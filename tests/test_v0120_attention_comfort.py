"""v0.1.20: attention gate, arm-fatigue mode, break reminders and their wiring into the pipeline, config, tray."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest
import yaml

from away import config as C
from away.app import tunables as T
from away.app.cursor_overlay import ring_spec
from away.app.runner import AwayApp, Pipeline
from away.app.tray import TrayController, comfort_menu_items
from away.comfort import actions, ui
from away.comfort.attention import AttentionGate, hand_height, hand_size
from away.comfort.fatigue import BreakReminder, FatigueMonitor
from away.comfort.hub import ComfortHub, off_screen_deg
from away.comfort.settings import (
    ADVANCED_SWITCHES,
    COMFORT_META,
    PLAIN_SWITCHES,
    AttentionConfig,
    ComfortConfig,
    comfort_tunables,
    set_switch,
)
from away.face.headpose import HeadPose
from away.hands.synthetic import pose

FPS = 30.0


def hand(y: float = 0.5, x: float = 0.5, scale: float = 0.12, name: str = "point") -> np.ndarray:
    """A pointing hand whose wrist sits at (x, y) (y grows downward): hand height = 1 - y."""
    return pose(name, center=(x, y), scale=scale)


def gate(**kw: Any) -> AttentionGate:
    return AttentionGate(AttentionConfig(enabled=True, **kw))


# ============================================================================== geometry
def test_hand_height_and_size() -> None:
    lm = hand(y=0.7)
    assert hand_height(lm) == pytest.approx(0.3)
    assert 0.05 < (hand_size(lm) or 0) < 0.2
    assert hand_size(hand(scale=0.03)) < hand_size(hand(scale=0.12))  # type: ignore[operator]
    assert hand_height(None) is None and hand_size(None) is None
    assert hand_height(np.zeros((3, 2))) is None  # not a hand


# ============================================================================== attention gate
def test_disabled_gate_always_lets_gestures_through() -> None:
    g = AttentionGate(AttentionConfig(enabled=False))
    assert g.update(0.0, None) is True and g.state == "off"
    assert g.update(0.1, hand(y=0.95)) is True
    assert g.voice_allowed(0.2) is True
    assert g.snapshot()["enabled"] is False and g.hint() == ""


def test_raised_hand_engages_after_the_onset_dwell() -> None:
    g = gate(onset_s=0.25)
    states = []
    for i in range(12):
        g.update(i / FPS, hand(y=0.5))
        states.append(g.state)
    assert states[0] == "arming" and g.hint() == ""
    assert states.index("engaged") == 8  # the first frame >= 0.25 s after the first good one (8/30 s)
    assert states[-1] == "engaged"


def test_arming_hint() -> None:
    g = gate(onset_s=1.0)
    g.update(0.0, hand())
    assert g.state == "arming" and g.hint() == "getting ready..."


def test_low_absent_and_far_hands_give_plain_hints() -> None:
    g = gate()
    assert g.update(0.0, hand(y=0.9)) is False and g.state == "idle"
    assert g.hint() == "raise your hand a little higher"
    g.update(0.1, None)
    assert g.hint() == "raise your hand into view"
    g.update(0.2, hand(y=0.4, scale=0.02))
    assert g.hint() == "bring your hand a little closer"
    assert g.snapshot()["hand_raised"] is False


def test_height_hysteresis_and_grace_period_stop_flicker() -> None:
    g = gate(min_hand_height=0.25, onset_s=0.0, grace_s=0.5)
    g.update(0.0, hand(y=0.70))  # height 0.30
    assert g.state == "engaged"
    g.update(0.1, hand(y=0.78))  # 0.22: below 0.25 but inside the 0.05 hysteresis
    assert g.state == "engaged"
    g.update(0.2, hand(y=0.82))  # 0.18: really low -> grace, still counts
    assert g.state == "grace" and g.engaged and g.hint() == "still with you..."
    g.update(0.5, hand(y=0.82))
    assert g.state == "grace"
    g.update(0.71, hand(y=0.82))  # 0.51 s after it went bad > grace_s
    assert g.state == "idle" and not g.engaged
    g.update(0.8, hand(y=0.78))  # coming back from idle needs the full threshold
    assert g.state == "idle"


def test_a_dip_shorter_than_the_grace_keeps_gestures_on() -> None:
    g = gate(onset_s=0.0, grace_s=0.8)
    g.update(0.0, hand())
    for t in (0.1, 0.3, 0.6):
        assert g.update(t, None) is True
    assert g.update(0.7, hand()) is True and g.state == "engaged"


def test_facing_uses_its_own_tolerance_with_hysteresis() -> None:
    g = gate(onset_s=0.0, grace_s=0.0, facing_tolerance=25.0)
    g.on_face(0.0, True, off_deg=10.0)
    assert g.update(0.0, hand()) and g.state == "engaged"
    g.on_face(0.1, False, off_deg=28.0)  # past 25 but inside the +5 hysteresis while engaged
    assert g.update(0.1, hand()) and g.checks.facing is True
    g.on_face(0.2, False, off_deg=32.0)
    assert g.update(0.2, hand()) is False and g.hint() == "turn toward the screen"
    g.on_face(0.3, False, off_deg=28.0)  # not engaged now: 28 > 25
    assert g.update(0.3, hand()) is False


def test_no_face_in_view_and_stopped_face_frames_mean_not_facing() -> None:
    g = gate(onset_s=0.0, grace_s=0.0)
    g.on_no_face(0.0)
    assert g.update(0.0, hand()) is False and g.checks.facing is False
    g.on_face(1.0, None)  # face back, pose baseline still learning: passes
    assert g.update(1.0, hand()) is True
    assert g.update(2.6, hand()) is False  # no face frame for > 1.5 s


def test_facing_check_is_skipped_when_it_cannot_be_done() -> None:
    g = gate(onset_s=0.0)  # face frames never arrived (no face model): never lock the user out
    assert g.update(0.0, hand()) is True and g.checks.facing is None
    g2 = AttentionGate(AttentionConfig(enabled=True, onset_s=0.0), face_available=False)
    g2.on_no_face(0.0)
    assert g2.update(0.0, hand()) is True and g2.checks.facing is None
    g3 = gate(onset_s=0.0, require_facing=False)
    g3.on_no_face(0.0)
    assert g3.update(0.0, hand()) is True


def test_gaze_check_only_with_a_calibration() -> None:
    g = gate(onset_s=0.0, grace_s=0.0, require_gaze=True)
    assert g.update(0.0, hand()) is True and g.checks.gaze is None  # never calibrated: skipped
    g.on_gaze(0.1, False)
    assert g.update(0.1, hand()) is False and g.hint() == "look at the screen"
    g.on_gaze(0.2, True)
    assert g.update(0.2, hand()) is True
    assert g.update(2.0, hand()) is False  # gaze went stale
    g4 = gate(onset_s=0.0, require_gaze=False)
    g4.on_gaze(0.0, False)
    assert g4.update(0.0, hand()) is True


def test_voice_gate_needs_facing_but_not_a_raised_hand() -> None:
    g = gate(gate_voice=False)
    g.on_no_face(0.0)
    assert g.voice_allowed(0.0) is True  # gate_voice off
    g = gate(gate_voice=True, grace_s=0.8)
    g.on_face(0.0, True, 3.0)
    g.update(0.0, None)  # no hand at all
    assert g.voice_allowed(0.0) is True
    g.on_no_face(0.5)
    assert g.voice_allowed(0.5) is True  # still inside the grace after the last good moment
    assert g.voice_allowed(0.9) is False


def test_snapshot_is_ui_ready() -> None:
    g = gate(onset_s=0.0)
    g.update(0.0, hand(y=0.6))
    s = g.snapshot()
    assert s["state"] == "engaged" and s["engaged"] and s["hand_height"] == pytest.approx(0.4)
    assert set(s) >= {"enabled", "state", "engaged", "facing", "gaze", "hand_raised", "hint"}


# ============================================================================== fatigue
def run_arm(mon: FatigueMonitor, t0: float, seconds: float, height_fn: Any, noise_fn: Any = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    events = []
    n = round(seconds * FPS)
    for i in range(n):
        t = t0 + i / FPS
        lm = hand(y=1.0 - height_fn(t - t0))
        if noise_fn is not None:
            lm = lm.copy()
            lm[:, :2] += rng.normal(0.0, noise_fn(t - t0), size=2)
        for ev in mon.update(t, lm):
            events.append((round(t - t0, 2), ev))
    return events, t0 + n / FPS


def first(events: list[tuple[float, str]], name: str) -> float | None:
    return next((t for t, e in events if e == name), None)


def test_a_steady_arm_is_tired_only_after_the_time_limit() -> None:
    mon = FatigueMonitor(ComfortConfig(raised_limit_s=240.0))
    events, _ = run_arm(mon, 0.0, 250, lambda _t: 0.6)
    t = first(events, "tired")
    assert t is not None and 239.0 <= t <= 241.0
    assert mon.snapshot()["tired"] is True and mon.reading.jitter_ratio <= 1.0  # no shake at all


def test_a_sinking_hand_is_tired_sooner() -> None:
    mon = FatigueMonitor(ComfortConfig(raised_limit_s=240.0, height_drop=0.08))

    def h(t: float) -> float:
        return 0.6 if t < 15 else max(0.45, 0.6 - 0.01 * (t - 15))

    events, _ = run_arm(mon, 0.0, 200, h)
    t = first(events, "tired")
    assert t is not None and 115 <= t <= 125  # one full sign: half the time limit
    assert mon.reading.drop >= 0.08 and mon.reading.jitter_ratio < 1.3  # the slow sink is not shake


def test_rising_tremor_is_a_tired_sign() -> None:
    mon = FatigueMonitor(ComfortConfig(raised_limit_s=240.0, jitter_ratio=1.8))

    def shake(t: float) -> float:
        return 0.002 if t < 20 else 0.006  # 3x the early shake

    events, _ = run_arm(mon, 0.0, 200, lambda _t: 0.6, shake)
    t = first(events, "tired")
    assert t is not None and t <= 130
    assert mon.reading.jitter_ratio > 1.8


def test_constant_noise_alone_is_not_tremor() -> None:
    mon = FatigueMonitor(ComfortConfig(raised_limit_s=240.0))
    events, _ = run_arm(mon, 0.0, 120, lambda _t: 0.6, lambda _t: 0.003)
    assert first(events, "tired") is None and mon.reading.jitter_ratio < 1.4


def test_both_signs_make_it_tired_at_the_minimum_time() -> None:
    mon = FatigueMonitor(ComfortConfig(min_raised_s=30.0))

    def h(t: float) -> float:
        return 0.6 if t < 10.5 else max(0.4, 0.6 - 0.05 * (t - 10.5))

    def shake(t: float) -> float:
        return 0.002 if t < 10.5 else 0.008

    events, _ = run_arm(mon, 0.0, 60, h, shake)
    t = first(events, "tired")
    assert t is not None and 29.5 <= t <= 31.0


def test_a_rest_resets_the_clock_but_a_short_dip_does_not() -> None:
    mon = FatigueMonitor(ComfortConfig(raised_limit_s=100.0, rest_s=5.0))
    _, t = run_arm(mon, 0.0, 60, lambda _t: 0.6)
    _, t = run_arm(mon, t, 2, lambda _t: 0.1)  # a 2 s dip below raised_height
    assert mon.reading.raised_s == pytest.approx(60, abs=0.5)
    _, t = run_arm(mon, t, 6, lambda _t: 0.1)  # a real rest
    assert mon.reading.raised_s == 0.0
    events, _ = run_arm(mon, t, 60, lambda _t: 0.6)
    assert first(events, "tired") is None


def test_switch_mode_hands_over_to_the_head_and_back() -> None:
    cfg = ComfortConfig(fatigue_switch=True, raised_limit_s=60.0, min_raised_s=5.0, rest_s=3.0, return_s=1.0)
    mon = FatigueMonitor(cfg)
    events, t = run_arm(mon, 0.0, 61, lambda _t: 0.6)
    assert [e for _, e in events] == ["tired", "switch_on"] and mon.switched
    events, t = run_arm(mon, t, 4, lambda _t: 0.1)  # arm down: rested, head still in control
    assert [e for _, e in events] == ["rested"] and mon.switched and mon.resting
    events, t = run_arm(mon, t, 0.5, lambda _t: 0.6)  # back up, not long enough yet
    assert events == [] and mon.switched
    events, t = run_arm(mon, t, 1.0, lambda _t: 0.6)
    assert [e for _, e in events] == ["switch_back"] and not mon.switched and not mon.tired


def test_switching_the_feature_off_meanwhile_gives_control_back() -> None:
    cfg = ComfortConfig(fatigue_switch=True, raised_limit_s=30.0, min_raised_s=5.0)
    mon = FatigueMonitor(cfg)
    run_arm(mon, 0.0, 31, lambda _t: 0.6)
    assert mon.switched
    cfg.fatigue_switch = False
    assert mon.update(40.0, None) == ["switch_back"] and not mon.switched


def test_a_long_gap_in_frames_does_not_count_as_arm_time() -> None:
    mon = FatigueMonitor(ComfortConfig())
    mon.update(0.0, hand(y=0.4))
    mon.update(100.0, hand(y=0.4))  # the camera stalled for 100 s
    assert mon.reading.raised_s == 0.0


def test_break_reminders_every_n_minutes_and_a_real_break_restarts() -> None:
    br = BreakReminder(ComfortConfig(break_every_min=10.0, break_reset_s=60.0))
    fired = [t for t in range(0, 1300, 5) if br.update(float(t), True)]
    assert fired == [600, 1200]
    assert br.update(1300.0, False) is False
    assert br.update(1370.0, True) is False  # 70 s away = a break: the clock restarts
    assert br.minutes_used(1370.0) == 0.0
    again = [t for t in range(1375, 1975, 5) if br.update(float(t), True)]
    assert again == [1970]  # 600 s after the break ended
    off = BreakReminder(ComfortConfig(break_reminders=False, break_every_min=5.0))
    assert not any(off.update(float(t), True) for t in range(0, 1000, 10))


# ============================================================================== hub
class FakeBaseline:
    def __init__(self, yaw: float | None = 0.0, pitch: float = 0.0, ready: bool = True) -> None:
        self.yaw, self.pitch, self.ready = yaw, pitch, ready


class FakeMon:
    def __init__(self, x0: float, x1: float) -> None:
        self.x0, self.x1 = x0, x1

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and 0 <= y < 1080


class FakeEstimator:
    def __init__(self, pt: Any) -> None:
        self.pt = pt

    def fresh(self, _t: float) -> Any:
        return self.pt


def fake_headgaze(yaw: float = 0.0, box: Any = None, gaze: Any = None) -> Any:
    mouse = types.SimpleNamespace(mapper=types.SimpleNamespace(monitors=[FakeMon(0, 1920)]))
    return types.SimpleNamespace(
        pose=HeadPose(yaw, 0.0, 0.0),
        baseline=FakeBaseline(),
        cal=types.SimpleNamespace(pose_box=box) if box else None,
        estimator=FakeEstimator(gaze) if gaze is not None else None,
        mouse=mouse,
    )


def test_off_screen_degrees() -> None:
    p = HeadPose(12.0, -3.0, 0.0)
    assert off_screen_deg(p, FakeBaseline(2.0, 0.0), None) == pytest.approx(10.0)
    assert off_screen_deg(p, FakeBaseline(ready=False), None) is None
    assert off_screen_deg(p, None, (-20.0, 10.0, -5.0, 5.0)) == pytest.approx(2.0)
    assert off_screen_deg(p, None, (-20.0, 20.0, -5.0, 5.0)) == 0.0
    assert off_screen_deg(None, FakeBaseline(), None) is None


def hub_for(**over: Any) -> tuple[ComfortHub, list[Any], list[Any]]:
    cfg = C.AppConfig()
    for path, value in over.items():
        sec, name = path.split("__")
        setattr(getattr(cfg, sec), name, value)
    published: list[Any] = []
    notes: list[Any] = []
    hub = ComfortHub(
        cfg,
        fake_headgaze(),
        publish=lambda topic, data: published.append((topic, data)),
        notify=lambda title, msg: notes.append((title, msg)),
    )
    return hub, published, notes


def test_hub_feeds_the_gate_from_head_pose_and_gaze() -> None:
    hub, published, _ = hub_for(
        attention__enabled=True, attention__onset_s=0.0, attention__grace_s=0.0, attention__require_gaze=True
    )
    hub.headgaze = fake_headgaze(yaw=5.0, gaze=(500.0, 500.0))
    hub.on_face(0.0, True)
    assert hub.allow_hand(0.0, hand()) is True
    assert published[-1][0] == "attention.state" and published[-1][1]["state"] == "engaged"
    hub.headgaze = fake_headgaze(yaw=5.0, gaze=(5000.0, 500.0))  # looking off every monitor
    hub.on_face(0.1, True)
    assert hub.allow_hand(0.1, hand()) is False and hub.gate.checks.gaze is False
    hub.headgaze = fake_headgaze(yaw=40.0, box=(-10.0, 10.0, -5.0, 5.0))  # outside the calibrated pose box
    hub.on_face(0.2, True)
    assert hub.allow_hand(0.2, hand()) is False and hub.gate.checks.facing is False
    hub.on_face(0.3, False)
    assert hub.allow_hand(0.3, hand()) is False and hub.gate.hint() == "turn toward the screen"


def test_hub_fatigue_switch_drives_head_assist_and_restores_it() -> None:
    hub, published, notes = hub_for(
        comfort__fatigue_switch=True,
        comfort__raised_limit_s=30.0,
        comfort__min_raised_s=5.0,
        comfort__rest_s=2.0,
        comfort__return_s=0.5,
    )
    t = 0.0
    while t < 31.0:
        hub.allow_hand(t, hand(y=0.4))
        t += 1 / FPS
    assert hub.cfg.hands.cursor.head_assist == "pointer"
    assert any("Lower your hand" in m for _, m in notes)
    assert "comfort.switch_on" in [p for p, _ in published]
    assert hub.allow_hand(t, hand(y=0.9)) is False  # a lowered hand in view must not block the head pointer
    assert hub.allow_hand(t + 0.1, hand(y=0.4)) is True  # a raised hand still works
    # a config hot-reload puts the file value back; the hub keeps the head pointer and remembers the new base
    new = C.AppConfig()
    new.hands.cursor.head_assist = "coarse"
    new.comfort = hub.cfg.comfort
    hub.cfg.hands.cursor.head_assist = "coarse"
    hub.apply_config(new)
    assert hub.cfg.hands.cursor.head_assist == "pointer" and hub.base_head_assist == "coarse"
    for i in range(int(3 * FPS)):
        hub.allow_hand(t + 1 + i / FPS, None)  # rest
    for i in range(int(1 * FPS)):
        hub.allow_hand(t + 5 + i / FPS, hand(y=0.4))  # hand back
    assert hub.cfg.hands.cursor.head_assist == "coarse"
    assert notes[-1][1] == ui.SWITCH_BACK_MESSAGE


def test_hub_without_face_tracking_never_switches_to_the_head() -> None:
    hub, _, notes = hub_for(
        comfort__fatigue_switch=True, comfort__raised_limit_s=30.0, comfort__min_raised_s=5.0, face__enabled=False
    )
    for i in range(int(31 * FPS)):
        hub.allow_hand(i / FPS, hand(y=0.4))
    assert hub.cfg.hands.cursor.head_assist == "off" and not hub.fatigue.switched
    assert "face tracking" in notes[-1][1]


def test_hub_tips_respect_the_cooldown() -> None:
    hub, _, notes = hub_for(comfort__raised_limit_s=30.0, comfort__min_raised_s=5.0, comfort__rest_s=1.0)
    t = 0.0
    for _round in range(2):
        for _ in range(int(31 * FPS)):
            hub.allow_hand(t, hand(y=0.4))
            t += 1 / FPS
        for _ in range(int(2 * FPS)):
            hub.allow_hand(t, None)
            t += 1 / FPS
        for _ in range(int(2 * FPS)):
            hub.allow_hand(t, hand(y=0.4))
            t += 1 / FPS
    tips = [m for title, m in notes if title == ui.FATIGUE_TITLE]
    assert len(tips) == 1 and "Settings > Getting started" in tips[0]  # the second fell inside the cooldown
    quiet, _, notes2 = hub_for(comfort__fatigue_tips=False, comfort__raised_limit_s=30.0, comfort__min_raised_s=5.0)
    for i in range(int(31 * FPS)):
        quiet.allow_hand(i / FPS, hand(y=0.4))
    assert notes2 == []


def test_hub_break_reminder_is_a_toast() -> None:
    hub, published, notes = hub_for(comfort__break_every_min=5.0)
    for i in range(0, 301):
        hub.allow_hand(float(i), hand(y=0.8))  # low hand: no fatigue, still "use"
    assert notes and notes[-1][0] == ui.BREAK_TITLE and "about 5 minutes" in notes[-1][1]
    assert ("comfort.break", {"minutes": 5}) in published


def test_hub_default_notify_publishes_an_app_notice() -> None:
    got: list[Any] = []
    hub = ComfortHub(C.AppConfig(), None, publish=lambda topic, data: got.append((topic, data)))
    hub.notify("t", "m")
    assert got == [("app.notice", {"title": "t", "message": "m"})]
    assert hub.voice_allowed(0.0) is True
    hub.on_face(0.0, True)  # no head tracker at all: nothing breaks
    assert hub.gate._face_t == 0.0


# ============================================================================== pipeline wiring
def move_count(p: Pipeline, frames: list[np.ndarray], t0: float = 0.0) -> int:
    n = 0
    for i, lm in enumerate(frames):
        n += p.on_hand(t0 + i / FPS, lm).count("move")
    return n


def test_pipeline_ignores_a_resting_hand_when_the_gate_is_on() -> None:
    cfg = C.AppConfig()
    cfg.attention.enabled = True
    p = Pipeline(cfg)
    low = [hand(y=0.9, x=0.4 + i * 0.01) for i in range(30)]
    assert move_count(p, low) == 0
    assert not p.gestures.hand_present  # the hand counts as absent: the head pointer could act
    assert p.attention_state()["hint"] == "raise your hand a little higher"
    raised = [hand(y=0.5, x=0.4 + i * 0.01) for i in range(30)]
    assert move_count(p, raised, 2.0) > 15
    assert p.cursor_state()["attention"] == "engaged"
    assert any(c[0] == "move" for c in p.mouse.backend.calls)  # the cursor really moved


def test_pipeline_gate_ends_a_drag_cleanly_when_attention_is_lost() -> None:
    cfg = C.AppConfig()
    cfg.attention.enabled = True
    cfg.attention.onset_s = 0.0
    cfg.attention.grace_s = 0.0
    p = Pipeline(cfg)
    t = 0.0
    for lm in [pose("pinch_open", center=(0.5, 0.5))] * 3 + [
        pose("pinch", center=(0.5 + 0.01 * i, 0.5)) for i in range(20)
    ]:
        p.on_hand(t, lm)
        t += 1 / FPS
    assert p.gestures.dragging
    out = p.on_hand(t, pose("pinch", center=(0.5, 0.95)))  # hand dropped out of the working area
    assert "drag_end" in out and not p.gestures.dragging


def test_pipeline_gate_off_changes_nothing() -> None:
    p = Pipeline(C.AppConfig())
    assert move_count(p, [hand(y=0.9, x=0.4 + i * 0.01) for i in range(30)]) > 20
    assert p.cursor_state()["attention"] == "off"


def test_pipeline_hot_reload_turns_the_gate_on_and_feeds_faces() -> None:
    p = Pipeline(C.AppConfig())
    new = C.AppConfig()
    new.attention.enabled = True
    new.attention.onset_s = 0.0
    p.apply_config(new)
    assert p.cfg.attention.enabled is True and p.comfort.cfg.attention.onset_s == 0.0
    p.on_faces(0.0, [], [], extras=[])  # a face frame with nobody in it
    p.on_hand(0.0, hand())
    assert p.comfort.gate.checks.facing is False and not p.comfort.gate.engaged
    obs = types.SimpleNamespace(pose=None, proxy=None, gaze=None)
    p.on_faces(1.0, [(0.5, 0.5, 0.3)], [{}], frontal=[True], extras=[obs])
    assert p.comfort.gate._face_t == 1.0
    obs2 = types.SimpleNamespace(pose=(1.0, 2.0, 0.0), proxy=None, gaze=None)
    p.on_faces(1.1, [(0.5, 0.5, 0.3)], [{}], frontal=[True], extras=[obs2])
    p.on_hand(1.1, hand())
    assert p.comfort.gate.checks.facing is True  # learning the usual pose: passes


def test_pipeline_indicator_switch_hides_the_ring_tint() -> None:
    cfg = C.AppConfig()
    cfg.attention.enabled = True
    cfg.attention.indicator = False
    p = Pipeline(cfg)
    p.on_hand(0.0, None)
    assert p.cursor_state()["attention"] == "off"


def test_ring_is_dim_while_gestures_wait() -> None:
    spec = ring_spec({"state": "idle", "attention": "idle"})
    assert spec["color"] == ui.ATTENTION_COLORS["idle"] and spec["width"] == 1
    assert ring_spec({"state": "idle", "attention": "arming"})["color"] == ui.ATTENTION_COLORS["arming"]
    assert ring_spec({"state": "paused", "attention": "idle"})["width"] == 2  # paused still wins
    assert ring_spec({"state": "pinch", "attention": "engaged"})["color"] != ui.ATTENTION_COLORS["idle"]


def test_voice_commands_are_dropped_when_not_facing_with_gate_voice() -> None:
    fired: list[str] = []
    comfort = types.SimpleNamespace(voice_allowed=lambda: False)

    def fire(_cat: str, phrase: str) -> Any:
        fired.append(phrase)
        return types.SimpleNamespace(kind="key")

    fake = types.SimpleNamespace(pipeline=types.SimpleNamespace(comfort=comfort, fire=fire))
    assert AwayApp._run_voice_command(fake, "scroll down") is False  # type: ignore[arg-type]
    AwayApp._ambient_command(fake, None, "scroll down")  # type: ignore[arg-type]
    assert fired == []
    comfort.voice_allowed = lambda: True
    assert AwayApp._run_voice_command(fake, "scroll down") is True  # type: ignore[arg-type]
    AwayApp._ambient_command(fake, None, "page up")  # type: ignore[arg-type]
    assert fired == ["scroll down", "page up"]


# ============================================================================== config
def test_config_defaults_round_trip_and_validation(tmp_path) -> None:
    cfg = C.AppConfig()
    assert cfg.attention.enabled is False and cfg.comfort.fatigue_switch is False
    assert cfg.comfort.break_reminders is True and cfg.comfort.fatigue_tips is True
    p = C.save(cfg, tmp_path / "c.yaml")
    data = yaml.safe_load(p.read_text())
    assert data["attention"]["min_hand_height"] == 0.25 and data["comfort"]["break_every_min"] == 30.0
    data["attention"]["enabled"] = True
    data["comfort"]["fatigue_switch"] = True
    back, unknown = C.from_dict(data)
    assert unknown == [] and back.attention.enabled and back.comfort.fatigue_switch
    for bad in (
        {"attention": {"enabled": "yes"}},
        {"comfort": {"break_reminders": 1}},
        {"attention": {"onset_s": 9.0}},
        {"comfort": {"jitter_ratio": 0.5}},
        {"comfort": {"min_raised_s": 300.0, "raised_limit_s": 100.0}},
    ):
        with pytest.raises(ValueError):
            C.from_dict(bad)
    _, unknown = C.from_dict({"attention": {"nope": 1}})
    assert unknown == ["attention.nope"]


def test_tunables_are_registered_in_the_tuning_tab_sections() -> None:
    paths = [p for p, *_ in C.tunable_ranges(C.AppConfig())]
    mine = [p for p, *_ in comfort_tunables(C.AppConfig())]
    assert set(mine) <= set(paths) and set(mine) == set(COMFORT_META)
    groups = {p: T.describe(p).group for p in mine}
    assert {groups[p] for p in mine if p.startswith("attention.")} == {"Face", "Head & eyes"}
    assert groups["attention.facing_tolerance"] == "Head & eyes"
    assert groups["comfort.raised_limit_s"] == "Head & eyes"
    assert groups["comfort.break_every_min"] == "Other"
    for _p, v, lo, hi in comfort_tunables(C.AppConfig()):
        assert lo <= v <= hi


def test_switch_helpers(tmp_path) -> None:
    cfg = C.AppConfig()
    assert set_switch(cfg, "attention.enabled", True) is None and cfg.attention.enabled
    assert set_switch(cfg, "comfort.raised_height", True) is not None  # numbers are not switches
    assert {p for p, *_ in PLAIN_SWITCHES} == {
        "attention.enabled",
        "comfort.fatigue_switch",
        "comfort.fatigue_tips",
        "comfort.break_reminders",
    }
    assert PLAIN_SWITCHES[0][1] == "Only respond when I'm looking at the screen"
    assert PLAIN_SWITCHES[1][1] == "Switch to head control when my arm is tired"
    assert all(set_switch(cfg, p, True) is None for p, _ in ADVANCED_SWITCHES)
    f = tmp_path / "cfg.yaml"
    live = C.AppConfig()
    assert actions.set_switch_in_file("comfort.fatigue_switch", True, f, live_cfg=live) is None
    assert C.load(f)[0].comfort.fatigue_switch is True and live.comfort.fatigue_switch is True
    f.write_text("attention: {enabled: maybe}\n")
    assert "settings file has a problem" in str(actions.set_switch_in_file("attention.enabled", True, f))
    assert actions.set_switch_in_file("nope.nope", True, f) is not None


def test_launch_first_run_uses_away_tour() -> None:
    seen: list[Any] = []
    assert actions.launch_first_run(popen=seen.append) is None
    assert seen[0][-1] == "tour"

    def boom(_argv: Any) -> None:
        raise OSError("nope")

    assert "could not open" in str(actions.launch_first_run(popen=boom))


# ============================================================================== tray + words
def test_tray_title_and_menu(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AWAY_CONFIG", str(tmp_path / "config.yaml"))
    cfg = C.AppConfig()
    cfg.attention.enabled = True
    p = Pipeline(cfg)
    app = types.SimpleNamespace(cfg=cfg, pipeline=p, voice_state=lambda: "listening")
    ctl = TrayController(app)
    p.on_hand(0.0, None)
    assert "waiting for you (raise your hand into view)" in ctl.title()
    cfg.attention.enabled = False
    p.on_hand(0.1, None)
    assert "waiting" not in ctl.title()

    class FakeItem:
        def __init__(self, text: str, action: Any, **kw: Any) -> None:
            self.text, self.action, self.kw = text, action, kw

    told: list[Any] = []
    items = comfort_menu_items(types.SimpleNamespace(MenuItem=FakeItem), ctl, lambda *a: told.append(a))
    assert [i.text for i in items] == [
        "Only respond when I'm looking at the screen",
        "Guided first run...",
        "Calibrate tracking...",
    ]
    assert items[0].kw["checked"](None) is False
    items[0].action(None, None)
    assert cfg.attention.enabled is True and C.load(tmp_path / "config.yaml")[0].attention.enabled is True
    assert items[0].kw["checked"](None) is True and "look at the screen" in told[-1][1]


def test_badges_messages_and_friendly_errors() -> None:
    assert ui.attention_badge(None)["label"] == "Always responding"
    b = ui.attention_badge({"enabled": True, "state": "idle", "hint": "turn toward the screen"})
    assert b["dim"] and b["label"] == "Waiting for you"
    assert ui.tray_suffix({"enabled": True, "state": "engaged"}) == " - listening to your hand"
    assert ui.tray_suffix({"enabled": False, "state": "idle"}) == ""
    assert "head" in ui.fatigue_message(True) and "Settings" in ui.fatigue_message(False)
    assert "face tracking" in ui.fatigue_message(True, face_ok=False)
    assert "30 minutes" in ui.break_message(30)
    cases = {
        "camera 0: Access denied by privacy settings": "camera_privacy",
        "the camera is busy (in use by another app)": "pause_app",
        "hand model missing (run `away setup`)": "download_models",
        "mediapipe not installed (pip install 'project-away[hands]')": "install_help",
        "camera 1 too slow: 1.0 fps": "retry",
        "camera 0 did not open": "retry",
        "weird thing": "retry",
    }
    for raw, fix in cases.items():
        err = ui.friendly_error(raw)
        assert err.fix == fix and err.fix in ui.FIX_ACTIONS, raw
        assert err.fix_label and err.title and "Traceback" not in err.message
    assert ui.friendly_error(None) is ui.UNKNOWN_ERROR
    assert ui.camera_privacy_command("win32")[-1] == "ms-settings:privacy-webcam"  # type: ignore[index]
    assert ui.camera_privacy_command("darwin")[0] == "open"  # type: ignore[index]
    assert ui.camera_privacy_command("linux") is None
