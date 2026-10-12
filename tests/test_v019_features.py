"""v0.1.9: big cursor while active, hand choice (hands.prefer), four-finger auto-scroll, clap control."""

from __future__ import annotations

import json
import queue

import numpy as np
import pytest

from away import config as C
from away.app import bigcursor as BC
from away.app.cursor_overlay import COLORS, arrow_points, ring_size, ring_spec
from away.app.runner import Pipeline
from away.app.settings_gui import SettingsModel, clap_test_handler
from away.bindings import Action, BindingsConfig
from away.config import AppConfig, GestureConfig
from away.control.mouse import RecordingBackend
from away.hands import tracker as T
from away.hands.autoscroll import LOST_STOP_S, OTHER_STOP_S, AutoScroller, axis_speed
from away.hands.gestures import GestureEvent, GestureStateMachine
from away.hands.landmarks import thumb_fold_ratio
from away.hands.select import HandObs, HandSelector, real_side
from away.hands.synthetic import pose
from away.voice import clap as CL

DT = 1 / 30
SR = 16000


# ============================================================================ big cursor
class FakeCursor:
    def __init__(self, size=40, ok=True, marker=None):
        self.size, self.ok, self.marker = size, ok, marker
        self.sets: list[int] = []
        self.marker_during_set: list[bool] = []

    def get(self):
        return self.size

    def set(self, size):
        self.sets.append(size)
        if self.marker is not None:
            self.marker_during_set.append(self.marker.exists())
        if self.ok:
            self.size = size
        return self.ok


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def make_big(tmp_path, level=2, **kw):
    marker = tmp_path / "big_cursor.json"
    be = FakeCursor(marker=marker, **kw)
    clock = Clock()
    return BC.BigCursor(be, level, marker, clock=clock), be, clock, marker


def test_target_sizes_follow_the_32px_scale_and_windows_limits():
    assert BC.target_size(0) == 0
    assert [BC.target_size(i) for i in range(1, 6)] == [48, 64, 96, 128, 192]
    assert all(BC.MIN_SIZE <= BC.target_size(i) <= BC.MAX_SIZE for i in range(1, 6))
    assert BC.ring_scale(0) == 1.0 and BC.ring_scale(5) > BC.ring_scale(1) > 1.0


def test_big_cursor_enlarges_saves_original_and_restores(tmp_path):
    big, be, _clock, marker = make_big(tmp_path)
    assert big.update(True)
    assert be.sets == [64] and big.enlarged and big.original == 40
    assert be.marker_during_set == [True]  # the original is on disk before the system cursor changes
    assert json.loads(marker.read_text())["original"] == 40
    assert big.restore()
    assert be.sets == [64, 40] and not big.enlarged and not marker.exists()
    assert not big.restore()  # nothing left to restore


def test_big_cursor_debounces_flapping_auto_pause(tmp_path):
    big, be, clock, _m = make_big(tmp_path)
    big.update(True)
    for i in range(10):  # auto-pause flapping every 50 ms
        clock.t += 0.05
        big.update(i % 2 == 0)
    assert len(be.sets) <= 3  # at most one change per 300 ms over 0.5 s
    clock.t += 0.3
    big.update(False)
    clock.t += 0.3
    big.update(False)
    assert be.sets[-1] == 40 and not big.enlarged


def test_big_cursor_pending_change_lands_after_the_debounce(tmp_path):
    big, be, clock, _m = make_big(tmp_path)
    big.update(True)
    clock.t += 0.1
    assert not big.update(False)  # too soon
    clock.t += 0.25
    assert big.update(False) and be.sets == [64, 40]


def test_restore_ignores_debounce(tmp_path):
    big, be, clock, marker = make_big(tmp_path)
    big.update(True)
    clock.t += 0.01
    big.restore()  # quit / manual pause: immediate
    assert be.sets == [64, 40] and not marker.exists()


def test_stale_marker_from_a_crash_is_restored_at_startup(tmp_path):
    marker = tmp_path / "big_cursor.json"
    marker.write_text(json.dumps({"original": 48, "applied": 128, "pid": 1}))
    be = FakeCursor(size=128)
    seen = []
    big = BC.BigCursor(be, 2, marker, publish=lambda t, d: seen.append((t, d)))
    assert big.recover()
    assert be.sets == [48] and not marker.exists()
    assert seen[0][0] == "cursor.size_recovered"
    assert not big.recover()  # nothing left


def test_unreadable_marker_is_dropped_without_touching_the_cursor(tmp_path):
    marker = tmp_path / "big_cursor.json"
    marker.write_text("{not json")
    be = FakeCursor()
    assert not BC.BigCursor(be, 2, marker).recover()
    assert be.sets == [] and not marker.exists()


def test_failed_resize_leaves_no_marker_and_does_not_retry_every_frame(tmp_path):
    big, be, clock, marker = make_big(tmp_path, ok=False)
    assert not big.update(True)
    assert not marker.exists() and not big.enlarged
    clock.t += 0.05
    big.update(True)
    assert len(be.sets) == 1


def test_size_zero_or_no_cursor_api_changes_nothing(tmp_path):
    big, be, _c, _m = make_big(tmp_path, level=0)
    assert not big.update(True) and be.sets == []
    null = BC.BigCursor(BC.NullCursorBackend(), 3, tmp_path / "m.json")
    assert not null.update(True) and not null.enlarged
    assert isinstance(BC.default_backend("linux"), BC.NullCursorBackend)
    assert isinstance(BC.default_backend("darwin"), BC.NullCursorBackend)


def test_changing_the_level_while_big_reapplies(tmp_path):
    big, be, _c, marker = make_big(tmp_path)
    big.update(True)
    big.set_level(4)
    assert be.sets[-1] == 128 and json.loads(marker.read_text())["applied"] == 128
    big.set_level(0)
    assert be.sets[-1] == 40 and not big.enlarged


def test_gestures_active_follows_every_kind_of_pause():
    p = Pipeline(AppConfig(), RecordingBackend())
    assert p.gestures_active()
    p.set_paused(True)  # tray / F9 / voice
    assert not p.gestures_active()
    p.set_paused(False)

    class Guard:
        active = True

    p.guard = Guard()  # physical mouse in use
    p.on_hand(0.0, None)
    assert not p.gestures_active()
    Guard.active = False
    p.on_hand(0.1, None)
    assert p.gestures_active()
    t = 0.2
    for _ in range(20):  # open-palm clutch
        t += 0.05
        p.on_hand(t, pose("palm"))
    assert p.paused and not p.gestures_active()
    cfg = AppConfig()
    cfg.hands.enabled = False
    assert not Pipeline(cfg, RecordingBackend()).gestures_active()


def test_big_cursor_shrinks_when_the_pipeline_pauses(tmp_path):
    p = Pipeline(AppConfig(), RecordingBackend())
    big, be, clock, _m = make_big(tmp_path)
    big.update(p.gestures_active())
    p.on_clap("double", 1.0)  # double clap pauses
    clock.t += 0.4
    big.update(p.gestures_active())
    assert be.sets == [64, 40]


def test_big_cursor_config_validation_and_setting():
    cfg = AppConfig()
    assert cfg.hands.cursor.big_cursor_size == 2
    for bad in (-1, 6, 2.5, True):
        cfg.hands.cursor.big_cursor_size = bad
        with pytest.raises(ValueError):
            C.validate(cfg)


def test_ring_grows_off_windows_and_shows_the_autoscroll_arrow():
    st = {"state": "idle", "size": 3}
    assert ring_size(st, "win32") == 34
    assert ring_size(st, "linux") > 34
    assert ring_size({"state": "paused", "size": 3}, "linux") == 34
    spec = ring_spec({"state": "autoscroll", "direction": "down"})
    assert spec["color"] == COLORS["autoscroll"] and spec["arrow"] == "down"
    assert ring_spec({"state": "autoscroll", "direction": None})["arrow"] is None
    pts = arrow_points("up", 34, 4)
    assert len(pts) == 6 and pts[1] < pts[3]  # tip above the base


# ============================================================================ hand choice
def obs(label, x=0.5, y=0.5, score=0.9):
    return HandObs(pose("point", center=(x, y)), label, score)


def test_mediapipe_label_vs_real_hand_with_and_without_mirroring():
    # Tasks API labels the hand as it looks in the frame; a flipped (mirrored) frame swaps left and right
    assert real_side("Right", mirrored=False) == "right"
    assert real_side("Left", mirrored=False) == "left"
    assert real_side("Left", mirrored=True) == "right"
    assert real_side("Right", mirrored=True) == "left"


def test_prefer_right_picks_the_real_right_hand_in_a_mirrored_frame():
    sel = HandSelector("right", mirrored=True)
    right_hand = obs("Left", x=0.7)  # mirrored frame: the user's right hand is labelled Left
    phone_hand = obs("Right", x=0.3)
    assert sel.select(0.0, [phone_hand, right_hand]) is right_hand
    assert HandSelector("right", mirrored=False).select(0.0, [phone_hand, right_hand]) is phone_hand


def test_the_other_hand_is_ignored_completely():
    sel = HandSelector("left", mirrored=True)
    assert sel.select(0.0, [obs("Left")]) is None  # only the (real) right hand is visible
    assert sel.wrong_side([obs("Left")])
    assert sel.select(0.1, []) is None


def test_label_flicker_on_the_tracked_hand_is_tolerated_briefly():
    sel = HandSelector("right", mirrored=True)
    assert sel.select(0.0, [obs("Left", x=0.5)]) is not None
    flick = obs("Right", x=0.51)  # same hand, label flipped for a moment
    assert sel.select(0.1, [flick]) is flick
    assert sel.select(0.2, [obs("Right", x=0.52)]) is not None
    assert sel.select(0.5, [obs("Right", x=0.52)]) is None  # stayed wrong past the grace: not the right hand


def test_any_locks_onto_the_tracked_hand_and_does_not_jump():
    sel = HandSelector("any", mirrored=True)
    main = obs("Left", x=0.6)
    assert sel.select(0.0, [main]) is main
    for i in range(5):  # a second hand appears (phone); the tracked one moves a little
        a, b = obs("Left", x=0.6 + 0.01 * i), obs("Right", x=0.2)
        assert sel.select(0.03 * (i + 1), [b, a]) is a
    # the tracked hand drops out for a frame: the other hand does not take over
    assert sel.select(0.2, [obs("Right", x=0.2)]) is None
    assert sel.select(0.25, [obs("Left", x=0.66), obs("Right", x=0.2)]).wrist[0] == pytest.approx(
        pose("point", center=(0.66, 0.5))[0, 0]
    )
    # it is really gone: after the lock expires the other hand may drive
    assert sel.select(0.9, [obs("Right", x=0.2)]) is not None


def test_any_follows_a_fast_move_of_the_only_hand():
    sel = HandSelector("any", mirrored=True)
    sel.select(0.0, [obs("Left", x=0.2)])
    assert sel.select(0.05, [obs("Left", x=0.8)]) is not None  # same side, only hand: it moved fast


def test_changing_the_preference_starts_over():
    state = {"p": "right"}
    sel = HandSelector(lambda: state["p"], mirrored=True)
    r, l_ = obs("Left", x=0.7), obs("Right", x=0.3)
    assert sel.select(0.0, [r, l_]) is r
    state["p"] = "left"
    assert sel.select(0.05, [r, l_]) is l_


class FakeLandmark:
    def __init__(self, hands_seq):
        self.hands_seq = hands_seq  # callable(frame_no) -> list of (label, x)
        self.calls = 0
        self.ms: list[int] = []

    def detect_for_video(self, image, ms):
        self.ms.append(ms)
        self.calls += 1
        return FakeResult(self.hands_seq())

    def close(self):
        pass


class FakeResult:
    def __init__(self, hands):
        class P:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class Cat:
            def __init__(self, name):
                self.category_name, self.score = name, 0.95

        self.hand_landmarks = [[P(*row) for row in pose("point", center=(x, 0.5))] for _lab, x in hands]
        self.handedness = [[Cat(lab)] for lab, _x in hands]


def make_tracker(visible, prefer="any", mirrored=True, max_hands=2):
    made = {}

    def factory(n):
        made[n] = FakeLandmark(lambda: visible["hands"] if n > 1 else visible["hands"][:1])
        return made[n]

    tr = T.HandTracker(None, max_hands, prefer=prefer, mirrored=mirrored, factory=factory)
    return tr, made


def test_two_hand_landmarker_only_probes_while_one_hand_is_visible():
    visible = {"hands": [("Left", 0.6)]}
    tr, made = make_tracker(visible)
    for i in range(60):
        fr = tr.process(None, i * DT)
        assert fr.landmarks is not None and fr.hand == "right"
    assert 8 <= made[2].calls <= 12  # one probe every PROBE_EVERY frames, not every frame
    assert not tr.dual


def test_two_hands_switch_to_dual_tracking_and_back():
    visible = {"hands": [("Left", 0.6)]}
    tr, made = make_tracker(visible, prefer="right")
    for i in range(6):
        tr.process(None, i * DT)
    visible["hands"] = [("Right", 0.2), ("Left", 0.6)]
    t = 6 * DT
    for _ in range(12):
        fr = tr.process(None, t)
        t += DT
    assert tr.dual and fr.n_hands == 2 and fr.hand == "right"
    assert fr.hands_gap == pytest.approx(0.4, abs=0.01)
    visible["hands"] = [("Left", 0.6)]
    for _ in range(T.DUAL_EXIT_FRAMES + 1):
        tr.process(None, t)
        t += DT
    assert not tr.dual
    assert all(b > a for a, b in zip(made[2].ms, made[2].ms[1:]))  # strictly increasing per landmarker


def test_wrong_hand_alone_is_ignored_and_probed_more_often():
    visible = {"hands": [("Right", 0.3)]}  # mirrored: the user's left hand
    tr, made = make_tracker(visible, prefer="right")
    frames = [tr.process(None, i * DT) for i in range(30)]
    assert all(f.landmarks is None for f in frames)
    assert made[2].calls >= 12


def test_max_hands_one_uses_a_single_landmarker():
    visible = {"hands": [("Left", 0.6)]}
    tr, made = make_tracker(visible, max_hands=1)
    tr.process(None, 0.0)
    assert 2 not in made and tr._multi is None


def test_hand_preference_config_and_old_config_migration(tmp_path):
    # v0.1.14: hands.prefer became hands.roles.mode (see tests/test_v0114_roles_palm.py)
    assert AppConfig().hands.roles.mode == "auto" and AppConfig().hands.max_hands == 2
    with pytest.raises(ValueError):
        C.from_dict({"hands": {"prefer": "both"}})
    cfg, _ = C.from_dict({"config_version": 2, "hands": {"max_hands": 1}})
    assert cfg.hands.max_hands == 2 and any("max_hands" in n for n in cfg.migrations)
    cfg3, _ = C.from_dict({"config_version": 3, "hands": {"max_hands": 1}})
    assert cfg3.hands.max_hands == 1  # chosen in v0.1.9: kept
    assert not any("max_hands" in n for n in cfg3.migrations)
    m = SettingsModel(AppConfig(), tmp_path / "c.yaml")
    assert m.set_voice("hands.roles.mode", "left") is None and m.cfg.hands.roles.mode == "left"
    assert m.set_voice("hands.roles.mode", "both") is not None


def test_voice_command_switches_the_hand():
    p = Pipeline(AppConfig(), RecordingBackend())
    assert p.fire("voice", "use left hand") == Action("hands", "left")
    assert p.hand_prefer == "left"
    p.fire("voice", "Use either hand!")
    assert p.hand_prefer == "auto"
    new = AppConfig()
    new.hands.roles.mode = "right"
    p.apply_config(new)  # a Settings change wins
    assert p.hand_prefer == "right"


# ============================================================================ auto-scroll
def four(**kw):
    return pose("four", **kw)


def feed(sm, frames, t0=0.0):
    out, t = [], t0
    for lm in frames:
        t += DT
        out += sm.update(t, lm)
    return out, t


def kinds(events):
    return [e.kind for e in events]


def test_thumb_fold_tells_four_fingers_from_the_open_palm():
    assert thumb_fold_ratio(pose("palm")) > 1.0 > 0.75 > thumb_fold_ratio(four())
    sm = GestureStateMachine()
    assert sm.pose_of(four(rotation=0.3)) == "four" and sm.pose_of(pose("palm")) == "palm"
    half = pose("palm", thumb_fold=0.45)  # in the hysteresis band: keeps the previous answer
    assert 0.75 <= thumb_fold_ratio(half) <= 0.95
    sm.pose_of(four())
    assert sm.pose_of(half) == "four"
    sm.pose_of(pose("palm"))
    assert sm.pose_of(half) == "palm"


def test_autoscroll_starts_only_after_the_hold():
    sm = GestureStateMachine(GestureConfig(autoscroll_hold=0.2))
    ev, t = feed(sm, [four()] * 4)  # ~0.13 s
    assert "autoscroll_start" not in kinds(ev)
    ev, t = feed(sm, [four()] * 4, t)  # held for ~0.23 s
    assert kinds(ev).count("autoscroll_start") == 1 and sm.autoscroll.active


def test_brief_four_finger_flick_does_not_start():
    sm = GestureStateMachine()
    ev, t = feed(sm, [four()] * 3 + [pose("point")] * 3 + [four()] * 3)
    assert "autoscroll_start" not in kinds(ev)


def test_hand_above_the_anchor_scrolls_up_and_keeps_scrolling_while_still():
    sm = GestureStateMachine()
    ev, t = feed(sm, [four(center=(0.5, 0.5))] * 8)
    assert sm.autoscroll.active
    ev, t = feed(sm, [four(center=(0.5, 0.5 - 0.12))] * 30, t)  # 1 palm length up, then held still
    steps = [e for e in ev if e.kind == "autoscroll"]
    assert all(e.data["dy"] >= 0 for e in steps) and sum(e.data["dy"] for e in steps) > 3
    assert steps[-1].data["dy"] > 0 and steps[-1].data["direction"] == "up"  # still scrolling, hand still
    ev, t = feed(sm, [four(center=(0.5, 0.5 + 0.12))] * 10, t)
    assert ev[-1].data["dy"] < 0 and ev[-1].data["direction"] == "down"


def test_dead_zone_speed_curve_and_max_speed():
    assert axis_speed(0.2, 0.25, 12, 30) == 0.0
    assert 0 < axis_speed(0.5, 0.25, 12, 30) < axis_speed(1.0, 0.25, 12, 30)
    assert axis_speed(-1.0, 0.25, 12, 30) == -axis_speed(1.0, 0.25, 12, 30)
    assert axis_speed(50.0, 0.25, 12, 30) == 30.0
    sm = GestureStateMachine()
    _ev, t = feed(sm, [four()] * 8)
    ev, t = feed(sm, [four(center=(0.5, 0.505))] * 10, t)  # inside the dead zone
    assert all(e.data["dy"] == 0 and e.data["dx"] == 0 for e in ev if e.kind == "autoscroll")


def test_horizontal_autoscroll_uses_the_dominant_axis():
    sm = GestureStateMachine()
    _ev, t = feed(sm, [four(center=(0.5, 0.5))] * 8)
    ev, _t = feed(sm, [four(center=(0.5 + 0.15, 0.5 - 0.03))] * 10, t)
    last = ev[-1].data
    assert last["dx"] > 0 and last["dy"] == 0 and last["direction"] == "right"
    sm2 = GestureStateMachine(GestureConfig(autoscroll_horizontal=False))
    _ev, t = feed(sm2, [four(center=(0.5, 0.5))] * 8)
    ev, _t = feed(sm2, [four(center=(0.65, 0.5))] * 10, t)
    assert ev[-1].data["dx"] == 0


def test_fist_stops_autoscroll():
    sm = GestureStateMachine(GestureConfig(autoscroll_stop_hold=0.2))
    _ev, t = feed(sm, [four()] * 8)
    ev, t = feed(sm, [pose("fist")] * 3, t)
    assert "autoscroll_stop" not in kinds(ev)  # not held long enough yet
    ev, t = feed(sm, [pose("fist")] * 5, t)
    stop = [e for e in ev if e.kind == "autoscroll_stop"]
    assert stop and stop[0].data["reason"] == "fist" and not sm.autoscroll.active
    ev, _t = feed(sm, [pose("fist")] * 10, t)
    assert "autoscroll" not in kinds(ev)


def test_short_hand_dropout_resumes_long_one_stops():
    sm = GestureStateMachine()
    _ev, t = feed(sm, [four()] * 8)
    ev, t = feed(sm, [None] * 6, t)  # 0.2 s
    assert sm.autoscroll.active and "autoscroll_stop" not in kinds(ev)
    ev, t = feed(sm, [four()] * 2, t)
    assert "autoscroll" in kinds(ev)
    ev, t = feed(sm, [None] * int((LOST_STOP_S + 0.2) / DT), t)
    assert "autoscroll_stop" in kinds(ev) and not sm.autoscroll.active


def test_pause_and_other_gestures_stop_autoscroll():
    sm = GestureStateMachine()
    _ev, t = feed(sm, [four()] * 8)
    ev, t = feed(sm, [pose("point")] * int(OTHER_STOP_S / DT + 2), t)
    assert [e.data["reason"] for e in ev if e.kind == "autoscroll_stop"] == ["pose:point"]
    p = Pipeline(AppConfig(), RecordingBackend())
    t = 0.0
    for _ in range(8):
        t += DT
        p.on_hand(t, four())
    assert p.gestures.autoscroll.active
    p.set_paused(True)
    assert not p.gestures.autoscroll.active


def test_four_finger_pose_never_triggers_the_clutch():
    sm = GestureStateMachine()
    ev, t = feed(sm, [four()] * 90)  # 3 s of four fingers, held still
    assert "clutch" not in kinds(ev) and not sm.paused
    ev, t = feed(sm, [pose("fist")] * 10, t)
    ev, _t = feed(sm, [pose("palm")] * 20, t)  # the open palm still pauses
    assert "clutch" in kinds(ev)


def test_open_palm_during_autoscroll_stops_it_before_pausing():
    sm = GestureStateMachine()
    _ev, t = feed(sm, [four()] * 8)
    ev, _t = feed(sm, [pose("palm")] * 40, t)
    k = kinds(ev)
    assert k.index("autoscroll_stop") < k.index("clutch")


def test_disabled_autoscroll_keeps_the_old_four_finger_clutch():
    sm = GestureStateMachine(GestureConfig(autoscroll_enabled=False))
    ev, _t = feed(sm, [four()] * 20)
    assert "autoscroll_start" not in kinds(ev) and "clutch" in kinds(ev)


def test_autoscroll_reaches_the_mouse_and_the_overlay():
    b = RecordingBackend()
    p = Pipeline(AppConfig(), b)
    t = 0.0
    for lm in [four(center=(0.5, 0.5))] * 8 + [four(center=(0.5, 0.65))] * 20:
        t += DT
        p.on_hand(t, lm)
    scrolls = [c[1][0] for c in b.calls if c[0] == "scroll"]
    assert scrolls and all(s < 0 for s in scrolls)  # hand below the start: scroll down
    st = p.cursor_state()
    assert st["state"] == "autoscroll" and st["direction"] == "down"
    for _ in range(10):
        t += DT
        p.on_hand(t, four(center=(0.75, 0.5)))
    assert any(c[0] == "hscroll" and c[1][0] > 0 for c in b.calls)
    assert ("gesture:four_finger_scroll", "mouse:autoscroll") in p.dispatcher.fired


def test_four_finger_scroll_binding_rules():
    bc = BindingsConfig()
    assert bc.resolve("gestures", "four_finger_scroll") == Action("mouse", "autoscroll")
    with pytest.raises(ValueError):
        bc.set("gesture:pinch", "mouse:autoscroll")
    bc.set("gesture:four_finger_scroll", "key:pagedown")  # a discrete action fires once, on start
    b = RecordingBackend()
    cfg = AppConfig()
    cfg.bindings = bc
    p = Pipeline(cfg, b)
    t = 0.0
    for _ in range(20):
        t += DT
        p.on_hand(t, four())
    presses = [c for c in b.calls if c[0] in ("key", "hotkey")]
    assert len(presses) == 1 and not any(c[0] == "scroll" for c in b.calls)


def test_autoscroller_unit_lost_then_stop_requires_enabled():
    a = AutoScroller(GestureConfig(autoscroll_hold=0.1))
    assert a.update(0.0, "four", (0.5, 0.5), 0.1) is None
    assert a.update(0.15, "four", (0.5, 0.5), 0.1)[0] == "autoscroll_start"
    assert a.lost(0.2) is None and a.lost(0.8) == {"reason": "hand_lost"}
    assert a.update(1.0, "palm", (0.5, 0.5), 0.1) is None  # thumb out never starts it


def test_test_tab_describes_autoscroll(tmp_path):
    m = SettingsModel(AppConfig(), tmp_path / "c.yaml")
    assert m.describe_event(GestureEvent("autoscroll_start", 0.0)) == "four_finger_scroll  ->  mouse:autoscroll"
    assert "down" in str(m.describe_event(GestureEvent("autoscroll", 0.0, {"direction": "down", "vy": -9.0})))
    assert "fist" in str(m.describe_event(GestureEvent("autoscroll_stop", 0.0, {"reason": "fist"})))
    assert m.set_voice("hands.gestures.autoscroll_enabled", False) is None
    labels = {t.path: t.label for t in m.tunables()}
    assert labels["hands.cursor.big_cursor_size"] == "Cursor size while gestures are active (0 = off)"
    assert "hands.gestures.autoscroll_dead_zone" in labels and "hands.gestures.autoscroll_max_speed" in labels
    assert m.set_tunable("hands.cursor.big_cursor_size", 4) is None and m.cfg.hands.cursor.big_cursor_size == 4


# ============================================================================ clap control
def noise_bed(seconds, rng, level=0.003):
    return rng.normal(0, level, int(seconds * SR))


def clap_sound(rng, amp=0.6, tau=0.006):
    n = int(0.05 * SR)
    return amp * rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR / tau)


def vowel(dur=0.4, f0=130.0, amp=0.3, attack=0.03):
    t = np.arange(int(dur * SR)) / SR
    x = sum((1 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 20))
    env = np.minimum(1, t / attack) * np.minimum(1, (dur - t) / 0.05)
    return amp * x / np.max(np.abs(x)) * env


def plosive_word(rng):
    n = int(0.012 * SR)
    burst = 0.4 * rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR / 0.004)
    return np.concatenate([burst, np.zeros(int(0.02 * SR)), vowel(attack=0.01)])


def beep():
    t = np.arange(int(0.03 * SR)) / SR
    return 0.5 * np.sin(2 * np.pi * 2000 * t) * np.exp(-t / 0.008)


def scene(parts, seconds=None, rng=None):
    rng = rng or np.random.default_rng(0)
    end = max(t + len(s) / SR for t, s in parts) + 1.0
    x = noise_bed(seconds or end, rng)
    for t, s in parts:
        i = int(t * SR)
        x[i : i + len(s)] += s
    return x.astype(np.float32)


def stream(x, consumer, chunk=480):
    out = []
    for k in range(0, len(x), chunk):
        part = x[k : k + chunk]
        out += consumer(part, (k + len(part)) / SR)
    return out


def detect(x):
    d = CL.ClapDetector(SR)
    return stream(x, d.feed)


def test_clap_is_detected_with_its_onset_time():
    rng = np.random.default_rng(1)
    c = detect(scene([(1.0, clap_sound(rng))], rng=rng))
    assert len(c) == 1 and c[0].t == pytest.approx(1.0, abs=0.02) and c[0].score > 0.8
    assert c[0].flatness > 0.3 and c[0].decay < CL.DECAY_RATIO


@pytest.mark.parametrize("name", ["vowel", "plosive", "beep", "noise", "quiet"])
def test_speech_noise_and_beeps_are_not_claps(name):
    rng = np.random.default_rng(2)
    x = {
        "vowel": lambda: scene([(1.0, vowel())], rng=rng),
        "plosive": lambda: scene([(1.0, plosive_word(rng)), (1.6, plosive_word(rng))], rng=rng),
        "beep": lambda: scene([(1.0, beep())], rng=rng),
        "noise": lambda: rng.normal(0, 0.1, SR * 3).astype(np.float32),
        "quiet": lambda: scene([(1.0, clap_sound(rng, amp=0.01))], rng=rng),
    }[name]()
    assert detect(x) == []


def test_spectral_flatness_separates_noise_from_tones():
    rng = np.random.default_rng(3)
    assert CL.spectral_flatness(rng.normal(0, 1, 320), SR) > 0.4
    t = np.arange(320) / SR
    assert CL.spectral_flatness(np.sin(2 * np.pi * 1000 * t), SR) < 0.05


class Voice:
    clap_enabled = True
    clap_sensitivity = 0.5
    clap_double_window = 0.6
    clap_require_hands = False


def control(view="seen", busy=lambda: False, **kw):
    v = Voice()
    for k, val in kw.items():
        setattr(v, k, val)
    return CL.ClapControl(v, SR, busy=busy, hands=lambda _t: view), v


def test_single_clap_waits_for_the_double_window():
    rng = np.random.default_rng(4)
    ctl, _v = control()
    got = stream(scene([(1.0, clap_sound(rng))], rng=rng), ctl.feed)
    assert [k for k, _t in got] == ["single"]
    assert got[0][1] >= 1.0 + 0.6  # acted on only after the window passed


def test_double_clap_fires_once_and_never_a_single():
    rng = np.random.default_rng(5)
    ctl, _v = control()
    got = stream(scene([(1.0, clap_sound(rng)), (1.35, clap_sound(rng))], rng=rng), ctl.feed)
    assert [k for k, _t in got] == ["double"]
    assert got[0][1] == pytest.approx(1.35, abs=0.02)


def test_two_claps_further_apart_than_the_window_are_two_singles():
    rng = np.random.default_rng(6)
    ctl, _v = control(clap_double_window=0.4)
    got = stream(scene([(1.0, clap_sound(rng)), (1.8, clap_sound(rng))], rng=rng), ctl.feed)
    assert [k for k, _t in got] == ["single", "single"]


def test_third_quick_clap_after_a_double_is_ignored():
    timing = CL.ClapTiming(0.6)
    assert timing.on_clap(0.0) is None
    assert timing.on_clap(0.3) == "double"
    assert timing.on_clap(0.5) is None
    assert timing.tick(2.0) is None


def test_claps_during_dictation_are_ignored():
    rng = np.random.default_rng(7)
    state = {"busy": False}
    ctl, _v = control(busy=lambda: state["busy"])
    x = scene([(1.0, clap_sound(rng))], rng=rng)
    # the user has been speaking since 0.5 s (mid-utterance) when a plosive-like transient arrives at 1.0 s
    out = []
    for k in range(0, len(x), 480):
        t = (k + 480) / SR
        state["busy"] = 0.5 <= t <= 1.5
        out += ctl.feed(x[k : k + 480], t)
    assert out == []


def test_a_clap_that_itself_starts_speech_detection_still_counts_as_a_double():
    rng = np.random.default_rng(8)
    state = {"busy": False}
    ctl, _v = control(busy=lambda: state["busy"])
    x = scene([(1.0, clap_sound(rng)), (1.3, clap_sound(rng))], rng=rng)
    out = []
    for k in range(0, len(x), 480):
        t = (k + 480) / SR
        state["busy"] = 1.02 <= t <= 2.0  # the endpointer thinks the claps are speech
        out += ctl.feed(x[k : k + 480], t)
    assert [k for k, _t in out] == ["double"]


def test_camera_confirmation_thresholds():
    base = CL.audio_threshold(0.5)
    assert CL.hand_threshold(0.5, "together") < base < CL.hand_threshold(0.5, "none")
    assert CL.hand_threshold(0.5, "seen") == base
    assert CL.hand_threshold(0.5, "seen", require_hands=True) is None
    assert CL.hand_threshold(0.5, "together", require_hands=True) == pytest.approx(base - 0.15)
    assert CL.audio_threshold(1.0) < CL.audio_threshold(0.0)


def test_require_hands_rejects_audio_only_claps():
    rng = np.random.default_rng(9)
    x = scene([(1.0, clap_sound(rng))], rng=rng)
    ctl, _v = control(view="none", clap_require_hands=True)
    assert stream(x, ctl.feed) == []
    ctl2, _v = control(view="together", clap_require_hands=True)
    assert [k for k, _t in stream(x, ctl2.feed)] == ["single"]


def test_disabled_clap_control_does_nothing():
    rng = np.random.default_rng(10)
    ctl, _v = control(clap_enabled=False)
    assert stream(scene([(1.0, clap_sound(rng))], rng=rng), ctl.feed) == []


def test_camera_view_from_hand_frames():
    views = [(0.5, 1, None), (0.8, 2, 0.1)]
    assert CL.camera_view(views, 1.0) == "together"
    assert CL.camera_view([(0.5, 2, 0.6)], 1.0) == "seen"
    assert CL.camera_view([(0.1, 2, 0.1)], 1.0) == "none"  # too long before the clap
    p = Pipeline(AppConfig(), RecordingBackend())
    assert p.hands_view(1.0) == "none"
    p.note_hands(0.9, 2, 0.05)
    assert p.hands_view(1.0) == "together"


def test_double_clap_pauses_and_single_clap_resumes():
    p = Pipeline(AppConfig(), RecordingBackend())
    assert p.on_clap("double", 1.0) == ["double_clap"]
    assert p.paused and not p.gestures_active()
    assert [e.data["paused"] for e in p.bus.history("app.paused")] == [True]
    p.on_clap("single", 2.0)
    assert not p.paused and p.gestures_active()
    assert ("gesture:clap", "pause:off") in p.dispatcher.fired


def test_clap_settings_and_test_tab(tmp_path):
    m = SettingsModel(AppConfig(), tmp_path / "c.yaml")
    assert m.cfg.voice.clap_enabled and m.cfg.voice.clap_require_hands is False
    assert m.set_voice("voice.clap_require_hands", True) is None
    paths = {t.path for t in m.tunables()}
    assert {"voice.clap_sensitivity", "voice.clap_double_window"} <= paths
    rng = np.random.default_rng(11)
    out: queue.Queue[str] = queue.Queue()
    handler = clap_test_handler(m, out, [])
    m.cfg.voice.clap_require_hands = False
    x = scene([(1.0, clap_sound(rng)), (1.3, clap_sound(rng))], rng=rng)
    for k in range(0, len(x), 480):
        handler(x[k : k + 480], (k + 480) / SR)
    assert out.get_nowait() == "double_clap  ->  pause:on"
