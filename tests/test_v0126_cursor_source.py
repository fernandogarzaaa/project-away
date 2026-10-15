"""v0.1.26: one cursor source at a time. With eye/head tracking on, a still pointing hand must keep the cursor
still (noisy face landmarks, hand-detection dropouts), and a fast reach must stay quick.

Fake landmarks only: synthetic hand poses with landmark noise and dropouts, synthetic face meshes looking at a
point on the desktop with gaze and head-pose noise, fed at the owner's camera rate (30 fps)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app.runner import Pipeline
from away.control.monitors import Monitor
from away.control.mouse import RecordingBackend
from away.face import gaze as gz
from away.face.gaze import LookToJump
from away.face.headpose import HeadPointer, HeadPose, rotation_matrix
from away.face.landmarker import observe_face
from away.face.synthetic import SyntheticViewer
from away.hands.synthetic import pose as hand_pose
from away.pointer.arbiter import CursorArbiter, RateMeter
from away.pointer.glide import Glide

TWO = [Monitor(0, 0, 1920, 1080, True, "A"), Monitor(1920, 0, 1920, 1080, False, "B")]
FPS = 30.0


def save_calibration() -> None:
    rng = np.random.default_rng(11)
    v = SyntheticViewer(bounds=gz.desktop_bounds(TWO), camera_yaw=0.0)
    samples = []
    for k, (mon, x, y) in enumerate(gz.calibration_targets(TWO, 9)):
        for _ in range(8):
            xs, ys, _pose, _box = v.look(x, y, rng)
            hyaw, hpitch = (a * v.head_share for a in v.angles(x, y))
            o = observe_face(xs, ys, {}, rotation_matrix(hyaw, hpitch, 0.0))
            samples.append(gz.Sample(k, mon, x, y, o.gaze, o.pose))
    gz.save_calibration(gz.calibration_from_samples(samples, TWO, 1.0, "0"))


def make(head_assist: str = "off", monitor_switch: bool = False, calibrate: bool = True) -> tuple[Pipeline, Any]:
    if calibrate:
        save_calibration()
    cfg = C.AppConfig()
    cfg.hands.cursor.head_assist = head_assist
    cfg.hands.cursor.head.monitor_switch = monitor_switch
    b = RecordingBackend()
    p = Pipeline(cfg, backend=b, screen=TWO)
    p.headgaze.reload(force=True)
    return p, b


class NoisyFace:
    """A face looking at ``look`` with gaze-point noise (px) and head-pose noise (degrees)."""

    def __init__(
        self, look: tuple[float, float], rng: np.random.Generator, gaze_px: float = 150.0, head_deg: float = 1.5
    ):
        self.v = SyntheticViewer(bounds=gz.desktop_bounds(TWO), camera_yaw=0.0)
        self.look, self.rng, self.gaze_px, self.head_deg = look, rng, gaze_px, head_deg

    def feed(self, p: Pipeline, t: float) -> None:
        x, y = self.look
        r = self.rng
        xs, ys, _pose, _box = self.v.look(x + r.normal(0, self.gaze_px), y + r.normal(0, self.gaze_px), r)
        hyaw, hpitch = (a * self.v.head_share for a in self.v.angles(x, y))
        hyaw += r.normal(0, self.head_deg)
        hpitch += r.normal(0, self.head_deg)
        o = observe_face(xs, ys, {}, rotation_matrix(hyaw, hpitch, 0.0))
        p.on_faces(t, [o.box], [o.blendshapes], [o.frontal], extras=[o])


def dropped(n: int) -> bool:
    """Hand-detection dropouts like a 640x480 webcam: a single missed frame now and then, and short bursts."""
    return n % 13 == 0 or n % 47 in (20, 21, 22)


SCENARIOS = {
    # gaze calibrated, look-to-jump on (the defaults), user reading the other monitor
    "gaze_look_to_jump": dict(head_assist="off", look=(3300, 300)),
    # head pointer: the head drives the cursor only without a hand
    "head_pointer": dict(head_assist="pointer", look=(3300, 300)),
    # head zones: the head turned toward a far zone while the hand rests
    "head_coarse": dict(head_assist="coarse", look=(3500, 900)),
    # head-turn monitor switch with the head toward the other monitor
    "head_monitor_switch": dict(head_assist="off", monitor_switch=True, look=(3300, 500)),
    # no detection dropouts at all: landmark noise alone must not trigger anything
    "gaze_look_to_jump_no_dropouts": dict(head_assist="off", look=(3300, 300), drops=False),
    "head_pointer_no_dropouts": dict(head_assist="pointer", look=(3300, 300), drops=False),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_still_hand_keeps_the_cursor_still_with_eye_and_head_tracking_on(name: str) -> None:
    sc = dict(SCENARIOS[name])
    look = sc.pop("look")
    drops = sc.pop("drops", True)
    p, b = make(**sc)  # type: ignore[arg-type]
    rng = np.random.default_rng(3)
    face = NoisyFace(look, rng)
    p.mouse.mapper.set_position(600, 500)
    t = 0.0
    for _ in range(int(4.0 * FPS)):  # the usual head pose is learned before the hand arrives
        t += 1 / FPS
        face.feed(p, t)
    settled: tuple[float, float] | None = None
    seen: list[tuple[float, float]] = []
    moves_before = 0
    writes: dict[str, int] = {}
    events_before = 0
    for n in range(1, int(7.0 * FPS) + 1):
        t += 1 / FPS
        c = (0.5 + rng.normal(0, 0.002), 0.5 + rng.normal(0, 0.002))  # a still hand, ~1 px landmark noise
        p.on_hand(t, None if drops and dropped(n) else hand_pose("point", center=c))
        face.feed(p, t)
        if n == int(1.5 * FPS):
            settled = p.mouse.mapper.pos
            moves_before = len(b.calls)
            writes = dict(p.cursor_arbiter.writes)
            events_before = len(p.bus.history())
        elif settled is not None:
            seen.append(p.mouse.mapper.pos)
    assert settled is not None
    worst = max(math.hypot(q[0] - settled[0], q[1] - settled[1]) for q in seen)
    sent = [c[1] for c in b.calls[moves_before:] if c[0] == "move"]
    worst_sent = max((math.hypot(x - settled[0], y - settled[1]) for x, y in sent), default=0.0)
    assert worst < 1.0, f"{name}: a still hand moved the cursor {worst:.1f} px"
    assert worst_sent < 1.0, f"{name}: the OS cursor was moved {worst_sent:.1f} px"
    late = [e.topic for e in p.bus.history()[events_before:]]
    assert "gaze.jump" not in late and "head.monitor_switch" not in late
    assert p.cursor_arbiter.writes["head"] == writes["head"] and p.cursor_arbiter.writes["gaze"] == writes["gaze"]
    assert p.cursor_arbiter.owner == "hand"


REACH_S = 0.25  # hand travel time of the reach
REACH = 0.15  # hand travel (frame widths)


def reach(p: Pipeline, face: NoisyFace | None, rng: np.random.Generator) -> list[tuple[float, tuple[float, float]]]:
    """Rest 1 s, reach REACH to the right in REACH_S, hold 1 s. Returns (time since the hand arrived, cursor)."""
    t, out = 0.0, []
    if face is not None:
        for _ in range(int(4.0 * FPS)):  # (the head pointer may move the cursor here: no hand yet)
            t += 1 / FPS
            face.feed(p, t)
    p.mouse.mapper.set_position(400, 540)
    t0 = t + 1.0
    for _ in range(int(2.4 * FPS)):
        t += 1 / FPS
        k = min(1.0, max(0.0, (t - t0) / REACH_S))
        x = 0.42 + REACH * (k * k * (3 - 2 * k))
        c = (x + rng.normal(0, 0.002), 0.5 + rng.normal(0, 0.002))
        p.on_hand(t, hand_pose("point", center=c))
        if face is not None:
            face.feed(p, t)
        out.append((t - t0 - REACH_S, p.mouse.mapper.pos))
    return out


def lag_90(path: list[tuple[float, tuple[float, float]]]) -> tuple[float, float]:
    """(seconds after the hand arrived until the cursor covered 90% of its move, the move in px)."""
    start = next(pos for dt, pos in path if dt >= -REACH_S)[0]
    final = path[-1][1][0]
    t90 = next(dt for dt, pos in path if pos[0] - start >= 0.9 * (final - start))
    return t90, final - start


def test_fast_reach_stays_quick_and_eye_head_tracking_adds_nothing() -> None:
    p0, _ = make(calibrate=False)
    hand_only = reach(p0, None, np.random.default_rng(5))
    p, _ = make(head_assist="pointer")
    # the user looks near the cursor (no look-to-jump), noisy gaze and head
    with_face = reach(p, NoisyFace((440, 560), np.random.default_rng(9), gaze_px=60.0), np.random.default_rng(5))
    t90, moved = lag_90(with_face)
    assert moved > 300  # it really is a reach
    assert t90 <= 0.17, f"the cursor reached 90% {t90 * 1000:.0f} ms after the hand stopped"
    # with eye/head tracking on, the hand path is exactly the hand-only path
    assert max(abs(a[1][0] - b[1][0]) + abs(a[1][1] - b[1][1]) for a, b in zip(hand_only, with_face)) < 1.0


# --------------------------------------------------------------------------------------------- the pieces
def test_head_pointer_still_works_without_a_hand_and_hands_over_on_dropouts_only_after_the_hold() -> None:
    p, b = make(head_assist="pointer", calibrate=False)
    rng = np.random.default_rng(1)
    t = 0.0
    for k in range(60):  # head turns right, no hand: the head drives
        t += 1 / FPS
        face = NoisyFace((960 + 40 * k, 540), rng, gaze_px=0.0, head_deg=0.0)
        face.feed(p, t)
    assert p.cursor_arbiter.writes["head"] > 0
    assert any(e.data["source"] == "head" for e in p.bus.history("cursor.source"))
    # the hand arrives: it owns the cursor, also through a 3-frame detection dropout
    p.on_hand(t + 0.03, hand_pose("point", center=(0.5, 0.5)))
    assert p.cursor_arbiter.owner == "hand"
    before = p.cursor_arbiter.writes["head"]
    for k in range(3):
        p.on_hand(t + 0.06 + k * 0.033, None)
        NoisyFace((3000, 540), rng, gaze_px=0.0, head_deg=0.0).feed(p, t + 0.06 + k * 0.033)
    assert p.cursor_arbiter.writes["head"] == before
    assert p.cursor_arbiter.hand_owns(t + 0.2) and not p.cursor_arbiter.hand_owns(t + 1.0)


def test_arbiter_logs_the_driving_source_and_per_source_rates() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    a = CursorArbiter(publish=lambda k, d: events.append((k, d)), stats_every_s=1.0)
    t = 0.0
    for i in range(60):
        t = i / 30
        a.note_hand(t, True)
        a.claim(t, "hand")
        if i % 2 == 0:
            a.note_face(t, pose=True, gaze=True)
    a.note_cost("hand", 12.0)
    assert not a.claim(t, "head") and a.denied["head"] == 1
    st = a.stats(t)
    assert (
        st["source"] == "hand"
        and st["hand_fps"] == pytest.approx(30, abs=1)
        and st["face_fps"] == pytest.approx(15, abs=1)
    )
    assert st["hand_ms"] == 12.0
    assert ("cursor.source", {"source": "hand", "was": "none"}) in events
    assert [k for k, _ in events].count("cursor.stats") >= 2
    assert a.claim(t + 1.0, "head") and a.owner == "head"
    assert RateMeter().fps(0.0) == 0.0


def test_face_model_runs_lighter_while_the_hand_drives() -> None:
    a = CursorArbiter()
    assert all(a.face_due(i / 30) for i in range(10))  # no hand: every frame
    a.note_hand(1.0, True)
    due = [a.face_due(1.0 + i / 30) for i in range(1, 16)]
    assert 6 <= sum(due) <= 9  # ~15 Hz of a 30 fps camera


def test_look_to_jump_ignores_landmark_noise_but_fires_on_a_deliberate_move() -> None:
    rng = np.random.default_rng(2)
    j = LookToJump(0.6, 250.0)
    t = 0.0
    for _ in range(600):  # 20 s of a resting hand with 640x480 landmark noise
        t += 1 / FPS
        h = (0.5 + rng.normal(0, 0.003), 0.5 + rng.normal(0, 0.003))
        assert j.update(t, h, (200, 200), (3000, 400)) is None
    t += 1 / FPS
    assert j.update(t, (0.53, 0.5), (200, 200), (3000, 400)) == (3000, 400)
    # a slow drift is not an onset
    j.reset()
    x = 0.5
    for _ in range(120):
        t += 1 / FPS
        x += 0.0008
        assert j.update(t, (x, 0.5), (200, 200), (3000, 400)) is None


def test_head_pointer_dead_zone_is_slack_not_a_staircase() -> None:
    hp = HeadPointer(gain=0.012, dead_zone=0.3, acceleration=1.4)
    rng = np.random.default_rng(4)
    total = 0.0
    hp.update(HeadPose(0.0, 0.0, 0.0), 1920)
    for _ in range(300):  # tremor within the dead zone never moves the cursor
        dx, dy = hp.update(HeadPose(rng.normal(0, 0.08), rng.normal(0, 0.08), 0.0), 1920)
        total += abs(dx) + abs(dy)
    assert total == 0.0
    moved = sum(hp.update(HeadPose(0.1 * k, 0.0, 0.0), 1920)[0] for k in range(1, 51))
    assert moved > 0.9 * 0.012 * 1920 * (5.0 - 0.3)


def test_short_hand_dropout_resumes_without_a_kick_long_one_reanchors() -> None:
    p, b = make(calibrate=False)
    m = p.mouse.mapper
    rng = np.random.default_rng(6)
    t = 0.0
    for _ in range(45):
        t += 1 / FPS
        p.on_hand(t, hand_pose("point", center=(0.5 + rng.normal(0, 0.002), 0.5 + rng.normal(0, 0.002))))
    pos = m.pos
    p.on_hand(t + 0.033, None)
    assert m._suspended is not None
    p.on_hand(t + 0.066, hand_pose("point", center=(0.5, 0.5)))
    assert m._suspended is None and m._prev is not None and m.pos == pos
    p.on_hand(t + 0.1, None)
    p.on_hand(t + 1.0, hand_pose("point", center=(0.8, 0.8)))  # back after 0.9 s somewhere else: re-anchored
    assert m.pos == pos and m._suspended is None


def test_head_writes_go_through_the_output_glide() -> None:
    p, b = make(head_assist="pointer", calibrate=False)
    p.mouse.glide = Glide()
    p.mouse.follow(100, 200)
    assert p.mouse.glide.target == (100.0, 200.0) and not b.calls
    p.mouse.jump_to(300, 400)
    assert p.mouse.glide.out == (300.0, 400.0) and b.calls[-1] == ("move", (300, 400))
