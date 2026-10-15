"""v0.1.19 head tracking: pose from the transformation matrix, sign alignment, filters, pointer, coarse zones,
monitor switch hysteresis, and the screen-facing test for the speech gate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from away.control.monitors import Monitor
from away.face.headpose import (
    HeadCoarse,
    HeadPointer,
    HeadPose,
    HeadPoseFilter,
    MonitorSwitch,
    PoseBaseline,
    SignAligner,
    facing_screen,
    nose_offsets,
    pose_from_matrix,
    pose_from_points,
    rotation_matrix,
)
from away.face.landmarker import observe_face
from away.face.synthetic import face_mesh

TWO = [Monitor(0, 0, 1920, 1080, True), Monitor(1920, 0, 1920, 1080)]


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"), [(0, 0, 0), (25, 0, 0), (-30, 10, 0), (12, -18, 7), (-5, 22, -15), (60, 30, 20)]
)
def test_pose_from_matrix_round_trips(yaw, pitch, roll):
    p = pose_from_matrix(rotation_matrix(yaw, pitch, roll))
    assert p is not None
    assert (p.yaw, p.pitch, p.roll) == pytest.approx((yaw, pitch, roll), abs=1e-9)


def test_pose_from_matrix_ignores_scale_and_translation_and_rejects_garbage():
    m = np.array(rotation_matrix(20, -10, 5))
    m[:3, :3] *= 3.5  # the face model scale
    m[:3, 3] = (1.0, -2.0, -40.0)
    p = pose_from_matrix(m)
    assert p is not None and (p.yaw, p.pitch, p.roll) == pytest.approx((20, -10, 5), abs=1e-9)
    assert pose_from_matrix(None) is None
    assert pose_from_matrix([[1, 2]]) is None
    assert pose_from_matrix(np.zeros((4, 4))) is None
    bad = np.eye(4)
    bad[0, 0] = np.nan
    assert pose_from_matrix(bad) is None


def test_nose_offsets_and_geometric_pose_are_screen_aligned():
    xs, ys = face_mesh(0, 0)
    h, v = nose_offsets(xs, ys)
    assert abs(h) < 1e-9 and abs(v) < 0.05
    xs, ys = face_mesh(30, 0)
    assert nose_offsets(xs, ys)[0] > 0.1  # turned toward image right
    p_right, p_left = pose_from_points(*face_mesh(30, 0)), pose_from_points(*face_mesh(-30, 0))
    assert p_right is not None and p_left is not None
    assert p_right.yaw > 10 and p_left.yaw == pytest.approx(-p_right.yaw)
    down, up = pose_from_points(*face_mesh(0, 20)), pose_from_points(*face_mesh(0, -20))
    assert down is not None and up is not None and down.pitch > up.pitch
    rolled = pose_from_points(*face_mesh(0, 0, roll=15))
    assert rolled is not None and rolled.roll == pytest.approx(15, abs=0.5)
    assert nose_offsets([0.1] * 10, [0.1] * 10) is None


def test_sign_aligner_flips_an_anti_correlated_matrix_and_defaults_to_plus():
    a = SignAligner(decay=1.0, min_evidence=1.0)
    p = a.update(HeadPose(10, 5, 0), None)
    assert (p.yaw, p.pitch) == (10, 5)  # no evidence yet
    for k in range(20):  # the matrix says -yaw when the nose goes right (+); pitch agrees
        s = 1 if k % 2 else -1
        a.update(HeadPose(-20 * s, 10 * s, 0), (0.2 * s, 0.1 * s))
    assert a.yaw_sign == -1.0 and a.pitch_sign == 1.0
    out = a.update(HeadPose(-15, 8, 3), None)
    assert (out.yaw, out.pitch, out.roll) == (15, 8, 3)


def test_pose_filter_smooths_jitter():
    f = HeadPoseFilter(min_cutoff=0.5, beta=0.0)
    rng = np.random.default_rng(3)
    raw = [10 + rng.normal(0, 2) for _ in range(120)]
    out = [f(i / 30, HeadPose(r, 0, 0)).yaw for i, r in enumerate(raw)]
    assert np.std(out[30:]) < 0.3 * np.std(raw[30:])
    f.configure(5.0, 0.1)
    assert f.f[0].min_cutoff == 5.0 and f.f[2].beta == 0.1


def test_pose_baseline_warms_up_fast_then_moves_slowly():
    b = PoseBaseline(tau_s=60.0, warm_tau_s=0.5, warm_s=3.0)
    t = 0.0
    for _ in range(100):  # 3.3 s at yaw 20
        t += 0.033
        b.update(t, HeadPose(20, -5, 0))
    assert b.ready and b.yaw == pytest.approx(20, abs=0.5) and b.pitch == pytest.approx(-5, abs=0.2)
    for _ in range(30):  # 1 s glance at yaw 50: barely moves the slow baseline
        t += 0.033
        b.update(t, HeadPose(50, -5, 0))
    assert b.yaw is not None and b.yaw < 21


def test_facing_screen_uses_the_calibrated_box_or_the_usual_pose():
    assert facing_screen(HeadPose(0, 0, 0), None, 20) is None
    box = (10.0, 40.0, -5.0, 10.0)  # an off-axis camera: the screens are at yaw 10..40
    assert facing_screen(HeadPose(35, 0, 0), None, 5, box) is True
    assert facing_screen(HeadPose(44, 0, 0), None, 5, box) is True  # within tolerance
    assert facing_screen(HeadPose(46, 0, 0), None, 5, box) is False
    assert facing_screen(HeadPose(20, 16, 0), None, 5, box) is False
    b = PoseBaseline(warm_tau_s=0.1, warm_s=0.5)
    for i in range(30):
        b.update(i * 0.033, HeadPose(30, 0, 0))
    assert facing_screen(HeadPose(50, 0, 0), b, 25) is True
    assert facing_screen(HeadPose(60, 0, 0), b, 25) is False
    assert facing_screen(HeadPose(30, 26, 0), b, 25) is False


def test_head_pointer_moves_with_turns_eats_tremor_and_accumulates_slow_turns():
    hp = HeadPointer(gain=0.01, dead_zone=0.3, acceleration=1.0)
    assert hp.update(HeadPose(0, 0, 0), 1000) == (0.0, 0.0)  # first frame anchors
    assert hp.update(HeadPose(0.1, 0, 0), 1000) == (0.0, 0.0)  # tremor
    assert hp.update(HeadPose(0.2, 0, 0), 1000) == (0.0, 0.0)
    dx, dy = hp.update(HeadPose(0.4, 0, 0), 1000)  # slow turn: everything past the dead zone goes through
    assert dx == pytest.approx(1.0) and dy == 0.0  # v0.1.26: slack (0.4 - 0.3), not a 4 px hop
    dx, dy = hp.update(HeadPose(0.4, 2.4, 0), 1000)  # look down -> cursor down
    assert dy == pytest.approx(21.0, abs=0.1) and 0.0 < dx < 0.15 * dy
    hp.reset()
    assert hp.update(HeadPose(50, 0, 0), 1000) == (0.0, 0.0)


def test_head_pointer_acceleration_makes_fast_turns_go_further():
    slow, fast = HeadPointer(0.01, 0.0, 1.5), HeadPointer(0.01, 0.0, 1.5)
    slow.update(HeadPose(0, 0, 0), 1000)
    fast.update(HeadPose(0, 0, 0), 1000)
    d_slow = slow.update(HeadPose(1, 0, 0), 1000)[0]
    d_fast = fast.update(HeadPose(4, 0, 0), 1000)[0]
    assert d_fast / d_slow == pytest.approx(4 * 2.0)  # 4x the turn, (4)^0.5 = 2x the gain


def test_head_coarse_zones_need_hysteresis_and_dwell():
    hc = HeadCoarse(cols=3, rows=3, yaw_span=30, pitch_span=15, hysteresis=0.25, dwell_s=0.3)
    t = 0.0

    def run(offset, secs):
        nonlocal t
        out = []
        for _ in range(round(secs / 0.033)):
            t += 0.033
            r = hc.update(t, offset, TWO)
            if r is not None:
                out.append(r)
        return out

    assert run((0.0, 0.0), 0.5) == []  # the first zone is adopted without a jump
    assert hc.zone == (0, 2, 1)  # centre of the desktop = right column of the left monitor
    # a little into the zone on the right (x 1984, monitor 2's first zone): inside the hysteresis band
    assert run((1.0, 0.0), 1.0) == []
    # a short glance far left: shorter than the dwell
    assert run((-25.0, 0.0), 0.2) == [] and hc.zone == (0, 2, 1)
    jumps = run((-25.0, 0.0), 0.5)
    assert jumps == [pytest.approx((320.0, 540.0))] and hc.zone == (0, 0, 1)
    assert run((-25.0, 0.0), 1.0) == []  # stays: no repeated jumps


def test_head_coarse_point_mapping_is_clamped():
    hc = HeadCoarse(yaw_span=20, pitch_span=10)
    assert hc.point((0, 0), TWO) == pytest.approx((3839 / 2, 1079 / 2))
    assert hc.point((100, 100), TWO) == pytest.approx((3839, 1079))
    assert hc.zone_of(5000, 500, TWO)[0] == 1
    assert hc.center_of((0, 2, 0), TWO) == pytest.approx((1600, 180))


def test_monitor_switch_hysteresis_and_dwell():
    ms = MonitorSwitch([(-15.0, 0.0), (15.0, 0.0)], margin=4.0, dwell_s=0.5)
    t = 0.0

    def run(yaw, secs, current):
        nonlocal t
        res = []
        for _ in range(round(secs / 0.033)):
            t += 0.033
            r = ms.update(t, HeadPose(yaw, 0, 0), current)
            if r is not None:
                res.append(r)
        return res

    got = run(10.0, 0.4, 0)  # 20 deg closer to monitor 2, but shorter than the dwell
    assert got == []
    got = run(10.0, 0.3, 0)
    assert got == [1]  # switches once the dwell is complete
    assert run(10.0, 1.0, 1) == []  # already there
    ms.reset()
    assert run(-14.0, 1.0, 1) == [0]


def test_monitor_switch_ignores_glances_and_the_middle():
    ms = MonitorSwitch([(-15.0, 0.0), (15.0, 0.0)], margin=4.0, dwell_s=0.5)
    t = 0.0
    out = []
    for _ in range(10):  # a 0.33 s glance at the other monitor
        t += 0.033
        out.append(ms.update(t, HeadPose(15, 0, 0), 0))
    for _ in range(30):  # back to the current one
        t += 0.033
        out.append(ms.update(t, HeadPose(-15, 0, 0), 0))
    for _ in range(60):  # 2 s in the middle (1.5 deg toward monitor 2: inside the margin)
        t += 0.033
        out.append(ms.update(t, HeadPose(1.5, 0, 0), 0))
    assert all(o is None for o in out)
    res = []
    for _ in range(20):  # a sustained turn: switches once after the dwell
        t += 0.033
        res.append(ms.update(t, HeadPose(14, 0, 0), 0))
    switched = [i for i, r in enumerate(res) if r is not None]
    assert switched and res[switched[0]] == 1 and switched[0] >= 14  # >= 0.5 s of frames
    assert MonitorSwitch([(0.0, 0.0)]).update(1.0, HeadPose(40, 0, 0), 0) is None


def test_observe_face_carries_pose_proxy_and_gaze_features():
    xs, ys = face_mesh(20, 5, iris_h=0.1)
    o = observe_face(xs, ys, {"jawOpen": 0.1}, rotation_matrix(-20, 5, 0))
    assert o.pose == pytest.approx((-20, 5, 0), abs=1e-9)  # matrix angles as given (sign aligned later)
    assert o.proxy is not None and o.proxy[0] > 0
    assert o.gaze is not None and o.gaze.shape == (11,) and o.gaze[0] > 0
    assert o.box[2] == pytest.approx(0.3, abs=0.01)
    no_matrix = observe_face(xs, ys, {})
    assert no_matrix.pose is not None and no_matrix.pose[0] > 0  # geometric fallback
    few = observe_face(xs[:468], ys[:468], {})
    assert few.gaze is None and few.pose is not None
    assert math.isfinite(few.box[0])


def test_sign_aligner_is_not_fooled_by_a_constant_proxy_offset():
    """A real nose sits below the brow-chin midpoint (v offset < 0): agreeing pitch must not be flipped."""
    a = SignAligner(min_evidence=0.5)
    for k in range(200):
        s = math.sin(k / 8)
        a.update(HeadPose(10 * s, 8 * s + 4, 0), (0.1 * s, 0.05 * s - 0.15))
    assert a.yaw_sign == 1.0 and a.pitch_sign == 1.0
    b = SignAligner(min_evidence=0.5)
    for k in range(200):
        s = math.sin(k / 8)
        b.update(HeadPose(-10 * s, -8 * s, 0), (0.1 * s, 0.05 * s - 0.15))
    assert b.yaw_sign == -1.0 and b.pitch_sign == -1.0
