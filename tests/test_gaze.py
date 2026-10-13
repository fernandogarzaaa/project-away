"""v0.1.19 eye tracking + screen map: iris features, ridge regression, calibration session and storage,
the gaze estimator with drift correction, and the look-to-jump trigger."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from away.app.gaze_ui import FaceSample, drive
from away.control.monitors import Monitor
from away.face import gaze as gz
from away.face.synthetic import SyntheticViewer, face_mesh

TWO = [Monitor(0, 0, 1920, 1080, True), Monitor(1920, 0, 1920, 1080)]


# ------------------------------------------------------------------------------------------ features
def test_features_need_iris_points():
    xs, ys = face_mesh()
    assert gz.gaze_features(xs[:468], ys[:468], None) is None
    assert gz.gaze_features(None, None, None) is None
    xs2 = xs.copy()
    xs2[5] = np.nan
    assert gz.gaze_features(xs2, ys, None) is None


def test_centred_iris_gives_zero_offsets_and_layout_is_stable():
    xs, ys = face_mesh(iris_h=0.0, iris_v=0.0, lid_open=0.3)
    f = gz.gaze_features(xs, ys, (10.0, -5.0, 2.0), (0.5, 0.5, 0.3))
    assert f is not None and f.shape == (len(gz.BASE_FEATURES),)
    assert f[:4] == pytest.approx([0, 0, 0, 0], abs=1e-9)
    assert f[4] == pytest.approx(0.3 * 0.25 / 0.25, abs=1e-6)  # lid gap / eye width
    assert f[5:8] == pytest.approx([10 / 30, -5 / 30, 2 / 30])
    assert f[8:11] == pytest.approx([0.5, 0.5, 0.3])
    fn = gz.gaze_features(xs, ys, None, None, net=(0.1, -0.2))
    assert fn is not None and fn.shape == (len(gz.BASE_FEATURES) + 2,) and fn[-2:] == pytest.approx([0.1, -0.2])


def test_iris_offsets_follow_the_eyes_and_ignore_position_size_and_roll():
    base = gz.gaze_features(*face_mesh(iris_h=0.1, iris_v=-0.05), None)
    assert base is not None
    assert base[0] == pytest.approx(0.1 * 0.25 / 0.25) and base[2] == pytest.approx(base[0])
    assert base[1] == pytest.approx(-0.05) and base[3] == pytest.approx(base[1])
    moved = gz.gaze_features(*face_mesh(iris_h=0.1, iris_v=-0.05, center=(0.3, 0.7), size=0.15), None)
    rolled = gz.gaze_features(*face_mesh(iris_h=0.1, iris_v=-0.05, roll=20), None)
    assert moved is not None and rolled is not None
    assert moved[:5] == pytest.approx(base[:5], abs=1e-9)
    assert rolled[:5] == pytest.approx(base[:5], abs=1e-9)
    assert moved[8:11] != pytest.approx(base[8:11])  # but the face position/size features do change


# ------------------------------------------------------------------------------------------ regression
def test_ridge_fit_recovers_a_known_mapping_and_round_trips():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(200, 11))
    tx = 1000 + 400 * f[:, 0] - 50 * f[:, 5] + 30 * f[:, 5] ** 2
    ty = 500 + 300 * f[:, 1] + 20 * f[:, 6]
    bounds = (0.0, 0.0, 3840.0, 1080.0)
    fit = gz.fit(f, np.stack([tx, ty], 1), bounds, alpha=1e-6, groups=list(range(200)))
    assert fit.rms_px < 1.0 and fit.loo_px is not None and fit.loo_px < 2.0
    x, y = fit.predict(f[7])
    assert (x, y) == pytest.approx((tx[7], ty[7]), abs=1.0)
    again = gz.GazeFit.from_dict(json.loads(json.dumps(fit.to_dict())))
    assert again.predict(f[7]) == pytest.approx((x, y))


def test_regularisation_shrinks_and_bad_input_is_refused():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(30, 11))
    t = np.stack([1000 + 300 * f[:, 0], 500 + 0 * f[:, 0]], 1)
    loose = gz.fit(f, t, (0, 0, 3840, 1080), alpha=1e-6)
    tight = gz.fit(f, t, (0, 0, 3840, 1080), alpha=1e4)
    assert tight.rms_px > loose.rms_px
    with pytest.raises(ValueError):
        gz.fit(f[:2], t[:2], (0, 0, 10, 10))
    with pytest.raises(ValueError):
        gz.fit(f, t, (0, 0, 0, 10))
    with pytest.raises(ValueError):
        gz.fit(f, t[:5], (0, 0, 10, 10))


# ------------------------------------------------------------------------------------------ targets
def test_targets_cover_every_monitor_inside_its_rectangle():
    for n in (5, 9, 16):
        tg = gz.calibration_targets(TWO, n)
        assert len(tg) == 2 * n
        for mon, x, y in tg:
            assert TWO[mon].contains(x, y)
    nine = gz.calibration_targets(TWO, 9)
    assert nine[0] == pytest.approx((0, 0.1 * 1919, 0.1 * 1079))
    assert nine[13] == pytest.approx((1, 1920 + 0.5 * 1919, 0.5 * 1079))
    assert gz.desktop_bounds(TWO) == (0.0, 0.0, 3840.0, 1080.0)


# ------------------------------------------------------------------------------------------ calibration session
def test_session_discards_the_settle_time_and_needs_min_frames():
    s = gz.CalibrationSession([(0, 10, 10), (0, 20, 20)], settle_s=0.5, collect_s=0.5, min_frames=5, timeout_s=5)
    f = np.zeros(11)
    t = 0.0
    moved = []
    while not s.done and t < 10:
        moved.append(s.feed(t, f))
        t += 0.05
    assert s.done and sum(moved) == 2 and not s.skipped
    for k in (0, 1):
        n = [x for x in s.samples if x.target == k]
        assert len(n) >= 5
    # no sample of dot 0 came from its settle window: of the ~21 frames it was shown, ~10 were dropped
    assert 10 <= len([x for x in s.samples if x.target == 0]) <= 11


def test_session_skips_a_dot_without_a_face_after_the_timeout():
    s = gz.CalibrationSession([(0, 1, 1), (0, 2, 2)], settle_s=0.1, collect_s=0.2, min_frames=2, timeout_s=1.0)
    t = 0.0
    while s.index == 0:
        s.feed(t, None)
        t += 0.05
    assert s.skipped == [0] and t == pytest.approx(1.05, abs=0.06)
    assert s.progress(t) < 0.5


def synthetic_samples(viewer, targets, rng, frames=12, net=False):
    out = []
    for k, (mon, x, y) in enumerate(targets):
        for _ in range(frames):
            xs, ys, pose, box = viewer.look(x, y, rng)
            netang = None
            if net:
                gyaw, gpitch = viewer.angles(x, y)
                netang = (math.radians(gyaw) + rng.normal(0, 0.02), math.radians(gpitch) + rng.normal(0, 0.02))
            f = gz.gaze_features(xs, ys, pose, box, netang)
            out.append(gz.Sample(k, mon, x, y, f, pose))
    return out


def test_end_to_end_calibration_on_a_synthetic_user_is_accurate_and_honest():
    rng = np.random.default_rng(5)
    viewer = SyntheticViewer(bounds=gz.desktop_bounds(TWO), noise=0.004)
    cal = gz.calibration_from_samples(synthetic_samples(viewer, gz.calibration_targets(TWO, 9), rng), TWO, 1.0, "cam")
    acc = cal.accuracy_px()
    assert acc is not None and acc < 150  # leave-one-dot-out on 3840 px of desktop
    est = gz.GazeEstimator(cal, min_cutoff=1e6, beta=0.0)
    errs = []
    for i, (x, y) in enumerate(((700, 300), (2600, 800), (1950, 540), (3300, 200))):  # points that were not dots
        xs, ys, pose, box = viewer.look(x, y, rng)
        px, py = est.update(float(i), gz.gaze_features(xs, ys, pose, box))
        errs.append(math.hypot(px - x, py - y))
    assert max(errs) < 250
    assert cal.monitor_poses[0][0] < cal.monitor_poses[1][0]  # the head turns right for the right monitor
    assert cal.pose_box is not None and cal.pose_box[0] < 20 < cal.pose_box[1]  # off-axis camera at +20 deg
    assert cal.n_dots == 18 and cal.net_model is None and set(cal.fits) == {"iris"}


def test_calibration_with_the_learned_network_adds_a_net_fit():
    rng = np.random.default_rng(6)
    viewer = SyntheticViewer(bounds=gz.desktop_bounds(TWO), noise=0.01)
    samples = synthetic_samples(viewer, gz.calibration_targets(TWO, 9), rng, net=True)
    cal = gz.calibration_from_samples(samples, TWO, 1.0, "cam", net_model="mobilegaze")
    assert set(cal.fits) == {"iris", "net"} and cal.net_model == "mobilegaze"
    assert cal.fits["net"].n_features == len(gz.BASE_FEATURES) + 2
    assert cal.best_fit(True)[0] == "net" and cal.best_fit(False)[0] == "iris"
    # the extra angles are really used (non-zero weights) and do not make the map worse on this synthetic user
    # (whose head pose is noise-free, so the network cannot add much here; real benefit is untested)
    w = np.asarray(cal.fits["net"].weights)
    assert np.abs(w[1 + len(gz.BASE_FEATURES) : 3 + len(gz.BASE_FEATURES)]).max() > 1e-4
    assert cal.fits["net"].loo_px < 1.1 * cal.fits["iris"].loo_px


def test_too_few_dots_is_an_error():
    rng = np.random.default_rng(7)
    viewer = SyntheticViewer(bounds=gz.desktop_bounds(TWO))
    samples = synthetic_samples(viewer, gz.calibration_targets(TWO, 9)[:3], rng)
    with pytest.raises(ValueError, match="at least 4"):
        gz.calibration_from_samples(samples, TWO)


def test_drive_runs_a_session_from_a_stream_and_reports_cancel():
    rng = np.random.default_rng(8)
    viewer = SyntheticViewer(bounds=gz.desktop_bounds(TWO))
    targets = gz.calibration_targets(TWO, 5)
    s = gz.CalibrationSession(targets, settle_s=0.2, collect_s=0.3, min_frames=4)
    shown = []

    def stream():
        t = 0.0
        while True:
            cur = s.current
            if cur is None:
                return
            xs, ys, pose, box = viewer.look(cur[1], cur[2], rng)
            yield FaceSample(t, gz.gaze_features(xs, ys, pose, box), pose)
            t += 0.033

    assert drive(s, stream(), lambda tgt, p: shown.append((tgt, p)))
    assert s.done and shown[0] == (targets[0], 0.0) and shown[-1][0] is None
    cal = s.result(TWO, 1.0, "cam")
    assert cal.n_dots == 10
    s2 = gz.CalibrationSession(targets)
    assert not drive(s2, stream(), cancelled=lambda: True)
    assert not drive(gz.CalibrationSession(targets), iter([]))


# ------------------------------------------------------------------------------------------ storage
def make_cal(camera="cam", monitors=TWO):
    rng = np.random.default_rng(9)
    viewer = SyntheticViewer(bounds=gz.desktop_bounds(monitors))
    return gz.calibration_from_samples(
        synthetic_samples(viewer, gz.calibration_targets(monitors, 5), rng, frames=6), monitors, 1.0, camera
    )


def test_calibrations_are_stored_per_camera_and_cleared(tmp_path):
    p = tmp_path / "gaze.json"
    a, b = make_cal("HD Webcam"), make_cal("Brio")
    a.bias = (12.0, -3.0)
    gz.save_calibration(a, p)
    gz.save_calibration(b, p)
    got = gz.load_calibration("HD Webcam", p)
    assert got is not None and got.bias == (12.0, -3.0) and got.camera == "HD Webcam"
    assert got.fits["iris"].predict(np.zeros(11)) == pytest.approx(a.fits["iris"].predict(np.zeros(11)))
    assert set(gz.list_calibrations(p)) == {"HD Webcam", "Brio"}
    assert gz.load_calibration("nope", p) is None
    assert gz.clear_calibration("Brio", p) and not gz.clear_calibration("Brio", p)
    assert set(gz.list_calibrations(p)) == {"HD Webcam"}
    assert gz.clear_calibration(None, p) and not p.exists()
    p.write_text("{not json", encoding="utf-8")
    assert gz.load_calibration("HD Webcam", p) is None
    p.write_text(json.dumps({"cameras": {"x": {"fits": "broken"}}}), encoding="utf-8")
    assert gz.load_calibration("x", p) is None
    assert gz.calibration_file().name == gz.FILE_NAME  # in the (test-isolated) config folder


def test_layout_matching_and_monitor_poses():
    cal = make_cal()
    assert cal.matches_layout(list(reversed(TWO)))
    assert not cal.matches_layout([TWO[0]])
    assert not cal.matches_layout([TWO[0], Monitor(1920, 0, 2560, 1440)])
    poses = cal.monitor_poses_for(TWO)
    assert poses is not None and len(poses) == 2
    assert cal.monitor_poses_for([Monitor(0, 0, 800, 600)]) is None


# ------------------------------------------------------------------------------------------ estimator + drift
def test_estimator_clamps_to_the_desktop_and_applies_the_bias():
    cal = make_cal()
    est = gz.GazeEstimator(cal, min_cutoff=1e6, beta=0.0)
    f = np.zeros(11)
    f[0] = 50.0  # absurd iris offset: the prediction is clamped onto the desktop
    x, y = est.update(0.0, f)
    assert 0 <= x <= 3839 and 0 <= y <= 1079
    assert est.fresh(0.3) is not None and est.fresh(1.0) is None
    est.bias = (100.0, 0.0)
    rng = np.random.default_rng(1)
    xs, ys, pose, box = SyntheticViewer(bounds=gz.desktop_bounds(TWO)).look(1000, 500, rng)
    feats = gz.gaze_features(xs, ys, pose, box)
    x1, _ = est.update(1.0, feats)
    assert est.raw is not None and x1 == pytest.approx(est.raw[0])
    est.bias = (0.0, 0.0)
    x0, _ = est.update(2.0, feats)
    assert x1 - x0 == pytest.approx(100.0, abs=0.01)
    assert est.update(3.0, np.array([np.nan] * 11)) == est.point  # garbage keeps the last point


def test_drift_correction_is_small_bounded_and_ignores_far_clicks():
    cal = make_cal()
    est = gz.GazeEstimator(cal, 1e6, 0.0)
    assert not est.correct((10, 10), 0.1, 100, 300)  # no prediction yet
    est.raw = (1000.0, 500.0)
    assert est.correct((1100.0, 500.0), 0.1, 150, 300)
    assert est.bias == pytest.approx((10.0, 0.0)) and cal.bias == est.bias
    assert not est.correct((1800.0, 500.0), 0.1, 150, 300)  # 800 px away: not looking there
    for _ in range(200):
        est.correct((1290.0, 500.0), 0.5, 150, 300)
    assert math.hypot(*est.bias) == pytest.approx(150.0)  # capped
    est.correct((1290.0, 500.0), 0.5, 0, 300)
    assert est.bias == (0.0, 0.0)
    assert not est.correct((1290.0, 500.0), 0.0, 150, 300)


# ------------------------------------------------------------------------------------------ look-to-jump
def run_jump(lj, frames, cursor=(100.0, 100.0), gaze=(2500.0, 500.0), **kw):
    out = []
    for t, hand in frames:
        out.append(lj.update(t, hand, cursor, gaze, **kw))
    return out


def test_look_to_jump_fires_once_when_moving_after_rest():
    lj = gz.LookToJump(idle_s=0.6, min_px=250)
    rest = [(i * 0.033, (0.5, 0.5)) for i in range(25)]  # 0.8 s still
    move = [(0.83 + i * 0.033, (0.5 + 0.01 * (i + 1), 0.5)) for i in range(5)]
    out = run_jump(lj, rest + move)
    fired = [o for o in out if o is not None]
    assert fired == [(2500.0, 500.0)] and out.index(fired[0]) == 25  # the first moving frame, once


def test_look_to_jump_needs_the_rest_the_distance_and_a_gaze_point():
    lj = gz.LookToJump(idle_s=0.6, min_px=250)
    short_rest = [(i * 0.033, (0.5, 0.5)) for i in range(10)] + [(0.4, (0.52, 0.5))]
    assert all(o is None for o in run_jump(lj, short_rest))
    lj.reset()
    rest_move = [(i * 0.033, (0.5, 0.5)) for i in range(25)] + [(0.9, (0.52, 0.5))]
    assert all(o is None for o in run_jump(lj, rest_move, cursor=(2400.0, 450.0)))  # gaze ~112 px away
    lj.reset()
    assert all(o is None for o in run_jump(lj, rest_move, gaze=None))
    lj.reset()
    jitter = [(i * 0.033, (0.5 + 0.002 * (i % 2), 0.5)) for i in range(25)] + [(0.9, (0.53, 0.5))]
    assert run_jump(lj, jitter)[-1] == (2500.0, 500.0)  # tracking jitter still counts as resting


def test_look_to_jump_across_monitors_only():
    lj = gz.LookToJump(idle_s=0.3, min_px=100)

    def mon(x, y):
        return 0 if x < 1920 else 1

    frames = [(i * 0.033, (0.5, 0.5)) for i in range(15)] + [(0.6, (0.53, 0.5))]
    same = run_jump(lj, frames, cursor=(200.0, 200.0), gaze=(1500.0, 800.0), monitor_of=mon, across_only=True)
    assert all(o is None for o in same)
    lj.reset()
    other = run_jump(lj, frames, cursor=(200.0, 200.0), gaze=(2500.0, 800.0), monitor_of=mon, across_only=True)
    assert other[-1] == (2500.0, 800.0)
