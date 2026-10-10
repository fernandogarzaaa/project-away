"""Multi-camera fusion on synthetic data: two virtual cameras with known extrinsics."""

import math

import cv2
import numpy as np
import pytest

from away import config as C
from away.app.runner import Pipeline
from away.config import AppConfig, CameraSpec, MultiCamConfig
from away.control.mouse import RecordingBackend
from away.hands.landmarks import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, pinch_ratio
from away.hands.synthetic import make_hand, pose
from away.multicam import geometry as G
from away.multicam.capture import CameraDevice, FramePairer, list_cameras, resolve_device
from away.multicam.fusion import (
    MultiViewFuser,
    PushDetector,
    ViewObs,
    hand_crop,
    pinch_ratio_3d,
    sharpness_score,
    view_confidence,
    visibility,
)

W1, H1, W2, H2 = 640, 480, 1280, 720


def rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) * math.radians(deg))[0]


def rig():
    """Primary at the origin; secondary 35 cm to the right, turned 25 degrees toward the user."""
    K1 = G.intrinsics_from_fov(W1, H1, 70)
    K2 = G.intrinsics_from_fov(W2, H2, 80)
    R = rot([0, 1, 0], 25) @ rot([1, 0, 0], 5)
    t = -R @ np.array([0.35, -0.05, 0.02])
    return K1, K2, R, t


def hand3d(lm2d=None):
    """A metric 3D hand (palm 9 cm) built from the synthetic 2D template, with some finger depth."""
    base = np.asarray(lm2d if lm2d is not None else pose("palm"), float)[:, :3].copy()
    base -= base[WRIST]
    base[:, 2] = 0.0
    h = base / np.linalg.norm(base[MIDDLE_MCP, :2]) * 0.09
    h[:, 2] = np.linspace(0.0, 0.02, 21)
    return h


def hand_frames(n, rng, depth=(0.4, 0.7)):
    h = hand3d()
    out = []
    for _ in range(n):
        Rh = rot(rng.normal(size=3), rng.uniform(0, 40))
        pos = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(*depth)])
        out.append(h @ Rh.T + pos)
    return out


def obs_from_3d(X, cam, K, R=None, t=None, size=(W1, H1), score=0.95, mirrored=True, ts=0.0):
    px = G.project(X, K, R, t)
    lm = G.from_pixels(px, *size, mirrored=mirrored)
    return ViewObs(cam, ts, np.column_stack([lm, np.zeros(21)]), "Right", score, size)


def extrinsics(K1, K2, R, t):
    return G.Extrinsics(
        G.CameraCalib("0", W1, H1, K1),
        [G.CameraCalib("1", W2, H2, K2, np.zeros(5), R, t)],
        "synthetic",
        0.0,
        True,
    )


CAMS = [CameraSpec(0, "primary", W1, H1), CameraSpec(1, "secondary", W2, H2, hfov_deg=80)]


# --------------------------------------------------------------------------- config
def test_single_camera_default_is_unchanged():
    cfg = AppConfig()
    assert cfg.hands.cameras == [] and cfg.hands.camera_index == 0
    cfg2, unknown = C.from_dict({"hands": {"camera_index": 2}})
    assert cfg2.hands.cameras == [] and unknown == []


def test_cameras_config_parse_and_validate(tmp_path):
    cfg, unknown = C.from_dict(
        {
            "hands": {
                "cameras": [
                    {"device": 0, "role": "primary", "width": 1280, "height": 720, "fps": 30},
                    {"device": "/dev/video2", "role": "secondary", "hfov_deg": 90, "bogus": 1},
                ]
            }
        }
    )
    a, b = cfg.hands.cameras
    assert (a.device, a.width, b.device, b.role, b.hfov_deg) == (0, 1280, "/dev/video2", "secondary", 90)
    assert unknown == ["hands.cameras[1].bogus"]
    short, _ = C.from_dict({"hands": {"cameras": [0, 2]}})
    assert [c.role for c in short.hands.cameras] == ["primary", "secondary"]
    p = C.save(cfg, tmp_path / "c.yaml")
    again, _ = C.load(p)
    assert again.hands.cameras == cfg.hands.cameras
    for bad in (
        [{"device": 0}, {"device": 1, "role": "primary"}],
        [{"device": 0, "role": "side"}],
        [{"device": 0, "fps": 1}],
        [{"device": 0}, {"device": 0, "role": "secondary"}],
    ):
        with pytest.raises(ValueError):
            C.from_dict({"hands": {"cameras": bad}})
    with pytest.raises(ValueError, match=r"multicam\.fusion"):
        C.from_dict({"hands": {"multicam": {"fusion": "magic"}}})


# --------------------------------------------------------------------------- geometry
def test_intrinsics_and_mirror_roundtrip():
    K = G.intrinsics_from_fov(640, 480, 90)
    assert K[0, 0] == pytest.approx(320.0) and K[0, 2] == 320 and K[1, 2] == 240
    lm = np.random.default_rng(0).uniform(0, 1, (21, 2))
    px = G.to_pixels(lm, 640, 480, mirrored=True)
    assert np.allclose(G.from_pixels(px, 640, 480, mirrored=True), lm)
    assert px[0, 0] == pytest.approx((1 - lm[0, 0]) * 640)


def test_triangulation_is_exact_without_noise():
    K1, K2, R, t = rig()
    X = hand_frames(1, np.random.default_rng(1))[0]
    Xr = G.triangulate(G.project(X, K1), G.project(X, K2, R, t), K1, K2, R, t)
    assert np.abs(Xr - X).max() < 1e-9


def test_pose_recovered_from_hand_motion(rng):
    K1, K2, R, t = rig()
    frames = hand_frames(40, rng)
    X = np.concatenate(frames)
    p1 = G.project(X, K1) + rng.normal(0, 0.5, (len(X), 2))
    p2 = G.project(X, K2, R, t) + rng.normal(0, 0.5, (len(X), 2))
    est = G.estimate_pose_from_hands(p1, p2, K1, K2, palm_length_m=0.09, frames=40)
    assert G.rotation_angle_deg(R, est.R) < 1.5
    direction = np.dot(est.t / np.linalg.norm(est.t), t / np.linalg.norm(t))
    assert direction > 0.995
    assert np.linalg.norm(est.t) == pytest.approx(np.linalg.norm(t), rel=0.1)  # palm-length scale
    assert est.reprojection_error_px < 5.0 and est.inliers > 0.6 * est.total
    # triangulation error with the recovered pose, noisy pixels: a few millimetres to a centimetre
    Xr = G.triangulate(p1, p2, K1, K2, est.R, est.t)
    assert np.median(np.linalg.norm(Xr - X, axis=1)) < 0.02
    with pytest.raises(ValueError):
        G.estimate_pose_from_hands(p1[:5], p2[:5], K1, K2)


def test_stereo_checkerboard_calibration():
    K1, K2, R, t = rig()
    board = G.checkerboard_points(7, 5, 0.03)
    rng = np.random.default_rng(3)
    obj, i1, i2 = [], [], []
    for _ in range(12):
        Rb = rot([rng.uniform(-1, 1), rng.uniform(-1, 1), 0], rng.uniform(10, 35))
        pos = np.array([rng.uniform(-0.12, 0.05), rng.uniform(-0.08, 0.03), rng.uniform(0.45, 0.7)])
        Xb = board @ Rb.T + pos
        obj.append(board)
        i1.append(G.project(Xb, K1))
        i2.append(G.project(Xb, K2, R, t))
    K1e, _d1, K2e, _d2, Re, te, rms = G.stereo_calibrate(obj, i1, i2, (W1, H1), (W2, H2))
    assert rms < 0.5
    assert G.rotation_angle_deg(R, Re) < 0.5
    assert np.linalg.norm(te - t) < 0.01
    img = G.make_checkerboard_image(7, 5, 40, 20)
    assert img.shape == (6 * 40 + 40, 8 * 40 + 40) and img[20, 20] == 0 and img[20, 60] == 255


def test_extrinsics_save_load_roundtrip(tmp_path):
    K1, K2, R, t = rig()
    ex = extrinsics(K1, K2, R, t)
    p = G.save_extrinsics(ex)
    assert p.parent.name == "data"  # user data dir (AWAY_DATA_DIR in tests)
    back = G.load_extrinsics()
    assert back is not None and np.allclose(back.secondaries[0].R, R) and back.metric
    assert back.for_device("1") is not None and back.for_device("9") is None
    (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(KeyError):
        G.Extrinsics.from_dict({})
    assert G.load_extrinsics(tmp_path / "bad.json") is None
    assert G.load_extrinsics(tmp_path / "missing.json") is None


def test_monocular_depth_tracks_hand_size():
    near = make_hand(scale=0.2)
    far = make_hand(scale=0.1)
    dn = G.monocular_depth(near, 640, 480, 70)
    df = G.monocular_depth(far, 640, 480, 70)
    assert dn is not None and df is not None and df == pytest.approx(2 * dn, rel=0.01)


# --------------------------------------------------------------------------- confidence + fusion
def test_view_confidence_terms():
    big = ViewObs(0, 0.0, make_hand(scale=0.2), score=0.9)
    small = ViewObs(0, 0.0, make_hand(scale=0.03), score=0.9)
    off = ViewObs(0, 0.0, make_hand(center=(0.98, 0.5), scale=0.2), score=0.9)
    blurry = ViewObs(0, 0.0, make_hand(scale=0.2), score=0.9, sharpness=5.0)
    assert view_confidence(big) > view_confidence(small)
    assert visibility(off.landmarks) < 1.0 and view_confidence(off) < view_confidence(big)
    assert view_confidence(blurry) < view_confidence(big)
    rng = np.random.default_rng(0)
    sharp = rng.uniform(0, 255, (40, 40))
    smooth = cv2.GaussianBlur(sharp, (9, 9), 4)
    assert sharpness_score(sharp) > 10 * sharpness_score(smooth)
    assert sharpness_score(np.zeros((2, 2))) == 0.0
    crop = hand_crop(np.zeros((480, 640)), big.landmarks)
    assert 0 < crop.shape[0] < 480 and 0 < crop.shape[1] < 640


def test_best_view_selection_and_weighted_average():
    good = ViewObs(0, 0.0, make_hand(center=(0.4, 0.5), scale=0.2), score=0.95)
    weak = ViewObs(1, 0.0, make_hand(center=(0.6, 0.5), scale=0.2), score=0.3, size=(W2, H2))
    f = MultiViewFuser(CAMS, MultiCamConfig(fusion="best_view", min_confidence=0.05))
    out = f.update(0.0, {0: good, 1: weak})
    assert out is not None and out.source == "cam0" and np.allclose(out.landmarks, good.landmarks)
    # secondary only (never seen together): its raw view is used
    f2 = MultiViewFuser(CAMS, MultiCamConfig(fusion="best_view", min_confidence=0.05))
    out2 = f2.update(0.0, [None, weak])
    assert out2 is not None and out2.source == "cam1"
    # weighted: the result lies between the two aligned views, closer to the confident one
    fw = MultiViewFuser(CAMS, MultiCamConfig(fusion="weighted", min_confidence=0.05))
    fw.update(0.0, {0: good, 1: good.__class__(1, 0.0, good.landmarks.copy(), score=0.95)})
    shifted = ViewObs(1, 0.1, good.landmarks + [0.02, 0.0, 0.0], score=0.3)
    out3 = fw.update(0.1, {0: good, 1: shifted})
    assert out3 is not None and "+" in out3.source and out3.confidences[0] > out3.confidences[1]
    assert fw.update(0.2, {0: None, 1: None}) is None
    with pytest.raises(ValueError):
        MultiViewFuser([])


def test_calibrated_fusion_triangulates_depth_and_3d_pinch():
    K1, K2, R, t = rig()
    f = MultiViewFuser(CAMS, MultiCamConfig(), extrinsics(K1, K2, R, t))
    assert f.calibrated
    X = hand3d() + [0.0, 0.0, 0.5]
    o1 = obs_from_3d(X, 0, K1)
    o2 = obs_from_3d(X, 1, K2, R, t, size=(W2, H2))
    out = f.update(0.0, {0: o1, 1: o2})
    assert out is not None and out.source == "3d" and out.points3d is not None
    assert np.abs(out.points3d - X).max() < 1e-6
    assert out.depth == pytest.approx(np.mean(X[[0, 5, 9, 13, 17], 2]), abs=1e-6)
    assert np.allclose(out.landmarks[:, :2], o1.landmarks[:, :2], atol=1e-9)  # reprojected into the primary
    assert out.pinch == pytest.approx(pinch_ratio_3d(X))
    # extrinsics for other devices are ignored
    f.set_extrinsics(G.Extrinsics(G.CameraCalib("7", W1, H1, K1), [], "x", 0.0, False))
    assert not f.calibrated


def test_3d_pinch_rejects_edge_on_false_pinch():
    """Thumb and index overlap in the primary image but are 5 cm apart in depth: 2D says pinch, 3D says no."""
    K1, K2, R, t = rig()
    X = hand3d() + [0.0, 0.0, 0.5]
    X[THUMB_TIP] = X[INDEX_TIP] + [0.0, 0.0, 0.05]
    ray = X[THUMB_TIP] / X[THUMB_TIP][2]
    X[THUMB_TIP] = ray * X[THUMB_TIP][2]  # keep it on the same pixel ray as the index tip...
    X[THUMB_TIP][:2] = X[INDEX_TIP][:2] / X[INDEX_TIP][2] * X[THUMB_TIP][2]
    o1 = obs_from_3d(X, 0, K1)
    o2 = obs_from_3d(X, 1, K2, R, t, size=(W2, H2))
    cfg = AppConfig()
    assert pinch_ratio(o1.landmarks) < cfg.hands.gestures.pinch_on  # a single camera would click
    out = MultiViewFuser(CAMS, MultiCamConfig(), extrinsics(K1, K2, R, t)).update(0.0, [o1, o2])
    assert out is not None and out.pinch is not None and out.pinch > cfg.hands.gestures.pinch_off


def test_dropout_handover_keeps_cursor_continuous():
    """Both cameras see the hand, then the primary is occluded: no hand_lost, no jump, offset fades."""
    mc = MultiCamConfig(fusion="best_view", handover_decay_s=0.5, min_confidence=0.05)
    f = MultiViewFuser(CAMS, mc)
    cfg = AppConfig()
    cfg.hands.cursor = C.CursorConfig(mode="absolute", min_cutoff=1e6, d_cutoff=1e6, dead_zone=0.0)
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    t = 0.0
    wrist = []
    for i in range(30):
        t += 1 / 30
        c = (0.45 + 0.002 * i, 0.5)
        prim = ViewObs(0, t, make_hand(center=c, scale=0.2), score=0.95)
        # the secondary sees the same hand shifted and smaller (different viewpoint)
        sec_lm = make_hand(center=(c[0] + 0.1, c[1] - 0.05), scale=0.15)
        sec = ViewObs(1, t, sec_lm, score=0.6, size=(W2, H2))
        occluded = i >= 15
        out = f.update(t, {0: None if occluded else prim, 1: sec})
        assert out is not None
        wrist.append(out.landmarks[WRIST, :2].copy())
        p.on_fused(out)
        if i == 15:
            assert out.source == "cam1"
    steps = np.linalg.norm(np.diff(np.array(wrist), axis=0), axis=1)
    assert steps.max() < 0.01  # the switch at frame 15 is not a jump
    assert not [e for e in p.bus.history() if e.topic == "gesture.hand_lost"]
    assert np.linalg.norm(f._current_offset(t + 5.0)) < 1e-3  # handover offset decays away
    p.on_fused(None, t + 0.1)
    assert p.bus.history("gesture.hand_lost")


# --------------------------------------------------------------------------- pairing + devices
def test_frame_pairer_nearest_timestamp_and_dropout():
    now = [0.0]
    fp = FramePairer(2, primary=0, tolerance_s=0.02, dropout_s=0.2, clock=lambda: now[0])
    assert fp.pop_set() is None
    for t in (0.000, 0.033, 0.066):
        fp.push(0, t, f"p{t}")
    for t in (0.010, 0.041, 0.075, 0.110):
        fp.push(1, t, f"s{t}")
    now[0] = 0.07
    t_ref, frames = fp.pop_set()
    assert t_ref == 0.066 and frames[0][1] == "p0.066" and frames[1][1] == "s0.075"  # nearest within 20 ms
    assert fp.pop_set() is None  # nothing new from the pacing camera
    fp.push(0, 0.2, "p0.2")
    now[0] = 0.2
    _t, frames = fp.pop_set()
    assert frames[1] is None  # closest secondary frame is 90 ms away: not paired
    # primary goes silent: the secondary takes over pacing
    fp.push(1, 0.5, "s0.5")
    now[0] = 0.5
    assert fp.alive() == [False, True]
    t_ref, frames = fp.pop_set()
    assert t_ref == 0.5 and frames[0] is None and frames[1][1] == "s0.5"
    now[0] = 2.0
    assert fp.pop_set() is None  # everyone dropped


def test_list_cameras_from_sysfs_and_resolve(tmp_path):
    sysfs = tmp_path / "v4l"
    for n, name, idx in ((0, "Integrated Webcam", "0"), (1, "Integrated Webcam", "1"), (2, "Logitech BRIO", "0")):
        d = sysfs / f"video{n}"
        d.mkdir(parents=True)
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(idx + "\n")
    devs = list_cameras(str(sysfs), "/dev")
    assert [(c.index, c.path, c.name) for c in devs] == [
        (0, "/dev/video0", "Integrated Webcam"),
        (2, "/dev/video2", "Logitech BRIO"),
    ]
    assert resolve_device(1) == 1 and resolve_device("3") == 3 and resolve_device("/dev/video9") == "/dev/video9"
    assert resolve_device("brio", devs) == 2
    assert resolve_device("x", [CameraDevice(None, "/dev/videoX", "x cam")]) == "/dev/videoX"
    with pytest.raises(ValueError, match="cameras list"):
        resolve_device("nope", devs)


# --------------------------------------------------------------------------- push gesture
def test_push_detector():
    pd = PushDetector(distance_m=0.06, time_s=0.35, cooldown_s=0.6)
    assert not pd.update(0.0, 0.60)
    assert not pd.update(0.1, 0.58)
    assert pd.update(0.2, 0.53)  # 7 cm closer in 0.2 s
    assert not pd.update(0.3, 0.45)  # cooldown
    assert not pd.update(1.0, None)
    for i in range(10):  # slow drift toward the screen is not a push
        assert not pd.update(2.0 + i * 0.2, 0.6 - i * 0.01)


def test_push_gesture_fires_binding_through_pipeline():
    cfg = AppConfig()
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    lm = pose("point")
    t = 0.0
    for d in (0.6, 0.6, 0.58, 0.55, 0.52):
        t += 0.05
        p.on_hand(t, lm, depth=d)
    assert p.bus.history("gesture.push") and ("click", (1,)) in b.calls
    assert ("gesture:push", "mouse:click") in p.dispatcher.fired
    cfg.bindings.set("gesture:push", "key:enter")
    p2 = Pipeline(cfg, RecordingBackend())
    for i, d in enumerate((0.6, 0.6, 0.55, 0.5)):
        p2.on_hand(i * 0.05, lm, depth=d)
    assert ("key", ("enter",)) in p2.mouse.backend.calls


def test_monocular_push_option():
    mc = MultiCamConfig(push_monocular=True)
    f = MultiViewFuser([CameraSpec(0)], mc)
    out = f.update(0.0, [ViewObs(0, 0.0, make_hand(scale=0.2))])
    assert out is not None and out.depth is not None and out.points3d is None


# --------------------------------------------------------------------------- calibration sessions + CLI
def test_hand_motion_calibrator_end_to_end(rng):
    K1, K2, R, t = rig()
    from away.multicam.calibration import HandMotionCalibrator

    cal = HandMotionCalibrator(CAMS, min_frames=30)
    assert cal.progress() == 0.0
    ex0, rep0 = cal.solve()
    assert ex0 is None and not rep0.ok
    for X in hand_frames(45, rng):
        o1 = obs_from_3d(X, 0, K1, mirrored=CAMS[0].mirror)
        o2 = obs_from_3d(X, 1, K2, R, t, size=(W2, H2), mirrored=CAMS[1].mirror)
        cal.add([o1, o2])
    cal.add([None, None])
    assert cal.progress() == 1.0
    ex, rep = cal.solve(0.09)
    assert ex is not None and rep.ok and ex.method == "hand"
    sec = ex.for_device("1")
    assert sec is not None and G.rotation_angle_deg(R, sec.R) < 1.0
    assert ex.reprojection_error_px < 2.0 and "reprojection error" in rep.message
    # the saved calibration drives triangulating fusion
    f = MultiViewFuser(CAMS, MultiCamConfig(), ex)
    X = hand3d() + [0.02, 0.0, 0.55]
    out = f.update(0.0, [obs_from_3d(X, 0, K1), obs_from_3d(X, 1, K2, R, t, size=(W2, H2))])
    assert out is not None and out.source == "3d"
    assert out.depth == pytest.approx(np.mean(X[[0, 5, 9, 13, 17], 2]), rel=0.15)
    with pytest.raises(ValueError):
        HandMotionCalibrator(CAMS[:1])


def test_checkerboard_calibrator_detect_and_solve():
    from away.multicam.calibration import CheckerboardCalibrator

    K1, K2, R, t = rig()
    cal = CheckerboardCalibrator(CAMS, 7, 5, 0.03)
    img = G.make_checkerboard_image(7, 5, 40, 40)
    corners = cal.detect(np.dstack([img] * 3), mirrored=False)
    assert corners is not None and corners.shape == (35, 2)
    assert cal.detect(np.zeros((100, 100, 3), np.uint8), mirrored=True) is None
    assert cal.add([np.dstack([img] * 3), None]) == []  # board only in the primary
    board = G.checkerboard_points(7, 5, 0.03)
    rng = np.random.default_rng(5)
    for _ in range(10):
        Rb = rot([rng.uniform(-1, 1), rng.uniform(-1, 1), 0], rng.uniform(10, 35))
        Xb = board @ Rb.T + [rng.uniform(-0.12, 0.05), rng.uniform(-0.08, 0.03), rng.uniform(0.45, 0.7)]
        cal.add_corners(1, G.project(Xb, K1), G.project(Xb, K2, R, t), (W1, H1), (W2, H2))
    ex, rep = cal.solve(min_views=8)
    assert ex is not None and rep.ok and ex.method == "checkerboard" and ex.reprojection_error_px < 0.5
    assert np.linalg.norm(ex.secondaries[0].t - t) < 0.01
    few = CheckerboardCalibrator(CAMS)
    assert few.solve()[0] is None


def test_cli_cameras_list_add_remove_and_board(tmp_path, capsys):
    from away.cli import main

    cfgp = tmp_path / "c.yaml"
    assert main(["cameras", "list", "--config", str(cfgp)]) == 0
    assert "single-camera mode" in capsys.readouterr().out
    assert main(["cameras", "add", "2", "--config", str(cfgp)]) == 0
    cfg, _ = C.load(cfgp)
    assert [(c.device, c.role) for c in cfg.hands.cameras] == [(0, "primary"), (2, "secondary")]
    assert main(["cameras", "add", "2", "--config", str(cfgp)]) == 2  # duplicate device rejected
    assert main(["cameras", "list", "--config", str(cfgp)]) == 0
    assert "configured (hands.cameras)" in capsys.readouterr().out
    assert main(["cameras", "remove", "9", "--config", str(cfgp)]) == 2
    assert main(["cameras", "remove", "2", "--config", str(cfgp)]) == 0
    cfg, _ = C.load(cfgp)
    assert cfg.hands.cameras == [] and cfg.hands.camera_index == 0
    png = tmp_path / "board.png"
    assert main(["calibrate-cameras", "--print-board", str(png), "--checkerboard", "7x5"]) == 0
    assert cv2.imread(str(png)).shape[:2] == (6 * 80 + 80, 8 * 80 + 80)
    assert main(["calibrate-cameras", "--config", str(cfgp)]) == 2
    assert "needs two cameras" in capsys.readouterr().err
