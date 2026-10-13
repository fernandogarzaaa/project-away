"""v0.1.19 head tracking + gaze wired into the Pipeline: head pointer, coarse zones, monitor switch,
look-to-jump, drift correction on clicks, the face gate, live calibration reload, config + tunables."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app.runner import Pipeline
from away.app.settings_gui import SettingsModel
from away.control.monitors import Monitor
from away.control.mouse import RecordingBackend
from away.face import gaze as gz
from away.face.headgaze import HeadGaze, calibration_camera
from away.face.headpose import rotation_matrix
from away.face.landmarker import observe_face
from away.face.synthetic import SyntheticViewer, face_mesh
from away.hands.gestures import GestureEvent
from away.hands.synthetic import pose as hand_pose

TWO = [Monitor(0, 0, 1920, 1080, True, "A"), Monitor(1920, 0, 1920, 1080, False, "B")]


def pipeline(**cursor: Any) -> tuple[Pipeline, RecordingBackend]:
    cfg = C.AppConfig()
    for k, v in cursor.items():
        setattr(cfg.hands.cursor, k, v)
    b = RecordingBackend()
    return Pipeline(cfg, backend=b, screen=TWO), b


def face(yaw: float = 0.0, pitch: float = 0.0, **kw: Any) -> Any:
    xs, ys = face_mesh(yaw, pitch, **kw)
    return observe_face(xs, ys, {}, rotation_matrix(yaw, pitch, 0.0))


def feed_faces(p: Pipeline, t0: float, secs: float, yaw: float, pitch: float = 0.0) -> float:
    t = t0
    for _ in range(round(secs / 0.033)):
        t += 0.033
        o = face(yaw, pitch)
        p.on_faces(t, [o.box], [o.blendshapes], [o.frontal], extras=[o])
    return t


def save_synthetic_calibration(camera: str = "0", monitors: list[Monitor] = TWO) -> gz.Calibration:
    rng = np.random.default_rng(11)
    v = SyntheticViewer(bounds=gz.desktop_bounds(monitors), camera_yaw=0.0)
    samples = []
    for k, (mon, x, y) in enumerate(gz.calibration_targets(monitors, 9)):
        for _ in range(8):
            xs, ys, _pose, box = v.look(x, y, rng)
            hyaw, hpitch = (a * v.head_share for a in v.angles(x, y))
            o = observe_face(xs, ys, {}, rotation_matrix(hyaw, hpitch, 0.0))
            samples.append(gz.Sample(k, mon, x, y, o.gaze, o.pose))
    cal = gz.calibration_from_samples(samples, monitors, 1.0, camera)
    gz.save_calibration(cal)
    return cal


def look_at(p: Pipeline, t: float, x: float, y: float) -> None:
    v = SyntheticViewer(bounds=gz.desktop_bounds(TWO), camera_yaw=0.0)
    xs, ys, _pose, _box = v.look(x, y)
    hyaw, hpitch = (a * v.head_share for a in v.angles(x, y))
    o = observe_face(xs, ys, {}, rotation_matrix(hyaw, hpitch, 0.0))
    p.on_faces(t, [o.box], [o.blendshapes], [o.frontal], extras=[o])


# ------------------------------------------------------------------------------------------ config
def test_defaults_are_off_or_inert_until_calibrated():
    cfg = C.AppConfig()
    c = cfg.hands.cursor
    assert c.head_assist == "off" and c.head.monitor_switch is False
    assert c.autocenter.enabled and c.autocenter.soft_recenter and c.autocenter.edge_reanchor
    assert cfg.gaze.model == "auto" and cfg.gaze.look_to_jump and cfg.gaze.drift_correction
    p, _ = pipeline()
    assert not p.headgaze.calibrated  # gaze does nothing without a calibration


@pytest.mark.parametrize(
    ("data", "msg"),
    [
        ({"hands": {"cursor": {"head_assist": "eyes"}}}, "head_assist"),
        ({"hands": {"cursor": {"autocenter": {"target": "middle"}}}}, "autocenter.target"),
        ({"hands": {"cursor": {"autocenter": {"enabled": "yes"}}}}, "autocenter.enabled"),
        ({"hands": {"cursor": {"autocenter": {"soft_strength": 0.9}}}}, "soft_strength"),
        ({"hands": {"cursor": {"head": {"coarse_cols": 2.5}}}}, "coarse_cols"),
        ({"hands": {"cursor": {"head": {"monitor_switch": 1}}}}, "monitor_switch"),
        ({"gaze": {"model": "tobii"}}, "gaze.model"),
        ({"gaze": {"points_per_monitor": 7}}, "points_per_monitor"),
        ({"gaze": {"jump_idle_s": 0.0}}, "jump_idle_s"),
        ({"gaze": {"look_to_jump": "on"}}, "look_to_jump"),
    ],
)
def test_validation_rejects_bad_values(data, msg):
    with pytest.raises(ValueError, match=msg):
        C.from_dict(data)


def test_config_round_trips_and_tunables_have_labels(tmp_path):
    cfg, unknown = C.from_dict(
        {"hands": {"cursor": {"head_assist": "pointer", "autocenter": {"edge_margin": 0.1}}}, "gaze": {"model": "iris"}}
    )
    assert not unknown and cfg.hands.cursor.head_assist == "pointer" and cfg.hands.cursor.autocenter.edge_margin == 0.1
    p = C.save(cfg, tmp_path / "c.yaml")
    again, unknown2 = C.load(p)
    assert not unknown2 and again.gaze.model == "iris" and again.hands.cursor.autocenter.edge_margin == 0.1
    m = SettingsModel(C.AppConfig(), tmp_path / "s.yaml")
    tun = {t.path: t for t in m.tunables()}
    for path, *_ in C.tracking_tunables(C.AppConfig()):
        assert path in tun and tun[path].label != path, path
        t = tun[path]
        assert t.lo <= t.value <= t.hi
        assert abs(round(t.value / t.resolution) * t.resolution - t.value) < 1e-9, path  # slider keeps defaults
    assert tun["hands.cursor.head.coarse_cols"].is_int
    assert m.set_tunable("hands.cursor.head.coarse_cols", 4.4) is None and m.cfg.hands.cursor.head.coarse_cols == 4
    assert m.set_tunable("gaze.jump_min_px", 400) is None and m.cfg.gaze.jump_min_px == 400


def test_bare_yaml_off_means_off():
    import yaml

    cfg, _ = C.from_dict(yaml.safe_load("hands:\n  cursor:\n    head_assist: off\n"))
    assert cfg.hands.cursor.head_assist == "off"


def test_calibration_camera_is_the_primary_or_single_camera():
    cfg = C.AppConfig()
    cfg.hands.camera_index = "HD User Facing"
    assert calibration_camera(cfg) == "HD User Facing"
    cfg.hands.cameras = [C.CameraSpec(device="Brio", role="secondary"), C.CameraSpec(device="Logi", role="primary")]
    assert calibration_camera(cfg) == "Logi"


# ------------------------------------------------------------------------------------------ head pointer
def test_head_pointer_moves_the_cursor_only_without_a_pointing_hand_and_not_paused():
    p, b = pipeline(head_assist="pointer")
    t = feed_faces(p, 0.0, 0.5, 0.0)
    start = p.mouse.mapper.pos
    t = feed_faces(p, t, 1.0, 10.0)  # turn the head toward screen right
    assert p.mouse.mapper.pos[0] > start[0] + 100
    assert b.calls and b.calls[-1][0] == "move"
    # a pointing hand takes over: the head no longer moves the cursor
    p.gestures.hand_present = True
    pos = p.mouse.mapper.pos
    feed_faces(p, t, 0.5, -10.0)
    assert p.mouse.mapper.pos == pos
    p.gestures.hand_present = False
    p.set_paused(True)
    t = feed_faces(p, t + 1, 0.5, 20.0)
    assert p.mouse.mapper.pos == pos


def test_head_pointer_direction_follows_the_screen_not_the_matrix_sign():
    """A camera whose matrix reports the opposite yaw sign still moves the cursor toward where the nose turns."""
    p, _ = pipeline(head_assist="pointer")
    t = 0.0
    for k in range(60):  # nose turns right while the matrix says "negative yaw"
        t += 0.033
        yaw = 10.0 * math.sin(k / 6)
        xs, ys = face_mesh(yaw, 0)
        o = observe_face(xs, ys, {}, rotation_matrix(-yaw, 0, 0))
        p.on_faces(t, [o.box], [o.blendshapes], [o.frontal], extras=[o])
    assert p.headgaze.aligner.yaw_sign == -1.0
    x0 = p.mouse.mapper.pos[0]
    for k in range(20):
        t += 0.033
        xs, ys = face_mesh(10 + k, 0)
        o = observe_face(xs, ys, {}, rotation_matrix(-(10 + k), 0, 0))
        p.on_faces(t, [o.box], [o.blendshapes], [o.frontal], extras=[o])
    assert p.mouse.mapper.pos[0] > x0 + 50


def test_head_off_does_nothing_and_faces_without_extras_still_work():
    p, b = pipeline()
    feed_faces(p, 0.0, 1.0, 20.0)
    assert b.calls == []
    p.on_faces(2.0, [(0.5, 0.5, 0.3)], [{}], [True])  # old callers (iPhone feed): no extras
    assert b.calls == []


# ------------------------------------------------------------------------------------------ coarse
def test_coarse_mode_jumps_between_zones_after_the_dwell():
    p, b = pipeline(head_assist="coarse")
    t = feed_faces(p, 0.0, 4.0, 0.0)  # the usual pose is learned
    assert p.headgaze.baseline.ready
    t = feed_faces(p, t, 1.5, -25.0)  # a sustained turn far left
    moves = [c for c in b.calls if c[0] == "move"]
    assert moves and TWO[0].contains(*moves[-1][1])
    assert moves[-1][1][0] < 960


# ------------------------------------------------------------------------------------------ calibration-driven
def test_gaze_estimate_monitor_switch_and_look_to_jump_with_a_calibration():
    save_synthetic_calibration("0")
    p, b = pipeline()
    p.cfg.hands.cursor.head.monitor_switch = True
    p.headgaze.reload(force=True)
    assert p.headgaze.calibrated and p.headgaze.switch is not None
    t = 0.0
    for _ in range(30):
        t += 0.033
        look_at(p, t, 2900, 500)  # looking at the right monitor
    g = p.headgaze.estimator.point
    assert g is not None and g[0] > 1920 and abs(g[0] - 2900) < 400
    # the head turned to the right monitor for longer than the dwell: the cursor moved there
    assert p.mouse.mapper.current_monitor() == 1
    assert any(e.topic == "head.monitor_switch" for e in p.bus.history())
    # look-to-jump: back on the left monitor, hand rests, then moves -> the cursor warps to the gaze point
    p.cfg.hands.cursor.head.monitor_switch = False  # (it would move the cursor next to the gaze point itself)
    p.mouse.mapper.set_position(200, 200)
    for _ in range(30):
        t += 0.033
        look_at(p, t, 3300, 300)
    hand = hand_pose("point", center=(0.5, 0.5))
    for _ in range(25):
        t += 0.033
        look_at(p, t, 3300, 300)
        p.on_hand(t, hand)
    t += 0.033
    look_at(p, t, 3300, 300)
    p.on_hand(t, hand_pose("point", center=(0.53, 0.5)))
    jumps = [e for e in p.bus.history() if e.topic == "gaze.jump"]
    assert len(jumps) == 1 and jumps[0].data["x"] > 2500


def test_drift_correction_runs_on_click_actions():
    save_synthetic_calibration("0")
    p, _ = pipeline()
    look_at(p, 0.1, 1000, 500)
    est = p.headgaze.estimator
    assert est is not None and est.raw is not None
    p.mouse.mapper.set_position(est.raw[0] + 80, est.raw[1])
    p.fire("voice", "click")
    assert est.bias[0] == pytest.approx(8.0, abs=0.01)
    p.fire("voice", "scroll up")  # not a click: no correction
    assert est.bias[0] == pytest.approx(8.0, abs=0.01)
    p.headgaze.save_bias()
    assert gz.load_calibration("0").bias[0] == pytest.approx(8.0, abs=0.01)


def test_look_here_voice_command_warps_to_the_gaze_point():
    save_synthetic_calibration("0")
    p, b = pipeline()
    for k in range(10):
        look_at(p, 0.1 + k * 0.033, 3000, 400)
    p.fire("voice", "look here")
    assert b.calls[-1][0] == "move" and b.calls[-1][1][0] > 1920


def test_calibration_for_another_layout_is_not_used_and_reload_picks_up_changes():
    p, _ = pipeline()
    assert not p.headgaze.calibrated
    save_synthetic_calibration("0", [TWO[0]])  # calibrated with one monitor only
    assert p.headgaze.reload(t=10.0)
    assert not p.headgaze.calibrated
    assert any(e.topic == "gaze.layout_changed" for e in p.bus.history())
    save_synthetic_calibration("0")
    p.headgaze._mtime = None  # same-second mtime on some filesystems
    assert p.headgaze.reload(t=20.0) and p.headgaze.calibrated
    assert not p.headgaze.reload(t=20.5)  # throttled


# ------------------------------------------------------------------------------------------ face gate
def test_off_axis_head_still_counts_as_facing_the_screen():
    p, _ = pipeline()
    t = 0.0
    seen = []
    orig = p.gate.observe

    def spy(*a: Any, **k: Any) -> None:
        seen.append(a[4] if len(a) > 4 else k.get("frontal"))
        orig(*a, **k)

    p.gate.observe = spy  # type: ignore[method-assign]
    for _ in range(150):  # camera off to the side: head at yaw 40, the geometric frontal test says no
        t += 0.033
        o = face(40.0, 0.0)
        assert o.frontal is False
        p.on_faces(t, [o.box], [o.blendshapes], [o.frontal], extras=[o])
    assert seen[-1] is True  # the usual pose is "facing the screens"
    o = face(85.0, 0.0)  # turned far away
    p.on_faces(t + 0.033, [o.box], [o.blendshapes], [o.frontal], extras=[o])
    assert seen[-1] is False


def test_headgaze_status_and_hand_hooks_are_safe_without_calibration():
    cfg = C.AppConfig()
    b = RecordingBackend()
    p = Pipeline(cfg, backend=b, screen=TWO)
    hg: HeadGaze = p.headgaze
    assert hg.on_hand_move(0.0, (0.5, 0.5)) is None
    assert hg.on_click((10.0, 10.0)) is False
    assert hg.facing(None) is None
    st = hg.status()
    assert st["calibrated"] is False and st["camera"] == "0"
    p.on_hand(0.0, None)  # hand lost path
    p.dispatcher.on_gesture(GestureEvent("hand_lost", 0.1))
    assert hg.on_face(0.2, None) == {}
    _ = SimpleNamespace()
