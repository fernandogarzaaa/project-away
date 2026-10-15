"""Head tracking + gaze in the running app: one controller fed by the face pipeline and the hand pipeline.

:class:`HeadGaze` owns the per-user state (pose filter, sign alignment, usual pose, gaze calibration, the
look-to-jump trigger) and acts on the cursor through the :class:`~away.control.mouse.MouseController`:

* ``hands.cursor.head_assist: pointer`` - the head moves the cursor while no hand is pointing;
* ``hands.cursor.head_assist: coarse`` - head zones make the cursor jump, the hand refines;
* ``hands.cursor.head.monitor_switch`` - a sustained head turn toward another monitor moves the cursor there
  (needs the screen map: per-monitor head poses recorded by ``away gaze calibrate``);
* gaze (``gaze.*``, needs the calibration for this camera and monitor layout) - look-to-jump and drift
  correction on clicks;
* :meth:`HeadGaze.facing` - head pose for the speech gate (an off-axis camera does not mean "not at the
  screen").

Nothing moves the cursor while output is paused or a pinch/drag is in progress, and (v0.1.26) nothing here moves
it while the hand drives: the :class:`~away.pointer.arbiter.CursorArbiter` gives the cursor to one source at a
time, and every write goes through the mouse controller's output (the glide), never straight to the backend.
Everything is pure logic on top of the injected mouse controller, so it is unit-tested with the recording backend.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from away.control.monitors import nearest_monitor
from away.face import gaze as gz
from away.face.headpose import (
    HeadCoarse,
    HeadPointer,
    HeadPose,
    HeadPoseFilter,
    MonitorSwitch,
    PoseBaseline,
    SignAligner,
    facing_screen,
)


def calibration_camera(cfg: Any) -> str:
    """The camera a gaze calibration belongs to: the primary of ``hands.cameras``, else ``hands.camera_index``."""
    prim = next((c for c in cfg.hands.cameras if c.role == "primary"), None)
    return str(prim.device if prim is not None else cfg.hands.camera_index)


class HeadGaze:
    def __init__(
        self,
        cfg: Any,
        mouse: Any,
        camera: str = "0",
        cal_path: Path | None = None,
        paused: Callable[[], bool] = lambda: False,
        hand_active: Callable[[], bool] = lambda: False,
        busy: Callable[[], bool] = lambda: False,
        publish: Callable[[str, dict[str, Any]], Any] | None = None,
        reload_s: float = 2.0,
        arbiter: Any = None,
    ) -> None:
        self.cfg = cfg
        self.mouse = mouse
        self.camera = camera
        self.cal_path = cal_path
        self.paused, self.hand_active, self.busy = paused, hand_active, busy
        self.publish = publish or (lambda _k, _d: None)
        self.reload_s = reload_s
        if arbiter is None:
            from away.pointer.arbiter import CursorArbiter

            arbiter = CursorArbiter()
        self.arbiter = arbiter
        hd = cfg.hands.cursor.head
        self.filter = HeadPoseFilter(hd.min_cutoff, hd.beta)
        self.aligner = SignAligner()
        self.baseline = PoseBaseline()
        self.pointer = HeadPointer(hd.pointer_gain, hd.pointer_dead_zone, hd.pointer_acceleration)
        self.coarse = HeadCoarse()
        self.switch: MonitorSwitch | None = None
        self.cal: gz.Calibration | None = None
        self.estimator: gz.GazeEstimator | None = None
        self.jump = gz.LookToJump(cfg.gaze.jump_idle_s, cfg.gaze.jump_min_px)
        self.pose: HeadPose | None = None  # filtered, matrix convention (what calibration stores)
        self.aligned: HeadPose | None = None  # filtered, screen-aligned signs
        self.last_t: float | None = None
        self._mtime: float | None = None
        self._checked: float | None = None
        self._layout_key: Any = None
        self.reload(force=True)

    # ---------------------------------------------------------------- calibration
    @property
    def path(self) -> Path:
        return self.cal_path or gz.calibration_file()

    def reload(self, force: bool = False, t: float | None = None) -> bool:
        """(Re)load the calibration when the file changed (``away gaze calibrate`` runs in another process).
        Checks the file at most every ``reload_s`` seconds. True when the calibration in use changed."""
        if not force and t is not None and self._checked is not None and t - self._checked < self.reload_s:
            return False
        self._checked = t
        try:
            mtime: float | None = self.path.stat().st_mtime
        except OSError:
            mtime = None
        layout = tuple((m.left, m.top, m.width, m.height) for m in self.mouse.mapper.monitors)
        if not force and mtime == self._mtime and layout == self._layout_key:
            return False
        self._mtime, self._layout_key = mtime, layout
        cal = gz.load_calibration(self.camera, self.path) if mtime is not None else None
        if cal is not None and not cal.matches_layout(self.mouse.mapper.monitors):
            self.publish("gaze.layout_changed", {"camera": self.camera})
            cal = None  # a different monitor layout: the screen map is wrong, recalibrate
        self.cal = cal
        g = self.cfg.gaze
        self.estimator = gz.GazeEstimator(cal, g.min_cutoff, g.beta) if cal is not None else None
        poses = cal.monitor_poses_for(self.mouse.mapper.monitors) if cal is not None else None
        hd = self.cfg.hands.cursor.head
        self.switch = MonitorSwitch(poses, hd.switch_margin, hd.switch_dwell_s) if poses else None
        self.publish("gaze.calibration", {"camera": self.camera, "calibrated": cal is not None})
        return True

    def save_bias(self) -> None:
        """Persist the drift-correction offset with the calibration (called on shutdown)."""
        if self.cal is not None and self.estimator is not None:
            with contextlib.suppress(OSError):
                gz.save_calibration(self.cal, self.path)
                self._mtime = self.path.stat().st_mtime

    @property
    def calibrated(self) -> bool:
        return self.estimator is not None

    def _sync_cfg(self) -> None:
        hd, g = self.cfg.hands.cursor.head, self.cfg.gaze
        self.filter.configure(hd.min_cutoff, hd.beta)
        self.pointer.gain, self.pointer.dead_zone, self.pointer.acceleration = (
            hd.pointer_gain,
            hd.pointer_dead_zone,
            hd.pointer_acceleration,
        )
        c = self.coarse
        c.cols, c.rows, c.yaw_span, c.pitch_span = (
            hd.coarse_cols,
            hd.coarse_rows,
            hd.coarse_yaw_span,
            hd.coarse_pitch_span,
        )
        c.hysteresis, c.dwell_s = hd.coarse_hysteresis, hd.coarse_dwell_s
        if self.switch is not None:
            self.switch.margin, self.switch.dwell_s = hd.switch_margin, hd.switch_dwell_s
        self.jump.idle_s, self.jump.min_px = g.jump_idle_s, g.jump_min_px
        if self.estimator is not None:
            self.estimator.configure(g.min_cutoff, g.beta)

    # ---------------------------------------------------------------- face frames
    def facing(self, raw_pose: tuple[float, float, float] | None) -> bool | None:
        """Speech gate: is the head turned toward the screens? None = unknown (no pose / nothing learned)."""
        if raw_pose is None:
            return None
        box = self.cal.pose_box if self.cal is not None else None
        tol = float(self.cfg.hands.cursor.head.gate_tolerance)
        return facing_screen(HeadPose(*raw_pose), self.baseline, tol, box)

    def on_face(
        self,
        t: float,
        raw_pose: tuple[float, float, float] | None,
        proxy: tuple[float, float] | None = None,
        feats: np.ndarray | None = None,
        net: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """One frame of the user's face. Returns what happened (for logs/tests)."""
        out: dict[str, Any] = {}
        self.reload(t=t)
        self._sync_cfg()
        if raw_pose is None:
            self.pointer.reset()
            return out
        if self.last_t is not None and t - self.last_t > 0.5:
            self.filter.reset()  # face was lost: do not smooth across the gap
            self.pointer.reset()
        self.last_t = t
        self.pose = self.filter(t, HeadPose(*raw_pose))
        self.baseline.update(t, self.pose)
        self.aligned = self.aligner.update(self.pose, proxy)
        mapper = self.mouse.mapper
        if self.estimator is not None and feats is not None and self.cfg.gaze.enabled:
            f = np.asarray(feats, float)
            if net is not None:
                f = np.concatenate([f[: len(gz.BASE_FEATURES)], np.asarray(net, float)])
            pt = self.estimator.update(t, f, net_available=net is not None)
            if pt is not None:
                out["gaze"] = pt
        if self.paused() or not self.mouse.enabled:
            self.pointer.reset()
            return out
        mode = self.cfg.hands.cursor.head_assist
        busy = self.busy()
        # v0.1.26: one cursor source at a time. While the hand drives (and for a moment after its last frame, so a
        # detection dropout does not hand the cursor over) the head only observes.
        hand = bool(self.hand_active()) or self.arbiter.hand_owns(t)
        free = not busy and not hand
        if mode == "pointer" and free:
            dx, dy = self.pointer.update(self.aligned, mapper.ref.width)
            if (dx or dy) and self.arbiter.claim(t, "head"):
                mapper.set_position(mapper.pos[0] + dx, mapper.pos[1] + dy)
                x, y = round(mapper.pos[0]), round(mapper.pos[1])
                self.mouse.follow(x, y)
                out["pointer"] = (x, y)
        else:
            self.pointer.reset()
        if mode == "coarse" and free and self.baseline.ready and self.baseline.yaw is not None:
            a = self.aligner
            off = (
                (self.pose.yaw - self.baseline.yaw) * a.yaw_sign,
                (self.pose.pitch - self.baseline.pitch) * a.pitch_sign,
            )
            target = self.coarse.update(t, off, mapper.monitors)
            if target is not None and self.arbiter.claim(t, "head"):
                out["coarse"] = self._warp(target)
        if self.cfg.hands.cursor.head.monitor_switch and self.switch is not None and free:
            j = self.switch.update(t, self.pose, mapper.current_monitor())
            if j is not None and self.arbiter.claim(t, "head"):
                m = mapper.monitors[j]
                g = self.estimator.fresh(t) if self.estimator is not None else None
                dest = g if g is not None and m.contains(*g) else m.center
                out["switch"] = j
                out["switch_to"] = self._warp(dest)
                self.publish("head.monitor_switch", {"monitor": j})
        return out

    def _warp(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = self.mouse.mapper.warp(point[0], point[1])
        if self.mouse.stabilizer is not None:
            self.mouse.stabilizer.reset()
        self.mouse.jump_to(x, y)  # through the output glide, so it does not pull the cursor back
        return x, y

    # ---------------------------------------------------------------- hand / clicks
    def on_hand_move(self, t: float, hand: tuple[float, float]) -> tuple[int, int] | None:
        """A pointing frame (before the cursor moves). Look-to-jump: returns the warp target when it fired."""
        g = self.cfg.gaze
        if self.estimator is None or not (g.enabled and g.look_to_jump):
            self.jump.reset()
            return None
        if self.paused() or self.busy() or not self.mouse.enabled:
            self.jump.reset()
            return None
        mapper = self.mouse.mapper
        mons = mapper.monitors
        target = self.jump.update(
            t,
            hand,
            mapper.pos,
            self.estimator.fresh(t),
            monitor_of=lambda x, y: nearest_monitor(x, y, mons),
            across_only=bool(g.jump_across_monitors_only),
        )
        if target is None:
            return None
        self.arbiter.claim(t, "gaze")  # the hand's movement onset: gaze picks the start, the hand drives on
        pos = self._warp(target)
        self.publish("gaze.jump", {"x": pos[0], "y": pos[1]})
        return pos

    def on_hand_lost(self) -> None:
        self.jump.reset()

    def on_click(self, pos: tuple[float, float]) -> bool:
        """Drift correction: the user looks where they click."""
        g = self.cfg.gaze
        if self.estimator is None or not (g.enabled and g.drift_correction):
            return False
        return self.estimator.correct(pos, g.drift_rate, g.drift_max_px, g.drift_max_err_px)

    def status(self) -> dict[str, Any]:
        acc = self.cal.accuracy_px() if self.cal is not None else None
        return {
            "camera": self.camera,
            "calibrated": self.calibrated,
            "accuracy_px": None if acc is None else round(acc),
            "pose": None if self.pose is None else (round(self.pose.yaw, 1), round(self.pose.pitch, 1)),
            "gaze": None if self.estimator is None or self.estimator.point is None else self.estimator.point,
            "monitor_switch": self.switch is not None,
        }


class NetRunner:
    """Runs the optional learned gaze network only when it can help: the current calibration was fitted with
    that network's angles (``Calibration.net_model``) and the network resolved from ``gaze.model`` is the same
    one. Re-resolves every ``check_s`` seconds, so a model downloaded from Settings (and a recalibration) is
    picked up without restarting the app."""

    def __init__(
        self,
        cfg: Any,
        models_dir: Path,
        factory: Callable[[Path, str, float], Any] | None = None,
        resolve: Callable[[str, Path], str] | None = None,
        check_s: float = 5.0,
    ) -> None:
        self.cfg = cfg
        self.models_dir = models_dir
        self.factory = factory
        self.resolve = resolve
        self.check_s = check_s
        self.net: Any = None
        self.name = "iris"
        self.error: str | None = None
        self._checked: float | None = None

    def _resolve(self) -> str:
        if self.resolve is not None:
            return self.resolve(self.cfg.gaze.model, self.models_dir)
        from away.app.optional_models import resolve_gaze_model

        return resolve_gaze_model(self.cfg.gaze.model, self.models_dir)

    def _make(self, name: str) -> Any:
        if self.factory is not None:
            return self.factory(self.models_dir, name, float(self.cfg.gaze.model_rate_hz))
        from away.app.optional_models import path_of  # pragma: no cover - real onnxruntime
        from away.face.gaze_net import GazeNet  # pragma: no cover

        return GazeNet(path_of(name, self.models_dir), name, float(self.cfg.gaze.model_rate_hz))  # pragma: no cover

    def wanted(self, t: float, cal: gz.Calibration | None) -> str | None:
        """The network to run now (None = none)."""
        if cal is None or cal.net_model is None or not self.cfg.gaze.enabled:
            return None
        if self._checked is None or t - self._checked >= self.check_s:
            self._checked = t
            self.name = self._resolve()
        return self.name if self.name == cal.net_model else None

    def update(
        self, t: float, rgb: Any, box: tuple[float, float, float] | None, cal: gz.Calibration | None
    ) -> tuple[float, float] | None:
        name = self.wanted(t, cal)
        if name is None:
            return None
        if self.net is None or getattr(self.net, "name", None) != name:
            try:
                self.net = self._make(name)
                self.error = None
            except Exception as exc:  # a broken download / onnxruntime problem must not stop the camera
                self.net, self.error = None, f"{type(exc).__name__}: {exc}"
                self._checked = t + 55.0  # retry in about a minute
                return None
        self.net.max_hz = float(self.cfg.gaze.model_rate_hz)
        return self.net.update(t, rgb, box)  # type: ignore[no-any-return]
