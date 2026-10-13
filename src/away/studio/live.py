"""Gesture Studio camera worker: one thread, no Tk. Borrows the camera from the running app exactly like the
Tuning tab's live view (:class:`~away.app.camera_share.CameraLease`, handed back when it stops).

Posts plain dicts on ``out``:

* ``{"type": "status", "text"}`` / ``{"type": "stopped"}``;
* ``{"type": "frame", ...}`` every ``every``-th frame: ``preview`` (``(w, h, PPM bytes)`` of the mirrored
  picture, or None), ``lm`` / ``helper``, ``point`` (the hand's zone point), ``zone_ok``, ``probs`` (class ->
  probability, ``nothing`` first), ``top`` / ``top_p`` / ``unknown``, ``states`` (gesture -> idle|arming|
  recognized|firing|cooldown), ``saliency`` (per landmark, at most ``saliency_hz`` times a second),
  ``nearest`` and ``builtin`` (what the built-in gesture detector saw: the old "Test gesture" tab);
* ``{"type": "fired", "name", "action", "dry_run"}``: a gesture fired (with dry run on nothing is executed);
* recording: ``countdown`` -> ``recording`` -> ``take`` (``frames``, ``handedness``, ``fps``, ``label``,
  ``session``, ``tag``); ``negative`` after "That was wrong" (the last 2 s as ``frames``).

Commands on ``commands``: ``record`` {label, seconds, countdown, session, tag}, ``cancel``, ``wrong``,
``dry_run`` {on}, ``reload`` (the trained model changed), ``zones`` {map} (live zone edits), ``source``
{which: mine|builtin}.
"""

from __future__ import annotations

import contextlib
import queue
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from away.app.tuning_live import handle_command, step_recording
from away.studio.engine import ZONE_GATED_EVENTS, GestureArbiter, StudioRecognizer, top_candidate
from away.studio.pack import NOTHING, GestureSettings, StudioStore, gesture_settings
from away.studio.zones import ZoneGate, anchor_point, zone_map_from_dict

WRONG_S = 2.0  # "That was wrong" keeps this many seconds as a 'nothing' example
PREVIEW_W, PREVIEW_H = 400, 300  # the Studio picture (normalised landmarks map straight onto it)


def ppm_preview(rgb: Any, width: int = PREVIEW_W, height: int = PREVIEW_H) -> tuple[int, int, bytes] | None:
    """Resize an RGB frame (H, W, 3 uint8) to exactly ``width`` x ``height`` (nearest pixel) and encode it as
    binary PPM (Tk shows PPM without Pillow). Returns ``(w, h, data)`` or None for anything that is not RGB."""
    if rgb is None:
        return None
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3 or a.shape[0] < 2 or a.shape[1] < 2:
        return None
    ys = np.linspace(0, a.shape[0] - 1, height).astype(int)
    xs = np.linspace(0, a.shape[1] - 1, width).astype(int)
    small = np.ascontiguousarray(a[ys][:, xs, :3]).astype(np.uint8)
    return width, height, f"P6 {width} {height} 255\n".encode() + small.tobytes()


class BuiltinSource:
    """The bundled temporal model as a probability source ('idle' shown as nothing)."""

    def __init__(self, path: Path) -> None:
        from away.hands.runtime import TemporalGestureClassifier

        self.clf = TemporalGestureClassifier(path)
        self.classes = [NOTHING if c == "idle" else c for c in self.clf.classes]

    def push(self, t: float, lm: Any, hand: str, helper: Any) -> dict[str, float] | None:
        if lm is None:
            self.clf.reset()
            return None
        p = self.clf.push(lm, hand)
        if p is None:
            return None
        return {(NOTHING if k == "idle" else k): v for k, v in p.items()}


def source_settings(source: Any, cfg: Any, store: StudioStore) -> dict[str, GestureSettings]:
    """Per-gesture settings for whatever the Studio is watching (built-in: the app's model threshold)."""
    if not isinstance(source, StudioRecognizer):
        thr = float(cfg.hands.gestures.model_min_conf)
        classes = [c for c in getattr(source, "classes", []) if c != NOTHING]
        return {c: GestureSettings(threshold=thr, hold_s=0.0) for c in classes}
    kinds = {g.name: g.kind for g in store.gestures()}
    out: dict[str, GestureSettings] = {}
    for c in source.classes[1:]:
        try:
            out[c] = gesture_settings(cfg, c, kinds.get(c, "static"))
        except ValueError:
            out[c] = GestureSettings(enabled=False)
    return out


def studio_camera_worker(
    model: Any,
    out: queue.Queue[dict[str, Any]],
    stop: Any,
    commands: queue.Queue[Any],
    *,
    store: StudioStore | None = None,
    share: Any = None,
    frames: Any = None,
    tracker_factory: Any = None,
    lease_timeout: float = 4.0,
    every: int = 2,
    saliency_hz: float = 3.0,
    executor: Any = None,
    zone_map: dict[str, Any] | None = None,
) -> None:
    """Camera -> landmarks -> zones + model probabilities + states. With dry run (the default) nothing runs;
    otherwise fired gestures go to ``executor(name)`` (the Settings window passes a real action runner)."""
    from away import paths
    from away.app.camera_share import LEASE_IDLE_S, CameraLease, CameraShare, camera_error_message
    from away.hands.gestures import GestureStateMachine

    cfg = model.cfg
    store = store or StudioStore()
    idx = cfg.hands.camera_index
    if frames is None:  # pragma: no cover - real camera
        from away.hands.tracker import camera_frames as frames
    if tracker_factory is None:  # pragma: no cover - real MediaPipe

        def tracker_factory() -> Any:
            from away.hands.tracker import HandTracker

            mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
            return HandTracker(
                mdir / "hand_landmarker.task",
                max(2, cfg.hands.max_hands),  # two-hand gestures need the second hand
                prefer=cfg.hands.roles.mode,
                mirrored=cfg.hands.mirror,
                release_s=cfg.hands.roles.release_s,
            )

    def load_source(which: str) -> tuple[Any, str]:
        if which == "builtin":
            from away.hands.runtime import find_model

            p = find_model("gesture_tcn.onnx", Path(cfg.models_dir) if cfg.models_dir else paths.models_dir())
            if p is None:
                return None, "The built-in gesture model is missing. Run setup again to restore it."
            try:
                return BuiltinSource(p), ""
            except Exception as exc:
                return None, f"The built-in gestures could not start ({exc})."
        rec = StudioRecognizer.load(store.model_dir)
        return rec, "" if rec is not None else "No gestures taught yet. Press 'Teach a new gesture' to start."

    lease = CameraLease(
        share or CameraShare(), timeout=lease_timeout, who="settings-studio", idle_s=LEASE_IDLE_S, stop=stop
    )
    try:
        lease.open()
        if lease.handed_over:
            out.put({"type": "status", "text": "Project Away paused itself while the Studio uses the camera"})
        elif lease.app_running:
            out.put({"type": "status", "text": "Waiting for Project Away to let go of the camera..."})
        try:
            tracker = tracker_factory()
        except Exception as exc:
            out.put({"type": "status", "text": f"Hand tracking could not start ({exc})."})
            return
        which = "mine"
        source, note = load_source(which)
        if note:
            out.put({"type": "status", "text": note})
        settings = source_settings(source, cfg, store)
        arbiter = GestureArbiter()
        sm = GestureStateMachine(cfg.hands.gestures)
        z = cfg.zones
        try:
            gate = ZoneGate(zone_map_from_dict(zone_map) if zone_map else None, z.hysteresis)
        except ValueError:
            gate = ZoneGate(None, z.hysteresis)
        dry_run = True
        rec: dict[str, Any] | None = None
        history: deque[tuple[float, Any, Any]] = deque()
        last_sal = -1e9
        last_builtin = ""
        n = 0
        try:
            out.put({"type": "status", "text": "Camera on"})
            for t, rgb in frames(idx, cfg.hands.mirror):
                if stop.is_set():
                    break
                lease.touch()  # v0.1.21: frames are flowing (a stuck preview gives the camera back)
                while True:
                    try:
                        cmd = commands.get_nowait()
                    except queue.Empty:
                        break
                    if not isinstance(cmd, dict):
                        continue
                    c = cmd.get("cmd")
                    if c in ("record", "cancel"):
                        rec = handle_command(rec, cmd, t)
                        if rec is not None:
                            rec["label"], rec["session"] = cmd.get("label"), cmd.get("session")
                    elif c == "dry_run":
                        dry_run = bool(cmd.get("on", True))
                    elif c in ("reload", "source"):
                        if c == "source":
                            which = "builtin" if cmd.get("which") == "builtin" else "mine"
                        source, note = load_source(which)
                        settings = source_settings(source, cfg, store)
                        arbiter.reset()
                        if note or c == "reload":
                            out.put({"type": "status", "text": note or "Using your newly taught gestures"})
                    elif c == "zones":
                        m = cmd.get("map")
                        try:
                            gate = ZoneGate(zone_map_from_dict(m) if m else None, z.hysteresis)
                        except ValueError as exc:
                            out.put({"type": "status", "text": f"Those areas could not be used: {exc}"})
                    elif c == "wrong":
                        recent = [(lm_, hp_) for ts, lm_, hp_ in history if ts >= t - WRONG_S]
                        out.put({"type": "negative", "frames": recent, "handedness": "Right"})
                fr = tracker.process(rgb, t)
                lm = getattr(fr, "landmarks", None)
                helper = getattr(fr, "helper", None)
                hand = str(getattr(fr, "handedness", "Right"))
                history.append((t, lm, helper))
                while history and history[0][0] < t - WRONG_S - 0.5:
                    history.popleft()
                if rec is not None:
                    label, session = rec.get("label"), rec.get("session")
                    msgs, done = step_recording(rec, t, lm, helper, hand)
                    for msg in msgs:
                        if msg["type"] == "trial":
                            msg = {**msg, "type": "take", "label": label, "session": session}
                        out.put(msg)
                    if done:
                        rec = None
                point = None
                if lm is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        point = anchor_point(lm, z.anchor, True)  # the frames are already mirrored for display
                zone_ok = gate.update(point)
                probs = None
                if source is not None:
                    try:
                        probs = source.push(t, lm, hand, helper)
                    except Exception as exc:
                        out.put({"type": "status", "text": f"The gesture recogniser stopped ({exc})"})
                        source = None
                for name in arbiter.update(t, probs, settings, allowed=zone_ok):
                    action = str(cfg.bindings.resolve("gestures", name, model.profile))
                    if not dry_run and executor is not None:
                        with contextlib.suppress(Exception):
                            executor(name)
                    out.put({"type": "fired", "t": t, "name": name, "action": action, "dry_run": dry_run})
                builtin = ""
                try:
                    events = sm.update(t, lm)
                except (ValueError, TypeError):
                    events = []
                for ev in events:
                    if ev.kind in ("move", "autoscroll"):
                        continue
                    txt = model.describe_event(ev)
                    if txt:
                        builtin = txt if zone_ok or ev.kind not in ZONE_GATED_EVENTS else f"{txt} (ignored here)"
                if builtin and builtin != last_builtin:
                    last_builtin = builtin
                else:
                    builtin = ""
                if n % max(1, every) == 0:
                    lowest = min([s.threshold for s in settings.values()] or [0.8])
                    top, top_p, unknown = top_candidate(probs, lowest)
                    sal = near = None
                    if isinstance(source, StudioRecognizer) and probs and not unknown and top != NOTHING:
                        near = source.nearest(top)
                        if t - last_sal >= 1.0 / max(0.1, saliency_hz):
                            last_sal = t
                            with contextlib.suppress(Exception):
                                sal = source.saliency(top)
                    out.put(
                        {
                            "type": "frame",
                            "t": t,
                            "preview": ppm_preview(rgb),
                            "lm": lm,
                            "helper": helper,
                            "point": point,
                            "zone_ok": zone_ok,
                            "probs": probs,
                            "top": top,
                            "top_p": top_p,
                            "unknown": unknown,
                            "states": dict(arbiter.state),
                            "saliency": sal,
                            "nearest": near,
                            "builtin": builtin,
                            "dry_run": dry_run,
                        }
                    )
                n += 1
        except Exception as exc:
            out.put({"type": "status", "text": camera_error_message(exc, idx, lease.app_running, lease.handed_over)})
        finally:
            with contextlib.suppress(Exception):
                tracker.close()
    finally:
        lease.close()
        out.put({"type": "stopped"})
