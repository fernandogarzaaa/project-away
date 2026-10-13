"""Settings > Tuning live view and auto-tune recording: one camera worker, no Tk.

The worker borrows the camera from the running app exactly like the "Test gesture" tab
(:class:`~away.app.camera_share.CameraLease`, given back when it stops), runs the hand tracker in-process and
posts plain dicts on ``out``:

* ``{"type": "status", "text": ...}``: lease / camera messages;
* ``{"type": "frame", "t", "lm", "helper", "gauges"}``: every ``every``-th frame, the pointer and helper landmarks
  and :func:`~away.hands.autotune.live_gauges` against the *current* settings (a slider move shows at once);
* ``{"type": "countdown", "tag", "left"}`` / ``{"type": "recording", "tag"}`` / ``{"type": "trial", "tag",
  "frames", "handedness", "fps"}``: answers to a ``{"cmd": "record", "tag", "seconds", "countdown"}`` sent on
  ``commands`` (``{"cmd": "cancel"}`` drops a recording in progress). Recording runs here, on the camera thread,
  so no frame is lost while the window is busy.

Timing uses the frame timestamps, so tests drive it with fake frames.
"""

from __future__ import annotations

import contextlib
import math
import queue
from pathlib import Path
from typing import Any

from away.hands.autotune import live_gauges


def handle_command(rec: dict[str, Any] | None, cmd: Any, t: float) -> dict[str, Any] | None:
    """The recording state after command ``cmd`` at time ``t`` (``None`` = not recording)."""
    if not isinstance(cmd, dict):
        return rec
    if cmd.get("cmd") == "cancel":
        return None
    if cmd.get("cmd") == "record":
        seconds = max(0.1, float(cmd.get("seconds", 2.0)))
        start = t + max(0.0, float(cmd.get("countdown", 1.5)))
        return {
            "tag": cmd.get("tag"),
            "start": start,
            "end": start + seconds,
            "seconds": seconds,
            "frames": [],
            "left": None,
            "announced": False,
            "hand": "Right",
        }
    return rec


def step_recording(rec: dict[str, Any], t: float, lm: Any, helper: Any, hand: str) -> tuple[list[dict[str, Any]], bool]:
    """Advance a recording by one frame: ``(messages, finished)``."""
    msgs: list[dict[str, Any]] = []
    if t < rec["start"]:
        left = math.ceil(rec["start"] - t)
        if left != rec["left"]:
            rec["left"] = left
            msgs.append({"type": "countdown", "tag": rec["tag"], "left": left})
        return msgs, False
    if not rec["announced"]:
        rec["announced"] = True
        msgs.append({"type": "recording", "tag": rec["tag"]})
    if t <= rec["end"]:
        rec["frames"].append((lm, helper))
        if lm is not None:
            rec["hand"] = hand or rec["hand"]
    if t >= rec["end"]:
        n = sum(1 for f, _h in rec["frames"] if f is not None)
        msgs.append(
            {
                "type": "trial",
                "tag": rec["tag"],
                "frames": rec["frames"],
                "handedness": rec["hand"],
                "fps": n / rec["seconds"] if rec["seconds"] > 0 else 0.0,
            }
        )
        return msgs, True
    return msgs, False


def tuning_camera_worker(
    model: Any,
    out: queue.Queue[dict[str, Any]],
    stop: Any,
    commands: queue.Queue[Any],
    *,
    share: Any = None,
    frames: Any = None,
    tracker_factory: Any = None,
    lease_timeout: float = 4.0,
    every: int = 2,
) -> None:
    """Camera -> landmarks -> live gauges (+ auto-tune trials). Nothing is executed."""
    from away import paths
    from away.app.camera_share import CameraLease, CameraShare, camera_error_message

    cfg = model.cfg
    idx = cfg.hands.camera_index
    if frames is None:  # pragma: no cover - real camera
        from away.hands.tracker import camera_frames as frames
    if tracker_factory is None:  # pragma: no cover - real MediaPipe

        def tracker_factory() -> Any:
            from away.hands.tracker import HandTracker

            mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
            return HandTracker(
                mdir / "hand_landmarker.task",
                max(2, cfg.hands.max_hands),  # the palm-depth gauge needs the helper hand too
                prefer=cfg.hands.roles.mode,
                mirrored=cfg.hands.mirror,
                release_s=cfg.hands.roles.release_s,
            )

    lease = CameraLease(share or CameraShare(), timeout=lease_timeout, who="settings-tuning")
    try:
        lease.open()
        if lease.handed_over:
            out.put({"type": "status", "text": "Project Away paused hand tracking for the live view"})
        elif lease.app_running:
            out.put({"type": "status", "text": "waiting for Project Away to release the camera..."})
        try:
            tracker = tracker_factory()
        except Exception as exc:
            out.put({"type": "status", "text": f"hand tracker unavailable: {exc}"})
            return
        rec: dict[str, Any] | None = None
        n = 0
        try:
            out.put({"type": "status", "text": "live view running"})
            for t, rgb in frames(idx, cfg.hands.mirror):
                if stop.is_set():
                    break
                while True:
                    try:
                        cmd = commands.get_nowait()
                    except queue.Empty:
                        break
                    rec = handle_command(rec, cmd, t)
                fr = tracker.process(rgb, t)
                lm = getattr(fr, "landmarks", None)
                helper = getattr(fr, "helper", None)
                if rec is not None:
                    msgs, done = step_recording(rec, t, lm, helper, str(getattr(fr, "handedness", "Right")))
                    for msg in msgs:
                        out.put(msg)
                    if done:
                        rec = None
                if n % max(1, every) == 0:
                    try:
                        gauges = live_gauges(lm, helper, model.cfg)
                    except ValueError:
                        gauges = []
                    out.put({"type": "frame", "t": t, "lm": lm, "helper": helper, "gauges": gauges})
                n += 1
        except Exception as exc:
            out.put({"type": "status", "text": camera_error_message(exc, idx, lease.app_running, lease.handed_over)})
        finally:
            with contextlib.suppress(Exception):
                tracker.close()
    finally:
        lease.close()
        out.put({"type": "stopped"})
