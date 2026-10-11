"""Camera enumeration, threaded capture and timestamp pairing.

:class:`FramePairer` is pure logic (thread-safe, no devices): every camera pushes
``(timestamp, frame)``; :meth:`FramePairer.pop_set` returns one frame per camera
taken at (nearly) the same moment, or ``None`` for a camera whose closest frame is
further than ``tolerance_s`` away or that has gone silent for ``dropout_s``.
The primary camera sets the pace; when it drops out, the freshest live camera does.
"""

from __future__ import annotations

import contextlib
import glob
import math
import os
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away.config import CameraSpec


@dataclass
class CameraDevice:
    index: int | None  # OpenCV index (None when only a path is known)
    path: str  # /dev/video0 on Linux, "camera 0" elsewhere
    name: str
    ok: bool | None = None  # opened and returned a frame (only when probed)


def list_cameras(
    sysfs: str = "/sys/class/video4linux", dev: str = "/dev", probe: bool = False, max_index: int = 8
) -> list[CameraDevice]:
    """Enumerate cameras. Linux: V4L2 capture nodes with their names (``probe`` also opens each one).
    Windows/macOS have no cheap device list, so OpenCV indices ``0..max_index-1`` are opened and the
    ones that return a frame are listed."""
    if not os.path.isdir(sysfs):  # pragma: no cover - Windows/macOS
        found = [_probe(CameraDevice(i, f"camera {i}", f"camera {i}")) for i in range(max_index)]
        return [c for c in found if c.ok]
    out: list[CameraDevice] = []
    for node in sorted(glob.glob(os.path.join(sysfs, "video*")), key=_node_num):
        base = os.path.basename(node)
        idx_file = Path(node) / "index"
        # each UVC camera exposes a capture node (index 0) plus metadata nodes; keep the capture node
        if idx_file.exists() and idx_file.read_text().strip() not in ("", "0"):
            continue
        name_file = Path(node) / "name"
        name = name_file.read_text().strip() if name_file.exists() else base
        out.append(CameraDevice(_node_num(node), dev.rstrip("/") + "/" + base, name))  # V4L paths are POSIX
    if probe:  # pragma: no cover - opens devices
        out = [_probe(c) for c in out]
    return out


def _node_num(path: str) -> int:
    m = re.search(r"(\d+)$", path)
    return int(m.group(1)) if m else 0


def _probe(c: CameraDevice) -> CameraDevice:  # pragma: no cover - opens a device
    os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")  # missing indices are expected while probing
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_FFMPEG", "0")
    try:
        import cv2
    except ImportError:
        return c
    with contextlib.suppress(Exception):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    cap = cv2.VideoCapture(c.index if c.index is not None else c.path)
    try:
        ok = bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()
    return CameraDevice(c.index, c.path, c.name, ok)


def resolve_device(device: int | str, devices: list[CameraDevice] | None = None) -> int | str:
    """Config ``device`` -> what ``cv2.VideoCapture`` takes: an index, a path, or the index of a camera by name."""
    if isinstance(device, int):
        return device
    d = device.strip()
    if d.isdigit():
        return int(d)
    if d.startswith(("/dev/", "\\\\", "http://", "https://", "rtsp://")) or os.path.exists(d):
        return d
    devs = list_cameras() if devices is None else devices
    for c in devs:
        if d.lower() in c.name.lower():
            return c.index if c.index is not None else c.path
    raise ValueError(f"no camera named {device!r}; run `away cameras list`")


class FramePairer:
    def __init__(
        self,
        n_cams: int,
        primary: int = 0,
        tolerance_s: float = 0.04,
        dropout_s: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
        maxlen: int = 6,
    ) -> None:
        if n_cams < 1:
            raise ValueError("n_cams must be at least 1")
        if not 0 <= primary < n_cams:
            raise ValueError("primary must identify a camera in range")
        if not math.isfinite(tolerance_s) or tolerance_s < 0:
            raise ValueError("tolerance_s must be finite and non-negative")
        if not math.isfinite(dropout_s) or dropout_s <= 0:
            raise ValueError("dropout_s must be finite and positive")
        if maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        self.n = n_cams
        self.primary = primary
        self.tolerance_s = tolerance_s
        self.dropout_s = dropout_s
        self.clock = clock
        self._buf: list[deque[tuple[float, Any]]] = [deque(maxlen=maxlen) for _ in range(n_cams)]
        self._last_seen = [-1e9] * n_cams
        self._last_ref = -1e9
        self._lock = threading.Lock()

    def push(self, cam: int, t: float, frame: Any) -> None:
        """Add a frame, rejecting invalid camera IDs and unordered timestamps."""
        if not 0 <= cam < self.n:
            raise ValueError(f"camera index {cam} is outside [0, {self.n})")
        if not math.isfinite(t):
            raise ValueError("frame timestamp must be finite")
        with self._lock:
            if self._buf[cam] and t < self._buf[cam][-1][0]:
                raise ValueError("frame timestamps must be non-decreasing per camera")
            self._buf[cam].append((t, frame))
            self._last_seen[cam] = max(self._last_seen[cam], t)

    def alive(self, now: float | None = None) -> list[bool]:
        now = self.clock() if now is None else now
        return [now - s <= self.dropout_s for s in self._last_seen]

    def pop_set(self, now: float | None = None) -> tuple[float, list[tuple[float, Any] | None]] | None:
        """Next synchronised set ``(t_ref, [ (t, frame) | None per camera ])``, or None if nothing new."""
        now = self.clock() if now is None else now
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        with self._lock:
            alive = self.alive(now)
            ref: int | None = None
            if alive[self.primary]:  # the primary sets the pace while it is alive
                if not self._buf[self.primary]:
                    return None
                ref = self.primary
            else:
                cands = [i for i in range(self.n) if alive[i] and self._buf[i]]
                if not cands:
                    return None
                ref = max(cands, key=lambda i: self._buf[i][-1][0])
            t_ref, f_ref = self._buf[ref][-1]
            if t_ref <= self._last_ref:
                return None
            out: list[tuple[float, Any] | None] = [None] * self.n
            out[ref] = (t_ref, f_ref)
            for i in range(self.n):
                if i == ref or not alive[i] or not self._buf[i]:
                    continue
                t_i, f_i = min(self._buf[i], key=lambda tf: abs(tf[0] - t_ref))
                if abs(t_i - t_ref) <= self.tolerance_s:
                    out[i] = (t_i, f_i)
            for i, item in enumerate(out):  # drop everything up to what was used
                if item is not None:
                    while self._buf[i] and self._buf[i][0][0] <= item[0]:
                        self._buf[i].popleft()
            self._last_ref = t_ref
            return t_ref, out


def open_capture(spec: CameraSpec) -> Any:  # pragma: no cover - opens a device
    import cv2

    src = resolve_device(spec.device)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if sys.platform.startswith("win") and isinstance(src, int) else None
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
    cap.set(cv2.CAP_PROP_FPS, spec.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low; not every backend honours it
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {spec.device!r} (no device, busy, or permission denied)")
    return cap


class CameraThread(threading.Thread):  # pragma: no cover - device thread
    """Reads one camera as fast as it delivers, stamping each frame with ``time.monotonic()`` on arrival."""

    def __init__(self, cam: int, spec: CameraSpec, pairer: FramePairer, stop: threading.Event) -> None:
        super().__init__(daemon=True, name=f"camera-{cam}")
        self.cam, self.spec, self.pairer, self.stop = cam, spec, pairer, stop
        self.error: str | None = None
        self.size: tuple[int, int] = (spec.width, spec.height)

    def run(self) -> None:
        import cv2

        try:
            cap = open_capture(self.spec)
        except Exception as exc:
            self.error = str(exc)
            return
        try:
            failures = 0
            while not self.stop.is_set():
                ok, frame = cap.read()
                t = time.monotonic()
                if not ok:
                    failures += 1
                    if failures > 30:  # unplugged: stop; the pairer marks this camera dropped
                        self.error = "camera stopped delivering frames"
                        return
                    time.sleep(0.02)
                    continue
                failures = 0
                if self.spec.mirror:
                    frame = cv2.flip(frame, 1)
                self.size = (frame.shape[1], frame.shape[0])
                self.pairer.push(self.cam, t, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
