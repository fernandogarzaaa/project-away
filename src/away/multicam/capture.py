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
    in_use: bool = False  # held by the running Project Away app (not probed, so it is not disturbed)

    @property
    def key(self) -> str:
        return str(self.index if self.index is not None else self.path)


def list_cameras(
    sysfs: str = "/sys/class/video4linux",
    dev: str = "/dev",
    probe: bool = False,
    max_index: int = 8,
    in_use: set[str] | None = None,
    platform: str | None = None,
    run: Callable[..., Any] | None = None,
    probe_fn: Callable[[CameraDevice], CameraDevice] | None = None,
) -> list[CameraDevice]:
    """Enumerate cameras without opening them where the OS allows it.

    * Linux: V4L2 capture nodes with their names (``probe`` also opens each one).
    * Windows: camera/image-class PnP devices by name (PowerShell CIM query, 15 s timeout), numbered in
      that order as OpenCV indices (best effort; DirectShow usually enumerates in the same order).
    * macOS: ``system_profiler SPCameraDataType``.
    * Otherwise (or when the query finds nothing): OpenCV indices ``0..max_index-1`` are opened.

    ``in_use``: device keys (``"0"``, ``"/dev/video0"``) the running app holds (camera_share owner file);
    they are listed and marked instead of probed, since an exclusive camera would fail to open and vanish
    from the list (and are added even when the OS query missed them). ``None`` reads the owner file.
    """
    plat = platform or sys.platform
    if in_use is None:
        in_use = _app_cameras()
    if os.path.isdir(sysfs):
        out: list[CameraDevice] = []
        for node in sorted(glob.glob(os.path.join(sysfs, "video*")), key=_node_num):
            base = os.path.basename(node)
            idx_file = Path(node) / "index"
            # each UVC camera exposes a capture node (index 0) plus metadata nodes; keep the capture node
            if idx_file.exists() and idx_file.read_text().strip() not in ("", "0"):
                continue
            name_file = Path(node) / "name"
            name = name_file.read_text().strip() if name_file.exists() else base
            num = _node_num(node)
            path = dev.rstrip("/") + "/" + base  # V4L paths are POSIX
            out.append(CameraDevice(num, path, name, in_use=str(num) in in_use or path in in_use))
        if probe:  # pragma: no cover - opens devices
            out = [c if c.in_use else _probe(c) for c in out]
        return out
    names: list[str] = []
    if plat == "win32":
        names = windows_camera_names(run)
    elif plat == "darwin":
        names = macos_camera_names(run)
    if names:
        devs = [CameraDevice(i, f"camera {i}", n, in_use=str(i) in in_use) for i, n in enumerate(names)]
        if probe:  # pragma: no cover - opens devices
            devs = [c if c.in_use else _probe(c) for c in devs]
        return with_in_use(devs, in_use)
    # no name query worked: open each free index (the ones the app holds are listed, not opened)
    prober = probe_fn or _probe
    found: list[CameraDevice] = []
    misses = 0
    for i in range(max_index):
        c = CameraDevice(i, f"camera {i}", f"camera {i}", in_use=str(i) in in_use)
        if c.in_use:
            found.append(c)
            misses = 0
            continue
        r = prober(c)
        if r.ok:
            found.append(r)
            misses = 0
        else:
            misses += 1
            if misses >= 2 and i >= 2:  # indices are dense: two misses in a row means no more cameras
                break
    return with_in_use(found, in_use)


def with_in_use(devs: list[CameraDevice], in_use: set[str]) -> list[CameraDevice]:
    """``devs`` plus a row for every numeric device the running app holds that the query missed (a camera
    held exclusively can drop out of an OS listing or fail a probe, and must never vanish from the list)."""
    have = {d.key for d in devs} | {d.path for d in devs}
    extra = [
        CameraDevice(int(k), f"camera {k}", f"camera {k}", in_use=True)
        for k in sorted((k for k in in_use if k.isdigit() and k not in have), key=int)
    ]
    return sorted([*devs, *extra], key=lambda d: (d.index is None, d.index if d.index is not None else 0, d.path))


def _app_cameras() -> set[str]:
    try:
        from away.app.camera_share import CameraShare

        return CameraShare().owned_devices()
    except Exception:
        return set()


# Filtered by WMI itself (listing every PnP entity and filtering in PowerShell can take longer than the
# timeout on a busy PC). USB webcams are class Camera (Windows 10+) or Image (older drivers); ``usbvideo``
# catches UVC cameras some drivers file under another class.
WINDOWS_CAMERA_QUERY = (
    "Get-CimInstance -ClassName Win32_PnPEntity -Filter \"PNPClass='Camera' OR PNPClass='Image' OR "
    "Service='usbvideo'\" | Where-Object { $_.Status -eq 'OK' } | Select-Object -ExpandProperty Name"
)
WINDOWS_QUERY_TIMEOUT_S = 15


def parse_name_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        n = line.strip()
        if n and n not in out:
            out.append(n)
    return out


def windows_camera_names(run: Callable[..., Any] | None = None) -> list[str]:
    """Names of the connected cameras from Windows PnP (does not open them). [] on any failure."""
    import subprocess

    runner = run or subprocess.run
    try:
        r = runner(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_CAMERA_QUERY],
            stdin=subprocess.DEVNULL,  # a windowed exe has no usable stdin handle to inherit
            capture_output=True,
            text=True,
            errors="replace",
            timeout=WINDOWS_QUERY_TIMEOUT_S,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return parse_name_lines(r.stdout or "") if r.returncode == 0 else []


def parse_system_profiler(text: str) -> list[str]:
    import json

    try:
        data = json.loads(text)
    except ValueError:
        return []
    items = data.get("SPCameraDataType", []) if isinstance(data, dict) else []
    return [str(i.get("_name")) for i in items if isinstance(i, dict) and i.get("_name")]


def macos_camera_names(run: Callable[..., Any] | None = None) -> list[str]:
    import subprocess

    runner = run or subprocess.run
    try:
        r = runner(
            ["system_profiler", "-json", "SPCameraDataType"], capture_output=True, text=True, timeout=8, check=False
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return parse_system_profiler(r.stdout or "") if r.returncode == 0 else []


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
    src: Any = c.index if c.index is not None else c.path
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if sys.platform.startswith("win") and isinstance(src, int) else None
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(src)
    try:
        ok = bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()
    return CameraDevice(c.index, c.path, c.name, ok, c.in_use)


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

    def seen(self, cam: int) -> bool:
        """Has camera ``cam`` delivered at least one frame?"""
        return self._last_seen[cam] > -1e9

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


class CameraThread(threading.Thread):
    """Reads one camera as fast as it delivers, stamping each frame with ``time.monotonic()`` on arrival.

    * ``open_lock``: when given, the device is opened while holding it, so several cameras open one after
      another instead of all at once (Windows MSMF/DirectShow is slow and flaky opening two USB webcams
      at the same moment). Reading still runs in parallel once a camera is open.
    * A freshly opened camera may return empty reads for a while before its first frame (Windows webcams
      can take seconds); that is tolerated for ``first_frame_s``. After the first frame, 30 failed reads
      in a row mean the camera is gone.
    * ``error`` is set when the thread ends because of a problem; ``opened`` once the device opened.
    """

    def __init__(
        self,
        cam: int,
        spec: CameraSpec,
        pairer: FramePairer,
        stop: threading.Event,
        open_lock: threading.Lock | None = None,
        first_frame_s: float = 10.0,
        opener: Callable[[CameraSpec], Any] | None = None,
        convert: Callable[[Any, bool], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(daemon=True, name=f"camera-{cam}")
        self.cam, self.spec, self.pairer, self.stop = cam, spec, pairer, stop
        self.open_lock = open_lock
        self.first_frame_s = first_frame_s
        self.opener = opener or open_capture
        self.convert = convert or _bgr_to_rgb
        self.clock = clock
        self.error: str | None = None
        self.opened = threading.Event()
        self.frames = 0
        self.size: tuple[int, int] = (spec.width, spec.height)

    def _open(self) -> Any:
        if self.open_lock is None:
            return self.opener(self.spec)
        while not self.open_lock.acquire(timeout=0.1):
            if self.stop.is_set():
                return None
        try:
            return None if self.stop.is_set() else self.opener(self.spec)
        finally:
            self.open_lock.release()

    def run(self) -> None:
        try:
            cap = self._open()
        except Exception as exc:
            self.error = f"cannot open camera {self.spec.device!r}: {exc}" if "camera" not in str(exc) else str(exc)
            return
        if cap is None:  # stopped while waiting for its turn to open
            return
        self.opened.set()
        try:
            failures = 0
            started = self.clock()
            while not self.stop.is_set():
                ok, frame = cap.read()
                t = self.clock()
                if not ok or frame is None:
                    failures += 1
                    if self.frames == 0:
                        if t - started > self.first_frame_s:
                            self.error = f"camera opened but sent no frame within {self.first_frame_s:g} s"
                            return
                    elif failures > 30:  # unplugged: stop; the pairer marks this camera dropped
                        self.error = "camera stopped delivering frames"
                        return
                    time.sleep(0.02)
                    continue
                failures = 0
                self.frames += 1
                self.size = (frame.shape[1], frame.shape[0])
                self.pairer.push(self.cam, t, self.convert(frame, self.spec.mirror))
        finally:
            with contextlib.suppress(Exception):
                cap.release()


def _bgr_to_rgb(frame: Any, mirror: bool) -> Any:  # pragma: no cover - needs OpenCV
    import cv2

    if mirror:
        frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
