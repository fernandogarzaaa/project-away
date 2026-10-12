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
import logging
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

log = logging.getLogger("away.cameras")


# Virtual cameras (software filters that DirectShow lists next to real webcams, but WMI/PnP does not). They are
# never auto-picked as a default or fallback camera; the user can still choose one explicitly.
VIRTUAL_CAMERA_RE = re.compile(
    r"\b(nvidia broadcast|obs virtual|obs-camera|obs camera|iriun|droidcam|camo|snap camera|manycam|xsplit|"
    r"splitcam|e2esoft|vcam|youcam|logi capture|mmhmm|chromacam|epoccam|ivcam|ndi webcam|ndi virtual|"
    r"streamlabs|virtual)\b",
    re.IGNORECASE,
)


def is_virtual_camera(name: str) -> bool:
    """True for a known virtual (software) camera: NVIDIA Broadcast, OBS Virtual Camera, Iriun, DroidCam..."""
    return bool(VIRTUAL_CAMERA_RE.search(name or ""))


@dataclass
class CameraDevice:
    index: int | None  # OpenCV index (None when only a path is known); on Windows the DirectShow index
    path: str  # /dev/video0 on Linux, "camera 0" elsewhere
    name: str
    ok: bool | None = None  # opened and returned a frame (only when probed)
    in_use: bool = False  # held by the running Project Away app (not probed, so it is not disturbed)
    virtual: bool = False  # a software camera (NVIDIA Broadcast, OBS...): never auto-picked
    device_path: str = ""  # Windows DirectShow DevicePath (tells two cameras with the same name apart)

    @property
    def key(self) -> str:
        return str(self.index if self.index is not None else self.path)

    @property
    def placeholder(self) -> bool:
        """The OS gave no real name (``camera 3``): only the index identifies it."""
        return self.name == f"camera {self.index}" or self.name == self.path


# where the last list_cameras() call got its list: v4l2 | dshow | wmi | system_profiler | probe
last_source = ""
_last_logged: tuple[Any, ...] | None = None


def describe_list(devs: list[CameraDevice], source: str) -> str:
    rows = [
        f"{d.index if d.index is not None else d.path}: {d.name}"
        + (" (virtual)" if d.virtual else "")
        + (" (in use)" if d.in_use else "")
        for d in devs
    ]
    return f"cameras ({source}): " + ("; ".join(rows) if rows else "none found")


def log_camera_list(devs: list[CameraDevice], source: str, force: bool = False) -> bool:
    """Log the enumerated cameras (source, index, name, virtual) to away.log: when ``force`` (start-up, Settings
    'Refresh list') or when the list changed since the last time (hot-plug). True when a line was written."""
    global _last_logged
    sig = (source, tuple((d.index, d.name, d.virtual, d.device_path) for d in devs))
    if not force and sig == _last_logged:
        return False
    _last_logged = sig
    log.info("%s", describe_list(devs, source))
    return True


def list_cameras(
    sysfs: str = "/sys/class/video4linux",
    dev: str = "/dev",
    probe: bool = False,
    max_index: int = 8,
    in_use: set[str] | None = None,
    platform: str | None = None,
    run: Callable[..., Any] | None = None,
    probe_fn: Callable[[CameraDevice], CameraDevice] | None = None,
    dshow: Callable[[], list[tuple[str, str]] | None] | None = None,
    log_list: bool | None = None,
) -> list[CameraDevice]:
    """Enumerate cameras without opening them where the OS allows it.

    * Linux: V4L2 capture nodes with their names (``probe`` also opens each one).
    * Windows: DirectShow's own video-input list (``ICreateDevEnum``, the order ``cv2.CAP_DSHOW`` numbers
      its indices in), so names and indices come from the same list, virtual cameras included and flagged.
      Only when that fails: the PnP/WMI camera names (PowerShell CIM query, 15 s timeout), which miss
      virtual cameras and may be numbered differently (logged as a fallback).
    * macOS: ``system_profiler SPCameraDataType``.
    * Otherwise (or when the query finds nothing): OpenCV indices ``0..max_index-1`` are opened.

    ``in_use``: device keys (``"0"``, ``"/dev/video0"``, a camera name) the running app holds (camera_share owner
    file); they are listed and marked instead of probed, since an exclusive camera would fail to open and vanish
    from the list (and are added even when the OS query missed them). ``None`` reads the owner file.
    ``dshow``: the DirectShow enumerator (tests inject a fake); ``log_list``: True logs the list now, None logs
    it only when it changed since the last log line, False never.
    """
    global last_source
    devs, source = _enumerate(sysfs, dev, probe, max_index, in_use, platform, run, probe_fn, dshow)
    last_source = source
    if log_list is not False:
        log_camera_list(devs, source, force=bool(log_list))
    return devs


def _held(in_use: set[str], *keys: Any) -> bool:
    return any(str(k) in in_use for k in keys if k not in (None, ""))


def _enumerate(
    sysfs: str,
    dev: str,
    probe: bool,
    max_index: int,
    in_use: set[str] | None,
    platform: str | None,
    run: Callable[..., Any] | None,
    probe_fn: Callable[[CameraDevice], CameraDevice] | None,
    dshow: Callable[[], list[tuple[str, str]] | None] | None,
) -> tuple[list[CameraDevice], str]:
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
            out.append(
                CameraDevice(num, path, name, in_use=_held(in_use, num, path, name), virtual=is_virtual_camera(name))
            )
        if probe:  # pragma: no cover - opens devices
            out = [c if c.in_use else _probe(c) for c in out]
        return out, "v4l2"
    names: list[str] = []
    source = ""
    if plat == "win32":
        entries = (dshow or windows_dshow_cameras)()
        if entries:
            devs = dshow_devices(entries, in_use)
            if probe:  # pragma: no cover - opens devices
                devs = [c if c.in_use else _probe(c) for c in devs]
            return with_in_use(devs, in_use), "dshow"
        log.warning(
            "DirectShow camera enumeration %s; falling back to the Windows PnP list (its numbering may not match)",
            "failed" if entries is None else "found no camera",
        )
        names, source = windows_camera_names(run), "wmi"
    elif plat == "darwin":
        names, source = macos_camera_names(run), "system_profiler"
    if names:
        devs = [
            CameraDevice(i, f"camera {i}", n, in_use=_held(in_use, i, n), virtual=is_virtual_camera(n))
            for i, n in enumerate(names)
        ]
        if probe:  # pragma: no cover - opens devices
            devs = [c if c.in_use else _probe(c) for c in devs]
        return with_in_use(devs, in_use), source
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
    return with_in_use(found, in_use), "probe"


def dshow_devices(entries: list[tuple[str, str]], in_use: set[str] | None = None) -> list[CameraDevice]:
    """DirectShow ``(FriendlyName, DevicePath)`` pairs, in enumeration order -> devices numbered exactly as
    ``cv2.VideoCapture(i, cv2.CAP_DSHOW)`` opens them. Duplicate names are kept (two identical webcams)."""
    held = in_use or set()
    devs = []
    for i, (name, dpath) in enumerate(entries):
        name = (name or "").strip() or f"camera {i}"
        devs.append(
            CameraDevice(
                i,
                f"camera {i}",
                name,
                in_use=_held(held, i, name, dpath),
                virtual=is_virtual_camera(name),
                device_path=dpath or "",
            )
        )
    for d in devs:  # a held camera may be recorded by its config label ("HD User Facing #2")
        if not d.in_use and device_label(d, devs) in held:
            d.in_use = True
    return devs


# DirectShow COM identifiers (strmif.h / uuids.h)
CLSID_SystemDeviceEnum = "{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}"
IID_ICreateDevEnum = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
CLSID_VideoInputDeviceCategory = "{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
CLSID_LegacyAmFilterCategory = "{083863F1-70DE-11D0-BD40-00A0C911CE86}"  # always populated (used by a CI test)
IID_IPropertyBag = "{55272A00-42CB-11CE-8135-00AA004BB851}"


def windows_dshow_cameras(category: str = CLSID_VideoInputDeviceCategory) -> list[tuple[str, str]] | None:
    """``[(FriendlyName, DevicePath), ...]`` of the DirectShow ``category`` in enumeration order, which is the
    order OpenCV's DirectShow backend numbers cameras in. ``[]`` when there are none, ``None`` when the
    enumeration itself failed (not Windows, COM error). Runs on its own thread so its COM apartment never
    clashes with the caller's (Tk, pystray)."""
    result: list[Any] = [None]

    def work() -> None:
        try:
            result[0] = _dshow_enumerate(category)
        except Exception as exc:  # pragma: no cover - Windows only
            log.warning("DirectShow camera enumeration failed: %s", exc)
            result[0] = None

    th = threading.Thread(target=work, name="dshow-enum", daemon=True)
    th.start()
    th.join(timeout=10.0)
    return result[0] if not th.is_alive() else None


def _dshow_enumerate(category: str) -> list[tuple[str, str]] | None:  # pragma: no cover - Windows only
    """Pure ctypes ICreateDevEnum -> IEnumMoniker -> IPropertyBag (no comtypes/pygrabber dependency)."""
    import ctypes
    import uuid

    if not hasattr(ctypes, "WINFUNCTYPE") or not hasattr(ctypes, "windll"):
        return None
    from ctypes import POINTER, byref, c_long, c_ulong, c_ushort, c_void_p, c_wchar_p

    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

    def guid(s: str) -> GUID:
        return GUID.from_buffer_copy(uuid.UUID(s).bytes_le)

    class VARIANT(ctypes.Structure):  # 16 bytes on 32-bit, 24 on 64-bit
        _fields_ = [
            ("vt", c_ushort),
            ("r1", c_ushort),
            ("r2", c_ushort),
            ("r3", c_ushort),
            ("val", c_void_p),
            ("pad", c_void_p),
        ]

    hresult = c_long
    windll: Any = getattr(ctypes, "windll")  # noqa: B009 - Windows-only attribute (mypy on Linux)
    winfunctype: Any = getattr(ctypes, "WINFUNCTYPE")  # noqa: B009
    ole32, oleaut32 = windll.ole32, windll.oleaut32
    ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    ole32.CoInitializeEx.restype = hresult
    ole32.CoCreateInstance.argtypes = [POINTER(GUID), c_void_p, c_ulong, POINTER(GUID), POINTER(c_void_p)]
    ole32.CoCreateInstance.restype = hresult
    oleaut32.VariantClear.argtypes = [POINTER(VARIANT)]
    oleaut32.VariantClear.restype = hresult

    def method(obj: Any, index: int, *argtypes: Any) -> Any:
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
        return winfunctype(hresult, c_void_p, *argtypes)(vtbl[index])

    def release(obj: c_void_p) -> None:
        if obj.value:
            method(obj, 2)(obj)

    hr = ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED, on this private thread
    initialised = hr in (0, 1)  # S_OK / S_FALSE
    dev_enum, enum_mk = c_void_p(), c_void_p()
    out: list[tuple[str, str]] = []
    try:
        hr = ole32.CoCreateInstance(
            byref(guid(CLSID_SystemDeviceEnum)), None, 1, byref(guid(IID_ICreateDevEnum)), byref(dev_enum)
        )
        if hr != 0 or not dev_enum.value:
            log.warning("DirectShow: CoCreateInstance(SystemDeviceEnum) failed (0x%08X)", hr & 0xFFFFFFFF)
            return None
        # ICreateDevEnum::CreateClassEnumerator(REFCLSID, IEnumMoniker**, DWORD) = vtable slot 3
        hr = method(dev_enum, 3, POINTER(GUID), POINTER(c_void_p), c_ulong)(
            dev_enum, byref(guid(category)), byref(enum_mk), 0
        )
        if hr == 1 or not enum_mk.value:  # S_FALSE: the category is empty
            return out
        if hr != 0:
            log.warning("DirectShow: CreateClassEnumerator failed (0x%08X)", hr & 0xFFFFFFFF)
            return None
        next_fn = method(enum_mk, 3, c_ulong, POINTER(c_void_p), POINTER(c_ulong))  # IEnumMoniker::Next
        iid_bag = guid(IID_IPropertyBag)
        while True:
            mk, fetched = c_void_p(), c_ulong(0)
            if next_fn(enum_mk, 1, byref(mk), byref(fetched)) != 0 or fetched.value != 1 or not mk.value:
                break
            bag = c_void_p()
            try:
                # IMoniker::BindToStorage(IBindCtx*, IMoniker*, REFIID, void**) = vtable slot 9
                hr = method(mk, 9, c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p))(
                    mk, None, None, byref(iid_bag), byref(bag)
                )
                props: dict[str, str] = {}
                if hr == 0 and bag.value:
                    read = method(bag, 3, c_wchar_p, POINTER(VARIANT), c_void_p)  # IPropertyBag::Read
                    for prop in ("FriendlyName", "DevicePath"):
                        var = VARIANT()
                        if read(bag, prop, byref(var), None) == 0 and var.vt == 8 and var.val:  # VT_BSTR
                            props[prop] = ctypes.wstring_at(var.val)
                        oleaut32.VariantClear(byref(var))
                out.append((props.get("FriendlyName", ""), props.get("DevicePath", "")))
            finally:
                release(bag)
                release(mk)
        return out
    finally:
        release(enum_mk)
        release(dev_enum)
        if initialised:
            ole32.CoUninitialize()


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


def _windows() -> bool:
    return sys.platform.startswith("win")


def load_cv2() -> Any:  # pragma: no cover - real OpenCV
    import cv2

    return cv2


def open_video(src: int | str, cv2: Any = None) -> Any:
    """``cv2.VideoCapture`` for ``src``. On Windows an integer index is opened with DirectShow, the backend whose
    numbering :func:`list_cameras` reports (and that lists virtual cameras); Media Foundation is tried only when
    DirectShow cannot open it, and that is logged (MSMF numbers cameras its own way). Everything else uses
    OpenCV's default backend."""
    if cv2 is None:  # pragma: no cover - real OpenCV
        cv2 = load_cv2()
    if not (_windows() and isinstance(src, int)):
        return cv2.VideoCapture(src)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    with contextlib.suppress(Exception):
        cap.release()
    log.warning("camera %s: DirectShow could not open it; trying Media Foundation (its numbering may differ)", src)
    cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
    if cap.isOpened():
        log.info("camera %s: opened with Media Foundation (fallback)", src)
    return cap


# ------------------------------------------------------------------------------ opening with a capture mode
# v0.1.12 (Acer Predator: JoyCam + HD User Facing + NVIDIA Broadcast): no pixel format was ever requested, so
# DirectShow negotiated uncompressed YUY2 (two USB 2.0 webcams at 640x480 YUY2 can exceed the bus).
# v0.1.13: the built-in camera failed even alone, with NVIDIA Broadcast not running, while Broadcast's virtual
# camera showed its picture. The Media Foundation fallback reused the DirectShow index, but Media Foundation
# numbers cameras its own way (it does not list DirectShow-only virtual cameras), so it opened the wrong device
# or none. Media Foundation is now opened at the index its own list gives the camera.
DEFAULT_FOURCC = "MJPG"
DSHOW_FOURCCS = ("MJPG", "YUY2", "NV12")  # the formats UVC webcams report; each is tried before giving up
LOW_RES = (320, 240)
LOW_RES_FPS = 15
FIRST_FRAME_S = 4.0  # the first capture mode may take a while to deliver (Windows webcams are slow to start)
RETRY_FRAME_S = 1.5  # every further mode (the multi-camera probe waits MULTICAM_PROBE_S = 20 s)
BANDWIDTH_HINT = "two USB cameras may exceed the USB bandwidth; try another USB port, or the other camera alone"
_BROADCAST_RE = re.compile(r"nvidia broadcast", re.IGNORECASE)


def is_broadcast_camera(name: str) -> bool:
    """NVIDIA Broadcast's virtual camera (``Camera (NVIDIA Broadcast)``)."""
    return bool(_BROADCAST_RE.search(name or ""))


def broadcast_hint(name: str) -> str:
    return f"{name} could not be opened directly; NVIDIA Broadcast's camera shows the same picture."


def _broadcast_name() -> str:
    return "Camera (NVIDIA Broadcast)"


@dataclass(frozen=True)
class CaptureMode:
    width: int
    height: int
    fps: int  # 0 = do not set
    fourcc: str  # "" = do not set (the driver's default format)

    def describe(self) -> str:
        if not self.width or not self.height:
            return "device default (nothing forced)" + (f" {self.fourcc}" if self.fourcc else "")
        return (
            f"{self.width}x{self.height}"
            + (f"@{self.fps}" if self.fps else "")
            + (f" {self.fourcc}" if self.fourcc else " driver format")
        )


def fourcc_code(text: str) -> int:
    """``cv2.VideoWriter_fourcc(*text)`` without needing it: four characters packed little-endian."""
    t = (str(text) + "    ")[:4]
    return ord(t[0]) | (ord(t[1]) << 8) | (ord(t[2]) << 16) | (ord(t[3]) << 24)


def normalize_fourcc(value: Any) -> str:
    """A configured FOURCC: ``""`` (do not set) or four printable ASCII characters, upper-cased."""
    if value is None:
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    if len(s) != 4 or not s.isascii() or not s.isprintable():
        raise ValueError(f"a FOURCC is 4 characters such as MJPG or YUY2 (or empty), got {value!r}")
    return s.upper()


DEVICE_DEFAULT = CaptureMode(0, 0, 0, "")  # set nothing: the camera's own default size and format


def capture_modes(
    width: int, height: int, fps: int, fourcc: str, low_res: bool, other_fourccs: bool = True
) -> list[CaptureMode]:
    """The modes to try, in order: as configured; the same size in every other format webcams report
    (``DSHOW_FOURCCS``: MJPG, YUY2, NV12; only when a FOURCC is configured and ``other_fourccs``); the same size
    without a FOURCC; nothing forced at all (the device default); with ``low_res`` (a second camera on a busy
    USB bus) 320x240, then 320x240 at 15 fps."""
    fcc = normalize_fourcc(fourcc)
    w, h, f = int(width), int(height), int(fps or 0)
    modes = [CaptureMode(w, h, f, fcc)]
    if fcc and other_fourccs:
        modes += [CaptureMode(w, h, f, x) for x in DSHOW_FOURCCS if x != fcc]
    if fcc:
        modes.append(CaptureMode(w, h, f, ""))
    modes.append(DEVICE_DEFAULT)
    if low_res:
        lw, lh = LOW_RES
        modes.append(CaptureMode(lw, lh, f, fcc))
        modes.append(CaptureMode(lw, lh, LOW_RES_FPS, fcc))
    out: list[CaptureMode] = []
    for m in modes:
        if m not in out:
            out.append(m)
    return out


_lock = threading.Lock()
_session_modes: dict[str, tuple[str, CaptureMode]] = {}  # camera -> (backend, mode) that worked (kept)
_open_now: dict[str, int] = {}  # cameras this process has open right now (index or path -> count)


def session_mode(key: str) -> tuple[str, CaptureMode] | None:
    with _lock:
        return _session_modes.get(key)


def reset_session_modes() -> None:
    with _lock:
        _session_modes.clear()


def held_cameras() -> set[str]:
    """Keys (``"0"``, ``"/dev/video0"``) of the cameras this process holds open."""
    with _lock:
        return {k for k, n in _open_now.items() if n > 0}


def _hold(key: str, delta: int) -> None:
    with _lock:
        _open_now[key] = max(0, _open_now.get(key, 0) + delta)
        if not _open_now[key]:
            del _open_now[key]


class HeldCapture:
    """A ``cv2.VideoCapture`` that counts as held by this process until :meth:`release` (so a second camera
    knows another one is open, and the NVIDIA Broadcast fallback never takes a camera already in use)."""

    def __init__(self, cap: Any, key: str) -> None:
        self._cap = cap
        self._key = key
        self._released = False
        _hold(key, 1)

    def release(self) -> None:
        if not self._released:
            self._released = True
            _hold(self._key, -1)
        self._cap.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cap, name)


class CameraOpenError(RuntimeError):
    """A camera could not be used. ``opened``: the device opened but delivered no frame in any mode."""

    def __init__(self, message: str, opened: bool = False, hints: list[str] | None = None) -> None:
        super().__init__(message)
        self.opened = opened
        self.hints = list(hints or [])


@dataclass
class OpenedCamera:
    cap: Any  # a HeldCapture
    frame: Any  # the first frame (already read to prove the mode works)
    mode: CaptureMode
    src: int | str  # what was opened (differs from the request after the NVIDIA Broadcast fallback)
    name: str
    fallback: str = ""  # why another camera was opened instead, '' when it was not
    backend: str = ""  # "DirectShow", "Media Foundation", "default" (the backend that worked)


def apply_mode(cap: Any, mode: CaptureMode, cv2: Any) -> None:
    """FOURCC first (DirectShow picks the media type when the size is set), then size and frame rate."""
    if mode.fourcc:
        with contextlib.suppress(Exception):
            cap.set(getattr(cv2, "CAP_PROP_FOURCC", 6), fourcc_code(mode.fourcc))
    if mode.width and mode.height:
        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
    if mode.fps:
        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_FPS, mode.fps)
    with contextlib.suppress(Exception):
        cap.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)  # low latency; not every backend honours it


def read_first(
    cap: Any, timeout_s: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Any] = time.sleep
) -> Any:
    """The first frame ``cap`` delivers within ``timeout_s``, or None."""
    deadline = clock() + timeout_s
    while True:
        try:
            ok, frame = cap.read()
        except Exception:
            ok, frame = False, None
        if ok and frame is not None:
            return frame
        if clock() >= deadline:
            return None
        sleep(0.05)


@dataclass
class _Attempt:
    backend: str  # DirectShow | Media Foundation | default
    api: Any  # cv2.CAP_* (None = OpenCV's default backend)
    src: int | str  # the index in *that* backend's numbering
    modes: list[CaptureMode]


def _try_open(
    attempts: list[_Attempt],
    label: str,
    cv2: Any,
    first_frame_s: float,
    retry_frame_s: float,
    clock: Callable[[], float],
    sleep: Callable[[float], Any],
) -> tuple[Any, Any, CaptureMode | None, str, list[str], bool]:
    """Try every backend and mode in order; every attempt is logged. Returns ``(cap, frame, mode, backend,
    tried, opened_any)``: ``cap`` None when nothing delivered a frame. A backend whose device does not open is
    left at its first mode (another mode cannot help)."""
    opened_any = False
    tried: list[str] = []
    first = True
    for at in attempts:
        for mode in at.modes:
            log.info("camera %s: trying %s index %s, mode %s", label, at.backend, at.src, mode.describe())
            cap = cv2.VideoCapture(at.src) if at.api is None else cv2.VideoCapture(at.src, at.api)
            if not cap.isOpened():
                with contextlib.suppress(Exception):
                    cap.release()
                tried.append(f"{at.backend} {at.src}: did not open")
                log.warning("camera %s: %s could not open index %s", label, at.backend, at.src)
                break
            opened_any = True
            apply_mode(cap, mode, cv2)
            frame = read_first(cap, first_frame_s if first else retry_frame_s, clock, sleep)
            first = False
            if frame is not None:
                log.info(
                    "camera %s: %s index %s delivers frames in mode %s", label, at.backend, at.src, mode.describe()
                )
                return cap, frame, mode, at.backend, tried, True
            tried.append(f"{at.backend} {mode.describe()}: no frames")
            log.warning(
                "camera %s: %s opened index %s but sent no frame in mode %s", label, at.backend, at.src, mode.describe()
            )
            with contextlib.suppress(Exception):
                cap.release()
    return None, None, None, "", tried, opened_any


# Media Foundation COM identifiers (mfidl.h)
MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE = "{c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3}"
MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID = "{8ac3587a-4ae7-42d8-99e0-0a6013eef90f}"
MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME = "{60d0e559-52f8-4fa2-bbce-acdb34a8ec01}"
MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK = "{58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f}"


def windows_msmf_cameras() -> list[tuple[str, str]] | None:
    """``[(FriendlyName, SymbolicLink), ...]`` of Media Foundation's video capture sources in
    ``MFEnumDeviceSources`` order, the order OpenCV's ``CAP_MSMF`` backend numbers cameras in. ``None`` when the
    enumeration failed (not Windows, COM error). Runs on its own thread (own COM apartment)."""
    result: list[Any] = [None]

    def work() -> None:
        try:
            result[0] = _msmf_enumerate()
        except Exception as exc:  # pragma: no cover - Windows only
            log.warning("Media Foundation camera enumeration failed: %s", exc)
            result[0] = None

    th = threading.Thread(target=work, name="msmf-enum", daemon=True)
    th.start()
    th.join(timeout=10.0)
    return result[0] if not th.is_alive() else None


def _msmf_enumerate() -> list[tuple[str, str]] | None:  # pragma: no cover - Windows only
    """Pure ctypes MFCreateAttributes -> SetGUID(source type = video capture) -> MFEnumDeviceSources ->
    IMFActivate::GetAllocatedString(friendly name, symbolic link)."""
    import ctypes
    import uuid

    if not hasattr(ctypes, "WINFUNCTYPE") or not hasattr(ctypes, "windll"):
        return None
    from ctypes import POINTER, byref, c_long, c_uint32, c_ulong, c_void_p, c_wchar_p

    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

    def guid(text: str) -> GUID:
        return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)

    hresult = c_long
    windll: Any = getattr(ctypes, "windll")  # noqa: B009 - Windows-only attribute (mypy on Linux)
    winfunctype: Any = getattr(ctypes, "WINFUNCTYPE")  # noqa: B009
    ole32 = windll.ole32
    mfplat = windll.mfplat
    mf = windll.mf
    ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    ole32.CoInitializeEx.restype = hresult
    ole32.CoTaskMemFree.argtypes = [c_void_p]
    mfplat.MFStartup.argtypes = [c_ulong, c_ulong]
    mfplat.MFStartup.restype = hresult
    mfplat.MFShutdown.restype = hresult
    mfplat.MFCreateAttributes.argtypes = [POINTER(c_void_p), c_uint32]
    mfplat.MFCreateAttributes.restype = hresult
    mf.MFEnumDeviceSources.argtypes = [c_void_p, POINTER(POINTER(c_void_p)), POINTER(c_uint32)]
    mf.MFEnumDeviceSources.restype = hresult

    def method(obj: Any, index: int, *argtypes: Any) -> Any:
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
        return winfunctype(hresult, c_void_p, *argtypes)(vtbl[index])

    def release(obj: Any) -> None:
        if obj:
            method(obj, 2)(obj)

    hr = ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED, on this private thread
    initialised = hr in (0, 1)
    started = mfplat.MFStartup(0x00020070, 0) == 0  # MF_VERSION, MFSTARTUP_FULL
    attrs = c_void_p()
    devices = POINTER(c_void_p)()
    count = c_uint32(0)
    out: list[tuple[str, str]] = []
    try:
        if not started:
            log.warning("Media Foundation: MFStartup failed")
            return None
        if mfplat.MFCreateAttributes(byref(attrs), 1) != 0 or not attrs.value:
            return None
        # IMFAttributes::SetGUID(REFGUID key, REFGUID value) = vtable slot 24
        hr = method(attrs, 24, POINTER(GUID), POINTER(GUID))(
            attrs,
            byref(guid(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE)),
            byref(guid(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)),
        )
        if hr != 0:
            return None
        hr = mf.MFEnumDeviceSources(attrs, byref(devices), byref(count))
        if hr != 0:
            log.warning("Media Foundation: MFEnumDeviceSources failed (0x%08X)", hr & 0xFFFFFFFF)
            return None
        # IMFActivate inherits IMFAttributes: GetAllocatedString(REFGUID, LPWSTR*, UINT32*) = vtable slot 13
        for i in range(count.value):
            act = devices[i]
            values: list[str] = []
            for key in (MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK):
                text, length = c_wchar_p(), c_uint32(0)
                ok = method(act, 13, POINTER(GUID), POINTER(c_wchar_p), POINTER(c_uint32))(
                    act, byref(guid(key)), byref(text), byref(length)
                )
                values.append((text.value or "") if ok == 0 else "")
                if ok == 0 and text:
                    ole32.CoTaskMemFree(ctypes.cast(text, c_void_p))
            out.append((values[0], values[1]))
            release(act)
        return out
    finally:
        if devices:
            ole32.CoTaskMemFree(ctypes.cast(devices, c_void_p))
        release(attrs)
        if started:
            mfplat.MFShutdown()
        if initialised:
            ole32.CoUninitialize()


def msmf_index(
    src: int,
    me: CameraDevice | None,
    devices: list[CameraDevice],
    label: str,
    lister: Callable[[], list[tuple[str, str]] | None] | None = None,
) -> int | None:
    """The index OpenCV's ``CAP_MSMF`` uses for the DirectShow camera ``src``: matched by symbolic link
    (= DirectShow DevicePath), else by name and occurrence (the 2nd "USB Camera" is the 2nd one in both lists).
    None when Media Foundation does not list it; ``src`` itself (logged) when the list is unavailable."""
    entries = (lister or windows_msmf_cameras)()
    if entries is None:
        log.warning(
            "camera %s: Media Foundation list unavailable; trying index %d (its numbering may differ)", label, src
        )
        return src
    listed = "; ".join(f"{i}: {n}" for i, (n, _l) in enumerate(entries)) or "none"
    log.info("cameras (msmf): %s", listed)
    if me is not None and me.device_path:
        dp = me.device_path.lower()
        for i, (_n, link) in enumerate(entries):
            if link and link.lower() == dp:
                log.info("camera %s: DirectShow index %d is Media Foundation index %d (by device path)", label, src, i)
                return i
    name = me.name if me is not None else label.strip("'\"")
    same_ds = [d for d in devices if d.name.lower() == name.lower()]
    k = next((j for j, d in enumerate(same_ds) if d is me), 0)
    same_mf = [i for i, (n, _l) in enumerate(entries) if n.lower() == name.lower()]
    if k < len(same_mf):
        i = same_mf[k]
        log.info("camera %s: DirectShow index %d is Media Foundation index %d (by name)", label, src, i)
        return i
    log.warning("camera %s: not in Media Foundation's camera list (%s); skipping Media Foundation", label, listed)
    return None


def open_camera(
    src: int | str,
    width: int = 640,
    height: int = 480,
    fps: int = 0,
    fourcc: str = DEFAULT_FOURCC,
    *,
    cv2: Any = None,
    devices: list[CameraDevice] | None = None,
    label: str | None = None,
    low_res: bool | None = None,
    virtual_fallback: bool = False,
    first_frame_s: float = FIRST_FRAME_S,
    retry_frame_s: float = RETRY_FRAME_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
    msmf_lister: Callable[[], list[tuple[str, str]] | None] | None = None,
) -> OpenedCamera:
    """Open ``src`` (an index or path from :func:`resolve_device`) and prove it works by reading a frame.

    On Windows (an integer index), in order, every attempt logged:

    1. DirectShow at that index (:func:`capture_modes`): the configured FOURCC (MJPG) at the configured size,
       YUY2 and NV12 at that size, no FOURCC, then nothing forced at all; with ``low_res`` (default: when this
       process already holds another camera) 320x240 and 320x240 at 15 fps;
    2. Media Foundation at the index *its own* list gives this camera (:func:`msmf_index`): the configured
       size, then nothing forced;
    3. ``virtual_fallback``: exactly one free NVIDIA Broadcast virtual camera, which shows the same picture
       when the built-in camera is its source.

    The backend and mode that worked are kept and tried first on later opens in this session. Virtual cameras
    are opened in their own format (no FOURCC). Raises :class:`CameraOpenError` saying whether the camera did
    not open or opened without frames, with hints (USB bandwidth, NVIDIA Broadcast) that are also logged.
    """
    if cv2 is None:  # pragma: no cover - real OpenCV
        cv2 = load_cv2()
    devs = list(devices or [])
    me = next((d for d in devs if (d.index if d.index is not None else d.path) == src), None)
    name = me.name if me is not None else str(label if label is not None else src)
    key = me.name if me is not None else str(src)
    virtual = bool(me.virtual) if me is not None else is_virtual_camera(str(label or ""))
    others = held_cameras() - {str(src)}
    if low_res is None:
        low_res = bool(others)
    modes = capture_modes(
        width, height, fps, "" if virtual else fourcc, bool(low_res) and not virtual, other_fourccs=not virtual
    )
    kept = session_mode(key)
    windows = _windows() and isinstance(src, int)
    tried: list[str] = []
    opened = False

    def attempt(at: _Attempt) -> OpenedCamera | None:
        nonlocal opened
        if kept is not None and kept[0] == at.backend and kept[1] in at.modes:
            at.modes.remove(kept[1])
            at.modes.insert(0, kept[1])
        cap, frame, mode, backend, t, op = _try_open([at], repr(name), cv2, first_frame_s, retry_frame_s, clock, sleep)
        tried.extend(t)
        opened = opened or op
        if cap is None or mode is None:
            return None
        _remember(key, name, backend, mode, (at.backend, modes[0]))
        return OpenedCamera(HeldCapture(cap, str(src)), frame, mode, src, name, backend=backend)

    if not windows:
        got = attempt(_Attempt("default", None, src, list(modes)))
        if got is not None:
            return got
    else:
        assert isinstance(src, int)
        mf_modes = [CaptureMode(int(width), int(height), int(fps or 0), ""), DEVICE_DEFAULT]
        order = ["DirectShow", "Media Foundation"]
        if kept is not None and kept[0] == "Media Foundation":
            order.reverse()
        for backend in order:
            if backend == "DirectShow":
                got = attempt(_Attempt("DirectShow", cv2.CAP_DSHOW, src, list(modes)))
            else:
                log.warning("camera %r: trying Media Foundation at its own index for this camera", name)
                mfi = msmf_index(src, me, devs, repr(name), msmf_lister)
                got = None if mfi is None else attempt(_Attempt("Media Foundation", cv2.CAP_MSMF, mfi, list(mf_modes)))
            if got is not None:
                return got
    hints: list[str] = []
    broadcast = [d for d in devs if d.virtual and is_broadcast_camera(d.name) and d.index is not None]
    if not virtual and broadcast:
        hints.append(broadcast_hint(name))
    if others or low_res:
        hints.append(BANDWIDTH_HINT)
    for h in hints:
        log.warning("camera %r: %s", name, h)
    detail = "; ".join(tried) if tried else "nothing to try"
    if opened:
        msg = f"camera {name!r} opened but sent no frames (tried {detail})"
    else:
        msg = f"cannot open camera {name!r} (busy, no device, or permission denied; tried {detail})"
    if virtual_fallback and not virtual and isinstance(src, int):
        held = held_cameras()
        free = [d for d in broadcast if d.index != src and not d.in_use and str(d.index) not in held]
        alt_index = free[0].index if len(free) == 1 else None
        if alt_index is not None:
            alt = free[0]
            why = f"{name} could not be opened directly ({'no frames' if opened else 'open failed'})"
            log.warning(
                "camera %r: %s; opening the virtual camera %r instead (hands.virtual_fallback); NVIDIA "
                "Broadcast's camera shows the same picture",
                name,
                why,
                alt.name,
            )
            alt_modes = capture_modes(width, height, fps, "", False, other_fourccs=False)
            api = cv2.CAP_DSHOW if _windows() else None
            acap, aframe, amode, _b, atried, _o = _try_open(
                [_Attempt("DirectShow" if api is not None else "default", api, alt_index, alt_modes)],
                repr(alt.name),
                cv2,
                first_frame_s,
                retry_frame_s,
                clock,
                sleep,
            )
            if acap is not None and amode is not None:
                log.info("camera %r: using %r (virtual fallback), mode %s", name, alt.name, amode.describe())
                return OpenedCamera(
                    HeldCapture(acap, str(alt_index)), aframe, amode, alt_index, alt.name, why, backend=_b
                )
            log.warning(
                "camera %r: the virtual fallback %r did not work either (%s)", name, alt.name, "; ".join(atried)
            )
    raise CameraOpenError(msg + (". " + "; ".join(hints) if hints else ""), opened=opened, hints=hints)


def _remember(key: str, name: str, backend: str, mode: CaptureMode, first: tuple[str, CaptureMode]) -> None:
    with _lock:
        before = _session_modes.get(key)
        _session_modes[key] = (backend, mode)
    if (backend, mode) != first or (before is not None and before != (backend, mode)):
        log.warning(
            "camera %r: works with %s in mode %s, not %s %s; keeping that for this session",
            name,
            backend,
            mode.describe(),
            first[0],
            first[1].describe(),
        )
    elif before is None:
        log.info("camera %r: opened with %s in mode %s", name, backend, mode.describe())
    if (backend, mode) != first:
        remember_mode(key, backend, mode)  # only works in a fallback mode: open it that way at the next start
    elif key in saved_modes():
        forget_mode(key)  # the configured mode works again


# ------------------------------------------------------------------------------ modes kept across starts
# v0.1.15: a camera that only works in a fallback mode (typically the second webcam at 320x240 or MJPG on a busy
# USB bus) is remembered by name in camera-modes.json, so the next start opens it in that mode right away
# instead of spending ~10 s trying the configured mode first. Enabled by the app, Settings and the CLI with
# :func:`use_mode_store`; tests leave it off unless they turn it on.
MODES_FILE = "camera-modes.json"
_mode_file: Path | None = None


def default_mode_file() -> Path:
    from away import paths

    return Path(paths.data_dir()) / MODES_FILE


def use_mode_store(path: Path | None) -> None:
    """Keep fallback modes in ``path`` (None: off). Modes saved there are tried first in this session."""
    global _mode_file
    _mode_file = Path(path) if path is not None else None
    if _mode_file is None:
        return
    for key, (backend, mode) in saved_modes().items():
        with _lock:
            _session_modes.setdefault(key, (backend, mode))


def mode_store() -> Path | None:
    return _mode_file


def saved_modes() -> dict[str, tuple[str, CaptureMode]]:
    """``{camera name: (backend, mode)}`` from the mode file; ``{}`` when off, missing or unreadable."""
    if _mode_file is None:
        return {}
    import json

    try:
        data = json.loads(_mode_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, tuple[str, CaptureMode]] = {}
    if not isinstance(data, dict):
        return out
    for key, v in data.items():
        try:
            m = CaptureMode(int(v["width"]), int(v["height"]), int(v["fps"]), normalize_fourcc(v["fourcc"]))
            out[str(key)] = (str(v["backend"]), m)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _write_modes(modes: dict[str, tuple[str, CaptureMode]]) -> None:
    if _mode_file is None:
        return
    import json

    data = {
        k: {"backend": b, "width": m.width, "height": m.height, "fps": m.fps, "fourcc": m.fourcc}
        for k, (b, m) in sorted(modes.items())
    }
    with contextlib.suppress(OSError):
        _mode_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = _mode_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, _mode_file)


def remember_mode(key: str, backend: str, mode: CaptureMode) -> None:
    """Save the mode a camera works in (when the store is on) and use it for the rest of this session."""
    with _lock:
        _session_modes[key] = (backend, mode)
    if _mode_file is None:
        return
    modes = saved_modes()
    if modes.get(key) != (backend, mode):
        modes[key] = (backend, mode)
        _write_modes(modes)
        log.info("camera %r: remembered %s %s for the next start", key, backend, mode.describe())


def forget_mode(key: str) -> None:
    modes = saved_modes()
    if key in modes:
        del modes[key]
        _write_modes(modes)


def camera_devices() -> list[CameraDevice]:
    """The listed cameras for :func:`open_camera` (Windows only; elsewhere an empty list). Never raises."""
    if not _windows():
        return []
    try:
        return list_cameras()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("camera listing failed: %s", exc)
        return []


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
    cap = open_video(src, cv2)
    try:
        ok = bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()
    return CameraDevice(c.index, c.path, c.name, ok, c.in_use, c.virtual, c.device_path)


_OCCURRENCE = re.compile(r"^(.*\S)\s+#(\d+)$")


def device_label(dev: CameraDevice, devices: list[CameraDevice]) -> str:
    """How a camera is stored in the config: its name; when several cameras share that name, its DevicePath
    (Windows) or the name plus its occurrence number (``"USB Camera #2"``)."""
    same = [d for d in devices if d.name.lower() == dev.name.lower()]
    if len(same) <= 1:
        return dev.name
    if dev.device_path:
        return dev.device_path
    return f"{dev.name} #{next((k for k, d in enumerate(same, 1) if d is dev), 1)}"


def config_device(dev: CameraDevice, devices: list[CameraDevice]) -> int | str:
    """What to save for ``dev``: its name (stable across hot-plug and reordering), or the bare index when the
    OS gave no real name for it."""
    if dev.placeholder and dev.index is not None:
        return dev.index
    return device_label(dev, devices)


def find_device(device: int | str, devices: list[CameraDevice]) -> CameraDevice | None:
    """The listed camera ``device`` (an index, a DevicePath, ``name``, ``name #k``, or part of a name) means."""
    if isinstance(device, int) or (isinstance(device, str) and device.strip().isdigit()):
        i = int(device)
        return next((c for c in devices if c.index == i), None)
    d = device.strip()
    low = d.lower()
    for c in devices:
        if c.device_path and c.device_path.lower() == low:
            return c
    exact = [c for c in devices if c.name.lower() == low]
    if exact:
        return exact[0]
    m = _OCCURRENCE.match(d)
    if m:
        same = [c for c in devices if c.name.lower() == m.group(1).lower()]
        k = int(m.group(2))
        if 1 <= k <= len(same):
            return same[k - 1]
    for c in devices:
        if c.path == d:
            return c
    return next((c for c in devices if low in c.name.lower()), None)


def first_real(devices: list[CameraDevice], exclude: set[int] | None = None) -> CameraDevice | None:
    """The first non-virtual camera (default / fallback pick); virtual ones are only used when chosen."""
    skip = exclude or set()
    return next((c for c in devices if not c.virtual and c.index is not None and c.index not in skip), None)


def resolve_device(device: int | str, devices: list[CameraDevice] | None = None, hint: int | None = None) -> int | str:
    """Config ``device`` -> what ``cv2.VideoCapture`` takes: an index, a path, or the current index of a camera
    by name (resolved again at every open, so a hot-plugged camera can never redirect it).

    * A name / ``name #k`` / DevicePath must match a listed camera; ``hint`` (the index saved with it) is
      used only when no list is available at all.
    * An integer index on Windows is a setting from an older version (numbered from the PnP list): if it now
      points at a virtual camera (NVIDIA Broadcast is usually DirectShow camera 0), the first real camera is
      used instead and that is logged. Choosing a virtual camera by name still works.
    """
    if isinstance(device, bool):
        raise ValueError(f"invalid camera {device!r}")
    if isinstance(device, str):
        d = device.strip()
        if d.isdigit():
            device = int(d)
        elif d.startswith(("/dev/", "http://", "https://", "rtsp://")) or (
            not d.startswith("\\\\?\\") and (d.startswith("\\\\") or os.path.exists(d))
        ):
            return d
    if isinstance(device, int):
        if devices is None and not _windows():
            return device
        devs = list_cameras() if devices is None else devices
        return _legacy_index(device, devs)
    devs = list_cameras() if devices is None else devices
    c = find_device(device, devs)
    if c is not None:
        return c.index if c.index is not None else c.path
    if not devs and hint is not None:
        log.warning("camera %r: no camera list available; using the saved index %d", device, hint)
        return hint
    raise ValueError(f"no camera named {device!r} is connected; run `away cameras list`")


def _legacy_index(index: int, devs: list[CameraDevice], taken: set[int] | None = None) -> int:
    c = next((d for d in devs if d.index == index), None)
    if c is None or not c.virtual:
        return index
    real = first_real(devs, taken)
    if real is None or real.index is None:
        return index
    log.info(
        "camera index %d is the virtual camera %r; using %d (%r) instead. Choose a camera by name in "
        "Settings > Cameras to use the virtual one.",
        index,
        c.name,
        real.index,
        real.name,
    )
    return real.index


def pin_camera_set(specs: list[CameraSpec], devices: list[CameraDevice]) -> list[CameraSpec]:
    """Multi-camera specs with every bare index from an older version pinned to a camera name, so the set is
    resolved by name at every open. An index that points at a virtual camera is moved to the first real camera
    no other spec uses (two specs never end up on one device). Named specs are left as they are."""
    import copy

    out = [copy.copy(s) for s in specs]
    if not devices:
        return out
    taken: set[int] = set()
    for s in out:
        if isinstance(s.device, str) and not s.device.strip().isdigit():
            c = find_device(s.device, devices)
            if c is not None and c.index is not None:
                taken.add(c.index)
        elif not isinstance(s.device, bool):
            c = next((d for d in devices if d.index == int(s.device)), None)
            if c is not None and not c.virtual and c.index is not None:
                taken.add(c.index)
    for s in out:
        if isinstance(s.device, bool) or (isinstance(s.device, str) and not s.device.strip().isdigit()):
            continue
        idx = int(s.device)
        c = next((d for d in devices if d.index == idx), None)
        if c is None:
            continue
        if c.virtual:
            real = first_real(devices, taken)
            if real is None or real.index is None:
                continue
            log.info("camera index %d is the virtual camera %r; using %r instead", idx, c.name, real.name)
            c = real
        if c.index is not None:
            taken.add(c.index)
        s.device = config_device(c, devices)
    return out


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


def open_capture(spec: CameraSpec, virtual_fallback: bool = True, cv2: Any = None, low_res: bool | None = None) -> Any:
    """Open one camera of a multi-camera set (see :func:`open_camera`); returns a reader whose first ``read()``
    gives the frame that proved the mode works. A secondary camera may fall back to 320x240 (``low_res``
    overrides that: True also for the primary, False never)."""
    if cv2 is None:  # pragma: no cover - real OpenCV
        cv2 = load_cv2()
    devs = camera_devices()
    src = resolve_device(spec.device, devs if devs else None, hint=spec.index_hint)  # by name, at every open
    low = low_res if low_res is not None else (True if spec.role != "primary" else None)
    got = open_camera(
        src,
        spec.width,
        spec.height,
        spec.fps,
        getattr(spec, "fourcc", DEFAULT_FOURCC),
        cv2=cv2,
        devices=devs,
        label=str(spec.device),
        low_res=low,
        virtual_fallback=virtual_fallback,
    )
    return FirstFrameCapture(got.cap, got.frame, got.mode, got.backend, got.name)


class FirstFrameCapture:
    """A capture whose first ``read()`` returns a frame that was already read (while proving the mode).
    ``mode`` / ``backend`` / ``name`` say how it was opened (shown in Settings and ``away cameras test``)."""

    def __init__(
        self, cap: Any, frame: Any, mode: CaptureMode | None = None, backend: str = "", name: str = ""
    ) -> None:
        self._cap = cap
        self._first = frame
        self.mode = mode
        self.backend = backend
        self.name = name

    def read(self) -> tuple[bool, Any]:
        if self._first is not None:
            f, self._first = self._first, None
            return True, f
        return self._cap.read()  # type: ignore[no-any-return]

    def isOpened(self) -> bool:
        return bool(self._cap.isOpened())

    def release(self) -> None:
        self._cap.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cap, name)


PRIMARY_SETTLE_S = 8.0  # a secondary camera waits at most this long for the primary's first frame


class CameraThread(threading.Thread):
    """Reads one camera as fast as it delivers, stamping each frame with ``time.monotonic()`` on arrival.

    * ``open_lock``: when given, the device is opened while holding it, so several cameras open one after
      another instead of all at once (Windows MSMF/DirectShow is slow and flaky opening two USB webcams
      at the same moment). Reading still runs in parallel once a camera is open.
    * ``after``: an event to wait for (at most ``after_s``) before opening: a secondary camera waits for the
      primary's first frame (its ``ready``), so the primary always opens first.
    * A freshly opened camera may return empty reads for a while before its first frame (Windows webcams
      can take seconds); that is tolerated for ``first_frame_s``. After the first frame, 30 failed reads
      in a row mean the camera is gone.
    * ``error`` is set when the thread ends because of a problem; ``opened`` once the device opened;
      ``ready`` once it delivered its first frame or gave up; ``state`` is ``waiting`` / ``opening`` /
      ``open`` / ``streaming`` / ``failed`` / ``stopped``; ``mode`` / ``backend`` say how it was opened.
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
        after: threading.Event | None = None,
        after_s: float = PRIMARY_SETTLE_S,
    ) -> None:
        super().__init__(daemon=True, name=f"camera-{cam}")
        self.cam, self.spec, self.pairer, self.stop = cam, spec, pairer, stop
        self.open_lock = open_lock
        self.first_frame_s = first_frame_s
        self.opener = opener or open_capture
        self.convert = convert or _bgr_to_rgb
        self.clock = clock
        self.after = after
        self.after_s = after_s
        self.error: str | None = None
        self.opened = threading.Event()
        self.ready = threading.Event()
        self.frames = 0
        self.size: tuple[int, int] = (spec.width, spec.height)
        self.state = "waiting"
        self.mode: CaptureMode | None = None
        self.backend = ""
        self.first_t: float | None = None
        self.last_t: float | None = None

    def fps(self) -> float:
        """Measured frame rate since the first frame (0.0 before two frames)."""
        if self.frames < 2 or self.first_t is None or self.last_t is None or self.last_t <= self.first_t:
            return 0.0
        return (self.frames - 1) / (self.last_t - self.first_t)

    def describe_mode(self) -> str:
        """``320x240@15 MJPG (DirectShow)``, or the frame size when the opener did not say."""
        if self.mode is not None:
            text = self.mode.describe()
        elif self.frames:
            text = f"{self.size[0]}x{self.size[1]}"
        else:
            return ""
        return text + (f" ({self.backend})" if self.backend else "")

    def _wait_turn(self) -> bool:
        """Wait for ``after`` (the primary's first frame), at most ``after_s``; False when stopped."""
        if self.after is None:
            return not self.stop.is_set()
        deadline = self.clock() + self.after_s
        while not self.after.is_set() and self.clock() < deadline:
            if self.stop.wait(0.05):
                return False
        return not self.stop.is_set()

    def _open(self) -> Any:
        if not self._wait_turn():
            return None
        if self.open_lock is None:
            self.state = "opening"
            return self.opener(self.spec)
        while not self.open_lock.acquire(timeout=0.1):
            if self.stop.is_set():
                return None
        try:
            if self.stop.is_set():
                return None
            self.state = "opening"
            return self.opener(self.spec)
        finally:
            self.open_lock.release()

    def run(self) -> None:
        try:
            self._run()
        finally:
            if self.state not in ("failed",):
                self.state = "failed" if self.error else "stopped"
            self.ready.set()

    def _run(self) -> None:
        try:
            cap = self._open()
        except Exception as exc:
            self.error = f"cannot open camera {self.spec.device!r}: {exc}" if "camera" not in str(exc) else str(exc)
            self.state = "failed"
            return
        if cap is None:  # stopped while waiting for its turn to open
            return
        mode = getattr(cap, "mode", None)
        self.mode = mode if isinstance(mode, CaptureMode) else None
        backend = getattr(cap, "backend", "")
        self.backend = backend if isinstance(backend, str) else ""
        self.state = "open"
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
                            self.state = "failed"
                            return
                    elif failures > 30:  # unplugged: stop; the pairer marks this camera dropped
                        self.error = "camera stopped delivering frames"
                        self.state = "failed"
                        return
                    time.sleep(0.02)
                    continue
                failures = 0
                self.frames += 1
                if self.first_t is None:
                    self.first_t = t
                self.last_t = t
                self.size = (frame.shape[1], frame.shape[0])
                self.state = "streaming"
                self.pairer.push(self.cam, t, self.convert(frame, self.spec.mirror))
                self.ready.set()
        finally:
            with contextlib.suppress(Exception):
                cap.release()


def primary_index(specs: list[CameraSpec]) -> int:
    return next((i for i, c in enumerate(specs) if getattr(c, "role", "") == "primary"), 0)


def camera_opener(virtual_fallback: bool = True, low_res: bool | None = None) -> Callable[[CameraSpec], Any]:
    """The opener every multi-camera user shares: :func:`open_capture` with the NVIDIA Broadcast fallback
    setting, a secondary allowed down to 320x240, and the backend/mode kept for the session (and across
    starts with :func:`use_mode_store`)."""
    import functools

    if low_res is None:
        return functools.partial(open_capture, virtual_fallback=virtual_fallback)
    return functools.partial(open_capture, virtual_fallback=virtual_fallback, low_res=low_res)


def start_camera_threads(
    specs: list[CameraSpec],
    pairer: FramePairer,
    stop: threading.Event,
    *,
    open_lock: threading.Lock | None = None,
    serial: bool | None = None,
    virtual_fallback: bool = True,
    opener: Callable[[CameraSpec], Any] | None = None,
    settle_s: float = PRIMARY_SETTLE_S,
    thread_factory: Callable[..., Any] = CameraThread,
    start: bool = True,
) -> list[Any]:
    """Start one capture thread per camera, the same way everywhere (the app, Settings > Live confidence and
    Test both cameras, ``away calibrate-cameras`` and ``away cameras test``; v0.1.15):

    * one opener (:func:`camera_opener`): NVIDIA Broadcast fallback, the secondary's 320x240 and other-FOURCC
      fallbacks, kept modes;
    * opened one after another under ``open_lock`` (made here when ``serial``; default: on Windows, where two
      USB webcams opening at the same moment is slow and flaky);
    * the primary first: every other camera waits for the primary's first frame (at most ``settle_s``).
    """
    if open_lock is None and (_windows() if serial is None else serial):
        open_lock = threading.Lock()
    op = opener or camera_opener(virtual_fallback)
    first = primary_index(specs)
    threads: list[Any] = [None] * len(specs)
    threads[first] = thread_factory(first, specs[first], pairer, stop, open_lock=open_lock, opener=op)
    for i, c in enumerate(specs):
        if i != first:
            threads[i] = thread_factory(
                i, c, pairer, stop, open_lock=open_lock, opener=op, after=threads[first].ready, after_s=settle_s
            )
    if start:
        threads[first].start()
        for i, th in enumerate(threads):
            if i != first:
                th.start()
    return threads


def stop_camera_threads(stop: threading.Event, threads: list[Any], timeout: float = 3.0) -> bool:
    """Stop the capture threads and wait (``timeout`` in total) until they released their devices.
    True when every thread ended."""
    stop.set()
    deadline = time.monotonic() + timeout
    for th in threads:
        if th.is_alive() if hasattr(th, "is_alive") else False:
            th.join(max(0.0, deadline - time.monotonic()))
    return not any(getattr(th, "is_alive", lambda: False)() for th in threads)


def _bgr_to_rgb(frame: Any, mirror: bool) -> Any:  # pragma: no cover - needs OpenCV
    import cv2

    if mirror:
        frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
