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
from dataclasses import dataclass, field
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
# v0.1.16 (the same laptop): the built-in camera "worked" at 1.0 fps (10 frames in 10 s) in every mode, even at
# 320x240, so USB bandwidth was ruled out, yet it was reported ok. Every attempted mode is now measured for
# MEASURE_S (not just one first frame), a camera below the minimum frame rate (hands.min_camera_fps) is too slow,
# the fastest mode wins, and a camera that stays slow in every mode on both backends is replaced by a free NVIDIA
# Broadcast camera when that reaches the minimum itself.
MIN_CAMERA_FPS = 10.0  # the default of hands.min_camera_fps
# v0.1.21 (JoyCam: "1280x720@30 YUY2 ... 10.0 fps measured; ok"): uncompressed 720p over USB 2 stops at ~10 fps.
# A mode between MIN_CAMERA_FPS and TRACKING_FPS still works, but the search goes on to smaller sizes and keeps the
# fastest mode that reaches TRACKING_FPS (hand tracking needs speed more than resolution).
TRACKING_FPS = 24.0
SPEED_SIZES = ((960, 540), (640, 360), (640, 480))  # tried when the configured size is larger than these
SPEED_FOURCCS = ("MJPG", "YUY2")
MEASURE_S = 1.5  # frames are counted this long in every attempted mode (0 = accept the first frame unmeasured)
SWEEP_FOURCCS = ("MJPG", "YUY2", "NV12", "")  # "" = the driver's default format
SWEEP_SIZES = ((640, 480), (320, 240))  # after the configured size
# v0.1.19: 1280x720 last (MJPG and the driver's format): some built-in cameras deliver 30 fps only in their HD
# mode; the frames are downscaled to the configured size (ScaledCapture), keeping the aspect ratio
SWEEP_HD = (1280, 720)
SWEEP_HD_FOURCCS = ("MJPG", "")
SWEEP_FPS = 30  # CAP_PROP_FPS requested in the sweep when the configuration asks for none
SWEEP_BUDGET_S = 60.0  # the slow-camera sweep stops trying further modes after this long
BANDWIDTH_HINT = "two USB cameras may exceed the USB bandwidth; try another USB port, or the other camera alone"
_BROADCAST_RE = re.compile(r"nvidia broadcast", re.IGNORECASE)


# v0.1.19: hands.camera_backend. "auto" tries DirectShow, Media Foundation, then WinRT MediaCapture (the API the
# Windows Camera app uses: the Acer's built-in camera opened only there); the others use that backend alone.
WINRT = "WinRT"
CAMERA_BACKENDS = ("auto", "dshow", "msmf", "winrt")
_BACKEND_NAMES = {"dshow": "DirectShow", "msmf": "Media Foundation", "winrt": WINRT}
_backend_pref = "auto"


def backends_for(pref: str | None, windows: bool = True) -> list[str]:
    """The backends to try, in order, for hands.camera_backend ``pref`` (None: the one set with
    :func:`set_camera_backend`). Not on Windows: OpenCV's default backend only."""
    if not windows:
        return ["default"]
    p = (pref if pref is not None else _backend_pref) or "auto"
    if p == "auto":
        return ["DirectShow", "Media Foundation", WINRT]
    return [_BACKEND_NAMES[p]]


def set_camera_backend(pref: str) -> None:
    """Use hands.camera_backend ``pref`` (auto | dshow | msmf | winrt) for every open in this process."""
    global _backend_pref
    p = str(pref or "auto").strip().lower()
    if p not in CAMERA_BACKENDS:
        raise ValueError(f"camera backend must be one of {', '.join(CAMERA_BACKENDS)}, got {pref!r}")
    if p != _backend_pref:
        log.info("camera backend: %s", p)
    _backend_pref = p


def camera_backend() -> str:
    return _backend_pref


def configure(hands: Any) -> None:
    """Apply the camera settings of ``hands`` (a HandsConfig) that hold for the whole process."""
    with contextlib.suppress(ValueError):
        set_camera_backend(str(getattr(hands, "camera_backend", "auto") or "auto"))


def winrt_name(me: CameraDevice | None, label: str | None) -> str | None:
    """The name WinRT looks the camera up by: the listed name, else a configured name (not a bare index)."""
    if me is not None and me.name:
        return me.name
    text = str(label or "").strip().strip("'\"")
    return text if text and not text.isdigit() and not text.startswith("/dev/") else None


def native_modes(
    name: str | None, width: int, height: int, fps: int, lister: Callable[..., Any] | None = None, limit: int = 4
) -> list[CaptureMode]:
    """v0.1.23: the camera's own formats (from WinRT), closest to the request first, as capture modes."""
    if not name:
        return []
    try:
        if lister is None:
            ok, _why = winrt_factory_available()
            if not ok:
                return []
            from away.multicam import winrt_capture

            lister = winrt_capture.list_formats
        formats, _err = lister(name)
        from away.multicam.winrt_capture import ranked_formats

        out: list[CaptureMode] = []
        for f in ranked_formats(formats, width, height, fps):
            m = CaptureMode(int(f.width), int(f.height), round(f.fps), f.fourcc())
            if m not in out:
                out.append(m)
        return out[:limit]
    except Exception as exc:
        log.warning("camera %r: could not list its formats: %s", name, exc)
        return []


def winrt_factory_available() -> tuple[bool, str]:
    from away.multicam import winrt_capture

    return winrt_capture.available()


def winrt_factory(name: str, width: int, height: int, occurrence: int = 0) -> Callable[[CaptureMode], Any]:
    """Opens camera ``name`` through WinRT for a mode: the native format closest to it (a remembered native
    mode is matched exactly), frames downscaled to fit ``width`` x ``height``."""
    from away.multicam import winrt_capture

    def make(mode: CaptureMode) -> Any:
        return winrt_capture.WinrtCapture(
            name, mode.width, mode.height, mode.fps, mode.fourcc, out_size=(width, height), occurrence=occurrence
        )

    return make


_INTERNAL_RE = re.compile(r"user facing|integrated|internal|built-?in|facetime|front camera|laptop", re.I)
_configured_internal: set[str] = set()


def is_internal_camera(name: str) -> bool:
    """A laptop's built-in camera (``HD User Facing``, ``Integrated Camera``...)."""
    return bool(_INTERNAL_RE.search(str(name or "")))


def configure_cameras(devices: Any) -> None:
    """v0.1.23: the cameras the app is set up to use. While one of them is the built-in camera, NVIDIA Broadcast's
    virtual camera is never opened as a fallback: opening it wakes Broadcast up, and Broadcast then takes the
    built-in camera for itself, so the camera we wanted fails again."""
    _configured_internal.clear()
    _configured_internal.update(str(d) for d in devices if is_internal_camera(str(d)))


def broadcast_fallback_allowed(name: str) -> bool:
    return not is_internal_camera(name) and not _configured_internal


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
    if fcc and other_fourccs:
        modes += speed_modes(w, h, f)
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


def speed_modes(width: int, height: int, fps: int) -> list[CaptureMode]:
    """v0.1.21: smaller sizes (960x540, 640x360, 640x480; MJPG then YUY2) to try when the configured size is only
    slow: a webcam that sends 720p uncompressed at 10 fps usually sends 640x480 at 30."""
    out: list[CaptureMode] = []
    if int(width) * int(height) <= 640 * 480:
        return out  # already a tracking-sized mode
    for sw, sh in SPEED_SIZES:
        if sw * sh >= int(width) * int(height):
            continue
        for fcc in SPEED_FOURCCS:
            out.append(CaptureMode(sw, sh, int(fps or 0) or SWEEP_FPS, fcc))
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
        _slow_known.clear()


_slow_known: dict[str, float] = {}  # camera -> best fps of a full sweep that stayed below the minimum
_opening = 0  # opens in progress in this process (a device may be held before HeldCapture exists)
_abort_opens = threading.Event()  # set while handing the cameras over: every open in progress gives up
_handover_gen = 0  # bumped every time wait_devices_free finishes: an open queued on the gate while a
# hand-over ran must not proceed afterwards (the event is already cleared by then)
_open_gate = threading.Lock()  # v0.1.29: native camera backends are not thread safe: one open at a time per process


def opens_in_progress() -> int:
    with _lock:
        return _opening


def wait_devices_free(
    timeout: float = 15.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
    step: float = 0.05,
) -> bool:
    """Hand-over guard (v0.1.16): abort every open in progress in this process and wait until no camera is
    held or being opened, so another process (Settings > Test both cameras, calibration) never opens a device
    this process is still opening or releasing. True when free within ``timeout``."""
    _abort_opens.set()
    try:
        deadline = clock() + timeout
        # v0.1.29: do not declare the devices free while a native open is still mid-flight. An
        # in-progress open sees _abort_opens and gives up between attempts, then releases the
        # gate; waiting for the gate (up to the timeout) keeps hand-over from interleaving
        # with it. The gate is released right away: new opens during hand-over still fail
        # fast on _abort_opens instead of blocking behind the whole hand-over.
        remaining = deadline - clock()
        if remaining > 0 and _open_gate.acquire(timeout=remaining):
            _open_gate.release()
        else:
            log.warning("camera hand-over: an open did not finish within %.1f s", timeout)
        while held_cameras() or opens_in_progress():
            if clock() >= deadline:
                log.warning(
                    "cameras still busy after %.0f s (held: %s, opening: %d)",
                    timeout,
                    ", ".join(sorted(held_cameras())) or "-",
                    opens_in_progress(),
                )
                return False
            sleep(step)
        return True
    finally:
        _abort_opens.clear()
        # v0.1.29: a hand-over ran. An open that queued on _open_gate while this ran must
        # see that and give up instead of opening after the hand-over finished: the event
        # above is already cleared, so the in-flight cancellation check cannot see it.
        # (A hand-over cannot return True while an open is queued: the queued open holds
        # _opening > 0, so the drain loop keeps waiting. A changed generation therefore
        # always means the hand-over timed out and the open is stale.)
        global _handover_gen
        with _lock:
            _handover_gen += 1


class OpenCancelled(RuntimeError):
    """An open was abandoned because the cameras are being handed over to another process."""


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
    fps: float | None = None  # frame rate measured while choosing the mode (None: not measured)
    slow: bool = False  # below the minimum frame rate in every mode tried (opened anyway: the best one)
    hints: list[str] = field(default_factory=list)  # what to tell the user (Broadcast offer, too slow)


def apply_mode(cap: Any, mode: CaptureMode, cv2: Any, label: str = "") -> str:
    """FOURCC first (DirectShow picks the media type when the size is set), then size and frame rate.

    v0.1.19 (JoyCam asked for MJPG read back YUY2): the FOURCC is read back after the size was set; when the
    driver reports another one it is set once more (some DirectShow drivers only take it once the size is
    known) and read back again. Returns what the driver reports (``""`` when it reports nothing); a camera
    that ignores the request is logged as such (``label`` names it) instead of claiming the requested format."""
    prop = getattr(cv2, "CAP_PROP_FOURCC", 6)
    if mode.fourcc:
        with contextlib.suppress(Exception):
            cap.set(prop, fourcc_code(mode.fourcc))
    if mode.width and mode.height:
        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
    if mode.fps:
        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_FPS, mode.fps)
    with contextlib.suppress(Exception):
        cap.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)  # low latency; not every backend honours it
    if not mode.fourcc:
        return ""
    got = _read_fourcc(cap, prop)
    if got and got != mode.fourcc:
        with contextlib.suppress(Exception):
            cap.set(prop, fourcc_code(mode.fourcc))
            if mode.fps:  # DirectShow resets the frame rate when the media type changes
                cap.set(cv2.CAP_PROP_FPS, mode.fps)
        got = _read_fourcc(cap, prop)
        if got == mode.fourcc:
            log.info("camera %s: %s accepted only after the size was set", label or "?", mode.fourcc)
    if got and got != mode.fourcc:
        log.warning(
            "camera %s: asked for %s but the driver reports %s; it ignores the requested pixel format (frames are %s)",
            label or "?",
            mode.fourcc,
            got,
            got,
        )
    elif not got:
        log.info("camera %s: asked for %s; the driver does not report its pixel format", label or "?", mode.fourcc)
    return got


def apply_mode_size_first(cap: Any, mode: CaptureMode, cv2: Any, label: str = "") -> str:
    """The other order some DirectShow drivers need (v0.1.21, for MJPG): size, FOURCC, size again, then the frame
    rate, on a freshly opened capture. Returns the FOURCC the driver reports."""
    prop = getattr(cv2, "CAP_PROP_FOURCC", 6)
    with contextlib.suppress(Exception):
        if mode.width and mode.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
        cap.set(prop, fourcc_code(mode.fourcc))
        if mode.width and mode.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
        if mode.fps:
            cap.set(cv2.CAP_PROP_FPS, mode.fps)
        cap.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)
    got = _read_fourcc(cap, prop)
    log.info(
        "camera %s: retried %s with the size set first -> driver reports %s", label or "?", mode.fourcc, got or "?"
    )
    return got


def _read_fourcc(cap: Any, prop: int) -> str:
    try:
        return decode_fourcc(cap.get(prop))
    except Exception:
        return ""


def fit_within(width: int, height: int, max_w: int, max_h: int) -> tuple[int, int]:
    """``width`` x ``height`` scaled down (aspect ratio kept, never up) to fit ``max_w`` x ``max_h``; unchanged
    when either limit is 0 (auto) or it already fits. Even sizes (video-friendly)."""
    w, h, mw, mh = int(width), int(height), int(max_w or 0), int(max_h or 0)
    if not (w and h and mw and mh) or (w <= mw and h <= mh):
        return w, h
    s = min(mw / w, mh / h)
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def resize_frame(frame: Any, width: int, height: int) -> Any:
    """``frame`` resized to ``width`` x ``height`` (OpenCV's area filter when available, else nearest pixel)."""
    try:
        import cv2

        return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    except Exception:
        import numpy as np

        h, w = frame.shape[:2]
        ys = (np.arange(int(height)) * h // int(height)).clip(0, h - 1)
        xs = (np.arange(int(width)) * w // int(width)).clip(0, w - 1)
        return frame[ys][:, xs]


class ScaledCapture:
    """A capture whose frames are downscaled to fit ``max_w`` x ``max_h`` (aspect ratio kept): the 1280x720 sweep
    mode delivers HD frames while the rest of the app asked for the configured size (v0.1.19)."""

    def __init__(self, cap: Any, max_w: int, max_h: int) -> None:
        self._cap = cap
        self.max_w, self.max_h = int(max_w), int(max_h)

    def scale(self, frame: Any) -> Any:
        if frame is None or not hasattr(frame, "shape"):
            return frame
        h, w = frame.shape[:2]
        ow, oh = fit_within(w, h, self.max_w, self.max_h)
        return frame if (ow, oh) == (w, h) else resize_frame(frame, ow, oh)

    def read(self) -> tuple[bool, Any]:
        ok, frame = self._cap.read()
        return ok, (self.scale(frame) if ok else frame)

    def release(self) -> None:
        self._cap.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cap, name)


def scaled(cap: Any, frame: Any, mode: CaptureMode, width: int, height: int) -> tuple[Any, Any]:
    """Wrap ``cap`` in :class:`ScaledCapture` when ``mode`` is larger than the configured size (the HD sweep
    mode); ``(cap, first frame)`` either way."""
    if not (width and height and mode.width and mode.height) or (mode.width <= width and mode.height <= height):
        return cap, frame
    sc = ScaledCapture(cap, width, height)
    return sc, sc.scale(frame)


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
    backend: str  # DirectShow | Media Foundation | WinRT | default
    api: Any  # cv2.CAP_* (None = OpenCV's default backend)
    src: int | str  # the index in *that* backend's numbering
    modes: list[CaptureMode]
    # v0.1.19: opens a capture for a mode itself (the WinRT backend picks the native format from the mode);
    # None = cv2.VideoCapture(src, api) + apply_mode
    factory: Callable[[CaptureMode], Any] | None = None


def decode_fourcc(value: Any) -> str:
    """``CAP_PROP_FOURCC`` read back (a float holding four packed characters) -> ``"MJPG"``; ``""`` for 0 or
    nothing; ``0x...`` when it is not printable."""
    try:
        code = int(value)
    except (TypeError, ValueError):
        return ""
    if code <= 0:
        return ""
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text.strip() if text.isascii() and text.isprintable() else f"0x{code:08x}"


def read_back(cap: Any, cv2: Any) -> dict[str, Any]:
    """What the driver reports after a mode was applied: ``fourcc``, ``width``, ``height``, ``fps`` and the
    ``exposure`` / ``auto_exposure`` values (logged only: auto-exposure is never changed, but a webcam that
    throttles its frame rate in low light is one explanation for a slow camera)."""
    out: dict[str, Any] = {}

    def get(name: str, default: int) -> Any:
        try:
            return cap.get(getattr(cv2, name, default))
        except Exception:
            return None

    out["fourcc"] = decode_fourcc(get("CAP_PROP_FOURCC", 6))
    for key, prop, default in (
        ("width", "CAP_PROP_FRAME_WIDTH", 3),
        ("height", "CAP_PROP_FRAME_HEIGHT", 4),
        ("fps", "CAP_PROP_FPS", 5),
        ("exposure", "CAP_PROP_EXPOSURE", 15),
        ("auto_exposure", "CAP_PROP_AUTO_EXPOSURE", 21),
    ):
        v = get(prop, default)
        out[key] = v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    return out


def _num(v: Any, fmt: str = "{:g}") -> str:
    return "?" if v is None else fmt.format(v)


def attempt_line(
    label: str, backend: str, src: Any, mode: CaptureMode, rb: dict[str, Any] | None, fps: float | None, result: str
) -> str:
    """One away.log line per attempted mode (v0.1.16): what was asked, what the driver reports back
    (FOURCC, size, fps, exposure) and the measured frame rate, so the cause of a slow camera shows in the log."""
    if rb is None:
        got = "no read-back"
    else:
        size = f"{_num(rb.get('width'), '{:.0f}')}x{_num(rb.get('height'), '{:.0f}')}"
        got = (
            f"fourcc {rb.get('fourcc') or '?'}, {size}, driver fps {_num(rb.get('fps'))}, exposure "
            f"{_num(rb.get('exposure'))}, auto-exposure {_num(rb.get('auto_exposure'))}"
        )
    measured = "not measured" if fps is None else f"{fps:.1f} fps measured"
    return f"camera {label}: attempt {backend} index {src}, asked {mode.describe()} -> {got}; {measured}; {result}"


def measure_fps(
    cap: Any,
    frame: Any,
    seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[float | None, Any]:
    """Count the frames ``cap`` delivers for ``seconds`` after the first one; ``(fps, latest frame)``.
    ``(None, frame)`` when ``seconds`` <= 0 (not measured)."""
    if seconds <= 0:
        return None, frame
    t0 = clock()
    n = 0
    last = frame
    while clock() - t0 < seconds:
        if cancelled is not None and cancelled():
            break
        try:
            ok, f = cap.read()
        except Exception:
            ok, f = False, None
        if ok and f is not None:
            n += 1
            last = f
        else:
            sleep(0.01)
    elapsed = clock() - t0
    return (n / elapsed if elapsed > 0 else 0.0), last


@dataclass
class _Candidate:
    backend: str
    api: Any
    src: int | str
    mode: CaptureMode
    fps: float | None  # None: not measured
    factory: Callable[[CaptureMode], Any] | None = None


class _Search:
    """Try modes one after another, measuring each (v0.1.16). A mode that reaches ``min_fps`` (or that is not
    measured: ``measure_s`` 0) ends the search; a slower one is released and remembered, and the fastest of
    those is reopened when nothing is fast enough. Every attempt is logged (:func:`attempt_line`)."""

    def __init__(
        self,
        label: str,
        cv2: Any,
        first_frame_s: float,
        retry_frame_s: float,
        clock: Callable[[], float],
        sleep: Callable[[float], Any],
        min_fps: float,
        measure_s: float,
        good_fps: float | None = None,
        trusted: CaptureMode | None = None,
    ) -> None:
        self.label, self.cv2 = label, cv2
        # v0.1.21: a mode below good_fps (but at least min_fps) is remembered in ``ok_best`` and the search goes on;
        # ``trusted`` (the best mode a full search found before, camera-modes.json "best") is accepted at min_fps
        self.good_fps = float(good_fps) if good_fps and good_fps > min_fps else None
        self.trusted = trusted
        self.ok_best: _Candidate | None = None
        self.first_frame_s, self.retry_frame_s = first_frame_s, retry_frame_s
        self.clock, self.sleep = clock, sleep
        self.min_fps, self.measure_s = float(min_fps), float(measure_s)
        self.tried: list[str] = []
        self.opened = False
        self.first = True
        self.best: _Candidate | None = None  # fastest mode below min_fps
        self.done: set[tuple[str, Any, CaptureMode]] = set()

    @staticmethod
    def cancelled() -> bool:
        return _abort_opens.is_set()

    def _check(self) -> None:
        if self.cancelled():
            raise OpenCancelled(f"camera {self.label}: open cancelled (the cameras are being handed over)")

    def run(self, at: _Attempt, deadline: float | None = None) -> tuple[Any, Any, _Candidate] | None:
        """``(cap, frame, candidate)`` for the first mode of ``at`` fast enough, else None. A backend whose
        device does not open is left at its first mode (another mode cannot help)."""
        cv2 = self.cv2
        for mode in at.modes:
            if deadline is not None and self.clock() >= deadline:
                log.warning("camera %s: stopping the mode search (it took too long)", self.label)
                return None
            k = (at.backend, at.src, mode)
            if k in self.done:
                continue
            self._check()
            self.done.add(k)
            log.info("camera %s: trying %s index %s, mode %s", self.label, at.backend, at.src, mode.describe())
            cap = self._create(at.src, at.api, at.factory, mode)
            if not cap.isOpened():
                why = str(getattr(cap, "error", "") or "")
                with contextlib.suppress(Exception):
                    cap.release()
                self.tried.append(f"{at.backend} {at.src}: did not open" + (f" ({why})" if why else ""))
                log.warning(
                    "camera %s: %s could not open index %s%s", self.label, at.backend, at.src, f": {why}" if why else ""
                )
                return None
            self.opened = True
            if at.factory is None:
                got_fcc = apply_mode(cap, mode, cv2, self.label)
                if mode.fourcc == "MJPG" and got_fcc and got_fcc != mode.fourcc:
                    # v0.1.21: the driver kept another format (JoyCam: YUY2). Retry on a fresh capture with the size
                    # first; if it still is not MJPG, skip this mode instead of measuring YUY2 under an MJPG name.
                    with contextlib.suppress(Exception):
                        cap.release()
                    cap = self._create(at.src, at.api, at.factory, mode)
                    if not cap.isOpened() or apply_mode_size_first(cap, mode, cv2, self.label) not in ("", "MJPG"):
                        with contextlib.suppress(Exception):
                            cap.release()
                        self.tried.append(f"{at.backend} {mode.describe()}: driver kept {got_fcc}")
                        log.info(
                            "camera %s: %s index %s will not send MJPG at %s (driver keeps %s); next mode",
                            self.label,
                            at.backend,
                            at.src,
                            mode.describe(),
                            got_fcc,
                        )
                        continue
            else:  # the backend chose a native format itself: that is the mode to remember
                native = getattr(cap, "mode", None)
                mode = native if isinstance(native, CaptureMode) else mode
            frame = read_first(cap, self.first_frame_s if self.first else self.retry_frame_s, self.clock, self.sleep)
            self.first = False
            if frame is None:
                self.tried.append(f"{at.backend} {mode.describe()}: no frames")
                log.info(
                    "%s", attempt_line(self.label, at.backend, at.src, mode, read_back(cap, cv2), 0.0, "no frames")
                )
                log.warning(
                    "camera %s: %s opened index %s but sent no frame in mode %s",
                    self.label,
                    at.backend,
                    at.src,
                    mode.describe(),
                )
                with contextlib.suppress(Exception):
                    cap.release()
                continue
            fps, frame = measure_fps(cap, frame, self.measure_s, self.clock, self.sleep, self.cancelled)
            rb = read_back(cap, cv2)
            cand = _Candidate(at.backend, at.api, at.src, mode, fps, at.factory)
            slow_for_tracking = (
                fps is not None
                and self.good_fps is not None
                and self.min_fps <= fps < self.good_fps
                and not (self.trusted is not None and mode == self.trusted)
            )
            if slow_for_tracking:
                assert fps is not None and self.good_fps is not None
                log.info(
                    "%s",
                    attempt_line(
                        self.label,
                        at.backend,
                        at.src,
                        mode,
                        rb,
                        fps,
                        f"works, but below {self.good_fps:g} fps for smooth tracking; trying smaller modes",
                    ),
                )
                self.tried.append(f"{at.backend} {mode.describe()}: {fps:.1f} fps")
                if self.ok_best is None or (self.ok_best.fps or 0.0) < fps:
                    self.ok_best = cand
                with contextlib.suppress(Exception):
                    cap.release()
                self._check()
                continue
            if fps is None or fps >= self.min_fps:
                log.info("%s", attempt_line(self.label, at.backend, at.src, mode, rb, fps, "ok"))
                log.info(
                    "camera %s: %s index %s delivers frames in mode %s", self.label, at.backend, at.src, mode.describe()
                )
                return cap, frame, cand
            log.info(
                "%s",
                attempt_line(self.label, at.backend, at.src, mode, rb, fps, f"too slow (minimum {self.min_fps:g})"),
            )
            self.tried.append(f"{at.backend} {mode.describe()}: too slow ({fps:.1f} fps)")
            with contextlib.suppress(Exception):
                cap.release()
            if self.best is None or (self.best.fps or 0.0) < fps:
                self.best = cand
            self._check()
        return None

    def _create(self, src: int | str, api: Any, factory: Callable[[CaptureMode], Any] | None, mode: CaptureMode) -> Any:
        """The capture for one attempt: ``factory(mode)`` (WinRT), else ``cv2.VideoCapture``. A factory that
        raises gives a capture that is not open and says why (``error``)."""
        if factory is not None:
            try:
                return factory(mode)
            except Exception as exc:
                return _NotOpened(str(exc))
        cv2 = self.cv2
        return cv2.VideoCapture(src) if api is None else cv2.VideoCapture(src, api)

    def reopen(self, cand: _Candidate) -> tuple[Any, Any] | None:
        """Open ``cand`` again (the fastest of the slow modes); ``(cap, frame)`` or None."""
        self._check()
        cv2 = self.cv2
        cap = self._create(cand.src, cand.api, cand.factory, cand.mode)
        if not cap.isOpened():
            with contextlib.suppress(Exception):
                cap.release()
            return None
        if cand.factory is None:
            apply_mode(cap, cand.mode, cv2, self.label)
        frame = read_first(cap, max(self.retry_frame_s, 2.0), self.clock, self.sleep)
        if frame is None:
            with contextlib.suppress(Exception):
                cap.release()
            return None
        return cap, frame


class _NotOpened:
    """A capture that did not open (a backend factory raised): ``error`` says why."""

    def __init__(self, error: str) -> None:
        self.error = error

    def isOpened(self) -> bool:
        return False

    def read(self) -> tuple[bool, Any]:
        return False, None

    def release(self) -> None:
        pass


def _try_open(
    attempts: list[_Attempt],
    label: str,
    cv2: Any,
    first_frame_s: float,
    retry_frame_s: float,
    clock: Callable[[], float],
    sleep: Callable[[float], Any],
    min_fps: float = 0.0,
    measure_s: float = 0.0,
) -> tuple[Any, Any, CaptureMode | None, str, list[str], bool]:
    """Try every backend and mode in order (see :class:`_Search`); ``(cap, frame, mode, backend, tried,
    opened_any)``, ``cap`` None when nothing delivered frames fast enough."""
    search = _Search(label, cv2, first_frame_s, retry_frame_s, clock, sleep, min_fps, measure_s)
    for at in attempts:
        got = search.run(at)
        if got is not None:
            cap, frame, cand = got
            return cap, frame, cand.mode, cand.backend, search.tried, True
    return None, None, None, "", search.tried, search.opened


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


def sweep_modes(width: int, height: int, fps: int, virtual: bool = False) -> list[CaptureMode]:
    """The slow-camera sweep (v0.1.16): MJPG, YUY2, NV12 and the driver's format at the configured size, then
    640x480, then 320x240, each asking for ``fps`` (or SWEEP_FPS) with ``CAP_PROP_FPS``; v0.1.19: then 1280x720
    in MJPG and the driver's format (downscaled to the configured size when it is used). A virtual camera only
    in its own format (and never in HD)."""
    f = int(fps or 0) or SWEEP_FPS
    sizes: list[tuple[int, int]] = []
    for size in ((int(width), int(height)), *SWEEP_SIZES):
        if size[0] and size[1] and size not in sizes:
            sizes.append(size)
    out: list[CaptureMode] = []
    for w, h in sizes:
        for fcc in ("",) if virtual else SWEEP_FOURCCS:
            m = CaptureMode(w, h, f, fcc)
            if m not in out:
                out.append(m)
    if not virtual:
        hw, hh = SWEEP_HD
        for fcc in SWEEP_HD_FOURCCS:
            m = CaptureMode(hw, hh, f, fcc)
            if m not in out:
                out.append(m)
    return out


def free_broadcast(devs: list[CameraDevice], src: int | str) -> CameraDevice | None:
    """The one NVIDIA Broadcast virtual camera that is listed, is not ``src`` and that nobody (this process or
    the running app) holds; None when there is none or more than one."""
    held = held_cameras()
    free = [
        d
        for d in devs
        if d.virtual
        and is_broadcast_camera(d.name)
        and d.index is not None
        and d.index != src
        and not d.in_use
        and str(d.index) not in held
    ]
    return free[0] if len(free) == 1 else None


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
    min_fps: float | None = None,
    measure_s: float | None = None,
    sweep_budget_s: float | None = None,
    backend: str | None = None,
    good_fps: float | None = None,
) -> OpenedCamera:
    """Open ``src`` (an index or path from :func:`resolve_device`) and prove it works: read a frame, then
    measure the frame rate for ``measure_s`` (default MEASURE_S; 0 = do not measure).

    On Windows (an integer index), in order, every attempt logged (one :func:`attempt_line` each):

    1. DirectShow at that index (:func:`capture_modes`): the configured FOURCC (MJPG) at the configured size,
       YUY2 and NV12 at that size, no FOURCC, then nothing forced at all; with ``low_res`` (default: when this
       process already holds another camera) 320x240 and 320x240 at 15 fps;
    2. Media Foundation at the index *its own* list gives this camera (:func:`msmf_index`): the configured
       size, then nothing forced;
    3. v0.1.19, when DirectShow and Media Foundation failed or stayed below ``min_fps``: WinRT MediaCapture
       (:mod:`away.multicam.winrt_capture`, the Windows Camera app's API) by the camera's name, in the native
       format closest to the configured one (1280x720 allowed, downscaled);
    4. only when every mode so far delivered frames but below ``min_fps`` (default MIN_CAMERA_FPS): the sweep
       (:func:`sweep_modes`) on DirectShow, then on Media Foundation, for at most ``sweep_budget_s``; the
       fastest mode wins and is remembered in camera-modes.json (with its fps);
    5. ``virtual_fallback``: exactly one free NVIDIA Broadcast virtual camera when the camera did not open,
       sent no frames, or stayed below ``min_fps`` in every mode (Broadcast must reach ``min_fps`` itself;
       the slow camera is closed first and not kept open: Broadcast usually shows that same camera).

    A mode that reaches ``min_fps`` is used at once. A camera that stays too slow is opened anyway in its
    fastest mode with ``slow=True`` and hints (callers report it as FAILED: too slow). The backend and mode that
    worked are kept and tried first on later opens in this session. Virtual cameras are opened in their own
    format (no FOURCC). Raises :class:`CameraOpenError` saying whether the camera did not open or opened without
    frames, with hints (USB bandwidth, NVIDIA Broadcast) that are also logged.

    ``backend`` (hands.camera_backend; None = :func:`set_camera_backend`): ``auto`` is the order above;
    ``dshow`` / ``msmf`` / ``winrt`` use that backend alone (virtual cameras always use DirectShow). A camera
    that WinRT opened is remembered as backend ``WinRT`` with its native format, and opened that way first.
    """
    if cv2 is None:  # pragma: no cover - real OpenCV
        cv2 = load_cv2()
    global _opening
    with _lock:
        _opening += 1
        # v0.1.29: record the hand-over generation before queuing on the gate. If a
        # hand-over finishes while this open is queued, the generation changes and the
        # open is stale: it must give up instead of opening after the hand-over.
        queued_gen = _handover_gen
    try:
        # v0.1.29: serialize whole opens. Two threads inside _open_camera raced in native
        # DirectShow/MediaFoundation/WinRT code (native access violations in the log); the
        # _opening counter alone never stopped that. No re-entrancy: _open_camera never
        # calls open_camera, and wait_devices_free never runs while the gate is held.
        with _open_gate:
            with _lock:
                if _handover_gen != queued_gen:
                    raise OpenCancelled("camera open cancelled (a hand-over finished while the open was queued)")
            return _open_camera(
                src,
                width,
                height,
                fps,
                fourcc,
                cv2=cv2,
                devices=devices,
                label=label,
                low_res=low_res,
                virtual_fallback=virtual_fallback,
                first_frame_s=first_frame_s,
                retry_frame_s=retry_frame_s,
                clock=clock,
                sleep=sleep,
                msmf_lister=msmf_lister,
                min_fps=MIN_CAMERA_FPS if min_fps is None else float(min_fps),
                measure_s=MEASURE_S if measure_s is None else float(measure_s),
                sweep_budget_s=SWEEP_BUDGET_S if sweep_budget_s is None else float(sweep_budget_s),
                backend=backend,
                good_fps=TRACKING_FPS if good_fps is None else float(good_fps),
            )
    finally:
        with _lock:
            _opening -= 1


def _open_camera(
    src: int | str,
    width: int,
    height: int,
    fps: int,
    fourcc: str,
    *,
    cv2: Any,
    devices: list[CameraDevice] | None,
    label: str | None,
    low_res: bool | None,
    virtual_fallback: bool,
    first_frame_s: float,
    retry_frame_s: float,
    clock: Callable[[], float],
    sleep: Callable[[float], Any],
    msmf_lister: Callable[[], list[tuple[str, str]] | None] | None,
    min_fps: float,
    measure_s: float,
    sweep_budget_s: float,
    backend: str | None = None,
    good_fps: float = 0.0,
) -> OpenedCamera:
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
    with _lock:
        known_slow = _slow_known.get(key)
    windows = _windows() and isinstance(src, int)
    allowed = backends_for(backend, windows)
    if windows and virtual and "DirectShow" not in allowed:
        allowed = ["DirectShow", *allowed]  # virtual cameras (NVIDIA Broadcast) are DirectShow filters
    if kept is not None and kept[0] not in allowed and kept[0] != "default":
        kept = None  # remembered for a backend that hands.camera_backend now rules out
    trusted = kept[1] if kept is not None and saved_mode_info().get(key, {}).get("best") else None
    search = _Search(repr(name), cv2, first_frame_s, retry_frame_s, clock, sleep, min_fps, measure_s, good_fps, trusted)
    mf_index: list[int | None] = []  # computed once, when Media Foundation is first needed
    winrt_tried: list[bool] = []

    def mf() -> int | None:
        if not mf_index:
            log.warning("camera %r: trying Media Foundation at its own index for this camera", name)
            assert isinstance(src, int)
            mf_index.append(msmf_index(src, me, devs, repr(name), msmf_lister))
        return mf_index[0]

    def found(got: tuple[Any, Any, _Candidate], first: tuple[str, CaptureMode], best: bool = False) -> OpenedCamera:
        cap, frame, cand = got
        _remember(key, name, cand.backend, cand.mode, first, cand.fps, min_fps, best=best)
        if cand.factory is None:
            cap, frame = scaled(cap, frame, cand.mode, int(width), int(height))
        return OpenedCamera(HeldCapture(cap, str(src)), frame, cand.mode, src, name, backend=cand.backend, fps=cand.fps)

    def attempt(at: _Attempt, deadline: float | None = None) -> OpenedCamera | None:
        if kept is not None and kept[0] == at.backend:  # the mode that worked last (also one the sweep found)
            if kept[1] in at.modes:
                at.modes.remove(kept[1])
            at.modes.insert(0, kept[1])
        got = search.run(at, deadline)
        # "first" = the configured way (DirectShow on Windows): anything else is remembered for the next start
        return None if got is None else found(got, (allowed[0] if windows else "default", modes[0]))

    def winrt_attempt(mode_list: list[CaptureMode]) -> _Attempt | None:
        """The WinRT attempt (v0.1.19), or None when it cannot be made here (logged once per open)."""
        if winrt_tried:
            return None
        winrt_tried.append(True)
        wname = winrt_name(me, label)
        if wname is None:
            search.tried.append("WinRT: no camera name to look up")
            log.warning("camera %r: WinRT needs the camera's name; skipping it", name)
            return None
        ok, why = winrt_factory_available()
        if not ok:
            search.tried.append(f"WinRT: {why}")
            log.warning("camera %r: skipping WinRT: %s", name, why)
            return None
        occurrence = 0
        if me is not None:
            same = [d for d in devs if d.name.lower() == me.name.lower()]
            occurrence = next((j for j, d in enumerate(same) if d is me), 0)
        factory = winrt_factory(wname, int(width), int(height), occurrence)
        return _Attempt(WINRT, None, src, list(mode_list), factory)

    def by_backend(b: str, mode_list: list[CaptureMode]) -> _Attempt | None:
        if not windows:
            return _Attempt("default", None, src, list(mode_list))
        if b == "Media Foundation":
            mfi = mf()
            return None if mfi is None else _Attempt("Media Foundation", cv2.CAP_MSMF, mfi, list(mode_list))
        if b == WINRT:
            return winrt_attempt(mode_list)
        return _Attempt("DirectShow", cv2.CAP_DSHOW, src, list(mode_list))

    if known_slow is not None and kept is not None:
        # a full search already found nothing fast enough in this session: check the remembered (fastest) mode
        # again only; still slow -> straight to the fallback, without another long sweep at every (re)try
        b, m = kept
        at = by_backend(b, [m])
        got = None if at is None else attempt(at)
        if got is not None:
            return got
        if search.best is None:  # it did not even stream any more: search the usual way
            known_slow = None
            with _lock:
                _slow_known.pop(key, None)
            winrt_tried.clear()
    if known_slow is not None:
        pass
    elif not windows:
        got = attempt(_Attempt("default", None, src, list(modes)))
        if got is not None:
            return got
    else:
        assert isinstance(src, int)
        mf_modes = [CaptureMode(int(width), int(height), int(fps or 0), ""), DEVICE_DEFAULT]
        usual = {"DirectShow": list(modes), "Media Foundation": mf_modes}
        order: list[str] = [b for b in ("DirectShow", "Media Foundation") if b in allowed]
        if kept is not None and kept[0] == "Media Foundation":
            order.reverse()
        if WINRT in allowed and not virtual:
            # after DirectShow / Media Foundation (failed or too slow), before the NVIDIA Broadcast fallback; first
            # when it is what worked last (camera-modes.json) or the only backend allowed
            if kept is not None and kept[0] == WINRT:
                order.insert(0, WINRT)
            else:
                order.append(WINRT)
        for b in order:
            if b == WINRT:
                at = winrt_attempt([CaptureMode(int(width), int(height), int(fps or 0), "")])
            else:
                at = by_backend(b, usual[b])
            got = None if at is None else attempt(at)
            if got is not None:
                return got

    if search.ok_best is not None:
        # v0.1.21: nothing reached TRACKING_FPS; use the fastest mode that works (e.g. 720p YUY2 at 10 fps when no
        # smaller mode is faster) and remember it as this camera's best, so the next start does not search again
        okb = search.ok_best
        log.warning(
            "camera %r: no mode reached %g fps; using the fastest that works: %s %s at %.1f fps",
            name,
            good_fps,
            okb.backend,
            okb.mode.describe(),
            okb.fps or 0.0,
        )
        reopened = search.reopen(okb)
        if reopened is not None:
            return found((reopened[0], reopened[1], okb), (allowed[0] if windows else "default", modes[0]), best=True)

    # every mode so far either failed or was too slow
    sweep_backends = [b for b in ("DirectShow", "Media Foundation") if b in allowed] if windows else ["default"]
    if search.best is not None and known_slow is None and sweep_backends:
        log.warning(
            "camera %r: below %g fps in every usual mode (best %.1f fps, %s %s); trying every format and size",
            name,
            min_fps,
            search.best.fps or 0.0,
            search.best.backend,
            search.best.mode.describe(),
        )
        deadline = clock() + sweep_budget_s
        sweep = sweep_modes(width, height, fps, virtual)
        sizes = list(dict.fromkeys((m.width, m.height) for m in sweep))
        got = None
        for size in sizes:  # per size: DirectShow's formats, then Media Foundation's (its own device list)
            group = [m for m in sweep if (m.width, m.height) == size]
            if clock() >= deadline:
                break
            for b in sweep_backends:
                if clock() >= deadline:
                    break
                at = by_backend(b, group)
                got = None if at is None else attempt(at, deadline)
                if got is not None:
                    return got

    if windows and not virtual and WINRT in allowed and "Media Foundation" in allowed:
        # v0.1.23: ask the camera which formats it really has (WinRT lists them) and try the closest ones through
        # Media Foundation; 640x480 MJPG may simply not exist on a built-in camera ("Element not found")
        native = native_modes(winrt_name(me, label), int(width), int(height), int(fps or 0))
        if native:
            log.warning("camera %r: trying the formats it reports: %s", name, ", ".join(m.describe() for m in native))
            at = by_backend("Media Foundation", native)
            got = None if at is None else attempt(at)
            if got is not None:
                return got

    hints: list[str] = []
    broadcast = [d for d in devs if d.virtual and is_broadcast_camera(d.name) and d.index is not None]
    alt = free_broadcast(devs, src) if not virtual and isinstance(src, int) else None
    if alt is not None and not broadcast_fallback_allowed(name):
        log.warning(
            "camera %r: not opening NVIDIA Broadcast's camera as a fallback (it would take the built-in camera)", name
        )
        alt = None

    if search.best is not None:  # it streams, but too slowly in every mode
        best = search.best
        why = f"{name} is too slow: {best.fps or 0.0:.1f} fps at best (minimum {min_fps:g})"
        log.warning(
            "camera %r: too slow in every mode: best %.1f fps (%s %s), minimum %g (hands.min_camera_fps)",
            name,
            best.fps or 0.0,
            best.backend,
            best.mode.describe(),
            min_fps,
        )
        remember_mode(key, best.backend, best.mode, fps=best.fps, slow=True)
        with _lock:
            _slow_known[key] = float(best.fps or 0.0)
        if alt is not None and virtual_fallback:
            got_alt = _open_broadcast(alt, name, why, width, height, fps, cv2, search, min_fps, measure_s)
            if got_alt is not None:
                return got_alt
            hints.append(f"NVIDIA Broadcast's camera did not reach {min_fps:g} fps either")
        elif alt is not None:
            hints.append(
                f"{alt.name} is free and may stream faster; turn on hands.virtual_fallback (Settings > Cameras) "
                "to use it instead"
            )
        reopened = search.reopen(best)
        if reopened is None:
            raise CameraOpenError(f"camera {name!r}: {why}, and it could not be opened again", opened=True, hints=hints)
        cap, frame = reopened
        for h in hints:
            log.warning("camera %r: %s", name, h)
        return OpenedCamera(
            HeldCapture(cap, str(src)),
            frame,
            best.mode,
            src,
            name,
            backend=best.backend,
            fps=best.fps,
            slow=True,
            hints=[why, *hints],
        )

    if not virtual and broadcast:
        hints.append(broadcast_hint(name))
    if others or low_res:
        hints.append(BANDWIDTH_HINT)
    for h in hints:
        log.warning("camera %r: %s", name, h)
    detail = "; ".join(search.tried) if search.tried else "nothing to try"
    opened = search.opened
    if opened:
        msg = f"camera {name!r} opened but sent no frames (tried {detail})"
    else:
        msg = f"cannot open camera {name!r} (busy, no device, or permission denied; tried {detail})"
    if virtual_fallback and alt is not None:
        why = f"{name} could not be opened directly ({'no frames' if opened else 'open failed'})"
        got_alt = _open_broadcast(alt, name, why, width, height, fps, cv2, search, min_fps, measure_s)
        if got_alt is not None:
            return got_alt
    raise CameraOpenError(msg + (". " + "; ".join(hints) if hints else ""), opened=opened, hints=hints)


def _open_broadcast(
    alt: CameraDevice,
    name: str,
    why: str,
    width: int,
    height: int,
    fps: int,
    cv2: Any,
    parent: _Search,
    min_fps: float,
    measure_s: float,
) -> OpenedCamera | None:
    """Open NVIDIA Broadcast's camera instead of ``name`` (in its own format); it must reach ``min_fps``."""
    assert alt.index is not None
    log.warning(
        "camera %r: %s; opening the virtual camera %r instead (hands.virtual_fallback); NVIDIA "
        "Broadcast's camera shows the same picture",
        name,
        why,
        alt.name,
    )
    alt_modes = capture_modes(width, height, fps, "", False, other_fourccs=False)
    api = cv2.CAP_DSHOW if _windows() else None
    search = _Search(
        repr(alt.name), cv2, parent.first_frame_s, parent.retry_frame_s, parent.clock, parent.sleep, min_fps, measure_s
    )
    got = search.run(_Attempt("DirectShow" if api is not None else "default", api, alt.index, alt_modes))
    if got is not None:
        acap, aframe, cand = got
        log.info(
            "camera %r: using %r (virtual fallback), mode %s%s",
            name,
            alt.name,
            cand.mode.describe(),
            "" if cand.fps is None else f", {cand.fps:.1f} fps measured",
        )
        return OpenedCamera(
            HeldCapture(acap, str(alt.index)),
            aframe,
            cand.mode,
            alt.index,
            alt.name,
            why,
            backend=cand.backend,
            fps=cand.fps,
        )
    log.warning("camera %r: the virtual fallback %r did not work either (%s)", name, alt.name, "; ".join(search.tried))
    return None


def _remember(
    key: str,
    name: str,
    backend: str,
    mode: CaptureMode,
    first: tuple[str, CaptureMode],
    fps: float | None = None,
    min_fps: float = 0.0,
    best: bool = False,
) -> None:
    with _lock:
        before = _session_modes.get(key)
        _session_modes[key] = (backend, mode)
        _slow_known.pop(key, None)  # fast enough now
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
    if (backend, mode) != first or best:
        # only works in a fallback mode (or this is the best of a full search): open it that way next start
        remember_mode(key, backend, mode, fps=fps, best=best)
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
    for key, info in saved_mode_info().items():
        if info.get("slow"):
            with _lock:
                _slow_known.setdefault(key, float(info.get("fps") or 0.0))


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


def saved_mode_info() -> dict[str, dict[str, Any]]:
    """``{camera name: {"measured_fps": float | None, "slow": bool}}`` from the mode file (v0.1.16)."""
    if _mode_file is None:
        return {}
    import json

    try:
        data = json.loads(_mode_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict):
        return out
    for key, v in data.items():
        if not isinstance(v, dict):
            continue
        mf = v.get("measured_fps")
        out[str(key)] = {
            "fps": float(mf) if isinstance(mf, (int, float)) and not isinstance(mf, bool) else None,
            "slow": v.get("slow") is True,
            "best": v.get("best") is True,  # v0.1.21: the fastest mode of a full search (no faster one exists)
        }
    return out


def _write_modes(modes: dict[str, tuple[str, CaptureMode]], info: dict[str, dict[str, Any]] | None = None) -> None:
    if _mode_file is None:
        return
    import json

    info = info if info is not None else saved_mode_info()
    data: dict[str, dict[str, Any]] = {}
    for k, (b, m) in sorted(modes.items()):
        row: dict[str, Any] = {"backend": b, "width": m.width, "height": m.height, "fps": m.fps, "fourcc": m.fourcc}
        extra = info.get(k, {})
        if extra.get("fps") is not None:
            row["measured_fps"] = round(float(extra["fps"]), 2)
        if extra.get("slow"):
            row["slow"] = True
        if extra.get("best"):
            row["best"] = True
        data[k] = row
    with contextlib.suppress(OSError):
        _mode_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = _mode_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, _mode_file)


def remember_mode(
    key: str, backend: str, mode: CaptureMode, fps: float | None = None, slow: bool = False, best: bool = False
) -> None:
    """Save the mode a camera works in (when the store is on) and use it for the rest of this session. ``fps``:
    the frame rate measured in it; ``slow``: it is the fastest mode of a full search and still below the
    minimum (the next start checks only this mode before falling back)."""
    with _lock:
        _session_modes[key] = (backend, mode)
    if _mode_file is None:
        return
    modes = saved_modes()
    info = saved_mode_info()
    new_info = {"fps": fps, "slow": bool(slow), "best": bool(best)}
    old_info = info.get(key, {"fps": None, "slow": False, "best": False})
    if modes.get(key) != (backend, mode) or old_info != new_info:
        modes[key] = (backend, mode)
        info[key] = new_info
        _write_modes(modes, info)
        log.info(
            "camera %r: remembered %s %s for the next start%s%s",
            key,
            backend,
            mode.describe(),
            "" if fps is None else f" ({fps:.1f} fps measured)",
            " - still too slow" if slow else "",
        )


def forget_mode(key: str) -> None:
    modes = saved_modes()
    if key in modes:
        del modes[key]
        info = saved_mode_info()
        info.pop(key, None)
        _write_modes(modes, info)
    with _lock:
        _slow_known.pop(key, None)


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


def open_capture(
    spec: CameraSpec,
    virtual_fallback: bool = True,
    cv2: Any = None,
    low_res: bool | None = None,
    min_fps: float | None = None,
    skip_held: bool = False,
    devices: list[CameraDevice] | None = None,
) -> Any:
    """Open one camera of a multi-camera set (see :func:`open_camera`); returns a reader whose first ``read()``
    gives the frame that proved the mode works. A secondary camera may fall back to 320x240 (``low_res``
    overrides that: True also for the primary, False never). ``min_fps``: hands.min_camera_fps.

    ``skip_held`` (the background re-probe of failed cameras, v0.1.19): a camera this process already holds
    open (the one that is working) is never opened again; that is an error instead (it means the failed
    camera's name now resolves to the working camera)."""
    if cv2 is None:  # pragma: no cover - real OpenCV
        cv2 = load_cv2()
    devs = camera_devices() if devices is None else list(devices)
    src = resolve_device(spec.device, devs if devs else None, hint=spec.index_hint)  # by name, at every open
    if skip_held and str(src) in held_cameras():
        raise CameraOpenError(
            f"camera {spec.device!r} resolves to camera {src}, which is already open and working; "
            "not reopening it during a retry"
        )
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
        min_fps=min_fps,
    )
    fc = FirstFrameCapture(got.cap, got.frame, got.mode, got.backend, got.name)
    fc.measured_fps, fc.slow, fc.hints = got.fps, got.slow, list(got.hints)
    if got.fallback:
        fc.fallback = got.fallback
        fc.opened_index = got.src if isinstance(got.src, int) else None
    return fc


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
        self.measured_fps: float | None = None  # while choosing the mode
        self.slow = False  # below the minimum in every mode (the fastest one is open)
        self.fallback = ""  # why another camera (NVIDIA Broadcast) streams instead of the configured one
        self.opened_index: int | None = None
        self.hints: list[str] = []

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
        self.measured_fps: float | None = None  # measured by the opener while choosing the mode
        self.slow = False  # the opener found it below the minimum in every mode
        self.fallback = ""  # why another camera streams in its place ("" = the configured camera itself)
        self.opened_name = ""  # that other camera (NVIDIA Broadcast)
        self.opened_index: int | None = None
        self.hints: list[str] = []

    def too_slow(self, min_fps: float, window_s: float = 2.0) -> bool:
        """Below ``min_fps``: the opener said so, or the rate measured over at least ``window_s`` of streaming
        is lower (v0.1.16)."""
        if min_fps <= 0 or self.error:
            return False
        if self.first_t is not None and self.last_t is not None and self.last_t - self.first_t >= window_s:
            return self.fps() < min_fps
        return bool(self.slow)

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
        mf = getattr(cap, "measured_fps", None)
        self.measured_fps = float(mf) if isinstance(mf, (int, float)) and not isinstance(mf, bool) else None
        self.slow = getattr(cap, "slow", False) is True
        fb = getattr(cap, "fallback", "")
        if isinstance(fb, str) and fb:
            self.fallback = fb
            nm = getattr(cap, "name", "")
            self.opened_name = nm if isinstance(nm, str) else ""
            oi = getattr(cap, "opened_index", None)
            self.opened_index = oi if isinstance(oi, int) else None
        hints = getattr(cap, "hints", None)
        self.hints = [str(h) for h in hints] if isinstance(hints, list) else []
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


def camera_opener(
    virtual_fallback: bool = True, low_res: bool | None = None, min_fps: float | None = None, skip_held: bool = False
) -> Callable[[CameraSpec], Any]:
    """The opener every multi-camera user shares: :func:`open_capture` with the NVIDIA Broadcast fallback
    setting, a secondary allowed down to 320x240, the minimum frame rate, and the backend/mode kept for the
    session (and across starts with :func:`use_mode_store`). ``skip_held``: see :func:`open_capture`."""
    import functools

    kw: dict[str, Any] = {"virtual_fallback": virtual_fallback}
    if low_res is not None:
        kw["low_res"] = low_res
    if min_fps is not None:
        kw["min_fps"] = min_fps
    if skip_held:
        kw["skip_held"] = True
    return functools.partial(open_capture, **kw)


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
    min_fps: float | None = None,
    skip_held: bool = False,
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
    op = opener or camera_opener(virtual_fallback, min_fps=min_fps, skip_held=skip_held)
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
