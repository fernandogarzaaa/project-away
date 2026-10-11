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


def open_video(src: int | str, cv2: Any = None) -> Any:
    """``cv2.VideoCapture`` for ``src``. On Windows an integer index is opened with DirectShow, the backend whose
    numbering :func:`list_cameras` reports (and that lists virtual cameras); Media Foundation is tried only when
    DirectShow cannot open it, and that is logged (MSMF numbers cameras its own way). Everything else uses
    OpenCV's default backend."""
    if cv2 is None:  # pragma: no cover - real OpenCV
        import cv2
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


def open_capture(spec: CameraSpec) -> Any:  # pragma: no cover - opens a device
    import cv2

    src = resolve_device(spec.device, hint=spec.index_hint)  # by name, at every open (hot-plug safe)
    cap = open_video(src, cv2)
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
