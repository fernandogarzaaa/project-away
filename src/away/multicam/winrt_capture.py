"""WinRT MediaCapture capture backend (Windows; v0.1.19, untested on real hardware).

Why: on an Acer Predator Helios 300 the built-in camera ("HD User Facing") never opens through OpenCV's
DirectShow or Media Foundation backends (or delivers 1.0 fps), yet the Windows Camera app shows it smoothly.
The Camera app uses WinRT ``MediaCapture`` through the Windows Camera Frame Server, so this backend uses the
same API through the pywinrt packages (``pip install "project-away[winrt]"``; bundled in the Windows installer).

* The camera is found by name among ``MediaFrameSourceGroup.find_all_async()`` (the n-th group of that name for
  the n-th camera of the same name).
* ``MediaCapture`` is initialised with ``SharingMode.EXCLUSIVE_CONTROL`` (so the frame format can be chosen),
  falling back to ``SHARED_READ_ONLY`` (the current format, when another app controls the camera).
* The native format closest to the requested size and frame rate is chosen (:func:`choose_format`): 30 fps
  first, then the smallest size that covers the request, up to 1280x720 (downscaled to the configured size,
  keeping the aspect ratio).
* Frames come from a ``MediaFrameReader`` asked for BGRA8; each one is copied out of its ``SoftwareBitmap``
  into a BGR numpy array.
* Everything WinRT runs on the capture's own thread with its own asyncio event loop; :class:`WinrtCapture`
  exposes the ``read()`` / ``isOpened()`` / ``release()`` / ``get()`` subset of ``cv2.VideoCapture`` that the
  camera threads use, so the rest of the app does not know which backend delivers the frames.

pywinrt is imported lazily (:func:`load_winrt`), so this module imports anywhere; every WinRT call sits behind
the small :class:`WinrtApi` interface that the tests replace with a fake.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from away.multicam.capture import CaptureMode, fit_within, fourcc_code, resize_frame

log = logging.getLogger("away.cameras")

BACKEND = "WinRT"  # the backend name in logs, camera-modes.json and the Settings / test output
MAX_NATIVE = (1280, 720)  # the largest native format chosen (downscaled to the configured size)
PREFERRED_FPS = 30.0
DEFAULT_SIZE = (640, 480)  # the size asked for when the configuration says 0 (auto)
OPEN_TIMEOUT_S = 10.0  # initialising MediaCapture + starting the reader
READ_TIMEOUT_S = 1.0  # read() waits this long for a new frame
SHARING_MODES = ("exclusive", "shared")  # EXCLUSIVE_CONTROL, then SHARED_READ_ONLY
SUBTYPE_RANK = ("MJPG", "NV12", "YUY2")  # equal size and rate: MJPG first (USB bandwidth), like the other backends
INSTALL_HINT = 'the WinRT camera backend needs the pywinrt packages: pip install "project-away[winrt]"'


@dataclass(frozen=True)
class NativeFormat:
    """One format a camera's colour source offers (``MediaFrameSource.SupportedFormats``)."""

    width: int
    height: int
    fps: float
    subtype: str  # "MJPG", "YUY2", "NV12", ... (a GUID string for a format without a FOURCC)
    index: int = -1  # position in the source's supported formats (-1: not from a list)

    def describe(self) -> str:
        return f"{self.width}x{self.height}@{self.fps:g} {self.subtype or '?'}"

    def fourcc(self) -> str:
        s = (self.subtype or "").strip().upper()
        return s if len(s) == 4 and s.isascii() and s.isprintable() else ""

    def mode(self) -> CaptureMode:
        """As a :class:`CaptureMode` (what camera-modes.json keeps): native size, rounded rate, FOURCC."""
        return CaptureMode(int(self.width), int(self.height), round(self.fps), self.fourcc())


def choose_format(
    formats: Sequence[NativeFormat],
    width: int,
    height: int,
    fps: float = 0,
    fourcc: str = "",
    max_size: tuple[int, int] = MAX_NATIVE,
) -> NativeFormat | None:
    """The native format to use for a request of ``width`` x ``height`` at ``fps`` (0 = auto: 640x480 / 30):

    1. reaches the requested rate (30 fps when none is asked for), then the rate closest to it;
    2. covers the requested size (it is downscaled), the smallest such; else the largest smaller one;
    3. the requested FOURCC (a remembered native mode), then MJPG, NV12, YUY2, anything else.

    Formats larger than ``max_size`` (1280x720) are used only when nothing else is offered. None for an empty
    list."""
    if not formats:
        return None
    tw, th = (int(width), int(height)) if width and height else DEFAULT_SIZE
    tf = float(fps) if fps and fps > 0 else PREFERRED_FPS
    mw, mh = max_size
    small = [f for f in formats if f.width <= mw and f.height <= mh]
    pool = small or list(formats)
    want = (fourcc or "").strip().upper()

    def rank(f: NativeFormat) -> tuple[Any, ...]:
        reaches = f.fps >= tf - 0.5
        covers = f.width >= tw and f.height >= th
        area = f.width * f.height
        sub = f.fourcc()
        sub_rank = SUBTYPE_RANK.index(sub) if sub in SUBTYPE_RANK else len(SUBTYPE_RANK)
        return (
            0 if reaches else 1,
            round(abs(f.fps - tf), 1),
            0 if covers else 1,
            area if covers else -area,
            0 if want and sub == want else 1,
            sub_rank,
            f.index,
        )

    return min(pool, key=rank)


def bgra_to_bgr(data: Any, width: int, height: int, stride: int = 0) -> Any:
    """A BGRA8 buffer (``SoftwareBitmap.CopyToBuffer``) -> a contiguous ``height x width x 3`` BGR array.
    ``stride``: bytes per row when rows are padded (0 = ``width * 4``)."""
    import numpy as np

    row = int(stride) or int(width) * 4
    raw = np.frombuffer(memoryview(data), dtype=np.uint8, count=row * int(height))
    img = raw.reshape(int(height), row)[:, : int(width) * 4].reshape(int(height), int(width), 4)
    return np.ascontiguousarray(img[:, :, :3])


def copy_pixels(bmp: Any, width: int, height: int, streams: Any = None) -> Any:
    """A BGRA8 ``SoftwareBitmap``'s pixels as a numpy byte array. pywinrt 2+ accepts any writable Python buffer
    for an ``IBuffer`` argument (copied straight into numpy); a ``Windows.Storage.Streams.Buffer`` otherwise."""
    import numpy as np

    out = np.empty(int(width) * int(height) * 4, np.uint8)
    try:
        bmp.copy_to_buffer(out)
        return out
    except TypeError:
        if streams is None:
            raise
        buf = streams.Buffer(out.size)
        bmp.copy_to_buffer(buf)
        return np.frombuffer(bytes(memoryview(buf)), np.uint8, count=out.size)


class WinrtApi(Protocol):
    """The WinRT calls the backend makes (:class:`RealWinrtApi`; the tests use a fake). The coroutine methods
    run on the capture thread's asyncio loop."""

    async def find_group(self, name: str, occurrence: int = 0) -> Any | None: ...

    async def group_names(self) -> list[str]: ...

    async def open(self, group: Any, sharing: str) -> Any: ...

    def color_source(self, capture: Any) -> Any: ...

    def formats(self, source: Any) -> list[NativeFormat]: ...

    def current_format(self, source: Any) -> NativeFormat | None: ...

    async def set_format(self, source: Any, fmt: NativeFormat) -> None: ...

    async def start_reader(self, capture: Any, source: Any, on_frame: Callable[[Any], None]) -> Any: ...

    async def stop(self, capture: Any, reader: Any) -> None: ...


_winrt_state: dict[str, Any] = {}


def preload_system_msvcp140(platform: str | None = None, loader: Callable[[str], Any] | None = None) -> bool:
    """Load the system (or the app's bundled) ``msvcp140.dll`` before pywinrt does.

    The ``winrt-runtime`` wheel ships its own old ``winrt/msvcp140.dll`` (14.29, VS 2019) next to ``_winrt.pyd``.
    Windows keeps one ``msvcp140.dll`` per process, the first one loaded, and onnxruntime 1.2x+ needs 14.40 or
    newer: once pywinrt has loaded the old copy, importing onnxruntime crashes the process (access violation;
    seen in the v0.1.19 Windows installer build). Loading the system copy first makes pywinrt reuse it.
    Returns True when a copy was loaded here; never raises."""
    import sys

    if not (platform or sys.platform).startswith("win"):
        return False
    if loader is None:  # pragma: no cover - Windows only
        import ctypes

        loader = ctypes.WinDLL  # type: ignore[attr-defined]
    try:
        loader("msvcp140.dll")  # application dir (the frozen app's bundle), then System32
    except OSError as exc:
        log.debug("msvcp140.dll preload failed (%s); pywinrt uses its own copy", exc)
        return False
    return True


def load_winrt() -> dict[str, Any]:  # pragma: no cover - needs the Windows-only pywinrt wheels
    """The pywinrt namespaces this backend uses (imported on first use). Raises ImportError when missing."""
    if "mods" not in _winrt_state:
        preload_system_msvcp140()
        import winrt.windows.devices.enumeration
        import winrt.windows.foundation
        import winrt.windows.foundation.collections  # noqa: F401  (IVectorView / IMapView results)
        import winrt.windows.graphics.imaging as imaging
        import winrt.windows.media.capture as mc
        import winrt.windows.media.capture.frames as frames
        import winrt.windows.media.mediaproperties as props
        import winrt.windows.storage.streams as streams

        _winrt_state["mods"] = {"mc": mc, "frames": frames, "imaging": imaging, "props": props, "streams": streams}
    return _winrt_state["mods"]  # type: ignore[no-any-return]


def available() -> tuple[bool, str]:
    """``(True, "")`` when the WinRT backend can be used here, else ``(False, why)`` (not Windows, or the pywinrt
    packages are not installed). Cached."""
    if "available" not in _winrt_state:
        import sys

        if not sys.platform.startswith("win"):
            _winrt_state["available"] = (False, "the WinRT camera backend exists only on Windows")
        else:  # pragma: no cover - Windows only
            try:
                load_winrt()
                _winrt_state["available"] = (True, "")
            except Exception as exc:
                _winrt_state["available"] = (False, f"{INSTALL_HINT} ({exc})")
    return _winrt_state["available"]  # type: ignore[no-any-return]


def reset_availability() -> None:
    _winrt_state.pop("available", None)


class RealWinrtApi:  # pragma: no cover - needs Windows, pywinrt and a camera
    """The real WinRT calls (pywinrt 2.x/3.x API names)."""

    def __init__(self) -> None:
        self.m = load_winrt()

    async def _groups(self) -> list[Any]:
        frames = self.m["frames"]
        groups = await frames.MediaFrameSourceGroup.find_all_async()
        out = []
        for g in groups:
            kinds = [si.source_kind for si in g.source_infos]
            if frames.MediaFrameSourceKind.COLOR in kinds:
                out.append(g)
        return out

    async def group_names(self) -> list[str]:
        return [str(g.display_name) for g in await self._groups()]

    async def find_group(self, name: str, occurrence: int = 0) -> Any | None:
        groups = await self._groups()
        same = [g for g in groups if str(g.display_name).lower() == name.lower()]
        if not same:  # "HD User Facing" vs "HD User Facing Camera": a unique partial match is good enough
            same = [g for g in groups if name.lower() in str(g.display_name).lower()]
        if not same:
            return None
        return same[occurrence] if occurrence < len(same) else same[0]

    async def open(self, group: Any, sharing: str) -> Any:
        mc = self.m["mc"]
        settings = mc.MediaCaptureInitializationSettings()
        settings.source_group = group
        settings.sharing_mode = (
            mc.MediaCaptureSharingMode.EXCLUSIVE_CONTROL
            if sharing == "exclusive"
            else mc.MediaCaptureSharingMode.SHARED_READ_ONLY
        )
        settings.streaming_capture_mode = mc.StreamingCaptureMode.VIDEO
        settings.memory_preference = mc.MediaCaptureMemoryPreference.CPU  # SoftwareBitmap, not a GPU surface
        capture = mc.MediaCapture()
        try:
            await capture.initialize_with_settings_async(settings)
        except BaseException:
            with contextlib.suppress(Exception):
                capture.close()
            raise
        return capture

    def color_source(self, capture: Any) -> Any:
        mc, frames = self.m["mc"], self.m["frames"]
        sources = list(capture.frame_sources.values())
        colour = [s for s in sources if s.info.source_kind == frames.MediaFrameSourceKind.COLOR]
        for kind in (mc.MediaStreamType.VIDEO_RECORD, mc.MediaStreamType.VIDEO_PREVIEW):
            for s in colour:
                if s.info.media_stream_type == kind:
                    return s
        if colour:
            return colour[0]
        raise RuntimeError("the camera has no colour video source")

    @staticmethod
    def _native(fmt: Any, index: int) -> NativeFormat:
        vf = fmt.video_format
        rate = fmt.frame_rate
        fps = float(rate.numerator) / float(rate.denominator) if rate is not None and rate.denominator else 0.0
        return NativeFormat(int(vf.width), int(vf.height), round(fps, 2), str(fmt.subtype), index)

    def formats(self, source: Any) -> list[NativeFormat]:
        out = []
        for i, fmt in enumerate(source.supported_formats):
            with contextlib.suppress(Exception):
                out.append(self._native(fmt, i))
        return out

    def current_format(self, source: Any) -> NativeFormat | None:
        try:
            return self._native(source.current_format, -1)
        except Exception:
            return None

    async def set_format(self, source: Any, fmt: NativeFormat) -> None:
        supported = list(source.supported_formats)
        if not 0 <= fmt.index < len(supported):
            raise ValueError(f"format {fmt.describe()} is not in the camera's list")
        await source.set_format_async(supported[fmt.index])

    async def start_reader(self, capture: Any, source: Any, on_frame: Callable[[Any], None]) -> Any:
        frames, imaging, props, streams = self.m["frames"], self.m["imaging"], self.m["props"], self.m["streams"]
        try:  # the reader converts to BGRA8 itself (in the Frame Server pipeline)
            reader = await capture.create_frame_reader_with_subtype_async(source, props.MediaEncodingSubtypes.bgra8)
        except Exception:
            reader = await capture.create_frame_reader_async(source)
        reader.acquisition_mode = frames.MediaFrameReaderAcquisitionMode.REALTIME

        def arrived(sender: Any, _args: Any) -> None:
            ref = sender.try_acquire_latest_frame()
            if ref is None:
                return
            try:
                vmf = ref.video_media_frame
                bmp = vmf.software_bitmap if vmf is not None else None
                if bmp is None:
                    return
                if bmp.bitmap_pixel_format != imaging.BitmapPixelFormat.BGRA8:
                    bmp = imaging.SoftwareBitmap.convert(bmp, imaging.BitmapPixelFormat.BGRA8)
                w, h = int(bmp.pixel_width), int(bmp.pixel_height)
                on_frame(bgra_to_bgr(copy_pixels(bmp, w, h, streams), w, h))
                with contextlib.suppress(Exception):
                    bmp.close()
            except Exception as exc:
                log.debug("WinRT frame conversion failed: %s", exc)
            finally:
                with contextlib.suppress(Exception):
                    ref.close()

        token = reader.add_frame_arrived(arrived)
        status = await reader.start_async()
        if status != frames.MediaFrameReaderStartStatus.SUCCESS:
            with contextlib.suppress(Exception):
                reader.remove_frame_arrived(token)
                reader.close()
            name = getattr(status, "name", str(status))
            raise RuntimeError(f"the frame reader did not start ({name})")
        reader._away_token = token  # type: ignore[attr-defined]  # removed again in stop()
        return reader

    async def stop(self, capture: Any, reader: Any) -> None:
        if reader is not None:
            with contextlib.suppress(Exception):
                await reader.stop_async()
            with contextlib.suppress(Exception):
                reader.remove_frame_arrived(reader._away_token)
            with contextlib.suppress(Exception):
                reader.close()
        if capture is not None:
            with contextlib.suppress(Exception):
                capture.close()


def default_api() -> WinrtApi:  # pragma: no cover - Windows only
    return RealWinrtApi()


class WinrtCapture:
    """A camera read through WinRT MediaCapture with the ``cv2.VideoCapture`` subset the camera threads use.

    The constructor starts the capture thread and waits (at most ``open_timeout``) until frames flow or the open
    failed: ``isOpened()`` is then False and ``error`` says why. ``read()`` returns each new frame once (so
    measured frame rates are honest), waiting at most ``read_timeout`` for it. ``native`` is the native format in
    use, ``sharing`` the sharing mode that worked; ``mode`` (a :class:`CaptureMode` of the native format) is what
    camera-modes.json remembers."""

    def __init__(
        self,
        name: str,
        width: int = 0,
        height: int = 0,
        fps: float = 0,
        fourcc: str = "",
        *,
        out_size: tuple[int, int] | None = None,
        occurrence: int = 0,
        api: WinrtApi | None = None,
        open_timeout: float = OPEN_TIMEOUT_S,
        read_timeout: float = READ_TIMEOUT_S,
        sharing_modes: Sequence[str] = SHARING_MODES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.request = (int(width or 0), int(height or 0), float(fps or 0), str(fourcc or ""))
        self.out_size = out_size if out_size is not None else (int(width or 0), int(height or 0))
        self.occurrence = occurrence
        self.read_timeout = read_timeout
        self.sharing_modes = tuple(sharing_modes)
        self.clock = clock
        self.error = ""
        self.native: NativeFormat | None = None
        self.formats: list[NativeFormat] = []
        self.sharing = ""
        self.frame_size: tuple[int, int] = (0, 0)
        self.frames = 0
        self._api = api
        self._cond = threading.Condition()
        self._latest: Any = None
        self._seq = 0
        self._taken = 0
        self._opened = False
        self._done = threading.Event()  # set once the open finished (ok or not)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, name=f"winrt-{name}", daemon=True)
        self._thread.start()
        if not self._done.wait(open_timeout):
            self.error = f"WinRT did not open {name!r} within {open_timeout:g} s"
            self._stop.set()
        if not self._opened and not self.error:
            self.error = f"WinRT could not open {name!r}"

    # ------------------------------------------------------------------ the capture thread
    def _thread_main(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # anything that escaped _main: the open failed
            if not self.error:
                self.error = f"WinRT: {exc}"
        finally:
            self._opened = self._opened and not self._stop.is_set()
            self._done.set()
            with self._cond:
                self._cond.notify_all()

    async def _main(self) -> None:
        api = self._api or default_api()
        self._api = api
        group = await api.find_group(self.name, self.occurrence)
        if group is None:
            names = []
            with contextlib.suppress(Exception):
                names = await api.group_names()
            self.error = f"WinRT does not list a camera named {self.name!r} (it lists: {', '.join(names) or 'none'})"
            return
        capture = None
        errors: list[str] = []
        for sharing in self.sharing_modes:
            try:
                capture = await api.open(group, sharing)
                self.sharing = sharing
                break
            except Exception as exc:
                errors.append(f"{sharing}: {exc}")
                log.info("camera %r: WinRT %s open failed: %s", self.name, sharing, exc)
        if capture is None:
            self.error = "WinRT could not initialise the camera (" + "; ".join(errors) + ")"
            return
        reader = None
        try:
            source = api.color_source(capture)
            self.formats = list(api.formats(source))
            w, h, f, fcc = self.request
            chosen: NativeFormat | None = None
            if self.sharing == "exclusive":
                chosen = choose_format(self.formats, w, h, f, fcc)
                if chosen is not None:
                    try:
                        await api.set_format(source, chosen)
                    except Exception as exc:
                        log.warning(
                            "camera %r: WinRT could not set %s (%s); using its current format",
                            self.name,
                            chosen.describe(),
                            exc,
                        )
                        chosen = None
            self.native = chosen or api.current_format(source)
            reader = await api.start_reader(capture, source, self._on_frame)
            self._opened = True
            log.info(
                "camera %r: WinRT %s, native format %s, delivered at %s",
                self.name,
                "exclusive control" if self.sharing == "exclusive" else "shared read-only",
                self.native.describe() if self.native else "unknown",
                self._describe_out(),
            )
            self._done.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.05)
        except Exception as exc:
            if not self._opened:
                self.error = f"WinRT: {exc}"
            else:
                log.warning("camera %r: WinRT capture stopped: %s", self.name, exc)
        finally:
            await api.stop(capture, reader)

    def _describe_out(self) -> str:
        if self.native is None:
            return "its own size"
        ow, oh = fit_within(self.native.width, self.native.height, *self.out_size)
        return f"{ow}x{oh}" + (" (downscaled)" if (ow, oh) != (self.native.width, self.native.height) else "")

    def _on_frame(self, frame: Any) -> None:
        """Called from WinRT's thread for each frame (BGR): downscale to the configured size, keep the newest."""
        if frame is None:
            return
        h, w = frame.shape[:2]
        ow, oh = fit_within(w, h, *self.out_size)
        if (ow, oh) != (w, h):
            frame = resize_frame(frame, ow, oh)
        with self._cond:
            self._latest = frame
            self._seq += 1
            self.frames += 1
            self.frame_size = (frame.shape[1], frame.shape[0])
            self._cond.notify_all()

    # ------------------------------------------------------------------ cv2.VideoCapture subset
    def isOpened(self) -> bool:
        return bool(self._opened) and not self._stop.is_set()

    def read(self) -> tuple[bool, Any]:
        """``(True, frame)`` for a frame not returned before, waiting at most ``read_timeout``; else
        ``(False, None)``."""
        if not self.isOpened():
            return False, None
        deadline = self.clock() + self.read_timeout
        with self._cond:
            while self._seq == self._taken:
                left = deadline - self.clock()
                if left <= 0 or self._stop.is_set() or (self._done.is_set() and not self._opened):
                    return False, None
                self._cond.wait(min(left, 0.05))
            self._taken = self._seq
            return True, self._latest

    def get(self, prop: int) -> float:
        """Width (3), height (4) of the delivered frames, the native frame rate (5) and FOURCC (6); 0 else."""
        n = self.native
        if self.frame_size[0]:
            w, h = self.frame_size
        elif n is not None:
            w, h = fit_within(n.width, n.height, *self.out_size)
        else:
            w, h = 0, 0
        if prop == 3:
            return float(w)
        if prop == 4:
            return float(h)
        if prop == 5:
            return float(n.fps) if n else 0.0
        if prop == 6:
            return float(fourcc_code(n.fourcc())) if n and n.fourcc() else 0.0
        return 0.0

    def set(self, _prop: int, _value: Any) -> bool:
        return False  # the format is chosen when opening (choose_format)

    @property
    def mode(self) -> CaptureMode | None:
        return self.native.mode() if self.native is not None else None

    def release(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        self._opened = False


def list_formats(
    name: str, occurrence: int = 0, api: WinrtApi | None = None, timeout: float = OPEN_TIMEOUT_S
) -> tuple[list[NativeFormat], str]:
    """The native formats WinRT reports for camera ``name`` (``away cameras diagnose``): ``(formats, error)``.
    Opens the camera shared read-only first (it may be in use), then with exclusive control."""
    result: dict[str, Any] = {"formats": [], "error": ""}

    async def main() -> None:
        a = api or default_api()
        group = await a.find_group(name, occurrence)
        if group is None:
            result["error"] = f"WinRT does not list a camera named {name!r}"
            return
        errors = []
        for sharing in ("shared", "exclusive"):
            capture = None
            try:
                capture = await a.open(group, sharing)
                result["formats"] = list(a.formats(a.color_source(capture)))
                result["error"] = ""
                return
            except Exception as exc:
                errors.append(f"{sharing}: {exc}")
            finally:
                if capture is not None:
                    await a.stop(capture, None)
        result["error"] = "; ".join(errors)

    def work() -> None:
        try:
            asyncio.run(main())
        except Exception as exc:
            result["error"] = str(exc)

    th = threading.Thread(target=work, name="winrt-formats", daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return [], f"WinRT did not answer within {timeout:g} s"
    return result["formats"], result["error"]


def group_names(api: WinrtApi | None = None, timeout: float = OPEN_TIMEOUT_S) -> list[str] | None:
    """The cameras WinRT lists (colour source groups), or None when it could not be asked."""
    result: list[Any] = [None]

    async def main() -> None:
        result[0] = await (api or default_api()).group_names()

    def work() -> None:
        with contextlib.suppress(Exception):
            asyncio.run(main())

    th = threading.Thread(target=work, name="winrt-list", daemon=True)
    th.start()
    th.join(timeout)
    return result[0] if not th.is_alive() else None
