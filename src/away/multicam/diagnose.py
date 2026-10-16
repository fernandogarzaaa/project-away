"""``away cameras diagnose`` and Settings > Cameras > Diagnose (v0.1.19).

For each camera and each capture backend (Windows: DirectShow, Media Foundation, WinRT MediaCapture; elsewhere
OpenCV's default backend) list the native formats where the backend can say (WinRT can; OpenCV cannot list
DirectShow / Media Foundation formats, so a set of common modes is tried instead), try to open each one and
print ``ok`` with the measured frame rate, or the error. Every device call goes through injectable pieces, so
the whole report is unit-tested with fake cameras.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from away.multicam import capture
from away.multicam.capture import CameraDevice, CaptureMode

BACKENDS = ("dshow", "msmf", "winrt")
BACKEND_LABELS = {"dshow": "DirectShow", "msmf": "Media Foundation", "winrt": "WinRT", "default": "default"}
# v0.1.28: the backend labels the mode file remembers (capture.saved_modes) back to diagnose's short names.
REMEMBERED_BACKENDS = {v: k for k, v in BACKEND_LABELS.items()}
MEASURE_S = 1.0  # frames are counted this long per mode
FIRST_FRAME_S = 3.0
MAX_WINRT_FORMATS = 6  # native formats opened per camera (all of them are listed)
BUDGET_S = 60.0  # v0.1.22: the whole diagnosis stops after this long (it ran for minutes on the user's PC)


def diagnose_modes(width: int = 640, height: int = 480, fps: int = 30) -> list[CaptureMode]:
    """The modes tried on an OpenCV backend: the configured size in MJPG, YUY2, NV12 and the driver's format,
    then 1280x720 MJPG and driver format, then 320x240 MJPG."""
    w, h = (int(width), int(height)) if width and height else (640, 480)
    f = int(fps or 0) or 30
    modes = [CaptureMode(w, h, f, x) for x in ("MJPG", "YUY2", "NV12", "")]
    modes += [CaptureMode(1280, 720, f, "MJPG"), CaptureMode(1280, 720, f, "")]
    # v0.1.21: the smaller uncompressed modes that usually reach 30 fps when 720p YUY2 only reaches 10 over USB 2
    modes += [CaptureMode(960, 540, f, "YUY2"), CaptureMode(640, 480, f, "YUY2"), CaptureMode(320, 240, f, "MJPG")]
    out: list[CaptureMode] = []
    for m in modes:
        if m not in out:
            out.append(m)
    return out


def winrt_candidates(formats: Sequence[Any], width: int, height: int, fps: int, limit: int) -> list[Any]:
    """Up to ``limit`` native formats to open, best first (the order :func:`choose_format` prefers)."""
    from away.multicam.winrt_capture import choose_format

    left = list(formats)
    out: list[Any] = []
    while left and len(out) < limit:
        pick = choose_format(left, width, height, fps, max_size=(10**6, 10**6))
        if pick is None:
            break
        out.append(pick)
        left.remove(pick)
    return out


def _probe(
    cap: Any, cv2: Any, mode: CaptureMode | None, measure_s: float, clock: Callable[[], float], sleep: Any
) -> tuple[bool, str]:
    """Read from an opened capture: ``(True, "ok 29.8 fps ...")`` or ``(False, "no frames")``."""
    frame = capture.read_first(cap, FIRST_FRAME_S, clock, sleep)
    if frame is None:
        return False, "opened, but no frames"
    fps, frame = capture.measure_fps(cap, frame, measure_s, clock, sleep)
    shape = getattr(frame, "shape", None)
    size = f"{shape[1]}x{shape[0]}" if shape is not None and len(shape) >= 2 else "?"
    rb = capture.read_back(cap, cv2) if cv2 is not None else {}
    got = rb.get("fourcc") or ""
    note = ""
    if mode is not None and mode.fourcc and got and got != mode.fourcc:
        note = f"; asked {mode.fourcc}, driver reports {got}"
    elif got:
        note = f"; driver reports {got}"
    rate = "?" if fps is None else f"{fps:.1f} fps measured"
    return True, f"ok {rate}, frames {size}{note}{speed_verdict(fps)}"


def speed_verdict(fps: float | None) -> str:
    """v0.1.21: what the measured rate means for hand tracking, in plain words."""
    if fps is None:
        return ""
    if fps >= capture.TRACKING_FPS:
        return " - fast enough for smooth tracking"
    if fps >= capture.MIN_CAMERA_FPS:
        return f" - works, but slow for tracking (smooth needs {capture.TRACKING_FPS:g}+ fps)"
    return " - too slow for tracking"


def _quick_probe(
    dev: CameraDevice,
    devices: Sequence[CameraDevice],
    emit: Callable[[str], None],
    width: int,
    height: int,
    fps: int,
    measure_s: float,
    cv2: Any,
    winrt_available: Callable[[], tuple[bool, str]] | None,
    winrt_formats: Callable[[str, int], tuple[list[Any], str]] | None,
    winrt_open: Callable[[str, Any, int], Any] | None,
    msmf_lister: Callable[[], list[tuple[str, str]] | None] | None,
    clock: Callable[[], float],
    sleep: Callable[[float], Any],
) -> None:
    """v0.1.28: quick triage for one camera: open it on its remembered backend in its remembered mode only,
    skipping the mode sweep. A camera with no remembered mode needs a full check once first (that is what
    records the backend and mode in the mode file)."""
    remembered = capture.saved_modes().get(dev.name)
    if remembered is None:
        emit("  quick: no remembered backend for this camera; run a full check once to record one")
        return
    label, mode = remembered
    b = REMEMBERED_BACKENDS.get(label, label)
    info = capture.saved_mode_info().get(dev.name, {})
    remembered_fps = info.get("fps")
    fps_txt = f" (remembered {remembered_fps:.1f} fps)" if isinstance(remembered_fps, (int, float)) else ""
    emit(f"  quick: remembered {label} {mode.describe()}{fps_txt}")
    if b == "winrt":
        from away.multicam import winrt_capture

        ok, why = (winrt_available or winrt_capture.available)()
        if not ok:
            emit(f"  WinRT: unavailable ({why})")
            return
        same = [d for d in devices if d.name.lower() == dev.name.lower()]
        occurrence = next((j for j, d in enumerate(same) if d is dev), 0)
        formats, err = (winrt_formats or (lambda n, o: winrt_capture.list_formats(n, o)))(dev.name, occurrence)
        if err and not formats:
            emit(f"  WinRT: {err}")
            return
        # v0.1.28 (CodeRabbit): match the whole remembered mode (size, FOURCC, rounded fps); never probe a
        # substitute mode and report it as the remembered one.
        match = next(
            (
                f
                for f in formats
                if f.width == mode.width
                and f.height == mode.height
                and f.fourcc() == mode.fourcc
                and round(float(f.fps or 0)) == int(mode.fps or 0)
            ),
            None,
        )
        if match is None:
            emit(f"  WinRT: remembered mode {mode.describe()} is not offered by this camera now")
            return
        _winrt(dev, devices, emit, width, height, fps, measure_s, cv2, winrt_available, winrt_formats, winrt_open,
               1, clock, sleep, formats_override=([match], ""))  # fmt: skip
        return
    if cv2 is None:
        emit("  quick: skipped (OpenCV not installed)")
        return
    where = dev.index if dev.index is not None else dev.path
    src: int | str | None = where
    api: Any = None
    if b == "dshow":
        api = getattr(cv2, "CAP_DSHOW", None)
    elif b == "msmf":
        api = getattr(cv2, "CAP_MSMF", None)
        if isinstance(where, int):
            src = capture.msmf_index(where, dev, list(devices), repr(dev.name), msmf_lister)
        if src is None:
            emit("  Media Foundation: not listed by Media Foundation")
            return
    emit(f"  {BACKEND_LABELS.get(b, b)} (index {src}): quick check of the remembered mode")
    try:
        cap = cv2.VideoCapture(src) if api is None else cv2.VideoCapture(src, api)
    except Exception as exc:
        emit(f"    {mode.describe()}: error {exc}")
        return
    try:
        if not cap.isOpened():
            emit(f"    {mode.describe()}: did not open")
            return
        capture.apply_mode(cap, mode, cv2, repr(dev.name))
        _ok, text = _probe(cap, cv2, mode, measure_s, clock, sleep)
        emit(f"    {mode.describe()}: {text}")
    except Exception as exc:
        emit(f"    {mode.describe()}: error {exc}")
    finally:
        with contextlib.suppress(Exception):
            cap.release()


def run_diagnose(
    devices: Sequence[CameraDevice],
    *,
    cv2: Any = None,
    backends: Sequence[str] | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    measure_s: float = MEASURE_S,
    out: Callable[[str], Any] | None = None,
    windows: bool | None = None,
    msmf_lister: Callable[[], list[tuple[str, str]] | None] | None = None,
    winrt_available: Callable[[], tuple[bool, str]] | None = None,
    winrt_formats: Callable[[str, int], tuple[list[Any], str]] | None = None,
    winrt_open: Callable[[str, Any, int], Any] | None = None,
    max_winrt_formats: int = MAX_WINRT_FORMATS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
    budget_s: float | None = BUDGET_S,
    include_virtual: bool = False,
    quick: bool = False,
) -> list[str]:
    """Diagnose every camera in ``devices`` on every backend; each line is passed to ``out`` as it is made
    (Settings shows them while the diagnosis runs) and all of them are returned.

    v0.1.22: virtual cameras (NVIDIA Broadcast, OBS...) are skipped unless ``include_virtual`` (or the camera was
    named), and the whole run stops after ``budget_s`` seconds (None or 0 = no limit) with a line saying so.
    v0.1.28: ``quick`` opens each camera on its remembered backend in its remembered mode only (no mode sweep),
    for a fast triage answer; a camera with no remembered mode needs a full check once first."""
    lines: list[str] = []
    say = out or (lambda _m: None)
    started = clock()
    budget = float(budget_s or 0.0)

    def over() -> bool:
        return budget > 0 and clock() - started > budget

    def emit(text: str) -> None:
        lines.append(text)
        say(text)

    win = capture._windows() if windows is None else windows
    if cv2 is None:  # pragma: no cover - real OpenCV
        try:
            cv2 = capture.load_cv2()
        except ImportError:
            emit("OpenCV is not installed (pip install 'project-away[hands]'); only WinRT can be checked")
    use = list(backends or (BACKENDS if win else ("default",)))
    if not devices:
        emit("no cameras found")
        return lines
    stopped = False
    for dev in devices:
        where = dev.index if dev.index is not None else dev.path
        if dev.virtual and not include_virtual and len(devices) > 1:
            emit(f"camera {where}: {dev.name} (virtual) - skipped; name it to check it anyway")
            continue
        if over():
            stopped = True
            break
        emit(f"camera {where}: {dev.name}" + (" (virtual)" if dev.virtual else ""))
        if quick:
            _quick_probe(
                dev,
                devices,
                emit,
                width,
                height,
                fps,
                measure_s,
                cv2,
                winrt_available,
                winrt_formats,
                winrt_open,
                msmf_lister,
                clock,
                sleep,
            )
            if over():  # v0.1.28 (CodeRabbit): a single probe can run past the budget too
                stopped = True
        else:
            for b in use:
                if over():
                    stopped = True
                    break
                label = BACKEND_LABELS.get(b, b)
                if b == "winrt":
                    _winrt(dev, devices, emit, width, height, fps, measure_s, cv2, winrt_available, winrt_formats,
                           winrt_open, max_winrt_formats, clock, sleep)  # fmt: skip
                    continue
                if cv2 is None:
                    emit(f"  {label}: skipped (OpenCV not installed)")
                    continue
                src: int | str | None = where
                api: Any = None
                if b == "dshow":
                    api = getattr(cv2, "CAP_DSHOW", None)
                elif b == "msmf":
                    api = getattr(cv2, "CAP_MSMF", None)
                    if isinstance(where, int):
                        src = capture.msmf_index(where, dev, list(devices), repr(dev.name), msmf_lister)
                    if src is None:
                        emit(f"  {label}: not listed by Media Foundation")
                        continue
                emit(f"  {label} (index {src}): OpenCV cannot list this backend's formats; trying common modes")
                best: tuple[float, str] | None = None
                for mode in diagnose_modes(width, height, fps):
                    if over():
                        stopped = True
                        break
                    try:
                        cap = cv2.VideoCapture(src) if api is None else cv2.VideoCapture(src, api)
                    except Exception as exc:
                        emit(f"    {mode.describe()}: error {exc}")
                        break
                    try:
                        if not cap.isOpened():
                            emit(f"    {mode.describe()}: did not open (skipping this backend's other modes)")
                            break
                        capture.apply_mode(cap, mode, cv2, repr(dev.name))
                        _ok, text = _probe(cap, cv2, mode, measure_s, clock, sleep)
                        emit(f"    {mode.describe()}: {text}")
                        rate = _rate_of(text)
                        if _ok and rate is not None and (best is None or rate > best[0]):
                            best = (rate, mode.describe())
                    except Exception as exc:
                        emit(f"    {mode.describe()}: error {exc}")
                    finally:
                        with contextlib.suppress(Exception):
                            cap.release()
                if best is not None:
                    emit(f"  {label}: fastest mode {best[1]} at {best[0]:.1f} fps{speed_verdict(best[0])}")
        saved = capture.saved_mode_info().get(dev.name)
        modes = capture.saved_modes().get(dev.name)
        if modes is not None:
            b, m = modes
            fps_txt = f" ({saved['fps']:.1f} fps measured)" if saved and saved.get("fps") is not None else ""
            emit(f"  remembered for this camera: {b} {m.describe()}{fps_txt}")
        if stopped:
            break
    if stopped:
        emit(
            f"stopped after {budget:g} s (time limit) - the cameras checked so far are above; "
            "name one camera to check just that one"
        )
    return lines


def _rate_of(text: str) -> float | None:
    import re

    m = re.search(r"ok ([0-9.]+) fps", text)
    return float(m.group(1)) if m else None


def _winrt(
    dev: CameraDevice,
    devices: Sequence[CameraDevice],
    emit: Callable[[str], None],
    width: int,
    height: int,
    fps: int,
    measure_s: float,
    cv2: Any,
    available: Callable[[], tuple[bool, str]] | None,
    formats_of: Callable[[str, int], tuple[list[Any], str]] | None,
    opener: Callable[[str, Any, int], Any] | None,
    limit: int,
    clock: Callable[[], float],
    sleep: Callable[[float], Any],
    formats_override: tuple[list[Any], str] | None = None,
) -> None:
    from away.multicam import winrt_capture

    ok, why = (available or winrt_capture.available)()
    if not ok:
        emit(f"  WinRT: unavailable ({why})")
        return
    same = [d for d in devices if d.name.lower() == dev.name.lower()]
    occurrence = next((j for j, d in enumerate(same) if d is dev), 0)
    if formats_override is not None:
        formats, err = formats_override  # v0.1.28: quick triage picks the remembered native format first
    else:
        formats, err = (formats_of or (lambda n, o: winrt_capture.list_formats(n, o)))(dev.name, occurrence)
    if err and not formats:
        emit(f"  WinRT: {err}")
        return
    emit(f"  WinRT: {len(formats)} native formats: " + (", ".join(f.describe() for f in formats) or "none"))
    if not formats:
        return

    def default_open(name: str, f: Any, occ: int) -> Any:
        return winrt_capture.WinrtCapture(
            name, f.width, f.height, f.fps, f.fourcc(), out_size=(f.width, f.height), occurrence=occ
        )

    for f in winrt_candidates(formats, width, height, fps, limit):
        try:
            cap = (opener or default_open)(dev.name, f, occurrence)
        except Exception as exc:
            emit(f"    {f.describe()}: error {exc}")
            continue
        try:
            if not cap.isOpened():
                emit(f"    {f.describe()}: did not open ({getattr(cap, 'error', '') or 'no reason given'})")
                continue
            _ok, text = _probe(cap, None, None, measure_s, clock, sleep)
            emit(f"    {f.describe()}: {text}")
        finally:
            with contextlib.suppress(Exception):
                cap.release()


# ------------------------------------------------------------------------------------------- driver noise
# v0.1.22: NVIDIA Broadcast's virtual camera driver writes "[VCAMDS] ..." lines to stderr every time a camera is
# opened, flooding the report. These patterns are camera-driver / OpenCV chatter, not results.
NOISE_PATTERNS = (
    re.compile(r"^\s*\[VCAMDS\]"),  # NVIDIA Broadcast virtual camera
    re.compile(r"^\s*\[\s*(WARN|INFO|DEBUG|ERROR):\d+@[\d.]+\]"),  # OpenCV's own log lines
    re.compile(r"^\s*\[[A-Z][A-Z0-9_]{2,15}\]\s"),  # other drivers' "[TAG] message" lines
)
NOISE_NAMES = {"VCAMDS": "NVIDIA Broadcast"}


def is_noise(line: str) -> bool:
    return any(p.search(line) for p in NOISE_PATTERNS)


class NoiseFilter:
    """Collapses camera-driver chatter in a stream of output lines: the first line of each kind is kept (shortened),
    the rest are counted, and :meth:`summary` says how many were hidden."""

    def __init__(self, keep_first: int = 1) -> None:
        self.keep_first = keep_first
        self.counts: dict[str, int] = {}

    @staticmethod
    def kind(line: str) -> str:
        m = re.match(r"^\s*\[\s*([A-Za-z0-9_]+)", line)
        return m.group(1).upper() if m else "driver"

    def feed(self, line: str) -> str | None:
        """The line to show, or None when it is hidden."""
        if not is_noise(line):
            return line
        k = self.kind(line)
        n = self.counts.get(k, 0) + 1
        self.counts[k] = n
        if n <= self.keep_first:
            return f"(camera driver) {line.strip()[:160]}"
        return None

    def hidden(self) -> int:
        return sum(max(0, n - self.keep_first) for n in self.counts.values())

    def summary(self) -> str:
        """ "(hid 812 camera-driver messages: NVIDIA Broadcast [VCAMDS])" or ""."""
        if not self.hidden():
            return ""
        parts = [
            f"{NOISE_NAMES[k]} [{k}]" if k in NOISE_NAMES else f"[{k}]"
            for k, n in self.counts.items()
            if n > self.keep_first
        ]
        n = self.hidden()
        return f"(hid {n} camera-driver message{'s' if n != 1 else ''}: {', '.join(parts)})"


def filter_lines(lines: Sequence[str]) -> list[str]:
    f = NoiseFilter()
    out = [x for x in (f.feed(line) for line in lines) if x is not None]
    if f.summary():
        out.append(f.summary())
    return out


def diagnose_devices(device: Any = None) -> list[CameraDevice]:  # pragma: no cover - lists real cameras
    """The cameras to diagnose: all listed ones, or the one ``device`` names."""
    devs = capture.list_cameras(log_list=True)
    if device is None or device == "":
        return devs
    found = capture.find_device(device, devs)
    return [found] if found is not None else []
