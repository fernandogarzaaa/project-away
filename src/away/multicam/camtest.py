"""``away cameras test`` and Settings > Cameras > Test both cameras (v0.1.15).

Opens every configured camera through the shared sequential start-up
(:func:`~away.multicam.capture.start_camera_threads`), streams for ~10 s, and reports per camera the backend,
the mode that worked, the measured fps, the frames received and any error. When the cameras do not all stream
together at the configured size, the secondary cameras are tried again at 320x240 MJPG @ 15 fps, and the
report says which combination works so it can be saved (``--save`` / the Settings button).

v0.1.16: a camera measured below ``hands.min_camera_fps`` (default 10) is FAILED ("too slow: X fps"), never
"ok", so the verdict no longer says two cameras stream together when one sends a frame per second. A camera that
was replaced by NVIDIA Broadcast's virtual camera (too slow in every mode, ``hands.virtual_fallback``) says so,
and saving the combination saves Broadcast's camera in its place.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from away.config import CameraSpec

LOW_RES_MODE = (320, 240, 15, "MJPG")  # the secondary's fallback: smallest common USB load
TEST_SECONDS = 10.0
# opening one after another may take a while on Windows: several modes per camera, each measured for 1.5 s, and
# a slow camera's sweep of every format and size (capture.SWEEP_BUDGET_S)
OPEN_TIMEOUT_S = 150.0


@dataclass
class CameraResult:
    device: str
    role: str
    backend: str = ""
    mode: str = ""  # "320x240@15 MJPG"
    width: int = 0
    height: int = 0
    fps_set: int = 0
    fourcc: str = ""
    frames: int = 0
    fps: float = 0.0
    error: str = ""
    min_fps: float = 0.0  # below this measured rate the camera is too slow (0 = no minimum)
    opened: str = ""  # the camera actually streaming when another replaced it (NVIDIA Broadcast), else ""
    opened_index: int | None = None
    fallback: str = ""  # why it was replaced
    hints: list[str] = field(default_factory=list)  # from the open (Broadcast offer, too slow...)

    @property
    def too_slow(self) -> bool:
        return self.min_fps > 0 and self.frames > 0 and not self.error and self.fps < self.min_fps

    @property
    def ok(self) -> bool:
        return self.frames > 0 and not self.error and not self.too_slow

    def line(self) -> str:
        head = f"{self.role:<9} {self.device}"
        how = self.mode + (f" via {self.backend}" if self.backend else "")
        if self.too_slow:
            text = f"{head}: FAILED - too slow: {self.fps:.1f} fps (minimum {self.min_fps:g}, hands.min_camera_fps)" + (
                f" - {how}, {self.frames} frames" if how else f" - {self.frames} frames"
            )
            extra = [h for h in self.hints if "too slow" not in h]
            return text + (f". {'; '.join(extra)}" if extra else "")
        if not self.ok:
            return f"{head}: FAILED - {self.error or 'no frames'}" + (f" ({self.frames} frames)" if self.frames else "")
        instead = f"using {self.opened} instead ({self.fallback}) - " if self.opened else ""
        return f"{head}: ok - {instead}{how}, {self.fps:.1f} fps measured, {self.frames} frames"


@dataclass
class Phase:
    label: str
    specs: list[CameraSpec]
    results: list[CameraResult]

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)


@dataclass
class TestReport:
    phases: list[Phase] = field(default_factory=list)

    @property
    def works(self) -> bool:
        return any(p.ok for p in self.phases)

    @property
    def working(self) -> Phase | None:
        return next((p for p in self.phases if p.ok), None)

    def combination(self) -> list[CameraSpec] | None:
        """The camera settings that worked together (each camera in the mode it actually streamed in), or
        None when no combination worked."""
        ph = self.working
        if ph is None:
            return None
        out = []
        for spec, r in zip(ph.specs, ph.results):
            s = copy.copy(spec)
            if r.opened:  # replaced by NVIDIA Broadcast's camera: save that camera (in its own format)
                s.device = r.opened
                s.index_hint = r.opened_index
                s.fourcc = ""
            if r.width and r.height:  # not the device default (0x0: nothing forced)
                s.width, s.height = r.width, r.height
                if r.fps_set:
                    s.fps = r.fps_set
                if r.fourcc:
                    s.fourcc = r.fourcc
            out.append(s)
        return out

    def save_offer(self, configured: list[CameraSpec]) -> list[CameraSpec] | None:
        """The combination worth saving: the one that worked, when it differs from ``configured``."""
        combo = self.combination()
        if combo is None:
            return None
        key = [(str(c.device), c.width, c.height, c.fps, c.fourcc) for c in combo]
        if key == [(str(c.device), c.width, c.height, c.fps, c.fourcc) for c in configured]:
            return None
        return combo

    def fallbacks(self) -> list[CameraResult]:
        ph = self.working
        return [r for r in ph.results if r.opened] if ph is not None else []

    def verdict(self) -> str:
        if not self.phases:
            return "no cameras were tested"
        n = len(self.phases[0].results)
        both = "both cameras" if n == 2 else f"all {n} cameras"
        first = self.phases[0]
        via = "".join(
            f" {r.device} is too slow on its own, so NVIDIA Broadcast's camera ({r.opened}) is used in its place"
            " - save the combination to keep it that way."
            for r in self.fallbacks()
        )
        if first.ok:
            return f"{both} stream at the same time with your current settings." + via
        slow = [r for r in first.results if r.too_slow]
        if slow and self.working is None:
            parts = []
            for r in slow:
                offer = next((h for h in r.hints if "Broadcast" in h), "")
                parts.append(
                    f"{r.device} streams, but far too slowly: {r.fps:.1f} fps (minimum {r.min_fps:g}, "
                    "hands.min_camera_fps) in every format and size tried, so it cannot track a hand. "
                    + (offer + "." if offer else "No free NVIDIA Broadcast camera could replace it.")
                )
            return (
                f"{both} do NOT work together: "
                + " ".join(parts)
                + " Use the other camera alone, or check this camera's driver and lighting (away.log lists every "
                "mode tried)."
            )
        ph = self.working
        if ph is not None:
            return (
                f"{both} do NOT stream together at the configured size, but they DO with the secondary camera at "
                f"{LOW_RES_MODE[0]}x{LOW_RES_MODE[1]} {LOW_RES_MODE[3]} @ {LOW_RES_MODE[2]} fps. Save that "
                "combination to use it." + via
            )
        return (
            f"{both} cannot stream at the same time, even with the secondary at "
            f"{LOW_RES_MODE[0]}x{LOW_RES_MODE[1]} {LOW_RES_MODE[3]}. Try another USB port (a different USB "
            "controller), close other camera apps, or use one camera."
        )

    def lines(self) -> list[str]:
        out: list[str] = []
        for p in self.phases:
            out.append(f"{p.label}:")
            out += [f"  {r.line()}" for r in p.results]
        out.append(self.verdict())
        return out


def low_res_specs(specs: list[CameraSpec]) -> list[CameraSpec]:
    """The same cameras with every secondary at 320x240 MJPG @ 15 fps."""
    w, h, f, fcc = LOW_RES_MODE
    out = []
    for c in specs:
        s = copy.copy(c)
        if c.role != "primary":
            s.width, s.height, s.fps, s.fourcc = w, h, f, fcc
        out.append(s)
    return out


def _result(spec: CameraSpec, th: Any, frames: int, elapsed: float, min_fps: float = 0.0) -> CameraResult:
    mode = getattr(th, "mode", None)
    r = CameraResult(
        device=str(spec.device),
        role=str(spec.role),
        backend=str(getattr(th, "backend", "") or ""),
        frames=frames,
        error=str(getattr(th, "error", "") or ""),
        min_fps=float(min_fps),
    )
    fallback = getattr(th, "fallback", "")
    opened = getattr(th, "opened_name", "")
    if isinstance(fallback, str) and fallback and isinstance(opened, str) and opened:
        r.fallback, r.opened = fallback, opened
        idx = getattr(th, "opened_index", None)
        r.opened_index = idx if isinstance(idx, int) else None
    hints = getattr(th, "hints", None)
    if isinstance(hints, list):
        r.hints = [str(h) for h in hints]
    r.fps = frames / elapsed if elapsed > 0 else 0.0
    if mode is not None:
        r.mode = mode.describe()
        r.width, r.height, r.fps_set, r.fourcc = mode.width, mode.height, mode.fps, mode.fourcc
    else:
        size = getattr(th, "size", None)
        if r.frames and size:
            r.width, r.height = int(size[0]), int(size[1])
            r.mode = f"{r.width}x{r.height}"
    if not r.error and r.frames == 0:
        r.error = "opened but sent no frames" if getattr(th, "opened", None) and th.opened.is_set() else "no frames"
    return r


def stream_cameras(
    specs: list[CameraSpec],
    seconds: float = TEST_SECONDS,
    *,
    start: Callable[..., list[Any]] | None = None,
    stop_threads: Callable[[threading.Event, list[Any]], bool] | None = None,
    open_timeout_s: float = OPEN_TIMEOUT_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
    cancel: threading.Event | None = None,
    min_fps: float = 0.0,
    virtual_fallback: bool = True,
) -> list[CameraResult]:
    """Open ``specs`` together (shared sequential start-up), wait until each streams or fails, stream for
    ``seconds`` with all of them running, and return one :class:`CameraResult` per camera (``min_fps``: a
    camera measured below it is too slow)."""
    from away.multicam.capture import FramePairer, primary_index, start_camera_threads, stop_camera_threads

    stop_threads = stop_threads or stop_camera_threads
    stop = threading.Event()
    pairer = FramePairer(len(specs), primary_index(specs), 0.04, 1.0, clock=clock)
    if start is None:
        threads = start_camera_threads(
            list(specs), pairer, stop, virtual_fallback=virtual_fallback, min_fps=min_fps or None
        )
    else:
        threads = start(list(specs), pairer, stop)
    cancelled = cancel or threading.Event()
    try:
        deadline = clock() + open_timeout_s
        while clock() < deadline and not cancelled.is_set():
            if all(th.frames > 0 or not th.is_alive() for th in threads):
                break
            sleep(0.05)
        base = [int(th.frames) for th in threads]  # count frames from here on, with every camera running
        t0 = clock()
        end = t0 + seconds
        while clock() < end and not cancelled.is_set():
            pairer.pop_set()  # keep the buffers moving like the app does
            if not any(th.is_alive() for th in threads):
                break
            sleep(0.02)
        elapsed = clock() - t0
        counts = [int(th.frames) - b for th, b in zip(threads, base)]
    finally:
        stop_threads(stop, threads)
    return [_result(s, th, n, elapsed, min_fps) for s, th, n in zip(specs, threads, counts)]


def run_camera_test(
    specs: list[CameraSpec],
    seconds: float = TEST_SECONDS,
    *,
    stream: Callable[[list[CameraSpec], float], list[CameraResult]] | None = None,
    fresh: bool = True,
    progress: Callable[[str], Any] | None = None,
    min_fps: float | None = None,
    virtual_fallback: bool = True,
) -> TestReport:
    """Configured settings first; when they do not all stream, the secondaries at 320x240 MJPG @ 15 (not when
    the only problem is a camera that is too slow: opening it already tried every format and size, 320x240
    included). ``min_fps``: default capture.MIN_CAMERA_FPS."""
    from away.multicam import capture

    floor = capture.MIN_CAMERA_FPS if min_fps is None else float(min_fps)
    run = stream or (lambda s, sec: stream_cameras(s, sec, min_fps=floor, virtual_fallback=virtual_fallback))
    say = progress or (lambda _m: None)

    def measured(results: list[CameraResult]) -> list[CameraResult]:
        for r in results:
            if not r.min_fps:
                r.min_fps = floor
        return results

    if fresh:  # measure the configured modes, not ones kept from an earlier open in this process
        capture.reset_session_modes()
    report = TestReport()
    say(f"opening {len(specs)} cameras one after another and streaming for {seconds:g} s...")
    report.phases.append(Phase("configured settings", list(specs), measured(run(list(specs), seconds))))
    first = report.phases[0]
    for r in first.results:
        if r.opened:
            say(f"{r.device}: {r.fallback}; using {r.opened} instead")
    if first.ok or not any(c.role != "primary" for c in specs):
        return report
    if all(r.ok or r.too_slow for r in first.results):
        return report  # too slow in every format and size already: a low-res retry cannot help
    low = low_res_specs(specs)
    w, h, f, fcc = LOW_RES_MODE
    say(f"not all cameras streamed; trying the secondary at {w}x{h} {fcc} @ {f} fps...")
    if fresh:
        capture.reset_session_modes()
    report.phases.append(Phase(f"secondary at {w}x{h} {fcc} @ {f}", low, measured(run(low, seconds))))
    return report


def save_combination(cfg: Any, combo: list[CameraSpec], configured: list[CameraSpec] | None = None) -> list[str]:
    """Write the working camera settings into ``cfg.hands.cameras`` (matched by position and device; with
    ``configured``, the cameras the test ran with, a camera NVIDIA Broadcast replaced is matched by its old
    device and saved as Broadcast's camera); a mode kept for that camera is dropped, since the config now asks
    for it. Returns what changed, for the user."""
    from away.multicam import capture

    changed: list[str] = []
    cams = cfg.hands.cameras
    for i, new in enumerate(combo):
        if i >= len(cams):
            continue
        was = str(configured[i].device) if configured is not None and i < len(configured) else str(new.device)
        if str(cams[i].device) not in (str(new.device), was):
            continue
        old = cams[i]
        if str(old.device) != str(new.device):
            changed.append(f"{old.device} -> {new.device} (NVIDIA Broadcast's camera)")
            capture.forget_mode(str(old.device))
            old.device, old.index_hint = new.device, new.index_hint
        before = (old.width, old.height, old.fps, old.fourcc)
        after = (new.width, new.height, new.fps, new.fourcc)
        if before != after:
            old.width, old.height, old.fps, old.fourcc = after
            changed.append(
                f"{old.device}: {before[0]}x{before[1]}@{before[2]} {before[3] or '-'} -> "
                f"{after[0]}x{after[1]}@{after[2]} {after[3] or '-'}"
            )
            capture.forget_mode(str(old.device))  # the config now asks for the mode that works
    return changed
