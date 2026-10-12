"""``away cameras test`` and Settings > Cameras > Test both cameras (v0.1.15).

Opens every configured camera through the shared sequential start-up
(:func:`~away.multicam.capture.start_camera_threads`), streams for ~10 s, and reports per camera the backend,
the mode that worked, the measured fps, the frames received and any error. When the cameras do not all stream
together at the configured size, the secondary cameras are tried again at 320x240 MJPG @ 15 fps, and the
report says which combination works so it can be saved (``--save`` / the Settings button).
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
OPEN_TIMEOUT_S = 30.0  # opening one after another may take a while on Windows (several modes per camera)


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

    @property
    def ok(self) -> bool:
        return self.frames > 0 and not self.error

    def line(self) -> str:
        head = f"{self.role:<9} {self.device}"
        if not self.ok:
            return f"{head}: FAILED - {self.error or 'no frames'}" + (f" ({self.frames} frames)" if self.frames else "")
        how = self.mode + (f" via {self.backend}" if self.backend else "")
        return f"{head}: ok - {how}, {self.fps:.1f} fps measured, {self.frames} frames"


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
        key = [(c.width, c.height, c.fps, c.fourcc) for c in combo]
        if key == [(c.width, c.height, c.fps, c.fourcc) for c in configured]:
            return None
        return combo

    def verdict(self) -> str:
        if not self.phases:
            return "no cameras were tested"
        n = len(self.phases[0].results)
        both = "both cameras" if n == 2 else f"all {n} cameras"
        first = self.phases[0]
        if first.ok:
            return f"{both} stream at the same time with your current settings."
        ph = self.working
        if ph is not None:
            return (
                f"{both} do NOT stream together at the configured size, but they DO with the secondary camera at "
                f"{LOW_RES_MODE[0]}x{LOW_RES_MODE[1]} {LOW_RES_MODE[3]} @ {LOW_RES_MODE[2]} fps. Save that "
                "combination to use it."
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


def _result(spec: CameraSpec, th: Any, frames: int, elapsed: float) -> CameraResult:
    mode = getattr(th, "mode", None)
    r = CameraResult(
        device=str(spec.device),
        role=str(spec.role),
        backend=str(getattr(th, "backend", "") or ""),
        frames=frames,
        error=str(getattr(th, "error", "") or ""),
    )
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
) -> list[CameraResult]:
    """Open ``specs`` together (shared sequential start-up), wait until each streams or fails, stream for
    ``seconds`` with all of them running, and return one :class:`CameraResult` per camera."""
    from away.multicam.capture import FramePairer, primary_index, start_camera_threads, stop_camera_threads

    start = start or start_camera_threads
    stop_threads = stop_threads or stop_camera_threads
    stop = threading.Event()
    pairer = FramePairer(len(specs), primary_index(specs), 0.04, 1.0, clock=clock)
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
    return [_result(s, th, n, elapsed) for s, th, n in zip(specs, threads, counts)]


def run_camera_test(
    specs: list[CameraSpec],
    seconds: float = TEST_SECONDS,
    *,
    stream: Callable[[list[CameraSpec], float], list[CameraResult]] | None = None,
    fresh: bool = True,
    progress: Callable[[str], Any] | None = None,
) -> TestReport:
    """Configured settings first; when they do not all stream, the secondaries at 320x240 MJPG @ 15."""
    from away.multicam import capture

    run = stream or (lambda s, sec: stream_cameras(s, sec))
    say = progress or (lambda _m: None)
    if fresh:  # measure the configured modes, not ones kept from an earlier open in this process
        capture.reset_session_modes()
    report = TestReport()
    say(f"opening {len(specs)} cameras one after another and streaming for {seconds:g} s...")
    report.phases.append(Phase("configured settings", list(specs), run(list(specs), seconds)))
    if report.phases[0].ok or not any(c.role != "primary" for c in specs):
        return report
    low = low_res_specs(specs)
    w, h, f, fcc = LOW_RES_MODE
    say(f"not all cameras streamed; trying the secondary at {w}x{h} {fcc} @ {f} fps...")
    if fresh:
        capture.reset_session_modes()
    report.phases.append(Phase(f"secondary at {w}x{h} {fcc} @ {f}", low, run(low, seconds)))
    return report


def save_combination(cfg: Any, combo: list[CameraSpec]) -> list[str]:
    """Write the working camera settings into ``cfg.hands.cameras`` (matched by position and device); a mode
    kept for that camera is dropped, since the config now asks for it. Returns what changed, for the user."""
    from away.multicam import capture

    changed: list[str] = []
    cams = cfg.hands.cameras
    for i, new in enumerate(combo):
        if i >= len(cams) or str(cams[i].device) != str(new.device):
            continue
        old = cams[i]
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
