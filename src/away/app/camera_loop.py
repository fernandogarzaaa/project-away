"""The app's camera thread body: frames -> trackers, with camera hand-over and a clean shutdown.

* Hand-over: when Settings asks for the camera (:class:`~away.app.camera_share.CameraShare`), the loop
  closes its capture, acknowledges, waits until the request goes away, then reopens it.
* A camera that will not open (unplugged, or held by another app) is retried every ``retry_s`` instead of
  killing the thread.
* Shutdown: once ``stop`` is set, an error from the tracker (MediaPipe raises ``RuntimeError: cannot
  schedule new futures after shutdown`` when the interpreter is exiting) ends the loop quietly.

Device access is injected (``open_frames``/``process``) so all of this is unit-tested without a camera.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from away.app.camera_share import CameraShare

Frames = Iterator[tuple[float, Any]]

SHUTDOWN_ERRORS = ("cannot schedule new futures after shutdown", "interpreter shutdown")


def is_shutdown_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and any(s in str(exc) for s in SHUTDOWN_ERRORS)


class CameraLoop:
    def __init__(
        self,
        open_frames: Callable[[], Frames],
        process: Callable[[int, float, Any], None],
        stop: threading.Event,
        *,
        cameras: list[Any] | None = None,
        share: CameraShare | None = None,
        publish: Callable[[str, dict[str, Any]], None] | None = None,
        on_release: Callable[[], None] | None = None,
        retry_s: float = 2.0,
        check_s: float = 0.5,
        owner_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        until: Callable[[], bool] | None = None,
    ) -> None:
        self.open_frames = open_frames
        self.process = process
        self.stop = stop
        self.cameras = list(cameras if cameras is not None else [0])
        self.share = share
        self.publish = publish or (lambda _t, _d: None)
        self.on_release = on_release
        self.retry_s = retry_s
        self.check_s = check_s
        self.owner_s = owner_s
        self.clock = clock
        self.until = until  # optional: end the loop (camera closed) once this returns True
        self.handovers = 0
        self.open_failures = 0
        self._last_error: str | None = None

    def _requested(self) -> bool:
        return self.share is not None and self.share.requested()

    def _done(self) -> bool:
        if self.stop.is_set():
            return True
        if self.until is None:
            return False
        try:
            return bool(self.until())
        except Exception:
            return False

    def run(self) -> None:
        try:
            while not self._done():
                if self._requested():
                    self._hand_over()
                    continue
                if not self._session():
                    self.stop.wait(self.retry_s)
        finally:
            if self.share is not None:
                self.share.clear_owner()
                self.share.clear_released()

    # one capture session; False when the camera could not be opened
    def _session(self) -> bool:
        try:
            frames = self.open_frames()
            first = next(frames, None)  # opening is lazy in a generator: make it happen here
        except Exception as exc:
            if self.stop.is_set() or is_shutdown_error(exc):
                return True
            self.open_failures += 1
            msg = str(exc)
            if msg != self._last_error:  # report each new problem once, not every retry
                self._last_error = msg
                self.publish("camera.error", {"error": msg, "retry_s": self.retry_s})
            return False
        if self._last_error is not None:
            self._last_error = None
            self.publish("camera.recovered", {"cameras": self.cameras})
        next_check = next_owner = self.clock()
        n = 0
        try:
            item = first
            while item is not None:
                if self.stop.is_set():
                    break
                now = self.clock()
                if now >= next_owner and self.share is not None:
                    self.share.publish_owner(self.cameras)
                    next_owner = now + self.owner_s
                if now >= next_check:
                    next_check = now + self.check_s
                    if self._requested() or self._done():
                        break
                t, frame = item
                try:
                    self.process(n, t, frame)
                except Exception as exc:
                    if self.stop.is_set() or is_shutdown_error(exc):
                        self.stop.set()  # quitting: MediaPipe's executor is already gone
                        return True
                    raise
                n += 1
                item = next(frames, None)
        finally:
            close = getattr(frames, "close", None)
            if close is not None:  # releases the capture (the generator's finally)
                with contextlib.suppress(Exception):
                    close()
        return True

    def _hand_over(self) -> None:
        assert self.share is not None
        self.handovers += 1
        if self.on_release is not None:
            with contextlib.suppress(Exception):
                self.on_release()  # e.g. "hand lost": releases a held drag
        self.share.clear_owner()
        self.share.ack_released(self.cameras)
        self.publish("camera.handover", {"released": True, "cameras": self.cameras})
        while self._requested() and not self.stop.is_set():
            self.stop.wait(self.check_s)
        self.share.clear_released()
        self.publish("camera.handover", {"released": False, "cameras": self.cameras})


def probe_cameras(
    threads: Sequence[Any],
    seen: Callable[[int], bool],
    timeout_s: float,
    stop: threading.Event,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
    step: float = 0.05,
) -> dict[int, str]:
    """Wait until every camera thread has delivered a frame or died, or ``timeout_s`` passed.

    Returns ``{camera: reason}`` for the cameras that failed. A camera fails only when its thread died
    with an error (``open error: ...``) or ended/stayed silent without a frame by the deadline
    (``no frames within N s``). A slow camera that is still opening is waited for until the deadline.
    """
    deadline = clock() + timeout_s
    while clock() < deadline and not stop.is_set():
        if all(seen(i) or not th.is_alive() for i, th in enumerate(threads)):
            break
        sleep(step)
    failed: dict[int, str] = {}
    for i, th in enumerate(threads):
        err = getattr(th, "error", None)
        alive = th.is_alive()
        if not alive and err:
            failed[i] = f"open error: {err}" if not seen(i) else f"error: {err}"
        elif not seen(i):
            failed[i] = "closed before sending a frame" if not alive else f"no frames within {timeout_s:g} s"
    return failed


def describe_failures(devices: Sequence[Any], failed: Mapping[int, str] | set[int]) -> str:
    """``camera 1 (no frames within 12 s)``, ``camera 0 (open error: ...), camera 1 (...)``."""
    reasons = failed if isinstance(failed, Mapping) else dict.fromkeys(failed, "")
    parts = []
    for i in sorted(reasons):
        why = reasons[i]
        parts.append(f"camera {devices[i]}" + (f" ({why})" if why else ""))
    return ", ".join(parts)


def degrade_plan(
    devices: Sequence[Any],
    primary: int,
    failed: Mapping[int, str] | set[int],
    retry_s: float | None = None,
) -> tuple[str, Any, str]:
    """What to do when some configured cameras could not be opened (or sent no frame).

    Returns ``(mode, device, message)``:

    * ``("multi", None, "")`` when two or more cameras work (a dead extra camera is left out of the fusion);
    * ``("single", device, why)`` with the one working camera to track on (a missing secondary camera
      never stops hand tracking; the full set is retried in the background);
    * ``("none", primary, why)`` when no camera works at all (nothing is tracked; everything is retried).

    The message names only the cameras that failed, with their real error.
    """
    working = [i for i in range(len(devices)) if i not in failed]
    if len(working) >= 2:
        return "multi", None, ""
    bad = describe_failures(devices, failed) or "-"
    every = f" every {retry_s:g} s" if retry_s else ""
    if not working:
        msg = (
            f"no camera is working: {bad}. Hand tracking is off; retrying{every}. Close other apps that use "
            "the cameras, or check Settings > Cameras."
        )
        return "none", devices[primary], msg
    pick = working[0]
    msg = (
        f"{bad} failed; tracking with camera {devices[pick]} only and retrying the full set{every}. "
        "Fix or remove it in Settings > Cameras."
    )
    return "single", devices[pick], msg
