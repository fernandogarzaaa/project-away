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
from typing import Any, Protocol

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
        wait_free: Callable[[], bool] | None = None,
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
        # v0.1.16: before telling the borrower the cameras are free, wait until this process really holds and
        # opens none (a background probe may still be opening one); see capture.wait_devices_free
        self.wait_free = wait_free
        self.handovers = 0
        self.in_use_by: str | None = None  # v0.1.21: "Settings (Gesture Studio)" while it has the camera
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
        if self.wait_free is not None:
            with contextlib.suppress(Exception):
                self.wait_free()
        from away.app.camera_share import borrower_label, in_use_status

        self.share.clear_owner()
        self.share.ack_released(self.cameras)
        req = getattr(self.share, "requester", None)
        who = (req() if callable(req) else None) or "settings"
        self.in_use_by = borrower_label(who)
        self.publish(
            "camera.handover",
            {
                "released": True,
                "cameras": self.cameras,
                "by": who,
                "by_label": self.in_use_by,
                "status": in_use_status(who),
            },
        )
        share = self.share
        # v0.1.21: a request that seems gone is read once more before the camera is reopened (see still_requested)
        still = getattr(share, "still_requested", None)
        while not self.stop.is_set() and (still(self.stop.wait) if callable(still) else share.requested()):
            self.stop.wait(self.check_s)
            share.ack_released(self.cameras)  # keep the acknowledgement fresh (it goes stale after 5 s)
        share.clear_released()
        self.in_use_by = None
        self.publish("camera.handover", {"released": False, "cameras": self.cameras, "by": who})


class StopFlag(Protocol):
    """Anything with ``is_set()`` (a ``threading.Event``, or the app's stop-or-handover flag)."""

    def is_set(self) -> bool: ...


def probe_cameras(
    threads: Sequence[Any],
    seen: Callable[[int], bool],
    timeout_s: float,
    stop: StopFlag,
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


def retry_labels(specs: Sequence[Any], bad: Mapping[int, str]) -> dict[str, str]:
    """``{device: reason}`` for a background retry: ``bad`` is keyed by position in the retried subset
    (``specs``), so key 0 is the first *retried* camera, not camera 0 of the whole set."""
    out: dict[str, str] = {}
    for k, why in bad.items():
        dev = getattr(specs[k], "device", specs[k]) if 0 <= k < len(specs) else k
        out[str(dev)] = why
    return out


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

    The message names only the cameras that failed, with their real error. (Virtual cameras are kept out
    before this: :func:`away.multicam.capture.pin_camera_set` never puts a legacy index on one.)
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


def camera_speed_lines(threads: Sequence[Any], min_fps: float) -> list[tuple[bool, str]]:
    """One away.log line per camera once it has streamed a while (v0.1.16): ``(failed, text)``. A camera below
    ``min_fps`` is FAILED (too slow), not ok; one that NVIDIA Broadcast's camera replaced says so."""
    out: list[tuple[bool, str]] = []
    for th in threads:
        dev = getattr(getattr(th, "spec", None), "device", "?")
        fps_fn = getattr(th, "fps", None)
        fps = float(fps_fn()) if callable(fps_fn) else 0.0
        describe = getattr(th, "describe_mode", None)
        mode = describe() if callable(describe) else ""
        via = getattr(th, "opened_name", "")
        why = getattr(th, "fallback", "")
        name = f"{dev!r}" + (f" (via {via}: {why})" if isinstance(via, str) and via and why else "")
        err = getattr(th, "error", None)
        if err:
            out.append((True, f"camera {name}: FAILED - {err}"))
            continue
        check = getattr(th, "too_slow", None)
        slow = bool(check(min_fps)) if callable(check) and min_fps > 0 else False
        if slow:
            out.append(
                (
                    True,
                    f"camera {name}: FAILED - too slow: {fps:.1f} fps (minimum {min_fps:g}, hands.min_camera_fps)"
                    + (f", {mode}" if mode else "")
                    + "; it adds nothing to hand tracking",
                )
            )
        else:
            out.append((False, f"camera {name}: ok - {mode or 'streaming'}, {fps:.1f} fps measured"))
    return out


# ------------------------------------------------------------------------------ camera retries (v0.1.19)
# v0.1.17 on the Acer: the background re-probe of a dead camera fired 306 times in one day (every 45 s, forever),
# each time reopening a camera for ~10 s. It now backs off exponentially and stops after a number of failures,
# with a notice; "Retry cameras" (tray, Settings > Cameras) starts it again.
RETRY_REQUEST_FILE = "camera-retry-request.json"  # Settings -> app: retry the cameras now
RETRY_STATUS_FILE = "camera-retry-status.json"  # app -> Settings: retrying / stopped, attempts, next retry


class RetryBackoff:
    """Exponential back-off for re-probing failed cameras: ``base_s``, then twice as long after every failure,
    capped at ``max_s``; ``exhausted`` once ``limit`` failures happened in a row (0 = never). Pure logic."""

    def __init__(
        self, base_s: float = 45.0, max_s: float = 900.0, limit: int = 8, factor: float = 2.0, last_max: bool = False
    ) -> None:
        self.base_s = max(0.0, float(base_s))
        self.max_s = max(self.base_s, float(max_s))
        self.limit = max(0, int(limit))
        self.factor = max(1.0, float(factor))
        self.last_max = bool(last_max)  # v0.1.23: the last attempt before stopping waits the full max_s
        self.failures = 0

    def _wait(self, f: int) -> float:
        if self.last_max and self.limit > 0 and f >= self.limit - 1:
            return float(self.max_s)
        return float(min(self.max_s, self.base_s * (self.factor ** min(f, 60))))

    def delay(self) -> float:
        """Seconds to wait before the next attempt."""
        return self._wait(self.failures)

    def failed(self) -> None:
        self.failures += 1

    def reset(self) -> None:
        self.failures = 0

    @property
    def exhausted(self) -> bool:
        return self.limit > 0 and self.failures >= self.limit

    def schedule(self, n: int | None = None) -> list[float]:
        """The waits of the next ``n`` attempts (default: until the limit) from a fresh start, for messages and
        tests: 45, 90, 180, 360, 720, 900, 900, 900."""
        count = self.limit if n is None else n
        return [self._wait(f) for f in range(count)]


# v0.1.23: words that mean another program holds the camera (NVIDIA Broadcast keeps its source camera open)
BUSY_WORDS = ("element not found", "in use", "busy", "being used", "0x80070490", "0xc00d3704", "access denied")


def unavailable_text(name: str, reason: str = "", broadcast_present: bool = False) -> str:
    """ "Internal camera unavailable (probably in use by NVIDIA Broadcast)" for the tray and Settings > Cameras."""
    low = str(name).lower()
    internal = any(w in low for w in ("user facing", "integrated", "internal", "built-in", "facetime"))
    label = "Internal camera" if internal else str(name or "A camera")
    why = str(reason or "").lower()
    busy = any(w in why for w in BUSY_WORDS)
    if broadcast_present and (busy or not why):
        return f"{label} unavailable (probably in use by NVIDIA Broadcast)"
    if busy:
        return f"{label} unavailable (probably in use by another app)"
    return f"{label} unavailable"


def _data_folder(folder: Any = None) -> Any:
    from pathlib import Path

    if folder is not None:
        return Path(folder)
    from away import paths

    return Path(paths.data_dir())


def request_camera_retry(folder: Any = None, clock: Callable[[], float] = time.time) -> bool:
    """Ask the running app to retry its cameras now (Settings > Cameras > Retry cameras). True when written."""
    import json

    p = _data_folder(folder) / RETRY_REQUEST_FILE
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"t": clock()}), encoding="utf-8")
        return True
    except OSError:
        return False


def take_retry_request(folder: Any = None, clock: Callable[[], float] = time.time, max_age_s: float = 3600.0) -> bool:
    """True (once) when Settings asked for a retry within ``max_age_s``; the request is removed."""
    import json

    p = _data_folder(folder) / RETRY_REQUEST_FILE
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    with contextlib.suppress(OSError):
        p.unlink()
    t = data.get("t") if isinstance(data, dict) else None
    return isinstance(t, (int, float)) and clock() - float(t) <= max_age_s


def write_retry_status(
    status: Mapping[str, Any] | None, folder: Any = None, clock: Callable[[], float] = time.time
) -> None:
    """The app's retry state for Settings (None: nothing is being retried; the file is removed)."""
    import json

    p = _data_folder(folder) / RETRY_STATUS_FILE
    with contextlib.suppress(OSError):
        if status is None:
            p.unlink(missing_ok=True)
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({**status, "t": clock()}), encoding="utf-8")
        import os

        os.replace(tmp, p)


def read_retry_status(folder: Any = None) -> dict[str, Any] | None:
    import json

    try:
        data = json.loads((_data_folder(folder) / RETRY_STATUS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def retry_status_text(status: Mapping[str, Any] | None) -> str:
    """One line for Settings > Cameras: what the app's camera retry is doing ('' when nothing)."""
    if not status:
        return ""
    failed = status.get("failed") or {}
    names = ", ".join(str(k) for k in failed) or "a camera"
    n = int(status.get("attempts") or 0)
    if status.get("stopped") and status.get("headline"):  # v0.1.23
        return (
            f"{status['headline']}. Project Away stopped retrying {names} after {n} failed attempts so it does not "
            "slow your other camera down. Press Retry to try again, or Turn off to stop using it."
        )
    if status.get("stopped"):
        return (
            f"Project Away stopped retrying {names} after {n} failed attempts. Fix the camera (or remove it), "
            "then press Retry cameras."
        )
    nxt = status.get("next_s")
    when = f" next retry in about {float(nxt):.0f} s" if isinstance(nxt, (int, float)) else ""
    return f"Project Away is retrying {names} in the background ({n} failed so far;{when or ' retrying now'})."


def wait_for_retry(
    delay: float,
    done: threading.Event,
    stop: threading.Event,
    manual: Callable[[], bool],
    step: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Wait ``delay`` seconds (``float('inf')``: until asked) for the next camera retry. ``"time"`` when it is
    due, ``"manual"`` when "Retry cameras" was pressed (``manual()`` True), ``"done"`` when ``done`` or ``stop``
    was set."""
    deadline = clock() + delay
    while True:
        if done.is_set() or stop.is_set():
            return "done"
        if manual():
            return "manual"
        left = deadline - clock()
        if left <= 0:
            return "time"
        done.wait(min(step, left))
