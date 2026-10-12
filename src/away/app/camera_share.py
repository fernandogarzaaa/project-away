"""Hand the webcam between the running app and the Settings window (separate processes).

Windows cameras are exclusive: while the tray app tracks your hand, Settings -> Test gesture, the camera
list probe and camera calibration cannot open the same device. This is a tiny file-based protocol in the
data folder (no sockets, nothing leaves the PC):

* ``camera-owner.json``     the app holds cameras ``[0]`` (pid, heartbeat every ~2 s, removed on exit)
* ``camera-request.json``   another process wants them (pid, owner, heartbeat every ~1 s while it needs them)
* ``camera-released.json``  the app has closed its cameras for that request (removed when it reopens them)

A request or owner file whose heartbeat is older than ``stale_s`` is ignored, so a crashed Settings window
can never keep the app's camera closed for more than a few seconds.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

OWNER_FILE = "camera-owner.json"
REQUEST_FILE = "camera-request.json"
RELEASED_FILE = "camera-released.json"

IN_USE_MESSAGE = (
    "the camera is in use by Project Away or another app - pause or quit it (or close the other app) and retry"
)


class CameraShare:
    def __init__(
        self,
        folder: Path | None = None,
        stale_s: float = 5.0,
        clock: Callable[[], float] = time.time,
        pid: int | None = None,
    ) -> None:
        if folder is None:
            from away import paths

            folder = paths.data_dir()
        self.folder = Path(folder)
        self.stale_s = stale_s
        self.clock = clock
        self.pid = os.getpid() if pid is None else pid

    # ------------------------------------------------------------------ files
    def _path(self, name: str) -> Path:
        return self.folder / name

    def _write(self, name: str, data: dict[str, Any]) -> None:
        with contextlib.suppress(OSError):
            self.folder.mkdir(parents=True, exist_ok=True)
            p = self._path(name)
            tmp = p.with_suffix(f".{self.pid}.tmp")
            tmp.write_text(json.dumps({**data, "pid": self.pid, "t": self.clock()}), encoding="utf-8")
            os.replace(tmp, p)

    def _read(self, name: str) -> dict[str, Any] | None:
        """The file's content when it exists and its heartbeat is fresh, else None."""
        try:
            data = json.loads(self._path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        t = data.get("t")
        if not isinstance(t, (int, float)) or self.clock() - float(t) > self.stale_s:
            return None
        return data

    def _remove(self, name: str, only_mine: bool = True) -> None:
        if only_mine:
            try:
                data = json.loads(self._path(name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            if isinstance(data, dict) and data.get("pid") not in (None, self.pid):
                return
        with contextlib.suppress(OSError):
            self._path(name).unlink()

    # ------------------------------------------------------------- app side
    def publish_owner(self, cameras: list[Any]) -> None:
        self._write(OWNER_FILE, {"cameras": [str(c) for c in cameras]})

    def clear_owner(self) -> None:
        self._remove(OWNER_FILE)

    def requested(self) -> bool:
        """Another process (Settings) asked for the cameras and is still alive."""
        req = self._read(REQUEST_FILE)
        return req is not None and req.get("pid") != self.pid

    def ack_released(self, cameras: list[Any]) -> None:
        self._write(RELEASED_FILE, {"cameras": [str(c) for c in cameras]})

    def clear_released(self) -> None:
        self._remove(RELEASED_FILE)

    # ------------------------------------------------------- requester side
    def owner(self) -> dict[str, Any] | None:
        """The running app's owner record (``{"pid", "cameras": ["0"], "t"}``) or None."""
        own = self._read(OWNER_FILE)
        return own if own is not None and own.get("pid") != self.pid else None

    def owned_devices(self) -> set[str]:
        own = self.owner()
        return {str(c) for c in own.get("cameras", [])} if own else set()

    def request(self, who: str = "settings") -> None:
        self._write(REQUEST_FILE, {"owner": who})

    def release(self) -> None:
        self._remove(REQUEST_FILE)

    def released(self) -> bool:
        return self._read(RELEASED_FILE) is not None

    def wait_released(
        self,
        timeout: float = 4.0,
        sleep: Callable[[float], Any] = time.sleep,
        step: float = 0.1,
        who: str = "settings",
    ) -> bool:
        """After :meth:`request`: True once the app has closed its cameras (False on timeout)."""
        deadline = self.clock() + timeout
        while True:
            if self.released():
                return True
            if self.clock() >= deadline:
                return False
            self.request(who)  # keep the heartbeat fresh while waiting
            sleep(step)


class CameraLease:
    """Borrow the cameras from the running app for as long as the lease is held.

    ``with CameraLease() as lease:`` -> ``lease.handed_over`` tells whether the app had them and let go;
    ``lease.app_running`` whether an app owned them at all. A heartbeat thread keeps the request fresh;
    leaving the block (or :meth:`close`) lets the app reopen them.
    """

    def __init__(
        self,
        share: CameraShare | None = None,
        timeout: float = 4.0,
        who: str = "settings",
        heartbeat_s: float = 1.0,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        self.share = share or CameraShare()
        self.timeout = timeout
        self.who = who
        self.heartbeat_s = heartbeat_s
        self.sleep = sleep
        self.app_running = False
        self.handed_over = False
        self.nested = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> CameraLease:
        """Ask for the cameras. The request is kept fresh for the whole lease even when no app holds them
        now (v0.1.15), so an app that starts, or retries its cameras, meanwhile leaves them alone."""
        self.app_running = self.share.owner() is not None
        if not self.app_running and self.share.requested():
            # another live process already holds the cameras for us (Settings runs `away calibrate-cameras`
            # while holding its own lease): leave its request alone, so closing ours cannot drop it
            self.nested = True
            return self
        self.share.request(self.who)
        self._thread = threading.Thread(target=self._beat, daemon=True, name="camera-lease")
        self._thread.start()
        if self.app_running:
            self.handed_over = self.share.wait_released(self.timeout, self.sleep, who=self.who)
        return self

    @property
    def blocked(self) -> bool:
        """The app holds the cameras and did not let go in time: opening them will fail."""
        return self.app_running and not self.handed_over

    def _beat(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            self.share.request(self.who)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if not self.nested:
            self.share.release()

    def __enter__(self) -> CameraLease:
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def camera_error_message(exc: BaseException, device: Any, app_running: bool, handed_over: bool) -> str:
    """What to tell the user when a camera would not open (instead of a bare 'permission denied')."""
    text = str(exc)
    if "cannot open camera" not in text and "could not be opened" not in text:
        return f"camera error: {text}"
    if app_running and not handed_over:
        return (
            f"camera {device}: Project Away is using it and did not hand it over in time - pause or quit it and retry"
        )
    return (
        f"camera {device}: cannot open it - {IN_USE_MESSAGE}; "
        "if no other app uses it, check the camera privacy settings"
    )


@contextlib.contextmanager
def borrowed_cameras(
    share: CameraShare | None = None, timeout: float = 4.0, who: str = "settings"
) -> Iterator[CameraLease]:
    """``with borrowed_cameras(who="calibrate") as lease:`` - a :class:`CameraLease` for a block."""
    lease = CameraLease(share, timeout, who)
    try:
        yield lease.open()
    finally:
        lease.close()


class SharedLease:
    """One :class:`CameraLease` shared by several users in one process (Settings: Live confidence, Test both
    cameras and Calibrate). The first :meth:`acquire` opens it, the last :meth:`release` closes it, so handing
    the cameras from one tool to the next never lets the app grab them in between."""

    def __init__(self, factory: Callable[[str], CameraLease] | None = None) -> None:
        self._factory = factory or (lambda who: CameraLease(who=who))
        self._lock = threading.Lock()
        self._lease: CameraLease | None = None
        self._users = 0

    def acquire(self, who: str = "settings") -> CameraLease:
        with self._lock:
            if self._lease is None:
                self._lease = self._factory(who).open()
            self._users += 1
            return self._lease

    def release(self) -> None:
        with self._lock:
            if self._users <= 0:
                return
            self._users -= 1
            if self._users == 0 and self._lease is not None:
                self._lease.close()
                self._lease = None

    @property
    def held(self) -> bool:
        return self._lease is not None
