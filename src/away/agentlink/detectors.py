"""Auto-detectors: notice trouble while the app runs and keep the evidence.

* ``fps_drop`` - a camera that streamed fine drops below ``fps_drop_ratio`` x its usual rate for 5 s
* ``open_failure`` - a camera could not be opened (``camera.error`` / ``cameras.degraded``)
* ``retry_loop`` - three or more camera failures within 2 minutes
* ``crash`` - crash.log grew, or a subscriber on the event bus raised
* ``misfire`` - ``misfire_threshold`` or more "that was wrong" / cancelled / undone actions within a minute

When one trips, the host saves a diagnostics bundle (with the last 30 s of landmarks) and tells connected agents
(event ``detector``); an optional, off-by-default GitHub issue can be filed with a token from the OS keyring.
Each detector has a cooldown (default 10 minutes) so a persistent problem does not fill the disk.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from collections.abc import Callable
from typing import Any

MISFIRE_TOPICS = ("action.cancelled", "action.undone", "feedback.wrong", "agent.wrong")
CAMERA_FAIL_TOPICS = ("camera.error", "cameras.degraded")


class Detectors:
    def __init__(
        self,
        on_trip: Callable[[str, dict[str, Any]], None],
        *,
        fps_drop_ratio: float = 0.5,
        misfire_threshold: int = 4,
        cooldown_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.on_trip = on_trip
        self.ratio = float(fps_drop_ratio)
        self.misfire_threshold = int(misfire_threshold)
        self.cooldown_s = float(cooldown_s)
        self.clock = clock
        self._last_trip: dict[str, float] = {}
        self._baseline: dict[str, float] = {}
        self._good_since: dict[str, float] = {}
        self._low_since: dict[str, float] = {}
        self._fails: deque[float] = deque(maxlen=50)
        self._misfires: deque[float] = deque(maxlen=50)
        self._crash_size: int | None = None
        self._bus_errors = 0
        self.tripped: list[tuple[float, str, dict[str, Any]]] = []

    def _trip(self, name: str, detail: dict[str, Any]) -> bool:
        now = self.clock()
        if now - self._last_trip.get(name, -1e18) < self.cooldown_s:
            return False
        self._last_trip[name] = now
        self.tripped.append((now, name, detail))
        with contextlib.suppress(Exception):
            self.on_trip(name, detail)
        return True

    def on_event(self, topic: str, data: dict[str, Any]) -> None:
        now = self.clock()
        if topic in CAMERA_FAIL_TOPICS:
            self._fails.append(now)
            recent = [t for t in self._fails if now - t <= 120.0]
            if len(recent) >= 3:
                self._trip("retry_loop", {"failures_in_2_min": len(recent), "last": str(data)[:300]})
            else:
                self._trip("open_failure", {"topic": topic, "detail": str(data)[:300]})
        elif topic in MISFIRE_TOPICS:
            self._misfires.append(now)
            recent = [t for t in self._misfires if now - t <= 60.0]
            if len(recent) >= self.misfire_threshold:
                self._trip("misfire", {"in_last_minute": len(recent), "last": topic})

    def tick(self, fps: dict[str, float], crash_size: int | None = None, bus_errors: int = 0) -> None:
        now = self.clock()
        for cam, f in fps.items():
            base = self._baseline.get(cam, 0.0)
            if f >= max(5.0, base * 0.8):
                self._good_since.setdefault(cam, now)
                if now - self._good_since[cam] >= 10.0:
                    self._baseline[cam] = max(base, f)
                self._low_since.pop(cam, None)
            elif base >= 10.0 and f < base * self.ratio:
                self._good_since.pop(cam, None)
                start = self._low_since.setdefault(cam, now)
                if now - start >= 5.0:
                    self._trip("fps_drop", {"camera": cam, "fps": f, "usual_fps": base})
                    self._low_since[cam] = now
            else:
                self._low_since.pop(cam, None)
        if crash_size is not None:
            if self._crash_size is not None and crash_size > self._crash_size:
                self._trip("crash", {"source": "crash.log", "grew_bytes": crash_size - self._crash_size})
            self._crash_size = crash_size
        if bus_errors > self._bus_errors:
            self._trip("crash", {"source": "event bus", "new_errors": bus_errors - self._bus_errors})
        self._bus_errors = max(self._bus_errors, bus_errors)


def file_github_issue(
    repo: str, title: str, body: str, token: str, opener: Any = None, timeout: float = 15.0
) -> str:  # pragma: no cover - network
    """POST an issue; returns its URL. The body must already be free of private data (it is: stats only)."""
    import json
    import urllib.request

    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({"title": title[:200], "body": body[:20000], "labels": ["auto-detector"]}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "project-away",
        },
        method="POST",
    )
    with (opener or urllib.request.urlopen)(req, timeout=timeout) as resp:
        return str(json.loads(resp.read().decode("utf-8")).get("html_url", ""))


KEYRING_SERVICE = "ProjectAway"
KEYRING_USER = "github-issues"


def github_token() -> str | None:
    """The user's GitHub token from the OS keyring (``away agent github-token set``), or None."""
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception:
        return None
