"""Clickable on-screen elements ("targets") and a cache that never blocks the cursor loop.

A :class:`TargetBackend` lists the clickable elements of the foreground window (Windows: UI Automation, see
:mod:`away.pointer.uia`; tests: a fake). :class:`TargetCache` asks it on a background thread a few times a second
and publishes an immutable :class:`Snapshot`; the cursor loop only ever reads the latest snapshot (one attribute
read, no lock, no COM call), so a slow window (a browser with thousands of elements) costs nothing per frame.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

# UI Automation control types that count as clickable (names as in UIA_*ControlTypeId)
CLICKABLE_TYPES = (
    "button",
    "hyperlink",
    "edit",
    "menuitem",
    "listitem",
    "checkbox",
    "radiobutton",
    "tabitem",
    "treeitem",
    "combobox",
    "splitbutton",
    "dataitem",
)
MAX_TARGETS = 2000  # a huge page is cut off here
MIN_SIZE_PX = 4  # zero-size / collapsed elements are skipped


@dataclass(frozen=True)
class Target:
    """One clickable element. ``rect`` is ``(left, top, right, bottom)`` in desktop pixels."""

    name: str
    kind: str
    rect: tuple[float, float, float, float]
    ref: Any = field(default=None, compare=False, repr=False)  # backend handle (e.g. the UIA element) for focus

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.rect
        return (left + right) / 2.0, (top + bottom) / 2.0

    @property
    def width(self) -> float:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> float:
        return self.rect[3] - self.rect[1]

    def contains(self, x: float, y: float) -> bool:
        left, top, right, bottom = self.rect
        return left <= x <= right and top <= y <= bottom

    def distance(self, x: float, y: float) -> float:
        """Distance from a point to the rectangle (0 inside)."""
        left, top, right, bottom = self.rect
        dx = max(left - x, 0.0, x - right)
        dy = max(top - y, 0.0, y - bottom)
        return math.hypot(dx, dy)


class TargetBackend(Protocol):
    def available(self) -> bool: ...
    def foreground_targets(self) -> list[Target]: ...
    def focus(self, target: Target) -> bool: ...


class NullBackend:
    """No accessibility API (macOS, Linux, or comtypes missing): no targets, every feature is a no-op."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def available(self) -> bool:
        return False

    def foreground_targets(self) -> list[Target]:
        return []

    def focus(self, target: Target) -> bool:
        return False


def clean_targets(items: Iterable[Target], limit: int = MAX_TARGETS) -> tuple[Target, ...]:
    """Drop tiny / degenerate rectangles and cap the count."""
    out = []
    for t in items:
        if not all(math.isfinite(v) for v in t.rect):
            continue
        if t.width < MIN_SIZE_PX or t.height < MIN_SIZE_PX:
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return tuple(out)


@dataclass(frozen=True)
class Snapshot:
    targets: tuple[Target, ...] = ()
    t: float = 0.0  # when it was taken (monotonic)
    error: str = ""  # the last refresh failed with this (the previous targets are kept until stale)

    def near(self, x: float, y: float, radius: float) -> list[tuple[float, Target]]:
        """``(distance, target)`` for every target within ``radius`` of the point, nearest first; when targets
        overlap (a list item inside a list), the smaller one wins a tie."""
        hits = [(t.distance(x, y), t) for t in self.targets]
        hits = [h for h in hits if h[0] <= radius]
        hits.sort(key=lambda h: (h[0], h[1].width * h[1].height))
        return hits

    def nearest(self, x: float, y: float, radius: float = math.inf) -> Target | None:
        hits = self.near(x, y, radius)
        return hits[0][1] if hits else None


class TargetCache:
    """Refreshes :attr:`snapshot` from ``backend`` every ``period_s`` on a daemon thread.

    ``start()`` / ``stop()``; :meth:`refresh_now` runs one refresh synchronously (voice commands want the freshest
    tree; tests call it directly). A refresh error is kept in the snapshot and reported once through ``on_error``;
    targets older than ``stale_s`` are dropped so the magnet never pulls toward a window that has gone away.
    """

    def __init__(
        self,
        backend: TargetBackend,
        period_s: float = 0.25,
        stale_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        on_error: Callable[[str], None] | None = None,
        enabled: Callable[[], bool] = lambda: True,
    ) -> None:
        self.backend = backend
        self.period_s = period_s
        self.stale_s = stale_s
        self.clock = clock
        self.on_error = on_error
        self.enabled = enabled
        self._snap = Snapshot()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()  # one refresh at a time (background vs refresh_now)
        self._reported = False
        self.refreshes = 0

    @property
    def snapshot(self) -> Snapshot:
        snap = self._snap
        if snap.targets and self.clock() - snap.t > self.stale_s:
            return Snapshot((), snap.t, snap.error)
        return snap

    def refresh_now(self) -> Snapshot:
        with self._lock:
            now = self.clock()
            try:
                found = clean_targets(self.backend.foreground_targets())
            except Exception as exc:  # a COM error, a window closing mid-walk...
                msg = str(exc) or type(exc).__name__
                self._snap = Snapshot(self._snap.targets, self._snap.t, msg)
                if not self._reported and self.on_error is not None:
                    self._reported = True
                    with contextlib.suppress(Exception):
                        self.on_error(msg)
                return self.snapshot
            self._reported = False
            self._snap = Snapshot(found, now)
            self.refreshes += 1
            return self._snap

    def poke(self) -> None:
        """Refresh soon (focus changed, a voice command is coming)."""
        self._wake.set()

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        if self._thread is not None or not self.backend.available():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pointer-targets", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        th, self._thread = self._thread, None
        if th is not None:
            th.join(timeout)

    def _run(self) -> None:
        init = getattr(self.backend, "thread_init", None)
        if init is not None:
            try:
                init()
            except Exception as exc:
                log.warning("pointer targets: %s", exc)
                if self.on_error is not None:
                    with contextlib.suppress(Exception):
                        self.on_error(str(exc) or type(exc).__name__)
                return
        while not self._stop.is_set():
            if self.enabled():
                self.refresh_now()
            else:
                self._snap = Snapshot()
            self._wake.wait(self.period_s)
            self._wake.clear()
