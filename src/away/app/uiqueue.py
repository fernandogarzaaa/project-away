"""Run Tk calls on the Tk thread (v0.1.16).

Tk is not thread-safe: a worker thread (a camera test, a download, an event-bus publisher) must never touch a
widget, not even ``widget.after(0, ...)``. Workers :meth:`UiQueue.post` a callable instead; the Tk thread runs
them from its own ``after`` loop with :meth:`UiQueue.drain` (or :meth:`UiQueue.pump`). One of the suspects for the
first Settings > Test both cameras run that took the whole app down.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger("away.ui")


class UiQueue:
    def __init__(self) -> None:
        self._q: queue.Queue[tuple[Callable[..., Any], tuple[Any, ...]]] = queue.Queue()
        self.owner = threading.get_ident()  # the Tk thread (the one that made the queue)

    def post(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` on the Tk thread soon. Safe from any thread."""
        self._q.put((fn, args))

    def drain(self, limit: int = 100) -> int:
        """Run what was posted (on the Tk thread); returns how many ran. An exception in one is logged."""
        n = 0
        while n < limit:
            try:
                fn, args = self._q.get_nowait()
            except queue.Empty:
                break
            n += 1
            try:
                fn(*args)
            except Exception:
                log.exception("UI callback failed")
        return n

    def pump(self, widget: Any, every_ms: int = 50) -> None:
        """Drain now and again every ``every_ms`` on ``widget``'s Tk loop (call once, on the Tk thread)."""
        self.drain()
        with contextlib.suppress(Exception):
            widget.after(every_ms, self.pump, widget, every_ms)
