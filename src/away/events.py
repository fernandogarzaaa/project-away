"""A tiny thread-safe publish/subscribe event bus.

Every subsystem (hands, voice, face, wifi) publishes typed events here; the
fusion layer and the OS controllers subscribe. Keeping it synchronous makes
the whole pipeline deterministic in tests.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

WILDCARD = "*"


@dataclass(frozen=True)
class Event:
    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.monotonic)


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self, history: int = 256) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.RLock()
        self._history: list[Event] = []
        self._history_max = history
        self.errors: list[tuple[str, BaseException]] = []

    def has_subscribers(self, topic: str) -> bool:
        """Someone listens to ``topic`` itself (not only ``*``): e.g. the tray for ``app.notice`` (v0.1.21)."""
        with self._lock:
            return bool(self._subs.get(topic))

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        with self._lock:
            self._subs[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._subs.get(topic, []):
                    self._subs[topic].remove(handler)

        return unsubscribe

    def publish(self, topic: str, data: dict[str, Any] | None = None, ts: float | None = None) -> Event:
        ev = Event(topic, dict(data or {}), time.monotonic() if ts is None else ts)
        with self._lock:
            self._history.append(ev)
            if len(self._history) > self._history_max:
                del self._history[0 : len(self._history) - self._history_max]
            handlers = list(self._subs.get(topic, [])) + list(self._subs.get(WILDCARD, []))
        for h in handlers:
            try:
                h(ev)
            except Exception as exc:  # one bad subscriber must not kill the pipeline
                self.errors.append((topic, exc))
        return ev

    def history(self, topic: str | None = None) -> list[Event]:
        with self._lock:
            if topic is None:
                return list(self._history)
            return [e for e in self._history if e.topic == topic]
