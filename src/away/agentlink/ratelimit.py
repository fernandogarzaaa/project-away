"""Rate limiting: a token bucket per key, and a lockout after repeated failed sign-ins."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rate = max(0.01, float(rate_per_s))
        self.burst = max(1, int(burst))
        self.clock = clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str = "") -> bool:
        now = self.clock()
        with self._lock:
            tokens, last = self._state.get(key, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.rate)
            ok = tokens >= 1.0
            self._state[key] = (tokens - 1.0 if ok else tokens, now)
            if len(self._state) > 4096:  # never grow without bound (one key per client address)
                self._state = dict(list(self._state.items())[-1024:])
            return ok

    def retry_after(self, key: str = "") -> float:
        with self._lock:
            tokens, _ = self._state.get(key, (float(self.burst), self.clock()))
        return max(0.0, (1.0 - tokens) / self.rate)


class FailureLockout:
    """After ``limit`` failures within ``window_s`` every attempt is refused for ``lock_s``."""

    def __init__(
        self, limit: int = 10, window_s: float = 60.0, lock_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit, self.window_s, self.lock_s, self.clock = limit, window_s, lock_s, clock
        self._fails: list[float] = []
        self._until = -1e18
        self._lock = threading.Lock()

    def locked(self) -> bool:
        with self._lock:
            return self.clock() < self._until

    def fail(self) -> None:
        now = self.clock()
        with self._lock:
            self._fails = [t for t in self._fails if now - t < self.window_s] + [now]
            if len(self._fails) >= self.limit:
                self._until = now + self.lock_s
                self._fails = []

    def reset(self) -> None:
        with self._lock:
            self._fails = []
