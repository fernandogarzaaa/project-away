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
    """After ``limit`` failures within ``window_s`` a source's unauthenticated attempts are refused for ``lock_s``.

    v0.1.29 (audit fix): failures are counted per source (the caller passes the source identity it
    computed) instead of in one global list, and a successful sign-in (``ok``) clears that source's
    failures and any active lock. One peer spraying bad credentials can therefore only ever throttle
    its own unauthenticated attempts; it can never deny service to other sources, or to a caller who
    presents valid credentials (the middleware judges those on their merits and never consults the
    lock for them).
    """

    def __init__(
        self, limit: int = 10, window_s: float = 60.0, lock_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit, self.window_s, self.lock_s, self.clock = limit, window_s, lock_s, clock
        self._fails: dict[str, list[float]] = {}
        self._until: dict[str, float] = {}
        self._lock = threading.Lock()

    def locked(self, key: str = "") -> bool:
        with self._lock:
            return self.clock() < self._until.get(key, -1e18)

    def fail(self, key: str = "") -> None:
        now = self.clock()
        with self._lock:
            fails = [t for t in self._fails.get(key, []) if now - t < self.window_s] + [now]
            if len(fails) >= self.limit:
                self._until[key] = now + self.lock_s
                fails = []
            if fails:
                self._fails[key] = fails
            else:
                self._fails.pop(key, None)
            if len(self._fails) > 4096:  # never grow without bound (one key per client address)
                self._fails = dict(list(self._fails.items())[-1024:])
            if len(self._until) > 4096:
                self._until = {k: v for k, v in self._until.items() if v > now}

    def ok(self, key: str = "") -> None:
        """A successful sign-in from this source clears its failures and any active lock."""
        with self._lock:
            self._fails.pop(key, None)
            self._until.pop(key, None)

    def reset(self, key: str = "") -> None:
        with self._lock:
            self._fails.pop(key, None)
            self._until.pop(key, None)


class SourceBackoff:
    """v0.1.24: failed sign-ins per source (client address) with exponential backoff.

    The first ``free`` failures within ``window_s`` cost nothing; each one after that blocks the source for
    ``base_s * 2**n`` seconds (capped at ``max_s``). A success clears the source. Other sources are not affected, so
    a stranger guessing tokens cannot lock the person's own agents out.
    """

    def __init__(
        self,
        free: int = 5,
        base_s: float = 2.0,
        max_s: float = 900.0,
        window_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.free, self.base_s, self.max_s, self.window_s, self.clock = int(free), base_s, max_s, window_s, clock
        self._state: dict[str, tuple[int, float, float]] = {}  # source -> (failures, last failure, blocked until)
        self._lock = threading.Lock()

    def retry_after(self, key: str = "") -> float:
        """Seconds this source still has to wait (0 = it may try)."""
        now = self.clock()
        with self._lock:
            st = self._state.get(key)
            return max(0.0, st[2] - now) if st else 0.0

    def fail(self, key: str = "") -> float:
        """Count a failure; returns the wait it now has (0 while it is within the free attempts)."""
        now = self.clock()
        with self._lock:
            n, last, until = self._state.get(key, (0, now, 0.0))
            if now - last > self.window_s and now >= until:
                n = 0  # quiet for a while: start over
            n += 1
            wait = 0.0 if n <= self.free else min(self.max_s, self.base_s * (2 ** (n - self.free - 1)))
            self._state[key] = (n, now, now + wait if wait else until)
            if len(self._state) > 4096:
                self._state = dict(list(self._state.items())[-1024:])
            return wait

    def ok(self, key: str = "") -> None:
        with self._lock:
            self._state.pop(key, None)
