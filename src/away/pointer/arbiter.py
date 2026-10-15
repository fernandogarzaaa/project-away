"""One cursor source at a time (v0.1.26): the hand, the head or the gaze, never two at once.

Why the cursor jittered with eye/head tracking on (found after v0.1.25): the face path (head pointer, head zones,
head-turn monitor switch, look-to-jump) wrote the OS cursor *directly*, next to the hand path:

* the head pointer only checked ``hand_present``, which drops on a single missed hand detection (common at
  640x480), so during every short detection dropout the head moved the cursor and the hand moved it back;
* head zones (``head_assist: coarse``) and the monitor switch warped the cursor while the hand was pointing;
* look-to-jump treated landmark noise on a resting hand (one frame-to-frame step over 0.006 of the frame) as
  "the hand started moving" and warped the cursor to the (noisy) gaze point;
* those writes bypassed the 120 Hz output glide, so the glide's next tick pulled the cursor back to the hand's
  old target: a visible flick on every face-driven write.

:class:`CursorArbiter` decides who may move the cursor. The hand owns it from its first pointing frame until it
has been gone for ``hand_hold_s`` (so detection dropouts never hand the cursor to the head), and while it owns
it the head and gaze only observe. It also measures each source's frame rate and the hand/face model time, and
publishes ``cursor.source`` when the driver changes and ``cursor.stats`` every ``stats_every_s`` seconds, so
``away.log`` shows which source drove the cursor and how fast each one ran.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

HAND_HOLD_S = 0.5  # the hand keeps the cursor this long after its last frame (detection dropouts)
CLAIM_HOLD_S = 0.5  # a head/gaze claim counts as "driving" this long after its last cursor write
FACE_HZ_WHILE_HAND = 15.0  # face model rate while the hand drives (it only feeds the speech gate / face commands)
SOURCES = ("hand", "face", "head", "gaze")


class RateMeter:
    """Frames per second over the last ``window_s`` seconds of timestamps."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = float(window_s)
        self._ts: deque[float] = deque()

    def tick(self, t: float) -> None:
        self._ts.append(t)
        self._trim(t)

    def _trim(self, now: float) -> None:
        while self._ts and now - self._ts[0] > self.window_s:
            self._ts.popleft()

    def fps(self, now: float) -> float:
        self._trim(now)
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


class CursorArbiter:
    def __init__(
        self,
        publish: Callable[[str, dict[str, Any]], Any] | None = None,
        hand_hold_s: float = HAND_HOLD_S,
        stats_every_s: float = 5.0,
    ) -> None:
        self.publish = publish or (lambda _k, _d: None)
        self.hand_hold_s = float(hand_hold_s)
        self.stats_every_s = float(stats_every_s)
        self._lock = threading.Lock()
        self.meters = {s: RateMeter() for s in SOURCES}
        self.cost_ms: dict[str, float] = {}
        self._hand_t: float | None = None  # last frame with a pointing hand
        self._claim: tuple[str, float] | None = None  # last head/gaze cursor write
        self.owner = "none"
        self.writes = {"hand": 0, "head": 0, "gaze": 0}
        self.denied = {"head": 0, "gaze": 0}
        self._stats_t: float | None = None
        self._last_face_t: float | None = None

    # ------------------------------------------------------------------ observations
    def note_hand(self, t: float, present: bool) -> None:
        """Every hand frame (``present``: a pointer hand was tracked)."""
        with self._lock:
            if present:
                self.meters["hand"].tick(t)
                self._hand_t = t
        self._update(t)

    def note_face(self, t: float, pose: bool = False, gaze: bool = False) -> None:
        with self._lock:
            self.meters["face"].tick(t)
            if pose:
                self.meters["head"].tick(t)
            if gaze:
                self.meters["gaze"].tick(t)
        self._update(t)

    def note_cost(self, source: str, ms: float) -> None:
        """Model time per frame (ms), smoothed: shows when the face model eats into the hand's frame budget."""
        with self._lock:
            old = self.cost_ms.get(source)
            self.cost_ms[source] = ms if old is None else 0.8 * old + 0.2 * ms

    # ------------------------------------------------------------------ decisions
    def hand_owns(self, t: float) -> bool:
        with self._lock:
            return self._hand_owns(t)

    def _hand_owns(self, t: float) -> bool:
        return self._hand_t is not None and -1e-6 <= t - self._hand_t <= self.hand_hold_s

    def claim(self, t: float, source: str) -> bool:
        """May ``source`` move the cursor now? The hand always may; the head and gaze only while no hand drives."""
        with self._lock:
            if source == "hand":
                self._hand_t = t if self._hand_t is None else max(self._hand_t, t)
                self.writes["hand"] += 1
                ok = True
            elif self._hand_owns(t):
                self.denied[source] = self.denied.get(source, 0) + 1
                ok = False
            else:
                self._claim = (source, t)
                self.writes[source] = self.writes.get(source, 0) + 1
                ok = True
        self._update(t)
        return ok

    def face_due(self, t: float) -> bool:
        """Run the face model on this frame? Always while the hand is not driving; at most FACE_HZ_WHILE_HAND
        while it is (the face model then only feeds the speech gate and face commands), so it does not stretch
        the hand's frame time."""
        with self._lock:
            gap = None if self._last_face_t is None else t - self._last_face_t
            if self._hand_owns(t) and gap is not None and gap < 1.0 / FACE_HZ_WHILE_HAND - 0.005:
                return False
            self._last_face_t = t
            return True

    # ------------------------------------------------------------------ reporting
    def _current(self, t: float) -> str:
        if self._hand_owns(t):
            return "hand"
        if self._claim is not None and 0 <= t - self._claim[1] <= CLAIM_HOLD_S:
            return self._claim[0]
        return "none"

    def _update(self, t: float) -> None:
        with self._lock:
            now = self._current(t)
            changed = now != self.owner
            was, self.owner = self.owner, now
            due = self._stats_t is None or t - self._stats_t >= self.stats_every_s
            if due:
                self._stats_t = t
            stats = self._stats(t) if due else None
        if changed:
            self.publish("cursor.source", {"source": now, "was": was})
        if stats is not None:
            self.publish("cursor.stats", stats)

    def _stats(self, t: float) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.owner}
        for s in SOURCES:
            out[f"{s}_fps"] = round(self.meters[s].fps(t), 1)
        for s, ms in self.cost_ms.items():
            out[f"{s}_ms"] = round(ms, 1)
        out["writes"] = dict(self.writes)
        out["denied"] = dict(self.denied)
        return out

    def stats(self, t: float) -> dict[str, Any]:
        with self._lock:
            return self._stats(t)


__all__ = ["CLAIM_HOLD_S", "FACE_HZ_WHILE_HAND", "HAND_HOLD_S", "CursorArbiter", "RateMeter"]
