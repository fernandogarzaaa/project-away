"""Click stabilisation and dwell detection (pure logic, no devices).

:class:`ClickStabilizer` ("steady click"): closing thumb and index drags the
index tip, which is the cursor anchor, so the click lands a few pixels off.
While the pinch ratio is closing (below ``pinch_on * (1 + margin)``) the cursor
is frozen, and the click is issued at the position the cursor had
``rewind_s`` before the pinch started closing. Idea from "Steady Clicks"
(Trewin, Keates & Moffatt, ASSETS 2006); implementation is original.

:class:`DwellDetector`: fires when the cursor rests inside ``radius_px`` for
``time_s`` (Camera Mouse style dwell click), then needs a move out of the
radius to re-arm.
"""

from __future__ import annotations

import math
from collections import deque


class ClickStabilizer:
    def __init__(self, rewind_s: float = 0.1, margin: float = 0.35, history_s: float = 1.5) -> None:
        self.rewind_s = rewind_s
        self.margin = margin
        self.history_s = history_s
        self.frozen = False
        self.closing_t: float | None = None  # when the current/last pinch started closing
        self.just_released = False  # set on unfreeze; the cursor re-anchors without a jump
        self._hist: deque[tuple[float, tuple[float, float]]] = deque()

    def reset(self) -> None:
        self.frozen = False
        self.closing_t = None
        self.just_released = False
        self._hist.clear()

    def record(self, t: float, pos: tuple[float, float]) -> None:
        """A cursor position that was actually sent to the OS."""
        self._hist.append((t, pos))
        while self._hist and t - self._hist[0][0] > self.history_s:
            self._hist.popleft()

    def observe(self, t: float, ratio: float, pinch_on: float, pinched: bool, dragging: bool) -> None:
        level = pinch_on * (1.0 + self.margin)
        if dragging:  # drags must move
            if self.frozen:
                self.frozen = False
                self.just_released = True
            self.closing_t = None
            return
        if not self.frozen and not pinched and ratio < level:
            self.frozen = True
            self.closing_t = t
        elif self.frozen and not pinched and ratio > level * 1.05:
            self.frozen = False
            self.just_released = True

    def position_at(self, t: float) -> tuple[float, float] | None:
        """Newest recorded position at or before ``t`` (oldest one if all are newer)."""
        if not self._hist:
            return None
        best = self._hist[0][1]
        for ts, pos in self._hist:
            if ts > t:
                break
            best = pos
        return best

    def click_position(self) -> tuple[float, float] | None:
        """Where the click should land, once per pinch (consumes the pinch start)."""
        if self.closing_t is None:
            return None
        pos = self.position_at(self.closing_t - self.rewind_s)
        self.closing_t = None
        return pos


class DwellDetector:
    def __init__(self, radius_px: float = 30.0, time_s: float = 0.8) -> None:
        self.radius_px = radius_px
        self.time_s = time_s
        self.reset()

    def reset(self) -> None:
        self._anchor: tuple[float, float] | None = None
        self._t0 = 0.0
        self._armed = True
        self.progress = 0.0

    def update(self, t: float, pos: tuple[float, float] | None) -> bool:
        """Feed the cursor position (``None`` = no pointing this frame). Returns True when a dwell fires."""
        if pos is None:
            self.reset()
            return False
        if self._anchor is None or math.dist(pos, self._anchor) > self.radius_px:
            self._anchor, self._t0, self._armed = pos, t, True
        self.progress = min(1.0, (t - self._t0) / self.time_s) if self._armed else 0.0
        if self._armed and t - self._t0 >= self.time_s:
            self._armed = False
            self.progress = 0.0
            return True
        return False


def pinch_progress(ratio: float | None, pinch_on: float, pinch_off: float) -> float:
    """0 = relaxed hand .. 1 = pinch closed; drives the at-cursor ring."""
    if ratio is None or not math.isfinite(ratio):
        return 0.0
    r_open = max(pinch_off * 1.6, pinch_on + 0.1)
    return float(min(1.0, max(0.0, (r_open - ratio) / (r_open - pinch_on))))
