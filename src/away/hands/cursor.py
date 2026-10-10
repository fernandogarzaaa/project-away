"""Map filtered hand positions to screen coordinates.

* ``absolute``: an "active region" box of the camera image maps linearly onto
  the screen (clamped). Simple, but arm fatigue is real.
* ``relative``: trackpad-style. Hand deltas move the cursor, with a dead zone
  that eats tremor and a power-law acceleration so slow moves are precise and
  fast moves cross the screen. Changing anchor (pointing -> pinch) or losing
  the hand re-anchors without a jump, like lifting a finger off a trackpad.
"""

from __future__ import annotations

import math

from away.config import CursorConfig
from away.hands.filters import OneEuro2D

ACCEL_REF = 0.01  # hand-units per frame at which acceleration is neutral


def accelerate(delta: float, exponent: float, ref: float = ACCEL_REF) -> float:
    """Scale a delta by (|delta|/ref)^(exponent-1), never below 1x."""
    mag = abs(delta)
    if mag == 0.0:
        return 0.0
    factor = max(1.0, (mag / ref) ** (exponent - 1.0))
    return delta * factor


def apply_dead_zone(dx: float, dy: float, dead_zone: float) -> tuple[float, float]:
    if math.hypot(dx, dy) < dead_zone:
        return 0.0, 0.0
    return dx, dy


class CursorMapper:
    def __init__(self, screen: tuple[int, int], cfg: CursorConfig | None = None) -> None:
        if screen[0] <= 0 or screen[1] <= 0:
            raise ValueError("screen size must be positive")
        self.cfg = cfg or CursorConfig()
        self.w, self.h = screen
        self.filter = OneEuro2D(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)
        self.pos = (self.w / 2.0, self.h / 2.0)
        self._prev: tuple[float, float] | None = None
        self._anchor: str | None = None

    def reset(self) -> None:
        self.filter.reset()
        self._prev = None
        self._anchor = None

    def set_position(self, x: float, y: float) -> None:
        self.pos = (self._clamp(x, self.w), self._clamp(y, self.h))

    @staticmethod
    def _clamp(v: float, hi: int) -> float:
        return min(max(v, 0.0), float(hi - 1))

    def update(self, t: float, x: float, y: float, anchor: str = "index") -> tuple[int, int]:
        if anchor != self._anchor:
            self.filter.reset()
            self._prev = None
            self._anchor = anchor
        fx, fy = self.filter(t, x, y)
        if self.cfg.mode == "absolute":
            return self._absolute(fx, fy)
        return self._relative(fx, fy)

    def _absolute(self, fx: float, fy: float) -> tuple[int, int]:
        x0, y0, x1, y1 = self.cfg.active_region
        u = (fx - x0) / (x1 - x0)
        v = (fy - y0) / (y1 - y0)
        target = (self._clamp(u * (self.w - 1), self.w), self._clamp(v * (self.h - 1), self.h))
        if self._prev is not None:
            ddx, ddy = apply_dead_zone(fx - self._prev[0], fy - self._prev[1], self.cfg.dead_zone)
            if ddx == 0.0 and ddy == 0.0:
                return round(self.pos[0]), round(self.pos[1])
        self._prev = (fx, fy)
        self.pos = target
        return round(self.pos[0]), round(self.pos[1])

    def _relative(self, fx: float, fy: float) -> tuple[int, int]:
        if self._prev is None:
            self._prev = (fx, fy)
            return round(self.pos[0]), round(self.pos[1])
        dx, dy = apply_dead_zone(fx - self._prev[0], fy - self._prev[1], self.cfg.dead_zone)
        if dx == 0.0 and dy == 0.0:
            return round(self.pos[0]), round(self.pos[1])
        self._prev = (fx, fy)
        sx = accelerate(dx, self.cfg.acceleration) * self.cfg.gain * self.w
        sy = accelerate(dy, self.cfg.acceleration) * self.cfg.gain * self.w  # same scale both axes
        self.set_position(self.pos[0] + sx, self.pos[1] + sy)
        return round(self.pos[0]), round(self.pos[1])
