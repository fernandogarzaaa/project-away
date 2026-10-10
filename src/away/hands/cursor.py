"""Map filtered hand positions to screen coordinates.

* ``absolute``: an "active region" box of the camera image maps linearly onto
  the screen (clamped). Simple, but arm fatigue is real.
* ``relative``: trackpad-style. Hand deltas move the cursor, with a dead zone
  that eats tremor and an acceleration curve so slow moves are precise and
  fast moves cross the screen. Two curves: ``power`` (|d|^k, the original) and
  ``sigmoid`` (a logistic gain between a low and a high plateau), plus optional
  per-direction speed multipliers. Changing anchor (pointing -> pinch) or losing
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


def sigmoid_gain(mag: float, low: float, high: float, mid: float, slope: float, ref: float = ACCEL_REF) -> float:
    """Gain multiplier for a per-frame delta of size ``mag``: ``low`` for tiny moves, ``high`` for fast ones.

    Clean-room implementation of the idea behind Project Gameface's sigmoid acceleration
    (https://github.com/google/project-gameface, Apache-2.0): a logistic curve centred at ``mid`` (in units
    of ``ref``) with steepness ``slope``. No code was copied.
    """
    x = mag / ref
    z = max(-60.0, min(60.0, -slope * (x - mid)))
    return low + (high - low) / (1.0 + math.exp(z))


def accelerate_sigmoid(
    delta: float, low: float, high: float, mid: float, slope: float, ref: float = ACCEL_REF
) -> float:
    if delta == 0.0:
        return 0.0
    return delta * sigmoid_gain(abs(delta), low, high, mid, slope, ref)


def direction_scale(dx: float, dy: float, up: float, down: float, left: float, right: float) -> tuple[float, float]:
    """Per-direction speed (screen y grows downwards): ``up`` scales negative dy, ``right`` positive dx..."""
    return dx * (right if dx > 0 else left), dy * (down if dy > 0 else up)


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
        c = self.cfg
        if c.curve == "sigmoid":
            # one gain from the 2D speed so diagonal moves keep their direction
            k = sigmoid_gain(math.hypot(dx, dy), c.sigmoid_low, c.sigmoid_high, c.sigmoid_mid, c.sigmoid_slope)
            ax, ay = dx * k, dy * k
        else:
            ax, ay = accelerate(dx, c.acceleration), accelerate(dy, c.acceleration)
        ax, ay = direction_scale(ax, ay, c.speed_up, c.speed_down, c.speed_left, c.speed_right)
        sx = ax * c.gain * self.w
        sy = ay * c.gain * self.w  # same scale both axes
        self.set_position(self.pos[0] + sx, self.pos[1] + sy)
        return round(self.pos[0]), round(self.pos[1])
