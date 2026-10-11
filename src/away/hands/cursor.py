"""Map filtered hand positions to screen coordinates.

* ``absolute``: an "active region" box of the camera image maps linearly onto
  the screen (clamped). Simple, but arm fatigue is real.
* ``relative``: trackpad-style. Hand deltas move the cursor, with a dead zone
  that eats tremor and an acceleration curve so slow moves are precise and
  fast moves cross the screen. Two curves: ``power`` (|d|^k, the original) and
  ``sigmoid`` (a logistic gain between a low and a high plateau), plus optional
  per-direction speed multipliers. Changing anchor (pointing -> pinch) or losing
  the hand re-anchors without a jump, like lifting a finger off a trackpad.

Multi-monitor: the cursor lives on the virtual desktop (the bounding box of the selected monitors, whose
origin can be negative) and every position is clamped onto the *union* of the monitor rectangles, so it
never parks in a dead zone between monitors of different sizes. Relative-mode speed is relative to the
primary monitor's width, so slow precise moves feel exactly like on one screen; only fast moves get extra
reach, ``(desktop span / primary span) ** monitor_reach`` (3 equal monitors, default 0.5: ~1.73x).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from away.config import CursorConfig
from away.control.monitors import (
    Layout,
    Monitor,
    clamp_to_monitors,
    nearest_monitor,
    select_monitors,
    sort_monitors,
)
from away.hands.filters import OneEuro2D

ACCEL_REF = 0.01  # hand-units per frame at which acceleration is neutral

Screen = tuple[int, int] | Layout | Sequence[Monitor]


def accel_factor(mag: float, exponent: float, ref: float = ACCEL_REF) -> float:
    """(|delta|/ref)^(exponent-1), never below 1x."""
    if mag == 0.0:
        return 1.0
    return max(1.0, (mag / ref) ** (exponent - 1.0))


def accelerate(delta: float, exponent: float, ref: float = ACCEL_REF, reach: float = 1.0) -> float:
    """Scale a delta by (|delta|/ref)^(exponent-1), never below 1x. ``reach`` multiplies only the part
    above 1x, so slow moves are unchanged and fast moves travel further (multi-monitor)."""
    mag = abs(delta)
    if mag == 0.0:
        return 0.0
    return delta * (1.0 + (accel_factor(mag, exponent, ref) - 1.0) * reach)


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


def as_monitors(screen: Screen) -> tuple[Monitor, ...]:
    """``(w, h)`` (one screen at the origin, the pre-0.1.6 form), a :class:`Layout`, or monitors."""
    if isinstance(screen, Layout):
        return screen.monitors
    if isinstance(screen, tuple) and len(screen) == 2 and all(isinstance(v, (int, float)) for v in screen):
        w, h = (float(v) for v in screen)  # type: ignore[arg-type]
        if w <= 0 or h <= 0:
            raise ValueError("screen size must be positive")
        return (Monitor(0, 0, int(w), int(h), True, "primary"),)
    mons = sort_monitors(screen)  # type: ignore[arg-type]
    if not mons:
        raise ValueError("at least one monitor with a positive size is needed")
    return mons


def reach_factor(mons: Sequence[Monitor], ref: Monitor, exponent: float) -> float:
    """How much further fast moves go on a desktop larger than the reference monitor."""
    x0 = min(m.left for m in mons)
    y0 = min(m.top for m in mons)
    span = max((max(m.right for m in mons) - x0) / ref.width, (max(m.bottom for m in mons) - y0) / ref.height)
    return float(max(1.0, span) ** max(0.0, exponent))


class CursorMapper:
    def __init__(self, screen: Screen, cfg: CursorConfig | None = None) -> None:
        self.cfg = cfg or CursorConfig()
        self.filter = OneEuro2D(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)
        self.all_monitors: tuple[Monitor, ...] = ()
        self.monitors: tuple[Monitor, ...] = ()
        self.pos: tuple[float, float] = (0.0, 0.0)
        self.set_monitors(screen, keep_position=False)
        self._prev: tuple[float, float] | None = None
        self._anchor: str | None = None

    # ------------------------------------------------------------------ monitors
    def set_monitors(self, screen: Screen, keep_position: bool = True) -> None:
        """Use a new monitor layout (hot-plug, config change). The cursor stays where it is when that is
        still on a monitor, else it moves to the nearest point that is."""
        self.all_monitors = as_monitors(screen)
        mons = select_monitors(Layout(self.all_monitors, "mapper"), getattr(self.cfg, "monitors", "all"))
        self.monitors = mons
        self.ref = next((m for m in mons if m.primary), mons[0])
        self.x0 = min(m.left for m in mons)
        self.y0 = min(m.top for m in mons)
        self.w = max(m.right for m in mons) - self.x0
        self.h = max(m.bottom for m in mons) - self.y0
        self.reach = reach_factor(mons, self.ref, float(getattr(self.cfg, "monitor_reach", 0.5)))
        if keep_position:
            self.pos = clamp_to_monitors(self.pos[0], self.pos[1], mons)
        else:
            self.pos = self.ref.center

    def refresh_selection(self) -> None:
        """Re-apply ``cfg.monitors`` / ``cfg.monitor_reach`` after a live config change."""
        self.set_monitors(self.all_monitors, keep_position=True)

    def current_monitor(self) -> int:
        """0-based index into :attr:`monitors` of the monitor under the cursor."""
        return nearest_monitor(self.pos[0], self.pos[1], self.monitors)

    def jump(self, target: str) -> tuple[int, int] | None:
        """``next`` | ``prev`` | ``primary`` | ``N`` (1-based number from `away monitors`): put the cursor in
        the centre of that monitor. Returns the new position, or None when there is no such monitor (or it
        is excluded by ``hands.cursor.monitors``)."""
        mons = self.monitors
        if target in ("next", "prev"):
            i = self.current_monitor()
            m = mons[(i + (1 if target == "next" else -1)) % len(mons)]
        elif target == "primary":
            m = self.ref
        else:
            try:
                n = int(target)
            except ValueError:
                return None
            if not 1 <= n <= len(self.all_monitors) or self.all_monitors[n - 1] not in mons:
                return None
            m = self.all_monitors[n - 1]
        self.pos = m.center
        self.reset()  # re-anchor: the next hand move continues from the new spot, no jump back
        return round(self.pos[0]), round(self.pos[1])

    # ------------------------------------------------------------------ mapping
    def reset(self) -> None:
        self.filter.reset()
        self._prev = None
        self._anchor = None

    def set_position(self, x: float, y: float) -> None:
        self.pos = clamp_to_monitors(x, y, self.monitors)

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
        u = min(max((fx - x0) / (x1 - x0), 0.0), 1.0)
        v = min(max((fy - y0) / (y1 - y0), 0.0), 1.0)
        target = clamp_to_monitors(self.x0 + u * (self.w - 1), self.y0 + v * (self.h - 1), self.monitors)
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
            # one gain from the 2D speed so diagonal moves keep their direction; the extra multi-monitor
            # reach raises only the fast plateau, the slow (precise) plateau is unchanged
            high = c.sigmoid_low + (c.sigmoid_high - c.sigmoid_low) * self.reach
            k = sigmoid_gain(math.hypot(dx, dy), c.sigmoid_low, high, c.sigmoid_mid, c.sigmoid_slope)
            ax, ay = dx * k, dy * k
        else:
            ax = accelerate(dx, c.acceleration, reach=self.reach)
            ay = accelerate(dy, c.acceleration, reach=self.reach)
        ax, ay = direction_scale(ax, ay, c.speed_up, c.speed_down, c.speed_left, c.speed_right)
        scale = c.gain * self.ref.width  # relative to the primary monitor, not the whole desktop
        self.set_position(self.pos[0] + ax * scale, self.pos[1] + ay * scale)
        return round(self.pos[0]), round(self.pos[1])
