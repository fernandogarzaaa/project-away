"""Auto-centering for the hand cursor (pure maths, no devices).

Three helpers keep the pointing hand comfortably inside the camera frame (``hands.cursor.autocenter``):

* :func:`soft_gain` (relative mode): a stroke *away* from the target gets slightly more gain and a stroke
  *back* slightly less. For any closed cursor trip (out and back by the same screen distance) the hand then
  travels a little further inward than outward, so its resting spot drifts toward the target over many
  strokes. The cursor itself never jumps; the effect is a few percent per stroke (imperceptible).
* :class:`ComfortZone`: an exponential moving average of where the hand *rests* (slow frames only). Absolute
  mode centres its active region there (:func:`shift_region`); relative mode can use it as the soft-recenter
  target (``target: comfort``), kept inside the frame.
* :class:`EdgeClutch`: when the hand reaches the edge band of the camera frame and swings back quickly, the
  swing back does not drag the cursor (like lifting a mouse at the edge of the mouse pad); the cursor
  re-anchors when the hand slows down or is back in the inner part of the frame. Slow moves near the edge
  (fine adjustments) are never clutched.

Coordinates are normalised camera coordinates (0..1, x right, y down), speeds in frame widths per second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

FRAME_CENTER = (0.5, 0.5)


def soft_gain(
    pos: tuple[float, float], delta: tuple[float, float], target: tuple[float, float], strength: float
) -> float:
    """Gain multiplier for a hand ``delta`` taken at ``pos``: ``1 + strength * c * r``.

    ``c`` is the cosine between the motion and the direction *away* from ``target`` (+1 straight out, -1
    straight back) and ``r`` the distance from the target relative to the distance to the frame edge in that
    direction (0 at the target, 1 at the edge), so moves near the target are unchanged."""
    if strength <= 0.0:
        return 1.0
    ox, oy = pos[0] - target[0], pos[1] - target[1]
    dist = math.hypot(ox, oy)
    mag = math.hypot(delta[0], delta[1])
    if dist < 1e-9 or mag < 1e-12:
        return 1.0
    cos = (ox * delta[0] + oy * delta[1]) / (dist * mag)
    r = min(1.0, dist / max(_edge_distance(target, (ox / dist, oy / dist)), 1e-6))
    return max(0.0, 1.0 + strength * cos * r)


def _edge_distance(origin: tuple[float, float], direction: tuple[float, float]) -> float:
    """Distance from ``origin`` to the unit frame's border along a unit ``direction``."""
    best = math.inf
    for o, d in zip(origin, direction):
        if d > 1e-12:
            best = min(best, (1.0 - o) / d)
        elif d < -1e-12:
            best = min(best, -o / d)
    return best if math.isfinite(best) else 0.5


def clamp_target(point: tuple[float, float], margin: float) -> tuple[float, float]:
    """Keep a target at least ``margin`` (plus a little) inside the frame so it is always trackable."""
    m = min(0.45, max(0.0, margin) + 0.1)
    return (min(max(point[0], m), 1.0 - m), min(max(point[1], m), 1.0 - m))


def shift_region(
    region: tuple[float, float, float, float], center: tuple[float, float]
) -> tuple[float, float, float, float]:
    """The absolute-mode active region moved (same size) so its centre is ``center``, kept inside 0..1."""
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    cx = min(max(center[0], w / 2), 1.0 - w / 2) if w < 1.0 else 0.5
    cy = min(max(center[1], h / 2), 1.0 - h / 2) if h < 1.0 else 0.5
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class ComfortZone:
    """EMA of where the hand rests (frames slower than ``rest_speed``), time constant ``tau_s``."""

    def __init__(self, tau_s: float = 20.0, rest_speed: float = 0.08, start: tuple[float, float] = FRAME_CENTER):
        self.tau_s = tau_s
        self.rest_speed = rest_speed
        self.center = start
        self.rest_time = 0.0  # seconds of resting seen (the zone is "learned" after a few seconds)
        self._prev: tuple[float, float, float] | None = None

    def reset_motion(self) -> None:
        """Forget the last sample (hand lost / re-anchored) but keep what was learned."""
        self._prev = None

    def update(self, t: float, x: float, y: float) -> bool:
        """Feed one hand position; returns True when this frame counted as resting."""
        prev, self._prev = self._prev, (t, x, y)
        if prev is None:
            return False
        dt = t - prev[0]
        if dt <= 0.0 or dt > 0.5:  # duplicate or a gap (hand was lost): no speed
            return False
        speed = math.hypot(x - prev[1], y - prev[2]) / dt
        if speed >= self.rest_speed:
            return False
        a = 1.0 - math.exp(-dt / max(self.tau_s, 1e-6))
        self.center = (self.center[0] + a * (x - self.center[0]), self.center[1] + a * (y - self.center[1]))
        self.rest_time += dt
        return True

    @property
    def learned(self) -> bool:
        return self.rest_time >= 2.0


@dataclass
class ClutchState:
    clutched: bool = False
    reanchor: bool = False  # True on the frame the clutch ends: the caller re-anchors the cursor there


class EdgeClutch:
    """Swing back from the camera edge without dragging the cursor back (see the module docstring)."""

    def __init__(self, margin: float = 0.08, return_speed: float = 0.5, rest_speed: float = 0.08) -> None:
        self.margin = margin
        self.return_speed = return_speed
        self.rest_speed = rest_speed
        self.clutched = False
        self._prev: tuple[float, float, float] | None = None

    def reset(self) -> None:
        self.clutched = False
        self._prev = None

    def edge_depth(self, x: float, y: float) -> tuple[float, float]:
        """Outward normal of the edge band the point is in, scaled 0..1 by how deep (0 = not in a band)."""
        m = self.margin
        if m <= 0.0:
            return 0.0, 0.0
        nx = (max(0.0, m - x) - max(0.0, x - (1.0 - m))) / m
        ny = (max(0.0, m - y) - max(0.0, y - (1.0 - m))) / m
        return -max(-1.0, min(1.0, nx)), -max(-1.0, min(1.0, ny))

    def update(self, t: float, x: float, y: float) -> ClutchState:
        prev, self._prev = self._prev, (t, x, y)
        if prev is None:
            return ClutchState(self.clutched)
        dt = t - prev[0]
        if dt <= 0.0:
            return ClutchState(self.clutched)
        vx, vy = (x - prev[1]) / dt, (y - prev[2]) / dt
        speed = math.hypot(vx, vy)
        if self.clutched:
            inner = self.margin * 2.0
            back_inside = inner <= x <= 1.0 - inner and inner <= y <= 1.0 - inner
            if speed < self.rest_speed or back_inside:
                self.clutched = False
                return ClutchState(False, reanchor=True)
            return ClutchState(True)
        # the previous position was in an edge band and the hand now moves quickly inward (away from that edge)
        ex, ey = self.edge_depth(prev[1], prev[2])
        if ex == 0.0 and ey == 0.0:
            return ClutchState(False)
        inward = -(vx * ex + vy * ey) / math.hypot(ex, ey)
        if inward >= self.return_speed:
            self.clutched = True
            return ClutchState(True)
        return ClutchState(False)


class AutoCenter:
    """The three helpers together, driven by :class:`~away.hands.cursor.CursorMapper` with the raw (filtered)
    hand position of every frame. Reads its settings live from an ``AutoCenterConfig``."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.comfort = ComfortZone(cfg.comfort_tau_s, cfg.rest_speed)
        self.clutch = EdgeClutch(cfg.edge_margin, cfg.edge_return_speed, cfg.rest_speed)

    def _sync(self) -> None:
        c = self.cfg
        self.comfort.tau_s, self.comfort.rest_speed = c.comfort_tau_s, c.rest_speed
        self.clutch.margin, self.clutch.return_speed, self.clutch.rest_speed = (
            c.edge_margin,
            c.edge_return_speed,
            c.rest_speed,
        )

    def reset(self) -> None:
        self.clutch.reset()
        self.comfort.reset_motion()

    def observe(self, t: float, x: float, y: float, clutch: bool = True) -> ClutchState:
        """One frame: learn the comfort zone and run the edge clutch (when enabled and ``clutch``: relative
        mode only, absolute mode maps positions)."""
        self._sync()
        c = self.cfg
        if not c.enabled:
            self.clutch.reset()
            return ClutchState()
        if c.comfort_zone:
            self.comfort.update(t, x, y)
        if c.edge_reanchor and clutch:
            return self.clutch.update(t, x, y)
        self.clutch.reset()
        return ClutchState()

    def target(self) -> tuple[float, float]:
        if self.cfg.target == "comfort" and self.cfg.comfort_zone and self.comfort.learned:
            return clamp_target(self.comfort.center, self.cfg.edge_margin)
        return FRAME_CENTER

    def gain(self, pos: tuple[float, float], delta: tuple[float, float]) -> float:
        c = self.cfg
        if not (c.enabled and c.soft_recenter):
            return 1.0
        return soft_gain(pos, delta, self.target(), float(c.soft_strength))
