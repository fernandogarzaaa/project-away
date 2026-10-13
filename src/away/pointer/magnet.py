"""Magnetic targets: gentle attraction toward clickable elements and sticky friction over them.

Works on the mapped desktop position, after :class:`~away.hands.cursor.CursorMapper` (so every cursor mode,
curve and monitor layout keeps working) and before the OS cursor moves. Two effects, both bounded:

* **Sticky friction.** While the cursor is over a target, small moves are scaled by ``1 - sticky``: webcam jitter
  of a few pixels mostly disappears, so a pinch lands on the button you are on. Friction fades out with speed
  (a move of ``radius_px`` per frame or more is not slowed at all), so a deliberate move leaves at once. The
  difference builds an *offset* between hand and cursor; it is capped at ``radius_px`` and bleeds away as soon as
  the cursor moves outside targets, so hand and cursor never drift apart.
* **Attraction.** Within ``radius_px`` of a target the shown cursor is pulled part of the way toward the target's
  core (:func:`anchor_point`; the centre of a small button): ``strength`` of the way at and inside the target's
  edge, fading with distance (smoothstep) to nothing at ``radius_px``, and never more than
  ``strength * radius_px``. The pull is a pure function of position (never accumulated), so the cursor cannot
  creep, and you can still point anywhere along a big text field.

A large jump in one frame (monitor jump, look-to-jump, "center", a re-anchor) resets the offset.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from away.pointer.targets import Snapshot, Target

JUMP_FACTOR = 4.0  # a raw move longer than JUMP_FACTOR * radius in one frame is a warp: reset the offset


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def anchor_point(x: float, y: float, target: Target, radius: float) -> tuple[float, float]:
    """Where the pull aims: the nearest point of the target's *core*, the rectangle shrunk by up to half the radius
    on each side. A small button's core is its centre; a wide text box's core is a line along its middle, so
    inside it the cursor is only steadied across the box, never dragged along it."""
    left, top, right, bottom = target.rect
    ix = min((right - left) / 2.0, radius / 2.0)
    iy = min((bottom - top) / 2.0, radius / 2.0)
    return min(max(x, left + ix), right - ix), min(max(y, top + iy), bottom - iy)


def pull_vector(x: float, y: float, target: Target, radius: float, strength: float) -> tuple[float, float]:
    """How far to shift the shown cursor at ``(x, y)`` toward ``target`` (desktop px)."""
    if radius <= 0 or strength <= 0:
        return 0.0, 0.0
    d = target.distance(x, y)
    if d > radius:
        return 0.0, 0.0
    ax, ay = anchor_point(x, y, target, radius)
    vx, vy = ax - x, ay - y
    dist = math.hypot(vx, vy)
    if dist < 1e-9:
        return 0.0, 0.0
    # `strength` of the way at and inside the edge, fading (smoothstep) to nothing at `radius` outside it
    w = strength * (1.0 - smoothstep(d / radius))
    shift = min(w * dist, radius * strength)
    return vx / dist * shift, vy / dist * shift


class Magnet:
    """Stateful refiner: ``apply(x, y) -> (x, y)`` once per cursor frame.

    ``cfg`` returns a ``hands.cursor.magnet`` section (read live, so Settings changes apply at once); ``snapshot``
    returns the latest :class:`~away.pointer.targets.Snapshot` (never blocks).
    """

    def __init__(self, cfg: Callable[[], Any], snapshot: Callable[[], Snapshot]) -> None:
        self._cfg = cfg
        self._snapshot = snapshot
        self.offset = (0.0, 0.0)
        self._last_raw: tuple[float, float] | None = None
        self.last_target: Target | None = None
        self.last_out: tuple[float, float] | None = None

    def reset(self) -> None:
        self.offset = (0.0, 0.0)
        self._last_raw = None
        self.last_target = None

    def resync(self, shown: tuple[float, float]) -> None:
        """The cursor was put at ``shown`` directly (the steady-click rewind) and the mapper now sits there too:
        start again from it without the pull shifting it a second time."""
        self.reset()
        cfg = self._cfg()
        radius, strength = float(cfg.radius_px), float(cfg.strength)
        snap = self._snapshot()
        qx, qy = shown
        for _ in range(6):  # solve q + pull(q) = shown (a few fixed-point steps converge: the pull is gentle)
            hit = snap.nearest(qx, qy, radius)
            px, py = pull_vector(qx, qy, hit, radius, strength) if hit is not None else (0.0, 0.0)
            qx, qy = shown[0] - px, shown[1] - py
        self.offset = (qx - shown[0], qy - shown[1])
        self._last_raw = shown

    def apply(self, x: float, y: float) -> tuple[float, float]:
        cfg = self._cfg()
        radius = float(cfg.radius_px)
        snap = self._snapshot()
        if not cfg.enabled or not snap.targets or radius <= 0:
            self.reset()
            self._last_raw = (x, y)
            self.last_out = (x, y)
            return x, y
        ox, oy = self.offset
        if self._last_raw is not None:
            dx, dy = x - self._last_raw[0], y - self._last_raw[1]
            step = math.hypot(dx, dy)
            if step > JUMP_FACTOR * radius:
                ox, oy = 0.0, 0.0
            elif step > 0:
                qx, qy = self._last_raw[0] + ox, self._last_raw[1] + oy  # where the frictioned cursor was
                if snap.nearest(qx, qy, 0.0) is not None:
                    friction = float(cfg.sticky) * (1.0 - smoothstep(step / radius))
                    ox -= dx * friction
                    oy -= dy * friction
                else:
                    keep = max(0.0, 1.0 - step / radius)  # moving outside targets bleeds the offset away
                    ox, oy = ox * keep, oy * keep
                mag = math.hypot(ox, oy)
                if mag > radius:
                    ox, oy = ox * radius / mag, oy * radius / mag
        self._last_raw = (x, y)
        self.offset = (ox, oy)
        qx, qy = x + ox, y + oy
        hit = snap.nearest(qx, qy, radius)
        self.last_target = hit
        if hit is None:
            self.last_out = (qx, qy)
            return qx, qy
        px, py = pull_vector(qx, qy, hit, radius, float(cfg.strength))
        self.last_out = (qx + px, qy + py)
        return self.last_out
