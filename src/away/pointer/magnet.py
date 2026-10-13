"""Magnetic targets: gentle attraction toward clickable elements and sticky friction over them.

Works on the mapped desktop position, after :class:`~away.hands.cursor.CursorMapper` (so every cursor mode,
curve and monitor layout keeps working) and before the OS cursor moves. Two effects, both bounded:

* **Sticky friction.** While the cursor is over a target, small moves are scaled by ``1 - sticky``: webcam jitter
  of a few pixels mostly disappears, so a pinch lands on the button you are on. Friction fades out with speed
  (a move of ``radius_px`` per frame or more is not slowed at all), so a deliberate move leaves at once. The
  difference builds an *offset* between hand and cursor; it is capped at ``radius_px`` and bleeds away as soon as
  the cursor moves outside targets, so hand and cursor never drift apart.
* **Attraction.** Within ``radius_px`` of a target the shown cursor is pulled part of the way toward the target's
  centre: ``strength`` of the way at the edge of the target, fading with distance (smoothstep) to nothing at
  ``radius_px``. The pull is a pure function of position (never accumulated), so the cursor cannot creep.
  Inside a target the pull fades with the depth from its edge and is capped at ``strength * radius_px``, so you
  can still point anywhere inside a big text field.

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


def pull_vector(x: float, y: float, target: Target, radius: float, strength: float) -> tuple[float, float]:
    """How far to shift the shown cursor at ``(x, y)`` toward ``target`` (desktop px)."""
    if radius <= 0 or strength <= 0:
        return 0.0, 0.0
    d = target.distance(x, y)
    if d > radius:
        return 0.0, 0.0
    cx, cy = target.center
    vx, vy = cx - x, cy - y
    dist_c = math.hypot(vx, vy)
    if dist_c < 1e-9:
        return 0.0, 0.0
    # outside: fades from `strength` at the edge to 0 at `radius`; inside: `strength` at the edge, fading with the
    # depth into the target so the middle of a big text box is not dragged anywhere
    if d > 0:
        w = strength * (1.0 - smoothstep(d / radius))
    else:
        left, top, right, bottom = target.rect
        depth = min(x - left, right - x, y - top, bottom - y)
        w = strength * (1.0 - smoothstep(depth / radius))
    shift = min(w * dist_c, radius * strength)
    return vx / dist_c * shift, vy / dist_c * shift


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
        hit = self._snapshot().nearest(shown[0], shown[1], float(cfg.radius_px))
        if hit is not None:
            px, py = pull_vector(shown[0], shown[1], hit, float(cfg.radius_px), float(cfg.strength))
            self.offset = (-px, -py)
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
