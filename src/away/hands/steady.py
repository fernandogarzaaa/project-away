"""A steadier cursor (v0.1.25): adaptive smoothing, a hold-still lock and a dead zone that never jumps.

Why the cursor was jittery (found in v0.1.24):

* **The dead zone made it jump.** ``apply_dead_zone`` dropped any per-frame move under ``dead_zone`` *without*
  moving its reference point, so small moves piled up until they passed the threshold and then went through all
  at once: a slow, careful move became a staircase of ~15-20 px hops, and a noisy still hand twitched back and
  forth whenever one noisy frame crossed it. It was also per *frame*, so it acted differently at 15 and 30 fps.
* **The speed part of the One Euro filter did nothing.** ``beta`` (0.02) multiplies the hand speed in frame widths
  per second (about 0.1-2), which adds 0.002-0.04 Hz to a 1 Hz cutoff: the filter was a fixed low-pass, too
  slow for quick moves and still not steady enough for a hand held still.
* **The 120 Hz glide kept moving after a stop.** It extrapolated the last frame-to-frame velocity, which includes
  the jitter (and the staircase hops), so a hop was followed by an overshoot and a snap back.

What is here instead:

* :class:`AdaptiveFilter2D`: one cutoff for both axes that slides from ``still_hz`` (hand still: heavy smoothing)
  to ``fast_hz`` (hand moving at ``fast_speed`` or faster: follows closely), with the smoothing factor computed
  from the real time between frames, so it behaves the same at 15, 21 or 30 fps.
* :func:`deadband`: the dead zone is now slack (the cursor point stays put while the hand wobbles within it, then
  follows at that distance): noise never moves the cursor and slow moves go through smoothly, at any frame rate.
* :class:`HoldStill`: once the hand has stayed inside a small circle for ``hold_s``, the cursor locks; it unlocks
  as soon as the hand leaves a circle twice that size, continuing from where it is (no jump).
* :mod:`away.hands.livecal` measures the user's own noise, reach and pinch and sets these per camera.
"""

from __future__ import annotations

import math
from typing import Any

PRE_HZ = 4.0  # pre-smoothing for the speed estimate only


def alpha(dt: float, cutoff_hz: float) -> float:
    """Smoothing factor of a first-order low-pass with this cutoff for a step of ``dt`` seconds."""
    r = 2.0 * math.pi * max(1e-6, cutoff_hz) * max(0.0, dt)
    return r / (r + 1.0)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class AdaptiveFilter2D:
    """Speed-adaptive low-pass on (x, y) in hand units (0..1 of the camera frame)."""

    def __init__(
        self, still_hz: float = 1.0, fast_hz: float = 8.0, fast_speed: float = 0.5, speed_hz: float = 3.0
    ) -> None:
        if still_hz <= 0 or fast_hz <= 0 or fast_speed <= 0 or speed_hz <= 0:
            raise ValueError("cutoffs and speeds must be positive")
        self.still_hz = float(still_hz)
        self.fast_hz = max(float(fast_hz), self.still_hz)
        self.fast_speed = float(fast_speed)
        self.speed_hz = float(speed_hz)
        self.reset()

    def reset(self) -> None:
        self._t: float | None = None
        self._pre = (0.0, 0.0)  # lightly pre-smoothed position, only used to estimate the speed
        self._out = (0.0, 0.0)
        self.speed = 0.0  # smoothed hand speed, hand units / s

    def cutoff(self, speed: float) -> float:
        return self.still_hz + (self.fast_hz - self.still_hz) * smoothstep(speed / self.fast_speed)

    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]:
        if self._t is None:
            self._t, self._pre, self._out, self.speed = t, (x, y), (x, y), 0.0
            return self._out
        dt = t - self._t
        if dt <= 0:  # duplicate / out-of-order timestamp: hold the last output
            return self._out
        dt = min(dt, 0.5)  # the hand was briefly lost: do not treat the gap as a huge slow step
        # the speed comes from a lightly smoothed position, so landmark noise on a still hand does not look like
        # movement (which would open the filter up exactly when it should be steadiest)
        ap = alpha(dt, PRE_HZ)
        pre = (self._pre[0] + ap * (x - self._pre[0]), self._pre[1] + ap * (y - self._pre[1]))
        raw_speed = math.hypot(pre[0] - self._pre[0], pre[1] - self._pre[1]) / dt
        self.speed += alpha(dt, self.speed_hz) * (raw_speed - self.speed)
        a = alpha(dt, self.cutoff(self.speed))
        ox, oy = self._out
        self._out = (ox + a * (x - ox), oy + a * (y - oy))
        self._t, self._pre = t, pre
        return self._out


def deadband(prev: tuple[float, float], cur: tuple[float, float], dead_zone: float) -> tuple[float, float]:
    """A dead zone with slack instead of a threshold (like the play in a gear): the returned point stays put while
    ``cur`` moves within ``dead_zone`` of it, then is dragged along at exactly ``dead_zone`` behind. Small noise
    never moves it, and a slow move goes through in full and smoothly (no hops), at any frame rate."""
    dx, dy = cur[0] - prev[0], cur[1] - prev[1]
    mag = math.hypot(dx, dy)
    if mag <= dead_zone:
        return prev
    k = (mag - dead_zone) / mag
    return prev[0] + dx * k, prev[1] + dy * k


class HoldStill:
    """Locks the position while the hand is held still; unlocks without a jump.

    ``radius``: the unlock distance (hand units). The hand counts as still while it stays within ``radius / 2`` of
    where it was ``hold_s`` ago. ``keep_offset``: after unlocking, keep the small difference between the locked
    point and the hand (relative mode: only moves matter, so the cursor never catches up by itself); otherwise it
    fades out over ``ease_s`` (absolute mode).
    """

    def __init__(
        self, radius: float = 0.006, hold_s: float = 0.25, keep_offset: bool = True, ease_s: float = 0.15
    ) -> None:
        self.radius = max(0.0, float(radius))
        self.hold_s = max(0.0, float(hold_s))
        self.keep_offset = bool(keep_offset)
        self.ease_s = max(1e-3, float(ease_s))
        self.reset()

    def reset(self) -> None:
        self.locked = False
        self._anchor: tuple[float, float] | None = None
        self._t_anchor = 0.0
        self._lock_at = (0.0, 0.0)
        self._offset = (0.0, 0.0)
        self._t: float | None = None

    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]:
        if self.radius <= 0:
            return x, y
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        if self._anchor is None:
            self._anchor, self._t_anchor = (x, y), t
            return x + self._offset[0], y + self._offset[1]
        if self.locked:
            if math.hypot(x - self._anchor[0], y - self._anchor[1]) <= self.radius:
                return self._lock_at
            self.locked = False  # unlock: carry on from the locked point
            self._offset = (self._lock_at[0] - x, self._lock_at[1] - y)
            self._anchor, self._t_anchor = (x, y), t
        elif math.hypot(x - self._anchor[0], y - self._anchor[1]) > self.radius / 2.0:
            self._anchor, self._t_anchor = (x, y), t  # moving: the stillness window starts again
        elif t - self._t_anchor >= self.hold_s:
            self.locked = True
            self._lock_at = (x + self._offset[0], y + self._offset[1])
            return self._lock_at
        if not self.keep_offset and dt > 0:
            k = math.exp(-dt / self.ease_s)
            self._offset = (self._offset[0] * k, self._offset[1] * k)
        return x + self._offset[0], y + self._offset[1]


def hold_from_config(c: Any) -> HoldStill | None:
    """The hold-still lock for a ``hands.cursor`` section (None when ``hold_still`` is off)."""
    if not bool(getattr(c, "hold_still", False)):
        return None
    return HoldStill(
        float(getattr(c, "hold_radius", 0.006)),
        float(getattr(c, "hold_ms", 250)) / 1000.0,
        keep_offset=getattr(c, "mode", "relative") != "absolute",
    )


def jitter_px(points: list[tuple[float, float]]) -> float:
    """How much a cursor that should be still moves: RMS distance from its mean position (px)."""
    if len(points) < 2:
        return 0.0
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    return math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in points) / len(points))


__all__ = ["AdaptiveFilter2D", "HoldStill", "alpha", "deadband", "hold_from_config", "jitter_px"]
