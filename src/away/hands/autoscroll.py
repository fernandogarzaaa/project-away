"""Four-finger auto-scroll (joystick / middle-click-autoscroll style). Pure logic, no devices.

Hold up four fingers (index..pinky extended, thumb folded) for ``autoscroll_hold`` seconds: the hand's position
at that moment becomes the anchor. Moving the hand above or below the anchor scrolls up or down, faster the
further it is (dead zone, power curve, max speed), and it keeps scrolling while the hand holds still. With
``autoscroll_horizontal`` the hand left/right of the anchor scrolls sideways; the axis with the larger offset
wins, so a vertical scroll never drifts sideways.

It stops when:

* the hand closes into a fist for ``autoscroll_stop_hold`` seconds (``reason: fist``);
* another gesture is held for :data:`OTHER_STOP_S` (point, two fingers, open palm with the thumb out, pinch);
* the hand is gone for more than :data:`LOST_STOP_S` (a shorter dropout pauses the scrolling, then resumes);
* gesture control pauses (:meth:`AutoScroller.stop`).

Offsets are measured in palm lengths (wrist -> middle MCP), so the feel does not change with the distance to
the camera. Speeds are wheel notches per second; :meth:`AutoScroller.update` returns the notches to scroll for
the time since the previous frame, which the backend accumulates (fractional notches carry over).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from away.config import GestureConfig

OTHER_STOP_S = 0.35  # another pose held this long ends auto-scroll (short flickers while moving do not)
LOST_STOP_S = 0.5  # hand gone longer than this ends auto-scroll
CURVE_EXPONENT = 1.5  # speed = gain * excess ** exponent (gentle near the dead zone, quick further out)
MAX_FRAME_DT = 0.1  # a stalled frame never produces a burst of scrolling


@dataclass
class ScrollStep:
    dx: float  # notches this frame (+ = right)
    dy: float  # notches this frame (+ = up)
    vx: float  # notches / s
    vy: float


def axis_speed(offset: float, dead_zone: float, gain: float, max_speed: float) -> float:
    """Signed speed (notches/s) for an offset in palm lengths."""
    excess = abs(offset) - dead_zone
    if excess <= 0:
        return 0.0
    speed = min(max_speed, gain * excess**CURVE_EXPONENT)
    return speed if offset > 0 else -speed


def direction(vx: float, vy: float) -> str | None:
    """``up`` | ``down`` | ``left`` | ``right`` | None (inside the dead zone)."""
    if vx == 0 and vy == 0:
        return None
    if abs(vy) >= abs(vx):
        return "up" if vy > 0 else "down"
    return "right" if vx > 0 else "left"


class AutoScroller:
    def __init__(self, cfg: GestureConfig | None = None) -> None:
        self.cfg = cfg or GestureConfig()
        self.active = False
        self.anchor: tuple[float, float] | None = None
        self.vx = 0.0
        self.vy = 0.0
        self._four_t0: float | None = None
        self._fist_t0: float | None = None
        self._other_t0: float | None = None
        self._lost_t0: float | None = None
        self._last_t: float | None = None

    @property
    def direction(self) -> str | None:
        return direction(self.vx, self.vy) if self.active else None

    def stop(self) -> bool:
        """End auto-scroll (pause, reset). True if it was running."""
        was = self.active
        self.active = False
        self.anchor = None
        self.vx = self.vy = 0.0
        self._four_t0 = self._fist_t0 = self._other_t0 = self._lost_t0 = None
        self._last_t = None
        return was

    def lost(self, t: float) -> dict[str, Any] | None:
        """No hand this frame. Returns the stop event data once the hand has been gone > LOST_STOP_S."""
        self._four_t0 = self._fist_t0 = self._other_t0 = None
        if not self.active:
            return None
        if self._lost_t0 is None:
            self._lost_t0 = t
        self._last_t = None  # no catch-up burst when the hand comes back
        if t - self._lost_t0 > LOST_STOP_S:
            self.stop()
            return {"reason": "hand_lost"}
        return None

    def update(
        self, t: float, pose: str, pos: tuple[float, float], palm: float, pinched: bool = False
    ) -> tuple[str, dict[str, Any]] | None:
        """Feed one frame with a hand.

        ``pose``: ``four`` (thumb folded) | ``palm`` (thumb out) | ``fist`` | ``point`` | ``two`` | ``other``;
        ``pos``: wrist (x, y) in normalised image coordinates; ``palm``: palm length in the same units.
        Returns ``("autoscroll_start" | "autoscroll" | "autoscroll_stop", data)`` or None.
        """
        self._lost_t0 = None
        cfg = self.cfg
        if not self.active:
            if not cfg.autoscroll_enabled or pose != "four" or pinched:
                self._four_t0 = None
                return None
            if self._four_t0 is None:
                self._four_t0 = t
            if t - self._four_t0 < cfg.autoscroll_hold:
                return None
            self.active = True
            self.anchor = pos
            self._last_t = t
            self._fist_t0 = self._other_t0 = None
            return "autoscroll_start", {"x": pos[0], "y": pos[1]}

        # running: stop on a held fist or a held other gesture
        if pose == "fist":
            self._other_t0 = None
            if self._fist_t0 is None:
                self._fist_t0 = t
            if t - self._fist_t0 >= cfg.autoscroll_stop_hold:
                self.stop()
                return "autoscroll_stop", {"reason": "fist"}
        elif pose != "four" or pinched:
            self._fist_t0 = None
            if self._other_t0 is None:
                self._other_t0 = t
            if t - self._other_t0 >= OTHER_STOP_S:
                self.stop()
                return "autoscroll_stop", {"reason": "pinch" if pinched else f"pose:{pose}"}
        else:
            self._fist_t0 = self._other_t0 = None
        if not cfg.autoscroll_enabled:
            self.stop()
            return "autoscroll_stop", {"reason": "disabled"}

        if pose == "four" and not pinched:  # steer only with the pose itself; a fist forming keeps the speed
            assert self.anchor is not None
            scale = max(palm, 1e-6)
            ox = (pos[0] - self.anchor[0]) / scale  # + = hand moved right (mirrored frame: the user's right)
            oy = (self.anchor[1] - pos[1]) / scale  # + = hand moved up
            vx = axis_speed(ox, cfg.autoscroll_dead_zone, cfg.autoscroll_gain, cfg.autoscroll_max_speed)
            vy = axis_speed(oy, cfg.autoscroll_dead_zone, cfg.autoscroll_gain, cfg.autoscroll_max_speed)
            if not cfg.autoscroll_horizontal:
                vx = 0.0
            elif abs(vx) > abs(vy):
                vy = 0.0
            else:
                vx = 0.0
            self.vx, self.vy = vx, vy
        dt = 0.0 if self._last_t is None else min(MAX_FRAME_DT, max(0.0, t - self._last_t))
        self._last_t = t
        step = ScrollStep(self.vx * dt, self.vy * dt, self.vx, self.vy)
        return "autoscroll", {
            "dx": step.dx,
            "dy": step.dy,
            "vx": step.vx,
            "vy": step.vy,
            "direction": direction(step.vx, step.vy),
        }
