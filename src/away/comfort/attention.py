"""Attention gate: "Only respond when I'm looking at the screen" (pure logic, no devices).

Engagement combines up to four checks, each one optional:

* **facing**: the head is turned toward the screens (head pose vs. the learned / calibrated screen-facing pose,
  the same reference :func:`away.face.headpose.facing_screen` uses). Face frames without the user's face = not
  facing. Face tracking switched off, or no face frame ever arrived (no face model) = the check is skipped: it
  cannot be done, and the gate must never lock the user out. Still learning the usual pose (first ~3 s) = passes.
* **gaze**: the eyes are on a monitor. Only when a gaze calibration exists; otherwise skipped.
* **hand raised**: the pointer hand's wrist is at least ``min_hand_height`` above the bottom of the picture and
  its palm at least ``min_hand_size`` big (a hand resting on the desk or someone far behind you does not count).
* **onset dwell**: all of the above held for ``onset_s`` before gestures count.

Hysteresis so it never flickers: once engaged, the hand may sink 0.05 lower and the head turn 5° further, and any
failing check is forgiven for ``grace_s`` before gestures stop. Voice commands (``gate_voice``) only need the
facing/gaze part: nobody raises a hand to speak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from away.hands.landmarks import MIDDLE_MCP, WRIST

HEIGHT_HYSTERESIS = 0.05  # picture heights
FACING_HYSTERESIS = 5.0  # degrees
FACE_TIMEOUT_S = 1.5  # face frames stop for this long = no face in view (slow PCs track faces every Nth frame)
GAZE_TIMEOUT_S = 1.5

STATES = ("off", "idle", "arming", "engaged", "grace")


def _points(landmarks: Any) -> Any:
    if landmarks is None:
        return None
    lm = np.asarray(landmarks, dtype=float)
    if lm.ndim != 2 or lm.shape[0] <= MIDDLE_MCP or lm.shape[1] < 2:
        return None
    return lm


def hand_height(landmarks: Any) -> float | None:
    """Wrist height above the bottom of the picture (0 = bottom edge, 1 = top edge), or None."""
    lm = _points(landmarks)
    return None if lm is None else float(1.0 - lm[WRIST, 1])


def hand_size(landmarks: Any) -> float | None:
    """Palm length (wrist -> middle-finger knuckle) as a fraction of the picture, or None."""
    lm = _points(landmarks)
    return None if lm is None else float(np.hypot(*(lm[MIDDLE_MCP, :2] - lm[WRIST, :2])))


@dataclass
class Checks:
    """One evaluation. ``None`` = the check is not used (switched off or impossible)."""

    facing: bool | None = None
    gaze: bool | None = None
    hand: bool | None = None

    def ok(self, need_hand: bool = True) -> bool:
        parts = [self.facing, self.gaze] + ([self.hand] if need_hand else [])
        return all(p is not False for p in parts)


class AttentionGate:
    """Feed it face, gaze and hand observations with timestamps; ask :meth:`update` whether gestures count."""

    def __init__(self, cfg: Any, face_available: bool = True) -> None:
        self.cfg = cfg  # an AttentionConfig (read live, so Settings changes apply at once)
        self.face_available = face_available  # False when face tracking is switched off
        self.state = "off" if not cfg.enabled else "idle"
        self._ok_since: float | None = None
        self._bad_since: float | None = None
        self._face_t: float | None = None
        self._facing: bool | None = None  # last facing verdict (None = cannot tell yet)
        self._off_deg: float | None = None  # degrees off the screen-facing pose, when known
        self._gaze_t: float | None = None
        self._gaze_on: bool | None = None
        self._voice_ok_until = float("-inf")
        self.checks = Checks()
        self.height: float | None = None
        self.size: float | None = None

    # ------------------------------------------------------------------ inputs
    def on_face(self, t: float, facing: bool | None, off_deg: float | None = None) -> None:
        """A face frame with the user's face. ``facing``: a verdict (None = still learning). ``off_deg``: how far
        (degrees) the head is from the screen-facing pose, when known; the gate then applies its own tolerance
        with hysteresis."""
        self._face_t = t
        self._facing = facing
        self._off_deg = off_deg

    def on_no_face(self, t: float) -> None:
        """A face frame in which the user's face was not found."""
        self._face_t = t
        self._facing = False
        self._off_deg = None

    def on_gaze(self, t: float, on_screen: bool | None) -> None:
        self._gaze_t = t
        self._gaze_on = on_screen

    # ------------------------------------------------------------------ checks
    def _facing_check(self, t: float, engaged: bool) -> bool | None:
        c = self.cfg
        if not c.require_facing or not self.face_available:
            return None
        if self._face_t is None:
            return None  # the face pipeline never ran (no face model, camera off): cannot check, never lock out
        if t - self._face_t > FACE_TIMEOUT_S:
            return False  # face frames stopped arriving
        if self._off_deg is not None:
            tol = float(c.facing_tolerance) + (FACING_HYSTERESIS if engaged else 0.0)
            return self._off_deg <= tol
        return True if self._facing is None else bool(self._facing)

    def _gaze_check(self, t: float) -> bool | None:
        if not self.cfg.require_gaze or self._gaze_on is None or self._gaze_t is None:
            return None  # not wanted, or no calibration: skipped
        if t - self._gaze_t > GAZE_TIMEOUT_S:
            return False
        return bool(self._gaze_on)

    def _hand_check(self, landmarks: Any, engaged: bool) -> bool:
        self.height, self.size = hand_height(landmarks), hand_size(landmarks)
        if self.height is None or self.size is None:
            return False
        lo = float(self.cfg.min_hand_height) - (HEIGHT_HYSTERESIS if engaged else 0.0)
        return self.height >= lo and self.size >= float(self.cfg.min_hand_size)

    # ------------------------------------------------------------------ verdicts
    @property
    def engaged(self) -> bool:
        return self.state in ("off", "engaged", "grace")

    def update(self, t: float, landmarks: Any) -> bool:
        """One hand frame (``landmarks`` = the pointer hand or None). True = gestures from it count."""
        if not self.cfg.enabled:
            self.state = "off"
            self._ok_since = self._bad_since = None
            self.checks = Checks()
            return True
        was = self.state in ("engaged", "grace")
        self.checks = Checks(self._facing_check(t, was), self._gaze_check(t), self._hand_check(landmarks, was))
        if self.checks.ok():
            self._bad_since = None
            if was:
                self.state = "engaged"
            else:
                if self._ok_since is None:
                    self._ok_since = t
                self.state = "engaged" if t - self._ok_since >= float(self.cfg.onset_s) else "arming"
        else:
            self._ok_since = None
            if was:
                if self._bad_since is None:
                    self._bad_since = t
                if t - self._bad_since < float(self.cfg.grace_s):
                    self.state = "grace"
                else:
                    self.state, self._bad_since = "idle", None
            else:
                self.state = "idle"
        return self.engaged

    def voice_allowed(self, t: float) -> bool:
        """Voice commands (``gate_voice``): facing (and gaze when calibrated/required) only, with the same grace."""
        c = self.cfg
        if not (c.enabled and c.gate_voice):
            return True
        if Checks(self._facing_check(t, True), self._gaze_check(t), None).ok(need_hand=False):
            self._voice_ok_until = t + float(c.grace_s)
            return True
        return t <= self._voice_ok_until

    def hint(self) -> str:
        """What the user should do to get gestures back, in plain words ("" when nothing is needed)."""
        if self.state in ("off", "engaged"):
            return ""
        if self.state == "grace":
            return "still with you..."
        if self.state == "arming":
            return "getting ready..."
        ch = self.checks
        if ch.facing is False:
            return "turn toward the screen"
        if ch.gaze is False:
            return "look at the screen"
        if ch.hand is False:
            if self.height is None:
                return "raise your hand into view"
            if self.size is not None and self.size < float(self.cfg.min_hand_size):
                return "bring your hand a little closer"
            return "raise your hand a little higher"
        return ""

    def snapshot(self) -> dict[str, Any]:
        """``attention_state`` for the UI (tray, ring, Settings, first run)."""
        return {
            "enabled": bool(self.cfg.enabled),
            "state": self.state,
            "engaged": self.engaged,
            "facing": self.checks.facing,
            "gaze": self.checks.gaze,
            "hand_raised": self.checks.hand,
            "hand_height": None if self.height is None else round(self.height, 3),
            "hint": self.hint(),
        }
