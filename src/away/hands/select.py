"""Which of the detected hands drives the cursor (``hands.prefer``). Pure logic, no devices.

Handedness and mirroring
------------------------
MediaPipe's *Tasks* Hand Landmarker (what Away uses) labels a hand by how it looks **in the image it is given**:
a real right hand in a raw, un-flipped webcam frame is labelled ``Right``. Checked two ways:

* MediaPipe's own test data: ``thumb_up_landmarks.pbtxt`` / ``pointing_up_landmarks.pbtxt`` (un-flipped photos)
  are labelled ``Right`` and their 3D landmarks have right-hand chirality;
* four photos run through the Tasks API (mediapipe 0.10.18, hand_landmarker.task float16/1): every raw photo of a
  right hand came back ``Right`` and the same photo flipped horizontally came back ``Left``.

(The legacy ``mp.solutions.hands`` docs say the opposite: that model assumed a mirrored selfie image.)

Away flips the frame when ``mirror`` is on (the default, so moving your hand right moves the cursor right), and
a flipped image turns a right hand into a left-looking one. So the user's real hand is::

    mirror on  (flipped frame):  real = the opposite of MediaPipe's label
    mirror off (raw frame):      real = MediaPipe's label

:func:`real_side` does this; each camera uses its own ``mirror`` flag.

Selection
---------
* ``right`` / ``left``: only hands whose real side matches; the other hand is ignored completely. A tracked hand
  whose label flickers to the wrong side for a moment (side-on, motion blur) is kept for up to
  ``label_grace_s`` so a drag is not dropped.
* ``any``: lock onto the hand being tracked and keep it while it stays visible (the candidate nearest to its
  last wrist position, within ``lock_radius``); only after it has been gone for ``lock_grace_s`` may another
  hand take over. So picking up a phone with the other hand does not steal the cursor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LOCK_RADIUS = 0.2  # normalised image units the wrist may move between two tracked frames
LOCK_GRACE_S = 0.5
LABEL_GRACE_S = 0.3


@dataclass
class HandObs:
    landmarks: Any  # (21, 3) normalised image coordinates
    label: str  # MediaPipe's handedness label for this image ("Right" | "Left")
    score: float = 1.0

    @property
    def wrist(self) -> tuple[float, float]:
        return float(self.landmarks[0][0]), float(self.landmarks[0][1])


def swap_side(label: str) -> str:
    return "Left" if label.lower().startswith("r") else "Right"


def real_side(label: str, mirrored: bool) -> str:
    """The user's real hand (``right`` | ``left``) for a MediaPipe Tasks label on a (possibly flipped) frame."""
    side = swap_side(label) if mirrored else ("Right" if label.lower().startswith("r") else "Left")
    return side.lower()


class HandSelector:
    def __init__(
        self,
        prefer: str | Callable[[], str] = "any",
        mirrored: bool = True,
        lock_radius: float = LOCK_RADIUS,
        lock_grace_s: float = LOCK_GRACE_S,
        label_grace_s: float = LABEL_GRACE_S,
    ) -> None:
        self._prefer = prefer
        self.mirrored = mirrored
        self.lock_radius = lock_radius
        self.lock_grace_s = lock_grace_s
        self.label_grace_s = label_grace_s
        self.reset()

    @property
    def prefer(self) -> str:
        p = self._prefer() if callable(self._prefer) else self._prefer
        return p if p in ("right", "left") else "any"

    @prefer.setter
    def prefer(self, value: str | Callable[[], str]) -> None:
        self._prefer = value
        self.reset()

    def reset(self) -> None:
        self._last_pos: tuple[float, float] | None = None
        self._last_t = -1e9
        self._last_side: str | None = None
        self._wrong_since: float | None = None
        self._last_prefer: str | None = None

    def side(self, h: HandObs) -> str:
        return real_side(h.label, self.mirrored)

    def wrong_side(self, hands: list[HandObs]) -> bool:
        """True when hands are visible but none is on the preferred side (the tracker should look for more)."""
        p = self.prefer
        return p != "any" and bool(hands) and all(self.side(h) != p for h in hands)

    def _nearest(self, hands: list[HandObs]) -> tuple[HandObs | None, float]:
        if self._last_pos is None or not hands:
            return None, math.inf
        best = min(hands, key=lambda h: math.dist(h.wrist, self._last_pos))  # type: ignore[arg-type]
        return best, math.dist(best.wrist, self._last_pos)

    def select(self, t: float, hands: list[HandObs]) -> HandObs | None:
        prefer = self.prefer
        if prefer != self._last_prefer:  # the preference changed (Settings / voice): start over
            self.reset()
            self._last_prefer = prefer
        if not hands:
            return None
        locked = self._last_pos is not None and t - self._last_t <= self.lock_grace_s
        near, dist = self._nearest(hands)
        tracked = near if locked and dist <= self.lock_radius else None
        pick: HandObs | None
        if prefer == "any":
            if tracked is not None:
                pick = tracked
            elif locked:
                # the tracked hand is not where it was; a lone hand of the same side is it (fast move), anything
                # else waits until the lock expires
                same = [h for h in hands if self.side(h) == self._last_side]
                pick = same[0] if len(hands) == 1 and same else None
            else:
                pick = near if near is not None else max(hands, key=lambda h: h.score)
        else:
            pick = None
            if tracked is not None:
                if self.side(tracked) == prefer:
                    pick, self._wrong_since = tracked, None
                else:
                    if self._wrong_since is None:
                        self._wrong_since = t
                    if t - self._wrong_since <= self.label_grace_s:
                        pick = tracked  # the label flickered on the hand we were following
            if pick is None:
                matching = [h for h in hands if self.side(h) == prefer]
                if matching:
                    self._wrong_since = None
                    last = self._last_pos
                    if last is not None:
                        pick = min(matching, key=lambda h: math.dist(h.wrist, last))
                    else:
                        pick = max(matching, key=lambda h: h.score)
        if pick is not None:
            self._last_pos, self._last_t = pick.wrist, t
            self._last_side = self.side(pick)
        return pick
