"""Hand signs for answering prompts: thumbs up, thumbs down, a finger count (1-4) and an open palm (5).

Pure geometry on the 21 MediaPipe landmarks (rotation and scale invariant for the finger tests): a finger is
extended when its tip is clearly further from the wrist than its middle joint; the thumb is extended when its
tip is far from the wrist; thumbs up/down need the four fingers folded and the thumb pointing up/down in the
image. :class:`SignHold` turns per-frame signs into one answer once a sign is held steadily.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

WRIST = 0
FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}  # (PIP, TIP)

THUMBS_UP = "thumbs_up"
THUMBS_DOWN = "thumbs_down"
OPEN_PALM = "open_palm"


def _d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def extended_fingers(lm: Any) -> list[str]:
    out = []
    w = lm[WRIST]
    for name, (pip, tip) in FINGERS.items():
        if _d(lm[tip], w) > 1.15 * _d(lm[pip], w):
            out.append(name)
    return out


def thumb_extended(lm: Any) -> bool:
    palm = _d(lm[WRIST], lm[9]) or 1e-6
    return _d(lm[4], lm[WRIST]) > 0.95 * palm


def classify(lm: Any) -> str | None:
    """``thumbs_up`` | ``thumbs_down`` | ``open_palm`` | ``"1"``..``"4"`` | None."""
    if lm is None:
        return None
    try:
        if len(lm) < 21:
            return None
    except TypeError:
        return None
    fingers = extended_fingers(lm)
    thumb = thumb_extended(lm)
    if not fingers and thumb:
        palm = _d(lm[WRIST], lm[9]) or 1e-6
        if _d(lm[4], lm[8]) < 0.45 * palm:
            return None  # thumb on the index tip: a pinch, not a thumbs up
        dx = float(lm[4][0]) - float(lm[2][0])
        dy = float(lm[4][1]) - float(lm[2][1])
        if abs(dy) > 0.25 * palm and abs(dy) >= 0.5 * abs(dx):
            return THUMBS_UP if dy < 0 else THUMBS_DOWN  # image y grows downwards
        return None
    if len(fingers) == 4 and thumb:
        return OPEN_PALM
    if fingers:
        return str(len(fingers))
    return None


class SignHold:
    """Emits a sign once it has been seen in (nearly) every frame for ``hold_s``; then waits for a change."""

    def __init__(self, hold_s: float = 0.6, gap_s: float = 0.25) -> None:
        self.hold_s, self.gap_s = hold_s, gap_s
        self._sign: str | None = None
        self._since = 0.0
        self._last = -1e9
        self._fired = False

    def update(self, t: float, sign: str | None) -> str | None:
        if sign is None:
            if t - self._last > self.gap_s:
                self._sign, self._fired = None, False
            return None
        if sign != self._sign or t - self._last > self.gap_s:
            self._sign, self._since, self._fired = sign, t, False
        self._last = t
        if not self._fired and t - self._since >= self.hold_s:
            self._fired = True
            return sign
        return None

    def reset(self) -> None:
        self._sign, self._fired = None, False


def answer_for(sign: str, options: Sequence[str]) -> str | None:
    """Map a held sign to one of ``options``: thumbs up = the first / "yes"-like option, thumbs down = the
    last / "no"-like option, a finger count N = option N (1-based), open palm = option 5."""
    if not options:
        return None
    low = [o.strip().lower() for o in options]
    yes = ("yes", "allow", "approve", "ok", "okay", "accept", "do it", "sure")
    no = ("no", "deny", "reject", "cancel", "stop", "don't", "dont")
    if sign == THUMBS_UP:
        for i, o in enumerate(low):
            if o in yes:
                return options[i]
        return options[0]
    if sign == THUMBS_DOWN:
        for i, o in enumerate(low):
            if o in no:
                return options[i]
        return options[-1] if len(options) > 1 else None
    n = 5 if sign == OPEN_PALM else (int(sign) if sign.isdigit() else 0)
    if 1 <= n <= len(options):
        return options[n - 1]
    return None
