"""Decision fusion: rule-based gesture events x temporal model x Wi-Fi CSI.

Continuous events (move, scroll, drag_end, hand_lost) always pass: blocking
them adds lag and can leave a mouse button stuck. Discrete events (click,
double click, drag start, clutch, swipes) are scored:

    score = (1 - w) * camera_conf + w * csi_support

camera_conf is the temporal model's recent max probability for the matching
class (1.0 when no model is loaded); csi_support is 1.0 if the latest CSI
label agrees, 0.5 when CSI is absent/stale/neutral, 0.0 when CSI says the
room is empty (a camera false positive with nobody there).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from away.hands.gestures import GestureEvent

PASS_THROUGH = frozenset({"move", "scroll", "drag_end", "hand_lost"})
EVENT_TO_CLASS = {
    "click": "pinch_click",
    "double_click": "double_pinch",
    "drag_start": "pinch_drag",
    "clutch": "palm_clutch",
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
}
# which CSI labels support which camera events
CSI_SUPPORT = {
    "swipe_left": {"swipe", "motion"},
    "swipe_right": {"swipe", "motion"},
}


@dataclass
class CSIState:
    label: str
    confidence: float
    t: float


class GestureFusion:
    def __init__(
        self, csi_weight: float = 0.25, threshold: float = 0.6, csi_max_age: float = 1.5, model_window: int = 8
    ) -> None:
        if not 0.0 <= csi_weight <= 1.0:
            raise ValueError("csi_weight must be in [0, 1]")
        self.w = csi_weight
        self.threshold = threshold
        self.csi_max_age = csi_max_age
        self.csi: CSIState | None = None
        self._probs: deque[dict[str, float]] = deque(maxlen=model_window)
        self.rejected: list[tuple[GestureEvent, float]] = []

    def update_model(self, probs: dict[str, float] | None) -> None:
        if probs:
            self._probs.append(probs)

    def update_csi(self, label: str, confidence: float, t: float) -> None:
        self.csi = CSIState(label, confidence, t)

    def camera_conf(self, cls: str) -> float:
        if not self._probs:
            return 1.0
        return max(p.get(cls, 0.0) for p in self._probs)

    def csi_support(self, kind: str, t: float) -> float:
        if self.csi is None or t - self.csi.t > self.csi_max_age:
            return 0.5
        if self.csi.label == "empty":
            return 0.5 * (1.0 - self.csi.confidence)
        if self.csi.label in CSI_SUPPORT.get(kind, set()):
            return 0.5 + 0.5 * self.csi.confidence
        return 0.5

    def score(self, ev: GestureEvent) -> float:
        if ev.kind in PASS_THROUGH or ev.kind not in EVENT_TO_CLASS:
            return 1.0
        cam = self.camera_conf(EVENT_TO_CLASS[ev.kind])
        return (1.0 - self.w) * cam + self.w * self.csi_support(ev.kind, ev.t)

    def accept(self, ev: GestureEvent) -> bool:
        s = self.score(ev)
        if s >= self.threshold:
            return True
        self.rejected.append((ev, s))
        return False
