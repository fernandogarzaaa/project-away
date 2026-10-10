"""Per-face "is this mouth talking" score from MediaPipe face blendshapes.

Speech is mouth *movement*, not an open mouth (a yawn is not speech), so the
score combines current openness with its short-term variability.
"""

from __future__ import annotations

from collections import deque

import numpy as np

MOUTH_KEYS = {
    "jawOpen": 1.0,
    "mouthLowerDownLeft": 0.25,
    "mouthLowerDownRight": 0.25,
    "mouthUpperUpLeft": 0.15,
    "mouthUpperUpRight": 0.15,
    "mouthFunnel": 0.2,
    "mouthPucker": 0.1,
}
_NORM = sum(MOUTH_KEYS.values())


def mouth_openness(blendshapes: dict[str, float]) -> float:
    """Weighted mouth activity in [0, 1]; ``mouthClose`` subtracts."""
    s = sum(w * float(blendshapes.get(k, 0.0)) for k, w in MOUTH_KEYS.items()) / _NORM
    s -= 0.5 * float(blendshapes.get("mouthClose", 0.0)) / _NORM
    return float(min(max(s, 0.0), 1.0))


class SpeakingScorer:
    """Rolling speaking score per face: 0.5 * mean openness + variability, clipped to [0, 1]."""

    def __init__(self, window: int = 15, var_gain: float = 6.0) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.var_gain = var_gain
        self._hist: dict[int, deque[float]] = {}

    def update(self, face_id: int, blendshapes: dict[str, float]) -> float:
        h = self._hist.setdefault(face_id, deque(maxlen=self.window))
        h.append(mouth_openness(blendshapes))
        arr = np.fromiter(h, dtype=float)
        variability = float(np.abs(np.diff(arr)).mean()) if len(arr) > 1 else 0.0
        return float(min(1.0, 0.5 * arr.mean() + self.var_gain * variability))

    def forget(self, face_id: int) -> None:
        self._hist.pop(face_id, None)
