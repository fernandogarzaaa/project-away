"""Active speaker detection: which visible face is producing the speech the mic hears.

Baseline: Pearson correlation between each face's mouth-activity series and
the mic energy series over a sliding window (both resampled to the video
frame times). A learned audio-visual model (TalkNet, Light-ASD) can be
plugged in through :class:`ASDModel`; its scores replace the correlation.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


class ASDModel(Protocol):
    """Pluggable learned active-speaker model."""

    def scores(self, face_series: dict[int, np.ndarray], audio_energy: np.ndarray) -> dict[int, float]: ...


class ActiveSpeakerDetector:
    def __init__(
        self,
        window: int = 45,
        min_correlation: float = 0.2,
        speaking_threshold: float = 0.25,
        model: ASDModel | None = None,
    ) -> None:
        self.window = window
        self.min_corr = min_correlation
        self.speaking_threshold = speaking_threshold
        self.model = model
        self._audio: deque[float] = deque(maxlen=window)
        self._faces: dict[int, deque[float]] = {}

    def update(self, speaking_scores: dict[int, float], audio_energy: float) -> int | None:
        """Feed one video frame's per-face speaking scores and the mic energy at that time."""
        self._audio.append(float(audio_energy))
        n = len(self._audio)
        for fid in list(self._faces):
            if fid not in speaking_scores:
                del self._faces[fid]
        for fid, s in speaking_scores.items():
            d = self._faces.setdefault(fid, deque([0.0] * (n - 1), maxlen=self.window))
            d.append(float(s))
        return self.active()

    def correlations(self) -> dict[int, float]:
        audio = np.fromiter(self._audio, dtype=float)
        series = {fid: np.fromiter(d, dtype=float) for fid, d in self._faces.items()}
        if self.model is not None:
            return self.model.scores(series, audio)
        return {fid: pearson(s, audio[-len(s) :]) for fid, s in series.items()}

    def active(self) -> int | None:
        if not self._faces:
            return None
        if len(self._faces) == 1:  # single face: no one to confuse it with; require it to be moving its mouth
            fid, d = next(iter(self._faces.items()))
            return fid if d and max(list(d)[-5:]) >= self.speaking_threshold else None
        corr = self.correlations()
        fid = max(corr, key=lambda k: corr[k])
        return fid if corr[fid] >= self.min_corr else None
