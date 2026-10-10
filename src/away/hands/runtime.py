"""Runtime model wrappers: sliding-window temporal classifier + static pose classifier."""

from __future__ import annotations

from collections import deque
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from away.hands.landmarks import as_array, infer_side, sequence_features, static_features
from away.hands.model import NumpyMLP, OnnxClassifier


def bundled_model(name: str) -> Path | None:
    """Path to a model shipped inside the wheel (``away/data``), if present."""
    try:
        p = resources.files("away").joinpath("data").joinpath(name)
        path = Path(str(p))
        return path if path.exists() else None
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def find_model(name: str, models_dir: Path | None) -> Path | None:
    """User-trained model in ``models_dir`` wins over the bundled one."""
    if models_dir is not None and (models_dir / name).exists():
        return models_dir / name
    return bundled_model(name)


class StaticPoseClassifier:
    def __init__(self, path: Path) -> None:
        self.model = NumpyMLP.load(path)
        self.classes = self.model.classes

    def predict(self, landmarks: object) -> tuple[str, float]:
        lm = as_array(landmarks)
        p = self.model.predict_proba(static_features(lm, infer_side(lm)))[0]
        i = int(np.argmax(p))
        return self.classes[i], float(p[i])


class TemporalGestureClassifier:
    """Keeps the last ``frames`` hand frames and classifies the window."""

    def __init__(self, path: Path, frames: int | None = None) -> None:
        self.model = OnnxClassifier(path)
        self.classes = self.model.classes
        self.frames = int(frames or self.model.meta.get("frames", 24))
        self.buf: deque[NDArray[np.float64]] = deque(maxlen=self.frames)
        self.handedness = "Right"

    def reset(self) -> None:
        self.buf.clear()

    def push(self, landmarks: object, handedness: str = "Right") -> dict[str, float] | None:
        if handedness != self.handedness:
            self.buf.clear()
            self.handedness = handedness
        self.buf.append(as_array(landmarks))
        if len(self.buf) < self.frames:
            return None
        feats = sequence_features(np.stack(self.buf), handedness)[None]
        p = self.model.predict_proba(feats)[0]
        return {c: float(v) for c, v in zip(self.classes, p)}
