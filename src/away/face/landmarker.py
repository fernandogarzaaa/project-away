"""MediaPipe Face Landmarker (Tasks API) with blendshapes. Needs the [face] extra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FaceObs:
    box: tuple[float, float, float]  # cx, cy, size (normalized)
    blendshapes: dict[str, float]


class FaceLandmarker:  # pragma: no cover - needs mediapipe + images
    def __init__(self, model_path: Path, max_faces: int = 4) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, rgb: Any, t: float) -> list[FaceObs]:
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(img, ms)
        out = []
        for i, lms in enumerate(res.face_landmarks):
            xs = np.array([p.x for p in lms])
            ys = np.array([p.y for p in lms])
            box = (float(xs.mean()), float(ys.mean()), float(max(np.ptp(xs), np.ptp(ys))))
            bs = {c.category_name: float(c.score) for c in res.face_blendshapes[i]} if res.face_blendshapes else {}
            out.append(FaceObs(box, bs))
        return out

    def close(self) -> None:
        self.landmarker.close()
