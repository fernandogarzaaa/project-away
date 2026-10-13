"""MediaPipe Face Landmarker (Tasks API) with blendshapes. Needs the [face] extra.

v0.1.19: the landmarker also outputs the facial transformation matrix (head pose, no extra model), and each
:class:`FaceObs` carries the head pose (:mod:`away.face.headpose`), the screen-aligned nose offsets and the
iris-based gaze features (:mod:`away.face.gaze`). Computing them is a few dozen arithmetic operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# MediaPipe face mesh indices: nose tip, outer corners of the two eyes, chin, point between the brows
NOSE_TIP, EYE_OUTER_A, EYE_OUTER_B, CHIN, BROWS_MID = 1, 33, 263, 152, 168
FRONTAL_MAX_YAW = 0.35  # nose offset from the eye midpoint / eye span (~35 degrees of head turn)
FRONTAL_PITCH_RANGE = (0.2, 0.8)  # nose height between the brows (0) and the chin (1)


def frontal_from_points(xs: Any, ys: Any, max_yaw: float = FRONTAL_MAX_YAW) -> bool | None:
    """Does a face look roughly at the camera? From the face mesh: the nose tip sits near the middle of the
    eyes (yaw) and between the brows and the chin (pitch). None when the mesh is too small to tell."""
    try:
        if len(xs) <= max(NOSE_TIP, EYE_OUTER_A, EYE_OUTER_B, CHIN, BROWS_MID):
            return None
        span = float(xs[EYE_OUTER_B]) - float(xs[EYE_OUTER_A])
        height = float(ys[CHIN]) - float(ys[BROWS_MID])
        if abs(span) < 1e-6 or abs(height) < 1e-6:
            return None
        mid = (float(xs[EYE_OUTER_A]) + float(xs[EYE_OUTER_B])) / 2.0
        yaw = abs(float(xs[NOSE_TIP]) - mid) / abs(span)
        pitch = (float(ys[NOSE_TIP]) - float(ys[BROWS_MID])) / height
    except (TypeError, ValueError, IndexError):
        return None
    lo, hi = FRONTAL_PITCH_RANGE
    return bool(yaw <= max_yaw and lo <= pitch <= hi)


@dataclass
class FaceObs:
    box: tuple[float, float, float]  # cx, cy, size (normalized)
    blendshapes: dict[str, float]
    frontal: bool | None = None  # looks roughly at the camera (None = unknown)
    pose: tuple[float, float, float] | None = None  # yaw, pitch, roll degrees (matrix convention, unaligned)
    proxy: tuple[float, float] | None = None  # screen-aligned nose offsets (headpose.nose_offsets)
    gaze: Any = None  # iris gaze feature vector (np.ndarray) or None


def observe_face(xs: Any, ys: Any, blendshapes: dict[str, float], matrix: Any = None) -> FaceObs:
    """Everything the app needs from one face mesh (pure; used by the landmarker and the tests)."""
    from away.face.gaze import gaze_features
    from away.face.headpose import nose_offsets, pose_from_matrix, pose_from_points

    box = (float(np.mean(xs)), float(np.mean(ys)), float(max(np.ptp(xs), np.ptp(ys))))
    hp = pose_from_matrix(matrix) if matrix is not None else None
    if hp is None:
        hp = pose_from_points(xs, ys)
    pose = (hp.yaw, hp.pitch, hp.roll) if hp is not None else None
    return FaceObs(
        box,
        blendshapes,
        frontal_from_points(xs, ys),
        pose,
        nose_offsets(xs, ys),
        gaze_features(xs, ys, pose, box),
    )


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
            output_facial_transformation_matrixes=True,  # head pose for head tracking / gaze (v0.1.19)
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, rgb: Any, t: float) -> list[FaceObs]:
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(img, ms)
        out = []
        mats = getattr(res, "facial_transformation_matrixes", None) or []
        for i, lms in enumerate(res.face_landmarks):
            xs = np.array([p.x for p in lms])
            ys = np.array([p.y for p in lms])
            bs = {c.category_name: float(c.score) for c in res.face_blendshapes[i]} if res.face_blendshapes else {}
            out.append(observe_face(xs, ys, bs, mats[i] if i < len(mats) else None))
        return out

    def close(self) -> None:
        self.landmarker.close()
