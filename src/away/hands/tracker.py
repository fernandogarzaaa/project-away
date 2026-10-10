"""Camera capture + MediaPipe Hand Landmarker (Tasks API). Needs the [hands] extra.

Untested in the build sandbox (no camera); the landmarker itself was smoke-tested
on a still image there.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class HandFrame:
    t: float
    landmarks: NDArray[np.float64] | None  # (21, 3) image coords of the first hand
    handedness: str
    image: Any = None
    score: float = 1.0  # MediaPipe handedness confidence (used by multi-camera fusion)


class HandTracker:
    def __init__(self, model_path: Path, max_hands: int = 1, min_conf: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, rgb: NDArray[np.uint8], t: float) -> HandFrame:
        ms = int(t * 1000)
        if ms <= self._last_ms:  # VIDEO mode needs strictly increasing timestamps
            ms = self._last_ms + 1
        self._last_ms = ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(image, ms)
        if not res.hand_landmarks:
            return HandFrame(t, None, "Right")
        lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float64)
        hand = res.handedness[0][0].category_name if res.handedness else "Right"
        score = float(res.handedness[0][0].score) if res.handedness else 1.0
        return HandFrame(t, lm, hand, score=score)

    def close(self) -> None:
        self.landmarker.close()


def camera_frames(
    index: int = 0, mirror: bool = True, width: int = 640, height: int = 480
) -> Iterator[tuple[float, Any]]:
    """Yield (monotonic time, RGB frame). Mirrored by default so motion matches the screen."""
    import cv2

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index} (permission denied or no device?)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            yield time.monotonic(), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def probe_camera(index: int = 0) -> tuple[bool, str]:
    try:
        import cv2
    except ImportError:
        return False, "opencv not installed (pip install 'project-away[hands]')"
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return False, f"camera {index} could not be opened (no device or permission denied)"
        ok, _ = cap.read()
        return (True, f"camera {index} OK") if ok else (False, f"camera {index} opened but returned no frame")
    finally:
        cap.release()
