"""Smoke-test MediaPipe Hand Landmarker + the bundled static pose model on still images.

    python scripts/smoke_mediapipe.py path/to/hand_landmarker.task img1.jpg img2.jpg ...

This checks the integration (Tasks API -> landmarks -> normalisation -> model).
It is NOT an accuracy benchmark.
"""

from __future__ import annotations

import sys

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from away.hands.landmarks import classify_pose
from away.hands.runtime import StaticPoseClassifier, bundled_model


def main(argv: list[str]) -> int:
    model, images = argv[0], argv[1:]
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), num_hands=2)
    clf = StaticPoseClassifier(bundled_model("static_pose.npz"))  # type: ignore[arg-type]
    with vision.HandLandmarker.create_from_options(opts) as lmk:
        for path in images:
            res = lmk.detect(mp.Image.create_from_file(path))
            if not res.hand_landmarks:
                print(f"{path}: no hand detected")
                continue
            for i, hand in enumerate(res.hand_landmarks):
                pts = [[p.x, p.y, p.z] for p in hand]
                label, conf = clf.predict(pts)
                side = res.handedness[i][0].category_name
                print(f"{path}: hand {i} ({side}) model={label} {conf:.2f} rules={classify_pose(pts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
