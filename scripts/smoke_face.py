"""Smoke-test MediaPipe Face Landmarker blendshapes -> speaking/brow features on still images.

python scripts/smoke_face.py path/to/face_landmarker.task img.jpg ...
"""

from __future__ import annotations

import sys

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from away.face.commands import brow_raise
from away.face.speaking import mouth_openness


def main(argv: list[str]) -> int:
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=argv[0]), output_face_blendshapes=True, num_faces=4
    )
    with vision.FaceLandmarker.create_from_options(opts) as lmk:
        for path in argv[1:]:
            res = lmk.detect(mp.Image.create_from_file(path))
            print(f"{path}: {len(res.face_landmarks)} face(s)")
            for i, cats in enumerate(res.face_blendshapes):
                bs = {c.category_name: c.score for c in cats}
                print(
                    f"  face {i}: {len(bs)} blendshapes, mouth_openness={mouth_openness(bs):.3f}, "
                    f"brow_raise={brow_raise(bs):.3f}, jawOpen={bs.get('jawOpen', 0):.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
