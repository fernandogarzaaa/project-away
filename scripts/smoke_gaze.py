"""Smoke-test head pose, iris gaze features and the optional gaze networks on still images.

python scripts/smoke_gaze.py path/to/face_landmarker.task [--net model.onnx ...] img.jpg ...

Prints, per image (as given and mirrored): the head pose from the facial transformation matrix, the
screen-aligned nose offsets, the iris feature vector, and each network's yaw/pitch with its time per run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(argv: list[str]) -> int:
    import cv2

    from away.face.gaze_net import GazeNet, default_session
    from away.face.landmarker import FaceLandmarker

    model, rest = Path(argv[0]), argv[1:]
    nets: list[Path] = []
    images: list[str] = []
    while rest:
        a = rest.pop(0)
        if a == "--net" and rest:
            nets.append(Path(rest.pop(0)))
        else:
            images.append(a)
    lm = FaceLandmarker(model, max_faces=1)
    t = 0.0
    try:
        for path in images:
            img = cv2.imread(path)
            if img is None:
                print(f"{path}: cannot read")
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for mirrored in (False, True):
                im = np.ascontiguousarray(rgb[:, ::-1]) if mirrored else rgb
                t += 1.0
                obs = lm.process(im, t)
                print(f"{path} {'mirrored' if mirrored else 'as is'}: {len(obs)} face(s)")
                for o in obs:
                    print(f"  pose (yaw, pitch, roll) {np.round(o.pose, 1)}  nose {np.round(o.proxy, 3)}")
                    print(f"  iris features {np.round(o.gaze, 3) if o.gaze is not None else None}")
                    for n in nets:
                        net = GazeNet(n, n.stem, 1000.0, session_factory=default_session)
                        ang = net.update(0.0, im, o.box)
                        times = []
                        for _ in range(5):
                            net.last_t = None
                            net.update(0.0, im, o.box)
                            times.append(net.last_ms)
                        deg = np.round(np.degrees(ang), 1) if ang is not None else None
                        print(f"  {n.name}: yaw/pitch {deg} deg, {np.median(times):.0f} ms per run")
    finally:
        lm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
