"""RTMPose-m Hand5 + RTMDet-nano on the graphics card - an **experimental** A/B alternative to MediaPipe hands
(v0.1.24 model refresh; ``models_next.hands_landmarker: rtmpose``). MediaPipe stays the default.

Both models are OpenMMLab's ONNX SDK exports (Apache-2.0; the catalog's ``hands-rtmpose-m-hand5``, sha256-checked):

* RTMDet-nano finds hand boxes (320x320 letterbox, BGR, mean/std from its ``pipeline.json``; NMS is in the graph).
  It runs every ``det_every`` frames; in between the box follows the landmarks, as MediaPipe does.
* RTMPose-m crops each box (x1.25, square, 256x256 affine, RGB) and returns SimCC x/y distributions (512 bins = 2x
  the input size); the arg-max gives 21 points in the hand layout MediaPipe uses.

What it does not give (honest gaps): no 3D/world coordinates (``z`` is 0) and no learned left/right label - the
label is a geometric guess (thumb vs little finger), so palm-scroll / two-hand roles may be less steady. Use the A/B
in :mod:`away.hands.ab` on a recorded session before switching; switching is undoable and rolls back by itself
when it is slower.

:class:`RTMPoseLandmarker` exposes ``detect_for_video(image, ms)`` returning a MediaPipe-shaped result, so
:class:`away.hands.tracker.HandTracker` uses it through its ``factory`` argument with no other change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

MODEL_ID = "hands-rtmpose-m-hand5"
DET_SIZE = 320
POSE_SIZE = 256
SIMCC_SPLIT = 2.0
DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)  # BGR
DET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB
POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
THUMB_TIP, PINKY_TIP = 4, 20


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0


@dataclass
class Category:
    category_name: str
    score: float


@dataclass
class Result:
    hand_landmarks: list[list[Point]] = field(default_factory=list)
    handedness: list[list[Category]] = field(default_factory=list)


def providers() -> list[str]:
    """GPU first: CUDA, then DirectML (any DirectX 12 card on Windows), then the CPU."""
    try:
        import onnxruntime as ort

        have = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    order = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
    return [p for p in order if p in have] or ["CPUExecutionProvider"]


def find_models(folder: Path) -> tuple[Path, Path]:
    pose = sorted(folder.glob("**/rtmpose*/end2end.onnx"))
    det = sorted(folder.glob("**/rtmdet*/end2end.onnx"))
    if not pose or not det:
        raise FileNotFoundError("RTMPose hands are not downloaded yet (Settings > Models)")
    return pose[0], det[0]


# ------------------------------------------------------------------------------------------------ geometry
def letterbox(rgb: NDArray[np.uint8], size: int = DET_SIZE) -> tuple[NDArray[np.float32], float]:
    """Resize keeping the aspect ratio, pad bottom/right with 114 -> (BGR float image, scale)."""
    import cv2

    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out[:, :, ::-1].astype(np.float32), r


def box_to_center_scale(box: NDArray[np.float64], padding: float = 1.25) -> tuple[NDArray[np.float64], float]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float64)
    side = max(x2 - x1, y2 - y1) * padding
    return center, max(side, 1.0)


def crop_matrix(center: NDArray[np.float64], side: float, size: int = POSE_SIZE) -> NDArray[np.float64]:
    """2x3 affine taking the square ``side`` around ``center`` to a ``size`` x ``size`` crop."""
    s = size / side
    return np.array([[s, 0.0, size / 2 - s * center[0]], [0.0, s, size / 2 - s * center[1]]], dtype=np.float64)


def decode_simcc(
    simcc_x: NDArray[np.float32], simcc_y: NDArray[np.float32], split: float = SIMCC_SPLIT
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(K, Wx), (K, Wy) -> keypoints (K, 2) in crop pixels and scores (K,) (the smaller of the two peaks)."""
    x = np.asarray(simcc_x, dtype=np.float64)
    y = np.asarray(simcc_y, dtype=np.float64)
    ix, iy = x.argmax(axis=1), y.argmax(axis=1)
    pts = np.stack([ix, iy], axis=1).astype(np.float64) / split
    scores = np.minimum(x.max(axis=1), y.max(axis=1))
    return pts, scores


def to_image(pts: NDArray[np.float64], center: NDArray[np.float64], side: float, size: int = POSE_SIZE) -> Any:
    return (pts - size / 2) * (side / size) + center


def box_from_points(pts: NDArray[np.float64], w: int, h: int) -> NDArray[np.float64]:
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return np.array([max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)], dtype=np.float64)


def guess_label(pts: NDArray[np.float64]) -> str:
    """MediaPipe-style label for the image (a real right hand, palm to a raw camera, shows its thumb on the image's
    left). A geometric guess: it flips when the back of the hand faces the camera."""
    return "Right" if pts[THUMB_TIP, 0] < pts[PINKY_TIP, 0] else "Left"


# ------------------------------------------------------------------------------------------------ model
class RTMPoseLandmarker:
    def __init__(
        self,
        folder: Path | None = None,
        num_hands: int = 1,
        det_every: int = 5,
        min_score: float = 0.3,
        det_session: Any = None,
        pose_session: Any = None,
    ) -> None:
        if det_session is None or pose_session is None:
            import onnxruntime as ort

            if folder is None:
                from away.app.model_fetch import model_dir

                folder = model_dir(MODEL_ID)
            pose_p, det_p = find_models(Path(folder))
            prov = providers()
            det_session = det_session or ort.InferenceSession(str(det_p), providers=prov)
            pose_session = pose_session or ort.InferenceSession(str(pose_p), providers=prov)
        self.det = det_session
        self.pose = pose_session
        self.num_hands = max(1, int(num_hands))
        self.det_every = max(1, int(det_every))
        self.min_score = float(min_score)
        self._boxes: list[NDArray[np.float64]] = []
        self._n = 0
        self.device = "gpu" if any(p != "CPUExecutionProvider" for p in self._providers()) else "cpu"

    def _providers(self) -> list[str]:
        fn = getattr(self.pose, "get_providers", None)
        return list(fn()) if fn else ["CPUExecutionProvider"]

    def detect_boxes(self, rgb: NDArray[np.uint8]) -> list[NDArray[np.float64]]:
        img, r = letterbox(rgb)
        x = ((img - DET_MEAN) / DET_STD).transpose(2, 0, 1)[None].astype(np.float32)
        dets, _labels = self.det.run(None, {"input": x})
        out = []
        for d in np.asarray(dets)[0]:
            if float(d[4]) < self.min_score:
                continue
            out.append(np.array([d[0] / r, d[1] / r, d[2] / r, d[3] / r, d[4]], dtype=np.float64))
        out.sort(key=lambda b: -float(b[4]))
        return out[: self.num_hands]

    def landmarks(self, rgb: NDArray[np.uint8], box: NDArray[np.float64]) -> tuple[Any, Any]:
        import cv2

        center, side = box_to_center_scale(box)
        m = crop_matrix(center, side)
        crop = cv2.warpAffine(rgb, m, (POSE_SIZE, POSE_SIZE), flags=cv2.INTER_LINEAR)
        x = ((crop.astype(np.float32) - POSE_MEAN) / POSE_STD).transpose(2, 0, 1)[None].astype(np.float32)
        sx, sy = self.pose.run(None, {"input": x})
        pts, scores = decode_simcc(np.asarray(sx)[0], np.asarray(sy)[0])
        return to_image(pts, center, side), scores

    def detect_for_video(self, image: Any, ms: int) -> Result:
        rgb = np.asarray(image)
        h, w = rgb.shape[:2]
        self._n += 1
        if not self._boxes or self._n % self.det_every == 0:
            self._boxes = [b[:4] for b in self.detect_boxes(rgb)]
        res = Result()
        kept = []
        for box in self._boxes:
            pts, scores = self.landmarks(rgb, box)
            score = float(np.mean(scores))
            if score < self.min_score:
                continue
            kept.append(box_from_points(pts, w, h))
            res.hand_landmarks.append([Point(float(px) / w, float(py) / h, 0.0) for px, py in pts])
            res.handedness.append([Category(guess_label(pts), min(1.0, score))])
        self._boxes = kept
        return res

    def close(self) -> None:
        pass


def tracker_factory(folder: Path | None = None, det_every: int = 5) -> Any:
    """``factory(num_hands)`` for :class:`away.hands.tracker.HandTracker`."""
    return lambda n: RTMPoseLandmarker(folder, num_hands=n, det_every=det_every)
