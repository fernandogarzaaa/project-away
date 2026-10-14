"""Hand tracking with the MediaPipe hand models as ONNX (v0.1.24): the same models as ``hand_landmarker.task``
(Google, Apache-2.0), run by ONNX Runtime on the graphics card (DirectML / CUDA) or the processor.

:class:`OnnxHandLandmarker` behaves like MediaPipe's ``HandLandmarker`` in VIDEO mode as far as
:class:`away.hands.tracker.HandTracker` is concerned: ``detect_for_video(rgb, ms)`` returns an object with
``hand_landmarks`` (lists of points with ``.x .y .z``, normalized) and ``handedness`` (lists of categories with
``.category_name .score``). Like MediaPipe it runs the palm detector only while it tracks fewer hands than it
looks for, and otherwise follows each hand from its previous landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from away.accel import mp_pipeline as mp

PALM_SIZE = 192
LANDMARK_SIZE = 224
PRESENCE_MIN = 0.5
DETECTION_MIN = 0.5
SAME_HAND_IOU = 0.5


@dataclass
class Point:
    x: float
    y: float
    z: float


@dataclass
class Category:
    category_name: str
    score: float


@dataclass
class HandResult:
    hand_landmarks: list[list[Point]] = field(default_factory=list)
    handedness: list[list[Category]] = field(default_factory=list)


def session(path: Path | str, providers: list[str]) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


class OnnxHandLandmarker:
    def __init__(
        self,
        palm_model: Path | Any,
        landmark_model: Path | Any,
        num_hands: int = 1,
        providers: list[str] | None = None,
        min_detection: float = DETECTION_MIN,
        min_presence: float = PRESENCE_MIN,
    ) -> None:
        prov = providers or ["CPUExecutionProvider"]
        self.palm = session(palm_model, prov) if isinstance(palm_model, (str, Path)) else palm_model
        self.lmk = session(landmark_model, prov) if isinstance(landmark_model, (str, Path)) else landmark_model
        self.num_hands = max(1, int(num_hands))
        self.min_detection = float(min_detection)
        self.min_presence = float(min_presence)
        self.anchors = mp.ssd_anchors(PALM_SIZE, (8, 16, 16, 16))
        self.tracked: list[mp.Rect] = []
        self.palm_runs = 0  # how often the palm detector ran (tests / diagnostics)
        self.providers = list(self.lmk.get_providers()) if hasattr(self.lmk, "get_providers") else prov

    # ------------------------------------------------------------------ models
    def _detect_palms(self, rgb: np.ndarray) -> list[mp.Rect]:
        self.palm_runs += 1
        h, w = rgb.shape[:2]
        sq, pad = mp.letterbox(rgb, PALM_SIZE)
        x = (sq.astype(np.float32) / 255.0)[None]
        name = self.palm.get_inputs()[0].name
        boxes, scores = self.palm.run(None, {name: x})[:2]
        dets = mp.decode(boxes[0], scores[0], self.anchors, PALM_SIZE, 7, self.min_detection)
        dets = [mp.unletterbox(d, pad) for d in mp.weighted_nms(dets, 0.3, self.num_hands)]
        return [mp.transform(mp.rect_from_detection(d, w, h, 0, 2, 90.0), w, h, 2.6, 2.6, 0.0, -0.5) for d in dets]

    def _landmarks(self, rgb: np.ndarray, r: mp.Rect) -> tuple[np.ndarray, float, float] | None:
        h, w = rgb.shape[:2]
        patch = mp.crop(rgb, r, LANDMARK_SIZE)
        x = (patch.astype(np.float32) / 255.0)[None]
        name = self.lmk.get_inputs()[0].name
        outs = self.lmk.run(None, {name: x})
        pts, presence, handed = (
            np.asarray(outs[0]).reshape(21, 3),
            float(np.ravel(outs[1])[0]),
            float(np.ravel(outs[2])[0]),
        )
        if presence < 0.0 or presence > 1.0:  # a logit: make it a probability
            presence = float(mp.sigmoid(presence))
        if presence < self.min_presence:
            return None
        return mp.project(pts, r, LANDMARK_SIZE, w, h), presence, handed

    # --------------------------------------------------------------- the API
    def detect_for_video(self, image: Any, ms: int) -> HandResult:
        rgb = np.asarray(getattr(image, "numpy_view", lambda: image)())
        h, w = rgb.shape[:2]
        rois = list(self.tracked)
        if len(rois) < self.num_hands:
            for r in self._detect_palms(rgb):
                if all(_overlap(r, t, w, h) < SAME_HAND_IOU for t in rois) and len(rois) < self.num_hands:
                    rois.append(r)
        res = HandResult()
        nxt: list[mp.Rect] = []
        for r in rois:
            got = self._landmarks(rgb, r)
            if got is None:
                continue
            lm, _presence, handed = got
            nr = mp.hand_rect_from_landmarks(lm, w, h)
            if any(_overlap(nr, t, w, h) > SAME_HAND_IOU for t in nxt):
                continue  # two regions found the same hand
            nxt.append(nr)
            res.hand_landmarks.append([Point(float(p[0]), float(p[1]), float(p[2])) for p in lm])
            # the model's handedness score is the probability of a right hand in the (unmirrored) picture
            label = "Right" if handed >= 0.5 else "Left"
            res.handedness.append([Category(label, float(handed if handed >= 0.5 else 1.0 - handed))])
        self.tracked = nxt
        return res

    def close(self) -> None:
        self.tracked = []


def _overlap(a: mp.Rect, b: mp.Rect, w: int, h: int) -> float:
    def box(r: mp.Rect) -> tuple[float, float, float, float]:
        return (r.cx - r.w / 2, r.cy - r.h / 2, r.w, r.h)

    return mp.iou(box(a), box(b))
