"""Face tracking with the MediaPipe face models as ONNX (v0.1.24): the detector, the 478-point face mesh and the
blendshape model from ``face_landmarker.task`` (Google, Apache-2.0), run by ONNX Runtime on the graphics card.

:class:`OnnxFaceLandmarker` has the same ``process(rgb, t) -> list[FaceObs]`` as
:class:`away.face.landmarker.FaceLandmarker`, so the face gate, head pose and gaze features work unchanged. The
head pose comes from the mesh points (``pose_from_points``) instead of MediaPipe's transformation matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from away.accel import mp_pipeline as mp
from away.accel.onnx_hands import SAME_HAND_IOU, _overlap, session

DET_SIZE = 128
MESH_SIZE = 256
PRESENCE_MIN = 0.5
# the 146 mesh points the blendshape model reads (MediaPipe face_blendshapes_graph.cc kLandmarksSubsetIdxs)
BLENDSHAPE_POINTS = (
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
    154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249,
    251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314,
    317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473,
    474, 475, 476, 477,
)  # fmt: skip
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)  # fmt: skip


DETECT_EVERY = 10  # frames between searches for another face while at least one is tracked


class OnnxFaceLandmarker:
    def __init__(
        self,
        detector: Path | Any,
        mesh: Path | Any,
        blendshapes: Path | Any | None = None,
        max_faces: int = 1,
        providers: list[str] | None = None,
        min_detection: float = 0.5,
    ) -> None:
        prov = providers or ["CPUExecutionProvider"]

        def load(m: Any) -> Any:
            return session(m, prov) if isinstance(m, (str, Path)) else m

        self.det, self.mesh = load(detector), load(mesh)
        self.bs = load(blendshapes) if blendshapes is not None else None
        self.max_faces = max(1, int(max_faces))
        self.min_detection = float(min_detection)
        self.anchors = mp.ssd_anchors(DET_SIZE, (8, 16, 16, 16))
        self.tracked: list[mp.Rect] = []
        self._frames = 0  # look for new faces every DETECT_EVERY frames while fewer than max_faces are tracked
        self.det_runs = 0

    def _detect(self, rgb: np.ndarray) -> list[mp.Rect]:
        self.det_runs += 1
        h, w = rgb.shape[:2]
        sq, pad = mp.letterbox(rgb, DET_SIZE)
        x = (sq.astype(np.float32) / 127.5 - 1.0)[None]
        boxes, scores = self.det.run(None, {self.det.get_inputs()[0].name: x})[:2]
        dets = mp.decode(boxes[0], scores[0], self.anchors, DET_SIZE, 6, self.min_detection)
        dets = [mp.unletterbox(d, pad) for d in mp.weighted_nms(dets, 0.3, self.max_faces)]
        return [mp.transform(mp.rect_from_detection(d, w, h, 0, 1, 0.0), w, h, 1.5, 1.5) for d in dets]

    def _mesh(self, rgb: np.ndarray, r: mp.Rect) -> np.ndarray | None:
        h, w = rgb.shape[:2]
        x = (mp.crop(rgb, r, MESH_SIZE).astype(np.float32) / 255.0)[None]
        outs = self.mesh.run(None, {self.mesh.get_inputs()[0].name: x})
        pts = np.asarray(outs[0]).reshape(-1, 3)
        presence = float(mp.sigmoid(float(np.ravel(outs[1])[0])))
        if presence < PRESENCE_MIN:
            return None
        return mp.project(pts, r, MESH_SIZE, w, h)

    def blendshapes(self, lm: np.ndarray, w: int, h: int) -> dict[str, float]:
        if self.bs is None:
            return {}
        sub = np.asarray(lm, np.float32)[list(BLENDSHAPE_POINTS), :2] * np.array([w, h], np.float32)
        out = np.ravel(self.bs.run(None, {self.bs.get_inputs()[0].name: sub[None]})[0])
        return {n: float(v) for n, v in zip(BLENDSHAPE_NAMES, out)}

    def detect(self, rgb: np.ndarray) -> list[tuple[np.ndarray, dict[str, float]]]:
        """Every face: (478, 3) normalized mesh points and the blendshape scores."""
        h, w = rgb.shape[:2]
        rois = list(self.tracked)
        self._frames += 1
        if not rois or (len(rois) < self.max_faces and self._frames % DETECT_EVERY == 0):
            for r in self._detect(rgb):  # someone new in view (the tracked faces alone never find them)
                if len(rois) < self.max_faces and all(_overlap(r, t, w, h) < SAME_HAND_IOU for t in rois):
                    rois.append(r)
        out: list[tuple[np.ndarray, dict[str, float]]] = []
        nxt: list[mp.Rect] = []
        for r in rois[: self.max_faces]:
            lm = self._mesh(rgb, r)
            if lm is None:
                continue
            nr = mp.face_rect_from_landmarks(lm, w, h)
            if any(_overlap(nr, t, w, h) > SAME_HAND_IOU for t in nxt):
                continue  # two regions found the same face
            nxt.append(nr)
            out.append((lm, self.blendshapes(lm, w, h)))
        self.tracked = nxt
        return out

    def process(self, rgb: Any, t: float) -> list[Any]:
        from away.face.landmarker import observe_face

        arr = np.asarray(rgb)
        return [observe_face(lm[:, 0], lm[:, 1], bs, None) for lm, bs in self.detect(arr)]

    def close(self) -> None:
        self.tracked = []
