"""MediaPipe's detector + landmark pipeline in plain numpy (v0.1.24), for running the MediaPipe models as ONNX.

The MediaPipe Tasks Python package has no GPU support on Windows ("GPU support is currently limited to Ubuntu
platforms"), so on Windows the same Google models (Apache-2.0), converted to ONNX, run in ONNX Runtime with the
DirectML (any DirectX 12 card) or CUDA execution provider. The glue MediaPipe does around the models is
re-implemented here, following its calculators:

* SSD anchors (``SsdAnchorsCalculator``), box + keypoint decoding and weighted non-max suppression
  (``TensorsToDetectionsCalculator``, ``NonMaxSuppressionCalculator``), letterbox removal;
* the rotated region of interest from a detection or from the previous frame's landmarks
  (``DetectionsToRectsCalculator``, ``HandLandmarksToRectCalculator``, ``RectTransformationCalculator``);
* cropping that rotated region to the model input (``ImageToTensorCalculator``) and projecting the landmarks back
  (``LandmarkProjectionCalculator``).

Coordinates: images are (H, W, 3) RGB uint8; detections and landmarks are normalized to [0, 1] like MediaPipe's.
Pure functions, no ONNX Runtime import: the tests check them without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FArr = NDArray[np.float32]


# ------------------------------------------------------------------------------------------------- anchors
def ssd_anchors(
    input_size: int,
    strides: tuple[int, ...],
    num_layers: int | None = None,
    min_scale: float = 0.1484375,
    max_scale: float = 0.75,
    offset: float = 0.5,
) -> FArr:
    """Anchor centres (N, 2) for MediaPipe's palm / face detectors (fixed anchor size, aspect ratio 1 plus the
    interpolated scale: two anchors per cell per layer; layers that share a stride are generated together)."""
    del min_scale, max_scale  # only the centres matter with fixed_anchor_size
    n = num_layers or len(strides)
    out: list[tuple[float, float]] = []
    layer = 0
    while layer < n:
        count = 0
        last = layer
        while last < n and strides[last] == strides[layer]:
            count += 2  # aspect ratio 1.0 + interpolated_scale_aspect_ratio 1.0
            last += 1
        stride = strides[layer]
        fm = int(math.ceil(input_size / stride))
        for y in range(fm):
            for x in range(fm):
                c = ((x + offset) / fm, (y + offset) / fm)
                out.extend([c] * count)
        layer = last
    return np.asarray(out, dtype=np.float32)


# ------------------------------------------------------------------------------------------------ decoding
@dataclass
class Detection:
    score: float
    box: tuple[float, float, float, float]  # xmin, ymin, w, h (normalized)
    keypoints: NDArray[np.float32]  # (K, 2) normalized


def sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100.0, 100.0)))


def decode(
    raw_boxes: NDArray[Any],
    raw_scores: NDArray[Any],
    anchors: FArr,
    input_size: int,
    num_keypoints: int,
    min_score: float = 0.5,
) -> list[Detection]:
    """``raw_boxes`` (N, 4 + 2K), ``raw_scores`` (N,) or (N, 1) logits -> detections above ``min_score``."""
    boxes = np.asarray(raw_boxes, dtype=np.float32).reshape(len(anchors), -1)
    scores = sigmoid(np.asarray(raw_scores, dtype=np.float32).reshape(-1))
    keep = np.nonzero(scores >= min_score)[0]
    out = []
    s = float(input_size)
    for i in keep:
        b = boxes[i]
        ax, ay = anchors[i]
        cx, cy = b[0] / s + ax, b[1] / s + ay
        w, h = b[2] / s, b[3] / s
        kp = b[4 : 4 + 2 * num_keypoints].reshape(num_keypoints, 2) / s + np.array([ax, ay], np.float32)
        out.append(Detection(float(scores[i]), (float(cx - w / 2), float(cy - h / 2), float(w), float(h)), kp))
    return out


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def weighted_nms(dets: list[Detection], threshold: float = 0.3, max_out: int = 2) -> list[Detection]:
    """MediaPipe's WEIGHTED suppression: overlapping detections are averaged, weighted by score."""
    rest = sorted(dets, key=lambda d: -d.score)
    out: list[Detection] = []
    while rest and len(out) < max_out:
        top = rest[0]
        group = [d for d in rest if iou(top.box, d.box) > threshold]
        rest = [d for d in rest if iou(top.box, d.box) <= threshold]
        w = np.array([d.score for d in group], np.float32)
        tot = float(w.sum()) or 1.0
        box = tuple(float(x) for x in (np.array([d.box for d in group], np.float32) * w[:, None]).sum(0) / tot)
        kp = (np.stack([d.keypoints for d in group]) * w[:, None, None]).sum(0) / tot
        out.append(Detection(top.score, box, kp.astype(np.float32)))  # type: ignore[arg-type]
    return out


# ------------------------------------------------------------------------------------------------ letterbox
def letterbox(image: NDArray[np.uint8], size: int) -> tuple[NDArray[np.uint8], tuple[float, float, float, float]]:
    """Scale ``image`` into a ``size`` square keeping its aspect ratio (zero border). Returns the square and the
    padding as fractions (left, top, right, bottom) for :func:`unletterbox`."""
    import cv2

    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out, (x0 / size, y0 / size, (size - nw - x0) / size, (size - nh - y0) / size)


def unletterbox(d: Detection, pad: tuple[float, float, float, float]) -> Detection:
    left, top, right, bottom = pad
    sx, sy = 1.0 - left - right, 1.0 - top - bottom
    x, y, w, h = d.box
    box = ((x - left) / sx, (y - top) / sy, w / sx, h / sy)
    kp = (d.keypoints - np.array([left, top], np.float32)) / np.array([sx, sy], np.float32)
    return Detection(d.score, box, kp.astype(np.float32))


# ------------------------------------------------------------------------------------------------------ ROI
@dataclass
class Rect:
    cx: float  # normalized centre
    cy: float
    w: float  # normalized size
    h: float
    rotation: float  # radians


def normalize_radians(a: float) -> float:
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def rotation_from(p0: Any, p1: Any, img_w: int, img_h: int, target_deg: float) -> float:
    x0, y0 = float(p0[0]) * img_w, float(p0[1]) * img_h
    x1, y1 = float(p1[0]) * img_w, float(p1[1]) * img_h
    return normalize_radians(math.radians(target_deg) - math.atan2(-(y1 - y0), x1 - x0))


def transform(
    r: Rect, img_w: int, img_h: int, scale_x: float, scale_y: float, shift_x: float = 0.0, shift_y: float = 0.0
) -> Rect:
    """``RectTransformationCalculator`` with ``square_long``: shift in the rotated frame, square, scale."""
    w_px, h_px = r.w * img_w, r.h * img_h
    c, s = math.cos(r.rotation), math.sin(r.rotation)
    dx = w_px * shift_x * c - h_px * shift_y * s
    dy = w_px * shift_x * s + h_px * shift_y * c
    side = max(w_px, h_px)
    return Rect(r.cx + dx / img_w, r.cy + dy / img_h, side * scale_x / img_w, side * scale_y / img_h, r.rotation)


def rect_from_detection(
    d: Detection, img_w: int, img_h: int, start: int, end: int, target_deg: float, use_keypoints_box: bool = False
) -> Rect:
    x, y, w, h = d.box
    rot = rotation_from(d.keypoints[start], d.keypoints[end], img_w, img_h, target_deg)
    return Rect(x + w / 2, y + h / 2, w, h, rot)


def rect_from_points(points: NDArray[Any], img_w: int, img_h: int, rotation: float) -> Rect:
    """The rotated box around ``points`` (normalized (N, 2+)): rotate them by -rotation about their centroid,
    take the axis-aligned box, and rotate its centre back (``HandLandmarksToRectCalculator``)."""
    px = np.asarray(points, np.float64)[:, 0] * img_w
    py = np.asarray(points, np.float64)[:, 1] * img_h
    cx0, cy0 = float(px.mean()), float(py.mean())
    c, s = math.cos(-rotation), math.sin(-rotation)
    rx = (px - cx0) * c - (py - cy0) * s
    ry = (px - cx0) * s + (py - cy0) * c
    mnx, mxx, mny, mxy = float(rx.min()), float(rx.max()), float(ry.min()), float(ry.max())
    mcx, mcy = (mnx + mxx) / 2, (mny + mxy) / 2
    c2, s2 = math.cos(rotation), math.sin(rotation)
    cx = cx0 + mcx * c2 - mcy * s2
    cy = cy0 + mcx * s2 + mcy * c2
    return Rect(cx / img_w, cy / img_h, (mxx - mnx) / img_w, (mxy - mny) / img_h, rotation)


HAND_ROI_POINTS = (0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)


def hand_rect_from_landmarks(lm: NDArray[Any], img_w: int, img_h: int) -> Rect:
    """Next frame's hand region from this frame's 21 landmarks (normalized)."""
    p = np.asarray(lm, np.float64)
    x1 = ((p[5, 0] + p[13, 0]) / 2 + p[9, 0]) / 2
    y1 = ((p[5, 1] + p[13, 1]) / 2 + p[9, 1]) / 2
    rot = rotation_from(p[0], (x1, y1), img_w, img_h, 90.0)
    r = rect_from_points(p[list(HAND_ROI_POINTS)], img_w, img_h, rot)
    return transform(r, img_w, img_h, 2.0, 2.0, 0.0, -0.1)


def face_rect_from_landmarks(lm: NDArray[Any], img_w: int, img_h: int) -> Rect:
    p = np.asarray(lm, np.float64)
    rot = rotation_from(p[33], p[263], img_w, img_h, 0.0)
    r = rect_from_points(p[:468], img_w, img_h, rot)
    return transform(r, img_w, img_h, 1.5, 1.5)


# ------------------------------------------------------------------------------------------- crop + project
def crop_matrix(r: Rect, img_w: int, img_h: int, size: int) -> NDArray[np.float64]:
    """The 2x3 affine matrix that maps the rotated region onto a ``size`` square (for ``cv2.warpAffine``)."""
    cx, cy = r.cx * img_w, r.cy * img_h
    w, h = r.w * img_w, r.h * img_h
    c, s = math.cos(r.rotation), math.sin(r.rotation)

    def corner(u: float, v: float) -> tuple[float, float]:  # u, v in [-0.5, 0.5] of the region
        return cx + u * w * c - v * h * s, cy + u * w * s + v * h * c

    src = np.array([corner(-0.5, -0.5), corner(0.5, -0.5), corner(-0.5, 0.5)], np.float32)
    dst = np.array([[0, 0], [size, 0], [0, size]], np.float32)
    import cv2

    return np.asarray(cv2.getAffineTransform(src, dst), np.float64)


def crop(image: NDArray[np.uint8], r: Rect, size: int) -> NDArray[np.uint8]:
    import cv2

    h, w = image.shape[:2]
    m = crop_matrix(r, w, h, size)
    return np.asarray(
        cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    )


def project(points: NDArray[Any], r: Rect, size: int, img_w: int, img_h: int) -> NDArray[np.float64]:
    """Landmarks in crop pixels (N, 3) -> normalized image coordinates (``LandmarkProjectionCalculator``), done
    in pixels so it is exact for any aspect ratio and rotation; z is scaled like the region's width."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    u = p[:, 0] / size - 0.5
    v = p[:, 1] / size - 0.5
    w, h = r.w * img_w, r.h * img_h
    c, s = math.cos(r.rotation), math.sin(r.rotation)
    x = r.cx * img_w + u * w * c - v * h * s
    y = r.cy * img_h + u * w * s + v * h * c
    return np.stack([x / img_w, y / img_h, p[:, 2] / size * r.w], axis=1)
