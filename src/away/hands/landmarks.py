"""Hand landmark geometry and normalization.

MediaPipe Hand Landmarker gives 21 points (x, y in [0,1] image coords, z
relative depth). Models in this package consume landmarks that are:

* wrist-relative (translation invariant),
* divided by the palm size (wrist -> middle-finger MCP), so scale invariant,
* mirrored so a left hand looks like a right hand (handedness invariant).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
NUM_LANDMARKS = 21

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

Array = NDArray[np.float64]


def as_array(landmarks: object) -> Array:
    arr = np.asarray(landmarks, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != NUM_LANDMARKS or arr.shape[1] not in (2, 3):
        raise ValueError(f"expected (21, 2|3) landmarks, got {arr.shape}")
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((NUM_LANDMARKS, 1))], axis=1)
    return arr


def palm_size(lm: Array) -> float:
    """Wrist to middle-finger MCP distance in the image plane."""
    return float(np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2]))


def normalize(landmarks: object, handedness: str = "Right", eps: float = 1e-6) -> Array:
    """Return (21, 3) wrist-relative, palm-size-scaled, right-hand-canonical landmarks."""
    lm = as_array(landmarks).copy()
    lm = lm - lm[WRIST]
    scale = palm_size(lm)
    if scale < eps:
        raise ValueError("degenerate hand: palm size ~ 0")
    lm = lm / scale
    if handedness.lower().startswith("l"):
        lm[:, 0] = -lm[:, 0]
    return lm


def normalize_sequence(seq: object, handedness: str = "Right") -> Array:
    """(T, 21, 2|3) -> (T, 21, 3) normalized per frame, plus wrist motion kept separately.

    Per-frame normalization removes global motion, which dynamic gestures
    (swipes) need, so :func:`sequence_features` re-adds wrist velocity.
    """
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (T, 21, C) sequence, got {arr.shape}")
    return np.stack([normalize(f, handedness) for f in arr])


def sequence_features(seq: object, handedness: str = "Right") -> Array:
    """Model input: (T, 21*2 + 2) = normalized x,y per landmark + wrist velocity (palm units/frame)."""
    arr = np.asarray(seq, dtype=np.float64)
    norm = normalize_sequence(arr, handedness)[:, :, :2].reshape(len(arr), -1)
    wrist = arr[:, WRIST, :2].astype(np.float64)
    scales = np.array([max(palm_size(as_array(f)), 1e-6) for f in arr])
    vel = np.zeros((len(arr), 2))
    vel[1:] = (wrist[1:] - wrist[:-1]) / scales[1:, None]
    if handedness.lower().startswith("l"):
        vel[:, 0] = -vel[:, 0]
    out: Array = np.concatenate([norm, vel], axis=1)
    return out


def static_features(landmarks: object, handedness: str = "Right") -> Array:
    """Model input for single frames: (42,) normalized x,y."""
    return normalize(landmarks, handedness)[:, :2].reshape(-1)


def infer_side(landmarks: object) -> str:
    """Handedness proxy from geometry: thumb tip right of pinky tip -> "Right".

    Used for the static-pose model because HaGRID has no handedness labels;
    the same rule is applied at train and run time so it stays consistent.
    """
    lm = as_array(landmarks)
    return "Right" if lm[THUMB_TIP, 0] >= lm[PINKY_TIP, 0] else "Left"


def pinch_ratio(lm: Array) -> float:
    """Thumb-tip to index-tip distance divided by palm size (scale invariant)."""
    s = palm_size(lm)
    if s <= 0:
        return float("inf")
    return float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[INDEX_TIP, :2]) / s)


def finger_extended(lm: Array, finger: str) -> bool:
    """A finger is extended when its tip is farther from the wrist than its PIP joint (by 10%)."""
    _mcp, pip, tip = FINGERS[finger]
    d_tip = np.linalg.norm(lm[tip, :2] - lm[WRIST, :2])
    d_pip = np.linalg.norm(lm[pip, :2] - lm[WRIST, :2])
    return bool(d_tip > 1.1 * d_pip)


def extended_fingers(lm: Array) -> dict[str, bool]:
    return {name: finger_extended(lm, name) for name in FINGERS}


def thumb_fold_ratio(lm_raw: object) -> float:
    """How far the thumb is tucked in: distance thumb tip -> pinky MCP over thumb MCP -> pinky MCP.

    About 1.1-1.4 for an open palm (thumb out), below ~0.75 when the thumb is folded across the palm or held
    against the side of the index finger. Scale-, rotation- and handedness-invariant (image plane).
    """
    lm = as_array(lm_raw)
    base = float(np.linalg.norm(lm[THUMB_MCP, :2] - lm[PINKY_MCP, :2]))
    if base <= 1e-9:
        return float("inf")
    return float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[PINKY_MCP, :2]) / base)


def classify_pose(lm_raw: object) -> str:
    """Rule-based static pose: point | two | palm | fist | other (used when no model is trained).

    ``palm`` only counts the four fingers; the gesture state machine splits it into the open palm (thumb out)
    and the four-finger auto-scroll pose (thumb folded) with :func:`thumb_fold_ratio` and hysteresis.
    """
    lm = as_array(lm_raw)
    ext = extended_fingers(lm)
    n = sum(ext.values())
    if n == 4:
        return "palm"
    if n == 0:
        return "fist"
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "two"
    if ext["index"] and n == 1:
        return "point"
    return "other"
