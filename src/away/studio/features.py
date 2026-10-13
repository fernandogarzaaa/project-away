"""Model inputs for Studio gestures, shared by training and the live recogniser (so they always match).

Gestures are trained in groups by ``(kind, hands)``: ``static1`` / ``static2`` (one frame, built on
:func:`~away.hands.landmarks.static_features`, the static pose model's input) and ``motion1`` / ``motion2``
(a window resampled to :data:`MOTION_FRAMES` frames, built on :func:`~away.hands.landmarks.sequence_features`,
the temporal TCN's input). Two-hand groups append the second hand (zeros while it is out of view) and where it
is relative to the first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.hands.dataset import fixed_length
from away.hands.landmarks import WRIST, as_array, infer_side, normalize, palm_size, sequence_features, static_features

Array = NDArray[np.float64]
MOTION_FRAMES = 24  # the TCN's window (away.hands.train.SEQ_FRAMES)
HAND_DIMS = 42  # 21 landmarks x (x, y)


def group_key(kind: str, hands: int) -> str:
    return f"{kind}{int(hands)}"


def parse_group(key: str) -> tuple[str, int]:
    return key[:-1], int(key[-1])


def static_dim(hands: int) -> int:
    return HAND_DIMS if hands == 1 else 2 * HAND_DIMS + 2


def motion_frame_dim(hands: int) -> int:
    return HAND_DIMS + 2 if hands == 1 else 2 * HAND_DIMS + 4


def _rel(p: Array, h: Array | None) -> Array:
    if h is None:
        return np.zeros(2)
    s = max(palm_size(p), 1e-6)
    out: Array = (h[WRIST, :2] - p[WRIST, :2]) / s
    return out


def static_vector(pointer: Any, helper: Any = None, hands: int = 1) -> Array:
    p = as_array(pointer)
    f = static_features(p, infer_side(p))
    if hands == 1:
        return f
    h = as_array(helper) if helper is not None else None
    hf = static_features(h, infer_side(h)) if h is not None else np.zeros(HAND_DIMS)
    out: Array = np.concatenate([f, hf, _rel(p, h)])
    return out


def motion_window(
    pointer_seq: Any, handedness: str = "Right", helper_seq: Any = None, helper_seen: Any = None, hands: int = 1
) -> Array:
    """(T, 21, 3) frames -> (MOTION_FRAMES, motion_frame_dim(hands))."""
    seq = np.asarray(pointer_seq, dtype=np.float64)
    idx = np.linspace(0, len(seq) - 1, MOTION_FRAMES).round().astype(int)
    base = sequence_features(fixed_length(seq, MOTION_FRAMES), handedness)
    if hands == 1:
        return base
    extra = np.zeros((MOTION_FRAMES, HAND_DIMS + 2))
    if helper_seq is not None:
        hs = np.asarray(helper_seq, dtype=np.float64)
        seen = np.ones(len(hs), dtype=bool) if helper_seen is None else np.asarray(helper_seen, dtype=bool)
        for k, i in enumerate(idx):
            if seen[i]:
                try:
                    extra[k, :HAND_DIMS] = normalize(hs[i], infer_side(hs[i]))[:, :2].reshape(-1)
                except ValueError:
                    continue
                extra[k, HAND_DIMS:] = _rel(seq[i], hs[i])
    out: Array = np.concatenate([base, extra], axis=1)
    return out


def occlude_static(x: Array, mean: Array, hands: int) -> Array:
    """One copy of ``x`` per landmark (21 per hand) with that landmark's (x, y) set to the training mean."""
    n = 21 * hands
    out = np.repeat(x[None, :], n, axis=0)
    for i in range(n):
        out[i, 2 * i : 2 * i + 2] = mean[2 * i : 2 * i + 2]
    return out


def occlude_motion(w: Array, mean: Array, hands: int) -> Array:
    """``w`` (F, D): one copy per landmark with its (x, y) replaced by the training mean in every frame. The
    second hand's landmarks sit after the first hand's 42 values and its 2 wrist-velocity values."""
    n = 21 * hands
    out = np.repeat(w[None], n, axis=0)
    m = mean.reshape(w.shape)
    for i in range(n):
        off = 2 * i if i < 21 else HAND_DIMS + 2 + 2 * (i - 21)
        out[i, :, off : off + 2] = m[:, off : off + 2]
    return out
