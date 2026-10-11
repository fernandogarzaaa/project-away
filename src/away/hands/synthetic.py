"""Procedural hand landmarks and gesture sequences.

Used for unit tests and as a *synthetic* pre-training set for the temporal
model. Nothing trained only on this data should be reported as real-world
accuracy.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from away.hands.landmarks import INDEX_TIP, NUM_LANDMARKS

Array = NDArray[np.float64]

GESTURES = (
    "idle",
    "move",
    "pinch_click",
    "double_pinch",
    "pinch_drag",
    "scroll",
    "palm_clutch",
    "swipe_left",
    "swipe_right",
)

_MCP = {
    "index": (0.30, -0.95),
    "middle": (0.0, -1.0),
    "ring": (-0.25, -0.95),
    "pinky": (-0.48, -0.85),
}
_FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
_SEG = (0.45, 0.30, 0.25)


def make_hand(
    curls: dict[str, float] | None = None,
    pinch: float = 0.0,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.12,
    rotation: float = 0.0,
    handedness: str = "Right",
    thumb_fold: float = 0.0,
) -> Array:
    """Return (21, 3) landmarks in image coordinates.

    curls: 0 = straight, 1 = fully curled, per finger (thumb ignored).
    pinch: 0..1 moves the thumb tip onto the index tip.
    thumb_fold: 0..1 tucks the thumb across the palm (1 = folded, the four-finger auto-scroll pose).
    """
    c = {"index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
    if curls:
        c.update(curls)
    lm = np.zeros((NUM_LANDMARKS, 3))
    # thumb chain 1..4
    lm[1] = (0.35, -0.25, 0)
    lm[2] = (0.60, -0.50, 0)
    lm[3] = (0.75, -0.75, 0)
    lm[4] = (0.85, -0.95, 0)
    for name, base in _FINGER_BASE.items():
        mx, my = _MCP[name]
        d = np.array([mx, my]) / math.hypot(mx, my)
        p = np.array([mx, my, 0.0])
        lm[base] = p
        cum = 0.0
        for k, seg in enumerate(_SEG):
            cum += c[name] * math.radians(90)
            p = p + np.array([d[0] * seg * math.cos(cum), d[1] * seg * math.cos(cum), seg * math.sin(cum)])
            lm[base + 1 + k] = p
    if thumb_fold > 0:
        palm = np.array([-0.05, -0.55, 0.0])
        lm[4] = lm[4] + thumb_fold * (palm - lm[4])
        lm[3] = lm[3] + 0.6 * thumb_fold * (np.array([0.2, -0.5, 0.0]) - lm[3])
    if pinch > 0:
        target = lm[INDEX_TIP].copy()
        lm[4] = lm[4] + pinch * (target - lm[4])
        lm[3] = lm[3] + 0.5 * pinch * (target - lm[3])
    # rotate, scale, place; left hand = mirror in x
    ca, sa = math.cos(rotation), math.sin(rotation)
    rot = np.array([[ca, -sa], [sa, ca]])
    xy = lm[:, :2] @ rot.T
    if handedness.lower().startswith("l"):
        xy[:, 0] = -xy[:, 0]
    out = np.zeros_like(lm)
    out[:, :2] = xy * scale + np.array(center)
    out[:, 2] = lm[:, 2] * scale
    return out


POSES: dict[str, dict[str, object]] = {
    "point": {"curls": {"index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}},
    "palm": {"curls": {}},
    "four": {"curls": {}, "thumb_fold": 1.0},  # four fingers up, thumb folded (auto-scroll)
    "fist": {"curls": {"index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}},
    "two": {"curls": {"index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0}},
    "pinch": {"curls": {"index": 0.35, "middle": 0.6, "ring": 0.8, "pinky": 0.8}, "pinch": 1.0},
    "pinch_open": {"curls": {"index": 0.35, "middle": 0.6, "ring": 0.8, "pinky": 0.8}, "pinch": 0.0},
}


def pose(name: str, **kw: object) -> Array:
    spec = dict(POSES[name])
    spec.update(kw)
    return make_hand(**spec)  # type: ignore[arg-type]


def _interp(a: float, b: float, n: int) -> Array:
    return np.linspace(a, b, n)


def gesture_sequence(
    label: str,
    frames: int = 24,
    rng: np.random.Generator | None = None,
    noise: float = 0.002,
    handedness: str = "Right",
) -> Array:
    """A (frames, 21, 3) synthetic sequence for one gesture label."""
    if label not in GESTURES:
        raise ValueError(f"unknown gesture {label!r}")
    r = rng or np.random.default_rng()
    cx, cy = r.uniform(0.35, 0.65), r.uniform(0.35, 0.65)
    scale = r.uniform(0.08, 0.16)
    rot = r.uniform(-0.35, 0.35)
    xs = np.full(frames, cx)
    ys = np.full(frames, cy)
    pinch = np.zeros(frames)
    base = "point"
    if label == "idle":
        base = str(r.choice(["fist", "palm", "point", "pinch_open"]))
        xs += np.cumsum(r.normal(0, 0.003, frames)) * 0.2
    elif label == "move":
        ang = r.uniform(0, 2 * math.pi)
        dist = r.uniform(0.08, 0.25)
        xs = cx + _interp(0, dist * math.cos(ang), frames)
        ys = cy + _interp(0, dist * math.sin(ang), frames)
    elif label in ("pinch_click", "double_pinch", "pinch_drag"):
        base = "pinch_open"
        if label == "pinch_click":
            s = int(r.integers(4, 10))
            pinch[s : s + int(r.integers(3, 6))] = 1.0
        elif label == "double_pinch":
            s = int(r.integers(2, 6))
            w = int(r.integers(2, 4))
            pinch[s : s + w] = 1.0
            s2 = s + w + int(r.integers(2, 4))
            pinch[s2 : s2 + w] = 1.0
        else:
            s = int(r.integers(2, 5))
            pinch[s:] = 1.0
            ang = r.uniform(0, 2 * math.pi)
            dist = r.uniform(0.08, 0.2)
            xs[s:] = cx + _interp(0, dist * math.cos(ang), frames - s)
            ys[s:] = cy + _interp(0, dist * math.sin(ang), frames - s)
    elif label == "scroll":
        base = "two"
        ys = cy + _interp(0, r.choice([-1, 1]) * r.uniform(0.08, 0.2), frames)
    elif label == "palm_clutch":
        base = "palm"
        xs += r.normal(0, 0.001, frames)
    elif label in ("swipe_left", "swipe_right"):
        base = "palm"
        sign = 1 if label == "swipe_right" else -1
        dist = r.uniform(0.25, 0.45)
        prog = np.clip((np.arange(frames) - frames * 0.2) / (frames * 0.5), 0, 1)
        prog = 0.5 - 0.5 * np.cos(np.pi * prog)
        xs = cx - sign * dist / 2 + sign * dist * prog
    spec = POSES[base]
    seq = []
    for i in range(frames):
        hand = make_hand(
            curls=spec.get("curls"),  # type: ignore[arg-type]
            pinch=float(pinch[i]) if base == "pinch_open" else float(spec.get("pinch", 0.0)),  # type: ignore[arg-type]
            center=(float(xs[i]), float(ys[i])),
            scale=scale,
            rotation=rot,
            handedness=handedness,
        )
        hand[:, :2] += r.normal(0, noise, (NUM_LANDMARKS, 2))
        seq.append(hand)
    return np.stack(seq)


def make_dataset(n_per_class: int = 200, frames: int = 24, seed: int = 0) -> tuple[Array, NDArray[np.int64], list[str]]:
    """Synthetic (N, frames, 21, 3) landmarks + labels, mixed handedness."""
    r = np.random.default_rng(seed)
    X, y, hands = [], [], []
    for ci, label in enumerate(GESTURES):
        for _ in range(n_per_class):
            h = "Left" if r.random() < 0.3 else "Right"
            X.append(gesture_sequence(label, frames, r, handedness=h))
            y.append(ci)
            hands.append(h)
    return np.stack(X), np.array(y, dtype=np.int64), hands
