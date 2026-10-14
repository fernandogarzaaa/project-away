"""A/B two hand landmarkers on the same recorded frames (v0.1.24; MediaPipe vs the experimental RTMPose).

Nobody has published a webcam jitter comparison of the two, so the switch is decided on your own camera:

* **jitter** - while the hand rests, how much each landmark shakes, as a fraction of the palm size (lower = steadier);
* **pinch flicker** - how often the thumb-index pinch state flips on frames where the hand holds still (a proxy for
  false clicks);
* **found** - share of frames with a hand;
* **ms per frame** - time per frame on this PC.

:func:`compare` returns both scores and a plain recommendation; RTMPose is only recommended when it is at least as
steady, finds the hand as often, and is not slower.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

PINCH_ON = 0.35  # pinch_ratio below this = pinched (the same order of magnitude Away's gestures use)
REST_MOVE = 0.03  # a frame "rests" when the wrist moved less than 3 % of the palm size


@dataclass
class Score:
    name: str
    frames: int
    found: float
    jitter: float | None
    pinch_flicker: float | None
    ms_per_frame: float

    def row(self) -> dict[str, Any]:
        return {
            "landmarker": self.name,
            "frames": self.frames,
            "found": round(self.found, 3),
            "jitter": None if self.jitter is None else round(self.jitter, 4),
            "pinch_flicker": None if self.pinch_flicker is None else round(self.pinch_flicker, 4),
            "ms_per_frame": round(self.ms_per_frame, 1),
        }


def palm_size(lm: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(lm[0, :2] - lm[9, :2])) or 1e-6


def pinch_ratio(lm: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(lm[4, :2] - lm[8, :2])) / palm_size(lm)


def score_sequence(name: str, seq: list[NDArray[np.float64] | None], ms: float = 0.0) -> Score:
    """Score a sequence of (21, 2+) landmark arrays (None = no hand that frame)."""
    found = [s for s in seq if s is not None]
    jit: list[float] = []
    flips, rest_frames = 0, 0
    prev: NDArray[np.float64] | None = None
    prev_pinch: bool | None = None
    for lm in seq:
        if lm is None:
            prev, prev_pinch = None, None
            continue
        a = np.asarray(lm, dtype=np.float64)[:, :2]
        pinched = pinch_ratio(a) < PINCH_ON
        if prev is not None:
            ps = palm_size(a)
            if float(np.linalg.norm(a[0] - prev[0])) / ps < REST_MOVE:
                rest_frames += 1
                jit.append(float(np.mean(np.linalg.norm(a - prev, axis=1))) / ps)
                if prev_pinch is not None and pinched != prev_pinch:
                    flips += 1
        prev, prev_pinch = a, pinched
    n = max(1, len(seq))
    return Score(
        name,
        len(seq),
        len(found) / n,
        float(np.median(jit)) if jit else None,
        flips / rest_frames if rest_frames else None,
        ms,
    )


def run(name: str, landmarker: Any, frames: Iterable[Any], clock: Callable[[], float] = time.perf_counter) -> Score:
    """Run ``landmarker.detect_for_video`` over ``frames`` (RGB arrays) and score the first hand of each frame."""
    seq: list[NDArray[np.float64] | None] = []
    spent = 0.0
    for i, f in enumerate(frames):
        t = clock()
        res = landmarker.detect_for_video(f, i * 33)
        spent += clock() - t
        hands = getattr(res, "hand_landmarks", None) or []
        seq.append(np.array([[p.x, p.y] for p in hands[0]], dtype=np.float64) if hands else None)
    return score_sequence(name, seq, 1000.0 * spent / max(1, len(seq)))


def compare(a: Score, b: Score, tolerance: float = 0.1) -> dict[str, Any]:
    """``a`` is the current landmarker (MediaPipe), ``b`` the candidate. Recommend ``b`` only when it is no worse."""
    reasons = []
    ok = True
    if b.found < a.found - 0.02:
        ok = False
        reasons.append(f"{b.name} found the hand less often ({b.found:.0%} vs {a.found:.0%})")
    if a.jitter is not None and (b.jitter is None or b.jitter > a.jitter * (1 + tolerance)):
        ok = False
        reasons.append(f"{b.name} was shakier")
    if a.pinch_flicker is not None and b.pinch_flicker is not None and b.pinch_flicker > a.pinch_flicker + 0.01:
        ok = False
        reasons.append(f"{b.name} flickered between pinch and open more often")
    if b.ms_per_frame > a.ms_per_frame * (1 + tolerance):
        ok = False
        reasons.append(f"{b.name} was slower on this PC ({b.ms_per_frame:.0f} vs {a.ms_per_frame:.0f} ms)")
    return {
        "recommend": b.name if ok else a.name,
        "switch": ok,
        "why": "; ".join(reasons) or f"{b.name} was as steady and as fast",
        "scores": [a.row(), b.row()],
    }
