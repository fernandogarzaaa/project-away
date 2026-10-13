"""Gesture measurements, live gauges and per-gesture auto-tuning. Pure logic (numpy only, no camera, no Tk).

Live view
---------
:func:`live_gauges` turns one frame's pointer (and helper) landmarks into the numbers the gesture detectors compare
against their thresholds: pinch distance (``pinch_on``), thumb fold (``autoscroll_thumb_fold``), index-finger
extension (a finger counts as out above :data:`EXTENDED_RATIO`) and how far the pointer's fingertip is off the
helper palm (``palm_scroll.depth_threshold``). Each :class:`Gauge` says whether the gesture condition is met right
now, so the Settings window can show *why* a gesture did not fire. :func:`skeleton_points` fits the hands into a
small canvas.

Auto-tune
---------
"Do the gesture 5 times, then do something else 5 times": each trial is a short clip; :func:`clip_statistic` takes
its most gesture-like value (a low percentile for "fires below" measurements, so one bad frame does not decide), and
:func:`choose_threshold` picks the threshold that best separates the two sets (balanced accuracy, ties broken by
the widest gap, the threshold at the gap's midpoint). :class:`AutoTuneSession` runs the flow and saves the trials as
labelled clips in the ``away record`` format with a ``session_id``, so the personal classifier can train on them.
"""

from __future__ import annotations

import itertools
import math
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from away.hands.landmarks import (
    FINGERS,
    INDEX_TIP,
    WRIST,
    as_array,
    pinch_ratio,
    thumb_fold_ratio,
)

# MediaPipe hand skeleton (pairs of landmark indices)
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
EXTENDED_RATIO = 1.1  # landmarks.finger_extended: tip farther from the wrist than 1.1 x the PIP joint


# ----------------------------------------------------------------------------------- measurements
def finite(v: float | None) -> bool:
    return v is not None and math.isfinite(v)


def pinch_distance(lm: Any) -> float | None:
    """Thumb tip to index tip over the palm size (``pinch_on`` compares against this)."""
    if lm is None:
        return None
    v = pinch_ratio(as_array(lm))
    return v if math.isfinite(v) else None


def thumb_fold(lm: Any) -> float | None:
    """:func:`~away.hands.landmarks.thumb_fold_ratio` (``autoscroll_thumb_fold`` compares against this)."""
    if lm is None:
        return None
    v = thumb_fold_ratio(lm)
    return v if math.isfinite(v) else None


def finger_extension(lm: Any) -> dict[str, float]:
    """Per finger: wrist->tip over wrist->PIP distance. Above :data:`EXTENDED_RATIO` the finger counts as out; a
    curled finger reads below 1."""
    a = as_array(lm)
    out: dict[str, float] = {}
    for name, (_mcp, pip, tip) in FINGERS.items():
        d_pip = float(np.linalg.norm(a[pip, :2] - a[WRIST, :2]))
        d_tip = float(np.linalg.norm(a[tip, :2] - a[WRIST, :2]))
        out[name] = d_tip / d_pip if d_pip > 1e-9 else 0.0
    return out


def palm_depth(pointer: Any, helper: Any, mirrored: bool = True) -> float | None:
    """How far the pointer's index fingertip is off the helper's palm plane, in palm lengths (what
    ``palm_scroll.depth_threshold`` compares against). None without both hands or with a degenerate palm."""
    if pointer is None or helper is None:
        return None
    from away.hands.palmscroll import hand_points, palm_frame

    frame = palm_frame(hand_points(helper, mirrored=mirrored))
    if frame is None:
        return None
    tip = hand_points(pointer, mirrored=mirrored)[INDEX_TIP]
    return abs(frame.local(tip)[2])


MEASURES: dict[str, Callable[[Any, Any], float | None]] = {
    "pinch": lambda lm, helper: pinch_distance(lm),
    "thumb_fold": lambda lm, helper: thumb_fold(lm),
    "index_extension": lambda lm, helper: finger_extension(lm)["index"] if lm is not None else None,
    "palm_depth": palm_depth,
}


def measure(name: str, lm: Any, helper: Any = None) -> float | None:
    if name not in MEASURES:
        raise ValueError(f"unknown measurement {name!r}; use {', '.join(MEASURES)}")
    try:
        v = MEASURES[name](lm, helper)
    except ValueError:  # malformed landmarks
        return None
    return v if finite(v) else None


# ----------------------------------------------------------------------------------- live gauges
@dataclass(frozen=True)
class Gauge:
    key: str
    label: str
    value: float | None
    threshold: float
    lo: float
    hi: float
    fires: str = "below"  # the gesture condition is value < threshold ("below") or value > threshold ("above")
    hint: str = ""

    @property
    def active(self) -> bool:
        """The gesture condition is met right now (False without a value)."""
        if self.value is None:
            return False
        return self.value < self.threshold if self.fires == "below" else self.value > self.threshold

    @property
    def fraction(self) -> float | None:
        return None if self.value is None else gauge_fraction(self.value, self.lo, self.hi)

    @property
    def threshold_fraction(self) -> float:
        return gauge_fraction(self.threshold, self.lo, self.hi)

    def text(self) -> str:
        if self.value is None:
            return f"{self.label}: (not measured){'  ' + self.hint if self.hint else ''}"
        sign = ("<" if self.active else ">=") if self.fires == "below" else ">" if self.active else "<="
        state = "ON" if self.active else "off"
        return f"{self.label}: {self.value:.2f} {sign} {self.threshold:.2f}  {state}"


def gauge_fraction(value: float, lo: float, hi: float) -> float:
    """``value`` as a 0..1 position on a gauge spanning ``lo``..``hi`` (clamped)."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))


def live_gauges(lm: Any, helper: Any, cfg: Any) -> list[Gauge]:
    """The four live gauges for one frame (``cfg``: the whole :class:`~away.config.AppConfig`)."""
    g, ps = cfg.hands.gestures, cfg.hands.palm_scroll
    ext = finger_extension(lm) if lm is not None else None
    n_up = sum(v > EXTENDED_RATIO for v in ext.values()) if ext else 0
    depth = palm_depth(lm, helper)
    depth_thr = float(ps.depth_threshold)
    return [
        Gauge("pinch", "Pinch distance", pinch_distance(lm), float(g.pinch_on), 0.0, 1.0, "below", "no hand"),
        Gauge(
            "thumb_fold",
            "Thumb fold (auto-scroll)",
            thumb_fold(lm),
            float(g.autoscroll_thumb_fold),
            0.0,
            1.6,
            "below",
            "no hand",
        ),
        Gauge(
            "curl",
            f"Index finger out ({n_up}/4 fingers up)",
            ext["index"] if ext else None,
            EXTENDED_RATIO,
            0.5,
            2.0,
            "above",
            "no hand",
        ),
        Gauge(
            "palm_depth",
            "Fingertip to palm" + (" (no depth check)" if depth_thr <= 0 else ""),
            depth,
            depth_thr if depth_thr > 0 else 3.0,
            0.0,
            2.0,
            "below",
            "needs both hands",
        ),
    ]


def skeleton_points(
    hands: Sequence[Any], width: float, height: float, margin: float = 10.0
) -> list[list[tuple[float, float]]]:
    """Canvas coordinates of each hand's 21 points: all hands together fitted into ``width`` x ``height`` (minus
    ``margin``) with one uniform scale, centred, so a small hand far from the camera still fills the view."""
    arrs = [as_array(h)[:, :2] for h in hands if h is not None]
    if not arrs:
        return []
    allp = np.concatenate(arrs)
    mn, mx = allp.min(axis=0), allp.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    avail = np.array([max(1.0, width - 2 * margin), max(1.0, height - 2 * margin)])
    s = float(min(avail[0] / span[0], avail[1] / span[1]))
    off = (np.array([width, height]) - span * s) / 2.0
    return [[(float(x), float(y)) for x, y in (a - mn) * s + off] for a in arrs]


# ----------------------------------------------------------------------------------- thresholds
MIN_TRIALS = 3
STAT_PERCENTILE = 10.0


def clip_statistic(values: Sequence[float | None], fires: str = "below") -> float | None:
    """A trial's most gesture-like value: the 10th percentile for "below" gestures (90th for "above"), so a single
    glitched frame does not decide. None with fewer than 3 measured frames."""
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 3:
        return None
    q = STAT_PERCENTILE if fires == "below" else 100.0 - STAT_PERCENTILE
    return float(np.percentile(vals, q))


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    balanced_accuracy: float  # 0.5 = no better than chance, 1.0 = every trial on the right side
    separable: bool
    margin: float  # distance from the threshold to the nearest trial value (0 when not separable)
    n_pos: int
    n_neg: int
    clamped: bool = False


def balanced_accuracy(pos: Sequence[float], neg: Sequence[float], threshold: float, fires: str = "below") -> float:
    """Mean of the share of gesture trials that fire and the share of other trials that do not."""
    if fires == "below":
        tp = sum(v < threshold for v in pos)
        tn = sum(v >= threshold for v in neg)
    else:
        tp = sum(v > threshold for v in pos)
        tn = sum(v <= threshold for v in neg)
    return 0.5 * (tp / len(pos) + tn / len(neg))


def choose_threshold(
    pos: Sequence[float],
    neg: Sequence[float],
    fires: str = "below",
    lo: float = -math.inf,
    hi: float = math.inf,
) -> ThresholdChoice:
    """The threshold that best separates gesture trials ``pos`` from other trials ``neg``.

    Candidates are the midpoints between neighbouring values (and a step outside both ends); the one with the
    highest balanced accuracy wins, ties go to the widest gap between its neighbours, then to the lowest value.
    The result is clamped to ``[lo, hi]`` (the slider range)."""
    if fires not in ("below", "above"):
        raise ValueError(f"fires must be 'below' or 'above', got {fires!r}")
    p = [float(v) for v in pos if finite(v)]
    n = [float(v) for v in neg if finite(v)]
    if len(p) < MIN_TRIALS or len(n) < MIN_TRIALS:
        raise ValueError(
            f"need at least {MIN_TRIALS} measured trials of each kind (got {len(p)} gesture, {len(n)} other)"
        )
    vals = sorted(set(p + n))
    edge = max(1e-3, (vals[-1] - vals[0]) * 0.05)
    cands: list[tuple[float, float]] = [(vals[0] - edge, edge), (vals[-1] + edge, edge)]  # (threshold, gap)
    cands += [((a + b) / 2.0, b - a) for a, b in itertools.pairwise(vals)]
    best: tuple[float, float, float] | None = None  # (ba, gap, -threshold)
    best_t = cands[0][0]
    for t, gap in cands:
        key = (balanced_accuracy(p, n, t, fires), gap, -t)
        if best is None or key > best:
            best, best_t = key, t
    assert best is not None
    ba = best[0]
    separable = ba >= 1.0
    margin = min(abs(v - best_t) for v in p + n) if separable else 0.0
    t = max(lo, min(hi, best_t))
    return ThresholdChoice(t, ba, separable, margin, len(p), len(n), clamped=t != best_t)


# ----------------------------------------------------------------------------------- auto-tune flow
@dataclass(frozen=True)
class AutoTuneSpec:
    key: str
    title: str
    path: str  # the tunable it sets
    measure: str  # a MEASURES name
    fires: str
    positive_label: str  # clip labels (away record format); custom names are saved with allow_custom
    negative_label: str
    positive_prompt: str
    negative_prompt: str
    linked: tuple[str, ...] = ()  # tunables moved by the same amount (pinch_off keeps its hysteresis gap)
    needs_helper: bool = False


AUTOTUNE_SPECS: dict[str, AutoTuneSpec] = {
    "pinch": AutoTuneSpec(
        "pinch",
        "Pinch click",
        "hands.gestures.pinch_on",
        "pinch",
        "below",
        "pinch_click",
        "move",
        "Pinch thumb and index together, then open again",
        "Point and move around WITHOUT pinching (bring the fingers near each other, but do not touch)",
        linked=("hands.gestures.pinch_off",),
    ),
    "thumb_fold": AutoTuneSpec(
        "thumb_fold",
        "Auto-scroll (thumb tucked)",
        "hands.gestures.autoscroll_thumb_fold",
        "thumb_fold",
        "below",
        "four_finger_scroll",
        "palm_clutch",
        "Four fingers up with the thumb tucked in, as for auto-scroll",
        "Open palm with the thumb OUT (the pause pose)",
    ),
    "palm_depth": AutoTuneSpec(
        "palm_depth",
        "Palm scroll (fingertip on the palm)",
        "hands.palm_scroll.depth_threshold",
        "palm_depth",
        "below",
        "palm_scroll",
        "move",
        "Hold one hand flat and touch its palm with the other hand's index fingertip",
        "Hold the hand flat and HOVER the fingertip a little above the palm, not touching",
        needs_helper=True,
    ),
}


def new_session_id(key: str, now: float | None = None) -> str:
    """``autotune-pinch-20261013-142501-3fa2``: unique per run, so the trainer keeps a run on one side of its split."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() if now is None else now))
    return f"autotune-{key}-{stamp}-{secrets.token_hex(2)}"


Frame = tuple[Any, Any]  # (pointer landmarks or None, helper landmarks or None)


@dataclass
class AutoTuneResult:
    spec: AutoTuneSpec
    choice: ThresholdChoice
    before: dict[str, float]
    after: dict[str, float]
    pos_stats: list[float]
    neg_stats: list[float]

    def summary(self) -> str:
        c = self.choice
        old, new = self.before[self.spec.path], self.after[self.spec.path]
        quality = (
            f"separates all {c.n_pos} + {c.n_neg} trials (margin {c.margin:.2f})"
            if c.separable
            else f"best split gets {c.balanced_accuracy:.0%} right; the two sets overlap, so try again more clearly"
        )
        clamp = " (limited by the slider range)" if c.clamped else ""
        return f"{self.spec.title}: {old:.2f} -> {new:.2f}{clamp}; {quality}"


@dataclass
class AutoTuneSession:
    spec: AutoTuneSpec
    count: int = 5
    session_id: str = ""
    positives: list[list[Frame]] = field(default_factory=list)
    negatives: list[list[Frame]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.count < MIN_TRIALS:
            raise ValueError(f"count must be at least {MIN_TRIALS}")
        if not self.session_id:
            self.session_id = new_session_id(self.spec.key)

    def next_step(self) -> tuple[str, int] | None:
        """``("positive", i)`` / ``("negative", i)`` (1-based) for the next trial to record, None when done."""
        if len(self.positives) < self.count:
            return "positive", len(self.positives) + 1
        if len(self.negatives) < self.count:
            return "negative", len(self.negatives) + 1
        return None

    def prompt(self) -> str:
        step = self.next_step()
        if step is None:
            return "Done: review the suggestion and apply it"
        kind, i = step
        text = self.spec.positive_prompt if kind == "positive" else self.spec.negative_prompt
        return f"{text} ({i}/{self.count})"

    def add_trial(self, frames: Sequence[Frame]) -> str | None:
        """Store the next trial. Returns why it was rejected (too few measured frames; record it again)."""
        step = self.next_step()
        if step is None:
            return "all trials are recorded"
        stat = self.trial_statistic(frames)
        if stat is None:
            need = "both hands" if self.spec.needs_helper else "your hand"
            return f"could not measure that one (keep {need} in view); again please"
        (self.positives if step[0] == "positive" else self.negatives).append(list(frames))
        return None

    def trial_statistic(self, frames: Sequence[Frame]) -> float | None:
        return clip_statistic([measure(self.spec.measure, lm, h) for lm, h in frames], self.spec.fires)

    def result(self, current: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]) -> AutoTuneResult:
        """The suggested change: the separating threshold for the spec's tunable, linked tunables shifted by the
        same amount (and clamped)."""
        if self.next_step() is not None:
            raise ValueError("record all trials first")
        ps = [s for s in (self.trial_statistic(f) for f in self.positives) if s is not None]
        ns = [s for s in (self.trial_statistic(f) for f in self.negatives) if s is not None]
        lo, hi = bounds.get(self.spec.path, (-math.inf, math.inf))
        choice = choose_threshold(ps, ns, self.spec.fires, lo, hi)
        before = {p: float(current[p]) for p in (self.spec.path, *self.spec.linked) if p in current}
        after = {self.spec.path: choice.threshold}
        delta = choice.threshold - before.get(self.spec.path, choice.threshold)
        for p in self.spec.linked:
            if p in before:
                llo, lhi = bounds.get(p, (-math.inf, math.inf))
                after[p] = max(llo, min(lhi, before[p] + delta))
        return AutoTuneResult(self.spec, choice, before, after, ps, ns)

    def save_clips(self, folder: Path, handedness: str = "Right", fps: float = 30.0) -> list[Path]:
        """Every trial as a labelled ``away record`` clip (pointer-hand frames) with this run's ``session_id`` and
        the per-frame measurement. Trials with fewer than 5 hand frames are skipped."""
        from away.hands.recorder import save_clip

        saved: list[Path] = []
        for label, trials, kind in (
            (self.spec.positive_label, self.positives, "positive"),
            (self.spec.negative_label, self.negatives, "negative"),
        ):
            for trial in trials:
                lms = [as_array(lm) for lm, _h in trial if lm is not None]
                if len(lms) < 5:
                    continue
                vals = np.array(
                    [
                        np.nan if (v := measure(self.spec.measure, lm, h)) is None else v
                        for lm, h in trial
                        if lm is not None
                    ]
                )
                saved.append(
                    save_clip(
                        folder,
                        label,
                        lms,
                        handedness,
                        fps,
                        allow_custom=True,
                        session_id=self.session_id,
                        extra={
                            "source": "autotune",
                            "autotune": self.spec.key,
                            "trial": kind,
                            "measure": self.spec.measure,
                            "values": vals,
                        },
                    )
                )
        return saved
