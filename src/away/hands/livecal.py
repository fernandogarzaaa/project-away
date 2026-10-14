"""Live tracking calibration (v0.1.25): ``away calibrate``, tray > Calibrate tracking, Settings > Getting started.

Three short steps, in plain words:

1. **Hold still** (about 3 s): measures how much the hand points shake on *this* camera when you do not move (the
   "noise"), and sets the smoothing, the dead zone and the hold-still lock from it.
2. **Corners**: you point at the four corners you can comfortably reach; that box becomes the active area and sets
   the cursor speed so the box covers the screen.
3. **Pinch** three times: your own closed and open pinch set when a pinch counts.

Everything here is pure logic (time and landmarks are passed in) so it can be tested with fake landmark streams;
:mod:`away.app.calibrate_gui` is the window on top. The result is saved per camera in ``hands.calibration`` and laid
over the cursor / gesture settings whenever that camera is in use (:func:`calibrated`).
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from away.hands.steady import jitter_px

INDEX_TIP = 8
STILL_S = 3.0  # step 1 length (seconds of hand-in-view samples)
STILL_MAX_NOISE = 0.02  # hand units: more than this is not "still" (the step starts again)
CORNERS_MIN_S = 3.0
CORNERS_MAX_S = 15.0
PINCHES = 3
FRAME_PX = 640.0  # the noise is shown in pixels of a 640-pixel-wide camera picture (the usual webcam mode)

# the keys a calibration may set (anything else in a saved entry is ignored)
CURSOR_KEYS = ("min_cutoff", "fast_cutoff", "fast_speed", "dead_zone", "hold_radius", "gain", "active_region")
GESTURE_KEYS = ("pinch_on", "pinch_off")
STEPS = ("still", "corners", "pinch", "done")
INSTRUCTIONS = {
    "still": "Hold your pointing hand still, as if resting on a mouse.",
    "corners": "Now point at the four corners you can reach comfortably, one after the other.",
    "pinch": "Pinch your thumb and index finger together, then open. Do it 3 times.",
    "done": "All done! Look at the cursor paths, then press Save.",
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


def camera_key(cfg: Any) -> str:
    """The camera a calibration belongs to: the primary camera's name / index / path, as text."""
    h = cfg.hands
    dev = h.cameras[0].device if getattr(h, "cameras", None) else h.camera_index
    return str(dev)


def measure_noise(points: Sequence[tuple[float, float]]) -> float:
    """How much a held-still point shakes (hand units): the RMS distance from a straight-line drift fitted over the
    samples, so a slow sag of the arm is not counted as noise."""
    n = len(points)
    if n < 3:
        return 0.0
    mt = (n - 1) / 2.0
    var_t = sum((t - mt) ** 2 for t in range(n)) or 1.0
    out = 0.0
    for axis in (0, 1):
        vals = [p[axis] for p in points]
        mv = sum(vals) / n
        slope = sum((t - mt) * (v - mv) for t, v in enumerate(vals)) / var_t
        out += sum((v - (mv + slope * (t - mt))) ** 2 for t, v in enumerate(vals)) / n
    return math.sqrt(out)


def _percentile(vals: Sequence[float], q: float) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    i = _clamp(q, 0.0, 1.0) * (len(s) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def reach_box(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    """The comfortable reach (x0, y0, x1, y1) from the corner step: 3rd-97th percentile (a stray frame is ignored)."""
    if len(points) < 5:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (_percentile(xs, 0.03), _percentile(ys, 0.03), _percentile(xs, 0.97), _percentile(ys, 0.97))


def corners_visited(points: Sequence[tuple[float, float]], frac: float = 0.25) -> int:
    """How many of the four corners of the reach box were pointed at (within ``frac`` of the box size)."""
    box = reach_box(points)
    if box is None:
        return 0
    x0, y0, x1, y1 = box
    w, h = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    return sum(
        1
        for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        if any(abs(px - cx) <= frac * w and abs(py - cy) <= frac * h for px, py in points)
    )


@dataclass
class Measurements:
    noise: float = 0.0  # hand units (RMS)
    fps: float = 30.0
    reach: tuple[float, float, float, float] | None = None
    pinch_closed: float | None = None  # pinch ratio when closed (typical)
    pinch_open: float | None = None  # pinch ratio when open (typical)

    @property
    def noise_px(self) -> float:
        return self.noise * FRAME_PX


def derive(m: Measurements) -> dict[str, Any]:
    """Settings from the measurements: ``{"cursor": {...}, "gestures": {...}}`` (only what was measured).

    * Noisier camera -> lower still cutoff (steadier), more dead-zone slack and a bigger hold-still circle.
    * Slower camera -> lower fast cutoff (no point following faster than the frames come in).
    * Smaller reach -> faster cursor, so the comfortable box still covers the screen; the box is the active area.
    * The pinch counts at 40% of the way from your closed to your open pinch and lets go at 70%.
    """
    cur: dict[str, Any] = {}
    ges: dict[str, Any] = {}
    if m.noise > 0:
        s = m.noise
        cur["min_cutoff"] = round(_clamp(math.sqrt(0.0015 / s), 0.4, 2.0), 3)
        cur["dead_zone"] = round(_clamp(0.9 * s, 0.0005, 0.006), 5)
        cur["hold_radius"] = round(_clamp(5.0 * s, 0.003, 0.03), 5)
    if m.fps > 0:
        cur["fast_cutoff"] = round(_clamp(m.fps / 4.0, 3.0, 10.0), 2)
    if m.reach is not None:
        x0, y0, x1, y1 = m.reach
        span = max(0.05, x1 - x0)
        margin = 0.03
        cur["active_region"] = [
            round(_clamp(x0 - margin, 0.0, 0.9), 3),
            round(_clamp(y0 - margin, 0.0, 0.9), 3),
            round(_clamp(x1 + margin, 0.1, 1.0), 3),
            round(_clamp(y1 + margin, 0.1, 1.0), 3),
        ]
        cur["gain"] = round(_clamp(1.54 / span, 1.0, 5.0), 2)
        cur["fast_speed"] = round(_clamp(0.8 * span, 0.2, 1.5), 3)
    if m.pinch_closed is not None and m.pinch_open is not None and m.pinch_open > m.pinch_closed + 0.05:
        c, o = m.pinch_closed, m.pinch_open
        on = _clamp(c + 0.4 * (o - c), 0.08, 0.9)
        off = _clamp(c + 0.7 * (o - c), on + 0.05, 1.4)
        ges["pinch_on"], ges["pinch_off"] = round(on, 3), round(off, 3)
    out: dict[str, Any] = {}
    if cur:
        out["cursor"] = cur
    if ges:
        out["gestures"] = ges
    return out


def apply_entry(cfg: Any, entry: dict[str, Any] | None) -> Any:
    """Lay a saved calibration over ``cfg`` (in place; returns it). Unknown keys are ignored."""
    if not entry:
        return cfg
    for sec, keys, obj in (
        ("cursor", CURSOR_KEYS, cfg.hands.cursor),
        ("gestures", GESTURE_KEYS, cfg.hands.gestures),
    ):
        vals = entry.get(sec) or {}
        if not isinstance(vals, dict):
            continue
        for k in keys:
            if k in vals:
                v = vals[k]
                setattr(obj, k, tuple(v) if k == "active_region" else v)
    return cfg


def calibrated(cfg: Any) -> Any:
    """A copy of ``cfg`` with the current camera's calibration laid over it (``cfg`` itself when there is none)."""
    entry = (getattr(cfg.hands, "calibration", None) or {}).get(camera_key(cfg))
    if not entry:
        return cfg
    out = copy.deepcopy(cfg)
    apply_entry(out, entry)
    try:
        from away.config import validate

        validate(out)
    except ValueError:
        return cfg  # a hand-edited, broken entry never stops the app: it is just not used
    return out


def entry_for(m: Measurements, now: float | None = None) -> dict[str, Any]:
    e = derive(m)
    e["noise_px"] = round(m.noise_px, 2)
    e["fps"] = round(m.fps, 1)
    e["saved"] = time.time() if now is None else now
    return e


def save(path: Path | None, key: str, entry: dict[str, Any] | None) -> Path:
    """Write (or with ``entry=None``, remove: "Reset to defaults") this camera's calibration in the config file."""
    from away import config as cfgmod

    cfg, _ = cfgmod.load(path)
    cal = dict(cfg.hands.calibration or {})
    if entry is None:
        cal.pop(key, None)
    else:
        cal[key] = entry
    cfg.hands.calibration = cal
    cfgmod.validate(cfg)
    return cfgmod.save(cfg, path)


def simulate(
    cursor_cfg: Any, samples: Sequence[tuple[float, float, float]], screen: tuple[int, int] = (1920, 1080)
) -> list[tuple[int, int]]:
    """Run the real cursor mapper over (t, x, y) hand samples; the cursor positions it produces (for the before /
    after trails and the tests)."""
    from away.hands.cursor import CursorMapper

    m = CursorMapper(screen, copy.deepcopy(cursor_cfg))
    m.set_position(screen[0] / 2, screen[1] / 2)
    return [m.update(t, x, y) for t, x, y in samples]


@dataclass
class Session:
    """The three steps, fed one frame at a time: :meth:`feed` with the time, the 21 hand points (x, y in 0..1, or
    None when no hand is seen) and the pinch ratio."""

    pinch_on: float = 0.32  # the current setting (to count pinches before we know the user's own)
    step: str = "still"
    still: list[tuple[float, float]] = field(default_factory=list)
    still_t: list[float] = field(default_factory=list)
    trail: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y) of steps 1-2, for the trails
    corners: list[tuple[float, float]] = field(default_factory=list)
    corners_t0: float | None = None
    ratios: list[float] = field(default_factory=list)
    pinch_mins: list[float] = field(default_factory=list)
    frame_t: list[float] = field(default_factory=list)
    message: str = ""
    _closed: bool = False
    _cur_min: float = 9.0

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[self.step]

    def fps(self) -> float:
        ts = self.frame_t[-31:]
        if len(ts) < 2 or ts[-1] <= ts[0]:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def live_noise_px(self) -> float:
        """The shake meter: noise of the last second of index-fingertip positions (camera px)."""
        return measure_noise([(x, y) for _t, x, y in self.trail[-30:]]) * FRAME_PX

    def progress(self) -> float:
        """0..1 within the current step."""
        if self.step == "still":
            if len(self.still_t) < 2:
                return 0.0
            return _clamp((self.still_t[-1] - self.still_t[0]) / STILL_S, 0.0, 1.0)
        if self.step == "corners":
            return corners_visited(self.corners) / 4.0
        if self.step == "pinch":
            return len(self.pinch_mins) / PINCHES
        return 1.0

    def feed(self, t: float, hand: Sequence[Sequence[float]] | None, ratio: float | None = None) -> str:
        """One frame. Returns the step it is in afterwards."""
        self.frame_t.append(t)
        self.frame_t = self.frame_t[-60:]
        if hand is None or len(hand) <= INDEX_TIP:
            self.message = "I can't see your hand. Hold it up in front of the camera."
            return self.step
        self.message = ""
        x, y = float(hand[INDEX_TIP][0]), float(hand[INDEX_TIP][1])
        if self.step in ("still", "corners"):
            self.trail.append((t, x, y))
        if self.step == "still":
            self.still.append((x, y))
            self.still_t.append(t)
            if len(self.still) >= 10 and measure_noise(self.still) > STILL_MAX_NOISE:
                self.still, self.still_t = [], []
                self.message = "That was a bit much movement. Rest your hand and keep it still."
            elif self.still_t[-1] - self.still_t[0] >= STILL_S:
                self.step = "corners"
                self.corners_t0 = t
        elif self.step == "corners":
            self.corners.append((x, y))
            t0 = self.corners_t0 if self.corners_t0 is not None else t
            box = reach_box(self.corners)
            big = box is not None and box[2] - box[0] >= 0.15 and box[3] - box[1] >= 0.1
            if (t - t0 >= CORNERS_MIN_S and big and corners_visited(self.corners) == 4) or t - t0 >= CORNERS_MAX_S:
                self.step = "pinch"
        elif self.step == "pinch" and ratio is not None and math.isfinite(ratio):
            self.ratios.append(float(ratio))
            if ratio < self.pinch_on * 1.15:
                self._closed = True
                self._cur_min = min(self._cur_min, ratio)
            elif self._closed and ratio > self.pinch_on * 1.4:
                self.pinch_mins.append(self._cur_min)
                self._closed, self._cur_min = False, 9.0
                if len(self.pinch_mins) >= PINCHES:
                    self.step = "done"
        return self.step

    def measurements(self) -> Measurements:
        m = Measurements(noise=measure_noise(self.still), fps=self.fps() or 30.0)
        box = reach_box(self.corners)
        if box is not None and box[2] - box[0] >= 0.15 and box[3] - box[1] >= 0.1:
            m.reach = box
        if self.pinch_mins:
            m.pinch_closed = _percentile(self.pinch_mins, 0.5)
            opened = [r for r in self.ratios if r > self.pinch_on * 1.4]
            if opened:
                m.pinch_open = _percentile(opened, 0.5)
        return m


def before_after(
    cfg: Any, entry: dict[str, Any], samples: Sequence[tuple[float, float, float]], screen: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], float, float]:
    """Cursor trails for the same hand samples with the settings before and after calibration, plus the jitter (px)
    of each over the hold-still part (the first :data:`STILL_S` seconds)."""
    before = simulate(cfg.hands.cursor, samples, screen)
    after_cfg = apply_entry(copy.deepcopy(cfg), entry)
    after = simulate(after_cfg.hands.cursor, samples, screen)
    jb = ja = 0.0
    if samples:
        t_end = samples[0][0] + STILL_S
        n = sum(1 for s in samples if s[0] <= t_end)
        skip = min(n // 3, 15)  # the first frames settle the filter
        jb = jitter_px([(float(a), float(b)) for a, b in before[skip:n]])
        ja = jitter_px([(float(a), float(b)) for a, b in after[skip:n]])
    return before, after, jb, ja


__all__ = [
    "INSTRUCTIONS",
    "Measurements",
    "Session",
    "apply_entry",
    "before_after",
    "calibrated",
    "camera_key",
    "corners_visited",
    "derive",
    "entry_for",
    "measure_noise",
    "reach_box",
    "save",
    "simulate",
]
