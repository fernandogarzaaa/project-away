"""Arm-fatigue detection and break reminders (pure logic, fed with timestamps).

:class:`FatigueMonitor` watches the pointer hand and combines three signs into one score:

* **time with the arm raised**: seconds the wrist has been above ``raised_height`` without a rest
  (dips shorter than ``rest_s`` pause the clock, they do not reset it);
* **sinking hand**: the hand's height now (5 s average) versus where it was during the first 10 s with the arm up;
* **rising tremor**: the high-frequency shake of the hand *while it is held still*, versus the shake measured
  during the first 10 s (webcam landmarks always shake a little, so only the ratio counts).

``score = raised_s / raised_limit_s + 0.5 * sinking + 0.5 * tremor`` with ``sinking`` and ``tremor`` scaled to
0..1 (full at ``height_drop`` and ``jitter_ratio``). The arm counts as tired when the score reaches 1 and it has
been up at least ``min_raised_s``: with no other sign after ``raised_limit_s`` (4 min), with one strong sign after
half of that, with both strong signs after ``min_raised_s``. A rest (hand down or out of view for ``rest_s``)
clears it. These thresholds are an informed starting point, **not** a validated model of muscle fatigue.

:class:`BreakReminder` counts continuous use (hand in view, gaps shorter than ``break_reset_s``) and asks for a
break every ``break_every_min`` minutes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from away.comfort.attention import hand_height, hand_size
from away.hands.landmarks import MIDDLE_MCP, WRIST

BASELINE_S = 10.0  # the first seconds of a raised-arm session give the reference height and tremor
HEIGHT_TAU_S = 5.0  # the "height now" average
JITTER_TAU_S = 3.0  # tremor energy average
SMOOTH_TAU_S = 0.12  # the low-pass the tremor is measured against (residual = position - this)
STILL_SPEED = 1.5  # palm lengths per second: slower than this counts as holding still (tremor is measured)
JITTER_FLOOR = 0.004  # palm lengths: the reference tremor is never taken as smaller than this
MAX_DT = 0.5  # gaps longer than this restart the filters


@dataclass
class FatigueReading:
    raised_s: float = 0.0
    drop: float = 0.0  # picture heights below the reference
    jitter_ratio: float = 1.0
    score: float = 0.0
    tired: bool = False


class FatigueMonitor:
    """Feed :meth:`update` every hand frame; it returns events: ``tired``, ``rested`` and, with
    ``comfort.fatigue_switch``, ``switch_on`` / ``switch_back``."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg  # a ComfortConfig, read live
        self.switched = False  # head control is on because of fatigue
        self.resting = False  # tired, then rested; waiting for the hand to come back
        self._reset_session()
        self._low_since: float | None = None
        self._back_since: float | None = None
        self._t: float | None = None

    def _reset_session(self) -> None:
        self.reading = FatigueReading()
        self._base_h: list[float] = []
        self._base_j: list[float] = []
        self._h_avg: float | None = None
        self._smooth: Any = None
        self._prev: Any = None
        self._j_energy: float | None = None

    @property
    def tired(self) -> bool:
        return self.reading.tired

    def _signs(self, dt: float, lm: Any, height: float) -> None:
        """Update the height average and the tremor estimate from one raised-arm frame."""
        r = self.reading
        a_h = 1.0 - math.exp(-dt / HEIGHT_TAU_S) if dt > 0 else 1.0
        self._h_avg = height if self._h_avg is None else self._h_avg + a_h * (height - self._h_avg)
        arr = np.asarray(lm, dtype=float)
        palm = max(1e-6, float(hand_size(arr) or 0.0))
        pos = (arr[WRIST, :2] + arr[MIDDLE_MCP, :2]) / 2.0 / palm
        if self._smooth is None or self._prev is None or dt <= 0:
            self._smooth, self._prev = pos.copy(), pos.copy()
        else:
            speed = float(np.linalg.norm(pos - self._prev)) / dt
            self._prev = pos.copy()
            a_s = 1.0 - math.exp(-dt / SMOOTH_TAU_S)
            self._smooth = self._smooth + a_s * (pos - self._smooth)
            if speed < STILL_SPEED:
                res2 = float(np.sum((pos - self._smooth) ** 2))
                a_j = 1.0 - math.exp(-dt / JITTER_TAU_S)
                self._j_energy = res2 if self._j_energy is None else self._j_energy + a_j * (res2 - self._j_energy)
        jitter = math.sqrt(self._j_energy) if self._j_energy is not None else None
        if r.raised_s <= BASELINE_S:
            self._base_h.append(height)
            if jitter is not None:
                self._base_j.append(jitter)
            return
        if self._base_h and self._h_avg is not None:
            r.drop = max(0.0, float(np.mean(self._base_h)) - self._h_avg)
        if self._base_j and jitter is not None:
            r.jitter_ratio = jitter / max(JITTER_FLOOR, float(np.mean(self._base_j)))

    def _score(self) -> None:
        c, r = self.cfg, self.reading
        sink = min(1.0, r.drop / max(1e-6, float(c.height_drop)))
        shake = min(1.0, max(0.0, (r.jitter_ratio - 1.0) / max(1e-6, float(c.jitter_ratio) - 1.0)))
        r.score = r.raised_s / max(1e-6, float(c.raised_limit_s)) + 0.5 * sink + 0.5 * shake

    def update(self, t: float, landmarks: Any) -> list[str]:
        c = self.cfg
        events: list[str] = []
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        if dt > MAX_DT:
            dt = 0.0
        h = hand_height(landmarks)
        r = self.reading
        if h is not None and h >= float(c.raised_height):
            self._low_since = None
            if self.resting:  # the hand came back after a rest
                if self._back_since is None:
                    self._back_since = t
                if t - self._back_since >= float(c.return_s):
                    self.resting = False
                    self._back_since = None
                    if self.switched:
                        self.switched = False
                        events.append("switch_back")
                return events
            r.raised_s += dt
            self._signs(dt, landmarks, h)
            self._score()
            if not r.tired and r.raised_s >= float(c.min_raised_s) and r.score >= 1.0:
                r.tired = True
                events.append("tired")
                if c.fatigue_switch and not self.switched:
                    self.switched = True
                    events.append("switch_on")
            return events
        # hand down or out of view
        self._back_since = None
        if self._low_since is None:
            self._low_since = t
        if t - self._low_since >= float(c.rest_s) and (r.raised_s > 0.0 or r.tired):
            was_tired = r.tired
            self._reset_session()
            if was_tired:
                self.resting = True
                events.append("rested")
        if self.switched and not c.fatigue_switch:  # switched off in Settings meanwhile
            self.switched = False
            events.append("switch_back")
        return events

    def snapshot(self) -> dict[str, Any]:
        r = self.reading
        return {
            "raised_s": round(r.raised_s, 1),
            "drop": round(r.drop, 3),
            "jitter_ratio": round(r.jitter_ratio, 2),
            "score": round(r.score, 2),
            "tired": r.tired,
            "head_control": self.switched,
        }


class BreakReminder:
    """Every ``break_every_min`` minutes of continuous use, one reminder; a real break restarts the clock."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.since: float | None = None  # start of the current stretch of use
        self.last_seen: float | None = None
        self.last_reminder: float | None = None

    def update(self, t: float, hand_present: bool) -> bool:
        c = self.cfg
        gap_was_break = self.last_seen is not None and t - self.last_seen >= float(c.break_reset_s)
        if gap_was_break:
            self.since = self.last_reminder = None
        if not hand_present:
            return False
        if self.since is None:
            self.since = t
        self.last_seen = t
        if not c.break_reminders:
            return False
        every = float(c.break_every_min) * 60.0
        ref = self.last_reminder if self.last_reminder is not None else self.since
        if t - ref >= every:
            self.last_reminder = t
            return True
        return False

    def minutes_used(self, t: float) -> float:
        return 0.0 if self.since is None else (t - self.since) / 60.0
