"""Real-time gesture state machine.

Turns a stream of (timestamp, landmarks) into discrete, debounced events:

======================  ====================================================
event                   trigger
======================  ====================================================
``move``                index finger pointing (cursor = index tip)
``click``               thumb-index pinch closed and released quickly
``double_click``        second quick pinch within ``double_pinch_window``
                        (the backend issues ONE extra click; the OS sees two)
``drag_start/drag_end`` pinch held > ``drag_hold`` or moved > ``drag_move``
``scroll``              index+middle extended ("two"), vertical hand motion
``clutch``              open palm held still for ``clutch_hold`` -> toggles
                        pause; while paused only the clutch is honoured
``swipe_left/right``    open palm moving fast horizontally
``hand_lost``           hand left the frame (ends any drag)
======================  ====================================================

Low-latency continuous control (move/drag/scroll) is rule-based; the learned
temporal classifier can veto or confirm discrete gestures through
:mod:`away.fusion`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from away.config import GestureConfig
from away.hands.landmarks import INDEX_TIP, THUMB_TIP, WRIST, as_array, classify_pose, pinch_ratio


@dataclass(frozen=True)
class GestureEvent:
    kind: str
    t: float
    data: dict[str, Any] = field(default_factory=dict)


SWIPE_COOLDOWN = 0.5
CLUTCH_MAX_DRIFT = 0.05
SCROLL_MIN_DELTA = 0.002


class GestureStateMachine:
    def __init__(self, cfg: GestureConfig | None = None) -> None:
        self.cfg = cfg or GestureConfig()
        # Detection vs mapping: the clutch pose is always *detected*; whether it pauses the machine is decided by
        # the bindings (open_palm -> pause:toggle sets this True; any other binding sets it False).
        self.clutch_pauses = True
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.paused = False
        self.pinched = False
        self.dragging = False
        self.hand_present = False
        self._pinch_t0 = 0.0
        self._pinch_pos = (0.0, 0.0)
        self._last_tap_t: float | None = None
        self._palm_t0: float | None = None
        self._palm_pos = (0.0, 0.0)
        self._clutch_latched = False
        self._wrist_hist: deque[tuple[float, float, float]] = deque()
        self._last_swipe_t = -1e9
        self._scroll_prev_y: float | None = None

    # ---------------------------------------------------------------- helpers
    def _update_pinch(self, ratio: float) -> bool:
        if self.pinched:
            return ratio < self.cfg.pinch_off
        return ratio < self.cfg.pinch_on

    def _check_swipe(self, t: float, x: float, y: float) -> GestureEvent | None:
        self._wrist_hist.append((t, x, y))
        while self._wrist_hist and t - self._wrist_hist[0][0] > self.cfg.swipe_max_time:
            self._wrist_hist.popleft()
        if t - self._last_swipe_t < SWIPE_COOLDOWN or len(self._wrist_hist) < 2:
            return None
        t0, x0, y0 = self._wrist_hist[0]
        dx, dy = x - x0, y - y0
        if abs(dx) >= self.cfg.swipe_min_dx and abs(dx) > 2 * abs(dy):
            self._last_swipe_t = t
            self._wrist_hist.clear()
            self._palm_t0 = None
            kind = "swipe_right" if dx > 0 else "swipe_left"
            return GestureEvent(kind, t, {"dx": dx, "duration": t - t0})
        return None

    # ------------------------------------------------------------------- step
    def update(self, t: float, landmarks: object | None, pinch: float | None = None) -> list[GestureEvent]:
        """Feed one frame (``None`` = no hand). Returns the events it produced.

        ``pinch`` overrides the 2D thumb-index ratio (multi-camera fusion passes a 3D one).
        """
        events: list[GestureEvent] = []
        if landmarks is None:
            if self.dragging:
                events.append(GestureEvent("drag_end", t, {"reason": "hand_lost"}))
            if self.hand_present:
                events.append(GestureEvent("hand_lost", t))
            paused = self.paused
            last_tap = self._last_tap_t
            self.reset()
            self.paused = paused  # pause survives the hand leaving
            self._last_tap_t = last_tap
            return events

        lm = as_array(landmarks)
        self.hand_present = True
        wx, wy = float(lm[WRIST, 0]), float(lm[WRIST, 1])
        pose = classify_pose(lm)
        ratio = pinch_ratio(lm) if pinch is None else float(pinch)

        # --- clutch (open palm held still) works even while paused
        if pose == "palm" and not self.pinched:
            if self._palm_t0 is None:
                self._palm_t0, self._palm_pos = t, (wx, wy)
            drift = float(np.hypot(wx - self._palm_pos[0], wy - self._palm_pos[1]))
            if drift > CLUTCH_MAX_DRIFT:
                self._palm_t0, self._palm_pos = t, (wx, wy)
            elif not self._clutch_latched and t - self._palm_t0 >= self.cfg.clutch_hold:
                if self.clutch_pauses:
                    self.paused = not self.paused
                self._clutch_latched = True
                events.append(GestureEvent("clutch", t, {"paused": self.paused}))
        else:
            self._palm_t0 = None
            self._clutch_latched = False

        if self.paused:
            self._wrist_hist.clear()
            return events

        # --- swipes (open palm, fast horizontal)
        if pose == "palm" and not self.pinched:
            sw = self._check_swipe(t, wx, wy)
            if sw is not None:
                return [*events, sw]
        else:
            self._wrist_hist.clear()

        # --- pinch: click / double click / drag
        was_pinched = self.pinched
        self.pinched = self._update_pinch(ratio)
        mid = (
            float((lm[THUMB_TIP, 0] + lm[INDEX_TIP, 0]) / 2),
            float((lm[THUMB_TIP, 1] + lm[INDEX_TIP, 1]) / 2),
        )
        if self.pinched and not was_pinched:
            self._pinch_t0, self._pinch_pos = t, mid
        elif self.pinched:
            moved = float(np.hypot(mid[0] - self._pinch_pos[0], mid[1] - self._pinch_pos[1]))
            if not self.dragging and (t - self._pinch_t0 >= self.cfg.drag_hold or moved >= self.cfg.drag_move):
                self.dragging = True
                events.append(GestureEvent("drag_start", t, {"x": mid[0], "y": mid[1]}))
            if self.dragging:
                events.append(GestureEvent("move", t, {"x": mid[0], "y": mid[1], "anchor": "pinch"}))
        elif was_pinched:  # released
            if self.dragging:
                self.dragging = False
                events.append(GestureEvent("drag_end", t, {"x": mid[0], "y": mid[1]}))
            elif self._last_tap_t is not None and t - self._last_tap_t <= self.cfg.double_pinch_window:
                events.append(GestureEvent("double_click", t))
                self._last_tap_t = None
            else:
                events.append(GestureEvent("click", t))
                self._last_tap_t = t
        if self.pinched:
            self._scroll_prev_y = None
            return events

        # --- scroll with two fingers
        if pose == "two":
            y = float(lm[INDEX_TIP, 1])
            if self._scroll_prev_y is not None:
                dy = self._scroll_prev_y - y  # hand up -> positive (scroll up)
                if abs(dy) >= SCROLL_MIN_DELTA:
                    events.append(GestureEvent("scroll", t, {"dy": dy * self.cfg.scroll_gain}))
                    self._scroll_prev_y = y
            else:
                self._scroll_prev_y = y
            return events
        self._scroll_prev_y = None

        # --- pointer
        if pose == "point":
            events.append(
                GestureEvent("move", t, {"x": float(lm[INDEX_TIP, 0]), "y": float(lm[INDEX_TIP, 1]), "anchor": "index"})
            )
        return events
