"""The one object :class:`away.app.runner.Pipeline` talks to for the attention gate, arm fatigue and breaks.

The pipeline calls three methods, each a single line in ``runner.py``:

* :meth:`ComfortHub.on_face` after head tracking ran on a face frame (facing + gaze for the gate),
* :meth:`ComfortHub.allow_hand` at the top of every hand frame: ``False`` = treat the hand as absent this frame
  (the gesture state machine then ends any drag cleanly, and the head pointer may take over),
* :meth:`ComfortHub.apply_config` on a config hot-reload.

Arm-fatigue "switch" mode turns ``hands.cursor.head_assist`` to ``pointer`` on the live config only (never
written to the file) and puts the user's own value back when the hand returns after a rest.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from away.comfort import ui
from away.comfort.attention import AttentionGate, hand_height
from away.comfort.fatigue import BreakReminder, FatigueMonitor


def off_screen_deg(pose: Any, baseline: Any, box: Any) -> float | None:
    """Degrees the head is turned away from the screens: outside the calibrated pose box (yaw/pitch min/max seen
    while looking at the calibration dots), or off the learned usual pose. None = not known yet."""
    if pose is None:
        return None
    if box is not None:
        y0, y1, p0, p1 = box
        dy = max(0.0, y0 - pose.yaw, pose.yaw - y1)
        dp = max(0.0, p0 - pose.pitch, pose.pitch - p1)
        return float(max(dy, dp))
    if baseline is None or not getattr(baseline, "ready", False) or baseline.yaw is None:
        return None
    return float(max(abs(pose.yaw - baseline.yaw), abs(pose.pitch - baseline.pitch)))


class ComfortHub:
    def __init__(
        self,
        cfg: Any,
        headgaze: Any = None,
        publish: Callable[[str, dict[str, Any]], Any] | None = None,
        notify: Callable[[str, str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.headgaze = headgaze
        self.publish = publish or (lambda _topic, _data: None)
        # notify(title, message): a toast. By default "app.notice" is published, which the tray shows.
        self.notify = notify or (lambda title, msg: self.publish("app.notice", {"title": title, "message": msg}))
        self.clock = clock
        self.gate = AttentionGate(cfg.attention, face_available=bool(cfg.face.enabled))
        self.fatigue = FatigueMonitor(cfg.comfort)
        self.breaks = BreakReminder(cfg.comfort)
        self.base_head_assist = str(cfg.hands.cursor.head_assist)
        self._last_state = self.gate.state
        self._last_tip: float | None = None

    # ------------------------------------------------------------------ config
    def apply_config(self, new: Any) -> None:
        """A reloaded config: take over the attention/comfort values and remember the user's head_assist."""
        for src, dst in ((new.attention, self.cfg.attention), (new.comfort, self.cfg.comfort)):
            if src is dst:
                continue
            for k, v in vars(src).items():
                setattr(dst, k, v)
        self.base_head_assist = str(new.hands.cursor.head_assist)
        self.gate.face_available = bool(new.face.enabled)
        if self.fatigue.switched:
            self.cfg.hands.cursor.head_assist = "pointer"  # a reload just put the file value back

    # ------------------------------------------------------------------ face
    def on_face(self, t: float, face_seen: bool) -> None:
        """After :meth:`HeadGaze.on_face` for the user's face (``face_seen``) or a face frame without them."""
        if not face_seen:
            self.gate.on_no_face(t)
            return
        hg = self.headgaze
        cal = getattr(hg, "cal", None)
        box = getattr(cal, "pose_box", None) if cal is not None else None
        off = off_screen_deg(getattr(hg, "pose", None), getattr(hg, "baseline", None), box)
        self.gate.on_face(t, None if off is None else off <= float(self.cfg.attention.facing_tolerance), off)
        est = getattr(hg, "estimator", None)
        if est is not None:
            pt = est.fresh(t)
            mons = getattr(getattr(getattr(hg, "mouse", None), "mapper", None), "monitors", None) or []
            on = None if pt is None else any(m.contains(pt[0], pt[1]) for m in mons)
            self.gate.on_gaze(t, on)

    # ------------------------------------------------------------------ hands
    def allow_hand(self, t: float, landmarks: Any) -> bool:
        """Every hand frame. False = the hand does not count this frame (not engaged, or lowered while the head
        drives the pointer)."""
        engaged = self.gate.update(t, landmarks)
        if self.gate.state != self._last_state:
            self._last_state = self.gate.state
            self.publish("attention.state", self.gate.snapshot())
        for ev in self.fatigue.update(t, landmarks):
            self._on_fatigue(t, ev)
        if self.fatigue.switched:
            self.cfg.hands.cursor.head_assist = "pointer"
            h = hand_height(landmarks)
            if h is None or h < float(self.cfg.comfort.raised_height):
                engaged = False  # a lowered hand still in view must not block the head pointer
        if self.breaks.update(t, landmarks is not None):
            mins = round(self.breaks.minutes_used(t))
            self.notify(ui.BREAK_TITLE, ui.break_message(mins))
            self.publish("comfort.break", {"minutes": mins})
        return engaged

    def _on_fatigue(self, t: float, ev: str) -> None:
        c = self.cfg.comfort
        self.publish(f"comfort.{ev}", self.fatigue.snapshot())
        if ev == "switch_on":
            if not self.cfg.face.enabled:  # head control needs face tracking: only the tip is shown
                self.fatigue.switched = False
                return
            self.cfg.hands.cursor.head_assist = "pointer"
        elif ev == "switch_back":
            self.cfg.hands.cursor.head_assist = self.base_head_assist
            self.notify(ui.FATIGUE_TITLE, ui.SWITCH_BACK_MESSAGE)
        elif ev == "tired" and (c.fatigue_tips or c.fatigue_switch):
            cool = float(c.tip_cooldown_min) * 60.0
            if self._last_tip is not None and t - self._last_tip < cool and not c.fatigue_switch:
                return
            self._last_tip = t
            self.notify(ui.FATIGUE_TITLE, ui.fatigue_message(bool(c.fatigue_switch), bool(self.cfg.face.enabled)))

    # ------------------------------------------------------------------ voice + UI
    def voice_allowed(self, t: float | None = None) -> bool:
        return self.gate.voice_allowed(self.clock() if t is None else t)

    def attention_state(self) -> dict[str, Any]:
        snap = self.gate.snapshot()
        snap["fatigue"] = self.fatigue.snapshot()
        return snap
