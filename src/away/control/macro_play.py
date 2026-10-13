"""Replay a shortcut safely.

:class:`MacroPlayer` runs the steps on an injected backend and window API (fakes in the tests). Safety rules:

* **Stop any time**: the ``abort`` event is checked before every step and during every pause; the service sets it
  from the abort hotkey (``macros.abort_hotkey``, default Esc), the voice command "stop", the tray, or real mouse
  movement. Held keys and buttons are always released.
* **Same app only**: a click is refused when its window's app is not open, is covered by another app, or the
  window has shrunk so the click would miss; text and key presses are refused when another app has focus.
  ``allow_other_app`` (per shortcut, or ``macros.allow_other_app``) lifts this.
* **Never into a password box**: text and single-character keys are skipped while a password box has focus.

Clicks are aimed by, in order: the recorded control found again by its UI Automation name, the position relative
to the window, the absolute screen position (no window information on this OS).

Every refusal comes back as a :class:`PlayResult` with a friendly message and a one-click fix.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from away.control.keys import KeyComboError, parse_combo
from away.control.macro_platform import NullWindowApi
from away.control.macro_record import MARKER, SelfInputMarker
from away.control.macros import MOD_BASE, SPEED_FACTOR, Macro, MacrosConfig, Step, Target, app_key, app_label

# fix id -> button text in the UI
FIX_LABELS = {
    "allow_other_app": "Allow any app",
    "try_again": "Try again",
    "record_again": "Record again",
    "edit_steps": "Edit steps",
}


@dataclass
class PlayResult:
    ok: bool
    steps_run: int = 0
    total: int = 0
    stopped: str = ""  # "" | aborted | refused | error | busy
    message: str = ""  # for people
    fix: str = ""  # FIX_LABELS key, "" = none
    step: int = 0  # 1-based step the problem happened at (0 = none)
    skipped_password: int = 0

    @property
    def fix_label(self) -> str:
        return FIX_LABELS.get(self.fix, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "steps_run": self.steps_run,
            "total": self.total,
            "stopped": self.stopped,
            "message": self.message,
            "fix": self.fix,
            "step": self.step,
            "skipped_password": self.skipped_password,
        }


class Refused(Exception):
    def __init__(self, message: str, fix: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix


class Aborted(Exception):
    pass


def _inside(pt: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    return rect[0] <= pt[0] < rect[2] and rect[1] <= pt[1] < rect[3]


def _wait(ev: threading.Event, seconds: float) -> bool:
    return ev.wait(seconds) if seconds > 0 else ev.is_set()


class MacroPlayer:
    def __init__(
        self,
        backend: Any,
        window_api: Any = None,
        cfg: MacrosConfig | None = None,
        marker: SelfInputMarker | None = None,
        sleep: Callable[[threading.Event, float], bool] = _wait,
    ) -> None:
        self.backend = backend
        self.win = window_api or NullWindowApi()
        self.cfg = cfg or MacrosConfig()
        self.marker = marker or MARKER
        self._sleep = sleep  # waits up to N seconds; True when aborted meanwhile
        self.abort = threading.Event()
        self.log: list[str] = []

    # ---------------------------------------------------------------- run
    def run(self, macro: Macro, speed: str | None = None, allow_other_app: bool | None = None) -> PlayResult:
        factor = SPEED_FACTOR.get(speed or macro.speed, 1.0)
        allow = bool(macro.allow_other_app or self.cfg.allow_other_app) if allow_other_app is None else allow_other_app
        res = PlayResult(ok=False, total=len(macro.steps))
        last_app = macro.app
        for i, step in enumerate(macro.steps, 1):
            res.step = i
            pause = step.delay / factor if factor else 0.0
            if i > 1:
                pause = max(pause, float(self.cfg.min_gap_s))
            if self.abort.is_set() or self._sleep(self.abort, pause):
                return self._stopped(res, macro)
            try:
                with self.marker.busy():
                    last_app = self._step(step, allow, last_app, res, factor)
            except Aborted:
                return self._stopped(res, macro)
            except Refused as exc:
                res.stopped, res.message, res.fix = "refused", exc.message, exc.fix
                self.log.append(f"refused step {i} ({step}): {exc.message}")
                return res
            except Exception as exc:  # the OS input layer failed; never leave the user stuck
                res.stopped, res.fix = "error", "edit_steps"
                res.message = f'"{macro.label}" stopped at step {i} because something went wrong ({exc}).'
                self.log.append(f"error step {i} ({step}): {exc}")
                return res
            res.steps_run = i
        res.ok, res.step = True, 0
        if res.skipped_password:
            res.message = "Done. Typing into a password box was skipped, as always."
        return res

    def _stopped(self, res: PlayResult, macro: Macro) -> PlayResult:
        res.stopped = "aborted"
        res.message = f'Stopped "{macro.label}" after {res.steps_run} of {res.total} steps.'
        return res

    # ---------------------------------------------------------------- steps
    def _step(self, s: Step, allow: bool, last_app: str, res: PlayResult, factor: float) -> str:
        b = self.backend
        if s.type == "wait":
            if self._sleep(self.abort, float(s.seconds)):  # waits you added always wait the full time
                raise Aborted
            return last_app
        if s.type in ("text", "key"):
            self._check_focus(s.app or last_app, allow, s)
            if s.types_text and self.win.focused_is_password():
                res.skipped_password += 1
                self.log.append(f"skipped {s}: a password box has focus")
                return last_app
            if s.type == "text":
                b.type_text(s.text)
            else:
                self._press_combo(s.keys)
            return last_app
        assert s.target is not None
        x, y = self._locate(s.target, allow)
        if s.type == "click":
            b.move(x, y)
            held: list[str] = []
            try:
                for m in s.modifiers:
                    held.append(m)
                    b.key_down(m)
                b.click(s.button, s.count)
            finally:
                for m in reversed(held):
                    b.key_up(m)
        elif s.type == "scroll":
            b.move(x, y)
            b.scroll(s.dx, s.dy)
        else:
            self._drag(s, x, y, factor)
        return s.target.app or last_app

    def _press_combo(self, keys: str) -> None:
        try:
            parts = parse_combo(keys)
        except KeyComboError as exc:
            raise Refused(f'The key "{keys}" is not valid ({exc}).', "edit_steps") from None
        held: list[str] = []
        try:
            for k in parts:
                held.append(k)
                self.backend.key_down(k)
        finally:
            for k in reversed(held):
                self.backend.key_up(k)

    def _drag(self, s: Step, x: int, y: int, factor: float) -> None:
        b = self.backend
        b.move(x, y)
        b.press(s.button)
        try:
            per = (s.duration / factor / max(1, len(s.path))) if factor else 0.0
            for dx, dy in s.path:
                if per and self._sleep(self.abort, min(per, 0.25)):
                    raise Aborted
                b.move(x + dx, y + dy)
        finally:
            b.release(s.button)

    # ---------------------------------------------------------------- safety
    def _check_focus(self, want: str, allow: bool, s: Step) -> None:
        if allow or not want:
            return
        try:
            fg = self.win.foreground()
        except Exception:
            fg = None
        if fg is None or not fg.process:
            return  # cannot tell: the click before it put the right app in front
        if app_key(fg.process) != app_key(want):
            what = "type" if s.type == "text" else f"press {s.keys}"
            raise Refused(
                f"This shortcut would {what} in {app_label(fg.process)}, but you recorded it in {app_label(want)}. "
                f"Bring {app_label(want)} to the front, or let this shortcut work in any app.",
                "allow_other_app",
            )

    def _locate(self, t: Target, allow: bool) -> tuple[int, int]:
        """Screen point for a pointer step, or :class:`Refused`."""
        if not t.app:
            return t.x, t.y
        name = app_label(t.app)
        if self.win.can_locate and t.rx is not None and t.ry is not None:
            w = self.win.find_window(t.app, t.window)
            if w is None:
                if allow:
                    return t.x, t.y
                raise Refused(f"{name} isn't open. Open {name}, then try again.", "try_again")
            pt: tuple[int, int] | None = None
            if t.element and self.cfg.use_ui_automation:
                el = self.win.find_element(w, t.element, t.element_type)
                if el is not None and el.rect[2] > el.rect[0] and el.rect[3] > el.rect[1]:
                    c = ((el.rect[0] + el.rect[2]) // 2, (el.rect[1] + el.rect[3]) // 2)
                    if _inside(c, w.rect):
                        pt = c
            if pt is None:
                pt = (w.rect[0] + t.rx, w.rect[1] + t.ry)
                if not _inside(pt, w.rect):
                    raise Refused(
                        f"{name}'s window is smaller than when you recorded this, so the click would miss. "
                        "Make the window bigger, or record the shortcut again.",
                        "record_again",
                    )
            if not allow:
                under = self.win.window_at(*pt)
                if under is not None and under.process and app_key(under.process) != app_key(t.app):
                    raise Refused(
                        f"{app_label(under.process)} is covering {name} where this shortcut clicks. "
                        f"Bring {name} to the front, or let this shortcut work in any app.",
                        "allow_other_app",
                    )
            return pt
        if not allow:
            fg = self.win.foreground()
            if fg is not None and fg.process and app_key(fg.process) != app_key(t.app):
                raise Refused(
                    f"This shortcut clicks in {name}, but {app_label(fg.process)} is in front. "
                    f"Bring {name} to the front, or let this shortcut work in any app.",
                    "allow_other_app",
                )
        return t.x, t.y


class AbortWatcher:
    """Decides whether a live input event should stop a replay: the abort hotkey or real mouse movement.

    Fed from a listener; events Away produced (marker/injected) never count."""

    def __init__(self, cfg: MacrosConfig, abort: threading.Event, marker: SelfInputMarker | None = None) -> None:
        self.cfg = cfg
        self.abort = abort
        self.marker = marker or MARKER
        self._combo = set(parse_combo(cfg.abort_hotkey)) if cfg.abort_hotkey else set()
        self._held: set[str] = set()
        self._anchor: tuple[int, int] | None = None
        self.reason = ""

    def feed(self, kind: str, p: dict[str, Any], injected: bool = False) -> None:
        if injected or self.marker.is_own():
            if kind == "move":
                self._anchor = (int(p.get("x", 0)), int(p.get("y", 0)))  # our own move: the new reference point
            return
        if kind == "key_down":
            k = str(p.get("key", ""))
            self._held.add(MOD_BASE.get(k, k))
            if self._combo and self._combo <= self._held:
                self.reason = "hotkey"
                self.abort.set()
        elif kind == "key_up":
            k = str(p.get("key", ""))
            self._held.discard(MOD_BASE.get(k, k))
        elif kind == "move" and self.cfg.abort_on_mouse_move:
            pt = (int(p.get("x", 0)), int(p.get("y", 0)))
            if self._anchor is None:
                self._anchor = pt
            elif abs(pt[0] - self._anchor[0]) + abs(pt[1] - self._anchor[1]) >= self.cfg.abort_move_px:
                self.reason = "mouse"
                self.abort.set()
