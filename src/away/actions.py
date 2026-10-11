"""Action mapping and dispatch.

The gesture state machine only *detects* (``click``, ``drag_start``, ``scroll``...).
:class:`Dispatcher` maps each detected event to a trigger (``pinch``,
``pinch_hold``, ``two_finger_scroll``...), looks the trigger up in the active
profile of :class:`~away.bindings.BindingsConfig` and hands the resulting
:class:`~away.bindings.Action` to :class:`ActionExecutor`, which talks to the
mouse/keyboard backend, launches apps, or calls the app's hooks for
dictation, mic, voice-mode and pause actions.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
import webbrowser
from collections.abc import Callable
from typing import Any

from away.bindings import NONE, Action, BindingsConfig, normalize_phrase
from away.control.keys import parse_combo, to_pynput_hotkey
from away.control.mouse import MouseController
from away.hands.gestures import GestureEvent, GestureStateMachine

# detected gesture event -> trigger name (``move`` depends on its anchor)
EVENT_TRIGGER = {
    "click": "pinch",
    "double_click": "double_pinch",
    "drag_start": "pinch_hold",
    "drag_end": "pinch_hold",
    "scroll": "two_finger_scroll",
    "clutch": "open_palm",
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
    "push": "push",  # multi-camera depth: hand pushed toward the screen
    "dwell": "dwell",  # cursor rested in place (hands.dwell)
}
PINCH_CLICK_EVENTS = ("click", "double_click", "drag_start")
SCROLL_STEP = 3.0  # wheel notches for mouse:scroll_up / scroll_down
URL_PREFIXES = ("http://", "https://", "mailto:", "file://", "ftp://")
HOOK_KINDS = ("dictation", "mic", "voice", "pause")

Hook = Callable[[str, dict[str, Any]], None]


def event_trigger(ev: GestureEvent) -> str | None:
    if ev.kind == "move":
        return "pinch_hold" if ev.data.get("anchor") == "pinch" else "point"
    if ev.kind == "custom":
        name = ev.data.get("name")
        return str(name) if name else None
    return EVENT_TRIGGER.get(ev.kind)


def _default_launch(argv: list[str]) -> None:  # pragma: no cover - starts a process
    subprocess.Popen(argv, start_new_session=True)


def _default_shell(cmd: str) -> None:  # pragma: no cover - starts a process
    subprocess.Popen(cmd, shell=True, start_new_session=True)


class ActionExecutor:
    """Runs one :class:`Action`. Launch/shell/URL openers are injectable for tests."""

    def __init__(
        self,
        mouse: MouseController,
        allow_shell: bool = False,
        hooks: dict[str, Hook] | None = None,
        launcher: Callable[[list[str]], None] = _default_launch,
        shell_runner: Callable[[str], None] = _default_shell,
        url_opener: Callable[[str], Any] = webbrowser.open,
    ) -> None:
        self.mouse = mouse
        self.allow_shell = allow_shell
        self.hooks: dict[str, Hook] = dict(hooks or {})
        self.launcher = launcher
        self.shell_runner = shell_runner
        self.url_opener = url_opener
        self.log: list[str] = []  # human-readable record of what ran / was refused
        self.errors: list[tuple[str, BaseException]] = []

    @property
    def backend(self) -> Any:
        return self.mouse.backend

    def run(self, action: Action, data: dict[str, Any] | None = None, *, extra_click_only: bool = False) -> bool:
        """Execute ``action``; returns False if nothing ran (none, refused, missing hook, error)."""
        if action.kind == "none":
            return False
        try:
            return self._run(action, data or {}, extra_click_only)
        except Exception as exc:  # an action must never take down the pipeline
            self.errors.append((str(action), exc))
            self.log.append(f"error {action}: {exc}")
            return False

    def _run(self, a: Action, data: dict[str, Any], extra_click_only: bool) -> bool:
        b = self.backend
        if a.kind == "mouse":
            if a.arg == "click":
                b.click(1)
            elif a.arg == "double_click":
                b.click(1 if extra_click_only else 2)
            elif a.arg in ("right_click", "middle_click"):
                btn = a.arg.split("_")[0]
                if hasattr(b, "click_button"):
                    b.click_button(btn, 1)
                else:
                    self.log.append(f"backend cannot {a.arg}")
                    return False
            elif a.arg in ("scroll_up", "scroll_down"):
                b.scroll(SCROLL_STEP if a.arg == "scroll_up" else -SCROLL_STEP)
            else:  # move/drag/scroll are continuous; the Dispatcher routes those
                return False
        elif a.kind == "key":
            keys = parse_combo(a.arg)
            if len(keys) == 1:
                b.key(keys[0])
            elif hasattr(b, "hotkey"):
                b.hotkey(keys)
            else:
                self.log.append(f"backend cannot press combo {a.arg}")
                return False
        elif a.kind == "type":
            if hasattr(b, "type_text"):
                b.type_text(a.arg)
            else:
                return False
        elif a.kind == "launch":
            if a.arg.lower().startswith(URL_PREFIXES):
                self.url_opener(a.arg)
            else:
                self.launcher(shlex.split(a.arg, posix=os.name != "nt"))
        elif a.kind == "shell":
            if not self.allow_shell:
                self.log.append(f"refused {a}: bindings.allow_shell is false")
                return False
            self.shell_runner(a.arg)
        elif a.kind == "cursor":
            if not self.mouse.jump_monitor(a.arg):
                self.log.append(f"no monitor for {a}")
                return False
        elif a.kind in HOOK_KINDS:
            hook = self.hooks.get(a.kind)
            if hook is None:
                self.log.append(f"no handler for {a}")
                return False
            hook(a.arg, data)
        else:
            return False
        self.log.append(f"ran {a}")
        return True


class Dispatcher:
    """Maps detected events and named triggers to actions under the active profile."""

    def __init__(
        self,
        bindings: BindingsConfig,
        executor: ActionExecutor,
        gestures: GestureStateMachine | None = None,
        active_app: Callable[[], str | None] | None = None,
        paused: Callable[[], bool] = lambda: False,
        app_poll: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bindings = bindings
        self.executor = executor
        self.gestures = gestures
        self.active_app = active_app
        self.paused = paused
        self.app_poll = app_poll
        self.clock = clock
        self._profile = bindings.profile_for_app(None)
        self._app_checked = -1e9
        self.last_app: str | None = None
        self._last_fire: dict[str, float] = {}
        self.fired: list[tuple[str, str]] = []  # (trigger_id, action) history for the GUI/tests
        self.on_fire: Callable[[str, Action], None] | None = None
        self.sync()

    # ------------------------------------------------------------ profile
    @property
    def profile(self) -> str:
        now = self.clock()
        if self.active_app is not None and now - self._app_checked >= self.app_poll:
            self._app_checked = now
            try:
                self.last_app = self.active_app()
            except Exception:
                self.last_app = None
            name = self.bindings.profile_for_app(self.last_app)
            if name != self._profile:
                self._profile = name
                self.sync()
        elif self.active_app is None:
            name = self.bindings.profile_for_app(None)
            if name != self._profile:
                self._profile = name
                self.sync()
        return self._profile

    def set_bindings(self, bindings: BindingsConfig) -> None:
        self.bindings = bindings
        self.executor.allow_shell = bindings.allow_shell
        self._app_checked = -1e9
        self._profile = bindings.profile_for_app(self.last_app)
        self.sync()

    def sync(self) -> None:
        """Push mapping-dependent switches into the detector (open palm pauses only if bound to pause:toggle)."""
        if self.gestures is not None:
            act = self.bindings.resolve("gestures", "open_palm", self._profile)
            self.gestures.clutch_pauses = act == Action("pause", "toggle")

    def resolve(self, category: str, trigger: str) -> Action:
        return self.bindings.resolve(category, trigger, self.profile)

    # ------------------------------------------------------------ gestures
    def on_gesture(self, ev: GestureEvent) -> Action:
        mouse = self.executor.mouse
        if ev.kind == "hand_lost":
            mouse.handle(ev)
            return NONE
        trig = event_trigger(ev)
        if trig is None:
            return NONE
        act = self.resolve("gestures", trig)
        if ev.kind == "drag_end":
            mouse.handle(ev)  # always release a held button
            return act
        if self.paused() and act.kind != "pause":
            return NONE
        if ev.kind in PINCH_CLICK_EVENTS and act.kind == "mouse":
            mouse.steady(ev.t)  # steady click: land where the cursor was before the pinch began
        cont = {"point": "move", "pinch_hold": "drag", "two_finger_scroll": "scroll"}.get(trig)
        if cont is not None and act == Action("mouse", cont):
            if ev.kind in ("move", "drag_start", "scroll"):
                mouse.handle(ev)
                self._record(f"gesture:{trig}", act, quiet=ev.kind != "drag_start")
            return act
        if ev.kind == "move":
            return NONE  # a non-drag action on pinch_hold fires once on drag_start, not every frame
        if trig in ("two_finger_scroll",) or ev.kind == "custom":
            last = self._last_fire.get(trig, -1e9)
            if ev.t - last < self.bindings.discrete_cooldown:
                return NONE
            self._last_fire[trig] = ev.t
        extra_only = False
        if trig == "double_pinch" and act == Action("mouse", "double_click"):
            # the first tap of a double pinch already produced a left click when pinch -> mouse:click
            extra_only = self.resolve("gestures", "pinch") == Action("mouse", "click")
        if self.executor.run(act, dict(ev.data), extra_click_only=extra_only):
            self._record(f"gesture:{trig}", act)
        return act

    # ------------------------------------------------------------ other sources
    def trigger(self, category: str, name: str, data: dict[str, Any] | None = None) -> Action:
        """Fire a voice/face/hotkey trigger by name. Returns the action (``none`` if unbound or paused)."""
        if category == "voice":
            name = normalize_phrase(name)
        act = self.resolve(category, name)
        if self.paused() and act.kind in ("mouse", "cursor"):
            return NONE
        if self.executor.run(act, dict(data or {})):
            prefix = {"voice": "voice", "face": "face", "hotkeys": "hotkey", "gestures": "gesture"}[category]
            self._record(f"{prefix}:{name}", act)
            return act
        return NONE if act.kind == "none" else act

    def _record(self, trig: str, act: Action, quiet: bool = False) -> None:
        if quiet:
            return
        self.fired.append((trig, str(act)))
        del self.fired[:-200]
        if self.on_fire is not None:
            self.on_fire(trig, act)


class CustomGestureDetector:
    """Fires a custom gesture when the temporal model's probability for it rises above ``min_conf``.

    Rising-edge + cooldown so one performance fires once.
    """

    def __init__(self, names: list[str], min_conf: float = 0.7, cooldown: float = 1.0, release: float = 0.4) -> None:
        self.names = list(names)
        self.min_conf = min_conf
        self.release = release
        self.cooldown = cooldown
        self._armed = dict.fromkeys(self.names, True)
        self._last = dict.fromkeys(self.names, -1e9)

    def update(self, t: float, probs: dict[str, float] | None) -> list[str]:
        if not probs:
            return []
        out = []
        for n in self.names:
            p = float(probs.get(n, 0.0))
            if p < self.release:
                self._armed[n] = True
            elif p >= self.min_conf and self._armed[n] and t - self._last[n] >= self.cooldown:
                self._armed[n] = False
                self._last[n] = t
                out.append(n)
        return out


def hotkey_slots(voice_cfg: Any) -> dict[str, str]:
    """Named hotkey slot -> key combo, from the voice config."""
    return {"ptt": str(voice_cfg.hotkey), "mute": str(voice_cfg.mute_hotkey)}


def hotkey_map(bindings: BindingsConfig, voice_cfg: Any, profile: str = "default") -> dict[str, str]:
    """pynput GlobalHotKeys combo -> hotkey trigger name for every bound, non-``none`` hotkey.

    In ``push_to_talk`` mode the ``ptt`` slot is handled by the hold-key listener instead.
    """
    slots = hotkey_slots(voice_cfg)
    out: dict[str, str] = {}
    for name, act in bindings.effective(profile)["hotkeys"].items():
        if act.kind == "none":
            continue
        if name == "ptt" and getattr(voice_cfg, "mode", "") == "push_to_talk":
            continue
        combo = slots.get(name, name)
        out[to_pynput_hotkey(combo)] = name
    return out
