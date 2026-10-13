"""Record a shortcut from your own mouse and keyboard.

:class:`MacroRecorder` takes raw events from a listener (``feed``), keeps the ones that are yours and turns them
into steps when you stop. What it leaves out:

* input Project Away produced itself (gesture clicks, voice commands, a shortcut replaying): reported through
  :class:`SelfInputMarker` by every Away output path, and on Windows also every event the OS marks as injected;
* clicks on Project Away's own windows (Settings, the "Recording" bar) and, when you stop from the tray, the
  click on the tray icon (its menu belongs to Away's own process);
* every keystroke while a password box has focus (UI Automation ``IsPassword`` / ``ES_PASSWORD``). Where that
  cannot be checked, typed characters are left out unless ``macros.record_text_without_password_check``.

The OS work (window under the click, control name, password check) runs on the recorder's worker thread, never in
the input hook, so a slow app cannot stall your mouse.
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from away.control.macro_platform import SHELL_TRAY_CLASSES, NullWindowApi
from away.control.macros import MOD_BASE, MODIFIERS, Macro, MacrosConfig, RawEvent, Target, build_steps

MAX_RAW_EVENTS = 50_000
_MOD_KEYS = {*MODIFIERS, *MOD_BASE}

Item = tuple[str, dict[str, Any], bool, float, bool]


class SelfInputMarker:
    """Remembers when Project Away itself is sending input, so recorders and abort watchers can ignore it."""

    def __init__(self, window_s: float = 0.15, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s = window_s
        self.clock = clock
        self._lock = threading.Lock()
        self._active = 0
        self._last = -1e9

    def mark(self) -> None:
        with self._lock:
            self._last = self.clock()

    @contextlib.contextmanager
    def busy(self) -> Iterator[None]:
        with self._lock:
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                self._last = self.clock()

    def is_own(self, t: float | None = None) -> bool:
        now = self.clock() if t is None else t
        with self._lock:
            return self._active > 0 or now - self._last <= self.window_s


MARKER = SelfInputMarker()  # process-wide: the app's mouse backend and replays report here


class MarkingBackend:
    """Wraps an output backend (the gesture mouse) and reports every call to the marker."""

    def __init__(self, backend: Any, marker: SelfInputMarker | None = None) -> None:
        self._b = backend
        self._marker = marker or MARKER

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._b, name)
        if not callable(attr) or name.startswith("_"):
            return attr

        def wrapped(*a: Any, **k: Any) -> Any:
            with self._marker.busy():
                return attr(*a, **k)

        return wrapped


class MacroRecorder:
    def __init__(
        self,
        cfg: MacrosConfig | None = None,
        window_api: Any = None,
        marker: SelfInputMarker | None = None,
        clock: Callable[[], float] = time.monotonic,
        own_pid: int | None = None,
        threaded: bool = True,
        cache_s: float = 0.15,
    ) -> None:
        self.cfg = cfg or MacrosConfig()
        self.win = window_api or NullWindowApi()
        self.marker = marker or MARKER
        self.clock = clock
        self.own_pid = os.getpid() if own_pid is None else own_pid
        self.threaded = threaded
        self.cache_s = cache_s
        self.exclude_rects: list[tuple[int, int, int, int]] = []  # screen areas whose clicks are never steps
        self.recording = False
        self.events: list[RawEvent] = []
        self.skipped_password_keys = 0
        self.skipped_text = False
        self.dropped_own = 0
        self._q: queue.Queue[Item | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._reset()

    def _reset(self) -> None:
        self.events = []
        self.skipped_password_keys = 0
        self.skipped_text = False
        self.dropped_own = 0
        self._dropped_buttons: set[str] = set()
        self._first_app = ""
        self._pw_t = -1e9
        self._pw: bool | None = None
        self._fg_t = -1e9
        self._fg = ""
        self._in_password = False

    @property
    def record_text(self) -> bool:
        return bool(self.win.can_check_password or self.cfg.record_text_without_password_check)

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._reset()
        self.recording = True
        if self.threaded:
            self._q = queue.Queue()
            self._worker = threading.Thread(target=self._run, name="macro-record", daemon=True)
            self._worker.start()

    def stop(self, trim_tail_s: float = 0.0, max_delay: float | None = None) -> Macro:
        """Stop and build the draft shortcut (not saved; it has no name yet).

        ``trim_tail_s``: drop pointer events from the last seconds (stopping from a menu whose clicks cannot be
        told apart from yours). Trailing clicks on the taskbar/tray are always dropped."""
        stop_t = self.clock()
        self.recording = False
        if self._worker is not None:
            self._q.put(None)
            self._worker.join(timeout=5.0)
            self._worker = None
        evs = list(self.events)
        if trim_tail_s > 0:
            evs = [e for e in evs if e.kind.startswith("key") or e.t < stop_t - trim_tail_s]
        drop_tray_tail(evs)
        steps = build_steps(evs, self.cfg.max_delay_s if max_delay is None else max_delay, self.record_text)
        app = next((s.target.app for s in steps if s.target is not None and s.target.app), "") or self._first_app
        return Macro(
            name="",
            steps=steps,
            app=app,
            speed=self.cfg.default_speed,
            created=time.time(),
            skipped_password_keys=self.skipped_password_keys,
            skipped_text=self.skipped_text,
        )

    # ---------------------------------------------------------------- input
    def feed(self, kind: str, payload: dict[str, Any], injected: bool = False) -> None:
        """Called from the input hook: cheap. The own-input decision is taken now, while it is still true."""
        if not self.recording:
            return
        t = self.clock()
        item = (kind, payload, injected, t, self.marker.is_own(t))
        if self.threaded:
            self._q.put(item)
        else:
            self._handle(item)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            with contextlib.suppress(Exception):  # one odd event must not end the recording
                self._handle(item)

    def _cached(self, t: float, which: str) -> Any:
        if which == "pw":
            if t - self._pw_t > self.cache_s:
                self._pw_t = t
                try:
                    self._pw = self.win.focused_is_password()
                except Exception:
                    self._pw = None
            return self._pw
        if t - self._fg_t > self.cache_s:
            self._fg_t = t
            try:
                fg = self.win.foreground()
                self._fg = fg.process if fg is not None and fg.pid != self.own_pid else ""
            except Exception:
                self._fg = ""
        return self._fg

    def _handle(self, item: Item) -> None:
        kind, p, injected, t, own = item
        if len(self.events) >= MAX_RAW_EVENTS:
            return
        if (injected or own) and not self.cfg.record_away_input:
            self.dropped_own += 1
            return
        if kind in ("key_down", "key_up"):
            self._key(kind, p, t)
            return
        x, y = int(p.get("x", 0)), int(p.get("y", 0))
        if kind == "move":
            self.events.append(RawEvent(t, "move", x, y))
            return
        button = str(p.get("button", "left"))
        if kind == "up":
            if button in self._dropped_buttons:
                self._dropped_buttons.discard(button)
                return
            self.events.append(RawEvent(t, "up", x, y, button=button))
            return
        if any(r[0] <= x < r[2] and r[1] <= y < r[3] for r in self.exclude_rects):
            if kind == "down":
                self._dropped_buttons.add(button)
            return
        target, win = self._target(x, y)
        if win is not None and win.pid == self.own_pid:
            if kind == "down":
                self._dropped_buttons.add(button)  # Settings / the recording bar: not part of the shortcut
            return
        tray = win is not None and win.cls in SHELL_TRAY_CLASSES
        ev = RawEvent(t, kind, x, y, button=button, dx=int(p.get("dx", 0)), dy=int(p.get("dy", 0)), target=target)
        ev.tray = tray
        self.events.append(ev)

    def _target(self, x: int, y: int) -> tuple[Target, Any]:
        tgt = Target(x, y)
        try:
            win = self.win.window_at(x, y)
        except Exception:
            win = None
        if win is not None:
            tgt.app, tgt.window = win.process, win.title
            if self.win.can_locate:
                tgt.rx, tgt.ry = x - win.rect[0], y - win.rect[1]
            if not self._first_app and win.pid != self.own_pid and win.cls not in SHELL_TRAY_CLASSES:
                self._first_app = win.process
        if self.cfg.use_ui_automation:
            try:
                el = self.win.element_at(x, y)
            except Exception:
                el = None
            if el is not None and el.name and len(el.name) <= 120:
                tgt.element, tgt.element_type = el.name, el.control_type
        return tgt, win

    def _key(self, kind: str, p: dict[str, Any], t: float) -> None:
        key = str(p.get("key", ""))
        char = str(p.get("char", ""))
        if key in _MOD_KEYS:  # modifiers always pass, so the held-key state stays right
            self.events.append(RawEvent(t, kind, key=key))
            return
        if kind == "key_down":
            pw = self._cached(t, "pw")
            self._in_password = bool(pw)
            if pw:
                self.skipped_password_keys += 1
                return
            app = self._cached(t, "fg")
            if not self._first_app and app:
                self._first_app = app
            if not self.record_text and (char or len(key) == 1 or key == "space"):
                self.skipped_text = True
            self.events.append(RawEvent(t, kind, key=key, char=char, app=app))
        elif not self._in_password:
            self.events.append(RawEvent(t, kind, key=key, char=char))


def drop_tray_tail(evs: list[RawEvent]) -> None:
    """Remove trailing clicks on the taskbar/tray (right-click the tray icon to reach "Stop recording"); the
    menu item itself belongs to Away's own process and was never recorded."""
    while evs:
        last = evs[-1]
        if last.kind == "move" or (last.kind == "down" and last.tray):
            evs.pop()
            continue
        if last.kind == "up":
            i = len(evs) - 2
            while i >= 0 and not (evs[i].kind == "down" and evs[i].button == last.button):
                i -= 1
            if i >= 0 and evs[i].tray:
                del evs[i:]
                continue
        break
