"""Shortcut recording and replay for a running process (the tray app, or the Settings window).

:class:`MacroService` owns the recorder, the player and their input listeners, runs replays on a worker thread
and is what the bindings call: ``macro:<name>`` runs a shortcut, ``macro_control:record | stop_recording |
toggle_recording | stop`` drives recording and stops a replay (voice: "record shortcut", "stop recording",
"stop"). Every device is injected (listener and backend factories, window API), so it is tested with fakes.

The outcome of the last replay is written to ``<macros folder>/_last_result.json`` so Settings can show what went
wrong with a fix button even when the shortcut ran from a gesture in the tray app.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.control.macro_play import AbortWatcher, MacroPlayer, PlayResult
from away.control.macro_record import MARKER, MacroRecorder, SelfInputMarker
from away.control.macros import Macro, MacroError, MacrosConfig, MacroStore, name_from_title

STATUS_FILE = "_last_result.json"

Notify = Callable[[str, str], None]
Sink = Callable[[str, dict[str, Any], bool], None]


def _default_listener(sink: Sink) -> Any:  # pragma: no cover - desktop
    from away.control.macro_platform import PynputListener

    return PynputListener(sink)


def _default_backend(mark: Callable[[], None]) -> Any:  # pragma: no cover - desktop
    from away.control.macro_platform import PynputMacroBackend

    return PynputMacroBackend(mark)


class MacroService:
    def __init__(
        self,
        cfg: MacrosConfig | None = None,
        store: MacroStore | None = None,
        window_api: Any = None,
        listener_factory: Callable[[Sink], Any] | None = None,
        backend_factory: Callable[[Callable[[], None]], Any] | None = None,
        notify: Notify | None = None,
        on_replay: Callable[[bool], None] | None = None,
        marker: SelfInputMarker | None = None,
        threaded: bool = True,
    ) -> None:
        self.cfg = cfg or MacrosConfig()
        self.store = store or MacroStore.default(self.cfg)
        self._window_api = window_api
        self._listener_factory = listener_factory or _default_listener
        self._backend_factory = backend_factory or _default_backend
        self.notify: Notify = notify or (lambda _t, _m: None)
        self.on_replay = on_replay
        self.marker = marker or MARKER
        self.threaded = threaded
        self.recorder: MacroRecorder | None = None
        self._rec_listener: Any = None
        self._play_thread: threading.Thread | None = None
        self._player: MacroPlayer | None = None
        self._lock = threading.Lock()
        self.last_result: PlayResult | None = None
        self.last_name = ""

    # ---------------------------------------------------------------- shared
    @property
    def window_api(self) -> Any:
        if self._window_api is None:  # pragma: no cover - OS specific
            from away.control.macro_platform import default_window_api

            self._window_api = default_window_api(self.cfg.use_ui_automation)
        return self._window_api

    def set_config(self, cfg: MacrosConfig) -> None:
        self.cfg = cfg
        if cfg.store_dir:
            self.store = MacroStore.default(cfg)

    @property
    def recording(self) -> bool:
        return self.recorder is not None and self.recorder.recording

    @property
    def playing(self) -> bool:
        return self._player is not None

    # ---------------------------------------------------------------- recording
    def start_recording(self, exclude_rects: list[tuple[int, int, int, int]] | None = None) -> str | None:
        """Start recording. Returns a friendly error, or None."""
        if not self.cfg.enabled:
            return "Shortcuts are turned off (macros.enabled: false in the config file)."
        if self.playing:
            return 'A shortcut is running. Wait for it to finish (or say "stop"), then record.'
        if self.recording:
            return None
        rec = MacroRecorder(self.cfg, self.window_api, self.marker, threaded=self.threaded)
        rec.exclude_rects = list(exclude_rects or [])
        rec.start()
        try:
            lst = self._listener_factory(rec.feed)
            lst.start()
        except Exception as exc:
            rec.stop()
            return f"Recording could not start because the mouse and keyboard can't be watched here ({exc})."
        self.recorder, self._rec_listener = rec, lst
        return None

    def stop_recording(self, trim_tail_s: float = 0.0) -> Macro | None:
        """Stop and return the draft (unnamed, unsaved). None when nothing was being recorded."""
        rec, lst = self.recorder, self._rec_listener
        if rec is None:
            return None
        if lst is not None:
            with contextlib.suppress(Exception):
                lst.stop()
        self.recorder, self._rec_listener = None, None
        return rec.stop(trim_tail_s=trim_tail_s)

    def save_draft(self, draft: Macro, title: str = "") -> Macro:
        """Save a recording under ``title`` (or the next "Shortcut N"). Raises :class:`MacroError`."""
        if title.strip():
            name = name_from_title(title)
            if not name:
                raise MacroError("Give the shortcut a name with at least one letter or number.")
            draft.name, draft.title = name, title.strip()
        else:
            draft.name, draft.title = self.store.next_untitled()
        if self.store.exists(draft.name):
            raise MacroError(f'You already have a shortcut called "{draft.title}". Pick another name.')
        self.store.save(draft)
        return draft

    # ---------------------------------------------------------------- replay
    def play(self, name: str, wait: bool = False, speed: str | None = None) -> PlayResult | None:
        """Run a saved shortcut (on a worker thread unless ``wait``). Returns the result when it ran inline or was
        refused before starting."""
        if not self.cfg.enabled:
            return self._finish(name, PlayResult(False, stopped="refused", message="Shortcuts are turned off."))
        if self.recording:
            return self._finish(
                name, PlayResult(False, stopped="busy", message="Finish recording first, then run a shortcut.")
            )
        with self._lock:
            if self.playing:
                return PlayResult(False, stopped="busy", message="Another shortcut is still running.")
            try:
                macro = self.store.load(name)
            except MacroError as exc:
                return self._finish(name, PlayResult(False, stopped="refused", message=str(exc), fix="record_again"))
            player = MacroPlayer(self._backend_factory(self.marker.mark), self.window_api, self.cfg, self.marker)
            self._player = player
            if not (wait or not self.threaded):
                self._play_thread = threading.Thread(
                    target=self._run, args=(macro, player, speed), name="macro-play", daemon=True
                )
                self._play_thread.start()
                return None
        return self._run(macro, player, speed)

    def _run(self, macro: Macro, player: MacroPlayer, speed: str | None) -> PlayResult:
        watcher = AbortWatcher(self.cfg, player.abort, self.marker)
        lst = None
        try:
            lst = self._listener_factory(watcher.feed)
            lst.start()
        except Exception:
            lst = None  # no abort hotkey on this system; voice "stop" and the tray still work
        if self.on_replay is not None:
            with contextlib.suppress(Exception):
                self.on_replay(True)
        try:
            res = player.run(macro, speed=speed)
        finally:
            if lst is not None:
                with contextlib.suppress(Exception):
                    lst.stop()
            if self.on_replay is not None:
                with contextlib.suppress(Exception):
                    self.on_replay(False)
            self._player = None
        return self._finish(macro.name, res)

    def stop_playback(self) -> bool:
        p = self._player
        if p is None:
            return False
        p.abort.set()
        return True

    def wait(self, timeout: float = 30.0) -> None:
        t = self._play_thread
        if t is not None:
            t.join(timeout)

    def _finish(self, name: str, res: PlayResult) -> PlayResult:
        self.last_result, self.last_name = res, name
        self._write_status(name, res)
        if not res.ok and res.stopped in ("refused", "error"):
            self.notify("Shortcut didn't run", res.message + " (Settings > Shortcuts has a fix button.)")
        return res

    def _write_status(self, name: str, res: PlayResult) -> None:
        with contextlib.suppress(OSError):
            self.store.folder.mkdir(parents=True, exist_ok=True)
            data = {"name": name, "time": time.time(), **res.to_dict()}
            (self.store.folder / STATUS_FILE).write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def read_status(folder: Path) -> dict[str, Any] | None:
        try:
            data = json.loads((Path(folder) / STATUS_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def clear_status(folder: Path) -> None:
        with contextlib.suppress(OSError):
            (Path(folder) / STATUS_FILE).unlink()

    # ---------------------------------------------------------------- bindings hooks
    def hook_macro(self, arg: str, _data: dict[str, Any]) -> None:
        """``macro:<name>`` from a gesture, phrase or hotkey."""
        self.play(arg)

    def hook_control(self, arg: str, _data: dict[str, Any]) -> None:
        """``macro_control:record | stop_recording | toggle_recording | stop``."""
        if arg == "stop":  # stops a replay only; "stop" alone never ends a recording (say "stop recording")
            self.stop_playback()
            return
        if arg == "toggle_recording":
            arg = "stop_recording" if self.recording else "record"
        if arg == "record":
            err = self.start_recording()
            if err:
                self.notify("Can't record", err)
            else:
                self.notify("Recording a shortcut", 'Do the steps now. Say "stop recording" when you are done.')
        elif arg == "stop_recording":
            self.finish_quick_recording()

    def finish_quick_recording(self, trim_tail_s: float = 0.0) -> Macro | None:
        """Stop a recording started by voice or the tray and save it as "Shortcut N" (named later in Settings)."""
        draft = self.stop_recording(trim_tail_s)
        if draft is None:
            return None
        if not draft.steps:
            self.notify("Nothing recorded", "No clicks or typing were recorded, so nothing was saved.")
            return None
        try:
            saved = self.save_draft(draft)
        except MacroError as exc:
            self.notify("Shortcut not saved", str(exc))
            return None
        self.notify(
            "Shortcut saved",
            f'Saved as "{saved.title}" ({len(saved.steps)} steps). Open Settings > Shortcuts to name it and pick a '
            "gesture or phrase for it.",
        )
        return saved
