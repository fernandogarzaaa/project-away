"""Wires smart cleanup and the personal dictionary into the running app (called from ``App._start_voice``).

Kept out of ``runner.py`` so the app only needs three calls: :func:`setup_voice_context`, ``hotwords()`` and
``attach()``. Nothing here may stop voice from starting: on any error the plain rule cleaner is used.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from away.voice.cleanup import RuleCleaner, Vocabulary
from away.voice.dictionary import CorrectionWatcher, Dictionary, default_path


@dataclass
class VoiceContext:
    cleaner: Any
    vocab: Vocabulary
    dictionary: Dictionary | None = None
    watcher: CorrectionWatcher | None = None
    terms: list[str] = field(default_factory=list)  # live list shared with the dictation sessions (Whisper prompt)
    _asr: Any = None

    def _merged(self) -> list[str]:
        words = self.dictionary.words() if self.dictionary is not None else []
        return list(dict.fromkeys([*self.vocab.terms(), *words]))

    def hotwords(self, limit: int = 60) -> str | None:
        seen = [t for t in self._merged() if t]
        return " ".join(seen[:limit]) or None

    def refresh(self) -> None:
        """The dictionary changed: update the Whisper prompt terms (in place) and the hotwords."""
        self.terms[:] = self._merged()
        if self._asr is not None and hasattr(self._asr, "hotwords") and self._asr.hotwords is not None:
            self._asr.hotwords = self.hotwords()

    def attach(self, asr: Any, *sessions: Any) -> None:
        """Share the live term list with the dictation sessions (they keep their own reference)."""
        self._asr = asr
        for s in sessions:
            if s is not None and hasattr(s, "vocabulary"):
                s.vocabulary = self.terms


def _dictionary_path(cfg: Any) -> Path:
    f = cfg.voice.dictionary.file
    return Path(f).expanduser() if f else default_path()


def setup_voice_context(app: Any, rules: RuleCleaner, vocab: Vocabulary) -> VoiceContext:
    cfg = app.cfg
    v = cfg.voice
    ctx = VoiceContext(cleaner=rules, vocab=vocab)
    try:
        dictionary = None
        if v.dictionary.enabled:
            dictionary = Dictionary.load(_dictionary_path(cfg), v.dictionary.learn_after)
        ctx.dictionary = dictionary
        ctx.terms[:] = ctx._merged()

        from away.voice import smart_cleanup
        from away.voice.context import active_app, active_process

        def publish(topic: str, data: dict[str, Any]) -> None:
            bus = getattr(app, "bus", None)
            if bus is not None:
                bus.publish(topic, data)

        def profile_for(process: str | None) -> str | None:
            try:
                return str(cfg.bindings.profile_for_app(process))
            except Exception:
                return None

        prof = getattr(app, "profile", None)
        gpu = bool(getattr(prof, "device", "cpu") == "cuda")
        cleaner = smart_cleanup.build(
            cfg,
            Path(app.models_dir),
            rules,
            gpu=gpu,
            process=active_process,
            profile=profile_for,
            on_event=publish,
            dictionary=dictionary,
        )
        if v.cleanup.context_clipboard and not getattr(app, "dry_run", False):
            cleaner.selection = _clipboard_selection(app)
        ctx.cleaner = cleaner
        if dictionary is not None:
            dictionary.on_change.append(ctx.refresh)
            if v.dictionary.learn and sys.platform.startswith("win"):
                from away.voice.surrounding import uia_field_text

                watcher = CorrectionWatcher(
                    dictionary,
                    uia_field_text,
                    app=active_app,
                    watch_seconds=float(v.dictionary.watch_seconds),
                    on_learned=lambda w, r: publish("dictionary.learned", {"wrong": w, "right": r}),
                )
                cleaner.on_inserted = watcher.on_inserted
                ctx.watcher = watcher
                _start_poller(app, watcher)
        if cleaner.mode == "llm":
            threading.Thread(target=_warm, args=(app, cleaner), daemon=True, name="away-cleanup-warmup").start()
    except Exception as exc:  # never let an optional feature stop dictation
        bus = getattr(app, "bus", None)
        if bus is not None:
            bus.publish("cleanup.unavailable", {"error": f"{type(exc).__name__}: {exc}", "fix": "none"})
        ctx.cleaner = rules
    return ctx


def _warm(app: Any, cleaner: Any) -> None:
    err = cleaner.warm_up()
    if err and hasattr(app, "notice"):
        app.notice("Project Away: Smart cleanup is off for now", f"{err} Open Settings > Voice to fix it.", "cleanup")


def _start_poller(app: Any, watcher: CorrectionWatcher) -> None:
    stop = getattr(app, "stop_event", None)
    if stop is None:
        return

    def loop() -> None:
        while not stop.wait(2.0):
            with contextlib.suppress(Exception):
                watcher.poll()
        watcher.flush()

    threading.Thread(target=loop, daemon=True, name="away-dictionary-watch").start()


def _clipboard_selection(app: Any) -> Any:  # pragma: no cover - needs a desktop session
    from away.control.keys import parse_combo
    from away.voice.edit import SystemClipboard
    from away.voice.surrounding import ClipboardSelection

    mod = "cmd" if sys.platform == "darwin" else "ctrl"
    b = app.pipeline.executor.backend
    return ClipboardSelection(SystemClipboard(), lambda: b.hotkey(parse_combo(f"{mod}+c"))).read
