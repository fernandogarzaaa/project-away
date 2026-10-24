"""Deliver text to the focused app: clipboard paste (fast, Unicode-safe) or synthetic typing."""

from __future__ import annotations

import contextlib
import sys
import time
from typing import Any, Protocol


class TextSink(Protocol):
    def emit(self, text: str) -> None: ...


class ListSink:
    def __init__(self) -> None:
        self.items: list[str] = []

    def emit(self, text: str) -> None:
        self.items.append(text)


class PasteSink:  # pragma: no cover - needs a desktop session
    """Copy to clipboard, press Ctrl/Cmd+V, then restore the previous clipboard. The key press goes through the
    action engine as ``dictation`` (Stop everything cancels it before the paste)."""

    def __init__(self, restore_delay: float = 0.3, backend: Any = None) -> None:
        import pyperclip

        from away.actions_engine import get_engine

        self._clip = pyperclip
        self._engine = get_engine()
        self._backend = backend if backend is not None else self._engine.os_backend()
        self._mod = "cmd" if sys.platform == "darwin" else "ctrl"
        self.restore_delay = restore_delay

    def emit(self, text: str) -> None:
        emit_paste(self._engine, self._backend, self._clip, self._mod, text, self.restore_delay)


class TypeSink:  # pragma: no cover - needs a desktop session
    """Synthetic typing through the action engine as ``dictation``, a few characters at a time so Stop everything
    interrupts a long dictation part-way."""

    def __init__(self, backend: Any = None) -> None:
        from away.actions_engine import get_engine

        self._engine = get_engine()
        self._backend = backend if backend is not None else self._engine.os_backend()

    def emit(self, text: str) -> None:
        emit_typed(self._engine, self._backend, text)


TYPE_CHUNK = 16  # characters typed per engine step (each step can be cancelled)


def emit_typed(engine: Any, backend: Any, text: str, chunk: int = TYPE_CHUNK) -> int:
    """Type ``text`` as the ``dictation`` source; returns how many characters went out (fewer when stopped)."""
    from away.actions_engine import DICTATION, ActionCancelled

    sent = 0
    try:
        with engine.begin(DICTATION, "type_text", backend=backend) as a:
            for i in range(0, len(text), chunk):
                part = text[i : i + chunk]
                a.step().type_text(part)
                sent += len(part)
            a.ok = True
    except ActionCancelled:
        pass  # Stop everything: the rest of the dictation is dropped
    return sent


def emit_paste(engine: Any, backend: Any, clip: Any, mod: str, text: str, restore_delay: float = 0.3) -> bool:
    """Paste ``text`` as the ``dictation`` source (clipboard, Ctrl/Cmd+V, clipboard restored). False if stopped."""
    from away.actions_engine import DICTATION, ActionCancelled

    try:
        old = clip.paste()
    except Exception:
        old = None
    try:
        with engine.begin(DICTATION, "paste", backend=backend) as a:
            a.check()
            clip.copy(text)
            a.step().hotkey([mod, "v"])
            a.ok = True
        time.sleep(restore_delay)
        return True
    except ActionCancelled:
        return False
    finally:
        if old is not None:
            with contextlib.suppress(Exception):
                clip.copy(old)


def make_sink(kind: str, backend: Any = None) -> TextSink:  # pragma: no cover
    """``backend``: the OS backend to type through (the app passes one wrapped by the auto-pause guard)."""
    return PasteSink(backend=backend) if kind == "paste" else TypeSink(backend=backend)
