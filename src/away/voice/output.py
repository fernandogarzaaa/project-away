"""Deliver text to the focused app: clipboard paste (fast, Unicode-safe) or synthetic typing."""

from __future__ import annotations

import sys
import time
from typing import Protocol


class TextSink(Protocol):
    def emit(self, text: str) -> None: ...


class ListSink:
    def __init__(self) -> None:
        self.items: list[str] = []

    def emit(self, text: str) -> None:
        self.items.append(text)


class PasteSink:  # pragma: no cover - needs a desktop session
    """Copy to clipboard, press Ctrl/Cmd+V, then restore the previous clipboard."""

    def __init__(self, restore_delay: float = 0.3) -> None:
        import pyperclip
        from pynput import keyboard

        self._clip = pyperclip
        self._kb = keyboard.Controller()
        self._mod = keyboard.Key.cmd if sys.platform == "darwin" else keyboard.Key.ctrl
        self.restore_delay = restore_delay

    def emit(self, text: str) -> None:
        try:
            old = self._clip.paste()
        except Exception:
            old = None
        self._clip.copy(text)
        with self._kb.pressed(self._mod):
            self._kb.press("v")
            self._kb.release("v")
        time.sleep(self.restore_delay)
        if old is not None:
            self._clip.copy(old)


class TypeSink:  # pragma: no cover - needs a desktop session
    def __init__(self) -> None:
        from pynput import keyboard

        self._kb = keyboard.Controller()

    def emit(self, text: str) -> None:
        self._kb.type(text)


def make_sink(kind: str) -> TextSink:  # pragma: no cover
    return PasteSink() if kind == "paste" else TypeSink()
