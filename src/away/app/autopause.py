"""Auto-pause gesture control while the physical mouse or keyboard is in use.

:class:`ManualInputGuard` is pure logic with an injectable clock. The app feeds it
events from pynput listeners; Away's own synthetic input is reported through
:class:`GuardedBackend` so it is not mistaken for the user (an OS listener sees
both). Idea from Tracky Mouse (MIT), implementation is original.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any


class ManualInputGuard:
    def __init__(
        self,
        idle_ms: float = 1500.0,
        clock: Callable[[], float] = time.monotonic,
        synthetic_window_s: float = 0.08,
        key_window_s: float = 0.5,
        tol_px: float = 3.0,
    ) -> None:
        self.idle_s = idle_ms / 1000.0
        self.clock = clock
        self.synthetic_window_s = synthetic_window_s
        self.key_window_s = key_window_s
        self.tol_px = tol_px
        self.enabled = True
        self._until = -1e9
        self._synth_pos: tuple[float, float] | None = None
        self._synth_t = -1e9
        self._synth_key_t = -1e9

    # Away's own output --------------------------------------------------------
    def note_synthetic_move(self, x: float, y: float) -> None:
        self._synth_pos, self._synth_t = (x, y), self.clock()

    def note_synthetic_key(self) -> None:
        self._synth_key_t = self.clock()

    # the user's input -----------------------------------------------------------
    def on_mouse_move(self, x: float, y: float) -> bool:
        now = self.clock()
        if self._synth_pos is not None and (
            now - self._synth_t <= self.synthetic_window_s or math.dist((x, y), self._synth_pos) <= self.tol_px
        ):
            return False
        return self._manual(now)

    def on_click(self) -> bool:
        now = self.clock()
        if now - self._synth_t <= self.synthetic_window_s or now - self._synth_key_t <= self.key_window_s:
            return False
        return self._manual(now)

    def on_key(self) -> bool:
        now = self.clock()
        if now - self._synth_key_t <= self.key_window_s:
            return False
        return self._manual(now)

    def _manual(self, now: float) -> bool:
        if not self.enabled:
            return False
        self._until = now + self.idle_s
        return True

    @property
    def active(self) -> bool:
        """True while gesture control should stay paused."""
        return self.enabled and self.clock() < self._until


class GuardedBackend:
    """Wraps a mouse backend and tells the guard about every synthetic event it sends."""

    def __init__(self, backend: Any, guard: ManualInputGuard) -> None:
        self._b = backend
        self._g = guard

    def move(self, x: int, y: int) -> None:
        self._g.note_synthetic_move(x, y)
        self._b.move(x, y)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._b, name)
        if callable(attr) and name in (
            "click",
            "press",
            "release",
            "scroll",
            "key",
            "click_button",
            "hotkey",
            "type_text",
        ):

            def wrapped(*a: Any, **k: Any) -> Any:
                if name in ("key", "hotkey", "type_text"):
                    self._g.note_synthetic_key()
                else:
                    self._g._synth_t = self._g.clock()
                return attr(*a, **k)

            return wrapped
        return attr


def start_listeners(guard: ManualInputGuard) -> list[Any]:  # pragma: no cover - needs a desktop session
    from pynput import keyboard, mouse

    ml = mouse.Listener(on_move=lambda x, y: guard.on_mouse_move(x, y), on_click=lambda *a: guard.on_click())
    kl = keyboard.Listener(on_press=lambda _k: guard.on_key())
    ml.start()
    kl.start()
    return [ml, kl]
