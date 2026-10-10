"""Apply gesture events to the OS mouse."""

from __future__ import annotations

from typing import Any, Protocol

from away.hands.cursor import CursorMapper
from away.hands.gestures import GestureEvent


class MouseBackend(Protocol):
    def move(self, x: int, y: int) -> None: ...
    def click(self, count: int = 1) -> None: ...
    def press(self) -> None: ...
    def release(self) -> None: ...
    def scroll(self, dy: float) -> None: ...
    def key(self, name: str) -> None: ...


class RecordingBackend:
    """In-memory backend for tests and ``--dry-run``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def move(self, x: int, y: int) -> None:
        self.calls.append(("move", (x, y)))

    def click(self, count: int = 1) -> None:
        self.calls.append(("click", (count,)))

    def press(self) -> None:
        self.calls.append(("press", ()))

    def release(self) -> None:
        self.calls.append(("release", ()))

    def scroll(self, dy: float) -> None:
        self.calls.append(("scroll", (dy,)))

    def key(self, name: str) -> None:
        self.calls.append(("key", (name,)))

    def click_button(self, button: str, count: int = 1) -> None:
        self.calls.append(("click_button", (button, count)))

    def hotkey(self, keys: list[str]) -> None:
        self.calls.append(("hotkey", tuple(keys)))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", (text,)))


class PynputBackend:  # pragma: no cover - needs a desktop session
    def __init__(self) -> None:
        from pynput import keyboard, mouse

        self._m = mouse.Controller()
        self._k = keyboard.Controller()
        self._button = mouse.Button.left
        self._keys = keyboard.Key
        self._scroll_acc = 0.0

    def move(self, x: int, y: int) -> None:
        self._m.position = (x, y)

    def click(self, count: int = 1) -> None:
        self._m.click(self._button, count)

    def press(self) -> None:
        self._m.press(self._button)

    def release(self) -> None:
        self._m.release(self._button)

    def scroll(self, dy: float) -> None:
        self._scroll_acc += dy
        steps = int(self._scroll_acc)
        if steps:
            self._m.scroll(0, steps)
            self._scroll_acc -= steps

    def key(self, name: str) -> None:
        k = getattr(self._keys, name, name)
        self._k.press(k)
        self._k.release(k)

    def click_button(self, button: str, count: int = 1) -> None:
        from pynput import mouse

        self._m.click(getattr(mouse.Button, button, mouse.Button.left), count)

    def hotkey(self, keys: list[str]) -> None:
        objs = [getattr(self._keys, k, k) for k in keys]
        for k in objs:
            self._k.press(k)
        for k in reversed(objs):
            self._k.release(k)

    def type_text(self, text: str) -> None:
        self._k.type(text)


# swipe -> key binding (browser/back-forward style by default)
DEFAULT_SWIPE_KEYS = {"swipe_left": "left", "swipe_right": "right"}


class MouseController:
    def __init__(self, backend: MouseBackend, mapper: CursorMapper, swipe_keys: dict[str, str] | None = None) -> None:
        self.backend = backend
        self.mapper = mapper
        self.swipe_keys = DEFAULT_SWIPE_KEYS if swipe_keys is None else swipe_keys
        self.enabled = True
        self.button_down = False

    def handle(self, ev: GestureEvent) -> None:
        if ev.kind == "clutch":
            self.enabled = not bool(ev.data.get("paused"))
            if not self.enabled:
                self._release_if_down()
            return
        if ev.kind == "hand_lost":
            self.mapper.reset()
            self._release_if_down()
            return
        if not self.enabled:
            return
        if ev.kind == "move":
            x, y = self.mapper.update(
                ev.t, float(ev.data["x"]), float(ev.data["y"]), str(ev.data.get("anchor", "index"))
            )
            self.backend.move(x, y)
        elif ev.kind in ("click", "double_click"):
            # double_click is the *second* tap: one more click makes the OS see a double click
            self.backend.click(1)
        elif ev.kind == "drag_start":
            self.backend.press()
            self.button_down = True
        elif ev.kind == "drag_end":
            self._release_if_down()
        elif ev.kind == "scroll":
            self.backend.scroll(float(ev.data["dy"]))
        elif ev.kind in self.swipe_keys:
            self.backend.key(self.swipe_keys[ev.kind])

    def _release_if_down(self) -> None:
        if self.button_down:
            self.backend.release()
            self.button_down = False


def screen_size() -> tuple[int, int]:  # pragma: no cover - needs a display
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080
