"""Apply gesture events to the OS mouse."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, Protocol

from away.hands.cursor import CursorMapper
from away.hands.gestures import GestureEvent
from away.hands.stabilize import ClickStabilizer


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

    def hscroll(self, dx: float) -> None:
        self.calls.append(("hscroll", (dx,)))

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
        self._hscroll_acc = 0.0

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

    def hscroll(self, dx: float) -> None:
        """Horizontal wheel notches (+ = right); fractions carry over like :meth:`scroll`."""
        self._hscroll_acc += dx
        steps = int(self._hscroll_acc)
        if steps:
            self._m.scroll(steps, 0)
            self._hscroll_acc -= steps

    def key(self, name: str) -> None:
        # Use the same guaranteed cleanup path as multi-key chords. A failure in
        # press() must not leave a modifier or ordinary key logically held down.
        self.hotkey([name])

    def click_button(self, button: str, count: int = 1) -> None:
        from pynput import mouse

        self._m.click(getattr(mouse.Button, button, mouse.Button.left), count)

    def hotkey(self, keys: list[str]) -> None:
        """Press a chord and always attempt to release every key touched.

        OS input APIs can fail partway through a chord (for example, when a
        desktop session loses access). Without cleanup, modifier keys can stay
        logically held and affect subsequent user input.
        """
        objs = [getattr(self._keys, k, k) for k in keys]
        attempted: list[Any] = []
        try:
            for key in objs:
                # Register before calling press(): the backend may raise after
                # the OS accepted the key-down event.
                attempted.append(key)
                self._k.press(key)
        finally:
            # Keep cleaning up after an individual release failure. Preserve an
            # original press error instead of replacing it with a cleanup error.
            active_error = sys.exc_info()[0] is not None
            release_error: BaseException | None = None
            for key in reversed(attempted):
                try:
                    self._k.release(key)
                except BaseException as exc:
                    if release_error is None:
                        release_error = exc
            if release_error is not None and not active_error:
                raise release_error

    def type_text(self, text: str) -> None:
        self._k.type(text)


# swipe -> key binding (browser/back-forward style by default)
DEFAULT_SWIPE_KEYS = {"swipe_left": "left", "swipe_right": "right"}


class MouseController:
    def __init__(
        self,
        backend: MouseBackend,
        mapper: CursorMapper,
        swipe_keys: dict[str, str] | None = None,
        stabilizer: ClickStabilizer | None = None,
    ) -> None:
        self.backend = backend
        self.mapper = mapper
        self.swipe_keys = DEFAULT_SWIPE_KEYS if swipe_keys is None else swipe_keys
        self.enabled = True
        self.button_down = False
        self.stabilizer = stabilizer
        # ``cursor:gaze``: returns the current gaze point on the desktop (None = not calibrated / no face)
        self.gaze_point: Callable[[], tuple[float, float] | None] | None = None

    # ------------------------------------------------------------ steady click
    def observe_pinch(self, t: float, ratio: float, pinch_on: float, pinched: bool, dragging: bool) -> None:
        st = self.stabilizer
        if st is None:
            return
        st.observe(t, ratio, pinch_on, pinched, dragging)
        if st.just_released:
            st.just_released = False
            self.mapper.reset()  # re-anchor: the next move continues from here, no jump

    def steady(self, t: float) -> None:
        """Before a pinch click/press: put the cursor back where it was just before the pinch began."""
        st = self.stabilizer
        if st is None or not self.enabled:
            return
        pos = st.click_position()
        if pos is None:
            return
        x, y = round(pos[0]), round(pos[1])
        if (x, y) != (round(self.mapper.pos[0]), round(self.mapper.pos[1])):
            self.mapper.set_position(pos[0], pos[1])
            self.backend.move(x, y)

    def handle(self, ev: GestureEvent) -> None:
        if ev.kind == "clutch":
            self.enabled = not bool(ev.data.get("paused"))
            if not self.enabled:
                self._release_if_down()
            return
        if ev.kind == "hand_lost":
            self.mapper.reset()
            if self.stabilizer is not None:
                self.stabilizer.reset()
            self._release_if_down()
            return
        if not self.enabled:
            return
        if ev.kind == "move":
            if self.stabilizer is not None and self.stabilizer.frozen:
                return  # pinch closing: hold still
            x, y = self.mapper.update(
                ev.t, float(ev.data["x"]), float(ev.data["y"]), str(ev.data.get("anchor", "index"))
            )
            self.backend.move(x, y)
            if self.stabilizer is not None:
                self.stabilizer.record(ev.t, self.mapper.pos)
        elif ev.kind in ("click", "double_click"):
            # double_click is the *second* tap: one more click makes the OS see a double click
            self.backend.click(1)
        elif ev.kind == "drag_start":
            # A backend call may raise after the OS accepted the button-down
            # event. Mark it as potentially held before pressing, then make a
            # best-effort release without masking the original failure.
            self.button_down = True
            try:
                self.backend.press()
            except BaseException:
                # The press may have reached the OS before raising. Keep the
                # state marked as potentially down if cleanup also fails, so a
                # later hand-loss, pause, or shutdown can retry the release.
                try:
                    self.backend.release()
                except BaseException:
                    self.button_down = True
                else:
                    self.button_down = False
                raise
        elif ev.kind == "drag_end":
            self._release_if_down()
        elif ev.kind == "scroll":
            self.backend.scroll(float(ev.data["dy"]))
        elif ev.kind in ("autoscroll", "palm_scroll"):
            dy, dx = float(ev.data.get("dy", 0.0)), float(ev.data.get("dx", 0.0))
            if dy:
                self.backend.scroll(dy)
            hscroll = getattr(self.backend, "hscroll", None)
            if dx and hscroll is not None:
                hscroll(dx)
        elif ev.kind in self.swipe_keys:
            self.backend.key(self.swipe_keys[ev.kind])

    def jump_monitor(self, arg: str) -> bool:
        """``cursor:`` action argument (``next_monitor`` | ``prev_monitor`` | ``primary_monitor`` |
        ``monitor N``): move the pointer to the centre of that monitor. False when there is none.
        ``center``: the centre of the current monitor; ``gaze``: the gaze point (v0.1.19). Both re-anchor."""
        if not self.enabled:
            return False
        if arg in ("center", "gaze"):
            to: tuple[int, int]
            if arg == "center":
                to = self.mapper.center()
            else:
                pt = self.gaze_point() if self.gaze_point is not None else None
                if pt is None:
                    return False
                to = self.mapper.warp(pt[0], pt[1])
            if self.stabilizer is not None:
                self.stabilizer.reset()
            self.backend.move(*to)
            return True
        target = {"next_monitor": "next", "prev_monitor": "prev", "primary_monitor": "primary"}.get(arg)
        if target is None and arg.startswith("monitor "):
            target = arg.split(" ", 1)[1]
        if target is None:
            return False
        # a held drag stays held: "next monitor" while dragging carries a window to the next screen
        pos = self.mapper.jump(target)
        if pos is None:
            return False
        if self.stabilizer is not None:
            self.stabilizer.reset()
        self.backend.move(*pos)
        return True

    def _release_if_down(self) -> None:
        if self.button_down:
            self.backend.release()
            self.button_down = False


def screen_size() -> tuple[int, int]:  # pragma: no cover - needs a display
    """Size of the primary monitor (kept for callers that only need one screen; the cursor uses
    :func:`away.control.monitors.enumerate_monitors`, which sees every monitor)."""
    from away.control.monitors import enumerate_monitors

    p = enumerate_monitors().primary
    return p.width, p.height
