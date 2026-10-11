"""Failure-path tests for gesture-driven mouse button lifecycle."""

import pytest

from away.control.mouse import MouseController
from away.hands.cursor import CursorMapper
from away.hands.gestures import GestureEvent


class _Backend:
    def __init__(self, *, fail_releases: int = 0, fail_press: bool = False) -> None:
        self.fail_releases = fail_releases
        self.fail_press = fail_press
        self.calls: list[str] = []

    def press(self) -> None:
        self.calls.append("press")
        if self.fail_press:
            raise RuntimeError("press failed")

    def release(self) -> None:
        self.calls.append("release")
        if self.fail_releases:
            self.fail_releases -= 1
            raise RuntimeError("release failed")

    def move(self, x: int, y: int) -> None:
        self.calls.append("move")

    def click(self, count: int = 1) -> None:
        self.calls.append("click")

    def scroll(self, dy: float) -> None:
        self.calls.append("scroll")

    def key(self, name: str) -> None:
        self.calls.append("key")


def _mouse(backend: _Backend) -> MouseController:
    return MouseController(backend, CursorMapper((100, 100)))


def test_hand_loss_releases_active_drag() -> None:
    backend = _Backend()
    mouse = _mouse(backend)
    mouse.handle(GestureEvent("drag_start", 0.0))
    mouse.handle(GestureEvent("hand_lost", 1.0))

    assert backend.calls == ["press", "release"]
    assert mouse.button_down is False


def test_pause_releases_active_drag_and_disables_output() -> None:
    backend = _Backend()
    mouse = _mouse(backend)
    mouse.handle(GestureEvent("drag_start", 0.0))
    mouse.handle(GestureEvent("clutch", 1.0, {"paused": True}))

    assert backend.calls == ["press", "release"]
    assert mouse.button_down is False
    assert mouse.enabled is False


def test_failed_release_keeps_button_marked_as_potentially_down_for_retry() -> None:
    backend = _Backend(fail_releases=1)
    mouse = _mouse(backend)
    mouse.button_down = True

    with pytest.raises(RuntimeError, match="release failed"):
        mouse.handle(GestureEvent("hand_lost", 1.0))

    assert mouse.button_down is True
    mouse.handle(GestureEvent("hand_lost", 2.0))
    assert backend.calls == ["release", "release"]
    assert mouse.button_down is False


def test_failed_drag_start_cleanup_remains_retryable() -> None:
    backend = _Backend(fail_press=True, fail_releases=1)
    mouse = _mouse(backend)

    with pytest.raises(RuntimeError, match="press failed"):
        mouse.handle(GestureEvent("drag_start", 0.0))

    assert backend.calls == ["press", "release"]
    assert mouse.button_down is True

    # A later lifecycle event retries cleanup rather than forgetting the
    # button may still be down after an ambiguous OS/backend failure.
    mouse.handle(GestureEvent("hand_lost", 1.0))
    assert backend.calls == ["press", "release", "release"]
    assert mouse.button_down is False
