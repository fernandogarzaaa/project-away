"""Failure-path tests for reliable release of synthetic keyboard input."""

import pytest

from away.control.mouse import MouseController, PynputBackend
from away.hands.cursor import CursorMapper
from away.hands.gestures import GestureEvent


class _KeyNamespace:
    def __getattr__(self, name: str) -> str:
        return name


class _FlakyKeyboard:
    def __init__(self, *, fail_press: str | None = None, fail_release: str | None = None) -> None:
        self.fail_press = fail_press
        self.fail_release = fail_release
        self.calls: list[tuple[str, str]] = []

    def press(self, key: str) -> None:
        self.calls.append(("press", key))
        if key == self.fail_press:
            raise RuntimeError(f"press failed: {key}")

    def release(self, key: str) -> None:
        self.calls.append(("release", key))
        if key == self.fail_release:
            raise RuntimeError(f"release failed: {key}")


def _backend(keyboard: _FlakyKeyboard) -> PynputBackend:
    backend = object.__new__(PynputBackend)
    backend._k = keyboard
    backend._keys = _KeyNamespace()
    return backend


def test_hotkey_releases_all_attempted_keys_when_press_fails() -> None:
    keyboard = _FlakyKeyboard(fail_press="x")
    backend = _backend(keyboard)

    with pytest.raises(RuntimeError, match="press failed: x"):
        backend.hotkey(["ctrl", "x"])

    assert keyboard.calls == [
        ("press", "ctrl"),
        ("press", "x"),
        ("release", "x"),
        ("release", "ctrl"),
    ]


def test_hotkey_continues_releasing_after_one_release_fails() -> None:
    keyboard = _FlakyKeyboard(fail_release="x")
    backend = _backend(keyboard)

    with pytest.raises(RuntimeError, match="release failed: x"):
        backend.hotkey(["ctrl", "x"])

    assert keyboard.calls == [
        ("press", "ctrl"),
        ("press", "x"),
        ("release", "x"),
        ("release", "ctrl"),
    ]


def test_original_press_error_is_not_masked_by_release_error() -> None:
    keyboard = _FlakyKeyboard(fail_press="x", fail_release="x")
    backend = _backend(keyboard)

    with pytest.raises(RuntimeError, match="press failed: x"):
        backend.hotkey(["ctrl", "x"])

    assert keyboard.calls[-2:] == [("release", "x"), ("release", "ctrl")]


def test_single_key_uses_the_same_cleanup_path() -> None:
    keyboard = _FlakyKeyboard(fail_press="enter")
    backend = _backend(keyboard)

    with pytest.raises(RuntimeError, match="press failed: enter"):
        backend.key("enter")

    assert keyboard.calls == [("press", "enter"), ("release", "enter")]


class _FlakyMouse:
    def __init__(self, *, fail_press: bool = False, fail_release: bool = False) -> None:
        self.fail_press = fail_press
        self.fail_release = fail_release
        self.calls: list[str] = []

    def press(self) -> None:
        self.calls.append("press")
        if self.fail_press:
            raise RuntimeError("mouse press failed")

    def release(self) -> None:
        self.calls.append("release")
        if self.fail_release:
            raise RuntimeError("mouse release failed")


def test_drag_start_attempts_release_when_mouse_press_fails() -> None:
    backend = _FlakyMouse(fail_press=True)
    mouse = MouseController(backend, CursorMapper((100, 100)))

    with pytest.raises(RuntimeError, match="mouse press failed"):
        mouse.handle(GestureEvent("drag_start", 0.0, {}))

    assert backend.calls == ["press", "release"]
    assert mouse.button_down is False


def test_drag_start_preserves_press_error_if_cleanup_also_fails() -> None:
    backend = _FlakyMouse(fail_press=True, fail_release=True)
    mouse = MouseController(backend, CursorMapper((100, 100)))

    with pytest.raises(RuntimeError, match="mouse press failed"):
        mouse.handle(GestureEvent("drag_start", 0.0, {}))

    assert backend.calls == ["press", "release"]
    assert mouse.button_down is True
