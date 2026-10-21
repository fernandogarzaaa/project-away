"""open_app keystroke safety (audit Phase 4 lane B, item 1.6).

Before the fix, ``_open_app`` pressed the launcher key, typed the app name and pressed Enter on fixed
sleeps, and only afterwards checked whether the app appeared. If the launcher (Start menu/search or
Spotlight) never opened, the name and Enter landed in whatever window had focus: a chat window (sending
the name as a message), a document, or a terminal (where name + Enter is a command line). The result
carried an honest warning, but the keystrokes were already delivered.

Now the launcher itself must be seen in the foreground before anything is typed, focus is re-checked
immediately before typing and again before Enter, and an unverifiable launcher refuses without typing
a single character. The post-launch "could not confirm the app opened" warning is unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element, best_match
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeUIA:
    available = True

    def __init__(self, focused: Any = None) -> None:
        self._focused = focused if focused is not None else Element("Message", "EditControl", rect=(0, 0, 9, 9))

    def focused(self) -> Any:
        return self._focused

    def element_at(self, x: int, y: int) -> Any:
        return Element("Canvas", "PaneControl")

    def find(self, name: str, control_type: str | None = None) -> Any:
        return best_match([], name, control_type)

    def tree(self, *a: Any) -> None:
        return None


CHAT = {"title": "Chat with Sam", "process": "chat.exe", "class": "", "rect": (0, 0, 9, 9)}
SEARCH = {"title": "Search", "process": "SearchHost.exe", "class": "", "rect": (0, 0, 9, 9)}
START = {"title": "Start", "process": "StartMenuExperienceHost.exe", "class": "", "rect": (0, 0, 9, 9)}
NOTEPAD = {"title": "Untitled - Notepad", "process": "notepad.exe", "class": "", "rect": (0, 0, 9, 9)}


def _control(clock: Clock, platform: str, foreground: Any) -> tuple[ComputerControl, RecordingBackend]:
    backend = RecordingBackend()

    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        return {"answer": "Allow", "method": "click"}

    hooks = ControlHooks(
        backend=backend,
        platform=platform,
        secure_desktop=lambda: False,
        foreground=foreground,
        grab=lambda: None,
        sleep=lambda s: clock.advance(s),
        uia=FakeUIA(),
    )
    ctl = ComputerControl(hooks, ask, ttl_s=1800, rate_per_s=1000, clock=clock)
    ctl.request("a1", "Claude")
    return ctl, backend


def _typed(backend: RecordingBackend) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in backend.calls if c[0] == "type_text"]


def test_launcher_never_opens_refuses_and_types_nothing() -> None:
    clock = Clock()
    ctl, backend = _control(clock, "win32", lambda: CHAT)  # the foreground stays a chat window
    with pytest.raises(ControlRefused, match=r"launcher|Start"):
        ctl.act("a1", "open_app", name="Notepad")
    assert _typed(backend) == [], f"the app name was typed into the chat window: {backend.calls}"
    assert ("key", ("enter",)) not in backend.calls, f"Enter was delivered too: {backend.calls}"
    assert [c for c in backend.calls if c[0] == "key"] == [("key", ("cmd",))]  # only the launcher key itself


def test_launcher_opens_slowly_typing_waits_until_it_is_verified() -> None:
    clock = Clock()
    backend_holder: dict[str, Any] = {}
    seen: dict[str, Any] = {"n": 0, "calls_at_verify": None}

    def fg() -> dict[str, Any]:
        seen["n"] += 1
        if seen["n"] <= 2:
            return CHAT  # the launcher is not up yet
        if seen["n"] <= 5:
            if seen["calls_at_verify"] is None:
                seen["calls_at_verify"] = list(backend_holder["backend"].calls)
            return SEARCH
        return NOTEPAD

    ctl, backend = _control(clock, "win32", fg)
    backend_holder["backend"] = backend
    res = ctl.act("a1", "open_app", name="Notepad")
    assert res["ok"] is True and "warning" not in res
    # when the launcher was first seen, nothing but the launcher key had been sent
    assert seen["calls_at_verify"] == [("key", ("cmd",))]
    assert backend.calls == [
        ("key", ("cmd",)),
        ("type_text", ("Notepad",)),
        ("key", ("enter",)),
    ]


def test_focus_taken_after_launcher_opens_refuses_before_typing() -> None:
    clock = Clock()
    seen = {"n": 0}

    def fg() -> dict[str, Any]:
        seen["n"] += 1
        return SEARCH if seen["n"] == 1 else CHAT  # verified once, then another window takes focus

    ctl, backend = _control(clock, "win32", fg)
    with pytest.raises(ControlRefused, match="focus"):
        ctl.act("a1", "open_app", name="Notepad")
    assert _typed(backend) == [], f"typed after focus was taken: {backend.calls}"
    assert ("key", ("enter",)) not in backend.calls


def test_focus_taken_before_enter_refuses_enter() -> None:
    clock = Clock()
    seen = {"n": 0}

    def fg() -> dict[str, Any]:
        seen["n"] += 1
        return SEARCH if seen["n"] <= 2 else CHAT  # launcher up for the poll and the pre-type check only

    ctl, backend = _control(clock, "win32", fg)
    with pytest.raises(ControlRefused, match="focus"):
        ctl.act("a1", "open_app", name="Notepad")
    assert _typed(backend) == [("type_text", ("Notepad",))]  # the name went into the launcher's search box
    assert ("key", ("enter",)) not in backend.calls, f"Enter went to the focused window: {backend.calls}"


def test_happy_path_types_after_verification_and_confirms_the_app() -> None:
    clock = Clock()
    seen = {"n": 0}

    def fg() -> dict[str, Any]:
        seen["n"] += 1
        return START if seen["n"] <= 3 else NOTEPAD  # the Start menu itself counts as the launcher

    ctl, backend = _control(clock, "win32", fg)
    res = ctl.act("a1", "open_app", name="Notepad")
    assert res["ok"] is True and "warning" not in res
    assert backend.calls == [
        ("key", ("cmd",)),
        ("type_text", ("Notepad",)),
        ("key", ("enter",)),
    ]
    assert ctl.log and ctl.log[-1].result == "done"


def test_spotlight_never_opens_refuses_and_types_nothing() -> None:
    clock = Clock()
    messages = {"title": "Messages", "process": "Messages", "class": "", "rect": (0, 0, 9, 9)}
    ctl, backend = _control(clock, "darwin", lambda: messages)
    with pytest.raises(ControlRefused, match=r"launcher|Spotlight"):
        ctl.act("a1", "open_app", name="TextEdit")
    assert _typed(backend) == [], f"the app name was typed into Messages: {backend.calls}"
    assert ("key", ("enter",)) not in backend.calls
    assert [c for c in backend.calls if c[0] in ("key", "hotkey")] == [("hotkey", ("cmd", "space"))]


def test_spotlight_opens_and_the_app_appears() -> None:
    clock = Clock()
    seen = {"n": 0}

    def fg() -> dict[str, Any]:
        seen["n"] += 1
        if seen["n"] <= 3:
            return {"title": "", "process": "Spotlight", "class": "", "rect": (0, 0, 9, 9)}
        return {"title": "Untitled", "process": "TextEdit.app", "class": "", "rect": (0, 0, 9, 9)}

    ctl, backend = _control(clock, "darwin", fg)
    res = ctl.act("a1", "open_app", name="TextEdit")
    assert res["ok"] is True and "warning" not in res
    assert backend.calls == [
        ("hotkey", ("cmd", "space")),
        ("type_text", ("TextEdit",)),
        ("key", ("enter",)),
    ]


def test_launcher_matching_uses_launcher_identity_only() -> None:
    m = ComputerControl._launcher_matches
    assert m(SEARCH, "win32") and m(START, "win32")
    assert m({"title": "Search", "process": "SearchUI.exe", "class": ""}, "win32")
    # Windows 10: Start/Search hosted inside explorer.exe, told apart by class + title
    assert m({"title": "Start", "process": "explorer.exe", "class": "Windows.UI.Core.CoreWindow"}, "win32")
    assert not m({"title": "", "process": "explorer.exe", "class": "Shell_TrayWnd"}, "win32")  # the taskbar
    assert not m(CHAT, "win32") and not m(NOTEPAD, "win32") and not m({}, "win32")
    assert m({"title": "", "process": "Spotlight", "class": ""}, "darwin")
    assert not m({"title": "Finder", "process": "Finder", "class": ""}, "darwin")
    assert not m(SEARCH, "darwin")  # a Windows search host is not Spotlight
