"""Consent receipts (Project Away proposal #17): when an agent's control grant ends, the app shows a toast
plus one receipt dialog - "{name} did N things" - with the existing thumbnails and an "Undo all" button, and
the tray menu offers "Recent agent activity" for later review.

Every test here failed on the code before the fix (checked with `git stash push src/`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks, describe_action, plain_end_reason
from away.agentlink.desktop import Element, best_match
from away.agentlink.ui_helper import receipt_texts
from away.app import tray as traymod
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

    def __init__(self, focused: Element | None = None, at: Any = None, elements: Any = ()) -> None:
        self._focused, self._at, self._els = focused, at, list(elements)

    def focused(self) -> Element | None:
        return self._focused

    def element_at(self, x: int, y: int) -> Element | None:
        if callable(self._at):
            return self._at(x, y)  # type: ignore[no-any-return]
        return self._at

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return best_match(self._els, name, control_type)

    def tree(self, *a: Any) -> None:
        return None


TEXT_BOX = Element("Message", "EditControl", rect=(0, 0, 9, 9))


def _control(
    clock: Clock,
    answers: list[str] | None = None,
    uia: Any = None,
    backend: Any = None,
    secure: Any = False,
    platform: str = "linux",
) -> tuple[ComputerControl, Any, list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    backend = backend if backend is not None else RecordingBackend()
    asked: list[dict[str, Any]] = []
    queue = list(answers or ["Allow"])
    events: list[tuple[str, dict[str, Any]]] = []

    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        asked.append({"question": question, "options": options, **kw})
        return {"answer": queue.pop(0) if queue else None, "method": "click"}

    hooks = ControlHooks(
        backend=backend,
        platform=platform,
        secure_desktop=secure if callable(secure) else (lambda: secure),
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=uia if uia is not None else FakeUIA(focused=TEXT_BOX, at=Element("OK", "ButtonControl")),
    )
    ctl = ComputerControl(
        hooks,
        ask,
        ttl_s=1800,
        rate_per_s=50,
        clock=clock,
        on_event=lambda topic, data: events.append((topic, data)),
    )
    return ctl, backend, asked, events


def _receipt(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    got = [d for t, d in events if t == "control.receipt"]
    assert got, f"no control.receipt event; topics seen: {[t for t, _ in events]}"
    return got[-1]


# ------------------------------------------------------------- the receipt event


def test_receipt_fires_when_the_grant_stops() -> None:
    clock = Clock()
    ctl, _, _, events = _control(clock)
    ctl.request("a1", "Claude")
    ctl.act("a1", "click", x=100, y=200)
    ctl.act("a1", "scroll", dy=5)
    ctl.act("a1", "type_text", text="hello world")
    ctl.stop("you pressed Stop")
    r = _receipt(events)
    assert r["agent"] == "Claude"
    assert r["count"] == 3
    assert r["ended"] == "you pressed Stop"
    assert r["title"] == "Claude finished using your computer"
    assert r["headline"] == "Claude did 3 things."
    texts = [i["text"] for i in r["items"]]
    assert texts == ["Clicked on 'OK'", "Scrolled", "Typed 11 characters"]
    assert r["undoable"] is True
    assert all("thumb" in i for i in r["items"])


def test_receipt_fires_when_the_grant_expires() -> None:
    clock = Clock()
    ctl, _, _, events = _control(clock)
    ctl.request("a1", "Claude")
    ctl.act("a1", "click", x=10, y=10)
    clock.advance(1801)
    assert ctl.status()["active"] is False  # expiry ends the grant
    r = _receipt(events)
    assert r["count"] == 1
    assert r["reason"] == "expired"
    assert r["ended"] == "its time ran out"


def test_receipt_with_no_actions_still_reports() -> None:
    clock = Clock()
    ctl, _, _, events = _control(clock)
    ctl.request("a1", "Claude")
    ctl.stop("stopped")
    r = _receipt(events)
    assert r["count"] == 0
    assert r["items"] == []
    assert r["undoable"] is False
    assert r["headline"] == "Claude did not do anything this time."


def test_receipt_covers_only_the_ended_grant() -> None:
    clock = Clock()
    ctl, _, _, events = _control(clock, ["Allow", "Allow"])
    ctl.request("a1", "Claude")
    ctl.act("a1", "click", x=1, y=1)
    ctl.act("a1", "click", x=2, y=2)
    ctl.stop("stopped")
    first = _receipt(events)
    assert first["count"] == 2
    events.clear()
    ctl.request("a1", "Claude")  # a later grant: the old actions stay in the log
    assert len(ctl.log) == 2
    ctl.act("a1", "scroll", dy=5)
    ctl.stop("stopped")
    second = _receipt(events)
    assert second["count"] == 1
    assert [i["text"] for i in second["items"]] == ["Scrolled"]


def test_receipt_mentions_what_was_already_put_back() -> None:
    clock = Clock()
    ctl, _, _, events = _control(clock)
    ctl.request("a1", "Claude")
    ctl.act("a1", "scroll", dy=5)
    ctl.undo_last("a1")
    ctl.stop("stopped")
    r = _receipt(events)
    assert r["items"][0]["text"] == "Scrolled (you already put this back)"
    assert r["undoable"] is False


# ------------------------------------------------------------------ undo all


def test_undo_all_by_user_puts_back_everything_undoable() -> None:
    clock = Clock()
    ctl, _, _, _ = _control(clock)
    ctl.hooks.cursor_pos = lambda: (50, 60)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=100, y=200)  # undoable (cursor position known)
    ctl.act("a1", "scroll", dy=5)  # undoable
    ctl.act("a1", "type_text", text="hello")  # undoable
    ctl.act("a1", "click", x=5, y=5)  # clicks can't be put back
    ctl.stop("stopped")
    res = ctl.undo_all_by_user()
    assert res["count"] == 3
    assert res["undone"] == ["Typed 5 characters", "Scrolled", "Moved the mouse"]
    again = ctl.undo_all_by_user()
    assert again == {"undone": [], "count": 0}


def test_undo_all_by_user_needs_no_grant() -> None:
    clock = Clock()
    ctl, _, _, _ = _control(clock)
    res = ctl.undo_all_by_user()  # nothing logged yet
    assert res == {"undone": [], "count": 0}


# ------------------------------------------------------- plain words, no jargon


@pytest.mark.parametrize(
    ("action", "detail", "expected"),
    [
        ("move", "", "Moved the mouse"),
        ("click", "(100, 200) 'OK'", "Clicked on 'OK'"),
        ("click", "(100, 200)", "Clicked at (100, 200)"),
        ("double_click", "(10, 10) 'Photo.jpg'", "Double-clicked 'Photo.jpg'"),
        ("right_click", "(10, 10) 'Desktop'", "Right-clicked 'Desktop'"),
        ("drag", "(1, 2) -> (3, 4)", "Dragged something across the screen"),
        ("scroll", "dy=3 dx=0", "Scrolled"),
        ("type_text", "24 characters", "Typed 24 characters"),
        ("press_keys", "ctrl+s", "Pressed ctrl+s"),
        ("open_app", "Notepad", "Opened Notepad"),
        ("open_url", "https://example.com", "Opened https://example.com"),
    ],
)
def test_describe_action_uses_plain_words(action: str, detail: str, expected: str) -> None:
    got = describe_action(action, detail)
    assert got == expected
    for jargon in ("grant", "undoable", "dequeue", "throttle", "payload"):
        assert jargon not in got.lower()


def test_plain_end_reason() -> None:
    assert plain_end_reason("expired") == "its time ran out"
    assert plain_end_reason("stopped") == "you stopped it"
    assert plain_end_reason("the user chose to stop") == "you chose to stop it"
    assert plain_end_reason("the agent was disconnected") == "the agent disconnected"
    assert plain_end_reason("something custom") == "something custom"


def test_receipt_texts_pure() -> None:
    t = receipt_texts(
        {
            "title": "Claude finished using your computer",
            "headline": "Claude did 2 things.",
            "ended": "its time ran out",
            "items": [{"text": "Clicked on 'OK'"}, {"text": "Typed 11 characters"}],
            "undoable": True,
        }
    )
    assert t["title"] == "Claude finished using your computer"
    assert t["headline"] == "Claude did 2 things."
    assert t["ended"] == "its time ran out"
    assert t["items"] == ["Clicked on 'OK'", "Typed 11 characters"]
    assert t["undoable"] is True


# ------------------------------------------------------------- tray menu entry


def test_tray_recent_activity_opens_the_agent_link_tab(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[list[str]] = []

    class FakePopen:
        def __init__(self, argv: list[str], **kw: Any) -> None:
            seen.append(argv)

    monkeypatch.setattr(traymod.subprocess, "Popen", FakePopen)
    app = type("App", (), {"cfg": type("Cfg", (), {})()})()
    ctl = traymod.TrayController(app)
    ctl.open_recent_activity()
    assert seen, "the settings window was not opened"
    argv = seen[0]
    assert "--tab" in argv
    assert argv[argv.index("--tab") + 1] == "Agent Link"
