"""Proposal #20: a toast shows when computer control ends (expiry, stop, deny, disconnect).

The 30-minute grant used to expire silently; every grant end now tells the user in
plain words via ``prompts.notify``. These tests FAIL on the unfixed code (no toast
emitted), proving the regression coverage.
"""

from __future__ import annotations

from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class UIA:
    available = True

    def focused(self) -> Element:
        return Element("Message", "EditControl", rect=(0, 0, 9, 9))


def _control(answer: str = "Allow", notify: Any = None, ttl_s: float = 60.0) -> tuple[ComputerControl, Clock]:
    clock = Clock()
    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=UIA(),
    )
    ctl = ComputerControl(
        hooks,
        lambda *_a, **_k: {"answer": answer},
        rate_per_s=50,
        ttl_s=ttl_s,
        clock=clock,
        notify=notify,
    )
    ctl.request("a1", "Claude")
    clock.advance(5)  # past the approval grace
    return ctl, clock


def _grant(notify: Any = None, ttl_s: float = 60.0) -> tuple[ComputerControl, Clock, list]:
    notes: list = []

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        notes.append((args, kwargs))
        return {"shown": True}

    ctl, clock = _control(notify=notify if notify is not None else capture, ttl_s=ttl_s)
    return ctl, clock, notes


def test_expiry_toast_says_time_ran_out() -> None:
    ctl, clock, notes = _grant()
    clock.advance(2000)  # past the 60 s grant
    assert ctl.status() == {"active": False}
    assert len(notes) == 1
    args, kwargs = notes[0]
    assert args[0] == "Claude's turn with your computer just ended (time ran out)."
    assert kwargs.get("level") == "info"


def test_stop_toast_says_you_stopped_it() -> None:
    ctl, _clock, notes = _grant()
    assert ctl.stop() is True
    assert len(notes) == 1
    assert notes[0][0][0] == "Claude's turn with your computer just ended (you stopped it)."


def test_resume_denied_toast_says_you_stopped_it() -> None:
    notes: list = []

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        notes.append((args, kwargs))
        return {"shown": True}

    ctl, clock = _control(notify=capture)
    assert ctl.user_input("mouse") is True  # pauses the grant
    ctl.ask = lambda *_a, **_k: {"answer": "Stop"}  # type: ignore[method-assign]
    with pytest.raises(ControlRefused):
        ctl.resume("a1")
    assert len(notes) == 1
    assert notes[0][0][0] == "Claude's turn with your computer just ended (you stopped it)."


def test_revoke_toast_says_disconnected() -> None:
    ctl, _clock, notes = _grant()
    ctl.revoke_agent("a1")
    assert len(notes) == 1
    assert notes[0][0][0] == "Claude's turn with your computer just ended (it was disconnected)."


def test_no_toast_when_there_was_no_grant() -> None:
    notes: list = []

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        notes.append((args, kwargs))
        return {"shown": True}

    clock = Clock()
    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=UIA(),
    )
    ctl = ComputerControl(hooks, lambda *_a, **_k: {"answer": "Allow"}, clock=clock, notify=capture)
    assert ctl.stop() is False  # nothing to end
    assert notes == []


def test_failing_notify_does_not_break_grant_teardown() -> None:
    def boom(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise RuntimeError("toast backend down")

    ctl, _clock, _notes = _grant(notify=boom)
    assert ctl.stop() is True  # no exception escapes
    assert ctl.grant is None
