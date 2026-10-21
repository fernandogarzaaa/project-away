"""Grant admission race safety for Agent Link computer control (audit fix, 2026-10-20).

ComputerControl.request() used to check the current grant, block on the approval
prompt, and then assign self.grant with no reservation and no recheck. Two
concurrent approved requests overwrote each other (the overwritten grant was
never ended: no cancel, no receipt, no ended event), and a re-request by the
grant's own agent while paused silently discarded the paused session.

The fixed contract, exercised here with deterministic threaded tests:

* at most one approval prompt is in flight; a request that arrives while one is
  pending is refused at once and can ask again once it is answered;
* a live grant is never overwritten: the grant is installed only while the
  caller's admission reservation is still its own;
* Stop / revocation during the prompt cancels the reservation, so a late
  approval installs nothing;
* a re-request while the agent's own grant is paused refuses and points at
  resume_control; the paused session is kept, not discarded.

Every test in this file fails on the unfixed code (checked by running it
against the pre-fix control.py).
"""

from __future__ import annotations

import threading
from pathlib import Path
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


class FakeUIA:
    available = True

    def __init__(self, at: Any = None) -> None:
        self._at = at

    def focused(self) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        return self._at() if callable(self._at) else self._at

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None


class BlockingAsk:
    """The first approval prompt blocks until the test releases it; every later
    prompt is answered at once. Counts the prompts actually shown."""

    def __init__(self, first: str = "Allow", later: str = "Allow") -> None:
        self.first, self.later = first, later
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, _q: str, _o: Any, **kw: Any) -> dict[str, Any]:
        with self._lock:
            self.calls += 1
            n = self.calls
        if n == 1:
            self.entered.set()
            if not self.release.wait(timeout=20):
                raise AssertionError("the test never released the first prompt")
            return {"answer": self.first, "method": "click"}
        return {"answer": self.later, "method": "click"}


def _control(
    tmp_path: Path,
    ask: Any,
    *,
    clock: Clock | None = None,
    events: list | None = None,
    backend: Any = None,
) -> ComputerControl:
    hooks = ControlHooks(
        backend=backend or RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=FakeUIA(at=Element("Canvas", "PaneControl")),
    )
    hooks.cursor_pos = lambda: (7, 8)
    return ComputerControl(
        hooks,
        ask,
        ttl_s=1800,
        rate_per_s=10000.0,
        log_folder=tmp_path / "actions",
        clock=clock or Clock(),
        on_event=(lambda topic, data: events.append((topic, data))) if events is not None else None,
    )


def _request_in_thread(ctl: ComputerControl, agent_id: str, name: str) -> tuple[threading.Thread, dict[str, Any]]:
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = ctl.request(agent_id, name)
        except Exception as exc:  # the test asserts on the captured outcome
            box["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    return t, box


# ------------------------------------------------------------------ the race


def test_two_agents_requesting_at_once_exactly_one_gets_the_grant(tmp_path: Path) -> None:
    ask = BlockingAsk()
    events: list = []
    ctl = _control(tmp_path, ask, events=events)
    t, box = _request_in_thread(ctl, "agent1", "Agent1")
    assert ask.entered.wait(timeout=10)  # agent1's prompt is on screen
    # agent2 asks while the first prompt is still up: refused at once, no second prompt.
    with pytest.raises(ControlRefused):
        ctl.request("agent2", "Agent2")
    assert ask.calls == 1
    ask.release.set()  # the person allows agent1
    t.join(15)
    assert not t.is_alive()
    assert box["result"]["active"] is True
    assert ctl.grant is not None and ctl.grant.agent_id == "agent1"
    started = [d for topic, d in events if topic == "control.started"]
    assert len(started) == 1  # exactly one grant was ever started


def test_the_same_agent_cannot_stack_a_second_prompt_on_its_own_pending_request(tmp_path: Path) -> None:
    ask = BlockingAsk()
    ctl = _control(tmp_path, ask)
    t, box = _request_in_thread(ctl, "agent1", "Agent1")
    assert ask.entered.wait(timeout=10)
    with pytest.raises(ControlRefused):
        ctl.request("agent1", "Agent1")  # a retry must not open a second prompt
    assert ask.calls == 1
    ask.release.set()
    t.join(15)
    assert not t.is_alive()
    assert box["result"]["active"] is True


def test_a_request_refused_while_another_is_pending_can_ask_again_afterwards(tmp_path: Path) -> None:
    ask = BlockingAsk(first="Deny")  # the person denies the first prompt
    ctl = _control(tmp_path, ask)
    t, box = _request_in_thread(ctl, "agent1", "Agent1")
    assert ask.entered.wait(timeout=10)
    with pytest.raises(ControlRefused):
        ctl.request("agent2", "Agent2")  # refused: a prompt is already waiting
    ask.release.set()
    t.join(15)
    assert not t.is_alive()
    assert isinstance(box.get("error"), ControlRefused)  # agent1 was denied
    assert ctl.grant is None
    # The admission slot is free again: agent2's own request prompts and succeeds.
    st = ctl.request("agent2", "Agent2")
    assert st["active"] is True
    assert ctl.grant is not None and ctl.grant.agent_id == "agent2"
    assert ask.calls == 2


def test_no_actions_run_under_the_losing_side_of_a_race(tmp_path: Path) -> None:
    ask = BlockingAsk()
    backend = RecordingBackend()
    ctl = _control(tmp_path, ask, backend=backend)
    t, box = _request_in_thread(ctl, "agent1", "Agent1")
    assert ask.entered.wait(timeout=10)
    with pytest.raises(ControlRefused):
        ctl.request("agent2", "Agent2")
    ask.release.set()
    t.join(15)
    assert not t.is_alive()
    assert box["result"]["active"] is True
    # The loser holds no grant (live or abandoned): its actions refuse and reach no backend.
    with pytest.raises(ControlRefused):
        ctl.act("agent2", "move", x=5, y=5)
    assert backend.calls == []
    # The winner acts normally.
    ctl.act("agent1", "move", x=5, y=5)
    assert backend.calls != []


# ------------------------------------------------------- stop / revoke mid-prompt


def test_stop_during_the_approval_prompt_installs_nothing(tmp_path: Path) -> None:
    ask = BlockingAsk()
    events: list = []
    ctl = _control(tmp_path, ask, events=events)
    t, box = _request_in_thread(ctl, "agent1", "Agent1")
    assert ask.entered.wait(timeout=10)
    ctl.stop("you pressed Stop")  # Stop Everything while the prompt is up
    ask.release.set()  # the person then clicks Allow on the now-stale prompt
    t.join(15)
    assert not t.is_alive()
    assert isinstance(box.get("error"), ControlRefused)
    assert ctl.grant is None
    assert ctl.status() == {"active": False}
    assert [d for topic, d in events if topic == "control.started"] == []


def test_revoke_during_the_approval_prompt_installs_nothing(tmp_path: Path) -> None:
    ask = BlockingAsk()
    events: list = []
    ctl = _control(tmp_path, ask, events=events)
    t, box = _request_in_thread(ctl, "agent1", "Agent1")
    assert ask.entered.wait(timeout=10)
    ctl.revoke_agent("agent1")  # the agent disconnects while its prompt is up
    ask.release.set()
    t.join(15)
    assert not t.is_alive()
    assert isinstance(box.get("error"), ControlRefused)
    assert ctl.grant is None
    assert ctl.status() == {"active": False}
    assert [d for topic, d in events if topic == "control.started"] == []


# ------------------------------------------------------- re-request while paused


def test_rerequest_while_paused_keeps_the_paused_session(tmp_path: Path) -> None:
    clock = Clock()
    calls = {"n": 0}

    def ask(_q: str, _o: Any, **kw: Any) -> dict[str, Any]:
        calls["n"] += 1
        return {"answer": "Continue" if kw.get("title") == "Resume?" else "Allow", "method": "click"}

    events: list = []
    ctl = _control(tmp_path, ask, clock=clock, events=events)
    ctl.request("agent1", "Agent1")
    old = ctl.grant
    assert old is not None
    ctl.act("agent1", "move", x=10, y=10)
    clock.advance(2.0)  # past the approval grace and the agent's own input echo
    assert ctl.user_input("mouse") is True  # the person takes over: paused, not ended
    assert old.paused
    # Re-requesting must not prompt again and must not discard the paused session.
    with pytest.raises(ControlRefused, match="resume_control"):
        ctl.request("agent1", "Agent1")
    assert calls["n"] == 1  # no second prompt was shown
    assert ctl.grant is old and old.ended == "" and old.paused
    assert [d for topic, d in events if topic == "control.ended"] == []
    assert [d for topic, d in events if topic == "control.receipt"] == []
    # The session is still there to resume: same session id, spent budget intact.
    st = ctl.resume("agent1")
    assert st["session"] == old.id and st["steps_used"] == 1
