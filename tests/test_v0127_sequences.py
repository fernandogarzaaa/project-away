"""v0.1.27+ action sequences: ``control(action="sequence", steps=[...])`` runs several computer actions in one
MCP call: one grant check, one log entry with sub-step detail, and it stops at the first refusal, cancel or
takeover. Every sub-step still runs through ``ComputerControl._act``, so the risky/unknown "Are you sure?"
prompts, the password-field and terminal refusals and the per-step engine audit apply unchanged - a batch must
not become a rubber stamp (mid-batch interception is tested below).

Each test here FAILS on the unfixed code (``ComputerControl.act_sequence`` does not exist there).
"""

from __future__ import annotations

import pytest

from away.agentlink.control import (
    ACTIONS,
    SEQUENCE_MAX_STEPS,
    ComputerControl,
    ControlHooks,
    ControlRefused,
)
from away.agentlink.desktop import Element, best_match
from away.agentlink.groups import resolve
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _control(clock: Clock, answers: list[str] | None = None, uia=None, fg=None, secure=False, platform="linux"):
    backend = RecordingBackend()
    asked: list[dict] = []
    queue = list(answers or ["Allow"])

    def ask(question: str, options: list[str], **kw):
        asked.append({"question": question, "options": options, **kw})
        return {"answer": queue.pop(0) if queue else None, "method": "click"}

    allowed: set[tuple[str, str]] = set()
    hooks = ControlHooks(
        backend=backend,
        platform=platform,
        secure_desktop=lambda: secure,
        foreground=lambda: fg,
        grab=lambda: None,
        sleep=lambda _s: None,
        **({"uia": uia} if uia is not None else {}),
    )
    ctl = ComputerControl(
        hooks,
        ask,
        ttl_s=1800,
        rate_per_s=4,
        clock=clock,
        always_allowed=lambda a, c: (a, c) in allowed,
        allow_always=lambda a, c: allowed.add((a, c)),
    )
    return ctl, backend, asked


class FakeUIA:
    available = True

    def __init__(self, focused: Element | None = None, at: Element | None = None, elements=()) -> None:
        self._focused, self._at, self._els = focused, at, list(elements)

    def focused(self):
        return self._focused

    def element_at(self, x, y):
        return self._at

    def find(self, name, control_type=None):
        return best_match(self._els, name, control_type)

    def tree(self, *a):
        return None


def _grant(ctl: ComputerControl, asked: list) -> None:
    """One approval: the sequence itself must not ask again for harmless steps."""
    ctl.request("a1", "Claude")
    assert len(asked) == 1 and asked[0]["kind"] == "approve"


def test_sequence_runs_steps_in_order_with_one_log_entry() -> None:
    clock = Clock()
    canvas = FakeUIA(at=Element("Canvas", "PaneControl", rect=(0, 0, 99, 99)))  # a named target: no questions
    ctl, backend, asked = _control(clock, ["Allow"], uia=canvas)
    _grant(ctl, asked)
    res = ctl.act_sequence(
        "a1",
        [
            {"action": "move", "x": 10, "y": 20},
            {"action": "click", "x": 10, "y": 20},
            {"action": "scroll", "dy": -3},
        ],
    )
    assert res["ok"] and res["action"] == "sequence"
    assert backend.calls == [("move", (10, 20)), ("move", (10, 20)), ("click", (1,)), ("scroll", (-3.0,))]
    assert [r["action"] for r in res["steps"]] == ["move", "click", "scroll"]
    entries = ctl.actions()
    assert len(entries) == 1  # one log entry for the whole sequence
    assert entries[0]["action"] == "sequence"
    assert entries[0]["result"] == "done"
    assert (
        "1. move" in entries[0]["detail"] and "2. click" in entries[0]["detail"] and "3. scroll" in entries[0]["detail"]
    )
    assert len(asked) == 1  # the grant approval was the only prompt: one grant check


def test_sequence_needs_a_grant_like_single_actions() -> None:
    clock = Clock()
    ctl, backend, _asked = _control(clock, ["Allow"])
    with pytest.raises(ControlRefused, match="request_control"):
        ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}])
    assert backend.calls == []


def test_sequence_rejects_bad_shape_up_front() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow"])
    ctl.request("a1", "Claude")
    for bad in ([], None, "move", [{"x": 1}]):
        with pytest.raises(ControlRefused):
            ctl.act_sequence("a1", bad)  # type: ignore[arg-type]
    with pytest.raises(ControlRefused, match="unknown action"):
        ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}, {"action": "teleport"}])
    with pytest.raises(ControlRefused, match="must be an object"):
        ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}, "click"])
    with pytest.raises(ControlRefused, match="limited to"):
        ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}] * (SEQUENCE_MAX_STEPS + 1))
    assert backend.calls == []  # nothing ran: validation happens before the grant is even used


def test_sequence_stops_at_first_refusal_and_later_steps_never_run() -> None:
    """Mid-batch interception: a risky sub-step asks the person one at a time, like a single action would; a
    "Deny" stops the batch and the steps after it never run."""
    clock = Clock()
    canvas = FakeUIA(at=Element("Canvas", "PaneControl", rect=(0, 0, 99, 99)))
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=canvas)  # grant ok, risky click denied
    _grant(ctl, asked)
    evil = FakeUIA(at=Element("Delete", "ButtonControl", rect=(0, 0, 99, 99)))
    ctl.hooks.uia = evil
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act_sequence(
            "a1",
            [
                {"action": "move", "x": 10, "y": 20},
                {"action": "click", "x": 10, "y": 20},  # asks, denied
                {"action": "move", "x": 99, "y": 99},  # never runs
            ],
        )
    assert [c for c in backend.calls if c[0] == "click"] == []
    assert backend.calls.count(("move", (99, 99))) == 0
    assert [q for q in asked if q.get("kind") == "confirm"], "the risky sub-step must ask, mid-batch"
    entries = ctl.actions()
    assert not entries or all(e["action"] != "sequence" or e["result"] != "done" for e in entries)


def test_sequence_password_refusal_stops_the_batch() -> None:
    clock = Clock()
    pwd = FakeUIA(focused=Element("Password", "EditControl", is_password=True, rect=(0, 0, 10, 10)))
    ctl, backend, asked = _control(clock, ["Allow"], uia=pwd)
    _grant(ctl, asked)
    with pytest.raises(ControlRefused, match="password"):
        ctl.act_sequence("a1", [{"action": "type_text", "text": "secret"}, {"action": "press_keys", "keys": "enter"}])
    assert [c for c in backend.calls if c[0] == "type_text"] == []
    assert [c for c in backend.calls if c[0] == "key"] == []  # the Enter after it never ran


def test_sequence_stops_when_the_grant_is_cancelled() -> None:
    clock = Clock()
    canvas = FakeUIA(at=Element("Canvas", "PaneControl", rect=(0, 0, 99, 99)))
    ctl, backend, asked = _control(clock, ["Allow"], uia=canvas)
    _grant(ctl, asked)
    assert ctl.grant is not None
    ctl.grant.cancel.set()  # e.g. a revoke, a global stop, or a takeover mid-batch
    with pytest.raises(ControlRefused, match="stopped part-way"):
        ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}])
    assert backend.calls == []
    entries = ctl.actions()
    assert len(entries) == 1 and entries[0]["result"] == "stopped" and "0 of 1 steps" in entries[0]["detail"]


def test_sequence_sub_steps_still_write_engine_audit_entries() -> None:
    """The engine's audit trail is per sub-step (not batched), while the action log has one sequence entry."""
    clock = Clock()
    canvas = FakeUIA(at=Element("Canvas", "PaneControl", rect=(0, 0, 99, 99)))
    ctl, backend, asked = _control(clock, ["Allow"], uia=canvas)
    _grant(ctl, asked)
    ctl.act_sequence("a1", [{"action": "move", "x": 10, "y": 20}, {"action": "click", "x": 10, "y": 20}])
    audit = [e for e in ctl.engine.recent(10) if e["actor"] == "Claude"]
    forward = list(reversed([e["action"] for e in audit]))  # recent() returns newest first
    # unchanged per-step audit logging, plus the one sequence-level entry the log entry writes
    assert forward == ["move", "click", "sequence"]
    assert len(ctl.actions()) == 1  # but one user-facing log entry


def test_control_group_resolves_sequence_to_act_sequence() -> None:
    steps = [{"action": "move", "x": 1, "y": 2}]
    method, args = resolve("control", {"action": "sequence", "steps": steps})
    assert method == "act_sequence" and args == {"steps": steps}
    with pytest.raises(ValueError, match="action must be one of"):
        resolve("control", {"action": "teleport"})


def test_sequence_action_names_cover_every_engine_action() -> None:
    assert set(ACTIONS) == {
        "move",
        "click",
        "double_click",
        "right_click",
        "drag",
        "scroll",
        "type_text",
        "press_keys",
        "open_app",
        "open_url",
    }
