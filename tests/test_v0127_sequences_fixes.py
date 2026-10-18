"""CodeRabbit follow-ups on proposal #1 (action sequences), all verified against the implementation:

- the per-action rate limit applies to every sub-step, not just the grant-time check;
- a mid-batch refusal or cancel records one combined log entry (no silent gap);
- a finished sequence keeps one combined undo record, reversed by ``undo_last``;
- the MCP bridge timeout covers the permitted batch (each risky sub-step can wait out a 60 s prompt).
"""

from __future__ import annotations

import pytest

from away.agentlink.bridge import _timeout_for
from away.agentlink.control import ControlRefused
from away.agentlink.desktop import Element
from test_v0127_sequences import Clock, FakeUIA, _control, _grant


def _canvas_control(clock, answers=None):
    canvas = FakeUIA(at=Element("Canvas", "PaneControl", rect=(0, 0, 99, 99)))  # named target: no questions
    return _control(clock, answers, uia=canvas)


def test_sequence_applies_rate_limit_per_sub_step() -> None:
    """One token per sub-step, like one per single action: a 2-token bucket allows the grant check plus one
    sub-step, and the second sub-step is refused - with a combined 'refused' log entry, not a silent gap."""
    clock = Clock()
    ctl, backend, asked = _canvas_control(clock, ["Allow"])
    _grant(ctl, asked)
    ctl.engine.set_agent_rate(1, clock)  # fresh bucket: 2 tokens
    with pytest.raises(ControlRefused, match="too many actions"):
        ctl.act_sequence(
            "a1",
            [
                {"action": "move", "x": 10, "y": 20},
                {"action": "move", "x": 30, "y": 40},
            ],
        )
    assert backend.calls == [("move", (10, 20))]  # only the first sub-step ran
    entries = ctl.actions()
    assert len(entries) == 1
    assert entries[0]["action"] == "sequence" and entries[0]["result"] == "refused"
    assert "1 of 2 steps" in entries[0]["detail"] and "1. move" in entries[0]["detail"]


def test_sequence_refusal_logs_one_combined_entry() -> None:
    """A password-field refusal on step 2 still records the sequence: what ran is in the action log."""
    clock = Clock()
    pwd = FakeUIA(focused=Element("Password", "EditControl", is_password=True, rect=(0, 0, 10, 10)))
    ctl, backend, asked = _control(clock, ["Allow"], uia=pwd)
    _grant(ctl, asked)
    with pytest.raises(ControlRefused, match="password"):
        ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}, {"action": "type_text", "text": "secret"}])
    assert [c for c in backend.calls if c[0] == "type_text"] == []
    entries = ctl.actions()
    assert len(entries) == 1  # one combined entry, not zero and not one per step
    assert entries[0]["action"] == "sequence" and entries[0]["result"] == "refused"
    assert "refused after 1 of 2 steps" in entries[0]["detail"]
    assert "1. move" in entries[0]["detail"]


def test_sequence_cancel_after_first_step_records_completed_detail() -> None:
    """The person takes over right after the first step lands: the 'stopped' entry names what ran."""
    clock = Clock()
    ctl, backend, asked = _canvas_control(clock, ["Allow"])
    _grant(ctl, asked)
    real_click = backend.click

    def click_and_takeover(*a):
        real_click(*a)
        assert ctl.grant is not None
        ctl.grant.cancel.set()  # the person grabs the mouse the moment the click lands

    backend.click = click_and_takeover  # type: ignore[method-assign]
    with pytest.raises(ControlRefused, match="stopped part-way"):
        ctl.act_sequence(
            "a1",
            [
                {"action": "click", "x": 10, "y": 20},  # runs
                {"action": "move", "x": 99, "y": 99},  # never runs: the batch was stopped
            ],
        )
    assert backend.calls.count(("move", (99, 99))) == 0
    entries = ctl.actions()
    assert len(entries) == 1
    assert entries[0]["action"] == "sequence" and entries[0]["result"] == "stopped"
    assert "stopped part-way after 1 of 2 steps" in entries[0]["detail"]
    assert "1. click" in entries[0]["detail"]


def test_sequence_undo_reverses_all_undoable_steps() -> None:
    """undo_last on a finished sequence reverses every undoable sub-step (newest first), and the sequence entry
    - not some older unrelated action - is what gets undone."""
    clock = Clock()
    ctl, backend, asked = _canvas_control(clock, ["Allow"])
    ctl.hooks.cursor_pos = lambda: (1, 2)
    _grant(ctl, asked)
    ctl.act_sequence("a1", [{"action": "move", "x": 300, "y": 300}, {"action": "scroll", "dy": -3}])
    entries = ctl.actions()
    assert len(entries) == 1 and entries[0].get("undoable")
    n = len(backend.calls)
    res = ctl.undo_last("a1")
    assert res["undone"] == "sequence"
    undone = backend.calls[n:]
    assert undone == [("scroll", (3.0,)), ("move", (1, 2))]  # newest sub-step undone first
    with pytest.raises(ControlRefused, match="nothing"):
        ctl.undo_last("a1")  # the sequence entry is spent; nothing older is silently picked instead


def test_bridge_sequence_timeout_covers_the_permitted_batch() -> None:
    """Each risky sub-step can wait out a 60 s 'Are you sure?' prompt, so the bridge waits 120 s plus 70 s per
    step instead of losing the result while the app keeps executing."""
    assert _timeout_for("act_sequence", {"steps": [{}]}) == 190.0
    assert _timeout_for("act_sequence", {"steps": [{}] * 25}) == 120.0 + 70.0 * 25
    assert _timeout_for("act", {}) == 90.0
    assert _timeout_for("ask_user", {}) == 330.0
