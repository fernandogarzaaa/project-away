"""Audit-trail integrity fixes (2026-10-20 audit, findings E1-E4).

E1: a sequence spent one budget step per sub-step but filed one combined trail row, so the session
summary disagreed with ``steps_used`` and sub-steps had no replayable evidence of their own.
E2: an undo left no durable status transition linked to the original action (a user undo and a
failed undo appended nothing at all; the persisted step read "done" forever).
E3: the persisted ``session-<id>.jsonl`` trails were write-only; a restart lost the Settings replay.
E4: ``actions()`` / ``session_trail()`` sliced with ``[-limit:]`` unvalidated, so limit=0 returned
everything and a negative limit returned all but a few.

Every test here fails on the unfixed code except the trail-off guard, which pins behavior the fix
must not change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from away.agentlink.control import (
    MAX_SESSIONS,
    ComputerControl,
    ControlHooks,
    ControlRefused,
)
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

    def __init__(self, focused: Element | None = None, at: Any = None) -> None:
        self._focused, self._at = focused, at

    def focused(self) -> Element | None:
        return self._focused

    def element_at(self, x: int, y: int) -> Element | None:
        return self._at() if callable(self._at) else self._at

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None


def _ask_allow(_q: str, _o: Any, **kw: Any) -> dict[str, Any]:
    return {"answer": "Continue" if kw.get("title") == "Resume?" else "Allow", "method": "click"}


def _ask_deny_confirm(_q: str, _o: Any, **kw: Any) -> dict[str, Any]:
    if kw.get("kind") == "confirm":
        return {"answer": "Deny", "method": "click"}
    return _ask_allow(_q, _o, **kw)


def _control(
    clock: Clock,
    tmp_path: Path,
    *,
    backend: RecordingBackend | None = None,
    ask: Any = None,
    audit_trail: bool = True,
    grab: Any = None,
    step_budget: int = 40,
) -> ComputerControl:
    hooks = ControlHooks(
        backend=backend or RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=grab if grab is not None else (lambda: None),  # no real screenshots unless a test asks
        sleep=lambda _s: None,
        uia=FakeUIA(at=Element("Canvas", "PaneControl")),
    )
    hooks.cursor_pos = lambda: (7, 8)  # so "move" is undoable in tests
    return ComputerControl(
        hooks,
        ask or _ask_allow,
        ttl_s=1800,
        rate_per_s=10000.0,
        log_folder=tmp_path / "actions",
        clock=clock,
        step_budget=step_budget,
        audit_trail=audit_trail,
    )


def _events(tmp_path: Path, session_id: str) -> list[dict[str, Any]]:
    path = tmp_path / "actions" / f"session-{session_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _session_id(ctl: ComputerControl) -> str:
    assert ctl.grant is not None
    return ctl.grant.id


class CancelOnFirstMove(RecordingBackend):
    """Sets the live grant's cancel event during the first move, like a takeover mid-sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.ctl: ComputerControl | None = None
        self._fired = False

    def move(self, x: int, y: int) -> None:
        super().move(x, y)
        if not self._fired and self.ctl is not None and self.ctl.grant is not None:
            self._fired = True
            self.ctl.grant.cancel.set()


class FailOnMove(RecordingBackend):
    """Raises on the Nth move call (1-based) once ``fail_on`` is armed."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_on: int | None = None
        self._moves = 0

    def move(self, x: int, y: int) -> None:
        self._moves += 1
        if self.fail_on is not None and self._moves == self.fail_on:
            raise RuntimeError("the mouse unplugged")
        super().move(x, y)


# ---------------------------------------------------------------------- E1


def test_sequence_files_one_trail_event_per_substep(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    res = ctl.act_sequence(
        "a1",
        [
            {"action": "move", "x": 10, "y": 20},
            {"action": "click", "x": 10, "y": 20},
            {"action": "scroll", "dy": -3},
        ],
    )
    sid = _session_id(ctl)
    # the persisted trail: one "step" event per executed sub-step, linked to the grouping entry
    steps = [ev for ev in _events(tmp_path, sid) if ev["event"] == "step"]
    assert [ev["action"] for ev in steps] == ["move", "click", "scroll"]
    assert all(ev["parent"] == res["log_id"] for ev in steps)
    assert all(ev["session"] == sid and ev["result"] == "done" and ev["ts"] for ev in steps)
    assert [ev["steps_used"] for ev in steps] == [1, 2, 3]  # budget accounting per sub-step
    assert [r["log_id"] for r in res["steps"]] == [ev["id"] for ev in steps]
    (seq,) = [ev for ev in _events(tmp_path, sid) if ev["event"] == "sequence"]
    assert seq["id"] == res["log_id"] and seq["steps"] == [ev["id"] for ev in steps] and seq["result"] == "done"
    # the in-memory trail the Settings replay reads: one entry per sub-step, oldest first
    trail = ctl.session_trail()
    assert [e["action"] for e in trail] == ["move", "click", "scroll"]
    assert all(e["parent"] == res["log_id"] for e in trail)
    # the summary agrees with the budget and names the grouped call for what it was
    (summary,) = ctl.sessions_summary()
    assert summary["steps"] == 3 == (ctl.grant.steps_used if ctl.grant else -1)
    assert summary["calls"] == 1
    # the action log keeps its old shape: one combined entry for the batch (receipt, undo)
    entries = ctl.actions()
    assert len(entries) == 1 and entries[0]["action"] == "sequence" and entries[0]["id"] == res["log_id"]


def test_summary_counts_singles_and_sequences_in_the_same_units(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=1, y=1)
    ctl.act_sequence("a1", [{"action": "move", "x": 2, "y": 2}, {"action": "scroll", "dy": 1.0}])
    (summary,) = ctl.sessions_summary()
    assert summary["steps"] == 3 == (ctl.grant.steps_used if ctl.grant else -1)
    assert summary["calls"] == 2  # one single action + one grouped sequence call


def test_cancelled_sequence_files_only_the_substeps_that_ran(tmp_path: Path) -> None:
    backend = CancelOnFirstMove()
    ctl = _control(Clock(), tmp_path, backend=backend)
    backend.ctl = ctl
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="stopped part-way"):
        ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}, {"action": "move", "x": 2, "y": 2}])
    sid = _session_id(ctl)
    trail = ctl.session_trail()
    assert [e["action"] for e in trail] == ["move"]  # only the executed sub-step is filed
    (summary,) = ctl.sessions_summary()
    assert summary["steps"] == 1 == (ctl.grant.steps_used if ctl.grant else -1)
    steps = [ev for ev in _events(tmp_path, sid) if ev["event"] == "step"]
    assert len(steps) == 1 and steps[0]["detail"].startswith("(1, 1)")
    (seq,) = [ev for ev in _events(tmp_path, sid) if ev["event"] == "sequence"]
    assert seq["result"] == "stopped" and seq["steps"] == [steps[0]["id"]]
    entries = ctl.actions()
    assert len(entries) == 1 and entries[0]["result"] == "stopped"


def test_refused_sequence_files_the_substeps_that_ran_before_the_refusal(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, ask=_ask_deny_confirm)
    ctl.hooks.uia = FakeUIA(at=Element("Delete", "ButtonControl"))  # a risky target: the click asks, is denied
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused):
        ctl.act_sequence(
            "a1",
            [
                {"action": "move", "x": 10, "y": 20},
                {"action": "click", "x": 10, "y": 20},  # asks, denied: never runs
                {"action": "move", "x": 99, "y": 99},  # never runs
            ],
        )
    sid = _session_id(ctl)
    assert [e["action"] for e in ctl.session_trail()] == ["move"]
    steps = [ev for ev in _events(tmp_path, sid) if ev["event"] == "step"]
    assert len(steps) == 1 and steps[0]["action"] == "move"
    (seq,) = [ev for ev in _events(tmp_path, sid) if ev["event"] == "sequence"]
    assert seq["result"] == "refused" and seq["steps"] == [steps[0]["id"]]
    entries = ctl.actions()
    assert len(entries) == 1 and entries[0]["result"] == "refused"


def test_substep_events_carry_their_own_screenshot_reference(tmp_path: Path) -> None:
    from PIL import Image

    ctl = _control(Clock(), tmp_path, grab=lambda: Image.new("RGB", (64, 48)))
    ctl.request("a1", "Claude")
    res = ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}, {"action": "move", "x": 2, "y": 2}])
    sid = _session_id(ctl)
    steps = [ev for ev in _events(tmp_path, sid) if ev["event"] == "step"]
    assert len(steps) == 2
    for ev in steps:
        assert ev["thumb"] == f"{ev['id']}.png"  # each sub-step has its own picture, not the batch's one
        assert (tmp_path / "actions" / ev["thumb"]).exists()
    assert res["steps"][0]["log_id"] != res["steps"][1]["log_id"]


# ---------------------------------------------------------------------- E2


def test_user_undo_appends_a_linked_status_transition(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    done = ctl.act("a1", "move", x=50, y=50)
    sid = _session_id(ctl)
    before = (tmp_path / "actions" / f"session-{sid}.jsonl").read_bytes()
    ctl.undo_last_by_user()
    after = (tmp_path / "actions" / f"session-{sid}.jsonl").read_bytes()
    assert after != before  # the persisted trail gained the transition (it used to gain nothing)
    (status,) = [ev for ev in _events(tmp_path, sid) if ev["event"] == "status"]
    assert status["id"] == done["log_id"] and status["status"] == "undone"
    assert status["applied"] == 1 and status["of"] == 1
    (entry,) = [e for e in ctl.session_trail() if e["id"] == done["log_id"]]
    assert entry["result"] == "undone"


def test_agent_undo_appends_a_linked_status_transition(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    done = ctl.act("a1", "scroll", dy=4.0)
    sid = _session_id(ctl)
    ctl.undo_last("a1")
    (status,) = [ev for ev in _events(tmp_path, sid) if ev["event"] == "status"]
    assert status["id"] == done["log_id"] and status["status"] == "undone"


def test_partial_undo_is_recorded_as_partial_and_not_reoffered(tmp_path: Path) -> None:
    backend = FailOnMove()
    ctl = _control(Clock(), tmp_path, backend=backend)
    ctl.request("a1", "Claude")
    res = ctl.act_sequence("a1", [{"action": "move", "x": 1, "y": 1}, {"action": "move", "x": 2, "y": 2}])
    sid = _session_id(ctl)
    backend.fail_on = 4  # two sequence moves ran; the undo's first inverse move is 3, the second is 4
    with pytest.raises(RuntimeError, match="unplugged"):
        ctl.undo_last_by_user()
    (status,) = [ev for ev in _events(tmp_path, sid) if ev["event"] == "status"]
    assert status["id"] == res["log_id"] and status["status"] == "partially undone"
    assert status["applied"] == 1 and status["of"] == 2
    (entry,) = [e for e in ctl.log if e.id == res["log_id"]]
    assert entry.result == "partially undone"
    with pytest.raises(ControlRefused, match="partly undone"):
        ctl.undo_last_by_user()  # a partly-applied inverse is never blindly repeated


def test_failed_undo_is_recorded_and_stays_retryable(tmp_path: Path) -> None:
    backend = FailOnMove()
    ctl = _control(Clock(), tmp_path, backend=backend)
    ctl.request("a1", "Claude")
    done = ctl.act("a1", "move", x=9, y=9)
    sid = _session_id(ctl)
    backend.fail_on = 2  # the action's move was 1; the undo's only inverse move is 2
    with pytest.raises(RuntimeError, match="unplugged"):
        ctl.undo_last_by_user()
    (status,) = [ev for ev in _events(tmp_path, sid) if ev["event"] == "status"]
    assert status["id"] == done["log_id"] and status["status"] == "undo failed"
    assert status["applied"] == 0 and status["of"] == 1
    (entry,) = [e for e in ctl.log if e.id == done["log_id"]]
    assert entry.result == "done"  # nothing ran: the undo may be retried
    backend.fail_on = None
    ctl.undo_last_by_user()
    statuses = [ev["status"] for ev in _events(tmp_path, sid) if ev["event"] == "status"]
    assert statuses == ["undo failed", "undone"]


# ---------------------------------------------------------------------- E3


def test_restart_reloads_the_session_history(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=1, y=1)
    res = ctl.act_sequence("a1", [{"action": "move", "x": 2, "y": 2}, {"action": "scroll", "dy": 2.0}])
    sid = _session_id(ctl)
    ctl.stop("done")
    ctl2 = _control(Clock(), tmp_path)  # a fresh process over the same folder
    (summary,) = [s for s in ctl2.sessions_summary() if s["session"] == sid]
    assert summary["agent"] == "Claude" and summary["steps"] == 3 and summary["calls"] == 2
    trail = ctl2.session_trail(session_id=sid)
    assert [e["action"] for e in trail] == ["move", "move", "scroll"]  # oldest first, sub-steps included
    assert trail[1]["parent"] == res["log_id"] and trail[0]["parent"] == ""
    assert all(e["result"] == "done" for e in trail)


def test_reloaded_history_applies_undo_transitions(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    done = ctl.act("a1", "move", x=5, y=5)
    sid = _session_id(ctl)
    ctl.undo_last_by_user()
    ctl.stop("done")
    ctl2 = _control(Clock(), tmp_path)
    (entry,) = [e for e in ctl2.session_trail(session_id=sid) if e["id"] == done["log_id"]]
    assert entry["result"] == "undone"  # the durable record no longer claims "done"


def test_reload_skips_corrupt_lines_and_keeps_the_valid_records(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=1, y=1)
    ctl.act("a1", "move", x=2, y=2)
    sid = _session_id(ctl)
    ctl.stop("done")
    path = tmp_path / "actions" / f"session-{sid}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    damaged = [
        lines[0],
        '{"event": "step", "id": "x", "act',  # truncated mid-record, as a crash would leave it
        "this is not json at all",
        "42",  # valid JSON, not a record
        *lines[1:],
        '{"event": "step", "id": "never-finished"',  # truncated tail
    ]
    path.write_text("\n".join(damaged) + "\n", encoding="utf-8")
    ctl2 = _control(Clock(), tmp_path)
    trail = ctl2.session_trail(session_id=sid)
    assert len(trail) == 2  # both valid steps survived the damage around them
    assert trail[0]["detail"].startswith("(1, 1)") and trail[1]["detail"].startswith("(2, 2)")


def test_reload_tolerates_a_missing_thumbnail(tmp_path: Path) -> None:
    folder = tmp_path / "actions"
    folder.mkdir(parents=True)
    (folder / "real.png").write_bytes(b"present, if not a real png")
    sid = "abc123def0"
    records = [
        {"ts": 1.0, "session": sid, "event": "started", "agent": "Claude", "agent_id": "a1", "step_budget": 40},
        {
            "ts": 2.0,
            "session": sid,
            "event": "step",
            "id": "step1",
            "agent": "Claude",
            "agent_id": "a1",
            "action": "move",
            "detail": "(1, 1)",
            "result": "done",
            "thumb": "gone.png",  # the picture was pruned or deleted; the record must survive
        },
        {
            "ts": 3.0,
            "session": sid,
            "event": "step",
            "id": "step2",
            "agent": "Claude",
            "agent_id": "a1",
            "action": "scroll",
            "detail": "dy=2 dx=0",
            "result": "done",
            "thumb": "real.png",
        },
    ]
    (folder / f"session-{sid}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    ctl = _control(Clock(), tmp_path)
    trail = ctl.session_trail(session_id=sid)
    assert [e["id"] for e in trail] == ["step1", "step2"]
    assert trail[0]["thumb"] == ""  # missing picture: cleared, record kept
    assert trail[1]["thumb"] == "real.png"
    (summary,) = ctl.sessions_summary()
    assert summary["agent"] == "Claude" and summary["steps"] == 2


def test_reload_is_bounded_to_the_newest_sessions(tmp_path: Path) -> None:
    folder = tmp_path / "actions"
    folder.mkdir(parents=True)
    base = 1_700_000_000
    for i in range(MAX_SESSIONS + 2):
        sid = f"s{i:02d}"
        records = [
            {"ts": float(i), "session": sid, "event": "started", "agent": "Claude", "agent_id": "a1"},
            {
                "ts": float(i),
                "session": sid,
                "event": "step",
                "id": f"step{i}",
                "agent": "Claude",
                "agent_id": "a1",
                "action": "move",
                "detail": "(1, 1)",
                "result": "done",
                "thumb": "",
            },
        ]
        path = folder / f"session-{sid}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        os.utime(path, (base + i * 10, base + i * 10))
    ctl = _control(Clock(), tmp_path)
    assert len(ctl.sessions) == MAX_SESSIONS  # the same bound as the live trail
    assert "s00" not in ctl.sessions and "s01" not in ctl.sessions  # the oldest are the ones dropped
    assert f"s{MAX_SESSIONS + 1:02d}" in ctl.sessions


def test_restored_sessions_count_toward_pruning(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ids = []
    for _ in range(MAX_SESSIONS):
        ctl.request("a1", "Claude")
        ids.append(_session_id(ctl))
        ctl.stop("done")
    ctl2 = _control(Clock(), tmp_path)  # restart: all of them come back
    assert len(ctl2.sessions) == MAX_SESSIONS
    ctl2.request("a1", "Claude")  # one new session pushes the oldest restored one out
    assert len(ctl2.sessions) == MAX_SESSIONS
    assert ids[0] not in ctl2.sessions
    assert not (tmp_path / "actions" / f"session-{ids[0]}.jsonl").exists()
    ctl2.stop("done")


def test_restored_pruning_uses_started_order_when_mtimes_tie(tmp_path: Path) -> None:
    # #96 CI on Windows: session files written in rapid sequence shared one
    # modification timestamp, and the name tiebreak (random hex ids) made
    # pruning drop an arbitrary session instead of the oldest. The started
    # line's wall-clock timestamp breaks the tie by actual creation order.
    ctl = _control(Clock(), tmp_path)
    ids = []
    for _ in range(MAX_SESSIONS):
        ctl.request("a1", "Claude")
        ids.append(_session_id(ctl))
        ctl.stop("done")
    for path in (tmp_path / "actions").glob("session-*.jsonl"):
        os.utime(path, (1700000000.0, 1700000000.0))
    ctl2 = _control(Clock(), tmp_path)
    assert len(ctl2.sessions) == MAX_SESSIONS
    ctl2.request("a1", "Claude")
    assert len(ctl2.sessions) == MAX_SESSIONS
    assert ids[0] not in ctl2.sessions
    assert not (tmp_path / "actions" / f"session-{ids[0]}.jsonl").exists()
    ctl2.stop("done")


def test_trail_off_persists_nothing_and_reloads_nothing(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=1, y=1)
    sid = _session_id(ctl)
    ctl.stop("done")
    assert (tmp_path / "actions" / f"session-{sid}.jsonl").exists()  # written while the trail was on
    off = _control(Clock(), tmp_path, audit_trail=False)
    assert off.sessions == {}  # the switch is off: the files on disk are not reloaded either
    assert off.sessions_summary() == []
    off.request("a1", "Claude")
    off.act("a1", "move", x=2, y=2)
    assert off.session_trail() == []
    assert [p for p in (tmp_path / "actions").glob("session-*.jsonl")] == [
        tmp_path / "actions" / f"session-{sid}.jsonl"
    ]  # and nothing new was persisted


# ---------------------------------------------------------------------- E4


def test_zero_and_negative_limits_return_nothing_not_everything(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    ctl.request("a1", "Claude")
    for i in range(3):
        ctl.act("a1", "move", x=i, y=i)
    assert ctl.actions(limit=0) == []
    assert ctl.actions(limit=-5) == []
    assert ctl.session_trail(limit=0) == []
    assert ctl.session_trail(limit=-1) == []
    assert len(ctl.actions(limit=2)) == 2  # a positive limit still means what it says
    assert len(ctl.session_trail(limit=2)) == 2
    assert len(ctl.actions()) == 3 and len(ctl.session_trail()) == 3  # the defaults are unchanged


def _core(tmp_path: Path) -> Any:
    from away import config as C
    from away.agentlink.host import build_core
    from away.agentlink.prompts import NullUI

    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    c = build_core(
        tmp_path,
        cfg=lambda: cfg,
        settings=lambda: cfg.agent_link,
        config_path=lambda: cfgp,
        ui=NullUI(),
    )
    c.prompts.polite = lambda: True
    c.cfg_obj = cfg  # type: ignore[attr-defined]
    return c


def test_service_action_log_clamps_the_agent_supplied_limit(tmp_path: Path) -> None:
    core = _core(tmp_path)
    ctl = core.control
    ctl.ask = lambda *a, **k: {"answer": "Allow"}
    a, _ = core.registry.issue("Claude", ["read", "control"], 3600)
    ctl.request(a.id, a.name)
    for i in range(3):
        ctl.act(a.id, "move", x=i, y=i)
    assert core.service.action_log(a, 0) == []
    assert core.service.action_log(a, -5) == []
    assert len(core.service.action_log(a, 2)) == 2
    assert len(core.service.action_log(a)) == 3
