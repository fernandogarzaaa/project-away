"""v0.1.28: per-session step budget + visual audit trail for Agent Link computer control.

Every test here fails on the unfixed code (checked with `git stash push src/`): the new constructor
kwargs, the `steps_left` results, the plain-words exhaustion refusal, and the session trails do not exist
there.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from away.agentlink.control import (
    DEFAULT_STEP_BUDGET,
    MAX_SESSIONS,
    TRAIL_CAP,
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


def _control(
    clock: Clock,
    tmp_path: Path,
    step_budget: int = DEFAULT_STEP_BUDGET,
    audit_trail: bool = True,
    settings: Any = None,
    rate: float = 10000.0,
) -> ComputerControl:
    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,  # no real screenshots in tests; thumbnails stay empty
        sleep=lambda _s: None,
        uia=FakeUIA(at=Element("Canvas", "PaneControl")),
    )
    hooks.cursor_pos = lambda: (7, 8)  # so "move" is undoable in tests
    return ComputerControl(
        hooks,
        # "Allow" for the approval prompt, "Continue" for the resume prompt
        lambda _q, _o, **kw: {"answer": "Continue" if kw.get("title") == "Resume?" else "Allow", "method": "click"},
        ttl_s=1800,
        rate_per_s=rate,
        log_folder=tmp_path / "actions",
        clock=clock,
        step_budget=step_budget,
        audit_trail=audit_trail,
        settings=settings,
    )


def _session(ctl: ComputerControl, agent_id: str = "a1", name: str = "Claude") -> dict[str, Any]:
    return ctl.request(agent_id, name)


# ------------------------------------------------------------------ step budget


def test_every_action_result_tells_the_agent_its_remaining_budget(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=3)
    _session(ctl)
    assert ctl.act("a1", "move", x=10, y=10)["steps_left"] == 2
    assert ctl.act("a1", "move", x=20, y=20)["steps_left"] == 1
    assert ctl.act("a1", "move", x=30, y=30)["steps_left"] == 0


def test_an_exhausted_session_refuses_new_actions_in_plain_words(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=2)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)
    ctl.act("a1", "move", x=2, y=2)
    with pytest.raises(ControlRefused) as exc:
        ctl.act("a1", "move", x=3, y=3)
    msg = str(exc.value)
    assert "all 2 of this session's steps" in msg  # names the budget, in plain words
    assert "ask" in msg and "approving" in msg  # says what to do next
    assert "Traceback" not in msg and "ControlRefused" not in msg  # no jargon for the agent


def test_the_budget_resets_with_each_new_grant(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=1)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)
    with pytest.raises(ControlRefused):
        ctl.act("a1", "move", x=2, y=2)
    ctl.stop("the agent finished")
    st = _session(ctl)  # the person approves again: a fresh session, a fresh budget
    assert st["steps_used"] == 0 and st["steps_left"] == 1
    assert ctl.act("a1", "move", x=3, y=3)["steps_left"] == 0


def test_pausing_and_resuming_keeps_counting_the_same_session(tmp_path: Path) -> None:
    clock = Clock()
    ctl = _control(clock, tmp_path, step_budget=2)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)
    clock.advance(1.0)  # past the approval grace: this mouse move is the person taking over
    assert ctl.user_input("mouse") is True  # the person takes over: paused, not ended
    st = ctl.resume("a1")
    assert st["steps_used"] == 1 and st["steps_left"] == 1
    ctl.act("a1", "move", x=2, y=2)
    with pytest.raises(ControlRefused):
        ctl.act("a1", "move", x=3, y=3)


def test_undo_counts_as_a_step(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=2)
    _session(ctl)
    ctl.act("a1", "move", x=50, y=50)  # undoable: moving back is one step
    ctl.undo_last("a1")
    with pytest.raises(ControlRefused):
        ctl.act("a1", "move", x=60, y=60)


def test_the_budget_follows_the_live_settings(tmp_path: Path) -> None:
    cfg = SimpleNamespace(control_step_budget=5, control_audit_trail=True)
    ctl = _control(Clock(), tmp_path, settings=lambda: cfg)
    assert _session(ctl)["step_budget"] == 5
    ctl.stop("done")
    cfg.control_step_budget = 7  # the person changes it in Settings: the next session uses it, no restart
    assert _session(ctl)["step_budget"] == 7


def test_status_reports_the_budget(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=9)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)
    st = ctl.status()
    assert (st["step_budget"], st["steps_used"], st["steps_left"]) == (9, 1, 8)
    assert st["session"]  # the session id the audit trail is filed under


# ------------------------------------------------------------------ audit trail


def test_the_trail_is_per_session_and_oldest_first(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=10)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)
    ctl.act("a1", "scroll", dy=2.0)
    first_session = ctl.grant.id if ctl.grant else ""
    ctl.stop("done")
    _session(ctl, agent_id="a2", name="Hark")
    ctl.act("a2", "move", x=9, y=9)
    trail = ctl.session_trail()
    assert [e["action"] for e in trail] == ["move"]  # only this session's steps
    assert all(e["session"] == trail[0]["session"] for e in trail)
    assert trail[0]["session"] != first_session
    old = ctl.session_trail(session_id=first_session)
    assert [e["action"] for e in old] == ["move", "scroll"]
    assert old[0]["ts"] <= old[1]["ts"]  # oldest first: replayable in order


def test_every_trail_step_carries_timestamp_action_and_result(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=10)
    _session(ctl)
    ctl.act("a1", "move", x=4, y=5)
    (e,) = ctl.session_trail()
    assert e["ts"] > 0 and e["action"] == "move" and e["result"] == "done"
    assert e["detail"].startswith("(4, 5)") and "thumb" in e and "id" in e


def test_the_trail_is_persisted_per_session_and_bounded(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=TRAIL_CAP + 50)
    _session(ctl)
    for i in range(TRAIL_CAP + 20):
        ctl.act("a1", "move", x=i % 100, y=1)
    trail = ctl.session_trail(limit=TRAIL_CAP + 1000)
    assert len(trail) == TRAIL_CAP  # bounded in memory
    assert ctl.grant is not None
    lines = (tmp_path / "actions" / f"session-{ctl.grant.id}.jsonl").read_text(encoding="utf-8").splitlines()
    import json

    events = [json.loads(line) for line in lines]
    assert events[0]["event"] == "started" and events[0]["step_budget"] == TRAIL_CAP + 50
    steps = [ev for ev in events if ev["event"] == "step"]
    assert len(steps) == TRAIL_CAP + 20  # every step persisted with its result
    assert all("ts" in ev and "action" in ev and "result" in ev and "thumb" in ev for ev in steps)
    assert steps[0]["session"] == ctl.grant.id


def test_old_sessions_are_pruned_in_memory_and_on_disk(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=10)
    ids = []
    for _ in range(MAX_SESSIONS + 3):
        _session(ctl)
        assert ctl.grant is not None
        ids.append(ctl.grant.id)
        ctl.stop("done")
    assert len(ctl.sessions) == MAX_SESSIONS
    assert ids[0] not in ctl.sessions  # the oldest session is gone
    assert not (tmp_path / "actions" / f"session-{ids[0]}.jsonl").exists()
    assert (tmp_path / "actions" / f"session-{ids[-1]}.jsonl").exists()


def test_turning_the_trail_off_keeps_no_pictures_or_session_files(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=10, audit_trail=False)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)
    assert list((tmp_path / "actions").glob("*.jsonl")) == []
    assert list((tmp_path / "actions").glob("*.png")) == []
    assert ctl.session_trail() == []  # nothing recorded
    assert len(ctl.log) == 1  # ...but the safety action log (undo) still works
    ctl.undo_last("a1")


def test_actions_can_be_filtered_to_one_session(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=10)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)
    first = ctl.grant.id if ctl.grant else ""
    ctl.stop("done")
    _session(ctl)
    ctl.act("a1", "move", x=2, y=2)
    mine = ctl.actions(50, session_id=first)
    assert len(mine) == 1 and mine[0]["detail"].startswith("(1, 1)")
    assert all("session" in e for e in ctl.actions(50))


# ------------------------------------------------------------------ config + model


def test_control_step_budget_must_stay_in_range() -> None:
    from away import config as cfgmod

    for bad in (0, -1, 1001, "40"):
        cfg = cfgmod.AppConfig()
        cfg.agent_link.control_step_budget = bad  # type: ignore[assignment]
        with pytest.raises(ValueError):
            cfgmod.validate_agent_link(cfg.agent_link)
    cfg = cfgmod.AppConfig()
    cfg.agent_link.control_step_budget = 40
    cfgmod.validate_agent_link(cfg.agent_link)  # the default is valid


def test_gui_model_roundtrips_the_budget_and_the_trail_switch(tmp_path: Path) -> None:
    from away import config as cfgmod
    from away.agentlink.gui_model import AgentLinkModel

    cfg_path = tmp_path / "config.yaml"
    m = AgentLinkModel(config_path=cfg_path)
    assert m.control_budget() == 40  # the default, no file yet
    assert m.audit_trail_on() is True
    assert m.set_control_budget(25) == 25
    assert m.control_budget() == 25
    assert m.set_control_budget(99999) == 1000  # clamped to the valid range
    m.set_audit_trail(False)
    assert m.audit_trail_on() is False
    cfg, _ = cfgmod.load(cfg_path)
    assert cfg.agent_link.control_step_budget == 1000
    assert cfg.agent_link.control_audit_trail is False


def test_a_sequence_consumes_one_step_per_sub_step(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=10)
    _session(ctl)
    res = ctl.act_sequence(
        "a1",
        [
            {"action": "move", "x": 1, "y": 1},
            {"action": "move", "x": 2, "y": 2},
            {"action": "move", "x": 3, "y": 3},
        ],
    )
    assert res["ok"] is True and len(res["steps"]) == 3
    assert ctl.grant is not None and ctl.grant.steps_used == 3  # not 1 for the outer call


def test_a_sequence_cannot_exceed_the_remaining_budget(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=3)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)  # 2 steps left
    with pytest.raises(ControlRefused, match="only 2 are left"):
        ctl.act_sequence(
            "a1",
            [
                {"action": "move", "x": 2, "y": 2},
                {"action": "move", "x": 3, "y": 3},
                {"action": "move", "x": 4, "y": 4},
            ],
        )
    assert ctl.grant is not None and ctl.grant.steps_used == 1  # refused up front: nothing ran
    ok = ctl.act_sequence("a1", [{"action": "move", "x": 2, "y": 2}, {"action": "move", "x": 3, "y": 3}])
    assert ok["ok"] is True and ctl.grant.steps_used == 3


def test_an_exhausted_budget_blocks_sequences_too(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path, step_budget=1)
    _session(ctl)
    ctl.act("a1", "move", x=1, y=1)
    with pytest.raises(ControlRefused, match="only 0 are left"):
        ctl.act_sequence("a1", [{"action": "move", "x": 2, "y": 2}])
