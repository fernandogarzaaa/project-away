"""Agent Link: semantic merge of step budget (#66) with element_id grounding (#68) and
undo scoping (#64).

- act_sequence passes element_id through each sub-step to _point, which resolves it via the
  agent's ui_query with staleness revalidation.
- _undo accepts the only_ids filter (from #64's consent-receipt scoping).
- Agent-triggered undo consumes one step; user-triggered receipt Undo All is not charged to an agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from away.agentlink.control import (
    DEFAULT_STEP_BUDGET,
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


def _tree() -> Element:
    send = Element("Send", "ButtonControl", rect=(100, 100, 140, 120))
    root = Element("Mail", "WindowControl", rect=(0, 0, 800, 600))
    root.children = [send]
    return root


class FakeUIA:
    available = True
    reason = ""

    def __init__(self, root: Element | None = None) -> None:
        self._root = root if root is not None else _tree()

    def tree(self, max_depth: int = 3, max_nodes: int = 200) -> Element | None:
        return self._root

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        # Model the live UIA lookup at the center of the Send button. The production code
        # deliberately refuses cached elements when this live lookup is absent or inconclusive.
        root = self._root
        if root is None:
            return None
        stack = [root]
        while stack:
            el = stack.pop()
            if el.rect is not None:
                left, top, right, bottom = el.rect
                if left <= x <= right and top <= y <= bottom:
                    return el
            stack.extend(el.children)
        return None

    def focused(self) -> Element | None:
        return None


def _control(
    tmp_path: Path,
    step_budget: int = DEFAULT_STEP_BUDGET,
    fg: dict[str, Any] | None = None,
) -> ComputerControl:
    clock = Clock()
    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: fg,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=FakeUIA(),
    )
    hooks.cursor_pos = lambda: (7, 8)
    return ComputerControl(
        hooks,
        lambda _q, _o, **kw: {"answer": "Allow", "method": "click"},
        ttl_s=1800,
        rate_per_s=10000.0,
        log_folder=tmp_path / "actions",
        clock=clock,
        step_budget=step_budget,
    )


def _session(ctl: ComputerControl, agent_id: str = "a1") -> None:
    ctl.request(agent_id, "Claude")


FG = {"title": "Mail", "process": "mail.exe", "class": "Win", "rect": [0, 0, 800, 600]}


# ------------------------------------------------- element_id through act_sequence


def test_sequence_sub_step_resolves_element_id(tmp_path: Path) -> None:
    ctl = _control(tmp_path, fg=dict(FG))
    _session(ctl)
    ctl.ui_query("a1")
    res = ctl.act_sequence("a1", [{"action": "click", "element_id": "e2"}])
    assert res["ok"] is True
    # the click landed on the Send button's center, not on raw coordinates
    moves = [c for c in ctl.hooks.backend.calls if c[0] == "move"]
    assert moves and moves[-1][1] == (120, 110)


def test_sequence_counts_element_id_steps_against_budget(tmp_path: Path) -> None:
    ctl = _control(tmp_path, step_budget=2, fg=dict(FG))
    _session(ctl)
    ctl.ui_query("a1")
    ctl.act_sequence("a1", [{"action": "click", "element_id": "e2"}])
    assert ctl.grant is not None and ctl.grant.steps_used == 1
    ctl.act_sequence("a1", [{"action": "click", "element_id": "e2"}])
    assert ctl.grant.steps_used == 2
    with pytest.raises(ControlRefused, match="only 0 are left in this session"):
        ctl.act_sequence("a1", [{"action": "click", "element_id": "e2"}])


def test_stale_element_id_in_sequence_is_refused(tmp_path: Path) -> None:
    ctl = _control(tmp_path, fg=dict(FG))
    _session(ctl)
    ctl.ui_query("a1")
    ctl.hooks.foreground = lambda: {"title": "Other", "process": "o.exe", "class": "W", "rect": [0, 0, 10, 10]}  # type: ignore[method-assign]
    with pytest.raises(ControlRefused, match="foreground window changed"):
        ctl.act_sequence("a1", [{"action": "click", "element_id": "e2"}])


# ------------------------------------------------- undo scoping and budget


def test_undo_consumes_budget(tmp_path: Path) -> None:
    ctl = _control(tmp_path, step_budget=3)
    _session(ctl)
    ctl.act("a1", "move", x=10, y=10)
    assert ctl.grant is not None and ctl.grant.steps_used == 1
    ctl.undo_last("a1")
    assert ctl.grant.steps_used == 2  # undo is a state-changing agent action


def test_undo_only_ids_filters_to_listed_entries(tmp_path: Path) -> None:
    ctl = _control(tmp_path)
    _session(ctl)
    ctl.act("a1", "move", x=10, y=10)
    first_id = ctl.log[-1].id
    ctl.act("a1", "move", x=20, y=20)
    second_id = ctl.log[-1].id
    # undo scoped to the first entry only: the second stays done
    ctl._undo("a1", only_ids={first_id})
    results = {e.id: e.result for e in ctl.log}
    assert results[first_id] == "undone"
    assert results[second_id] == "done"


def test_undo_only_ids_empty_set_finds_nothing(tmp_path: Path) -> None:
    ctl = _control(tmp_path)
    _session(ctl)
    ctl.act("a1", "move", x=10, y=10)
    with pytest.raises(ControlRefused, match="nothing that can be undone"):
        ctl._undo("a1", only_ids=set())
