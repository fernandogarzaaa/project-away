"""Docs/API contract tests (audit 2026-10-20, lane B items 5.1 and 5.2).

The README's tool table and about.py's budget wording are part of the
published contract. These tests hold them to the code:

* the tool count the README claims equals the number of tools the MCP
  surface actually registers, every registered tool is named in the
  README table, and the `open` row documents the `system` action;
* every agent action result carries `steps_left` (single actions already
  did; sequence results, their per-step results and undo results did
  not), and the value agrees with `status()` at the same moment.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks
from away.agentlink.desktop import Element
from away.control.mouse import RecordingBackend

README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_tool_count_and_table_match_registered_mcp_tools() -> None:
    pytest.importorskip("mcp")
    from away.agentlink import mcp_tools

    class StubBackend:
        def call(self, ctx: Any, method: str, **kw: Any) -> Any:
            raise AssertionError("listing tools must not call the backend")

    mcp = mcp_tools.build(StubBackend())
    names = {t.name for t in asyncio.run(mcp.list_tools())}

    text = README.read_text(encoding="utf-8")
    m = re.search(r"\*\*Tools \((\d+);", text)
    assert m, "README no longer states a tool count in its Tools heading"
    assert int(m.group(1)) == len(names), (
        f"README claims {m.group(1)} tools but the MCP surface registers {len(names)}: {sorted(names)}"
    )
    # The names must come from the Tools table itself, not from anywhere in the
    # README: a whole-document search would still pass if a row were deleted
    # while its name survived in the heading or elsewhere. Rows start with
    # "| `name`", and one row groups two tools with a comma.
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("**Tools ("))
    rows = []
    for line in lines[start:]:
        if line.startswith("| `"):
            rows.append(line)
        elif rows and not line.startswith("|"):
            break
    table_names = set()
    for row in rows:
        table_names.update(re.findall(r"`([^`]+)`", row.split("|")[1]))
    assert table_names == names, (
        f"README tool table names {sorted(table_names)} but the MCP surface registers {sorted(names)}"
    )
    open_row = next((line for line in rows if line.startswith("| `open`")), "")
    assert "`system`" in open_row, "README's `open` row omits the `system` action"


class FakeUIA:
    available = True

    def focused(self) -> Any:
        return Element("Message", "EditControl", rect=(0, 0, 9, 9))

    def element_at(self, x: int, y: int) -> Any:
        return Element("Canvas", "PaneControl", rect=(0, 0, 9, 9))

    def find(self, name: str, control_type: str | None = None) -> Any:
        return None

    def tree(self, *a: Any) -> None:
        return None


def _control() -> ComputerControl:
    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        return {"answer": "Allow", "method": "click"}

    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=FakeUIA(),
    )
    ctl = ComputerControl(hooks, ask, ttl_s=1800, rate_per_s=1000, clock=lambda: 1000.0)
    ctl.hooks.cursor_pos = lambda: (1, 2)
    return ctl


def test_sequence_and_undo_results_carry_steps_left_consistent_with_status() -> None:
    ctl = _control()
    ctl.request("a1", "Claude")

    single = ctl.act("a1", "move", x=10, y=10)
    assert single["steps_left"] == ctl.status()["steps_left"]

    seq = ctl.act_sequence("a1", [{"action": "move", "x": 20, "y": 20}, {"action": "scroll", "dy": -3}])
    assert seq["steps_left"] == ctl.status()["steps_left"], "sequence result omits or misreports steps_left"
    assert all("steps_left" in s for s in seq["steps"]), "sequence per-step results omit steps_left"
    assert seq["steps"][-1]["steps_left"] == seq["steps_left"]

    und = ctl.undo_last("a1")
    assert und["steps_left"] == ctl.status()["steps_left"], "undo result omits or misreports steps_left"
