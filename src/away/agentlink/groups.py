"""Grouped tools (v0.1.21): the MCP tool list went from 37 tools to 15, which an agent can reason over.

v0.1.28: 17 tools (``shell`` added).

Related tools became one tool with an ``action`` argument (``settings(action="change", ...)``,
``pointer(action="click", ...)``). Each group action maps to an existing service method, so nothing below the MCP
layer changed: every REST route (``/v1/settings/change``...) and every ``/v1/rpc/<method>`` name still works, and
the group names work on ``/v1/rpc/<group>`` too (``{"action": "change", ...}``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroupAction:
    method: str  # the service method (also the /v1/rpc name)
    params: tuple[str, ...] = ()  # arguments passed through (others are ignored)
    rename: Mapping[str, str] = field(default_factory=dict)  # group argument -> method argument
    fixed: Mapping[str, Any] = field(default_factory=dict)  # always passed (e.g. act's action)


POINT = ("x", "y", "element", "from_image")

GROUPS: dict[str, dict[str, GroupAction]] = {
    "events": {
        "poll": GroupAction("poll", ("since", "landmarks")),
        "wait": GroupAction("wait_for_event", ("since", "timeout_s", "topics")),
    },
    "settings": {
        "list": GroupAction("list_settings", ("domain",)),
        "change": GroupAction("change_settings", ("changes", "ops", "why"), {"changes": "ops"}),
        "undo": GroupAction("undo_changes", ("what",)),
        "history": GroupAction("change_history", ("limit",)),
        "config": GroupAction("get_config"),
    },
    "preferences": {
        "list": GroupAction("get_preferences"),
        "add": GroupAction("add_preference", ("text",)),
        "remove": GroupAction("remove_preference", ("number",)),
    },
    "diagnostics": {
        "run": GroupAction("run_diagnose", ("kind", "device", "seconds", "quick")),
        "job": GroupAction("get_job", ("job_id",)),
        "record": GroupAction("record_session", ("seconds", "note")),
        "bundle": GroupAction("get_bundle", ("note",)),
        "files": GroupAction("list_files"),
        "download": GroupAction("read_file", ("path",), {"path": "rel"}),
        "logs": GroupAction("get_logs", ("lines",)),
        "selftest": GroupAction("run_selftest", ("wait", "only")),  # v0.1.24
        "selftest_latest": GroupAction("selftest_latest"),
    },
    "control": {
        "request": GroupAction("request_control", ("why", "minutes")),
        "resume": GroupAction("resume_control"),
        "stop": GroupAction("stop_control"),
        "undo": GroupAction("undo_last_action"),
        "log": GroupAction("action_log", ("limit",)),
        "sequence": GroupAction("act_sequence", ("steps",)),
    },
    "pointer": {
        "move": GroupAction("act", POINT, fixed={"action": "move"}),
        "click": GroupAction("act", POINT, fixed={"action": "click"}),
        "double_click": GroupAction("act", POINT, fixed={"action": "double_click"}),
        "right_click": GroupAction("act", POINT, fixed={"action": "right_click"}),
        "drag": GroupAction("act", (*POINT, "x2", "y2", "to_element"), fixed={"action": "drag"}),
        "scroll": GroupAction("act", ("dy", "dx", "x", "y", "element", "from_image"), fixed={"action": "scroll"}),
    },
    "keyboard": {
        "type": GroupAction("act", ("text",), fixed={"action": "type_text"}),
        "press": GroupAction("act", ("keys",), fixed={"action": "press_keys"}),
    },
    "inbox": {  # v0.1.22 "Talk to your agent"
        "list": GroupAction("inbox_list", ("unread_only",)),
        "read": GroupAction("inbox_read", ("id",)),
        "reply": GroupAction("inbox_reply", ("id", "text", "speak")),
    },
    "open": {
        "app": GroupAction("act", ("name",), fixed={"action": "open_app"}),
        "url": GroupAction("act", ("url",), fixed={"action": "open_url"}),
        "system": GroupAction("act", ("name",), fixed={"action": "open_app", "system": True}),
    },
    "shell": {  # v0.1.28: staged shell (proposal #26)
        "list": GroupAction("run_shell", (), fixed={"action": "list"}),
        "run": GroupAction("run_shell", ("command", "argv", "why")),
        "log": GroupAction("run_shell", ("limit",), fixed={"action": "log"}),
    },
    "shell": {  # v0.1.28: staged shell (proposal #26)
        "list": GroupAction("run_shell", (), fixed={"action": "list"}),
        "run": GroupAction("run_shell", ("command", "argv", "why")),
        "log": GroupAction("run_shell", ("limit",), fixed={"action": "log"}),
    },
}


def actions(group: str) -> tuple[str, ...]:
    return tuple(GROUPS[group])


def resolve(group: str, kwargs: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """``("settings", {"action": "change", "changes": [...], "why": "..."})`` ->
    ``("change_settings", {"ops": [...], "why": "..."})``. Unset (None) arguments are left out; arguments that do
    not belong to the action are ignored. Raises ValueError for an unknown action."""
    if group not in GROUPS:
        raise KeyError(f"unknown tool {group!r}")
    action = str(kwargs.get("action") or "").strip()
    acts = GROUPS[group]
    if action not in acts:
        raise ValueError(f"{group}: action must be one of {', '.join(acts)} (got {action!r})")
    ga = acts[action]
    out: dict[str, Any] = dict(ga.fixed)
    for p in ga.params:
        v = kwargs.get(p)
        if v is None:
            continue
        out[ga.rename.get(p, p)] = v
    return ga.method, out


def describe() -> list[str]:
    """One line per group for the README / about: ``settings(action=list|change|undo|history|config)``."""
    return [f"{g}(action={'|'.join(a)})" for g, a in GROUPS.items()]
