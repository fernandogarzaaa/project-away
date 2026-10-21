"""Proposal #2: allowlisted first-class system-tool opening (open action="system").

Opening an allowlisted Windows system tool (Device Manager, Task Manager, Settings pages) is navigation
only: it launches straight through the shell (no synthetic keystrokes, no command line), while terminals,
PowerShell, regedit and mmc-for-other-snapins stay blocked. Every test here fails on the unfixed code
(system=True was ignored, so "device-manager" fell through to Start-menu typing and unknown tools were
either blocked with the wrong message or refused with no allowlist to offer).
"""

from __future__ import annotations

from typing import Any

import pytest

from away.agentlink import groups
from away.agentlink.control import (
    SAFE_SYSTEM_TOOLS,
    ComputerControl,
    ControlHooks,
    ControlRefused,
)
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _control(clock: Clock, platform: str = "win32") -> tuple[ComputerControl, Any, list[str]]:
    launched: list[str] = []
    queue = ["Allow"]

    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        return {"answer": queue.pop(0) if queue else None, "method": "click"}

    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform=platform,
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        system_launcher=launched.append,
    )
    ctl = ComputerControl(hooks, ask, ttl_s=1800, rate_per_s=50, clock=clock)
    ctl.request("a1", "Claude")
    return ctl, hooks.backend, launched


def test_system_open_launches_allowlisted_tool_through_the_shell() -> None:
    clock = Clock()
    ctl, backend, launched = _control(clock)
    res = ctl.act("a1", "open_app", name="device-manager", system=True)
    assert res["ok"] is True
    assert launched == [r"C:\Windows\System32\devmgmt.msc"]  # the .msc path, not a bare "mmc"
    # no synthetic keystrokes: the Start-menu type-the-name path was not used
    assert [c for c in backend.calls if c[0] in ("key", "type_text")] == []


def test_system_open_normalizes_the_name() -> None:
    ctl, _backend, launched = _control(Clock())
    ctl.act("a1", "open_app", name="Device Manager", system=True)
    assert launched == [SAFE_SYSTEM_TOOLS["device-manager"]]
    ctl.act("a1", "open_app", name="WINDOWS-UPDATE", system=True)
    assert launched[-1] == "ms-settings:windowsupdate"


def test_system_open_settings_page_uses_ms_settings_uri() -> None:
    ctl, _backend, launched = _control(Clock())
    ctl.act("a1", "open_app", name="windows-update", system=True)
    assert launched == ["ms-settings:windowsupdate"]


def test_task_manager_is_not_allowlisted() -> None:
    # Regression: Task Manager must stay out of the allowlist because its
    # File > Run new task dialog executes arbitrary commands, which would
    # bypass the terminal block through an allowlisted tool.
    assert "task-manager" not in SAFE_SYSTEM_TOOLS
    ctl, backend, launched = _control(Clock())
    with pytest.raises(ControlRefused, match="unknown system tool"):
        ctl.act("a1", "open_app", name="task-manager", system=True)
    with pytest.raises(ControlRefused, match="unknown system tool"):
        ctl.act("a1", "open_app", name="Task Manager", system=True)
    assert launched == []
    assert backend.calls == []


def test_system_open_rejects_unknown_tools() -> None:
    ctl, backend, launched = _control(Clock())
    for name in ("regedit", "Registry Editor", "mmc", "PowerShell", "cmd", "not-a-tool"):
        with pytest.raises(ControlRefused, match="unknown system tool"):
            ctl.act("a1", "open_app", name=name, system=True)
    assert launched == []
    assert backend.calls == []  # nothing was typed or launched


def test_system_open_is_windows_only() -> None:
    ctl, backend, launched = _control(Clock(), platform="linux")
    with pytest.raises(ControlRefused, match="only available on Windows"):
        ctl.act("a1", "open_app", name="device-manager", system=True)
    assert launched == []
    assert backend.calls == []


def test_system_open_needs_a_live_grant() -> None:
    ctl, _backend, launched = _control(Clock())
    ctl.stop("user pressed stop")
    with pytest.raises(ControlRefused):
        ctl.act("a1", "open_app", name="device-manager", system=True)
    assert launched == []


def test_plain_open_app_still_blocks_terminals_and_system_tools() -> None:
    ctl, backend, launched = _control(Clock())
    for name in ("PowerShell", "Terminal", "Command Prompt", "taskmgr", "mmc", "Registry Editor"):
        with pytest.raises(ControlRefused, match="terminals or system tools"):
            ctl.act("a1", "open_app", name=name)  # no system=True: the old blocklist rules
    assert launched == []
    assert backend.calls == []


def test_plain_open_app_still_works_for_normal_apps() -> None:
    clock = Clock()
    ctl, backend, launched = _control(clock)
    n = {"i": 0}

    def fg() -> dict[str, Any]:
        n["i"] += 1
        # the Start search UI is in front while open_app verifies the launcher and types;
        # the app itself is foreground by the post-Enter verification poll
        if n["i"] < 4:
            return {"title": "Search", "process": "SearchHost.exe"}
        return {"title": "Untitled - Notepad", "process": "notepad.exe"}

    ctl.hooks.foreground = fg
    res = ctl.act("a1", "open_app", name="Notepad")
    assert res["ok"] is True
    assert launched == []  # normal apps still go through the Start-menu keystroke path
    assert any(c[0] == "type_text" for c in backend.calls)


def test_allowlist_targets_are_hardcoded_and_shell_safe() -> None:
    assert set(SAFE_SYSTEM_TOOLS) == {
        "device-manager",
        "windows-update",
        "network",
        "bluetooth",
        "display",
        "sound",
        "apps",
        "storage",
        "about",
    }
    for key, target in SAFE_SYSTEM_TOOLS.items():
        assert key == key.lower() and " " not in key
        ok_path = target.lower().startswith("c:\\windows\\system32\\")
        ok_uri = target.startswith("ms-settings:")
        assert ok_path or ok_uri, f"{key}: {target!r} is not a System32 path or ms-settings: URI"
        assert all(c not in target for c in (";", "|", "&", "`", "$"))  # no command-line smuggling


def test_open_group_resolves_system_action() -> None:
    method, kw = groups.resolve("open", {"action": "system", "name": "Device Manager"})
    assert method == "act"
    assert kw == {"action": "open_app", "system": True, "name": "Device Manager"}
