"""Proposal #19 (v0.1.28): the tray icon gets a state dot while an AI agent holds snapshots/screen/control
and was recently active, and clicking the tray icon shows one line per agent ("who can see what") with an
Open Agent Link button in the popup."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from away.agentlink import perms
from away.agentlink.auth import Agent
from away.agentlink.host import (
    WATCHER_ACTIVE_S,
    relative_time_ago,
    watcher_dot,
    watchers,
)
from away.app.tray import (
    AGENT_DOT_COLOR,
    POPUP_MENU_LINES,
    TrayController,
    agent_popup_lines,
    agent_popup_menu_items,
    make_icon_image,
)


def _agent(name, scopes, last_seen, aid="a1", expired=False):
    now = time.time()
    return Agent(
        id=aid,
        name=name,
        kind="remote",
        scopes=list(scopes),
        token_hash="h",
        created=now - 3600,
        expires=now - 10 if expired else now + 3600,
        last_seen=last_seen,
    )


class _Registry:
    """Mirrors AgentRegistry.list(): only agents with access now, unless include_inactive."""

    def __init__(self, agents):
        self._agents = agents

    def list(self, include_inactive=False):
        now = time.time()
        return [a for a in self._agents if include_inactive or a.active(now)]


class _Control:
    def __init__(self, agent_id=None):
        self._agent_id = agent_id

    def status(self):
        return {
            "active": self._agent_id is not None,
            "agent": "X",
            "agent_id": self._agent_id,
            "paused": False,
        }


class _Host:
    def __init__(self, agents, control_agent=None):
        self.core = SimpleNamespace(registry=_Registry(agents), control=_Control(control_agent))


# ---------------------------------------------------------------- relative time


def test_relative_time_ago():
    assert relative_time_ago(5) == "just now"
    assert relative_time_ago(59) == "just now"
    assert relative_time_ago(60) == "1 min ago"
    assert relative_time_ago(120) == "2 min ago"
    assert relative_time_ago(3600) == "1 hour ago"
    assert relative_time_ago(7200) == "2 hours ago"
    assert relative_time_ago(90000) == "yesterday"
    assert relative_time_ago(3 * 86400) == "3 days ago"


# ---------------------------------------------------------------- the state dot


def test_watcher_dot_on_for_recent_screen_agent():
    h = _Host([_agent("Claude", [perms.SCREEN], time.time() - 10)])
    assert watcher_dot(h) is True


def test_watcher_dot_on_for_snapshots_and_control():
    now = time.time()
    h = _Host(
        [
            _agent("A", [perms.SNAPSHOTS], now - 5, aid="a"),
            _agent("B", [perms.CONTROL], now - 5, aid="b"),
        ]
    )
    assert watcher_dot(h) is True


def test_watcher_dot_off_for_read_only_agent():
    h = _Host([_agent("Claude", [perms.READ, perms.DIAGNOSE], time.time() - 10)])
    assert watcher_dot(h) is False


def test_watcher_dot_off_when_stale():
    h = _Host([_agent("Claude", [perms.SCREEN], time.time() - WATCHER_ACTIVE_S - 1)])
    assert watcher_dot(h) is False


def test_watcher_dot_off_when_expired_or_no_host():
    now = time.time()
    h = _Host([_agent("Claude", [perms.SCREEN], now - 10, expired=True)])
    assert watcher_dot(h) is False
    assert watcher_dot(None) is False


# ---------------------------------------------------------------- the per-agent lines


def test_watchers_capabilities_in_plain_words():
    now = time.time()
    h = _Host([_agent("Claude", [perms.SCREEN, perms.CONTROL, perms.TUNE], now - 120)])
    (w,) = watchers(h, now=now)
    assert w["name"] == "Claude"
    assert w["capabilities"] == ["can see your screen", "can use your computer", "can change your settings"]
    assert w["active"] == "2 min ago"
    assert w["connected"] is False
    assert w["controlling"] is False


def test_watchers_marks_controlling_agent():
    now = time.time()
    h = _Host([_agent("Claude", [perms.CONTROL], now - 5)], control_agent="a1")
    (w,) = watchers(h, now=now)
    assert w["controlling"] is True
    assert w["connected"] is True


def test_watchers_sorts_controlling_first():
    now = time.time()
    h = _Host(
        [
            _agent("Idle", [perms.SCREEN], now - 5, aid="idle"),
            _agent("Busy", [perms.CONTROL], now - 50, aid="busy"),
        ],
        control_agent="busy",
    )
    names = [w["name"] for w in watchers(h, now=now)]
    assert names == ["Busy", "Idle"]


def test_watchers_empty_without_host():
    assert watchers(None) == []


def test_agent_popup_lines_match_proposal_format():
    now = time.time()
    h = _Host([_agent("Claude", [perms.SCREEN], now - 120)])
    assert agent_popup_lines(h, now=now) == ["Claude — can see your screen · used your computer 2 min ago"]


def test_agent_popup_lines_controlling_agent():
    now = time.time()
    h = _Host([_agent("Claude", [perms.CONTROL], now - 5)], control_agent="a1")
    assert agent_popup_lines(h, now=now) == ["Claude — can use your computer · using your computer right now"]


def test_agent_popup_lines_empty_without_host():
    assert agent_popup_lines(None) == []


# ---------------------------------------------------------------- the icon dot


def test_make_icon_image_agent_dot():
    pytest.importorskip("PIL")
    img = make_icon_image(False, 64, agent_dot=True)
    r = 64 // 8
    cx, cy = 64 - r - 2, r + 2
    assert img.getpixel((cx, cy))[:3] == AGENT_DOT_COLOR[:3]
    plain = make_icon_image(False, 64)
    assert plain.getpixel((cx, cy))[3] == 0  # no dot: transparent corner


# ---------------------------------------------------------------- the controller


def test_tray_controller_agent_dot_and_popup_lines():
    now = time.time()
    ctl = TrayController(SimpleNamespace(agent_link=_Host([_agent("Claude", [perms.SCREEN], now - 10)])))
    assert ctl.agent_dot() is True
    assert ctl.popup_lines() == ["Claude — can see your screen · used your computer just now"]
    cmd = TrayController.agent_link_command()
    assert cmd[-3:] == ["settings", "--tab", "Agent Link"]


def test_tray_controller_without_agent_link():
    ctl = TrayController(SimpleNamespace())
    assert ctl.agent_dot() is False
    assert ctl.popup_lines() == []


def test_agent_popup_menu_items():
    try:
        import pystray
    except Exception as exc:  # headless sandbox/CI: no tray backend (Xlib needs a display)
        pytest.skip(f"pystray backend unavailable: {exc}")
    now = time.time()
    ctl = TrayController(
        SimpleNamespace(
            agent_link=_Host(
                [
                    _agent("Claude", [perms.SCREEN], now - 10, aid="a1"),
                    _agent("Hark", [perms.READ], now - 20, aid="a2"),
                ]
            )
        )
    )
    items = agent_popup_menu_items(pystray, ctl)
    assert len(items) == POPUP_MENU_LINES + 1  # one slot per agent line, then "Who can see what..."
    assert "Who can see what (2)" in items[-1].text(items[-1])
    # no agents: every slot is hidden
    empty = TrayController(SimpleNamespace())
    assert all(not it.visible(it) for it in agent_popup_menu_items(pystray, empty))
