"""v0.1.29: Settings merges the "Live agent" and "Agent Link" tabs into one "Agents" tab.

Pure composition: both panels keep their models and their builders; the new container
(:mod:`away.app.agents_tab`) only hosts them in one scrollable tab. Tk-free: the widget
calls are faked, the real models are constructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_tab_order_has_single_agents_tab() -> None:
    from away.app.settings_gui import TAB_ORDER

    assert "Agents" in TAB_ORDER
    assert "Agent Link" not in TAB_ORDER
    assert "Live agent" not in TAB_ORDER
    # it sits where the agent tabs were: with the everyday tabs, before Cameras
    assert TAB_ORDER.index("Agents") < TAB_ORDER.index("Cameras")


def test_old_tab_names_resolve_to_agents() -> None:
    from away.app.settings_gui import resolve_tab_alias

    assert resolve_tab_alias("Agent Link") == "Agents"
    assert resolve_tab_alias("Live agent") == "Agents"
    assert resolve_tab_alias("agents") == "agents"
    assert resolve_tab_alias("AGENTS") == "AGENTS"  # unknown names come back unchanged
    assert resolve_tab_alias("studio") == "studio"


class _FakeFrame:
    def pack(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FakeTtk:
    def Label(self, *args: Any, **kwargs: Any) -> _FakeFrame:
        return _FakeFrame()

    def LabelFrame(self, *args: Any, **kwargs: Any) -> _FakeFrame:
        return _FakeFrame()


class _FakeWin:
    tk = _FakeTtk()
    ttk = _FakeTtk()

    def __init__(self) -> None:
        self.tabs_made: list[str] = []

    def _scroll_tab(self, nb: Any, text: str, padding: int = 8) -> _FakeFrame:
        self.tabs_made.append(text)
        return _FakeFrame()


def test_agents_tab_composes_both_panels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from away.agentlink import gui as agent_gui
    from away.app import agents_tab
    from away.liveagent import gui as live_gui

    calls: dict[str, Any] = {}

    def fake_build_panel(win: Any, parent: Any) -> str:
        calls["live_parent"] = parent
        return "live-panel"

    class FakeLinkTab:
        def __init__(self, win: Any, parent: Any, model: Any) -> None:
            calls["link_parent"] = parent
            calls["link_model"] = model

    monkeypatch.setattr(live_gui, "build_panel", fake_build_panel)
    monkeypatch.setattr(agent_gui, "AgentLinkTab", FakeLinkTab)

    win = _FakeWin()
    agents_tab.build_tab(win, object())

    assert win.tabs_made == ["Agents"]  # one tab, not two
    assert calls["live_parent"] is not None
    assert calls["link_parent"] is not None
    assert calls["live_parent"] is not calls["link_parent"]  # separate sections
    assert calls["link_model"] is not None
    assert win.live_agent_panel == "live-panel"  # SettingsWindow keeps this attribute


def test_both_models_still_constructible(tmp_path: Path) -> None:
    from away.agentlink.gui_model import AgentLinkModel
    from away.liveagent.gui import LivePanel

    assert AgentLinkModel(data_dir=tmp_path / "al") is not None
    assert LivePanel(data_dir=tmp_path / "la") is not None
