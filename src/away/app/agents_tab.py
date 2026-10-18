"""Settings > Agents: the Live agent and Agent Link panels in one tab (v0.1.29).

Pure composition: the "Live agent" tab (the on-device voice agent) and the "Agent Link" tab
(remote agents connecting in through pairing links) were two tabs for one idea, AI agents that
work on this PC. This module only hosts the existing panels; every control lives where it
always did, in :mod:`away.liveagent.gui` and :mod:`away.agentlink.gui`, whose models stay
untouched and unit-testable.
"""

from __future__ import annotations

from typing import Any


def build_tab(win: Any, nb: Any) -> Any:  # pragma: no cover - widgets; smoke-tested under Xvfb
    """One scrollable "Agents" tab: "On this PC" (live agent) above "Connected remotely" (agent link).

    Sets ``win.live_agent_panel`` like the old Live agent tab did, and returns the Agent Link
    tab object.
    """
    from away.agentlink import gui as agent_gui
    from away.agentlink.gui_model import AgentLinkModel
    from away.liveagent import gui as live_gui

    ttk = win.ttk
    f = win._scroll_tab(nb, "Agents")
    ttk.Label(f, text="Agents", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
    ttk.Label(
        f,
        text="The AI agents that work for you: the voice agent on this PC, and remote agents you connect.",
        foreground="#555",
        wraplength=620,
    ).pack(anchor="w", pady=(0, 6))

    on_pc = ttk.LabelFrame(f, text="On this PC", padding=6)
    on_pc.pack(fill="x", pady=6)
    win.live_agent_panel = live_gui.build_panel(win, on_pc)

    remote = ttk.LabelFrame(f, text="Connected remotely", padding=6)
    remote.pack(fill="x", pady=6)
    link_tab = agent_gui.AgentLinkTab(win, remote, AgentLinkModel())
    return link_tab
