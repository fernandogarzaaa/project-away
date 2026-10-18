"""Agent Link (v0.1.28, proposal #24): hybrid grounding.

UI Automation gives exact element bounds for free; raw pixel grounding is the weakest link. ``ui_query``
lists the foreground window's interactable elements with short ids, and pointer/keyboard actions accept an
``element_id`` that resolves to bounds, then act exactly as with coordinates.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element
from away.agentlink.groups import resolve
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _tree() -> Element:
    send = Element("Send", "ButtonControl", rect=(100, 100, 140, 120))
    subject = Element("Subject", "EditControl", rect=(0, 0, 50, 20))
    hidden = Element("Pane", "PaneControl", rect=(0, 0, 5, 5), enabled=False)  # not interactable: skipped
    root = Element("Mail", "WindowControl", rect=(0, 0, 800, 600))
    root.children = [send, subject, hidden]
    return root


class FakeUIA:
    """A UI Automation source with a fixed tree: ids come out e1=Mail, e2=Send, e3=Subject."""

    available = True
    reason = ""

    def __init__(self, root: Element | None = None, focused: Element | None = None) -> None:
        self._root = root if root is not None else _tree()
        self._focused = focused

    def tree(self, max_depth: int = 3, max_nodes: int = 200) -> Element | None:
        return self._root

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        return None

    def focused(self) -> Element | None:
        return self._focused


def _control(
    clock: Clock,
    answers: list[str] | None = None,
    uia: Any = None,
    fg: dict[str, Any] | None = None,
) -> tuple[ComputerControl, RecordingBackend, list[dict[str, Any]]]:
    backend = RecordingBackend()
    asked: list[dict[str, Any]] = []
    queue = list(answers or ["Allow"])

    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        asked.append({"question": question, "options": options, **kw})
        return {"answer": queue.pop(0) if queue else None, "method": "click"}

    hooks = ControlHooks(
        backend=backend,
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: fg,
        grab=lambda: None,
        sleep=lambda _s: None,
        **({"uia": uia} if uia is not None else {}),
    )
    ctl = ComputerControl(hooks, ask, ttl_s=1800, rate_per_s=100, clock=clock)
    return ctl, backend, asked


# ----------------------------------------------------------------------------------- ui_query
def test_ui_query_lists_elements_with_ids_names_roles_and_bounds() -> None:
    clock = Clock()
    ctl, _, _ = _control(clock, uia=FakeUIA(), fg={"title": "Mail", "process": "mail.exe"})
    res = ctl.ui_query("a1")
    assert res["available"] is True
    assert res["count"] == 3  # the disabled pane is not interactable
    assert res["window"] == {"title": "Mail", "process": "mail.exe"}
    assert res["elements"] == [
        {"id": "e1", "name": "Mail", "role": "WindowControl", "bounds": [0, 0, 800, 600]},
        {"id": "e2", "name": "Send", "role": "ButtonControl", "bounds": [100, 100, 140, 120]},
        {"id": "e3", "name": "Subject", "role": "EditControl", "bounds": [0, 0, 50, 20]},
    ]


def test_ui_query_degrades_when_uia_is_missing() -> None:
    clock = Clock()
    ctl, _, _ = _control(clock)  # default hooks: NullUIA
    res = ctl.ui_query("a1")
    assert res["available"] is False and res["reason"]


def test_ui_query_needs_no_grant_but_the_service_needs_control_permission(core: Any) -> None:
    from away.agentlink.service import Forbidden

    reader, _ = core.registry.issue("Eve", ["read"], 3600)
    with pytest.raises(Forbidden, match="Let it use my computer"):
        core.service.ui_query(reader)
    agent, _ = core.registry.issue("Claude", ["read", "control"], 3600)
    res = core.service.ui_query(agent)  # no control grant: reading the tree is not acting
    assert res["available"] is False  # the core fixture has no UI Automation
    core.service.control.hooks.uia = FakeUIA()
    res = core.service.ui_query(agent)
    assert res["available"] is True and res["count"] == 3


def test_ids_expire_on_requery() -> None:
    clock = Clock()
    uia = FakeUIA()
    ctl, _, _ = _control(clock, ["Allow"], uia=uia)
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "click", element_id="e3")  # the old Subject
    cancel = Element("Cancel", "ButtonControl", rect=(0, 0, 10, 10))
    root = Element("Dialog", "WindowControl", rect=(0, 0, 40, 40), children=[cancel])
    uia._root = root
    ctl.ui_query("a1")
    with pytest.raises(ControlRefused, match="unknown element id"):
        ctl.act("a1", "click", element_id="e3")  # the old Subject is gone; ids are per-query
    ctl.act("a1", "click", element_id="e2")  # Cancel is now e2


# ------------------------------------------------------------------ id resolution in pointer
def test_click_by_element_id_uses_exact_bounds() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "click", element_id="e3")  # Subject: named and harmless, no question asked
    assert len(asked) == 1 and asked[0]["kind"] == "approve"  # only the control request asked
    assert backend.calls[-2:] == [("move", (25, 10)), ("click", (1,))]


def test_unknown_element_id_is_refused() -> None:
    clock = Clock()
    ctl, _, _ = _control(clock, ["Allow"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    with pytest.raises(ControlRefused, match="unknown element id"):
        ctl.act("a1", "click", element_id="e99")


def test_element_id_beats_name_and_coordinates() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "click", x=1, y=2, element="Whatever", element_id="e3")
    assert backend.calls[-2:] == [("move", (25, 10)), ("click", (1,))]


def test_risky_element_id_still_asks() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Allow"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "click", element_id="e2")  # Send: exact bounds do not skip the risky-click question
    assert asked[-1]["kind"] == "confirm"
    assert backend.calls[-2:] == [("move", (120, 110)), ("click", (1,))]


def test_drag_to_element_id() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow", "Allow"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "drag", element_id="e3", to_element_id="e2")  # dropping on Send asks
    assert backend.calls[0] == ("move", (25, 10))
    assert backend.calls[1] == ("press", ())
    assert backend.calls[-1] == ("release", ())


def test_scroll_to_element_id_moves_there_first() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "scroll", dy=-3.0, element_id="e3")
    assert backend.calls[:2] == [("move", (25, 10)), ("scroll", (-3.0,))]


# ----------------------------------------------------------------- id resolution in keyboard
def test_type_with_element_id_clicks_the_field_first() -> None:
    clock = Clock()
    uia = FakeUIA(focused=Element("Subject", "EditControl", rect=(0, 0, 50, 20)))
    ctl, backend, _ = _control(clock, ["Allow"], uia=uia)
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "type_text", text="hello", element_id="e3")
    assert backend.calls[0] == ("move", (25, 10))
    assert backend.calls[1] == ("click", (1,))
    assert ("type_text", ("hello",)) in backend.calls


def test_press_keys_with_element_id_focuses_first() -> None:
    clock = Clock()
    uia = FakeUIA(focused=Element("Subject", "EditControl", rect=(0, 0, 50, 20)))
    ctl, backend, _ = _control(clock, ["Allow"], uia=uia)
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "press_keys", keys="a", element_id="e3")
    assert backend.calls[:3] == [("move", (25, 10)), ("click", (1,)), ("key", ("a",))]


# ------------------------------------------------------------------------------------ groups
def test_groups_pass_element_ids_through() -> None:
    assert resolve("pointer", {"action": "click", "element_id": "e1"}) == (
        "act",
        {"action": "click", "element_id": "e1"},
    )
    assert resolve("pointer", {"action": "drag", "element_id": "e1", "to_element_id": "e2"}) == (
        "act",
        {"action": "drag", "element_id": "e1", "to_element_id": "e2"},
    )
    assert resolve("pointer", {"action": "scroll", "dy": -3.0, "element_id": "e1"}) == (
        "act",
        {"action": "scroll", "dy": -3.0, "element_id": "e1"},
    )
    assert resolve("keyboard", {"action": "type", "text": "hi", "element_id": "e1"}) == (
        "act",
        {"action": "type_text", "text": "hi", "element_id": "e1"},
    )
    assert resolve("keyboard", {"action": "press", "keys": "enter", "element_id": "e1"}) == (
        "act",
        {"action": "press_keys", "keys": "enter", "element_id": "e1"},
    )


# --------------------------------------------------------------------------------- mcp + rest
def test_mcp_has_ui_query_and_passes_element_ids() -> None:
    pytest.importorskip("mcp")
    from away.agentlink import mcp_tools

    calls: list[tuple[str, dict[str, Any]]] = []

    class Recorder:
        def call(self, ctx: Any, method: str, **kw: Any) -> Any:
            calls.append((method, kw))
            return {"ok": True}

    mcp = mcp_tools.build(Recorder())
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "ui_query" in names
    asyncio.run(mcp.call_tool("ui_query", {}))
    assert calls[-1] == ("ui_query", {})
    asyncio.run(mcp.call_tool("pointer", {"action": "click", "element_id": "e1"}))
    assert calls[-1] == ("act", {"action": "click", "element_id": "e1"})
    asyncio.run(mcp.call_tool("keyboard", {"action": "type", "text": "hi", "element_id": "e2"}))
    assert calls[-1] == ("act", {"action": "type_text", "text": "hi", "element_id": "e2"})
    doc = next(t for t in asyncio.run(mcp.list_tools()) if t.name == "ui_query").description
    assert doc and "element_id" in doc


def test_rest_knows_ui_query() -> None:
    from away.agentlink import api

    assert api.TOOLS["ui_query"][2] == "/v1/control/ui_query"
    enum = api.openapi()["paths"]["/v1/rpc/{tool}"]["post"]["parameters"][0]["schema"]["enum"]
    assert "ui_query" in enum


@pytest.fixture
def core(tmp_path: Path) -> Any:
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
        flash=lambda _t: None,
        camera_frame=lambda _cam: None,
        screen_grab=lambda: None,
    )
    c.prompts.polite = lambda: True
    return c


# ------------------------------------------------- stale-id hardening (audit: risk D)
class MovedUIA(FakeUIA):
    """element_at reports a different control at the old spot: the queried element moved or is gone."""

    def element_at(self, x: int, y: int) -> Element | None:
        return Element("Delete", "ButtonControl", rect=(x - 5, y - 5, x + 5, y + 5))


def test_element_id_refused_after_foreground_window_changes() -> None:
    clock = Clock()
    fg = {"title": "Mail", "process": "mail.exe", "class": "MailWnd", "rect": [0, 0, 800, 600]}
    ctl, backend, _ = _control(clock, ["Allow"], uia=FakeUIA(), fg=fg)
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")  # e2 = Send at (120, 110) in Mail
    fg.update({"title": "Browser", "process": "browser.exe", "class": "BrowserWnd", "rect": [0, 0, 1024, 768]})
    with pytest.raises(ControlRefused, match="foreground window changed"):
        ctl.act("a1", "click", element_id="e2")  # must not click the stale spot in the new window
    assert backend.calls == []


def test_element_ids_are_isolated_per_agent() -> None:
    clock = Clock()
    uia = FakeUIA()
    fg = {"title": "Mail", "process": "mail.exe"}
    ctl, backend, _ = _control(clock, ["Allow", "Allow"], uia=uia, fg=fg)
    ctl.ui_query("a1")  # a1 sees Mail: e2 = Send at (120, 110)
    cancel = Element("Cancel", "ButtonControl", rect=(0, 0, 10, 10))
    uia._root = Element("Dialog", "WindowControl", rect=(0, 0, 40, 40), children=[cancel])
    ctl.ui_query("a2")  # a2 sees Dialog: e2 = Cancel at (5, 5)
    ctl.request("a1", "Claude")
    ctl.act("a1", "click", element_id="e2")  # a1's e2 is still Send, not a2's Cancel
    assert backend.calls[-2:] == [("move", (120, 110)), ("click", (1,))]


def test_element_id_refused_when_element_moved() -> None:
    clock = Clock()
    fg = {"title": "Mail", "process": "mail.exe"}
    ctl, backend, _ = _control(clock, ["Allow"], uia=MovedUIA(), fg=fg)
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")  # e3 = Subject at (25, 10)
    with pytest.raises(ControlRefused, match="gone or has moved"):
        ctl.act("a1", "click", element_id="e3")  # something else is at that spot now
    assert backend.calls == []


def test_element_id_without_query_is_refused() -> None:
    clock = Clock()
    ctl, _, _ = _control(clock, ["Allow"], uia=FakeUIA(), fg={"title": "Mail", "process": "mail.exe"})
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="unknown element id"):
        ctl.act("a1", "click", element_id="e1")  # this agent never queried


def test_element_id_still_works_when_window_unchanged() -> None:
    clock = Clock()
    fg = {"title": "Mail", "process": "mail.exe", "class": "MailWnd", "rect": [0, 0, 800, 600]}
    ctl, backend, _ = _control(clock, ["Allow"], uia=FakeUIA(), fg=fg)
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    ctl.act("a1", "click", element_id="e3")  # Subject: same window, still there
    assert backend.calls[-2:] == [("move", (25, 10)), ("click", (1,))]
