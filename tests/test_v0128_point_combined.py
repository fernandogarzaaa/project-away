"""Agent Link: combined _point contract (set-of-mark + element_id), proposals #25 + #24.

_point resolves in this order: (1) set-of-mark references like "mark:1" from the agent's last
annotated screenshot, (2) element_id from the agent's last ui_query with staleness revalidation,
(3) fuzzy element-name matching, (4) raw coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from away import config as C
from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element
from away.agentlink.host import build_core
from away.agentlink.prompts import NullUI
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
    """Fixed tree: e1=Mail (0,0,800,600), e2=Send (100,100,140,120)."""

    available = True
    reason = ""

    def __init__(self, root: Element | None = None) -> None:
        self._root = root if root is not None else _tree()

    def tree(self, max_depth: int = 3, max_nodes: int = 200) -> Element | None:
        return self._root

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        # For revalidation: return the Send button if coordinates match its center
        send = Element("Send", "ButtonControl", rect=(100, 100, 140, 120))
        cx, cy = (100 + 140) // 2, (100 + 120) // 2
        if abs(x - cx) < 5 and abs(y - cy) < 5:
            return send
        return None

    def focused(self) -> Element | None:
        return None


def _control(fg: dict[str, Any] | None = None) -> ComputerControl:
    clock = Clock()
    backend = RecordingBackend()
    hooks = ControlHooks(
        backend=backend,
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: fg,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=FakeUIA(),
    )
    return ComputerControl(hooks, lambda *a, **k: {"answer": "Allow", "method": "click"}, clock=clock)


@pytest.fixture
def core(tmp_path: Path):
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    ui = NullUI()
    c = build_core(
        tmp_path,
        cfg=lambda: cfg,
        settings=lambda: cfg.agent_link,
        config_path=lambda: cfgp,
        ui=ui,
        flash=lambda _m: None,
        camera_frame=lambda _cam: None,
        screen_grab=lambda: None,
    )
    c.prompts.polite = lambda: True
    return c


def _screen_with_boxes() -> object:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1920, 1080), (245, 245, 245))
    d = ImageDraw.Draw(img)
    for x, y, w, h in ((200, 150, 300, 120), (700, 150, 300, 120)):
        d.rectangle([x, y, x + w, y + h], outline=(20, 20, 20), width=4)
    return img


# ------------------------------------------------------------------ mark references


def test_mark_reference_resolves_with_new_signature(core) -> None:
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    _jpeg, meta = core.service.get_screen(a, annotate=True)
    first = meta["annotations"][0]
    x, y, label = core.service.control._point(a.id, None, None, "mark:1", None, False)
    assert (x, y) == (first["center"]["x"], first["center"]["y"])
    assert label == "mark 1"


def test_mark_reference_wins_over_element_id(core) -> None:
    """When both a mark and an element_id are given, the mark resolves first."""
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen", "control"], 3600)
    _jpeg, meta = core.service.get_screen(a, annotate=True)
    first = meta["annotations"][0]
    x, y, label = core.service.control._point(a.id, None, None, "mark:1", "e999", False)
    assert (x, y) == (first["center"]["x"], first["center"]["y"])
    assert label == "mark 1"


def test_unknown_mark_still_refused_with_new_signature(core) -> None:
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    core.service.get_screen(a, annotate=True)
    with pytest.raises(ControlRefused, match="no annotation 999"):
        core.service.control._point(a.id, None, None, "mark:999", None, False)


# ------------------------------------------------------------------ element_id path


def test_fresh_element_id_resolves() -> None:
    ctl = _control(fg={"title": "Mail", "process": "mail.exe", "class": "Win", "rect": [0, 0, 800, 600]})
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    assert res["available"] is True
    x, y, label = ctl._point("a1", None, None, None, "e2", False)
    assert (x, y) == (120, 110)  # center of the Send button rect
    assert label == "Send"


def test_stale_element_id_refused_after_window_change() -> None:
    ctl = _control(fg={"title": "Mail", "process": "mail.exe", "class": "Win", "rect": [0, 0, 800, 600]})
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    # the foreground window changes: the id must stop resolving
    ctl.hooks.foreground = lambda: {
        "title": "Browser",
        "process": "browser.exe",
        "class": "Win",
        "rect": [0, 0, 800, 600],
    }  # type: ignore[method-assign]
    with pytest.raises(ControlRefused, match="foreground window changed"):
        ctl._point("a1", None, None, None, "e2", False)


def test_element_id_of_other_agent_refused() -> None:
    ctl = _control(fg={"title": "Mail", "process": "mail.exe", "class": "Win", "rect": [0, 0, 800, 600]})
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    with pytest.raises(ControlRefused, match="call ui_query first"):
        ctl._point("a2", None, None, None, "e2", False)


def test_unknown_element_id_refused() -> None:
    ctl = _control(fg={"title": "Mail", "process": "mail.exe", "class": "Win", "rect": [0, 0, 800, 600]})
    ctl.request("a1", "Claude")
    ctl.ui_query("a1")
    with pytest.raises(ControlRefused, match="unknown element id"):
        ctl._point("a1", None, None, None, "e99", False)


def test_ui_query_unavailable_without_uia() -> None:
    ctl = _control(fg={"title": "Mail", "process": "mail.exe"})
    ctl.request("a1", "Claude")
    ctl.hooks.uia = type("NullUIA", (), {"available": False, "reason": "nope"})()
    res = ctl.ui_query("a1")
    assert res["available"] is False
