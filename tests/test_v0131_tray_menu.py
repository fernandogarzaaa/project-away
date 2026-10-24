"""The tray menu must be accepted by the real pystray (v0.1.31).

v0.1.29 wrapped every menu callback in ``menu_safe`` with a bare ``*args`` wrapper. pystray decides how to call a
menu action from ``action.__code__.co_argcount`` (minus one for a bound method) and raises ValueError for anything
else, so the app crashed building its menu on start-up ("Failed to execute script 'away_entry'"). The tests of the
menu used a fake pystray, which accepts anything.
"""

from __future__ import annotations

import inspect
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from away.app import tray


def pystray_argcount(action: Any) -> int:
    """What pystray's ``MenuItem._assert_action`` computes (pystray 0.19)."""
    return action.__code__.co_argcount - (1 if inspect.ismethod(action) else 0)


def _callbacks() -> tuple[Any, Any, Any]:
    app = MagicMock()
    ctl = tray.TrayController(app)
    cb = tray.TrayMenuCallbacks(ctl, app, MagicMock(), tray.TrayGuard())
    return app, ctl, cb


def test_every_menu_callback_takes_the_icon_and_the_item_for_pystray() -> None:
    _app, _ctl, cb = _callbacks()
    # the menu actions are the menu_safe ones (on_gate_hint is an event handler, called with one argument)
    names = [n for n in dir(cb) if n.startswith("on_") and hasattr(getattr(cb, n), "__wrapped__")]
    assert "on_pause" in names and "on_quit" in names and len(names) >= 15
    for name in names:
        assert pystray_argcount(getattr(cb, name)) == 2, name  # 0, 1 or 2 are accepted; the callbacks take both


def test_menu_safe_keeps_the_argument_count_and_still_guards() -> None:
    calls: list[Any] = []

    @tray.menu_safe("plain")
    def plain(icon: Any, item: Any) -> str:
        calls.append((icon, item))
        return "ran"

    @tray.menu_safe("boom")
    def boom(icon: Any, item: Any) -> None:
        raise RuntimeError("COM error")

    class C:
        @tray.menu_safe("method")
        def act(self, icon: Any, item: Any) -> str:
            return f"{icon}-{item}"

    assert pystray_argcount(plain) == 2 and pystray_argcount(C().act) == 2
    assert plain("i", "m") == "ran" and calls == [("i", "m")]
    assert boom("i", "m") is None  # an escaping error must not take the tray's message loop down
    assert C().act("i", "m") == "i-m"
    assert plain.__name__ == "plain"  # functools.wraps still applies
    with pytest.raises(TypeError, match="takes 4 arguments"):

        @tray.menu_safe("too many")
        def nope(a: Any, b: Any, c: Any, d: Any) -> None: ...


def test_the_real_pystray_accepts_the_whole_tray_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYSTRAY_BACKEND", "dummy")  # no display needed; the menu code is backend independent
    pystray = pytest.importorskip("pystray")
    if os.environ.get("PYSTRAY_BACKEND") != "dummy":  # pragma: no cover
        pytest.skip("pystray was imported with another backend")
    app, ctl, cb = _callbacks()
    from away.liveagent.integrate import menu_items

    # the real live-agent item too (a stub there would hide an argument-count problem of its own)
    menu = tray.build_tray_menu(pystray, ctl, app, cb, MagicMock(), menu_items, lambda tab: None)
    items = list(menu.items)
    assert len(items) > 20
    icon = MagicMock()
    pause = next(i for i in items if i.default)
    assert pause.text == "Pause"
    before = ctl.paused
    pause(icon)  # a click: pystray calls the action with (icon, item)
    assert ctl.paused is not before
