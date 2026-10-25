"""v0.1.34: on a Mac the tray changes its status item (menu, icon, tooltip, hide) only on the main thread. macOS 27
kills the app (SIGTRAP in -[NSStatusItem setMenu:]) when a bus handler on a worker thread rebuilds the menu."""

from __future__ import annotations

import sys
import threading
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from away.app import tray


class FakeMainQueue:
    def __init__(self) -> None:
        self.queued: list[Any] = []

    def addOperationWithBlock_(self, fn: Any) -> None:
        self.queued.append(fn)

    def drain(self) -> None:
        while self.queued:
            self.queued.pop(0)()


@pytest.fixture
def main_queue(monkeypatch: pytest.MonkeyPatch) -> FakeMainQueue:
    q = FakeMainQueue()
    foundation = types.ModuleType("Foundation")
    foundation.NSOperationQueue = types.SimpleNamespace(mainQueue=lambda: q)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "Foundation", foundation)
    return q


def _on_worker(fn: Any) -> None:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            fn()
        except BaseException as exc:  # surfaced on the test thread, not lost in the worker
            errors.append(exc)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if errors:
        raise errors[0]


def test_off_the_mac_or_on_the_main_thread_it_runs_now(main_queue: FakeMainQueue) -> None:
    ran: list[str] = []
    tray.on_main_thread(lambda: ran.append("main"), platform="darwin")
    _on_worker(lambda: tray.on_main_thread(lambda: ran.append("win"), platform="win32"))
    _on_worker(lambda: tray.on_main_thread(lambda: ran.append("linux"), platform="linux"))
    assert ran == ["main", "win", "linux"] and not main_queue.queued


def test_on_a_mac_a_worker_thread_queues_it_to_the_main_thread(main_queue: FakeMainQueue) -> None:
    ran: list[str] = []
    _on_worker(lambda: tray.on_main_thread(lambda: ran.append("x"), platform="darwin"))
    assert ran == [] and len(main_queue.queued) == 1
    main_queue.drain()
    assert ran == ["x"]


def _callbacks() -> tuple[Any, Any]:
    app = MagicMock()
    app.pipeline.paused = False
    cb = tray.TrayMenuCallbacks(tray.TrayController(app), app, MagicMock(), tray.TrayGuard())
    icon = MagicMock()
    cb.icon = icon
    return cb, icon


def test_a_bus_event_on_a_worker_never_touches_the_mac_status_item(
    monkeypatch: pytest.MonkeyPatch, main_queue: FakeMainQueue
) -> None:
    monkeypatch.setattr(tray.sys, "platform", "darwin")
    monkeypatch.setattr(tray, "tray_icon_image", lambda *a, **k: "img")
    cb, icon = _callbacks()
    _on_worker(lambda: cb.sync_icon(icon))  # what "voice.state" and friends do
    _on_worker(cb.safe_update_menu)  # the update manager's on_change
    assert not icon.update_menu.called and main_queue.queued
    main_queue.drain()
    assert icon.update_menu.call_count == 2 and icon.icon == "img"


def test_a_queued_change_after_quit_is_dropped(monkeypatch: pytest.MonkeyPatch, main_queue: FakeMainQueue) -> None:
    monkeypatch.setattr(tray.sys, "platform", "darwin")
    cb, icon = _callbacks()
    _on_worker(cb.safe_update_menu)
    cb.guard.stop(icon)
    main_queue.drain()
    assert not icon.update_menu.called and icon.stop.called


def test_an_update_quit_hides_the_icon_on_the_main_thread(
    monkeypatch: pytest.MonkeyPatch, main_queue: FakeMainQueue
) -> None:
    monkeypatch.setattr(tray.sys, "platform", "darwin")
    cb, icon = _callbacks()
    _on_worker(lambda: cb.on_quit(icon, None))  # quit_for_update runs it on the update worker
    assert not icon.stop.called
    main_queue.drain()
    assert icon.stop.called


def test_elsewhere_the_icon_still_changes_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tray.sys, "platform", "win32")
    cb, icon = _callbacks()
    _on_worker(cb.safe_update_menu)
    assert icon.update_menu.called
