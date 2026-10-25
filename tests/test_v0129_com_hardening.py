"""v0.1.29: harden COM usage against the fatal 0x80040155/0x80010108/0x8001010d crashes.

The runtime log (2026-10-12 to 2026-10-17) showed repeated Windows fatal exceptions with
``pointer/uia.py`` ``foreground_targets`` on the current thread, pystray tray callbacks,
and pycaw ``other_audio`` sampling in the dumps. Regression tests for the fixes:

* ``UIABackend`` holds one lock around every use of the shared COM objects (the lock used
  to guard creation only), and drops + re-creates them when a COM call reports a
  disconnect instead of staying wedged.
* ``pycaw_reader().read()`` guards ``AudioUtilities.GetAllSessions()`` itself.
* tray menu callbacks never let an exception escape onto pystray's message thread.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import types
from typing import Any

import pytest

from away.pointer import targets as PT
from away.pointer import uia as U


class FakeCOMError(Exception):
    """Stand-in for comtypes.COMError (e.g. RPC_E_DISCONNECTED 0x80010108)."""


class FakeElement:
    def __init__(self, name: Any = "OK") -> None:
        self.CachedName = name
        self.CachedControlType = 50000  # button
        self.CachedBoundingRectangle = (10, 20, 110, 40)
        self.focused = False
        self.fail_focus = False

    def SetFocus(self) -> None:
        if self.fail_focus:
            raise FakeCOMError("0x80010108 RPC_E_DISCONNECTED")
        self.focused = True


class FakeArray:
    def __init__(self, items: list[Any]) -> None:
        self.items = items
        self.Length = len(items)

    def GetElement(self, i: int) -> Any:
        return self.items[i]


class FakeUIA:
    """Duck-typed CUIAutomation. ``fail_next_handle`` makes the next ElementFromHandle raise."""

    def __init__(self) -> None:
        self.fail_next_handle = False
        self.return_none_array = False
        self.barrier: threading.Barrier | None = None
        self.overlapped = False
        self.conditions: list[tuple[int, Any]] = []
        self.cache_properties: list[int] = []

    def CreatePropertyCondition(self, prop: int, value: Any) -> Any:
        self.conditions.append((prop, value))
        return ("cond", prop, value)

    def CreateOrConditionFromArray(self, conds: list[Any]) -> Any:
        return ("or", len(conds))

    def CreateAndCondition(self, a: Any, b: Any) -> Any:
        return ("and", a, b)

    def CreateCacheRequest(self) -> Any:
        uia = self

        class FakeCache:
            def AddProperty(self, p: int) -> None:
                uia.cache_properties.append(p)

        return FakeCache()

    def ElementFromHandle(self, hwnd: int) -> Any:
        if self.fail_next_handle:
            self.fail_next_handle = False
            raise FakeCOMError("0x80010108 RPC_E_DISCONNECTED")
        return FakeRoot(self)


class FakeRoot:
    def __init__(self, uia: FakeUIA) -> None:
        self._uia = uia

    def FindAllBuildCache(self, scope: int, cond: Any, cache: Any) -> Any:
        if self._uia.return_none_array:
            return None
        # Overlap detector: two threads inside this COM call at once trip the barrier.
        barrier = self._uia.barrier
        if barrier is not None:
            try:
                barrier.wait(timeout=1.0)
                self._uia.overlapped = True
            except threading.BrokenBarrierError:
                pass
        return FakeArray([FakeElement()])


def install_fake_comtypes(monkeypatch: pytest.MonkeyPatch) -> list[FakeUIA]:
    """A fake ``comtypes`` module (with ``COMError``), like test_v0120_smart_pointer.py's."""
    created: list[FakeUIA] = []
    comtypes = types.ModuleType("comtypes")
    comtypes.COMError = FakeCOMError  # type: ignore[attr-defined]
    comtypes.COINIT_MULTITHREADED = 0  # type: ignore[attr-defined]
    comtypes.CoInitializeEx = lambda *a, **k: 0  # type: ignore[attr-defined]
    client = types.ModuleType("comtypes.client")

    def GetModule(name: str) -> Any:
        return types.SimpleNamespace(CUIAutomation="CLSID", IUIAutomation="IID")

    def CreateObject(clsid: Any, interface: Any = None) -> Any:
        uia = FakeUIA()
        created.append(uia)
        return uia

    client.GetModule = GetModule  # type: ignore[attr-defined]
    client.CreateObject = CreateObject  # type: ignore[attr-defined]
    comtypes.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comtypes", comtypes)
    monkeypatch.setitem(sys.modules, "comtypes.client", client)
    return created


def _backend(monkeypatch: pytest.MonkeyPatch) -> tuple[U.UIABackend, list[FakeUIA]]:
    created = install_fake_comtypes(monkeypatch)
    return U.UIABackend(foreground=lambda: 0x1234), created


def test_disconnect_drops_objects_and_next_call_recreates(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    assert [(t.name, t.kind) for t in backend.foreground_targets()] == [("OK", "button")]
    assert len(created) == 1
    created[0].fail_next_handle = True
    assert backend.foreground_targets() == []  # the disconnect never escapes
    assert backend._uia is None  # dropped, not wedged
    assert [(t.name, t.kind) for t in backend.foreground_targets()] == [("OK", "button")]
    assert len(created) == 2  # re-created on the next call


def test_concurrent_foreground_targets_are_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    backend.foreground_targets()  # create the shared objects once
    uia = created[0]
    uia.barrier = threading.Barrier(2)
    threads = [threading.Thread(target=backend.foreground_targets) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(10)
    assert not any(th.is_alive() for th in threads)
    assert not uia.overlapped  # the two COM walks never overlapped


def test_focus_disconnect_returns_false_and_drops_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    el = FakeElement()
    el.fail_focus = True
    target = PT.Target("OK", "button", (10.0, 20.0, 110.0, 40.0), el)
    assert backend.focus(target) is False
    assert backend._uia is None


def test_pycaw_reader_guards_get_all_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.voice import other_audio

    monkeypatch.setattr(other_audio.sys, "platform", "win32")

    class FakeAudioUtilities:
        @staticmethod
        def GetAllSessions() -> list[Any]:
            raise FakeCOMError("0x80010108 RPC_E_DISCONNECTED")

    parent = types.ModuleType("pycaw")
    fake = types.ModuleType("pycaw.pycaw")
    fake.AudioUtilities = FakeAudioUtilities  # type: ignore[attr-defined]
    fake.IAudioMeterInformation = object()  # type: ignore[attr-defined]
    parent.pycaw = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycaw", parent)
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", fake)
    reader = other_audio.pycaw_reader()
    assert reader is not None
    assert reader() == []  # the COM failure never escapes read()


def test_sample_records_reader_error_instead_of_raising() -> None:
    from away.voice import other_audio

    def boom() -> list[tuple[int, str, float]]:
        raise FakeCOMError("0x80010108 RPC_E_DISCONNECTED")

    meter = other_audio.OtherAudioMeter(reader=boom, own_pids=set())
    assert meter.sample() is False
    assert meter.error != ""
    assert not meter.available


def test_menu_callback_exception_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    from away.app import tray

    @tray.menu_safe("boom-item")
    def boom(_icon: Any, _item: Any) -> None:
        raise RuntimeError("COM blew up")

    with caplog.at_level(logging.WARNING, logger="away.tray"):
        assert boom(None, None) is None
    assert "boom-item" in caplog.text


def test_menu_callback_passthrough() -> None:
    from away.app import tray

    @tray.menu_safe("ok-item")
    def ok(_icon: Any, _item: Any) -> str:
        return "done"

    assert ok(None, None) == "done"


def test_com_error_types_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_com_error_types falls back to (OSError,) when comtypes cannot be imported."""
    monkeypatch.setitem(sys.modules, "comtypes", None)
    assert U._com_error_types() == (OSError,)


def test_com_error_types_invalid_comerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """_com_error_types falls back to (OSError,) when comtypes.COMError is not an exception type."""
    comtypes = types.ModuleType("comtypes")
    comtypes.COMError = "not-an-exception-type"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comtypes", comtypes)
    assert U._com_error_types() == (OSError,)


def test_thread_init_creates_objects_under_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    backend.thread_init()
    assert len(created) == 1
    assert backend._uia is not None


def test_ensure_initializes_com_apartment(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ensure calls _com_init (the mutant that drops it must change observable state)."""
    backend, created = _backend(monkeypatch)
    assert getattr(backend._local, "com", False) is False
    backend.foreground_targets()
    assert backend._local.com is True


def test_ensure_offscreen_condition_uses_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """The onscreen condition filters OUT offscreen elements (False, not True)."""
    backend, created = _backend(monkeypatch)
    backend.foreground_targets()
    uia = created[0]
    # (prop, value) pairs passed to CreatePropertyCondition
    offscreen = [c for c in uia.conditions if c[0] == U.UIA_IS_OFFSCREEN]
    assert offscreen == [(U.UIA_IS_OFFSCREEN, False)]


def test_ensure_adds_all_cache_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    backend.foreground_targets()
    uia = created[0]
    assert uia.cache_properties == [U.UIA_NAME, U.UIA_CONTROL_TYPE, U.UIA_BOUNDING_RECTANGLE]


def test_foreground_targets_none_array_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    backend.foreground_targets()  # create the shared objects
    created[0].return_none_array = True
    assert backend.foreground_targets() == []


def test_focus_none_ref_returns_false_without_touching_com(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    target = PT.Target("OK", "button", (10.0, 20.0, 110.0, 40.0), None)
    assert backend.focus(target) is False
    assert created == []  # _ensure never ran


def test_focus_success_creates_com_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    el = FakeElement()
    target = PT.Target("OK", "button", (10.0, 20.0, 110.0, 40.0), el)
    assert backend.focus(target) is True
    assert el.focused
    assert len(created) == 1  # the shared objects were created under the lock


def test_foreground_targets_no_hwnd_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_comtypes(monkeypatch)
    backend = U.UIABackend(foreground=lambda: 0)
    assert backend.foreground_targets() == []


def test_foreground_targets_none_root_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, created = _backend(monkeypatch)
    backend.foreground_targets()  # create the shared objects
    created[0].ElementFromHandle = lambda hwnd: None
    assert backend.foreground_targets() == []


def _install_fake_pycaw(monkeypatch: pytest.MonkeyPatch, sessions_fn: Any) -> None:
    from away.voice import other_audio

    monkeypatch.setattr(other_audio.sys, "platform", "win32")

    class FakeAudioUtilities:
        @staticmethod
        def GetAllSessions() -> list[Any]:
            return sessions_fn()

    parent = types.ModuleType("pycaw")
    fake = types.ModuleType("pycaw.pycaw")
    fake.AudioUtilities = FakeAudioUtilities  # type: ignore[attr-defined]
    fake.IAudioMeterInformation = object()  # type: ignore[attr-defined]
    parent.pycaw = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycaw", parent)
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", fake)


def test_pycaw_reader_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """read() returns (pid, name, peak) rows when the session list is healthy."""
    from away.voice import other_audio

    class FakeMeter:
        def GetPeakValue(self) -> float:
            return 0.5

    class FakeSession:
        ProcessId = 1234

        def __init__(self) -> None:
            self._ctl = self

        def QueryInterface(self, _iface: Any) -> Any:
            return FakeMeter()

        @property
        def Process(self) -> Any:
            return types.SimpleNamespace(name=lambda: "chrome.exe")

    def sessions() -> list[Any]:
        return [FakeSession()]

    _install_fake_pycaw(monkeypatch, sessions)
    reader = other_audio.pycaw_reader()
    assert reader is not None
    assert reader() == [(1234, "chrome.exe", 0.5)]


def test_pycaw_reader_import_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """pycaw_reader() is None when pycaw cannot be imported (the except branch)."""
    from away.voice import other_audio

    monkeypatch.setattr(other_audio.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", None)
    assert other_audio.pycaw_reader() is None


def test_pycaw_reader_bad_session_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session whose meter query fails is skipped; the good session is still read."""
    from away.voice import other_audio

    class FakeMeter:
        def GetPeakValue(self) -> float:
            return 0.5

    class BadSession:
        ProcessId = 111

        def __init__(self) -> None:
            self._ctl = self

        def QueryInterface(self, _iface: Any) -> Any:
            raise FakeCOMError("0x80010108")

    class GoodSession:
        ProcessId = 222

        def __init__(self) -> None:
            self._ctl = self

        def QueryInterface(self, _iface: Any) -> Any:
            return FakeMeter()

        @property
        def Process(self) -> Any:
            return types.SimpleNamespace(name=lambda: "music.exe")

    def sessions() -> list[Any]:
        return [BadSession(), GoodSession()]

    _install_fake_pycaw(monkeypatch, sessions)
    reader = other_audio.pycaw_reader()
    assert reader is not None
    assert reader() == [(222, "music.exe", 0.5)]


def test_pycaw_reader_session_without_pid_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session without a ProcessId gets pid 0 and is skipped (system sounds)."""
    from away.voice import other_audio

    class FakeMeter:
        def GetPeakValue(self) -> float:
            return 0.5

    class NoPidSession:
        def __init__(self) -> None:
            self._ctl = self

        def QueryInterface(self, _iface: Any) -> Any:
            return FakeMeter()

    def sessions() -> list[Any]:
        return [NoPidSession()]

    _install_fake_pycaw(monkeypatch, sessions)
    reader = other_audio.pycaw_reader()
    assert reader is not None
    # pid 0 is skipped by OtherAudioMeter.sample, but read() itself returns the row;
    # the mutant changing the default would produce pid 1, which is NOT skipped.
    assert reader() == [(0, "", 0.5)]


def test_pycaw_reader_session_pid_zero_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session with ProcessId=0 gets pid 0 (the `or 0` matters, not `or 1`)."""
    from away.voice import other_audio

    class FakeMeter:
        def GetPeakValue(self) -> float:
            return 0.5

    class ZeroPidSession:
        ProcessId = 0

        def __init__(self) -> None:
            self._ctl = self

        def QueryInterface(self, _iface: Any) -> Any:
            return FakeMeter()

    def sessions() -> list[Any]:
        return [ZeroPidSession()]

    _install_fake_pycaw(monkeypatch, sessions)
    reader = other_audio.pycaw_reader()
    assert reader is not None
    assert reader() == [(0, "", 0.5)]


class _FakeGuard:
    """Duck-typed TrayGuard: records calls, never touches pystray."""

    def __init__(self) -> None:
        self.stopping = False
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []
        self._lock = threading.RLock()
        self.stopped: list[Any] = []

    def call(self, fn: Any, *args: Any, **kw: Any) -> Any:
        self.calls.append((fn, args))
        return fn(*args, **kw)

    def stop(self, icon: Any) -> None:
        self.stopped.append(icon)


class _FakeCtl:
    """Duck-typed TrayController."""

    def __init__(self) -> None:
        self.paused = False
        self.log: list[str] = []

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        self.log.append("toggle_pause")
        return self.paused

    def toggle_dictation(self) -> None:
        self.log.append("toggle_dictation")

    def toggle_mic(self) -> None:
        self.log.append("toggle_mic")

    def quit(self) -> None:
        self.log.append("quit")

    def retry_cameras(self) -> None:
        self.log.append("retry_cameras")

    def toggle_shortcut_recording(self) -> None:
        self.log.append("toggle_shortcut_recording")

    def stop_shortcut(self) -> None:
        self.log.append("stop_shortcut")

    def stop_everything(self) -> None:
        self.log.append("stop_everything")

    def disconnect_agents(self) -> None:
        self.log.append("disconnect_agents")

    def run_check_pc(self) -> None:
        self.log.append("run_check_pc")

    def run_selftest(self) -> None:
        self.log.append("run_selftest")

    def face_gate_on(self) -> bool:
        return False

    def set_face_gate(self, on: bool) -> bool:
        self.log.append(f"set_face_gate:{on}")
        return on

    def gpu_pack_offer(self) -> str | None:
        return None

    def icon_state(self) -> str:
        return "active"

    def tooltip(self) -> str:
        return "Project Away"

    def is_idle(self, idle_s: float = 600.0) -> bool:
        return False


class _FakeUpdates:
    def __init__(self) -> None:
        self.log: list[str] = []

    def stop(self) -> None:
        self.log.append("stop")

    def install_now(self) -> None:
        self.log.append("install_now")

    def check_now(self) -> None:
        self.log.append("check_now")

    def run_installer(self) -> None:
        self.log.append("run_installer")


class _FakeIcon:
    def __init__(self) -> None:
        self.notified: list[tuple[str, str]] = []
        self.updated = 0

    def notify(self, message: str, title: str) -> None:
        self.notified.append((title, message))

    def update_menu(self) -> None:
        self.updated += 1


def _callbacks() -> tuple[Any, _FakeCtl, Any, _FakeUpdates, _FakeGuard, _FakeIcon]:
    from away.app import tray

    ctl = _FakeCtl()
    app = types.SimpleNamespace(
        turn_off_camera=lambda d: ctl.log.append(f"turn_off_camera:{d}"),
        reset_cursor_size=lambda: ctl.log.append("reset_cursor_size"),
        camera_unavailable={"devices": ["cam1"]},
    )
    updates = _FakeUpdates()
    guard = _FakeGuard()
    cb = tray.TrayMenuCallbacks(ctl, app, updates, guard)
    cb.icon_notifications = True  # these tests drive pystray's notify path on every OS (macOS uses its own window)
    icon = _FakeIcon()
    cb.icon = icon
    return cb, ctl, app, updates, guard, icon


def test_callbacks_on_pause_toggles_and_refreshes() -> None:
    cb, ctl, _app, _updates, guard, icon = _callbacks()
    cb.on_pause(icon, None)
    assert ctl.log == ["toggle_pause"]
    assert ctl.paused is True
    assert guard.calls  # refresh ran through the guard


def test_callbacks_on_pause_exception_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()

    def boom() -> bool:
        raise RuntimeError("COM blew up")

    ctl.toggle_pause = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="away.tray"):
        assert cb.on_pause(icon, None) is None
    assert "pause" in caplog.text


def test_callbacks_on_quit_stops_everything() -> None:
    cb, ctl, _app, updates, guard, icon = _callbacks()
    cb.on_quit(icon, None)
    assert updates.log == ["stop"]
    assert ctl.log == ["quit"]
    assert guard.stopped == [icon]


def test_callbacks_on_update_spawns_install_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    cb, _ctl, _app, updates, _guard, icon = _callbacks()
    started: list[Any] = []

    class FakeThread:
        def __init__(self, target: Any = None, **kw: Any) -> None:
            self._target = target

        def start(self) -> None:
            started.append(self._target)
            self._target()

    monkeypatch.setattr("threading.Thread", FakeThread)
    cb.on_update(icon, None)
    assert updates.log == ["install_now"]
    assert len(started) == 1


def test_callbacks_on_check_updates_spawns_check_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    cb, _ctl, _app, updates, _guard, icon = _callbacks()
    ran: list[bool] = []

    class FakeThread:
        def __init__(self, target: Any = None, **kw: Any) -> None:
            ran.append(True)
            target()

        def start(self) -> None:
            pass

    monkeypatch.setattr("threading.Thread", FakeThread)
    cb.on_check_updates(icon, None)
    assert updates.log == ["check_now"]
    assert ran == [True]


def test_callbacks_on_run_installer() -> None:
    cb, _ctl, _app, updates, _guard, icon = _callbacks()
    cb.on_run_installer(icon, None)
    assert updates.log == ["run_installer"]


def test_callbacks_notify_uses_icon() -> None:
    cb, _ctl, _app, _updates, _guard, icon = _callbacks()
    cb.notify("T", "M")
    assert icon.notified == [("T", "M")]


def test_callbacks_notify_falls_back_to_desktop(monkeypatch: pytest.MonkeyPatch) -> None:

    cb, _ctl, _app, _updates, _guard, icon = _callbacks()
    fell_back: list[tuple[str, str]] = []
    monkeypatch.setattr("away.app.frozen.notify", lambda t, m: fell_back.append((t, m)))

    def boom(_m: str, _t: str) -> None:
        raise RuntimeError("tray gone")

    icon.notify = boom  # type: ignore[method-assign]
    cb.notify("T", "M")
    assert fell_back == [("T", "M")]


def test_callbacks_notify_skipped_when_stopping() -> None:
    cb, _ctl, _app, _updates, guard, icon = _callbacks()
    guard.stopping = True
    cb.notify("T", "M")
    assert icon.notified == []


def test_callbacks_on_face_gate_toggles() -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    cb.on_face_gate(icon, None)
    assert "set_face_gate:True" in ctl.log
    assert icon.notified == [("Project Away", "Face gate on")]
    assert icon.updated == 1


def test_callbacks_on_face_gate_error_notifies(monkeypatch: pytest.MonkeyPatch) -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()

    def boom(_on: bool) -> bool:
        raise RuntimeError("config locked")

    ctl.set_face_gate = boom  # type: ignore[method-assign]
    cb.on_face_gate(icon, None)
    assert icon.notified[0][0] == "Project Away"
    assert "could not change the face gate" in icon.notified[0][1]


def test_callbacks_on_retry_cameras() -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    cb.on_retry_cameras(icon, None)
    assert ctl.log == ["retry_cameras"]


def test_callbacks_on_turn_off_camera() -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    cb.on_turn_off_camera(icon, None)
    assert ctl.log == ["turn_off_camera:cam1"]


def test_callbacks_shortcut_callbacks() -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    cb.on_record_shortcut(icon, None)
    cb.on_stop_shortcut(icon, None)
    cb.on_stop_everything(icon, None)
    assert ctl.log == ["toggle_shortcut_recording", "stop_shortcut", "stop_everything"]


def test_callbacks_on_disconnect_agents() -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    cb.on_disconnect_agents(icon, None)
    assert ctl.log == ["disconnect_agents"]


def test_callbacks_on_reset_cursor() -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    cb.on_reset_cursor(icon, None)
    assert ctl.log == ["reset_cursor_size"]


def test_callbacks_on_check_pc_and_selftest() -> None:
    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    cb.on_check_pc(icon, None)
    cb.on_run_selftest(icon, None)
    assert ctl.log == ["run_check_pc", "run_selftest"]


def test_callbacks_on_dictation_and_mic() -> None:
    cb, ctl, _app, _updates, guard, icon = _callbacks()
    cb.on_dictation(icon, None)
    cb.on_mic(icon, None)
    assert ctl.log == ["toggle_dictation", "toggle_mic"]
    assert len(guard.calls) == 2  # both refreshed the icon


def test_callbacks_safe_update_menu_uses_guard() -> None:
    cb, _ctl, _app, _updates, guard, icon = _callbacks()
    cb.safe_update_menu(icon)
    assert icon.updated == 1
    assert guard.calls  # went through guard.call


def test_callbacks_quit_for_update(monkeypatch: pytest.MonkeyPatch) -> None:

    cb, ctl, _app, updates, guard, icon = _callbacks()
    exited: list[Any] = []
    monkeypatch.setattr("away.app.updater.exit_after_update", lambda fn: exited.append(fn))
    cb.quit_for_update()
    assert len(exited) == 1
    exited[0]()  # the deferred quit
    assert updates.log == ["stop"]
    assert ctl.log == ["quit"]
    assert guard.stopped == [icon]


class _FakePystray:
    """Duck-typed pystray module: records Menu/MenuItem constructions."""

    def __init__(self) -> None:
        self.items: list[Any] = []

    def Menu(self, *items: Any) -> Any:
        self.items = list(items)
        return ("menu", len(items))

    def MenuItem(self, *args: Any, **kwargs: Any) -> Any:
        return ("item", args, kwargs)

    def Icon(self, *args: Any, **kwargs: Any) -> Any:
        return ("icon", args)


def test_build_tray_menu_wires_callbacks() -> None:
    """build_tray_menu constructs the menu with the callbacks' methods (not inline lambdas)."""
    from away.app import tray

    cb, ctl, app, updates, _guard, _icon = _callbacks()
    fake_pystray = _FakePystray()

    def fake_live_menu_items(_pystray: Any, _app: Any, _opener: Any) -> list[Any]:
        return []

    menu = tray.build_tray_menu(fake_pystray, ctl, app, cb, updates, fake_live_menu_items, lambda tab: None)
    assert menu[0] == "menu"
    # comfort_menu_items adds 3 items; update_menu_items adds 3; the rest are literal
    assert menu[1] > 20
    # Every MenuItem's callback (where present) is a bound method of our callbacks object
    callbacks_seen = set()
    for item in fake_pystray.items:
        if not isinstance(item, tuple) or item[0] != "item":
            continue
        args = item[1]
        if len(args) >= 2 and args[1] is not None:
            cb_fn = args[1]
            if hasattr(cb_fn, "__self__"):
                assert cb_fn.__self__ is cb
                callbacks_seen.add(cb_fn.__name__)
    # The key callbacks are wired
    assert "on_pause" in callbacks_seen
    assert "on_quit" in callbacks_seen
    assert "on_stop_everything" in callbacks_seen


def test_callbacks_init_defaults() -> None:
    """The gpu_offer starts not-installing and the gate hint starts hidden."""

    cb, _ctl, _app, _updates, _guard, _icon = _callbacks()
    assert cb.gpu_offer == {"text": None, "installing": False}
    assert cb.gate_hint == {"shown": False}


def test_callbacks_safe_update_menu_uses_self_icon_when_none() -> None:
    """safe_update_menu() with no arg uses self.icon; with an arg uses that icon."""
    cb, _ctl, _app, _updates, _guard, icon = _callbacks()
    other = _FakeIcon()
    cb.safe_update_menu()  # no arg -> self.icon
    assert icon.updated == 1
    assert other.updated == 0
    cb.safe_update_menu(other)  # explicit arg -> that icon
    assert other.updated == 1
    assert icon.updated == 1  # unchanged


def test_callbacks_on_settings_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    opened: list[Any] = []
    monkeypatch.setattr(tray, "open_settings", lambda c: opened.append(c))
    cb.on_settings(icon, None)
    assert opened == [ctl]


def test_callbacks_on_edit_yaml_opens_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    cb, ctl, _app, _updates, _guard, icon = _callbacks()
    opened: list[Any] = []
    monkeypatch.setattr(tray, "open_path", lambda p: opened.append(p))
    ctl.settings_path = lambda: "/fake/config.yaml"  # type: ignore[method-assign]
    cb.on_edit_yaml(icon, None)
    assert opened == ["/fake/config.yaml"]


def test_callbacks_on_gpu_install_failure_notifies(monkeypatch: pytest.MonkeyPatch) -> None:
    """When gpupack.install() raises, the failure is notified (not swallowed silently)."""

    cb, _ctl, _app, _updates, _guard, icon = _callbacks()

    def boom() -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr("away.app.gpupack.install", boom)

    ran: list[bool] = []

    class FakeThread:
        def __init__(self, target: Any = None, **kw: Any) -> None:
            self._target = target

        def start(self) -> None:
            ran.append(True)
            self._target()

    monkeypatch.setattr("threading.Thread", FakeThread)
    cb.on_gpu_install(icon, None)
    assert ran == [True]
    # The failure notification was sent
    assert any("GPU pack" in title for title, _msg in icon.notified)
    assert any("network down" in msg for _title, msg in icon.notified)
    # installing flag was reset
    assert cb.gpu_offer["installing"] is False


def test_callbacks_on_gate_hint_shows_item() -> None:
    cb, _ctl, _app, _updates, _guard, icon = _callbacks()
    assert cb.gate_hint["shown"] is False
    cb.on_gate_hint(None)
    assert cb.gate_hint["shown"] is True
    assert icon.updated == 1


def test_callbacks_sync_icon_refreshes() -> None:
    cb, ctl, app, _updates, guard, icon = _callbacks()
    app.pipeline = types.SimpleNamespace(paused=True)
    cb.sync_icon(icon)
    assert ctl.paused is True
    assert guard.calls  # refresh went through the guard
    assert icon.updated == 1  # update_menu was called


def test_callbacks_sync_icon_skipped_when_stopping() -> None:
    cb, _ctl, _app, _updates, guard, icon = _callbacks()
    guard.stopping = True
    cb.sync_icon(icon)
    assert guard.calls == []
    assert icon.updated == 0


# ---------------------------------------------------------------------------
# main() startup wiring: extracted units plus full main() with a mocked desktop
#
# These tests drive the startup sequence that used to live inline in main()
# (instance acquisition, config load, app creation, tray runtime wiring, bus
# subscriptions) with fakes, asserting the wiring actually happens.
# ---------------------------------------------------------------------------


class _FakeBus:
    """Duck-typed app bus: records subscriptions."""

    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe(self, topic: str, handler: Any) -> None:
        self.subscriptions.append((topic, handler))

    def topics(self) -> list[str]:
        return [t for t, _ in self.subscriptions]

    def handlers(self, topic: str) -> list[Any]:
        return [h for t, h in self.subscriptions if t == topic]


def _patch_submodule(monkeypatch: pytest.MonkeyPatch, dotted: str, fake: Any) -> None:
    """Make `from <pkg> import <name>` and `import <dotted>` resolve to `fake`."""
    import importlib

    monkeypatch.setitem(sys.modules, dotted, fake)
    pkg_name, _, attr = dotted.rpartition(".")
    pkg = importlib.import_module(pkg_name)
    monkeypatch.setattr(pkg, attr, fake, raising=False)


class _FakeCfgMod:
    def __init__(self, cfg: Any, had_config: bool) -> None:
        self._cfg = cfg
        self._had_config = had_config
        self.load_calls = 0

    def config_path(self) -> Any:
        return types.SimpleNamespace(exists=lambda: self._had_config)

    def load(self) -> tuple[Any, dict]:
        self.load_calls += 1
        return (self._cfg, {})


def _make_cfg(always_on: bool = True, start_minimized: bool = False) -> Any:
    return types.SimpleNamespace(
        app=types.SimpleNamespace(always_on=always_on, start_minimized=start_minimized),
        updates=types.SimpleNamespace(check=True),
    )


def test_acquire_single_instance_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    calls: list[int] = []
    _patch_submodule(
        monkeypatch,
        "away.app.instance",
        types.SimpleNamespace(acquire=lambda: calls.append(1) or True),
    )
    assert tray.acquire_single_instance() is True
    assert calls == [1]


def test_acquire_single_instance_false_when_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    _patch_submodule(monkeypatch, "away.app.instance", types.SimpleNamespace(acquire=lambda: False))
    assert tray.acquire_single_instance() is False


def test_open_running_settings_popens_settings_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    popen_calls: list[Any] = []
    monkeypatch.setattr(tray, "subprocess", types.SimpleNamespace(Popen=lambda argv: popen_calls.append(argv)))
    monkeypatch.setattr(tray.TrayController, "settings_command", staticmethod(lambda: ["away", "settings"]))
    tray.open_running_settings()
    assert popen_calls == [["away", "settings"]]


def test_open_running_settings_swallows_popen_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    def boom(argv: Any) -> Any:
        raise OSError("no desktop")

    monkeypatch.setattr(tray, "subprocess", types.SimpleNamespace(Popen=boom))
    monkeypatch.setattr(tray.TrayController, "settings_command", staticmethod(lambda: ["away", "settings"]))
    tray.open_running_settings()  # must not raise


def test_load_tray_config_new_user(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    cfg = _make_cfg()
    fake_cfgmod = _FakeCfgMod(cfg, had_config=False)
    monkeypatch.setattr(tray, "cfgmod", fake_cfgmod)
    got_cfg, had_config = tray.load_tray_config()
    assert got_cfg is cfg
    assert had_config is False
    assert fake_cfgmod.load_calls == 1


def test_load_tray_config_existing_user(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    cfg = _make_cfg()
    monkeypatch.setattr(tray, "cfgmod", _FakeCfgMod(cfg, had_config=True))
    got_cfg, had_config = tray.load_tray_config()
    assert got_cfg is cfg
    assert had_config is True


def test_create_tray_app_starts_paused_when_not_always_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from away.app import tray

    created: list[Any] = []

    class FakeAwayApp:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg
            self.start_calls = 0
            created.append(self)

        def start(self) -> None:
            self.start_calls += 1

    _patch_submodule(monkeypatch, "away.app.runner", types.SimpleNamespace(AwayApp=FakeAwayApp))
    cfg = _make_cfg(always_on=False)
    app, ctl = tray.create_tray_app(cfg)
    assert len(created) == 1
    assert app.cfg is cfg
    assert ctl.paused is True
    assert app.start_calls == 1


def test_create_tray_app_unpaused_when_always_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    _patch_submodule(
        monkeypatch,
        "away.app.runner",
        types.SimpleNamespace(AwayApp=lambda cfg: types.SimpleNamespace(cfg=cfg, start=lambda: None)),
    )
    cfg = _make_cfg(always_on=True)
    _app, ctl = tray.create_tray_app(cfg)
    assert ctl.paused is False


class _WiringCtl:
    def __init__(self) -> None:
        self.mark_activity_calls = 0

    def mark_activity(self, now: Any = None) -> None:
        self.mark_activity_calls += 1


class _WiringCallbacks:
    def __init__(self) -> None:
        self.notify_calls: list[tuple[str, str]] = []
        self.sync_icon_calls: list[Any] = []
        self.gate_hint_calls: list[Any] = []

    def notify(self, title: str, message: str) -> None:
        self.notify_calls.append((title, message))

    def sync_icon(self, icon: Any) -> None:
        self.sync_icon_calls.append(icon)

    def on_gate_hint(self, event: Any) -> None:
        self.gate_hint_calls.append(event)


def test_subscribe_tray_bus_registers_all_topics() -> None:
    from away.app import tray

    app = types.SimpleNamespace(bus=_FakeBus())
    ctl = _WiringCtl()
    cb = _WiringCallbacks()
    icon = object()
    tray.subscribe_tray_bus(app, ctl, cb, icon)
    topics = app.bus.topics()
    for expected in (
        "voice.state",
        "app.paused",
        "gesture.clutch",
        "cameras.retry_stopped",
        "cameras.retry_requested",
        "cameras.turned_off",
        "app.health_hint",
        "attention.state",
        "agentlink.changed",
        "camera.handover",
        "action.fired",
        "app.notice",
        "voice.gate_hint",
    ):
        assert expected in topics, expected
    assert len(topics) == 13


def test_subscribe_tray_bus_handlers_do_the_wiring() -> None:
    from away.app import tray

    app = types.SimpleNamespace(bus=_FakeBus())
    ctl = _WiringCtl()
    cb = _WiringCallbacks()
    icon = object()
    tray.subscribe_tray_bus(app, ctl, cb, icon)

    for handler in app.bus.handlers("voice.state"):
        handler(None)
    assert cb.sync_icon_calls == [icon] * len(app.bus.handlers("voice.state"))

    for handler in app.bus.handlers("action.fired"):
        handler(None)
    assert ctl.mark_activity_calls == len(app.bus.handlers("action.fired"))

    event = types.SimpleNamespace(data={"title": "T", "message": "M"})
    for handler in app.bus.handlers("app.notice"):
        handler(event)
    assert ("T", "M") in cb.notify_calls

    gate_handlers = app.bus.handlers("voice.gate_hint")
    assert len(gate_handlers) == 1
    assert gate_handlers[0] == cb.on_gate_hint


def test_setup_tray_runtime_wires_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    updates_created: list[Any] = []

    class FakeUpdateManager:
        def __init__(self, cfg_updates: Any, **kw: Any) -> None:
            self.cfg_updates = cfg_updates
            self.kwargs = kw
            updates_created.append(self)

    _patch_submodule(monkeypatch, "away.app.updater", types.SimpleNamespace(UpdateManager=FakeUpdateManager))
    hook_calls: list[Any] = []
    _patch_submodule(
        monkeypatch,
        "away.liveagent.integrate",
        types.SimpleNamespace(hook_updates=lambda u: hook_calls.append(u), menu_items=lambda *a: ["live"]),
    )
    menu_calls: list[Any] = []
    monkeypatch.setattr(tray, "build_tray_menu", lambda *a: menu_calls.append(a) or "MENU")
    image_calls: list[Any] = []
    monkeypatch.setattr(tray, "tray_icon_image", lambda *a, **k: image_calls.append((a, k)) or "IMAGE")

    icon_created: list[Any] = []

    class FakePystray:
        @staticmethod
        def Icon(name: str, image: Any, title: Any, menu: Any) -> Any:
            icon = types.SimpleNamespace(name=name, image=image, title=title, menu=menu)
            icon_created.append(icon)
            return icon

    ctl = _FakeCtl()
    ctl.paused = True
    app = types.SimpleNamespace(bus=_FakeBus(), tray_icon=None)
    cfg = _make_cfg()
    callbacks, updates, icon = tray.setup_tray_runtime(app, ctl, cfg, FakePystray)

    assert len(updates_created) == 1
    assert updates_created[0].cfg_updates is cfg.updates
    assert updates_created[0].kwargs["notify"] == callbacks.notify
    assert updates_created[0].kwargs["on_change"] == callbacks.safe_update_menu
    assert updates_created[0].kwargs["quit_app"] == callbacks.quit_for_update
    assert callbacks.updates is updates_created[0]
    assert hook_calls == [updates_created[0]]
    assert len(menu_calls) == 1
    assert menu_calls[0][:5] == (FakePystray, ctl, app, callbacks, updates_created[0])
    assert callable(menu_calls[0][5])  # live_menu_items factory
    assert callable(menu_calls[0][6])  # open_settings_tab factory
    assert image_calls and image_calls[0][0] == (True,)
    assert image_calls[0][1].get("state") == "active"
    assert len(icon_created) == 1
    assert icon_created[0].name == "project-away"
    assert icon_created[0].image == "IMAGE"
    assert icon_created[0].menu == "MENU"
    assert icon_created[0].title == "Project Away"
    assert callbacks.icon is icon_created[0]
    assert app.tray_icon is icon_created[0]
    assert "voice.gate_hint" in app.bus.topics()


def test_maybe_offer_gpu_pack_notifies_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    once_calls: list[str] = []
    _patch_submodule(
        monkeypatch,
        "away.app.notices",
        types.SimpleNamespace(once=lambda name: once_calls.append(name) or True),
    )
    cb = _WiringCallbacks()
    cb.gpu_offer = {"text": "faster speech", "installing": False}  # type: ignore[attr-defined]
    tray.maybe_offer_gpu_pack(cb)
    assert once_calls == ["gpu-pack-offer"]
    assert cb.notify_calls == [
        ("Project Away: faster speech available", "faster speech. Use the tray menu: Install GPU pack.")
    ]


def test_maybe_offer_gpu_pack_skipped_when_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    _patch_submodule(monkeypatch, "away.app.notices", types.SimpleNamespace(once=lambda name: False))
    cb = _WiringCallbacks()
    cb.gpu_offer = {"text": "faster speech", "installing": False}  # type: ignore[attr-defined]
    tray.maybe_offer_gpu_pack(cb)
    assert cb.notify_calls == []


def test_maybe_offer_gpu_pack_skipped_without_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import tray

    once_calls: list[str] = []
    _patch_submodule(
        monkeypatch,
        "away.app.notices",
        types.SimpleNamespace(once=lambda name: once_calls.append(name) or True),
    )
    cb = _WiringCallbacks()
    cb.gpu_offer = {"text": None, "installing": False}  # type: ignore[attr-defined]
    tray.maybe_offer_gpu_pack(cb)
    assert once_calls == []
    assert cb.notify_calls == []


class _MainStage:
    """Installs fakes for every import tray.main() touches."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from away.app import tray

        self.tray = tray
        self.com_apartment_calls: list[int] = []
        monkeypatch.setattr(tray, "com_apartment", lambda: self.com_apartment_calls.append(1) or "")
        self.dpi_calls: list[int] = []
        _patch_submodule(
            monkeypatch,
            "away.control.monitors",
            types.SimpleNamespace(enable_dpi_awareness=lambda *a, **k: self.dpi_calls.append(1) or "unsupported"),
        )
        self.icon_instances: list[Any] = []

        stage = self

        class FakeIcon:
            def __init__(self, name: str, image: Any, title: Any, menu: Any) -> None:
                self.name, self.image, self.title, self.menu = name, image, title, menu
                self.run_calls = 0
                stage.icon_instances.append(self)

            def run(self) -> None:
                self.run_calls += 1

        monkeypatch.setitem(sys.modules, "pystray", types.SimpleNamespace(Icon=FakeIcon))
        self.setup_logging_calls: list[int] = []
        _patch_submodule(
            monkeypatch,
            "away.app.frozen",
            types.SimpleNamespace(setup_logging=lambda: self.setup_logging_calls.append(1)),
        )
        self.crash_install_calls: list[str] = []
        self.crash_note: Any = None
        _patch_submodule(
            monkeypatch,
            "away.app.crashlog",
            types.SimpleNamespace(
                install=lambda name: self.crash_install_calls.append(name) or self.crash_note,
                crash_log_path=lambda: "crash.log",
            ),
        )
        self.acquire_value = True
        self.acquire_calls: list[int] = []
        _patch_submodule(
            monkeypatch,
            "away.app.instance",
            types.SimpleNamespace(acquire=lambda: self.acquire_calls.append(1) or self.acquire_value),
        )
        self.popen_calls: list[Any] = []
        monkeypatch.setattr(tray, "subprocess", types.SimpleNamespace(Popen=lambda argv: self.popen_calls.append(argv)))

        class FakeTrayController:
            def __init__(self, app: Any) -> None:
                self.app = app
                self.paused = False
                self.mark_activity_calls = 0
                self.crash_note = None
                stage.controllers.append(self)

            @staticmethod
            def settings_command() -> list[str]:
                return ["away", "settings"]

            def mark_activity(self, now: Any = None) -> None:
                self.mark_activity_calls += 1

            def icon_state(self) -> str:
                return "active"

            def tooltip(self) -> str:
                return "Project Away"

            def is_idle(self, idle_s: float = 600.0) -> bool:
                return False

        self.controllers: list[Any] = []
        monkeypatch.setattr(tray, "TrayController", FakeTrayController)
        self.cfg = _make_cfg()
        self.had_config = True
        self.cfgmod_load_calls: list[int] = []
        monkeypatch.setattr(tray, "cfgmod", _FakeCfgMod(self.cfg, True))
        # keep had_config in sync: tests flip stage.had_config after setup
        self._cfgmod = tray.cfgmod

        def config_path() -> Any:
            return types.SimpleNamespace(exists=lambda: self.had_config)

        self._cfgmod.config_path = config_path  # type: ignore[method-assign]
        self.apps: list[Any] = []

        class FakeAwayApp:
            def __init__(self, cfg: Any) -> None:
                self.cfg = cfg
                self.bus = _FakeBus()
                self.tray_icon = None
                self.start_calls = 0
                stage.apps.append(self)

            def start(self) -> None:
                self.start_calls += 1

        _patch_submodule(monkeypatch, "away.app.runner", types.SimpleNamespace(AwayApp=FakeAwayApp))
        self.open_settings_calls: list[Any] = []
        monkeypatch.setattr(tray, "open_settings", lambda ctl: self.open_settings_calls.append(ctl))
        self.should_auto_open = False
        self.mark_seen_calls: list[int] = []
        _patch_submodule(
            monkeypatch,
            "away.app.first_run",
            types.SimpleNamespace(
                should_auto_open=lambda: self.should_auto_open,
                mark_seen=lambda: self.mark_seen_calls.append(1),
            ),
        )
        self.callbacks_instances: list[Any] = []

        class FakeCallbacks:
            def __init__(self, ctl: Any, app: Any, updates: Any, guard: Any) -> None:
                self.ctl, self.app, self.updates_arg, self.guard = ctl, app, updates, guard
                self.icon = None
                self.notify_calls: list[tuple[str, str]] = []
                self.sync_icon_calls: list[Any] = []
                self.gpu_offer: dict[str, Any] = {"text": None, "installing": False}
                stage.callbacks_instances.append(self)

            def notify(self, title: str, message: str) -> None:
                self.notify_calls.append((title, message))

            def safe_update_menu(self, icon: Any = None) -> None:
                pass

            def quit_for_update(self) -> None:
                pass

            def on_gate_hint(self, event: Any) -> None:
                pass

            def sync_icon(self, icon: Any) -> None:
                self.sync_icon_calls.append(icon)

        monkeypatch.setattr(tray, "TrayMenuCallbacks", FakeCallbacks)
        self.update_managers: list[Any] = []

        class FakeUpdateManager:
            def __init__(self, cfg_updates: Any, **kw: Any) -> None:
                self.cfg_updates = cfg_updates
                self.kwargs = kw
                self.start_calls = 0
                self.exiting = False
                stage.update_managers.append(self)

            def start(self) -> None:
                self.start_calls += 1

        _patch_submodule(monkeypatch, "away.app.updater", types.SimpleNamespace(UpdateManager=FakeUpdateManager))
        self.hook_updates_calls: list[Any] = []
        _patch_submodule(
            monkeypatch,
            "away.liveagent.integrate",
            types.SimpleNamespace(
                hook_updates=lambda u: self.hook_updates_calls.append(u),
                menu_items=lambda *a: ["live"],
            ),
        )
        monkeypatch.setattr(tray, "build_tray_menu", lambda *a: "MENU")
        monkeypatch.setattr(tray, "tray_icon_image", lambda *a, **k: "IMAGE")
        self.once_calls: list[str] = []
        self.once_value = False
        _patch_submodule(
            monkeypatch,
            "away.app.notices",
            types.SimpleNamespace(once=lambda name: self.once_calls.append(name) or self.once_value),
        )
        self.launch_first_run_calls: list[int] = []
        _patch_submodule(
            monkeypatch,
            "away.comfort.actions",
            types.SimpleNamespace(launch_first_run=lambda: self.launch_first_run_calls.append(1)),
        )
        self.exit_calls: list[int] = []
        monkeypatch.setattr(os, "_exit", lambda code: self.exit_calls.append(code))


def test_main_already_running_opens_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)
    stage.acquire_value = False
    rc = stage.tray.main()
    assert rc == 0
    assert stage.acquire_calls == [1]
    assert stage.popen_calls == [["away", "settings"]]
    assert stage.apps == []
    assert stage.icon_instances == []
    assert stage.open_settings_calls == []


def test_main_full_startup_wires_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)
    rc = stage.tray.main()
    assert rc == 0
    assert stage.dpi_calls == [1]
    assert stage.com_apartment_calls == [1]
    assert stage.setup_logging_calls == [1]
    assert stage.crash_install_calls == ["tray"]
    assert len(stage.apps) == 1
    app = stage.apps[0]
    assert app.cfg is stage.cfg
    assert app.start_calls == 1
    assert len(stage.controllers) == 1
    assert stage.controllers[0].paused is False  # always_on True
    assert len(stage.open_settings_calls) == 1  # not start_minimized
    assert len(stage.callbacks_instances) == 1
    cb = stage.callbacks_instances[0]
    assert len(stage.update_managers) == 1
    um = stage.update_managers[0]
    assert um.cfg_updates is stage.cfg.updates
    assert um.start_calls == 1
    assert stage.hook_updates_calls == [um]
    assert len(stage.icon_instances) == 1
    icon = stage.icon_instances[0]
    assert (icon.name, icon.image, icon.menu) == ("project-away", "IMAGE", "MENU")
    assert icon.run_calls == 1
    assert cb.icon is icon
    assert app.tray_icon is icon
    topics = app.bus.topics()
    for expected in (
        "voice.state",
        "attention.state",
        "agentlink.changed",
        "camera.handover",
        "action.fired",
        "app.notice",
        "voice.gate_hint",
    ):
        assert expected in topics
    assert stage.controllers[0].mark_activity_calls >= 1  # idle clock starts at launch
    assert stage.exit_calls == []


def test_main_bus_handlers_are_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)
    stage.tray.main()
    app = stage.apps[0]
    cb = stage.callbacks_instances[0]
    ctl = stage.controllers[0]
    icon = stage.icon_instances[0]
    for handler in app.bus.handlers("gesture.clutch"):
        handler(None)
    assert cb.sync_icon_calls == [icon] * len(app.bus.handlers("gesture.clutch"))
    for handler in app.bus.handlers("action.fired"):
        handler(None)
    assert ctl.mark_activity_calls >= 1
    event = types.SimpleNamespace(data={"title": "T", "message": "M"})
    for handler in app.bus.handlers("app.notice"):
        handler(event)
    assert ("T", "M") in cb.notify_calls
    assert app.bus.handlers("voice.gate_hint") == [cb.on_gate_hint]


def test_main_start_minimized_skips_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)
    stage.cfg.app.start_minimized = True
    assert stage.tray.main() == 0
    assert stage.open_settings_calls == []
    assert stage.icon_instances[0].run_calls == 1


def test_main_first_run_new_user_launches_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)
    stage.should_auto_open = True
    stage.had_config = False
    assert stage.tray.main() == 0
    assert stage.mark_seen_calls == [1]
    assert stage.launch_first_run_calls == [1]


def test_main_first_run_existing_user_gets_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)
    stage.should_auto_open = True
    stage.had_config = True
    assert stage.tray.main() == 0
    assert stage.mark_seen_calls == [1]
    assert stage.launch_first_run_calls == []
    cb = stage.callbacks_instances[0]
    assert (
        "Project Away",
        "New: a guided tour and comfort helpers. Tray menu > Guided first run...",
    ) in cb.notify_calls


def test_main_crash_note_notifies(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)
    stage.crash_note = "boom"
    assert stage.tray.main() == 0
    cb = stage.callbacks_instances[0]
    ctl = stage.controllers[0]
    assert ctl.crash_note == "boom (crash.log)"
    assert ("Project Away", "boom (crash.log)") in cb.notify_calls


def test_main_gpu_offer_notifies_once(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)
    stage.once_value = True

    orig_init = stage.tray.TrayMenuCallbacks.__init__

    def init_with_offer(self: Any, ctl: Any, app: Any, updates: Any, guard: Any) -> None:
        orig_init(self, ctl, app, updates, guard)
        self.gpu_offer = {"text": "fast", "installing": False}

    monkeypatch.setattr(stage.tray.TrayMenuCallbacks, "__init__", init_with_offer)
    assert stage.tray.main() == 0
    cb = stage.callbacks_instances[0]
    assert ("Project Away: faster speech available", "fast. Use the tray menu: Install GPU pack.") in cb.notify_calls


def test_main_exiting_for_update_calls_os_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _MainStage(monkeypatch)

    orig_icon_cls = sys.modules["pystray"].Icon

    class ExitingIcon(orig_icon_cls):  # type: ignore[valid-type,misc]
        def run(self) -> None:
            for um in stage.update_managers:
                um.exiting = True
            super().run()

    monkeypatch.setitem(sys.modules, "pystray", types.SimpleNamespace(Icon=ExitingIcon))
    assert stage.tray.main() == 0
    assert stage.exit_calls == [0]


def test_all_pystray_menu_callbacks_are_menu_safe() -> None:
    """Every on_* menu callback (icon, item) must be wrapped in menu_safe.

    An unwrapped callback that raises on pystray's message thread breaks the
    loop and takes the tray down. functools.wraps sets __wrapped__, so an
    undecorated callback is detectable.
    """
    import inspect

    from away.app import tray

    unprotected = []
    for name in dir(tray.TrayMenuCallbacks):
        if not name.startswith("on_"):
            continue
        fn = getattr(tray.TrayMenuCallbacks, name)
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        # pystray menu callbacks take (icon, item); bus handlers take (event,) or (icon,)
        is_menu_cb = params[1:] == ["icon", "_item"] or params[1:] == ["_icon", "_item"]
        if is_menu_cb and not hasattr(fn, "__wrapped__"):
            unprotected.append(name)
    assert not unprotected, f"menu callbacks missing @menu_safe: {unprotected}"
