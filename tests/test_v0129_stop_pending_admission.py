"""Stop surfaces during a pending grant admission (Codex P2 on #94, 2026-10-21).

The open-palm and voice stop paths only fired while a grant existed or the
engine was busy. While an approval prompt waited, neither was true, so
holding up an open palm or saying "stop" left the pending admission in
place and a later Allow still installed the grant. The stop hotkey never
had this gap: it calls engine.stop_all unconditionally, and the control's
stop listener cancels the pending admission.

Both tests below fail on the unfixed code (the Allow installs the grant)
and pass once the stop gates also admit a pending admission.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element
from away.agentlink.host import AgentLinkHost
from away.agentlink.prompts import NullUI
from away.control.mouse import RecordingBackend
from away.hands.synthetic import pose


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeUIA:
    available = True

    def __init__(self, at: Any = None) -> None:
        self._at = at

    def focused(self) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        return self._at() if callable(self._at) else self._at

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None


class BlockingAsk:
    """The first approval prompt blocks until the test releases it."""

    def __init__(self, first: str = "Allow") -> None:
        self.first = first
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, _q: str, _o: Any, **kw: Any) -> dict[str, Any]:
        self.entered.set()
        if not self.release.wait(timeout=20):
            raise AssertionError("the test never released the first prompt")
        return {"answer": self.first, "method": "click"}


def _control(tmp_path: Path, ask: Any) -> ComputerControl:
    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=FakeUIA(at=Element("Canvas", "PaneControl")),
    )
    hooks.cursor_pos = lambda: (7, 8)
    return ComputerControl(
        hooks,
        ask,
        ttl_s=1800,
        rate_per_s=10000.0,
        log_folder=tmp_path / "actions",
        clock=Clock(),
    )


def _request_in_thread(ctl: ComputerControl) -> tuple[threading.Thread, dict[str, Any]]:
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = ctl.request("agent1", "Agent1")
        except Exception as exc:  # the test asserts on the captured outcome
            box["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    return t, box


def _host_with_control(tmp_path: Path, ctl: ComputerControl) -> AgentLinkHost:
    host = AgentLinkHost(None, data_dir=tmp_path / "host", ui=NullUI(), headless=True)
    host.core.control = ctl
    return host


def _allow_after_stop(ask: BlockingAsk, t: threading.Thread, box: dict[str, Any]) -> None:
    ask.release.set()  # the person allows, after the stop
    t.join(15)
    assert not t.is_alive()
    assert "error" in box
    assert isinstance(box["error"], ControlRefused)
    assert "stopped before it took effect" in str(box["error"])


def test_open_palm_hold_during_approval_cancels_the_pending_admission(tmp_path: Path) -> None:
    ask = BlockingAsk()
    ctl = _control(tmp_path, ask)
    t, box = _request_in_thread(ctl)
    assert ask.entered.wait(timeout=10)  # the approval prompt is on screen
    host = _host_with_control(tmp_path, ctl)
    t_frame = 1000.0
    while t_frame <= 1001.2:  # frames 0.1s apart: the 1s hold needs gaps under 0.25s
        host._on_hand(t_frame, pose("palm"), "Right", {})
        t_frame += 0.1
    assert ctl.grant is None
    _allow_after_stop(ask, t, box)


def test_voice_stop_during_approval_cancels_the_pending_admission(tmp_path: Path) -> None:
    ask = BlockingAsk()
    ctl = _control(tmp_path, ask)
    t, box = _request_in_thread(ctl)
    assert ask.entered.wait(timeout=10)  # the approval prompt is on screen
    host = _host_with_control(tmp_path, ctl)
    host._on_bus(SimpleNamespace(topic="voice.utterance", data={"text": "stop"}, ts=1000.0))
    assert ctl.grant is None
    _allow_after_stop(ask, t, box)
