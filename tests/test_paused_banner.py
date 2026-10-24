"""Proposal #16: the persistent paused-state banner.

On takeover the grant pauses and a slim top-center banner stays on screen ("Paused - you took over from
<name>. [Let <name> continue] [Stop]") instead of the border/banner vanishing with no explanation. The
banner's buttons answer directly (the click IS the answer): no second approval round-trip through the agent.

Every test here fails on the unfixed code: ``overlay_paused`` / ``paused_banner_answer`` /
``paused_banner_texts`` do not exist there, and ``user_input`` hides the overlay on pause.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from away.agentlink import ui_helper
from away.agentlink.control import GRACE_S, ComputerControl, ControlHooks
from away.agentlink.prompts import NullUI, SubprocessUI
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _control(clock: Clock, answers: list[str] | None = None) -> tuple[ComputerControl, dict[str, Any]]:
    """A grant-ready ComputerControl whose overlay calls are recorded."""
    calls: dict[str, Any] = {"on": [], "off": 0, "paused": []}
    asked: list[dict[str, Any]] = []
    queue = list(answers or ["Allow"])

    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        asked.append({"question": question, "options": options, **kw})
        return {"answer": queue.pop(0) if queue else None, "method": "click"}

    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        overlay_on=lambda name: calls["on"].append(name),
        overlay_off=lambda: calls.__setitem__("off", calls["off"] + 1),
        overlay_paused=lambda name: calls["paused"].append(name),
    )
    ctl = ComputerControl(hooks, ask, ttl_s=1800, rate_per_s=50, clock=clock)
    return ctl, {"calls": calls, "asked": asked}


def _paused(ctl: ComputerControl, clock: Clock) -> None:
    ctl.request("a1", "Sunday")
    clock.advance(GRACE_S + 1)  # the approval click is long past: real input is a takeover
    assert ctl.user_input("mouse")
    assert ctl.status()["paused"]


# ------------------------------------------------------- the banner replaces overlay_off on pause
def test_pause_shows_the_paused_banner_instead_of_hiding_the_overlay() -> None:
    clock = Clock()
    ctl, rec = _control(clock)
    _paused(ctl, clock)
    assert rec["calls"]["paused"] == ["Sunday"]
    assert rec["calls"]["off"] == 0  # the old behavior hid everything with no explanation


def test_paused_banner_texts_are_plain_words() -> None:
    t = ui_helper.paused_banner_texts({"agent": "Sunday"})
    assert t["title"] == "Paused"
    assert "Sunday" in t["text"] and "took over" in t["text"]
    assert [v for v, _ in t["buttons"]] == ["continue", "stop"]
    assert t["buttons"][0][1] == "Let Sunday continue"
    assert t["buttons"][1][1] == "Stop"


def test_paused_banner_texts_default_name() -> None:
    t = ui_helper.paused_banner_texts({})
    assert "Your AI agent" in t["text"]


# ------------------------------------------------------- the buttons answer directly (no round-trip)
def test_banner_continue_resumes_without_a_new_prompt() -> None:
    clock = Clock()
    ctl, rec = _control(clock)
    _paused(ctl, clock)
    asked_before = len(rec["asked"])
    assert ctl.paused_banner_answer("continue") is True
    assert not ctl.status()["paused"]
    assert len(rec["asked"]) == asked_before  # no second approval round-trip through the agent
    assert rec["calls"]["on"] == ["Sunday", "Sunday"]  # request's banner, then the control banner back on resume


def test_banner_stop_ends_the_grant() -> None:
    clock = Clock()
    ctl, rec = _control(clock)
    _paused(ctl, clock)
    assert ctl.paused_banner_answer("stop") is True
    assert ctl.status() == {"active": False}
    assert rec["calls"]["off"] == 1


def test_banner_answer_with_no_live_grant_is_a_noop() -> None:
    clock = Clock()
    ctl, rec = _control(clock)
    assert ctl.paused_banner_answer("continue") is False
    assert ctl.paused_banner_answer("stop") is False
    assert rec["calls"]["on"] == [] and rec["calls"]["off"] == 0


def test_stale_banner_click_after_agent_resume_clears_the_banner() -> None:
    """The grant was already resumed through the agent (a new approval): the stale banner's "continue" click
    must not re-pause or ask again, just clear the banner and keep running."""
    clock = Clock()
    ctl, rec = _control(clock, ["Allow", "Continue"])
    _paused(ctl, clock)
    ctl.resume("a1")
    assert not ctl.status()["paused"]
    assert ctl.paused_banner_answer("continue") is True
    assert not ctl.status()["paused"]  # still running
    assert rec["calls"]["on"][-1] == "Sunday"  # the stale paused banner is gone, control banner back


# ------------------------------------------------------- grace: the banner click is not a takeover
def test_banner_continue_click_is_not_a_takeover_but_real_input_still_pauses() -> None:
    """The click on "Let Sunday continue" must not instantly re-pause (the grace window covers it), while real
    input a moment later still pauses instantly."""
    clock = Clock()
    ctl, _rec = _control(clock)
    _paused(ctl, clock)
    assert ctl.paused_banner_answer("continue") is True
    assert ctl.user_input("mouse") is False  # the banner click itself: not a takeover
    clock.advance(GRACE_S + 1)
    assert ctl.user_input("mouse") is True  # real input still pauses instantly
    assert ctl.status()["paused"]


# ------------------------------------------------------- ui_helper routes paused specs to _control
def test_main_routes_paused_control_specs(tmp_path: Path, monkeypatch: Any) -> None:
    # main() builds a tk.Tk before _control: fake it (no display on CI), like the window code is faked for toasts.
    pytest.importorskip("tkinter")

    class FakeRoot:
        def withdraw(self) -> None:
            pass

        def attributes(self, *a: Any) -> None:
            pass

        def mainloop(self) -> None:
            pass

    monkeypatch.setattr("tkinter.Tk", FakeRoot)
    seen: dict[str, Any] = {}

    def fake_control(root: Any, spec: dict[str, Any], spec_path: Path) -> None:
        seen["spec"], seen["path"] = spec, spec_path

    monkeypatch.setattr(ui_helper, "_control", fake_control)
    spec = tmp_path / "paused-x.json"
    spec.write_text(json.dumps({"id": "paused-x", "agent": "Sunday", "stop_file": "s", "mode": "paused"}))
    assert ui_helper.main("control", spec) == 0
    assert seen["spec"]["mode"] == "paused"
    assert seen["path"] == spec  # the banner answers next to its own spec file


# ------------------------------------------------------- prompts.paused_banner writes a paused spec
def test_paused_banner_spec_and_answer_path(tmp_path: Path, monkeypatch: Any) -> None:
    ui = SubprocessUI(tmp_path, command=lambda args: args)
    monkeypatch.setattr(ui, "_spawn", lambda args: "proc")
    proc, answer = ui.paused_banner("Sunday", tmp_path / "s.stop")
    assert proc == "proc"
    assert answer.name.endswith(".answer.json") and answer.parent == tmp_path
    specs = list(tmp_path.glob("paused-*.json"))
    assert len(specs) == 1
    data = json.loads(specs[0].read_text(encoding="utf-8"))
    assert data["mode"] == "paused" and data["agent"] == "Sunday"
    assert answer.name == f"{specs[0].stem}.answer.json"  # the answer lands next to the spec, same id


def test_paused_banner_spawn_failure_raises(tmp_path: Path, monkeypatch: Any) -> None:
    ui = SubprocessUI(tmp_path, command=lambda args: args)
    monkeypatch.setattr(ui, "_spawn", lambda args: None)
    ui.last_spawn_error = "no display"
    try:
        ui.paused_banner("Sunday", tmp_path / "s.stop")
    except RuntimeError as exc:
        assert "paused banner" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


# ------------------------------------------------------- host: the watcher feeds the click to the grant
class _BannerUI(NullUI):
    def __init__(self, folder: Path) -> None:
        super().__init__()
        self.folder = folder
        self.banners: list[tuple[str, Path]] = []

    def paused_banner(self, name: str, stop_file: Path) -> tuple[Any, Path]:
        self.banners.append((name, stop_file))
        return None, self.folder / "paused-test.answer.json"


def _host(tmp_path: Path) -> Any:
    from away.agentlink.host import AgentLinkHost

    ui = _BannerUI(tmp_path)
    h = AgentLinkHost(None, data_dir=tmp_path, ui=ui, headless=True)
    h.core.control.ask = lambda *a, **k: {"answer": "Allow"}
    h.core.control.hooks.secure_desktop = lambda: False
    h.core.control.hooks.foreground = lambda: None
    return h


def test_host_paused_banner_watcher_feeds_continue_to_the_grant(tmp_path: Path) -> None:
    h = _host(tmp_path)
    ctl = h.core.control
    ctl.request("a1", "Sunday")
    ctl._grace_until = 0.0  # past the approval click
    assert ctl.user_input("mouse")  # takeover: the paused banner shows
    assert ctl.grant is not None and ctl.grant.paused
    _proc, answer_file = h.ui.paused_banner("Sunday", tmp_path / "s.stop")  # the banner the host showed
    answer_file.write_text(json.dumps({"answer": "continue"}), encoding="utf-8")
    assert h._overlay_stop is not None
    h._watch_paused_banner(answer_file, h._overlay_stop, time.monotonic() + 5)
    assert not ctl.grant.paused  # resumed, no new approval prompt
    assert not answer_file.exists()  # consumed exactly once


def test_host_paused_banner_watcher_feeds_stop_to_the_grant(tmp_path: Path) -> None:
    h = _host(tmp_path)
    ctl = h.core.control
    ctl.request("a1", "Sunday")
    ctl._grace_until = 0.0
    assert ctl.user_input("mouse")
    _proc, answer_file = h.ui.paused_banner("Sunday", tmp_path / "s.stop")
    answer_file.write_text(json.dumps({"answer": "stop"}), encoding="utf-8")
    assert h._overlay_stop is not None
    h._watch_paused_banner(answer_file, h._overlay_stop, time.monotonic() + 5)
    assert ctl.grant is None  # ended


def test_host_paused_banner_watcher_exits_when_the_banner_is_gone(tmp_path: Path) -> None:
    h = _host(tmp_path)
    h._overlay_stop = None  # the banner was cleared another way (end, resume, expiry)
    answer_file = tmp_path / "paused-test.answer.json"
    h._watch_paused_banner(answer_file, tmp_path / "gone.stop", time.monotonic() + 5)  # must return, not spin


def test_host_paused_banner_watcher_exits_when_a_newer_overlay_replaced_it(tmp_path: Path) -> None:
    """A stale watcher must not keep polling after its banner was replaced: it is bound to its own stop path,
    not to _overlay_stop being non-None."""
    h = _host(tmp_path)
    old_stop, new_stop = tmp_path / "old.stop", tmp_path / "new.stop"
    h._overlay_stop = new_stop  # a newer overlay took over (e.g. the agent resumed through a prompt)
    answer_file = tmp_path / "paused-test.answer.json"
    answer_file.write_text(json.dumps({"answer": "continue"}), encoding="utf-8")
    h._watch_paused_banner(answer_file, old_stop, time.monotonic() + 5)  # must return, not spin
    assert answer_file.exists()  # the stale watcher consumed nothing
