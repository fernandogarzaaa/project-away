"""Trust and safety hardening (external audit, nine findings + two small fixes).

Every test here failed on the code before its fix (checked by running it against the unfixed commit).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element, best_match
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeUIA:
    available = True

    def __init__(self, focused: Element | None = None, at: Any = None, elements: Any = ()) -> None:
        self._focused, self._at, self._els = focused, at, list(elements)

    def focused(self) -> Element | None:
        return self._focused

    def element_at(self, x: int, y: int) -> Element | None:
        if callable(self._at):
            return self._at(x, y)  # type: ignore[no-any-return]
        return self._at

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return best_match(self._els, name, control_type)

    def tree(self, *a: Any) -> None:
        return None


TEXT_BOX = Element("Message", "EditControl", rect=(0, 0, 9, 9))


def _control(
    clock: Clock,
    answers: list[str] | None = None,
    uia: Any = None,
    backend: Any = None,
    secure: Any = False,
    platform: str = "linux",
    sleep: Any = None,
) -> tuple[ComputerControl, Any, list[dict[str, Any]]]:
    backend = backend if backend is not None else RecordingBackend()
    asked: list[dict[str, Any]] = []
    queue = list(answers or ["Allow"])

    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        asked.append({"question": question, "options": options, **kw})
        return {"answer": queue.pop(0) if queue else None, "method": "click"}

    hooks = ControlHooks(
        backend=backend,
        platform=platform,
        secure_desktop=secure if callable(secure) else (lambda: secure),
        foreground=lambda: None,
        grab=lambda: None,
        sleep=sleep or (lambda _s: None),
        uia=uia if uia is not None else FakeUIA(focused=TEXT_BOX, at=Element("Canvas", "PaneControl")),
    )
    ctl = ComputerControl(hooks, ask, ttl_s=1800, rate_per_s=50, clock=clock)
    return ctl, backend, asked


# ------------------------------------------------------------------------- 1. undo needs a live grant
def test_undo_needs_a_live_unpaused_grant_owned_by_the_calling_agent() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow", "Allow"])
    ctl.hooks.cursor_pos = lambda: (50, 60)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=300, y=300)
    n = len(backend.calls)
    with pytest.raises(ControlRefused):
        ctl.undo_last("a2")  # another agent cannot undo (or even trigger) anything
    ctl.stop("you pressed Stop")
    with pytest.raises(ControlRefused):
        ctl.undo_last("a1")  # the grant ended: no more input from this agent
    assert len(backend.calls) == n
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=310, y=310)
    clock.advance(5)
    assert ctl.user_input("mouse")  # the person took over
    n = len(backend.calls)
    with pytest.raises(ControlRefused, match="paused"):
        ctl.undo_last("a1")
    assert len(backend.calls) == n


def test_undo_only_reverses_the_agents_own_actions_and_expires() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow", "Allow"])
    ctl.hooks.cursor_pos = lambda: (1, 2)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=300, y=300)
    ctl.stop()
    ctl.request("a2", "Gemini")
    with pytest.raises(ControlRefused, match="nothing"):
        ctl.undo_last("a2")  # a1's move is not a2's to undo
    clock.advance(1801)
    with pytest.raises(ControlRefused):
        ctl.undo_last("a2")
    assert ctl.undo_last_by_user()["undone"] == "move"  # the person's own Undo button still works


def test_service_undo_passes_the_agent(tmp_path: Path) -> None:
    from away import config as C
    from away.agentlink.host import build_core

    cfg = C.AppConfig()
    core = build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: tmp_path / "c")
    seen: list[Any] = []
    core.control.undo_last = lambda *a, **k: seen.append((a, k)) or {}  # type: ignore[method-assign]
    a, _ = core.registry.issue("Claude", ["read", "control"], 3600)
    core.service.undo_last_action(a)
    assert seen == [((a.id,), {})]


# --------------------------------------------------------------- 2. takeover windows (mouse vs keyboard echo)
def test_real_mouse_movement_pauses_the_agent_while_it_types_a_long_text() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock)
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    ctl.act("a1", "type_text", text="x" * 2000)  # used to hide every mouse movement for 40.5 s
    clock.advance(0.05)
    assert ctl.user_input("mouse")
    assert ctl.status()["paused"]


def test_keyboard_echo_is_ignored_only_briefly_after_each_typed_chunk() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock)
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    ctl.act("a1", "type_text", text="y" * 2000)
    assert not ctl.user_input("keyboard")  # the agent's own keys echoing back
    typed = [c for c in backend.calls if c[0] == "type_text"]
    assert len(typed) > 1 and all(len(c[1][0]) <= 16 for c in typed)  # small chunks, one short window each
    clock.advance(1.0)
    assert ctl.user_input("keyboard")  # one second later a key press is the person


def test_mouse_echo_of_the_agents_own_move_is_ignored_but_not_for_long() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock)
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    ctl.act("a1", "move", x=5, y=5)
    assert not ctl.user_input("mouse")
    assert ctl.user_input("keyboard")  # a key press during a mouse move is never the agent's echo


def test_approval_grace_is_short_and_never_blocks_a_deliberate_stop() -> None:
    clock = Clock()
    ctl, _, _ = _control(clock)
    ctl.request("a1", "Claude")
    assert not ctl.user_input("hand")  # the thumbs up that approved it
    clock.advance(1.0)
    assert ctl.user_input("mouse")  # a second later, moving the mouse takes over
    ctl2, backend2, _ = _control(clock)
    ctl2.request("a1", "Claude")
    assert ctl2.stop("you held up an open palm")  # inside the grace window: the stop still works at once
    with pytest.raises(ControlRefused):
        ctl2.act("a1", "move", x=1, y=1)
    assert backend2.calls == []


# ------------------------------------------------------------- 3. stopping an action that is already running
class BlockingBackend(RecordingBackend):
    """Records like RecordingBackend, and blocks inside the ``n``-th call of ``method`` until ``go`` is set (so a
    test can stop or take over while the action is in the middle of a drag or of typing)."""

    def __init__(self, method: str, n: int) -> None:
        super().__init__()
        self.method, self.n, self.seen = method, n, 0
        self.reached, self.go = threading.Event(), threading.Event()

    def _maybe_block(self, name: str) -> None:
        if name == self.method:
            self.seen += 1
            if self.seen == self.n:
                self.reached.set()
                assert self.go.wait(5.0)

    def move(self, x: int, y: int) -> None:
        super().move(x, y)
        self._maybe_block("move")

    def type_text(self, text: str) -> None:
        super().type_text(text)
        self._maybe_block("type_text")

    def release_modifiers(self) -> None:
        self.calls.append(("release_modifiers", ()))


def _run_in_thread(fn: Any) -> tuple[threading.Thread, dict[str, Any]]:
    out: dict[str, Any] = {}

    def run() -> None:
        try:
            out["result"] = fn()
        except Exception as exc:
            out["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_stop_interrupts_a_drag_in_the_middle_and_lets_go_of_the_button() -> None:
    clock = Clock()
    backend = BlockingBackend("move", 4)  # start point + 2 steps done, blocked in the 3rd step
    ctl, _, _ = _control(clock, backend=backend)
    ctl.request("a1", "Claude")
    t, out = _run_in_thread(lambda: ctl.act("a1", "drag", x=0, y=0, x2=800, y2=0))
    assert backend.reached.wait(5.0)
    assert ctl.stop("you pressed the stop keys")
    backend.go.set()
    t.join(5.0)
    assert isinstance(out.get("error"), ControlRefused) and "stop" in str(out["error"])
    after = backend.calls[[c[0] for c in backend.calls].index("press") :]
    assert [c[0] for c in after].count("move") == 3  # no step after the stop
    assert ("release", ()) in after  # the mouse button is never left held down
    assert ("release_modifiers", ()) in after
    assert ctl.log[-1].result == "stopped"


def test_moving_the_mouse_interrupts_typing_between_chunks() -> None:
    clock = Clock()
    backend = BlockingBackend("type_text", 2)
    ctl, _, _ = _control(clock, backend=backend)
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    t, out = _run_in_thread(lambda: ctl.act("a1", "type_text", text="abcdefgh" * 50))
    assert backend.reached.wait(5.0)
    assert ctl.user_input("mouse")  # the person grabs the mouse while it types
    backend.go.set()
    t.join(5.0)
    assert isinstance(out.get("error"), ControlRefused) and "mouse" in str(out["error"])
    assert [c[0] for c in backend.calls].count("type_text") == 2  # the rest of the text is never typed
    assert backend.calls[-1] == ("release_modifiers", ())


def test_after_resume_a_new_action_is_not_cancelled_by_the_old_takeover() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow", "Continue"])
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    assert ctl.user_input("mouse")
    ctl.resume("a1")
    ctl.act("a1", "type_text", text="hello world, again")
    assert "".join(c[1][0] for c in backend.calls if c[0] == "type_text") == "hello world, again"


def test_open_app_checks_for_a_stop_before_each_step() -> None:
    clock = Clock()
    stopped: dict[str, Any] = {}
    ctl, backend, _ = _control(clock, platform="win32", uia=FakeUIA(focused=TEXT_BOX))

    def sleep(_s: float) -> None:
        if not stopped:
            stopped["x"] = ctl.stop("you said stop")

    ctl.hooks.sleep = sleep
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="stop"):
        ctl.act("a1", "open_app", name="Notepad")
    assert backend.calls == [("key", ("cmd",)), ("release_modifiers", ())]  # never typed the name or pressed Enter
