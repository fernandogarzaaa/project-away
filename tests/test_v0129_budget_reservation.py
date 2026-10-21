"""v0.1.29 audit fix: sequence budget reservation + whole-operation serialization.

Defects (audit 2026-10-20, phase2-verification.md sections C and C2, reproduced there):

* ``act_sequence()`` preflighted the remaining budget but reserved nothing, so a concurrent
  action could spend budget after the preflight and the sequence refused mid-way, after
  earlier sub-steps had already produced side effects.
* The action engine serialized single OS calls only (``_io``), so the individual steps of
  two agent operations (a drag's press, moves, release) could interleave.

Fix: a sequence reserves its steps atomically at admission (``Grant.steps_reserved``);
unused reserved steps are released when the sequence is stopped or refused part-way, and
``steps_used`` keeps the documented policy (agent-issued attempts count, including undo).
The engine holds a whole-operation lock for agent sources across ``begin()``; waiting for
it is cancellation-aware, so takeover, stop and shutdown never deadlock behind it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from away.actions_engine import AGENT as ENGINE_AGENT
from away.actions_engine import ActionCancelled, ActionEngine
from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element
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

    def __init__(self, at: Any = None) -> None:
        self._at = at

    def focused(self) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        return self._at() if callable(self._at) else self._at

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None


def _control(
    tmp_path: Path,
    *,
    backend: Any = None,
    step_budget: int = 40,
    clock: Clock | None = None,
) -> ComputerControl:
    hooks = ControlHooks(
        backend=backend or RecordingBackend(),
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
        # "Allow" for the approval prompt, "Continue" for the resume prompt
        lambda _q, _o, **kw: {"answer": "Continue" if kw.get("title") == "Resume?" else "Allow", "method": "click"},
        ttl_s=1800,
        rate_per_s=10000.0,
        log_folder=tmp_path / "actions",
        clock=clock or Clock(),
        step_budget=step_budget,
    )


class GatedScrollBackend(RecordingBackend):
    """Blocks inside the Nth scroll call until the test releases it."""

    def __init__(self, block_on: int) -> None:
        super().__init__()
        self.block_on = block_on
        self.count = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def scroll(self, dy: float) -> None:
        self.count += 1
        if self.count == self.block_on:
            self.entered.set()
            self.release.wait(timeout=10)
        super().scroll(dy)


class GatedPressBackend(RecordingBackend):
    """Records the press, then blocks until the test releases it (the button stays held)."""

    def __init__(self) -> None:
        super().__init__()
        self.press_entered = threading.Event()
        self.release_press = threading.Event()

    def press(self) -> None:
        super().press()
        self.press_entered.set()
        self.release_press.wait(timeout=10)


def _run(fn: Any) -> tuple[threading.Thread, dict[str, Any]]:
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = fn()
        except Exception as exc:  # captured, not swallowed: the test asserts on the outcome
            outcome["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, outcome


def _scrolls(backend: RecordingBackend) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in backend.calls if c[0] == "scroll"]


# ------------------------------------------------------------- budget reservation


def test_sequence_reservation_completes_despite_a_concurrent_action(tmp_path: Path) -> None:
    backend = GatedScrollBackend(block_on=1)
    ctl = _control(tmp_path, backend=backend, step_budget=3)
    ctl.request("a1", "Agent1")
    t, outcome = _run(
        lambda: ctl.act_sequence(
            "a1",
            [{"action": "scroll", "dy": 1}, {"action": "scroll", "dy": 2}, {"action": "scroll", "dy": 3}],
        )
    )
    assert backend.entered.wait(timeout=10)  # the sequence was admitted and is inside sub-step 1
    # The whole budget is reserved by the admitted sequence: a concurrent action is refused
    # up front instead of spending a step the sequence is counting on.
    with pytest.raises(ControlRefused):
        ctl.act("a1", "move", x=1, y=1)
    backend.release.set()
    t.join(15)
    assert not t.is_alive()
    assert outcome.get("result", {}).get("ok") is True, f"sequence must run to completion: {outcome}"
    assert len(_scrolls(backend)) == 3
    g = ctl.grant
    assert g is not None and g.steps_used == 3 and g.steps_reserved == 0


def test_two_concurrent_sequences_only_one_is_admitted(tmp_path: Path) -> None:
    backend = GatedScrollBackend(block_on=1)
    ctl = _control(tmp_path, backend=backend, step_budget=5)
    ctl.request("a1", "Agent1")
    t, outcome = _run(
        lambda: ctl.act_sequence(
            "a1",
            [{"action": "scroll", "dy": 1}, {"action": "scroll", "dy": 2}, {"action": "scroll", "dy": 3}],
        )
    )
    assert backend.entered.wait(timeout=10)  # sequence A admitted (3 of 5 reserved), inside sub-step 1
    # Sequence B needs 3 more but only 2 are unreserved: refused at admission, zero side effects.
    with pytest.raises(ControlRefused):
        ctl.act_sequence(
            "a1",
            [{"action": "scroll", "dy": 7}, {"action": "scroll", "dy": 8}, {"action": "scroll", "dy": 9}],
        )
    assert _scrolls(backend) == []  # B ran nothing at all (A's first scroll is still gated)
    backend.release.set()
    t.join(15)
    assert not t.is_alive()
    assert outcome.get("result", {}).get("ok") is True, f"sequence A must complete: {outcome}"
    assert [c[1] for c in _scrolls(backend)] == [(1,), (2,), (3,)]
    g = ctl.grant
    assert g is not None and g.steps_used == 3 and g.steps_reserved == 0
    # No reservation leaked: exactly the 2 remaining steps are still usable.
    ctl.act_sequence("a1", [{"action": "scroll", "dy": 4}, {"action": "scroll", "dy": 5}])
    assert ctl.grant is not None and ctl.grant.steps_used == 5
    with pytest.raises(ControlRefused):
        ctl.act("a1", "scroll", dy=6)


def test_reservation_refusal_happens_before_any_side_effect(tmp_path: Path) -> None:
    backend = GatedScrollBackend(block_on=1)
    ctl = _control(tmp_path, backend=backend, step_budget=4)
    ctl.request("a1", "Agent1")
    t, outcome = _run(
        lambda: ctl.act_sequence(
            "a1",
            [{"action": "scroll", "dy": 1}, {"action": "scroll", "dy": 2}, {"action": "scroll", "dy": 3}],
        )
    )
    assert backend.entered.wait(timeout=10)  # A admitted: 1 used + 2 reserved, 1 step truly left
    # A 2-step sequence does not fit in the 1 unreserved step: refused at reservation time.
    with pytest.raises(ControlRefused):
        ctl.act_sequence("a1", [{"action": "scroll", "dy": 7}, {"action": "scroll", "dy": 8}])
    assert _scrolls(backend) == [], "the refused sequence must not have executed anything"
    backend.release.set()
    t.join(15)
    assert not t.is_alive()
    assert outcome.get("result", {}).get("ok") is True
    assert ctl.grant is not None and ctl.grant.steps_used == 3 and ctl.grant.steps_reserved == 0


def test_a_stopped_sequence_releases_its_unused_reservation(tmp_path: Path) -> None:
    clock = Clock()
    backend = GatedScrollBackend(block_on=2)
    ctl = _control(tmp_path, backend=backend, step_budget=3, clock=clock)
    ctl.request("a1", "Agent1")
    t, outcome = _run(
        lambda: ctl.act_sequence(
            "a1",
            [{"action": "scroll", "dy": 1}, {"action": "scroll", "dy": 2}, {"action": "scroll", "dy": 3}],
        )
    )
    assert backend.entered.wait(timeout=10)  # sub-step 2 executing: 2 used, 1 still reserved
    with pytest.raises(ControlRefused):  # nothing is left for a concurrent action
        ctl.act("a1", "move", x=1, y=1)
    clock.advance(60.0)  # past the approval grace and the agent's own-input echo windows
    assert ctl.user_input("mouse") is True  # the person takes over: the sequence is cancelled
    backend.release.set()
    t.join(15)
    assert not t.is_alive()
    err = outcome.get("error")
    assert isinstance(err, ControlRefused) and "stopped part-way" in str(err), f"unexpected outcome: {outcome}"
    g = ctl.grant
    assert g is not None and g.steps_used == 2, "the two executed sub-steps stay spent (attempts count)"
    assert g.steps_reserved == 0, "the unexecuted sub-step's reservation was released"
    # After resuming, exactly one step remains: one action fits, the next is refused.
    assert ctl.resume("a1")["steps_left"] == 1
    ctl.act("a1", "scroll", dy=9)
    with pytest.raises(ControlRefused):
        ctl.act("a1", "scroll", dy=10)


# ------------------------------------------------------- operation serialization


def test_agent_operations_do_not_interleave_their_steps(tmp_path: Path) -> None:
    backend = GatedPressBackend()
    ctl = _control(tmp_path, backend=backend, step_budget=40)
    ctl.request("a1", "Agent1")
    ta, out_a = _run(lambda: ctl.act("a1", "drag", x=0, y=0, x2=10, y2=10))
    assert backend.press_entered.wait(timeout=10)  # the drag holds its button and is mid-operation
    b_started = threading.Event()

    def click_b() -> dict[str, Any]:
        b_started.set()
        return ctl.act("a1", "click", x=99, y=99)

    tb, out_b = _run(click_b)
    assert b_started.wait(timeout=10)
    time.sleep(0.5)  # give B every chance to interleave (on unfixed code it finishes here)
    backend.release_press.set()
    ta.join(15)
    tb.join(15)
    assert not ta.is_alive() and not tb.is_alive(), "an operation deadlocked"
    assert out_a.get("result", {}).get("ok") is True, f"drag failed: {out_a}"
    assert out_b.get("result", {}).get("ok") is True, f"click failed: {out_b}"
    calls = backend.calls
    press_at = calls.index(("press", ()))
    release_at = calls.index(("release", ()))
    between = calls[press_at + 1 : release_at]
    assert all(c[0] == "move" for c in between), f"foreign steps interleaved inside the drag: {calls}"
    assert ("move", (99, 99)) not in between and ("click", (1,)) not in between
    assert calls.index(("move", (99, 99))) > release_at, f"the click ran inside the drag: {calls}"
    assert calls.index(("click", (1,))) > release_at


def test_takeover_interrupts_the_running_operation_and_refuses_the_waiting_one(tmp_path: Path) -> None:
    clock = Clock()
    backend = GatedPressBackend()
    ctl = _control(tmp_path, backend=backend, step_budget=40, clock=clock)
    ctl.request("a1", "Agent1")
    ta, out_a = _run(lambda: ctl.act("a1", "drag", x=0, y=0, x2=10, y2=10))
    assert backend.press_entered.wait(timeout=10)  # drag in progress, button held
    b_started = threading.Event()

    def click_b() -> dict[str, Any]:
        b_started.set()
        return ctl.act("a1", "click", x=99, y=99)

    tb, out_b = _run(click_b)
    assert b_started.wait(timeout=10)
    time.sleep(0.3)  # B is now queued behind the drag's operation (or running, on unfixed code)
    clock.advance(60.0)  # past the approval grace and echo windows
    assert ctl.user_input("mouse") is True  # the person takes over
    backend.release_press.set()
    ta.join(15)
    tb.join(15)
    assert not ta.is_alive() and not tb.is_alive(), "takeover deadlocked behind the operation lock"
    err_a = out_a.get("error")
    assert isinstance(err_a, ControlRefused) and "stopped part-way" in str(err_a), f"drag outcome: {out_a}"
    # The held button was let go, not left pressed on the person's desktop.
    assert ("release", ()) in backend.calls[backend.calls.index(("press", ())) :]
    # The queued operation was refused, never executed: no click, no move to its target.
    assert isinstance(out_b.get("error"), ControlRefused), f"queued click outcome: {out_b}"
    assert ("click", (1,)) not in backend.calls and ("move", (99, 99)) not in backend.calls
    assert ctl.grant is not None and ctl.grant.paused


def test_engine_stop_all_reaches_an_operation_waiting_for_its_turn() -> None:
    backend = GatedPressBackend()
    eng = ActionEngine(backend=backend)
    out_a: dict[str, Any] = {}
    out_b: dict[str, Any] = {}

    def hold() -> None:
        try:
            with eng.begin(ENGINE_AGENT, "drag", actor="a1", backend=backend) as a:
                a.step().press()
                a.step().release()
            out_a["completed"] = True
        except ActionCancelled:
            out_a["cancelled"] = True

    def queued() -> None:
        out_b["started"] = True
        try:
            with eng.begin(ENGINE_AGENT, "click", actor="a1", backend=backend) as a:
                a.step().click(1)
            out_b["completed"] = True
        except ActionCancelled:
            out_b["cancelled"] = True

    ta = threading.Thread(target=hold, daemon=True)
    ta.start()
    assert backend.press_entered.wait(timeout=10)  # the first operation holds the floor
    tb = threading.Thread(target=queued, daemon=True)
    tb.start()
    deadline = time.monotonic() + 10
    while "started" not in out_b and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.3)  # the second operation is queued behind the first
    eng.stop_all("test stop")
    tb.join(5)
    assert not tb.is_alive(), "stop_all deadlocked behind the running operation"
    assert out_b.get("cancelled") is True, f"the queued operation was not cancelled: {out_b}"
    assert ("click", (1,)) not in backend.calls
    backend.release_press.set()
    ta.join(5)
    assert not ta.is_alive()
    assert out_a.get("cancelled") is True, f"the running operation was not cancelled: {out_a}"


def test_op_lock_is_released_when_the_post_acquire_check_raises() -> None:
    """Regression (CodeRabbit review on PR #75): ``_acquire_op`` runs one final ``a.check()``
    AFTER the operation lock is acquired. If that check raises (the action was cancelled in
    the window between acquiring and checking), the lock must be released exactly once, or
    every later serialized operation waits forever on a lock owned by a thread that left."""
    backend = RecordingBackend()
    eng = ActionEngine(backend=backend)
    token = threading.Event()
    token.set()  # already cancelled: the wait loop is skipped, the post-acquire check raises
    first: dict[str, Any] = {}

    def cancelled_first() -> None:
        try:
            with eng.begin(ENGINE_AGENT, "click", actor="a1", token=token, backend=backend):
                first["entered"] = True
        except ActionCancelled:
            first["cancelled"] = True

    t1 = threading.Thread(target=cancelled_first, daemon=True)
    t1.start()
    t1.join(5)
    assert not t1.is_alive(), "the cancelled operation hung instead of raising"
    assert first.get("cancelled") is True and "entered" not in first, f"unexpected outcome: {first}"
    # The lock must be free again (probe with a timeout so a leak fails fast, never hangs).
    assert eng._op_lock.acquire(timeout=2), "the operation lock leaked: still held after the exception"
    eng._op_lock.release()
    # Behaviorally: a later operation acquires the lock and runs to completion.
    second: dict[str, Any] = {}

    def second_op() -> None:
        try:
            with eng.begin(ENGINE_AGENT, "click", actor="a1", backend=backend) as a:
                a.step().click(1)
                a.ok = True
            second["completed"] = True
        except ActionCancelled:
            second["cancelled"] = True

    t2 = threading.Thread(target=second_op, daemon=True)
    t2.start()
    t2.join(5)
    assert not t2.is_alive(), "a later operation deadlocked behind the leaked operation lock"
    assert second.get("completed") is True, f"the later operation did not run: {second}"
    assert ("click", (1,)) in backend.calls

def test_steps_left_includes_inflight_reservations(tmp_path: Path) -> None:
    ctl = _control(tmp_path, step_budget=10)
    ctl.request("agent-a", "Claude")
    grant = ctl.grant
    assert grant is not None

    ctl._reserve_steps(grant, 4)
    assert ctl._steps_left(grant) == 6
    assert ctl.status()["steps_left"] == 6

    # A separate single action must not report reserved steps as available.
    result = ctl.act("agent-a", "move", x=10, y=10)
    assert result["steps_left"] == ctl.status()["steps_left"] == 5


def test_sequence_and_undo_results_report_reserved_aware_steps_left(tmp_path: Path) -> None:
    ctl = _control(tmp_path, step_budget=10)
    ctl.request("agent-a", "Claude")

    result = ctl.act_sequence("agent-a", [{"action": "move", "x": 20, "y": 20}, {"action": "move", "x": 30, "y": 30}])
    assert result["steps_left"] == ctl.status()["steps_left"]
    assert all("steps_left" in step for step in result["steps"])
    assert result["steps"][-1]["steps_left"] == result["steps_left"]

    undone = ctl.undo_last("agent-a")
    assert undone["steps_left"] == ctl.status()["steps_left"]
