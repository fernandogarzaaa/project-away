"""Undo progress tracking (audit 2026-10-20, claim B): an undo runs its inverse steps one by one, and a
failure part-way used to reset the entry to "done" even when some inverses had already executed - so a
retry re-ran the whole inverse and applied the completed steps twice.

Undo now records each inverse step's outcome on the entry. An interruption before any inverse input
leaves the entry fully retryable. A failure after some steps executed leaves a distinct
"partially undone" state: completed and failed steps are never blindly repeated, a retry runs only the
steps that were never attempted, and the per-step outcome is reported. A failure with nothing
completed leaves the entry fully undoable (main's rule: a transient backend failure must not brick
its undo); the never-repeat rule still protects completed steps. Sequence parents are not
session-trail rows on main (sub-steps only), so a sequence's per-step outcome surfaces via the
undo result, not the trail. Receipt "Undo all" stays scoped
to the receipt's own items.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from away.actions_engine import ActionCancelled
from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeUIA:
    available = True

    def tree(self, depth: int, nodes: int) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        return Element("Canvas", "PaneControl")

    def focused(self) -> Element | None:
        return None

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None


class FlakyScrollBackend(RecordingBackend):
    """Counts every scroll attempt; attempts whose number is in fail_on raise instead of recording."""

    def __init__(self) -> None:
        super().__init__()
        self.scroll_calls = 0
        self.fail_on: set[int] = set()

    def scroll(self, dy: float) -> None:
        self.scroll_calls += 1
        if self.scroll_calls in self.fail_on:
            raise RuntimeError("injected scroll failure")
        super().scroll(dy)


def _control(tmp_path: Path, *, backend: Any = None, step_budget: int = 40) -> ComputerControl:
    hooks = ControlHooks(
        backend=backend or RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=FakeUIA(),
    )
    hooks.cursor_pos = lambda: (7, 8)
    return ComputerControl(
        hooks,
        lambda _q, _o, **kw: {"answer": "Allow", "method": "click"},
        ttl_s=1800,
        rate_per_s=10000.0,
        log_folder=tmp_path / "actions",
        clock=Clock(),
        step_budget=step_budget,
    )


def _scroll_sequence(c: ComputerControl, *dys: int) -> Any:
    c.act_sequence("agent1", [{"action": "scroll", "dy": dy} for dy in dys])
    entry = c.log[-1]
    assert entry.result == "done" and entry.undo is not None
    return entry


def _recorded(backend: RecordingBackend, dy: float) -> int:
    return len([call for call in backend.calls if call == ("scroll", (dy,))])


# ------------------------------------------------------- cancellation before any inverse input


def test_undo_cancelled_before_any_inverse_input_stays_fully_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FlakyScrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    entry = _scroll_sequence(c, 5, 7)

    # The cancel token fires at the first inverse step's check, before any input is sent
    # (exactly what Action.step does when the action was cancelled during setup).
    def cancelled_step(a: Any, b: Any, u: dict[str, Any]) -> None:
        raise ActionCancelled("the person took over")

    monkeypatch.setattr(ComputerControl, "_apply_undo_step", staticmethod(cancelled_step))
    with pytest.raises(ControlRefused):
        c.undo_last("agent1")
    monkeypatch.undo()

    assert entry.result == "done", "nothing was applied, so the entry must stay fully undoable"
    assert entry.undo_progress == ["pending", "pending"], "the attempt is tracked, nothing attempted"
    assert backend.scroll_calls == 2, "only the two forward scrolls reached the backend"

    res = c.undo_last("agent1")
    assert entry.result == "undone"
    assert [s["status"] for s in res["undo_steps"]] == ["done", "done"]
    assert _recorded(backend, -7) == 1 and _recorded(backend, -5) == 1


# ------------------------------------------------------- failure mid-sequence: partial state


def test_undo_failure_mid_sequence_marks_partial_and_retry_repeats_nothing(tmp_path: Path) -> None:
    backend = FlakyScrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    entry = _scroll_sequence(c, 5, 7)
    # Inverses run newest-first: scroll(-7) is attempt 3, scroll(-5) is attempt 4. Fail attempt 4.
    backend.fail_on = {4}
    with pytest.raises(RuntimeError):
        c.undo_last("agent1")

    assert entry.result == "partially undone", "a partially executed undo must not read as fully undoable"
    assert entry.undo_progress == ["done", "failed"]
    # No trail assertion: a sequence parent is not a session-trail row on main (sub-steps only),
    # so its per-step outcome surfaces via the entry and the undo result, not the trail.

    with pytest.raises(ControlRefused, match="partly undone"):
        c.undo_last("agent1")
    assert backend.scroll_calls == 4, "the retry must not re-attempt any inverse step"
    assert _recorded(backend, -7) == 1, "the completed inverse was applied exactly once"
    assert entry.result == "partially undone"


def test_undo_resume_runs_only_the_unattempted_steps(tmp_path: Path) -> None:
    backend = FlakyScrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    entry = _scroll_sequence(c, 1, 2, 3)
    # Forward scrolls are attempts 1-3; inverses newest-first: -3 (ok), -2 (fail), -1 (never attempted).
    backend.fail_on = {5}
    with pytest.raises(RuntimeError):
        c.undo_last("agent1")
    assert entry.result == "partially undone"
    assert entry.undo_progress == ["done", "failed", "pending"]
    assert c.grant is not None and c.grant.steps_used == 4, "3 sequence steps + 1 undo attempt"

    backend.fail_on = set()
    res = c.undo_last("agent1")
    assert res["partial"] is True, "the resume reports that one inverse could not be completed"
    assert [s["status"] for s in res["undo_steps"]] == ["done", "failed", "done"]
    assert entry.result == "partially undone"
    assert _recorded(backend, -3) == 1, "the completed inverse was not repeated"
    assert _recorded(backend, -1) == 1, "the unattempted inverse ran on the resume"
    assert backend.scroll_calls == 6, "the failed inverse was not re-attempted (5 before + 1 resume)"
    assert c.grant.steps_used == 5, "the resume attempt spends one step, like any undo attempt"


def test_single_scroll_undo_failure_stays_retryable(tmp_path: Path) -> None:
    # Composition decision, documented in the module docstring: nothing completed means the
    # entry stays fully undoable. The old #80 expectation (a failed inverse bricks its undo)
    # contradicted main's "undo failed" contract for transient backend failures.
    backend = FlakyScrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    c.act("agent1", "scroll", dy=5)
    entry = c.log[-1]
    backend.fail_on = {2}  # the inverse scroll(-5) itself raises
    with pytest.raises(RuntimeError):
        c.undo_last("agent1")
    assert entry.result == "done"  # nothing completed: the undo may be retried
    backend.fail_on = set()
    res = c.undo_last("agent1")
    assert "partial" not in res
    assert entry.result == "undone"
    assert backend.scroll_calls == 3, "forward + failed attempt + genuine retry"
    assert _recorded(backend, -5) == 1, "the retry genuinely re-ran the inverse once"


def test_scroll_with_both_components_failing_mid_step_is_uncertain_not_retryable(tmp_path: Path) -> None:
    # Review finding on #97: a scroll inverse with dy and dx is two backend calls, so a
    # failure on the second leaves the first applied. Resetting it to pending would
    # double-apply dy on retry, so the entry is partly undone and the step never repeats.
    class FailHscrollBackend(FlakyScrollBackend):
        fail_hscroll = False

        def hscroll(self, dx: float) -> None:
            if self.fail_hscroll:
                raise RuntimeError("injected hscroll failure")
            super().hscroll(dx)

    backend = FailHscrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    c.act("agent1", "scroll", dy=5, dx=3)
    entry = c.log[-1]
    backend.fail_hscroll = True
    with pytest.raises(RuntimeError, match="injected hscroll failure"):
        c.undo_last("agent1")
    assert entry.result == "partially undone"
    assert entry.undo_progress == ["failed"]
    assert _recorded(backend, -5) == 1, "the vertical inverse applied exactly once"
    assert ("hscroll", (-3,)) not in backend.calls, "the failed horizontal inverse never applied"
    with pytest.raises(ControlRefused, match="partly undone"):
        c.undo_last("agent1")
    assert _recorded(backend, -5) == 1, "the retry must not re-apply the vertical inverse"


def test_undo_cancelled_mid_sequence_never_repeats_completed_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FlakyScrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    entry = _scroll_sequence(c, 1, 2, 3)

    orig = ComputerControl.__dict__["_apply_undo_step"].__func__
    armed = {"on": True}

    def cancelling_step(a: Any, b: Any, u: dict[str, Any]) -> None:
        if armed["on"] and u.get("scroll") == [-2, 0]:
            raise ActionCancelled("stopped")  # token fired at this step's check: no input sent for it
        orig(a, b, u)

    monkeypatch.setattr(ComputerControl, "_apply_undo_step", staticmethod(cancelling_step))
    with pytest.raises(ControlRefused):
        c.undo_last("agent1")
    armed["on"] = False

    assert entry.result == "partially undone"
    assert entry.undo_progress == ["done", "cancelled", "pending"]
    res = c.undo_last("agent1")
    assert res["partial"] is True
    assert [s["status"] for s in res["undo_steps"]] == ["done", "cancelled", "done"]
    assert _recorded(backend, -3) == 1, "the completed inverse ran exactly once"
    assert _recorded(backend, -2) == 0, "the cancelled inverse was not blindly repeated"
    assert _recorded(backend, -1) == 1, "the unattempted inverse ran on the retry"


# ------------------------------------------------------- successful undo reporting


def test_successful_undo_reports_every_step(tmp_path: Path) -> None:
    backend = FlakyScrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    entry = _scroll_sequence(c, 5, 7)
    res = c.undo_last("agent1")
    assert entry.result == "undone"
    assert entry.undo_progress == ["done", "done"]
    assert [s["status"] for s in res["undo_steps"]] == ["done", "done"]
    # No trail assertion here either: see the mid-sequence test's comment.


# ------------------------------------------------------- receipt Undo All scoping


def test_undo_all_stays_scoped_to_the_receipts_own_session(tmp_path: Path) -> None:
    backend = RecordingBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    c.act("agent1", "scroll", dy=5)
    first = c.log[-1]
    g1 = c.grant
    assert g1 is not None
    c.stop("stopped")
    ids1 = [i["id"] for i in c.receipt(g1)["items"]]

    c.request("agent1", "Agent1")
    c.act("agent1", "scroll", dy=9)
    second = c.log[-1]
    g2 = c.grant
    assert g2 is not None
    c.stop("stopped")
    ids2 = [i["id"] for i in c.receipt(g2)["items"]]

    assert ids1 == [first.id] and ids2 == [second.id]
    res = c.undo_all_by_user(ids1)
    assert res["count"] == 1
    assert first.result == "undone"
    assert second.result == "done", "Undo all on session 1's receipt must not touch session 2"
    res2 = c.undo_all_by_user(ids2)
    assert res2["count"] == 1
    assert second.result == "undone"


def test_undo_all_resumes_a_partial_entry_and_reports_it(tmp_path: Path) -> None:
    backend = FlakyScrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    entry = _scroll_sequence(c, 1, 2, 3)
    backend.fail_on = {5}  # second inverse fails during an agent undo, leaving step 3 unattempted
    with pytest.raises(RuntimeError):
        c.undo_last("agent1")
    assert entry.result == "partially undone"
    g = c.grant
    assert g is not None
    c.stop("stopped")
    receipt = c.receipt(g)
    item = next(i for i in receipt["items"] if i["id"] == entry.id)
    assert item["undoable"] is False, "a partially undone entry is not offered as plainly undoable"
    assert "partly put back" in item["text"]

    backend.fail_on = set()
    res = c.undo_all_by_user([entry.id])
    assert res["count"] == 0, "the sequence as a whole was not fully put back"
    assert res.get("partial") is True and res["failed"], "the partial entry is reported, not hidden"
    assert _recorded(backend, -1) == 1, "the press resumed the one unattempted inverse"
    assert _recorded(backend, -3) == 1 and backend.scroll_calls == 6, "nothing completed was repeated"
    assert entry.result == "partially undone"


def test_undo_all_reports_a_backend_failure_instead_of_raising(tmp_path: Path) -> None:
    backend = FlakyScrollBackend()
    c = _control(tmp_path, backend=backend)
    c.request("agent1", "Agent1")
    c.act("agent1", "scroll", dy=5)
    entry = c.log[-1]
    g = c.grant
    assert g is not None
    c.stop("stopped")
    ids = [i["id"] for i in c.receipt(g)["items"]]
    backend.fail_on = {2}  # the inverse raises during the press itself
    res = c.undo_all_by_user(ids)
    assert res["count"] == 0
    assert res.get("partial") is True and "injected scroll failure" in res["failed"][0]
    # The failure itself is reported above; the entry stays fully undoable because nothing
    # completed (same composition decision as the single-failure test).
    assert entry.result == "done"
