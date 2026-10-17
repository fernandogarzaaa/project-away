"""v0.1.29: one camera open at a time per process.

The runtime log showed native access violations in multicam/capture.py (apply_mode,
_create, _open_camera) while the first-run tour camera probe and a settings diagnose
probe ran on two threads at once. open_camera() counted in-progress opens via
_opening but never serialized them: two threads could be inside _open_camera
simultaneously, racing in native DirectShow/MediaFoundation/WinRT code.

The fix: a process-wide _open_gate held for the whole _open_camera() call, and
wait_devices_free() waits for the gate (after setting _abort_opens) before it
declares the devices free for hand-over.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time

import pytest

from away.multicam import capture


@pytest.fixture(autouse=True)
def _clean_camera_state():
    capture._abort_opens.clear()
    capture._open_now.clear()
    capture._open_state.handover_gen = None
    yield
    capture._abort_opens.clear()
    capture._open_now.clear()
    capture._open_state.handover_gen = None
    assert capture.opens_in_progress() == 0


def test_concurrent_opens_never_overlap_inside_open_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two threads in open_camera() must never be inside _open_camera at once.

    Regression test for the native AV: fails on unfixed code (max concurrency 2),
    passes with the gate (max concurrency 1).
    """
    state = {"current": 0, "max": 0}
    state_lock = threading.Lock()
    go = threading.Barrier(2)

    def fake_open_camera(src: object, *args: object, **kwargs: object) -> None:
        with state_lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        try:
            time.sleep(0.2)  # real sleep: without the gate the other thread overlaps here
        finally:
            with state_lock:
                state["current"] -= 1
        return None

    monkeypatch.setattr(capture, "_open_camera", fake_open_camera)

    errors: list[BaseException] = []

    def run() -> None:
        go.wait(5)
        try:
            capture.open_camera(0, cv2=object())
        except BaseException as exc:  # the test must not hide thread errors
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
        assert not t.is_alive()
    assert not errors
    assert state["max"] == 1


def test_wait_devices_free_aborts_an_in_progress_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hand-over aborts a mid-flight open via _abort_opens and then reports free.

    Contract test for the wait_devices_free side of the fix: the in-progress open
    sees the abort event (like the real _AttemptSearch._check) and gives up, so
    wait_devices_free can take the gate and return True instead of timing out.
    """
    entered = threading.Event()
    outcome: dict[str, BaseException] = {}

    def fake_open_camera(src: object, *args: object, **kwargs: object) -> None:
        entered.set()
        while not capture._abort_opens.is_set():
            time.sleep(0.01)
        raise capture.OpenCancelled("camera 0: open cancelled (the cameras are being handed over)")

    monkeypatch.setattr(capture, "_open_camera", fake_open_camera)

    def run_open() -> None:
        try:
            capture.open_camera(0, cv2=object())
        except BaseException as exc:  # record, do not hide
            outcome["exc"] = exc

    t = threading.Thread(target=run_open, name="test-open")
    t.start()
    assert entered.wait(5)
    assert capture.wait_devices_free(timeout=5.0) is True
    t.join(5)
    assert not t.is_alive()
    assert isinstance(outcome.get("exc"), capture.OpenCancelled)


def test_open_camera_passes_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_camera forwards its documented default geometry to _open_camera."""
    seen: dict[str, object] = {}

    def fake_open_camera(src: object, *args: object, **kwargs: object) -> None:
        seen["args"] = (src, *args)
        seen["kwargs"] = kwargs
        return None

    monkeypatch.setattr(capture, "_open_camera", fake_open_camera)
    capture.open_camera(0, cv2=object())
    assert seen["args"] == (0, 640, 480, 0, capture.DEFAULT_FOURCC)
    assert seen["kwargs"]["virtual_fallback"] is False  # type: ignore[index]


def test_wait_devices_free_default_timeout_and_step() -> None:
    """The hand-over wait keeps its documented 15 s timeout and 0.05 s step."""
    sig = inspect.signature(capture.wait_devices_free)
    assert sig.parameters["timeout"].default == 15.0
    assert sig.parameters["step"].default == 0.05


def test_wait_devices_free_gives_up_waiting_for_gate_after_timeout() -> None:
    """A stuck native open must not hang hand-over: remaining is deadline - now.

    wait_devices_free waits for the gate only up to the timeout, then proceeds
    to the drain loop instead of blocking forever.
    """
    outcome: dict[str, object] = {}
    capture._open_gate.acquire()
    try:

        def run() -> None:
            start = time.monotonic()
            outcome["result"] = capture.wait_devices_free(timeout=0.3)
            outcome["elapsed"] = time.monotonic() - start

        t = threading.Thread(target=run, name="test-wait-free")
        t.start()
        t.join(5)
        assert not t.is_alive(), "wait_devices_free hung waiting for the gate"
    finally:
        capture._open_gate.release()
    assert outcome["result"] is True
    assert outcome["elapsed"] < 2.0  # type: ignore[operator]


def test_wait_devices_free_uses_remaining_timeout_for_gate() -> None:
    """wait_devices_free really waits for the gate while time remains.

    The gate holder releases after 0.3 s; with timeout=0.5 the wait must take
    the gate instead of skipping it.
    """
    acquired = threading.Event()
    release_gate = threading.Event()

    def holder() -> None:
        capture._open_gate.acquire()
        acquired.set()
        assert release_gate.wait(5)
        capture._open_gate.release()

    h = threading.Thread(target=holder, name="test-gate-holder")
    h.start()
    try:
        assert acquired.wait(5)
        threading.Timer(0.3, release_gate.set).start()
        start = time.monotonic()
        assert capture.wait_devices_free(timeout=0.5) is True
        assert time.monotonic() - start >= 0.2
    finally:
        release_gate.set()
        h.join(5)


class _GateSpy:
    """Records acquire calls while delegating to the real gate."""

    def __init__(self, real: threading.Lock) -> None:
        self._real = real
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def acquire(self, *args: object, **kwargs: object) -> bool:
        self.calls.append((args, kwargs))
        return self._real.acquire(*args, **kwargs)  # type: ignore[arg-type]

    def release(self) -> None:
        self._real.release()

    def __enter__(self) -> _GateSpy:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


def test_wait_devices_free_does_not_take_gate_without_time_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no time remaining the gate must not be touched at all."""
    spy = _GateSpy(capture._open_gate)
    monkeypatch.setattr(capture, "_open_gate", spy)
    now = [1000.0]
    assert capture.wait_devices_free(timeout=0.0, clock=lambda: now[0], sleep=lambda s: None) is True
    assert spy.calls == []


def test_wait_devices_free_stops_at_deadline() -> None:
    """The drain loop stops as soon as the clock reaches the deadline."""
    now = [1000.0]
    sleeps: list[float] = []
    capture._hold("0", 1)  # fake a held camera so the loop has to wait
    try:

        def sleep(s: float) -> None:
            sleeps.append(s)
            now[0] += s

        assert capture.wait_devices_free(timeout=1.0, clock=lambda: now[0], sleep=sleep, step=0.5) is False
    finally:
        capture._hold("0", -1)
    assert sleeps == [0.5, 0.5]


def test_wait_devices_free_logs_held_cameras_comma_separated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The 'still busy' warning lists held cameras separated by ', '."""
    now = [1000.0]
    capture._hold("0", 1)
    capture._hold("1", 1)
    try:
        with caplog.at_level(logging.WARNING, logger="away.cameras"):
            assert (
                capture.wait_devices_free(
                    timeout=0.5,
                    clock=lambda: now[0],
                    sleep=lambda s: now.__setitem__(0, now[0] + s),
                    step=0.5,
                )
                is False
            )
    finally:
        capture._hold("0", -1)
        capture._hold("1", -1)
    assert "0, 1" in caplog.text


def test_queued_open_is_cancelled_if_handover_finished_while_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open queued on the gate while a hand-over timed out must not open afterwards.

    Regression test for the CodeRabbit finding on the gate fix: if a native open holds
    _open_gate past the hand-over timeout, a queued open_camera() call would previously
    acquire the gate after wait_devices_free() cleared _abort_opens and open a camera
    after the runner handed the devices over. The hand-over generation must cancel it.
    """
    opened: list[bool] = []
    errors: list[BaseException] = []

    def fake_open_camera(src: object, *args: object, **kwargs: object) -> None:
        opened.append(True)
        return None

    monkeypatch.setattr(capture, "_open_camera", fake_open_camera)

    # Wedge the gate: simulates a native open stuck past the hand-over timeout.
    assert capture._open_gate.acquire(blocking=False)

    def run() -> None:
        try:
            capture.open_camera(0, cv2=object())
        except BaseException as exc:  # the test must not hide thread errors
            errors.append(exc)

    t = threading.Thread(target=run)
    try:
        t.start()
        # Wait until the open is queued (counted, blocked on the gate).
        deadline = time.monotonic() + 5
        while capture.opens_in_progress() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert capture.opens_in_progress() == 1
        time.sleep(0.1)  # let the thread reach the gate
        assert t.is_alive()

        # Hand-over: cannot get the gate, drain loop sees the queued open, times out.
        assert capture.wait_devices_free(timeout=0.5) is False
    finally:
        capture._open_gate.release()

    t.join(10)
    assert not t.is_alive()
    # The queued open must have been cancelled, never reaching _open_camera.
    assert not opened
    assert len(errors) == 1
    assert isinstance(errors[0], capture.OpenCancelled)


def test_queued_open_proceeds_when_no_handover_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generation check must not cancel opens when no hand-over happened."""
    opened: list[bool] = []

    def fake_open_camera(src: object, *args: object, **kwargs: object) -> None:
        opened.append(True)
        return None

    monkeypatch.setattr(capture, "_open_camera", fake_open_camera)

    assert capture._open_gate.acquire(blocking=False)

    def run() -> None:
        capture.open_camera(0, cv2=object())

    t = threading.Thread(target=run)
    try:
        t.start()
        deadline = time.monotonic() + 5
        while capture.opens_in_progress() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert capture.opens_in_progress() == 1
        time.sleep(0.1)
    finally:
        capture._open_gate.release()

    t.join(10)
    assert not t.is_alive()
    assert opened == [True]


def test_handover_generation_increments_by_exactly_one() -> None:
    """wait_devices_free must bump the generation counter by exactly one per run.

    Pins the counter semantics the stale-open cancellation relies on: a queued open
    compares the recorded generation against the current one, so the increment must
    be exactly one (not zero, not two, not a decrement).
    """
    before = capture._handover_gen
    assert capture.wait_devices_free(timeout=0.5) is True
    assert capture._handover_gen == before + 1
    assert capture.wait_devices_free(timeout=0.5) is True
    assert capture._handover_gen == before + 2


def test_search_cancelled_sees_handover_that_finished_mid_open() -> None:
    """_Search.cancelled() must fire when the generation changed mid-open.

    Regression test for the follow-up CodeRabbit finding: the gate-acquisition check
    alone leaves a window where a hand-over finishes after the open takes the gate
    but before _open_camera's first _check(). Publishing the generation thread-locally
    closes it: every in-flight check sees the finished hand-over.
    """
    from away.multicam.capture import _Search

    search = _Search("cam", object(), 1.0, 1.0, time.monotonic, time.sleep, 30.0, 0.5)
    # Simulate open_camera() having published this thread's generation.
    capture._open_state.handover_gen = capture._handover_gen
    try:
        assert search.cancelled() is False
        # A hand-over runs to completion while the open is in flight.
        assert capture.wait_devices_free(timeout=0.5) is True
        assert search.cancelled() is True
        with pytest.raises(capture.OpenCancelled):
            search._check()
    finally:
        capture._open_state.handover_gen = None


def test_search_without_thread_generation_uses_event_only() -> None:
    """_Search without a thread-local generation behaves exactly like before (event only)."""
    from away.multicam.capture import _Search

    capture._open_state.handover_gen = None
    search = _Search("cam", object(), 1.0, 1.0, time.monotonic, time.sleep, 30.0, 0.5)
    assert search.cancelled() is False
    capture._abort_opens.set()
    try:
        assert search.cancelled() is True
    finally:
        capture._abort_opens.clear()


def test_open_camera_publishes_generation_thread_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_camera must publish its recorded generation for the calling thread."""
    seen: dict[str, object] = {}

    def fake_open_camera(src: object, *args: object, **kwargs: object) -> None:
        seen["thread_gen"] = getattr(capture._open_state, "handover_gen", "MISSING")
        return None

    monkeypatch.setattr(capture, "_open_camera", fake_open_camera)
    gen_before = capture._handover_gen
    capture.open_camera(0, cv2=object())
    assert seen.get("thread_gen") == gen_before
    # The generation must not leak to later opens on this thread.
    assert getattr(capture._open_state, "handover_gen", "MISSING") is None
