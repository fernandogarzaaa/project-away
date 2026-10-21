"""Microphone thread supervision (audit 2026-10-20, phase 4 item 3.4).

Before this fix the runner's mic thread body was
``with MicStream(...): stop_event.wait()``: if the stream failed to open
(device unplugged, driver hiccup, permission denied) the exception left
the thread, and dictation stayed silently dead until the app restarted.
These tests pin the supervised lifecycle instead: the failure is caught
and recorded, the voice state surface shows it, retries back off, a
recovered device reopens the stream, and stop shuts the supervisor down.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np

from away.events import EventBus
from away.voice.always_on import VoiceController
from away.voice.dictation import MicSupervisor

CHUNK = np.zeros(480, dtype=np.float32)


class FakeStream:
    """MicStream stand-in: a context manager the test feeds chunks through."""

    def __init__(self, on_chunk, feed_on_enter: bool = False) -> None:
        self.on_chunk = on_chunk
        self.feed_on_enter = feed_on_enter
        self.entered = False
        self.exited = False

    def __enter__(self) -> FakeStream:
        self.entered = True
        if self.feed_on_enter:
            self.deliver()
        return self

    def __exit__(self, *exc: object) -> None:
        self.exited = True

    def deliver(self) -> None:
        self.on_chunk(CHUNK)


class ScriptedFactory:
    """Stream factory following a script, then a default.

    Script items: an exception instance (the open raises it), "live" (a
    stream opens and waits), "feed" (a stream that delivers one chunk the
    moment it opens). Once the script runs out, ``default`` applies: an
    exception instance to keep failing, or a plain live stream.
    """

    def __init__(self, script: list | None = None, default: object = None) -> None:
        self.script = list(script or [])
        self.default = default
        self.calls = 0
        self.streams: list[FakeStream] = []

    def __call__(self, on_chunk, sample_rate: int) -> FakeStream:
        self.calls += 1
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, BaseException):
            raise item
        stream = FakeStream(on_chunk, feed_on_enter=item == "feed")
        self.streams.append(stream)
        return stream


class RecordingWaiter:
    """Supervisor wait that records requested delays instead of sleeping.

    Returns True (stop) once more than ``stop_after`` waits were recorded.
    """

    def __init__(self, stop_after: int) -> None:
        self.delays: list[float] = []
        self.stop_after = stop_after

    def __call__(self, seconds: float) -> bool:
        self.delays.append(seconds)
        return len(self.delays) > self.stop_after


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# --------------------------------------------------------------------- state


def test_open_failure_is_recorded_and_shown_in_voice_state() -> None:
    bus = EventBus()
    events: list[dict] = []
    bus.subscribe("voice.mic_error", lambda ev: events.append(ev.data))
    vc = VoiceController("push_to_talk")
    factory = ScriptedFactory(default=RuntimeError("device unplugged"))
    sup = MicSupervisor(
        lambda chunk: None,
        16000,
        stream_factory=factory,
        bus=bus,
        on_error=vc.set_mic_error,
        wait=RecordingWaiter(stop_after=2),
    )
    sup.run()  # the recording waiter ends the loop; no thread needed
    assert sup.error is not None and "device unplugged" in sup.error
    assert sup.attempts == 3
    assert not sup.live
    # The failure is on the existing status surface, not just a log line.
    assert vc.state == "mic_error"
    assert len(events) == 1
    assert "device unplugged" in events[0]["error"]


def test_real_micstream_open_failure_is_caught(monkeypatch) -> None:
    # The real MicStream over a fake sounddevice whose InputStream raises:
    # the supervisor must catch it, not let it kill the thread silently.
    def broken_input_stream(**kwargs):
        raise RuntimeError("no input device")

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=broken_input_stream))
    sup = MicSupervisor(lambda chunk: None, 16000, wait=RecordingWaiter(stop_after=0))
    sup.run()
    assert sup.attempts == 1
    assert sup.error is not None and "no input device" in sup.error


# -------------------------------------------------------------------- backoff


def test_repeated_failures_back_off_and_cap() -> None:
    waiter = RecordingWaiter(stop_after=5)
    factory = ScriptedFactory(default=RuntimeError("still gone"))
    sup = MicSupervisor(
        lambda chunk: None,
        16000,
        stream_factory=factory,
        wait=waiter,
        base_delay_s=1.0,
        max_delay_s=8.0,
    )
    sup.run()
    # Doubling from the base, capped at the max: no hot loop, no unbounded wait.
    assert waiter.delays == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
    assert sup.attempts == 6
    assert factory.calls == 6


# ------------------------------------------------------------------ recovery


def test_recovers_and_chunks_flow_again() -> None:
    stop = threading.Event()
    factory = ScriptedFactory([RuntimeError("gone"), RuntimeError("gone")])
    received: list = []
    transitions: list = []
    vc = VoiceController("push_to_talk")

    def on_error(error: str | None) -> None:
        transitions.append(error)
        vc.set_mic_error(error)

    sup = MicSupervisor(
        received.append,
        16000,
        stop_event=stop,
        stream_factory=factory,
        on_error=on_error,
        base_delay_s=0.01,
        max_delay_s=0.02,
    )
    th = threading.Thread(target=sup.run, daemon=True, name="mic")
    th.start()
    try:
        assert _wait_until(lambda: sup.live)
        assert sup.error is None
        assert sup.attempts == 0  # reset by the successful open
        assert transitions[0] is not None and "gone" in transitions[0]
        assert transitions[-1] is None  # recovery was reported too
        assert vc.state == "idle"  # back to the mode state, no error shown
        factory.streams[-1].deliver()
        assert len(received) == 1  # dictation hears again without a restart
    finally:
        stop.set()
        th.join(timeout=2)
    assert not th.is_alive()
    assert factory.streams[-1].exited


def test_stalled_stream_is_reopened() -> None:
    # A stream that opens but never delivers audio (device died mid-stream)
    # must be treated as failed and reopened, not waited on forever.
    clock_now = [0.0]

    def clock() -> float:
        return clock_now[0]

    stop = threading.Event()

    def wait(seconds: float) -> bool:
        clock_now[0] += seconds
        time.sleep(0.001)
        return stop.is_set() or clock_now[0] > 30.0  # fake-time budget ends the run

    factory = ScriptedFactory(["live", "feed"])  # first stream silent, second feeds on open
    received: list = []
    errors: list = []
    sup = MicSupervisor(
        received.append,
        16000,
        stop_event=stop,
        stream_factory=factory,
        on_error=errors.append,
        base_delay_s=0.5,
        max_delay_s=1.0,
        stall_timeout_s=1.0,
        clock=clock,
        wait=wait,
    )
    th = threading.Thread(target=sup.run, daemon=True, name="mic")
    th.start()
    th.join(timeout=10)
    assert not th.is_alive()
    assert factory.calls >= 2  # the silent stream was reopened
    assert any(e and "stopped delivering" in e for e in errors)
    assert received  # the reopened stream delivered audio


def test_supervisor_defaults() -> None:
    sup = MicSupervisor(lambda chunk: None)
    assert sup.sample_rate == 16000
    assert sup.base_delay_s == 1.0
    assert sup.max_delay_s == 30.0
    assert sup.stall_timeout_s == 5.0
    assert sup.error is None
    assert sup.attempts == 0
    assert not sup.live


def test_stall_boundary_needs_elapsed_strictly_past_the_timeout() -> None:
    # The stall check must fire only once the silence is strictly longer
    # than the timeout, measured from the last chunk (not from any other
    # origin). A fake clock starting at 100 makes the arithmetic exact:
    # polls advance 0.25 each, so at clock 102.0 the silence is exactly
    # the 2.0 timeout and the stream must still count as live.
    clock_now = [100.0]
    stop = threading.Event()
    snapshots: list[tuple[float, int]] = []
    factory = ScriptedFactory(default="feed")  # every stream feeds one chunk on open
    errors: list = []
    received: list = []

    def wait(seconds: float) -> bool:
        clock_now[0] += seconds
        snapshots.append((clock_now[0], factory.calls))
        if clock_now[0] > 105.0:
            stop.set()
        return stop.is_set()

    sup = MicSupervisor(
        received.append,
        16000,
        stop_event=stop,
        stream_factory=factory,
        on_error=errors.append,
        base_delay_s=0.5,
        max_delay_s=0.5,
        stall_timeout_s=2.0,
        clock=lambda: clock_now[0],
        wait=wait,
    )
    sup.run()
    assert all(calls == 1 for t, calls in snapshots if t <= 102.25)
    first_reopen = min(t for t, calls in snapshots if calls >= 2)
    assert first_reopen >= 103.0
    assert any(e and "stopped delivering" in e for e in errors)
    assert received


# ------------------------------------------------------------------- shutdown


def test_stop_while_live_closes_the_stream() -> None:
    stop = threading.Event()
    factory = ScriptedFactory()
    sup = MicSupervisor(lambda chunk: None, 16000, stop_event=stop, stream_factory=factory)
    th = threading.Thread(target=sup.run, daemon=True, name="mic")
    th.start()
    assert _wait_until(lambda: sup.live)
    stop.set()
    th.join(timeout=2)
    assert not th.is_alive()  # no leaked supervisor thread
    assert factory.streams[-1].exited


def test_stop_during_backoff_exits_promptly() -> None:
    # Shutdown must not wait out a long backoff delay.
    stop = threading.Event()
    factory = ScriptedFactory(default=RuntimeError("gone"))
    sup = MicSupervisor(
        lambda chunk: None,
        16000,
        stop_event=stop,
        stream_factory=factory,
        base_delay_s=60.0,
        max_delay_s=60.0,
    )
    th = threading.Thread(target=sup.run, daemon=True, name="mic")
    th.start()
    assert _wait_until(lambda: sup.attempts >= 1)
    stop.set()
    th.join(timeout=2)
    assert not th.is_alive()


# ------------------------------------------------------------- state surface


def test_voice_state_mic_error_precedence() -> None:
    published: list[dict] = []

    class Bus:
        def publish(self, topic: str, data: dict) -> None:
            published.append(data)

    vc = VoiceController("push_to_talk", bus=Bus())  # type: ignore[arg-type]
    assert vc.state == "idle"
    vc.set_mic_error("boom")
    assert vc.state == "mic_error"
    assert published[-1]["state"] == "mic_error"
    vc.set_muted(True)
    assert vc.state == "muted"  # the user's own mute still wins
    vc.set_muted(False)
    assert vc.state == "mic_error"  # and the failure returns once unmuted
    vc.set_mic_error(None)
    assert vc.state == "idle"
