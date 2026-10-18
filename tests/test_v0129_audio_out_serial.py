"""v0.1.29: sounddevice output-stream calls are serialized through one lock.

0.1.27 crashed twice with a native access violation inside PortAudio: ``sd.stop()``
from the barge-in interrupt path (``liveagent/speech.py`` ``_stop_sounddevice``) while
TTS played, and ``sd.play()`` from the earcon thread (``voice/earcon.py`` ``_sd_play``)
while ``voice/natural.py`` ``play_chunks`` spoke on another thread. PortAudio stream
handles are not thread safe, and a native access violation cannot be caught from
Python, so every ``sd.play`` / ``sd.stop`` on the default output stream must be
serialized through ``away.voice.audio_out``'s single process-wide lock.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import Any

import numpy as np
import pytest


class _FakeStream:
    """Stand-in for sounddevice's stream: ``active`` until stopped or the fake
    playback elapses, ``close()`` recorded so tests can see the teardown."""

    def __init__(self, play_s: float = 0.05, on_close: Any = None) -> None:
        self._active = True
        self._done_at = time.perf_counter() + play_s
        self.close_calls = 0
        self._on_close = on_close

    @property
    def active(self) -> bool:
        if self._active and time.perf_counter() >= self._done_at:
            self._active = False
        return self._active

    @property
    def closed(self) -> bool:
        return self.close_calls > 0

    def close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        self.close_calls += 1


class _FakeSD:
    """Fake ``sounddevice`` module recording (name, enter, exit) for every call.

    ``play``/``stop`` sleep briefly to widen the race window, the way the real
    native calls do. Each ``play()`` mints a fresh stream (like the real
    convenience API); ``stop()`` deactivates and closes the current stream, the
    way ``sounddevice.stop()`` stops and closes it.
    """

    def __init__(self, play_s: float = 0.05, op_s: float = 0.01) -> None:
        self.events: list[tuple[str, float, float]] = []
        self._ev_lock = threading.Lock()
        self.op_s = op_s
        self.play_s = play_s
        self._stream = _FakeStream(play_s=play_s, on_close=lambda: self._span("close"))
        self.play_calls: list[tuple[Any, int]] = []

    def _span(self, name: str) -> None:
        enter = time.perf_counter()
        time.sleep(self.op_s)  # stand-in for the native call duration
        done = time.perf_counter()
        with self._ev_lock:
            self.events.append((name, enter, done))

    def get_stream(self) -> _FakeStream:
        return self._stream

    def play(self, samples: Any, sr: int) -> None:
        # Like the real sd.play(): stop (and close) any current stream first.
        self.stop()
        self._stream = _FakeStream(play_s=self.play_s, on_close=lambda: self._span("close"))
        with self._ev_lock:
            self.play_calls.append((samples, sr))
        self._span("play")

    def stop(self) -> None:
        self._stream._active = False
        self._span("stop")
        self._stream.close()


def _assert_no_overlap(events: list[tuple[str, float, float]]) -> None:
    spans = sorted((enter, done) for name, enter, done in events if name in ("play", "stop", "close"))
    assert spans, "expected some recorded play/stop/close calls"
    for (_, done1), (enter2, _) in itertools.pairwise(spans):
        assert enter2 >= done1, "two sounddevice output calls overlapped in time"


def test_concurrent_output_calls_never_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression test: earcon play, barge-in stop and TTS play hammering at once
    must never have two native output-stream calls in flight. Fails on unfixed code,
    where each call site hits ``sd`` directly with no lock."""
    fake = _FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.liveagent.speech import _stop_sounddevice
    from away.voice.earcon import _sd_play
    from away.voice.natural import sd_play

    samples = np.zeros(16, dtype=np.float32)
    n_threads, iters = 8, 12
    barrier = threading.Barrier(n_threads)

    def worker(i: int) -> None:
        barrier.wait()  # release all threads at the same instant for maximum collision
        for _ in range(iters):
            op = i % 3
            if op == 0:
                _sd_play(samples, 24000)
            elif op == 1:
                _stop_sounddevice()
            else:
                sd_play(samples, 24000)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "worker thread hung"
    _assert_no_overlap(fake.events)


def test_play_blocking_does_not_hold_lock_during_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Barge-in latency guard: ``play_blocking`` polls for completion OUTSIDE the
    lock, so an interrupt ``stop()`` lands at once instead of queueing behind the
    whole utterance. The teardown ``close()`` still happens under the lock, so it
    cannot race ``stop()``'s own close (``Stream.close()`` is not idempotent)."""
    fake = _FakeSD(play_s=30.0)  # stays active until stop(); no auto-finish flakiness
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(16, dtype=np.float32)
    player = threading.Thread(target=audio_out.play_blocking, args=(samples, 24000))
    player.start()
    deadline = time.perf_counter() + 15
    while not fake.play_calls and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert fake.play_calls, "playback never started"
    time.sleep(0.05)  # let the player enter its poll loop
    stop_done = threading.Event()

    def do_stop() -> None:
        audio_out.stop()
        stop_done.set()

    stopper = threading.Thread(target=do_stop)
    stopper.start()
    assert stop_done.wait(timeout=10), "stop() blocked behind play_blocking's wait"
    player.join(timeout=15)
    stopper.join(timeout=15)
    assert not player.is_alive() and not stopper.is_alive()
    assert fake.get_stream().close_calls >= 1, "teardown close never ran"


def test_stop_without_sounddevice_loaded_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_stop_sounddevice`` keeps its lazy contract: with no sounddevice imported it
    does nothing and does not import it."""
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    from away.liveagent.speech import _stop_sounddevice

    _stop_sounddevice()  # must not raise
    assert "sounddevice" not in sys.modules, "stop() must not import sounddevice by itself"


def test_audio_out_play_invokes_sd_play(monkeypatch: pytest.MonkeyPatch) -> None:
    """``audio_out.play`` must actually call ``sd.play`` (mutant: remove-call survived)."""
    fake = _FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(8, dtype=np.float32)
    audio_out.play(samples, 24000)
    assert len(fake.play_calls) == 1
    got, sr = fake.play_calls[0]
    assert sr == 24000
    assert got is samples


def test_audio_out_stop_invokes_sd_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """``audio_out.stop`` must actually call ``sd.stop`` (mutant: remove-call survived)."""
    fake = _FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    audio_out.stop()
    assert any(name == "stop" for name, _, _ in fake.events)


def test_play_blocking_calls_play_not_just_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """``play_blocking`` must start playback via ``play`` (mutant: ``play(...)`` ->
    ``pass`` survived because the old test only observed the wait)."""
    fake = _FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(8, dtype=np.float32)
    audio_out.play_blocking(samples, 24000)
    assert len(fake.play_calls) == 1, "play_blocking never started playback"


def test_earcon_sd_play_reaches_output_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The earcon path must reach the output stream (mutant: remove-call survived)."""
    fake = _FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice.earcon import _sd_play

    samples = np.zeros(8, dtype=np.float32)
    _sd_play(samples, 24000)
    assert len(fake.play_calls) == 1, "_sd_play never reached sd.play"


def test_stop_sounddevice_stops_stream_when_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_stop_sounddevice`` with sounddevice loaded must stop the stream (mutant:
    ``audio_out.stop()`` -> ``pass`` survived)."""
    fake = _FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.liveagent.speech import _stop_sounddevice

    _stop_sounddevice()
    assert any(name == "stop" for name, _, _ in fake.events)


def test_sd_play_default_volume_leaves_samples_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sd_play`` default ``volume=1.0`` must not rescale samples (mutant:
    ``1.0`` -> ``2.0`` survived)."""
    fake = _FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice.natural import sd_play

    samples = np.full(8, 0.5, dtype=np.float32)
    sd_play(samples, 24000)
    assert fake.play_calls, "sd_play never reached sd.play"
    got, sr = fake.play_calls[0]
    assert sr == 24000
    np.testing.assert_allclose(got, samples, err_msg="default volume must be 1.0")
    # explicit volume still scales
    sd_play(samples, 24000, volume=0.5)
    got2, _ = fake.play_calls[1]
    np.testing.assert_allclose(got2, samples * 0.5, err_msg="volume scaling broken")


def test_play_blocking_waits_for_stream_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """``play_blocking`` must not return while the stream is still active (mutant:
    the wait loop removed would return at once, cutting the audio off)."""
    fake = _FakeSD(play_s=30.0)  # stays active until stopped
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(8, dtype=np.float32)
    done = threading.Event()

    def run() -> None:
        audio_out.play_blocking(samples, 24000)
        done.set()

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.2)
    try:
        assert not done.is_set(), "play_blocking returned while the stream was active"
    finally:
        audio_out.stop()  # like barge-in: ends the wait
    assert done.wait(timeout=10), "play_blocking did not return after stop"
    t.join(timeout=10)
    assert not t.is_alive()


def test_play_blocking_teardown_closes_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """``play_blocking`` must close the stream when playback finishes (mutant: the
    teardown ``close()`` -> ``pass`` survived because every other test also called
    ``stop()``, which closes too)."""
    fake = _FakeSD(play_s=0.05)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(8, dtype=np.float32)
    audio_out.play_blocking(samples, 24000)  # returns when the stream auto-completes
    assert fake.get_stream().close_calls == 1, "play_blocking never closed the stream"


def test_barge_in_stop_does_not_double_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """``play_blocking`` + barge-in ``stop()`` must close the stream exactly once.

    ``sd.stop()`` closes the stream itself; the teardown ``close()`` must skip a
    stream ``stop()`` already closed instead of re-entering Pa_CloseStream on a
    dead handle. Fails on the unfixed code with close_calls == 2.
    """
    fake = _FakeSD(play_s=30.0)  # stays active until stop()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(16, dtype=np.float32)
    player = threading.Thread(target=audio_out.play_blocking, args=(samples, 24000))
    player.start()
    deadline = time.perf_counter() + 15
    while not fake.play_calls and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert fake.play_calls, "playback never started"
    time.sleep(0.05)
    stream = fake.get_stream()
    audio_out.stop()  # barge-in
    player.join(timeout=15)
    assert not player.is_alive()
    assert stream.close_calls == 1, f"stream closed {stream.close_calls} times (double close)"


def test_play_interrupting_play_blocking_does_not_double_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``play()`` while ``play_blocking`` polls must close the old stream once.

    ``sd.play()`` stops (and closes) any current stream first; the polling
    ``play_blocking`` teardown must recognize the already-closed stream.
    """
    fake = _FakeSD(play_s=30.0)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(16, dtype=np.float32)
    player = threading.Thread(target=audio_out.play_blocking, args=(samples, 24000))
    player.start()
    deadline = time.perf_counter() + 15
    while not fake.play_calls and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert fake.play_calls, "playback never started"
    time.sleep(0.05)
    old_stream = fake.get_stream()
    audio_out.play(samples, 24000)
    player.join(timeout=15)
    assert not player.is_alive()
    assert old_stream.close_calls == 1, f"old stream closed {old_stream.close_calls} times"


def test_stop_after_play_does_not_overwrite_closed_stream_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a stop() after play() must not overwrite the closed-stream record.

    Sequence: play_blocking() starts (stream1); play() stops stream1 (recorded);
    stop() stops stream2 (must NOT overwrite stream1's record). The play_blocking
    teardown must skip closing stream1 exactly once, not double-close it.
    Fails on the old single-slot _closed_by_stop implementation.
    """
    fake = _FakeSD(play_s=30.0)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(16, dtype=np.float32)
    player = threading.Thread(target=audio_out.play_blocking, args=(samples, 24000))
    player.start()
    deadline = time.perf_counter() + 15
    while not fake.play_calls and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert fake.play_calls, "playback never started"
    time.sleep(0.05)
    stream1 = fake.get_stream()
    # play() stops stream1 (recorded in _closed_by_stop)
    audio_out.play(samples, 24000)
    assert stream1.close_calls == 1, "sd.play() should have closed stream1 once"
    stream2 = fake.get_stream()
    assert stream2 is not stream1
    # stop() stops stream2; must not overwrite stream1's record
    audio_out.stop()
    assert stream2.close_calls == 1, "sd.stop() should have closed stream2 once"
    # play_blocking teardown: stream1 was already closed, skip it (no double close)
    player.join(timeout=15)
    assert not player.is_alive()
    assert stream1.close_calls == 1, (
        f"stream1 closed {stream1.close_calls} times (double close from overwritten record)"
    )


def test_two_overlapping_play_blocking_no_double_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two play_blocking() calls with interleaved stop()s: each stream closed once."""
    fake = _FakeSD(play_s=30.0)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out

    samples = np.zeros(16, dtype=np.float32)
    started = threading.Event()

    def player() -> None:
        started.wait(timeout=15)
        audio_out.play_blocking(samples, 24000)

    t1 = threading.Thread(target=player)
    t2 = threading.Thread(target=player)
    t1.start()
    t2.start()
    time.sleep(0.1)
    started.set()  # release both players at once
    time.sleep(0.3)  # let both enter play_blocking
    # Interrupt with stop(); each stream must be closed exactly once total
    audio_out.stop()
    t1.join(timeout=15)
    t2.join(timeout=15)
    assert not t1.is_alive() and not t2.is_alive()
    # No leaked entries: all waiters tore down
    assert not audio_out._play_blocking_streams, "waiter set leaked entries"
    assert not audio_out._closed_by_stop, "closed-stream set leaked entries"
