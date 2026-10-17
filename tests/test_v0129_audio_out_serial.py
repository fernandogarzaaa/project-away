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


class _FakeSD:
    """Fake ``sounddevice`` module recording (name, enter, exit) for every call.

    ``play``/``stop`` sleep briefly to widen the race window, the way the real
    native calls do. ``wait`` either returns at once or blocks on ``wait_release``,
    so tests can observe a thread parked inside playback.
    """

    def __init__(self, block_wait: bool = False, op_s: float = 0.01) -> None:
        self.events: list[tuple[str, float, float]] = []
        self._ev_lock = threading.Lock()
        self.op_s = op_s
        self.block_wait = block_wait
        self.wait_started = threading.Event()
        self.wait_release = threading.Event()
        self.play_calls: list[tuple[Any, int]] = []

    def _span(self, name: str) -> None:
        enter = time.perf_counter()
        time.sleep(self.op_s)  # stand-in for the native call duration
        done = time.perf_counter()
        with self._ev_lock:
            self.events.append((name, enter, done))

    def play(self, samples: Any, sr: int) -> None:
        with self._ev_lock:
            self.play_calls.append((samples, sr))
        self._span("play")

    def stop(self) -> None:
        self._span("stop")

    def wait(self) -> None:
        if not self.block_wait:
            return
        self.wait_started.set()
        assert self.wait_release.wait(timeout=15), "fake wait() was never released"


def _assert_no_overlap(events: list[tuple[str, float, float]]) -> None:
    spans = sorted((enter, done) for name, enter, done in events if name in ("play", "stop"))
    assert spans, "expected some recorded play/stop calls"
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
    """Barge-in latency guard: ``play_blocking`` must wait for the stream OUTSIDE the
    lock, so an interrupt ``stop()`` lands at once instead of queueing behind the whole
    utterance. Fails if the wait is moved inside the lock."""
    fake = _FakeSD(block_wait=True)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from away.voice import audio_out
    from away.voice.natural import sd_play

    samples = np.zeros(16, dtype=np.float32)
    player = threading.Thread(target=sd_play, args=(samples, 24000))
    player.start()
    assert fake.wait_started.wait(timeout=15), "playback never reached wait()"
    stop_done = threading.Event()

    def do_stop() -> None:
        audio_out.stop()
        stop_done.set()

    stopper = threading.Thread(target=do_stop)
    stopper.start()
    try:
        assert stop_done.wait(timeout=10), "stop() blocked behind play_blocking's wait()"
    finally:
        fake.wait_release.set()
    player.join(timeout=15)
    stopper.join(timeout=15)
    assert not player.is_alive() and not stopper.is_alive()


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
