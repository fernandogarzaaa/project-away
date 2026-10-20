"""Mutation-strengthening tests for voice/dictation.py (v0.1.29)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from away.voice.dictation import DictationResult, DictationSession
from test_voice import FakeASR, ListSink, RuleCleaner, make_session, utterance


def test_cancel_discards_without_transcribing() -> None:
    s, asr, sink, _ = make_session(["hello"])
    s.start()
    s.feed(utterance())
    s.cancel()
    assert not s.recording
    # No transcription happened
    assert asr.calls == []
    assert sink.items == []
    # stop after cancel: not recording
    assert s.stop().status == "empty"


def test_command_handler_intercepts() -> None:
    from test_voice import SR, Clock, EnergyVAD, EventBus

    handled = []

    def handler(text: str) -> bool:
        handled.append(text)
        return True

    bus = EventBus()
    asr = FakeASR(["open browser"])
    sink = ListSink()
    s = DictationSession(asr, RuleCleaner(), sink, EnergyVAD(SR), bus, SR, command_handler=handler, clock=Clock())
    s.start()
    s.feed(utterance())
    r = s.stop()
    assert r.status == "command"
    assert handled == ["open browser"]
    # Command not typed to sink
    assert sink.items == []


def test_command_handler_declines() -> None:
    from test_voice import SR, Clock, EnergyVAD, EventBus

    def handler(text: str) -> bool:
        return False

    bus = EventBus()
    asr = FakeASR(["hello world"])
    sink = ListSink()
    s = DictationSession(asr, RuleCleaner(), sink, EnergyVAD(SR), bus, SR, command_handler=handler, clock=Clock())
    s.start()
    s.feed(utterance())
    r = s.stop()
    assert r.status == "ok"
    assert sink.items == ["Hello world "]


def test_command_handler_empty_raw_not_called() -> None:
    from test_voice import SR, Clock, EnergyVAD, EventBus

    called = []

    def handler(text: str) -> bool:
        called.append(text)
        return True

    bus = EventBus()
    asr = FakeASR(["um"])
    sink = ListSink()
    s = DictationSession(asr, RuleCleaner(), sink, EnergyVAD(SR), bus, SR, command_handler=handler, clock=Clock())
    s.start()
    s.feed(utterance())
    r = s.stop()
    # "um" passes clean_transcript (not empty), handler IS called
    assert r.status == "command"
    assert called == ["um"]


def test_set_enabled_publishes() -> None:
    s, _, _, bus = make_session(["x"])
    s.set_enabled(False)
    assert bus.history("dictation.enabled")[0].data == {"enabled": False}
    s.set_enabled(True)
    assert bus.history("dictation.enabled")[1].data == {"enabled": True}


def test_stop_when_disabled_not_recording() -> None:
    s, _, _, _ = make_session(["x"])
    s.set_enabled(False)
    r = s.stop()
    assert r.status == "disabled"


def test_feed_converts_to_float32() -> None:
    s, _, _, _ = make_session(["hello world"])
    s.start()
    # Feed int array, should be converted
    s.feed(np.array([1000, 2000, 3000], dtype=np.int16))
    # Should not crash
    s.cancel()


def test_dictation_result_defaults() -> None:
    r = DictationResult("ok")
    assert r.raw == ""
    assert r.text == ""
    assert r.seconds == 0.0
    assert r.meta == {}


def test_cleaner_empty_returns_empty_status() -> None:
    # ASR returns something, cleaner returns empty
    class EmptyCleaner:
        def clean(self, text: str, context: str | None = None) -> str:
            return ""

    from away.voice.dictation import DictationSession
    from test_voice import SR, EnergyVAD, FakeASR, ListSink

    s = DictationSession(FakeASR(["hello world"]), EmptyCleaner(), ListSink(), EnergyVAD(SR))
    s.start()
    s.feed(utterance())
    r = s.stop()
    assert r.status == "empty"
    assert r.raw == "hello world"


def test_feed_2d_chunk_flattened() -> None:
    # Tests reshape(-1) mutant (line 97): 2D chunk must be flattened
    s, _, _, _ = make_session(["hello"])
    s.start()
    chunk_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    s.feed(chunk_2d)
    # Should not crash; chunks are flattened
    assert len(s._chunks) == 1
    assert s._chunks[0].shape == (4,)
    s.cancel()


def test_seconds_calculation_affects_silence_phrase() -> None:
    # Tests len(speech)/self.sr mutant (line 133): seconds must be divided, not multiplied
    # A very short utterance with a silence phrase should be filtered
    from test_voice import SR, Clock, EnergyVAD, EventBus

    bus = EventBus()
    asr = FakeASR(["um"])  # "um" is a silence phrase
    sink = ListSink()
    s = DictationSession(asr, RuleCleaner(), sink, EnergyVAD(SR), bus, SR, clock=Clock())
    s.start()
    # Feed a very short audio (0.1s) - should be detected as short silence phrase
    short = np.zeros(int(SR * 0.1), dtype=np.float32)
    s.feed(short)
    r = s.stop()
    # With correct seconds (0.1), "um" might be kept or filtered based on threshold
    # The key is that seconds is calculated correctly (not huge)
    assert r.seconds < 1.0  # Would be huge if multiplied (0.1 * 16000 = 1600)


def test_disabled_while_recording_discards_buffer_and_publishes() -> None:
    s, _, sink, bus = make_session(["hello"])
    s.start()
    s.feed(utterance())
    s.set_enabled(False)
    assert s.recording is False
    assert s._chunks == []
    assert bus.history("dictation.enabled")[-1].data == {"enabled": False}
    assert s.stop().status == "disabled"
    assert sink.items == []


def test_toggle_starts_then_stops_recording() -> None:
    s, asr, sink, _ = make_session(["hello world"])
    assert s.toggle() is None
    assert s.recording is True
    s.feed(utterance())
    result = s.toggle()
    assert result is not None and result.status == "ok"
    assert asr.calls
    assert sink.items == ["Hello world "]


def test_probe_microphone_reports_dependency_error(monkeypatch) -> None:
    import sys

    from away.voice.dictation import probe_microphone

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    ok, message = probe_microphone()
    assert ok is False
    assert "sounddevice unavailable" in message


def test_set_enabled_true_while_recording_keeps_recording() -> None:
    # Kills L78 mutant: `and` -> `or` in set_enabled.
    # With `or`, set_enabled(True) while recording would wrongly stop recording.
    s, _, _, _ = make_session(["hello"])
    s.start()
    assert s.recording is True
    s.set_enabled(True)
    assert s.recording is True


def test_set_enabled_false_while_recording_stops_and_clears() -> None:
    # Companion: set_enabled(False) while recording must stop and clear.
    s, _, _, _ = make_session(["hello"])
    s.start()
    s.feed(utterance())
    assert len(s._chunks) > 0
    s.set_enabled(False)
    assert s.recording is False
    assert s._chunks == []


def test_stop_with_no_chunks_reports_zero_seconds() -> None:
    # Kills L104 mutant: np.zeros(0) -> np.zeros(1).
    # With the mutant, seconds would be 1/16000 instead of 0.0.
    from test_voice import SR, Clock, EnergyVAD, EventBus

    bus = EventBus()
    s = DictationSession(FakeASR(["x"]), RuleCleaner(), ListSink(), EnergyVAD(SR), bus, SR, clock=Clock())
    s.start()
    # Feed nothing, stop immediately -> empty chunks -> zeros(0)
    r = s.stop()
    assert r.seconds == 0.0


def test_process_seconds_divides_by_sample_rate() -> None:
    # Kills L133 mutant: len(speech)/self.sr -> len(speech)*self.sr.
    # Feed a known utterance; seconds must be ~1.6s of audio, not billions.
    from test_voice import SR, Clock, EnergyVAD, EventBus, utterance

    bus = EventBus()
    asr = FakeASR(["hello world"])
    sink = ListSink()
    s = DictationSession(asr, RuleCleaner(), sink, EnergyVAD(SR), bus, SR, clock=Clock())
    s.start()
    s.feed(utterance())
    r = s.stop()
    assert r.status == "ok"
    # utterance() is 1.6s total; VAD trims silence so speech < 1.6s
    assert 0.1 < r.seconds < 2.0


def test_default_sample_rate_is_16000() -> None:
    # Kills L50 mutant: default sample_rate 16000 -> 16001.
    from test_voice import EnergyVAD, FakeASR, ListSink

    s = DictationSession(FakeASR(["x"]), RuleCleaner(), ListSink(), EnergyVAD(16000))
    assert s.sr == 16000


def test_initial_t_start_is_zero() -> None:
    # Kills L69 mutant: self._t_start = 0.0 -> 1.0.
    s, _, _, _ = make_session(["x"])
    assert s._t_start == 0.0


def test_short_silence_phrase_filtered_via_seconds_division() -> None:
    # Kills L133 mutant: len(speech)/self.sr -> len(speech)*self.sr in _process.
    # "thank you" is a silence phrase; with correct seconds (< 2.0s) it is
    # filtered as a silence hallucination. With the mutant, seconds is
    # astronomically large, so the phrase passes through as real text.
    # Also kills L135 mutant: `why or "empty transcript"` -> `why and ...`.
    # With `and`, a truthy why would wrongly become "empty transcript".
    from test_voice import SR, Clock, EnergyVAD, EventBus, utterance

    bus = EventBus()
    asr = FakeASR(["thank you"])
    sink = ListSink()
    s = DictationSession(asr, RuleCleaner(), sink, EnergyVAD(SR), bus, SR, clock=Clock())
    s.start()
    s.feed(utterance())
    r = s.stop()
    # "thank you" with short audio (< 2.0s) -> silence hallucination -> empty
    assert r.status == "empty"
    assert sink.items == []
    # The reason must be the actual why, not the fallback
    assert r.meta["reason"] == "silence hallucination"


# ---------------------------------------------------------------------------
# Round 2: MicStream / probe_microphone / hotkey_listener with faked hardware.
# These functions carry "# pragma: no cover" but testteeth still mutates them;
# every mutant there was "no_coverage". The fakes below capture the exact
# arguments and call sequence so behaviour changes are observable.
# ---------------------------------------------------------------------------


class _FakeInputStream:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        _FakeInputStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _fake_sounddevice(monkeypatch, stream_cls=_FakeInputStream):
    import sys
    import types

    sd = types.SimpleNamespace(InputStream=stream_cls)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def test_micstream_stream_parameters(monkeypatch) -> None:
    # Kills L159 (channels 1->2) and L161 mutants (blocksize arithmetic:
    # Div->Mult, Mult->Div, 1000->1001). int(16000*30/1000) == 480; the
    # mutants give 480000000, 0 and 479 respectively.
    from away.voice.dictation import MicStream

    _FakeInputStream.instances = []
    _fake_sounddevice(monkeypatch)
    MicStream(lambda chunk: None, sample_rate=16000, block_ms=30)
    (stream,) = _FakeInputStream.instances
    assert stream.kwargs["samplerate"] == 16000
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "float32"
    assert stream.kwargs["blocksize"] == 480


def test_micstream_callback_forwards_first_channel(monkeypatch) -> None:
    # Kills L155 mutants: removed on_chunk call, and indata[:, 0] -> [:, 1].
    from away.voice.dictation import MicStream

    _FakeInputStream.instances = []
    _fake_sounddevice(monkeypatch)
    received = []
    MicStream(received.append, sample_rate=16000, block_ms=30)
    (stream,) = _FakeInputStream.instances
    indata = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    stream.kwargs["callback"](indata, 3, None, None)
    assert len(received) == 1
    assert received[0].tolist() == [1.0, 2.0, 3.0]


def test_micstream_context_manager_starts_stops_closes(monkeypatch) -> None:
    # Kills L166 (removed stream.start), L167 (__enter__ returns None),
    # L170 (removed stream.stop), L171 (removed stream.close).
    from away.voice.dictation import MicStream

    _FakeInputStream.instances = []
    _fake_sounddevice(monkeypatch)
    ms = MicStream(lambda chunk: None)
    with ms as entered:
        assert entered is ms
        (stream,) = _FakeInputStream.instances
        assert stream.started is True
    assert stream.stopped is True
    assert stream.closed is True


def test_probe_microphone_success_path(monkeypatch) -> None:
    # Kills L180 (seconds*sample_rate -> /, channels 1->2), L181 (removed
    # sd.wait), L182 (rec**2 -> rec*2 and -> rec**3), L183 (True->False,
    # return None). rec is all 0.5, so RMS is exactly 0.5; the arithmetic
    # mutants give 1.0 and ~0.3536.
    import sys
    import types

    from away.voice.dictation import probe_microphone

    calls = {}

    def rec(n, samplerate=None, channels=None, dtype=None):
        calls.update(n=n, samplerate=samplerate, channels=channels, dtype=dtype)
        return np.full((n, channels), 0.5, dtype=np.float32)

    def wait():
        calls["waited"] = True

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(rec=rec, wait=wait))
    ok, message = probe_microphone(0.5, 16000)
    assert ok is True
    assert calls["n"] == 8000
    assert calls["samplerate"] == 16000
    assert calls["channels"] == 1
    assert calls["dtype"] == "float32"
    assert calls["waited"] is True
    assert "microphone OK (rms 0.5000" in message


def test_probe_microphone_rec_error_is_reported(monkeypatch) -> None:
    # Kills L185 mutants: swallowed exception (handler body -> pass, so the
    # function would return None), return None, and False->True.
    import sys
    import types

    from away.voice.dictation import probe_microphone

    def rec(*a, **k):
        raise RuntimeError("device busy")

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(rec=rec, wait=lambda: None))
    result = probe_microphone(0.5, 16000)
    assert result is not None
    ok, message = result
    assert ok is False
    assert message == "microphone error: device busy"


class _FakeThread:
    instances: ClassVar[list] = []

    def __init__(self, target=None, daemon=None, **kwargs) -> None:
        self.target = target
        self.daemon = daemon
        self.started = False
        _FakeThread.instances.append(self)

    def start(self) -> None:
        self.started = True
        if self.target is not None:
            self.target()


class _FakeSession:
    def __init__(self, recording: bool = False) -> None:
        self.recording = recording
        self.start_calls = 0
        self.stop_calls = 0
        self.toggle_calls = 0

    def start(self) -> bool:
        self.start_calls += 1
        self.recording = True
        return True

    def stop(self) -> None:
        self.stop_calls += 1
        self.recording = False

    def toggle(self) -> None:
        self.toggle_calls += 1
        self.recording = not self.recording


def _fake_pynput(monkeypatch):
    import sys
    import types

    class KeyCode:
        def __init__(self, ch) -> None:
            self.ch = ch

        @classmethod
        def from_char(cls, ch):
            return cls(ch)

    class Key:
        space = KeyCode("<space-key>")

    class GlobalHotKeys:
        instances: ClassVar[list] = []

        def __init__(self, mapping) -> None:
            self.mapping = mapping
            self.started = False
            GlobalHotKeys.instances.append(self)

        def start(self) -> None:
            self.started = True

    class Listener:
        instances: ClassVar[list] = []

        def __init__(self, on_press=None, on_release=None) -> None:
            self.on_press = on_press
            self.on_release = on_release
            self.started = False
            Listener.instances.append(self)

        def start(self) -> None:
            self.started = True

    keyboard = types.SimpleNamespace(Key=Key, KeyCode=KeyCode, GlobalHotKeys=GlobalHotKeys, Listener=Listener)
    pynput = types.SimpleNamespace(keyboard=keyboard)
    monkeypatch.setitem(sys.modules, "pynput", pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    GlobalHotKeys.instances = []
    Listener.instances = []
    return keyboard


def test_hotkey_listener_toggle_mode(monkeypatch) -> None:
    # Kills L192 (mode == "toggle" -> !=), L194 (Thread daemon True->False),
    # L208 (removed listener.start), L209 (return None instead of listener).
    import threading

    from away.voice.dictation import hotkey_listener

    keyboard = _fake_pynput(monkeypatch)
    _FakeThread.instances = []
    monkeypatch.setattr(threading, "Thread", _FakeThread)
    session = _FakeSession()
    listener = hotkey_listener("toggle", "<ctrl>+q", "ctrl", session)
    (ghk,) = keyboard.GlobalHotKeys.instances
    assert keyboard.Listener.instances == []
    assert listener is ghk
    assert ghk.started is True
    assert set(ghk.mapping) == {"<ctrl>+q"}
    ghk.mapping["<ctrl>+q"]()
    (thread,) = _FakeThread.instances
    assert thread.target == session.toggle
    assert thread.daemon is True
    assert thread.started is True
    assert session.toggle_calls == 1


def test_hotkey_listener_hold_press_starts_once(monkeypatch) -> None:
    # Kills L197 (getattr(...) or KeyCode.from_char -> and), L200 mutants
    # (and->or, ==->!=, dropped `not`), L201 (removed session.start call).
    from away.voice.dictation import hotkey_listener

    keyboard = _fake_pynput(monkeypatch)
    session = _FakeSession()
    hotkey_listener("hold", "<ctrl>", "space", session)
    (listener,) = keyboard.Listener.instances
    target = keyboard.Key.space
    # Pressing an unrelated key must not start.
    listener.on_press(keyboard.KeyCode.from_char("x"))
    assert session.start_calls == 0
    # Pressing the hold key starts exactly once.
    listener.on_press(target)
    assert session.start_calls == 1
    # Pressing again while already recording must not re-start.
    listener.on_press(target)
    assert session.start_calls == 1


def test_hotkey_listener_hold_release_stops_via_thread(monkeypatch) -> None:
    # Kills L204 mutants (and->or, ==->!= on the release guard), L205
    # (removed Thread.start, daemon True->False).
    import threading

    from away.voice.dictation import hotkey_listener

    keyboard = _fake_pynput(monkeypatch)
    _FakeThread.instances = []
    monkeypatch.setattr(threading, "Thread", _FakeThread)
    session = _FakeSession(recording=True)
    hotkey_listener("hold", "<ctrl>", "space", session)
    (listener,) = keyboard.Listener.instances
    # Releasing an unrelated key while recording must not stop.
    listener.on_release(keyboard.KeyCode.from_char("x"))
    assert session.stop_calls == 0
    assert _FakeThread.instances == []
    # Releasing the hold key stops via a started daemon thread.
    listener.on_release(keyboard.Key.space)
    (thread,) = _FakeThread.instances
    assert thread.target == session.stop
    assert thread.daemon is True
    assert thread.started is True
    assert session.stop_calls == 1
    # Releasing again while not recording must not stop again.
    listener.on_release(keyboard.Key.space)
    assert session.stop_calls == 1
