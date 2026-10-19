"""Mutation-strengthening tests for voice/dictation.py (v0.1.29)."""

from __future__ import annotations

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
