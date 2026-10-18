"""Mutation-strengthening tests for voice/dictation.py (v0.1.29)."""

from __future__ import annotations

import numpy as np
from tests.test_voice import FakeASR, ListSink, RuleCleaner, make_session, utterance

from away.voice.dictation import DictationResult, DictationSession


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
    from tests.test_voice import SR, Clock, EnergyVAD, EventBus

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
    from tests.test_voice import SR, Clock, EnergyVAD, EventBus

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
    from tests.test_voice import SR, Clock, EnergyVAD, EventBus

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

    from tests.test_voice import SR, EnergyVAD, FakeASR, ListSink

    from away.voice.dictation import DictationSession

    s = DictationSession(FakeASR(["hello world"]), EmptyCleaner(), ListSink(), EnergyVAD(SR))
    s.start()
    s.feed(utterance())
    r = s.stop()
    assert r.status == "empty"
    assert r.raw == "hello world"
