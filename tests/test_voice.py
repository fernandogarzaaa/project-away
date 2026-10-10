import numpy as np
import pytest

from away.events import EventBus
from away.voice.asr import FakeASR, make_asr
from away.voice.cleanup import RuleCleaner
from away.voice.context import whisper_prompt
from away.voice.dictation import DictationSession
from away.voice.output import ListSink
from away.voice.vad import EnergyVAD, frame_rms, make_vad

SR = 16000


def tone(seconds, amp=0.3, f=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds, amp=0.001, seed=0):
    return (np.random.default_rng(seed).normal(0, amp, int(seconds * SR))).astype(np.float32)


def utterance():
    return np.concatenate([silence(0.5), tone(0.6), silence(0.5, seed=1)])


def test_frame_rms():
    x = np.array([3, -3, 4, -4, 1], dtype=np.float32)
    assert np.allclose(frame_rms(x, 2), [3.0, 4.0])
    assert len(frame_rms(x, 10)) == 0
    with pytest.raises(ValueError):
        frame_rms(x, 0)


def test_energy_vad_segments_and_trim():
    vad = EnergyVAD(SR)
    audio = utterance()
    segs = vad.segments(audio)
    assert len(segs) == 1
    s = segs[0]
    assert abs(s.start - int(0.5 * SR)) <= vad.frame
    # hangover extends the end by up to 8 frames
    assert int(1.1 * SR) - vad.frame <= s.end <= int(1.1 * SR) + 9 * vad.frame
    trimmed = vad.trim(audio, pad_ms=0)
    assert len(trimmed) == s.end - s.start
    padded = vad.trim(audio, pad_ms=100)
    assert len(padded) == len(trimmed) + 2 * int(0.1 * SR)
    assert vad.is_speech(audio)


def test_energy_vad_silence():
    vad = EnergyVAD(SR)
    a = silence(1.0)
    assert vad.segments(a) == []
    assert len(vad.trim(a)) == 0
    assert not vad.is_speech(a)
    assert len(vad.frame_flags(np.zeros(10, dtype=np.float32))) == 0


def test_energy_vad_min_speech_and_hangover():
    vad = EnergyVAD(SR, frame_ms=10, min_speech_frames=5, hangover_frames=0)
    blip = np.concatenate([silence(0.3), tone(0.03), silence(0.3)])  # 3 frames < 5
    assert vad.segments(blip) == []
    vad2 = EnergyVAD(SR, frame_ms=10, min_speech_frames=1, hangover_frames=4)
    flags = vad2.frame_flags(blip)
    n_speech = int(flags.sum())
    assert 3 + 4 - 1 <= n_speech <= 3 + 4 + 1


def test_energy_vad_min_rms_floor():
    vad = EnergyVAD(SR, min_rms=0.5)
    assert vad.segments(utterance()) == []  # 0.3 amplitude tone stays under the absolute floor


def test_make_vad_fallback(monkeypatch):
    import sys

    assert isinstance(make_vad("energy"), EnergyVAD)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)
    monkeypatch.setitem(sys.modules, "silero_vad", None)
    assert isinstance(make_vad("silero"), EnergyVAD)  # falls back when no Silero backend is installed


def test_whisper_prompt():
    assert whisper_prompt(None, []) is None
    assert whisper_prompt("Slack", ["Inan", "GitHub"]) == "Vocabulary: Inan, GitHub. Dictating into Slack."
    assert whisper_prompt(None, ["x"] * 50).count("x") == 40


def test_fake_asr():
    a = FakeASR(["one"])
    assert a.transcribe(np.zeros(5, dtype=np.float32), "p") == "one"
    assert a.transcribe(np.zeros(3, dtype=np.float32)) == ""
    assert a.calls == [(5, "p"), (3, None)]


def test_make_asr_imports_backend_lazily(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "nemo", None)
    monkeypatch.setitem(sys.modules, "nemo.collections", None)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", None)
    with pytest.raises(ImportError):
        make_asr("faster-whisper", "tiny")
    with pytest.raises(ImportError):
        make_asr("parakeet", "large-v3-turbo")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def make_session(outputs, gate=None, context=None, vocabulary=None):
    bus = EventBus()
    asr = FakeASR(outputs)
    sink = ListSink()
    s = DictationSession(
        asr,
        RuleCleaner(),
        sink,
        EnergyVAD(SR),
        bus,
        SR,
        gate=gate,
        context=context,
        vocabulary=vocabulary,
        clock=Clock(),
    )
    return s, asr, sink, bus


def feed_chunks(s, audio, n=480):
    for i in range(0, len(audio), n):
        s.feed(audio[i : i + n])


def test_dictation_happy_path():
    s, asr, sink, bus = make_session(["um hello there friend"], context=lambda: "Notes", vocabulary=["Inan"])
    assert s.start() is True
    assert s.start() is False  # already recording
    feed_chunks(s, utterance())
    r = s.stop()
    assert r.status == "ok"
    assert r.raw == "um hello there friend"
    assert r.text == "Hello there friend."
    assert sink.items == ["Hello there friend. "]
    assert r.meta["app"] == "Notes"
    assert r.seconds == pytest.approx(len(utterance()) / SR)
    n_samples, prompt = asr.calls[0]
    assert n_samples < len(utterance())  # VAD trimmed the silence
    assert prompt == "Vocabulary: Inan. Dictating into Notes."
    topics = [e.topic for e in bus.history()]
    assert topics == ["dictation.start", "dictation.result"]
    assert bus.history("dictation.result")[0].data == {"status": "ok", "text": "Hello there friend."}


def test_dictation_feed_ignored_when_not_recording():
    s, asr, sink, _ = make_session(["x y z"])
    feed_chunks(s, utterance())
    s.start()
    r = s.stop()
    assert r.status == "silence"
    assert asr.calls == [] and sink.items == []


def test_dictation_silence_and_empty():
    s, asr, sink, _ = make_session(["um"])
    s.start()
    feed_chunks(s, silence(1.0))
    assert s.stop().status == "silence"
    s.start()
    feed_chunks(s, utterance())
    r = s.stop()
    assert r.status == "empty" and r.raw == "um"
    assert sink.items == []


def test_dictation_gate_blocks_and_gets_time_window():
    calls = []

    def gate(t0, t1):
        calls.append((t0, t1))
        return False

    s, asr, sink, _ = make_session(["secret"], gate=gate)
    s.start()
    feed_chunks(s, utterance())
    r = s.stop()
    assert r.status == "gated"
    assert calls == [(1.0, 2.0)]
    assert asr.calls == [] and sink.items == []


def test_dictation_gate_allows():
    s, _, sink, _ = make_session(["fine then ok"], gate=lambda a, b: True)
    s.start()
    feed_chunks(s, utterance())
    assert s.stop().status == "ok"
    assert sink.items == ["Fine then ok. "]


def test_dictation_disabled_and_toggle():
    s, _, sink, bus = make_session(["one two three"])
    s.set_enabled(False)
    assert s.start() is False
    assert s.stop().status == "disabled"
    assert bus.history("dictation.enabled")[0].data == {"enabled": False}
    s.set_enabled(True)
    assert s.toggle() is None and s.recording
    feed_chunks(s, utterance())
    r = s.toggle()
    assert r is not None and r.status == "ok"
    assert s.stop().status == "empty"  # not recording anymore


def test_disable_while_recording_discards_audio():
    s, asr, _, _ = make_session(["x"])
    s.start()
    feed_chunks(s, utterance())
    s.set_enabled(False)
    assert not s.recording
    s.set_enabled(True)
    s.start()
    assert s.stop().status == "silence"
    assert asr.calls == []


def test_dictation_without_bus_or_context():
    s = DictationSession(FakeASR(["a b c"]), RuleCleaner(), ListSink(), EnergyVAD(SR))
    s.start()
    s.feed(utterance())
    r = s.stop()
    assert r.status == "ok" and r.meta["app"] is None
