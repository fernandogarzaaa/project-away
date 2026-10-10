import os

import numpy as np
import pytest

from away.bindings import Action, BindingsConfig
from away.events import EventBus
from away.face.gate import SpeechGate
from away.voice.always_on import (
    AmbientSession,
    StreamingEndpointer,
    VoiceController,
    decide,
    match_wake,
    sensitivity_params,
)
from away.voice.asr import FakeASR
from away.voice.dictation import DictationSession
from away.voice.output import ListSink
from away.voice.vad import EnergyVAD

SR = 16000


def noise(sec, amp=0.002, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(sec * SR)) * amp).astype(np.float32)


def speech(sec, amp=0.2, f=220.0):
    t = np.arange(int(sec * SR)) / SR
    # amplitude-modulated tone: a crude syllable rhythm
    return (amp * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) * np.sin(2 * np.pi * f * t)).astype(np.float32)


def stream(ep, audio, chunk=480):
    out = []
    for i in range(0, len(audio), chunk):
        out += ep.feed(audio[i : i + chunk])
    return out


COMMANDS = BindingsConfig().voice_phrases()


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# ----------------------------------------------------------------- end-pointing


def test_endpointer_segments_utterances_and_drops_blips():
    ep = StreamingEndpointer(SR, end_silence_ms=500, min_utterance_ms=250)
    audio = np.concatenate(
        [noise(1.0), speech(0.8), noise(0.9, seed=1), speech(0.08), noise(0.9, seed=2), speech(1.2), noise(0.8, seed=3)]
    )
    utts = stream(ep, audio)
    assert len(utts) == 2  # the 80 ms click is not an utterance
    (a, b) = utts
    assert 0.6 <= a.t0 <= 1.0 and 1.75 <= a.t1 <= 2.2  # pre-roll before, ~150 ms tail after
    assert 3.3 <= b.t0 <= 3.7 and 4.9 <= b.t1 <= 5.2
    assert a.audio.dtype == np.float32 and len(a.audio) == pytest.approx((a.t1 - a.t0) * SR, abs=1)
    assert not ep.in_speech


def test_endpointer_waits_for_end_silence_then_flush():
    ep = StreamingEndpointer(SR, end_silence_ms=700)
    assert stream(ep, np.concatenate([noise(0.5), speech(0.6), noise(0.3)])) == []  # 300 ms < 700 ms
    assert ep.in_speech
    u = ep.flush()
    assert u is not None and ep.flush() is None


def test_endpointer_max_length_splits_long_speech():
    ep = StreamingEndpointer(SR, max_utterance_s=2.0)
    utts = stream(ep, np.concatenate([noise(0.5), speech(5.0)]))
    assert len(utts) == 2 and all(u.t1 - u.t0 <= 2.01 for u in utts)


def test_sensitivity_changes_what_counts_as_speech():
    quiet = np.concatenate([noise(1.0, amp=0.001), speech(0.8, amp=0.006), noise(1.0, amp=0.001, seed=4)])
    assert stream(StreamingEndpointer(SR, sensitivity=0.0), quiet) == []
    assert len(stream(StreamingEndpointer(SR, sensitivity=1.0), quiet)) == 1
    lo, hi = sensitivity_params(0.0), sensitivity_params(1.0)
    assert lo[0] > hi[0] and lo[1] > hi[1]
    assert sensitivity_params(5) == hi  # clamped


def test_endpointer_reset_drops_buffers():
    ep = StreamingEndpointer(SR)
    stream(ep, np.concatenate([noise(0.5), speech(0.5)]))
    assert ep.in_speech
    ep.reset()
    assert not ep.in_speech and ep.flush() is None and len(ep._rest) == 0 and ep._utt == []


# --------------------------------------------------------------------- wake word


@pytest.mark.parametrize(
    "text,ok,rest",
    [
        ("Away, scroll down.", True, "scroll down."),
        ("hey away click", True, "click"),
        ("OK Away open browser", True, "open browser"),
        ("A way, type this please", True, "type this please"),
        ("away", True, ""),
        ("I was away all day", False, "I was away all day"),
        ("anyway click", False, "anyway click"),
        ("", False, ""),
    ],
)
def test_match_wake(text, ok, rest):
    assert match_wake(text, "away") == (ok, rest)


def test_match_custom_multiword_wake_word():
    assert match_wake("Hey Jarvis, new tab", "jarvis") == (True, "new tab")
    assert match_wake("computer stop listening", "Computer") == (True, "stop listening")


def test_decide_wake_word_gating():
    kw = {"commands": COMMANDS, "wake_word": "away"}
    assert decide("scroll down", **kw).kind == "ignore"  # no wake word: ignored, even a command
    d = decide("away scroll down", **kw)
    assert d.kind == "command" and d.action == Action("mouse", "scroll_down")
    assert decide("Away.", **kw).kind == "wake"
    assert decide("scroll down", in_wake_window=True, **kw).kind == "command"
    d = decide("Away, meeting at noon tomorrow.", **kw)
    assert d.kind == "type" and d.text == "meeting at noon tomorrow."
    # the face gate is not consulted when a wake word is set
    assert decide("away hello", gate=False, **kw).kind == "type"


def test_decide_face_gate_without_wake_word():
    assert decide("hello there", commands=COMMANDS, gate=True).kind == "type"
    d = decide("hello there", commands=COMMANDS, gate=False)
    assert d.kind == "ignore" and "face gate" in d.reason
    assert decide("click", commands=COMMANDS, gate=False).kind == "ignore"  # TV saying "click"
    # no camera gate at all: commands only by default, typing only if the user opted in
    assert decide("hello there", commands=COMMANDS, gate=None).kind == "ignore"
    assert decide("click", commands=COMMANDS, gate=None).kind == "command"
    assert decide("hello there", commands=COMMANDS, gate=None, fallback="type").kind == "type"


def test_decide_listening_intent_and_commands_switch():
    assert decide("hello", commands=COMMANDS, gate=True, listening=False).kind == "ignore"
    assert decide("click", commands=COMMANDS, gate=True, listening=False).kind == "ignore"
    d = decide("Start listening!", commands=COMMANDS, gate=True, listening=False)
    assert d.kind == "command" and d.action == Action("dictation", "start")
    assert decide("hello", commands=COMMANDS, gate=True, intent="command").kind == "ignore"
    assert decide("click", commands=COMMANDS, gate=True, intent="command").kind == "command"
    assert decide("click", commands=COMMANDS, gate=True, commands_enabled=False).kind == "type"
    assert decide("  ...  ", commands=COMMANDS, gate=True).kind == "ignore"


# ----------------------------------------------------------------- face gate


def test_speech_gate_allow_ambient_requires_face_and_mouth():
    g = SpeechGate()
    assert g.allow_ambient(0, 1) is False  # nothing observed
    for i in range(10):
        g.observe(i * 0.1, None, 0)  # nobody at the screen (TV in the background)
    assert g.allow_ambient(0, 1) is False and g.allow(0, 1) is True  # PTT gate is lenient, ambient is not
    g.enroll({0: 0.3, 1: 0.1})
    for i in range(10, 20):
        g.observe(i * 0.1, 0, 2)  # the user is talking
    assert g.allow_ambient(1.0, 1.9)
    for i in range(20, 30):
        g.observe(i * 0.1, 1, 2)  # someone else talking
    assert not g.allow_ambient(2.0, 2.9)
    for i in range(30, 40):
        g.observe(i * 0.1, None, 1)  # user present, mouth closed
    assert not g.allow_ambient(3.0, 3.9)


def test_speech_gate_ambient_unenrolled_and_identity():
    g = SpeechGate()
    for i in range(10):
        g.observe(i * 0.1, 3, 1)
    assert g.allow_ambient(0, 1)
    g.enroll({3: 0.2})

    class No:
        def is_user(self, fid):
            return False

    g.identity = No()
    assert not g.allow_ambient(0, 1)


# ----------------------------------------------------------------- session


def make_session(outputs, gate=None, wake=None, clock=None, **kw):
    asr = FakeASR(outputs)
    sink = ListSink()
    ran = []

    class Clean:
        def clean(self, text, context=None):
            return text.strip()

    bus = EventBus()
    s = AmbientSession(
        asr,
        Clean(),
        sink,
        StreamingEndpointer(SR, end_silence_ms=400),
        commands=lambda: COMMANDS,
        run_command=lambda act, phrase: ran.append((str(act), phrase)),
        bus=bus,
        gate=gate,
        wake_word=wake,
        clock=clock or Clock(),
        **kw,
    )
    return s, asr, sink, ran, bus


def one_utterance(sec=0.6):
    return np.concatenate([noise(0.5), speech(sec), noise(0.6, seed=9)])


def test_session_types_when_face_gate_allows():
    windows = []

    def gate(t0, t1):
        windows.append((t0, t1))
        return True

    s, asr, sink, ran, bus = make_session(["Hello there."], gate=gate)
    res = [r for i in range(0, len(one_utterance()), 480) for r in s.feed(one_utterance()[i : i + 480])]
    assert [r.decision.kind for r in res] == ["type"]
    assert sink.items == ["Hello there. "] and ran == []
    t0, t1 = windows[0]  # gate is asked about the utterance's own time window (clock-anchored)
    assert 100.0 <= t0 < t1 <= 102.0
    assert bus.history("voice.utterance")[-1].data["decision"] == "type"


def test_session_ignores_background_talk_when_gate_denies():
    s, asr, sink, ran, _ = make_session(["Breaking news tonight"], gate=lambda a, b: False)
    stream(s, one_utterance())
    assert sink.items == [] and ran == [] and s.results[-1].decision.kind == "ignore"


def test_session_wake_word_window_and_commands():
    clock = Clock()
    s, asr, sink, ran, _ = make_session(
        ["away", "scroll down", "nice weather", "away new paragraph"], wake="away", clock=clock
    )
    for _ in range(4):
        stream(s, one_utterance())
        clock.t += 2.0
    kinds = [r.decision.kind for r in s.results]
    assert kinds == ["wake", "command", "type", "type"]
    assert ran == [("mouse:scroll_down", "scroll down")]
    assert sink.items == ["nice weather ", "new paragraph "]
    clock.t += 10.0  # window expired
    asr.outputs.append("nice weather")
    stream(s, one_utterance())
    assert s.results[-1].decision.kind == "ignore"


def test_session_stop_and_start_listening_via_controller():
    s, asr, sink, ran, bus = make_session(
        ["stop listening", "hello", "start listening", "hello again"], gate=lambda a, b: True
    )
    vc = VoiceController("always_on", ambient=s, bus=bus)
    s.run_command = lambda act, phrase: vc.handle_action(act.kind, act.arg)
    for _ in range(4):
        stream(s, one_utterance())
    assert sink.items == ["hello again "]
    states = [e.data["state"] for e in bus.history("voice.state")]
    assert states == ["listening", "stopped", "listening"]


def test_session_submit_runs_off_thread_and_keeps_no_audio(tmp_path, monkeypatch):
    jobs = []
    s, asr, sink, ran, _ = make_session(["hello"], gate=lambda a, b: True, submit=jobs.append)
    monkeypatch.chdir(tmp_path)
    assert stream(s, one_utterance()) == []  # queued, not processed inline
    assert len(jobs) == 1 and asr.calls == []
    jobs[0]()
    assert sink.items == ["hello "]
    assert len(s.results[-1].meta) >= 1 and os.listdir(tmp_path) == []  # nothing written to disk
    assert not hasattr(s.results[-1], "audio")


# --------------------------------------------------------------- controller


def make_dictation(outputs):
    return DictationSession(
        FakeASR(outputs), type("C", (), {"clean": lambda self, t, c=None: t})(), ListSink(), EnergyVAD(SR)
    )


def test_controller_mode_state_machine():
    bus = EventBus()
    d = make_dictation(["hi there"])
    s, *_ = make_session([], gate=lambda a, b: True)
    vc = VoiceController("toggle", dictation=d, ambient=s, bus=bus)
    assert vc.state == "idle"
    vc.toggle()
    assert vc.state == "recording" and d.recording
    vc.feed(np.concatenate([noise(0.3), speech(0.5), noise(0.3)]))
    res = vc.toggle()
    assert res.status == "ok" and d.sink.items == ["hi there "] and vc.state == "idle"
    vc.set_mode("push_to_talk")
    vc.ptt_press()
    assert vc.state == "recording"
    vc.set_muted(True)  # mute while recording: audio dropped, nothing transcribed
    assert vc.state == "muted" and not d.recording
    vc.ptt_press()
    assert not d.recording  # muted: the hold key does nothing
    vc.set_muted(False)
    assert vc.ptt_release() is None
    vc.set_mode("always_on")
    assert vc.state == "listening"
    vc.toggle()  # the ptt hotkey in always_on toggles listening
    assert vc.state == "stopped"
    vc.handle_action("dictation", "start")
    vc.handle_action("mic", "off")
    assert vc.state == "muted"
    vc.handle_action("mic", "toggle")
    assert vc.state == "listening"
    vc.handle_action("voice", "command_mode")
    assert vc.intent == "command"
    vc.handle_action("voice", "toggle_mode")
    assert vc.intent == "dictation"
    states = [e.data["state"] for e in bus.history("voice.state")]
    assert states == [
        "idle",
        "recording",
        "idle",
        "recording",
        "muted",
        "idle",
        "listening",
        "stopped",
        "listening",
        "muted",
        "listening",
        "listening",  # intent changes are published too, so the tray/overlay can show them
        "listening",
    ]
    assert [e.data["intent"] for e in bus.history("voice.state")][-2:] == ["command", "dictation"]
    with pytest.raises(ValueError, match="always_on\\|push_to_talk\\|toggle"):
        vc.set_mode("ambient")
    with pytest.raises(ValueError):
        VoiceController("ambient")


def test_controller_mute_drops_audio_before_processing():
    s, asr, sink, ran, _ = make_session(["hello"], gate=lambda a, b: True)
    vc = VoiceController("always_on", ambient=s)
    vc.set_muted(True)
    for i in range(0, len(one_utterance()), 480):
        vc.feed(one_utterance()[i : i + 480])
    assert asr.calls == [] and not s.ep.in_speech and s.ep._utt == []
    vc.enabled = False
    assert vc.state == "off" and vc.feed(np.zeros(480)) is None


def test_controller_ptt_dictation_actions():
    d = make_dictation(["x"])
    vc = VoiceController("push_to_talk", dictation=d)
    vc.handle_action("dictation", "start")
    assert d.recording
    vc.handle_action("dictation", "stop")
    assert not d.recording
    vc.handle_action("dictation", "toggle")
    assert d.recording
    vc.handle_action("voice", "command_mode")  # no ambient session: ignored, no crash
    assert vc.intent == "dictation" and vc.listening


def test_dictation_command_handler_skips_typing():
    seen = []
    d = make_dictation(["Scroll down."])
    d.command_handler = lambda raw: seen.append(raw) or True
    d.start()
    d.feed(np.concatenate([noise(0.3), speech(0.5), noise(0.3)]))
    r = d.stop()
    assert r.status == "command" and d.sink.items == [] and seen == ["Scroll down."]
    d.start()
    d.cancel()
    assert not d.recording and d.stop().status == "empty"
