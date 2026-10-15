"""v0.1.26 conversational voice: the listening window, conversational mode, barge-in, lower latency.

Each test names the old behaviour it guards against; most fail on v0.1.25."""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from away.liveagent import settings as st
from away.voice import latency
from away.voice.always_on import AmbientSession, StreamingEndpointer, Utterance, VoiceController
from away.voice.output import ListSink
from away.voice.speech import EchoGuard, Speaker
from away.voice.talk import TalkHook, TalkState, is_stop

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class ScriptASR:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.last_confidence = 0.95

    def transcribe(self, audio: Any, prompt: Any = None) -> str:
        return self.texts.pop(0)


def _session(texts: list[str], talk: TalkHook, clock: Clock) -> tuple[AmbientSession, ListSink]:
    sink = ListSink()
    amb = AmbientSession(
        ScriptASR(texts),
        SimpleNamespace(clean=lambda t, app=None: t),  # type: ignore[arg-type]
        sink,
        StreamingEndpointer(16000),
        commands=dict,
        run_command=lambda a, p: None,
        talk=talk,
        clock=clock,
        intent="dictation",
        fallback="type",
    )
    return amb, sink


def _utt(clock: Clock) -> Utterance:
    return Utterance(np.full(16000, 0.1, dtype=np.float32), clock.t - 1.0, clock.t, closed_at=clock.t)


def _talk(clock: Clock, window: Any = 8.0, conversational: bool = False) -> tuple[TalkHook, list[tuple[str, str]]]:
    talk = TalkHook(window_s=window, conversational=lambda: conversational)
    talk.state.clock = clock
    talk.agents = lambda: [{"id": "cookie", "name": "Cookie", "wake_word": ""}]
    sent: list[tuple[str, str]] = []
    talk.deliver = lambda t, x, s: (sent.append((t, x)), "Cookie is answering")[1]
    return talk, sent


# ------------------------------------------------------------------------------------------------ listen window
def test_listen_window_is_the_setting_and_is_honoured() -> None:
    """v0.1.25: a fixed 8 s; now Settings > Voice & input "Keep listening after the wake word" (read live)."""
    clock = Clock()
    secs = [3.0]
    talk, sent = _talk(clock, window=lambda: secs[0])
    amb, sink = _session(["Hey Cookie", "what's the weather"], talk, clock)
    amb.process(_utt(clock))  # bare wake phrase: waits for the message
    clock.t += 2.9
    assert talk.state.armed() == "cookie"
    clock.t += 0.2  # 3.1 s: the window closed
    assert talk.state.armed() is None
    clock.t += 1.1  # v0.1.27: routing goes by when speech STARTED; this 1 s utterance starts after the window
    amb.process(_utt(clock))
    assert sent == [] and sink.items  # typed as dictation, not sent
    secs[0] = 20.0
    talk.arm("cookie", "voice")
    clock.t += 19.0
    assert talk.state.armed() == "cookie"


def test_listen_seconds_config_cli_and_range(tmp_path) -> None:
    from away import cli
    from away import config as cfgmod

    p = tmp_path / "c.yaml"
    assert cli.main(["voice", "listen-seconds", "12", "--config", str(p)]) == 0
    assert cli.main(["voice", "conversational", "on", "--config", str(p)]) == 0
    assert cli.main(["voice", "agent-pause-ms", "400", "--config", str(p)]) == 0
    cfg, _warn = cfgmod.load(p)
    assert cfg.voice.listen_seconds == 12.0 and cfg.voice.conversational and cfg.voice.agent_end_silence_ms == 400
    assert cli.main(["voice", "listen-seconds", "45", "--config", str(p)]) != 0  # 3..30
    assert cli.main(["voice", "conversational", "maybe", "--config", str(p)]) != 0
    d = cfgmod.AppConfig().voice
    assert d.listen_seconds == 8.0 and d.conversational is False and d.agent_end_silence_ms < d.end_silence_ms


# ------------------------------------------------------------------------------------------------ conversation
def test_conversational_follow_up_needs_no_wake_word() -> None:
    clock = Clock()
    talk, sent = _talk(clock, window=5.0, conversational=True)
    amb, sink = _session(["Hey Cookie, what's the weather?", "and tomorrow?", "will it rain?"], talk, clock)
    amb.process(_utt(clock))
    clock.t += 1.0
    amb.process(_utt(clock))
    clock.t += 4.0
    amb.process(_utt(clock))
    assert sent == [("cookie", "what's the weather?"), ("cookie", "and tomorrow?"), ("cookie", "will it rain?")]
    assert not sink.items and talk.conversation == "cookie"


def test_without_conversational_mode_a_follow_up_is_not_sent() -> None:
    clock = Clock()
    talk, sent = _talk(clock, conversational=False)
    amb, sink = _session(["Hey Cookie, what's the weather?", "and tomorrow?"], talk, clock)
    amb.process(_utt(clock))
    clock.t += 1.0
    amb.process(_utt(clock))
    assert sent == [("cookie", "what's the weather?")] and "".join(sink.items).strip() == "and tomorrow?"


def test_follow_up_window_waits_while_the_agent_answers_then_times_out() -> None:
    clock = Clock()
    talk, sent = _talk(clock, window=4.0, conversational=True)
    busy = [True]
    talk.busy = lambda: busy[0]
    talk.send("cookie", "tell me a long story", "voice")
    clock.t += 30.0  # a long spoken answer: the window does not run out meanwhile
    assert talk.state.armed() == "cookie"
    busy[0] = False  # the agent went quiet: 4 s from now
    clock.t += 3.9
    assert talk.state.armed() == "cookie"
    clock.t += 0.2
    assert talk.state.armed() is None


@pytest.mark.parametrize("said", ["stop", "That's all.", "okay, that's all", "that's all, thanks", "Goodbye!"])
def test_stop_phrases_end_the_conversation(said: str) -> None:
    clock = Clock()
    talk, sent = _talk(clock, conversational=True)
    stopped: list[str] = []
    talk.on_stop = stopped.append
    amb, sink = _session(["Hey Cookie, what's the weather?", said, "and tomorrow?"], talk, clock)
    amb.process(_utt(clock))
    res = amb.process(_utt(clock))
    assert res.decision.reason == "conversation ended" and stopped == ["cookie"]
    amb.process(_utt(clock))
    assert sent == [("cookie", "what's the weather?")] and talk.state.armed() is None and sink.items


def test_is_stop_does_not_eat_real_questions() -> None:
    assert is_stop("stop") and is_stop("That\u2019s all")
    assert not is_stop("stop the music in Spotify") and not is_stop("what is all of this")


def test_wake_word_model_gate_stays_open_for_a_follow_up() -> None:
    """v0.1.25: with the wake-word model in front, audio without "Hey <name>" never reached speech-to-text."""
    from away.voice.wakeword import WakeGate

    clock = Clock()
    talk, _ = _talk(clock, conversational=True)
    session = SimpleNamespace(talk=talk, ep=SimpleNamespace(in_speech=False), feed=lambda a: [], reset=lambda: None)
    gate = WakeGate(session, SimpleNamespace(feed=lambda a: False), clock=clock)
    assert not gate.awake
    talk.send("cookie", "hi", "voice")
    assert gate.awake


# ------------------------------------------------------------------------------------------------ end of speech
def _close_after(ep: StreamingEndpointer) -> float:
    rng = np.random.default_rng(1)
    ep.feed(rng.normal(0, 0.001, 9600).astype(np.float32))
    ep.feed(rng.normal(0, 0.12, 16000).astype(np.float32))
    sil = rng.normal(0, 0.001, 480).astype(np.float32)
    for i in range(100):
        if ep.feed(sil):
            return (i + 1) * 0.03
    return 99.0


def test_quicker_end_of_speech_while_talking_to_an_agent() -> None:
    slow = StreamingEndpointer(16000, end_silence_ms=700, quick_end_silence_ms=450)
    fast = StreamingEndpointer(16000, end_silence_ms=700, quick_end_silence_ms=450)
    fast.quick = lambda: True
    assert _close_after(slow) == pytest.approx(0.69, abs=0.031)  # dictation keeps the longer pause
    assert _close_after(fast) == pytest.approx(0.45, abs=0.031)
    assert fast.last_end_wait_s == pytest.approx(0.45, abs=0.031)


# ------------------------------------------------------------------------------------------------ barge-in
def test_barge_in_keeps_your_first_words_in_a_conversation() -> None:
    """v0.1.25: the microphone was deaf while the voice spoke (and 0.7 s after), so the words you talked over it
    with were lost even though the voice stopped."""
    clock = Clock()
    echo = EchoGuard(clock=clock)
    fed: list[int] = []
    amb = SimpleNamespace(feed=lambda c: fed.append(len(c)), reset=lambda: None, listening=True, intent="dictation")
    vc = VoiceController("always_on", None, amb)  # type: ignore[arg-type]
    vc.echo = echo
    echo.begin()
    for _ in range(40):  # 1.2 s of 30 ms chunks while the voice speaks
        vc.feed(np.zeros(480, dtype=np.float32))
        clock.t += 0.03
    assert fed == []
    echo.cut()  # barge-in fired
    vc.feed(np.zeros(480, dtype=np.float32))
    assert len(fed) == 1 and fed[0] >= 16000  # ~1 s held while it spoke + this chunk, replayed
    assert not echo.overlaps(clock.t - 1.0, clock.t + 1.0)  # those words are not "the computer's own voice"


def test_without_a_barge_in_held_audio_is_dropped() -> None:
    clock = Clock()
    echo = EchoGuard(tail_s=0.0, clock=clock)
    fed: list[int] = []
    amb = SimpleNamespace(feed=lambda c: fed.append(len(c)), reset=lambda: None, listening=True, intent="dictation")
    vc = VoiceController("always_on", None, amb)  # type: ignore[arg-type]
    vc.echo = echo
    with echo.speaking():
        vc.feed(np.zeros(480, dtype=np.float32))
    clock.t += 0.1
    vc.feed(np.zeros(480, dtype=np.float32))
    assert fed == [480]  # only the new chunk: the computer's voice is never transcribed


def _fake_app(conversational: bool = True) -> Any:
    talk = TalkHook(window_s=5.0, conversational=lambda: conversational)
    talk.agents = list
    talk.deliver = lambda t, x, s: ""
    return SimpleNamespace(
        talk=talk,
        bus=SimpleNamespace(publish=lambda *a, **k: None),
        agent_link=None,
        voice=None,
        cfg=SimpleNamespace(speech=None, voice=SimpleNamespace(sample_rate=16000)),
    )


def test_barge_in_stops_the_voice_after_the_model_finished_writing(tmp_path) -> None:
    """v0.1.25: barge-in only worked while ``agent.busy`` (the model writing); the voice reads the last sentences
    after that, and talking over them did nothing."""
    from away.liveagent.integrate import LiveAgentHost

    app = _fake_app()
    host = LiveAgentHost(app, tmp_path)
    st.save(st.LiveSettings(enabled=True, providers=[st.ProviderConfig("groq", "groq", "m")]), host.watcher.path)
    host.watcher.poll()
    app.voice = SimpleNamespace(feed=lambda c: None)
    host.speech.speaking = lambda: True  # type: ignore[method-assign]
    assert host.hook_barge_in()
    stopped: list[int] = []
    host.agent.cancel = lambda: stopped.append(1)  # type: ignore[method-assign]
    host.agent.busy = False
    clock = Clock()
    host.barge.clock = clock
    for i in range(30):
        loud = i > 8
        app.voice.feed(np.full(1600, 0.3 if loud else 0.01, dtype=np.float32))
        clock.t += 0.1
    assert stopped


def test_live_speech_interrupt_stops_tts_mid_reply() -> None:
    from away.liveagent.speech import LiveSpeech

    stops: list[int] = []
    sp = Speaker(lambda: None, factories=[("slow", lambda: SlowEngine())], echo=EchoGuard(tail_s=0.0))
    ls = LiveSpeech(sp, stop_audio=lambda: stops.append(1))
    ls.begin()
    for s in ("One sentence here.", "Two sentences here.", "Three sentences here."):
        ls.say(s)
    time.sleep(0.05)
    ls.interrupt()
    assert ls.say("Four.") is False and stops
    sp.wait(2)
    assert len(sp.spoken) <= 1  # the waiting sentences were dropped


class SlowEngine:
    name = "slow"

    def speak(self, text: str, v: Any) -> None:
        time.sleep(0.2)

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------ latency
class StreamFunnel:
    def __init__(self, parts: list[str], gap: float = 0.05) -> None:
        self.parts, self.gap = parts, gap
        self.done_at: float | None = None
        self.router = SimpleNamespace(candidates=lambda task: [])

    def stream(self, task: str, messages: Any, rec: Any = None, **_k: Any) -> Any:
        for p in self.parts:
            time.sleep(self.gap)
            yield p
        self.done_at = time.monotonic()


class TimedSpeech:
    def __init__(self) -> None:
        self.at: list[float] = []

    def begin(self) -> None:
        pass

    def say(self, s: str) -> bool:
        self.at.append(time.monotonic())
        return True

    def interrupt(self) -> None:
        pass


def test_speaking_starts_before_the_reply_is_complete(tmp_path) -> None:
    from away.liveagent.agent import LiveAgent

    f = StreamFunnel(["Sure thing, here it is. ", "The second ", "part is ", "longer and ", "comes later."])
    sp = TimedSpeech()
    agent = LiveAgent(lambda: st.LiveSettings(enabled=True), lambda: f, None, None, sp)
    agent.handle("how are you today")
    assert f.done_at is not None and sp.at and sp.at[0] < f.done_at - 0.1


def test_plain_questions_skip_the_routing_call(tmp_path) -> None:
    """v0.1.25: every request first asked Jev or a model what kind of request it was (often ~1 s or more)."""
    from away.liveagent.agent import LiveAgent

    calls: list[str] = []

    class Decider:
        def route(self, text: str) -> Any:
            calls.append(text)
            return SimpleNamespace(value="chat", source="jev")

    f = StreamFunnel(["Twelve."], gap=0.0)
    agent = LiveAgent(lambda: st.LiveSettings(enabled=True), lambda: f, lambda: Decider(), None, TimedSpeech())
    assert agent.handle("what's fifteen percent of eighty").route_by == "quick" and calls == []
    agent.handle("open Notepad and type hello")
    agent.handle("can you make the screen brighter")
    assert len(calls) == 2  # anything that may be an action still gets the routing decision
    slow = LiveAgent(lambda: st.LiveSettings(enabled=True, quick_route=False), lambda: f, lambda: Decider(), None, None)
    slow.handle("what's the time")
    assert len(calls) == 3


def test_providers_are_reused_between_turns(tmp_path) -> None:
    from away.liveagent.integrate import LiveAgentHost

    host = LiveAgentHost(_fake_app(), tmp_path)
    built: list[int] = []
    host.runtime.funnel = lambda s=None: built.append(1) or object()  # type: ignore[method-assign]
    a, b = host.funnel(), host.funnel()
    assert a is b and len(built) == 1
    st.save(st.LiveSettings(enabled=True, name="Jarvis"), host.watcher.path)
    host.watcher.poll()
    host.funnel()
    assert len(built) == 2  # a settings change rebuilds them


def test_voice_engine_is_loaded_before_the_first_sentence() -> None:
    """v0.1.25: the voice engine loaded when the first sentence arrived, and unloaded 30 s after the last one."""
    made: list[float] = []
    sp = Speaker(lambda: None, factories=[("e", lambda: made.append(time.monotonic()) or SlowEngine())])
    assert sp.warm()
    deadline = time.monotonic() + 2
    while not made and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(made) == 1 and sp.spoken == []  # loaded, nothing said
    sp.say("Hello there.")
    sp.wait(2)
    assert len(made) == 1 and sp.spoken == ["Hello there."]  # the warm engine said it
    sp.stop()


def test_live_agent_host_keeps_the_voice_loaded(tmp_path) -> None:
    from away.liveagent.integrate import WARM_IDLE_S, LiveAgentHost

    host = LiveAgentHost(_fake_app(), tmp_path)
    assert getattr(host.speech.speaker, "idle_s", WARM_IDLE_S) >= WARM_IDLE_S


def test_listening_sound_plays_on_wake_and_when_a_message_is_taken() -> None:
    from away.voice.earcon import Earcon, tone

    played: list[int] = []
    ev = threading.Event()
    e = Earcon(play=lambda s, sr: (played.append(len(s)), ev.set()))
    clock = Clock()
    talk, _ = _talk(clock)
    talk.sound = e
    amb, _sink = _session(["Hey Cookie", "what's up"], talk, clock)
    amb.process(_utt(clock))
    amb.process(_utt(clock))
    assert e.played == ["listen", "heard"] and ev.wait(1) and played
    assert 0.05 < len(tone("listen")) / 24000 < 0.25  # shorter than the shortest utterance
    quiet = Earcon(enabled=lambda: False, play=lambda s, sr: None)
    assert quiet("listen") is False
    busy = Earcon(speaking=lambda: True, play=lambda s, sr: None)
    assert busy("listen") is False  # never over the computer's own voice


def test_timing_log_covers_every_stage_end_to_end(tmp_path) -> None:
    """The whole path with fakes: endpoint, speech-to-text, route, first token, first sentence, first sound."""
    bench = _bench()
    rows = bench.run("after", turns=1, scale=0.1)
    assert rows and set(latency.STAGES) <= set(rows[0]) and rows[0]["total"] > 0


def test_before_and_after_numbers(tmp_path) -> None:
    """The fixes, measured by the timing log on the same fakes (scaled down): see scripts/bench_voice_latency.py."""
    bench = _bench()
    before = latency.summary(bench.run("before", turns=2, scale=0.2))
    after = latency.summary(bench.run("after", turns=2, scale=0.2))
    assert after["endpoint"] < before["endpoint"] - 150  # 700 -> 450 ms pause
    assert after["route"] < 50 < before["route"]  # no routing call for a plain question
    assert after["first_audio"] < before["first_audio"] - 100  # the voice was loaded while you spoke
    assert after["total"] < before["total"] * 0.7


def _bench() -> Any:
    spec = importlib.util.spec_from_file_location("bench_voice_latency", ROOT / "scripts" / "bench_voice_latency.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trace_breakdown_math() -> None:
    clock = Clock(10.0)
    tr = latency.Trace(9.5, clock=clock)
    tr.mark("endpoint", 10.0)
    tr.mark("stt", 10.2)
    tr.mark("first_audio", 11.0)
    b = tr.breakdown()
    assert b == {"endpoint": 500.0, "stt": 200.0, "first_audio": 800.0, "total": 1500.0}
    assert "total=1500ms" in tr.line()


def test_settings_tab_lists_the_new_switches() -> None:
    src = (ROOT / "src" / "away" / "app" / "settings_gui.py").read_text(encoding="utf-8")
    assert "voice.listen_seconds" in src and "voice.conversational" in src and "Conversational mode" in src
    assert TalkState  # imported for the type
