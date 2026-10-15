"""v0.1.27 regression tests from the real-PC report (v0.1.26, live agent "sunday", JoyCam):

1. the dictation face gate (face_fraction 0) dropped "Hey sunday" so the live agent never heard it;
2. a reply begun inside the listening window was typed as dictation because the window was checked only after
   speech-to-text finished;
3. Agent Link re-assigning ``talk.agents`` / ``deliver`` (a restart, a remote agent pairing) unhooked the live agent;
4. Auto speech language heard English as Malay.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from away.liveagent import settings as st
from away.voice.always_on import AmbientSession, StreamingEndpointer, Utterance
from away.voice.output import ListSink
from away.voice.talk import TalkHook


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


def _live(tmp_path: Any, clock: Clock | None = None) -> tuple[Any, Any, list[str], list[tuple[str, str]]]:
    """A fake app with Agent Link's talk hook (a remote "AI agent") and the live agent named Sunday, on."""
    from away.liveagent.integrate import LiveAgentHost

    talk = TalkHook()
    if clock is not None:
        talk.state.clock = clock
    remote: list[tuple[str, str]] = []
    talk.agents = lambda: [{"id": "r1", "name": "AI agent", "wake_word": ""}]
    talk.deliver = lambda t, x, s: (remote.append((t, x)), "Sent to AI agent")[1]
    app = SimpleNamespace(
        talk=talk,
        bus=SimpleNamespace(publish=lambda *a, **k: None),
        agent_link=None,
        voice=None,
        cfg=SimpleNamespace(speech=None, voice=SimpleNamespace(sample_rate=16000)),
    )
    host = LiveAgentHost(app, tmp_path)
    st.save(
        st.LiveSettings(enabled=True, name="Sunday", providers=[st.ProviderConfig("opencode_free", "opencode_free")]),
        host.watcher.path,
    )
    host.watcher.poll()
    host.hook_talk()
    got: list[str] = []
    host.agent.submit = lambda text, source="voice": got.append(text) or True  # type: ignore[method-assign]
    host.warm = lambda: None  # type: ignore[method-assign]
    return host, talk, got, remote


def _session(texts: list[str], talk: TalkHook, clock: Clock, gate: Any = None) -> tuple[AmbientSession, ListSink]:
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
        gate=gate,
        intent="dictation",
        fallback="commands_only",
    )
    return amb, sink


def _utt(t0: float, t1: float, closed_at: float | None = None, end_wait: float = 0.0) -> Utterance:
    n = max(1600, int((t1 - t0) * 16000))
    return Utterance(np.full(n, 0.1, dtype=np.float32), t0, t1, closed_at=closed_at, end_wait_s=end_wait)


# ------------------------------------------------------------------------------------------------ 1. face gate
def test_wake_phrase_with_no_face_at_the_screen_still_reaches_the_live_agent(tmp_path) -> None:
    clock = Clock()
    host, talk, got, remote = _live(tmp_path, clock)
    no_face = lambda t0, t1: False  # noqa: E731 - face_fraction 0.0: "face gate: no face at the screen"
    amb, sink = _session(
        ["Hey Sunday, can you see my screen?", "this stray speech must not be typed"], talk, clock, no_face
    )
    r = amb.process(_utt(99.0, 100.0))
    assert got == ["can you see my screen?"] and r.decision.kind == "message"
    clock.t += 30
    r2 = amb.process(_utt(clock.t - 1, clock.t))
    assert r2.decision.kind == "ignore" and r2.decision.reason.startswith("face gate") and not sink.items


def test_bare_wake_phrase_then_message_with_the_gate_closed(tmp_path) -> None:
    clock = Clock()
    host, talk, got, remote = _live(tmp_path, clock)
    amb, sink = _session(["Hey Sunday", "Can you see my screen?"], talk, clock, lambda t0, t1: False)
    assert amb.process(_utt(99.0, 100.0)).decision.kind == "listen"
    clock.t += 3
    amb.process(_utt(clock.t - 1, clock.t))
    assert got == ["Can you see my screen?"] and not sink.items and remote == []


# ------------------------------------------------------------------------------------------------ 2. window
def test_speech_starting_at_7_5s_of_an_8s_window_still_routes_to_the_agent(tmp_path) -> None:
    clock = Clock(100.0)
    host, talk, got, remote = _live(tmp_path, clock)
    amb, sink = _session(["Hey Sunday", "Yeah. Can you see my screen?"], talk, clock, lambda t0, t1: True)
    # "Hey Sunday" ended at 99.5 (0.5 s of silence closed it at 100.0); speech-to-text finished at 101.0
    clock.t = 101.0
    assert amb.process(_utt(99.0, 99.5, closed_at=100.0, end_wait=0.5)).decision.kind == "listen"
    # the reply STARTED at 107.0 (7.5 s into the 8 s window), lasted 2 s, closed at 109.5, transcribed at 111.0
    clock.t = 111.0
    r = amb.process(_utt(107.0, 109.0, closed_at=109.5, end_wait=0.5))
    assert got == ["Yeah. Can you see my screen?"] and not sink.items, r.decision


def test_speech_starting_after_the_window_is_dictation_and_says_so(tmp_path) -> None:
    clock = Clock(100.0)
    host, talk, got, remote = _live(tmp_path, clock)
    amb, sink = _session(["Hey Sunday", "Good morning everyone"], talk, clock, lambda t0, t1: True)
    amb.fallback = "type"
    clock.t = 100.2
    amb.process(_utt(99.0, 99.5, closed_at=100.0, end_wait=0.5))
    clock.t = 110.0
    amb.process(_utt(108.0, 109.0, closed_at=109.5, end_wait=0.5))  # started 0.5 s after the window closed
    assert got == [] and sink.items
    assert any(a["kind"] == "expired" for a in host.attempts)


# ------------------------------------------------------------------------------------------------ 3. re-hook
def test_agent_link_restart_does_not_unhook_the_live_agent(tmp_path) -> None:
    host, talk, got, remote = _live(tmp_path)
    armed: list[str] = []
    # Agent Link (re)starts: a new private link, a remote agent paired; it assigns its own callbacks again
    talk.agents = lambda: [{"id": "r2", "name": "AI agent", "wake_word": ""}]
    talk.deliver = lambda t, x, s: (remote.append((t, x)), "Sent to AI agent. AI agent isn't connected")[1]
    talk.on_armed = armed.append
    route = talk.route("Hey Sunday, what's on my screen?")
    assert route is not None and route.target == "live-agent"
    assert talk.send(route.target, route.body, "voice") == "Sunday is answering"
    assert got == ["what's on my screen?"] and remote == []
    talk.arm("live-agent", "voice")
    assert armed == []  # Agent Link's "say your message for AI agent" is not shown for the live agent
    hey = talk.route("Hey AI agent, hello")  # the remote agent still works
    assert hey is not None and talk.send(hey.target, hey.body, "voice").startswith("Sent to AI agent")


def test_live_agent_answers_even_when_agent_link_is_off(tmp_path) -> None:
    host, talk, got, remote = _live(tmp_path)
    talk.deliver = None
    talk.agents = list
    route = talk.route("Hey Sunday, hello")
    assert route is not None and talk.send(route.target, route.body, "voice") == "Sunday is answering"
    host.hook_talk()
    host.hook_talk()
    assert len(talk.providers) == 1  # idempotent


# ------------------------------------------------------------------------------------------------ 4. language
class _Model:
    """Whisper guessing Malay for English unless told the language."""

    def __init__(self, prob: float = 0.55) -> None:
        self.calls: list[Any] = []
        self.prob = prob

    def transcribe(self, audio: Any, language: Any = None, **kw: Any) -> Any:
        self.calls.append(language)
        if language is None:
            seg = SimpleNamespace(text="Bersihkanlah anda lihat skrin saya?", no_speech_prob=0.0, avg_logprob=-0.4)
            return iter([seg]), SimpleNamespace(language="ms", language_probability=self.prob)
        seg = SimpleNamespace(text="Can you see my screen?", no_speech_prob=0.0, avg_logprob=-0.1)
        return iter([seg]), SimpleNamespace(language=language, language_probability=1.0)


def test_auto_language_falls_back_to_english_when_unsure() -> None:
    from away.voice.asr import whisper_transcribe
    from away.voice.language import LanguageMemory

    m = _Model()
    stats: dict[str, Any] = {}
    text = whisper_transcribe(m, np.zeros(1600, np.float32), None, None, None, 5, stats, LanguageMemory())
    assert text == "Can you see my screen?" and m.calls == [None, "en"] and stats["language"] == "en"


def test_a_confident_language_you_never_speak_also_falls_back_until_it_is_learned() -> None:
    from away.voice.language import LEARN_AFTER, LanguageMemory

    mem = LanguageMemory()
    assert mem.fallback("ms", 0.95) == "en"  # not one of your languages yet
    for _ in range(LEARN_AFTER - 2):
        mem.fallback("ms", 0.95)
    assert mem.fallback("ms", 0.95) is None and "ms" in mem.languages()  # you really speak it now
    assert mem.fallback("en", 0.99) is None and mem.fallback("ja", 0.3) is not None


def test_a_chosen_language_is_passed_straight_through() -> None:
    from away.voice.asr import whisper_transcribe
    from away.voice.language import LanguageMemory

    m = _Model()
    assert whisper_transcribe(m, np.zeros(1600, np.float32), "en", None, None, 5, None, LanguageMemory()) == (
        "Can you see my screen?"
    )
    assert m.calls == ["en"]


def test_speech_language_in_settings_and_for_agents_with_confirmation() -> None:
    from away import config as cfgmod
    from away.agentlink import registry as reg
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(cfgmod.AppConfig())
    assert m.language_label().startswith("Auto") and "English" in m.language_choices()
    assert m.set_language_label("English") is None and m.cfg.voice.language == "en"
    assert m.set_language_label(m.language_choices()[0]) is None and m.cfg.voice.language is None
    cfg = cfgmod.AppConfig()
    a = reg.settings_apply(cfg, reg.Op("settings", "set", "voice.language", "en"))
    assert cfg.voice.language == "en" and not a.low_risk  # you confirm it on the PC
    reg.settings_apply(cfg, a.inverse[0])
    assert cfg.voice.language is None
    b = reg.settings_apply(cfg, reg.Op("settings", "set", "voice.listen_seconds", 12))
    assert cfg.voice.listen_seconds == 12 and not b.low_risk
    rows = {r["key"]: r for r in reg.settings_list(cfg)}
    assert rows["voice.language"]["value"] == "auto" and "en" in rows["voice.language"]["choices"]
