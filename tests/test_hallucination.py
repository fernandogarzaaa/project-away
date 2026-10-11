"""v0.1.6: Whisper prompt echo / silence hallucinations are never typed.

Real bug: dictating typed "Dictating into <app>." because Whisper repeated its initial_prompt on short or
noisy audio. The app is no longer in the prompt, and transcripts that echo the prompt are filtered.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from away.bindings import Action, BindingsConfig
from away.events import EventBus
from away.voice.always_on import AmbientSession, StreamingEndpointer
from away.voice.asr import FakeASR, whisper_transcribe
from away.voice.cleanup import RuleCleaner
from away.voice.context import whisper_prompt
from away.voice.dictation import DictationSession
from away.voice.edit import SelectionEditor
from away.voice.hallucination import (
    clean_transcript,
    is_silence_phrase,
    is_vocabulary_list,
    join_segments,
    segment_ok,
    strip_prompt_sentences,
)
from away.voice.output import ListSink
from away.voice.vad import EnergyVAD

SR = 16000
VOCAB = ["Inan", "GitHub", "Ruff", "Calabanga"]
PROMPT = whisper_prompt("Slack", VOCAB)


def tone(sec, amp=0.2, f=220.0):
    t = np.arange(int(sec * SR)) / SR
    return (amp * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) * np.sin(2 * np.pi * f * t)).astype(np.float32)


def quiet(sec, amp=0.001, seed=0):
    return (np.random.default_rng(seed).normal(0, amp, int(sec * SR))).astype(np.float32)


# ------------------------------------------------------------------ prompt


@pytest.mark.parametrize("app", ["Slack", "main.py - Visual Studio Code", "Inbox (3) - Gmail", None])
def test_prompt_never_names_the_app(app):
    p = whisper_prompt(app, VOCAB)
    assert p == "Vocabulary: Inan, GitHub, Ruff, Calabanga."
    assert "Dictating" not in p
    if app:
        assert app not in p
    assert whisper_prompt(app, []) is None


# ------------------------------------------------------------------ prompt echo


@pytest.mark.parametrize(
    "heard",
    [
        "Dictating into Slack.",
        "Dictating into main.py - Visual Studio Code.",
        "dictating into notepad",
        "Vocabulary: Inan, GitHub, Ruff, Calabanga.",
        "Vocabulary: Inan, GitHub.",
        "Vocabulary: Inan, GitHub, Ruff, Calabanga. Dictating into Slack.",
        "Inan, GitHub, Ruff, Calabanga.",
        "GitHub, Ruff.",
        "Inan GitHub Ruff",
        "GitHub, Ruff, Calabanga",
    ],
)
def test_prompt_echo_is_dropped(heard):
    text, why = clean_transcript(heard, PROMPT, VOCAB, seconds=1.0)
    assert text == ""
    assert "echo" in why


def test_prompt_echo_dropped_even_for_long_audio():
    # an echo is an echo however long the noisy recording was
    assert clean_transcript("Dictating into Slack.", PROMPT, VOCAB, seconds=8.0)[0] == ""
    assert clean_transcript("Vocabulary: Inan, GitHub, Ruff, Calabanga.", PROMPT, VOCAB, seconds=8.0)[0] == ""


def test_prompt_sentence_stripped_but_real_speech_kept():
    assert clean_transcript("Dictating into Slack. Send the report today.", PROMPT, VOCAB, 3.0) == (
        "Send the report today.",
        "prompt echo stripped",
    )
    assert clean_transcript("Hello there. Dictating into Code.exe.", PROMPT, VOCAB, 3.0)[0] == "Hello there."
    assert strip_prompt_sentences("Vocabulary: a, b. Ship it!") == "Ship it!"


@pytest.mark.parametrize(
    "heard",
    [
        "GitHub",  # one vocabulary word is a real dictation
        "Push the fix to GitHub and run Ruff.",
        "Inan, can you check GitHub?",
        "The vocabulary test is on Monday.",  # 'vocabulary' mid-sentence is fine
        "I am dictating into the mic now.",
        "Thank you for the review, I will merge it after lunch.",
    ],
)
def test_real_speech_is_kept(heard):
    assert clean_transcript(heard, PROMPT, VOCAB, seconds=2.5) == (heard, "")


def test_vocabulary_list_detection():
    assert is_vocabulary_list("Inan, GitHub.", VOCAB)
    assert not is_vocabulary_list("Inan, hello.", VOCAB)
    assert not is_vocabulary_list("GitHub", VOCAB)
    assert not is_vocabulary_list("Inan, GitHub.", [])


# ------------------------------------------------------------------ silence hallucinations


@pytest.mark.parametrize(
    "heard",
    [
        "Thank you.",
        "Thanks for watching!",
        "Thank you for watching.",
        "Please subscribe.",
        "you",
        "Bye.",
        "Thank you. Thank you.",
        "Thank you so much.",
    ],
)
def test_stock_phrases_dropped_on_short_utterances(heard):
    assert clean_transcript(heard, PROMPT, VOCAB, seconds=0.9) == ("", "silence hallucination")


def test_stock_phrases_kept_on_longer_utterances():
    assert clean_transcript("Thank you.", PROMPT, VOCAB, seconds=3.0) == ("Thank you.", "")
    assert not is_silence_phrase("Thank you.", 2.5)


def test_subtitle_credits_dropped_at_any_length():
    assert is_silence_phrase("Subtitles by the Amara.org community", 9.0)
    assert clean_transcript("Transcribed by ESO, translated by -", None, (), 6.0)[0] == ""


def seg(text, nsp=None, alp=None):
    d = {"text": text}
    if nsp is not None:
        d["no_speech_prob"] = nsp
    if alp is not None:
        d["avg_logprob"] = alp
    return SimpleNamespace(**d)


def test_segment_ok_needs_both_signals():
    assert not segment_ok(seg("Dictating into Slack.", 0.9, -1.4))  # silent and unsure: dropped
    assert segment_ok(seg("hello", 0.9, -0.3))  # confident text despite a quiet start
    assert segment_ok(seg("hello", 0.1, -1.6))  # speech with an unusual word
    assert segment_ok(seg("no attributes"))  # other backends / fakes
    assert segment_ok(SimpleNamespace(text="x", no_speech_prob="bad", avg_logprob=None))


def test_join_segments_drops_silent_ones():
    segs = [seg(" Hello there.", 0.05, -0.2), seg(" Thank you.", 0.95, -1.3), seg(" Bye now.", 0.2, -0.4)]
    assert join_segments(segs) == "Hello there. Bye now."


def test_whisper_transcribe_filters_segments():
    class Model:
        def transcribe(
            self,
            audio,
            language=None,
            initial_prompt=None,
            vad_filter=False,
            beam_size=5,
            condition_on_previous_text=False,
        ):
            self.prompt = initial_prompt
            return iter([seg(" Dictating into Slack.", 0.92, -1.2), seg(" Ship it.", 0.03, -0.1)]), None

    m = Model()
    assert whisper_transcribe(m, np.zeros(SR, np.float32), "en", PROMPT, None) == "Ship it."
    assert m.prompt == PROMPT


# ------------------------------------------------------------------ pipelines


def dictation(outputs, vocabulary=VOCAB):
    asr = FakeASR(outputs)
    sink = ListSink()
    bus = EventBus()
    s = DictationSession(
        asr, RuleCleaner(), sink, EnergyVAD(SR), bus, SR, context=lambda: "Slack", vocabulary=list(vocabulary)
    )
    return s, asr, sink


def record(s, audio):
    s.start()
    for i in range(0, len(audio), 480):
        s.feed(audio[i : i + 480])
    return s.stop()


def test_push_to_talk_never_types_prompt_echo():
    s, asr, sink = dictation(["Dictating into Slack."])
    r = record(s, np.concatenate([quiet(0.3), tone(0.5), quiet(0.3, seed=1)]))
    assert r.status == "empty" and r.meta["reason"] == "prompt echo"
    assert sink.items == []
    assert "Slack" not in (asr.calls[0][1] or "")


def test_push_to_talk_drops_short_thank_you_but_types_real_text():
    s, _asr, sink = dictation(["Thank you.", "Ship the build tonight."])
    r = record(s, np.concatenate([quiet(0.3), tone(0.4), quiet(0.3, seed=1)]))
    assert r.status == "empty" and r.meta["reason"] == "silence hallucination"
    r = record(s, np.concatenate([quiet(0.3), tone(1.2), quiet(0.3, seed=1)]))
    assert r.status == "ok" and sink.items == ["Ship the build tonight. "]


def test_push_to_talk_strips_echo_from_real_text():
    s, _asr, sink = dictation(["Dictating into Slack. Ship the build tonight."])
    r = record(s, np.concatenate([quiet(0.3), tone(1.0), quiet(0.3, seed=1)]))
    assert r.status == "ok" and sink.items == ["Ship the build tonight. "]


def ambient(outputs, commands=None):
    asr = FakeASR(outputs)
    sink = ListSink()
    ran = []
    bus = EventBus()

    class Clean:
        def clean(self, text, context=None):
            return text.strip()

    s = AmbientSession(
        asr,
        Clean(),
        sink,
        StreamingEndpointer(SR, end_silence_ms=400),
        commands=lambda: commands if commands is not None else BindingsConfig().voice_phrases(),
        run_command=lambda act, phrase: ran.append((str(act), phrase)),
        bus=bus,
        gate=lambda t0, t1: True,
        context=lambda: "Slack",
        vocabulary=list(VOCAB),
    )
    return s, asr, sink, ran, bus


def feed(s, audio):
    return [r for i in range(0, len(audio), 480) for r in s.feed(audio[i : i + 480])]


def test_always_on_never_types_prompt_echo():
    s, asr, sink, _ran, bus = ambient(["Dictating into Slack."])
    res = feed(s, np.concatenate([quiet(0.5), tone(0.6), quiet(0.6, seed=9)]))
    assert [r.decision.kind for r in res] == ["ignore"]
    assert res[0].decision.reason == "dropped: prompt echo"
    assert sink.items == []
    assert asr.calls[0][1] == PROMPT
    assert bus.history("voice.utterance")[-1].data["reason"] == "dropped: prompt echo"


def test_always_on_drops_vocab_list_and_short_thanks():
    s, _asr, sink, _ran, _bus = ambient(["Calabanga, Inan.", "Thank you."])
    audio = np.concatenate([quiet(0.5), tone(0.6), quiet(0.8, seed=9), tone(0.5), quiet(0.8, seed=3)])
    res = feed(s, audio)
    assert [r.decision.reason for r in res] == ["dropped: vocabulary list echo", "dropped: silence hallucination"]
    assert sink.items == []


def test_always_on_bound_phrase_is_never_treated_as_hallucination():
    cmds = {"thank you": Action("key", "enter")}
    s, _asr, _sink, ran, _bus = ambient(["Thank you."], commands=cmds)
    res = feed(s, np.concatenate([quiet(0.5), tone(0.5), quiet(0.6, seed=9)]))
    assert [r.decision.kind for r in res] == ["command"]
    assert ran == [("key:enter", "thank you")]


def test_always_on_types_real_speech_with_echo_removed():
    s, _asr, sink, _ran, _bus = ambient(["Vocabulary: Inan, GitHub. Hello Inan, the build is green."])
    feed(s, np.concatenate([quiet(0.5), tone(1.0), quiet(0.6, seed=9)]))
    assert sink.items == ["Hello Inan, the build is green. "]


def test_edit_instruction_ignores_silence_hallucination():
    class Clip:
        def __init__(self):
            self.v = "some text"

        def get(self):
            return self.v

        def set(self, v):
            self.v = v

    class LLM:  # would rewrite the selection for any instruction it gets
        configured = True

        def __init__(self):
            self.calls = []

        def edit(self, instruction, selection):
            self.calls.append(instruction)
            return "REWRITTEN"

    clip, llm = Clip(), LLM()
    e = SelectionEditor(
        FakeASR(["Thank you."]),
        clip,
        copy_keys=lambda: clip.set("some text"),
        paste_keys=lambda: None,
        llm=llm,
        sleep=lambda s: None,
    )
    assert e.start()
    e.feed(tone(0.5))
    new, method = e.stop()
    assert method == "unchanged" and llm.calls == []
