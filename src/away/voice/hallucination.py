"""Drop Whisper hallucinations before they are typed.

Whisper invents text on short, quiet or noisy audio. Three kinds matter for dictation:

* **Prompt echo.** On near-silence it repeats its ``initial_prompt`` ("Vocabulary: Inan, GitHub."; before
  v0.1.6 also "Dictating into Slack."), or a run of the vocabulary list. :func:`clean_transcript` strips
  those sentences and drops a transcript that is substantially the prompt or a bare list of vocabulary terms.
* **Silent segments.** A segment with a high ``no_speech_prob`` *and* a low ``avg_logprob`` is Whisper
  guessing over silence (the same rule faster-whisper uses for its own silence skip): :func:`segment_ok`.
* **Stock phrases.** "Thank you.", "Thanks for watching!", "Please subscribe" etc. come from subtitled
  training video. They are dropped only when the utterance is very short (a real "thank you" at the end of
  a longer sentence is kept); subtitle credits ("Subtitles by the Amara.org community") always are.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence
from typing import Any

NO_SPEECH_PROB = 0.6
LOW_AVG_LOGPROB = -1.0
SHORT_UTTERANCE_S = 2.0  # stock phrases are only treated as hallucinations below this length
PROMPT_SIMILARITY = 0.8

# sentences that only ever come from our own prompt (current and pre-0.1.6 wording)
# (an app title or a term may contain dots, "main.py - Code", so a sentence ends at [.!?] + space/end)
PROMPT_SENTENCE = re.compile(
    r"(?:^|(?<=[.!?]))\s*(?:vocabulary\s*:|dictating\s+into\b).*?(?:[.!?](?=\s|$)|$)", re.I | re.S
)

# Whisper's stock outputs on silence/noise (normalized: lowercase, no punctuation)
SILENCE_PHRASES = frozenset(
    {
        "thank you",
        "thank you very much",
        "thank you so much",
        "thanks",
        "thanks for watching",
        "thank you for watching",
        "thanks for watching and see you next time",
        "thank you for watching and see you next time",
        "please subscribe",
        "please like and subscribe",
        "like and subscribe",
        "subscribe to my channel",
        "dont forget to like and subscribe",
        "see you next time",
        "see you in the next video",
        "bye",
        "bye bye",
        "you",
        "music",
        "applause",
        "silence",
    }
)
# subtitle credits: hallucinations at any length
CREDIT_PATTERNS = (
    re.compile(r"subtitles? (?:by|made by|created by)", re.I),
    re.compile(r"amara\.org", re.I),
    re.compile(r"transcribed by", re.I),
    re.compile(r"translated by", re.I),
)


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'").replace("'", "")
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def segment_ok(seg: Any, no_speech_prob: float = NO_SPEECH_PROB, low_logprob: float = LOW_AVG_LOGPROB) -> bool:
    """False for a segment Whisper produced over silence: ``no_speech_prob`` high AND ``avg_logprob`` low.

    Segments without those attributes (other backends, fakes) are kept."""
    nsp = getattr(seg, "no_speech_prob", None)
    alp = getattr(seg, "avg_logprob", None)
    if nsp is None or alp is None:
        return True
    try:
        return not (float(nsp) > no_speech_prob and float(alp) < low_logprob)
    except (TypeError, ValueError):
        return True


def join_segments(segments: Iterable[Any]) -> str:
    """Text of the segments that are not silence hallucinations."""
    return " ".join(str(s.text).strip() for s in segments if segment_ok(s)).strip()


def is_silence_phrase(text: str, seconds: float | None, short_s: float = SHORT_UTTERANCE_S) -> bool:
    if any(p.search(text) for p in CREDIT_PATTERNS):
        return True
    if seconds is not None and seconds >= short_s:
        return False
    n = normalize(text)
    if n in SILENCE_PHRASES:
        return True
    # "Thank you. Thank you." / "Bye. Bye. Bye."
    parts = [normalize(p) for p in re.split(r"[.!?,]+", text) if normalize(p)]
    return bool(parts) and all(p in SILENCE_PHRASES for p in parts)


def _vocab_set(vocabulary: Sequence[str]) -> set[str]:
    return {normalize(v) for v in vocabulary if normalize(v)}


def is_vocabulary_list(text: str, vocabulary: Sequence[str]) -> bool:
    """True when the text is just two or more vocabulary terms in a list ("Inan, GitHub, Ruff.")."""
    vocab = _vocab_set(vocabulary)
    if not vocab:
        return False
    parts = [normalize(p) for p in re.split(r"[,;.\n]+", text) if normalize(p)]
    if len(parts) < 2:
        # space-separated run of terms with no other words: "Inan GitHub Ruff"
        words = normalize(text).split()
        return len(words) >= 3 and all(w in vocab for w in words)
    return all(p in vocab for p in parts)


def strip_prompt_sentences(text: str) -> str:
    return " ".join(PROMPT_SENTENCE.sub(" ", text).split()).strip()


def clean_transcript(
    text: str,
    prompt: str | None = None,
    vocabulary: Sequence[str] = (),
    seconds: float | None = None,
) -> tuple[str, str]:
    """``(text, reason)``: the transcript with prompt echoes removed, or ``("", why)`` when nothing real is
    left. ``reason`` is "" when the text was not changed."""
    original = text.strip()
    if not original:
        return "", ""
    if prompt:
        a, b = normalize(original), normalize(prompt)
        if a and b and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= PROMPT_SIMILARITY):
            return "", "prompt echo"
    out = strip_prompt_sentences(original)
    reason = "prompt echo stripped" if out != original else ""
    if not normalize(out):
        return "", "prompt echo"
    if prompt:
        # what is left is a fragment of the prompt (e.g. the tail of the vocabulary list)
        a, b = normalize(out), normalize(prompt)
        if len(a.split()) >= 3 and a in b:
            return "", "prompt echo"
    if is_vocabulary_list(out, vocabulary):
        return "", "vocabulary list echo"
    if is_silence_phrase(out, seconds):
        return "", "silence hallucination"
    return out, reason
