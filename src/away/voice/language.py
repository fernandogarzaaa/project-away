"""Speech language (v0.1.27): a plain choice in Settings > Voice & input, and a safer "Auto".

With ``voice.language`` unset (Auto) Whisper guesses the language of every utterance on its own, and a short or
unclear one is easily guessed wrong: English came out as Malay ("Bersihkanlah anda lihat skrin saya?"). Auto now
keeps a short memory of the languages it was confident about; a guess that is unsure, or is a language you have
not been speaking, is transcribed again in your most-used language (English until it knows better).

Pure logic, unit-tested without a model.
"""

from __future__ import annotations

import threading
from collections import Counter, deque

# Settings > Voice & input > Speech language: (code, plain label). "" = Auto.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("", "Auto (detect, falls back to the language you use most)"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("tl", "Filipino (Tagalog)"),
    ("ms", "Malay"),
    ("id", "Indonesian"),
    ("hi", "Hindi"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("zh", "Chinese"),
    ("ar", "Arabic"),
    ("ru", "Russian"),
    ("tr", "Turkish"),
    ("pl", "Polish"),
    ("uk", "Ukrainian"),
    ("vi", "Vietnamese"),
)
CODES: tuple[str, ...] = tuple(c for c, _ in LANGUAGES if c)
AGENT_CHOICES: tuple[str, ...] = ("auto", *CODES)  # what an agent may set (Agent Link, with confirmation)

MIN_PROBABILITY = 0.75  # below this, Whisper's guess is not trusted
CONFIDENT = 0.9  # a guess this sure counts towards "a language you speak"
LEARN_AFTER = 3  # a new language joins your languages after this many confident guesses
DEFAULT = "en"


def label_for(code: str | None) -> str:
    c = normalize(code) or ""
    return next((label for k, label in LANGUAGES if k == c), c)


def code_for_label(label: str) -> str | None:
    """The code for a Settings label (or a code typed directly); None = Auto. Raises ValueError when unknown."""
    t = str(label or "").strip()
    for k, lab in LANGUAGES:
        if t.lower() in (lab.lower(), k):
            return k or None
    if t.lower() in ("auto", "automatic", ""):
        return None
    raise ValueError(f"unknown speech language {label!r}")


def normalize(code: str | None) -> str | None:
    c = str(code or "").strip().lower()
    return None if c in ("", "auto", "none") else c


class LanguageMemory:
    """Which languages you speak, learned from Whisper's confident guesses (in memory only; seeded with English).

    :meth:`fallback` answers: given Whisper's guess and how sure it was, which language should the utterance be
    transcribed in instead (None = keep the guess)."""

    def __init__(self, seed: str = DEFAULT, size: int = 50) -> None:
        self._lock = threading.Lock()
        self._recent: deque[str] = deque(maxlen=size)
        self._candidates: Counter[str] = Counter()
        self.seed = seed
        if seed:
            self._recent.append(seed)

    def languages(self) -> list[str]:
        with self._lock:
            return [c for c, _ in Counter(self._recent).most_common()]

    def most_used(self) -> str:
        langs = self.languages()
        return langs[0] if langs else (self.seed or DEFAULT)

    def note(self, code: str) -> None:
        """An utterance was transcribed in ``code`` (kept or fallen back to)."""
        if code:
            with self._lock:
                self._recent.append(code)

    def fallback(self, detected: str | None, probability: float | None) -> str | None:
        d = normalize(detected)
        p = float(probability) if probability is not None else 0.0
        known = set(self.languages())
        if d is not None and d in known and p >= MIN_PROBABILITY:
            self.note(d)
            return None
        if d is not None and p >= CONFIDENT:
            with self._lock:
                self._candidates[d] += 1
                learned = self._candidates[d] >= LEARN_AFTER
            if learned:  # you really are speaking it now
                self.note(d)
                return None
        target = self.most_used()
        if d == target:
            return None
        return target
