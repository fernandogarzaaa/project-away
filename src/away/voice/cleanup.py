"""Second-pass transcript cleanup.

:class:`RuleCleaner` is deterministic and always available:

* spoken self-corrections: "scratch that" drops the current sentence;
  "no wait" / "no, wait" / "actually no" replace the phrase they correct;
* filler removal (um, uh, er, hmm ...) and stutter de-duplication ("the the");
* spoken punctuation ("comma", "question mark", "new line" ...);
* custom vocabulary (``wrong -> right`` lines or plain terms for casing);
* sentence capitalisation, standalone "i" -> "I", terminal period.

:class:`LLMCleaner` optionally polishes the rule output with any
OpenAI-compatible ``/chat/completions`` endpoint (Ollama, LM Studio, vLLM,
llama.cpp server, a cloud API...) and silently falls back to the rule output
on any error or suspicious answer.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

FILLERS = ("um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mhm")
_FILLER_RE = re.compile(r"(?<![\w'])(?:" + "|".join(FILLERS) + r")(?![\w'])[,.]?\s*", re.IGNORECASE)
_STUTTER_RE = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)

SPOKEN_PUNCT = [
    (r"\s*\bnew paragraph\b\s*", "\n\n"),
    (r"\s*\bnew line\b\s*", "\n"),
    (r"\s*\bquestion mark\b", "?"),
    (r"\s*\bexclamation (?:mark|point)\b", "!"),
    (r"\s*\b(?:full stop|period)\b", "."),
    (r"\s*\bcomma\b", ","),
    (r"\s*\bsemicolon\b", ";"),
    (r"\s*\bcolon\b", ":"),
]

SCRATCH_RE = re.compile(r"\bscratch that\b[,.!?]?", re.IGNORECASE)
CORRECTION_RE = re.compile(r"[,.]?\s*\b(?:no,? wait|actually,? no)\b[,.]?\s*", re.IGNORECASE)
# a correction marker at the very end of the text before "scratch that" is a
# command, not part of the sentence being scratched
_TRAILING_CORRECTION_RE = re.compile(r"[,.]?\s*\b(?:no,? wait|actually,? no)\b[,.!?]?\s*$", re.IGNORECASE)
_SENT_END = re.compile(r"[.!?\n]")
_NUM = re.compile(r"^\d+(?:[.:]\d+)?$")
# "no wait, make it Tuesday": the lead-in says "replace", the rest is the replacement
_CORR_LEAD = re.compile(r"^(?:make (?:it|that)|change (?:it|that) to|i mean|rather|instead)\s+", re.IGNORECASE)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip
_TIME = re.compile(r"^(?:\d{1,2}(?::\d\d)?(?:am|pm)?|noon|midnight)$", re.IGNORECASE)


def remove_fillers(text: str) -> str:
    text = _FILLER_RE.sub("", text)
    return _STUTTER_RE.sub(r"\1", text)


def apply_scratch_that(text: str) -> str:
    """Delete the sentence in progress before each "scratch that"."""
    while True:
        m = SCRATCH_RE.search(text)
        if not m:
            return text
        before = text[: m.start()].rstrip()
        # "no wait" / "actually no" are correction commands, not sentence text:
        # strip a trailing one so "X. no wait scratch that Y" deletes X (the
        # sentence the user wants gone), not just the marker
        while True:
            stripped = _TRAILING_CORRECTION_RE.sub("", before).rstrip()
            if stripped == before:
                break
            before = stripped
        # strip a trailing terminator that belongs to the scratched clause
        core = before.rstrip(" ,")
        ends = [e.end() for e in _SENT_END.finditer(core[:-1] if core.endswith((".", "!", "?")) else core)]
        keep = core[: ends[-1]] if ends else ""
        text = (keep + " " + text[m.end() :].lstrip()).strip()


def _same_class(a: str, b: str, clock: bool = False) -> bool:
    """Both weekdays, both months, or both clock times ("Monday"/"Tuesday", "3pm"/"4:30"). A bare number only
    counts as a time when the correction says am/pm (``clock``); otherwise numbers keep their own rule."""
    if any(a in group and b in group for group in (_WEEKDAYS, _MONTHS)):
        return True
    if not (_TIME.match(a) and _TIME.match(b)):
        return False
    return clock or not (a.isdigit() or b.isdigit())


def _correct_once(before: str, after: str) -> str:
    words = before.split()
    after = _CORR_LEAD.sub("", after.lstrip(), count=1)
    corr = re.split(r"[.,!?\n]", after, maxsplit=1)[0].split()
    if not corr or not words:
        return (before + " " + after).strip()
    first = corr[0].lower().strip(",.")
    cut = None
    # 0) a weekday / month / time correction replaces the last one of its kind: a lone one keeps the rest
    #    ("on Monday at 3, no wait, Tuesday" -> "on Tuesday at 3"), a longer one restarts there
    ampm = len(corr) == 2 and corr[1].lower().strip(",.") in ("am", "pm")
    said = any(w.lower().strip(",.") == first for w in words)  # "30 then 40 no wait 30 again": rule 1 anchors
    for i in range(len(words) - 1, -1, -1) if not said else range(0):
        if _same_class(words[i].lower().strip(",."), first, ampm):
            if len(corr) == 1 or ampm:
                clause = re.match(r"([^.,!?\n]*)(.*)", after, re.S)
                rest = clause.group(2) if clause else ""
                j = i + 1
                if ampm and j < len(words) and words[j].lower().strip(",.") in ("am", "pm"):
                    j += 1  # "3 pm" is replaced as a whole
                src = words[j - 1]
                end = src[len(src.rstrip(",.")) :]  # keep the punctuation that ended the replaced words
                fixed = " ".join([*words[:i], " ".join(corr) + end, *words[j:]])
                return (fixed + rest).strip()
            cut = i
            break
    # 1) correction starts with a word we already said -> restart from its last occurrence
    for i in range(len(words) - 1, -1, -1) if cut is None else range(0):
        if words[i].lower().strip(",.") == first:
            cut = i
            break
    # 2) numbers replace the last number
    if cut is None and _NUM.match(first):
        for i in range(len(words) - 1, -1, -1):
            if _NUM.match(words[i].strip(",.")):
                cut = i
                break
    # 3) otherwise replace as many words as the correction has (max 3)
    if cut is None:
        cut = max(0, len(words) - min(len(corr), 3))
    return (" ".join(words[:cut]) + " " + after).strip()


def apply_corrections(text: str) -> str:
    while True:
        m = CORRECTION_RE.search(text)
        if not m:
            return text
        text = _correct_once(text[: m.start()], text[m.end() :])


def apply_spoken_punctuation(text: str) -> str:
    for pat, rep in SPOKEN_PUNCT:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


@dataclass
class Vocabulary:
    replacements: list[tuple[re.Pattern[str], str]] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> Vocabulary:
        reps: list[tuple[re.Pattern[str], str]] = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "->" in line:
                wrong, right = (s.strip() for s in line.split("->", 1))
            else:
                wrong = right = line
            if wrong:
                reps.append((re.compile(r"(?<!\w)" + re.escape(wrong) + r"(?!\w)", re.IGNORECASE), right))
        return cls(reps)

    @classmethod
    def load(cls, path: str | Path | None) -> Vocabulary:
        if not path or not Path(path).exists():
            return cls()
        return cls.parse(Path(path).read_text(encoding="utf-8"))

    def apply(self, text: str) -> str:
        for pat, rep in self.replacements:
            text = pat.sub(lambda _m, r=rep: r, text)  # literal: "C:\\Users" is not a regex template
        return text

    def terms(self) -> list[str]:
        return [rep for _, rep in self.replacements]

    def hotwords(self, limit: int = 60) -> str | None:
        """Distinct target terms as one space-separated string for faster-whisper ``hotwords``."""
        seen = list(dict.fromkeys(t for t in self.terms() if t))
        return " ".join(seen[:limit]) or None


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.!?;:])", r"\1", text)
    text = re.sub(r"([,;:])(?=\w)", r"\1 ", text)
    text = re.sub(r"([,.!?;:])\1+", r"\1", text)
    text = re.sub(r"\s*\n\s*", lambda m: "\n\n" if m.group(0).count("\n") > 1 else "\n", text)
    text = re.sub(r"^[,.;:\s]+", "", text)
    return text.strip()


_PROPER_DATES = re.compile(
    r"(?<![\w'])(" + "|".join(d for d in (*_WEEKDAYS, *_MONTHS) if d not in ("may", "march")) + r")(?![\w'])"
)


def capitalize(text: str) -> str:
    text = re.sub(r"(?<![\w'])i(?=(?:'(?:m|ve|ll|d))?(?![\w']))", "I", text)
    text = _PROPER_DATES.sub(lambda m: m.group(1).capitalize(), text)  # "tuesday" -> "Tuesday" (not may/march)
    return re.sub(r"(^|[.!?]\s+|\n\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def ensure_terminal(text: str, min_words: int = 3) -> str:
    if text and len(text.split()) >= min_words and text[-1] not in ".!?:;\n":
        return text + "."
    return text


class RuleCleaner:
    def __init__(
        self,
        vocabulary: Vocabulary | None = None,
        terminal_period: bool = True,
        numbers_as_digits: bool = False,
        numbers_min_value: int = 10,
    ) -> None:
        self.vocab = vocabulary or Vocabulary()
        self.terminal_period = terminal_period
        self.numbers_as_digits = numbers_as_digits
        self.numbers_min_value = numbers_min_value

    def clean(self, text: str, context: str | None = None) -> str:
        t = " ".join(text.split())
        t = remove_fillers(t)
        t = apply_spoken_punctuation(t)
        t = apply_scratch_that(t)
        t = apply_corrections(t)
        if self.numbers_as_digits:
            from away.voice.numbers import words_to_digits

            t = words_to_digits(t, self.numbers_min_value)
        t = self.vocab.apply(t)
        t = tidy(t)
        if _is_code_context(context):  # shells: no sentence casing, no trailing period
            return t
        t = capitalize(t)
        if self.terminal_period:
            t = ensure_terminal(t)
        return t


def _is_code_context(context: str | None) -> bool:
    if not context:
        return False
    c = context.lower()
    return any(k in c for k in ("terminal", "powershell", "cmd.exe", "iterm", "konsole", "bash", "zsh"))


SYSTEM_PROMPT = (
    "You clean up dictated text. Fix punctuation, capitalization and obvious transcription errors. "
    "Apply spoken self-corrections. Keep the user's wording and language; do not add content, "
    "do not answer questions in the text. Output only the cleaned text."
)


class LLMCleaner:
    """Optional polish via an OpenAI-compatible endpoint configured by env vars:

    ``AWAY_LLM_BASE_URL`` (e.g. http://localhost:11434/v1), ``AWAY_LLM_MODEL``,
    ``AWAY_LLM_API_KEY`` (optional). Nothing is sent anywhere unless the base
    URL is set *and* ``voice.llm_cleanup`` is true.
    """

    def __init__(
        self,
        rules: RuleCleaner,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> None:
        self.rules = rules
        self.base_url = (base_url or os.environ.get("AWAY_LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("AWAY_LLM_MODEL") or "llama3.2"
        self.api_key = api_key or os.environ.get("AWAY_LLM_API_KEY")
        self.timeout = timeout
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _request(self, text: str, context: str | None) -> str:
        user = text if not context else f"[active app: {context}]\n{text}"
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            }
        ).encode()
        from away.voice.llm_backends import user_agent

        headers = {"Content-Type": "application/json", "User-Agent": user_agent()}  # v0.1.23: Groq needs one
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return str(data["choices"][0]["message"]["content"]).strip()

    def clean(self, text: str, context: str | None = None) -> str:
        base = self.rules.clean(text, context)
        if not self.configured or not base:
            return base
        try:
            out = self._request(base, context)
        except Exception as exc:  # network, HTTP, JSON shape: never lose the dictation
            self.last_error = f"{type(exc).__name__}: {exc}"
            return base
        if not out or not (0.5 <= len(out) / max(len(base), 1) <= 1.6):
            self.last_error = "rejected LLM output (empty or length changed too much)"
            return base
        self.last_error = None
        return out
