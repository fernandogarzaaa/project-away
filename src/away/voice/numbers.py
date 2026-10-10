"""Spoken numbers -> digits for dictation ("twenty five" -> "25", "three point five" -> "3.5").

Original implementation (the option set is inspired by nerd-dictation's
``--numbers-as-digits`` / ``--numbers-min-value``; no code from it is used).
Numbers below ``min_value`` stay as words so "one of them" is not turned into "1 of them".
"""

from __future__ import annotations

import re

_UNIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
UNITS = {w: i for i, w in enumerate(_UNIT_WORDS)}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES) | {"hundred"}
_TOKEN = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\S|\s+")


def _kind(w: str) -> str:
    if w in TENS:
        return "tens"
    if w in UNITS:
        return "teen" if UNITS[w] >= 10 else "unit"
    if w == "hundred":
        return "hundred"
    return "scale"


def _can_follow(prev: str | None, cur: str) -> bool:
    """Whether ``cur`` continues the same number after ``prev`` ("twenty five" yes, "five six" no)."""
    if prev is None:
        return True
    if cur == "unit":
        return prev in ("tens", "hundred", "scale")
    if cur in ("teen", "tens"):
        return prev in ("hundred", "scale")
    if cur == "hundred":
        return prev in ("unit", "teen")
    return prev in ("unit", "teen", "tens", "hundred")  # thousand/million after a quantity


def _value(words: list[str]) -> int:
    total, current = 0, 0
    for w in words:
        if w in UNITS:
            current += UNITS[w]
        elif w in TENS:
            current += TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        else:
            total += max(current, 1) * SCALES[w]
            current = 0
    return total + current


def words_to_digits(text: str, min_value: int = 10) -> str:
    toks = _TOKEN.findall(text)
    words = []  # (token index, lowercase word) for word tokens
    for i, tk in enumerate(toks):
        if tk[0].isalpha():
            words.append((i, tk.lower()))
    out = list(toks)
    k = 0
    while k < len(words):
        i0, w0 = words[k]
        parts0 = w0.split("-")
        if not all(p in NUMBER_WORDS for p in parts0) or parts0[0] in SCALES or parts0[0] == "hundred":
            k += 1
            continue
        run: list[str] = []
        kinds: list[str] = []
        j = k
        last_tok = i0
        while j < len(words):
            idx, w = words[j]
            if j > k and any(not toks[m].isspace() for m in range(last_tok + 1, idx)):
                break  # punctuation between words ends the number
            if w == "and" and run and j + 1 < len(words) and kinds[-1] in ("hundred", "scale"):
                nxt = words[j + 1][1].split("-")[0]
                if nxt in UNITS or nxt in TENS:
                    j += 1
                    last_tok = idx
                    continue
            parts = w.split("-")
            if not all(p in NUMBER_WORDS for p in parts):
                break
            ok = True
            for p in parts:
                kd = _kind(p)
                if not _can_follow(kinds[-1] if kinds else None, kd):
                    ok = False
                    break
                kinds.append(kd)
                run.append(p)
            if not ok:
                # keep what parsed of a hyphenated word only if it was the first part
                break
            last_tok = idx
            j += 1
        if not run:
            k += 1
            continue
        value = _value(run)
        end_tok = last_tok
        decimal = ""
        if j + 1 < len(words) and words[j][1] == "point" and words[j + 1][1] in UNITS and UNITS[words[j + 1][1]] < 10:
            m = j + 1
            while m < len(words) and words[m][1] in UNITS and UNITS[words[m][1]] < 10:
                decimal += str(UNITS[words[m][1]])
                end_tok = words[m][0]
                m += 1
            j = m
        if value >= min_value or decimal:
            out[i0] = str(value) + (f".{decimal}" if decimal else "")
            for m in range(i0 + 1, end_tok + 1):
                out[m] = ""
        k = max(j, k + 1)
    return "".join(out)
