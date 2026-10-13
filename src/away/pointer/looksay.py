"""Look + say: "click <name>", "click that", "double click <name>", "focus <name>" and numbered choices.

Pure logic (tested headless): :func:`parse` turns a spoken phrase into a :class:`Command`, :func:`match_names`
fuzzy-matches a name against the foreground window's targets, :func:`nearest_to` picks the element nearest the
gaze point (or the cursor), and :class:`Chooser` holds a numbered list while the user says which one.
:class:`away.pointer.smart.SmartPointer` runs them.
"""

from __future__ import annotations

import difflib
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass

from away.pointer.targets import Snapshot, Target

ACTIONS = ("click", "double_click", "focus")
THAT_WORDS = frozenset({"that", "this", "here", "it", "there", "that one", "this one"})
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "to": 2,  # what a recogniser often writes for a bare "two"
    "too": 2,
    "three": 3,
    "four": 4,
    "for": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "first": 1,
    "second": 2,
    "third": 3,
}
CANCEL_WORDS = frozenset({"cancel", "never mind", "nevermind", "stop", "none", "no"})
MAX_CHOICES = 9
MATCH_THRESHOLD = 0.62  # below this a name does not count as heard
TIE_MARGIN = 0.06  # candidates this close to the best score are ambiguous
LOOK_RADIUS_PX = 160.0  # "click that": the element must be this close to the gaze point / cursor
_PUNCT = {ord(c): " " for c in string.punctuation}


def norm(text: str) -> str:
    """Lower case, punctuation to spaces, single spaces (``"Save as..."`` -> ``"save as"``)."""
    return " ".join(str(text).lower().translate(_PUNCT).split())


@dataclass(frozen=True)
class Command:
    action: str  # click | double_click | focus | choose | cancel
    name: str = ""  # the spoken element name ("" for "that")
    that: bool = False
    number: int = 0  # choose


_PREFIXES = (
    ("double click ", "double_click"),
    ("click ", "click"),
    ("focus ", "focus"),
)
# "click ..." is never ordinary dictation, so it is always taken as a command (and says so when nothing matches);
# "focus ..." also starts normal sentences ("focus on the budget"), so it only counts when something matches
ALWAYS_COMMAND = frozenset({"click", "double_click"})
_FILLER = re.compile(r"^(?:on |the |at |in |into )+")


def parse_number(phrase: str) -> int:
    """``"2"``, ``"two"``, ``"number two"``, ``"pick 3"``, ``"the second one"`` -> the number (0 when not one)."""
    p = norm(phrase)
    p = re.sub(r"^(?:number|choose|pick|select|option|click|the)\s+", "", p)
    if p != "one":
        p = re.sub(r"\s+(?:one|please)$", "", p)
    if p.isdigit():
        n = int(p)
        return n if 1 <= n <= MAX_CHOICES else 0
    return NUMBER_WORDS.get(p, 0)


def parse(phrase: str, choosing: bool = False) -> Command | None:
    """A spoken phrase -> :class:`Command`, or None when it is not a smart pointer command.

    ``choosing``: a numbered list is on screen, so a bare number ("two") or "cancel" counts too.
    """
    p = norm(phrase)
    if not p:
        return None
    if choosing:
        if p in CANCEL_WORDS:
            return Command("cancel")
        n = parse_number(p)
        if n:
            return Command("choose", number=n)
    for prefix, action in _PREFIXES:
        if p.startswith(prefix):
            rest = _FILLER.sub("", p[len(prefix) :] + " ").strip()
            if not rest:
                return None
            if rest in THAT_WORDS:
                return Command(action, that=True)
            return Command(action, name=rest)
    return None


def name_score(query: str, name: str) -> float:
    """How well a spoken ``query`` matches an element ``name`` (0..1): exact 1.0, a whole-word prefix 0.92,
    every spoken word present 0.85, else the best of a character ratio over the whole name and over each run of
    name words as long as the query (so "send" matches "Send message" but "sand" only weakly)."""
    q, n = norm(query), norm(name)
    if not q or not n:
        return 0.0
    if q == n:
        return 1.0
    nw, qw = n.split(), q.split()
    if nw[: len(qw)] == qw:
        return 0.92
    if all(w in nw for w in qw):
        return 0.85
    best = difflib.SequenceMatcher(None, q, n).ratio()
    k = len(qw)
    for i in range(max(1, len(nw) - k + 1)):
        part = " ".join(nw[i : i + k])
        best = max(best, difflib.SequenceMatcher(None, q, part).ratio() * 0.95)
    return best


@dataclass(frozen=True)
class Match:
    target: Target
    score: float


def _same_spot(a: Target, b: Target, tol: float = 3.0) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a.rect, b.rect))


def match_names(query: str, targets: Sequence[Target], threshold: float = MATCH_THRESHOLD) -> list[Match]:
    """The targets that match ``query`` well enough, best first, *only* those tied with the best.

    One result: act on it. Several: they are equally plausible (two "OK" buttons), so the caller shows numbered
    badges. Empty: nothing on screen is called that.
    """
    scored = [Match(t, name_score(query, t.name)) for t in targets if t.name]
    scored = [m for m in scored if m.score >= threshold]
    if not scored:
        return []
    scored.sort(key=lambda m: (-m.score, m.target.rect[1], m.target.rect[0]))
    best = scored[0].score
    tied = [m for m in scored if m.score >= best - TIE_MARGIN]
    if best >= 1.0:
        tied = [m for m in tied if m.score >= 1.0]  # an exact name beats near misses
    out: list[Match] = []
    for m in tied:  # the same element reported twice (a button and its inner text) collapses to one
        if any(_same_spot(m.target, o.target) for o in out):
            continue
        out.append(m)
    return out[:MAX_CHOICES]


def nearest_to(snapshot: Snapshot, point: tuple[float, float], radius: float = LOOK_RADIUS_PX) -> Target | None:
    """The target nearest a point (distance to its rectangle; 0 inside, the smallest one wins a tie)."""
    return snapshot.nearest(point[0], point[1], radius)


@dataclass
class PendingChoice:
    action: str
    targets: list[Target]
    expires: float


class Chooser:
    """A numbered list on screen ("1", "2", ...) waiting for the user to say a number or "cancel"."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        self.timeout_s = timeout_s
        self.pending: PendingChoice | None = None

    def offer(self, action: str, targets: Sequence[Target], now: float) -> PendingChoice:
        self.pending = PendingChoice(action, list(targets)[:MAX_CHOICES], now + self.timeout_s)
        return self.pending

    def active(self, now: float) -> bool:
        if self.pending is not None and now > self.pending.expires:
            self.pending = None
        return self.pending is not None

    def choose(self, number: int, now: float) -> tuple[str, Target] | None:
        if not self.active(now) or self.pending is None:
            return None
        if not 1 <= number <= len(self.pending.targets):
            return None
        p, self.pending = self.pending, None
        return p.action, p.targets[number - 1]

    def cancel(self) -> None:
        self.pending = None
