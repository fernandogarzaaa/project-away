"""Talk to your agent (v0.1.22): say "Hey <agent name>, ..." (or use the summon gesture, then speak) and the words
go to that AI agent's inbox instead of being typed.

* Every connected agent answers to "Hey <its name>" automatically ("Hey Claude", "Hey Hark"; for a longer name
  such as "Claude Desktop" the first word works too). The person can give an agent its own wake word in
  Settings > Agent Link ("Jarvis" also works as "Hey Jarvis").
* "Hey agent" and the summon gesture go to the person's default agent (Settings > Agent Link); with no default
  they go to every connected agent.
* "Hey Claude" on its own (nothing after it) waits ``window_s`` seconds for the message; so does the summon gesture.

This module only recognises the phrase and keeps the "waiting for your message" state; Agent Link stores the
message and tells the agent (:mod:`away.agentlink.inbox`). Pure logic, unit-tested without audio.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from away.bindings import normalize_phrase

LEADS = ("hey", "hi", "ok", "okay", "hello")
GENERIC = ("agent", "my agent", "assistant", "ai")  # "Hey agent" -> the default agent (or everyone)
WINDOW_S = 8.0
DEFAULT = ""  # target id of the generic route ("Hey agent", the summon gesture)


@dataclass(frozen=True)
class Route:
    target: str  # an agent id, or DEFAULT ("" = the default agent / every connected agent)
    body: str  # the message ("" = only the wake phrase: wait for the message)
    phrase: str  # what was matched ("hey claude")


def _tokens(text: str) -> list[str]:
    return [t for t in (normalize_phrase(w).replace(" ", "") for w in str(text).split()) if t]


def phrases_for(name: str, wake_word: str = "") -> list[tuple[str, bool]]:
    """The phrases an agent answers to, as ``(phrase, lead required)``: its custom wake word (with or without
    "hey"), then "hey <full name>" and "hey <first word of the name>"."""
    out: list[tuple[str, bool]] = []
    custom = " ".join(_tokens(wake_word))
    if custom:
        out.append((custom, False))  # works with or without a "hey" in front
    toks = _tokens(name)
    if toks:
        out.append((" ".join(toks), True))
        if len(toks) > 1 and len(toks[0]) >= 3:
            out.append((toks[0], True))
    seen: set[str] = set()
    uniq = []
    for p, lead in out:
        if p not in seen:
            seen.add(p)
            uniq.append((p, lead))
    return uniq


def match(text: str, agents: Iterable[Mapping[str, Any]]) -> Route | None:
    """Which agent ``text`` is addressed to. ``agents``: ``{"id", "name", "wake_word"}`` dicts. The longest
    matching phrase wins (so "Hey Claude Desktop" beats "Hey Claude" when both are agents). None: not addressed
    to an agent (normal dictation)."""
    raw = str(text).split()
    toks = [normalize_phrase(w).replace(" ", "") for w in raw]
    i = 0
    while i < len(toks) and not toks[i]:
        i += 1
    lead = i < len(toks) and toks[i] in LEADS
    start = i + 1 if lead else i
    cands: list[tuple[str, bool, str]] = []
    for a in agents:
        for p, need_lead in phrases_for(str(a.get("name", "")), str(a.get("wake_word", "") or "")):
            cands.append((p, need_lead, str(a.get("id", ""))))
    for g in GENERIC:
        cands.append((g, True, DEFAULT))
    best: tuple[int, Route] | None = None
    for phrase, need_lead, target in cands:
        words = phrase.split()
        if words and words[0] in LEADS:  # a custom wake word that already says "hey ..."
            s, need_lead = i, False
        else:
            s = start
        if need_lead and not lead:
            continue
        end = _match_at(toks, s, words)
        if end is None:
            continue
        body = " ".join(raw[end:]).lstrip(" ,.:;!?-").strip()
        score = len("".join(words))
        if best is None or score > best[0]:
            label = (toks[i] + " " if lead and s == start else "") + phrase
            best = (score, Route(target, body, label))
    if best is None and lead and start < len(toks):
        return _sounds_like(toks, raw, i, start, cands)
    return best[1] if best else None


# v0.1.24: speech recognition often hears a short name as a common word ("Hey Hark" -> "hey heart", "Hey Claude"
# -> "hey cloud"). After "hey"/"ok"/... a word that sounds like exactly one agent's one-word name still reaches it.
_SOUND_CLASS = {
    **dict.fromkeys("bdgkpqtc", "P"),  # stops: t/k/d/g sound alike in a quick "hey ..."
    **dict.fromkeys("fv", "f"),
    **dict.fromkeys("szx", "s"),
    **dict.fromkeys("mn", "n"),
    "l": "l",
    "r": "r",
    "j": "j",
}


def sound_key(word: str) -> str:
    """A rough sound of ``word``: its first letter, then its consonants by sound class, vowels and h/w/y dropped,
    repeats merged ("hark" and "heart" -> "hrP", "claude" and "cloud" -> "PlP")."""
    w = "".join(ch for ch in str(word).lower() if "a" <= ch <= "z")
    if not w:
        return ""
    out = [_SOUND_CLASS.get(w[0], w[0])]
    for ch in w[1:]:
        c = _SOUND_CLASS.get(ch)
        if c and c != out[-1]:
            out.append(c)
    return "".join(out)


def _edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sounds_like(heard: str, name: str) -> bool:
    """``heard`` is a near-homophone of the one-word name ``name``: the same rough sound (and about as long), or,
    for names of 5+ letters, one letter off ("jemini" for "gemini")."""
    if heard == name:
        return True
    if len(name) >= 5 and _edits(heard, name) <= 1:
        return True
    key = sound_key(name)
    return len(name) >= 3 and len(key) >= 3 and abs(len(heard) - len(name)) <= 2 and sound_key(heard) == key


def _sounds_like(
    toks: list[str], raw: list[str], i: int, start: int, cands: list[tuple[str, bool, str]]
) -> Route | None:
    """The one agent whose one-word name the word after the lead sounds like (None when none, or more than one)."""
    heard = toks[start]
    hits = {
        target: phrase
        for phrase, _need_lead, target in cands
        if target != DEFAULT and " " not in phrase and phrase not in LEADS and sounds_like(heard, phrase)
    }
    if len(hits) != 1:
        return None
    target, phrase = next(iter(hits.items()))
    body = " ".join(raw[start + 1 :]).lstrip(" ,.:;!?-").strip()
    return Route(target, body, f"{toks[i]} {phrase}")


def _match_at(toks: list[str], s: int, words: list[str]) -> int | None:
    """The token index after ``words`` when they start at ``s`` (ASR may split or join words: "jar vis")."""
    want = "".join(words)
    joined = ""
    for k in range(s, min(len(toks), s + len(words) + 3)):
        joined += toks[k]
        if joined == want:
            return k + 1
        if not want.startswith(joined):
            return None
    return None


# v0.1.26 conversational mode: these end the conversation (the agent stops talking and stops listening)
STOP_PHRASES = (
    "stop",
    "stop it",
    "stop talking",
    "that's all",
    "thats all",
    "that is all",
    "that's it",
    "thats it",
    "thanks that's all",
    "thank you that's all",
    "no thanks",
    "never mind",
    "nevermind",
    "goodbye",
    "bye",
    "cancel",
    "be quiet",
    "quiet",
)


def is_stop(text: str) -> bool:
    """ "stop", "that's all", "okay, that's all. thanks" ... (the whole utterance, not a sentence that has "stop")."""
    t = " ".join(str(text).lower().replace("\u2019", "'").replace(",", " ").replace(".", " ").replace("!", " ").split())
    for lead in ("okay ", "ok ", "alright ", "no "):
        if t.startswith(lead) and t[len(lead) :] in STOP_PHRASES:
            return True
    for tail in (" thanks", " thank you"):
        if t.endswith(tail) and t[: -len(tail)] in STOP_PHRASES:
            return True
    return t in STOP_PHRASES


class TalkState:
    """The "waiting for your message" window after a bare wake phrase or the summon gesture.

    v0.1.26: ``window_s`` may be a getter (Settings > Voice & input "Keep listening", read at every arm), and
    ``hold`` (True while the agent is still answering or speaking) keeps the window open and restarts it from the
    moment the agent goes quiet, so a long reply never eats the time you have to answer."""

    def __init__(
        self,
        window_s: float | Callable[[], float] = WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        hold: Callable[[], bool] | None = None,
    ) -> None:
        self._window = window_s
        self.clock = clock
        self.hold = hold
        self._lock = threading.Lock()
        self._until = -1e9
        self._target = DEFAULT
        self._held = False
        self.source = "voice"

    @property
    def window_s(self) -> float:
        w = self._window
        try:
            v = float(w() if callable(w) else w)
        except Exception:
            v = WINDOW_S
        return max(0.5, min(60.0, v))

    @window_s.setter
    def window_s(self, value: float | Callable[[], float]) -> None:
        self._window = value

    def arm(self, target: str = DEFAULT, source: str = "voice", held: bool = False) -> None:
        """``held``: the window waits for :attr:`hold` to go False before it starts counting (conversational)."""
        with self._lock:
            self._until = self.clock() + self.window_s
            self._target = target
            self.source = source
            self._held = held

    def _holding(self) -> bool:
        if not self._held or self.hold is None:
            return False
        try:
            return bool(self.hold())
        except Exception:
            return False

    def armed(self) -> str | None:
        """The waiting target (DEFAULT or an agent id), or None when nothing is waiting."""
        holding = self._holding()
        with self._lock:
            if self._until <= -1e8:
                return None
            if holding:
                self._until = self.clock() + self.window_s  # counts from when the agent goes quiet
                return self._target
            return self._target if self.clock() <= self._until else None

    def disarm(self) -> None:
        with self._lock:
            self._until = -1e9
            self._held = False


class TalkHook:
    """What the voice input needs from Agent Link: the agents to listen for, where to deliver a message, and the
    waiting window. The app creates one; Agent Link fills in ``agents`` and ``deliver`` when it starts.

    v0.1.26: ``window_s`` (a getter: Settings "Keep listening after the wake word"), ``conversational`` (a getter:
    after a message, keep listening for a follow-up to the same agent without the wake word), ``busy`` (True while
    an agent answers or the computer speaks: holds the follow-up window open) and ``sound`` (the short listening
    sound)."""

    def __init__(
        self,
        enabled: Callable[[], bool] | None = None,
        window_s: float | Callable[[], float] = WINDOW_S,
        conversational: Callable[[], bool] | None = None,
    ) -> None:
        self.enabled = enabled or (lambda: True)
        self.agents: Callable[[], list[dict[str, Any]]] = list
        # deliver(target, text, source) -> what to show ("Sent to Claude"), "" when it could not be delivered
        self.deliver: Callable[[str, str, str], str] | None = None
        self.on_armed: Callable[[str], None] | None = None  # "Listening for your message to Claude..."
        # v0.1.26: the short listening sound ("listen": waiting for your words; "heard": your message was taken)
        self.sound: Callable[[str], Any] | None = None
        self.on_stop: Callable[[str], None] | None = None  # v0.1.26: "stop" / "that's all" ended the conversation
        self.conversational = conversational or (lambda: False)
        self.busy: Callable[[], bool] = lambda: False
        self.state = TalkState(window_s, hold=lambda: self.busy())
        self.conversation: str | None = None  # the agent a conversation is going on with (conversational mode)

    def ready(self) -> bool:
        try:
            return self.deliver is not None and bool(self.enabled())
        except Exception:
            return False

    def is_conversational(self) -> bool:
        try:
            return bool(self.conversational())
        except Exception:
            return False

    def route(self, text: str) -> Route | None:
        if not self.ready():
            return None
        try:
            agents = list(self.agents())
        except Exception:
            agents = []
        return match(text, agents)

    def _sound(self, kind: str) -> None:
        if self.sound is not None:
            with contextlib.suppress(Exception):
                self.sound(kind)

    def arm(self, target: str = DEFAULT, source: str = "gesture") -> None:
        self.state.arm(target, source)
        self._sound("listen")
        if self.on_armed is not None:
            with contextlib.suppress(Exception):
                self.on_armed(target)

    def send(self, target: str, text: str, source: str) -> str:
        if self.deliver is None:
            return ""
        self._sound("heard")
        try:
            label = str(self.deliver(target, text, source) or "")
        except Exception:
            label = ""
        if label and self.is_conversational():
            # v0.1.26: the agent answers, then listens for your follow-up (no wake word) for ``window_s``
            self.conversation = target
            self.state.arm(target, "conversation", held=True)
        return label

    def stop_conversation(self, target: str | None = None) -> None:
        """ "stop" / "that's all": no more follow-ups until the next wake word."""
        t = target if target is not None else (self.conversation or DEFAULT)
        self.state.disarm()
        self.conversation = None
        if self.on_stop is not None:
            with contextlib.suppress(Exception):
                self.on_stop(t)
