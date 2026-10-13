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
    return best[1] if best else None


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


class TalkState:
    """The "waiting for your message" window after a bare wake phrase or the summon gesture."""

    def __init__(self, window_s: float = WINDOW_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s = float(window_s)
        self.clock = clock
        self._lock = threading.Lock()
        self._until = -1e9
        self._target = DEFAULT
        self.source = "voice"

    def arm(self, target: str = DEFAULT, source: str = "voice") -> None:
        with self._lock:
            self._until = self.clock() + self.window_s
            self._target = target
            self.source = source

    def armed(self) -> str | None:
        """The waiting target (DEFAULT or an agent id), or None when nothing is waiting."""
        with self._lock:
            return self._target if self.clock() <= self._until else None

    def disarm(self) -> None:
        with self._lock:
            self._until = -1e9


class TalkHook:
    """What the voice input needs from Agent Link: the agents to listen for, where to deliver a message, and the
    waiting window. The app creates one; Agent Link fills in ``agents`` and ``deliver`` when it starts."""

    def __init__(self, enabled: Callable[[], bool] | None = None, window_s: float = WINDOW_S) -> None:
        self.enabled = enabled or (lambda: True)
        self.agents: Callable[[], list[dict[str, Any]]] = list
        # deliver(target, text, source) -> what to show ("Sent to Claude"), "" when it could not be delivered
        self.deliver: Callable[[str, str, str], str] | None = None
        self.on_armed: Callable[[str], None] | None = None  # "Listening for your message to Claude..."
        self.state = TalkState(window_s)

    def ready(self) -> bool:
        try:
            return self.deliver is not None and bool(self.enabled())
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

    def arm(self, target: str = DEFAULT, source: str = "gesture") -> None:
        self.state.arm(target, source)
        if self.on_armed is not None:
            with contextlib.suppress(Exception):
                self.on_armed(target)

    def send(self, target: str, text: str, source: str) -> str:
        if self.deliver is None:
            return ""
        try:
            return str(self.deliver(target, text, source) or "")
        except Exception:
            return ""
