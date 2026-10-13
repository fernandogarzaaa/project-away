"""Jev (TypeSafe AI's "System One" model) for quick, typed decisions.

Jev does not write text: it answers typed questions about a ``state`` with probabilities, in about 0.3 s.
``noul`` = yes/no (probability that the statement is true), ``choice`` = one of named options, ``score`` = a level
on a rubric. Away uses it for four computer-use decisions:

1. :meth:`Decider.risk` - is this action a send / pay / delete / anything that cannot be undone? (feeds the existing
   approval: a risky action is confirmed on the PC first);
2. :meth:`Decider.pick_element` - which on-screen element (UI Automation names) matches "the blue Save button";
3. :meth:`Decider.succeeded` - did the action work, judged from the screen / element text afterwards;
4. :meth:`Decider.route` - is this a chat, a job on this PC, or a big job for another agent?

``jev-1.13-free`` answers without a key at https://opencode.ai/inference/v1/systemone (checked 2026-10-14; free
"for a limited time"). With an OpenCode key ``jev-1.13`` is used. When Jev is off or not answering, the same
questions go to the chat model as a JSON question (:class:`ChatDecider`), and when that fails too, to plain rules.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from away.liveagent.http import Client, HttpError

ENDPOINT = "https://opencode.ai/inference/v1/systemone"
FREE_MODEL = "jev-1.13-free"
PAID_MODEL = "jev-1.13"
ROUTES = {
    "chat": "Talking, a question, a fact, advice, or anything answered with words only",
    "computer": "Doing something on this computer now: open, click, type, scroll, switch apps or change a setting",
    "handoff": "A big or long job for another AI agent: write code, research, long documents, many steps over time",
}
RISK_QUESTION = (
    "Is this action irreversible or high-stakes, such as sending a message or e-mail, posting publicly, paying or "
    "buying, deleting or erasing something, submitting a form, installing software, or signing out?"
)
RISKY_WORDS = re.compile(
    r"\b(send|submit|pay|buy|purchase|delete|remove|erase|confirm|install|uninstall|checkout|transfer|post|"
    r"publish|sign out|log out|format|reset|empty trash|wipe)\b",
    re.IGNORECASE,
)
COMPUTER_WORDS = re.compile(
    r"^(please )?(open|close|click|type|press|scroll|go to|switch|start|launch|show|minimi[sz]e|maximi[sz]e|"
    r"turn (on|off)|mute|unmute|play|pause|select|copy|paste|save|find|search for|move|drag|zoom)\b",
    re.IGNORECASE,
)
HANDOFF_WORDS = re.compile(
    r"\b(write (me )?(a|an|the) (program|script|app|essay|report|article|document)|build (me )?(a|an)|"
    r"research|refactor|fix (the|my) (code|bug|repo)|pull request|deploy|every (day|week|morning)|remind me)\b",
    re.IGNORECASE,
)


@dataclass
class Answer:
    value: Any  # True/False (noul), an option name (choice), a number (score)
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    source: str = "jev"  # jev | chat | rules
    latency_s: float = 0.0


class JevClient:
    def __init__(self, key: str = "", client: Client | None = None, endpoint: str = ENDPOINT,
                 timeout: float = 6.0) -> None:
        self.key = key
        self.client = client or Client()
        self.endpoint = endpoint
        self.timeout = timeout

    @property
    def model(self) -> str:
        return PAID_MODEL if self.key else FREE_MODEL

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        body = {"model": self.model, "state": state, "questions": questions}
        data = self.client.post_json(self.endpoint, body, headers, self.timeout)
        answers = (data or {}).get("answers")
        if not isinstance(answers, dict):
            raise HttpError(502, "Jev gave no answers")
        return answers

    def noul(self, state: Any, instructions: str, criteria: Any = None) -> Answer:
        q: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            q["criteria"] = criteria
        t0 = time.monotonic()
        a = self.ask(state, {"q": q})["q"]
        p = float(a.get("noul", 0.0))
        return Answer(p >= 0.5, abs(p - 0.5) * 2, {"yes": p, "no": 1 - p}, "jev", time.monotonic() - t0)

    def choice(self, state: Any, instructions: str, options: dict[str, str]) -> Answer:
        t0 = time.monotonic()
        a = self.ask(state, {"q": {"type": "choice", "instructions": instructions, "criteria": options}})["q"]
        probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
        return Answer(str(a.get("choice")), float(a.get("confidence", 0.0)), probs, "jev", time.monotonic() - t0)

    def score(self, state: Any, instructions: str, levels: list[str]) -> Answer:
        t0 = time.monotonic()
        a = self.ask(state, {"q": {"type": "score", "instructions": instructions, "criteria": levels}})["q"]
        probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
        return Answer(float(a.get("score", 0.0)), float(a.get("confidence", 0.0)), probs, "jev", time.monotonic() - t0)


class ChatDecider:
    """The same questions asked to a chat model, answered as JSON (the fallback when Jev is not available)."""

    def __init__(self, complete: Callable[[str, str], str]) -> None:
        self.complete = complete  # (system, user) -> text

    def _json(self, user: str) -> dict[str, Any]:
        from away.liveagent.textstream import strip_reasoning

        text = strip_reasoning(self.complete("Answer with one JSON object only, no other text.", user))
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("no JSON in the answer")
        return dict(json.loads(m.group(0)))

    def noul(self, state: Any, instructions: str, criteria: Any = None) -> Answer:
        d = self._json(f'State:\n{state}\n\nQuestion: {instructions}\nReply as {{"yes": <probability 0..1>}}.')
        p = max(0.0, min(1.0, float(d.get("yes", 0.0))))
        return Answer(p >= 0.5, abs(p - 0.5) * 2, {"yes": p, "no": 1 - p}, "chat")

    def choice(self, state: Any, instructions: str, options: dict[str, str]) -> Answer:
        opts = "\n".join(f"- {k}: {v}" for k, v in options.items())
        d = self._json(f'State:\n{state}\n\n{instructions}\nOptions:\n{opts}\nReply as {{"choice": "<option>"}}.')
        c = str(d.get("choice", ""))
        if c not in options:
            raise ValueError(f"not an option: {c!r}")
        return Answer(c, 0.6, {c: 0.6}, "chat")


class Decider:
    """Jev first, then the chat model, then rules. Each step's failure is remembered for a minute so a broken Jev
    never slows every turn down."""

    def __init__(self, jev: JevClient | None, chat: ChatDecider | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.jev = jev
        self.chat = chat
        self.clock = clock
        self._jev_down_until = -1.0
        self.last_source = ""

    def _try(self, method: str, *args: Any) -> Answer | None:
        if self.jev is not None and self.clock() >= self._jev_down_until:
            try:
                a: Answer = getattr(self.jev, method)(*args)
                self.last_source = "jev"
                return a
            except Exception:
                self._jev_down_until = self.clock() + 60.0
        if self.chat is not None and hasattr(self.chat, method):
            try:
                a = getattr(self.chat, method)(*args)
                self.last_source = "chat"
                return a
            except Exception:
                pass
        return None

    # 1. risk gate
    def risk(self, action: str, target: str = "", context: str = "") -> Answer:
        state = {"action": action, "target": target, "screen": context[:1500]}
        a = self._try("noul", state, RISK_QUESTION)
        rule = bool(RISKY_WORDS.search(f"{action} {target}"))
        if a is None:
            self.last_source = "rules"
            return Answer(rule, 1.0 if rule else 0.5, {}, "rules")
        if rule and not a.value:  # the word rule is never overruled towards "safe"
            return Answer(True, a.confidence, a.probabilities, a.source + "+rules", a.latency_s)
        return a

    # 2. element choice
    def pick_element(self, target: str, candidates: list[str]) -> Answer | None:
        names = [c for c in dict.fromkeys(c.strip() for c in candidates if c and c.strip())][:64]
        if not names:
            return None
        if len(names) == 1:
            return Answer(names[0], 1.0, {names[0]: 1.0}, "rules")
        opts = {f"e{i}": n[:120] for i, n in enumerate(names)}
        a = self._try("choice", f"The user wants: {target}", "Which on-screen element is the user's target?", opts)
        if a is None or a.value not in opts:
            from difflib import get_close_matches

            m = get_close_matches(target, names, 1, 0.0)
            self.last_source = "rules"
            return Answer(m[0], 0.3, {}, "rules") if m else None
        return Answer(opts[str(a.value)], a.confidence, a.probabilities, a.source, a.latency_s)

    # 3. post-action success check
    def succeeded(self, goal: str, after_text: str) -> Answer:
        a = self._try("noul", {"goal": goal, "screen_after": after_text[:3000]},
                      "Does the screen show that the goal was achieved?")
        if a is None:
            self.last_source = "rules"
            return Answer(True, 0.0, {}, "rules")  # unknown: do not claim failure
        return a

    # 4. intent routing
    def route(self, text: str) -> Answer:
        a = self._try("choice", text, "What kind of request is this?", ROUTES)
        if a is None or a.value not in ROUTES:
            self.last_source = "rules"
            if HANDOFF_WORDS.search(text):
                return Answer("handoff", 0.6, {}, "rules")
            if COMPUTER_WORDS.search(text.strip()):
                return Answer("computer", 0.6, {}, "rules")
            return Answer("chat", 0.5, {}, "rules")
        return a


def probe(client: Client | None = None, key: str = "") -> str:
    """Live check for Settings ("Jev: works, free, 0.3 s")."""
    j = JevClient(key, client)
    a = j.noul("Click the Send button in the mail app", RISK_QUESTION)
    return f"Jev works ({'your key' if key else 'free, no key'}, {a.latency_s:.1f} s)"
