"""Timing logs for talking to an agent (v0.1.26): where the time goes between the end of what you said and the
first sound of the answer.

One :class:`Trace` per message, stages in order:

* ``endpoint`` - the silence waited for before deciding you finished (``voice.agent_end_silence_ms``);
* ``stt`` - speech-to-text;
* ``route`` - deciding what kind of request it is (chat / do something / hand off);
* ``first_token`` - the model's first word;
* ``first_sentence`` - the first whole sentence, handed to the voice;
* ``first_audio`` - the voice's first sound (after the engine started and synthesized the first piece).

Times are ``time.monotonic()`` seconds; the speech end is ``closed_at - endpoint``. Each finished trace is logged
(``away.latency`` logger, one line) and kept in :data:`LOG` (the last 50; ``away doctor`` / tests read it). Only
durations are kept: never the words, never audio.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("away.latency")

STAGES = ("endpoint", "stt", "route", "first_token", "first_sentence", "first_audio")


@dataclass
class Trace:
    speech_end: float  # when you actually stopped talking
    marks: dict[str, float] = field(default_factory=dict)  # stage -> clock time it finished
    clock: Callable[[], float] = time.monotonic
    done: bool = False

    def mark(self, stage: str, at: float | None = None) -> None:
        if stage not in self.marks:
            self.marks[stage] = self.clock() if at is None else float(at)

    def breakdown(self) -> dict[str, float]:
        """Milliseconds per stage (each from the previous finished stage) plus ``total`` (speech end to sound)."""
        out: dict[str, float] = {}
        prev = self.speech_end
        for st in STAGES:
            if st in self.marks:
                t = self.marks[st]
                out[st] = round(max(0.0, t - prev) * 1000.0, 1)
                prev = t
        if "first_audio" in self.marks:
            out["total"] = round(max(0.0, self.marks["first_audio"] - self.speech_end) * 1000.0, 1)
        return out

    def line(self) -> str:
        b = self.breakdown()
        return "latency: " + " ".join(f"{k}={v:.0f}ms" for k, v in b.items())


LOG: deque[dict[str, float]] = deque(maxlen=50)
_lock = threading.Lock()
_pending: Trace | None = None  # made by the voice input, picked up by the agent that answers
_active: Trace | None = None  # the turn being answered (the voice marks its first sound on it)
listeners: list[Callable[[dict[str, float]], None]] = []


def start(closed_at: float, end_wait_s: float, clock: Callable[[], float] = time.monotonic) -> Trace:
    """The voice input decided you finished speaking at ``closed_at`` after ``end_wait_s`` of silence."""
    global _pending
    tr = Trace(float(closed_at) - float(end_wait_s), clock=clock)
    tr.mark("endpoint", closed_at)
    with _lock:
        _pending = tr
    return tr


def take() -> Trace | None:
    """The agent answering the message takes its trace (and makes it the active one)."""
    global _pending, _active
    with _lock:
        tr, _pending = _pending, None
        _active = tr
    return tr


def active() -> Trace | None:
    return _active


def mark(stage: str) -> None:
    tr = _active
    if tr is not None and not tr.done:
        tr.mark(stage)
        if stage == "first_audio":
            finish(tr)


def spoken() -> None:
    """The voice finished a piece: an engine that never marked ``first_audio`` (the classic Windows voice) started
    its sound right when it was asked to speak (``tts_start``)."""
    tr = _active
    if tr is not None and not tr.done and "first_audio" not in tr.marks:
        if "tts_start" in tr.marks:
            tr.mark("first_audio", tr.marks["tts_start"])
        finish(tr)


def finish(tr: Trace | None) -> dict[str, float]:
    global _active
    if tr is None or tr.done:
        return {}
    tr.done = True
    b = tr.breakdown()
    LOG.append(b)
    log.info(tr.line())
    for fn in list(listeners):
        with contextlib.suppress(Exception):
            fn(b)
    with _lock:
        if _active is tr:
            _active = None
    return b


def reset() -> None:
    """Tests."""
    global _pending, _active
    with _lock:
        _pending = _active = None
    LOG.clear()


def summary(rows: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Median milliseconds per stage over the logged traces."""
    rows = list(LOG) if rows is None else rows
    out: dict[str, float] = {}
    for k in (*STAGES, "total"):
        vals = sorted(float(r[k]) for r in rows if k in r)
        if vals:
            out[k] = vals[len(vals) // 2]
    return out
