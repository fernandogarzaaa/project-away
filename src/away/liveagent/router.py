"""Which model answers, and what happens when it cannot.

**Router** - one ordered list of models per kind of job:

* ``chat``   quick spoken answers: the fastest suitable model;
* ``plan``   doing things on the PC (several steps): the strongest model;
* ``vision`` looking at the screen: only models that accept pictures, only private ones (providers that may keep
  prompts never get a screenshot), and only when the person allowed screen access;
* ``decide`` safety and routing decisions: Jev first (:mod:`.jev`), else a fast model answering in JSON.

Order (the "funnel"): the person's choice for that job or their preferred model -> their other configured models ->
free models -> models on this PC. Paid models are used only while today's spend is under the daily limit.

**Funnel** - tries that list in order with a first-word timeout, one quick retry on 429/5xx when the provider says
how long to wait, and a circuit breaker per provider (three failures in a row: rest it for a minute, doubling up to
ten; a refused key: an hour). Once a model has started answering, it keeps the turn.

Ported with attribution from the author's MIT projects: GHOST-Chimera ``model_layer/free_router.py`` (ordered
failover, privacy routing, honest skip reasons) and ape-mcp freepool (cost ladder local < free < paid, failure
classes, Retry-After).
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from away.liveagent import privacy
from away.liveagent.catalog import ModelInfo
from away.liveagent.http import HttpError
from away.liveagent.ledger import CostLedger

TASKS = ("chat", "plan", "vision", "decide")
# a paid model whose price is unknown is counted at this (USD per 1M tokens, in/out), so the daily limit still holds
UNKNOWN_PRICE = (1.0, 5.0)


# ------------------------------------------------------------------------------------------------ failures
def classify(exc: BaseException) -> str:
    """rate_limit | server | timeout | auth | not_found | bad_request | budget | network (ape-mcp classifyFailure)."""
    from away.liveagent.ledger import BudgetExceeded

    if isinstance(exc, BudgetExceeded):
        return "budget"
    if isinstance(exc, HttpError):
        s = exc.status
        if s == 429:
            return "rate_limit"
        if s in (401, 403):
            return "auth"
        if s == 404:
            return "not_found"
        if s >= 500:
            return "server"
        if s >= 400:
            return "bad_request"
        if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
            return "timeout"
        return "network"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "network"


@dataclass
class Breaker:
    failures: int = 0
    open_until: float = 0.0
    reason: str = ""
    base_s: float = 60.0


class Breakers:
    THRESHOLD = 3
    MAX_S = 600.0
    AUTH_S = 3600.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.state: dict[str, Breaker] = {}

    def get(self, pid: str) -> Breaker:
        return self.state.setdefault(pid, Breaker())

    def is_open(self, pid: str) -> bool:
        return self.clock() < self.get(pid).open_until

    def ok(self, pid: str) -> None:
        b = self.get(pid)
        b.failures, b.open_until, b.reason, b.base_s = 0, 0.0, "", 60.0

    def fail(self, pid: str, kind: str, retry_after: float | None = None) -> None:
        b = self.get(pid)
        b.failures += 1
        b.reason = kind
        now = self.clock()
        if kind == "auth":
            b.open_until = now + self.AUTH_S
        elif kind == "rate_limit" and retry_after:
            b.open_until = max(b.open_until, now + min(self.MAX_S, retry_after))
        elif b.failures >= self.THRESHOLD:
            b.open_until = now + b.base_s
            b.base_s = min(self.MAX_S, b.base_s * 2)

    def public(self) -> dict[str, dict[str, Any]]:
        now = self.clock()
        return {
            k: {"open": now < b.open_until, "rest_s": max(0, int(b.open_until - now)), "reason": b.reason,
                "failures": b.failures}
            for k, b in self.state.items()
        }


# ------------------------------------------------------------------------------------------------ router
@dataclass
class Candidate:
    info: ModelInfo
    provider: Any  # providers.Provider
    rank: int = 0  # 0 = the person's choice, 1 = configured, 2 = free, 3 = local
    why: str = ""

    @property
    def ref(self) -> str:
        return self.info.ref

    @property
    def paid(self) -> bool:
        return not (self.info.free or self.info.local)


class Router:
    def __init__(
        self,
        providers: dict[str, Any],
        models: list[ModelInfo],
        preferred: str = "",
        routing: dict[str, str] | None = None,
        allow_screen: bool = False,
    ) -> None:
        self.providers = providers  # provider id -> Provider
        self.models = [m for m in models if m.provider in providers]
        self.preferred = preferred
        self.routing = dict(routing or {})
        self.allow_screen = allow_screen

    def find(self, ref: str) -> ModelInfo | None:
        if not ref or "/" not in ref:
            return None
        pid, mid = ref.split("/", 1)
        for m in self.models:
            if m.provider == pid and m.id == mid:
                return m
        if pid in self.providers:  # a model the provider has but we did not list: allow it
            from away.liveagent.catalog import guess

            p = self.providers[pid]
            return guess(pid, mid, free=bool(getattr(p, "free", False)), local=bool(getattr(p, "local", False)),
                         may_keep_prompts=bool(getattr(p, "may_keep_prompts", False)))
        return None

    def candidates(self, task: str) -> list[Candidate]:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        pool = list(self.models)
        if task == "vision":
            if not self.allow_screen:
                return []
            pool = [m for m in pool if m.vision and not m.may_keep_prompts]
        first: list[ModelInfo] = []
        for ref in (self.routing.get(task, ""), self.preferred):
            m = self.find(ref)
            if m is not None and m not in first and (task != "vision" or (m.vision and not m.may_keep_prompts)):
                first.append(m)

        def rank(m: ModelInfo) -> int:
            if m.local:
                return 3
            if m.free:
                return 2
            return 1

        def suit(m: ModelInfo) -> tuple[int, ...]:
            if task in ("chat", "decide"):
                return (not m.fast, m.reasoning, m.strong)
            if task == "plan":
                return (not m.strong, not m.tools, m.fast)
            return (not m.strong, not m.fast)

        rest = sorted((m for m in pool if m not in first), key=lambda m: (rank(m), *suit(m), m.provider, m.id))
        # at most two models per provider in the list (a dead provider should not hide the others)
        per: dict[str, int] = {}
        out = [Candidate(m, self.providers[m.provider], 0, "your choice") for m in first]
        for m in rest:
            if per.get(m.provider, 0) >= 2:
                continue
            per[m.provider] = per.get(m.provider, 0) + 1
            r = rank(m)
            out.append(Candidate(m, self.providers[m.provider], r, ("configured", "free", "on this PC")[r - 1]))
        return out

    def table(self) -> dict[str, list[str]]:
        """For Settings > Advanced and Agent Link: the order per job."""
        return {t: [c.ref for c in self.candidates(t)] for t in TASKS}


# ------------------------------------------------------------------------------------------------ funnel
@dataclass
class CallRecord:
    task: str
    chosen: str = ""
    title: str = ""
    attempts: list[tuple[str, str]] = field(default_factory=list)  # (ref, outcome)
    first_token_s: float | None = None
    total_s: float = 0.0
    cost_usd: float = 0.0
    redacted: bool = False
    dropped_images: int = 0
    text: str = ""


class NoModel(RuntimeError):
    def __init__(self, message: str, attempts: list[tuple[str, str]]) -> None:
        super().__init__(message)
        self.attempts = attempts


_DONE = object()


class Funnel:
    def __init__(
        self,
        router: Router,
        ledger: CostLedger | None = None,
        breakers: Breakers | None = None,
        first_token_timeout: float = 8.0,
        turn_timeout: float = 60.0,
        max_retry_wait: float = 2.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.router = router
        self.ledger = ledger or CostLedger()
        self.breakers = breakers or Breakers(clock)
        self.first_token_timeout = first_token_timeout
        self.turn_timeout = turn_timeout
        self.max_retry_wait = max_retry_wait
        self.clock = clock
        self.sleep = sleep

    def _producer(self, cand: Candidate, model: str, messages: list[dict[str, Any]], q: queue.Queue[Any],
                  kw: dict[str, Any], stop: threading.Event) -> None:
        try:
            for piece in cand.provider.stream(model, messages, **kw):
                if stop.is_set():
                    break
                q.put(piece)
            q.put(_DONE)
        except BaseException as exc:  # handed to the consumer
            q.put(exc)

    def stream(
        self,
        task: str,
        messages: list[dict[str, Any]],
        record: CallRecord | None = None,
        *,
        max_tokens: int = 400,
        json_mode: bool = False,
        candidates: list[Candidate] | None = None,
    ) -> Iterator[str]:
        rec = record if record is not None else CallRecord(task)
        rec.task = task
        cands = candidates if candidates is not None else self.router.candidates(task)
        if not cands:
            msg = "no model can do this yet" if task != "vision" else "no private picture-capable model is set up"
            raise NoModel(msg, rec.attempts)
        t_start = self.clock()
        for cand in cands:
            pid = cand.info.provider
            if self.breakers.is_open(pid):
                rec.attempts.append((cand.ref, f"resting ({self.breakers.get(pid).reason})"))
                continue
            if cand.paid:
                if not self.ledger.can_spend():
                    rec.attempts.append((cand.ref, "skipped: daily limit reached" if self.ledger.limit_usd > 0
                                         else "skipped: paid models are off"))
                    continue
            prep = privacy.prepare(messages, cand.info.may_keep_prompts, allow_images=cand.info.vision)
            if task == "vision" and prep.dropped_images:
                rec.attempts.append((cand.ref, "skipped: pictures stay private"))
                continue
            rec.redacted = rec.redacted or prep.redacted
            rec.dropped_images += prep.dropped_images
            reasoning = cand.info.reasoning
            kw = {"max_tokens": max(max_tokens, 1200) if reasoning else max_tokens, "json_mode": json_mode,
                  "timeout": self.turn_timeout}
            for attempt in (1, 2):
                q: queue.Queue[Any] = queue.Queue()
                stop = threading.Event()
                t0 = self.clock()
                th = threading.Thread(target=self._producer, args=(cand, cand.info.id, prep.messages, q, kw, stop),
                                      daemon=True, name="liveagent-call")
                th.start()
                try:
                    first = q.get(timeout=self.first_token_timeout)
                except queue.Empty:
                    stop.set()
                    rec.attempts.append((cand.ref, f"no answer in {self.first_token_timeout:.0f} s"))
                    self.breakers.fail(pid, "timeout")
                    break
                if isinstance(first, BaseException):
                    kind = classify(first)
                    ra = getattr(first, "retry_after", None)
                    rec.attempts.append((cand.ref, f"{kind}: {str(first)[:80]}"))
                    if attempt == 1 and kind in ("rate_limit", "server") and ra is not None and ra <= self.max_retry_wait:
                        self.sleep(ra)
                        continue
                    self.breakers.fail(pid, kind, ra)
                    break
                # committed: this model answers the turn
                rec.chosen, rec.title = cand.ref, cand.info.label()
                rec.first_token_s = self.clock() - t_start
                out: list[str] = []
                item: Any = first
                try:
                    while item is not _DONE:
                        if isinstance(item, BaseException):
                            self.breakers.fail(pid, classify(item))
                            rec.attempts.append((cand.ref, f"stopped mid-answer: {str(item)[:80]}"))
                            break
                        out.append(str(item))
                        yield str(item)
                        left = self.turn_timeout - (self.clock() - t0)
                        if left <= 0:
                            stop.set()
                            rec.attempts.append((cand.ref, "answer cut at the time limit"))
                            break
                        try:
                            item = q.get(timeout=left)
                        except queue.Empty:
                            stop.set()
                            rec.attempts.append((cand.ref, "answer cut at the time limit"))
                            break
                    else:
                        self.breakers.ok(pid)
                        rec.attempts.append((cand.ref, "ok"))
                finally:
                    stop.set()
                    rec.text = "".join(out)
                    rec.total_s = self.clock() - t_start
                    usage = getattr(cand.provider, "last_usage", None)
                    rec.cost_usd = self.ledger.record(
                        pid,
                        cand.info.id,
                        price_in=0.0 if not cand.paid else (cand.info.price_in or UNKNOWN_PRICE[0]),
                        price_out=0.0 if not cand.paid else (cand.info.price_out or UNKNOWN_PRICE[1]),
                        input_text=" ".join(privacy._text_of(m.get("content")) for m in prep.messages),
                        output_text=rec.text,
                        input_tokens=getattr(usage, "input_tokens", None),
                        output_tokens=getattr(usage, "output_tokens", None),
                        latency_s=rec.first_token_s,
                    )
                return
        rec.total_s = self.clock() - t_start
        raise NoModel("every model is unavailable right now", rec.attempts)

    def complete(self, task: str, messages: list[dict[str, Any]], record: CallRecord | None = None,
                 **kw: Any) -> str:
        return "".join(self.stream(task, messages, record, **kw))
