"""Daily cost ledger for paid API keys, with a stop at the person's daily limit.

Ported and trimmed from GHOST-Chimera ``ghostchimera/model_layer/cost_monitor.py`` (MIT, Fernando Garza):
provider-reported token counts when given, else ~4 characters per token; per-provider and per-day totals;
latency samples for p50/p95; atomic save. Changed for Away: the budget is one *daily* limit across all paid
providers (free and local calls always record $0 and never count), days are local days, and only 30 days are kept.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    """Today's spend reached the daily limit; paid models wait until tomorrow."""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class CostLedger:
    LATENCY_SAMPLES = 200
    KEEP_DAYS = 30

    def __init__(self, path: Path | None = None, today: Callable[[], str] = _today) -> None:
        self.path = Path(path) if path is not None else None
        self.today = today
        self._lock = threading.Lock()
        self.daily: dict[str, dict[str, dict[str, float]]] = {}  # day -> provider -> {spend, calls, in, out}
        self.latency: dict[str, list[float]] = {}
        self.limit_usd = 0.0  # 0 = paid models are not used at all
        self._load()

    # ------------------------------------------------------------ record
    def record(
        self,
        provider: str,
        model: str,
        *,
        price_in: float = 0.0,
        price_out: float = 0.0,
        input_text: str = "",
        output_text: str = "",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_s: float | None = None,
    ) -> float:
        """Record one call; prices are USD per 1M tokens. Returns its cost."""
        tin = max(0, input_tokens if input_tokens is not None else estimate_tokens(input_text))
        tout = max(0, output_tokens if output_tokens is not None else estimate_tokens(output_text))
        cost = tin / 1e6 * max(0.0, price_in) + tout / 1e6 * max(0.0, price_out)
        with self._lock:
            day = self.daily.setdefault(self.today(), {})
            e = day.setdefault(provider, {"spend": 0.0, "calls": 0.0, "in": 0.0, "out": 0.0})
            e["spend"] += cost
            e["calls"] += 1
            e["in"] += tin
            e["out"] += tout
            if latency_s is not None and latency_s >= 0:
                s = self.latency.setdefault(provider, [])
                s.append(float(latency_s))
                del s[: max(0, len(s) - self.LATENCY_SAMPLES)]
            for old in sorted(self.daily)[: -self.KEEP_DAYS]:
                del self.daily[old]
        self.save()
        return cost

    # ------------------------------------------------------------ budget
    def spent_today(self, provider: str | None = None) -> float:
        with self._lock:
            day = self.daily.get(self.today(), {})
            if provider is not None:
                return float(day.get(provider, {}).get("spend", 0.0))
            return float(sum(e.get("spend", 0.0) for e in day.values()))

    def can_spend(self, estimate_usd: float = 0.0) -> bool:
        return self.limit_usd > 0 and self.spent_today() + max(0.0, estimate_usd) <= self.limit_usd

    def check(self, estimate_usd: float = 0.0) -> None:
        if not self.can_spend(estimate_usd):
            if self.limit_usd <= 0:
                raise BudgetExceeded("paid models are turned off (daily limit is $0)")
            raise BudgetExceeded(
                f"today's limit of ${self.limit_usd:.2f} is reached (${self.spent_today():.2f} spent); "
                "paid models wait until tomorrow"
            )

    # ------------------------------------------------------------ report
    @staticmethod
    def _pct(samples: list[float], pct: float) -> float:
        if not samples:
            return 0.0
        o = sorted(samples)
        return o[min(len(o) - 1, max(0, int(pct / 100.0 * len(o))))]

    def latency_stats(self, provider: str) -> dict[str, float]:
        with self._lock:
            s = list(self.latency.get(provider, []))
        return {"calls": float(len(s)), "p50_s": self._pct(s, 50), "p95_s": self._pct(s, 95)}

    def summary(self) -> dict[str, Any]:
        with self._lock:
            day = {k: dict(v) for k, v in self.daily.get(self.today(), {}).items()}
        return {
            "day": self.today(),
            "spent_usd": round(sum(e["spend"] for e in day.values()), 6),
            "limit_usd": self.limit_usd,
            "providers": day,
        }

    # ------------------------------------------------------------ disk
    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {"version": 1, "daily": self.daily}
        with contextlib.suppress(OSError):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self.path)

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        daily = data.get("daily") if isinstance(data, dict) else None
        if not isinstance(daily, dict):
            return
        for day, provs in list(daily.items())[-self.KEEP_DAYS :]:
            if not isinstance(provs, dict):
                continue
            clean: dict[str, dict[str, float]] = {}
            for prov, st in provs.items():
                if not isinstance(st, dict):
                    continue
                try:
                    vals = {k: float(st.get(k, 0) or 0) for k in ("spend", "calls", "in", "out")}
                except (TypeError, ValueError):
                    continue
                if all(math.isfinite(v) and v >= 0 for v in vals.values()):
                    clean[str(prov)] = vals
            if clean:
                self.daily[str(day)] = clean
