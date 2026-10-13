"""How busy the PC is, in plain words (v0.1.23).

* :func:`top_processes` - the five programs using the most CPU (name and percent of the whole PC), for status,
  health and ``away cameras diagnose`` (psutil when installed; otherwise ``tasklist``/``ps`` is not used: empty).
* :class:`LoadHints` - turns readings into one tray hint: "Your PC is busy (NVIDIA Broadcast 40%), so gestures may
  be choppy" when the whole PC stays above ``BUSY_PCT`` for ``BUSY_S`` seconds, and "Hand tracking is choppy
  (12 fps): <likely reason>" when the tracked camera runs below ``CHOPPY_FPS``.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable
from typing import Any

BUSY_PCT = 85.0
BUSY_S = 20.0
CHOPPY_FPS = 20.0
CHOPPY_S = 10.0
FRIENDLY = {
    "nvidia broadcast": "NVIDIA Broadcast",
    "nvbroadcast": "NVIDIA Broadcast",
    "msedge": "Microsoft Edge",
    "chrome": "Google Chrome",
    "firefox": "Firefox",
    "obs64": "OBS",
    "teams": "Microsoft Teams",
    "ms-teams": "Microsoft Teams",
    "zoom": "Zoom",
    "discord": "Discord",
    "msmpeng": "Windows Security scan",
    "searchindexer": "Windows Search indexing",
    "projectaway": "Project Away",
    "away": "Project Away",
}


def friendly_name(proc_name: str) -> str:
    base = os.path.basename(str(proc_name or "")).lower()
    for ext in (".exe", ".app"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    for k, v in FRIENDLY.items():
        if k in base:
            return v
    return os.path.basename(str(proc_name or "")).removesuffix(".exe") or "?"


def top_processes(n: int = 5, psutil_mod: Any = None, interval: float = 0.0) -> list[dict[str, Any]]:
    """``[{"name": "NVIDIA Broadcast", "cpu_pct": 40.1, "pid": 123}, ...]`` - percent of the whole PC (all cores),
    biggest first. Programs with the same name are added up. [] when psutil is not available."""
    try:
        ps = psutil_mod
        if ps is None:
            import psutil as ps  # type: ignore[no-redef]
    except Exception:
        return []
    cores = max(1, int(ps.cpu_count() or 1))
    totals: dict[str, dict[str, Any]] = {}
    procs = list(ps.process_iter(["name", "pid"]))
    for p in procs:
        try:
            p.cpu_percent(None)  # prime
        except Exception:
            continue
    if interval > 0:
        time.sleep(interval)
    for p in procs:
        try:
            pct = float(p.cpu_percent(None)) / cores
            name = friendly_name(p.info.get("name") or "")
        except Exception:
            continue
        if name.lower() in ("system idle process", "idle") or pct <= 0:
            continue
        row = totals.setdefault(name, {"name": name, "cpu_pct": 0.0, "pid": p.info.get("pid")})
        row["cpu_pct"] += pct
    out = sorted(totals.values(), key=lambda r: -r["cpu_pct"])[: max(0, n)]
    for r in out:
        r["cpu_pct"] = round(r["cpu_pct"], 1)
    return out


def describe_top(rows: list[dict[str, Any]], skip: str = "Project Away", n: int = 2) -> str:
    others = [r for r in rows if r.get("name") != skip][:n]
    return ", ".join(f"{r['name']} {r['cpu_pct']:.0f}%" for r in others)


class LoadHints:
    """Feed it readings every few seconds; :meth:`hint` is "" or one sentence for the tray."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._busy_since: float | None = None
        self._slow_since: float | None = None
        self.system_pct: float | None = None
        self.fps: float | None = None
        self.top: list[dict[str, Any]] = []
        self.reason = ""
        self.history: deque[float] = deque(maxlen=60)

    def update(
        self,
        system_pct: float | None = None,
        fps: float | None = None,
        top: list[dict[str, Any]] | None = None,
        reason: str = "",
    ) -> str:
        now = self.clock()
        self.system_pct, self.fps, self.reason = system_pct, fps, reason
        if top is not None:
            self.top = list(top)
        if system_pct is not None:
            self.history.append(float(system_pct))
        if system_pct is not None and system_pct >= BUSY_PCT:
            self._busy_since = self._busy_since if self._busy_since is not None else now
        else:
            self._busy_since = None
        if fps is not None and 0 < fps < CHOPPY_FPS:
            self._slow_since = self._slow_since if self._slow_since is not None else now
        else:
            self._slow_since = None
        return self.hint()

    def busy(self) -> bool:
        return self._busy_since is not None and self.clock() - self._busy_since >= BUSY_S

    def choppy(self) -> bool:
        return self._slow_since is not None and self.clock() - self._slow_since >= CHOPPY_S

    def hint(self) -> str:
        if self.busy():
            who = describe_top(self.top)
            return f"Your PC is busy{f' ({who})' if who else ''}, so gestures may be choppy"
        if self.choppy():
            why = self.reason or "the camera is sending few pictures"
            return f"Hand tracking is choppy ({self.fps:.0f} fps): {why}"
        return ""


def choppy_reason(app: Any) -> str:
    """The likely reason for a low hand-tracking rate, in plain words."""
    if getattr(app, "camera_unavailable", None):
        return "a camera that keeps failing is being retried"
    if getattr(app, "camera_retry_stopped", False):
        return "a camera failed; the other one may be short of USB bandwidth"
    return "the camera is slow (light, USB bandwidth or another app using it)"
