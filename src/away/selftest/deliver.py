"""Where a self-test report goes (v0.1.24): to every agent the person has connected, never to just one.

For each agent that is connected (or paired and able to reconnect by itself), through the channels Agent Link
already has:

* its **inbox**: a short note ("Self-test: 1 thing not working ...") it reads with the ``inbox`` tool;
* its **wake-up address** (webhook), when the person set one: ``{"event": "selftest.completed", "report": {...}}``;
* the Agent Link event **selftest.completed** (``wait_for_event`` / the event stream / ``away://events``);
* and the full report is always readable as ``away://selftest/latest`` and ``GET /v1/selftest/latest``.

When no agent is connected the report stays on this PC: the window shows it with "Copy report" and "Save as file".
The report is redacted before any of this (:mod:`away.selftest.redact`).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterable
from typing import Any

from away.selftest.model import FAIL, WARN

MAX_NOTE_LINES = 6


def note_text(report: dict[str, Any]) -> str:
    """The inbox note: the headline, then the lines that need attention (with their fix)."""
    lines = [f"Project Away self-test: {report.get('headline', '')}"]
    bad = [r for r in report.get("results") or [] if r.get("status") in (FAIL, WARN)]
    for r in bad[:MAX_NOTE_LINES]:
        fix = f" (fix: {r['fix']['label']})" if r.get("fix") else ""
        lines.append(
            f"- {'NOT WORKING' if r.get('status') == FAIL else 'look at'}: {r.get('title')}: {r.get('summary')}{fix}"
        )
    if len(bad) > MAX_NOTE_LINES:
        lines.append(f"- and {len(bad) - MAX_NOTE_LINES} more")
    lines.append("Full report: away://selftest/latest or GET /v1/selftest/latest")
    return "\n".join(lines)


def compact(report: dict[str, Any]) -> dict[str, Any]:
    """The event payload: everything but the bulky details."""
    return {
        "id": report.get("id"),
        "status": report.get("status"),
        "headline": report.get("headline"),
        "counts": report.get("counts"),
        "time": report.get("time"),
        "trigger": report.get("trigger"),
        "results": [
            {k: r.get(k) for k in ("id", "title", "status", "summary", "fix")} for r in report.get("results") or []
        ],
        "full": "away://selftest/latest",
    }


def deliver(
    report: dict[str, Any],
    agents: Iterable[Any],
    *,
    inbox: Any = None,
    events: Any = None,
    post: Callable[[str, dict[str, Any]], Any] | None = None,
    background: bool = True,
) -> dict[str, Any]:
    """Send ``report`` (already redacted) to every agent in ``agents``. Returns what went where."""
    targets = list(agents)
    out: dict[str, Any] = {"delivered": bool(targets), "agents": []}
    if events is not None:
        with contextlib.suppress(Exception):
            ev = events.publish("selftest.completed", compact(report))
            out["event_seq"] = ev.get("seq") if isinstance(ev, dict) else None
    if not targets:
        out["local_only"] = True
        return out
    if inbox is not None:
        with contextlib.suppress(Exception):
            m = inbox.add(note_text(report), targets, source="system", wake=False)
            out["inbox_id"] = m.id
    jobs: list[Callable[[], None]] = []
    for a in targets:
        row: dict[str, Any] = {"agent": getattr(a, "name", "?"), "inbox": inbox is not None, "webhook": "none"}
        url = str(getattr(a, "wake_url", "") or "")
        if url and post is not None:
            row["webhook"] = "sending"
            payload = {"event": "selftest.completed", "agent": getattr(a, "name", ""), "report": report}

            def send(u: str = url, p: dict[str, Any] = payload, r: dict[str, Any] = row) -> None:
                try:
                    post(u, p)
                    r["webhook"] = "ok"
                except Exception as exc:
                    r["webhook"] = f"failed: {str(exc)[:120]}"

            jobs.append(send)
        out["agents"].append(row)
    if jobs:
        if background:
            threading.Thread(target=lambda: [j() for j in jobs], daemon=True, name="selftest-deliver").start()
        else:
            for j in jobs:
                j()
    return out
