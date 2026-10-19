"""Where a report goes (v0.1.27): one audience policy for both reports.

The one-click self-test and Check my PC are the same class of report (redacted environment detail), so they
share one audience: only agents the person let "Share my logs and recordings" (``audience="share_logs"``,
:func:`away.selftest.pccheck.may_receive`). The safer direction won; "Run camera and speed checks" alone is
not enough to be sent either report.

For each agent in the audience, through the channels Agent Link already has:

* its **inbox**: a short note ("Self-test: 1 thing not working ...") it reads with the ``inbox`` tool;
* its **wake-up address** (webhook), when the person set one: ``{"event": "selftest.completed", "report": {...}}``;
* and the full report is always readable as ``away://selftest/latest`` and ``GET /v1/selftest/latest``.

The Agent Link event stream is deliberately not used for the share_logs audience: every connected agent can
read it, including ones outside the audience. When no agent is in the audience the report stays on this PC:
the window shows it with "Copy report" and "Save as file".
The report is redacted before any of this (:mod:`away.selftest.redact`).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterable
from typing import Any

from away.selftest.model import FAIL, WARN

MAX_NOTE_LINES = 6

AUDIENCES = ("all", "share_logs")


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
    event: str = "selftest.completed",
    audience: str = "all",
) -> dict[str, Any]:
    """Send ``report`` (already redacted) to the agents the audience policy allows. Returns what went where.

    ``audience="all"`` sends to every agent passed in (inbox, wake-up address, and the event stream).
    ``audience="share_logs"`` keeps only agents the person let "Share my logs and recordings"
    (:func:`away.selftest.pccheck.may_receive`); the event stream is not used then, because every connected
    agent can read it, including ones outside the audience. The returned dict always carries
    ``skipped_agents``: the names the audience policy kept the report from.
    """
    if audience not in AUDIENCES:
        raise ValueError(f"unknown audience {audience!r}; want one of {AUDIENCES}")
    targets = list(agents)
    skipped: list[str] = []
    if audience == "share_logs":
        from away.selftest import pccheck  # deferred: pccheck imports this module the same lazy way

        kept: list[Any] = []
        for a in targets:
            if pccheck.may_receive(a):
                kept.append(a)
            else:
                skipped.append(str(getattr(a, "name", "?")))
        targets = kept
    out: dict[str, Any] = {"delivered": bool(targets), "agents": [], "skipped_agents": skipped}
    if events is not None and audience == "all":
        with contextlib.suppress(Exception):
            ev = events.publish(event, compact(report))
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
            payload = {"event": event, "agent": getattr(a, "name", ""), "report": report}

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
