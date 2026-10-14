"""The self-test's results (v0.1.24): pass / warn / fail per check, each in plain words, with a fix button.

A :class:`Result` is one line of the report ("Camera "JoyCam": works, 24 fps, MJPG 640x480"). ``fix`` names an
action the app can do for you (:mod:`away.selftest.fixes`): the button's words are ``fix.label`` ("Close NVIDIA
Broadcast", "Try again", "Use the processor instead").
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
STATUSES = (PASS, WARN, FAIL, SKIP)
ICONS = {PASS: "\u2714", WARN: "!", FAIL: "\u2716", SKIP: "-"}
WORDS = {PASS: "OK", WARN: "Needs a look", FAIL: "Not working", SKIP: "Skipped"}


@dataclass(frozen=True)
class Fix:
    action: str  # "retry_cameras", "close_app:<path>", "set:hands.compute=cpu", "open:settings/cameras" ...
    label: str  # the button: "Close NVIDIA Broadcast"

    def public(self) -> dict[str, str]:
        return {"action": self.action, "label": self.label}


@dataclass
class Result:
    id: str  # "camera:JoyCam", "tracking", "mic" ...
    title: str  # "Camera \u201cJoyCam\u201d"
    status: str  # pass | warn | fail | skip
    summary: str  # one plain sentence
    details: dict[str, Any] = field(default_factory=dict)  # numbers for agents and "Show details"
    fix: Fix | None = None
    ms: float = 0.0  # how long the check took

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["fix"] = self.fix.public() if self.fix else None
        d["ms"] = round(float(self.ms), 1)
        return d

    def line(self) -> str:
        fix = f"  [{self.fix.label}]" if self.fix else ""
        return f"{ICONS.get(self.status, '?')} {self.title}: {self.summary}{fix}"


def overall(results: list[Result]) -> str:
    sts = {r.status for r in results}
    if FAIL in sts:
        return FAIL
    if WARN in sts:
        return WARN
    return PASS if PASS in sts else SKIP


@dataclass
class Report:
    version: str
    platform: str
    started: float
    finished: float = 0.0
    results: list[Result] = field(default_factory=list)
    trigger: str = "you"  # you | tray | settings | cli | <agent name>
    id: str = ""
    kind: str = "away.selftest"  # v0.1.24: "away.pccheck" for Check my PC

    @property
    def status(self) -> str:
        return overall(self.results)

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for r in self.results if r.status == s) for s in STATUSES}

    def headline(self) -> str:
        c = self.counts()
        if c[FAIL]:
            return f"{c[FAIL]} thing{'s' if c[FAIL] != 1 else ''} not working, {c[WARN]} to look at, {c[PASS]} OK"
        if c[WARN]:
            return f"Everything works; {c[WARN]} thing{'s' if c[WARN] != 1 else ''} to look at ({c[PASS]} OK)"
        return f"Everything works ({c[PASS]} checks OK)"

    def public(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "version": self.version,
            "platform": self.platform,
            "started": self.started,
            "finished": self.finished,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.finished or self.started)),
            "trigger": self.trigger,
            "status": self.status,
            "headline": self.headline(),
            "counts": self.counts(),
            "results": [r.public() for r in self.results],
        }

    def text(self) -> str:
        return report_text(self.public())


ORDER = {FAIL: 0, WARN: 1, PASS: 2, SKIP: 3}


def report_text(p: dict[str, Any]) -> str:
    """A (redacted) report dict as plain text: "Copy report", "Save as file" and the CLI."""
    lines = [
        f"Project Away self-test {p.get('time', '')} (v{p.get('version', '?')}, {p.get('platform', '')})",
        str(p.get("headline", "")),
        "",
    ]
    for r in sorted(p.get("results") or [], key=lambda r: ORDER.get(r.get("status"), 9)):
        fix = f"  [{r['fix']['label']}]" if r.get("fix") else ""
        lines.append(f"{ICONS.get(r.get('status'), '?')} {r.get('title')}: {r.get('summary')}{fix}")
    if p.get("delivered_to"):
        lines += ["", "Sent to: " + ", ".join(p["delivered_to"])]
    return "\n".join(lines)
