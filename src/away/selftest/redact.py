"""Take private things out of a self-test report before it leaves the app (v0.1.24).

Removed: tokens and keys (Agent Link ``pa_``/``par_`` tokens, bearer headers, Cloudflare tunnel tokens, anything
that looks like a long secret), the Windows / macOS / Linux user name, and file paths outside the app's own
folders (``C:\\Users\\fern\\Videos\\x.mp4`` becomes ``<path>``; ``<data>/logs/away.log`` stays). Public link
addresses stay (agents need them); rendezvous keys never appear in a report in the first place.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import re
from pathlib import Path
from typing import Any

TOKEN_RES = (
    re.compile(r"\b(?:paway|par?)_[A-Za-z0-9_-]{8,}"),  # Agent Link access / refresh / persistent tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]+)?"),  # JWT / tunnel tokens
    re.compile(r"\b(?:ghp|gho|ghs|github_pat|sk|hf)_[A-Za-z0-9_]{16,}"),
    re.compile(r"(?i)\b(token|secret|password|key)(\s*[=:]\s*)[^\s,;\"']{6,}"),
)
WIN_PATH = re.compile(r"(?i)\b[A-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n,;)]*")
POSIX_PATH = re.compile(r"(?<![\w:/])/(?:Users|home|root|var|tmp|private|mnt|media|opt|etc)/[^\s,;\"')]*")


def _user_names() -> list[str]:
    names = set()
    for v in (os.environ.get("USERNAME"), os.environ.get("USER"), os.environ.get("LOGNAME")):
        if v:
            names.add(v)
    with contextlib.suppress(Exception):
        names.add(getpass.getuser())
    with contextlib.suppress(Exception):
        names.add(Path.home().name)
    return sorted(
        (n for n in names if n and len(n) >= 2 and n.lower() not in ("root", "user", "admin")), key=len, reverse=True
    )


class Redactor:
    def __init__(self, keep_dirs: list[Path] | None = None, users: list[str] | None = None) -> None:
        self.keep = []
        for d in keep_dirs or []:
            try:
                self.keep.append((str(Path(d)), self._label(d)))
            except Exception:
                continue
        self.users = _user_names() if users is None else [u for u in users if u]

    @staticmethod
    def _label(d: Path) -> str:
        return "<data>"

    def text(self, s: str) -> str:
        out = str(s)
        for rx in TOKEN_RES:
            out = rx.sub(
                lambda m: (m.group(1) + m.group(2) + "***") if m.lastindex and m.lastindex >= 2 else "***", out
            )
        for d, label in sorted(self.keep, key=lambda x: -len(x[0])):  # app folders become <data>
            for variant in {d, d.replace("\\", "/"), d.replace("/", "\\")}:
                if variant and len(variant) > 3:
                    out = out.replace(variant, label)
        out = WIN_PATH.sub(lambda m: m.group(0) if m.group(0).startswith("<data>") else "<path>", out)
        out = POSIX_PATH.sub("<path>", out)
        for u in self.users:
            out = re.sub(rf"(?i)(?<![A-Za-z0-9]){re.escape(u)}(?![A-Za-z0-9])", "<user>", out)
        return out

    def value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if isinstance(k, str) and re.search(r"(?i)token|secret|password|rdv_key|^key$", k):
                    out[k] = "***" if x else x
                else:
                    out[self.text(k) if isinstance(k, str) else k] = self.value(x)
            return out
        if isinstance(v, (list, tuple)):
            return [self.value(x) for x in v]
        return v


def redact_report(
    report: dict[str, Any], keep_dirs: list[Path] | None = None, users: list[str] | None = None
) -> dict[str, Any]:
    """A redacted copy of a report's public dict (the original is not changed)."""
    out: dict[str, Any] = Redactor(keep_dirs, users).value(report)
    out["redacted"] = True
    return out
