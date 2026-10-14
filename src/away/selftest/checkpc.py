"""``away check`` (v0.1.25): Check my PC from a terminal, the tray or Settings > Getting started.

* ``away check`` - the guided check in this terminal (press Enter when asked, answer y / n);
* ``away check --window`` - the same in a window with fix buttons, "Copy report" and "Send to my agents" (what the
  tray's "Check my PC" and the Getting started button open);
* ``away check --only cameras,voices`` - some parts only; ``--send`` - also send the report to the agents that are
  allowed to receive it.

The report is saved (redacted) in ``<data>/check/``: ``check-<time>.json``, ``.html``, ``.txt``, ``latest.json``
and ``verified.json``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def make_env(prompt: Callable[[str], Any] | None, confirm: Callable[[str], bool | None] | None) -> tuple[Any, bool]:
    """(env, in_app): measurements inside the running app when there is one, otherwise right here."""
    from away.agentlink import host

    if host.host_alive():
        from away.selftest import cli as stcli
        from away.selftest.pcenv import RemotePcEnv

        def ask_host(action: str, timeout: float = 20.0, **data: Any) -> dict[str, Any] | None:
            return stcli._ask(action, timeout, **data)

        return RemotePcEnv(ask_host, prompt, confirm), True
    from away.selftest.pcenv import RealPcEnv

    return RealPcEnv(interactive=True, ask=prompt, confirm=confirm), False


def run_check(
    prompt: Callable[[str], Any] | None,
    confirm: Callable[[str], bool | None] | None,
    only: list[str] | None = None,
    trigger: str = "you",
    progress: Any = None,
    data_dir: Path | None = None,
    env: Any = None,
) -> dict[str, Any]:
    from away import paths
    from away.selftest import pccheck

    in_app = False
    if env is None:
        env, in_app = make_env(prompt, confirm)
    dd = Path(data_dir or paths.data_dir())
    rep = pccheck.run(env, only=only, trigger=trigger, progress=progress)
    public, files = pccheck.finish(rep, keep_dirs=[dd], data_dir=dd)
    return {"report": public, "files": {k: str(v) for k, v in files.items()}, "in_app": in_app}


def send(report: dict[str, Any]) -> dict[str, Any]:
    """Send to the allowed agents through the running app (nobody to send to when it is not running)."""
    from away.agentlink import host

    if not host.host_alive():
        return {
            "delivered": False,
            "local_only": True,
            "note": "Project Away is not running, so no agent is connected.",
        }
    from away.selftest import cli as stcli

    res = stcli._ask("pccheck_share", 30.0, report=report)
    return res or {"delivered": False, "note": "Project Away did not answer."}


def sent_text(res: dict[str, Any]) -> str:
    to = [str(a.get("agent")) for a in res.get("agents") or []]
    skipped = res.get("skipped_agents") or []
    if to:
        msg = "Sent to " + ", ".join(to) + "."
    else:
        msg = str(res.get("note") or "No connected agent is allowed to receive reports.")
    if skipped:
        msg += (
            " Not sent to "
            + ", ".join(skipped)
            + " (they need \u201cShare my logs\u201d or \u201cRun camera and speed checks\u201d)."
        )
    return msg


def main(args: argparse.Namespace) -> int:
    from away.selftest import pccheck

    only = [c.strip() for c in str(getattr(args, "only", "") or "").split(",") if c.strip()] or None
    bad = [c for c in only or [] if c not in pccheck.PC_CHECK_IDS]
    if bad:
        print(f"unknown part(s): {', '.join(bad)}; choose from {', '.join(pccheck.PC_CHECK_IDS)}", file=sys.stderr)
        return 2
    if getattr(args, "window", False):
        from away.selftest import pcui

        return pcui.run_window(only, trigger=str(getattr(args, "trigger", "") or "you"))
    tty = bool(sys.stdin and sys.stdin.isatty())

    def prompt(text: str) -> None:
        print("\n>> " + text, flush=True)
        if tty:
            input("   Press Enter when ready... ")

    def confirm(question: str) -> bool | None:
        if not tty:
            return None
        ans = input(f"?? {question} [y/n] ").strip().lower()
        return True if ans.startswith("y") else (False if ans.startswith("n") else None)

    print("Check my PC: about 2 minutes. You will be asked to raise your hand, pinch and say a few words.", flush=True)
    got = run_check(
        prompt,
        confirm,
        only,
        "you (command line)",
        lambda label, i, n: print(f"[{i + 1}/{n}] {label}" if i < n else "", flush=True),
    )
    rep = got["report"]
    print("\n" + pccheck.text(rep))
    print(f"\nSaved: {got['files'].get('html')}\nVerified list: {got['files'].get('verified')}")
    if getattr(args, "send", False):
        print(sent_text(send(rep)))
    return 0 if rep.get("verdicts", {}).get("fail", 0) == 0 else 1


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--window", action="store_true", help="run it in a window with fix buttons")
    p.add_argument("--only", help="some parts only, comma-separated (cameras,tracking,voices,echo,link,...)")
    p.add_argument("--send", action="store_true", help="also send the report to agents allowed to receive it")
    p.add_argument("--trigger", help=argparse.SUPPRESS)
