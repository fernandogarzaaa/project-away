"""``away selftest`` (v0.1.24): the one-click self-test from a terminal, the tray or Settings.

* ``away selftest`` - runs every check and prints the report in plain words (inside the running app when it runs,
  so the cameras are not taken away from it; otherwise right here);
* ``away selftest --window`` - the same in a small window with fix buttons, "Copy report" and "Save as file"
  (what the tray's "Run self-test" and Settings' "Run self-test" open);
* ``away selftest --only cameras,mic`` - some checks only; ``--save FILE`` - write the redacted report (JSON);
* ``away selftest --json FILE`` / ``--bundle`` - the installer bundle check the release workflow runs (unchanged).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

POLL_S = 0.5
WAIT_S = 240.0


def _via_app(only: list[str] | None, trigger: str) -> dict[str, Any] | None:
    """Run inside the running app: start it there, wait for the report. None when the app is not running."""
    from away.agentlink import host

    if not host.host_alive():
        return None
    before = (_ask("selftest_state") or {}).get("latest") or {}
    started = _ask("selftest", only=only, trigger=trigger)
    if started is None:
        return None
    t_end = time.monotonic() + WAIT_S
    while time.monotonic() < t_end:
        time.sleep(POLL_S)
        st = _ask("selftest_state") or {}
        state, latest = st.get("state") or {}, st.get("latest") or {}
        if not state.get("running") and latest and latest.get("id") != before.get("id"):
            return {"report": latest, "in_app": True}
    return {"report": None, "in_app": True, "error": "the app did not finish the self-test in time"}


def _ask(action: str, timeout: float = 15.0, **data: Any) -> dict[str, Any] | None:
    from away.agentlink import host
    from away.agentlink.cli import _wait_result

    rid = host.send_request(action, **data)
    res = _wait_result(rid, timeout)
    if not res or not res.get("ok"):
        return None
    out = res.get("result")
    return out if isinstance(out, dict) else {"value": out}


def run_here(only: list[str] | None, trigger: str, ask: Any = None) -> dict[str, Any]:
    """Run in this process (the app is not running): nothing to deliver to, the report stays here."""
    from away import paths
    from away.selftest.env import RealEnv
    from away.selftest.service import SelftestService

    svc = SelftestService(
        paths.data_dir(), lambda interactive: RealEnv(interactive=interactive, ask=ask), background=False
    )
    return svc.run_now(trigger=trigger, only=only, interactive=ask is not None)


def fix_runner(in_app: bool) -> Any:
    def run(action: str) -> dict[str, Any]:
        if in_app:
            res = _ask("selftest_fix", timeout=60.0, fix=action)
            return res or {"ok": False, "message": "Project Away did not answer."}
        from away.selftest import fixes

        return fixes.apply(action)

    return run


def main(args: argparse.Namespace) -> int:
    if getattr(args, "json", None) or getattr(args, "bundle", False):  # the release workflow's bundle check
        from away.app import selftest as bundle

        return bundle.main(args.json)
    only = [c.strip() for c in str(args.only or "").split(",") if c.strip()] or None
    from away.selftest.checks import CHECK_IDS

    bad = [c for c in only or [] if c not in CHECK_IDS]
    if bad:
        print(f"unknown check(s): {', '.join(bad)}; choose from {', '.join(CHECK_IDS)}", file=sys.stderr)
        return 2
    if args.window:
        from away.selftest import ui

        return ui.run_window(only, trigger=str(args.trigger or "you"))
    print(
        "Running the self-test (cameras, tracking, pointer, microphone, voices, link, PC load, models)...", flush=True
    )
    got = _via_app(only, "you (command line)")
    if got is None:

        def ask(text: str) -> None:
            print(text, flush=True)

        got = run_here(only, "you (command line)", ask if sys.stdin and sys.stdin.isatty() else None)
        where = "here (Project Away is not running, so nothing was sent to agents)"
    else:
        where = "inside the running app"
    rep = got.get("report")
    if not rep:
        print(f"error: {got.get('error') or 'no report'}", file=sys.stderr)
        return 1
    from away.selftest.model import report_text

    print(report_text(rep))
    print(f"\n(ran {where}; the latest report is also at <data>/selftest/latest.json)")
    if args.save:
        Path(args.save).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        print(f"saved {args.save}")
    return 0 if rep.get("status") != "fail" else 1


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--window", action="store_true", help="show the report in a window with fix buttons")
    p.add_argument("--only", help="some checks only, comma-separated (cameras,tracking,gpu,glide,big_cursor,...)")
    p.add_argument("--save", metavar="FILE", help="also save the redacted report as JSON")
    p.add_argument("--trigger", help=argparse.SUPPRESS)
    p.add_argument("--json", metavar="FILE", help=argparse.SUPPRESS)  # installer bundle check (release workflow)
    p.add_argument("--bundle", action="store_true", help=argparse.SUPPRESS)
