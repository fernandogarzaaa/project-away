"""Command line for Agent Link (registered by :func:`add_parsers`, so ``away/cli.py`` only gains two lines).

* ``away agent status|start|stop|pair|agents|revoke|disconnect|connect|snippets|prefs|changes|undo|github-token``
  (aliases ``away link`` and ``away remote``)
* ``away mcp [--scopes read,tune] [--name NAME]`` - the stdio MCP server local agent apps launch
* ``away replay FILE [--model] [--json]`` - re-run a recording through the gesture pipeline headless
* ``away diagnose bundle`` - write a diagnostics bundle
* ``away agent-ui ...`` - hidden: the on-screen prompt / toast / border windows the app starts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from away.agentlink import perms


def _wait_result(rid: str, timeout: float = 60.0) -> dict[str, Any] | None:
    from away.agentlink.host import read_state

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        res = (read_state().get("results") or {}).get(rid)
        if res is not None:
            return dict(res)
        time.sleep(0.3)
    return None


def _print_pairing(p: dict[str, Any]) -> None:
    from away.agentlink import qr

    print(f"\nPairing link (works once, for 10 minutes):\n  {p['url']}\n")
    print(f"6-digit code: {p['pin'][:3]} {p['pin'][3:]}")
    print("Permissions asked: " + ", ".join(perms.describe(p.get("scopes") or [])))
    art = qr.ascii_art(p["url"])
    if art:
        print("\n" + art + "\n")
    print("Paste the link into your AI app as a custom connector (MCP URL). You will be asked to Allow it here.")


def cmd_agent(args: argparse.Namespace) -> int:
    from away.agentlink import clients, host

    action = args.action
    alive = host.host_alive()
    if action == "status":
        st = host.read_state() if alive else {}
        if not alive:
            print("Project Away is not running (Agent Link lives in the running app). Start it, or `away agent start`.")
            return 1
        agents = st.get("agents") or []
        print(
            f"Agent Link: {'on' if (st.get('server') or {}).get('running') else 'idle'}"
            + (f", secure link {st['link']['url']}" if (st.get("link") or {}).get("url") else "")
        )
        for a in agents:
            mark = "\u2713 connected" if a.get("connected") else "not connected right now"
            print(f"  {a['name']} ({a['kind']}, {mark}): {', '.join(a['permissions'])}  [id {a['id']}]")
        if not agents:
            print("  no agents connected")
        ctl = st.get("control") or {}
        if ctl.get("active"):
            print(f"  {ctl['agent']} is using the computer" + (" (paused)" if ctl.get("paused") else ""))
        return 0
    if action in ("start", "pair"):
        scopes = perms.to_list(perms.normalize(args.scopes))
        remote = not args.local
        if alive:
            rid = host.send_request("pair" if action == "pair" else "serve", scopes=scopes, remote=remote)
            res = _wait_result(rid, 90.0)
            if res is None:
                print("error: Project Away did not answer", file=sys.stderr)
                return 1
            if not res.get("ok"):
                print(f"error: {res.get('error')}", file=sys.stderr)
                return 1
            if action == "pair":
                _print_pairing(res["result"])
            else:
                print(f"Agent Link is on (local port {res['result']['port']}).")
            return 0
        print(
            "Project Away is not running: starting Agent Link here (diagnostics and settings only, no live "
            "camera). Press Ctrl+C to stop.",
            flush=True,
        )
        h = host.AgentLinkHost(None)
        h.start()
        try:
            if action == "pair":
                _print_pairing(h.start_pairing(scopes, remote=remote))
            else:
                print(f"local port {h.ensure_server()}")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            h.stop()
        return 0
    if action in ("stop", "disconnect"):
        if not alive:
            print("Project Away is not running; nothing is connected.")
            return 0
        rid = host.send_request("disconnect_all")
        res = _wait_result(rid, 15.0) or {}
        print(f"Disconnected {(res.get('result') or {}).get('disconnected', 0)} agent(s); the secure link is closed.")
        return 0
    if action == "agents":
        for a in (host.read_state().get("agents") or []) if alive else []:
            print(json.dumps(a))
        return 0
    if action == "revoke":
        if not args.value:
            print("usage: away agent revoke AGENT_ID (see `away agent status`)", file=sys.stderr)
            return 2
        res = _wait_result(host.send_request("revoke", agent_id=args.value), 15.0) or {}
        print("revoked" if (res.get("result") or {}).get("revoked") else "no such agent")
        return 0
    if action == "connect":
        key = {"claude": "claude_desktop", "claude-desktop": "claude_desktop", "cursor": "cursor"}.get(
            str(args.value or "").lower()
        )
        if key is None:
            print("usage: away agent connect claude|cursor [--scopes ...] [--yes]", file=sys.stderr)
            return 2
        app = clients.APPS[key]
        path = clients.config_path(key)
        if path is None:
            return 2
        if not clients.installed(key):
            print(f"{app.name} isn't installed. Get it: {app.download}")
            return 1
        entry = clients.server_entry(perms.to_list(perms.normalize(args.scopes)))
        new, _old = clients.plan(path, entry)
        print(f"This will add Project Away to {path}:\n{new}")
        if not args.yes:
            ans = input("Write it (a backup is kept)? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("nothing changed")
                return 1
        backup = clients.write(path, entry)
        print(f"done{f' (backup: {backup.name})' if backup else ''}. {app.steps}")
        return 0
    if action == "snippets":
        for title, text in clients.snippets(perms.to_list(perms.normalize(args.scopes))).items():
            print(f"--- {title}\n{text}\n")
        return 0
    if action == "prefs":
        from away import paths
        from away.agentlink.prefs import Preferences

        prefs = Preferences(host.base_dir(paths.data_dir()) / "preferences.md")
        if args.value:
            prefs.add(" ".join([args.value, *args.rest]))
        print(prefs.text())
        return 0
    if action == "changes":
        from away import paths
        from away.agentlink.changes import ChangeManager

        for e in ChangeManager(host.base_dir(paths.data_dir())).history(30):
            print(
                f"{time.strftime('%m-%d %H:%M', time.localtime(e['ts']))} {e['who']}: {e['text']}"
                + (" (undone)" if e["undone"] else "")
                + f"  [{e['batch']}]"
            )
        return 0
    if action == "undo":
        from away import paths
        from away.agentlink.changes import ChangeManager

        cm = ChangeManager(host.base_dir(paths.data_dir()))
        what = args.value or "last"
        res = cm.undo_last("you") if what == "last" else cm.undo_batch(what, "you")
        print("; ".join(c["text"] for c in res["changes"]))
        return 0
    if action == "github-token":
        from away.agentlink.detectors import KEYRING_SERVICE, KEYRING_USER

        try:
            import keyring
        except Exception:
            print("error: the 'keyring' package is not installed (pip install keyring)", file=sys.stderr)
            return 1
        if args.value == "clear":
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
            print("removed")
            return 0
        import getpass

        tok = getpass.getpass("GitHub token (fine-grained, Issues: write on the repo): ").strip()
        if not tok:
            return 1
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, tok)
        print("saved in the OS keyring; turn on agent_link.github_issues to use it")
        return 0
    print(f"unknown action {action}", file=sys.stderr)
    return 2


def cmd_mcp(args: argparse.Namespace) -> int:
    from away.agentlink.bridge import serve_stdio

    return serve_stdio(args.scopes, args.name)


def cmd_replay(args: argparse.Namespace) -> int:
    from away.agentlink import replay

    res = replay.replay_file(Path(args.file), use_model=args.model)
    s = res.summary()
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        print(f"{s['frames']} frames replayed (digest {s['digest']})")
        for k, v in sorted(s["events"].items()):
            print(f"  {k:<24} {v}")
        print("  mouse: " + ", ".join(f"{k} x{v}" for k, v in sorted(s["mouse"].items())))
        if res.recorded_events:
            print(f"  (the live session logged {len(res.recorded_events)} gesture events)")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from away import paths
    from away.agentlink import bundle, host

    if host.host_alive():
        res = _wait_result(host.send_request("bundle"), 60.0)
        if res and res.get("ok"):
            print(f"saved {res['result']['path']} (in {paths.data_dir()})")
            return 0
    p = bundle.build(paths.data_dir(), note="away diagnose bundle (app not running: no recording)")
    print(f"saved {p}")
    return 0


def cmd_agent_ui(args: argparse.Namespace) -> int:
    from away.agentlink import ui_helper

    return ui_helper.main(args.kind, Path(args.spec))


def add_parsers(sub: Any) -> None:
    ag = sub.add_parser(
        "agent",
        aliases=["link", "remote"],
        help="connect AI agents (Claude, ChatGPT, Cursor...): status | pair | connect claude|cursor | disconnect",
    )
    ag.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=[
            "status",
            "start",
            "stop",
            "pair",
            "agents",
            "revoke",
            "disconnect",
            "connect",
            "snippets",
            "prefs",
            "changes",
            "undo",
            "github-token",
        ],
    )
    ag.add_argument("value", nargs="?")
    ag.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
    ag.add_argument(
        "--scopes",
        default="read,diagnose",
        help=f"permissions to ask for, comma-separated ({', '.join(perms.ALL)}); default read,diagnose",
    )
    ag.add_argument("--local", action="store_true", help="pair: no secure internet link (agents on this PC only)")
    ag.add_argument("--yes", "-y", action="store_true")
    ag.set_defaults(fn=cmd_agent)

    mc = sub.add_parser("mcp", help="MCP server over stdio for AI apps on this PC (Claude Desktop, Cursor...)")
    mc.add_argument("--scopes", default="read,diagnose")
    mc.add_argument("--name", help="agent name shown on the approval prompt (default: from the app)")
    mc.set_defaults(fn=cmd_mcp)

    rp = sub.add_parser("replay", help="re-run a recorded session (.awayrec) through the gesture pipeline headless")
    rp.add_argument("file")
    rp.add_argument("--model", action="store_true", help="also run the learned gesture model")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_replay)

    dg = sub.add_parser("diagnose", help="bundle: zip logs, redacted config, camera checks and the last 30 s")
    dg.add_argument("what", choices=["bundle"])
    dg.set_defaults(fn=cmd_diagnose)

    ui = sub.add_parser("agent-ui")  # hidden: started by the app
    ui.add_argument("kind", choices=["prompt", "toast", "flash", "control"])
    ui.add_argument("spec")
    ui.set_defaults(fn=cmd_agent_ui)
