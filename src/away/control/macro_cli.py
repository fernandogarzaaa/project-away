"""``away shortcut ...`` (alias ``away macro``): list, run, record, bind, import and export shortcuts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from away.control.macros import SPEEDS, MacroError, name_from_title

ACTIONS = ("list", "show", "run", "record", "rename", "delete", "bind", "unbind", "export", "import")


def add_parser(sub: Any) -> None:
    sp = sub.add_parser(
        "shortcut",
        aliases=["macro"],
        help="list | show | run | record | bind | unbind | rename | delete | export | import recorded shortcuts",
    )
    sp.add_argument("action", choices=ACTIONS)
    sp.add_argument("args", nargs="*", help="shortcut id(s), a new title, or a file")
    sp.add_argument("--config")
    sp.add_argument("--speed", choices=SPEEDS, help="run: override the shortcut's speed")
    sp.add_argument("--seconds", type=float, help="record: stop after this many seconds (default: press Enter)")
    sp.add_argument("--name", help="record: save under this name (default: Shortcut N)")
    sp.add_argument("--say", help="bind: the phrase that runs it")
    sp.add_argument("--gesture", help="bind: the gesture that runs it (e.g. swipe_left, palm_tap)")
    sp.add_argument("--replace", action="store_true", help="bind: take the phrase/gesture over from another action")
    sp.add_argument("--include-private", action="store_true", help="export shortcuts that type text you entered")
    sp.add_argument("--overwrite", action="store_true", help="import: replace shortcuts with the same id")
    sp.set_defaults(fn=cmd_shortcut)


def _fail(msg: object) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def _model(args: argparse.Namespace, service: Any = None) -> Any:
    from away.app.macro_panel import ShortcutsModel
    from away.app.macro_service import MacroService
    from away.app.settings_gui import SettingsModel

    sm = SettingsModel(path=Path(args.config) if args.config else None)
    return ShortcutsModel(sm, service or MacroService(sm.cfg.macros))


def cmd_shortcut(args: argparse.Namespace, service: Any = None, wait_stop: Any = None) -> int:
    try:
        m = _model(args, service)
    except ValueError as exc:
        return _fail(exc)
    a, rest = args.action, list(args.args)
    store = m.store
    try:
        if a == "list":
            rows = m.rows()
            if not rows:
                print("no shortcuts yet: record one with `away shortcut record` or Settings > Shortcuts")
            for r in rows:
                lock = "  [private: types text]" if r.sensitive else ""
                print(f"{r.name:24} {r.title:24} {r.steps:3} steps   {r.runs_when}{lock}")
            return 0
        if a == "export":
            if not rest:
                return _fail("usage: away shortcut export FILE [ID ...] [--include-private]")
            names = rest[1:] or store.names()
            p = m.export(names, rest[0], include_private=args.include_private)
            if p is not None:
                return _fail(p.message + (" (add --include-private)" if p.fix == "export_private" else ""))
            print(f"exported {len(names)} shortcut(s) to {rest[0]}")
            return 0
        if a == "import":
            if len(rest) != 1:
                return _fail("usage: away shortcut import FILE [--overwrite]")
            got = store.import_file(rest[0], overwrite=args.overwrite)
            print("imported: " + ", ".join(x.name for x in got) + " (bind each with `away shortcut bind`)")
            return 0
        if a == "record":
            return _record(m, args, wait_stop)
        if not rest:
            return _fail(f"usage: away shortcut {a} ID")
        name = rest[0]
        if a == "show":
            mac = store.load(name)
            print(f"{mac.label} ({mac.name}) speed {mac.speed}, recorded in {mac.app or 'unknown app'}")
            print(f"runs when: {m.runs_when(name)}" + ("  [private: types text]" if mac.sensitive else ""))
            for i, s in enumerate(mac.steps, 1):
                print(f"  {i:3}. {s.describe()}" + (f"  (after {s.delay:g} s)" if s.delay else ""))
            return 0
        if a == "run":
            res = m.service.play(name, wait=True, speed=args.speed)
            if res is None or res.ok:
                print(res.message if res is not None and res.message else "done")
                return 0
            return _fail(res.message)
        if a == "rename":
            if len(rest) != 2:
                return _fail('usage: away shortcut rename ID "New name"')
            p = m.rename(name, rest[1])
        elif a == "delete":
            if not store.exists(name):
                return _fail(f"no shortcut {name!r}")
            p = m.delete(name)
        elif a == "unbind":
            p = m.unbind(name)
        else:  # bind
            if not store.exists(name):
                return _fail(f"no shortcut {name!r} (see `away shortcut list`)")
            if bool(args.say) == bool(args.gesture):
                return _fail('usage: away shortcut bind ID --say "open mail" | --gesture palm_tap [--replace]')
            kind, value = ("phrase", args.say) if args.say else ("gesture", args.gesture)
            p = m.bind(name, kind, value, replace=args.replace)
            if p is not None and p.fix == "replace_binding":
                return _fail(p.message + " (add --replace)")
        if p is not None:
            return _fail(p.message)
        print("ok")
        return 0
    except MacroError as exc:
        return _fail(exc)


def _record(m: Any, args: argparse.Namespace, wait_stop: Any) -> int:  # the waiting is injectable for tests
    err = m.start_recording()
    if err is not None:
        return _fail(err.message)
    if wait_stop is not None:
        wait_stop()
    elif args.seconds:
        print(f"recording for {args.seconds:g} s: do your steps now...", flush=True)
        time.sleep(args.seconds)
    else:  # pragma: no cover - interactive
        input("recording: do your steps, then come back here and press Enter... ")
    draft, p = m.stop_recording()
    if p is not None:
        return _fail(p.message)
    note = m.recording_note(draft)
    title = args.name or m.suggested_title()
    p = m.save_new(title)
    if p is not None:
        return _fail(p.message)
    print(f"saved {name_from_title(title)} ({len(draft.steps)} steps)" + (f". {note}" if note else ""))
    print('bind it: away shortcut bind ID --say "..." | --gesture palm_tap')
    return 0
