"""``python -m away.liveagent`` - try the live agent from a terminal (no app, no microphone).

python -m away.liveagent status                 what is set up (never prints keys)
python -m away.liveagent free                   the "Use free models" button: get opencode, check, turn on
python -m away.liveagent models                 free opencode models right now
python -m away.liveagent ask "what is 15% of 80"   one chat turn, printed sentence by sentence with timings
python -m away.liveagent jev                    check Jev (free decisions model) answers
python -m away.liveagent update-engine          update Away's copy of opencode from its releases
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m away.liveagent", description="Project Away live agent")
    ap.add_argument("--data-dir", type=Path, default=None, help="use this data folder instead of Away's own")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("free")
    sub.add_parser("models")
    sub.add_parser("jev")
    sub.add_parser("update-engine")
    ask = sub.add_parser("ask")
    ask.add_argument("text")
    ask.add_argument("--model", default="", help='"provider/model" for this turn only')
    a = ap.parse_args(argv)

    from away.liveagent import opencode, runtime
    from away.liveagent import settings as st

    rt = runtime.Runtime(a.data_dir)
    path = st.path(rt.data_dir)
    s = st.load(path)
    rt._settings_fn = lambda: s
    try:
        if a.cmd == "status":
            print(json.dumps(rt.status(s), indent=2, default=str))
        elif a.cmd == "free":

            def progress(msg: str, done: int, total: int) -> None:
                extra = f" {done / 1e6:.1f}/{total / 1e6:.1f} MB" if total else ""
                print(f"  {msg}{extra}", file=sys.stderr)

            print(runtime.use_free_models(rt, s, progress))
            st.save(s, path)
        elif a.cmd == "models":
            with_server = opencode.installed_engine(rt.data_dir) is not None or opencode.find_cli() is not None
            if with_server:
                rt.opencode_server()
            for m in rt.free_models(refresh=True):
                print(f"{m.ref:55} {m.label():40} vision={m.vision} reasoning={m.reasoning}")
        elif a.cmd == "jev":
            from away.liveagent import jev

            print(jev.probe(rt.client))
            d = rt.decider(s)
            for q in ("open notepad and type hello", "what's the capital of France", "write a full market report"):
                ans = d.route(q)
                print(f"  route {q!r}: {ans.value} ({ans.source}, {ans.confidence:.2f})")
            r = d.risk("click", "Send")
            print(f"  risk click 'Send': {r.value} ({r.source})")
        elif a.cmd == "update-engine":
            print(json.dumps(opencode.check_update(rt.data_dir)))
        elif a.cmd == "ask":
            from away.liveagent.agent import chat_system
            from away.liveagent.router import CallRecord
            from away.liveagent.textstream import sentences

            if a.model:
                s.preferred = a.model
            f = rt.funnel(s)
            rec = CallRecord("chat")
            t0 = time.monotonic()
            msgs = [{"role": "system", "content": chat_system(s.name)}, {"role": "user", "content": a.text}]
            first = None
            from away.liveagent.router import NoModel

            try:
                for sentence in sentences(f.stream("chat", msgs, rec, max_tokens=300)):
                    if first is None:
                        first = time.monotonic() - t0
                    print(f"[{time.monotonic() - t0:5.2f}s] {sentence}")
            except NoModel as exc:
                print(f"No model answered: {exc}", file=sys.stderr)
                for ref, outcome in exc.attempts:
                    print(f"  {ref}: {outcome}", file=sys.stderr)
                return 1
            print(
                json.dumps(
                    {
                        "model": rec.chosen,
                        "first_token_s": rec.first_token_s,
                        "first_sentence_s": first,
                        "total_s": rec.total_s,
                        "attempts": rec.attempts,
                        "cost_usd": rec.cost_usd,
                    },
                    default=str,
                ),
                file=sys.stderr,
            )
    finally:
        rt.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
