"""Measure Smart cleanup on this PC: latency per sentence and how often the guard accepts the model's answer.

    pip install "project-away[cleanup]"
    python scripts/bench_cleanup.py path/to/model.gguf [--gpu-layers -1] [--timeout-ms 1500]

Prints one line per sample (method, ms, output) and a summary. Nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from away.voice.cleanup import RuleCleaner
from away.voice.llm_backends import LlamaCppBackend
from away.voice.smart_cleanup import SmartCleaner

SAMPLES = [
    ("slack.exe", "hey are we still on for lunch tomorrow"),
    ("slack.exe", "um yeah i think the the build is broken can you check it"),
    ("OUTLOOK.EXE", "hi maria thanks for sending the contract i will review it and get back to you by friday"),
    ("OUTLOOK.EXE", "please find the updated budget attached let me know if you have any questions"),
    ("WINWORD.EXE", "the results show a twelve percent increase in retention over the last quarter"),
    ("WINWORD.EXE", "we should meet on monday no wait make it tuesday at three to discuss the roadmap"),
    ("notepad.exe", "what time is the standup tomorrow"),
    ("notepad.exe", "remind me to call the dentist and pick up groceries after work"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--timeout-ms", type=int, default=30000, help="high by default to measure the model itself")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    backend = LlamaCppBackend(args.model, gpu_layers=args.gpu_layers, threads=args.threads or None)
    t = time.perf_counter()
    backend.load()
    print(f"load: {time.perf_counter() - t:.2f} s")
    proc = {"p": None}
    sc = SmartCleaner(RuleCleaner(), "llm", None, backend, args.timeout_ms, process=lambda: proc["p"])
    sc.warm_up()
    times, methods = [], []
    for p, text in SAMPLES:
        proc["p"] = p
        out = sc.clean(text)
        times.append(sc.stats.latency_ms)
        methods.append(sc.stats.method)
        why = f" ({sc.stats.rejected or sc.stats.error})" if sc.stats.method == "fallback" else ""
        print(f"{p:12} {sc.stats.method:8} {sc.stats.latency_ms:7.0f} ms  {out}{why}")
    print(
        f"median {statistics.median(times):.0f} ms, max {max(times):.0f} ms; "
        f"model text used {methods.count('llm')}/{len(methods)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
