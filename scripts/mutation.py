"""Run testteeth per core module against its own test file(s) and aggregate.

    python scripts/mutation.py                 # all groups
    python scripts/mutation.py filters cursor  # selected groups
    python scripts/mutation.py --summary       # just aggregate existing reports
    python scripts/mutation.py --summary --fail-under 80  # fail if any report is missing or below 80
    python scripts/mutation.py --list-groups   # print the group list as JSON (for CI matrix)

Reports land in .testteeth-reports/<group>.json (git-ignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".testteeth-reports"
GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "filters": (["src/away/hands/filters.py"], ["tests/test_filters.py"]),
    "landmarks": (["src/away/hands/landmarks.py"], ["tests/test_landmarks.py"]),
    "gestures": (["src/away/hands/gestures.py"], ["tests/test_gestures.py"]),
    "cursor": (["src/away/hands/cursor.py"], ["tests/test_cursor.py"]),
    "cleanup": (
        ["src/away/voice/cleanup.py"],
        ["tests/test_cleanup.py", "tests/test_v0129_cleanup_mut.py"],
    ),
    "vad": (
        ["src/away/voice/vad.py"],
        ["tests/test_voice.py", "tests/test_v0129_vad_mut.py"],
    ),
    "dictation": (
        ["src/away/voice/dictation.py"],
        ["tests/test_voice.py", "tests/test_v0129_dictation_mut.py", "tests/test_v0129_mic_supervision.py"],
    ),
    "wifi_features": (["src/away/wifi/features.py"], ["tests/test_wifi.py"]),
    "wifi_classifier": (["src/away/wifi/classifier.py"], ["tests/test_wifi.py"]),
    "face": (
        [
            "src/away/face/speaking.py",
            "src/away/face/speaker.py",
            "src/away/face/gate.py",
            "src/away/face/commands.py",
            "src/away/face/tracking.py",
        ],
        ["tests/test_face.py", "tests/test_v0129_face_mut.py"],
    ),
    "fusion_events": (["src/away/fusion.py", "src/away/events.py"], ["tests/test_fusion_events.py"]),
    # v0.1.16
    "autoscroll": (
        ["src/away/hands/autoscroll.py"],
        ["tests/test_v0116_fixes.py", "tests/test_v019_features.py", "tests/test_v0129_autoscroll_mut.py"],
    ),
    "crashlog": (
        ["src/away/app/crashlog.py", "src/away/app/uiqueue.py"],
        ["tests/test_v0116_fixes.py", "tests/test_v0129_crashlog_mut.py"],
    ),
    # Settings > Tuning redesign
    "tunables": (["src/away/app/tunables.py"], ["tests/test_tuning_tab.py"]),
    "autotune": (["src/away/hands/autotune.py", "src/away/app/tuning_live.py"], ["tests/test_autotune.py"]),
}


# boundary/default tests written after the first mutation pass apply to every group
SHARED_TESTS = ["tests/test_spec_edges.py", "tests/test_spec_edges2.py"]


def run(group: str, workers: int = 2) -> None:
    paths, own = GROUPS[group]
    tests = own + SHARED_TESTS
    OUT.mkdir(exist_ok=True)
    cmd = [
        "testteeth", "run", *paths,
        "--pytest-args", " ".join([*tests, "-q", "-p", "no:cacheprovider", "-p", "no:warnings"]),
        "--workers", str(workers), "-q", "--json", str(OUT / f"{group}.json"),
    ]  # fmt: skip
    subprocess.run(cmd, cwd=ROOT, check=False)


def summary(fail_under: float = 0.0) -> int:
    # One denominator definition, matching testteeth's Report.score: every
    # mutant in the report counts, no_coverage included. The numerator is the
    # detected mutants (killed + timeouts). Percentage, ratio and gate all
    # derive from these same counts, so they can never describe different
    # fractions of the same report.
    total = detected = 0
    sums = {"killed": 0, "survived": 0, "no_coverage": 0, "timeouts": 0}
    rows = []
    for g in GROUPS:
        p = OUT / f"{g}.json"
        counts = None
        if p.exists():
            try:
                d = json.loads(p.read_text())
                counts = {
                    "killed": d["killed"],
                    "survived": d["survived"],
                    "no_coverage": d.get("no_coverage", 0),
                    "timeouts": d["timeouts"],
                }
                n = sum(counts.values())
                k = counts["killed"] + counts["timeouts"]
                pct = round(100.0 * k / n, 1) if n else None
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"warning: unreadable mutation report for {g}: {e}")
                counts = None
        if counts is None:
            rows.append((g, None, 0, 0, dict.fromkeys(sums, 0)))
            continue
        total += n
        detected += k
        for key in sums:
            sums[key] += counts[key]
        rows.append((g, pct, k, n, counts))
    for g, s, k, n, c in rows:
        print(
            f"{g:<16} {'n/a' if s is None else f'{s:5.1f}%'}  {k}/{n}"
            f"  killed={c['killed']} survived={c['survived']}"
            f" no_coverage={c['no_coverage']} timeouts={c['timeouts']}"
        )
    overall = round(100.0 * detected / total, 1) if total else 0.0
    print(
        f"{'OVERALL':<16} {overall:5.1f}%  {detected}/{total}"
        f"  killed={sums['killed']} survived={sums['survived']}"
        f" no_coverage={sums['no_coverage']} timeouts={sums['timeouts']}"
    )
    if fail_under > 0:
        missing = [g for g, s, *_ in rows if s is None]
        if missing:
            print(f"missing mutation reports: {', '.join(missing)}")
            return 1
        weak = [(g, s) for g, s, *_ in rows if s is not None and s < fail_under]
        if weak:
            details = ", ".join(f"{g} {s:.1f}%" for g, s in weak)
            print(f"below --fail-under {fail_under:.1f}%: {details}")
            return 1
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list-groups" in args:
        print(json.dumps(list(GROUPS)))
        raise SystemExit(0)
    fail_under = 0.0
    if "--fail-under" in args:
        i = args.index("--fail-under")
        try:
            fail_under = float(args[i + 1])
        except (IndexError, ValueError):
            raise SystemExit("--fail-under requires a numeric threshold") from None
        del args[i : i + 2]
    if "--summary" not in args:
        for grp in args or list(GROUPS):
            if grp not in GROUPS:
                raise SystemExit(f"unknown mutation group: {grp}")
            run(grp)
    raise SystemExit(summary(fail_under))
