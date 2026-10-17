"""Run testteeth per core module against its own test file(s) and aggregate.

    python scripts/mutation.py                 # all groups
    python scripts/mutation.py filters cursor  # selected groups
    python scripts/mutation.py --summary       # just aggregate existing reports
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
    "cleanup": (["src/away/voice/cleanup.py"], ["tests/test_cleanup.py"]),
    "vad": (["src/away/voice/vad.py"], ["tests/test_voice.py"]),
    "dictation": (["src/away/voice/dictation.py"], ["tests/test_voice.py"]),
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
        ["tests/test_face.py"],
    ),
    "fusion_events": (["src/away/fusion.py", "src/away/events.py"], ["tests/test_fusion_events.py"]),
    # v0.1.16
    "autoscroll": (["src/away/hands/autoscroll.py"], ["tests/test_v0116_fixes.py", "tests/test_v019_features.py"]),
    "crashlog": (["src/away/app/crashlog.py", "src/away/app/uiqueue.py"], ["tests/test_v0116_fixes.py"]),
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


def summary() -> int:
    total = killed = 0
    rows = []
    for g in GROUPS:
        p = OUT / f"{g}.json"
        if not p.exists():
            rows.append((g, None, 0, 0))
            continue
        d = json.loads(p.read_text())
        scored = d["killed"] + d["survived"] + d["timeouts"]
        k = d["killed"] + d["timeouts"]
        total += scored
        killed += k
        rows.append((g, d["score"], k, scored))
    for g, s, k, n in rows:
        print(f"{g:<16} {'n/a' if s is None else f'{s:5.1f}%'}  {k}/{n}")
    overall = 100.0 * killed / total if total else 0.0
    print(f"{'OVERALL':<16} {overall:5.1f}%  {killed}/{total}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list-groups" in args:
        print(json.dumps(list(GROUPS)))
        raise SystemExit(0)
    if "--summary" not in args:
        for grp in args or list(GROUPS):
            if grp not in GROUPS:
                raise SystemExit(f"unknown mutation group: {grp}")
            run(grp)
    raise SystemExit(summary())
