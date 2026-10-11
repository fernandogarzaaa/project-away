"""Release-workflow gate for the frozen installer builds (stdlib only, runs on every OS runner).

    python scripts/check_selftest.py selftest.json [...]          # every bundled feature loaded (ok: true)
    python scripts/check_selftest.py --models DIR --asr tiny        # `setup --yes` downloaded the models
    python scripts/check_selftest.py --max-size-mb 1900 FILE [...]  # GitHub release assets must be < 2 GB

Exit code 1 on any failure, so a release never ships with a feature silently missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MEDIAPIPE = ("hand_landmarker.task", "face_landmarker.task")


def check_reports(files: list[str], allow_skip: bool = False) -> list[str]:
    errors = []
    for f in files:
        rep = json.loads(Path(f).read_text(encoding="utf-8"))
        if not rep.get("frozen"):
            errors.append(f"{f}: not produced by the frozen app (frozen=false)")
        for name, r in rep.get("checks", {}).items():
            ok = r.get("ok")
            status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
            print(f"  {status:4} {name:15} {r.get('feature', '')}: {r.get('detail', '')}")
            if ok is False or (ok is None and not allow_skip):
                errors.append(f"{f}: {name} ({r.get('feature')}) {status}: {r.get('detail')}")
    return errors


def check_models(models_dir: str, asr: str | None) -> list[str]:
    d = Path(models_dir)
    want = [d / m for m in MEDIAPIPE]
    if asr:
        want.append(d / f"whisper-{asr}" / "model.bin")
    errors = [f"missing model: {p}" for p in want if not p.exists()]
    for p in want:
        if p.exists():
            print(f"  ok  {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return errors


def check_sizes(files: list[str], max_mb: float) -> list[str]:
    errors = []
    for f in files:
        mb = Path(f).stat().st_size / (1024 * 1024)
        print(f"  {mb:8.1f} MiB  {f}")
        if mb > max_mb:
            errors.append(f"{f} is {mb:.0f} MiB, over the {max_mb:.0f} MiB limit (GitHub release assets: 2 GiB)")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*")
    p.add_argument("--models")
    p.add_argument("--asr")
    p.add_argument("--max-size-mb", type=float)
    p.add_argument("--allow-skip", action="store_true", help="accept checks skipped for lack of a display")
    a = p.parse_args(argv)
    if a.max_size_mb is not None:
        errors = check_sizes(a.files, a.max_size_mb)
    elif a.models:
        errors = check_models(a.models, a.asr)
    else:
        errors = check_reports(a.files, a.allow_skip)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
