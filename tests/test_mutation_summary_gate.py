"""Mutation summary gate: missing reports and weak scores must fail loudly.

scripts/mutation.py --summary --fail-under 80 is the CI gate. These tests pin
the contract: with a threshold set, a missing report or any group below the
threshold fails; without a threshold the historical always-zero behavior holds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_mutation_module(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mutation_under_test", ROOT / "scripts" / "mutation.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path / ".testteeth-reports")
    return mod


def write_report(out_dir: Path, group: str, score: float, killed: int, total: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    survived = total - killed
    (out_dir / f"{group}.json").write_text(
        json.dumps(
            {
                "score": score,
                "killed": killed,
                "survived": survived,
                "timeouts": 0,
            }
        )
    )


def write_all_reports(mod, tmp_path, score=95.0, skip=(), weak=()):
    out = tmp_path / ".testteeth-reports"
    for g in mod.GROUPS:
        if g in skip:
            continue
        s = 60.0 if g in weak else score
        write_report(out, g, s, killed=int(s), total=100)


def test_summary_passes_when_all_groups_meet_threshold(tmp_path, monkeypatch, capsys):
    mod = load_mutation_module(tmp_path, monkeypatch)
    write_all_reports(mod, tmp_path, score=95.0)
    assert mod.summary(fail_under=80) == 0
    out = capsys.readouterr().out
    assert "OVERALL" in out
    assert "missing" not in out


def test_summary_fails_on_missing_report(tmp_path, monkeypatch, capsys):
    mod = load_mutation_module(tmp_path, monkeypatch)
    first = next(iter(mod.GROUPS))
    write_all_reports(mod, tmp_path, score=95.0, skip={first})
    assert mod.summary(fail_under=80) == 1
    out = capsys.readouterr().out
    assert "missing mutation reports" in out
    assert first in out


def test_summary_fails_on_weak_group(tmp_path, monkeypatch, capsys):
    mod = load_mutation_module(tmp_path, monkeypatch)
    groups = list(mod.GROUPS)
    write_all_reports(mod, tmp_path, score=95.0, weak={groups[3]})
    assert mod.summary(fail_under=80) == 1
    out = capsys.readouterr().out
    assert "below --fail-under 80.0%" in out
    assert groups[3] in out


def test_summary_treats_corrupt_report_as_missing(tmp_path, monkeypatch, capsys):
    mod = load_mutation_module(tmp_path, monkeypatch)
    write_all_reports(mod, tmp_path, score=95.0)
    first = next(iter(mod.GROUPS))
    (tmp_path / ".testteeth-reports" / f"{first}.json").write_text("{not json")
    assert mod.summary(fail_under=80) == 1
    out = capsys.readouterr().out
    assert "unreadable mutation report" in out
    assert "missing mutation reports" in out


def test_summary_without_threshold_keeps_historical_behavior(tmp_path, monkeypatch):
    # No --fail-under: missing reports and weak scores still aggregate to 0,
    # preserving the pre-gate local behavior.
    mod = load_mutation_module(tmp_path, monkeypatch)
    groups = list(mod.GROUPS)
    write_all_reports(mod, tmp_path, score=60.0, skip={groups[0]})
    assert mod.summary() == 0


def test_fail_under_rejects_non_numeric_threshold(monkeypatch):
    import runpy
    import sys

    with pytest.raises(SystemExit) as exc:
        monkeypatch.setattr(sys, "argv", ["mutation.py", "--summary", "--fail-under", "abc"])
        runpy.run_path(str(ROOT / "scripts" / "mutation.py"), run_name="__main__")
    assert "numeric threshold" in str(exc.value.code)
