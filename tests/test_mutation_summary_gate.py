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


def write_report_counts(out_dir: Path, group: str, killed: int, survived: int, timeouts: int = 0, no_coverage: int = 0):
    # A report in the real testteeth shape: score is round(100 * detected /
    # total, 1) where detected = killed + timeouts and total counts every
    # mutant, no_coverage included (testteeth models.GradeReport).
    total = killed + survived + timeouts + no_coverage
    score = round(100.0 * (killed + timeouts) / total, 1) if total else None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{group}.json").write_text(
        json.dumps(
            {
                "score": score,
                "total": total,
                "killed": killed,
                "survived": survived,
                "timeouts": timeouts,
                "no_coverage": no_coverage,
            }
        )
    )


def write_all_counts(mod, tmp_path, **counts):
    out = tmp_path / ".testteeth-reports"
    for g in mod.GROUPS:
        write_report_counts(out, g, **counts)
    return out


def row_for(out: str, name: str) -> str:
    return next(line for line in out.splitlines() if line.startswith(name))


def test_summary_ratio_denominator_includes_no_coverage(tmp_path, monkeypatch, capsys):
    # The dictation report shape from CI run 39363900879: 94 mutants total,
    # 60 killed, 1 survived, 33 no_coverage. The printed percentage (63.8%,
    # from 60/94) and the printed ratio must describe the same fraction.
    mod = load_mutation_module(tmp_path, monkeypatch)
    out_dir = write_all_counts(mod, tmp_path, killed=85, survived=5, timeouts=5, no_coverage=5)
    write_report_counts(out_dir, "dictation", killed=60, survived=1, timeouts=0, no_coverage=33)
    assert mod.summary() == 0
    line = row_for(capsys.readouterr().out, "dictation")
    assert "63.8%" in line
    assert "60/94" in line
    assert "60/61" not in line
    assert "no_coverage=33" in line


def test_summary_reports_each_count_and_counts_timeouts_as_detected(tmp_path, monkeypatch, capsys):
    mod = load_mutation_module(tmp_path, monkeypatch)
    out_dir = write_all_counts(mod, tmp_path, killed=85, survived=5, timeouts=5, no_coverage=5)
    write_report_counts(out_dir, "cursor", killed=70, survived=10, timeouts=10, no_coverage=10)
    assert mod.summary() == 0
    line = row_for(capsys.readouterr().out, "cursor")
    assert "80.0%" in line
    assert "80/100" in line
    assert "killed=70" in line
    assert "survived=10" in line
    assert "no_coverage=10" in line
    assert "timeouts=10" in line


def test_summary_overall_uses_the_same_denominator(tmp_path, monkeypatch, capsys):
    mod = load_mutation_module(tmp_path, monkeypatch)
    write_all_counts(mod, tmp_path, killed=60, survived=1, timeouts=0, no_coverage=33)
    assert mod.summary(fail_under=80) == 1
    out = capsys.readouterr().out
    overall = row_for(out, "OVERALL")
    assert "63.8%" in overall
    assert "900/1410" in overall
    assert "below --fail-under 80.0%" in out


def test_summary_passes_with_no_coverage_when_true_score_meets_threshold(tmp_path, monkeypatch, capsys):
    mod = load_mutation_module(tmp_path, monkeypatch)
    write_all_counts(mod, tmp_path, killed=85, survived=5, timeouts=5, no_coverage=5)
    assert mod.summary(fail_under=80) == 0
    overall = row_for(capsys.readouterr().out, "OVERALL")
    assert "90.0%" in overall
    assert "1350/1500" in overall


def test_fail_under_rejects_non_numeric_threshold(monkeypatch):
    import runpy
    import sys

    with pytest.raises(SystemExit) as exc:
        monkeypatch.setattr(sys, "argv", ["mutation.py", "--summary", "--fail-under", "abc"])
        runpy.run_path(str(ROOT / "scripts" / "mutation.py"), run_name="__main__")
    assert "numeric threshold" in str(exc.value.code)
