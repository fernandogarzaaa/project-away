"""The CI mutation matrix must match GROUPS in scripts/mutation.py.

The matrix in .github/workflows/ci.yml is a hand-maintained copy of GROUPS.
Nothing in GitHub Actions compares the two, so drift used to surface only
late and incidentally (a missing artifact in the summary job, or a matrix job
failing with "unknown mutation group"). This test compares the complete sets
so drift fails fast. It also runs as the mutation-groups-check CI job, which
the mutation matrix job depends on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"


def load_groups() -> list[str]:
    spec = importlib.util.spec_from_file_location("mutation_groups", ROOT / "scripts" / "mutation.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.GROUPS)


def ci_matrix_groups() -> list[str]:
    lines = CI_YML.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  mutation:"))
    end = next(i for i, line in enumerate(lines) if line.startswith("  mutation-summary:"))
    section = lines[start:end]
    key = next(i for i, line in enumerate(section) if line.strip() == "group:")
    key_indent = len(section[key]) - len(section[key].lstrip())
    groups = []
    for line in section[key + 1 :]:
        stripped = line.strip()
        if stripped.startswith("- "):
            groups.append(stripped[2:].strip())
        elif stripped and len(line) - len(line.lstrip()) <= key_indent:
            break
    return groups


def test_ci_matrix_matches_mutation_groups():
    matrix = ci_matrix_groups()
    groups = load_groups()
    assert matrix, "parsed no groups from the CI mutation matrix; the parser or the workflow layout changed"
    assert len(matrix) == len(set(matrix)), f"duplicate entries in the CI mutation matrix: {matrix}"
    missing = [g for g in groups if g not in matrix]
    extra = [g for g in matrix if g not in groups]
    assert not missing and not extra, (
        "CI mutation matrix out of sync with GROUPS in scripts/mutation.py: "
        f"missing from matrix: {missing}, in matrix but not in GROUPS: {extra}"
    )
