from __future__ import annotations

import time
from pathlib import Path

from away.app import models


def test_progress_bar_formats_and_clamps() -> None:
    assert models.progress_bar(0, 100_000_000, width=10) == "  [..........]   0%  0/100 MB"
    assert models.progress_bar(50_000_000, 100_000_000, width=10) == "  [#####.....]  50%  50/100 MB"
    assert models.progress_bar(300_000_000, 100_000_000, width=10).startswith("  [##########] 100%")
    assert models.progress_bar(-5, 100_000_000, width=4).startswith("  [....]   0%")
    assert models.progress_bar(2_000_000, 0) == "  2 MB"


def test_dir_bytes_counts_nested_and_partial_files(tmp_path: Path) -> None:
    assert models.dir_bytes(tmp_path / "missing") == 0
    (tmp_path / ".cache" / "dl").mkdir(parents=True)
    (tmp_path / "model.bin").write_bytes(b"x" * 10)
    (tmp_path / ".cache" / "dl" / "a.incomplete").write_bytes(b"y" * 5)
    assert models.dir_bytes(tmp_path) == 15


def test_dir_progress_redraws_and_finishes_at_100(tmp_path: Path) -> None:
    out: list[str] = []
    with models._DirProgress(tmp_path, 1_000_000, out.append, interval=0.01):
        (tmp_path / "f").write_bytes(b"z" * 500_000)
        time.sleep(0.1)
    assert any(" 50%" in s for s in out)
    assert out[-2].startswith("\r") and "100%" in out[-2]
    assert out[-1] == "\n"
    assert all(s.startswith("\r") for s in out[:-1])


def test_dir_progress_does_not_claim_100_on_failure(tmp_path: Path) -> None:
    out: list[str] = []
    try:
        with models._DirProgress(tmp_path, 1_000_000, out.append, interval=0.01):
            raise RuntimeError("network down")
    except RuntimeError:
        pass
    assert not any("100%" in s for s in out)
    assert out[-1] == "\n"


def test_quiet_hf_hub_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)
    models.quiet_hf_hub()
    import os

    assert os.environ["HF_HUB_VERBOSITY"] == "error"
