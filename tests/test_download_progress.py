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


def test_dir_bytes_ignores_fastdl_preallocated_part_files(tmp_path: Path) -> None:
    """v0.1.4 bug: the sparse full-size model.bin.part made the fallback bar show 100% immediately."""
    (tmp_path / "model.bin.part").write_bytes(b"\0" * 1000)
    (tmp_path / "model.bin.part.json").write_text("{}")
    (tmp_path / "config.json").write_bytes(b"c" * 7)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "x.incomplete").write_bytes(b"i" * 3)
    assert models.dir_bytes(tmp_path) == 10
    assert models.cleanup_fastdl_temp(tmp_path) == 2 and models.dir_bytes(tmp_path) == 10


def test_progress_bar_never_overshoots_and_caps_at_99_while_running() -> None:
    line = models.progress_bar(1_695_000_000, 1_620_000_000)
    assert "1,620/1,620 MB" in line and "1,695" not in line
    running = models.progress_bar(1_695_000_000, 1_620_000_000, final=False)
    assert " 99%" in running and "100%" not in running
    assert models.progress_bar(500_000_000, 1_000_000_000, width=10, final=False) == "  [#####.....]  50%  500/1,000 MB"


def test_dir_progress_does_not_count_a_leftover_part_or_existing_files(tmp_path: Path) -> None:
    (tmp_path / "model.bin.part").write_bytes(b"\0" * 900_000)  # left by a failed fast download
    (tmp_path / "config.json").write_bytes(b"c" * 100_000)  # already finished before the fallback
    out: list[str] = []
    with models._DirProgress(tmp_path, 1_000_000, out.append, interval=0.01) as dp:
        assert dp.total == 900_000
        time.sleep(0.05)
        assert all("100%" not in s and " 0%" in s for s in out)
        (tmp_path / "model.bin").write_bytes(b"z" * 450_000)
        time.sleep(0.08)
    assert any(" 50%" in s for s in out)
    assert "100%" in out[-2]
