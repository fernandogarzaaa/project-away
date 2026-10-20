"""Crash-safety of config persistence (audit 2026-10-20, item 4.1).

config.save() used to truncate config.yaml in place, so a crash or power
loss mid-save left a truncated file and config.load() raised with no
fallback: the app could not start and no copy of the settings survived.
save() now writes atomically (temp file in the same directory, then
os.replace) and keeps the previous valid file as config.yaml.bak; load()
falls back to that backup when the main file is missing or unreadable.
"""

import os
from pathlib import Path

import pytest
import yaml

from away import config as C

BAK_NAME = "config.yaml.bak"


def _cfg(camera_index: int) -> C.AppConfig:
    return C.from_dict({"hands": {"camera_index": camera_index}})[0]


def _truncate(p: Path) -> None:
    """Leave a prefix of the file, cut inside a quoted scalar: the shape an
    interrupted write leaves that no longer parses. (A plain 1/3 cut can
    still parse: YAML closes block structures implicitly at EOF.)"""
    data = p.read_bytes()
    cut = data.index(b"head_assist: '") + len(b"head_assist: 'of")
    p.write_bytes(data[:cut])


def test_interrupted_save_leaves_previous_config_intact(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    C.save(_cfg(3), p)
    before = p.read_bytes()

    real_replace = os.replace

    def flaky_replace(src, dst):
        if Path(dst) == p:
            raise OSError("simulated crash during save")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated crash"):
        C.save(_cfg(7), p)
    monkeypatch.undo()

    assert p.read_bytes() == before
    cfg, _ = C.load(p)
    assert cfg.hands.camera_index == 3


def test_truncated_config_loads_from_backup(tmp_path):
    p = tmp_path / "config.yaml"
    C.save(_cfg(3), p)
    C.save(_cfg(7), p)  # the first save's content is now the last-good backup

    _truncate(p)  # what an interrupted old-style write can leave

    cfg, notes = C.load(p)
    assert cfg.hands.camera_index == 3
    assert any("backup" in n for n in notes)


def test_missing_config_loads_from_backup(tmp_path):
    p = tmp_path / "config.yaml"
    C.save(_cfg(3), p)
    C.save(_cfg(7), p)
    p.unlink()

    cfg, notes = C.load(p)
    assert cfg.hands.camera_index == 3
    assert any("backup" in n for n in notes)


def test_both_files_corrupt_still_raises(tmp_path):
    p = tmp_path / "config.yaml"
    C.save(_cfg(3), p)
    C.save(_cfg(7), p)
    p.write_text("hands: [1, 2\n")
    (tmp_path / BAK_NAME).write_text("hands: [3, 4\n")

    with pytest.raises(yaml.YAMLError):
        C.load(p)


def test_normal_save_updates_backup_to_previous_good(tmp_path):
    p = tmp_path / "config.yaml"
    C.save(_cfg(3), p)
    C.save(_cfg(7), p)

    bak = tmp_path / BAK_NAME
    assert bak.exists()
    assert yaml.safe_load(bak.read_text())["hands"]["camera_index"] == 3
    assert yaml.safe_load(p.read_text())["hands"]["camera_index"] == 7


def test_save_over_corrupt_main_keeps_existing_backup(tmp_path):
    p = tmp_path / "config.yaml"
    C.save(_cfg(3), p)
    C.save(_cfg(7), p)  # backup now holds camera_index 3
    _truncate(p)  # main is damaged; backup is the last good copy

    C.save(_cfg(9), p)  # must not overwrite the good backup with the damaged main

    assert yaml.safe_load((tmp_path / BAK_NAME).read_text())["hands"]["camera_index"] == 3
    cfg, _ = C.load(p)
    assert cfg.hands.camera_index == 9
