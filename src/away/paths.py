"""Per-OS user directories (no platformdirs dependency)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "ProjectAway"


def _home() -> Path:
    return Path(os.environ.get("AWAY_HOME_OVERRIDE") or Path.home())


def data_dir(platform: str | None = None) -> Path:
    """Where models, recordings and trained weights live."""
    if os.environ.get("AWAY_DATA_DIR"):
        return Path(os.environ["AWAY_DATA_DIR"])
    plat = platform or sys.platform
    home = _home()
    if plat.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
        return base / APP
    if plat == "darwin":
        return home / "Library" / "Application Support" / APP
    base = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return base / "project-away"


def config_dir(platform: str | None = None) -> Path:
    if os.environ.get("AWAY_CONFIG_DIR"):
        return Path(os.environ["AWAY_CONFIG_DIR"])
    plat = platform or sys.platform
    home = _home()
    if plat.startswith("win"):
        base = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
        return base / APP
    if plat == "darwin":
        return home / "Library" / "Application Support" / APP
    base = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    return base / "project-away"


def config_file() -> Path:
    return config_dir() / "config.yaml"


def models_dir() -> Path:
    return data_dir() / "models"


def recordings_dir() -> Path:
    return data_dir() / "recordings"
