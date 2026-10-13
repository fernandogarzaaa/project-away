"""Small actions shared by the tray, Settings and the guided first run (testable without a desktop)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from away.comfort.settings import get_path, set_switch


def set_switch_in_file(path: str, value: bool, file: Path | None = None, live_cfg: Any = None) -> str | None:
    """Flip a plain switch (``attention.enabled``...) in the config file; the running app hot-reloads it.
    ``live_cfg``: also update this in-memory config (the tray's copy). Returns an error text or None."""
    from away import config as cfgmod

    p = Path(file) if file is not None else cfgmod.config_path()
    try:
        cfg, _ = cfgmod.load(p) if p.exists() else (cfgmod.AppConfig(), [])
    except ValueError as exc:
        return f"your settings file has a problem: {exc}"
    err = set_switch(cfg, path, value)
    if err:
        return err
    try:
        cfgmod.save(cfg, p)
    except OSError as exc:
        return f"could not save your settings: {exc}"
    if live_cfg is not None:
        set_switch(live_cfg, path, value)
    return None


def switch_value(cfg: Any, path: str) -> bool:
    return bool(get_path(cfg, path))


def first_run_command() -> list[str]:
    """argv for the guided first run in its own process (Tk and pystray each want their own main thread)."""
    from away.app.frozen import away_command

    return away_command(["tour"])


def launch_first_run(popen: Any = subprocess.Popen) -> str | None:
    """Open the guided first run. Returns an error text or None."""
    try:
        popen(first_run_command())
    except Exception as exc:
        return f"could not open the guided first run: {exc}"
    return None
