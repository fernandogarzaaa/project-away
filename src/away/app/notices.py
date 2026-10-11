"""One-time notices that survive restarts (a marker file per key in the data folder)."""

from __future__ import annotations

import re
from pathlib import Path


def _marker(key: str, folder: Path | None) -> Path:
    if folder is None:
        from away import paths

        folder = paths.data_dir()
    return Path(folder) / "notices" / (re.sub(r"[^a-z0-9_.-]", "_", key.lower()) + ".shown")


def seen(key: str, folder: Path | None = None) -> bool:
    return _marker(key, folder).exists()


def once(key: str, folder: Path | None = None) -> bool:
    """True the first time ``key`` is asked about (and remembers it); False ever after. Never raises:
    if the marker cannot be written the notice is shown (better twice than never)."""
    m = _marker(key, folder)
    if m.exists():
        return False
    try:
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text("1", encoding="utf-8")
    except OSError:
        pass
    return True
