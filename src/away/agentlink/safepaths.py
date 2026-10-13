"""The only files an agent may download: diagnostics bundles, recordings and logs inside the app's data folder.

Paths are relative (``diagnostics/bundle-....zip``); the first part picks one of :data:`ROOTS`. Absolute paths,
drive letters, ``..``, backslashes, NUL bytes and anything that resolves (symlinks included) outside its root
are refused. Agents can never write files through this module; it only resolves paths for reading.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOTS = ("diagnostics", "recordings", "logs")
_BAD = re.compile(r"(^[/\\])|(^[A-Za-z]:)|\\|\x00")


class UnsafePath(ValueError):
    pass


def resolve(data_dir: Path, rel: str) -> Path:
    """The real path of ``rel`` inside one of the allowed roots under ``data_dir``; raises :class:`UnsafePath`."""
    rel = str(rel or "").strip()
    if not rel or _BAD.search(rel):
        raise UnsafePath("path must be relative, like diagnostics/<file>")
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise UnsafePath("path must not contain '.', '..' or empty parts")
    if parts[0] not in ROOTS:
        raise UnsafePath(f"only {', '.join(ROOTS)} can be downloaded")
    root = (Path(data_dir) / parts[0]).resolve()
    target = (Path(data_dir) / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise UnsafePath("path escapes its folder") from None
    if not target.is_file():
        raise FileNotFoundError(rel)
    return target


def listing(data_dir: Path, limit: int = 200) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for root in ROOTS:
        base = Path(data_dir) / root
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*"), key=lambda q: q.stat().st_mtime if q.exists() else 0, reverse=True):
            if p.is_file() and not p.is_symlink():
                out.append({"path": p.relative_to(data_dir).as_posix(), "bytes": p.stat().st_size})
                if len(out) >= limit:
                    return out
    return out
