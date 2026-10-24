"""``away uninstall``: remove user data, config and autostart; print how to remove the program."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from away import paths
from away.app import autostart


def targets(keep_config: bool = False) -> list[Path]:
    out = [paths.data_dir()]
    if not keep_config and paths.config_dir() != paths.data_dir():
        out.append(paths.config_dir())
    return out


def config_files() -> list[Path]:
    d = paths.config_dir()
    return [d / "config.yaml", d / "config.yaml.bak"]


def _remove(p: Path, keep: list[Path]) -> None:
    """Delete ``p``, except the files in ``keep`` inside it (macOS keeps config and data in one folder)."""
    inside = [k for k in keep if k == p or p in k.parents]
    if not inside:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
        return
    if p in inside:
        return
    for child in p.iterdir():
        _remove(child, keep)


def program_removal_hint(platform: str | None = None) -> str:
    plat = platform or sys.platform
    lines = [
        "Remove the program itself with the method you installed it:",
        "  pipx:  pipx uninstall project-away",
        "  pip:   pip uninstall project-away",
    ]
    if plat.startswith("win"):
        lines.append("  Installer: Settings > Apps > Installed apps > Project Away > Uninstall")
    elif plat == "darwin":
        lines.append("  .app: drag /Applications/Project Away.app to the Trash")
    else:
        lines.append("  .deb: sudo apt remove project-away   |   AppImage: delete the .AppImage file")
    return "\n".join(lines)


def run(yes: bool = False, keep_config: bool = False, dry_run: bool = False) -> list[Path]:
    victims = [p for p in targets(keep_config) if p.exists()]
    for p in victims:
        print(f"  will delete: {p}")
    if not victims:
        print("  no Project Away user data found")
    if not dry_run and victims and not yes:
        from away.app.setup_wizard import ask

        if not ask("Delete these folders (models, recordings, config)?", False, None):
            print("aborted")
            return []
    removed = []
    if not dry_run:
        keep = config_files() if keep_config else []
        for p in victims:
            _remove(p, keep)
            removed.append(p)
        if autostart.disable():
            print("  removed start-at-login entry")
    print(program_removal_hint())
    return removed
