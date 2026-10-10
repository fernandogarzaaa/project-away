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
        ans = input("Delete these folders (models, recordings, config)? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("aborted")
            return []
    removed = []
    if not dry_run:
        for p in victims:
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p)
        if autostart.disable():
            print("  removed start-at-login entry")
    print(program_removal_hint())
    return removed
