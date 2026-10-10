"""Frozen-app entry point.

Double-clicking the installed app (no arguments) starts the tray app and,
on first launch, runs the setup wizard in non-interactive mode so the
MediaPipe models are present. With arguments it behaves like the `away` CLI.
"""

from __future__ import annotations

import sys


def main() -> int:
    from away import paths
    from away.cli import main as cli_main

    args = sys.argv[1:]
    if not args:
        if not paths.config_file().exists():
            cli_main(["setup", "--yes", "--skip-asr", "--skip-probe"])
        args = ["tray"]
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
