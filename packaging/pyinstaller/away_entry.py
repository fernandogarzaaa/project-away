"""Frozen-app entry point (PyInstaller).

* No arguments (double-click, Start menu, Dock, .desktop launcher) or ``tray``: start the tray app.
  If the models are missing (first launch of the .dmg / AppImage / .deb, or a Windows install where
  the download was unticked), first run ``setup --yes --skip-probe`` with a desktop notification.
* Any other arguments behave like the ``away`` CLI (``setup``, ``doctor``, ``demo``, ``selftest``...).

The Windows/macOS exe is a windowed (no console) app, so stdin/stdout/stderr start as ``None``;
``ensure_streams`` makes ``input()`` return defaults and sends output to a console (Windows, for
commands that print) or to a log file, so nothing blocks or crashes.
"""

from __future__ import annotations

import sys


def main() -> int:
    from away.app import frozen
    from away.cli import main as cli_main

    args = sys.argv[1:]
    if args and args != ["tray"]:
        frozen.ensure_streams(args)
        return cli_main(args)
    if frozen.needs_setup():
        # show the download progress (Windows: a console window; elsewhere: notification + log/terminal)
        where = frozen.ensure_streams(["setup"])
        frozen.first_launch(cli_main)
        if where == "console":
            frozen.detach_console()
    else:
        frozen.ensure_streams(["tray"])
    return cli_main(["tray"])


if __name__ == "__main__":
    raise SystemExit(main())
