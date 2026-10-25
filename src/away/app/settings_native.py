"""Open the native macOS Settings app (v0.1.35) instead of the Tk window when it ships with Project Away.

``Project Away Settings.app`` (SwiftUI, Liquid Glass on macOS 26+; source in ``macos/SettingsApp``) lives inside
the bundle at ``Project Away.app/Contents/Helpers/``. It reads and saves the config through ``away settings-bridge
serve`` (:mod:`away.app.settings_bridge`), so the rules stay in Python. The app keeps itself to one window: a
second launch hands its section to the running copy and quits.

:func:`launch` is called by ``away settings`` (the tray's Settings... runs that) and returns False whenever the Tk
window should open instead: not macOS, no native app, ``AWAY_NATIVE_SETTINGS=0``, or a ``--tab`` the native app does
not have yet (Bindings, Gesture Studio, Shortcuts, Agents, Tuning, Performance stay in Tk for now).
"""

from __future__ import annotations

import json
import logging
import os
import plistlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

APP_NAME = "Project Away Settings.app"
ENV = "AWAY_NATIVE_SETTINGS"  # "0" = always Tk; a path = this .app (development)


def candidates(executable: str | None = None, frozen: bool | None = None) -> list[Path]:
    """Where the native app may be: inside the frozen bundle, else the local SwiftPM build of a checkout."""
    exe = Path(executable or sys.executable)
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if is_frozen:  # .../Project Away.app/Contents/MacOS/ProjectAway
        return [exe.parent.parent / "Helpers" / APP_NAME]
    root = Path(__file__).resolve().parents[3]
    return [root / "macos" / "SettingsApp" / "build" / APP_NAME]


def find_app(env: dict[str, str] | None = None, platform: str | None = None, **kw: Any) -> Path | None:
    """The native Settings .app, or None (not macOS, turned off, or not built / not bundled)."""
    env = dict(os.environ) if env is None else env
    if (platform or sys.platform) != "darwin":
        return None
    want = env.get(ENV, "").strip()
    if want == "0":
        return None
    paths = [Path(want)] if want and want != "1" else candidates(**kw)
    return next((p for p in paths if executable_of(p) is not None), None)


def executable_of(app: Path) -> Path | None:
    """``Contents/MacOS/<CFBundleExecutable>`` of an .app, when it exists."""
    try:
        with open(app / "Contents" / "Info.plist", "rb") as fh:
            name = plistlib.load(fh).get("CFBundleExecutable") or app.stem
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None
    exe = app / "Contents" / "MacOS" / str(name)
    return exe if exe.is_file() else None


def section_for_tab(tab: str | None) -> str | None:
    """The native section for ``away settings --tab NAME``; None = a tab only the Tk window has."""
    from away.app.settings_bridge import SECTIONS, TK_ONLY_TABS
    from away.app.settings_gui import resolve_tab_alias

    if not tab or not str(tab).strip():
        return "general"
    want = resolve_tab_alias(str(tab).strip()).lower()
    for s in SECTIONS:
        if want in (s["id"], s["title"].lower()) or (s["tk"] and want in s["tk"].lower()):
            return s["id"]
    if any(want in t.lower() for t in TK_ONLY_TABS):
        return None
    return next((s["id"] for s in SECTIONS if want in s["title"].lower()), None)


def command(app: Path, section: str, config: str | None = None) -> list[str]:
    """argv that opens the native app on ``section``, with the bridge and the Tk window it may start."""
    from away.app.frozen import away_command

    extra = ["--config", str(config)] if config else []
    exe = executable_of(app)
    return [
        str(exe),
        "--section",
        section,
        "--bridge",
        json.dumps(away_command(["settings-bridge", "serve", *extra])),
    ]


def launch(
    tab: str | None = None,
    config: str | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
    **kw: Any,
) -> bool:
    """Open the native Settings app; False when the Tk window should open instead (never raises)."""
    try:
        app = find_app(**kw)
        if app is None:
            return False
        section = section_for_tab(tab)
        if section is None:
            return False
        popen(
            command(app, section, config),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # outlives `away settings` (and the tray's helper process)
        )
        return True
    except Exception:
        logging.getLogger("away.settings").warning("native Settings did not open; using the Tk window", exc_info=True)
        return False
