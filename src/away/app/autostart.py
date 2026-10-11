"""Start-at-login integration per OS (user level, no admin rights)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

LABEL = "com.fernandogarzaaa.projectaway"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def launch_command() -> list[str]:
    if getattr(sys, "frozen", False):  # PyInstaller bundle
        # inside an AppImage sys.executable is a temporary mount; $APPIMAGE is the file the user keeps
        return [os.environ.get("APPIMAGE") or sys.executable, "tray"]
    return [sys.executable, "-m", "away", "tray"]


def linux_desktop_file(home: Path) -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    return base / "autostart" / "project-away.desktop"


def mac_plist(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def desktop_entry(cmd: list[str]) -> str:
    exec_line = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    return (
        "[Desktop Entry]\nType=Application\nName=Project Away\n"
        f"Exec={exec_line}\nX-GNOME-Autostart-enabled=true\nNoDisplay=false\n"
        "Comment=Hands-free PC control\n"
    )


def plist(cmd: list[str]) -> str:
    args = "\n".join(f"    <string>{c}</string>" for c in cmd)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>\n'
        f"  <key>Label</key><string>{LABEL}</string>\n"
        f"  <key>ProgramArguments</key><array>\n{args}\n  </array>\n"
        "  <key>RunAtLoad</key><true/>\n"
        "</dict></plist>\n"
    )


def enable(platform: str | None = None, home: Path | None = None, cmd: list[str] | None = None) -> str:
    plat = platform or sys.platform
    home = home or Path.home()
    cmd = cmd or launch_command()
    if plat.startswith("win"):  # pragma: no cover - needs Windows
        import winreg  # type: ignore[import-not-found,unused-ignore]

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, "ProjectAway", 0, winreg.REG_SZ, " ".join(f'"{c}"' for c in cmd))
        return "HKCU\\" + RUN_KEY + "\\ProjectAway"
    path = mac_plist(home) if plat == "darwin" else linux_desktop_file(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plist(cmd) if plat == "darwin" else desktop_entry(cmd))
    return str(path)


def disable(platform: str | None = None, home: Path | None = None) -> bool:
    plat = platform or sys.platform
    home = home or Path.home()
    if plat.startswith("win"):  # pragma: no cover - needs Windows
        import winreg  # type: ignore[import-not-found,unused-ignore]

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, "ProjectAway")
            return True
        except FileNotFoundError:
            return False
    path = mac_plist(home) if plat == "darwin" else linux_desktop_file(home)
    if path.exists():
        path.unlink()
        return True
    return False
