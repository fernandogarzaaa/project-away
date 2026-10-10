"""Best-effort name of the focused application/window (used as a cleanup hint)."""

from __future__ import annotations

import shutil
import subprocess
import sys


def active_app(platform: str | None = None) -> str | None:  # pragma: no cover - OS specific
    plat = platform or sys.platform
    try:
        if plat.startswith("win"):
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            hwnd = user32.GetForegroundWindow()
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            return buf.value or None
        if plat == "darwin":
            out = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first process whose frontmost is true',
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            return out.stdout.strip() or None
        if shutil.which("xdotool"):
            out = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True, timeout=1.0
            )
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def whisper_prompt(app: str | None, vocabulary: list[str]) -> str | None:
    """A short initial prompt nudging Whisper toward your terms and the app's register."""
    parts = []
    if vocabulary:
        parts.append("Vocabulary: " + ", ".join(vocabulary[:40]) + ".")
    if app:
        parts.append(f"Dictating into {app}.")
    return " ".join(parts) or None
