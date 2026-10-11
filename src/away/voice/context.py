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


def active_process(platform: str | None = None) -> str | None:  # pragma: no cover - OS specific
    """Process name of the focused window (``chrome.exe``, ``Code``, ``firefox``), for per-app profiles."""
    plat = platform or sys.platform
    try:
        if plat.startswith("win"):
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
            h = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                buf = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(1024)
                if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    return buf.value.replace("\\", "/").rsplit("/", 1)[-1] or None
            finally:
                kernel32.CloseHandle(h)
            return None
        if plat == "darwin":
            return active_app(plat)  # System Events already returns the process name
        if shutil.which("xdotool"):
            out = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowpid"], capture_output=True, text=True, timeout=1.0
            )
            xpid = out.stdout.strip()
            if xpid.isdigit():
                with open(f"/proc/{xpid}/comm", encoding="utf-8") as fh:
                    return fh.read().strip() or None
    except Exception:
        return None
    return None


def whisper_prompt(app: str | None, vocabulary: list[str]) -> str | None:
    """A short initial prompt nudging Whisper toward your terms.

    ``app`` is accepted for callers that pass it but is deliberately *not* put in the prompt: Whisper
    repeats its prompt on short or noisy audio, and before v0.1.6 that typed "Dictating into <app>." into
    the app. The focused app is used only by the text cleanup (see :mod:`away.voice.cleanup`)."""
    del app
    if not vocabulary:
        return None
    return "Vocabulary: " + ", ".join(vocabulary[:40]) + "."
