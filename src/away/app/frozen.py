"""Helpers for the installed (PyInstaller) app.

* A windowed exe (Windows/macOS) starts with ``sys.stdin``/``stdout``/``stderr``
  set to ``None``: any ``input()`` raises and any ``sys.stderr.write`` (tqdm
  progress bars from the model download) crashes. :func:`ensure_streams` makes
  them safe: stdin becomes an empty stream (``input()`` -> ``EOFError`` -> the
  default answer), output goes to a visible console on Windows for commands
  that print (``setup``, ``doctor``...) or to a log file otherwise.
* :func:`missing_models` / :func:`first_launch` run the non-interactive setup
  the first time the tray app starts without its models (macOS .dmg, Linux
  AppImage/.deb, or a Windows install where the model download was unticked).
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from away import config as cfgmod
from away import paths
from away.app import models

# commands whose output the user should see in a console window (Windows windowed exe)
CONSOLE_COMMANDS = frozenset(
    {"setup", "doctor", "selftest", "demo", "config", "bindings", "cameras", "uninstall", "update"}
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def stdin_usable(stream: Any = None) -> bool:
    """True when ``input()`` can actually read from a user (not a GUI exe with no stdin)."""
    s = sys.stdin if stream is None else stream
    if s is None or getattr(s, "closed", False):
        return False
    try:
        return bool(s.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def log_file(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("AWAY_LOG_FILE"):
        return Path(env["AWAY_LOG_FILE"])
    return paths.data_dir() / "logs" / "away.log"


def wants_console(argv: Sequence[str]) -> bool:
    return bool(argv) and (argv[0] in CONSOLE_COMMANDS or argv[0] in ("--version", "-h", "--help"))


def _win_console() -> TextIO | None:  # pragma: no cover - Windows only
    """Attach to the parent console (cmd/PowerShell) or open a new one; return a writer for it."""
    import ctypes

    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    if not (k32.AttachConsole(-1) or k32.AllocConsole()):
        return None
    try:
        return open("CONOUT$", "w", buffering=1, encoding="utf-8", errors="replace")
    except OSError:
        return None


def ensure_streams(
    argv: Sequence[str],
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
    console: Callable[[], TextIO | None] | None = None,
) -> str:
    """Make stdin/stdout/stderr safe to use. Returns where output goes: ``tty`` | ``console`` | a log path."""
    env = os.environ if env is None else env
    plat = platform or sys.platform
    if sys.stdin is None:
        sys.stdin = io.StringIO("")  # input() -> EOFError -> default answer, never a hang or crash
    if sys.stdout is not None and sys.stderr is not None:
        return "tty"
    out: TextIO | None = None
    where = ""
    if plat == "win32" and wants_console(argv) and env.get("AWAY_NO_CONSOLE") != "1":
        out = (console or _win_console)()
        where = "console"
    if out is None:
        path = log_file(env)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = open(path, "a", buffering=1, encoding="utf-8", errors="replace")  # noqa: SIM115
        where = str(path)
    if sys.stdout is None:
        sys.stdout = out
    if sys.stderr is None:
        sys.stderr = out
    return where


def missing_models(cfg: cfgmod.AppConfig | None = None, models_dir: Path | None = None) -> list[str]:
    """Models the installed app needs that are not on disk yet (MediaPipe tasks + the Whisper model)."""
    if cfg is None:
        cfg, _ = cfgmod.load()
    mdir = models_dir or (Path(cfg.models_dir) if cfg.models_dir else paths.models_dir())
    missing = [mf.name for mf in models.MEDIAPIPE_MODELS if not (mdir / mf.name).exists()]
    if cfg.voice.enabled and cfg.voice.asr_backend == "faster-whisper":
        wdir = mdir / f"whisper-{cfg.voice.asr_model}"
        if not (wdir / "model.bin").exists():
            missing.append(f"whisper-{cfg.voice.asr_model}")
    return missing


def needs_setup(cfg: cfgmod.AppConfig | None = None) -> bool:
    return not paths.config_file().exists() or bool(missing_models(cfg))


def detach_console() -> None:  # pragma: no cover - Windows only
    """After first-launch setup in a console window: close it and keep logging to the log file."""
    path = log_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(path, "a", buffering=1, encoding="utf-8", errors="replace")  # noqa: SIM115
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.FreeConsole()  # type: ignore[attr-defined]


def notify(title: str, message: str, platform: str | None = None) -> bool:  # pragma: no cover - desktop only
    """Best-effort desktop notification (no extra dependencies). Always also prints."""
    print(f"{title}: {message}", flush=True)
    plat = platform or sys.platform
    try:
        if plat == "darwin":
            script = f"display notification {_osa(message)} with title {_osa(title)}"
            subprocess.run(["osascript", "-e", script], timeout=5, check=False)
            return True
        if plat.startswith("linux") and shutil.which("notify-send"):
            subprocess.run(["notify-send", "-a", "Project Away", title, message], timeout=5, check=False)
            return True
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def _osa(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def first_launch(
    cli_main: Callable[[list[str]], int],
    cfg: cfgmod.AppConfig | None = None,
    notifier: Callable[[str, str], object] = notify,
) -> int | None:
    """Run ``setup --yes --skip-probe`` when models are missing. Returns its exit code, or None if not needed."""
    if cfg is None:
        cfg, _ = cfgmod.load()
    missing = missing_models(cfg)
    if not missing and paths.config_file().exists():
        return None
    size = sum(mf.size_mb for mf in models.MEDIAPIPE_MODELS if mf.name in missing)
    asr = cfg.voice.asr_model
    if f"whisper-{asr}" in missing:
        size += models.ASR_SIZES_MB.get(asr, 0)
    notifier(
        "Project Away: first-time setup",
        f"Downloading hand-tracking and speech models (~{size / 1000:.1f} GB). The app starts when done.",
    )
    rc = cli_main(["setup", "--yes", "--skip-probe", "--asr-model", asr])
    notifier(
        "Project Away",
        "Models ready." if rc == 0 else f"Some downloads failed; they retry next launch. Log: {log_file()}",
    )
    return rc
