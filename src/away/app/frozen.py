"""Helpers for the installed (PyInstaller) app.

* A windowed exe (Windows/macOS) starts with ``sys.stdin``/``stdout``/``stderr``
  set to ``None``: any ``input()`` raises and any ``sys.stderr.write`` (tqdm
  progress bars from the model download) crashes. :func:`ensure_streams` makes
  them safe: stdin becomes an empty stream (``input()`` -> ``EOFError`` -> the
  default answer), output goes to a visible console on Windows for commands
  that print (``setup``, ``doctor``...) or to a log file otherwise.
* :func:`disable_quick_edit` turns off the console's QuickEdit mode: with it on, one stray click in the
  setup window starts a text selection ("Select ..." in the title) and Windows *freezes the process*
  until a key is pressed, so a download silently stops.
* :func:`missing_models` / :func:`first_launch` run the non-interactive setup
  the first time the tray app starts without its models (macOS .dmg, Linux
  AppImage/.deb, or a Windows install where the model download was unticked).
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from away import config as cfgmod
from away import paths
from away.app import models

# commands whose output the user should see in a console window (Windows windowed exe)
CONSOLE_COMMANDS = frozenset(
    {
        "setup",
        "doctor",
        "selftest",
        "demo",
        "config",
        "bindings",
        "cameras",
        "calibrate-cameras",
        "monitors",
        "uninstall",
        "update",
        "hardware",
        "gpu",
    }
)


def away_command(args: Sequence[str], frozen: bool | None = None, executable: str | None = None) -> list[str]:
    """argv that runs ``away ARGS`` in a new process: the installed app exe *is* the CLI when frozen
    (``ProjectAway.exe calibrate-cameras``); from Python it is ``python -m away ...``."""
    exe = executable or sys.executable
    if is_frozen() if frozen is None else frozen:
        return [exe, *args]
    return [exe, "-m", "away", *args]


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


def setup_logging(stream: Any = None, level: int | None = None) -> Any:
    """INFO logging for the ``away.*`` loggers (updater steps, monitors...) to ``stream`` (default stderr,
    which the windowed app has already pointed at away.log). Idempotent; other libraries stay at WARNING."""
    import logging

    lg = logging.getLogger("away")
    lg.setLevel(logging.INFO if level is None else level)
    for h in lg.handlers:
        if getattr(h, "_away_handler", False):
            return h
    h = logging.StreamHandler(stream if stream is not None else sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    h._away_handler = True  # type: ignore[attr-defined]
    lg.addHandler(h)
    return h


def wants_console(argv: Sequence[str]) -> bool:
    return bool(argv) and (argv[0] in CONSOLE_COMMANDS or argv[0] in ("--version", "-h", "--help"))


ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080
GENERIC_READ_WRITE = 0x80000000 | 0x40000000
FILE_SHARE_READ_WRITE = 0x1 | 0x2
OPEN_EXISTING = 3
INVALID_HANDLE = (-1, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF, None, 0)


def quick_edit_off(mode: int) -> int:
    """Console input mode with QuickEdit cleared. ``ENABLE_EXTENDED_FLAGS`` must be set or Windows ignores
    the QuickEdit bit."""
    return (mode & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS


_saved_console_mode: list[tuple[Any, Any, int]] = []


def disable_quick_edit(kernel32: Any = None, platform: str | None = None, restore_at_exit: bool = True) -> bool:
    """Turn off QuickEdit for this process's console so a click cannot pause setup. Best effort, never raises.

    Uses ``CONIN$`` (works for an attached or newly allocated console, where the std input handle of a GUI
    exe is invalid). The previous mode is restored at exit, so a cmd window the user ran us from keeps
    their setting."""
    plat = platform or sys.platform
    if plat != "win32":
        return False
    try:
        import ctypes

        k32 = kernel32
        if k32 is None:  # pragma: no cover - Windows only
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        with contextlib.suppress(AttributeError, TypeError):
            k32.CreateFileW.restype = ctypes.c_void_p  # a HANDLE is pointer-sized; the c_int default truncates it
        raw = k32.CreateFileW("CONIN$", GENERIC_READ_WRITE, FILE_SHARE_READ_WRITE, None, OPEN_EXISTING, 0, None)
        if raw in INVALID_HANDLE:
            return False
        handle = ctypes.c_void_p(raw)
        mode = ctypes.c_uint32(0)
        if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        old = int(mode.value)
        new = quick_edit_off(old)
        if new != old and not k32.SetConsoleMode(handle, ctypes.c_uint32(new)):
            return False
        if restore_at_exit and new != old:
            import atexit

            _saved_console_mode.append((k32, handle, old))
            atexit.register(_restore_console_mode)
        return True
    except Exception:
        return False


def _restore_console_mode() -> None:
    while _saved_console_mode:
        k32, handle, old = _saved_console_mode.pop()
        with contextlib.suppress(Exception):
            k32.SetConsoleMode(handle, old & 0xFFFFFFFF)


def _win_console() -> TextIO | None:  # pragma: no cover - Windows only
    """Attach to the parent console (cmd/PowerShell) or open a new one; return a writer for it."""
    import ctypes

    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    if not (k32.AttachConsole(-1) or k32.AllocConsole()):
        return None
    disable_quick_edit(k32)
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
        if plat == "win32" and is_frozen() and wants_console(argv):
            disable_quick_edit(platform=plat)  # frozen exe run from cmd/PowerShell: a click must not pause setup
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
        model = _asr_model(cfg)
        wdir = mdir / f"whisper-{model}"
        if not (wdir / "model.bin").exists():
            missing.append(f"whisper-{model}")
    return missing


def _asr_model(cfg: cfgmod.AppConfig) -> str:
    """The Whisper model in use (``voice.asr_model``, or the hardware profile's pick when it is ``auto``)."""
    if cfg.voice.asr_model != "auto":
        return cfg.voice.asr_model
    try:
        from away.app.performance import effective_asr_model

        return effective_asr_model(cfg)
    except Exception:
        return "small"


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


def mac_message(
    title: str,
    message: str,
    folder: Path | None = None,
    spawn: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.time,
) -> bool:
    """macOS: show ``message`` in Project Away's own small window (``away agent-ui toast``, top right, closes by
    itself or on a click) instead of a system notification. Notifications sent with ``osascript`` (the only kind an
    unsigned app can send) are credited to Script Editor, and clicking one opens Script Editor. False when the
    window could not be started."""
    import json
    import tempfile
    import uuid

    from away.agentlink.prompts import window_process_options

    d = Path(folder) if folder is not None else Path(tempfile.gettempdir()) / "project-away-messages"
    try:
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("message-*.json"):  # the window does not delete its file; keep the folder small
            with contextlib.suppress(OSError):
                if clock() - old.stat().st_mtime > 3600:
                    old.unlink()
        spec = d / f"message-{uuid.uuid4().hex[:8]}.json"
        spec.write_text(json.dumps({"title": title, "text": message, "level": "info", "seconds": 8}), encoding="utf-8")
        (spawn or subprocess.Popen)(away_command(["agent-ui", "toast", str(spec)]), **window_process_options())
        return True
    except (OSError, ValueError):
        return False


def notify(title: str, message: str, platform: str | None = None) -> bool:  # pragma: no cover - desktop only
    """Best-effort desktop notification (no extra dependencies). Always also prints."""
    print(f"{title}: {message}", flush=True)
    plat = platform or sys.platform
    try:
        if plat == "darwin":
            # v0.1.35: a real macOS notification (Project Away's name and icon, Notification Center, Focus) when the
            # app bundle may post one; Project Away's own window when it may not (a source checkout, permission
            # denied, the framework missing), also when the user's "no" arrives after the message was queued
            from away.app import macnotify

            if macnotify.notify(title, message, fallback=mac_window_notify):
                return True
            return mac_window_notify(title, message)
        if plat.startswith("linux") and shutil.which("notify-send"):
            subprocess.run(["notify-send", "-a", "Project Away", title, message], timeout=5, check=False)
            return True
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def mac_window_notify(title: str, message: str) -> bool:  # pragma: no cover - desktop only
    """macOS without system notifications: Project Away's own message window (:func:`mac_message`)."""
    try:
        if mac_message(title, message):
            return True
        # the window could not start: a system notification (credited to Script Editor) beats nothing
        script = f"display notification {_osa(message)} with title {_osa(title)}"
        subprocess.run(["osascript", "-e", script], timeout=5, check=False)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _osa(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def setup_download_mb(cfg: cfgmod.AppConfig, missing: list[str] | None = None) -> float:
    """How much the first-launch setup downloads (MB): the missing MediaPipe models and Whisper model."""
    missing = missing_models(cfg) if missing is None else missing
    size = float(sum(mf.size_mb for mf in models.MEDIAPIPE_MODELS if mf.name in missing))
    asr = _asr_model(cfg)
    if f"whisper-{asr}" in missing:
        size += models.ASR_SIZES_MB.get(asr, 0)
    return size


def setup_window_process(argv: Sequence[str], run: Callable[..., Any] = subprocess.run) -> int | None:
    """Run ``setup`` with its progress window (:mod:`away.app.setup_window`) in its own process and wait for it.
    Its exit code, or None when that process could not be started (then the caller sets up without a window)."""
    from away.agentlink.prompts import window_process_options

    try:
        return int(run(away_command(["setup-window", *argv]), **window_process_options()).returncode)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def first_launch(
    cli_main: Callable[[list[str]], int],
    cfg: cfgmod.AppConfig | None = None,
    notifier: Callable[[str, str], object] = notify,
    window: Callable[[list[str]], int | None] | None = None,
) -> int | None:
    """Run ``setup --yes --skip-probe`` when models are missing. Returns its exit code, or None if not needed.

    ``window`` (macOS, Linux): runs the setup with a progress window instead (:func:`setup_window_process`); when it
    cannot start, the setup runs here with a notification at the start and the end, as before."""
    if cfg is None:
        cfg, _ = cfgmod.load()
    missing = missing_models(cfg)
    if not missing and paths.config_file().exists():
        return None
    size = setup_download_mb(cfg, missing)
    # an explicit model is passed on; "auto" stays auto (setup offers the GPU pack first, then picks the model)
    argv = ["setup", "--yes", "--skip-probe"]
    if cfg.voice.asr_model != "auto":
        argv += ["--asr-model", cfg.voice.asr_model]
    if window is not None:
        rc = window(argv)
        if rc is not None:
            return rc
    notifier(
        "Project Away: first-time setup",
        f"Downloading hand-tracking and speech models (~{size / 1000:.1f} GB). The app starts when done.",
    )
    rc = cli_main(argv)
    notifier(
        "Project Away",
        "Models ready." if rc == 0 else f"Some downloads failed; they retry next launch. Log: {log_file()}",
    )
    return rc
