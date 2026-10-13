"""Crash capture (v0.1.16): ``crash.log`` next to ``away.log`` (``data_dir()/logs``, on Windows
``%LOCALAPPDATA%\\ProjectAway\\logs\\crash.log``).

The very first Settings > Test both cameras on the user's laptop took the whole app down and left no trace: a
native crash (OpenCV / DirectShow / Tk) kills the process before Python logging writes anything. Now every entry
point (the tray app, the Settings window, the CLI and the frozen exe) calls :func:`install` at start-up:

* :mod:`faulthandler` writes a dump of every thread to crash.log on a fatal error (segfault, access violation,
  abort), after a timestamped "started" line for that process;
* ``sys.excepthook`` and ``threading.excepthook`` log any uncaught exception to away.log and crash.log;
* at the next start, a crash recorded since the last start is logged as a warning and returned, so the tray
  and Settings can say "the app crashed last time; details in crash.log".

Everything is best effort: a crash log that cannot be written never stops the app. ``AWAY_NO_CRASHLOG=1`` turns
it off (the test suite does, so pytest's own fault handler stays in charge).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

CRASH_FILE = "crash.log"
SEEN_FILE = "crash-seen.json"  # how far into crash.log the last start had read
MAX_BYTES = 1_000_000  # crash.log is moved to crash.log.1 beyond this
CRASHED_MESSAGE = "the app crashed last time; details in crash.log"
ERROR_MESSAGE = "an unexpected error was logged last time; details in crash.log"
# a fatal crash (faulthandler output); "Windows fatal exception: code 0x..." alone is NOT one: faulthandler also
# reports C++ exceptions that OpenCV / MediaPipe throw and catch themselves
FATAL_MARKERS = (
    "Fatal Python error",
    "Windows fatal exception: access violation",
    "Windows fatal exception: stack overflow",
    "Windows fatal exception: illegal instruction",
    "Windows fatal exception: in page error",
    "Windows fatal exception: int divide by zero",
    "Segmentation fault",
    "Bus error",
    "Floating point exception",
    "Illegal instruction",
    "Aborted",
)
UNCAUGHT_MARKER = "Uncaught exception"

log = logging.getLogger("away.crash")

_lock = threading.Lock()
_state: dict[str, Any] = {"file": None, "path": None, "note": None, "entry": None}


def logs_dir(folder: Path | None = None) -> Path:
    if folder is not None:
        return Path(folder)
    from away import paths

    return Path(paths.data_dir()) / "logs"


def crash_log_path(folder: Path | None = None) -> Path:
    return logs_dir(folder) / CRASH_FILE


def _stamp(clock: Any = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((clock or time.time)()))


def _read_seen(seen: Path) -> int:
    try:
        data = json.loads(seen.read_text(encoding="utf-8"))
        return int(data.get("offset", 0)) if isinstance(data, dict) else 0
    except (OSError, ValueError, TypeError):
        return 0


def _write_seen(seen: Path, offset: int, note: str | None = None, clock: Any = None) -> None:
    data: dict[str, Any] = {"offset": int(offset)}
    if note is None:  # keep a note another process found recently (the tray finds it, Settings shows it too)
        with contextlib.suppress(OSError, ValueError, TypeError):
            old = json.loads(seen.read_text(encoding="utf-8"))
            if isinstance(old, dict) and old.get("note"):
                data["note"], data["noted_at"] = old["note"], old.get("noted_at", 0)
    else:
        data["note"], data["noted_at"] = note, (clock or time.time)()
    with contextlib.suppress(OSError):
        tmp = seen.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, seen)


NOTE_MAX_AGE_S = 12 * 3600.0


def recent_note(folder: Path | None = None, max_age_s: float = NOTE_MAX_AGE_S, clock: Any = None) -> str | None:
    """The previous-run note any process of the app found within ``max_age_s`` (the tray app starts first and
    reads crash.log; the Settings window it opens later shows the same note)."""
    try:
        data = json.loads((logs_dir(folder) / SEEN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("note"):
        return None
    try:
        age = (clock or time.time)() - float(data.get("noted_at", 0))
    except (TypeError, ValueError):
        return None
    return str(data["note"]) if 0 <= age <= max_age_s else None


def dismiss_note(folder: Path | None = None) -> bool:
    """v0.1.22: the person closed the "an unexpected error was logged last time" notice. It is forgotten in this
    process and in crash-seen.json (so Settings and the tray stop showing it); crash.log itself is kept. True when a
    note was cleared."""
    had = _state.get("note") is not None
    with _lock:
        _state["note"] = None
    seen = logs_dir(folder) / SEEN_FILE
    try:
        data = json.loads(seen.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return had
    if not isinstance(data, dict):
        return had
    if data.get("note"):
        had = True
    data.pop("note", None)
    data.pop("noted_at", None)
    with contextlib.suppress(OSError):
        tmp = seen.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, seen)
    return had


def tail(folder: Path | None = None, lines: int = 80, max_bytes: int = 200_000) -> list[str]:
    """v0.1.22: the last ``lines`` lines of crash.log (diagnostics logs); [] when there is none."""
    p = crash_log_path(folder)
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-max(1, int(lines)) :]


def new_entries(path: Path, seen: Path) -> str:
    """The crash.log text written since the last start read it ("" when nothing new or no file)."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    offset = _read_seen(seen)
    if offset > size:  # rotated or truncated since
        offset = 0
    if offset >= size:
        return ""
    try:
        with path.open("rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def classify(text: str) -> str | None:
    """``crashed`` (a fatal error), ``error`` (an uncaught exception) or None for ``text`` from crash.log."""
    if any(m in text for m in FATAL_MARKERS):
        return "crashed"
    if UNCAUGHT_MARKER in text:
        return "error"
    return None


def first_crash_line(text: str) -> str:
    for line in text.splitlines():
        if any(m in line for m in (*FATAL_MARKERS, UNCAUGHT_MARKER)):
            return line.strip()
    return ""


def last_crash_note() -> str | None:
    """What :func:`install` found about the previous run (None: nothing new)."""
    return _state["note"]  # type: ignore[no-any-return]


def write(text: str, clock: Any = None) -> None:
    """Append a timestamped entry to crash.log (when installed). Never raises."""
    f = _state["file"]
    if f is None:
        return
    with _lock, contextlib.suppress(Exception):
        f.write(f"[{_stamp(clock)}] {text.rstrip()}\n")
        f.flush()


def _excepthook(prev: Any) -> Any:
    def hook(exc_type: Any, exc: Any, tb: Any) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        if not issubclass(exc_type, KeyboardInterrupt):
            with contextlib.suppress(Exception):
                log.error("Uncaught exception (main thread):\n%s", text.rstrip())
            write(f"{UNCAUGHT_MARKER} in the main thread ({_state['entry']}):\n{text}")
        if prev is not None:
            with contextlib.suppress(Exception):
                prev(exc_type, exc, tb)

    return hook


def _thread_excepthook(prev: Any) -> Any:
    def hook(args: Any) -> None:
        exc_type = getattr(args, "exc_type", None)
        if exc_type is not None and not issubclass(exc_type, SystemExit):
            name = getattr(getattr(args, "thread", None), "name", "?")
            text = "".join(traceback.format_exception(exc_type, args.exc_value, args.exc_traceback))
            with contextlib.suppress(Exception):
                log.error("Uncaught exception in thread %s:\n%s", name, text.rstrip())
            write(f"{UNCAUGHT_MARKER} in thread {name} ({_state['entry']}):\n{text}")
        if prev is not None:
            with contextlib.suppress(Exception):
                prev(args)

    return hook


def install(
    entry: str,
    folder: Path | None = None,
    *,
    faulthandler_mod: Any = None,
    sys_mod: Any = None,
    threading_mod: Any = None,
    clock: Any = None,
    version: str | None = None,
    force: bool = False,
) -> str | None:
    """Start crash capture for this process (idempotent). Returns the note about the previous run ("the app
    crashed last time; details in crash.log") or None. ``entry``: which entry point ("tray", "settings",
    "cli", "frozen"). The ``*_mod`` arguments are for tests."""
    if os.environ.get("AWAY_NO_CRASHLOG") == "1" and not force:
        return None
    with _lock:
        if _state["file"] is not None:
            return _state["note"]  # type: ignore[no-any-return]
    import faulthandler as _fh

    fh: Any = faulthandler_mod or _fh
    sm: Any = sys_mod or sys
    tm: Any = threading_mod or threading
    if version is None:
        from away import __version__ as version
    try:
        d = logs_dir(folder)
        d.mkdir(parents=True, exist_ok=True)
        path = d / CRASH_FILE
        seen = d / SEEN_FILE
        text = new_entries(path, seen)
        kind = classify(text)
        note = None
        if kind is not None:
            note = CRASHED_MESSAGE if kind == "crashed" else ERROR_MESSAGE
            log.warning("%s (%s): %s", note, path, first_crash_line(text) or "see the file")
        with contextlib.suppress(OSError):
            if path.stat().st_size > MAX_BYTES:
                os.replace(path, path.with_name(CRASH_FILE + ".1"))
        f = path.open("a", encoding="utf-8", buffering=1)
        f.write(f"[{_stamp(clock)}] {entry} started (pid {os.getpid()}, Project Away {version})\n")
        f.flush()
        _write_seen(seen, f.tell(), note, clock)
    except Exception as exc:  # never stop the app over its crash log
        log.warning("crash log unavailable: %s", exc)
        return None
    with _lock:
        _state.update(file=f, path=path, note=note, entry=entry)
    with contextlib.suppress(Exception):
        fh.enable(file=f, all_threads=True)
    sm.excepthook = _excepthook(getattr(sm, "excepthook", None))
    with contextlib.suppress(Exception):
        tm.excepthook = _thread_excepthook(getattr(tm, "excepthook", None))
    return note


def uninstall(faulthandler_mod: Any = None) -> None:
    """Stop writing crash.log (tests; the app keeps it until it exits)."""
    import faulthandler as _fh

    fh = faulthandler_mod or _fh
    with _lock:
        f = _state["file"]
        _state.update(file=None, path=None, note=None, entry=None)
    if f is not None:
        with contextlib.suppress(Exception):
            fh.disable()
        with contextlib.suppress(Exception):
            f.close()
