"""One Project Away app per user session (v0.1.23).

Crash logs showed three or four app processes started within seconds. Two running apps shared the big-cursor
marker: one enlarged the cursor, the other took the big size as the "original" and the pointer stayed big. Now the
app takes a lock at start:

* Windows: a named mutex ``Local\\ProjectAway.App`` (released by Windows when the process ends, even on a crash);
* elsewhere: an exclusive ``flock`` on ``<data>/app.lock``.

A second launch does not start another app: it opens the Settings window of the running one (see
:func:`away.app.tray.main`) and exits. Only the process that holds the lock changes the system cursor size.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

MUTEX_NAME = "Local\\ProjectAway.App"
ERROR_ALREADY_EXISTS = 183

_held: dict[str, Any] = {}


def acquire(name: str = "app", folder: Path | None = None, platform: str | None = None, kernel32: Any = None) -> bool:
    """Take the single-instance lock ``name``. True when this process holds it (also when it already did)."""
    if name in _held:
        return True
    plat = platform or sys.platform
    if plat.startswith("win"):
        mutex = MUTEX_NAME if name == "app" else f"{MUTEX_NAME}.{name}"
        try:
            if kernel32 is None:  # pragma: no cover - Windows only
                import ctypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
                h = kernel32.CreateMutexW(None, False, mutex)
                err = ctypes.get_last_error()  # type: ignore[attr-defined]
            else:
                h = kernel32.CreateMutexW(None, False, mutex)
                err = kernel32.GetLastError()
        except Exception:
            return True  # cannot tell: never stop the app over the lock
        if not h:
            return True
        if err == ERROR_ALREADY_EXISTS:
            with contextlib.suppress(Exception):
                kernel32.CloseHandle(h)
            return False
        _held[name] = h
        return True
    from away import paths

    d = Path(folder) if folder is not None else Path(paths.data_dir())
    try:
        d.mkdir(parents=True, exist_ok=True)
        fh = (d / f"{name}.lock").open("a+")
    except OSError:
        return True
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return False
    except Exception:
        pass
    with contextlib.suppress(OSError):
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
    _held[name] = fh
    return True


def owned(name: str = "app") -> bool:
    """This process holds the lock (it is the one app that may change the system cursor size)."""
    return name in _held


def release(name: str = "app", kernel32: Any = None) -> None:
    h = _held.pop(name, None)
    if h is None:
        return
    with contextlib.suppress(Exception):
        if hasattr(h, "close"):
            h.close()
        elif kernel32 is not None:
            kernel32.CloseHandle(h)
        else:  # pragma: no cover - Windows only
            import ctypes

            ctypes.windll.kernel32.CloseHandle(h)  # type: ignore[attr-defined]
