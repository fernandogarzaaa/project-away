"""Big system cursor while gesture control is active (``hands.cursor.big_cursor_size``).

Users sit far from the screen, so while gestures drive the pointer the Windows cursor is enlarged with the
accessibility cursor size; it shrinks back to the user's own size whenever gesture control pauses (manual pause,
the F9 hotkey, auto-pause from physical mouse/keyboard use, the open-palm clutch), is disabled, or the app quits.
Shrinking back also tells the user that gestures are paused.

* :class:`BigCursor` is the controller: pure logic with an injectable clock, a backend and a marker file.
  At most one change per ``debounce_s`` (auto-pause can flap); :meth:`BigCursor.restore` is immediate.
* The marker file records the user's original size before anything is changed and is removed after the restore.
  If the app crashed or was killed while the cursor was big, :meth:`BigCursor.recover` (called at start-up)
  finds the marker and puts the original size back.
* :class:`WindowsCursorBackend` writes ``HKCU\\Control Panel\\Cursors\\CursorBaseSize`` (32..256 px), applies it
  live with ``SystemParametersInfoW(0x2029, 0, size, SPIF_UPDATEINIFILE | SPIF_SENDCHANGE)`` (the undocumented
  action the Settings app uses for the pointer-size slider) and reloads the cursors with ``SPI_SETCURSORS``.
  :class:`NullCursorBackend` (macOS, Linux) changes nothing; the at-cursor ring is scaled up instead.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from away import paths

BASE_SIZE = 32  # Windows' normal cursor size (px)
MIN_SIZE, MAX_SIZE = 32, 256  # what Windows accepts for CursorBaseSize
SCALES = {1: 1.5, 2: 2.0, 3: 3.0, 4: 4.0, 5: 6.0}  # big_cursor_size -> multiple of 32 px
DEBOUNCE_S = 0.3
ENLARGE_DEBOUNCE_S = 1.5  # v0.1.23: re-enlarging waits this long after a change (restoring stays immediate)
MARKER_NAME = "big_cursor.json"
USER_SIZE_NAME = "cursor_user_size.json"  # v0.1.23: the last size seen that was the user's own (not ours)


def target_size(level: int) -> int:
    """Pixel size for ``big_cursor_size`` 1..5 (0 = off -> 0)."""
    if level <= 0:
        return 0
    scale = SCALES.get(int(level), SCALES[max(SCALES)])
    return max(MIN_SIZE, min(MAX_SIZE, round(BASE_SIZE * scale)))


def our_sizes() -> set[int]:
    """Every size Project Away sets (big_cursor_size 1..5): never mistaken for the user's own size."""
    return {target_size(lv) for lv in SCALES}


def ring_scale(level: int) -> float:
    """How much to scale the at-cursor ring (macOS/Linux stand-in for the big cursor)."""
    return 1.0 if level <= 0 else min(2.5, SCALES.get(int(level), 2.0) ** 0.5 * 1.1)


def default_marker() -> Path:
    return paths.data_dir() / MARKER_NAME


class CursorSizeBackend(Protocol):
    def get(self) -> int | None: ...  # current size in px (None = unknown)
    def set(self, size: int) -> bool: ...  # True when applied


class NullCursorBackend:
    """macOS / Linux: no system cursor size API is used."""

    def get(self) -> int | None:
        return None

    def set(self, size: int) -> bool:
        return False


class WindowsCursorBackend:  # pragma: no cover - needs Windows
    KEY = r"Control Panel\Cursors"
    VALUE = "CursorBaseSize"
    SPI_SETCURSORS = 0x0057
    SPI_SETCURSORSIZE = 0x2029  # undocumented; what Settings > Accessibility > Mouse pointer size calls
    SPIF_UPDATEINIFILE = 0x01
    SPIF_SENDCHANGE = 0x02

    def get(self) -> int | None:
        if sys.platform != "win32":
            return None
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.KEY) as k:
                val, _typ = winreg.QueryValueEx(k, self.VALUE)
            return int(val)
        except (OSError, ValueError, TypeError):
            return BASE_SIZE  # never set: Windows uses the default 32 px

    def set(self, size: int) -> bool:
        if sys.platform != "win32":
            return False
        import ctypes
        import winreg

        size = max(MIN_SIZE, min(MAX_SIZE, int(size)))
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.KEY, 0, winreg.KEY_SET_VALUE) as k:
                winreg.SetValueEx(k, self.VALUE, 0, winreg.REG_DWORD, size)
        except OSError:
            return False
        spi = ctypes.windll.user32.SystemParametersInfoW
        flags = self.SPIF_UPDATEINIFILE | self.SPIF_SENDCHANGE
        live = bool(spi(self.SPI_SETCURSORSIZE, 0, ctypes.c_void_p(size), flags))
        reloaded = bool(spi(self.SPI_SETCURSORS, 0, None, self.SPIF_SENDCHANGE))
        return live or reloaded


def default_backend(platform: str | None = None) -> CursorSizeBackend:
    if (platform or sys.platform).startswith("win"):
        return WindowsCursorBackend()
    return NullCursorBackend()


class BigCursor:
    def __init__(
        self,
        backend: CursorSizeBackend,
        level: int = 2,
        marker: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        debounce_s: float = DEBOUNCE_S,
        publish: Callable[[str, dict[str, Any]], None] | None = None,
        enlarge_debounce_s: float = ENLARGE_DEBOUNCE_S,
    ) -> None:
        self.backend = backend
        self.level = int(level)
        self.marker = marker or default_marker()
        self.clock = clock
        self.debounce_s = debounce_s
        self.enlarge_debounce_s = max(float(debounce_s), float(enlarge_debounce_s))
        self.stuck = False  # v0.1.23: a restore did not take (the size read back was not the original)
        self.publish = publish or (lambda _t, _d: None)
        self.enlarged = False  # we changed the system cursor and have not restored it
        self.original: int | None = None
        self.applied: int | None = None
        self.changes = 0
        self._last_change = -1e9
        self._want = False

    # ----------------------------------------------------------------- marker
    def _write_marker(self, original: int, applied: int) -> None:
        with contextlib.suppress(OSError):
            self.marker.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.marker.with_suffix(".tmp")
            tmp.write_text(json.dumps({"original": original, "applied": applied, "pid": os.getpid()}), "utf-8")
            os.replace(tmp, self.marker)

    def _clear_marker(self) -> None:
        with contextlib.suppress(OSError):
            self.marker.unlink()

    @property
    def user_size_file(self) -> Path:
        return self.marker.with_name(USER_SIZE_NAME)

    def _remember_user_size(self, size: int) -> None:
        # only called with a size seen while Away has not enlarged the cursor: 48 or 64 px are ordinary steps of
        # the Windows pointer-size slider too, so they are recorded like any other size
        with contextlib.suppress(OSError):
            self.user_size_file.write_text(json.dumps({"size": int(size)}), "utf-8")

    def user_size(self) -> int:
        """The last cursor size seen that was the user's own (BASE_SIZE when never seen)."""
        try:
            v = json.loads(self.user_size_file.read_text(encoding="utf-8")).get("size")
            if isinstance(v, int) and MIN_SIZE <= v <= MAX_SIZE:
                return v
        except (OSError, ValueError, AttributeError):
            pass
        return BASE_SIZE

    def read_marker(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("original"), int):
            return None
        return data

    def recover(self) -> bool:
        """Start-up: a marker left by a crash means the cursor is still big. Put the original back."""
        data = self.read_marker()
        if data is None:
            if self.marker.exists():  # unreadable leftover
                self._clear_marker()
            return False
        original = int(data["original"])
        if original in our_sizes() and original != self.user_size():
            # v0.1.23: a marker overwritten by a second copy of the app held our own size as the "original"
            # (unless that size is the user's own: 48/64/96 px are also steps of the Windows slider)
            original = self.user_size()
        ok = self.backend.set(original)
        self._clear_marker()
        self.publish("cursor.size_recovered", {"size": original, "applied": ok})
        return True

    # ---------------------------------------------------------------- control
    def update(self, want_big: bool) -> bool:
        """Desired state (call often). Applies it unless the last change was < ``debounce_s`` ago; a pending
        change goes through on a later call. Returns True when the cursor was changed now."""
        self._want = bool(want_big) and target_size(self.level) > 0
        if self._want == self.enlarged:
            return False
        if not self._want:  # v0.1.23: shrinking back (paused) is immediate
            return self._restore()
        if self.clock() - self._last_change < self.enlarge_debounce_s:
            return False
        return self._enlarge()

    def set_level(self, level: int) -> None:
        """New ``big_cursor_size`` (Settings). Re-applies the size if the cursor is big now."""
        level = int(level)
        if level == self.level:
            return
        self.level = level
        if self.enlarged:
            if target_size(level) <= 0:
                self.restore()
            else:
                self.applied = target_size(level)
                if self.backend.set(self.applied):
                    self._write_marker(self.original or BASE_SIZE, self.applied)
                    self.changes += 1

    def restore(self) -> bool:
        """Put the user's size back now (pause, disable, quit): no debounce."""
        self._want = False
        return self._restore() if self.enlarged else False

    def _enlarge(self) -> bool:
        size = target_size(self.level)
        current = self.backend.get()
        if current is None:  # no system cursor API (macOS/Linux): nothing to change or restore
            return False
        # v0.1.23: the original is never one of our own sizes. A marker left by another run (or another copy of
        # the app) already holds the real original: keep it. A current size that is one of ours means a big
        # cursor was left behind: the original is the last size known to be the user's. With no size known yet the
        # current one is the user's (48 and 64 px are also ordinary steps of the Windows pointer-size slider).
        marked = self.read_marker()
        user = self.user_size()
        if marked is not None and (
            int(marked["original"]) not in our_sizes() - {BASE_SIZE} or marked["original"] == user
        ):
            original = int(marked["original"])
        elif current in our_sizes() and current != BASE_SIZE and current != user and self.user_size_file.exists():
            original = user  # a size of ours that is not the user's last known one: a big cursor left behind
        else:
            original = current
            self._remember_user_size(current)
        self.original = original
        self._write_marker(original, size)  # before touching the system: a crash right after can be undone
        if not self.backend.set(size):
            self._clear_marker()
            self.original = None
            self._last_change = self.clock()  # do not retry every frame
            return False
        self.enlarged, self.applied = True, size
        self._last_change = self.clock()
        self.changes += 1
        self.publish("cursor.size", {"size": size, "big": True})
        return True

    def _restore(self) -> bool:
        original = self.original if self.original is not None else BASE_SIZE
        ok = self.backend.set(original)
        ok = self._verify(original, ok)
        if ok:
            self._clear_marker()
        self.enlarged, self.applied = False, None
        self._last_change = self.clock()
        self.changes += 1
        self.publish("cursor.size", {"size": original, "big": False, "applied": ok})
        return True

    def _verify(self, original: int, ok: bool) -> bool:
        """v0.1.23: read the size back after a restore; try once more, then report it stuck (the tray offers
        "Reset cursor size to normal")."""
        now = self.backend.get()
        if now is None or now == original:
            self.stuck = False
            return ok
        ok = self.backend.set(original)
        now = self.backend.get()
        if now is None or now == original:
            self.stuck = False
            return ok
        self.stuck = True
        self.publish(
            "cursor.size_stuck",
            {
                "size": now,
                "want": original,
                "message": "Your mouse pointer is still bigger than normal. Use \u201cReset cursor size to "
                "normal\u201d in the tray menu (or Settings) to fix it.",
            },
        )
        return False

    def reset(self) -> int:
        """v0.1.23 "Reset cursor size to normal" (tray, Settings, ``away cursor reset``): the user's last known
        own size (32 px when unknown); the marker is cleared. Returns the size set."""
        marked = self.read_marker()
        size = self.user_size()
        if marked is not None and int(marked["original"]) not in our_sizes() - {BASE_SIZE}:
            size = int(marked["original"])
        self.backend.set(size)
        self._verify(size, True)
        self._clear_marker()
        self.enlarged, self.applied, self.original = False, None, None
        self._last_change = self.clock()
        self.publish("cursor.size_reset", {"size": size})
        return size
