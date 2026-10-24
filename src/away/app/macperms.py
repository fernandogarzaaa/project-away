"""macOS privacy permissions for pointer and keyboard control (v0.1.32).

Without **Accessibility** macOS silently ignores the pointer moves, clicks and typing Project Away sends, and
without **Input Monitoring** the hotkeys never fire. Before, the only sign was a line in the log ("This process is
not trusted! Input event monitoring will not be possible...") that nobody sees from a menu-bar app.

At start-up the app now checks both. A missing one makes macOS show its own request (the system dialog with
"Open System Settings"), and the app shows one message saying what to allow and that the app must be reopened
afterwards. The tray menu has "Allow pointer and keyboard control..." to open the right System Settings page.

* Accessibility: ``AXIsProcessTrustedWithOptions`` from ``HIServices`` (PyObjC; pynput already uses it).
* Input Monitoring: ``IOHIDCheckAccess`` / ``IOHIDRequestAccess`` from IOKit (ctypes; macOS 10.15+).

Every call is best effort: on another OS, an older macOS or any error the answer is ``None`` (unknown), never a
crash and never a nag.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from collections.abc import Callable
from typing import Any

IOKIT = "/System/Library/Frameworks/IOKit.framework/IOKit"
LISTEN_EVENT = 1  # kIOHIDRequestTypeListenEvent: Input Monitoring
GRANTED = 0  # kIOHIDAccessTypeGranted (1 = denied, 2 = not decided yet)

SETTINGS = {
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "input_monitoring": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
}

MESSAGE_TITLE = "Project Away needs two permissions"
MESSAGE = (
    "To move the pointer, click and use hotkeys, allow Project Away in System Settings > Privacy & Security > "
    "{missing}. Then quit Project Away (menu bar icon > Quit) and open it again."
)


def _hiservices() -> Any:
    import HIServices  # type: ignore[import-not-found]  # PyObjC, bundled on macOS (pynput needs it)

    return HIServices


def _iokit() -> Any:
    import ctypes

    lib = ctypes.cdll.LoadLibrary(IOKIT)
    for name in ("IOHIDCheckAccess", "IOHIDRequestAccess"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_uint32 if name == "IOHIDCheckAccess" else ctypes.c_bool
    return lib


def accessibility(ask: bool = False, platform: str | None = None, hiservices: Any = None) -> bool | None:
    """Whether macOS lets this app control the pointer and keyboard. ``ask``: when not, macOS shows its own
    request (and lists the app in System Settings, switched off). None: unknown (not macOS, or no API)."""
    if (platform or sys.platform) != "darwin":
        return None
    try:
        hs = hiservices if hiservices is not None else _hiservices()
        if ask:
            return bool(hs.AXIsProcessTrustedWithOptions({hs.kAXTrustedCheckOptionPrompt: True}))
        return bool(hs.AXIsProcessTrusted())
    except Exception:
        return None


def input_monitoring(ask: bool = False, platform: str | None = None, iokit: Any = None) -> bool | None:
    """Whether macOS lets this app see key presses (hotkeys). ``ask``: when not decided yet, macOS asks."""
    if (platform or sys.platform) != "darwin":
        return None
    try:
        lib = iokit if iokit is not None else _iokit()
        if int(lib.IOHIDCheckAccess(LISTEN_EVENT)) == GRANTED:
            return True
        if ask:
            return bool(lib.IOHIDRequestAccess(LISTEN_EVENT))
        return False
    except Exception:
        return None


def missing_names(state: dict[str, bool | None]) -> list[str]:
    names = {"accessibility": "Accessibility", "input_monitoring": "Input Monitoring"}
    return [names[k] for k in ("accessibility", "input_monitoring") if state.get(k) is False]


def check_and_ask(
    notify: Callable[[str, str], object],
    platform: str | None = None,
    hiservices: Any = None,
    iokit: Any = None,
) -> dict[str, bool | None]:
    """Start-up (macOS): check both permissions, let macOS ask for missing ones, and say once what to allow."""
    state = {
        "accessibility": accessibility(True, platform, hiservices),
        "input_monitoring": input_monitoring(True, platform, iokit),
    }
    missing = missing_names(state)
    if missing:
        with contextlib.suppress(Exception):
            notify(
                MESSAGE_TITLE if len(missing) > 1 else "Project Away needs a permission",
                MESSAGE.format(missing=" and ".join(missing)),
            )
    return state


def open_settings(which: str = "accessibility", run: Callable[..., Any] = subprocess.run) -> bool:
    """Open System Settings at the Accessibility (or Input Monitoring) list."""
    try:
        run(["open", SETTINGS[which]], timeout=10, check=False)
        return True
    except (OSError, subprocess.SubprocessError, KeyError):
        return False
