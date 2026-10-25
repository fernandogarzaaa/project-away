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
from pathlib import Path
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


def warm_up(platform: str | None = None, hiservices: Any = None) -> bool:
    """Resolve the HIServices names on the calling thread, before any other thread needs them.

    PyObjC fills a framework module in lazily on its first attribute access, and that is not thread-safe: pynput's
    keyboard and mouse listeners (and the permission check) each ask for ``AXIsProcessTrusted`` at the same moment
    on start-up, and the loser dies with ``KeyError: 'AXIsProcessTrusted'`` (a dead hotkey or click listener).
    Touching the names once from the main thread makes every later access a plain lookup."""
    if (platform or sys.platform) != "darwin":
        return False
    try:
        hs = hiservices if hiservices is not None else _hiservices()
        for name in ("AXIsProcessTrusted", "AXIsProcessTrustedWithOptions", "kAXTrustedCheckOptionPrompt"):
            getattr(hs, name)
        return True
    except Exception:
        return False


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


BUNDLE_ID = "com.fernandogarzaaa.projectaway"
TCC_SERVICES = ("Accessibility", "ListenEvent")  # tccutil's names for Accessibility and Input Monitoring
IDENTITY_FILE = "macperms-identity.json"


def app_bundle(executable: str | None = None) -> Path | None:
    """The ``.app`` this process runs from (``.../Project Away.app/Contents/MacOS/ProjectAway``), or None."""
    exe = Path(executable or sys.executable).resolve()
    app = exe.parent.parent.parent
    return app if exe.parent.name == "MacOS" and app.suffix == ".app" else None


def signing_identity(app: Path, run: Callable[..., Any] = subprocess.run) -> str | None:
    """The designated requirement macOS ties the app's permissions to (``codesign -d -r-``), or None."""
    try:
        r = run(["codesign", "-d", "-r-", str(app)], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    out = f"{r.stdout or ''}{r.stderr or ''}"
    line = next((ln for ln in out.splitlines() if ln.startswith("designated =>")), "")
    if not line:
        return None
    return line.split("=>", 1)[1].strip() or None


def clear_stale(
    data_dir: Path,
    platform: str | None = None,
    hiservices: Any = None,
    executable: str | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> bool:
    """v0.1.35: drop Accessibility / Input Monitoring entries left by a build signed differently. True when cleared.

    macOS ties a permission to the app's signing identity. Before v0.1.35 every release was ad-hoc signed, so each
    one was a new app: System Settings kept showing Project Away switched on, but the switch belonged to the old
    build and the new one was asked again and again, however often it was toggled. When this build is not trusted
    and its identity differs from the one recorded on the last start, its entries are reset once, so the prompt
    that follows adds a fresh entry for this build. Never twice for one identity: a permission you turn off stays off.
    """
    import json

    if (platform or sys.platform) != "darwin" or accessibility(False, platform, hiservices) is not False:
        return False
    app = app_bundle(executable)
    ident = signing_identity(app, run) if app is not None else None
    if not ident:
        return False
    marker = Path(data_dir) / IDENTITY_FILE
    try:
        seen = json.loads(marker.read_text(encoding="utf-8")).get("identity")
    except (OSError, ValueError, AttributeError):
        seen = None
    if seen == ident:
        return False
    for service in TCC_SERVICES:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            run(["tccutil", "reset", service, BUNDLE_ID], capture_output=True, timeout=15, check=False)
    with contextlib.suppress(OSError):
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"identity": ident}), encoding="utf-8")
    import logging

    logging.getLogger("away.macperms").info("permissions: cleared entries of a differently signed build (%s)", ident)
    return True


def missing_names(state: dict[str, bool | None]) -> list[str]:
    names = {"accessibility": "Accessibility", "input_monitoring": "Input Monitoring"}
    return [names[k] for k in ("accessibility", "input_monitoring") if state.get(k) is False]


def check_and_ask(
    notify: Callable[[str, str], object],
    platform: str | None = None,
    hiservices: Any = None,
    iokit: Any = None,
    data_dir: Path | None = None,
) -> dict[str, bool | None]:
    """Start-up (macOS): check both permissions, let macOS ask for missing ones, and say once what to allow.
    ``data_dir`` (v0.1.35): first clear entries a differently signed build left behind (:func:`clear_stale`)."""
    if data_dir is not None:
        with contextlib.suppress(Exception):
            clear_stale(data_dir, platform, hiservices)
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
