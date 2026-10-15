"""The fix buttons of the self-test report (v0.1.24).

Each :class:`~away.selftest.model.Fix` names an action; :func:`apply` does it and answers in plain words
(``{"ok": True, "message": "Closed NVIDIA Broadcast. Press Try again."}``). Only the person presses these buttons:
agents get the same report, but no fix runs without the person. Actions that close another program say so on the
button itself ("Close NVIDIA Broadcast").

v0.1.28 (proposal #11): ``open:device_manager`` opens Windows Device Manager (an MMC console, ``devmgmt.msc``,
not a Settings page), and ``guide:uninstall_driver`` walks the person through the driver repair (uninstall the
camera, restart the PC) after a Yes / No confirm, in plain words.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

OPEN_TARGETS = {  # Windows Settings pages (ms-settings: URIs); other systems are told where to look
    "camera_privacy": ("ms-settings:privacy-webcam", "Settings > Privacy > Camera"),
    "sound_settings": ("ms-settings:sound", "your sound settings"),
    "notification_settings": ("ms-settings:notifications", "your notification settings"),
}
SETTABLE = {"hands.compute": ("auto", "gpu", "cpu")}


def _ok(message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": True, "message": message, **extra}


def _no(message: str) -> dict[str, Any]:
    return {"ok": False, "message": message}


def apply(
    action: str,
    app: Any = None,
    host: Any = None,
    psutil_mod: Any = None,
    launcher: Any = None,
    confirm: Callable[[str], bool | None] | None = None,
) -> dict[str, Any]:
    """``confirm`` (v0.1.28): an injected Yes / No answer for the guided flows (the app's own prompt window is
    used when ``host`` carries one)."""
    kind, _, arg = str(action or "").partition(":")
    try:
        if kind == "retry_cameras":
            if app is None or not hasattr(app, "retry_cameras"):
                return _no("Start Project Away, then run the self-test again.")
            app.retry_cameras()
            return _ok("Trying the cameras again now.")
        if kind in ("close_app", "close_pid"):
            return _close(kind, arg, psutil_mod)
        if kind == "close_duplicates":
            return _close_duplicates(psutil_mod)
        if kind == "set":
            return _set(arg, host)
        if kind == "download" and arg == "gpu_tracking":
            from away.accel import models as gm

            gm.download("hands")
            with contextlib.suppress(Exception):
                gm.download("face")
            return _ok("Downloaded. Hand tracking moves to the graphics card the next time Project Away starts.")
        if kind == "reset_cursor":
            from away.app import bigcursor as bc

            ctl = getattr(app, "big_cursor", None) or bc.BigCursor(bc.default_backend())
            size = ctl.reset()
            return _ok(f"Your pointer is back to {size} px." if size else "Your pointer size is back.")
        if kind in ("new_link", "backup_link", "restart_link"):
            if host is None:
                return _no("Start Project Away first.")
            if kind == "backup_link":
                host.backup_link()
            elif kind == "new_link":
                host.new_link()
            else:
                host.stop_tunnel()
                host.restore_link("restarted from the self-test")
            return _ok("Done. Your agents are told the new address by themselves.")
        if kind == "repair_models":
            from away.app.frozen import away_command

            (launcher or subprocess.Popen)(away_command(["setup", "--yes"]))
            return _ok("Downloading the missing models in the background; this can take a few minutes.")
        if kind == "open":
            return _open(arg, launcher)
        if kind == "guide" and arg == "uninstall_driver":
            return _guide_uninstall_driver(launcher, host, confirm)
        if kind == "camera_on":
            return _camera_on(arg, host)
        if kind == "use_broadcast":
            return _set("hands.virtual_fallback=true", host)
        if kind == "restart_app":
            return _ok("Quit Project Away from its tray icon and start it again.")
        if kind == "rerun":
            st = getattr(host, "selftest", None)
            if st is None:
                return _no("Start Project Away first.")
            st.start(trigger="you", only=[arg])
            return _ok("Running that check again.")
    except Exception as exc:
        return _no(f"That did not work: {str(exc)[:160]}")
    return _no("Nothing to do for this one.")


def _close(kind: str, arg: str, psutil_mod: Any = None) -> dict[str, Any]:
    ps = psutil_mod
    if ps is None:
        import psutil as ps  # type: ignore[no-redef]
    want = arg.replace("/", "\\").lower()
    base = want.rsplit("\\", 1)[-1]
    closed = []
    for p in ps.process_iter(["pid", "name", "exe"]):
        try:
            pid = int(p.info["pid"])
            if kind == "close_pid":
                hit = pid == int(arg)
            else:
                exe = str(p.info.get("exe") or "").replace("/", "\\").lower()
                name = str(p.info.get("name") or "").lower()
                hit = (bool(exe) and (exe == want or exe.endswith("\\" + base))) or name in (want, base)
            if hit and pid != os.getpid():
                closed.append(str(p.info.get("name") or pid))
                p.terminate()
        except Exception:
            continue
    if not closed:
        return _no("That program is not running any more.")
    return _ok(f"Closed {', '.join(sorted(set(closed)))}. Press Try again if a camera was waiting for it.")


def _close_duplicates(psutil_mod: Any = None) -> dict[str, Any]:
    from away.selftest.env import app_processes

    ps = psutil_mod
    if ps is None:
        import psutil as ps  # type: ignore[no-redef]
    pids = app_processes(ps)
    keep = os.getpid() if os.getpid() in pids else (min(pids) if pids else None)
    n = 0
    for pid in pids:
        if pid == keep:
            continue
        with contextlib.suppress(Exception):
            ps.Process(pid).terminate()
            n += 1
    return _ok(f"Closed {n} extra cop{'y' if n == 1 else 'ies'}.") if n else _no("Only one copy is running now.")


def _set(arg: str, host: Any) -> dict[str, Any]:
    key, _, value = arg.partition("=")
    val: Any = value
    if value.lower() in ("true", "false"):
        val = value.lower() == "true"
    elif key in SETTABLE and value not in SETTABLE[key]:
        return _no("That value is not allowed.")
    changes = getattr(getattr(host, "core", None), "changes", None)
    if changes is not None:
        from away.agentlink.registry import Op

        changes.apply([Op.from_dict({"domain": "settings", "op": "set", "key": key, "value": val})], who="self-test")
        return _ok("Changed (undo it in Agent Link > Changes). It applies the next time tracking starts.")
    from away import config as cfgmod

    path = cfgmod.config_path()
    cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
    obj: Any = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], val)
    cfgmod.validate(cfg)
    cfgmod.save(cfg, path)
    return _ok("Changed. It applies the next time Project Away starts.")


def _camera_on(name: str, host: Any) -> dict[str, Any]:
    changes = getattr(getattr(host, "core", None), "changes", None)
    if changes is None:
        return _no("Turn it on in Settings > Cameras.")
    from away.agentlink.registry import Op

    changes.apply([Op.from_dict({"domain": "cameras", "op": "enable", "key": name})], who="self-test")
    return _ok(f"Turned on {name}.")


def _open(target: str, launcher: Any = None) -> dict[str, Any]:
    if target.startswith("settings/"):
        from away.app.frozen import away_command

        (launcher or subprocess.Popen)(away_command(["settings", "--tab", target.split("/", 1)[1]]))
        return _ok("Opening Settings.")
    if target == "device_manager":
        return _open_device_manager(launcher)
    uri, words = OPEN_TARGETS.get(target, ("", target))
    if sys.platform.startswith("win") and uri:
        os.startfile(uri)  # type: ignore[attr-defined]
        return _ok(f"Opened {words}.")
    return _ok(f"Open {words} on this computer.")


def _open_device_manager(launcher: Any = None) -> dict[str, Any]:
    """v0.1.28 (proposal #11): Device Manager is an MMC console (``devmgmt.msc``), not a Settings page, so it is
    launched directly, through the same injectable ``launcher`` pattern as the other opens. Safe: only the person
    presses the button, on this PC; agents never reach this."""
    if sys.platform.startswith("win"):
        (launcher or subprocess.Popen)(["mmc.exe", "devmgmt.msc"])
        return _ok("Opening Device Manager (Windows may ask for permission).")
    return _ok("Open Device Manager on this computer to look at the camera's driver.")


# --------------------------------------------------------------------------------- driver repair guide
UNINSTALL_DRIVER_QUESTION = (
    "Your camera's driver looks broken. The repair is to uninstall the camera in Device Manager and restart the PC "
    "(Windows installs the driver again by itself). Do that now?"
)
UNINSTALL_DRIVER_STEPS = (
    "In Device Manager, open Cameras, right-click the camera, choose Uninstall device (leave 'Delete the driver "
    "software' unticked), then restart the PC. Windows installs the camera again when it starts up; afterwards, "
    "press Try again in the self-test."
)


def _confirm_person(question: str, host: Any, confirm: Callable[[str], bool | None] | None) -> bool | None:
    """Yes / No from the person, reusing the Check-my-PC confirm pattern (v0.1.24): the app's own prompt window
    when the app runs (``host.core.prompts``), an injected answer in tests, None when there is no way to ask."""
    if confirm is not None:
        try:
            return confirm(question)
        except Exception:
            return None
    pr = getattr(getattr(host, "core", None), "prompts", None)
    if pr is None:
        return None
    try:
        res = pr.ask(question, ["Yes", "No"], title="Project Away self-test", agent="Project Away", timeout_s=60.0)
    except Exception:
        return None
    ans = str(res.get("answer") or "").lower()
    return True if ans == "yes" else (False if ans == "no" else None)


def _guide_uninstall_driver(launcher: Any, host: Any, confirm: Callable[[str], bool | None] | None) -> dict[str, Any]:
    """v0.1.28 (proposal #11): the guided repair for a driver-broken camera. Confirms with the person first, then
    opens Device Manager and hands over the steps in plain words. A "no" changes nothing; with no way to ask, the
    person just gets the steps."""
    ans = _confirm_person(UNINSTALL_DRIVER_QUESTION, host, confirm)
    if ans is False:
        return _ok("OK, nothing changed.")
    opened = _open_device_manager(launcher)
    tail = " (no answer came back, so here are the steps)" if ans is None else ""
    return _ok(f"{opened.get('message')}{tail} {UNINSTALL_DRIVER_STEPS}")
