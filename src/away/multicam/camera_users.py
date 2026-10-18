"""Which app is using a camera right now (v0.1.23, Windows).

Windows records camera use per app under
``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\webcam``: packaged
(Store) apps as subkeys like ``Microsoft.WindowsCamera_8wekyb3d8bbwe``, desktop apps under ``NonPackaged`` as the
program path with ``#`` for ``\\`` (``C:#Program Files#NVIDIA Corporation#NVIDIA Broadcast#NVIDIA Broadcast.exe``).
An entry whose ``LastUsedTimeStop`` is 0 (with a ``LastUsedTimeStart``) is using a camera now. Windows does not
say *which* camera, so the message names the app(s) and the camera that failed.

NVIDIA Broadcast keeps its source camera open; when that source is the built-in camera, Broadcast's virtual
camera *is* the built-in camera, filtered. :func:`broadcast_source` reads Broadcast's own settings to find out.

:func:`holders` (proposal #13, v0.1.28) answers "who is holding my camera" for the diagnose tool: the apps
Windows says hold a camera now, a live scan of known camera apps (a hint, not proof), and the last-used
timestamps Windows stores per app.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEY = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
OWN = ("projectaway", "project away", "away.exe", "python")


@dataclass(frozen=True)
class CameraUser:
    name: str  # "NVIDIA Broadcast"
    path: str  # program path (desktop apps) or package name
    packaged: bool

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "packaged": self.packaged}


def _pretty(key: str, packaged: bool) -> str:
    if packaged:
        base = key.split("_", 1)[0]
        base = base.split(".")[-1] if "." in base else base
        return {"WindowsCamera": "Windows Camera", "MicrosoftTeams": "Microsoft Teams"}.get(base, base)
    exe = key.replace("#", "\\").rsplit("\\", 1)[-1]
    name = exe[:-4] if exe.lower().endswith(".exe") else exe
    from away.app.sysload import friendly_name

    return friendly_name(name)


def parse(entries: list[tuple[str, bool, dict[str, Any]]]) -> list[CameraUser]:
    """``entries``: (subkey name, packaged, values). Users with LastUsedTimeStop == 0 and a start time."""
    out: list[CameraUser] = []
    for key, packaged, vals in entries:
        start = int(vals.get("LastUsedTimeStart") or 0)
        stop = vals.get("LastUsedTimeStop")
        if stop is None or int(stop) != 0 or start == 0:
            continue
        name = _pretty(key, packaged)
        low = (key + " " + name).lower()
        if any(o in low for o in OWN):
            continue  # Project Away itself
        out.append(CameraUser(name, key.replace("#", "\\") if not packaged else key, packaged))
    return out


def read_registry(winreg_mod: Any = None) -> list[tuple[str, bool, dict[str, Any]]]:  # pragma: no cover - Windows
    wr = winreg_mod
    if wr is None:
        if sys.platform != "win32":
            return []
        import winreg as wr  # type: ignore[no-redef]

    def values(k: Any) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for name in ("LastUsedTimeStart", "LastUsedTimeStop"):
            with contextlib.suppress(OSError):
                d[name] = wr.QueryValueEx(k, name)[0]
        return d

    def subkeys(k: Any) -> list[str]:
        names, i = [], 0
        while True:
            try:
                names.append(wr.EnumKey(k, i))
            except OSError:
                return names
            i += 1

    out: list[tuple[str, bool, dict[str, Any]]] = []
    try:
        root = wr.OpenKey(wr.HKEY_CURRENT_USER, KEY)
    except OSError:
        return out
    with root:
        for name in subkeys(root):
            with contextlib.suppress(OSError), wr.OpenKey(root, name) as k:
                if name == "NonPackaged":
                    for sub in subkeys(k):
                        with contextlib.suppress(OSError), wr.OpenKey(k, sub) as k2:
                            out.append((sub, False, values(k2)))
                else:
                    out.append((name, True, values(k)))
    return out


def current_users(reader: Callable[[], list[tuple[str, bool, dict[str, Any]]]] = read_registry) -> list[CameraUser]:
    try:
        return parse(reader())
    except Exception:
        return []


# ---------------------------------------------------------------------------------- holders (v0.1.28)
# Proposal #13: first-class "who is holding my camera". The registry only says who used a camera *last* (one
# start/stop pair per app; Windows does not say which camera either), so "holders" answers the question with
# three honest pieces: what Windows says holds it now (LastUsedTimeStop == 0), a live scan of known camera apps
# (running now, so they *may* hold it - a hint, never a claim), and who used it before with their last timestamps.

# 100-ns ticks between the Windows epoch (1601-01-01) and the Unix epoch (1970-01-01)
_FILETIME_UNIX_DIFF = 116444736000000000


def filetime_to_iso(ft: Any) -> str:
    """A Windows FILETIME value (100-ns ticks since 1601-01-01 UTC) as an ISO timestamp, or "" when it is not
    a plausible timestamp."""
    try:
        seconds = (int(ft) - _FILETIME_UNIX_DIFF) / 10_000_000
        if not 0 <= seconds <= 4102444800:  # 1970-01-01 .. 2100-01-01
            return ""
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def last_used(entries: list[tuple[str, bool, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Apps the registry says used a camera and stopped (``LastUsedTimeStop`` != 0), most recent first:
    ``[{"name", "path", "packaged", "at"}]`` where ``at`` is an ISO timestamp ("" when Windows stored junk).
    Project Away itself is excluded."""
    out: list[dict[str, Any]] = []
    for key, packaged, vals in entries:
        start = int(vals.get("LastUsedTimeStart") or 0)
        stop = int(vals.get("LastUsedTimeStop") or 0)
        if start == 0 or stop == 0:
            continue  # never used it, or still using it now (see parse())
        name = _pretty(key, packaged)
        if any(o in (key + " " + name).lower() for o in OWN):
            continue
        out.append(
            {
                "name": name,
                "path": key.replace("#", "\\") if not packaged else key,
                "packaged": packaged,
                "at": filetime_to_iso(stop),
            }
        )
    out.sort(key=lambda d: (d["at"] != "", d["at"]), reverse=True)  # newest first; broken timestamps last
    return out


# process base name (lowercase, no .exe) -> friendly name
CAMERA_PROCESSES = {
    "nvbroadcast": "NVIDIA Broadcast",
    "obs64": "OBS Studio",
    "obs32": "OBS Studio",
    "windowscamera": "Windows Camera",
    "ms-teams": "Microsoft Teams",
    "teams": "Microsoft Teams",
    "zoom": "Zoom",
    "discord": "Discord",
    "skype": "Skype",
    "slack": "Slack",
    "webex": "Cisco Webex",
    "cct": "Cisco Webex",
    "gotomeeting": "GoTo Meeting",
    "whereby": "Whereby",
    "meetnow": "Meet Now",
}


def running_camera_apps(psutil_mod: Any = None) -> list[dict[str, Any]]:
    """The known camera apps (see :data:`CAMERA_PROCESSES`) running right now: ``[{"name", "path"}]``, [] when
    psutil is not installed or the process list cannot be read. A hint only: being open does not mean the app
    holds the camera."""
    try:
        ps = psutil_mod
        if ps is None:
            import psutil as ps  # type: ignore[no-redef]
    except Exception:
        return []
    try:
        procs = list(ps.process_iter(["name", "exe"]))
    except Exception:
        return []
    seen: dict[str, dict[str, Any]] = {}
    for p in procs:
        try:
            info = p.info or {}
            name = str(info.get("name") or "")
            base = os.path.splitext(os.path.basename(name).lower())[0]
            if base in CAMERA_PROCESSES and base not in seen:
                seen[base] = {
                    "name": CAMERA_PROCESSES[base],
                    "path": str(info.get("exe") or name),
                }
        except Exception:
            continue
    return sorted(seen.values(), key=lambda d: d["name"])


def holders(
    read: Callable[[], list[tuple[str, bool, dict[str, Any]]]] = read_registry,
    psutil_mod: Any = None,
) -> dict[str, Any]:
    """Who holds a camera now plus who used it last (proposal #13).

    ``{"now": [{"name", "path", "source"}], "last_used": [{"name", "path", "packaged", "at"}]}``. ``source`` is
    "registry" when Windows reports the app holds a camera now, or "process" when a known camera app is running
    (a hint, not proof). Windows never names *which* camera an app holds, so this answers "which app", not
    "which camera".
    """
    try:
        entries = read()
    except Exception:
        entries = []
    try:
        now: list[dict[str, Any]] = [{"name": u.name, "path": u.path, "source": "registry"} for u in parse(entries)]
    except Exception:
        now = []
    have = {n["name"].lower() for n in now}
    for app in running_camera_apps(psutil_mod):
        if app["name"].lower() not in have:
            now.append({**app, "source": "process"})
    try:
        last = last_used(entries)
    except Exception:
        last = []
    return {"now": now, "last_used": last}


def broadcast_settings_paths(environ: dict[str, str] | None = None) -> list[Path]:
    env = os.environ if environ is None else environ
    roots = [env.get("LOCALAPPDATA", ""), env.get("APPDATA", ""), env.get("PROGRAMDATA", "")]
    out = []
    for r in roots:
        if r:
            out += [Path(r) / "NVIDIA Corporation" / "NVIDIA Broadcast" / "settings.json"]
    return out


def broadcast_source(read: Callable[[Path], str] | None = None, paths: list[Path] | None = None) -> str:
    """The camera NVIDIA Broadcast takes its picture from ("HD User Facing"), "" when unknown."""
    rd = read or (lambda p: p.read_text(encoding="utf-8", errors="replace"))
    for p in paths if paths is not None else broadcast_settings_paths():
        try:
            data = json.loads(rd(p))
        except Exception:
            continue
        found = _find_source(data)
        if found:
            return found
    return ""


def _find_source(obj: Any) -> str:
    if isinstance(obj, dict):
        cam = None
        for k, v in obj.items():
            lk = str(k).lower()
            if isinstance(v, str) and ("camera" in lk or "device" in lk) and ("source" in lk or "input" in lk):
                return v
            if "camera" in lk:
                cam = v
        for v in ([cam] if cam is not None else []) + list(obj.values()):
            got = _find_source(v)
            if got:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_source(v)
            if got:
                return got
    return ""


def explain(camera: str, users: list[CameraUser], broadcast_from: str = "") -> dict[str, Any]:
    """A plain message + buttons for a camera that will not open.

    ``{"message": "Your built-in camera is being used by NVIDIA Broadcast", "users": [...], "actions":
    ["close:<path>", "retry"], "broadcast_uses_it": bool}``."""
    from away.multicam.capture import is_internal_camera

    label = "Your built-in camera" if is_internal_camera(camera) else f"{camera or 'The camera'}"
    via = bool(broadcast_from) and broadcast_from.strip().lower() in str(camera).strip().lower()
    names = [u.name for u in users]
    actions: list[str] = []
    if via:
        msg = (
            f"{label} is used by NVIDIA Broadcast (its camera shows your {label.split()[-1]} with effects). Use the "
            "built-in camera through NVIDIA Broadcast, or release it in Broadcast."
        )
        actions = ["use_broadcast", "release_broadcast", "retry"]
    elif names:
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        msg = f"{label} is being used by {who}. Close it, then Retry."
        actions = [*[f"close:{u.path}" for u in users if not u.packaged][:1], "retry"]
    else:
        msg = f"{label} could not be opened and no other app seems to be using it right now."
        actions = ["retry"]
    return {"message": msg, "users": [u.public() for u in users], "actions": actions, "broadcast_uses_it": via}
