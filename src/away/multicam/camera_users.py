"""Which app is using a camera right now (v0.1.23, Windows).

Windows records camera use per app under
``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\webcam``: packaged
(Store) apps as subkeys like ``Microsoft.WindowsCamera_8wekyb3d8bbwe``, desktop apps under ``NonPackaged`` as the
program path with ``#`` for ``\\`` (``C:#Program Files#NVIDIA Corporation#NVIDIA Broadcast#NVIDIA Broadcast.exe``).
An entry whose ``LastUsedTimeStop`` is 0 (with a ``LastUsedTimeStart``) is using a camera now. Windows does not
say *which* camera, so the message names the app(s) and the camera that failed.

NVIDIA Broadcast keeps its source camera open; when that source is the built-in camera, Broadcast's virtual
camera *is* the built-in camera, filtered. :func:`broadcast_source` reads Broadcast's own settings to find out.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
        actions = [f"close:{u.path}" for u in users if not u.packaged][:1] + ["retry"]
    else:
        msg = f"{label} could not be opened and no other app seems to be using it right now."
        actions = ["retry"]
    return {"message": msg, "users": [u.public() for u in users], "actions": actions, "broadcast_uses_it": via}
