"""Turn a camera off and on again (v0.1.23), from the tray, Settings > Cameras and Agent Link (``cameras`` changes).

A camera that is turned off moves from ``hands.cameras`` to ``hands.disabled_cameras`` with all its settings, so it
is never opened or retried while off and "Turn on" puts it back exactly. The last camera cannot be turned off (hand
tracking needs one); the camera left alone becomes the primary one.
"""

from __future__ import annotations

import copy
from typing import Any

from away import config as cfgmod


class CameraSwitchError(ValueError):
    pass


def _same(a: Any, b: str) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def configured(cfg: Any) -> list[str]:
    h = cfg.hands
    return [str(c.device) for c in h.cameras] if h.cameras else [str(h.camera_index)]


def listing(cfg: Any) -> list[dict[str, Any]]:
    """Every camera Project Away knows, on or off (what an agent sees in ``settings list cameras``)."""
    h = cfg.hands
    out = [{"key": d, "title": d, "value": "on", "kind": "camera"} for d in configured(cfg)]
    out += [{"key": str(c.device), "title": str(c.device), "value": "off", "kind": "camera"} for c in h.disabled_cameras]
    return out


def turn_off(cfg: Any, device: str) -> str:
    h = cfg.hands
    if any(_same(c.device, device) for c in h.disabled_cameras):
        return f"{device} is already off"
    idx = next((i for i, c in enumerate(h.cameras) if _same(c.device, device)), None)
    if idx is None:
        if not h.cameras and _same(h.camera_index, device):
            raise CameraSwitchError(f"{device} is your only camera; it can't be turned off (hand tracking needs one)")
        raise CameraSwitchError(f"there is no camera called {device!r} (cameras: {', '.join(configured(cfg))})")
    if len(h.cameras) <= 1:
        raise CameraSwitchError(f"{device} is your only camera; it can't be turned off (hand tracking needs one)")
    spec = h.cameras[idx]
    rest = [copy.copy(c) for j, c in enumerate(h.cameras) if j != idx]
    if not any(c.role == "primary" for c in rest):
        rest[0].role = "primary"
    if len(rest) == 1:  # one camera left: single-camera mode on it (same as removing it in Settings)
        only = rest[0]
        h.camera_index = only.device
        h.camera_hint = only.index_hint
        h.camera_fourcc = only.fourcc
        rest = []
    off = copy.copy(spec)
    h.cameras = rest
    h.disabled_cameras = [*h.disabled_cameras, off]
    return f"Turned off {spec.device}"


def turn_on(cfg: Any, device: str) -> str:
    h = cfg.hands
    idx = next((i for i, c in enumerate(h.disabled_cameras) if _same(c.device, device)), None)
    if idx is None:
        if any(_same(d, device) for d in configured(cfg)):
            return f"{device} is already on"
        raise CameraSwitchError(f"{device!r} is not a camera that was turned off")
    spec = copy.copy(h.disabled_cameras[idx])
    cams = [copy.copy(c) for c in h.cameras]
    if not cams:  # single-camera mode: the current camera becomes the primary of a two-camera set
        cams = [
            cfgmod.CameraSpec(
                device=h.camera_index, role="primary", index_hint=h.camera_hint, fourcc=h.camera_fourcc
            )
        ]
    if len(cams) == 1:
        cams[0].role = "primary"
    if spec.role == "primary" and any(c.role == "primary" for c in cams):
        spec.role = "secondary"
    h.cameras = [*cams, spec]
    h.disabled_cameras = [c for j, c in enumerate(h.disabled_cameras) if j != idx]
    return f"Turned on {spec.device}"
