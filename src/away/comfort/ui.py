"""Words and colours for the attention gate / comfort features, and friendly error messages with a fix button.

Pure functions only (no tkinter, no pystray), so every text a user can see is unit-tested.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

FATIGUE_TITLE = "Project Away: time to rest your arm?"
BREAK_TITLE = "Project Away: time for a short break"
SWITCH_BACK_MESSAGE = "Welcome back! Your hand is in control again."

# indicator colours (at-cursor ring tint, Getting started tab dot)
ATTENTION_COLORS = {
    "off": "#9E9E9E",
    "idle": "#9E9E9E",
    "arming": "#FFD54F",
    "engaged": "#7CFC9A",
    "grace": "#C5E1A5",
}
ATTENTION_LABELS = {
    "off": "Always responding",
    "idle": "Waiting for you",
    "arming": "Getting ready",
    "engaged": "Listening to your hand",
    "grace": "Still with you",
}


def fatigue_message(switch: bool, face_ok: bool = True) -> str:
    if switch and face_ok:
        return (
            "Your arm looks tired. Lower your hand and move your head to steer the pointer. "
            "Raise your hand again whenever you want it back."
        )
    if switch:
        return "Your arm looks tired. Rest it for a moment (head control needs face tracking, which is switched off)."
    return (
        "Your arm looks tired. Rest it for a moment. Tip: Settings > Getting started > "
        "'Switch to head control when my arm is tired'."
    )


def break_message(minutes: int) -> str:
    return (
        f"You've been using gestures for about {minutes} minutes. Lower your arms, roll your shoulders and look "
        "away from the screen for a minute."
    )


def attention_badge(state: dict[str, Any] | None) -> dict[str, Any]:
    """``{"color", "label", "hint", "dim"}`` for an ``attention_state`` snapshot. ``dim``: gestures are being
    ignored right now (the at-cursor ring and the tray say so)."""
    st = state or {}
    s = str(st.get("state", "off")) if st.get("enabled") else "off"
    return {
        "color": ATTENTION_COLORS.get(s, ATTENTION_COLORS["off"]),
        "label": ATTENTION_LABELS.get(s, s),
        "hint": str(st.get("hint") or ""),
        "dim": s in ("idle", "arming"),
    }


def tray_suffix(state: dict[str, Any] | None) -> str:
    """Appended to the tray title while the gate is on: ``" - waiting for you (raise your hand...)"``."""
    if not (state or {}).get("enabled"):
        return ""
    b = attention_badge(state)
    text = f" - {b['label'].lower()}"
    return text + (f" ({b['hint']})" if b["dim"] and b["hint"] else "")


# ------------------------------------------------------------------------------------------- friendly errors
@dataclass(frozen=True)
class FriendlyError:
    title: str
    message: str
    fix_label: str  # the big button ("Try again", "Open camera settings"...)
    fix: str  # an action id the window knows how to run (see FIX_ACTIONS)


FIX_ACTIONS = ("retry", "camera_privacy", "download_models", "install_help", "pause_app")

_RULES: tuple[tuple[tuple[str, ...], FriendlyError], ...] = (
    (
        ("permission", "access denied", "not authorized", "privacy"),
        FriendlyError(
            "Your computer is blocking the camera",
            "Project Away needs permission to use the camera. Open the camera settings, allow desktop apps to use "
            "the camera, then come back and press Try again.",
            "Open camera settings",
            "camera_privacy",
        ),
    ),
    (
        ("busy", "in use", "did not hand", "being used"),
        FriendlyError(
            "Another app is using the camera",
            "Close video-call apps (Zoom, Teams, the Camera app), then try again. Project Away itself lends its "
            "camera to this guide automatically.",
            "Try again",
            "pause_app",
        ),
    ),
    (
        ("hand model missing", "landmarker.task", "model missing", "away setup"),
        FriendlyError(
            "A small download is missing",
            "The hand-tracking files have not been downloaded yet (about 10 MB).",
            "Download now",
            "download_models",
        ),
    ),
    (
        ("not installed", "no module named", "pip install"),
        FriendlyError(
            "A part of Project Away is not installed",
            "Camera support is missing from this installation. The installer from the Releases page includes "
            "everything; with pip, install project-away[all].",
            "Show me how",
            "install_help",
        ),
    ),
    (
        ("too slow",),
        FriendlyError(
            "The camera is very slow",
            "The camera sends too few pictures per second for smooth control. Try more light, another USB port "
            "or another camera (Settings > Cameras).",
            "Try again",
            "retry",
        ),
    ),
    (
        ("no camera", "did not open", "could not open", "not found", "no frames", "failed", "cannot open"),
        FriendlyError(
            "We can't see your camera",
            "Check that the camera is plugged in and its cover or privacy switch is open, then try again.",
            "Try again",
            "retry",
        ),
    ),
)

UNKNOWN_ERROR = FriendlyError(
    "Something went wrong",
    "That step did not work. You can try again, or skip it and carry on: everything works with the defaults.",
    "Try again",
    "retry",
)


def friendly_error(text: str | BaseException | None) -> FriendlyError:
    """A plain-language message and one fix for a raw error text (camera, models, packages...)."""
    raw = str(text or "").lower()
    for needles, err in _RULES:
        if any(n in raw for n in needles):
            return err
    return UNKNOWN_ERROR


def camera_privacy_command(platform: str | None = None) -> list[str] | None:
    """The command that opens the OS camera-privacy page (None where there is no such page to open)."""
    plat = platform or sys.platform
    if plat.startswith("win"):
        return ["cmd", "/c", "start", "", "ms-settings:privacy-webcam"]
    if plat == "darwin":
        return ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"]
    return None


INSTALL_HELP_URL = "https://github.com/fernandogarzaaa/project-away/blob/main/docs/INSTALL.md"
