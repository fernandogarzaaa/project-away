"""Permissions an agent can hold, in plain language for the UI and as short scope names for the code.

The UI never says "scope": it shows :data:`PERMISSIONS` labels with an icon and a toggle. ``read`` is always
granted (an agent that cannot see anything is not connected). Everything that shows the user's camera or screen
or touches their computer is off by default and needs its own toggle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

READ = "read"  # status, gestures, landmarks, health, logs, config (redacted), context
DIAGNOSE = "diagnose"  # camera checks, benchmarks, recordings, diagnostics bundles
TUNE = "tune"  # change settings, zones, bindings, profiles, presets, preferences (undoable)
SNAPSHOTS = "snapshots"  # one camera still at a time, each one flashes an on-screen indicator
SCREEN = "screen"  # one screenshot or a low-fps screen view, with the same indicator
CONTROL = "control"  # use the mouse/keyboard; approved on the PC every session, expires, takeover pauses it

ALL = (READ, DIAGNOSE, TUNE, SNAPSHOTS, SCREEN, CONTROL)
DEFAULT = frozenset({READ, DIAGNOSE})


@dataclass(frozen=True)
class Permission:
    scope: str
    label: str  # what the user sees
    icon: str  # a symbol Tk can draw everywhere (no emoji fonts needed)
    default: bool
    locked: bool = False  # always on, the toggle is greyed out
    advanced: bool = False  # shown only under "Advanced"
    help: str = ""


PERMISSIONS: tuple[Permission, ...] = (
    Permission(READ, "See my gestures", "\u261d", True, locked=True, help="What your hands, face and voice do."),
    Permission(SNAPSHOTS, "See my camera when I allow", "\u25c9", False, help="One still photo at a time."),
    Permission(SCREEN, "See my screen when I allow", "\u25a3", False, help="Screenshots, never sound."),
    Permission(TUNE, "Change my settings", "\u2699", False, help="Every change can be undone."),
    Permission(CONTROL, "Let it use my computer", "\u2328", False, help="You approve it each time. Move to stop."),
    Permission(DIAGNOSE, "Run camera and speed checks", "\u2695", True, advanced=True),
)
BY_SCOPE = {p.scope: p for p in PERMISSIONS}


def normalize(scopes: Iterable[str] | str | None) -> frozenset[str]:
    """Scope names from a list or a comma-separated string; ``read`` is always added. Unknown names raise."""
    if scopes is None:
        return DEFAULT
    items = [s.strip().lower() for s in (scopes.split(",") if isinstance(scopes, str) else scopes)]
    items = [s for s in items if s]
    bad = [s for s in items if s not in ALL]
    if bad:
        raise ValueError(f"unknown permission(s) {', '.join(bad)}; use {', '.join(ALL)}")
    return frozenset({READ, *items})


def describe(scopes: Iterable[str]) -> list[str]:
    """Plain-language lines for an approval prompt, in UI order."""
    have = set(scopes)
    return [p.label for p in PERMISSIONS if p.scope in have]


def to_list(scopes: Iterable[str]) -> list[str]:
    have = set(scopes)
    return [s for s in ALL if s in have]
