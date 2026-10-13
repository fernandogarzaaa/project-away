"""The ``attention:`` and ``comfort:`` config sections (v0.1.20).

They live in their own module so the feature touches :mod:`away.config` in only a few lines (two fields, one
validation call, one tunables splat). Plain switches (what a non-technical user sees as a check box) and the
numbers (Settings > Tuning, or the Advanced part of the Getting started tab) are kept apart on purpose:
everything works with the defaults, and the numbers are only for people who want them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AttentionConfig:
    """"Only respond when I'm looking at the screen": gestures (and optionally voice commands) count only while
    the user is engaged: facing the screen, looking at it (with a gaze calibration), hand raised into the working
    area, held for a moment. See :class:`away.comfort.attention.AttentionGate`."""

    enabled: bool = False  # the one switch (off by default: nothing changes for existing users)
    require_facing: bool = True  # head turned toward the screens (head pose from the face landmarker)
    require_gaze: bool = False  # eyes on a screen; only used when a gaze calibration exists (`away gaze calibrate`)
    min_hand_height: float = 0.25  # wrist at least this high above the bottom of the camera picture (0..1)
    min_hand_size: float = 0.03  # palm at least this big in the picture (a far-away hand behind you is smaller)
    onset_s: float = 0.25  # everything must hold this long before gestures count
    grace_s: float = 0.8  # a short look away / dip of the hand keeps gestures on for this long
    facing_tolerance: float = 25.0  # degrees of head turn from the usual screen-facing pose that still count
    gate_voice: bool = False  # also ignore voice commands while not facing the screen (dictation is the face gate's)
    indicator: bool = True  # tint the at-cursor ring / tray title while not engaged


@dataclass
class ComfortConfig:
    """Arm-fatigue mode and break reminders. See :class:`away.comfort.fatigue.FatigueMonitor`."""

    fatigue_switch: bool = False  # "Switch to head control when my arm is tired" (head_assist -> pointer)
    fatigue_tips: bool = True  # "Tell me when my arm looks tired" (a gentle notification)
    break_reminders: bool = True  # "Remind me to take a break"
    raised_height: float = 0.30  # wrist this high (0..1 from the picture bottom) = arm raised
    raised_limit_s: float = 240.0  # arm raised this long without a rest = tired, even without other signs
    min_raised_s: float = 30.0  # never call it fatigue before the arm has been up this long
    height_drop: float = 0.08  # the hand sinking this much (picture heights) below where it started = tired sign
    jitter_ratio: float = 1.8  # hand tremor this many times the early-session tremor = tired sign
    rest_s: float = 5.0  # hand down / out of view this long = the arm rested (fatigue clears)
    return_s: float = 1.0  # after a rest, the hand back up this long = switch back to hand control
    tip_cooldown_min: float = 10.0  # at most one "arm looks tired" tip per this many minutes
    break_every_min: float = 30.0  # a break reminder after this many minutes of continuous use
    break_reset_s: float = 120.0  # no hand in view this long = you took a break (the clock restarts)


ATTENTION_SWITCHES = ("enabled", "require_facing", "require_gaze", "gate_voice", "indicator")
COMFORT_SWITCHES = ("fatigue_switch", "fatigue_tips", "break_reminders")

# The plain toggles (Getting started tab, tray, first run). Wording requested by the user for the first two.
PLAIN_SWITCHES: tuple[tuple[str, str, str], ...] = (
    (
        "attention.enabled",
        "Only respond when I'm looking at the screen",
        "Gestures count only while you face the screen with your hand raised. Stops accidental clicks when you "
        "turn away, talk to someone or rest your hand.",
    ),
    (
        "comfort.fatigue_switch",
        "Switch to head control when my arm is tired",
        "After a long time with your arm up, lower your hand and steer the pointer with your head. Raise your "
        "hand again to take over.",
    ),
    (
        "comfort.fatigue_tips",
        "Tell me when my arm looks tired",
        "A gentle note when your hand starts to sink or shake.",
    ),
    (
        "comfort.break_reminders",
        "Remind me to take a break",
        "A short reminder after a long stretch of gesture use (every 30 minutes by default).",
    ),
)
# Advanced switches (shown under Advanced, not as the main toggles)
ADVANCED_SWITCHES: tuple[tuple[str, str], ...] = (
    ("attention.require_facing", "Looking at the screen: check where my head points"),
    ("attention.require_gaze", "Looking at the screen: also check my eyes (needs eye calibration)"),
    ("attention.gate_voice", "Also ignore voice commands while I look away"),
    ("attention.indicator", "Show when gestures are waiting for me (dim ring)"),
)


def validate_comfort(cfg: Any) -> None:
    """Switches must be true/false (numbers are checked through :func:`comfort_tunables`)."""
    for section, names in (("attention", ATTENTION_SWITCHES), ("comfort", COMFORT_SWITCHES)):
        obj = getattr(cfg, section)
        for name in names:
            if not isinstance(getattr(obj, name), bool):
                raise ValueError(f"{section}.{name} must be true or false, got {getattr(obj, name)!r}")
    c = cfg.comfort
    lo, hi = c.min_raised_s, c.raised_limit_s
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
        raise ValueError("comfort.min_raised_s must be <= comfort.raised_limit_s")


def comfort_tunables(cfg: Any) -> list[tuple[str, Any, float, float]]:
    """(yaml path, value, min, max) for :func:`away.config.tunable_ranges` (the Tuning tab sliders)."""
    a, c = cfg.attention, cfg.comfort
    return [
        ("attention.min_hand_height", a.min_hand_height, 0.0, 0.9),
        ("attention.min_hand_size", a.min_hand_size, 0.0, 0.3),
        ("attention.onset_s", a.onset_s, 0.0, 2.0),
        ("attention.grace_s", a.grace_s, 0.0, 5.0),
        ("attention.facing_tolerance", a.facing_tolerance, 5.0, 90.0),
        ("comfort.raised_height", c.raised_height, 0.0, 0.9),
        ("comfort.raised_limit_s", c.raised_limit_s, 30.0, 1800.0),
        ("comfort.min_raised_s", c.min_raised_s, 5.0, 600.0),
        ("comfort.height_drop", c.height_drop, 0.01, 0.5),
        ("comfort.jitter_ratio", c.jitter_ratio, 1.1, 5.0),
        ("comfort.rest_s", c.rest_s, 1.0, 60.0),
        ("comfort.return_s", c.return_s, 0.2, 10.0),
        ("comfort.tip_cooldown_min", c.tip_cooldown_min, 1.0, 120.0),
        ("comfort.break_every_min", c.break_every_min, 5.0, 240.0),
        ("comfort.break_reset_s", c.break_reset_s, 10.0, 1800.0),
    ]


# (group, title, help, unit) for the Tuning tab; registered into away.app.tunables.TUNABLE_META by register_meta()
COMFORT_META: dict[str, tuple[str, str, str, str]] = {
    "attention.min_hand_height": (
        "Face",
        "Looking at the screen: how high to raise your hand",
        "Higher: raise your hand further before gestures count (fewer accidental gestures).",
        "",
    ),
    "attention.min_hand_size": (
        "Face",
        "Looking at the screen: how close your hand must be",
        "Higher: a hand must be closer to the camera to count (ignores people behind you).",
        "",
    ),
    "attention.onset_s": (
        "Face",
        "Looking at the screen: wait before responding",
        "Higher: gestures start a little later after you look back (fewer accidental starts).",
        "s",
    ),
    "attention.grace_s": (
        "Face",
        "Looking at the screen: keep going during a glance away",
        "Higher: a longer glance away or dip of the hand still keeps gestures on.",
        "s",
    ),
    "attention.facing_tolerance": (
        "Head & eyes",
        "Looking at the screen: head turn still counted",
        "Higher: you may turn your head further and still count as looking at the screen.",
        "°",
    ),
    "comfort.raised_height": (
        "Head & eyes",
        "Arm-fatigue: hand height that counts as arm raised",
        "Higher: only a hand held high counts toward arm fatigue.",
        "",
    ),
    "comfort.raised_limit_s": (
        "Head & eyes",
        "Arm-fatigue: longest time with the arm up",
        "Higher: more time with your arm up before it counts as tired.",
        "s",
    ),
    "comfort.min_raised_s": (
        "Head & eyes",
        "Arm-fatigue: earliest it may say tired",
        "Higher: never says your arm is tired before this much time.",
        "s",
    ),
    "comfort.height_drop": (
        "Head & eyes",
        "Arm-fatigue: hand sinking that counts",
        "Higher: your hand must sink further before it counts as a tired sign.",
        "",
    ),
    "comfort.jitter_ratio": (
        "Head & eyes",
        "Arm-fatigue: hand shake that counts",
        "Higher: your hand must shake more before it counts as a tired sign.",
        "×",
    ),
    "comfort.rest_s": (
        "Head & eyes",
        "Arm-fatigue: rest that clears it",
        "Higher: rest your arm longer before it counts as rested.",
        "s",
    ),
    "comfort.return_s": (
        "Head & eyes",
        "Arm-fatigue: hand back up before switching back",
        "Higher: hold your hand up longer before control goes back to the hand.",
        "s",
    ),
    "comfort.tip_cooldown_min": (
        "Other",
        "Arm-fatigue tips: at most one every",
        "Higher: fewer 'your arm looks tired' notes.",
        "min",
    ),
    "comfort.break_every_min": (
        "Other",
        "Break reminder: every",
        "Higher: break reminders come less often.",
        "min",
    ),
    "comfort.break_reset_s": (
        "Other",
        "Break reminder: time away that counts as a break",
        "Higher: you must be away longer before the break clock restarts.",
        "s",
    ),
}


def register_meta() -> None:
    """Describe the sliders in the Tuning tab (idempotent; called when this module is imported)."""
    from away.app import tunables

    for path, (group, title, help_text, unit) in COMFORT_META.items():
        tunables.TUNABLE_META.setdefault(path, tunables.TunableMeta(group, title, help_text, unit))


def get_path(cfg: Any, path: str) -> Any:
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_switch(cfg: Any, path: str, value: bool) -> str | None:
    """Flip one of the plain/advanced switches on a config (validated). Returns an error text or None."""
    known = {p for p, *_ in PLAIN_SWITCHES} | {p for p, _ in ADVANCED_SWITCHES}
    if path not in known:
        return f"unknown switch {path!r}"
    section, name = path.split(".", 1)
    obj = getattr(cfg, section)
    old = getattr(obj, name)
    setattr(obj, name, bool(value))
    try:
        validate_comfort(cfg)
    except ValueError as exc:
        setattr(obj, name, old)
        return str(exc)
    return None


register_meta()
