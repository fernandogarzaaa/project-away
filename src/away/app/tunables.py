"""Settings > Tuning: what each slider means, how they group, presets, the master sensitivity, undo. Pure logic.

Nothing here imports tkinter, so all of it is unit-tested headless. :class:`TuningController` is the brain of the
Tuning tab: it works on a :class:`~away.app.settings_gui.SettingsModel` (anything with ``tunables()``,
``set_tunable()``, ``voice_value()``, ``bindings``, ``new_profile()``) and records every change for undo.

Data-driven grouping
--------------------
The tunables themselves are declared once, in :func:`away.config.tunable_ranges`. This module only *describes*
them: :data:`TUNABLE_META` gives a known path its section, plain-language title, help line and unit. A tunable
that is not described here still shows up in the right section: :func:`group_for` tries the explicit entry, then
:data:`GROUP_KEYWORDS` (``head``, ``gaze``, ``eye``, ``auto_center``... -> "Head & eyes") and then
:data:`GROUP_PREFIXES` (``hands.cursor.`` -> "Cursor"...), and anything else lands in "Other". So a feature
branch that adds ``("face.head.gain", ...)`` to ``tunable_ranges`` gets a slider under "Head & eyes" without
touching the GUI; it can add a :class:`TunableMeta` here for a nicer title and help text.

Master sensitivity and presets
------------------------------
:data:`SENSITIVITY_RULES` scale cursor gain, acceleration, the dead zone and the pinch thresholds around their
*shipped defaults*: ``s = 0`` is the defaults, ``s = +1`` faster / easier to trigger, ``s = -1`` slower / more
deliberate (log-interpolated, clamped to each slider's range, ``pinch_off`` follows ``pinch_on`` so the hysteresis
is kept). The presets Precise / Balanced / Fast are a sensitivity level plus a smoothing tweak.
"""

from __future__ import annotations

import copy
import functools
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from away.bindings import PROFILE_TUNING_PREFIXES
from away.pointer.config import POINTER_TUNABLE_TEXT

GROUPS = ("Cursor", "Clicks", "Scroll", "Gestures", "Face", "Head & eyes", "Voice", "Advanced", "Other")
DEFAULT_GROUP = "Other"


@dataclass(frozen=True)
class TunableMeta:
    group: str
    title: str = ""
    help: str = ""
    unit: str = ""


M = TunableMeta
# Raising / lowering is spelled out in every help line, so nobody has to guess which way a slider goes.
TUNABLE_META: dict[str, TunableMeta] = {
    # ------------------------------------------------------------------ cursor
    "hands.cursor.gain": M("Cursor", "Cursor speed", "Higher: the pointer travels further per hand movement.", "×"),
    "hands.cursor.dead_zone": M(
        "Cursor", "Hold-still zone", "Higher: small hand tremors are ignored, but tiny moves are lost.", "hand units"
    ),
    "hands.cursor.acceleration": M(
        "Cursor", "Acceleration", "Higher: fast flicks go much further than slow moves. 1 = no acceleration.", ""
    ),
    "hands.cursor.min_cutoff": M(
        "Cursor", "Smoothing (still)", "Lower: steadier pointer when you hold still, but it lags a little.", "Hz"
    ),
    "hands.cursor.beta": M(
        "Cursor", "Smoothing (moving)", "Higher: less lag on fast moves; lower: smoother but laggier.", ""
    ),
    "hands.cursor.d_cutoff": M(
        "Cursor", "Smoothing (speed estimate)", "Rarely needs changing. Higher: snappier.", "Hz"
    ),
    "hands.cursor.big_cursor_size": M(
        "Cursor", "Cursor size while gestures are on", "Higher: a bigger pointer you can see from across the room.", ""
    ),
    "hands.cursor.sigmoid_low": M("Cursor", "Curve: slow-move speed", "Higher: slow, precise moves cover more.", "×"),
    "hands.cursor.sigmoid_high": M("Cursor", "Curve: fast-move speed", "Higher: fast moves cover more screen.", "×"),
    "hands.cursor.sigmoid_mid": M(
        "Cursor", "Curve: switch-over speed", "Higher: you must move faster before the fast speed kicks in.", ""
    ),
    "hands.cursor.sigmoid_slope": M(
        "Cursor", "Curve: steepness", "Higher: a sharper switch between slow and fast speed.", ""
    ),
    "hands.cursor.speed_up": M("Cursor", "Speed upward", "Higher: the pointer moves faster upward.", "×"),
    "hands.cursor.speed_down": M("Cursor", "Speed downward", "Higher: the pointer moves faster downward.", "×"),
    "hands.cursor.speed_left": M("Cursor", "Speed to the left", "Higher: the pointer moves faster to the left.", "×"),
    "hands.cursor.speed_right": M(
        "Cursor", "Speed to the right", "Higher: the pointer moves faster to the right.", "×"
    ),
    "hands.cursor.monitor_reach": M(
        "Cursor", "Reach of other monitors", "Higher: less hand travel to reach the other screens.", ""
    ),
    "hands.roles.release_s": M(
        "Cursor",
        "Keep pointer hand while out of view",
        "Higher: your pointer hand keeps its role longer when it briefly leaves the camera.",
        "s",
    ),
    # ------------------------------------------------------------------ clicks
    "hands.gestures.pinch_on": M(
        "Clicks",
        "Pinch to click: how close",
        "Higher: a looser pinch already clicks (easier, but more accidental clicks).",
        "palm",
    ),
    "hands.gestures.pinch_off": M(
        "Clicks", "Pinch release distance", "Higher: fingers must open wider before the pinch lets go.", "palm"
    ),
    "hands.gestures.double_pinch_window": M(
        "Clicks", "Double-click time", "Higher: more time allowed between the two pinches of a double click.", "s"
    ),
    "hands.gestures.drag_hold": M(
        "Clicks", "Hold before drag", "Higher: hold the pinch longer before it becomes a drag.", "s"
    ),
    "hands.cursor.click_rewind_ms": M(
        "Clicks", "Steady click: rewind", "Higher: the click lands where the pointer was a bit earlier.", "ms"
    ),
    "hands.cursor.freeze_margin": M(
        "Clicks", "Steady click: freeze early", "Higher: the pointer freezes sooner as your fingers close.", ""
    ),
    "hands.dwell.radius_px": M(
        "Clicks", "Dwell click: hold-still radius", "Higher: the pointer may wander more while dwelling.", "px"
    ),
    "hands.dwell.time_s": M("Clicks", "Dwell click: time", "Higher: hold still longer before it clicks.", "s"),
    "hands.multicam.push_distance_m": M(
        "Clicks", "Push to click: distance", "Higher: push your hand further toward the screen to click.", "m"
    ),
    # ------------------------------------------------------------------ scroll
    "hands.gestures.scroll_gain": M("Scroll", "Two-finger scroll speed", "Higher: faster scrolling.", "×"),
    "hands.gestures.autoscroll_hold": M(
        "Scroll", "Auto-scroll: hold before it starts", "Higher: hold four fingers up longer first.", "s"
    ),
    "hands.gestures.autoscroll_thumb_fold": M(
        "Scroll",
        "Auto-scroll: thumb tucked enough",
        "Higher: a half-tucked thumb already counts (easier, but may catch an open palm).",
        "",
    ),
    "hands.gestures.autoscroll_dead_zone": M(
        "Scroll", "Auto-scroll: still zone", "Higher: move the hand further before scrolling starts.", "palm"
    ),
    "hands.gestures.autoscroll_gain": M("Scroll", "Auto-scroll: speed", "Higher: faster auto-scroll.", "×"),
    "hands.gestures.autoscroll_max_speed": M(
        "Scroll", "Auto-scroll: top speed", "Higher: the fastest auto-scroll allowed.", "notches/s"
    ),
    "hands.gestures.autoscroll_stop_hold": M(
        "Scroll", "Auto-scroll: fist to stop", "Higher: hold the fist longer to stop.", "s"
    ),
    "hands.palm_scroll.engage_s": M(
        "Scroll", "Palm scroll: touch before it scrolls", "Higher: touch the palm longer before scrolling.", "s"
    ),
    "hands.palm_scroll.depth_threshold": M(
        "Scroll",
        "Palm scroll: how close to the palm",
        "Higher: the fingertip may hover further off the palm and still count as touching (0 = any).",
        "palm",
    ),
    "hands.palm_scroll.margin": M(
        "Scroll", "Palm scroll: palm edge padding", "Higher: touches just outside the palm still count.", "palm"
    ),
    "hands.palm_scroll.gain": M("Scroll", "Palm scroll: speed", "Higher: more scrolling per glide.", "notches/palm"),
    "hands.palm_scroll.dead_zone": M(
        "Scroll", "Palm scroll: still zone", "Higher: glide further before it scrolls (and taps may move more).", "palm"
    ),
    "hands.palm_scroll.friction": M(
        "Scroll", "Palm scroll: momentum friction", "Higher: the coasting page stops sooner.", ""
    ),
    # ------------------------------------------------------------------ gestures
    "hands.gestures.swipe_min_dx": M(
        "Gestures", "Swipe distance", "Higher: swipe further across the camera view.", "frame"
    ),
    "hands.gestures.swipe_max_time": M(
        "Gestures", "Swipe time limit", "Lower: the swipe must be quicker to count.", "s"
    ),
    "hands.gestures.clutch_hold": M(
        "Gestures", "Open palm to pause: hold", "Higher: hold the open palm longer before it pauses.", "s"
    ),
    "app.auto_pause_idle_ms": M(
        "Gestures",
        "Auto-pause: resume after idle",
        "Higher: gestures wait longer after you stop using the real mouse.",
        "ms",
    ),
    # ------------------------------------------------------------------ face
    "face.mouth_threshold": M("Face", "Mouth open: how wide", "Higher: open your mouth wider to trigger.", ""),
    "face.mouth_hold": M("Face", "Mouth open: hold", "Higher: keep it open longer to trigger.", "s"),
    "face.brow_threshold": M("Face", "Brow raise: how high", "Higher: raise your eyebrows more to trigger.", ""),
    "face.wink_threshold": M("Face", "Wink: closed eye", "Higher: a half-closed eye already counts as a wink.", ""),
    "face.wink_open_max": M(
        "Face", "Wink: other eye open", "Higher: the other eye may be more closed and it still counts.", ""
    ),
    "face.wink_hold": M("Face", "Wink: hold", "Higher: hold the wink longer.", "s"),
    # ------------------------------------------------------------------ head & eyes (v0.1.19)
    "hands.cursor.autocenter.soft_strength": M(
        "Head & eyes",
        "Auto-center: soft recenter strength",
        "Higher: your resting hand drifts back to the middle of the camera view faster (0 = off).",
        "",
    ),
    "hands.cursor.autocenter.comfort_tau_s": M(
        "Head & eyes",
        "Auto-center: comfort zone learning time",
        "Higher: the learned resting spot follows your hand more slowly.",
        "s",
    ),
    "hands.cursor.autocenter.rest_speed": M(
        "Head & eyes",
        "Auto-center: resting hand speed",
        "Higher: a slowly moving hand still counts as resting.",
        "frame/s",
    ),
    "hands.cursor.autocenter.edge_margin": M(
        "Head & eyes",
        "Auto-center: camera edge band",
        "Higher: a wider band at the frame edge where a swing back does not drag the cursor.",
        "frame",
    ),
    "hands.cursor.autocenter.edge_return_speed": M(
        "Head & eyes",
        "Auto-center: swing-back speed at the edge",
        "Higher: only a quicker swing back from the edge lifts the cursor.",
        "frame/s",
    ),
    "hands.cursor.head.min_cutoff": M(
        "Head & eyes", "Head: smoothing (still)", "Lower: steadier head readings, but they lag a little.", "Hz"
    ),
    "hands.cursor.head.beta": M(
        "Head & eyes", "Head: smoothing (moving)", "Higher: less lag on quick head turns; lower: smoother.", ""
    ),
    "hands.cursor.head.pointer_gain": M(
        "Head & eyes", "Head pointer: speed", "Higher: the pointer moves further per degree of head turn.", "×"
    ),
    "hands.cursor.head.pointer_dead_zone": M(
        "Head & eyes", "Head pointer: hold-still zone", "Higher: small head tremors are ignored.", "°"
    ),
    "hands.cursor.head.pointer_acceleration": M(
        "Head & eyes",
        "Head pointer: acceleration",
        "Higher: quick head turns go much further than slow ones. 1 = none.",
        "",
    ),
    "hands.cursor.head.coarse_cols": M(
        "Head & eyes", "Head zones: columns per monitor", "Higher: more, smaller zones across each monitor.", ""
    ),
    "hands.cursor.head.coarse_rows": M(
        "Head & eyes", "Head zones: rows per monitor", "Higher: more, smaller zones down each monitor.", ""
    ),
    "hands.cursor.head.coarse_yaw_span": M(
        "Head & eyes",
        "Head zones: turn across the desktop",
        "Higher: turn your head further to cross the desktop (without a screen map).",
        "°",
    ),
    "hands.cursor.head.coarse_pitch_span": M(
        "Head & eyes",
        "Head zones: tilt across the desktop",
        "Higher: tilt your head further to go top to bottom (without a screen map).",
        "°",
    ),
    "hands.cursor.head.coarse_hysteresis": M(
        "Head & eyes",
        "Head zones: border stickiness",
        "Higher: look further past a border before the zone changes.",
        "",
    ),
    "hands.cursor.head.coarse_dwell_s": M(
        "Head & eyes", "Head zones: hold before a jump", "Higher: hold a new zone longer before the cursor jumps.", "s"
    ),
    "hands.cursor.head.switch_margin": M(
        "Head & eyes",
        "Monitor switch by head turn: margin",
        "Higher: turn further toward the other monitor before it switches.",
        "°",
    ),
    "hands.cursor.head.switch_dwell_s": M(
        "Head & eyes",
        "Monitor switch by head turn: hold",
        "Higher: face the other monitor longer before it switches (a glance does not).",
        "s",
    ),
    "hands.cursor.head.gate_tolerance": M(
        "Head & eyes",
        "Face gate: head turn still facing the screen",
        "Higher: you may turn your head further and still count as facing the screen.",
        "°",
    ),
    "gaze.model_rate_hz": M(
        "Head & eyes", "Gaze: learned model rate", "Higher: the optional gaze network runs more often (more CPU).", "Hz"
    ),
    "gaze.min_cutoff": M(
        "Head & eyes", "Gaze: smoothing (still)", "Lower: a steadier gaze point, but it lags a little.", "Hz"
    ),
    "gaze.beta": M(
        "Head & eyes", "Gaze: smoothing (moving)", "Higher: less lag when your eyes jump; lower: smoother.", ""
    ),
    "gaze.ridge_alpha": M(
        "Head & eyes",
        "Gaze: calibration regularisation",
        "Rarely needs changing. Higher: a smoother, less exact fit to the calibration dots.",
        "",
    ),
    "gaze.jump_idle_s": M(
        "Head & eyes",
        "Look-to-jump: hand rest before a jump",
        "Higher: rest your hand longer before the cursor may jump to where you look.",
        "s",
    ),
    "gaze.jump_min_px": M(
        "Head & eyes",
        "Look-to-jump: minimum distance",
        "Higher: the cursor only jumps when you look further away from it.",
        "px",
    ),
    "gaze.drift_rate": M(
        "Head & eyes",
        "Gaze drift correction: rate per click",
        "Higher: each click corrects the gaze point more (0 = off).",
        "",
    ),
    "gaze.drift_max_px": M(
        "Head & eyes",
        "Gaze drift correction: largest offset",
        "Higher: clicks may shift the gaze point further.",
        "px",
    ),
    "gaze.drift_max_err_px": M(
        "Head & eyes",
        "Gaze drift correction: ignore far clicks",
        "Higher: clicks further from where you look still correct the gaze point.",
        "px",
    ),
    # ------------------------------------------------------------------ voice
    "voice.vad_sensitivity": M("Voice", "Voice sensitivity", "Higher: quieter speech is picked up.", ""),
    "voice.end_silence_ms": M(
        "Voice", "Pause that ends a sentence", "Higher: you may pause longer mid-sentence.", "ms"
    ),
    "voice.min_utterance_ms": M("Voice", "Shortest utterance", "Higher: very short sounds are ignored.", "ms"),
    "voice.max_utterance_s": M("Voice", "Longest utterance", "Higher: longer dictation in one go.", "s"),
    "voice.wake_window": M(
        "Voice", "Wake word: follow-up time", "Higher: more time to speak after the wake word.", "s"
    ),
    "voice.listen_seconds": M(
        "Voice",
        "Keep listening after the wake word",
        "Higher: more time to start talking to your agent (and, in conversational mode, to answer it).",
        "s",
    ),
    "voice.agent_end_silence_ms": M(
        "Voice",
        "Pause that ends what you say to your agent",
        "Lower: the agent answers sooner; higher: you may pause longer mid-sentence.",
        "ms",
    ),
    "voice.numbers_min_value": M("Voice", "Write numbers as digits from", "Higher: more numbers stay spelled out.", ""),
    "voice.clap_sensitivity": M("Voice", "Clap sensitivity", "Higher: softer claps count.", ""),
    "voice.clap_double_window": M("Voice", "Double-clap time", "Higher: more time allowed between the two claps.", "s"),
    # ------------------------------------------------------------------ other
    "hands.min_camera_fps": M(
        "Other", "Camera: lowest usable frame rate", "Higher: slow cameras are flagged sooner.", "fps"
    ),
    "hands.multicam.pair_tolerance_ms": M(
        "Other", "Multi-camera pairing tolerance", "Higher: frames further apart in time are paired.", "ms"
    ),
}
# v0.1.20 smart pointer (away.pointer.config): the numbers behind the three plain toggles, kept out of the way
TUNABLE_META.update({p: M("Advanced", *text) for p, text in POINTER_TUNABLE_TEXT.items()})
del M

# A path segment containing one of these -> "Head & eyes" (head pose, gaze, auto-centre features).
GROUP_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("head", "Head & eyes"),
    ("gaze", "Head & eyes"),
    ("eye", "Head & eyes"),
    ("auto_center", "Head & eyes"),
    ("autocenter", "Head & eyes"),
    ("auto_centre", "Head & eyes"),
    ("recenter", "Head & eyes"),
)
# Longest matching prefix wins.
GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("hands.cursor.", "Cursor"),
    ("hands.roles.", "Cursor"),
    ("hands.dwell.", "Clicks"),
    ("hands.palm_scroll.", "Scroll"),
    ("hands.gestures.autoscroll", "Scroll"),
    ("hands.gestures.scroll", "Scroll"),
    ("hands.gestures.pinch", "Clicks"),
    ("hands.gestures.", "Gestures"),
    ("face.", "Face"),
    ("voice.", "Voice"),
)
UNIT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_ms", "ms"),
    ("_px", "px"),
    ("_fps", "fps"),
    ("_deg", "°"),
    ("_m", "m"),
    ("_s", "s"),
    ("_hold", "s"),
    ("_window", "s"),
)


def group_for(path: str, declared: str | None = None) -> str:
    """The Tuning section of ``path``: a known meta entry, else the ``declared`` group (when it is one of
    :data:`GROUPS`), else a keyword, else the longest prefix rule, else :data:`DEFAULT_GROUP`."""
    meta = TUNABLE_META.get(path)
    if meta is not None and meta.group in GROUPS:
        return meta.group
    if declared in GROUPS:
        return str(declared)
    parts = path.lower().split(".")
    for word, group in GROUP_KEYWORDS:
        if any(word in part for part in parts):
            return group
    best = ""
    found = DEFAULT_GROUP
    for prefix, group in GROUP_PREFIXES:
        if path.startswith(prefix) and len(prefix) > len(best):
            best, found = prefix, group
    return found


def humanize(path: str) -> str:
    """``face.head.dead_zone`` -> ``Head dead zone`` (a fallback title for undescribed tunables)."""
    parts = [p for p in path.split(".") if p]
    if not parts:
        return path
    tail = parts[-2:] if len(parts) >= 3 else parts[-1:]
    words = " ".join(tail).replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def unit_for(path: str) -> str:
    meta = TUNABLE_META.get(path)
    if meta is not None:
        return meta.unit
    last = path.rsplit(".", 1)[-1]
    for suffix, unit in UNIT_SUFFIXES:
        if last.endswith(suffix):
            return unit
    return ""


def describe(path: str, label: str | None = None, declared_group: str | None = None) -> TunableMeta:
    """Section, title, help and unit for any tunable path (known or not)."""
    meta = TUNABLE_META.get(path)
    title = (meta.title if meta and meta.title else "") or (label if label and label != path else "") or humanize(path)
    help_text = meta.help if meta else ""
    if not help_text:
        help_text = f"Advanced setting ({path})."
    return TunableMeta(group_for(path, declared_group), title, help_text, unit_for(path))


def format_value(value: float, unit: str = "", is_int: bool = False, step: float | None = None) -> str:
    """``0.32 palm``, ``800 ms``, ``2.2×``: the number at the slider's precision plus its unit."""
    if is_int:
        text = str(round(value))
    else:
        decimals = 3
        if step is not None and step > 0:
            decimals = max(0, min(4, -math.floor(math.log10(step))))
        text = f"{value:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text in ("-0", ""):
            text = "0"
    if not unit:
        return text
    if unit in ("×", "°"):
        return f"{text}{unit}"
    return f"{text} {unit}"


def group_items(items: Iterable[Any], key: Callable[[Any], str]) -> list[tuple[str, list[Any]]]:
    """Items bucketed by ``key(item)`` in :data:`GROUPS` order (unknown group names -> "Other"); empty groups are
    left out and each group keeps the items' original order."""
    buckets: dict[str, list[Any]] = {g: [] for g in GROUPS}
    for it in items:
        g = key(it)
        buckets[g if g in buckets else DEFAULT_GROUP].append(it)
    return [(g, buckets[g]) for g in GROUPS if buckets[g]]


# ------------------------------------------------------------------------------------- defaults
def default_values() -> dict[str, float]:
    """The shipped default of every tunable (a fresh :class:`~away.config.AppConfig`)."""
    from away.config import AppConfig, tunable_ranges

    return {p: float(v) for p, v, _lo, _hi in tunable_ranges(AppConfig())}


@functools.lru_cache(maxsize=1)
def _defaults_cache() -> tuple[tuple[str, float], ...]:
    return tuple(default_values().items())


def cached_defaults() -> dict[str, float]:
    """:func:`default_values`, computed once per process (the shipped defaults never change while it runs)."""
    return dict(_defaults_cache())


def short_label(text: str, n: int = 24) -> str:
    """``text`` cut to ``n`` characters with an ellipsis (the Undo button names the step it reverts)."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: max(1, n - 1)].rstrip() + "…"


def tunable_bounds() -> dict[str, tuple[float, float]]:
    from away.config import AppConfig, tunable_ranges

    return {p: (float(lo), float(hi)) for p, _v, lo, hi in tunable_ranges(AppConfig())}


# --------------------------------------------------------------------------- master sensitivity
SENSITIVITY_MIN, SENSITIVITY_MAX = -1.0, 1.0


@dataclass(frozen=True)
class SensitivityRule:
    """At sensitivity ``+1`` the value is ``anchor + (default - anchor) * up``, at ``-1`` ``... * down``;
    log-interpolated in between (``s = 0`` gives the default)."""

    path: str
    up: float
    down: float
    anchor: float = 0.0


SENSITIVITY_RULES: tuple[SensitivityRule, ...] = (
    SensitivityRule("hands.cursor.gain", 2.0, 0.5),
    SensitivityRule("hands.cursor.acceleration", 1.5, 0.5, anchor=1.0),
    SensitivityRule("hands.cursor.dead_zone", 0.5, 2.0),
    SensitivityRule("hands.gestures.pinch_on", 1.25, 0.8),
    SensitivityRule("hands.gestures.pinch_off", 1.25, 0.8),
)
SENSITIVITY_PATHS = tuple(r.path for r in SENSITIVITY_RULES)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rule_factor(rule: SensitivityRule, s: float) -> float:
    s = clamp(float(s), SENSITIVITY_MIN, SENSITIVITY_MAX)
    if s >= 0:
        return float(rule.up**s)
    return float(rule.down ** (-s))


def sensitivity_values(
    s: float, defaults: Mapping[str, float], bounds: Mapping[str, tuple[float, float]] | None = None
) -> dict[str, float]:
    """Values of the :data:`SENSITIVITY_RULES` tunables for master sensitivity ``s`` in [-1, 1]."""
    out: dict[str, float] = {}
    for rule in SENSITIVITY_RULES:
        if rule.path not in defaults:
            continue
        d = float(defaults[rule.path])
        v = rule.anchor + (d - rule.anchor) * rule_factor(rule, s)
        if bounds and rule.path in bounds:
            v = clamp(v, *bounds[rule.path])
        out[rule.path] = v
    return out


def estimate_sensitivity(values: Mapping[str, float], defaults: Mapping[str, float]) -> float:
    """Where the master slider sits for ``values``: inverted from the cursor gain (the first rule), 0 when the
    gain or its default is missing or not positive."""
    rule = SENSITIVITY_RULES[0]
    v, d = values.get(rule.path), defaults.get(rule.path)
    if v is None or d is None or v <= 0 or d <= 0:
        return 0.0
    ratio = math.log(float(v) / float(d))
    s = ratio / math.log(rule.up) if ratio >= 0 else -ratio / math.log(rule.down)
    return clamp(s, SENSITIVITY_MIN, SENSITIVITY_MAX)


def slider_to_sensitivity(pos: float) -> float:
    """The master slider (0..100, 50 = defaults) -> sensitivity in [-1, 1]."""
    return clamp((float(pos) - 50.0) / 50.0, SENSITIVITY_MIN, SENSITIVITY_MAX)


def sensitivity_to_slider(s: float) -> float:
    return clamp(float(s), SENSITIVITY_MIN, SENSITIVITY_MAX) * 50.0 + 50.0


@dataclass(frozen=True)
class Preset:
    name: str
    sensitivity: float
    extra: tuple[tuple[str, float], ...] = ()  # (path, factor of the default)
    blurb: str = ""


PRESETS: dict[str, Preset] = {
    "Precise": Preset(
        "Precise",
        -0.5,
        (("hands.cursor.min_cutoff", 0.6), ("hands.cursor.beta", 0.7)),
        "slower, steadier pointer and a tighter pinch",
    ),
    "Balanced": Preset("Balanced", 0.0, (), "the shipped defaults"),
    "Fast": Preset(
        "Fast",
        0.5,
        (("hands.cursor.min_cutoff", 1.6), ("hands.cursor.beta", 1.5)),
        "quicker pointer, less smoothing, an easier pinch",
    ),
}
PRESET_SMOOTHING_PATHS = ("hands.cursor.min_cutoff", "hands.cursor.beta")


def preset_values(
    name: str, defaults: Mapping[str, float], bounds: Mapping[str, tuple[float, float]] | None = None
) -> dict[str, float]:
    """Every value a preset sets (the sensitivity tunables plus the smoothing ones, which go back to their
    defaults for Balanced)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; use {', '.join(PRESETS)}")
    p = PRESETS[name]
    out = sensitivity_values(p.sensitivity, defaults, bounds)
    factors = dict(p.extra)
    for path in PRESET_SMOOTHING_PATHS:
        if path not in defaults:
            continue
        v = float(defaults[path]) * factors.get(path, 1.0)
        if bounds and path in bounds:
            v = clamp(v, *bounds[path])
        out[path] = v
    return out


def matching_preset(values: Mapping[str, float], defaults: Mapping[str, float], tol: float = 1e-6) -> str | None:
    """The preset the current values equal, or None (custom)."""
    bounds = tunable_bounds()
    for name in PRESETS:
        want = preset_values(name, defaults, bounds)
        if all(path in values and abs(float(values[path]) - v) <= tol * max(1.0, abs(v)) for path, v in want.items()):
            return name
    return None


# ------------------------------------------------------------------------------------- undo
@dataclass(frozen=True)
class Change:
    path: str
    old: float
    new: float


@dataclass
class UndoEntry:
    label: str
    changes: list[Change]
    t: float = 0.0


@dataclass
class UndoStack:
    """Batches of changes, newest last. Repeated changes of one slider within ``coalesce_s`` (a drag fires a
    callback per pixel) merge into one entry that keeps the values from before the drag."""

    max_depth: int = 100
    coalesce_s: float = 1.0
    entries: list[UndoEntry] = field(default_factory=list)

    def push(self, label: str, changes: Iterable[Change], t: float = 0.0, coalesce: bool = False) -> bool:
        """Record a batch; no-op changes are dropped. With ``coalesce``, a batch with the same label and the same
        settings as the newest entry, within ``coalesce_s`` of it, is merged into it (a slider drag, the master
        sensitivity drag). Returns False when nothing was recorded."""
        real = [c for c in changes if c.old != c.new]
        if not real:
            return False
        last = self.entries[-1] if self.entries else None
        if (
            coalesce
            and last is not None
            and last.label == label
            and [c.path for c in last.changes] == [c.path for c in real]
            and 0.0 <= t - last.t <= self.coalesce_s
        ):
            merged = [Change(a.path, a.old, b.new) for a, b in zip(last.changes, real)]
            if all(c.old == c.new for c in merged):  # dragged back to where it started: nothing to undo
                self.entries.pop()
            else:
                last.changes = merged
                last.t = t
            return True
        self.entries.append(UndoEntry(label, real, t))
        if len(self.entries) > self.max_depth:
            del self.entries[: len(self.entries) - self.max_depth]
        return True

    def pop(self) -> UndoEntry | None:
        return self.entries.pop() if self.entries else None

    def peek(self) -> UndoEntry | None:
        return self.entries[-1] if self.entries else None

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def can_undo(self) -> bool:
        return bool(self.entries)


# --------------------------------------------------------------------------- per-app profiles
def profile_tuning_snapshot(
    values: Mapping[str, float], defaults: Mapping[str, float], tol: float = 1e-9
) -> dict[str, float]:
    """The tunables a per-app profile keeps: those that differ from the shipped default and that the app applies
    per focused app (:data:`~away.bindings.PROFILE_TUNING_PREFIXES`)."""
    out: dict[str, float] = {}
    for path, v in values.items():
        if not path.startswith(PROFILE_TUNING_PREFIXES):
            continue
        d = defaults.get(path)
        if d is None or abs(float(v) - float(d)) > tol:
            out[path] = float(v)
    return dict(sorted(out.items()))


def _set_path(cfg: Any, path: str, value: Any) -> None:
    *head, last = path.split(".")
    obj = cfg
    for part in head:
        obj = getattr(obj, part)
    old = getattr(obj, last)
    if isinstance(old, bool):
        return
    setattr(obj, last, round(value) if isinstance(old, int) else float(value))


def profile_tuned(cfg: Any, profile: str) -> Any:
    """A copy of ``cfg`` with the tuning of bindings profile ``profile`` applied (``cfg`` itself is untouched).
    Unknown paths are skipped (validation rejects them when the config loads)."""
    out = copy.deepcopy(cfg)
    prof = cfg.bindings.profiles.get(profile)
    for path, value in (getattr(prof, "tuning", None) or {}).items():
        try:
            _set_path(out, path, value)
        except AttributeError:
            continue
    g, base = out.hands.gestures, cfg.hands.gestures
    if not g.pinch_on < g.pinch_off:  # the profile and the global values together broke the pinch hysteresis
        g.pinch_on, g.pinch_off = base.pinch_on, base.pinch_off
    return out


# --------------------------------------------------------------------------- the controller
class TuningController:
    """Everything the Tuning tab does, minus the widgets. Every change goes through :meth:`apply` and is undoable."""

    def __init__(self, model: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self.m = model
        self.clock = clock
        self.undo_stack = UndoStack()
        self._defaults: dict[str, float] | None = None
        self._bounds: dict[str, tuple[float, float]] | None = None

    # ---------------------------------------------------------------- reading
    @property
    def defaults(self) -> dict[str, float]:
        if self._defaults is None:
            self._defaults = cached_defaults()
        return self._defaults

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        if self._bounds is None:
            self._bounds = tunable_bounds()
        return self._bounds

    def tunables(self) -> list[Any]:
        return list(self.m.tunables())

    def groups(self) -> list[tuple[str, list[Any]]]:
        return group_items(self.tunables(), key=lambda t: getattr(t, "group", DEFAULT_GROUP))

    def values(self) -> dict[str, float]:
        return {t.path: float(t.value) for t in self.tunables()}

    def value(self, path: str) -> float:
        return float(self.m.voice_value(path))

    def default(self, path: str) -> float | None:
        return self.defaults.get(path)

    def is_default(self, path: str) -> bool:
        d = self.default(path)
        return d is not None and abs(self.value(path) - d) <= 1e-9 * max(1.0, abs(d))

    def sensitivity(self) -> float:
        return estimate_sensitivity(self.values(), self.defaults)

    def sensitivity_matches(self, tol: float = 1e-6) -> bool:
        """The master-sensitivity tunables all sit where the master slider says (False once one of them was moved
        on its own)."""
        cur = self.values()
        want = sensitivity_values(self.sensitivity(), self.defaults, self.bounds)
        return all(p in cur and abs(cur[p] - v) <= tol * max(1.0, abs(v)) for p, v in want.items())

    def preset(self) -> str | None:
        return matching_preset(self.values(), self.defaults)

    # ---------------------------------------------------------------- writing
    def set(self, path: str, value: float) -> str | None:
        """One slider moved (coalesced with the previous move of the same slider for undo)."""
        return self.apply({path: value}, label=describe(path).title, coalesce=True)

    def apply(self, values: Mapping[str, float], label: str = "change", coalesce: bool = False) -> str | None:
        """Set several tunables as one undoable step. Order-independent: a value that fails only because of
        another pending one (``pinch_on < pinch_off``) is retried after the others; when something still fails,
        every value set in this call is rolled back and the error returned."""
        before = {p: self.value(p) for p in values if self._known(p)}
        unknown = [p for p in values if p not in before]
        if unknown:
            return f"unknown setting {unknown[0]!r}"
        pending = list(values.items())
        done: list[str] = []
        err: str | None = None
        while pending:
            failed: list[tuple[str, float]] = []
            for path, v in pending:
                e = self.m.set_tunable(path, float(v))
                if e is None:
                    done.append(path)
                else:
                    failed.append((path, v))
                    err = e
            stuck = len(failed) == len(pending)
            pending = failed
            if stuck:
                break
        if pending:
            for path in reversed(done):
                self.m.set_tunable(path, before[path])
            return err
        changes = [Change(p, before[p], self.value(p)) for p in values]
        self.undo_stack.push(label, changes, self.clock(), coalesce)
        return None

    def _known(self, path: str) -> bool:
        return any(t.path == path for t in self.tunables())

    def reset(self, path: str) -> str | None:
        d = self.default(path)
        if d is None:
            return f"{path} has no default"
        return self.apply({path: d}, label=f"reset {describe(path).title}")

    def reset_group(self, group: str) -> str | None:
        vals = {
            t.path: self.defaults[t.path]
            for g, ts in self.groups()
            if g == group
            for t in ts
            if t.path in self.defaults
        }
        if not vals:
            return f"nothing to reset in {group}"
        return self.apply(vals, label=f"reset {group}")

    def apply_preset(self, name: str) -> str | None:
        try:
            vals = preset_values(name, self.defaults, self.bounds)
        except ValueError as exc:
            return str(exc)
        return self.apply(vals, label=f"preset {name}")

    def set_sensitivity(self, s: float) -> str | None:
        return self.apply(sensitivity_values(s, self.defaults, self.bounds), label="sensitivity", coalesce=True)

    def undo(self) -> tuple[str | None, str | None]:
        """``(error, done_message)``: revert the newest step."""
        entry = self.undo_stack.pop()
        if entry is None:
            return "nothing to undo", None
        vals = {c.path: c.old for c in entry.changes}
        before = {p: self.value(p) for p in vals}
        pending = list(vals.items())
        err: str | None = None
        while pending:
            failed = [(p, v) for p, v in pending if self.m.set_tunable(p, v) is not None]
            if len(failed) == len(pending):
                err = self.m.set_tunable(*failed[0])
                break
            pending = failed
        if err:
            for p, v in before.items():
                self.m.set_tunable(p, v)
            self.undo_stack.entries.append(entry)
            return err, None
        return None, f"undid {entry.label}"

    # ---------------------------------------------------------------- profiles
    def save_as_profile(self, name: str, apps_text: str = "") -> tuple[str | None, str | None]:
        """Keep the current hand / face tuning (what differs from the defaults) in per-app profile ``name``
        (created with ``apps_text`` when it does not exist). ``(error, done_message)``."""
        name = (name or "").strip()
        if not name:
            return "type a profile name", None
        if name == "default":
            return "the default profile is your global tuning; pick or create an app profile", None
        snap = profile_tuning_snapshot(self.values(), self.defaults)
        b = self.m.bindings
        if name not in b.profiles:
            err = self.m.new_profile(name, apps_text)
            if err:
                return err, None
        elif apps_text.strip():
            self.m.select_profile(name)
            err = self.m.set_profile_apps(apps_text)
            if err:
                return err, None
        b.profiles[name].tuning = dict(snap)
        apps = ", ".join(b.profiles[name].apps) or "no apps yet (set them at the top)"
        n = len(snap)
        what = f"{n} tuned setting{'s' if n != 1 else ''}" if n else "no differences from the defaults"
        return None, f"profile {name!r}: {what}; used in {apps}. Save to keep it."
