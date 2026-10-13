"""Settings editor: bindings table, tunables, voice mode, profiles, live gesture test.

:class:`SettingsModel` holds all the logic and never imports tkinter, so it is
unit-tested headless. :class:`SettingsWindow` is a thin tkinter view over it
(tkinter ships with Python; nothing extra to install). ``main()`` degrades to a
clear message when there is no display or no Tk.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import math
import os
import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from away import config as cfgmod
from away.actions import event_trigger
from away.bindings import (
    ACTION_ARGS,
    ACTION_KINDS,
    CATEGORY_PREFIX,
    BindingsError,
    list_rows,
    load_bindings_file,
    merge_bindings,
    parse_action,
    split_trigger,
)
from away.config import CAMERA_BACKENDS, PERFORMANCE_MODES, ROLE_MODES, UPDATE_CHANNELS, VOICE_MODES, AppConfig
from away.hands.gestures import GestureEvent

TUNABLE_LABELS = {
    "hands.cursor.gain": "Cursor sensitivity (gain)",
    "hands.cursor.dead_zone": "Dead zone",
    "hands.cursor.acceleration": "Acceleration",
    "hands.cursor.min_cutoff": "Smoothing: One Euro min cutoff",
    "hands.cursor.beta": "Smoothing: One Euro beta",
    "hands.cursor.d_cutoff": "Smoothing: One Euro d cutoff",
    "hands.gestures.pinch_on": "Pinch threshold (close)",
    "hands.gestures.pinch_off": "Pinch threshold (release)",
    "hands.gestures.scroll_gain": "Scroll speed",
    "hands.gestures.swipe_min_dx": "Swipe distance",
    "hands.gestures.swipe_max_time": "Swipe max time (lower = faster swipe needed)",
    "hands.gestures.double_pinch_window": "Double-pinch window (s)",
    "hands.gestures.drag_hold": "Pinch-hold before drag (s)",
    "hands.gestures.clutch_hold": "Open-palm hold to pause (s)",
    "hands.cursor.big_cursor_size": "Cursor size while gestures are active (0 = off)",
    "hands.gestures.autoscroll_hold": "Auto-scroll: four-finger hold before it starts (s)",
    "hands.gestures.autoscroll_thumb_fold": "Auto-scroll: thumb-fold threshold (higher = easier to trigger)",
    "hands.gestures.autoscroll_dead_zone": "Auto-scroll: dead zone (palm lengths)",
    "hands.gestures.autoscroll_gain": "Auto-scroll: speed (gain)",
    "hands.gestures.autoscroll_max_speed": "Auto-scroll: max speed (notches/s)",
    "hands.gestures.autoscroll_stop_hold": "Auto-scroll: fist hold to stop (s)",
    "hands.min_camera_fps": "Cameras: minimum usable frame rate (fps)",
    "voice.vad_sensitivity": "Voice sensitivity (always-on)",
    "voice.end_silence_ms": "End of utterance silence (ms)",
    "voice.min_utterance_ms": "Shortest utterance (ms)",
    "voice.max_utterance_s": "Longest utterance (s)",
    "voice.wake_window": "Wake-word follow-up window (s)",
    "face.mouth_threshold": "Mouth-open threshold",
    "face.mouth_hold": "Mouth-open hold (s)",
    "hands.cursor.sigmoid_low": "Sigmoid curve: slow-move gain",
    "hands.cursor.sigmoid_high": "Sigmoid curve: fast-move gain",
    "hands.cursor.sigmoid_mid": "Sigmoid curve: midpoint speed",
    "hands.cursor.sigmoid_slope": "Sigmoid curve: steepness",
    "hands.cursor.speed_up": "Speed up (per direction)",
    "hands.cursor.speed_down": "Speed down",
    "hands.cursor.speed_left": "Speed left",
    "hands.cursor.speed_right": "Speed right",
    "hands.cursor.click_rewind_ms": "Steady click: rewind (ms)",
    "hands.cursor.freeze_margin": "Steady click: freeze margin",
    "hands.dwell.radius_px": "Dwell click radius (px)",
    "hands.dwell.time_s": "Dwell click time (s)",
    "hands.multicam.pair_tolerance_ms": "Multi-camera pairing tolerance (ms)",
    "hands.multicam.push_distance_m": "Push gesture distance (m)",
    "face.brow_threshold": "Brow-raise threshold",
    "face.wink_threshold": "Wink: closed-eye threshold",
    "face.wink_open_max": "Wink: other eye open below",
    "face.wink_hold": "Wink hold (s)",
    "app.auto_pause_idle_ms": "Auto-pause: resume after idle (ms)",
    "voice.numbers_min_value": "Numbers as digits from",
    "voice.clap_sensitivity": "Clap sensitivity (higher = softer claps count)",
    "voice.clap_double_window": "Double-clap window (s)",
    "hands.roles.release_s": "Hand roles: pointer kept while out of view (s)",
    "hands.palm_scroll.engage_s": "Palm scroll: touch time before it scrolls (s)",
    "hands.palm_scroll.depth_threshold": "Palm scroll: fingertip distance from the palm (palm lengths, 0 = any)",
    "hands.palm_scroll.margin": "Palm scroll: palm outline padding (palm lengths)",
    "hands.palm_scroll.gain": "Palm scroll: speed (notches per palm length)",
    "hands.palm_scroll.dead_zone": "Palm scroll: dead zone / tap movement (palm lengths)",
    "hands.palm_scroll.friction": "Palm scroll: momentum friction (higher = stops sooner)",
}
# Settings > Tuning check boxes for palm-glide scroll
PALM_SCROLL_SWITCHES = (
    ("hands.palm_scroll.enabled", "Palm scroll: glide a fingertip along the other, flat hand"),
    ("hands.palm_scroll.invert", "Palm scroll: invert (toward the fingers scrolls down)"),
    ("hands.palm_scroll.horizontal", "Palm scroll: two fingers scroll sideways"),
    ("hands.palm_scroll.momentum", "Palm scroll: momentum after the fingertip lifts"),
)
# Settings > Voice & input "Hand roles" dropdown (hands.roles.mode; "any" is config-only)
ROLE_LABELS = {"auto": "Auto", "right": "Right", "left": "Left", "any": "Any (first hand seen)"}
ROLE_CHOICES = ("auto", "right", "left")
INT_TUNABLES = {
    "hands.cursor.big_cursor_size",
    "voice.end_silence_ms",
    "voice.min_utterance_ms",
    "hands.cursor.click_rewind_ms",
    "app.auto_pause_idle_ms",
    "voice.numbers_min_value",
    *cfgmod.TRACKING_INT_TUNABLES,  # auto-centering / head / gaze (v0.1.19)
}
TUNABLE_LABELS.update(cfgmod.TRACKING_TUNABLE_LABELS)
VOICE_FIELDS = {
    "hands.roles.mode": ROLE_MODES,
    "hands.cursor.curve": ("sigmoid", "power"),
    "hands.multicam.fusion": ("weighted", "best_view"),
    "voice.mode": VOICE_MODES,
    "voice.intent": ("dictation", "command"),
    "voice.ambient_fallback": ("commands_only", "type"),
    "updates.channel": UPDATE_CHANNELS,
    "performance.mode": PERFORMANCE_MODES,
    "hands.camera_backend": CAMERA_BACKENDS,
}
# Tuning tab glyphs, escaped: some tests read this file with the Windows default code page
OPEN_MARK, CLOSED_MARK, RESET_MARK, RECORDING_MARK = "\u25be ", "\u25b8 ", "\u21ba", "\u25cf"
PERFORMANCE_LABELS = {"auto": "Auto (detect)", "cpu": "CPU only", "gpu": "GPU (NVIDIA CUDA)"}


@dataclass
class Row:
    trigger: str  # gesture:pinch
    action: str  # key:ctrl+tab
    source: str  # profile that defines it
    kind: str
    arg: str


@dataclass
class Tunable:
    path: str
    label: str
    lo: float
    hi: float
    value: float
    is_int: bool
    # Settings > Tuning (away.app.tunables): the section, a plain-language title, what raising / lowering does,
    # the unit and the shipped default
    group: str = "Other"
    title: str = ""
    help: str = ""
    unit: str = ""
    default: float | None = None

    @property
    def resolution(self) -> float:
        """A round slider step (1, 0.01, 0.001...) about 1/200 of the range, so defaults are not snapped."""
        if self.is_int:
            return 1.0
        return float(10 ** math.floor(math.log10((self.hi - self.lo) / 200.0)))


def _get(cfg: Any, path: str) -> Any:
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set(cfg: Any, path: str, value: Any) -> None:
    *head, last = path.split(".")
    obj = cfg
    for part in head:
        obj = getattr(obj, part)
    setattr(obj, last, value)


class SettingsModel:
    """Everything the settings window does, minus the widgets."""

    def __init__(self, cfg: AppConfig | None = None, path: Path | None = None) -> None:
        self.path = path or cfgmod.config_path()
        if cfg is None:
            cfg, _ = cfgmod.load(self.path)
        self.cfg = cfg
        self._saved = copy.deepcopy(cfg)
        self.profile = "default"

    # ---------------------------------------------------------------- state
    @property
    def dirty(self) -> bool:
        return cfgmod.to_dict(self.cfg) != cfgmod.to_dict(self._saved)

    @property
    def bindings(self) -> Any:
        return self.cfg.bindings

    def profiles(self) -> list[str]:
        return list(self.bindings.profiles)

    def select_profile(self, name: str) -> None:
        if name not in self.bindings.profiles:
            raise ValueError(f"no profile named {name!r}")
        self.profile = name

    def profile_apps(self) -> list[str]:
        return list(self.bindings.profiles[self.profile].apps)

    def set_profile_apps(self, apps_text: str) -> str | None:
        if self.profile == "default":
            return "the default profile applies to every app"
        apps = [a.strip() for a in apps_text.replace(";", ",").split(",") if a.strip()]
        old = self.bindings.profiles[self.profile].apps
        self.bindings.profiles[self.profile].apps = apps
        try:
            from away.bindings import parse_bindings

            parse_bindings(self.bindings.to_dict())
        except BindingsError as exc:
            self.bindings.profiles[self.profile].apps = old
            return "; ".join(exc.errors)
        return None

    def new_profile(self, name: str, apps_text: str = "", copy_current: bool = False) -> str | None:
        apps = [a.strip() for a in apps_text.replace(";", ",").split(",") if a.strip()]
        try:
            self.bindings.new_profile(name, apps, self.profile if copy_current else None)
        except ValueError as exc:
            return str(exc)
        self.profile = name
        return None

    def active_profile_choices(self) -> list[str]:
        return ["auto", *self.profiles()]

    def set_active_profile(self, name: str) -> str | None:
        if name not in self.active_profile_choices():
            return f"{name!r} is not 'auto' or a profile"
        self.bindings.active_profile = name
        return None

    # ------------------------------------------------------------- bindings
    def rows(self) -> list[Row]:
        out = []
        for trig, act, src in list_rows(self.bindings, self.profile):
            kind, _, arg = act.partition(":")
            out.append(Row(trig, act, src, kind, arg))
        return out

    @staticmethod
    def action_kinds() -> list[str]:
        return list(ACTION_KINDS)

    def arg_choices(self, kind: str, trigger: str | None = None) -> list[str] | None:
        """Choices for the argument dropdown, or None when the argument is free text."""
        if kind not in ACTION_ARGS:
            return None
        choices = ACTION_ARGS[kind]
        if choices is None:
            return None
        if kind == "mouse" and trigger is not None:
            from away.bindings import CONTINUOUS_ONLY

            cat, name = split_trigger(trigger)
            choices = tuple(
                c for c in choices if c not in CONTINUOUS_ONLY or (cat == "gestures" and CONTINUOUS_ONLY[c] == name)
            )
        return list(choices)

    def compose(self, kind: str, arg: str) -> str:
        return "none" if kind == "none" else f"{kind}:{arg}"

    def set_binding(self, trigger: str, kind: str, arg: str = "") -> str | None:
        """Bind ``trigger`` in the current profile. Returns an error message or None."""
        try:
            self.bindings.set(trigger, self.compose(kind, arg.strip()), self.profile)
        except ValueError as exc:
            return str(exc)
        return None

    def add_trigger(self, category: str, name: str, kind: str = "none", arg: str = "") -> str | None:
        prefix = CATEGORY_PREFIX.get(category, category)
        return self.set_binding(f"{prefix}:{name}", kind, arg)

    def remove_binding(self, trigger: str) -> str | None:
        try:
            self.bindings.unset(trigger, self.profile)
        except (ValueError, KeyError) as exc:
            return str(exc)
        return None

    def validate_action_text(self, text: str) -> str | None:
        try:
            parse_action(text, allow_shell=self.bindings.allow_shell)
        except ValueError as exc:
            return str(exc)
        return None

    def set_allow_shell(self, on: bool) -> None:
        self.bindings.allow_shell = bool(on)

    # ------------------------------------------------------------- tunables
    def tunables(self) -> list[Tunable]:
        """Every slider, from :func:`away.config.tunable_ranges`; a tunable added there shows up on the Tuning tab
        in its section automatically (:func:`away.app.tunables.describe`, unknown ones under "Other")."""
        from away.app.tunables import cached_defaults, describe

        defaults = cached_defaults()
        out = []
        for p, v, lo, hi in cfgmod.tunable_ranges(self.cfg):
            label = TUNABLE_LABELS.get(p, p)
            meta = describe(p, label)
            out.append(
                Tunable(
                    p,
                    label,
                    float(lo),
                    float(hi),
                    float(v),
                    p in INT_TUNABLES or (isinstance(v, int) and not isinstance(v, bool) and isinstance(lo, int)),
                    meta.group,
                    meta.title,
                    meta.help,
                    meta.unit,
                    defaults.get(p),
                )
            )
        return out

    def set_tunable(self, path: str, value: float) -> str | None:
        spec = {t.path: t for t in self.tunables()}.get(path)
        if spec is None:
            return f"unknown setting {path!r}"
        v: float = round(value) if spec.is_int else float(value)
        if not spec.lo <= v <= spec.hi:
            return f"{spec.label} must be between {spec.lo:g} and {spec.hi:g}"
        old = _get(self.cfg, path)
        _set(self.cfg, path, int(v) if spec.is_int else v)
        try:
            cfgmod.validate(self.cfg)
        except ValueError as exc:
            _set(self.cfg, path, old)
            return str(exc)
        return None

    # ---------------------------------------------------------------- voice
    def voice_value(self, path: str) -> Any:
        return _get(self.cfg, path)

    def set_voice(self, path: str, value: Any) -> str | None:
        """voice.mode / voice.intent / voice.wake_word(_enabled) / voice.mute_hotkey / app.always_on ..."""
        allowed = {
            "voice.mode",
            "voice.intent",
            "voice.ambient_fallback",
            "voice.wake_word_enabled",
            "voice.wake_word",
            "voice.mute_hotkey",
            "voice.hotkey",
            "voice.hold_key",
            "voice.commands_enabled",
            "voice.enabled",
            "app.always_on",
            "app.start_minimized",
            "face.brow_raise_toggle",
            "face.mouth_open_trigger",
            "face.wink_trigger",
            "hands.dwell.enabled",
            "hands.cursor.click_freeze",
            "app.auto_pause",
            "app.toggle_hotkey",
            "voice.edit_hotkey",
            "voice.numbers_as_digits",
            "cursor_overlay",
            "hands.cursor.curve",
            "hands.multicam.fusion",
            "updates.check",
            "updates.auto_install",
            "updates.channel",
            "performance.mode",
            "face.gate_dictation",
            "hands.roles.mode",
            "hands.palm_scroll.enabled",
            "hands.palm_scroll.invert",
            "hands.palm_scroll.horizontal",
            "hands.palm_scroll.momentum",
            "hands.gestures.autoscroll_enabled",
            "hands.gestures.autoscroll_horizontal",
            "voice.clap_enabled",
            "voice.clap_require_hands",
            "hands.camera_backend",
        }
        if path not in allowed:
            return f"unknown setting {path!r}"
        if path in VOICE_FIELDS and value not in VOICE_FIELDS[path]:
            return f"{path} must be one of {', '.join(VOICE_FIELDS[path])}"
        old = _get(self.cfg, path)
        if isinstance(old, bool):
            value = bool(value)
        elif isinstance(old, str):
            value = str(value).strip()
        _set(self.cfg, path, value)
        try:
            cfgmod.validate(self.cfg)
        except ValueError as exc:
            _set(self.cfg, path, old)
            return str(exc)
        if path == "face.gate_dictation" and not value:
            # no face gate and no wake word must still type (commands_only would silently drop dictation)
            self.cfg.voice.ambient_fallback = "type"
        return None

    # ------------------------------------------------------------ hand roles
    @staticmethod
    def role_choices() -> list[str]:
        """The Hand roles dropdown: Auto / Right / Left."""
        return [ROLE_LABELS[m] for m in ROLE_CHOICES]

    def role_label(self) -> str:
        return ROLE_LABELS.get(self.cfg.hands.roles.mode, str(self.cfg.hands.roles.mode))

    def set_role_label(self, label: str) -> str | None:
        mode = next((k for k, v in ROLE_LABELS.items() if v.lower() == str(label).strip().lower()), None)
        if mode is None:
            mode = str(label).strip().lower()
        return self.set_voice("hands.roles.mode", mode)

    # ------------------------------------------------------------ file ops
    def save(self) -> Path:
        p = cfgmod.save(self.cfg, self.path)
        self._saved = copy.deepcopy(self.cfg)
        return p

    def revert(self) -> None:
        self.cfg = copy.deepcopy(self._saved)
        if self.profile not in self.bindings.profiles:
            self.profile = "default"

    def reset_profile(self) -> None:
        self.bindings.reset(self.profile)

    def reset_all(self) -> None:
        """Every setting back to the shipped defaults (not saved until :meth:`save`)."""
        self.cfg = AppConfig()
        self.profile = "default"

    def export_bindings(self, path: Path) -> Path:
        import yaml

        Path(path).write_text(yaml.safe_dump({"bindings": self.bindings.to_dict()}, sort_keys=False), "utf-8")
        return Path(path)

    def import_bindings(self, path: Path, replace: bool = False) -> str | None:
        try:
            incoming, raw = load_bindings_file(path)
            self.cfg.bindings = incoming if replace else merge_bindings(self.bindings, incoming, raw)
        except BindingsError as exc:
            return "\n".join(exc.errors)
        except Exception as exc:
            return f"cannot import {path}: {exc}"
        if self.profile not in self.bindings.profiles:
            self.profile = "default"
        return None

    # -------------------------------------------------------------- cameras
    MULTI_ONLY_HINT = "Calibrate and Live confidence need two cameras: add a second camera first."

    def app_cameras(self) -> set[str]:
        """Devices the running Project Away app holds right now (from its camera-owner heartbeat)."""
        share = getattr(self, "camera_share", None)
        if share is None:
            from away.app.camera_share import CameraShare

            share = self.camera_share = CameraShare()
        try:
            return set(share.owned_devices())
        except Exception:
            return set()

    def _listed(self) -> list[Any]:
        """The cameras the last 'Refresh list' found (empty before the first one, or when listing failed)."""
        return list(getattr(self, "listed", []) or [])

    def device_display(self, device: Any) -> str:
        """The true name of a configured camera for the Device column: ``1: HD User Facing``,
        ``0: Camera (NVIDIA Broadcast) (virtual)``, or ``Logitech BRIO (not connected)``."""
        from away.multicam.capture import find_device

        devs = self._listed()
        c = find_device(device, devs) if devs and not isinstance(device, bool) else None
        if c is None:
            if devs and isinstance(device, str) and not device.strip().isdigit():
                return f"{device} (not connected)"
            return str(device)
        idx = c.index if c.index is not None else c.path
        return f"{idx}: {c.name}" + (" (virtual)" if c.virtual else "")

    def camera_rows(self) -> list[dict[str, Any]]:
        """Cameras for the Cameras tab. Single-camera mode shows the implicit camera
        (``hands.camera_index``) as the primary row, marked ``implicit``."""
        busy = self.app_cameras()
        h, perf = self.cfg.hands, self.cfg.performance
        if not h.cameras:
            rows = [
                {
                    "device": h.camera_index,
                    "role": "primary",
                    "width": perf.camera_width or 640,
                    "height": perf.camera_height or 480,
                    "fps": perf.camera_fps or 30,
                    "hfov_deg": 70.0,
                    "mirror": h.mirror,
                    "implicit": True,
                }
            ]
        else:
            rows = [{**vars(c), "implicit": False} for c in h.cameras]
        for r in rows:
            r["in_use"] = str(r["device"]) in busy
            r["label"] = self.device_display(r["device"])
        return rows

    def camera_mode(self) -> str:
        n = len(self.cfg.hands.cameras)
        return f"{n} cameras (fusion: {self.cfg.hands.multicam.fusion})" if n >= 2 else "single camera"

    def role_default(self, selected: int | None = None) -> str:
        """What the Role picker shows: primary in single-camera mode (the only camera is the primary, so Apply on
        it just works), the selected camera's own role, or secondary for a camera about to be added. v0.1.19:
        the only camera (also a one-entry hands.cameras) and a set without a primary default to primary."""
        cams = self.cfg.hands.cameras
        if len(cams) <= 1:
            return "primary"
        if selected is not None and 0 <= selected < len(cams):
            return cams[selected].role
        if not any(c.role == "primary" for c in cams):
            return "primary"
        return "secondary"

    def device_choice(self, selected: int | None = None, choices: list[str] | None = None) -> str:
        """The Device picker entry for a camera row (v0.1.19: the picker was blank until a row was selected):
        the selected row's camera, else the only (or primary) camera; ``""`` when it is not in the list."""
        from away.multicam.capture import find_device

        h = self.cfg.hands
        if h.cameras:
            if selected is not None and 0 <= selected < len(h.cameras):
                dev: Any = h.cameras[selected].device
            else:
                dev = next((c.device for c in h.cameras if c.role == "primary"), h.cameras[0].device)
        else:
            dev = h.camera_index
        values = getattr(self, "choice_values", {}) or {}
        labels = list(choices) if choices is not None else list(values)
        for label in labels:
            if label in values and str(values[label]) == str(dev):
                return label
        listed = self._listed()
        if listed:
            d = find_device(dev, listed)
            if d is not None:
                for label in labels:
                    if label.split(":", 1)[0].strip() == str(d.index if d.index is not None else d.path):
                        return label
        for label in labels:  # "HD User Facing (configured)" / "1: camera 1 (in use by Project Away)"
            if label.split(" (", 1)[0].strip() == str(dev) or label.split(":", 1)[0].strip() == str(dev):
                return label
        return ""

    def camera_backend(self) -> str:
        return str(self.cfg.hands.camera_backend)

    @staticmethod
    def retry_status_text(folder: Any = None) -> str:
        """What the running app's camera retry is doing (Settings > Cameras), '' when nothing."""
        from away.app.camera_loop import read_retry_status, retry_status_text

        try:
            return retry_status_text(read_retry_status(folder))
        except Exception:
            return ""

    @staticmethod
    def request_camera_retry(folder: Any = None) -> str | None:
        """Retry cameras: ask the running app to probe its failed cameras now. An error message, or None."""
        from away.app.camera_loop import request_camera_retry

        return None if request_camera_retry(folder) else "could not ask Project Away to retry (data folder)"

    def multi_tools(self) -> tuple[bool, str]:
        """Are Calibrate / Live confidence usable? ``(enabled, hint)``: they need two configured cameras."""
        if len(self.cfg.hands.cameras) >= 2:
            return True, ""
        return False, self.MULTI_ONLY_HINT

    def device_choices(self, devices: list[Any] | None = None) -> list[str]:
        """``"1: HD User Facing"`` entries for the device pickers (``(virtual)`` marks NVIDIA Broadcast & co).

        Never empty while a camera is configured: devices the running app holds and the configured ones are
        always listed, even when the OS listing fails or misses them (Windows cameras are exclusive, so a
        held camera cannot be probed)."""
        from away.multicam.capture import config_device, device_label, find_device, list_cameras

        busy = self.app_cameras()
        if devices is None:
            try:
                devices = list_cameras(in_use=busy, log_list=True)  # 'Refresh list' -> away.log
            except Exception:
                devices = []
        devices = list(devices)
        self.listed = devices
        values: dict[str, Any] = {}
        have = {str(d.index if d.index is not None else d.path) for d in devices} | {str(d.path) for d in devices}
        have |= {d.name for d in devices} | {device_label(d, devices) for d in devices}
        have |= {d.device_path for d in devices if d.device_path}
        out: list[str] = []
        for d in devices:
            held = getattr(d, "in_use", False) or str(d.index) in busy or device_label(d, devices) in busy
            label = (
                f"{d.index if d.index is not None else d.path}: {d.name}"
                + (" (virtual)" if d.virtual else "")
                + (" (in use by Project Away)" if held else "")
            )
            values[label] = config_device(d, devices) if d.index is not None else d.path
            out.append(label)
        h = self.cfg.hands
        configured = [c.device for c in h.cameras] if h.cameras else [h.camera_index]
        extra: list[tuple[str, str]] = []
        for dev in [*sorted(busy), *configured]:
            key = str(dev)
            if key in have or key in (k for k, _ in extra):
                continue
            if devices and find_device(dev, devices) is not None:
                continue
            state = "in use by Project Away" if key in busy else "configured"
            label = f"{key}: camera {key} ({state})" if key.isdigit() else f"{key} ({state})"
            values[label] = int(key) if key.isdigit() else dev
            extra.append((key, label))
        self.known_devices = have | {k for k, _ in extra}
        self.choice_values = values
        return out + [label for _, label in extra]

    def parse_device(self, choice: Any) -> int | str:
        """A picker entry -> what is saved: the camera's name (stable when cameras are plugged in or reordered),
        or its index when the OS gave no name."""
        if isinstance(choice, int) and not isinstance(choice, bool):
            return choice
        values = getattr(self, "choice_values", {}) or {}
        if choice in values:
            return values[choice]  # type: ignore[no-any-return]
        head = str(choice).split(":", 1)[0].strip()
        return int(head) if head.isdigit() else head

    def _hint(self, device: Any) -> int | None:
        """The current index of a camera saved by name (kept as a hint for when no list is available)."""
        from away.multicam.capture import find_device

        if isinstance(device, int) or not self._listed():
            return None
        c = find_device(device, self._listed())
        return c.index if c is not None else None

    def _cams_try(self, new: list[Any]) -> str | None:
        from away.config import validate_cameras

        try:
            validate_cameras(new)
        except ValueError as exc:
            return str(exc)
        self.cfg.hands.cameras = new
        return None

    def _unknown_device(self, device: int | str) -> str | None:
        """Refuse a device the last 'Refresh list' did not find (a typo or a missing camera would make
        the multi-camera setup start with a dead view)."""
        known = getattr(self, "known_devices", None)
        if known and str(device) not in known:
            return f"camera {device} not found; pick one from the list (Refresh list)"
        return None

    def add_camera(self, device: int | str, role: str = "secondary") -> str | None:
        from away.config import CameraSpec

        device = self.parse_device(device)
        err = self._unknown_device(device)
        if err:
            return err
        h = self.cfg.hands
        new = [copy.copy(c) for c in h.cameras]
        if not new:  # leaving single-camera mode: the current camera becomes the primary
            new.append(
                CameraSpec(h.camera_index, "primary", index_hint=h.camera_hint, mirror=h.mirror, fourcc=h.camera_fourcc)
            )
            if role == "primary":
                new[0].role = "secondary"
        new.append(CameraSpec(device, role, index_hint=self._hint(device), mirror=h.mirror))
        return self._cams_try(new)

    def remove_camera(self, i: int) -> str | None:
        if not self.cfg.hands.cameras:
            return "it is the only camera; pick another device for it instead"
        new = [copy.copy(c) for j, c in enumerate(self.cfg.hands.cameras) if j != i]
        if len(new) == 1:  # back to single-camera mode
            self.cfg.hands.camera_index = new[0].device
            self.cfg.hands.camera_hint = new[0].index_hint
            self.cfg.hands.camera_fourcc = new[0].fourcc
            new = []
        return self._cams_try(new)

    def set_camera(self, i: int, field: str, value: Any) -> str | None:
        if field not in ("device", "role", "width", "height", "fps", "hfov_deg", "mirror"):
            return f"unknown camera setting {field!r}"
        if not self.cfg.hands.cameras and i == 0:
            return self._set_single_camera(field, value)
        new = [copy.copy(c) for c in self.cfg.hands.cameras]
        if not 0 <= i < len(new):
            return "no such camera"
        if field in ("width", "height", "fps"):
            try:
                value = int(value)
            except (TypeError, ValueError):
                return f"{field} must be a whole number"
        elif field == "hfov_deg":
            try:
                value = float(value)
            except (TypeError, ValueError):
                return "field of view must be a number"
        elif field == "device":
            value = self.parse_device(value)
            new[i].index_hint = self._hint(value)
        if field == "role" and value == "primary":  # promoting one demotes the old primary
            for c in new:
                c.role = "secondary"
        setattr(new[i], field, value)
        return self._cams_try(new)

    def _set_single_camera(self, field: str, value: Any) -> str | None:
        """Edit the implicit camera of single-camera mode (hands.camera_index + performance.camera_*)."""
        h, perf = self.cfg.hands, self.cfg.performance
        if field == "device":
            dev = self.parse_device(value)
            try:
                cfgmod.validate_camera_device("hands.camera_index", dev)
            except ValueError as exc:
                return str(exc)
            h.camera_index = dev
            h.camera_hint = self._hint(dev)
            return None
        if field == "role":
            return None if value == "primary" else "the only camera is always the primary"
        if field == "mirror":
            h.mirror = bool(value)
            return None
        if field == "hfov_deg":
            return None  # only used to triangulate two cameras
        try:
            v = int(value)
        except (TypeError, ValueError):
            return f"{field} must be a whole number"
        old = getattr(perf, f"camera_{field}")
        setattr(perf, f"camera_{field}", v)
        try:
            cfgmod.validate(self.cfg)
        except ValueError as exc:
            setattr(perf, f"camera_{field}", old)
            return str(exc)
        return None

    def calibration_command(self, status_file: Path) -> list[str]:
        """argv for the calibration subprocess (the installed exe is the CLI; ``-m away`` only from Python)."""
        from away.app.frozen import away_command

        return away_command(["calibrate-cameras", "--config", str(self.path), "--status-file", str(status_file)])

    @staticmethod
    def calibration_status_file() -> Path:
        from away import paths

        d = paths.data_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d / "calibration-status.json"

    @staticmethod
    def calibration_status() -> str:
        import time as _time

        from away.multicam.geometry import load_extrinsics

        ex = load_extrinsics()
        if ex is None:
            return "not calibrated (fusion works; depth/push/3D pinch need calibration)"
        when = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(ex.created))
        return f"calibrated {when} by {ex.method}, reprojection error {ex.reprojection_error_px:.2f} px"

    # -------------------------------------------------------- performance
    def performance_info(self, hw: Any = None) -> dict[str, Any]:
        """Detected hardware, the profile picked for it, and the GPU pack state for the Performance tab."""
        from away.app import gpupack, hardware, performance
        from away.app.frozen import is_frozen

        hw = hw or hardware.cached()
        prof = performance.current_profile(self.cfg, hw)
        frozen = is_frozen()
        if prof.gpu_pack == "installed":
            pack = "GPU pack installed"
        elif prof.gpu_pack == "recommended":
            pack = "GPU pack not installed: download it to use the GPU" if frozen else gpupack.pip_instructions()
        elif prof.gpu_pack == "not-needed" and prof.gpu_capable:
            pack = "CUDA libraries found (pip / CUDA toolkit)"
        elif prof.gpu_pack == "unavailable":
            pack = "GPU present but not usable for speech (see reasons)"
        else:
            pack = "no NVIDIA GPU: not needed"
        return {
            "lines": performance.describe(hw, prof),
            "mode": self.cfg.performance.mode,
            "profile": prof,
            "pack_status": pack,
            "can_install": bool(frozen and prof.gpu_pack == "recommended" and gpupack.platform_tag() is not None),
            "can_remove": gpupack.installed(),
        }

    @staticmethod
    def install_gpu_pack(progress: Any = print) -> str | None:
        """Download the GPU pack (blocking; the window runs it on a thread). Returns an error or None."""
        from away.app import gpupack

        try:
            gpupack.install(write=lambda s: None, progress=progress)
        except Exception as exc:
            return f"GPU pack download failed: {exc} (try again to resume)"
        return None

    # -------------------------------------------------------- gesture test
    def describe_event(self, ev: GestureEvent) -> str | None:
        """Text for the live 'test gesture' indicator: ``pinch -> mouse:click``."""
        if ev.kind in ("hand_lost",):
            return "no hand"
        if ev.kind == "autoscroll":
            d = ev.data.get("direction")
            if not d:
                return "auto-scroll: hold (move the hand up / down to scroll)"
            speed = max(abs(float(ev.data.get("vx", 0.0))), abs(float(ev.data.get("vy", 0.0))))
            return f"auto-scroll {d}  {speed:.0f} notches/s"
        if ev.kind == "autoscroll_stop":
            return f"auto-scroll stopped ({str(ev.data.get('reason', '')).replace('pose:', '')})"
        trig = event_trigger(ev)
        if trig is None:
            return None
        act = self.bindings.resolve("gestures", trig, self.profile)
        return f"{trig}  ->  {act}"


def read_calibration_status(path: Path) -> dict[str, Any] | None:
    import json

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def calibration_outcome(status: dict[str, Any] | None, rc: int, stderr: str) -> tuple[str | None, str | None]:
    """``(error, ok)`` for the status line once the calibration process has exited."""
    if status and status.get("state") == "done":
        return None, str(status.get("message") or "calibrated")
    if status and status.get("state") == "failed":
        return str(status.get("message") or "calibration failed"), None
    if rc == 0:
        return None, "calibrated"
    tail = (stderr.strip().splitlines() or [""])[-1][-200:]
    return f"calibration failed (exit {rc}){': ' + tail if tail else ''}", None


def camera_available() -> tuple[bool, str]:
    """Can the live gesture test run? (cv2 + mediapipe + downloaded hand model)."""
    import importlib.util

    for mod in ("cv2", "mediapipe"):
        try:
            if importlib.util.find_spec(mod) is None:
                return False, f"{mod} not installed (pip install 'project-away[hands]')"
        except (ImportError, ValueError):
            return False, f"{mod} not installed"
    from away import paths

    cfg, _ = cfgmod.load()
    mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
    if not (mdir / "hand_landmarker.task").exists():
        return False, "hand model missing (run `away setup`)"
    return True, "camera test ready"


def gesture_test_worker(
    model: SettingsModel,
    out: queue.Queue[str],
    stop: threading.Event,
    *,
    share: Any = None,
    frames: Any = None,
    tracker_factory: Any = None,
    lease_timeout: float = 4.0,
    clap_mic: Any = None,
) -> None:
    """Camera -> hand landmarks -> state machine -> indicator text (no actions are executed).

    The running app holds the webcam (exclusive on Windows), so the camera is borrowed from it first
    (:class:`~away.app.camera_share.CameraLease`) and given back when the test stops or the window closes.
    """
    from away import paths
    from away.app.camera_share import CameraLease, CameraShare, camera_error_message
    from away.hands.gestures import GestureStateMachine

    cfg = model.cfg
    idx = cfg.hands.camera_index
    real_devices = frames is None  # with a real camera the real microphone listens for claps too
    if frames is None:  # pragma: no cover - real camera
        from away.hands.tracker import camera_frames as frames
    if tracker_factory is None:  # pragma: no cover - real MediaPipe

        def tracker_factory() -> Any:
            from away.hands.tracker import HandTracker

            mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
            return HandTracker(
                mdir / "hand_landmarker.task",
                cfg.hands.max_hands,
                prefer=cfg.hands.roles.mode,
                mirrored=cfg.hands.mirror,
                release_s=cfg.hands.roles.release_s,
            )

    lease = CameraLease(share or CameraShare(), timeout=lease_timeout, who="settings-test")
    try:
        lease.open()
        if lease.handed_over:
            out.put("Project Away paused hand tracking for the test (it resumes when you stop)")
        elif lease.app_running:
            out.put("waiting for Project Away to release the camera...")
        tracker = tracker_factory()
        sm = GestureStateMachine(cfg.hands.gestures)
        last: str | None = None
        views: Any = collections.deque(maxlen=90)
        mic = start_clap_test(model, out, views, clap_mic) if (real_devices or clap_mic is not None) else None
        try:
            for t, rgb in frames(idx, cfg.hands.mirror):
                if stop.is_set():
                    break
                fr = tracker.process(rgb, t)
                views.append((t, getattr(fr, "n_hands", 0), getattr(fr, "hands_gap", None)))
                for ev in sm.update(t, fr.landmarks):
                    if ev.kind == "move":
                        continue
                    txt = model.describe_event(ev)
                    if txt and txt != last:  # auto-scroll reports every frame: only show changes
                        out.put(txt)
                        last = txt
        except Exception as exc:
            out.put(camera_error_message(exc, idx, lease.app_running, lease.handed_over))
        finally:
            tracker.close()
            if mic is not None:
                with contextlib.suppress(Exception):
                    mic.__exit__(None, None, None)
    finally:
        lease.close()


def clap_test_handler(model: SettingsModel, out: Any, views: Any) -> Any:
    """Mic chunk -> clap detection -> indicator text (``clap -> pause:off``). Nothing is executed."""
    import time as _time

    from away.voice.clap import ClapControl, camera_view

    ctl = ClapControl(model.cfg.voice, model.cfg.voice.sample_rate, hands=lambda t0: camera_view(views, t0))

    def on_chunk(chunk: Any, t: float | None = None) -> None:
        for kind, tc in ctl.feed(chunk, _time.monotonic() if t is None else t):
            name = "clap" if kind == "single" else "double_clap"
            txt = model.describe_event(GestureEvent(name, tc, {}))
            if txt:
                out.put(txt)

    return on_chunk


def start_clap_test(model: SettingsModel, out: Any, views: Any, mic: Any = None) -> Any:
    """Listen for claps during the camera test (voice.clap_enabled). ``mic(on_chunk)`` returns a started
    context manager; the default opens the microphone with sounddevice. None when unavailable."""
    if not model.cfg.voice.clap_enabled:
        return None
    handler = clap_test_handler(model, out, views)
    try:
        if mic is None:  # pragma: no cover - real microphone
            from away.voice.dictation import MicStream

            stream = MicStream(handler, model.cfg.voice.sample_rate)
            stream.__enter__()
            return stream
        return mic(handler)
    except Exception as exc:  # no mic / sounddevice missing: the gesture test still runs
        out.put(f"(clap test unavailable: {exc})")
        return None


def conf_bar(conf: float) -> str:
    return ("#" * round(conf * 10)).ljust(10, ".") + f" {conf:.2f}"


def short_camera_error(err: str) -> str:
    """A camera thread's error, short enough for the Confidence column."""
    low = err.lower()
    if "did not hand" in low:
        return "error: app kept it"
    if "cannot open" in low or "did not open" in low:
        return "error: busy / did not open"
    if "no frame" in low or "sent no frames" in low:
        return "no frames"
    if "stopped delivering" in low:
        return "error: stopped"
    return "error: " + (err if len(err) <= 32 else err[:31] + "\u2026")


def thread_too_slow(th: Any, min_fps: float) -> float | None:
    """The measured fps when camera thread ``th`` is below ``min_fps`` (v0.1.16), else None."""
    check = getattr(th, "too_slow", None)
    if min_fps <= 0 or not callable(check):
        return None
    try:
        if not check(min_fps):
            return None
    except Exception:
        return None
    fps = getattr(th, "fps", None)
    live = fps() if callable(fps) else 0.0
    measured = getattr(th, "measured_fps", None)
    if not live and isinstance(measured, (int, float)):
        live = float(measured)
    return float(live)


def live_cells(threads: list[Any], confs: list[float | None] | None = None, min_fps: float = 0.0) -> list[str]:
    """The Confidence column during Live confidence: ``opening\u2026`` until the camera opened, its hand
    confidence once frames arrive, ``no frames`` when it opened but nothing comes, ``too slow: 1.0 fps`` below
    ``min_fps``, or a short error."""
    out: list[str] = []
    for i, th in enumerate(threads):
        conf = confs[i] if confs is not None and i < len(confs) else None
        err = getattr(th, "error", None)
        state = getattr(th, "state", "")
        slow = thread_too_slow(th, min_fps)
        if err:
            out.append(short_camera_error(str(err)))
        elif slow is not None:
            out.append(f"FAILED: too slow ({slow:.1f} fps)")
        elif conf is not None:
            out.append(conf_bar(conf))
        elif state in ("waiting", "opening"):
            out.append("opening\u2026")
        elif getattr(th, "frames", 0) > 0:
            out.append("streaming")
        elif state == "open":
            out.append("no frames")
        else:
            out.append(state or "-")
    return out


def live_status_line(threads: list[Any], min_fps: float = 0.0) -> str:
    """``live: JoyCam 640x480@30 MJPG (DirectShow) | HD User Facing 320x240@15 MJPG (DirectShow)``; a camera
    below ``min_fps`` says ``FAILED - too slow: 1.0 fps (minimum 10)``, one NVIDIA Broadcast replaced says so."""
    parts = []
    for th in threads:
        dev = str(getattr(getattr(th, "spec", None), "device", "?"))
        describe = getattr(th, "describe_mode", None)
        mode = describe() if callable(describe) else ""
        err = getattr(th, "error", None)
        slow = thread_too_slow(th, min_fps)
        via = getattr(th, "opened_name", "")
        if isinstance(via, str) and via and getattr(th, "fallback", ""):
            dev = f"{dev} (via {via}: {th.fallback})"
        if slow is not None and not err:
            parts.append(f"{dev} FAILED - too slow: {slow:.1f} fps (minimum {min_fps:g})")
        else:
            parts.append(f"{dev} {mode}" if mode else f"{dev} {'failed' if err else 'opening...'}")
    return "live: " + " | ".join(parts)


class CameraTools:
    """Which camera tool of the Cameras tab is running: ``live`` (Live confidence), ``test`` (Test both
    cameras) or ``calibrate``. Only one at a time: they would fight over the same devices (v0.1.15)."""

    LABELS: ClassVar[dict[str, str]] = {
        "live": "Live confidence",
        "test": "the camera test",
        "calibrate": "calibration",
    }

    def __init__(self) -> None:
        self.busy: str | None = None
        self._lock = threading.Lock()

    def begin(self, kind: str) -> str | None:
        """Start ``kind``: None, or why it cannot start now. Calibrate may take over from Live confidence
        (it stops it first); nothing else may take over."""
        with self._lock:
            if self.busy is None or self.busy == kind or (kind == "calibrate" and self.busy == "live"):
                self.busy = kind
                return None
            return f"{self.LABELS[self.busy]} is running; wait for it to finish (or stop it) first"

    def end(self, kind: str) -> None:
        with self._lock:
            if self.busy == kind:
                self.busy = None

    def states(self, multi_ok: bool) -> dict[str, str]:
        """Button states: ``normal`` or ``disabled`` for ``calibrate``, ``live``, ``test``."""
        b = self.busy
        return {
            "calibrate": "normal" if multi_ok and b in (None, "live") else "disabled",
            "live": "normal" if (multi_ok and b is None) or b == "live" else "disabled",
            "test": "normal" if b is None else "disabled",
        }


def stop_live_view(stop: threading.Event, worker: threading.Thread | None, timeout: float = 6.0) -> bool:
    """Stop Live confidence and wait until its camera threads have let go of the devices (the worker joins
    them before it ends). True when it ended in time."""
    stop.set()
    if worker is None:
        return True
    worker.join(timeout)
    return not worker.is_alive()


def calibration_flow(
    leases: Any,
    stop_live: Callable[[], bool],
    spawn: Callable[[], dict[str, Any]],
    who: str = "settings-calibrate",
) -> dict[str, Any]:
    """Calibrate from Settings: hold the cameras (``leases``: a :class:`~away.app.camera_share.SharedLease`,
    so the app cannot grab them between Live confidence and calibration), stop Live confidence and wait until
    its devices are free, then run the calibration subprocess; give the cameras back afterwards."""
    leases.acquire(who)
    try:
        if not stop_live():
            return {"rc": -1, "err": "Live confidence did not release the cameras in time; try again in a moment"}
        return spawn()
    finally:
        leases.release()


def camera_confidence_worker(
    model: SettingsModel,
    out: queue.Queue[Any],
    stop: threading.Event,
    *,
    leases: Any = None,
    start: Callable[..., list[Any]] | None = None,
    trackers: list[Any] | None = None,
    sleep: Callable[[float], Any] | None = None,
    report_s: float = 0.25,
) -> None:
    """Live per-camera hand confidence for the Cameras tab (no actions are executed).

    Puts ``{"cells": [...], "status": "live: ..."}`` on ``out`` (each camera: opening / confidence /
    no frames / error, and the mode that worked), or a string on a fatal error. The cameras are borrowed from
    the running app and opened the same way the app opens them (:func:`start_camera_threads`); when the
    worker ends, its camera threads have released their devices."""
    import time as _time

    from away.app.camera_share import SharedLease, camera_error_message
    from away.multicam import capture
    from away.multicam.capture import FramePairer, start_camera_threads, stop_camera_threads

    sleep = sleep or _time.sleep
    cams = model.cfg.hands.cameras
    floor = float(getattr(model.cfg.hands, "min_camera_fps", 0.0) or 0.0)
    primary = next((i for i, c in enumerate(cams) if c.role == "primary"), 0)
    pairer = FramePairer(len(cams), primary, model.cfg.hands.multicam.pair_tolerance_ms / 1000.0, 0.5)
    leases = leases or SharedLease()
    lease = leases.acquire("settings-confidence")  # borrow the cameras from the running app
    cam_stop = threading.Event()
    threads: list[Any] = []
    try:
        if trackers is None:  # pragma: no cover - real models
            from away import paths
            from away.hands.tracker import HandTracker

            mdir = Path(model.cfg.models_dir) if model.cfg.models_dir else paths.models_dir()
            trackers = [HandTracker(mdir / "hand_landmarker.task", 1) for _ in cams]
        from away.multicam.fusion import ViewObs, view_confidence

        if start is None:  # pragma: no cover - real cameras
            import functools

            capture.configure(model.cfg.hands)
            capture.use_mode_store(capture.default_mode_file())
            start = functools.partial(start_camera_threads, min_fps=floor)
        threads = start(list(cams), pairer, cam_stop, virtual_fallback=bool(model.cfg.hands.virtual_fallback))
        confs: list[float | None] = [None] * len(cams)
        last = -1.0
        while not stop.is_set():
            st = pairer.pop_set()
            if st is not None:
                alive = pairer.alive()
                for i, item in enumerate(st[1]):
                    if item is None:
                        if not alive[i]:
                            confs[i] = None
                        continue
                    fr = trackers[i].process(item[1], item[0])
                    h, w = item[1].shape[:2]
                    confs[i] = (
                        0.0
                        if fr.landmarks is None
                        else view_confidence(ViewObs(i, item[0], fr.landmarks, fr.handedness, fr.score, (w, h)))
                    )
            now = _time.monotonic()
            if now - last >= report_s:
                last = now
                for i, th in enumerate(threads):
                    if getattr(th, "error", None) or not th.is_alive():
                        confs[i] = None
                out.put({"cells": live_cells(threads, confs, floor), "status": live_status_line(threads, floor)})
            if st is None:
                sleep(0.005)
    except Exception as exc:
        out.put(camera_error_message(exc, "", lease.app_running, lease.handed_over))
    finally:
        stop.set()
        stop_camera_threads(cam_stop, threads)  # the devices are free before the lease hands them back
        if threads:
            out.put(
                {"cells": live_cells(threads, None, floor), "status": live_status_line(threads, floor), "final": True}
            )
        leases.release()


def camera_test_worker(
    model: SettingsModel,
    out: queue.Queue[Any],
    *,
    leases: Any = None,
    tester: Callable[..., Any] | None = None,
    seconds: float = 10.0,
) -> None:
    """Settings > Cameras > Test both cameras: the same test as ``away cameras test``. Puts progress
    strings and finally ``{"report": TestReport}`` (or ``{"error": str}``) on ``out``."""
    from away.app.camera_share import SharedLease, confirm_handover
    from away.multicam import camtest, capture

    leases = leases or SharedLease()
    lease = leases.acquire("settings-camera-test")
    try:
        confirm_handover(lease, say=lambda m: out.put(str(m)))  # never open what the app is still releasing
        if lease.app_running and not lease.handed_over:
            out.put({"error": "Project Away is using the cameras and did not hand them over in time - pause it"})
            return
        capture.configure(model.cfg.hands)
        if tester is None:  # pragma: no cover - real cameras
            import functools

            capture.use_mode_store(capture.default_mode_file())
            tester = functools.partial(
                camtest.run_camera_test,
                min_fps=float(model.cfg.hands.min_camera_fps),
                virtual_fallback=bool(model.cfg.hands.virtual_fallback),
            )
        report = tester(list(model.cfg.hands.cameras), seconds, progress=lambda m: out.put(str(m)))
        out.put({"report": report})
    except Exception as exc:
        out.put({"error": f"camera test failed: {exc}"})
    finally:
        leases.release()


def camera_diagnose_worker(
    model: SettingsModel,
    out: queue.Queue[Any],
    *,
    leases: Any = None,
    runner: Callable[..., Any] | None = None,
    lister: Callable[[], list[Any]] | None = None,
) -> None:
    """Settings > Cameras > Diagnose (v0.1.19): the same report as ``away cameras diagnose``, one line at a time
    on ``out`` (strings), then ``{"done": True}`` (or ``{"error": str}``). The cameras are borrowed from the app."""
    from away.app.camera_share import SharedLease, confirm_handover
    from away.multicam import capture, diagnose

    leases = leases or SharedLease()
    lease = leases.acquire("settings-camera-diagnose")
    try:
        confirm_handover(lease, say=lambda m: out.put(str(m)))
        if lease.app_running and not lease.handed_over:
            out.put({"error": "Project Away is using the cameras and did not hand them over in time - pause it"})
            return
        capture.configure(model.cfg.hands)
        h = model.cfg.hands
        spec = h.cameras[0] if h.cameras else None
        perf = getattr(model.cfg, "performance", None)
        width = spec.width if spec is not None else int(getattr(perf, "camera_width", 0) or 0) or 640
        height = spec.height if spec is not None else int(getattr(perf, "camera_height", 0) or 0) or 480
        fps = spec.fps if spec is not None else 30
        devs = (lister or (lambda: diagnose.diagnose_devices(None)))()
        out.put("diagnosing every camera on every backend (this takes a while)...")
        (runner or diagnose.run_diagnose)(devs, width=width, height=height, fps=fps, out=lambda m: out.put(str(m)))
        out.put({"done": True})
    except Exception as exc:
        out.put({"error": f"camera diagnosis failed: {exc}"})
    finally:
        leases.release()


# ---------------------------------------------------------------------------------- layout helpers
# Pure functions behind the window layout, so the sizes and texts are unit-tested without a display.
BASE_DPI = 96.0
DEFAULT_SIZE = (900, 640)  # logical (96 dpi) pixels; fits a 1366x768 screen with the taskbar at 100 %
MIN_SIZE = (600, 420)  # logical; the scroll bars handle anything smaller than the content
FONT_PT = 10  # ttk default fonts (positive = points, so they follow `tk scaling`)
BUTTON_PADDING = (10, 4)  # ttk.Button padding in pixels (x, y): text never touches the border


def tk_scaling(pixels_per_inch: float) -> float:
    """``tk scaling`` value (pixels per point) for a display: 96 dpi -> 1.333, 144 dpi (150 %) -> 2.0."""
    ppi = float(pixels_per_inch) if pixels_per_inch and pixels_per_inch > 0 else BASE_DPI
    return ppi / 72.0


def ui_scale(pixels_per_inch: float) -> float:
    """Windows scale factor (1.0 = 100 %, 1.25, 1.5...) from the display DPI; never below 1."""
    ppi = float(pixels_per_inch) if pixels_per_inch and pixels_per_inch > 0 else BASE_DPI
    return max(1.0, ppi / BASE_DPI)


def window_geometry(screen_w: int, screen_h: int, scale: float = 1.0) -> tuple[int, int, int, int]:
    """``(width, height, x, y)``: :data:`DEFAULT_SIZE` scaled for the DPI, shrunk to fit the screen (minus a
    margin and the taskbar) and centred. 1366x768 at 100 % -> 900x640."""
    max_w, max_h = max(320, screen_w - 40), max(240, int(screen_h * 0.88))
    w = min(int(DEFAULT_SIZE[0] * scale), max_w)
    h = min(int(DEFAULT_SIZE[1] * scale), max_h)
    return w, h, max(0, (screen_w - w) // 2), max(0, (screen_h - h) // 3)


def min_window_size(screen_w: int, screen_h: int, scale: float = 1.0) -> tuple[int, int]:
    """Smallest window that still shows every bottom-bar button: :data:`MIN_SIZE` scaled, never larger than
    what :func:`window_geometry` opens."""
    w, h, _x, _y = window_geometry(screen_w, screen_h, scale)
    return min(int(MIN_SIZE[0] * scale), w), min(int(MIN_SIZE[1] * scale), h)


def _wheel_int(value: Any) -> int:
    """A Tk event field as an int; Tk fills fields an event does not carry with ``'??'`` (v0.1.12)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return 0


def wheel_steps(delta: Any = 0, num: Any = 0, platform: str | None = None) -> int:
    """Scroll units for a mouse-wheel event (negative = up). Windows: ``delta`` in multiples of 120 (a
    high-resolution wheel sends smaller values: still one step); macOS: ``delta`` is already in steps;
    X11: Button-4 = up, Button-5 = down. A field Tk could not fill (``'??'``) counts as 0, never an error."""
    plat = platform or sys.platform
    n = _wheel_int(num)
    if n == 4:
        return -1
    if n == 5:
        return 1
    d = _wheel_int(delta)
    if not d:
        return 0
    if plat == "darwin":
        return -d
    steps = int(-d / 120)
    return steps if steps else (-1 if d > 0 else 1)


def short_path(path: Path | str, max_len: int = 48) -> str:
    """A path that fits a status line: the whole path when short, else ``…\\<parent>\\<name>``."""
    s = str(path)
    if len(s) <= max_len:
        return s
    sep = "\\" if "\\" in s else "/"
    parts = [p for p in s.replace("\\", "/").split("/") if p]
    tail = sep.join(parts[-2:])
    if len(tail) + 2 > max_len:
        tail = parts[-1]
    return "…" + sep + tail


def status_text(kind: str, path: Path | str | None = None, dirty: bool = False) -> str:
    """The bottom status line, short enough for one line of a narrow window."""
    p = short_path(path) if path is not None else ""
    if kind == "saved":
        return "Saved. Changes apply within a second."
    if kind == "imported":
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1] if path is not None else "bindings"
        return f"Imported {name}."
    if kind == "exported":
        return f"Exported to {p}."
    return ("Unsaved changes. " if dirty else "") + f"Config file: {p}"


def status_wraplength(width: int, margin: int = 16) -> int:
    """``wraplength`` for the full-width status label of a window ``width`` pixels wide."""
    return max(120, int(width) - margin)


def scroll_window_height(canvas_h: int, content_h: int) -> int:
    """Height of the frame inside a scrollable tab: the visible height when the content fits (so a tab that
    stretches, such as the bindings list, still fills the window), else the content's own height."""
    return max(int(canvas_h), int(content_h))


class SettingsWindow:  # pragma: no cover - widgets; smoke-tested under Xvfb only
    def __init__(self, root: Any, model: SettingsModel) -> None:
        import tkinter as tk
        from tkinter import ttk

        from away.control.keys import ComboCapture

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.m = model
        self.capture = ComboCapture()
        self._capture_target: Any = None
        from away.app.uiqueue import UiQueue

        self._ui = UiQueue()  # worker threads post Tk work here; only the Tk thread touches widgets (v0.1.16)
        self._test_q: queue.Queue[str] = queue.Queue()
        self._test_stop = threading.Event()
        root.title("Project Away settings")
        self._scrollers: list[tuple[Any, Any]] = []  # (outer tab frame, canvas) of every scrollable tab
        scale = self._apply_style()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h, x, y = window_geometry(sw, sh, scale)
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.minsize(*min_window_size(sw, sh, scale))

        top = ttk.Frame(root, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Profile:").pack(side="left")
        self.profile_var = tk.StringVar(value=model.profile)
        self.profile_cb = ttk.Combobox(top, textvariable=self.profile_var, state="readonly", width=16)
        self.profile_cb.pack(side="left", padx=4)
        self.profile_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_profile())
        ttk.Label(top, text="Apps:").pack(side="left", padx=(8, 0))
        self.apps_var = tk.StringVar()
        self.apps_entry = ttk.Entry(top, textvariable=self.apps_var, width=28)
        self.apps_entry.pack(side="left", padx=4)
        ttk.Button(top, text="Set apps", command=self._on_apps).pack(side="left")
        ttk.Button(top, text="New profile...", command=self._on_new_profile).pack(side="left", padx=4)
        ttk.Label(top, text="Use:").pack(side="left", padx=(8, 0))
        self.active_var = tk.StringVar(value=model.bindings.active_profile)
        self.active_cb = ttk.Combobox(top, textvariable=self.active_var, state="readonly", width=12)
        self.active_cb.pack(side="left", padx=4)
        self.active_cb.bind(
            "<<ComboboxSelected>>", lambda _e: self._report(model.set_active_profile(self.active_var.get()))
        )

        # The bottom bar is packed before the notebook so it keeps its space however small the window gets;
        # the tabs scroll instead. Status text gets its own full-width row under the buttons.
        status_row = ttk.Frame(root, padding=(8, 0, 8, 6))
        status_row.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value=status_text("config", model.path))
        status_lbl = ttk.Label(status_row, textvariable=self.status, foreground="#555", anchor="w", justify="left")
        status_lbl.pack(fill="x")
        status_row.bind("<Configure>", lambda e: status_lbl.configure(wraplength=status_wraplength(e.width)))
        bottom = ttk.Frame(root, padding=(6, 6, 6, 2))
        bottom.pack(side="bottom", fill="x")
        for text, cmd in (
            ("Save", self._on_save),
            ("Reset profile", self._on_reset_profile),
            ("Reset all", self._on_reset_all),
            ("Import...", self._on_import),
            ("Export...", self._on_export),
            ("Close", self._on_close),
        ):
            ttk.Button(bottom, text=text, command=cmd).pack(side="left", padx=3)

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=6)
        self._build_bindings(nb)
        self._build_tuning(nb)
        self._build_voice(nb)
        self._build_cameras(nb)
        self._build_performance(nb)
        self._build_studio(nb)  # v0.1.20: Gesture Studio replaces the old "Test gesture" tab
        self._bind_wheel()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh()
        self._ui.pump(root)
        from away.app import crashlog

        note = crashlog.last_crash_note() or crashlog.recent_note()
        if note:
            self._report(None, f"{note} ({crashlog.crash_log_path()})")

    # ---------------------------------------------------------------- layout
    def _apply_style(self) -> float:
        """Readable fonts (about 10 pt, following the real DPI) and button padding; returns the UI scale."""
        from tkinter import font as tkfont

        root = self.root
        ppi = 96.0
        with contextlib.suppress(Exception):
            ppi = float(root.winfo_fpixels("1i"))
            root.tk.call("tk", "scaling", tk_scaling(ppi))
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
            with contextlib.suppress(Exception):
                tkfont.nametofont(name).configure(size=FONT_PT)
        style = self.ttk.Style(root)
        style.configure("TButton", padding=BUTTON_PADDING)
        with contextlib.suppress(Exception):  # Treeview rows sized for the font, not a fixed 20 px
            line = tkfont.nametofont("TkDefaultFont").metrics("linespace")
            style.configure("Treeview", rowheight=int(line) + 6)
        return ui_scale(ppi)

    def _scroll_tab(self, nb: Any, text: str, padding: int = 8) -> Any:
        """A notebook tab whose content scrolls vertically (scroll bar + mouse wheel). Returns the frame to
        fill. The inner frame is as wide as the tab and at least as tall, so stretching layouts still fill it."""
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(nb)
        nb.add(outer, text=text)
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        f = ttk.Frame(canvas, padding=padding)
        win = canvas.create_window((0, 0), window=f, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def fit(_e: Any = None) -> None:
            h = scroll_window_height(canvas.winfo_height(), f.winfo_reqheight())
            canvas.itemconfigure(win, width=canvas.winfo_width(), height=h)
            canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), h))

        f.bind("<Configure>", lambda _e: canvas.after_idle(fit))
        canvas.bind("<Configure>", fit)
        self._scrollers.append((outer, canvas))
        return f

    def _bind_wheel(self) -> None:
        """One mouse-wheel handler for the window: it scrolls the tab under the pointer, and only when that
        tab's content is taller than the window. Lists keep their own wheel scrolling; a combo box no
        longer changes its value when the wheel passes over it (it scrolls the tab instead)."""
        root = self.root

        def canvas_under(e: Any) -> Any:
            w = None
            with contextlib.suppress(Exception):
                w = root.winfo_containing(e.x_root, e.y_root)
            while w is not None:
                for outer, canvas in self._scrollers:
                    if w is outer or w is canvas:
                        return canvas
                w = getattr(w, "master", None)
            return None

        def on_wheel(e: Any, own_list: bool = True) -> str | None:
            if own_list:
                with contextlib.suppress(Exception):
                    if root.winfo_containing(e.x_root, e.y_root).winfo_class() in ("Treeview", "Listbox", "Text"):
                        return None
            canvas = canvas_under(e)
            steps = wheel_steps(getattr(e, "delta", 0), getattr(e, "num", 0))
            if canvas is not None and steps and canvas.yview() != (0.0, 1.0):
                canvas.yview_scroll(steps, "units")
            return None

        def combo_wheel(e: Any) -> str:
            on_wheel(e, own_list=False)
            return "break"

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.bind_all(seq, on_wheel, add="+")
            root.bind_class("TCombobox", seq, combo_wheel)

    # ---------------------------------------------------------------- tabs
    def _build_bindings(self, nb: Any) -> None:
        tk, ttk = self.tk, self.ttk
        f = self._scroll_tab(nb, "Bindings", 6)
        cols = ("trigger", "action", "from")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=8)  # grows with the window
        for c, w in zip(cols, (240, 320, 100)):
            self.tree.heading(c, text=c.title())
            self.tree.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, columnspan=6, sticky="nsew")
        sb.grid(row=0, column=6, sticky="ns")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(3, weight=1)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        ttk.Label(f, text="Action:").grid(row=1, column=0, sticky="w", pady=4)
        self.kind_var = tk.StringVar(value="none")
        self.kind_cb = ttk.Combobox(
            f, textvariable=self.kind_var, values=self.m.action_kinds(), state="readonly", width=10
        )
        self.kind_cb.grid(row=1, column=1, sticky="w")
        self.kind_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_kind())
        self.arg_var = tk.StringVar()
        self.arg_cb = ttk.Combobox(f, textvariable=self.arg_var, width=34)
        self.arg_cb.grid(row=1, column=2, columnspan=2, sticky="we", padx=4)
        self.capture_btn = ttk.Button(f, text="Capture keys", command=lambda: self._start_capture(self.arg_var))
        self.capture_btn.grid(row=1, column=4, padx=2)
        ttk.Button(f, text="Apply", command=self._on_apply).grid(row=1, column=5, padx=2)

        ttk.Label(f, text="Add trigger:").grid(row=2, column=0, sticky="w")
        self.new_cat = tk.StringVar(value="voice")
        ttk.Combobox(
            f, textvariable=self.new_cat, values=["voice", "hotkeys", "gestures"], state="readonly", width=10
        ).grid(row=2, column=1, sticky="w")
        self.new_name = tk.StringVar()
        ttk.Entry(f, textvariable=self.new_name, width=34).grid(row=2, column=2, columnspan=2, sticky="we", padx=4)
        ttk.Button(f, text="Capture keys", command=lambda: self._start_capture(self.new_name)).grid(row=2, column=4)
        ttk.Button(f, text="Add", command=self._on_add).grid(row=2, column=5)
        ttk.Button(f, text="Remove / reset selected", command=self._on_remove).grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        self.shell_var = tk.BooleanVar(value=self.m.bindings.allow_shell)
        ttk.Checkbutton(
            f,
            text="Allow shell-command actions (off = safer)",
            variable=self.shell_var,
            command=lambda: self.m.set_allow_shell(self.shell_var.get()),
        ).grid(row=3, column=2, columnspan=3, sticky="w")

    def _build_tuning(self, nb: Any) -> None:
        """Settings > Tuning: presets + master sensitivity + undo + per-app profile, a live view, auto-tune, then
        one collapsible section per area (sections and sliders come from the tunables, see away.app.tunables)."""
        from away.app.tunables import PRESETS, TuningController, sensitivity_to_slider
        from away.hands.autotune import AUTOTUNE_SPECS

        tk, ttk = self.tk, self.ttk
        f = self._scroll_tab(nb, "Tuning", 6)
        f.columnconfigure(0, weight=1)
        scale = 1.0
        with contextlib.suppress(Exception):
            scale = ui_scale(float(self.root.winfo_fpixels("1i")))
        self._tune_scale = scale
        self.tuning = TuningController(self.m)
        self.scales: dict[str, tuple[Any, Any]] = {}
        self.tvars: dict[str, Any] = {}
        self._tune_rows: dict[str, tuple[Any, Any, Any]] = {}  # path -> (value label, reset button, Tunable)
        self._tune_sections: dict[str, tuple[Any, dict[str, bool], Any]] = {}
        self._tune_sync_pending = False

        bar = ttk.Frame(f)
        bar.grid(row=0, column=0, sticky="we")
        ttk.Label(bar, text="Preset:").pack(side="left")
        for name in PRESETS:
            ttk.Button(bar, text=name, command=lambda n=name: self._tune_preset(n)).pack(side="left", padx=2)
        ttk.Label(bar, text="Sensitivity:").pack(side="left", padx=(12, 2))
        self.sens_var = tk.DoubleVar(value=sensitivity_to_slider(self.tuning.sensitivity()))
        tk.Scale(
            bar,
            from_=0,
            to=100,
            resolution=1,
            orient="horizontal",
            length=int(150 * scale),
            showvalue=False,
            variable=self.sens_var,
            command=lambda v: self._tune_sensitivity(float(v)),
        ).pack(side="left")
        self.sens_note = tk.StringVar()
        ttk.Label(bar, textvariable=self.sens_note, width=12).pack(side="left", padx=4)
        bar2 = ttk.Frame(f)
        bar2.grid(row=1, column=0, sticky="we", pady=(2, 0))
        self.undo_btn = ttk.Button(bar2, text="Undo", command=self._tune_undo, state="disabled")
        self.undo_btn.pack(side="left")
        ttk.Button(bar2, text="Save as profile...", command=self._tune_save_profile).pack(side="left", padx=4)
        self.preset_note = tk.StringVar()
        ttk.Label(bar2, textvariable=self.preset_note, foreground="#555").pack(side="left", padx=6)

        # ---- live view
        live = self._tune_section(f, 2, "Live view: why did a gesture (not) fire?", False)
        self._live_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._live_cmd: queue.Queue[Any] = queue.Queue()
        self._live_stop = threading.Event()
        self._live_thread: threading.Thread | None = None
        self.live_canvas = tk.Canvas(
            live, width=int(200 * scale), height=int(170 * scale), background="#111", highlightthickness=0
        )
        self.live_canvas.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 8))
        self.gauge_canvas = tk.Canvas(
            live, width=int(380 * scale), height=int(170 * scale), background="#fafafa", highlightthickness=0
        )
        self.gauge_canvas.grid(row=0, column=1, sticky="nw")
        lrow = ttk.Frame(live)
        lrow.grid(row=1, column=1, sticky="w", pady=4)
        self.live_btn = ttk.Button(lrow, text="Start live view", command=self._live_toggle)
        self.live_btn.pack(side="left")
        self.live_status = tk.StringVar(value="Shows your hand and the measurements each gesture compares.")
        ttk.Label(lrow, textvariable=self.live_status, foreground="#555").pack(side="left", padx=6)
        self._live_draw(None)

        # ---- auto-tune
        auto = self._tune_section(f, 3, "Auto-tune a gesture", False)
        ttk.Label(
            auto,
            text="Do the gesture 5 times, then something else 5 times; the threshold that best tells them apart "
            "is suggested. The recordings are saved for `away calibrate`.",
            foreground="#555",
            wraplength=int(560 * scale),
            justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        self.at_var = tk.StringVar(value=next(iter(AUTOTUNE_SPECS.values())).title)
        ttk.Combobox(
            auto,
            textvariable=self.at_var,
            values=[s.title for s in AUTOTUNE_SPECS.values()],
            state="readonly",
            width=32,
        ).grid(row=1, column=0, sticky="w", pady=4)
        self.at_start_btn = ttk.Button(auto, text="Start", command=self._at_start)
        self.at_start_btn.grid(row=1, column=1, padx=4)
        self.at_apply_btn = ttk.Button(auto, text="Apply", command=self._at_apply, state="disabled")
        self.at_apply_btn.grid(row=1, column=2, padx=2)
        ttk.Button(auto, text="Cancel", command=self._at_cancel).grid(row=1, column=3, padx=2)
        self.at_prompt = tk.StringVar(value="")
        tk.Label(
            auto,
            textvariable=self.at_prompt,
            font=("TkDefaultFont", 12, "bold"),
            fg="#1b5e9f",
            wraplength=int(560 * scale),
            justify="left",
        ).grid(row=2, column=0, columnspan=4, sticky="w")
        self.at_result = tk.StringVar(value="")
        ttk.Label(auto, textvariable=self.at_result, wraplength=int(560 * scale), justify="left").grid(
            row=3, column=0, columnspan=4, sticky="w"
        )
        self._at_session: Any = None
        self._at_res: Any = None
        self._at_meta: tuple[str, float] = ("Right", 30.0)

        # ---- one section per area
        row = 4
        switches_done = False
        for group, items in self.tuning.groups():
            body = self._tune_section(
                f, row, group, group in ("Cursor", "Clicks"), reset=lambda g=group: self._tune_reset_group(g)
            )
            body.columnconfigure(1, weight=1)
            r = 0
            for t in items:
                self._tune_slider(body, r, t)
                r += 2
            if group == "Scroll":
                self._tune_switches(body, r)
                switches_done = True
            row += 1
        if not switches_done:
            self._tune_switches(self._tune_section(f, row, "Palm scroll switches", False), 0)
        self._tune_sync()

    def _tune_section(self, parent: Any, row: int, title: str, open_: bool, reset: Any = None) -> Any:
        """A collapsible section: a header button (triangle + title) and an optional "Reset section" button."""
        ttk = self.ttk
        box = ttk.Frame(parent)
        box.grid(row=row, column=0, sticky="we", pady=(6, 0))
        head = ttk.Frame(box)
        head.pack(fill="x")
        body = ttk.Frame(box, padding=(16, 2, 2, 4))
        state = {"open": open_}
        btn = ttk.Button(head, text=(OPEN_MARK if open_ else CLOSED_MARK) + title)

        def toggle() -> None:
            state["open"] = not state["open"]
            if state["open"]:
                body.pack(fill="x")
            else:
                body.pack_forget()
            btn.configure(text=(OPEN_MARK if state["open"] else CLOSED_MARK) + title)

        btn.configure(command=toggle)
        btn.pack(side="left")
        if reset is not None:
            ttk.Button(head, text="Reset section", command=reset).pack(side="right")
        ttk.Separator(box, orient="horizontal").pack(fill="x", pady=(2, 0))
        if open_:
            body.pack(fill="x")
        self._tune_sections[title] = (body, state, btn)
        return body

    def _tune_slider(self, body: Any, r: int, t: Any) -> None:
        """One slider row: plain title, slider, value with units, reset; the help line under it."""
        tk, ttk = self.tk, self.ttk
        s = self._tune_scale
        ttk.Label(body, text=t.title).grid(row=r, column=0, sticky="w", padx=(0, 6))
        var = tk.DoubleVar(value=t.value)
        sc = tk.Scale(
            body,
            from_=t.lo,
            to=t.hi,
            resolution=t.resolution,
            orient="horizontal",
            length=int(240 * s),
            variable=var,
            showvalue=False,
            command=lambda v, p=t.path: self._on_scale(p, float(v)),
        )
        sc.grid(row=r, column=1, sticky="we")
        val = ttk.Label(body, width=16, anchor="w")
        val.grid(row=r, column=2, sticky="w", padx=4)
        rb = ttk.Button(body, text=RESET_MARK, width=3, command=lambda p=t.path: self._tune_reset(p))
        rb.grid(row=r, column=3)
        ttk.Label(body, text=t.help, foreground="#666", wraplength=int(560 * s), justify="left").grid(
            row=r + 1, column=0, columnspan=4, sticky="w", pady=(0, 4)
        )
        self.scales[t.path] = (var, sc)
        self._tune_rows[t.path] = (val, rb, t)

    def _tune_switches(self, body: Any, r: int) -> None:
        tk, ttk = self.tk, self.ttk
        for path, label in PALM_SCROLL_SWITCHES:
            bvar = tk.BooleanVar(value=bool(self.m.voice_value(path)))
            ttk.Checkbutton(
                body,
                text=label,
                variable=bvar,
                command=lambda p=path, v=bvar: self._report(self.m.set_voice(p, v.get())),
            ).grid(row=r, column=0, columnspan=4, sticky="w")
            self.tvars[path] = bvar
            r += 1

    def _on_scale(self, path: str, value: float) -> None:
        self._tune_schedule_sync()
        cur = float(self.m.voice_value(path))
        step = {t.path: t.resolution for t in self.m.tunables()}[path]
        if abs(value - cur) >= step / 2:  # ignore the callback Tk fires when refresh() sets the variable
            tuning = getattr(self, "tuning", None)
            self._report(tuning.set(path, value) if tuning is not None else self.m.set_tunable(path, value))

    def _tune_schedule_sync(self) -> None:
        if not getattr(self, "_tune_sync_pending", True):
            self._tune_sync_pending = True
            self.root.after_idle(self._tune_sync)

    def _tune_sync(self) -> None:
        """Sliders, value labels, reset buttons, the master slider, the preset note and Undo from the model."""
        from away.app.tunables import PRESETS, format_value, sensitivity_to_slider, short_label

        self._tune_sync_pending = False
        tc = self.tuning
        for path, (val, rb, t) in self._tune_rows.items():
            v = tc.value(path)
            var = self.scales[path][0]
            if abs(float(var.get()) - v) > 1e-12:
                var.set(v)
            val.configure(text=format_value(v, t.unit, t.is_int, t.resolution))
            rb.configure(state="disabled" if tc.is_default(path) else "normal")
        pos = round(sensitivity_to_slider(tc.sensitivity()))
        if abs(float(self.sens_var.get()) - pos) >= 0.5:
            self.sens_var.set(pos)
        note = "default" if pos == 50 else ("faster " if pos > 50 else "slower ") + f"{abs(pos - 50) * 2}%"
        self.sens_note.set(note if tc.sensitivity_matches() else "mixed")
        preset = tc.preset()
        self.preset_note.set(
            f"Preset: {preset} ({PRESETS[preset].blurb})" if preset else "Preset: custom (your own adjustments)"
        )
        top = tc.undo_stack.peek()
        self.undo_btn.configure(
            state="normal" if top else "disabled", text=f"Undo {short_label(top.label)}" if top else "Undo"
        )

    def _tune_done(self, err: str | None, ok: str | None = None) -> None:
        self._report(err, ok)
        self._tune_sync()

    def _tune_preset(self, name: str) -> None:
        from away.app.tunables import PRESETS

        self._tune_done(self.tuning.apply_preset(name), f"preset {name}: {PRESETS[name].blurb}")

    def _tune_sensitivity(self, pos: float) -> None:
        from away.app.tunables import sensitivity_to_slider, slider_to_sensitivity

        if abs(pos - round(sensitivity_to_slider(self.tuning.sensitivity()))) < 0.5:
            return  # the callback for our own update
        self._report(self.tuning.set_sensitivity(slider_to_sensitivity(pos)))
        self._tune_schedule_sync()

    def _tune_reset(self, path: str) -> None:
        self._tune_done(self.tuning.reset(path))

    def _tune_reset_group(self, group: str) -> None:
        self._tune_done(self.tuning.reset_group(group), f"{group} back to the defaults")

    def _tune_undo(self) -> None:
        err, ok = self.tuning.undo()
        self._tune_done(err, ok)

    def _tune_save_profile(self) -> None:
        from tkinter import simpledialog

        cur = self.m.profile if self.m.profile != "default" else ""
        name = simpledialog.askstring(
            "Save tuning as profile",
            "Per-app profile name (existing or new). Its hand and face tuning is used while that app is focused:",
            initialvalue=cur,
            parent=self.root,
        )
        if not name:
            return
        apps = ""
        if name.strip() not in self.m.bindings.profiles:
            apps = (
                simpledialog.askstring("Apps", "Apps for the new profile (e.g. chrome, firefox.exe):", parent=self.root)
                or ""
            )
        err, ok = self.tuning.save_as_profile(name, apps)
        if err is None:
            self.refresh()
        self._tune_done(err, ok)

    # ---- live view
    def _live_running(self) -> bool:
        return self._live_thread is not None and self._live_thread.is_alive()

    def _live_toggle(self) -> None:
        if self._live_running():
            self._live_stop.set()
            self.live_btn.configure(text="Start live view")
            return
        self._live_start()

    def _live_start(self) -> bool:
        from away.app.tuning_live import tuning_camera_worker

        if self._live_running():
            return True
        ok, msg = camera_available()
        if not ok:
            self.live_status.set(msg)
            return False
        test = getattr(self, "_test_thread", None)
        if test is not None and test.is_alive():
            self.live_status.set("turn the camera off in the Gesture Studio tab first")
            return False
        self._live_stop.clear()
        self._live_q = queue.Queue()
        self._live_cmd = queue.Queue()
        self._live_thread = threading.Thread(
            target=tuning_camera_worker, args=(self.m, self._live_q, self._live_stop, self._live_cmd), daemon=True
        )
        self._live_thread.start()
        self.live_btn.configure(text="Stop live view")
        self.root.after(100, self._live_poll)
        return True

    def _live_poll(self) -> None:
        latest = None
        while True:
            try:
                msg = self._live_q.get_nowait()
            except queue.Empty:
                break
            kind = msg.get("type")
            if kind == "frame":
                latest = msg
            elif kind == "status":
                self.live_status.set(str(msg.get("text", "")))
            elif kind in ("countdown", "recording", "trial"):
                self._at_message(msg)
            elif kind == "stopped":
                self.live_btn.configure(text="Start live view")
                if self._at_session is not None:
                    self._at_session = None
                    self.at_prompt.set("auto-tune stopped: the live view ended")
        if latest is not None:
            self._live_draw(latest)
        if self._live_running() or not self._live_q.empty():
            self.root.after(66, self._live_poll)

    def _live_draw(self, frame: dict[str, Any] | None) -> None:
        from away.hands.autotune import HAND_CONNECTIONS, skeleton_points

        c, g = self.live_canvas, self.gauge_canvas
        c.delete("all")
        g.delete("all")
        cw, ch = int(c.cget("width")), int(c.cget("height"))
        hands = [frame.get("lm"), frame.get("helper")] if frame else []
        if not frame or (frame.get("lm") is None and frame.get("helper") is None):
            c.create_text(cw // 2, ch // 2, text="no hand", fill="#888")
        else:
            present = [h for h in hands if h is not None]
            colours = ["#4fc3f7" if frame.get("lm") is not None else "#aaa", "#aaa"]
            for pts, colour in zip(skeleton_points(present, cw, ch), colours):
                for a, b in HAND_CONNECTIONS:
                    c.create_line(*pts[a], *pts[b], fill=colour, width=2)
                for x, y in pts:
                    c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="white", outline="")
        gauges = frame.get("gauges", []) if frame else []
        gw = int(g.cget("width"))
        row_h = max(30, int(g.cget("height")) // 4)
        bar_x0, bar_x1 = 8, gw - 8
        for i, gauge in enumerate(gauges):
            y = i * row_h + 4
            g.create_text(bar_x0, y, text=gauge.text(), anchor="nw", fill="#1b7f3b" if gauge.active else "#333")
            by = y + 16
            g.create_rectangle(bar_x0, by, bar_x1, by + 8, fill="#e3e3e3", outline="")
            frac = gauge.fraction
            if frac is not None:
                fill = "#43a047" if gauge.active else "#90a4ae"
                g.create_rectangle(bar_x0, by, bar_x0 + frac * (bar_x1 - bar_x0), by + 8, fill=fill, outline="")
            tx = bar_x0 + gauge.threshold_fraction * (bar_x1 - bar_x0)
            g.create_line(tx, by - 3, tx, by + 11, fill="#d32f2f", width=2)
        if not gauges:
            g.create_text(8, 8, anchor="nw", fill="#666", text="Start the live view to see the measurements.")

    # ---- auto-tune
    def _at_start(self) -> None:
        from away.hands.autotune import AUTOTUNE_SPECS, AutoTuneSession

        spec = next((s for s in AUTOTUNE_SPECS.values() if s.title == self.at_var.get()), None)
        if spec is None:
            return
        if not self._live_start():
            self.at_prompt.set(f"the live view could not start: {self.live_status.get()}")
            return
        self._at_session = AutoTuneSession(spec)
        self._at_res = None
        self.at_apply_btn.configure(state="disabled")
        self.at_result.set("")
        self._at_next(first=True)

    def _at_next(self, first: bool = False) -> None:
        sess = self._at_session
        if sess is None:
            return
        step = sess.next_step()
        if step is None:
            try:
                self._at_res = sess.result(self.tuning.values(), self.tuning.bounds)
            except ValueError as exc:
                self.at_prompt.set(str(exc))
                return
            self.at_prompt.set("Done. Apply the suggestion?")
            self.at_result.set(self._at_res.summary())
            self.at_apply_btn.configure(state="normal")
            return
        self.at_prompt.set(sess.prompt())
        long_wait = first or step == ("negative", 1)
        self._live_cmd.put(
            {"cmd": "record", "tag": (sess.session_id, step), "seconds": 2.0, "countdown": 3.0 if long_wait else 1.0}
        )

    def _at_message(self, msg: dict[str, Any]) -> None:
        sess = self._at_session
        if sess is None or not isinstance(msg.get("tag"), tuple) or msg["tag"][0] != sess.session_id:
            return
        if msg["type"] == "countdown":
            self.at_prompt.set(f"{sess.prompt()}  ...starting in {msg['left']}")
        elif msg["type"] == "recording":
            self.at_prompt.set(f"{sess.prompt()}  " + RECORDING_MARK + " recording")
        elif msg["type"] == "trial":
            self._at_meta = (str(msg.get("handedness", "Right")), float(msg.get("fps", 30.0)) or 30.0)
            err = sess.add_trial(msg.get("frames", []))
            if err:
                self.at_result.set(err)
            self._at_next()

    def _at_apply(self) -> None:
        from away import paths

        sess, res = self._at_session, self._at_res
        if sess is None or res is None:
            return
        err = self.tuning.apply(res.after, label=f"auto-tune {res.spec.title}")
        if err:
            self._tune_done(err)
            return
        saved: list[Path] = []
        note = ""
        try:
            saved = sess.save_clips(paths.recordings_dir(), *self._at_meta)
        except (OSError, ValueError) as exc:
            note = f" (recordings not saved: {exc})"
        self._at_session = self._at_res = None
        self.at_apply_btn.configure(state="disabled")
        self.at_prompt.set("")
        self._tune_done(None, f"{res.summary()}; saved {len(saved)} clips for `away calibrate`{note}. Undo reverts it.")

    def _at_cancel(self) -> None:
        if self._at_session is not None:
            self._live_cmd.put({"cmd": "cancel"})
        self._at_session = self._at_res = None
        self.at_apply_btn.configure(state="disabled")
        self.at_prompt.set("")
        self.at_result.set("")

    def _build_voice(self, nb: Any) -> None:
        tk, ttk = self.tk, self.ttk
        f = self._scroll_tab(nb, "Voice & input")
        self.vvars: dict[str, Any] = {}
        r = 0
        ttk.Label(f, text="Hand roles (which hand is the pointer)").grid(row=r, column=0, sticky="w", pady=3)
        self.roles_var = tk.StringVar()
        rcb = ttk.Combobox(
            f, textvariable=self.roles_var, values=[ROLE_LABELS[m] for m in ROLE_CHOICES], state="readonly", width=16
        )
        rcb.grid(row=r, column=1, sticky="w")
        rcb.bind("<<ComboboxSelected>>", lambda _e: self._report(self.m.set_role_label(self.roles_var.get())))
        r += 1
        for path, label in (
            ("hands.cursor.curve", "Cursor acceleration curve"),
            ("voice.mode", "Input mode"),
            ("voice.intent", "Always-on intent"),
            ("voice.ambient_fallback", "No wake word + no camera"),
            ("updates.channel", "Update channel"),
        ):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3)
            var = tk.StringVar()
            cb = ttk.Combobox(f, textvariable=var, values=list(VOICE_FIELDS[path]), state="readonly", width=16)
            cb.grid(row=r, column=1, sticky="w")
            cb.bind("<<ComboboxSelected>>", lambda _e, p=path, v=var: self._report(self.m.set_voice(p, v.get())))
            self.vvars[path] = var
            r += 1
        for path, label in (
            ("voice.wake_word_enabled", "Require wake word"),
            ("face.gate_dictation", "Face gate: only type when I face the camera and speak (off = always type)"),
            ("voice.commands_enabled", "Voice commands"),
            ("app.always_on", "Active as soon as the app starts (gestures + voice)"),
            ("app.start_minimized", "Start minimized to tray"),
            ("face.brow_raise_toggle", "Brow-raise trigger"),
            ("face.mouth_open_trigger", "Mouth-open trigger"),
            ("face.wink_trigger", "Wink triggers (left / right)"),
            ("hands.dwell.enabled", "Dwell click (rest the cursor)"),
            ("hands.gestures.autoscroll_enabled", "Four-finger auto-scroll (thumb folded; a fist stops it)"),
            ("hands.gestures.autoscroll_horizontal", "Auto-scroll sideways too (hand left / right)"),
            ("hands.cursor.click_freeze", "Steady click (freeze while pinching)"),
            ("app.auto_pause", "Pause while I use the mouse / keyboard"),
            ("voice.numbers_as_digits", "Spoken numbers as digits"),
            ("voice.clap_enabled", "Clap: single = activate, double = pause"),
            ("voice.clap_require_hands", "Clap: only when the camera sees both hands together"),
            ("cursor_overlay", "Ring at the cursor (next start)"),
            ("updates.check", "Check for updates (GitHub releases, daily; nothing is sent about you)"),
            ("updates.auto_install", "Install updates automatically when idle"),
        ):
            var = tk.BooleanVar()
            ttk.Checkbutton(
                f, text=label, variable=var, command=lambda p=path, v=var: self._report(self.m.set_voice(p, v.get()))
            ).grid(row=r, column=0, columnspan=2, sticky="w")
            self.vvars[path] = var
            r += 1
        for path, label, cap in (
            ("voice.wake_word", "Wake word", False),
            ("voice.hotkey", "Push-to-talk / toggle hotkey", True),
            ("voice.mute_hotkey", "Mute hotkey", True),
            ("voice.hold_key", "Hold-to-talk key", True),
            ("app.toggle_hotkey", "Gestures on/off hotkey", True),
            ("voice.edit_hotkey", "Edit selected text hotkey", True),
        ):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3)
            var = tk.StringVar()
            e = ttk.Entry(f, textvariable=var, width=24)
            e.grid(row=r, column=1, sticky="w")
            e.bind("<FocusOut>", lambda _e, p=path, v=var: self._report(self.m.set_voice(p, v.get())))
            e.bind("<Return>", lambda _e, p=path, v=var: self._report(self.m.set_voice(p, v.get())))
            if cap:
                ttk.Button(f, text="Capture", command=lambda v=var, p=path: self._start_capture(v, p)).grid(
                    row=r, column=2
                )
            self.vvars[path] = var
            r += 1
        cspec = {t.path: t for t in self.m.tunables()}["hands.cursor.big_cursor_size"]
        ttk.Label(f, text=cspec.label).grid(row=r, column=0, sticky="w", pady=3)
        cvar = tk.DoubleVar(value=cspec.value)
        tk.Scale(
            f,
            from_=cspec.lo,
            to=cspec.hi,
            resolution=1,
            orient="horizontal",
            length=240,
            variable=cvar,
            command=lambda v: self._on_scale("hands.cursor.big_cursor_size", float(v)),
        ).grid(row=r, column=1, columnspan=2, sticky="w")
        self.scales["hands.cursor.big_cursor_size#voice"] = (cvar, None)
        r += 1
        for cpath in ("voice.clap_sensitivity", "voice.clap_double_window"):
            cs = {t.path: t for t in self.m.tunables()}[cpath]
            ttk.Label(f, text=cs.label).grid(row=r, column=0, sticky="w", pady=3)
            csv = tk.DoubleVar(value=cs.value)
            tk.Scale(
                f,
                from_=cs.lo,
                to=cs.hi,
                resolution=cs.resolution,
                orient="horizontal",
                length=240,
                variable=csv,
                command=lambda val, p=cpath: self._on_scale(p, float(val)),
            ).grid(row=r, column=1, columnspan=2, sticky="w")
            self.scales[f"{cpath}#voice"] = (csv, None)
            r += 1
        spec = {t.path: t for t in self.m.tunables()}["voice.vad_sensitivity"]
        ttk.Label(f, text="Voice sensitivity (always-on)").grid(row=r, column=0, sticky="w", pady=3)
        svar = tk.DoubleVar(value=spec.value)
        tk.Scale(
            f,
            from_=spec.lo,
            to=spec.hi,
            resolution=spec.resolution,
            orient="horizontal",
            length=240,
            variable=svar,
            command=lambda v: self._on_scale("voice.vad_sensitivity", float(v)),
        ).grid(row=r, column=1, columnspan=2, sticky="w")
        self.scales["voice.vad_sensitivity#voice"] = (svar, None)
        r += 1
        ttk.Label(f, text="More timing sliders are on the Tuning tab.", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=6
        )

    def _build_cameras(self, nb: Any) -> None:
        tk, ttk = self.tk, self.ttk
        f = self._scroll_tab(nb, "Cameras")
        self.cam_mode = tk.StringVar()
        ttk.Label(f, textvariable=self.cam_mode, font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=6, sticky="w"
        )
        cols = ("role", "device", "resolution", "fps", "fov", "confidence")
        self.cam_tree = ttk.Treeview(f, columns=cols, show="headings", height=5)
        for c, w in zip(cols, (90, 220, 110, 50, 60, 140)):
            self.cam_tree.heading(c, text=c.title())
            self.cam_tree.column(c, width=w, anchor="w")
        self.cam_tree.grid(row=1, column=0, columnspan=6, sticky="nsew", pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Device:").grid(row=2, column=0, sticky="w")
        self.cam_dev = tk.StringVar()
        self.cam_dev_cb = ttk.Combobox(f, textvariable=self.cam_dev, width=34, state="readonly")
        self.cam_dev_cb.grid(row=2, column=1, sticky="we", padx=4)
        ttk.Button(f, text="Refresh list", command=self._cam_refresh_devices).grid(row=2, column=2)
        ttk.Label(f, text="Role:").grid(row=3, column=0, sticky="w")
        self.cam_role = tk.StringVar(value=self.m.role_default())
        ttk.Combobox(f, textvariable=self.cam_role, values=["primary", "secondary"], state="readonly", width=12).grid(
            row=3, column=1, sticky="w", padx=4
        )
        ttk.Label(f, text="Resolution / fps:").grid(row=4, column=0, sticky="w")
        self.cam_res = tk.StringVar(value="640x480")
        self.cam_fps = tk.StringVar(value="30")
        ttk.Entry(f, textvariable=self.cam_res, width=12).grid(row=4, column=1, sticky="w", padx=4)
        ttk.Entry(f, textvariable=self.cam_fps, width=6).grid(row=4, column=2, sticky="w")
        btns = ttk.Frame(f)
        btns.grid(row=5, column=0, columnspan=6, sticky="w", pady=6)
        ttk.Button(btns, text="Add camera", command=self._cam_add).pack(side="left", padx=2)
        ttk.Button(btns, text="Apply to selected", command=self._cam_apply).pack(side="left", padx=2)
        ttk.Button(btns, text="Remove selected", command=self._cam_remove).pack(side="left", padx=2)
        self.cam_calib_btn = ttk.Button(btns, text="Calibrate (wave hand ~20 s)", command=self._cam_calibrate)
        self.cam_calib_btn.pack(side="left", padx=8)
        self.cam_live_btn = ttk.Button(btns, text="Live confidence", command=self._cam_live)
        self.cam_live_btn.pack(side="left", padx=2)
        self.cam_test_btn = ttk.Button(btns, text="Test both cameras", command=self._cam_test)
        self.cam_test_btn.pack(side="left", padx=2)
        self.cam_save_combo_btn = ttk.Button(btns, text="Save working combination", command=self._cam_save_combo)
        self.cam_save_combo_btn.pack(side="left", padx=2)
        self.cam_save_combo_btn.configure(state="disabled")
        self.cam_multi_hint = tk.StringVar()
        ttk.Label(f, textvariable=self.cam_multi_hint, foreground="#555").grid(
            row=9, column=0, columnspan=6, sticky="w", pady=(2, 0)
        )
        self.cam_tree.bind("<<TreeviewSelect>>", lambda _e: self._cam_on_select())
        ttk.Label(f, text="Fusion:").grid(row=6, column=0, sticky="w")
        self.cam_fusion = tk.StringVar(value=self.m.cfg.hands.multicam.fusion)
        cb = ttk.Combobox(f, textvariable=self.cam_fusion, values=["weighted", "best_view"], state="readonly", width=12)
        cb.grid(row=6, column=1, sticky="w", padx=4)
        cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._report(self.m.set_voice("hands.multicam.fusion", self.cam_fusion.get())),
        )
        # v0.1.19: the Windows capture backend (auto: DirectShow, Media Foundation, then WinRT MediaCapture)
        ttk.Label(f, text="Capture backend:").grid(row=6, column=2, sticky="e", padx=(12, 0))
        self.cam_backend = tk.StringVar(value=self.m.camera_backend())
        bcb = ttk.Combobox(f, textvariable=self.cam_backend, values=list(CAMERA_BACKENDS), state="readonly", width=8)
        bcb.grid(row=6, column=3, sticky="w", padx=4)
        bcb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._report(
                self.m.set_voice("hands.camera_backend", self.cam_backend.get()),
                "camera backend saved on Save; applies the next time the cameras open",
            ),
        )
        self.cam_calib = tk.StringVar()
        ttk.Label(f, textvariable=self.cam_calib, foreground="#555").grid(
            row=7, column=0, columnspan=6, sticky="w", pady=4
        )
        ttk.Label(
            f,
            text="Save first; calibration and live confidence use the saved config. "
            "The primary camera is the cursor's frame of reference.",
            foreground="#555",
        ).grid(row=8, column=0, columnspan=6, sticky="w")
        # v0.1.19: the app's camera retry state, Diagnose / Retry cameras, and the test / diagnose output in a
        # read-only, word-wrapped, scrollable text box (long lines were cut off in a label)
        self.cam_retry_var = tk.StringVar()
        ttk.Label(f, textvariable=self.cam_retry_var, foreground="#a33", wraplength=640, justify="left").grid(
            row=10, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
        tools = ttk.Frame(f)
        tools.grid(row=11, column=0, columnspan=6, sticky="w", pady=(4, 0))
        self.cam_diag_btn = ttk.Button(tools, text="Diagnose", command=self._cam_diagnose)
        self.cam_diag_btn.pack(side="left", padx=2)
        ttk.Button(tools, text="Retry cameras", command=self._cam_retry).pack(side="left", padx=2)
        outf = ttk.Frame(f)
        outf.grid(row=12, column=0, columnspan=6, sticky="nsew", pady=(4, 0))
        f.rowconfigure(12, weight=1)
        self.cam_out = tk.Text(outf, height=12, width=96, wrap="word", relief="flat", background="#f6f6f6")
        osb = ttk.Scrollbar(outf, orient="vertical", command=self.cam_out.yview)
        self.cam_out.configure(yscrollcommand=osb.set, state="disabled")
        osb.pack(side="right", fill="y")
        self.cam_out.pack(side="left", fill="both", expand=True)
        self._cam_diag_q: queue.Queue[Any] = queue.Queue()
        self._cam_diag_thread: threading.Thread | None = None
        self._cam_q: queue.Queue[Any] = queue.Queue()
        self._cam_stop = threading.Event()
        self._cam_thread: threading.Thread | None = None
        self._cam_cells: list[str] = []
        self._cam_tools = CameraTools()
        from away.app.camera_share import SharedLease

        self._cam_leases = SharedLease()
        self._cam_test_q: queue.Queue[Any] = queue.Queue()
        self._cam_test_thread: threading.Thread | None = None
        self._cam_combo: list[Any] | None = None
        self._cam_refresh_devices()

    def _cam_refresh_devices(self) -> None:
        choices = self.m.device_choices()
        self.cam_dev_cb["values"] = choices
        self._cam_refresh()  # the Device column shows the names from the fresh list
        if not self.cam_dev.get() or self.cam_dev.get() not in choices:
            self.cam_dev.set(self.m.device_choice(self._cam_selected(), choices))

    def _cam_on_select(self) -> None:
        i = self._cam_selected()
        self.cam_role.set(self.m.role_default(i))
        choice = self.m.device_choice(i, list(self.cam_dev_cb["values"] or []))
        if choice:
            self.cam_dev.set(choice)

    def _cam_out_set(self, text: str) -> None:
        """Replace the test / diagnose output (read-only text box)."""
        self.cam_out.configure(state="normal")
        self.cam_out.delete("1.0", "end")
        self.cam_out.insert("end", text)
        self.cam_out.configure(state="disabled")

    def _cam_out_add(self, line: str) -> None:
        self.cam_out.configure(state="normal")
        self.cam_out.insert("end", ("\n" if self.cam_out.get("1.0", "end-1c") else "") + line)
        self.cam_out.see("end")
        self.cam_out.configure(state="disabled")

    def _cam_retry(self) -> None:
        err = self.m.request_camera_retry()
        self._report(err, "asked Project Away to retry the cameras now")
        self.cam_retry_var.set("")

    def _cam_diagnose(self) -> None:
        err = self._cam_tools.begin("test")  # it opens every camera: never at the same time as the other tools
        if err:
            self._report(err)
            return
        self._cam_out_set("diagnosing: every camera on every backend; this takes a while...")
        self.cam_diag_btn.configure(state="disabled")
        self._cam_diag_thread = threading.Thread(
            target=camera_diagnose_worker,
            args=(self.m, self._cam_diag_q),
            kwargs={"leases": self._cam_leases},
            daemon=True,
            name="camera-diagnose",
        )
        self._cam_diag_thread.start()
        self.root.after(200, self._cam_diag_poll)

    def _cam_diag_poll(self) -> None:
        running = self._cam_diag_thread is not None and self._cam_diag_thread.is_alive()
        while not self._cam_diag_q.empty():
            msg = self._cam_diag_q.get_nowait()
            if isinstance(msg, str):
                self._cam_out_add(msg)
            elif isinstance(msg, dict) and "error" in msg:
                self._cam_out_add(f"error: {msg['error']}")
        if running:
            self.root.after(200, self._cam_diag_poll)
            return
        self._cam_tools.end("test")
        self.cam_diag_btn.configure(state="normal")
        self._cam_refresh()

    def _cam_selected(self) -> int | None:
        sel = self.cam_tree.selection()
        return self.cam_tree.index(sel[0]) if sel else None

    def _cam_refresh(self) -> None:
        self.cam_mode.set(self.m.camera_mode())
        self.cam_tree.delete(*self.cam_tree.get_children())
        for i, r in enumerate(self.m.camera_rows()):
            bar = self._cam_cells[i] if i < len(self._cam_cells) else "-"
            dev = str(r.get("label") or r["device"]) + (" (in use by Project Away)" if r.get("in_use") else "")
            role = r["role"] + (" (default)" if r.get("implicit") else "")
            self.cam_tree.insert(
                "", "end", values=(role, dev, f"{r['width']}x{r['height']}", r["fps"], r["hfov_deg"], bar)
            )
        self.cam_calib.set(self.m.calibration_status())
        ok, hint = self.m.multi_tools()
        st = self._cam_tools.states(ok)
        self.cam_calib_btn.configure(state=st["calibrate"])
        self.cam_live_btn.configure(state=st["live"])
        self.cam_test_btn.configure(state=st["test"] if self.m.cfg.hands.cameras else "disabled")
        self.cam_multi_hint.set(hint)
        self.cam_retry_var.set(self.m.retry_status_text())
        rows = self.cam_tree.get_children()
        if self._cam_selected() is None and len(rows) == 1:
            self.cam_tree.selection_set(rows[0])  # the only camera: editing it needs no extra click
        if self._cam_selected() is None:
            self.cam_role.set(self.m.role_default())

    def _cam_add(self) -> None:
        if not self.cam_dev.get():
            self._report("pick a device first")
            return
        self._report(self.m.add_camera(self.m.parse_device(self.cam_dev.get()), self.cam_role.get()))
        self._cam_refresh()

    def _cam_apply(self) -> None:
        i = self._cam_selected()
        if i is None:
            self._report("select a camera row first")
            return
        w, _, h = self.cam_res.get().lower().partition("x")
        for field, val in (("width", w), ("height", h), ("fps", self.cam_fps.get()), ("role", self.cam_role.get())):
            err = self.m.set_camera(i, field, val)
            if err:
                self._report(err)
                break
        if self.cam_dev.get():
            self._report(self.m.set_camera(i, "device", self.cam_dev.get()))
        self._cam_refresh()

    def _cam_remove(self) -> None:
        i = self._cam_selected()
        if i is not None:
            self._report(self.m.remove_camera(i))
            self._cam_refresh()

    def _cam_calibrate(self) -> None:
        import subprocess

        ok, hint = self.m.multi_tools()
        if not ok:
            self._report(hint)
            return
        err = self._cam_tools.begin("calibrate")  # Live confidence is stopped below; the test blocks this
        if err:
            self._report(err)
            return
        if self.m.dirty:
            self._on_save()
        status_file = self.m.calibration_status_file()
        with contextlib.suppress(OSError):
            status_file.unlink()
        live = self._cam_thread
        self._report(None, "calibrating: move your open hand through both cameras' view for ~20 s...")
        self.cam_live_btn.configure(text="Live confidence")
        self._cam_refresh()
        result: dict[str, Any] = {}

        def spawn() -> dict[str, Any]:
            try:
                r = subprocess.run(
                    self.m.calibration_command(status_file),
                    capture_output=True,
                    text=True,
                    env={**os.environ, "AWAY_NO_CONSOLE": "1"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                # keep the full output next to the status file for troubleshooting (GUI exe: no console)
                with contextlib.suppress(OSError):
                    status_file.with_name("calibration.log").write_text(
                        f"$ {' '.join(self.m.calibration_command(status_file))}\nexit {r.returncode}\n"
                        f"--- stdout ---\n{r.stdout or ''}\n--- stderr ---\n{r.stderr or ''}",
                        encoding="utf-8",
                    )
                return {"rc": r.returncode, "err": r.stderr or ""}
            except Exception as exc:
                return {"rc": -1, "err": str(exc)}

        def run() -> None:
            result.update(calibration_flow(self._cam_leases, lambda: stop_live_view(self._cam_stop, live), spawn))

        th = threading.Thread(target=run, daemon=True, name="calibrate")
        th.start()

        def poll() -> None:
            st = read_calibration_status(status_file)
            if th.is_alive():
                if st and st.get("message"):
                    self._report(None, str(st["message"]))
                self.root.after(500, poll)
                return
            self._cam_tools.end("calibrate")
            self._report(*calibration_outcome(st, result.get("rc", -1), str(result.get("err", ""))))
            self._cam_refresh()

        self.root.after(500, poll)

    def _cam_live(self) -> None:
        if self._cam_thread is not None and self._cam_thread.is_alive():
            self._cam_stop.set()
            self.cam_live_btn.configure(text="Live confidence")
            return
        ok, hint = self.m.multi_tools()
        if not ok:
            self._report(hint)
            return
        err = self._cam_tools.begin("live")
        if err:
            self._report(err)
            return
        self._cam_stop.clear()
        self._cam_cells = ["opening\u2026"] * len(self.m.cfg.hands.cameras)
        self._report(None, "live confidence: opening the cameras one after another...")  # clears an old error
        self._cam_thread = threading.Thread(
            target=camera_confidence_worker,
            args=(self.m, self._cam_q, self._cam_stop),
            kwargs={"leases": self._cam_leases},
            daemon=True,
        )
        self._cam_thread.start()
        self.cam_live_btn.configure(text="Stop live view")
        self._cam_refresh()
        self.root.after(150, self._cam_poll)

    def _cam_poll(self) -> None:
        running = self._cam_thread is not None and self._cam_thread.is_alive()  # before draining
        latest = None
        while not self._cam_q.empty():
            latest = self._cam_q.get_nowait()
        if isinstance(latest, str):
            self._report(latest)
        elif isinstance(latest, dict):
            self._cam_cells = list(latest.get("cells", []))
            if self._cam_tools.busy == "live":
                self._report(None, str(latest.get("status", "")))
            self._cam_refresh()
        if running:
            self.root.after(150, self._cam_poll)
            return
        self._cam_tools.end("live")
        self.cam_live_btn.configure(text="Live confidence")
        self._cam_refresh()

    def _cam_test(self) -> None:
        err = self._cam_tools.begin("test")
        if err:
            self._report(err)
            return
        if self.m.dirty:
            self._on_save()
        self._cam_combo = None
        self.cam_save_combo_btn.configure(state="disabled")
        self._cam_out_set("testing: opening every camera one after another, then streaming ~10 s...")
        self._cam_refresh()
        self._cam_test_thread = threading.Thread(
            target=camera_test_worker,
            args=(self.m, self._cam_test_q),
            kwargs={"leases": self._cam_leases},
            daemon=True,
            name="camera-test",
        )
        self._cam_test_thread.start()
        self.root.after(200, self._cam_test_poll)

    def _cam_test_poll(self) -> None:
        running = self._cam_test_thread is not None and self._cam_test_thread.is_alive()  # before draining
        while not self._cam_test_q.empty():
            msg = self._cam_test_q.get_nowait()
            if isinstance(msg, str):
                self._cam_out_set(msg)
            elif isinstance(msg, dict) and "error" in msg:
                self._cam_out_set(f"error: {msg['error']}")
            elif isinstance(msg, dict) and "report" in msg:
                rep = msg["report"]
                self._cam_out_set("\n".join(rep.lines()))
                self._cam_tested = list(self.m.cfg.hands.cameras)
                self._cam_combo = rep.save_offer(self._cam_tested)
                if self._cam_combo is not None:
                    self.cam_save_combo_btn.configure(state="normal")
        if running:
            self.root.after(200, self._cam_test_poll)
            return
        self._cam_tools.end("test")
        self._cam_refresh()

    def _cam_save_combo(self) -> None:
        from away.multicam import camtest

        if not self._cam_combo:
            return
        changed = camtest.save_combination(self.m.cfg, self._cam_combo, getattr(self, "_cam_tested", None))
        self._on_save()
        self._cam_combo = None
        self.cam_save_combo_btn.configure(state="disabled")
        self._report(None, "saved camera settings: " + ("; ".join(changed) or "no change"))
        self._cam_refresh()

    def _build_performance(self, nb: Any) -> None:
        tk, ttk = self.tk, self.ttk
        f = self._scroll_tab(nb, "Performance")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Speech recognition runs on:").pack(side="left")
        self.perf_var = tk.StringVar()
        labels = [PERFORMANCE_LABELS[m] for m in PERFORMANCE_MODES]
        cb = ttk.Combobox(row, textvariable=self.perf_var, values=labels, state="readonly", width=20)
        cb.pack(side="left", padx=6)

        def on_mode(_e: Any = None) -> None:
            mode = next(k for k, v in PERFORMANCE_LABELS.items() if v == self.perf_var.get())
            self._report(self.m.set_voice("performance.mode", mode), "saved on Save; applies at the next start")
            self._perf_refresh()

        cb.bind("<<ComboboxSelected>>", on_mode)
        self.perf_text = tk.Text(f, height=18, width=96, wrap="word", relief="flat", background="#f6f6f6")
        self.perf_text.pack(fill="both", expand=True, pady=6)
        pack_row = ttk.Frame(f)
        pack_row.pack(fill="x")
        self.pack_var = tk.StringVar()
        ttk.Label(pack_row, textvariable=self.pack_var, wraplength=560).pack(side="left")
        self.pack_btn = ttk.Button(pack_row, text="Download GPU pack (~570 MB)", command=self._perf_install)
        self.pack_btn.pack(side="right")
        self.pack_rm = ttk.Button(pack_row, text="Remove GPU pack", command=self._perf_remove)
        self.pack_rm.pack(side="right", padx=4)
        self._perf_refresh()
        from away.app import models_panel  # optional models + eye tracking (v0.1.19), isolated in its own module

        models_panel.build(self, f)

    def _perf_refresh(self) -> None:
        try:
            info = self.m.performance_info()
        except Exception as exc:
            self.pack_var.set(f"hardware detection failed: {exc}")
            return
        self.perf_var.set(PERFORMANCE_LABELS.get(info["mode"], info["mode"]))
        self.perf_text.configure(state="normal")
        self.perf_text.delete("1.0", "end")
        self.perf_text.insert("end", "\n".join(info["lines"]))
        self.perf_text.configure(state="disabled")
        self.pack_var.set(info["pack_status"])
        self.pack_btn.configure(state="normal" if info["can_install"] else "disabled")
        self.pack_rm.configure(state="normal" if info["can_remove"] else "disabled")

    def _perf_install(self) -> None:
        self.pack_btn.configure(state="disabled")
        self.pack_var.set("Downloading the GPU pack... (you can keep using this window)")

        def run() -> None:
            msgs: list[str] = []
            err = self.m.install_gpu_pack(msgs.append)

            def done() -> None:
                from away.app import gpupack

                getattr(gpupack.cublas_available, "cache_clear", lambda: None)()
                self._report(err, "GPU pack installed: speech recognition uses the GPU from the next start")
                self._perf_refresh()

            self._ui.post(done)  # never touch Tk from this worker thread

        threading.Thread(target=run, daemon=True, name="gpu-pack").start()

    def _perf_remove(self) -> None:
        from away.app import gpupack

        gpupack.remove()
        self._report(None, "GPU pack removed")
        self._perf_refresh()

    def _build_studio(self, nb: Any) -> None:
        from away.studio.ui import StudioTab

        self._test_thread = None  # the Studio camera thread (the Tuning live view waits for it)
        self.studio = StudioTab(self, nb)

    def _build_test(self, nb: Any) -> None:
        tk, ttk = self.tk, self.ttk
        f = self._scroll_tab(nb, "Test gesture")
        ok, msg = camera_available()
        self.test_status = tk.StringVar(value=msg)
        ttk.Label(f, textvariable=self.test_status).pack(anchor="w")
        self.indicator = tk.StringVar(value="(no gesture yet)")
        tk.Label(f, textvariable=self.indicator, font=("TkDefaultFont", 18, "bold"), fg="#1b7f3b").pack(
            anchor="w", pady=12
        )
        self.test_btn = ttk.Button(
            f, text="Start camera test", command=self._toggle_test, state="normal" if ok else "disabled"
        )
        self.test_btn.pack(anchor="w")
        ttk.Label(f, text="Shows the gesture or clap detected and what it is bound to. Nothing is executed.").pack(
            anchor="w", pady=6
        )
        self._test_thread: threading.Thread | None = None

    # ------------------------------------------------------------- helpers
    def _report(self, err: str | None, ok: str | None = None) -> None:
        if err:
            self.status.set(f"error: {err}")
        elif ok:
            self.status.set(ok)
        else:
            self.status.set(status_text("config", self.m.path, self.m.dirty))

    def refresh(self) -> None:
        m = self.m
        self.profile_cb["values"] = m.profiles()
        self.profile_var.set(m.profile)
        self.active_cb["values"] = m.active_profile_choices()
        self.active_var.set(m.bindings.active_profile)
        self.apps_var.set(", ".join(m.profile_apps()))
        self.apps_entry.configure(state="disabled" if m.profile == "default" else "normal")
        sel = self.tree.selection()
        keep = self.tree.item(sel[0])["values"][0] if sel else None
        self.tree.delete(*self.tree.get_children())
        for row in m.rows():
            iid = self.tree.insert("", "end", values=(row.trigger, row.action, row.source))
            if row.trigger == keep:
                self.tree.selection_set(iid)
        for path, (var, _sc) in self.scales.items():
            var.set(m.voice_value(path.split("#")[0]))
        for path, var in self.vvars.items():
            var.set(m.voice_value(path))
        for path, var in getattr(self, "tvars", {}).items():
            var.set(bool(m.voice_value(path)))
        if hasattr(self, "roles_var"):
            self.roles_var.set(m.role_label())
        self.shell_var.set(m.bindings.allow_shell)
        self._cam_refresh()
        self._report(None)

    def _selected_trigger(self) -> str | None:
        sel = self.tree.selection()
        return str(self.tree.item(sel[0])["values"][0]) if sel else None

    def _on_select(self) -> None:
        trig = self._selected_trigger()
        if trig is None:
            return
        row = next(r for r in self.m.rows() if r.trigger == trig)
        self.kind_var.set(row.kind)
        self._on_kind()
        self.arg_var.set(row.arg)

    def _on_kind(self) -> None:
        kind = self.kind_var.get()
        choices = self.m.arg_choices(kind, self._selected_trigger())
        self.arg_cb["values"] = choices or []
        self.arg_cb.configure(state="readonly" if choices else ("disabled" if kind == "none" else "normal"))
        self.capture_btn.configure(state="normal" if kind == "key" else "disabled")
        if choices and self.arg_var.get() not in choices:
            self.arg_var.set(choices[0])
        if kind == "none":
            self.arg_var.set("")

    def _on_apply(self) -> None:
        trig = self._selected_trigger()
        if trig is None:
            self._report("select a row first")
            return
        err = self.m.set_binding(trig, self.kind_var.get(), self.arg_var.get())
        self.refresh()
        self._report(err, None if err else f"{trig} -> {self.m.compose(self.kind_var.get(), self.arg_var.get())}")

    def _on_add(self) -> None:
        err = self.m.add_trigger(self.new_cat.get(), self.new_name.get(), self.kind_var.get(), self.arg_var.get())
        if not err:
            self.new_name.set("")
        self.refresh()
        self._report(err)

    def _on_remove(self) -> None:
        trig = self._selected_trigger()
        if trig:
            self._report(self.m.remove_binding(trig))
            self.refresh()

    def _on_profile(self) -> None:
        self.m.select_profile(self.profile_var.get())
        self.refresh()

    def _on_apps(self) -> None:
        self._report(self.m.set_profile_apps(self.apps_var.get()))
        self.refresh()

    def _on_new_profile(self) -> None:
        from tkinter import simpledialog

        name = simpledialog.askstring("New profile", "Profile name (e.g. chrome):", parent=self.root)
        if not name:
            return
        apps = simpledialog.askstring(
            "New profile", "Process names, comma separated (e.g. chrome, chrome.exe):", parent=self.root
        )
        err = self.m.new_profile(name, apps or "")
        self.refresh()
        self._report(err)

    def _start_capture(self, var: Any, voice_path: str | None = None) -> None:
        self.capture.reset()
        self._capture_target = (var, voice_path)
        self.status.set("press the key combo now (Esc alone cancels)...")
        self.root.bind_all("<KeyPress>", self._on_key_press)
        self.root.bind_all("<KeyRelease>", self._on_key_release)

    def _finish_capture(self, combo: str | None) -> None:
        self.root.unbind_all("<KeyPress>")
        self.root.unbind_all("<KeyRelease>")
        var, vpath = self._capture_target
        self._capture_target = None
        if combo is None:
            self._report(None, "capture cancelled")
            return
        var.set(combo)
        if vpath:
            self._report(self.m.set_voice(vpath, combo))
        else:
            self._report(None, f"captured {combo}")

    def _on_key_press(self, ev: Any) -> str:
        if ev.keysym == "Escape" and not self.capture.held:
            self._finish_capture(None)
            return "break"
        res = self.capture.press(ev.keysym)
        if res:
            self._finish_capture(res)
        return "break"

    def _on_key_release(self, ev: Any) -> str:
        res = self.capture.release(ev.keysym)
        if res and self._capture_target is not None:
            self._finish_capture(res)
        return "break"

    def _on_save(self) -> None:
        try:
            p = self.m.save()
        except ValueError as exc:
            self._report(str(exc))
            return
        self._report(None, status_text("saved", p))

    def _on_reset_profile(self) -> None:
        self.m.reset_profile()
        self.refresh()

    def _on_reset_all(self) -> None:
        from tkinter import messagebox

        if messagebox.askyesno("Reset all", "Reset every setting to the defaults? (not saved until you press Save)"):
            self.m.reset_all()
            self.refresh()

    def _on_import(self) -> None:
        from tkinter import filedialog, messagebox

        p = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All", "*")])
        if not p:
            return
        replace = messagebox.askyesno("Import", "Replace all bindings? (No = merge into the current ones)")
        err = self.m.import_bindings(Path(p), replace)
        self.refresh()
        self._report(err, None if err else status_text("imported", p))

    def _on_export(self) -> None:
        from tkinter import filedialog

        p = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML", "*.yaml")])
        if p:
            self._report(None, status_text("exported", self.m.export_bindings(Path(p))))

    def _toggle_test(self) -> None:
        if self._test_thread is not None and self._test_thread.is_alive():
            self._test_stop.set()
            self.test_btn.configure(text="Start camera test")
            return
        self._test_stop.clear()
        self._test_thread = threading.Thread(
            target=gesture_test_worker, args=(self.m, self._test_q, self._test_stop), daemon=True
        )
        self._test_thread.start()
        self.test_btn.configure(text="Stop camera test")
        self.root.after(100, self._poll_test)

    def _poll_test(self) -> None:
        latest = None
        while not self._test_q.empty():
            latest = self._test_q.get_nowait()
        if latest:
            self.indicator.set(latest)
        if self._test_thread is not None and self._test_thread.is_alive():
            self.root.after(100, self._poll_test)

    def _on_close(self) -> None:
        from tkinter import messagebox

        self._test_stop.set()
        if getattr(self, "studio", None) is not None:
            self.studio.stop.set()  # Gesture Studio camera
        self._cam_stop.set()
        self._live_stop.set()  # Tuning tab live view
        for th in (self._test_thread, self._cam_thread, self._cam_test_thread, self._live_thread):  # cameras back first
            if th is not None and th.is_alive():
                th.join(timeout=1.5)
        if self.m.dirty and messagebox.askyesno("Unsaved changes", "Save changes before closing?"):
            self._on_save()
        self.root.destroy()


def main(config_path: Path | None = None, argv: list[str] | None = None) -> int:
    from away.app import crashlog

    crashlog.install("settings")  # crash.log: faulthandler + uncaught exceptions (v0.1.16); idempotent
    if getattr(sys, "frozen", False):  # pragma: no cover - installed app
        from away.app.frozen import setup_logging

        setup_logging()  # away.* INFO (the camera list on 'Refresh list'...) -> away.log
    try:
        model = SettingsModel(path=config_path)
    except ValueError as exc:
        print(f"error: your config is invalid, fix it or run `away config init`:\n{exc}", file=sys.stderr)
        return 2
    try:
        import tkinter as tk

        from away.control.monitors import enable_dpi_awareness

        enable_dpi_awareness()  # Windows: before Tk, or text is bitmap-scaled (blurry) or clipped; no-op if set
        root = tk.Tk()
    except Exception as exc:
        print(
            "The settings window needs a desktop session with Tk "
            f"({type(exc).__name__}: {exc}).\n"
            "Edit controls from the terminal instead:\n"
            "  away bindings list | set TRIGGER ACTION | reset | export FILE | import FILE\n"
            "  away profile list | new NAME --apps chrome | use NAME|auto\n"
            "  away voice mode always_on|push_to_talk|toggle ; away voice wake-word away|off\n"
            f"or edit {model.path}",
            file=sys.stderr,
        )
        return 1
    SettingsWindow(root, model)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
