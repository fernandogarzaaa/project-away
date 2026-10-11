"""Settings editor: bindings table, tunables, voice mode, profiles, live gesture test.

:class:`SettingsModel` holds all the logic and never imports tkinter, so it is
unit-tested headless. :class:`SettingsWindow` is a thin tkinter view over it
(tkinter ships with Python; nothing extra to install). ``main()`` degrades to a
clear message when there is no display or no Tk.
"""

from __future__ import annotations

import copy
import math
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from away.config import UPDATE_CHANNELS, VOICE_MODES, AppConfig
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
}
INT_TUNABLES = {
    "voice.end_silence_ms",
    "voice.min_utterance_ms",
    "hands.cursor.click_rewind_ms",
    "app.auto_pause_idle_ms",
    "voice.numbers_min_value",
}
VOICE_FIELDS = {
    "hands.cursor.curve": ("sigmoid", "power"),
    "hands.multicam.fusion": ("weighted", "best_view"),
    "voice.mode": VOICE_MODES,
    "voice.intent": ("dictation", "command"),
    "voice.ambient_fallback": ("commands_only", "type"),
    "updates.channel": UPDATE_CHANNELS,
}


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
        return [
            Tunable(p, TUNABLE_LABELS.get(p, p), float(lo), float(hi), float(v), p in INT_TUNABLES)
            for p, v, lo, hi in cfgmod.tunable_ranges(self.cfg)
        ]

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
        return None

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
    def camera_rows(self) -> list[dict[str, Any]]:
        """Configured cameras for the Cameras tab (empty list = single-camera mode)."""
        return [dict(vars(c)) for c in self.cfg.hands.cameras]

    def camera_mode(self) -> str:
        n = len(self.cfg.hands.cameras)
        return f"{n} cameras (fusion: {self.cfg.hands.multicam.fusion})" if n >= 2 else "single camera"

    def device_choices(self, devices: list[Any] | None = None) -> list[str]:
        """``"0: Integrated Webcam"`` entries for the device pickers."""
        if devices is None:
            from away.multicam.capture import list_cameras

            try:
                devices = list_cameras()
            except Exception:
                devices = []
        return [f"{d.index if d.index is not None else d.path}: {d.name}" for d in devices]

    @staticmethod
    def parse_device(choice: str) -> int | str:
        head = str(choice).split(":", 1)[0].strip()
        return int(head) if head.isdigit() else head

    def _cams_try(self, new: list[Any]) -> str | None:
        from away.config import validate_cameras

        try:
            validate_cameras(new)
        except ValueError as exc:
            return str(exc)
        self.cfg.hands.cameras = new
        return None

    def add_camera(self, device: int | str, role: str = "secondary") -> str | None:
        from away.config import CameraSpec

        new = [copy.copy(c) for c in self.cfg.hands.cameras]
        if not new:  # leaving single-camera mode: the current camera becomes the primary
            new.append(CameraSpec(self.cfg.hands.camera_index, "primary", mirror=self.cfg.hands.mirror))
        new.append(CameraSpec(device, role, mirror=self.cfg.hands.mirror))
        return self._cams_try(new)

    def remove_camera(self, i: int) -> str | None:
        new = [copy.copy(c) for j, c in enumerate(self.cfg.hands.cameras) if j != i]
        if len(new) == 1:  # back to single-camera mode
            if isinstance(new[0].device, int):
                self.cfg.hands.camera_index = new[0].device
            new = []
        return self._cams_try(new)

    def set_camera(self, i: int, field: str, value: Any) -> str | None:
        if field not in ("device", "role", "width", "height", "fps", "hfov_deg", "mirror"):
            return f"unknown camera setting {field!r}"
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
        if field == "role" and value == "primary":  # promoting one demotes the old primary
            for c in new:
                c.role = "secondary"
        setattr(new[i], field, value)
        return self._cams_try(new)

    @staticmethod
    def calibration_status() -> str:
        import time as _time

        from away.multicam.geometry import load_extrinsics

        ex = load_extrinsics()
        if ex is None:
            return "not calibrated (fusion works; depth/push/3D pinch need calibration)"
        when = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(ex.created))
        return f"calibrated {when} by {ex.method}, reprojection error {ex.reprojection_error_px:.2f} px"

    # -------------------------------------------------------- gesture test
    def describe_event(self, ev: GestureEvent) -> str | None:
        """Text for the live 'test gesture' indicator: ``pinch -> mouse:click``."""
        if ev.kind in ("hand_lost",):
            return "no hand"
        trig = event_trigger(ev)
        if trig is None:
            return None
        act = self.bindings.resolve("gestures", trig, self.profile)
        return f"{trig}  ->  {act}"


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


def gesture_test_worker(model: SettingsModel, out: queue.Queue[str], stop: threading.Event) -> None:  # pragma: no cover
    """Camera -> hand landmarks -> state machine -> indicator text (no actions are executed)."""
    from away import paths
    from away.hands.gestures import GestureStateMachine
    from away.hands.tracker import HandTracker, camera_frames

    cfg = model.cfg
    mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
    tracker = HandTracker(mdir / "hand_landmarker.task", 1)
    sm = GestureStateMachine(cfg.hands.gestures)
    try:
        for t, rgb in camera_frames(cfg.hands.camera_index, cfg.hands.mirror):
            if stop.is_set():
                break
            fr = tracker.process(rgb, t)
            for ev in sm.update(t, fr.landmarks):
                if ev.kind == "move":
                    continue
                txt = model.describe_event(ev)
                if txt:
                    out.put(txt)
    except Exception as exc:
        out.put(f"camera error: {exc}")
    finally:
        tracker.close()


def camera_confidence_worker(
    model: SettingsModel, out: queue.Queue[Any], stop: threading.Event
) -> None:  # pragma: no cover
    """Live per-camera hand confidence for the Cameras tab (no actions are executed)."""
    from away import paths
    from away.hands.tracker import HandTracker
    from away.multicam.capture import CameraThread, FramePairer
    from away.multicam.fusion import ViewObs, view_confidence

    cams = model.cfg.hands.cameras
    mdir = Path(model.cfg.models_dir) if model.cfg.models_dir else paths.models_dir()
    primary = next((i for i, c in enumerate(cams) if c.role == "primary"), 0)
    pairer = FramePairer(len(cams), primary, model.cfg.hands.multicam.pair_tolerance_ms / 1000.0, 0.5)
    threads = [CameraThread(i, c, pairer, stop) for i, c in enumerate(cams)]
    try:
        trackers = [HandTracker(mdir / "hand_landmarker.task", 1) for _ in cams]
        for th in threads:
            th.start()
        import time as _time

        while not stop.is_set():
            st = pairer.pop_set()
            if st is None:
                _time.sleep(0.005)
                continue
            confs: list[float | None] = []
            for i, item in enumerate(st[1]):
                if item is None:
                    confs.append(None)
                    continue
                fr = trackers[i].process(item[1], item[0])
                h, w = item[1].shape[:2]
                confs.append(
                    0.0
                    if fr.landmarks is None
                    else view_confidence(ViewObs(i, item[0], fr.landmarks, fr.handedness, fr.score, (w, h)))
                )
            out.put(confs)
    except Exception as exc:
        out.put(f"camera error: {exc}")
    finally:
        stop.set()


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
        self._test_q: queue.Queue[str] = queue.Queue()
        self._test_stop = threading.Event()
        root.title("Project Away settings")
        root.geometry("860x620")
        root.minsize(720, 520)

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

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=6)
        self._build_bindings(nb)
        self._build_tuning(nb)
        self._build_voice(nb)
        self._build_cameras(nb)
        self._build_test(nb)

        bottom = ttk.Frame(root, padding=6)
        bottom.pack(fill="x")
        for text, cmd in (
            ("Save", self._on_save),
            ("Reset profile", self._on_reset_profile),
            ("Reset all", self._on_reset_all),
            ("Import...", self._on_import),
            ("Export...", self._on_export),
            ("Close", self._on_close),
        ):
            ttk.Button(bottom, text=text, command=cmd).pack(side="left", padx=3)
        self.status = tk.StringVar(value=f"config: {model.path}")
        ttk.Label(bottom, textvariable=self.status, foreground="#555").pack(side="left", padx=8)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh()

    # ---------------------------------------------------------------- tabs
    def _build_bindings(self, nb: Any) -> None:
        tk, ttk = self.tk, self.ttk
        f = ttk.Frame(nb, padding=6)
        nb.add(f, text="Bindings")
        cols = ("trigger", "action", "from")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=14)
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
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(nb)
        nb.add(outer, text="Tuning")
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        f = ttk.Frame(canvas, padding=6)
        f.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=f, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.scales: dict[str, tuple[Any, Any]] = {}
        for i, t in enumerate(self.m.tunables()):
            ttk.Label(f, text=t.label).grid(row=i, column=0, sticky="w", padx=4, pady=1)
            var = tk.DoubleVar(value=t.value)
            sc = tk.Scale(
                f,
                from_=t.lo,
                to=t.hi,
                resolution=t.resolution,
                orient="horizontal",
                length=320,
                variable=var,
                showvalue=True,
                command=lambda v, p=t.path: self._on_scale(p, float(v)),
            )
            sc.grid(row=i, column=1, sticky="we")
            self.scales[t.path] = (var, sc)

    def _on_scale(self, path: str, value: float) -> None:
        cur = float(self.m.voice_value(path))
        step = {t.path: t.resolution for t in self.m.tunables()}[path]
        if abs(value - cur) >= step / 2:  # ignore the callback Tk fires when refresh() sets the variable
            self._report(self.m.set_tunable(path, value))

    def _build_voice(self, nb: Any) -> None:
        tk, ttk = self.tk, self.ttk
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="Voice & input")
        self.vvars: dict[str, Any] = {}
        r = 0
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
            ("voice.commands_enabled", "Voice commands"),
            ("app.always_on", "Active as soon as the app starts (gestures + voice)"),
            ("app.start_minimized", "Start minimized to tray"),
            ("face.brow_raise_toggle", "Brow-raise trigger"),
            ("face.mouth_open_trigger", "Mouth-open trigger"),
            ("face.wink_trigger", "Wink triggers (left / right)"),
            ("hands.dwell.enabled", "Dwell click (rest the cursor)"),
            ("hands.cursor.click_freeze", "Steady click (freeze while pinching)"),
            ("app.auto_pause", "Pause while I use the mouse / keyboard"),
            ("voice.numbers_as_digits", "Spoken numbers as digits"),
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
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="Cameras")
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
        self.cam_dev_cb = ttk.Combobox(f, textvariable=self.cam_dev, width=34)
        self.cam_dev_cb.grid(row=2, column=1, sticky="we", padx=4)
        ttk.Button(f, text="Refresh list", command=self._cam_refresh_devices).grid(row=2, column=2)
        ttk.Label(f, text="Role:").grid(row=3, column=0, sticky="w")
        self.cam_role = tk.StringVar(value="secondary")
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
        ttk.Button(btns, text="Calibrate (wave hand ~20 s)", command=self._cam_calibrate).pack(side="left", padx=8)
        self.cam_live_btn = ttk.Button(btns, text="Live confidence", command=self._cam_live)
        self.cam_live_btn.pack(side="left", padx=2)
        ttk.Label(f, text="Fusion:").grid(row=6, column=0, sticky="w")
        self.cam_fusion = tk.StringVar(value=self.m.cfg.hands.multicam.fusion)
        cb = ttk.Combobox(f, textvariable=self.cam_fusion, values=["weighted", "best_view"], state="readonly", width=12)
        cb.grid(row=6, column=1, sticky="w", padx=4)
        cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._report(self.m.set_voice("hands.multicam.fusion", self.cam_fusion.get())),
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
        self._cam_q: queue.Queue[Any] = queue.Queue()
        self._cam_stop = threading.Event()
        self._cam_thread: threading.Thread | None = None
        self._cam_confs: list[Any] = []
        self._cam_refresh_devices()

    def _cam_refresh_devices(self) -> None:
        self.cam_dev_cb["values"] = self.m.device_choices()

    def _cam_selected(self) -> int | None:
        sel = self.cam_tree.selection()
        return self.cam_tree.index(sel[0]) if sel else None

    def _cam_refresh(self) -> None:
        self.cam_mode.set(self.m.camera_mode())
        self.cam_tree.delete(*self.cam_tree.get_children())
        for i, r in enumerate(self.m.camera_rows()):
            conf = self._cam_confs[i] if i < len(self._cam_confs) else None
            bar = "-" if conf is None else ("#" * round(conf * 10)).ljust(10, ".") + f" {conf:.2f}"
            self.cam_tree.insert(
                "", "end", values=(r["role"], r["device"], f"{r['width']}x{r['height']}", r["fps"], r["hfov_deg"], bar)
            )
        self.cam_calib.set(self.m.calibration_status())

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

        if len(self.m.cfg.hands.cameras) < 2:
            self._report("add a second camera first")
            return
        if self.m.dirty:
            self._on_save()
        self._report(None, "calibrating: move your open hand through both cameras' view for ~20 s...")

        def run() -> None:
            r = subprocess.run(
                [sys.executable, "-m", "away", "calibrate-cameras", "--config", str(self.m.path)],
                capture_output=True,
                text=True,
            )
            msg = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else r.stderr.strip()[-200:]

            def done() -> None:
                self._report(None, msg)
                self._cam_refresh()

            self.root.after(0, done)

        threading.Thread(target=run, daemon=True).start()

    def _cam_live(self) -> None:
        if self._cam_thread is not None and self._cam_thread.is_alive():
            self._cam_stop.set()
            self.cam_live_btn.configure(text="Live confidence")
            return
        if len(self.m.cfg.hands.cameras) < 2:
            self._report("add a second camera first")
            return
        self._cam_stop.clear()
        self._cam_thread = threading.Thread(
            target=camera_confidence_worker, args=(self.m, self._cam_q, self._cam_stop), daemon=True
        )
        self._cam_thread.start()
        self.cam_live_btn.configure(text="Stop live view")
        self.root.after(150, self._cam_poll)

    def _cam_poll(self) -> None:
        latest = None
        while not self._cam_q.empty():
            latest = self._cam_q.get_nowait()
        if isinstance(latest, str):
            self._report(latest)
        elif latest is not None:
            self._cam_confs = latest
            self._cam_refresh()
        if self._cam_thread is not None and self._cam_thread.is_alive():
            self.root.after(150, self._cam_poll)

    def _build_test(self, nb: Any) -> None:
        tk, ttk = self.tk, self.ttk
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="Test gesture")
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
        ttk.Label(f, text="Shows which trigger was detected and what it is bound to. Nothing is executed.").pack(
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
            self.status.set(("unsaved changes - " if self.m.dirty else "") + f"config: {self.m.path}")

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
        self._report(None, f"saved to {p} (a running Away picks it up within a second)")

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
        self._report(err, None if err else f"imported {p}")

    def _on_export(self) -> None:
        from tkinter import filedialog

        p = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML", "*.yaml")])
        if p:
            self._report(None, f"exported to {self.m.export_bindings(Path(p))}")

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
        self._cam_stop.set()
        if self.m.dirty and messagebox.askyesno("Unsaved changes", "Save changes before closing?"):
            self._on_save()
        self.root.destroy()


def main(config_path: Path | None = None, argv: list[str] | None = None) -> int:
    try:
        model = SettingsModel(path=config_path)
    except ValueError as exc:
        print(f"error: your config is invalid, fix it or run `away config init`:\n{exc}", file=sys.stderr)
        return 2
    try:
        import tkinter as tk

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
