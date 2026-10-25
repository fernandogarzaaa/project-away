"""JSON bridge between the native macOS Settings app and the Python settings logic (v0.1.35).

The SwiftUI Settings app (``macos/SettingsApp``) never re-implements config rules: it asks this module for one
*settings document* (every section, group and field with its value, type, choices, range, label and help) and
sends changes back as ``{"voice.mode": "toggle", ...}``. Changes go through :class:`SettingsModel`, the same code
path the Tk window uses (``set_voice`` / ``set_tunable`` / the comfort switches / the speech language), and are
saved with :func:`away.config.save`. The running app hot-reloads ``config.yaml`` when its mtime changes
(``runner._watch_config``), exactly as after a Save in the Tk window, so nothing else has to be signalled.

Three ways in (``away settings-bridge ...``):

* ``get`` prints the document as JSON.
* ``set --json '{"voice.mode": "toggle"}'`` validates every value, saves only when all of them are valid
  (all or nothing), prints ``{"ok": ..., "errors": {key: message}, "document": ...}`` and exits 1 on an error.
* ``serve`` answers one JSON request per stdin line (``{"id": 1, "op": "get" | "set" | "status"}``) with one JSON
  line on stdout, so the native window pays for Python's start-up once instead of on every switch.

Only the keys in the document can be set (a whitelist); a field whose type is ``bool`` needs a JSON boolean, so a
string ``"false"`` is not quietly read as true. The document is built from data, so a tunable added to
:func:`away.config.tunable_ranges` appears in the native window (under its Tuning group) without a Swift change.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

# v0.1.35: the sidebar of the native Settings app. "tk" names the Tk tab a section stands in for, so
# `away settings --tab Cameras` opens the right section (see away.app.settings_native).
SECTIONS: tuple[dict[str, str], ...] = (
    {"id": "general", "title": "General", "symbol": "gearshape", "tk": "Getting started"},
    {"id": "pointer", "title": "Hands & Pointer", "symbol": "hand.point.up.left", "tk": ""},
    {"id": "voice", "title": "Voice & Dictation", "symbol": "waveform", "tk": "Voice & input"},
    {"id": "face", "title": "Face & Eyes", "symbol": "face.smiling", "tk": ""},
    {"id": "cameras", "title": "Cameras", "symbol": "web.camera", "tk": "Cameras"},
    {"id": "models", "title": "Models", "symbol": "cpu", "tk": "Models"},
    {"id": "advanced", "title": "Advanced", "symbol": "slider.horizontal.3", "tk": ""},
    {"id": "permissions", "title": "Permissions", "symbol": "lock.shield", "tk": ""},
    {"id": "about", "title": "About", "symbol": "info.circle", "tk": ""},
)

# Tk tabs the native app does not have yet: it links to them ("More settings...") and `--tab` opens them in Tk.
TK_ONLY_TABS = ("Bindings", "Gesture Studio", "Shortcuts", "Agents", "Tuning", "Performance")

# Settings > Tuning groups (away.app.tunables.GROUPS) -> the section their sliders show in. A group not listed
# here (a new one) lands in "advanced", so a new tunable is never lost.
TUNABLE_SECTIONS = {
    "Cursor": "pointer",
    "Clicks": "pointer",
    "Scroll": "pointer",
    "Gestures": "pointer",
    "Voice": "voice",
    "Face": "face",
    "Head & eyes": "face",
    "Advanced": "advanced",
    "Other": "advanced",
}

# Choice labels for the drop-downs (values are what config.yaml stores)
CHOICE_LABELS = {
    "hands.roles.mode": {"auto": "Auto", "right": "Right hand", "left": "Left hand", "any": "Any (first hand seen)"},
    "hands.cursor.curve": {"sigmoid": "Sigmoid (precise when slow)", "power": "Power (classic)"},
    "voice.mode": {"always_on": "Always on", "push_to_talk": "Push to talk", "toggle": "Toggle with hotkey"},
    "voice.intent": {"dictation": "Dictation and commands", "command": "Commands only"},
    "voice.ambient_fallback": {"commands_only": "Commands only", "type": "Type what I say"},
    "face.gate_when_camera_busy": {"block": "Don't type", "allow": "Type anyway"},
    "updates.channel": {"stable": "Stable", "beta": "Beta (pre-releases too)"},
    "performance.mode": {"auto": "Auto (detect)", "cpu": "CPU only", "gpu": "GPU"},
    "hands.camera_backend": {"auto": "Auto", "dshow": "DirectShow", "msmf": "Media Foundation", "winrt": "WinRT"},
}

# Hand-picked switches, drop-downs and text fields, grouped like the Tk tabs they come from
# (Getting started, Voice & input, Performance). Each: (key, kind, label, help[, platforms]).
# kind: bool | choice | text | hotkey. Sliders come from SettingsModel.tunables() instead.
_W = ("win32",)
CURATED: tuple[tuple[str, str, str, tuple[tuple[Any, ...], ...]], ...] = (
    (
        "general",
        "Startup",
        "",
        (
            ("app.always_on", "bool", "Active as soon as the app starts", "Gestures and voice are on at launch."),
            ("app.start_minimized", "bool", "Start in the menu bar only", "Off also opens Settings at launch."),
            ("cursor_overlay", "bool", "Ring at the cursor", "Shows where gestures act. Applies at the next start."),
        ),
    ),
    ("general", "Comfort", "", ()),  # filled from away.comfort.settings.PLAIN_SWITCHES
    (
        "general",
        "Pausing",
        "",
        (
            ("app.auto_pause", "bool", "Pause while I use the mouse or keyboard", ""),
            ("app.toggle_hotkey", "hotkey", "Gestures on/off hotkey", "Empty = no hotkey."),
        ),
    ),
    (
        "general",
        "Updates",
        "The only network call is an anonymous request to GitHub Releases; nothing is sent about you.",
        (
            ("updates.check", "bool", "Check for updates daily", ""),
            ("updates.auto_install", "bool", "Install updates automatically when idle", ""),
            ("updates.channel", "choice", "Update channel", ""),
        ),
    ),
    (
        "pointer",
        "Pointer",
        "",
        (
            ("hands.roles.mode", "choice", "Pointer hand", "Which hand moves the pointer."),
            ("hands.cursor.curve", "choice", "Acceleration curve", ""),
            ("pointer.smooth", "bool", "Smoother cursor", "Predicts where your hand is going: less lag and shake."),
            ("hands.cursor.click_freeze", "bool", "Steady click", "Holds the pointer still while a pinch closes."),
            ("hands.dwell.enabled", "bool", "Dwell click", "Rest the pointer on something to click it."),
            (
                "hands.cursor.big_cursor_size",
                "slider",
                "Bigger cursor while gestures are on",
                "0 = off.",
                _W,
            ),
        ),
    ),
    (
        "pointer",
        "Scrolling",
        "",
        (
            ("hands.gestures.autoscroll_enabled", "bool", "Four-finger auto-scroll", "Thumb folded; a fist stops it."),
            ("hands.gestures.autoscroll_horizontal", "bool", "Auto-scroll sideways too", ""),
            ("hands.palm_scroll.enabled", "bool", "Palm scroll", "Glide a fingertip along the other, flat hand."),
            ("hands.palm_scroll.invert", "bool", "Invert palm scroll", ""),
            ("hands.palm_scroll.horizontal", "bool", "Two fingers scroll sideways", ""),
            ("hands.palm_scroll.momentum", "bool", "Momentum after the fingertip lifts", ""),
        ),
    ),
    (
        "voice",
        "Listening",
        "",
        (
            ("voice.mode", "choice", "Input mode", ""),
            ("voice.intent", "choice", "Always-on intent", ""),
            ("voice.language", "choice", "Speech language", "Auto falls back to the language you use most."),
            ("voice.wake_word_enabled", "bool", "Require the wake word", ""),
            ("voice.wake_word", "text", "Wake word", ""),
            ("voice.ambient_fallback", "choice", "No wake word and no camera", ""),
            ("voice.commands_enabled", "bool", "Voice commands", "A phrase you bound runs instead of being typed."),
            ("voice.numbers_as_digits", "bool", "Spoken numbers as digits", "“twenty five” becomes 25."),
        ),
    ),
    (
        "voice",
        "Face gate",
        "",
        (
            (
                "face.gate_dictation",
                "bool",
                "Only type when I face the camera and speak",
                "Off = always type what it hears.",
            ),
            ("face.gate_when_camera_busy", "choice", "While a camera check has the camera", ""),
        ),
    ),
    (
        "voice",
        "Hotkeys",
        "Use pynput names, e.g. <ctrl>+<alt>+space. Empty turns a hotkey off.",
        (
            ("voice.hotkey", "hotkey", "Push-to-talk / toggle", ""),
            ("voice.mute_hotkey", "hotkey", "Mute microphone", ""),
            ("voice.hold_key", "hotkey", "Hold-to-talk key", ""),
            ("voice.edit_hotkey", "hotkey", "Edit selected text", ""),
        ),
    ),
    (
        "voice",
        "Claps",
        "",
        (
            ("voice.clap_enabled", "bool", "Clap control", "One clap activates, two claps pause."),
            ("voice.clap_require_hands", "bool", "Only when the camera sees both hands together", ""),
        ),
    ),
    (
        "voice",
        "Talking to your agent",
        "",
        (
            ("voice.conversational", "bool", "Keep listening after the agent replies", ""),
            ("voice.listen_sound", "bool", "Soft sound when the agent listens", ""),
        ),
    ),
    (
        "face",
        "Face triggers",
        "Bind them to actions in More settings > Bindings.",
        (
            ("face.brow_raise_toggle", "bool", "Brow raise", ""),
            ("face.mouth_open_trigger", "bool", "Mouth open", ""),
            ("face.wink_trigger", "bool", "Winks (left / right)", ""),
        ),
    ),
    ("face", "Looking at the screen", "", ()),  # filled from away.comfort.settings.ADVANCED_SWITCHES
    (
        "cameras",
        "Capture",
        "",
        (("hands.camera_backend", "choice", "Capture backend", "", _W),),
    ),
    (
        "models",
        "Performance",
        "New models load the next time Project Away starts.",
        (("performance.mode", "choice", "Speech recognition runs on", ""),),
    ),
)

# Tunables shown inside a curated group instead of under their Tuning group
TUNABLE_HOME = {
    "app.auto_pause_idle_ms": ("general", "Pausing"),
    "voice.clap_sensitivity": ("voice", "Claps"),
    "voice.clap_double_window": ("voice", "Claps"),
    "voice.vad_sensitivity": ("voice", "Listening"),
    "voice.listen_seconds": ("voice", "Talking to your agent"),
    "voice.agent_end_silence_ms": ("voice", "Talking to your agent"),
    "hands.min_camera_fps": ("cameras", "Capture"),
}

PERMISSION_URLS = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
}
PERMISSIONS = (
    ("accessibility", "Accessibility", "Moves the pointer, clicks and types for you."),
    ("input_monitoring", "Input Monitoring", "Lets the hotkeys work."),
    ("camera", "Camera", "Tracks your hands and face, on this Mac only."),
    ("microphone", "Microphone", "Hears dictation and voice commands, on this Mac only."),
)


def _get(cfg: Any, path: str) -> Any:
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _model(path: Path | None = None) -> Any:
    from away.app.settings_gui import SettingsModel

    return SettingsModel(path=path)


def _choices(key: str) -> list[dict[str, Any]]:
    from away.app.settings_gui import VOICE_FIELDS

    if key == "voice.language":
        from away.voice.language import LANGUAGES

        return [{"value": code, "label": label} for code, label in LANGUAGES]
    values = VOICE_FIELDS.get(key, ())
    if key == "hands.roles.mode":
        from away.app.settings_gui import ROLE_CHOICES

        values = ROLE_CHOICES  # "any" is config-only, like the Tk drop-down
    labels = CHOICE_LABELS.get(key, {})
    return [{"value": v, "label": labels.get(v, str(v))} for v in values]


def _value(model: Any, key: str) -> Any:
    if key == "pointer.smooth":
        from away.pointer.settings_ui import toggle_on

        return bool(toggle_on(model.cfg, "smooth"))
    if key == "voice.language":
        from away.voice.language import normalize

        return normalize(model.cfg.voice.language) or ""
    return _jsonable(_get(model.cfg, key))


def _tunable_field(t: Any) -> dict[str, Any]:
    step = 1.0 if t.is_int else t.resolution
    return {
        "key": t.path,
        "type": "number",
        "label": t.title or t.label,
        "help": t.help,
        "value": int(t.value) if t.is_int else t.value,
        "min": t.lo,
        "max": t.hi,
        "step": step,
        "unit": t.unit,
        "is_int": bool(t.is_int),
        "default": t.default,
        "tuning_group": t.group,
    }


def build_document(model: Any, platform: str | None = None) -> dict[str, Any]:
    """The whole settings document for the native window: sections > groups > fields (with values)."""
    from away import __version__
    from away.comfort.settings import ADVANCED_SWITCHES, PLAIN_SWITCHES

    plat = platform or sys.platform
    tunables = {t.path: t for t in model.tunables()}
    groups: dict[str, list[dict[str, Any]]] = {s["id"]: [] for s in SECTIONS}
    by_title: dict[tuple[str, str], dict[str, Any]] = {}

    def group(section: str, title: str, footer: str = "", tuning: bool = False) -> dict[str, Any]:
        g = by_title.get((section, title))
        if g is None:
            g = {"title": title, "footer": footer, "tuning": tuning, "fields": []}
            by_title[(section, title)] = g
            groups.setdefault(section, []).append(g)
        return g

    for section, title, footer, rows in CURATED:
        g = group(section, title, footer)
        for row in rows:
            key, kind, label, help_text, *rest = row
            if rest and plat not in rest[0]:
                continue
            if kind == "slider":
                if key in tunables:
                    g["fields"].append({**_tunable_field(tunables[key]), "label": label, "help": help_text})
                continue
            field: dict[str, Any] = {"key": key, "type": kind, "label": label, "help": help_text}
            field["value"] = _value(model, key)
            if kind == "choice":
                field["choices"] = _choices(key)
            g["fields"].append(field)
    for key, label, help_text in PLAIN_SWITCHES:
        group("general", "Comfort")["fields"].append(
            {"key": key, "type": "bool", "label": label, "help": help_text, "value": _value(model, key)}
        )
    for key, label in ADVANCED_SWITCHES:
        group("face", "Looking at the screen")["fields"].append(
            {"key": key, "type": "bool", "label": label, "help": "", "value": _value(model, key)}
        )
    placed = {f["key"] for gs in groups.values() for g in gs for f in g["fields"]}
    for path, t in tunables.items():  # every other slider, under its Tuning group (data-driven)
        if path in placed:
            continue
        if path in TUNABLE_HOME:
            section, title = TUNABLE_HOME[path]
            group(section, title)["fields"].append(_tunable_field(t))
            continue
        if path == "hands.cursor.big_cursor_size" and plat not in _W:
            continue  # Windows-only cursor enlarging
        section = TUNABLE_SECTIONS.get(t.group, "advanced")
        group(section, t.group, tuning=True)["fields"].append(_tunable_field(t))
    for gs in groups.values():  # drop groups left empty on this platform
        gs[:] = [g for g in gs if g["fields"]]
    return {
        "version": __version__,
        "config_path": str(model.path),
        "platform": plat,
        "sections": [{**s, "groups": groups.get(s["id"], [])} for s in SECTIONS],
        "tk_tabs": list(TK_ONLY_TABS),
        "models": _models(model),
        "cameras": _cameras(model),
        "permissions": permissions(plat),
        "commands": commands(model.path),
    }


def commands(config: Path | None = None) -> dict[str, list[str]]:
    """argv for the windows the native app opens as their own processes (the same ones the Tk tabs open).
    ``tk_settings`` gets ``--tab NAME`` appended for "More settings..." on an unported tab."""
    from away.app.frozen import away_command
    from away.comfort import actions

    tk = ["settings", "--tk"]
    if config is not None and Path(config) != _default_config():
        tk += ["--config", str(config)]
    return {
        "tk_settings": away_command(tk),
        "first_run": actions.first_run_command(),
        "calibrate": actions.calibrate_command(),
        "check_pc": actions.check_pc_command(),
    }


def _default_config() -> Path:
    from away import config as cfgmod

    return cfgmod.config_path()


def fields(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every settable field of a document, by key."""
    return {f["key"]: f for s in doc["sections"] for g in s["groups"] for f in g["fields"]}


def _models(model: Any) -> list[dict[str, Any]]:
    """Settings > Models, read-only (changing models stays in the Tk Models tab for now)."""
    try:
        from away.app import model_registry as MR
        from away.app import models_tab

        reg = MR.Registry()
        return [
            {
                "slot": r["slot"],
                "title": r["title"],
                "current": r["label"],  # the Models tab's wording ("Automatic (picked for this PC)"...)
                "runs_on": r.get("runs_on", ""),
                "optional": bool(r.get("optional")),
                "on": bool(r.get("on", True)),
            }
            for r in models_tab.rows(reg, model.cfg)
        ]
    except Exception:
        return []


def _cameras(model: Any) -> list[dict[str, Any]]:
    """The configured cameras, read-only. Never opens a camera (no device scan: that can take seconds)."""
    try:
        h, perf = model.cfg.hands, model.cfg.performance
        if not h.cameras:
            return [
                {
                    "device": _jsonable(h.camera_index),
                    "role": "primary",
                    "width": perf.camera_width or 640,
                    "height": perf.camera_height or 480,
                    "fps": perf.camera_fps or 30,
                    "mirror": bool(h.mirror),
                    "implicit": True,
                }
            ]
        return [
            {
                "device": _jsonable(c.device),
                "role": c.role,
                "width": c.width,
                "height": c.height,
                "fps": c.fps,
                "mirror": bool(c.mirror),
                "implicit": False,
            }
            for c in h.cameras
        ]
    except Exception:
        return []


def permissions(platform: str | None = None) -> list[dict[str, Any]]:
    """Accessibility / Input Monitoring state (None = unknown) and the System Settings page for each permission.
    Camera and Microphone state is read natively by the Swift app (AVFoundation); here it is None."""
    from away.app import macperms

    plat = platform or sys.platform
    state = {
        "accessibility": macperms.accessibility(False, plat),
        "input_monitoring": macperms.input_monitoring(False, plat),
    }
    urls = {**macperms.SETTINGS, **PERMISSION_URLS}
    return [
        {"id": k, "title": title, "help": help_text, "granted": state.get(k), "url": urls[k]}
        for k, title, help_text in PERMISSIONS
    ]


def apply_changes(model: Any, changes: dict[str, Any], platform: str | None = None) -> dict[str, str]:
    """Validate and apply ``changes`` to ``model`` through SettingsModel's own setters. Returns {key: error}."""
    from away.comfort.settings import ADVANCED_SWITCHES, PLAIN_SWITCHES, set_switch

    known = fields(build_document(model, platform))
    comfort = {p for p, *_ in PLAIN_SWITCHES} | {p for p, _ in ADVANCED_SWITCHES}
    errors: dict[str, str] = {}
    for key, value in changes.items():
        spec = known.get(key)
        if spec is None:
            errors[key] = f"unknown setting {key!r}"
            continue
        kind = spec["type"]
        err: str | None = None
        if kind == "bool" and not isinstance(value, bool):
            err = f"{key} must be true or false"
        elif kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            err = f"{key} must be a number"
        elif kind in ("text", "hotkey", "choice") and not isinstance(value, str):
            err = f"{key} must be text"
        elif kind == "number":
            err = model.set_tunable(key, float(value))
        elif key == "pointer.smooth":
            from away.pointer.settings_ui import set_toggle

            set_toggle(model.cfg, "smooth", value)
        elif key == "voice.language":
            from away.voice.language import label_for

            err = model.set_language_label(label_for(value) if value else "auto")
        elif key in comfort:
            err = set_switch(model.cfg, key, value)
        else:
            err = model.set_voice(key, value)
        if err:
            errors[key] = err
    return errors


def set_values(changes: Any, path: Path | None = None, platform: str | None = None) -> dict[str, Any]:
    """Apply ``changes`` to a fresh copy of the config file and save it when every value is valid."""
    if not isinstance(changes, dict):
        return {"ok": False, "errors": {"": "expected a JSON object of {key: value}"}}
    model = _model(path)  # re-read: the tray (face gate) or the Tk window may have saved since the last request
    errors = apply_changes(model, changes, platform)
    saved = None
    if not errors and changes:
        try:
            saved = str(model.save())
        except ValueError as exc:
            errors[""] = str(exc)
    out: dict[str, Any] = {"ok": not errors, "errors": errors, "saved": saved}
    out["document"] = build_document(_model(path) if errors else model, platform)
    return out


def handle(request: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """One ``serve`` request -> its response (never raises)."""
    rid = request.get("id") if isinstance(request, dict) else None
    try:
        op = request.get("op")
        if op == "get":
            out: dict[str, Any] = {"ok": True, "document": build_document(_model(path))}
        elif op == "set":
            out = set_values(request.get("values"), path)
        elif op == "status":
            out = {"ok": True, "permissions": permissions()}
        else:
            out = {"ok": False, "errors": {"": f"unknown op {op!r}"}}
    except Exception as exc:  # a bad config file, a bug: report it to the window instead of dying
        out = {"ok": False, "errors": {"": f"{type(exc).__name__}: {exc}"}}
    return {"id": rid, **out}


def serve(lines: Iterable[str], write: Callable[[str], Any], path: Path | None = None) -> int:
    """JSON lines in, JSON lines out, until stdin closes (the window quit)."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as exc:
            resp: dict[str, Any] = {"id": None, "ok": False, "errors": {"": f"bad JSON: {exc}"}}
        else:
            # anything a setter prints must not corrupt the JSON stream on stdout
            with contextlib.redirect_stdout(sys.stderr):
                resp = handle(req, path)
        write(json.dumps(resp) + "\n")
    return 0


def main(action: str, values: str | None = None, config: str | None = None) -> int:
    """``away settings-bridge get | set --json ... | serve``."""
    path = Path(config) if config else None
    out = sys.stdout
    if action == "serve":

        def write(text: str) -> None:
            out.write(text)
            out.flush()

        return serve(sys.stdin, write, path)
    with contextlib.redirect_stdout(sys.stderr):
        if action == "get":
            resp = handle({"op": "get"}, path)
        elif action == "status":
            resp = handle({"op": "status"}, path)
        else:
            try:
                changes = json.loads(values or "{}")
            except ValueError as exc:
                resp = {"id": None, "ok": False, "errors": {"": f"bad JSON: {exc}"}}
            else:
                resp = handle({"op": "set", "values": changes}, path)
    out.write(json.dumps(resp, indent=None) + "\n")
    out.flush()
    return 0 if resp.get("ok") else 1
