"""User-remappable controls: triggers -> actions, with per-app profiles.

Triggers (what the user does) are grouped by source:

* ``gestures``: built-in hand gestures (:data:`GESTURE_TRIGGERS`) plus any
  custom gesture listed in ``custom_gestures`` (record it with ``away record``).
* ``voice``: spoken phrases, matched case- and punctuation-insensitively.
* ``face``: ``brow_raise``, ``mouth_open``, ``wink_left``, ``wink_right``.
* ``hotkeys``: key combos (``ctrl+alt+k``) plus the named slots ``ptt``
  (= ``voice.hotkey``) and ``mute`` (= ``voice.mute_hotkey``).

Actions (what happens) are written ``kind:argument``::

    none
    mouse:click | double_click | right_click | middle_click | scroll_up | scroll_down
    mouse:move (point only) | mouse:drag (pinch_hold only) | mouse:scroll (two_finger_scroll only)
    key:ctrl+tab              press a key combo
    type:Hello there          type text
    launch:firefox            start an app (no shell); launch:https://... opens a URL
    shell:notify-send hi      run a shell command (needs bindings.allow_shell: true)
    dictation:toggle | start | stop
    mic:toggle | on | off     mute/unmute the microphone
    voice:command_mode | dictation_mode | toggle_mode
    pause:toggle | on | off   pause/resume gesture control
    cursor:next_monitor | prev_monitor | primary_monitor | monitor 2
                              jump the cursor to the centre of a monitor (numbers from `away monitors`)

Profiles: ``default`` holds the full map. Any other profile lists the
``apps`` (process names, case-insensitive, ``.exe`` optional) it applies to
and overrides only the triggers it names; everything else falls through to
``default``. ``active_profile: auto`` picks by the focused app.

This module is pure data + validation (no devices) so it is fully testable.
"""

from __future__ import annotations

import copy
import difflib
import re
import string
from dataclasses import dataclass, field
from typing import Any

from away.control.keys import KeyComboError, canonical

GESTURE_TRIGGERS = (
    "point",
    "pinch",
    "double_pinch",
    "pinch_hold",
    "two_finger_scroll",
    "open_palm",
    "swipe_left",
    "swipe_right",
    "push",
    "dwell",
)
FACE_TRIGGERS = ("brow_raise", "mouth_open", "wink_left", "wink_right")
HOTKEY_SLOTS = ("ptt", "mute")
CATEGORIES = ("gestures", "voice", "face", "hotkeys")
CATEGORY_PREFIX = {"gestures": "gesture", "voice": "voice", "face": "face", "hotkeys": "hotkey"}
PREFIX_CATEGORY = {v: k for k, v in CATEGORY_PREFIX.items()}

MAX_MONITORS = 9
CURSOR_ARGS = ("next_monitor", "prev_monitor", "primary_monitor", *(f"monitor {i}" for i in range(1, MAX_MONITORS + 1)))
_CURSOR_ALIASES = {"previous_monitor": "prev_monitor", "next": "next_monitor", "prev": "prev_monitor"}
_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

ACTION_ARGS: dict[str, tuple[str, ...] | None] = {
    "none": (),
    "mouse": (
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "scroll_up",
        "scroll_down",
        "move",
        "drag",
        "scroll",
    ),
    "key": None,  # free: a key combo
    "type": None,  # free text
    "launch": None,  # app command or URL
    "shell": None,  # shell command (opt-in)
    "dictation": ("toggle", "start", "stop"),
    "mic": ("toggle", "on", "off"),
    "voice": ("command_mode", "dictation_mode", "toggle_mode"),
    "pause": ("toggle", "on", "off"),
    "cursor": CURSOR_ARGS,
}
ACTION_KINDS = tuple(ACTION_ARGS)
# continuous mouse actions only make sense on the matching continuous gesture
CONTINUOUS_ONLY = {"move": "point", "drag": "pinch_hold", "scroll": "two_finger_scroll"}
LABEL_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")


class BindingsError(ValueError):
    """Raised with every problem found, one per line, each prefixed by its YAML path."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("invalid bindings:\n  - " + "\n  - ".join(errors))


@dataclass(frozen=True)
class Action:
    kind: str
    arg: str = ""

    def __str__(self) -> str:
        return "none" if self.kind == "none" else f"{self.kind}:{self.arg}"


NONE = Action("none")


_PUNCT = {ord(c): " " for c in string.punctuation if c != "'"}


def normalize_phrase(text: str) -> str:
    """'Scroll  Down!' -> 'scroll down' (what voice triggers are matched on)."""
    t = text.lower().translate(_PUNCT)
    return " ".join(t.split())


def normalize_cursor_arg(arg: str) -> str:
    """``Monitor_2`` / ``monitor2`` / ``monitor two`` -> ``monitor 2``; ``previous_monitor`` -> ``prev_monitor``."""
    a = " ".join(arg.strip().lower().replace("-", " ").split())
    m = re.fullmatch(r"(?:monitor|screen|display)[ _]?(\w+)", a)
    if m and m.group(1) not in ("next", "prev", "previous", "primary"):
        n = m.group(1)
        if n in _NUMBER_WORDS:
            n = str(_NUMBER_WORDS.index(n))
        return f"monitor {n}"
    a = a.replace(" ", "_")
    return _CURSOR_ALIASES.get(a, a)


def phrase_variants(phrase: str) -> list[str]:
    """Other spellings a recogniser may produce for a normalised phrase: single digits <-> words
    (``monitor 2`` <-> ``monitor two``), so "monitor two" matches whichever way Whisper writes it."""
    words = phrase.split()
    swapped = []
    for w in words:
        if w.isdigit() and len(w) == 1:
            swapped.append(_NUMBER_WORDS[int(w)])
        elif w in _NUMBER_WORDS:
            swapped.append(str(_NUMBER_WORDS.index(w)))
        else:
            swapped.append(w)
    alt = " ".join(swapped)
    return [alt] if alt != phrase else []


def parse_action(value: Any, *, allow_shell: bool = False) -> Action:
    """Parse ``"key:ctrl+tab"`` or ``{"key": "ctrl+tab"}``; raises ValueError with a clear message."""
    if value is None:
        return NONE
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"action mapping must have exactly one key (kind: argument), got {value!r}")
        ((k, v),) = value.items()
        value = f"{k}:{'' if v is None else v}"
    if not isinstance(value, str):
        raise ValueError(f"action must be a string like 'key:ctrl+tab', got {value!r}")
    text = value.strip()
    if text.lower() in ("none", "", "off", "disabled"):
        return NONE
    kind, sep, arg = text.partition(":")
    kind = kind.strip().lower()
    arg = arg.strip()
    if kind not in ACTION_ARGS:
        hint = difflib.get_close_matches(kind, ACTION_KINDS, n=1)
        sug = f" (did you mean {hint[0]!r}?)" if hint else ""
        raise ValueError(f"unknown action {text!r}{sug}; kinds: {', '.join(ACTION_KINDS)}")
    if kind == "none":
        return NONE
    allowed = ACTION_ARGS[kind]
    if kind == "cursor" and arg:
        arg = normalize_cursor_arg(arg)
    if not sep or not arg:
        if allowed:
            raise ValueError(f"action {kind!r} needs an argument: {kind}:{'|'.join(allowed)}")
        raise ValueError(f"action {kind!r} needs an argument, e.g. {kind}:...")
    if allowed is not None:
        arg = arg.lower()
        if arg not in allowed:
            raise ValueError(f"{kind}:{arg} is not valid; use one of {kind}:{'|'.join(allowed)}")
    if kind == "key":
        try:
            arg = canonical(arg)
        except KeyComboError as exc:
            raise ValueError(str(exc)) from None
    if kind == "shell" and not allow_shell:
        raise ValueError("shell actions are disabled for safety; set bindings.allow_shell: true to enable them")
    return Action(kind, arg)


@dataclass
class Profile:
    name: str
    apps: list[str] = field(default_factory=list)
    gestures: dict[str, Action] = field(default_factory=dict)
    voice: dict[str, Action] = field(default_factory=dict)
    face: dict[str, Action] = field(default_factory=dict)
    hotkeys: dict[str, Action] = field(default_factory=dict)

    def table(self, category: str) -> dict[str, Action]:
        if category not in CATEGORIES:
            raise ValueError(f"unknown trigger category {category!r}; use {', '.join(CATEGORIES)}")
        tbl: dict[str, Action] = getattr(self, category)
        return tbl


def default_profile() -> Profile:
    """Matches the behaviour before bindings existed, plus the always-on voice commands."""
    a = parse_action
    return Profile(
        "default",
        gestures={
            "point": a("mouse:move"),
            "pinch": a("mouse:click"),
            "double_pinch": a("mouse:double_click"),
            "pinch_hold": a("mouse:drag"),
            "two_finger_scroll": a("mouse:scroll"),
            "open_palm": a("pause:toggle"),
            "swipe_left": a("key:left"),
            "swipe_right": a("key:right"),
            "push": a("mouse:click"),  # only detected with depth (multi-camera, or hands.multicam.push_monocular)
            "dwell": a("mouse:click"),  # only detected when hands.dwell.enabled
        },
        voice={
            "click": a("mouse:click"),
            "double click": a("mouse:double_click"),
            "right click": a("mouse:right_click"),
            "scroll up": a("mouse:scroll_up"),
            "scroll down": a("mouse:scroll_down"),
            "go back": a("key:alt+left"),
            "go forward": a("key:alt+right"),
            "open browser": a("launch:https://www.google.com"),
            "stop listening": a("dictation:stop"),
            "start listening": a("dictation:start"),
            "command mode": a("voice:command_mode"),
            "dictation mode": a("voice:dictation_mode"),
            "pause gestures": a("pause:on"),
            "resume gestures": a("pause:off"),
            "next monitor": a("cursor:next_monitor"),
            "previous monitor": a("cursor:prev_monitor"),
            "monitor one": a("cursor:monitor 1"),
            "monitor two": a("cursor:monitor 2"),
            "monitor three": a("cursor:monitor 3"),
        },
        face={
            "brow_raise": a("dictation:toggle"),
            "mouth_open": NONE,
            "wink_left": a("mouse:click"),  # only detected when face.wink_trigger
            "wink_right": a("mouse:right_click"),
        },
        hotkeys={"ptt": a("dictation:toggle"), "mute": a("mic:toggle")},
    )


@dataclass
class BindingsConfig:
    active_profile: str = "auto"  # "auto" = pick by focused app, else a profile name
    allow_shell: bool = False
    custom_gestures: list[str] = field(default_factory=list)
    discrete_cooldown: float = 0.5  # min seconds between repeats of a discrete action on a continuous gesture
    profiles: dict[str, Profile] = field(default_factory=lambda: {"default": default_profile()})

    @property
    def default(self) -> Profile:
        return self.profiles["default"]

    # ------------------------------------------------------------ resolution
    def profile_for_app(self, process: str | None) -> str:
        """Name of the profile to use for the focused process (``active_profile`` wins unless auto)."""
        if self.active_profile != "auto" and self.active_profile in self.profiles:
            return self.active_profile
        key = app_key(process)
        if key:
            for name, prof in self.profiles.items():
                if name != "default" and key in {app_key(a) for a in prof.apps}:
                    return name
        return "default"

    def resolve(self, category: str, trigger: str, profile: str = "default") -> Action:
        """Action for ``trigger`` under ``profile`` (falls back to default, then none)."""
        if category == "voice":
            trigger = normalize_phrase(trigger)
        prof = self.profiles.get(profile)
        for t in [trigger, *(phrase_variants(trigger) if category == "voice" else [])]:
            if prof is not None and t in prof.table(category):
                return prof.table(category)[t]
            if t in self.default.table(category):
                return self.default.table(category)[t]
        return NONE

    def effective(self, profile: str = "default") -> dict[str, dict[str, Action]]:
        out: dict[str, dict[str, Action]] = {}
        for cat in CATEGORIES:
            merged = dict(self.default.table(cat))
            if profile != "default" and profile in self.profiles:
                merged.update(self.profiles[profile].table(cat))
            out[cat] = merged
        return out

    def voice_phrases(self, profile: str = "default") -> dict[str, Action]:
        """Every bound phrase, plus digit/word spellings of it (``monitor 2`` for ``monitor two``)."""
        phrases = dict(self.effective(profile)["voice"])
        for p, act in list(phrases.items()):
            for alt in phrase_variants(p):
                phrases.setdefault(alt, act)
        return phrases

    # ----------------------------------------------------------------- edits
    def set(self, trigger_id: str, action: str | Action, profile: str = "default") -> Action:
        cat, trig = split_trigger(trigger_id)
        if profile not in self.profiles:
            raise ValueError(f"no profile named {profile!r}; create it with `away profile new {profile}`")
        act = action if isinstance(action, Action) else parse_action(action, allow_shell=self.allow_shell)
        trig = _check_trigger(cat, trig, self.custom_gestures)
        _check_placement(cat, trig, act)
        self.profiles[profile].table(cat)[trig] = act
        return act

    def unset(self, trigger_id: str, profile: str = "default") -> None:
        cat, trig = split_trigger(trigger_id)
        if cat == "voice":
            trig = normalize_phrase(trig)
        elif cat == "hotkeys" and trig not in HOTKEY_SLOTS:
            trig = canonical(trig)
        tbl = self.profiles[profile].table(cat)
        if profile == "default":
            dflt = default_profile().table(cat)
            if trig in dflt:
                tbl[trig] = dflt[trig]
                return
        tbl.pop(trig, None)

    def reset(self, profile: str | None = None) -> None:
        if profile in (None, "default"):
            self.profiles["default"] = default_profile()
            if profile is None:
                for name in [n for n in self.profiles if n != "default"]:
                    self.profiles[name] = Profile(name, apps=list(self.profiles[name].apps))
        else:
            if profile not in self.profiles:
                raise ValueError(f"no profile named {profile!r}")
            self.profiles[profile] = Profile(profile, apps=list(self.profiles[profile].apps))

    def new_profile(self, name: str, apps: list[str] | None = None, copy_from: str | None = None) -> Profile:
        if not PROFILE_RE.match(name or ""):
            raise ValueError(f"profile name {name!r} must be letters, digits, '_', '-' or '.' (max 40)")
        if name in self.profiles or name == "auto":
            raise ValueError(f"profile {name!r} already exists" if name != "auto" else "'auto' is reserved")
        if copy_from is not None:
            if copy_from not in self.profiles:
                raise ValueError(f"no profile named {copy_from!r} to copy from")
            p = copy.deepcopy(self.profiles[copy_from])
            p.name, p.apps = name, list(apps or [])
        else:
            p = Profile(name, apps=list(apps or []))
        self.profiles[name] = p
        return p

    def to_dict(self) -> dict[str, Any]:
        profiles: dict[str, Any] = {}
        for name, p in self.profiles.items():
            d: dict[str, Any] = {}
            if name != "default":
                d["apps"] = list(p.apps)
            for cat in CATEGORIES:
                tbl = p.table(cat)
                if tbl or name == "default":
                    d[cat] = {k: str(v) for k, v in tbl.items()}
            profiles[name] = d
        return {
            "active_profile": self.active_profile,
            "allow_shell": self.allow_shell,
            "custom_gestures": list(self.custom_gestures),
            "discrete_cooldown": self.discrete_cooldown,
            "profiles": profiles,
        }


def app_key(process: str | None) -> str:
    if not process:
        return ""
    p = process.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    return p[:-4] if p.endswith(".exe") else p


def split_trigger(trigger_id: str) -> tuple[str, str]:
    """``gesture:pinch`` -> (``gestures``, ``pinch``). Also accepts the plural category."""
    prefix, sep, name = trigger_id.partition(":")
    prefix = prefix.strip().lower()
    cat = PREFIX_CATEGORY.get(prefix) or (prefix if prefix in CATEGORIES else None)
    if not sep or cat is None or not name.strip():
        raise ValueError(
            f"trigger {trigger_id!r} must look like gesture:pinch, voice:scroll down, "
            "face:brow_raise or hotkey:ctrl+alt+k"
        )
    return cat, name.strip()


def _check_trigger(cat: str, trig: str, custom: list[str]) -> str:
    if cat == "gestures":
        if trig not in GESTURE_TRIGGERS and trig not in custom:
            pool = [*GESTURE_TRIGGERS, *custom]
            hint = difflib.get_close_matches(trig, pool, n=1)
            sug = f" (did you mean {hint[0]!r}?)" if hint else ""
            raise ValueError(
                f"unknown gesture {trig!r}{sug}; built-ins: {', '.join(GESTURE_TRIGGERS)}; "
                "custom gestures must be listed in bindings.custom_gestures (away record adds them)"
            )
        return trig
    if cat == "face":
        if trig not in FACE_TRIGGERS:
            raise ValueError(f"unknown face trigger {trig!r}; use {', '.join(FACE_TRIGGERS)}")
        return trig
    if cat == "voice":
        p = normalize_phrase(trig)
        if not p:
            raise ValueError(f"voice phrase {trig!r} is empty after normalising")
        return p
    if trig in HOTKEY_SLOTS:
        return trig
    try:
        return canonical(trig)
    except KeyComboError as exc:
        raise ValueError(f"hotkey {trig!r}: {exc}") from None


def _check_placement(cat: str, trig: str, act: Action) -> None:
    if act.kind == "mouse" and act.arg in CONTINUOUS_ONLY:
        need = CONTINUOUS_ONLY[act.arg]
        if not (cat == "gestures" and trig == need):
            raise ValueError(f"mouse:{act.arg} is continuous and can only be bound to gesture {need!r}")


def parse_bindings(data: Any, prefix: str = "bindings") -> BindingsConfig:
    """Build a :class:`BindingsConfig` from YAML data. Missing defaults are filled in; all errors are collected."""
    errors: list[str] = []
    cfg = BindingsConfig()
    if data is None:
        return cfg
    if not isinstance(data, dict):
        raise BindingsError([f"{prefix}: must be a mapping"])
    known = {"active_profile", "allow_shell", "custom_gestures", "discrete_cooldown", "profiles"}
    for k in data:
        if k not in known:
            hint = difflib.get_close_matches(str(k), sorted(known), n=1)
            errors.append(f"{prefix}.{k}: unknown key" + (f" (did you mean {hint[0]!r}?)" if hint else ""))
    shell = data.get("allow_shell", False)
    if not isinstance(shell, bool):
        errors.append(f"{prefix}.allow_shell: must be true or false")
        shell = False
    cfg.allow_shell = shell
    cd = data.get("discrete_cooldown", cfg.discrete_cooldown)
    if isinstance(cd, bool) or not isinstance(cd, (int, float)) or not 0.0 <= float(cd) <= 10.0:
        errors.append(f"{prefix}.discrete_cooldown: must be a number of seconds in [0, 10]")
    else:
        cfg.discrete_cooldown = float(cd)
    custom = data.get("custom_gestures") or []
    if not isinstance(custom, list):
        errors.append(f"{prefix}.custom_gestures: must be a list of names")
        custom = []
    for c in custom:
        if not isinstance(c, str) or not LABEL_RE.match(c):
            errors.append(f"{prefix}.custom_gestures: {c!r} must be lower_snake_case (a-z, 0-9, _), max 32 chars")
        elif c in GESTURE_TRIGGERS:
            errors.append(f"{prefix}.custom_gestures: {c!r} is already a built-in gesture")
        elif c not in cfg.custom_gestures:
            cfg.custom_gestures.append(c)
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        errors.append(f"{prefix}.profiles: must be a mapping of profile name -> bindings")
        profiles = {}
    for name, pdata in profiles.items():
        ppath = f"{prefix}.profiles.{name}"
        name = str(name)
        if not PROFILE_RE.match(name) or name == "auto":
            errors.append(f"{ppath}: invalid profile name (letters, digits, '_', '-', '.'; 'auto' is reserved)")
            continue
        if pdata is None:
            pdata = {}
        if not isinstance(pdata, dict):
            errors.append(f"{ppath}: must be a mapping")
            continue
        prof = cfg.profiles["default"] if name == "default" else Profile(name)
        for k in pdata:
            if k not in (*CATEGORIES, "apps"):
                hint = difflib.get_close_matches(str(k), [*CATEGORIES, "apps"], n=1)
                errors.append(f"{ppath}.{k}: unknown key" + (f" (did you mean {hint[0]!r}?)" if hint else ""))
        apps = pdata.get("apps") or []
        if name == "default" and apps:
            errors.append(f"{ppath}.apps: the default profile applies to every app; remove 'apps'")
        elif not isinstance(apps, list) or not all(isinstance(a, str) and a.strip() for a in apps):
            errors.append(f"{ppath}.apps: must be a list of process names, e.g. [chrome, firefox.exe]")
        else:
            prof.apps = [a.strip() for a in apps]
        for cat in CATEGORIES:
            tbl = pdata.get(cat)
            if tbl is None:
                continue
            if not isinstance(tbl, dict):
                errors.append(f"{ppath}.{cat}: must be a mapping of trigger -> action")
                continue
            for trig, raw in tbl.items():
                tpath = f"{ppath}.{cat}.{trig}"
                try:
                    t = _check_trigger(cat, str(trig), cfg.custom_gestures)
                    act = parse_action(raw, allow_shell=cfg.allow_shell)
                    _check_placement(cat, t, act)
                except ValueError as exc:
                    errors.append(f"{tpath}: {exc}")
                    continue
                prof.table(cat)[t] = act
        cfg.profiles[name] = prof
    ap = data.get("active_profile", "auto")
    if not isinstance(ap, str) or (ap != "auto" and ap not in cfg.profiles):
        errors.append(f"{prefix}.active_profile: {ap!r} is not 'auto' or a defined profile ({', '.join(cfg.profiles)})")
    else:
        cfg.active_profile = ap
    seen: dict[str, str] = {}
    for name, prof in cfg.profiles.items():
        for a in prof.apps:
            k = app_key(a)
            if k in seen and seen[k] != name:
                errors.append(f"{prefix}.profiles.{name}.apps: {a!r} is already claimed by profile {seen[k]!r}")
            seen[k] = name
    if errors:
        raise BindingsError(errors)
    return cfg


def list_rows(cfg: BindingsConfig, profile: str = "default") -> list[tuple[str, str, str]]:
    """``[(trigger_id, action, source)]`` where source is the profile that defines it."""
    rows: list[tuple[str, str, str]] = []
    prof = cfg.profiles.get(profile)
    eff = cfg.effective(profile)
    for cat in CATEGORIES:
        names: list[str] = list(eff[cat])
        if cat == "gestures":
            names = [*GESTURE_TRIGGERS, *cfg.custom_gestures] + [n for n in names if n not in GESTURE_TRIGGERS]
            names = list(dict.fromkeys(names))
        if cat == "face":
            names = list(dict.fromkeys([*FACE_TRIGGERS, *names]))
        for n in names:
            src = profile if prof is not None and n in prof.table(cat) else "default"
            rows.append((f"{CATEGORY_PREFIX[cat]}:{n}", str(eff[cat].get(n, NONE)), src))
    return rows


def merge_bindings(current: BindingsConfig, incoming: BindingsConfig, raw: Any) -> BindingsConfig:
    """Merge an imported file into ``current``: only what the file explicitly sets is taken over.

    ``incoming`` is ``parse_bindings(raw)`` (already validated); ``raw`` is the YAML mapping it came from,
    used to tell explicit entries from the defaults that parsing filled in.
    """
    out = copy.deepcopy(current)
    raw = raw if isinstance(raw, dict) else {}
    for c in incoming.custom_gestures:
        if c not in out.custom_gestures:
            out.custom_gestures.append(c)
    out.allow_shell = out.allow_shell or incoming.allow_shell
    for name, pdata in (raw.get("profiles") or {}).items():
        name = str(name)
        src = incoming.profiles[name]
        if name not in out.profiles:
            out.profiles[name] = Profile(name)
        dst = out.profiles[name]
        if isinstance(pdata, dict) and "apps" in pdata and name != "default":
            dst.apps = list(src.apps)
        for cat in CATEGORIES:
            tbl = pdata.get(cat) if isinstance(pdata, dict) else None
            for trig in tbl or {}:
                key = _check_trigger(cat, str(trig), incoming.custom_gestures)
                dst.table(cat)[key] = src.table(cat)[key]
    for key in ("active_profile", "discrete_cooldown"):
        if key in raw:
            setattr(out, key, getattr(incoming, key))
    # re-validate the merged result (e.g. two profiles claiming one app)
    return parse_bindings(out.to_dict())


def load_bindings_file(path: Any) -> tuple[BindingsConfig, Any]:
    """Read a YAML file holding either ``bindings:`` or the bindings mapping itself. Returns (parsed, raw)."""
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and "bindings" in raw:
        raw = raw["bindings"]
    return parse_bindings(raw), raw
