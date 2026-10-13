"""What agents may change (the "Change my settings" permission), as a registry of domains.

Every domain knows how to list its items, read one, and apply ``set`` / ``create`` / ``delete`` (and ``apply``
for presets) to an :class:`~away.config.AppConfig`, validated with the same range checks the Settings window
uses. Each applied operation returns its inverse, which is how every change can be undone.

Built-in domains on ``main``:

* ``settings`` - every slider in :func:`away.config.tunable_ranges` plus a few safe choices/switches
* ``zones`` - the absolute-mode active region and the dead zones
* ``bindings`` - trigger -> action, per profile (never ``shell``, never launching programs; URLs are fine)
* ``custom_gestures`` - enable/disable a recorded custom gesture, the model threshold and the repeat cooldown
* ``profiles`` - per-app profiles: create/delete, which apps, which is active, per-profile tuning
* ``presets`` - named snapshots of the sliders (saved in the data folder), applied as ordinary setting changes

Since the v0.1.20 integration (:mod:`away.agentlink.integrations`):

* ``gestures`` - Gesture Studio, per gesture: enabled / how sure / hold for / rest time
* ``macros`` - recorded shortcuts: list (no typed text), create, delete (undoable); run one with ``macro:<name>``
* ``settings`` also covers the smart pointer, attention gate, arm-fatigue and Studio zone switches and sliders

Other modules can still add a domain with :func:`register`.

Never reachable through here: ``bindings.allow_shell``, ``agent_link.*`` (the agents' own permissions),
``updates.*``, ``models_dir``, file paths, hotkeys that could lock the user out, and anything outside the config.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from away import config as cfgmod

MERGE_TODO: tuple[str, ...] = (
    "blocked zones from an agent ('ignore my hand in this part of the camera'): Gesture Studio has zone maps; "
    "a 'zones' op that edits them is not exposed yet",
)

FORBIDDEN_PREFIXES = (
    "bindings.allow_shell",
    "agent_link.",
    "updates.",
    "models_dir",
    "telemetry",
    "speech.enabled",  # "Let agents talk to me" is the person's switch, never an agent's
)
CHOICES: dict[str, tuple[str, ...]] = {
    "hands.cursor.mode": ("relative", "absolute"),
    "hands.cursor.curve": ("power", "sigmoid"),
    "hands.roles.mode": ("auto", "right", "left", "any"),
    "hands.cursor.smoothing": ("one_euro", "predictive"),  # smart pointer: "Smoother cursor"
    "zones.anchor": ("palm", "index_tip", "wrist"),  # Gesture Studio zones: which point of the hand is tested
    "speech.rate": cfgmod.SPEECH_RATES,  # v0.1.21: how fast agents speak to you
    "speech.volume": cfgmod.SPEECH_VOLUMES,
}
SWITCHES = (
    "hands.dwell.enabled",
    "hands.cursor.click_freeze",
    "hands.gestures.autoscroll_enabled",
    "hands.cursor.autocenter.enabled",
    "hands.cursor.magnet.enabled",  # smart pointer: "Snap to buttons"
    "voice.click_by_name",  # smart pointer: "Click things by saying their name"
    "attention.enabled",  # "Only respond when I'm looking at the screen"
    "attention.require_gaze",
    "attention.gate_voice",
    "comfort.fatigue_switch",  # "Switch to head control when my arm is tired"
    "comfort.fatigue_tips",
    "comfort.break_reminders",
    "zones.enabled",  # Gesture Studio zones
    "zones.block_cursor",
    "speech.read_messages_aloud",  # v0.1.21 (speech.enabled, "Let agents talk to me", stays the person's own)
)
ZONE_KEYS = ("hands.cursor.active_region",)
SAFE_ACTION_KINDS = (
    "none",
    "mouse",
    "key",
    "type",
    "dictation",
    "mic",
    "voice",
    "pause",
    "hands",
    "cursor",
    "agent",
    "macro",  # run a recorded shortcut (only clicks, keys, text, scrolls and waits; never programs)
    "macro_control",
)


class ChangeError(ValueError):
    pass


@dataclass
class Op:
    """One change. ``domain``/``op``/``key``/``value`` (+ ``profile`` for bindings)."""

    domain: str
    op: str
    key: str
    value: Any = None
    profile: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "op": self.op, "key": self.key, "value": self.value, "profile": self.profile}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Op:
        return Op(
            str(d.get("domain") or "settings"),
            str(d.get("op") or "set"),
            str(d.get("key") or ""),
            d.get("value"),
            str(d.get("profile") or "default"),
        )


@dataclass
class Applied:
    op: Op
    before: Any
    after: Any
    inverse: list[Op]
    text: str  # plain language: "Cursor speed: 2.2 -> 1.8"
    low_risk: bool = False


@dataclass
class Domain:
    name: str
    title: str
    list: Callable[[Any], list[dict[str, Any]]]
    apply: Callable[[Any, Op], Applied] | None
    available: bool = True
    note: str = ""
    ops: tuple[str, ...] = ("set",)
    extra: dict[str, Any] = field(default_factory=dict)


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


def _forbid(key: str) -> None:
    if any(key == p.rstrip(".") or key.startswith(p) for p in FORBIDDEN_PREFIXES):
        raise ChangeError(f"{key} can never be changed by an agent")


def ranges(cfg: Any) -> dict[str, tuple[float, float, Any]]:
    return {p: (float(lo), float(hi), v) for p, v, lo, hi in cfgmod.tunable_ranges(cfg)}


# v0.1.21: plain titles for settings that are not sliders (the speech section)
TITLES = {
    "speech.rate": "How fast agents speak (slow, normal, fast)",
    "speech.volume": "How loud agents speak (quiet, normal, loud)",
    "speech.read_messages_aloud": "Read every agent message aloud",
}


def title_for(key: str) -> str:
    if key in TITLES:
        return TITLES[key]
    try:
        from away.app.tunables import describe

        return describe(key).title or key
    except Exception:
        return key


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return "none" if v is None else str(v)


# ---------------------------------------------------------------------------- settings / zones
def _coerce_number(key: str, value: Any, lo: float, hi: float, current: Any) -> float | int:
    if isinstance(value, bool):
        raise ChangeError(f"{key} needs a number")
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ChangeError(f"{key} needs a number, got {value!r}") from None
    if not lo <= num <= hi:
        raise ChangeError(f"{title_for(key)} must be between {lo:g} and {hi:g} (you asked for {num:g})")
    return round(num) if isinstance(current, int) and not isinstance(current, bool) else num


def settings_list(cfg: Any) -> list[dict[str, Any]]:
    out = []
    for key, (lo, hi, v) in ranges(cfg).items():
        out.append({"key": key, "title": title_for(key), "value": v, "min": lo, "max": hi, "kind": "number"})
    for key, ch in CHOICES.items():
        out.append(
            {"key": key, "title": title_for(key), "value": _get(cfg, key), "choices": list(ch), "kind": "choice"}
        )
    for key in SWITCHES:
        out.append({"key": key, "title": title_for(key), "value": _get(cfg, key), "kind": "switch"})
    return out


def settings_apply(cfg: Any, op: Op) -> Applied:
    if op.op != "set":
        raise ChangeError("settings only support 'set'")
    key = op.key
    _forbid(key)
    rng = ranges(cfg)
    before = _get(cfg, key) if key in rng or key in CHOICES or key in SWITCHES else None
    low = False
    if key in rng:
        lo, hi, cur = rng[key]
        new: Any = _coerce_number(key, op.value, lo, hi, cur)
        low = True
    elif key in CHOICES:
        new = str(op.value)
        if new not in CHOICES[key]:
            raise ChangeError(f"{title_for(key)} must be one of {', '.join(CHOICES[key])}")
    elif key in SWITCHES:
        if not isinstance(op.value, bool):
            raise ChangeError(f"{title_for(key)} is on/off (true/false)")
        new = op.value
    else:
        raise ChangeError(f"{key} is not a setting agents can change (see list_settings)")
    _set(cfg, key, new)
    return Applied(
        op, before, new, [Op("settings", "set", key, before)], f"{title_for(key)}: {_fmt(before)} -> {_fmt(new)}", low
    )


def zones_list(cfg: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "key": "hands.cursor.active_region",
            "title": "Active area of the camera (absolute cursor mode)",
            "value": list(cfg.hands.cursor.active_region),
            "kind": "region",
            "help": "[left, top, right, bottom], fractions of the camera image",
        }
    ]
    for key, (lo, hi, v) in ranges(cfg).items():
        if "dead_zone" in key:
            out.append({"key": key, "title": title_for(key), "value": v, "min": lo, "max": hi, "kind": "number"})
    return out


def zones_apply(cfg: Any, op: Op) -> Applied:
    if op.key in ZONE_KEYS:
        if op.op != "set":
            raise ChangeError("zones only support 'set'")
        try:
            box = tuple(float(x) for x in op.value)
        except (TypeError, ValueError):
            raise ChangeError("a region is [left, top, right, bottom] between 0 and 1") from None
        if len(box) != 4 or not (0.0 <= box[0] < box[2] <= 1.0 and 0.0 <= box[1] < box[3] <= 1.0):
            raise ChangeError("a region is [left, top, right, bottom] with 0 <= left < right <= 1, same for top/bottom")
        if (box[2] - box[0]) < 0.1 or (box[3] - box[1]) < 0.1:
            raise ChangeError("the active area must be at least 10% of the camera image wide and tall")
        before = list(_get(cfg, op.key))
        _set(cfg, op.key, box)
        return Applied(
            op,
            before,
            list(box),
            [Op("zones", "set", op.key, before)],
            f"Active camera area: {_fmt(before)} -> {_fmt(box)}",
            True,
        )
    if "dead_zone" in op.key:
        a = settings_apply(cfg, Op("settings", op.op, op.key, op.value))
        a.op, a.inverse = op, [Op("zones", "set", op.key, a.before)]
        return a
    raise ChangeError(f"{op.key} is not a zone (see list_settings zones)")


# ---------------------------------------------------------------------------- bindings
def _check_action(value: Any) -> str:
    from away.bindings import parse_action

    text = str(value or "none").strip()
    act = parse_action(text, allow_shell=False)
    if act.kind == "launch":
        if not act.arg.lower().startswith(("http://", "https://")):
            raise ChangeError("agents can bind web links (https://...) but not programs or commands")
    elif act.kind not in SAFE_ACTION_KINDS:
        raise ChangeError(f"agents cannot bind '{act.kind}' actions")
    return str(act)


def bindings_list(cfg: Any) -> list[dict[str, Any]]:
    out = []
    for name, prof in cfg.bindings.profiles.items():
        for cat in ("gestures", "voice", "face", "hotkeys"):
            for trig, act in prof.table(cat).items():
                out.append({"profile": name, "key": f"{_prefix(cat)}:{trig}", "value": str(act)})
    return out


def _prefix(cat: str) -> str:
    from away.bindings import CATEGORY_PREFIX

    return CATEGORY_PREFIX[cat]


def bindings_apply(cfg: Any, op: Op) -> Applied:
    from away.bindings import split_trigger

    b = cfg.bindings
    if op.profile not in b.profiles:
        raise ChangeError(f"there is no profile named {op.profile!r}")
    cat, trig = split_trigger(op.key)
    if cat == "hotkeys":
        raise ChangeError("agents cannot change hotkeys (it could lock you out of your own controls)")
    before = None
    tbl = b.profiles[op.profile].table(cat)
    try:
        before = (
            str(b.resolve(cat, trig, op.profile))
            if op.profile == "default"
            else (str(tbl[trig]) if trig in tbl else None)
        )
    except Exception:
        before = None
    own_before = trig in tbl
    if op.op == "set":
        action = _check_action(op.value)
        try:
            act = b.set(op.key, action, op.profile)
        except ValueError as exc:
            raise ChangeError(str(exc)) from None
        inverse = (
            [Op("bindings", "set", op.key, before, op.profile)]
            if own_before and before
            else [Op("bindings", "delete", op.key, None, op.profile)]
        )
        return Applied(op, before, str(act), inverse, f"{op.key} ({op.profile}): {_fmt(before)} -> {act}")
    if op.op == "delete":
        b.unset(op.key, op.profile)
        after = None
        try:
            after = str(b.resolve(cat, trig, op.profile))
        except Exception:
            after = None
        inverse = [Op("bindings", "set", op.key, before, op.profile)] if before and before != "none" else []
        return Applied(op, before, after, inverse, f"{op.key} ({op.profile}): {_fmt(before)} -> {_fmt(after)}")
    raise ChangeError("bindings support 'set' and 'delete'")


# ---------------------------------------------------------------------------- custom gestures
def custom_list(cfg: Any) -> list[dict[str, Any]]:
    return [{"key": name, "kind": "custom_gesture", "enabled": True} for name in cfg.bindings.custom_gestures] + [
        {
            "key": "threshold",
            "title": "Custom gesture confidence needed",
            "value": cfg.hands.gestures.model_min_conf,
            "min": 0.3,
            "max": 0.99,
            "kind": "number",
        },
        {
            "key": "cooldown",
            "title": "Seconds between repeats",
            "value": cfg.bindings.discrete_cooldown,
            "min": 0.0,
            "max": 5.0,
            "kind": "number",
        },
    ]


def custom_apply(cfg: Any, op: Op) -> Applied:
    from away.bindings import LABEL_RE

    b = cfg.bindings
    if op.op == "set" and op.key in ("threshold", "cooldown"):
        path = "hands.gestures.model_min_conf" if op.key == "threshold" else "bindings.discrete_cooldown"
        lo, hi = (0.3, 0.99) if op.key == "threshold" else (0.0, 5.0)
        before = _get(cfg, path)
        new = _coerce_number(path, op.value, lo, hi, before)
        _set(cfg, path, float(new))
        label = "Custom gesture confidence" if op.key == "threshold" else "Repeat cooldown"
        return Applied(
            op,
            before,
            new,
            [Op("custom_gestures", "set", op.key, before)],
            f"{label}: {_fmt(before)} -> {_fmt(new)}",
            True,
        )
    name = op.key
    if not LABEL_RE.match(name):
        raise ChangeError("a custom gesture name is lower_snake_case")
    if op.op == "create":
        if name in b.custom_gestures:
            raise ChangeError(f"{name} is already enabled")
        b.custom_gestures.append(name)
        return Applied(op, False, True, [Op("custom_gestures", "delete", name)], f"Custom gesture {name}: off -> on")
    if op.op == "delete":
        if name not in b.custom_gestures:
            raise ChangeError(f"{name} is not enabled")
        removed: list[Op] = []
        for pname, prof in b.profiles.items():
            if name in prof.gestures:
                removed.append(Op("bindings", "set", f"gesture:{name}", str(prof.gestures.pop(name)), pname))
        b.custom_gestures.remove(name)
        return Applied(
            op, True, False, [Op("custom_gestures", "create", name), *removed], f"Custom gesture {name}: on -> off"
        )
    raise ChangeError("custom gestures support create (enable), delete (disable) and set threshold/cooldown")


# ---------------------------------------------------------------------------- profiles
def profiles_list(cfg: Any) -> list[dict[str, Any]]:
    b = cfg.bindings
    return [{"key": "active", "value": b.active_profile, "kind": "choice", "choices": ["auto", *b.profiles]}] + [
        {"key": name, "apps": list(p.apps), "tuning": dict(p.tuning), "kind": "profile"}
        for name, p in b.profiles.items()
    ]


def profiles_apply(cfg: Any, op: Op) -> Applied:
    b = cfg.bindings
    if op.op == "create":
        spec = op.value if isinstance(op.value, dict) else {}
        try:
            b.new_profile(op.key, [str(a) for a in spec.get("apps") or []], spec.get("copy_from"))
        except ValueError as exc:
            raise ChangeError(str(exc)) from None
        return Applied(
            op,
            None,
            op.key,
            [Op("profiles", "delete", op.key)],
            f"New profile {op.key} for {', '.join(spec.get('apps') or []) or 'no apps yet'}",
        )
    if op.op == "delete":
        if op.key == "default" or op.key not in b.profiles:
            raise ChangeError("that profile cannot be deleted" if op.key == "default" else f"no profile {op.key!r}")
        snap = copy.deepcopy(b.profiles[op.key])
        del b.profiles[op.key]
        if b.active_profile == op.key:
            b.active_profile = "auto"
        restore = Op(
            "profiles",
            "restore",
            op.key,
            {
                "apps": snap.apps,
                "tuning": snap.tuning,
                "bindings": {
                    c: {k: str(v) for k, v in snap.table(c).items()} for c in ("gestures", "voice", "face", "hotkeys")
                },
            },
        )
        return Applied(op, op.key, None, [restore], f"Deleted profile {op.key}")
    if op.op == "restore":  # only produced by undo
        spec = op.value or {}
        b.new_profile(op.key, list(spec.get("apps") or []))
        b.profiles[op.key].tuning = dict(spec.get("tuning") or {})
        for cat, tbl in (spec.get("bindings") or {}).items():
            for trig, act in tbl.items():
                b.set(f"{_prefix(cat)}:{trig}", act, op.key)
        return Applied(op, None, op.key, [Op("profiles", "delete", op.key)], f"Restored profile {op.key}")
    if op.op != "set":
        raise ChangeError("profiles support create, delete and set")
    if op.key == "active":
        name = str(op.value)
        if name != "auto" and name not in b.profiles:
            raise ChangeError(f"no profile {name!r}")
        before = b.active_profile
        b.active_profile = name
        return Applied(
            op, before, name, [Op("profiles", "set", "active", before)], f"Profile in use: {before} -> {name}"
        )
    m = re.match(r"^([^.]+)\.(apps|tuning\.(.+))$", op.key)
    if not m or m.group(1) not in b.profiles:
        raise ChangeError("set 'active', '<profile>.apps' or '<profile>.tuning.<setting>'")
    prof = b.profiles[m.group(1)]
    if m.group(2) == "apps":
        before = list(prof.apps)
        prof.apps = [str(a) for a in (op.value or [])]
        return Applied(
            op,
            before,
            list(prof.apps),
            [Op("profiles", "set", op.key, before)],
            f"Apps for {prof.name}: {_fmt(before)} -> {_fmt(prof.apps)}",
        )
    path = m.group(3)
    _forbid(path)
    rng = ranges(cfg)
    from away.bindings import PROFILE_TUNING_PREFIXES

    if path not in rng or not path.startswith(PROFILE_TUNING_PREFIXES):
        raise ChangeError(f"{path} cannot be tuned per app")
    before_v = prof.tuning.get(path)
    if op.value is None:
        prof.tuning.pop(path, None)
        new_v = None
    else:
        lo, hi, cur = rng[path]
        new_v = float(_coerce_number(path, op.value, lo, hi, cur))
        prof.tuning[path] = new_v
    return Applied(
        op,
        before_v,
        new_v,
        [Op("profiles", "set", op.key, before_v)],
        f"{title_for(path)} in {prof.name}: {_fmt(before_v)} -> {_fmt(new_v)}",
        True,
    )


# ---------------------------------------------------------------------------- presets
class PresetStore:
    """Named snapshots of the sliders in ``<data>/agentlink/presets/<name>.json`` (never inside the config)."""

    NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}$")

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    def _path(self, name: str) -> Path:
        if not self.NAME.match(name or ""):
            raise ChangeError("a preset name is letters, digits, spaces, '_', '-' or '.' (max 40)")
        return self.folder / f"{name}.json"

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.folder.glob("*.json")) if self.folder.is_dir() else []

    def load(self, name: str) -> dict[str, float]:
        p = self._path(name)
        if not p.exists():
            raise ChangeError(f"no preset named {name!r}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in data.get("values", {}).items()}

    def save(self, name: str, values: dict[str, float]) -> None:
        p = self._path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"values": values}, indent=1), encoding="utf-8")

    def delete(self, name: str) -> dict[str, float] | None:
        p = self._path(name)
        if not p.exists():
            return None
        vals = self.load(name)
        p.unlink()
        return vals


def make_presets_domain(store: PresetStore) -> Domain:
    def lst(cfg: Any) -> list[dict[str, Any]]:
        from away.app.tunables import PRESETS

        return [{"key": n, "kind": "preset", "builtin": True} for n in PRESETS] + [
            {"key": n, "kind": "preset", "builtin": False} for n in store.names()
        ]

    def apply(cfg: Any, op: Op) -> Applied:
        if op.op == "create":
            existed = op.key in store.names()
            old = store.load(op.key) if existed else None
            vals = {k: float(v) for k, (_lo, _hi, v) in ranges(cfg).items() if not isinstance(v, bool)}
            store.save(op.key, vals)
            inv = [Op("presets", "restore_file", op.key, old)] if existed else [Op("presets", "delete", op.key)]
            return Applied(op, None, op.key, inv, f"Saved your current settings as preset '{op.key}'")
        if op.op == "restore_file":
            if op.value is None:
                store.delete(op.key)
            else:
                store.save(op.key, dict(op.value))
            return Applied(op, None, op.key, [], f"Restored preset file '{op.key}'")
        if op.op == "delete":
            old = store.delete(op.key)
            if old is None:
                raise ChangeError(f"no saved preset named {op.key!r}")
            return Applied(op, op.key, None, [Op("presets", "restore_file", op.key, old)], f"Deleted preset '{op.key}'")
        raise ChangeError("use apply_preset to apply a preset; presets support create and delete")

    return Domain("presets", "Presets", lst, apply, ops=("create", "delete", "apply"))


def preset_ops(cfg: Any, name: str, store: PresetStore) -> list[Op]:
    """A preset expanded into ordinary setting changes (so applying it is undoable like anything else)."""
    from away.app.tunables import PRESETS, cached_defaults, preset_values, tunable_bounds

    vals = preset_values(name, cached_defaults(), tunable_bounds()) if name in PRESETS else store.load(name)
    return [Op("settings", "set", k, v) for k, v in sorted(vals.items()) if k in ranges(cfg)]


# ---------------------------------------------------------------------------- registry
_EXTRA: dict[str, Domain] = {}


def register(domain: Domain) -> None:
    """Feature modules add their own domain (e.g. Gesture Studio's per-gesture settings)."""
    _EXTRA[domain.name] = domain


def unregister(name: str) -> None:
    _EXTRA.pop(name, None)


def placeholder(name: str, title: str, todo: str) -> Domain:
    return Domain(name, title, lambda _cfg: [], None, available=False, note=f"not available in this version ({todo})")


def domains(store: PresetStore | None = None) -> dict[str, Domain]:
    from away.agentlink import integrations

    out = {
        "settings": Domain("settings", "Settings", settings_list, settings_apply),
        "zones": Domain("zones", "Zones", zones_list, zones_apply),
        "bindings": Domain(
            "bindings",
            "Gestures, voice commands and face triggers",
            bindings_list,
            bindings_apply,
            ops=("set", "delete"),
        ),
        "custom_gestures": Domain(
            "custom_gestures", "Custom gestures", custom_list, custom_apply, ops=("create", "delete", "set")
        ),
        "profiles": Domain(
            "profiles", "Per-app profiles", profiles_list, profiles_apply, ops=("create", "delete", "set")
        ),
        "gestures": integrations.gestures_domain(),
        "macros": integrations.macros_domain(),
    }
    if store is not None:
        out["presets"] = make_presets_domain(store)
    out.update(_EXTRA)
    return out


def apply_ops(cfg: Any, ops: list[Op], store: PresetStore | None = None) -> list[Applied]:
    """Apply ``ops`` to ``cfg`` in place (all or nothing: raises before anything is kept if one fails)."""
    doms = domains(store)
    work = copy.deepcopy(cfg)
    done: list[Applied] = []
    for op in ops:
        _forbid(op.key)
        d = doms.get(op.domain)
        if d is None:
            raise ChangeError(f"unknown kind of change {op.domain!r}; use {', '.join(doms)}")
        if not d.available or d.apply is None:
            raise ChangeError(f"{d.title}: {d.note}")
        try:
            done.append(d.apply(work, op))
        except ChangeError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            raise ChangeError(str(exc)) from None
    try:
        cfgmod.validate(work)
        cfgmod.from_dict(cfgmod.to_dict(work))  # the saved file must load again
    except ValueError as exc:
        raise ChangeError(f"that change would make the settings invalid: {exc}") from None
    for f in cfgmod.fields(cfg):
        setattr(cfg, f.name, getattr(work, f.name))
    return done
