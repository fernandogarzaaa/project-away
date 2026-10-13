"""Teach-by-demonstration shortcuts ("macros" in code and config, "shortcuts" in the UI).

A shortcut is a short list of :class:`Step` s recorded from your own mouse and keyboard: clicks (stored relative
to the window they landed in, plus the name of the control under the pointer when Windows UI Automation can
tell), drags, wheel scrolls, typed text, key presses and waits. Any gesture or voice phrase runs one through the
normal bindings with the action ``macro:<name>``.

This module is pure data: the model, JSON (de)serialisation and validation, the local store, the config section
and :func:`build_steps`, which turns raw input events into editable steps. Recording, replay and the OS glue live
in :mod:`away.control.macro_record`, :mod:`away.control.macro_play` and :mod:`away.control.macro_platform`.

Privacy: shortcuts that type text are marked ``sensitive``. They live only in the local data folder
(``<data_dir>/macros``, never the roaming config folder), are written owner-only where the OS supports it, and are
not exported unless you say so explicitly. Keystrokes typed into a password box are never recorded.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from away.control.keys import MODIFIERS, KeyComboError, canonical

FORMAT_VERSION = 1
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
STEP_TYPES = ("click", "drag", "scroll", "text", "key", "wait")
BUTTONS = ("left", "right", "middle")
SPEEDS = ("1x", "2x", "instant")
SPEED_FACTOR = {"1x": 1.0, "2x": 2.0, "instant": 0.0}
MAX_STEPS = 2000
MAX_TEXT = 10_000
MAX_WAIT_S = 600.0
# side-specific modifiers fold onto the base name (shortcuts care about "ctrl", not which ctrl)
MOD_BASE = {
    "ctrl_l": "ctrl",
    "ctrl_r": "ctrl",
    "alt_l": "alt",
    "alt_r": "alt",
    "alt_gr": "alt",
    "shift_l": "shift",
    "shift_r": "shift",
    "cmd_l": "cmd",
    "cmd_r": "cmd",
}


class MacroError(ValueError):
    """A shortcut file or edit is invalid. The message is written for people, not programmers."""


class SensitiveExportError(MacroError):
    """Export refused: a shortcut types text you entered (pass ``include_sensitive=True`` to export anyway)."""


# --------------------------------------------------------------------------- config
@dataclass
class MacrosConfig:
    """``macros:`` section of config.yaml."""

    enabled: bool = True
    abort_hotkey: str = "esc"  # a physical press stops a running shortcut (Away's own key presses are ignored)
    abort_on_mouse_move: bool = True  # moving the real mouse during a replay stops it
    abort_move_px: float = 40.0  # how far the real mouse must move to count
    allow_other_app: bool = False  # True: replay even when a step's window belongs to another app than recorded
    default_speed: str = "1x"  # 1x | 2x | instant, for newly recorded shortcuts
    max_delay_s: float = 3.0  # recorded pauses longer than this are shortened to it
    min_gap_s: float = 0.05  # at "instant" speed, still wait this long between steps so apps keep up
    record_away_input: bool = False  # also record clicks/keys Project Away itself produced (gestures, voice)
    # off = typed characters are not recorded where password boxes cannot be detected (macOS/Linux, or Windows
    # without UI Automation); keys like enter, tab and ctrl+c are still recorded
    record_text_without_password_check: bool = False
    use_ui_automation: bool = True  # Windows: remember/find the control under each click by its name
    store_dir: str = ""  # "" = <data_dir>/macros


def validate_macros_config(m: MacrosConfig) -> None:
    for name in (
        "enabled",
        "abort_on_mouse_move",
        "allow_other_app",
        "record_away_input",
        "record_text_without_password_check",
        "use_ui_automation",
    ):
        if not isinstance(getattr(m, name), bool):
            raise ValueError(f"macros.{name} must be true or false, got {getattr(m, name)!r}")
    if m.default_speed not in SPEEDS:
        raise ValueError(f"macros.default_speed must be {'|'.join(SPEEDS)}, got {m.default_speed!r}")
    for name, lo, hi in (("abort_move_px", 5.0, 2000.0), ("max_delay_s", 0.1, 60.0), ("min_gap_s", 0.0, 2.0)):
        v = getattr(m, name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
            raise ValueError(f"macros.{name} must be a number in [{lo:g}, {hi:g}], got {v!r}")
    if m.abort_hotkey:
        try:
            m.abort_hotkey = canonical(str(m.abort_hotkey))
        except KeyComboError as exc:
            raise ValueError(f"macros.abort_hotkey: {exc}") from None
    if not isinstance(m.store_dir, str):
        raise ValueError("macros.store_dir must be a folder path or empty")


# --------------------------------------------------------------------------- model
@dataclass
class Target:
    """Where a pointer step lands: screen position, position inside its window, and the control there."""

    x: int = 0  # absolute screen pixels (fallback when the window cannot be found)
    y: int = 0
    rx: int | None = None  # relative to the window's top-left corner
    ry: int | None = None
    app: str = ""  # process name of the window (chrome.exe), "" = unknown
    window: str = ""  # window title when recorded (used to pick between windows of one app)
    element: str = ""  # UI Automation name of the control under the pointer ("Send", "File")
    element_type: int = 0  # UI Automation control type id (0 = unknown)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"x": self.x, "y": self.y}
        if self.rx is not None and self.ry is not None:
            d["rx"], d["ry"] = self.rx, self.ry
        for k in ("app", "window", "element"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.element_type:
            d["element_type"] = self.element_type
        return d

    @classmethod
    def from_dict(cls, d: Any, where: str) -> Target:
        if not isinstance(d, dict):
            raise MacroError(f"{where}: the click position is missing")
        try:
            t = cls(
                x=int(d.get("x", 0)),
                y=int(d.get("y", 0)),
                rx=None if d.get("rx") is None else int(d["rx"]),
                ry=None if d.get("ry") is None else int(d["ry"]),
                app=str(d.get("app", "")),
                window=str(d.get("window", "")),
                element=str(d.get("element", "")),
                element_type=int(d.get("element_type", 0)),
            )
        except (TypeError, ValueError):
            raise MacroError(f"{where}: the click position is not a number") from None
        if (t.rx is None) != (t.ry is None):
            t.rx = t.ry = None
        return t


@dataclass
class Step:
    """One editable step. ``delay`` is the pause before it (seconds, at 1x speed)."""

    type: str
    delay: float = 0.0
    target: Target | None = None  # click / drag start / scroll position
    button: str = "left"  # click, drag
    count: int = 1  # click: 1 = single, 2 = double
    modifiers: list[str] = field(default_factory=list)  # click: keys held during it (ctrl+click)
    path: list[tuple[int, int]] = field(default_factory=list)  # drag: offsets from the start, the last is the drop
    duration: float = 0.0  # drag: how long the move took
    dx: int = 0  # scroll: wheel notches (+ = right / up)
    dy: int = 0
    text: str = ""  # text: what to type
    keys: str = ""  # key: canonical combo (enter, ctrl+c)
    seconds: float = 0.0  # wait
    app: str = ""  # text / key: the app that had focus when it was recorded (replay checks it still does)

    # ---------------------------------------------------------------- people-facing text
    def describe(self) -> str:
        """Plain words for the step list. Typed text is never shown in full (it may be private)."""
        where = ""
        if self.target is not None:
            if self.target.element:
                where = f' on "{self.target.element}"'
            elif self.target.app:
                where = f" in {app_label(self.target.app)}"
        if self.type == "click":
            kind = {1: "Click", 2: "Double-click", 3: "Triple-click"}.get(self.count, "Click")
            if self.button != "left":
                kind = f"{self.button.capitalize()}-{kind.lower()}"
            mods = "+".join(self.modifiers)
            return f"{mods + '+' if mods else ''}{kind}{where}"
        if self.type == "drag":
            return f"Drag{where}"
        if self.type == "scroll":
            if self.dy:
                return f"Scroll {'up' if self.dy > 0 else 'down'} {abs(self.dy)}{where}"
            return f"Scroll {'right' if self.dx > 0 else 'left'} {abs(self.dx)}{where}"
        if self.type == "text":
            n = len(self.text)
            return f"Type text ({n} character{'s' if n != 1 else ''})"
        if self.type == "key":
            return f"Press {self.keys}"
        return f"Wait {self.seconds:g} s"

    def __str__(self) -> str:  # never leaks typed text into logs
        return self.describe()

    @property
    def is_pointer(self) -> bool:
        return self.type in ("click", "drag", "scroll")

    @property
    def types_text(self) -> bool:
        """Types characters a person entered (text, or a lone printable key like ``a``)."""
        if self.type == "text":
            return bool(self.text)
        if self.type == "key":
            parts = self.keys.split("+")
            return all(p == "shift" for p in parts[:-1]) and len(parts[-1]) == 1
        return False

    # ---------------------------------------------------------------- JSON
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "delay": round(float(self.delay), 3)}
        if self.target is not None and self.is_pointer:
            d["target"] = self.target.to_dict()
        if self.type in ("click", "drag"):
            d["button"] = self.button
        if self.type == "click":
            d["count"] = self.count
            if self.modifiers:
                d["modifiers"] = list(self.modifiers)
        elif self.type == "drag":
            d["path"] = [[int(a), int(b)] for a, b in self.path]
            d["duration"] = round(float(self.duration), 3)
        elif self.type == "scroll":
            d["dx"], d["dy"] = self.dx, self.dy
        elif self.type == "text":
            d["text"] = self.text
        elif self.type == "key":
            d["keys"] = self.keys
        elif self.type == "wait":
            d["seconds"] = round(float(self.seconds), 3)
        if self.type in ("text", "key") and self.app:
            d["app"] = self.app
        return d

    @classmethod
    def from_dict(cls, d: Any, where: str = "step") -> Step:
        if not isinstance(d, dict):
            raise MacroError(f"{where}: not a step")
        st = Step(type=str(d.get("type", "")))
        st.delay = _num(d.get("delay", 0.0), f"{where}: delay", 0.0, MAX_WAIT_S)
        if st.type in ("click", "drag", "scroll"):
            st.target = Target.from_dict(d.get("target"), where)
        if st.type in ("click", "drag"):
            st.button = str(d.get("button", "left"))
        if st.type == "click":
            st.count = int(_num(d.get("count", 1), f"{where}: count", 1, 3))
            mods = d.get("modifiers") or []
            if not isinstance(mods, list):
                raise MacroError(f"{where}: modifiers must be a list")
            st.modifiers = [str(m) for m in mods]
        elif st.type == "drag":
            path = d.get("path") or []
            try:
                st.path = [(int(p[0]), int(p[1])) for p in path]
            except (TypeError, ValueError, IndexError, KeyError):
                raise MacroError(f"{where}: the drag path is not a list of points") from None
            st.duration = _num(d.get("duration", 0.0), f"{where}: duration", 0.0, MAX_WAIT_S)
        elif st.type == "scroll":
            st.dx = int(_num(d.get("dx", 0), f"{where}: dx", -1000, 1000))
            st.dy = int(_num(d.get("dy", 0), f"{where}: dy", -1000, 1000))
        elif st.type == "text":
            st.text = str(d.get("text", ""))
        elif st.type == "key":
            st.keys = str(d.get("keys", ""))
        elif st.type == "wait":
            st.seconds = _num(d.get("seconds", 0.0), f"{where}: seconds", 0.0, MAX_WAIT_S)
        if st.type in ("text", "key"):
            st.app = str(d.get("app", ""))
        st.validate(where)
        return st

    def validate(self, where: str = "step") -> None:
        if self.type not in STEP_TYPES:
            raise MacroError(f"{where}: unknown step type {self.type!r} (use {', '.join(STEP_TYPES)})")
        if not 0.0 <= float(self.delay) <= MAX_WAIT_S:
            raise MacroError(f"{where}: the pause must be between 0 and {MAX_WAIT_S:g} seconds")
        if self.is_pointer and self.target is None:
            raise MacroError(f"{where}: the click position is missing")
        if self.type in ("click", "drag") and self.button not in BUTTONS:
            raise MacroError(f"{where}: mouse button must be left, right or middle")
        if self.type == "click":
            if self.count not in (1, 2, 3):
                raise MacroError(f"{where}: a click happens 1, 2 or 3 times")
            bad = [m for m in self.modifiers if m not in MODIFIERS]
            if bad:
                raise MacroError(f"{where}: {', '.join(bad)} is not a modifier key ({', '.join(MODIFIERS)})")
        if self.type == "drag" and not self.path:
            raise MacroError(f"{where}: a drag needs somewhere to drop")
        if self.type == "scroll" and not (self.dx or self.dy):
            raise MacroError(f"{where}: a scroll must move at least one notch")
        if self.type == "text":
            if not self.text:
                raise MacroError(f"{where}: there is no text to type")
            if len(self.text) > MAX_TEXT:
                raise MacroError(f"{where}: the text is too long (max {MAX_TEXT} characters)")
        if self.type == "key":
            try:
                self.keys = canonical(self.keys)
            except KeyComboError as exc:
                raise MacroError(f"{where}: {exc}") from None
        if self.type == "wait" and not 0.0 < float(self.seconds) <= MAX_WAIT_S:
            raise MacroError(f"{where}: a wait must be between 0 and {MAX_WAIT_S:g} seconds")


@dataclass
class Macro:
    name: str  # lower_snake id used in bindings (macro:open_mail)
    title: str = ""  # what you called it ("Open my mail")
    steps: list[Step] = field(default_factory=list)
    app: str = ""  # main app it was recorded in (process name)
    speed: str = "1x"
    allow_other_app: bool = False
    created: float = 0.0
    skipped_password_keys: int = 0  # keystrokes not recorded because a password box had focus
    skipped_text: bool = False  # typed characters were left out (password boxes could not be detected)

    @property
    def label(self) -> str:
        return self.title or title_from_name(self.name)

    @property
    def sensitive(self) -> bool:
        """Types text you entered: kept on this PC only and not exported unless asked."""
        return any(s.types_text for s in self.steps)

    @property
    def apps(self) -> list[str]:
        out: list[str] = []
        for s in self.steps:
            app = s.target.app if s.target is not None else s.app
            if app and app not in out:
                out.append(app)
        return out

    def validate(self) -> None:
        if not NAME_RE.match(self.name or ""):
            raise MacroError(
                f"{self.name!r} can't be used as a shortcut id (letters, digits and _ only, starting with a letter)"
            )
        if not self.steps:
            raise MacroError("This shortcut has no steps. Record it again and click or type something.")
        if len(self.steps) > MAX_STEPS:
            raise MacroError(f"This shortcut is too long ({len(self.steps)} steps, max {MAX_STEPS}).")
        if self.speed not in SPEEDS:
            raise MacroError(f"Speed must be {', '.join(SPEEDS)}")
        for i, s in enumerate(self.steps, 1):
            s.validate(f"step {i}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "name": self.name,
            "title": self.title,
            "app": self.app,
            "speed": self.speed,
            "allow_other_app": self.allow_other_app,
            "sensitive": self.sensitive,
            "created": round(self.created, 3),
            "skipped_password_keys": self.skipped_password_keys,
            "skipped_text": self.skipped_text,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: Any) -> Macro:
        if not isinstance(d, dict):
            raise MacroError("This file is not a Project Away shortcut.")
        fmt = d.get("format", FORMAT_VERSION)
        if not isinstance(fmt, int) or fmt > FORMAT_VERSION:
            raise MacroError("This shortcut was made by a newer Project Away. Update the app to use it.")
        steps = d.get("steps")
        if not isinstance(steps, list):
            raise MacroError("This file is not a Project Away shortcut (it has no steps).")
        m = cls(
            name=str(d.get("name", "")),
            title=str(d.get("title", "")),
            steps=[Step.from_dict(s, f"step {i}") for i, s in enumerate(steps, 1)],
            app=str(d.get("app", "")),
            speed=str(d.get("speed", "1x")),
            allow_other_app=bool(d.get("allow_other_app", False)),
            created=float(d.get("created", 0.0) or 0.0),
            skipped_password_keys=int(d.get("skipped_password_keys", 0) or 0),
            skipped_text=bool(d.get("skipped_text", False)),
        )
        m.validate()
        return m


def _num(v: Any, where: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise MacroError(f"{where} must be a number")
    if not lo <= float(v) <= hi:
        raise MacroError(f"{where} must be between {lo:g} and {hi:g}")
    return float(v)


# --------------------------------------------------------------------------- names
def name_from_title(title: str) -> str:
    """``"Open my mail!"`` -> ``open_my_mail`` (the id used in ``macro:<name>``)."""
    s = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    if not s:
        return ""
    if not s[0].isalpha():
        s = "shortcut_" + s
    return s[:40].rstrip("_")


def title_from_name(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def app_label(process: str) -> str:
    """``chrome.exe`` -> ``Chrome`` (what people call the app)."""
    p = process.replace("\\", "/").rsplit("/", 1)[-1]
    if p.lower().endswith(".exe"):
        p = p[:-4]
    return p[:1].upper() + p[1:] if p else "the app"


def app_key(process: str) -> str:
    p = (process or "").strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    return p[:-4] if p.endswith(".exe") else p


# --------------------------------------------------------------------------- store
class MacroStore:
    """One JSON file per shortcut in a local folder. Writes are atomic; sensitive files are owner-only."""

    SUFFIX = ".json"

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)

    @classmethod
    def default(cls, cfg: MacrosConfig | None = None) -> MacroStore:
        if cfg is not None and cfg.store_dir:
            return cls(Path(cfg.store_dir).expanduser())
        from away import paths

        return cls(paths.data_dir() / "macros")

    def path(self, name: str) -> Path:
        if not NAME_RE.match(name or ""):
            raise MacroError(f"{name!r} is not a shortcut id")
        return self.folder / f"{name}{self.SUFFIX}"

    def names(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted(p.stem for p in self.folder.glob(f"*{self.SUFFIX}") if NAME_RE.match(p.stem))

    def exists(self, name: str) -> bool:
        return NAME_RE.match(name or "") is not None and self.path(name).is_file()

    def load(self, name: str) -> Macro:
        if not NAME_RE.match(name or ""):
            raise MacroError(f'There is no shortcut called "{name}".')
        p = self.path(name)
        if not p.is_file():
            raise MacroError(f'There is no shortcut called "{title_from_name(name)}".')
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise MacroError(f'The shortcut "{title_from_name(name)}" is damaged. Record it again.') from None
        m = Macro.from_dict({**data, "name": name} if isinstance(data, dict) else data)
        return m

    def all(self) -> list[Macro]:
        out = []
        for n in self.names():
            with contextlib.suppress(MacroError):
                out.append(self.load(n))
        return out

    def save(self, m: Macro) -> Path:
        m.validate()
        self.folder.mkdir(parents=True, exist_ok=True)
        p = self.path(m.name)
        tmp = p.with_suffix(".tmp")
        text = json.dumps(m.to_dict(), indent=2, ensure_ascii=False)
        tmp.write_text(text, encoding="utf-8")
        if m.sensitive and os.name != "nt":
            with contextlib.suppress(OSError):
                os.chmod(tmp, 0o600)
        os.replace(tmp, p)
        return p

    def delete(self, name: str) -> bool:
        p = self.path(name)
        if p.is_file():
            p.unlink()
            return True
        return False

    def rename(self, old: str, new: str, title: str | None = None) -> Macro:
        m = self.load(old)
        if new != old and self.exists(new):
            raise MacroError(f'A shortcut called "{title_from_name(new)}" already exists. Pick another name.')
        m.name = new
        if title is not None:
            m.title = title
        self.save(m)
        if new != old:
            self.delete(old)
        return m

    def unique_name(self, base: str) -> str:
        base = base if NAME_RE.match(base or "") else "shortcut"
        if not self.exists(base):
            return base
        i = 2
        while self.exists(f"{base[:36]}_{i}"):
            i += 1
        return f"{base[:36]}_{i}"

    def next_untitled(self) -> tuple[str, str]:
        """(``shortcut_3``, ``Shortcut 3``) for a recording that has not been named yet."""
        i = 1
        while self.exists(f"shortcut_{i}"):
            i += 1
        return f"shortcut_{i}", f"Shortcut {i}"

    # ---------------------------------------------------------------- import / export
    def export(self, names: Iterable[str], dest: Path | str, include_sensitive: bool = False) -> Path:
        macros = [self.load(n) for n in names]
        if not macros:
            raise MacroError("Pick at least one shortcut to export.")
        private = [m.label for m in macros if m.sensitive]
        if private and not include_sensitive:
            one = len(private) == 1
            raise SensitiveExportError(
                f"{', '.join(private)} type{'s' if one else ''} text you entered, so "
                f"{'it stays' if one else 'they stay'} on this PC. Export anyway only if the file will not be shared."
            )
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = {"format": FORMAT_VERSION, "shortcuts": [m.to_dict() for m in macros]}
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return dest

    def import_file(self, src: Path | str, overwrite: bool = False) -> list[Macro]:
        try:
            data = json.loads(Path(src).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise MacroError("That file could not be read as shortcuts (expected a .json export).") from None
        items = data.get("shortcuts") if isinstance(data, dict) and "shortcuts" in data else data
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list) or not items:
            raise MacroError("That file has no shortcuts in it.")
        parsed = [Macro.from_dict(it) for it in items]  # validate all before writing any
        out = []
        for m in parsed:
            if self.exists(m.name) and not overwrite:
                m.name = self.unique_name(m.name)
            if not m.created:
                m.created = time.time()
            self.save(m)
            out.append(m)
        return out


# --------------------------------------------------------------------------- raw events -> steps
@dataclass
class RawEvent:
    """One input event captured by the recorder (already filtered for Away's own input and passwords)."""

    t: float
    kind: str  # down | up | move | scroll | key_down | key_up
    x: int = 0
    y: int = 0
    button: str = "left"
    dx: int = 0
    dy: int = 0
    key: str = ""  # canonical key name (ctrl, enter, a) for key events
    char: str = ""  # the character it produced, when printable
    app: str = ""  # key events: the focused app
    target: Target | None = None  # down/scroll: what was under the pointer
    tray: bool = False  # down: landed on the taskbar / tray (dropped when it ends the recording)


DOUBLE_CLICK_S = 0.4
CLICK_SLOP_PX = 6
DRAG_SAMPLE_PX = 8


def build_steps(events: list[RawEvent], max_delay: float = 3.0, record_text: bool = True) -> list[Step]:
    """Turn raw events into editable steps.

    * press + release in place -> ``click`` (quick clicks in place -> one double-click);
      press, move, release elsewhere -> ``drag`` with its path.
    * runs of printable characters (shift allowed) -> one ``text`` step; with ctrl/alt/cmd held -> ``key``
      (``ctrl+c``); enter, tab, arrows, f-keys -> ``key``; a modifier tapped alone -> ``key``.
    * pauses longer than ``max_delay`` are shortened to it. ``record_text=False`` drops typed characters.
    """
    steps: list[Step] = []
    last_t: float | None = None
    held_mods: list[str] = []
    mod_used: dict[str, bool] = {}
    down: RawEvent | None = None
    down_mods: list[str] = []
    drag_path: list[tuple[int, int]] = []

    def gap(t: float) -> float:
        nonlocal last_t
        d = 0.0 if last_t is None else max(0.0, t - last_t)
        last_t = t
        return round(min(d, max_delay), 3)

    def add_text(ch: str, t: float, app: str) -> None:
        prev = steps[-1] if steps else None
        if prev is not None and prev.type == "text" and prev.app == app and t - (last_t or t) <= max(1.5, max_delay):
            prev.text += ch
            gap(t)
            return
        steps.append(Step("text", delay=gap(t), text=ch, app=app))

    for ev in events:
        if ev.kind == "down":
            if down is not None:  # a second button while one is held: ignored
                continue
            down, down_mods, drag_path = ev, [m for m in held_mods if m in MODIFIERS], []
            for m in held_mods:
                mod_used[m] = True
        elif ev.kind == "move":
            if down is not None:
                rel = (ev.x - down.x, ev.y - down.y)
                last = drag_path[-1] if drag_path else (0, 0)
                if abs(rel[0] - last[0]) + abs(rel[1] - last[1]) >= DRAG_SAMPLE_PX:
                    drag_path.append(rel)
        elif ev.kind == "up":
            if down is None or ev.button != down.button:
                continue
            tgt = down.target or Target(down.x, down.y)
            moved = abs(ev.x - down.x) + abs(ev.y - down.y)
            if moved <= CLICK_SLOP_PX:
                prev = steps[-1] if steps else None
                if (
                    prev is not None
                    and prev.type == "click"
                    and prev.button == down.button
                    and prev.count < 3
                    and prev.target is not None
                    and abs(prev.target.x - down.x) + abs(prev.target.y - down.y) <= CLICK_SLOP_PX
                    and last_t is not None
                    and down.t - last_t <= DOUBLE_CLICK_S
                ):
                    prev.count += 1
                else:
                    steps.append(
                        Step("click", delay=gap(down.t), target=tgt, button=down.button, modifiers=list(down_mods))
                    )
                last_t = ev.t
            else:
                end = (ev.x - down.x, ev.y - down.y)
                if not drag_path or drag_path[-1] != end:
                    drag_path.append(end)
                steps.append(
                    Step(
                        "drag",
                        delay=gap(down.t),
                        target=tgt,
                        button=down.button,
                        path=drag_path,
                        duration=round(max(0.0, ev.t - down.t), 3),
                    )
                )
                last_t = ev.t
            down = None
        elif ev.kind == "scroll":
            tgt = ev.target or Target(ev.x, ev.y)
            prev = steps[-1] if steps else None
            if (
                prev is not None
                and prev.type == "scroll"
                and prev.target is not None
                and abs(prev.target.x - ev.x) + abs(prev.target.y - ev.y) <= CLICK_SLOP_PX * 4
                and last_t is not None
                and ev.t - last_t <= 0.5
                and (prev.dy == 0) == (ev.dy == 0)
            ):
                prev.dx += ev.dx
                prev.dy += ev.dy
                last_t = ev.t
                if not (prev.dx or prev.dy):
                    steps.pop()
            elif ev.dx or ev.dy:
                steps.append(Step("scroll", delay=gap(ev.t), target=tgt, dx=ev.dx, dy=ev.dy))
        elif ev.kind == "key_down":
            k = MOD_BASE.get(ev.key, ev.key)
            if k in MODIFIERS:
                if k not in held_mods:
                    held_mods.append(k)
                    mod_used[k] = False
                continue
            for m in held_mods:
                mod_used[m] = True
            combo_mods = [m for m in held_mods if m != "shift"]
            printable = ev.char if (ev.char and ev.char.isprintable()) else (" " if k == "space" else "")
            if printable and not combo_mods:
                if record_text:
                    add_text(printable, ev.t, ev.app)
                continue
            if not k or (len(k) == 1 and not combo_mods and not record_text):
                continue
            try:
                combo = canonical("+".join([*held_mods, k]))
            except KeyComboError:
                continue
            steps.append(Step("key", delay=gap(ev.t), keys=combo, app=ev.app))
        elif ev.kind == "key_up":
            k = MOD_BASE.get(ev.key, ev.key)
            if k in held_mods:
                held_mods.remove(k)
                if not mod_used.pop(k, True):
                    steps.append(Step("key", delay=gap(ev.t), keys=k, app=ev.app))  # a lone tap (the Windows key)
    return steps
