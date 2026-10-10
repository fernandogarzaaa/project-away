"""Key-combo parsing shared by bindings, the hotkey listener and the settings GUI.

One canonical spelling is used everywhere: lower-case key names joined with
``+`` and modifiers first, e.g. ``ctrl+shift+t``, ``alt+left``, ``f5``.
pynput's ``<ctrl>+<alt>+space`` spelling is accepted on input and produced by
:func:`to_pynput_hotkey` for :class:`pynput.keyboard.GlobalHotKeys`.
"""

from __future__ import annotations

MODIFIERS = ("ctrl", "alt", "shift", "cmd")
ALIASES = {
    "control": "ctrl",
    "ctl": "ctrl",
    "option": "alt",
    "opt": "alt",
    "win": "cmd",
    "windows": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "command": "cmd",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
    "ins": "insert",
    "pgup": "page_up",
    "pageup": "page_up",
    "pgdn": "page_down",
    "pagedown": "page_down",
    "prtsc": "print_screen",
    "printscreen": "print_screen",
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
    "plus": "+",
}
# pynput.keyboard.Key member names (the portable subset) + side-specific modifiers
NAMED_KEYS = frozenset(
    {
        *MODIFIERS,
        "ctrl_l",
        "ctrl_r",
        "alt_l",
        "alt_r",
        "alt_gr",
        "shift_l",
        "shift_r",
        "cmd_l",
        "cmd_r",
        "backspace",
        "caps_lock",
        "delete",
        "down",
        "end",
        "enter",
        "esc",
        "home",
        "insert",
        "left",
        "menu",
        "num_lock",
        "page_down",
        "page_up",
        "pause",
        "print_screen",
        "right",
        "scroll_lock",
        "space",
        "tab",
        "up",
        "media_play_pause",
        "media_next",
        "media_previous",
        "media_volume_up",
        "media_volume_down",
        "media_volume_mute",
        *(f"f{i}" for i in range(1, 25)),
    }
)
_MOD_BASE = {"ctrl_l": "ctrl", "ctrl_r": "ctrl", "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt", "shift_l": "shift"}
_MOD_BASE |= {"shift_r": "shift", "cmd_l": "cmd", "cmd_r": "cmd"}


class KeyComboError(ValueError):
    pass


def _norm_token(tok: str) -> str:
    t = tok.strip()
    if t.startswith("<") and t.endswith(">") and len(t) > 2:
        t = t[1:-1]
    if len(t) == 1:
        return t.lower()
    t = t.lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(t, t)


def parse_combo(text: str) -> list[str]:
    """Parse ``"Ctrl+Shift+T"`` / ``"<ctrl>+<alt>+space"`` into ``["ctrl", "shift", "t"]``.

    Raises :class:`KeyComboError` with a readable message on bad input.
    """
    if not isinstance(text, str) or not text.strip():
        raise KeyComboError("empty key combo")
    raw = text.strip()
    # a literal "+" key: "ctrl++" -> ["ctrl", "+"]
    parts = raw.split("+")
    toks: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == "" and i + 1 < len(parts) and parts[i + 1] == "":
            toks.append("+")
            i += 2
            continue
        if p.strip() == "":
            raise KeyComboError(f"bad key combo {text!r}: empty key between '+' signs")
        toks.append(p)
        i += 1
    keys = [_norm_token(t) for t in toks]
    for k in keys:
        if len(k) == 1:
            if not k.isprintable() or k.isspace():
                raise KeyComboError(f"bad key {k!r} in {text!r}")
            continue
        if k not in NAMED_KEYS:
            raise KeyComboError(
                f"unknown key {k!r} in {text!r}; use a single character or one of: modifiers "
                f"{', '.join(MODIFIERS)}, f1-f24, enter, tab, esc, space, left/right/up/down, home, end, page_up, ..."
            )
    if len(set(keys)) != len(keys):
        raise KeyComboError(f"key repeated in {text!r}")
    mods = [k for k in keys if k in MODIFIERS or k in _MOD_BASE]
    rest = [k for k in keys if k not in mods]
    if len(rest) > 1:
        raise KeyComboError(f"{text!r} has more than one non-modifier key ({', '.join(rest)})")
    order = {m: i for i, m in enumerate(MODIFIERS)}
    mods.sort(key=lambda m: order.get(_MOD_BASE.get(m) or m, 9))
    return mods + rest


def canonical(text: str) -> str:
    return "+".join(parse_combo(text))


def is_modifier_only(keys: list[str]) -> bool:
    return all(k in MODIFIERS or k in _MOD_BASE for k in keys)


def to_pynput_hotkey(text: str) -> str:
    """``ctrl+alt+space`` -> ``<ctrl>+<alt>+<space>`` (GlobalHotKeys format)."""
    return "+".join(k if len(k) == 1 else f"<{k}>" for k in parse_combo(text))


# Tk keysym -> canonical key name (for the settings GUI's key-capture field)
TK_KEYSYMS = {
    "control_l": "ctrl",
    "control_r": "ctrl",
    "alt_l": "alt",
    "alt_r": "alt",
    "meta_l": "cmd",
    "meta_r": "cmd",
    "super_l": "cmd",
    "super_r": "cmd",
    "shift_l": "shift",
    "shift_r": "shift",
    "iso_level3_shift": "alt",
    "return": "enter",
    "kp_enter": "enter",
    "escape": "esc",
    "prior": "page_up",
    "next": "page_down",
    "backspace": "backspace",
    "delete": "delete",
    "insert": "insert",
    "print": "print_screen",
    "caps_lock": "caps_lock",
    "num_lock": "num_lock",
    "scroll_lock": "scroll_lock",
    "space": "space",
    "tab": "tab",
    "iso_left_tab": "tab",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "home": "home",
    "end": "end",
    "pause": "pause",
    "menu": "menu",
    "plus": "+",
    "minus": "-",
    "equal": "=",
    "comma": ",",
    "period": ".",
    "slash": "/",
    "backslash": "\\",
    "semicolon": ";",
    "apostrophe": "'",
    "bracketleft": "[",
    "bracketright": "]",
    "grave": "`",
}


def keysym_to_key(keysym: str) -> str | None:
    """Map one Tk keysym (``Control_L``, ``a``, ``F5``, ``Prior``) to a canonical key, or None."""
    k = keysym.lower()
    if k in TK_KEYSYMS:
        return TK_KEYSYMS[k]
    if len(keysym) == 1 and keysym.isprintable() and not keysym.isspace():
        return keysym.lower()
    if k.startswith("f") and k[1:].isdigit() and 1 <= int(k[1:]) <= 24:
        return k
    return None


class ComboCapture:
    """Accumulates key presses until the first non-modifier key; widget-free so it is testable."""

    def __init__(self) -> None:
        self.held: list[str] = []
        self.result: str | None = None

    def press(self, keysym: str) -> str | None:
        k = keysym_to_key(keysym)
        if k is None or self.result is not None:
            return self.result
        if k not in self.held:
            self.held.append(k)
        if not is_modifier_only([k]):
            self.result = canonical("+".join(self.held))
        return self.result

    def release(self, keysym: str) -> str | None:
        """Releasing a lone modifier (e.g. tap Ctrl_R) captures it as the combo."""
        k = keysym_to_key(keysym)
        if self.result is None and k is not None and self.held and is_modifier_only(self.held):
            self.result = canonical("+".join(self.held))
        return self.result

    def reset(self) -> None:
        self.held = []
        self.result = None
