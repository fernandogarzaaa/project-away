"""Settings for the smart pointer, in plain words: three switches, a status line and a fix button.

Pure parts (tested): :data:`TOGGLES`, :func:`toggle_on`, :func:`set_toggle` and :func:`health`. The numbers
(snap reach, strength, ...) are not here: they are sliders in Settings > Tuning > "Advanced".
:func:`build_section` adds the switches to a Tk frame of the Settings window (one call from ``settings_gui``).
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from away.pointer.uia import availability


@dataclass(frozen=True)
class Toggle:
    key: str
    label: str
    help: str
    path: str  # the config value behind it
    windows_only: bool


TOGGLES: tuple[Toggle, ...] = (
    Toggle(
        "snap",
        "Snap to buttons",
        "The pointer is gently drawn to buttons and links and holds still over them, so clicks land.",
        "hands.cursor.magnet.enabled",
        True,
    ),
    Toggle(
        "say",
        "Click things by saying their name",
        "Say “click Save”, “double click Photos”, “focus Search” or look at something and say “click that”.",
        "voice.click_by_name",
        True,
    ),
    Toggle(
        "smooth",
        "Smoother cursor",
        "Predicts where your hand is going, so the pointer keeps up with less lag and less shake.",
        "hands.cursor.smoothing",
        False,
    ),
)
TOGGLE_BY_KEY = {t.key: t for t in TOGGLES}


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


def raw_to_on(key: str, raw: Any) -> bool:
    """The config value behind a toggle -> on/off (``smoothing: predictive`` is "Smoother cursor" on)."""
    if key == "smooth":
        return raw == "predictive"
    return bool(raw)


def toggle_on(cfg: Any, key: str) -> bool:
    return raw_to_on(key, _get(cfg, TOGGLE_BY_KEY[key].path))


def set_toggle(cfg: Any, key: str, on: bool) -> None:
    t = TOGGLE_BY_KEY[key]
    value: Any = ("predictive" if on else "one_euro") if key == "smooth" else bool(on)
    _set(cfg, t.path, value)


# --------------------------------------------------------------------------------------------- health
INSTALL_CMD = (sys.executable, "-m", "pip", "install", "comtypes>=1.2")


@dataclass(frozen=True)
class Health:
    ok: bool
    message: str
    fix_label: str = ""  # empty: no fix button
    fix: str = ""  # install_comtypes | retry


def health(state: str | None = None, frozen: bool | None = None) -> Health:
    """What the Settings status line says about the Windows parts (snap + click by name)."""
    state = state or availability()
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if state == "ok":
        return Health(True, "Ready. Snap to buttons and click-by-name work in most Windows apps.")
    if state == "not_windows":
        return Health(
            True,
            "Snap to buttons and click-by-name need Windows; the smoother cursor works here too.",
        )
    if frozen:  # the installer bundles comtypes, so this means a damaged install
        return Health(
            False,
            "A part needed for snap to buttons is missing. Reinstalling Project Away fixes it.",
            "Open download page",
            "open_download",
        )
    return Health(
        False,
        "Snap to buttons needs one small free add-on (comtypes). Click Install, then restart Project Away.",
        "Install",
        "install_comtypes",
    )


def run_fix(fix: str, done: Callable[[bool, str], None], runner: Callable[..., Any] = subprocess.run) -> None:
    """Run a fix in the background; ``done(ok, message)`` gets a plain-language result."""

    def work() -> None:
        if fix == "install_comtypes":
            try:
                r = runner(list(INSTALL_CMD), capture_output=True, text=True, timeout=300)
                ok = getattr(r, "returncode", 1) == 0
            except Exception:
                ok = False
            done(
                ok,
                "Installed. Restart Project Away to turn snap to buttons on."
                if ok
                else "The install didn't work (no internet?). Try again later, or run: pip install comtypes",
            )
        elif fix == "open_download":
            import webbrowser

            webbrowser.open("https://github.com/fernandogarzaaa/project-away/releases/latest")
            done(True, "Opened the download page.")
        else:
            done(False, "")

    threading.Thread(target=work, name="pointer-fix", daemon=True).start()


# --------------------------------------------------------------------------------------------- Tk
class _ToggleVar:
    """Lets Settings' refresh() (``var.set(model.voice_value(path))``) drive a toggle whose value is not a bool."""

    def __init__(self, var: Any, key: str) -> None:
        self.var, self.key = var, key

    def set(self, raw: Any) -> None:
        self.var.set(raw_to_on(self.key, raw))

    def get(self) -> bool:
        return bool(self.var.get())


def build_section(gui: Any, parent: Any, row: int) -> int:  # pragma: no cover - Tk
    """Add "Smart pointer" switches + status/fix to ``parent`` at ``row``; returns the next free row.

    ``gui`` is the Settings window (``tk``, ``ttk``, ``m`` = SettingsModel, ``vvars``, ``_report``).
    """
    tk, ttk = gui.tk, gui.ttk
    from away import config as cfgmod

    box = ttk.LabelFrame(parent, text="Smart pointer", padding=(8, 4))
    box.grid(row=row, column=0, columnspan=3, sticky="we", pady=(8, 4))
    h = health()
    is_windows = availability() != "not_windows"
    r = 0
    for t in TOGGLES:
        var = tk.BooleanVar(value=toggle_on(gui.m.cfg, t.key))

        def changed(key: str = t.key, v: Any = var) -> None:
            old = _get(gui.m.cfg, TOGGLE_BY_KEY[key].path)
            set_toggle(gui.m.cfg, key, bool(v.get()))
            try:
                cfgmod.validate(gui.m.cfg)
            except ValueError as exc:
                _set(gui.m.cfg, TOGGLE_BY_KEY[key].path, old)
                gui._report(str(exc))
                return
            gui._report(None)

        cb = ttk.Checkbutton(box, text=t.label, variable=var, command=changed)
        cb.grid(row=r, column=0, sticky="w")
        if t.windows_only and not is_windows:
            cb.state(["disabled"])
        ttk.Label(box, text=t.help, foreground="#555", wraplength=520, justify="left").grid(
            row=r + 1, column=0, sticky="w", padx=(22, 0)
        )
        gui.vvars[t.path] = _ToggleVar(var, t.key)
        r += 2
    status = tk.StringVar(value=h.message)
    line = ttk.Frame(box)
    line.grid(row=r, column=0, sticky="w", pady=(4, 0))
    ttk.Label(line, textvariable=status, foreground="#2e7d32" if h.ok else "#b26a00", wraplength=440).pack(side="left")
    if h.fix:
        btn = ttk.Button(line, text=h.fix_label)

        def fix() -> None:
            btn.state(["disabled"])
            status.set("Working on it…")

            def done(ok: bool, msg: str) -> None:
                gui.root.after(0, lambda: (status.set(msg), None if ok else btn.state(["!disabled"])))

            run_fix(h.fix, done)

        btn.configure(command=fix)
        btn.pack(side="left", padx=8)
    ttk.Label(box, text="Fine-tuning: Tuning tab > Advanced.", foreground="#777").grid(row=r + 1, column=0, sticky="w")
    return row + 1
