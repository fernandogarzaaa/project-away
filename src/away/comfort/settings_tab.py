"""Settings > Getting started (v0.1.20): the guided first run button, the plain comfort switches, and the numbers
under Advanced. Built by :func:`build_tab`, called once from :class:`away.app.settings_gui.SettingsWindow`.

:func:`tab_rows` and :func:`advanced_rows` are pure (tested); the widgets are smoke-tested under Xvfb.
"""

from __future__ import annotations

from typing import Any

from away.comfort.settings import ADVANCED_SWITCHES, COMFORT_META, PLAIN_SWITCHES, get_path, set_switch


def tab_rows(cfg: Any) -> list[tuple[str, str, str, bool]]:
    """(path, label, help, current value) for the plain switches."""
    return [(p, label, help_text, bool(get_path(cfg, p))) for p, label, help_text in PLAIN_SWITCHES]


def advanced_rows(model: Any) -> list[Any]:
    """The Settings sliders (``Tunable``) of the attention/comfort numbers, in :data:`COMFORT_META` order."""
    by_path = {t.path: t for t in model.tunables()}
    return [by_path[p] for p in COMFORT_META if p in by_path]


def build_tab(win: Any, nb: Any) -> Any:  # pragma: no cover - widgets
    from away.comfort import actions

    tk, ttk = win.tk, win.ttk
    f = win._scroll_tab(nb, "Getting started", 12)
    nb.insert(0, nb.tabs()[-1])  # first tab: where a new user looks
    nb.select(0)
    big = ("TkDefaultFont", 13, "bold")
    tk.Label(f, text="New here? The guided first run sets everything up in about five minutes.", anchor="w").pack(
        fill="x"
    )
    tk.Button(
        f,
        text="Run the guided first run",
        font=big,
        bg="#2E7D32",
        fg="white",
        activebackground="#1B5E20",
        activeforeground="white",
        relief="flat",
        padx=22,
        pady=10,
        cursor="hand2",
        command=lambda: win._report(actions.launch_first_run(), "opening the guided first run..."),
    ).pack(anchor="w", pady=(6, 16))
    tk.Label(  # v0.1.24: one click, about 2 minutes, a report of what really works on this PC
        f,
        text="Something not working? Check my PC tests your cameras, voice and pointer in about 2 minutes.",
        anchor="w",
    ).pack(fill="x")
    tk.Button(
        f,
        text="Check my PC",
        font=big,
        relief="flat",
        padx=22,
        pady=8,
        cursor="hand2",
        command=lambda: win._report(actions.launch_check_pc(), "opening Check my PC..."),
    ).pack(anchor="w", pady=(6, 16))
    ttk.Separator(f).pack(fill="x", pady=4)
    vars_: dict[str, Any] = {}

    def flip(path: str) -> None:
        err = set_switch(win.m.cfg, path, bool(vars_[path].get()))
        if err:
            vars_[path].set(not vars_[path].get())
        win._report(err, None if err else "changed (press Save to keep it)")

    for path, label, help_text, value in tab_rows(win.m.cfg):
        vars_[path] = tk.BooleanVar(value=value)
        tk.Checkbutton(f, text=label, variable=vars_[path], font=big, anchor="w", command=lambda p=path: flip(p)).pack(
            fill="x", pady=(8, 0)
        )
        tk.Label(f, text=help_text, fg="#616161", anchor="w", justify="left", wraplength=640).pack(fill="x", padx=26)

    adv = ttk.Frame(f)
    shown = {"open": False}
    toggle = ttk.Button(f, text="Advanced \u25b8")

    def open_close() -> None:
        shown["open"] = not shown["open"]
        toggle.configure(text="Advanced \u25be" if shown["open"] else "Advanced \u25b8")
        if shown["open"]:
            adv.pack(fill="x", pady=4)
        else:
            adv.pack_forget()

    toggle.configure(command=open_close)
    toggle.pack(anchor="w", pady=(18, 4))
    for path, label in ADVANCED_SWITCHES:
        vars_[path] = tk.BooleanVar(value=bool(get_path(win.m.cfg, path)))
        ttk.Checkbutton(adv, text=label, variable=vars_[path], command=lambda p=path: flip(p)).pack(fill="x")
    ttk.Label(adv, text="The same numbers are in Tuning > Face, Head & eyes and Other.", foreground="#616161").pack(
        fill="x", pady=(8, 2)
    )
    for t in advanced_rows(win.m):
        row = ttk.Frame(adv)
        row.pack(fill="x")
        ttk.Label(row, text=t.title, width=46).pack(side="left")
        val = ttk.Label(row, text=f"{t.value:g}", width=8)

        def moved(v: str, path: str = t.path, lbl: Any = val, step: float = t.resolution) -> None:
            if abs(float(get_path(win.m.cfg, path)) - float(v)) < step / 2:
                return  # the slider snapping to its step is not a change
            err = win.m.set_tunable(path, float(v))
            lbl.configure(text=f"{float(v):.3g}")
            if err:
                win._report(err)

        scale = tk.Scale(
            row,
            from_=t.lo,
            to=t.hi,
            resolution=t.resolution,
            orient="horizontal",
            showvalue=False,
            length=220,
            command=moved,
        )
        scale.set(t.value)  # before packing; moved() ignores this (no change)
        scale.pack(side="left")
        val.pack(side="left")
    return f
