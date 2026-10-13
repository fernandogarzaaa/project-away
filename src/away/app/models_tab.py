"""Settings > Models (v0.1.23): one row per model slot, from the model registry (:mod:`away.app.model_registry`).

Each row: the slot ("Speech to text"), a drop-down of the models for it (permissively licensed ones only), an
On switch for the optional slots (face, eye tracking, cleanup, voice), where it runs (CPU / GPU), and Download /
Remove for the natural voices. On top: Performance mode (Light / Balanced / Best) sets every slot at once. Every
change is saved at once; models load at the next start. Agents make the same changes with the ``models`` op.

The logic lives in small pure functions (:func:`rows`, :func:`choose`, :func:`toggle`, :func:`apply_mode`) so it
is tested without a display; :func:`build` only lays out Tk widgets around them.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.app import model_registry as MR

RESTART_NOTE = "Saved. New models load the next time Project Away starts."


def rows(reg: MR.Registry, cfg: Any) -> list[dict[str, Any]]:
    """The registry's rows plus the drop-down labels (``labels``) and the current label (``label``)."""
    out = []
    for r in reg.rows(cfg):
        labels = [c["title"] for c in r["choices"]]
        label = r["current_title"]
        if r["current"] is None:
            labels = [label, *labels]  # "custom": a setting no listed model matches
        out.append({**r, "labels": labels, "label": label})
    return out


def model_for(reg: MR.Registry, slot: str, label: str) -> MR.Model:
    for m in reg.for_slot(slot):
        if m.title == label:
            return m
    raise ValueError(f"no {MR.SLOT_TITLES.get(slot, slot)} model called {label!r}")


def choose(reg: MR.Registry, cfg: Any, slot: str, label: str) -> str:
    m = model_for(reg, slot, label)
    reg.switch(cfg, slot, m.id)
    return f"{MR.SLOT_TITLES[slot]}: {m.title}. {RESTART_NOTE}"


def toggle(reg: MR.Registry, cfg: Any, slot: str, on: bool) -> str:
    reg.set_enabled(cfg, slot, on)
    return f"{MR.SLOT_TITLES[slot]}: {'on' if on else 'off'}. {RESTART_NOTE}"


def apply_mode(reg: MR.Registry, cfg: Any, title: str) -> str:
    mode = next((k for k, v in MR.MODE_TITLES.items() if v == title), title)
    reg.apply_mode(cfg, mode)
    return f"Performance mode: {MR.MODE_TITLES.get(mode, mode)}. {RESTART_NOTE}"


def voice_engine(m: MR.Model | None) -> str:
    """``voice:kokoro`` / ``voice:piper`` models have a Download / Remove button ("" for the rest)."""
    if m is None or not m.download.startswith("voice:"):
        return ""
    return m.download.split(":", 1)[1]


def download_hint(m: MR.Model | None) -> str:
    """Where a model that is not built in is downloaded ("" when it needs nothing)."""
    if m is None or m.bundled or not m.download or m.download.startswith("voice:"):
        return ""
    return f"Download: {m.download}"


def build(win: Any, nb: Any, data_dir: Path | None = None) -> Any:  # pragma: no cover - Tk; smoke-tested
    """The Models tab. ``win``: the Settings window (``tk``, ``ttk``, ``m.cfg``, ``m.save``, ``_ui.post``)."""
    from away.voice import natural

    tk, ttk = win.tk, win.ttk
    cfg = win.m.cfg
    reg = MR.Registry(data_dir)
    f = win._scroll_tab(nb, "Models") if hasattr(win, "_scroll_tab") else ttk.Frame(nb)
    if not hasattr(win, "_scroll_tab"):
        nb.add(f, text="Models")
    msg = tk.StringVar(value="Which model does each job. Only permissively licensed models are listed.")

    def save(text: str) -> None:
        try:
            win.m.save()
            msg.set(text)
        except Exception as exc:
            msg.set(f"Could not save: {exc}")
        refresh()

    top = ttk.Frame(f)
    top.pack(fill="x", pady=(0, 6))
    ttk.Label(top, text="Performance mode:").pack(side="left")
    mode_var = tk.StringVar(value="")
    mode_cb = ttk.Combobox(
        top, textvariable=mode_var, values=[MR.MODE_TITLES[m] for m in MR.MODES], state="readonly", width=12
    )
    mode_cb.pack(side="left", padx=6)
    mode_cb.bind("<<ComboboxSelected>>", lambda _e: _run(lambda: apply_mode(reg, cfg, mode_var.get())))

    grid = ttk.Frame(f)
    grid.pack(fill="x")
    widgets: dict[str, dict[str, Any]] = {}

    def _run(fn: Callable[[], str]) -> None:
        try:
            save(fn())
        except Exception as exc:
            msg.set(str(exc))
            refresh()

    for i, r in enumerate(rows(reg, cfg)):
        slot = r["slot"]
        ttk.Label(grid, text=r["title"], width=22).grid(row=i, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=r["label"])
        cb = ttk.Combobox(grid, textvariable=var, values=r["labels"], state="readonly", width=40)
        cb.grid(row=i, column=1, sticky="w", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e, s=slot, v=var: _run(lambda: choose(reg, cfg, s, v.get())))
        on = tk.BooleanVar(value=r["on"])
        chk = ttk.Checkbutton(
            grid, text="On", variable=on, command=lambda s=slot, o=on: _run(lambda: toggle(reg, cfg, s, o.get()))
        )
        if r["optional"]:
            chk.grid(row=i, column=2, padx=4)
        where = tk.StringVar(value=r["runs_on"])
        ttk.Label(grid, textvariable=where, width=10).grid(row=i, column=3, sticky="w")
        hint = tk.StringVar(value="")
        dl = ttk.Button(grid, text="Download", command=lambda s=slot: _download(s))
        rm = ttk.Button(grid, text="Remove", command=lambda s=slot: _remove(s))
        ttk.Label(grid, textvariable=hint, foreground="#555").grid(row=i, column=6, sticky="w")
        widgets[slot] = {"var": var, "cb": cb, "on": on, "where": where, "hint": hint, "dl": dl, "rm": rm, "row": i}

    ttk.Label(f, textvariable=msg, wraplength=640, foreground="#333").pack(fill="x", pady=8)

    def refresh() -> None:
        for r in rows(reg, cfg):
            w = widgets[r["slot"]]
            w["var"].set(r["label"])
            w["cb"].configure(values=r["labels"])
            w["on"].set(r["on"])
            w["where"].set(r["runs_on"])
            cur = reg.current(cfg, r["slot"])
            eng = voice_engine(cur)
            w["hint"].set(download_hint(cur))
            if eng:
                ready = natural.installed(eng, getattr(cfg.speech, "piper_voice", ""))
                w["dl"].grid(row=w["row"], column=4, padx=2)
                w["dl"].configure(state="disabled" if ready else "normal")
                w["rm"].grid(row=w["row"], column=5, padx=2)
                w["rm"].configure(state="normal" if ready else "disabled")
                if not ready:
                    w["hint"].set(natural.LICENSE_NOTE.split(". ")[0] + ".")
            else:
                w["dl"].grid_remove()
                w["rm"].grid_remove()

    def _download(slot: str) -> None:
        eng = voice_engine(reg.current(cfg, slot))
        if not eng:
            return
        widgets[slot]["dl"].configure(state="disabled")
        msg.set(f"Downloading the voice... ({natural.LICENSE_NOTE})")

        def run() -> None:
            try:
                natural.download(eng, getattr(cfg.speech, "piper_voice", ""))
                text = "Voice downloaded."
            except Exception as exc:
                text = f"The download did not finish ({str(exc)[:120]})."
            win._ui.post(lambda: (msg.set(text), refresh()))

        threading.Thread(target=run, daemon=True, name="models-download").start()

    def _remove(slot: str) -> None:
        natural.remove()
        msg.set("Natural voices removed; the next available voice is used.")
        refresh()

    refresh()
    return f
