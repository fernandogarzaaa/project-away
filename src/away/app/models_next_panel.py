"""Settings > Performance > "Newer models (optional)" (v0.1.24): plain words, one click, the details folded away.

What the user sees, top to bottom:

1. one line: what is in use and whether "Test my PC" has passed ("Using the standard models. Test my PC to see if the
   newer ones run well here.");
2. **Test my PC** - one click: downloads nothing, measures what is downloaded, then says which preset fits and offers
   **Use it** (undoable; anything that later fails or runs slower switches back by itself);
3. **Light / Balanced / Best** - each shows its estimated graphics memory and what still needs downloading; one click
   downloads what is missing (licence notes shown first) and switches;
4. **Advanced** (closed by default) - every new model per job, with Download / Use / Try it (Try it switches back
   automatically when it is slower), plus "Go back to the standard models".

:class:`NextModelsPanel` holds all the logic (tested without Tk); :func:`build` only draws it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.app.model_registry import MODE_TITLES, MODES, SLOT_TITLES, Registry

LICENSE_NOTES = {
    "OpenMDW-1.1": "OpenMDW-1.1 (permissive; its licence text is kept with the model)",
    "CC-BY-4.0 (weights, credit Kyutai); code MIT": "CC-BY-4.0: credit Kyutai; clone a voice only with consent",
}
GPL_NOTE = (
    "espeak-ng (Kokoro's pronunciation) and Piper are GPL-3.0: they are separate, optional downloads and are never "
    "bundled with Project Away."
)


def state_text(m: Any, ready: bool) -> str:
    """Plain words for what a model still needs on this PC."""
    if ready:
        return "ready"
    if "wake_model" in (getattr(m, "requires", None) or []):
        return "train a model for your name first (Settings > Models > Wake word)"
    dl = str(getattr(m, "download", "") or "")
    if dl.startswith("pip install"):
        return f"optional install: {dl}"
    size = round(int(getattr(m, "size", 0) or 0) / 1e6)
    return f"{size} MB to download" if size else "to download"


def title_with_flag(c: dict[str, Any]) -> str:
    """The title, plus "(experimental)" unless the title already says so."""
    title = str(c.get("title", ""))
    return title + (" (experimental)" if c.get("experimental") and "experimental" not in title.lower() else "")


class NextModelsPanel:
    def __init__(
        self,
        cfg: Any,
        save: Callable[[], None] | None = None,
        data_dir: Path | None = None,
        reg: Registry | None = None,
        ready: Callable[[Any], bool] | None = None,
        downloader: Callable[..., Any] | None = None,
        bench: Callable[..., Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.save = save or (lambda: None)
        self.data_dir = data_dir
        self.reg = reg or Registry(data_dir)
        self._ready = ready
        self._download = downloader
        self._bench = bench
        self.undo: dict[str, Any] | None = None

    # ----------------------------------------------------------------- reads
    def is_ready(self, m: Any) -> bool:
        if self._ready is not None:
            return bool(self._ready(m))
        from away.app import model_bench, model_fetch

        return model_bench._ready(m, self.cfg, self.data_dir, model_fetch)

    def headline(self) -> str:
        from away.app import model_bench

        rep = model_bench.last_report(self.data_dir)
        using = "the newer models" if self.cfg.models_next.presets == "next" else "the standard models"
        if rep is None:
            return f"Using {using}. Press Test my PC to see if the newer ones run well on this computer."
        return f"Using {using}. {model_bench.summary(rep)}"

    def presets(self) -> list[dict[str, Any]]:
        out = []
        for mode in MODES:
            try:
                plan = self.reg.preset_plan(mode, self.is_ready)
            except ValueError:
                continue
            missing = [r for r in plan if not r["ready"] and r["model"]]
            gb = sum(int(r["vram_mb"]) for r in plan) / 1000
            out.append(
                {
                    "mode": mode,
                    "title": MODE_TITLES[mode],
                    "vram": "no graphics card needed" if gb < 0.05 else f"about {gb:.1f} GB of graphics memory",
                    "missing": [r["title"] for r in missing],
                    "download_mb": sum(int(r.get("size_mb") or 0) for r in missing),
                    "experimental": [r["title"] for r in plan if r.get("experimental")],
                }
            )
        return out

    def advanced(self, show_restrictive: bool = False) -> list[dict[str, Any]]:
        """New models per job (the Advanced section)."""
        rows = []
        for slot in SLOT_TITLES:
            choices = []
            for m in self.reg.for_slot(slot, show_restrictive):
                if not (m.source == "catalog" or m.notes):
                    continue
                ok = self.is_ready(m)
                choices.append(
                    {
                        **m.public(),
                        "ready": ok,
                        "state": state_text(m, ok),
                        "license_note": LICENSE_NOTES.get(m.license, m.license),
                    }
                )
            if choices:
                cur = self.reg.current(self.cfg, slot)
                rows.append(
                    {"slot": slot, "title": SLOT_TITLES[slot], "current": cur.id if cur else None, "choices": choices}
                )
        return rows

    def license_notes(self, mode: str) -> list[str]:
        notes = []
        for r in self.reg.preset_plan(mode):
            m = self.reg.models.get(str(r["model"])) if r["model"] else None
            if m is not None and m.license in LICENSE_NOTES:
                notes.append(f"{m.title}: {LICENSE_NOTES[m.license]}")
        notes.append(GPL_NOTE)
        return notes

    # ----------------------------------------------------------------- one click
    def download_missing(self, mode: str, progress: Callable[[str], None] | None = None) -> list[str]:
        """Download what ``mode`` still needs. Returns plain-language problems (empty = all good)."""
        from away.app import model_fetch

        dl = self._download or (lambda m: model_fetch.download(m, self.data_dir))
        errors = []
        for r in self.reg.preset_plan(mode, self.is_ready):
            if r["ready"] or not r["model"]:
                continue
            m = self.reg.models[str(r["model"])]
            if "wake_model" in (m.requires or []) or (m.download and not m.url and not m.files):
                continue  # trained per name / its own v0.1.23 button: not a one-click download
            if progress is not None:
                progress(f"Downloading {m.title}...")
            try:
                dl(m)
            except Exception as exc:
                errors.append(f"{m.title}: {exc}")
        return errors

    def use_preset(self, mode: str) -> str:
        """Switch to the 2026-10 ``mode`` (whatever is ready); remembered per job for automatic rollback."""
        from away.app import model_guard

        before = {"models_next.presets": self.cfg.models_next.presets}
        self.cfg.models_next.presets = "next"
        for k, v in self.reg.apply_mode(self.cfg, mode, ready=self.is_ready).items():
            before.setdefault(k, v)
        model_guard.remember_all(before, self.data_dir)
        self.undo = before
        self.save()
        return f"Now using the newer {MODE_TITLES[mode]} models. Anything that fails or runs slower switches back."

    def test_my_pc(self, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        from away.app import model_bench

        run = self._bench or (lambda: model_bench.run_bench(self.cfg, self.data_dir, progress=progress).to_dict())
        rep = run()
        rec = rep.get("recommendation") or {}
        return {"passed": bool(rec.get("passed")), "preset": rec.get("preset"), "text": model_bench.summary(rep)}

    def standard(self) -> str:
        """Go back to the standard (v0.1.23) models."""
        from away.app import model_guard

        self.cfg.models_next.presets = "classic"
        for slot in list(model_guard.pending(self.data_dir)):
            model_guard.rollback(self.cfg, slot, "you chose the standard models", self.data_dir)
        self.cfg.models_next.stt = "whisper"
        self.save()
        return "Back to the standard models."


def build(win: Any, parent: Any) -> NextModelsPanel:  # pragma: no cover - Tk widgets
    """Under Settings > Performance (``win.m`` is the settings model: ``.cfg`` and ``.save()``)."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    m = getattr(win, "m", win)
    panel = NextModelsPanel(m.cfg, lambda: m.save())
    box = ttk.LabelFrame(parent, text="Newer models (optional)")
    box.pack(fill="x", pady=(10, 0))
    head = tk.StringVar(value=panel.headline())
    ttk.Label(box, textvariable=head, wraplength=640).pack(anchor="w", padx=6, pady=4)
    status = tk.StringVar(value="")
    row = ttk.Frame(box)
    row.pack(fill="x", padx=6)

    def in_bg(work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        def go() -> None:
            try:
                res: Any = work()
            except Exception as exc:
                res = exc
            parent.after(0, lambda: done(res))

        threading.Thread(target=go, daemon=True).start()

    def say(text: str) -> None:
        parent.after(0, lambda: status.set(text))

    def test() -> None:
        status.set("Testing this computer... (about a minute)")

        def done(res: Any) -> None:
            head.set(panel.headline())
            if isinstance(res, Exception):
                status.set(f"The test could not run: {res}")
                return
            status.set(res["text"])
            if res["passed"] and messagebox.askyesno("Project Away", f"{res['text']}\n\nUse it now?"):
                status.set(panel.use_preset(str(res["preset"])))
                head.set(panel.headline())

        in_bg(lambda: panel.test_my_pc(say), done)

    ttk.Button(row, text="Test my PC", command=test).pack(side="left")
    presets_box = ttk.Frame(box)
    presets_box.pack(fill="x", padx=6, pady=(4, 0))
    for i, p in enumerate(panel.presets()):

        def pick(mode: str = p["mode"], info: dict[str, Any] = p) -> None:
            text = f"{info['title']}: {info['vram']}."
            if info["missing"]:
                text += f"\nTo download (~{info['download_mb']} MB): " + ", ".join(info["missing"])
            if info["experimental"]:
                text += "\nExperimental: " + ", ".join(info["experimental"])
            text += "\n\n" + "\n".join(panel.license_notes(mode))
            if not messagebox.askyesno("Project Away", text + "\n\nContinue?"):
                return

            def work() -> str:
                errs = panel.download_missing(mode, say)
                msg = panel.use_preset(mode)
                return msg + (" Not downloaded: " + "; ".join(errs) if errs else "")

            def finish(res: Any) -> None:
                status.set(str(res))
                head.set(panel.headline())

            in_bg(work, finish)

        ttk.Button(presets_box, text=p["title"], width=10, command=pick).grid(row=i, column=0, sticky="w", pady=1)
        ttk.Label(presets_box, text=p["vram"]).grid(row=i, column=1, sticky="w", padx=8)
    ttk.Label(box, textvariable=status, wraplength=640, foreground="#444").pack(anchor="w", padx=6, pady=2)

    adv = ttk.Frame(box)
    shown = tk.BooleanVar(value=False)

    def toggle() -> None:
        if shown.get():
            adv.pack_forget()
            shown.set(False)
            adv_btn.configure(text="Advanced ▸")
            return
        adv_btn.configure(text="Advanced ▾")
        for w in adv.winfo_children():
            w.destroy()
        for r in panel.advanced():
            ttk.Label(adv, text=r["title"], font=("", 9, "bold")).pack(anchor="w", pady=(4, 0))
            for c in r["choices"]:
                ttk.Label(
                    adv,
                    text=f"  {title_with_flag(c)} - {c['runs_on']}, {c['license_note']}, {c['state']}",
                    wraplength=620,
                ).pack(anchor="w")
        ttk.Button(adv, text="Go back to the standard models", command=lambda: status.set(panel.standard())).pack(
            anchor="w", pady=4
        )
        adv.pack(fill="x", padx=12)
        shown.set(True)

    adv_btn = ttk.Button(box, text="Advanced ▸", command=toggle)
    adv_btn.pack(anchor="w", padx=6, pady=(0, 6))
    return panel
