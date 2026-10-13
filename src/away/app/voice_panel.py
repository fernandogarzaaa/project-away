"""Settings > Voice & input: "Smart cleanup" and "My words" (the personal dictionary), made for non-technical users.

* one **Smart cleanup** switch; the first time it needs its model, one **Download** button shows the size and a
  progress bar (no model names here; Settings > Performance > Models lists the details);
* **My words**: a plain word list with Add / Remove (plus Import / Export);
* **Advanced** (collapsed): engine, time limit, tone per app;
* problems are shown as one friendly sentence with a button that fixes them (Download, Get Ollama, Learn how).

:class:`VoicePanel` holds the logic and is tested without Tk; :func:`build` only creates widgets. Kept out of
``settings_gui.py``, which calls :func:`build` once.
"""

from __future__ import annotations

import contextlib
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away.app import optional_models as om
from away.voice.cleanup_config import CLEANUP_BACKENDS, TONES
from away.voice.dictionary import Dictionary, default_path

HELP_URL = "https://github.com/fernandogarzaaa/project-away#smart-cleanup-and-my-words"
OLLAMA_URL = "https://ollama.com/download"
BACKEND_LABELS = {
    "llama_cpp": "Built-in (runs inside Project Away)",
    "ollama": "Ollama (a separate local app)",
    "openai": "Custom server (AWAY_LLM_BASE_URL)",
}
FIX_LABELS = {
    "download": "Download",
    "install_engine": "Learn how",
    "start_ollama": "Get Ollama",
    "pick_model": "Open advanced",
}


@dataclass
class Problem:
    message: str  # one friendly sentence
    fix: str  # download | install_engine | start_ollama | pick_model | none
    button: str  # the fix button's label ("" = no button)


def human_size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class VoicePanel:
    def __init__(
        self,
        cfg: Any,
        models_dir: Path | None = None,
        dictionary_path: Path | None = None,
        gpu: bool | None = None,
        engine_available: Callable[[], bool] | None = None,
        engine_offer: Callable[[], Any] | None = None,
        engine_install: Callable[..., Any] | None = None,
    ) -> None:
        self.cfg = cfg
        if models_dir is None:
            from away import paths

            models_dir = Path(cfg.models_dir) if getattr(cfg, "models_dir", None) else paths.models_dir()
        self.models_dir = models_dir
        f = cfg.voice.dictionary.file
        self.dictionary = Dictionary.load(
            dictionary_path or (Path(f).expanduser() if f else default_path()), cfg.voice.dictionary.learn_after
        )
        self._gpu = gpu
        if engine_available is None:
            from away.voice.llm_backends import llama_cpp_available

            engine_available = llama_cpp_available
        self.engine_available = engine_available
        if engine_offer is None or engine_install is None:
            from away.voice import engine

            engine_offer = engine_offer or engine.wheel_for
            engine_install = engine_install or engine.install
        self.engine_offer = engine_offer  # the ready-made engine wheel for this computer, or None (v0.1.20)
        self.engine_install = engine_install

    # ------------------------------------------------------------------ smart cleanup
    @property
    def c(self) -> Any:
        return self.cfg.voice.cleanup

    def gpu(self) -> bool:
        if self._gpu is None:
            try:
                from away.app.performance import current_profile

                self._gpu = current_profile(self.cfg).device == "cuda"
            except Exception:
                self._gpu = False
        return bool(self._gpu)

    def smart_on(self) -> bool:
        return bool(self.c.mode == "llm" or getattr(self.cfg.voice, "llm_cleanup", False))

    def model_key(self) -> str:
        from away.voice.llm_backends import pick_model_key

        return pick_model_key(self.models_dir, self.gpu(), self.c.model)

    def _registry_model(self) -> om.OptionalModel | None:
        try:
            return om.get(self.model_key())
        except KeyError:
            return None  # a .gguf path of the user's own

    def model_ready(self) -> bool:
        if self.c.backend != "llama_cpp":
            return True
        m = self._registry_model()
        if m is None:
            return Path(self.model_key()).expanduser().is_file()
        return om.installed(m.key, self.models_dir)

    def _engine_wheel(self) -> Any:
        """The prebuilt engine to fetch with Download, when the engine is missing and one exists."""
        if self.c.backend != "llama_cpp" or self.engine_available():
            return None
        try:
            return self.engine_offer()
        except Exception:
            return None

    def download_bytes(self) -> int:
        m = self._registry_model()
        n = m.size_bytes if m is not None and not self.model_ready() else 0
        w = self._engine_wheel()
        return n + (int(w.size) if w is not None else 0)

    def download_label(self) -> str:
        n = self.download_bytes()
        return f"Download ({human_size(n)})" if n else "Download"

    def problem(self, check_server: bool = False) -> Problem | None:
        """What stops Smart cleanup from working, in plain words (None = ready, or Smart cleanup is off)."""
        if not self.smart_on():
            return None
        backend = self.c.backend
        if backend == "llama_cpp":
            if not self.model_ready() or self._engine_wheel() is not None:
                n = self.download_bytes()
                size = f" ({human_size(n)}, one time)" if n else ""
                return Problem(f"Smart cleanup needs a download{size}.", "download", self.download_label())
            if not self.engine_available():
                return Problem(
                    "This copy of Project Away can't run Smart cleanup yet. Basic cleanup keeps working.",
                    "install_engine",
                    FIX_LABELS["install_engine"],
                )
            return None
        if backend == "ollama" and check_server:
            from away.voice.llm_backends import EngineError, OllamaBackend

            model = self.c.model if self.c.model != "auto" else ("qwen2.5:1.5b" if self.gpu() else "qwen2.5:0.5b")
            try:
                OllamaBackend(self.c.ollama_url, model).check()
            except EngineError as exc:
                if exc.fix == "start_ollama":
                    return Problem("Ollama isn't running. Start it, or install it first.", exc.fix, "Get Ollama")
                return Problem(f"Ollama needs the model: run 'ollama pull {model}'.", exc.fix, "Learn how")
        if backend == "openai":
            import os

            if not os.environ.get("AWAY_LLM_BASE_URL"):
                return Problem("No custom server is set up (AWAY_LLM_BASE_URL).", "install_engine", "Learn how")
        return None

    def set_smart(self, on: bool) -> Problem | None:
        """Turn Smart cleanup on/off. Turning it on is kept even when a download is still needed: the app
        then uses basic cleanup until the model is there."""
        self.c.mode = "llm" if on else "rules"
        if not on and getattr(self.cfg.voice, "llm_cleanup", False):
            self.cfg.voice.llm_cleanup = False
        return self.problem()

    def download(self, progress: Callable[[int, int], None] | None = None) -> str | None:
        """One-click download (resumes): the model and, in the installed app, the engine (v0.1.20: a pinned,
        sha256-checked llama-cpp-python wheel, see :mod:`away.voice.engine`). Returns a friendly error or None."""
        m = self._registry_model()
        w = self._engine_wheel()
        if m is None and w is None:
            return "Your own model file is used; nothing to download."
        total = max(1, self.download_bytes())
        done_before = [0]

        def step(done: int, _total: int) -> None:
            if progress is not None:
                progress(min(total, done_before[0] + done), total)

        if m is not None and not self.model_ready():
            try:
                om.install(m.key, self.models_dir, progress=step)
            except Exception as exc:
                return f"The download stopped ({type(exc).__name__}). Check your internet and press Download to resume."
            done_before[0] += m.size_bytes
        if w is not None:
            try:
                self.engine_install(progress=step)
            except Exception as exc:
                msg = str(exc) if type(exc).__name__ == "EngineInstallError" else ""
                return msg or f"The engine download stopped ({type(exc).__name__}). Press Download to try again."
        return None

    def fix_action(self, fix: str) -> str | None:
        """For the buttons that only open a page; returns the URL opened (None for download/advanced)."""
        url = {"install_engine": HELP_URL, "start_ollama": OLLAMA_URL}.get(fix)
        if url:
            with contextlib.suppress(Exception):
                webbrowser.open(url)
        return url

    # ------------------------------------------------------------------ advanced
    def backend_choices(self) -> list[str]:
        return [BACKEND_LABELS[b] for b in CLEANUP_BACKENDS]

    def backend_label(self) -> str:
        return BACKEND_LABELS.get(self.c.backend, self.c.backend)

    def set_backend_label(self, label: str) -> str | None:
        for key, lab in BACKEND_LABELS.items():
            if lab == label or key == label:
                self.c.backend = key
                return None
        return f"unknown engine {label!r}"

    def set_timeout_ms(self, value: Any) -> str | None:
        try:
            v = int(float(value))
        except (TypeError, ValueError):
            return "the time limit must be a number of milliseconds"
        if not 100 <= v <= 30000:
            return "the time limit must be between 100 and 30000 ms"
        self.c.timeout_ms = v
        return None

    def tones_text(self) -> str:
        return "\n".join(f"{app} = {tone}" for app, tone in self.c.tones.items())

    def set_tones_text(self, text: str) -> str | None:
        """Lines of ``app = tone`` (``slack.exe = casual``). Returns an error for the first bad line."""
        out: dict[str, str] = {}
        for n, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                return f"line {n}: write it as app = tone (e.g. slack.exe = casual)"
            app, tone = (s.strip() for s in line.split("=", 1))
            if not app or tone.lower() not in TONES:
                return f"line {n}: the tone must be one of {', '.join(TONES)}"
            out[app] = tone.lower()
        self.c.tones = out
        return None

    # ------------------------------------------------------------------ my words
    def words(self) -> list[str]:
        self.dictionary.reload_if_changed()
        return sorted(self.dictionary.words(), key=str.lower)

    def add_word(self, word: str) -> str | None:
        err = self.dictionary.add(word)
        return f"Couldn't add that: {err}." if err else None

    def remove_word(self, word: str) -> bool:
        return self.dictionary.remove(word)

    def import_words(self, path: Path) -> tuple[str | None, int]:
        try:
            return None, self.dictionary.import_file(path)
        except Exception as exc:
            return f"Couldn't read that file ({type(exc).__name__}). Use a .txt word list or a .json export.", 0

    def export_words(self, path: Path) -> str | None:
        try:
            self.dictionary.export_file(path)
        except Exception as exc:
            return f"Couldn't save the file ({type(exc).__name__})."
        return None


def build(win: Any, parent: Any, row: int) -> VoicePanel:  # pragma: no cover - Tk widgets
    """Add Smart cleanup + My words (+ collapsed Advanced) to the Voice tab grid ``parent`` from ``row`` on."""
    tk, ttk = win.tk, win.ttk
    panel = VoicePanel(win.m.cfg)
    box = ttk.LabelFrame(parent, text="Smart cleanup")
    box.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 4))
    smart_var = tk.BooleanVar(value=panel.smart_on())
    ttk.Checkbutton(
        box,
        text="Smart cleanup: fix punctuation and match the app you type in (casual in chats, formal in email)",
        variable=smart_var,
        command=lambda: on_toggle(),
    ).pack(anchor="w", padx=4, pady=(4, 0))
    ttk.Label(
        box,
        text="Runs on this PC. If it is ever slow, basic cleanup is typed instead, so you never wait.",
        foreground="#555",
        wraplength=640,
    ).pack(anchor="w", padx=22)
    prob_row = ttk.Frame(box)
    prob_row.pack(fill="x", padx=4, pady=4)
    prob_var = tk.StringVar()
    ttk.Label(prob_row, textvariable=prob_var, wraplength=420, foreground="#a15c00").pack(side="left")
    fix_btn = ttk.Button(prob_row, text="", command=lambda: on_fix())
    bar = ttk.Progressbar(prob_row, length=140, maximum=100.0)
    state: dict[str, Any] = {"fix": "none", "downloading": False}

    adv_open = tk.BooleanVar(value=False)
    adv_btn = ttk.Button(box, text="\u25b8 Advanced", command=lambda: toggle_adv())
    adv_btn.pack(anchor="w", padx=4)
    adv = ttk.Frame(box)

    def show_problem() -> None:
        p = panel.problem()
        fix_btn.pack_forget()
        if state["downloading"]:
            return
        bar.pack_forget()
        if p is None:
            prob_var.set("Ready." if panel.smart_on() else "")
            state["fix"] = "none"
            return
        prob_var.set(p.message)
        state["fix"] = p.fix
        if p.button:
            fix_btn.configure(text=p.button)
            fix_btn.pack(side="right", padx=4)

    def mark() -> None:
        win._report(None, None)

    def on_toggle() -> None:
        panel.set_smart(bool(smart_var.get()))
        show_problem()
        mark()

    def on_fix() -> None:
        fix = state["fix"]
        if fix == "download":
            start_download()
        elif fix == "pick_model":
            if not adv_open.get():
                toggle_adv()
        else:
            panel.fix_action(fix)

    def start_download() -> None:
        state["downloading"] = True
        fix_btn.pack_forget()
        bar["value"] = 0
        bar.pack(side="right", padx=4)
        prob_var.set("Downloading... you can keep using Project Away.")

        def progress(done: int, total: int) -> None:
            win._ui.post(lambda: bar.configure(value=100.0 * done / max(total, 1)))

        def run() -> None:
            err = panel.download(progress)

            def done() -> None:
                state["downloading"] = False
                show_problem()
                if err:
                    prob_var.set(err)
                    state["fix"] = "download"
                    fix_btn.configure(text=panel.download_label())
                    fix_btn.pack(side="right", padx=4)
                else:
                    win._report(None, "Smart cleanup is ready (restart Project Away if it is running)")

            win._ui.post(done)

        threading.Thread(target=run, daemon=True, name="download-cleanup").start()

    # advanced (collapsed)
    ttk.Label(adv, text="Engine").grid(row=0, column=0, sticky="w", pady=2)
    be_var = tk.StringVar(value=panel.backend_label())
    be = ttk.Combobox(adv, textvariable=be_var, values=panel.backend_choices(), state="readonly", width=34)
    be.grid(row=0, column=1, sticky="w")

    def on_backend(_e: Any = None) -> None:
        win._report(panel.set_backend_label(be_var.get()), "engine changed")
        show_problem()

    be.bind("<<ComboboxSelected>>", on_backend)
    ttk.Label(adv, text="Time limit (ms)").grid(row=1, column=0, sticky="w", pady=2)
    to_var = tk.StringVar(value=str(panel.c.timeout_ms))
    to = ttk.Entry(adv, textvariable=to_var, width=8)
    to.grid(row=1, column=1, sticky="w")
    for ev in ("<FocusOut>", "<Return>"):
        to.bind(ev, lambda _e: win._report(panel.set_timeout_ms(to_var.get()), "time limit saved"))
    ttk.Label(adv, text="Tone per app (one per line, e.g. slack.exe = casual; tones: " + ", ".join(TONES) + ")").grid(
        row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
    )
    tones = tk.Text(adv, height=4, width=48)
    tones.insert("1.0", panel.tones_text())
    tones.grid(row=3, column=0, columnspan=2, sticky="w")
    tones.bind("<FocusOut>", lambda _e: win._report(panel.set_tones_text(tones.get("1.0", "end")), "tones saved"))

    def toggle_adv() -> None:
        if adv_open.get():
            adv.pack_forget()
            adv_open.set(False)
            adv_btn.configure(text="\u25b8 Advanced")
        else:
            adv.pack(fill="x", padx=22, pady=(0, 6))
            adv_open.set(True)
            adv_btn.configure(text="\u25be Advanced")

    # my words
    wbox = ttk.LabelFrame(parent, text="My words (names and terms dictation should get right)")
    wbox.grid(row=row + 1, column=0, columnspan=3, sticky="we", pady=4)
    lb = tk.Listbox(wbox, height=6, width=36)
    lb.grid(row=0, column=0, rowspan=4, sticky="w", padx=4, pady=4)
    word_var = tk.StringVar()
    ent = ttk.Entry(wbox, textvariable=word_var, width=24)
    ent.grid(row=0, column=1, sticky="w", padx=4)

    def refresh_words() -> None:
        lb.delete(0, "end")
        for w in panel.words():
            lb.insert("end", w)

    def add() -> None:
        err = panel.add_word(word_var.get())
        if not err:
            word_var.set("")
        win._report(err, "word added")
        refresh_words()

    def remove() -> None:
        sel = lb.curselection()
        if not sel:
            win._report("Pick a word in the list first.")
            return
        panel.remove_word(lb.get(sel[0]))
        win._report(None, "word removed")
        refresh_words()

    def do_import() -> None:
        from tkinter import filedialog

        p = filedialog.askopenfilename(parent=parent, filetypes=[("Word lists", "*.txt *.json"), ("All files", "*")])
        if p:
            err, n = panel.import_words(Path(p))
            win._report(err, f"imported {n} word(s)")
            refresh_words()

    def do_export() -> None:
        from tkinter import filedialog

        p = filedialog.asksaveasfilename(
            parent=parent, defaultextension=".txt", filetypes=[("Text", "*.txt"), ("JSON", "*.json")]
        )
        if p:
            win._report(panel.export_words(Path(p)), "word list saved")

    ent.bind("<Return>", lambda _e: add())
    ttk.Button(wbox, text="Add", command=add).grid(row=0, column=2, padx=2)
    ttk.Button(wbox, text="Remove", command=remove).grid(row=1, column=1, sticky="w", padx=4)
    ttk.Button(wbox, text="Import...", command=do_import).grid(row=2, column=1, sticky="w", padx=4)
    ttk.Button(wbox, text="Export...", command=do_export).grid(row=3, column=1, sticky="w", padx=4)
    ttk.Label(
        wbox,
        text='Tip: say "add Jonie to my dictionary", or "correct Jonnie to Jonie" right after a mistake. '
        "Words you fix by hand are learned too. Saved right away.",
        foreground="#555",
        wraplength=640,
    ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 4))
    refresh_words()
    show_problem()
    return panel
