"""Settings > Voice & input > "Voice" (v0.1.23): which voice reads your agents' messages aloud.

A picker (Natural (Kokoro) with five voices, Light (Piper) with two, Windows natural, Classic), a speed choice, a
Download button with a progress bar for the natural voices, and a Preview button. Plain words only; a voice that
cannot be used falls back to the next one without bothering the user.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.voice import natural

CHOICES: list[tuple[str, str, str]] = [  # (label, engine, voice)
    *[(f"Natural (Kokoro) - {v}", "kokoro", v) for v in natural.KOKORO_VOICES],
    *[(f"Light (Piper) - {v}", "piper", v) for v in natural.PIPER_VOICES],
    ("Windows natural", "windows", ""),
    ("Classic system voice", "classic", ""),
    ("Best available", "auto", ""),
]


def label_for(cfg: Any) -> str:
    s = cfg.speech
    if s.engine == "kokoro":
        return f"Natural (Kokoro) - {s.kokoro_voice}"
    if s.engine == "piper":
        return f"Light (Piper) - {s.piper_voice}"
    return next((lbl for lbl, e, _v in CHOICES if e == s.engine), "Best available")


def apply_label(cfg: Any, label: str) -> str:
    """Set speech.engine (+ its voice) from a picker label. Returns the engine."""
    for lbl, eng, v in CHOICES:
        if lbl == label:
            cfg.speech.engine = eng
            if eng == "kokoro":
                cfg.speech.kokoro_voice = v
            elif eng == "piper":
                cfg.speech.piper_voice = v
            return eng
    raise ValueError(f"unknown voice {label!r}")


def needs_download(cfg: Any, data_dir: Path | None = None) -> int:
    """Bytes still to download for the chosen voice (0 = ready or nothing to download)."""
    s = cfg.speech
    if s.engine == "kokoro":
        return natural.download_size("kokoro", data_dir=data_dir)
    if s.engine == "piper":
        return natural.download_size("piper", s.piper_voice, data_dir=data_dir)
    return 0


def status_text(cfg: Any, data_dir: Path | None = None) -> str:
    n = needs_download(cfg, data_dir)
    if n:
        return f"This voice needs a one-time download ({n / 1e6:.0f} MB). Until then the next voice is used."
    return "Ready."


def preview(cfg: Any, speaker_factory: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """Say a short sentence with the chosen voice (falls back like a real message)."""
    from away.voice.speech import Speaker

    sp = speaker_factory(cfg) if speaker_factory is not None else Speaker(lambda: cfg.speech)
    return dict(sp.say(natural.preview_text(cfg.speech.engine), force=True))


def build(win: Any, parent: Any, row: int) -> int:  # pragma: no cover - Tk widgets; smoke-tested under Xvfb
    tk, ttk = win.tk, win.ttk
    cfg = win.m.cfg
    box = ttk.LabelFrame(parent, text="Voice (how your agents sound)")
    box.grid(row=row, column=0, columnspan=4, sticky="we", pady=(10, 4))
    top = ttk.Frame(box)
    top.pack(fill="x", padx=4, pady=4)
    var = tk.StringVar(value=label_for(cfg))
    cb = ttk.Combobox(top, textvariable=var, values=[c[0] for c in CHOICES], state="readonly", width=30)
    cb.pack(side="left")
    speed = tk.StringVar(value=cfg.speech.rate)
    ttk.Label(top, text="  Speed").pack(side="left")
    sp = ttk.Combobox(top, textvariable=speed, values=["slow", "normal", "fast"], state="readonly", width=8)
    sp.pack(side="left")
    msg = tk.StringVar(value=status_text(cfg))
    row2 = ttk.Frame(box)
    row2.pack(fill="x", padx=4)
    ttk.Label(row2, textvariable=msg, wraplength=520, foreground="#555").pack(side="left")
    bar = ttk.Progressbar(row2, length=160, maximum=100.0)
    sp_holder: dict[str, Any] = {}

    def save() -> None:
        try:
            win.m.save()
        except Exception as exc:
            msg.set(str(exc))

    def picked(_e: Any = None) -> None:
        apply_label(cfg, var.get())
        save()
        msg.set(status_text(cfg))
        dl.configure(state="normal" if needs_download(cfg) else "disabled")

    def sped(_e: Any = None) -> None:
        cfg.speech.rate = speed.get()
        save()

    def do_preview() -> None:
        def run() -> None:
            try:
                from away.voice.speech import Speaker

                spk = sp_holder.get("s") or Speaker(lambda: cfg.speech)
                sp_holder["s"] = spk
                res = spk.say(natural.preview_text(cfg.speech.engine), force=True)
                text = (
                    f"Speaking with: {res.get('voice') or res.get('engine')}"
                    if res.get("spoken")
                    else res.get("reason", "")
                )
            except Exception as exc:
                text = f"Could not play a preview ({exc})"
            win._ui.post(lambda: msg.set(str(text)))

        threading.Thread(target=run, daemon=True, name="voice-preview").start()

    def do_download() -> None:
        s = cfg.speech
        eng = s.engine if s.engine in ("kokoro", "piper") else "kokoro"
        dl.configure(state="disabled")
        bar.pack(side="left", padx=6)
        msg.set("Downloading the voice...")

        def prog(done: int, total: int) -> None:
            win._ui.post(lambda: bar.configure(value=100.0 * done / max(total, 1)))

        def run() -> None:
            try:
                natural.download(eng, s.piper_voice, progress=prog)
                text = "Downloaded. Press Preview to hear it."
            except Exception as exc:
                text = f"The download did not finish ({str(exc)[:120]}). The next voice is used meanwhile."

            def done() -> None:
                bar.pack_forget()
                msg.set(text)
                dl.configure(state="normal" if needs_download(cfg) else "disabled")

            win._ui.post(done)

        threading.Thread(target=run, daemon=True, name="voice-download").start()

    btns = ttk.Frame(box)
    btns.pack(fill="x", padx=4, pady=4)
    ttk.Button(btns, text="\u25b6 Preview", command=do_preview).pack(side="left")
    dl = ttk.Button(btns, text="Download", command=do_download)
    dl.pack(side="left", padx=4)
    dl.configure(state="normal" if needs_download(cfg) else "disabled")
    cb.bind("<<ComboboxSelected>>", picked)
    sp.bind("<<ComboboxSelected>>", sped)
    return row + 1
