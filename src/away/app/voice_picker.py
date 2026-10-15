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

from away.voice import natural, voice_profiles

CREATE_LABEL = "Manage custom voices..."
DISABLE_CUSTOM_LABEL = "Use preset voice (disable custom voice)"


CHOICES: list[tuple[str, str, str]] = [  # (label, engine, voice)
    *[(f"Natural (Kokoro) - {v}", "kokoro", v) for v in natural.KOKORO_VOICES],
    *[(f"Light (Piper) - {v}", "piper", v) for v in natural.PIPER_VOICES],
    ("Windows natural", "windows", ""),
    ("Classic system voice", "classic", ""),
    ("Best available", "auto", ""),
]


def custom_choices(cfg: Any, data_dir: Path | None = None) -> dict[str, str]:
    return {f"Custom voice - {p.name} [{p.id[:6]}]": p.id for p in voice_profiles.list_profiles(data_dir)}


def label_for(cfg: Any) -> str:
    s = cfg.speech
    selected = str(getattr(s, "custom_voice_profile", "") or "")
    if selected:
        for label, profile_id in custom_choices(cfg).items():
            if profile_id == selected:
                return label
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
    if str(getattr(s, "custom_voice_profile", "") or ""):
        return 0
    if s.engine == "kokoro":
        return natural.download_size("kokoro", data_dir=data_dir)
    if s.engine == "piper":
        return natural.download_size("piper", s.piper_voice, data_dir=data_dir)
    return 0


def status_text(cfg: Any, data_dir: Path | None = None) -> str:
    selected = str(getattr(cfg.speech, "custom_voice_profile", "") or "")
    if selected:
        try:
            profile = voice_profiles.get_profile(selected, data_dir)
            if profile.engine == "pocket-clone":
                from away.voice.voice_cloning import pocket_available

                status = (
                    "Ready; Pocket TTS."
                    if pocket_available()
                    else ("Install the optional Pocket TTS extra; preset fallback remains available.")
                )
                return "Custom voice is local. " + status
            from away.voice.voice_cloning import qwen_available

            status = (
                "Ready; Qwen3-TTS."
                if qwen_available()
                else ("Install the optional Qwen3-TTS extra; preset fallback remains available.")
            )
            return "Custom voice is local. " + status
        except Exception:
            return "Custom voice profile is unavailable; the existing voice fallback chain will be used."
    n = needs_download(cfg, data_dir)
    if n:
        return f"This voice needs a one-time download ({n / 1e6:.0f} MB). Until then the next voice is used."
    return "Ready."


def license_text(cfg: Any) -> str:
    """Explain optional voice-engine licences before downloads."""
    selected = str(getattr(cfg.speech, "custom_voice_profile", "") or "")
    if selected:
        try:
            profile = voice_profiles.get_profile(selected)
            if profile.engine == "pocket-clone":
                return (
                    "Pocket TTS weights are CC-BY-4.0 (credit Kyutai). The reference WAV and reusable voice state "
                    "stay in your local Project Away data folder."
                )
            return (
                "Qwen3-TTS uses its own model licence. The reference WAV stays local; the clone prompt is held "
                "in memory, not serialized."
            )
        except Exception:
            return "Custom voice profile is unavailable."
    eng = cfg.speech.engine if cfg.speech.engine in ("kokoro", "piper") else ""
    if not eng:
        return ""
    return natural.LICENSE_NOTE + "\n" + "; ".join(natural.license_lines(eng))


def preview(cfg: Any, speaker_factory: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """Say a short sentence with the chosen voice (falls back like a real message)."""
    from away.voice.speech import Speaker

    sp = speaker_factory(cfg) if speaker_factory is not None else Speaker(lambda: cfg.speech)
    return dict(sp.say(natural.preview_text(cfg.speech.engine), force=True))


def manage_custom_voices(win: Any, parent: Any, cfg: Any, on_change: Callable[[], None]) -> None:
    """Local-only profile manager with explicit speaker permission and deletion."""
    tk, ttk = win.tk, win.ttk
    from tkinter import filedialog, messagebox

    dialog = tk.Toplevel(parent)
    dialog.title("Voice Studio · Custom voices")
    dialog.transient(parent.winfo_toplevel())
    dialog.grab_set()
    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Create a custom voice", font=("", 11, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w"
    )
    ttk.Label(
        frame, text="Import 3–30 seconds of clear PCM WAV. The sample and profile stay on this PC.", wraplength=480
    ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 8))
    name = tk.StringVar(value="My voice")
    speaker = tk.StringVar(value="Me")
    engine = tk.StringVar(value="pocket-clone")
    transcript = tk.StringVar()
    sample = tk.StringVar()
    consent = tk.BooleanVar(value=False)
    ttk.Label(frame, text="Profile name").grid(row=2, column=0, sticky="w")
    ttk.Entry(frame, textvariable=name, width=38).grid(row=2, column=1, columnspan=2, sticky="we")
    ttk.Label(frame, text="Whose voice?").grid(row=3, column=0, sticky="w")
    ttk.Entry(frame, textvariable=speaker, width=38).grid(row=3, column=1, columnspan=2, sticky="we")
    ttk.Label(frame, text="Engine").grid(row=4, column=0, sticky="w")
    ttk.Combobox(frame, textvariable=engine, values=["pocket-clone", "qwen-clone"], state="readonly", width=35).grid(
        row=4, column=1, columnspan=2, sticky="we"
    )
    ttk.Label(frame, text="Reference transcript (Qwen required)").grid(row=5, column=0, sticky="w")
    ttk.Entry(frame, textvariable=transcript, width=38).grid(row=5, column=1, columnspan=2, sticky="we")
    ttk.Label(frame, text="WAV sample").grid(row=6, column=0, sticky="w")
    ttk.Entry(frame, textvariable=sample, width=30).grid(row=6, column=1, sticky="we")

    def browse() -> None:
        chosen = filedialog.askopenfilename(parent=dialog, filetypes=[("WAV audio", "*.wav")])
        if chosen:
            sample.set(chosen)

    ttk.Button(frame, text="Browse…", command=browse).grid(row=6, column=2, padx=(4, 0))
    ttk.Checkbutton(
        frame, text="I have permission from this speaker to create and use this voice clone.", variable=consent
    ).grid(row=7, column=0, columnspan=3, sticky="w", pady=6)
    ttk.Label(
        frame,
        text="Do not clone another person's voice without their informed permission. Voice samples are sensitive data.",
        wraplength=480,
    ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(0, 6))

    def refresh() -> None:
        listing.delete(0, "end")
        profile_ids.clear()
        for profile in voice_profiles.list_profiles():
            profile_ids.append(profile.id)
            listing.insert("end", f"{profile.name} · {profile.engine} · {profile.duration_s:.1f}s")

    def create() -> None:
        try:
            profile = voice_profiles.create_profile(
                name.get(),
                engine.get(),
                Path(sample.get()),
                consent_confirmed=bool(consent.get()),
                speaker_name=speaker.get(),
                reference_text=transcript.get(),
            )
        except Exception as exc:
            messagebox.showerror("Could not create voice", str(exc), parent=dialog)
            return
        cfg.speech.custom_voice_profile = profile.id
        on_change()
        refresh()
        messagebox.showinfo("Voice created", f"{profile.name} was saved locally and selected.", parent=dialog)

    ttk.Button(frame, text="Create and select", command=create).grid(row=9, column=1, sticky="e", pady=(2, 8))
    ttk.Separator(frame).grid(row=10, column=0, columnspan=3, sticky="we", pady=5)
    ttk.Label(frame, text="Saved profiles").grid(row=11, column=0, columnspan=3, sticky="w")
    listing = tk.Listbox(frame, height=5, width=58)
    listing.grid(row=12, column=0, columnspan=2, sticky="we")
    profile_ids: list[str] = []

    def delete_selected() -> None:
        sel = listing.curselection()
        if not sel:
            return
        profile_id = profile_ids[sel[0]]
        if not messagebox.askyesno("Delete voice", "Delete its WAV, metadata and cached voice state?", parent=dialog):
            return
        voice_profiles.delete_profile(profile_id)
        if getattr(cfg.speech, "custom_voice_profile", "") == profile_id:
            cfg.speech.custom_voice_profile = ""
        on_change()
        refresh()

    ttk.Button(frame, text="Delete selected", command=delete_selected).grid(row=12, column=2, sticky="n", padx=(4, 0))
    ttk.Button(frame, text="Done", command=dialog.destroy).grid(row=13, column=2, sticky="e", pady=(8, 0))
    refresh()
    dialog.columnconfigure(1, weight=1)


def build(win: Any, parent: Any, row: int) -> int:  # pragma: no cover - Tk widgets; smoke-tested under Xvfb
    tk, ttk = win.tk, win.ttk
    cfg = win.m.cfg
    box = ttk.LabelFrame(parent, text="Voice (how your agents sound)")
    box.grid(row=row, column=0, columnspan=4, sticky="we", pady=(10, 4))
    top = ttk.Frame(box)
    top.pack(fill="x", padx=4, pady=4)
    var = tk.StringVar(value=label_for(cfg))
    custom_labels = custom_choices(cfg)
    cb = ttk.Combobox(
        top,
        textvariable=var,
        values=[c[0] for c in CHOICES] + list(custom_labels) + [CREATE_LABEL, DISABLE_CUSTOM_LABEL],
        state="readonly",
        width=38,
    )
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
    lic = tk.StringVar(value=license_text(cfg))  # v0.1.23: licences of the opt-in download, shown up front
    ttk.Label(box, textvariable=lic, wraplength=620, foreground="#666", justify="left").pack(fill="x", padx=4)
    sp_holder: dict[str, Any] = {}

    def save() -> None:
        try:
            win.m.save()
        except Exception as exc:
            msg.set(str(exc))

    def refresh_choices() -> None:
        nonlocal custom_labels
        custom_labels = custom_choices(cfg)
        cb.configure(values=[c[0] for c in CHOICES] + list(custom_labels) + [CREATE_LABEL, DISABLE_CUSTOM_LABEL])
        var.set(label_for(cfg))
        msg.set(status_text(cfg))
        lic.set(license_text(cfg))
        dl.configure(state="normal" if needs_download(cfg) else "disabled")

    def profile_changed() -> None:
        save()
        refresh_choices()

    def picked(_e: Any = None) -> None:
        label = var.get()
        if label == CREATE_LABEL:
            manage_custom_voices(win, parent, cfg, profile_changed)
            refresh_choices()
            return
        if label == DISABLE_CUSTOM_LABEL:
            cfg.speech.custom_voice_profile = ""
        elif label in custom_labels:
            cfg.speech.custom_voice_profile = custom_labels[label]
        else:
            cfg.speech.custom_voice_profile = ""
            apply_label(cfg, label)
        save()
        refresh_choices()

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
