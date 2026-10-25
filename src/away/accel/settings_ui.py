"""Settings > Performance (v0.1.24): "Hand and face tracking runs on" (Automatic / Graphics card / Processor), what
it really uses right now in plain words, a Download button for the graphics-card models, and "Run self-test".
"""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import Any

LABELS = {"auto": "Automatic (recommended)", "gpu": "Graphics card", "cpu": "Processor"}
MAC_GPU_LABEL = "Apple GPU / Neural Engine"  # v0.1.35: Core ML on macOS


def labels(platform: str | None = None) -> dict[str, str]:
    """The choices as shown on this OS (v0.1.35: macOS names the Apple GPU / Neural Engine; others unchanged)."""
    plat = platform or sys.platform
    return {**LABELS, "gpu": MAC_GPU_LABEL} if plat == "darwin" else dict(LABELS)


def status_line(st: dict[str, Any]) -> str:
    """One plain sentence for the current state (tests read it too)."""
    using = st.get("using") or "processor"
    if using == "graphics card" and "Core ML" in str(st.get("provider") or ""):
        # v0.1.35: Core ML is kept per model only where it measured faster; the reason names which ones
        return (
            f"Tracking uses the Apple Neural Engine / GPU (Core ML): {st.get('reason') or 'faster than the processor'}."
        )
    if using == "graphics card":
        return f"Tracking runs on your graphics card ({st.get('provider') or 'GPU'})."
    if st.get("provider") and not st.get("models_ready"):
        return (
            f"Tracking runs on the processor. Your graphics card ({st['provider']}) can take it over after a one-time "
            f"~{st.get('download_mb') or 23} MB download."
        )
    return f"Tracking runs on the processor: {st.get('reason') or 'your choice'}."


def label_to_value(label: str) -> str:
    if label == MAC_GPU_LABEL:
        return "gpu"
    return next((k for k, v in LABELS.items() if v == label), "auto")


def build(win: Any, parent: Any) -> None:  # pragma: no cover - Tk widgets; smoke-tested under Xvfb
    from away import paths
    from away.accel import compute
    from away.accel import models as gm
    from away.app.frozen import away_command

    tk, ttk = win.tk, win.ttk
    box = ttk.LabelFrame(parent, text="Hand and face tracking")
    box.pack(fill="x", pady=(10, 4))
    row = ttk.Frame(box)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Runs on:").pack(side="left")
    shown = labels()
    var = tk.StringVar(value=shown.get(str(win.m.cfg.hands.compute), shown["auto"]))
    cb = ttk.Combobox(row, textvariable=var, values=list(shown.values()), state="readonly", width=24)
    cb.pack(side="left", padx=6)
    line = tk.StringVar()
    ttk.Label(box, textvariable=line, wraplength=640).pack(anchor="w", pady=2)
    brow = ttk.Frame(box)
    brow.pack(anchor="w", pady=2)
    dl = ttk.Button(brow, text="Download graphics-card models (~23 MB)")
    dl.pack(side="left", padx=2)

    def refresh() -> None:
        st = compute.status(win.m.cfg, paths.data_dir())
        line.set(status_line(st))
        dl.configure(state="normal" if st.get("provider") and not st.get("models_ready") else "disabled")

    def on_pick(_e: Any = None) -> None:
        win.m.cfg.hands.compute = label_to_value(var.get())
        win._report(None, "saved on Save; applies the next time tracking starts")
        refresh()

    def download() -> None:
        dl.configure(state="disabled")
        line.set("Downloading the graphics-card models...")

        def work() -> None:
            try:
                gm.download("hands", paths.data_dir())
                gm.download("face", paths.data_dir())
                msg = None
            except Exception as exc:
                msg = str(exc)

            def done() -> None:
                win._report(msg, "downloaded")
                refresh()

            parent.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    dl.configure(command=download)
    cb.bind("<<ComboboxSelected>>", on_pick)
    st_box = ttk.LabelFrame(parent, text="Self-test")
    st_box.pack(fill="x", pady=(10, 4))
    ttk.Label(
        st_box,
        text="Checks your cameras, hand tracking, pointer, microphone, voices, pop-ups, Agent Link and models, and "
        "says what to fix. The report goes to every connected AI agent.",
        wraplength=640,
    ).pack(anchor="w")
    ttk.Button(
        st_box,
        text="Run self-test",
        command=lambda: subprocess.Popen(away_command(["selftest", "--window", "--trigger", "Settings"])),
    ).pack(anchor="w", pady=4)
    refresh()
