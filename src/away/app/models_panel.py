"""Settings > Performance: the "Models" list (optional downloads) and the "Eye tracking" buttons.

Kept out of ``settings_gui.py`` on purpose (that file only calls :func:`build`). :class:`ModelsPanel` holds
the logic and is tested without Tk; the widgets are built from :data:`away.app.optional_models.MODELS`, so a
new optional model shows up here by adding one entry there.

* every model row: name, size, license note, status, Download / Remove, a progress bar while downloading;
  a model with restricted terms asks for an explicit opt-in first;
* the running app picks a downloaded gaze model up by itself (``gaze.model: auto`` is re-resolved every few
  seconds), so no restart is needed; the screen map needs a recalibration to use the model's features;
* Eye tracking: Calibrate (runs ``away gaze calibrate`` in its own process, it borrows the camera from the
  app), Test (``away gaze test``) and Clear.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.app import optional_models as om


class ModelsPanel:
    def __init__(self, cfg: Any, config_path: Path | None = None, models_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        if models_dir is None:
            from away import paths

            models_dir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
        self.models_dir = models_dir

    def rows(self) -> list[dict[str, Any]]:
        return om.describe(self.models_dir)

    def needs_consent(self, key: str) -> str | None:
        """The notice to confirm before downloading (None = no opt-in needed)."""
        m = om.get(key)
        return m.notice if m.restricted else None

    def install(self, key: str, accepted: bool, progress: Callable[[int, int], None] | None = None) -> str | None:
        """Download (resumes an interrupted one). Returns an error message or None."""
        try:
            om.install(key, self.models_dir, accept_license=accepted, progress=progress)
        except om.LicenseNotAccepted:
            return "not downloaded: the license was not accepted"
        except Exception as exc:
            return f"download failed: {exc} (Download again to resume)"
        return None

    def remove(self, key: str) -> str:
        return "removed" if om.remove(key, self.models_dir) else "was not installed"

    def gaze_backend(self) -> str:
        return om.resolve_gaze_model(self.cfg.gaze.model, self.models_dir)

    def gaze_command(self, action: str) -> list[str]:
        from away.app.frozen import away_command

        args = ["gaze", action]
        if self.config_path is not None:
            args += ["--config", str(self.config_path)]
        return away_command(args)

    def gaze_status(self) -> str:
        from away.face import gaze as gz
        from away.face.headgaze import calibration_camera

        cam = calibration_camera(self.cfg)
        cal = gz.load_calibration(cam)
        backend = self.gaze_backend()
        if cal is None:
            return f"Eye tracking: not calibrated for camera {cam!r} (gaze features stay off). Backend: {backend}."
        acc = cal.accuracy_px()
        extra = f", ~{acc:.0f} px typical error" if acc is not None else ""
        net = f"; uses {cal.net_model}" if cal.net_model else ""
        hint = ""
        if backend != "iris" and cal.net_model != backend:
            hint = f" Recalibrate to use the {backend} model."
        return f"Eye tracking: calibrated ({cal.n_dots} dots{extra}{net}). Backend: {backend}.{hint}"

    def clear_gaze(self) -> str:
        from away.face import gaze as gz
        from away.face.headgaze import calibration_camera

        return "screen map cleared" if gz.clear_calibration(calibration_camera(self.cfg)) else "nothing to clear"


def build(win: Any, parent: Any) -> ModelsPanel:  # pragma: no cover - Tk widgets
    """Add the Models + Eye tracking sections to ``parent`` (a SettingsWindow tab frame)."""
    tk, ttk = win.tk, win.ttk
    panel = ModelsPanel(win.m.cfg, getattr(win.m, "path", None))
    box = ttk.LabelFrame(parent, text="Models (optional downloads)")
    box.pack(fill="x", pady=(10, 4))
    widgets: dict[str, dict[str, Any]] = {}

    def refresh() -> None:
        for row in panel.rows():
            w = widgets[row["key"]]
            w["status"].set(row["status"])
            w["dl"].configure(state="disabled" if row["installed"] else "normal")
            w["rm"].configure(state="normal" if row["installed"] else "disabled")
        gaze_var.set(panel.gaze_status())

    def download(key: str) -> None:
        notice = panel.needs_consent(key)
        accepted = True
        if notice is not None:
            from tkinter import messagebox

            accepted = bool(
                messagebox.askyesno("License: research use only", notice + "\n\nAccept and download?", parent=parent)
            )
            if not accepted:
                win._report(None, "not downloaded (license not accepted)")
                return
        w = widgets[key]
        w["dl"].configure(state="disabled")
        w["bar"]["value"] = 0

        def progress(done: int, total: int) -> None:
            win._ui.post(lambda: w["bar"].configure(value=100.0 * done / max(total, 1)))

        def run() -> None:
            err = panel.install(key, accepted, progress)

            def done() -> None:
                win._report(err, f"{om.get(key).name} installed")
                refresh()

            win._ui.post(done)

        threading.Thread(target=run, daemon=True, name=f"download-{key}").start()

    def remove(key: str) -> None:
        win._report(None, f"{om.get(key).name}: {panel.remove(key)}")
        widgets[key]["bar"]["value"] = 0
        refresh()

    for row in panel.rows():
        line = ttk.Frame(box)
        line.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(line, text=row["name"]).pack(side="left")
        sv = tk.StringVar(value=row["status"])
        rm = ttk.Button(line, text="Remove", width=8, command=lambda k=row["key"]: remove(k))
        rm.pack(side="right", padx=2)
        dl = ttk.Button(line, text="Download", width=9, command=lambda k=row["key"]: download(k))
        dl.pack(side="right", padx=2)
        bar = ttk.Progressbar(line, length=110, maximum=100.0)
        bar.pack(side="right", padx=4)
        ttk.Label(line, textvariable=sv, width=13).pack(side="right")
        ttk.Label(box, text=f"{row['size']} - {row['license']}. {row['cost']}", foreground="#555", wraplength=640).pack(
            anchor="w", padx=16
        )
        widgets[row["key"]] = {"status": sv, "bar": bar, "dl": dl, "rm": rm}

    gbox = ttk.LabelFrame(parent, text="Eye tracking and screen map")
    gbox.pack(fill="x", pady=4)
    gaze_var = tk.StringVar()
    ttk.Label(gbox, textvariable=gaze_var, wraplength=620).pack(anchor="w", padx=4)
    brow = ttk.Frame(gbox)
    brow.pack(anchor="w", pady=2)

    def spawn(action: str) -> None:
        argv = panel.gaze_command(action)
        win._report(None, f"running: away gaze {action} (Esc closes it)")

        def run() -> None:
            try:
                r = subprocess.run(argv, capture_output=True, text=True, timeout=900)
                err = None if r.returncode == 0 else (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
            except Exception as exc:
                err = str(exc)

            def done() -> None:
                win._report(err, f"gaze {action}: done")
                refresh()

            win._ui.post(done)

        threading.Thread(target=run, daemon=True, name=f"gaze-{action}").start()

    ttk.Button(brow, text="Calibrate gaze...", command=lambda: spawn("calibrate")).pack(side="left", padx=2)
    ttk.Button(brow, text="Test gaze", command=lambda: spawn("test")).pack(side="left", padx=2)

    def clear() -> None:
        win._report(None, panel.clear_gaze())
        refresh()

    ttk.Button(brow, text="Clear", command=clear).pack(side="left", padx=2)
    refresh()
    return panel
