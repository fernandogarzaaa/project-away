"""The "Calibrate tracking" window (v0.1.25): ``away calibrate``, tray > Calibrate tracking..., Settings > Getting
started. The steps and the maths are in :mod:`away.hands.livecal`; this is only the window.

It shows the camera with the hand points, a shake meter (pixels of noise while you hold still), the frame rate and
the delay, walks through the three steps, then draws the cursor path before (grey) and after (green) so you can see
the difference. Save keeps it for this camera; Reset to defaults forgets it.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from away import config as cfgmod
from away.hands import livecal

GREEN, GREY, BLUE, TEXT = "#2E7D32", "#9E9E9E", "#1565C0", "#212121"
PREVIEW_W, PREVIEW_H = 320, 240
TRAIL_W, TRAIL_H = 320, 180


def status_line(noise_px: float, fps: float, latency_ms: float | None) -> str:
    """ "Shake 1.2 px  |  30 fps  |  delay 45 ms" (plain words for the meter row)."""
    parts = [f"Shake {noise_px:.1f} px", f"{fps:.0f} fps" if fps > 0 else "-- fps"]
    if latency_ms is not None:
        parts.append(f"delay {latency_ms:.0f} ms")
    return "  |  ".join(parts)


def result_line(before_px: float, after_px: float) -> str:
    if before_px <= 0:
        return f"Cursor shake while still: {after_px:.1f} px"
    if after_px < before_px:
        pct = 100.0 * (1.0 - after_px / before_px)
        return f"Cursor shake while still: {before_px:.1f} px before, {after_px:.1f} px now ({pct:.0f}% steadier)"
    return f"Cursor shake while still: {before_px:.1f} px before, {after_px:.1f} px now"


def trail_points(
    pts: list[tuple[int, int]], screen: tuple[int, int], w: int = TRAIL_W, h: int = TRAIL_H
) -> list[float]:
    """Screen positions scaled into the trail box, flattened for ``Canvas.create_line``."""
    out: list[float] = []
    for x, y in pts:
        out += [x * w / max(1, screen[0]), y * h / max(1, screen[1])]
    return out


class CalibrateWindow:  # pragma: no cover - widgets; smoke-tested under Xvfb
    def __init__(self, root: Any, cfg: Any, config_path: Path | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root, self.cfg, self.path = root, cfg, config_path
        self.key = livecal.camera_key(cfg)
        self.session = livecal.Session(pinch_on=float(cfg.hands.gestures.pinch_on))
        self.entry: dict[str, Any] | None = None
        self.cam_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        self.cam_stop = threading.Event()
        self.cam_thread: threading.Thread | None = None
        self.latency_ms: float | None = None
        self._photo: Any = None
        root.title("Calibrate tracking - Project Away")
        root.configure(padx=16, pady=12)
        big = ("TkDefaultFont", 13, "bold")
        tk.Label(root, text=f"Calibrate tracking for camera: {self.key}", font=big, anchor="w").pack(fill="x")
        self.instr = tk.Label(root, text="", wraplength=640, justify="left", fg=BLUE, font=("TkDefaultFont", 12))
        self.instr.pack(fill="x", pady=(6, 4))
        self.bar = ttk.Progressbar(root, maximum=1.0, length=640)
        self.bar.pack(fill="x")
        row = tk.Frame(root)
        row.pack(fill="x", pady=8)
        self.preview = tk.Canvas(row, width=PREVIEW_W, height=PREVIEW_H, bg="black", highlightthickness=0)
        self.preview.pack(side="left")
        right = tk.Frame(row)
        right.pack(side="left", padx=(12, 0), fill="both", expand=True)
        self.meter = tk.Label(right, text=status_line(0.0, 0.0, None), anchor="w", font=("TkFixedFont", 11))
        self.meter.pack(fill="x")
        self.msg = tk.Label(right, text="", anchor="w", fg="#C62828", wraplength=300, justify="left")
        self.msg.pack(fill="x", pady=(4, 8))
        tk.Label(right, text="Cursor path: grey = before, green = after", anchor="w", fg=GREY).pack(fill="x")
        self.trail = tk.Canvas(right, width=TRAIL_W, height=TRAIL_H, bg="white", highlightthickness=1)
        self.trail.pack()
        self.result = tk.Label(right, text="", anchor="w", wraplength=320, justify="left")
        self.result.pack(fill="x", pady=(4, 0))
        btns = tk.Frame(root)
        btns.pack(fill="x", pady=(8, 0))
        self.save_btn = tk.Button(
            btns,
            text="Save",
            bg=GREEN,
            fg="white",
            relief="flat",
            padx=18,
            pady=6,
            state="disabled",
            command=self._save,
        )
        self.save_btn.pack(side="left")
        tk.Button(btns, text="Start again", relief="flat", padx=12, pady=6, command=self._restart).pack(
            side="left", padx=8
        )
        tk.Button(btns, text="Reset to defaults", relief="flat", padx=12, pady=6, command=self._reset).pack(side="left")
        tk.Button(btns, text="Close", relief="flat", padx=12, pady=6, command=self._close).pack(side="right")
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._start_camera()
        self._poll()

    # ------------------------------------------------------------------ camera
    def _start_camera(self) -> None:
        from away.app import first_run as fr
        from away.app.first_run_gui import _Dropping
        from away.comfort.ui import friendly_error

        ok, msg = fr.tracking_ready(self.cfg)
        if not ok:
            self.msg.configure(text=friendly_error(msg).message)
            return
        self.cam_stop.clear()
        self.cam_thread = threading.Thread(
            target=fr.camera_session,
            args=(self.cfg, _Dropping(self.cam_q), self.cam_stop),
            daemon=True,
            name="calibrate-live",
        )
        self.cam_thread.start()

    def _stop_camera(self) -> None:
        self.cam_stop.set()
        th = self.cam_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.5)
        self.cam_thread = None

    def _poll(self) -> None:
        with contextlib.suppress(Exception):
            self._drain()
        with contextlib.suppress(Exception):
            self.root.after(40, self._poll)

    def _drain(self) -> None:
        from away.comfort.ui import friendly_error
        from away.hands.landmarks import as_array, pinch_ratio

        last: dict[str, Any] | None = None
        image = None
        while not self.cam_q.empty():
            msg = self.cam_q.get_nowait()
            if "error" in msg:
                self.msg.configure(text=friendly_error(msg["error"]).message)
                return
            t = float(msg["t"])
            self.latency_ms = max(0.0, (time.monotonic() - t) * 1000.0)
            hand = msg.get("hand")
            ratio = None
            if hand:
                with contextlib.suppress(Exception):
                    ratio = pinch_ratio(as_array([[p[0], p[1], 0.0] for p in hand]))
            was = self.session.step
            self.session.feed(t, hand, ratio)
            if was != "done" and self.session.step == "done":
                self._finish()
            image = msg.get("image", image)
            last = msg
        s = self.session
        self.instr.configure(text=s.instruction)
        self.bar.configure(value=s.progress())
        self.msg.configure(text=s.message)
        noise = s.live_noise_px() if s.step == "still" else livecal.measure_noise(s.still) * livecal.FRAME_PX
        self.meter.configure(text=status_line(noise, s.fps(), self.latency_ms))
        if image is not None:
            self._preview(image, last)

    def _preview(self, image: Any, msg: dict[str, Any] | None) -> None:
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return
        self._photo = ImageTk.PhotoImage(Image.fromarray(image).resize((PREVIEW_W, PREVIEW_H)))
        c = self.preview
        c.delete("all")
        c.create_image(0, 0, image=self._photo, anchor="nw")
        for i, (x, y) in enumerate((msg or {}).get("hand") or []):
            r = 5 if i == livecal.INDEX_TIP else 3
            col = "#FFEB3B" if i == livecal.INDEX_TIP else "#76FF03"
            px, py = x * PREVIEW_W, y * PREVIEW_H
            c.create_oval(px - r, py - r, px + r, py + r, fill=col, outline="")

    # ------------------------------------------------------------------ result
    def _screen(self) -> tuple[int, int]:
        return (max(800, self.root.winfo_screenwidth()), max(600, self.root.winfo_screenheight()))

    def _finish(self) -> None:
        self.entry = livecal.entry_for(self.session.measurements())
        screen = self._screen()
        before, after, jb, ja = livecal.before_after(self.cfg, self.entry, self.session.trail, screen)
        c = self.trail
        c.delete("all")
        if len(before) > 1:
            c.create_line(*trail_points(before, screen), fill=GREY, width=2)
        if len(after) > 1:
            c.create_line(*trail_points(after, screen), fill=GREEN, width=2)
        self.result.configure(text=result_line(jb, ja))
        self.save_btn.configure(state="normal")
        self._stop_camera()

    def _save(self) -> None:
        if not self.entry:
            return
        try:
            livecal.save(self.path, self.key, self.entry)
        except Exception as exc:
            self.msg.configure(text=f"Could not save: {exc}")
            return
        self.result.configure(text="Saved \u2714  Project Away uses it right away for this camera.", fg=GREEN)
        self.save_btn.configure(state="disabled")

    def _reset(self) -> None:
        try:
            livecal.save(self.path, self.key, None)
        except Exception as exc:
            self.msg.configure(text=f"Could not reset: {exc}")
            return
        self.result.configure(text="Back to the default settings for this camera.", fg=TEXT)

    def _restart(self) -> None:
        self.session = livecal.Session(pinch_on=float(self.cfg.hands.gestures.pinch_on))
        self.entry = None
        self.save_btn.configure(state="disabled")
        self.trail.delete("all")
        self.result.configure(text="", fg=TEXT)
        if self.cam_thread is None:
            self._start_camera()

    def _close(self) -> None:
        self._stop_camera()
        with contextlib.suppress(Exception):
            self.root.destroy()


def main(config_path: Path | None = None) -> int:  # pragma: no cover - needs a display
    from away.app import crashlog

    crashlog.install("calibrate")
    try:
        cfg, _ = cfgmod.load(config_path)
    except ValueError as exc:
        print(f"error: your config is invalid, fix it or run `away config init`:\n{exc}", file=sys.stderr)
        return 2
    try:
        import tkinter as tk

        from away.control.monitors import enable_dpi_awareness

        enable_dpi_awareness()
        root = tk.Tk()
    except Exception as exc:
        print(f"Calibrate tracking needs a desktop session with Tk ({type(exc).__name__}: {exc}).", file=sys.stderr)
        return 1
    CalibrateWindow(root, cfg, config_path)
    root.mainloop()
    return 0
