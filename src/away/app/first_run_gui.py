"""The guided first run window (tkinter): big friendly buttons, progress dots, every step skippable.

All decisions live in :mod:`away.app.first_run` (tested headless); this is the view. :func:`dot_style` and
:func:`format_left` are pure and tested; the window itself is smoke-tested under Xvfb.
"""

from __future__ import annotations

import contextlib
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from away import config as cfgmod
from away.app import first_run as fr
from away.comfort import actions, ui
from away.comfort.settings import PLAIN_SWITCHES

GREEN, GREEN_DARK = "#2E7D32", "#1B5E20"
BLUE = "#1565C0"
GREY, LIGHT = "#9E9E9E", "#E0E0E0"
ORANGE = "#E65100"
BG = "#FFFFFF"
TEXT, SOFT = "#212121", "#616161"
LIVE_STEPS = ("framing", "tutorial")


def dot_style(status: str) -> tuple[str, str]:
    """(fill, outline) of a progress dot: done green, skipped hollow grey, failed orange, current blue."""
    return {
        fr.DONE: (GREEN, GREEN),
        fr.SKIPPED: (BG, GREY),
        fr.FAILED: (ORANGE, ORANGE),
        fr.CURRENT: (BLUE, BLUE),
    }.get(status, (LIGHT, LIGHT))


def format_left(seconds: float) -> str:
    s = max(0, round(seconds))
    return f"{s // 60}:{s % 60:02d} left"


class FirstRunWindow:  # pragma: no cover - widgets; smoke-tested under Xvfb
    def __init__(self, root: Any, cfg: Any, config_path: Path | None = None, state_path: Path | None = None) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.cfg = cfg
        self.config_path = config_path
        fr.discover_plugins()
        self.flow = fr.FirstRun(state_path=state_path)
        self.flow.save()  # "seen": the tray never opens the guide by itself again
        self.q: queue.Queue[Any] = queue.Queue()
        self.cam_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.cam_stop = threading.Event()
        self.cam_thread: threading.Thread | None = None
        self.coach = fr.TutorialCoach()
        self.framing = fr.FramingCheck()
        self._photo: Any = None
        self._agent_wizard: Any = None  # kept alive while the Agent Link wizard window is open
        self._step_ok = False
        root.title("Project Away - getting started")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(900, sw - 40), min(680, sh - 80)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        root.minsize(640, 520)
        self.f_title = ("TkDefaultFont", 22, "bold")
        self.f_body = ("TkDefaultFont", 13)
        self.f_big = ("TkDefaultFont", 14, "bold")
        self.f_small = ("TkDefaultFont", 11)

        top = tk.Frame(root, bg=BG, padx=24, pady=14)
        top.pack(fill="x")
        self.dots = tk.Canvas(top, height=26, bg=BG, highlightthickness=0)
        self.dots.pack(side="left", fill="x", expand=True)
        self.counter = tk.Label(top, text="", font=self.f_small, fg=SOFT, bg=BG)
        self.counter.pack(side="right")

        bottom = tk.Frame(root, bg=BG, padx=24, pady=16)
        bottom.pack(side="bottom", fill="x")
        self.back_btn = self._button(bottom, "Back", self._back, primary=False)
        self.back_btn.pack(side="left")
        self.main_btn = self._button(bottom, "Next", self._main)
        self.main_btn.pack(side="right")
        self.skip_btn = self._button(bottom, "Skip this step", self._skip, primary=False)
        self.skip_btn.pack(side="right", padx=12)

        self.body = tk.Frame(root, bg=BG, padx=32, pady=4)
        self.body.pack(fill="both", expand=True)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.bind("<Configure>", lambda _e: self._draw_dots())
        self._show()
        root.after(50, self._poll)

    # ------------------------------------------------------------------ widgets
    def _button(self, parent: Any, text: str, cmd: Any, primary: bool = True) -> Any:
        tk = self.tk
        return tk.Button(
            parent,
            text=text,
            command=cmd,
            font=self.f_big if primary else self.f_body,
            bg=GREEN if primary else BG,
            fg="white" if primary else TEXT,
            activebackground=GREEN_DARK if primary else LIGHT,
            activeforeground="white" if primary else TEXT,
            relief="flat" if primary else "groove",
            padx=26 if primary else 16,
            pady=12 if primary else 9,
            cursor="hand2",
            borderwidth=1,
        )

    def _label(self, parent: Any, text: str, font: Any = None, fg: str = TEXT, **kw: Any) -> Any:
        lbl = self.tk.Label(parent, text=text, font=font or self.f_body, fg=fg, bg=BG, justify="left", anchor="w", **kw)
        lbl.bind("<Configure>", lambda e: lbl.configure(wraplength=max(200, e.width - 8)))
        return lbl

    def _draw_dots(self) -> None:
        c = self.dots
        c.delete("all")
        r, gap = 8, 30
        for i, (_key, status) in enumerate(self.flow.dots()):
            fill, outline = dot_style(status)
            x = 10 + i * gap
            c.create_oval(x - r, 13 - r, x + r, 13 + r, fill=fill, outline=outline, width=2)
        n, total = self.flow.position()
        self.counter.configure(text=f"Step {n} of {total}")

    def _clear(self) -> None:
        for w in self.body.winfo_children():
            w.destroy()

    # ------------------------------------------------------------------ navigation
    def _show(self) -> None:
        step = self.flow.step
        self._clear()
        self._step_ok = False
        self._draw_dots()
        self._label(self.body, step.title, self.f_title).pack(fill="x", pady=(6, 8))
        self._label(self.body, step.body, self.f_body, fg=SOFT).pack(fill="x", pady=(0, 12))
        self.content = self.tk.Frame(self.body, bg=BG)
        self.content.pack(fill="both", expand=True)
        self.main_btn.configure(text=step.button)
        self.back_btn.configure(state="normal" if self.flow.index > 0 else "disabled")
        self.skip_btn.configure(state="disabled" if step.key in ("welcome", "done") else "normal")
        self.skip_btn.configure(text="Later" if step.key == "agent" else "Skip this step")
        if step.key in LIVE_STEPS:
            self._start_camera()
        else:
            self._stop_camera()
        getattr(self, f"_step_{step.key}", self._step_plugin)()

    def _main(self) -> None:
        key = self.flow.step.key
        handler = getattr(self, f"_main_{key}", None)
        if handler is not None and handler():
            return  # the step ran its own action; it moves on when that is done
        if key == "done":
            self.flow.finish()
            self._close()
            return
        self.flow.next(fr.step_result(key, self._step_ok, self.coach.score()[0]))
        self._show()

    def _skip(self) -> None:
        self.flow.skip()
        self._show()

    def _back(self) -> None:
        self.flow.back()
        self._show()

    def _close(self) -> None:
        self._stop_camera()
        self.flow.save()
        with contextlib.suppress(Exception):
            self.root.destroy()

    # ------------------------------------------------------------------ friendly errors
    def _error(self, raw: Any, retry: Any) -> None:
        err = ui.friendly_error(raw)
        box = self.tk.Frame(self.content, bg="#FFF3E0", padx=16, pady=14)
        box.pack(fill="x", pady=8)
        self.tk.Label(box, text=err.title, font=self.f_big, fg=ORANGE, bg="#FFF3E0", anchor="w").pack(fill="x")
        self.tk.Label(
            box, text=err.message, font=self.f_body, fg=TEXT, bg="#FFF3E0", justify="left", anchor="w", wraplength=700
        ).pack(fill="x", pady=(4, 10))
        row = self.tk.Frame(box, bg="#FFF3E0")
        row.pack(fill="x")
        self._button(row, err.fix_label, lambda: self._fix(err.fix, retry)).pack(side="left")
        if err.fix != "retry":
            self._button(row, "Try again", retry, primary=False).pack(side="left", padx=10)
        self.tk.Label(box, text=f"Details: {raw}", font=self.f_small, fg=SOFT, bg="#FFF3E0", anchor="w").pack(
            fill="x", pady=(10, 0)
        )
        self.flow.mark(fr.FAILED)
        self._draw_dots()

    def _fix(self, fix: str, retry: Any) -> None:
        if fix == "camera_privacy":
            cmd = ui.camera_privacy_command()
            if cmd:
                with contextlib.suppress(Exception):
                    subprocess.Popen(cmd)
            return
        if fix == "install_help":
            import webbrowser

            webbrowser.open(ui.INSTALL_HELP_URL)
            return
        if fix == "download_models":
            self._busy("Downloading (about 10 MB)...")

            def work() -> None:
                err = fr.download_models(self.cfg, out=lambda _m: None)
                self.q.put(("after_download", err, retry))

            threading.Thread(target=work, daemon=True, name="first-run-download").start()
            return
        retry()

    def _busy(self, text: str) -> None:
        for w in self.content.winfo_children():
            w.destroy()
        self._label(self.content, text, self.f_big, fg=BLUE).pack(fill="x", pady=20)

    # ------------------------------------------------------------------ steps
    def _step_welcome(self) -> None:
        self._label(
            self.content,
            "You'll need a webcam. Sit at your desk as you normally would.",
            self.f_body,
        ).pack(fill="x")

    def _step_camera(self) -> None:
        self._label(self.content, "Press the green button when you're ready.", self.f_body, fg=SOFT).pack(fill="x")

    def _main_camera(self) -> bool:
        if self._step_ok:
            return False
        self._busy("Turning your camera on...")

        def work() -> None:
            ok, msg = fr.camera_check(self.cfg)
            self.q.put(("camera", ok, msg))

        threading.Thread(target=work, daemon=True, name="first-run-camera").start()
        return True

    def _camera_result(self, ok: bool, msg: str) -> None:
        if self.flow.step.key != "camera":
            return
        for w in self.content.winfo_children():
            w.destroy()
        if ok:
            self._step_ok = True
            self.flow.mark(fr.DONE)
            self._draw_dots()
            self._label(self.content, "Your camera works!", self.f_big, fg=GREEN).pack(fill="x", pady=10)
            self.main_btn.configure(text="Next")
        else:
            self._error(msg, self._main_camera)
            self._button(self.content, "Run a full camera check", self._diagnose, primary=False).pack(
                anchor="w", pady=6
            )

    def _diagnose(self) -> None:
        """The same report as `away cameras diagnose` (Settings > Cameras > Diagnose)."""
        from away.app.settings_gui import SettingsModel, camera_diagnose_worker

        text = self.tk.Text(self.content, height=10, font=("TkFixedFont", 9), wrap="word")
        text.pack(fill="both", expand=True, pady=6)
        out: queue.Queue[Any] = queue.Queue()
        model = SettingsModel(self.cfg, self.config_path)
        threading.Thread(target=camera_diagnose_worker, args=(model, out), daemon=True).start()

        def pump() -> None:
            with contextlib.suppress(Exception):
                while not out.empty():
                    item = out.get_nowait()
                    line = item if isinstance(item, str) else str(item.get("error") or "finished")
                    text.insert("end", line + "\n")
                    text.see("end")
                    if isinstance(item, dict):
                        return
                self.root.after(200, pump)

        pump()

    def _step_framing(self) -> None:
        row = self.tk.Frame(self.content, bg=BG)
        row.pack(fill="both", expand=True)
        self.preview = self.tk.Canvas(row, width=320, height=240, bg="#263238", highlightthickness=0)
        self.preview.pack(side="left", anchor="n")
        self.preview.create_text(160, 120, text="starting the camera...", fill="white", tags="msg")
        side = self.tk.Frame(row, bg=BG, padx=20)
        side.pack(side="left", fill="both", expand=True)
        self.tips = self._label(side, "", self.f_big, fg=BLUE)
        self.tips.pack(fill="x")

    def _step_tutorial(self) -> None:
        self.coach = fr.TutorialCoach()
        row = self.tk.Frame(self.content, bg=BG)
        row.pack(fill="both", expand=True)
        self.practice = self.tk.Canvas(row, width=420, height=300, bg="#F5F5F5", highlightthickness=1)
        self.practice.pack(side="left", anchor="n")
        side = self.tk.Frame(row, bg=BG, padx=18)
        side.pack(side="left", fill="both", expand=True)
        self.task_labels: dict[str, Any] = {}
        for task in self.coach.tasks:
            lbl = self.tk.Label(side, text="", font=self.f_body, bg=BG, anchor="w")
            lbl.pack(fill="x")
            self.task_labels[task.key] = lbl
        self.task_how = self._label(side, "", self.f_big, fg=BLUE)
        self.task_how.pack(fill="x", pady=(12, 4))
        self.task_hint = self._label(side, "", self.f_small, fg=SOFT)
        self.task_hint.pack(fill="x")
        self.flash = self._label(side, "", self.f_big, fg=GREEN)
        self.flash.pack(fill="x", pady=6)
        bar = self.tk.Frame(side, bg=BG)
        bar.pack(fill="x", pady=6)
        self._button(bar, "Skip this one", self._skip_task, primary=False).pack(side="left")
        self.timer = self._label(bar, "", self.f_small, fg=SOFT)
        self.timer.pack(side="left", padx=12)
        self._draw_tutorial(time.monotonic())

    def _skip_task(self) -> None:
        self.coach.skip(time.monotonic())
        self._draw_tutorial(time.monotonic())

    def _draw_tutorial(self, t: float) -> None:
        c, co = self.practice, self.coach
        c.delete("all")
        w, h = int(c.cget("width")), int(c.cget("height"))
        cur = co.current
        if cur is not None and cur.key == "move":
            for (tx, ty), hit in zip(fr.MOVE_TARGETS, co.targets_hit):
                r = fr.TARGET_RADIUS * w
                c.create_oval(
                    tx * w - r,
                    ty * h - r,
                    tx * w + r,
                    ty * h + r,
                    outline=GREEN,
                    width=3,
                    fill="#C8E6C9" if hit else "",
                )
        elif cur is not None and cur.key == "click":
            c.create_rectangle(w * 0.3, h * 0.35, w * 0.7, h * 0.65, fill=GREEN, outline="")
            c.create_text(w / 2, h / 2, text="Pinch!", fill="white", font=self.f_big)
        elif cur is not None and cur.key == "scroll":
            c.create_rectangle(w - 30, 20, w - 14, h - 20, outline=GREY)
            fill = (h - 40) * next(x for x in co.tasks if x.key == "scroll").progress
            c.create_rectangle(w - 30, h - 20 - fill, w - 14, h - 20, fill=BLUE, outline="")
        elif cur is not None and cur.key == "pause":
            c.create_text(w / 2, h / 2, text="Open palm \u270b", font=self.f_title, fill=SOFT)
        else:
            c.create_text(w / 2, h / 2, text="All done!", font=self.f_title, fill=GREEN)
        if co.dot is not None:
            x, y = co.dot[0] * w, co.dot[1] * h
            c.create_oval(x - 9, y - 9, x + 9, y + 9, fill=BLUE, outline="white", width=2)
        for task in co.tasks:
            mark = "\u2714" if task.done else ("\u2013" if task.skipped else "\u25cb")
            colour = GREEN if task.done else (GREY if task.skipped else (BLUE if task is cur else TEXT))
            self.task_labels[task.key].configure(text=f"{mark}  {task.title}", fg=colour)
        self.task_how.configure(text=cur.how if cur else "Great job! Press Next to continue.")
        self.task_hint.configure(text=("Tip: " + cur.hint) if cur and co.show_hint(t) else "")
        self.flash.configure(text=co.flash)
        left = co.remaining(t)
        self.timer.configure(text=format_left(left) if left > 0 else "Take your time")
        if co.complete:
            self.flow.mark(fr.DONE)
            self._draw_dots()

    def _step_comfort(self) -> None:
        self.switch_vars: dict[str, Any] = {}
        for path, label, help_text in PLAIN_SWITCHES:
            var = self.tk.BooleanVar(value=actions.switch_value(self.cfg, path))
            row = self.tk.Frame(self.content, bg=BG, pady=6)
            row.pack(fill="x")
            self.tk.Checkbutton(
                row,
                text=label,
                variable=var,
                font=self.f_big,
                bg=BG,
                activebackground=BG,
                anchor="w",
                command=self._switch_command(path, var),
            ).pack(fill="x")
            self._label(row, help_text, self.f_small, fg=SOFT).pack(fill="x", padx=28)
            self.switch_vars[path] = var
        self.switch_msg = self._label(self.content, "", self.f_small, fg=ORANGE)
        self.switch_msg.pack(fill="x", pady=6)

    def _switch_command(self, path: str, var: Any) -> Any:
        return lambda: self._set_switch(path, var)

    def _set_switch(self, path: str, var: Any) -> None:
        err = actions.set_switch_in_file(path, bool(var.get()), self.config_path, live_cfg=self.cfg)
        self.switch_msg.configure(text=err or "")
        if err:
            var.set(not var.get())

    def _main_gaze(self) -> bool:
        if self._step_ok:
            return False
        self._busy("Follow the dots with your eyes. Press Esc to stop.")

        def work() -> None:
            from away.app.frozen import away_command

            try:
                rc = subprocess.run(away_command(["gaze", "calibrate"]), capture_output=True, text=True)
                tail = (rc.stderr.strip().splitlines() or [""])[-1]
                self.q.put(("gaze", rc.returncode == 0, tail))
            except Exception as exc:
                self.q.put(("gaze", False, str(exc)))

        threading.Thread(target=work, daemon=True, name="first-run-gaze").start()
        return True

    def _step_gaze(self) -> None:
        self._label(self.content, "You can skip this: eye tracking is extra.", self.f_body, fg=SOFT).pack(fill="x")

    def _gaze_result(self, ok: bool, msg: str) -> None:
        if self.flow.step.key != "gaze":
            return
        for w in self.content.winfo_children():
            w.destroy()
        if ok:
            self._step_ok = True
            self.flow.mark(fr.DONE)
            self._draw_dots()
            self._label(self.content, "Eye setup done!", self.f_big, fg=GREEN).pack(fill="x", pady=10)
            self.main_btn.configure(text="Next")
        else:
            self._error(msg or "eye setup did not finish", self._main_gaze)

    def _step_agent(self) -> None:
        self._label(
            self.content,
            "You can connect, change or remove agents any time from the tray menu.",
            self.f_small,
            fg=SOFT,
        ).pack(fill="x", pady=(12, 0))

    def _main_agent(self) -> bool:
        """The page's one button: open the existing 3-step Agent Link wizard; the guide stays behind it."""
        from away.agentlink import gui as agent_gui

        self._agent_wizard = agent_gui.Wizard(_FirstRunAgentLinkHost(self.root))
        return True

    def _step_done(self) -> None:
        self._label(self.content, self.flow.summary() + ".", self.f_big, fg=GREEN).pack(fill="x", pady=8)

    def _step_plugin(self) -> None:
        step = self.flow.step
        if step.run is not None:
            self._button(self.content, step.button, lambda: self._run_plugin(step), primary=False).pack(anchor="w")
            self.main_btn.configure(text="Next")

    def _run_plugin(self, step: fr.Step) -> None:
        try:
            if step.run is not None:
                step.run()
            self._step_ok = True
            self.flow.mark(fr.DONE)
            self._draw_dots()
        except Exception as exc:
            self._error(exc, lambda: self._run_plugin(step))

    # ------------------------------------------------------------------ camera
    def _start_camera(self) -> None:
        if self.cam_thread is not None and self.cam_thread.is_alive():
            return
        ok, msg = fr.tracking_ready(self.cfg)
        if not ok:
            self.root.after(10, lambda: self._error(msg, self._show))
            return
        self.cam_stop.clear()
        self.cam_q = queue.Queue(maxsize=8)
        self.cam_thread = threading.Thread(
            target=fr.camera_session,
            args=(self.cfg, _Dropping(self.cam_q), self.cam_stop),
            daemon=True,
            name="first-run-live",
        )
        self.cam_thread.start()

    def _stop_camera(self) -> None:
        self.cam_stop.set()
        th = self.cam_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.5)
        self.cam_thread = None

    def _poll(self) -> None:
        try:
            self._drain()
        except Exception as exc:  # a bug in a view update must not kill the window
            with contextlib.suppress(Exception):
                self.flow.mark(fr.FAILED)
                print(f"first run: {exc}", file=sys.stderr)
        with contextlib.suppress(Exception):
            self.root.after(50, self._poll)

    def _drain(self) -> None:
        while not self.q.empty():
            item = self.q.get_nowait()
            if item[0] == "camera":
                self._camera_result(item[1], item[2])
            elif item[0] == "gaze":
                self._gaze_result(item[1], item[2])
            elif item[0] == "after_download":
                if item[1]:
                    self._show()
                    self._error(item[1], item[2])
                else:
                    item[2]()
        key = self.flow.step.key
        last: dict[str, Any] | None = None
        image = None
        while not self.cam_q.empty():
            msg = self.cam_q.get_nowait()
            if "error" in msg:
                if key in LIVE_STEPS:
                    self._stop_camera()
                    self._error(msg["error"], self._show)
                return
            if key == "tutorial":
                for kind, data in msg.get("events", []):
                    self.coach.feed(kind, data, float(msg["t"]))
            if key == "framing":
                self.framing.face_tracking = bool(msg.get("face_tracking"))
                res = self.framing.update(
                    float(msg["t"]),
                    msg.get("face_box"),
                    msg.get("hand"),
                    msg.get("brightness"),
                    bool(msg.get("face_checked")),
                )
                self._framing_view(res)
            image = msg.get("image", image)
            last = msg
        if key == "tutorial":
            self._draw_tutorial(time.monotonic())
        if key == "framing" and image is not None:
            self._preview(image, last)

    def _framing_view(self, res: fr.FramingResult) -> None:
        if res.stable:
            self._step_ok = True
            self.flow.mark(fr.DONE)
            self.tips.configure(text="\u2714 Looks great! Press the green button.", fg=GREEN)
            self._draw_dots()
        elif res.tips:
            self.tips.configure(text="\n\n".join(res.tips[:2]), fg=BLUE)
        else:
            self.tips.configure(text="Almost there... hold still.", fg=BLUE)

    def _preview(self, image: Any, msg: dict[str, Any] | None) -> None:
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return
        img = Image.fromarray(image).resize((320, 240))
        self._photo = ImageTk.PhotoImage(img)
        c = self.preview
        c.delete("all")
        c.create_image(0, 0, image=self._photo, anchor="nw")
        hand = (msg or {}).get("hand")
        if hand:
            for x, y in hand:
                c.create_oval(x * 320 - 3, y * 240 - 3, x * 320 + 3, y * 240 + 3, fill="#76FF03", outline="")


class _FirstRunAgentLinkHost:
    """Enough of the Agent Link tab's shape to host its 3-step connect wizard in the first-run window
    (see :class:`away.agentlink.gui.WizardHost`)."""

    tk: Any
    ttk: Any
    root: Any
    frame: Any
    m: Any

    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        from away.agentlink import gui_model

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.frame = root  # the wizard window opens on top of the guide
        self.m = gui_model.AgentLinkModel()

    def _bg(self, fn: Any, done: Any = None) -> None:
        def run() -> None:
            try:
                res = fn()
            except Exception as exc:
                res = exc
            if done is not None:
                self.root.after(0, lambda: done(res))

        threading.Thread(target=run, daemon=True, name="first-run-agent-wizard").start()

    def _fix(self, p: Any) -> None:
        if p.fix.startswith("url:"):
            import webbrowser

            webbrowser.open(p.fix[4:])
        elif p.fix == "start_app":
            import subprocess

            from away.app.frozen import away_command

            with contextlib.suppress(Exception):
                subprocess.Popen(away_command(["tray"]))


class _Dropping:
    """A queue front that drops the oldest frame when the window falls behind (the camera never blocks)."""

    def __init__(self, q: queue.Queue[dict[str, Any]]) -> None:
        self.q = q

    def put(self, item: dict[str, Any]) -> None:
        while True:
            try:
                self.q.put_nowait(item)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self.q.get_nowait()


def main(config_path: Path | None = None) -> int:
    from away.app import crashlog

    crashlog.install("first-run")
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
        print(
            f"The guided first run needs a desktop session with Tk ({type(exc).__name__}: {exc}).\n"
            "Run `away tour --text` for the text version.",
            file=sys.stderr,
        )
        return 1
    FirstRunWindow(root, cfg, config_path)
    root.mainloop()
    return 0
