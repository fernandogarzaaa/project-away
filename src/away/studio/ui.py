"""Settings > Gesture Studio: the Tk widgets (logic lives in controller / live / engine; smoke-tested under Xvfb).

Simple view by default: a big camera button, the live picture with the areas where gestures work, what it sees
right now with a "How sure it is" bar, the list of your gestures and a 3-step "Teach a new gesture" wizard.
"Show details" adds the 5-second timeline, the points it pays attention to, the closest practice example and
what it mixed up when it checked itself. Every control has a tooltip; problems show a plain message with a fix
button. Replaces the old "Test gesture" tab (the built-in gesture line shows what that tab showed).
"""

from __future__ import annotations

import contextlib
import queue
import threading
from pathlib import Path
from typing import Any

from away.studio.controller import StudioController, WizardPlan, action_runner
from away.studio.pack import NOTHING, StudioStore
from away.studio.words import (
    ACTION_CHOICES,
    FIXES,
    HELP,
    KIND_WORDS,
    LABELS,
    TAB_TITLE,
    TOOLTIPS,
    WIZARD,
    action_text,
    explain,
    name_for,
    score_text,
    state_text,
    summary_split,
    sure_text,
)
from away.studio.zones import ACTIVE, IGNORE, ActivationLine, ZoneMap, ZoneRect

PALETTE = ("#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#e53935", "#00897b", "#6d4c41", "#3949ab")
EDGE_GRAB_PX = 6
CANVAS_W, CANVAS_H = 400, 300


class Tooltip:  # pragma: no cover - widgets
    """A small yellow note that appears when the pointer rests on a widget."""

    def __init__(self, widget: Any, text: str, delay_ms: int = 500) -> None:
        self.widget, self.text, self.delay = widget, text, delay_ms
        self._after: Any = None
        self._tip: Any = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e: Any = None) -> None:
        self._hide()
        self._after = self.widget.after(self.delay, self._show)

    def _show(self) -> None:
        import tkinter as tk

        if self._tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip, text=self.text, background="#fff8c4", relief="solid", borderwidth=1, wraplength=320,
            justify="left", padx=6, pady=4,
        ).pack()  # fmt: skip

    def _hide(self, _e: Any = None) -> None:
        if self._after is not None:
            with contextlib.suppress(Exception):
                self.widget.after_cancel(self._after)
            self._after = None
        if self._tip is not None:
            with contextlib.suppress(Exception):
                self._tip.destroy()
            self._tip = None


def tip(widget: Any, key: str) -> Any:  # pragma: no cover - widgets
    Tooltip(widget, TOOLTIPS.get(key, key))
    return widget


class StudioTab:  # pragma: no cover - widgets; smoke-tested under Xvfb only
    def __init__(self, window: Any, nb: Any, store: StudioStore | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.w = window
        self.tk, self.ttk = tk, ttk
        self.root = window.root
        self.nb = nb
        self.ctl = StudioController(window.m, store)
        self.q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.cmd: queue.Queue[Any] = queue.Queue()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.zprofile = "default"
        self.zmap: ZoneMap = self.ctl.zone_map(self.zprofile)
        self.frame: dict[str, Any] | None = None
        self.photo: Any = None
        self.timeline: list[tuple[float, dict[str, float]]] = []
        self.events: list[tuple[float, str]] = []
        self.wizard: Wizard | None = None
        self.relax_pending = False
        self._drag: tuple[str, Any] | None = None
        self._fix_id = ""
        self._runner: Any = None
        style = ttk.Style(self.root)
        with contextlib.suppress(Exception):
            style.configure("Big.TButton", padding=(14, 8), font=("TkDefaultFont", 12, "bold"))
        self.f = f = window._scroll_tab(nb, TAB_TITLE, 6)
        f.columnconfigure(0, weight=1)

        # ---- top bar
        bar = ttk.Frame(f)
        bar.grid(row=0, column=0, sticky="we")
        self.cam_btn = tip(ttk.Button(bar, text=LABELS["camera_on"], style="Big.TButton", command=self.toggle), "camera_on")
        self.cam_btn.pack(side="left")
        self.dry = tk.BooleanVar(value=True)
        tip(ttk.Checkbutton(bar, text=LABELS["practice_mode"], variable=self.dry, command=self._dry_changed), "practice_mode").pack(
            side="left", padx=10
        )
        self.details = tk.BooleanVar(value=False)
        tip(ttk.Checkbutton(bar, text=LABELS["details"], variable=self.details, command=self._details_changed), "details").pack(
            side="left", padx=4
        )
        tip(ttk.Button(bar, text=LABELS["thats_wrong"], command=self._wrong), "thats_wrong").pack(side="right")
        self.watch = tk.StringVar(value=LABELS["my_gestures"])
        cb = ttk.Combobox(bar, textvariable=self.watch, state="readonly", width=16,
                          values=[LABELS["my_gestures"], LABELS["builtin_gestures"]])  # fmt: skip
        cb.bind("<<ComboboxSelected>>", lambda _e: self._source_changed())
        tip(cb, "watching").pack(side="right", padx=4)
        ttk.Label(bar, text=LABELS["watching"] + ":").pack(side="right")

        # ---- friendly problem bar
        self.msg_bar = ttk.Frame(f)
        self.msg_bar.grid(row=1, column=0, sticky="we", pady=(6, 0))
        self.msg = tk.StringVar()
        tk.Label(self.msg_bar, textvariable=self.msg, fg="#8a4b00", anchor="w", justify="left", wraplength=640).pack(
            side="left", fill="x", expand=True
        )
        self.fix_btn = ttk.Button(self.msg_bar, text="", command=self._fix)
        self.status = tk.StringVar(value="Turn the camera on to start.")
        ttk.Label(f, textvariable=self.status, foreground="#555").grid(row=2, column=0, sticky="w")

        # ---- picture + what it sees
        mid = ttk.Frame(f)
        mid.grid(row=3, column=0, sticky="we", pady=6)
        left = ttk.LabelFrame(mid, text=LABELS["where"], padding=6)
        left.pack(side="left", anchor="n")
        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, background="#222", highlightthickness=0)
        self.canvas.pack()
        tip(self.canvas, "grid")
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        tools = ttk.Frame(left)
        tools.pack(fill="x", pady=(4, 0))
        self.tool = tk.StringVar(value="ignore")
        for key, val in (("works_here", "works"), ("ignore_here", "ignore"), ("draw_box", "box"), ("add_line", "line")):
            tip(ttk.Radiobutton(tools, text=LABELS[key], value=val, variable=self.tool), key).pack(side="left", padx=2)
        tools2 = ttk.Frame(left)
        tools2.pack(fill="x", pady=(4, 0))
        ttk.Label(tools2, text=LABELS["grid"] + ":").pack(side="left")
        self.cols = tk.IntVar(value=self.zmap.cols)
        self.rows = tk.IntVar(value=self.zmap.rows)
        for var in (self.cols, self.rows):
            sp = ttk.Spinbox(tools2, from_=1, to=12, width=3, textvariable=var, command=self._resize)
            tip(sp, "grid").pack(side="left", padx=1)
        tip(ttk.Button(tools2, text=LABELS["clear_zones"], command=self._clear_zones), "clear_zones").pack(side="left", padx=6)
        ttk.Label(tools2, text=LABELS["zones_for"] + ":").pack(side="left")
        self.zprof_var = tk.StringVar(value="default")
        self.zprof_cb = ttk.Combobox(tools2, textvariable=self.zprof_var, state="readonly", width=12)
        self.zprof_cb.bind("<<ComboboxSelected>>", lambda _e: self._zone_profile_changed())
        tip(self.zprof_cb, "zones_for").pack(side="left", padx=2)
        self.block_var = tk.BooleanVar(value=bool(self.ctl.cfg.zones.block_cursor))
        self.block_chk = ttk.Checkbutton(left, text="Also stop the cursor there", variable=self.block_var,
                                         command=lambda: self._report(self.ctl.set_block_cursor(self.block_var.get())))  # fmt: skip
        Tooltip(self.block_chk, "When ticked, the cursor also stops moving in 'Ignore here' areas. "
                "Normally only clicks and gestures are ignored there.")  # fmt: skip
        self.zone_note = tk.StringVar()
        ttk.Label(left, textvariable=self.zone_note, foreground="#555").pack(anchor="w")

        right = ttk.LabelFrame(mid, text=LABELS["sees"], padding=6)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0), anchor="n")
        self.top_var = tk.StringVar(value="-")
        tk.Label(right, textvariable=self.top_var, font=("TkDefaultFont", 20, "bold"), fg="#1b5e20").pack(anchor="w")
        self.sure_canvas = tk.Canvas(right, width=260, height=26, highlightthickness=0)
        self.sure_canvas.pack(anchor="w")
        tip(self.sure_canvas, "how_sure")
        self.state_var = tk.StringVar()
        ttk.Label(right, textvariable=self.state_var).pack(anchor="w")
        self.last_var = tk.StringVar(value=f"{LABELS['last_action']}: -")
        ttk.Label(right, textvariable=self.last_var, wraplength=300).pack(anchor="w", pady=(4, 0))
        self.builtin_var = tk.StringVar()
        ttk.Label(right, textvariable=self.builtin_var, foreground="#555", wraplength=300).pack(anchor="w")
        self.bars = tk.Canvas(right, width=300, height=120, highlightthickness=0)
        self.bars.pack(anchor="w", pady=(6, 0))
        tip(self.bars, "how_sure")
        self.near_var = tk.StringVar()
        self.near_lbl = tip(ttk.Label(right, textvariable=self.near_var, foreground="#555"), "looks_like")

        # ---- details (hidden by default)
        self.det = ttk.LabelFrame(f, text=LABELS["details"], padding=6)
        ttk.Label(self.det, text=LABELS["timeline"]).grid(row=0, column=0, sticky="w")
        self.tl_canvas = tip(tk.Canvas(self.det, width=420, height=140, background="white", highlightthickness=0), "timeline")
        self.tl_canvas.grid(row=1, column=0, sticky="w")
        ttk.Label(self.det, text=LABELS["mixups"]).grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.cm_canvas = tip(tk.Canvas(self.det, width=260, height=160, background="white", highlightthickness=0), "mixups")
        self.cm_canvas.grid(row=1, column=1, sticky="w", padx=(12, 0))
        self.split_var = tk.StringVar()
        ttk.Label(self.det, textvariable=self.split_var, foreground="#555", wraplength=680).grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(self.det, text=LABELS["important_points"] + ": shown as circles on the picture",
                  foreground="#555").grid(row=3, column=0, columnspan=2, sticky="w")  # fmt: skip

        # ---- your gestures
        lst = ttk.LabelFrame(f, text=LABELS["your_gestures"], padding=6)
        lst.grid(row=5, column=0, sticky="we", pady=(6, 0))
        lst.columnconfigure(0, weight=1)
        cols = ("gesture", "kind", "examples", "status", "does")
        self.tree = ttk.Treeview(lst, columns=cols, show="headings", height=5, selectmode="browse")
        for c, title, w in zip(
            cols, ("Gesture", "Kind", LABELS["examples"], "Status", "What it does"), (130, 150, 120, 170, 200)
        ):
            self.tree.heading(c, text=title)
            self.tree.column(c, width=w, anchor="w")
        self.tree.grid(row=0, column=0, sticky="we")
        btns = ttk.Frame(lst)
        btns.grid(row=1, column=0, sticky="w", pady=(6, 0))
        tip(ttk.Button(btns, text=LABELS["teach"], style="Big.TButton", command=self.teach), "teach").pack(side="left")
        for key, cmd in (("practise_more", self.practise), ("learn", self.learn),
                         ("gesture_settings", self.edit_gesture), ("remove", self.remove)):  # fmt: skip
            tip(ttk.Button(btns, text=LABELS[key], command=cmd), key).pack(side="left", padx=3)
        btns2 = ttk.Frame(lst)
        btns2.grid(row=2, column=0, sticky="w", pady=(4, 0))
        for key, cmd in (("share", self.share), ("add_file", self.add_file),
                         ("delete_recordings", self.delete_recordings), ("improve_builtin", self.improve_builtin)):  # fmt: skip
            tip(ttk.Button(btns2, text=LABELS[key], command=cmd), key).pack(side="left", padx=3)
        self.result_var = tk.StringVar()
        tk.Label(lst, textvariable=self.result_var, justify="left", anchor="w", wraplength=760).grid(
            row=3, column=0, sticky="we", pady=(6, 0)
        )
        self.refresh()
        self._draw_canvas()

    # ---------------------------------------------------------------- general
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.5)

    def _report(self, err: str | None, ok: str | None = None) -> None:
        if err:
            self.show_problem(err)
        elif ok:
            self.status.set(ok)
            self.clear_problem()

    def show_problem(self, text: str) -> None:
        ex = explain(text)
        msg, fix, label = ex if ex else (text, "", "")
        self.msg.set(msg)
        self._fix_id = fix
        if fix:
            self.fix_btn.configure(text=label or FIXES.get(fix, "Fix it"))
            self.fix_btn.pack(side="right")
        else:
            self.fix_btn.pack_forget()

    def clear_problem(self) -> None:
        self.msg.set("")
        self._fix_id = ""
        self.fix_btn.pack_forget()

    def _fix(self) -> None:
        from tkinter import messagebox

        fix = self._fix_id
        self.clear_problem()
        if fix == "open_cameras":
            for tab in self.nb.tabs():
                if self.nb.tab(tab, "text") == "Cameras":
                    self.nb.select(tab)
        elif fix in ("retry", "camera_on"):
            if self.running():
                self.stop.set()
                self.root.after(800, self.start)
            else:
                self.start()
        elif fix == "record_relax":
            self.record_relax()
        elif fix == "practise_more":
            self.practise()
        elif fix == "teach":
            self.teach()
        elif fix in HELP:
            messagebox.showinfo(TAB_TITLE, HELP[fix], parent=self.root)

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for r in self.ctl.rows():
            ex = f"{r.examples} ({r.rounds} round{'s' if r.rounds != 1 else ''})"
            self.tree.insert("", "end", iid=r.name, values=(r.title, f"{r.kind}, {r.hands.lower()}", ex, r.status, r.action))
        self.zprof_cb["values"] = self.ctl.zone_profiles()
        res = self.ctl.last_result()
        if res:
            self._show_result(res)

    def selected(self) -> str | None:
        sel = self.tree.selection()
        if sel:
            return str(sel[0])
        kids = self.tree.get_children()
        return str(kids[0]) if len(kids) == 1 else None

    # ---------------------------------------------------------------- camera
    def toggle(self) -> None:
        if self.running():
            self.stop.set()
            self.cam_btn.configure(text=LABELS["camera_on"])
        else:
            self.start()

    def start(self) -> bool:
        from away.app.settings_gui import camera_available
        from away.studio.live import studio_camera_worker

        if self.running():
            return True
        ok, msg = camera_available()
        if not ok:
            self.show_problem(msg)
            return False
        if self.w._live_running():
            self.w._live_stop.set()  # the Tuning tab's live view gives the camera back first
            self.root.after(700, self.start)
            return False
        self.stop.clear()
        self.q = queue.Queue()
        self.cmd = queue.Queue()
        self.cmd.put({"cmd": "dry_run", "on": bool(self.dry.get())})
        if self.watch.get() == LABELS["builtin_gestures"]:
            self.cmd.put({"cmd": "source", "which": "builtin"})
        self.thread = threading.Thread(
            target=studio_camera_worker,
            args=(self.w.m, self.q, self.stop, self.cmd),
            kwargs={"store": self.ctl.store, "executor": self._execute, "zone_map": self.zmap.to_dict()},
            daemon=True,
        )
        self.thread.start()
        self.w._test_thread = self.thread  # the Tuning tab's live view waits for this camera user too
        self.cam_btn.configure(text=LABELS["camera_off"])
        self.clear_problem()
        self.root.after(60, self.poll)
        return True

    def _execute(self, name: str) -> None:
        """Practice mode off: really run the gesture's action (on the camera thread; actions are quick)."""
        if self._runner is None:
            try:
                self._runner = action_runner(self.w.m)
            except Exception as exc:
                self.w._ui.post(lambda: self.show_problem(f"Actions cannot run on this computer ({exc})"))
                self._runner = lambda _n: False
        self._runner(name)

    def _dry_changed(self) -> None:
        self.cmd.put({"cmd": "dry_run", "on": bool(self.dry.get())})
        if not self.dry.get():
            self.status.set("Practice mode is off: your gestures will really press keys and open things.")

    def _source_changed(self) -> None:
        which = "builtin" if self.watch.get() == LABELS["builtin_gestures"] else "mine"
        self.cmd.put({"cmd": "source", "which": which})
        self.timeline.clear()

    def _details_changed(self) -> None:
        if self.details.get():
            self.det.grid(row=4, column=0, sticky="we", pady=(6, 0))
            self.near_lbl.pack(anchor="w", pady=(4, 0))
            self.block_chk.pack(anchor="w")
            res = self.ctl.last_result()
            if res:
                self._draw_confusion(res)
        else:
            self.det.grid_forget()
            self.near_lbl.pack_forget()
            self.block_chk.pack_forget()
        self._draw_canvas()

    def _wrong(self) -> None:
        if not self.running():
            self.show_problem("Turn the camera on first, then press this right after something went wrong.")
            self._fix_id = "camera_on"
            self.fix_btn.configure(text=FIXES["camera_on"])
            self.fix_btn.pack(side="right")
            return
        self.cmd.put({"cmd": "wrong"})

    def poll(self) -> None:
        latest = None
        while True:
            try:
                msg = self.q.get_nowait()
            except queue.Empty:
                break
            kind = msg.get("type")
            if kind == "frame":
                latest = msg
            elif kind == "status":
                text = str(msg.get("text", ""))
                if explain(text):
                    self.show_problem(text)
                else:
                    self.status.set(text)
            elif kind == "fired":
                self._fired(msg)
            elif kind in ("countdown", "recording", "take"):
                if self.wizard is not None and self.wizard.alive():
                    self.wizard.on_message(msg)
                elif kind == "take" and msg.get("label") == NOTHING and self.relax_pending:
                    self.relax_pending = False
                    _p, err = self.ctl.save_take(msg)
                    self._report(err, "Relax round saved. Press 'Learn my gestures now'.")
                    self.refresh()
            elif kind == "negative":
                err = self.ctl.save_negative(list(msg.get("frames") or []))
                self._report(err, "Saved as 'no gesture'. Press 'Learn my gestures now' so it learns from it.")
            elif kind == "stopped":
                self.cam_btn.configure(text=LABELS["camera_on"])
                if self.wizard is not None and self.wizard.alive():
                    self.wizard.camera_stopped()
        if latest is not None:
            self._show_frame(latest)
        if self.running() or not self.q.empty():
            self.root.after(50, self.poll)

    def _fired(self, msg: dict[str, Any]) -> None:
        what = action_text(str(msg.get("action", "none")))
        dry = " (practice mode: not really done)" if msg.get("dry_run") else ""
        self.last_var.set(f"{LABELS['last_action']}: {name_for(str(msg.get('name')))} -> {what}{dry}")
        t = float(msg.get("t", 0.0))
        self.events.append((t, name_for(str(msg.get("name")))))
        self.events = [e for e in self.events if e[0] >= t - 5.0]
        if self.wizard is not None and self.wizard.alive():
            self.wizard.on_fired(str(msg.get("name")))

    # ---------------------------------------------------------------- drawing
    def _show_frame(self, fr: dict[str, Any]) -> None:
        self.frame = fr
        probs = fr.get("probs") or {}
        if fr.get("lm") is None:
            self.top_var.set("No hand in view")
        elif not probs:
            self.top_var.set("Watching...")
        elif fr.get("unknown"):
            self.top_var.set(LABELS["not_sure"])
        else:
            self.top_var.set(name_for(str(fr.get("top"))))
        top = str(fr.get("top", NOTHING))
        states = fr.get("states") or {}
        st = states.get(top)
        zone = "" if fr.get("zone_ok", True) else "  (your hand is in an 'Ignore here' area)"
        self.state_var.set((state_text(st) if st and top != NOTHING else "") + zone)
        if fr.get("builtin"):
            self.builtin_var.set(f"Built-in gesture: {fr['builtin']}")
        near = fr.get("nearest")
        self.near_var.set(
            f"{LABELS['looks_like']}: practice example #{near[1]} of {name_for(near[0])}" if near else ""
        )
        thr = self._threshold(top)
        self._bar(self.sure_canvas, float(fr.get("top_p", 0.0)) if probs else 0.0, thr, 260)
        self._draw_bars(probs, states)
        if probs:
            self.timeline.append((float(fr.get("t", 0.0)), dict(probs)))
            t = float(fr.get("t", 0.0))
            self.timeline = [p for p in self.timeline if p[0] >= t - 5.0]
        if self.details.get():
            self._draw_timeline()
        if self.wizard is not None and self.wizard.alive():
            self.wizard.on_frame(fr)
        self._draw_canvas()

    def _threshold(self, name: str) -> float:
        from away.studio.pack import gesture_settings

        g = self.ctl.store.get(name)
        try:
            return gesture_settings(self.ctl.cfg, name, g.kind if g else "static").threshold
        except ValueError:
            return 0.8

    def _bar(self, c: Any, p: float, thr: float, width: int) -> None:
        c.delete("all")
        c.create_rectangle(0, 6, width, 20, fill="#e3e3e3", outline="")
        c.create_rectangle(0, 6, width * max(0.0, min(1.0, p)), 20, fill="#43a047" if p >= thr else "#90a4ae", outline="")
        c.create_line(width * thr, 2, width * thr, 24, fill="#d32f2f", width=2)
        c.create_text(width - 4, 13, text=sure_text(p), anchor="e", fill="#222")

    def _draw_bars(self, probs: dict[str, float], states: dict[str, str]) -> None:
        c = self.bars
        c.delete("all")
        if not probs:
            c.create_text(4, 10, anchor="w", fill="#777", text=f"{LABELS['how_sure']}: shows here when a hand is in view")
            return
        items = list(probs.items())[:8]
        c.configure(height=max(40, 22 * len(items) + 6))
        for i, (name, p) in enumerate(items):
            y = 4 + 22 * i
            label = name_for(name)
            st = states.get(name)
            if st in ("firing", "arming"):
                label += f"  ({state_text(st)})"
            c.create_text(0, y + 7, anchor="w", text=label[:26], fill="#222")
            x0, x1 = 150, 290
            c.create_rectangle(x0, y + 2, x1, y + 14, fill="#eee", outline="")
            c.create_rectangle(x0, y + 2, x0 + (x1 - x0) * p, y + 14, fill=PALETTE[i % len(PALETTE)], outline="")
            if name != NOTHING:
                tx = x0 + (x1 - x0) * self._threshold(name)
                c.create_line(tx, y, tx, y + 16, fill="#d32f2f")

    def _draw_canvas(self) -> None:
        import tkinter as tk

        c = self.canvas
        c.delete("all")
        fr = self.frame
        w, h = CANVAS_W, CANVAS_H
        prev = fr.get("preview") if fr else None
        if prev:
            pw, ph, data = prev
            try:
                self.photo = tk.PhotoImage(data=data, format="PPM")
                zx = max(1, w // max(1, pw))
                if zx > 1:
                    self.photo = self.photo.zoom(zx, zx)
                c.create_image(w // 2, h // 2, image=self.photo)
            except Exception:
                self.photo = None
        zm = self.zmap
        for r in range(zm.rows):
            for col in range(zm.cols):
                if zm.cells[r][col] == IGNORE:
                    c.create_rectangle(zm.xs[col] * w, zm.ys[r] * h, zm.xs[col + 1] * w, zm.ys[r + 1] * h,
                                       fill="#c62828", stipple="gray50", outline="")  # fmt: skip
        for x in zm.xs[1:-1]:
            c.create_line(x * w, 0, x * w, h, fill="#ffeb3b", dash=(3, 3))
        for y in zm.ys[1:-1]:
            c.create_line(0, y * h, w, y * h, fill="#ffeb3b", dash=(3, 3))
        for rect in zm.rects:
            colour = "#c62828" if rect.mode == IGNORE else "#2e7d32"
            c.create_rectangle(rect.x0 * w, rect.y0 * h, rect.x1 * w, rect.y1 * h, outline=colour, width=2,
                               fill=colour, stipple="gray25")  # fmt: skip
        if zm.line is not None:
            ln = zm.line
            c.create_line(ln.x0 * w, ln.y0 * h, ln.x1 * w, ln.y1 * h, fill="#00e5ff", width=3)
            c.create_text(8, ln.y0 * h - 10, anchor="w", text="gestures count above this line", fill="#00e5ff")
        if self._drag and self._drag[0] == "box":
            x0, y0, x1, y1 = self._drag[1]
            c.create_rectangle(x0 * w, y0 * h, x1 * w, y1 * h, outline="white", dash=(4, 2), width=2)
        if fr:
            self._draw_hand(c, fr.get("helper"), "#bbbbbb", None)
            self._draw_hand(c, fr.get("lm"), "#4fc3f7", fr.get("saliency") if self.details.get() else None)
            pt = fr.get("point")
            if pt:
                colour = "#00c853" if fr.get("zone_ok", True) else "#ff1744"
                c.create_oval(pt[0] * w - 7, pt[1] * h - 7, pt[0] * w + 7, pt[1] * h + 7, outline=colour, width=3)
        else:
            c.create_text(w // 2, h // 2, text="Turn the camera on to see your hand here", fill="#aaa")
        share = round(100 * zm.active_fraction(20))
        self.zone_note.set(f"Gestures work in {share}% of the picture ({self.zprofile}).")

    def _draw_hand(self, c: Any, lm: Any, colour: str, saliency: Any) -> None:
        from away.hands.autotune import HAND_CONNECTIONS

        if lm is None:
            return
        pts = [(float(p[0]) * CANVAS_W, float(p[1]) * CANVAS_H) for p in lm]
        if len(pts) != 21:
            return
        for a, b in HAND_CONNECTIONS:
            c.create_line(*pts[a], *pts[b], fill=colour, width=2)
        for i, (x, y) in enumerate(pts):
            r = 2.5
            if saliency is not None and i < len(saliency):
                r = 3 + 10 * float(saliency[i])
                c.create_oval(x - r, y - r, x + r, y + r, outline="#ffab00", width=2)
            else:
                c.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")

    def _draw_timeline(self) -> None:
        c = self.tl_canvas
        c.delete("all")
        w, h = int(c.cget("width")), int(c.cget("height"))
        if not self.timeline:
            c.create_text(8, h // 2, anchor="w", fill="#777", text="Turn the camera on to see the last 5 seconds.")
            return
        t1 = self.timeline[-1][0]
        names = list(self.timeline[-1][1])
        for i, name in enumerate(names[:8]):
            pts: list[float] = []
            for t, p in self.timeline:
                pts += [w * (1 - (t1 - t) / 5.0), (h - 16) * (1 - p.get(name, 0.0)) + 2]
            if len(pts) >= 4:
                c.create_line(*pts, fill=PALETTE[i % len(PALETTE)], width=2)
            c.create_text(4, 10 + 12 * i, anchor="w", text=name_for(name), fill=PALETTE[i % len(PALETTE)])
        for t, text in self.events:
            x = w * (1 - (t1 - t) / 5.0)
            c.create_line(x, h - 14, x, h, fill="#d32f2f", width=2)
            c.create_text(x + 2, h - 8, anchor="w", text=text, fill="#d32f2f")

    def _draw_confusion(self, res: dict[str, Any]) -> None:
        c = self.cm_canvas
        c.delete("all")
        labels = list(res.get("labels") or [])
        cm = res.get("confusion") or []
        self.split_var.set(summary_split(res.get("split") or {}))
        if not labels or not cm or not sum(map(sum, cm)):
            c.create_text(8, 20, anchor="w", fill="#777", text="Practise 3 rounds to see this.")
            return
        n = len(labels)
        cell = min(28, 150 // max(1, n))
        x0, y0 = 90, 20
        for j, lab in enumerate(labels):
            c.create_text(x0 + cell * j + cell / 2, 10, text=name_for(lab)[:6], font=("TkDefaultFont", 7))
        for i, row in enumerate(cm):
            c.create_text(x0 - 4, y0 + cell * i + cell / 2, anchor="e", text=name_for(labels[i])[:12])
            total = max(1, sum(row))
            for j, v in enumerate(row):
                shade = int(255 - 180 * v / total)
                colour = f"#{shade:02x}{255 if i == j else shade:02x}{shade:02x}" if v else "#ffffff"
                c.create_rectangle(x0 + cell * j, y0 + cell * i, x0 + cell * (j + 1), y0 + cell * (i + 1),
                                   fill=colour, outline="#ccc")  # fmt: skip
                if v:
                    c.create_text(x0 + cell * j + cell / 2, y0 + cell * i + cell / 2, text=str(v))

    # ---------------------------------------------------------------- zones (click / drag on the picture)
    def _xy(self, e: Any) -> tuple[float, float]:
        return min(max(e.x / CANVAS_W, 0.0), 1.0), min(max(e.y / CANVAS_H, 0.0), 1.0)

    def _press(self, e: Any) -> None:
        x, y = self._xy(e)
        zm = self.zmap
        ix = zm.nearest_edge("x", x, EDGE_GRAB_PX / CANVAS_W)
        iy = zm.nearest_edge("y", y, EDGE_GRAB_PX / CANVAS_H)
        tool = self.tool.get()
        if tool in ("works", "ignore") and (ix is not None or iy is not None):
            self._drag = ("edge", ("x", ix) if ix is not None else ("y", iy))
            return
        if tool == "box":
            self._drag = ("box", [x, y, x, y])
        elif tool == "line":
            zm.line = ActivationLine(0.0, y, 1.0, y, keep=-1)
            self._drag = ("line", None)
        else:
            mode = ACTIVE if tool == "works" else IGNORE
            r, col = zm.cell_at(x, y)
            zm.set_cell(r, col, mode)
            self._drag = ("paint", mode)
        self._draw_canvas()

    def _motion(self, e: Any) -> None:
        if not self._drag:
            return
        x, y = self._xy(e)
        kind, data = self._drag
        if kind == "edge":
            axis, i = data
            self.zmap.move_edge(axis, i, x if axis == "x" else y)
        elif kind == "box":
            data[2], data[3] = x, y
        elif kind == "line" and self.zmap.line is not None:
            self.zmap.line = ActivationLine(0.0, y, 1.0, y, keep=-1)
        elif kind == "paint":
            r, col = self.zmap.cell_at(x, y)
            self.zmap.set_cell(r, col, data)
        self._draw_canvas()

    def _release(self, _e: Any) -> None:
        if self._drag and self._drag[0] == "box":
            x0, y0, x1, y1 = self._drag[1]
            if abs(x1 - x0) > 0.02 and abs(y1 - y0) > 0.02:
                self.zmap.rects.append(ZoneRect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), IGNORE))
        self._drag = None
        self._zones_changed()

    def _zones_changed(self) -> None:
        self.cmd.put({"cmd": "zones", "map": self.zmap.to_dict()})
        err = self.ctl.save_zone_map(self.zmap, self.zprofile)
        self._report(err, f"Areas saved for this camera ({self.zprofile}).")
        self._draw_canvas()

    def _resize(self) -> None:
        try:
            self.zmap.resize(int(self.cols.get()), int(self.rows.get()))
        except (ValueError, Exception):
            return
        self._zones_changed()

    def _clear_zones(self) -> None:
        self.zmap = ZoneMap()
        self.zmap.resize(int(self.cols.get() or 6), int(self.rows.get() or 4))
        self._zones_changed()

    def _zone_profile_changed(self) -> None:
        self.zprofile = self.zprof_var.get() or "default"
        self.zmap = self.ctl.zone_map(self.zprofile)
        self.cols.set(self.zmap.cols)
        self.rows.set(self.zmap.rows)
        self.cmd.put({"cmd": "zones", "map": self.zmap.to_dict()})
        self._draw_canvas()

    # ---------------------------------------------------------------- gestures
    def teach(self) -> None:
        if self.wizard is not None and self.wizard.alive():
            self.wizard.top.lift()
            return
        self.wizard = Wizard(self)

    def practise(self) -> None:
        name = self.selected()
        if name is None:
            self.show_problem("Click one of your gestures in the list first.")
            return
        g = self.ctl.store.get(name)
        self.wizard = Wizard(self, existing=name, kind=g.kind if g else "static", rounds=1)

    def record_relax(self) -> None:
        if not self.running() and not self.start():
            return
        from away.studio.pack import new_session_id

        self.relax_pending = True
        self.cmd.put({"cmd": "record", "label": NOTHING, "session": new_session_id(), "seconds": 4.0,
                      "countdown": 2.0, "tag": ("relax", 0)})  # fmt: skip
        self.status.set("Relax round: in 2 seconds, keep your hand in view and move it naturally for 4 seconds.")

    def learn(self, after: Any = None) -> None:
        self.status.set(WIZARD["learning"])
        self.result_var.set(WIZARD["learning"])

        def work() -> None:
            res = self.ctl.learn(progress=lambda text: self.w._ui.post(lambda: self.status.set(text)))
            self.w._ui.post(lambda: self._learned(res, after))

        threading.Thread(target=work, daemon=True).start()

    def _learned(self, res: Any, after: Any = None) -> None:
        if not res.ok:
            self.show_problem(res.message)
            self.result_var.set(res.message)
        else:
            self.clear_problem()
            self.status.set(res.message)
            self._show_result(res.to_dict())
            self.cmd.put({"cmd": "reload"})
        self.refresh()
        if after is not None:
            after(res)

    def _show_result(self, res: dict[str, Any]) -> None:
        n = int(sum(sum(r) for r in res.get("confusion") or []))
        lines = [score_text(res.get("score"), n)]
        lines += [f"• {w}" for w in res.get("warnings") or []]
        self.result_var.set("\n".join(lines))
        if self.details.get():
            self._draw_confusion(res)

    def remove(self) -> None:
        from tkinter import messagebox

        name = self.selected()
        if name is None:
            self.show_problem("Click one of your gestures in the list first.")
            return
        if messagebox.askyesno(TAB_TITLE, f"Remove '{name_for(name)}' and its practice examples?", parent=self.root):
            self._report(self.ctl.remove_gesture(name), f"Removed '{name_for(name)}'. Press 'Learn my gestures now'.")
            self.refresh()

    def edit_gesture(self) -> None:
        name = self.selected()
        if name is None:
            self.show_problem("Click one of your gestures in the list first.")
            return
        GestureDialog(self, name)

    def share(self) -> None:
        from tkinter import filedialog, messagebox

        if not self.ctl.store.gestures():
            self.show_problem("Teach a gesture first: press 'Teach a new gesture'.")
            return
        clips = messagebox.askyesno(
            TAB_TITLE,
            "Also include your practice recordings?\n\nThey are only movements of your hand points (no pictures), "
            "but say No if you want to keep them to yourself.",
            default="no",
            parent=self.root,
        )
        p = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Gesture pack", "*.zip")],
                                         initialfile="my-gestures.zip")  # fmt: skip
        if p:
            self._report(self.ctl.export(Path(p), include_clips=clips), "Saved your gestures to the file.")

    def add_file(self) -> None:
        from tkinter import filedialog

        p = filedialog.askopenfilename(filetypes=[("Gesture pack", "*.zip")])
        if not p:
            return
        notes, err = self.ctl.import_(Path(p))
        self._report(err, "; ".join(notes))
        self.refresh()
        self.cmd.put({"cmd": "reload"})

    def delete_recordings(self) -> None:
        from tkinter import messagebox

        if messagebox.askyesno(TAB_TITLE, "Delete every practice recording on this computer?\n\n"
                               "Your gestures keep working, but you will need new recordings to teach them again.",
                               parent=self.root):  # fmt: skip
            n = self.ctl.delete_all_recordings()
            self._report(None, f"Deleted {n} recording{'s' if n != 1 else ''}.")
            self.refresh()

    def improve_builtin(self) -> None:
        BuiltinDialog(self)


class Wizard:  # pragma: no cover - widgets
    """'Teach a new gesture' in 3 steps (or, for 'Practise more', step 2 only)."""

    def __init__(self, tab: StudioTab, existing: str | None = None, kind: str = "static", rounds: int = 3,
                 saver: Any = None, title: str | None = None, after: Any = None) -> None:  # fmt: skip
        tk, ttk = tab.tk, tab.ttk
        self.tab = tab
        self.name = existing
        self.kind = kind
        self.rounds = rounds
        self.saver = saver
        self.after = after
        self.plan: WizardPlan | None = None
        self.recording = False
        self.top = tk.Toplevel(tab.root)
        self.top.title(title or WIZARD["title"])
        self.top.transient(tab.root)
        self.top.protocol("WM_DELETE_WINDOW", self.cancel)
        self.body = ttk.Frame(self.top, padding=14)
        self.body.pack(fill="both", expand=True)
        self.big = tk.StringVar()
        self.small = tk.StringVar()
        self.err = tk.StringVar()
        if existing is None:
            self.step1()
        else:
            self.step2()

    def alive(self) -> bool:
        try:
            return bool(self.top.winfo_exists())
        except Exception:
            return False

    def _clear(self) -> None:
        for wdg in self.body.winfo_children():
            wdg.destroy()

    def _title(self, text: str, help_text: str) -> None:
        ttk = self.tab.ttk
        ttk.Label(self.body, text=text, font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        ttk.Label(self.body, text=help_text, wraplength=460, justify="left").pack(anchor="w", pady=(4, 10))

    # ---- step 1
    def step1(self) -> None:
        tk, ttk = self.tab.tk, self.tab.ttk
        self._clear()
        self._title(WIZARD["step1"], WIZARD["step1_help"])
        self.name_var = tk.StringVar()
        e = ttk.Entry(self.body, textvariable=self.name_var, width=30, font=("TkDefaultFont", 12))
        e.pack(anchor="w")
        e.focus_set()
        self.kind_var = tk.StringVar(value="static")
        box = ttk.Frame(self.body)
        box.pack(anchor="w", pady=8)
        for k, text in KIND_WORDS.items():
            ttk.Radiobutton(box, text=text, value=k, variable=self.kind_var).pack(anchor="w")
        self.hands_var = tk.IntVar(value=1)
        box2 = ttk.Frame(self.body)
        box2.pack(anchor="w")
        ttk.Radiobutton(box2, text="One hand", value=1, variable=self.hands_var).pack(side="left")
        ttk.Radiobutton(box2, text="Two hands", value=2, variable=self.hands_var).pack(side="left", padx=8)
        tk.Label(self.body, textvariable=self.err, fg="#b71c1c", wraplength=460, justify="left").pack(anchor="w", pady=4)
        nav = ttk.Frame(self.body)
        nav.pack(fill="x", pady=(8, 0))
        ttk.Button(nav, text=WIZARD["cancel"], command=self.cancel).pack(side="left")
        ttk.Button(nav, text=WIZARD["next"], style="Big.TButton", command=self._step1_next).pack(side="right")

    def _step1_next(self) -> None:
        name, err = self.tab.ctl.create_gesture(self.name_var.get(), self.kind_var.get(), int(self.hands_var.get()))
        if name is None:
            self.err.set(err or "")
            return
        self.name, self.kind = name, self.kind_var.get()
        self.tab.refresh()
        self.step2()

    # ---- step 2
    def step2(self) -> None:
        tk, ttk = self.tab.tk, self.tab.ttk
        self._clear()
        title = WIZARD["step2"] if self.saver is None and self.rounds > 1 else LABELS["practise_more"]
        self._title(title, WIZARD["step2_help"])
        assert self.name is not None
        self.plan = WizardPlan(self.name, self.kind, rounds=self.rounds)
        tk.Label(self.body, textvariable=self.big, font=("TkDefaultFont", 28, "bold"), fg="#1b5e20").pack(pady=6)
        ttk.Label(self.body, textvariable=self.small).pack()
        self.prog = ttk.Progressbar(self.body, length=420, maximum=len(self.plan.steps))
        self.prog.pack(pady=6)
        tk.Label(self.body, textvariable=self.err, fg="#b71c1c", wraplength=460, justify="left").pack(anchor="w")
        nav = ttk.Frame(self.body)
        nav.pack(fill="x", pady=(8, 0))
        ttk.Button(nav, text=WIZARD["cancel"], command=self.cancel).pack(side="left")
        ttk.Button(nav, text=WIZARD["bad_one"], command=self._redo).pack(side="left", padx=6)
        self.start_btn = ttk.Button(nav, text=WIZARD["start"], style="Big.TButton", command=self._start_recording)
        self.start_btn.pack(side="right")
        self.big.set(f"Show me '{name_for(self.name)}'")
        self.small.set(self.plan.progress_text())

    def _start_recording(self) -> None:
        if not self.tab.running() and not self.tab.start():
            self.err.set("The camera could not start. See the message in the Studio tab.")
            return
        self.start_btn.configure(state="disabled")
        self.err.set("")
        self._send()

    def _send(self) -> None:
        if self.plan is None or self.plan.done:
            self._recorded_all()
            return
        cmd = self.plan.command()
        if cmd is not None:
            self.recording = True
            self.tab.cmd.put(cmd)
            step = self.plan.current
            self.small.set(self.plan.progress_text())
            self.big.set(WIZARD["relax"] if step is not None and step.kind == "relax" else WIZARD["get_ready"])

    def on_message(self, msg: dict[str, Any]) -> None:
        if self.plan is None:
            return
        step = self.plan.current
        relax = step is not None and step.kind == "relax"
        if msg["type"] == "countdown":
            self.big.set(WIZARD["relax"] if relax else str(msg.get("left", "")))
        elif msg["type"] == "recording":
            self.big.set("Relax and move naturally..." if relax else WIZARD["go"])
        elif msg["type"] == "take":
            tag = msg.get("tag")
            if not (isinstance(tag, tuple) and tag and tag[0] == "wizard"):
                return
            self.recording = False
            if self.saver is not None:
                path, err = self.saver(msg)
            else:
                path, err = self.tab.ctl.save_take(msg)
            if path is None:
                self.err.set(err or "")
                self.tab.root.after(1200, self._send)  # same step again
                return
            self.err.set("")
            self.plan.accept(path)
            self.prog.configure(value=self.plan.pos)
            self.tab.root.after(500, self._send)

    def _redo(self) -> None:
        if self.plan is None:
            return
        if self.recording:
            self.tab.cmd.put({"cmd": "cancel"})
            self.recording = False
        for p in self.plan.redo_last():
            with contextlib.suppress(OSError):
                Path(p).unlink()
        self.prog.configure(value=self.plan.pos)
        self.tab.root.after(300, self._send)

    def camera_stopped(self) -> None:
        if self.recording:
            self.err.set("The camera stopped. Turn it on again and press Start.")
            self.start_btn.configure(state="normal")
            self.recording = False

    def _recorded_all(self) -> None:
        if self.saver is not None:
            self.top.destroy()
            if self.after is not None:
                self.after()
            return
        self.tab.refresh()
        self.step3()

    # ---- step 3
    def step3(self) -> None:
        tk, ttk = self.tab.tk, self.tab.ttk
        self._clear()
        self._title(WIZARD["step3"], WIZARD["step3_help"])
        self.big.set(WIZARD["learning"])
        tk.Label(self.body, textvariable=self.big, font=("TkDefaultFont", 18, "bold"), fg="#1b5e20").pack(pady=4)
        self.try_canvas = tk.Canvas(self.body, width=300, height=26, highlightthickness=0)
        self.try_canvas.pack()
        Tooltip(self.try_canvas, TOOLTIPS["how_sure"])
        tk.Label(self.body, textvariable=self.small, wraplength=460, justify="left").pack(anchor="w", pady=4)
        self.small.set("")
        self.picker = ActionPicker(self.tab, self.body, self.name or "")
        tk.Label(self.body, textvariable=self.err, fg="#b71c1c", wraplength=460, justify="left").pack(anchor="w")
        nav = ttk.Frame(self.body)
        nav.pack(fill="x", pady=(8, 0))
        ttk.Button(nav, text=LABELS["practise_more"], command=self._more).pack(side="left")
        ttk.Button(nav, text=WIZARD["finish"], style="Big.TButton", command=self._finish).pack(side="right")
        self.tab.learn(after=self._learned)

    def _learned(self, res: Any) -> None:
        if not self.alive():
            return
        if res.ok:
            self.big.set(f"Now try '{name_for(self.name or '')}'!")
            self.small.set("\n".join([score_text(res.score, res.n_checked), *[f"• {w}" for w in res.warnings]]))
        else:
            self.big.set("Almost there")
            self.small.set(res.message)

    def on_frame(self, fr: dict[str, Any]) -> None:
        if self.plan is None or self.plan.done is False or not hasattr(self, "try_canvas"):
            return
        with contextlib.suppress(Exception):
            p = float((fr.get("probs") or {}).get(self.name or "", 0.0))
            self.tab._bar(self.try_canvas, p, self.tab._threshold(self.name or ""), 300)

    def on_fired(self, name: str) -> None:
        if name == self.name and hasattr(self, "try_canvas"):
            with contextlib.suppress(Exception):
                self.big.set("It saw it!")

    def _more(self) -> None:
        self.rounds = 1
        self.step2()

    def _finish(self) -> None:
        err = self.picker.save()
        if err:
            self.err.set(err)
            return
        self.tab.refresh()
        self.top.destroy()

    def cancel(self) -> None:
        if self.recording:
            self.tab.cmd.put({"cmd": "cancel"})
        self.top.destroy()
        self.tab.refresh()


class ActionPicker:  # pragma: no cover - widgets
    """'What should it do?': plain choices, with a text box or a list for the details."""

    def __init__(self, tab: StudioTab, parent: Any, name: str) -> None:
        tk, ttk = tab.tk, tab.ttk
        self.tab, self.name = tab, name
        frm = ttk.Frame(parent)
        frm.pack(anchor="w", pady=6, fill="x")
        ttk.Label(frm, text="What should it do?").grid(row=0, column=0, sticky="w")
        kind, arg = tab.ctl.current_action(name)
        labels = [c[0] for c in ACTION_CHOICES]
        current = next((c[0] for c in ACTION_CHOICES if c[1] == kind), labels[0])
        self.choice = tk.StringVar(value=current)
        cb = ttk.Combobox(frm, textvariable=self.choice, values=labels, state="readonly", width=26)
        cb.grid(row=1, column=0, sticky="w")
        Tooltip(cb, TOOLTIPS["action"])
        self.arg = tk.StringVar(value=arg)
        self.arg_box = ttk.Combobox(frm, textvariable=self.arg, width=28)
        self.arg_box.grid(row=1, column=1, sticky="w", padx=6)
        Tooltip(self.arg_box, TOOLTIPS["action_detail"])
        cb.bind("<<ComboboxSelected>>", lambda _e: self._kind_changed())
        self._kind_changed(keep=True)

    def kind(self) -> str:
        return next((c[1] for c in ACTION_CHOICES if c[0] == self.choice.get()), "none")

    def _kind_changed(self, keep: bool = False) -> None:
        args = self.tab.ctl.choice_args(self.kind())
        if args:
            self.arg_box.configure(values=list(args), state="readonly")
            if not keep or self.arg.get() not in args:
                self.arg.set(args[0])
        elif args == ():
            self.arg_box.configure(values=[], state="disabled")
            self.arg.set("")
        else:
            self.arg_box.configure(values=[], state="normal")
            if not keep:
                self.arg.set("")

    def save(self) -> str | None:
        return self.tab.ctl.set_action(self.name, self.kind(), self.arg.get())


class GestureDialog:  # pragma: no cover - widgets
    """'Change what it does...': the action, on/off, how sure it must be, hold time, rest time, extra steps."""

    def __init__(self, tab: StudioTab, name: str) -> None:
        from away.studio.pack import gesture_settings

        tk, ttk = tab.tk, tab.ttk
        self.tab, self.name = tab, name
        g = tab.ctl.store.get(name)
        s = gesture_settings(tab.ctl.cfg, name, g.kind if g else "static")
        self.top = tk.Toplevel(tab.root)
        self.top.title(f"{name_for(name)}: what it does")
        self.top.transient(tab.root)
        body = ttk.Frame(self.top, padding=12)
        body.pack(fill="both", expand=True)
        self.picker = ActionPicker(tab, body, name)
        self.enabled = tk.BooleanVar(value=s.enabled)
        tip(ttk.Checkbutton(body, text="Switched on", variable=self.enabled), "enabled").pack(anchor="w", pady=4)
        self.thr = tk.DoubleVar(value=round(s.threshold * 100))
        self.hold = tk.DoubleVar(value=s.hold_s)
        self.rest = tk.DoubleVar(value=s.cooldown_s)
        for text, var, lo, hi, res, key in (
            ("How sure it must be (%)", self.thr, 30, 99, 1, "threshold"),
            ("Hold for (seconds)", self.hold, 0.0, 2.0, 0.05, "hold"),
            ("Rest time (seconds)", self.rest, 0.0, 5.0, 0.1, "rest"),
        ):
            row = ttk.Frame(body)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=text, width=24).pack(side="left")
            sc = tk.Scale(row, from_=lo, to=hi, resolution=res, orient="horizontal", length=240, variable=var)
            tip(sc, key).pack(side="left")
        ttk.Label(body, text="Extra steps (optional, one per line):").pack(anchor="w", pady=(6, 0))
        self.then = tk.Text(body, width=40, height=4)
        self.then.insert("1.0", "\n".join(s.then))
        tip(self.then, "then").pack(anchor="w")
        self.err = tk.StringVar()
        tk.Label(body, textvariable=self.err, fg="#b71c1c", wraplength=420, justify="left").pack(anchor="w")
        nav = ttk.Frame(body)
        nav.pack(fill="x", pady=(8, 0))
        ttk.Button(nav, text=WIZARD["cancel"], command=self.top.destroy).pack(side="left")
        ttk.Button(nav, text="Save", style="Big.TButton", command=self.save).pack(side="right")

    def save(self) -> None:
        err = self.picker.save() or self.tab.ctl.set_settings(
            self.name,
            bool(self.enabled.get()),
            float(self.thr.get()) / 100.0,
            float(self.hold.get()),
            float(self.rest.get()),
            self.then.get("1.0", "end"),
        )
        if err:
            ex = explain(err)
            self.err.set(ex[0] + ("\n" + HELP.get(ex[1], "") if ex and ex[1] in HELP else "") if ex else err)
            return
        self.tab.refresh()
        self.tab.cmd.put({"cmd": "reload"})
        self.top.destroy()


class BuiltinDialog:  # pragma: no cover - widgets
    """'Improve the built-in gestures': record 5 examples of one built-in gesture, then fine-tune the built-in
    model with them (needs the training add-on)."""

    def __init__(self, tab: StudioTab) -> None:
        from away.hands.synthetic import GESTURES

        tk, ttk = tab.tk, tab.ttk
        self.tab = tab
        self.top = tk.Toplevel(tab.root)
        self.top.title(LABELS["improve_builtin"].rstrip("."))
        self.top.transient(tab.root)
        body = ttk.Frame(self.top, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Which built-in gesture should work better for you?", wraplength=380).pack(anchor="w")
        choices = [g for g in GESTURES if g != "idle"]
        self.var = tk.StringVar(value=choices[0])
        ttk.Combobox(body, textvariable=self.var, values=choices, state="readonly").pack(anchor="w", pady=6)
        ttk.Label(body, text="You will show it 5 times. Then the app updates the built-in gestures (this can take "
                  "a minute).", wraplength=380).pack(anchor="w")  # fmt: skip
        nav = ttk.Frame(body)
        nav.pack(fill="x", pady=(8, 0))
        ttk.Button(nav, text=WIZARD["cancel"], command=self.top.destroy).pack(side="left")
        ttk.Button(nav, text=WIZARD["start"], style="Big.TButton", command=self.go).pack(side="right")

    def go(self) -> None:
        from away.studio.trainer import torch_available

        if not torch_available():
            self.tab.show_problem("Improving the built-in gestures needs the training add-on.")
            self.top.destroy()
            return
        label = self.var.get()
        self.top.destroy()
        from away import paths
        from away.hands.recorder import save_clip

        def saver(msg: dict[str, Any]) -> tuple[Path | None, str | None]:
            frames = [f for f, _h in msg.get("frames") or [] if f is not None]
            if len(frames) < 3:
                return None, "Your hand was not in the picture. Move it into view and try again."
            if msg.get("label") == NOTHING:
                return Path(""), None  # built-in training has its own 'idle' examples
            p = save_clip(paths.recordings_dir(), label, frames, str(msg.get("handedness", "Right")),
                          float(msg.get("fps") or 30), session_id=str(msg.get("session")))  # fmt: skip
            return p, None

        def retrain() -> None:
            from away.studio.trainer import retrain_builtin

            self.tab.status.set("Updating the built-in gestures... (this can take a minute)")

            def work() -> None:
                from away import paths as pth

                try:
                    retrain_builtin(pth.models_dir(), pth.recordings_dir())
                    text = "The built-in gestures were updated with your examples. Restart Project Away to use them."
                except Exception as exc:
                    text = f"Could not update the built-in gestures ({exc})."
                self.tab.w._ui.post(lambda: self.tab.status.set(text))

            threading.Thread(target=work, daemon=True).start()

        self.tab.wizard = Wizard(self.tab, existing=label, kind="motion", rounds=1, saver=saver,
                                 title=LABELS["improve_builtin"].rstrip("."), after=retrain)  # fmt: skip
