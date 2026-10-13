"""Settings > Shortcuts: record a shortcut once, name it, pick a gesture or phrase. That's it.

The tab is deliberately plain: one big "Record a shortcut" button, a list of your shortcuts with what runs them,
and friendly problems with a fix button. The step editor exists but sits behind "Edit steps".

:class:`ShortcutsModel` and :class:`StepEditor` hold all the logic and never import tkinter (tested headless);
:func:`build_shortcuts_tab` is the thin Tk view. Shortcuts are bound in the ``default`` profile, so they work in
every app; the binding is an ordinary ``macro:<id>`` action, visible (and editable) on the Bindings tab too.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away.app.macro_service import MacroService
from away.bindings import GESTURE_TRIGGERS, NONE, Action, normalize_phrase
from away.control.macro_play import FIX_LABELS, PlayResult
from away.control.macros import (
    SPEEDS,
    Macro,
    MacroError,
    MacroStore,
    SensitiveExportError,
    Step,
    name_from_title,
    title_from_name,
)

PROFILE = "default"
# gestures that drive the pointer itself; offering them would take away basic control
CORE_GESTURES = ("point", "pinch", "pinch_hold", "two_finger_scroll", "four_finger_scroll", "open_palm")
FIX_LABELS_PANEL = {
    **FIX_LABELS,
    "replace_binding": "Replace it",
    "record": "Record a shortcut",
    "export_private": "Export anyway",
}
CATEGORY_PREFIX = {"gestures": "gesture", "voice": "voice", "face": "face", "hotkeys": "hotkey"}
CATEGORY_TABLE = {v: k for k, v in CATEGORY_PREFIX.items()}


@dataclass
class Problem:
    """Something went wrong, in plain words, with at most one button that fixes it."""

    message: str
    fix: str = ""
    name: str = ""  # the shortcut it is about

    @property
    def fix_label(self) -> str:
        return FIX_LABELS_PANEL.get(self.fix, "")


@dataclass
class ShortcutRow:
    name: str
    title: str
    runs_when: str  # 'Say "open mail"' / "Gesture: Swipe left" / "Not set yet"
    steps: int
    sensitive: bool


def gesture_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def action_label(act: Action) -> str:
    """Plain words for what a trigger does now ("Press left", "Click", "Run Open mail")."""
    if act.kind == "none":
        return "nothing"
    if act.kind == "macro":
        return f'run "{title_from_name(act.arg)}"'
    if act.kind == "key":
        return f"press {act.arg}"
    if act.kind == "mouse":
        return act.arg.replace("_", " ")
    if act.kind == "type":
        return "type text"
    return str(act)


def shortcut_ids(cfg: Any = None) -> list[str]:
    """Saved shortcut ids (the Bindings tab offers them for the ``macro`` action)."""
    try:
        return MacroStore.default(getattr(cfg, "macros", None)).names()
    except Exception:
        return []


class ShortcutsModel:
    def __init__(self, settings: Any, service: MacroService) -> None:
        self.settings = settings  # SettingsModel: .bindings, .save(), .cfg
        self.service = service
        self.draft: Macro | None = None

    @property
    def store(self) -> MacroStore:
        return self.service.store

    @property
    def bindings(self) -> Any:
        return self.settings.bindings

    # ---------------------------------------------------------------- list
    def triggers_for(self, name: str) -> list[str]:
        target = Action("macro", name)
        out = []
        for cat, table in self.bindings.effective(PROFILE).items():
            out += [f"{CATEGORY_PREFIX[cat]}:{trig}" for trig, act in table.items() if act == target]
        return out

    def runs_when(self, name: str) -> str:
        parts = []
        for t in self.triggers_for(name):
            cat, _, trig = t.partition(":")
            if cat == "voice":
                parts.append(f'Say "{trig}"')
            elif cat == "gesture":
                parts.append(f"Gesture: {gesture_label(trig)}")
            elif cat == "hotkey":
                parts.append(f"Keys: {trig}")
            else:
                parts.append(f"Face: {gesture_label(trig)}")
        return ", ".join(parts) or "Not set yet"

    def rows(self) -> list[ShortcutRow]:
        return [
            ShortcutRow(m.name, m.label, self.runs_when(m.name), len(m.steps), m.sensitive) for m in self.store.all()
        ]

    def gesture_choices(self) -> list[tuple[str, str]]:
        """``[(gesture id, "Swipe left (now: press left)")]`` a shortcut may take over."""
        out = []
        names = [g for g in GESTURE_TRIGGERS if g not in CORE_GESTURES] + list(self.bindings.custom_gestures)
        for g in dict.fromkeys(names):
            act = self.bindings.resolve("gestures", g, PROFILE)
            now = "" if act.kind == "none" else f" (now: {action_label(act)})"
            out.append((g, f"{gesture_label(g)}{now}"))
        return out

    # ---------------------------------------------------------------- record + save
    def start_recording(self, exclude_rects: list[tuple[int, int, int, int]] | None = None) -> Problem | None:
        err = self.service.start_recording(exclude_rects)
        return Problem(err) if err else None

    def stop_recording(self) -> tuple[Macro | None, Problem | None]:
        draft = self.service.stop_recording()
        if draft is None or not draft.steps:
            self.draft = None
            return None, Problem("Nothing was recorded. Click or type something while recording.", "record")
        self.draft = draft
        return draft, None

    def suggested_title(self) -> str:
        return self.store.next_untitled()[1]

    def recording_note(self, m: Macro) -> str:
        """What was left out of a recording, in plain words ("" when nothing was)."""
        notes = []
        if m.skipped_password_keys:
            notes.append("Typing in a password box was not recorded.")
        if m.skipped_text:
            notes.append(
                "Typed text was not recorded because password boxes can't be detected on this computer. "
                "Add it with Edit steps if you need it."
            )
        return " ".join(notes)

    def save_new(self, title: str, kind: str = "none", value: str = "", replace: bool = False) -> Problem | None:
        """Save the current draft as ``title`` and bind it (``kind``: phrase | gesture | none)."""
        if self.draft is None:
            return Problem("There is no new recording to save. Record a shortcut first.", "record")
        if not title.strip():
            return Problem("Give your shortcut a name, like “Open my mail”.")
        name = name_from_title(title)
        if not name:
            return Problem("Use at least one letter or number in the name.")
        if self.store.exists(name):
            return Problem(f'You already have a shortcut called "{title.strip()}". Pick another name.')
        if kind != "none":
            p = self._check_trigger(name, kind, value, replace)
            if p is not None:
                return p
        try:
            saved = self.service.save_draft(self.draft, title)
        except MacroError as exc:
            return Problem(str(exc))
        self.draft = None
        if kind != "none":
            return self.bind(saved.name, kind, value, replace=True)
        return None

    # ---------------------------------------------------------------- binding
    def _trigger_id(self, kind: str, value: str) -> tuple[str, Problem | None]:
        if kind == "phrase":
            phrase = normalize_phrase(value)
            if not phrase:
                return "", Problem("Type the words you want to say, like “open mail”.")
            return f"voice:{phrase}", None
        if kind == "gesture":
            if value not in dict(self.gesture_choices()):
                return "", Problem("Pick a gesture from the list.")
            return f"gesture:{value}", None
        return "", Problem("Choose a phrase or a gesture.")

    def _check_trigger(self, name: str, kind: str, value: str, replace: bool) -> Problem | None:
        trig, p = self._trigger_id(kind, value)
        if p is not None:
            return p
        cat, _, t = trig.partition(":")
        current = self.bindings.resolve("voice" if cat == "voice" else "gestures", t, PROFILE)
        if current.kind != "none" and current != Action("macro", name) and not replace:
            what = f'Saying "{t}"' if cat == "voice" else f"The {gesture_label(t).lower()} gesture"
            return Problem(
                f"{what} already does: {action_label(current)}. Use it for this shortcut instead?",
                "replace_binding",
                name,
            )
        return None

    def bind(self, name: str, kind: str, value: str, replace: bool = False) -> Problem | None:
        """Make ``kind``/``value`` run shortcut ``name`` (other triggers of it are removed) and save the config."""
        p = self._check_trigger(name, kind, value, replace)
        if p is not None:
            return p
        trig, _ = self._trigger_id(kind, value)
        self._unbind(name)
        try:
            self.bindings.set(trig, Action("macro", name), PROFILE)
        except ValueError as exc:
            return Problem(str(exc))
        return self._save_config()

    def _unbind(self, name: str) -> None:
        for t in self.triggers_for(name):
            with contextlib.suppress(ValueError, KeyError):
                self.bindings.unset(t, PROFILE)
            # a default trigger restored to its default is fine; anything still pointing here is cleared
            cat, _, trig = t.partition(":")
            prof = self.bindings.profiles[PROFILE].table(CATEGORY_TABLE[cat])
            if prof.get(trig) == Action("macro", name):
                prof[trig] = NONE

    def unbind(self, name: str) -> Problem | None:
        self._unbind(name)
        return self._save_config()

    def _save_config(self) -> Problem | None:
        try:
            self.settings.save()
        except Exception as exc:
            return Problem(f"The settings could not be saved ({exc}).")
        return None

    # ---------------------------------------------------------------- manage
    def delete(self, name: str) -> Problem | None:
        self._unbind(name)
        self.store.delete(name)
        return self._save_config()

    def rename(self, name: str, title: str) -> Problem | None:
        new = name_from_title(title)
        if not new:
            return Problem("Use at least one letter or number in the name.")
        triggers = self.triggers_for(name)
        try:
            self.store.rename(name, new, title.strip())
        except MacroError as exc:
            return Problem(str(exc))
        if new != name:
            self._unbind(name)
            for t in triggers:
                with contextlib.suppress(ValueError):
                    self.bindings.set(t, Action("macro", new), PROFILE)
            return self._save_config()
        return None

    def run(self, name: str) -> Problem | None:
        res = self.service.play(name, wait=True)
        return self.problem_from(res, name)

    @staticmethod
    def problem_from(res: PlayResult | None, name: str) -> Problem | None:
        if res is None or res.ok or res.stopped == "aborted":
            return None
        return Problem(res.message, res.fix, name)

    def last_problem(self) -> Problem | None:
        """A problem from the last replay anywhere (the tray app writes it), until dismissed."""
        data = MacroService.read_status(self.store.folder)
        if not data or data.get("ok") or data.get("stopped") not in ("refused", "error"):
            return None
        return Problem(str(data.get("message", "")), str(data.get("fix", "")), str(data.get("name", "")))

    def dismiss_problem(self) -> None:
        MacroService.clear_status(self.store.folder)

    def apply_fix(self, p: Problem) -> Problem | None:
        """Run the fix behind a problem's button. ``record`` / ``record_again`` / ``edit_steps`` are UI flows and
        return the problem unchanged for the view to handle."""
        if p.fix == "allow_other_app":
            try:
                m = self.store.load(p.name)
            except MacroError as exc:
                return Problem(str(exc))
            m.allow_other_app = True
            self.store.save(m)
            self.dismiss_problem()
            return None
        if p.fix == "try_again":
            self.dismiss_problem()
            return self.run(p.name)
        return p

    # ---------------------------------------------------------------- import / export
    def export(self, names: list[str], path: str, include_private: bool = False) -> Problem | None:
        try:
            self.store.export(names, path, include_sensitive=include_private)
        except SensitiveExportError as exc:
            return Problem(str(exc), "export_private")
        except MacroError as exc:
            return Problem(str(exc))
        return None

    def import_file(self, path: str) -> tuple[list[Macro], Problem | None]:
        try:
            return self.store.import_file(path), None
        except MacroError as exc:
            return [], Problem(str(exc))


class StepEditor:
    """Edits a copy of one shortcut; nothing is written until :meth:`save`."""

    def __init__(self, store: MacroStore, name: str) -> None:
        self.store = store
        self.macro = store.load(name)

    @property
    def steps(self) -> list[Step]:
        return self.macro.steps

    def lines(self) -> list[str]:
        return [
            f"{i}. {s.describe()}" + (f"  (after {s.delay:g} s)" if s.delay else "")
            for i, s in enumerate(self.steps, 1)
        ]

    def _ok(self, i: int) -> bool:
        return 0 <= i < len(self.steps)

    def move(self, i: int, by: int) -> int:
        j = i + by
        if self._ok(i) and self._ok(j):
            self.steps[i], self.steps[j] = self.steps[j], self.steps[i]
            return j
        return i

    def delete(self, i: int) -> Problem | None:
        if not self._ok(i):
            return Problem("Pick a step first.")
        if len(self.steps) == 1:
            return Problem("A shortcut needs at least one step. Delete the whole shortcut instead.")
        del self.steps[i]
        return None

    def set_delay(self, i: int, text: str) -> Problem | None:
        if not self._ok(i):
            return Problem("Pick a step first.")
        try:
            v = float(str(text).replace(",", "."))
        except ValueError:
            return Problem("The pause is a number of seconds, like 0.5.")
        if not 0 <= v <= 600:
            return Problem("The pause must be between 0 and 600 seconds.")
        self.steps[i].delay = v
        return None

    def _insert(self, at: int, step: Step) -> Problem | None:
        try:
            step.validate()
        except MacroError as exc:
            return Problem(str(exc).replace("step: ", ""))
        at = len(self.steps) if at < 0 or at > len(self.steps) else at
        self.steps.insert(at, step)
        return None

    def insert_text(self, at: int, text: str) -> Problem | None:
        if not text:
            return Problem("Type the text the shortcut should type.")
        return self._insert(at, Step("text", text=text))

    def insert_key(self, at: int, keys: str) -> Problem | None:
        if not keys.strip():
            return Problem("Type a key, like enter or ctrl+s.")
        return self._insert(at, Step("key", keys=keys))

    def insert_wait(self, at: int, seconds: str) -> Problem | None:
        try:
            v = float(str(seconds).replace(",", "."))
        except ValueError:
            return Problem("A wait is a number of seconds, like 2.")
        return self._insert(at, Step("wait", seconds=v))

    def set_speed(self, speed: str) -> Problem | None:
        if speed not in SPEEDS:
            return Problem(f"Speed is one of {', '.join(SPEEDS)}.")
        self.macro.speed = speed
        return None

    def save(self) -> Problem | None:
        try:
            self.store.save(self.macro)
        except MacroError as exc:
            return Problem(str(exc))
        return None


# --------------------------------------------------------------------------- Tk view
def build_shortcuts_tab(win: Any, nb: Any) -> Any:  # pragma: no cover - widgets; needs a display
    """Add the Shortcuts tab to the Settings notebook. ``win`` is the SettingsWindow."""
    view = ShortcutsTab(win, nb)
    return view


class ShortcutsTab:  # pragma: no cover - widgets; needs a display
    def __init__(self, win: Any, nb: Any) -> None:
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.win = win
        self.root = win.root
        cfg = win.m.cfg.macros
        self.m = ShortcutsModel(win.m, MacroService(cfg, notify=lambda _t, _m: None))
        f = win._scroll_tab(nb, "Shortcuts", 10)
        f.columnconfigure(0, weight=1)
        style = ttk.Style(self.root)
        with contextlib.suppress(Exception):
            big = tkfont.nametofont("TkDefaultFont").copy()
            big.configure(size=big.cget("size") + 4, weight="bold")
            self._big = big
            style.configure("Big.TButton", font=big, padding=(24, 14))
        ttk.Button(f, text="●  Record a shortcut", style="Big.TButton", command=self.record).grid(
            row=0, column=0, sticky="w", pady=(4, 2)
        )
        ttk.Label(
            f,
            text="Do it once (clicks and typing), then name it and choose a gesture or phrase. "
            "Press Esc or say “stop” to stop a running shortcut.",
            foreground="#555",
            wraplength=640,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.banner = ttk.Frame(f, padding=8, relief="groove")
        self.banner_msg = tk.StringVar()
        ttk.Label(self.banner, textvariable=self.banner_msg, wraplength=520, justify="left").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(self.banner, text="Dismiss", command=self.dismiss).pack(side="right", padx=2)
        self.fix_btn = ttk.Button(self.banner, text="Fix", command=self.fix)
        self.fix_btn.pack(side="right", padx=2)
        self.problem: Problem | None = None
        self._pending_export: tuple[list[str], str] | None = None

        cols = ("shortcut", "runs", "steps")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=7)
        for c, label, w in zip(cols, ("Shortcut", "Runs when", "Steps"), (220, 320, 60)):
            self.tree.heading(c, text=label)
            self.tree.column(c, width=w, anchor="w")
        self.tree.grid(row=3, column=0, sticky="nsew")
        f.rowconfigure(3, weight=1)
        btns = ttk.Frame(f)
        btns.grid(row=4, column=0, sticky="w", pady=6)
        for text, cmd in (
            ("Run now", self.run),
            ("Change gesture or phrase", self.change_trigger),
            ("Rename", self.rename),
            ("Delete", self.delete),
            ("Edit steps", self.edit_steps),
        ):
            ttk.Button(btns, text=text, command=cmd).pack(side="left", padx=2)
        more = ttk.Frame(f)
        more.grid(row=5, column=0, sticky="w")
        ttk.Button(more, text="Import...", command=self.import_).pack(side="left", padx=2)
        ttk.Button(more, text="Export...", command=self.export).pack(side="left", padx=2)
        self.refresh()
        self.show(self.m.last_problem())

    # ---------------------------------------------------------------- helpers
    def selected(self) -> str | None:
        sel = self.tree.selection()
        return str(sel[0]) if sel else None

    def need_selection(self) -> str | None:
        name = self.selected()
        if name is None:
            self.show(Problem("Pick a shortcut in the list first."))
        return name

    def refresh(self) -> None:
        keep = self.selected()
        self.tree.delete(*self.tree.get_children())
        for r in self.m.rows():
            title = r.title + ("  (private: types text)" if r.sensitive else "")
            self.tree.insert("", "end", iid=r.name, values=(title, r.runs_when, r.steps))
        if keep and self.tree.exists(keep):
            self.tree.selection_set(keep)
        with contextlib.suppress(Exception):
            self.win.refresh()  # the Bindings tab shows macro:... too

    def show(self, p: Problem | None, info: str = "") -> None:
        self.problem = p
        if p is None and not info:
            self.banner.grid_remove()
            return
        self.banner_msg.set(p.message if p is not None else info)
        if p is not None and p.fix_label:
            self.fix_btn.configure(text=p.fix_label)
            self.fix_btn.pack(side="right", padx=2)
        else:
            self.fix_btn.pack_forget()
        self.banner.grid(row=2, column=0, sticky="we", pady=(0, 8))

    def dismiss(self) -> None:
        self.m.dismiss_problem()
        self.show(None)

    def fix(self) -> None:
        p = self.problem
        if p is None:
            return
        if p.fix in ("record", "record_again"):
            self.show(None)
            self.record()
            return
        if p.fix == "edit_steps":
            self.show(None)
            self.edit_steps(p.name)
            return
        if p.fix == "export_private" and self._pending_export is not None:
            names, path = self._pending_export
            self._pending_export = None
            res = self.m.export(names, path, include_private=True)
            self.show(res, "" if res else f"Exported to {path}. It contains text you typed: keep it private.")
            return
        res = self.m.apply_fix(p)
        self.show(res, "" if res else "Fixed. Run it again whenever you like.")
        self.refresh()

    # ---------------------------------------------------------------- record
    def record(self) -> None:
        tk = self.tk
        self.show(None)
        bar = tk.Toplevel(self.root)
        bar.overrideredirect(True)
        with contextlib.suppress(Exception):
            bar.attributes("-topmost", True)
        frm = self.ttk.Frame(bar, padding=10, relief="ridge")
        frm.pack(fill="both", expand=True)
        self.ttk.Label(frm, text="●  Recording — do your steps now", foreground="#c00").pack(side="left", padx=6)
        self.ttk.Button(frm, text="Stop", command=lambda: self._stop(bar)).pack(side="left", padx=6)
        bar.update_idletasks()
        w, h = bar.winfo_reqwidth(), bar.winfo_reqheight()
        x = (bar.winfo_screenwidth() - w) // 2
        bar.geometry(f"{w}x{h}+{x}+8")
        bar.update_idletasks()
        rect = (x, 8, x + w, 8 + h)
        self.root.iconify()
        err = self.m.start_recording([rect])
        if err is not None:
            bar.destroy()
            self.root.deiconify()
            self.show(err)

    def _stop(self, bar: Any) -> None:
        draft, err = self.m.stop_recording()
        bar.destroy()
        self.root.deiconify()
        self.root.lift()
        if err is not None or draft is None:
            self.show(err)
            return
        self.save_dialog(draft)

    def save_dialog(self, draft: Macro, rename: str | None = None) -> None:
        tk, ttk = self.tk, self.ttk
        dlg = tk.Toplevel(self.root)
        dlg.title("Save your shortcut" if rename is None else "Change gesture or phrase")
        dlg.transient(self.root)
        body = ttk.Frame(dlg, padding=14)
        body.pack(fill="both", expand=True)
        row = 0
        name_var = tk.StringVar(value=self.m.suggested_title())
        if rename is None:
            n = len(draft.steps)
            ttk.Label(body, text=f"Recorded {n} step{'s' if n != 1 else ''}. Name your shortcut:").grid(
                row=row, column=0, columnspan=2, sticky="w"
            )
            row += 1
            e = ttk.Entry(body, textvariable=name_var, width=36)
            e.grid(row=row, column=0, columnspan=2, sticky="we", pady=(2, 10))
            e.select_range(0, "end")
            e.focus_set()
            row += 1
            note = self.m.recording_note(draft)
            if note:
                ttk.Label(body, text=note, foreground="#555", wraplength=380, justify="left").grid(
                    row=row, column=0, columnspan=2, sticky="w", pady=(0, 8)
                )
                row += 1
        ttk.Label(body, text="Run it when I…").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        kind = tk.StringVar(value="phrase")
        phrase = tk.StringVar()
        gesture = tk.StringVar()
        choices = self.m.gesture_choices()
        labels = [lbl for _g, lbl in choices]
        ttk.Radiobutton(body, text="say", variable=kind, value="phrase").grid(row=row, column=0, sticky="w")
        ttk.Entry(body, textvariable=phrase, width=28).grid(row=row, column=1, sticky="we")
        row += 1
        ttk.Radiobutton(body, text="make a gesture", variable=kind, value="gesture").grid(row=row, column=0, sticky="w")
        ttk.Combobox(body, textvariable=gesture, values=labels, state="readonly", width=26).grid(
            row=row, column=1, sticky="we"
        )
        row += 1
        ttk.Radiobutton(body, text="not now", variable=kind, value="none").grid(row=row, column=0, sticky="w")
        row += 1
        msg = tk.StringVar()
        ttk.Label(body, textvariable=msg, foreground="#b00", wraplength=380, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        row += 1
        replace = {"on": False}

        def value() -> str:
            if kind.get() == "phrase":
                return phrase.get()
            return next((g for g, lbl in choices if lbl == gesture.get()), "")

        def save() -> None:
            if rename is None:
                p = self.m.save_new(name_var.get(), kind.get(), value(), replace=replace["on"])
            elif kind.get() == "none":
                p = self.m.unbind(rename)
            else:
                p = self.m.bind(rename, kind.get(), value(), replace=replace["on"])
            if p is not None and p.fix == "replace_binding":
                msg.set(p.message + " Press Save again to replace it.")
                replace["on"] = True
                return
            if p is not None:
                msg.set(p.message)
                return
            dlg.destroy()
            self.refresh()
            self.show(None, "Saved. Your shortcut is ready." if rename is None else "Saved.")

        bar = ttk.Frame(body)
        bar.grid(row=row, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(bar, text="Save", command=save).pack(side="right", padx=2)
        ttk.Button(bar, text="Cancel", command=dlg.destroy).pack(side="right", padx=2)
        dlg.bind("<Return>", lambda _e: save())

    # ---------------------------------------------------------------- list actions
    def run(self) -> None:
        name = self.need_selection()
        if name is None:
            return
        self.root.iconify()
        self.root.after(400, lambda: self._run(name))

    def _run(self, name: str) -> None:
        p = self.m.run(name)
        self.root.deiconify()
        self.show(p, "" if p else "Done.")

    def change_trigger(self) -> None:
        name = self.need_selection()
        if name is None:
            return
        with contextlib.suppress(MacroError):
            self.save_dialog(self.m.store.load(name), rename=name)

    def rename(self) -> None:
        from tkinter import simpledialog

        name = self.need_selection()
        if name is None:
            return
        new = simpledialog.askstring(
            "Rename shortcut", "New name:", initialvalue=title_from_name(name), parent=self.root
        )
        if new:
            self.show(self.m.rename(name, new))
            self.refresh()

    def delete(self) -> None:
        from tkinter import messagebox

        name = self.need_selection()
        if name is None:
            return
        if messagebox.askyesno("Delete shortcut", f'Delete "{title_from_name(name)}"?', parent=self.root):
            self.show(self.m.delete(name))
            self.refresh()

    def import_(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("Shortcuts", "*.json")], parent=self.root)
        if path:
            got, p = self.m.import_file(path)
            self.show(
                p,
                ""
                if p
                else f"Imported {len(got)} shortcut{'s' if len(got) != 1 else ''}. "
                "Pick a gesture or phrase for each with “Change gesture or phrase”.",
            )
            self.refresh()

    def export(self) -> None:
        from tkinter import filedialog

        sel = list(self.tree.selection()) or [r.name for r in self.m.rows()]
        if not sel:
            self.show(Problem("There are no shortcuts to export yet.", "record"))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("Shortcuts", "*.json")], parent=self.root
        )
        if not path:
            return
        p = self.m.export(sel, path)
        if p is not None and p.fix == "export_private":
            self._pending_export = (sel, path)
        self.show(p, "" if p else f"Exported to {Path(path).name}.")

    # ---------------------------------------------------------------- step editor
    def edit_steps(self, name: str | None = None) -> None:
        tk, ttk = self.tk, self.ttk
        name = name or self.need_selection()
        if not name:
            return
        try:
            ed = StepEditor(self.m.store, name)
        except MacroError as exc:
            self.show(Problem(str(exc), "record_again", name))
            return
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Edit steps: {ed.macro.label}")
        dlg.transient(self.root)
        body = ttk.Frame(dlg, padding=10)
        body.pack(fill="both", expand=True)
        lb = tk.Listbox(body, width=60, height=14, exportselection=False)
        lb.grid(row=0, column=0, rowspan=8, sticky="nsew")
        msg = tk.StringVar()

        def fill(sel: int | None = None) -> None:
            lb.delete(0, "end")
            for line in ed.lines():
                lb.insert("end", line)
            if sel is not None and 0 <= sel < lb.size():
                lb.selection_set(sel)
                lb.see(sel)

        def cur() -> int:
            s = lb.curselection()
            return int(s[0]) if s else -1

        def report(p: Problem | None, sel: int | None = None) -> None:
            msg.set(p.message if p else "")
            fill(cur() if sel is None else sel)

        delay = tk.StringVar(value="0")
        lb.bind("<<ListboxSelect>>", lambda _e: delay.set(f"{ed.steps[cur()].delay:g}" if cur() >= 0 else "0"))
        col = ttk.Frame(body, padding=(8, 0))
        col.grid(row=0, column=1, sticky="n")
        ttk.Button(col, text="Move up", command=lambda: report(None, ed.move(cur(), -1))).pack(fill="x", pady=1)
        ttk.Button(col, text="Move down", command=lambda: report(None, ed.move(cur(), 1))).pack(fill="x", pady=1)
        ttk.Button(col, text="Delete step", command=lambda: report(ed.delete(cur()))).pack(fill="x", pady=1)
        ttk.Label(col, text="Pause before (s):").pack(anchor="w", pady=(8, 0))
        ttk.Entry(col, textvariable=delay, width=8).pack(anchor="w")
        ttk.Button(col, text="Set pause", command=lambda: report(ed.set_delay(cur(), delay.get()))).pack(fill="x")

        def ask(kind: str) -> None:
            from tkinter import simpledialog

            prompt = {"text": "Text to type:", "key": "Key to press (enter, tab, ctrl+s):", "wait": "Wait (seconds):"}
            v = simpledialog.askstring("Add a step", prompt[kind], parent=dlg)
            if v is None:
                return
            at = cur() + 1 if cur() >= 0 else -1
            fn = {"text": ed.insert_text, "key": ed.insert_key, "wait": ed.insert_wait}[kind]
            report(fn(at, v), at if at >= 0 else len(ed.steps) - 1)

        ttk.Label(col, text="Add after the selected step:").pack(anchor="w", pady=(8, 0))
        ttk.Button(col, text="Type text...", command=lambda: ask("text")).pack(fill="x", pady=1)
        ttk.Button(col, text="Press a key...", command=lambda: ask("key")).pack(fill="x", pady=1)
        ttk.Button(col, text="Wait...", command=lambda: ask("wait")).pack(fill="x", pady=1)
        speed = tk.StringVar(value=ed.macro.speed)
        sp = ttk.Frame(body)
        sp.grid(row=8, column=0, sticky="w", pady=(8, 0))
        ttk.Label(sp, text="Speed:").pack(side="left")
        for s, label in (("1x", "Normal"), ("2x", "Fast"), ("instant", "Instant")):
            ttk.Radiobutton(sp, text=label, value=s, variable=speed, command=lambda: ed.set_speed(speed.get())).pack(
                side="left", padx=4
            )
        anyapp = tk.BooleanVar(value=ed.macro.allow_other_app)

        def set_any() -> None:
            ed.macro.allow_other_app = bool(anyapp.get())

        ttk.Checkbutton(body, text="Allow it to click and type in any app", variable=anyapp, command=set_any).grid(
            row=9, column=0, sticky="w"
        )
        ttk.Label(body, textvariable=msg, foreground="#b00", wraplength=420).grid(row=10, column=0, sticky="w")

        def save() -> None:
            p = ed.save()
            if p is not None:
                msg.set(p.message)
                return
            dlg.destroy()
            self.refresh()
            self.show(None, "Steps saved.")

        bar = ttk.Frame(body)
        bar.grid(row=11, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bar, text="Save", command=save).pack(side="right", padx=2)
        ttk.Button(bar, text="Cancel", command=dlg.destroy).pack(side="right", padx=2)
        fill(0)
