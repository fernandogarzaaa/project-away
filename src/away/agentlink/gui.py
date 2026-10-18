"""Settings > Agent Link: the "Connect an AI agent" button and wizard, connected agents, what they changed, what
they did on the computer, preferences, a live health panel, and (collapsed) the technical details.

Kept out of ``settings_gui.py`` on purpose (it only calls :func:`build_tab`); all decisions live in
:class:`~away.agentlink.gui_model.AgentLinkModel`, which is tested without Tk.
"""

from __future__ import annotations

import contextlib
import threading
import time
import webbrowser
from typing import Any

from away.agentlink import perms
from away.agentlink.gui_model import AgentLinkModel, Problem

TILE_COLORS = {
    "claude_desktop": "#d97757",
    "claude_ai": "#d97757",
    "chatgpt": "#10a37f",
    "cursor": "#222222",
    "other": "#6e7781",
}
TILE_LETTER = {"claude_desktop": "C", "claude_ai": "C", "chatgpt": "G", "cursor": "\u25b2", "other": "\u2026"}


def _hhmmss(ts: Any) -> str:
    """v0.1.28: a short local timestamp for the action log rows."""
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts or 0)))
    except (TypeError, ValueError, OverflowError):
        return "--:--:--"


def build_tab(win: Any, nb: Any, model: AgentLinkModel | None = None) -> Any:  # pragma: no cover - widgets
    return AgentLinkTab(win, nb, model or AgentLinkModel())


class AgentLinkTab:  # pragma: no cover - widgets; smoke-tested under Xvfb
    def __init__(self, win: Any, nb: Any, model: AgentLinkModel) -> None:
        self.win, self.m = win, model
        tk, ttk = win.tk, win.ttk
        self.tk, self.ttk = tk, ttk
        f = win._scroll_tab(nb, "Agent Link")
        self.frame = f
        ttk.Label(
            f,
            text="The bridge between you and your AI agents, controlled by your hands, face and voice.",
            foreground="#555",
        ).pack(anchor="w")
        top = ttk.Frame(f)
        top.pack(fill="x", pady=(8, 4))
        self.status = tk.StringVar(value="...")
        self.status_lbl = tk.Label(top, textvariable=self.status, font=("TkDefaultFont", 13, "bold"))
        self.status_lbl.pack(side="left")
        ttk.Button(top, text="Disconnect all", command=self._disconnect_all).pack(side="right")
        # v0.1.22: a fresh web address for remote agents, without disconnecting anyone
        ttk.Button(top, text="New link", command=self._new_link).pack(side="right", padx=4)
        # v0.1.23: a link that does not use Cloudflare (when Cloudflare answers 403 to an agent's servers)
        ttk.Button(top, text="Blocked? Try the backup link", command=self._backup_link).pack(side="right", padx=4)
        lt = ttk.Frame(f)
        lt.pack(anchor="w")
        ttk.Label(lt, text="Link type:").pack(side="left")
        self.link_type = tk.StringVar(value=model.link_type())
        for val, text in (("standard", "Standard (Cloudflare)"), ("backup", "Backup")):
            ttk.Radiobutton(
                lt,
                text=text,
                value=val,
                variable=self.link_type,
                command=lambda: self.m.set_link_type(self.link_type.get()),
            ).pack(side="left", padx=4)
        big = tk.Button(
            f,
            text="  Connect an AI agent  ",
            font=("TkDefaultFont", 13, "bold"),
            bg="#1f6feb",
            fg="white",
            activebackground="#1158c7",
            relief="flat",
            padx=12,
            pady=8,
            command=self.open_wizard,
        )
        big.pack(anchor="w", pady=6)
        self.problem = ttk.Frame(f)
        self.problem.pack(fill="x")
        self.link_note = tk.StringVar(value="")  # v0.1.22: the result of "New link"
        ttk.Label(f, textvariable=self.link_note, wraplength=640, justify="left", foreground="#1a7f37").pack(anchor="w")

        self.prompt_box = ttk.LabelFrame(f, text="Waiting for your answer", padding=6)
        self.agents_box = ttk.LabelFrame(f, text="Your agents", padding=6)
        self.agents_box.pack(fill="x", pady=6)
        # v0.1.24: agents that stay connected (persistent agent tokens)
        self.tokens_box = ttk.LabelFrame(f, text="Agent tokens (agents that stay connected)", padding=6)
        self.tokens_box.pack(fill="x", pady=6)
        self.token_note = tk.StringVar(value="")
        # how long a new agent token works (agent_link.token_expiry): 90 days by default, 30 days, or until revoked
        te = ttk.Frame(f)
        te.pack(fill="x")
        ttk.Label(te, text="New agent tokens last:").pack(side="left")
        labels = dict(self.m.EXPIRY_CHOICES)
        self.token_expiry_var = tk.StringVar(value=labels.get(self.m.token_expiry(), labels["90d"]))
        te_box = ttk.Combobox(
            te, textvariable=self.token_expiry_var, values=list(labels.values()), state="readonly", width=18
        )
        te_box.pack(side="left", padx=6)
        te_box.bind("<<ComboboxSelected>>", lambda _e: self.m.set_token_expiry(self.token_expiry_var.get()))
        ttk.Label(te, text="(tokens already given keep theirs)", foreground="#555").pack(side="left")
        # v0.1.22 "Talk to your agent"
        tb = ttk.LabelFrame(f, text="Talk to your agent", padding=6)
        tb.pack(fill="x", pady=6)
        ttk.Label(
            tb,
            text="Say \u201cHey\u201d and an agent's name (\u201cHey Claude, ...\u201d) and your words go to that "
            "agent instead of being typed. \u201cHey agent\u201d or the summon gesture goes to:",
            wraplength=640,
            justify="left",
        ).pack(anchor="w")
        self.talk_on = tk.BooleanVar(value=model.talk_enabled())
        ttk.Checkbutton(
            tb,
            text="Listen for \u201cHey\u201d + an agent's name (off: those words are typed like any others)",
            variable=self.talk_on,
            command=lambda: self.m.set_talk_enabled(self.talk_on.get()),
        ).pack(anchor="w")
        drow = ttk.Frame(tb)
        drow.pack(fill="x", pady=2)
        self.default_var = tk.StringVar(value=model.default_agent() or "Every connected agent")
        self.default_cb = ttk.Combobox(drow, textvariable=self.default_var, state="readonly", width=28)
        self.default_cb.pack(side="left")
        self.default_cb.bind("<<ComboboxSelected>>", lambda _e: self._set_default())
        self.inbox_text = tk.StringVar(value="")  # v0.1.23: unread messages per agent
        ttk.Label(tb, textvariable=self.inbox_text, justify="left", foreground="#1f6feb").pack(anchor="w")
        self.control_box = ttk.LabelFrame(f, text="Using your computer", padding=6)
        self.control_box.pack(fill="x", pady=6)
        self.control_text = tk.StringVar(value="No agent is using your computer.")
        ttk.Label(self.control_box, textvariable=self.control_text).pack(anchor="w")
        row = ttk.Frame(self.control_box)
        row.pack(fill="x")
        ttk.Button(row, text="Stop now", command=lambda: self.m.stop_control()).pack(side="left")
        ttk.Button(row, text="Undo last", command=lambda: self.m.undo_action()).pack(side="left", padx=4)
        # v0.1.28: the step budget (fresh every approved session) and the visual audit trail switch
        brow = ttk.Frame(self.control_box)
        brow.pack(fill="x", pady=2)
        ttk.Label(brow, text="Step budget per session:").pack(side="left")
        self.budget_var = tk.IntVar(value=self.m.control_budget())
        self.budget_spin = tk.Spinbox(
            brow, from_=1, to=1000, width=6, textvariable=self.budget_var, command=self._set_budget
        )
        self.budget_spin.pack(side="left", padx=4)
        self.budget_spin.bind("<Return>", lambda _e: self._set_budget())
        self.budget_spin.bind("<FocusOut>", lambda _e: self._set_budget())
        ttk.Label(brow, text="steps (next session)").pack(side="left")
        self.trail_var = tk.BooleanVar(value=self.m.audit_trail_on())
        ttk.Checkbutton(
            brow,
            text="Keep a step-by-step record",
            variable=self.trail_var,
            command=lambda: self.m.set_audit_trail(self.trail_var.get()),
        ).pack(side="left", padx=8)
        self.actions = tk.Listbox(self.control_box, height=5)
        self.actions.pack(fill="x", pady=4)
        self.actions.bind("<<ListboxSelect>>", lambda _e: self._show_thumb())
        # v0.1.28: replay the session's audit trail step by step (timestamp, action, result, picture)
        rrow = ttk.Frame(self.control_box)
        rrow.pack(fill="x")
        ttk.Button(rrow, text="\u25c0", width=3, command=lambda: self._replay_step(-1)).pack(side="left")
        ttk.Button(rrow, text="\u25b6", width=3, command=lambda: self._replay_step(1)).pack(side="left", padx=2)
        self.replay_text = tk.StringVar(value="")
        ttk.Label(rrow, textvariable=self.replay_text, foreground="#555").pack(side="left", padx=4)
        self.thumb = ttk.Label(self.control_box)
        self.thumb.pack(anchor="w")
        self._thumb_img: Any = None
        self._action_rows: list[dict[str, Any]] = []
        self._trail_rows: list[dict[str, Any]] = []
        self._replay_idx: int | None = None
        self._replay_session: str = ""

        # v0.1.28 "Recently": who looked through your camera or at your screen (and what you sent)
        rf = ttk.LabelFrame(f, text="Recently", padding=6)
        rf.pack(fill="x", pady=6)
        self.recently = tk.Listbox(rf, height=4)
        self.recently.pack(fill="x")

        # v0.1.28 "Recently": who looked through your camera or at your screen (and what you sent)
        rf = ttk.LabelFrame(f, text="Recently", padding=6)
        rf.pack(fill="x", pady=6)
        self.recently = tk.Listbox(rf, height=4)
        self.recently.pack(fill="x")

        ch = ttk.LabelFrame(f, text="Changes your agents made", padding=6)
        ch.pack(fill="x", pady=6)
        self.autofix = tk.StringVar(value="ask")
        mrow = ttk.Frame(ch)
        mrow.pack(fill="x")
        ttk.Radiobutton(mrow, text="Ask me first", value="ask", variable=self.autofix, command=self._set_autofix).pack(
            side="left"
        )
        ttk.Radiobutton(
            mrow,
            text="Just do it and tell me (small adjustments only)",
            value="auto_low_risk",
            variable=self.autofix,
            command=self._set_autofix,
        ).pack(side="left", padx=8)
        self.changes = tk.Listbox(ch, height=5)
        self.changes.pack(fill="x", pady=4)
        crow = ttk.Frame(ch)
        crow.pack(fill="x")
        ttk.Button(crow, text="Undo", command=self._undo_change).pack(side="left")
        self._change_rows: list[dict[str, Any]] = []

        pf = ttk.LabelFrame(f, text="My preferences (your agents read these)", padding=6)
        pf.pack(fill="x", pady=6)
        self.prefs = tk.Listbox(pf, height=4)
        self.prefs.pack(fill="x")
        prow = ttk.Frame(pf)
        prow.pack(fill="x", pady=2)
        self.pref_entry = ttk.Entry(prow)
        self.pref_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(prow, text="Add", command=self._add_pref).pack(side="left", padx=4)
        ttk.Button(prow, text="Remove selected", command=self._remove_pref).pack(side="left")

        hf = ttk.LabelFrame(f, text="How it's running", padding=6)
        hf.pack(fill="x", pady=6)
        self.health = tk.StringVar(value="")
        ttk.Label(hf, textvariable=self.health, justify="left").pack(anchor="w")

        self.adv_open = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Advanced", variable=self.adv_open, command=self._toggle_adv).pack(anchor="w", pady=4)
        self.adv = ttk.Frame(f)
        self.adv_text = tk.Text(self.adv, height=14, wrap="word", relief="flat", background="#f6f6f6")
        self.adv_text.pack(fill="x")
        arow = ttk.Frame(self.adv)
        arow.pack(fill="x", pady=4)
        ttk.Button(arow, text="Copy", command=self._copy_adv).pack(side="left")
        ttk.Button(arow, text="Save a diagnostics bundle", command=self._bundle).pack(side="left", padx=4)
        self._tick()

    # ------------------------------------------------------------------ helpers
    def _bg(self, fn: Any, done: Any = None) -> None:
        def run() -> None:
            try:
                res = fn()
            except Exception as exc:
                res = exc
            if done is not None:
                self.win._ui.post(lambda: done(res))

        threading.Thread(target=run, daemon=True, name="agentlink-gui").start()

    def _show_problem(self, p: Problem | None) -> None:
        for w in self.problem.winfo_children():
            w.destroy()
        if p is None:
            return
        self.ttk.Label(self.problem, text=p.text, foreground="#cf222e").pack(side="left")
        if p.fix_label:
            self.ttk.Button(self.problem, text=p.fix_label, command=lambda: self._fix(p)).pack(side="left", padx=6)

    def _fix(self, p: Problem) -> None:
        if p.fix.startswith("url:"):
            webbrowser.open(p.fix[4:])
        elif p.fix == "start_app":
            import subprocess

            from away.app.frozen import away_command

            with contextlib.suppress(Exception):
                subprocess.Popen(away_command(["tray"]))

    def _tick(self) -> None:
        try:
            self.refresh()
        finally:
            self.frame.after(1500, self._tick)

    def refresh(self) -> None:
        st = self.m.state() if self.m.alive() else {}
        text, ok = self.m.status_line(st)
        self.status.set(text)
        self.status_lbl.configure(fg="#1a7f37" if ok else "#333")
        self._show_problem(self.m.not_running())
        self._render_prompts(st.get("prompts") or [])
        self._render_agents(st.get("agents") or [])
        self._render_tokens(st)
        self.inbox_text.set("\n".join(self.m.inbox_lines(st)))
        ctl = st.get("control") or {}
        if ctl.get("active"):
            mins = int(ctl.get("expires_in_s", 0)) // 60
            # v0.1.28: the session's step budget, so the person sees how far the agent is through it
            steps = f" \u2014 step {ctl.get('steps_used', 0)} of {ctl.get('step_budget', 40)}"
            self.control_text.set(
                f"{ctl['agent']} is using your computer"
                + (" (paused: you took over)" if ctl.get("paused") else f" for {mins} more min")
                + steps
            )
        else:
            self.control_text.set("No agent is using your computer.")
        self._action_rows = list(st.get("actions") or [])
        self._fill(
            self.actions,
            [
                f"{_hhmmss(a.get('ts'))}  {a['agent']}: {a['action']} {a['detail']}"
                + ("" if a["result"] == "done" else f"  ({a['result']})")
                for a in self._action_rows
            ],
        )
        # v0.1.28: the session's audit trail for replay; a new session resets the replay position
        trail = list(st.get("control_trail") or [])
        session = str(trail[-1].get("session", "")) if trail else ""
        if session != self._replay_session:
            self._replay_session = session
            self._replay_idx = None
            self.replay_text.set("")
        self._trail_rows = trail
        self._change_rows = list(st.get("changes") or [])
        self._fill(
            self.changes,
            [
                f"{c['who']}: {c['text']}"
                + ("  (undone)" if c["undone"] else "")
                + (f"  - {c['why']}" if c.get("why") and not c["who"].startswith("undo") else "")
                for c in self._change_rows
            ],
        )
        if st.get("autofix") and self.autofix.get() != st["autofix"]:
            self.autofix.set(st["autofix"])
        self._fill(self.prefs, self._prefs_items())
        self._fill(self.recently, self.m.capture_lines(st))
        self.m.want_health()
        self.health.set("\n".join(self.m.health_lines(st.get("health"))))
        if self.adv_open.get():
            self._render_adv(st)

    @staticmethod
    def _fill(lb: Any, rows: list[str]) -> None:
        if list(lb.get(0, "end")) == rows:
            return
        lb.delete(0, "end")
        for r in rows:
            lb.insert("end", r)

    def _render_prompts(self, prompts: list[dict[str, Any]]) -> None:
        box = self.prompt_box
        for w in box.winfo_children():
            w.destroy()
        if not prompts:
            box.pack_forget()
            return
        box.pack(fill="x", pady=6, before=self.agents_box)
        for p in prompts[:2]:
            self.ttk.Label(box, text=p["question"], font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
            row = self.ttk.Frame(box)
            row.pack(anchor="w")
            for opt in p["options"]:
                self.ttk.Button(row, text=opt, command=lambda i=p["id"], o=opt: self.m.answer(i, o)).pack(
                    side="left", padx=2
                )

    def _render_agents(self, agents: list[dict[str, Any]]) -> None:
        box = self.agents_box
        key = [
            (a["id"], a["connected"], tuple(a["scopes"]), a.get("wake_phrase"), a.get("wake_me"), a.get("unread"))
            for a in agents
        ]
        if getattr(self, "_agents_key", None) == key:
            return
        self._agents_key = key
        for w in box.winfo_children():
            w.destroy()
        if not agents:
            self.ttk.Label(box, text="None yet. Press \u201cConnect an AI agent\u201d.").pack(anchor="w")
            return
        for a in agents:
            row = self.ttk.Frame(box)
            row.pack(fill="x", pady=2)
            mark = "\u2713" if a["connected"] else "\u2022"
            self.ttk.Label(row, text=f"{mark} {a['name']}", font=("TkDefaultFont", 11, "bold")).pack(side="left")
            if a.get("unread"):
                self.ttk.Label(row, text=f"  {a['unread']} unread", foreground="#cf222e").pack(side="left")
            if a.get("wake_phrase"):
                self.ttk.Label(row, text=f"  say \u201c{a['wake_phrase']}\u201d", foreground="#1f6feb").pack(
                    side="left"
                )
            self.ttk.Label(row, text="  " + ", ".join(a["permissions"]), foreground="#555").pack(side="left")
            self.ttk.Button(row, text="Disconnect", command=lambda i=a["id"]: self.m.revoke(i)).pack(side="right")
            self.ttk.Button(
                row, text="Undo its changes today", command=lambda i=a["id"]: self.m.undo_agent_today(i)
            ).pack(side="right", padx=2)
            self.ttk.Button(row, text="Permissions", command=lambda ag=a: self._edit_perms(ag)).pack(side="right")
            self.ttk.Button(row, text="Wake word", command=lambda ag=a: self._edit_talk(ag)).pack(side="right", padx=2)
        from away.agentlink.gui_model import AgentLinkModel

        self.default_cb["values"] = AgentLinkModel.default_choices(agents)

    def _render_tokens(self, st: dict[str, Any]) -> None:
        """v0.1.24 "Agent tokens": who stays connected, what it can do, when it was last used and from where."""
        rows = self.m.token_rows(st)
        key = [(r["id"], r["can"], r["when"], r["ends"], r["old_key"], r["connected"]) for r in rows]
        key.append(("note", self.token_note.get(), "", "", "", False))
        if getattr(self, "_tokens_key", None) == key:
            return
        self._tokens_key = key
        box = self.tokens_box
        for w in box.winfo_children():
            w.destroy()
        if self.token_note.get():
            self.ttk.Label(box, textvariable=self.token_note, foreground="#1a7f37", wraplength=640).pack(anchor="w")
        if not rows:
            self.ttk.Label(
                box,
                text="None. When you allow an agent, choose \u201cKeep this agent connected\u201d and it comes back "
                "by itself after restarts, without asking again.",
                wraplength=640,
                justify="left",
                foreground="#555",
            ).pack(anchor="w")
            return
        for r in rows:
            row = self.ttk.Frame(box)
            row.pack(fill="x", pady=3)
            head = self.ttk.Frame(row)
            head.pack(fill="x")
            mark = "\u2713" if r["connected"] else "\u2022"
            self.ttk.Label(head, text=f"{mark} {r['name']}", font=("TkDefaultFont", 11, "bold")).pack(side="left")
            self.ttk.Button(head, text="Revoke", command=lambda x=r: self._revoke_token(x)).pack(side="right")
            self.ttk.Button(head, text="New token", command=lambda x=r: self._rotate_token(x)).pack(
                side="right", padx=2
            )
            self.ttk.Button(head, text="What it can do", command=lambda x=r: self._edit_token_perms(x)).pack(
                side="right"
            )
            for line in (r["can"], r["when"], r["ends"], r["old_key"]):
                if line:
                    self.ttk.Label(row, text="   " + line, foreground="#555").pack(anchor="w")

    def _revoke_token(self, r: dict[str, Any]) -> None:
        from tkinter import messagebox

        if not messagebox.askyesno(
            "Revoke?",
            f"{r['name']} will be disconnected right away and can only come back if you allow it again.",
            parent=self.frame,
        ):
            return
        self._bg(lambda: self.m.revoke_token(r["id"]), lambda msg: self.token_note.set(str(msg)))

    def _rotate_token(self, r: dict[str, Any]) -> None:
        def done(res: Any) -> None:
            if isinstance(res, Exception):
                self.token_note.set(f"Something went wrong: {res}")
                return
            problem, secret = res
            if problem is not None or not secret:
                self.token_note.set(problem.text if problem is not None else "Something went wrong.")
                return
            self._show_token_once(r["name"], secret)

        self._bg(lambda: self.m.rotate_token(r["id"]), done)

    def _show_token_once(self, name: str, secret: str) -> None:
        tk, ttk = self.tk, self.ttk
        top = tk.Toplevel(self.frame)
        top.title(f"New token for {name}")
        ttk.Label(
            top,
            text=f"Give this new token to {name}. It is shown only now; after you close this window nobody can see "
            "it again. The old token keeps working for one hour, then stops.",
            wraplength=480,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(12, 6))
        var = tk.StringVar(value=secret)
        ent = ttk.Entry(top, textvariable=var, width=60)
        ent.state(["readonly"])
        ent.pack(padx=12, pady=4)

        def copy() -> None:
            top.clipboard_clear()
            top.clipboard_append(secret)
            copied.set("Copied")

        copied = tk.StringVar(value="")
        row = ttk.Frame(top)
        row.pack(pady=8)
        ttk.Button(row, text="Copy", command=copy).pack(side="left")
        ttk.Label(row, textvariable=copied, foreground="#1a7f37").pack(side="left", padx=6)
        ttk.Button(row, text="Done", command=top.destroy).pack(side="left", padx=6)

    def _edit_token_perms(self, r: dict[str, Any]) -> None:
        tk, ttk = self.tk, self.ttk
        top = tk.Toplevel(self.frame)
        top.title(f"What {r['name']} can do")
        vars_: dict[str, Any] = {}
        for p in perms.PERMISSIONS:
            v = tk.BooleanVar(value=p.scope in r["scopes"])
            vars_[p.scope] = v
            cb = ttk.Checkbutton(top, text=f"{p.icon}  {p.label}", variable=v)
            if p.locked:
                cb.state(["disabled"])
            cb.pack(anchor="w", padx=12, pady=3)

        def save() -> None:
            self.m.set_token_scopes(r["id"], [s for s, v in vars_.items() if v.get()])
            top.destroy()

        ttk.Button(top, text="Save", command=save).pack(pady=8)

    def _edit_perms(self, agent: dict[str, Any]) -> None:
        tk, ttk = self.tk, self.ttk
        top = tk.Toplevel(self.frame)
        top.title(f"What {agent['name']} can do")
        vars_: dict[str, Any] = {}
        for p in perms.PERMISSIONS:
            v = tk.BooleanVar(value=p.scope in agent["scopes"])
            vars_[p.scope] = v
            cb = ttk.Checkbutton(top, text=f"{p.icon}  {p.label}", variable=v)
            if p.locked:
                cb.state(["disabled"])
            cb.pack(anchor="w", padx=12, pady=3)

        def save() -> None:
            self.m.set_scopes(agent["id"], [s for s, v in vars_.items() if v.get()])
            top.destroy()

        ttk.Button(top, text="Save", command=save).pack(pady=8)

    def _edit_talk(self, agent: dict[str, Any]) -> None:
        """v0.1.22: this agent's own wake word and the optional "wake me" address."""
        tk, ttk = self.tk, self.ttk
        top = tk.Toplevel(self.frame)
        top.title(f"Talking to {agent['name']}")
        ttk.Label(top, text=f"Wake word (empty = \u201cHey {agent['name']}\u201d):").pack(
            anchor="w", padx=12, pady=(10, 2)
        )
        word = tk.StringVar(value=str(agent.get("wake_word") or ""))
        ttk.Entry(top, textvariable=word, width=36).pack(anchor="w", padx=12)
        ttk.Label(
            top,
            text="Wake-up address (optional). If your agent gave you one, paste it here and it is told right away "
            "when you talk to it. Leave empty if you don't know.",
            wraplength=380,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(10, 2))
        url = tk.StringVar(value="")
        ttk.Entry(top, textvariable=url, width=48).pack(anchor="w", padx=12)
        on = "on" if agent.get("wake_me") else "off"
        ttk.Label(top, text=f"Wake-up is {on} now (leave the box empty to keep it).", foreground="#555").pack(
            anchor="w", padx=12
        )
        off = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Turn wake-up off", variable=off).pack(anchor="w", padx=12)
        msg = tk.StringVar(value="")
        ttk.Label(top, textvariable=msg, foreground="#cf222e").pack(anchor="w", padx=12)

        def save() -> None:
            u = "" if off.get() else (url.get().strip() or None)
            err = self.m.set_talk(agent["id"], word.get().strip(), u)
            if err:
                msg.set(err)
                return
            top.destroy()

        ttk.Button(top, text="Save", command=save).pack(pady=8)

    def _set_default(self) -> None:
        from away.agentlink.gui_model import EVERYONE

        v = self.default_var.get()
        self.m.set_default_agent("" if v == EVERYONE else v)

    def _backup_link(self) -> None:
        self.link_note.set("Getting a backup link (not Cloudflare)...")

        def done(res: Any) -> None:
            if isinstance(res, Exception):
                self.link_note.set(str(res))
                return
            problem, info = res
            if problem is not None:
                self.link_note.set(problem.text)
                return
            with contextlib.suppress(Exception):
                self.win.root.clipboard_clear()
                self.win.root.clipboard_append(str(info.get("url", "")))
            self.link_type.set("backup")
            with contextlib.suppress(Exception):
                self.m.set_link_type("backup")
            self.link_note.set(f"Backup link (copied): {info.get('url')}  {info.get('note', '')}")

        self._bg(self.m.backup_link, done)

    def _new_link(self) -> None:
        self.link_note.set("Getting a new link...")

        def done(res: Any) -> None:
            if isinstance(res, Exception):
                self.link_note.set(str(res))
                return
            problem, info = res
            if problem is not None:
                self.link_note.set(problem.text)
                return
            with contextlib.suppress(Exception):
                self.win.root.clipboard_clear()
                self.win.root.clipboard_append(str(info.get("url", "")))
            self.link_note.set(
                f"New link (copied): {info.get('url')}  Paste it into your AI app. {info.get('note', '')}"
            )

        self._bg(self.m.new_link, done)

    def _prefs_items(self) -> list[str]:
        from away import paths
        from away.agentlink.host import base_dir
        from away.agentlink.prefs import Preferences

        return Preferences(base_dir(self.m.data_dir or paths.data_dir()) / "preferences.md").items()

    def _add_pref(self) -> None:
        from away import paths
        from away.agentlink.host import base_dir
        from away.agentlink.prefs import Preferences

        text = self.pref_entry.get().strip()
        if text:
            Preferences(base_dir(self.m.data_dir or paths.data_dir()) / "preferences.md").add(text)
            self.pref_entry.delete(0, "end")

    def _remove_pref(self) -> None:
        from away import paths
        from away.agentlink.host import base_dir
        from away.agentlink.prefs import Preferences

        sel = self.prefs.curselection()
        if sel:
            Preferences(base_dir(self.m.data_dir or paths.data_dir()) / "preferences.md").remove(int(sel[0]) + 1)

    def _undo_change(self) -> None:
        sel = self.changes.curselection()
        if sel and int(sel[0]) < len(self._change_rows):
            self.m.undo_change(self._change_rows[int(sel[0])]["batch"])

    def _set_autofix(self) -> None:
        self.m.set_autofix(self.autofix.get())

    def _show_thumb(self) -> None:
        sel = self.actions.curselection()
        if not sel or int(sel[0]) >= len(self._action_rows):
            return
        row = self._action_rows[int(sel[0])]
        self._replay_idx = None  # picking a row leaves the replay walk
        self.replay_text.set("")
        self._show_thumb_file(row.get("thumb"))

    def _show_thumb_file(self, name: Any) -> None:
        if not name:
            self.thumb.configure(image="", text="(no picture)")
            return
        from away import paths
        from away.agentlink.host import base_dir

        p = base_dir(self.m.data_dir or paths.data_dir()) / "actions" / name
        with contextlib.suppress(Exception):
            self._thumb_img = self.tk.PhotoImage(file=str(p))
            self.thumb.configure(image=self._thumb_img, text="")

    def _set_budget(self) -> None:
        """v0.1.28: the step budget (1-1000) is saved to the config and applies to the next approved session."""
        try:
            saved = self.m.set_control_budget(int(self.budget_var.get()))
        except Exception:  # not a number: the spinbox raises TclError, int() raises ValueError
            saved = self.m.control_budget()
        self.budget_var.set(saved)

    def _replay_step(self, direction: int) -> None:
        """v0.1.28: walk the session's audit trail one step at a time (timestamp, action, result, picture)."""
        rows = self._trail_rows
        if not rows:
            self.replay_text.set("No steps recorded in this session yet.")
            return
        if self._replay_idx is None:
            self._replay_idx = len(rows) - 1 if direction < 0 else 0
        else:
            self._replay_idx = max(0, min(len(rows) - 1, self._replay_idx + direction))
        row = rows[self._replay_idx]
        self.replay_text.set(
            f"Step {self._replay_idx + 1} of {len(rows)}: {row['action']} {row.get('detail', '')}"
            f" \u2014 {row.get('result', '')} ({_hhmmss(row.get('ts'))})"
        )
        self._show_thumb_file(row.get("thumb"))

    def _disconnect_all(self) -> None:
        self._bg(self.m.disconnect_all, lambda msg: self.win._report(None, str(msg)))

    def _bundle(self) -> None:
        self._bg(self.m.bundle, lambda msg: self.win._report(None, str(msg)))

    def _toggle_adv(self) -> None:
        if self.adv_open.get():
            self.adv.pack(fill="x")
        else:
            self.adv.pack_forget()

    def _render_adv(self, st: dict[str, Any]) -> None:
        from away.agentlink import clients

        lines = [
            "How agents connect (technical):",
            "- Apps on this PC run `away mcp` (MCP over stdio); the first time you approve them here.",
            "- Remote apps use a private https link (a Cloudflare quick tunnel, or the backup SSH relay, to "
            "127.0.0.1) that works once,",
            "  for 10 minutes, and only after you press Allow. Remote access ends after 1 hour.",
            f"- Local server: 127.0.0.1:{(st.get('server') or {}).get('port') or '-'}; private link: "
            f"{(st.get('link') or {}).get('url') or 'off'}",
            "- REST API + OpenAPI at <link>/openapi.json; MCP at <link>/mcp with a bearer token.",
            "",
        ]
        p = st.get("pairing") or {}
        for title, text in clients.snippets(perms.to_list(perms.DEFAULT), p.get("url")).items():
            lines += [f"[{title}]", text, ""]
        lines.append("Recent requests:")
        lines += [f"  {a['method']} {a['path']} -> {a['status']} ({a['agent'] or '-'})" for a in st.get("access") or []]
        text = "\n".join(lines)
        if self.adv_text.get("1.0", "end").strip() != text.strip():
            self.adv_text.delete("1.0", "end")
            self.adv_text.insert("end", text)

    def _copy_adv(self) -> None:
        self.frame.clipboard_clear()
        self.frame.clipboard_append(self.adv_text.get("1.0", "end"))

    # ------------------------------------------------------------------ wizard
    def open_wizard(self) -> None:
        Wizard(self)


class Wizard:  # pragma: no cover - widgets
    def __init__(self, tab: AgentLinkTab) -> None:
        self.tab, self.m = tab, tab.m
        tk = tab.tk
        self.top = tk.Toplevel(tab.frame)
        self.top.title("Connect an AI agent")
        self.top.geometry("620x560")
        self.body = tab.ttk.Frame(self.top, padding=14)
        self.body.pack(fill="both", expand=True)
        self.choice: str = ""
        self.scopes = set(self.m.default_scopes())
        self.known = {a["id"] for a in self.m.agents()} if self.m.alive() else set()
        self.step1()

    def _clear(self, title: str, step: int) -> Any:
        for w in self.body.winfo_children():
            w.destroy()
        ttk = self.tab.ttk
        ttk.Label(self.body, text=f"Step {step} of 3", foreground="#777").pack(anchor="w")
        ttk.Label(self.body, text=title, font=("TkDefaultFont", 15, "bold")).pack(anchor="w", pady=(2, 10))
        return self.body

    def step1(self) -> None:
        tk = self.tab.tk
        b = self._clear("Which AI do you use?", 1)
        grid = self.tab.ttk.Frame(b)
        grid.pack()
        for i, t in enumerate(self.m.tiles()):
            cell = tk.Frame(grid, padx=8, pady=8)
            cell.grid(row=i // 3, column=i % 3)
            c = tk.Canvas(cell, width=84, height=84, highlightthickness=0)
            c.create_oval(4, 4, 80, 80, fill=TILE_COLORS[t["key"]], outline="")
            c.create_text(42, 42, text=TILE_LETTER[t["key"]], fill="white", font=("TkDefaultFont", 26, "bold"))
            c.pack()
            c.bind("<Button-1>", lambda _e, k=t["key"]: self.pick(k))
            label = t["name"] + ("\n(found on this PC)" if t.get("installed") else "")
            tk.Button(cell, text=label, relief="flat", command=lambda k=t["key"]: self.pick(k)).pack()

    def pick(self, key: str) -> None:
        self.choice = key
        self.step2()

    def step2(self) -> None:
        tk, ttk = self.tab.tk, self.tab.ttk
        b = self._clear("What can it do?", 2)
        vars_: dict[str, Any] = {}
        for p in self.m.permission_rows():
            v = tk.BooleanVar(value=p.scope in self.scopes)
            vars_[p.scope] = v
            row = ttk.Frame(b)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=p.icon, font=("TkDefaultFont", 18), width=2).pack(side="left")
            cb = ttk.Checkbutton(row, text=p.label, variable=v)
            if p.locked:
                cb.state(["disabled"])
            cb.pack(side="left")
            ttk.Label(row, text=p.help, foreground="#777").pack(side="left", padx=8)
        ttk.Label(
            b, text="You can change this any time. Nothing is shared until you press Allow.", foreground="#555"
        ).pack(anchor="w", pady=10)

        def nxt() -> None:
            self.scopes = {s for s, v in vars_.items() if v.get()} | {perms.READ}
            self.scopes |= {p.scope for p in self.m.permission_rows(advanced=True) if p.default}
            self.step3()

        nav = ttk.Frame(b)
        nav.pack(fill="x", side="bottom")
        ttk.Button(nav, text="Back", command=self.step1).pack(side="left")
        ttk.Button(nav, text="Next", command=nxt).pack(side="right")

    def step3(self) -> None:
        from away.agentlink import clients

        ttk = self.tab.ttk
        app = clients.APPS[self.choice]
        b = self._clear(f"Connect {app.name}", 3)
        self.msg = self.tab.tk.StringVar(value="")
        if app.local:
            problem, preview, _path = self.m.local_plan(self.choice, sorted(self.scopes))
            if problem is not None:
                self._problem(b, problem)
                return
            ttk.Label(b, text=preview, wraplength=560).pack(anchor="w")
            ttk.Button(b, text="Connect", command=lambda: self._do_local(b)).pack(anchor="w", pady=10)
            ttk.Label(b, textvariable=self.msg, wraplength=560, foreground="#1a7f37").pack(anchor="w")
        else:
            self.msg.set("Making a private link...")
            ttk.Label(b, textvariable=self.msg, wraplength=560).pack(anchor="w")
            self.tab._bg(lambda: self.m.start_remote(sorted(self.scopes)), lambda res: self._show_link(b, res))
        self._wait_connected()

    def _problem(self, b: Any, p: Problem) -> None:
        ttk = self.tab.ttk
        ttk.Label(b, text=p.text, foreground="#cf222e", font=("TkDefaultFont", 12)).pack(anchor="w")
        if p.fix_label:
            ttk.Button(
                b, text=p.fix_label, command=lambda: self.step3() if p.fix == "retry" else self.tab._fix(p)
            ).pack(anchor="w", pady=6)

    def _do_local(self, b: Any) -> None:
        problem, text = self.m.connect_local(self.choice, sorted(self.scopes))
        if problem is not None:
            self._problem(b, problem)
        else:
            self.msg.set(text)

    def _show_link(self, b: Any, res: Any) -> None:
        from away.agentlink import qr

        if isinstance(res, Exception):
            self._problem(b, Problem(str(res), "Try again", "retry"))
            return
        problem, p = res
        if problem is not None:
            self.msg.set("")
            self._problem(b, problem)
            return
        tk, ttk = self.tab.tk, self.tab.ttk
        self.msg.set("Your private link (works once, for 10 minutes):")
        row = ttk.Frame(b)
        row.pack(fill="x", pady=4)
        e = ttk.Entry(row, width=56)
        e.insert(0, p["url"])
        e.pack(side="left")

        def copy() -> None:
            self.top.clipboard_clear()
            self.top.clipboard_append(p["url"])

        ttk.Button(row, text="Copy", command=copy).pack(side="left", padx=4)
        mid = ttk.Frame(b)
        mid.pack(fill="x")
        m = qr.matrix(p["url"])
        if m is not None:
            n = len(m)
            px = max(2, 180 // n)
            c = tk.Canvas(mid, width=n * px, height=n * px, bg="white", highlightthickness=0)
            for y, r in enumerate(m):
                for x, on in enumerate(r):
                    if on:
                        c.create_rectangle(x * px, y * px, x * px + px, y * px + px, fill="black", outline="")
            c.pack(side="left", padx=(0, 12))
        info = ttk.Frame(mid)
        info.pack(side="left", fill="both")
        ttk.Label(info, text="Code", foreground="#777").pack(anchor="w")
        ttk.Label(info, text=f"{p['pin'][:3]} {p['pin'][3:]}", font=("TkDefaultFont", 22, "bold")).pack(anchor="w")
        ttk.Label(info, text="When your AI connects, you'll be asked to Allow it here.", wraplength=300).pack(
            anchor="w"
        )
        for name, line in self.m.remote_steps(self.choice):
            ttk.Label(b, text=f"{name}: {line}", wraplength=580, justify="left").pack(anchor="w", pady=2)

    def _wait_connected(self) -> None:
        if not self.top.winfo_exists():
            return
        new = [a for a in (self.m.agents() if self.m.alive() else []) if a["id"] not in self.known and a["connected"]]
        if new:
            self.msg.set(f"Connected to {new[0]['name']} \u2713  You can close this window.")
            return
        self.top.after(1500, self._wait_connected)
