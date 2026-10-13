"""Settings > Live agent.

Plain words, defaults that work, advanced settings folded away:

* an On/Off switch;
* **Use free models (opencode)** - one click: finds opencode or downloads it (progress, size, MIT license note),
  checks a free model answers, turns the live agent on. If that fails: **Use a free key instead** (Gemini or Groq,
  with a link to get the key) or Ollama when it runs on this PC;
* **Add my API key** - provider, key (saved in Windows Credential Manager, never in a file), optional model, Test;
* **Preferred model**, the wake name, and what it may do (look at the screen, use mouse and keyboard, change
  settings);
* **Advanced** (folded): which model does which job, the daily limit for paid keys, Jev, privacy note;
* **Recent turns**: what was asked, which model answered, how fast, what happened.

:class:`LivePanel` holds the logic and is tested without Tk; :func:`build_tab` draws it.
"""

from __future__ import annotations

import contextlib
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.liveagent import catalog, keys, opencode
from away.liveagent import settings as st
from away.liveagent.agent import load_log
from away.liveagent.runtime import Runtime, add_provider, fallback_options, use_free_models

PRIVACY_NOTE = (
    "Free models (like Big Pickle) and Google's free tier may keep what you send. They never get pictures of your "
    "screen or camera, or anything typed into password fields, and e-mail addresses, phone and card numbers and "
    "keys are hidden before anything is sent to them. Your own paid keys and models on this PC can see the screen "
    "when you allow it. Every action on your PC still asks you first."
)
AUTO = "Automatic"


class LivePanel:
    def __init__(self, data_dir: Path | None = None, keystore: keys.KeyStore | None = None,
                 runtime: Runtime | None = None) -> None:
        if data_dir is None:
            from away import paths

            data_dir = paths.data_dir()
        self.data_dir = Path(data_dir)
        self.path = st.path(self.data_dir)
        self.s = st.load(self.path)
        self.keys = keystore or keys.KeyStore()
        self.runtime = runtime or Runtime(self.data_dir, settings=lambda: self.s, keystore=self.keys)

    # ------------------------------------------------------------ basics
    def save(self) -> str:
        st.save(self.s, self.path)
        return "Saved"

    def set_enabled(self, on: bool) -> str:
        if on and not self.s.providers:
            return "Choose free models or add a key first"
        self.s.enabled = bool(on)
        self.save()
        return "Live agent is on" if on else "Live agent is off"

    def set_name(self, name: str) -> str:
        if not st.NAME.match(name.strip() or "-"):
            return "Use a short name with letters (like Away or Jarvis)"
        self.s.name = name.strip()
        self.save()
        return f'Say "Hey {self.s.name}"'

    def status_line(self) -> str:
        if not self.s.providers:
            return "Not set up yet: click Use free models (no key needed)."
        state = "On" if self.s.enabled else "Off"
        model = ""
        with contextlib.suppress(Exception):
            model = self.runtime.current_model(self.s)
        bad = [f"{p.title or p.id}: {e}" for p, e in ((p, self.runtime.errors.get(p.id)) for p in self.s.providers) if e]
        line = f'{state}. Say "Hey {self.s.name}".' + (f" Answers with {model}." if model else "")
        return line + (f" Problem: {bad[0]}" if bad else "")

    # ------------------------------------------------------------ free models
    def engine_text(self) -> str:
        from away.liveagent.runtime import engine_status

        e = engine_status(self.data_dir)
        if not e.get("installed"):
            return "opencode is not on this PC yet; the button downloads it (about 60 MB, free, MIT license)."
        if e.get("where") == "away":
            return f"opencode {e.get('version')} (downloaded by Away, MIT license)."
        return "Using the opencode you installed."

    def use_free(self, progress: Callable[[str, int, int], None] | None = None,
                 installer: Callable[..., dict[str, Any]] = opencode.install) -> tuple[bool, str]:
        try:
            msg = use_free_models(self.runtime, self.s, progress, installer)
            self.save()
            return True, msg
        except Exception as exc:
            return False, f"Couldn't start the free models: {exc}"
        finally:
            self.runtime.stop()  # the running app starts its own copy when needed

    def fallbacks(self) -> list[dict[str, str]]:
        return fallback_options(self.data_dir, self.runtime.client)

    # ------------------------------------------------------------ keys
    def provider_choices(self) -> list[tuple[str, str]]:
        return [(p.id, p.title) for p in catalog.PROVIDERS if p.kind != "opencode"]

    def add_key(self, kind: str, key: str, model: str = "", base_url: str = "", test: bool = True) -> tuple[bool, str]:
        sp = catalog.spec(kind)
        if sp.needs_key and not key.strip() and not (self.s.provider(kind) and self.keys.has(kind)):
            return False, "Paste the key first"
        cfg = add_provider(self.s, kind, model.strip(), base_url.strip())
        if key.strip():
            try:
                self.keys.set(cfg.id, key.strip())
            except keys.KeyStoreError as exc:
                return False, str(exc)
            cfg.has_key = True
        if not self.s.preferred:
            self.s.preferred = f"{cfg.id}/{cfg.model}" if cfg.model else ""
        self.save()
        if not test:
            return True, "Saved"
        return self.test(cfg.id)

    def test(self, pid: str) -> tuple[bool, str]:
        from away.liveagent import providers

        cfg = self.s.provider(pid)
        if cfg is None:
            return False, "not set up"
        try:
            sp = catalog.spec(cfg.kind)
            if sp.kind == "opencode":
                srv = self.runtime.opencode_server()
                free = self.runtime.free_models(refresh=True)
                model = cfg.model or (free[0].id if free else "big-pickle")
                from away.liveagent.textstream import strip_reasoning

                txt = strip_reasoning("".join(srv.chat_stream(model, "Reply with the single word: ready", timeout=45)))
                return (bool(txt), f"Works ({catalog.pretty(model)})" if txt else "Empty answer")
            p = providers.build(cfg, self.keys.get(cfg.id), self.runtime.client)
            if not cfg.model:
                ids = [m.id for m in p.list_models()]
                cfg.model = catalog.pick_default(ids, sp.default_model)
                self.save()
            msg = p.test(cfg.model)
            if not self.s.preferred:
                self.s.preferred = f"{cfg.id}/{cfg.model}"
                self.save()
            return True, msg
        except Exception as exc:
            return False, f"Didn't work: {plain_error(exc)}"
        finally:
            if catalog.spec(cfg.kind).kind == "opencode":
                self.runtime.stop()

    def remove(self, pid: str) -> str:
        self.s.providers = [p for p in self.s.providers if p.id != pid]
        self.keys.delete(pid)
        if self.s.preferred.startswith(pid + "/"):
            self.s.preferred = ""
        self.s.routing = {k: v for k, v in self.s.routing.items() if not v.startswith(pid + "/")}
        if not self.s.providers:
            self.s.enabled = False
        self.save()
        return "Removed"

    # ------------------------------------------------------------ models
    def model_choices(self) -> list[tuple[str, str]]:
        """(ref, label) for the Preferred model and routing dropdowns."""
        out: list[tuple[str, str]] = [("", AUTO)]
        for p in self.s.providers:
            sp = catalog.spec(p.kind)
            if sp.kind == "opencode":
                for m in self.runtime.free_models():
                    out.append((m.ref, f"{m.label()} - free"))
            elif p.model:
                out.append((f"{p.id}/{p.model}", f"{catalog.pretty(p.model)} - {p.title or sp.title}"))
        return out

    def set_preferred(self, ref: str) -> str:
        self.s.preferred = ref
        self.save()
        return "Saved"

    def set_route(self, task: str, ref: str) -> None:
        if ref:
            self.s.routing[task] = ref
        else:
            self.s.routing.pop(task, None)
        self.save()

    def recent_turns(self, n: int = 30) -> list[dict[str, Any]]:
        return load_log(self.data_dir / "liveagent" / "turns.jsonl", n)

    def test_jev(self) -> str:
        from away.liveagent import jev

        try:
            key = self.keys.get("jev") or self.keys.get("opencode_zen")
        except Exception:
            key = ""
        try:
            return jev.probe(self.runtime.client, key)
        except Exception as exc:
            return f"Jev is not answering ({plain_error(exc)}); the chat model makes those decisions instead."


def plain_error(exc: BaseException) -> str:
    from away.liveagent.http import HttpError

    if isinstance(exc, HttpError):
        if exc.status in (401, 403):
            return "the key was refused (check it, or make a new one)"
        if exc.status == 429:
            return "too many requests right now (try again in a minute)"
        if exc.status == 404:
            return "that model or address was not found"
        if exc.status == 0:
            return "could not connect (internet, or the address)"
    return str(exc)[:160]


# ------------------------------------------------------------------------------------------------ Tk
def build_tab(win: Any, nb: Any) -> LivePanel:  # pragma: no cover - widgets; smoke-tested under Xvfb
    tk, ttk = win.tk, win.ttk
    panel = LivePanel()
    f = win._scroll_tab(nb, "Live agent")
    root = win.root
    status = tk.StringVar(value=panel.status_line())

    def later(fn: Callable[[], None]) -> None:
        with contextlib.suppress(Exception):
            root.after(0, fn)

    def bg(work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        def run() -> None:
            res = work()
            later(lambda: done(res))

        threading.Thread(target=run, daemon=True, name="live-agent-settings").start()

    ttk.Label(f, text="Live agent", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
    ttk.Label(f, text="Talk to your computer. It answers out loud and can do small jobs for you.", wraplength=620).pack(
        anchor="w", pady=(0, 6)
    )
    on = tk.BooleanVar(value=panel.s.enabled)

    def toggle() -> None:
        msg = panel.set_enabled(on.get())
        on.set(panel.s.enabled)
        status.set(f"{msg}. {panel.status_line()}")

    ttk.Checkbutton(f, text="Live agent is on", variable=on, command=toggle).pack(anchor="w")
    ttk.Label(f, textvariable=status, wraplength=620, foreground="#1a5e20").pack(anchor="w", pady=(2, 8))

    # --- free models
    free = ttk.LabelFrame(f, text="Free models (no key needed)", padding=8)
    free.pack(fill="x", pady=4)
    eng = tk.StringVar(value=panel.engine_text())
    ttk.Label(free, textvariable=eng, wraplength=600).pack(anchor="w")
    bar = ttk.Progressbar(free, mode="determinate", maximum=100)
    prog = tk.StringVar(value="")
    fb = ttk.Frame(free)
    btn = ttk.Button(free, text="Use free models (opencode)")

    def progress(msg: str, done: int, total: int) -> None:
        def ui() -> None:
            if total:
                bar["value"] = 100.0 * done / total
                prog.set(f"{msg}: {done / 1e6:.0f} of {total / 1e6:.0f} MB")
            else:
                prog.set(msg)

        later(ui)

    def show_fallbacks(reason: str) -> None:
        for w in fb.winfo_children():
            w.destroy()
        ttk.Label(fb, text=f"{reason}\nUse a free key instead:", wraplength=600, foreground="#9a3412").pack(anchor="w")
        for o in panel.fallbacks():
            row = ttk.Frame(fb)
            row.pack(anchor="w", pady=2)
            if o["key_url"]:
                ttk.Button(row, text=f"Get a free {o['title']} key",
                           command=lambda u=o["key_url"]: webbrowser.open(u)).pack(side="left")
                ttk.Button(row, text="I have it: add it below",
                           command=lambda k=o["kind"]: pick_provider(k)).pack(side="left", padx=4)
            else:
                ttk.Button(row, text=f"Use {o['title']}", command=lambda k=o["kind"]: use_local(k)).pack(side="left")
            ttk.Label(row, text=o.get("note", ""), foreground="#555").pack(side="left", padx=6)
        fb.pack(fill="x", pady=4)

    def use_free() -> None:
        btn.state(["disabled"])
        bar.pack(fill="x", pady=2)
        ttk.Label(free, textvariable=prog).pack(anchor="w")
        ttk.Label(free, text=opencode.LICENSE_NOTE, wraplength=600, foreground="#555").pack(anchor="w")

        def done(res: tuple[bool, str]) -> None:
            ok, msg = res
            btn.state(["!disabled"])
            bar.pack_forget()
            eng.set(panel.engine_text())
            on.set(panel.s.enabled)
            status.set(msg)
            refresh_models()
            refresh_providers()
            if not ok:
                show_fallbacks(msg)

        bg(lambda: panel.use_free(progress), done)

    btn.configure(command=use_free)
    btn.pack(anchor="w", pady=4)

    # --- API keys
    kf = ttk.LabelFrame(f, text="Add my API key", padding=8)
    kf.pack(fill="x", pady=4)
    choices = panel.provider_choices()
    prov = tk.StringVar(value=choices[0][1])
    key_v, model_v, url_v = tk.StringVar(), tk.StringVar(), tk.StringVar()
    grid = ttk.Frame(kf)
    grid.pack(fill="x")
    ttk.Label(grid, text="Provider").grid(row=0, column=0, sticky="w")
    combo = ttk.Combobox(grid, textvariable=prov, values=[c[1] for c in choices], state="readonly", width=28)
    combo.grid(row=0, column=1, sticky="w", padx=4)
    link = ttk.Label(grid, text="", foreground="#1d4ed8", cursor="hand2")
    link.grid(row=0, column=2, sticky="w")
    ttk.Label(grid, text="Key").grid(row=1, column=0, sticky="w")
    ttk.Entry(grid, textvariable=key_v, show="\u2022", width=44).grid(row=1, column=1, columnspan=2, sticky="w", padx=4)
    ttk.Label(grid, text="Model (optional)").grid(row=2, column=0, sticky="w")
    ttk.Entry(grid, textvariable=model_v, width=30).grid(row=2, column=1, sticky="w", padx=4)
    url_lbl = ttk.Label(grid, text="Address")
    url_ent = ttk.Entry(grid, textvariable=url_v, width=44)
    kmsg = tk.StringVar(value="Keys are kept in Windows Credential Manager, never in a file.")

    def kind() -> str:
        return next(c[0] for c in choices if c[1] == prov.get())

    def on_kind(_e: Any = None) -> None:
        sp = catalog.spec(kind())
        link.configure(text="Get a key" if sp.key_url else "")
        link.bind("<Button-1>", lambda _e: sp.key_url and webbrowser.open(sp.key_url))
        if sp.id in ("custom", "ollama", "lmstudio"):
            url_lbl.grid(row=3, column=0, sticky="w")
            url_ent.grid(row=3, column=1, columnspan=2, sticky="w", padx=4)
            url_v.set(sp.base_url)
        else:
            url_lbl.grid_remove()
            url_ent.grid_remove()

    def pick_provider(k: str) -> None:
        prov.set(catalog.spec(k).title)
        on_kind()

    combo.bind("<<ComboboxSelected>>", on_kind)
    on_kind()

    def save_key() -> None:
        kmsg.set("Testing...")
        k, kv, mv, uv = kind(), key_v.get(), model_v.get(), url_v.get()
        key_v.set("")

        def done(res: tuple[bool, str]) -> None:
            kmsg.set(res[1])
            refresh_models()
            refresh_providers()
            status.set(panel.status_line())

        bg(lambda: panel.add_key(k, kv, mv, uv), done)

    def use_local(k: str) -> None:
        pick_provider(k)
        save_key()

    ttk.Button(kf, text="Save and test", command=save_key).pack(anchor="w", pady=4)
    ttk.Label(kf, textvariable=kmsg, wraplength=600).pack(anchor="w")
    plist = ttk.Frame(kf)
    plist.pack(fill="x", pady=4)

    def refresh_providers() -> None:
        for w in plist.winfo_children():
            w.destroy()
        for p in panel.s.providers:
            row = ttk.Frame(plist)
            row.pack(fill="x")
            label = f"{p.title or catalog.spec(p.kind).title}" + (f" - {catalog.pretty(p.model)}" if p.model else "")
            ttk.Label(row, text=label, width=48).pack(side="left")
            res = tk.StringVar()

            def t(pid: str = p.id, v: Any = res) -> None:
                v.set("Testing...")
                bg(lambda: panel.test(pid), lambda r, v=v: v.set(r[1]))

            ttk.Button(row, text="Test", command=t).pack(side="left")
            ttk.Button(row, text="Remove",
                       command=lambda pid=p.id: (panel.remove(pid), refresh_providers(), refresh_models())).pack(
                side="left", padx=4)
            ttk.Label(row, textvariable=res).pack(side="left")

    # --- preferred model, name, permissions
    mf = ttk.LabelFrame(f, text="How it works for you", padding=8)
    mf.pack(fill="x", pady=4)
    ttk.Label(mf, text="Preferred model").grid(row=0, column=0, sticky="w")
    pref = tk.StringVar()
    pref_box = ttk.Combobox(mf, textvariable=pref, state="readonly", width=40)
    pref_box.grid(row=0, column=1, sticky="w", padx=4)
    model_map: dict[str, str] = {}

    def refresh_models() -> None:
        ch = panel.model_choices()
        model_map.clear()
        model_map.update({lbl: ref for ref, lbl in ch})
        pref_box["values"] = [lbl for _r, lbl in ch]
        pref.set(next((lbl for ref, lbl in ch if ref == panel.s.preferred), AUTO))
        for task, (var, box) in route_boxes.items():
            box["values"] = [lbl for _r, lbl in ch]
            var.set(next((lbl for ref, lbl in ch if ref == panel.s.routing.get(task, "")), AUTO))

    pref_box.bind("<<ComboboxSelected>>", lambda _e: status.set(panel.set_preferred(model_map.get(pref.get(), ""))))
    ttk.Label(mf, text="Wake name: Hey").grid(row=1, column=0, sticky="w")
    name_v = tk.StringVar(value=panel.s.name)
    ent = ttk.Entry(mf, textvariable=name_v, width=16)
    ent.grid(row=1, column=1, sticky="w", padx=4)
    ent.bind("<FocusOut>", lambda _e: status.set(panel.set_name(name_v.get())))
    perms = (
        ("allow_screen", "Let it look at my screen (only private models see the picture)"),
        ("allow_control", "Let it use my mouse and keyboard (you approve each time; move to stop)"),
        ("allow_settings", "Let it change Away's settings (you confirm; can be undone)"),
    )
    for i, (attr, text) in enumerate(perms):
        v = tk.BooleanVar(value=bool(getattr(panel.s, attr)))

        def set_perm(a: str = attr, var: Any = v) -> None:
            setattr(panel.s, a, bool(var.get()))
            panel.save()

        ttk.Checkbutton(mf, text=text, variable=v, command=set_perm).grid(row=2 + i, column=0, columnspan=2, sticky="w")

    # --- advanced (folded)
    adv_btn = ttk.Button(f, text="Show advanced \u25b8")
    adv = ttk.Frame(f, padding=(8, 4))
    route_boxes: dict[str, tuple[Any, Any]] = {}
    ttk.Label(adv, text="Which model does which job", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    rt = ttk.Frame(adv)
    rt.pack(fill="x")
    for i, task in enumerate(st.TASKS):
        ttk.Label(rt, text=st.TASK_TITLES[task]).grid(row=i, column=0, sticky="w")
        var = tk.StringVar(value=AUTO)
        box = ttk.Combobox(rt, textvariable=var, state="readonly", width=40)
        box.grid(row=i, column=1, sticky="w", padx=4, pady=1)
        box.bind("<<ComboboxSelected>>", lambda _e, t=task, v=var: panel.set_route(t, model_map.get(v.get(), "")))
        route_boxes[task] = (var, box)
    bf = ttk.Frame(adv)
    bf.pack(fill="x", pady=6)
    ttk.Label(bf, text="Daily limit for paid keys ($)").pack(side="left")
    bud = tk.StringVar(value=f"{panel.s.budget_usd_per_day:g}")
    sp_box = ttk.Spinbox(bf, from_=0, to=100, increment=0.5, textvariable=bud, width=6)
    sp_box.pack(side="left", padx=4)

    def set_budget(_e: Any = None) -> None:
        with contextlib.suppress(ValueError):
            panel.s.budget_usd_per_day = float(bud.get())
            panel.save()

    sp_box.bind("<FocusOut>", set_budget)
    sp_box.configure(command=set_budget)
    spent = tk.StringVar(value=f"Spent today: ${panel.runtime.ledger.spent_today():.2f}")
    ttk.Label(bf, textvariable=spent).pack(side="left", padx=8)
    jf = ttk.Frame(adv)
    jf.pack(fill="x", pady=4)
    ttk.Label(jf, text="Safety decisions with Jev").pack(side="left")
    jv = tk.StringVar(value=panel.s.jev)
    jb = ttk.Combobox(jf, textvariable=jv, values=["auto", "on", "off"], state="readonly", width=6)
    jb.pack(side="left", padx=4)

    def set_jev(_e: Any = None) -> None:
        panel.s.jev = jv.get()
        panel.save()

    jb.bind("<<ComboboxSelected>>", set_jev)
    jmsg = tk.StringVar()
    ttk.Button(jf, text="Test Jev", command=lambda: bg(panel.test_jev, jmsg.set)).pack(side="left", padx=4)
    ttk.Label(jf, textvariable=jmsg, wraplength=380).pack(side="left")
    for attr, text in (("speak", "Answer out loud"), ("barge_in", "Stop talking when I start talking")):
        v = tk.BooleanVar(value=bool(getattr(panel.s, attr)))

        def set_flag(a: str = attr, var: Any = v) -> None:
            setattr(panel.s, a, bool(var.get()))
            panel.save()

        ttk.Checkbutton(adv, text=text, variable=v, command=set_flag).pack(anchor="w")
    ttk.Label(adv, text="Privacy", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(6, 0))
    ttk.Label(adv, text=PRIVACY_NOTE, wraplength=600).pack(anchor="w")
    shown = {"on": False}

    def toggle_adv() -> None:
        shown["on"] = not shown["on"]
        if shown["on"]:
            adv.pack(fill="x", after=adv_btn)
            adv_btn.configure(text="Hide advanced \u25be")
        else:
            adv.pack_forget()
            adv_btn.configure(text="Show advanced \u25b8")

    adv_btn.configure(command=toggle_adv)
    adv_btn.pack(anchor="w", pady=(8, 0))

    # --- recent turns
    tf = ttk.LabelFrame(f, text="Recent turns", padding=6)
    tf.pack(fill="both", expand=True, pady=6)
    cols = ("time", "asked", "model", "first", "result")
    tree = ttk.Treeview(tf, columns=cols, show="headings", height=6)
    for c, w, title in zip(cols, (70, 260, 140, 80, 160), ("Time", "You asked", "Model", "First word", "Result")):
        tree.heading(c, text=title)
        tree.column(c, width=w, anchor="w")
    tree.pack(fill="both", expand=True)

    def refresh_turns() -> None:
        tree.delete(*tree.get_children())
        for t in panel.recent_turns():
            first = t.get("first_spoken_s") or t.get("first_token_s")
            tree.insert("", "end", values=(t.get("time", ""), t.get("text", ""), t.get("model", ""),
                                           f"{first:.1f} s" if first else "-", t.get("outcome", "")))
        spent.set(f"Spent today: ${panel.runtime.ledger.spent_today():.2f}")

    ttk.Button(tf, text="Refresh", command=refresh_turns).pack(anchor="w", pady=2)
    refresh_providers()
    bg(panel.model_choices, lambda _r: refresh_models())  # the free list may need the network: off the UI thread
    refresh_turns()
    return panel
