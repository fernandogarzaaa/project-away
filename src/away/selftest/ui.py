"""The self-test window (v0.1.24): one big button's worth of checks, the result of each in plain words, and a fix
button next to anything that is not right.

Opened by the tray ("Run self-test"), Settings ("Run self-test") and ``away selftest --window``. While the checks
run it shows which one is going; then one row per result (a coloured mark, the title, one sentence, the fix
button). At the bottom: who the report was sent to (agents allowed to "Share my logs and recordings"), or, when
no agent may receive it, "Copy report" and "Save as file" so the person can share it themselves. Both copies are
redacted.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from away.selftest.model import FAIL, ICONS, ORDER, PASS, SKIP, WARN, WORDS, report_text

COLOURS = {PASS: "#1b7f3b", WARN: "#a86400", FAIL: "#b3261e", SKIP: "#6b6b6b"}


def sorted_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(report.get("results") or [], key=lambda r: ORDER.get(r.get("status"), 9))


def footer_text(report: dict[str, Any]) -> str:
    sent = report.get("delivered_to") or []
    if sent:
        return "Sent to " + ", ".join(sent) + ". They can read it any time as \u201cselftest latest\u201d."
    return "No AI agent is connected, so the report stays on this PC. Copy it or save it to share it."


def run_window(only: list[str] | None = None, trigger: str = "you", tk_mod: Any = None) -> int:  # pragma: no cover
    """Run the self-test (in the running app when there is one) and show it. Needs tkinter and a display."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    from away.selftest import cli as stcli

    tkm = tk_mod or tk
    root = tkm.Tk()
    root.title("Project Away self-test")
    root.geometry("760x580")
    head = tkm.Label(root, text="Checking your PC...", font=("Segoe UI", 15, "bold"), anchor="w")
    head.pack(fill="x", padx=16, pady=(14, 4))
    sub = tkm.Label(root, text="This takes about a minute. You may hear a short test voice.", anchor="w")
    sub.pack(fill="x", padx=16)
    body = tkm.Frame(root)
    body.pack(fill="both", expand=True, padx=12, pady=8)
    bottom = tkm.Frame(root)
    bottom.pack(fill="x", padx=12, pady=(0, 12))
    state: dict[str, Any] = {"report": None, "in_app": False}

    def ask(text: str) -> None:
        root.after(0, lambda: sub.config(text=text))
        messagebox.showinfo("Project Away self-test", text + "\n\nPress OK, then say it.")

    def work() -> None:
        got = stcli._via_app(only, trigger)
        if got is None:
            got = stcli.run_here(only, trigger, ask)
            state["in_app"] = False
        else:
            state["in_app"] = True
        state["report"] = got.get("report")
        root.after(0, show)

    def show() -> None:
        rep = state["report"]
        for w in body.winfo_children():
            w.destroy()
        if not rep:
            head.config(text="The self-test could not finish")
            return
        head.config(text=str(rep.get("headline", "")), fg=COLOURS.get(rep.get("status"), "black"))
        sub.config(text=footer_text(rep), wraplength=600, justify="left")
        canvas = tkm.Canvas(body, highlightthickness=0)
        bar = tkm.Scrollbar(body, orient="vertical", command=canvas.yview)
        inner = tkm.Frame(canvas)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        inner.columnconfigure(1, weight=1)
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        run_fix = stcli.fix_runner(bool(state["in_app"]))
        for i, r in enumerate(sorted_results(rep)):
            st = str(r.get("status"))
            tkm.Label(inner, text=ICONS.get(st, "?"), fg=COLOURS.get(st), font=("Segoe UI", 13, "bold"), width=2).grid(
                row=i, column=0, sticky="n", pady=3
            )
            text = f"{r.get('title')} \u2014 {WORDS.get(st, st)}\n{r.get('summary')}"
            tkm.Label(inner, text=text, justify="left", anchor="w", wraplength=360).grid(
                row=i, column=1, sticky="w", pady=3
            )
            fx = r.get("fix")
            if fx:
                tkm.Button(inner, text=fx["label"], command=_fix_cmd(root, run_fix, fx["action"], messagebox)).grid(
                    row=i, column=2, sticky="ne", padx=(6, 10), pady=3
                )

    def copy() -> None:
        rep = state["report"]
        if rep:
            root.clipboard_clear()
            root.clipboard_append(report_text(rep))
            sub.config(text="Copied. Paste it wherever you like (it has no passwords, names or private paths).")

    def save() -> None:
        rep = state["report"]
        if not rep:
            return
        path = filedialog.asksaveasfilename(
            title="Save the self-test report",
            defaultextension=".txt",
            initialfile="project-away-selftest.txt",
            filetypes=[("Text", "*.txt"), ("JSON", "*.json")],
        )
        if path:
            data = json.dumps(rep, indent=2, default=str) if path.lower().endswith(".json") else report_text(rep)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(data)
            sub.config(text=f"Saved: {path}")

    def again() -> None:
        head.config(text="Checking your PC...", fg="black")
        threading.Thread(target=work, daemon=True).start()

    tkm.Button(bottom, text="Copy report", command=copy).pack(side="left")
    tkm.Button(bottom, text="Save as file", command=save).pack(side="left", padx=6)
    tkm.Button(bottom, text="Run again", command=again).pack(side="left")
    tkm.Button(bottom, text="Close", command=root.destroy).pack(side="right")
    threading.Thread(target=work, daemon=True).start()
    root.mainloop()
    return 0


def _fix_cmd(root: Any, run_fix: Callable[[str], dict[str, Any]], action: str, messagebox: Any) -> Callable[[], None]:
    def go() -> None:
        def work() -> None:
            res = run_fix(action)
            root.after(0, lambda: messagebox.showinfo("Project Away", str(res.get("message") or "Done.")))

        threading.Thread(target=work, daemon=True).start()

    return go
