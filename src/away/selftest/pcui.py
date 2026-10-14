"""The Check my PC window (v0.1.24): instructions while it runs, then Pass / Fail / Skipped per item with fix
buttons, "Copy report", "Open full report", "Send to my agents" and "Run again". Reuses the self-test window's
colours, ordering and fix buttons (:mod:`away.selftest.ui`)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from away.selftest.pccheck import VERDICT_WORDS

COLOURS = {"pass": "#1b7f3b", "fail": "#b3261e", "skipped": "#6b6b6b"}
ORDER = {"fail": 0, "pass": 1, "skipped": 2}


def rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(report.get("results") or [], key=lambda r: ORDER.get(str(r.get("verdict")), 9))


def on_main(root: Any, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on Tk's thread and wait for its answer (dialogs must not open from the worker thread)."""
    box: dict[str, Any] = {}
    done = threading.Event()

    def go() -> None:
        try:
            box["v"] = fn()
        finally:
            done.set()

    root.after(0, go)
    done.wait()
    return box.get("v")


def run_window(only: list[str] | None = None, trigger: str = "you") -> int:  # pragma: no cover - needs a display
    import tkinter as tk
    import webbrowser
    from pathlib import Path
    from tkinter import messagebox

    from away.selftest import checkpc, pccheck
    from away.selftest import cli as stcli
    from away.selftest.ui import _fix_cmd

    root = tk.Tk()
    root.title("Project Away - Check my PC")
    root.geometry("780x600")
    head = tk.Label(root, text="Check my PC", font=("Segoe UI", 15, "bold"), anchor="w")
    head.pack(fill="x", padx=16, pady=(14, 4))
    sub = tk.Label(
        root,
        text="About 2 minutes. Follow the instructions; you will hear a few test voices.",
        anchor="w",
        justify="left",
        wraplength=720,
    )
    sub.pack(fill="x", padx=16)
    body = tk.Frame(root)
    body.pack(fill="both", expand=True, padx=12, pady=8)
    bottom = tk.Frame(root)
    bottom.pack(fill="x", padx=12, pady=(0, 12))
    state: dict[str, Any] = {"got": None}

    def prompt(text: str) -> None:
        root.after(0, lambda: sub.config(text=text))
        on_main(root, lambda: messagebox.showinfo("Check my PC", text))

    def confirm(question: str) -> bool | None:
        return bool(on_main(root, lambda: messagebox.askyesno("Check my PC", question)))

    def progress(label: str, i: int, n: int) -> None:
        if i < n:
            root.after(0, lambda: head.config(text=f"Checking {i + 1} of {n}: {label}"))

    def work() -> None:
        try:
            state["got"] = checkpc.run_check(prompt, confirm, only, trigger, progress)
        except Exception as exc:
            state["got"] = {"error": str(exc)}
        root.after(0, show)

    def show() -> None:
        for w in body.winfo_children():
            w.destroy()
        got = state["got"] or {}
        rep = got.get("report")
        if not rep:
            head.config(text="Check my PC could not finish: " + str(got.get("error") or "no report"))
            return
        bad = rep.get("verdicts", {}).get("fail", 0)
        head.config(text=str(rep.get("headline", "")), fg=COLOURS["fail" if bad else "pass"])
        sub.config(
            text="Saved on this PC. Nothing was sent to any agent; use \u201cSend to my agents\u201d if you want to."
        )
        canvas = tk.Canvas(body, highlightthickness=0)
        bar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        inner.columnconfigure(1, weight=1)
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        run_fix = stcli.fix_runner(bool(got.get("in_app")))
        for i, r in enumerate(rows(rep)):
            v = str(r.get("verdict"))
            tk.Label(
                inner, text=VERDICT_WORDS.get(v, v), fg=COLOURS.get(v, "black"), font=("Segoe UI", 10, "bold"), width=8
            ).grid(row=i, column=0, sticky="n", pady=3)
            tk.Label(
                inner, text=f"{r.get('title')}\n{r.get('summary')}", justify="left", anchor="w", wraplength=420
            ).grid(row=i, column=1, sticky="w", pady=3)
            fx = r.get("fix")
            if fx:
                tk.Button(inner, text=fx["label"], command=_fix_cmd(root, run_fix, fx["action"], messagebox)).grid(
                    row=i, column=2, sticky="ne", padx=(6, 10), pady=3
                )

    def copy() -> None:
        rep = (state["got"] or {}).get("report")
        if rep:
            root.clipboard_clear()
            root.clipboard_append(pccheck.text(rep))
            sub.config(text="Copied. It has no passwords, keys, names or private paths.")

    def open_full() -> None:
        p = ((state["got"] or {}).get("files") or {}).get("html")
        if p:
            webbrowser.open(Path(p).as_uri())

    def send_it() -> None:
        rep = (state["got"] or {}).get("report")
        if not rep:
            return
        sub.config(text="Sending...")

        def go() -> None:
            msg = checkpc.sent_text(checkpc.send(rep))
            root.after(0, lambda: sub.config(text=msg))

        threading.Thread(target=go, daemon=True).start()

    def again() -> None:
        head.config(text="Check my PC", fg="black")
        threading.Thread(target=work, daemon=True).start()

    tk.Button(bottom, text="Copy report", command=copy).pack(side="left")
    tk.Button(bottom, text="Open full report", command=open_full).pack(side="left", padx=6)
    tk.Button(bottom, text="Send to my agents", command=send_it).pack(side="left")
    tk.Button(bottom, text="Run again", command=again).pack(side="left", padx=6)
    tk.Button(bottom, text="Close", command=root.destroy).pack(side="right")
    threading.Thread(target=work, daemon=True).start()
    root.mainloop()
    return 0
