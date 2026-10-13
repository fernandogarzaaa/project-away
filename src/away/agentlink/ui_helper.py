"""``away agent-ui prompt|toast|flash|control SPEC``: the small always-on-top windows Agent Link shows.

They run in their own process (the tray owns the app's main thread, Tk wants its own) and talk to the app
through files next to SPEC: a click writes ``<id>.answer.json``; the app writes ``<id>.done`` when the prompt was
answered another way (gesture, voice, timeout), and the window closes itself. ``control`` draws a bright border
around the screen with the banner "<agent> is using your computer - move your hand or mouse to take over" (or, in
``watch`` mode, a small "watching your screen" badge) until its stop file appears.
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any

COLORS = {"info": "#1f6feb", "success": "#1a7f37", "warning": "#bf8700", "error": "#cf222e", "camera": "#cf222e"}
BORDER = "#ff2d55"


def prompt_texts(spec: dict[str, Any]) -> dict[str, Any]:
    """What the prompt window shows (pure, for tests)."""
    options = list(spec.get("options") or ["Yes", "No"])
    hints = [
        "thumbs up" if i == 0 else "thumbs down" if i == len(options) - 1 and len(options) > 1 else f"{i + 1} fingers"
        for i in range(len(options))
    ]
    if len(options) > 2:
        hints = [f"{i + 1} finger{'s' if i else ''}" for i in range(len(options))]
        hints[0] = "thumbs up or 1 finger"
    return {
        "title": str(spec.get("title") or "Your AI agent asks"),
        "question": str(spec.get("question") or ""),
        "details": [str(d) for d in spec.get("details") or []],
        "buttons": [(o, f"{o}  ({h} or say \u201c{o.lower()}\u201d)") for o, h in zip(options, hints)],
        "footer": "Answer with a click, a hand sign, or your voice.",
    }


TOAST_W = 380
TOAST_MARGIN = 16
TOAST_SECONDS = 6.0


def toast_texts(spec: dict[str, Any]) -> dict[str, str]:
    """What a message window shows (pure, for tests): "Message from <agent>" over the text."""
    agent = str(spec.get("agent") or "").strip()
    title = str(spec.get("title") or "").strip() or (f"Message from {agent}" if agent else "Message from your AI agent")
    return {"title": title[:80], "text": str(spec.get("text") or "")[:300], "hint": "click to close"}


def toast_geometry(work: tuple[int, int, int, int], w: int, h: int, margin: int = TOAST_MARGIN) -> tuple[int, int]:
    """Top-left corner for a ``w`` x ``h`` window in the bottom-right corner of ``work`` (left, top, right, bottom:
    the monitor's area without the taskbar), kept inside it."""
    left, top, right, bottom = (int(v) for v in work)
    x = max(left, right - w - margin)
    y = max(top, bottom - h - margin)
    return x, y


def write_ack(path: Path | None, data: dict[str, Any]) -> None:
    """Tell the app whether the window really showed (``{"shown": true, "x", "y", ...}`` or an ``error``)."""
    if path is None:
        return
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def main(kind: str, spec_path: Path) -> int:  # pragma: no cover - GUI; smoke-tested under Xvfb
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if kind == "toast":
        ack = Path(spec["ack"]) if spec.get("ack") else None
        try:
            return _message(spec, ack)
        except Exception as exc:  # the app reports this instead of "shown": no silent failures
            import logging

            logging.getLogger("away.agentlink").exception("message window failed")
            write_ack(ack, {"shown": False, "error": f"{type(exc).__name__}: {exc}"})
            return 1
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if kind == "prompt":
        _prompt(root, spec, Path(spec_path))
    elif kind in ("toast", "flash"):
        _toast(root, spec, flash=kind == "flash")
    elif kind == "control":
        _control(root, spec)
    root.mainloop()
    return 0


def _work_area(root: Any) -> tuple[int, int, int, int]:  # pragma: no cover - desktop
    """The usable area (no taskbar) of the monitor the mouse pointer is on (Windows); elsewhere the screen with
    room for a panel or dock at the bottom."""
    import sys

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD),
                ]

            u32 = ctypes.windll.user32  # type: ignore[attr-defined]
            pt = wintypes.POINT()
            u32.GetCursorPos(ctypes.byref(pt))
            u32.MonitorFromPoint.restype = ctypes.c_void_p
            mon = u32.MonitorFromPoint(pt, 2)  # MONITOR_DEFAULTTONEAREST
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            if mon and u32.GetMonitorInfoW(ctypes.c_void_p(mon), ctypes.byref(mi)):
                r = mi.rcWork
                return int(r.left), int(r.top), int(r.right), int(r.bottom)
        except Exception:
            pass
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    return 0, 0, int(sw), int(sh) - 56


def _hwnd(win: Any) -> int:  # pragma: no cover - Windows
    try:
        frame = win.wm_frame()
        h = int(str(frame), 16) if str(frame).startswith("0x") else int(frame)
        if h:
            return h
    except Exception:
        pass
    import ctypes

    return int(ctypes.windll.user32.GetParent(win.winfo_id()) or win.winfo_id())  # type: ignore[attr-defined]


def _pin_on_top(win: Any, x: int, y: int, w: int, h: int) -> dict[str, Any]:  # pragma: no cover - desktop
    """Keep the window above everything without taking the focus; returns what Windows reports about it."""
    import sys

    with __import__("contextlib").suppress(Exception):
        win.attributes("-topmost", True)
        win.lift()
    if sys.platform != "win32":
        return {"visible": bool(win.winfo_viewable())}
    import ctypes

    u32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = _hwnd(win)
    # HWND_TOPMOST, SWP_NOACTIVATE | SWP_SHOWWINDOW: on top of full-screen apps, the focus stays where it was
    u32.SetWindowPos(ctypes.c_void_p(hwnd), ctypes.c_void_p(-1), x, y, w, h, 0x0010 | 0x0040)
    ex = int(u32.GetWindowLongW(ctypes.c_void_p(hwnd), -20))  # GWL_EXSTYLE
    return {"visible": bool(u32.IsWindowVisible(ctypes.c_void_p(hwnd))), "topmost": bool(ex & 0x8), "hwnd": hwnd}


def _message(spec: dict[str, Any], ack: Path | None) -> int:  # pragma: no cover - GUI; smoke-tested under Xvfb
    """The agent's message: a small window in the bottom-right corner of the screen the pointer is on, above every
    other window (without taking the keyboard focus) for ``seconds`` (6 by default); a click closes it. Writes
    ``ack`` once the window is really on screen, so ``notify_user`` can say whether it was shown."""
    import tkinter as tk

    t = toast_texts(spec)
    cancel = ack.with_name(ack.name + ".cancel") if ack is not None else None
    if cancel is not None and cancel.exists():  # the app gave up waiting (it showed a tray notification instead)
        return 0
    color = COLORS.get(str(spec.get("level")), COLORS["info"])
    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg=color, highlightthickness=0)
    body = tk.Frame(root, bg="white")
    body.pack(fill="both", expand=True, padx=(6, 0))  # the colored strip on the left says info/warning/error
    tk.Label(body, text=t["title"], bg="white", fg="#555", font=("Segoe UI", 10, "bold"), anchor="w").pack(
        fill="x", padx=12, pady=(10, 0)
    )
    tk.Label(
        body,
        text=t["text"],
        bg="white",
        fg="#111",
        font=("Segoe UI", 12),
        wraplength=TOAST_W - 40,
        justify="left",
        anchor="w",
    ).pack(fill="x", padx=12, pady=(2, 4))
    tk.Label(body, text=t["hint"], bg="white", fg="#999", font=("Segoe UI", 8), anchor="e").pack(
        fill="x", padx=10, pady=(0, 6)
    )
    root.update_idletasks()
    w, h = TOAST_W, max(70, root.winfo_reqheight())
    x, y = toast_geometry(_work_area(root), w, h)
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.deiconify()
    root.update()
    for widget in (root, body, *body.winfo_children()):
        widget.bind("<Button-1>", lambda _e: root.destroy())
    seconds = float(spec.get("seconds") or TOAST_SECONDS)
    state: dict[str, Any] = {"acked": False}

    def keep_on_top() -> None:
        try:
            info = _pin_on_top(root, x, y, w, h)
        except Exception as exc:
            info = {"visible": bool(root.winfo_viewable()), "pin_error": str(exc)}
        if not state["acked"]:
            state["acked"] = True
            shown = bool(info.get("visible", True)) and bool(root.winfo_ismapped())
            write_ack(ack, {"shown": shown, "x": x, "y": y, "w": w, "h": h, **info})
            if not shown:
                root.after(50, root.destroy)
                return
        root.after(500, keep_on_top)  # stay above windows that become topmost later (a full-screen video)

    root.after(30, keep_on_top)
    root.after(int(seconds * 1000), root.destroy)
    root.mainloop()
    return 0


def _place(win: Any, w: int, h: int, where: str = "top-right") -> None:  # pragma: no cover
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    x = sw - w - 24 if where.endswith("right") else (sw - w) // 2
    y = 24 if where.startswith("top") else (sh - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")


def _prompt(root: Any, spec: dict[str, Any], path: Path) -> None:  # pragma: no cover
    import tkinter as tk

    t = prompt_texts(spec)
    win = tk.Toplevel(root)
    win.title(t["title"])
    win.attributes("-topmost", True)
    win.configure(bg="white", highlightthickness=3, highlightbackground="#1f6feb")
    win.overrideredirect(True)
    tk.Label(win, text=t["title"], bg="white", fg="#555", font=("Segoe UI", 10)).pack(anchor="w", padx=16, pady=(12, 0))
    tk.Label(win, text=t["question"], bg="white", font=("Segoe UI", 15, "bold"), wraplength=440, justify="left").pack(
        anchor="w", padx=16, pady=(4, 6)
    )
    for d in t["details"]:
        tk.Label(
            win, text="\u2022 " + d, bg="white", fg="#333", font=("Segoe UI", 10), wraplength=440, justify="left"
        ).pack(anchor="w", padx=20)
    answer_file = path.with_name(f"{spec['id']}.answer.json")
    done_file = path.with_name(f"{spec['id']}.done")

    def pick(opt: str) -> None:
        answer_file.write_text(json.dumps({"answer": opt}), encoding="utf-8")
        root.after(300, root.destroy)

    for opt, label in t["buttons"]:
        tk.Button(
            win, text=label, font=("Segoe UI", 11), command=functools.partial(pick, opt), relief="groove", bg="#f6f8fa"
        ).pack(fill="x", padx=16, pady=3)
    tk.Label(win, text=t["footer"], bg="white", fg="#777", font=("Segoe UI", 9)).pack(pady=(4, 10))
    win.update_idletasks()
    _place(win, 480, win.winfo_reqheight(), "center")
    deadline = time.monotonic() + float(spec.get("timeout_s") or 60) + 5

    def poll() -> None:
        if done_file.exists():
            done_file.unlink(missing_ok=True)
            root.destroy()
            return
        if time.monotonic() > deadline:
            root.destroy()
            return
        root.after(200, poll)

    poll()


def _toast(root: Any, spec: dict[str, Any], flash: bool) -> None:  # pragma: no cover
    import tkinter as tk

    color = COLORS.get(str(spec.get("level")), COLORS["info"])
    win = tk.Toplevel(root)
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    win.configure(bg=color)
    text = ("\u25c9  " if flash else "") + str(spec.get("text", ""))
    tk.Label(
        win,
        text=text,
        bg=color,
        fg="white",
        font=("Segoe UI", 12, "bold"),
        wraplength=380,
        justify="left",
        padx=14,
        pady=10,
    ).pack()
    win.update_idletasks()
    _place(win, max(260, win.winfo_reqwidth()), win.winfo_reqheight(), "top-right")
    if flash:
        edges = _border(root, color, 8)
        root.after(int(float(spec.get("seconds") or 1.5) * 1000), lambda: [e.destroy() for e in edges])
    root.after(int(float(spec.get("seconds") or 6) * 1000), root.destroy)


def _border(root: Any, color: str, width: int) -> list[Any]:  # pragma: no cover
    import tkinter as tk

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    edges = []
    for x, y, w, h in ((0, 0, sw, width), (0, sh - width, sw, width), (0, 0, width, sh), (sw - width, 0, width, sh)):
        e = tk.Toplevel(root)
        e.overrideredirect(True)
        e.attributes("-topmost", True)
        e.configure(bg=color)
        e.geometry(f"{w}x{h}+{x}+{y}")
        edges.append(e)
    return edges


def _control(root: Any, spec: dict[str, Any]) -> None:  # pragma: no cover
    import tkinter as tk

    stop = Path(str(spec.get("stop_file")))
    name = str(spec.get("agent") or "Your AI agent")
    watch = spec.get("mode") == "watch"
    edges = [] if watch else _border(root, BORDER, 6)
    banner = tk.Toplevel(root)
    banner.overrideredirect(True)
    banner.attributes("-topmost", True)
    color = "#6e40c9" if watch else BORDER
    banner.configure(bg=color)
    text = (
        f"\u25c9  {name} is watching your screen"
        if watch
        else f"{name} is using your computer \u2014 move your hand or mouse to take over"
    )
    tk.Label(
        banner, text=text, bg=color, fg="white", font=("Segoe UI", 11 if watch else 13, "bold"), padx=16, pady=6
    ).pack()
    banner.update_idletasks()
    _place(banner, banner.winfo_reqwidth(), banner.winfo_reqheight(), "top-right" if watch else "top-center")
    started = time.monotonic()

    def poll() -> None:
        if stop.exists() or time.monotonic() - started > 4 * 3600:
            stop.unlink(missing_ok=True)
            for e in edges:
                e.destroy()
            root.destroy()
            return
        banner.lift()
        root.after(300, poll)

    poll()
