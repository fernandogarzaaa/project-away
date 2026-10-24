"""First-time setup in a window (v0.1.32; the frozen app on macOS and Linux).

The first launch downloads the models (~1.6 GB). Before, that happened with no window at all: only a notification
at the start and the end (on macOS credited to Script Editor), and no menu-bar icon until it finished, so the app
looked dead. Now ``away setup-window setup --yes ...`` runs ``setup`` and shows its progress: a title, a progress
bar and the current download line.

It is its own process (the app waits for it, then starts the menu-bar icon): Tk and the tray each want the main
thread, and on macOS both drive the same ``NSApplication``, so the app never mixes them in one process (the agent
windows are separate processes for the same reason). ``setup`` runs on a worker thread here while the window runs
on the main thread. Without a display (or without Tk) the setup still runs, just without the window.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

PERCENT_RE = re.compile(r"(\d{1,3})%")


@dataclass
class Snapshot:
    line: str
    percent: int | None  # None: no percentage known yet (the bar just moves)
    done: bool
    rc: int | None


class Progress:
    """What the setup printed, reduced to what the window shows (thread-safe; the worker writes, the window reads)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._line = ""
        self._percent: int | None = None
        self._done = False
        self._rc: int | None = None

    def feed(self, text: str) -> None:
        """One write from ``setup``: lines end in ``\\n`` and the download bar redraws itself with ``\\r``."""
        for seg in re.split(r"[\r\n]+", text):
            seg = seg.strip()
            if not seg:
                continue
            m = PERCENT_RE.search(seg)
            with self._lock:
                self._line = seg
                if m:
                    self._percent = min(100, int(m.group(1)))
                elif "downloading" in seg.lower():
                    self._percent = None  # the next file starts: no percentage for it yet

    def finish(self, rc: int) -> None:
        with self._lock:
            self._done, self._rc = True, int(rc)

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(self._line, self._percent, self._done, self._rc)


class Tee(io.TextIOBase):
    """``sys.stdout`` while the setup runs: everything still goes to ``out`` (the log), and to ``progress``."""

    def __init__(self, out: TextIO | None, progress: Progress) -> None:
        self.out = out
        self.progress = progress

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if self.out is not None:
            with contextlib.suppress(OSError, ValueError):  # a closed log must not stop the download
                self.out.write(s)
        self.progress.feed(s)
        return len(s)

    def flush(self) -> None:
        if self.out is not None:
            with contextlib.suppress(OSError, ValueError):
                self.out.flush()


def tray_place(platform: str | None = None) -> str:
    plat = platform or sys.platform
    if plat == "darwin":
        return "the menu bar at the top right of the screen"
    return "the system tray (the icons in your panel or taskbar)"


def run(
    setup_argv: Sequence[str],
    cli_main: Callable[[list[str]], int],
    window: Callable[[Progress, str], None] | None = None,
    log: TextIO | None = None,
    intro: str = "",
) -> int:
    """Run ``cli_main(setup_argv)`` on a worker thread with its output in ``progress``; show ``window`` meanwhile.
    Returns setup's exit code. A window that cannot open (no display, no Tk) just means no window."""
    progress = Progress()
    old_out, old_err = sys.stdout, sys.stderr
    tee = Tee(log if log is not None else old_out, progress)
    sys.stdout = sys.stderr = tee  # the worker's prints (and the download bar) go through the tee

    def work() -> None:
        rc = 1
        try:
            rc = int(cli_main(list(setup_argv)))
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
        except BaseException as exc:  # never leave the window waiting for a thread that died
            print(f"setup failed: {type(exc).__name__}: {exc}")
        finally:
            progress.finish(rc)

    worker = threading.Thread(target=work, name="away-setup", daemon=True)
    worker.start()
    try:
        (window or tk_window)(progress, intro)
    except Exception as exc:  # no display / no Tk: the download goes on without a window
        print(f"setup window unavailable ({type(exc).__name__}: {exc}); downloading without it")
    finally:
        worker.join()  # the window may close early (the user closed it); the setup still finishes
        sys.stdout, sys.stderr = old_out, old_err
    snap = progress.snapshot()
    return snap.rc if snap.rc is not None else 1


def intro_text(size_mb: float) -> str:
    size = f"about {size_mb / 1000:.1f} GB" if size_mb >= 100 else "a few MB"
    return (
        f"Downloading the hand-tracking and speech models ({size}). This happens only once; "
        "Project Away starts by itself when it is done."
    )


def main(setup_argv: Sequence[str], cli_main: Callable[[list[str]], int]) -> int:  # pragma: no cover - frozen app
    """``away setup-window setup --yes ...``: the setup with its window; output also goes to the app's log file."""
    from away import config as cfgmod
    from away.app import frozen

    try:
        cfg, _ = cfgmod.load()
        size = frozen.setup_download_mb(cfg)
    except Exception:
        size = 0.0
    log: Any = None
    try:
        path = frozen.log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        log = open(path, "a", buffering=1, encoding="utf-8", errors="replace")  # noqa: SIM115
    except OSError:
        log = None
    try:
        return run(setup_argv, cli_main, log=log, intro=intro_text(size))
    finally:
        if log is not None:
            log.close()


def tk_window(progress: Progress, intro: str) -> None:  # pragma: no cover - GUI; the frozen app on a desktop
    """The window itself: returns when the setup is done (a few seconds after, so "done" can be read) or when the
    user closes it."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Project Away: first-time setup")
    root.resizable(False, False)
    frame = tk.Frame(root, padx=22, pady=18)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, text="Setting up Project Away", font=("TkDefaultFont", 15, "bold"), anchor="w").pack(fill="x")
    tk.Label(frame, text=intro, wraplength=440, justify="left", anchor="w").pack(fill="x", pady=(6, 12))
    bar = ttk.Progressbar(frame, length=440, mode="indeterminate")
    bar.pack(fill="x")
    status = tk.Label(frame, text="Starting...", fg="#555", wraplength=440, justify="left", anchor="w")
    status.pack(fill="x", pady=(8, 0))
    close = tk.Button(frame, text="Close", command=root.destroy)
    bar.start(12)
    state = {"determinate": False}

    def tick() -> None:
        snap = progress.snapshot()
        if snap.done:
            bar.stop()
            bar.configure(mode="determinate")
            if snap.rc == 0:
                bar["value"] = 100
                status["text"] = f"Done. Project Away is starting: look for its icon in {tray_place()}."
                root.after(5000, root.destroy)
            else:
                status["text"] = (
                    "Some downloads did not finish. Project Away starts anyway and tries again the next time you "
                    "open it (the log is in Project Away's logs folder)."
                )
                close.pack(anchor="e", pady=(12, 0))
                root.after(30000, root.destroy)
            return
        if snap.percent is not None:
            if not state["determinate"]:
                bar.stop()
                bar.configure(mode="determinate")
                state["determinate"] = True
            bar["value"] = snap.percent
        elif state["determinate"]:
            bar.configure(mode="indeterminate")
            bar.start(12)
            state["determinate"] = False
        if snap.line:
            status["text"] = snap.line[-140:]
        root.after(200, tick)

    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    root.geometry(f"+{max(0, (root.winfo_screenwidth() - w) // 2)}+{max(0, (root.winfo_screenheight() - h) // 3)}")
    root.lift()
    root.attributes("-topmost", True)  # a menu-bar app has no Dock icon: make sure the window is seen
    root.after(2000, lambda: root.attributes("-topmost", False))
    root.after(200, tick)
    root.mainloop()
