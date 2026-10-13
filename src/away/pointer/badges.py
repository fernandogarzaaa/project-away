"""Numbered badges over the candidates of an ambiguous "click <name>" ("say a number").

:func:`badge_specs` is pure (tested): where each badge goes and what it says. :class:`BadgeOverlay` draws them in
a borderless, click-through, always-on-top tkinter window on its own thread (Windows; elsewhere it only records
what it would show, which is also what tests look at). Every Tk call happens on that thread.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from away.pointer.targets import Target

BADGE_PX = 26


@dataclass(frozen=True)
class Badge:
    number: int
    x: int  # top-left of the badge, desktop px
    y: int
    label: str  # the element name, shortened


def short(text: str, n: int = 28) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def badge_specs(targets: Sequence[Target], size: int = BADGE_PX) -> list[Badge]:
    """One badge per target at its top-left corner, nudged right so two badges never sit on top of each other."""
    out: list[Badge] = []
    taken: list[tuple[int, int]] = []
    for i, t in enumerate(targets, start=1):
        x, y = int(t.rect[0]) - size // 3, int(t.rect[1]) - size // 3
        while any(abs(x - a) < size and abs(y - b) < size for a, b in taken):
            x += size
        taken.append((x, y))
        out.append(Badge(i, x, y, short(t.name)))
    return out


class BadgeOverlay:
    """``show(badges)`` / ``hide()`` from any thread."""

    def __init__(self, platform: str | None = None) -> None:
        self.enabled = (platform or sys.platform).startswith("win")
        self.shown: list[Badge] = []
        self._q: queue.Queue[list[Badge] | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def show(self, badges: list[Badge]) -> None:
        self.shown = list(badges)
        if self.enabled:
            self._ensure()
            self._q.put(list(badges))

    def hide(self) -> None:
        self.shown = []
        if self.enabled and self._thread is not None:
            self._q.put([])

    def _ensure(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="pointer-badges", daemon=True)
            self._thread.start()

    def _run(self) -> None:  # pragma: no cover - needs a Windows desktop
        try:
            import tkinter as tk

            from away.app.cursor_overlay import _click_through
        except Exception:
            return
        key = "#010203"
        try:
            root = tk.Tk()
        except Exception:
            return
        root.withdraw()
        wins: list[Any] = []

        def draw(badges: list[Badge]) -> None:
            for w in wins:
                with contextlib.suppress(Exception):
                    w.destroy()
            wins.clear()
            for b in badges:
                w = tk.Toplevel(root)
                w.overrideredirect(True)
                w.attributes("-topmost", True)
                with contextlib.suppress(tk.TclError):
                    w.attributes("-transparentcolor", key)
                w.configure(bg=key)
                tk.Label(
                    w, text=str(b.number), bg="#FFD54F", fg="#111", font=("Segoe UI", 12, "bold"), width=2
                ).pack(side="left")
                tk.Label(w, text=b.label, bg="#111", fg="#FFD54F", font=("Segoe UI", 9), padx=4).pack(side="left")
                w.geometry(f"+{b.x}+{b.y}")
                w.update_idletasks()
                with contextlib.suppress(Exception):
                    _click_through(w)
                wins.append(w)

        def pump() -> None:
            try:
                while True:
                    item = self._q.get_nowait()
                    if item is None:
                        root.destroy()
                        return
                    draw(item)
            except queue.Empty:
                pass
            root.after(50, pump)

        pump()
        root.mainloop()
