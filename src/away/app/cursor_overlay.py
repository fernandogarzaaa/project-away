"""At-cursor feedback ring (tkinter): pinch progress, drag, paused, dwell, hand lost.

:func:`ring_spec` is pure (tested). :func:`run_cursor_overlay` polls a state
function (``Pipeline.cursor_state``) about 30 times a second and draws a small
ring next to the cursor in a borderless, always-on-top window. On Windows the
window is made click-through; elsewhere it sits offset from the hotspot so it
never covers what you click. Idea from Tracky Mouse's dwell/gesture overlay (MIT);
implementation is original.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from typing import Any

COLORS = {
    "idle": "#7CFC9A",
    "pinch": "#FFD54F",
    "drag": "#4FC3F7",
    "paused": "#9E9E9E",
    "dwell": "#CE93D8",
    "hand_lost": "#9E9E9E",
    "click": "#FFFFFF",
}
SIZE = 34  # window size in px
OFFSET = (18, 18)  # from the cursor hotspot (keeps the ring out from under the pointer)


def ring_spec(state: dict[str, Any]) -> dict[str, Any]:
    """What to draw: ``{"color", "extent", "filled", "visible", "width"}``; ``extent`` in degrees (0..360)."""
    s = str(state.get("state", "idle"))
    pinch = float(state.get("pinch", 0.0) or 0.0)
    dwell = float(state.get("dwell", 0.0) or 0.0)
    if s == "paused":
        return {"color": COLORS["paused"], "extent": 360.0, "filled": False, "visible": True, "width": 2}
    if s == "hand_lost":
        return {"color": COLORS["hand_lost"], "extent": 360.0, "filled": False, "visible": True, "width": 1}
    if s == "drag":
        return {"color": COLORS["drag"], "extent": 360.0, "filled": True, "visible": True, "width": 4}
    if s == "dwell":
        return {"color": COLORS["dwell"], "extent": 360.0 * dwell, "filled": False, "visible": True, "width": 4}
    if s == "pinch":
        return {"color": COLORS["pinch"], "extent": 360.0, "filled": True, "visible": True, "width": 4}
    # idle: the ring fills as the pinch closes, invisible when the hand is relaxed
    return {"color": COLORS["idle"], "extent": 360.0 * pinch, "filled": False, "visible": pinch > 0.05, "width": 3}


def _click_through(root: Any) -> None:  # pragma: no cover - Windows only
    if not sys.platform.startswith("win"):
        return
    import ctypes

    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
    ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x00080000 | 0x00000020)  # LAYERED | TRANSPARENT


def run_cursor_overlay(state_fn: Callable[[], dict[str, Any]], status_bus: Any = None) -> None:  # pragma: no cover
    import tkinter as tk

    key = "#010203"
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-transparentcolor", key)  # Windows
    except tk.TclError:
        with contextlib.suppress(tk.TclError):
            root.attributes("-alpha", 0.85)  # X11/macOS: no colour keying; a slightly see-through square
    cv = tk.Canvas(root, width=SIZE, height=SIZE, bg=key, highlightthickness=0)
    cv.pack()
    root.update_idletasks()
    _click_through(root)
    if status_bus is not None:
        from away.app.overlay import status_text

        top = tk.Toplevel(root)
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        top.geometry("+20+20")
        label = tk.Label(top, text="Project Away", bg="#111", fg="#7CFC9A", padx=10, pady=4)
        label.pack()
        status_bus.subscribe("*", lambda ev: (s := status_text(ev)) and label.after(0, label.config, {"text": s}))

    def tick() -> None:
        try:
            st = state_fn()
        except Exception:
            st = {"state": "hand_lost"}
        spec = ring_spec(st)
        cv.delete("all")
        if spec["visible"]:
            pad = spec["width"] + 1
            box = (pad, pad, SIZE - pad, SIZE - pad)
            if spec["filled"]:
                cv.create_oval(*box, outline=spec["color"], width=spec["width"], fill=spec["color"])
            elif spec["extent"] >= 359.9:
                cv.create_oval(*box, outline=spec["color"], width=spec["width"])
            elif spec["extent"] > 0:
                cv.create_arc(
                    *box, start=90, extent=-spec["extent"], style="arc", outline=spec["color"], width=spec["width"]
                )
        x, y = int(st.get("x", 0)), int(st.get("y", 0))
        root.geometry(f"{SIZE}x{SIZE}+{x + OFFSET[0]}+{y + OFFSET[1]}")
        root.after(33, tick)

    tick()
    root.mainloop()
