"""Optional minimal always-on-top status overlay (tkinter, ships with Python)."""

from __future__ import annotations

from typing import Any

from away.events import Event, EventBus

STATUS_TOPICS = {
    "gesture.clutch": lambda d: "PAUSED" if d.get("paused") else "ACTIVE",
    "dictation.start": lambda d: "LISTENING...",
    "dictation.result": lambda d: (
        f"typed: {d.get('text', '')[:40]}" if d.get("status") == "ok" else d.get("status", "")
    ),
    "gesture.click": lambda d: "click",
    "gesture.double_click": lambda d: "double click",
    "gesture.drag_start": lambda d: "drag",
    "gesture.swipe_left": lambda d: "swipe left",
    "gesture.swipe_right": lambda d: "swipe right",
    "voice.state": lambda d: f"voice: {d.get('state', '')}" + (" (commands)" if d.get("intent") == "command" else ""),
    "action.fired": lambda d: f"{d.get('trigger', '')} -> {d.get('action', '')}",
    "app.paused": lambda d: "PAUSED" if d.get("paused") else "ACTIVE",
}


def status_text(ev: Event) -> str | None:
    fn = STATUS_TOPICS.get(ev.topic)
    return fn(ev.data) if fn else None


def run_overlay(bus: EventBus) -> None:  # pragma: no cover - needs a display
    import tkinter as tk

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry("+20+20")
    label = tk.Label(root, text="Project Away", bg="#111", fg="#7CFC9A", font=("Segoe UI", 11), padx=10, pady=4)
    label.pack()
    pending: list[str] = []

    def on_event(ev: Any) -> None:
        s = status_text(ev)
        if s:
            pending.append(s)

    bus.subscribe("*", on_event)

    def tick() -> None:
        if pending:
            label.config(text=pending[-1])
            pending.clear()
        root.after(100, tick)

    tick()
    root.mainloop()
