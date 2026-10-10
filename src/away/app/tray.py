"""System tray app (pystray): listening state, pause, mic on/off, dictation, settings, quit."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from away import config as cfgmod
from away import paths

STATE_COLORS = {
    "active": (46, 160, 67, 255),  # gestures on, mic listening (or push-to-talk idle)
    "recording": (33, 118, 214, 255),  # push-to-talk / toggle capturing
    "stopped": (214, 160, 33, 255),  # always-on: "stop listening" was said
    "muted": (200, 60, 60, 255),  # mic off
    "paused": (140, 140, 140, 255),  # gestures paused
}


def icon_state(paused: bool, voice_state: str) -> str:
    """Which colour the tray icon shows: paused wins, then mic muted/stopped/recording, else active."""
    if paused:
        return "paused"
    if voice_state in ("muted", "stopped", "recording"):
        return voice_state
    return "active"


def make_icon_image(paused: bool = False, size: int = 64, state: str | None = None) -> Any:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = STATE_COLORS[state or ("paused" if paused else "active")]
    d.ellipse((4, 4, size - 4, size - 4), fill=color)
    # a simple open-hand glyph: palm + four fingers
    w = size // 10
    for i in range(4):
        x = size // 4 + i * (w + 2)
        d.rounded_rectangle((x, size // 5, x + w, size // 2 + 4), radius=w // 2, fill="white")
    d.rounded_rectangle((size // 4 - 2, size // 2, size * 3 // 4 + 2, size * 4 // 5), radius=6, fill="white")
    return img


def open_path(path: str) -> None:  # pragma: no cover - launches a GUI app
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class TrayController:
    """Menu state, separated from pystray so it can be unit-tested."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.paused = False

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        self.app.set_paused(self.paused)
        return self.paused

    def toggle_dictation(self) -> bool:
        self.app.toggle_dictation()
        d = getattr(self.app, "dictation", None)
        return bool(d.enabled) if d is not None else False

    def dictation_enabled(self) -> bool:
        d = getattr(self.app, "dictation", None)
        return bool(d.enabled) if d is not None else False

    def settings_path(self) -> str:
        p = paths.config_file()
        if not p.exists():
            cfgmod.save(self.app.cfg, p)
        return str(p)

    # mic / listening ------------------------------------------------------
    def voice_state(self) -> str:
        fn = getattr(self.app, "voice_state", None)
        return str(fn()) if callable(fn) else "off"

    def mic_on(self) -> bool:
        return self.voice_state() not in ("muted", "off")

    def toggle_mic(self) -> bool:
        """Mic on/off (mute). Returns the new 'mic on' state."""
        on = not self.mic_on()
        fn = getattr(self.app, "set_mic", None)
        if callable(fn):
            fn(on)
        return self.mic_on()

    def title(self) -> str:
        state = self.voice_state()
        voice = {
            "listening": "listening",
            "stopped": "not listening",
            "muted": "mic off",
            "recording": "recording",
            "idle": "push-to-talk ready",
            "off": "voice off",
        }.get(state, state)
        return f"Project Away - {'paused' if self.paused else 'active'}, {voice}"

    def icon_state(self) -> str:
        return icon_state(self.paused, self.voice_state())

    @staticmethod
    def settings_command() -> list[str]:
        """argv that opens the settings window in its own process (Tk and pystray each want the main thread)."""
        if getattr(sys, "frozen", False):  # PyInstaller build: the app exe is the `away` CLI
            return [sys.executable, "settings"]
        return [sys.executable, "-m", "away", "settings"]

    def quit(self) -> None:
        self.app.stop()


def open_settings(ctl: TrayController) -> None:  # pragma: no cover - launches a GUI process
    try:
        subprocess.Popen(ctl.settings_command())
    except Exception:
        open_path(ctl.settings_path())  # fall back to the YAML file in the default editor


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a desktop session
    import pystray

    from away.app.runner import AwayApp

    cfg, _ = cfgmod.load()
    app = AwayApp(cfg)
    ctl = TrayController(app)
    ctl.paused = not cfg.app.always_on  # app.always_on: false starts paused (and muted)
    app.start()
    if not cfg.app.start_minimized:
        open_settings(ctl)

    def refresh(icon: Any) -> None:
        icon.icon = make_icon_image(ctl.paused, state=ctl.icon_state())
        icon.title = ctl.title()

    def on_pause(icon: Any, _item: Any) -> None:
        ctl.toggle_pause()
        refresh(icon)

    def on_dictation(icon: Any, _item: Any) -> None:
        ctl.toggle_dictation()
        refresh(icon)

    def on_mic(icon: Any, _item: Any) -> None:
        ctl.toggle_mic()
        refresh(icon)

    def on_settings(_icon: Any, _item: Any) -> None:
        open_settings(ctl)

    def on_edit_yaml(_icon: Any, _item: Any) -> None:
        open_path(ctl.settings_path())

    def on_quit(icon: Any, _item: Any) -> None:
        ctl.quit()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem(lambda _i: ctl.title(), None, enabled=False),
        pystray.MenuItem(lambda _i: "Resume" if ctl.paused else "Pause", on_pause, default=True),
        pystray.MenuItem("Mic on", on_mic, checked=lambda _i: ctl.mic_on()),
        pystray.MenuItem(
            lambda _i: "Start listening" if ctl.voice_state() == "stopped" else "Stop listening / dictation",
            on_dictation,
        ),
        pystray.MenuItem("Settings...", on_settings),
        pystray.MenuItem("Edit config file", on_edit_yaml),
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon("project-away", make_icon_image(ctl.paused, state=ctl.icon_state()), ctl.title(), menu)
    for topic in ("voice.state", "app.paused", "gesture.clutch"):
        app.bus.subscribe(topic, lambda _e: _sync(icon))

    def _sync(ic: Any) -> None:
        ctl.paused = bool(app.pipeline.paused)
        refresh(ic)
        ic.update_menu()

    icon.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
