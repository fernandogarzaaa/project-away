"""System tray app (pystray): start/pause, dictation toggle, settings, quit."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from away import config as cfgmod
from away import paths


def make_icon_image(paused: bool = False, size: int = 64) -> Any:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (140, 140, 140, 255) if paused else (46, 160, 67, 255)
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

    def quit(self) -> None:
        self.app.stop()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a desktop session
    import pystray

    from away.app.runner import AwayApp

    cfg, _ = cfgmod.load()
    app = AwayApp(cfg)
    ctl = TrayController(app)
    app.start()

    def refresh(icon: Any) -> None:
        icon.icon = make_icon_image(ctl.paused)
        icon.title = "Project Away (paused)" if ctl.paused else "Project Away"

    def on_pause(icon: Any, _item: Any) -> None:
        ctl.toggle_pause()
        refresh(icon)

    def on_dictation(icon: Any, _item: Any) -> None:
        ctl.toggle_dictation()

    def on_settings(_icon: Any, _item: Any) -> None:
        open_path(ctl.settings_path())

    def on_quit(icon: Any, _item: Any) -> None:
        ctl.quit()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem(lambda _i: "Resume" if ctl.paused else "Pause", on_pause, default=True),
        pystray.MenuItem("Dictation enabled", on_dictation, checked=lambda _i: ctl.dictation_enabled()),
        pystray.MenuItem("Open settings", on_settings),
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon("project-away", make_icon_image(), "Project Away", menu)
    icon.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
