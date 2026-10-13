"""v0.1.10: Settings window layout on high-DPI Windows screens (clipped buttons, cramped status, no scroll)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from away.app import settings_gui as G

USER_CFG = r"C:\Users\garza\AppData\Local\ProjectAway\config.yaml"


def test_saved_status_is_short_and_has_no_path() -> None:
    msg = G.status_text("saved", USER_CFG)
    assert msg == "Saved. Changes apply within a second."
    assert "C:\\" not in msg and len(msg) < 45


def test_status_paths_are_shortened_to_the_tail() -> None:
    assert G.short_path(USER_CFG) == "…\\ProjectAway\\config.yaml"
    assert G.short_path("/home/someone-long/.config/project-away/config.yaml") == "…/project-away/config.yaml"
    assert G.short_path(r"C:\x\config.yaml") == r"C:\x\config.yaml"  # short paths stay whole
    long_name = "C:\\a\\" + "b" * 60 + "\\bindings.yaml"
    assert G.short_path(long_name) == "…\\bindings.yaml"
    assert G.status_text("config", USER_CFG) == "Config file: …\\ProjectAway\\config.yaml"
    assert G.status_text("config", USER_CFG, dirty=True).startswith("Unsaved changes. Config file: ")
    assert G.status_text("imported", r"C:\Users\garza\Desktop\my bindings.yaml") == "Imported my bindings.yaml."
    assert G.status_text("exported", r"C:\Users\garza\Documents\Project Away\exports\bindings.yaml") == (
        "Exported to …\\exports\\bindings.yaml."
    )
    for kind in ("saved", "config", "imported", "exported"):
        assert len(G.status_text(kind, USER_CFG, True)) <= 60


def test_tk_scaling_and_ui_scale_follow_the_real_dpi() -> None:
    assert G.tk_scaling(96) == pytest.approx(1.3333, abs=1e-3)
    assert G.tk_scaling(120) == pytest.approx(1.6667, abs=1e-3)  # 125 %
    assert G.tk_scaling(144) == pytest.approx(2.0)  # 150 %
    assert G.tk_scaling(0) == G.tk_scaling(96)  # unknown -> 96 dpi
    assert G.ui_scale(144) == 1.5 and G.ui_scale(120) == 1.25 and G.ui_scale(72) == 1.0


@pytest.mark.parametrize(
    ("screen", "scale"),
    [((1366, 768), 1.0), ((1920, 1080), 1.0), ((1920, 1080), 1.25), ((1920, 1080), 1.5), ((1280, 720), 1.5)],
)
def test_window_opens_inside_the_screen_with_room_for_the_taskbar(screen: tuple[int, int], scale: float) -> None:
    sw, sh = screen
    w, h, x, y = G.window_geometry(sw, sh, scale)
    assert x >= 0 and x + w <= sw and y >= 0 and y + h <= sh
    assert h <= int(sh * 0.88)  # the bottom button bar is never under the taskbar
    mw, mh = G.min_window_size(sw, sh, scale)
    assert mw <= w and mh <= h and mw >= int(G.MIN_SIZE[0] * min(scale, 1.0))


def test_default_size_on_a_1366x768_screen() -> None:
    assert G.window_geometry(1366, 768, 1.0)[:2] == (900, 640)
    assert G.window_geometry(1920, 1080, 1.5)[:2] == (1350, 950)  # the user's primary monitor at 150 %


def test_wheel_steps_per_platform() -> None:
    assert G.wheel_steps(120, platform="win32") == -1  # wheel up = scroll up
    assert G.wheel_steps(-360, platform="win32") == 3
    assert G.wheel_steps(30, platform="win32") == -1  # high-resolution wheel: still one step
    assert G.wheel_steps(-2, platform="darwin") == 2
    assert G.wheel_steps(num=4, platform="linux") == -1 and G.wheel_steps(num=5, platform="linux") == 1
    assert G.wheel_steps(0, platform="win32") == 0


def test_status_wraps_to_the_window_and_scroll_frame_fills_the_tab() -> None:
    assert G.status_wraplength(900) == 884 and G.status_wraplength(50) == 120
    assert G.scroll_window_height(500, 300) == 500  # content fits: stretch to the visible height
    assert G.scroll_window_height(500, 900) == 900  # content taller: scroll


def test_readable_font_and_button_padding() -> None:
    assert 9 <= G.FONT_PT <= 11
    px, py = G.BUTTON_PADDING
    assert px >= 6 and py >= 3


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and os.environ.get("DISPLAY")), reason="needs an X display (Xvfb)"
)
def test_settings_window_bottom_bar_visible_and_tabs_scroll(tmp_path: Path) -> None:
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - no Tk
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        w = G.SettingsWindow(root, G.SettingsModel(path=tmp_path / "config.yaml"))
        root.geometry("700x450")
        root.update()
        buttons = [b for b in root.winfo_children() if b.winfo_class() == "TFrame" for b in b.winfo_children()]
        labels = {b.cget("text"): b for b in buttons if b.winfo_class() == "TButton"}
        for name in ("Save", "Reset profile", "Reset all", "Import...", "Export...", "Close"):
            b = labels[name]
            assert b.winfo_ismapped() and b.winfo_rooty() + b.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            assert b.winfo_height() >= b.winfo_reqheight()  # not squashed: the label is never clipped
        assert len(w._scrollers) == 7  # every tab scrolls (v0.1.20: + Agent Link)
        assert w.status.get().startswith("Config file: ")
    finally:
        root.destroy()
