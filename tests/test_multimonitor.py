"""v0.1.6: the gesture cursor reaches every monitor (geometry, enumeration, config, jump actions, voice)."""

from __future__ import annotations

import ctypes
import types
from typing import Any, ClassVar

import pytest

from away import config as C
from away.actions import ActionExecutor, Dispatcher
from away.bindings import BindingsConfig, parse_action, phrase_variants
from away.config import CursorConfig
from away.control import monitors as M
from away.control.monitors import Layout, Monitor
from away.control.mouse import MouseController, RecordingBackend
from away.hands.cursor import CursorMapper, reach_factor

THREE = [Monitor(0, 0, 1920, 1080, True, "A"), Monitor(1920, 0, 1920, 1080), Monitor(3840, 0, 1920, 1080)]


def cfg(**kw: Any) -> CursorConfig:
    base: dict[str, Any] = dict(min_cutoff=1e6, beta=0.0, dead_zone=0.0, acceleration=1.0, curve="power")
    base.update(kw)
    return CursorConfig(**base)


def drive(m: CursorMapper, steps: list[tuple[float, float]]) -> tuple[int, int]:
    t = 0.0
    out = m.update(t, 0.5, 0.5)
    x, y = 0.5, 0.5
    for dx, dy in steps:
        t += 0.033
        x, y = x + dx, y + dy
        out = m.update(t, x, y)
    return out


# ------------------------------------------------------------------------------------------ geometry
def test_three_side_by_side_bounds_and_numbering():
    lay = Layout(M.sort_monitors(reversed(THREE)), "test")
    assert lay.bounds == (0, 0, 5760, 1080)
    assert [m.left for m in lay.monitors] == [0, 1920, 3840]  # numbered left to right
    assert lay.primary.name == "A"
    assert "3 monitors" in lay.lines()[0] and "virtual desktop: 5760x1080 at (0, 0)" in "\n".join(lay.lines())


def test_monitor_left_of_primary_has_negative_origin():
    mons = M.sort_monitors([Monitor(0, 0, 1920, 1080, True), Monitor(-2560, -360, 2560, 1440)])
    assert M.virtual_bounds(mons) == (-2560, -360, 4480, 1440)
    m = CursorMapper(mons, cfg(gain=1.0))
    assert m.pos == (960.0, 540.0)  # starts on the primary
    x, _ = drive(m, [(-0.2, 0.0)] * 10)  # sweep left across onto the other monitor
    assert x < 0
    assert M.clamp_to_monitors(-5000, 0, mons) == (-2560.0, 0.0)


def test_stacked_vertically():
    mons = M.sort_monitors([Monitor(0, 0, 1920, 1080, True), Monitor(0, -1080, 1920, 1080)])
    assert M.virtual_bounds(mons) == (0, -1080, 1920, 2160)
    m = CursorMapper(mons, cfg(gain=1.0))
    _, y = drive(m, [(0.0, -0.2)] * 10)
    assert y == -1080  # top edge of the upper monitor


def test_mixed_sizes_dead_zone_is_clamped():
    # 1920x1080 primary next to a taller 2560x1440: the area right of x=1920 below y=1080 is not dead,
    # but below the primary (x<1920, y>=1080) is
    mons = M.sort_monitors([Monitor(0, 0, 1920, 1080, True), Monitor(1920, 0, 2560, 1440)])
    assert M.clamp_to_monitors(1900, 1300, mons) == (1920.0, 1300.0)  # nearest is the tall monitor's edge
    assert M.clamp_to_monitors(500, 1300, mons) == (500.0, 1079.0)  # straight up onto the primary
    for x, y in [(1900, 1300), (500, 1300), (-50, -50), (9999, 9999)]:
        cx, cy = M.clamp_to_monitors(x, y, mons)
        assert any(mm.contains(cx, cy) for mm in mons)
    m = CursorMapper(mons, cfg(gain=1.0))
    m.set_position(1900, 1300)
    assert any(mm.contains(*m.pos) for mm in mons)


def test_single_monitor_behaves_as_before():
    old = CursorMapper((1000, 800), cfg(gain=2.0))
    assert (old.w, old.h, old.x0, old.y0, old.reach) == (1000, 800, 0, 0, 1.0)
    old.update(0.0, 0.5, 0.5)
    assert old.update(0.1, 0.51, 0.5) == (520, 400)
    with pytest.raises(ValueError):
        CursorMapper((0, 10))


def test_reach_scales_fast_moves_only():
    one = CursorMapper((1920, 1080), cfg(gain=1.0, acceleration=2.0))
    three = CursorMapper(THREE, cfg(gain=1.0, acceleration=2.0))
    assert three.reach == pytest.approx(3**0.5)
    assert reach_factor(THREE, THREE[0], 0.0) == 1.0 and reach_factor(THREE, THREE[0], 1.0) == pytest.approx(3.0)
    # slow move (below the acceleration knee): identical pixels on 1 or 3 monitors (precision kept)
    a = drive(one, [(0.005, 0.0)])[0] - 960
    b = drive(three, [(0.005, 0.0)])[0] - 960
    assert a == b
    # fast move: further on the 3-monitor desktop, but less than 3x
    one2 = CursorMapper((1920, 1080), cfg(gain=1.0, acceleration=2.0))
    three2 = CursorMapper(THREE, cfg(gain=1.0, acceleration=2.0))
    fa = drive(one2, [(0.03, 0.0)])[0] - 960
    fb = drive(three2, [(0.03, 0.0)])[0] - 960
    assert fa < fb < 3 * fa


def test_sigmoid_reach_keeps_low_plateau():
    c = cfg(curve="sigmoid", gain=1.0, sigmoid_low=0.6, sigmoid_high=3.0)
    one, three = CursorMapper((1920, 1080), c), CursorMapper(THREE, c)
    assert drive(one, [(0.001, 0.0)]) == drive(three, [(0.001, 0.0)])
    one2, three2 = CursorMapper((1920, 1080), c), CursorMapper(THREE, c)
    assert drive(three2, [(0.05, 0.0)])[0] > drive(one2, [(0.05, 0.0)])[0]


def test_absolute_mode_spans_virtual_desktop_with_negative_origin():
    mons = [Monitor(0, 0, 1000, 1000, True), Monitor(-1000, 0, 1000, 1000)]
    m = CursorMapper(mons, cfg(mode="absolute", active_region=(0.0, 0.0, 1.0, 1.0)))
    assert m.update(0.0, 0.0, 0.0) == (-1000, 0)
    m.reset()
    assert m.update(1.0, 1.0, 1.0) == (999, 999)


# --------------------------------------------------------------------------------------- selection
def test_parse_monitor_spec_and_selection():
    assert M.parse_monitor_spec("all") == "all" and M.parse_monitor_spec(None) == "all"
    assert M.parse_monitor_spec("Primary") == "primary"
    assert M.parse_monitor_spec([3, 1, 3]) == [3, 1] and M.parse_monitor_spec(2) == [2]
    assert M.parse_monitor_spec("1, 3") == [1, 3]
    for bad in ("left", [0], [True], [], 1.5):
        with pytest.raises(ValueError):
            M.parse_monitor_spec(bad)
    lay = Layout(tuple(THREE), "t")
    assert M.select_monitors(lay, "primary") == (THREE[0],)
    assert M.select_monitors(lay, [2, 3]) == (THREE[1], THREE[2])
    assert M.select_monitors(lay, [7]) == tuple(THREE)  # unplugged -> all


def test_config_monitors_key(tmp_path):
    cfg_, unknown = C.from_dict({"hands": {"cursor": {"monitors": [1, 3], "monitor_reach": 0.25}}})
    assert unknown == [] and cfg_.hands.cursor.monitors == [1, 3] and cfg_.hands.cursor.monitor_reach == 0.25
    assert C.AppConfig().hands.cursor.monitors == "all"
    with pytest.raises(ValueError):
        C.from_dict({"hands": {"cursor": {"monitors": "left"}}})
    with pytest.raises(ValueError):
        C.from_dict({"hands": {"cursor": {"monitor_refresh_s": -1}}})
    with pytest.raises(ValueError):
        C.from_dict({"hands": {"cursor": {"monitor_reach": 2}}})
    p = C.save(cfg_, tmp_path / "c.yaml")
    assert C.load(p)[0].hands.cursor.monitors == [1, 3]


def test_mapper_respects_monitor_selection():
    m = CursorMapper(THREE, cfg(monitors=[2, 3]))
    assert m.monitors == (THREE[1], THREE[2]) and m.x0 == 1920
    m.set_position(100, 100)
    assert m.pos == (1920.0, 100.0)
    m.cfg.monitors = "all"
    m.refresh_selection()
    assert len(m.monitors) == 3


# ------------------------------------------------------------------------------------- enumeration
class FakeUser32:
    """EnumDisplayMonitors / GetMonitorInfoW with three monitors, one left of the primary."""

    RECTS: ClassVar[list[tuple[int, int, int, int, int, str]]] = [
        (-1920, 0, 0, 1080, 0, "\\\\.\\DISPLAY2"),
        (0, 0, 2560, 1440, 1, "\\\\.\\DISPLAY1"),
        (2560, 0, 4480, 1080, 0, "D3"),
    ]

    def EnumDisplayMonitors(self, hdc: Any, clip: Any, cb: Any, data: Any) -> int:
        for i in range(len(self.RECTS)):
            cb(i + 1, None, None, 0)
        return 1

    def GetMonitorInfoW(self, h: int, ptr: Any) -> int:
        info = ptr.contents
        left, top, right, bottom, flags, name = self.RECTS[h - 1]
        info.rcMonitor.left, info.rcMonitor.top, info.rcMonitor.right, info.rcMonitor.bottom = left, top, right, bottom
        info.dwFlags = flags
        info.szDevice = name
        return 1


def test_win32_enumeration_mocked():
    try:
        from ctypes import wintypes  # noqa: F401
    except (ImportError, ValueError):  # pragma: no cover - very old non-Windows Pythons
        pytest.skip("ctypes.wintypes unavailable")
    mons = M.sort_monitors(M._win32(FakeUser32()))
    assert [(m.left, m.width, m.primary) for m in mons] == [(-1920, 1920, False), (0, 2560, True), (2560, 1920, False)]
    assert mons[1].name == "\\\\.\\DISPLAY1"
    lay = M.enumerate_monitors([("win32", lambda: M._win32(FakeUser32()))])
    assert lay.source == "win32" and lay.bounds == (-1920, 0, 6400, 1440) and lay.note == ""


def test_enumeration_failure_falls_back_to_primary_only():
    def boom() -> list[Monitor]:
        raise OSError("no display")

    lay = M.enumerate_monitors([("a", boom), ("b", lambda: [])])
    assert lay.source == "fallback" and len(lay.monitors) == 1 and lay.primary.width == 1920
    assert "no display" in lay.note and "b: no monitors reported" in lay.note
    lay2 = M.enumerate_monitors([("a", boom), ("tkinter", lambda: [Monitor(0, 0, 1366, 768, True)])])
    assert lay2.source == "tkinter" and "only the primary" in lay2.note
    m = CursorMapper(lay, cfg())
    assert (m.w, m.h) == (1920, 1080)


def test_parse_xrandr_and_appkit_flip():
    text = (
        "Monitors: 3\n 0: +*eDP-1 1920/344x1080/194+0+0  eDP-1\n 1: +HDMI-1 2560/597x1440/336+1920+0  HDMI-1\n"
        " 2: +DP-1 1920/500x1080/300+0+1080  DP-1\n"
    )
    mons = M.parse_xrandr(text)
    assert [(m.left, m.top, m.width, m.height, m.primary, m.name) for m in mons] == [
        (0, 0, 1920, 1080, True, "eDP-1"),
        (1920, 0, 2560, 1440, False, "HDMI-1"),
        (0, 1080, 1920, 1080, False, "DP-1"),
    ]
    run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=text)  # noqa: E731
    import os

    os.environ["DISPLAY"] = ":0"
    try:
        assert len(M._xrandr(run)) == 3
    finally:
        del os.environ["DISPLAY"]
    # AppKit: primary 1440 tall at origin, second above it (y up) -> negative top in pynput space
    flipped = M.appkit_to_quartz([(0, 0, 2560, 1440), (0, 1440, 1920, 1080)])
    assert (flipped[1].left, flipped[1].top) == (0, -1080) and flipped[0].primary


def test_monitor_watcher_detects_hotplug_and_keeps_good_layout():
    one = Layout((THREE[0],), "win32")
    three = Layout(tuple(THREE), "win32")
    fall = M.single(1920, 1080)
    results = [one, three, fall]
    w = M.MonitorWatcher(one, 3.0, detect=lambda: results.pop(0))
    assert w.poll(0.0) is None  # unchanged
    assert w.poll(1.0) is None  # not due yet
    assert w.poll(3.5) == three
    assert w.poll(7.0) is None and w.layout == three  # transient failure keeps the 3 monitors
    assert not M.MonitorWatcher(M.single(800, 600, source="tkinter")).enabled
    assert not M.MonitorWatcher(three, 0).enabled


def test_pipeline_set_monitors_keeps_cursor_on_screen():
    from away.app.runner import Pipeline

    p = Pipeline(C.AppConfig(), screen=Layout(tuple(THREE), "t"))
    p.mouse.mapper.set_position(5000, 500)
    p.set_monitors(Layout((THREE[0],), "t"))
    assert p.mouse.mapper.pos == (1919.0, 500.0)


def test_dpi_awareness_fallback_chain():
    class U:
        def __init__(self, ctx: bool, legacy: bool = True) -> None:
            self.ctx, self.legacy = ctx, legacy
            self.calls: list[str] = []

        def SetProcessDpiAwarenessContext(self, v: Any) -> int:
            self.calls.append("ctx")
            assert v.value == ctypes.c_void_p(-4).value
            return int(self.ctx)

        def SetProcessDPIAware(self) -> int:
            self.calls.append("legacy")
            return int(self.legacy)

    class Sh:
        def __init__(self, hr: int) -> None:
            self.hr = hr

        def SetProcessDpiAwareness(self, v: int) -> int:
            assert v == 2
            return self.hr

    assert M.enable_dpi_awareness(U(True), Sh(0), "win32", force=True) == "per_monitor_v2"
    assert M.enable_dpi_awareness(U(False), Sh(0), "win32", force=True) == "per_monitor"
    assert M.enable_dpi_awareness(U(False), Sh(M.E_ACCESSDENIED), "win32", force=True) == "already"
    assert M.enable_dpi_awareness(U(False), Sh(1), "win32", force=True) == "system"
    assert M.enable_dpi_awareness(U(False, False), Sh(1), "win32", force=True) == "failed"
    assert M.enable_dpi_awareness(platform="linux", force=True) == "unsupported"
    assert M.enable_dpi_awareness(platform="linux") == "unsupported"  # cached


# ------------------------------------------------------------------------------- jump actions/voice
def mouse(mons: Any = THREE) -> tuple[MouseController, RecordingBackend]:
    b = RecordingBackend()
    return MouseController(b, CursorMapper(mons, cfg())), b


def test_cursor_action_parsing():
    assert str(parse_action("cursor:next_monitor")) == "cursor:next_monitor"
    assert str(parse_action("cursor:Previous Monitor")) == "cursor:prev_monitor"
    assert str(parse_action("cursor:monitor_2")) == "cursor:monitor 2"
    assert str(parse_action("cursor:monitor three")) == "cursor:monitor 3"
    assert str(parse_action({"cursor": "monitor2"})) == "cursor:monitor 2"
    for bad in ("cursor:monitor 12", "cursor:sideways", "cursor"):
        with pytest.raises(ValueError):
            parse_action(bad)


def test_monitor_jump_actions():
    m, b = mouse()
    ex = ActionExecutor(m)
    assert ex.run(parse_action("cursor:next_monitor")) and b.calls[-1] == ("move", (2880, 540))
    assert ex.run(parse_action("cursor:next_monitor")) and b.calls[-1] == ("move", (4800, 540))
    assert ex.run(parse_action("cursor:next_monitor")) and b.calls[-1] == ("move", (960, 540))  # wraps
    assert ex.run(parse_action("cursor:prev_monitor")) and b.calls[-1] == ("move", (4800, 540))
    assert ex.run(parse_action("cursor:monitor 2")) and b.calls[-1] == ("move", (2880, 540))
    assert ex.run(parse_action("cursor:primary_monitor")) and b.calls[-1] == ("move", (960, 540))
    assert not ex.run(parse_action("cursor:monitor 5"))  # not connected
    m.mapper.cfg.monitors = [1, 2]
    m.mapper.refresh_selection()
    assert not ex.run(parse_action("cursor:monitor 3"))  # excluded by hands.cursor.monitors
    m.enabled = False
    assert not ex.run(parse_action("cursor:monitor 1"))


def test_monitor_jump_reanchors_relative_motion():
    m, b = mouse()
    m.mapper.update(0.0, 0.5, 0.5)
    m.jump_monitor("monitor 3")
    assert m.mapper._prev is None  # next hand frame continues from the new spot, no jump back


def test_default_voice_commands_jump_monitors():
    m, b = mouse()
    d = Dispatcher(BindingsConfig(), ActionExecutor(m))
    assert str(d.trigger("voice", "Next monitor.")) == "cursor:next_monitor"
    assert b.calls[-1] == ("move", (2880, 540))
    assert str(d.trigger("voice", "previous monitor")) == "cursor:prev_monitor"
    assert str(d.trigger("voice", "monitor three")) == "cursor:monitor 3" and b.calls[-1] == ("move", (4800, 540))
    assert str(d.trigger("voice", "Monitor 2")) == "cursor:monitor 2"  # Whisper may write the digit
    paused = Dispatcher(BindingsConfig(), ActionExecutor(m), paused=lambda: True)
    assert paused.trigger("voice", "monitor one").kind == "none"


def test_voice_phrases_include_digit_spellings_and_binding():
    b = BindingsConfig()
    ph = b.voice_phrases()
    assert str(ph["monitor one"]) == "cursor:monitor 1" and str(ph["monitor 1"]) == "cursor:monitor 1"
    assert phrase_variants("monitor 2") == ["monitor two"] and phrase_variants("scroll down") == []
    b.set("gesture:swipe_left", "cursor:prev_monitor")  # bindable from any trigger
    b.set("hotkey:ctrl+alt+right", "cursor:next_monitor")
    assert str(b.resolve("gestures", "swipe_left")) == "cursor:prev_monitor"
    from away.app.settings_gui import SettingsModel

    assert "cursor" in SettingsModel.action_kinds()


def test_monitors_cli_and_selftest(capsys, monkeypatch):
    from away import cli
    from away.app import selftest

    lay = Layout(tuple(THREE), "win32")
    monkeypatch.setattr(M, "enumerate_monitors", lambda *a, **k: lay)
    assert cli.main(["monitors"]) == 0
    out = capsys.readouterr().out
    assert "3 monitors (source: win32)" in out and "next monitor" in out
    assert cli.main(["monitors", "--json"]) == 0
    assert '"cursor_monitors": [' in capsys.readouterr().out
    res = selftest.run([c for c in selftest.CHECKS if c.name == "monitors"])
    assert res["monitors"]["ok"] is True and "3 via win32" in str(res["monitors"]["detail"])
