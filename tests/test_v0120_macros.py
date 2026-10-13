"""v0.1.20 shortcuts (teach-by-demonstration macros): model, recording, replay safety, bindings, panel, CLI.

Every input backend is a fake: the window API, the listener and the replay backend are injected."""

from __future__ import annotations

import argparse
import json
import os
import threading
from typing import Any, ClassVar

import pytest

from away import config as C
from away.actions import ActionExecutor, Dispatcher
from away.app.macro_panel import Problem, ShortcutsModel, StepEditor, action_label
from away.app.macro_service import STATUS_FILE, DryRunBackend, MacroService
from away.app.settings_gui import SettingsModel
from away.app.tray import TrayController
from away.bindings import Action, BindingsConfig, parse_action, parse_bindings
from away.control.macro_platform import ElementInfo, NullWindowApi, WindowInfo
from away.control.macro_play import AbortWatcher, MacroPlayer
from away.control.macro_record import MacroRecorder, MarkingBackend, SelfInputMarker
from away.control.macros import (
    Macro,
    MacroError,
    MacrosConfig,
    MacroStore,
    RawEvent,
    SensitiveExportError,
    Step,
    Target,
    build_steps,
    name_from_title,
    validate_macros_config,
)
from away.control.mouse import MouseController, RecordingBackend
from away.hands.cursor import CursorMapper


# --------------------------------------------------------------------------- fakes
class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeWin:
    """A desktop with windows (topmost first), a focused app and an optional password box."""

    can_locate = True
    can_check_password = True

    def __init__(self, windows: list[WindowInfo] | None = None, focused: str = "notepad.exe") -> None:
        self.windows = windows or []
        self.focused = focused
        self.password = False
        self.elements: dict[str, ElementInfo] = {}  # name -> element (find_element)
        self.at: dict[tuple[int, int], ElementInfo] = {}  # point -> element (element_at)

    def window_at(self, x: int, y: int) -> WindowInfo | None:
        for w in self.windows:
            if w.rect[0] <= x < w.rect[2] and w.rect[1] <= y < w.rect[3]:
                return w
        return None

    def foreground(self) -> WindowInfo | None:
        hit = next((w for w in self.windows if w.process == self.focused), None)
        return hit or WindowInfo(0, 0, self.focused, "", "", (0, 0, 0, 0))

    def find_window(self, process: str, title: str = "") -> WindowInfo | None:
        return next((w for w in self.windows if w.process.lower() == process.lower()), None)

    def element_at(self, x: int, y: int) -> ElementInfo | None:
        return self.at.get((x, y))

    def find_element(self, window: WindowInfo, name: str, control_type: int = 0) -> ElementInfo | None:
        return self.elements.get(name)

    def focused_is_password(self) -> bool | None:
        return self.password


def win(process: str, rect: tuple[int, int, int, int], pid: int = 50, cls: str = "App") -> WindowInfo:
    return WindowInfo(1, pid, process, process, cls, rect)


class FakeBackend:
    def __init__(self, fail_on: str = "", on_call: Any = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail_on = fail_on
        self.on_call = on_call

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*a: Any) -> None:
            self.calls.append((name, *a))
            if self.on_call is not None:
                self.on_call(name, a)
            if name == self.fail_on:
                raise OSError("input blocked")

        return call


class FakeListener:
    instances: ClassVar[list[FakeListener]] = []

    def __init__(self, sink: Any) -> None:
        self.sink = sink
        self.started = self.stopped = False
        FakeListener.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class Sleeper:
    """Records requested pauses instead of sleeping; ``abort_on`` makes the n-th pause report an abort."""

    def __init__(self, abort_on: int = -1) -> None:
        self.total = 0.0
        self.n = 0
        self.abort_on = abort_on

    def __call__(self, ev: threading.Event, s: float) -> bool:
        self.total += s
        self.n += 1
        if self.n == self.abort_on:
            ev.set()
        return ev.is_set()


def click_macro(app: str = "chrome.exe", **kw: Any) -> Macro:
    t = Target(110, 220, rx=10, ry=20, app=app, window=app)
    return Macro("open_mail", "Open mail", [Step("click", target=t, **kw)], app=app)


# --------------------------------------------------------------------------- build_steps
def ev(t: float, kind: str, **kw: Any) -> RawEvent:
    return RawEvent(t, kind, **kw)


def test_click_double_click_and_drag_become_steps():
    tgt = Target(10, 10, app="x.exe")
    steps = build_steps(
        [
            ev(0.0, "down", x=10, y=10, target=tgt),
            ev(0.05, "up", x=11, y=10),
            ev(0.2, "down", x=10, y=11, target=tgt),
            ev(0.25, "up", x=10, y=11),
            ev(1.0, "down", x=100, y=100, target=Target(100, 100)),
            ev(1.1, "move", x=120, y=100),
            ev(1.2, "move", x=160, y=130),
            ev(1.3, "up", x=200, y=150),
        ]
    )
    assert [s.type for s in steps] == ["click", "drag"]
    assert steps[0].count == 2
    assert steps[1].path[-1] == (100, 50) and len(steps[1].path) == 3
    assert steps[1].duration == pytest.approx(0.3)
    assert steps[1].delay == pytest.approx(0.75)


def test_typing_shortcuts_and_lone_modifier():
    steps = build_steps(
        [
            ev(0.0, "key_down", key="shift"),
            ev(0.1, "key_down", key="h", char="H"),
            ev(0.15, "key_up", key="shift"),
            ev(0.2, "key_down", key="i", char="i"),
            ev(0.3, "key_down", key="space"),
            ev(0.4, "key_down", key="enter"),
            ev(0.5, "key_down", key="ctrl_l"),
            ev(0.6, "key_down", key="c"),
            ev(0.7, "key_up", key="ctrl_l"),
            ev(0.8, "key_down", key="cmd"),
            ev(0.9, "key_up", key="cmd"),
        ]
    )
    assert [(s.type, s.text or s.keys) for s in steps] == [
        ("text", "Hi "),
        ("key", "enter"),
        ("key", "ctrl+c"),
        ("key", "cmd"),
    ]


def test_ctrl_click_keeps_the_modifier():
    steps = build_steps(
        [
            ev(0.0, "key_down", key="ctrl"),
            ev(0.1, "down", x=5, y=5),
            ev(0.2, "up", x=5, y=5),
            ev(0.3, "key_up", key="ctrl"),
        ]
    )
    assert [(s.type, s.modifiers) for s in steps] == [("click", ["ctrl"])]


def test_record_text_off_keeps_keys_but_drops_characters():
    steps = build_steps(
        [
            ev(0.0, "key_down", key="a"),
            ev(0.1, "key_down", key="shift"),
            ev(0.15, "key_down", key="b"),
            ev(0.2, "key_up", key="shift"),
            ev(0.3, "key_down", key="tab"),
            ev(0.4, "key_down", key="ctrl"),
            ev(0.5, "key_down", key="s"),
            ev(0.6, "key_up", key="ctrl"),
        ],
        record_text=False,
    )
    assert [s.keys for s in steps] == ["tab", "ctrl+s"]


def test_long_pauses_are_capped_and_scrolls_merge():
    t = Target(5, 5)
    steps = build_steps(
        [
            ev(0.0, "scroll", x=5, y=5, dy=-1, target=t),
            ev(0.1, "scroll", x=6, y=5, dy=-1, target=t),
            ev(0.2, "scroll", x=6, y=6, dy=-1, target=t),
            ev(30.0, "key_down", key="f5"),
        ],
        max_delay=2.0,
    )
    assert [(s.type, s.dy) for s in steps] == [("scroll", -3), ("key", 0)]
    assert steps[1].delay == 2.0


# --------------------------------------------------------------------------- model + store
def test_macro_round_trip_sensitive_and_describe_never_shows_text():
    steps = [click_macro().steps[0], Step("text", text="hunter2"), Step("key", keys="Enter")]
    m = Macro("login_flow", "Login flow", steps)
    m.validate()
    back = Macro.from_dict(json.loads(json.dumps(m.to_dict())))
    assert back == m
    assert m.sensitive and m.to_dict()["sensitive"] is True
    assert "hunter2" not in m.steps[1].describe() and "7 characters" in str(m.steps[1])
    assert back.steps[2].keys == "enter"
    assert not Macro("x", steps=[Step("key", keys="ctrl+c")]).sensitive
    assert Macro("x", steps=[Step("key", keys="shift+a")]).sensitive
    assert m.steps[0].describe() == "Click in Chrome.exe" or m.steps[0].describe() == "Click in Chrome"


@pytest.mark.parametrize(
    "bad, words",
    [
        ({"format": 99, "steps": []}, "newer Project Away"),
        ({"name": "x", "steps": [{"type": "fly"}]}, "unknown step type"),
        ({"name": "x", "steps": [{"type": "click"}]}, "click position is missing"),
        ({"name": "x", "steps": [{"type": "key", "keys": "ctrl+nope"}]}, "unknown key"),
        ({"name": "x", "steps": [{"type": "wait", "seconds": 0}]}, "wait must be"),
        ({"name": "x", "steps": []}, "no steps"),
        ({"name": "Bad Name", "steps": [{"type": "key", "keys": "a"}]}, "can't be used"),
    ],
)
def test_invalid_shortcuts_fail_with_friendly_messages(bad, words):
    with pytest.raises(MacroError, match=words):
        Macro.from_dict(bad)


def test_names_from_titles():
    assert name_from_title("Open my mail!") == "open_my_mail"
    assert name_from_title("2nd tab") == "shortcut_2nd_tab"
    assert name_from_title("!!!") == ""


def test_store_crud_unique_names_and_private_files(tmp_path):
    st = MacroStore(tmp_path / "m")
    assert st.names() == [] and st.next_untitled() == ("shortcut_1", "Shortcut 1")
    m = click_macro()
    st.save(m)
    assert st.names() == ["open_mail"] and st.load("open_mail") == m
    assert st.unique_name("open_mail") == "open_mail_2"
    st.rename("open_mail", "mail", "Mail")
    assert st.names() == ["mail"] and st.load("mail").title == "Mail"
    with pytest.raises(MacroError, match="no shortcut called"):
        st.load("open_mail")
    p = st.save(Macro("pw", steps=[Step("text", text="secret")]))
    if os.name != "nt":
        assert (p.stat().st_mode & 0o777) == 0o600
    (st.folder / "broken.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(MacroError, match="damaged"):
        st.load("broken")
    assert [x.name for x in st.all()] == ["mail", "pw"]  # the damaged one is skipped
    assert st.delete("mail") and not st.delete("mail")


def test_export_keeps_private_shortcuts_unless_asked_and_import_never_overwrites(tmp_path):
    st = MacroStore(tmp_path / "a")
    st.save(click_macro())
    st.save(Macro("say_hi", steps=[Step("text", text="hi")]))
    with pytest.raises(SensitiveExportError, match="stays on this PC"):
        st.export(["open_mail", "say_hi"], tmp_path / "out.json")
    st.export(["open_mail"], tmp_path / "out.json")
    st.export(["say_hi"], tmp_path / "private.json", include_sensitive=True)
    other = MacroStore(tmp_path / "b")
    other.save(click_macro())
    got = other.import_file(tmp_path / "out.json")
    assert [m.name for m in got] == ["open_mail_2"] and len(other.names()) == 2
    assert [m.name for m in other.import_file(tmp_path / "out.json", overwrite=True)] == ["open_mail"]
    (tmp_path / "junk.json").write_text('{"shortcuts": [{"name": "x", "steps": [{"type": "fly"}]}]}')
    with pytest.raises(MacroError):
        other.import_file(tmp_path / "junk.json")
    assert len(other.names()) == 2  # nothing written from an invalid file


def test_config_section_validates_and_round_trips():
    cfg, unknown = C.from_dict({"macros": {"abort_hotkey": "Ctrl+Alt+X", "default_speed": "2x"}})
    assert unknown == [] and cfg.macros.abort_hotkey == "ctrl+alt+x"
    again, _ = C.from_dict(C.to_dict(cfg))
    assert again.macros == cfg.macros
    for bad in ({"default_speed": "9x"}, {"abort_hotkey": "nope+x"}, {"max_delay_s": 0}, {"enabled": "yes"}):
        with pytest.raises(ValueError, match="macros"):
            C.from_dict({"macros": bad})
    validate_macros_config(MacrosConfig(abort_hotkey=""))  # empty = no abort hotkey


# --------------------------------------------------------------------------- recording
def recorder(w: Any = None, **cfg: Any) -> tuple[MacroRecorder, SelfInputMarker, Clock]:
    clock = Clock()
    marker = SelfInputMarker(clock=clock)
    rec = MacroRecorder(MacrosConfig(**cfg), w or FakeWin(), marker, clock=clock, own_pid=999, threaded=False)
    rec.start()
    return rec, marker, clock


def test_recorder_stores_window_relative_clicks_with_the_control_name():
    w = FakeWin([win("chrome.exe", (100, 200, 900, 800))])
    w.at[(150, 260)] = ElementInfo("Compose", 50000, (140, 250, 200, 270))
    rec, _, clock = recorder(w)
    rec.feed("down", {"x": 150, "y": 260, "button": "left"})
    clock.t += 0.05
    rec.feed("up", {"x": 150, "y": 260, "button": "left"})
    m = rec.stop()
    t = m.steps[0].target
    assert (t.rx, t.ry, t.app, t.element, t.element_type) == (50, 60, "chrome.exe", "Compose", 50000)
    assert m.app == "chrome.exe"


def test_recorder_threaded_worker_processes_events():
    w = FakeWin([win("chrome.exe", (0, 0, 900, 900))])
    rec = MacroRecorder(MacrosConfig(), w, SelfInputMarker(), own_pid=999, threaded=True)
    rec.start()
    rec.feed("down", {"x": 1, "y": 2})
    rec.feed("up", {"x": 1, "y": 2})
    m = rec.stop()
    assert [s.type for s in m.steps] == ["click"]


def test_recorder_leaves_out_away_own_input_and_own_windows():
    w = FakeWin([win("away.exe", (0, 0, 50, 50), pid=999), win("chrome.exe", (0, 0, 900, 900))])
    rec, marker, clock = recorder(w)
    with marker.busy():  # a gesture click Away sent
        rec.feed("down", {"x": 300, "y": 300})
        rec.feed("up", {"x": 300, "y": 300})
    clock.t += 1
    rec.feed("key_down", {"key": "a", "char": "a"}, True)  # Windows says: injected
    rec.feed("down", {"x": 10, "y": 10})  # the Settings window (own pid)
    rec.feed("up", {"x": 10, "y": 10})
    clock.t += 1
    rec.feed("down", {"x": 400, "y": 400})
    rec.feed("up", {"x": 400, "y": 400})
    m = rec.stop()
    assert len(m.steps) == 1 and m.steps[0].target.x == 400
    assert rec.dropped_own == 3


def test_recorder_can_include_away_input_when_asked():
    rec, marker, _ = recorder(FakeWin([win("chrome.exe", (0, 0, 900, 900))]), record_away_input=True)
    with marker.busy():
        rec.feed("down", {"x": 300, "y": 300})
        rec.feed("up", {"x": 300, "y": 300})
    assert len(rec.stop().steps) == 1


def test_recorder_never_captures_password_boxes():
    w = FakeWin([win("chrome.exe", (0, 0, 900, 900))], focused="chrome.exe")
    rec, _, clock = recorder(w)
    rec.feed("key_down", {"key": "u", "char": "u"})
    clock.t += 1
    w.password = True
    for ch in "secret":
        rec.feed("key_down", {"key": ch, "char": ch})
        rec.feed("key_up", {"key": ch, "char": ch})
        clock.t += 0.2
    rec.feed("key_down", {"key": "enter"})
    clock.t += 1
    w.password = False
    rec.feed("key_down", {"key": "tab"})
    m = rec.stop()
    assert [(s.type, s.text or s.keys) for s in m.steps] == [("text", "u"), ("key", "tab")]
    assert m.skipped_password_keys == 7
    assert m.steps[0].app == "chrome.exe"


def test_without_password_detection_typed_text_is_left_out():
    rec, _, _ = recorder(NullWindowApi(lambda: "notepad.exe"))
    rec.feed("key_down", {"key": "h", "char": "h"})
    rec.feed("key_down", {"key": "enter"})
    m = rec.stop()
    assert [s.keys for s in m.steps] == ["enter"] and m.skipped_text
    assert m.steps[0].app == "notepad.exe"
    rec2, _, _ = recorder(NullWindowApi(), record_text_without_password_check=True)
    rec2.feed("key_down", {"key": "h", "char": "h"})
    assert rec2.stop().steps[0].text == "h"


def test_clicks_on_the_recording_bar_and_the_tray_that_ends_it_are_dropped():
    tray = win("explorer.exe", (0, 1000, 1920, 1040), cls="Shell_TrayWnd")
    rec, _, clock = recorder(FakeWin([tray, win("chrome.exe", (0, 0, 900, 900))]))
    rec.exclude_rects = [(800, 0, 1000, 40)]
    rec.feed("down", {"x": 100, "y": 100})
    rec.feed("up", {"x": 100, "y": 100})
    clock.t += 1
    rec.feed("down", {"x": 900, "y": 10})  # the Stop button on the recording bar
    rec.feed("up", {"x": 900, "y": 10})
    rec.feed("down", {"x": 1800, "y": 1020, "button": "right"})  # right-click the tray icon
    rec.feed("up", {"x": 1800, "y": 1020, "button": "right"})
    m = rec.stop()
    assert len(m.steps) == 1 and m.steps[0].target.x == 100


def test_trim_tail_drops_late_pointer_events_only():
    rec, _, clock = recorder(FakeWin([win("chrome.exe", (0, 0, 900, 900))]))
    rec.feed("down", {"x": 1, "y": 1})
    rec.feed("up", {"x": 1, "y": 1})
    clock.t += 5
    rec.feed("key_down", {"key": "f5"})
    rec.feed("down", {"x": 2, "y": 2})
    rec.feed("up", {"x": 2, "y": 2})
    clock.t += 0.5
    m = rec.stop(trim_tail_s=1.0)
    assert [s.type for s in m.steps] == ["click", "key"]


def test_marking_backend_reports_every_call():
    clock = Clock()
    marker = SelfInputMarker(window_s=0.1, clock=clock)
    inner = RecordingBackend()
    b = MarkingBackend(inner, marker)
    b.click(1)
    assert inner.calls == [("click", (1,))] and marker.is_own()
    assert b.calls is inner.calls  # plain attributes pass through
    clock.t += 1
    assert not marker.is_own()


# --------------------------------------------------------------------------- replay
def player(w: Any = None, backend: Any = None, sleeper: Any = None, **cfg: Any) -> MacroPlayer:
    return MacroPlayer(
        backend or FakeBackend(), w or FakeWin(), MacrosConfig(**cfg), SelfInputMarker(), sleep=sleeper or Sleeper()
    )


def test_click_follows_the_window_after_it_moved():
    w = FakeWin([win("chrome.exe", (500, 300, 1300, 900))], focused="chrome.exe")
    p = player(w)
    res = p.run(click_macro())
    assert res.ok and res.steps_run == 1
    assert p.backend.calls == [("move", 510, 320), ("click", "left", 1)]


def test_click_aims_at_the_recorded_control_when_it_is_found():
    w = FakeWin([win("chrome.exe", (0, 0, 800, 600))], focused="chrome.exe")
    m = click_macro()
    m.steps[0].target.element = "Send"
    w.elements["Send"] = ElementInfo("Send", 50000, (300, 400, 340, 420))
    p = player(w)
    assert p.run(m).ok
    assert p.backend.calls[0] == ("move", 320, 410)


def test_without_window_info_replay_uses_screen_position_and_focus_check():
    m = click_macro()
    p = player(NullWindowApi(lambda: "chrome.exe"))
    assert p.run(m).ok and p.backend.calls[0] == ("move", 110, 220)
    res = player(NullWindowApi(lambda: "notepad.exe")).run(m)
    assert res.fix == "allow_other_app" and "Notepad is in front" in res.message


def test_refuses_when_the_app_is_not_open_or_covered_and_allow_lifts_it():
    p = player(FakeWin([]))
    res = p.run(click_macro())
    assert not res.ok and res.stopped == "refused" and res.fix == "try_again" and "Chrome isn't open" in res.message
    assert res.fix_label == "Try again" and p.backend.calls == []
    covered = FakeWin([win("notepad.exe", (0, 0, 400, 400)), win("chrome.exe", (0, 0, 800, 600))])
    res = player(covered).run(click_macro())
    assert res.fix == "allow_other_app" and "Notepad is covering Chrome" in res.message
    assert player(covered, allow_other_app=True).run(click_macro()).ok
    m = click_macro()
    m.allow_other_app = True
    assert player(covered).run(m).ok


def test_refuses_a_click_outside_a_shrunken_window():
    res = player(FakeWin([win("chrome.exe", (0, 0, 5, 5))])).run(click_macro())
    assert res.fix == "record_again" and "smaller" in res.message


def test_typing_is_refused_in_another_app_and_skipped_in_password_boxes():
    m = Macro("t", steps=[Step("text", text="hello", app="notepad.exe"), Step("key", keys="enter", app="notepad.exe")])
    w = FakeWin(focused="chrome.exe")
    res = player(w).run(m)
    assert res.stopped == "refused" and res.fix == "allow_other_app" and "recorded it in Notepad" in res.message
    w.focused = "notepad.exe"
    w.password = True
    p = player(w)
    res = p.run(m)
    assert res.ok and res.skipped_password == 1 and "password box" in res.message
    assert p.backend.calls == [("key_down", "enter"), ("key_up", "enter")]
    w.password = False
    p = player(w)
    assert p.run(m).ok and p.backend.calls[0] == ("type_text", "hello")


def test_modifiers_and_buttons_are_released_even_when_input_fails():
    w = FakeWin([win("chrome.exe", (0, 0, 800, 600))])
    p = player(w, FakeBackend(fail_on="click"))
    res = p.run(click_macro(modifiers=["ctrl"]))
    assert res.stopped == "error" and res.fix == "edit_steps" and "input blocked" in res.message
    assert p.backend.calls[-1] == ("key_up", "ctrl")
    drag = Macro("d", steps=[Step("drag", target=Target(10, 10), path=[(5, 0), (50, 0)], duration=0.2)])

    def fail_mid_drag(name: str, a: tuple[Any, ...]) -> None:
        if name == "move" and a == (15, 10):
            raise OSError("input blocked")

    p2 = player(backend=FakeBackend(on_call=fail_mid_drag))
    assert p2.run(drag).stopped == "error"
    assert p2.backend.calls[-1] == ("release", "left")
    p3 = player(backend=FakeBackend(fail_on="key_down"))
    p3.run(Macro("k", steps=[Step("key", keys="ctrl+s")]))
    assert p3.backend.calls == [("key_down", "ctrl"), ("key_up", "ctrl")]


def test_abort_stops_between_steps_and_during_a_drag():
    m = Macro("abc", steps=[Step("key", keys="a"), Step("key", keys="b", delay=1.0), Step("key", keys="c")])
    holder: dict[str, MacroPlayer] = {}

    def on_call(name: str, a: tuple[Any, ...]) -> None:
        if name == "key_up" and a == ("a",):
            holder["p"].abort.set()

    p = player(backend=FakeBackend(on_call=on_call))
    holder["p"] = p
    res = p.run(m)
    assert res.stopped == "aborted" and res.steps_run == 1 and "after 1 of 3" in res.message
    drag = Macro("d", steps=[Step("drag", target=Target(10, 10), path=[(5, 0), (50, 0)], duration=2.0)])
    p2 = player(sleeper=Sleeper(abort_on=2))  # pause 1 = before the step, pause 2 = inside the drag
    res = p2.run(drag)
    assert res.stopped == "aborted" and p2.backend.calls[-1] == ("release", "left")
    p3 = player(sleeper=Sleeper(abort_on=1))
    assert p3.run(Macro("w", steps=[Step("wait", seconds=5.0)])).stopped == "aborted"


def test_speed_scales_recorded_pauses_but_not_added_waits():
    m = Macro("s", steps=[Step("key", keys="a"), Step("key", keys="b", delay=1.0), Step("wait", seconds=2.0)])
    for speed, expected in (("1x", 3.05), ("2x", 2.55), ("instant", 2.1)):
        s = Sleeper()
        assert player(sleeper=s, min_gap_s=0.05).run(m, speed=speed).ok
        assert s.total == pytest.approx(expected)


def test_abort_watcher_hotkey_and_real_mouse_but_never_away_input():
    clock = Clock()
    marker = SelfInputMarker(clock=clock)
    ev_ = threading.Event()
    w = AbortWatcher(MacrosConfig(abort_hotkey="ctrl+q", abort_move_px=40), ev_, marker)
    w.feed("key_down", {"key": "ctrl_l"}, True)
    w.feed("key_down", {"key": "q"}, True)
    assert not ev_.is_set()  # injected: Away pressed them
    w.feed("key_down", {"key": "ctrl_l"})
    w.feed("key_up", {"key": "ctrl_l"})
    w.feed("key_down", {"key": "q"})
    assert not ev_.is_set()  # ctrl was let go
    w.feed("key_down", {"key": "ctrl_r"})
    assert ev_.is_set() and w.reason == "hotkey"
    ev2 = threading.Event()
    w2 = AbortWatcher(MacrosConfig(abort_move_px=40), ev2, marker)
    with marker.busy():
        w2.feed("move", {"x": 500, "y": 500})  # our own move resets the reference point
    clock.t += 1
    w2.feed("move", {"x": 510, "y": 505})
    assert not ev2.is_set()
    w2.feed("move", {"x": 560, "y": 520})
    assert ev2.is_set() and w2.reason == "mouse"
    ev3 = threading.Event()
    AbortWatcher(MacrosConfig(), ev3, marker).feed("key_down", {"key": "esc"})
    assert ev3.is_set()  # Esc is the default
    ev4 = threading.Event()
    w4 = AbortWatcher(MacrosConfig(abort_on_mouse_move=False), ev4, marker)
    w4.feed("move", {"x": 0, "y": 0})
    w4.feed("move", {"x": 900, "y": 900})
    assert not ev4.is_set()


# --------------------------------------------------------------------------- service + bindings
def service(tmp_path, **cfg: Any) -> tuple[MacroService, list[tuple[str, str]], list[FakeBackend]]:
    notes: list[tuple[str, str]] = []
    backends: list[FakeBackend] = []

    def backend(_mark: Any) -> FakeBackend:
        backends.append(FakeBackend())
        return backends[-1]

    FakeListener.instances.clear()
    svc = MacroService(
        MacrosConfig(**cfg),
        MacroStore(tmp_path / "macros"),
        window_api=FakeWin([win("chrome.exe", (0, 0, 800, 600))], focused="chrome.exe"),
        listener_factory=FakeListener,
        backend_factory=backend,
        notify=lambda t, m: notes.append((t, m)),
        marker=SelfInputMarker(),
        threaded=False,
    )
    return svc, notes, backends


def test_voice_record_and_stop_recording_save_shortcut_n(tmp_path):
    svc, notes, _ = service(tmp_path)
    svc.hook_control("record", {})
    assert svc.recording and FakeListener.instances[0].started
    assert notes[-1][0] == "Recording a shortcut"
    sink = FakeListener.instances[0].sink
    sink("down", {"x": 50, "y": 60, "button": "left"}, False)
    sink("up", {"x": 50, "y": 60, "button": "left"}, False)
    svc.hook_control("stop", {})  # "stop" alone never ends a recording
    assert svc.recording
    svc.hook_control("stop_recording", {})
    assert not svc.recording and FakeListener.instances[0].stopped
    assert svc.store.names() == ["shortcut_1"]
    assert notes[-1][0] == "Shortcut saved" and "Settings > Shortcuts" in notes[-1][1]
    svc.hook_control("toggle_recording", {})
    svc.hook_control("toggle_recording", {})
    assert notes[-1][0] == "Nothing recorded" and svc.store.names() == ["shortcut_1"]
    assert svc.stop_recording() is None


def test_recording_fails_friendly_when_input_cannot_be_watched(tmp_path):
    svc, notes, _ = service(tmp_path)

    def broken(_sink: Any) -> Any:
        raise RuntimeError("no display")

    svc._listener_factory = broken
    svc.hook_control("record", {})
    assert not svc.recording and notes[-1][0] == "Can't record" and "no display" in notes[-1][1]


def test_play_through_hook_writes_status_and_notifies_on_refusal(tmp_path):
    svc, notes, backends = service(tmp_path)
    svc.store.save(click_macro())
    svc.hook_macro("open_mail", {})
    assert backends[0].calls == [("move", 10, 20), ("click", "left", 1)]
    assert json.loads((svc.store.folder / STATUS_FILE).read_text())["ok"] is True
    assert FakeListener.instances[-1].stopped  # the abort listener is cleaned up
    svc.store.save(click_macro("excel.exe"))
    res = svc.play("open_mail", wait=True)
    assert res is not None and res.fix == "try_again"
    status = MacroService.read_status(svc.store.folder)
    assert status is not None and status["name"] == "open_mail" and status["fix"] == "try_again"
    assert notes[-1][0] == "Shortcut didn't run"
    assert svc.play("missing", wait=True).fix == "record_again"
    MacroService.clear_status(svc.store.folder)
    assert MacroService.read_status(svc.store.folder) is None


def test_play_is_refused_while_recording_or_disabled(tmp_path):
    svc, _, _ = service(tmp_path)
    svc.store.save(click_macro())
    svc.start_recording()
    assert svc.play("open_mail", wait=True).stopped == "busy"
    svc.stop_recording()
    svc.cfg.enabled = False
    assert "turned off" in svc.play("open_mail", wait=True).message
    assert "turned off" in svc.start_recording()


def test_threaded_replay_can_be_stopped_by_voice(tmp_path):
    svc, _, _ = service(tmp_path)
    svc.threaded = True
    gate = threading.Event()
    svc.store.save(Macro("slow", steps=[Step("key", keys="a"), Step("wait", seconds=30.0)]))
    replays: list[bool] = []

    def on_replay(started: bool) -> None:
        replays.append(started)
        if started:
            gate.set()

    svc.on_replay = on_replay
    svc.hook_macro("slow", {})
    assert gate.wait(5)
    assert svc.play("slow").stopped == "busy"  # one at a time
    assert "running" in svc.start_recording()
    svc.hook_control("stop", {})
    svc.wait(5)
    assert svc.last_result is not None and svc.last_result.stopped == "aborted"
    assert replays == [True, False] and not svc.playing
    assert not svc.stop_playback()


def test_macro_actions_parse_and_dispatch_through_bindings():
    assert parse_action("macro:Open_Mail") == Action("macro", "open_mail")
    assert parse_action("macro_control:record") == Action("macro_control", "record")
    with pytest.raises(ValueError, match="not a shortcut id"):
        parse_action("macro:9 lives")
    with pytest.raises(ValueError):
        parse_action("macro_control:explode")
    b = parse_bindings(
        {"profiles": {"default": {"voice": {"open mail": "macro:open_mail"}, "gestures": {"palm_tap": "macro:x"}}}}
    )
    calls: list[tuple[str, str]] = []
    ex = ActionExecutor(
        MouseController(RecordingBackend(), CursorMapper((800, 600), C.CursorConfig())),
        hooks={
            "macro": lambda a, _d: calls.append(("macro", a)),
            "macro_control": lambda a, _d: calls.append(("c", a)),
        },
    )
    d = Dispatcher(b, ex)
    assert d.trigger("voice", "Open mail!") == Action("macro", "open_mail")
    d.trigger("voice", "record shortcut")
    d.trigger("voice", "stop recording")
    d.trigger("voice", "stop")
    assert calls == [("macro", "open_mail"), ("c", "record"), ("c", "stop_recording"), ("c", "stop")]
    assert d.fired[0] == ("voice:open mail", "macro:open_mail")


def test_without_a_handler_the_macro_action_is_logged_not_crashing():
    ex = ActionExecutor(MouseController(RecordingBackend(), CursorMapper((800, 600), C.CursorConfig())))
    assert ex.run(Action("macro", "x")) is False and "no handler for macro:x" in ex.log[-1]


def test_default_voice_phrases_include_recording_commands():
    v = BindingsConfig().default.voice
    assert v["record macro"] == v["record shortcut"] == Action("macro_control", "record")
    assert v["stop recording"] == Action("macro_control", "stop_recording")
    assert v["stop"] == Action("macro_control", "stop")


# --------------------------------------------------------------------------- Settings model
def panel(tmp_path) -> tuple[ShortcutsModel, SettingsModel]:
    sm = SettingsModel(C.AppConfig(), tmp_path / "config.yaml")
    svc, _, _ = service(tmp_path)
    return ShortcutsModel(sm, svc), sm


def record_one(m: ShortcutsModel) -> None:
    assert m.start_recording() is None
    sink = FakeListener.instances[-1].sink
    sink("down", {"x": 5, "y": 5, "button": "left"}, False)
    sink("up", {"x": 5, "y": 5, "button": "left"}, False)
    draft, p = m.stop_recording()
    assert p is None and draft is not None


def test_record_name_and_say_a_phrase_saves_and_binds(tmp_path):
    m, _ = panel(tmp_path)
    record_one(m)
    assert m.suggested_title() == "Shortcut 1"
    assert m.save_new("Open my mail", "phrase", "Open mail") is None
    saved, _ = C.load(tmp_path / "config.yaml")
    assert saved.bindings.resolve("voice", "open mail") == Action("macro", "open_my_mail")
    rows = m.rows()
    assert [(r.title, r.runs_when, r.steps, r.sensitive) for r in rows] == [
        ("Open my mail", 'Say "open mail"', 1, False)
    ]


def test_taking_over_a_used_gesture_asks_first(tmp_path):
    m, _ = panel(tmp_path)
    ids = [g for g, _ in m.gesture_choices()]
    assert "swipe_left" in ids and "palm_tap" in ids and "pinch" not in ids and "point" not in ids
    assert dict(m.gesture_choices())["swipe_left"] == "Swipe left (now: press left)"
    assert dict(m.gesture_choices())["palm_tap"] == "Palm tap"
    record_one(m)
    p = m.save_new("Next tab", "gesture", "swipe_left")
    assert p is not None and p.fix == "replace_binding" and p.fix_label == "Replace it"
    assert "press left" in p.message
    assert m.store.names() == []  # nothing saved until you agree
    assert m.save_new("Next tab", "gesture", "swipe_left", replace=True) is None
    assert m.runs_when("next_tab") == "Gesture: Swipe left"
    # moving it to a phrase gives the gesture its old job back
    assert m.bind("next_tab", "phrase", "next tab") is None
    assert m.bindings.resolve("gestures", "swipe_left") == Action("key", "left")
    assert m.unbind("next_tab") is None and m.runs_when("next_tab") == "Not set yet"


def test_custom_gestures_from_gesture_studio_are_offered(tmp_path):
    m, sm = panel(tmp_path)
    sm.bindings.custom_gestures.append("thumbs_up")
    assert "thumbs_up" in dict(m.gesture_choices())


def test_friendly_problems_for_bad_names_and_empty_recordings(tmp_path):
    m, _ = panel(tmp_path)
    assert m.save_new("x").fix == "record"
    assert m.start_recording() is None
    draft, p = m.stop_recording()
    assert draft is None and p is not None and p.fix == "record" and "Nothing was recorded" in p.message
    record_one(m)
    assert "name" in m.save_new("   ").message
    assert m.save_new("!!!").message.startswith("Use at least one letter")
    assert m.save_new("Mail", "phrase", "...").message.startswith("Type the words")
    assert m.save_new("Mail", "gesture", "fly").message == "Pick a gesture from the list."
    assert m.save_new("Mail", "wave", "x").message.startswith("Choose a phrase")
    assert m.save_new("Mail") is None
    record_one(m)
    assert "already have" in m.save_new("mail").message


def test_recording_note_explains_what_was_left_out():
    m = ShortcutsModel(None, None)  # type: ignore[arg-type]
    assert m.recording_note(Macro("x")) == ""
    note = m.recording_note(Macro("x", skipped_password_keys=3, skipped_text=True))
    assert "password box was not recorded" in note and "Edit steps" in note


def test_rename_and_delete_keep_bindings_in_step(tmp_path):
    m, _ = panel(tmp_path)
    record_one(m)
    m.save_new("Mail", "phrase", "open mail")
    assert m.rename("mail", "Inbox") is None
    assert m.bindings.resolve("voice", "open mail") == Action("macro", "inbox")
    assert m.store.names() == ["inbox"]
    assert m.rename("inbox", "Inbox") is None  # same id, new title only
    assert m.rename("inbox", "???").message.startswith("Use at least one letter")
    assert m.delete("inbox") is None
    assert m.store.names() == [] and m.bindings.resolve("voice", "open mail").kind == "none"


def test_fix_buttons_allow_other_app_and_last_problem(tmp_path):
    m, _ = panel(tmp_path)
    m.store.save(click_macro("excel.exe"))
    p = m.run("open_mail")
    assert p is not None and p.fix == "try_again" and m.last_problem() == p
    desk = m.service.window_api
    desk.windows.insert(0, win("notepad.exe", (0, 0, 800, 600)))
    desk.windows.append(win("excel.exe", (0, 0, 800, 600)))
    p = m.apply_fix(p)
    assert p is not None and p.fix == "allow_other_app"
    assert m.apply_fix(p) is None and m.store.load("open_mail").allow_other_app
    assert m.last_problem() is None
    assert m.run("open_mail") is None
    handed_back = Problem("x", "edit_steps", "open_mail")
    assert m.apply_fix(handed_back) is handed_back  # a UI flow: the view opens the editor
    assert m.apply_fix(Problem("x", "allow_other_app", "gone")).message.startswith("There is no shortcut")


def test_panel_export_and_import_problems(tmp_path):
    m, _ = panel(tmp_path)
    m.store.save(Macro("say_hi", steps=[Step("text", text="hi")]))
    p = m.export(["say_hi"], str(tmp_path / "x.json"))
    assert p is not None and p.fix == "export_private" and p.fix_label == "Export anyway"
    assert m.export(["say_hi"], str(tmp_path / "x.json"), include_private=True) is None
    assert m.export(["nope"], str(tmp_path / "y.json")).message.startswith("There is no shortcut")
    got, p = m.import_file(str(tmp_path / "x.json"))
    assert p is None and [g.name for g in got] == ["say_hi_2"]
    got, p = m.import_file(str(tmp_path / "missing.json"))
    assert got == [] and p is not None and "could not be read" in p.message


def test_step_editor(tmp_path):
    st = MacroStore(tmp_path)
    st.save(Macro("m", steps=[Step("key", keys="a"), Step("key", keys="b")]))
    ed = StepEditor(st, "m")
    assert ed.move(0, 1) == 1 and [s.keys for s in ed.steps] == ["b", "a"]
    assert ed.move(1, 1) == 1
    assert ed.set_delay(0, "0,5") is None and ed.steps[0].delay == 0.5
    assert ed.set_delay(0, "soon").message.startswith("The pause is a number")
    assert ed.set_delay(0, "999").message.startswith("The pause must be")
    assert ed.set_delay(9, "1").message == "Pick a step first."
    assert ed.insert_text(1, "hello") is None
    assert ed.insert_text(1, "").message.startswith("Type the text")
    assert ed.insert_key(-1, "Ctrl+S") is None and ed.steps[-1].keys == "ctrl+s"
    assert "unknown key" in ed.insert_key(0, "ctrl+nope").message
    assert ed.insert_key(0, " ").message.startswith("Type a key")
    assert ed.insert_wait(0, "2") is None and ed.steps[0].type == "wait"
    assert "wait must be" in ed.insert_wait(0, "0").message
    assert ed.insert_wait(0, "later").message.startswith("A wait is a number")
    assert ed.set_speed("2x") is None and ed.set_speed("9x") is not None
    assert ed.delete(99).message == "Pick a step first."
    assert ed.save() is None
    back = st.load("m")
    assert [s.type for s in back.steps] == ["wait", "key", "text", "key", "key"] and back.speed == "2x"
    assert back.sensitive
    assert ed.lines()[0] == "1. Wait 2 s" and ed.lines()[1] == "2. Press b  (after 0.5 s)"
    one = StepEditor(st, "m")
    one.macro.steps = one.macro.steps[:1]
    assert "at least one step" in one.delete(0).message
    one.macro.steps = []
    assert "no steps" in one.save().message


def test_action_labels_are_plain_words():
    assert action_label(Action("key", "left")) == "press left"
    assert action_label(Action("mouse", "right_click")) == "right click"
    assert action_label(Action("macro", "open_mail")) == 'run "Open mail"'
    assert action_label(Action("none")) == "nothing"
    assert action_label(Action("type", "secret")) == "type text"


def test_bindings_tab_offers_saved_shortcuts(tmp_path, monkeypatch):
    monkeypatch.setenv("AWAY_DATA_DIR", str(tmp_path))
    sm = SettingsModel(C.AppConfig(), tmp_path / "config.yaml")
    assert sm.arg_choices("macro") is None  # none yet: free text
    MacroStore(tmp_path / "macros").save(click_macro())
    assert sm.arg_choices("macro") == ["open_mail"]
    assert sm.set_binding("voice:open mail", "macro", "open_mail") is None


# --------------------------------------------------------------------------- CLI + tray
def cli_args(tmp_path, action: str, *rest: str, **kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        config=str(tmp_path / "config.yaml"),
        speed=None,
        seconds=None,
        name=None,
        say=None,
        gesture=None,
        replace=False,
        include_private=False,
        overwrite=False,
    )
    base.update(kw)
    return argparse.Namespace(action=action, args=list(rest), **base)


def test_cli_record_bind_run_export_import(tmp_path, capsys):
    from away.control.macro_cli import cmd_shortcut

    svc, _, backends = service(tmp_path)

    def do_steps() -> None:
        sink = FakeListener.instances[-1].sink
        sink("down", {"x": 5, "y": 5, "button": "left"}, False)
        sink("up", {"x": 5, "y": 5, "button": "left"}, False)

    assert cmd_shortcut(cli_args(tmp_path, "list"), svc) == 0
    assert "no shortcuts yet" in capsys.readouterr().out
    assert cmd_shortcut(cli_args(tmp_path, "record", name="Open mail"), svc, do_steps) == 0
    assert "saved open_mail (1 steps)" in capsys.readouterr().out
    assert cmd_shortcut(cli_args(tmp_path, "bind", "open_mail", say="open mail"), svc) == 0
    assert cmd_shortcut(cli_args(tmp_path, "bind", "open_mail", gesture="swipe_left"), svc) == 1
    assert "add --replace" in capsys.readouterr().err
    assert cmd_shortcut(cli_args(tmp_path, "bind", "open_mail"), svc) == 1
    assert cmd_shortcut(cli_args(tmp_path, "bind", "nope", say="x"), svc) == 1
    assert cmd_shortcut(cli_args(tmp_path, "list"), svc) == 0
    assert 'Say "open mail"' in capsys.readouterr().out
    assert cmd_shortcut(cli_args(tmp_path, "show", "open_mail"), svc) == 0
    assert "1. Click" in capsys.readouterr().out
    assert cmd_shortcut(cli_args(tmp_path, "run", "open_mail"), svc) == 0
    assert backends[-1].calls[-1] == ("click", "left", 1)
    svc.store.save(Macro("say_hi", steps=[Step("text", text="hi")]))
    out = tmp_path / "x.json"
    assert cmd_shortcut(cli_args(tmp_path, "export", str(out)), svc) == 1
    assert "--include-private" in capsys.readouterr().err
    assert cmd_shortcut(cli_args(tmp_path, "export", str(out), "open_mail"), svc) == 0
    assert cmd_shortcut(cli_args(tmp_path, "import", str(out)), svc) == 0
    assert "open_mail_2" in capsys.readouterr().out
    assert cmd_shortcut(cli_args(tmp_path, "delete", "nope"), svc) == 1
    assert cmd_shortcut(cli_args(tmp_path, "rename", "open_mail", "Inbox"), svc) == 0
    assert cmd_shortcut(cli_args(tmp_path, "unbind", "inbox"), svc) == 0
    assert cmd_shortcut(cli_args(tmp_path, "delete", "inbox"), svc) == 0
    assert cmd_shortcut(cli_args(tmp_path, "run", "missing"), svc) == 1
    assert cmd_shortcut(cli_args(tmp_path, "show"), svc) == 1
    assert cmd_shortcut(cli_args(tmp_path, "record"), svc, lambda: None) == 1  # nothing recorded


def test_cli_is_registered_with_an_alias():
    from away.cli import build_parser

    p = build_parser()
    assert p.parse_args(["shortcut", "list"]).action == "list"
    assert p.parse_args(["macro", "run", "x", "--speed", "2x"]).speed == "2x"


def test_tray_controller_shortcut_items():
    class App:
        macros: Any = None

    t = TrayController(App())
    assert not t.shortcut_recording() and not t.shortcut_playing()
    t.toggle_shortcut_recording()
    t.stop_shortcut()  # no service: nothing happens

    class Svc:
        recording = True
        playing = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        def hook_control(self, arg: str, _d: Any) -> None:
            self.calls.append(arg)

        def stop_playback(self) -> None:
            self.calls.append("stop")

    app = App()
    app.macros = Svc()
    t = TrayController(app)
    assert t.shortcut_recording() and t.shortcut_playing()
    t.toggle_shortcut_recording()
    t.stop_shortcut()
    assert app.macros.calls == ["toggle_recording", "stop"]


def test_dry_run_backend_records_calls():
    b = DryRunBackend()
    b.click("left", 1)
    assert b.calls == [("click", ("left", 1))]
    with pytest.raises(AttributeError):
        b._private  # noqa: B018
