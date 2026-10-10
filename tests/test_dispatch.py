import pytest

from away.actions import ActionExecutor, CustomGestureDetector, Dispatcher, event_trigger, hotkey_map
from away.app.runner import Pipeline
from away.bindings import NONE, Action, BindingsConfig
from away.config import AppConfig, CursorConfig, VoiceConfig
from away.control.mouse import MouseController, RecordingBackend
from away.hands.cursor import CursorMapper
from away.hands.gestures import GestureEvent, GestureStateMachine
from away.hands.synthetic import gesture_sequence, pose


def make(bindings=None, **kw):
    b = RecordingBackend()
    cfg = CursorConfig(min_cutoff=1e6, d_cutoff=1e6, beta=0.0, dead_zone=0.0, acceleration=1.0, gain=1.0)
    mouse = MouseController(b, CursorMapper((1000, 1000), cfg))
    launched, shelled, opened = [], [], []
    ex = ActionExecutor(
        mouse,
        launcher=launched.append,
        shell_runner=shelled.append,
        url_opener=opened.append,
        **kw,
    )
    sm = GestureStateMachine()
    d = Dispatcher(bindings or BindingsConfig(), ex, sm)
    return d, b, (launched, shelled, opened), sm


def ev(kind, t=0.0, **data):
    return GestureEvent(kind, t, data)


def test_event_trigger_mapping():
    assert event_trigger(ev("click")) == "pinch"
    assert event_trigger(ev("double_click")) == "double_pinch"
    assert event_trigger(ev("drag_start")) == "pinch_hold"
    assert event_trigger(ev("move", anchor="index")) == "point"
    assert event_trigger(ev("move", anchor="pinch")) == "pinch_hold"
    assert event_trigger(ev("scroll")) == "two_finger_scroll"
    assert event_trigger(ev("clutch")) == "open_palm"
    assert event_trigger(ev("custom", name="wave")) == "wave"
    assert event_trigger(ev("hand_lost")) is None


def test_default_bindings_reproduce_old_mouse_calls():
    d, b, _, _ = make()
    for e in [
        ev("move", 0.0, x=0.5, y=0.5),
        ev("move", 0.1, x=0.6, y=0.5, anchor="index"),
        ev("click", 0.2),
        ev("double_click", 0.3),
        ev("drag_start", 0.4),
        ev("move", 0.45, x=0.6, y=0.6, anchor="pinch"),
        ev("drag_end", 0.5),
        ev("scroll", 0.6, dy=2.5),
        ev("swipe_left", 0.7),
        ev("swipe_right", 0.8),
    ]:
        d.on_gesture(e)
    names = [c for c in b.calls if c[0] != "move"]
    assert names == [
        ("click", (1,)),
        ("click", (1,)),  # double pinch: ONE extra click (first tap already clicked)
        ("press", ()),
        ("release", ()),
        ("scroll", (2.5,)),
        ("key", ("left",)),
        ("key", ("right",)),
    ]
    assert ("gesture:pinch", "mouse:click") in d.fired


def test_remapped_gestures():
    bnd = BindingsConfig()
    bnd.set("gesture:pinch", "mouse:right_click")
    bnd.set("gesture:double_pinch", "mouse:double_click")
    bnd.set("gesture:swipe_left", "key:alt+left")
    bnd.set("gesture:pinch_hold", "key:ctrl+c")
    bnd.set("gesture:two_finger_scroll", "key:ctrl+tab")
    bnd.set("gesture:point", "none")
    d, b, _, _ = make(bnd)
    d.on_gesture(ev("click", 0.0))
    d.on_gesture(ev("double_click", 0.1))
    d.on_gesture(ev("swipe_left", 0.2))
    d.on_gesture(ev("drag_start", 0.3))
    d.on_gesture(ev("move", 0.35, x=0.1, y=0.1, anchor="pinch"))  # not a drag: no cursor motion
    d.on_gesture(ev("drag_end", 0.4))
    for i in range(10):  # continuous scroll frames -> discrete key at most once per cooldown
        d.on_gesture(ev("scroll", 1.0 + i * 0.05, dy=1.0))
    d.on_gesture(ev("scroll", 1.6, dy=1.0))
    d.on_gesture(ev("move", 2.0, x=0.2, y=0.2, anchor="index"))
    assert b.calls == [
        ("click_button", ("right", 1)),
        ("click", (2,)),  # pinch is not a left click any more -> a real double click
        ("hotkey", ("alt", "left")),
        ("hotkey", ("ctrl", "c")),
        ("hotkey", ("ctrl", "tab")),
        ("hotkey", ("ctrl", "tab")),
    ]


def test_type_launch_url_shell_and_refusal():
    bnd = BindingsConfig(allow_shell=True)
    bnd.set("gesture:swipe_left", "type:hello world")
    bnd.set("gesture:swipe_right", "launch:firefox --new-window")
    bnd.set("gesture:pinch", "launch:https://example.com")
    bnd.set("gesture:double_pinch", "shell:echo hi > /dev/null")
    d, b, (launched, shelled, opened), _ = make(bnd, allow_shell=True)
    d.on_gesture(ev("swipe_left"))
    d.on_gesture(ev("swipe_right"))
    d.on_gesture(ev("click"))
    d.on_gesture(ev("double_click"))
    assert b.calls == [("type_text", ("hello world",))]
    assert launched == [["firefox", "--new-window"]]
    assert opened == ["https://example.com"]
    assert shelled == ["echo hi > /dev/null"]
    d.executor.allow_shell = False  # executor double-checks even if a binding slipped through
    assert d.executor.run(Action("shell", "echo no")) is False
    assert shelled == ["echo hi > /dev/null"] and "refused" in d.executor.log[-1]


def test_executor_errors_never_raise_and_missing_hooks():
    d, b, _, _ = make()

    def boom(arg, data):
        raise RuntimeError("x")

    d.executor.hooks["mic"] = boom
    assert d.executor.run(Action("mic", "toggle")) is False
    assert d.executor.errors and d.executor.errors[0][0] == "mic:toggle"
    assert d.executor.run(Action("dictation", "toggle")) is False  # no hook registered
    assert d.executor.run(NONE) is False
    assert d.executor.run(Action("mouse", "scroll_up")) and b.calls[-1] == ("scroll", (3.0,))


def test_open_palm_rebinding_controls_detector_pause():
    bnd = BindingsConfig()
    d, b, _, sm = make(bnd)
    assert sm.clutch_pauses is True
    bnd.set("gesture:open_palm", "key:cmd+d")
    d.set_bindings(bnd)
    assert sm.clutch_pauses is False
    t = 0.0
    events = []
    for _ in range(20):
        t += 0.05
        events += sm.update(t, pose("palm"))
    clutch = [e for e in events if e.kind == "clutch"]
    assert clutch and clutch[0].data["paused"] is False and not sm.paused
    d.on_gesture(clutch[0])
    assert b.calls[-1] == ("hotkey", ("cmd", "d"))


def test_profile_switch_follows_active_app():
    bnd = BindingsConfig()
    bnd.new_profile("chrome", ["chrome.exe"])
    bnd.set("gesture:swipe_left", "key:ctrl+shift+tab", "chrome")
    bnd.set("gesture:open_palm", "none", "chrome")
    now = [0.0]
    focused = ["notepad.exe"]
    b = RecordingBackend()
    mouse = MouseController(b, CursorMapper((100, 100)))
    sm = GestureStateMachine()
    d = Dispatcher(bnd, ActionExecutor(mouse), sm, active_app=lambda: focused[0], clock=lambda: now[0])
    d.on_gesture(ev("swipe_left"))
    assert b.calls[-1] == ("key", ("left",)) and d.profile == "default"
    focused[0] = "chrome.exe"
    d.on_gesture(ev("swipe_left"))  # cached for app_poll seconds
    assert b.calls[-1] == ("key", ("left",))
    now[0] = 1.0
    d.on_gesture(ev("swipe_left"))
    assert b.calls[-1] == ("hotkey", ("ctrl", "shift", "tab")) and d.profile == "chrome"
    assert sm.clutch_pauses is False  # chrome profile unbinds the open palm
    d.active_app = lambda: (_ for _ in ()).throw(OSError("no X"))
    now[0] = 2.0
    assert d.profile == "default"  # provider errors fall back to default


def test_trigger_voice_face_hotkey_and_pause_blocks_mouse():
    paused = [False]
    d, b, _, _ = make()
    d.paused = lambda: paused[0]
    calls = []
    d.executor.hooks["dictation"] = lambda arg, data: calls.append(("dictation", arg))
    d.executor.hooks["mic"] = lambda arg, data: calls.append(("mic", arg))
    assert d.trigger("voice", "Scroll Down!") == Action("mouse", "scroll_down")
    assert b.calls[-1] == ("scroll", (-3.0,))
    d.trigger("face", "brow_raise")
    d.trigger("hotkeys", "mute")
    assert calls == [("dictation", "toggle"), ("mic", "toggle")]
    assert d.trigger("voice", "unbound phrase") is NONE
    paused[0] = True
    n = len(b.calls)
    assert d.trigger("voice", "click") is NONE and len(b.calls) == n
    d.on_gesture(ev("click"))
    assert len(b.calls) == n
    d.trigger("hotkeys", "ptt")  # non-mouse actions still work while gestures are paused
    assert calls[-1] == ("dictation", "toggle")


def test_custom_gesture_detector_edges_and_cooldown():
    det = CustomGestureDetector(["wave"], min_conf=0.7, cooldown=1.0)
    assert det.update(0.0, None) == []
    assert det.update(0.1, {"wave": 0.9}) == ["wave"]
    assert det.update(0.2, {"wave": 0.95}) == []  # still held: no repeat
    assert det.update(0.3, {"wave": 0.1}) == []
    assert det.update(0.4, {"wave": 0.9}) == []  # re-armed but in cooldown
    det.update(0.5, {"wave": 0.0})
    assert det.update(1.6, {"wave": 0.8}) == ["wave"]


class FakeTemporal:
    def __init__(self):
        self.p = {"wave": 0.0}

    def push(self, lm, hand):
        return dict(self.p)


def test_pipeline_custom_gesture_fires_binding():
    cfg = AppConfig()
    cfg.hands.gestures.use_model = True
    cfg.bindings.custom_gestures.append("wave")
    cfg.bindings.set("gesture:wave", "key:ctrl+w")
    b = RecordingBackend()
    tm = FakeTemporal()
    p = Pipeline(cfg, b, temporal=tm)
    p.on_hand(0.0, pose("fist"))
    tm.p = {"wave": 0.95}
    out = p.on_hand(0.05, pose("fist"))
    assert "custom:wave" in out and ("hotkey", ("ctrl", "w")) in b.calls
    assert p.bus.history("gesture.custom")[0].data == {"name": "wave"}
    assert p.bus.history("action.fired")[-1].data == {"trigger": "gesture:wave", "action": "key:ctrl+w"}


def test_pipeline_remapped_pinch_end_to_end(rng):
    cfg = AppConfig()
    cfg.bindings.set("gesture:pinch", "mouse:right_click")
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    t = 0.0
    for lm in [pose("pinch_open"), pose("pinch"), pose("pinch_open")]:
        t += 0.05
        p.on_hand(t, lm)
    assert ("click_button", ("right", 1)) in b.calls and ("click", (1,)) not in b.calls


def test_pipeline_voice_pause_blocks_gestures_and_open_palm_resumes():
    cfg = AppConfig()
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    p.fire("voice", "pause gestures")
    assert p.paused and p.gestures.paused and p.bus.history("app.paused")[-1].data == {"paused": True}
    t = 0.0
    for lm in [pose("pinch_open"), pose("pinch"), pose("pinch_open")]:
        t += 0.05
        p.on_hand(t, lm)
    assert not any(c[0] == "click" for c in b.calls)
    for _ in range(20):  # open palm held -> resume (default open_palm -> pause:toggle)
        t += 0.05
        p.on_hand(t, pose("palm"))
    assert not p.paused
    p.fire("voice", "pause gestures")
    p.fire("voice", "resume gestures")
    assert not p.paused


def test_pipeline_mouth_open_trigger():
    cfg = AppConfig()
    cfg.face.mouth_open_trigger = True
    cfg.bindings.set("face:mouth_open", "key:esc")
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    for i in range(30):
        p.on_faces(i * 0.1, [(0.5, 0.5, 0.3)], [{"jawOpen": 0.9}])
    assert ("key", ("esc",)) in b.calls and len(p.bus.history("face.mouth_open")) == 1


def test_pipeline_apply_config_live_reload():
    cfg = AppConfig()
    p = Pipeline(cfg, RecordingBackend())
    new = AppConfig()
    new.hands.cursor.gain = 5.0
    new.hands.gestures.pinch_on = 0.2
    new.bindings.set("gesture:open_palm", "none")
    p.apply_config(new)
    assert p.mouse.mapper.cfg.gain == 5.0 and p.gestures.cfg.pinch_on == 0.2
    assert p.gestures.clutch_pauses is False


def test_hotkey_map_modes():
    bnd = BindingsConfig()
    bnd.set("hotkey:ctrl+shift+k", "type:hi")
    v = VoiceConfig()
    m = hotkey_map(bnd, v)
    assert m == {"<ctrl>+<alt>+<space>": "ptt", "<ctrl>+<alt>+m": "mute", "<ctrl>+<shift>+k": "ctrl+shift+k"}
    v.mode = "push_to_talk"  # ptt handled by the hold-key listener
    assert "ptt" not in hotkey_map(bnd, v).values()
    bnd.set("hotkey:mute", "none")
    assert "mute" not in hotkey_map(bnd, v).values()


def test_demo_script_still_works_with_bindings(rng):
    cfg = AppConfig()
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    t = 0.0
    for lm in gesture_sequence("swipe_right", 24, rng):
        t += 1 / 30
        p.on_hand(t, lm)
    assert ("key", ("right",)) in b.calls


@pytest.mark.parametrize("arg,count", [("click", 1), ("double_click", 2), ("middle_click", None)])
def test_executor_mouse_variants(arg, count):
    d, b, _, _ = make()
    d.executor.run(Action("mouse", arg))
    if count:
        assert b.calls[-1] == ("click", (count,))
    else:
        assert b.calls[-1] == ("click_button", ("middle", 1))
