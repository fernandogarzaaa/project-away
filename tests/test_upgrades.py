"""Research-driven upgrades: steady click, sigmoid curve, overlay state, auto-pause, dwell/face clicks, dictation."""

import itertools

import numpy as np
import pytest

from away import config as C
from away.app.runner import Pipeline
from away.config import AppConfig, CursorConfig
from away.control.mouse import MouseController, RecordingBackend
from away.hands.cursor import CursorMapper, accelerate, direction_scale, sigmoid_gain
from away.hands.gestures import GestureEvent
from away.hands.stabilize import ClickStabilizer, DwellDetector, pinch_progress
from away.hands.synthetic import make_hand


# --------------------------------------------------------------------------- B1 steady click
def test_click_stabilizer_freeze_rewind_and_release():
    st = ClickStabilizer(rewind_s=0.1, margin=0.35)
    for i in range(10):
        st.record(i * 0.033, (100.0 + i * 10, 200.0))
    st.observe(0.33, 0.60, 0.32, False, False)
    assert not st.frozen
    st.observe(0.36, 0.40, 0.32, False, False)  # below 0.32 * 1.35 = 0.432: closing
    assert st.frozen and st.closing_t == 0.36
    assert st.click_position() == (170.0, 200.0)  # newest sample at or before 0.26 s (0.231)
    assert st.click_position() is None  # consumed once per pinch
    st.observe(0.40, 0.50, 0.32, False, False)
    assert not st.frozen and st.just_released
    st.observe(0.45, 0.40, 0.32, False, False)
    st.observe(0.50, 0.30, 0.32, True, True)  # drag: never frozen
    assert not st.frozen and st.closing_t is None
    st.reset()
    assert st.position_at(1.0) is None


def test_mouse_controller_holds_cursor_while_pinch_closes():
    b = RecordingBackend()
    cfg = CursorConfig(mode="absolute", min_cutoff=1e6, d_cutoff=1e6, dead_zone=0.0, active_region=(0, 0, 1, 1))
    m = MouseController(b, CursorMapper((1001, 1001), cfg), stabilizer=ClickStabilizer(0.1, 0.35))
    t = 0.0
    for i in range(6):
        t += 0.033
        m.handle(GestureEvent("move", t, {"x": 0.5 + 0.01 * i, "y": 0.5}))
    before = m.mapper.pos
    m.observe_pinch(t, 0.40, 0.32, False, False)  # closing starts
    m.handle(GestureEvent("move", t + 0.03, {"x": 0.62, "y": 0.52}))  # index tip dragged by the pinch
    assert m.mapper.pos == before and b.calls[-1][0] == "move"
    n = len(b.calls)
    m.steady(t + 0.1)
    assert b.calls[n][0] == "move" and b.calls[n][1][0] < round(before[0])  # rewound ~100 ms
    m.observe_pinch(t + 0.2, 0.60, 0.32, False, False)
    assert not m.stabilizer.frozen and m.mapper._prev is None  # re-anchored, no jump on the next move


def test_pipeline_steady_click_lands_before_the_pinch():
    cfg = AppConfig()
    cfg.hands.cursor = CursorConfig(
        mode="absolute", min_cutoff=1e6, d_cutoff=1e6, dead_zone=0.0, active_region=(0, 0, 1, 1)
    )
    b = RecordingBackend()
    p = Pipeline(cfg, b, screen=(1001, 1001))
    t = 0.0
    for i in range(10):  # point and move right
        t += 0.033
        p.on_hand(t, make_hand({"middle": 1, "ring": 1, "pinky": 1}, center=(0.4 + 0.005 * i, 0.5), scale=0.2))
    rest = b.calls[-1][1]
    # pinch closes; the synthetic thumb pulls the index tip down-left while closing
    for k, pinch in enumerate((0.5, 0.8, 1.0, 1.0, 0.0)):
        t += 0.033
        lm = make_hand(
            {"middle": 1, "ring": 1, "pinky": 1}, pinch=pinch, center=(0.445 - 0.01 * k, 0.5 + 0.01 * k), scale=0.2
        )
        p.on_hand(t, lm)
    clicks = [i for i, c in enumerate(b.calls) if c[0] == "click"]
    assert clicks, b.calls[-6:]
    moves_before = [c for c in b.calls[: clicks[0]] if c[0] == "move"]
    landed = moves_before[-1][1]
    assert abs(landed[0] - rest[0]) <= 12 and abs(landed[1] - rest[1]) <= 12  # where the user was pointing
    off = AppConfig()
    off.hands.cursor.click_freeze = False
    assert Pipeline(off).mouse.stabilizer is None


# --------------------------------------------------------------------------- B2 sigmoid + asymmetry
def test_sigmoid_gain_shape():
    lo = sigmoid_gain(0.0, 0.6, 3.0, 1.5, 2.5)
    hi = sigmoid_gain(1.0, 0.6, 3.0, 1.5, 2.5)
    mid = sigmoid_gain(0.015, 0.6, 3.0, 1.5, 2.5)  # mid * ACCEL_REF
    assert lo == pytest.approx(0.6, abs=0.06) and hi == pytest.approx(3.0) and mid == pytest.approx(1.8)
    xs = np.linspace(0, 0.05, 50)
    gains = [sigmoid_gain(x, 0.6, 3.0, 1.5, 2.5) for x in xs]
    assert all(b >= a for a, b in itertools.pairwise(gains))  # monotonic
    assert sigmoid_gain(1e9, 0.6, 3.0, 1.5, 2.5) == pytest.approx(3.0)  # no overflow
    assert accelerate(0.04, 3.0) == pytest.approx(0.04 * 16)  # the old curve is still there


def test_direction_scale_and_mapper_curves():
    assert direction_scale(1.0, -1.0, up=2.0, down=1.0, left=1.0, right=0.5) == (0.5, -2.0)
    assert direction_scale(-1.0, 1.0, up=2.0, down=3.0, left=4.0, right=0.5) == (-4.0, 3.0)
    base = dict(min_cutoff=1e6, d_cutoff=1e6, dead_zone=0.0, gain=1.0)

    def move(cfg, dx, dy):
        m = CursorMapper((1000, 1000), cfg)
        m.update(0.0, 0.5, 0.5)
        x, y = m.update(0.033, 0.5 + dx, 0.5 + dy)
        return x - 500, y - 500

    sig = CursorConfig(curve="sigmoid", **base)
    slow, fast = move(sig, 0.002, 0.0), move(sig, 0.05, 0.0)
    assert slow[0] < 0.002 * 1000 and fast[0] > 0.05 * 1000 * 2.5  # precise when slow, fast when fast
    asym = CursorConfig(curve="power", acceleration=1.0, speed_left=2.0, speed_up=0.5, **base)
    assert move(asym, -0.01, 0.0)[0] == -20 and move(asym, 0.0, -0.01)[1] == -5
    assert AppConfig().hands.cursor.curve == "sigmoid" and CursorConfig().curve == "power"
    with pytest.raises(ValueError, match="curve"):
        C.from_dict({"hands": {"cursor": {"curve": "cubic"}}})
    with pytest.raises(ValueError, match="sigmoid_low"):
        C.from_dict({"hands": {"cursor": {"sigmoid_low": 4.0, "sigmoid_high": 2.0}}})


# --------------------------------------------------------------------------- dwell + overlay state
def test_dwell_detector():
    d = DwellDetector(radius_px=30, time_s=0.5)
    assert not d.update(0.0, (100, 100))
    assert not d.update(0.3, (110, 105)) and d.progress == pytest.approx(0.6)
    assert d.update(0.55, (112, 100))
    assert not d.update(1.5, (112, 100))  # needs to move away to re-arm
    assert not d.update(1.6, (200, 100)) and d.progress == 0.0
    assert d.update(2.2, (205, 100))
    assert not d.update(2.3, None) and d.progress == 0.0
    assert pinch_progress(None, 0.32, 0.45) == 0.0
    assert pinch_progress(0.1, 0.32, 0.45) == 1.0 and pinch_progress(2.0, 0.32, 0.45) == 0.0
    assert 0.0 < pinch_progress(0.5, 0.32, 0.45) < 1.0


def test_pipeline_dwell_click_and_cursor_state():
    cfg = AppConfig()
    cfg.hands.dwell.enabled = True
    cfg.hands.dwell.time_s = 0.3
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    assert p.cursor_state()["state"] == "hand_lost"
    point = make_hand({"middle": 1, "ring": 1, "pinky": 1}, scale=0.2)
    t = 0.0
    states = []
    for _ in range(15):
        t += 0.033
        p.on_hand(t, point)
        states.append(p.cursor_state()["state"])
    assert "dwell" in states and p.bus.history("gesture.dwell")
    assert ("click", (1,)) in b.calls and ("gesture:dwell", "mouse:click") in p.dispatcher.fired
    p.set_paused(True)
    assert p.cursor_state()["state"] == "paused"
    p.set_paused(False)
    p.on_hand(t + 0.05, make_hand({"middle": 1, "ring": 1, "pinky": 1}, pinch=1.0, scale=0.2))
    s = p.cursor_state()
    assert s["state"] in ("pinch", "drag") and s["pinch"] > 0.9 and set(s) >= {"x", "y", "dwell", "frozen"}


# --------------------------------------------------------------------------- B3 cursor overlay
def test_cursor_overlay_ring_spec():
    from away.app.cursor_overlay import COLORS, ring_spec
    from away.app.overlay import status_text
    from away.events import Event

    assert ring_spec({"state": "idle", "pinch": 0.0})["visible"] is False
    half = ring_spec({"state": "idle", "pinch": 0.5})
    assert half["visible"] and half["extent"] == pytest.approx(180.0) and not half["filled"]
    assert ring_spec({"state": "pinch"})["filled"] and ring_spec({"state": "drag"})["color"] == COLORS["drag"]
    assert ring_spec({"state": "dwell", "dwell": 0.25})["extent"] == pytest.approx(90.0)
    assert ring_spec({"state": "paused"})["color"] == COLORS["paused"]
    assert ring_spec({"state": "hand_lost"})["width"] == 1
    assert status_text(Event("app.auto_paused", {"paused": True}, 0.0)).startswith("PAUSED")
    assert status_text(Event("gesture.push", {}, 0.0)) == "push"


# --------------------------------------------------------------------------- B4 auto-pause + toggle hotkey
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_manual_input_guard_ignores_own_events():
    from away.app.autopause import GuardedBackend, ManualInputGuard

    clk = Clock()
    g = ManualInputGuard(idle_ms=1000, clock=clk)
    b = RecordingBackend()
    gb = GuardedBackend(b, g)
    gb.move(100, 100)
    assert not g.on_mouse_move(100, 100)  # the listener seeing our own move
    clk.t = 0.5
    assert not g.on_mouse_move(101, 101)  # within tolerance of our last synthetic position
    assert not g.active
    assert g.on_mouse_move(300, 300) and g.active  # the user moved the real mouse
    clk.t = 1.4
    assert g.active
    clk.t = 1.6
    assert not g.active
    gb.key("enter")
    assert not g.on_key()  # our own key press
    gb.click(1)
    assert not g.on_click()
    clk.t = 3.0
    assert g.on_key() and g.active
    gb.type_text("hi")
    assert b.calls[-1] == ("type_text", ("hi",)) and b.calls[0] == ("move", (100, 100))
    g.enabled = False
    clk.t = 10.0
    assert not g.on_click() and not g.active


def test_pipeline_auto_pause_blocks_output_then_resumes_without_jump():
    from away.app.autopause import ManualInputGuard

    clk = Clock()
    g = ManualInputGuard(idle_ms=500, clock=clk)
    b = RecordingBackend()
    p = Pipeline(AppConfig(), b, guard=g)
    point = make_hand({"middle": 1, "ring": 1, "pinky": 1}, scale=0.2)
    p.on_hand(0.0, point)
    clk.t = 0.1
    g.on_mouse_move(5, 5)
    n = len(b.calls)
    for i in range(5):
        p.on_hand(
            0.1 + i * 0.03, make_hand({"middle": 1, "ring": 1, "pinky": 1}, center=(0.5 + 0.02 * i, 0.5), scale=0.2)
        )
    assert len(b.calls) == n and p.output_paused() and p.cursor_state()["state"] == "paused"
    assert p.bus.history("app.auto_paused")[-1].data == {"paused": True}
    clk.t = 1.0
    p.on_hand(1.0, make_hand({"middle": 1, "ring": 1, "pinky": 1}, center=(0.7, 0.5), scale=0.2))
    assert not p.output_paused() and p.bus.history("app.auto_paused")[-1].data == {"paused": False}
    assert p.mouse.mapper._prev is not None  # re-anchored on the first frame after resuming


def test_new_hotkeys_validate():
    cfg, _ = C.from_dict({"app": {"toggle_hotkey": "ctrl+alt+g"}, "voice": {"edit_hotkey": ""}})
    assert cfg.app.toggle_hotkey == "ctrl+alt+g" and cfg.voice.edit_hotkey == ""
    with pytest.raises(ValueError, match="toggle_hotkey"):
        C.from_dict({"app": {"toggle_hotkey": "ctrl+notakey"}})


# --------------------------------------------------------------------------- B5 alternate clicks
def test_wink_trigger():
    from away.face.commands import WinkTrigger

    w = WinkTrigger(threshold=0.5, open_max=0.3, hold=0.2, cooldown=0.5)
    assert w.update(0.0, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}) is None  # blink: both closed
    assert w.update(0.3, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}) is None
    assert w.update(0.4, {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.1}) is None  # hold starts
    assert w.update(0.65, {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.1}) == "wink_left"
    assert w.update(0.9, {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.1}) is None  # still closed: fires once
    assert w.update(1.3, {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0}) is None
    assert w.update(1.4, {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.9}) is None
    assert w.update(1.7, {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.9}) == "wink_right"


def test_pipeline_face_clicks_are_bindable():
    cfg = AppConfig()
    cfg.face.wink_trigger = True
    cfg.face.mouth_open_trigger = True
    cfg.bindings.set("face:mouth_open", "mouse:click")
    cfg.bindings.set("face:wink_right", "key:esc")
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    box = [(0.5, 0.5, 0.3)]
    p.on_faces(0.0, box, [{}])
    for i in range(1, 12):
        p.on_faces(i * 0.05, box, [{"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.0}])
    assert ("click", (1,)) in b.calls and p.bus.history("face.wink_left")
    for i in range(12, 30):
        p.on_faces(i * 0.05, box, [{"eyeBlinkRight": 0.9}])
    assert ("key", ("esc",)) in b.calls
    n = b.calls.count(("click", (1,)))
    for i in range(30, 60):
        p.on_faces(i * 0.05, box, [{"jawOpen": 0.9}])
    assert b.calls.count(("click", (1,))) == n + 1 and p.bus.history("face.mouth_open")
    new = AppConfig()
    new.face.wink_hold = 1.0
    p.apply_config(new)
    assert p.wink.hold == 1.0


def test_settings_model_new_switches(tmp_path):
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(AppConfig(), tmp_path / "c.yaml")
    assert m.set_voice("face.wink_trigger", True) is None and m.cfg.face.wink_trigger
    assert m.set_voice("hands.dwell.enabled", 1) is None and m.cfg.hands.dwell.enabled is True
    assert m.set_voice("hands.cursor.curve", "power") is None and m.cfg.hands.cursor.curve == "power"
    assert m.set_voice("hands.cursor.curve", "cubic") is not None
    assert m.set_voice("app.toggle_hotkey", "ctrl+notakey") is not None and m.cfg.app.toggle_hotkey == "f9"
    assert m.set_voice("cursor_overlay", True) is None and m.cfg.cursor_overlay
    assert m.set_tunable("hands.dwell.radius_px", 50) is None and m.cfg.hands.dwell.radius_px == 50
    assert m.set_tunable("hands.cursor.click_rewind_ms", 120.4) is None and m.cfg.hands.cursor.click_rewind_ms == 120
