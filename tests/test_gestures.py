import pytest

from away.config import GestureConfig
from away.hands.gestures import GestureStateMachine
from away.hands.synthetic import pose

DT = 1 / 30


def run(sm, frames, t0=0.0):
    """frames: list of landmarks-or-None; returns [(t, kind, data)]."""
    out = []
    t = t0
    for lm in frames:
        t += DT
        out += [(e.t, e.kind, e.data) for e in sm.update(t, lm)]
    return out, t


def kinds(events):
    return [k for _, k, _ in events]


def test_point_moves_cursor_with_index_tip():
    sm = GestureStateMachine()
    lm = pose("point", center=(0.4, 0.6))
    ev, _ = run(sm, [lm])
    assert kinds(ev) == ["move"]
    assert ev[0][2]["x"] == pytest.approx(lm[8, 0])
    assert ev[0][2]["y"] == pytest.approx(lm[8, 1])
    assert ev[0][2]["anchor"] == "index"


def test_fist_does_nothing():
    sm = GestureStateMachine()
    ev, _ = run(sm, [pose("fist")] * 5)
    assert ev == []


def test_quick_pinch_is_click():
    sm = GestureStateMachine()
    frames = [pose("pinch_open")] * 3 + [pose("pinch")] * 3 + [pose("pinch_open")] * 2
    ev, _ = run(sm, frames)
    assert kinds(ev).count("click") == 1
    assert "drag_start" not in kinds(ev)


def test_double_pinch_within_window():
    sm = GestureStateMachine()
    f = [pose("pinch")] * 2 + [pose("pinch_open")] * 2
    ev, _ = run(sm, [pose("pinch_open")] + f + f)
    clicks = [k for k in kinds(ev) if "click" in k]
    assert clicks == ["click", "double_click"]


def test_third_pinch_starts_new_click_pair():
    sm = GestureStateMachine()
    f = [pose("pinch")] * 2 + [pose("pinch_open")] * 2
    ev, _ = run(sm, [pose("pinch_open")] + f * 3)
    assert [k for k in kinds(ev) if "click" in k] == ["click", "double_click", "click"]


def test_slow_second_pinch_is_two_clicks():
    sm = GestureStateMachine(GestureConfig(double_pinch_window=0.1))
    f = [pose("pinch")] * 2 + [pose("pinch_open")] * 4
    ev, _ = run(sm, [pose("pinch_open")] + f + f)
    assert [k for k in kinds(ev) if "click" in k] == ["click", "click"]


def test_double_window_boundary_inclusive():
    cfg = GestureConfig(double_pinch_window=0.5)
    sm = GestureStateMachine(cfg)
    sm.update(0.0, pose("pinch_open"))
    sm.update(0.1, pose("pinch"))
    assert [e.kind for e in sm.update(0.2, pose("pinch_open"))] == ["click"]
    sm.update(0.6, pose("pinch"))
    assert [e.kind for e in sm.update(0.7, pose("pinch_open"))] == ["double_click"]


def test_hold_pinch_becomes_drag_and_moves_then_releases():
    sm = GestureStateMachine(GestureConfig(drag_hold=0.2))
    frames = [pose("pinch_open")] + [pose("pinch", center=(0.5 + 0.0 * i, 0.5)) for i in range(10)]
    frames += [pose("pinch_open")]
    ev, _ = run(sm, frames)
    k = kinds(ev)
    assert k.count("drag_start") == 1 and k.count("drag_end") == 1
    assert "click" not in k
    assert k.index("drag_start") < k.index("drag_end")
    moves = [d for _, kk, d in ev if kk == "move"]
    assert moves and all(m["anchor"] == "pinch" for m in moves)


def test_drag_starts_on_movement_before_hold_time():
    sm = GestureStateMachine(GestureConfig(drag_hold=5.0, drag_move=0.02))
    frames = [pose("pinch_open"), pose("pinch", center=(0.5, 0.5)), pose("pinch", center=(0.53, 0.5))]
    ev, _ = run(sm, frames)
    assert "drag_start" in kinds(ev)


def test_small_motion_before_hold_is_not_drag():
    sm = GestureStateMachine(GestureConfig(drag_hold=5.0, drag_move=0.05))
    frames = [pose("pinch_open"), pose("pinch", center=(0.5, 0.5)), pose("pinch", center=(0.51, 0.5))]
    ev, _ = run(sm, frames)
    assert "drag_start" not in kinds(ev)


def test_pinch_hysteresis():
    cfg = GestureConfig(pinch_on=0.3, pinch_off=0.5)
    sm = GestureStateMachine(cfg)
    assert sm._update_pinch(0.29) is True
    assert sm._update_pinch(0.3) is False  # strict
    sm.pinched = True
    assert sm._update_pinch(0.45) is True
    assert sm._update_pinch(0.5) is False


def test_hand_lost_ends_drag_and_keeps_pause():
    sm = GestureStateMachine(GestureConfig(drag_hold=0.0))
    run(sm, [pose("pinch_open"), pose("pinch"), pose("pinch")])
    assert sm.dragging
    sm.paused = True
    ev = sm.update(1.0, None)
    assert [e.kind for e in ev] == ["drag_end", "hand_lost"]
    assert ev[0].data["reason"] == "hand_lost"
    assert sm.paused and not sm.dragging and not sm.hand_present
    assert sm.update(1.1, None) == []  # only reported once


def test_double_click_survives_brief_hand_loss():
    sm = GestureStateMachine()
    sm.update(0.0, pose("pinch_open"))
    sm.update(0.05, pose("pinch"))
    sm.update(0.1, pose("pinch_open"))
    sm.update(0.15, None)
    sm.update(0.2, pose("pinch_open"))
    sm.update(0.25, pose("pinch"))
    assert [e.kind for e in sm.update(0.3, pose("pinch_open"))] == ["double_click"]


def test_scroll_with_two_fingers():
    sm = GestureStateMachine(GestureConfig(scroll_gain=10.0))
    up = [pose("two", center=(0.5, 0.6 - 0.02 * i)) for i in range(4)]
    ev, _ = run(sm, up)
    sc = [d["dy"] for _, k, d in ev if k == "scroll"]
    assert len(sc) == 3
    assert all(v == pytest.approx(0.2) for v in sc)  # 0.02 * gain 10, hand up = positive
    down = [pose("two", center=(0.5, 0.56 + 0.02 * i)) for i in range(3)]
    ev2, _ = run(sm, down, t0=1.0)
    assert all(d["dy"] < 0 for _, k, d in ev2 if k == "scroll")


def test_scroll_ignores_tiny_motion_and_accumulates():
    sm = GestureStateMachine()
    frames = [pose("two", center=(0.5, 0.5 - 0.001 * i)) for i in range(4)]
    ev, _ = run(sm, frames)
    # 0.001 steps are below the 0.002 threshold individually, but they accumulate
    assert kinds(ev) == ["scroll"]


def test_scroll_resets_after_other_pose():
    sm = GestureStateMachine()
    run(sm, [pose("two", center=(0.5, 0.5))])
    run(sm, [pose("fist")], t0=0.1)
    ev, _ = run(sm, [pose("two", center=(0.5, 0.2))], t0=0.2)
    assert ev == []


def test_swipe_right_and_left():
    sm = GestureStateMachine()
    right = [pose("palm", center=(0.2 + 0.06 * i, 0.5)) for i in range(8)]
    ev, t = run(sm, right)
    assert kinds(ev) == ["swipe_right"]
    assert ev[0][2]["dx"] >= 0.25
    left = [pose("palm", center=(0.8 - 0.06 * i, 0.5)) for i in range(8)]
    ev2, _ = run(sm, left, t0=t + 1.0)
    assert kinds(ev2) == ["swipe_left"]


def test_swipe_cooldown():
    sm = GestureStateMachine()
    frames = [pose("palm", center=(0.1 + 0.08 * i, 0.5)) for i in range(11)]
    ev, _ = run(sm, frames)
    assert kinds(ev).count("swipe_right") == 1


def test_slow_or_vertical_palm_is_not_swipe():
    sm = GestureStateMachine()
    slow = [pose("palm", center=(0.2 + 0.005 * i, 0.5)) for i in range(40)]
    assert "swipe_right" not in kinds(run(sm, slow)[0])
    sm2 = GestureStateMachine()
    diag = [pose("palm", center=(0.2 + 0.05 * i, 0.2 + 0.05 * i)) for i in range(8)]
    assert not any(k.startswith("swipe") for k in kinds(run(sm2, diag)[0]))


def test_clutch_toggles_pause_and_blocks_other_events():
    sm = GestureStateMachine(GestureConfig(clutch_hold=0.3))
    ev, t = run(sm, [pose("palm")] * 15)
    assert kinds(ev) == ["clutch"]
    assert ev[0][2]["paused"] is True and sm.paused
    ev, t = run(sm, [pose("point")] * 3, t0=t)
    assert ev == []
    ev, t = run(sm, [pose("palm")] * 15, t0=t)
    assert kinds(ev) == ["clutch"] and ev[0][2]["paused"] is False
    ev, _ = run(sm, [pose("point")], t0=t)
    assert kinds(ev) == ["move"]


def test_clutch_requires_still_palm():
    sm = GestureStateMachine(GestureConfig(clutch_hold=0.3, swipe_min_dx=5.0))
    drifting = [pose("palm", center=(0.3 + 0.03 * i, 0.5)) for i in range(20)]
    assert "clutch" not in kinds(run(sm, drifting)[0])


def test_clutch_latched_until_palm_released():
    sm = GestureStateMachine(GestureConfig(clutch_hold=0.2))
    ev, _ = run(sm, [pose("palm")] * 40)
    assert kinds(ev).count("clutch") == 1


def test_clutch_hold_boundary():
    sm = GestureStateMachine(GestureConfig(clutch_hold=0.5))
    sm.update(0.0, pose("palm"))
    assert sm.update(0.49, pose("palm")) == []
    assert [e.kind for e in sm.update(0.5, pose("palm"))] == ["clutch"]


def test_reset_clears_state():
    sm = GestureStateMachine()
    sm.paused = sm.pinched = sm.dragging = True
    sm.reset()
    assert not (sm.paused or sm.pinched or sm.dragging)
