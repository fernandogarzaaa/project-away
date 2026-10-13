"""v0.1.19 auto-centering: soft recenter gain, learned comfort zone, edge clutch, cursor:center."""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest

from away.actions import ActionExecutor
from away.bindings import BindingsConfig, parse_action
from away.config import AutoCenterConfig, CursorConfig
from away.control.monitors import Monitor
from away.control.mouse import MouseController, RecordingBackend
from away.hands.autocenter import (
    FRAME_CENTER,
    AutoCenter,
    ComfortZone,
    EdgeClutch,
    clamp_target,
    shift_region,
    soft_gain,
)
from away.hands.cursor import CursorMapper

TWO = [Monitor(0, 0, 1920, 1080, True, "A"), Monitor(1920, 0, 1920, 1080, False, "B")]


def ccfg(**kw: Any) -> CursorConfig:
    ac = kw.pop("autocenter", None)
    base: dict[str, Any] = dict(min_cutoff=1e6, d_cutoff=1e6, beta=0.0, dead_zone=0.0, acceleration=1.0, gain=1.0)
    base.update(kw)
    c = CursorConfig(curve="power", **base)
    if ac is not None:
        c.autocenter = ac
    return c


# ------------------------------------------------------------------------------------------ soft gain
def test_soft_gain_is_neutral_at_the_target_and_for_zero_strength():
    assert soft_gain((0.5, 0.5), (0.01, 0.0), FRAME_CENTER, 0.2) == 1.0
    assert soft_gain((0.8, 0.5), (0.01, 0.0), FRAME_CENTER, 0.0) == 1.0
    assert soft_gain((0.8, 0.5), (0.0, 0.0), FRAME_CENTER, 0.2) == 1.0


def test_soft_gain_boosts_outward_and_cuts_inward_symmetrically():
    out = soft_gain((0.75, 0.5), (0.01, 0.0), FRAME_CENTER, 0.2)
    back = soft_gain((0.75, 0.5), (-0.01, 0.0), FRAME_CENTER, 0.2)
    assert out == pytest.approx(1.1)  # halfway to the edge: r = 0.5
    assert back == pytest.approx(0.9)
    assert soft_gain((1.0, 0.5), (0.01, 0.0), FRAME_CENTER, 0.2) == pytest.approx(1.2)  # at the edge: full
    assert soft_gain((0.75, 0.5), (0.0, 0.01), FRAME_CENTER, 0.2) == pytest.approx(1.0)  # perpendicular: none


def test_soft_gain_scales_with_distance_and_respects_the_target():
    near = soft_gain((0.55, 0.5), (0.01, 0.0), FRAME_CENTER, 0.2)
    far = soft_gain((0.9, 0.5), (0.01, 0.0), FRAME_CENTER, 0.2)
    assert 1.0 < near < far
    # with a target at (0.7, 0.5) the same position/motion is *toward* the target from the left
    assert soft_gain((0.6, 0.5), (0.01, 0.0), (0.7, 0.5), 0.2) < 1.0


def test_soft_recenter_drifts_the_resting_spot_toward_the_centre_over_closed_trips():
    """Out-and-back strokes of the same cursor distance: the hand ends a little closer to the centre each trip."""
    x = 0.8
    d_cursor = 0.05  # in hand units at gain 1
    history = [x]
    for _ in range(40):
        g_out = soft_gain((x, 0.5), (1.0, 0.0), FRAME_CENTER, 0.12)
        x += d_cursor / g_out
        g_back = soft_gain((x, 0.5), (-1.0, 0.0), FRAME_CENTER, 0.12)
        x -= d_cursor / g_back
        history.append(x)
    assert all(b < a for a, b in itertools.pairwise(history))  # strictly toward the centre every trip
    assert 0.5 < history[-1] < 0.7


# ------------------------------------------------------------------------------------------ comfort zone
def test_comfort_zone_learns_only_while_resting():
    cz = ComfortZone(tau_s=1.0, rest_speed=0.1)
    t = 0.0
    for _ in range(200):  # rest at (0.3, 0.6) for ~6.6 s
        t += 0.033
        cz.update(t, 0.3, 0.6)
    assert cz.center == pytest.approx((0.3, 0.6), abs=0.01)
    assert cz.learned and cz.rest_time == pytest.approx(199 * 0.033, rel=1e-3)
    before = cz.center
    for i in range(30):  # fast sweeps: 0.3 frame widths per frame, far above rest speed
        t += 0.033
        cz.update(t, 0.3 + (0.0 if i % 2 else 0.3), 0.6)
    assert cz.center == before


def test_comfort_zone_time_constant_and_gaps():
    cz = ComfortZone(tau_s=10.0, rest_speed=0.1, start=(0.5, 0.5))
    cz.update(0.0, 0.2, 0.5)
    cz.update(10.0, 0.2, 0.5)  # a 10 s gap (hand lost): not a resting sample
    assert cz.center == (0.5, 0.5)
    t = 10.0
    for _ in range(1000):  # 10 s of resting at 0.2
        t += 0.01
        cz.update(t, 0.2, 0.5)
    assert cz.center[0] == pytest.approx(0.5 - 0.3 * (1 - math.exp(-1)), abs=0.005)  # ~63 % of the way


def test_shift_region_keeps_size_and_stays_in_frame():
    reg = (0.15, 0.15, 0.85, 0.85)
    assert shift_region(reg, (0.4, 0.6)) == pytest.approx((0.05, 0.25, 0.75, 0.95))
    assert shift_region(reg, (0.0, 1.0)) == pytest.approx((0.0, 0.3, 0.7, 1.0))  # clamped
    assert shift_region((0.0, 0.0, 1.0, 1.0), (0.2, 0.2)) == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_clamp_target_keeps_targets_trackable():
    assert clamp_target((0.0, 1.0), 0.08) == pytest.approx((0.18, 0.82))
    assert clamp_target((0.4, 0.5), 0.08) == (0.4, 0.5)


# ------------------------------------------------------------------------------------------ edge clutch
def feed(ec: EdgeClutch, pts: list[tuple[float, float]], dt: float = 0.033) -> list[Any]:
    return [ec.update(i * dt, x, y) for i, (x, y) in enumerate(pts)]


def test_quick_swing_back_from_the_edge_clutches_then_reanchors_when_slow():
    ec = EdgeClutch(margin=0.1, return_speed=0.5, rest_speed=0.08)
    # out to the right edge, then a fast swing back (0.05 / frame = 1.5 widths/s) that stays outside the
    # inner frame (x > 0.8), then rest
    pts = [(0.9, 0.5), (0.95, 0.5), (0.99, 0.5), (0.94, 0.5), (0.89, 0.5), (0.85, 0.5), (0.85, 0.5)]
    st = feed(ec, pts)
    assert [s.clutched for s in st] == [False, False, False, True, True, True, False]
    assert [s.reanchor for s in st] == [False] * 6 + [True]


def test_clutch_ends_when_back_in_the_inner_frame():
    ec = EdgeClutch(margin=0.1, return_speed=0.5, rest_speed=0.01)
    st = feed(ec, [(0.95, 0.5), (0.9, 0.5), (0.82, 0.5), (0.75, 0.5)])
    assert [s.clutched for s in st] == [False, True, True, False] and st[-1].reanchor


def test_slow_moves_and_outward_moves_at_the_edge_never_clutch():
    ec = EdgeClutch(margin=0.1, return_speed=0.5)
    slow = feed(ec, [(0.95, 0.5), (0.945, 0.5), (0.94, 0.5), (0.935, 0.5)])  # 0.15 widths/s inward
    assert not any(s.clutched for s in slow)
    ec.reset()
    outward = feed(ec, [(0.91, 0.5), (0.95, 0.5), (0.99, 0.5)])
    assert not any(s.clutched for s in outward)
    ec.reset()
    middle = feed(ec, [(0.6, 0.5), (0.5, 0.5), (0.4, 0.5)])  # fast, but nowhere near an edge
    assert not any(s.clutched for s in middle)


def test_clutch_works_on_every_edge():
    for start, swing in (((0.03, 0.5), (0.1, 0.5)), ((0.5, 0.03), (0.5, 0.1)), ((0.5, 0.97), (0.5, 0.9))):
        ec = EdgeClutch(margin=0.1, return_speed=0.5)
        assert feed(ec, [start, swing])[-1].clutched, start


# ------------------------------------------------------------------------------------------ mapper integration
def test_mapper_swing_back_from_edge_does_not_drag_the_cursor_back():
    m = CursorMapper((1000, 1000), ccfg(autocenter=AutoCenterConfig(soft_recenter=False, edge_margin=0.1)))
    t = 0.0
    m.update(t, 0.6, 0.5)
    for x in (0.7, 0.8, 0.9, 0.95):  # move right: the cursor follows
        t += 0.1
        m.update(t, x, 0.5)
    right = m.pos[0]
    assert right > 500 + 300
    for x in (0.85, 0.75, 0.7):  # fast swing back (1 width/s): clutched, back inside at 0.75 -> re-anchor
        t += 0.1
        m.update(t, x, 0.5)
        assert m.pos[0] == right
    for x in (0.72, 0.75):  # the hand continues from where it is now: no jump back, normal moves
        t += 0.1
        m.update(t, x, 0.5)
    assert m.pos[0] == pytest.approx(right + 0.05 * 1000, abs=1)


def test_mapper_without_autocenter_drags_back_as_before():
    off = AutoCenterConfig(enabled=False)
    m = CursorMapper((1000, 1000), ccfg(autocenter=off))
    t = 0.0
    m.update(t, 0.6, 0.5)
    for x in (0.7, 0.8, 0.9, 0.95, 0.85, 0.75, 0.7):
        t += 0.1
        m.update(t, x, 0.5)
    assert m.pos[0] == pytest.approx(500 + 0.1 * 1000, abs=1)


def test_mapper_soft_gain_changes_relative_speed_off_centre_only():
    ac = AutoCenterConfig(edge_reanchor=False, soft_strength=0.2)
    m = CursorMapper((1000, 1000), ccfg(autocenter=ac))
    m.update(0.0, 0.75, 0.5)
    m.update(0.1, 0.76, 0.5)  # outward at r = 0.5: gain 1.1
    assert m.pos[0] == pytest.approx(500 + 10 * 1.1, abs=0.6)
    m2 = CursorMapper((1000, 1000), ccfg(autocenter=AutoCenterConfig(enabled=False)))
    m2.update(0.0, 0.75, 0.5)
    m2.update(0.1, 0.76, 0.5)
    assert m2.pos[0] == pytest.approx(510, abs=0.6)


def test_absolute_region_follows_the_comfort_zone_only_at_reanchor():
    ac = AutoCenterConfig(comfort_tau_s=0.5)
    m = CursorMapper((1000, 1000), ccfg(mode="absolute", autocenter=ac))
    t = 0.0
    for _ in range(150):  # rest at (0.3, 0.3) for 5 s: comfort zone learned there
        t += 0.033
        m.update(t, 0.3, 0.3)
    resting = m.pos
    assert m.autocenter is not None and m.autocenter.comfort.learned
    assert m.region == pytest.approx((0.15, 0.15, 0.85, 0.85))  # not applied under the resting hand
    assert m.pos == resting
    m.reset()  # hand lost / re-anchor: now the region moves to the comfort zone
    assert m.region == pytest.approx((0.0, 0.0, 0.7, 0.7), abs=0.02)
    m.update(t + 1, 0.35, 0.35)
    assert m.pos[0] == pytest.approx(0.5 * 999, abs=25)  # the comfort spot maps to the middle of the screen


def test_live_config_change_reaches_the_autocenter():
    c = ccfg()
    m = CursorMapper((1000, 1000), c)
    c.autocenter = AutoCenterConfig(enabled=False)
    m.update(0.0, 0.5, 0.5)
    assert m.autocenter is not None and m.autocenter.cfg is c.autocenter


def test_autocenter_comfort_target_used_once_learned():
    ac = AutoCenter(AutoCenterConfig(target="comfort", comfort_tau_s=0.2))
    assert ac.target() == FRAME_CENTER
    t = 0.0
    for _ in range(100):
        t += 0.033
        ac.observe(t, 0.02, 0.5)
    assert ac.target() == pytest.approx((0.18, 0.5), abs=0.01)  # learned, clamped inside the frame


# ------------------------------------------------------------------------------------------ cursor:center
def test_center_moves_to_the_current_monitor_centre_and_reanchors():
    b = RecordingBackend()
    mc = MouseController(b, CursorMapper(TWO, ccfg()))
    mc.mapper.set_position(3000, 100)
    assert mc.jump_monitor("center")
    assert b.calls[-1] == ("move", (2880, 540))
    assert mc.mapper._prev is None  # re-anchored: the next hand frame does not jump back
    mc.mapper.set_position(10, 10)
    assert mc.jump_monitor("center") and b.calls[-1] == ("move", (960, 540))


def test_center_and_gaze_actions_parse_and_run():
    assert str(parse_action("cursor:centre")) == "cursor:center"
    assert str(parse_action("cursor:Center Cursor")) == "cursor:center"
    assert str(parse_action("cursor:look here")) == "cursor:gaze"
    b = RecordingBackend()
    mc = MouseController(b, CursorMapper(TWO, ccfg()))
    ex = ActionExecutor(mc, False)
    assert not ex.run(parse_action("cursor:gaze"))  # no gaze source: refused, nothing moved
    assert b.calls == []
    mc.gaze_point = lambda: (2500.0, 300.0)
    assert ex.run(parse_action("cursor:gaze")) and b.calls[-1] == ("move", (2500, 300))
    mc.enabled = False
    assert not ex.run(parse_action("cursor:center"))


def test_default_voice_commands_for_center_and_gaze():
    b = BindingsConfig()
    for phrase in ("center", "Centre!", "center cursor"):
        assert str(b.resolve("voice", phrase)) == "cursor:center", phrase
    assert str(b.resolve("voice", "look here")) == "cursor:gaze"
