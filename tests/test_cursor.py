import math

import pytest

from away.config import AutoCenterConfig, CursorConfig
from away.control.monitors import Layout, Monitor
from away.hands.autocenter import AutoCenter, shift_region
from away.hands.cursor import (
    ACCEL_REF,
    CursorMapper,
    accel_factor,
    accelerate,
    accelerate_sigmoid,
    apply_dead_zone,
    as_monitors,
    direction_scale,
    reach_factor,
    sigmoid_gain,
)


def fast_cfg(**kw):
    # very high cutoff => filter ~ passthrough so mapping math is testable exactly
    base = {"min_cutoff": 1e6, "beta": 0.0, "d_cutoff": 1e6}
    base.update(kw)
    return CursorConfig(**base)


def test_accelerate():
    assert accelerate(0.0, 2.0) == 0.0
    assert accelerate(ACCEL_REF / 2, 2.0) == pytest.approx(ACCEL_REF / 2)  # never below 1x
    assert accelerate(ACCEL_REF * 4, 2.0) == pytest.approx(ACCEL_REF * 16)
    assert accelerate(-ACCEL_REF * 4, 2.0) == pytest.approx(-ACCEL_REF * 16)
    assert accelerate(ACCEL_REF * 4, 1.0) == pytest.approx(ACCEL_REF * 4)
    assert accelerate(0.05, 1.5, ref=0.05) == pytest.approx(0.05)


def test_dead_zone():
    assert apply_dead_zone(0.003, 0.0, 0.004) == (0.0, 0.0)
    assert apply_dead_zone(0.003, 0.003, 0.004) == (0.003, 0.003)  # hypot 0.00424 > dz
    assert apply_dead_zone(0.004, 0.0, 0.004) == (0.004, 0.0)  # boundary passes


def test_invalid_screen():
    with pytest.raises(ValueError):
        CursorMapper((0, 100))
    with pytest.raises(ValueError):
        CursorMapper((100, -1))


def test_absolute_mapping_corners_and_clamp():
    m = CursorMapper((1001, 501), fast_cfg(mode="absolute", active_region=(0.2, 0.2, 0.8, 0.8), dead_zone=0.0))
    assert m.update(0.0, 0.2, 0.2) == (0, 0)
    assert m.update(0.1, 0.8, 0.8) == (1000, 500)
    assert m.update(0.2, 0.5, 0.5) == (500, 250)
    assert m.update(0.3, 0.95, 0.05) == (1000, 0)


def test_absolute_dead_zone_holds_position():
    m = CursorMapper((1000, 1000), fast_cfg(mode="absolute", active_region=(0, 0, 1, 1), dead_zone=0.01))
    p0 = m.update(0.0, 0.5, 0.5)
    assert m.update(0.1, 0.505, 0.5) == p0
    assert m.update(0.2, 0.6, 0.5) != p0


def test_relative_first_frame_anchors_without_jump():
    m = CursorMapper((1000, 800), fast_cfg())
    assert m.update(0.0, 0.9, 0.9) == (500, 400)


def test_relative_motion_gain_and_axes():
    cfg = fast_cfg(gain=2.0, acceleration=1.0, dead_zone=0.0)
    m = CursorMapper((1000, 800), cfg)
    m.update(0.0, 0.5, 0.5)
    assert m.update(0.1, 0.51, 0.5) == (520, 400)  # 0.01 * 2 * 1000
    assert m.update(0.2, 0.51, 0.49) == (520, 380)  # y uses the same (width) scale


def test_relative_dead_zone_is_slack_not_a_hop():
    """v0.1.25: tremor inside the dead zone never moves the cursor, and a slow move goes through bit by bit (it used
    to pile up under the threshold and then jump by the whole 0.012 at once)."""
    m = CursorMapper((1000, 1000), fast_cfg(gain=1.0, acceleration=1.0, dead_zone=0.01, hold_still=False))
    m.update(0.0, 0.5, 0.5)
    assert m.update(0.1, 0.505, 0.5) == (500, 500)  # inside the slack
    assert m.update(0.2, 0.512, 0.5) == (502, 500)  # the slack is taken up: only the 0.002 beyond it moves
    assert m.update(0.3, 0.515, 0.5) == (505, 500)  # from now on every bit of the slow move goes through
    assert m.update(0.4, 0.508, 0.5) == (505, 500)  # a small wobble back stays inside the slack


def test_relative_acceleration_makes_fast_moves_bigger():
    slow = CursorMapper((1000, 1000), fast_cfg(gain=1.0, acceleration=2.0, dead_zone=0.0))
    slow.update(0.0, 0.5, 0.5)
    x1, _ = slow.update(0.1, 0.505, 0.5)
    fast = CursorMapper((1000, 1000), fast_cfg(gain=1.0, acceleration=2.0, dead_zone=0.0))
    fast.update(0.0, 0.5, 0.5)
    x2, _ = fast.update(0.1, 0.54, 0.5)
    assert x1 - 500 == 5
    assert x2 - 500 == 160  # 0.04 * (0.04/0.01) * 1000


def test_relative_clamps_to_screen():
    m = CursorMapper((100, 100), fast_cfg(gain=10.0, acceleration=1.0, dead_zone=0.0))
    m.update(0.0, 0.5, 0.5)
    assert m.update(0.1, 0.9, 0.0) == (99, 0)


def test_anchor_change_reanchors():
    m = CursorMapper((1000, 1000), fast_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0))
    m.update(0.0, 0.5, 0.5, "index")
    m.update(0.1, 0.6, 0.5, "index")
    pos = m.pos
    assert m.update(0.2, 0.1, 0.1, "pinch") == (round(pos[0]), round(pos[1]))


def test_reset_and_set_position():
    m = CursorMapper((1000, 1000), fast_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0))
    m.update(0.0, 0.5, 0.5)
    m.reset()
    assert m.update(0.1, 0.9, 0.9) == (500, 500)
    m.set_position(-5, 5000)
    assert m.pos == (0.0, 999.0)


def test_filter_smooths_jitter():
    m = CursorMapper(
        (1000, 1000), CursorConfig(mode="absolute", active_region=(0, 0, 1, 1), dead_zone=0.0, min_cutoff=0.5, beta=0.0)
    )
    m.update(0.0, 0.5, 0.5)
    x, _ = m.update(1 / 30, 0.6, 0.5)
    assert 500 < x < 600


# ---------------------------------------------------------------------------
# Mutation-killing tests (v0.1.29). Each test pins an exact observable value
# that the corresponding mutation in src/away/hands/cursor.py changes.
# ---------------------------------------------------------------------------


def mut_cfg(**kw):
    # passthrough filter, and hold-still OFF unless a test asks for it, so the
    # mapping math below is exact and no other subsystem interferes
    base = {"min_cutoff": 1e6, "beta": 0.0, "d_cutoff": 1e6, "hold_still": False}
    base.update(kw)
    return CursorConfig(**base)


def slow_abs_cfg():
    # absolute mapper behind a very slow one-euro filter: after a reset the next
    # frame maps exactly, without a reset it barely moves
    return CursorConfig(
        mode="absolute",
        active_region=(0, 0, 1, 1),
        dead_zone=0.0,
        min_cutoff=0.01,
        beta=0.0,
        d_cutoff=1.0,
        smoothing="one_euro",
        hold_still=False,
    )


MON_A = Monitor(0, 0, 1000, 800, True, "A")
MON_B = Monitor(1000, 0, 1000, 800, False, "B")
MON_C = Monitor(2000, 0, 500, 800, False, "C")


def test_accel_factor_edges():
    assert accel_factor(0.0, 2.0) == 1.0  # no motion, no boost, and not None
    assert accel_factor(0.0, 3.0) == 1.0
    assert accel_factor(1.0, 2.0) == pytest.approx(100.0)  # (1.0 / 0.01) ** 1: mag == 1.0 is not the zero case
    assert accel_factor(0.02, 2.0) == pytest.approx(2.0)
    assert accel_factor(0.001, 2.0) == 1.0  # below the reference: clamped at 1x


def test_accelerate_reach_scales_only_the_boost():
    # factor 4 at reach 2: delta * (1 + (4 - 1) * 2) = 7x, not delta * (1 + 3 / 2)
    assert accelerate(0.04, 2.0, reach=2.0) == pytest.approx(0.28)
    assert accelerate(-0.04, 2.0, reach=2.0) == pytest.approx(-0.28)
    assert accelerate(0.04, 2.0, reach=0.0) == pytest.approx(0.04)


def test_sigmoid_gain_midpoint_and_shape():
    # at mag == mid * ref, z == 0 and the gain is exactly halfway: low + (high - low) / 2
    assert sigmoid_gain(0.015, 0.6, 3.0, 1.5, 2.5) == pytest.approx(1.8)
    # z = -slope * (x - mid) is a product: at x - mid == 2 the gain uses exp(-5), not exp(-2.5 / 2)
    assert sigmoid_gain(0.035, 0.6, 3.0, 1.5, 2.5) == pytest.approx(0.6 + 2.4 / (1.0 + math.exp(-5.0)))
    # the lower plateau tends to low, the upper plateau reaches high exactly in float64
    assert sigmoid_gain(0.0001, 0.6, 3.0, 1.5, 2.5) == pytest.approx(0.6 + 2.4 / (1.0 + math.exp(3.725)))
    assert sigmoid_gain(0.5, 0.6, 3.0, 1.5, 2.5) == 3.0


def test_sigmoid_gain_exponent_clamps_at_sixty():
    # z = -2.5 * (0 - 30) = 75, clamped to 60 before exp()
    assert sigmoid_gain(0.0, 0.0, 1.0, 30.0, 2.5) == 1.0 / (1.0 + math.exp(60.0))
    # the -60 side is indistinguishable in float64: 1 + exp(-60) rounds to exactly 1.0,
    # so a -61 clamp gives the same result; this pins the plateau value only
    assert sigmoid_gain(0.275, 0.0, 1.0, 1.5, 2.5) == 1.0


def test_accelerate_sigmoid_values():
    assert accelerate_sigmoid(0.0, 0.6, 3.0, 1.5, 2.5) == 0.0  # exactly zero, not None and not 1.0
    assert accelerate_sigmoid(0.015, 0.6, 3.0, 1.5, 2.5) == pytest.approx(0.027)  # 0.015 * midpoint gain 1.8
    assert accelerate_sigmoid(-0.015, 0.6, 3.0, 1.5, 2.5) == pytest.approx(-0.027)  # sign preserved via abs()
    assert accelerate_sigmoid(1.0, 0.6, 3.0, 1.5, 2.5) == pytest.approx(3.0)  # a delta of 1.0 is not "zero"


def test_direction_scale_picks_the_facing_side():
    assert direction_scale(3.0, -4.0, 2.0, 3.0, 5.0, 7.0) == (21.0, -8.0)  # right x7, up x2
    assert direction_scale(-3.0, 4.0, 2.0, 3.0, 5.0, 7.0) == (-15.0, 12.0)  # left x5, down x3
    assert direction_scale(0.0, 0.0, 2.0, 3.0, 5.0, 7.0) == (0.0, 0.0)


def test_direction_scale_small_moves_use_the_positive_side():
    # 0.5 is not > 1, but it is > 0: the right/down multipliers apply
    assert direction_scale(0.5, 0.0, 2.0, 3.0, 5.0, 7.0) == (3.5, 0.0)
    assert direction_scale(0.0, 0.5, 2.0, 3.0, 5.0, 7.0) == (0.0, 1.5)


def test_as_monitors_accepts_layouts_and_lists():
    lay = Layout((MON_B, MON_A), "test")
    assert as_monitors(lay) == lay.monitors  # a Layout passes through, not None
    assert as_monitors([MON_B, MON_A]) == (MON_A, MON_B)  # sorted left to right
    assert as_monitors([MON_A]) == (MON_A,)


def test_as_monitors_wh_tuple_becomes_the_primary():
    mons = as_monitors((100, 200))
    assert len(mons) == 1
    assert mons[0].primary is True
    assert (mons[0].left, mons[0].top, mons[0].width, mons[0].height) == (0, 0, 100, 200)
    assert mons[0].name == "primary"


def test_as_monitors_without_a_usable_monitor_raises():
    with pytest.raises(ValueError):
        as_monitors([])
    with pytest.raises(ValueError):
        as_monitors([Monitor(0, 0, 0, 100, False, "zero")])


def test_as_monitors_three_numbers_are_not_a_screen_size():
    # (w, h) is the only numeric form; three numbers fall through to the monitor
    # path and fail on the missing geometry attributes instead of unpacking
    with pytest.raises((AttributeError, TypeError)):
        as_monitors((100, 200, 300))


def test_reach_factor_spans():
    ref = Monitor(0, 0, 1000, 1000, True, "ref")
    wide = [Monitor(-500, -500, 1000, 1000, True, "a"), Monitor(500, -500, 1000, 500, False, "b")]
    assert reach_factor(wide, ref, 0.5) == pytest.approx(math.sqrt(2.0))  # width spans 2000 of 1000
    tall = [Monitor(-500, -500, 1000, 1000, True, "a"), Monitor(-500, 500, 500, 1500, False, "b")]
    assert reach_factor(tall, ref, 0.5) == pytest.approx(math.sqrt(2.5))  # height spans 2500 of 1000
    assert reach_factor([ref], ref, 0.5) == pytest.approx(1.0)


def test_reach_factor_negative_exponent_clamps_to_zero():
    ref = Monitor(0, 0, 1000, 1000, True, "ref")
    mons = [Monitor(0, 0, 1000, 1000, True, "a"), Monitor(1000, 0, 1000, 1000, False, "b")]
    assert reach_factor(mons, ref, -2.0) == 1.0  # span ** max(0, -2) == span ** 0


def test_mapper_initial_state():
    m = CursorMapper((1000, 800), mut_cfg())
    assert m.clutched is False
    assert m.autocenter is not None  # the default config carries an AutoCenterConfig
    assert m.hold is None  # hold_still off in mut_cfg
    m2 = CursorMapper((1000, 800), mut_cfg(hold_still=True))
    assert m2.hold is not None  # the constructor syncs the hold from the config


def test_mapper_without_autocenter_config():
    cfg = mut_cfg()
    cfg.autocenter = None
    m = CursorMapper((1000, 800), cfg)
    assert m.autocenter is None
    assert m.update(0.0, 0.5, 0.5) == (500, 400)


def test_set_monitors_keeps_position_by_default():
    m = CursorMapper((1000, 800), mut_cfg())
    m.set_position(300, 200)
    m.set_monitors((2000, 1000))  # no keep_position argument: the position survives
    assert m.pos == (300.0, 200.0)


def test_set_monitors_clamps_kept_position_axis_by_axis():
    m = CursorMapper((3000, 1000), mut_cfg())
    m.set_position(1500, 100)
    m.set_monitors((1000, 800))
    assert m.pos == (999.0, 100.0)  # x clamped from pos[0], y kept from pos[1]


def test_set_monitors_desktop_size_with_negative_origin():
    mons = [Monitor(-1000, -500, 1000, 500, True, "a"), Monitor(0, 0, 1000, 500, False, "b")]
    m = CursorMapper(mons, mut_cfg())
    assert (m.x0, m.y0) == (-1000, -500)
    assert m.w == 2000  # 1000 - (-1000), a subtraction across the origin
    assert m.h == 1000  # 500 - (-500)


class BareCfg:
    """Config-shaped stub with NO hold_*, monitor_reach or autocenter attributes,
    so the getattr defaults inside CursorMapper are the behavior under test."""

    mode = "relative"
    min_cutoff = 1e6
    beta = 0.0
    d_cutoff = 1e6
    dead_zone = 0.0
    gain = 1.0
    acceleration = 1.0
    curve = "power"
    active_region = (0.0, 0.0, 1.0, 1.0)
    speed_up = 1.0
    speed_down = 1.0
    speed_left = 1.0
    speed_right = 1.0
    sigmoid_low = 0.6
    sigmoid_high = 3.0
    sigmoid_mid = 1.5
    sigmoid_slope = 2.5


def test_set_monitors_reach_uses_the_default_exponent_with_a_bare_cfg():
    mons = [
        Monitor(0, 0, 1000, 1000, True, "a"),
        Monitor(1000, 0, 1000, 1000, False, "b"),
        Monitor(2000, 0, 1000, 1000, False, "c"),
    ]
    m = CursorMapper(mons, BareCfg())
    assert m.reach == pytest.approx(math.sqrt(3.0))  # span 3 ** the 0.5 default, not ** 1.5


def test_refresh_selection_reapplies_the_monitor_spec():
    m = CursorMapper([MON_A, MON_B], mut_cfg())
    assert len(m.monitors) == 2
    m.cfg.monitors = "primary"
    m.refresh_selection()  # must actually re-run the selection
    assert m.monitors == (MON_A,)


def test_refresh_selection_keeps_the_position():
    m = CursorMapper([MON_A, MON_B], mut_cfg())
    m.set_position(300, 400)
    m.cfg.monitors = "primary"
    m.refresh_selection()
    assert m.pos == (300.0, 400.0)  # kept, not recentered on the primary


def test_current_monitor_under_the_cursor():
    stacked = [Monitor(0, 0, 1000, 500, True, "top"), Monitor(0, 500, 1000, 500, False, "bot")]
    m = CursorMapper(stacked, mut_cfg())
    m.set_position(200, 750)
    assert m.current_monitor() == 1  # looked up as (x=200, y=750), not (750, 200)
    m.set_position(200, 250)
    assert m.current_monitor() == 0


def test_jump_next_and_prev_wrap_around():
    m = CursorMapper([MON_A, MON_B, MON_C], mut_cfg())
    assert m.jump("next") == (1500, 400)
    assert m.jump("next") == (2250, 400)
    assert m.jump("next") == (500, 400)  # wraps to the first monitor
    assert m.jump("prev") == (2250, 400)  # and back the other way
    assert m.jump("prev") == (1500, 400)


def test_jump_primary_and_numbered_targets():
    m = CursorMapper([MON_A, MON_B, MON_C], mut_cfg())
    m.jump("next")
    m.jump("next")
    assert m.jump("primary") == (500, 400)
    assert m.jump("2") == (1500, 400)  # 1-based: the second monitor
    assert m.jump("3") == (2250, 400)  # the last valid number
    assert m.jump("1") == (500, 400)


def test_jump_rejects_unknown_targets():
    m = CursorMapper([MON_A, MON_B, MON_C], mut_cfg())
    assert m.jump("bogus") is None  # not a number: swallowed ValueError, no crash
    assert m.jump("0") is None  # numbers are 1-based
    assert m.jump("4") is None  # past the last monitor: None, not an index error


def test_jump_to_an_excluded_monitor_is_none():
    m = CursorMapper([MON_A, MON_B, MON_C], mut_cfg(monitors="primary"))
    assert m.jump("2") is None  # connected but not in the active selection
    assert m.jump("next") == (500, 400)  # next cycles inside the selection only


def test_jump_reanchors_the_hand():
    m = CursorMapper([MON_A, MON_B], mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0))
    m.update(0.0, 0.5, 0.5)
    m.update(0.1, 0.6, 0.5)
    assert m.jump("next") == (1500, 400)
    # the jump reset the mapper: a teleported hand anchors instead of dragging the cursor
    assert m.update(0.2, 0.1, 0.1) == (1500, 400)


def test_center_uses_the_current_monitor_and_reanchors():
    m = CursorMapper([MON_A, MON_B], mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0))
    m.update(0.0, 0.5, 0.5)  # gives the mapper a _prev the reset must clear
    m.set_position(1200, 200)  # on MON_B
    assert m.center() == (1500, 400)  # MON_B's center, x from pos[0] and y from pos[1]
    assert m.update(0.1, 0.9, 0.9) == (1500, 400)  # re-anchored by the reset inside center()


def test_warp_moves_reanchors_and_clamps():
    m = CursorMapper((1000, 800), mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0))
    m.update(0.0, 0.5, 0.5)
    assert m.warp(123, 456) == (123, 456)  # the warp actually moves the position
    assert m.update(0.1, 0.9, 0.1) == (123, 456)  # and re-anchors the hand
    assert m.warp(-50, 9999) == (0, 799)  # clamped onto the desktop


def test_sync_hold_defaults_with_a_bare_cfg():
    cfg = BareCfg()
    m = CursorMapper((1000, 800), cfg)
    assert m._hold_key == (False, 0.0, 0, "relative")  # the getattr defaults, verbatim
    assert m.hold is None
    cfg.hold_still = True
    m._sync_hold()  # the key changed (False to True): the hold appears
    assert m.hold is not None


def test_sync_hold_keeps_the_same_hold_when_defaults_become_explicit():
    cfg = BareCfg()
    cfg.hold_still = True
    m = CursorMapper((1000, 800), cfg)
    first = m.hold
    assert first is not None
    cfg.hold_radius = 0.0  # same value the default already produced: no rebuild
    m._sync_hold()
    assert m.hold is first
    cfg.hold_ms = 0  # same for the milliseconds default
    m._sync_hold()
    assert m.hold is first


def test_sync_hold_follows_live_config_changes():
    m = CursorMapper((1000, 800), mut_cfg(hold_still=True, hold_ms=250))
    assert m.hold.hold_s == pytest.approx(0.25)
    m.cfg.hold_ms = 900
    m._sync_hold()
    assert m.hold.hold_s == pytest.approx(0.9)  # rebuilt because the key changed
    first = m.hold
    m._sync_hold()
    assert m.hold is first  # and not rebuilt again while nothing changed


def test_hold_lock_applies_in_update_and_anchor_change_resets_it():
    cfg = mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0, hold_still=True, hold_ms=250, hold_radius=0.006)
    m = CursorMapper((1000, 1000), cfg)
    assert m.update(0.0, 0.5, 0.5) == (500, 500)
    m.update(0.1, 0.5, 0.5)
    m.update(0.2, 0.5, 0.5)
    m.update(0.3, 0.5, 0.5)
    assert m.hold.locked  # 300 ms perfectly still inside the radius: locked
    # still inside the lock radius: the hold swallows the move entirely
    assert m.update(0.4, 0.504, 0.5) == (500, 500)
    # an anchor change resets the hold: the same hand position re-anchors...
    assert m.update(0.5, 0.504, 0.5, "pinch") == (500, 500)
    # ...and the next move goes through instead of unlocking against the stale anchor
    assert m.update(0.6, 0.51, 0.5, "pinch") == (506, 500)


def test_suspend_and_resume_inside_the_window_keeps_the_filter():
    m = CursorMapper((1001, 1001), slow_abs_cfg())
    m.update(0.0, 0.2, 0.2)
    m.update(0.1, 0.8, 0.3)  # the suspended raw point is (0.8, 0.3): asymmetric on purpose
    m.suspend(0.2)
    out = m.update(0.3, 0.81, 0.31)  # 0.1 s later, 0.014 away: resume, no reset
    assert out[0] < 700 and out[1] < 700  # the slow filter continued; a reset would map exactly (810, 310)


def test_suspend_and_resume_after_the_window_reanchors():
    m = CursorMapper((1001, 1001), slow_abs_cfg())
    m.update(0.0, 0.2, 0.2)
    m.update(0.1, 0.8, 0.3)
    m.suspend(0.2)
    # 0.31 s > the 0.3 s resume window: reset, so the frame maps exactly
    assert m.update(0.51, 0.81, 0.31) == (810, 310)


def test_suspend_and_resume_too_far_reanchors():
    m = CursorMapper((1001, 1001), slow_abs_cfg())
    m.update(0.0, 0.2, 0.2)
    m.update(0.1, 0.8, 0.3)
    m.suspend(0.2)
    # 0.1 frame units > the 0.04 resume distance: reset, exact mapping
    assert m.update(0.25, 0.9, 0.3) == (900, 300)


def test_suspend_and_resume_at_the_window_edges():
    # exactly 0.3 s later still resumes (the comparison is <=)
    m = CursorMapper((1001, 1001), slow_abs_cfg())
    m.update(0.0, 0.8, 0.3)
    m.suspend(0.0)
    assert m.update(0.3, 0.81, 0.31) == (800, 300)  # resumed: barely moved off (0.8, 0.3)
    # exactly 0.04 away still resumes
    m2 = CursorMapper((1001, 1001), slow_abs_cfg())
    m2.update(0.0, 0.0, 0.3)
    m2.suspend(0.0)
    assert m2.update(0.1, 0.04, 0.3) == (0, 300)
    # zero elapsed time still resumes (0.0 <= dt)
    m3 = CursorMapper((1001, 1001), slow_abs_cfg())
    m3.update(0.4, 0.2, 0.2)
    m3.update(0.45, 0.8, 0.3)
    m3.suspend(0.5)
    out = m3.update(0.5, 0.81, 0.31)
    assert out[0] < 700 and out[1] < 700


def test_suspend_without_a_hand_is_a_clean_reset():
    m = CursorMapper((1001, 1001), slow_abs_cfg())
    m.suspend(0.0)  # no update yet: nothing to suspend, and nothing stored either
    assert m.update(0.1, 0.5, 0.5) == (500, 500)


def test_reset_clears_the_filter_state():
    m = CursorMapper((1000, 1000), mut_cfg())
    m.update(0.0, 0.5, 0.5)
    m.update(0.1, 0.6, 0.5)
    assert m.filter._t is not None
    m.reset()
    assert m.filter._t is None


def test_reset_clears_a_locked_hold():
    cfg = mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0, hold_still=True, hold_ms=250, hold_radius=0.006)
    m = CursorMapper((1000, 1000), cfg)
    for t in (0.0, 0.1, 0.2, 0.3):
        m.update(t, 0.5, 0.5)
    assert m.hold.locked
    m.reset()  # must call hold.reset(): the lock is gone afterwards
    assert m.hold.locked is False


def test_reset_without_a_hold_does_not_raise():
    m = CursorMapper((1000, 1000), mut_cfg(hold_still=False))
    assert m.hold is None
    m.reset()  # the hold branch is guarded by "is not None"


def test_reset_after_the_autocenter_is_removed():
    cfg = mut_cfg()
    m = CursorMapper((1000, 800), cfg)
    cfg.autocenter = None
    m.reset()  # syncs to None first, then must skip the reset on it
    assert m.autocenter is None


def test_sync_autocenter_lifecycle():
    cfg = mut_cfg()
    m = CursorMapper((1000, 800), cfg)
    ac = m.autocenter
    assert ac is not None
    replacement = AutoCenterConfig()
    cfg.autocenter = replacement
    assert m._sync_autocenter() is ac  # same object, retargeted, and returned (not None)
    assert ac.cfg is replacement
    cfg.autocenter = None
    m._sync_autocenter()
    assert m.autocenter is None
    cfg.autocenter = AutoCenterConfig()
    out = m._sync_autocenter()  # from None back to a config: a fresh AutoCenter appears
    assert isinstance(out, AutoCenter)
    assert m.autocenter is out


def test_edge_clutch_freezes_releases_and_reanchors():
    cfg = mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0)  # autocenter on, hold off
    m = CursorMapper((1000, 800), cfg)
    assert m.update(0.0, 0.02, 0.5) == (500, 400)  # hand parked in the left edge band
    # swinging back inward at 1.6 frame widths per second: the clutch engages and
    # this frame's motion is swallowed, exactly (500, 400), not a moved tuple or None
    assert m.update(0.05, 0.10, 0.5) == (500, 400)
    assert m.clutched is True
    assert m.update(0.10, 0.13, 0.5) == (500, 400)  # still fast, still outside the inner zone
    assert m.update(0.15, 0.145, 0.5) == (500, 400)  # sustained clutch: clutched alone holds the frame
    assert m.update(0.20, 0.147, 0.5) == (500, 400)  # slowing down releases the clutch with a re-anchor
    assert m.clutched is False
    assert m.update(0.25, 0.20, 0.5) == (500, 400)  # the re-anchored frame does not jump
    # free again: 0.01 of hand travel, softened toward the center (soft 0.928), is 9 px
    assert m.update(0.30, 0.21, 0.5) == (509, 400)


def test_reset_clears_the_autocenters_clutch():
    cfg = mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0)
    m = CursorMapper((1000, 800), cfg)
    m.update(0.0, 0.02, 0.5)
    m.update(0.05, 0.10, 0.5)
    assert m.autocenter.clutch.clutched is True
    assert m.clutched is True
    m.reset()  # calls autocenter.reset(), and clears the mapper's own flag too
    assert m.autocenter.clutch.clutched is False
    assert m.clutched is False


def test_absolute_mode_never_clutches():
    cfg = mut_cfg(mode="absolute", active_region=(0, 0, 1, 1), dead_zone=0.0)
    m = CursorMapper((1001, 1001), cfg)
    assert m.update(0.0, 0.02, 0.5) == (20, 500)
    # the same edge swing that clutches in relative mode maps normally here:
    # observe() is told clutch=False in absolute mode
    assert m.update(0.05, 0.10, 0.5) == (100, 500)


def _learned_comfort_mapper():
    cfg = mut_cfg(mode="absolute", active_region=(0.15, 0.15, 0.85, 0.85), dead_zone=0.0)
    m = CursorMapper((1001, 1001), cfg)
    for i in range(10):  # resting at (0.3, 0.3) for 2.25 s of small steps: the zone learns
        m.update(i * 0.25, 0.3, 0.3)
    assert m.autocenter.comfort.learned
    return m


def test_region_follows_the_learned_comfort_zone_after_reset():
    m = _learned_comfort_mapper()
    m.reset()
    assert m.region_center == pytest.approx((0.4787, 0.4787), abs=0.01)
    assert m.region == pytest.approx(shift_region((0.15, 0.15, 0.85, 0.85), m.region_center))
    assert m.region != (0.15, 0.15, 0.85, 0.85)  # not the raw config region, and not None


def test_region_stays_unset_when_the_comfort_path_is_off():
    m = _learned_comfort_mapper()
    m.cfg.autocenter.enabled = False
    m.reset()
    assert m.region_center is None  # enabled=False alone keeps the region unset
    m2 = _learned_comfort_mapper()
    m2.cfg.autocenter.comfort_zone = False
    m2.reset()
    assert m2.region_center is None  # comfort_zone=False alone keeps it unset too
    m3 = CursorMapper((1001, 1001), mut_cfg(mode="absolute", dead_zone=0.0))
    m3.reset()
    assert m3.region_center is None  # nothing learned: unset


def test_relative_soft_recenter_gain_off_centre():
    cfg = mut_cfg(gain=2.0, acceleration=1.0, dead_zone=0.0)  # autocenter on with soft recenter
    m = CursorMapper((1000, 1000), cfg)
    assert m.update(0.0, 0.9, 0.5) == (500, 500)
    # moving away from the center off-centre: soft = 1 + 0.12 * 0.8 = 1.096,
    # so 0.02 * 2.0 * 1.096 * 1000 = 43.84 px, not the 40 px of a unit soft gain
    assert m.update(0.1, 0.92, 0.5) == (544, 500)
    # moving back toward the center: soft = 1 - 0.12 * 0.84 = 0.8992, shedding speed
    assert m.update(0.2, 0.90, 0.5) == (508, 500)


def test_relative_without_autocenter_uses_unit_soft_gain():
    cfg = mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0)
    cfg.autocenter = None
    m = CursorMapper((1000, 1000), cfg)
    m.update(0.0, 0.5, 0.5)
    assert m.update(0.1, 0.51, 0.5) == (510, 500)  # exactly gain * delta * width, no doubling


def test_relative_sigmoid_curve_on_both_axes():
    cfg = mut_cfg(curve="sigmoid", gain=1.0, dead_zone=0.0)
    cfg.autocenter = None
    # k = 0.6 + 2.4 / (1 + exp(-6.25)) = 2.9954 at mag 0.04: 119.8 px of travel
    mx = CursorMapper((1000, 1000), cfg)
    mx.update(0.0, 0.5, 0.5)
    assert mx.update(0.1, 0.54, 0.5) == (620, 500)
    my = CursorMapper((1000, 1000), cfg)
    my.update(0.0, 0.5, 0.5)
    assert my.update(0.1, 0.5, 0.54) == (500, 620)


def test_relative_sigmoid_plateau_scales_with_reach():
    cfg = mut_cfg(curve="sigmoid", gain=1.0, dead_zone=0.0)
    cfg.autocenter = None
    mons = [Monitor(0, 0, 1000, 1000, True, "a"), Monitor(1000, 0, 1000, 1000, False, "b")]
    m = CursorMapper(mons, cfg)
    assert m.reach == pytest.approx(math.sqrt(2.0))
    m.update(0.0, 0.5, 0.5)
    # high = 0.6 + 2.4 * reach (a product): k = 3.9936 at mag 0.05, so 199.7 px
    assert m.update(0.1, 0.55, 0.5) == (700, 500)


def test_absolute_first_frame_uses_the_region_formula():
    cfg = mut_cfg(mode="absolute", active_region=(0.2, 0.2, 0.8, 0.8), dead_zone=0.0)
    m = CursorMapper((1001, 501), cfg)
    # u = (0.5 - 0.2) / (0.8 - 0.2) = 0.5 of 1000; v = (0.35 - 0.2) / 0.6 = 0.25 of 500
    assert m.update(0.0, 0.5, 0.35) == (500, 125)


def test_relative_full_frame_move_is_not_swallowed():
    cfg = mut_cfg(gain=1.0, acceleration=1.0, dead_zone=0.0)
    cfg.autocenter = None
    m = CursorMapper((1000, 1000), cfg)
    m.update(0.0, 0.0, 0.5)
    # dx is exactly 1.0 with dy exactly 0.0: a real move that must reach the clamp
    assert m.update(0.1, 1.0, 0.5) == (999, 500)
