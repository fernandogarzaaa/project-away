"""Mutation-strengthening tests for hands/autoscroll.py (v0.1.29)."""

from __future__ import annotations

import pytest

from away.hands.autoscroll import AutoScroller, axis_speed, direction


def test_axis_speed_dead_zone() -> None:
    assert axis_speed(0.05, dead_zone=0.1, gain=10.0, max_speed=5.0) == 0.0
    assert axis_speed(-0.05, dead_zone=0.1, gain=10.0, max_speed=5.0) == 0.0


def test_axis_speed_positive() -> None:
    s = axis_speed(0.5, dead_zone=0.1, gain=10.0, max_speed=5.0)
    assert s > 0


def test_axis_speed_negative() -> None:
    s = axis_speed(-0.5, dead_zone=0.1, gain=10.0, max_speed=5.0)
    assert s < 0


def test_axis_speed_capped() -> None:
    s = axis_speed(10.0, dead_zone=0.1, gain=10.0, max_speed=5.0)
    assert s == 5.0


def test_axis_speed_symmetric() -> None:
    pos = axis_speed(0.5, dead_zone=0.1, gain=10.0, max_speed=5.0)
    neg = axis_speed(-0.5, dead_zone=0.1, gain=10.0, max_speed=5.0)
    assert pos == -neg


def test_direction_none() -> None:
    assert direction(0, 0) is None


def test_direction_cardinal() -> None:
    assert direction(0, 1) == "up"
    assert direction(0, -1) == "down"
    assert direction(1, 0) == "right"
    assert direction(-1, 0) == "left"


def test_direction_diagonal_prefers_larger() -> None:
    assert direction(0.5, 1.0) == "up"  # |vy| > |vx|
    assert direction(1.0, 0.5) == "right"  # |vx| > |vy|


def test_scroller_stop_returns_was_active() -> None:
    sc = AutoScroller()
    assert sc.stop() is False  # not active
    sc.active = True
    assert sc.stop() is True
    assert sc.stop() is False  # now inactive


def test_scroller_stop_resets_state() -> None:
    sc = AutoScroller()
    sc.active = True
    sc.anchor = (0.5, 0.5)
    sc.vx, sc.vy = 1.0, 2.0
    sc.stop()
    assert sc.anchor is None
    assert sc.vx == 0.0 and sc.vy == 0.0
    assert sc.direction is None


def test_scroller_direction_none_when_inactive() -> None:
    sc = AutoScroller()
    sc.vx, sc.vy = 1.0, 1.0
    assert sc.direction is None  # not active


def test_scroller_lost_not_active() -> None:
    sc = AutoScroller()
    assert sc.lost(1.0) is None


def test_scroller_lost_stops_after_timeout() -> None:
    from away.hands.autoscroll import LOST_STOP_S

    sc = AutoScroller()
    sc.active = True
    # First call sets _lost_t0
    assert sc.lost(0.0) is None
    # Before timeout: None
    assert sc.lost(LOST_STOP_S / 2) is None
    # After timeout: stops
    result = sc.lost(LOST_STOP_S + 0.1)
    assert result == {"reason": "hand_lost"}
    assert not sc.active


def test_axis_speed_exact_dead_zone_and_curve_exponent() -> None:
    from away.hands.autoscroll import CURVE_EXPONENT

    assert axis_speed(0.1, dead_zone=0.1, gain=10.0, max_speed=5.0) == 0.0
    actual = axis_speed(0.3, dead_zone=0.1, gain=10.0, max_speed=5.0)
    expected = 10.0 * (0.2**CURVE_EXPONENT)
    assert abs(actual - expected) < 1e-9


def test_direction_tie_prefers_vertical_and_zero_speed_is_none() -> None:
    assert direction(1.0, 1.0) == "up"
    assert direction(-1.0, -1.0) == "down"
    assert direction(0.0, 0.0) is None


def test_start_hold_anchor_averages_recent_positions() -> None:
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        autoscroll_enabled=True,
        autoscroll_hold=0.2,
        autoscroll_dead_zone=0.1,
        autoscroll_gain=1.0,
        autoscroll_max_speed=5.0,
        autoscroll_horizontal=True,
        autoscroll_stop_hold=0.3,
    )
    sc = AutoScroller(cfg)
    sc.update(0.0, "four", (0.0, 0.0), 1.0)
    sc.update(0.1, "four", (0.2, 0.2), 1.0)
    ev = sc.update(0.2, "four", (0.4, 0.4), 1.0)
    assert ev is not None and ev[0] == "autoscroll_start"
    assert sc.anchor == pytest.approx((0.2, 0.2))


# Round 2: precise survivor targeting


def _active_scroller() -> AutoScroller:
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        autoscroll_enabled=True,
        autoscroll_hold=0.2,
        autoscroll_dead_zone=0.1,
        autoscroll_gain=1.0,
        autoscroll_max_speed=5.0,
        autoscroll_horizontal=True,
        autoscroll_stop_hold=0.3,
    )
    sc = AutoScroller(cfg)
    sc.update(0.0, "four", (0.0, 0.0), 1.0)
    sc.update(0.1, "four", (0.2, 0.2), 1.0)
    ev = sc.update(0.2, "four", (0.4, 0.4), 1.0)
    assert ev is not None and ev[0] == "autoscroll_start"
    return sc


def test_frame_dt_is_capped_at_max_frame_dt() -> None:
    # Kills MAX_FRAME_DT 0.1 -> 1.1: a 1.0s gap must still only scroll for 0.1s.
    sc = _active_scroller()
    ev = sc.update(1.2, "four", (0.2, -0.4), 1.0)  # hand moved up: vy = 0.5 notches/s
    assert ev is not None and ev[0] == "autoscroll"
    assert ev[1]["vy"] == pytest.approx(0.5)
    assert ev[1]["dy"] == pytest.approx(0.05)  # 0.5 * 0.1, not 0.5 * 1.0


def test_lost_exact_timeout_boundary_does_not_stop() -> None:
    # Kills `t - _lost_t0 > LOST_STOP_S` -> `>=`: at exactly 0.5s the hand is not lost yet.
    from away.hands.autoscroll import LOST_STOP_S

    sc = AutoScroller()
    sc.active = True
    assert sc.lost(0.0) is None
    assert sc.lost(LOST_STOP_S) is None  # exactly at the boundary: still scrolling
    assert sc.active is True


def test_hold_anchor_averages_last_hold_samples() -> None:
    # Kills HOLD_SAMPLES 8 -> 9 and HOLD_SAMPLES - 1 -> HOLD_SAMPLES + 1:
    # the neutral point is the mean of the last 8 positions, not 9 or 10.
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        autoscroll_enabled=True,
        autoscroll_hold=0.9,
        autoscroll_dead_zone=0.1,
        autoscroll_gain=1.0,
        autoscroll_max_speed=5.0,
        autoscroll_horizontal=True,
        autoscroll_stop_hold=0.3,
    )
    sc = AutoScroller(cfg)
    ev = None
    for i in range(10):
        ev = sc.update(i * 0.1, "four", (i * 0.1, i * 0.1), 1.0)
    assert ev is not None and ev[0] == "autoscroll_start"
    # mean of positions 2..9 (the last 8): (0.2 + ... + 0.9) / 8 = 0.55
    assert sc.anchor == pytest.approx((0.55, 0.55))


def test_other_pose_first_frame_does_not_stop_immediately() -> None:
    # Kills `t - _other_t0` -> `t + _other_t0`: on the first other-pose frame the
    # elapsed hold time is 0, so scrolling continues (mutant stops instantly at large t).
    sc = _active_scroller()
    ev = sc.update(1.0, "point", (0.2, 0.2), 1.0)
    assert ev is not None and ev[0] == "autoscroll"  # not "autoscroll_stop"
    assert sc.active is True


def test_diagonal_tie_prefers_vertical() -> None:
    # Kills `abs(vx) > abs(vy)` -> `>=`: on a tie the vertical axis wins (vx = 0).
    sc = _active_scroller()
    ev = sc.update(0.3, "four", (0.5, -0.1), 1.0)  # |ox| == |oy| == 0.3
    assert ev is not None and ev[0] == "autoscroll"
    assert ev[1]["vx"] == 0.0
    assert ev[1]["vy"] == pytest.approx(0.2)


# Round 3: constructor state margin


def test_new_scroller_has_zero_velocity_and_no_anchor() -> None:
    # Kills the 0.0 -> 1.0 constant mutants on the initial vx/vy.
    sc = AutoScroller()
    assert sc.vx == 0.0
    assert sc.vy == 0.0
    assert sc.anchor is None
    assert sc.active is False
