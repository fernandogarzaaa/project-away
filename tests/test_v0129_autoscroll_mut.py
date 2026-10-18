"""Mutation-strengthening tests for hands/autoscroll.py (v0.1.29)."""

from __future__ import annotations

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
