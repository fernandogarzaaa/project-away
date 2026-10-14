import pytest

from away.config import CursorConfig
from away.hands.cursor import ACCEL_REF, CursorMapper, accelerate, apply_dead_zone


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
