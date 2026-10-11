"""Input validation regression tests for camera frame pairing."""

import math

import pytest

from away.multicam.capture import FramePairer


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_cams": 0}, "n_cams"),
        ({"n_cams": 2, "primary": -1}, "primary"),
        ({"n_cams": 2, "primary": 2}, "primary"),
        ({"n_cams": 1, "tolerance_s": -0.1}, "tolerance_s"),
        ({"n_cams": 1, "tolerance_s": math.inf}, "tolerance_s"),
        ({"n_cams": 1, "dropout_s": 0}, "dropout_s"),
        ({"n_cams": 1, "dropout_s": math.nan}, "dropout_s"),
        ({"n_cams": 1, "maxlen": 0}, "maxlen"),
    ],
)
def test_frame_pairer_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FramePairer(**kwargs)


def test_push_rejects_invalid_camera_index():
    pairer = FramePairer(2)
    with pytest.raises(ValueError, match="camera index"):
        pairer.push(2, 1.0, object())
    with pytest.raises(ValueError, match="camera index"):
        pairer.push(-1, 1.0, object())


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, -math.inf])
def test_push_rejects_non_finite_timestamps(timestamp):
    pairer = FramePairer(1)
    with pytest.raises(ValueError, match="timestamp must be finite"):
        pairer.push(0, timestamp, object())


def test_push_rejects_out_of_order_timestamps_without_corrupting_buffer():
    pairer = FramePairer(1, dropout_s=2.0)
    pairer.push(0, 1.0, "first")
    with pytest.raises(ValueError, match="non-decreasing"):
        pairer.push(0, 0.5, "older")
    pairer.push(0, 1.5, "second")

    result = pairer.pop_set(now=1.5)
    assert result == (1.5, [(1.5, "second")])


@pytest.mark.parametrize("now", [math.nan, math.inf, -math.inf])
def test_pop_set_rejects_non_finite_clock_override(now):
    pairer = FramePairer(1)
    with pytest.raises(ValueError, match="now must be finite"):
        pairer.pop_set(now=now)


def test_frame_pairer_accepts_equal_timestamps_and_zero_tolerance():
    pairer = FramePairer(2, tolerance_s=0.0, dropout_s=1.0)
    pairer.push(0, 2.0, "primary")
    pairer.push(1, 2.0, "secondary")

    assert pairer.pop_set(now=2.0) == (2.0, [(2.0, "primary"), (2.0, "secondary")])
