import math

import pytest

from away.hands.filters import OneEuro2D, OneEuroFilter, exponential_smoothing, smoothing_factor


def test_smoothing_factor_formula():
    r = 2 * math.pi * 1.0 * 0.01
    assert smoothing_factor(0.01, 1.0) == pytest.approx(r / (r + 1))
    assert smoothing_factor(1.0, 1000.0) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("t_e,cutoff", [(0, 1.0), (-1, 1.0), (0.1, 0), (0.1, -2)])
def test_smoothing_factor_rejects_nonpositive(t_e, cutoff):
    with pytest.raises(ValueError):
        smoothing_factor(t_e, cutoff)


def test_exponential_smoothing():
    assert exponential_smoothing(0.25, 10.0, 2.0) == pytest.approx(4.0)


@pytest.mark.parametrize("kw", [{"min_cutoff": 0}, {"d_cutoff": 0}, {"beta": -0.1}])
def test_filter_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        OneEuroFilter(**kw)


def test_first_sample_passes_through():
    f = OneEuroFilter()
    assert f(0.0, 5.0) == 5.0


def test_exact_second_step_without_beta():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f(0.0, 0.0)
    a = smoothing_factor(0.1, 1.0)
    assert f(0.1, 1.0) == pytest.approx(a * 1.0)


def test_beta_raises_cutoff_for_fast_motion():
    slow, fast = OneEuroFilter(1.0, 0.0), OneEuroFilter(1.0, 1.0)
    for f in (slow, fast):
        f(0.0, 0.0)
    assert fast(0.1, 10.0) > slow(0.1, 10.0)


def test_derivative_uses_d_cutoff():
    a, b = OneEuroFilter(1.0, 1.0, d_cutoff=0.1), OneEuroFilter(1.0, 1.0, d_cutoff=10.0)
    for f in (a, b):
        f(0.0, 0.0)
    assert a(0.1, 5.0) != pytest.approx(b(0.1, 5.0))


def test_converges_on_constant_and_reduces_jitter(rng):
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = 1.0 + rng.normal(0, 0.05, 300)
    out = [f(i / 60, x) for i, x in enumerate(raw)]
    import numpy as np

    assert np.std(out[100:]) < 0.5 * np.std(raw[100:])
    assert np.mean(out[100:]) == pytest.approx(1.0, abs=0.02)


def test_duplicate_timestamp_holds_value():
    f = OneEuroFilter()
    f(0.0, 0.0)
    v = f(0.1, 1.0)
    assert f(0.1, 50.0) == v
    assert f(0.05, 50.0) == v


def test_state_is_updated_between_calls():
    f = OneEuroFilter(1.0, 0.0)
    f(0.0, 0.0)
    v1 = f(0.1, 1.0)
    a = smoothing_factor(0.1, 1.0)
    assert f(0.2, 1.0) == pytest.approx(a * 1.0 + (1 - a) * v1)


def test_reset():
    f = OneEuroFilter()
    f(0.0, 0.0)
    f(0.1, 1.0)
    f.reset()
    assert f(5.0, 7.0) == 7.0


def test_2d_filters_axes_independently():
    f = OneEuro2D(1.0, 0.0)
    assert f(0.0, 1.0, 2.0) == (1.0, 2.0)
    x, y = f(0.1, 2.0, 2.0)
    assert y == pytest.approx(2.0)
    assert 1.0 < x < 2.0
    f.reset()
    assert f(1.0, 9.0, 9.0) == (9.0, 9.0)
