"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

Low jitter at low speed, low lag at high speed: the cutoff frequency rises
with the (smoothed) speed of the signal.
"""

from __future__ import annotations

import math


def smoothing_factor(t_e: float, cutoff: float) -> float:
    """alpha for an exponential smoother with the given cutoff (Hz) and period t_e (s)."""
    if t_e <= 0:
        raise ValueError("t_e must be positive")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(alpha: float, x: float, x_prev: float) -> float:
    return alpha * x + (1 - alpha) * x_prev


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    def __call__(self, t: float, x: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev, self._t_prev, self._dx_prev = x, t, 0.0
            return x
        t_e = t - self._t_prev
        if t_e <= 0:  # duplicate / out-of-order timestamp: hold the last output
            return self._x_prev
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self._dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self._x_prev)
        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


class OneEuro2D:
    """Independent One Euro filters on x and y."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]:
        return self.fx(t, x), self.fy(t, y)
