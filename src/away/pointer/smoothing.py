"""Cursor smoothing: the existing One Euro filter or a predictive (lookahead) filter.

``hands.cursor.smoothing``:

* ``adaptive`` (default since v0.1.25): :class:`away.hands.steady.AdaptiveFilter2D`.
* ``one_euro``: :class:`away.hands.filters.OneEuro2D`, unchanged.
* ``predictive``: :class:`PredictiveFilter2D`, a speed-adaptive constant-velocity Kalman filter whose output is
  extrapolated ``lookahead_ms`` into the future. A webcam pipeline shows the hand roughly one frame late (exposure,
  transfer, hand landmarking); predicting that frame ahead cancels most of the lag while the Kalman filter keeps
  the jitter down. The prediction fades out at low speed (``speed_gate``), so a resting hand does not shimmer
  from noisy velocity estimates, and it is capped so an abrupt stop overshoots by at most a few pixels.

Both filters take ``(t, x, y)`` in hand units (0..1 of the camera frame) and return the filtered ``(x, y)``;
:class:`~away.hands.cursor.CursorMapper` calls whichever :func:`make_cursor_filter` returns. The benchmark in
:mod:`away.pointer.bench` measures them against each other on synthetic hand motion with jitter and latency.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

from away.hands.filters import OneEuro2D
from away.pointer.config import SMOOTHING_MODES

__all__ = ["SMOOTHING_MODES", "CursorFilter", "PredictiveFilter2D", "make_cursor_filter", "predictive_from_config"]


class CursorFilter(Protocol):
    def reset(self) -> None: ...
    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]: ...


class _Axis:
    """One axis of the constant-velocity Kalman filter: state ``[position, velocity]`` and its covariance."""

    __slots__ = ("p00", "p01", "p11", "v", "x")

    def __init__(self, x: float, r2: float) -> None:
        self.x, self.v = x, 0.0
        self.p00, self.p01, self.p11 = r2, 0.0, 1.0

    def step(self, z: float, dt: float, q: float, r2: float) -> None:
        # predict (continuous white-noise acceleration, spectral density q)
        x = self.x + self.v * dt
        p00 = self.p00 + dt * (2 * self.p01 + dt * self.p11) + q * dt**3 / 3.0
        p01 = self.p01 + dt * self.p11 + q * dt**2 / 2.0
        p11 = self.p11 + q * dt
        # update with the measurement z
        s = p00 + r2
        k0, k1 = p00 / s, p01 / s
        innov = z - x
        self.x = x + k0 * innov
        self.v = self.v + k1 * innov
        self.p00 = (1 - k0) * p00
        self.p01 = (1 - k0) * p01
        self.p11 = p11 - k1 * p01


REST_NOISE_FRACTION = 0.002  # process noise at rest, as a fraction of ``process_noise``
ADAPT_SPEED = 0.3  # hand units / s at which the full ``process_noise`` applies


class PredictiveFilter2D:
    """Speed-adaptive constant-velocity Kalman filter on x and y, with lookahead.

    * The process noise (how quickly the filter believes the hand can change speed) scales with the squared
      estimated speed between ``process_noise * REST_NOISE_FRACTION`` (resting: heavy smoothing, a still cursor)
      and ``process_noise`` (moving at ``ADAPT_SPEED`` or faster: the filter follows closely). Same idea as the
      One Euro filter's speed-dependent cutoff, in a model that also estimates velocity.
    * ``measurement_noise``: landmark jitter, standard deviation in hand units.
    * Output: ``position + velocity * lookahead_s * gate``; ``gate`` rises smoothly from 0 to 1 as the 2D speed
      goes from 0 to ``speed_gate`` (no prediction for a resting hand) and the lead is capped at ``max_lead``.
    """

    def __init__(
        self,
        lookahead_s: float = 0.033,
        process_noise: float = 5.0,
        measurement_noise: float = 0.004,
        speed_gate: float = 0.3,
        max_lead: float = 0.05,
    ) -> None:
        if lookahead_s < 0:
            raise ValueError("lookahead must be >= 0")
        if process_noise <= 0 or measurement_noise <= 0:
            raise ValueError("noise levels must be positive")
        self.lookahead_s = float(lookahead_s)
        self.q = float(process_noise)
        self.r2 = float(measurement_noise) ** 2
        self.speed_gate = max(1e-9, float(speed_gate))
        self.max_lead = float(max_lead)
        self.reset()

    def reset(self) -> None:
        self._t: float | None = None
        self._ax: _Axis | None = None
        self._ay: _Axis | None = None
        self._out = (0.0, 0.0)

    @property
    def velocity(self) -> tuple[float, float]:
        if self._ax is None or self._ay is None:
            return 0.0, 0.0
        return self._ax.v, self._ay.v

    @property
    def speed(self) -> float:
        return math.hypot(*self.velocity)

    def gate(self, speed: float) -> float:
        g = min(1.0, abs(speed) / self.speed_gate)
        return g * g * (3 - 2 * g)  # smoothstep: no kink where the prediction starts

    def process_noise_at(self, speed: float) -> float:
        k = min(1.0, abs(speed) / ADAPT_SPEED)
        lo = self.q * REST_NOISE_FRACTION
        return lo + (self.q - lo) * k * k

    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]:
        if self._t is None or self._ax is None or self._ay is None:
            self._t = t
            self._ax, self._ay = _Axis(x, self.r2), _Axis(y, self.r2)
            self._out = (x, y)
            return self._out
        dt = t - self._t
        if dt <= 0:  # duplicate / out-of-order timestamp: hold the last output
            return self._out
        dt = min(dt, 0.5)  # a long gap (hand briefly lost) must not blow the covariance up
        self._t = t
        q = self.process_noise_at(self.speed)
        self._ax.step(x, dt, q, self.r2)
        self._ay.step(y, dt, q, self.r2)
        vx, vy = self._ax.v, self._ay.v
        g = self.gate(math.hypot(vx, vy))
        lx, ly = vx * self.lookahead_s * g, vy * self.lookahead_s * g
        lead = math.hypot(lx, ly)
        if lead > self.max_lead > 0:
            lx, ly = lx * self.max_lead / lead, ly * self.max_lead / lead
        self._out = (self._ax.x + lx, self._ay.x + ly)
        return self._out


def predictive_from_config(p: Any) -> PredictiveFilter2D:
    """A :class:`PredictiveFilter2D` from a ``hands.cursor.predictive`` section."""
    return PredictiveFilter2D(
        lookahead_s=float(p.lookahead_ms) / 1000.0,
        process_noise=float(p.process_noise),
        measurement_noise=float(p.measurement_noise),
        speed_gate=float(p.speed_gate),
    )


def make_cursor_filter(cfg: Any) -> CursorFilter:
    """The hand-position filter for a ``hands.cursor`` section (``smoothing`` picks it; One Euro otherwise)."""
    mode = getattr(cfg, "smoothing", "one_euro")
    if mode == "predictive" and getattr(cfg, "predictive", None) is not None:
        return predictive_from_config(cfg.predictive)
    if mode == "adaptive":  # v0.1.25
        from away.hands.steady import AdaptiveFilter2D

        return AdaptiveFilter2D(
            float(cfg.min_cutoff), float(getattr(cfg, "fast_cutoff", 8.0)), float(getattr(cfg, "fast_speed", 0.5))
        )
    return OneEuro2D(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
