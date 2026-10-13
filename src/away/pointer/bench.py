"""Synthetic jitter / latency benchmark for the cursor filters (``hands.cursor.smoothing``).

The "true" hand moves in minimum-jerk reaches between random points with rests in between (how people point).
The filter sees it the way the webcam pipeline does: ``latency_frames`` late and with Gaussian landmark jitter.
Each filter output is compared with where the hand *is now*:

* ``lag_rmse``: RMS distance to the true position while the hand moves (lag + noise), hand units.
* ``still_jitter``: RMS frame-to-frame movement of the output for a hand held perfectly still (what makes a cursor
  shimmer; the cursor's ``dead_zone``, 0.004 by default, swallows per-frame moves below it).
* ``overshoot``: the largest distance past the end point of a reach during the following rest.

Everything is deterministic for a given ``seed``; tests assert the predictive filter beats One Euro on lag
while a resting cursor stays inside the dead zone. ``python -m away.pointer.bench`` prints the table.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from away.pointer.smoothing import CursorFilter


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray  # (n,) seconds
    truth: np.ndarray  # (n, 2) true hand position now
    measured: np.ndarray  # (n, 2) what the filter sees (late + jitter)
    moving: np.ndarray  # (n,) bool
    reach_end: np.ndarray  # (n,) index of the end of the last reach (-1 before the first)


@dataclass(frozen=True)
class Score:
    lag_rmse: float
    still_jitter: float
    overshoot: float


def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def synthetic(
    seed: int = 0,
    fps: float = 30.0,
    reaches: int = 40,
    jitter: float = 0.002,
    latency_frames: int = 1,
) -> Trajectory:
    rng = np.random.default_rng(seed)
    pos = np.array([0.5, 0.5])
    pts: list[np.ndarray] = []
    moving: list[bool] = []
    ends: list[int] = []
    last_end = -1
    for _ in range(reaches):
        goal = rng.uniform(0.2, 0.8, size=2)
        dur = float(rng.uniform(0.25, 0.8))
        n = max(2, round(dur * fps))
        s = min_jerk(np.linspace(0, 1, n + 1)[1:])
        for k in s:
            pts.append(pos + (goal - pos) * k)
            moving.append(True)
            ends.append(last_end)
        pos = goal
        last_end = len(pts) - 1
        for _ in range(round(float(rng.uniform(0.4, 1.0)) * fps)):
            pts.append(pos.copy())
            moving.append(False)
            ends.append(last_end)
    truth = np.array(pts)
    n = len(truth)
    late = np.vstack([np.repeat(truth[:1], latency_frames, axis=0), truth[: n - latency_frames]])
    measured = late + rng.normal(0.0, jitter, size=truth.shape)
    t = np.arange(n) / fps
    return Trajectory(t, truth, measured, np.array(moving), np.array(ends))


def run(filt: CursorFilter, traj: Trajectory) -> np.ndarray:
    filt.reset()
    return np.array([filt(float(t), float(x), float(y)) for t, (x, y) in zip(traj.t, traj.measured)])


def still_jitter(filt: CursorFilter, jitter: float = 0.002, frames: int = 150, fps: float = 30.0, seed: int = 0) -> float:
    """RMS frame-to-frame output movement for a resting hand seen with landmark jitter (first second skipped)."""
    rng = np.random.default_rng(seed)
    filt.reset()
    noise = rng.normal(0.0, jitter, size=(frames, 2))
    out = np.array([filt(i / fps, 0.5 + float(e[0]), 0.5 + float(e[1])) for i, e in enumerate(noise)])
    d = np.linalg.norm(np.diff(out[int(fps) :], axis=0), axis=1)
    return float(math.sqrt(np.mean(d**2)))


def lag_rmse(out: np.ndarray, traj: Trajectory, warmup: int = 15) -> float:
    err = np.linalg.norm(out - traj.truth, axis=1)
    mv = traj.moving.copy()
    mv[:warmup] = False
    return float(math.sqrt(np.mean(err[mv] ** 2))) if mv.any() else 0.0


def overshoot(out: np.ndarray, traj: Trajectory, warmup: int = 15) -> float:
    over = 0.0
    for i in np.flatnonzero(~traj.moving):
        e = int(traj.reach_end[i])
        if e < warmup:
            continue
        direction = traj.truth[e] - traj.truth[max(0, e - 1)]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        over = max(over, float(np.dot(out[i] - traj.truth[e], direction / norm)))
    return over


def compare(
    filters: dict[str, Callable[[], CursorFilter]],
    seeds: Iterable[int] = range(5),
    **traj_kw: float,
) -> dict[str, Score]:
    """Average :class:`Score` per filter over several seeds."""
    seeds = list(seeds)
    acc: dict[str, list[Score]] = {k: [] for k in filters}
    jit = float(traj_kw.get("jitter", 0.002))
    for s in seeds:
        traj = synthetic(seed=s, **traj_kw)  # type: ignore[arg-type]
        for name, make in filters.items():
            out = run(make(), traj)
            acc[name].append(Score(lag_rmse(out, traj), still_jitter(make(), jit, seed=s), overshoot(out, traj)))
    return {
        k: Score(
            float(np.mean([x.lag_rmse for x in v])),
            float(np.mean([x.still_jitter for x in v])),
            float(np.mean([x.overshoot for x in v])),
        )
        for k, v in acc.items()
    }


def default_filters() -> dict[str, Callable[[], CursorFilter]]:
    """The shipped defaults of both ``hands.cursor.smoothing`` modes."""
    from away.config import CursorConfig
    from away.pointer.smoothing import make_cursor_filter

    def one_euro() -> CursorFilter:
        return make_cursor_filter(CursorConfig())

    def predictive() -> CursorFilter:
        c = CursorConfig()
        c.smoothing = "predictive"
        return make_cursor_filter(c)

    return {"one_euro": one_euro, "predictive": predictive}


def main() -> None:  # pragma: no cover - prints a table
    res = compare(default_filters())
    print(f"{'filter':<12}{'lag RMSE':>12}{'still jitter':>14}{'overshoot':>12}   (hand units; 1 frame latency)")
    for name, s in res.items():
        print(f"{name:<12}{s.lag_rmse:>12.5f}{s.still_jitter:>14.6f}{s.overshoot:>12.5f}")


if __name__ == "__main__":  # pragma: no cover
    main()
