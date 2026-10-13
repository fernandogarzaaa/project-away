"""Synthetic practice recordings for the Gesture Studio tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from away.hands.synthetic import gesture_sequence, pose
from away.studio.pack import NOTHING, StudioStore

OTHER_POSES = ("palm", "point", "two")


def jitter(lm: np.ndarray, rng: np.random.Generator, amount: float = 0.004) -> np.ndarray:
    out = lm.copy()
    out[:, :2] += rng.normal(0, amount, (21, 2))
    return out


def record_static(store: StudioStore, name: str, base: str, rounds: int, per_round: int, rng) -> None:
    for r in range(rounds):
        sid = f"{name}-round-{r}"
        for _ in range(per_round):
            center = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
            scale = rng.uniform(0.1, 0.15)
            rot = rng.uniform(-0.3, 0.3)
            frames = [(jitter(pose(base, center=center, scale=scale, rotation=rot), rng), None) for _ in range(12)]
            store.save_take(name, frames, "Right", 30.0, sid)


def record_nothing(store: StudioStore, rounds: int, rng, frames_per: int = 60) -> None:
    for r in range(rounds):
        frames = []
        for i in range(frames_per):
            base = OTHER_POSES[(i // 10) % len(OTHER_POSES)]
            frames.append((jitter(pose(base, center=(0.4 + 0.002 * i, 0.5)), rng), None))
        store.save_take(NOTHING, frames, "Right", 30.0, f"relax-{r}")


def motion_store(root: Path, rng, per_round: int = 4) -> StudioStore:
    store = StudioStore(root)
    store.add("flick", "motion", 1)
    for r in range(3):
        for _ in range(per_round):
            seq = gesture_sequence("swipe_right", frames=36, rng=rng)
            store.save_take("flick", [(f, None) for f in seq], "Right", 30.0, f"flick-{r}")
    for r in range(3):
        seq = np.concatenate([gesture_sequence("idle", frames=45, rng=rng) for _ in range(2)])
        store.save_take(NOTHING, [(f, None) for f in seq], "Right", 30.0, f"relax-{r}")
    return store
