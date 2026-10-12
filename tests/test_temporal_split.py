from __future__ import annotations

import numpy as np

from away.hands.train import temporal_split


def test_recorded_groups_never_cross_splits() -> None:
    source = np.array(["recorded"] * 12 + ["synthetic"] * 30)
    groups = [f"session-{i // 2}" for i in range(12)] + [f"synthetic-{i}" for i in range(30)]
    train, val, test = temporal_split(source, groups, np.random.default_rng(7))
    assert np.all(train.astype(int) + val.astype(int) + test.astype(int) == 1)
    assignments: dict[str, set[str]] = {}
    for i, group in enumerate(groups[:12]):
        split = "train" if train[i] else "val" if val[i] else "test"
        assignments.setdefault(group, set()).add(split)
    assert all(len(splits) == 1 for splits in assignments.values())
    assert np.any(test & (source == "recorded"))
    assert np.any(val & (source == "recorded"))


def test_too_few_recorded_groups_are_not_claimed_as_test() -> None:
    source = np.array(["recorded", "recorded", "recorded", "synthetic", "synthetic", "synthetic"])
    groups = ["one", "one", "two", "s0", "s1", "s2"]
    train, val, test = temporal_split(source, groups, np.random.default_rng(1))
    assert np.all(train[:3])
    assert not np.any(val[:3])
    assert not np.any(test[:3])


def test_split_is_deterministic_for_same_seed() -> None:
    source = np.array(["recorded"] * 9 + ["synthetic"] * 12)
    groups = [f"g{i // 3}" for i in range(9)] + [f"s{i}" for i in range(12)]
    a = temporal_split(source, groups, np.random.default_rng(42))
    b = temporal_split(source, groups, np.random.default_rng(42))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
