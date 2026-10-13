"""Gesture Studio with the [train] extra: movements are learned with the temporal TCN and exported to ONNX."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from away.hands.synthetic import gesture_sequence
from away.studio import trainer as T
from away.studio.engine import StudioRecognizer
from away.studio.pack import NOTHING
from studio_helpers import motion_store, record_nothing, record_static

pytestmark = pytest.mark.torch


def test_movement_learned_with_the_tcn_runs_through_onnx(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(6)
    store = motion_store(tmp_path, rng, per_round=3)
    res = T.train_studio(store, backend="torch")
    assert res.ok and res.backend == "torch" and res.score is not None
    meta = json.loads((store.model_dir / "studio.json").read_text())
    assert meta["groups"][0]["backend"] == "tcn" and (store.model_dir / "motion1.onnx").exists()
    rec = StudioRecognizer(store.model_dir)
    assert rec.groups[0].backend == "tcn"
    probs = None
    for i, f in enumerate(gesture_sequence("swipe_right", frames=36, rng=rng)):
        probs = rec.push(i / 30.0, f) or probs
    assert probs is not None and set(probs) == {NOTHING, "flick"} and probs["flick"] > probs[NOTHING]


def test_pose_learned_with_torch_uses_the_static_pose_mlp(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from away.hands.synthetic import pose
    from away.studio.pack import StudioStore

    rng = np.random.default_rng(7)
    store = StudioStore(tmp_path)
    store.add("fist_hold")
    record_static(store, "fist_hold", "fist", 3, 3, rng)
    record_nothing(store, 3, rng)
    res = T.train_studio(store, backend="torch")
    assert res.ok
    meta = json.loads((store.model_dir / "studio.json").read_text())
    assert meta["groups"][0]["backend"] == "mlp" and meta["backend"] == "torch"  # numpy runtime either way
    with np.load(store.model_dir / "static1.npz") as d:
        assert {"w0", "w1", "w2"} <= set(d.files)  # the StaticPoseMLP's three Linear layers
    rec = StudioRecognizer(store.model_dir)
    p = rec.push(0.0, pose("fist"))
    assert p is not None and p["fist_hold"] > 0.5
