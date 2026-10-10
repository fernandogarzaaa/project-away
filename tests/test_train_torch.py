"""Training/export tests. Need the [train] extra (torch, onnx, onnxruntime)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from away.hands import train as T  # noqa: E402
from away.hands.landmarks import static_features  # noqa: E402
from away.hands.model import NumpyMLP, OnnxClassifier  # noqa: E402
from away.hands.runtime import TemporalGestureClassifier  # noqa: E402
from away.hands.synthetic import GESTURES, gesture_sequence, pose  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.slow]


def test_confusion_and_f1():
    cm = T.confusion(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
    assert cm.tolist() == [[1, 1], [0, 2]]
    assert T.macro_f1(cm) == pytest.approx((2 / 3 + 0.8) / 2)
    assert T.macro_f1(np.zeros((2, 2), dtype=np.int64)) == 0.0


def _fake_hagrid(folder, n=60):
    rng = np.random.default_rng(0)
    for cls, p in (("palm", "palm"), ("fist", "fist"), ("peace", "two")):
        data = {}
        for i in range(n):
            lm = pose(
                p, center=(rng.uniform(0.3, 0.7), 0.5), rotation=rng.uniform(-0.3, 0.3), scale=rng.uniform(0.08, 0.15)
            )[:, :2]
            lm = lm + rng.normal(0, 0.002, lm.shape)
            data[f"{cls}{i}"] = {"labels": [cls], "hand_landmarks": [lm.tolist()], "user_id": f"u{i}"}
        (folder / f"test_{cls}.json").write_text(json.dumps(data))


def test_train_static_exports_consistent_models(tmp_path):
    _fake_hagrid(tmp_path)
    m = T.train_static(tmp_path, tmp_path / "out", epochs=15)
    assert m["test_accuracy"] > 0.9
    assert m["n_test"] > 0 and m["classes_present"] == ["palm", "fist", "two"]
    npm = NumpyMLP.load(tmp_path / "out" / "static_pose.npz")
    onx = OnnxClassifier(tmp_path / "out" / "static_pose.onnx")
    x = np.stack([static_features(pose(p)) for p in ("palm", "fist", "two")])
    assert np.allclose(npm.predict_proba(x), onx.predict_proba(x), atol=1e-4)
    assert onx.classes == npm.classes


def test_train_temporal_and_runtime(tmp_path):
    rec = tmp_path / "rec"
    from away.hands.recorder import save_clip

    rng = np.random.default_rng(0)
    for _ in range(5):
        save_clip(rec, "swipe_left", list(gesture_sequence("swipe_left", 20, rng)), "Right", 30)
    m = T.train_temporal(tmp_path / "out", recordings=rec, n_synthetic=12, epochs=3)
    assert m["classes"] == list(GESTURES)
    assert "synthetic_holdout_accuracy" in m
    clf = TemporalGestureClassifier(tmp_path / "out" / "gesture_tcn.onnx")
    seq = gesture_sequence("swipe_right", 30, rng)
    outs = [clf.push(f) for f in seq]
    assert outs[clf.frames - 2] is None and outs[-1] is not None
    assert set(outs[-1]) == set(GESTURES)
    assert sum(outs[-1].values()) == pytest.approx(1.0, abs=1e-5)
    clf.push(seq[0], "Left")
    assert len(clf.buf) == 1  # handedness switch clears the window
    clf.reset()
    assert len(clf.buf) == 0
    m2 = T.calibrate(tmp_path / "out", rec, epochs=1)
    assert m2["model"] == "gesture_tcn"


def test_train_temporal_needs_data(tmp_path):
    with pytest.raises(ValueError):
        T.train_temporal(tmp_path, n_synthetic=0, epochs=1)
