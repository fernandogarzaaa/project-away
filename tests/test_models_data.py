import http.server
import json
import threading
import zipfile

import numpy as np
import pytest

from away.hands import dataset as D
from away.hands.landmarks import infer_side, static_features
from away.hands.model import STATIC_CLASSES, NumpyMLP, softmax
from away.hands.recorder import save_clip
from away.hands.runtime import StaticPoseClassifier, bundled_model, find_model
from away.hands.synthetic import GESTURES, gesture_sequence, make_dataset, make_hand, pose


def test_softmax():
    p = softmax(np.array([[0.0, 0.0], [1000.0, 0.0]]))
    assert np.allclose(p, [[0.5, 0.5], [1.0, 0.0]])


def test_numpy_mlp_forward():
    w = {"w0": np.eye(2), "b0": np.array([0.0, -10.0]), "w1": np.array([[1.0, 0.0], [0.0, 1.0]]), "b1": np.zeros(2)}
    m = NumpyMLP(w, ["a", "b"])
    p = m.predict_proba(np.array([1.0, 1.0]))  # relu kills the second unit
    assert p.shape == (1, 2)
    assert p[0, 0] == pytest.approx(np.exp(1) / (np.exp(1) + 1))
    with pytest.raises(ValueError):
        NumpyMLP({}, [])


def test_bundled_static_model_on_synthetic_poses():
    path = bundled_model("static_pose.npz")
    assert path is not None
    clf = StaticPoseClassifier(path)
    assert clf.classes == list(STATIC_CLASSES)
    # NOTE: the model is trained on *real* HaGRID hands; the procedural "point" hand is
    # outside its distribution (it reads as "two"), so only clear-cut poses are checked here.
    for name, expected in (("palm", "palm"), ("fist", "fist"), ("two", "two")):
        label, conf = clf.predict(pose(name))
        assert label == expected, (name, label, conf)
        assert 0 < conf <= 1


def test_find_model(tmp_path):
    assert find_model("static_pose.npz", None) == bundled_model("static_pose.npz")
    (tmp_path / "static_pose.npz").write_bytes(b"x")
    assert find_model("static_pose.npz", tmp_path) == tmp_path / "static_pose.npz"
    assert find_model("does_not_exist.onnx", tmp_path) is None


def test_synthetic_sequences():
    rng = np.random.default_rng(0)
    for g in GESTURES:
        s = gesture_sequence(g, 16, rng)
        assert s.shape == (16, 21, 3)
    with pytest.raises(ValueError):
        gesture_sequence("dab")
    X, y, hands = make_dataset(3, 8, seed=1)
    assert X.shape == (27, 8, 21, 3) and list(np.bincount(y)) == [3] * 9
    assert set(hands) <= {"Left", "Right"}
    sw = gesture_sequence("swipe_right", 24, np.random.default_rng(1), noise=0)
    assert sw[-1, 0, 0] > sw[0, 0, 0] + 0.2
    sl = gesture_sequence("swipe_left", 24, np.random.default_rng(1), noise=0)
    assert sl[-1, 0, 0] < sl[0, 0, 0] - 0.2


def test_make_hand_pinch_and_mirror():
    h = make_hand(pinch=1.0)
    assert np.allclose(h[4, :2], h[8, :2])
    r, left = make_hand(handedness="Right"), make_hand(handedness="Left")
    assert np.allclose(r[:, 0] - 0.5, -(left[:, 0] - 0.5))


def write_hagrid(path, cls, users, label=None):
    data = {}
    for i, u in enumerate(users):
        lm = pose("palm" if cls == "palm" else "point", center=(0.5, 0.5))[:, :2].tolist()
        data[f"img{i}"] = {"labels": [label or cls, "no_gesture"], "hand_landmarks": [lm, []], "user_id": u}
    data["bad"] = {"labels": [cls], "hand_landmarks": [[[0, 0]] * 21], "user_id": "z"}  # degenerate hand
    data["short"] = {"labels": [cls], "hand_landmarks": [[[0, 0]] * 5], "user_id": "z"}
    path.write_text(json.dumps(data))


def test_load_hagrid(tmp_path):
    write_hagrid(tmp_path / "test_palm.json", "palm", [f"u{i}" for i in range(10)])
    write_hagrid(tmp_path / "test_peace.json", "peace", ["a", "b"])
    (tmp_path / "test_unknownclass.json").write_text("{}")
    rows = D.load_hagrid_json(tmp_path / "test_palm.json", "palm")
    assert len(rows) == 10 and rows[0][1] == "palm" and rows[0][2] == "u0"
    assert np.allclose(rows[0][0], static_features(pose("palm")))
    X, y, users = D.load_hagrid_dir(tmp_path)
    assert X.shape == (12, 42)
    assert sorted(set(STATIC_CLASSES[i] for i in y)) == ["palm", "two"]
    X2, _, _ = D.load_hagrid_dir(tmp_path, max_per_class=3)
    assert len(X2) == 5
    with pytest.raises(FileNotFoundError):
        D.load_hagrid_dir(tmp_path / "empty")


def test_subject_split_is_disjoint_and_deterministic():
    users = [f"user{i % 50}" for i in range(500)]
    tr, va, te = D.subject_split(users)
    assert not (tr & te).any() and not (tr & va).any() and not (va & te).any()
    assert (tr | va | te).all()
    test_users = {u for u, m in zip(users, te) if m}
    train_users = {u for u, m in zip(users, tr) if m}
    assert test_users and not (test_users & train_users)
    assert all(np.array_equal(a, b) for a, b in zip(D.subject_split(users), (tr, va, te)))


def test_fixed_length():
    s = np.arange(10)
    assert list(D.fixed_length(s, 4)) == [0, 3, 6, 9]
    assert list(D.fixed_length(s[:2], 4)) == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        D.fixed_length(s[:0], 3)


def test_recordings_roundtrip(tmp_path):
    frames = list(gesture_sequence("swipe_left", 12, np.random.default_rng(0)))
    p = save_clip(tmp_path, "swipe_left", frames, "Left", 30.0)
    assert p.parent.name == "swipe_left"
    seqs, labels, hands = D.load_recordings(tmp_path)
    assert labels == ["swipe_left"] and hands == ["Left"] and seqs[0].shape == (12, 21, 3)
    with pytest.raises(ValueError):
        save_clip(tmp_path, "moonwalk", frames, "Left", 30.0)
    with pytest.raises(ValueError):
        save_clip(tmp_path, "idle", [], "Left", 30.0)


def test_infer_side_matches_training_assumption():
    assert infer_side(pose("point")) == "Right"


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            chunk = data[int(a) : int(b) + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            super().do_GET()


def test_fetch_hagrid_subset_over_http_ranges(tmp_path, monkeypatch):
    for v in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    src = tmp_path / "srv"
    src.mkdir()
    with zipfile.ZipFile(src / "annotations.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations/test/palm.json", json.dumps({"x": 1}))
        z.writestr("annotations/test/fist.json", json.dumps({"y": 2}) * 1000)
        z.writestr("annotations/train/palm.json", "big")

    def handler(*a, **k):
        return RangeHandler(*a, directory=str(src), **k)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/annotations.zip"
        out = D.fetch_hagrid_subset(tmp_path / "dl", ["palm", "fist"], url=url)
        assert sorted(p.name for p in out) == ["test_fist.json", "test_palm.json"]
        assert json.loads((tmp_path / "dl" / "test_palm.json").read_text()) == {"x": 1}
        f = D.HTTPRangeFile(url)
        assert f.seek(-4, 2) == f.size - 4 and f.tell() == f.size - 4
        assert f.seek(2, 1) == f.size - 2
        f.seek(f.size)
        assert f.read(10) == b""
        # second call is cached: no network needed
        srv.shutdown()
        assert len(D.fetch_hagrid_subset(tmp_path / "dl", ["palm", "fist"], url=url)) == 2
    finally:
        srv.server_close()
