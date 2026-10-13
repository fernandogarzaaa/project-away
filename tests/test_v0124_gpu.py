"""v0.1.24 tracking on the graphics card: the choice (auto | gpu | cpu, every platform/provider combination), the
CPU fallback, the pinned ONNX model download, and the numpy re-implementation of MediaPipe's detector/landmark
glue (anchors, decoding, weighted NMS, rotated crops, projection) with fake ONNX sessions. With AWAY_NET_TESTS=1
and onnxruntime installed the real models are downloaded, checked and run (DirectML on the Windows CI job)."""

from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.accel import compute
from away.accel import models as gm
from away.accel import mp_pipeline as mp

cv2 = pytest.importorskip("cv2")


# ------------------------------------------------------------------------------------------------ choosing
@pytest.mark.parametrize(
    ("setting", "plat", "providers", "ready", "backend", "using"),
    [
        ("auto", "win32", ["CPUExecutionProvider"], False, "mediapipe-cpu", "processor"),
        ("auto", "win32", ["DmlExecutionProvider", "CPUExecutionProvider"], False, "mediapipe-cpu", "processor"),
        ("auto", "win32", ["DmlExecutionProvider", "CPUExecutionProvider"], True, "onnx-gpu", "graphics card"),
        ("gpu", "win32", ["DmlExecutionProvider", "CPUExecutionProvider"], False, "onnx-gpu", "graphics card"),
        ("gpu", "win32", ["CPUExecutionProvider"], False, "mediapipe-cpu", "processor"),
        ("cpu", "win32", ["CUDAExecutionProvider"], True, "mediapipe-cpu", "processor"),
        ("auto", "linux", ["CPUExecutionProvider"], False, "mediapipe-cpu", "processor"),
        ("gpu", "linux", ["CPUExecutionProvider"], False, "mediapipe-gpu", "graphics card"),
        ("auto", "linux", ["CUDAExecutionProvider", "CPUExecutionProvider"], True, "onnx-gpu", "graphics card"),
    ],
)
def test_choice(setting: str, plat: str, providers: list[str], ready: bool, backend: str, using: str) -> None:
    c = compute.choose(setting, platform=plat, providers=providers, models_ready=ready)
    assert (c.backend, c.using) == (backend, using), c.reason
    assert c.reason


def test_cuda_is_preferred_over_directml() -> None:
    assert compute.gpu_provider(["DmlExecutionProvider", "CUDAExecutionProvider"]) == "CUDAExecutionProvider"


def test_setting_is_validated_and_shown() -> None:
    cfg = C.AppConfig()
    assert cfg.hands.compute == "auto"
    cfg.hands.compute = "tpu"
    with pytest.raises(ValueError, match=r"hands\.compute"):
        C.validate(cfg)
    from away.accel.settings_ui import LABELS, label_to_value, status_line

    assert label_to_value(LABELS["gpu"]) == "gpu" and label_to_value("?") == "auto"
    assert "one-time" in status_line({"using": "processor", "provider": "DirectML", "models_ready": False})
    assert "graphics card" in status_line({"using": "graphics card", "provider": "DirectML"})


def test_auto_never_downloads_and_falls_back_to_the_processor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = C.AppConfig()
    called: list[str] = []
    monkeypatch.setattr(gm, "download", lambda *a, **k: called.append("dl"))
    assert compute.hand_factory(cfg, tmp_path / "hand.task", tmp_path, providers=["DmlExecutionProvider"]) is None
    assert called == []
    st = compute.status(cfg, tmp_path, providers=["DmlExecutionProvider"])
    assert st["using"] == "processor" and st["provider"] == "DirectML" and not st["models_ready"]


def test_gpu_that_fails_to_start_falls_back_with_a_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = C.AppConfig()
    cfg.hands.compute = "gpu"

    def broken(*_a: Any, **_k: Any) -> None:
        raise gm.ModelDownloadError("no internet")

    monkeypatch.setattr(gm, "download", broken)
    notes: list[str] = []
    assert (
        compute.hand_factory(
            cfg, tmp_path / "h.task", tmp_path, on_note=notes.append, providers=["DmlExecutionProvider"]
        )
        is None
    )
    assert notes and "processor" in notes[0]
    st = compute.status(cfg, tmp_path, providers=["DmlExecutionProvider"])
    assert st["using"] == "processor" and "could not start" in st["reason"]


def test_face_stays_on_mediapipe_when_the_gpu_is_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import away.face.landmarker as fl

    made: list[Any] = []

    class FakeFL:
        def __init__(self, path: Any, n: int) -> None:
            made.append((path, n))

    monkeypatch.setattr(fl, "FaceLandmarker", FakeFL)
    got = compute.face_landmarker(
        C.AppConfig(), tmp_path / "face.task", tmp_path, 4, providers=["CPUExecutionProvider"]
    )
    assert isinstance(got, FakeFL) and made == [(tmp_path / "face.task", 4)]


# ------------------------------------------------------------------------------------------------ download
def test_download_checks_the_pinned_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good, bad = b"good model bytes", b"tampered!"
    m = gm.OnnxModel("palm_detector", "onnx/hand/palm.onnx", hashlib.sha256(good).hexdigest(), len(good))
    monkeypatch.setitem(gm.SETS, "hands", (m,))
    served = {"body": bad}

    class Resp(io.BytesIO):
        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

    def opener(req: Any, timeout: float = 0) -> Resp:
        assert req.full_url.startswith("https://huggingface.co/") and gm.REVISION in req.full_url
        return Resp(served["body"])

    with pytest.raises(gm.ModelDownloadError, match="checksum"):
        gm.download("hands", tmp_path, opener=opener)
    assert not gm.ready("hands", tmp_path) and not list(gm.folder(tmp_path).glob("*.part"))
    served["body"] = good
    got = gm.download("hands", tmp_path, opener=opener)
    assert gm.ready("hands", tmp_path) and got["palm_detector"].read_bytes() == good
    assert gm.remove(tmp_path) and not gm.ready("hands", tmp_path)


def test_manifest_is_pinned_and_permissive() -> None:
    assert len(gm.REVISION) == 40 and gm.LICENSE == "Apache-2.0"
    total = sum(m.size for ms in gm.SETS.values() for m in ms)
    assert 20e6 < total < 26e6  # "about 23 MB"
    for ms in gm.SETS.values():
        for m in ms:
            assert len(m.sha256) == 64 and m.url.startswith("https://huggingface.co/")


# ------------------------------------------------------------------------------------------------ geometry
def test_anchor_counts_match_the_models() -> None:
    assert mp.ssd_anchors(192, (8, 16, 16, 16)).shape == (2016, 2)  # palm detector output
    assert mp.ssd_anchors(128, (8, 16, 16, 16)).shape == (896, 2)  # BlazeFace short range
    a = mp.ssd_anchors(192, (8, 16, 16, 16))
    assert np.allclose(a[0], [0.5 / 24, 0.5 / 24]) and np.allclose(a[-1], [11.5 / 12, 11.5 / 12])


def test_decode_and_weighted_nms() -> None:
    anchors = mp.ssd_anchors(192, (8, 16, 16, 16))
    boxes = np.zeros((2016, 18), np.float32)
    scores = np.full((2016, 1), -10.0, np.float32)
    for i, dx in ((500, 0.0), (502, 2.0)):  # two overlapping detections of one palm
        boxes[i, :4] = [dx, 0, 40, 40]
        boxes[i, 4:6] = [0, 10]
        scores[i] = 3.0
    dets = mp.decode(boxes, scores, anchors, 192, 7)
    assert len(dets) == 2
    out = mp.weighted_nms(dets, 0.3, 2)
    assert len(out) == 1 and out[0].box[2] == pytest.approx(40 / 192, rel=1e-4)


def test_letterbox_roundtrip() -> None:
    img = np.zeros((480, 640, 3), np.uint8)
    sq, pad = mp.letterbox(img, 192)
    assert sq.shape == (192, 192, 3) and pad[0] == 0 and pad[1] > 0
    d = mp.Detection(1.0, (0.25, pad[1] + 0.1, 0.1, 0.1), np.array([[0.5, 0.5]], np.float32))
    u = mp.unletterbox(d, pad)
    assert u.keypoints[0, 0] == pytest.approx(0.5) and u.box[1] == pytest.approx(0.1 / (1 - 2 * pad[1]), rel=1e-4)


@pytest.mark.parametrize("rot", [0.0, 0.4, -1.2, math.pi / 2])
def test_crop_and_project_are_inverse(rot: float) -> None:
    w, h, size = 640, 480, 224
    r = mp.Rect(0.45, 0.55, 0.3, 0.4, rot)
    m = mp.crop_matrix(r, w, h, size)
    pts_img = np.array([[300.0, 250.0], [280.0, 300.0], [320.0, 240.0]])
    in_crop = (m @ np.c_[pts_img, np.ones(3)].T).T  # where cv2.warpAffine puts these pixels
    back = mp.project(np.c_[in_crop, np.zeros(3)], r, size, w, h)
    assert np.allclose(back[:, 0] * w, pts_img[:, 0], atol=1e-3) and np.allclose(
        back[:, 1] * h, pts_img[:, 1], atol=1e-3
    )


def test_transform_squares_and_shifts_in_the_rotated_frame() -> None:
    r = mp.Rect(0.5, 0.5, 0.1, 0.2, math.pi / 2)
    t = mp.transform(r, 1000, 1000, 2.0, 2.0, 0.0, -0.5)
    assert t.w == pytest.approx(0.4) and t.h == pytest.approx(0.4)
    assert t.cx == pytest.approx(0.6) and t.cy == pytest.approx(0.5)  # "up" in the hand's frame is +x here


def test_hand_rect_from_landmarks_points_along_the_hand() -> None:
    lm = np.zeros((21, 3))
    lm[:, 0] = 0.5
    lm[:, 1] = np.linspace(0.8, 0.4, 21)  # an upright hand: wrist at the bottom
    r = mp.hand_rect_from_landmarks(lm, 640, 480)
    assert abs(r.rotation) < 1e-6 and 0.4 < r.cy < 0.8


# ------------------------------------------------------------------------------------- the ONNX landmarker
class FakeSession:
    def __init__(self, fn: Any, provider: str = "CPUExecutionProvider") -> None:
        self.fn, self.calls, self.provider = fn, 0, provider

    def get_inputs(self) -> list[Any]:
        class Input:
            name = "input_1"

        return [Input()]

    def get_providers(self) -> list[str]:
        return [self.provider, "CPUExecutionProvider"]

    def run(self, _outs: Any, feed: dict[str, Any]) -> list[Any]:
        self.calls += 1
        return self.fn(next(iter(feed.values())))


def palm_outputs(_x: Any) -> list[Any]:
    boxes = np.zeros((1, 2016, 18), np.float32)
    scores = np.full((1, 2016, 1), -10.0, np.float32)
    boxes[0, 600, :4] = [0, 0, 40, 40]
    boxes[0, 600, 4:6] = [0, 12]  # wrist below the palm centre
    boxes[0, 600, 8:10] = [0, -12]  # middle finger above
    scores[0, 600] = 4.0
    return [boxes, scores]


def hand_outputs(_x: Any) -> list[Any]:
    pts = np.zeros((21, 3), np.float32)
    pts[:, 0] = 112.0
    pts[:, 1] = np.linspace(180, 40, 21)
    return [
        pts.reshape(1, 63),
        np.array([[0.98]], np.float32),
        np.array([[0.9]], np.float32),
        np.zeros((1, 63), np.float32),
    ]


def test_onnx_hand_landmarker_detects_then_tracks() -> None:
    from away.accel.onnx_hands import OnnxHandLandmarker

    palm, lmk = FakeSession(palm_outputs), FakeSession(hand_outputs, "DmlExecutionProvider")
    tr = OnnxHandLandmarker(palm, lmk, num_hands=1)
    img = np.zeros((480, 640, 3), np.uint8)
    r1 = tr.detect_for_video(img, 0)
    assert len(r1.hand_landmarks) == 1 and r1.handedness[0][0].category_name == "Right"
    assert 0 < r1.hand_landmarks[0][0].y < 1 and r1.hand_landmarks[0][0].y > r1.hand_landmarks[0][20].y
    r2 = tr.detect_for_video(img, 33)
    assert len(r2.hand_landmarks) == 1 and palm.calls == 1  # followed from the landmarks, no new palm search
    assert "DmlExecutionProvider" in tr.providers


def test_onnx_hand_landmarker_plugs_into_hand_tracker() -> None:
    from away.accel.onnx_hands import OnnxHandLandmarker
    from away.hands.tracker import HandTracker

    tr = HandTracker(
        None,
        1,
        factory=lambda n: OnnxHandLandmarker(FakeSession(palm_outputs), FakeSession(hand_outputs), n),
        image_fn=lambda x: x,
    )
    fr = tr.process(np.zeros((480, 640, 3), np.uint8), 0.0)
    assert fr.landmarks is not None and fr.landmarks.shape == (21, 3) and fr.n_hands == 1


def test_lost_hand_goes_back_to_palm_search() -> None:
    from away.accel.onnx_hands import OnnxHandLandmarker

    state = {"present": 0.98}

    def lmk(_x: Any) -> list[Any]:
        out = hand_outputs(_x)
        out[1] = np.array([[state["present"]]], np.float32)
        return out

    palm = FakeSession(palm_outputs)
    tr = OnnxHandLandmarker(palm, FakeSession(lmk), 1)
    img = np.zeros((240, 320, 3), np.uint8)
    tr.detect_for_video(img, 0)
    state["present"] = 0.1
    assert tr.detect_for_video(img, 33).hand_landmarks == [] and tr.tracked == []
    tr.detect_for_video(img, 66)
    assert palm.calls == 2  # the next frame searches again (like MediaPipe)


def test_onnx_face_landmarker_returns_face_obs() -> None:
    from away.accel.onnx_face import BLENDSHAPE_NAMES, BLENDSHAPE_POINTS, OnnxFaceLandmarker

    assert len(BLENDSHAPE_POINTS) == 146 and len(BLENDSHAPE_NAMES) == 52

    def det(_x: Any) -> list[Any]:
        boxes = np.zeros((1, 896, 16), np.float32)
        scores = np.full((1, 896, 1), -10.0, np.float32)
        boxes[0, 300, :4] = [0, 0, 50, 50]
        boxes[0, 300, 4:8] = [-10, -5, 10, -5]  # eyes
        scores[0, 300] = 5.0
        return [boxes, scores]

    def mesh(_x: Any) -> list[Any]:
        rng = np.random.default_rng(0)
        pts = np.c_[rng.uniform(60, 196, (478, 2)), np.zeros(478)].astype(np.float32)
        return [pts.reshape(1, 1, 1, 1434), np.array([[[[4.0]]]], np.float32), np.zeros((1, 1), np.float32)]

    seen: list[Any] = []

    def bs(x: Any) -> list[Any]:
        seen.append(x.shape)
        out = np.zeros(52, np.float32)
        out[25] = 0.7  # jawOpen
        return [out]

    f = OnnxFaceLandmarker(FakeSession(det), FakeSession(mesh), FakeSession(bs), 1)
    obs = f.process(np.zeros((480, 640, 3), np.uint8), 0.0)
    assert len(obs) == 1 and obs[0].blendshapes["jawOpen"] == pytest.approx(0.7) and seen == [(1, 146, 2)]
    f.process(np.zeros((480, 640, 3), np.uint8), 0.033)
    assert f.det_runs == 1  # tracked from the mesh


# ------------------------------------------------------------------------------------------- real models
@pytest.mark.skipif(os.environ.get("AWAY_NET_TESTS") != "1", reason="downloads ~23 MB (set AWAY_NET_TESTS=1)")
def test_real_models_download_and_run(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    gm.download("hands", tmp_path)
    gm.download("face", tmp_path)
    assert gm.ready("hands", tmp_path) and gm.ready("face", tmp_path)
    from away.accel.onnx_face import OnnxFaceLandmarker
    from away.accel.onnx_hands import OnnxHandLandmarker

    avail = ort.get_available_providers()
    prov = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider") if p in avail] + ["CPUExecutionProvider"]
    hp, fp = gm.paths("hands", tmp_path), gm.paths("face", tmp_path)
    hands = OnnxHandLandmarker(hp["palm_detector"], hp["hand_landmark"], 2, prov)
    print("providers available:", avail, "hand session uses:", hands.providers)
    if os.environ.get("AWAY_EXPECT_DML") == "1":  # the Windows CI job with onnxruntime-directml
        assert "DmlExecutionProvider" in avail  # the DirectML runtime itself loads
        if "DmlExecutionProvider" not in hands.providers:
            # GitHub's Windows runners have no DirectX 12 graphics card ("No devices detected that match the filter
            # criteria"), so ONNX Runtime quietly runs the session on the processor. The app must notice that and
            # fall back to MediaPipe with a note instead of claiming the graphics card.
            notes: list[str] = []
            cfg = C.AppConfig()
            cfg.hands.compute = "gpu"
            assert compute.hand_factory(cfg, tmp_path / "hand.task", tmp_path, on_note=notes.append) is None
            assert notes, "the fallback to the processor must say why"
            print("DirectML loaded but found no device; fallback note:", notes[0])
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert hands.detect_for_video(blank, 0).hand_landmarks == []
    face = OnnxFaceLandmarker(fp["face_detector"], fp["face_landmark"], fp["face_blendshape"], 1, prov)
    assert face.process(blank, 0.0) == []
    # the landmark models themselves run and give sane shapes on any picture
    out = hands.lmk.run(None, {hands.lmk.get_inputs()[0].name: np.zeros((1, 224, 224, 3), np.float32)})
    assert np.asarray(out[0]).size == 63
    st = compute.status(C.AppConfig(), tmp_path)
    assert st["models_ready"] is True
