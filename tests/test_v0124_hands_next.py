"""v0.1.24 model refresh: the experimental RTMPose hand landmarker (decode + geometry + a full frame through fake
ONNX sessions) and the MediaPipe-vs-RTMPose A/B scoring (away.hands.ab)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from away.hands import ab, rtmpose

# a hand in crop pixels (256x256): wrist at the bottom, fingers up; thumb on the image's left
HAND_CROP = np.array(
    [[128, 220]]
    + [[90 - 6 * i, 190 - 20 * i] for i in range(4)]  # thumb 1-4
    + [[110, 150 - 22 * i] for i in range(4)]  # index 5-8
    + [[128, 145 - 24 * i] for i in range(4)]  # middle 9-12
    + [[146, 150 - 22 * i] for i in range(4)]  # ring 13-16
    + [[164, 160 - 18 * i] for i in range(4)],  # little 17-20
    dtype=np.float64,
)


def _simcc(pts_crop, peak=0.9, bins=512, split=2.0):
    k = len(pts_crop)
    sx, sy = np.full((k, bins), 0.01, np.float32), np.full((k, bins), 0.01, np.float32)
    for i, (x, y) in enumerate(pts_crop):
        sx[i, round(float(x) * split)] = peak
        sy[i, round(float(y) * split)] = peak
    return sx, sy


# ============================================================================================ decode + geometry
def test_decode_simcc_returns_crop_pixels_and_the_weaker_peak():
    sx, sy = _simcc(HAND_CROP)
    sy[3, int(HAND_CROP[3, 1] * 2)] = 0.4
    pts, scores = rtmpose.decode_simcc(sx, sy)
    assert pts.shape == (21, 2) and np.allclose(pts, HAND_CROP)
    assert scores[3] == pytest.approx(0.4) and scores[0] == pytest.approx(0.9)


def test_decode_simcc_half_pixel_resolution():
    sx = np.zeros((1, 512), np.float32)
    sy = np.zeros((1, 512), np.float32)
    sx[0, 101], sy[0, 33] = 1.0, 1.0
    pts, _ = rtmpose.decode_simcc(sx, sy)
    assert pts.tolist() == [[50.5, 16.5]]


def test_box_to_center_scale_is_square_and_padded():
    c, side = rtmpose.box_to_center_scale(np.array([100, 50, 200, 250, 0.9]))
    assert c.tolist() == [150, 150] and side == pytest.approx(250)  # max(100, 200) x 1.25
    assert rtmpose.box_to_center_scale(np.array([5, 5, 5, 5]))[1] == 1.0


def test_crop_matrix_and_to_image_are_inverse():
    center, side = np.array([320.0, 200.0]), 180.0
    m = rtmpose.crop_matrix(center, side)
    img_pts = np.array([[320.0, 200.0], [230.0, 110.0], [410.0, 290.0], [300.0, 250.0]])
    crop = img_pts @ m[:, :2].T + m[:, 2]
    assert crop[0].tolist() == pytest.approx([128, 128])  # the centre lands mid-crop
    assert crop[1].tolist() == pytest.approx([0, 0]) and crop[2].tolist() == pytest.approx([256, 256])
    assert np.allclose(rtmpose.to_image(crop, center, side), img_pts)


def test_letterbox_keeps_aspect_pads_114_and_flips_to_bgr():
    pytest.importorskip("cv2")
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[..., 0] = 200  # red
    img, r = rtmpose.letterbox(rgb)
    assert img.shape == (320, 320, 3) and img.dtype == np.float32 and r == pytest.approx(0.5)
    assert img[0, 0].tolist() == [0, 0, 200]  # BGR: red is last
    assert img[-1, -1].tolist() == [114, 114, 114]  # padded bottom
    assert img[239, 0].tolist() == [0, 0, 200] and img[240, 0].tolist() == [114, 114, 114]


def test_box_from_points_is_clipped_to_the_image():
    pts = np.array([[-10.0, 5.0], [700.0, 300.0]])
    assert rtmpose.box_from_points(pts, 640, 480).tolist() == [0, 5, 640, 300]


def test_guess_label_from_thumb_and_little_finger():
    assert rtmpose.guess_label(HAND_CROP) == "Right"
    mirrored = HAND_CROP.copy()
    mirrored[:, 0] = 256 - mirrored[:, 0]
    assert rtmpose.guess_label(mirrored) == "Left"


def test_find_models_and_factory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not downloaded yet"):
        rtmpose.find_models(tmp_path)
    for sub in ("rtmpose-m_hand5/end2end.onnx", "rtmdet-nano/end2end.onnx"):
        p = tmp_path / "bundle" / sub
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x")
    pose, det = rtmpose.find_models(tmp_path)
    assert pose.parent.name.startswith("rtmpose") and det.parent.name.startswith("rtmdet")
    assert callable(rtmpose.tracker_factory(tmp_path))
    assert "CPUExecutionProvider" in rtmpose.providers()


# ============================================================================================ full frames (fake ONNX)
class FakeDet:
    def __init__(self, boxes_320):
        self.boxes = np.array([boxes_320], np.float32)  # (1, N, 5) in letterboxed 320 px
        self.calls = 0
        self.inputs = []

    def run(self, _o, feeds):
        self.calls += 1
        self.inputs.append(feeds["input"].shape)
        return self.boxes, np.zeros((1, self.boxes.shape[1]), np.int64)


class FakePose:
    def __init__(self, score=0.9, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        self.sx, self.sy = _simcc(HAND_CROP, peak=score)
        self.calls = 0
        self.inputs = []
        self._prov = list(providers)

    def run(self, _o, feeds):
        self.calls += 1
        self.inputs.append(feeds["input"].shape)
        return self.sx[None], self.sy[None]

    def get_providers(self):
        return self._prov


def test_rtmpose_frame_gives_a_mediapipe_shaped_result():
    pytest.importorskip("cv2")
    # one hand box in the 640x480 image (letterbox scale 0.5): x 200..360, y 100..300
    det = FakeDet([[100, 50, 180, 150, 0.95], [0, 0, 10, 10, 0.1]])
    pose = FakePose()
    lm = rtmpose.RTMPoseLandmarker(det_session=det, pose_session=pose, num_hands=2, det_every=5)
    assert lm.device == "gpu"
    frame = np.zeros((480, 640, 3), np.uint8)
    res = lm.detect_for_video(frame, 0)
    assert det.inputs == [(1, 3, 320, 320)] and pose.inputs == [(1, 3, 256, 256)]
    assert len(res.hand_landmarks) == 1  # the low-score box is dropped
    pts = res.hand_landmarks[0]
    assert len(pts) == 21 and all(p.z == 0.0 for p in pts)
    center, side = rtmpose.box_to_center_scale(np.array([200, 100, 360, 300]))
    expect = rtmpose.to_image(HAND_CROP, center, side)
    got = np.array([[p.x * 640, p.y * 480] for p in pts])
    assert np.allclose(got, expect, atol=1e-6)
    cat = res.handedness[0][0]
    assert cat.category_name == "Right" and cat.score == pytest.approx(0.9)


def test_rtmpose_runs_the_detector_every_n_frames_and_follows_the_hand_in_between():
    pytest.importorskip("cv2")
    det, pose = FakeDet([[100, 50, 180, 150, 0.95]]), FakePose()
    lm = rtmpose.RTMPoseLandmarker(det_session=det, pose_session=pose, det_every=5)
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(10):
        lm.detect_for_video(frame, i * 33)
    assert det.calls == 3  # frame 1 (no box yet), frame 5, frame 10
    assert pose.calls == 10


def test_rtmpose_drops_weak_hands_and_redetects():
    pytest.importorskip("cv2")
    det = FakeDet([[100, 50, 180, 150, 0.95]])
    lm = rtmpose.RTMPoseLandmarker(
        det_session=det, pose_session=FakePose(score=0.1, providers=("CPUExecutionProvider",))
    )
    assert lm.device == "cpu"
    frame = np.zeros((480, 640, 3), np.uint8)
    assert lm.detect_for_video(frame, 0).hand_landmarks == []
    lm.detect_for_video(frame, 33)
    assert det.calls == 2  # no box kept -> the detector runs again on the next frame


def test_hand_kwargs_rolls_back_to_mediapipe_when_rtmpose_is_missing(monkeypatch, tmp_path):
    from away.app import model_guard, next_wiring
    from away.config import AppConfig

    cfg = AppConfig()
    assert next_wiring.hand_kwargs(cfg) == {}
    model_guard.remember("hands", {"models_next.hands_landmarker": "mediapipe"})
    cfg.models_next.hands_landmarker = "rtmpose"
    notes, saved = [], []
    assert next_wiring.hand_kwargs(cfg, lambda t, m: notes.append(m), lambda: saved.append(1)) == {}
    assert cfg.models_next.hands_landmarker == "mediapipe" and saved == [1]
    assert "RTMPose hands unavailable" in notes[0]


# ============================================================================================ A/B
def _hand(offset=(0.0, 0.0), pinch=False, noise=0.0, rng=None):
    pts = HAND_CROP / 256.0 + np.array(offset)
    if pinch:
        pts[4] = pts[8]  # thumb tip on index tip
    if noise:
        pts = pts + (rng or np.random.default_rng(0)).normal(0, noise, pts.shape)
    return pts


def test_palm_size_and_pinch_ratio():
    h = _hand()
    assert ab.palm_size(h) == pytest.approx(np.linalg.norm(h[0] - h[9]))
    assert ab.pinch_ratio(_hand(pinch=True)) == pytest.approx(0.0)
    assert ab.pinch_ratio(h) > ab.PINCH_ON


def test_score_sequence_jitter_found_and_flicker():
    rng = np.random.default_rng(1)
    steady = ab.score_sequence("steady", [_hand(noise=0.0005, rng=rng) for _ in range(50)], ms=5.0)
    shaky = ab.score_sequence("shaky", [_hand(noise=0.004, rng=rng) for _ in range(50)])
    assert steady.found == 1.0 and steady.frames == 50 and steady.ms_per_frame == 5.0
    assert steady.jitter < shaky.jitter
    assert steady.pinch_flicker == 0.0
    flick = ab.score_sequence("flick", [_hand(pinch=i % 2 == 0) for i in range(20)])
    assert flick.pinch_flicker == pytest.approx(1.0)  # flips on every resting frame
    gaps = ab.score_sequence("gaps", [_hand() if i % 2 else None for i in range(10)])
    assert gaps.found == 0.5 and gaps.jitter is None and gaps.pinch_flicker is None  # never two in a row
    moving = ab.score_sequence("moving", [_hand(offset=(0.05 * i, 0)) for i in range(10)])
    assert moving.jitter is None  # the hand never rested
    row = steady.row()
    assert set(row) == {"landmarker", "frames", "found", "jitter", "pinch_flicker", "ms_per_frame"}


def _score(name, found=1.0, jitter=0.01, flicker=0.0, ms=10.0):
    return ab.Score(name, 100, found, jitter, flicker, ms)


def test_compare_recommends_rtmpose_only_when_no_worse():
    mp = _score("MediaPipe")
    ok = ab.compare(mp, _score("RTMPose", jitter=0.008, ms=9.0))
    assert ok["switch"] is True and ok["recommend"] == "RTMPose" and "as steady and as fast" in ok["why"]
    assert len(ok["scores"]) == 2
    for cand, word in (
        (_score("RTMPose", found=0.9), "less often"),
        (_score("RTMPose", jitter=0.02), "shakier"),
        (_score("RTMPose", jitter=None), "shakier"),
        (_score("RTMPose", flicker=0.05), "flickered"),
        (_score("RTMPose", ms=20.0), "slower"),
    ):
        res = ab.compare(mp, cand)
        assert res["switch"] is False and res["recommend"] == "MediaPipe" and word in res["why"]
    # within tolerance counts as no worse
    assert ab.compare(mp, _score("RTMPose", jitter=0.0105, ms=10.5))["switch"] is True


def test_run_times_a_landmarker_over_frames():
    class LM:
        def __init__(self):
            self.ms = []

        def detect_for_video(self, frame, ms):
            self.ms.append(ms)
            if frame is None:
                return SimpleNamespace(hand_landmarks=[])
            return SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=x, y=y) for x, y in _hand()]])

    ticks = iter(np.arange(0, 100, 0.002))
    lm = LM()
    s = ab.run("fake", lm, [1, None, 1, 1], clock=lambda: float(next(ticks)))
    assert lm.ms == [0, 33, 66, 99]
    assert s.found == 0.75 and s.ms_per_frame == pytest.approx(2.0)
