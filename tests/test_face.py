import numpy as np
import pytest

from away.face.commands import BrowRaiseToggle, brow_raise
from away.face.gate import SpeechGate
from away.face.speaker import ActiveSpeakerDetector, pearson
from away.face.speaking import SpeakingScorer, mouth_openness
from away.face.tracking import FaceTracker


def test_mouth_openness():
    assert mouth_openness({}) == 0.0
    assert mouth_openness({"jawOpen": 1.0}) == pytest.approx(1.0 / 2.1)
    full = {
        k: 1.0
        for k in (
            "jawOpen",
            "mouthLowerDownLeft",
            "mouthLowerDownRight",
            "mouthUpperUpLeft",
            "mouthUpperUpRight",
            "mouthFunnel",
            "mouthPucker",
        )
    }
    assert mouth_openness(full) == pytest.approx(1.0)
    assert mouth_openness({"jawOpen": 0.2, "mouthClose": 1.0}) == 0.0
    assert mouth_openness({"jawOpen": 0.8, "mouthClose": 0.2}) == pytest.approx((0.8 - 0.1) / 2.1)


def test_speaking_scorer_movement_beats_static_open_mouth():
    s = SpeakingScorer(window=10)
    for i in range(10):
        talk = s.update(1, {"jawOpen": 0.6 if i % 2 else 0.1})
        yawn = s.update(2, {"jawOpen": 0.6})
    assert talk > yawn
    assert yawn == pytest.approx(0.5 * 0.6 / 2.1)
    assert 0.0 <= talk <= 1.0


def test_speaking_scorer_first_frame_and_forget():
    s = SpeakingScorer(window=4, var_gain=1.0)
    assert s.update(7, {"jawOpen": 0.42}) == pytest.approx(0.5 * 0.2)
    assert s.update(7, {"jawOpen": 0.0}) == pytest.approx(0.5 * 0.1 + 0.2)
    s.forget(7)
    assert s.update(7, {}) == 0.0
    s.forget(99)
    with pytest.raises(ValueError):
        SpeakingScorer(window=1)


def test_speaking_scorer_window_is_rolling():
    s = SpeakingScorer(window=2, var_gain=0.0)
    s.update(1, {"jawOpen": 2.1 / 2.1})
    s.update(1, {"jawOpen": 0.0})
    assert s.update(1, {"jawOpen": 0.0}) == 0.0  # the old open frame fell out


def test_tracker_ids_stable_and_new_faces():
    tr = FaceTracker(max_dist=0.1)
    a = tr.update([(0.3, 0.5, 0.2), (0.7, 0.5, 0.2)])
    assert a == [0, 1]
    b = tr.update([(0.71, 0.5, 0.2), (0.31, 0.5, 0.2)])
    assert b == [1, 0]
    c = tr.update([(0.5, 0.9, 0.1)])
    assert c == [2]


def test_tracker_expiry_and_reacquire():
    tr = FaceTracker(max_dist=0.1, max_missing=2)
    tr.update([(0.3, 0.5, 0.2)])
    tr.update([])
    tr.update([])
    assert 0 in tr.tracks
    assert tr.update([(0.3, 0.5, 0.2)]) == [0]
    for _ in range(3):
        tr.update([])
    assert 0 not in tr.tracks
    assert tr.update([(0.3, 0.5, 0.2)]) == [1]


def test_tracker_one_track_per_box():
    tr = FaceTracker(max_dist=0.5)
    tr.update([(0.5, 0.5, 0.2)])
    ids = tr.update([(0.5, 0.5, 0.2), (0.52, 0.5, 0.2)])
    assert ids[0] == 0 and ids[1] == 1


def test_tracker_distance_boundary():
    tr = FaceTracker(max_dist=0.1)
    tr.update([(0.0, 0.0, 0.1)])
    assert tr.update([(0.1, 0.0, 0.1)]) == [0]
    assert tr.update([(0.25, 0.0, 0.1)]) == [1]


def test_pearson():
    a = np.array([1.0, 2, 3, 4])
    assert pearson(a, a * 2 + 1) == pytest.approx(1.0)
    assert pearson(a, -a) == pytest.approx(-1.0)
    assert pearson(a, np.ones(4)) == 0.0
    assert pearson(a[:2], a[:2]) == 0.0
    assert pearson(a, a[:3]) == 0.0
    assert pearson(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == pytest.approx(1.0)


def two_face_session(det, n=40, speaker=1, seed=0):
    rng = np.random.default_rng(seed)
    active = None
    for i in range(n):
        voice = 0.5 + 0.5 * np.sin(i / 2.0)
        mouth_a = voice + rng.normal(0, 0.05) if speaker == 1 else rng.uniform(0, 1)
        mouth_b = voice + rng.normal(0, 0.05) if speaker == 2 else rng.uniform(0, 1)
        active = det.update({1: mouth_a, 2: mouth_b}, voice)
    return active


@pytest.mark.parametrize("speaker", [1, 2])
def test_active_speaker_correlation(speaker):
    det = ActiveSpeakerDetector(window=30, min_correlation=0.5)
    assert two_face_session(det, speaker=speaker) == speaker
    corr = det.correlations()
    assert corr[speaker] > 0.9


def test_active_speaker_none_when_uncorrelated():
    det = ActiveSpeakerDetector(window=30, min_correlation=0.5)
    rng = np.random.default_rng(3)
    for _ in range(30):
        res = det.update({1: rng.uniform(), 2: rng.uniform()}, rng.uniform())
    assert res is None
    assert ActiveSpeakerDetector().active() is None


def test_active_speaker_single_face_needs_mouth_activity():
    det = ActiveSpeakerDetector(speaking_threshold=0.3)
    assert det.update({5: 0.1}, 0.0) is None
    assert det.update({5: 0.3}, 0.0) == 5
    for _ in range(5):
        det.update({5: 0.0}, 0.0)
    assert det.active() is None


def test_active_speaker_drops_departed_faces_and_aligns_new():
    det = ActiveSpeakerDetector(window=10)
    for _ in range(5):
        det.update({1: 0.5}, 0.1)
    det.update({2: 0.5}, 0.1)
    assert set(det._faces) == {2}
    assert len(det._faces[2]) == 6  # back-filled to line up with audio


class FixedModel:
    def scores(self, face_series, audio_energy):
        return {fid: (1.0 if fid == 2 else 0.0) for fid in face_series}


def test_pluggable_asd_model():
    det = ActiveSpeakerDetector(model=FixedModel(), min_correlation=0.5)
    det.update({1: 0.9, 2: 0.0}, 1.0)
    assert det.update({1: 0.9, 2: 0.0}, 1.0) == 2


def test_gate_enroll_and_allow():
    g = SpeechGate(min_fraction=0.5)
    assert g.enroll({}) is None
    assert g.enroll({3: 0.1, 4: 0.3}) == 4
    for i in range(10):
        g.observe(i, 4 if i < 6 else None, 2)
    assert g.allow(0, 9) is True  # 6/10 >= 0.5
    assert g.allow(6, 9) is False  # nobody detected speaking
    g.observe(20, 3, 2)
    g.observe(21, 3, 2)
    g.observe(22, 4, 2)
    assert g.allow(20, 22) is False  # someone else spoke more


def test_gate_fraction_boundary():
    g = SpeechGate(min_fraction=0.5)
    g.enroll({1: 1.0})
    g.observe(0, 1, 1)
    g.observe(1, None, 1)
    assert g.allow(0, 1) is True


def test_gate_tie_with_other_speaker_allows():
    g = SpeechGate(min_fraction=0.5)
    g.enroll({1: 1.0})
    g.observe(0, 1, 2)
    g.observe(1, 2, 2)
    assert g.allow(0, 1) is True


def test_gate_without_face_and_not_enrolled():
    g = SpeechGate(allow_without_face=False)
    assert g.allow(0, 1) is False
    g.observe(0.5, None, 0)
    assert g.allow(0, 1) is False
    assert SpeechGate(allow_without_face=True).allow(0, 1) is True
    g2 = SpeechGate()
    g2.observe(0.5, 7, 1)
    assert g2.allow(0, 1) is True  # not enrolled => no-op


class Deny:
    def is_user(self, face_id):
        return False


def test_gate_identity_check():
    g = SpeechGate()
    g.enroll({1: 1.0})
    g.observe(0.5, 1, 1)
    g.identity = Deny()
    assert g.allow(0, 1) is False


def test_brow_raise_value():
    assert brow_raise({"browInnerUp": 0.2, "browOuterUpLeft": 0.8, "browOuterUpRight": 0.6}) == pytest.approx(0.7)
    assert brow_raise({"browInnerUp": 0.9}) == 0.9
    assert brow_raise({}) == 0.0


def test_brow_toggle_hold_rearm_cooldown():
    b = BrowRaiseToggle(threshold=0.5, hold=0.5, cooldown=1.0)
    up, down, mid = {"browInnerUp": 0.8}, {"browInnerUp": 0.0}, {"browInnerUp": 0.4}
    assert b.update(0.0, up) is False
    assert b.update(0.49, up) is False
    assert b.update(0.5, up) is True
    assert b.update(0.6, up) is False  # needs re-arming
    assert b.update(0.7, mid) is False  # above release (0.3): still not re-armed
    assert b.update(0.8, up) is False
    b.update(0.9, down)
    b.update(1.0, up)
    assert b.update(1.4, up) is False  # hold not reached again (restarted at 1.0)
    assert b.update(1.5, up) is True


def test_brow_toggle_cooldown_blocks():
    b = BrowRaiseToggle(threshold=0.5, hold=0.0, cooldown=2.0)
    assert b.update(0.0, {"browInnerUp": 1.0}) is True
    b.update(0.1, {"browInnerUp": 0.0})
    assert b.update(0.2, {"browInnerUp": 1.0}) is False
    b.update(0.3, {"browInnerUp": 0.0})
    assert b.update(2.0, {"browInnerUp": 1.0}) is True


def test_brow_dip_between_release_and_threshold_restarts_hold():
    b = BrowRaiseToggle(threshold=0.5, hold=0.5)
    b.update(0.0, {"browInnerUp": 0.9})
    b.update(0.3, {"browInnerUp": 0.4})
    assert b.update(0.6, {"browInnerUp": 0.9}) is False
    assert b.update(1.1, {"browInnerUp": 0.9}) is True
