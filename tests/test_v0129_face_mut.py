"""Mutation-strengthening tests for face/* (v0.1.29)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from away.face.commands import (
    BrowRaiseToggle,
    MouthOpenTrigger,
    WinkTrigger,
    brow_raise,
    mouth_open,
)
from away.face.gate import SpeechGate
from away.face.speaker import pearson
from away.face.speaking import SpeakingScorer, mouth_openness
from away.face.tracking import FaceTracker


def test_mouth_openness_zero() -> None:
    assert mouth_openness({}) == 0.0


def test_mouth_openness_jaw() -> None:
    # jawOpen=1.0 with weight 1.0, norm ~2.1
    score = mouth_openness({"jawOpen": 1.0})
    assert 0.4 < score < 0.6


def test_mouth_openness_clipped() -> None:
    # All maxed: clipped to 1.0
    bs = {k: 1.0 for k in ["jawOpen", "mouthLowerDownLeft", "mouthLowerDownRight"]}
    assert mouth_openness(bs) <= 1.0


def test_mouth_openness_mouth_close_subtracts() -> None:
    open_score = mouth_openness({"jawOpen": 1.0})
    closed_score = mouth_openness({"jawOpen": 1.0, "mouthClose": 1.0})
    assert closed_score < open_score


def test_mouth_openness_never_negative() -> None:
    assert mouth_openness({"mouthClose": 1.0}) == 0.0


def test_speaking_scorer_window_validation() -> None:
    with pytest.raises(ValueError):
        SpeakingScorer(window=1)


def test_speaking_scorer_silence() -> None:
    scorer = SpeakingScorer()
    score = scorer.update(0, {})
    assert score == 0.0


def test_speaking_scorer_movement_scores_higher() -> None:
    scorer = SpeakingScorer()
    # Static: low score
    for _ in range(5):
        static = scorer.update(0, {"jawOpen": 0.5})
    # Moving: higher score (variability term)
    scorer2 = SpeakingScorer()
    for i in range(5):
        moving = scorer2.update(0, {"jawOpen": 0.1 if i % 2 == 0 else 0.9})
    assert moving > static


def test_speaking_scorer_forget() -> None:
    scorer = SpeakingScorer()
    scorer.update(0, {"jawOpen": 0.5})
    scorer.forget(0)
    # After forget, history is empty
    assert 0 not in scorer._hist


def test_speaking_scorer_multiple_faces() -> None:
    scorer = SpeakingScorer()
    s0 = scorer.update(0, {"jawOpen": 0.1})
    s1 = scorer.update(1, {"jawOpen": 0.9})
    assert s1 > s0


def test_face_tracker_assigns_new_ids() -> None:
    tracker = FaceTracker()
    ids = tracker.update([(0.5, 0.5, 0.1)])
    assert ids == [0]
    ids = tracker.update([(0.6, 0.6, 0.1)])
    # Close: same ID (within max_dist)
    assert ids == [0]


def test_face_tracker_new_face_gets_new_id() -> None:
    tracker = FaceTracker()
    tracker.update([(0.1, 0.1, 0.1)])
    ids = tracker.update([(0.9, 0.9, 0.1)])
    # Far: new ID
    assert ids == [1]


def test_face_tracker_missing_increments() -> None:
    tracker = FaceTracker(max_missing=2)
    tracker.update([(0.5, 0.5, 0.1)])
    tracker.update([])  # missing 1
    assert 0 in tracker.tracks
    assert tracker.tracks[0].missing == 1
    tracker.update([])  # missing 2
    tracker.update([])  # missing 3 > max_missing: removed
    assert 0 not in tracker.tracks


def test_face_tracker_greedy_nearest() -> None:
    tracker = FaceTracker(max_dist=1.0)
    tracker.update([(0.1, 0.1, 0.1), (0.9, 0.9, 0.1)])
    # Two tracks: 0 and 1
    ids = tracker.update([(0.11, 0.11, 0.1), (0.89, 0.89, 0.1)])
    assert ids == [0, 1]  # nearest matching


def test_brow_raise_max() -> None:
    assert brow_raise({"browInnerUp": 0.8}) == 0.8
    assert brow_raise({"browOuterUpLeft": 0.6, "browOuterUpRight": 0.4}) == 0.5
    assert brow_raise({}) == 0.0


def test_brow_toggle_fires_after_hold() -> None:
    tog = BrowRaiseToggle(threshold=0.5, hold=0.5, cooldown=0.0)
    bs = {"browInnerUp": 0.9}
    assert tog.update(0.0, bs) is False
    assert tog.update(0.6, bs) is True  # held past 0.5s


def test_brow_toggle_no_fire_below_threshold() -> None:
    tog = BrowRaiseToggle(threshold=0.5, hold=0.5, cooldown=0.0)
    bs = {"browInnerUp": 0.4}
    assert tog.update(0.0, bs) is False
    assert tog.update(1.0, bs) is False


def test_brow_toggle_rearms_on_release() -> None:
    tog = BrowRaiseToggle(threshold=0.5, hold=0.1, cooldown=0.0)
    bs = {"browInnerUp": 0.9}
    assert tog.update(0.0, bs) is False
    assert tog.update(0.2, bs) is True  # first fire
    assert tog.update(0.3, bs) is False  # cooldown/disarm
    tog.update(0.4, {"browInnerUp": 0.1})  # release
    assert tog.update(0.5, bs) is False
    assert tog.update(0.7, bs) is True  # re-armed, fires again


def test_mouth_open_value() -> None:
    assert mouth_open({"jawOpen": 0.7}) == 0.7
    assert mouth_open({}) == 0.0


def test_mouth_open_trigger_fires() -> None:
    trig = MouthOpenTrigger(threshold=0.5, hold=0.2, cooldown=0.0)
    bs = {"jawOpen": 0.9}
    assert trig.update(0.0, bs) is False
    assert trig.update(0.3, bs) is True


def test_wink_trigger_left() -> None:
    trig = WinkTrigger(threshold=0.7, hold=0.2, cooldown=0.0)
    bs = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}
    assert trig.update(0.0, bs) is None
    assert trig.update(0.3, bs) == "wink_left"


def test_wink_trigger_both_closed_no_fire() -> None:
    trig = WinkTrigger(threshold=0.7, hold=0.2, cooldown=0.0)
    bs = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}  # blink, not wink
    assert trig.update(0.0, bs) is None
    assert trig.update(0.5, bs) is None


# speaker.py pearson (line 23 mutants)


def test_pearson_identical() -> None:

    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(a, a) == pytest.approx(1.0)


def test_pearson_uncorrelated() -> None:

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert pearson(a, b) == pytest.approx(-1.0)


def test_pearson_too_short() -> None:

    assert pearson(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_pearson_mismatched_length() -> None:

    assert pearson(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])) == 0.0


def test_pearson_zero_variance() -> None:

    # Constant array: den = 0, returns 0.0 (tests den > 1e-12)
    a = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(a, b) == 0.0


# gate.py defaults (lines 38-45 mutants)


def test_gate_defaults() -> None:
    g = SpeechGate()
    assert g.allow_without_face is True
    # Verify defaults are stored (kills constant mutants)

    sig = inspect.signature(SpeechGate.__init__)
    assert sig.parameters["ambient_fraction"].default == 0.12
    assert sig.parameters["loose_score"].default == 0.08
    assert sig.parameters["pad_s"].default == 0.3
    assert sig.parameters["starved_s"].default == 10.0
    assert sig.parameters["presence_fraction"].default == 0.6
    assert sig.parameters["min_loose"].default == 0.03
    assert sig.parameters["baseline_frames"].default == 300
    assert sig.parameters["baseline_min_frames"].default == 15
