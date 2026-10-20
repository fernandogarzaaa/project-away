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


def test_speech_gate_observe_respects_explicit_user_presence() -> None:
    gate = SpeechGate()
    gate.observe(1.0, active_speaker=0, n_faces=1, score=0.8, frontal=True, user_present=False)
    assert gate._pose[-1] == (False, None)
    assert gate._scores[-1] == (1.0, 0.8)


def test_speech_gate_camera_frame_only_moves_forward() -> None:
    gate = SpeechGate()
    gate.note_camera_frame(10.0)
    gate.note_camera_frame(9.0)
    assert gate.last_camera_frame == 10.0
    gate.note_camera_frame(11.0)
    assert gate.last_camera_frame == 11.0


def test_speech_gate_holder_exception_and_truthy_value() -> None:
    gate = SpeechGate()
    gate.camera_holder = lambda: (_ for _ in ()).throw(RuntimeError("camera query failed"))
    assert gate._holder() is None
    gate.camera_holder = lambda: 123
    assert gate._holder() == "123"


def test_speech_gate_mouth_threshold_uses_only_preceding_known_scores() -> None:
    gate = SpeechGate(baseline_min_frames=3, baseline_frames=3, loose_score=0.8, min_loose=0.03)
    for t, score in [(1.0, 0.1), (2.0, 0.1), (3.0, 0.1), (4.0, 0.9)]:
        gate.observe(t, active_speaker=None, n_faces=1, score=score)
    threshold, baseline = gate.mouth_threshold(4.0)
    assert baseline == pytest.approx(0.1)
    assert threshold == pytest.approx(0.115)  # baseline margin dominates zero MAD


def test_speech_gate_mouth_threshold_falls_back_without_enough_history() -> None:
    gate = SpeechGate(baseline_min_frames=3, loose_score=0.08)
    gate.observe(1.0, active_speaker=None, n_faces=1, score=0.4)
    threshold, baseline = gate.mouth_threshold(2.0)
    assert threshold == 0.08
    assert baseline is None


def test_speech_gate_explain_blocks_when_camera_is_busy_by_default() -> None:
    gate = SpeechGate()
    gate.camera_holder = lambda: "Settings camera check"
    result = gate.explain(1.0, 2.0)
    assert result["allowed"] is False
    assert result["camera_busy"] == "Settings camera check"
    assert "not typed" in result["reason"]


def test_speech_gate_explain_honors_allow_when_camera_is_busy() -> None:
    gate = SpeechGate()
    gate.camera_holder = lambda: "Settings camera check"
    gate.when_camera_busy = "allow"
    result = gate.explain(1.0, 2.0)
    assert result["allowed"] is True
    assert "typing anyway" in result["reason"]


def test_speech_gate_explain_rejects_other_speaker() -> None:
    gate = SpeechGate(ambient_fraction=0.9, min_fraction=0.9)
    gate.enroll({1: 0.5})
    gate.observe(1.0, active_speaker=2, n_faces=1, score=0.0, frontal=False)
    result = gate.explain(1.0, 1.0)
    assert result["allowed"] is False
    assert result["reason"] == "someone else was speaking"


def test_speech_gate_explain_allows_frontal_enrolled_face_with_weak_mouth_motion() -> None:
    gate = SpeechGate(presence_fraction=0.6)
    gate.enroll({1: 0.5})
    for t in (1.0, 1.1, 1.2):
        gate.observe(t, active_speaker=None, n_faces=1, score=0.0, frontal=True, user_present=True)
    result = gate.explain(1.0, 1.2)
    assert result["allowed"] is True
    assert "facing it" in result["reason"]


def test_speech_gate_allow_without_face_and_enrollment_rules() -> None:
    gate = SpeechGate(allow_without_face=False)
    assert gate.allow(1.0, 2.0) is False
    gate.observe(1.5, active_speaker=None, n_faces=0)
    assert gate.allow(1.0, 2.0) is False
    gate.enroll({1: 0.5})
    gate.observe(2.0, active_speaker=1, n_faces=1)
    assert gate.allow(2.0, 2.0) is True


def test_speech_gate_median_empty_odd_and_even() -> None:
    from away.face.gate import _median

    assert _median([]) == 0.0
    assert _median([1.0]) == 1.0
    assert _median([1.0, 2.0, 3.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_face_tracker_exact_distance_and_reacquire_resets_missing() -> None:
    tracker = FaceTracker(max_dist=0.5, max_missing=1)
    assert tracker.update([(0.0, 0.0, 0.2)]) == [0]
    tracker.update([])
    assert tracker.tracks[0].missing == 1
    assert tracker.update([(0.5, 0.0, 0.2)]) == [0]  # max_dist is inclusive
    assert tracker.tracks[0].missing == 0


def test_face_tracker_expires_only_after_max_missing() -> None:
    tracker = FaceTracker(max_missing=2)
    tracker.update([(0.5, 0.5, 0.2)])
    tracker.update([])
    tracker.update([])
    assert 0 in tracker.tracks
    tracker.update([])
    assert 0 not in tracker.tracks


def test_face_tracker_preserves_input_order_for_new_boxes() -> None:
    tracker = FaceTracker()
    assert tracker.update([(0.9, 0.9, 0.1), (0.1, 0.1, 0.2)]) == [0, 1]
    assert tracker.tracks[0].cx == 0.9
    assert tracker.tracks[1].cx == 0.1


def test_speaking_scorer_clips_score_and_bounds_history() -> None:
    scorer = SpeakingScorer(window=3, var_gain=100.0)
    for value in (0.0, 1.0, 0.0, 1.0, 0.0):
        score = scorer.update(7, {"jawOpen": value})
    assert 0.0 <= score <= 1.0
    assert len(scorer._hist[7]) == 3
    scorer.forget(7)
    assert 7 not in scorer._hist


def test_active_speaker_detector_empty_and_single_face_threshold() -> None:
    from away.face.speaker import ActiveSpeakerDetector

    detector = ActiveSpeakerDetector(window=5, speaking_threshold=0.25)
    assert detector.active() is None
    detector.update({4: 0.1}, 0.1)
    assert detector.active() is None
    detector.update({4: 0.25}, 0.1)
    assert detector.active() == 4
    detector.update({}, 0.1)
    assert detector.active() is None


def test_active_speaker_detector_uses_pluggable_model() -> None:
    from away.face.speaker import ActiveSpeakerDetector

    class Model:
        def scores(self, face_series, audio_energy):
            assert set(face_series) == {1, 2}
            assert len(audio_energy) >= 1
            return {1: 0.1, 2: 0.9}

    detector = ActiveSpeakerDetector(window=5, model=Model(), min_correlation=0.5)
    detector.update({1: 0.2, 2: 0.3}, 0.1)
    detector.update({1: 0.1, 2: 0.9}, 0.9)
    detector.update({1: 0.2, 2: 0.8}, 0.8)
    assert detector.active() == 2
    assert detector.correlations() == {1: 0.1, 2: 0.9}


def test_active_speaker_detector_drops_missing_face_series() -> None:
    from away.face.speaker import ActiveSpeakerDetector

    detector = ActiveSpeakerDetector()
    detector.update({1: 0.2, 2: 0.3}, 0.1)
    detector.update({1: 0.3}, 0.2)
    assert set(detector._faces) == {1}


def test_brow_toggle_exact_hold_and_cooldown_boundaries() -> None:
    # hold/cooldown use float-exact boundaries: 1.2 - 1.0 is 0.19999... in binary
    # floating point, so a 0.2s hold at t=1.0 can never read as fully elapsed.
    toggle = BrowRaiseToggle(threshold=0.5, hold=0.25, cooldown=0.5)
    raised = {"browInnerUp": 0.5}
    assert toggle.update(1.0, raised) is False
    assert toggle.update(1.249, raised) is False
    assert toggle.update(1.25, raised) is True
    toggle.update(1.3, {"browInnerUp": 0.0})  # re-arm
    assert toggle.update(1.4, raised) is False  # cooldown has not elapsed
    assert toggle.update(1.75, raised) is True


def test_brow_toggle_midrange_resets_hold_without_rearming() -> None:
    toggle = BrowRaiseToggle(threshold=0.5, hold=0.2, cooldown=0.0)
    assert toggle.update(0.0, {"browInnerUp": 0.8}) is False
    assert toggle.update(0.1, {"browInnerUp": 0.4}) is False  # between release and threshold
    assert toggle.update(0.25, {"browInnerUp": 0.8}) is False
    assert toggle.update(0.46, {"browInnerUp": 0.8}) is True


def test_wink_threshold_and_open_max_are_inclusive() -> None:
    trigger = WinkTrigger(threshold=0.7, open_max=0.2, hold=0.1, cooldown=0.0)
    assert trigger.update(0.0, {"eyeBlinkLeft": 0.7, "eyeBlinkRight": 0.2}) is None
    assert trigger.update(0.1, {"eyeBlinkLeft": 0.7, "eyeBlinkRight": 0.2}) == "wink_left"


def test_wink_requires_one_eye_open_and_rearms_after_release() -> None:
    trigger = WinkTrigger(threshold=0.7, open_max=0.2, hold=0.1, cooldown=0.0)
    assert trigger.update(0.0, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}) is None
    assert trigger.update(0.2, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) is None  # hold starts
    assert trigger.update(0.31, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) == "wink_left"
    assert trigger.update(0.4, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) is None  # disarmed
    trigger.update(0.5, {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0})  # release re-arms
    assert trigger.update(0.6, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) is None  # hold restarts
    assert trigger.update(0.71, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) == "wink_left"


# Round 2: constructor defaults (signature-value mutants)


def test_gate_remaining_defaults() -> None:
    sig = inspect.signature(SpeechGate.__init__)
    assert sig.parameters["baseline_mads"].default == 3.0
    assert sig.parameters["baseline_margin"].default == 0.015
    gate = SpeechGate()
    assert gate.last_camera_frame == -1e9
    assert gate.camera_recent_s == 2.0
    assert gate.when_camera_busy == "block"


def test_wink_trigger_defaults() -> None:
    sig = inspect.signature(WinkTrigger.__init__)
    assert sig.parameters["threshold"].default == 0.5
    assert sig.parameters["open_max"].default == 0.3
    assert sig.parameters["hold"].default == 0.25
    assert sig.parameters["cooldown"].default == 0.8
    trigger = WinkTrigger()
    assert trigger._armed == {"left": True, "right": True}
    assert trigger._last == -1e9
    assert trigger._t0 == {"left": None, "right": None}


def test_mouth_open_trigger_defaults() -> None:
    sig = inspect.signature(MouthOpenTrigger.__init__)
    assert sig.parameters["threshold"].default == 0.5
    assert sig.parameters["hold"].default == 0.8
    assert sig.parameters["cooldown"].default == 1.0


# Round 2: pearson tiny-variance boundary


def test_pearson_tiny_but_nonzero_variance() -> None:
    # den = 6.67e-7: above 1e-12 but far below 1.000000000001.
    # Kills the 1e-12 -> 1.000000000001 mutant (which would return 0.0 here).
    a = np.array([0.0, 0.001, 0.0])
    assert pearson(a, a) == pytest.approx(1.0)


# Round 2: observe() presence boundaries


def test_observe_n_faces_zero_is_not_present() -> None:
    # Kills n_faces > 0 -> n_faces >= 0 (both occurrences): a frame with no
    # faces is not "present", even though 0 >= 0.
    gate = SpeechGate()
    gate.observe(1.0, active_speaker=None, n_faces=0, frontal=True)
    assert gate._pose[-1] == (False, None)


def test_observe_single_face_is_present() -> None:
    # Kills n_faces > 0 -> n_faces > 1: exactly one face counts as present.
    gate = SpeechGate()
    gate.observe(1.0, active_speaker=None, n_faces=1, frontal=True)
    assert gate._pose[-1] == (True, True)


# Round 2: mouth_threshold() frame selection


def test_mouth_threshold_excludes_the_boundary_frame() -> None:
    # Kills t < before -> t <= before (and the `and` -> `or` mutant on the
    # same comprehension): a score at exactly `before` is history for later
    # calls, not for this one (2 known < baseline_min_frames=3).
    gate = SpeechGate(baseline_min_frames=3, baseline_frames=3)
    for t, score in [(1.0, 0.1), (2.0, 0.1), (3.0, 0.1)]:
        gate.observe(t, active_speaker=None, n_faces=1, score=score)
    threshold, baseline = gate.mouth_threshold(3.0)
    assert (threshold, baseline) == (0.08, None)


# Round 2: WinkTrigger.update() edge semantics


def test_wink_missing_key_defaults_to_open() -> None:
    # Kills the 0.0 -> 1.0 default mutants on both blendshape lookups:
    # a missing key means "open" (0.0), never "closed" (1.0).
    trigger = WinkTrigger(threshold=0.7, open_max=0.2, hold=0.1, cooldown=0.0)
    # eyeBlinkLeft missing -> open: right eye closed alone is a wink_right
    assert trigger.update(0.0, {"eyeBlinkRight": 0.9}) is None
    assert trigger.update(0.2, {"eyeBlinkRight": 0.9}) == "wink_right"
    trigger2 = WinkTrigger(threshold=0.7, open_max=0.2, hold=0.1, cooldown=0.0)
    # eyeBlinkRight missing -> open: left eye closed alone is a wink_left
    assert trigger2.update(0.0, {"eyeBlinkLeft": 0.9}) is None
    assert trigger2.update(0.2, {"eyeBlinkLeft": 0.9}) == "wink_left"


def test_wink_rearm_needs_strictly_below_release() -> None:
    # Kills closed < threshold*0.6 -> closed <= threshold*0.6: exactly at
    # 0.42 the eye is not "open", so the trigger must not re-arm.
    trigger = WinkTrigger(threshold=0.7, open_max=0.2, hold=0.1, cooldown=0.0)
    assert trigger.update(0.0, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) is None
    assert trigger.update(0.2, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) == "wink_left"
    trigger.update(0.3, {"eyeBlinkLeft": 0.42, "eyeBlinkRight": 0.42})  # exactly at release
    # Still disarmed: the next wink pose cannot fire
    assert trigger.update(0.4, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) is None
    assert trigger.update(0.6, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) is None


def test_wink_cooldown_boundary_is_inclusive() -> None:
    # Kills t - _last >= cooldown -> t - _last > cooldown: exactly at the
    # cooldown boundary the trigger may fire again.
    trigger = WinkTrigger(threshold=0.7, open_max=0.2, hold=0.1, cooldown=0.5)
    assert trigger.update(1.0, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) is None
    assert trigger.update(1.25, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) == "wink_left"
    trigger.update(1.3, {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0})  # re-arm
    assert trigger.update(1.4, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) is None
    assert trigger.update(1.75, {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) == "wink_left"


# Round 2: explain() window statistics


def test_explain_reports_exact_window_statistics() -> None:
    gate = SpeechGate()
    gate.enroll({1: 0.5, 2: 0.3})
    # Window for explain(1.0, 1.0) is [0.7, 1.3]: boundary frames included.
    gate.observe(0.7, active_speaker=1, n_faces=1, score=0.5, frontal=True)
    gate.observe(1.0, active_speaker=2, n_faces=1, score=0.9, frontal=True)
    gate.observe(1.1, active_speaker=None, n_faces=1, score=0.08, frontal=False)
    gate.observe(1.2, active_speaker=1, n_faces=1, score=0.9, frontal=True)
    gate.observe(1.25, active_speaker=None, n_faces=1, score=None, frontal=False)
    gate.observe(1.3, active_speaker=None, n_faces=0, score=None, frontal=None)
    info = gate.explain(1.0, 1.0)
    assert info["frames"] == 6
    assert info["enrolled"] is True
    assert info["face_fraction"] == 0.83  # 5/6
    assert info["max_score"] == 0.9
    assert info["speaker_frames"] == 2
    assert info["other_speaker_frames"] == 1
    assert info["mouth_moving_frames"] == 1
    assert info["frontal_fraction"] == 0.5
    assert info["speaking_fraction"] == 0.5
    assert info["allowed"] is True
    assert info["reason"] == "facing the screen and speaking"


def test_explain_rounding_precision() -> None:
    gate = SpeechGate()
    gate.observe(1.0, active_speaker=None, n_faces=1, score=0.126)
    gate.observe(1.1, active_speaker=None, n_faces=0, score=None)
    gate.observe(1.2, active_speaker=None, n_faces=0, score=None)
    info = gate.explain(1.0, 1.2)
    assert info["face_fraction"] == 0.33  # round(1/3, 2), not 0.333
    assert info["max_score"] == 0.13  # round(0.126, 2), not 0.126


def test_explain_identity_check_blocks_non_user() -> None:
    import types

    gate = SpeechGate()
    gate.enroll({1: 0.5})
    gate.identity = types.SimpleNamespace(is_user=lambda fid: False)
    gate.observe(1.0, active_speaker=1, n_faces=1, score=0.9)
    info = gate.explain(1.0, 1.0)
    assert info["allowed"] is False
    assert info["reason"] == "identity check: not the enrolled user"


def test_explain_identity_check_passes_for_user() -> None:
    import types

    gate = SpeechGate()
    gate.enroll({1: 0.5})
    gate.identity = types.SimpleNamespace(is_user=lambda fid: True)
    gate.observe(1.0, active_speaker=1, n_faces=1, score=0.9, frontal=True)
    info = gate.explain(1.0, 1.0)
    assert info["allowed"] is True


def test_explain_speaking_fraction_boundary_is_inclusive() -> None:
    # frac exactly == min(ambient_fraction, min_fraction) must allow.
    gate = SpeechGate(ambient_fraction=0.5, min_fraction=0.5)
    gate.enroll({1: 0.5})
    gate.observe(1.0, active_speaker=1, n_faces=1, score=0.9)
    gate.observe(1.1, active_speaker=1, n_faces=1, score=0.9)
    gate.observe(1.2, active_speaker=None, n_faces=1, score=0.0)
    gate.observe(1.3, active_speaker=None, n_faces=1, score=0.0)
    info = gate.explain(1.0, 1.3)
    assert info["speaking_fraction"] == 0.5  # (2 + 0) / 4
    assert info["allowed"] is True


def test_explain_presence_boundary_is_inclusive() -> None:
    # presence exactly == presence_fraction must allow (mouth motion weak).
    gate = SpeechGate(presence_fraction=0.6)
    for i, t in enumerate([1.0, 1.1, 1.2, 1.3, 1.4]):
        gate.observe(t, active_speaker=None, n_faces=1, score=0.0, frontal=(i < 3))
    info = gate.explain(1.0, 1.4)
    assert info["frontal_fraction"] == 0.6  # 3/5
    assert info["allowed"] is True
    assert info["reason"] == "enrolled face at the screen, facing it (mouth motion weak)"


# Round 3: remaining reachable survivors


def test_observe_user_present_with_no_faces_is_not_present() -> None:
    # Kills the second n_faces > 0 -> n_faces >= 0 (the user_present branch):
    # an explicit user_present=True still needs n_faces > 0.
    gate = SpeechGate()
    gate.observe(1.0, active_speaker=None, n_faces=0, frontal=True, user_present=True)
    assert gate._pose[-1] == (False, None)


def test_explain_unenrolled_speaker_frames_count_each_frame_once() -> None:
    # Kills the sum(1 ...) -> sum(2 ...) in the not-enrolled branch of explain.
    gate = SpeechGate()
    gate.observe(1.0, active_speaker=1, n_faces=1, score=0.9)
    gate.observe(1.1, active_speaker=2, n_faces=1, score=0.9)
    info = gate.explain(1.0, 1.1)
    assert info["enrolled"] is False
    assert info["speaker_frames"] == 2
    assert info["other_speaker_frames"] == 0


def test_explain_loose_ignores_frames_without_faces() -> None:
    # Kills n > 0 -> n >= 0 in the loose counter: a mouth score on a frame
    # with no face must not count as mouth movement.
    gate = SpeechGate()
    gate.observe(1.0, active_speaker=None, n_faces=0, score=0.9)
    gate.observe(1.1, active_speaker=1, n_faces=1, score=0.9)
    gate.enroll({1: 0.5})
    info = gate.explain(1.0, 1.1)
    assert info["mouth_moving_frames"] == 0


def test_explain_others_compared_against_hits_plus_loose() -> None:
    # Kills others > hits + loose -> others > hits - loose:
    # hits=2, loose=2, others=3: 3 > 4 is False (allowed path),
    # but 3 > 0 is True (mutant would claim someone else was speaking).
    gate = SpeechGate()
    gate.enroll({1: 0.5, 2: 0.3})
    gate.observe(1.0, active_speaker=1, n_faces=1, score=0.9)
    gate.observe(1.1, active_speaker=1, n_faces=1, score=0.9)
    gate.observe(1.2, active_speaker=None, n_faces=1, score=0.9)
    gate.observe(1.3, active_speaker=None, n_faces=1, score=0.9)
    gate.observe(1.4, active_speaker=2, n_faces=1, score=0.9)
    gate.observe(1.5, active_speaker=2, n_faces=1, score=0.9)
    gate.observe(1.6, active_speaker=2, n_faces=1, score=0.9)
    info = gate.explain(1.0, 1.6)
    assert info["speaker_frames"] == 2
    assert info["mouth_moving_frames"] == 2
    assert info["other_speaker_frames"] == 3
    assert info["allowed"] is True


def test_explain_fraction_rounding_precision() -> None:
    # Kills round(x, 2) -> round(x, 3) on speaking_fraction and frontal_fraction.
    gate = SpeechGate()
    gate.observe(1.0, active_speaker=None, n_faces=1, score=0.5, frontal=True)
    gate.observe(1.1, active_speaker=None, n_faces=1, score=0.0, frontal=False)
    gate.observe(1.2, active_speaker=None, n_faces=1, score=0.0, frontal=False)
    info = gate.explain(1.0, 1.2)
    assert info["speaking_fraction"] == 0.33  # round(1/3, 2)
    assert info["frontal_fraction"] == 0.33  # round(1/3, 2)
