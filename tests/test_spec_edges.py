"""Boundary / exact-value tests added after mutation testing (testteeth) showed gaps."""

import inspect
import math

import numpy as np
import pytest

from away.config import CursorConfig, GestureConfig
from away.hands import landmarks as L
from away.hands.cursor import CursorMapper, accelerate
from away.hands.filters import OneEuro2D, OneEuroFilter, smoothing_factor
from away.hands.gestures import GestureStateMachine
from away.hands.synthetic import pose
from away.voice import cleanup as C
from away.voice import dictation as Dm
from away.voice import vad as V
from away.voice.asr import FakeASR
from away.voice.output import ListSink
from away.wifi.classifier import CoarseCSIClassifier, CSIThresholds

# ------------------------------------------------------------------ filters


def test_one_euro_defaults():
    f = OneEuroFilter()
    assert (f.min_cutoff, f.beta, f.d_cutoff) == (1.0, 0.0, 1.0)
    g = OneEuro2D()
    assert (g.fx.min_cutoff, g.fx.beta, g.fx.d_cutoff) == (1.0, 0.0, 1.0)
    assert (g.fy.min_cutoff, g.fy.beta, g.fy.d_cutoff) == (1.0, 0.0, 1.0)


def expected_step(x0, x1, dt, min_cutoff, beta, d_cutoff, dx_prev=0.0):
    a_d = smoothing_factor(dt, d_cutoff)
    dx_hat = a_d * (x1 - x0) / dt + (1 - a_d) * dx_prev
    a = smoothing_factor(dt, min_cutoff + beta * abs(dx_hat))
    return a * x1 + (1 - a) * x0


@pytest.mark.parametrize("x0,x1", [(0.0, 1.0), (2.0, 3.0), (5.0, 1.0)])
def test_one_euro_exact_with_beta(x0, x1):
    f = OneEuroFilter(min_cutoff=0.5, beta=0.7, d_cutoff=2.0)
    f(0.0, x0)
    assert f(0.25, x1) == pytest.approx(expected_step(x0, x1, 0.25, 0.5, 0.7, 2.0))


def test_one_euro_reset_clears_derivative():
    f = OneEuroFilter(min_cutoff=0.5, beta=0.7, d_cutoff=2.0)
    f(0.0, 0.0)
    f(0.25, 5.0)
    f.reset()
    assert f._dx_prev == 0.0 and f._x_prev is None and f._t_prev is None
    f(1.0, 2.0)
    assert f(1.25, 3.0) == pytest.approx(expected_step(2.0, 3.0, 0.25, 0.5, 0.7, 2.0))


# ---------------------------------------------------------------- landmarks


def test_mediapipe_landmark_indices():
    assert (L.WRIST, L.THUMB_TIP) == (0, 4)
    assert (L.INDEX_MCP, L.INDEX_PIP, L.INDEX_TIP) == (5, 6, 8)
    assert (L.MIDDLE_MCP, L.MIDDLE_PIP, L.MIDDLE_TIP) == (9, 10, 12)
    assert (L.RING_MCP, L.RING_PIP, L.RING_TIP) == (13, 14, 16)
    assert (L.PINKY_MCP, L.PINKY_PIP, L.PINKY_TIP) == (17, 18, 20)
    assert L.NUM_LANDMARKS == 21


def test_normalize_eps_boundary():
    lm = np.zeros((21, 3))
    lm[L.MIDDLE_MCP] = (1e-6, 0, 0)
    out = L.normalize(lm, eps=1e-6)  # exactly eps is allowed
    assert out[L.MIDDLE_MCP, 0] == pytest.approx(1.0)


def test_normalize_sequence_message():
    with pytest.raises(ValueError, match="sequence"):
        L.normalize_sequence(np.zeros((21, 3)))


def test_sequence_velocity_uses_current_frame_scale():
    seq = np.stack(
        [
            pose("palm", center=(0.3, 0.5), scale=0.1),
            pose("palm", center=(0.32, 0.5), scale=0.1),
            pose("palm", center=(0.36, 0.5), scale=0.2),
        ]
    )
    v = L.sequence_features(seq)[:, -2]
    assert v[0] == 0.0
    assert v[1] == pytest.approx(0.02 / 0.1)
    assert v[2] == pytest.approx(0.04 / 0.2)


def test_static_features_flatten_order():
    lm = pose("point")
    f = L.static_features(lm)
    n = L.normalize(lm)
    assert f[0] == n[0, 0] and f[1] == n[0, 1] and f[2] == n[1, 0]


def test_infer_side_uses_x_not_y():
    lm = np.zeros((21, 3))
    lm[L.THUMB_TIP] = (0.9, 0.1, 0)
    lm[L.PINKY_TIP] = (0.1, 0.9, 0)
    assert L.infer_side(lm) == "Right"
    lm[L.THUMB_TIP] = (0.1, 0.9, 0)
    lm[L.PINKY_TIP] = (0.9, 0.1, 0)
    assert L.infer_side(lm) == "Left"


# ------------------------------------------------------------------- cursor


def test_accelerate_more():
    assert accelerate(1.0, 1.0) == 1.0
    assert accelerate(0.04, 3.0) == pytest.approx(0.04 * 16)  # (4)^(3-1)


def test_screen_validation_each_axis():
    CursorMapper((1, 1))
    with pytest.raises(ValueError):
        CursorMapper((100, 0))
    with pytest.raises(ValueError):
        CursorMapper((0, 100))


def slow_abs():
    return CursorConfig(mode="absolute", active_region=(0, 0, 1, 1), dead_zone=0.0, min_cutoff=0.01, beta=0.0)


def test_reset_clears_filter():
    m = CursorMapper((1001, 1001), slow_abs())
    m.update(0.0, 0.2, 0.2)
    m.reset()
    assert m.update(0.1, 0.8, 0.6) == (800, 600)


def test_anchor_change_clears_filter():
    m = CursorMapper((1001, 1001), slow_abs())
    m.update(0.0, 0.2, 0.2, "index")
    assert m.update(0.1, 0.8, 0.6, "pinch") == (800, 600)
    held = m.update(0.2, 0.2, 0.2, "pinch")
    assert held[0] > 700  # same anchor: heavily smoothed


def test_absolute_dead_zone_keeps_asymmetric_position():
    cfg = CursorConfig(mode="absolute", active_region=(0, 0, 1, 1), dead_zone=0.05, min_cutoff=1e6, d_cutoff=1e6)
    m = CursorMapper((1001, 501), cfg)
    assert m.update(0.0, 0.2, 0.6) == (200, 300)
    assert m.update(0.1, 0.21, 0.6) == (200, 300)


def test_relative_axes_are_not_swapped():
    cfg = CursorConfig(gain=1.0, acceleration=1.0, dead_zone=0.0, min_cutoff=1e6, d_cutoff=1e6)
    m = CursorMapper((1000, 600), cfg)
    assert m.update(0.0, 0.1, 0.7) == (500, 300)
    assert m.update(0.1, 0.12, 0.7) == (520, 300)
    assert m.update(0.2, 0.12, 0.69) == (520, 290)


# ----------------------------------------------------------------- gestures


def palm_at(x, y=0.5):
    return pose("palm", center=(x, y))


def test_reset_initial_state():
    sm = GestureStateMachine()
    assert sm._pinch_t0 == 0.0 and sm._pinch_pos == (0.0, 0.0)
    assert sm._palm_pos == (0.0, 0.0) and sm._palm_t0 is None
    assert sm._last_swipe_t == -1e9 and sm._last_tap_t is None and sm._scroll_prev_y is None


def test_two_frame_swipe_and_duration():
    sm = GestureStateMachine(GestureConfig(swipe_min_dx=0.25))
    assert sm.update(0.0, palm_at(0.2)) == []
    ev = sm.update(0.125, palm_at(0.5))
    assert [e.kind for e in ev] == ["swipe_right"]
    assert ev[0].data["duration"] == 0.125
    assert ev[0].data["dx"] == pytest.approx(0.3)


def test_swipe_window_boundary_inclusive():
    sm = GestureStateMachine(GestureConfig(swipe_max_time=0.5, swipe_min_dx=0.25))
    sm.update(0.0, palm_at(0.25))
    assert [e.kind for e in sm.update(0.5, palm_at(0.5))] == ["swipe_right"]
    sm2 = GestureStateMachine(GestureConfig(swipe_max_time=0.5, swipe_min_dx=0.25))
    sm2.update(0.0, palm_at(0.25))
    assert sm2.update(0.625, palm_at(0.5)) == []


def test_swipe_measures_from_oldest_sample():
    sm = GestureStateMachine(GestureConfig(swipe_min_dx=0.25))
    sm.update(0.0, palm_at(0.2))
    assert sm.update(0.0625, palm_at(0.3)) == []
    assert [e.kind for e in sm.update(0.125, palm_at(0.46))] == ["swipe_right"]


def test_swipe_min_dx_inclusive_and_diagonal_ratio_strict():
    sm = GestureStateMachine(GestureConfig(swipe_min_dx=0.25))
    sm.update(0.0, palm_at(0.25))
    assert [e.kind for e in sm.update(0.125, palm_at(0.5))] == ["swipe_right"]
    sm2 = GestureStateMachine(GestureConfig(swipe_min_dx=0.25))
    sm2.update(0.0, palm_at(0.25, 0.25))
    assert sm2.update(0.125, palm_at(0.75, 0.5)) == []  # |dx| == 2|dy| is not horizontal enough


def test_swipe_cooldown_boundary_and_history_cleared():
    cfg = GestureConfig(swipe_min_dx=0.25, swipe_max_time=1.0)
    sm = GestureStateMachine(cfg)
    sm.update(0.0, palm_at(0.0))
    assert [e.kind for e in sm.update(0.125, palm_at(0.3))] == ["swipe_right"]
    assert sm.update(0.625, palm_at(0.31)) == []  # history was cleared by the swipe
    sm2 = GestureStateMachine(cfg)
    sm2.update(0.0, palm_at(0.0))
    sm2.update(0.125, palm_at(0.3))
    sm2.update(0.25, palm_at(0.3))
    assert [e.kind for e in sm2.update(0.625, palm_at(0.0))] == ["swipe_left"]  # 0.5 s later: allowed


def test_pointing_hand_never_swipes():
    sm = GestureStateMachine()
    kinds = []
    for i in range(6):
        kinds += [e.kind for e in sm.update(i / 30, pose("point", center=(0.2 + 0.08 * i, 0.5)))]
    assert set(kinds) == {"move"}


def test_non_palm_frame_clears_swipe_history():
    sm = GestureStateMachine(GestureConfig(swipe_min_dx=0.25, swipe_max_time=2.0))
    sm.update(0.0, palm_at(0.2))
    sm.update(0.1, pose("fist", center=(0.3, 0.5)))
    assert sm.update(0.2, palm_at(0.46)) == []


def test_pause_clears_swipe_history():
    cfg = GestureConfig(swipe_min_dx=0.25, swipe_max_time=5.0, clutch_hold=0.25)
    sm = GestureStateMachine(cfg)
    t = 0.0
    sm.update(t, palm_at(0.0))
    for _ in range(4):
        t += 0.125
        sm.update(t, palm_at(0.2))
    assert sm.paused
    t += 0.125
    sm.update(t, pose("fist", center=(0.2, 0.5)))
    for _ in range(4):
        t += 0.125
        sm.update(t, palm_at(0.2))
    assert not sm.paused
    t += 0.125
    assert sm.update(t, palm_at(0.26)) == []


def test_clutch_with_asymmetric_position():
    sm = GestureStateMachine(GestureConfig(clutch_hold=0.25))
    ev = []
    for i in range(6):
        ev += sm.update(i * 0.125, pose("palm", center=(0.3, 0.7)))
    assert [e.kind for e in ev] == ["clutch"]


def test_drag_events_carry_pinch_midpoint_and_reset():
    sm = GestureStateMachine(GestureConfig(drag_hold=0.25, drag_move=1.0))
    sm.update(0.0, pose("pinch_open", center=(0.3, 0.6)))
    p = pose("pinch", center=(0.3, 0.6))
    mid = ((p[4, 0] + p[8, 0]) / 2, (p[4, 1] + p[8, 1]) / 2)
    sm.update(0.25, p)
    ev = sm.update(0.5, p)  # exactly drag_hold later
    assert [e.kind for e in ev] == ["drag_start", "move"]
    assert ev[0].data == {"x": pytest.approx(mid[0]), "y": pytest.approx(mid[1])}
    end = sm.update(0.625, pose("pinch_open", center=(0.3, 0.6)))
    assert end[0].kind == "drag_end"
    assert end[0].data["y"] > 0.3
    assert not sm.dragging
    sm.update(2.0, p)
    assert [e.kind for e in sm.update(2.0625, pose("pinch_open", center=(0.3, 0.6)))][0] == "click"


def test_double_click_window_exact_boundary():
    sm = GestureStateMachine(GestureConfig(double_pinch_window=0.5))
    sm.update(0.0, pose("pinch"))
    assert [e.kind for e in sm.update(0.25, pose("pinch_open"))][0] == "click"
    sm.update(0.5, pose("pinch"))
    assert [e.kind for e in sm.update(0.75, pose("pinch_open"))][0] == "double_click"


# ------------------------------------------------------------------ cleanup


def test_scratch_that_with_double_terminator():
    assert C.apply_scratch_that("Done?! scratch that ok") == "Done? ok"


def test_correction_anchor_on_last_word():
    assert C.apply_corrections("buy apples no wait apples and pears") == "buy apples and pears"


def test_correction_word_anchor_beats_number_rule():
    assert C.apply_corrections("volume 30 then 40 no wait 30 again") == "volume 30 again"


def test_correction_caps_replacement_at_three_words():
    assert C.apply_corrections("please buy red apples today no wait get green pears now") == (
        "please buy get green pears now"
    )


def test_vocabulary_arrow_in_replacement():
    v = C.Vocabulary.parse("x -> y -> z")
    assert v.apply("x") == "y -> z"


def test_is_code_context_returns_bool():
    assert C._is_code_context("Notes") is False
    assert C._is_code_context("") is False


def test_llm_default_timeout():
    assert C.LLMCleaner(C.RuleCleaner(), base_url="").timeout == 8.0


@pytest.mark.parametrize("n,ok", [(10, True), (9, False), (32, True), (33, False)])
def test_llm_length_ratio_bounds(monkeypatch, n, ok):
    import io
    import json

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    out = "y" * n
    monkeypatch.setattr(
        C.urllib.request,
        "urlopen",
        lambda req, timeout: R(json.dumps({"choices": [{"message": {"content": out}}]}).encode()),
    )
    llm = C.LLMCleaner(C.RuleCleaner(), base_url="http://x")
    base = "Abcd efgh ijkl mnop."
    assert len(base) == 20
    assert llm.clean("abcd efgh ijkl mnop") == (out if ok else base)


# ---------------------------------------------------------------------- vad


def test_frame_rms_single_frame_and_frame_one():
    assert np.allclose(V.frame_rms(np.array([3.0, -3.0]), 2), [3.0])
    assert np.allclose(V.frame_rms(np.array([1.0, -2.0]), 1), [1.0, 2.0])


def test_energy_vad_defaults():
    v = V.EnergyVAD()
    assert (v.sr, v.frame, v.ratio, v.min_rms, v.hangover, v.min_speech) == (16000, 480, 3.0, 0.005, 8, 3)
    assert V.make_vad("energy").sr == 16000


def frames_audio(levels, frame=4):
    return np.concatenate([np.full(frame, lv, dtype=np.float64) for lv in levels])


def test_energy_threshold_strict_and_ratio():
    v = V.EnergyVAD(sample_rate=1000, frame_ms=4, ratio=2.0, min_rms=0.0, hangover_frames=0, min_speech_frames=1)
    levels = [0.25] * 18 + [0.5, 0.5001]
    flags = v.frame_flags(frames_audio(levels))
    assert flags.tolist() == [False] * 19 + [True]


def test_hangover_exact_and_initial_state():
    v = V.EnergyVAD(sample_rate=1000, frame_ms=4, ratio=2.0, min_rms=0.0, hangover_frames=2, min_speech_frames=1)
    levels = [0.25] * 10 + [1.0] + [0.25] * 5
    assert v.frame_flags(frames_audio(levels)).tolist() == [False] * 10 + [True, True, True] + [False] * 3


def test_segment_closing_at_end_and_min_speech_inclusive():
    v = V.EnergyVAD(sample_rate=1000, frame_ms=4, ratio=2.0, min_rms=0.0, hangover_frames=0, min_speech_frames=2)
    audio = frames_audio([0.25] * 10 + [1.0, 1.0])
    segs = v.segments(audio)
    assert [(s.start, s.end) for s in segs] == [(40, 48)]


def test_trim_keeps_speech_at_start():
    v = V.EnergyVAD(sample_rate=1000, frame_ms=4, ratio=2.0, min_rms=0.0, hangover_frames=0, min_speech_frames=1)
    audio = frames_audio([1.0, 1.0] + [0.25] * 10)
    out = v.trim(audio, pad_ms=4)
    assert len(out) == 12 and out[0] == 1.0


def test_device_default_signatures():
    assert inspect.signature(V.SileroVAD).parameters["sample_rate"].default == 16000
    assert inspect.signature(V.SileroVAD).parameters["threshold"].default == 0.5
    assert inspect.signature(V.SileroVAD.trim).parameters["pad_ms"].default == 150
    assert inspect.signature(V.EnergyVAD.trim).parameters["pad_ms"].default == 150
    assert inspect.signature(V.make_vad).parameters["sample_rate"].default == 16000
    mic = inspect.signature(Dm.MicStream).parameters
    assert mic["sample_rate"].default == 16000 and mic["block_ms"].default == 30
    probe = inspect.signature(Dm.probe_microphone).parameters
    assert probe["seconds"].default == 0.5 and probe["sample_rate"].default == 16000
    assert inspect.signature(Dm.Trimmer.trim).parameters["pad_ms"].default == 150


# ---------------------------------------------------------------- dictation


def test_dictation_defaults_and_empty_stop():
    s = Dm.DictationSession(FakeASR(), C.RuleCleaner(), ListSink(), V.EnergyVAD())
    assert s.sr == 16000 and s._t_start == 0.0
    assert Dm.DictationResult("ok").seconds == 0.0
    s.start()
    r = s.stop()
    assert r.status == "silence" and r.seconds == 0.0


def test_enabling_while_recording_keeps_recording():
    s = Dm.DictationSession(FakeASR(), C.RuleCleaner(), ListSink(), V.EnergyVAD())
    s.start()
    s.feed(np.ones(10))
    s.set_enabled(True)
    assert s.recording and len(s._chunks) == 1
    s.set_enabled(False)
    assert not s.recording and s._chunks == []


# ------------------------------------------------------------ wifi classifier


def test_csi_threshold_defaults():
    th = CSIThresholds()
    assert (th.presence, th.motion, th.rssi_presence, th.rssi_motion) == (0.015, 0.08, 0.3, 1.5)
    assert (th.min_net_rotations, th.directional_ratio, th.swipe_ratio) == (1.0, 0.6, 0.3)
    assert (th.min_swipe_rotations, th.direction_sign) == (2.0, 1.0)


def test_classifier_exact_confidences():
    c = CoarseCSIClassifier(CSIThresholds(presence=0.1, motion=0.2, rssi_presence=1.0, rssi_motion=2.0))
    assert c.classify({"energy": 0.05}) == ("empty", 0.75)
    assert c.classify({"energy": 0.25}) == ("motion", 0.625)
    assert c.classify({"energy": 0.3, "rotation_net": 3.0, "rotation_total": 2.5}) == ("push", 1.0)
    assert c.classify({"energy": 0.3, "rotation_net": 0.5, "rotation_total": 1.0}) == ("motion", 0.5)
    assert c.classify({"rssi_energy": 0.5}) == ("empty", 0.75)
    assert c.classify({"rssi_energy": 1.5}) == ("presence", 0.5)
    assert c.classify({"rssi_energy": 3.0}) == ("motion", 0.75)


def test_calibration_edge_cases():
    c = CoarseCSIClassifier(CSIThresholds(presence=0.015, motion=0.08))
    th = c.calibrate([({"energy": 0.01}, "empty"), ({"energy": 0.09}, "presence"), ({}, "swipe")])
    assert th.presence == pytest.approx(math.sqrt(0.01 * 0.09))
    assert th.motion == 0.08  # moving group had no energy feature
    c2 = CoarseCSIClassifier(CSIThresholds(presence=0.015))
    assert c2.calibrate([({"energy": 0.05}, "empty"), ({"energy": 0.05}, "presence")]).presence == 0.015
    c3 = CoarseCSIClassifier(CSIThresholds(motion=0.08))
    assert c3.calibrate([({"energy": 0.1}, "presence"), ({"energy": 0.1}, "motion")]).motion == 0.08
