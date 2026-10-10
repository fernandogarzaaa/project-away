"""More boundary/default tests from the mutation-testing pass (wifi features, face, fusion)."""

import inspect

import numpy as np
import pytest
from scipy import signal

from away.events import EventBus
from away.face.commands import BrowRaiseToggle
from away.face.gate import SpeechGate
from away.face.speaker import ActiveSpeakerDetector
from away.face.speaking import SpeakingScorer, mouth_openness
from away.face.tracking import FaceTracker, Track
from away.fusion import GestureFusion
from away.hands.gestures import GestureEvent
from away.wifi import features as F
from away.wifi.sources import synthetic_csi


def ref_hampel(x, k, n_sigma):
    """Straight-from-the-definition reference implementation."""
    x = np.asarray(x, float)
    y = x.copy()
    for i in range(len(x)):
        w = x[max(0, i - k) : min(len(x), i + k + 1)]
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        if mad > 0 and abs(x[i] - med) > n_sigma * mad:
            y[i] = med
    return y


@pytest.mark.parametrize("k,n_sigma", [(1, 3.0), (2, 2.0), (5, 3.0)])
def test_hampel_matches_reference(k, n_sigma):
    rng = np.random.default_rng(k)
    x = rng.normal(0, 1, 60)
    x[rng.choice(60, 6, replace=False)] += rng.choice([-8, 8], 6)
    assert np.allclose(F.hampel(x, k=k, n_sigma=n_sigma), ref_hampel(x, k, n_sigma))


def test_hampel_defaults():
    p = inspect.signature(F.hampel).parameters
    assert p["k"].default == 5 and p["n_sigma"].default == 3.0
    x = np.random.default_rng(0).normal(0, 1, 40)
    x[20] = 30
    assert np.allclose(F.hampel(x), ref_hampel(x, 5, 3.0))


def test_sanitize_phase_per_packet_rows():
    k = np.arange(20)
    rows = [np.exp(1j * (0.1 * r + (0.05 * (r + 1)) * k)) for r in range(3)]
    out = F.sanitize_phase(np.stack(rows))
    assert out.shape == (3, 20)
    assert np.allclose(out, 0, atol=1e-9)


def test_csi_ratio_eps_sign():
    h = np.zeros((1, 2, 2), dtype=complex)
    h[0, 0] = 1.0
    r = F.csi_ratio(h, eps=1e-6)
    assert np.all(r.real > 0)


def test_normalize_amplitude_shape_and_means():
    amp = np.array([[0.25, 0.0, 2.0], [0.75, 0.0, 6.0]])
    n = F.normalize_amplitude(amp)
    assert n.shape == (2, 3)
    assert np.allclose(n.mean(axis=0), [1.0, 0.0, 1.0])


def test_first_pc_is_centered():
    x = np.stack([np.arange(10.0), 2 * np.arange(10.0)], axis=1) + 100
    pc = F.first_pc(x)
    assert abs(pc.mean()) < 1e-9
    assert np.ptp(pc) == pytest.approx(np.sqrt(5) * 9)


def test_doppler_noverlap_default_and_explicit():
    s = np.sin(np.arange(256) / 3)
    f, t, S = F.doppler_spectrogram(s, 100.0, nperseg=64)
    f2, t2, _ = signal.stft(s, fs=100.0, nperseg=64, noverlap=48)
    assert np.allclose(t, t2) and np.allclose(f, f2)
    _, t3, _ = F.doppler_spectrogram(s, 100.0, nperseg=64, noverlap=0)
    assert len(t3) < len(t)


def test_ratio_rotation_exact_turns():
    theta = 1.0 + np.linspace(0, 4 * np.pi, 40, endpoint=False)
    a = 3.0 + 0.5 * np.exp(1j * theta)
    h = np.ones((40, 2, 3), dtype=complex)
    h[:, 0, :] = a[:, None]
    net, total = F.ratio_rotation(h)
    assert net == pytest.approx(1.95, abs=1e-6)
    assert total == pytest.approx(1.95, abs=1e-6)
    net_rev, _ = F.ratio_rotation(h[::-1])
    assert net_rev == pytest.approx(-1.95, abs=1e-6)


def test_window_features_definition():
    _, H = synthetic_csi("push", seed=4)
    f = F.window_features(H, None, 100.0)
    amp = F.hampel(F.amplitude(H).reshape(len(H), -1), k=3)
    assert f["energy"] == pytest.approx(F.motion_energy(amp))
    pc = F.first_pc(F.normalize_amplitude(amp))
    fr, _, S = F.doppler_spectrogram(pc, 100.0)
    prof = S.mean(axis=1)
    prof[0] = 0.0
    assert f["doppler_peak_hz"] == pytest.approx(fr[int(np.argmax(prof))])
    assert f["doppler_peak_hz"] > 0
    st = np.convolve(np.abs(np.diff(pc)), np.ones(5) / 5, mode="same")
    assert f["burst_fraction"] == pytest.approx(float((st > 0.5 * st.max()).mean()))


def test_window_features_static_channel():
    H = np.ones((50, 2, 8), dtype=complex)
    f = F.window_features(H, None, 50.0)
    assert f["energy"] == 0.0 and f["doppler_peak_hz"] == 0.0 and f["burst_fraction"] == 0.0


# --------------------------------------------------------------------- face
def test_face_defaults():
    s = SpeakingScorer()
    assert (s.window, s.var_gain) == (15, 6.0)
    d = ActiveSpeakerDetector()
    assert (d.window, d.min_corr, d.speaking_threshold) == (45, 0.2, 0.25)
    g = SpeechGate()
    assert (g.allow_without_face, g.min_fraction, g._log.maxlen) == (True, 0.4, 600)
    b = BrowRaiseToggle()
    assert (b.threshold, b.hold, b.cooldown) == (0.55, 0.6, 1.0)
    assert b.release == pytest.approx(0.33) and b._last_fire == -1e9
    t = FaceTracker()
    assert (t.max_dist, t.max_missing) == (0.15, 15)
    assert Track(0, 0.0, 0.0, 0.1).missing == 0


def test_mouth_openness_clips_out_of_range_input():
    assert mouth_openness({"jawOpen": 5.0}) == 1.0


def test_asd_backfill_is_zero():
    d = ActiveSpeakerDetector(window=10)
    for _ in range(3):
        d.update({1: 0.5}, 0.1)
    d.update({1: 0.5, 2: 0.9}, 0.1)
    assert list(d._faces[2]) == [0.0, 0.0, 0.0, 0.9]


class Const:
    def __init__(self, v):
        self.v = v

    def scores(self, face_series, audio_energy):
        return dict.fromkeys(face_series, self.v)


def test_asd_min_correlation_inclusive():
    d = ActiveSpeakerDetector(model=Const(0.2), min_correlation=0.2)
    d.update({1: 0.0, 2: 0.0}, 0.0)
    assert d.active() in (1, 2)


def test_gate_window_inclusive_end_and_counting():
    g = SpeechGate(min_fraction=0.1)
    g.enroll({1: 1.0})
    g.observe(1.0, 1, 1)
    assert g.allow(0.0, 1.0) is True
    g2 = SpeechGate(min_fraction=0.1)
    g2.enroll({1: 1.0})
    for t, a in ((0, 1), (1, None), (2, None), (3, None)):
        g2.observe(t, a, 1)
    assert g2.allow(0, 3) is True  # silence frames are not "someone else"
    g3 = SpeechGate(min_fraction=0.1)
    g3.enroll({1: 1.0})
    for t, a in ((0, 1), (1, 2), (2, 2)):
        g3.observe(t, a, 2)
    assert g3.allow(0, 2) is False
    g4 = SpeechGate(min_fraction=0.3)
    g4.enroll({1: 1.0})
    for t, a in ((0, 1), (1, None), (2, None), (3, None)):
        g4.observe(t, a, 1)
    assert g4.allow(0, 3) is False  # 1/4 < 0.3


def test_brow_release_boundary_and_returns():
    b = BrowRaiseToggle(threshold=0.5, hold=0.0, cooldown=0.0)
    assert b.update(0.0, {"browInnerUp": 0.9}) is True
    assert b.update(0.1, {"browInnerUp": b.release}) is False  # equal to release: not re-armed
    assert b.update(0.2, {"browInnerUp": 0.9}) is False
    assert b.update(0.3, {"browInnerUp": 0.0}) is False
    assert b.update(0.4, {"browInnerUp": 0.5}) is True  # exactly threshold counts as raised
    assert b.update(0.5, {"browInnerUp": 0.9}) is False  # fired once; needs re-arming


# ------------------------------------------------------------------- fusion
def test_fusion_defaults_and_bool_accept():
    f = GestureFusion()
    assert (f.w, f.threshold, f.csi_max_age, f._probs.maxlen) == (0.25, 0.6, 1.5, 8)
    f.update_csi("empty", 1.0, 0.0)
    assert f.accept(GestureEvent("click", 0.0)) is True  # camera dominates: 0.75 * 1.0 + 0.25 * 0.0
    f.update_model({"pinch_click": 0.5})
    assert f.accept(GestureEvent("click", 0.0)) is False  # 0.375
    assert f.accept(GestureEvent("move", 0.0)) is True


def test_bus_default_history():
    bus = EventBus()
    for i in range(300):
        bus.publish("t", {"i": i})
    assert len(bus.history()) == 256
