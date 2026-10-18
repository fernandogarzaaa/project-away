"""Mutation-strengthening tests for voice/vad.py (v0.1.29)."""

from __future__ import annotations

import numpy as np

from away.voice.vad import EnergyVAD, Segment, frame_rms, make_vad

SR = 16000


def _tone(seconds: float = 0.5, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float = 0.5) -> np.ndarray:
    return (np.random.default_rng(0).normal(0, 0.001, int(SR * seconds))).astype(np.float32)


def test_trim_empty_when_silence() -> None:
    vad = EnergyVAD(SR)
    out = vad.trim(_silence(1.0))
    assert len(out) == 0


def test_trim_keeps_speech_with_pad() -> None:
    vad = EnergyVAD(SR)
    audio = _utterance()
    out = vad.trim(audio, pad_ms=100)
    # Should include the tone plus padding, but not all the silence
    assert len(out) < len(audio)
    assert len(out) > int(SR * 0.5)


def _utterance() -> np.ndarray:
    # silence + tone + silence (VAD needs silence for noise floor)
    return np.concatenate([_silence(0.3), _tone(0.5), _silence(0.3)])


def test_is_speech_true_false() -> None:
    vad = EnergyVAD(SR)
    assert vad.is_speech(_utterance()) is True
    assert vad.is_speech(_silence(0.5)) is False


def test_segment_fields() -> None:
    vad = EnergyVAD(SR)
    segs = vad.segments(_utterance())
    assert len(segs) == 1
    assert isinstance(segs[0], Segment)
    assert segs[0].start >= 0
    assert segs[0].end > segs[0].start


def test_make_vad_energy() -> None:
    vad = make_vad("energy", SR)
    assert isinstance(vad, EnergyVAD)


def test_make_vad_unknown_defaults_to_energy() -> None:
    vad = make_vad("nonexistent", SR)
    assert isinstance(vad, EnergyVAD)


def test_frame_rms_silence() -> None:
    rms = frame_rms(_silence(0.1), frame=160)
    assert np.all(rms < 0.01)


def test_frame_rms_tone() -> None:
    rms = frame_rms(_tone(0.1), frame=160)
    assert np.all(rms > 0.1)
