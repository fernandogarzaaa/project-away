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


def test_frame_rms_drops_partial_tail_and_rejects_bad_frame() -> None:
    import pytest

    audio = np.array([1.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float32)
    rms = frame_rms(audio, frame=2)
    assert len(rms) == 2
    assert np.allclose(rms, [1.0, 1.0])  # the one-sample tail is intentionally dropped
    with pytest.raises(ValueError, match="frame must be positive"):
        frame_rms(audio, frame=0)


def test_frame_flags_applies_hangover_after_speech() -> None:
    class FixedVAD(EnergyVAD):
        def __init__(self) -> None:
            super().__init__(sample_rate=1000, frame_ms=10, min_rms=0.1, ratio=1.0, hangover_frames=2)

        def frame_flags(self, audio: np.ndarray) -> np.ndarray:
            return super().frame_flags(audio)

    vad = FixedVAD()
    # Five 10ms frames: speech in the first frame, then silence. Hangover extends two frames.
    audio = np.concatenate([np.ones(10, dtype=np.float32), np.zeros(40, dtype=np.float32)])
    flags = vad.frame_flags(audio)
    assert flags[0]
    assert flags[1] and flags[2]
    assert not flags[3]


def test_trim_padding_is_bounded_to_input() -> None:
    vad = EnergyVAD(SR, frame_ms=10, min_rms=0.01, ratio=1.0, hangover_frames=0, min_speech_frames=1)
    audio = np.concatenate([_tone(0.2), _silence(0.1)])
    trimmed = vad.trim(audio, pad_ms=1000)
    assert 0 < len(trimmed) <= len(audio)


def test_make_vad_silero_falls_back_when_backend_unavailable(monkeypatch) -> None:
    from away.voice import vad

    class BrokenSilero:
        def __init__(self, sample_rate: int) -> None:
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr(vad, "SileroVAD", BrokenSilero)
    assert isinstance(vad.make_vad("silero", SR), EnergyVAD)


def test_silero_vad_segments_trim_and_is_speech_without_loading_model() -> None:
    from away.voice.vad import SileroVAD

    vad = object.__new__(SileroVAD)
    vad.sr = SR
    vad.threshold = 0.5
    vad._fn = lambda _audio: [{"start": 100, "end": 300}, {"start": 500, "end": 700}]
    audio = np.arange(1000, dtype=np.float32)
    segments = vad.segments(audio)
    assert [(s.start, s.end) for s in segments] == [(100, 300), (500, 700)]
    trimmed = vad.trim(audio, pad_ms=10)
    assert len(trimmed) == 860  # 160 samples of padding, clipped at the start of the input
    assert vad.is_speech(audio) is True


def test_silero_vad_empty_segments_return_empty_audio() -> None:
    from away.voice.vad import SileroVAD

    vad = object.__new__(SileroVAD)
    vad.sr = SR
    vad.threshold = 0.5
    vad._fn = lambda _audio: []
    audio = np.arange(100, dtype=np.float32)
    assert vad.trim(audio).size == 0
    assert vad.is_speech(audio) is False


def test_make_vad_silero_success_and_silero6_dispatch(monkeypatch) -> None:
    from away.voice import vad, vad_next

    class FakeSilero:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate

    monkeypatch.setattr(vad, "SileroVAD", FakeSilero)
    made = vad.make_vad("silero", SR)
    assert isinstance(made, FakeSilero) and made.sample_rate == SR

    sentinel = object()
    monkeypatch.setattr(vad_next, "make_silero6", lambda sample_rate, fallback: sentinel)
    assert vad.make_vad("silero6", SR) is sentinel
