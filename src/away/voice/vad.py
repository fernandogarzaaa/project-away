"""Voice activity detection.

:class:`EnergyVAD` is a dependency-free adaptive-threshold detector (used in
tests, as a fallback, and as the per-frame "mic energy" signal for active
speaker detection). :class:`SileroVAD` wraps the Silero model when the
[voice] extra is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32] | NDArray[np.float64]


def frame_rms(audio: FloatArray, frame: int) -> NDArray[np.float64]:
    """RMS per non-overlapping frame (the tail shorter than a frame is dropped)."""
    if frame <= 0:
        raise ValueError("frame must be positive")
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0)
    x = np.asarray(audio[: n * frame], dtype=np.float64).reshape(n, frame)
    out: NDArray[np.float64] = np.sqrt((x * x).mean(axis=1))
    return out


@dataclass
class Segment:
    start: int  # samples
    end: int


class EnergyVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        ratio: float = 3.0,
        min_rms: float = 0.005,
        hangover_frames: int = 8,
        min_speech_frames: int = 3,
    ) -> None:
        self.sr = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.ratio = ratio
        self.min_rms = min_rms
        self.hangover = hangover_frames
        self.min_speech = min_speech_frames

    def frame_flags(self, audio: FloatArray) -> NDArray[np.bool_]:
        rms = frame_rms(audio, self.frame)
        if len(rms) == 0:
            return np.zeros(0, dtype=bool)
        noise = float(np.percentile(rms, 10))
        thr = max(self.min_rms, noise * self.ratio)
        raw = rms > thr
        flags = raw.copy()
        hang = 0
        for i, v in enumerate(raw):  # hangover keeps word endings
            if v:
                hang = self.hangover
            elif hang > 0:
                flags[i] = True
                hang -= 1
        return flags

    def segments(self, audio: FloatArray) -> list[Segment]:
        flags = self.frame_flags(audio)
        segs: list[Segment] = []
        start = None
        for i, v in enumerate([*flags.tolist(), False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= self.min_speech:
                    segs.append(Segment(start * self.frame, i * self.frame))
                start = None
        return segs

    def trim(self, audio: FloatArray, pad_ms: int = 150) -> FloatArray:
        """Cut leading/trailing silence; returns an empty array when nothing was said."""
        segs = self.segments(audio)
        if not segs:
            return audio[:0]
        pad = int(self.sr * pad_ms / 1000)
        return audio[max(0, segs[0].start - pad) : min(len(audio), segs[-1].end + pad)]

    def is_speech(self, audio: FloatArray) -> bool:
        return bool(self.segments(audio))


class SileroVAD:  # pragma: no cover - needs the silero-vad package
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5) -> None:
        from silero_vad import get_speech_timestamps, load_silero_vad

        self.model = load_silero_vad(onnx=True)
        self._ts = get_speech_timestamps
        self.sr = sample_rate
        self.threshold = threshold

    def segments(self, audio: FloatArray) -> list[Segment]:
        import torch

        ts = self._ts(
            torch.from_numpy(np.asarray(audio, dtype=np.float32)),
            self.model,
            sampling_rate=self.sr,
            threshold=self.threshold,
        )
        return [Segment(int(t["start"]), int(t["end"])) for t in ts]

    def trim(self, audio: FloatArray, pad_ms: int = 150) -> FloatArray:
        segs = self.segments(audio)
        if not segs:
            return audio[:0]
        pad = int(self.sr * pad_ms / 1000)
        return audio[max(0, segs[0].start - pad) : min(len(audio), segs[-1].end + pad)]

    def is_speech(self, audio: FloatArray) -> bool:
        return bool(self.segments(audio))


def make_vad(kind: str, sample_rate: int = 16000) -> EnergyVAD | SileroVAD:
    if kind == "silero":
        try:
            return SileroVAD(sample_rate)
        except Exception:
            return EnergyVAD(sample_rate)
    return EnergyVAD(sample_rate)
