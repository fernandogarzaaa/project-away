"""Silero VAD v6.2 (MIT) run directly with onnxruntime (v0.1.24 model refresh; ``models_next.vad: silero6``).

The model is the catalog's ``vad-silero-v6.2`` (``silero_vad.onnx`` pinned to the v6.2 tag, sha256-checked) in
``<data>/models/vad-silero-v6.2/``. No torch and no ``silero-vad`` package are needed: the ONNX graph takes
512-sample windows at 16 kHz (256 at 8 kHz) with 64 (32) samples of context from the previous window and a
``[2, 1, 128]`` recurrent state, and returns the speech probability of the window.

:class:`SileroV6VAD` has the same ``segments`` / ``trim`` / ``is_speech`` interface as the other detectors in
:mod:`away.voice.vad`, plus :meth:`SileroV6VAD.stream` for a chunk-by-chunk probability.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.voice.vad import FloatArray, Segment

MODEL_ID = "vad-silero-v6.2"
MODEL_FILE = "silero_vad.onnx"


def model_path(data_dir: Path | None = None) -> Path:
    from away.app.model_fetch import model_dir

    return model_dir(MODEL_ID, data_dir) / MODEL_FILE


class SileroV6VAD:
    """``session``: an onnxruntime session (or anything with ``run(None, feeds)``) - injectable for tests."""

    backend = "silero-v6.2-onnx"

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        session: Any = None,
        path: Path | None = None,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD runs at 8000 or 16000 Hz")
        self.sr = sample_rate
        self.threshold = float(threshold)
        self.neg_threshold = max(0.01, self.threshold - 0.15)
        self.window = 512 if sample_rate == 16000 else 256
        self.context_size = 64 if sample_rate == 16000 else 32
        self.min_speech = int(sample_rate * min_speech_ms / 1000)
        self.min_silence = int(sample_rate * min_silence_ms / 1000)
        if session is None:
            import onnxruntime as ort

            p = path or model_path()
            if not Path(p).is_file():
                raise FileNotFoundError("Silero VAD v6.2 is not downloaded yet (Settings > Models)")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"])
        self.session = session
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)
        self._rest = np.zeros(0, dtype=np.float32)

    def window_prob(self, win: NDArray[np.float32]) -> float:
        """Speech probability of exactly one window (keeps the state and context)."""
        x = np.asarray(win, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.window:
            raise ValueError(f"a window is {self.window} samples")
        inp = np.concatenate([self._context, x], axis=1)
        out, state = self.session.run(None, {"input": inp, "state": self._state, "sr": np.array(self.sr, np.int64)})
        self._state = np.asarray(state, dtype=np.float32)
        self._context = inp[:, -self.context_size :]
        return float(np.asarray(out).reshape(-1)[0])

    def stream(self, chunk: FloatArray) -> list[float]:
        """Probabilities of the whole windows in ``chunk`` (any length; the remainder waits for the next call)."""
        buf = np.concatenate([self._rest, np.asarray(chunk, dtype=np.float32).reshape(-1)])
        n = len(buf) // self.window
        probs = [self.window_prob(buf[i * self.window : (i + 1) * self.window]) for i in range(n)]
        self._rest = buf[n * self.window :]
        return probs

    def probabilities(self, audio: FloatArray) -> list[float]:
        self.reset()
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        pad = (-len(a)) % self.window
        if pad:
            a = np.concatenate([a, np.zeros(pad, dtype=np.float32)])
        probs = self.stream(a)
        self.reset()
        return probs

    def segments(self, audio: FloatArray) -> list[Segment]:
        return segments_from_probs(
            self.probabilities(audio),
            self.window,
            self.threshold,
            self.neg_threshold,
            self.min_speech,
            self.min_silence,
            len(np.asarray(audio).reshape(-1)),
        )

    def trim(self, audio: FloatArray, pad_ms: int = 150) -> FloatArray:
        segs = self.segments(audio)
        if not segs:
            return audio[:0]
        pad = int(self.sr * pad_ms / 1000)
        return audio[max(0, segs[0].start - pad) : min(len(audio), segs[-1].end + pad)]

    def is_speech(self, audio: FloatArray) -> bool:
        return bool(self.segments(audio))


def segments_from_probs(
    probs: list[float],
    window: int,
    threshold: float,
    neg_threshold: float,
    min_speech: int,
    min_silence: int,
    total: int,
) -> list[Segment]:
    """Hysteresis over window probabilities (Silero's own rule): speech starts above ``threshold`` and ends after
    ``min_silence`` samples below ``neg_threshold``; segments shorter than ``min_speech`` are dropped."""
    segs: list[Segment] = []
    start: int | None = None
    silence_at: int | None = None
    for i, p in enumerate(probs):
        pos = i * window
        if p >= threshold:
            silence_at = None
            if start is None:
                start = pos
        elif start is not None and p < neg_threshold:
            if silence_at is None:
                silence_at = pos
            if pos + window - silence_at >= min_silence:
                if silence_at - start >= min_speech:
                    segs.append(Segment(start, silence_at))
                start, silence_at = None, None
    if start is not None:
        end = min(total, len(probs) * window)
        if end - start >= min_speech:
            segs.append(Segment(start, end))
    return segs


def make_silero6(
    sample_rate: int = 16000, data_dir: Path | None = None, fallback: Callable[[], Any] | None = None
) -> Any:
    """Silero v6.2 when it is downloaded; otherwise ``fallback()`` (the v0.1.23 detector)."""
    try:
        return SileroV6VAD(sample_rate, path=model_path(data_dir))
    except Exception:
        if fallback is None:
            raise
        return fallback()
