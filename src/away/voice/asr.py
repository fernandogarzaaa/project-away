"""Speech recognition backends behind one small interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

WHISPER_MODELS = ("large-v3-turbo", "large-v3", "medium", "small", "base", "tiny", "distil-large-v3")


class ASRBackend(Protocol):
    name: str

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str: ...


class FakeASR:
    """Deterministic backend for tests / dry runs: returns queued strings."""

    name = "fake"

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.calls: list[tuple[int, str | None]] = []

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        self.calls.append((len(audio), prompt))
        return self.outputs.pop(0) if self.outputs else ""


class FasterWhisperASR:  # pragma: no cover - downloads a model
    name = "faster-whisper"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
        download_root: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        if compute_type == "auto":
            compute_type = "default"
        self.model = WhisperModel(model, device=device, compute_type=compute_type, download_root=download_root)
        self.language = language

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        segments, _info = self.model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.language,
            initial_prompt=prompt,
            vad_filter=False,
            beam_size=5,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class ParakeetASR:  # pragma: no cover - optional NVIDIA NeMo backend
    """NVIDIA Parakeet (e.g. ``nvidia/parakeet-tdt-0.6b-v2``) via NeMo. English only; GPU recommended."""

    name = "parakeet"

    def __init__(self, model: str = "nvidia/parakeet-tdt-0.6b-v2") -> None:
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model)

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        out = self.model.transcribe([np.asarray(audio, dtype=np.float32)])
        first = out[0]
        return str(getattr(first, "text", first)).strip()


def make_asr(
    backend: str,
    model: str,
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    download_root: str | None = None,
) -> ASRBackend:
    if backend == "parakeet":
        return ParakeetASR() if model in WHISPER_MODELS else ParakeetASR(model)
    return FasterWhisperASR(model, device, compute_type, language, download_root)
