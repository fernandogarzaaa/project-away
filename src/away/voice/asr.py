"""Speech recognition backends behind one small interface."""

from __future__ import annotations

from pathlib import Path
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
        hotwords: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        if compute_type == "auto":
            compute_type = "default"
        self.model = WhisperModel(
            resolve_whisper_model(model, download_root),
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        self.language = language
        self.hotwords = hotwords

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        return whisper_transcribe(self.model, audio, self.language, prompt, self.hotwords)


def resolve_whisper_model(model: str, download_root: str | None) -> str:
    """``away setup`` downloads the model flat into ``download_root`` (``model.bin`` + tokenizer). Load that
    directory directly; passing the size name with ``download_root`` would make faster-whisper fetch a second
    copy into a Hugging Face cache layout on first dictation."""
    if download_root and (Path(download_root) / "model.bin").exists():
        return str(download_root)
    return model


def supports_kwarg(fn: object, name: str) -> bool:
    import inspect

    try:
        return name in inspect.signature(fn).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def whisper_transcribe(
    model: object, audio: NDArray[np.float32], language: str | None, prompt: str | None, hotwords: str | None
) -> str:
    """Call ``WhisperModel.transcribe`` with the vocabulary as ``initial_prompt`` and, when the installed
    faster-whisper has it (1.1+; verified on 1.2.1), also as ``hotwords``."""
    fn = model.transcribe  # type: ignore[attr-defined]
    kw: dict[str, object] = {
        "language": language,
        "initial_prompt": prompt,
        "vad_filter": False,
        "beam_size": 5,
        "condition_on_previous_text": False,
    }
    if hotwords and supports_kwarg(fn, "hotwords"):
        kw["hotwords"] = hotwords
    segments, _info = fn(np.asarray(audio, dtype=np.float32), **kw)
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


PARAKEET_MISSING = (
    "voice.asr_backend 'parakeet' needs NVIDIA NeMo, which the one-click installers do not bundle "
    "(multi-GB, Linux/NVIDIA-oriented). Install from Python instead: "
    "pipx install 'project-away[all,parakeet]', or set voice.asr_backend: faster-whisper."
)


def make_asr(
    backend: str,
    model: str,
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    download_root: str | None = None,
    hotwords: str | None = None,
) -> ASRBackend:
    if backend == "parakeet":
        try:
            import nemo.collections.asr  # noqa: F401
        except ImportError as exc:
            raise ImportError(PARAKEET_MISSING) from exc
        return ParakeetASR() if model in WHISPER_MODELS else ParakeetASR(model)
    return FasterWhisperASR(model, device, compute_type, language, download_root, hotwords)
