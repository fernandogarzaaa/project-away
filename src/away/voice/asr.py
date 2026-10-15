"""Speech recognition backends behind one small interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from away.voice.hallucination import join_segments

WHISPER_MODELS = ("large-v3-turbo", "large-v3", "medium", "small", "base", "tiny", "distil-large-v3")


class ASRBackend(Protocol):
    name: str

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str: ...


class FakeASR:
    """Deterministic backend for tests / dry runs: returns queued strings."""

    name = "fake"

    def __init__(self, outputs: list[str] | None = None, confidences: list[float | None] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.confidences = list(confidences or [])
        self.calls: list[tuple[int, str | None]] = []
        self.last_confidence: float | None = None

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        self.calls.append((len(audio), prompt))
        self.last_confidence = self.confidences.pop(0) if self.confidences else None
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
        cpu_threads: int = 0,
        beam_size: int = 5,
    ) -> None:
        from faster_whisper import WhisperModel

        if compute_type == "auto":
            compute_type = "default"
        self.model = WhisperModel(
            resolve_whisper_model(model, download_root),
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            cpu_threads=max(0, int(cpu_threads)),
        )
        self.device = device
        self.compute_type = compute_type
        self.language = language
        from away.voice.language import LanguageMemory

        self.languages = LanguageMemory()  # v0.1.27: a safer Auto (falls back to the language you use most)
        self.hotwords = hotwords
        self.beam_size = max(1, int(beam_size))
        self.last_confidence: float | None = None

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        stats: dict[str, object] = {}
        text = whisper_transcribe(
            self.model, audio, self.language, prompt, self.hotwords, self.beam_size, stats, self.languages
        )
        conf = stats.get("confidence")
        self.last_confidence = float(conf) if isinstance(conf, (int, float)) else None
        return text


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
    model: object,
    audio: NDArray[np.float32],
    language: str | None,
    prompt: str | None,
    hotwords: str | None,
    beam_size: int = 5,
    stats: dict[str, object] | None = None,
    memory: Any = None,
) -> str:
    """Call ``WhisperModel.transcribe`` with the vocabulary as ``initial_prompt`` and, when the installed
    faster-whisper has it (1.1+; verified on 1.2.1), also as ``hotwords``. Silent-segment hallucinations are
    removed (:func:`away.voice.hallucination.segment_ok`).

    v0.1.27: with no language set (Auto) and a ``memory`` (:class:`away.voice.language.LanguageMemory`), an unsure
    guess or a language you have not been speaking is transcribed again in your most-used language."""
    from away.voice.language import normalize

    language = normalize(language)
    fn = model.transcribe  # type: ignore[attr-defined]
    kw: dict[str, object] = {
        "language": language,
        "initial_prompt": prompt,
        "vad_filter": False,
        "beam_size": beam_size,
        "condition_on_previous_text": False,
    }
    if hotwords and supports_kwarg(fn, "hotwords"):
        kw["hotwords"] = hotwords
    segments, info = fn(np.asarray(audio, dtype=np.float32), **kw)
    segs = list(segments)  # a generator: the text and the confidence both need it
    used = language or str(getattr(info, "language", "") or "")
    if language is None and memory is not None and hasattr(memory, "fallback"):
        fb = memory.fallback(getattr(info, "language", None), getattr(info, "language_probability", None))
        if fb:
            kw["language"] = fb
            segments, info = fn(np.asarray(audio, dtype=np.float32), **kw)
            segs = list(segments)
            used = fb
            memory.note(fb)
    if stats is not None:
        stats["language"] = used
    if stats is not None:
        from away.voice.hallucination import segment_ok
        from away.voice.weak_speech import confidence_of

        stats["confidence"] = confidence_of([s for s in segs if segment_ok(s)])  # v0.1.21 (weak-evidence filter)
    # segments Whisper produced over silence (high no_speech_prob AND low avg_logprob) are dropped
    return join_segments(segs)


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
    cpu_threads: int = 0,
    beam_size: int = 5,
    on_fallback: Callable[[str], object] | None = None,
    factory: Callable[..., ASRBackend] | None = None,
    cpu_model: str | None = None,
    cpu_download_root: str | None = None,
    cpu_beam_size: int | None = None,
) -> ASRBackend:
    """Build the ASR backend. On ``device="cuda"`` the GPU libraries are activated first; if CUDA cannot
    load or run (no cuBLAS, old driver, out of VRAM) it falls back to the CPU (int8) and calls
    ``on_fallback(reason)`` so the app can tell the user why.

    ``cpu_model``: the model to use on that CPU fallback (the hardware profile's CPU pick). A GPU-sized
    model such as large-v3-turbo cannot keep up on most CPUs, so falling back to it would just drop
    speech. A CUDA error that only shows up later, on a real transcription, switches the same way."""
    if backend == "parakeet":
        try:
            import nemo.collections.asr  # noqa: F401
        except ImportError as exc:
            raise ImportError(PARAKEET_MISSING) from exc
        return ParakeetASR() if model in WHISPER_MODELS else ParakeetASR(model)
    build = factory or FasterWhisperASR

    def build_cpu() -> ASRBackend:
        m = cpu_model or model
        root = cpu_download_root if cpu_model and cpu_model != model else download_root
        beam = cpu_beam_size if cpu_beam_size else beam_size
        return build(m, "cpu", "int8", language, root, hotwords, cpu_threads, beam)

    if device == "cuda":
        from away.app import gpupack

        gpupack.activate()
        try:
            asr = build(model, "cuda", compute_type, language, download_root, hotwords, cpu_threads, beam_size)
            warm_up(asr)  # cuBLAS is loaded lazily: make it fail here, not on the first dictation
            return FallbackASR(asr, build_cpu, on_fallback, cpu_model or model)
        except Exception as exc:
            reason = cuda_failure_reason(exc, cpu_model if cpu_model and cpu_model != model else None)
            if on_fallback is not None:
                on_fallback(reason)
            return build_cpu()
    return build(model, device, compute_type, language, download_root, hotwords, cpu_threads, beam_size)


CUDA_ERROR_WORDS = ("cublas", "cudnn", "cuda", "cudart")


def is_cuda_error(exc: BaseException) -> bool:
    low = f"{type(exc).__name__}: {exc}".lower()
    return any(w in low for w in CUDA_ERROR_WORDS)


class FallbackASR:
    """A GPU backend that switches itself to the CPU (once) when CUDA fails during a transcription."""

    def __init__(
        self,
        gpu: ASRBackend,
        build_cpu: Callable[[], ASRBackend],
        on_fallback: Callable[[str], object] | None = None,
        cpu_model: str = "",
    ) -> None:
        self.gpu = gpu
        self.build_cpu = build_cpu
        self.on_fallback = on_fallback
        self.cpu_model = cpu_model
        self.cpu: ASRBackend | None = None
        self.name = getattr(gpu, "name", "faster-whisper")

    @property
    def device(self) -> str:
        return "cpu" if self.cpu is not None else "cuda"

    @property
    def last_confidence(self) -> float | None:
        backend = self.cpu if self.cpu is not None else self.gpu
        v = getattr(backend, "last_confidence", None)
        return float(v) if isinstance(v, (int, float)) else None

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        if self.cpu is not None:
            return self.cpu.transcribe(audio, prompt)
        try:
            return self.gpu.transcribe(audio, prompt)
        except Exception as exc:
            if not is_cuda_error(exc):
                raise
            self.cpu = self.build_cpu()
            if self.on_fallback is not None:
                self.on_fallback(cuda_failure_reason(exc, self.cpu_model or None))
            return self.cpu.transcribe(audio, prompt)


def warm_up(asr: ASRBackend) -> None:
    asr.transcribe(np.zeros(16000, dtype=np.float32))


def cuda_failure_reason(exc: BaseException, cpu_model: str | None = None) -> str:
    """One line for the user: what went wrong on the GPU and what fixes it."""
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    if "cublas" in low:
        fix = "the CUDA library cuBLAS 12 is missing: run `away gpu install` (or Settings > Performance)"
    elif "out of memory" in low:
        fix = "the GPU ran out of memory: pick a smaller model (voice.asr_model: small)"
    elif "driver" in low or "insufficient" in low:
        fix = "the NVIDIA driver is too old for CUDA 12: update it from nvidia.com"
    elif "no cuda" in low or "cuda device" in low or "not compiled" in low:
        fix = "no usable CUDA device"
    else:
        fix = "CUDA failed to start"
    cpu = f"the CPU with the smaller '{cpu_model}' model" if cpu_model else "the CPU"
    return f"GPU unavailable ({fix}); using {cpu} instead. Details: {msg[:300]}"
