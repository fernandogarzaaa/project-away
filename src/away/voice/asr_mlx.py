"""Whisper on the Apple Silicon GPU through MLX (``mlx-whisper``, Metal), v0.1.35.

faster-whisper's engine, CTranslate2, has no Metal backend, so up to v0.1.34 a Mac ran Whisper on the CPU (int8).
On an Apple Silicon Mac with ``mlx-whisper`` installed (the ``[mlx]`` extra; bundled in the macOS app) and
``voice.asr_backend: auto`` (the default), :func:`away.voice.asr.make_asr` builds :class:`MLXWhisperASR` instead.
It answers the same ``transcribe(audio, prompt) -> str`` (plus ``last_confidence``) as
:class:`~away.voice.asr.FasterWhisperASR`, and goes through the same :func:`~away.voice.asr.whisper_transcribe`
(vocabulary as ``initial_prompt``, silent-segment and prompt-echo filtering, the v0.1.27 safer Auto language), via
:class:`MLXModelAdapter`, which makes ``mlx_whisper.transcribe`` look like ``WhisperModel.transcribe``.

Differences from faster-whisper: MLX decodes greedily (mlx-whisper has no beam search; ``beam_size`` is ignored,
the temperature fallback is the same) and runs in float16. ``hotwords`` does not exist there either; the
vocabulary still goes in through ``initial_prompt``.

Any failure (package missing, no Metal device, weights missing or corrupt, an error during a transcription) falls
back to faster-whisper on the CPU and says why: dictation never breaks because of this backend.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import logging
import platform as _platform
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

MLX_BACKEND = "mlx-whisper"
ASR_BACKENDS = ("auto", "faster-whisper", MLX_BACKEND, "parakeet")
MLX_MISSING = (
    "voice.asr_backend 'mlx-whisper' needs the mlx-whisper package (Apple Silicon Macs only): "
    "pipx install 'project-away[all,mlx]'"
)


def apple_silicon(platform: str | None = None, machine: str | None = None) -> bool:
    """macOS on an arm64 (M-series) CPU: the only place MLX runs. With no arguments, the app's (cached) hardware
    detection answers, so tests that pin the hardware (tests/conftest.py) never start the real MLX."""
    if platform is None and machine is None:
        try:
            from away.app import hardware

            return bool(hardware.cached().apple_silicon)
        except Exception:
            pass
    plat = sys.platform if platform is None else platform
    mach = _platform.machine() if machine is None else machine
    return plat == "darwin" and mach.lower() in ("arm64", "aarch64")


def mlx_installed() -> bool:
    """``mlx_whisper`` and ``mlx.core`` can be found (cheap: nothing is imported, no GPU is touched)."""
    try:
        return importlib.util.find_spec("mlx_whisper") is not None and importlib.util.find_spec("mlx.core") is not None
    except (ImportError, ValueError):  # a broken / half-installed namespace package
        return False


def resolve_backend(
    backend: str,
    platform: str | None = None,
    machine: str | None = None,
    installed: bool | None = None,
) -> str:
    """The engine ``voice.asr_backend`` means on this machine.

    ``auto`` -> ``mlx-whisper`` on Apple Silicon with mlx-whisper installed, else ``faster-whisper``. An explicit
    value is kept (``mlx-whisper`` on a machine without MLX then falls back inside :func:`away.voice.asr.make_asr`,
    with the reason)."""
    if backend not in ("", "auto"):
        return backend
    if apple_silicon(platform, machine) and (mlx_installed() if installed is None else installed):
        return MLX_BACKEND
    return "faster-whisper"


def setup_backend(backend: str, model: str, platform: str | None = None, machine: str | None = None) -> str:
    """Whose weights first-run setup fetches for ``model``: ``mlx-whisper`` (``whisper-mlx-<model>``) when the
    app will really run it (Apple Silicon, mlx-whisper importable, an MLX build of the model exists), else
    ``faster-whisper`` (``whisper-<model>``), or ``parakeet`` (NeMo downloads its own)."""
    from away.app.models import MLX_MODELS

    chosen = resolve_backend(backend, platform, machine)
    if chosen == "parakeet":
        return chosen
    if chosen == MLX_BACKEND and apple_silicon(platform, machine) and mlx_installed() and model in MLX_MODELS:
        return MLX_BACKEND
    return "faster-whisper"


def failure_reason(exc: BaseException | None, cpu_model: str | None = None) -> str:
    """One line for the user and the log: why the Apple GPU is not used and what runs instead."""
    cpu = f"Whisper '{cpu_model}' on the CPU" if cpu_model else "Whisper on the CPU"
    if exc is None or isinstance(exc, ImportError):
        why = "mlx-whisper is not installed" + (f" ({exc})" if exc is not None else "")
    elif isinstance(exc, KeyError):
        why = f"no MLX build of the model {exc}"
    else:
        why = f"MLX failed ({type(exc).__name__}: {str(exc)[:300]})"
    return f"Apple GPU speech recognition unavailable: {why}; using {cpu} instead."


def quiet_tqdm() -> None:
    """``mlx_whisper.transcribe`` opens a (disabled) tqdm bar on every call, and tqdm's first bar creates a
    multiprocessing lock, which starts multiprocessing's resource tracker: in the frozen app that is the app
    executable again, started with Python's ``-B -s -S -c ...`` arguments (found with the v0.1.35 bundle test).
    A plain thread lock is all one process needs."""
    try:
        import threading

        import tqdm

        if getattr(tqdm.tqdm, "_lock", None) is None:
            tqdm.tqdm.set_lock(threading.RLock())
    except Exception as exc:  # tqdm missing or changed: only the bundle's extra process is at stake
        log.debug("tqdm lock not replaced: %s", exc)


def ensure_numba() -> None:
    """``mlx_whisper.timing`` imports numba at import time, only for word timestamps (never used here). The macOS
    app leaves numba + llvmlite (~150 MB) out of the bundle, so stand in a no-op ``numba.jit`` when it is missing."""
    try:
        import numba  # noqa: F401
    except Exception:
        stub = types.ModuleType("numba")

        def jit(*args: Any, **kwargs: Any) -> Any:
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]
            return lambda fn: fn

        stub.jit = jit  # type: ignore[attr-defined]
        sys.modules["numba"] = stub


class MLXModelAdapter:
    """``mlx_whisper`` dressed as faster-whisper's ``WhisperModel``: ``transcribe(audio, **kw)`` returns
    ``(segments, info)`` with attribute access (``text``, ``avg_logprob``, ``no_speech_prob``; ``info.language``,
    ``info.language_probability``), so :func:`away.voice.asr.whisper_transcribe` works on it unchanged."""

    def __init__(self, path: str, api: Any = None) -> None:
        self.path = path
        self._api = api  # tests pass a fake; the real one is imported lazily

    @property
    def api(self) -> Any:
        if self._api is None:
            ensure_numba()
            quiet_tqdm()
            import mlx_whisper

            self._api = mlx_whisper
        return self._api

    def load(self) -> Any:
        """Load the weights onto the GPU (mlx_whisper keeps the last model loaded in ``ModelHolder``)."""
        import importlib

        self.api  # noqa: B018 - imports mlx_whisper (with the numba stand-in when needed)
        import mlx.core as mx

        # the package's ``transcribe`` attribute is the function; the module holds the cache
        holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
        return holder.get_model(self.path, mx.float16)

    def detect_language(self, audio: NDArray[np.float32]) -> tuple[str | None, float | None]:
        """Detect the language like faster-whisper does (first 30 s), so the caller gets its probability (the
        v0.1.27 safer Auto needs it; ``mlx_whisper.transcribe`` detects but does not report it)."""
        model = self.load()
        if model is None or not getattr(model, "is_multilingual", True):
            return None, None
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim

        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
        seg = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(seg)
        if not probs:
            return None, None
        lang = max(probs, key=probs.get)
        return str(lang), float(probs[lang])

    def transcribe(
        self,
        audio: NDArray[np.float32],
        language: str | None = None,
        initial_prompt: str | None = None,
        condition_on_previous_text: bool = False,
        **_ignored: Any,  # beam_size (no beam search in mlx-whisper), vad_filter (Away runs its own VAD)
    ) -> tuple[list[types.SimpleNamespace], types.SimpleNamespace]:
        audio = np.asarray(audio, dtype=np.float32)
        prob: float | None = None
        if language is None:
            try:
                language, prob = self.detect_language(audio)
            except Exception as exc:  # leave it to mlx_whisper's own detection
                log.debug("MLX language detection failed: %s", exc)
        result = self.api.transcribe(
            audio,
            path_or_hf_repo=self.path,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            verbose=None,
            fp16=True,
        )
        segs = [
            types.SimpleNamespace(
                text=str(s.get("text", "")),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                avg_logprob=s.get("avg_logprob"),
                no_speech_prob=s.get("no_speech_prob"),
                compression_ratio=s.get("compression_ratio"),
            )
            for s in result.get("segments", [])
        ]
        lang = result.get("language") or language
        info = types.SimpleNamespace(language=lang, language_probability=prob if lang == language else None)
        return segs, info


class MLXWhisperASR:  # the real model is exercised by the v0.1.35 benchmark, the logic by fakes
    """Whisper on the Apple GPU (MLX). Every MLX call runs on one worker thread of its own: MLX's GPU streams
    are per thread in some releases, and dictation, the edit hotkey and partials call in from different threads."""

    name = MLX_BACKEND
    device = "metal"
    compute_type = "float16"

    def __init__(
        self,
        path: str,
        language: str | None = None,
        beam_size: int = 5,
        adapter: Any = None,
        warm: bool = True,
    ) -> None:
        from away.voice.language import LanguageMemory

        self.path = path
        self.model = adapter or MLXModelAdapter(path)
        self.language = language
        self.languages = LanguageMemory()
        self.beam_size = max(1, int(beam_size))  # kept for the interface; MLX decodes greedily
        self.last_confidence: float | None = None
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="away-mlx")
        if warm:
            # load the weights and compile the Metal kernels now, so a broken MLX fails here (and falls back to
            # the CPU) instead of on the first dictation
            self._run(self._warm)

    def _run(self, fn: Any, *args: Any) -> Any:
        return self._pool.submit(fn, *args).result()

    def _warm(self) -> None:
        load = getattr(self.model, "load", None)
        if callable(load):
            load()
        self.model.transcribe(np.zeros(16000, dtype=np.float32), language=self.language or "en")

    def _transcribe(self, audio: NDArray[np.float32], prompt: str | None) -> str:
        from away.voice.asr import whisper_transcribe

        stats: dict[str, object] = {}
        text = whisper_transcribe(self.model, audio, self.language, prompt, None, self.beam_size, stats, self.languages)
        conf = stats.get("confidence")
        self.last_confidence = float(conf) if isinstance(conf, (int, float)) else None
        return text

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        return str(self._run(self._transcribe, audio, prompt))


def model_path(model: str, download_root: str | None, fetch: Any = None) -> str:
    """The folder with the MLX weights of ``model``: ``<models>/whisper-mlx-<model>`` (``away setup`` fills it),
    downloaded there now (pinned and checksum-verified) when it is missing. Without a models folder (a library
    caller), the pinned Hub repo id, which mlx_whisper downloads itself. ``KeyError``: no MLX build of ``model``."""
    from away.app import models

    spec = models.MLX_MODELS[model]
    if not download_root:
        return spec.repo
    out = models.mlx_dir(Path(download_root).parent, model)
    if not models.mlx_ready(out, model):
        (fetch or models.download_mlx)(model, Path(download_root).parent, lambda m: log.info(m.strip()))
    return str(out)
