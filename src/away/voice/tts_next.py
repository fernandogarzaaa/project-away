"""Voices from the 2026-10 model refresh (v0.1.24), in front of the v0.1.23 voice chain (:mod:`away.voice.natural`).

* **Kokoro-82M v1.0 fp16 on the graphics card** (``models_next.kokoro_precision``: ``fp16``, or ``auto`` with a CUDA
  onnxruntime): the same Kokoro voice and voices file, run in half precision with the CUDA provider. The fp16 model
  is the catalog's ``voice-kokoro-fp16-gpu`` (kokoro-onnx model-files-v1.0, sha256-checked). ``int8`` keeps the
  v0.1.23 int8 model on the processor.
* **Instant voice, Kyutai pocket-tts** (``models_next.instant_voice``; experimental): 100M parameters on two CPU
  cores, first audio in well under a second, used for short replies. It needs PyTorch, so it is the
  ``project-away[pocket]`` pip extra, never a bundled download. Weights CC-BY-4.0 (credit: Kyutai); voice cloning
  only with the speaker's consent.
* **Voice Studio clones**: a consent-gated local WAV profile can select Pocket TTS (reusable safetensors
  voice state) or Qwen3-TTS 0.6B Base (reusable in-memory clone prompt). Both are optional extras, and the selected
  clone is prepended to the existing chain; missing dependencies or engine errors fall back to existing voices.

Licence notes: Kokoro's phonemes come from espeak-ng (GPL-3.0) and Piper's engine is GPL-3.0; both stay optional
runtime downloads that are never bundled (as in v0.1.23).

The runner calls :func:`configure` with a getter for ``cfg.models_next``; :func:`next_factories` returns the engines
to put first (empty when nothing new is chosen, so v0.1.23's order is untouched).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

KOKORO_FP16_ID = "voice-kokoro-fp16-gpu"
KOKORO_FP16_FILE = "kokoro-v1.0.fp16.onnx"
INSTANT_MAX_WORDS = 25  # longer messages keep the natural voice
POCKET_VOICE = "alba"

_opts: Callable[[], Any] | None = None


def configure(opts: Callable[[], Any] | None) -> None:
    global _opts
    _opts = opts


def _get_opts() -> Any:
    try:
        return _opts() if _opts is not None else None
    except Exception:
        return None


def cuda_onnxruntime() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in set(ort.get_available_providers())
    except Exception:
        return False


def kokoro_precision(opts: Any, cuda: Callable[[], bool] = cuda_onnxruntime) -> str:
    want = str(getattr(opts, "kokoro_precision", "auto") or "auto")
    if want in ("int8", "fp16"):
        return want
    return "fp16" if cuda() else "int8"


def fp16_path(data_dir: Path | None = None) -> Path:
    from away.app.model_fetch import model_dir

    return model_dir(KOKORO_FP16_ID, data_dir) / KOKORO_FP16_FILE


def make_kokoro_fp16(settings: Callable[[], Any], data_dir: Path | None = None) -> Any:
    """The v0.1.23 Kokoro engine with the fp16 model on the CUDA provider."""
    from away.voice import natural

    p = fp16_path(data_dir)
    if not p.is_file():
        raise RuntimeError("the graphics-card Kokoro model is not downloaded yet (Settings > Models)")
    if not natural.installed("kokoro", data_dir=data_dir):
        raise RuntimeError("the natural voice is not downloaded yet (Settings > Voice & input > Download)")
    natural.activate(data_dir)
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    sess = ort.InferenceSession(str(p), providers=natural.providers())
    model = Kokoro.from_session(sess, str(natural.voices_dir(data_dir) / natural.KOKORO_MODEL[1].name))

    def voice() -> str:
        s = settings()
        return str(getattr(s, "kokoro_voice", "af_heart") or "af_heart") if s is not None else "af_heart"

    eng = natural.KokoroEngine(data_dir, voice, model=model)
    eng.name = "Natural (Kokoro, graphics card)"
    return eng


def pocket_available() -> bool:
    try:
        return importlib.util.find_spec("pocket_tts") is not None
    except (ImportError, ValueError):
        return False


def _qwen_available() -> bool:
    try:
        return importlib.util.find_spec("qwen_tts") is not None
    except (ImportError, ValueError):
        return False


class PocketEngine:
    """Kyutai pocket-tts for short replies; longer text goes to the next engine (raises so the chain moves on)."""

    name = "Instant (pocket-tts)"

    def __init__(self, model: Any = None, voice: str = POCKET_VOICE, play: Callable[..., None] | None = None) -> None:
        if model is None:
            from pocket_tts import TTSModel

            model = TTSModel.load_model()
        self.model = model
        self.state = model.get_state_for_audio_prompt(voice)
        from away.voice import natural

        self.play = play or natural.sd_play

    def speak(self, text: str, v: Any) -> None:
        if len(str(text).split()) > INSTANT_MAX_WORDS:
            raise RuntimeError("long message: the natural voice reads it")
        from away.voice import natural

        vol = natural.VOLUMES.get(str(getattr(v, "volume", "normal")), 0.9)

        def synth(piece: str) -> tuple[Any, int]:
            audio = self.model.generate_audio(self.state, piece)
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio
            return arr, int(self.model.sample_rate)

        natural.play_chunks(synth, natural.chunks(text), lambda s, sr: self.play(s, sr, vol))

    def close(self) -> None:
        pass


def next_factories(
    settings: Callable[[], Any],
    data_dir: Path | None = None,
    opts: Any = None,
    have_fp16: Callable[[], bool] | None = None,
    have_pocket: Callable[[], bool] = pocket_available,
    cuda: Callable[[], bool] = cuda_onnxruntime,
) -> list[tuple[str, Callable[[], Any]]]:
    """Engines to try before the v0.1.23 chain: pocket-tts for short replies when chosen, then Kokoro fp16 on the
    graphics card when that precision is chosen and downloaded. Empty by default."""
    o = opts if opts is not None else _get_opts()
    out: list[tuple[str, Callable[[], Any]]] = []
    # Voice Studio is independent of models_next: the selected custom profile takes precedence only when its
    # optional engine is installed. Speaker retains the existing voices as fallback on any failure.
    try:
        from away.voice.voice_cloning import profile_factory
        out += profile_factory(settings, data_dir, have_pocket=have_pocket, have_qwen=_qwen_available)
    except Exception:
        pass
    if o is None:
        return out
    if bool(getattr(o, "instant_voice", False)) and have_pocket():
        out.append((PocketEngine.name, lambda: PocketEngine()))
    s = settings()
    engine = str(getattr(s, "engine", "auto") or "auto") if s is not None else "auto"
    hf = have_fp16 or (lambda: fp16_path(data_dir).is_file())
    if engine in ("auto", "kokoro") and kokoro_precision(o, cuda) == "fp16" and hf():
        out.append(("Natural (Kokoro, graphics card)", lambda: make_kokoro_fp16(settings, data_dir)))
    return out
