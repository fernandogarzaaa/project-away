"""Optional local voice-cloning engines used by Voice Studio.

Pocket TTS caches its reusable voice state in safetensors. Qwen3-TTS keeps its derived clone prompt in memory
and rebuilds it from the local WAV on next launch rather than serializing Python objects or tensors unsafely.
Both engines are opt-in, local-only, and rely on a consent-gated profile created by voice_profiles.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.voice import natural, voice_profiles


def pocket_available() -> bool:
    try:
        return importlib.util.find_spec("pocket_tts") is not None
    except (ImportError, ValueError):
        return False


def qwen_available() -> bool:
    try:
        return importlib.util.find_spec("qwen_tts") is not None
    except (ImportError, ValueError):
        return False


class PocketCloneEngine:
    """Pocket TTS speaker state from a user-provided WAV, cached as safetensors in that profile's directory."""

    def __init__(
        self,
        profile_id: str,
        data_dir: Path | None = None,
        model: Any = None,
        play: Callable[..., None] | None = None,
    ) -> None:
        self.profile = voice_profiles.get_profile(profile_id, data_dir)
        if self.profile.engine != "pocket-clone":
            raise ValueError("selected voice profile is not a Pocket TTS clone")
        self.name = f"Custom voice (Pocket TTS): {self.profile.name}"
        self.data_dir = data_dir
        if model is None:
            from pocket_tts import TTSModel

            model = TTSModel.load_model()
        self.model = model
        self.play = play or natural.sd_play
        cache = voice_profiles.cache_path(profile_id, "pocket-clone", data_dir)
        reference = voice_profiles.reference_path(profile_id, data_dir)
        if cache.is_file():
            self.state = model.get_state_for_audio_prompt(str(cache))
        else:
            self.state = model.get_state_for_audio_prompt(str(reference))
            try:
                from pocket_tts import export_model_state

                export_model_state(self.state, str(cache))
                if cache.is_file() and __import__("os").name != "nt":
                    cache.chmod(0o600)
            except Exception:
                # Cache is an optimization only. Voice synthesis still works without it.
                pass

    def speak(self, text: str, v: Any) -> None:
        volume = natural.VOLUMES.get(str(getattr(v, "volume", "normal")), 0.9)

        def synth(piece: str) -> tuple[Any, int]:
            audio = self.model.generate_audio(self.state, piece)
            samples = audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio
            return samples, int(self.model.sample_rate)
        natural.play_chunks(synth, natural.chunks(text), lambda samples, sr: self.play(samples, sr, volume))

    def close(self) -> None:
        # Pocket's small model is shared only for this engine lifetime; no profile data is deleted here.
        pass


class QwenCloneEngine:
    """Qwen3-TTS 0.6B Base. The reusable clone prompt stays in memory; never pickle model-derived objects."""

    def __init__(
        self,
        profile_id: str,
        data_dir: Path | None = None,
        model: Any = None,
        play: Callable[..., None] | None = None,
    ) -> None:
        self.profile = voice_profiles.get_profile(profile_id, data_dir)
        if self.profile.engine != "qwen-clone":
            raise ValueError("selected voice profile is not a Qwen3-TTS clone")
        if not self.profile.reference_text.strip():
            raise ValueError("Qwen3-TTS needs a transcript for the reference audio")
        self.name = f"Custom voice (Qwen3-TTS): {self.profile.name}"
        self.play = play or natural.sd_play
        if model is None:
            from qwen_tts import Qwen3TTSModel

            model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cpu")
        self.model = model
        reference = voice_profiles.reference_path(profile_id, data_dir)
        self.prompt = model.create_voice_clone_prompt(
            ref_audio=str(reference),
            ref_text=self.profile.reference_text,
            x_vector_only_mode=False,
        )

    def speak(self, text: str, v: Any) -> None:
        volume = natural.VOLUMES.get(str(getattr(v, "volume", "normal")), 0.9)
        def synth(piece: str) -> tuple[Any, int]:
            wavs, sr = self.model.generate_voice_clone(
                text=piece,
                language="English",
                voice_clone_prompt=self.prompt,
            )
            return wavs[0], int(sr)
        natural.play_chunks(synth, natural.chunks(text), lambda samples, sr: self.play(samples, sr, volume))

    def close(self) -> None:
        pass


def profile_factory(
    settings: Callable[[], Any],
    data_dir: Path | None = None,
    *,
    have_pocket: Callable[[], bool] = pocket_available,
    have_qwen: Callable[[], bool] = qwen_available,
) -> list[tuple[str, Callable[[], Any]]]:
    """Return the selected custom voice first, if its optional runtime is installed.

    The Speaker's existing engine-failure chain handles a missing dependency, broken profile, or model load failure.
    """
    try:
        cfg = settings()
        profile_id = str(getattr(cfg, "custom_voice_profile", "") or "") if cfg is not None else ""
        if not profile_id:
            return []
        profile = voice_profiles.get_profile(profile_id, data_dir)
        if profile.engine == "pocket-clone" and have_pocket():
            return [(f"Custom voice (Pocket TTS): {profile.name}", lambda: PocketCloneEngine(profile_id, data_dir))]
        if profile.engine == "qwen-clone" and have_qwen():
            return [(f"Custom voice (Qwen3-TTS): {profile.name}", lambda: QwenCloneEngine(profile_id, data_dir))]
    except Exception:
        return []
    return []
