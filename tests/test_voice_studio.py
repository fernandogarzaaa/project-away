from __future__ import annotations

import queue
import sys
import threading
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from away.voice import voice_profiles
from away.voice.voice_cloning import PocketCloneEngine, QwenCloneEngine, profile_factory


def _wav(path: Path, seconds: float = 5.0, sample_rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return path


def _profile(tmp_path: Path, engine: str = "pocket-clone", transcript: str = ""):
    src = _wav(tmp_path / "source.wav")
    return voice_profiles.create_profile(
        "My voice", engine, src, consent_confirmed=True, speaker_name="Me",
        reference_text=transcript, data_dir=tmp_path / "data",
    )


def test_voice_profile_requires_explicit_permission_and_speaker_identity(tmp_path: Path):
    src = _wav(tmp_path / "source.wav")
    with pytest.raises(PermissionError, match="permission"):
        voice_profiles.create_profile("Test", "pocket-clone", src, consent_confirmed=False,
                                      speaker_name="Me", data_dir=tmp_path / "data")
    with pytest.raises(ValueError, match="whose voice|permission"):
        voice_profiles.create_profile("Test", "pocket-clone", src, consent_confirmed=True,
                                      speaker_name="", data_dir=tmp_path / "data")


def test_qwen_profile_requires_reference_transcript(tmp_path: Path):
    src = _wav(tmp_path / "source.wav")
    with pytest.raises(ValueError, match="transcript"):
        voice_profiles.create_profile("Qwen test", "qwen-clone", src, consent_confirmed=True,
                                      speaker_name="Me", data_dir=tmp_path / "data")


@pytest.mark.parametrize("seconds", [1.0, 31.0])
def test_voice_profile_rejects_wav_outside_duration_bounds(tmp_path: Path, seconds: float):
    src = _wav(tmp_path / "source.wav", seconds=seconds)
    with pytest.raises(ValueError, match="3–30 seconds"):
        voice_profiles.create_profile("Test", "pocket-clone", src, consent_confirmed=True,
                                      speaker_name="Me", data_dir=tmp_path / "data")


def test_profile_storage_is_local_and_delete_removes_reference_and_cache(tmp_path: Path):
    profile = _profile(tmp_path)
    wav_path = voice_profiles.reference_path(profile.id, tmp_path / "data")
    cache = voice_profiles.cache_path(profile.id, "pocket-clone", tmp_path / "data")
    cache.write_bytes(b"test-cache")
    assert wav_path.is_file() and cache.is_file()
    assert voice_profiles.get_profile(profile.id, tmp_path / "data").name == "My voice"
    assert voice_profiles.delete_profile(profile.id, tmp_path / "data")
    assert not wav_path.parent.exists()
    assert not voice_profiles.delete_profile(profile.id, tmp_path / "data")


def test_profile_id_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="id"):
        voice_profiles.reference_path("../../outside", tmp_path / "data")


def test_pocket_clone_caches_reusable_voice_state_as_safetensors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profile = _profile(tmp_path)
    module = ModuleType("pocket_tts")
    def export(state, path):
        Path(path).write_bytes(b"safe-state")
    module.export_model_state = export  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pocket_tts", module)

    class FakeModel:
        sample_rate = 24000
        def __init__(self):
            self.prompts = []
        def get_state_for_audio_prompt(self, prompt):
            self.prompts.append(str(prompt))
            return {"prompt": str(prompt)}
        def generate_audio(self, state, text):
            return np.zeros(240, dtype=np.float32)

    model = FakeModel()
    played = []
    engine = PocketCloneEngine(profile.id, tmp_path / "data", model=model,
                               play=lambda samples, sr, volume: played.append((len(samples), sr)))
    cache = voice_profiles.cache_path(profile.id, "pocket-clone", tmp_path / "data")
    assert cache.is_file() and cache.read_bytes() == b"safe-state"
    assert model.prompts[0].endswith("reference.wav")
    engine.speak("Hello there.", SimpleNamespace(volume="normal"))
    assert played and all(sr == 24000 for _, sr in played)

    second_model = FakeModel()
    PocketCloneEngine(profile.id, tmp_path / "data", model=second_model,
                      play=lambda *_: None)
    assert second_model.prompts == [str(cache)]


def test_qwen_clone_builds_one_reusable_in_memory_prompt_and_speaks(tmp_path: Path):
    profile = _profile(tmp_path, "qwen-clone", "Hello, my name is Alex.")
    class FakeModel:
        def __init__(self):
            self.prompt_calls = []
            self.generation_calls = []
        def create_voice_clone_prompt(self, **kwargs):
            self.prompt_calls.append(kwargs)
            return ["reusable-prompt"]
        def generate_voice_clone(self, **kwargs):
            self.generation_calls.append(kwargs)
            return [np.zeros(240, dtype=np.float32)], 24000

    model = FakeModel()
    played = []
    engine = QwenCloneEngine(profile.id, tmp_path / "data", model=model,
                             play=lambda samples, sr, volume: played.append(sr))
    engine.speak("Hello there. How are you?", SimpleNamespace(volume="normal"))
    assert len(model.prompt_calls) == 1
    assert model.prompt_calls[0]["ref_text"] == "Hello, my name is Alex."
    assert model.generation_calls and model.generation_calls[0]["voice_clone_prompt"] == ["reusable-prompt"]
    assert played and all(sr == 24000 for sr in played)


def test_selected_custom_profile_precedes_existing_voice_chain_but_is_optional(tmp_path: Path):
    profile = _profile(tmp_path)
    cfg = SimpleNamespace(custom_voice_profile=profile.id, engine="auto")
    got = profile_factory(lambda: cfg, tmp_path / "data", have_pocket=lambda: True, have_qwen=lambda: True)
    assert len(got) == 1 and got[0][0].startswith("Custom voice (Pocket TTS)")
    missing = profile_factory(lambda: cfg, tmp_path / "data", have_pocket=lambda: False, have_qwen=lambda: False)
    assert missing == []


def test_live_speech_interrupt_cannot_race_with_queue_insertion():
    from away.liveagent.speech import LiveSpeech

    class BlockingSpeaker:
        def __init__(self):
            self._q = queue.Queue()
            self._lock = threading.Lock()
            self.entered = threading.Event()
            self.release = threading.Event()
        def say(self, text, force=False):
            self.entered.set()
            assert self.release.wait(2)
            with self._lock:
                self._q.put(text)
            return {"spoken": True}

    speaker = BlockingSpeaker()
    speech = LiveSpeech(speaker, stop_audio=lambda: None)
    said = threading.Thread(target=lambda: speech.say("Do not play after interruption."))
    said.start()
    assert speaker.entered.wait(1)
    interrupted = threading.Thread(target=speech.interrupt)
    interrupted.start()
    speaker.release.set()
    said.join(2)
    interrupted.join(2)
    assert not said.is_alive() and not interrupted.is_alive()
    assert speaker._q.empty()
    assert speech.say("Still cancelled.") is False


def test_barge_in_only_arms_when_audio_is_actually_playing_and_rejects_bad_samples():
    from away.liveagent.speech import BargeIn, LiveSpeech

    class Speaker:
        def __init__(self):
            self._idle = threading.Event()
            self._idle.clear()  # queued/loading, but not actually playing
            self._speaking = threading.Event()
    speaker = Speaker()
    live = LiveSpeech(speaker, stop_audio=lambda: None)
    assert live.speaking() is False
    speaker._speaking.set()
    assert live.speaking() is True

    fired = []
    barge = BargeIn(lambda: True, lambda: fired.append(True), sample_rate=16000)
    assert barge.feed(np.array([np.nan, np.inf], dtype=np.float32)) is False
    assert fired == []
    with pytest.raises(ValueError, match="sample_rate"):
        BargeIn(lambda: True, lambda: None, sample_rate=0)


def test_speaker_playback_flag_is_false_while_idle_and_true_during_engine_playback():
    from away.voice.speech import Speaker

    entered = threading.Event()
    release = threading.Event()
    class BlockingEngine:
        name = "blocking"
        def speak(self, text, voice):
            entered.set()
            assert release.wait(2)
        def close(self):
            pass

    speaker = Speaker(lambda: SimpleNamespace(enabled=True, rate="normal", volume="normal", voice=""),
                      factories=[("blocking", BlockingEngine)])
    assert not speaker._speaking.is_set()
    assert speaker.say("hello")["spoken"]
    assert entered.wait(1)
    assert speaker._speaking.is_set()
    release.set()
    assert speaker.wait(2)
    assert not speaker._speaking.is_set()
    speaker.stop()
