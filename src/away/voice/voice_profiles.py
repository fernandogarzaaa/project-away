"""Local, consent-gated voice profiles for optional voice cloning engines.

Profiles are deliberately stored outside logs and settings. Only a WAV reference, minimal metadata,
and engine-specific local caches are written. Nothing in this module uploads audio or contacts a service.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_WAV_BYTES = 25 * 1024 * 1024
MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
PROFILE_ID = re.compile(r"^[a-f0-9]{32}$")
ENGINES = ("pocket-clone", "qwen-clone")


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    engine: str
    reference_text: str
    created_at: str
    duration_s: float

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "engine": self.engine,
            "reference_text": self.reference_text,
            "created_at": self.created_at,
            "duration_s": self.duration_s,
        }


def profile_dir(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        from away import paths

        data_dir = paths.data_dir()
    return Path(data_dir) / "voice_profiles"


def _profile_path(profile_id: str, data_dir: Path | None = None) -> Path:
    if not PROFILE_ID.fullmatch(str(profile_id)):
        raise ValueError("invalid voice profile id")
    root = profile_dir(data_dir).resolve()
    path = (root / profile_id).resolve()
    if path.parent != root:
        raise ValueError("invalid voice profile path")
    return path


def validate_wav(path: Path) -> float:
    """Accept ordinary uncompressed PCM WAV only; bound duration and file size before model processing."""
    path = Path(path)
    try:
        size = path.stat().st_size
        if size < 44 or size > MAX_WAV_BYTES:
            raise ValueError("WAV must be between 44 bytes and 25 MB")
        with wave.open(str(path), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ValueError("voice sample must be uncompressed PCM WAV")
            if wav.getnchannels() not in (1, 2):
                raise ValueError("voice sample must be mono or stereo")
            if wav.getsampwidth() not in (2, 3, 4):
                raise ValueError("voice sample must use 16-, 24-, or 32-bit PCM")
            sr = wav.getframerate()
            if not 16000 <= sr <= 48000:
                raise ValueError("voice sample sample rate must be 16–48 kHz")
            frames = wav.getnframes()
            duration = frames / float(sr)
            if not MIN_SECONDS <= duration <= MAX_SECONDS:
                raise ValueError("voice sample must be 3–30 seconds long")
            if frames <= 0:
                raise ValueError("voice sample is empty")
            return duration
    except (wave.Error, EOFError) as exc:
        raise ValueError("file is not a valid PCM WAV") from exc


def create_profile(
    name: str,
    engine: str,
    wav_path: Path,
    *,
    consent_confirmed: bool,
    speaker_name: str = "",
    reference_text: str = "",
    data_dir: Path | None = None,
) -> VoiceProfile:
    """Copy a validated WAV into local storage after an explicit consent acknowledgement."""
    clean_name = " ".join(str(name).split()).strip()
    speaker = " ".join(str(speaker_name).split()).strip()
    transcript = " ".join(str(reference_text).split()).strip()
    if not clean_name or len(clean_name) > 48 or any(ord(c) < 32 for c in clean_name):
        raise ValueError("voice profile name must be 1–48 printable characters")
    if engine not in ENGINES:
        raise ValueError("choose Pocket TTS or Qwen3-TTS")
    if not consent_confirmed:
        raise PermissionError("confirm you have the speaker's permission before creating a voice profile")
    if not speaker or len(speaker) > 80:
        raise ValueError("enter whose voice this is and confirm you have permission to use it")
    if engine == "qwen-clone" and not transcript:
        raise ValueError("Qwen3-TTS needs an accurate transcript of the reference audio")
    if len(transcript) > 1000:
        raise ValueError("reference transcript must be at most 1000 characters")
    src = Path(wav_path).expanduser().resolve()
    duration = validate_wav(src)
    profile_id = uuid.uuid4().hex
    root = _profile_path(profile_id, data_dir)
    root.mkdir(parents=True, exist_ok=False)
    if os.name != "nt":
        os.chmod(root, 0o700)
    try:
        dest = root / "reference.wav"
        shutil.copyfile(src, dest)
        if os.name != "nt":
            os.chmod(dest, 0o600)
        created = datetime.now(timezone.utc).isoformat()
        meta = {
            "version": 1,
            "id": profile_id,
            "name": clean_name,
            "engine": engine,
            "speaker_name": speaker,
            "reference_text": transcript,
            "consent_confirmed": True,
            "consent_statement": "The creator confirms they have permission to synthesize this speaker's voice.",
            "created_at": created,
            "duration_s": duration,
        }
        temp = root / "profile.json.tmp"
        temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if os.name != "nt":
            os.chmod(temp, 0o600)
        os.replace(temp, root / "profile.json")
        return VoiceProfile(profile_id, clean_name, engine, transcript, created, duration)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def get_profile(profile_id: str, data_dir: Path | None = None) -> VoiceProfile:
    path = _profile_path(profile_id, data_dir)
    try:
        raw = json.loads((path / "profile.json").read_text(encoding="utf-8"))
        if raw.get("version") != 1 or raw.get("id") != profile_id or not raw.get("consent_confirmed"):
            raise ValueError("voice profile is invalid or has no consent acknowledgement")
        engine = str(raw.get("engine", ""))
        if engine not in ENGINES:
            raise ValueError("voice profile uses an unsupported engine")
        return VoiceProfile(
            id=profile_id,
            name=str(raw["name"]),
            engine=engine,
            reference_text=str(raw.get("reference_text", "")),
            created_at=str(raw["created_at"]),
            duration_s=float(raw["duration_s"]),
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("voice profile could not be read") from exc


def reference_path(profile_id: str, data_dir: Path | None = None) -> Path:
    profile = get_profile(profile_id, data_dir)
    path = _profile_path(profile.id, data_dir) / "reference.wav"
    if not path.is_file():
        raise ValueError("voice reference audio is missing")
    validate_wav(path)
    return path


def list_profiles(data_dir: Path | None = None) -> list[VoiceProfile]:
    root = profile_dir(data_dir)
    if not root.is_dir():
        return []
    out: list[VoiceProfile] = []
    for item in root.iterdir():
        if item.is_dir() and PROFILE_ID.fullmatch(item.name):
            try:
                out.append(get_profile(item.name, data_dir))
            except (OSError, ValueError):
                continue
    return sorted(out, key=lambda p: (p.name.casefold(), p.id))


def delete_profile(profile_id: str, data_dir: Path | None = None) -> bool:
    """Delete original audio, metadata, and every cached embedding for the selected profile."""
    path = _profile_path(profile_id, data_dir)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def cache_path(profile_id: str, engine: str, data_dir: Path | None = None) -> Path:
    if engine not in ENGINES:
        raise ValueError("unsupported voice cloning engine")
    _profile_path(profile_id, data_dir)
    return _profile_path(profile_id, data_dir) / (
        "pocket-state.safetensors" if engine == "pocket-clone" else "qwen-prompt.cache"
    )
