"""Model registry and downloads (stdlib only, sha256-verified where pinned)."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str | None
    size_mb: float
    purpose: str


MEDIAPIPE_MODELS = (
    ModelFile(
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        7.8,
        "hand tracking (21 landmarks)",
    ),
    ModelFile(
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        3.8,
        "face landmarks + blendshapes (speaking / brow raise)",
    ),
)

ASR_SIZES_MB = {"large-v3-turbo": 1620, "small": 485, "base": 150, "tiny": 75, "medium": 1530, "large-v3": 3100}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected: str | None) -> bool:
    return path.exists() and (expected is None or sha256_file(path) == expected)


def download(
    mf: ModelFile,
    dest_dir: Path,
    progress: Callable[[str], None] = print,
    opener: Callable[[str], object] | None = None,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / mf.name
    if verify(dest, mf.sha256):
        progress(f"  {mf.name}: already present")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    progress(f"  {mf.name}: downloading {mf.size_mb:.1f} MB ...")
    open_url = opener or (lambda u: urllib.request.urlopen(u, timeout=120))
    with open_url(mf.url) as r, tmp.open("wb") as fh:  # type: ignore[attr-defined,union-attr]
        shutil.copyfileobj(r, fh)
    if mf.sha256 and sha256_file(tmp) != mf.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{mf.name}: checksum mismatch (download corrupted or model updated upstream)")
    tmp.replace(dest)
    return dest


def download_asr(model: str, dest_dir: Path, progress: Callable[[str], None] = print) -> Path:  # pragma: no cover
    from faster_whisper import download_model

    progress(f"  ASR '{model}': downloading ~{ASR_SIZES_MB.get(model, 0)} MB from Hugging Face ...")
    path = download_model(model, output_dir=str(dest_dir / f"whisper-{model}"))
    return Path(path)


def model_path(models_dir: Path, name: str) -> Path | None:
    p = models_dir / name
    return p if p.exists() else None
