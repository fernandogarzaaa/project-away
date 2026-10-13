"""The ONNX versions of the MediaPipe hand and face models (v0.1.24), downloaded on demand for graphics-card
tracking (``hands.compute``).

Source: the Hugging Face repository ``takoyakisoft/vtuberkit-vision-landmarkers-onnx`` at a pinned revision,
whose files are Google's official MediaPipe ``hand_landmarker.task`` / ``face_landmarker.task`` models converted to
ONNX. License: Apache-2.0, the same as the MediaPipe originals (weights (c) Google LLC; see the repository's
``licenses/`` folder). Each file is checked against the SHA-256 pinned below before it is used; a partial or
changed file is never loaded. Stored in ``<data>/models/gpu-tracking/``.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = "takoyakisoft/vtuberkit-vision-landmarkers-onnx"
REVISION = "3f3d0bf8e56f4474bc670b9eee1b697856241ccb"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class OnnxModel:
    key: str
    path: str  # in the repository
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return BASE + self.path

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]


SETS: dict[str, tuple[OnnxModel, ...]] = {
    "hands": (
        OnnxModel("palm_detector", "onnx/hand/palm_detector.onnx", "939e53b8c438bdbdcaaf2e5589c8204cb6c92d383a6fe87c939ca3bb52f605c7", 4589448),
        OnnxModel("hand_landmark", "onnx/hand/hand_landmark.onnx", "716f2d75eb41cbb8f222cda9099eb617ec0a3248ecb96e77c0dc916759b5bb1c", 10903281),
    ),
    "face": (
        OnnxModel("face_detector", "onnx/face/face_detector.onnx", "0acb84ef882f18005675b14b3d8021038fab9fc2d2e6365fc73787022d7dda48", 418530),
        OnnxModel("face_landmark", "onnx/face/face_landmark.onnx", "38ea4d75bffea5a72cb93ec027ddeaf31763049ffa57ec944fa3840119083bec", 4921071),
        OnnxModel("face_blendshape", "onnx/face/face_blendshape.onnx", "4d97c181a04c1ac1778daa7ffbd9ae1e80b65300750078a7edbcf456d29b53c3", 1883864),
    ),
}  # fmt: skip


class ModelDownloadError(RuntimeError):
    pass


def folder(data_dir: Path | None = None) -> Path:
    from away import paths

    return Path(data_dir or paths.data_dir()) / "models" / "gpu-tracking"


def paths(which: str, data_dir: Path | None = None) -> dict[str, Path]:
    d = folder(data_dir)
    return {m.key: d / m.filename for m in SETS[which]}


def _ok(p: Path, m: OnnxModel) -> bool:
    try:
        return p.is_file() and p.stat().st_size == m.size
    except OSError:
        return False


def ready(which: str, data_dir: Path | None = None) -> bool:
    d = folder(data_dir)
    return all(_ok(d / m.filename, m) for m in SETS[which])


def download_size(which: str = "hands", data_dir: Path | None = None) -> int:
    d = folder(data_dir) if data_dir is not None else None
    return sum(m.size for m in SETS[which] if d is None or not _ok(d / m.filename, m))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    which: str,
    data_dir: Path | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
    progress: Callable[[int, int], Any] | None = None,
) -> dict[str, Path]:
    """Download (or finish) one set; returns its paths. Raises :class:`ModelDownloadError` in plain words."""
    d = folder(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    total = sum(m.size for m in SETS[which])
    done = 0
    for m in SETS[which]:
        dest = d / m.filename
        if _ok(dest, m) and sha256(dest) == m.sha256:
            done += m.size
            continue
        tmp = dest.with_suffix(".part")
        try:
            req = urllib.request.Request(m.url, headers={"User-Agent": "ProjectAway"})
            with opener(req, timeout=60) as resp, tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise ModelDownloadError(f"could not download the graphics-card models ({exc}); check the internet") from None
        got = sha256(tmp)
        if got != m.sha256:
            tmp.unlink(missing_ok=True)
            raise ModelDownloadError(f"{m.filename} did not match its pinned checksum; nothing was installed")
        os.replace(tmp, dest)
    return paths(which, data_dir)


def remove(data_dir: Path | None = None) -> bool:
    d = folder(data_dir)
    gone = False
    for ms in SETS.values():
        for m in ms:
            p = d / m.filename
            if p.exists():
                p.unlink()
                gone = True
    return gone
