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


def dir_bytes(path: Path) -> int:
    """Bytes under ``path``, counting partial (``.incomplete``) download files too."""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    return total


def progress_bar(done: int, total: int, width: int = 30) -> str:
    """``[#####.....]  41%  664/1620 MB`` (clamped; ``total`` <= 0 shows MB only)."""
    mb = done / 1_000_000
    if total <= 0:
        return f"  {mb:,.0f} MB"
    frac = min(max(done / total, 0.0), 1.0)
    filled = round(frac * width)
    return f"  [{'#' * filled}{'.' * (width - filled)}] {frac * 100:3.0f}%  {mb:,.0f}/{total / 1_000_000:,.0f} MB"


class _DirProgress:
    """Background thread that redraws one progress line while a folder fills up."""

    def __init__(self, path: Path, total_bytes: int, write: Callable[[str], object], interval: float = 0.5) -> None:
        import threading

        self.path, self.total, self.write, self.interval = path, total_bytes, write, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="away-download-progress", daemon=True)

    def _draw(self) -> None:
        self.write("\r" + progress_bar(dir_bytes(self.path), self.total))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._draw()

    def __enter__(self) -> _DirProgress:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        if exc[0] is None:
            self.write("\r" + progress_bar(self.total, self.total))
        self.write("\n")


def quiet_hf_hub() -> None:
    """Hide huggingface_hub's harmless warnings (no HF_TOKEN, hf_xet missing, deprecated args)."""
    import logging
    import os
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    warnings.filterwarnings("ignore", module=r"huggingface_hub(\..*)?")
    for name in ("huggingface_hub", "huggingface_hub.file_download", "huggingface_hub.utils._http"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _stdout_writer() -> Callable[[str], None]:
    import sys

    stream = sys.stdout  # None in a windowed frozen exe

    def write(text: str) -> None:
        if stream is None:
            return
        try:
            stream.write(text)
            stream.flush()
        except (OSError, ValueError):
            pass

    return write


def _hub_download(model: str, out: Path, write: Callable[[str], object]) -> Path:  # pragma: no cover - network
    """Fallback: faster-whisper's own huggingface_hub download (uses hf_xet when it is installed)."""
    quiet_hf_hub()
    from faster_whisper import download_model

    with _DirProgress(out, ASR_SIZES_MB.get(model, 0) * 1_000_000, write):
        path = download_model(model, output_dir=str(out))
    return Path(path)


def download_asr(
    model: str,
    dest_dir: Path,
    progress: Callable[[str], None] = print,
    connections: int = 8,
    write: Callable[[str], object] | None = None,
    fast: Callable[..., Path] | None = None,
    slow: Callable[[str, Path, Callable[[str], object]], Path] | None = None,
) -> Path:
    """Download Whisper ``model`` into ``dest_dir/whisper-<model>``.

    Primary path: parallel, resumable, checksum-verified ranged download (``away.app.fastdl``).
    Any failure other than a checksum mismatch falls back to faster-whisper's huggingface_hub download.
    """
    from away.app import fastdl

    out = dest_dir / f"whisper-{model}"
    size_mb = ASR_SIZES_MB.get(model, 0)
    write = write or _stdout_writer()
    fast = fast or fastdl.download_model
    slow = slow or _hub_download
    progress(f"  ASR '{model}': downloading ~{size_mb} MB from Hugging Face ({connections} connections) ...")
    try:
        return Path(fast(model, out, write, connections=connections))
    except fastdl.ChecksumError:
        raise
    except Exception as exc:
        progress(f"  fast download unavailable ({type(exc).__name__}: {exc}); using the standard downloader ...")
    return Path(slow(model, out, write))


def model_path(models_dir: Path, name: str) -> Path | None:
    p = models_dir / name
    return p if p.exists() else None
