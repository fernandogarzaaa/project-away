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


# fastdl's own temporary files: a preallocated (sparse) full-size ``.part`` and its JSON sidecar. Counting
# them made the fallback bar jump to 100% at once (v0.1.4) while huggingface_hub restarted from zero.
FASTDL_TEMP_SUFFIXES = (".part", ".part.json", ".part.json.tmp")


def is_fastdl_temp(path: Path) -> bool:
    return path.name.endswith(FASTDL_TEMP_SUFFIXES)


def dir_bytes(path: Path) -> int:
    """Bytes really downloaded under ``path``: finished files plus huggingface_hub's ``.incomplete`` files,
    but not fastdl's preallocated ``.part`` files (their size is the final size, not progress)."""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            try:
                if f.is_file() and not is_fastdl_temp(f):
                    total += f.stat().st_size
            except OSError:
                continue
    return total


def progress_bar(done: int, total: int, width: int = 30, final: bool = True) -> str:
    """``[#####.....]  41%  664/1620 MB`` (clamped to ``total``; ``total`` <= 0 shows MB only).

    ``final=False`` (still running) never shows more than 99%: the expected size is an estimate."""
    if total <= 0:
        return f"  {max(done, 0) / 1_000_000:,.0f} MB"
    done = min(max(done, 0), total)
    frac = done / total
    if not final:
        frac = min(frac, 0.99)
    filled = round(frac * width)
    pct = frac * 100
    shown = f"{pct:3.0f}" if final or pct < 99 else " 99"
    return f"  [{'#' * filled}{'.' * (width - filled)}] {shown}%  {done / 1_000_000:,.0f}/{total / 1_000_000:,.0f} MB"


def cleanup_fastdl_temp(path: Path) -> int:
    """Delete fastdl ``.part`` leftovers once a model is complete (they can be 1.6 GB). Returns files removed."""
    n = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file() and is_fastdl_temp(f):
                try:
                    f.unlink()
                    n += 1
                except OSError:
                    pass
    return n


class _DirProgress:
    """Background thread that redraws one progress line while a folder fills up."""

    def __init__(self, path: Path, total_bytes: int, write: Callable[[str], object], interval: float = 0.5) -> None:
        import threading

        self.path, self.total, self.write, self.interval = path, total_bytes, write, interval
        self.base = 0  # bytes already there when the download started (finished small files)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="away-download-progress", daemon=True)

    def _draw(self) -> None:
        self.write("\r" + progress_bar(dir_bytes(self.path) - self.base, self.total, final=False))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._draw()

    def __enter__(self) -> _DirProgress:
        self.base = 0
        self.base = dir_bytes(self.path)
        self.total = max(self.total - self.base, 0) if self.total > 0 else self.total
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


FAST_ATTEMPTS = 3  # fast-download runs (each resumes the last) before the huggingface_hub fallback
FAST_PAUSE_S = (5.0, 20.0)


def slow_hint(model: str, progress: Callable[[str], None]) -> Callable[[float, float | None], None]:
    """Build fastdl's ``on_slow`` callback: estimate the time left and suggest the smaller model."""
    from away.app import fastdl

    def hint(rate: float, eta: float | None) -> None:
        msg = f"  Slow connection ({rate / 1_000_000:.2f} MB/s): about {fastdl.format_eta(eta)} left for '{model}'."
        smaller = "small" if model not in ("small", "base", "tiny") else None
        if smaller:
            mb = ASR_SIZES_MB[smaller]
            msg += (
                f"\n  Tip: press Ctrl+C and run `away setup --asr-model {smaller}` (~{mb} MB, about "
                f"{fastdl.format_eta(mb * 1_000_000 / rate)}) for a quicker, slightly less accurate model. "
                "This download keeps its progress and resumes if you rerun setup later."
            )
        progress(msg)

    return hint


def download_asr(
    model: str,
    dest_dir: Path,
    progress: Callable[[str], None] = print,
    connections: int = 8,
    write: Callable[[str], object] | None = None,
    fast: Callable[..., Path] | None = None,
    slow: Callable[[str, Path, Callable[[str], object]], Path] | None = None,
    attempts: int = FAST_ATTEMPTS,
    sleep: Callable[[float], None] | None = None,
    engine: str = "faster-whisper",
) -> Path:
    """Download Whisper ``model`` into ``dest_dir/whisper-<model>``.

    v0.1.35: ``engine="mlx-whisper"`` fetches the MLX weights into ``dest_dir/whisper-mlx-<model>`` instead
    (:func:`download_mlx`; Apple Silicon, see :func:`away.app.frozen.asr_engine`).

    Primary path: parallel, resumable, checksum-verified ranged download (``away.app.fastdl``). A failed run
    keeps its ``.part`` + sidecar, waits, and resumes (up to ``attempts`` runs); only then does it fall back to
    faster-whisper's huggingface_hub download, as a last resort. A checksum mismatch never falls back.
    """
    import time

    from away.app import fastdl

    if engine == "mlx-whisper":
        return download_mlx(model, dest_dir, progress, connections, write, fast)
    out = dest_dir / f"whisper-{model}"
    size_mb = ASR_SIZES_MB.get(model, 0)
    write = write or _stdout_writer()
    sleep = sleep or time.sleep
    progress(f"  ASR '{model}': downloading ~{size_mb} MB from Hugging Face ({connections} connections) ...")
    last: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            if fast is not None:
                return Path(fast(model, out, write, connections=connections))
            return Path(
                fastdl.download_model(model, out, write, connections=connections, on_slow=slow_hint(model, progress))
            )
        except fastdl.ChecksumError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                pause = FAST_PAUSE_S[min(i, len(FAST_PAUSE_S) - 1)]
                progress(
                    f"  download interrupted ({type(exc).__name__}: {exc}); "
                    f"progress kept, resuming in {pause:.0f} s ..."
                )
                sleep(pause)
    progress(f"  fast download unavailable ({type(last).__name__}: {last}); using the standard downloader ...")
    path = Path((slow or _hub_download)(model, out, write))
    cleanup_fastdl_temp(out)
    return path


@dataclass(frozen=True)
class MLXModel:
    """v0.1.35: Whisper weights in MLX format (Apple Silicon GPU, :mod:`away.voice.asr_mlx`), pinned to one commit
    of the mlx-community repo and to the sha256 of the weights file (checked against the Hub listing before the
    download and again on the downloaded bytes by :mod:`away.app.fastdl`)."""

    repo: str
    revision: str
    weights: str  # weights.npz | weights.safetensors
    sha256: str
    size_mb: int


MLX_MODELS = {
    "tiny": MLXModel(
        "mlx-community/whisper-tiny-mlx",
        "6caf9c55601caafbe6508a8b0d216bdf4783c4e8",
        "weights.npz",
        "0e03a5993d6eea43b07ee2dcc772b0e4cef5bb227257dacc24bf289387d49186",
        75,
    ),
    "base": MLXModel(
        "mlx-community/whisper-base-mlx",
        "1e3e249fb8d01c655324bd6841b1deadffd6d04c",
        "weights.npz",
        "2f57d5f3ef473054c638961f90716f4ee415e8108de81313eccb2c5fd62eff0b",
        144,
    ),
    "small": MLXModel(
        "mlx-community/whisper-small-mlx",
        "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
        "weights.npz",
        "55b6674c9b339702d486e2b1573839a66f8ec8f821ed2886993ef717a86b09f5",
        481,
    ),
    "medium": MLXModel(
        "mlx-community/whisper-medium-mlx",
        "7fc08c4eac4c316526498f147dfdee6f6303f975",
        "weights.npz",
        "10b597c2bcb1bcc38b2d3d24cd4f0885f461a7cd70e8444d6ad5a763ece549ea",
        1525,
    ),
    "large-v3": MLXModel(
        "mlx-community/whisper-large-v3-mlx",
        "49e6aa286ad60c14352c404340ded53710378a11",
        "weights.npz",
        "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d",
        3084,
    ),
    "large-v3-turbo": MLXModel(
        "mlx-community/whisper-large-v3-turbo",
        "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
        "weights.safetensors",
        "951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6",
        1614,
    ),
    "distil-large-v3": MLXModel(
        "mlx-community/distil-whisper-large-v3",
        "e1c3c155644be59f8b477c0186719442f7e3fbb0",
        "weights.npz",
        "dacfaf0b80d75c8b217a77a1e1da49d935a17a7581618f4811bd1e94aa7a92d5",
        1509,
    ),
}


def mlx_dir(models_dir: Path, model: str) -> Path:
    """Where ``away setup`` puts the MLX weights of ``model`` (next to the faster-whisper ``whisper-<model>``)."""
    return Path(models_dir) / f"whisper-mlx-{model}"


def mlx_ready(path: Path, model: str | None = None) -> bool:
    """``config.json`` and the weights file are on disk (the files mlx_whisper's ``load_model`` reads)."""
    path = Path(path)
    spec = MLX_MODELS.get(model or "")
    names = (spec.weights,) if spec else ("weights.safetensors", "weights.npz")
    return (path / "config.json").is_file() and any((path / n).is_file() for n in names)


def download_mlx(
    model: str,
    dest_dir: Path,
    progress: Callable[[str], None] = print,
    connections: int = 8,
    write: Callable[[str], object] | None = None,
    fast: Callable[..., Path] | None = None,
) -> Path:
    """v0.1.35: download the pinned MLX Whisper weights of ``model`` into ``dest_dir/whisper-mlx-<model>`` with the
    same parallel, resumable, checksum-verified downloader as the faster-whisper models. Raises ``KeyError`` for a
    model with no MLX build (the app then uses faster-whisper on the CPU)."""
    from away.app import fastdl

    spec = MLX_MODELS[model]
    out = mlx_dir(dest_dir, model)
    if mlx_ready(out, model):
        progress(f"  ASR '{model}' (Apple GPU, MLX): already present")
        return out
    write = write or _stdout_writer()
    progress(f"  ASR '{model}' (Apple GPU, MLX): downloading ~{spec.size_mb} MB from Hugging Face ...")
    return Path(
        (fast or fastdl.download_model)(
            model,
            out,
            write,
            connections=connections,
            on_slow=slow_hint(model, progress),
            repo=spec.repo,
            revision=spec.revision,
            patterns=("config.json", spec.weights),
            required=(spec.weights,),
            expect={spec.weights: spec.sha256},
        )
    )


def model_path(models_dir: Path, name: str) -> Path | None:
    p = models_dir / name
    return p if p.exists() else None
