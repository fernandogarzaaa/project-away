"""Optional model downloads, listed as data (Settings > Performance > Models, ``away gaze model``).

Each :class:`OptionalModel` says what it is, how big, under which license, and where it comes from. Downloads
use :func:`away.app.fastdl.download_file` (resumable ``.part`` + sidecar, parallel ranges for big files,
sha256-verified, atomic rename). A model whose license restricts use (``restricted``) is only downloaded
after an explicit opt-in (:class:`LicenseNotAccepted` otherwise); nothing here is ever bundled in the
installers. Add a new entry to :data:`MODELS` to list another model (e.g. a personal gesture classifier).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GAZE360_LICENSE_URL = "https://github.com/erkil1452/gaze360/blob/master/LICENSE.md"
GAZE_NOTICE = (
    "Research-only weights: the code (github.com/yakhyo/gaze-estimation, an L2CS-Net implementation) is MIT, but "
    "the network was trained on the Gaze360 dataset, whose license allows research use only and excludes "
    "models trained on it from commercial applications. Download it only if your use is non-commercial "
    f"research/personal experimentation and you accept those terms ({GAZE360_LICENSE_URL}). Project Away never "
    "bundles it; it is fetched from the author's GitHub release."
)


@dataclass(frozen=True)
class OptionalModel:
    key: str  # config value / CLI name (gaze.model: l2cs)
    name: str  # what the user sees
    filename: str  # under the models folder
    url: str
    sha256: str
    size_bytes: int
    license: str  # short license line
    notice: str  # longer note shown before an opt-in download
    restricted: bool  # needs an explicit opt-in before downloading
    kind: str  # gaze | gesture | ...
    purpose: str
    cost: str  # measured per-run cost, for the list

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_000_000


_YAKHYO = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/"

MODELS: tuple[OptionalModel, ...] = (
    OptionalModel(
        key="l2cs",
        name="Gaze network: L2CS-Net (ResNet-18)",
        filename="gaze_l2cs_resnet18.onnx",
        url=_YAKHYO + "resnet18_gaze.onnx",
        sha256="404fec1efd07ff49f981e47f461c20c2627119e465ec441bbd1c067d3f16e657",
        size_bytes=45_066_134,
        license="code MIT; weights trained on Gaze360 (research-only)",
        notice=GAZE_NOTICE,
        restricted=True,
        kind="gaze",
        purpose="learned gaze direction from the face crop (extra calibration features); best with a GPU",
        cost="~117 ms per run on 2 CPU cores (448x448 crop; measured on ARM Neoverse-V2); GPU untested",
    ),
    OptionalModel(
        key="mobilegaze",
        name="Gaze network: MobileGaze (MobileOne-S0)",
        filename="gaze_mobileone_s0.onnx",
        url=_YAKHYO + "mobileone_s0_gaze.onnx",
        sha256="8b4fdc4e3da44733c9a82e7776b411e4a39f94e8e285aee0fc85a548a55f7d9f",
        size_bytes=4_974_521,
        license="code MIT; weights trained on Gaze360 (research-only)",
        notice=GAZE_NOTICE,
        restricted=True,
        kind="gaze",
        purpose="the same, ~5x lighter: the CPU choice",
        cost="~21 ms per run on 2 CPU cores (448x448 crop; measured on ARM Neoverse-V2)",
    ),
)


class LicenseNotAccepted(PermissionError):
    """A restricted model was requested without the explicit opt-in."""


def get(key: str) -> OptionalModel:
    for m in MODELS:
        if m.key == key:
            return m
    raise KeyError(f"unknown model {key!r}; choose from {', '.join(m.key for m in MODELS)}")


def path_of(key: str, models_dir: Path) -> Path:
    return models_dir / get(key).filename


def installed(key: str, models_dir: Path) -> bool:
    """Present with the right size (a full checksum on every status call would read 45 MB)."""
    p = path_of(key, models_dir)
    try:
        return p.is_file() and p.stat().st_size == get(key).size_bytes
    except OSError:
        return False


@dataclass
class Status:
    model: OptionalModel
    installed: bool
    partial_bytes: int  # bytes of an interrupted download that a retry resumes

    def line(self) -> str:
        m = self.model
        state = "installed" if self.installed else "not installed"
        if not self.installed and self.partial_bytes:
            state += f" ({self.partial_bytes / 1_000_000:.1f} MB downloaded, resumes)"
        return f"{m.name} [{m.key}]: {state} - {m.size_mb:.1f} MB - {m.license}"


def status(models_dir: Path, kind: str | None = None) -> list[Status]:
    out = []
    for m in MODELS:
        if kind is not None and m.kind != kind:
            continue
        part = models_dir / (m.filename + ".part")
        try:
            pb = part.stat().st_size if part.exists() else 0
        except OSError:
            pb = 0
        out.append(Status(m, installed(m.key, models_dir), pb))
    return out


Progress = Callable[[int, int], None]  # (bytes done, total bytes)


class _ProgressMeter:
    """A fastdl Meter whose updates are forwarded as (done, total) to a GUI/CLI callback."""

    def __init__(self, total: int, progress: Progress | None, write: Callable[[str], object]) -> None:
        from away.app import fastdl

        outer = self

        class M(fastdl.Meter):
            def add(self, n: int) -> None:
                super().add(n)
                if outer.progress is not None:
                    outer.progress(self.done, self.total)

        self.progress = progress
        self.meter = M(total, write)


def install(
    key: str,
    models_dir: Path,
    accept_license: bool = False,
    progress: Progress | None = None,
    write: Callable[[str], object] | None = None,
    downloader: Callable[..., Path] | None = None,
) -> Path:
    """Download (or resume) a model; verified by sha256 before it is moved into place."""
    from away.app import fastdl

    m = get(key)
    if m.restricted and not accept_license:
        raise LicenseNotAccepted(f"{m.name}: {m.notice}")
    dest = models_dir / m.filename
    if installed(key, models_dir):
        if progress is not None:
            progress(m.size_bytes, m.size_bytes)
        return dest
    rf = fastdl.RemoteFile(m.filename, m.size_bytes, sha256=m.sha256)
    pm = _ProgressMeter(m.size_bytes, progress, write or (lambda _s: None))
    ok = False
    try:
        (downloader or fastdl.download_file)(m.url, dest, rf, pm.meter)
        ok = True
    finally:
        pm.meter.finish(ok)
    _write_receipt(models_dir, m, accept_license)
    if progress is not None:
        progress(m.size_bytes, m.size_bytes)
    return dest


def _write_receipt(models_dir: Path, m: OptionalModel, accepted: bool) -> None:
    """Note next to the model when (and that) its license was accepted."""
    import json

    receipt = models_dir / (m.filename + ".license.json")
    with contextlib.suppress(OSError):
        receipt.write_text(
            json.dumps({"model": m.key, "license": m.license, "accepted": accepted, "at": time.time()}),
            encoding="utf-8",
        )


def remove(key: str, models_dir: Path) -> bool:
    m = get(key)
    gone = False
    for name in (m.filename, m.filename + ".part", m.filename + ".part.json", m.filename + ".license.json"):
        p = models_dir / name
        if p.exists():
            try:
                p.unlink()
                gone = gone or name == m.filename
            except OSError:
                pass
    return gone


def resolve_gaze_model(
    choice: str, models_dir: Path, gpu: bool | None = None, gpu_check: Callable[[], bool] | None = None
) -> str:
    """``gaze.model`` -> the backend to run now: ``iris`` | ``l2cs`` | ``mobilegaze``.

    ``auto``: l2cs when it is downloaded and a GPU provider is usable, else mobilegaze when downloaded, else
    iris. An explicit choice whose file is missing falls back to iris (the app keeps working)."""
    if choice in ("l2cs", "mobilegaze"):
        return choice if installed(choice, models_dir) else "iris"
    if choice != "auto":
        return "iris"
    if installed("l2cs", models_dir):
        if gpu is None:
            if gpu_check is None:
                from away.face.gaze_net import gpu_available

                gpu_check = gpu_available
            gpu = bool(gpu_check())
        if gpu:
            return "l2cs"
    if installed("mobilegaze", models_dir):
        return "mobilegaze"
    return "iris"


def describe(models_dir: Path) -> list[dict[str, Any]]:
    """Rows for the Settings list: name, size, license, status, notice."""
    return [
        {
            "key": s.model.key,
            "name": s.model.name,
            "size": f"{s.model.size_mb:.1f} MB",
            "license": s.model.license,
            "status": "installed" if s.installed else "not installed",
            "installed": s.installed,
            "restricted": s.model.restricted,
            "notice": s.model.notice,
            "purpose": s.model.purpose,
            "cost": s.model.cost,
        }
        for s in status(models_dir)
    ]
