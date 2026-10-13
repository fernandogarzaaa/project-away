"""Downloads for catalog models (v0.1.24 model refresh): one file (``url`` + ``sha256``) or a pinned bundle
(``files``: ``[{"path", "url", "sha256", "size"}]``), into ``<data>/models/<model id>/``.

Every file is sha256-checked (``away.app.fastdl``: resumable, atomic rename); a bundle is marked complete only when
every file is there and verified. Archives (``.zip``) are unpacked after the check with path checks. Nothing here
is ever bundled with the app.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DONE = "complete.json"


@dataclass(frozen=True)
class FileSpec:
    path: str
    url: str
    sha256: str
    size: int

    @staticmethod
    def of(item: dict[str, Any]) -> FileSpec:
        return FileSpec(str(item["path"]), str(item["url"]), str(item["sha256"]).lower(), int(item.get("size") or 0))


class FetchError(RuntimeError):
    pass


def files_of(model: Any) -> list[FileSpec]:
    """The files a model needs: its ``files`` bundle, or its single ``url``."""
    files = list(getattr(model, "files", None) or [])
    if files:
        return [FileSpec.of(f) for f in files]
    url = str(getattr(model, "url", "") or "")
    if not url:
        return []
    name = url.rsplit("/", 1)[-1].split("?", 1)[0] or "model.bin"
    return [FileSpec(name, url, str(model.sha256).lower(), int(getattr(model, "size", 0) or 0))]


def model_dir(mid: str, data_dir: Path | None = None) -> Path:
    from away.app.model_registry import models_dir

    return models_dir(data_dir) / mid


def _safe(rel: str) -> bool:
    p = PurePosixPath(rel)
    return bool(rel) and not (p.is_absolute() or ".." in p.parts or "\\" in rel or ":" in rel)


def have(model: Any, data_dir: Path | None = None) -> bool:
    if getattr(model, "bundled", False):
        return True
    specs = files_of(model)
    if not specs:
        return False
    d = model_dir(model.id, data_dir)
    try:
        info = json.loads((d / DONE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    want = sorted(s.sha256 for s in specs)
    return bool(info.get("sha256") == want) and all((d / s.path).exists() for s in specs)


def fetch(
    model: Any,
    data_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    downloader: Callable[..., Path] | None = None,
) -> Path:
    """Download (and verify) every file of ``model``; returns its folder."""
    from away.app import fastdl
    from away.app.optional_models import _ProgressMeter

    specs = files_of(model)
    if not specs:
        raise FetchError(f"{getattr(model, 'title', model.id)} has nothing to download here")
    for s in specs:
        if not _safe(s.path) or len(s.sha256) != 64:
            raise FetchError(f"{model.id}: bad file entry {s.path!r}")
    d = model_dir(model.id, data_dir)
    if have(model, data_dir):
        return d
    total = sum(s.size for s in specs) or 1
    pm = _ProgressMeter(total, progress, lambda _s: None)
    ok = False
    try:
        for s in specs:
            dest = d / s.path
            if dest.exists() and fastdl.file_digest(dest, "sha256") == s.sha256:
                pm.meter.add(s.size)
                continue
            (downloader or fastdl.download_file)(
                s.url, dest, fastdl.RemoteFile(s.path, s.size, sha256=s.sha256), pm.meter
            )
            if fastdl.file_digest(dest, "sha256") != s.sha256:
                dest.unlink(missing_ok=True)
                raise FetchError(f"{s.path} did not match its checksum; it was removed")
        ok = True
    except FetchError:
        raise
    except Exception as exc:
        raise FetchError(f"The download stopped ({type(exc).__name__}). Check your internet and try again.") from exc
    finally:
        pm.meter.finish(ok)
    for s in specs:
        if s.path.endswith(".zip"):
            unzip(d / s.path, d / s.path[: -len(".zip")])
    (d / DONE).write_text(json.dumps({"sha256": sorted(s.sha256 for s in specs)}), encoding="utf-8")
    return d


def unzip(zf_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zf_path) as zf:
        for info in zf.infolist():
            if not _safe(info.filename):
                raise FetchError(f"{zf_path.name} has an unsafe path; it was not unpacked")
        zf.extractall(dest)


def find(model_folder: Path, pattern: str) -> Path | None:
    hits = sorted(model_folder.glob(f"**/{pattern}"))
    return hits[0] if hits else None


# ----------------------------------------------------------------------------------------------- one click
def cleanup_key(model: Any) -> str | None:
    """A cleanup model handled by :mod:`away.app.optional_models` (downloaded where Smart cleanup looks)."""
    from away.app import optional_models as om

    key = str((getattr(model, "set", None) or {}).get("voice.cleanup.model", "") or "")
    try:
        om.get(key)
    except KeyError:
        return None
    return key


def engine_archive(model: Any, data_dir: Path | None = None) -> Any:
    from away.app import engines

    name = str(getattr(model, "engine", "") or "")
    if not name:
        return None
    if name == "nemo-speech":
        from away.voice.asr_next import nemotron_plan

        return engines.archive_for(name, str(nemotron_plan("auto", data_dir)["variant"]))
    return engines.archive_for(name)


def ready(model: Any, data_dir: Path | None = None) -> bool:
    """Everything this model needs is on this PC (its files and its engine)."""
    from away.app import engines
    from away.app import optional_models as om
    from away.app.model_registry import models_dir

    if getattr(model, "bundled", False):
        return True
    key = cleanup_key(model)
    if key is not None:
        return om.installed(key, models_dir(data_dir))
    if files_of(model) and not have(model, data_dir):
        return False
    if getattr(model, "engine", ""):
        mod = engines.MODULES.get(str(model.engine))
        if not ((mod and engines.pip_installed(mod)) or engines.installed(engine_archive(model, data_dir), data_dir)):
            return False
    return bool(files_of(model)) or bool(getattr(model, "engine", ""))


def download(
    model: Any, data_dir: Path | None = None, progress: Callable[[int, int], None] | None = None
) -> Path | None:
    """One click: the model's files (verified) and the engine it runs in. Returns the model folder."""
    from away.app import engines
    from away.app import optional_models as om
    from away.app.model_registry import models_dir

    key = cleanup_key(model)
    if key is not None:
        om.install(key, models_dir(data_dir), progress=progress)
        return models_dir(data_dir)
    folder = fetch(model, data_dir, progress) if files_of(model) else None
    a = engine_archive(model, data_dir)
    if a is not None:
        mod = engines.MODULES.get(a.engine)
        if not (mod and engines.pip_installed(mod)):
            engines.install(a, data_dir, progress)
    return folder
