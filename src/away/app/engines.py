"""Runtime engines for the optional 2026-10 models (v0.1.24 model refresh). Nothing here is bundled.

Each engine is a pinned, sha256-verified file fetched on request into ``<data>/engines/<name>-<version>-<tag>/``:

* ``moonshine`` - the ``moonshine-voice`` wheel (MIT; its native ``libmoonshine`` runs Moonshine v2 streaming
  speech-to-text on the CPU). Only the ``moonshine_voice`` package is unpacked.
* ``livekit-wakeword`` - the ``livekit-wakeword`` wheel (Apache-2.0; pure Python + the bundled Apache-2.0 mel and
  speech-embedding ONNX files). It needs only numpy and onnxruntime, which the app already ships.
* ``nemo-speech`` - NVIDIA NeMo-Speech.cpp (Apache-2.0) release archives: the ``nemo-speech`` CLI/server that runs
  Nemotron 3.5 ASR Streaming. Variants: ``cuda`` (NVIDIA), ``vulkan`` (any GPU), ``cpu``.

Sizes and sha256 come from the publishers (PyPI digests, the GitHub release asset digests) and are re-checked
after every download; an engine whose check fails is removed again. A pip-installed copy always wins.

Licence notes: espeak-ng and Piper are GPL-3.0 and are never part of an engine here.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import platform as _platform
import shutil
import sys
import tarfile
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MARKER = "engine.json"
PYPI = "https://files.pythonhosted.org/packages/"
NEMO_VERSION = "0.2.0"
NEMO_RELEASE = f"https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{NEMO_VERSION}/"


@dataclass(frozen=True)
class Archive:
    engine: str  # moonshine | livekit-wakeword | nemo-speech
    version: str
    tag: str  # platform (and variant) tag used in the folder name
    filename: str
    url: str
    size: int
    sha256: str
    kind: str  # wheel | zip | tgz
    keep: tuple[str, ...] = ()  # wheel: top-level folders to unpack (nothing else)
    license: str = ""


class EngineError(RuntimeError):
    """A friendly sentence for Settings / the agent."""


def _wheel(engine: str, version: str, tag: str, path: str, size: int, sha: str, keep: str, lic: str) -> Archive:
    return Archive(engine, version, tag, path.rsplit("/", 1)[1], PYPI + path, size, sha, "wheel", (keep,), lic)


def _nemo(tag: str, ext: str, size: int, sha: str) -> Archive:
    name = f"nemo-speech-{NEMO_VERSION}-{tag}.{ext}"
    kind = "zip" if ext == "zip" else "tgz"
    return Archive("nemo-speech", NEMO_VERSION, tag, name, NEMO_RELEASE + name, size, sha, kind, (), "Apache-2.0")


_MS = "moonshine-voice"
_MS_LIC = "MIT (moonshine-voice library and the English streaming models)"
ARCHIVES: dict[str, dict[tuple[str, str], Archive]] = {
    "moonshine": {
        ("win32", "amd64"): _wheel(
            _MS,
            "0.1.5",
            "win_amd64",
            "ea/05/ad3457baa56170a13829c98327598a4d99d6cd4bdff8ae59df4c47ebdd11/moonshine_voice-0.1.5-py3-none-win_amd64.whl",
            16_542_073,
            "7fc9a84e827360ddb85c2735f44a1502c5575c27dcb476a2555646cd20a70e9d",
            "moonshine_voice/",
            _MS_LIC,
        ),
        ("linux", "x86_64"): _wheel(
            _MS,
            "0.1.5",
            "linux_x86_64",
            "a9/9d/228f738b48e0e7cc1de97c3f842b5470c6a3c5f7a0bffbf13c3d00eefb87/"
            "moonshine_voice-0.1.5-py3-none-manylinux_2_34_x86_64.whl",
            19_897_307,
            "1ed9e0ccf94be4845d69e7fa862dcf4c8b5801a758796966494c694f48184496",
            "moonshine_voice/",
            _MS_LIC,
        ),
        ("linux", "aarch64"): _wheel(
            _MS,
            "0.1.5",
            "linux_aarch64",
            "23/1c/cb821747b2a7b40216dd576a62d1ebc77d04fab97064efc97fa460f137ed/"
            "moonshine_voice-0.1.5-py3-none-manylinux_2_31_aarch64.whl",
            18_505_651,
            "605300afe33eb7920301e73f11e0ea7493770eb655f389ba65be4aa1db8dedd6",
            "moonshine_voice/",
            _MS_LIC,
        ),
        ("darwin", "arm64"): _wheel(
            _MS,
            "0.1.5",
            "macos_arm64",
            "86/2a/9eca2f000e4990315d18aa360c6eb679417b29252e6307c273bcbef38461/"
            "moonshine_voice-0.1.5-py3-none-macosx_15_0_arm64.whl",
            18_589_534,
            "9866b1956d12fff11923ad42f1140bcb314bf05bf8d8576fb7b3aa819ec40e51",
            "moonshine_voice/",
            _MS_LIC,
        ),
    },
    "livekit-wakeword": {
        ("any", "any"): _wheel(
            "livekit-wakeword",
            "0.3.0",
            "any",
            "0b/33/cc1682755d3e626faefa1def3b761f6b6826a0498a514246b54bd5e9ecbc/livekit_wakeword-0.3.0-py3-none-any.whl",
            1_917_287,
            "6bb00b832c21720ebb94410e005854740b2d36b20970c7cecc74b5e889d94f4b",
            "livekit/",
            "Apache-2.0",
        ),
    },
    # NeMo-Speech.cpp v0.2.0 (2026-10-02); sizes/sha256 = the release's asset digests
    "nemo-speech": {
        ("win32", "amd64-cuda"): _nemo(
            "windows-x86_64-cuda",
            "zip",
            102_183_465,
            "9ffe5e842c39a7d3211f3b2c3b3aa1bca8c0b2455d96d961ed5f22ec5108f98d",
        ),
        ("win32", "amd64-vulkan"): _nemo(
            "windows-x86_64-vulkan",
            "zip",
            19_302_464,
            "edf15a04ba98740aa0ab75ae4681ac7ad0a488b22b35ce2cbebd10590bb1c1e1",
        ),
        ("win32", "amd64-cpu"): _nemo(
            "windows-x86_64-cpu", "zip", 5_528_418, "82c80451086e86194aba9af19f800da77cd66d1c28d993d0efceffd6cdb49bb8"
        ),
        ("linux", "x86_64-cuda"): _nemo(
            "linux-x86_64-cuda",
            "tar.gz",
            105_010_050,
            "aef573fcc6c2c759401d2ed03e5ca9e401ed23b0e38cb852d917080c1e2d4515",
        ),
        ("linux", "x86_64-cpu"): _nemo(
            "linux-x86_64-cpu", "tar.gz", 6_683_477, "f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507"
        ),
        ("linux", "aarch64-cpu"): _nemo(
            "linux-aarch64-cpu", "tar.gz", 6_309_769, "ac50694d4729f9eae390a9851d38f51269d9af9829b535325539090a3b95c787"
        ),
        ("darwin", "arm64-metal"): _nemo(
            "macos-aarch64-metal",
            "tar.gz",
            5_031_510,
            "5cb02ba7c04f0b5585ce5cde9c830be5f0b7dfb4c83083c500109c381b4f2da9",
        ),
    },
}
# the Python module each wheel engine provides (a pip-installed copy wins)
MODULES = {"moonshine": "moonshine_voice", "livekit-wakeword": "livekit.wakeword"}


def platform_key(system: str | None = None, machine: str | None = None) -> tuple[str, str]:
    s = system or sys.platform
    s = "win32" if s.startswith("win") else "darwin" if s == "darwin" else "linux" if s.startswith("linux") else s
    m = (machine or _platform.machine() or "").lower()
    if m in ("x86_64", "amd64"):
        m = "amd64" if s == "win32" else "x86_64"
    elif m in ("arm64", "aarch64"):
        m = "arm64" if s == "darwin" else "aarch64"
    return s, m


def archive_for(
    engine: str, variant: str = "", system: str | None = None, machine: str | None = None
) -> Archive | None:
    """The pinned archive of ``engine`` for this computer (``variant``: cuda | vulkan | cpu | metal for nemo-speech)."""
    table = ARCHIVES.get(engine) or {}
    if ("any", "any") in table:
        return table[("any", "any")]
    s, m = platform_key(system, machine)
    if engine == "nemo-speech":
        return table.get((s, f"{m}-{variant or 'cpu'}"))
    return table.get((s, m))


def variants(engine: str, system: str | None = None, machine: str | None = None) -> list[str]:
    s, m = platform_key(system, machine)
    return [k[1].split("-", 1)[1] for k in (ARCHIVES.get(engine) or {}) if k[0] == s and k[1].startswith(m + "-")]


def engines_root(data_dir: Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir) / "engines"
    from away import paths

    return Path(paths.data_dir()) / "engines"


def engine_dir(a: Archive, data_dir: Path | None = None) -> Path:
    return engines_root(data_dir) / f"{a.engine}-{a.version}-{a.tag}"


def installed(a: Archive | None, data_dir: Path | None = None) -> bool:
    if a is None:
        return False
    d = engine_dir(a, data_dir)
    try:
        info = json.loads((d / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(info.get("sha256") == a.sha256 and info.get("checked") is True)


def pip_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def activate(engine: str, data_dir: Path | None = None) -> bool:
    """Make a downloaded wheel engine importable (appended to sys.path: a pip copy still wins)."""
    mod = MODULES.get(engine)
    if mod and pip_installed(mod):
        return True
    a = archive_for(engine)
    if a is None or a.kind != "wheel" or not installed(a, data_dir):
        return False
    d = str(engine_dir(a, data_dir))
    if d not in sys.path:
        sys.path.append(d)
        importlib.invalidate_caches()
    return True


def binary(a: Archive, data_dir: Path | None = None, name: str = "nemo-speech") -> Path | None:
    """The executable inside an unpacked archive engine (``bin/nemo-speech[.exe]``)."""
    d = engine_dir(a, data_dir)
    for exe in (name + ".exe", name):
        hits = sorted(d.glob(f"**/bin/{exe}")) or sorted(d.glob(f"**/{exe}"))
        if hits:
            return hits[0]
    return None


def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return not (p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name)


def unpack(a: Archive, file: Path, dest: Path) -> None:
    """Unpack ``file`` into ``dest`` (atomically replaced). Wheels keep only ``a.keep``; paths are checked."""
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    damaged = EngineError(f"The {a.engine} download looks damaged. Press Download to try again.")
    try:
        if a.kind in ("wheel", "zip"):
            with zipfile.ZipFile(file) as zf:
                members = zf.infolist()
                if any(not _safe_name(i.filename) for i in members):
                    raise damaged
                if a.kind == "wheel":
                    members = [i for i in members if i.filename.startswith(a.keep)]
                    if not members:
                        raise damaged
                for info in members:
                    zf.extract(info, tmp)
                    mode = (info.external_attr >> 16) & 0o777
                    if mode and not info.is_dir():
                        with contextlib.suppress(OSError):
                            (tmp / info.filename).chmod(mode | 0o600)
        else:
            with tarfile.open(file, "r:gz") as tf:
                for m in tf.getmembers():
                    if not _safe_name(m.name) or m.islnk() or (m.issym() and not _safe_name(m.linkname)):
                        raise damaged
                    if m.isdev():
                        raise damaged
                for m in tf.getmembers():
                    tf.extract(m, tmp, set_attrs=False)
                    target = tmp / m.name
                    if m.isfile() and m.mode & 0o111:
                        with contextlib.suppress(OSError):
                            target.chmod(0o755)
    except (zipfile.BadZipFile, tarfile.TarError, zlib.error, EOFError, OSError) as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise damaged from exc
    except EngineError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)


def install(
    a: Archive,
    data_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    downloader: Callable[..., Path] | None = None,
    checker: Callable[[Path], tuple[bool, str]] | None = None,
) -> Path:
    """Download (verified), unpack and check an engine. Returns its folder; :class:`EngineError` otherwise."""
    from away.app import fastdl
    from away.app.optional_models import _ProgressMeter

    dest = engine_dir(a, data_dir)
    if installed(a, data_dir):
        return dest
    dl = engines_root(data_dir) / ".downloads"
    dl.mkdir(parents=True, exist_ok=True)
    f = dl / a.filename
    pm = _ProgressMeter(a.size, progress, lambda _s: None)
    ok = False
    try:
        (downloader or fastdl.download_file)(a.url, f, fastdl.RemoteFile(a.filename, a.size, sha256=a.sha256), pm.meter)
        ok = True
    except Exception as exc:
        raise EngineError(
            f"The {a.engine} download stopped ({type(exc).__name__}). Check your internet and try again."
        ) from exc
    finally:
        pm.meter.finish(ok)
    if fastdl.file_digest(f, "sha256") != a.sha256:
        f.unlink(missing_ok=True)
        raise EngineError(f"The {a.engine} download looks damaged. Press Download to try again.")
    unpack(a, f, dest)
    marker: dict[str, Any] = {"engine": a.engine, "version": a.version, "sha256": a.sha256, "checked": False}
    (dest / MARKER).write_text(json.dumps(marker), encoding="utf-8")
    good, detail = (checker or (lambda d: (True, "")))(dest)
    if not good:
        shutil.rmtree(dest, ignore_errors=True)
        raise EngineError(f"{a.engine} could not start on this computer ({detail[:200]}). Nothing was changed.")
    marker["checked"] = True
    (dest / MARKER).write_text(json.dumps(marker), encoding="utf-8")
    f.unlink(missing_ok=True)
    return dest


def remove(a: Archive, data_dir: Path | None = None) -> bool:
    d = engine_dir(a, data_dir)
    gone = d.exists()
    shutil.rmtree(d, ignore_errors=True)
    return gone


def missing_modules(names: tuple[str, ...], find: Callable[[str], Any] = importlib.util.find_spec) -> list[str]:
    out = []
    for n in names:
        try:
            if find(n) is None:
                out.append(n)
        except (ImportError, ValueError):
            out.append(n)
    return out


def status(engine: str, variant: str = "", data_dir: Path | None = None) -> dict[str, Any]:
    """``{"engine", "available", "installed", "pip", "size_mb", "license"}`` for Settings and the benchmark."""
    a = archive_for(engine, variant)
    mod = MODULES.get(engine)
    pip = bool(mod and pip_installed(mod))
    return {
        "engine": engine,
        "variant": variant,
        "available": a is not None,
        "installed": pip or installed(a, data_dir),
        "pip": pip,
        "size_mb": round(a.size / 1e6, 1) if a else 0,
        "license": a.license if a else "",
    }
