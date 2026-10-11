"""NVIDIA GPU pack: the CUDA library Whisper needs, downloaded once from PyPI into the user data dir.

The one-click installers stay CPU-sized. CTranslate2 4.8 (what faster-whisper runs on) is built with
``CUDA_DYNAMIC_LOADING=ON`` and ``WITH_CUDNN=OFF`` (``python/tools/prepare_build_environment_*.sh``):
the CUDA runtime is linked statically and the only library it loads at run time is **cuBLAS 12**
(``cublas64_12.dll`` / ``libcublas.so.12``, which pulls in ``cublasLt``). cuDNN is not needed.

:func:`install` downloads NVIDIA's official ``nvidia-cublas-cu12`` wheel (pinned to 12.8.4.1, the
version CTranslate2 4.8 was built against) with :mod:`away.app.fastdl` (parallel, resumable,
SHA-256 verified against the digest PyPI publishes, which must also match the digest pinned here),
extracts just the two libraries, and deletes the wheel. :func:`activate` puts them on the DLL search
path and preloads them before CTranslate2 asks for them by name (``LoadLibraryA`` / ``dlopen``), so
an already-loaded module is found. It also picks up pip-installed ``nvidia-cublas-cu12``.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away import paths

PACKAGE = "nvidia-cublas-cu12"
VERSION = "12.8.4.1"
# sha256 of each wheel as published on PyPI (https://pypi.org/pypi/nvidia-cublas-cu12/12.8.4.1/json)
PINNED_SHA256 = {
    "win_amd64": "47e9b82132fa8d2b4944e708049229601448aaad7e6f296f630f2d1a32de35af",
    "manylinux_2_27_x86_64": "8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142",
}
# wheel members to keep, in load order (cublasLt first: cublas depends on it)
MEMBERS = {
    "win32": ("nvidia/cublas/bin/cublasLt64_12.dll", "nvidia/cublas/bin/cublas64_12.dll"),
    "linux": ("nvidia/cublas/lib/libcublasLt.so.12", "nvidia/cublas/lib/libcublas.so.12"),
}
LICENSE_SUFFIX = ".dist-info/License.txt"
CUBLAS_NAME = {"win32": "cublas64_12.dll", "linux": "libcublas.so.12"}
PIP_HINT = f"pip install {PACKAGE}=={VERSION}"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"


@dataclass(frozen=True)
class Wheel:
    filename: str
    url: str
    size: int
    sha256: str


def _os(platform: str | None = None) -> str:
    plat = platform or sys.platform
    return "linux" if plat.startswith("linux") else plat


def platform_tag(platform: str | None = None, arch: str | None = None) -> str | None:
    """The wheel tag for this machine, or ``None`` where CTranslate2 has no CUDA build."""
    import platform as _p

    os_name = _os(platform)
    machine = (arch or _p.machine()).lower()
    if machine not in ("x86_64", "amd64"):
        return None
    return {"win32": "win_amd64", "linux": "manylinux_2_27_x86_64"}.get(os_name)


def pack_dir() -> Path:
    return paths.data_dir() / "gpu" / "cuda12"


def manifest_path(d: Path | None = None) -> Path:
    return (d or pack_dir()) / "manifest.json"


def read_manifest(d: Path | None = None) -> dict[str, Any] | None:
    try:
        m = json.loads(manifest_path(d).read_text(encoding="utf-8"))
        return m if isinstance(m, dict) else None
    except (OSError, ValueError):
        return None


def installed(d: Path | None = None) -> bool:
    """The pack is complete: a manifest whose every library is on disk with the recorded size."""
    d = d or pack_dir()
    m = read_manifest(d)
    if not m or not m.get("files"):
        return False
    try:
        return all((d / name).stat().st_size == int(meta["size"]) for name, meta in m["files"].items())
    except (OSError, KeyError, TypeError, ValueError):
        return False


def status(d: Path | None = None) -> dict[str, Any]:
    d = d or pack_dir()
    m = read_manifest(d) or {}
    ok = installed(d)
    size = sum(int(v.get("size", 0)) for v in (m.get("files") or {}).values()) if ok else 0
    return {
        "installed": ok,
        "dir": str(d),
        "package": m.get("package"),
        "version": m.get("version"),
        "size_mb": round(size / 1_000_000),
        "files": sorted((m.get("files") or {}).keys()) if ok else [],
        "pip_cublas": pip_cublas_dir() is not None,
    }


# ---------------------------------------------------------------- download
def fetch_wheel_info(tag: str, opener: Any = None, name: str = PACKAGE, version: str = VERSION) -> Wheel:
    """Find the wheel for ``tag`` in PyPI's JSON API; its digest must match the pinned one."""
    from away.app import fastdl

    opener = opener or fastdl.default_opener
    url = PYPI_JSON.format(name=name, version=version)
    with opener(fastdl._request(url), fastdl.TIMEOUT_S) as r:
        data = json.loads(r.read().decode("utf-8"))
    for f in data.get("urls", []):
        fn = str(f.get("filename", ""))
        if fn.endswith(f"-{tag}.whl"):
            sha = str((f.get("digests") or {}).get("sha256", ""))
            pinned = PINNED_SHA256.get(tag) if (name, version) == (PACKAGE, VERSION) else None
            if not sha or (pinned and sha != pinned):
                raise fastdl.ChecksumError(f"{fn}: PyPI digest {sha[:12]}... does not match the pinned digest")
            return Wheel(fn, str(f["url"]), int(f["size"]), sha)
    raise RuntimeError(f"{name} {version}: no wheel for {tag} on PyPI")


def extract(wheel_path: Path, out: Path, members: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Copy ``members`` (by path inside the wheel) flat into ``out``; returns {file name: {size, sha256}}."""
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(wheel_path) as z:
        names = set(z.namelist())
        missing = [m for m in members if m not in names]
        if missing:
            raise RuntimeError(f"{wheel_path.name}: missing {', '.join(missing)}")
        lic = next((n for n in names if n.endswith(LICENSE_SUFFIX)), None)
        for member in [*members, *([lic] if lic else [])]:
            target = out / ("NVIDIA-cuBLAS-LICENSE.txt" if member == lic else Path(member).name)
            tmp = target.with_name(target.name + ".tmp")
            h = hashlib.sha256()
            with z.open(member) as src, tmp.open("wb") as dst:
                for block in iter(lambda s=src: s.read(1 << 20), b""):  # type: ignore[misc]
                    h.update(block)
                    dst.write(block)
            tmp.replace(target)
            if member != lic:
                files[target.name] = {"size": target.stat().st_size, "sha256": h.hexdigest(), "member": member}
    return files


def install(
    write: Callable[[str], object] | None = None,
    progress: Callable[[str], None] = print,
    connections: int = 8,
    opener: Any = None,
    platform: str | None = None,
    arch: str | None = None,
    d: Path | None = None,
    fetch_info: Callable[[str], Wheel] | None = None,
) -> Path:
    """Download + verify + extract the GPU pack. Idempotent; an interrupted download resumes next time."""
    from away.app import fastdl, models

    os_name = _os(platform)
    tag = platform_tag(platform, arch)
    if tag is None:
        raise RuntimeError("the GPU pack is only available for Windows and Linux on x86-64 (NVIDIA CUDA)")
    d = d or pack_dir()
    if installed(d):
        progress(f"  GPU pack already installed in {d}")
        return d
    opener = opener or fastdl.default_opener
    whl = (fetch_info or (lambda t: fetch_wheel_info(t, opener)))(tag)
    progress(f"  GPU pack: downloading {whl.filename} (~{whl.size / 1_000_000:,.0f} MB, once) from PyPI ...")
    dl = d / ".download"
    write = write or models._stdout_writer()
    meter = fastdl.Meter(whl.size, write)
    ok = False
    try:
        path = fastdl.download_file(
            whl.url,
            dl / whl.filename,
            fastdl.RemoteFile(whl.filename, whl.size, sha256=whl.sha256),
            meter,
            connections=connections,
            opener=opener,
        )
        ok = True
    finally:
        meter.finish(ok)
    progress("  GPU pack: verified SHA-256, extracting cuBLAS ...")
    files = extract(path, d, MEMBERS[os_name])
    manifest = {
        "package": PACKAGE,
        "version": VERSION,
        "wheel": whl.filename,
        "wheel_sha256": whl.sha256,
        "files": files,
    }
    manifest_path(d).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.rmtree(dl, ignore_errors=True)
    total = sum(v["size"] for v in files.values())
    progress(f"  GPU pack installed: {len(files)} libraries, {total / 1_000_000:,.0f} MB in {d}")
    getattr(cublas_available, "cache_clear", lambda: None)()
    return d


def remove(d: Path | None = None) -> bool:
    d = d or pack_dir()
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    getattr(cublas_available, "cache_clear", lambda: None)()
    return not d.exists()


# ---------------------------------------------------------------- activation
def pip_cublas_dir(platform: str | None = None) -> Path | None:
    """``site-packages/nvidia/cublas/{bin,lib}`` from a pip-installed ``nvidia-cublas-cu12``."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("nvidia.cublas")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    sub = "bin" if _os(platform) == "win32" else "lib"
    for loc in spec.submodule_search_locations:
        p = Path(loc) / sub
        if p.is_dir():
            return p
    return None


def library_dirs(platform: str | None = None) -> list[Path]:
    dirs = []
    if installed():
        dirs.append(pack_dir())
    pip = pip_cublas_dir(platform)
    if pip is not None:
        dirs.append(pip)
    return dirs


_activated: list[str] = []
_dll_handles: list[Any] = []  # keep os.add_dll_directory cookies + CDLL objects alive


def activate(platform: str | None = None, dirs: list[Path] | None = None, loader: Any = None) -> list[str]:
    """Make cuBLAS findable by CTranslate2: DLL search path + preload. Returns the libraries loaded."""
    os_name = _os(platform)
    if os_name not in MEMBERS:
        return []
    loaded: list[str] = []
    for d in dirs if dirs is not None else library_dirs(platform):
        names = [Path(m).name for m in MEMBERS[os_name]]
        if not all((d / n).exists() for n in names):
            continue
        if os_name == "win32":
            add = getattr(os, "add_dll_directory", None)
            if add is not None:
                with contextlib.suppress(OSError):
                    _dll_handles.append(add(str(d)))
            if str(d) not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        for n in names:
            full = str(d / n)
            if full in _activated:
                loaded.append(full)
                continue
            try:
                if loader is not None:
                    _dll_handles.append(loader(full))
                elif os_name == "win32":  # pragma: no cover - Windows only
                    _dll_handles.append(ctypes.WinDLL(full))  # type: ignore[attr-defined]
                else:
                    _dll_handles.append(ctypes.CDLL(full, mode=getattr(os, "RTLD_GLOBAL", 0)))
            except OSError:
                break
            _activated.append(full)
            loaded.append(full)
        if len(loaded) >= len(names):
            break
    return loaded


@functools.lru_cache(maxsize=1)
def cublas_available(platform: str | None = None) -> bool:
    """Can cuBLAS 12 be loaded the way CTranslate2 loads it (by name), after :func:`activate`?"""
    os_name = _os(platform)
    if os_name not in CUBLAS_NAME:
        return False
    activate(platform)
    try:
        if os_name == "win32":  # pragma: no cover - Windows only
            ctypes.WinDLL(CUBLAS_NAME[os_name])  # type: ignore[attr-defined]
        else:
            ctypes.CDLL(CUBLAS_NAME[os_name])
        return True
    except OSError:
        return False


def pip_instructions() -> str:
    return (
        f"Install the CUDA library Whisper needs into this Python environment:\n"
        f"  {PIP_HINT}\n"
        "(CTranslate2 4.8 needs only cuBLAS 12. Older CTranslate2 builds also need:"
        " pip install nvidia-cudnn-cu12==9.*)\n"
        "Project Away finds it automatically; no LD_LIBRARY_PATH needed."
    )
