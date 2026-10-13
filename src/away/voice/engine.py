"""The Smart cleanup engine for the installed app (v0.1.20): a pinned, sha256-verified llama-cpp-python wheel.

The installers do not bundle llama-cpp-python. Instead Settings > Voice > Smart cleanup's **Download** button
fetches the official prebuilt CPU wheel for this computer from the llama-cpp-python GitHub release (one file,
8-27 MB), checks its size and sha256 against the values pinned below, and unpacks only the ``llama_cpp``
package (plus the bundled OpenMP library on Linux) into ``<data>/engines/llama-cpp-python-<version>-<tag>/``.
:func:`activate` appends that folder to ``sys.path`` before ``llama_cpp`` is imported.

Why this is low risk: these wheels are ``py3-none`` (llama-cpp-python talks to llama.cpp through ctypes, so
they do not depend on the Python version), the Windows DLLs only need the system Visual C++ runtime the app
already relies on (no ``msvcp140.dll`` is unpacked), and a fresh engine is first loaded in a *separate*
process (``away cleanup-engine check``), so a broken native library can never take the app down. When the
check fails the engine is removed again and Settings keeps the friendly "Learn how" / Ollama path. A
``pip install llama-cpp-python`` copy, when present, always wins (it is found before the appended folder).

Python dependencies the wheel needs at import time (numpy, typing_extensions, diskcache, jinja2) ship in the
installers; :func:`missing_dependencies` names any that are missing in a pip install.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import platform as _platform
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

VERSION = "0.3.36"
RELEASE_URL = f"https://github.com/abetlen/llama-cpp-python/releases/download/v{VERSION}/"
LICENSE = "MIT (llama-cpp-python and llama.cpp)"
RUNTIME_DEPS = ("numpy", "typing_extensions", "diskcache", "jinja2")
KEEP_TOP = ("llama_cpp/", "llama_cpp_python.libs/")  # nothing else from the wheel (no headers, no import libs)
MARKER = "engine.json"


@dataclass(frozen=True)
class Wheel:
    tag: str  # short platform tag for the folder name
    filename: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return RELEASE_URL + self.filename


# Official CPU wheels of llama-cpp-python v0.3.36 (sizes and sha256 from the GitHub release's asset digests,
# re-checked by downloading each file). CUDA builds are not offered: they need a matching CUDA runtime.
WHEELS: dict[tuple[str, str], Wheel] = {
    ("win32", "amd64"): Wheel(
        "win_amd64",
        f"llama_cpp_python-{VERSION}-py3-none-win_amd64.whl",
        7_767_370,
        "ae5e88a2cf464e5a2dde7c8898c2d5c71cc24d03cac6396bf7509456bde0b9fb",
    ),
    ("darwin", "arm64"): Wheel(
        "macos_arm64",
        f"llama_cpp_python-{VERSION}-py3-none-macosx_11_0_arm64.whl",
        24_928_299,
        "1bec5e1d9fc7c4919aaece647bbd3e1a7f36ac8fd2714c8cda7e6991a544557b",
    ),
    ("linux", "x86_64"): Wheel(
        "linux_x86_64",
        f"llama_cpp_python-{VERSION}-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        27_077_498,
        "c8a3e98093e950d7a770f794842e36f2db82b2933e8762ed7b5ffd435011e8e4",
    ),
    ("linux", "aarch64"): Wheel(
        "linux_aarch64",
        f"llama_cpp_python-{VERSION}-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        24_942_850,
        "410bdf3cf66478f55de0ad51dd6ab241424591bdd36aff086b000d155ac182c6",
    ),
}


class EngineInstallError(RuntimeError):
    """A friendly sentence for Settings."""


def platform_key(system: str | None = None, machine: str | None = None) -> tuple[str, str]:
    s = system or sys.platform
    s = "win32" if s.startswith("win") else "darwin" if s == "darwin" else "linux" if s.startswith("linux") else s
    m = (machine or _platform.machine() or "").lower()
    m = {"x86_64": "amd64" if s == "win32" else "x86_64", "amd64": "amd64" if s == "win32" else "x86_64"}.get(m, m)
    m = "arm64" if (s == "darwin" and m in ("arm64", "aarch64")) else m
    return s, m


def wheel_for(system: str | None = None, machine: str | None = None) -> Wheel | None:
    w = WHEELS.get(platform_key(system, machine))
    return w if w is not None and w.size > 0 and len(w.sha256) == 64 else None


def engines_dir(data_dir: Path | None = None) -> Path:
    from away import paths

    return Path(data_dir) if data_dir is not None else paths.data_dir() / "engines"


def engine_dir(wheel: Wheel, data_dir: Path | None = None) -> Path:
    return engines_dir(data_dir) / f"llama-cpp-python-{VERSION}-{wheel.tag}"


def pip_installed() -> bool:
    """A llama-cpp-python installed with pip (not our downloaded copy)."""
    try:
        spec = importlib.util.find_spec("llama_cpp")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.origin:
        return False
    return engines_dir().resolve() not in Path(spec.origin).resolve().parents


def installed(data_dir: Path | None = None, wheel: Wheel | None = None) -> bool:
    w = wheel or wheel_for()
    if w is None:
        return False
    d = engine_dir(w, data_dir)
    try:
        info = json.loads((d / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        info.get("sha256") == w.sha256 and info.get("checked") is True and (d / "llama_cpp" / "__init__.py").is_file()
    )


def activate(data_dir: Path | None = None) -> bool:
    """Make a downloaded (and checked) engine importable. True when one is on ``sys.path``."""
    w = wheel_for()
    if w is None or not installed(data_dir, w):
        return False
    d = str(engine_dir(w, data_dir))
    if d not in sys.path:
        sys.path.append(d)  # appended: a pip-installed llama_cpp still wins
        importlib.invalidate_caches()
    return True


def missing_dependencies(find_spec: Callable[[str], Any] = importlib.util.find_spec) -> list[str]:
    out = []
    for mod in RUNTIME_DEPS:
        try:
            if find_spec(mod) is None:
                out.append(mod)
        except (ImportError, ValueError):
            out.append(mod)
    return out


def _safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    keep = []
    for info in zf.infolist():
        name = info.filename
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
            raise EngineInstallError("The engine download looks damaged. Press Download to try again.")
        if name.startswith(KEEP_TOP):
            keep.append(info)
    if not any(i.filename == "llama_cpp/__init__.py" for i in keep):
        raise EngineInstallError("The engine download looks damaged. Press Download to try again.")
    return keep


def unpack(wheel_file: Path, dest: Path) -> None:
    """Unpack only the runtime parts into ``dest`` (replaced atomically)."""
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(wheel_file) as zf:
        for info in _safe_members(zf):
            zf.extract(info, tmp)
            mode = (info.external_attr >> 16) & 0o777
            if mode and not info.is_dir():
                with contextlib.suppress(OSError):
                    (tmp / info.filename).chmod(mode | 0o600)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)


def check_command() -> list[str]:
    from away.app.frozen import away_command

    return away_command(["cleanup-engine", "check"])


def run_check(
    runner: Callable[..., Any] = subprocess.run, timeout: float = 120.0, data_dir: Path | None = None
) -> tuple[bool, str]:
    """Load the engine in a separate process (a broken DLL cannot crash the app). (ok, detail)."""
    try:
        cp = runner(check_command(), capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    out = ((cp.stdout or "") + (cp.stderr or "")).strip()
    return cp.returncode == 0, out[-600:]


def friendly_check_error(detail: str) -> str:
    low = detail.lower()
    if "vcomp140" in low or "msvcp140" in low or "vcruntime140" in low or "dll load failed" in low:
        return (
            "Smart cleanup's engine needs the Microsoft Visual C++ Redistributable (x64). Install it from "
            "aka.ms/vs/17/release/vc_redist.x64.exe, then press Download again. Basic cleanup keeps working."
        )
    return "Smart cleanup's engine could not start on this computer. Basic cleanup keeps working; Ollama is an option."


def install(
    progress: Callable[[int, int], None] | None = None,
    data_dir: Path | None = None,
    downloader: Callable[..., Path] | None = None,
    checker: Callable[[], tuple[bool, str]] | None = None,
    wheel: Wheel | None = None,
) -> Path:
    """Download, verify, unpack and check the engine. Returns its folder; raises :class:`EngineInstallError`."""
    from away.app import fastdl
    from away.app.optional_models import _ProgressMeter

    w = wheel or wheel_for()
    if w is None:
        raise EngineInstallError("There is no ready-made Smart cleanup engine for this computer. Use Ollama instead.")
    dest = engine_dir(w, data_dir)
    if installed(data_dir, w):
        if progress is not None:
            progress(w.size, w.size)
        return dest
    missing = missing_dependencies()
    if missing:
        raise EngineInstallError(f"Smart cleanup's engine needs: pip install {' '.join(missing)}")
    dl = engines_dir(data_dir) / ".downloads"
    dl.mkdir(parents=True, exist_ok=True)
    wf = dl / w.filename
    pm = _ProgressMeter(w.size, progress, lambda _s: None)
    ok = False
    try:
        (downloader or fastdl.download_file)(
            w.url, wf, fastdl.RemoteFile(w.filename, w.size, sha256=w.sha256), pm.meter
        )
        ok = True
    except Exception as exc:
        raise EngineInstallError(
            f"The engine download stopped ({type(exc).__name__}). Check your internet and press Download again."
        ) from exc
    finally:
        pm.meter.finish(ok)
    if fastdl.file_digest(wf, "sha256") != w.sha256:  # download_file verifies too; never unpack an unverified file
        wf.unlink(missing_ok=True)
        raise EngineInstallError("The engine download looks damaged. Press Download to try again.")
    unpack(wf, dest)
    marker = {"version": VERSION, "wheel": w.filename, "sha256": w.sha256, "checked": False}
    (dest / MARKER).write_text(json.dumps(marker), encoding="utf-8")
    ok_check, detail = (checker or (lambda: run_check(data_dir=data_dir)))()
    if not ok_check:
        shutil.rmtree(dest, ignore_errors=True)
        raise EngineInstallError(friendly_check_error(detail))
    marker["checked"] = True
    (dest / MARKER).write_text(json.dumps(marker), encoding="utf-8")
    wf.unlink(missing_ok=True)
    if progress is not None:
        progress(w.size, w.size)
    return dest


def remove(data_dir: Path | None = None) -> bool:
    w = wheel_for()
    if w is None:
        return False
    d = engine_dir(w, data_dir)
    gone = d.exists()
    shutil.rmtree(d, ignore_errors=True)
    return gone


def self_check(data_dir: Path | None = None) -> str:
    """What ``away cleanup-engine check`` runs (in its own process): import llama_cpp and touch the library."""
    w = wheel_for()
    if w is not None and not pip_installed():
        d = str(engine_dir(w, data_dir))
        if (Path(d) / "llama_cpp" / "__init__.py").is_file() and d not in sys.path:
            sys.path.append(d)
    import llama_cpp

    info = llama_cpp.llama_print_system_info()
    text = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)
    return f"llama-cpp-python {llama_cpp.__version__} loaded: {text.strip()[:300]}"


def status_line(data_dir: Path | None = None) -> str:
    w = wheel_for()
    if pip_installed():
        return "llama-cpp-python installed with pip (used as is)"
    if w is None:
        return "no ready-made engine for this computer (use Ollama, or pip install llama-cpp-python)"
    if installed(data_dir, w):
        return f"downloaded engine {VERSION} ({w.tag}) in {engine_dir(w, data_dir)}"
    return f"not installed (Download fetches {w.filename}, {w.size / 1e6:.0f} MB)"
