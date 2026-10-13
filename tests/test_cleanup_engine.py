"""v0.1.20: the Smart cleanup engine for installer users (a pinned, sha256-verified llama-cpp-python wheel)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from away.voice import engine


def make_wheel(path: Path, extra: dict[str, bytes] | None = None) -> engine.Wheel:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("llama_cpp/__init__.py", "__version__ = 'fake'\n")
        zf.writestr("llama_cpp/lib/libllama.so", b"\x7fELF")
        zf.writestr("llama_cpp_python.libs/libgomp.so.1", b"\x7fELF")
        zf.writestr("include/llama.h", "/* not unpacked */")
        zf.writestr("bin/llama.dll", b"MZ")
        zf.writestr("llama_cpp_python-0.3.36.dist-info/METADATA", "Name: llama_cpp_python\n")
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    data = buf.getvalue()
    path.write_bytes(data)
    return engine.Wheel("test", path.name, len(data), hashlib.sha256(data).hexdigest())


def copying_downloader(src: Path) -> Any:
    def download(url: str, dest: Path, rf: Any, meter: Any) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        meter.add(rf.size)
        return dest

    return download


@pytest.mark.parametrize(
    ("system", "machine", "tag"),
    [
        ("win32", "AMD64", "win_amd64"),
        ("linux", "x86_64", "linux_x86_64"),
        ("linux", "aarch64", "linux_aarch64"),
    ],
)
def test_a_pinned_wheel_for_each_installer_platform(system: str, machine: str, tag: str) -> None:
    w = engine.wheel_for(system, machine)
    assert w is not None and w.tag == tag and len(w.sha256) == 64 and w.size > 1_000_000
    assert w.url.startswith("https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/")
    assert "py3-none" in w.filename  # ctypes-based: independent of the Python version


def test_no_wheel_for_unsupported_platforms() -> None:
    assert engine.wheel_for("darwin", "x86_64") is None  # Intel Macs: Ollama or pip
    assert engine.wheel_for("darwin", "arm64") is None  # the published 0.3.36 macOS wheel is damaged
    assert engine.wheel_for("win32", "arm64") is None


def test_install_verifies_unpacks_only_the_runtime_checks_and_activates(tmp_path: Path, monkeypatch: Any) -> None:
    w = make_wheel(tmp_path / "x.whl")
    data = tmp_path / "data"
    seen: list[tuple[int, int]] = []
    checks: list[str] = []
    d = engine.install(
        progress=lambda a, b: seen.append((a, b)),
        data_dir=data,
        downloader=copying_downloader(tmp_path / "x.whl"),
        checker=lambda: (checks.append("ran"), (True, "ok"))[1],
        wheel=w,
    )
    assert checks == ["ran"] and seen[-1] == (w.size, w.size)
    assert (d / "llama_cpp" / "__init__.py").is_file() and (d / "llama_cpp_python.libs").is_dir()
    assert not (d / "include").exists() and not (d / "bin").exists()  # no headers, no stray DLL copies
    assert not list(d.rglob("msvcp140.dll"))
    assert json.loads((d / engine.MARKER).read_text())["checked"] is True
    assert not (data / ".downloads" / w.filename).exists()
    assert engine.installed(data, w)
    monkeypatch.setattr(engine, "wheel_for", lambda *a: w)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert engine.activate(data) and str(d) == sys.path[-1]
    assert engine.activate(data) and sys.path.count(str(d)) == 1  # idempotent
    assert engine.install(data_dir=data, wheel=w) == d  # already there: nothing downloaded


def test_a_failed_load_check_removes_the_engine_with_a_friendly_message(tmp_path: Path) -> None:
    w = make_wheel(tmp_path / "x.whl")
    with pytest.raises(engine.EngineInstallError, match="Visual C\\+\\+"):
        engine.install(
            data_dir=tmp_path / "data",
            downloader=copying_downloader(tmp_path / "x.whl"),
            checker=lambda: (False, "OSError: [WinError 126] VCOMP140.DLL not found"),
            wheel=w,
        )
    assert not engine.engine_dir(w, tmp_path / "data").exists()
    assert not engine.installed(tmp_path / "data", w)


def test_a_tampered_download_is_never_unpacked(tmp_path: Path) -> None:
    w = make_wheel(tmp_path / "x.whl")
    bad = engine.Wheel(w.tag, w.filename, w.size, "0" * 64)

    def download(url: str, dest: Path, rf: Any, meter: Any) -> Path:  # a downloader that skips verification
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((tmp_path / "x.whl").read_bytes())
        return dest

    with pytest.raises(engine.EngineInstallError, match="damaged"):
        engine.install(data_dir=tmp_path / "data", downloader=download, checker=lambda: (True, ""), wheel=bad)
    assert not engine.engine_dir(bad, tmp_path / "data").exists()


@pytest.mark.parametrize("evil", ["../evil.py", "llama_cpp/../../evil.py", "/abs/evil.py"])
def test_unpack_refuses_paths_outside_the_folder(tmp_path: Path, evil: str) -> None:
    make_wheel(tmp_path / "x.whl", {evil: b"x"})
    with pytest.raises(engine.EngineInstallError):
        engine.unpack(tmp_path / "x.whl", tmp_path / "out")
    assert not (tmp_path / "evil.py").exists()


def test_a_damaged_archive_is_a_friendly_error_and_leaves_nothing(tmp_path: Path) -> None:
    make_wheel(tmp_path / "x.whl")
    raw = bytearray((tmp_path / "x.whl").read_bytes())
    i = raw.index(b"__version__")  # stored data of llama_cpp/__init__.py: its CRC no longer matches
    raw[i : i + 11] = b"XXXXXXXXXXX"
    (tmp_path / "x.whl").write_bytes(bytes(raw))
    with pytest.raises(engine.EngineInstallError, match="damaged"):
        engine.unpack(tmp_path / "x.whl", tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.tmp").exists()


def test_download_errors_and_missing_dependencies_are_friendly(tmp_path: Path, monkeypatch: Any) -> None:
    w = make_wheel(tmp_path / "x.whl")

    def offline(*_a: Any, **_k: Any) -> Path:
        raise ConnectionError("offline")

    with pytest.raises(engine.EngineInstallError, match="internet"):
        engine.install(data_dir=tmp_path / "d", downloader=offline, checker=lambda: (True, ""), wheel=w)
    assert engine.missing_dependencies(lambda m: None if m == "diskcache" else object()) == ["diskcache"]
    monkeypatch.setattr(engine, "missing_dependencies", lambda: ["diskcache", "jinja2"])
    with pytest.raises(engine.EngineInstallError, match="pip install diskcache jinja2"):
        engine.install(data_dir=tmp_path / "d2", wheel=w)
    monkeypatch.setattr(engine, "wheel_for", lambda *a: None)
    with pytest.raises(engine.EngineInstallError, match="Ollama"):
        engine.install(data_dir=tmp_path / "d3")


def test_run_check_uses_a_separate_process() -> None:
    seen: dict[str, Any] = {}

    def runner(argv: list[str], **kw: Any) -> Any:
        seen["argv"] = argv
        return SimpleNamespace(returncode=0, stdout="llama-cpp-python 0.3.36 loaded", stderr="")

    ok, out = engine.run_check(runner=runner)
    assert ok and "loaded" in out and seen["argv"][-2:] == ["cleanup-engine", "check"]
    ok, out = engine.run_check(runner=lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 1)))
    assert not ok and "TimeoutExpired" in out


def test_cli_status(capsys: Any) -> None:
    from away import cli

    assert cli.main(["cleanup-engine", "status"]) == 0
    assert capsys.readouterr().out.strip()


@pytest.mark.skipif(
    os.environ.get("AWAY_NET_TESTS") != "1", reason="downloads the real engine (CI sets AWAY_NET_TESTS)"
)
def test_real_engine_downloads_verifies_and_loads_in_its_own_process(tmp_path: Path) -> None:
    """On the CI runners (Windows, macOS, Ubuntu): the pinned wheel downloads, matches its sha256, unpacks and
    loads llama.cpp in a separate process, exactly like Settings' Download button in the installed app."""
    if engine.wheel_for() is None:
        pytest.skip("no ready-made engine for this runner")
    if engine.missing_dependencies():
        pytest.skip(f"missing {engine.missing_dependencies()}")
    if engine.pip_installed():
        pytest.skip("llama-cpp-python is pip-installed here")
    env = {**os.environ, "AWAY_DATA_DIR": str(tmp_path / "data")}

    def checker() -> tuple[bool, str]:
        cp = subprocess.run(
            [sys.executable, "-m", "away", "cleanup-engine", "check"],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
        return cp.returncode == 0, cp.stdout + cp.stderr

    d = engine.install(data_dir=tmp_path / "data" / "engines", checker=checker)
    assert (d / "llama_cpp" / "__init__.py").is_file()
    ok, out = checker()
    assert ok and "loaded" in out, out
