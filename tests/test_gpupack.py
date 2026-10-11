"""GPU pack: PyPI wheel lookup, verified download + extraction, activation, CLI. No network, no GPU."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import urllib.request
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from away.app import fastdl, gpupack
from away.cli import main


def _opener(req: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)


def make_wheel(os_name: str = "linux", extra: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, m in enumerate(gpupack.MEMBERS[os_name]):
            z.writestr(m, bytes([65 + i]) * (3000 + i))
        z.writestr("nvidia/cublas/lib/libnvblas.so.12", b"not needed")
        z.writestr("nvidia_cublas_cu12-12.8.4.1.dist-info/License.txt", b"NVIDIA EULA")
        for k, v in (extra or {}).items():
            z.writestr(k, v)
    return buf.getvalue()


class PyPI:
    def __init__(self, wheel: bytes, tag: str = "manylinux_2_27_x86_64") -> None:
        self.wheel = wheel
        self.tag = tag
        self.sha = hashlib.sha256(wheel).hexdigest()
        self.fn = f"nvidia_cublas_cu12-12.8.4.1-py3-none-{tag}.whl"
        self.base = ""
        self.hits: list[str] = []

    def json(self) -> bytes:
        return json.dumps(
            {
                "urls": [
                    {
                        "filename": "nvidia_cublas_cu12-12.8.4.1-py3-none-other.whl",
                        "url": "x",
                        "size": 1,
                        "digests": {"sha256": "0"},
                    },
                    {
                        "filename": self.fn,
                        "url": f"{self.base}/files/{self.fn}",
                        "size": len(self.wheel),
                        "digests": {"sha256": self.sha},
                    },
                ]
            }
        ).encode()


@pytest.fixture
def pypi(monkeypatch: pytest.MonkeyPatch) -> Iterator[PyPI]:
    srv_state = PyPI(make_wheel())

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def do_GET(self) -> None:
            srv_state.hits.append(self.path)
            if self.path == "/pypi/nvidia-cublas-cu12/12.8.4.1/json":
                body = srv_state.json()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == f"/files/{srv_state.fn}":
                data = srv_state.wheel
                rng = self.headers.get("Range")
                if rng:
                    s, e = (int(x) for x in rng.split("=")[1].split("-"))
                    e = min(e, len(data) - 1)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {s}-{e}/{len(data)}")
                    self.send_header("Content-Length", str(e - s + 1))
                    self.end_headers()
                    self.wfile.write(data[s : e + 1])
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv_state.base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setattr(gpupack, "PYPI_JSON", srv_state.base + "/pypi/{name}/{version}/json")
    monkeypatch.setitem(gpupack.PINNED_SHA256, srv_state.tag, srv_state.sha)
    yield srv_state
    srv.shutdown()
    srv.server_close()


def test_platform_tags():
    assert gpupack.platform_tag("win32", "AMD64") == "win_amd64"
    assert gpupack.platform_tag("linux", "x86_64") == "manylinux_2_27_x86_64"
    assert gpupack.platform_tag("linux", "aarch64") is None  # CTranslate2 has no CUDA build there
    assert gpupack.platform_tag("darwin", "x86_64") is None


def test_pins_match_the_bundled_ctranslate2_build():
    assert (gpupack.PACKAGE, gpupack.VERSION) == ("nvidia-cublas-cu12", "12.8.4.1")
    assert set(gpupack.PINNED_SHA256) == {"win_amd64", "manylinux_2_27_x86_64"}
    assert all(len(v) == 64 for v in gpupack.PINNED_SHA256.values())
    assert gpupack.MEMBERS["win32"][-1].endswith("cublas64_12.dll")
    assert gpupack.MEMBERS["linux"][-1].endswith("libcublas.so.12")


def test_fetch_wheel_info_and_digest_pin(pypi: PyPI, monkeypatch: pytest.MonkeyPatch):
    w = gpupack.fetch_wheel_info(pypi.tag, _opener)
    assert (w.filename, w.size, w.sha256) == (pypi.fn, len(pypi.wheel), pypi.sha)
    monkeypatch.setitem(gpupack.PINNED_SHA256, pypi.tag, "f" * 64)
    with pytest.raises(fastdl.ChecksumError):
        gpupack.fetch_wheel_info(pypi.tag, _opener)
    with pytest.raises(RuntimeError, match="no wheel"):
        gpupack.fetch_wheel_info("win_arm64", _opener)


def test_install_downloads_verifies_extracts_and_is_idempotent(pypi: PyPI, tmp_path: Path):
    d = tmp_path / "gpu"
    msgs: list[str] = []
    out = gpupack.install(
        write=lambda s: None, progress=msgs.append, opener=_opener, platform="linux", arch="x86_64", d=d
    )
    assert out == d and gpupack.installed(d)
    names = sorted(p.name for p in d.iterdir())
    assert names == ["NVIDIA-cuBLAS-LICENSE.txt", "libcublas.so.12", "libcublasLt.so.12", "manifest.json"]
    m = gpupack.read_manifest(d)
    assert m["wheel_sha256"] == pypi.sha and m["version"] == "12.8.4.1"
    assert m["files"]["libcublas.so.12"]["sha256"] == hashlib.sha256((d / "libcublas.so.12").read_bytes()).hexdigest()
    assert not (d / ".download").exists()  # the wheel is deleted after extraction
    st = gpupack.status(d)
    assert st["installed"] and st["files"] == ["libcublas.so.12", "libcublasLt.so.12"]
    hits = len(pypi.hits)
    gpupack.install(write=lambda s: None, progress=msgs.append, opener=_opener, platform="linux", arch="x86_64", d=d)
    assert len(pypi.hits) == hits and "already installed" in msgs[-1]
    # a truncated library means "not installed"
    (d / "libcublas.so.12").write_bytes(b"x")
    assert not gpupack.installed(d)
    assert gpupack.remove(d) and not d.exists() and not gpupack.remove(d)


def test_install_rejects_corrupt_download(pypi: PyPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    good = pypi.wheel
    pypi.wheel = good[:-10] + b"0123456789"  # served bytes differ from the published digest
    pypi.sha = hashlib.sha256(good).hexdigest()
    monkeypatch.setitem(gpupack.PINNED_SHA256, pypi.tag, pypi.sha)
    with pytest.raises(fastdl.ChecksumError):
        gpupack.install(
            write=lambda s: None,
            progress=lambda s: None,
            opener=_opener,
            platform="linux",
            arch="x86_64",
            d=tmp_path / "g",
        )
    assert not gpupack.installed(tmp_path / "g")


def test_install_refuses_unsupported_platforms(tmp_path: Path):
    with pytest.raises(RuntimeError, match="x86-64"):
        gpupack.install(platform="darwin", arch="arm64", d=tmp_path)


def test_extract_reports_missing_members(tmp_path: Path):
    w = tmp_path / "w.whl"
    with zipfile.ZipFile(w, "w") as z:
        z.writestr("other.txt", b"x")
    with pytest.raises(RuntimeError, match="missing"):
        gpupack.extract(w, tmp_path / "o", gpupack.MEMBERS["linux"])


def test_activate_preloads_in_dependency_order(tmp_path: Path):
    for m in gpupack.MEMBERS["linux"]:
        (tmp_path / Path(m).name).write_bytes(b"lib")
    loaded: list[str] = []
    got = gpupack.activate("linux", [tmp_path / "missing", tmp_path], loader=lambda p: loaded.append(p) or object())
    assert [Path(p).name for p in loaded] == ["libcublasLt.so.12", "libcublas.so.12"]
    assert got == loaded
    assert gpupack.activate("darwin", [tmp_path]) == []


def test_activate_windows_adds_dll_directory_and_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for m in gpupack.MEMBERS["win32"]:
        (tmp_path / Path(m).name).write_bytes(b"dll")
    added: list[str] = []
    monkeypatch.setattr(gpupack.os, "add_dll_directory", lambda p: added.append(p) or object(), raising=False)
    monkeypatch.setenv("PATH", "C:\\Windows")
    gpupack.activate("win32", [tmp_path], loader=lambda p: object())
    assert added == [str(tmp_path)] and gpupack.os.environ["PATH"].startswith(str(tmp_path))


def test_activate_stops_when_a_library_fails_to_load(tmp_path: Path):
    for m in gpupack.MEMBERS["linux"]:
        (tmp_path / Path(m).name).write_bytes(b"lib")

    def bad(p: str) -> Any:
        raise OSError("wrong ELF class")

    assert gpupack.activate("linux", [tmp_path], loader=bad) == []


def test_pip_instructions_name_the_package():
    text = gpupack.pip_instructions()
    assert "pip install nvidia-cublas-cu12==12.8.4.1" in text and "cudnn" in text


def test_cli_gpu_install_on_pip_prints_instructions(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(gpupack, "platform_tag", lambda *a: "manylinux_2_27_x86_64")
    assert main(["gpu", "install"]) == 0
    assert "pip install nvidia-cublas-cu12" in capsys.readouterr().out


def test_cli_gpu_install_frozen_without_gpu_and_with_force(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from away.app import frozen, hardware

    monkeypatch.setattr(gpupack, "platform_tag", lambda *a: "win_amd64")
    monkeypatch.setattr(frozen, "is_frozen", lambda: True)
    monkeypatch.setattr(hardware, "detect", lambda: hardware.Hardware(os="win32", arch="amd64"))
    assert main(["gpu", "install"]) == 2
    calls: list[int] = []
    monkeypatch.setattr(gpupack, "install", lambda connections=8: calls.append(connections) or Path("x"))
    assert main(["gpu", "install", "--force", "--connections", "4"]) == 0 and calls == [4]
    monkeypatch.setattr(gpupack, "install", lambda connections=8: (_ for _ in ()).throw(OSError("offline")))
    assert main(["gpu", "install", "--force"]) == 2
    assert "resume" in capsys.readouterr().err


def test_cli_gpu_install_unsupported_platform(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gpupack, "platform_tag", lambda *a: None)
    assert main(["gpu", "install"]) == 2


def test_cli_gpu_status_and_remove(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    from away.app import hardware

    monkeypatch.setattr(hardware, "detect", lambda: hardware.Hardware(os="linux", arch="x86_64"))
    assert main(["gpu", "status"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["installed"] is False and st["nvidia_gpu"] is None and st["cublas_loadable"] is False
    assert main(["gpu", "remove"]) == 0 and "not installed" in capsys.readouterr().out
