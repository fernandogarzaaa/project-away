"""v0.1.24 model refresh: runtime engines (away.app.engines) and catalog downloads (away.app.model_fetch).

Everything runs on fakes: a fake downloader writes bytes we choose, so the sha256 checks, the unpacking rules and
the "complete" markers are exercised without the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from away.app import engines, model_fetch


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fake_downloader(payloads: dict[str, bytes], calls: list[str] | None = None):
    """``download_file(url, dest, remote, meter)`` that writes ``payloads[url]``."""

    def dl(url, dest, remote, meter):
        if calls is not None:
            calls.append(url)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = payloads[url]
        dest.write_bytes(data)
        meter.add(len(data))
        return dest

    return dl


def _wheel_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _tgz_bytes(files: dict[str, bytes], exe: tuple[str, ...] = ()) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mode = 0o755 if name in exe else 0o644
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _archive(kind: str, data: bytes, engine: str = "moonshine", keep: tuple[str, ...] = ()) -> engines.Archive:
    fname = {"wheel": "fake-0.1-py3-none-any.whl", "zip": "fake.zip", "tgz": "fake.tar.gz"}[kind]
    return engines.Archive(
        engine,
        "0.1",
        "test",
        fname,
        "https://files.pythonhosted.org/packages/x/" + fname,
        len(data),
        _sha(data),
        kind,
        keep,
    )


# ----------------------------------------------------------------------------------------------- engines
def test_platform_key_normalises_os_and_cpu():
    assert engines.platform_key("win32", "AMD64") == ("win32", "amd64")
    assert engines.platform_key("linux", "x86_64") == ("linux", "x86_64")
    assert engines.platform_key("linux", "arm64") == ("linux", "aarch64")
    assert engines.platform_key("darwin", "arm64") == ("darwin", "arm64")


def test_archive_for_picks_pinned_files_per_platform():
    win = engines.archive_for("moonshine", system="win32", machine="AMD64")
    assert win is not None and win.kind == "wheel" and win.filename.endswith("win_amd64.whl")
    assert win.keep == ("moonshine_voice/",)
    # the wake-word wheel is pure Python: one file for every computer
    assert engines.archive_for("livekit-wakeword", system="darwin", machine="arm64").tag == "any"
    # nemo-speech: one archive per variant, cpu when none is asked for
    assert engines.archive_for("nemo-speech", "cuda", "win32", "amd64").tag == "windows-x86_64-cuda"
    assert engines.archive_for("nemo-speech", "", "linux", "x86_64").tag == "linux-x86_64-cpu"
    assert engines.archive_for("nemo-speech", "cuda", "darwin", "arm64") is None  # no CUDA on a Mac
    assert engines.archive_for("moonshine", system="darwin", machine="x86_64") is None  # no Intel Mac wheel
    assert engines.archive_for("nope") is None


def test_every_pinned_archive_is_https_with_a_full_sha256():
    for table in engines.ARCHIVES.values():
        for a in table.values():
            assert a.url.startswith("https://")
            assert len(a.sha256) == 64 and int(a.sha256, 16) >= 0
            assert a.size > 0 and a.license
            assert "gpl" not in a.license.lower()  # espeak-ng / Piper are never an engine


def test_variants_lists_what_this_platform_has():
    assert sorted(engines.variants("nemo-speech", "win32", "amd64")) == ["cpu", "cuda", "vulkan"]
    assert sorted(engines.variants("nemo-speech", "linux", "x86_64")) == ["cpu", "cuda"]
    assert engines.variants("nemo-speech", "darwin", "arm64") == ["metal"]


def test_install_wheel_keeps_only_its_package_and_marks_checked(tmp_path):
    data = _wheel_bytes(
        {
            "moonshine_voice/__init__.py": b"X = 1\n",
            "moonshine_voice/lib/libmoonshine.so": b"\x7fELF",
            "moonshine_voice-0.1.dist-info/METADATA": b"Name: x",
            "evil_other_pkg/__init__.py": b"import os",
        }
    )
    a = _archive("wheel", data, keep=("moonshine_voice/",))
    seen = []
    d = engines.install(
        a, tmp_path, progress=lambda done, total: seen.append((done, total)), downloader=_fake_downloader({a.url: data})
    )
    assert (d / "moonshine_voice" / "__init__.py").read_text() == "X = 1\n"
    assert not (d / "evil_other_pkg").exists() and not (d / "moonshine_voice-0.1.dist-info").exists()
    assert engines.installed(a, tmp_path)
    marker = json.loads((d / engines.MARKER).read_text())
    assert marker["checked"] is True and marker["sha256"] == a.sha256
    assert seen and seen[-1][0] == len(data)
    assert not (tmp_path / "engines" / ".downloads" / a.filename).exists()  # the download is cleaned up
    # a second install is a no-op (no download)
    calls: list[str] = []
    engines.install(a, tmp_path, downloader=_fake_downloader({a.url: data}, calls))
    assert calls == []


def test_install_rejects_a_sha_mismatch_and_leaves_nothing(tmp_path):
    good = _wheel_bytes({"moonshine_voice/__init__.py": b""})
    a = _archive("wheel", good, keep=("moonshine_voice/",))
    tampered = _wheel_bytes({"moonshine_voice/__init__.py": b"import evil"})
    with pytest.raises(engines.EngineError, match="damaged"):
        engines.install(a, tmp_path, downloader=_fake_downloader({a.url: tampered}))
    assert not engines.installed(a, tmp_path)
    assert not engines.engine_dir(a, tmp_path).exists()
    assert not (tmp_path / "engines" / ".downloads" / a.filename).exists()


def test_install_download_failure_is_a_friendly_error(tmp_path):
    a = _archive("wheel", b"x")

    def broken(*_a, **_k):
        raise TimeoutError("slow")

    with pytest.raises(engines.EngineError, match="Check your internet"):
        engines.install(a, tmp_path, downloader=broken)


def test_install_removes_an_engine_whose_check_fails(tmp_path):
    data = _wheel_bytes({"moonshine_voice/__init__.py": b""})
    a = _archive("wheel", data, keep=("moonshine_voice/",))
    with pytest.raises(engines.EngineError, match="could not start"):
        engines.install(
            a,
            tmp_path,
            downloader=_fake_downloader({a.url: data}),
            checker=lambda d: (False, "missing VCRUNTIME140.dll"),
        )
    assert not engines.engine_dir(a, tmp_path).exists()
    assert not engines.installed(a, tmp_path)


def test_unpack_refuses_path_traversal_in_zip_and_tar(tmp_path):
    zdata = _wheel_bytes({"../escape.txt": b"x"})
    za = _archive("zip", zdata)
    with pytest.raises(engines.EngineError):
        engines.install(za, tmp_path, downloader=_fake_downloader({za.url: zdata}))
    assert not (tmp_path / "engines" / "escape.txt").exists()

    tdata = _tgz_bytes({"/etc/passwd": b"x"})
    ta = _archive("tgz", tdata, engine="nemo-speech")
    with pytest.raises(engines.EngineError):
        engines.install(ta, tmp_path, downloader=_fake_downloader({ta.url: tdata}))


def test_unpack_wheel_without_its_package_is_damaged(tmp_path):
    data = _wheel_bytes({"other/__init__.py": b""})
    a = _archive("wheel", data, keep=("moonshine_voice/",))
    with pytest.raises(engines.EngineError, match="damaged"):
        engines.install(a, tmp_path, downloader=_fake_downloader({a.url: data}))


def test_tgz_engine_binary_is_found_and_executable(tmp_path):
    data = _tgz_bytes(
        {"nemo/bin/nemo-speech": b"#!/bin/sh\necho hi\n", "nemo/README": b"r"}, exe=("nemo/bin/nemo-speech",)
    )
    a = _archive("tgz", data, engine="nemo-speech")
    engines.install(a, tmp_path, downloader=_fake_downloader({a.url: data}))
    exe = engines.binary(a, tmp_path)
    assert exe is not None and exe.name == "nemo-speech"
    if sys.platform != "win32":
        assert exe.stat().st_mode & 0o111
    assert engines.remove(a, tmp_path) is True
    assert engines.binary(a, tmp_path) is None and not engines.installed(a, tmp_path)


def test_installed_needs_matching_sha_and_checked(tmp_path):
    a = _archive("wheel", b"abc")
    d = engines.engine_dir(a, tmp_path)
    d.mkdir(parents=True)
    assert not engines.installed(a, tmp_path)  # no marker
    (d / engines.MARKER).write_text(json.dumps({"sha256": a.sha256, "checked": False}))
    assert not engines.installed(a, tmp_path)  # unpacked but never checked
    (d / engines.MARKER).write_text(json.dumps({"sha256": "0" * 64, "checked": True}))
    assert not engines.installed(a, tmp_path)  # an older pin
    (d / engines.MARKER).write_text(json.dumps({"sha256": a.sha256, "checked": True}))
    assert engines.installed(a, tmp_path)
    assert not engines.installed(None, tmp_path)


def test_activate_appends_a_downloaded_wheel_after_pip(tmp_path, monkeypatch):
    data = _wheel_bytes({"livekit/__init__.py": b"", "livekit/wakeword/__init__.py": b"OK = 1\n"})
    a = _archive("wheel", data, engine="livekit-wakeword", keep=("livekit/",))
    monkeypatch.setitem(engines.ARCHIVES, "livekit-wakeword", {("any", "any"): a})
    monkeypatch.setattr(engines, "pip_installed", lambda mod: False)
    assert engines.activate("livekit-wakeword", tmp_path) is False  # not downloaded yet
    engines.install(a, tmp_path, downloader=_fake_downloader({a.url: data}))
    before = list(sys.path)
    try:
        assert engines.activate("livekit-wakeword", tmp_path) is True
        assert sys.path[-1] == str(engines.engine_dir(a, tmp_path))  # appended: a pip copy still wins
        n = len(sys.path)
        engines.activate("livekit-wakeword", tmp_path)
        assert len(sys.path) == n  # only once
    finally:
        sys.path[:] = before
    # a pip-installed copy wins without touching sys.path
    monkeypatch.setattr(engines, "pip_installed", lambda mod: True)
    assert engines.activate("livekit-wakeword", tmp_path) is True
    assert engines.activate("nemo-speech", tmp_path) is False  # an archive engine, never imported


def test_missing_modules_and_status(tmp_path, monkeypatch):
    def find(name):
        if name == "boom":
            raise ImportError("x")
        return object() if name == "numpy" else None

    assert engines.missing_modules(("numpy", "nope", "boom"), find) == ["nope", "boom"]
    monkeypatch.setattr(engines, "pip_installed", lambda mod: False)
    st = engines.status("nemo-speech", "cpu", tmp_path)
    assert st["engine"] == "nemo-speech" and st["installed"] is False and st["pip"] is False
    assert st["license"] == "Apache-2.0"
    assert engines.status("nope", "", tmp_path) == {
        "engine": "nope",
        "variant": "",
        "available": False,
        "installed": False,
        "pip": False,
        "size_mb": 0,
        "license": "",
    }


# ----------------------------------------------------------------------------------------------- model_fetch
def _model(mid="m-test", files=None, url="", sha256="", size=0, **kw):
    return SimpleNamespace(id=mid, title="Test model", files=files or [], url=url, sha256=sha256, size=size, **kw)


def test_files_of_single_url_and_bundle():
    m = _model(url="https://huggingface.co/x/resolve/abc/model.onnx?download=1", sha256="A" * 64, size=5)
    (spec,) = model_fetch.files_of(m)
    assert spec.path == "model.onnx" and spec.sha256 == "a" * 64 and spec.size == 5
    b = _model(files=[{"path": "a/b.bin", "url": "https://x/b", "sha256": "B" * 64, "size": "3"}])
    (fs,) = model_fetch.files_of(b)
    assert fs == model_fetch.FileSpec("a/b.bin", "https://x/b", "b" * 64, 3)
    assert model_fetch.files_of(_model()) == []


def test_fetch_bundle_verifies_and_marks_complete(tmp_path):
    a, b = b"first file", b"second file!"
    m = _model(
        files=[
            {"path": "a.onnx", "url": "https://h/a", "sha256": _sha(a), "size": len(a)},
            {"path": "sub/b.json", "url": "https://h/b", "sha256": _sha(b), "size": len(b)},
        ]
    )
    calls: list[str] = []
    prog = []
    d = model_fetch.fetch(
        m,
        tmp_path,
        progress=lambda done, tot: prog.append(done),
        downloader=_fake_downloader({"https://h/a": a, "https://h/b": b}, calls),
    )
    assert d == tmp_path / "models" / "m-test"
    assert (d / "sub" / "b.json").read_bytes() == b
    assert model_fetch.have(m, tmp_path)
    assert calls == ["https://h/a", "https://h/b"] and prog[-1] == len(a) + len(b)
    # already complete: nothing is downloaded again
    calls.clear()
    model_fetch.fetch(m, tmp_path, downloader=_fake_downloader({}, calls))
    assert calls == []


def test_fetch_rejects_a_sha_mismatch_and_removes_the_file(tmp_path):
    good = b"the real model"
    m = _model(files=[{"path": "m.onnx", "url": "https://h/m", "sha256": _sha(good), "size": len(good)}])
    with pytest.raises(model_fetch.FetchError, match="did not match its checksum"):
        model_fetch.fetch(m, tmp_path, downloader=_fake_downloader({"https://h/m": b"something else"}))
    d = model_fetch.model_dir("m-test", tmp_path)
    assert not (d / "m.onnx").exists()
    assert not (d / model_fetch.DONE).exists()
    assert not model_fetch.have(m, tmp_path)


def test_fetch_resumes_files_already_verified(tmp_path):
    a, b = b"aaa", b"bbb"
    m = _model(
        files=[
            {"path": "a", "url": "https://h/a", "sha256": _sha(a), "size": 3},
            {"path": "b", "url": "https://h/b", "sha256": _sha(b), "size": 3},
        ]
    )
    d = model_fetch.model_dir("m-test", tmp_path)
    d.mkdir(parents=True)
    (d / "a").write_bytes(a)
    calls: list[str] = []
    model_fetch.fetch(m, tmp_path, downloader=_fake_downloader({"https://h/b": b}, calls))
    assert calls == ["https://h/b"]


def test_fetch_refuses_unsafe_or_unpinned_entries(tmp_path):
    for bad in ("../x", "/abs", "C:\\x", "a\\b"):
        m = _model(files=[{"path": bad, "url": "https://h/x", "sha256": "a" * 64}])
        with pytest.raises(model_fetch.FetchError, match="bad file entry"):
            model_fetch.fetch(m, tmp_path, downloader=_fake_downloader({}))
    short = _model(files=[{"path": "x", "url": "https://h/x", "sha256": "abc"}])
    with pytest.raises(model_fetch.FetchError):
        model_fetch.fetch(short, tmp_path, downloader=_fake_downloader({}))
    with pytest.raises(model_fetch.FetchError, match="nothing to download"):
        model_fetch.fetch(_model(), tmp_path)


def test_fetch_network_error_is_plain(tmp_path):
    m = _model(files=[{"path": "x", "url": "https://h/x", "sha256": "a" * 64, "size": 1}])

    def broken(*_a, **_k):
        raise ConnectionResetError()

    with pytest.raises(model_fetch.FetchError, match="Check your internet"):
        model_fetch.fetch(m, tmp_path, downloader=broken)


def test_fetch_unpacks_zip_after_the_check_and_refuses_unsafe_archives(tmp_path):
    z = _wheel_bytes({"inner/model.onnx": b"m"})
    m = _model(files=[{"path": "pack.zip", "url": "https://h/z", "sha256": _sha(z), "size": len(z)}])
    d = model_fetch.fetch(m, tmp_path, downloader=_fake_downloader({"https://h/z": z}))
    assert (d / "pack" / "inner" / "model.onnx").read_bytes() == b"m"
    assert model_fetch.find(d, "model.onnx") == d / "pack" / "inner" / "model.onnx"
    assert model_fetch.find(d, "nothing.bin") is None

    evil = _wheel_bytes({"../../owned.txt": b"x"})
    m2 = _model("m-evil", files=[{"path": "e.zip", "url": "https://h/e", "sha256": _sha(evil), "size": len(evil)}])
    with pytest.raises(model_fetch.FetchError, match="unsafe path"):
        model_fetch.fetch(m2, tmp_path, downloader=_fake_downloader({"https://h/e": evil}))
    assert not (tmp_path / "owned.txt").exists()
    assert not model_fetch.have(m2, tmp_path)


def test_have_is_false_when_a_file_went_missing_or_the_pin_changed(tmp_path):
    a = b"x"
    m = _model(files=[{"path": "a", "url": "https://h/a", "sha256": _sha(a), "size": 1}])
    d = model_fetch.fetch(m, tmp_path, downloader=_fake_downloader({"https://h/a": a}))
    assert model_fetch.have(m, tmp_path)
    m2 = _model(files=[{"path": "a", "url": "https://h/a", "sha256": "f" * 64, "size": 1}])
    assert not model_fetch.have(m2, tmp_path)  # a newer pin is not "downloaded"
    (d / "a").unlink()
    assert not model_fetch.have(m, tmp_path)
    assert model_fetch.have(_model(bundled=True), tmp_path)


def test_ready_and_download_cover_files_and_engines(tmp_path, monkeypatch):
    """One click = the files (verified) + the engine they run in; ready() needs both."""
    data = b"model"
    m = _model(
        "stt-x",
        files=[{"path": "m.bin", "url": "https://h/m", "sha256": _sha(data), "size": 5}],
        engine="moonshine",
        set={},
    )
    wheel = _wheel_bytes({"moonshine_voice/__init__.py": b""})
    arch = _archive("wheel", wheel, keep=("moonshine_voice/",))
    monkeypatch.setattr(engines, "archive_for", lambda name, variant="", system=None, machine=None: arch)
    monkeypatch.setattr(engines, "pip_installed", lambda mod: False)
    real_fetch, real_install = model_fetch.fetch, engines.install
    monkeypatch.setattr(
        model_fetch,
        "fetch",
        lambda mm, dd, pp=None: real_fetch(mm, dd, pp, downloader=_fake_downloader({"https://h/m": data})),
    )
    monkeypatch.setattr(
        engines,
        "install",
        lambda aa, dd=None, pp=None: real_install(aa, dd, pp, downloader=_fake_downloader({arch.url: wheel})),
    )
    assert not model_fetch.ready(m, tmp_path)
    folder = model_fetch.download(m, tmp_path)
    assert folder == model_fetch.model_dir("stt-x", tmp_path)
    assert model_fetch.have(m, tmp_path) and engines.installed(arch, tmp_path)
    assert model_fetch.ready(m, tmp_path)
    # the engine removed again -> not ready
    engines.remove(arch, tmp_path)
    assert not model_fetch.ready(m, tmp_path)


def test_cleanup_models_go_through_the_v0123_downloader(tmp_path, monkeypatch):
    from away.app import optional_models as om

    m = _model("cleanup-qwen3.5-0.8b-q4km", set={"voice.cleanup.model": "qwen3.5-0.8b"})
    assert model_fetch.cleanup_key(m) == "qwen3.5-0.8b"
    assert model_fetch.cleanup_key(_model(set={"voice.cleanup.model": "not-a-model"})) is None
    got = []
    monkeypatch.setattr(om, "install", lambda key, d, progress=None: got.append((key, d)))
    monkeypatch.setattr(om, "installed", lambda key, d: key == "qwen3.5-0.8b")
    assert model_fetch.download(m, tmp_path) == tmp_path / "models"
    assert got == [("qwen3.5-0.8b", tmp_path / "models")]
    assert model_fetch.ready(m, tmp_path)


def test_qwen35_cleanup_models_are_pinned_and_kept_out_of_the_v0123_list(tmp_path):
    from away.app import optional_models as om

    keys = {m.key for m in om.CLEANUP_MODELS_NEXT}
    assert keys == {"qwen3.5-2b", "qwen3.5-2b-q8", "qwen3.5-0.8b"}
    for m in om.CLEANUP_MODELS_NEXT:
        assert m.kind == "cleanup_next" and len(m.sha256) == 64 and "/resolve/main/" not in m.url
        assert m.license == "Apache-2.0" and not m.restricted
    assert keys.isdisjoint({row["key"] for row in om.describe(tmp_path)})
