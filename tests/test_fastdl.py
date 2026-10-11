"""Parallel, resumable Whisper downloader (away.app.fastdl): no network, a local Range-capable server."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from away.app import fastdl, models

REPO = "org/faster-whisper-test"


def _no_proxy_opener(req: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)


class Hub:
    """A fake Hugging Face Hub: tree API, /resolve/ redirect to a CDN path with Range support."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.lfs_min = 1000  # files at least this big are listed as LFS (sha256), others as git blobs (sha1)
        self.no_range = False
        self.bad_sha: set[str] = set()
        self.fail: dict[int, int] = {}  # range start -> remaining failures (HTTP 500)
        self.truncate: dict[int, int] = {}  # range start -> remaining truncated (short) responses
        self.api_down = False
        self.log: list[tuple[str, str | None]] = []
        self.lock = threading.Lock()

    def listing(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"type": "directory", "path": "sub", "oid": "x", "size": 0}]
        for path, data in self.files.items():
            e: dict[str, Any] = {"type": "file", "path": path, "size": len(data)}
            if len(data) >= self.lfs_min:
                sha = hashlib.sha256(data).hexdigest()
                e["oid"] = "gitpointer"
                e["lfs"] = {"oid": "0" * 64 if path in self.bad_sha else sha, "size": len(data), "pointerSize": 133}
            else:
                blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                e["oid"] = "f" * 40 if path in self.bad_sha else blob
            out.append(e)
        return out

    def ranges(self, path: str) -> list[str]:
        return [r for p, r in self.log if p == f"/cdn/{path}" and r]


def _handler(hub: Hub) -> type[BaseHTTPRequestHandler]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def do_GET(self) -> None:
            rng = self.headers.get("Range")
            with hub.lock:
                hub.log.append((self.path, rng))
            if self.path == f"/api/models/{REPO}/tree/main":
                if hub.api_down:
                    self.send_error(503)
                    return
                body = json.dumps(hub.listing()).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            prefix = f"/{REPO}/resolve/main/"
            if self.path.startswith(prefix):
                self.send_response(302)
                self.send_header("Location", "/cdn/" + self.path[len(prefix) :])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.startswith("/cdn/") and self.path[5:] in hub.files:
                data = hub.files[self.path[5:]]
                if rng and not hub.no_range:
                    s, e = (int(x) for x in rng.split("=")[1].split("-"))
                    e = min(e, len(data) - 1)
                    with hub.lock:
                        if hub.fail.get(s, 0) > 0:
                            hub.fail[s] -= 1
                            self.send_error(500)
                            return
                        short = hub.truncate.get(s, 0) > 0
                        if short:
                            hub.truncate[s] -= 1
                    part = data[s : e + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {s}-{e}/{len(data)}")
                    self.send_header("Content-Length", str(len(part)))
                    self.end_headers()
                    self.wfile.write(part[: len(part) // 2] if short else part)
                    if short:
                        self.close_connection = True
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

    return H


@pytest.fixture
def hub() -> Iterator[tuple[Hub, str]]:
    rnd = os.urandom
    h = Hub(
        {
            "config.json": b'{"a": 1}',
            "model.bin": rnd(10_007),  # 10 chunks of 1000 + a 7-byte tail
            "tokenizer.json": rnd(1_500),  # LFS, but under the parallel threshold: single stream
            "vocabulary.txt": b"hello\nworld\n",
            "README.md": b"not downloaded",
        }
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(h))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield h, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _run(base: str, out: Path, **kw: Any) -> list[str]:
    lines: list[str] = []
    sleeps: list[float] = kw.pop("sleeps", [])
    fastdl.download_model(
        REPO,
        out,
        lines.append,
        opener=_no_proxy_opener,
        base=base,
        chunk_size=kw.pop("chunk_size", 1000),
        threshold=kw.pop("threshold", 5000),
        sleep=sleeps.append,
        **kw,
    )
    return lines


# ---------------------------------------------------------------- planning / listing
@pytest.mark.parametrize("size,chunk", [(100, 30), (90, 30), (1, 30), (29, 30), (30, 30), (31, 30), (10_007, 1000)])
def test_plan_chunks_covers_file_exactly(size: int, chunk: int) -> None:
    chunks = fastdl.plan_chunks(size, chunk)
    assert chunks[0].start == 0 and chunks[-1].end == size - 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for a, b in itertools.pairwise(chunks):
        assert b.start == a.end + 1  # no gaps, no overlaps
    assert all(0 < c.length <= chunk for c in chunks)
    assert sum(c.length for c in chunks) == size
    assert len(chunks) == -(-size // chunk)


def test_plan_chunks_exact_values_and_edges() -> None:
    assert fastdl.plan_chunks(100, 30) == [
        fastdl.Chunk(0, 0, 29),
        fastdl.Chunk(1, 30, 59),
        fastdl.Chunk(2, 60, 89),
        fastdl.Chunk(3, 90, 99),
    ]
    assert fastdl.plan_chunks(0, 30) == []
    assert fastdl.Chunk(3, 90, 99).length == 10
    with pytest.raises(ValueError):
        fastdl.plan_chunks(10, 0)


def test_repo_for_matches_faster_whisper() -> None:
    assert fastdl.repo_for("small") == "Systran/faster-whisper-small"
    assert fastdl.repo_for("large-v3-turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert fastdl.repo_for("me/my-ct2-model") == "me/my-ct2-model"
    with pytest.raises(ValueError, match="nope"):
        fastdl.repo_for("nope")


def test_select_files_uses_faster_whisper_patterns_and_checksums() -> None:
    entries = [
        {"type": "file", "path": "model.bin", "size": 133, "oid": "ptr", "lfs": {"oid": "a" * 64, "size": 999}},
        {"type": "file", "path": "config.json", "size": 5, "oid": "b" * 40},
        {"type": "file", "path": "vocabulary.json", "size": 7, "oid": "c" * 40},
        {"type": "file", "path": "README.md", "size": 1, "oid": "d" * 40},
        {"type": "file", "path": ".gitattributes", "size": 1, "oid": "e" * 40},
        {"type": "directory", "path": "config.json", "size": 0, "oid": "f"},
    ]
    files = fastdl.select_files(entries)
    assert files == [
        fastdl.RemoteFile("model.bin", 999, sha256="a" * 64, git_sha1=None),
        fastdl.RemoteFile("config.json", 5, sha256=None, git_sha1="b" * 40),
        fastdl.RemoteFile("vocabulary.json", 7, sha256=None, git_sha1="c" * 40),
    ]
    assert fastdl.ALLOW_PATTERNS == (
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    )


def test_file_url_and_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert fastdl.file_url("o/r", "model.bin") == "https://huggingface.co/o/r/resolve/main/model.bin"
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert fastdl.file_url("o/r", "a b.json") == "https://hf-mirror.com/o/r/resolve/main/a%20b.json"


# ---------------------------------------------------------------- end to end against the local hub
def test_parallel_download_assembles_ranges_and_verifies(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    out = tmp_path / "whisper-x"
    lines = _run(base, out, connections=4)
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        assert (out / name).read_bytes() == h.files[name], name
    assert not (out / "README.md").exists()
    assert sorted(p.name for p in out.iterdir()) == ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    want = {"bytes=0-0"} | {f"bytes={c.start}-{c.end}" for c in fastdl.plan_chunks(10_007, 1000)}
    got = h.ranges("model.bin")
    assert sorted(got) == sorted(want) and len(got) == 12  # 1 probe + 11 chunks, none twice
    assert h.ranges("tokenizer.json") == []  # below the threshold: one plain stream
    total = sum(len(h.files[n]) for n in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"))
    assert lines[-2].startswith("\r") and "100%" in lines[-2] and lines[-1] == "\n"
    assert f"{total / 1e6:,.0f} MB" in lines[-2]


def test_second_run_downloads_nothing(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    _run(base, tmp_path, connections=4)
    h.log.clear()
    _run(base, tmp_path, connections=4)
    assert [p for p, _ in h.log] == [f"/api/models/{REPO}/tree/main"]


def test_lfs_checksum_mismatch_deletes_and_raises(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    h.bad_sha = {"model.bin"}
    with pytest.raises(fastdl.ChecksumError, match=r"model\.bin"):
        _run(base, tmp_path, connections=4)
    assert not (tmp_path / "model.bin").exists()
    assert not (tmp_path / "model.bin.part").exists()
    assert not (tmp_path / "model.bin.part.json").exists()


def test_small_file_git_sha1_mismatch_raises(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    h.bad_sha = {"config.json"}
    with pytest.raises(fastdl.ChecksumError, match=r"config\.json"):
        _run(base, tmp_path)
    assert not (tmp_path / "config.json").exists() and not (tmp_path / "config.json.part").exists()


def test_resume_skips_completed_chunks(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    data = h.files["model.bin"]
    chunks = fastdl.plan_chunks(len(data), 1000)
    done = [0, 1, 2, 5, 10]
    part = bytearray(len(data))
    for i in done:
        part[chunks[i].start : chunks[i].end + 1] = data[chunks[i].start : chunks[i].end + 1]
    (tmp_path / "model.bin.part").write_bytes(bytes(part))
    sha = hashlib.sha256(data).hexdigest()
    meta = {"size": len(data), "sha256": sha, "chunk_size": 1000, "done": done}
    (tmp_path / "model.bin.part.json").write_text(json.dumps(meta))
    lines = _run(base, tmp_path, connections=3)
    assert (tmp_path / "model.bin").read_bytes() == data
    fetched = {r for r in h.ranges("model.bin") if r != "bytes=0-0"}
    assert fetched == {f"bytes={c.start}-{c.end}" for c in chunks if c.index not in done}
    assert not (tmp_path / "model.bin.part.json").exists() and not (tmp_path / "model.bin.part").exists()
    assert "100%" in lines[-2]


def test_stale_sidecar_restarts_from_scratch(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    data = h.files["model.bin"]
    (tmp_path / "model.bin.part").write_bytes(b"\0" * len(data))  # nothing valid in it
    meta = {"size": len(data), "sha256": "different", "chunk_size": 1000, "done": list(range(11))}
    (tmp_path / "model.bin.part.json").write_text(json.dumps(meta))
    _run(base, tmp_path, connections=2)
    assert (tmp_path / "model.bin").read_bytes() == data
    assert len([r for r in h.ranges("model.bin") if r != "bytes=0-0"]) == 11


def test_server_without_range_falls_back_to_single_stream(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    h.no_range = True
    _run(base, tmp_path, connections=4)
    assert (tmp_path / "model.bin").read_bytes() == h.files["model.bin"]
    cdn = [r for p, r in h.log if p == "/cdn/model.bin"]
    assert cdn == ["bytes=0-0", None]  # the probe, then one plain GET


def test_transient_chunk_failures_are_retried(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    h.fail = {3000: 2}  # HTTP 500 twice
    h.truncate = {7000: 1}  # connection closes half way once
    sleeps: list[float] = []
    lines = _run(base, tmp_path, connections=4, sleeps=sleeps, backoff=0.5)
    assert (tmp_path / "model.bin").read_bytes() == h.files["model.bin"]
    assert h.ranges("model.bin").count("bytes=3000-3999") == 3
    assert h.ranges("model.bin").count("bytes=7000-7999") == 2
    assert sorted(sleeps) == [0.5, 0.5, 1.0]  # backoff doubles per attempt of the same chunk
    total = sum(len(h.files[n]) for n in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"))
    # partial bytes from failed attempts are not double counted
    assert f"{total / 1e6:,.0f}/{total / 1e6:,.0f} MB" in lines[-2]


def test_gives_up_after_max_retries(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    h.fail = {10_000: 99}  # the last (7-byte) chunk never succeeds
    sleeps: list[float] = []
    with pytest.raises(urllib.error.HTTPError):
        _run(base, tmp_path, connections=2, sleeps=sleeps, retries=4, backoff=1.0)
    assert h.ranges("model.bin").count("bytes=10000-10006") == 4
    assert sorted(sleeps) == [1.0, 2.0, 4.0]
    assert not (tmp_path / "model.bin").exists()
    # the finished chunks are recorded so the next run resumes
    meta = json.loads((tmp_path / "model.bin.part.json").read_text())
    assert set(range(9)) <= set(meta["done"]) and 10 not in meta["done"]
    h.fail.clear()
    h.log.clear()
    _run(base, tmp_path, connections=2)
    assert (tmp_path / "model.bin").read_bytes() == h.files["model.bin"]
    assert "bytes=0-999" not in h.ranges("model.bin")  # resumed, not restarted


def test_connections_one_uses_single_stream(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    _run(base, tmp_path, connections=1)
    assert (tmp_path / "model.bin").read_bytes() == h.files["model.bin"]
    assert h.ranges("model.bin") == []


def test_api_down_raises_for_fallback(hub: tuple[Hub, str], tmp_path: Path) -> None:
    h, base = hub
    h.api_down = True
    sleeps: list[float] = []
    with pytest.raises(urllib.error.HTTPError):
        _run(base, tmp_path, sleeps=sleeps, retries=2)
    assert len(sleeps) == 1


# ---------------------------------------------------------------- progress line
def test_progress_line_with_speed_and_eta() -> None:
    line = fastdl.progress_line(664_000_000, 1_620_000_000, 5_200_000)
    assert line == "  [############..................]  41%  664/1,620 MB  5.2 MB/s  ETA 3m04s"
    assert fastdl.progress_line(0, 100_000_000, None).endswith("0/100 MB  -- MB/s  ETA --")
    assert fastdl.progress_line(0, 100_000_000, 0.0).endswith("-- MB/s  ETA --")


@pytest.mark.parametrize(
    "secs,text",
    [
        (0, "0s"),
        (59.4, "59s"),
        (60, "1m00s"),
        (184, "3m04s"),
        (3600, "1h00m"),
        (3725, "1h02m"),
        (None, "--"),
        (-1, "--"),
    ],
)
def test_format_eta(secs: float | None, text: str) -> None:
    assert fastdl.format_eta(secs) == text


def test_meter_throttles_warms_up_and_excludes_resumed_bytes() -> None:
    now = [0.0]
    out: list[str] = []
    m = fastdl.Meter(100_000_000, out.append, already=40_000_000, interval=0.5, clock=lambda: now[0], warmup=1.0)
    m.add(1_000_000)  # first draw, during warmup: no speed
    assert len(out) == 1 and out[0].endswith("-- MB/s  ETA --") and " 41%" in out[0]
    now[0] = 0.2
    m.add(1_000_000)  # throttled
    assert len(out) == 1
    now[0] = 2.0
    m.add(8_000_000)  # 10 MB this run in 2 s = 5.0 MB/s; 50 MB left -> 10 s
    assert len(out) == 2
    assert out[1] == "\r" + fastdl.progress_line(50_000_000, 100_000_000, 5_000_000)
    assert out[1].endswith("5.0 MB/s  ETA 10s")
    m.finish(ok=False)
    assert out[-1] == "\n" and len(out) == 3
    m.finish(ok=True)
    assert out[-2].startswith("\r") and out[-1] == "\n"


# ---------------------------------------------------------------- download_asr: fast path + fallback
def test_download_asr_uses_fast_path(tmp_path: Path) -> None:
    calls: list[tuple[str, Path, int]] = []

    def fast(model: str, out: Path, write: Any, connections: int) -> Path:
        calls.append((model, out, connections))
        return out

    def slow(*a: Any) -> Path:
        raise AssertionError("fallback must not run")

    msgs: list[str] = []
    p = models.download_asr("small", tmp_path, msgs.append, connections=3, write=lambda s: None, fast=fast, slow=slow)
    assert p == tmp_path / "whisper-small"
    assert calls == [("small", tmp_path / "whisper-small", 3)]
    assert "3 connections" in msgs[0] and "485 MB" in msgs[0]


def test_download_asr_falls_back_on_network_error(tmp_path: Path) -> None:
    got: list[tuple[str, Path]] = []

    def fast(*a: Any, **k: Any) -> Path:
        raise urllib.error.URLError("api unreachable")

    def slow(model: str, out: Path, write: Any) -> Path:
        got.append((model, out))
        return out

    msgs: list[str] = []
    p = models.download_asr("tiny", tmp_path, msgs.append, write=lambda s: None, fast=fast, slow=slow)
    assert got == [("tiny", tmp_path / "whisper-tiny")] and p == tmp_path / "whisper-tiny"
    assert any("standard downloader" in m and "api unreachable" in m for m in msgs)


def test_download_asr_does_not_fall_back_on_checksum_mismatch(tmp_path: Path) -> None:
    def fast(*a: Any, **k: Any) -> Path:
        raise fastdl.ChecksumError("model.bin: checksum mismatch")

    def slow(*a: Any) -> Path:
        raise AssertionError("fallback must not run")

    with pytest.raises(fastdl.ChecksumError):
        models.download_asr("tiny", tmp_path, lambda s: None, write=lambda s: None, fast=fast, slow=slow)


def test_download_asr_end_to_end_against_local_hub(
    hub: tuple[Hub, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, base = hub
    monkeypatch.setenv("HF_ENDPOINT", base)
    monkeypatch.setattr(fastdl, "default_opener", _no_proxy_opener)
    monkeypatch.setattr(fastdl, "repo_for", lambda m: REPO)

    def fast(model: str, out: Path, write: Any, connections: int) -> Path:
        return fastdl.download_model(
            model, out, write, connections=connections, opener=_no_proxy_opener, chunk_size=1000, threshold=5000
        )

    p = models.download_asr("tiny", tmp_path, lambda s: None, write=lambda s: None, fast=fast)
    assert (p / "model.bin").read_bytes() == h.files["model.bin"]


def test_stdout_writer_tolerates_missing_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(sys, "stdout", None)
    models._stdout_writer()("\rabc")  # no exception


# ---------------------------------------------------------------- CLI
def test_setup_connections_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from away import cli
    from away.app import setup_wizard

    seen: dict[str, Any] = {}

    def fake_run(**kw: Any) -> setup_wizard.SetupReport:
        seen.update(kw)
        return setup_wizard.SetupReport()

    monkeypatch.setattr(setup_wizard, "run", fake_run)
    assert cli.main(["setup", "--yes", "--skip-probe", "--connections", "3"]) == 0
    assert seen["connections"] == 3
    assert cli.main(["setup", "--yes", "--skip-probe"]) == 0
    assert seen["connections"] == 8


def test_setup_wizard_passes_connections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from away.app import setup_wizard

    got: list[int] = []
    monkeypatch.setattr(models, "download_asr", lambda m, d, out, connections: got.append(connections))
    setup_wizard.run(
        yes=True,
        asr_model="tiny",
        skip_probe=True,
        out=lambda s: None,
        downloader=lambda mf, d, out: d / mf.name,
        connections=0,
    )
    assert got == [1]  # clamped to at least one connection
