"""Fast, resumable, parallel Hugging Face downloads for the Whisper models (stdlib only).

``huggingface_hub.snapshot_download`` fetches each file over one HTTP stream, which crawls on slow or
lossy links (a 1.6 GB ``large-v3-turbo`` took >50 min on some Philippine connections). This module:

* lists the repo with the Hub API (``/api/models/{repo}/tree/{rev}``) to get every file's size and
  checksum (LFS ``sha256`` for big files, the git blob ``sha1`` for small ones),
* downloads big files as N parallel HTTP ``Range`` requests pulled from a shared work queue (fast
  connections take more chunks), each written at its offset into a preallocated ``.part`` file,
* retries each chunk with exponential backoff and records finished chunks in a JSON sidecar, so a
  rerun of ``away setup`` resumes where the last one stopped,
* verifies the checksum, then atomically renames the file into place,
* draws one live progress line with speed and ETA.

The output folder has the same layout ``faster_whisper.download_model(output_dir=...)`` produces, so
the rest of the app is unchanged. Servers that ignore ``Range`` get a single-stream download.
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# faster_whisper.utils.download_model(allow_patterns=...)
ALLOW_PATTERNS = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*")
DEFAULT_CONNECTIONS = 8
CHUNK_SIZE = 16 * 1024 * 1024
PARALLEL_THRESHOLD = 32 * 1024 * 1024  # smaller files: one stream (the parallel setup costs more than it saves)
RETRIES = 5
BACKOFF_S = 1.0
TIMEOUT_S = 60.0
BLOCK = 256 * 1024

Opener = Callable[[urllib.request.Request, float], Any]  # urllib.request.urlopen(req, timeout=...) compatible


class ChecksumError(RuntimeError):
    """The downloaded bytes do not match the Hub's checksum (corrupt download or file changed upstream)."""


class RangeUnsupported(RuntimeError):
    """The server answered a Range request with the whole file."""


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None = None  # LFS files
    git_sha1: str | None = None  # small (non-LFS) files: git blob id


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int  # inclusive, like the HTTP Range header

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def default_opener(req: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(req, timeout=timeout)


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def repo_for(model: str) -> str:
    """The Hub repo faster-whisper downloads for ``model`` (``small`` -> ``Systran/faster-whisper-small``)."""
    if "/" in model:
        return model
    try:
        from faster_whisper.utils import _MODELS
    except Exception:  # faster-whisper missing or its private table moved
        _MODELS = {}
    repo = _MODELS.get(model)
    if repo is None:
        raise ValueError(f"unknown Whisper model {model!r}")
    return str(repo)


def plan_chunks(size: int, chunk_size: int = CHUNK_SIZE) -> list[Chunk]:
    """Split ``size`` bytes into contiguous inclusive ranges of at most ``chunk_size`` bytes."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [Chunk(i, s, min(s + chunk_size, size) - 1) for i, s in enumerate(range(0, size, chunk_size))]


def select_files(entries: Iterable[dict[str, Any]], patterns: Sequence[str] = ALLOW_PATTERNS) -> list[RemoteFile]:
    files = []
    for e in entries:
        if e.get("type") != "file":
            continue
        path = str(e["path"])
        if not any(fnmatch.fnmatchcase(path, p) for p in patterns):
            continue
        lfs = e.get("lfs") or {}
        files.append(
            RemoteFile(
                path,
                int(lfs.get("size", e.get("size", 0))),
                sha256=lfs.get("oid") or None,
                git_sha1=None if lfs else (e.get("oid") or None),
            )
        )
    return files


def list_files(
    repo: str,
    opener: Opener = default_opener,
    revision: str = "main",
    base: str | None = None,
    patterns: Sequence[str] = ALLOW_PATTERNS,
) -> list[RemoteFile]:
    url = f"{base or endpoint()}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}"
    with opener(_request(url), TIMEOUT_S) as r:
        entries = json.loads(r.read().decode("utf-8"))
    files = select_files(entries, patterns)
    if not any(f.path == "model.bin" for f in files):
        raise RuntimeError(f"{repo}: no model.bin in the repo listing")
    return files


def file_url(repo: str, path: str, revision: str = "main", base: str | None = None) -> str:
    return f"{base or endpoint()}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path)}"


# ---------------------------------------------------------------- progress
def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0 or seconds > 99 * 3600:
        return "--"
    s = round(seconds)
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def progress_line(done: int, total: int, rate: float | None) -> str:
    """``[####......]  41%  664/1620 MB  5.2 MB/s  ETA 3m04s``."""
    from away.app.models import progress_bar

    if not rate or rate <= 0:
        return f"{progress_bar(done, total)}  -- MB/s  ETA --"
    eta = (total - done) / rate if total > 0 else None
    return f"{progress_bar(done, total)}  {rate / 1_000_000:.1f} MB/s  ETA {format_eta(eta)}"


class Meter:
    """Thread-safe byte counter that redraws one ``\\r`` progress line at most every ``interval`` s.

    Speed counts only bytes fetched in this run (resumed bytes are not "fast"), averaged over the run
    (shown after ``warmup`` seconds).
    """

    def __init__(
        self,
        total: int,
        write: Callable[[str], object],
        already: int = 0,
        interval: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        warmup: float = 1.0,
    ) -> None:
        self.total, self.write, self.interval, self.clock, self.warmup = total, write, interval, clock, warmup
        self.done = already
        self.session = 0
        self._lock = threading.Lock()
        self._t0 = clock()
        self._last = -1e18

    def rate(self) -> float | None:
        dt = self.clock() - self._t0
        # no speed/ETA during the first ``warmup`` s: one early block would extrapolate nonsense
        return self.session / dt if dt >= self.warmup and dt > 0 and self.session > 0 else None

    def line(self) -> str:
        return progress_line(self.done, self.total, self.rate())

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n
            self.session += n
            now = self.clock()
            if now - self._last < self.interval:
                return
            self._last = now
            text = "\r" + self.line()
        self.write(text)

    def finish(self, ok: bool) -> None:
        with self._lock:
            text = "\r" + self.line()
        if ok:
            self.write(text)
        self.write("\n")


# ---------------------------------------------------------------- HTTP
def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    from away import __version__

    h = {"User-Agent": f"project-away/{__version__}", "Accept-Encoding": "identity"}
    h.update(headers or {})
    return urllib.request.Request(url, headers=h)


def _status(r: Any) -> int:
    code = getattr(r, "status", None)
    return int(code if code is not None else r.getcode())


def _content_range_start(r: Any) -> int | None:
    cr = r.headers.get("Content-Range", "") if getattr(r, "headers", None) is not None else ""
    # "bytes 0-99/75538270"
    try:
        return int(cr.split()[1].split("-")[0])
    except (IndexError, ValueError):
        return None


def _retry(fn: Callable[[], Any], retries: int, backoff: float, sleep: Callable[[float], None]) -> Any:
    for attempt in range(retries):
        try:
            return fn()
        except (ChecksumError, RangeUnsupported):
            raise
        except Exception:
            if attempt == retries - 1:
                raise
            sleep(backoff * (2**attempt))
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- checksums / sidecar
def file_digest(path: Path, algo: str, git_size: int | None = None) -> str:
    h = hashlib.new(algo)
    if git_size is not None:
        h.update(f"blob {git_size}\0".encode())
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def matches(path: Path, rf: RemoteFile) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != rf.size:
            return False
    except OSError:
        return False
    if rf.sha256:
        return file_digest(path, "sha256") == rf.sha256
    if rf.git_sha1:
        return file_digest(path, "sha1", git_size=rf.size) == rf.git_sha1
    return True


def _verify_or_raise(tmp: Path, rf: RemoteFile, sidecar: Path | None = None) -> None:
    if rf.sha256:
        got, want = file_digest(tmp, "sha256"), rf.sha256
    elif rf.git_sha1:
        got, want = file_digest(tmp, "sha1", git_size=rf.size), rf.git_sha1
    else:
        return
    if got != want:
        tmp.unlink(missing_ok=True)
        if sidecar is not None:
            sidecar.unlink(missing_ok=True)
        raise ChecksumError(f"{rf.path}: checksum mismatch (expected {want[:12]}..., got {got[:12]}...)")


def _load_sidecar(sidecar: Path, tmp: Path, rf: RemoteFile, chunk_size: int) -> set[int]:
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            meta.get("size") == rf.size
            and meta.get("sha256") == rf.sha256
            and meta.get("chunk_size") == chunk_size
            and tmp.stat().st_size == rf.size
        ):
            return {int(i) for i in meta.get("done", [])}
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return set()


def _save_sidecar(sidecar: Path, rf: RemoteFile, chunk_size: int, done: set[int]) -> None:
    meta = {"size": rf.size, "sha256": rf.sha256, "chunk_size": chunk_size, "done": sorted(done)}
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(sidecar)


# ---------------------------------------------------------------- one file
def _single_stream(
    url: str, tmp: Path, meter: Meter, opener: Opener, retries: int, backoff: float, sleep: Callable[[float], None]
) -> None:
    def attempt() -> None:
        written = 0
        try:
            with opener(_request(url), TIMEOUT_S) as r, tmp.open("wb") as fh:
                for block in iter(lambda: r.read(BLOCK), b""):
                    fh.write(block)
                    written += len(block)
                    meter.add(len(block))
        except BaseException:
            meter.add(-written)
            raise

    _retry(attempt, retries, backoff, sleep)


def _probe(url: str, opener: Opener) -> tuple[str, bool]:
    """Follow redirects to the CDN with a 1-byte Range request: (final url, server honours Range)."""
    with opener(_request(url, {"Range": "bytes=0-0"}), TIMEOUT_S) as r:
        final = r.geturl() if hasattr(r, "geturl") else url
        ok = _status(r) == 206 and _content_range_start(r) == 0
        r.read(1)
    return str(final or url), ok


def _fetch_chunk(url: str, tmp: Path, ch: Chunk, meter: Meter, opener: Opener, stop: threading.Event) -> None:
    written = 0
    try:
        with opener(_request(url, {"Range": f"bytes={ch.start}-{ch.end}"}), TIMEOUT_S) as r:
            if _status(r) != 206 or _content_range_start(r) != ch.start:
                raise RangeUnsupported(f"server ignored Range for bytes {ch.start}-{ch.end}")
            with tmp.open("r+b") as fh:
                fh.seek(ch.start)
                while written < ch.length:
                    if stop.is_set():
                        raise RuntimeError("download cancelled")
                    block = r.read(min(BLOCK, ch.length - written))
                    if not block:
                        raise ConnectionError(f"connection closed at {written}/{ch.length} bytes of chunk {ch.index}")
                    fh.write(block)
                    written += len(block)
                    meter.add(len(block))
    except BaseException:
        meter.add(-written)
        raise


def download_file(
    url: str,
    dest: Path,
    rf: RemoteFile,
    meter: Meter,
    connections: int = DEFAULT_CONNECTIONS,
    chunk_size: int = CHUNK_SIZE,
    threshold: int = PARALLEL_THRESHOLD,
    opener: Opener = default_opener,
    retries: int = RETRIES,
    backoff: float = BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download ``url`` to ``dest`` (verified, atomic). Big files go in parallel ranges and can resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    sidecar = dest.with_name(dest.name + ".part.json")

    if rf.size < threshold or connections <= 1:
        _single_stream(url, tmp, meter, opener, retries, backoff, sleep)
        sidecar.unlink(missing_ok=True)
    else:
        final, ranged = _retry(lambda: _probe(url, opener), retries, backoff, sleep)
        if not ranged:
            _single_stream(final, tmp, meter, opener, retries, backoff, sleep)
            sidecar.unlink(missing_ok=True)
        else:
            _parallel(final, tmp, sidecar, rf, meter, connections, chunk_size, opener, retries, backoff, sleep)

    if tmp.stat().st_size != rf.size:
        got = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise ConnectionError(f"{rf.path}: got {got} bytes, expected {rf.size}")
    _verify_or_raise(tmp, rf, sidecar)
    tmp.replace(dest)
    sidecar.unlink(missing_ok=True)
    return dest


def _parallel(
    url: str,
    tmp: Path,
    sidecar: Path,
    rf: RemoteFile,
    meter: Meter,
    connections: int,
    chunk_size: int,
    opener: Opener,
    retries: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> None:
    chunks = plan_chunks(rf.size, chunk_size)
    done = _load_sidecar(sidecar, tmp, rf, chunk_size)
    if done:
        meter.done += sum(c.length for c in chunks if c.index in done)  # resumed bytes: progress, not speed
    else:
        with tmp.open("wb") as fh:  # preallocate so every worker can write at its offset
            fh.truncate(rf.size)
        _save_sidecar(sidecar, rf, chunk_size, done)

    work: queue.Queue[Chunk] = queue.Queue()
    for c in chunks:
        if c.index not in done:
            work.put(c)
    stop = threading.Event()
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        while not stop.is_set():
            try:
                ch = work.get_nowait()
            except queue.Empty:
                return
            try:
                _retry(functools.partial(_fetch_chunk, url, tmp, ch, meter, opener, stop), retries, backoff, sleep)
            except BaseException as exc:
                with lock:
                    errors.append(exc)
                stop.set()
                return
            with lock:
                done.add(ch.index)
                _save_sidecar(sidecar, rf, chunk_size, done)

    threads = [
        threading.Thread(target=worker, name=f"away-dl-{i}", daemon=True)
        for i in range(max(1, min(connections, work.qsize())))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


# ---------------------------------------------------------------- whole model
def download_model(
    model: str,
    out_dir: Path,
    write: Callable[[str], object],
    connections: int = DEFAULT_CONNECTIONS,
    opener: Opener = default_opener,
    base: str | None = None,
    revision: str = "main",
    chunk_size: int = CHUNK_SIZE,
    threshold: int = PARALLEL_THRESHOLD,
    retries: int = RETRIES,
    backoff: float = BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    interval: float = 0.5,
    warmup: float = 1.0,
) -> Path:
    """Download the faster-whisper files of ``model`` into ``out_dir``; returns ``out_dir``."""
    repo = repo_for(model)
    files = _retry(lambda: list_files(repo, opener, revision, base), retries, backoff, sleep)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not matches(out_dir / f.path, f)]
    total = sum(f.size for f in files)
    meter = Meter(
        total,
        write,
        already=total - sum(f.size for f in todo),
        interval=interval,
        clock=clock,
        warmup=warmup,
    )
    ok = False
    try:
        for rf in todo:
            download_file(
                file_url(repo, rf.path, revision, base),
                out_dir / rf.path,
                rf,
                meter,
                connections=connections,
                chunk_size=chunk_size,
                threshold=threshold,
                opener=opener,
                retries=retries,
                backoff=backoff,
                sleep=sleep,
            )
        ok = True
    finally:
        meter.finish(ok)
    return out_dir
