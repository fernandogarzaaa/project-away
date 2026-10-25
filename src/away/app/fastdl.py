"""Fast, resumable, parallel Hugging Face downloads for the Whisper models (stdlib only).

``huggingface_hub.snapshot_download`` fetches each file over one HTTP stream, which crawls on slow or
lossy links (a 1.6 GB ``large-v3-turbo`` took >50 min on some Philippine connections). This module:

* lists the repo with the Hub API (``/api/models/{repo}/tree/{rev}``) to get every file's size and
  checksum (LFS ``sha256`` for big files, the git blob ``sha1`` for small ones),
* downloads big files as N parallel HTTP ``Range`` requests pulled from a shared work queue (fast
  connections take more chunks), each written at its offset into a preallocated ``.part`` file,
* retries each chunk with exponential backoff, continuing a broken chunk from the byte where it stopped,
  and records finished (and partly finished) chunks in a JSON sidecar, so a rerun of ``away setup``
  resumes where the last one stopped (including a ``.part`` left by v0.1.4),
* re-resolves the Hub ``/resolve/`` URL when the signed CDN link it redirected to expires (HTTP
  401/403/410 after ~1 h on ``cas-bridge.xethub.hf.co`` / ``cdn-lfs``) and on every retry, so a slow
  download is never abandoned half way,
* verifies the checksum, then atomically renames the file into place,
* draws one live progress line with speed and ETA, and suggests the smaller model once when the
  measured speed is very slow.

The output folder has the same layout ``faster_whisper.download_model(output_dir=...)`` produces, so
the rest of the app is unchanged. Servers that ignore ``Range`` get a single-stream download.
"""

from __future__ import annotations

import contextlib
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
MAX_BACKOFF_S = 30.0
EXPIRED_STATUSES = (401, 403, 410)  # what an expired signed CDN URL answers
MAX_REFRESHES = 50  # URL re-resolves per chunk before giving up (an hour-long link expires ~once an hour)
SIDECAR_EVERY = 4 * 1024 * 1024  # persist partial-chunk progress at least this often
SLOW_BPS = 500_000  # below 0.5 MB/s, suggest the smaller model
SLOW_AFTER_S = 20.0
TIMEOUT_S = 60.0
BLOCK = 256 * 1024

Opener = Callable[[urllib.request.Request, float], Any]  # urllib.request.urlopen(req, timeout=...) compatible


class ChecksumError(RuntimeError):
    """The downloaded bytes do not match the Hub's checksum (corrupt download or file changed upstream)."""


class RangeUnsupported(RuntimeError):
    """The server answered a Range request with the whole file."""


class UrlExpired(RuntimeError):
    """The signed CDN URL stopped working (401/403/410): resolve the Hub URL again for a fresh one."""


def is_expired(exc: BaseException) -> bool:
    import urllib.error

    return isinstance(exc, UrlExpired) or (
        isinstance(exc, urllib.error.HTTPError) and int(getattr(exc, "code", 0) or 0) in EXPIRED_STATUSES
    )


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


# Copy of faster_whisper.utils._MODELS (1.2): used when faster-whisper is missing or its private table moved.
KNOWN_REPOS = {
    **{
        m: f"Systran/faster-whisper-{m}"
        for m in ("tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium")
    },
    **{m: f"Systran/faster-whisper-{m}" for m in ("large-v1", "large-v2", "large-v3")},
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def repo_for(model: str) -> str:
    """The Hub repo faster-whisper downloads for ``model`` (``small`` -> ``Systran/faster-whisper-small``)."""
    if "/" in model:
        return model
    try:
        from faster_whisper.utils import _MODELS
    except Exception:  # faster-whisper missing or its private table moved
        _MODELS = {}
    repo = _MODELS.get(model) or KNOWN_REPOS.get(model)
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
    required: Sequence[str] = ("model.bin",),
) -> list[RemoteFile]:
    """The repo's files matching ``patterns``. ``required``: at least one of these must be listed (v0.1.35: the
    MLX Whisper repos ship ``weights.npz`` or ``weights.safetensors`` instead of ``model.bin``)."""
    url = f"{base or endpoint()}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}"
    with opener(_request(url), TIMEOUT_S) as r:
        entries = json.loads(r.read().decode("utf-8"))
    files = select_files(entries, patterns)
    if required and not any(f.path in required for f in files):
        raise RuntimeError(f"{repo}: no {' or '.join(required)} in the repo listing")
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
        on_slow: Callable[[float, float | None], object] | None = None,
        slow_bps: float = SLOW_BPS,
        slow_after: float = SLOW_AFTER_S,
    ) -> None:
        self.total, self.write, self.interval, self.clock, self.warmup = total, write, interval, clock, warmup
        self.done = already
        self.session = 0
        self._lock = threading.Lock()
        self._t0 = clock()
        self._last = -1e18
        self.on_slow, self.slow_bps, self.slow_after = on_slow, slow_bps, slow_after
        self._slow_told = False

    def _check_slow(self, now: float) -> None:
        """Call ``on_slow(rate, eta_s)`` once when, after ``slow_after`` s, the speed is below ``slow_bps``."""
        if self.on_slow is None or self._slow_told or now - self._t0 < self.slow_after:
            return
        rate = self.rate()
        if rate is None or rate >= self.slow_bps:
            return
        self._slow_told = True
        left = max(self.total - self.done, 0)
        self.write("\n")
        self.on_slow(rate, left / rate if rate > 0 else None)

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
        self._check_slow(now)

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


def _load_sidecar(sidecar: Path, tmp: Path, rf: RemoteFile, chunk_size: int) -> tuple[set[int], dict[int, int]]:
    """(finished chunk indexes, {chunk index: bytes already written from its start}) of a valid sidecar.

    Accepts sidecars written with another chunk size (v0.1.4 wrote 16 MiB chunks without ``partial``):
    the byte ranges they cover are mapped onto the current chunks so nothing already fetched is lost."""
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("size") != rf.size or meta.get("sha256") != rf.sha256 or tmp.stat().st_size != rf.size:
            return set(), {}
        old_cs = int(meta.get("chunk_size"))
        if old_cs <= 0:
            return set(), {}
        old = {c.index: c for c in plan_chunks(rf.size, old_cs)}
        covered: list[tuple[int, int]] = []  # [start, end) byte ranges known to be on disk
        for i in meta.get("done", []):
            c = old.get(int(i))
            if c is not None:
                covered.append((c.start, c.end + 1))
        for i, n in (meta.get("partial") or {}).items():
            c = old.get(int(i))
            if c is not None and int(n) > 0:
                covered.append((c.start, c.start + min(int(n), c.length)))
    except (OSError, ValueError, TypeError, AttributeError):
        return set(), {}
    done: set[int] = set()
    partial: dict[int, int] = {}
    for c in plan_chunks(rf.size, chunk_size):
        have = _prefix_covered(c.start, c.end + 1, covered)
        if have >= c.length:
            done.add(c.index)
        elif have > 0:
            partial[c.index] = have
    return done, partial


def _prefix_covered(start: int, end: int, ranges: list[tuple[int, int]]) -> int:
    """How many bytes from ``start`` (up to ``end``) are contiguously covered by ``ranges``."""
    pos = start
    for s, e in sorted(ranges):
        if s <= pos < e:
            pos = min(e, end)
        if pos >= end:
            break
    return pos - start


def _save_sidecar(
    sidecar: Path, rf: RemoteFile, chunk_size: int, done: set[int], partial: dict[int, int] | None = None
) -> None:
    meta: dict[str, Any] = {"size": rf.size, "sha256": rf.sha256, "chunk_size": chunk_size, "done": sorted(done)}
    if partial:
        meta["partial"] = {str(k): v for k, v in sorted(partial.items()) if v > 0 and k not in done}
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


class Resolver:
    """The current signed CDN URL for a Hub ``/resolve/`` URL, shared by the download workers.

    When a worker sees the URL expire (or any error) it calls :meth:`refresh` with the generation it used;
    only the first caller per generation re-probes, the others reuse the fresh URL."""

    def __init__(self, url: str, opener: Opener, final: str | None = None) -> None:
        self.url, self.opener = url, opener
        self.final = final or url
        self.generation = 0
        self.refreshes = 0
        self._lock = threading.Lock()

    def current(self) -> tuple[str, int]:
        with self._lock:
            return self.final, self.generation

    def refresh(self, seen_generation: int) -> str:
        with self._lock:
            if seen_generation == self.generation:
                final, _ranged = _probe(self.url, self.opener)
                self.final = final
                self.generation += 1
                self.refreshes += 1
            return self.final


def _fetch_chunk(
    url: str,
    tmp: Path,
    ch: Chunk,
    meter: Meter,
    opener: Opener,
    stop: threading.Event,
    have: int = 0,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Fetch the rest of ``ch`` (its first ``have`` bytes are already on disk). Returns bytes on disk for
    ``ch``; on failure raises with ``progress`` already told how far it got (those bytes are kept)."""
    start = ch.start + have
    written = have
    try:
        with opener(_request(url, {"Range": f"bytes={start}-{ch.end}"}), TIMEOUT_S) as r:
            if _status(r) != 206 or _content_range_start(r) != start:
                raise RangeUnsupported(f"server ignored Range for bytes {start}-{ch.end}")
            with tmp.open("r+b") as fh:
                fh.seek(start)
                while written < ch.length:
                    if stop.is_set():
                        raise RuntimeError("download cancelled")
                    block = r.read(min(BLOCK, ch.length - written))
                    if not block:
                        raise ConnectionError(f"connection closed at {written}/{ch.length} bytes of chunk {ch.index}")
                    fh.write(block)
                    written += len(block)
                    meter.add(len(block))
                    if progress is not None:
                        progress(written)
    except BaseException as exc:
        if progress is not None:
            progress(written)
        if is_expired(exc) and not isinstance(exc, UrlExpired):
            raise UrlExpired(f"download link expired (HTTP {getattr(exc, 'code', '?')})") from exc
        raise
    return written


def fetch_chunk_with_retries(
    resolver: Resolver,
    tmp: Path,
    ch: Chunk,
    meter: Meter,
    stop: threading.Event,
    have: int = 0,
    progress: Callable[[int], None] | None = None,
    retries: int = RETRIES,
    backoff: float = BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
    max_refreshes: int = MAX_REFRESHES,
) -> None:
    """One chunk, robustly: an expired link is re-resolved immediately (does not use up a retry); any other
    error waits with exponential backoff, re-resolves the link, and continues from the last byte written."""
    attempt = refreshed = 0
    pos = [have]

    def track(n: int) -> None:
        pos[0] = n
        if progress is not None:
            progress(n)

    while True:
        url, gen = resolver.current()
        try:
            _fetch_chunk(url, tmp, ch, meter, resolver.opener, stop, pos[0], track)
            return
        except (ChecksumError, RangeUnsupported):
            raise
        except UrlExpired:
            refreshed += 1
            if refreshed > max_refreshes or stop.is_set():
                raise
            _retry(functools.partial(resolver.refresh, gen), retries, backoff, sleep)
        except Exception:
            attempt += 1
            if attempt >= retries or stop.is_set():
                raise
            sleep(min(backoff * (2 ** (attempt - 1)), MAX_BACKOFF_S))
            # a fresh link per retry (the old one may be the problem); on failure keep the old link and let the
            # next attempt report the real error
            with contextlib.suppress(Exception):
                resolver.refresh(gen)


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
            resolver = Resolver(url, opener, final)
            _parallel(resolver, tmp, sidecar, rf, meter, connections, chunk_size, retries, backoff, sleep)

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
    resolver: Resolver,
    tmp: Path,
    sidecar: Path,
    rf: RemoteFile,
    meter: Meter,
    connections: int,
    chunk_size: int,
    retries: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> None:
    chunks = plan_chunks(rf.size, chunk_size)
    done, partial = _load_sidecar(sidecar, tmp, rf, chunk_size)
    if done or partial:
        # resumed bytes count as progress, not speed
        meter.done += sum(c.length for c in chunks if c.index in done) + sum(partial.values())
    else:
        with tmp.open("wb") as fh:  # preallocate so every worker can write at its offset
            fh.truncate(rf.size)
    _save_sidecar(sidecar, rf, chunk_size, done, partial)

    work: queue.Queue[Chunk] = queue.Queue()
    for c in chunks:
        if c.index not in done:
            work.put(c)
    stop = threading.Event()
    errors: list[BaseException] = []
    lock = threading.Lock()
    saved_at: dict[int, int] = {}

    def record(idx: int, n: int, force: bool = False) -> None:
        with lock:
            partial[idx] = n
            if force or n - saved_at.get(idx, 0) >= SIDECAR_EVERY:
                saved_at[idx] = n
                _save_sidecar(sidecar, rf, chunk_size, done, partial)

    def worker() -> None:
        while not stop.is_set():
            try:
                ch = work.get_nowait()
            except queue.Empty:
                return
            try:
                fetch_chunk_with_retries(
                    resolver,
                    tmp,
                    ch,
                    meter,
                    stop,
                    have=partial.get(ch.index, 0),
                    progress=functools.partial(record, ch.index),
                    retries=retries,
                    backoff=backoff,
                    sleep=sleep,
                )
            except BaseException as exc:
                with lock:
                    errors.append(exc)
                    _save_sidecar(sidecar, rf, chunk_size, done, partial)  # keep every byte for the next run
                stop.set()
                return
            with lock:
                done.add(ch.index)
                partial.pop(ch.index, None)
                _save_sidecar(sidecar, rf, chunk_size, done, partial)

    threads = [
        threading.Thread(target=worker, name=f"away-dl-{i}", daemon=True)
        for i in range(max(1, min(connections, work.qsize())))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        err = errors[0]
        if isinstance(err, UrlExpired) and err.__cause__ is not None:
            raise err.__cause__
        raise err


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
    on_slow: Callable[[float, float | None], object] | None = None,
    repo: str | None = None,
    patterns: Sequence[str] = ALLOW_PATTERNS,
    required: Sequence[str] = ("model.bin",),
    expect: dict[str, str] | None = None,
) -> Path:
    """Download the faster-whisper files of ``model`` into ``out_dir``; returns ``out_dir``.

    ``on_slow(rate_bytes_per_s, eta_s)`` is called once if the connection turns out to be very slow.

    v0.1.35: ``repo`` / ``patterns`` / ``required`` fetch another layout the same way (the MLX Whisper weights for
    Apple Silicon, :data:`away.app.models.MLX_MODELS`, pinned to a commit with ``revision``). ``expect`` maps a
    file name to the sha256 it must have; the Hub listing is checked against it before anything is downloaded."""
    repo = repo or repo_for(model)
    files = _retry(lambda: list_files(repo, opener, revision, base, patterns, required), retries, backoff, sleep)
    for rf in files:
        want = (expect or {}).get(rf.path)
        if want and rf.sha256 != want:
            raise ChecksumError(f"{repo}@{revision[:12]}: {rf.path} is not the pinned file (sha256 {rf.sha256})")
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
        on_slow=on_slow,
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
