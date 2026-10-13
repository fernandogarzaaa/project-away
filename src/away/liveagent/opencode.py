"""The opencode engine: free models with no API key.

opencode's free models only answer requests made from within opencode, so Away runs opencode itself, privately:

* **Find it.** An opencode you already installed (``opencode`` on PATH, npm, ``~/.opencode/bin``) is used as is.
* **Or download it (one click, no Node/npm).** The official standalone build for this PC from opencode's GitHub
  releases (anomalyco/opencode, formerly sst/opencode; MIT license) goes to ``<data>/engines/opencode/<version>``.
  The download is checked against the SHA-256 GitHub publishes for the file; a mismatch is thrown away.
  Updated from releases when Away checks for updates.
* **Run it privately.** ``opencode serve`` on a random port of 127.0.0.1 only, with a random password (HTTP basic
  auth), its own config/data folders under ``<data>/engines/opencode/home`` (your own opencode setup and keys are
  not touched, and API keys in your environment are not passed to it), an empty working folder, and every tool it
  might want to use (shell, files, web) answered "no" automatically. It only ever writes text for Away.
* It stops when Away quits, when the live agent is turned off, or after a long idle time.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform as _platform
import queue
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away.liveagent.http import Client, HttpError

REPOS = ("anomalyco/opencode", "sst/opencode")
RELEASES_API = "https://api.github.com/repos/{repo}/releases/latest"
LICENSE_NOTE = (
    "opencode is free, open-source software (MIT license) by Anomaly (anomalyco/opencode on GitHub). "
    "Away downloads the official build and checks its fingerprint (SHA-256) before using it."
)
CREATE_NO_WINDOW = 0x08000000
SECRET_ENV_MARKERS = ("token", "secret", "api_key", "apikey", "password", "credential", "authorization", "bearer")


class EngineError(RuntimeError):
    """A plain-words problem with the opencode engine."""


# ------------------------------------------------------------------------------------------------ paths
def engine_root(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        from away import paths

        data_dir = paths.data_dir()
    return Path(data_dir) / "engines" / "opencode"


def exe_name(platform: str | None = None) -> str:
    return "opencode.exe" if (platform or sys.platform) == "win32" else "opencode"


# ------------------------------------------------------------------------------------------------ find
def find_cli(
    which: Callable[[str], str | None] = shutil.which,
    home: Path | None = None,
    platform: str | None = None,
    exists: Callable[[Path], bool] = Path.is_file,
) -> list[str] | None:
    """The command for an opencode the person installed themselves, or None."""
    plat = platform or sys.platform
    for name in (("opencode.exe", "opencode.cmd", "opencode") if plat == "win32" else ("opencode",)):
        p = which(name)
        if p:
            return command_for(p, plat)
    h = home or Path.home()
    for cand in (h / ".opencode" / "bin" / exe_name(plat), h / ".local" / "bin" / exe_name(plat)):
        if exists(cand):
            return command_for(str(cand), plat)
    return None


def command_for(path: str, platform: str | None = None) -> list[str]:
    """npm installs a ``.cmd`` shim on Windows: it has to run through ``cmd /c``."""
    if (platform or sys.platform) == "win32" and path.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", path]
    return [path]


def installed_engine(data_dir: Path | None = None) -> dict[str, Any] | None:
    """``{"version", "path", "sha256", ...}`` of the engine Away downloaded, or None."""
    root = engine_root(data_dir)
    try:
        info = json.loads((root / "current.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    p = Path(str(info.get("path", "")))
    if not p.is_file():
        return None
    return info


def resolve(data_dir: Path | None = None, which: Callable[[str], str | None] = shutil.which) -> tuple[list[str], str]:
    """``(command, where)`` for the engine to run: the downloaded one first (known version, private), else the
    person's own CLI. Raises :class:`EngineError` when neither exists."""
    eng = installed_engine(data_dir)
    if eng is not None:
        return [str(eng["path"])], f"Away's copy (v{eng.get('version', '?')})"
    cli = find_cli(which)
    if cli is not None:
        return cli, "your installed opencode"
    raise EngineError("opencode is not installed yet")


# ------------------------------------------------------------------------------------------------ releases
@dataclass
class Asset:
    name: str
    url: str
    size: int
    sha256: str
    version: str


def asset_candidates(platform: str | None = None, machine: str | None = None, musl: bool | None = None) -> list[str]:
    """Release file names for this PC, best first (the "baseline" build is for older CPUs without AVX2)."""
    plat = platform or sys.platform
    m = (machine or _platform.machine() or "").lower()
    arm = m in ("arm64", "aarch64", "armv8", "armv8l")
    arch = "arm64" if arm else "x64"
    if plat == "win32":
        names = [f"opencode-windows-{arch}.zip"]
        if not arm:
            names.append("opencode-windows-x64-baseline.zip")
        return names
    if plat == "darwin":
        names = [f"opencode-darwin-{arch}.zip"]
        if not arm:
            names.append("opencode-darwin-x64-baseline.zip")
        return names
    if musl is None:
        musl = Path("/etc/alpine-release").exists()
    suffix = "-musl" if musl else ""
    names = [f"opencode-linux-{arch}{suffix}.tar.gz"]
    if not arm:
        names.append(f"opencode-linux-x64-baseline{suffix}.tar.gz")
    return names


def latest_release(client: Client | None = None, repos: tuple[str, ...] = REPOS) -> dict[str, Any]:
    client = client or Client()
    last: Exception | None = None
    for repo in repos:
        try:
            data = client.get_json(RELEASES_API.format(repo=repo), {"Accept": "application/vnd.github+json"}, 20)
            if isinstance(data, dict) and data.get("assets"):
                return data
        except Exception as exc:
            last = exc
    raise EngineError(f"could not reach opencode's releases on GitHub ({last})" if last else "no release found")


def pick_asset(release: dict[str, Any], names: list[str]) -> Asset:
    version = str(release.get("tag_name") or "").lstrip("v")
    by_name = {str(a.get("name")): a for a in release.get("assets") or [] if isinstance(a, dict)}
    for n in names:
        a = by_name.get(n)
        if a is None:
            continue
        digest = str(a.get("digest") or "")
        sha = digest.split(":", 1)[1] if digest.startswith("sha256:") else ""
        if len(sha) != 64:
            raise EngineError(f"opencode {version}: GitHub publishes no fingerprint for {n}, so it was not downloaded")
        return Asset(n, str(a.get("browser_download_url")), int(a.get("size") or 0), sha.lower(), version)
    raise EngineError(f"opencode {version} has no download for this computer ({', '.join(names)})")


# ------------------------------------------------------------------------------------------------ download
def download(
    url: str, dest: Path, progress: Callable[[int, int], None] | None = None, total: int = 0,
    timeout: float = 60.0, cancel: threading.Event | None = None,
) -> str:  # pragma: no cover - network (tests inject their own downloader)
    """Streams ``url`` to ``dest``; returns the SHA-256 of what arrived."""
    import urllib.request

    from away.liveagent.http import user_agent

    req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    h = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or total or 0)
        while True:
            if cancel is not None and cancel.is_set():
                raise EngineError("download cancelled")
            buf = resp.read(1 << 16)
            if not buf:
                break
            f.write(buf)
            h.update(buf)
            done += len(buf)
            if progress is not None:
                progress(done, total)
    return h.hexdigest()


def _safe_extract(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = (dest / info.filename).resolve()
                if not str(target).startswith(str(dest)):
                    raise EngineError("the download contains an unsafe file path")
            z.extractall(dest)
    else:
        with tarfile.open(archive, "r:gz") as t:
            for member in t.getmembers():
                target = (dest / member.name).resolve()
                if not str(target).startswith(str(dest)) or member.issym() or member.islnk():
                    raise EngineError("the download contains an unsafe file path")
            t.extractall(dest)  # noqa: S202 - every member checked above


def install(
    data_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    client: Client | None = None,
    downloader: Callable[..., str] = download,
    platform: str | None = None,
    machine: str | None = None,
    release: dict[str, Any] | None = None,
    names: list[str] | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Downloads, verifies and unpacks the newest opencode build; returns the new ``current.json`` info."""
    root = engine_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    rel = release if release is not None else latest_release(client)
    asset = pick_asset(rel, names or asset_candidates(platform, machine))
    part = root / (asset.name + ".part")
    try:
        got = downloader(asset.url, part, progress, asset.size, cancel=cancel)
    except EngineError:
        with contextlib.suppress(OSError):
            part.unlink()
        raise
    except Exception as exc:
        with contextlib.suppress(OSError):
            part.unlink()
        raise EngineError(f"the download failed ({exc})") from None
    if str(got).lower() != asset.sha256:
        with contextlib.suppress(OSError):
            part.unlink()
        raise EngineError("the downloaded file did not match its published fingerprint, so it was deleted")
    target = root / asset.version
    tmp = root / f".{asset.version}.{os.getpid()}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    archive = tmp / asset.name
    os.replace(part, archive)
    _safe_extract(archive, tmp)
    archive.unlink()
    exe = next((p for p in tmp.rglob(exe_name(platform)) if p.is_file()), None)
    if exe is None:
        shutil.rmtree(tmp, ignore_errors=True)
        raise EngineError("the download did not contain the opencode program")
    if (platform or sys.platform) != "win32":
        exe.chmod(0o755)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(tmp, target)
    exe_final = target / exe.relative_to(tmp)
    info = {
        "version": asset.version,
        "path": str(exe_final),
        "asset": asset.name,
        "sha256": asset.sha256,
        "size": asset.size,
        "installed": time.time(),
        "license": "MIT",
    }
    (root / "current.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    for old in root.iterdir():  # keep only the version in use
        if old.is_dir() and old.name not in (asset.version, "home", "work") and not old.name.startswith("."):
            shutil.rmtree(old, ignore_errors=True)
    return info


def _newer(a: str, b: str) -> bool:
    def key(v: str) -> tuple[int, ...]:
        out = []
        for part in v.lstrip("v").split("."):
            digits = "".join(c for c in part if c.isdigit())
            out.append(int(digits or 0))
        return tuple(out)

    return key(a) > key(b)


def check_update(data_dir: Path | None = None, client: Client | None = None, apply: bool = True,
                 **kw: Any) -> dict[str, Any]:
    """Called when Away checks for updates. Only updates a copy Away downloaded itself (never your own CLI)."""
    cur = installed_engine(data_dir)
    if cur is None:
        return {"updated": False, "reason": "Away has no copy of opencode"}
    rel = latest_release(client)
    latest = str(rel.get("tag_name") or "").lstrip("v")
    if not latest or not _newer(latest, str(cur.get("version", "0"))):
        return {"updated": False, "version": cur.get("version"), "latest": latest}
    if not apply:
        return {"updated": False, "available": latest, "version": cur.get("version")}
    info = install(data_dir, client=client, release=rel, **kw)
    return {"updated": True, "version": info["version"], "previous": cur.get("version")}


# ------------------------------------------------------------------------------------------------ server
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def private_env(home: Path, password: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for the private server: no API keys or tokens from the person's environment, its own
    config/data/cache folders, no self-update, no telemetry sharing."""
    src = dict(os.environ if base is None else base)
    env = {k: v for k, v in src.items() if not any(m in k.lower() for m in SECRET_ENV_MARKERS)}
    for drop in [k for k in env if k.upper().endswith("_KEY") or k.upper().startswith(("OPENCODE_", "AWS_"))]:
        env.pop(drop, None)
    env.update(
        {
            "OPENCODE_SERVER_PASSWORD": password,
            "OPENCODE_SERVER_USERNAME": "opencode",
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_DATA_HOME": str(home / "data"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "XDG_STATE_HOME": str(home / "state"),
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
        }
    )
    return env


PRIVATE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "autoupdate": False,
    "share": "disabled",
    # every tool asks, and Away always answers "no" (a denied tool would be removed from the request, which the
    # free tier refuses; "ask" keeps the request normal and Away rejects each ask)
    "permission": {"*": "ask"},
}


class OpenCodeServer:
    """A private ``opencode serve``. ``command`` is the engine command (tests pass a fake server script)."""

    def __init__(
        self,
        command: list[str],
        home: Path,
        client: Client | None = None,
        start_timeout: float = 120.0,
        idle_stop_s: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.command = list(command)
        self.home = Path(home)
        self.client = client or Client()
        self.start_timeout = float(start_timeout)
        self.idle_stop_s = float(idle_stop_s)
        self.clock = clock
        self.proc: subprocess.Popen[bytes] | None = None
        self.port = 0
        self.password = ""
        self._lock = threading.RLock()
        self._subs: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._events: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_used = clock()
        self.log_path = self.home / "serve.log"
        self.rejected = 0  # tool asks answered "no" (shown in the turn log)
        self.models_cache: dict[str, Any] = {}

    # ------------------------------------------------------------ lifecycle
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _auth(self) -> dict[str, str]:
        import base64

        tok = base64.b64encode(f"opencode:{self.password}".encode()).decode()
        return {"Authorization": f"Basic {tok}"}

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self) -> OpenCodeServer:
        with self._lock:
            self.last_used = self.clock()
            if not self.running():
                self.start()
            return self

    def start(self) -> None:
        with self._lock:
            self.stop()
            for sub in ("config/opencode", "data", "cache", "state"):
                (self.home / sub).mkdir(parents=True, exist_ok=True)
            work = self.home.parent / "work"
            work.mkdir(parents=True, exist_ok=True)
            (self.home / "config" / "opencode" / "opencode.json").write_text(
                json.dumps(PRIVATE_CONFIG, indent=2), encoding="utf-8"
            )
            self.port = free_port()
            self.password = secrets.token_urlsafe(24)
            env = private_env(self.home, self.password)
            flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
            log = open(self.log_path, "wb")  # noqa: SIM115 - handed to the child
            try:
                self.proc = subprocess.Popen(
                    [*self.command, "serve", "--hostname", "127.0.0.1", "--port", str(self.port)],
                    cwd=str(work),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
            except OSError as exc:
                log.close()
                raise EngineError(f"could not start opencode ({exc})") from None
            finally:
                with contextlib.suppress(Exception):
                    log.close()
            self._wait_ready()
            self._stop.clear()
            self._events = threading.Thread(target=self._event_loop, daemon=True, name="opencode-events")
            self._events.start()

    def _wait_ready(self) -> None:
        deadline = self.clock() + self.start_timeout
        last = ""
        while self.clock() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise EngineError(f"opencode stopped while starting (exit code {self.proc.returncode}); see {self.log_path}")
            try:
                sid = self.client.post_json(f"{self.url}/session", {"title": "away-ready"}, self._auth(), 5)
                if isinstance(sid, dict) and sid.get("id"):
                    with contextlib.suppress(Exception):
                        self.client.send("DELETE", f"{self.url}/session/{sid['id']}", None, self._auth(), 5).read()
                    return
            except HttpError as exc:
                last = str(exc)
                if exc.status == 401:
                    raise EngineError("opencode refused Away's password") from None
            time.sleep(0.3)
        self.stop()
        raise EngineError(f"opencode did not start in time ({last or 'no answer'})")

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            p, self.proc = self.proc, None
            if p is None:
                return
            if p.poll() is None:
                if sys.platform == "win32":  # pragma: no cover - Windows: the whole process tree
                    with contextlib.suppress(Exception):
                        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True,
                                       creationflags=CREATE_NO_WINDOW, timeout=10)
                else:
                    p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    with contextlib.suppress(Exception):
                        p.wait(3)

    def idle_check(self) -> bool:
        """Stops the server after ``idle_stop_s`` without use (frees memory). True when it stopped."""
        if self.running() and not self._subs and self.clock() - self.last_used > self.idle_stop_s:
            self.stop()
            return True
        return False

    # ------------------------------------------------------------ events
    def _event_loop(self) -> None:
        while not self._stop.is_set():
            try:
                for ev in self.client.sse("GET", f"{self.url}/event", None, self._auth(), timeout=600):
                    if self._stop.is_set():
                        return
                    self._dispatch(ev)
            except Exception:
                if self._stop.is_set() or not self.running():
                    return
                time.sleep(0.5)

    def _dispatch(self, ev: dict[str, Any]) -> None:
        t = str(ev.get("type", ""))
        props = ev.get("properties") or {}
        if t.startswith("permission") or t.startswith("question"):
            threading.Thread(target=self.reject_pending, daemon=True).start()
        sid = props.get("sessionID") or (props.get("info") or {}).get("sessionID") or (props.get("part") or {}).get(
            "sessionID"
        )
        if sid and sid in self._subs:
            self._subs[sid].put(ev)

    def reject_pending(self) -> int:
        """Answers "no" to every tool the model asked to use (shell, files, web...)."""
        n = 0
        with contextlib.suppress(Exception):
            for pr in self.client.get_json(f"{self.url}/permission", self._auth(), 5) or []:
                with contextlib.suppress(Exception):
                    self.client.post_json(f"{self.url}/permission/{pr['id']}/reply", {"reply": "reject"}, self._auth(), 5)
                    n += 1
        with contextlib.suppress(Exception):
            for q in self.client.get_json(f"{self.url}/question", self._auth(), 5) or []:
                with contextlib.suppress(Exception):
                    self.client.post_json(f"{self.url}/question/{q['id']}/reject", {}, self._auth(), 5)
                    n += 1
        self.rejected += n
        return n

    # ------------------------------------------------------------ models
    def free_models_raw(self) -> dict[str, Any]:
        """``{model_id: models.dev-style info}`` the server offers for its ``opencode`` provider."""
        self.ensure()
        data = self.client.get_json(f"{self.url}/config/providers", self._auth(), 15) or {}
        for p in data.get("providers") or []:
            if p.get("id") == "opencode":
                self.models_cache = dict(p.get("models") or {})
                return self.models_cache
        return {}

    # ------------------------------------------------------------ chat
    def chat_stream(self, model: str, prompt: str, system: str = "", timeout: float = 60.0) -> Iterator[str]:
        self.ensure()
        auth = self._auth()
        ses = self.client.post_json(
            f"{self.url}/session",
            {"title": "away-live", "permission": [{"permission": "*", "pattern": "*", "action": "ask"}]},
            auth,
            10,
        )
        sid = str(ses["id"])
        q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._subs[sid] = q
        part_types: dict[str, str] = {}
        pending: dict[str, list[str]] = {}
        got_text = False
        try:
            body: dict[str, Any] = {
                "model": {"providerID": "opencode", "modelID": model},
                "parts": [{"type": "text", "text": prompt}],
            }
            if system:
                body["system"] = system
            self.client.send("POST", f"{self.url}/session/{sid}/prompt_async", body, auth, 15).read()
            deadline = self.clock() + timeout
            while True:
                left = deadline - self.clock()
                if left <= 0:
                    with contextlib.suppress(Exception):
                        self.client.send("POST", f"{self.url}/session/{sid}/abort", {}, auth, 5).read()
                    raise HttpError(0, "opencode timed out")
                try:
                    ev = q.get(timeout=min(left, 1.0))
                except queue.Empty:
                    if not self.running():
                        raise HttpError(503, "opencode stopped") from None
                    continue
                t = ev.get("type")
                props = ev.get("properties") or {}
                if t == "message.part.updated":
                    part = props.get("part") or {}
                    pid, ptype = str(part.get("id", "")), str(part.get("type", ""))
                    if pid and ptype:
                        part_types[pid] = ptype
                        if ptype == "text" and pid in pending:
                            for piece in pending.pop(pid):
                                got_text = True
                                yield piece
                        elif ptype != "text":
                            pending.pop(pid, None)
                elif t == "message.part.delta":
                    if props.get("field", "text") != "text":
                        continue
                    pid, piece = str(props.get("partID", "")), str(props.get("delta", ""))
                    kind = part_types.get(pid)
                    if kind == "text":
                        got_text = True
                        yield piece
                    elif kind is None:
                        pending.setdefault(pid, []).append(piece)
                elif t == "message.updated":
                    info = props.get("info") or {}
                    err = info.get("error")
                    if info.get("role") == "assistant" and err:
                        raise _error_from(err)
                elif t == "session.error":
                    err = props.get("error")
                    if err:
                        raise _error_from(err)
                elif t == "session.idle" or (t == "session.status" and (props.get("status") or {}).get("type") == "idle"):
                    break
            if not got_text:
                for piece in self._final_text(sid, auth):
                    yield piece
        finally:
            self._subs.pop(sid, None)
            self.last_used = self.clock()
            with contextlib.suppress(Exception):
                self.client.send("DELETE", f"{self.url}/session/{sid}", None, auth, 5).read()

    def _final_text(self, sid: str, auth: dict[str, str]) -> list[str]:
        msgs = self.client.get_json(f"{self.url}/session/{sid}/message", auth, 10) or []
        for m in reversed(msgs):
            info = m.get("info") or {}
            if info.get("role") != "assistant":
                continue
            if info.get("error"):
                raise _error_from(info["error"])
            return [str(p.get("text", "")) for p in m.get("parts") or [] if p.get("type") == "text"]
        return []


def _error_from(err: dict[str, Any]) -> HttpError:
    data = err.get("data") or {}
    msg = str(data.get("message") or err.get("message") or err.get("name") or "opencode error")
    status = int(data.get("statusCode") or 0)
    low = msg.lower()
    if not status:
        status = 429 if "rate limit" in low else 500
    if "free tier" in low:
        status = 403
    return HttpError(status, msg[:200])


def make_server(data_dir: Path | None = None, **kw: Any) -> OpenCodeServer:
    cmd, _where = resolve(data_dir)
    return OpenCodeServer(cmd, engine_root(data_dir) / "home", **kw)
