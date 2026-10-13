"""cloudflared quick tunnels (``https://<random>.trycloudflare.com``, free, no account) for agents that are not on
this PC. The local server stays bound to 127.0.0.1; cloudflared connects out to Cloudflare and forwards to it.

The binary is found in ``agent_link.cloudflared_path``, then ``<data>/bin``, then on PATH; otherwise (only when
the user picks a remote agent and ``agent_link.cloudflared_download`` is on) it is downloaded from Cloudflare's
GitHub release :data:`VERSION` and checked against the SHA-256 pinned below before it is ever run.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

VERSION = "2026.10.0"
BASE = f"https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/"
# (sys.platform prefix, machine) -> (asset, sha256 of the downloaded file, from the GitHub release asset digests)
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("win32", "amd64"): (
        "cloudflared-windows-amd64.exe",
        "86aee4017b26625cee8484c113558f48effa4cd47f7aa05fcf425604e5d2b23c",
    ),
    ("win32", "386"): (
        "cloudflared-windows-386.exe",
        "0630a8779e9823a1a3b091698b8e71874e0f7b205559219f52fdd301466b5546",
    ),
    ("linux", "amd64"): ("cloudflared-linux-amd64", "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db"),
    ("linux", "arm64"): ("cloudflared-linux-arm64", "e6422b9d4f72d3194bc5a38676f13667c06666523217b842a877d72a80b5ac08"),
    ("darwin", "amd64"): (
        "cloudflared-darwin-amd64.tgz",
        "903845b81828c8cb3c5d13d816a2de71c06a3da5785469df8eb0e1b736d92f9f",
    ),
    ("darwin", "arm64"): (
        "cloudflared-darwin-arm64.tgz",
        "a2f79ff7b9420aa537d74af239f376da170bbabeb529aec416002adac6a72e70",
    ),
}
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class TunnelError(RuntimeError):
    pass


def machine_key(plat: str | None = None, machine: str | None = None) -> tuple[str, str]:
    plat = plat or sys.platform
    m = (machine or platform.machine()).lower()
    arch = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "x86": "386",
        "i386": "386",
        "i686": "386",
    }.get(m, m)
    osname = "win32" if plat.startswith("win") else "darwin" if plat == "darwin" else "linux"
    return osname, arch


def exe_name(plat: str | None = None) -> str:
    return "cloudflared.exe" if (plat or sys.platform).startswith("win") else "cloudflared"


def find(data_dir: Path, configured: str = "") -> Path | None:
    if configured:
        p = Path(configured)
        return p if p.is_file() else None
    local = Path(data_dir) / "bin" / exe_name()
    if local.is_file():
        return local
    w = shutil.which("cloudflared")
    return Path(w) if w else None


def download(
    data_dir: Path,
    progress: Callable[[int, int], None] | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
    key: tuple[str, str] | None = None,
) -> Path:
    """Download the pinned cloudflared for this machine into ``<data>/bin`` and verify its SHA-256."""
    key = key or machine_key()
    if key not in ASSETS:
        raise TunnelError(f"no cloudflared download for {key[0]}/{key[1]}; install cloudflared yourself")
    asset, want = ASSETS[key]
    dest_dir = Path(data_dir) / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest_dir / (asset + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(BASE + asset, headers={"User-Agent": "project-away"})
    with opener(req, timeout=60) as resp, part.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if h.hexdigest() != want:
        part.unlink(missing_ok=True)
        raise TunnelError("the cloudflared download did not match its pinned checksum; nothing was installed")
    target = dest_dir / exe_name("win32" if key[0] == "win32" else key[0])
    if asset.endswith(".tgz"):
        with tarfile.open(part, "r:gz") as tf:
            member = next(
                (m for m in tf.getmembers() if m.name.rsplit("/", 1)[-1] == "cloudflared" and m.isfile()), None
            )
            if member is None:
                raise TunnelError("the cloudflared archive has no binary")
            src = tf.extractfile(member)
            assert src is not None
            target.write_bytes(src.read())
        part.unlink(missing_ok=True)
    else:
        os.replace(part, target)
    if key[0] != "win32":
        target.chmod(0o755)
    return target


class Tunnel:
    def __init__(self, exe: Path, popen: Callable[..., Any] = subprocess.Popen) -> None:
        self.exe = Path(exe)
        self.popen = popen
        self.proc: Any = None
        self.url: str | None = None
        self.lines: list[str] = []
        self._ready = threading.Event()

    def start(self, port: int, timeout: float = 45.0) -> str:
        kw: dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE, "text": True}
        if sys.platform == "win32":  # pragma: no cover
            kw["creationflags"] = 0x08000000
        self.proc = self.popen(
            [str(self.exe), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{int(port)}"], **kw
        )
        threading.Thread(target=self._read, daemon=True, name="cloudflared-log").start()
        if not self._ready.wait(timeout) or not self.url:
            tail = " | ".join(self.lines[-3:])
            self.stop()
            raise TunnelError(f"the secure link did not start ({tail or 'no output'})")
        return self.url

    def _read(self) -> None:
        stream = getattr(self.proc, "stderr", None)
        if stream is None:
            return
        for line in stream:
            self.lines.append(line.rstrip()[:300])
            self.lines = self.lines[-50:]
            m = URL_RE.search(line)
            if m and not self.url:
                self.url = m.group(0)
                self._ready.set()
        self._ready.set()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        p, self.proc = self.proc, None
        self.url = None
        if p is None:
            return
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                p.kill()


# ------------------------------------------------------------------ named tunnel (v0.1.24)
KEYRING_SERVICE = "ProjectAway"
KEYRING_NAMED = "cloudflare-tunnel-token"
READY_RE = re.compile(r"Registered tunnel connection|Connection [0-9a-f-]+ registered", re.IGNORECASE)


def named_token(keyring_mod: Any = None) -> str:
    """The named tunnel's token from the system keyring ("" when not set or no keyring)."""
    try:
        kr = keyring_mod
        if kr is None:
            import keyring as kr  # type: ignore[no-redef]
        return str(kr.get_password(KEYRING_SERVICE, KEYRING_NAMED) or "")
    except Exception:
        return ""


def set_named_token(token: str, keyring_mod: Any = None) -> None:
    """Keep (or with "" forget) the named tunnel's token in the system keyring, never in the config file."""
    kr = keyring_mod
    if kr is None:
        import keyring as kr  # type: ignore[no-redef]
    tok = str(token or "").strip()
    if not tok:
        with contextlib.suppress(Exception):
            kr.delete_password(KEYRING_SERVICE, KEYRING_NAMED)
        return
    if len(tok) < 20 or any(c.isspace() for c in tok):
        raise TunnelError("that does not look like a Cloudflare tunnel token (copy it from the tunnel's install command)")
    kr.set_password(KEYRING_SERVICE, KEYRING_NAMED, tok)


class NamedTunnel(Tunnel):
    """Your own Cloudflare tunnel (free account): ``cloudflared tunnel run --token <token>``. Its public hostname
    (set in the Cloudflare dashboard to ``http://127.0.0.1:<port>``) never changes, so agents keep the same address
    across restarts. The token is passed through the TUNNEL_TOKEN environment variable, never on the command line."""

    def __init__(self, exe: Path, token: str, hostname: str, popen: Callable[..., Any] = subprocess.Popen) -> None:
        super().__init__(exe, popen)
        self.token = token
        self.hostname = hostname.strip().lower().removeprefix("https://").rstrip("/")

    def start(self, port: int, timeout: float = 45.0) -> str:
        env = dict(os.environ, TUNNEL_TOKEN=self.token)
        kw: dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE, "text": True, "env": env}
        if sys.platform == "win32":  # pragma: no cover
            kw["creationflags"] = 0x08000000
        self.proc = self.popen([str(self.exe), "tunnel", "--no-autoupdate", "run"], **kw)
        threading.Thread(target=self._read, daemon=True, name="cloudflared-named-log").start()
        if not self._ready.wait(timeout) or not self.url:
            tail = " | ".join(self.lines[-3:])
            self.stop()
            raise TunnelError(f"your Cloudflare tunnel did not start ({tail or 'no output'})")
        return self.url

    def _read(self) -> None:
        stream = getattr(self.proc, "stderr", None)
        if stream is None:
            return
        for line in stream:
            self.lines.append(line.rstrip()[:300].replace(self.token, "***"))
            self.lines = self.lines[-50:]
            if READY_RE.search(line) and not self.url:
                self.url = f"https://{self.hostname}"
                self._ready.set()
        self._ready.set()


def wait_for(cond: Callable[[], bool], timeout: float, every: float = 0.2) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(every)
    return cond()
