"""The backup link (v0.1.23): a second, non-Cloudflare way to reach this PC from the internet, free and with no
account, over the computer's own SSH client.

Cloudflare's edge sometimes refuses one agent's servers on every ``trycloudflare.com`` address ("403 Your request
was blocked."). The backup link uses a different company's relay instead:

* ``localhost.run`` (default): ``ssh -R 80:127.0.0.1:PORT nokey@localhost.run`` prints
  ``<random>.lhr.life tunneled with tls termination, https://<random>.lhr.life``.
* ``pinggy``: ``ssh -p 443 -R 0:127.0.0.1:PORT a.pinggy.io`` prints ``https://<random>.run.pinggy-free.link``
  (free addresses last about 60 minutes; the link restarts itself with a new address).

The local server stays bound to 127.0.0.1; SSH connects *out* and forwards to it, exactly like cloudflared. The
pairing URL and ``/c/<code>`` work the same on either link. Windows 10+ ships the OpenSSH client
(``C:\\Windows\\System32\\OpenSSH\\ssh.exe``); macOS and Linux have ``ssh``.

When the SSH process dies (network drop, the relay closed it, a free address timed out) :class:`Relay` starts it
again with a short back-off and reports the new address through ``on_url``; after ``max_restarts`` failures in
``restart_window_s`` it gives up and :meth:`Relay.alive` turns False.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROVIDERS = ("localhost.run", "pinggy")
LINK_TYPES = ("standard", "backup")
RESTART_DELAYS_S = (1.0, 3.0, 10.0, 30.0, 60.0)


class RelayError(RuntimeError):
    pass


class SshMissing(RelayError):
    pass


@dataclass(frozen=True)
class Provider:
    name: str
    host_args: tuple[str, ...]  # everything after the common options, with {port}
    url_res: tuple[re.Pattern[str], ...]  # tried in order on every output line
    batch: bool  # BatchMode=yes (never ask for a password); pinggy wants an empty password, so no


PROVIDER_INFO: dict[str, Provider] = {
    "localhost.run": Provider(
        "localhost.run",
        ("-R", "80:127.0.0.1:{port}", "nokey@localhost.run"),
        (
            re.compile(r"tunneled with tls termination,\s*(https://[A-Za-z0-9.-]+)"),
            re.compile(r"(https://[a-z0-9-]+\.lhr\.(?:life|rocks|pro))"),
        ),
        True,
    ),
    "pinggy": Provider(
        "pinggy",
        ("-p", "443", "-R", "0:127.0.0.1:{port}", "a.pinggy.io"),
        (re.compile(r"(https://[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:pinggy-free\.link|pinggy\.link|pinggy\.net|pinggy\.online))"),),
        False,
    ),
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_url(line: str, provider: str = "localhost.run") -> str | None:
    """The public https address in one line of the relay's output, or None."""
    text = ANSI_RE.sub("", line)
    for rx in PROVIDER_INFO[provider].url_res:
        m = rx.search(text)
        if m:
            url = m.group(1).rstrip(".,")
            if url.startswith(("https://admin.", "https://dashboard.")):
                continue
            return url
    return None


def ssh_candidates(plat: str | None = None, environ: dict[str, str] | None = None) -> list[str]:
    plat = plat or sys.platform
    env = os.environ if environ is None else environ
    if not plat.startswith("win"):
        return ["/usr/bin/ssh", "/usr/local/bin/ssh", "/opt/homebrew/bin/ssh"]
    root = env.get("SystemRoot") or env.get("WINDIR") or r"C:\Windows"
    # Sysnative: a 32-bit process sees the real System32 there (WOW64 redirects System32 itself)
    return [
        rf"{root}\System32\OpenSSH\ssh.exe",
        rf"{root}\Sysnative\OpenSSH\ssh.exe",
        r"C:\Program Files\OpenSSH\ssh.exe",
        r"C:\Program Files\Git\usr\bin\ssh.exe",
    ]


def find_ssh(
    configured: str = "",
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.isfile,
    plat: str | None = None,
) -> str:
    """The system's ssh client. Raises :class:`SshMissing` with a plain-language fix when there is none."""
    if configured:
        if exists(configured):
            return configured
        raise SshMissing(f"the SSH program set in agent_link.ssh_path was not found ({configured})")
    hit = which("ssh")
    if hit:
        return hit
    for c in ssh_candidates(plat):
        if exists(c):
            return c
    raise SshMissing(missing_help(plat))


def missing_help(plat: str | None = None) -> str:
    plat = plat or sys.platform
    if plat.startswith("win"):
        return (
            "The backup link needs Windows' built-in SSH program (OpenSSH Client), which is turned off on this PC. "
            "Turn it on in Settings > System > Optional features > Add a feature > OpenSSH Client, then try again."
        )
    if plat == "darwin":
        return "The backup link needs the ssh program, which macOS normally includes. Reinstall the command line tools."
    return "The backup link needs the ssh program. Install it (for example: sudo apt install openssh-client)."


def ssh_command(ssh: str, port: int, provider: str, known_hosts: Path | None = None) -> list[str]:
    info = PROVIDER_INFO[provider]
    opts = [
        "-T",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=15",
    ]
    if known_hosts is not None:
        opts += ["-o", f"UserKnownHostsFile={known_hosts}"]
    if info.batch:
        opts += ["-o", "BatchMode=yes"]
    return [ssh, *opts, *(a.format(port=int(port)) for a in info.host_args)]


class Relay:
    """One SSH relay process, kept running. Same surface as :class:`away.agentlink.tunnel.Tunnel` (``start``,
    ``alive``, ``stop``, ``url``, ``lines``) plus ``on_url`` (the address changed after a restart) and
    ``restarts``."""

    kind = "backup"

    def __init__(
        self,
        ssh: str,
        provider: str = "localhost.run",
        known_hosts: Path | None = None,
        popen: Callable[..., Any] = subprocess.Popen,
        on_url: Callable[[str | None, str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_restarts: int = 5,
        restart_window_s: float = 600.0,
        delays: tuple[float, ...] = RESTART_DELAYS_S,
    ) -> None:
        if provider not in PROVIDER_INFO:
            raise RelayError(f"unknown backup link service {provider!r} (use {' or '.join(PROVIDERS)})")
        self.ssh = ssh
        self.provider = provider
        self.known_hosts = known_hosts
        self.popen = popen
        self.on_url = on_url
        self.sleep = sleep
        self.clock = clock
        self.max_restarts = int(max_restarts)
        self.restart_window_s = float(restart_window_s)
        self.delays = delays
        self.proc: Any = None
        self.url: str | None = None
        self.port = 0
        self.lines: list[str] = []
        self.restarts = 0
        self.gave_up = ""
        self._restart_times: list[float] = []
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._watch: threading.Thread | None = None

    # ---------------------------------------------------------------- process
    def _spawn(self) -> Any:
        kw: dict[str, Any] = {
            "stdin": subprocess.PIPE,  # kept open: the relay closes the session when its input ends
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if sys.platform == "win32":  # pragma: no cover
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        if self.known_hosts is not None:
            with contextlib.suppress(OSError):
                self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
        return self.popen(ssh_command(self.ssh, self.port, self.provider, self.known_hosts), **kw)

    def _read(self, proc: Any, stream: Any) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                with self._lock:
                    self.lines.append(ANSI_RE.sub("", line.rstrip())[:300])
                    self.lines = self.lines[-60:]
                url = parse_url(line, self.provider)
                if url and proc is self.proc and not self._ready.is_set():
                    old, self.url = self.url, url
                    self._ready.set()
                    if old is not None and old != url and self.on_url is not None:
                        with contextlib.suppress(Exception):
                            self.on_url(old, url)
        except (OSError, ValueError):
            pass

    def _launch(self) -> Any:
        self._ready.clear()
        proc = self._spawn()
        self.proc = proc
        for s in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            threading.Thread(target=self._read, args=(proc, s), daemon=True, name="relay-log").start()
        return proc

    def start(self, port: int, timeout: float = 30.0) -> str:
        self.port = int(port)
        self._stopped.clear()
        self.gave_up = ""
        proc = self._launch()
        if not self._wait_url(proc, timeout):
            tail = " | ".join(x for x in self.lines[-3:] if x)
            self.stop()
            raise RelayError(f"the backup link did not start ({tail or 'no answer from ' + self.provider})")
        self._watch = threading.Thread(target=self._supervise, daemon=True, name="relay-watch")
        self._watch.start()
        assert self.url is not None
        return self.url

    def _wait_url(self, proc: Any, timeout: float) -> bool:
        end = self.clock() + timeout
        while self.clock() < end:
            if self._ready.wait(0.1):
                return bool(self.url)
            if proc.poll() is not None:  # died before it printed an address
                self._ready.wait(0.3)  # let the readers finish the last lines
                return bool(self._ready.is_set() and self.url)
        return False

    def _supervise(self) -> None:
        """Restart a relay that died, with back-off; give up after too many failures in a short time."""
        while not self._stopped.is_set():
            proc = self.proc
            if proc is None:
                return
            if proc.poll() is None:
                self._stopped.wait(0.5)
                continue
            if self._stopped.is_set():
                return
            now = self.clock()
            self._restart_times = [t for t in self._restart_times if now - t < self.restart_window_s] + [now]
            if len(self._restart_times) > self.max_restarts:
                self.gave_up = "the backup link kept dropping; press Blocked? Try the backup link to start it again"
                self.proc = None
                return
            delay = self.delays[min(len(self._restart_times) - 1, len(self.delays) - 1)]
            self.sleep(delay)
            if self._stopped.is_set():
                return
            self.restarts += 1
            try:
                nproc = self._launch()
            except Exception as exc:
                self.lines.append(f"restart failed: {exc}"[:300])
                continue
            self._wait_url(nproc, 30.0)

    def alive(self) -> bool:
        """True while the relay runs or is being restarted; False once stopped or given up."""
        if self._stopped.is_set() or self.gave_up:
            return False
        return self.proc is not None

    def stop(self) -> None:
        self._stopped.set()
        p, self.proc = self.proc, None
        self.url = None
        if p is None:
            return
        with contextlib.suppress(Exception):
            if p.stdin is not None:
                p.stdin.close()
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                p.kill()
