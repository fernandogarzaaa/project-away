"""Safety-gated shell with staged access (proposal #26).

Two tiers, both behind the "use my computer" permission and an approved per-session computer grant:

* **Tier 1 (read-only allowlist):** a small set of named, fixed commands (device/camera/process lists, the
  ``away`` CLI, the app log tail). The argv is built here, never from agent input, and no shell is ever
  involved (``shell=False`` always), so shell metacharacters cannot mean anything. Runs without per-call
  approval inside an approved session.
* **Tier 2 (anything else):** any other argv asks the person first, every single time, through the same
  prompt path as risky clicks; the exact command is shown and only an explicit "Allow" runs it.

Hard rules, both tiers: never a shell (``shell=False``), never credential stores or password material
(refused outright, not even with approval), a 64 KB output cap like the REST body cap, secrets redacted
from output and the audit log, and every invocation (ran, denied or refused) appended to the audit log.

Threat model (see also the PR body): the agent is untrusted input. The tier-1 allowlist is code, reviewed
like code; exact-match classification means an agent cannot smuggle arguments into a tier-1 command. Tier-2
commands run with the user's interactive approval, shown verbatim; approval is per call, never sticky. The
remaining risk is a user approving a malicious command they did not understand: the prompt shows the full
argv, flags shell-looking characters, and names the agent's stated reason. ``shell=False`` plus fixed argv
removes shell-injection as a class; the credential-store refuse-list removes the most valuable targets.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OUTPUT_CAP = 65_536  # 64 KB, like the REST body cap
DEFAULT_TIMEOUT_S = 30.0
MAX_ARGV = 32

# Anything touching these is refused outright (ValueError -> HTTP 400), even with user approval: the tool
# never touches password fields or credential stores.
_CREDENTIAL_RE = re.compile(
    r"(?i)(password|passwd|pwd|secret|credential|private[_-]?key|id_rsa|\.pem\b"
    r"|keychain|lsass|sekurlsa|ntds\.dit"
    r"|[/\\.](ssh|aws|gnupg|pki)([/\\]|$))"
)

# Characters that do nothing special without a shell, but are worth flagging in the approval prompt.
_METACHAR_RE = re.compile(r"[;&|`$<>(){}\[\]!*?\n\r]")


class ShellRefused(ValueError):
    """The command touches something the shell tool never touches (credential stores, password material)."""


class ShellNotAvailable(RuntimeError):
    """The command is not available on this PC (missing program, unsupported platform)."""


_PS = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]


@dataclass(frozen=True)
class Tier1Command:
    name: str
    help: str  # plain words for the agent
    argv: dict[str, Any] = field(default_factory=dict)  # platform -> fixed argv (or () -> argv callable)
    serve: str = ""  # in-process source when there is no subprocess: "log_tail" | "linux_cameras"


def _away_version_argv() -> list[str]:
    from away.app.frozen import away_command

    return away_command(["--version"])


TIER1: tuple[Tier1Command, ...] = (
    Tier1Command(
        name="devices",
        help="Every hardware device the system currently sees (name, id, status): the first step when "
        "hardware misbehaves.",
        argv={
            "win32": [
                *_PS,
                "Get-PnpDevice -PresentOnly | "
                "Select-Object FriendlyName,InstanceId,Status,Class | ConvertTo-Json -Compress",
            ],
            "darwin": ["system_profiler", "SPUSBDataType", "-detailLevel", "mini"],
            "linux": ["lsusb"],
        },
    ),
    Tier1Command(
        name="cameras",
        help="Cameras the system sees right now: one query that usually settles 'the camera is not found'.",
        argv={
            "win32": [
                *_PS,
                "Get-PnpDevice -Class Camera -PresentOnly | "
                "Select-Object FriendlyName,InstanceId,Status | ConvertTo-Json -Compress",
            ],
            "darwin": ["system_profiler", "SPCameraDataType", "-detailLevel", "mini"],
        },
        serve="linux_cameras",
    ),
    Tier1Command(
        name="processes",
        help="The running process list (pid and name): for 'is X running' questions.",
        argv={
            "win32": ["tasklist", "/FO", "CSV"],
            "darwin": ["ps", "-eo", "pid,comm"],
            "linux": ["ps", "-eo", "pid,comm"],
        },
    ),
    Tier1Command(
        name="away_version",
        help="The installed away CLI's version (the same binary the diagnostics jobs use).",
        argv={"win32": _away_version_argv, "darwin": _away_version_argv, "linux": _away_version_argv},
    ),
    Tier1Command(
        name="logs",
        help="The tail of the app log, secrets hidden (read-only).",
        serve="log_tail",
    ),
)
BY_NAME = {c.name: c for c in TIER1}


def tier1_names() -> tuple[str, ...]:
    return tuple(c.name for c in TIER1)


def tier1_argv(name: str, platform: str = sys.platform) -> list[str] | None:
    """The fixed argv for a tier-1 command on this platform; None when it is served in-process or the
    platform has no entry. Raises KeyError for an unknown command."""
    spec = BY_NAME[name]
    entry = spec.argv.get(platform)
    if entry is None:
        return None
    return list(entry() if callable(entry) else entry)


def tier1_kind(name: str, platform: str = sys.platform) -> str:
    """'subprocess' | 'in_process' | 'unavailable'. Raises KeyError for an unknown command."""
    spec = BY_NAME[name]
    if tier1_argv(name, platform) is not None:
        return "subprocess"
    if spec.serve:
        return "in_process"
    return "unavailable"


def templates(platform: str = sys.platform) -> dict[tuple[str, ...], str]:
    """Exact argv -> command name for every tier-1 command available on this platform."""
    out: dict[tuple[str, ...], str] = {}
    for cmd in TIER1:
        argv = tier1_argv(cmd.name, platform)
        if argv is not None:
            out[tuple(argv)] = cmd.name
    return out


def describe(platform: str = sys.platform) -> list[dict[str, Any]]:
    return [{"name": c.name, "help": c.help, "available": tier1_kind(c.name, platform) != "unavailable"} for c in TIER1]


def has_metachars(argv: Sequence[str]) -> bool:
    return any(_METACHAR_RE.search(a) for a in argv)


def classify(argv: Sequence[str], platform: str = sys.platform) -> tuple[int, str | None]:
    """Which tier an argv runs in. Tier 1 only for an *exact* allowlist match (fixed argv, nothing from the
    agent); anything else is tier 2. Raises ShellRefused for credential-store / password touching argv."""
    args = [str(a) for a in argv]
    if _CREDENTIAL_RE.search("\x00".join(args)):
        raise ShellRefused(
            "refused: the shell tool never touches credential stores or password material "
            f"(matched in {args[0][:40]!r}...)"
        )
    name = templates(platform).get(tuple(args))
    if name is not None:
        return 1, name
    return 2, None


def _decode(data: Any) -> str:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data).decode("utf-8", "replace")
    return str(data or "")


def _bounded_popen_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
    """A ``subprocess.run``-compatible runner with bounded output retention: two drain threads keep at most
    :data:`OUTPUT_CAP` bytes per stream while the child runs, so a chatty approved command cannot grow
    service memory without bound. The child is never killed for being chatty: draining continues until it
    exits or ``timeout`` fires (which kills it, as ``subprocess.run`` does). The result carries
    ``discarded_output=True`` when any bytes were dropped."""
    create = kw.pop("creationflags", 0)
    proc = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=create)
    out, err = bytearray(), bytearray()
    discarded = {"out": False, "err": False}

    def _drain(stream: Any, buf: bytearray, key: str) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                room = OUTPUT_CAP - len(buf)
                if room > 0:
                    buf += chunk[:room]
                if len(chunk) > room:
                    discarded[key] = True
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, out, "out"), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err, "err"), daemon=True),
    ]
    for t in threads:
        t.start()
    timeout = float(kw.get("timeout") or DEFAULT_TIMEOUT_S)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
        for t in threads:
            t.join(timeout=5)
        raise subprocess.TimeoutExpired(args, timeout, output=bytes(out), stderr=bytes(err)) from None
    for t in threads:
        t.join(timeout=5)
    res = subprocess.CompletedProcess(args, rc, bytes(out), bytes(err))
    res.discarded_output = discarded["out"] or discarded["err"]  # type: ignore[attr-defined]
    return res


def run(
    argv: Sequence[str],
    timeout_s: float = DEFAULT_TIMEOUT_S,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run a fixed argv with no shell. Returns rc, stdout, stderr (each redacted later by the caller),
    timed_out, truncated and out_bytes. Raises ShellNotAvailable when the program cannot start.

    Output retention is bounded even before the 64 KB cap is applied: the default runner drains both
    streams through bounded buffers, so a chatty child cannot grow service memory without bound.
    """
    args = [str(a) for a in argv]
    if not args or any(not a for a in args):
        raise ValueError("shell: argv must be a non-empty list of non-empty strings")
    if len(args) > MAX_ARGV:
        raise ValueError(f"shell: argv too long (max {MAX_ARGV})")
    kw: dict[str, Any] = {"shell": False, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "timeout": timeout_s}
    if sys.platform == "win32":  # pragma: no cover - same as jobs.py: no console window flash
        kw["creationflags"] = 0x08000000
    try:
        proc = (runner or _bounded_popen_run)(args, **kw)
    except FileNotFoundError:
        raise ShellNotAvailable(f"{args[0]!r} is not installed on this PC") from None
    except subprocess.TimeoutExpired as e:
        stdout, stderr = _decode(e.stdout), _decode(e.stderr)
        stdout, stderr, truncated = _cap(stdout, stderr)
        return {
            "rc": None,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
            "truncated": True,  # a killed child may have emitted more than was drained
            "out_bytes": len(stdout.encode("utf-8")) + len(stderr.encode("utf-8")),
        }
    except OSError as e:
        raise ShellNotAvailable(f"could not start {args[0]!r}: {e.strerror or e}") from None
    stdout, stderr, truncated = _cap(_decode(proc.stdout), _decode(proc.stderr))
    truncated = truncated or bool(getattr(proc, "discarded_output", False))
    return {
        "rc": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
        "truncated": truncated,
        "out_bytes": len(stdout.encode("utf-8")) + len(stderr.encode("utf-8")),
    }


def _cap(stdout: str, stderr: str) -> tuple[str, str, bool]:
    """Cap combined output at OUTPUT_CAP: stderr keeps up to 8 KB (the diagnostics that matter on failure),
    stdout gets the rest."""
    if len(stdout) + len(stderr) <= OUTPUT_CAP:
        return stdout, stderr, False
    err_keep = min(len(stderr), 8192)
    return stdout[: OUTPUT_CAP - err_keep], stderr[:err_keep], True


def serve_in_process(name: str, log_tail: Callable[[], str] | None = None) -> str:
    """Stdout text for tier-1 commands with no subprocess. Raises KeyError / ShellNotAvailable."""
    spec = BY_NAME[name]
    if spec.serve == "log_tail":
        if log_tail is None:
            raise ShellNotAvailable("the app log tail is not wired up on this host")
        return log_tail()
    if spec.serve == "linux_cameras":
        dev = Path("/dev")
        try:
            cams = sorted(p.name for p in dev.iterdir() if p.name.startswith("video"))
        except OSError:
            cams = []
        return "\n".join(cams) if cams else "(no /dev/video* devices)"
    raise ShellNotAvailable(f"{name} has no in-process source")


def summarize(stdout: str, rc: Any) -> str:
    """One short result summary for the audit log (redacted by the caller)."""
    first = next((ln for ln in stdout.splitlines() if ln.strip()), "")
    if len(first) > 160:
        first = first[:157] + "..."
    return f"rc={rc} {first}".strip() or f"rc={rc}"


class ShellAudit:
    """Every shell invocation, appended as JSONL (``<data>/agentlink/shell-audit.jsonl``); a short in-memory
    ring plus the file tail serve ``action="log"``. Best-effort: auditing never breaks the call itself."""

    def __init__(self, path: Path, keep: int = 200) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mem: deque[dict[str, Any]] = deque(maxlen=keep)

    def append(self, entry: dict[str, Any]) -> None:
        rec = {"t": time.time(), **entry}
        with self._lock:
            self._mem.append(rec)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass

    def tail(self, limit: int = 30) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
            recs = [json.loads(ln) for ln in lines if ln.strip()]
            if recs:
                return recs[::-1]
        except (OSError, ValueError):
            pass
        with self._lock:
            return list(self._mem)[-limit:][::-1]
