"""Auto-update from GitHub Releases (stdlib only: urllib, json, hashlib).

Privacy: the only network traffic is one unauthenticated ``GET`` to the public GitHub Releases API
(plus the asset download when the user installs). No identifiers, no telemetry.

Flow::

    check()                 -> CheckResult (never raises; offline / rate limit / bad JSON -> .error)
    download(asset, ...)    -> temp file, with a (done, total) progress callback
    verify_download(...)    -> SHA-256 from the API ``digest`` field ("sha256:..."), else a SHA256SUMS.txt
                               asset; refuses to install unverified files unless ``updates.allow_unverified``
    install(path, target)   -> per packaging: Windows Inno Setup silent install + relaunch, macOS .dmg
                               swap by a detached helper after the app exits, Linux AppImage atomic
                               replace, .deb via ``pkexec apt install`` (only after the user confirmed).
                               A pip/pipx install is never modified: the user gets the upgrade command.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import platform as _platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

from away import __version__

log = logging.getLogger("away.updater")

REPO = "fernandogarzaaa/project-away"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
API_RELEASES = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
SUMS_ASSET = "SHA256SUMS.txt"
CHANNELS = ("stable", "beta")  # beta also considers GitHub pre-releases
CHECK_INTERVAL_S = 24 * 3600
ERROR_RETRY_S = 30 * 60  # a failed check (offline at sign-in, rate limit) is retried after 30 min, not 24 h
EXIT_GRACE_S = 8.0  # after launching the installer: force the process to exit if a clean quit hangs
STATE_FILE = "update-state.json"  # written before installing; the next launch reports the outcome
EXIT_CODE_FILE = "update-exit-code.txt"  # the Windows helper writes the installer's exit code here
INSTALL_LOG = "update-install.log"  # Inno Setup /LOG (in <data dir>/logs)
HELPER_LOG = "update-helper.log"  # the Windows wait-then-install helper's own log
TIMEOUT_S = 10.0
PIP_HINT = "pipx upgrade project-away   (or: pip install -U project-away)"

# Inno Setup silent install. /AUTOUPDATE=1 makes away.iss relaunch the tray app and skip the model
# download when the models are already there; /NORESTARTAPPLICATIONS avoids a second launch by the
# Restart Manager (the [Run] entry does the relaunch). The updater adds /DIR= (the running install's
# folder) and /LOG= (<data dir>/logs/update-install.log). The installer is started by a small detached
# PowerShell helper only after this process has exited, so no file of the running app is locked.
WINDOWS_SILENT_ARGS = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/NORESTARTAPPLICATIONS")

Fetch = Callable[[str, float], bytes]
Opener = Callable[[str, float], Any]  # returns a context manager with .read(n) (and optionally .headers)


class UpdateError(Exception):
    """A user-facing update problem (offline, checksum mismatch, unsupported install...)."""


# ------------------------------------------------------------------------------------------ versions
_SEMVER = re.compile(r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?\s*$")


def parse_version(text: str) -> tuple[Any, ...] | None:
    """Sortable key for ``X.Y.Z[-pre][+build]`` (leading ``v`` allowed); None if not a version.

    A release sorts after its pre-releases (1.0.0 > 1.0.0-rc.1); numeric pre-release parts sort
    numerically and before alphanumeric ones, as in SemVer 2.0.
    """
    m = _SEMVER.match(str(text or ""))
    if not m:
        return None
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    pre = m.group(4)
    if pre is None:
        return (major, minor, patch, 1, ())
    parts = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return (major, minor, patch, 0, parts)


def is_newer(candidate: str, current: str) -> bool:
    """True when ``candidate`` is a strictly higher version than ``current`` (unparseable -> False)."""
    a, b = parse_version(candidate), parse_version(current)
    if a is None:
        return False
    if b is None:
        return True  # a dev/unknown local version: any real release is newer
    return a > b


def is_prerelease_version(text: str) -> bool:
    key = parse_version(text)
    return key is not None and key[3] == 0


# ------------------------------------------------------------------------------------------ releases
@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int = 0
    digest: str | None = None  # "sha256:<hex>" from the GitHub API (newer releases)


@dataclass(frozen=True)
class Release:
    tag: str
    version: str
    assets: tuple[Asset, ...] = ()
    html_url: str = RELEASES_PAGE
    draft: bool = False
    prerelease: bool = False

    def asset(self, name: str) -> Asset | None:
        return next((a for a in self.assets if a.name == name), None)


def release_from_json(d: Mapping[str, Any]) -> Release | None:
    tag = str(d.get("tag_name") or "")
    if parse_version(tag) is None:
        return None
    assets = []
    for a in d.get("assets") or []:
        if not isinstance(a, Mapping) or not a.get("name") or not a.get("browser_download_url"):
            continue
        digest = a.get("digest")
        assets.append(
            Asset(
                name=str(a["name"]),
                url=str(a["browser_download_url"]),
                size=int(a.get("size") or 0),
                digest=str(digest) if digest else None,
            )
        )
    return Release(
        tag=tag,
        version=tag.strip().lstrip("v"),
        assets=tuple(assets),
        html_url=str(d.get("html_url") or RELEASES_PAGE),
        draft=bool(d.get("draft")),
        prerelease=bool(d.get("prerelease")),
    )


def _default_fetch(url: str, timeout: float) -> bytes:  # pragma: no cover - network
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"project-away/{__version__} (update check)",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data: bytes = r.read()
        return data


def _friendly(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (403, 429):
            return "GitHub API rate limit reached; try again in an hour"
        if exc.code == 404:
            return "no published release found"
        return f"GitHub returned HTTP {exc.code}"
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError)):
        return "could not reach GitHub (offline?)"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "unexpected response from GitHub"
    return f"update check failed: {exc}"


def fetch_release(channel: str = "stable", fetch: Fetch | None = None, timeout: float = TIMEOUT_S) -> Release | None:
    """Newest published release for the channel (drafts never; pre-releases only on ``beta``).

    Raises :class:`UpdateError` with a friendly message on network/HTTP/JSON problems.
    """
    get = fetch or _default_fetch
    try:
        if channel == "beta":
            data = json.loads(get(API_RELEASES, timeout).decode("utf-8"))
            if not isinstance(data, list):
                raise ValueError("expected a list")
            items = [r for r in (release_from_json(x) for x in data if isinstance(x, Mapping)) if r is not None]
        else:
            data = json.loads(get(API_LATEST, timeout).decode("utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("expected an object")
            r = release_from_json(data)
            items = [r] if r is not None else []
    except Exception as exc:
        raise UpdateError(_friendly(exc)) from exc
    allow_pre = channel == "beta"
    ok = [r for r in items if not r.draft and (allow_pre or not (r.prerelease or is_prerelease_version(r.version)))]
    if not ok:
        return None
    return max(ok, key=lambda r: parse_version(r.version) or ())


# ------------------------------------------------------------------------------------- install kind
@dataclass(frozen=True)
class InstallTarget:
    kind: str  # windows | macos | appimage | deb | pip | unsupported
    path: Path | None = None  # what gets replaced/relaunched: setup dir exe, .app bundle, AppImage file, deb exe
    arch: str = "x86_64"


def _norm_arch(machine: str) -> str:
    """``x86_64`` | ``arm64`` | the raw machine name (lowercase)."""
    m = machine.lower()
    if m in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if m in ("arm64", "aarch64", "armv8", "armv8l"):
        return "arm64"
    return m or "x86_64"


def detect_install(
    platform: str | None = None,
    frozen: bool | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    machine: str | None = None,
) -> InstallTarget:
    """How this copy was installed, which decides the asset and the install method."""
    plat = platform or sys.platform
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    env = os.environ if env is None else env
    exe = Path(executable or sys.executable)
    arch = _norm_arch(machine if machine is not None else _platform.machine())
    if not is_frozen:
        return InstallTarget("pip", None, arch)
    if plat.startswith("win"):
        # the Inno install folder ({localappdata}\Programs\ProjectAway) holds ProjectAway.exe + unins000.exe;
        # a portable copy is updated in place the same way (/DIR= its folder)
        return InstallTarget("windows", exe, arch)
    if plat == "darwin":
        bundle = next((p for p in (exe, *exe.parents) if p.suffix == ".app"), None)
        return InstallTarget("macos", bundle, arch)
    linux_arch = "aarch64" if arch == "arm64" else arch  # build_appimage.sh / build_deb.sh use uname -m
    if env.get("APPIMAGE"):
        return InstallTarget("appimage", Path(env["APPIMAGE"]), linux_arch)
    if exe.as_posix().startswith("/opt/project-away/"):
        return InstallTarget("deb", exe, linux_arch)
    return InstallTarget("unsupported", exe, linux_arch)


def asset_suffix(target: InstallTarget) -> tuple[str, str] | None:
    """(prefix, suffix) of the release asset for this install, or None when there is no native asset."""
    if target.kind == "windows":
        return ("ProjectAway-", "-windows-x64-setup.exe")
    if target.kind == "macos":
        return ("ProjectAway-", f"-macos-{target.arch}.dmg")
    if target.kind == "appimage":
        return ("ProjectAway-", f"-linux-{target.arch}.AppImage")
    if target.kind == "deb":
        debarch = {"x86_64": "amd64", "aarch64": "arm64"}.get(target.arch, target.arch)
        return ("project-away_", f"_{debarch}.deb")
    return None


def select_asset(release: Release, target: InstallTarget) -> Asset | None:
    """The installer for this platform/packaging; prefers the exact versioned file name."""
    ps = asset_suffix(target)
    if ps is None:
        return None
    prefix, suffix = ps
    exact = release.asset(f"{prefix}{release.version}{suffix}")
    if exact is not None:
        return exact
    return next((a for a in release.assets if a.name.startswith(prefix) and a.name.endswith(suffix)), None)


# -------------------------------------------------------------------------------------------- check
@dataclass
class CheckResult:
    current: str
    latest: str | None = None
    update_available: bool = False
    release: Release | None = None
    asset: Asset | None = None
    target: InstallTarget = field(default_factory=lambda: InstallTarget("pip"))
    error: str | None = None

    @property
    def can_install(self) -> bool:
        return self.update_available and self.asset is not None and self.target.kind in INSTALLABLE

    def summary(self) -> str:
        if self.error:
            return f"Project Away {self.current}: {self.error}"
        if not self.update_available:
            latest = f" (latest release: {self.latest})" if self.latest else ""
            return f"Project Away {self.current} is up to date{latest}"
        msg = f"Project Away {self.latest} is available (you have {self.current})"
        if self.target.kind == "pip":
            return f"{msg}. Upgrade with: {PIP_HINT}"
        if self.asset is None:
            return f"{msg}. No installer for this system; download it from {self.release_url}"
        return msg

    @property
    def release_url(self) -> str:
        return self.release.html_url if self.release else RELEASES_PAGE


INSTALLABLE = frozenset({"windows", "macos", "appimage", "deb"})


def check(
    current: str = __version__,
    channel: str = "stable",
    target: InstallTarget | None = None,
    fetch: Fetch | None = None,
    timeout: float = TIMEOUT_S,
) -> CheckResult:
    """Compare the newest release with ``current``. Never raises."""
    tgt = target or detect_install()
    res = CheckResult(current=current, target=tgt)
    try:
        rel = fetch_release(channel, fetch, timeout)
    except UpdateError as exc:
        res.error = str(exc)
        return res
    except Exception as exc:  # belt and braces: an update check must never crash the app
        res.error = _friendly(exc)
        return res
    if rel is None:
        return res
    res.latest, res.release = rel.version, rel
    res.update_available = is_newer(rel.version, current)
    if res.update_available:
        res.asset = select_asset(rel, tgt)
    return res


# ---------------------------------------------------------------------------------- download/verify
def _default_opener(url: str, timeout: float) -> Any:  # pragma: no cover - network
    req = urllib.request.Request(url, headers={"User-Agent": f"project-away/{__version__} (updater)"})
    return urllib.request.urlopen(req, timeout=timeout)


def download(
    asset: Asset,
    dest_dir: Path | None = None,
    progress: Callable[[int, int], object] | None = None,
    opener: Opener | None = None,
    timeout: float = 60.0,
    chunk: int = 1 << 16,
) -> Path:
    """Stream the asset to ``dest_dir/<asset name>`` (via a ``.part`` file). Raises UpdateError."""
    out_dir = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="away-update-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(asset.name).name  # never trust a path in the asset name
    dest = out_dir / name
    part = out_dir / (name + ".part")
    open_url = opener or _default_opener
    done = 0
    try:
        with open_url(asset.url, timeout) as r, part.open("wb") as fh:
            headers = getattr(r, "headers", None)
            length = headers.get("Content-Length") if headers is not None else None
            total = int(length) if length and str(length).isdigit() else asset.size
            if progress:
                progress(0, total)
            while True:
                block = r.read(chunk)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if progress:
                    progress(done, total)
        part.replace(dest)
    except UpdateError:
        part.unlink(missing_ok=True)
        raise
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise UpdateError(f"download failed: {_friendly(exc)}") from exc
    return dest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    """``sha256sum`` output (``<hex>  name`` or ``<hex> *name``) -> {name: hex}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if m:
            out[Path(m.group(2).strip()).name] = m.group(1).lower()
    return out


def _api_digest(asset: Asset) -> str | None:
    if asset.digest and asset.digest.lower().startswith("sha256:"):
        hexd = asset.digest.split(":", 1)[1].strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", hexd):
            return hexd
    return None


def expected_sha256(asset: Asset, release: Release | None, fetch: Fetch | None = None) -> tuple[str | None, str]:
    """(hex digest, source) where source is ``api`` | ``SHA256SUMS.txt`` | ``none``."""
    api = _api_digest(asset)
    if api:
        return api, "api"
    sums = release.asset(SUMS_ASSET) if release else None
    if sums is not None:
        try:
            text = (fetch or _default_fetch)(sums.url, TIMEOUT_S).decode("utf-8", "replace")
        except Exception as exc:
            raise UpdateError(f"could not fetch {SUMS_ASSET}: {_friendly(exc)}") from exc
        hexd = parse_sums(text).get(asset.name)
        if hexd:
            return hexd, SUMS_ASSET
    return None, "none"


def verify_download(
    path: Path,
    asset: Asset,
    release: Release | None,
    fetch: Fetch | None = None,
    allow_unverified: bool = False,
) -> str:
    """Check the file's SHA-256; returns the source used. Deletes the file and raises on any failure."""
    try:
        expected, source = expected_sha256(asset, release, fetch)
    except UpdateError:
        Path(path).unlink(missing_ok=True)
        raise
    if expected is None:
        if allow_unverified:
            return "none"
        Path(path).unlink(missing_ok=True)
        raise UpdateError(
            f"{asset.name}: no SHA-256 published for this file; refusing to install "
            "(set updates.allow_unverified: true to override)"
        )
    actual = sha256_file(path)
    if actual != expected:
        Path(path).unlink(missing_ok=True)
        raise UpdateError(f"{asset.name}: checksum mismatch (expected {expected[:12]}…, got {actual[:12]}…)")
    return source


# ------------------------------------------------------------------------------------------ install
@dataclass(frozen=True)
class InstallOutcome:
    exit_app: bool  # the caller should quit now so the helper/installer can finish (and relaunch)
    message: str


def windows_command(
    installer: Path, relaunch: bool = True, install_dir: PurePath | None = None, log_path: PurePath | None = None
) -> list[str]:
    cmd = [str(installer), *WINDOWS_SILENT_ARGS]
    if relaunch:
        cmd.append("/AUTOUPDATE=1")
    if install_dir is not None:
        cmd.append(f"/DIR={install_dir}")
    if log_path is not None:
        cmd.append(f"/LOG={log_path}")
    return cmd


def win_arg(arg: str) -> str:
    """One Inno Setup argument for a command line: ``/DIR=C:\a b`` -> ``/DIR="C:\a b"``."""
    if not any(c in arg for c in ' \t"'):
        return arg
    if arg.startswith("/") and "=" in arg:
        k, v = arg.split("=", 1)
        return f'{k}="{v}"'
    return f'"{arg}"'


def _ps_str(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def windows_helper_script(
    pid: int, argv: Sequence[str], app_exe: Path | None, exit_file: Path, helper_log: Path, wait_s: int = 60
) -> str:
    """PowerShell: wait for this app (and any other copy of ProjectAway.exe from the install folder, e.g. the
    Settings window) to exit, stop leftovers after ``wait_s``, then run the installer, wait for it and write
    its exit code to ``exit_file`` so the next launch can report the result."""
    installer, args = str(argv[0]), " ".join(win_arg(str(a)) for a in argv[1:])
    exe = _ps_str(str(app_exe) if app_exe else "")
    name = _ps_str(PureWindowsPath(str(app_exe)).stem if app_exe else "ProjectAway")
    return "\n".join(
        [
            "$ErrorActionPreference = 'SilentlyContinue'",
            f"$log = {_ps_str(str(helper_log))}",
            "function Log($m) { Add-Content -Path $log -Value ((Get-Date -Format s) + ' ' + $m) }",
            f"Log 'update helper: waiting for Project Away (pid {pid}) to exit'",
            f"Wait-Process -Id {pid} -Timeout {wait_s}",
            f"$exe = {exe}",
            f"$deadline = (Get-Date).AddSeconds({max(5, wait_s // 2)})",
            "do {",
            f"  $left = @(Get-Process -Name {name} | Where-Object {{ $exe -eq '' -or $_.Path -eq $exe }})",
            "  if ($left.Count -eq 0) { break }",
            "  Start-Sleep -Milliseconds 500",
            "} while ((Get-Date) -lt $deadline)",
            "if ($left.Count -gt 0) { Log ('stopping ' + $left.Count + ' Project Away process(es) still running');"
            " $left | Stop-Process -Force; Start-Sleep -Seconds 1 }",
            f"Log ('starting installer: ' + {_ps_str(installer)} + ' ' + {_ps_str(args)})",
            f"$p = Start-Process -FilePath {_ps_str(installer)} -ArgumentList {_ps_str(args)} -PassThru",
            "if ($p) {",
            "  $h = $p.Handle",  # cache the handle, or ExitCode is empty after WaitForExit
            "  $p.WaitForExit()",
            "  Log ('installer exit code ' + $p.ExitCode)",
            f"  Set-Content -Path {_ps_str(str(exit_file))} -Value $p.ExitCode",
            "} else {",
            "  Log ('could not start the installer: ' + $Error[0])",
            f"  Set-Content -Path {_ps_str(str(exit_file))} -Value 'launch-failed'",
            "}",
        ]
    )


def windows_helper_command(script: str) -> list[str]:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")  # no quoting pitfalls
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-WindowStyle",
        "Hidden",
        "-EncodedCommand",
        encoded,
    ]


def windows_start_command(argv: Sequence[str]) -> str:
    """Last-resort launch (no PowerShell): ``cmd /c start "" "setup.exe" /VERYSILENT ...``."""
    return 'cmd.exe /c start "" ' + " ".join(win_arg(str(a)) for a in argv)


def deb_command(deb: Path) -> list[str]:
    # an absolute path (or ./file) tells apt this is a local file, not a package name
    return ["pkexec", "apt", "install", "-y", str(Path(deb).resolve())]


def wait_then_run_command(pid: int, argv: Sequence[str]) -> list[str]:
    """POSIX helper: wait for ``pid`` to exit, then exec ``argv`` (used to relaunch after quitting)."""
    script = 'while kill -0 "$0" 2>/dev/null; do sleep 0.5; done; exec "$@"'
    return ["/bin/sh", "-c", script, str(pid), *[str(a) for a in argv]]


MACOS_HELPER = r"""#!/bin/sh
# Project Away updater (macOS): wait for the app to quit, swap the .app from the .dmg, relaunch.
PID="$1"; DMG="$2"; DEST="$3"; RELAUNCH="$4"
while kill -0 "$PID" 2>/dev/null; do sleep 0.5; done
MNT="$(mktemp -d /tmp/away-update.XXXXXX)" || exit 1
hdiutil attach -nobrowse -readonly -noautoopen -mountpoint "$MNT" "$DMG" || exit 1
SRC="$(find "$MNT" -maxdepth 1 -name '*.app' -print | head -n 1)"
NEW="$DEST.new.$$"; OLD="$DEST.old.$$"
rm -rf "$NEW"
if [ -n "$SRC" ] && ditto "$SRC" "$NEW"; then
  xattr -dr com.apple.quarantine "$NEW" 2>/dev/null
  if mv "$DEST" "$OLD" 2>/dev/null || [ ! -e "$DEST" ]; then
    if mv "$NEW" "$DEST"; then rm -rf "$OLD"; else mv "$OLD" "$DEST"; fi
  fi
fi
rm -rf "$NEW"
hdiutil detach "$MNT" -quiet || hdiutil detach "$MNT" -force -quiet
rmdir "$MNT" 2>/dev/null
rm -f "$DMG"
[ "$RELAUNCH" = "1" ] && open "$DEST"
exit 0
"""


def macos_command(helper: Path, pid: int, dmg: Path, dest: Path, relaunch: bool = True) -> list[str]:
    return ["/bin/sh", str(helper), str(pid), str(dmg), str(dest), "1" if relaunch else "0"]


def macos_destination(target: InstallTarget) -> Path:
    """The .app to replace: the running bundle, else /Applications/Project Away.app."""
    bundle = target.path
    if bundle is not None and bundle.as_posix().startswith("/Volumes/"):
        raise UpdateError("Project Away is running from the disk image; drag it to Applications first")
    return bundle or Path("/Applications/Project Away.app")


def replace_appimage(new_file: Path, target: Path) -> Path:
    """Atomically replace ``target`` with ``new_file``: copy next to it, chmod +x, ``os.replace``.

    The running AppImage keeps working (its old inode stays mounted until it exits).
    """
    target = Path(target)
    tmp = target.with_name(f".{target.name}.update-{os.getpid()}")
    try:
        shutil.copyfile(new_file, tmp)
        mode = target.stat().st_mode if target.exists() else 0o755
        os.chmod(tmp, (mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH | stat.S_IRUSR) & 0o7777)
        os.replace(tmp, target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise UpdateError(f"cannot replace {target}: {exc}") from exc
    return target


DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000  # survive the app's exit even if it runs inside a kill-on-close job


def _detached_kwargs(platform: str, breakaway: bool = False) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if platform.startswith("win"):
        kw["creationflags"] = (
            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | (CREATE_BREAKAWAY_FROM_JOB if breakaway else 0)
        )
    else:
        kw["start_new_session"] = True
    return kw


def launch_windows_installer(
    installer: Path,
    target: InstallTarget,
    pid: int,
    relaunch: bool,
    launcher: Callable[..., Any],
    data_dir: Path,
    platform: str = "win32",
) -> str:
    """Start the detached helper that installs once this process is gone. Returns how it was started.

    Tries, in order: PowerShell helper with CREATE_BREAKAWAY_FROM_JOB, the helper without it (a job that
    forbids breakaway rejects the flag), then ``cmd /c start`` of the installer itself."""
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    install_dir = PureWindowsPath(str(target.path)).parent if target.path is not None else None
    argv = windows_command(installer, relaunch, install_dir, logs / INSTALL_LOG)
    exit_file = data_dir / EXIT_CODE_FILE
    exit_file.unlink(missing_ok=True)
    script = windows_helper_script(pid, argv, target.path, exit_file, logs / HELPER_LOG)
    helper = windows_helper_command(script)
    attempts: list[tuple[str, Any, dict[str, Any]]] = [
        ("powershell helper (breakaway)", helper, _detached_kwargs(platform, breakaway=True)),
        ("powershell helper", helper, _detached_kwargs(platform)),
        ("cmd start (breakaway)", windows_start_command(argv), _detached_kwargs(platform, breakaway=True)),
        ("cmd start", windows_start_command(argv), _detached_kwargs(platform)),
    ]
    errors = []
    for how, cmd, kw in attempts:
        try:
            launcher(cmd, **kw)
        except OSError as exc:
            log.warning("update: could not start the installer via %s: %s", how, exc)
            errors.append(f"{how}: {exc}")
            continue
        log.info("update: installer scheduled via %s: %s", how, " ".join(argv))
        return how
    raise UpdateError("could not start the installer (" + "; ".join(errors) + f"); run {installer} yourself")


def install(
    path: Path,
    target: InstallTarget,
    pid: int | None = None,
    relaunch: bool = True,
    launcher: Callable[..., Any] = subprocess.Popen,
    runner: Callable[..., Any] = subprocess.run,
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    data_dir: Path | None = None,
) -> InstallOutcome:
    """Install the verified file for ``target``. Raises UpdateError for anything it cannot do."""
    plat = platform or sys.platform
    pid = os.getpid() if pid is None else pid
    kind = target.kind
    if kind == "pip":
        raise UpdateError(f"installed with pip/pipx; upgrade with: {PIP_HINT}")
    if kind not in INSTALLABLE:
        raise UpdateError(f"this copy cannot update itself; download the new version from {RELEASES_PAGE}")
    if kind == "windows":
        launch_windows_installer(Path(path), target, pid, relaunch, launcher, data_dir or _data_dir(), plat)
        return InstallOutcome(True, "Installing the update; Project Away closes now and restarts when it is done.")
    if kind == "macos":
        dest = macos_destination(target)
        if not os.access(dest.parent, os.W_OK):
            raise UpdateError(f"no permission to replace {dest}; open {path} and drag the app to Applications")
        helper = Path(path).with_name("away-update-macos.sh")
        helper.write_text(MACOS_HELPER, encoding="utf-8")
        launcher(macos_command(helper, pid, Path(path), dest, relaunch), **_detached_kwargs(plat))
        if relaunch:
            return InstallOutcome(True, "Quitting to install the update; Project Away reopens in a moment.")
        return InstallOutcome(True, f"Update installs when this process exits; then start {dest.name} again.")
    if kind == "appimage":
        assert target.path is not None
        replace_appimage(Path(path), target.path)
        Path(path).unlink(missing_ok=True)
        if relaunch:
            launcher(wait_then_run_command(pid, [str(target.path)]), **_detached_kwargs(plat))
            return InstallOutcome(True, "Updated; restarting Project Away.")
        return InstallOutcome(False, f"Updated {target.path}; restart Project Away to use the new version.")
    # deb: the caller has asked the user first (pkexec also asks for the admin password)
    if which("pkexec") is None:
        raise UpdateError(f"pkexec not found; install the downloaded package with: sudo apt install {path}")
    proc = runner(deb_command(Path(path)), check=False)
    if getattr(proc, "returncode", 1) != 0:
        raise UpdateError(f"apt install failed or was cancelled; the package is at {path}")
    if relaunch and target.path is not None:
        launcher(wait_then_run_command(pid, [str(target.path), "tray"]), **_detached_kwargs(plat))
        return InstallOutcome(True, "Updated; restarting Project Away.")
    return InstallOutcome(False, "Updated; restart Project Away to use the new version.")


def _data_dir() -> Path:
    from away import paths

    return paths.data_dir()


# ------------------------------------------------------------------------------------ install state
INNO_EXIT_CODES = {
    "1": "the installer failed to start",
    "2": "the installer was cancelled",
    "3": "the installer hit a fatal error while preparing",
    "4": "the installer hit a fatal error while installing",
    "5": "the installer was stopped at 'Preparing to install' (the app could not be closed)",
    "6": "the installer was killed",
    "7": "the installer could not proceed (files in use?)",
    "8": "Windows needs a restart to finish the update",
    "launch-failed": "the installer could not be started",
}


def write_state(data_dir: Path, current: str, latest: str, asset: str, installer: Path | None = None) -> Path:
    """Record an install that is about to happen, for :func:`previous_outcome` on the next launch."""
    path = Path(data_dir) / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"from": current, "to": latest, "asset": asset, "installer": str(installer or ""), "t": time.time()}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def previous_outcome(data_dir: Path, current: str = __version__, clear: bool = True) -> tuple[bool, str, str] | None:
    """``(ok, version, message)`` for the install recorded by :func:`write_state`, or None when there is none.

    ``ok`` is True when this copy now runs the version that was being installed (or newer). The state is
    removed (``clear``) so the outcome is reported once."""
    path = Path(data_dir) / STATE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    finally:
        if clear:
            path.unlink(missing_ok=True)
    if not isinstance(data, dict) or not data.get("to"):
        return None
    to = str(data["to"])
    exit_file = Path(data_dir) / EXIT_CODE_FILE
    code = ""
    with contextlib.suppress(OSError):
        code = exit_file.read_text(encoding="utf-8", errors="replace").strip().lstrip("\ufeff")
    if clear:
        exit_file.unlink(missing_ok=True)
    pv, pc = parse_version(to), parse_version(current)
    if pv is not None and pc is not None and pc >= pv:
        return True, to, f"Project Away was updated to v{current}."
    why = INNO_EXIT_CODES.get(code, f"the installer exited with code {code}" if code else "the installer did not run")
    logs = Path(data_dir) / "logs"
    msg = f"The update to v{to} did not install (still v{current}): {why}. Details: {logs / INSTALL_LOG}"
    return False, to, msg


def exit_after_update(
    shutdown: Callable[[], object],
    grace_s: float = EXIT_GRACE_S,
    exit_fn: Callable[[int], object] = os._exit,
    timer: Callable[..., Any] = threading.Timer,
) -> None:
    """Quit so the installer can replace the files: a clean ``shutdown()``, and ``exit_fn(0)`` after ``grace_s``
    no matter what (a hung camera/voice thread or tray loop must never keep the old files locked)."""

    def force() -> None:
        log.info("update: forcing exit so the installer can run")
        with contextlib.suppress(Exception):
            logging.shutdown()
        exit_fn(0)

    t = timer(grace_s, force)
    t.daemon = True
    t.start()
    try:
        shutdown()
    except Exception as exc:
        log.warning("update: clean shutdown failed (%s); the forced exit follows", exc)


# -------------------------------------------------------------------------------------- tray manager
class UpdateManager:
    """Background checks for the tray app: on start, then every 24 h; notify or install when idle.

    ``tick(now)`` holds all the decisions so it can be tested without threads or network. Every step is
    logged (logger ``away.updater``, INFO; the tray app writes it to away.log) and every failure is shown
    as a notification.
    """

    def __init__(
        self,
        updates_cfg: Any,
        notify: Callable[[str, str], object] = lambda _t, _m: None,
        on_change: Callable[[], object] = lambda: None,
        quit_app: Callable[[], object] = lambda: None,
        is_idle: Callable[[], bool] = lambda: False,
        checker: Callable[[], CheckResult] | None = None,
        installer: Callable[[CheckResult], InstallOutcome] | None = None,
        interval_s: float = CHECK_INTERVAL_S,
        error_retry_s: float = ERROR_RETRY_S,
        data_dir: Path | None = None,
        current: str = __version__,
    ) -> None:
        self.cfg = updates_cfg
        self.notify, self.on_change, self.quit_app, self.is_idle = notify, on_change, quit_app, is_idle
        self.checker = checker or (lambda: check(channel=str(getattr(updates_cfg, "channel", "stable"))))
        self.installer = installer or self._download_and_install
        self.interval_s = interval_s
        self.error_retry_s = error_retry_s
        self._data_dir = data_dir
        self.current = current
        self.result: CheckResult | None = None
        self.last_check: float | None = None
        self.last_check_failed = False
        self.notified: str | None = None
        self.installing = False
        self.checking = False
        self.exiting = False  # set once the installer was launched and the app is quitting for it
        self.failed_version: str | None = None  # an automatic install that failed is not retried
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def data_dir(self) -> Path:
        return self._data_dir if self._data_dir is not None else _data_dir()

    # -------------------------------------------------------------- state for the menu
    @property
    def available(self) -> CheckResult | None:
        r = self.result
        return r if r is not None and r.update_available else None

    def menu_label(self) -> str | None:
        r = self.available
        if r is None:
            return None
        if self.installing:
            return f"Installing v{r.latest}..."
        return f"Update to v{r.latest}"

    # -------------------------------------------------------------- decisions
    def _run_check(self) -> CheckResult:
        log.info("update: checking GitHub releases (current v%s)", self.current)
        fresh = self.checker()
        self.last_check_failed = fresh.error is not None
        if fresh.error is None or self.available is None:  # a failed re-check keeps a known update
            self.result = fresh
        if fresh.error:
            log.info("update: check failed: %s", fresh.error)
        else:
            asset = fresh.asset.name if fresh.asset else None
            log.info(
                "update: latest v%s, update available: %s, install kind: %s, asset: %s",
                fresh.latest,
                fresh.update_available,
                fresh.target.kind,
                asset,
            )
        return fresh

    def tick(self, now: float) -> None:
        if not getattr(self.cfg, "check", True):
            return
        wait = self.error_retry_s if self.last_check_failed else self.interval_s
        if self.last_check is None or now - self.last_check >= wait:
            self.last_check = now
            fresh = self._run_check()
            if fresh.error is None and fresh.update_available and self.notified != fresh.latest:
                self.notified = fresh.latest
                if fresh.target.kind == "pip":
                    self.notify("Project Away update", f"v{fresh.latest} is available. Run: {PIP_HINT}")
                else:
                    self.notify("Project Away update", f"v{fresh.latest} is available. Use the tray menu to update.")
            self.on_change()
        r = self.available
        if (
            r is not None
            and r.can_install
            and getattr(self.cfg, "auto_install", False)
            and r.target.kind != "deb"  # needs an admin password: never silently
            and not self.installing
            and self.failed_version != r.latest
            and self.is_idle()
        ):
            log.info("update: idle, installing v%s automatically", r.latest)
            self.install_now()

    def check_now(self) -> CheckResult | None:
        """Tray "Check for updates": check right away and always say what was found. Never raises."""
        with self._lock:
            if self.checking or self.installing:
                return None
            self.checking = True
        try:
            fresh = self._run_check()
            self.last_check = time.monotonic()
            if fresh.error:
                self.notify("Project Away update check failed", fresh.error)
            elif fresh.update_available:
                self.notified = fresh.latest
                if fresh.target.kind == "pip":
                    self.notify("Project Away update", f"v{fresh.latest} is available. Run: {PIP_HINT}")
                elif fresh.asset is None:
                    self.notify("Project Away update", fresh.summary())
                else:
                    self.notify("Project Away update", f"v{fresh.latest} is available. Use the tray menu to update.")
            else:
                self.notify("Project Away update", fresh.summary())
            return fresh
        except Exception as exc:  # pragma: no cover - checker never raises; defensive
            log.exception("update: check crashed")
            self.notify("Project Away update check failed", str(exc))
            return None
        finally:
            self.checking = False
            self.on_change()

    def report_previous(self) -> tuple[bool, str, str] | None:
        """On start: tell the user how the last install went (from the state file it left)."""
        try:
            out = previous_outcome(self.data_dir, self.current)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("update: could not read the previous install state: %s", exc)
            return None
        if out is None:
            return None
        ok, version, msg = out
        log.info("update: previous install of v%s %s: %s", version, "succeeded" if ok else "FAILED", msg)
        if ok:
            self.notify("Project Away updated", msg)
        else:
            self.failed_version = version  # do not auto-install the same version in a loop
            self.last_error = msg
            self.notify("Project Away update failed", msg)
        return out

    def install_now(self) -> InstallOutcome | None:
        """Download, verify and install the available update (tray menu click). Never raises."""
        r = self.available
        with self._lock:
            if r is None or self.installing or self.exiting:
                return None
            self.installing = True
        self.on_change()
        log.info("update: install of v%s requested (install kind: %s)", r.latest, r.target.kind)
        try:
            if r.target.kind == "pip" or r.asset is None:
                log.info("update: nothing to install here: %s", r.summary())
                self.notify("Project Away update", r.summary())
                return None
            outcome = self.installer(r)
        except Exception as exc:  # UpdateError, or anything unexpected: never take the tray down
            self.last_error = str(exc) if isinstance(exc, UpdateError) else f"update failed: {exc}"
            self.failed_version = r.latest
            log.error("update: install of v%s failed: %s", r.latest, self.last_error)
            self.notify("Project Away update failed", self.last_error)
            return None
        finally:
            self.installing = False
            self.on_change()
        log.info("update: %s", outcome.message)
        self.notify("Project Away update", outcome.message)
        if outcome.exit_app:
            self.exiting = True
            log.info("update: quitting so the installer can replace the files")
            try:
                self.quit_app()
            except Exception as exc:
                log.error("update: quitting failed: %s", exc)
                self.notify("Project Away update", f"Quit Project Away to finish the update ({exc}).")
        return outcome

    def _download_and_install(self, r: CheckResult) -> InstallOutcome:
        assert r.asset is not None
        dl_dir = self.data_dir / "updates"
        if dl_dir.exists():  # old installers from earlier attempts
            for old in dl_dir.iterdir():
                if old.is_file() and old.name != r.asset.name:
                    with contextlib.suppress(OSError):
                        old.unlink()
        log.info("update: downloading %s (%d bytes) to %s", r.asset.name, r.asset.size, dl_dir)
        path = download(r.asset, dl_dir)
        log.info("update: downloaded %s", path)
        source = verify_download(
            path, r.asset, r.release, allow_unverified=bool(getattr(self.cfg, "allow_unverified", False))
        )
        log.info("update: SHA-256 verified (source: %s)", source)
        write_state(self.data_dir, self.current, str(r.latest), r.asset.name, path)
        try:
            outcome = install(path, r.target, data_dir=self.data_dir)
        except Exception:
            (self.data_dir / STATE_FILE).unlink(missing_ok=True)
            raise
        log.info("update: installer launched for v%s", r.latest)
        return outcome

    # -------------------------------------------------------------- thread
    def start(self, poll_s: float = 60.0, report_delay_s: float = 3.0) -> None:
        """Background thread: report the last install (after ``report_delay_s``, once the tray icon is up),
        then check right away and every ``interval_s``."""
        if self._thread is not None:
            return
        if not getattr(self.cfg, "check", True):
            log.info("update: automatic checks are off (updates.check: false)")

            def report_only() -> None:
                if not self._stop.wait(report_delay_s):
                    self.report_previous()

            threading.Thread(target=report_only, name="away-update-report", daemon=True).start()
            return

        def loop() -> None:
            if not self._stop.wait(report_delay_s):
                self.report_previous()
            while not self._stop.is_set():
                try:
                    self.tick(time.monotonic())
                except Exception as exc:  # pragma: no cover - defensive
                    self.last_error = str(exc)
                    log.exception("update: background check crashed")
                self._stop.wait(poll_s)

        self._thread = threading.Thread(target=loop, name="away-updater", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
