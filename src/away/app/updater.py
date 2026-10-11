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

import hashlib
import json
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
from pathlib import Path
from typing import Any

from away import __version__

REPO = "fernandogarzaaa/project-away"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
API_RELEASES = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
SUMS_ASSET = "SHA256SUMS.txt"
CHANNELS = ("stable", "beta")  # beta also considers GitHub pre-releases
CHECK_INTERVAL_S = 24 * 3600
TIMEOUT_S = 10.0
PIP_HINT = "pipx upgrade project-away   (or: pip install -U project-away)"

# Inno Setup silent install. /AUTOUPDATE=1 makes away.iss relaunch the tray app and skip the model
# download when the models are already there; /NORESTARTAPPLICATIONS avoids a second launch by the
# Restart Manager (the [Run] entry does the relaunch).
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


def windows_command(installer: Path, relaunch: bool = True) -> list[str]:
    cmd = [str(installer), *WINDOWS_SILENT_ARGS]
    if relaunch:
        cmd.append("/AUTOUPDATE=1")
    return cmd


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


def _detached_kwargs(platform: str) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if platform.startswith("win"):
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return kw


def install(
    path: Path,
    target: InstallTarget,
    pid: int | None = None,
    relaunch: bool = True,
    launcher: Callable[..., Any] = subprocess.Popen,
    runner: Callable[..., Any] = subprocess.run,
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
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
        launcher(windows_command(path, relaunch), **_detached_kwargs(plat))
        return InstallOutcome(True, "Installing the update; Project Away restarts when it is done.")
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


# -------------------------------------------------------------------------------------- tray manager
class UpdateManager:
    """Background checks for the tray app: on start, then every 24 h; notify or install when idle.

    ``tick(now)`` holds all the decisions so it can be tested without threads or network.
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
    ) -> None:
        self.cfg = updates_cfg
        self.notify, self.on_change, self.quit_app, self.is_idle = notify, on_change, quit_app, is_idle
        self.checker = checker or (lambda: check(channel=str(getattr(updates_cfg, "channel", "stable"))))
        self.installer = installer or self._download_and_install
        self.interval_s = interval_s
        self.result: CheckResult | None = None
        self.last_check: float | None = None
        self.notified: str | None = None
        self.installing = False
        self.failed_version: str | None = None  # an automatic install that failed is not retried
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
    def tick(self, now: float) -> None:
        if not getattr(self.cfg, "check", True):
            return
        if self.last_check is None or now - self.last_check >= self.interval_s:
            self.last_check = now
            fresh = self.checker()
            if fresh.error is None or self.available is None:  # a failed re-check keeps a known update
                self.result = fresh
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
            self.install_now()

    def install_now(self) -> InstallOutcome | None:
        """Download, verify and install the available update (tray menu click). Never raises."""
        r = self.available
        with self._lock:
            if r is None or self.installing:
                return None
            self.installing = True
        self.on_change()
        try:
            if r.target.kind == "pip" or r.asset is None:
                self.notify("Project Away update", r.summary())
                return None
            outcome = self.installer(r)
        except Exception as exc:  # UpdateError, or anything unexpected: never take the tray down
            self.last_error = str(exc) if isinstance(exc, UpdateError) else f"update failed: {exc}"
            self.failed_version = r.latest
            self.notify("Project Away update failed", self.last_error)
            return None
        finally:
            self.installing = False
            self.on_change()
        self.notify("Project Away update", outcome.message)
        if outcome.exit_app:
            self.quit_app()
        return outcome

    def _download_and_install(self, r: CheckResult) -> InstallOutcome:  # pragma: no cover - network + OS
        assert r.asset is not None
        path = download(r.asset)
        verify_download(path, r.asset, r.release, allow_unverified=bool(getattr(self.cfg, "allow_unverified", False)))
        return install(path, r.target)

    # -------------------------------------------------------------- thread
    def start(self, poll_s: float = 60.0) -> None:
        if not getattr(self.cfg, "check", True) or self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.tick(time.monotonic())
                except Exception as exc:  # pragma: no cover - defensive
                    self.last_error = str(exc)
                self._stop.wait(poll_s)

        self._thread = threading.Thread(target=loop, name="away-updater", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
