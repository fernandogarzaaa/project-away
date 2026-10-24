"""Auto-update (away.app.updater): no network, every fetch/opener/launcher is injected."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from away import __version__
from away import config as C
from away.app import updater as U
from away.app.settings_gui import SettingsModel
from away.app.tray import TrayController
from away.cli import EXIT_UPDATE_AVAILABLE, build_parser, cmd_update

PAYLOAD = b"new installer bytes \x00\x01" * 100
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
V = "9.8.7"
NAMES = {
    "windows": f"ProjectAway-{V}-windows-x64-setup.exe",
    "macos": f"ProjectAway-{V}-macos-arm64.dmg",
    "appimage": f"ProjectAway-{V}-linux-x86_64.AppImage",
    "deb": f"project-away_{V}_amd64.deb",
}


def _asset_json(name: str, digest: str | None = "auto") -> dict:
    d = {"name": name, "browser_download_url": f"https://dl.example/{name}", "size": len(PAYLOAD)}
    if digest == "auto":
        d["digest"] = f"sha256:{PAYLOAD_SHA}"
    elif digest is not None:
        d["digest"] = digest
    return d


def _release_json(tag: str = f"v{V}", *, draft=False, prerelease=False, digest="auto", extra=()) -> dict:
    names = [*NAMES.values(), f"ProjectAway-{V}-linux-aarch64.AppImage", f"project-away_{V}_arm64.deb"]
    names += [f"project_away-{V}-py3-none-any.whl", *extra]
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/{U.REPO}/releases/tag/{tag}",
        "draft": draft,
        "prerelease": prerelease,
        "assets": [_asset_json(n, digest) for n in names],
    }


def _fetcher(payloads: dict[str, object], calls: list[str] | None = None):
    def fetch(url: str, timeout: float) -> bytes:
        if calls is not None:
            calls.append(url)
        assert timeout > 0
        value = payloads[url]
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, bytes) else json.dumps(value).encode()

    return fetch


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, length: str | None = None) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": length} if length is not None else {}


def _opener(data: bytes = PAYLOAD, urls: list[str] | None = None, length: str | None = None):
    def open_url(url: str, timeout: float):
        if urls is not None:
            urls.append(url)
        return _Resp(data, length)

    return open_url


def _release(**kw) -> U.Release:
    r = U.release_from_json(_release_json(**kw))
    assert r is not None
    return r


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(U.API_LATEST, code, "err", {}, None)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------- versions
@pytest.mark.parametrize(
    ("newer", "older"),
    [
        ("0.1.10", "0.1.9"),
        ("v0.1.3", "0.1.2"),
        ("0.2.0", "0.1.99"),
        ("1.0.0", "0.99.99"),
        ("1.0.0", "1.0.0-rc.1"),
        ("1.0.0-rc.2", "1.0.0-rc.1"),
        ("1.0.0-rc.10", "1.0.0-rc.2"),
        ("1.0.0-beta", "1.0.0-alpha"),
        ("1.0.0-alpha.beta", "1.0.0-alpha.1"),
        ("0.1.3", "0.1.2+local.5"),
    ],
)
def test_is_newer_orders_semver(newer: str, older: str) -> None:
    assert U.is_newer(newer, older) is True
    assert U.is_newer(older, newer) is False


def test_equal_and_unparseable_versions_are_not_newer() -> None:
    assert U.is_newer("0.1.2", "0.1.2") is False
    assert U.is_newer("v0.1.2", "0.1.2") is False
    assert U.is_newer("0.1.2+build", "0.1.2") is False
    assert U.is_newer("latest", "0.1.2") is False
    assert U.is_newer("", "0.1.2") is False
    assert U.is_newer("0.1.3", "not-a-version") is True  # unknown local build: any real release wins


def test_parse_version_details() -> None:
    assert U.parse_version("1.2") == U.parse_version("1.2.0") == (1, 2, 0, 1, ())
    assert U.parse_version("v10.20.30") == (10, 20, 30, 1, ())
    assert U.parse_version("1.2.3-rc.4") == (1, 2, 3, 0, ((1, 0, "rc"), (0, 4, "")))
    assert U.parse_version("1.2.3.4") is None
    assert U.parse_version("x1.2.3") is None
    assert U.is_prerelease_version("1.0.0-rc.1") is True
    assert U.is_prerelease_version("1.0.0") is False
    assert U.is_prerelease_version("junk") is False


# ------------------------------------------------------------------------------------ releases
def test_release_from_json_reads_assets_and_digest() -> None:
    r = _release()
    assert r.tag == f"v{V}" and r.version == V
    a = r.asset(NAMES["windows"])
    assert a == U.Asset(
        NAMES["windows"], f"https://dl.example/{NAMES['windows']}", len(PAYLOAD), f"sha256:{PAYLOAD_SHA}"
    )
    assert U.release_from_json({"tag_name": "nightly"}) is None
    bad = U.release_from_json({"tag_name": "v1.0.0", "assets": [{"name": "x"}, "junk", _asset_json("ok.txt", None)]})
    assert bad is not None and [a.name for a in bad.assets] == ["ok.txt"] and bad.assets[0].digest is None


def test_fetch_release_stable_uses_latest_endpoint() -> None:
    calls: list[str] = []
    rel = U.fetch_release("stable", _fetcher({U.API_LATEST: _release_json()}, calls))
    assert calls == [U.API_LATEST]
    assert rel is not None and rel.version == V


@pytest.mark.parametrize("kw", [{"draft": True}, {"prerelease": True}, {"tag": "v9.9.9-rc.1"}])
def test_stable_ignores_drafts_and_prereleases(kw: dict) -> None:
    assert U.fetch_release("stable", _fetcher({U.API_LATEST: _release_json(**kw)})) is None


def test_beta_channel_picks_highest_including_prereleases_but_never_drafts() -> None:
    items = [
        _release_json("v1.0.0"),
        _release_json("v1.1.0-rc.1", prerelease=True),
        _release_json("v2.0.0", draft=True),
        _release_json("v1.0.5"),
        {"tag_name": "garbage"},
    ]
    calls: list[str] = []
    rel = U.fetch_release("beta", _fetcher({U.API_RELEASES: items}, calls))
    assert calls == [U.API_RELEASES]
    assert rel is not None and rel.version == "1.1.0-rc.1"


@pytest.mark.parametrize(
    ("exc", "needle"),
    [
        (_http_error(403), "rate limit"),
        (_http_error(429), "rate limit"),
        (_http_error(404), "no published release"),
        (_http_error(500), "HTTP 500"),
        (urllib.error.URLError("dns"), "offline"),
        (TimeoutError("slow"), "offline"),
        (ConnectionResetError("reset"), "offline"),
    ],
)
def test_network_problems_become_friendly_errors(exc: BaseException, needle: str) -> None:
    with pytest.raises(U.UpdateError, match=needle):
        U.fetch_release("stable", _fetcher({U.API_LATEST: exc}))
    res = U.check("0.1.2", target=U.InstallTarget("windows"), fetch=_fetcher({U.API_LATEST: exc}))
    assert res.error is not None and needle in res.error
    assert res.update_available is False and res.asset is None and res.latest is None


@pytest.mark.parametrize("body", [b"<html>not json", b"[]", b'"string"'])
def test_bad_json_is_handled(body: bytes) -> None:
    res = U.check("0.1.2", target=U.InstallTarget("windows"), fetch=_fetcher({U.API_LATEST: body}))
    assert res.error == "unexpected response from GitHub"
    assert res.update_available is False


def test_check_never_raises_even_on_unexpected_errors() -> None:
    res = U.check("0.1.2", target=U.InstallTarget("pip"), fetch=_fetcher({U.API_LATEST: RuntimeError("boom")}))
    assert res.error is not None and "boom" in res.error
    assert "boom" in res.summary()


# ------------------------------------------------------------------------- install detection
def test_detect_install_kinds() -> None:
    d = U.detect_install
    assert d(platform="linux", frozen=False, env={"APPIMAGE": "/a"}, machine="x86_64").kind == "pip"
    w = d(platform="win32", frozen=True, env={}, executable=r"C:\P\ProjectAway.exe", machine="AMD64")
    assert w.kind == "windows" and w.arch == "x86_64"
    m = d(
        platform="darwin",
        frozen=True,
        env={},
        executable="/Applications/Project Away.app/Contents/MacOS/ProjectAway",
        machine="arm64",
    )
    assert m == U.InstallTarget("macos", Path("/Applications/Project Away.app"), "arm64")
    a = d(
        platform="linux",
        frozen=True,
        env={"APPIMAGE": "/home/u/PA.AppImage"},
        executable="/tmp/.mount_x/ProjectAway",
        machine="aarch64",
    )
    assert a == U.InstallTarget("appimage", Path("/home/u/PA.AppImage"), "aarch64")
    deb = d(platform="linux", frozen=True, env={}, executable="/opt/project-away/ProjectAway", machine="x86_64")
    assert deb == U.InstallTarget("deb", Path("/opt/project-away/ProjectAway"), "x86_64")
    raw = d(platform="linux", frozen=True, env={}, executable="/home/u/dist/ProjectAway/ProjectAway", machine="x86_64")
    assert raw.kind == "unsupported"


def test_detect_install_default_is_pip_in_tests() -> None:
    assert U.detect_install().kind == "pip"  # the test run is never a frozen app


# ---------------------------------------------------------------------------- asset selection
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (U.InstallTarget("windows", arch="x86_64"), NAMES["windows"]),
        (U.InstallTarget("windows", arch="arm64"), NAMES["windows"]),  # x64 build runs under emulation
        (U.InstallTarget("macos", arch="arm64"), NAMES["macos"]),
        (U.InstallTarget("appimage", arch="x86_64"), NAMES["appimage"]),
        (U.InstallTarget("appimage", arch="aarch64"), f"ProjectAway-{V}-linux-aarch64.AppImage"),
        (U.InstallTarget("deb", arch="x86_64"), NAMES["deb"]),
        (U.InstallTarget("deb", arch="aarch64"), f"project-away_{V}_arm64.deb"),
    ],
)
def test_select_asset_per_platform(target: U.InstallTarget, expected: str) -> None:
    a = U.select_asset(_release(), target)
    assert a is not None and a.name == expected


@pytest.mark.parametrize("kind", ["pip", "unsupported"])
def test_no_asset_for_pip_or_unknown_installs(kind: str) -> None:
    assert U.select_asset(_release(), U.InstallTarget(kind)) is None


def test_no_asset_for_intel_mac() -> None:
    assert U.select_asset(_release(), U.InstallTarget("macos", arch="x86_64")) is None


def test_select_asset_prefers_exact_versioned_name() -> None:
    rel = U.Release(
        "v1.2.3",
        "1.2.3",
        (
            U.Asset("ProjectAway-old-windows-x64-setup.exe", "u1"),
            U.Asset("ProjectAway-1.2.3-windows-x64-setup.exe", "u2"),
        ),
    )
    a = U.select_asset(rel, U.InstallTarget("windows"))
    assert a is not None and a.url == "u2"
    rel2 = U.Release("v1.2.3", "1.2.3", (U.Asset("ProjectAway-1.2.3a-windows-x64-setup.exe", "u3"),))
    a2 = U.select_asset(rel2, U.InstallTarget("windows"))
    assert a2 is not None and a2.url == "u3"


def test_check_reports_update_and_asset() -> None:
    res = U.check(
        "0.1.2",
        target=U.InstallTarget("appimage", Path("/x"), "x86_64"),
        fetch=_fetcher({U.API_LATEST: _release_json()}),
    )
    assert res.error is None and res.update_available is True and res.latest == V
    assert res.asset is not None and res.asset.name == NAMES["appimage"]
    assert res.can_install is True
    assert res.summary() == f"Project Away {V} is available (you have 0.1.2)"


def test_check_up_to_date() -> None:
    res = U.check(V, target=U.InstallTarget("windows"), fetch=_fetcher({U.API_LATEST: _release_json()}))
    assert res.update_available is False and res.asset is None and res.can_install is False
    assert res.summary() == f"Project Away {V} is up to date (latest release: {V})"


def test_check_pip_install_gets_upgrade_hint_not_installer() -> None:
    res = U.check("0.1.2", target=U.InstallTarget("pip"), fetch=_fetcher({U.API_LATEST: _release_json()}))
    assert res.update_available is True and res.asset is None and res.can_install is False
    assert "pipx upgrade project-away" in res.summary()


def test_check_without_any_release() -> None:
    res = U.check("0.1.2", target=U.InstallTarget("windows"), fetch=_fetcher({U.API_LATEST: _release_json(draft=True)}))
    assert res.error is None and res.latest is None and res.update_available is False


# ------------------------------------------------------------------------------------ download
def test_download_streams_with_progress(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []
    urls: list[str] = []
    asset = U.Asset(NAMES["windows"], "https://dl.example/x", len(PAYLOAD))
    p = U.download(asset, tmp_path, progress=lambda d, t: seen.append((d, t)), opener=_opener(urls=urls), chunk=256)
    assert p == tmp_path / NAMES["windows"] and p.read_bytes() == PAYLOAD
    assert urls == ["https://dl.example/x"]
    assert seen[0] == (0, len(PAYLOAD)) and seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen) and len(seen) == 2 + len(PAYLOAD) // 256
    assert not list(tmp_path.glob("*.part"))


def test_download_uses_content_length_when_present(tmp_path: Path) -> None:
    seen: list[int] = []
    U.download(U.Asset("a.bin", "u", 0), tmp_path, progress=lambda d, t: seen.append(t), opener=_opener(length="2400"))
    assert set(seen) == {2400}


def test_download_strips_path_from_asset_name(tmp_path: Path) -> None:
    p = U.download(U.Asset("../../evil.exe", "u"), tmp_path / "d", opener=_opener())
    assert p == tmp_path / "d" / "evil.exe"


def test_download_failure_cleans_up(tmp_path: Path) -> None:
    class Broken(_Resp):
        def read(self, n: int = -1) -> bytes:  # type: ignore[override]
            raise ConnectionResetError("dropped")

    with pytest.raises(U.UpdateError, match="download failed"):
        U.download(U.Asset("a.bin", "u"), tmp_path, opener=lambda u, t: Broken(b""))
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------------------------- verify
def _file(tmp_path: Path, data: bytes = PAYLOAD, name: str = "dl.bin") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_verify_with_api_digest_passes(tmp_path: Path) -> None:
    p = _file(tmp_path)
    asset = U.Asset("dl.bin", "u", digest=f"sha256:{PAYLOAD_SHA.upper()}")
    calls: list[str] = []
    assert U.verify_download(p, asset, _release(), _fetcher({}, calls)) == "api"
    assert calls == [] and p.exists()


def test_verify_with_api_digest_mismatch_refuses_and_deletes(tmp_path: Path) -> None:
    p = _file(tmp_path, b"tampered")
    asset = U.Asset("dl.bin", "u", digest=f"sha256:{PAYLOAD_SHA}")
    with pytest.raises(U.UpdateError, match="checksum mismatch"):
        U.verify_download(p, asset, _release(), allow_unverified=True)  # a mismatch is never overridable
    assert not p.exists()


def _sums_release(sums_text: str) -> tuple[U.Release, object]:
    rel = U.Release("v1.0.0", "1.0.0", (U.Asset("dl.bin", "u"), U.Asset(U.SUMS_ASSET, "https://dl.example/sums")))
    return rel, _fetcher({"https://dl.example/sums": sums_text.encode()})


def test_sha256sums_fallback_passes(tmp_path: Path) -> None:
    p = _file(tmp_path)
    rel, fetch = _sums_release(f"{'0' * 64}  other.exe\n{PAYLOAD_SHA} *dl.bin\n")
    assert U.verify_download(p, rel.assets[0], rel, fetch) == U.SUMS_ASSET  # type: ignore[arg-type]
    assert p.exists()


def test_sha256sums_fallback_mismatch(tmp_path: Path) -> None:
    p = _file(tmp_path)
    rel, fetch = _sums_release(f"{'a' * 64}  dl.bin\n")
    with pytest.raises(U.UpdateError, match="checksum mismatch"):
        U.verify_download(p, rel.assets[0], rel, fetch)  # type: ignore[arg-type]
    assert not p.exists()


def test_sha256sums_without_entry_refuses_by_default(tmp_path: Path) -> None:
    p = _file(tmp_path)
    rel, fetch = _sums_release(f"{PAYLOAD_SHA}  something-else.bin\n")
    with pytest.raises(U.UpdateError, match="refusing to install"):
        U.verify_download(p, rel.assets[0], rel, fetch)  # type: ignore[arg-type]
    assert not p.exists()


def test_sums_fetch_failure_refuses(tmp_path: Path) -> None:
    p = _file(tmp_path)
    rel, _ = _sums_release("")
    with pytest.raises(U.UpdateError, match=re.escape("could not fetch SHA256SUMS.txt")):
        U.verify_download(p, rel.assets[0], rel, _fetcher({"https://dl.example/sums": _http_error(403)}))
    assert not p.exists()


def test_no_digest_anywhere_refuses_unless_allowed(tmp_path: Path) -> None:
    p = _file(tmp_path)
    asset = U.Asset("dl.bin", "u", digest="md5:abc")  # unknown algorithm counts as no digest
    rel = U.Release("v1.0.0", "1.0.0", (asset,))
    assert U.verify_download(p, asset, rel, allow_unverified=True) == "none"
    assert p.exists()
    with pytest.raises(U.UpdateError, match=re.escape("updates.allow_unverified: true")):
        U.verify_download(p, asset, rel)
    assert not p.exists()


def test_parse_sums_formats() -> None:
    h1, h2 = "A" * 64, "b" * 64
    got = U.parse_sums(f"{h1}  ProjectAway.dmg\n# comment\n{h2} *dir/x.deb\nshort  y\n")
    assert got == {"ProjectAway.dmg": "a" * 64, "x.deb": h2}


# ------------------------------------------------------------------------------------- install
class _Recorder:
    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.returncode = returncode

    def __call__(self, cmd, **kw):
        self.calls.append((list(cmd), kw))
        return SimpleNamespace(returncode=self.returncode)


def test_windows_command_line() -> None:
    exe = Path("C:/Temp/ProjectAway-1.0.0-windows-x64-setup.exe")
    assert U.windows_command(exe) == [
        str(exe),
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/CLOSEAPPLICATIONS",
        "/NORESTARTAPPLICATIONS",
        "/AUTOUPDATE=1",
    ]
    assert U.windows_command(exe, relaunch=False)[-1] == "/NORESTARTAPPLICATIONS"


def test_windows_install_launches_detached_installer(tmp_path: Path) -> None:
    setup = _file(tmp_path, name=NAMES["windows"])
    launch, run = _Recorder(), _Recorder()
    out = U.install(setup, U.InstallTarget("windows"), pid=42, launcher=launch, runner=run, platform="win32")
    assert out.exit_app is True
    assert len(launch.calls) == 1 and run.calls == []
    chars, kw = launch.calls[0]
    cmd = "".join(chars)  # _Recorder stores list(cmd); the helper command line is one string
    # v0.1.10: a hidden cmd.exe runs away-update.cmd, which waits for this process to exit, then installs
    assert cmd.endswith(f'/d /s /c ""{setup.parent / U.HELPER_SCRIPT}""')
    script = (setup.parent / U.HELPER_SCRIPT).read_text(encoding="utf-8")
    assert '"PID eq 42"' in script and str(setup) in script and "/AUTOUPDATE=1" in script
    assert kw["creationflags"] == 0x08000000 | 0x00000200 | 0x01000000 and "start_new_session" not in kw
    assert kw["stdin"] is U.subprocess.DEVNULL


@pytest.mark.parametrize("kind", ["pip", "unsupported"])
def test_no_install_when_not_frozen_or_unknown(tmp_path: Path, kind: str) -> None:
    launch, run = _Recorder(), _Recorder()
    with pytest.raises(U.UpdateError, match="pipx upgrade" if kind == "pip" else "cannot update itself"):
        U.install(_file(tmp_path), U.InstallTarget(kind), launcher=launch, runner=run)
    assert launch.calls == [] and run.calls == []


@pytest.mark.skipif(sys.platform == "win32", reason="AppImages are Linux-only; Windows has no POSIX exec bits")
def test_appimage_atomic_replace(tmp_path: Path) -> None:
    target = tmp_path / "apps" / "ProjectAway-0.1.2-linux-x86_64.AppImage"
    target.parent.mkdir()
    target.write_bytes(b"old")
    os.chmod(target, 0o700)
    inode_before = target.stat().st_ino
    new = _file(tmp_path, name="dl.AppImage")
    assert U.replace_appimage(new, target) == target
    assert target.read_bytes() == PAYLOAD
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode & stat.S_IXUSR and mode & stat.S_IXGRP and mode & stat.S_IXOTH and mode & stat.S_IRUSR
    assert mode & stat.S_IWUSR  # kept the original owner bits
    assert target.stat().st_ino != inode_before  # replaced (new inode), not rewritten in place
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]  # no temp file left behind
    assert new.exists()  # replace_appimage copies; install() removes the download


def test_appimage_replace_failure_leaves_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "PA.AppImage"
    target.write_bytes(b"old")

    def boom(src, dst):
        raise PermissionError("read-only")

    monkeypatch.setattr(U.os, "replace", boom)
    with pytest.raises(U.UpdateError, match="cannot replace"):
        U.replace_appimage(_file(tmp_path, name="n"), target)
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["PA.AppImage", "n"]


def test_appimage_install_replaces_and_relaunches_after_exit(tmp_path: Path) -> None:
    target = tmp_path / "PA.AppImage"
    target.write_bytes(b"old")
    dl = _file(tmp_path, name="dl.AppImage")
    launch = _Recorder()
    out = U.install(dl, U.InstallTarget("appimage", target), pid=4242, launcher=launch, platform="linux")
    assert out.exit_app is True and target.read_bytes() == PAYLOAD and not dl.exists()
    cmd, kw = launch.calls[0]
    assert cmd == U.wait_then_run_command(4242, [str(target)])
    assert kw["start_new_session"] is True and "creationflags" not in kw


def test_appimage_install_without_relaunch(tmp_path: Path) -> None:
    target = tmp_path / "PA.AppImage"
    target.write_bytes(b"old")
    launch = _Recorder()
    out = U.install(_file(tmp_path), U.InstallTarget("appimage", target), relaunch=False, launcher=launch)
    assert out.exit_app is False and launch.calls == [] and target.read_bytes() == PAYLOAD


def test_wait_then_run_command() -> None:
    cmd = U.wait_then_run_command(77, ["/opt/a b/ProjectAway", "tray"])
    assert cmd[:2] == ["/bin/sh", "-c"]
    assert 'kill -0 "$0"' in cmd[2] and 'exec "$@"' in cmd[2]
    assert cmd[3:] == ["77", "/opt/a b/ProjectAway", "tray"]


def test_deb_install_runs_pkexec_apt_then_relaunches(tmp_path: Path) -> None:
    deb = _file(tmp_path, name=NAMES["deb"])
    launch, run = _Recorder(), _Recorder(0)
    exe = Path("/opt/project-away/ProjectAway")
    out = U.install(
        deb,
        U.InstallTarget("deb", exe),
        pid=9,
        launcher=launch,
        runner=run,
        platform="linux",
        which=lambda n: "/usr/bin/pkexec",
    )
    assert run.calls[0][0] == ["pkexec", "apt", "install", "-y", str(deb.resolve())]
    assert run.calls[0][1] == {"check": False}
    assert launch.calls[0][0] == U.wait_then_run_command(9, [str(exe), "tray"])
    assert out.exit_app is True


def test_deb_install_failure_keeps_package(tmp_path: Path) -> None:
    deb = _file(tmp_path, name=NAMES["deb"])
    launch, run = _Recorder(), _Recorder(126)
    with pytest.raises(U.UpdateError, match="cancelled"):
        U.install(
            deb,
            U.InstallTarget("deb", Path("/opt/project-away/ProjectAway")),
            launcher=launch,
            runner=run,
            which=lambda n: "/x",
        )
    assert launch.calls == [] and deb.exists()


def test_deb_without_pkexec_shows_path(tmp_path: Path) -> None:
    deb = _file(tmp_path, name=NAMES["deb"])
    run = _Recorder()
    with pytest.raises(U.UpdateError, match=re.escape(f"sudo apt install {deb}")):
        U.install(deb, U.InstallTarget("deb", Path("/opt/project-away/ProjectAway")), runner=run, which=lambda n: None)
    assert run.calls == []


def test_macos_install_writes_helper_and_waits_for_exit(tmp_path: Path) -> None:
    dmg = _file(tmp_path, name=NAMES["macos"])
    bundle = tmp_path / "Applications" / "Project Away.app"
    bundle.mkdir(parents=True)
    launch = _Recorder()
    out = U.install(dmg, U.InstallTarget("macos", bundle, "arm64"), pid=31, launcher=launch, platform="darwin")
    helper = tmp_path / "away-update-macos.sh"
    assert out.exit_app is True and helper.read_text() == U.MACOS_HELPER
    cmd, kw = launch.calls[0]
    assert cmd == ["/bin/sh", str(helper), "31", str(dmg), str(bundle), "1"]
    assert kw["start_new_session"] is True


def test_macos_helper_script_steps_in_order() -> None:
    s = U.MACOS_HELPER
    steps = [
        'kill -0 "$PID"',
        "hdiutil attach",
        'ditto "$SRC" "$NEW"',
        "xattr -dr com.apple.quarantine",
        'mv "$NEW" "$DEST"',
        "hdiutil detach",
        'open "$DEST"',
    ]
    idx = [s.index(x) for x in steps]
    assert idx == sorted(idx)
    assert '[ "$RELAUNCH" = "1" ] && open' in s


def test_macos_refuses_running_from_dmg_and_defaults_to_applications(tmp_path: Path) -> None:
    with pytest.raises(U.UpdateError, match="drag it to Applications"):
        U.macos_destination(U.InstallTarget("macos", Path("/Volumes/Project Away/Project Away.app")))
    assert U.macos_destination(U.InstallTarget("macos", None)) == Path("/Applications/Project Away.app")


def test_macos_unwritable_destination_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(U.os, "access", lambda p, m: False)
    launch = _Recorder()
    with pytest.raises(U.UpdateError, match="no permission"):
        U.install(_file(tmp_path), U.InstallTarget("macos", tmp_path / "X.app"), launcher=launch, platform="darwin")
    assert launch.calls == []


# ------------------------------------------------------------------------------- tray manager
def _result(kind: str = "windows", version: str = V, error: str | None = None) -> U.CheckResult:
    rel = _release()
    tgt = U.InstallTarget(kind, Path("/opt/project-away/ProjectAway") if kind == "deb" else None)
    if error:
        return U.CheckResult("0.1.2", target=tgt, error=error)
    avail = U.is_newer(version, "0.1.2")
    return U.CheckResult("0.1.2", version, avail, rel, U.select_asset(rel, tgt) if avail else None, tgt)


class _Mgr:
    def __init__(self, cfg=None, result=None, idle=True, outcome=None, fail=False) -> None:
        self.checks = 0
        self.notes: list[tuple[str, str]] = []
        self.installs: list[U.CheckResult] = []
        self.quits = 0
        self.result = result or _result()

        def checker():
            self.checks += 1
            return self.result

        def installer(r):
            self.installs.append(r)
            if fail:
                raise U.UpdateError("checksum mismatch")
            return outcome or U.InstallOutcome(True, "Installing")

        self.m = U.UpdateManager(
            cfg or C.UpdatesConfig(),
            notify=lambda t, msg: self.notes.append((t, msg)),
            quit_app=lambda: setattr(self, "quits", self.quits + 1),
            is_idle=lambda: idle,
            checker=checker,
            installer=installer,
            interval_s=100.0,
        )


def test_manager_checks_on_start_then_every_interval_and_notifies_once() -> None:
    t = _Mgr()
    t.m.tick(1000.0)
    t.m.tick(1050.0)
    assert t.checks == 1
    t.m.tick(1100.0)
    assert t.checks == 2
    assert t.notes == [("Project Away update", f"v{V} is available. Use the tray menu to update.")]
    assert t.m.menu_label() == f"Update to v{V}"
    assert t.installs == []  # auto_install is off by default: notify only


def test_manager_failed_recheck_keeps_known_update() -> None:
    t = _Mgr()
    t.m.tick(0.0)
    t.result = _result(error="could not reach GitHub (offline?)")
    t.m.tick(200.0)
    assert t.checks == 2 and t.m.menu_label() == f"Update to v{V}"
    assert len(t.notes) == 1


def test_manager_disabled_never_checks() -> None:
    t = _Mgr(cfg=C.UpdatesConfig(check=False))
    t.m.tick(0.0)
    t.m.start()
    assert t.checks == 0 and t.m._thread is None and t.m.menu_label() is None


def test_manager_errors_and_up_to_date_stay_quiet() -> None:
    for res in (_result(error="could not reach GitHub (offline?)"), _result(version="0.1.2")):
        t = _Mgr(result=res, cfg=C.UpdatesConfig(auto_install=True))
        t.m.tick(0.0)
        assert t.checks == 1 and t.notes == [] and t.m.menu_label() is None and t.installs == []


def test_manager_pip_install_notifies_command_and_never_installs() -> None:
    t = _Mgr(result=_result("pip"), cfg=C.UpdatesConfig(auto_install=True))
    t.m.tick(0.0)
    assert "pipx upgrade project-away" in t.notes[0][1]
    assert t.m.install_now() is None
    assert t.installs == [] and t.quits == 0


def test_manager_menu_click_installs_and_quits() -> None:
    t = _Mgr()
    t.m.tick(0.0)
    out = t.m.install_now()
    assert out == U.InstallOutcome(True, "Installing")
    assert len(t.installs) == 1 and t.quits == 1 and t.m.installing is False
    assert t.notes[-1] == ("Project Away update", "Installing")


def test_manager_auto_install_only_when_idle() -> None:
    busy = _Mgr(cfg=C.UpdatesConfig(auto_install=True), idle=False)
    busy.m.tick(0.0)
    assert busy.installs == []
    idle = _Mgr(cfg=C.UpdatesConfig(auto_install=True), idle=True)
    idle.m.tick(0.0)
    assert len(idle.installs) == 1 and idle.quits == 1


def test_manager_never_auto_installs_deb() -> None:
    t = _Mgr(cfg=C.UpdatesConfig(auto_install=True), result=_result("deb"))
    t.m.tick(0.0)
    assert t.installs == [] and t.m.menu_label() == f"Update to v{V}"


def test_manager_failed_auto_install_is_not_retried() -> None:
    t = _Mgr(cfg=C.UpdatesConfig(auto_install=True), fail=True)
    t.m.tick(0.0)
    t.m.tick(10.0)
    t.m.tick(20.0)
    assert len(t.installs) == 1 and t.quits == 0
    assert t.notes[-1] == ("Project Away update failed", "checksum mismatch")
    assert t.m.last_error == "checksum mismatch" and t.m.installing is False


def test_manager_outcome_without_exit_keeps_running() -> None:
    t = _Mgr(outcome=U.InstallOutcome(False, "restart later"))
    t.m.tick(0.0)
    t.m.install_now()
    assert t.quits == 0


def test_tray_idle_detection() -> None:
    app = SimpleNamespace(pipeline=SimpleNamespace(paused=False))
    ctl = TrayController(app)
    ctl.mark_activity(now=100.0)
    assert ctl.is_idle(idle_s=600, now=699.0) is False
    assert ctl.is_idle(idle_s=600, now=700.0) is True
    app.pipeline.paused = True
    assert ctl.is_idle(idle_s=600, now=101.0) is True
    app.pipeline.paused = False
    ctl.paused = True
    assert ctl.is_idle(idle_s=600, now=101.0) is True


# --------------------------------------------------------------------------------------- config
def test_updates_config_defaults() -> None:
    u = C.AppConfig().updates
    assert (u.check, u.auto_install, u.channel, u.allow_unverified) == (True, False, "stable", False)
    assert C.to_dict(C.AppConfig())["updates"] == {
        "check": True,
        "auto_install": False,
        "channel": "stable",
        "allow_unverified": False,
    }


def test_updates_config_merge_validate_roundtrip(tmp_path: Path) -> None:
    cfg, unknown = C.from_dict({"updates": {"auto_install": True, "channel": "beta", "bogus": 1}})
    assert unknown == ["updates.bogus"]
    assert cfg.updates.auto_install is True and cfg.updates.channel == "beta" and cfg.updates.check is True
    p = C.save(cfg, tmp_path / "c.yaml")
    again, _ = C.load(p)
    assert again.updates == cfg.updates
    with pytest.raises(ValueError, match=re.escape("updates.channel must be stable|beta, got 'nightly'")):
        C.from_dict({"updates": {"channel": "nightly"}})


def test_settings_model_update_options(tmp_path: Path) -> None:
    m = SettingsModel(C.AppConfig(), tmp_path / "config.yaml")
    assert m.set_voice("updates.channel", "beta") is None and m.cfg.updates.channel == "beta"
    assert m.set_voice("updates.channel", "nightly") is not None and m.cfg.updates.channel == "beta"
    assert m.set_voice("updates.auto_install", 1) is None and m.cfg.updates.auto_install is True
    assert m.set_voice("updates.check", False) is None and m.cfg.updates.check is False
    assert m.set_voice("updates.allow_unverified", True) == "unknown setting 'updates.allow_unverified'"


# ------------------------------------------------------------------------------------------ CLI
def _args(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(["update", *argv])


def _cli_env(monkeypatch: pytest.MonkeyPatch, kind: str, tmp_path: Path) -> U.InstallTarget:
    tgt = U.InstallTarget(kind, tmp_path / "PA.AppImage" if kind == "appimage" else None, "x86_64")
    monkeypatch.setattr(U, "detect_install", lambda: tgt)
    return tgt


def test_cli_parser_has_update() -> None:
    a = _args("--check", "--yes")
    assert a.check is True and a.yes is True and a.fn is cmd_update
    assert EXIT_UPDATE_AVAILABLE == 10


def test_cli_check_up_to_date_exit_0(monkeypatch, tmp_path, capsys) -> None:
    _cli_env(monkeypatch, "windows", tmp_path)
    rc = cmd_update(_args("--check"), fetch=_fetcher({U.API_LATEST: _release_json(f"v{__version__}")}))
    out = capsys.readouterr().out
    assert rc == 0
    assert f"current: {__version__}" in out and f"latest:  {__version__}" in out and "up to date" in out


def test_cli_check_update_available_exit_10(monkeypatch, tmp_path, capsys) -> None:
    _cli_env(monkeypatch, "windows", tmp_path)
    urls: list[str] = []
    rc = cmd_update(_args("--check"), fetch=_fetcher({U.API_LATEST: _release_json()}), opener=_opener(urls=urls))
    assert rc == 10 and urls == []  # --check never downloads
    assert f"latest:  {V}" in capsys.readouterr().out


def test_cli_check_offline_exit_1(monkeypatch, tmp_path, capsys) -> None:
    _cli_env(monkeypatch, "windows", tmp_path)
    rc = cmd_update(_args("--check"), fetch=_fetcher({U.API_LATEST: _http_error(403)}))
    assert rc == 1 and "rate limit" in capsys.readouterr().err


def test_cli_update_pip_install_only_prints_command(monkeypatch, tmp_path, capsys) -> None:
    _cli_env(monkeypatch, "pip", tmp_path)
    urls: list[str] = []
    installs: list = []
    rc = cmd_update(
        _args("--yes"),
        fetch=_fetcher({U.API_LATEST: _release_json()}),
        opener=_opener(urls=urls),
        installer=lambda *a, **k: installs.append(a),
    )
    assert rc == 10 and urls == [] and installs == []
    assert "pipx upgrade project-away" in capsys.readouterr().out


def test_cli_update_yes_downloads_verifies_installs(monkeypatch, tmp_path, capsys) -> None:
    tgt = _cli_env(monkeypatch, "appimage", tmp_path)
    installs: list = []

    def fake_install(path, target, relaunch):
        installs.append((Path(path).read_bytes(), target, relaunch))
        return U.InstallOutcome(False, "Updated; restart Project Away.")

    urls: list[str] = []
    rc = cmd_update(
        _args("--yes"),
        fetch=_fetcher({U.API_LATEST: _release_json()}),
        opener=_opener(urls=urls),
        installer=fake_install,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert urls == [f"https://dl.example/{NAMES['appimage']}"]
    assert installs == [(PAYLOAD, tgt, False)]
    assert "100%" in out and "verified SHA-256 (api)" in out and "Updated; restart Project Away." in out


def test_cli_update_checksum_mismatch_exit_1(monkeypatch, tmp_path, capsys) -> None:
    _cli_env(monkeypatch, "appimage", tmp_path)
    installs: list = []
    rc = cmd_update(
        _args("--yes"),
        fetch=_fetcher({U.API_LATEST: _release_json()}),
        opener=_opener(b"evil"),
        installer=lambda *a, **k: installs.append(a),
    )
    assert rc == 1 and installs == []
    assert "checksum mismatch" in capsys.readouterr().err


def test_cli_update_without_yes_and_no_terminal_does_not_install(monkeypatch, tmp_path, capsys) -> None:
    _cli_env(monkeypatch, "appimage", tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # not a tty
    monkeypatch.setattr("builtins.input", lambda p: (_ for _ in ()).throw(AssertionError("must not prompt")))
    urls: list[str] = []
    rc = cmd_update(_args(), fetch=_fetcher({U.API_LATEST: _release_json()}), opener=_opener(urls=urls))
    assert rc == 10 and urls == []
    assert "away update --yes" in capsys.readouterr().out


def test_cli_update_windows_relaunches(monkeypatch, tmp_path) -> None:
    _cli_env(monkeypatch, "windows", tmp_path)
    seen: list[bool] = []
    rc = cmd_update(
        _args("-y"),
        fetch=_fetcher({U.API_LATEST: _release_json()}),
        opener=_opener(),
        installer=lambda p, t, relaunch: seen.append(relaunch) or U.InstallOutcome(True, "Installing"),
    )
    assert rc == 0 and seen == [True]


def test_cli_update_respects_channel_from_config(monkeypatch, tmp_path) -> None:
    _cli_env(monkeypatch, "windows", tmp_path)
    cfg, _ = C.from_dict({"updates": {"channel": "beta"}})
    C.save(cfg)
    calls: list[str] = []
    rc = cmd_update(_args("--check"), fetch=_fetcher({U.API_RELEASES: [_release_json("v0.0.1")]}, calls))
    assert rc == 0 and calls == [U.API_RELEASES]


def test_cli_main_dispatches_update(monkeypatch, capsys) -> None:
    from away.cli import main

    monkeypatch.setattr(U, "_default_fetch", _fetcher({U.API_LATEST: urllib.error.URLError("offline")}))
    assert main(["update", "--check"]) == 1
    assert "offline" in capsys.readouterr().err


def test_update_is_a_console_command_for_the_windowed_exe() -> None:
    from away.app import frozen

    assert frozen.wants_console(["update", "--check"]) is True


def test_inno_script_supports_silent_update_and_relaunch() -> None:
    iss = (Path(__file__).parents[1] / "packaging" / "windows" / "away.iss").read_text()
    run = iss.split("[Run]")[1].split("[UninstallRun]")[0]
    relaunch = [ln for ln in run.splitlines() if "Check: IsAutoUpdate" in ln]
    assert len(relaunch) == 1 and 'Parameters: "tray"' in relaunch[0] and "skipifsilent" not in relaunch[0]
    assert "postinstall skipifsilent" in run  # the interactive "Launch now" checkbox is unchanged
    assert sum("Check: ShouldDownloadModels" in ln for ln in run.splitlines() if "setup --yes" in ln) == 2
    code = iss.split("[Code]")[1]
    assert "{param:AUTOUPDATE|0}" in code and "WizardSilent" in code and "hand_landmarker.task" in code


def test_readme_updates_block_is_valid_config() -> None:
    import yaml

    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    block = readme.split("## Updates")[1].split("```yaml")[1].split("```")[0]
    cfg, unknown = C.from_dict(yaml.safe_load(block))
    assert unknown == [] and cfg.updates == C.UpdatesConfig()


# =========================================================== v0.1.10: batch helper + failure remedy
# The user's real machine (v0.1.8 -> v0.1.9): the hidden PowerShell helper started but the installer never ran.
G_DATA = Path(r"C:\Users\garza\AppData\Local\ProjectAway")
G_EXE = Path(r"C:\Users\garza\AppData\Local\Programs\ProjectAway\ProjectAway.exe")
G_SETUP = "ProjectAway-0.1.10-windows-x64-setup.exe"


def _g_argv(data: Path = G_DATA, exe: Path = G_EXE) -> list[str]:
    from pathlib import PureWindowsPath as W

    return U.windows_command(
        W(str(data)) / "updates" / G_SETUP, True, W(str(exe)).parent, W(str(data)) / "logs" / U.INSTALL_LOG
    )


def _g_batch(data: Path = G_DATA, exe: Path = G_EXE, pid: int = 4321) -> str:
    from pathlib import PureWindowsPath as W

    d = W(str(data))
    return U.windows_helper_batch(pid, _g_argv(data, exe), W(str(exe)), d / U.EXIT_CODE_FILE, d / "logs" / U.HELPER_LOG)


def test_version_parsing_orders_0_1_10_after_0_1_9() -> None:
    assert U.is_newer("0.1.10", "0.1.9") and not U.is_newer("0.1.9", "0.1.10")
    assert U.parse_version("v0.1.10") > U.parse_version("0.1.9") > U.parse_version("0.1.8")  # type: ignore[operator]
    assert max(["0.1.8", "0.1.10", "0.1.9"], key=lambda v: U.parse_version(v) or ()) == "0.1.10"
    assert __version__ == "0.1.31"


def test_batch_helper_with_the_users_real_paths() -> None:
    s = _g_batch()
    lines = s.splitlines()
    assert lines[0] == "@echo off"
    first_cmd = [ln for ln in lines if not ln.startswith(("@echo", "rem ", "setlocal", "set ", "chcp"))][0]
    assert first_cmd.startswith('>>"%LOG%" echo %DATE% %TIME% update helper started')  # logging comes first
    assert 'set "LOG=C:\\Users\\garza\\AppData\\Local\\ProjectAway\\logs\\update-helper.log"' in s
    assert 'set "EXITFILE=C:\\Users\\garza\\AppData\\Local\\ProjectAway\\update-exit-code.txt"' in s
    assert f'set "SETUP=C:\\Users\\garza\\AppData\\Local\\ProjectAway\\updates\\{G_SETUP}"' in s
    start = next(ln for ln in lines if ln.startswith('start "" /wait '))
    assert start == (
        f'start "" /wait "C:\\Users\\garza\\AppData\\Local\\ProjectAway\\updates\\{G_SETUP}" /VERYSILENT '
        "/SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS /NORESTARTAPPLICATIONS /AUTOUPDATE=1 "
        '/DIR="C:\\Users\\garza\\AppData\\Local\\Programs\\ProjectAway" '
        '/LOG="C:\\Users\\garza\\AppData\\Local\\ProjectAway\\logs\\update-install.log"'
    )
    # order: wait for the pid (bounded), kill leftovers, run the installer, record the exit code
    i_wait = s.index('tasklist /FI "PID eq 4321" /FO CSV /NH 2>nul | find /I "%APPEXE%" >nul')
    i_bound = s.index("if %WAITED% GEQ 60 goto app_timeout")
    i_kill = s.index('taskkill /F /IM "%APPEXE%"')
    i_start = s.index('start "" /wait')
    i_code = s.index('>"%EXITFILE%" echo %RC%')
    assert i_wait < i_bound < i_kill < i_start < s.index('set "RC=%ERRORLEVEL%"') < i_code
    assert s.rstrip().endswith("exit /b 0") and f'>>"%LOG%" echo %DATE% %TIME% {U.HELPER_DONE}' in s
    assert "ping -n 2 127.0.0.1 >nul" in s and "timeout /t" not in s  # timeout.exe dies with redirected stdin
    assert "powershell" not in s.lower() and "chcp" not in s  # all ASCII: no code page switch
    assert '>"%EXITFILE%" echo launch-failed' in s  # installer file missing


def test_batch_helper_quotes_spaces_specials_percent_and_unicode(tmp_path: Path) -> None:
    data = Path(r"C:\Users\Ana María (Work) & Co\AppData\Local\ProjectAway 100%")
    exe = Path(r"C:\Program Files (x86)\Project Away\ProjectAway.exe")
    s = _g_batch(data, exe)
    assert 'set "LOG=C:\\Users\\Ana María (Work) & Co\\AppData\\Local\\ProjectAway 100%%\\logs\\update-helper.log"' in s
    start = next(ln for ln in s.splitlines() if ln.startswith('start "" /wait '))
    assert '/DIR="C:\\Program Files (x86)\\Project Away"' in start
    assert f'"C:\\Users\\Ana María (Work) & Co\\AppData\\Local\\ProjectAway 100%%\\updates\\{G_SETUP}"' in start
    assert start.count('"') % 2 == 0 and "100%\\" not in start
    echo = next(ln for ln in s.splitlines() if "installer file not found" in ln)
    assert "^(Work^) ^& Co" in echo and "100%%" in echo  # echo prints the path, never runs "& Co"
    assert "chcp 65001 >nul 2>&1" in s.splitlines()[:5]  # UTF-8 file with a non-ASCII path
    assert U.cmd_arg("/AUTOUPDATE=1") == "/AUTOUPDATE=1" and U.cmd_arg("/VERYSILENT") == "/VERYSILENT"
    assert U.cmd_arg(r"/DIR=C:\a&b") == r'/DIR="C:\a&b"' and U.cmd_arg(r"C:\x\s.exe") == r'"C:\x\s.exe"'
    p = U.write_helper_batch(tmp_path / U.HELPER_SCRIPT, s)
    raw = p.read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")  # CRLF only (cmd's goto needs it)
    assert raw.decode("utf-8") == s.replace("\n", "\r\n")


def test_helper_command_line_and_launch_flags() -> None:
    script = G_DATA / "updates" / U.HELPER_SCRIPT
    cmd = U.windows_helper_command(script, {"SystemRoot": r"C:\WINDOWS"})
    assert cmd == f'"C:\\WINDOWS\\System32\\cmd.exe" /d /s /c ""{script}""'
    assert U.windows_helper_command(Path(r"C:\A B\x.cmd"), {}) == '"cmd.exe" /d /s /c ""C:\\A B\\x.cmd""'
    helper = U._detached_kwargs("win32", breakaway=True, console=True)["creationflags"]
    assert helper == 0x08000000 | 0x00000200 | 0x01000000  # CREATE_NO_WINDOW, never with DETACHED_PROCESS
    assert U._detached_kwargs("win32", console=True)["creationflags"] == 0x08000000 | 0x00000200
    assert U._detached_kwargs("win32", breakaway=True)["creationflags"] == 0x00000008 | 0x00000200 | 0x01000000
    assert U.windows_direct_command(_g_argv()).startswith(f'"{G_DATA}\\updates\\{G_SETUP}" /VERYSILENT')


class _Proc:
    """Popen stand-in: ``codes`` is what poll() returns on successive calls (last value repeats)."""

    def __init__(self, codes: list[int | None], on_poll=None) -> None:
        self.codes, self.on_poll, self.polls = list(codes), on_poll, 0

    def poll(self) -> int | None:
        self.polls += 1
        if self.on_poll:
            self.on_poll(self.polls)
        return self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_wait_alive_log_line_running_or_dead(tmp_path: Path) -> None:
    logf = tmp_path / U.HELPER_LOG
    c = _Clock()
    alive, why = U.wait_alive(_Proc([None]), logf, 3.0, c.sleep, c)  # running, no log yet
    assert alive and "still running" in why and c.t >= 3.0
    c = _Clock()

    def write(n: int) -> None:
        if n == 3:
            logf.write_text("x update helper started\n")

    alive, why = U.wait_alive(_Proc([None], write), logf, 3.0, c.sleep, c)
    assert alive and "helper log" in why and c.t < 1.0  # returns as soon as the first line is there
    logf.unlink()
    c = _Clock()
    alive, why = U.wait_alive(_Proc([None, None, 1]), logf, 3.0, c.sleep, c)
    assert not alive and "code 1" in why and "no log" in why
    assert U.wait_alive(object(), logf)[0] is True  # cannot poll: trust the launch


class _ProcLauncher:
    def __init__(self, procs: list[Any]) -> None:
        self.procs, self.calls = list(procs), []

    def __call__(self, cmd: Any, **kw: Any) -> Any:
        self.calls.append((cmd, kw))
        p = self.procs.pop(0)
        if isinstance(p, BaseException):
            raise p
        return p


def test_dead_helper_falls_back_to_the_next_method_and_logs_each(tmp_path: Path, caplog) -> None:
    setup = tmp_path / "updates" / G_SETUP
    setup.parent.mkdir()
    setup.write_bytes(b"x")
    tgt = U.InstallTarget("windows", G_EXE)
    c = _Clock()
    launch = _ProcLauncher([_Proc([None, 0]), _Proc([1]), _Proc([None])])  # helper dies twice, installer runs
    caplog.set_level("INFO", logger="away.updater")
    how = U.launch_windows_installer(setup, tgt, 7, True, launch, tmp_path, "win32", 3.0, c.sleep, c)
    assert how == "installer direct (breakaway)" and len(launch.calls) == 3
    assert [bool(kw["creationflags"] & U.CREATE_NO_WINDOW) for _c, kw in launch.calls] == [True, True, False]
    text = caplog.text
    assert "batch helper (breakaway) exited at once with code 0 and wrote no log; trying the next method" in text
    assert "batch helper exited at once with code 1" in text
    assert "installer scheduled via installer direct (breakaway) (still running after 3 s)" in text

    launch = _ProcLauncher([_Proc([2]), _Proc([2]), _Proc([2]), OSError(5, "denied")])
    with pytest.raises(U.UpdateError, match=r"installer direct: \[Errno 5\] denied"):
        U.launch_windows_installer(setup, tgt, 7, True, launch, tmp_path, "win32", 3.0, c.sleep, c)


def test_live_helper_is_accepted_and_old_helper_log_is_rotated(tmp_path: Path) -> None:
    setup = tmp_path / "updates" / G_SETUP
    setup.parent.mkdir()
    setup.write_bytes(b"x")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / U.HELPER_LOG).write_text("old attempt\n")
    (tmp_path / U.EXIT_CODE_FILE).write_text("5")

    def started(_n: int) -> None:
        (logs / U.HELPER_LOG).write_text("2026 update helper started: waiting\n")

    c = _Clock()
    launch = _ProcLauncher([_Proc([None], started)])
    tgt = U.InstallTarget("windows", G_EXE)
    how = U.launch_windows_installer(setup, tgt, 7, True, launch, tmp_path, "win32", 3.0, c.sleep, c)
    assert how == "batch helper (breakaway)" and len(launch.calls) == 1
    assert (logs / U.HELPER_OLD_LOG).read_text() == "old attempt\n"
    assert not (tmp_path / U.EXIT_CODE_FILE).exists()


def _failed_state(tmp: Path, installer: bool = True, helper: str | None = None, age: float = 0.0, code: str = ""):
    setup = tmp / "updates" / G_SETUP
    if installer:
        setup.parent.mkdir(parents=True, exist_ok=True)
        setup.write_bytes(b"x")
    U.write_state(tmp, "0.1.9", "0.1.10", G_SETUP, setup)
    if helper is not None:
        (tmp / "logs").mkdir(exist_ok=True)
        hl = tmp / "logs" / U.HELPER_LOG
        hl.write_text(helper)
        os.utime(hl, (hl.stat().st_mtime - age, hl.stat().st_mtime - age))
    if code:
        (tmp / U.EXIT_CODE_FILE).write_text(code)
    return setup


def test_report_no_helper_log_says_did_not_run_with_tray_remedy(tmp_path: Path) -> None:
    setup = _failed_state(tmp_path)
    out = U.previous_install(tmp_path, "0.1.9")
    assert out is not None and out.status == "failed" and out.installer == setup
    assert "the installer did not run" in out.message and "still v0.1.9" in out.message
    assert "Right-click the tray icon → Run update installer." in out.message
    assert setup.exists()  # the installer is kept
    assert U.load_failure(tmp_path, "0.1.9") == ("0.1.10", setup)  # survives restarts
    assert U.previous_install(tmp_path, "0.1.9") is None  # reported once


def test_report_still_installing_keeps_state_and_quotes_last_line(tmp_path: Path) -> None:
    line = "Mon 10/12/2026  3:33:05.12 starting installer: C:\\...\\setup.exe /VERYSILENT"
    _failed_state(tmp_path, helper=f"x update helper started\nx Project Away has exited\n{line}\n")
    out = U.previous_install(tmp_path, "0.1.9")
    assert out is not None and out.status == "installing" and not out.ok
    assert "still installing" in out.message and "the installer is running" in out.message
    assert line in out.message and out.helper_line == line
    assert (tmp_path / U.STATE_FILE).exists()  # reported again (as success) once the installer relaunches us
    _failed_state(tmp_path, helper="x update helper started: waiting for Project Away (pid 1) to exit\n")
    out = U.previous_install(tmp_path, "0.1.9")
    assert out is not None and out.status == "installing" and "getting ready" in out.message


def test_report_stale_or_finished_helper_log_is_a_failure_with_last_line(tmp_path: Path) -> None:
    _failed_state(tmp_path, helper="x update helper started\nx starting installer: s.exe\n", age=3600)
    out = U.previous_install(tmp_path, "0.1.9")
    assert out is not None and out.status == "failed"
    assert "the update helper stopped before the installer finished" in out.message
    assert "Last step: x starting installer: s.exe." in out.message
    _failed_state(tmp_path, helper=f"x installer exit code 5\nx {U.HELPER_DONE}\n", code="5")
    out = U.previous_install(tmp_path, "0.1.9")
    assert out is not None and out.status == "failed" and "aborted during the installation (exit code 5" in out.message
    assert f"Last step: x {U.HELPER_DONE}." in out.message
    (tmp_path / "updates" / G_SETUP).unlink()
    (tmp_path / "logs" / U.HELPER_LOG).unlink()
    _failed_state(tmp_path, installer=False)
    out = U.previous_install(tmp_path, "0.1.9")
    assert out is not None and out.installer is None and U.RELEASES_PAGE in out.message
    assert "Run update installer" not in out.message
    _failed_state(tmp_path)
    U.update_state(tmp_path, method="installer direct (breakaway)")
    out = U.previous_install(tmp_path, "0.1.9")
    assert out is not None and "started directly" in out.message


def test_report_success_clears_failure_record(tmp_path: Path) -> None:
    setup = _failed_state(tmp_path)
    U.record_failure(tmp_path, "0.1.10", setup, "earlier failure")
    out = U.previous_install(tmp_path, "0.1.10")
    assert out is not None and out.ok and out.message == "Project Away was updated to v0.1.10."
    assert not (tmp_path / U.FAILED_FILE).exists() and U.load_failure(tmp_path, "0.1.10") is None


def _mgr(tmp: Path, notes: list, opened: list, current: str = "0.1.9", **kw: Any) -> U.UpdateManager:
    return U.UpdateManager(
        C.UpdatesConfig(),
        notify=lambda t, m: notes.append((t, m)),
        data_dir=tmp,
        current=current,
        open_file=opened.append,
        **kw,
    )


def test_manager_failed_report_enables_run_installer_item(tmp_path: Path) -> None:
    setup = _failed_state(tmp_path)
    notes: list = []
    opened: list = []
    m = _mgr(tmp_path, notes, opened)
    assert m.run_installer_label() is None
    m.report_previous()
    assert notes[-1][0] == "Project Away update failed" and "Run update installer" in notes[-1][1]
    assert m.failed_version == "0.1.10" and m.run_installer_label() == "Run update installer"
    assert m.run_installer() is True and opened == [setup]  # a visible, double-click style run
    # next start: the failure was reported already, but the one-click remedy is still there
    m2 = _mgr(tmp_path, notes2 := [], opened)
    m2.report_previous()
    assert notes2 == [] and m2.run_installer_label() == "Run update installer" and m2.failed_version == "0.1.10"
    setup.unlink()
    assert m2.run_installer_label() is None
    assert m2.run_installer() is False and "installer is gone" in notes2[-1][1]


def test_manager_still_installing_is_not_called_a_failure(tmp_path: Path) -> None:
    _failed_state(tmp_path, helper="x update helper started: waiting\n")
    notes: list = []
    m = _mgr(tmp_path, notes, [])
    assert m.report_previous() is not None
    assert notes[-1][0] == "Project Away update still installing" and "did not" not in notes[-1][1]
    assert m.failed_version == "0.1.10" and m.run_installer_label() is None


def test_manager_failure_during_launch_offers_run_installer(tmp_path: Path) -> None:
    setup = tmp_path / "updates" / G_SETUP
    setup.parent.mkdir()
    setup.write_bytes(b"x")
    asset = U.Asset(G_SETUP, "https://dl/x", 1, "sha256:" + "0" * 64)
    rel = U.Release("v0.1.10", "0.1.10", (asset,))
    res = U.CheckResult("0.1.9", "0.1.10", True, rel, asset, U.InstallTarget("windows", G_EXE))

    def boom(_r: Any) -> U.InstallOutcome:
        raise U.UpdateError("could not start the installer (batch helper: exited at once with code 1)")

    notes: list = []
    m = _mgr(tmp_path, notes, [], checker=lambda: res, installer=boom)
    m.tick(0.0)
    m.install_now()
    assert notes[-1][0] == "Project Away update failed"
    assert notes[-1][1].endswith("Right-click the tray icon → Run update installer.")
    assert m.run_installer_label() == "Run update installer" and m.menu_label() == "Update to v0.1.10"


def test_tray_update_menu_shows_run_installer_only_after_a_failure(tmp_path: Path) -> None:
    from away.app.tray import update_menu_items

    class Item:
        def __init__(self, text: Any, action: Any, **kw: Any) -> None:
            self.text, self.action, self.kw = text, action, kw

        def label(self) -> str:
            return self.text(None) if callable(self.text) else self.text

        def visible(self) -> bool:
            v = self.kw.get("visible", True)
            return bool(v(None) if callable(v) else v)

    fake = SimpleNamespace(MenuItem=Item)
    m = _mgr(tmp_path, [], [])
    items = update_menu_items(fake, m, "upd", "run", "chk")
    shown = [i.label() for i in items if i.visible()]
    assert shown == ["Check for updates"]
    setup = _failed_state(tmp_path)
    m.report_previous()
    shown = {i.label(): i.action for i in items if i.visible()}
    assert shown == {"Run update installer": "run", "Check for updates": "chk"}
    assert m.failed_installer == setup
