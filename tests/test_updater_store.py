"""v0.1.25: a Microsoft Store (MSIX) copy runs from ``...\\WindowsApps\\...`` and is updated by the Store only.

The in-app updater must never check GitHub, download, or start the Inno installer there (it would install a
second copy beside the Store one), while ordinary Windows installs keep updating exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from away import config as C
from away.app import updater as U

STORE_EXE = r"C:\Program Files\WindowsApps\FernandoGarza.ProjectAway_0.1.25.0_x64__abc123\ProjectAway\ProjectAway.exe"
INNO_EXE = r"C:\Users\me\AppData\Local\Programs\ProjectAway\ProjectAway.exe"


def test_is_store_install_matches_the_windowsapps_folder_only() -> None:
    assert U.is_store_install(STORE_EXE)
    assert U.is_store_install(STORE_EXE.lower())
    assert U.is_store_install("D:/WindowsApps/Pkg_1.0_x64__x/ProjectAway.exe")  # Store apps moved to another drive
    assert not U.is_store_install(INNO_EXE)
    assert not U.is_store_install(r"C:\Tools\MyWindowsAppsBackup\ProjectAway.exe")  # a folder merely named like it
    assert not U.is_store_install(r"C:\Tools\WindowsApps.exe")  # the file name is not a folder


def test_detect_install_store_vs_inno() -> None:
    store = U.detect_install(platform="win32", frozen=True, env={}, executable=STORE_EXE, machine="AMD64")
    assert store.kind == "store" and store.kind not in U.INSTALLABLE
    assert U.asset_suffix(store) is None
    inno = U.detect_install(platform="win32", frozen=True, env={}, executable=INNO_EXE, machine="AMD64")
    assert inno.kind == "windows"  # unchanged for the normal installer


def test_check_never_fetches_for_a_store_copy() -> None:
    calls: list[Any] = []

    def fetch(url: str, timeout: float) -> bytes:
        calls.append(url)
        raise AssertionError("a Store copy must not ask GitHub for releases")

    res = U.check("0.1.25", target=U.InstallTarget("store", Path(STORE_EXE)), fetch=fetch)
    assert calls == []
    assert not res.update_available and not res.can_install and res.error is None
    assert "Microsoft Store" in res.summary()


def test_install_refuses_a_store_target(tmp_path: Path) -> None:
    launched: list[Any] = []
    with pytest.raises(U.UpdateError, match="Microsoft Store"):
        U.install(
            tmp_path / "setup.exe",
            U.InstallTarget("store", Path(STORE_EXE)),
            launcher=lambda *a, **k: launched.append(a),
            platform="win32",
            data_dir=tmp_path,
        )
    assert launched == []


def test_manager_never_auto_installs_or_nags_in_a_store_copy(tmp_path: Path) -> None:
    notes: list[tuple[str, str]] = []
    checks: list[int] = []

    def checker() -> U.CheckResult:
        checks.append(1)
        return U.check("0.1.25", target=U.InstallTarget("store", Path(STORE_EXE)), fetch=lambda u, t: b"")

    cfg = C.UpdatesConfig()
    cfg.auto_install = True
    m = U.UpdateManager(cfg, notify=lambda t, msg: notes.append((t, msg)), data_dir=tmp_path, current="0.1.25")
    m.checker = checker
    m.installer = lambda r: (_ for _ in ()).throw(AssertionError("no installer in a Store copy"))  # type: ignore[assignment]
    m.is_idle = lambda: True  # type: ignore[assignment]
    m.tick(0.0)
    m.tick(10_000_000.0)  # long after the interval: still no second check
    assert len(checks) == 1
    assert m.available is None and m.menu_label() is None
    assert notes == []
    assert m.install_now() is None
    m.check_now()  # the tray's "Check for updates" says where updates come from
    assert notes and "Microsoft Store" in notes[-1][1]
