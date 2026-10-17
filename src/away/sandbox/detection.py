"""Runtime capability detection for the agent's own desktop.

All functions take injected runners/probes so they are pure and unit-testable on any OS,
including Linux CI where no Windows capability exists. The app calls these at runtime and
enables the sandboxed desktop only when a backend is actually available; otherwise it keeps
host control under the step budget (no silent fallback).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class EditionInfo:
    """What we know about the host OS. ``edition`` is e.g. ``"Pro"``, ``"Home"``, or ``""``."""

    platform: str  # sys.platform
    edition: str = ""
    build: str = ""


_SANDBOX_EDITIONS = {"pro", "enterprise", "education"}


def detect_edition(
    run: Callable[[list[str]], str] | None = None,
    platform: str | None = None,
) -> EditionInfo:
    """Detect the Windows edition. ``run`` executes a command and returns its stdout.

    On non-Windows (or when detection fails) the edition is ``""`` and nothing raises.
    """
    plat = sys.platform if platform is None else platform
    if plat != "win32":
        return EditionInfo(platform=plat)
    if run is None:  # pragma: no cover - real Windows only
        import subprocess

        def run(cmd: list[str]) -> str:  # type: ignore[misc]
            return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout

    edition, build = "", ""
    try:
        out = run(["wmic", "os", "get", "Caption,BuildNumber", "/format:list"])
        for line in out.splitlines():
            line = line.strip()
            if line.lower().startswith("caption="):
                caption = line.split("=", 1)[1]
                # e.g. "Microsoft Windows 11 Pro" -> "Pro"
                edition = caption.rsplit(" ", 1)[-1] if " " in caption else caption
            elif line.lower().startswith("buildnumber="):
                build = line.split("=", 1)[1].strip()
    except Exception:
        return EditionInfo(platform=plat)
    return EditionInfo(platform=plat, edition=edition, build=build)


def sandbox_available(
    info: EditionInfo,
    sandbox_exe_present: bool,
    virtualization_on: bool,
) -> bool:
    """True when Windows Sandbox can host the agent desktop.

    Needs Pro/Enterprise/Education (Sandbox is not on Home), the Sandbox feature installed,
    and virtualization enabled in BIOS.
    """
    return (
        info.platform == "win32"
        and info.edition.strip().lower() in _SANDBOX_EDITIONS
        and bool(sandbox_exe_present)
        and bool(virtualization_on)
    )


def hyperv_available(info: EditionInfo, hyperv_present: bool) -> bool:
    """True when Hyper-V can host the fallback VM. Same edition gate as Sandbox."""
    return info.platform == "win32" and info.edition.strip().lower() in _SANDBOX_EDITIONS and bool(hyperv_present)
