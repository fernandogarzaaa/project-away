"""Sandbox session lifecycle: launch a disposable agent desktop, watch it, throw it away.

The recommended backend is Windows Sandbox, driven by a generated ``.wsb`` file: mapped
folders ship the guest bridge into the sandbox and ``LogonCommand`` starts it at logon.
Closing the session discards everything, which matches an agent task's lifecycle.

The ``.wsb`` writer is pure (unit-tested). Launching is Windows-only and is marked
accordingly; tests drive the lifecycle through :class:`FakeSessionBackend`.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class MappedFolder:
    host_path: str
    sandbox_path: str
    readonly: bool = True


@dataclass(frozen=True)
class SessionConfig:
    """Everything needed to describe one disposable agent desktop."""

    memory_mb: int = 4096
    networking: bool = True
    vgpu: bool = True
    audio_input: bool = False
    video_input: bool = False
    protected_client: bool = False
    clipboard_redirection: bool = False
    printer_redirection: bool = False
    mapped_folders: tuple[MappedFolder, ...] = ()
    logon_command: str = ""


def _bool_text(value: bool) -> str:
    return "Enable" if value else "Disable"


def write_wsb(config: SessionConfig) -> str:
    """Render ``config`` as a Windows Sandbox ``.wsb`` file. Pure; unit-tested."""
    root = ET.Element("Configuration")
    ET.SubElement(root, "VGpu").text = _bool_text(config.vgpu)
    ET.SubElement(root, "Networking").text = _bool_text(config.networking)
    if config.memory_mb > 0:
        ET.SubElement(root, "MemoryInMB").text = str(int(config.memory_mb))
    if config.mapped_folders:
        folders = ET.SubElement(root, "MappedFolders")
        for m in config.mapped_folders:
            folder = ET.SubElement(folders, "MappedFolder")
            ET.SubElement(folder, "HostFolder").text = m.host_path
            ET.SubElement(folder, "SandboxFolder").text = m.sandbox_path
            ET.SubElement(folder, "ReadOnly").text = "true" if m.readonly else "false"
    ET.SubElement(root, "AudioInput").text = _bool_text(config.audio_input)
    ET.SubElement(root, "VideoInput").text = _bool_text(config.video_input)
    ET.SubElement(root, "ProtectedClient").text = _bool_text(config.protected_client)
    ET.SubElement(root, "ClipboardRedirection").text = _bool_text(config.clipboard_redirection)
    ET.SubElement(root, "PrinterRedirection").text = _bool_text(config.printer_redirection)
    if config.logon_command:
        logon = ET.SubElement(root, "LogonCommand")
        ET.SubElement(logon, "Command").text = config.logon_command
    tree = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + tree


@dataclass
class SessionHandle:
    """An opaque handle to a launched session. Backends define what ``token`` holds."""

    backend_name: str
    token: Any = None
    wsb_path: str = ""
    launched_at: float = 0.0


class SessionBackend(Protocol):
    """A way to host the agent desktop. Windows-only backends raise on other platforms."""

    name: str

    def launch(self, config: SessionConfig) -> SessionHandle: ...
    def is_alive(self, handle: SessionHandle) -> bool: ...
    def terminate(self, handle: SessionHandle) -> None: ...


class FakeSessionBackend:
    """Scripted backend for tests: launch/terminate/is_alive with no Windows."""

    name = "fake"

    def __init__(self) -> None:
        self.launched: list[SessionConfig] = []
        self.terminated: list[SessionHandle] = []
        self._alive: dict[int, bool] = {}

    def launch(self, config: SessionConfig) -> SessionHandle:
        handle = SessionHandle(backend_name=self.name, token=len(self.launched), launched_at=time.monotonic())
        self.launched.append(config)
        self._alive[handle.token] = True
        return handle

    def is_alive(self, handle: SessionHandle) -> bool:
        return bool(self._alive.get(handle.token, False))

    def terminate(self, handle: SessionHandle) -> None:
        self.terminated.append(handle)
        self._alive[handle.token] = False


class SandboxSession:
    """Owns one disposable agent desktop from launch to teardown."""

    def __init__(
        self,
        backend: SessionBackend,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self._clock = clock
        self._handle: SessionHandle | None = None

    @property
    def handle(self) -> SessionHandle | None:
        return self._handle

    def start(self, config: SessionConfig, wsb_dir: Path | None = None) -> SessionHandle:
        """Write the .wsb (Sandbox backends) and launch. Idempotent while alive."""
        if self._handle is not None and self._backend.is_alive(self._handle):
            return self._handle
        wsb_path = ""
        if self._backend.name == "windows-sandbox":
            target = Path(wsb_dir) if wsb_dir is not None else Path.cwd()
            wsb_path = str(target / f"away-agent-{int(self._clock())}.wsb")
            Path(wsb_path).write_text(write_wsb(config), encoding="utf-8")
        self._handle = self._backend.launch(config)
        self._handle.wsb_path = wsb_path
        return self._handle

    def wait_ready(
        self,
        ready: Callable[[], bool],
        timeout_s: float = 120.0,
        poll_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> bool:
        """Wait until the guest bridge answers. ``ready`` is polled, never slept blindly."""
        if self._handle is None:
            return False
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            if ready():
                return True
            if not self._backend.is_alive(self._handle):
                return False
            sleep(poll_s)
        return False

    def stop(self) -> None:
        """Terminate the session. Closing a Sandbox discards everything inside."""
        if self._handle is not None:
            try:
                self._backend.terminate(self._handle)
            finally:
                self._handle = None

    @property
    def active(self) -> bool:
        return self._handle is not None and self._backend.is_alive(self._handle)


@dataclass
class HyperVBackend:
    """Fallback backend interface (not implemented in the spike).

    A real implementation provisions a Windows 11 VM, enables RDP (or a VNC server),
    and returns a handle carrying the guest address. Kept as a dataclass so the shape
    of the fallback is pinned down before anyone builds it.
    """

    name: str = "hyper-v"
    vm_name: str = "away-agent-desktop"
    rdp_port: int = 3389

    def launch(self, config: SessionConfig) -> SessionHandle:  # pragma: no cover - needs Hyper-V
        raise NotImplementedError("Hyper-V backend is a spike stub; validate on the Acer first")

    def is_alive(self, handle: SessionHandle) -> bool:  # pragma: no cover - needs Hyper-V
        raise NotImplementedError("Hyper-V backend is a spike stub; validate on the Acer first")

    def terminate(self, handle: SessionHandle) -> None:  # pragma: no cover - needs Hyper-V
        raise NotImplementedError("Hyper-V backend is a spike stub; validate on the Acer first")


__all__ = [
    "FakeSessionBackend",
    "HyperVBackend",
    "MappedFolder",
    "SandboxSession",
    "SessionBackend",
    "SessionConfig",
    "SessionHandle",
    "write_wsb",
]
