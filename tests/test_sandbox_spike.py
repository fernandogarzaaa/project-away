"""Spike tests for the agent's own desktop (src/away/sandbox).

Everything here runs on any OS: detection takes injected runners, the session lifecycle
runs against FakeSessionBackend, and the bridges run against fakes. Anything that needs
Windows Sandbox or Hyper-V is behind the on-Acer checklist in docs/sandbox-vm-spike.md,
not in this file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from away.sandbox import bridge as B
from away.sandbox import detection as D
from away.sandbox import session as S


def _wmic(caption: str, build: str = "26300") -> str:
    return f"Caption={caption}\r\nBuildNumber={build}\r\n"


# ------------------------------------------------------------- detection


def test_detect_edition_parses_pro() -> None:
    info = D.detect_edition(run=lambda cmd: _wmic("Microsoft Windows 11 Pro"), platform="win32")
    assert info.edition == "Pro"
    assert info.build == "26300"


def test_detect_edition_non_windows_is_unknown() -> None:
    info = D.detect_edition(run=lambda cmd: (_ for _ in ()).throw(AssertionError("no run on linux")))
    assert info.platform != "win32"
    assert info.edition == ""


def test_detect_edition_failure_never_raises() -> None:
    def boom(cmd: object) -> str:
        raise OSError("wmic missing")

    info = D.detect_edition(run=boom, platform="win32")
    assert info.edition == ""


def test_sandbox_available_needs_pro_edition_and_feature() -> None:
    pro = D.EditionInfo(platform="win32", edition="Pro")
    home = D.EditionInfo(platform="win32", edition="Home")
    assert D.sandbox_available(pro, sandbox_exe_present=True, virtualization_on=True)
    assert not D.sandbox_available(home, sandbox_exe_present=True, virtualization_on=True)
    assert not D.sandbox_available(pro, sandbox_exe_present=False, virtualization_on=True)
    assert not D.sandbox_available(pro, sandbox_exe_present=True, virtualization_on=False)
    assert not D.sandbox_available(D.EditionInfo(platform="linux"), True, True)


def test_hyperv_available_same_edition_gate() -> None:
    pro = D.EditionInfo(platform="win32", edition="Pro")
    home = D.EditionInfo(platform="win32", edition="Home")
    assert D.hyperv_available(pro, True)
    assert not D.hyperv_available(home, True)
    assert not D.hyperv_available(pro, False)


# ------------------------------------------------------------- session


def test_write_wsb_renders_config() -> None:
    config = S.SessionConfig(
        memory_mb=4096,
        networking=False,
        vgpu=False,
        mapped_folders=(S.MappedFolder(r"C:\Temp\away-share", r"C:\AwayShare", True),),
        logon_command="powershell -File C:\\AwayShare\\start-bridge.ps1",
    )
    xml = S.write_wsb(config)
    root = ET.fromstring(xml.split("\n", 1)[1])
    assert root.findtext("Networking") == "Disable"
    assert root.findtext("VGpu") == "Disable"
    assert root.findtext("MemoryInMB") == "4096"
    folder = root.find("MappedFolders/MappedFolder")
    assert folder is not None
    assert folder.findtext("HostFolder") == r"C:\Temp\away-share"
    assert folder.findtext("ReadOnly") == "true"
    assert root.findtext("LogonCommand/Command") == "powershell -File C:\\AwayShare\\start-bridge.ps1"
    # isolation defaults: nothing leaks unless asked
    assert root.findtext("ClipboardRedirection") == "Disable"
    assert root.findtext("PrinterRedirection") == "Disable"


def test_session_lifecycle_with_fake_backend() -> None:
    backend = S.FakeSessionBackend()
    session = S.SandboxSession(backend)
    assert not session.active
    handle = session.start(S.SessionConfig())
    assert session.active
    assert backend.launched and backend.is_alive(handle)
    # starting again while alive is idempotent
    assert session.start(S.SessionConfig()) is handle
    assert len(backend.launched) == 1
    session.stop()
    assert not session.active
    assert backend.terminated == [handle]


def test_wait_ready_polls_until_true() -> None:
    backend = S.FakeSessionBackend()
    session = S.SandboxSession(backend)
    session.start(S.SessionConfig())
    attempts = {"n": 0}

    def ready() -> bool:
        attempts["n"] += 1
        return attempts["n"] >= 3

    assert session.wait_ready(ready, timeout_s=30.0, poll_s=0.001, sleep=lambda s: None)
    assert attempts["n"] == 3


def test_wait_ready_false_when_session_dies() -> None:
    backend = S.FakeSessionBackend()
    session = S.SandboxSession(backend)
    handle = session.start(S.SessionConfig())
    backend.terminate(handle)  # guest died before the bridge answered
    assert not session.wait_ready(lambda: False, timeout_s=30.0, poll_s=0.001, sleep=lambda s: None)


# ------------------------------------------------------------- bridge


def test_frame_valid_rejects_bad_shapes() -> None:
    good = B.Frame(width=2, height=1, pixels=bytes(6))
    assert good.valid()
    assert not B.Frame(width=0, height=1, pixels=b"").valid()
    assert not B.Frame(width=2, height=1, pixels=bytes(5)).valid()


def test_clamp_point_stays_in_frame() -> None:
    assert B.clamp_point(-5, 999, 320, 200) == (0, 199)
    assert B.clamp_point(10, 20, 320, 200) == (10, 20)


def test_fake_screenshot_bridge_cycles_frames() -> None:
    br = B.FakeScreenshotBridge(width=16, height=8)
    f1 = br.capture()
    f2 = br.capture()
    assert f1.valid() and f2.valid()
    assert f1.pixels != f2.pixels  # frames change
    assert (f1.cursor_x, f1.cursor_y) != (f2.cursor_x, f2.cursor_y)


def test_fake_input_bridge_records_and_clamps() -> None:
    br = B.FakeInputBridge(width=100, height=50)
    br.move(1000, -3)
    br.click(10, 20, button="right", count=2)
    br.key("Enter")
    br.type_text("hello")
    kinds = [c[0] for c in br.calls]
    assert kinds == ["move", "click", "key", "type_text"]
    assert br.calls[0] == ("move", (99, 0))  # clamped into the guest frame
    assert br.calls[1][1][2:] == ("right", 2)
