import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Never touch the real user folders from tests."""
    monkeypatch.setenv("AWAY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AWAY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("AWAY_CONFIG", raising=False)
    monkeypatch.delenv("AWAY_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("AWAY_NO_CRASHLOG", "1")  # cli.main must not replace pytest's fault handler / excepthooks
    monkeypatch.setenv("AWAY_NATIVE_SETTINGS", "0")  # v0.1.35: `away settings` on a Mac never opens the real native app
    yield


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture(autouse=True)
def fixed_hardware(monkeypatch):
    """Hermetic hardware: every test sees the same 8-core / 16 GB, no-GPU machine (CI runners differ),
    unless it builds its own Hardware and passes it in."""
    from away.app import gpupack, hardware

    hw = hardware.Hardware(
        os="linux",
        arch="x86_64",
        cpu_name="Test CPU",
        cores_physical=8,
        cores_logical=16,
        ram_mb=16384,
        disk_free_mb=100_000,
    )
    monkeypatch.setattr(hardware, "cached", lambda: hw)
    monkeypatch.setattr(gpupack, "cublas_available", lambda platform=None: False)
    yield hw


@pytest.fixture(autouse=True)
def no_real_camera_listing(monkeypatch):
    """Hermetic cameras: no DirectShow COM enumeration (Windows CI) and no camera listing from the app runner,
    unless a test injects its own enumerator / list."""
    from away.app.runner import AwayApp
    from away.multicam import capture

    monkeypatch.setattr(capture, "windows_dshow_cameras", lambda category=capture.CLSID_VideoInputDeviceCategory: [])
    monkeypatch.setattr(AwayApp, "_camera_list", lambda self, log=False: None)
    capture._last_logged = None
    yield


@pytest.fixture(autouse=True)
def unmeasured_camera_modes(monkeypatch):
    """v0.1.16 measures every camera mode for 1.5 s (capture.MEASURE_S). Tests with fake cameras that are not
    about the frame rate accept the first frame unmeasured; frame-rate tests pass ``measure_s`` and a fake clock."""
    from away.multicam import capture

    monkeypatch.setattr(capture, "MEASURE_S", 0.0)
    yield


@pytest.fixture(autouse=True)
def no_camera_mode_store():
    """Camera modes kept across starts (camera-modes.json) are off unless a test turns them on."""
    from away.multicam import capture

    capture.use_mode_store(None)
    yield
    capture.use_mode_store(None)


@pytest.fixture(autouse=True)
def no_real_winrt(monkeypatch):
    """v0.1.19: the WinRT MediaCapture backend never touches real WinRT in tests (the Windows CI runner has the
    pywinrt packages but no camera); tests that exercise it inject fakes. The backend preference is reset."""
    from away.multicam import capture

    monkeypatch.setattr(capture, "winrt_factory_available", lambda: (False, "WinRT disabled in tests"))
    capture._backend_pref = "auto"
    yield
    capture._backend_pref = "auto"
