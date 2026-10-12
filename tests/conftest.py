import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Never touch the real user folders from tests."""
    monkeypatch.setenv("AWAY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AWAY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("AWAY_CONFIG", raising=False)
    monkeypatch.delenv("AWAY_LLM_BASE_URL", raising=False)
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
def no_camera_mode_store():
    """Camera modes kept across starts (camera-modes.json) are off unless a test turns them on."""
    from away.multicam import capture

    capture.use_mode_store(None)
    yield
    capture.use_mode_store(None)
