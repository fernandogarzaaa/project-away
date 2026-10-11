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
