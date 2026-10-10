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
