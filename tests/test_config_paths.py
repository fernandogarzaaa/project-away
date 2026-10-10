import pytest
import yaml

from away import config as C
from away import paths


def test_defaults_valid_and_roundtrip(tmp_path):
    cfg, unknown = C.from_dict(None)
    assert unknown == []
    p = C.save(cfg, tmp_path / "c.yaml")
    data = yaml.safe_load(p.read_text())
    assert data["hands"]["cursor"]["active_region"] == [0.15, 0.15, 0.85, 0.85]
    cfg2, _ = C.load(p)
    assert cfg2 == cfg


def test_merge_nested_unknown_and_tuple():
    cfg, unknown = C.from_dict(
        {
            "hands": {"cursor": {"mode": "absolute", "active_region": [0, 0, 1, 1], "bogus": 1}, "camera_index": 2},
            "nope": 3,
            "overlay": True,
        }
    )
    assert cfg.hands.cursor.mode == "absolute"
    assert cfg.hands.cursor.active_region == (0, 0, 1, 1)
    assert cfg.hands.camera_index == 2
    assert cfg.overlay is True
    assert unknown == ["hands.cursor.bogus", "nope"]


@pytest.mark.parametrize(
    "bad",
    [
        {"hands": {"cursor": {"mode": "warp"}}},
        {"voice": {"output": "fax"}},
        {"voice": {"asr_backend": "siri"}},
        {"voice": {"ptt_mode": "maybe"}},
        {"wifi": {"source": "esp32"}},
        {"hands": {"gestures": {"pinch_on": 0.5, "pinch_off": 0.5}}},
    ],
)
def test_validation(bad):
    with pytest.raises(ValueError):
        C.from_dict(bad)


def test_valid_alternatives():
    C.from_dict({"voice": {"output": "type", "asr_backend": "parakeet", "ptt_mode": "hold"}})
    for src in ("rssi", "picoscenes", "iax", "synthetic"):
        assert C.from_dict({"wifi": {"source": src}})[0].wifi.source == src


def test_load_missing_and_bad(tmp_path, monkeypatch):
    cfg, unknown = C.load(tmp_path / "none.yaml")
    assert cfg == C.AppConfig() and unknown == []
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        C.load(bad)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert C.load(empty)[0] == C.AppConfig()
    env = tmp_path / "env.yaml"
    env.write_text("overlay: true\n")
    monkeypatch.setenv("AWAY_CONFIG", str(env))
    assert C.load()[0].overlay is True


def test_save_default_location():
    p = C.save(C.AppConfig())
    assert p == paths.config_file() and p.exists()


@pytest.mark.parametrize(
    "plat,data_tail,cfg_tail",
    [
        ("win32", ("AppData", "Local", "ProjectAway"), ("AppData", "Roaming", "ProjectAway")),
        (
            "darwin",
            ("Library", "Application Support", "ProjectAway"),
            ("Library", "Application Support", "ProjectAway"),
        ),
        ("linux", (".local", "share", "project-away"), (".config", "project-away")),
    ],
)
def test_paths_per_os(monkeypatch, tmp_path, plat, data_tail, cfg_tail):
    for v in ("AWAY_DATA_DIR", "AWAY_CONFIG_DIR", "LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AWAY_HOME_OVERRIDE", str(tmp_path))
    assert paths.data_dir(plat) == tmp_path.joinpath(*data_tail)
    assert paths.config_dir(plat) == tmp_path.joinpath(*cfg_tail)


def test_paths_env_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("AWAY_DATA_DIR")
    monkeypatch.delenv("AWAY_CONFIG_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xd"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xc"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "la"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "ra"))
    assert paths.data_dir("linux") == tmp_path / "xd" / "project-away"
    assert paths.config_dir("linux") == tmp_path / "xc" / "project-away"
    assert paths.data_dir("win32") == tmp_path / "la" / "ProjectAway"
    assert paths.config_dir("win32") == tmp_path / "ra" / "ProjectAway"
    monkeypatch.setenv("AWAY_DATA_DIR", str(tmp_path / "d"))
    assert paths.models_dir() == tmp_path / "d" / "models"
    assert paths.recordings_dir() == tmp_path / "d" / "recordings"
