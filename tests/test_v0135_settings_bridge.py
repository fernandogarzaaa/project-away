"""v0.1.35: the JSON bridge behind the native macOS Settings app, and how `away settings` picks native vs Tk.

The bridge must validate through the same SettingsModel setters as the Tk window, save only when every value is
valid, never accept a key that is not in its document, and keep stdout pure JSON in ``serve`` mode."""

from __future__ import annotations

import io
import json
import plistlib
from pathlib import Path
from typing import Any

import pytest

from away import config as cfgmod
from away.app import settings_bridge as sb
from away.app import settings_native as sn


def _doc(platform: str = "darwin") -> dict[str, Any]:
    return sb.build_document(sb._model(), platform)


def _saved() -> cfgmod.AppConfig:
    cfg, _ = cfgmod.load(cfgmod.config_path())
    return cfg


# ------------------------------------------------------------------------------------------------ the document
def test_document_has_every_sidebar_section_in_order_with_symbols():
    doc = _doc()
    assert [s["id"] for s in doc["sections"]] == [s["id"] for s in sb.SECTIONS]
    assert all(s["symbol"] and s["title"] for s in doc["sections"])
    assert doc["version"] and doc["config_path"].endswith("config.yaml")


def test_every_field_carries_type_label_and_current_value():
    fields = sb.fields(_doc())
    assert fields["voice.mode"]["type"] == "choice"
    assert {c["value"] for c in fields["voice.mode"]["choices"]} == set(cfgmod.VOICE_MODES)
    assert fields["voice.mode"]["value"] == cfgmod.AppConfig().voice.mode
    assert fields["app.always_on"] == {**fields["app.always_on"], "type": "bool", "value": True}
    for f in fields.values():
        assert f["type"] in {"bool", "choice", "text", "hotkey", "number"}, f
        assert f["label"], f


def test_every_tunable_slider_appears_with_its_range_so_new_tunables_need_no_swift_change():
    model = sb._model()
    fields = sb.fields(sb.build_document(model, "darwin"))
    for t in model.tunables():
        if t.path == "hands.cursor.big_cursor_size":
            continue  # Windows-only
        f = fields[t.path]
        assert (f["type"], f["min"], f["max"], f["is_int"]) == ("number", t.lo, t.hi, t.is_int)


def test_unknown_tuning_group_lands_in_advanced(monkeypatch):
    from away.app.settings_gui import SettingsModel, Tunable

    real = SettingsModel.tunables

    def plus_one(self):
        return [*real(self), Tunable("hands.cursor.gain", "x", 0.1, 10.0, 2.2, False, group="Brand new group")]

    monkeypatch.setattr(SettingsModel, "tunables", plus_one)
    doc = _doc()
    adv = next(s for s in doc["sections"] if s["id"] == "advanced")
    assert any(g["title"] == "Brand new group" for g in adv["groups"])


def test_windows_only_fields_are_hidden_on_a_mac_and_shown_on_windows():
    mac, win = sb.fields(_doc("darwin")), sb.fields(_doc("win32"))
    for key in ("hands.camera_backend", "hands.cursor.big_cursor_size"):
        assert key not in mac and key in win


def test_language_choices_use_codes_with_auto_as_empty_string():
    f = sb.fields(_doc())["voice.language"]
    assert f["choices"][0]["value"] == "" and "Auto" in f["choices"][0]["label"]
    assert {"value": "en", "label": "English"} in f["choices"]


def test_permissions_list_the_four_macos_privacy_pages():
    perms = {p["id"]: p for p in sb.permissions("darwin")}
    assert list(perms) == ["accessibility", "input_monitoring", "camera", "microphone"]
    assert perms["accessibility"]["url"].endswith("Privacy_Accessibility")
    assert perms["input_monitoring"]["url"].endswith("Privacy_ListenEvent")
    assert perms["camera"]["url"].endswith("Privacy_Camera")
    assert perms["microphone"]["url"].endswith("Privacy_Microphone")
    assert all(p["granted"] is None for p in sb.permissions("linux"))  # unknown off a Mac, never a crash


def test_document_carries_the_commands_for_windows_the_native_app_opens():
    cmds = _doc()["commands"]
    assert cmds["tk_settings"][-2:] == ["settings", "--tk"]
    assert cmds["first_run"][-1] == "tour" and cmds["calibrate"][-1] == "calibrate"


# --------------------------------------------------------------------------------------------- set
def test_set_saves_valid_changes_through_the_settings_model():
    out = sb.set_values({"voice.mode": "toggle", "hands.cursor.gain": 3.1, "app.always_on": False})
    assert out["ok"] and out["errors"] == {} and out["saved"]
    cfg = _saved()
    assert (cfg.voice.mode, cfg.hands.cursor.gain, cfg.app.always_on) == ("toggle", 3.1, False)
    assert sb.fields(out["document"])["voice.mode"]["value"] == "toggle"


def test_set_is_all_or_nothing_when_one_value_is_invalid():
    out = sb.set_values({"voice.mode": "toggle", "hands.cursor.gain": 999})
    assert not out["ok"] and "hands.cursor.gain" in out["errors"] and out["saved"] is None
    assert not cfgmod.config_path().exists() or _saved().voice.mode == cfgmod.AppConfig().voice.mode
    assert sb.fields(out["document"])["voice.mode"]["value"] == cfgmod.AppConfig().voice.mode


@pytest.mark.parametrize(
    ("changes", "key", "needle"),
    [
        ({"voice.mode": "bogus"}, "voice.mode", "must be one of"),
        ({"voice.wake_word_enabled": "false"}, "voice.wake_word_enabled", "true or false"),
        ({"hands.cursor.gain": "fast"}, "hands.cursor.gain", "number"),
        ({"hands.cursor.gain": True}, "hands.cursor.gain", "number"),
        ({"voice.hotkey": "<ctrl>+<nosuchkey>"}, "voice.hotkey", "nosuchkey"),
        ({"bindings.allow_shell": True}, "bindings.allow_shell", "unknown setting"),
        ({"no.such.key": 1}, "no.such.key", "unknown setting"),
    ],
)
def test_set_rejects_bad_values_with_the_models_message(changes, key, needle):
    out = sb.set_values(changes)
    assert not out["ok"]
    assert needle in out["errors"][key]
    assert not cfgmod.config_path().exists()  # nothing saved


def test_set_routes_language_smooth_pointer_and_comfort_switches():
    out = sb.set_values({"voice.language": "es", "pointer.smooth": True, "attention.enabled": True})
    assert out["ok"], out["errors"]
    cfg = _saved()
    assert cfg.voice.language == "es"
    assert cfg.hands.cursor.smoothing == "predictive"
    assert cfg.attention.enabled is True
    assert sb.set_values({"voice.language": ""})["ok"] and _saved().voice.language is None


def test_set_keeps_changes_saved_by_someone_else_since_the_window_opened():
    sb.set_values({"voice.mode": "toggle"})
    cfg = _saved()
    cfg.face.gate_dictation = False  # e.g. the tray's face-gate switch writes the file
    cfgmod.save(cfg, cfgmod.config_path())
    assert sb.set_values({"app.always_on": False})["ok"]
    cfg = _saved()
    assert (cfg.voice.mode, cfg.face.gate_dictation, cfg.app.always_on) == ("toggle", False, False)


def test_set_writes_config_yaml_so_the_running_app_hot_reloads_it():
    # runner._watch_config reloads on a new mtime of config_path(): a save through the bridge is enough
    sb.set_values({"voice.mode": "toggle"})
    assert cfgmod.config_path().exists()
    assert "mode: toggle" in cfgmod.config_path().read_text("utf-8")


# --------------------------------------------------------------------------------------------- CLI / serve
def test_cli_get_prints_one_json_document(capsys):
    from away.cli import main

    assert main(["settings-bridge", "get"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["document"]["sections"]


def test_cli_set_exits_1_and_prints_errors_on_a_bad_value(capsys):
    from away.cli import main

    assert main(["settings-bridge", "set", "--json", '{"voice.mode": "nope"}']) == 1
    out = json.loads(capsys.readouterr().out)
    assert not out["ok"] and "voice.mode" in out["errors"]
    assert main(["settings-bridge", "set", "--json", "{not json"]) == 1
    assert "bad JSON" in json.loads(capsys.readouterr().out)["errors"][""]


def test_serve_answers_each_line_with_its_id_and_survives_bad_json():
    lines = [
        json.dumps({"id": 1, "op": "get"}),
        "this is not json",
        json.dumps({"id": 2, "op": "set", "values": {"voice.mode": "toggle"}}),
        json.dumps({"id": 3, "op": "status"}),
        json.dumps({"id": 4, "op": "explode"}),
    ]
    buf = io.StringIO()
    assert sb.serve(lines, buf.write) == 0
    out = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert [o["id"] for o in out] == [1, None, 2, 3, 4]
    assert out[0]["ok"] and out[2]["ok"] and out[3]["permissions"]
    assert not out[1]["ok"] and not out[4]["ok"]


def test_serve_keeps_stdout_pure_json_when_a_setter_prints(monkeypatch, capsys):
    from away.app.settings_gui import SettingsModel

    real = SettingsModel.set_voice

    def chatty(self, path, value):
        print("noise from a setter")
        return real(self, path, value)

    monkeypatch.setattr(SettingsModel, "set_voice", chatty)
    buf = io.StringIO()
    sb.serve([json.dumps({"id": 1, "op": "set", "values": {"voice.mode": "toggle"}})], buf.write)
    assert json.loads(buf.getvalue())["ok"]
    assert "noise" in capsys.readouterr().err


# ------------------------------------------------------------------------------------- native vs Tk launch
def _fake_app(tmp_path: Path) -> Path:
    app = tmp_path / "Project Away.app" / "Contents" / "Helpers" / sn.APP_NAME
    (app / "Contents" / "MacOS").mkdir(parents=True)
    with open(app / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleExecutable": "ProjectAwaySettings"}, fh)
    (app / "Contents" / "MacOS" / "ProjectAwaySettings").write_text("#!/bin/sh\n")
    return app


def test_frozen_app_finds_the_helper_inside_its_bundle(tmp_path):
    app = _fake_app(tmp_path)
    exe = str(tmp_path / "Project Away.app" / "Contents" / "MacOS" / "ProjectAway")
    assert sn.find_app(env={}, platform="darwin", executable=exe, frozen=True) == app
    assert sn.find_app(env={sn.ENV: "0"}, platform="darwin", executable=exe, frozen=True) is None
    assert sn.find_app(env={}, platform="win32", executable=exe, frozen=True) is None
    assert sn.find_app(env={}, platform="linux", executable=exe, frozen=True) is None


@pytest.mark.parametrize(
    ("tab", "section"),
    [
        (None, "general"),
        ("Getting started", "general"),
        ("Voice & input", "voice"),
        ("voice", "voice"),
        ("Cameras", "cameras"),
        ("Models", "models"),
        ("permissions", "permissions"),
        ("studio", None),
        ("Gesture Studio", None),
        ("Bindings", None),
        ("Shortcuts", None),
        ("agent link", None),  # alias of Agents (Tk only)
        ("Tuning", None),
        ("Performance", None),
    ],
)
def test_tab_maps_to_a_native_section_or_falls_back_to_tk(tab, section):
    assert sn.section_for_tab(tab) == section


def test_launch_starts_the_native_app_with_the_section_and_bridge(tmp_path):
    app = _fake_app(tmp_path)
    calls: list[list[str]] = []
    ok = sn.launch(
        "Voice & input", None, popen=lambda argv, **kw: calls.append(argv), env={sn.ENV: str(app)}, platform="darwin"
    )
    assert ok
    argv = calls[0]
    assert argv[0].endswith("ProjectAwaySettings")
    assert argv[argv.index("--section") + 1] == "voice"
    bridge = json.loads(argv[argv.index("--bridge") + 1])
    assert bridge[-2:] == ["settings-bridge", "serve"]


def test_launch_returns_false_for_tk_only_tabs_and_on_other_platforms(tmp_path):
    app = _fake_app(tmp_path)
    calls: list[Any] = []
    env = {sn.ENV: str(app)}
    assert not sn.launch("studio", popen=lambda *a, **k: calls.append(a), env=env, platform="darwin")
    assert not sn.launch(None, popen=lambda *a, **k: calls.append(a), env=env, platform="win32")
    assert calls == []


def test_launch_never_raises_when_the_app_cannot_start(tmp_path):
    app = _fake_app(tmp_path)

    def boom(*a: Any, **k: Any) -> None:
        raise OSError("no exec")

    assert sn.launch(None, popen=boom, env={sn.ENV: str(app)}, platform="darwin") is False


def test_away_settings_opens_tk_when_native_says_no(monkeypatch):
    from away import cli
    from away.app import settings_gui

    seen: dict[str, Any] = {}
    monkeypatch.setattr(sn, "launch", lambda tab, config: False)
    monkeypatch.setattr(settings_gui, "main", lambda **kw: seen.update(kw) or 0)
    assert cli.main(["settings", "--tab", "studio"]) == 0
    assert seen["tab"] == "studio"


def test_away_settings_tk_flag_and_teach_gesture_skip_the_native_app(monkeypatch):
    from away import cli
    from away.app import settings_gui

    monkeypatch.setattr(sn, "launch", lambda tab, config: pytest.fail("native app must not open"))
    monkeypatch.setattr(settings_gui, "main", lambda **kw: 0)
    assert cli.main(["settings", "--tk"]) == 0
    assert cli.main(["settings", "--teach-gesture"]) == 0


def test_away_settings_returns_at_once_when_the_native_app_opened(monkeypatch):
    from away import cli
    from away.app import settings_gui

    monkeypatch.setattr(sn, "launch", lambda tab, config: True)
    monkeypatch.setattr(settings_gui, "main", lambda **kw: pytest.fail("Tk must not open"))
    assert cli.main(["settings", "--tab", "voice"]) == 0


def test_spec_copies_the_native_app_into_contents_helpers():
    spec = (Path(__file__).resolve().parents[1] / "packaging" / "pyinstaller" / "away.spec").read_text("utf-8")
    assert '"Contents" / "Helpers"' in spec and "Project Away Settings.app" in spec


def test_release_workflow_builds_the_native_app_before_pyinstaller():
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml").read_text("utf-8")
    mac = wf[wf.index("\n  macos:") :]
    assert mac.index("build_settings_app.sh") < mac.index("pyinstaller packaging/pyinstaller/away.spec")
