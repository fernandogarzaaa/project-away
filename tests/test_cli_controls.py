import json

import numpy as np
import pytest
import yaml

from away import config as C
from away.bindings import Action
from away.cli import main, register_custom_gesture
from away.hands.recorder import check_label, save_clip


@pytest.fixture
def cfgfile(tmp_path, monkeypatch):
    p = tmp_path / "cfg" / "config.yaml"
    monkeypatch.setenv("AWAY_CONFIG", str(p))
    return p


def run(capsys, *argv):
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_bindings_set_list_json(cfgfile, capsys):
    assert run(capsys, "bindings", "set", "gesture:swipe_left", "key:alt+left")[0] == 0
    rc, out, _ = run(capsys, "bindings", "list", "--json")
    rows = {r["trigger"]: r["action"] for r in json.loads(out)}
    assert rc == 0 and rows["gesture:swipe_left"] == "key:alt+left" and rows["hotkey:mute"] == "mic:toggle"
    rc, out, _ = run(capsys, "bindings", "list")
    assert "gesture:swipe_left" in out and "key:alt+left" in out
    assert C.load(cfgfile)[0].bindings.resolve("gestures", "swipe_left") == Action("key", "alt+left")


@pytest.mark.parametrize(
    "argv,msg",
    [
        (("bindings", "set", "gesture:pinchh", "mouse:click"), "did you mean 'pinch'"),
        (("bindings", "set", "gesture:pinch", "shell:ls"), "disabled for safety"),
        (("bindings", "set", "gesture:pinch"), "usage: away bindings set"),
        (("bindings", "list", "--profile", "nope"), "no profile named 'nope'"),
        (("profile", "use", "nope"), "no profile named 'nope'"),
        (("profile", "new", "bad name"), "must be letters"),
        (("profile", "remove", "default"), "cannot remove"),
        (("voice", "mode", "ambient"), "voice.mode must be always_on|push_to_talk|toggle"),
        (("voice", "sensitivity", "3"), "vad_sensitivity"),
        (("bindings", "import"), "usage: away bindings import"),
    ],
)
def test_cli_errors_exit_2_with_message(cfgfile, capsys, argv, msg):
    rc, _out, err = run(capsys, *argv)
    assert rc == 2 and msg in err


def test_profiles_cli(cfgfile, capsys):
    assert run(capsys, "profile", "new", "chrome", "--apps", "chrome", "chrome.exe")[0] == 0
    assert run(capsys, "bindings", "set", "gesture:two_finger_scroll", "key:ctrl+tab", "--profile", "chrome")[0] == 0
    assert run(capsys, "profile", "new", "chrome2", "--copy-from", "chrome", "--apps", "chromium")[0] == 0
    rc, out, _ = run(capsys, "profile", "list")
    assert "chrome: chrome, chrome.exe" in out and "active_profile: auto" in out
    assert run(capsys, "profile", "use", "chrome")[0] == 0
    cfg = C.load(cfgfile)[0]
    assert cfg.bindings.active_profile == "chrome"
    assert cfg.bindings.resolve("gestures", "two_finger_scroll", "chrome2") == Action("key", "ctrl+tab")
    assert run(capsys, "profile", "remove", "chrome")[0] == 0
    assert C.load(cfgfile)[0].bindings.active_profile == "auto"


def test_export_import_round_trip(cfgfile, tmp_path, capsys):
    run(capsys, "profile", "new", "code", "--apps", "Code.exe")
    run(capsys, "bindings", "set", "voice:Format Document", "key:shift+alt+f", "--profile", "code")
    run(capsys, "bindings", "set", "hotkey:ctrl+alt+j", "type:Hi!", "--profile", "code")
    exported = tmp_path / "mine.yaml"
    assert run(capsys, "bindings", "export", str(exported))[0] == 0
    before = C.load(cfgfile)[0].bindings.to_dict()
    assert yaml.safe_load(exported.read_text())["bindings"] == before
    run(capsys, "bindings", "reset", "--all")
    run(capsys, "profile", "remove", "code")
    assert "code" not in C.load(cfgfile)[0].bindings.profiles
    assert run(capsys, "bindings", "import", str(exported))[0] == 0
    assert C.load(cfgfile)[0].bindings.to_dict() == before
    # stdout export
    rc, out, _ = run(capsys, "bindings", "export", "-")
    assert rc == 0 and yaml.safe_load(out)["bindings"] == before


def test_import_merge_vs_replace_and_invalid_file(cfgfile, tmp_path, capsys):
    run(capsys, "bindings", "set", "gesture:swipe_left", "key:alt+left")
    part = tmp_path / "part.yaml"
    part.write_text("profiles:\n  chrome:\n    apps: [chrome]\n    gestures:\n      pinch: key:ctrl+t\n")
    assert run(capsys, "bindings", "import", str(part))[0] == 0
    b = C.load(cfgfile)[0].bindings
    assert b.resolve("gestures", "swipe_left") == Action("key", "alt+left")  # merge kept it
    assert b.resolve("gestures", "pinch", "chrome") == Action("key", "ctrl+t")
    assert run(capsys, "bindings", "import", str(part), "--replace")[0] == 0
    assert C.load(cfgfile)[0].bindings.resolve("gestures", "swipe_left") == Action("key", "left")
    bad = tmp_path / "bad.yaml"
    bad.write_text("bindings:\n  profiles:\n    default:\n      gestures:\n        pinch: warp:9\n")
    rc, _, err = run(capsys, "bindings", "import", str(bad))
    assert rc == 2 and "bindings.profiles.default.gestures.pinch: unknown action 'warp:9'" in err
    rc, _, err = run(capsys, "bindings", "import", str(tmp_path / "missing.yaml"))
    assert rc == 2 and "cannot read" in err


def test_bindings_reset_one_trigger_and_profile(cfgfile, capsys):
    run(capsys, "bindings", "set", "gesture:pinch", "mouse:right_click")
    run(capsys, "bindings", "reset", "gesture:pinch")
    assert C.load(cfgfile)[0].bindings.resolve("gestures", "pinch") == Action("mouse", "click")
    run(capsys, "profile", "new", "x", "--apps", "x")
    run(capsys, "bindings", "set", "gesture:pinch", "none", "--profile", "x")
    run(capsys, "bindings", "reset", "--profile", "x")
    b = C.load(cfgfile)[0].bindings
    assert b.profiles["x"].apps == ["x"] and b.resolve("gestures", "pinch", "x") == Action("mouse", "click")


def test_voice_cli(cfgfile, capsys):
    assert run(capsys, "voice", "mode")[1].strip() == "always_on"
    assert run(capsys, "voice", "mode", "push_to_talk")[0] == 0
    assert run(capsys, "voice", "wake-word", "Jarvis")[0] == 0
    assert run(capsys, "voice", "intent", "command")[0] == 0
    assert run(capsys, "voice", "sensitivity", "0.8")[0] == 0
    v = C.load(cfgfile)[0].voice
    assert (v.mode, v.wake_word_enabled, v.wake_word, v.intent, v.vad_sensitivity) == (
        "push_to_talk",
        True,
        "Jarvis",
        "command",
        0.8,
    )
    assert run(capsys, "voice", "wake-word")[1].strip() == "Jarvis"
    run(capsys, "voice", "wake-word", "off")
    assert run(capsys, "voice", "wake-word")[1].strip() == "off"
    rc, out, _ = run(capsys, "voice", "show")
    assert rc == 0 and "voice.mode: push_to_talk" in out and "app.always_on: True" in out


def test_invalid_existing_config_is_reported(cfgfile, capsys):
    cfgfile.parent.mkdir(parents=True)
    cfgfile.write_text("voice:\n  mode: loud\n")
    rc, _, err = run(capsys, "bindings", "list")
    assert rc == 2 and "voice.mode" in err


def test_custom_gesture_labels_and_registration(cfgfile, tmp_path):
    assert check_label("swipe_left") == "swipe_left"
    with pytest.raises(ValueError, match="--custom"):
        check_label("wave")
    assert check_label("wave", allow_custom=True) == "wave"
    with pytest.raises(ValueError, match="lower_snake_case"):
        check_label("Wave Hi", allow_custom=True)
    frames = [np.zeros((21, 3)) for _ in range(6)]
    p = save_clip(tmp_path / "rec", "wave", frames, "Right", 30.0, allow_custom=True)
    assert p.parent.name == "wave"
    assert register_custom_gesture("wave", cfgfile) is True
    assert register_custom_gesture("wave", cfgfile) is False
    assert register_custom_gesture("pinch_click", cfgfile) is False  # built-in label
    assert C.load(cfgfile)[0].bindings.custom_gestures == ["wave"]
    assert main(["bindings", "set", "gesture:wave", "key:ctrl+w"]) == 0


def test_temporal_classes_include_custom_recordings():
    from away.hands.synthetic import GESTURES
    from away.hands.train import temporal_classes

    assert temporal_classes(["swipe_left", "wave", "Bad Label", "clap", "wave"]) == [*GESTURES, "clap", "wave"]


def test_settings_command_without_tkinter(cfgfile, capsys, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "tkinter", None)  # Python built without Tk
    rc, _, err = run(capsys, "settings")
    assert rc == 1 and "import of tkinter halted" in err and "away voice mode" in err


def test_settings_command_headless(cfgfile, capsys, monkeypatch):
    tkinter = pytest.importorskip("tkinter", exc_type=ImportError)  # libtk may be missing

    def no_display():
        raise tkinter.TclError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(tkinter, "Tk", no_display)
    rc, _, err = run(capsys, "settings")
    assert rc == 1 and "needs a desktop session" in err and "away bindings list" in err
