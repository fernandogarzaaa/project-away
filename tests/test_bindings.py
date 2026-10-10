import pytest

from away import config as C
from away.bindings import (
    NONE,
    Action,
    BindingsConfig,
    BindingsError,
    app_key,
    default_profile,
    list_rows,
    merge_bindings,
    normalize_phrase,
    parse_action,
    parse_bindings,
    split_trigger,
)
from away.control.keys import ComboCapture, KeyComboError, canonical, keysym_to_key, parse_combo, to_pynput_hotkey

# ----------------------------------------------------------------------------- keys


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ctrl+tab", ["ctrl", "tab"]),
        ("Tab+Ctrl", ["ctrl", "tab"]),
        ("<ctrl>+<alt>+space", ["ctrl", "alt", "space"]),
        ("shift+alt+ctrl+T", ["ctrl", "alt", "shift", "t"]),
        ("Control+PgUp", ["ctrl", "page_up"]),
        ("win+d", ["cmd", "d"]),
        ("ctrl++", ["ctrl", "+"]),
        ("F5", ["f5"]),
        ("ctrl_r", ["ctrl_r"]),
        ("Return", ["enter"]),
    ],
)
def test_parse_combo(text, expected):
    assert parse_combo(text) == expected


@pytest.mark.parametrize(
    "bad,msg",
    [
        ("", "empty"),
        ("ctrl+", "empty key"),
        ("ctrl+banana", "unknown key 'banana'"),
        ("a+b", "more than one non-modifier"),
        ("ctrl+ctrl", "repeated"),
        ("f25", "unknown key"),
    ],
)
def test_parse_combo_errors(bad, msg):
    with pytest.raises(KeyComboError, match=msg):
        parse_combo(bad)


def test_pynput_format_and_canonical():
    assert to_pynput_hotkey("ctrl+alt+space") == "<ctrl>+<alt>+<space>"
    assert to_pynput_hotkey("ctrl+shift+k") == "<ctrl>+<shift>+k"
    assert canonical("<ctrl>+<alt>+m") == "ctrl+alt+m"


def test_combo_capture_from_tk_keysyms():
    c = ComboCapture()
    assert c.press("Control_L") is None
    assert c.press("Shift_L") is None
    assert c.press("T") == "ctrl+shift+t"
    assert c.press("x") == "ctrl+shift+t"  # already captured
    c.reset()
    c.press("Control_R")
    assert c.release("Control_R") == "ctrl"  # a lone modifier tap
    c.reset()
    assert c.press("Prior") == "page_up"
    assert keysym_to_key("F12") == "f12" and keysym_to_key("XF86Unknown") is None
    c.reset()
    assert c.press("Alt_L") is None and c.press("Left") == "alt+left"


# ------------------------------------------------------------------------- actions


def test_parse_action_forms():
    assert parse_action("key:Ctrl+Tab") == Action("key", "ctrl+tab")
    assert parse_action({"key": "alt+left"}) == Action("key", "alt+left")
    assert parse_action("none") is NONE and parse_action(None) is NONE
    assert parse_action("type:Hello, World") == Action("type", "Hello, World")
    assert parse_action("launch:https://example.com") == Action("launch", "https://example.com")
    assert str(parse_action("mouse:RIGHT_CLICK")) == "mouse:right_click"
    assert parse_action("shell:echo hi", allow_shell=True) == Action("shell", "echo hi")


@pytest.mark.parametrize(
    "bad,msg",
    [
        ("clik", "unknown action"),
        ("mouse:triple_click", "mouse:click|double_click"),
        ("mouse", "needs an argument"),
        ("key:ctrl+nope", "unknown key"),
        ("shell:rm -rf /", "disabled for safety"),
        ("launch:", "needs an argument"),
        (42, "must be a string"),
        ({"a": 1, "b": 2}, "exactly one key"),
    ],
)
def test_parse_action_errors(bad, msg):
    with pytest.raises(ValueError, match=msg):
        parse_action(bad)


def test_unknown_action_suggests():
    with pytest.raises(ValueError, match="did you mean 'mouse'"):
        parse_action("mous:click")


# ------------------------------------------------------------------------- model


def test_defaults_match_previous_behaviour():
    p = default_profile()
    assert p.gestures["pinch"] == Action("mouse", "click")
    assert p.gestures["pinch_hold"] == Action("mouse", "drag")
    assert p.gestures["two_finger_scroll"] == Action("mouse", "scroll")
    assert p.gestures["open_palm"] == Action("pause", "toggle")
    assert p.gestures["swipe_left"] == Action("key", "left")  # was DEFAULT_SWIPE_KEYS
    assert p.gestures["swipe_right"] == Action("key", "right")
    assert p.face["brow_raise"] == Action("dictation", "toggle")
    assert p.hotkeys == {"ptt": Action("dictation", "toggle"), "mute": Action("mic", "toggle")}
    assert p.voice["stop listening"] == Action("dictation", "stop")


def test_normalize_phrase_and_split_trigger():
    assert normalize_phrase("  Scroll, DOWN!! ") == "scroll down"
    assert normalize_phrase("don't stop") == "don't stop"
    assert split_trigger("gesture:pinch") == ("gestures", "pinch")
    assert split_trigger("hotkeys:ctrl+k") == ("hotkeys", "ctrl+k")
    with pytest.raises(ValueError, match="must look like"):
        split_trigger("pinch")


def test_parse_bindings_full_example():
    b = parse_bindings(
        {
            "active_profile": "auto",
            "custom_gestures": ["wave"],
            "profiles": {
                "default": {"gestures": {"swipe_left": "key:alt+left", "wave": "launch:https://x.org"}},
                "chrome": {
                    "apps": ["chrome", "chrome.exe"],
                    "gestures": {"two_finger_scroll": "key:ctrl+tab"},
                    "voice": {"New Tab": "key:ctrl+t"},
                    "hotkeys": {"<ctrl>+<alt>+k": "type:hello"},
                },
            },
        }
    )
    assert b.resolve("gestures", "swipe_left") == Action("key", "alt+left")
    assert b.resolve("gestures", "pinch") == Action("mouse", "click")  # default kept
    assert b.resolve("gestures", "two_finger_scroll", "chrome") == Action("key", "ctrl+tab")
    assert b.resolve("gestures", "pinch", "chrome") == Action("mouse", "click")  # inherits default
    assert b.resolve("voice", "new tab!", "chrome") == Action("key", "ctrl+t")
    assert b.resolve("voice", "new tab", "default") is NONE
    assert b.resolve("hotkeys", "ctrl+alt+k", "chrome") == Action("type", "hello")
    assert b.resolve("gestures", "wave") == Action("launch", "https://x.org")


def test_parse_bindings_collects_every_error_with_paths():
    with pytest.raises(BindingsError) as ei:
        parse_bindings(
            {
                "profils": {},
                "profiles": {
                    "default": {"gestures": {"pinchh": "mouse:click", "swipe_left": "mouse:drag"}, "apps": ["x"]},
                    "auto": {},
                    "chrome": {"apps": "chrome", "voice": {"go": "shell:ls"}, "gesturs": {}},
                    "edge": {"apps": ["msedge"]},
                    "edge2": {"apps": ["MSEDGE.exe"]},
                },
                "active_profile": "nope",
                "custom_gestures": ["Bad Name", "pinch"],
            }
        )
    errs = "\n".join(ei.value.errors)
    assert "bindings.profils: unknown key (did you mean 'profiles'?)" in errs
    assert "bindings.profiles.default.gestures.pinchh: unknown gesture 'pinchh' (did you mean 'pinch'?)" in errs
    assert "mouse:drag is continuous and can only be bound to gesture 'pinch_hold'" in errs
    assert "default.apps" in errs and "bindings.profiles.auto: invalid profile name" in errs
    assert "bindings.profiles.chrome.apps: must be a list" in errs
    assert "chrome.voice.go: shell actions are disabled" in errs
    assert "chrome.gesturs: unknown key (did you mean 'gestures'?)" in errs
    assert "already claimed by profile 'edge'" in errs
    assert "active_profile: 'nope'" in errs
    assert "'Bad Name' must be lower_snake_case" in errs and "'pinch' is already a built-in" in errs
    assert str(ei.value).startswith("invalid bindings:\n  - ")


def test_profile_resolution_by_active_app():
    b = BindingsConfig()
    b.new_profile("chrome", ["chrome", "Google Chrome"])
    b.new_profile("code", ["Code.exe"])
    assert b.profile_for_app("chrome.exe") == "chrome"
    assert b.profile_for_app(r"C:\Program Files\Google\chrome.EXE") == "chrome"
    assert b.profile_for_app("Google Chrome") == "chrome"
    assert b.profile_for_app("code") == "code"
    assert b.profile_for_app("firefox") == "default"
    assert b.profile_for_app(None) == "default"
    b.active_profile = "code"  # explicit pin wins over the focused app
    assert b.profile_for_app("chrome.exe") == "code"
    assert app_key("/usr/bin/firefox") == "firefox"


def test_set_unset_reset_new_profile():
    b = BindingsConfig()
    b.set("gesture:swipe_left", "key:alt+left")
    b.new_profile("chrome", ["chrome"])
    b.set("gesture:pinch", "mouse:right_click", "chrome")
    b.set("hotkey:<ctrl>+<shift>+y", "type:yo", "chrome")
    assert b.resolve("hotkeys", "ctrl+shift+y", "chrome") == Action("type", "yo")
    with pytest.raises(ValueError, match="no profile named 'x'"):
        b.set("gesture:pinch", "none", "x")
    with pytest.raises(ValueError, match="unknown face trigger"):
        b.set("face:wink", "none")
    b.unset("gesture:swipe_left")  # default profile: back to the shipped default
    assert b.resolve("gestures", "swipe_left") == Action("key", "left")
    b.unset("gesture:pinch", "chrome")  # other profile: falls through to default
    assert b.resolve("gestures", "pinch", "chrome") == Action("mouse", "click")
    b.new_profile("c2", ["c2"], copy_from="chrome")
    assert b.resolve("hotkeys", "ctrl+shift+y", "c2") == Action("type", "yo")
    with pytest.raises(ValueError, match="already exists"):
        b.new_profile("c2")
    with pytest.raises(ValueError, match="reserved"):
        b.new_profile("auto")
    with pytest.raises(ValueError, match="must be letters"):
        b.new_profile("bad name")
    b.reset()
    assert b.profiles["chrome"].apps == ["chrome"] and not b.profiles["chrome"].hotkeys


def test_to_dict_round_trip_and_rows():
    b = BindingsConfig(custom_gestures=["wave"])
    b.new_profile("chrome", ["chrome"])
    b.set("gesture:wave", "key:ctrl+w", "chrome")
    b2 = parse_bindings(b.to_dict())
    assert b2.to_dict() == b.to_dict()
    rows = dict((t, (a, s)) for t, a, s in list_rows(b2, "chrome"))
    assert rows["gesture:wave"] == ("key:ctrl+w", "chrome")
    assert rows["gesture:pinch"] == ("mouse:click", "default")
    assert rows["face:mouth_open"] == ("none", "default")


def test_merge_only_takes_explicit_entries():
    cur = BindingsConfig()
    cur.set("gesture:swipe_left", "key:alt+left")
    raw = {"profiles": {"chrome": {"apps": ["chrome"], "gestures": {"pinch": "key:ctrl+t"}}}}
    merged = merge_bindings(cur, parse_bindings(raw), raw)
    assert merged.resolve("gestures", "swipe_left") == Action("key", "alt+left")  # not clobbered by defaults
    assert merged.resolve("gestures", "pinch", "chrome") == Action("key", "ctrl+t")
    raw2 = {"active_profile": "chrome", "profiles": {"default": {"gestures": {"swipe_left": "none"}}}}
    merged2 = merge_bindings(merged, parse_bindings({**raw2, "profiles": {**raw2["profiles"], "chrome": {}}}), raw2)
    assert merged2.resolve("gestures", "swipe_left") is NONE and merged2.active_profile == "chrome"


# ------------------------------------------------------------------------- config


def test_config_round_trip_with_bindings_and_voice():
    data = {
        "voice": {"mode": "toggle", "wake_word_enabled": True, "wake_word": "computer"},
        "app": {"always_on": False},
        "bindings": {"profiles": {"default": {"gestures": {"swipe_left": "key:alt+left"}}}},
    }
    cfg, unknown = C.from_dict(data)
    assert unknown == []
    assert cfg.voice.mode == "toggle" and cfg.app.always_on is False
    cfg2, _ = C.from_dict(C.to_dict(cfg))
    assert C.to_dict(cfg2) == C.to_dict(cfg)
    assert cfg2.bindings.resolve("gestures", "swipe_left") == Action("key", "alt+left")


def test_config_defaults_always_on():
    cfg = C.AppConfig()
    assert cfg.voice.mode == "always_on" and cfg.app.always_on and cfg.app.start_minimized
    assert cfg.bindings.allow_shell is False


def test_legacy_ptt_hold_maps_to_push_to_talk():
    cfg, _ = C.from_dict({"voice": {"ptt_mode": "hold"}})
    assert cfg.voice.mode == "push_to_talk"
    cfg, _ = C.from_dict({"voice": {"ptt_mode": "hold", "mode": "always_on"}})
    assert cfg.voice.mode == "always_on"


@pytest.mark.parametrize(
    "data,msg",
    [
        ({"voice": {"mode": "ambient"}}, "voice.mode must be always_on|push_to_talk|toggle"),
        ({"voice": {"intent": "chat"}}, "voice.intent"),
        ({"voice": {"wake_word_enabled": True, "wake_word": " "}}, "wake_word must not be empty"),
        ({"voice": {"mute_hotkey": "ctrl+nope"}}, "voice.mute_hotkey: unknown key"),
        ({"voice": {"hold_key": "ctrl+t"}}, "voice.hold_key must be a single key"),
        ({"voice": {"vad_sensitivity": 2}}, r"voice.vad_sensitivity must be a number in \[0.0, 1.0\]"),
        ({"hands": {"cursor": {"gain": 0}}}, "hands.cursor.gain"),
        ({"hands": {"cursor": {"dead_zone": "big"}}}, "hands.cursor.dead_zone"),
        ({"hands": {"gestures": {"scroll_gain": True}}}, "scroll_gain"),
        ({"bindings": {"profiles": {"default": {"gestures": {"x": "none"}}}}}, "unknown gesture 'x'"),
        ({"bindings": "nope"}, "bindings: must be a mapping"),
    ],
)
def test_config_validation_messages(data, msg):
    with pytest.raises(ValueError, match=msg):
        C.from_dict(data)


def test_save_validates_and_respects_away_config(tmp_path, monkeypatch):
    target = tmp_path / "x" / "away.yaml"
    monkeypatch.setenv("AWAY_CONFIG", str(target))
    cfg = C.AppConfig()
    cfg.voice.mode = "toggle"
    assert C.save(cfg) == target and C.config_path() == target
    assert C.load()[0].voice.mode == "toggle"
    cfg.voice.mode = "bad"
    with pytest.raises(ValueError):
        C.save(cfg)
