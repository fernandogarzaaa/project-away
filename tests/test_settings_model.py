"""Settings editor logic, tested without any widgets (the GUI is a thin view over SettingsModel)."""

import pytest

from away import config as C
from away.app.settings_gui import SettingsModel, Tunable, camera_available
from away.app.tray import TrayController, icon_state
from away.bindings import Action
from away.hands.gestures import GestureEvent


@pytest.fixture
def model(tmp_path):
    return SettingsModel(C.AppConfig(), tmp_path / "config.yaml")


def test_rows_and_dropdown_choices(model):
    rows = {r.trigger: r for r in model.rows()}
    assert rows["gesture:pinch"].kind == "mouse" and rows["gesture:pinch"].arg == "click"
    assert rows["hotkey:ptt"].action == "dictation:toggle"
    assert "shell" in model.action_kinds() and "none" in model.action_kinds()
    # continuous mouse ops are only offered on their own gesture
    assert "drag" in model.arg_choices("mouse", "gesture:pinch_hold")
    assert "drag" not in model.arg_choices("mouse", "gesture:pinch")
    assert "scroll" not in model.arg_choices("mouse", "voice:scroll")
    assert model.arg_choices("key") is None and model.arg_choices("mic") == ["toggle", "on", "off"]
    assert model.arg_choices("nope") is None


def test_set_binding_validation_and_dirty(model):
    assert not model.dirty
    assert model.set_binding("gesture:swipe_left", "key", "alt+left") is None
    assert model.dirty
    assert model.bindings.resolve("gestures", "swipe_left") == Action("key", "alt+left")
    assert "unknown key" in model.set_binding("gesture:swipe_left", "key", "alt+nope")
    assert "disabled for safety" in model.set_binding("gesture:pinch", "shell", "ls")
    model.set_allow_shell(True)
    assert model.set_binding("gesture:pinch", "shell", "ls") is None
    assert "continuous" in model.set_binding("gesture:pinch", "mouse", "drag")
    assert model.set_binding("gesture:pinch", "none") is None
    assert model.bindings.resolve("gestures", "pinch").kind == "none"
    assert model.validate_action_text("mouse:click") is None and model.validate_action_text("x") is not None


def test_add_and_remove_triggers(model):
    assert model.add_trigger("voice", "Next Tab", "key", "ctrl+tab") is None
    assert model.add_trigger("hotkeys", "ctrl+alt+n", "launch", "notepad") is None
    assert "unknown key" in model.add_trigger("hotkeys", "ctrl+alt+nope", "none")
    rows = {r.trigger for r in model.rows()}
    assert "voice:next tab" in rows and "hotkey:ctrl+alt+n" in rows
    assert model.remove_binding("voice:next tab") is None
    assert "voice:next tab" not in {r.trigger for r in model.rows()}


def test_profiles_in_model(model):
    assert model.new_profile("chrome", "chrome, chrome.exe") is None
    assert model.profile == "chrome" and model.profile_apps() == ["chrome", "chrome.exe"]
    assert model.set_binding("gesture:two_finger_scroll", "key", "ctrl+tab") is None
    rows = {r.trigger: r for r in model.rows()}
    assert rows["gesture:two_finger_scroll"].source == "chrome" and rows["gesture:pinch"].source == "default"
    assert "already exists" in model.new_profile("chrome")
    model.new_profile("edge", "msedge")
    assert "already claimed" in model.set_profile_apps("chrome")
    assert model.profile_apps() == ["msedge"]
    model.select_profile("default")
    assert "every app" in model.set_profile_apps("x")
    assert model.set_active_profile("chrome") is None and model.bindings.active_profile == "chrome"
    assert model.set_active_profile("nope") is not None
    assert model.active_profile_choices() == ["auto", "default", "chrome", "edge"]
    with pytest.raises(ValueError):
        model.select_profile("nope")
    model.select_profile("chrome")
    model.reset_profile()
    assert model.bindings.resolve("gestures", "two_finger_scroll", "chrome") == Action("mouse", "scroll")


def test_tunables_sliders(model):
    tun = {t.path: t for t in model.tunables()}
    for path in (
        "hands.cursor.gain",
        "hands.cursor.dead_zone",
        "hands.cursor.acceleration",
        "hands.cursor.min_cutoff",
        "hands.cursor.beta",
        "hands.gestures.pinch_on",
        "hands.gestures.scroll_gain",
        "hands.gestures.swipe_min_dx",
        "hands.gestures.swipe_max_time",
        "voice.vad_sensitivity",
    ):
        t = tun[path]
        assert t.lo <= t.value <= t.hi
        # the slider step never snaps the shipped default
        assert abs(round(t.value / t.resolution) * t.resolution - t.value) < 1e-9
    assert model.set_tunable("hands.cursor.gain", 3.5) is None and model.cfg.hands.cursor.gain == 3.5
    assert "between" in model.set_tunable("hands.cursor.gain", 99)
    assert "pinch_on must be < pinch_off" in model.set_tunable("hands.gestures.pinch_on", 0.9)
    assert model.cfg.hands.gestures.pinch_on == 0.32  # rolled back
    assert model.set_tunable("voice.end_silence_ms", 812.6) is None and model.cfg.voice.end_silence_ms == 813
    assert "unknown setting" in model.set_tunable("hands.nope", 1)
    assert Tunable("x", "x", 0, 10, 1, True).resolution == 1.0


def test_voice_settings(model):
    assert model.voice_value("voice.mode") == "always_on"
    assert model.set_voice("voice.mode", "toggle") is None
    assert "must be one of" in model.set_voice("voice.mode", "loud")
    assert model.set_voice("voice.wake_word", "  jarvis ") is None and model.cfg.voice.wake_word == "jarvis"
    assert model.set_voice("voice.wake_word_enabled", True) is None
    assert "must not be empty" in model.set_voice("voice.wake_word", "")
    assert model.cfg.voice.wake_word == "jarvis"
    assert "unknown key" in model.set_voice("voice.mute_hotkey", "ctrl+nope")
    assert model.set_voice("voice.mute_hotkey", "ctrl+shift+m") is None
    assert model.set_voice("app.always_on", False) is None and model.cfg.app.always_on is False
    assert "unknown setting" in model.set_voice("voice.asr_model", "tiny")


def test_save_revert_reset_import_export(model, tmp_path):
    model.set_binding("gesture:swipe_left", "key", "alt+left")
    model.set_voice("voice.mode", "push_to_talk")
    p = model.save()
    assert not model.dirty
    loaded = C.load(p)[0]
    assert loaded.voice.mode == "push_to_talk"
    assert loaded.bindings.resolve("gestures", "swipe_left") == Action("key", "alt+left")
    model.set_binding("gesture:swipe_left", "none")
    model.revert()
    assert model.bindings.resolve("gestures", "swipe_left") == Action("key", "alt+left") and not model.dirty
    exp = model.export_bindings(tmp_path / "b.yaml")
    model.reset_all()
    assert model.cfg.voice.mode == "always_on" and model.bindings.resolve("gestures", "swipe_left") == Action(
        "key", "left"
    )
    assert model.import_bindings(exp) is None
    assert model.bindings.resolve("gestures", "swipe_left") == Action("key", "alt+left")
    bad = tmp_path / "bad.yaml"
    bad.write_text("profiles: {default: {face: {wink: none}}}\n")
    assert "unknown face trigger 'wink'" in model.import_bindings(bad)
    assert "cannot import" in model.import_bindings(tmp_path / "missing.yaml")


def test_model_loads_from_path_and_reports_invalid(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("voice: {mode: toggle}\n")
    assert SettingsModel(path=p).voice_value("voice.mode") == "toggle"
    p.write_text("voice: {mode: nope}\n")
    with pytest.raises(ValueError):
        SettingsModel(path=p)


def test_live_test_indicator_text(model):
    assert model.describe_event(GestureEvent("click", 0.0)) == "pinch  ->  mouse:click"
    assert model.describe_event(GestureEvent("move", 0.0, {"anchor": "pinch"})) == "pinch_hold  ->  mouse:drag"
    assert model.describe_event(GestureEvent("hand_lost", 0.0)) == "no hand"
    assert model.describe_event(GestureEvent("weird", 0.0)) is None
    model.new_profile("chrome", "chrome")
    model.set_binding("gesture:swipe_left", "key", "ctrl+shift+tab")
    assert model.describe_event(GestureEvent("swipe_left", 0.0)) == "swipe_left  ->  key:ctrl+shift+tab"


def test_camera_available_reports_reason():
    ok, msg = camera_available()
    assert isinstance(ok, bool) and msg  # in CI: missing model or package -> a reason, never an exception
    if not ok:
        assert "away setup" in msg or "not installed" in msg


class VoiceApp:
    def __init__(self):
        from away.config import AppConfig

        self.cfg = AppConfig()
        self.state = "listening"
        self.paused_calls = []
        self.dictation = None

    def voice_state(self):
        return self.state

    def set_mic(self, on):
        self.state = "listening" if on else "muted"

    def set_paused(self, p):
        self.paused_calls.append(p)

    def toggle_dictation(self):
        self.state = "stopped" if self.state == "listening" else "listening"

    def stop(self):
        pass


def test_tray_mic_toggle_and_listening_state():
    app = VoiceApp()
    t = TrayController(app)
    assert t.mic_on() and t.title() == "Project Away - active, listening" and t.icon_state() == "active"
    assert t.toggle_mic() is False and app.state == "muted"
    assert t.title() == "Project Away - active, mic off" and t.icon_state() == "muted"
    assert t.toggle_mic() is True
    t.toggle_dictation()
    assert t.icon_state() == "stopped" and "not listening" in t.title()
    t.toggle_pause()
    assert t.icon_state() == "paused" and t.title().startswith("Project Away - paused")
    cmd = t.settings_command()
    assert cmd[-1] == "settings" and cmd[1:3] == ["-m", "away"]


def test_tray_without_voice_and_icon_states(monkeypatch):
    class Bare:
        cfg = None

    t = TrayController(Bare())
    assert t.voice_state() == "off" and not t.mic_on() and t.toggle_mic() is False
    assert icon_state(False, "recording") == "recording" and icon_state(False, "idle") == "active"
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert t.settings_command() == [sys.executable, "settings"]


def test_tray_icon_colours():
    pytest.importorskip("PIL")
    from away.app.tray import STATE_COLORS, make_icon_image

    for state, colour in STATE_COLORS.items():
        assert make_icon_image(state=state).getpixel((32, 6)) == colour


def test_cameras_tab_model(tmp_path):
    from away.app.settings_gui import SettingsModel
    from away.config import AppConfig
    from away.multicam.capture import CameraDevice

    m = SettingsModel(AppConfig(), tmp_path / "c.yaml")
    assert m.camera_rows() == [] and m.camera_mode() == "single camera"
    devs = [CameraDevice(0, "/dev/video0", "Integrated Webcam"), CameraDevice(2, "/dev/video2", "Logitech BRIO")]
    choices = m.device_choices(devs)
    assert choices == ["0: Integrated Webcam", "2: Logitech BRIO"]
    assert m.parse_device(choices[1]) == 2 and m.parse_device("/dev/video4: x") == "/dev/video4"
    assert m.add_camera(m.parse_device(choices[1])) is None
    assert [(r["device"], r["role"]) for r in m.camera_rows()] == [(0, "primary"), (2, "secondary")]
    assert m.camera_mode().startswith("2 cameras")
    assert m.add_camera(2) is not None  # same device twice
    assert m.set_camera(1, "width", "1280") is None and m.cfg.hands.cameras[1].width == 1280
    assert m.set_camera(1, "fps", "abc") is not None
    assert m.set_camera(1, "fps", 1) is not None and m.cfg.hands.cameras[1].fps == 30  # rejected, unchanged
    assert m.set_camera(1, "role", "primary") is None
    assert [c.role for c in m.cfg.hands.cameras] == ["secondary", "primary"]  # promotion demotes the old one
    assert m.set_camera(5, "fps", 30) == "no such camera" and m.set_camera(0, "bogus", 1) is not None
    assert m.set_camera(0, "device", "4: Other") is None and m.cfg.hands.cameras[0].device == 4
    assert m.set_camera(0, "hfov_deg", "85") is None and m.cfg.hands.cameras[0].hfov_deg == 85.0
    m.save()
    assert C.load(tmp_path / "c.yaml")[0].hands.cameras == m.cfg.hands.cameras
    assert m.remove_camera(1) is None and m.cfg.hands.cameras == [] and m.cfg.hands.camera_index == 4
    assert "not calibrated" in m.calibration_status()


def test_performance_tab_model(monkeypatch, fixed_hardware):
    from away.app import frozen, gpupack, hardware
    from away.app.settings_gui import SettingsModel

    model = SettingsModel(C.AppConfig())
    info = model.performance_info()
    assert info["mode"] == "auto" and info["profile"].asr_model == "small"
    assert any(line.startswith("Profile (cpu-mid)") for line in info["lines"])
    assert info["pack_status"].startswith("no NVIDIA GPU") and not info["can_install"]
    assert model.set_voice("performance.mode", "gpu") is None and model.cfg.performance.mode == "gpu"
    assert "must be one of" in model.set_voice("performance.mode", "turbo")
    # an RTX 2060 in the frozen app without the pack: the download button is offered
    g = hardware.GPU("nvidia", "NVIDIA GeForce RTX 2060", 6144, 5000, (7, 5), "560.94", "12.6", "nvml")
    rtx = hardware.Hardware(os="win32", arch="amd64", cores_physical=6, ram_mb=16384, gpus=[g])
    monkeypatch.setattr(frozen, "is_frozen", lambda: True)
    monkeypatch.setattr(gpupack, "platform_tag", lambda *a: "win_amd64")
    model.set_voice("performance.mode", "auto")
    info = model.performance_info(rtx)
    assert info["can_install"] and "not installed" in info["pack_status"]
    monkeypatch.setattr(gpupack, "install", lambda **k: (_ for _ in ()).throw(OSError("offline")))
    assert "resume" in SettingsModel.install_gpu_pack(lambda s: None)
    monkeypatch.setattr(gpupack, "install", lambda **k: None)
    assert SettingsModel.install_gpu_pack(lambda s: None) is None
