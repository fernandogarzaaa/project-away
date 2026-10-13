import io
import json

import numpy as np
import pytest

from away import config as C
from away import paths
from away.app import autostart, models, setup_wizard, uninstall
from away.app.overlay import status_text
from away.app.runner import Pipeline
from away.app.tray import TrayController
from away.cli import main
from away.config import AppConfig, CursorConfig
from away.control.mouse import MouseController, RecordingBackend
from away.events import Event
from away.hands.cursor import CursorMapper
from away.hands.gestures import GestureEvent
from away.hands.synthetic import pose
from away.wifi.sources import synthetic_csi


def ctl(mode="relative"):
    b = RecordingBackend()
    cfg = CursorConfig(mode=mode, min_cutoff=1e6, d_cutoff=1e6, beta=0.0, dead_zone=0.0, acceleration=1.0, gain=1.0)
    return MouseController(b, CursorMapper((1000, 1000), cfg)), b


def test_mouse_controller_maps_events():
    m, b = ctl()
    m.handle(GestureEvent("move", 0.0, {"x": 0.5, "y": 0.5}))
    m.handle(GestureEvent("move", 0.1, {"x": 0.6, "y": 0.5, "anchor": "index"}))
    m.handle(GestureEvent("click", 0.2))
    m.handle(GestureEvent("double_click", 0.3))
    m.handle(GestureEvent("drag_start", 0.4))
    m.handle(GestureEvent("drag_end", 0.5))
    m.handle(GestureEvent("drag_end", 0.55))  # no second release
    m.handle(GestureEvent("scroll", 0.6, {"dy": 2.5}))
    m.handle(GestureEvent("swipe_left", 0.7))
    m.handle(GestureEvent("swipe_right", 0.8))
    assert b.calls == [
        ("move", (500, 500)),
        ("move", (600, 500)),
        ("click", (1,)),
        ("click", (1,)),
        ("press", ()),
        ("release", ()),
        ("scroll", (2.5,)),
        ("key", ("left",)),
        ("key", ("right",)),
    ]


def test_mouse_controller_clutch_and_hand_lost_release_button():
    m, b = ctl()
    m.handle(GestureEvent("drag_start", 0.0))
    m.handle(GestureEvent("clutch", 0.1, {"paused": True}))
    assert b.calls[-1] == ("release", ()) and not m.enabled
    m.handle(GestureEvent("click", 0.2))
    assert ("click", (1,)) not in b.calls
    m.handle(GestureEvent("clutch", 0.3, {"paused": False}))
    m.handle(GestureEvent("drag_start", 0.4))
    m.handle(GestureEvent("hand_lost", 0.5))
    assert b.calls[-1] == ("release", ())
    assert m.button_down is False
    m2, b2 = ctl()
    m2.swipe_keys = {}
    m2.handle(GestureEvent("swipe_left", 0.0))
    assert b2.calls == []


def test_pipeline_click_and_pause():
    cfg = AppConfig()
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    t = 0.0
    for lm in [pose("pinch_open"), pose("pinch"), pose("pinch_open")]:
        t += 0.05
        p.on_hand(t, lm)
    assert ("click", (1,)) in b.calls
    assert [e.topic for e in p.bus.history() if e.topic == "gesture.click"] == ["gesture.click"]
    n = len(b.calls)
    for _ in range(20):
        t += 0.05
        p.on_hand(t, pose("palm"))
    assert p.paused
    p.on_hand(t + 0.05, pose("point"))
    assert len(b.calls) == n


class FakeTemporal:
    def __init__(self, probs):
        self.probs = probs

    def push(self, lm, hand):
        return self.probs


def test_pipeline_model_veto():
    cfg = AppConfig()
    cfg.hands.gestures.use_model = True
    b = RecordingBackend()
    p = Pipeline(cfg, b, temporal=FakeTemporal({"pinch_click": 0.1}))
    t = 0.0
    for lm in [pose("pinch_open"), pose("pinch"), pose("pinch_open")]:
        t += 0.05
        p.on_hand(t, lm)
    assert ("click", (1,)) not in b.calls
    assert p.bus.history("gesture.rejected")[0].data == {"kind": "click"}
    cfg.hands.gestures.use_model = False
    assert Pipeline(cfg, temporal=FakeTemporal({})).temporal is None


def test_pipeline_faces_enroll_gate_and_brow():
    cfg = AppConfig()
    cfg.face.brow_raise_toggle = True
    p = Pipeline(cfg)
    boxes = [(0.3, 0.5, 0.3), (0.7, 0.5, 0.1)]
    p.mic_energy = 0.0
    p.on_faces(0.0, boxes, [{}, {}])
    assert p.gate.primary == 0
    assert p.bus.history("face.enrolled")[0].data == {"face_id": 0}
    for i in range(1, 30):
        voice = 0.5 + 0.5 * np.sin(i / 2)
        p.mic_energy = voice
        p.on_faces(i * 0.1, boxes, [{"jawOpen": voice * 2.1}, {"jawOpen": 0.3 * (i % 3)}])
    assert p.gate.allow(1.0, 3.0)
    for i in range(30, 60):
        p.on_faces(i * 0.1, boxes, [{"browInnerUp": 1.0}, {}])
    assert len(p.bus.history("face.brow_toggle")) == 1
    p.re_enroll()
    p.on_faces(9.0, [(0.7, 0.5, 0.1)], [{}])
    assert p.gate.primary == 1


def test_pipeline_wifi_window():
    cfg = AppConfig()
    cfg.wifi.enabled = True
    p = Pipeline(cfg)
    _, H = synthetic_csi("swipe", seed=0)
    label, conf = p.on_csi_window(5.0, H, None, 100.0)
    assert label == "swipe"
    assert p.fusion.csi.label == "swipe" and p.fusion.csi.t == 5.0
    assert p.bus.history("wifi.state")[0].data["label"] == "swipe"


# ---------------------------------------------------------------- models / setup
class FakeHTTP(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_verifies_checksum(tmp_path):
    payload = b"model-bytes"
    import hashlib

    mf = models.ModelFile("m.task", "http://x/m.task", hashlib.sha256(payload).hexdigest(), 0.1, "t")
    msgs = []
    p = models.download(mf, tmp_path, msgs.append, opener=lambda u: FakeHTTP(payload))
    assert p.read_bytes() == payload
    assert models.verify(p, mf.sha256)
    assert "downloading" in msgs[0]
    models.download(mf, tmp_path, msgs.append, opener=lambda u: pytest.fail("should not refetch"))
    assert "already present" in msgs[-1]
    bad = models.ModelFile("b.task", "http://x/b", "0" * 64, 0.1, "t")
    with pytest.raises(RuntimeError, match="checksum"):
        models.download(bad, tmp_path, msgs.append, opener=lambda u: FakeHTTP(payload))
    assert not (tmp_path / "b.task").exists()
    assert not (tmp_path / "b.task.part").exists()
    assert models.model_path(tmp_path, "m.task") == p
    assert models.model_path(tmp_path, "zzz") is None
    assert not models.verify(tmp_path / "nope", None)
    nohash = models.ModelFile("n.task", "u", None, 0.1, "t")
    assert models.download(nohash, tmp_path, msgs.append, opener=lambda u: FakeHTTP(b"x")).exists()


def test_mediapipe_registry_is_pinned():
    names = {m.name for m in models.MEDIAPIPE_MODELS}
    assert names == {"hand_landmarker.task", "face_landmarker.task"}
    assert all(m.sha256 and len(m.sha256) == 64 and m.url.startswith("https://") for m in models.MEDIAPIPE_MODELS)


def test_setup_wizard_noninteractive(tmp_path):
    fetched = []

    def fake_dl(mf, d, out):
        fetched.append(mf.name)
        return d / mf.name

    out = []
    rep = setup_wizard.run(yes=True, skip_asr=True, skip_probe=True, out=out.append, downloader=fake_dl)
    assert fetched == ["hand_landmarker.task", "face_landmarker.task"]
    assert rep.models["asr"] == "skipped"
    assert rep.probes == {"camera": "skipped", "microphone": "skipped"}
    cfg, _ = C.load()
    assert cfg.models_dir == str(paths.models_dir())
    assert rep.config_path == str(paths.config_file())


def test_setup_wizard_reports_failures(monkeypatch):
    def bad_dl(mf, d, out):
        raise OSError("offline")

    monkeypatch.setattr(models, "download_asr", lambda *a, **k: (_ for _ in ()).throw(OSError("hf down")))
    rep = setup_wizard.run(yes=True, skip_probe=True, out=lambda s: None, downloader=bad_dl)
    assert rep.models["hand_landmarker.task"].startswith("failed")
    assert rep.models["asr"].startswith("failed")
    assert len(rep.warnings) == 3


def test_setup_wizard_asr_choice(monkeypatch):
    got = []
    monkeypatch.setattr(models, "download_asr", lambda m, d, out, **k: got.append(m))
    rep = setup_wizard.run(
        yes=True, asr_model="small", skip_probe=True, out=lambda s: None, downloader=lambda mf, d, out: d / mf.name
    )
    assert got == ["small"] and rep.models["asr"] == "ok"
    assert C.load()[0].voice.asr_model == "small"
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    rep2 = setup_wizard.run(skip_probe=True, out=lambda s: None, downloader=lambda mf, d, out: d / mf.name)
    assert rep2.models["asr"].startswith("deferred")


def test_ask():
    assert setup_wizard.ask("q", False, True) is True
    assert setup_wizard.ask("q", True, False) is False


def test_ask_interactive(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda p: "")
    assert setup_wizard.ask("q", True, None) is True
    monkeypatch.setattr("builtins.input", lambda p: "YES")
    assert setup_wizard.ask("q", False, None) is True
    monkeypatch.setattr("builtins.input", lambda p: (_ for _ in ()).throw(EOFError()))
    assert setup_wizard.ask("q", False, None) is False


# ---------------------------------------------------------------- autostart / uninstall
def test_autostart_linux_and_mac(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    p = autostart.enable("linux", tmp_path, ["/opt/away", "tray"])
    assert p == str(tmp_path / ".config" / "autostart" / "project-away.desktop")
    text = (tmp_path / ".config" / "autostart" / "project-away.desktop").read_text()
    assert "Exec=/opt/away tray" in text and text.startswith("[Desktop Entry]")
    assert autostart.disable("linux", tmp_path) is True
    assert autostart.disable("linux", tmp_path) is False
    p = autostart.enable("darwin", tmp_path, ["/Applications/Project Away.app/x", "tray"])
    plist = (tmp_path / "Library" / "LaunchAgents" / f"{autostart.LABEL}.plist").read_text()
    assert "<string>/Applications/Project Away.app/x</string>" in plist and "<key>RunAtLoad</key><true/>" in plist
    assert autostart.disable("darwin", tmp_path) is True


def test_desktop_entry_quotes_spaces():
    assert 'Exec="/a b/away" tray' in autostart.desktop_entry(["/a b/away", "tray"])


def test_launch_command(monkeypatch):
    import sys

    assert autostart.launch_command()[1:] == ["-m", "away", "tray"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert autostart.launch_command() == [sys.executable, "tray"]


def test_uninstall(tmp_path, capsys):
    paths.data_dir().mkdir(parents=True)
    (paths.data_dir() / "models").mkdir()
    paths.config_dir().mkdir(parents=True)
    assert uninstall.run(dry_run=True) == []
    assert paths.data_dir().exists()
    removed = uninstall.run(yes=True, keep_config=True)
    assert removed == [paths.data_dir()]
    assert not paths.data_dir().exists() and paths.config_dir().exists()
    uninstall.run(yes=True)
    assert not paths.config_dir().exists()
    assert uninstall.run(yes=True) == []
    out = capsys.readouterr().out
    assert "pipx uninstall project-away" in out
    assert "no Project Away user data found" in out


def test_uninstall_abort(monkeypatch):
    paths.data_dir().mkdir(parents=True)
    monkeypatch.setattr("builtins.input", lambda p: "n")
    assert uninstall.run() == []
    assert paths.data_dir().exists()


def test_uninstall_hints():
    assert "Installed apps" in uninstall.program_removal_hint("win32")
    assert "Trash" in uninstall.program_removal_hint("darwin")
    assert "apt remove" in uninstall.program_removal_hint("linux")


# ---------------------------------------------------------------- tray / overlay
class FakeApp:
    def __init__(self):
        self.cfg = AppConfig()
        self.paused_calls = []
        self.stopped = False
        self.dictation = type("D", (), {"enabled": True})()

    def set_paused(self, p):
        self.paused_calls.append(p)

    def toggle_dictation(self):
        if self.dictation is not None:
            self.dictation.enabled = not self.dictation.enabled

    def stop(self):
        self.stopped = True


def test_tray_controller():
    app = FakeApp()
    t = TrayController(app)
    assert t.toggle_pause() is True and t.toggle_pause() is False
    assert app.paused_calls == [True, False]
    assert t.dictation_enabled() is True
    assert t.toggle_dictation() is False
    p = t.settings_path()
    assert p == str(paths.config_file()) and paths.config_file().exists()
    t.quit()
    assert app.stopped
    app.dictation = None
    assert t.toggle_dictation() is False and t.dictation_enabled() is False


def test_tray_icon_image():
    pytest.importorskip("PIL")
    from away.app.tray import make_icon_image

    img = make_icon_image(False, 32)
    assert img.size == (32, 32)
    assert img.getpixel((16, 4))[3] == 255
    assert make_icon_image(True).getpixel((32, 6))[:3] == (140, 140, 140)


def test_overlay_status_text():
    assert status_text(Event("gesture.clutch", {"paused": True})) == "PAUSED"
    assert status_text(Event("gesture.clutch", {"paused": False})) == "ACTIVE"
    assert status_text(Event("dictation.result", {"status": "ok", "text": "hi"})) == "typed: hi"
    assert status_text(Event("dictation.result", {"status": "gated"})) == "gated"
    assert status_text(Event("gesture.move", {})) is None


# ---------------------------------------------------------------- CLI
def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "pinch_click  -> click" in out
    assert "swipe_right  -> swipe_right" in out
    assert "palm_clutch  -> clutch" in out
    assert "wifi window 'push' -> push" in out


def test_cli_doctor_and_config(capsys):
    assert main(["doctor"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["bundled_models"]["static_pose.npz"] is True
    assert "hands/face" in info["extras"]
    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(paths.config_file())
    assert main(["config", "init"]) == 0
    capsys.readouterr()
    assert main(["config"]) == 0
    assert "hands:" in capsys.readouterr().out


def test_cli_autostart_and_uninstall(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(autostart, "enable", lambda: "X")
    monkeypatch.setattr(autostart, "disable", lambda: False)
    assert main(["autostart", "on"]) == 0
    assert "enabled: X" in capsys.readouterr().out
    assert main(["autostart", "off"]) == 0
    assert "was not enabled" in capsys.readouterr().out
    assert main(["uninstall", "--dry-run"]) == 0


def test_cli_setup(monkeypatch):
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return setup_wizard.SetupReport(models={"a": "ok"})

    monkeypatch.setattr(setup_wizard, "run", fake_run)
    assert main(["setup", "-y", "--skip-asr", "--skip-probe", "--asr-model", "small"]) == 0
    assert seen == {
        "yes": True,
        "asr_model": "small",
        "skip_asr": True,
        "skip_probe": True,
        "connections": 8,
        "gpu": None,
    }
    monkeypatch.setattr(setup_wizard, "run", lambda **kw: setup_wizard.SetupReport(models={"a": "failed: x"}))
    assert main(["setup", "-y"]) == 1


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "project-away 0.1.21" in capsys.readouterr().out
