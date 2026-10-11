"""v0.1.6 real-user fixes: camera hand-over to Settings, camera list/defaults, dead secondary camera, CPU
speech fallback + config migration, ASR backlog, face gate diagnostics, shutdown race, frozen calibration."""

from __future__ import annotations

import json
import queue
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app import camera_share as CS
from away.app.camera_loop import CameraLoop, degrade_plan, is_shutdown_error
from away.events import EventBus


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ----------------------------------------------------------------------------------- camera share
def test_share_protocol_and_staleness(tmp_path):
    clock = Clock()
    app = CS.CameraShare(tmp_path, clock=clock, pid=1)
    ui = CS.CameraShare(tmp_path, clock=clock, pid=2)
    assert ui.owner() is None and not app.requested()
    app.publish_owner([0])
    assert ui.owner()["cameras"] == ["0"] and ui.owned_devices() == {"0"}
    assert app.owner() is None  # its own file
    ui.request()
    assert app.requested() and not ui.requested()
    app.ack_released([0])
    assert ui.released()
    clock.t += 10  # Settings crashed: its request goes stale, the app takes the camera back
    assert not app.requested() and ui.owner() is None
    ui.request()
    ui.release()
    assert not app.requested()


def test_lease_without_running_app_is_a_noop(tmp_path):
    lease = CS.CameraLease(CS.CameraShare(tmp_path, pid=2), timeout=0.2).open()
    assert not lease.app_running and not lease.handed_over
    lease.close()
    assert not (tmp_path / CS.REQUEST_FILE).exists()


def fake_frames(n: int = 10_000, delay: float = 0.002):
    def gen():
        for i in range(n):
            time.sleep(delay)
            yield float(i), np.zeros((2, 2, 3), np.uint8)

    return gen


def test_app_hands_camera_to_settings_and_takes_it_back(tmp_path):
    stop = threading.Event()
    app_share = CS.CameraShare(tmp_path, pid=1)
    opened: list[int] = []
    events: list[tuple[str, dict[str, Any]]] = []
    released: list[int] = []

    def open_frames():
        opened.append(1)
        return fake_frames()()

    loop = CameraLoop(
        open_frames,
        lambda n, t, f: None,
        stop,
        cameras=[0],
        share=app_share,
        publish=lambda t, d: events.append((t, d)),
        on_release=lambda: released.append(1),
        check_s=0.02,
        owner_s=0.05,
    )
    th = threading.Thread(target=loop.run, daemon=True)
    th.start()
    ui = CS.CameraShare(tmp_path, pid=2)
    deadline = time.time() + 3
    while ui.owner() is None and time.time() < deadline:
        time.sleep(0.01)
    assert ui.owner() is not None
    with CS.CameraLease(ui, timeout=3.0, heartbeat_s=0.05) as lease:
        assert lease.app_running and lease.handed_over
        assert released == [1] and ("camera.handover", {"released": True, "cameras": [0]}) in events
    deadline = time.time() + 3
    while len(opened) < 2 and time.time() < deadline:
        time.sleep(0.01)
    stop.set()
    th.join(2)
    assert len(opened) == 2 and loop.handovers == 1  # reacquired after the test
    assert not (tmp_path / CS.OWNER_FILE).exists()


def test_camera_loop_retries_unopenable_camera_and_reports_once():
    stop = threading.Event()
    events: list[str] = []
    tries: list[int] = []

    def open_frames():
        tries.append(1)
        if len(tries) < 3:
            raise RuntimeError("cannot open camera 0 (in use by another app, no device, or permission denied)")
        stop.set()
        return iter([])

    CameraLoop(open_frames, lambda *a: None, stop, publish=lambda t, d: events.append(t), retry_s=0.01).run()
    assert len(tries) == 3 and events.count("camera.error") == 1 and "camera.recovered" in events


def test_camera_loop_swallows_shutdown_race():
    stop = threading.Event()

    def process(n, t, f):
        raise RuntimeError("cannot schedule new futures after shutdown")

    CameraLoop(fake_frames(5, 0.0), process, stop).run()  # must not raise
    assert stop.is_set()
    assert is_shutdown_error(RuntimeError("cannot schedule new futures after shutdown"))
    assert not is_shutdown_error(ValueError("x"))

    stop2 = threading.Event()

    def other(n, t, f):
        raise ValueError("real bug")

    with pytest.raises(ValueError):
        CameraLoop(fake_frames(5, 0.0), other, stop2).run()


def test_app_stop_joins_camera_thread_before_returning():
    from away.app.runner import AwayApp

    app = AwayApp.__new__(AwayApp)
    app.stop_event = threading.Event()
    done: list[int] = []

    def cam():
        app.stop_event.wait()
        time.sleep(0.1)  # finishing the frame + closing MediaPipe
        done.append(1)

    th = threading.Thread(target=cam, name="camera", daemon=True)
    th.start()
    app.threads = [th]
    app._listeners = []
    app.pipeline = types.SimpleNamespace(set_paused=lambda p: None)
    app.bus = EventBus()
    app.stop()
    assert done == [1]


def test_camera_error_message_is_clear():
    exc = RuntimeError("cannot open camera 0 (in use by another app, no device, or permission denied)")
    msg = CS.camera_error_message(exc, 0, app_running=False, handed_over=False)
    assert "in use by Project Away or another app" in msg and "permission denied" not in msg.split(" - ")[0]
    assert "did not hand it over" in CS.camera_error_message(exc, 0, True, False)
    assert CS.camera_error_message(ValueError("boom"), 0, False, False) == "camera error: boom"


def test_gesture_test_borrows_camera_and_reports_clearly(tmp_path):
    from away.app.settings_gui import SettingsModel, gesture_test_worker

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    out: queue.Queue[str] = queue.Queue()

    class Tracker:
        closed = False

        def process(self, rgb, t):
            return types.SimpleNamespace(landmarks=None)

        def close(self):
            Tracker.closed = True

    def no_camera(idx, mirror):
        raise RuntimeError(f"cannot open camera {idx} (in use by another app, no device, or permission denied)")
        yield  # pragma: no cover

    share = CS.CameraShare(tmp_path / "share", pid=2)
    gesture_test_worker(m, out, threading.Event(), share=share, frames=no_camera, tracker_factory=Tracker)
    msgs = [out.get_nowait() for _ in range(out.qsize())]
    assert any("in use by Project Away or another app" in x for x in msgs) and Tracker.closed

    # with the app running: it is asked to hand over first, and the request is withdrawn afterwards
    app = CS.CameraShare(tmp_path / "share", pid=1)
    app.publish_owner([0])

    def app_side():
        for _ in range(200):
            if app.requested():
                app.ack_released([0])
                return
            time.sleep(0.01)

    threading.Thread(target=app_side, daemon=True).start()
    frames = lambda idx, mirror: iter([(0.0, None)])  # noqa: E731
    gesture_test_worker(m, out, threading.Event(), share=share, frames=frames, tracker_factory=Tracker)
    msgs = [out.get_nowait() for _ in range(out.qsize())]
    assert any("paused hand tracking" in x for x in msgs)
    assert not app.requested()


# ---------------------------------------------------------------------------------- cameras tab
def test_cameras_tab_shows_default_camera_and_in_use(tmp_path):
    from away.app.settings_gui import SettingsModel
    from away.multicam.capture import CameraDevice

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    m.camera_share = CS.CameraShare(tmp_path / "s", pid=2)
    CS.CameraShare(tmp_path / "s", pid=1).publish_owner([0])
    rows = m.camera_rows()
    assert rows[0]["device"] == 0 and rows[0]["role"] == "primary" and rows[0]["implicit"] and rows[0]["in_use"]
    assert m.set_camera(0, "device", "1: Other") is None and m.cfg.hands.camera_index == 1
    assert m.set_camera(0, "width", "1280") is None and m.cfg.performance.camera_width == 1280
    assert m.set_camera(0, "role", "secondary") is not None
    assert m.remove_camera(0) is not None  # the only camera
    choices = m.device_choices([CameraDevice(0, "camera 0", "Integrated Webcam", in_use=True)])
    assert choices == ["0: Integrated Webcam (in use by Project Away)"]
    assert m.add_camera(2) is not None and "not found" in str(m.add_camera(2))  # only real devices
    assert m.cfg.hands.cameras == []


def test_windows_camera_list_does_not_open_devices():
    from away.multicam import capture

    def run(argv, **kw):
        assert argv[0] == "powershell" and kw["timeout"] <= 10
        return types.SimpleNamespace(returncode=0, stdout="Integrated Webcam\r\nLogitech BRIO\r\n\r\n")

    devs = capture.list_cameras(sysfs="/nonexistent", platform="win32", run=run, in_use={"0"})
    assert [(d.index, d.name, d.in_use, d.ok) for d in devs] == [
        (0, "Integrated Webcam", True, None),
        (1, "Logitech BRIO", False, None),
    ]
    assert capture.windows_camera_names(lambda *a, **k: (_ for _ in ()).throw(OSError("no ps"))) == []
    js = json.dumps({"SPCameraDataType": [{"_name": "FaceTime HD Camera"}]})
    assert capture.parse_system_profiler(js) == ["FaceTime HD Camera"]


def test_dead_secondary_camera_never_stops_tracking():
    assert degrade_plan([0, 2], 0, set()) == ("multi", None, "")
    mode, dev, msg = degrade_plan([0, 2], 0, {1})
    assert (mode, dev) == ("single", 0) and "camera 2 could not be opened" in msg and "camera 0 only" in msg
    assert degrade_plan([0, 2], 0, {0})[:2] == ("single", 2)  # primary dead: track on the one that works
    assert degrade_plan([0, 2, 4], 0, {2})[0] == "multi"
    assert degrade_plan([0, 2], 0, {0, 1})[:2] == ("single", 0)  # nothing works: keep retrying the primary


# ----------------------------------------------------------------------------- speech: CPU fallback
def test_v014_asr_model_is_migrated_to_auto(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("voice:\n  asr_model: large-v3-turbo\n", encoding="utf-8")
    cfg, _ = C.load(p)
    assert cfg.voice.asr_model == "auto" and cfg.migrations and cfg.config_version == C.CONFIG_VERSION
    C.save(cfg, p)
    assert "config_version: 2" in p.read_text() and "migrations" not in p.read_text()
    # a choice made in v0.1.6+ (config_version 2) is the user's: kept
    p.write_text("config_version: 2\nvoice:\n  asr_model: large-v3-turbo\n", encoding="utf-8")
    cfg2, _ = C.load(p)
    assert cfg2.voice.asr_model == "large-v3-turbo" and cfg2.migrations == []
    assert C.from_dict({"voice": {"asr_model": "small"}})[0].voice.asr_model == "small"


def test_asr_plan_uses_cpu_model_when_gpu_unusable():
    from away.app.performance import asr_plan
    from away.app.tuning import Profile

    cfg = C.AppConfig()
    gpu = Profile(device="cuda", asr_model="large-v3-turbo", gpu_capable=True)
    cpu = Profile(device="cpu", asr_model="small")
    assert asr_plan(cfg, gpu, cpu) == ("large-v3-turbo", "small", None)
    cfg.voice.asr_model = "large-v3-turbo"  # explicit, but the GPU pack is missing -> CPU profile
    model, fb, note = asr_plan(cfg, Profile(device="cpu", asr_model="large-v3-turbo", gpu_capable=True), cpu)
    assert (model, fb) == ("small", "small") and "GPU pack" in str(note)
    cfg.voice.device = "cpu"  # the user insists
    assert asr_plan(cfg, Profile(device="cpu", asr_model="large-v3-turbo"), cpu)[0] == "large-v3-turbo"


def test_cuda_failure_falls_back_to_cpu_model(monkeypatch):
    from away.app import gpupack
    from away.voice import asr as A

    monkeypatch.setattr(gpupack, "activate", lambda *a, **k: [])
    built: list[tuple[str, str]] = []

    class W:
        def __init__(self, model, device, *rest):
            built.append((model, device))
            self.device = device
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

        def transcribe(self, audio, prompt=None):
            return "ok"

    reasons: list[str] = []
    a: Any = A.make_asr(
        "faster-whisper", "large-v3-turbo", "cuda", "float16", on_fallback=reasons.append, factory=W, cpu_model="small"
    )
    assert built == [("large-v3-turbo", "cuda"), ("small", "cpu")]
    assert "smaller 'small' model" in reasons[0] and a is not None


def test_cuda_error_at_runtime_switches_to_cpu():
    from away.voice.asr import FallbackASR

    class G:
        def transcribe(self, audio, prompt=None):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    class Cpu:
        def transcribe(self, audio, prompt=None):
            return "hello"

    reasons: list[str] = []
    f = FallbackASR(G(), Cpu, reasons.append, "small")
    assert f.device == "cuda"
    assert f.transcribe(np.zeros(10, np.float32)) == "hello" and f.device == "cpu" and len(reasons) == 1
    assert f.transcribe(np.zeros(10, np.float32)) == "hello" and len(reasons) == 1

    class Bug:
        def transcribe(self, audio, prompt=None):
            raise ValueError("boom")

    with pytest.raises(ValueError):
        FallbackASR(Bug(), Cpu).transcribe(np.zeros(1, np.float32))


def test_asr_queue_keeps_newest_and_warns_once():
    from away.voice.always_on import ASRQueue

    drops: list[int] = []
    warned: list[int] = []
    q = ASRQueue(2, on_drop=drops.append, on_overload=lambda: warned.append(1))
    jobs = [lambda i=i: i for i in range(5)]
    for j in jobs:
        q.put(j)
    assert len(q) == 2 and drops == [1, 2, 3] and warned == [1]
    assert q.get(0.01)() == 3 and q.get(0.01)() == 4 and q.get(0.01) is None
    with pytest.raises(ValueError):
        ASRQueue(0)


def test_gpu_pack_offer_notice_once(tmp_path):
    from away.app import notices

    assert notices.once("gpu-pack-offer", tmp_path) and not notices.once("gpu-pack-offer", tmp_path)
    assert notices.seen("gpu-pack-offer", tmp_path)


def test_app_notice_is_once_per_key():
    from away.app.runner import AwayApp

    app = AwayApp.__new__(AwayApp)
    app.bus, app._notices = EventBus(), set()
    seen: list[dict[str, Any]] = []
    app.bus.subscribe("app.notice", lambda e: seen.append(e.data))
    assert app.notice("t", "m", key="k") and not app.notice("t", "m", key="k")
    assert len(seen) == 1


# --------------------------------------------------------------------------------------- face gate
def test_gate_accepts_turned_head_with_moving_mouth():
    from away.face.gate import SpeechGate

    g = SpeechGate()
    g.enroll({0: 0.3})
    for i in range(20):  # face visible, ASD does not pick it (off-axis), but the mouth moves
        g.observe(i * 0.1, None, 1, 0.2 if i % 2 else 0.05)
    info = g.explain(0.0, 1.9)
    assert info["allowed"] and info["mouth_moving_frames"] >= 5 and "facing" in info["reason"]
    assert g.allow_ambient(0.0, 1.9)


def test_gate_tolerates_starved_face_tracking_but_not_absence():
    from away.face.gate import SpeechGate

    g = SpeechGate()
    for i in range(5):
        g.observe(i * 0.1, 0, 1, 0.4)
    # no face frames during the utterance (CPU busy with Whisper) right after the user was there
    assert g.explain(3.0, 4.0)["allowed"] and "CPU busy" in g.last_explain["reason"]
    tv = SpeechGate()
    for i in range(5):
        tv.observe(i * 0.1, None, 0, None)
    assert not tv.allow_ambient(3.0, 4.0)
    assert not SpeechGate().allow_ambient(0, 1)


def test_gate_still_rejects_someone_else():
    from away.face.gate import SpeechGate

    g = SpeechGate()
    g.enroll({0: 0.3, 1: 0.1})
    for i in range(10):
        g.observe(i * 0.1, 1, 2, 0.0)
    info = g.explain(0.0, 0.9)
    assert not info["allowed"] and info["reason"] == "someone else was speaking"


def test_utterance_log_carries_gate_evidence():
    from away.bindings import BindingsConfig
    from away.voice.always_on import AmbientSession, StreamingEndpointer, Utterance
    from away.voice.asr import FakeASR
    from away.voice.output import ListSink

    bus = EventBus()
    got: list[dict[str, Any]] = []
    bus.subscribe("voice.utterance", lambda e: got.append(e.data))

    class Clean:
        def clean(self, text, context=None):
            return text

    s = AmbientSession(
        FakeASR(["hello there"]),
        Clean(),
        ListSink(),
        StreamingEndpointer(16000),
        commands=lambda: BindingsConfig().voice_phrases(),
        run_command=lambda a, p: None,
        bus=bus,
        gate=lambda t0, t1: False,
        gate_info=lambda: {"frames": 3, "reason": "face present but mouth not moving"},
    )
    s.process(Utterance(np.zeros(1600, np.float32), 0.0, 0.1))
    assert got[0]["decision"] == "ignore" and got[0]["gate"]["frames"] == 3


def test_gate_hint_after_consecutive_rejections():
    from away.voice.always_on import GateHint

    h = GateHint(3)
    rej = {"decision": "ignore", "reason": "face gate: you were not facing the screen and speaking"}
    assert not h.observe(rej) and not h.observe(rej)
    assert not h.observe({"decision": "type"})  # a success resets the streak
    assert [h.observe(rej) for _ in range(4)] == [False, False, True, False]  # once per session


def test_disabling_face_gate_still_types():
    from away.app.runner import effective_ambient_fallback
    from away.app.settings_gui import SettingsModel
    from away.voice.always_on import decide

    cfg = C.AppConfig()
    assert effective_ambient_fallback(cfg) == "commands_only"
    cfg.face.gate_dictation = False
    assert effective_ambient_fallback(cfg) == "type"
    d = decide("hello world", commands={}, gate=None, fallback=effective_ambient_fallback(cfg))
    assert d.kind == "type"
    m = SettingsModel(C.AppConfig())
    assert m.set_voice("face.gate_dictation", False) is None
    assert m.cfg.voice.ambient_fallback == "type"  # set together


def test_tray_face_gate_toggle_writes_config(tmp_path):
    from away.app.tray import TrayController

    app = types.SimpleNamespace(cfg=C.AppConfig())
    t = TrayController(app)
    assert t.face_gate_on()
    p = tmp_path / "c.yaml"
    assert t.set_face_gate(False, p) is False
    saved, _ = C.load(p)
    assert not saved.face.gate_dictation and saved.voice.ambient_fallback == "type" and not t.face_gate_on()
    assert t.set_face_gate(True, p) and C.load(p)[0].face.gate_dictation


def test_single_face_with_new_track_id_is_still_the_user():
    from away.app.runner import Pipeline

    p = Pipeline(C.AppConfig())
    p.on_faces(0.0, [(0.5, 0.5, 0.3)], [{}])
    first = p.gate.primary
    p.faces = types.SimpleNamespace(update=lambda boxes: [first + 7])  # tracker re-identified the face
    p.on_faces(0.1, [(0.5, 0.5, 0.3)], [{"jawOpen": 0.5}])
    assert p.gate.primary == first + 7


# ------------------------------------------------------------------------------- frozen calibration
def test_away_command_frozen_and_source():
    from away.app.frozen import CONSOLE_COMMANDS, away_command

    assert away_command(["calibrate-cameras"], frozen=True, executable="C:/PA/ProjectAway.exe") == [
        "C:/PA/ProjectAway.exe",
        "calibrate-cameras",
    ]
    assert away_command(["settings"], frozen=False, executable="py") == ["py", "-m", "away", "settings"]
    assert "calibrate-cameras" in CONSOLE_COMMANDS and "monitors" in CONSOLE_COMMANDS


def test_settings_calibration_command_and_outcome(tmp_path, monkeypatch):
    import sys

    from away.app.settings_gui import SettingsModel, calibration_outcome, read_calibration_status

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    status = tmp_path / "st.json"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    cmd = m.calibration_command(status)
    assert cmd[1] == "calibrate-cameras" and "-m" not in cmd and cmd[-1] == str(status)
    monkeypatch.delattr(sys, "frozen")
    assert m.calibration_command(status)[1:4] == ["-m", "away", "calibrate-cameras"]
    assert read_calibration_status(status) is None
    status.write_text(json.dumps({"state": "done", "message": "calibrated, reprojection error 1.20 px"}))
    assert calibration_outcome(read_calibration_status(status), 0, "") == (
        None,
        "calibrated, reprojection error 1.20 px",
    )
    assert calibration_outcome({"state": "failed", "message": "camera 2: no frames"}, 1, "")[0] == "camera 2: no frames"
    err, ok = calibration_outcome(None, 2, "Traceback...\nRuntimeError: cannot open camera 2")
    assert ok is None and "exit 2" in str(err) and "cannot open camera 2" in str(err)


def test_calibrate_cli_writes_status_on_error(tmp_path):
    from away import cli

    st = tmp_path / "st.json"
    cfgp = tmp_path / "c.yaml"
    C.save(C.AppConfig(), cfgp)
    assert cli.main(["calibrate-cameras", "--config", str(cfgp), "--status-file", str(st)]) != 0
    data = json.loads(Path(st).read_text())
    assert data["state"] == "failed" and "two cameras" in data["message"]


def test_no_subprocess_hardcodes_python_dash_m_away():
    """``sys.executable -m away`` is wrong in the frozen build (sys.executable is ProjectAway.exe): every
    child-process argv goes through ``away.app.frozen.away_command``."""
    src = Path(__file__).resolve().parents[1] / "src" / "away"
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if p.name != "frozen.py" and '"-m", "away"' in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_tray_settings_command_frozen(monkeypatch):
    import sys

    from away.app.tray import TrayController

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert TrayController.settings_command() == [sys.executable, "settings"]
    monkeypatch.delattr(sys, "frozen")
    assert TrayController.settings_command()[1:] == ["-m", "away", "settings"]
