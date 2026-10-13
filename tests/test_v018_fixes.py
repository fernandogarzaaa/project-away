"""v0.1.8 fixes: two webcams on Windows (slow open, fallback + retry, honest notices, device list) and the
Windows auto-updater (wait-then-install helper, job breakaway, forced exit, install-state report, logging)."""

from __future__ import annotations

import json
import logging
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app import camera_share as CS
from away.app import updater as U
from away.app.camera_loop import CameraLoop, degrade_plan, describe_failures, probe_cameras
from away.events import EventBus
from away.multicam import capture
from away.multicam.capture import CameraDevice, CameraThread, FramePairer

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================== camera: simulation
class FakeCap:
    """A webcam: ``silent`` cameras open but never deliver a frame."""

    def __init__(self, silent: bool = False, first_delay: float = 0.0) -> None:
        self.silent = silent
        self.t0 = time.monotonic()
        self.first_delay = first_delay
        self.released = False

    def read(self) -> tuple[bool, Any]:
        time.sleep(1 / 60)
        if self.silent or time.monotonic() - self.t0 < self.first_delay:
            return False, None
        return True, np.zeros((4, 6, 3), np.uint8)

    def release(self) -> None:
        self.released = True


class FakeDevices:
    """Per-device behaviour, changeable while a test runs: open delay, open error, silent."""

    def __init__(self, **behaviour: dict[str, Any]) -> None:
        self.behaviour = {int(k[1:]): v for k, v in behaviour.items()}  # d0=..., d1=...
        self.lock = threading.Lock()
        self.opening = 0
        self.max_parallel_opens = 0
        self.opens: list[int] = []

    def set(self, dev: int, **kw: Any) -> None:
        self.behaviour[dev] = kw

    def opener(self, spec: C.CameraSpec) -> FakeCap:
        with self.lock:  # read under the lock so a test can change several devices between two probes
            b = self.behaviour.get(int(spec.device), {})
            self.opening += 1
            self.max_parallel_opens = max(self.max_parallel_opens, self.opening)
            self.opens.append(int(spec.device))
        try:
            time.sleep(b.get("open_delay", 0.0))
            if b.get("error"):
                raise RuntimeError(f"cannot open camera {spec.device!r} (no device, busy, or permission denied)")
            return FakeCap(silent=b.get("silent", False), first_delay=b.get("first_delay", 0.0))
        finally:
            with self.lock:
                self.opening -= 1


def _threads(devs: FakeDevices, specs, pairer, session, lock, first_frame_s=10.0):
    ths = [
        CameraThread(
            i,
            c,
            pairer,
            session,
            open_lock=lock,
            first_frame_s=first_frame_s,
            opener=devs.opener,
            convert=lambda f, m: f,
        )
        for i, c in enumerate(specs)
    ]
    for th in ths:
        th.start()
    return ths


class _Pipeline:
    def __init__(self, app: Any, stop_after: int | None = None) -> None:
        self.app, self.fused, self.stop_after = app, 0, stop_after

    def on_fused(self, fused: Any, t: float) -> None:
        self.fused += 1
        if self.stop_after is not None and self.fused >= self.stop_after:
            self.app.stop_event.set()

    def on_faces(self, *a: Any) -> None:
        pass

    def on_hand(self, *a: Any) -> None:
        pass


class _Tracker:
    def process(self, rgb: Any, t: float) -> Any:
        return types.SimpleNamespace(landmarks=None, handedness=None, score=0.0)

    def close(self) -> None:
        pass


def _app(tmp_path: Path, devs: FakeDevices, serial: bool = False, stop_after: int | None = 5):
    from away.app.runner import AwayApp

    app = AwayApp.__new__(AwayApp)
    cfg = C.AppConfig()
    cfg.hands.cameras = [C.CameraSpec(device=0, role="primary"), C.CameraSpec(device=1, role="secondary")]
    cfg.hands.multicam.fusion = "best_view"
    app.cfg = cfg
    app.stop_event = threading.Event()
    app.bus = EventBus()
    app.camera_share = CS.CameraShare(tmp_path / "share", pid=1)
    app.camera_confidences = {}
    app.models_dir = tmp_path
    app.pipeline = _Pipeline(app, stop_after)
    app._notices = set()
    app._camera_episode = 0
    app.events = []
    app.bus.subscribe("*", lambda e: app.events.append((e.topic, dict(e.data))))
    app.bus.subscribe("cameras.degraded", app._on_cameras_degraded)
    app.bus.subscribe("cameras.restored", app._on_cameras_restored)
    app._start_cameras = lambda specs, pairer, session, lock: _threads(devs, specs, pairer, session, lock)
    app._serial_open = lambda: serial
    return app


def _topics(app: Any, topic: str) -> list[dict[str, Any]]:
    return [d for t, d in app.events if t == topic]


def _run_loop(app: Any, run_single: Any = None, timeout: float = 30.0, **kw: Any) -> None:
    th = threading.Thread(
        target=app._multicam_loop, args=(None, None, run_single), kwargs={"trackers": [_Tracker(), _Tracker()], **kw}
    )
    th.daemon = True
    th.start()
    th.join(timeout)
    app.stop_event.set()
    th.join(5)
    assert not th.is_alive()


# =========================================================================== camera: (a) slow open
def test_slow_opening_webcams_are_not_marked_failed(tmp_path):
    """The v0.1.7 regression: two webcams that take 3.5 s to open (Windows MSMF/DSHOW) were marked failed by
    the 3 s probe and multi-camera was never tried again. Now both are waited for and fused."""
    devs = FakeDevices(d0={"open_delay": 3.5}, d1={"open_delay": 3.5})
    app = _app(tmp_path, devs, serial=False, stop_after=5)
    _run_loop(app)
    assert _topics(app, "cameras.degraded") == []
    assert _topics(app, "cameras.started") and app.pipeline.fused >= 5


def test_windows_opens_cameras_one_after_another(tmp_path):
    devs = FakeDevices(d0={"open_delay": 0.4}, d1={"open_delay": 0.4})
    app = _app(tmp_path, devs, serial=True, stop_after=5)
    _run_loop(app, probe_s=2.0)
    assert devs.max_parallel_opens == 1 and sorted(devs.opens[:2]) == [0, 1]
    assert _topics(app, "cameras.started") and _topics(app, "cameras.degraded") == []


def test_probe_timeout_is_generous_and_grows_with_serial_cameras(tmp_path, monkeypatch):
    from away.app import runner

    assert runner.MULTICAM_PROBE_S >= 10.0
    app = _app(tmp_path, FakeDevices())
    app._serial_open = lambda: True
    assert app._probe_timeout(2, 12.0) == 18.0
    app._serial_open = lambda: False
    assert app._probe_timeout(2, 12.0) == 12.0


class _T:
    def __init__(self, alive: bool, error: str | None = None) -> None:
        self._alive, self.error = alive, error

    def is_alive(self) -> bool:
        return self._alive


def test_probe_marks_failed_only_on_error_or_no_frames_by_deadline():
    clock = {"t": 0.0}
    seen = {0: False, 1: False, 2: False, 3: False}
    threads = [_T(True), _T(False, "cannot open camera 1"), _T(True), _T(False)]

    def sleep(dt: float) -> None:
        clock["t"] += dt
        if clock["t"] >= 4.0:  # camera 0 is slow: first frame after 4 s (the old 3 s probe failed it)
            seen[0] = True

    failed = probe_cameras(threads, lambda i: seen[i], 12.0, threading.Event(), clock=lambda: clock["t"], sleep=sleep)
    assert 0 not in failed
    assert failed[1] == "open error: cannot open camera 1"
    assert failed[2] == "no frames within 12 s"
    assert failed[3] == "closed before sending a frame"
    assert clock["t"] >= 12.0  # waited the whole deadline for the silent camera 2


def test_probe_returns_as_soon_as_every_camera_delivered():
    clock = {"t": 0.0}

    def sleep(dt: float) -> None:
        clock["t"] += dt

    out = probe_cameras(
        [_T(True), _T(True)], lambda i: True, 12.0, threading.Event(), clock=lambda: clock["t"], sleep=sleep
    )
    assert out == {} and clock["t"] == 0.0


def test_camera_thread_waits_for_a_slow_first_frame_then_gives_up_clearly():
    pairer = FramePairer(1)
    stop = threading.Event()
    devs = FakeDevices(d0={"first_delay": 0.8})  # opens, but the first real frame takes 0.8 s
    th = CameraThread(
        0, C.CameraSpec(device=0), pairer, stop, first_frame_s=3.0, opener=devs.opener, convert=lambda f, m: f
    )
    th.start()
    t0 = time.monotonic()
    while not pairer.seen(0) and time.monotonic() - t0 < 5:
        time.sleep(0.02)
    stop.set()
    th.join(2)
    assert pairer.seen(0) and th.error is None and th.opened.is_set()

    stop2 = threading.Event()
    silent = FakeDevices(d0={"silent": True})
    th2 = CameraThread(
        0,
        C.CameraSpec(device=0),
        FramePairer(1),
        stop2,
        first_frame_s=0.3,
        opener=silent.opener,
        convert=lambda f, m: f,
    )
    th2.start()
    th2.join(3)
    assert th2.error == "camera opened but sent no frame within 0.3 s"

    bad = FakeDevices(d0={"error": True})
    th3 = CameraThread(0, C.CameraSpec(device=0), FramePairer(1), threading.Event(), opener=bad.opener)
    th3.start()
    th3.join(2)
    assert th3.error is not None and "cannot open camera 0" in th3.error and not th3.opened.is_set()


def test_camera_thread_waiting_for_its_turn_stops_cleanly():
    lock = threading.Lock()
    lock.acquire()  # another camera is still opening
    stop = threading.Event()
    devs = FakeDevices()
    th = CameraThread(0, C.CameraSpec(device=0), FramePairer(1), stop, open_lock=lock, opener=devs.opener)
    th.start()
    time.sleep(0.2)
    stop.set()
    th.join(2)
    assert not th.is_alive() and devs.opens == [] and th.error is None


# ======================================================================= camera: (b)(d) honest notices
def test_degrade_message_names_only_failed_cameras_with_real_error():
    mode, dev, msg = degrade_plan([0, 1], 0, {1: "no frames within 12 s"}, 45)
    assert (mode, dev) == ("single", 0)
    assert msg.startswith("camera 1 (no frames within 12 s) failed; tracking with camera 0 only")
    assert "camera 0 (" not in msg and "every 45 s" in msg
    mode, dev, msg = degrade_plan([0, 1], 0, {0: "open error: cannot open camera 0"})
    assert (mode, dev) == ("single", 1) and "camera 0 (open error: cannot open camera 0)" in msg
    assert "camera 1 (" not in msg


def test_no_working_camera_never_claims_to_track():
    mode, dev, msg = degrade_plan([0, 1], 0, {0: "open error: busy", 1: "no frames within 12 s"}, 10)
    assert mode == "none" and dev == 0
    assert "tracking with" not in msg and "no camera is working" in msg and "retrying every 10 s" in msg
    assert "camera 0 (open error: busy), camera 1 (no frames within 12 s)" in msg
    assert describe_failures([0, 1], {1}) == "camera 1"


def test_degraded_notice_once_per_problem_and_again_after_recovery(tmp_path):
    app = _app(tmp_path, FakeDevices())
    notes = []
    app.bus.subscribe("app.notice", lambda e: notes.append(e.data["title"]))
    ev = {"failed": {"1": "no frames within 12 s"}, "message": "m", "mode": "single"}
    app.bus.publish("cameras.degraded", ev)
    app.bus.publish("cameras.degraded", ev)  # the background retry failing the same way: quiet
    assert notes == ["Project Away: camera skipped"]
    app.bus.publish("cameras.degraded", {**ev, "mode": "none", "failed": {"0": "x", "1": "y"}})
    assert notes[-1] == "Project Away: no camera"
    app.bus.publish("cameras.restored", {"message": "back"})
    app.bus.publish("cameras.restored", {"message": "back"})
    assert notes.count("Project Away: cameras back") == 1
    app.bus.publish("cameras.degraded", ev)  # a new episode is reported again
    assert notes.count("Project Away: camera skipped") == 2


# ======================================================================= camera: (c) fallback + retry
def test_failed_secondary_falls_back_then_returns_to_fusion(tmp_path):
    devs = FakeDevices(d1={"error": True})
    app = _app(tmp_path, devs, stop_after=5)
    singles: list[Any] = []

    def run_single(device: Any, until: Any) -> None:
        singles.append(device)
        t0 = time.monotonic()
        devs.set(1)  # the user plugs the secondary camera back in while camera 0 keeps tracking
        while not until() and not app.stop_event.is_set() and time.monotonic() - t0 < 10:
            time.sleep(0.02)

    _run_loop(app, run_single, probe_s=1.0, retry_s=0.2)
    deg = _topics(app, "cameras.degraded")
    assert len(deg) == 1 and deg[0]["mode"] == "single" and list(deg[0]["failed"]) == ["1"]
    assert deg[0]["failed"]["1"].startswith("open error: cannot open camera 1")
    assert singles == [0]  # tracked on the working primary meanwhile
    retries = _topics(app, "cameras.retry")
    assert retries and retries[-1]["ok"] is True
    assert len(_topics(app, "cameras.restored")) == 1 and _topics(app, "cameras.started")
    assert app.pipeline.fused >= 5  # fusion is back


def test_background_retry_that_still_fails_never_interrupts_tracking(tmp_path):
    devs = FakeDevices(d1={"error": True})
    app = _app(tmp_path, devs, stop_after=None)
    ends: list[str] = []

    def run_single(device: Any, until: Any) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            if until():
                ends.append("until")
                return
            time.sleep(0.02)
        ends.append("still tracking")
        app.stop_event.set()

    _run_loop(app, run_single, probe_s=0.3, retry_s=0.1)
    assert ends == ["still tracking"]
    assert len([r for r in _topics(app, "cameras.retry") if not r["ok"]]) >= 2
    assert _topics(app, "cameras.started") == []


def test_no_camera_keeps_retrying_and_recovers(tmp_path):
    devs = FakeDevices(d0={"error": True}, d1={"error": True})
    app = _app(tmp_path, devs, stop_after=3)
    notes: list[str] = []
    app.bus.subscribe("app.notice", lambda e: notes.append(e.data["message"]))

    def fix_later() -> None:
        # after at least one retry of the whole set, and between two probes: fixing camera 0 while a probe has
        # already opened it but not yet camera 1 would (correctly) leave the app on camera 1 alone
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with devs.lock:
                n0, n1 = devs.opens.count(0), devs.opens.count(1)
                if n0 >= 2 and n0 == n1 and devs.opening == 0:
                    devs.behaviour[0] = {}
                    devs.behaviour[1] = {}
                    return
            time.sleep(0.005)
        devs.set(0)
        devs.set(1)

    threading.Thread(target=fix_later, daemon=True).start()
    _run_loop(app, lambda d, u: pytest.fail("must not track on a dead camera"), probe_s=0.5, none_retry_s=0.1)
    deg = _topics(app, "cameras.degraded")
    assert deg and all(d["mode"] == "none" for d in deg)
    assert sum("no camera is working" in n for n in notes) == 1  # notified once, not every retry
    assert len(_topics(app, "cameras.restored")) == 1 and app.pipeline.fused >= 3


def test_camera_loop_until_ends_the_session():
    stop = threading.Event()
    flag = {"end": False}
    frames: list[int] = []

    def gen():
        for i in range(100_000):
            time.sleep(0.002)
            yield float(i), None

    def process(n, t, f):
        frames.append(n)
        if n == 20:
            flag["end"] = True

    CameraLoop(gen, process, stop, until=lambda: flag["end"], check_s=0.01).run()
    assert 20 <= len(frames) < 1000 and not stop.is_set()


# ========================================================================== camera: (e) device list
def test_device_list_never_empty_when_listing_fails(tmp_path, monkeypatch):
    from away.app.settings_gui import SettingsModel

    cfg = C.AppConfig()
    cfg.hands.cameras = [C.CameraSpec(device=0, role="primary"), C.CameraSpec(device=1, role="secondary")]
    m = SettingsModel(cfg, tmp_path / "c.yaml")
    m.camera_share = CS.CameraShare(tmp_path / "s", pid=2)
    CS.CameraShare(tmp_path / "s", pid=1).publish_owner([0])  # the app holds camera 0 (the reported state)

    def boom(**kw):
        raise RuntimeError("enumeration failed")

    monkeypatch.setattr(capture, "list_cameras", boom)
    assert m.device_choices() == ["0: camera 0 (in use by Project Away)", "1: camera 1 (configured)"]
    assert m.parse_device(m.device_choices()[1]) == 1 and m._unknown_device(1) is None


def test_windows_listing_adds_held_cameras_and_closes_stdin():
    seen: dict[str, Any] = {}

    def run(argv, **kw):
        seen.update(kw)
        seen["cmd"] = argv[-1]
        return types.SimpleNamespace(returncode=0, stdout="USB Camera\r\n")  # the held camera is missing

    devs = capture.list_cameras(sysfs="/nonexistent", platform="win32", run=run, in_use={"1"})
    assert [(d.index, d.name, d.in_use) for d in devs] == [(0, "USB Camera", False), (1, "camera 1", True)]
    import subprocess

    assert seen["stdin"] == subprocess.DEVNULL
    assert "-Filter" in seen["cmd"] and "usbvideo" in seen["cmd"] and seen["timeout"] >= 10


def test_fallback_probe_lists_held_cameras_without_opening_them():
    probed: list[int] = []

    def probe(c: CameraDevice) -> CameraDevice:
        probed.append(c.index or 0)
        return CameraDevice(c.index, c.path, c.name, ok=c.index == 1)

    devs = capture.list_cameras(
        sysfs="/nonexistent",
        platform="win32",
        run=lambda *a, **k: types.SimpleNamespace(returncode=1, stdout=""),
        in_use={"0"},
        probe_fn=probe,
    )
    assert [(d.index, d.in_use) for d in devs] == [(0, True), (1, False)]
    assert 0 not in probed and probed[:3] == [1, 2, 3] and len(probed) <= 4  # stops after two misses


# ===================================================================================== updater
SETUP = "ProjectAway-0.1.8-windows-x64-setup.exe"
INNO_EXE = r"C:\Users\Inan\AppData\Local\Programs\ProjectAway\ProjectAway.exe"


def test_frozen_inno_install_is_detected_and_gets_the_setup_asset():
    t = U.detect_install(platform="win32", frozen=True, env={}, executable=INNO_EXE, machine="AMD64")
    assert t.kind == "windows" and t.path == Path(INNO_EXE)
    names = [SETUP, "ProjectAway-0.1.8-linux-x86_64.AppImage", "project_away-0.1.8-py3-none-any.whl", "SHA256SUMS.txt"]
    rel = U.Release("v0.1.8", "0.1.8", tuple(U.Asset(n, f"https://x/{n}", 1) for n in names))
    assert U.select_asset(rel, t).name == SETUP  # type: ignore[union-attr]


def test_windows_command_adds_dir_and_log_and_quotes_spaces():
    d, lg = Path(r"C:\Users\A B\Programs\ProjectAway"), Path("C:/L/u.log")
    cmd = U.windows_command(Path("C:/T/setup.exe"), True, d, lg)
    assert cmd[-2] == "/DIR=" + str(d) and cmd[-1] == "/LOG=" + str(lg)
    assert "/CLOSEAPPLICATIONS" in cmd and "/AUTOUPDATE=1" in cmd
    assert U.win_arg(r"/DIR=C:\A B") == r'/DIR="C:\A B"' and U.win_arg("/VERYSILENT") == "/VERYSILENT"
    assert U.win_arg(r"C:\A B\s.exe") == r'"C:\A B\s.exe"'


class _Launcher:
    def __init__(self, fail: int = 0) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.fail = fail

    def __call__(self, cmd: Any, **kw: Any) -> Any:
        self.calls.append((cmd, kw))
        if len(self.calls) <= self.fail:
            raise OSError(5, "Access is denied")
        return types.SimpleNamespace(pid=99)


def test_windows_install_waits_for_the_app_then_installs_and_records_exit_code(tmp_path):
    setup = tmp_path / "updates" / SETUP
    setup.parent.mkdir()
    setup.write_bytes(b"x")
    data = tmp_path / "data"
    launch = _Launcher()
    tgt = U.InstallTarget("windows", Path(r"C:\Users\A B\AppData\Local\Programs\ProjectAway\ProjectAway.exe"))
    U.write_state(data, "0.1.7", "0.1.8", SETUP, setup)
    out = U.install(setup, tgt, pid=4321, launcher=launch, platform="win32", data_dir=data)
    assert out.exit_app and len(launch.calls) == 1
    cmd, kw = launch.calls[0]
    assert "cmd.exe" in cmd and "/d /s /c" in cmd and U.HELPER_SCRIPT in cmd  # v0.1.10: batch, not PowerShell
    assert kw["creationflags"] & U.CREATE_BREAKAWAY_FROM_JOB and kw["creationflags"] & U.CREATE_NO_WINDOW
    assert not kw["creationflags"] & U.DETACHED_PROCESS
    script = (setup.parent / U.HELPER_SCRIPT).read_text(encoding="utf-8")
    assert script.index('"PID eq 4321"') < script.index('start "" /wait')
    assert 'taskkill /F /IM "%APPEXE%"' in script and 'set "APPEXE=ProjectAway.exe"' in script
    assert '/DIR="C:\\Users\\A B\\AppData\\Local\\Programs\\ProjectAway"' in script
    assert f'/LOG="{data / "logs" / U.INSTALL_LOG}"' in script
    assert f'set "EXITFILE={data / U.EXIT_CODE_FILE}"' in script and '>"%EXITFILE%" echo %RC%' in script
    assert (data / "logs").is_dir()
    assert json.loads((data / U.STATE_FILE).read_text())["method"] == "batch helper (breakaway)"


def test_windows_launch_falls_back_without_breakaway_then_direct_installer(tmp_path):
    setup = tmp_path / SETUP
    setup.write_bytes(b"x")
    tgt = U.InstallTarget("windows", Path(INNO_EXE))
    launch = _Launcher(fail=1)  # a job that forbids breakaway rejects the flag
    assert U.launch_windows_installer(setup, tgt, 1, True, launch, tmp_path, "win32") == "batch helper"
    assert not launch.calls[1][1]["creationflags"] & U.CREATE_BREAKAWAY_FROM_JOB
    launch = _Launcher(fail=2)  # cmd.exe cannot be started at all
    assert U.launch_windows_installer(setup, tgt, 1, True, launch, tmp_path, "win32") == "installer direct (breakaway)"
    cmd, kw = launch.calls[2]
    assert isinstance(cmd, str) and cmd.startswith(f'"{setup}" /VERYSILENT') and "/AUTOUPDATE=1" in cmd
    assert kw["creationflags"] & U.DETACHED_PROCESS and not kw["creationflags"] & U.CREATE_NO_WINDOW
    with pytest.raises(U.UpdateError, match="could not start the installer"):
        U.launch_windows_installer(setup, tgt, 1, True, _Launcher(fail=9), tmp_path, "win32")


def test_previous_outcome_reports_success_and_failure_once(tmp_path):
    U.write_state(tmp_path, "0.1.7", "0.1.8", SETUP)
    assert U.previous_outcome(tmp_path, "0.1.8") == (True, "0.1.8", "Project Away was updated to v0.1.8.")
    assert U.previous_outcome(tmp_path, "0.1.8") is None  # reported once

    U.write_state(tmp_path, "0.1.7", "0.1.8", SETUP)
    (tmp_path / U.EXIT_CODE_FILE).write_text("\ufeff5\r\n", encoding="utf-8")
    ok, ver, msg = U.previous_outcome(tmp_path, "0.1.7")  # type: ignore[misc]
    assert not ok and ver == "0.1.8" and "still v0.1.7" in msg and "exit code 5" in msg
    assert U.INSTALL_LOG in msg and not (tmp_path / U.EXIT_CODE_FILE).exists()

    U.write_state(tmp_path, "0.1.7", "0.1.8", SETUP)
    assert "the installer did not run" in U.previous_outcome(tmp_path, "0.1.7")[2]  # type: ignore[index]


def test_manager_reports_failed_previous_install_and_blocks_auto_loop(tmp_path):
    U.write_state(tmp_path, "0.1.7", "0.1.8", SETUP)
    notes: list[tuple[str, str]] = []
    m = U.UpdateManager(
        C.UpdatesConfig(check=False), notify=lambda t, s: notes.append((t, s)), data_dir=tmp_path, current="0.1.7"
    )
    m.start(report_delay_s=0.0)
    t0 = time.monotonic()
    while not notes and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    assert notes and notes[0][0] == "Project Away update failed" and m.failed_version == "0.1.8"


def _ready(version: str = "0.1.8") -> U.CheckResult:
    asset = U.Asset(SETUP, "https://dl/x", 3, "sha256:" + "0" * 64)
    rel = U.Release(f"v{version}", version, (asset,))
    return U.CheckResult("0.1.7", version, True, rel, asset, U.InstallTarget("windows", Path(INNO_EXE)))


def test_install_failure_is_notified_and_resets_installing(tmp_path):
    notes: list[tuple[str, str]] = []
    quits: list[int] = []

    def bad(r):
        raise U.UpdateError("ProjectAway-0.1.8-windows-x64-setup.exe: checksum mismatch")

    m = U.UpdateManager(
        C.UpdatesConfig(),
        notify=lambda t, s: notes.append((t, s)),
        quit_app=lambda: quits.append(1),
        checker=_ready,
        installer=bad,
        data_dir=tmp_path,
    )
    m.tick(0.0)
    assert m.install_now() is None
    assert m.installing is False and not m.exiting and quits == []
    assert notes[-1] == ("Project Away update failed", "ProjectAway-0.1.8-windows-x64-setup.exe: checksum mismatch")
    assert m.menu_label() == "Update to v0.1.8"  # the user can retry from the menu


def test_install_success_quits_and_quit_errors_are_reported(tmp_path):
    notes: list[tuple[str, str]] = []

    def quit_app() -> None:
        raise RuntimeError("tray loop is gone")

    m = U.UpdateManager(
        C.UpdatesConfig(),
        notify=lambda t, s: notes.append((t, s)),
        quit_app=quit_app,
        checker=_ready,
        installer=lambda r: U.InstallOutcome(True, "Installing"),
        data_dir=tmp_path,
    )
    m.tick(0.0)
    assert m.install_now() == U.InstallOutcome(True, "Installing")
    assert m.exiting and m.installing is False
    assert "Quit Project Away to finish the update" in notes[-1][1]
    assert m.install_now() is None  # never a second installer while quitting


def test_download_and_install_uses_data_dir_verifies_and_writes_state(tmp_path, monkeypatch):
    import hashlib

    payload = b"setup-bytes"
    asset = U.Asset(SETUP, "https://dl/x", len(payload), "sha256:" + hashlib.sha256(payload).hexdigest())
    r = U.CheckResult(
        "0.1.7",
        "0.1.8",
        True,
        U.Release("v0.1.8", "0.1.8", (asset,)),
        asset,
        U.InstallTarget("windows", Path(INNO_EXE)),
    )

    class Resp:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.data = [payload, b""]

        def read(self, n: int = -1) -> bytes:
            return self.data.pop(0)

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

    monkeypatch.setattr(U, "_default_opener", lambda url, timeout: Resp())
    calls: list[Any] = []
    monkeypatch.setattr(
        U, "install", lambda path, target, data_dir=None: calls.append((path, data_dir)) or U.InstallOutcome(True, "ok")
    )
    (tmp_path / "updates").mkdir()
    (tmp_path / "updates" / "ProjectAway-0.1.6-windows-x64-setup.exe").write_bytes(b"old")
    m = U.UpdateManager(C.UpdatesConfig(), data_dir=tmp_path, current="0.1.7")
    with caplog_at_info() as records:
        assert m._download_and_install(r) == U.InstallOutcome(True, "ok")
    path, ddir = calls[0]
    assert path == tmp_path / "updates" / SETUP and ddir == tmp_path
    assert sorted(p.name for p in (tmp_path / "updates").iterdir()) == [SETUP]  # old installer cleaned
    state = json.loads((tmp_path / U.STATE_FILE).read_text())
    assert state["from"] == "0.1.7" and state["to"] == "0.1.8" and state["asset"] == SETUP
    text = "\n".join(records)
    assert "downloading" in text and "SHA-256 verified (source: api)" in text

    bad = U.CheckResult("0.1.7", "0.1.8", True, r.release, U.Asset(SETUP, "u", 0, "sha256:" + "1" * 64), r.target)
    (tmp_path / U.STATE_FILE).unlink()
    with pytest.raises(U.UpdateError, match="checksum mismatch"):
        m._download_and_install(bad)
    assert not (tmp_path / U.STATE_FILE).exists()  # nothing to report next launch: no install was attempted


class caplog_at_info:
    def __enter__(self) -> list[str]:
        self.records: list[str] = []
        self.h = logging.Handler()
        self.h.emit = lambda rec: self.records.append(rec.getMessage())  # type: ignore[method-assign]
        lg = logging.getLogger("away.updater")
        self.old = lg.level
        lg.setLevel(logging.INFO)
        lg.addHandler(self.h)
        return self.records

    def __exit__(self, *a: Any) -> None:
        lg = logging.getLogger("away.updater")
        lg.removeHandler(self.h)
        lg.setLevel(self.old)


def test_exit_after_update_forces_exit_even_if_shutdown_hangs():
    timers: list[Any] = []
    exits: list[int] = []

    class FakeTimer:
        def __init__(self, s: float, fn: Any) -> None:
            self.s, self.fn, self.daemon, self.started = s, fn, False, False
            timers.append(self)

        def start(self) -> None:
            self.started = True

    def shutdown() -> None:
        raise RuntimeError("camera thread stuck")

    U.exit_after_update(shutdown, grace_s=8.0, exit_fn=exits.append, timer=FakeTimer)
    assert timers[0].started and timers[0].daemon and timers[0].s == 8.0  # armed before the shutdown ran
    timers[0].fn()  # the grace period passed and the process is still alive
    assert exits == [0]


def test_failed_check_is_retried_sooner_and_manual_check_always_answers(tmp_path):
    results = [U.CheckResult("0.1.7", error="could not reach GitHub (offline?)")]
    calls = {"n": 0}

    def checker() -> U.CheckResult:
        calls["n"] += 1
        return results[-1]

    notes: list[tuple[str, str]] = []
    m = U.UpdateManager(
        C.UpdatesConfig(),
        notify=lambda t, s: notes.append((t, s)),
        checker=checker,
        interval_s=86400,
        error_retry_s=1800,
        data_dir=tmp_path,
    )
    m.tick(0.0)
    m.tick(1700.0)
    assert calls["n"] == 1
    m.tick(1800.0)  # offline at sign-in: retried after 30 min, not 24 h
    assert calls["n"] == 2
    m.check_now()
    assert notes[-1] == ("Project Away update check failed", "could not reach GitHub (offline?)")
    results.append(U.CheckResult("0.1.7", latest="0.1.7"))
    m.check_now()
    assert notes[-1][1] == "Project Away 0.1.7 is up to date (latest release: 0.1.7)"
    results.append(_ready())
    m.check_now()
    assert notes[-1][1] == "v0.1.8 is available. Use the tray menu to update." and m.menu_label() == "Update to v0.1.8"


def test_same_version_rebuild_is_not_offered_again():
    """Version is the only signal: a rebuilt installer with the same version is never offered (no loop)."""
    asset = U.Asset("ProjectAway-0.1.7-windows-x64-setup.exe", "u", 1, "sha256:" + "0" * 64)
    rel = U.Release("v0.1.7", "0.1.7", (asset,))
    res = U.check("0.1.7", target=U.InstallTarget("windows", Path(INNO_EXE)), fetch=lambda u, t: b"{}")
    assert not res.update_available
    assert not U.is_newer(rel.version, "0.1.7")


def test_setup_logging_writes_info_lines(tmp_path):
    import io

    from away.app.frozen import setup_logging

    buf = io.StringIO()
    lg = logging.getLogger("away")
    before = list(lg.handlers)
    try:
        h = setup_logging(buf)
        assert setup_logging(buf) is h  # idempotent
        logging.getLogger("away.updater").info("update: hello")
        assert "INFO away.updater: update: hello" in buf.getvalue()
    finally:
        for x in list(lg.handlers):
            if x not in before:
                lg.removeHandler(x)


def test_inno_script_closes_running_app_and_relaunches_on_auto_update():
    iss = (ROOT / "packaging" / "windows" / "away.iss").read_text(encoding="utf-8")
    assert "CloseApplications=force" in iss and "RestartApplications=no" in iss
    assert "PrivilegesRequired=lowest" in iss and r"DefaultDirName={localappdata}\Programs\ProjectAway" in iss
    assert "OutputBaseFilename=ProjectAway-{#AppVersion}-windows-x64-setup" in iss
    assert 'Parameters: "tray"; Flags: nowait; Check: IsAutoUpdate' in iss


def test_tray_has_check_for_updates_and_forced_exit():
    src = (ROOT / "src" / "away" / "app" / "tray.py").read_text(encoding="utf-8")
    assert '"Check for updates"' in src and "updates.check_now" in src
    assert "exit_after_update(lambda: on_quit(icon, None))" in src
    assert "if updates.exiting:" in src and "os._exit(0)" in src
