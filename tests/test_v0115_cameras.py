"""v0.1.15: two webcams at the same time on Windows (JoyCam + the built-in "HD User Facing").

The user's report: in Settings > Cameras > Live confidence the built-in camera showed "-" the whole time (its
light came on briefly, then off), and Calibrate failed with "cannot open camera 'HD User Facing' ... DirectShow
1: did not open; Media Foundation 1: did not open". Causes fixed here:

* Settings' live view and `away calibrate-cameras` opened both cameras at the same moment with the default
  opener (no sequential open, no fallbacks); the app used a lock and its own opener. -> one shared start-up
  (:func:`capture.start_camera_threads`): primary first, one after another, the same opener and fallbacks.
* Camera thread errors were never shown ("-"). -> per-camera status cells and the mode that worked.
* Calibrate (a subprocess) and Live confidence (in-process) could run at once and fight over the devices.
  -> mutually exclusive; Calibrate stops the live view and waits until its devices are free.
* The app held the cameras while it probed them (up to 30 s) without saying so, so a lease never asked for
  them, and a probe did not stop when asked. -> owner record while probing; the probe ends on a request.
* New: `away cameras test` / Test both cameras, and per-camera fallback modes kept across starts.
"""

from __future__ import annotations

import argparse
import functools
import json
import queue
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import cli
from away import config as C
from away.app import camera_share as CS
from away.app import settings_gui as SG
from away.multicam import camtest, capture
from away.multicam.capture import CameraOpenError, CameraThread, CaptureMode, FramePairer

JOY, HD = "JoyCam", "HD User Facing"


def _specs(fourcc: str = "MJPG", w: int = 640, h: int = 480) -> list[C.CameraSpec]:
    return [
        C.CameraSpec(device=JOY, role="primary", width=w, height=h, fourcc=fourcc),
        C.CameraSpec(device=HD, role="secondary", width=w, height=h, fourcc=fourcc),
    ]


def _wait(cond: Any, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


# =============================================================================== a simulated USB bus
LOAD = {"YUY2": 2.0, "NV12": 2.0, "": 2.0, "MJPG": 1.0}  # 640x480 bus load per format ("" = driver YUY2)


def _load(mode: CaptureMode) -> float:
    if not mode.width:  # device default: 640x480 YUY2 on these webcams
        return 2.0
    scale = (mode.width * mode.height) / (640 * 480)
    return LOAD.get(mode.fourcc, 2.0) * scale


class Bus:
    """Two USB webcams on one controller, like the user's laptop:

    * a device that is already open cannot be opened again (any process): "did not open" in every backend;
    * the bus carries ``limit`` units: 640x480 YUY2 = 2, 640x480 MJPG = 1, 320x240 = a quarter of that. A mode
      that would exceed it opens but sends no frames. Both at 640x480 YUY2 fail; MJPG / 320x240 succeed.
    """

    def __init__(self, limit: float = 3.0, open_delay: float = 0.02) -> None:
        self.limit = limit
        self.open_delay = open_delay
        self.lock = threading.Lock()
        self.holder: dict[str, str] = {}  # device -> who holds it
        self.modes: dict[str, CaptureMode] = {}
        self.violations: list[str] = []  # a device opened while someone else held it
        self.opening = 0
        self.max_parallel = 0
        self.log: list[tuple[str, str, str]] = []  # (who, device, event)

    def load(self, but: str = "") -> float:
        return sum(_load(m) for d, m in self.modes.items() if d != but)

    def open(self, who: str, device: str, mode: CaptureMode) -> FakeCap:
        with self.lock:
            if device in self.holder:
                self.violations.append(f"{who} opened {device} held by {self.holder[device]}")
                raise CameraOpenError(
                    f"cannot open camera {device!r} (busy, no device, or permission denied; tried DirectShow 1: "
                    "did not open; Media Foundation 1: did not open)"
                )
            if self.load(device) + _load(mode) > self.limit:
                raise CameraOpenError(f"camera {device!r} opened but sent no frames", opened=True)
            self.holder[device] = who
            self.modes[device] = mode
            self.log.append((who, device, "open"))
        return FakeCap(self, device, mode)

    def close(self, device: str) -> None:
        with self.lock:
            who = self.holder.pop(device, "?")
            self.modes.pop(device, None)
            self.log.append((who, device, "close"))

    def opener(self, who: str, fallbacks: bool = True) -> Any:
        """An opener like :func:`capture.open_capture`: the real :func:`capture.capture_modes` order (a secondary
        may go down to 320x240), the first mode the bus accepts wins; a busy device fails at once."""

        def op(spec: C.CameraSpec) -> FakeCap:
            with self.lock:
                self.opening += 1
                self.max_parallel = max(self.max_parallel, self.opening)
            try:
                time.sleep(self.open_delay)
                modes = capture.capture_modes(
                    spec.width, spec.height, spec.fps, spec.fourcc, low_res=spec.role != "primary"
                )
                last: Exception | None = None
                for m in modes if fallbacks else modes[:1]:
                    try:
                        return self.open(who, str(spec.device), m)
                    except CameraOpenError as exc:
                        if not exc.opened:
                            raise  # busy: no mode can help
                        last = exc
                raise last or CameraOpenError("no modes")
            finally:
                with self.lock:
                    self.opening -= 1

        return op


class FakeCap:
    def __init__(self, bus: Bus, device: str, mode: CaptureMode) -> None:
        self.bus, self.device, self.mode, self.backend = bus, device, mode, "DirectShow"
        self.released = False

    def read(self) -> tuple[bool, Any]:
        time.sleep(1 / 120)
        return True, np.zeros((self.mode.height or 480, self.mode.width or 640, 3), np.uint8)

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.bus.close(self.device)


_noconv = functools.partial(CameraThread, convert=lambda f, m: f)


def _starter(bus: Bus, who: str, fallbacks: bool = True, settle_s: float = 2.0) -> Any:
    """:func:`capture.start_camera_threads` as the call sites use it, with the simulated bus as the device."""
    return functools.partial(
        capture.start_camera_threads,
        serial=True,
        opener=bus.opener(who, fallbacks),
        thread_factory=_noconv,
        settle_s=settle_s,
    )


class FakeApp:
    """The tray app holding both cameras, handing them over on a CameraShare request and taking them back."""

    def __init__(self, folder: Path, bus: Bus, specs: list[C.CameraSpec]) -> None:
        self.share = CS.CameraShare(folder, pid=1)
        self.bus, self.specs = bus, specs
        self.stop = threading.Event()
        self.caps: list[FakeCap] = []
        self.acquired = 0
        self.handovers = 0
        self.th = threading.Thread(target=self.run, daemon=True)

    def _open_all(self) -> None:
        op = self.bus.opener("app")
        self.caps = [op(s) for s in self.specs]
        self.acquired += 1

    def run(self) -> None:
        self._open_all()
        while not self.stop.is_set():
            self.share.publish_owner([s.device for s in self.specs])
            if self.share.requested():
                for c in self.caps:
                    c.release()
                self.caps = []
                self.share.clear_owner()
                self.share.ack_released([s.device for s in self.specs])
                self.handovers += 1
                while self.share.requested() and not self.stop.is_set():
                    time.sleep(0.02)
                self.share.clear_released()
                if not self.stop.is_set():
                    self._open_all()
            time.sleep(0.02)
        for c in self.caps:
            c.release()
        self.share.clear_owner()


class _Tracker:
    def process(self, rgb: Any, t: float) -> Any:
        return types.SimpleNamespace(landmarks=None, handedness=None, score=0.0)


def _model(specs: list[C.CameraSpec]) -> Any:
    cfg = C.AppConfig()
    cfg.hands.cameras = specs
    return types.SimpleNamespace(cfg=cfg)


def _leases(folder: Path, pid: int = 2) -> CS.SharedLease:
    return CS.SharedLease(lambda who: CS.CameraLease(CS.CameraShare(folder, pid=pid), 3.0, who, heartbeat_s=0.05))


# =============================================================================== 1. shared start-up
class _Recorder:
    def __init__(self, delay: float = 0.05, fail: set[str] | None = None) -> None:
        self.lock = threading.Lock()
        self.events: list[tuple[str, str, float]] = []
        self.opening = 0
        self.max_parallel = 0
        self.delay = delay
        self.fail = fail or set()

    def __call__(self, spec: C.CameraSpec) -> Any:
        with self.lock:
            self.opening += 1
            self.max_parallel = max(self.max_parallel, self.opening)
            self.events.append(("start", str(spec.device), time.monotonic()))
        try:
            time.sleep(self.delay)
            if str(spec.device) in self.fail:
                raise CameraOpenError(f"cannot open camera {spec.device!r} (busy; tried DirectShow 1: did not open)")
            return FakeCap(Bus(limit=99), str(spec.device), CaptureMode(spec.width, spec.height, spec.fps, "MJPG"))
        finally:
            with self.lock:
                self.opening -= 1
                self.events.append(("end", str(spec.device), time.monotonic()))


def test_shared_startup_opens_one_after_another_primary_first_even_when_listed_second() -> None:
    rec = _Recorder()
    specs = list(reversed(_specs()))  # secondary listed first
    stop = threading.Event()
    pairer = FramePairer(2, 1, 0.04, 1.0)
    ths = capture.start_camera_threads(specs, pairer, stop, serial=True, opener=rec, thread_factory=_noconv)
    assert _wait(lambda: all(t.frames > 0 for t in ths))
    assert capture.stop_camera_threads(stop, ths)
    assert rec.max_parallel == 1
    starts = [d for kind, d, _t in rec.events if kind == "start"]
    assert starts == [JOY, HD]  # the primary always first
    assert all(t.state == "stopped" and not t.is_alive() for t in ths)


def test_secondary_waits_for_the_primarys_first_frame_even_without_a_lock() -> None:
    rec = _Recorder(delay=0.15)
    stop = threading.Event()
    ths = capture.start_camera_threads(_specs(), FramePairer(2), stop, serial=False, opener=rec, thread_factory=_noconv)
    assert ths[1].after is ths[0].ready and ths[0].after is None
    assert _wait(lambda: all(t.frames > 0 for t in ths))
    capture.stop_camera_threads(stop, ths)
    first_frame = ths[0].first_t
    hd_start = next(t for kind, d, t in rec.events if kind == "start" and d == HD)
    assert first_frame is not None and hd_start >= first_frame
    assert rec.max_parallel == 1


def test_secondary_opens_after_the_settle_delay_when_the_primary_is_slow_or_fails() -> None:
    # primary fails: the secondary does not wait for the settle delay at all
    rec = _Recorder(delay=0.01, fail={JOY})
    stop = threading.Event()
    ths = capture.start_camera_threads(
        _specs(), FramePairer(2), stop, serial=False, opener=rec, thread_factory=_noconv, settle_s=30.0
    )
    assert _wait(lambda: ths[1].frames > 0, 3.0)
    capture.stop_camera_threads(stop, ths)
    assert ths[0].state == "failed" and "did not open" in (ths[0].error or "")

    # primary hangs while opening: the secondary goes after settle_s
    gate = threading.Event()

    def slow(spec: C.CameraSpec) -> Any:
        if spec.role == "primary":
            gate.wait(5)
        return FakeCap(Bus(limit=99), str(spec.device), CaptureMode(320, 240, 15, "MJPG"))

    stop2 = threading.Event()
    t0 = time.monotonic()
    ths2 = capture.start_camera_threads(
        _specs(), FramePairer(2), stop2, serial=False, opener=slow, thread_factory=_noconv, settle_s=0.3
    )
    assert _wait(lambda: ths2[1].frames > 0, 3.0)
    assert time.monotonic() - t0 >= 0.25
    gate.set()
    capture.stop_camera_threads(stop2, ths2)


def test_a_waiting_secondary_stops_without_opening() -> None:
    gate = threading.Event()
    opened: list[str] = []

    def op(spec: C.CameraSpec) -> Any:
        opened.append(str(spec.device))
        gate.wait(5)
        return FakeCap(Bus(limit=99), str(spec.device), CaptureMode(640, 480, 30, "MJPG"))

    stop = threading.Event()
    ths = capture.start_camera_threads(
        _specs(), FramePairer(2), stop, serial=True, opener=op, thread_factory=_noconv, settle_s=30
    )
    assert _wait(lambda: opened == [JOY])
    stop.set()
    gate.set()
    assert capture.stop_camera_threads(stop, ths, timeout=3.0)
    assert opened == [JOY] and ths[1].state == "stopped" and ths[1].error is None


def test_default_opener_is_open_capture_with_the_apps_fallbacks() -> None:
    op = capture.camera_opener(virtual_fallback=False)
    assert op.func is capture.open_capture and op.keywords == {"virtual_fallback": False}
    low = capture.camera_opener(True, low_res=True)
    assert low.keywords == {"virtual_fallback": True, "low_res": True}
    ths = capture.start_camera_threads(_specs(), FramePairer(2), threading.Event(), virtual_fallback=False, start=False)
    assert ths[0].opener.func is capture.open_capture and ths[0].opener.keywords == {"virtual_fallback": False}
    assert ths[0].opener is ths[1].opener


def test_serial_default_follows_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "_windows", lambda: True)
    ths = capture.start_camera_threads(_specs(), FramePairer(2), threading.Event(), start=False)
    assert ths[0].open_lock is not None and ths[0].open_lock is ths[1].open_lock
    monkeypatch.setattr(capture, "_windows", lambda: False)
    ths = capture.start_camera_threads(_specs(), FramePairer(2), threading.Event(), start=False)
    assert ths[0].open_lock is None
    lock = threading.Lock()
    ths = capture.start_camera_threads(_specs(), FramePairer(2), threading.Event(), open_lock=lock, start=False)
    assert ths[1].open_lock is lock


def test_all_three_call_sites_use_the_shared_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    bus = Bus(limit=99)
    real = capture.start_camera_threads

    def spy(specs: list[Any], pairer: Any, stop: Any, **kw: Any) -> list[Any]:
        calls.append(([str(s.device) for s in specs], kw))
        return real(specs, pairer, stop, serial=True, opener=bus.opener("spy"), thread_factory=_noconv)

    monkeypatch.setattr(capture, "start_camera_threads", spy)

    # (1) the app
    from away.app.runner import AwayApp

    app = AwayApp.__new__(AwayApp)
    app.cfg = C.AppConfig()
    app.cfg.hands.virtual_fallback = False
    stop = threading.Event()
    ths = app._start_cameras(_specs(), FramePairer(2), stop, threading.Lock())
    capture.stop_camera_threads(stop, ths)
    assert calls[-1][1]["serial"] is True and calls[-1][1]["virtual_fallback"] is False
    assert calls[-1][1]["open_lock"] is not None

    # (2) Settings > Live confidence (default start)
    out: queue.Queue[Any] = queue.Queue()
    stop2 = threading.Event()
    w = threading.Thread(
        target=SG.camera_confidence_worker,
        args=(_model(_specs()), out, stop2),
        kwargs={"leases": _leases(tmp_path), "trackers": [_Tracker(), _Tracker()], "sleep": lambda s: time.sleep(s)},
    )
    monkeypatch.setattr(capture, "default_mode_file", lambda: tmp_path / "modes.json")
    w.start()
    assert _wait(lambda: len(calls) == 2)
    stop2.set()
    w.join(5)
    assert not w.is_alive()

    # (3) away calibrate-cameras
    stop3 = threading.Event()
    ths3 = cli.open_calibration_cameras(_specs(), FramePairer(2), stop3, virtual_fallback=True, wait_s=2.0)
    capture.stop_camera_threads(stop3, ths3)
    # v0.1.16: the calibration start-up also passes the minimum frame rate (None here: the default)
    assert len(calls) == 3 and calls[2][1] == {"virtual_fallback": True, "min_fps": None}

    # (4) away cameras test
    camtest.stream_cameras(_specs(), 0.1)
    assert len(calls) == 4
    assert all(c[0] == [JOY, HD] for c in calls)


# =============================================================================== 2. error surfacing
def test_camera_thread_surfaces_errors_state_mode_and_fps() -> None:
    bad = CameraThread(
        1, _specs()[1], FramePairer(2), threading.Event(), opener=_Recorder(0.0, fail={HD}), convert=lambda f, m: f
    )
    bad.start()
    bad.join(2)
    assert bad.state == "failed" and "did not open" in (bad.error or "") and bad.ready.is_set()

    stop = threading.Event()
    cap = capture.FirstFrameCapture(
        FakeCap(Bus(99), HD, CaptureMode(320, 240, 15, "MJPG")), None, CaptureMode(320, 240, 15, "MJPG"), "DirectShow"
    )
    good = CameraThread(1, _specs()[1], FramePairer(2), stop, opener=lambda s: cap, convert=lambda f, m: f)
    good.start()
    assert _wait(lambda: good.frames > 5)
    stop.set()
    good.join(2)
    assert good.describe_mode() == "320x240@15 MJPG (DirectShow)"
    assert good.fps() > 10 and good.state == "stopped"


def test_camera_thread_without_frames_says_so() -> None:
    class Silent:
        def read(self) -> tuple[bool, Any]:
            return False, None

        def release(self) -> None:
            pass

    th = CameraThread(0, _specs()[0], FramePairer(2), threading.Event(), first_frame_s=0.1, opener=lambda s: Silent())
    th.start()
    th.join(3)
    assert th.state == "failed" and "no frame" in (th.error or "")
    assert SG.live_cells([th]) == ["no frames"]


def test_live_cells_and_status_line() -> None:
    def t(**kw: Any) -> Any:
        base = {"error": None, "state": "waiting", "frames": 0, "spec": C.CameraSpec(device=HD)}
        base.update(kw)
        ns = types.SimpleNamespace(**base)
        ns.describe_mode = lambda: kw.get("mode", "")
        return ns

    cells = SG.live_cells(
        [
            t(state="opening"),
            t(state="streaming", frames=9),
            t(state="open"),
            t(state="failed", error="cannot open camera 'HD User Facing' (busy; tried DirectShow 1: did not open)"),
            t(state="streaming", frames=3),
        ],
        [None, 0.5, None, None, None],
    )
    assert cells == ["opening\u2026", "#####..... 0.50", "no frames", "error: busy / did not open", "streaming"]
    assert SG.short_camera_error("camera opened but sent no frame within 10 s") == "no frames"
    assert SG.short_camera_error("x" * 50).endswith("\u2026") and len(SG.short_camera_error("x" * 50)) < 45
    line = SG.live_status_line([t(spec=C.CameraSpec(device=JOY), mode="640x480 MJPG (DirectShow)"), t(error="x")])
    assert line == "live: JoyCam 640x480 MJPG (DirectShow) | HD User Facing failed"


def test_live_worker_reports_each_camera_and_releases_everything(tmp_path: Path) -> None:
    bus = Bus(limit=99)
    bus_fail = _Recorder(0.0, fail={HD})

    def start(specs: list[Any], pairer: Any, stop: Any, **kw: Any) -> list[Any]:
        def op(spec: C.CameraSpec) -> Any:
            return bus_fail(spec) if spec.role == "secondary" else bus.opener("live")(spec)

        return capture.start_camera_threads(specs, pairer, stop, serial=True, opener=op, thread_factory=_noconv)

    leases = _leases(tmp_path)
    out: queue.Queue[Any] = queue.Queue()
    stop = threading.Event()
    w = threading.Thread(
        target=SG.camera_confidence_worker,
        args=(_model(_specs()), out, stop),
        kwargs={"leases": leases, "start": start, "trackers": [_Tracker(), _Tracker()], "report_s": 0.05},
    )
    w.start()
    seen: list[dict[str, Any]] = []

    def got_both() -> bool:
        while not out.empty():
            seen.append(out.get_nowait())
        return any(
            isinstance(m, dict) and m["cells"][0].endswith("0.00") and m["cells"][1].startswith("error") for m in seen
        )

    assert _wait(got_both)
    assert leases.held  # the cameras are borrowed while the live view runs
    last = [m for m in seen if isinstance(m, dict)][-1]
    assert last["cells"][1] == "error: busy / did not open"
    assert "JoyCam 640x480@30 MJPG (DirectShow)" in last["status"] and "HD User Facing failed" in last["status"]
    stop.set()
    w.join(5)
    assert not w.is_alive() and not leases.held and bus.holder == {}  # devices free, lease given back


def test_live_worker_reports_a_fatal_error_as_text(tmp_path: Path) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("cannot open camera 'JoyCam'")

    out: queue.Queue[Any] = queue.Queue()
    leases = _leases(tmp_path)
    SG.camera_confidence_worker(
        _model(_specs()), out, threading.Event(), leases=leases, start=boom, trackers=[_Tracker(), _Tracker()]
    )
    msg = out.get_nowait()
    assert isinstance(msg, str) and "cannot open" in msg
    assert not leases.held


# =============================================================================== 3. mutual exclusion
def test_camera_tools_are_mutually_exclusive() -> None:
    tools = SG.CameraTools()
    assert tools.states(True) == {"calibrate": "normal", "live": "normal", "test": "normal"}
    assert tools.begin("live") is None
    assert tools.states(True) == {"calibrate": "normal", "live": "normal", "test": "disabled"}  # live: Stop
    assert "Live confidence is running" in (tools.begin("test") or "")
    assert tools.begin("calibrate") is None  # calibrate takes over (it stops the live view first)
    assert tools.states(True) == {"calibrate": "disabled", "live": "disabled", "test": "disabled"}
    assert "calibration is running" in (tools.begin("live") or "")
    assert "calibration is running" in (tools.begin("test") or "")
    tools.end("live")  # the old live view finishing does not end the calibration
    assert tools.busy == "calibrate"
    tools.end("calibrate")
    assert tools.begin("test") is None
    assert tools.begin("calibrate") and tools.begin("live")
    tools.end("test")
    assert tools.states(False) == {"calibrate": "disabled", "live": "disabled", "test": "normal"}


def test_calibrate_stops_live_view_and_waits_for_the_devices_before_spawning(tmp_path: Path) -> None:
    bus = Bus(limit=99)
    leases = _leases(tmp_path)
    out: queue.Queue[Any] = queue.Queue()
    stop = threading.Event()
    live = threading.Thread(
        target=SG.camera_confidence_worker,
        args=(_model(_specs()), out, stop),
        kwargs={"leases": leases, "start": _starter(bus, "live"), "trackers": [_Tracker(), _Tracker()]},
    )
    live.start()
    assert _wait(lambda: set(bus.holder) == {JOY, HD})
    order: list[str] = []
    at_spawn: dict[str, Any] = {}

    def spawn() -> dict[str, Any]:
        order.append("spawn")
        at_spawn.update(holder=dict(bus.holder), live_alive=live.is_alive(), held=leases.held)
        return {"rc": 0, "err": ""}

    def stop_live() -> bool:
        order.append("stop-live")
        assert leases.held  # the cameras stay borrowed across the hand-over (the app cannot grab them)
        return SG.stop_live_view(stop, live, timeout=5.0)

    res = SG.calibration_flow(leases, stop_live, spawn)
    assert res == {"rc": 0, "err": ""} and order == ["stop-live", "spawn"]
    assert at_spawn == {"holder": {}, "live_alive": False, "held": True}
    assert not leases.held


def test_calibrate_does_not_spawn_when_live_view_will_not_let_go(tmp_path: Path) -> None:
    leases = _leases(tmp_path)
    spawned: list[int] = []
    res = SG.calibration_flow(leases, lambda: False, lambda: spawned.append(1) or {"rc": 0})
    assert res["rc"] == -1 and "did not release" in res["err"] and spawned == [] and not leases.held

    hung = threading.Thread(target=time.sleep, args=(1.0,), daemon=True)
    hung.start()
    assert not SG.stop_live_view(threading.Event(), hung, timeout=0.05)
    assert SG.stop_live_view(threading.Event(), None)


def test_shared_lease_is_reference_counted(tmp_path: Path) -> None:
    opened: list[str] = []
    closed: list[int] = []

    class L:
        app_running = handed_over = False

        def __init__(self, who: str) -> None:
            opened.append(who)

        def open(self) -> L:
            return self

        def close(self) -> None:
            closed.append(1)

    sl = CS.SharedLease(L)  # type: ignore[arg-type]
    a = sl.acquire("live")
    b = sl.acquire("calibrate")
    assert a is b and opened == ["live"]
    sl.release()
    assert sl.held and closed == []
    sl.release()
    sl.release()  # extra releases are ignored
    assert not sl.held and closed == [1]


# =============================================================================== 4. calibrate CLI
def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(status_file=str(tmp_path / "status.json"), config=None)


def test_calibrate_borrows_the_cameras_from_the_running_app_and_gives_them_back(tmp_path: Path) -> None:
    from away import paths

    data = Path(paths.data_dir())
    app = CS.CameraShare(data, pid=999_001)
    app.publish_owner([JOY, HD])
    done = threading.Event()

    def app_side() -> None:  # the tray app: hands over on request
        while not done.is_set():
            if app.requested():
                app.clear_owner()
                app.ack_released([JOY, HD])
                return
            app.publish_owner([JOY, HD])
            time.sleep(0.02)

    th = threading.Thread(target=app_side, daemon=True)
    th.start()
    seen: dict[str, Any] = {}

    def fake_run(cfg: Any, args: Any) -> int:
        req = json.loads((data / CS.REQUEST_FILE).read_text())
        seen.update(owner=req["owner"], released=(data / CS.RELEASED_FILE).exists())
        return 0

    assert cli.calibrate_with_lease(C.AppConfig(), _args(tmp_path), run=fake_run) == 0
    done.set()
    assert seen == {"owner": "calibrate", "released": True}
    assert not (data / CS.REQUEST_FILE).exists()  # handed back


def test_calibrate_stops_with_a_clear_message_when_the_app_keeps_the_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from away import paths

    monkeypatch.setattr(cli, "CALIBRATE_LEASE_S", 0.3)
    app = CS.CameraShare(Path(paths.data_dir()), pid=999_002)
    app.publish_owner([JOY, HD])  # never acknowledges
    ran: list[int] = []
    rc = cli.calibrate_with_lease(C.AppConfig(), _args(tmp_path), run=lambda c, a: ran.append(1) or 0)
    assert rc == 2 and ran == []
    assert "did not hand them over in time" in capsys.readouterr().err
    st = json.loads((tmp_path / "status.json").read_text())
    assert st["state"] == "failed" and "pause or quit" in st["message"]


def test_cmd_calibrate_cameras_goes_through_the_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = C.AppConfig()
    cfg.hands.cameras = _specs()
    cfg_path = tmp_path / "config.toml"
    C.save(cfg, cfg_path)
    who: list[str] = []
    real = CS.borrowed_cameras

    def spy(share: Any = None, timeout: float = 4.0, who_: str = "settings", **kw: Any) -> Any:
        who.append(kw.get("who", who_))
        return real(share, timeout, kw.get("who", who_))

    monkeypatch.setattr(CS, "borrowed_cameras", spy)
    monkeypatch.setattr(cli, "_run_camera_calibration", lambda c, a: 0)
    args = argparse.Namespace(
        print_board=None, config=str(cfg_path), status_file=str(tmp_path / "s.json"), checkerboard="9x6"
    )
    assert cli.cmd_calibrate_cameras(args) == 0
    assert who == ["calibrate"]


def test_lease_inside_a_settings_lease_leaves_the_parents_request_alone(tmp_path: Path) -> None:
    parent = CS.CameraLease(CS.CameraShare(tmp_path, pid=10), heartbeat_s=0.05, who="settings-calibrate").open()
    child = CS.CameraLease(CS.CameraShare(tmp_path, pid=11), who="calibrate").open()
    assert child.nested and not child.blocked
    child.close()
    assert CS.CameraShare(tmp_path, pid=99).requested()  # still held for Settings
    parent.close()
    assert not CS.CameraShare(tmp_path, pid=99).requested()


def test_lease_requests_even_when_no_app_runs(tmp_path: Path) -> None:
    lease = CS.CameraLease(CS.CameraShare(tmp_path, pid=2), timeout=0.2, heartbeat_s=0.05).open()
    assert not lease.app_running and not lease.handed_over and not lease.blocked
    assert CS.CameraShare(tmp_path, pid=1).requested()  # an app starting now leaves the cameras alone
    lease.close()
    assert not (tmp_path / CS.REQUEST_FILE).exists()


def _thread(frames: int, alive: bool, error: str | None = None) -> Any:
    ns = types.SimpleNamespace(frames=frames, error=error, spec=None)
    ns.is_alive = lambda: alive
    return ns


def test_calibrate_retries_a_failed_secondary_once_at_low_res_after_the_primary_is_open() -> None:
    specs = _specs()
    prim, sec = _thread(5, True), _thread(0, False, "cannot open camera 'HD User Facing': did not open")
    used: list[C.CameraSpec] = []

    def retry(spec: C.CameraSpec) -> Any:
        used.append(spec)
        return FakeCap(Bus(99), str(spec.device), CaptureMode(320, 240, 15, "MJPG"))

    stop = threading.Event()
    said: list[str] = []
    ths = cli.open_calibration_cameras(
        specs,
        FramePairer(2),
        stop,
        start=lambda *a, **k: [prim, sec],
        retry_opener=retry,
        wait_s=1.0,
        say=said.append,
    )
    assert ths[0] is prim and ths[1] is not sec and isinstance(ths[1], CameraThread)
    assert [(s.width, s.height, s.fps, s.fourcc) for s in used] == [(320, 240, 15, "MJPG")]
    assert specs[1].width == 640  # the configured spec is not changed
    assert "retrying it at 320x240 MJPG" in said[0]
    stop.set()
    ths[1].join(2)


def test_calibrate_does_not_retry_when_the_primary_failed_or_the_secondary_works() -> None:
    calls: list[int] = []
    for pair in ([_thread(0, False, "x"), _thread(0, False, "y")], [_thread(3, True), _thread(4, True)]):
        ths = cli.open_calibration_cameras(
            _specs(),
            FramePairer(2),
            threading.Event(),
            start=lambda *a, p=pair, **k: list(p),
            retry_opener=lambda s: calls.append(1),
            wait_s=0.2,
            say=lambda m: None,
        )
        assert ths == pair
    assert calls == []
    # a default retry opener forces the low-res modes
    assert capture.camera_opener(True, low_res=True).keywords["low_res"] is True


def test_open_capture_low_res_override(monkeypatch: pytest.MonkeyPatch) -> None:
    got: dict[str, Any] = {}

    def fake_open(src: Any, *a: Any, **kw: Any) -> Any:
        got.update(kw)
        return capture.OpenedCamera(object(), np.zeros((2, 2, 3)), CaptureMode(320, 240, 15, "MJPG"), src, HD)

    monkeypatch.setattr(capture, "open_camera", fake_open)
    monkeypatch.setattr(capture, "camera_devices", lambda: [])
    cap = capture.open_capture(C.CameraSpec(device=1, role="primary"), cv2=object(), low_res=True)
    assert got["low_res"] is True and cap.mode == CaptureMode(320, 240, 15, "MJPG")
    capture.open_capture(C.CameraSpec(device=1, role="primary"), cv2=object())
    assert got["low_res"] is None
    capture.open_capture(C.CameraSpec(device=1, role="secondary"), cv2=object())
    assert got["low_res"] is True


# =============================================================================== 5. modes kept across starts
def test_a_fallback_mode_is_remembered_per_camera_name_and_used_first_next_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_v0112_fixes as T

    monkeypatch.setattr(capture, "_windows", lambda: True)
    monkeypatch.setattr(capture, "list_cameras", lambda **kw: T._devs())
    monkeypatch.setattr(capture, "windows_msmf_cameras", lambda: list(T.MF_LIST))
    capture.reset_session_modes()
    capture._open_now.clear()
    store = tmp_path / "camera-modes.json"
    capture.use_mode_store(store)
    only_low = T.FakeCam(works=lambda fcc, w, h, fps: w == 320 and fps == 15, frames=10**6)

    def widths(cv2: Any) -> list[Any]:
        return [v for cap in cv2.caps for p, v in cap.sets if p == T.W_PROP]

    cv2 = T.FakeCV2({1: only_low})
    got = capture.open_camera(1, 640, 480, 30, "MJPG", cv2=cv2, devices=T._devs(), low_res=True, **T._quick())
    got.cap.release()
    assert got.mode == CaptureMode(320, 240, 15, "MJPG") and widths(cv2)[0] == 640  # found the slow way
    saved = json.loads(store.read_text())
    assert saved == {HD: {"backend": "DirectShow", "width": 320, "height": 240, "fps": 15, "fourcc": "MJPG"}}

    # the next start (a new process): the remembered mode is the first one tried
    capture.reset_session_modes()
    capture.use_mode_store(None)
    capture.use_mode_store(store)
    cv2b = T.FakeCV2({1: only_low})
    again = capture.open_camera(1, 640, 480, 30, "MJPG", cv2=cv2b, devices=T._devs(), low_res=True, **T._quick())
    again.cap.release()
    assert widths(cv2b) == [320] and again.mode.fps == 15  # right away, nothing else tried

    # once the configured mode works at the first try, the remembered entry is dropped
    capture.reset_session_modes()
    cv2c = T.FakeCV2({1: T.FakeCam(frames=10**6)})
    capture.open_camera(1, 640, 480, 30, "MJPG", cv2=cv2c, devices=T._devs(), low_res=True, **T._quick()).cap.release()
    assert HD not in capture.saved_modes() and widths(cv2c)[0] == 640
    capture._open_now.clear()
    capture.reset_session_modes()


def test_mode_store_is_off_by_default_and_tolerates_bad_files(tmp_path: Path) -> None:
    assert capture.mode_store() is None and capture.saved_modes() == {}
    capture.remember_mode(HD, "DirectShow", CaptureMode(320, 240, 15, "MJPG"))  # session only
    assert capture.session_mode(HD) == ("DirectShow", CaptureMode(320, 240, 15, "MJPG"))
    capture.reset_session_modes()
    bad = tmp_path / "m.json"
    bad.write_text("{not json")
    capture.use_mode_store(bad)
    assert capture.saved_modes() == {}
    bad.write_text(
        json.dumps(
            {
                HD: {"backend": "DirectShow"},
                JOY: [1],
                "ok": {"backend": "Media Foundation", "width": 640, "height": 480, "fps": 30, "fourcc": ""},
            }
        )
    )
    assert capture.saved_modes() == {"ok": ("Media Foundation", CaptureMode(640, 480, 30, ""))}
    bad.write_text("[1, 2]")
    assert capture.saved_modes() == {}
    assert capture.default_mode_file().name == "camera-modes.json"


def test_the_app_turns_the_mode_store_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from away.app.runner import AwayApp

    monkeypatch.setattr(capture, "default_mode_file", lambda: tmp_path / "modes.json")
    app = AwayApp.__new__(AwayApp)
    app._use_mode_store()
    assert capture.mode_store() == tmp_path / "modes.json"


# =============================================================================== 6. the app lets go while probing
def test_probe_stop_flag_keeps_the_owner_fresh_and_ends_on_a_request(tmp_path: Path) -> None:
    from away.app.runner import _ProbeStop

    share = CS.CameraShare(tmp_path, pid=1)
    stop = threading.Event()
    flag = _ProbeStop(stop, share, [JOY, HD], every_s=0.0)
    assert not flag.is_set()
    assert CS.CameraShare(tmp_path, pid=2).owner() is not None  # visible while probing
    CS.CameraShare(tmp_path, pid=2).request("settings")
    assert flag.is_set()
    stop.set()
    assert _ProbeStop(stop, share, None).is_set()


def test_app_hands_over_cameras_while_still_probing(tmp_path: Path) -> None:
    """The user's case: HD User Facing is slow (the app is still probing it), and Calibrate asks for the
    cameras. Before v0.1.15 the app published no owner while probing and ignored the request for up to 30 s."""
    import test_v018_fixes as T

    devs = T.FakeDevices(d0={}, d1={"silent": True})  # the built-in camera never sends a frame: long probe
    app = T._app(tmp_path, devs, serial=True, stop_after=None)
    th = threading.Thread(
        target=app._multicam_loop,
        args=(None, None, None),
        kwargs={"trackers": [T._Tracker(), T._Tracker()], "probe_s": 30.0},
        daemon=True,
    )
    th.start()
    ui = CS.CameraShare(tmp_path / "share", pid=2)
    assert _wait(lambda: ui.owner() is not None, 5.0)
    t0 = time.monotonic()
    with CS.CameraLease(ui, timeout=8.0, heartbeat_s=0.05) as lease:
        assert lease.app_running and lease.handed_over
        assert time.monotonic() - t0 < 8.0
        assert ("camera.handover", {"released": True, "cameras": [0, 1]}) in app.events
    app.stop_event.set()
    th.join(10)
    assert not th.is_alive()


# =============================================================================== 7. away cameras test
def test_run_camera_test_falls_back_to_a_low_res_secondary_and_offers_to_save() -> None:
    def stream(specs: list[C.CameraSpec], seconds: float) -> list[camtest.CameraResult]:
        out = []
        for s in specs:
            r = camtest.CameraResult(str(s.device), s.role, "DirectShow", f"{s.width}x{s.height}")
            r.width, r.height, r.fps_set, r.fourcc = s.width, s.height, s.fps, s.fourcc
            if s.role == "secondary" and s.width == 640:
                r.error = "cannot open camera 'HD User Facing': did not open"
            else:
                r.frames, r.fps = 300, 30.0
            out.append(r)
        return out

    progress: list[str] = []
    rep = camtest.run_camera_test(_specs(), 10, stream=stream, progress=progress.append)
    assert [p.ok for p in rep.phases] == [False, True] and rep.works
    assert "do NOT stream together" in rep.verdict() and "320x240 MJPG @ 15" in rep.verdict()
    offer = rep.save_offer(_specs())
    assert offer is not None and [(c.width, c.height, c.fps, c.fourcc) for c in offer] == [
        (640, 480, 30, "MJPG"),
        (320, 240, 15, "MJPG"),
    ]
    text = "\n".join(rep.lines())
    assert (
        "FAILED - cannot open camera" in text and "ok - 320x240 via DirectShow, 30.0 fps measured, 300 frames" in text
    )
    assert any("trying the secondary at 320x240 MJPG @ 15" in p for p in progress)

    cfg = C.AppConfig()
    cfg.hands.cameras = _specs()
    changed = camtest.save_combination(cfg, offer)
    assert changed == ["HD User Facing: 640x480@30 MJPG -> 320x240@15 MJPG"]
    assert (cfg.hands.cameras[1].width, cfg.hands.cameras[1].fps) == (320, 15)
    assert cfg.hands.cameras[0].width == 640


def test_run_camera_test_verdicts() -> None:
    ok = camtest.CameraResult(JOY, "primary", frames=10, fps=30.0, mode="640x480@30 MJPG", width=640, height=480)
    rep = camtest.run_camera_test(_specs(), 1, stream=lambda s, sec: [ok, ok])
    assert len(rep.phases) == 1 and "stream at the same time with your current settings" in rep.verdict()
    assert rep.save_offer(_specs()) is None  # same as configured (fps 30, MJPG)
    bad = camtest.CameraResult(HD, "secondary", error="no frames")
    rep2 = camtest.run_camera_test(_specs(), 1, stream=lambda s, sec: [ok, bad])
    assert len(rep2.phases) == 2 and not rep2.works and "cannot stream at the same time" in rep2.verdict()
    assert rep2.combination() is None and rep2.save_offer(_specs()) is None
    single = camtest.run_camera_test(_specs()[:1], 1, stream=lambda s, sec: [bad])
    assert len(single.phases) == 1  # no secondary to try at low res
    assert camtest.TestReport().verdict() == "no cameras were tested"


def test_stream_cameras_on_the_simulated_bus_measures_each_camera() -> None:
    bus = Bus(limit=3.0)
    specs = _specs(fourcc="YUY2")
    res = camtest.stream_cameras(specs, 0.4, start=_starter(bus, "test", fallbacks=False))
    assert res[0].ok and res[0].mode == "640x480@30 YUY2" and res[0].backend == "DirectShow"
    assert not res[1].ok and "sent no frames" in res[1].error  # both at 640x480 YUY2: over the bus limit
    # a simulated 30 fps camera; a busy CI runner (macOS) measured 17 fps, so only check it really streamed
    assert res[0].fps > 10 and res[0].frames > 3
    assert bus.holder == {}  # released after the test

    res2 = camtest.stream_cameras(camtest.low_res_specs(specs), 0.3, start=_starter(bus, "test", fallbacks=False))
    assert all(r.ok for r in res2) and res2[1].mode == "320x240@15 MJPG"


def test_full_camera_test_on_the_simulated_bus_finds_the_working_combination() -> None:
    bus = Bus(limit=2.5)
    rep = camtest.run_camera_test(
        _specs(fourcc="YUY2"),
        0.3,
        stream=lambda s, sec: camtest.stream_cameras(s, sec, start=_starter(bus, "test", fallbacks=False)),
    )
    assert not rep.phases[0].ok and rep.phases[1].ok
    combo = rep.save_offer(_specs(fourcc="YUY2"))
    assert combo is not None and (combo[1].width, combo[1].height, combo[1].fourcc) == (320, 240, "MJPG")


class _FakeLease:
    def __init__(self, running: bool = False, handed: bool = False) -> None:
        self.app_running, self.handed_over = running, handed
        self.closed = False

    def __enter__(self) -> _FakeLease:
        return self

    def __exit__(self, *a: Any) -> None:
        self.closed = True


def _cfg_file(tmp_path: Path) -> Path:
    cfg = C.AppConfig()
    cfg.hands.cameras = _specs()
    p = tmp_path / "config.toml"
    C.save(cfg, p)
    return p


def _report_low() -> camtest.TestReport:
    good = camtest.CameraResult(JOY, "primary", "DirectShow", "640x480@30 MJPG", 640, 480, 30, "MJPG", 300, 30.0)
    low = camtest.CameraResult(HD, "secondary", "DirectShow", "320x240@15 MJPG", 320, 240, 15, "MJPG", 150, 15.0)
    fail = camtest.CameraResult(HD, "secondary", error="did not open")
    return camtest.TestReport(
        [
            camtest.Phase("configured settings", _specs(), [good, fail]),
            camtest.Phase("low", camtest.low_res_specs(_specs()), [good, low]),
        ]
    )


def test_cli_cameras_test_reports_and_saves_with_flag(tmp_path: Path, capsys: Any) -> None:
    p = _cfg_file(tmp_path)
    lease = _FakeLease(running=True, handed=True)
    tested: list[Any] = []

    def tester(cams: list[Any], seconds: float, progress: Any) -> camtest.TestReport:
        tested.append((len(cams), seconds))
        return _report_low()

    args = cli.build_parser().parse_args(["cameras", "test", "--config", str(p), "--seconds", "3"])
    assert cli.cmd_cameras_test(args, lease_factory=lambda: lease, tester=tester) == 0
    out = capsys.readouterr().out
    assert tested == [(2, 3.0)] and lease.closed
    assert "released the cameras for the test" in out and "do NOT stream together" in out
    assert "away cameras test --save" in out
    assert C.load(p)[0].hands.cameras[1].width == 640  # not saved without --save

    args = cli.build_parser().parse_args(["cameras", "test", "--config", str(p), "--save"])
    assert cli.cmd_cameras_test(args, lease_factory=lambda: _FakeLease(), tester=tester) == 0
    assert "saved: HD User Facing: 640x480@30 MJPG -> 320x240@15 MJPG" in capsys.readouterr().out
    cams = C.load(p)[0].hands.cameras
    assert (cams[1].width, cams[1].height, cams[1].fps, cams[1].fourcc) == (320, 240, 15, "MJPG")


def test_cli_cameras_test_failure_paths(tmp_path: Path, capsys: Any) -> None:
    p = _cfg_file(tmp_path)
    args = cli.build_parser().parse_args(["cameras", "test", "--config", str(p)])
    assert cli.cmd_cameras_test(args, lease_factory=lambda: _FakeLease(running=True), tester=None) == 2
    assert "did not hand them over" in capsys.readouterr().err
    bad = camtest.CameraResult(HD, "secondary", error="no frames")
    rep = camtest.TestReport([camtest.Phase("configured settings", _specs(), [bad, bad])])
    assert cli.cmd_cameras_test(args, lease_factory=lambda: _FakeLease(), tester=lambda *a, **k: rep) == 1
    empty = tmp_path / "empty.toml"
    C.save(C.AppConfig(), empty)
    args = cli.build_parser().parse_args(["cameras", "test", "--config", str(empty)])
    assert cli.cmd_cameras_test(args) == 2
    assert "no cameras in hands.cameras" in capsys.readouterr().err


def test_settings_camera_test_worker(tmp_path: Path) -> None:
    out: queue.Queue[Any] = queue.Queue()
    leases = _leases(tmp_path)

    def tester(cams: list[Any], seconds: float, progress: Any) -> camtest.TestReport:
        assert leases.held
        progress("opening...")
        return _report_low()

    SG.camera_test_worker(_model(_specs()), out, leases=leases, tester=tester, seconds=1)
    msgs = [out.get_nowait() for _ in range(out.qsize())]
    assert msgs[0] == "opening..." and isinstance(msgs[1]["report"], camtest.TestReport) and not leases.held

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("bus error")

    SG.camera_test_worker(_model(_specs()), out, leases=leases, tester=boom)
    assert out.get_nowait() == {"error": "camera test failed: bus error"}

    class Blocked:
        app_running, handed_over = True, False

        def open(self) -> Blocked:
            return self

        def close(self) -> None:
            pass

    blocked = CS.SharedLease(lambda who: Blocked())  # type: ignore[arg-type,return-value]
    SG.camera_test_worker(_model(_specs()), out, leases=blocked, tester=tester)
    assert "did not hand them over" in out.get_nowait()["error"]


# =============================================================================== 8. the whole real-world sequence
def test_end_to_end_app_lease_live_view_calibrate_hand_back(tmp_path: Path) -> None:
    """App holds both cameras -> Settings borrows them -> Live confidence -> Calibrate (a separate process with
    its own lease) -> the cameras go back to the app. The simulated devices reject a second open of a held
    device, and both at 640x480 YUY2 exceed the USB bus (MJPG / 320x240 fit)."""
    folder = tmp_path / "share"
    bus = Bus(limit=3.0)
    specs = _specs(fourcc="YUY2")
    app_specs = _specs(fourcc="MJPG")  # the app runs them in MJPG (what v0.1.12+ asks for)
    app = FakeApp(folder, bus, app_specs)
    app.th.start()
    assert _wait(lambda: set(bus.holder) == {JOY, HD} and app.acquired == 1)

    # before the fix: opening while the app holds the device fails exactly like the user's report
    with pytest.raises(CameraOpenError, match="DirectShow 1: did not open; Media Foundation 1: did not open"):
        bus.opener("old-calibrate")(specs[1])
    bus.violations.clear()

    leases = _leases(folder, pid=2)
    out: queue.Queue[Any] = queue.Queue()
    stop = threading.Event()
    live = threading.Thread(
        target=SG.camera_confidence_worker,
        args=(_model(specs), out, stop),
        kwargs={
            "leases": leases,
            "start": _starter(bus, "live"),
            "trackers": [_Tracker(), _Tracker()],
            "report_s": 0.05,
        },
    )
    live.start()
    msgs: list[Any] = []

    def live_ok() -> bool:
        while not out.empty():
            msgs.append(out.get_nowait())
        return any(isinstance(m, dict) and all(c.endswith("0.00") for c in m["cells"]) for m in msgs)

    assert _wait(live_ok, 8.0), msgs[-3:]
    assert app.handovers == 1 and bus.holder == {JOY: "live", HD: "live"}
    status = [m for m in msgs if isinstance(m, dict)][-1]["status"]
    assert "JoyCam 640x480@30 YUY2 (DirectShow)" in status
    assert "HD User Facing 640x480@30 MJPG (DirectShow)" in status  # YUY2 + YUY2 exceed the bus: MJPG fits

    calib: dict[str, Any] = {}

    def calibration_process() -> dict[str, Any]:  # what `away calibrate-cameras` does, in another "process"
        child = CS.CameraShare(folder, pid=3)

        def run(cfg: Any, args: Any) -> int:
            cstop = threading.Event()
            pairer = FramePairer(2, 0, 0.04, 1.0)
            ths = cli.open_calibration_cameras(
                specs, pairer, cstop, start=_starter(bus, "calibrate"), wait_s=5.0, say=lambda m: None
            )
            calib["streaming"] = _wait(lambda: all(t.frames > 3 for t in ths), 5.0)
            calib["holder"] = dict(bus.holder)
            calib["app_owner"] = child.owner()
            capture.stop_camera_threads(cstop, ths)
            return 0

        rc = cli.calibrate_with_lease(
            _model(specs).cfg,
            _args(tmp_path),
            lease_factory=lambda: CS.borrowed_cameras(child, 2.0, "calibrate"),
            run=run,
        )
        return {"rc": rc, "err": ""}

    res = SG.calibration_flow(leases, lambda: SG.stop_live_view(stop, live, 5.0), calibration_process)
    assert res == {"rc": 0, "err": ""}
    assert calib == {"streaming": True, "holder": {JOY: "calibrate", HD: "calibrate"}, "app_owner": None}
    assert app.handovers == 1  # the app never grabbed the cameras between live view and calibration

    # hand back: the app reopens both cameras
    assert _wait(lambda: app.acquired == 2 and bus.holder == {JOY: "app", HD: "app"}, 5.0)
    assert bus.violations == []  # nobody ever opened a device someone else held
    assert bus.max_parallel == 1  # every open, by anyone, one after another
    app.stop.set()
    app.th.join(3)
    assert bus.holder == {}
