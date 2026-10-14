"""v0.1.16: honest camera verdicts, a slow camera brought up to speed (or replaced by NVIDIA Broadcast), crash
capture, and a four-finger auto-scroll that triggers on a real hand.

The user's Acer Predator Helios 300 (DirectShow: 0 JoyCam, 1 HD User Facing, 2 Camera (NVIDIA Broadcast)):

* Settings > Test both cameras: JoyCam "ok - 640x480@30 MJPG, 30.0 fps measured, 300 frames", HD User Facing
  "ok - 640x480@30 driver format, 1.0 fps measured, 10 frames" (also exactly 1.0 fps at 320x240: not USB
  bandwidth), yet the verdict said "both cameras stream at the same time" and calibration then failed with
  "no secondary camera could be calibrated".
* The very first Test both cameras run crashed the whole app; no log was captured.
* Four-finger auto-scroll never triggered: the real hand read as the open palm (thumb-fold threshold 0.75) and
  toggled pause instead.

The fakes below simulate devices with a fake clock, so every frame rate is exact and the tests take no time.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import __version__, cli
from away import config as C
from away.app import camera_loop as CL
from away.app import camera_share as CS
from away.app import crashlog
from away.app import settings_gui as SG
from away.app.uiqueue import UiQueue
from away.config import GestureConfig
from away.hands.autoscroll import AutoScroller, axis_speed
from away.hands.gestures import THUMB_FOLD_HYSTERESIS, GestureStateMachine, thumb_fold_thresholds
from away.hands.landmarks import palm_size, thumb_fold_ratio
from away.hands.synthetic import pose
from away.multicam import camtest, capture
from away.multicam.capture import CameraDevice, CaptureMode

JOYCAM, BUILTIN, BROADCAST = "JoyCam", "HD User Facing", "Camera (NVIDIA Broadcast)"
PREDATOR = [(JOYCAM, r"\\?\usb#vid_aaaa&pid_0001"), (BUILTIN, r"\\?\usb#vid_0408&pid_a061"), (BROADCAST, "")]
MF_LIST = [(BUILTIN, PREDATOR[1][1].upper()), (JOYCAM, PREDATOR[0][1])]  # Media Foundation: built-in first
DSHOW, MSMF = 700, 1400
FOURCC_PROP, W_PROP, H_PROP, FPS_PROP, EXPOSURE_PROP, AUTO_EXP_PROP = 6, 3, 4, 5, 15, 21
DT = 1 / 30


# =============================================================================== a camera with a real frame rate
class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(float(s), 1e-4)


def fourcc_text(code: Any) -> str:
    return capture.decode_fourcc(code)


class TimedCam:
    """``rate(api, fourcc, w, h, fps_requested) -> frames per second`` (0 = opens but sends nothing)."""

    def __init__(self, rate: Any, opens: bool = True, exposure: float = -6.0) -> None:
        self.rate = rate
        self.opens = opens
        self.exposure = exposure


class TimedCap:
    def __init__(self, cv2: TimedCV2, src: Any, api: Any) -> None:
        self.cv2, self.src, self.api = cv2, src, api
        self.cam: TimedCam = cv2.cams.get((src, api)) or TimedCam(lambda *a: 30.0)
        self.props: dict[int, Any] = {}
        self.released = False
        self.reads = 0
        cv2.log.append(("open", src, api))

    def isOpened(self) -> bool:
        return self.cam.opens

    def set(self, prop: int, value: Any) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> Any:
        if prop == FOURCC_PROP:
            return float(self.props.get(FOURCC_PROP, capture.fourcc_code("YUY2")))
        if prop == W_PROP:
            return float(self.props.get(W_PROP) or 640)
        if prop == H_PROP:
            return float(self.props.get(H_PROP) or 480)
        if prop == FPS_PROP:
            return float(self.props.get(FPS_PROP) or 30)
        if prop == EXPOSURE_PROP:
            return self.cam.exposure
        if prop == AUTO_EXP_PROP:
            return 0.75
        return 0.0

    def _rate(self) -> float:
        fcc = fourcc_text(self.props.get(FOURCC_PROP, 0))
        return float(
            self.cam.rate(self.api, fcc, self.props.get(W_PROP), self.props.get(H_PROP), self.props.get(FPS_PROP, 0))
        )

    def read(self) -> tuple[bool, Any]:
        r = self._rate()
        if not self.cam.opens or r <= 0 or self.released:
            self.cv2.clock.t += 0.05
            return False, None
        self.cv2.clock.t += 1.0 / r
        self.reads += 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.cv2.log.append(("release", self.src, self.api))


class TimedCV2:
    CAP_DSHOW, CAP_MSMF = DSHOW, MSMF
    CAP_PROP_FOURCC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = FOURCC_PROP, W_PROP, H_PROP, FPS_PROP
    CAP_PROP_EXPOSURE, CAP_PROP_AUTO_EXPOSURE, CAP_PROP_BUFFERSIZE = EXPOSURE_PROP, AUTO_EXP_PROP, 38

    def __init__(self, cams: dict[Any, TimedCam], clock: Clock) -> None:
        self.cams = cams
        self.clock = clock
        self.caps: list[TimedCap] = []
        self.log: list[tuple[str, Any, Any]] = []

    def VideoCapture(self, src: Any, api: Any = None) -> TimedCap:
        cap = TimedCap(self, src, api)
        self.caps.append(cap)
        return cap

    def open_now(self) -> set[tuple[Any, Any]]:
        return {(c.src, c.api) for c in self.caps if not c.released and c.cam.opens}


def _devs(broadcast_in_use: bool = False) -> list[CameraDevice]:
    devs = capture.dshow_devices(PREDATOR)
    if broadcast_in_use:
        devs[2].in_use = True
    return devs


def _open(cv2: TimedCV2, clock: Clock, src: int = 1, **kw: Any) -> capture.OpenedCamera:
    args: dict[str, Any] = {
        "cv2": cv2,
        "devices": _devs(),
        "msmf_lister": lambda: list(MF_LIST),
        "clock": clock,
        "sleep": clock.sleep,
        "min_fps": 10.0,
        "measure_s": 1.5,
        "first_frame_s": 4.0,
        "retry_frame_s": 1.5,
    }
    args.update(kw)
    return capture.open_camera(src, 640, 480, 30, **args)


def builtin_only_mjpg_on_msmf(api: Any, fcc: str, w: Any, h: Any, fps: Any) -> float:
    """The built-in camera as reported: 1 fps in every DirectShow mode and size; Media Foundation reaches 30
    fps only when asked for MJPG."""
    return 30.0 if (api == MSMF and fcc == "MJPG") else 1.0


@pytest.fixture(autouse=True)
def _windows_and_clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(capture, "_windows", lambda: True)
    capture.reset_session_modes()
    capture._open_now.clear()
    yield
    capture.reset_session_modes()
    capture._open_now.clear()
    capture._abort_opens.clear()


# =============================================================================== 2. the slow camera, up to speed


@pytest.fixture
def external_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """v0.1.23: NVIDIA Broadcast is never a fallback for the built-in camera (opening it makes Broadcast take the
    built-in camera). These older tests check the fallback mechanics, which still apply to an external camera."""
    monkeypatch.setattr(capture, "broadcast_fallback_allowed", lambda _name: True)


def test_slow_camera_reaches_30fps_only_in_mjpg_on_media_foundation(caplog: Any, tmp_path: Path) -> None:
    clock = Clock()
    cv2 = TimedCV2(
        {(1, DSHOW): TimedCam(builtin_only_mjpg_on_msmf), (0, MSMF): TimedCam(builtin_only_mjpg_on_msmf)}, clock
    )
    capture.use_mode_store(tmp_path / "camera-modes.json")
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = _open(cv2, clock)
    assert got.backend == "Media Foundation" and got.src == 1 and got.cap._key == "1"
    assert got.mode == CaptureMode(640, 480, 30, "MJPG") and not got.slow and got.fallback == ""
    assert got.fps is not None and 29.0 <= got.fps <= 31.0
    # every attempted mode was measured for 1.5 s and logged on one line with the read-back FOURCC and the fps
    lines = [r.getMessage() for r in caplog.records if ": attempt " in r.getMessage()]
    assert all("fps measured" in ln and "fourcc " in ln and "exposure -6" in ln for ln in lines)
    slow_lines = [ln for ln in lines if "too slow (minimum 10)" in ln]
    assert len(slow_lines) == len(lines) - 1 and lines[-1].endswith("; ok")
    assert "attempt DirectShow index 1, asked 640x480@30 MJPG -> fourcc MJPG, 640x480" in lines[0]
    assert "1.0 fps measured" in lines[0] or "0.7 fps measured" in lines[0]
    assert "attempt Media Foundation index 0, asked 640x480@30 MJPG" in lines[-1] and "30.0 fps measured" in lines[-1]
    # DirectShow first (MJPG, YUY2, NV12, driver format, device default), Media Foundation's usual modes, then the
    # sweep: Media Foundation MJPG at the configured size wins before any 320x240 attempt
    assert not any("asked 320x240" in ln for ln in lines)
    assert [cap.src for cap in cv2.caps if cap.api == MSMF][:1] == [0]  # MSMF at ITS index for the camera
    # only the winner is still open; every slow attempt was released
    assert cv2.open_now() == {(0, MSMF)}
    saved = json.loads((tmp_path / "camera-modes.json").read_text())
    assert saved[BUILTIN]["backend"] == "Media Foundation" and saved[BUILTIN]["fourcc"] == "MJPG"
    assert 29.0 <= saved[BUILTIN]["measured_fps"] <= 31.0 and "slow" not in saved[BUILTIN]
    got.cap.release()
    # next start: the remembered mode is tried first and wins at once
    capture.reset_session_modes()
    capture.use_mode_store(tmp_path / "camera-modes.json")
    cv2.caps.clear()
    again = _open(cv2, clock)
    assert len(cv2.caps) == 1 and again.backend == "Media Foundation" and again.mode.fourcc == "MJPG"
    again.cap.release()


def test_the_fastest_mode_wins_not_the_first_that_delivers(tmp_path: Path) -> None:
    def rates(api: Any, fcc: str, w: Any, h: Any, fps: Any) -> float:
        return {"MJPG": 2.0, "YUY2": 6.0, "NV12": 4.0}.get(fcc, 3.0)  # always below 10 fps

    clock = Clock()
    cv2 = TimedCV2({(1, DSHOW): TimedCam(rates), (0, MSMF): TimedCam(rates)}, clock)
    capture.use_mode_store(tmp_path / "camera-modes.json")
    got = _open(cv2, clock, devices=[d for d in _devs() if d.name != BROADCAST])  # no Broadcast to fall back to
    assert got.slow and got.mode.fourcc == "YUY2" and got.fps is not None and 5.5 <= got.fps <= 6.5
    assert got.hints and "too slow: 6.0 fps at best (minimum 10)" in got.hints[0]
    assert len(cv2.open_now()) == 1  # the fastest mode reopened, nothing else left open
    saved = json.loads((tmp_path / "camera-modes.json").read_text())[BUILTIN]
    assert saved["slow"] is True and saved["fourcc"] == "YUY2" and 5.5 <= saved["measured_fps"] <= 6.5
    got.cap.release()
    # the next start checks only the remembered mode (no second sweep of every format and size)
    capture.reset_session_modes()
    capture.use_mode_store(tmp_path / "camera-modes.json")
    n = len(cv2.caps)
    again = _open(cv2, clock, devices=[d for d in _devs() if d.name != BROADCAST])
    assert again.slow and len(cv2.caps) - n == 2  # one measured attempt + the reopen
    again.cap.release()


def test_sweep_modes_order_and_fps() -> None:
    modes = capture.sweep_modes(1280, 720, 0)
    assert modes[:4] == [CaptureMode(1280, 720, 30, f) for f in ("MJPG", "YUY2", "NV12", "")]
    assert [(m.width, m.height) for m in modes[4::4]] == [(640, 480), (320, 240)] and len(modes) == 12
    assert {m.fps for m in modes} == {capture.SWEEP_FPS}  # CAP_PROP_FPS is always asked for
    assert (
        capture.sweep_modes(640, 480, 15)[0] == CaptureMode(640, 480, 15, "MJPG")
        and len(capture.sweep_modes(640, 480, 15)) == 10  # v0.1.19: + 1280x720 MJPG and driver format
    )
    assert capture.sweep_modes(640, 480, 15)[-2:] == [
        CaptureMode(1280, 720, 15, "MJPG"),
        CaptureMode(1280, 720, 15, ""),
    ]
    assert capture.sweep_modes(640, 480, 30, virtual=True) == [
        CaptureMode(640, 480, 30, ""),
        CaptureMode(320, 240, 30, ""),
    ]


def test_fps_is_set_and_auto_exposure_is_never_changed() -> None:
    clock = Clock()
    cv2 = TimedCV2(
        {(1, DSHOW): TimedCam(builtin_only_mjpg_on_msmf), (0, MSMF): TimedCam(builtin_only_mjpg_on_msmf)}, clock
    )
    _open(cv2, clock).cap.release()
    assert all(c.props.get(FPS_PROP) == 30 for c in cv2.caps if W_PROP in c.props)
    assert not any(AUTO_EXP_PROP in c.props or EXPOSURE_PROP in c.props for c in cv2.caps)


def test_measure_fps_and_read_back() -> None:
    clock = Clock()
    cv2 = TimedCV2({(5, None): TimedCam(lambda *a: 20.0)}, clock)
    cap = cv2.VideoCapture(5, None)
    fps, frame = capture.measure_fps(cap, "first", 1.5, clock, clock.sleep)
    assert fps is not None and 19.0 <= fps <= 21.0 and isinstance(frame, np.ndarray)
    assert capture.measure_fps(cap, "first", 0.0, clock, clock.sleep) == (None, "first")
    rb = capture.read_back(cap, cv2)
    assert rb["fourcc"] == "YUY2" and rb["width"] == 640 and rb["auto_exposure"] == 0.75
    assert capture.decode_fourcc(capture.fourcc_code("MJPG")) == "MJPG" and capture.decode_fourcc(0) == ""
    assert capture.decode_fourcc(None) == "" and capture.decode_fourcc(0x01020304).startswith("0x")


# =============================================================================== 3. NVIDIA Broadcast fallback
def _always_slow(broadcast_rate: float) -> dict[Any, TimedCam]:
    return {
        (1, DSHOW): TimedCam(lambda *a: 1.0),
        (0, MSMF): TimedCam(lambda *a: 1.0),
        (2, DSHOW): TimedCam(lambda *a: broadcast_rate),
    }


def test_always_slow_camera_falls_back_to_broadcast_and_is_not_kept_open(caplog: Any, external_camera: None) -> None:
    clock = Clock()
    cv2 = TimedCV2(_always_slow(30.0), clock)
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = _open(cv2, clock, virtual_fallback=True)
    assert got.src == 2 and got.name == BROADCAST and got.backend == "DirectShow" and not got.slow
    assert "HD User Facing is too slow: 1.0 fps at best (minimum 10)" in got.fallback
    assert got.fps is not None and got.fps >= 10
    # the slow built-in camera was closed before Broadcast (whose source it is) was opened, and stays closed
    first_broadcast = cv2.log.index(("open", 2, DSHOW))
    assert all(c.released for c in cv2.caps if c.src != 2) and cv2.open_now() == {(2, DSHOW)}
    assert ("release", 1, DSHOW) in cv2.log[:first_broadcast]
    assert capture.held_cameras() == {"2"}
    assert "opening the virtual camera 'Camera (NVIDIA Broadcast)' instead" in caplog.text
    # the CameraThread / camtest view of it
    fc = capture.FirstFrameCapture(got.cap, got.frame, got.mode, got.backend, got.name)
    assert fc.fallback == "" and fc.measured_fps is None
    got.cap.release()


def test_open_capture_reports_the_fallback_to_the_camera_thread(
    monkeypatch: pytest.MonkeyPatch, external_camera: None
) -> None:
    clock = Clock()
    cv2 = TimedCV2(_always_slow(30.0), clock)
    monkeypatch.setattr(capture, "camera_devices", _devs)
    real = capture.open_camera
    monkeypatch.setattr(
        capture,
        "open_camera",
        lambda *a, **k: real(
            *a, **{**k, "clock": clock, "sleep": clock.sleep, "measure_s": 1.5, "msmf_lister": lambda: list(MF_LIST)}
        ),
    )
    spec = C.CameraSpec(device=BUILTIN, role="secondary")
    fc = capture.open_capture(spec, virtual_fallback=True, cv2=cv2, min_fps=10.0)
    assert fc.name == BROADCAST and fc.opened_index == 2 and "too slow" in fc.fallback and not fc.slow
    th = capture.CameraThread(1, spec, capture.FramePairer(2), threading.Event(), opener=lambda s: fc)
    th.convert = lambda f, m: (th.stop.set(), f)[1]  # stop after the first frame
    th.run()
    assert th.opened_name == BROADCAST and th.opened_index == 2 and "too slow" in th.fallback
    assert th.measured_fps is not None and th.measured_fps >= 10


def test_broadcast_must_reach_the_minimum_itself(external_camera: None) -> None:
    clock = Clock()
    cv2 = TimedCV2(_always_slow(5.0), clock)
    got = _open(cv2, clock, virtual_fallback=True)
    assert got.slow and got.src == 1 and got.name == BUILTIN
    assert any("did not reach 10 fps either" in h for h in got.hints)
    assert cv2.open_now() == {(1, DSHOW)}  # Broadcast released again; only the (slow) camera itself is open
    got.cap.release()


def test_broadcast_is_offered_when_virtual_fallback_is_off(external_camera: None) -> None:
    clock = Clock()
    cv2 = TimedCV2(_always_slow(30.0), clock)
    got = _open(cv2, clock, virtual_fallback=False)
    assert got.slow and got.name == BUILTIN and not any(c.src == 2 for c in cv2.caps)
    assert any("Camera (NVIDIA Broadcast) is free" in h and "hands.virtual_fallback" in h for h in got.hints)
    got.cap.release()


def test_no_broadcast_fallback_when_it_is_already_in_use() -> None:
    clock = Clock()
    cv2 = TimedCV2(_always_slow(30.0), clock)
    got = _open(cv2, clock, virtual_fallback=True, devices=_devs(broadcast_in_use=True))
    assert got.slow and got.name == BUILTIN and not any(c.src == 2 for c in cv2.caps)
    assert not any("Broadcast" in h for h in got.hints)
    got.cap.release()
    held = capture.HeldCapture(types.SimpleNamespace(release=lambda: None), "2")  # this process holds Broadcast
    assert capture.free_broadcast(_devs(), 1) is None
    held.release()
    assert capture.free_broadcast(_devs(), 1) is not None and capture.free_broadcast(_devs(), 2) is None


# =============================================================================== 1. honest verdicts
def test_a_camera_below_the_minimum_is_failed_not_ok() -> None:
    slow = camtest.CameraResult(BUILTIN, "secondary", "DirectShow", "640x480@30 driver format", frames=10, fps=1.0)
    slow.min_fps = 10.0
    assert slow.too_slow and not slow.ok
    assert slow.line() == (
        "secondary HD User Facing: FAILED - too slow: 1.0 fps (minimum 10, hands.min_camera_fps) - "
        "640x480@30 driver format via DirectShow, 10 frames"
    )
    edge = camtest.CameraResult(JOYCAM, "primary", frames=100, fps=10.0, min_fps=10.0)
    assert edge.ok and not edge.too_slow
    under = camtest.CameraResult(JOYCAM, "primary", frames=99, fps=9.99, min_fps=10.0)
    assert under.too_slow and not under.ok
    assert camtest.CameraResult(JOYCAM, "primary", frames=10, fps=1.0).ok  # no minimum given: as before
    assert not camtest.CameraResult(JOYCAM, "primary", frames=0, fps=0.0, min_fps=10).too_slow  # failed, not slow


def _joy() -> camtest.CameraResult:
    return camtest.CameraResult(
        JOYCAM, "primary", "DirectShow", "640x480@30 MJPG", 640, 480, 30, "MJPG", frames=300, fps=30.0
    )


def _specs() -> list[C.CameraSpec]:
    return [C.CameraSpec(device=JOYCAM, role="primary"), C.CameraSpec(device=BUILTIN, role="secondary")]


def test_the_reported_test_no_longer_says_both_cameras_stream() -> None:
    def stream(specs: list[C.CameraSpec], seconds: float) -> list[camtest.CameraResult]:
        hd = camtest.CameraResult(BUILTIN, "secondary", "DirectShow", "640x480@30 driver format", frames=10, fps=1.0)
        hd.hints = ["HD User Facing is too slow: 1.0 fps at best (minimum 10)"]
        return [_joy(), hd]

    rep = camtest.run_camera_test(_specs(), 10, stream=stream, min_fps=10)
    v = rep.verdict()
    assert "stream at the same time" not in v and not rep.works
    assert "HD User Facing streams, but far too slowly: 1.0 fps (minimum 10" in v
    assert "No free NVIDIA Broadcast camera could replace it" in v
    assert len(rep.phases) == 1  # every size was tried while opening: no pointless 320x240 retry
    text = "\n".join(rep.lines())
    assert "JoyCam: ok - 640x480@30 MJPG via DirectShow, 30.0 fps measured, 300 frames" in text
    assert "HD User Facing: FAILED - too slow: 1.0 fps" in text
    assert rep.save_offer(_specs()) is None
    # with the default minimum (capture.MIN_CAMERA_FPS = 10) too
    assert not camtest.run_camera_test(_specs(), 10, stream=stream).works
    # a lower minimum accepts it
    assert camtest.run_camera_test(_specs(), 10, stream=stream, min_fps=1.0).works


def test_broadcast_offer_is_part_of_the_verdict() -> None:
    hd = camtest.CameraResult(BUILTIN, "secondary", frames=10, fps=1.0, min_fps=10)
    hd.hints = ["Camera (NVIDIA Broadcast) is free and may stream faster; turn on hands.virtual_fallback"]
    rep = camtest.TestReport([camtest.Phase("configured settings", _specs(), [_joy(), hd])])
    assert "Camera (NVIDIA Broadcast) is free" in rep.verdict()
    assert "Camera (NVIDIA Broadcast) is free" in hd.line()


def test_a_broadcast_replacement_is_reported_and_saved(tmp_path: Path) -> None:
    def stream(specs: list[C.CameraSpec], seconds: float) -> list[camtest.CameraResult]:
        b = camtest.CameraResult(BUILTIN, "secondary", "DirectShow", "640x480@30", 640, 480, 30, "", 290, 29.0)
        b.opened, b.opened_index = BROADCAST, 2
        b.fallback = "HD User Facing is too slow: 1.0 fps at best (minimum 10)"
        return [_joy(), b]

    progress: list[str] = []
    rep = camtest.run_camera_test(_specs(), 10, stream=stream, progress=progress.append)
    assert rep.works and len(rep.phases) == 1
    assert "stream at the same time" in rep.verdict() and "NVIDIA Broadcast's camera (Camera (NVIDIA" in rep.verdict()
    assert any("using Camera (NVIDIA Broadcast) instead" in p for p in progress)
    line = rep.phases[0].results[1].line()
    assert line.startswith("secondary HD User Facing: ok - using Camera (NVIDIA Broadcast) instead (HD User Facing is")
    offer = rep.save_offer(_specs())
    assert offer is not None and offer[1].device == BROADCAST and offer[1].index_hint == 2 and offer[1].fourcc == ""
    cfg = C.AppConfig()
    cfg.hands.cameras = _specs()
    capture.use_mode_store(tmp_path / "m.json")
    capture.remember_mode(BUILTIN, "DirectShow", CaptureMode(640, 480, 30, ""), fps=1.0, slow=True)
    changed = camtest.save_combination(cfg, offer, _specs())
    assert changed[0] == "HD User Facing -> Camera (NVIDIA Broadcast) (NVIDIA Broadcast's camera)"
    assert cfg.hands.cameras[1].device == BROADCAST and cfg.hands.cameras[1].index_hint == 2
    assert BUILTIN not in capture.saved_modes()
    assert camtest.save_combination(C.AppConfig(), offer) == []  # nothing configured: nothing to change


def test_cli_cameras_test_save_writes_broadcast(tmp_path: Path, capsys: Any) -> None:
    cfg = C.AppConfig()
    cfg.hands.cameras = _specs()
    p = tmp_path / "config.yaml"
    C.save(cfg, p)
    b = camtest.CameraResult(BUILTIN, "secondary", "DirectShow", "640x480@30", 640, 480, 30, "", 290, 29.0, min_fps=10)
    b.opened, b.opened_index, b.fallback = BROADCAST, 2, "HD User Facing is too slow"
    j = _joy()
    j.min_fps = 10
    rep = camtest.TestReport([camtest.Phase("configured settings", _specs(), [j, b])])

    class Lease:
        app_running = handed_over = False

        def __enter__(self) -> Lease:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

    args = cli.build_parser().parse_args(["cameras", "test", "--config", str(p), "--save"])
    assert cli.cmd_cameras_test(args, lease_factory=Lease, tester=lambda *a, **k: rep) == 0
    out = capsys.readouterr().out
    assert "saved: HD User Facing -> Camera (NVIDIA Broadcast)" in out
    assert C.load(p)[0].hands.cameras[1].device == BROADCAST


def test_stream_result_carries_the_fallback_and_the_minimum() -> None:
    th = types.SimpleNamespace(
        mode=CaptureMode(640, 480, 30, ""),
        backend="DirectShow",
        error=None,
        opened=threading.Event(),
        fallback="HD User Facing is too slow",
        opened_name=BROADCAST,
        opened_index=2,
        hints=["x"],
    )
    r = camtest._result(C.CameraSpec(device=BUILTIN, role="secondary"), th, 300, 10.0, 10.0)
    assert r.ok and r.opened == BROADCAST and r.opened_index == 2 and r.min_fps == 10.0 and r.hints == ["x"]
    slow = camtest._result(C.CameraSpec(device=BUILTIN, role="secondary"), types.SimpleNamespace(), 10, 10.0, 10.0)
    assert slow.too_slow and slow.fps == 1.0 and slow.opened == ""


def _thread(frames: int, seconds: float, slow: bool = False, error: str | None = None) -> capture.CameraThread:
    th = capture.CameraThread(
        1, C.CameraSpec(device=BUILTIN, role="secondary"), capture.FramePairer(2), threading.Event()
    )
    th.frames, th.first_t, th.last_t, th.slow, th.error = frames, 0.0, seconds, slow, error
    th.mode, th.backend = CaptureMode(640, 480, 30, ""), "DirectShow"
    th.state = "failed" if error else ("streaming" if frames else "open")
    return th


def test_camera_thread_too_slow_and_settings_live_status() -> None:
    slow, fast, fresh = _thread(11, 10.0), _thread(301, 10.0), _thread(3, 0.5, slow=True)
    assert slow.too_slow(10.0) and not fast.too_slow(10.0) and fresh.too_slow(10.0)
    assert not _thread(3, 0.5).too_slow(10.0)  # too early to tell, and the opener did not say so
    assert not slow.too_slow(0.0) and not _thread(11, 10.0, error="gone").too_slow(10.0)
    assert SG.live_cells([fast, slow], None, 10.0) == ["streaming", "FAILED: too slow (1.0 fps)"]
    line = SG.live_status_line([fast, slow], 10.0)
    assert line == (
        "live: HD User Facing 640x480@30 driver format (DirectShow) | HD User Facing FAILED - too slow: 1.0 fps "
        "(minimum 10)"
    )
    assert SG.live_cells([slow], None) == ["streaming"]  # no minimum: as before
    slow.opened_name, slow.fallback = BROADCAST, "too slow"
    assert "HD User Facing (via Camera (NVIDIA Broadcast): too slow)" in SG.live_status_line([slow], 10.0)


def test_runner_logs_each_camera_ok_or_failed_too_slow(caplog: Any) -> None:
    lines = CL.camera_speed_lines([_thread(301, 10.0), _thread(11, 10.0), _thread(0, 0.0, error="no frames")], 10.0)
    assert lines[0] == (False, "camera 'HD User Facing': ok - 640x480@30 driver format (DirectShow), 30.0 fps measured")
    assert lines[1][0] is True and lines[1][1].startswith(
        "camera 'HD User Facing': FAILED - too slow: 1.0 fps (minimum 10, hands.min_camera_fps)"
    )
    assert lines[2] == (True, "camera 'HD User Facing': FAILED - no frames")
    from away.app.runner import AwayApp

    app = AwayApp.__new__(AwayApp)
    app.cfg = C.AppConfig()
    events: list[tuple[str, Any]] = []
    app.bus = types.SimpleNamespace(publish=lambda t, d: events.append((t, d)))
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        app._log_camera_speed([_thread(301, 10.0), _thread(11, 10.0)])
    assert "FAILED - too slow: 1.0 fps" in caplog.text and "ok - 640x480@30" in caplog.text
    assert [t for t, _ in events] == ["cameras.slow", "app.notice"]
    assert events[1][1]["message"].startswith("camera 'HD User Facing': FAILED - too slow: 1.0 fps")
    events.clear()
    app._log_camera_speed([_thread(301, 10.0)])
    assert events == []


def test_calibration_refuses_a_too_slow_camera() -> None:
    clock = Clock()
    cams = _specs()
    msg = cli.calibration_speed_check(
        cams, [_thread(301, 10.0), _thread(11, 10.0)], 10.0, clock=clock, sleep=clock.sleep
    )
    assert msg is not None and msg.startswith("calibration cannot run: camera 'HD User Facing' is too slow")
    assert "1.0 fps (minimum 10, hands.min_camera_fps)" in msg and "Test both cameras" in msg
    assert clock.t >= 1000.0 + cli.CALIBRATION_SPEED_S  # it streamed first, then measured
    assert cli.calibration_speed_check(cams, [_thread(301, 10.0), _thread(301, 10.0)], 10.0, settle_s=0) is None
    dead = cli.calibration_speed_check(cams, [_thread(301, 10.0), _thread(0, 0.0)], 10.0, settle_s=0)
    assert dead is not None and "camera 'HD User Facing' sent no frames" in dead


def test_min_camera_fps_config() -> None:
    assert C.AppConfig().hands.min_camera_fps == 10.0
    cfg, _ = C.from_dict({"hands": {"min_camera_fps": 15}})
    assert cfg.hands.min_camera_fps == 15
    for bad in (0, 0.5, 61, "fast"):
        with pytest.raises(ValueError, match=r"hands\.min_camera_fps"):
            C.from_dict({"hands": {"min_camera_fps": bad}})
    assert capture.MIN_CAMERA_FPS == 10.0 and capture.MEASURE_S == 0.0  # (tests: unmeasured; the app: 1.5 s)


def test_real_measure_window_default() -> None:
    # conftest sets capture.MEASURE_S to 0 for the fake cameras; the shipped default measures 1.5 s per mode
    src = Path(capture.__file__).read_text(encoding="utf-8")
    assert "\nMEASURE_S = 1.5 " in src and "\nMIN_CAMERA_FPS = 10.0 " in src and "\nSWEEP_BUDGET_S = 60.0 " in src


# =============================================================================== 4. crash capture
class FakeFaultHandler:
    def __init__(self) -> None:
        self.enabled: list[dict[str, Any]] = []
        self.disabled = 0

    def enable(self, file: Any = None, all_threads: bool = False) -> None:
        self.enabled.append({"file": file, "all_threads": all_threads})

    def disable(self) -> None:
        self.disabled += 1


@pytest.fixture
def crash_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("AWAY_NO_CRASHLOG", raising=False)
    crashlog.uninstall(FakeFaultHandler())
    fh = FakeFaultHandler()
    sysm = types.SimpleNamespace(excepthook=lambda *a: None)
    thr = types.SimpleNamespace(excepthook=lambda a: None)
    yield tmp_path / "logs", fh, sysm, thr
    crashlog.uninstall(FakeFaultHandler())


def _install(env: Any, entry: str = "tray") -> str | None:
    folder, fh, sysm, thr = env
    return crashlog.install(entry, folder, faulthandler_mod=fh, sys_mod=sysm, threading_mod=thr, version="0.1.16")


def test_crash_log_faulthandler_and_hooks(crash_env: Any, caplog: Any) -> None:
    folder, fh, sysm, thr = crash_env
    prev_calls: list[Any] = []
    sysm.excepthook = lambda *a: prev_calls.append(a)
    assert _install(crash_env) is None
    path = folder / "crash.log"
    # v0.1.23: faulthandler writes to its own per-process file, folded into crash.log later (handled codes tagged)
    native = folder / f"crash-native-{os.getpid()}.log"
    assert fh.enabled and fh.enabled[0]["all_threads"] is True and fh.enabled[0]["file"].name == str(native)
    head = path.read_text(encoding="utf-8")
    assert "] tray started (pid " in head and "Project Away 0.1.16" in head and head.startswith("[20")
    assert _install(crash_env, "settings") is None and len(fh.enabled) == 1  # idempotent
    try:
        raise ValueError("boom in main")
    except ValueError as exc:
        with caplog.at_level(logging.ERROR, logger="away.crash"):
            sysm.excepthook(ValueError, exc, exc.__traceback__)
    assert len(prev_calls) == 1  # the previous hook still runs
    try:
        raise RuntimeError("boom in thread")
    except RuntimeError as exc:
        args = types.SimpleNamespace(
            exc_type=RuntimeError,
            exc_value=exc,
            exc_traceback=exc.__traceback__,
            thread=types.SimpleNamespace(name="camera-1"),
        )
        thr.excepthook(args)
    text = path.read_text(encoding="utf-8")
    assert "Uncaught exception in the main thread (tray):" in text and "ValueError: boom in main" in text
    assert "Uncaught exception in thread camera-1 (tray):" in text and "RuntimeError: boom in thread" in text
    assert "Uncaught exception (main thread)" in caplog.text
    sysm.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert path.read_text(encoding="utf-8").count("Uncaught exception") == 2  # Ctrl+C is not a crash


def test_next_start_reports_a_crash_once(crash_env: Any, caplog: Any) -> None:
    folder = crash_env[0]
    assert _install(crash_env) is None
    crashlog.write("simulated")  # timestamped entries
    # a native crash: faulthandler appends its dump to the open file, then the process dies
    with (folder / "crash.log").open("a", encoding="utf-8") as f:
        f.write("Fatal Python error: Segmentation fault\n\nThread 0x0001 (most recent call first):\n  File x\n")
    crashlog.uninstall(FakeFaultHandler())
    with caplog.at_level(logging.WARNING, logger="away.crash"):
        note = _install(crash_env)
    assert note == crashlog.CRASHED_MESSAGE == "the app crashed last time; details in crash.log"
    assert "Fatal Python error: Segmentation fault" in caplog.text
    assert crashlog.last_crash_note() == note and crashlog.recent_note(folder) == note  # Settings shows it too
    crashlog.uninstall(FakeFaultHandler())
    assert _install(crash_env) is None  # already reported
    assert crashlog.recent_note(folder) == note  # still recent for a Settings window opened later
    assert crashlog.recent_note(folder, clock=lambda: time.time() + crashlog.NOTE_MAX_AGE_S + 60) is None


def test_handled_cpp_exception_noise_is_not_a_crash_but_uncaught_errors_are_noted(crash_env: Any) -> None:
    folder = crash_env[0]
    _install(crash_env)
    with (folder / "crash.log").open("a", encoding="utf-8") as f:
        f.write("Windows fatal exception: code 0xe06d7363\n\nCurrent thread 0x1 (most recent call first):\n")
    crashlog.uninstall(FakeFaultHandler())
    assert _install(crash_env) is None
    crashlog.write("Uncaught exception in thread x (tray):\nTraceback ...")
    crashlog.uninstall(FakeFaultHandler())
    assert _install(crash_env) == crashlog.ERROR_MESSAGE
    assert crashlog.classify("Windows fatal exception: access violation") == "crashed"
    assert crashlog.classify("nothing") is None


def test_crash_log_rotates_and_can_be_turned_off(crash_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = crash_env[0]
    folder.mkdir(parents=True)
    (folder / "crash.log").write_text("x" * (crashlog.MAX_BYTES + 10), encoding="utf-8")
    _install(crash_env)
    assert (folder / "crash.log.1").stat().st_size > crashlog.MAX_BYTES
    assert (folder / "crash.log").stat().st_size < 200
    crashlog.uninstall(FakeFaultHandler())
    monkeypatch.setenv("AWAY_NO_CRASHLOG", "1")
    fh = FakeFaultHandler()
    assert crashlog.install("cli", folder, faulthandler_mod=fh) is None and fh.enabled == []
    assert crashlog.crash_log_path(folder) == folder / "crash.log"


def test_every_entry_point_installs_the_crash_log(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    entry = (root / "packaging" / "pyinstaller" / "away_entry.py").read_text()
    assert "crashlog.install(crashlog.entry_label(sys.argv[1:]))" in entry  # v0.1.23: labelled entries
    assert 'crashlog.install("tray")' in (root / "src" / "away" / "app" / "tray.py").read_text()
    assert 'crashlog.install("settings")' in (root / "src" / "away" / "app" / "settings_gui.py").read_text()
    seen: list[str] = []
    monkeypatch.setattr(crashlog, "install", lambda entry, *a, **k: seen.append(entry))
    assert cli.main(["demo"]) == 0
    assert seen == ["cli cmd_demo"]


def test_default_crash_log_location_is_the_data_logs_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AWAY_DATA_DIR", str(tmp_path / "ProjectAway"))
    assert crashlog.crash_log_path() == tmp_path / "ProjectAway" / "logs" / "crash.log"


def test_tray_title_mentions_the_crash() -> None:
    from away.app.tray import TrayController

    ctl = TrayController(types.SimpleNamespace(pipeline=None, voice=None))
    ctl.voice_state = lambda: "off"  # type: ignore[method-assign]
    assert "crashed" not in ctl.title()
    ctl.crash_note = "the app crashed last time; details in crash.log"
    assert ctl.title().endswith(" - crashed last time, see crash.log")


# =============================================================================== 4b. hand-over and Tk thread
def test_wait_devices_free_waits_for_held_and_opening_cameras() -> None:
    clock = Clock()
    held = capture.HeldCapture(types.SimpleNamespace(release=lambda: None), "1")
    state = {"n": 0}

    def sleep(s: float) -> None:
        clock.sleep(s)
        assert capture._abort_opens.is_set()  # opens in progress are told to give up meanwhile
        state["n"] += 1
        if state["n"] == 5:
            held.release()

    assert capture.wait_devices_free(5.0, clock, sleep) is True and state["n"] == 5
    assert not capture._abort_opens.is_set()
    capture._opening += 1
    try:
        assert capture.wait_devices_free(0.5, clock, clock.sleep) is False
    finally:
        capture._opening -= 1
    assert capture.opens_in_progress() == 0 and capture.wait_devices_free(0.5, clock, clock.sleep) is True


def test_an_open_in_progress_gives_up_during_a_handover() -> None:
    clock = Clock()
    cv2 = TimedCV2(_always_slow(30.0), clock)
    capture._abort_opens.set()
    with pytest.raises(capture.OpenCancelled):
        _open(cv2, clock)
    assert capture.opens_in_progress() == 0 and cv2.open_now() == set()


def test_runner_hands_over_only_after_the_devices_are_free() -> None:
    from away.app.runner import AwayApp

    order: list[str] = []
    app = AwayApp.__new__(AwayApp)
    app.stop_event = threading.Event()
    app.bus = types.SimpleNamespace(publish=lambda t, d: order.append(f"publish {t} {d['released']}"))
    app._wait_devices_free = lambda timeout=0: order.append("wait_free") or True  # type: ignore[method-assign]
    asked = iter([True, False])
    app.camera_share = types.SimpleNamespace(
        clear_owner=lambda: order.append("clear_owner"),
        ack_released=lambda d: order.append("ack"),
        requested=lambda: next(asked),
        clear_released=lambda: order.append("clear_released"),
    )
    app.stop_event.wait = lambda s: None  # type: ignore[method-assign]
    app._wait_handover([0, 1])
    assert order[:3] == ["wait_free", "clear_owner", "ack"]
    assert order.count("ack") == 2  # the acknowledgement is kept fresh while the borrower works
    assert order[-1] == "publish camera.handover False"


def test_single_camera_loop_hands_over_only_after_the_devices_are_free(tmp_path: Path) -> None:
    order: list[str] = []
    share = CS.CameraShare(tmp_path, pid=1)
    loop = CL.CameraLoop(
        lambda: iter([]),
        lambda *a: None,
        threading.Event(),
        share=share,
        wait_free=lambda: order.append("wait_free") or True,
        check_s=0.01,
    )
    real_ack = share.ack_released
    share.ack_released = lambda cams: (order.append("ack"), real_ack(cams))  # type: ignore[method-assign]
    loop._hand_over()
    assert order == ["wait_free", "ack"]


def test_camera_test_waits_for_the_handover_to_be_confirmed() -> None:
    clock = Clock()
    released = {"after": 3}

    class Share:
        def released(self) -> bool:
            released["after"] -= 1
            return released["after"] <= 0

        def request(self, who: str) -> None:
            pass

    lease = types.SimpleNamespace(app_running=True, handed_over=False, share=Share(), who="t")
    said: list[str] = []
    assert CS.confirm_handover(lease, 5.0, clock, clock.sleep, said.append) and lease.handed_over
    assert said == ["waiting for Project Away to release the cameras..."]
    never = types.SimpleNamespace(
        app_running=True, handed_over=False, share=types.SimpleNamespace(released=lambda: False, request=lambda w: None)
    )
    assert not CS.confirm_handover(never, 1.0, clock, clock.sleep) and not never.handed_over
    assert CS.confirm_handover(types.SimpleNamespace(app_running=False, handed_over=False))
    assert not CS.confirm_handover(types.SimpleNamespace(app_running=True, handed_over=False))


def test_camera_share_survives_windows_sharing_violations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    share = CS.CameraShare(tmp_path, pid=1)
    other = CS.CameraShare(tmp_path, pid=2)
    share.publish_owner([JOYCAM])
    real = Path.read_text
    fails = {"n": 3}

    def flaky(self: Path, *a: Any, **k: Any) -> str:
        if self.name == CS.OWNER_FILE and fails["n"] > 0:
            fails["n"] -= 1
            raise PermissionError(32, "being used by another process")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert other.owner() is not None and fails["n"] == 0  # not mistaken for "no app holds the cameras"
    fails["n"] = 99
    assert other.owner() is None  # a file that stays locked is still treated as unreadable, not an error
    calls = {"n": 0}

    def flaky_replace() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "denied")
        return "ok"

    assert CS._retry_sharing(flaky_replace) == "ok" and calls["n"] == 3


def test_ui_queue_runs_worker_posts_only_on_the_tk_thread() -> None:
    ui = UiQueue()
    ran: list[tuple[int, int]] = []
    th = threading.Thread(target=lambda: ui.post(lambda x: ran.append((x, threading.get_ident())), 7))
    th.start()
    th.join()
    assert ran == []  # nothing runs on the worker
    assert ui.drain() == 1 and ran == [(7, threading.get_ident())] and ui.owner == threading.get_ident()
    ui.post(lambda: 1 / 0)
    ui.post(lambda: ran.append((8, 0)))
    assert ui.drain() == 2 and ran[-1] == (8, 0)  # one failing callback does not stop the rest
    afters: list[Any] = []
    ui.pump(types.SimpleNamespace(after=lambda ms, fn, *a: afters.append((ms, fn, a))), 25)
    assert afters and afters[0][0] == 25 and afters[0][1] == ui.pump


def test_no_worker_thread_touches_tk_directly() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "away" / "app"
    assert "root.after(0" not in (root / "settings_gui.py").read_text()
    assert "label.after(0" not in (root / "cursor_overlay.py").read_text()


# =============================================================================== 5. four-finger auto-scroll
REAL_FOUR = {"thumb_fold": 0.45}  # thumb held against the side of the index finger: ratio ~0.82


def feed(sm: GestureStateMachine, frames: list[Any], t0: float = 0.0) -> tuple[list[Any], float]:
    out, t = [], t0
    for lm in frames:
        t += DT
        out += sm.update(t, lm)
    return out, t


def kinds(events: list[Any]) -> list[str]:
    return [e.kind for e in events]


def test_the_real_four_finger_pose_triggers_and_never_pauses() -> None:
    real = pose("palm", **REAL_FOUR)
    r = thumb_fold_ratio(real)
    assert 0.75 < r < 0.95  # v0.1.15 (fixed 0.75) read this as the open palm
    sm = GestureStateMachine()
    ev, t = feed(sm, [real] * 60)  # 2 s: far longer than clutch_hold (0.4 s)
    assert "clutch" not in kinds(ev) and not sm.paused
    assert kinds(ev).count("autoscroll_start") == 1 and sm.autoscroll.active
    # the old fixed threshold (0.75) reproduces the bug: a thumb at the side of the index finger (ratio ~0.93)
    # read as the open palm, so holding the pose toggled pause instead of scrolling
    side = pose("palm", thumb_fold=0.3)
    assert 0.85 < thumb_fold_ratio(side) < 0.95
    old = GestureStateMachine(GestureConfig(autoscroll_thumb_fold=0.75))
    ev_old, _ = feed(old, [side] * 60)
    assert "autoscroll_start" not in kinds(ev_old) and "clutch" in kinds(ev_old) and old.paused
    new = GestureStateMachine()
    ev_new, _ = feed(new, [side] * 60)
    assert "autoscroll_start" in kinds(ev_new) and "clutch" not in kinds(ev_new) and not new.paused


def test_thumb_fold_threshold_is_configurable_with_hysteresis() -> None:
    on, off = thumb_fold_thresholds(GestureConfig())
    assert on == 0.95 and off == pytest.approx(1.05) and pytest.approx(0.1) == THUMB_FOLD_HYSTERESIS
    sm = GestureStateMachine(GestureConfig(autoscroll_thumb_fold=0.8))
    assert sm.pose_of(pose("palm", **REAL_FOUR)) == "palm"  # 0.82 >= 0.8
    assert sm.pose_of(pose("four")) == "four"
    assert sm.pose_of(pose("palm", **REAL_FOUR)) == "four"  # held: released only above 0.9
    assert sm.pose_of(pose("palm")) == "palm"
    assert C.AppConfig().hands.gestures.autoscroll_thumb_fold == 0.95
    with pytest.raises(ValueError, match="autoscroll_thumb_fold"):
        C.from_dict({"hands": {"gestures": {"autoscroll_thumb_fold": 2.0}}})
    paths = [p for p, *_ in C.tunable_ranges(C.AppConfig())]
    assert (
        "hands.gestures.autoscroll_thumb_fold" in paths and "hands.gestures.autoscroll_thumb_fold" in SG.TUNABLE_LABELS
    )


def test_a_hand_in_the_hysteresis_band_never_toggles_pause() -> None:
    band = pose("palm", thumb_fold=0.2)  # ratio ~1.01: neither clearly folded nor clearly out
    assert 0.95 <= thumb_fold_ratio(band) <= 1.05
    sm = GestureStateMachine()
    ev, _ = feed(sm, [band] * 60)
    assert "clutch" not in kinds(ev) and not sm.paused and "autoscroll_start" not in kinds(ev)
    ev2, _ = feed(GestureStateMachine(), [pose("palm")] * 30)  # a real open palm still pauses
    assert "clutch" in kinds(ev2)
    off = GestureStateMachine(GestureConfig(autoscroll_enabled=False))  # auto-scroll off: any palm pauses
    ev3, _ = feed(off, [band] * 30)
    assert "clutch" in kinds(ev3)


def test_scrolling_starts_from_the_current_position_as_neutral() -> None:
    sm = GestureStateMachine()
    # the hand drifts a little while the thumb folds: the neutral point is where it is during the hold
    hold = [pose("four", center=(0.5, 0.50 + 0.002 * i)) for i in range(8)]
    ev, t = feed(sm, hold)
    start = next(e for e in ev if e.kind == "autoscroll_start")
    assert sm.autoscroll.anchor is not None
    ys = [float(h[0, 1]) for h in hold]
    assert start.data["y"] == pytest.approx(sm.autoscroll.anchor[1])
    assert min(ys) < sm.autoscroll.anchor[1] < max(ys)
    # held still where it is: no scrolling (no need to move away from the start first, nothing jumps)
    ev, t = feed(sm, [hold[-1]] * 10, t)
    assert all(e.data["dy"] == 0 and e.data["dx"] == 0 for e in ev if e.kind == "autoscroll")
    # half a palm length up: speed proportional to the distance beyond the dead zone
    palm = palm_size(hold[-1])
    g = GestureConfig()
    for lengths in (0.5, 1.0):
        up = pose("four", center=(0.5, 0.5 + 0.002 * 3.5 - palm * lengths))  # the hold's mean centre - lengths
        ev, t = feed(sm, [up] * 5, t)
        offset = (sm.autoscroll.anchor[1] - float(up[0, 1])) / palm
        assert ev[-1].data["vy"] == pytest.approx(g.autoscroll_gain * (offset - g.autoscroll_dead_zone), rel=1e-6)
        assert offset == pytest.approx(lengths, abs=1e-6)
    # a fist stops it
    ev, t = feed(sm, [pose("fist")] * 10, t)
    assert [e.data["reason"] for e in ev if e.kind == "autoscroll_stop"] == ["fist"] and not sm.autoscroll.active


def test_speed_is_proportional_with_a_dead_zone() -> None:
    g = GestureConfig()
    assert g.autoscroll_dead_zone == 0.15 and g.autoscroll_gain == 20.0
    assert axis_speed(0.15, 0.15, 20.0, 30.0) == 0.0
    assert axis_speed(0.65, 0.15, 20.0, 30.0) == pytest.approx(10.0)
    assert axis_speed(1.15, 0.15, 20.0, 30.0) == pytest.approx(20.0)  # twice as far, twice as fast
    assert axis_speed(-0.65, 0.15, 20.0, 30.0) == pytest.approx(-10.0)
    assert axis_speed(5.0, 0.15, 20.0, 30.0) == 30.0


def test_neutral_point_is_the_mean_of_the_hold() -> None:
    a = AutoScroller(GestureConfig(autoscroll_hold=0.1))
    assert a.update(0.00, "four", (0.40, 0.50), 0.1) is None
    assert a.update(0.05, "four", (0.42, 0.52), 0.1) is None
    res = a.update(0.10, "four", (0.44, 0.54), 0.1)
    assert res is not None and res[0] == "autoscroll_start"
    assert res[1]["x"] == pytest.approx(0.42) and res[1]["y"] == pytest.approx(0.52)
    assert a.anchor == (pytest.approx(0.42), pytest.approx(0.52))
    a.stop()
    assert a._hold_pos == [] and a.anchor is None
    b = AutoScroller(GestureConfig(autoscroll_hold=0.1))
    b.update(0.0, "four", (0.1, 0.1), 0.1)
    b.update(0.05, "palm", (0.9, 0.9), 0.1)  # the pose broke: the hold (and its positions) start over
    b.update(0.06, "four", (0.5, 0.5), 0.1)
    res = b.update(0.2, "four", (0.5, 0.5), 0.1)
    assert res is not None and res[1]["x"] == pytest.approx(0.5)


def test_palm_scroll_suppression_still_ends_auto_scroll() -> None:
    sm = GestureStateMachine()
    _ev, t = feed(sm, [pose("palm", **REAL_FOUR)] * 10)
    assert sm.autoscroll.active
    out = sm.update(t + DT, pose("palm", **REAL_FOUR), suppress=True)
    assert kinds(out) == ["autoscroll_stop"] and out[0].data["reason"] == "palm_scroll"


# =============================================================================== 7. version
def test_version_is_0_1_17() -> None:
    root = Path(__file__).resolve().parents[1]
    assert __version__ == "0.1.24"
    assert 'version = "0.1.24"' in (root / "pyproject.toml").read_text()
    assert '#define AppVersion "0.1.24"' in (root / "packaging" / "windows" / "away.iss").read_text()
    assert '"AWAY_VERSION", "0.1.24"' in (root / "packaging" / "pyinstaller" / "away.spec").read_text()
    for script in ("linux/build_appimage.sh", "linux/build_deb.sh", "macos/build_dmg.sh"):
        assert 'VERSION="${1:-0.1.23}"' in (root / "packaging" / script).read_text()
