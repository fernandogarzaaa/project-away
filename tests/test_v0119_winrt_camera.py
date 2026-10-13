"""v0.1.19: the WinRT MediaCapture camera backend, honest MJPG forcing, camera retries that back off and stop,
``away cameras diagnose``, and the Settings > Cameras fixes.

Real-PC findings (Acer Predator Helios 300, Windows 11, v0.1.17) these tests pin down:

* "HD User Facing" never opens through OpenCV DSHOW or MSMF (a fast "did not open", even alone), while the
  Windows Camera app (WinRT MediaCapture / Frame Server) shows it smoothly -> a WinRT backend after DSHOW/MSMF,
  before the NVIDIA Broadcast fallback.
* JoyCam asked for MJPG reads back YUY2 -> FOURCC before the size, read back, logged honestly.
* cameras.retry fired 306 times in one day (every ~45 s, forever), each round reopening JoyCam and sweeping the
  dead camera -> exponential back-off, a stop after N failures, "Retry cameras", never reopen a working camera.
* Settings: test output cut off, Device picker blank until a row is selected, Role "secondary" for the only one.

No real WinRT or camera is touched: WinRT sits behind :class:`~away.multicam.winrt_capture.WinrtApi` (faked
here) and the OpenCV fakes run on a fake clock.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import cli
from away import config as C
from away.app import camera_loop as CL
from away.app import camera_share as CS
from away.app import selftest
from away.app import settings_gui as SG
from away.events import EventBus
from away.multicam import capture, diagnose
from away.multicam import winrt_capture as W
from away.multicam.capture import CameraDevice, CaptureMode
from away.multicam.winrt_capture import NativeFormat, choose_format

JOYCAM, BUILTIN, BROADCAST = "JoyCam", "HD User Facing", "Camera (NVIDIA Broadcast)"
PREDATOR = [(JOYCAM, r"\\?\usb#vid_aaaa&pid_0001"), (BUILTIN, r"\\?\usb#vid_0408&pid_a061"), (BROADCAST, "")]
MF_LIST = [(BUILTIN, PREDATOR[1][1].upper()), (JOYCAM, PREDATOR[0][1])]
DSHOW, MSMF = 700, 1400
FOURCC_PROP, W_PROP, H_PROP, FPS_PROP = 6, 3, 4, 5


def nf(w: int, h: int, fps: float, sub: str = "MJPG", i: int = 0) -> NativeFormat:
    return NativeFormat(w, h, fps, sub, i)


# =============================================================================== 1. native format choice
def test_30fps_beats_the_exact_size_and_720p_is_allowed_for_it() -> None:
    fmts = [nf(640, 480, 15, i=0), nf(1280, 720, 30, i=1), nf(1920, 1080, 30, i=2)]
    got = choose_format(fmts, 640, 480, 30)
    assert got == fmts[1]  # 30 fps wins; 1080p is above the 1280x720 ceiling


def test_exact_size_at_30fps_wins_and_mjpg_breaks_ties() -> None:
    fmts = [nf(640, 480, 30, "YUY2", 0), nf(640, 480, 30, "MJPG", 1), nf(1280, 720, 30, "MJPG", 2)]
    assert choose_format(fmts, 640, 480, 30) == fmts[1]
    assert choose_format(fmts, 640, 480, 30, "YUY2") == fmts[0]  # a remembered native mode is matched exactly
    nv = [nf(640, 480, 30, "YUY2", 0), nf(640, 480, 30, "NV12", 1)]
    assert choose_format(nv, 640, 480, 30) == nv[1]  # NV12 before YUY2


def test_smallest_covering_size_else_the_largest_smaller_one() -> None:
    fmts = [nf(320, 240, 30, i=0), nf(1280, 720, 30, i=1), nf(800, 600, 30, i=2)]
    assert choose_format(fmts, 640, 480, 30) == fmts[2]
    small = [nf(320, 240, 30, i=0), nf(424, 240, 30, i=1)]
    assert choose_format(small, 640, 480, 30) == small[1]


def test_rate_preference_30_over_60_but_60_over_15() -> None:
    assert choose_format([nf(640, 480, 60, i=0), nf(640, 480, 30, i=1)], 640, 480, 0).fps == 30
    assert choose_format([nf(640, 480, 15, i=0), nf(640, 480, 60, i=1)], 640, 480, 30).fps == 60
    # 29.97 counts as 30
    assert choose_format([nf(640, 480, 29.97, i=0), nf(640, 480, 15, i=1)], 640, 480, 30).fps == 29.97


def test_auto_size_and_edge_cases() -> None:
    assert choose_format([], 640, 480, 30) is None
    only_big = [nf(1920, 1080, 30)]
    assert choose_format(only_big, 640, 480, 30) == only_big[0]  # used when nothing else is offered
    fmts = [nf(640, 480, 30, i=0), nf(1280, 720, 30, i=1)]
    assert choose_format(fmts, 0, 0, 0) == fmts[0]  # 0 = auto: 640x480 @ 30


def test_native_format_mode_and_fourcc() -> None:
    assert nf(1280, 720, 29.97).mode() == CaptureMode(1280, 720, 30, "MJPG")
    guid = NativeFormat(640, 480, 30, "{32315659-0000-0010-8000-00AA00389B71}")
    assert guid.fourcc() == "" and guid.mode().fourcc == "" and "640x480@30" in guid.describe()
    assert nf(640, 480, 30, "nv12").fourcc() == "NV12"


def test_bgra_buffer_to_bgr_drops_alpha_and_row_padding() -> None:
    w, h = 3, 2
    bgra = np.zeros((h, w, 4), np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = 10, 20, 30, 255
    bgra[1, 2] = (1, 2, 3, 4)
    out = W.bgra_to_bgr(bgra.tobytes(), w, h)
    assert out.shape == (2, 3, 3) and out.flags["C_CONTIGUOUS"]
    assert tuple(out[0, 0]) == (10, 20, 30) and tuple(out[1, 2]) == (1, 2, 3)
    padded = np.zeros((h, w * 4 + 4), np.uint8)  # 4 bytes of padding per row
    padded[:, : w * 4] = bgra.reshape(h, w * 4)
    out2 = W.bgra_to_bgr(padded.tobytes(), w, h, stride=w * 4 + 4)
    assert np.array_equal(out, out2)


# =============================================================================== 2. downscaling
def test_fit_within_keeps_the_aspect_ratio_and_never_upscales() -> None:
    assert capture.fit_within(1280, 720, 640, 480) == (640, 360)
    assert capture.fit_within(640, 480, 640, 480) == (640, 480)
    assert capture.fit_within(320, 240, 640, 480) == (320, 240)
    assert capture.fit_within(1280, 720, 0, 0) == (1280, 720)  # 0 = auto: unchanged
    assert capture.fit_within(1920, 1080, 320, 240) == (320, 180)


def test_resize_frame_and_scaled_capture() -> None:
    frame = np.arange(720 * 1280 * 3, dtype=np.uint32).astype(np.uint8).reshape(720, 1280, 3)
    assert capture.resize_frame(frame, 640, 360).shape == (360, 640, 3)

    class Cap:
        released = False

        def read(self) -> tuple[bool, Any]:
            return True, frame

        def release(self) -> None:
            self.released = True

        def isOpened(self) -> bool:
            return True

    inner = Cap()
    sc, first = capture.scaled(inner, frame, CaptureMode(1280, 720, 30, "MJPG"), 640, 480)
    assert isinstance(sc, capture.ScaledCapture) and first.shape == (360, 640, 3)
    ok, f = sc.read()
    assert ok and f.shape == (360, 640, 3) and sc.isOpened()
    sc.release()
    assert inner.released
    same, f2 = capture.scaled(inner, frame, CaptureMode(640, 480, 30, "MJPG"), 640, 480)
    assert same is inner and f2 is frame  # the configured size: nothing wrapped


# =============================================================================== 3. WinrtCapture on a fake WinRT
class FakeSource:
    def __init__(self, formats: list[NativeFormat], current: NativeFormat | None) -> None:
        self.formats = formats
        self.current = current


class FakeApi:
    """A camera as WinRT reports it; frames are pushed from a thread at ``rate`` while the reader runs."""

    def __init__(
        self,
        names: list[str] | None = None,
        formats: list[NativeFormat] | None = None,
        fail: dict[str, str] | None = None,
        rate: float = 200.0,
        current: NativeFormat | None = None,
        set_format_error: str = "",
        reader_error: str = "",
    ) -> None:
        self.names = names if names is not None else [BUILTIN]
        self.fmts = formats if formats is not None else [nf(640, 480, 15, i=0), nf(1280, 720, 30, i=1)]
        self.fail = fail or {}
        self.rate = rate
        self.current = current or self.fmts[0]
        self.set_format_error = set_format_error
        self.reader_error = reader_error
        self.calls: list[tuple[Any, ...]] = []
        self.active: NativeFormat | None = None
        self.stopped = threading.Event()
        self._pump: threading.Thread | None = None

    async def group_names(self) -> list[str]:
        return list(self.names)

    async def find_group(self, name: str, occurrence: int = 0) -> Any:
        self.calls.append(("find", name, occurrence))
        same = [n for n in self.names if n.lower() == name.lower()]
        return (name, occurrence) if same else None

    async def open(self, group: Any, sharing: str) -> Any:
        self.calls.append(("open", sharing))
        if sharing in self.fail:
            raise OSError(self.fail[sharing])
        return {"group": group, "sharing": sharing}

    def color_source(self, capture_: Any) -> Any:
        return FakeSource(self.fmts, self.current)

    def formats(self, source: Any) -> list[NativeFormat]:
        return list(source.formats)

    def current_format(self, source: Any) -> NativeFormat | None:
        return source.current

    async def set_format(self, source: Any, fmt: NativeFormat) -> None:
        self.calls.append(("set_format", fmt))
        if self.set_format_error:
            raise OSError(self.set_format_error)
        source.current = fmt

    async def start_reader(self, capture_: Any, source: Any, on_frame: Any) -> Any:
        self.calls.append(("start_reader",))
        if self.reader_error:
            raise RuntimeError(self.reader_error)
        self.active = source.current
        fmt = source.current

        def pump() -> None:
            n = 0
            while not self.stopped.is_set() and self.rate > 0:
                frame = np.full((fmt.height, fmt.width, 3), n % 255, np.uint8)
                on_frame(frame)
                n += 1
                time.sleep(1.0 / self.rate)

        self._pump = threading.Thread(target=pump, daemon=True)
        self._pump.start()
        return "reader"

    async def stop(self, capture_: Any, reader: Any) -> None:
        self.calls.append(("stop", reader))
        self.stopped.set()


def _wcap(api: FakeApi, **kw: Any) -> W.WinrtCapture:
    args: dict[str, Any] = {"api": api, "open_timeout": 5.0, "read_timeout": 1.0, "out_size": (640, 480)}
    args.update(kw)
    return W.WinrtCapture(BUILTIN, 640, 480, 30, "", **args)


def test_winrt_opens_exclusive_picks_720p30_and_downscales(caplog: Any) -> None:
    api = FakeApi()
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        cap = _wcap(api)
    try:
        assert cap.isOpened() and cap.sharing == "exclusive" and cap.error == ""
        assert ("set_format", api.fmts[1]) in api.calls and api.active == api.fmts[1]
        assert cap.native == api.fmts[1] and cap.mode == CaptureMode(1280, 720, 30, "MJPG")
        ok, frame = cap.read()
        assert ok and frame.shape == (360, 640, 3)  # 1280x720 fitted into 640x480, aspect kept
        assert (
            cap.get(3) == 640 and cap.get(4) == 360 and cap.get(5) == 30 and cap.get(6) == capture.fourcc_code("MJPG")
        )
        assert cap.get(15) == 0.0 and cap.set(3, 99) is False
        assert any("WinRT exclusive control, native format 1280x720@30 MJPG" in r.getMessage() for r in caplog.records)
        # each frame is returned once: two reads give two different frames
        ok2, frame2 = cap.read()
        assert ok2 and frame2 is not frame
    finally:
        cap.release()
    assert api.stopped.is_set() and not cap.isOpened() and ("stop", "reader") in api.calls
    assert cap.read() == (False, None)


def test_winrt_falls_back_to_shared_read_only_and_keeps_the_current_format() -> None:
    api = FakeApi(fail={"exclusive": "another app has exclusive control"}, current=nf(640, 480, 30, "YUY2", 0))
    cap = _wcap(api)
    try:
        assert cap.isOpened() and cap.sharing == "shared"
        assert [c for c in api.calls if c[0] == "open"] == [("open", "exclusive"), ("open", "shared")]
        assert not any(c[0] == "set_format" for c in api.calls)  # shared read-only cannot set a format
        assert cap.native == nf(640, 480, 30, "YUY2", 0)
    finally:
        cap.release()


def test_winrt_reports_why_it_did_not_open() -> None:
    both = FakeApi(fail={"exclusive": "busy", "shared": "denied"})
    cap = _wcap(both)
    assert not cap.isOpened() and "exclusive: busy" in cap.error and "shared: denied" in cap.error
    cap.release()
    missing = FakeApi(names=["Some Other Camera"])
    cap = _wcap(missing)
    assert not cap.isOpened() and "does not list a camera named 'HD User Facing'" in cap.error
    assert "Some Other Camera" in cap.error
    cap.release()
    broken = FakeApi(reader_error="frame reader did not start (DEVICE_NOT_AVAILABLE)")
    cap = _wcap(broken)
    assert not cap.isOpened() and "DEVICE_NOT_AVAILABLE" in cap.error and broken.stopped.is_set()
    cap.release()


def test_winrt_set_format_failure_uses_the_current_format(caplog: Any) -> None:
    api = FakeApi(set_format_error="not supported", current=nf(640, 480, 15, "YUY2", 0))
    with caplog.at_level(logging.WARNING, logger="away.cameras"):
        cap = _wcap(api)
    try:
        assert cap.isOpened() and cap.native == nf(640, 480, 15, "YUY2", 0)
        assert any("could not set 1280x720@30 MJPG" in r.getMessage() for r in caplog.records)
    finally:
        cap.release()


def test_winrt_read_times_out_without_frames_and_open_timeout() -> None:
    api = FakeApi(rate=0)
    cap = _wcap(api, read_timeout=0.05)
    try:
        assert cap.isOpened()
        t0 = time.monotonic()
        assert cap.read() == (False, None) and time.monotonic() - t0 < 1.0
    finally:
        cap.release()

    class Hangs(FakeApi):
        async def open(self, group: Any, sharing: str) -> Any:
            import asyncio

            await asyncio.sleep(5)
            return {}

    slow = _wcap(Hangs(), open_timeout=0.2)
    assert not slow.isOpened() and "within 0.2 s" in slow.error
    slow.release()


def test_winrt_passes_the_occurrence_for_two_cameras_of_one_name() -> None:
    api = FakeApi(names=["USB Camera", "USB Camera"])
    cap = W.WinrtCapture("USB Camera", 640, 480, 30, api=api, occurrence=1, open_timeout=5.0)
    try:
        assert ("find", "USB Camera", 1) in api.calls
    finally:
        cap.release()


def test_list_formats_and_group_names_on_the_fake() -> None:
    api = FakeApi()
    fmts, err = W.list_formats(BUILTIN, api=api)
    assert err == "" and fmts == api.fmts and ("open", "shared") in api.calls  # shared first: it may be in use
    assert W.group_names(api=api) == [BUILTIN]
    none, err = W.list_formats("nope", api=FakeApi())
    assert none == [] and "does not list" in err
    _f, err = W.list_formats(BUILTIN, api=FakeApi(fail={"shared": "x", "exclusive": "y"}))
    assert "shared: x" in err and "exclusive: y" in err


def test_winrt_is_unavailable_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    W.reset_availability()
    monkeypatch.setattr(sys, "platform", "linux")
    ok, why = W.available()
    assert not ok and "only on Windows" in why
    W.reset_availability()


# =============================================================================== 4. the open chain on Windows
class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(float(s), 1e-4)


class FakeCap:
    """An OpenCV capture: ``opens``; ``rate`` frames per second on the fake clock (0 = no frames)."""

    def __init__(self, cv2: FakeCV2, src: Any, api: Any) -> None:
        self.cv2, self.src, self.api = cv2, src, api
        self.opens, self.rate = cv2.devices.get((src, api), (True, 30.0))
        self.props: dict[int, Any] = {}
        self.sets: list[int] = []
        self.released = False
        cv2.log.append(("open", src, api))

    def isOpened(self) -> bool:
        return self.opens

    def set(self, prop: int, value: Any) -> bool:
        self.props[prop] = value
        self.sets.append(prop)
        return True

    def get(self, prop: int) -> Any:
        if prop == FOURCC_PROP:
            return float(self.props.get(FOURCC_PROP, capture.fourcc_code("YUY2")))
        return float(self.props.get(prop, 0) or 0)

    def read(self) -> tuple[bool, Any]:
        if not self.opens or self.rate <= 0 or self.released:
            self.cv2.clock.t += 0.05
            return False, None
        self.cv2.clock.t += 1.0 / self.rate
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.cv2.log.append(("release", self.src, self.api))


class FakeCV2:
    CAP_DSHOW, CAP_MSMF = DSHOW, MSMF
    CAP_PROP_FOURCC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = FOURCC_PROP, W_PROP, H_PROP, FPS_PROP
    CAP_PROP_EXPOSURE, CAP_PROP_AUTO_EXPOSURE, CAP_PROP_BUFFERSIZE = 15, 21, 38

    def __init__(self, devices: dict[Any, tuple[bool, float]], clock: Clock) -> None:
        self.devices = devices
        self.clock = clock
        self.log: list[tuple[Any, ...]] = []
        self.caps: list[FakeCap] = []

    def VideoCapture(self, src: Any, api: Any = None) -> FakeCap:
        cap = FakeCap(self, src, api)
        self.caps.append(cap)
        return cap


class FakeWinrtCap:
    """What the WinRT factory returns in the chain tests: streams ``rate`` fps on the fake clock."""

    def __init__(self, cv2: FakeCV2, mode: CaptureMode, rate: float, opens: bool = True, error: str = "") -> None:
        self.cv2, self.rate, self.opens = cv2, rate, opens
        self.error = error if not opens else ""
        self.mode = CaptureMode(1280, 720, 30, "MJPG")  # the native format it chose
        self.asked = mode
        self.released = False

    def isOpened(self) -> bool:
        return self.opens and not self.released

    def read(self) -> tuple[bool, Any]:
        if self.rate <= 0 or not self.opens:
            self.cv2.clock.t += 0.05
            return False, None
        self.cv2.clock.t += 1.0 / self.rate
        return True, np.zeros((360, 640, 3), np.uint8)

    def get(self, prop: int) -> float:
        return {3: 640.0, 4: 360.0, 5: 30.0, 6: float(capture.fourcc_code("MJPG"))}.get(prop, 0.0)

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.cv2.log.append(("release", "winrt", None))


@pytest.fixture(autouse=True)
def _windows_and_clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(capture, "_windows", lambda: True)
    capture.reset_session_modes()
    capture._open_now.clear()
    yield
    capture.reset_session_modes()
    capture._open_now.clear()
    capture._abort_opens.clear()


def _winrt(monkeypatch: pytest.MonkeyPatch, cv2: FakeCV2, rate: float = 30.0, opens: bool = True) -> list[Any]:
    """Make the WinRT backend available with a fake factory; returns the list of (name, w, h, occurrence, mode)."""
    calls: list[Any] = []

    def factory(name: str, width: int, height: int, occurrence: int = 0) -> Any:
        def make(mode: CaptureMode) -> Any:
            calls.append((name, width, height, occurrence, mode))
            cv2.log.append(("winrt", name, mode))
            return FakeWinrtCap(cv2, mode, rate, opens, error="WinRT could not initialise the camera (busy)")

        return make

    monkeypatch.setattr(capture, "winrt_factory_available", lambda: (True, ""))
    monkeypatch.setattr(capture, "winrt_factory", factory)
    return calls


def _devs() -> list[CameraDevice]:
    return capture.dshow_devices(PREDATOR)


def _open(cv2: FakeCV2, clock: Clock, src: int = 1, **kw: Any) -> capture.OpenedCamera:
    args: dict[str, Any] = {
        "cv2": cv2,
        "devices": _devs(),
        "msmf_lister": lambda: list(MF_LIST),
        "clock": clock,
        "sleep": clock.sleep,
        "min_fps": 10.0,
        "measure_s": 1.0,
        "first_frame_s": 2.0,
        "retry_frame_s": 1.0,
    }
    args.update(kw)
    return capture.open_camera(src, 640, 480, 30, **args)


DEAD_BUILTIN = {(1, DSHOW): (False, 0.0), (0, MSMF): (False, 0.0)}  # never opens through OpenCV (the Acer)


def test_dead_builtin_opens_through_winrt_after_dshow_and_msmf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clock = Clock()
    cv2 = FakeCV2(dict(DEAD_BUILTIN), clock)
    calls = _winrt(monkeypatch, cv2)
    capture.use_mode_store(tmp_path / "camera-modes.json")
    got = _open(cv2, clock, virtual_fallback=True)
    assert got.backend == "WinRT" and got.name == BUILTIN and got.src == 1 and got.fallback == ""
    assert got.mode == CaptureMode(1280, 720, 30, "MJPG") and got.fps is not None and got.fps >= 29
    # the order: DirectShow, then Media Foundation (at its own index), then WinRT; Broadcast never touched
    kinds = [(e[0], e[2] if e[0] == "open" else None) for e in cv2.log if e[0] in ("open", "winrt")]
    assert kinds == [("open", DSHOW), ("open", MSMF), ("winrt", None)]
    assert not any(e[0] == "open" and e[1] == 2 for e in cv2.log)  # Broadcast's DirectShow index 2
    assert calls[0][:4] == (BUILTIN, 640, 480, 0) and calls[0][4] == CaptureMode(640, 480, 30, "")
    assert got.cap._key == "1" and "1" in capture.held_cameras()
    saved = json.loads((tmp_path / "camera-modes.json").read_text())[BUILTIN]
    assert saved["backend"] == "WinRT" and (saved["width"], saved["height"], saved["fourcc"]) == (1280, 720, "MJPG")
    got.cap.release()
    assert "1" not in capture.held_cameras()
    # next start: WinRT first, with the remembered native format; DirectShow / Media Foundation not even opened
    capture.reset_session_modes()
    capture.use_mode_store(tmp_path / "camera-modes.json")
    cv2.log.clear()
    again = _open(cv2, clock)
    assert again.backend == "WinRT" and [e[0] for e in cv2.log if e[0] in ("open", "winrt")] == ["winrt"]
    assert calls[-1][4] == CaptureMode(1280, 720, 30, "MJPG")
    again.cap.release()


def test_winrt_comes_before_the_slow_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2({(1, DSHOW): (True, 1.0), (0, MSMF): (True, 1.0)}, clock)  # 1.0 fps everywhere (v0.1.15)
    _winrt(monkeypatch, cv2)
    got = _open(cv2, clock)
    assert got.backend == "WinRT" and not got.slow
    asked = [c.props.get(W_PROP) for c in cv2.caps]
    assert 320 not in asked and 1280 not in asked  # no sweep size was tried: WinRT ended the search first


def test_slow_winrt_still_lets_the_sweep_find_a_fast_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()

    class HDOnly(FakeCV2):
        def VideoCapture(self, src: Any, api: Any = None) -> FakeCap:
            cap = super().VideoCapture(src, api)
            orig = cap.read

            def read() -> tuple[bool, Any]:
                cap.rate = 30.0 if cap.props.get(W_PROP) == 1280 else 1.0  # 30 fps only in its HD mode
                return orig()

            cap.read = read  # type: ignore[method-assign]
            return cap

    cv2 = HDOnly({(1, DSHOW): (True, 1.0), (0, MSMF): (True, 1.0)}, clock)
    _winrt(monkeypatch, cv2, rate=2.0)
    got = _open(cv2, clock, devices=[d for d in _devs() if d.name != BROADCAST])
    assert got.backend == "DirectShow" and got.mode.width == 1280 and not got.slow
    assert isinstance(got.cap._cap, capture.ScaledCapture)  # HD frames are downscaled to the configured size
    winrt_at = next(i for i, e in enumerate(cv2.log) if e[0] == "winrt")
    opens = [i for i, e in enumerate(cv2.log) if e[0] == "open"]
    first_hd = next(i for i, c in enumerate(cv2.caps) if c.props.get(W_PROP) == 1280)
    first_320 = next(i for i, c in enumerate(cv2.caps) if c.props.get(W_PROP) == 320)
    assert winrt_at < opens[first_320] < opens[first_hd]  # WinRT, then the sweep (HD last)
    got.cap.release()


def test_broadcast_fallback_only_after_winrt_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2({**DEAD_BUILTIN, (2, DSHOW): (True, 30.0)}, clock)
    _winrt(monkeypatch, cv2, opens=False)
    got = _open(cv2, clock, virtual_fallback=True)
    assert got.name == BROADCAST and got.src == 2 and "could not be opened directly" in got.fallback
    order = [e[0] if e[0] == "winrt" else e[1] for e in cv2.log if e[0] in ("open", "winrt")]
    assert order.index("winrt") < order.index(2)  # WinRT before NVIDIA Broadcast
    got.cap.release()


def test_winrt_failure_reason_is_in_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2(dict(DEAD_BUILTIN), clock)
    _winrt(monkeypatch, cv2, opens=False)
    with pytest.raises(capture.CameraOpenError) as ei:
        _open(cv2, clock, devices=[d for d in _devs() if d.name != BROADCAST])
    assert "WinRT 1: did not open (WinRT could not initialise the camera (busy))" in str(ei.value)


def test_unavailable_winrt_is_skipped_and_said(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2(dict(DEAD_BUILTIN), clock)
    monkeypatch.setattr(capture, "winrt_factory_available", lambda: (False, W.INSTALL_HINT))
    with pytest.raises(capture.CameraOpenError) as ei:
        _open(cv2, clock, devices=[d for d in _devs() if d.name != BROADCAST])
    assert 'pip install "project-away[winrt]"' in str(ei.value)


def test_winrt_factory_raising_counts_as_did_not_open(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2(dict(DEAD_BUILTIN), clock)
    monkeypatch.setattr(capture, "winrt_factory_available", lambda: (True, ""))

    def factory(*_a: Any, **_k: Any) -> Any:
        def make(_mode: CaptureMode) -> Any:
            raise OSError("RPC server unavailable")

        return make

    monkeypatch.setattr(capture, "winrt_factory", factory)
    with pytest.raises(capture.CameraOpenError) as ei:
        _open(cv2, clock, devices=[d for d in _devs() if d.name != BROADCAST])
    assert "RPC server unavailable" in str(ei.value)


@pytest.mark.parametrize(
    ("pref", "expect"),
    [("dshow", [("open", DSHOW)]), ("msmf", [("open", MSMF)]), ("winrt", [("winrt", None)])],
)
def test_one_backend_only(monkeypatch: pytest.MonkeyPatch, pref: str, expect: list[Any]) -> None:
    clock = Clock()
    cv2 = FakeCV2(dict(DEAD_BUILTIN), clock)
    _winrt(monkeypatch, cv2, opens=False)
    with pytest.raises(capture.CameraOpenError):
        _open(cv2, clock, devices=[d for d in _devs() if d.name != BROADCAST], backend=pref)
    assert [(e[0], e[2] if e[0] == "open" else None) for e in cv2.log if e[0] in ("open", "winrt")] == expect


def test_backend_preference_is_process_wide_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2(dict(DEAD_BUILTIN), clock)
    _winrt(monkeypatch, cv2)
    capture.set_camera_backend("WINRT ")
    assert capture.camera_backend() == "winrt"
    got = _open(cv2, clock)
    assert got.backend == "WinRT" and not any(e[0] == "open" for e in cv2.log)
    got.cap.release()
    with pytest.raises(ValueError):
        capture.set_camera_backend("vfw")
    capture.configure(types_ns(camera_backend="nonsense"))  # ignored, never raises
    assert capture.camera_backend() == "winrt"
    capture.configure(types_ns(camera_backend="dshow"))
    assert capture.camera_backend() == "dshow"
    assert capture.backends_for("auto") == ["DirectShow", "Media Foundation", "WinRT"]
    assert capture.backends_for("msmf") == ["Media Foundation"]
    assert capture.backends_for("winrt", windows=False) == ["default"]


def types_ns(**kw: Any) -> Any:
    import types

    return types.SimpleNamespace(**kw)


def test_a_virtual_camera_always_uses_directshow(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2({(2, DSHOW): (True, 30.0)}, clock)
    calls = _winrt(monkeypatch, cv2)
    got = _open(cv2, clock, src=2, backend="winrt")
    assert got.backend == "DirectShow" and got.name == BROADCAST and calls == []
    got.cap.release()


def test_no_winrt_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "_windows", lambda: False)
    clock = Clock()
    cv2 = FakeCV2({("/dev/video0", None): (True, 30.0)}, clock)
    calls = _winrt(monkeypatch, cv2)
    got = capture.open_camera(
        "/dev/video0", 640, 480, 30, cv2=cv2, devices=[], clock=clock, sleep=clock.sleep, backend="winrt"
    )
    assert got.backend == "default" and calls == []
    got.cap.release()


def test_winrt_name_comes_from_the_list_or_a_configured_name() -> None:
    devs = _devs()
    assert capture.winrt_name(devs[1], "1") == BUILTIN
    assert capture.winrt_name(None, "'HD User Facing'") == BUILTIN
    assert capture.winrt_name(None, "1") is None and capture.winrt_name(None, "/dev/video0") is None


def test_winrt_occurrence_for_the_second_camera_of_one_name(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    devs = capture.dshow_devices([("USB Camera", "a"), ("USB Camera", "b")])
    cv2 = FakeCV2({(1, DSHOW): (False, 0.0)}, clock)
    calls = _winrt(monkeypatch, cv2)
    got = _open(cv2, clock, devices=devs, msmf_lister=lambda: [])
    assert calls[0][0] == "USB Camera" and calls[0][3] == 1
    got.cap.release()


def test_winrt_factory_builds_a_winrt_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Spy:
        def __init__(self, name: str, w: int, h: int, fps: int, fourcc: str, **kw: Any) -> None:
            seen.update(name=name, w=w, h=h, fps=fps, fourcc=fourcc, **kw)

    monkeypatch.setattr(W, "WinrtCapture", Spy)
    make = capture.winrt_factory(BUILTIN, 640, 480, occurrence=1)
    make(CaptureMode(1280, 720, 30, "MJPG"))
    assert seen == {
        "name": BUILTIN,
        "w": 1280,
        "h": 720,
        "fps": 30,
        "fourcc": "MJPG",
        "out_size": (640, 480),
        "occurrence": 1,
    }


# =============================================================================== 5. MJPG forcing
class FourccCap:
    """``mode``: "honours" (MJPG at once), "after_size" (only once the size is set), "ignores" (stays YUY2)."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.fourcc = capture.fourcc_code("YUY2")
        self.order: list[int] = []
        self.size_set = False

    def set(self, prop: int, value: Any) -> bool:
        self.order.append(prop)
        if prop == W_PROP:
            self.size_set = True
        if prop == FOURCC_PROP and (self.mode == "honours" or (self.mode == "after_size" and self.size_set)):
            self.fourcc = int(value)
        return True

    def get(self, prop: int) -> float:
        return float(self.fourcc) if prop == FOURCC_PROP else 0.0


def test_fourcc_is_set_before_the_size_and_read_back(caplog: Any) -> None:
    cap = FourccCap("honours")
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = capture.apply_mode(cap, CaptureMode(640, 480, 30, "MJPG"), FakeCV2({}, Clock()), "'JoyCam'")
    assert got == "MJPG" and cap.order.index(FOURCC_PROP) < cap.order.index(W_PROP)
    assert cap.order.count(FOURCC_PROP) == 1  # accepted at once: not set twice
    assert not any("ignores" in r.getMessage() for r in caplog.records)


def test_fourcc_set_again_after_the_size_when_the_driver_wants_that(caplog: Any) -> None:
    cap = FourccCap("after_size")
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = capture.apply_mode(cap, CaptureMode(640, 480, 30, "MJPG"), FakeCV2({}, Clock()), "'JoyCam'")
    assert got == "MJPG" and cap.order.count(FOURCC_PROP) == 2
    assert any("accepted only after the size was set" in r.getMessage() for r in caplog.records)


def test_a_camera_that_ignores_mjpg_is_logged_honestly(caplog: Any) -> None:
    cap = FourccCap("ignores")
    with caplog.at_level(logging.WARNING, logger="away.cameras"):
        got = capture.apply_mode(cap, CaptureMode(640, 480, 30, "MJPG"), FakeCV2({}, Clock()), "'JoyCam'")
    assert got == "YUY2"
    msg = next(r.getMessage() for r in caplog.records if "ignores" in r.getMessage())
    assert "'JoyCam': asked for MJPG but the driver reports YUY2" in msg


def test_no_fourcc_requested_reads_nothing_back() -> None:
    cap = FourccCap("honours")
    assert capture.apply_mode(cap, CaptureMode(640, 480, 30, ""), FakeCV2({}, Clock())) == ""
    assert FOURCC_PROP not in cap.order


def test_sweep_has_720p_in_mjpg_and_driver_format_last() -> None:
    modes = capture.sweep_modes(640, 480, 30)
    assert modes[-2:] == [CaptureMode(1280, 720, 30, "MJPG"), CaptureMode(1280, 720, 30, "")]
    assert all(m.width != 1280 for m in capture.sweep_modes(640, 480, 30, virtual=True))


# =============================================================================== 6. retries back off and stop
def test_backoff_schedule_doubles_caps_at_15_minutes_and_stops() -> None:
    b = CL.RetryBackoff(45, 900, 8)
    assert b.schedule() == [45, 90, 180, 360, 720, 900, 900, 900]
    seen = []
    while not b.exhausted:
        seen.append(b.delay())
        b.failed()
    assert seen == b.schedule() and b.failures == 8
    b.reset()
    assert b.delay() == 45 and not b.exhausted
    forever = CL.RetryBackoff(10, 40, 0)
    for _ in range(100):
        forever.failed()
    assert not forever.exhausted and forever.delay() == 40
    assert CL.RetryBackoff(45, 10, 3).max_s == 45  # a ceiling below the start is raised to it


def test_backoff_cuts_a_day_of_retries_from_306_to_8() -> None:
    b = CL.RetryBackoff(45, 900, 8)
    assert len(b.schedule()) == 8 and sum(b.schedule()) == 45 + 90 + 180 + 360 + 720 + 900 * 3


def test_wait_for_retry_due_manual_and_done() -> None:
    t = [0.0]
    done, stop = threading.Event(), threading.Event()

    class FakeDone:
        def is_set(self) -> bool:
            return done.is_set()

        def wait(self, s: float) -> bool:
            t[0] += s
            return done.is_set()

    clock = lambda: t[0]  # noqa: E731
    assert CL.wait_for_retry(5.0, FakeDone(), stop, lambda: False, clock=clock) == "time"  # type: ignore[arg-type]
    assert 5.0 <= t[0] <= 5.01
    presses = iter([False, False, True])
    assert CL.wait_for_retry(float("inf"), FakeDone(), stop, lambda: next(presses), clock=clock) == "manual"  # type: ignore[arg-type]
    stop.set()
    assert CL.wait_for_retry(float("inf"), FakeDone(), stop, lambda: False, clock=clock) == "done"  # type: ignore[arg-type]


def test_retry_request_and_status_files(tmp_path: Path) -> None:
    assert not CL.take_retry_request(tmp_path)
    assert CL.request_camera_retry(tmp_path, clock=lambda: 100.0)
    assert CL.take_retry_request(tmp_path, clock=lambda: 101.0)
    assert not CL.take_retry_request(tmp_path, clock=lambda: 101.0)  # taken once
    CL.request_camera_retry(tmp_path, clock=lambda: 0.0)
    assert not CL.take_retry_request(tmp_path, clock=lambda: 7200.0)  # a stale request is dropped
    CL.write_retry_status({"stopped": True, "attempts": 8, "failed": {BUILTIN: ""}}, tmp_path)
    st = CL.read_retry_status(tmp_path)
    assert st is not None and st["stopped"] and "t" in st
    text = CL.retry_status_text(st)
    assert "stopped retrying HD User Facing after 8 failed attempts" in text and "Retry cameras" in text
    CL.write_retry_status({"stopped": False, "attempts": 2, "failed": {BUILTIN: ""}, "next_s": 180}, tmp_path)
    assert "next retry in about 180 s" in CL.retry_status_text(CL.read_retry_status(tmp_path))
    CL.write_retry_status(None, tmp_path)
    assert CL.read_retry_status(tmp_path) is None and CL.retry_status_text(None) == ""


def _app(tmp_path: Path, limit: int = 3, base: float = 0.01) -> Any:
    from away.app.runner import AwayApp

    app = AwayApp.__new__(AwayApp)
    app.cfg = C.AppConfig()
    app.cfg.hands.camera_retry_s = base
    app.cfg.hands.camera_retry_max_s = base * 4
    app.cfg.hands.camera_retry_limit = limit
    app.stop_event = threading.Event()
    app.bus = EventBus()
    app.camera_share = CS.CameraShare(tmp_path / "share", pid=1)
    return app


def test_the_retry_backs_off_stops_and_notifies_once(tmp_path: Path) -> None:
    app = _app(tmp_path, limit=3, base=0.05)
    probes: list[float] = []
    retries: list[dict[str, Any]] = []
    notices: list[dict[str, Any]] = []
    stopped: list[dict[str, Any]] = []
    app.bus.subscribe("cameras.retry", lambda e: retries.append(e.data))
    app.bus.subscribe("app.notice", lambda e: notices.append(e.data))
    app.bus.subscribe("cameras.retry_stopped", lambda e: stopped.append(e.data))

    def probe(specs: list[Any], probe_s: float) -> dict[int, str]:
        probes.append(time.monotonic())
        return {0: "open error: did not open"}

    app._probe_specs = probe

    def run_single(device: Any, until: Any) -> None:
        t0 = time.monotonic()
        while not stopped and time.monotonic() - t0 < 5:
            time.sleep(0.005)
        time.sleep(0.15)  # nothing more is probed once the retries stopped

    t0 = time.monotonic()
    app._single_until_retry(run_single, JOYCAM, [C.CameraSpec(BUILTIN, "secondary")], 0.05, 0.1)
    assert len(probes) == 3 and len(retries) == 3 and len(stopped) == 1 and len(notices) == 1
    assert [r["attempt"] for r in retries] == [1, 2, 3] and retries[-1]["next_s"] is None
    assert retries[0]["next_s"] == pytest.approx(0.1) and retries[1]["next_s"] == pytest.approx(0.2)
    # every wait is at least as long as scheduled: 0.05, then 0.1, then 0.2 (the cap is 4 x the start)
    assert probes[0] - t0 >= 0.045 and probes[1] - probes[0] >= 0.095 and probes[2] - probes[1] >= 0.19
    assert "stopped retrying HD User Facing after 3 failed attempts" in notices[0]["message"]
    assert app.camera_retry_stopped
    st = CL.read_retry_status(tmp_path / "share")
    assert st is not None and st["stopped"] and st["attempts"] == 3


def test_retry_cameras_starts_the_probes_again_and_success_returns(tmp_path: Path) -> None:
    app = _app(tmp_path, limit=1)
    results = iter([{0: "did not open"}, {}])
    probes: list[int] = []

    def probe(specs: list[Any], probe_s: float) -> dict[int, str]:
        probes.append(1)
        return next(results)

    app._probe_specs = probe
    stopped = threading.Event()
    app.bus.subscribe("cameras.retry_stopped", lambda e: stopped.set())
    returned: list[bool] = []

    def run_single(device: Any, until: Any) -> None:
        assert stopped.wait(5)
        CL.request_camera_retry(tmp_path / "share")  # Settings > Cameras > Retry cameras
        t0 = time.monotonic()
        while not until() and time.monotonic() - t0 < 5:
            time.sleep(0.005)
        returned.append(until())

    app._single_until_retry(run_single, JOYCAM, [C.CameraSpec(BUILTIN, "secondary")], 0.01, 0.1)
    assert probes == [1, 1] and returned == [True] and not app.camera_retry_stopped
    assert CL.read_retry_status(tmp_path / "share") is None  # cleared once the camera works


def test_tray_retry_cameras_sets_the_event(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.camera_retry_stopped = True
    seen: list[str] = []
    app.bus.subscribe("cameras.retry_requested", lambda e: seen.append("x"))
    app.retry_cameras()
    assert not app.camera_retry_stopped and seen == ["x"] and app._manual_retry() and not app._manual_retry()

    from away.app.tray import TrayController

    ctl = TrayController.__new__(TrayController)
    ctl.app = app
    app.camera_retry_stopped = True
    assert ctl.camera_retry_stopped()
    ctl.retry_cameras()
    assert not ctl.camera_retry_stopped()


def test_no_camera_at_all_backs_off_too(tmp_path: Path) -> None:
    app = _app(tmp_path, limit=2, base=0.01)
    stopped: list[Any] = []
    app.bus.subscribe("cameras.retry_stopped", lambda e: stopped.append(e.data))
    b = app._retry_backoff(0.01)
    t0 = time.monotonic()
    app._wait_none_retry(b, {BUILTIN: ""})
    assert b.failures == 1 and not stopped
    threading.Timer(0.1, app.retry_cameras).start()  # the second failure stops; Retry cameras resumes
    app._wait_none_retry(b, {BUILTIN: ""})
    assert len(stopped) == 1 and b.failures == 0 and time.monotonic() - t0 < 3


def test_a_retry_probe_never_reopens_the_working_camera() -> None:
    capture._hold("0", 1)  # JoyCam (index 0) is tracking right now
    try:
        cv2 = FakeCV2({}, Clock())
        spec = C.CameraSpec(0, "secondary")  # the failed camera's entry now resolves to index 0
        with pytest.raises(capture.CameraOpenError, match="already open and working"):
            capture.open_capture(spec, cv2=cv2, skip_held=True, devices=_devs())
        assert cv2.caps == []  # nothing was opened
        opener = capture.camera_opener(skip_held=True)
        assert opener.keywords["skip_held"] is True  # type: ignore[attr-defined]
    finally:
        capture._hold("0", -1)


def test_probe_specs_asks_the_opener_to_skip_held_cameras(tmp_path: Path) -> None:
    app = _app(tmp_path)
    seen: dict[str, Any] = {}

    def start(specs: Any, pairer: Any, session: Any, lock: Any, skip_held: bool = False) -> list[Any]:
        seen["skip_held"] = skip_held
        return []

    app._start_cameras = start
    app._end_cameras = lambda s, t: None
    app._probe_timeout = lambda n, p: 0.01
    app._probe_specs([C.CameraSpec(BUILTIN, "secondary")], 0.01)
    assert seen == {"skip_held": True}


def test_a_session_does_not_resweep_a_camera_measured_too_slow(tmp_path: Path) -> None:
    clock = Clock()
    cv2 = FakeCV2({(1, DSHOW): (True, 2.0), (0, MSMF): (True, 2.0)}, clock)
    devs = [d for d in _devs() if d.name != BROADCAST]
    first = _open(cv2, clock, devices=devs)
    assert first.slow
    first.cap.release()
    n = len(cv2.caps)
    again = _open(cv2, clock, devices=devs)  # a background retry in the same session
    assert again.slow and len(cv2.caps) - n == 2  # the remembered mode once + reopen: no new sweep
    again.cap.release()


# =============================================================================== 7. away cameras diagnose
def test_diagnose_reports_every_backend_and_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2({(0, DSHOW): (True, 30.0), (1, DSHOW): (False, 0.0), (0, MSMF): (False, 0.0)}, clock)
    lines: list[str] = []
    opened: list[Any] = []

    def winrt_open(name: str, f: NativeFormat, occ: int) -> Any:
        opened.append((name, f))
        return FakeWinrtCap(cv2, f.mode(), 30.0 if f.fps >= 30 else 0.0)

    fmts = [nf(640, 480, 30, i=0), nf(1280, 720, 30, i=1), nf(640, 480, 15, "YUY2", 2)]
    out = diagnose.run_diagnose(
        [d for d in _devs() if d.name != BROADCAST],
        cv2=cv2,
        out=lines.append,
        windows=True,
        msmf_lister=lambda: list(MF_LIST),
        winrt_available=lambda: (True, ""),
        winrt_formats=lambda n, o: (fmts, ""),
        winrt_open=winrt_open,
        max_winrt_formats=2,
        measure_s=0.5,
        clock=clock,
        sleep=clock.sleep,
    )
    assert out == lines
    text = "\n".join(lines)
    assert "camera 0: JoyCam" in text and "camera 1: HD User Facing" in text
    assert "DirectShow (index 0): OpenCV cannot list this backend's formats" in text
    assert "640x480@30 MJPG: ok 30.0 fps" in text
    assert "1280x720@30 MJPG: ok" in text and "320x240@30 MJPG: ok" in text
    assert "DirectShow (index 1)" in text and "did not open (skipping this backend's other modes)" in text
    assert "Media Foundation (index 1)" in text  # JoyCam is Media Foundation's index 1
    assert "WinRT: 3 native formats: 640x480@30 MJPG, 1280x720@30 MJPG, 640x480@15 YUY2" in text
    assert len(opened) == 4 and opened[0][1] == fmts[0]  # the best 2 formats per camera, best first


def test_diagnose_without_winrt_and_off_windows() -> None:
    clock = Clock()
    cv2 = FakeCV2({}, clock)
    lines = diagnose.run_diagnose(
        _devs()[:1],
        cv2=cv2,
        windows=True,
        backends=["winrt"],
        winrt_available=lambda: (False, W.INSTALL_HINT),
        clock=clock,
        sleep=clock.sleep,
    )
    assert any("WinRT: unavailable" in ln and "project-away[winrt]" in ln for ln in lines)
    nix = diagnose.run_diagnose(
        [CameraDevice(None, "/dev/video0", "Integrated Camera")],
        cv2=FakeCV2({("/dev/video0", None): (True, 30.0)}, clock),
        windows=False,
        measure_s=0.2,
        clock=clock,
        sleep=clock.sleep,
    )
    assert any("default (index /dev/video0)" in ln for ln in nix) and any(": ok 30.0 fps" in ln for ln in nix)
    assert diagnose.run_diagnose([], cv2=cv2, windows=False) == ["no cameras found"]


def test_diagnose_winrt_errors_are_printed() -> None:
    clock = Clock()
    lines = diagnose.run_diagnose(
        _devs()[1:2],
        cv2=FakeCV2({}, clock),
        windows=True,
        backends=["winrt"],
        winrt_available=lambda: (True, ""),
        winrt_formats=lambda n, o: ([], "WinRT does not list a camera named 'HD User Facing'"),
        clock=clock,
        sleep=clock.sleep,
    )
    assert lines[-1] == "  WinRT: WinRT does not list a camera named 'HD User Facing'"

    class Dead:
        error = "busy"

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    lines = diagnose.run_diagnose(
        _devs()[1:2],
        cv2=FakeCV2({}, clock),
        windows=True,
        backends=["winrt"],
        winrt_available=lambda: (True, ""),
        winrt_formats=lambda n, o: ([nf(640, 480, 30)], ""),
        winrt_open=lambda n, f, o: Dead(),
        clock=clock,
        sleep=clock.sleep,
    )
    assert lines[-1] == "    640x480@30 MJPG: did not open (busy)"


def test_diagnose_modes_and_candidates() -> None:
    modes = diagnose.diagnose_modes(640, 480, 30)
    assert modes[0] == CaptureMode(640, 480, 30, "MJPG") and CaptureMode(1280, 720, 30, "") in modes
    assert len(modes) == len(set(modes)) == 7
    assert diagnose.diagnose_modes(0, 0, 0)[0] == CaptureMode(640, 480, 30, "MJPG")
    fmts = [nf(640, 480, 15, i=0), nf(1280, 720, 30, i=1), nf(1920, 1080, 30, i=2)]
    assert diagnose.winrt_candidates(fmts, 640, 480, 30, 5)[:2] == [fmts[1], fmts[2]]


def test_cli_cameras_diagnose(tmp_path: Path, capsys: Any) -> None:
    import contextlib

    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)

    class Lease:
        app_running = False
        handed_over = False

    @contextlib.contextmanager
    def lease() -> Any:
        yield Lease()

    seen: dict[str, Any] = {}

    def runner(devs: Any, **kw: Any) -> list[str]:
        seen.update(devs=devs, **kw)
        kw["out"]("  WinRT: 2 native formats: ...")
        return []

    args = cli.build_parser().parse_args(["cameras", "diagnose", "1", "--config", str(cfgp), "--seconds", "0.5"])
    rc = cli.cmd_cameras_diagnose(args, lease_factory=lease, runner=runner, lister=lambda d: [_devs()[1]])
    assert rc == 0 and seen["measure_s"] == 0.5 and (seen["width"], seen["height"]) == (640, 480)
    assert "WinRT: 2 native formats" in capsys.readouterr().out
    rc = cli.cmd_cameras_diagnose(args, lease_factory=lease, runner=runner, lister=lambda d: [])
    assert rc != 0


# =============================================================================== 8. config
def test_new_config_keys_defaults_and_validation() -> None:
    h = C.AppConfig().hands
    assert (h.camera_backend, h.camera_retry_s, h.camera_retry_max_s, h.camera_retry_limit) == ("auto", 45.0, 900.0, 8)
    cfg, _ = C.from_dict({"hands": {"camera_backend": "winrt", "camera_retry_limit": 0}})
    assert cfg.hands.camera_backend == "winrt" and cfg.hands.camera_retry_limit == 0
    for bad in (
        {"camera_backend": "vfw"},
        {"camera_retry_limit": -1},
        {"camera_retry_limit": 2.5},
        {"camera_retry_s": 1},
        {"camera_retry_s": 100, "camera_retry_max_s": 50},
    ):
        with pytest.raises(ValueError):
            C.from_dict({"hands": bad})


def test_the_cli_and_runner_apply_the_backend(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    cfg.hands.camera_backend = "msmf"
    capture.configure(cfg.hands)
    assert capture.camera_backend() == "msmf"


# =============================================================================== 9. Settings > Cameras
def test_role_defaults_to_primary_for_the_only_camera(tmp_path: Path) -> None:
    m = SG.SettingsModel(path=tmp_path / "config.yaml")
    assert m.role_default() == "primary" and m.role_default(0) == "primary"
    m.cfg.hands.cameras = [C.CameraSpec(BUILTIN, "primary")]  # a one-entry list (hand-edited config)
    assert m.role_default() == "primary" and m.role_default(0) == "primary"
    m.cfg.hands.cameras = [C.CameraSpec(JOYCAM, "primary"), C.CameraSpec(BUILTIN, "secondary")]
    assert m.role_default(1) == "secondary" and m.role_default() == "secondary"  # adding a third one


def test_device_picker_shows_the_current_camera(tmp_path: Path) -> None:
    m = SG.SettingsModel(path=tmp_path / "config.yaml")
    m.cfg.hands.camera_index = BUILTIN
    choices = m.device_choices(_devs())
    assert m.device_choice(None, choices) == "1: HD User Facing"
    m.cfg.hands.cameras = [C.CameraSpec(JOYCAM, "primary"), C.CameraSpec(BUILTIN, "secondary")]
    choices = m.device_choices(_devs())
    assert m.device_choice(None, choices) == "0: JoyCam"  # the primary when nothing is selected
    assert m.device_choice(1, choices) == "1: HD User Facing"
    m.cfg.hands.cameras = []
    m.cfg.hands.camera_index = "Logitech BRIO"
    choices = m.device_choices(_devs())
    assert m.device_choice(None, choices) == "Logitech BRIO (configured)"


def test_settings_backend_picker_and_retry(tmp_path: Path) -> None:
    m = SG.SettingsModel(path=tmp_path / "config.yaml")
    assert m.camera_backend() == "auto"
    assert m.set_voice("hands.camera_backend", "winrt") is None and m.camera_backend() == "winrt"
    assert m.set_voice("hands.camera_backend", "vfw") is not None and m.camera_backend() == "winrt"
    assert m.request_camera_retry(tmp_path) is None and CL.take_retry_request(tmp_path)
    assert m.retry_status_text(tmp_path) == ""
    CL.write_retry_status({"stopped": True, "attempts": 8, "failed": {BUILTIN: ""}}, tmp_path)
    assert "stopped retrying HD User Facing" in m.retry_status_text(tmp_path)


def test_settings_diagnose_worker_streams_lines(tmp_path: Path) -> None:
    import queue

    m = SG.SettingsModel(path=tmp_path / "config.yaml")
    q: queue.Queue[Any] = queue.Queue()

    class Lease:
        app_running = False
        handed_over = False

    class Leases:
        def acquire(self, who: str) -> Any:
            return Lease()

        def release(self) -> None:
            q.put("released")

    def runner(devs: Any, **kw: Any) -> list[str]:
        kw["out"]("camera 1: HD User Facing")
        kw["out"]("  WinRT: unavailable (x)")
        return []

    SG.camera_diagnose_worker(m, q, leases=Leases(), runner=runner, lister=lambda: _devs())
    items = []
    while not q.empty():
        items.append(q.get())
    assert "camera 1: HD User Facing" in items and "  WinRT: unavailable (x)" in items
    assert {"done": True} in items and items[-1] == "released"

    def boom(devs: Any, **kw: Any) -> list[str]:
        raise RuntimeError("no OpenCV")

    SG.camera_diagnose_worker(m, q, leases=Leases(), runner=boom, lister=lambda: [])
    items = []
    while not q.empty():
        items.append(q.get())
    assert {"error": "camera diagnosis failed: no OpenCV"} in items


# =============================================================================== 10. packaging
def test_winrt_extra_is_windows_only_and_bundled() -> None:
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+

    root = Path(__file__).resolve().parents[1]
    proj = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert proj["winrt"] and all("sys_platform == 'win32'" in d for d in proj["winrt"])
    names = {d.split(">")[0] for d in proj["winrt"]}
    assert {"winrt-runtime", "winrt-Windows.Media.Capture", "winrt-Windows.Media.Capture.Frames"} <= names
    assert {"winrt-Windows.Graphics.Imaging", "winrt-Windows.Devices.Enumeration", "winrt-Windows.Foundation"} <= names
    assert set(proj["winrt"]) <= set(proj["all"])  # the installers install [all]
    spec = (root / "packaging" / "pyinstaller" / "away.spec").read_text()
    assert 'optional("winrt"' in spec and "winrt.windows.media.capture.frames" in spec
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    assert '".[dev,winrt]"' in ci


def test_preload_system_msvcp140_before_pywinrt() -> None:
    """pywinrt's own msvcp140.dll (14.29) must never be the process's copy: onnxruntime needs 14.40+."""
    calls: list[str] = []
    assert W.preload_system_msvcp140("linux", calls.append) is False and calls == []
    assert W.preload_system_msvcp140("darwin", calls.append) is False and calls == []
    assert W.preload_system_msvcp140("win32", calls.append) is True and calls == ["msvcp140.dll"]

    def missing(name: str) -> None:
        raise OSError(126, "not found", name)

    assert W.preload_system_msvcp140("win32", missing) is False  # never raises: pywinrt then uses its own copy


def test_installer_never_ships_or_loads_pywinrt_msvcp140() -> None:
    import re

    root = Path(__file__).resolve().parents[1]
    spec = (root / "packaging" / "pyinstaller" / "away.spec").read_text()
    src = spec[spec.index("def _is_winrt_msvcp") :]
    src = src[: src.index("\n\n\n")]
    ns: dict[str, Any] = {"re": re}
    exec(src, ns)  # our own spec file
    drop = ns["_is_winrt_msvcp"]
    assert drop("winrt\\msvcp140.dll") and drop("winrt/MSVCP140.dll")
    assert not drop("msvcp140.dll") and not drop("winrt\\_winrt.cp312-win_amd64.pyd")
    assert not drop("onnxruntime\\capi\\msvcp140.dll")
    assert "a.binaries = [b for b in a.binaries if not _is_winrt_msvcp(b[0])]" in spec
    rel = (root / ".github" / "workflows" / "release.yml").read_text()
    win = rel[rel.index("\n  windows:") : rel.index("\n  macos:")]
    assert win.index("Remove pywinrt's old msvcp140.dll") < win.index("pyinstaller packaging/pyinstaller/away.spec")
    assert "import winrt.windows.media.capture, onnxruntime" in win


def test_selftest_has_a_winrt_check_that_passes_off_windows() -> None:
    check = next(c for c in selftest.CHECKS if c.name == "winrt-camera")
    if not sys.platform.startswith("win"):
        assert check.fn() == "n/a (Windows only)"


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="the pywinrt wheels exist only on Windows")
def test_real_pywinrt_imports_and_answers_on_windows() -> None:  # pragma: no cover - Windows CI
    W.reset_availability()
    ok, why = W.available()
    assert ok, why
    mods = W.load_winrt()
    assert mods["mc"].MediaCaptureSharingMode.EXCLUSIVE_CONTROL == 0
    assert mods["mc"].MediaCaptureSharingMode.SHARED_READ_ONLY == 1
    assert mods["frames"].MediaFrameReaderAcquisitionMode.REALTIME == 0
    assert mods["imaging"].BitmapPixelFormat.BGRA8 == 87
    assert isinstance(mods["props"].MediaEncodingSubtypes.bgra8, str)
    # the asyncio plumbing against the real API: the CI runner has no camera, so this lists none (or WinRT
    # media capture is missing on the server image: None); it must not hang or crash
    names = W.group_names(timeout=20.0)
    assert names is None or isinstance(names, list)
    cap = W.WinrtCapture("No Such Camera 1234", 640, 480, 30, open_timeout=20.0)
    assert not cap.isOpened() and cap.error
    cap.release()


def test_copy_pixels_into_numpy_and_the_buffer_fallback() -> None:
    raw = np.arange(2 * 3 * 4, dtype=np.uint8)

    class Bitmap:
        def copy_to_buffer(self, buf: Any) -> None:
            if isinstance(buf, np.ndarray):
                buf[:] = raw
            else:
                buf.data[:] = raw.tobytes()

    out = W.copy_pixels(Bitmap(), 3, 2)
    assert out.dtype == np.uint8 and np.array_equal(out, raw)

    class OldBitmap:
        def copy_to_buffer(self, buf: Any) -> None:
            if isinstance(buf, np.ndarray):
                raise TypeError("IBuffer expected")
            buf.data[:] = raw.tobytes()

    class Streams:
        class Buffer:
            def __init__(self, n: int) -> None:
                self.data = bytearray(n)

            def __buffer__(self, flags: int) -> memoryview:
                return memoryview(self.data)

    if sys.version_info >= (3, 12):  # __buffer__ (PEP 688)
        assert np.array_equal(W.copy_pixels(OldBitmap(), 3, 2, Streams), raw)
    with pytest.raises(TypeError):
        W.copy_pixels(OldBitmap(), 3, 2)
