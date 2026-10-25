"""v0.1.12/v0.1.13: four bugs from the user's real Acer Predator session (away.log, v0.1.11).

1. ``HD User Facing`` never opened: DirectShow negotiated uncompressed YUY2 (no FOURCC was set) and NVIDIA
   Broadcast keeps its source camera open, so desktop opens fail. -> MJPG, frame-verified opens, a 320x240
   fallback for a second camera, clearer errors with hints, and an automatic NVIDIA Broadcast fallback.
2. The updater said "succeeded" after Inno Setup exit code 5 because the version matched. -> the exit code
   is read first; the helper retries the installer once and relaunches the app itself.
3. Settings crashed on the mouse wheel: Tk passes ``'??'`` for fields an event does not carry.
4. The face gate ignored nearly every utterance ("face present but mouth not moving").
"""

from __future__ import annotations

import logging
import os
import types
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app import updater as U
from away.app.settings_gui import wheel_steps
from away.face.gate import SpeechGate
from away.multicam import capture
from away.multicam.capture import CameraDevice, CameraOpenError, CaptureMode

BROADCAST = "Camera (NVIDIA Broadcast)"
JOYCAM = "JoyCam"
BUILTIN = "HD User Facing"
# the user's DirectShow list (away.log, 2026-10-12): 0: JoyCam; 1: HD User Facing; 2: Camera (NVIDIA Broadcast)
PREDATOR = [(JOYCAM, r"\\?\usb#vid_aaaa&pid_0001"), (BUILTIN, r"\\?\usb#vid_0408&pid_a061"), (BROADCAST, "")]
FOURCC_PROP, W_PROP, H_PROP, FPS_PROP, BUF_PROP = 6, 3, 4, 5, 38
MJPG = capture.fourcc_code("MJPG")


# ====================================================================== fake OpenCV
class FakeCam:
    """One device: ``opens`` per backend, and ``works(fourcc, w, h, fps)`` says whether frames arrive."""

    def __init__(self, opens: dict[Any, bool] | None = None, works: Any = None, frames: int = 3) -> None:
        self.opens = opens or {}
        self.works = works or (lambda fcc, w, h, fps: True)
        self.frames = frames


DEAD = {700: False, 1400: False}


class FakeCap:
    def __init__(self, cv2: FakeCV2, src: Any, api: Any) -> None:
        self.cv2, self.src, self.api = cv2, src, api
        cam = cv2.cams.get((src, api)) or cv2.cams.get(src) or FakeCam()  # (index, backend) first: MSMF numbering
        self.cam = cam
        self.ok = cam.opens.get(api, True)
        self.props: dict[int, Any] = {}
        self.sets: list[tuple[int, Any]] = []
        self.reads = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.ok

    def set(self, prop: int, value: Any) -> bool:
        self.sets.append((prop, value))
        self.props[prop] = value
        return True

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        fcc = self.props.get(FOURCC_PROP, 0)
        w, h, fps = self.props.get(W_PROP), self.props.get(H_PROP), self.props.get(FPS_PROP, 0)
        if not self.ok or not self.cam.works(fcc, w, h, fps) or self.reads > self.cam.frames:
            return False, None
        return True, np.zeros((h or 480, w or 640, 3), np.uint8)

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_DSHOW, CAP_MSMF = 700, 1400
    CAP_PROP_FOURCC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = FOURCC_PROP, W_PROP, H_PROP, FPS_PROP
    CAP_PROP_BUFFERSIZE, COLOR_BGR2RGB = BUF_PROP, 4

    def __init__(self, cams: dict[Any, FakeCam] | None = None) -> None:
        self.cams = cams or {}
        self.caps: list[FakeCap] = []

    def VideoCapture(self, src: Any, api: Any = None) -> FakeCap:
        cap = FakeCap(self, src, api)
        self.caps.append(cap)
        return cap

    @staticmethod
    def flip(f: Any, _c: Any) -> Any:
        return f

    @staticmethod
    def cvtColor(f: Any, _c: Any) -> Any:
        return f

    @property
    def opened(self) -> list[tuple[Any, Any]]:
        return [(c.src, c.api) for c in self.caps]


def _devs() -> list[CameraDevice]:
    return capture.dshow_devices(PREDATOR)


# Media Foundation's own list on the same PC: no virtual camera, and the built-in camera comes first, so its
# MSMF index (0) differs from its DirectShow index (1); MSMF index 1 is JoyCam
MF_LIST = [(BUILTIN, PREDATOR[1][1].upper()), (JOYCAM, PREDATOR[0][1])]
MF_BUILTIN, MF_JOYCAM = 0, 1


def _quick(**kw: Any) -> dict[str, Any]:
    return {"first_frame_s": 0.0, "retry_frame_s": 0.0, "sleep": lambda _s: None, **kw}


def _builtin_dead() -> dict[Any, FakeCam]:
    """The reported failure: the built-in camera opens neither with DirectShow nor with Media Foundation."""
    return {1: FakeCam(opens=DEAD), (MF_BUILTIN, 1400): FakeCam(opens=DEAD)}


@pytest.fixture(autouse=True)
def _windows_and_clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(capture, "_windows", lambda: True)
    monkeypatch.setattr(capture, "list_cameras", lambda **kw: _devs())
    monkeypatch.setattr(capture, "windows_msmf_cameras", lambda: list(MF_LIST))
    capture.reset_session_modes()
    capture._open_now.clear()
    yield
    capture.reset_session_modes()
    capture._open_now.clear()


# ====================================================================== bug 1: MJPG + verified open


@pytest.fixture
def external_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """v0.1.23: NVIDIA Broadcast is never a fallback for the built-in camera (opening it makes Broadcast take the
    built-in camera). These older tests check the fallback mechanics, which still apply to an external camera."""
    monkeypatch.setattr(capture, "broadcast_fallback_allowed", lambda _name: True)


def test_mjpg_is_set_before_size_and_fps_and_a_frame_is_verified() -> None:
    cv2 = FakeCV2()
    got = capture.open_camera(1, 640, 480, 30, cv2=cv2, devices=_devs(), **_quick())
    cap = cv2.caps[0]
    assert cv2.opened == [(1, 700)]  # DirectShow, one open
    props = [p for p, _v in cap.sets]
    assert props[:4] == [FOURCC_PROP, W_PROP, H_PROP, FPS_PROP]  # FOURCC first: DirectShow picks the type on size
    assert cap.sets[0] == (FOURCC_PROP, MJPG)
    assert got.frame is not None and cap.reads == 1 and got.mode == CaptureMode(640, 480, 30, "MJPG")
    assert capture.held_cameras() == {"1"}
    got.cap.release()
    assert capture.held_cameras() == set()


def test_every_reported_fourcc_then_no_fourcc_and_the_mode_is_kept(caplog: Any) -> None:
    cam = FakeCam(works=lambda fcc, w, h, fps: fcc == 0)  # a camera that only streams its default format
    cv2 = FakeCV2({1: cam})
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = capture.open_camera(1, 640, 480, 0, cv2=cv2, devices=_devs(), **_quick())
    assert got.mode == CaptureMode(640, 480, 0, "")
    assert [c.props.get(FOURCC_PROP, 0) for c in cv2.caps] == [
        MJPG,
        capture.fourcc_code("YUY2"),
        capture.fourcc_code("NV12"),
        0,
    ]
    assert all(c.released for c in cv2.caps[:3])
    for m in ("640x480 MJPG", "640x480 YUY2", "640x480 NV12", "640x480 driver format"):
        assert f"trying DirectShow index 1, mode {m}" in caplog.text  # every attempt is logged
    assert "keeping that for this session" in caplog.text
    got.cap.release()
    cv2.caps.clear()
    again = capture.open_camera(1, 640, 480, 0, cv2=cv2, devices=_devs(), **_quick())  # a later retry
    assert len(cv2.caps) == 1 and again.mode.fourcc == ""  # the mode that worked is tried first


def test_device_default_mode_forces_nothing() -> None:
    cam = FakeCam(works=lambda fcc, w, h, fps: w is None and h is None and fcc == 0)
    cv2 = FakeCV2({1: cam})
    got = capture.open_camera(1, 640, 480, 30, cv2=cv2, devices=_devs(), **_quick())
    assert got.mode == capture.DEVICE_DEFAULT and got.mode.describe().startswith("device default")
    assert cv2.caps[-1].sets == [(BUF_PROP, 1)]  # no FOURCC, no size, no fps
    got.cap.release()


def test_empty_fourcc_config_never_sets_one() -> None:
    cv2 = FakeCV2()
    capture.open_camera(1, 640, 480, 0, "", cv2=cv2, devices=_devs(), **_quick()).cap.release()
    assert FOURCC_PROP not in cv2.caps[0].props
    assert capture.capture_modes(640, 480, 0, "", False) == [CaptureMode(640, 480, 0, ""), capture.DEVICE_DEFAULT]


def test_second_camera_falls_back_to_320x240_then_15fps_and_logs_it(caplog: Any) -> None:
    cv2 = FakeCV2(
        {
            0: FakeCam(),
            1: FakeCam(works=lambda fcc, w, h, fps: (w, h, fps) == (320, 240, 15) and fcc == MJPG),
        }
    )
    first = capture.open_camera(0, cv2=cv2, devices=_devs(), **_quick())  # JoyCam is open
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        second = capture.open_camera(1, 640, 480, 30, cv2=cv2, devices=_devs(), **_quick())
    assert second.mode == CaptureMode(320, 240, 15, "MJPG")
    tried = [(c.props.get(W_PROP), c.props.get(FPS_PROP), c.props.get(FOURCC_PROP, 0)) for c in cv2.caps[1:]]
    y, n = capture.fourcc_code("YUY2"), capture.fourcc_code("NV12")
    assert tried == [
        (640, 30, MJPG),
        (640, 30, y),
        (640, 30, n),
        (640, 30, 0),
        (None, None, 0),
        (320, 30, MJPG),
        (320, 15, MJPG),
    ]
    assert "320x240@15 MJPG" in caplog.text and "keeping that for this session" in caplog.text
    second.cap.release()
    n_caps = len(cv2.caps)
    capture.open_camera(1, 640, 480, 30, cv2=cv2, devices=_devs(), **_quick()).cap.release()
    assert len(cv2.caps) == n_caps + 1  # retry 45 s later: the low-res mode works at once
    first.cap.release()


def test_alone_a_camera_does_not_drop_to_low_res() -> None:
    silent = FakeCam(works=lambda fcc, w, h, fps: False)
    cv2 = FakeCV2({1: silent, (MF_BUILTIN, 1400): silent})
    with pytest.raises(CameraOpenError) as err:
        capture.open_camera(1, cv2=cv2, devices=_devs(), **_quick())
    ds = [c for c in cv2.caps if c.api == 700]
    assert len(ds) == 5 and all(c.props.get(W_PROP) in (640, None) for c in ds)  # no 320x240 when alone
    assert err.value.opened and "opened but sent no frames" in str(err.value)
    assert capture.BANDWIDTH_HINT not in str(err.value)


def test_open_failed_and_no_frames_are_different_errors_with_bandwidth_hint(caplog: Any) -> None:
    cv2 = FakeCV2({0: FakeCam(), **_builtin_dead()})
    first = capture.open_camera(0, cv2=cv2, devices=_devs(), **_quick())
    with caplog.at_level(logging.INFO, logger="away.cameras"), pytest.raises(CameraOpenError) as err:
        capture.open_camera(1, cv2=cv2, devices=_devs(), virtual_fallback=False, **_quick())
    e = err.value
    assert not e.opened and str(e).startswith("cannot open camera") and "busy" in str(e)
    assert "DirectShow 1: did not open" in str(e) and "Media Foundation 0: did not open" in str(e)
    assert capture.BANDWIDTH_HINT in str(e) and capture.BANDWIDTH_HINT in caplog.text
    assert "DirectShow could not open index 1" in caplog.text
    assert cv2.opened[1:] == [(1, 700), (MF_BUILTIN, 1400)]  # one open per backend, never per mode
    first.cap.release()
    cv2 = FakeCV2({1: FakeCam(works=lambda *a: False), (MF_BUILTIN, 1400): FakeCam(works=lambda *a: False)})
    with pytest.raises(CameraOpenError) as err2:
        capture.open_camera(1, cv2=cv2, devices=_devs(), low_res=True, **_quick())
    assert err2.value.opened and "opened but sent no frames (tried DirectShow 640x480 MJPG: no frames" in str(
        err2.value
    )


def test_camera_frames_yields_the_verified_frame_first() -> None:
    from away.hands.tracker import camera_frames, probe_camera

    cv2 = FakeCV2({1: FakeCam(frames=3)})
    frames = list(camera_frames(BUILTIN, cv2=cv2, virtual_fallback=False))
    assert len(frames) == 3 and cv2.opened == [(1, 700)] and cv2.caps[0].sets[0] == (FOURCC_PROP, MJPG)
    assert capture.held_cameras() == set()  # released when the generator ends
    assert probe_camera(BUILTIN, cv2=FakeCV2()) == (True, f"camera {BUILTIN} OK")
    ok, msg = probe_camera(BUILTIN, cv2=FakeCV2(_builtin_dead()))
    assert not ok and "cannot open camera" in msg


def test_open_capture_uses_the_spec_fourcc_and_low_res_for_a_secondary() -> None:
    cv2 = FakeCV2({1: FakeCam(works=lambda fcc, w, h, fps: w == 320)})
    spec = C.CameraSpec(BUILTIN, "secondary", fourcc="YUY2")
    cap = capture.open_capture(spec, cv2=cv2)
    assert cv2.caps[0].sets[0] == (FOURCC_PROP, capture.fourcc_code("YUY2"))
    ok, frame = cap.read()  # the frame read while proving the mode comes first
    assert ok and frame.shape[:2] == (240, 320)
    cap.release()


# ====================================================================== v0.1.13: Media Foundation numbering
def test_media_foundation_is_opened_at_its_own_index_not_the_directshow_one(caplog: Any) -> None:
    """DirectShow: 0 JoyCam, 1 HD User Facing, 2 NVIDIA Broadcast. Media Foundation: 0 HD User Facing, 1 JoyCam.
    The old fallback opened MSMF index 1, which is JoyCam (already in use), so the built-in never opened."""
    cv2 = FakeCV2({1: FakeCam(opens={700: False}), (MF_JOYCAM, 1400): FakeCam(opens=DEAD)})
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = capture.open_camera(1, 640, 480, 30, cv2=cv2, devices=_devs(), virtual_fallback=True, **_quick())
    assert cv2.opened == [(1, 700), (MF_BUILTIN, 1400)]  # never (1, 1400)
    assert got.src == 1 and got.name == BUILTIN and not got.fallback  # no NVIDIA Broadcast needed
    assert got.mode == CaptureMode(640, 480, 30, "")
    assert "DirectShow index 1 is Media Foundation index 0 (by device path)" in caplog.text
    assert "cameras (msmf): 0: HD User Facing; 1: JoyCam" in caplog.text
    assert "trying Media Foundation index 0, mode 640x480@30 driver format" in caplog.text
    assert capture.held_cameras() == {"1"}  # held under its DirectShow index (the camera's identity)
    got.cap.release()
    cv2.caps.clear()
    capture.open_camera(1, 640, 480, 30, cv2=cv2, devices=_devs(), **_quick()).cap.release()
    assert cv2.opened == [(MF_BUILTIN, 1400)]  # later retries go straight to what worked


def test_msmf_index_matching_by_name_occurrence_and_fallbacks(caplog: Any) -> None:
    twins = [("USB Camera", ""), ("USB Camera", ""), (BROADCAST, "")]
    devs = capture.dshow_devices(twins)
    mf = [("USB Camera", "a"), ("Other", "b"), ("USB Camera", "c")]
    assert capture.msmf_index(1, devs[1], devs, "'USB Camera'", lambda: mf) == 2  # the 2nd of the same name
    assert capture.msmf_index(0, devs[0], devs, "'USB Camera'", lambda: mf) == 0
    with caplog.at_level(logging.WARNING, logger="away.cameras"):
        assert capture.msmf_index(2, devs[2], devs, f"'{BROADCAST}'", lambda: mf) is None  # DirectShow-only
        assert "not in Media Foundation's camera list" in caplog.text
        assert capture.msmf_index(1, devs[1], devs, "'USB Camera'", lambda: None) == 1  # list unavailable
        assert "Media Foundation list unavailable" in caplog.text
    assert capture.msmf_index(1, None, [], f"'{BUILTIN}'", lambda: list(MF_LIST)) == MF_BUILTIN  # by name only


def test_media_foundation_is_tried_before_the_virtual_fallback(external_camera: None) -> None:
    cv2 = FakeCV2(_builtin_dead())
    got = capture.open_camera(1, cv2=cv2, devices=_devs(), virtual_fallback=True, **_quick())
    assert [o for o in cv2.opened] == [(1, 700), (MF_BUILTIN, 1400), (2, 700)]  # DirectShow, MSMF, Broadcast
    assert got.src == 2 and got.name == BROADCAST
    got.cap.release()


def test_camera_missing_from_media_foundation_skips_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "windows_msmf_cameras", lambda: [(JOYCAM, "")])
    cv2 = FakeCV2({1: FakeCam(opens=DEAD)})
    with pytest.raises(CameraOpenError) as err:
        capture.open_camera(1, cv2=cv2, devices=_devs(), **_quick())
    assert cv2.opened == [(1, 700)] and "Media Foundation" not in str(err.value)


# ====================================================================== bug 1: NVIDIA Broadcast
def test_broadcast_hint_when_a_physical_camera_cannot_be_opened(caplog: Any) -> None:
    cv2 = FakeCV2(_builtin_dead())
    with caplog.at_level(logging.WARNING, logger="away.cameras"), pytest.raises(CameraOpenError) as err:
        capture.open_camera(1, cv2=cv2, devices=_devs(), virtual_fallback=False, **_quick())
    hint = f"{BUILTIN} could not be opened directly; NVIDIA Broadcast's camera shows the same picture."
    assert hint in str(err.value) and hint in caplog.text
    assert "probably using" not in str(err.value)  # Broadcast is not holding it (the user checked)
    assert [s for s, _a in cv2.opened] == [1, MF_BUILTIN]  # no fallback when it is turned off


def test_no_broadcast_hint_without_a_broadcast_camera() -> None:
    cv2 = FakeCV2(_builtin_dead())
    devs = [d for d in _devs() if d.name != BROADCAST]
    with pytest.raises(CameraOpenError) as err:
        capture.open_camera(1, cv2=cv2, devices=devs, virtual_fallback=True, **_quick())
    assert "NVIDIA Broadcast" not in str(err.value)


def test_unopenable_camera_falls_back_to_the_one_free_broadcast_camera(caplog: Any, external_camera: None) -> None:
    cv2 = FakeCV2(_builtin_dead())
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = capture.open_camera(1, 640, 480, 30, cv2=cv2, devices=_devs(), virtual_fallback=True, **_quick())
    assert got.src == 2 and got.name == BROADCAST and got.fallback
    assert "opening the virtual camera 'Camera (NVIDIA Broadcast)' instead" in caplog.text
    assert "virtual fallback" in caplog.text
    assert FOURCC_PROP not in cv2.caps[-1].props  # a virtual camera is opened in its own format
    assert capture.held_cameras() == {"2"}
    got.cap.release()


def test_no_fallback_when_broadcast_is_in_use_or_ambiguous_or_the_failing_camera_is_virtual() -> None:
    cv2 = FakeCV2({0: FakeCam(), 2: FakeCam(), **_builtin_dead()})
    held = capture.open_camera(2, cv2=cv2, devices=_devs(), **_quick())  # Broadcast already open (camera 0 of set)
    with pytest.raises(CameraOpenError):
        capture.open_camera(1, cv2=cv2, devices=_devs(), virtual_fallback=True, **_quick())
    held.cap.release()
    two = [*_devs(), CameraDevice(3, "camera 3", "Camera (NVIDIA Broadcast) #2", virtual=True)]
    with pytest.raises(CameraOpenError):
        capture.open_camera(1, cv2=cv2, devices=two, virtual_fallback=True, **_quick())
    in_use = _devs()
    in_use[2].in_use = True
    with pytest.raises(CameraOpenError):
        capture.open_camera(1, cv2=cv2, devices=in_use, virtual_fallback=True, **_quick())
    cv2 = FakeCV2({2: FakeCam(opens=DEAD)})
    with pytest.raises(CameraOpenError) as err:
        capture.open_camera(2, cv2=cv2, devices=_devs(), virtual_fallback=True, **_quick())
    assert "could not be opened directly" not in str(err.value)


def test_choosing_the_broadcast_camera_explicitly_works_single_and_secondary() -> None:
    from away.hands.tracker import camera_frames

    devs = _devs()
    assert capture.resolve_device(BROADCAST, devs) == 2  # by name: never swapped for a real camera
    cv2 = FakeCV2()
    assert len(list(camera_frames(BROADCAST, cv2=cv2))) == 3
    assert cv2.opened == [(2, 700)] and FOURCC_PROP not in cv2.caps[0].props
    specs = [C.CameraSpec(JOYCAM, "primary"), C.CameraSpec(BROADCAST, "secondary")]
    pinned = capture.pin_camera_set(specs, devs)
    assert [s.device for s in pinned] == [JOYCAM, BROADCAST]  # not auto-skipped
    cv2 = FakeCV2()
    cap = capture.open_capture(pinned[1], cv2=cv2)
    assert cv2.opened == [(2, 700)] and cap.read()[0]
    cap.release()


def test_virtual_fallback_and_fourcc_config() -> None:
    cfg = C.AppConfig()
    assert cfg.hands.virtual_fallback is True and cfg.hands.camera_fourcc == "MJPG"
    assert C.CameraSpec().fourcc == "MJPG"
    cfg2, unknown = C.from_dict(
        {
            "hands": {
                "virtual_fallback": False,
                "camera_fourcc": "",
                "cameras": [{"device": JOYCAM, "fourcc": "yuy2"}, {"device": BUILTIN, "fourcc": ""}],
            }
        }
    )
    assert unknown == [] and cfg2.hands.virtual_fallback is False and cfg2.hands.camera_fourcc == ""
    assert [c.fourcc for c in cfg2.hands.cameras] == ["YUY2", ""]
    for bad in ({"camera_fourcc": "MJPEG"}, {"camera_fourcc": 5}, {"virtual_fallback": "yes"}):
        with pytest.raises(ValueError):
            C.from_dict({"hands": bad})
    with pytest.raises(ValueError, match="fourcc"):
        C.from_dict({"hands": {"cameras": [{"device": 0, "fourcc": "MJ"}, {"device": 1}]}})
    assert capture.capture_modes(640, 480, 30, "MJPG", True) == [
        CaptureMode(640, 480, 30, "MJPG"),
        CaptureMode(640, 480, 30, "YUY2"),
        CaptureMode(640, 480, 30, "NV12"),
        CaptureMode(640, 480, 30, ""),
        capture.DEVICE_DEFAULT,
        CaptureMode(320, 240, 30, "MJPG"),
        CaptureMode(320, 240, 15, "MJPG"),
    ]


def test_config_round_trip_keeps_new_camera_fields(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    cfg.hands.virtual_fallback = False
    cfg.hands.cameras = [C.CameraSpec(JOYCAM, "primary", fourcc=""), C.CameraSpec(BUILTIN, "secondary")]
    back, unknown = C.load(C.save(cfg, tmp_path / "c.yaml"))
    assert unknown == [] and back.hands.virtual_fallback is False
    assert [c.fourcc for c in back.hands.cameras] == ["", "MJPG"]


def test_runner_notifies_the_broadcast_hint_once() -> None:
    from away.app.runner import AwayApp as App

    notes: list[tuple[str, str, Any]] = []
    fake = types.SimpleNamespace(_camera_episode=0, notice=lambda t, m, key=None: notes.append((t, m, key)))
    ev = types.SimpleNamespace(data={"error": capture.broadcast_hint(BUILTIN)})
    App._on_camera_error(fake, ev)  # type: ignore[arg-type]
    App._on_camera_error(fake, types.SimpleNamespace(data={"error": "cannot open camera 'x'"}))  # type: ignore[arg-type]
    assert len(notes) == 1 and notes[0][0] == "Project Away: camera could not be opened"
    assert "shows the same picture" in notes[0][1]


# ====================================================================== bug 2: updater
G_DATA = PureWindowsPath(r"C:\Users\garza\AppData\Local\ProjectAway")
G_EXE = PureWindowsPath(r"C:\Users\garza\AppData\Local\Programs\ProjectAway\ProjectAway.exe")


def _batch(relaunch: bool = True) -> str:
    argv = U.windows_command(G_DATA / "updates" / "s.exe", relaunch, G_EXE.parent, G_DATA / "logs" / U.INSTALL_LOG)
    return U.windows_helper_batch(4321, argv, G_EXE, G_DATA / U.EXIT_CODE_FILE, G_DATA / "logs" / U.HELPER_LOG)


def test_batch_clears_pyinstaller_env_before_the_installer() -> None:
    s = _batch()
    i_env = s.index('for /f "delims==" %%V in (\'set _PYI_ 2^>nul\') do set "%%V="')
    assert 'set "_MEIPASS2="' in s and 'set "PYINSTALLER_RESET_ENVIRONMENT=1"' in s
    assert s.index(U.HELPER_STARTED) < i_env < s.index('start "" /wait')
    assert "cleared inherited PyInstaller variables" in s


def test_batch_kills_the_app_tree_and_retries_a_failed_installer_once() -> None:
    s = _batch()
    assert s.count('taskkill /F /IM "%APPEXE%" /T >>"%LOG%" 2>&1') == 2  # before the install and before the retry
    assert s.count('start "" /wait ') == 2  # the installer runs at most twice
    i_first = s.index('start "" /wait ')
    i_check = s.index('if "%RC%"=="0" goto setup_done')
    i_wait5 = s.index("ping -n 6 127.0.0.1 >nul")
    i_retry = s.index('start "" /wait ', i_first + 1)
    assert i_first < i_check < i_wait5 < i_retry < s.index(":setup_done") < s.index('>"%EXITFILE%" echo %RC%')
    assert "retrying once in 5 s" in s and "installer exit code %RC% (retry)" in s
    assert 'if "%RC%"=="8" goto setup_done' in s  # "restart needed" is not retried


def test_batch_relaunches_once_and_never_doubles_inno_run_entry() -> None:
    s = _batch()
    lines = s.splitlines()
    launch = [ln for ln in lines if ln.startswith('start "" "%APPDIR%')]
    assert launch == ['start "" "%APPDIR%\\%APPEXE%" tray']  # exactly one relaunch line
    assert 'set "APPDIR=C:\\Users\\garza\\AppData\\Local\\Programs\\ProjectAway"' in s
    i_code = s.index('>"%EXITFILE%" echo %RC%')
    i_loop = s.index(":wait_relaunch")
    i_seen = s.index("if not errorlevel 1 goto already_running")
    i_bound = s.index("if %RWAIT% GEQ 10 goto do_relaunch")
    i_launch = s.index(launch[0])
    assert i_code < i_loop < i_seen < i_bound < s.index(":do_relaunch") < i_launch
    # the "already running" branch skips the launch
    block = s[s.index(":already_running") : s.index(":do_relaunch")]
    assert "goto finish" in block and "not starting another copy" in block
    assert s.index(":finish") > i_launch and s.rstrip().endswith("exit /b 0")
    # a missing installer still relaunches the old copy
    missing = s[s.index("installer file not found") : s.index(":setup_found")]
    assert "goto relaunch" in missing and '>"%EXITFILE%" echo launch-failed' in missing
    assert "relaunching Project Away" in s


def test_batch_without_relaunch_has_no_relaunch() -> None:
    s = _batch(relaunch=False)
    assert 'set "APPDIR="' in s and 'if "%APPDIR%"=="" goto finish' in s


def _simulate_relaunch(s: str, running: list[bool]) -> list[str]:
    """A tiny interpreter for the relaunch block: ``running[k]`` = tasklist finds the app on check k."""
    lines = s.splitlines()
    labels = {ln[1:]: i for i, ln in enumerate(lines) if ln.startswith(":")}
    pc, env, checks, out = labels["relaunch"], {"RWAIT": 0, "APPDIR": "x"}, 0, []
    errorlevel = 0
    while pc < len(lines):
        ln = lines[pc]
        pc += 1
        if ln.startswith("goto "):
            pc = labels[ln[5:]]
        elif ln.startswith("set /a RWAIT=0"):
            env["RWAIT"] = 0
        elif ln.startswith("set /a RWAIT+=1"):
            env["RWAIT"] = int(env["RWAIT"]) + 1
        elif ln.startswith("tasklist"):
            errorlevel = 0 if running[min(checks, len(running) - 1)] else 1
            checks += 1
        elif ln == "if not errorlevel 1 goto already_running":
            if errorlevel == 0:
                pc = labels["already_running"]
        elif ln.startswith("if %RWAIT% GEQ"):
            if int(env["RWAIT"]) >= int(ln.split()[3]):
                pc = labels[ln.split()[-1]]
        elif ln.startswith("if exist"):
            pc = labels[ln.split()[-1]]
        elif ln.startswith('start ""'):
            out.append("launch")
        elif ln.startswith("exit /b"):
            break
    return out


def test_relaunch_logic_simulated() -> None:
    s = _batch()
    assert _simulate_relaunch(s, [False]) == ["launch"]  # failed install: nobody relaunched it -> we do
    assert _simulate_relaunch(s, [False, False, True]) == []  # Inno's [Run] started it 2 s later: no double
    assert _simulate_relaunch(s, [True]) == []


def test_inno_exit_codes_are_explained() -> None:
    expect = {
        "1": "failed to initialize",
        "2": "cancelled before the installation started",
        "3": "fatal error while preparing",
        "4": "fatal error during the installation",
        "5": "aborted",
        "6": "killed",
        "7": "'Preparing to install' step decided setup cannot proceed",
        "8": "needs a restart",
    }
    for code, words in expect.items():
        assert words in U.inno_exit_text(code) and f"exit code {code}" in U.inno_exit_text(code)
    assert U.inno_exit_text("42") == "the installer exited with code 42"


def _state(tmp: Path, code: str | None, to: str = "0.1.11") -> Path:
    setup = tmp / "updates" / f"ProjectAway-{to}-windows-x64-setup.exe"
    setup.parent.mkdir(parents=True, exist_ok=True)
    setup.write_bytes(b"MZ")
    U.write_state(tmp, "0.1.10", to, setup.name, setup)
    (tmp / "logs").mkdir(exist_ok=True)
    (tmp / "logs" / U.HELPER_LOG).write_text(f"x installer exit code {code}\nx {U.HELPER_DONE}\n")
    if code is not None:
        (tmp / U.EXIT_CODE_FILE).write_text(f"\ufeff{code}\r\n", encoding="utf-8")
    return setup


def test_failed_installer_is_reported_even_when_the_version_matches(tmp_path: Path) -> None:
    """The user's case: exit code 5, the next manual start runs v0.1.11 and used to say 'succeeded'."""
    setup = _state(tmp_path, "5")
    out = U.previous_install(tmp_path, "0.1.11")
    assert out is not None and out.status == "partial" and not out.ok
    assert "may not have installed completely" in out.message and "exit code 5" in out.message
    assert "Run update installer" in out.message and out.installer == setup
    assert not (tmp_path / U.EXIT_CODE_FILE).exists() and U.previous_install(tmp_path, "0.1.11") is None


def test_zero_or_missing_exit_code_with_matching_version_is_success(tmp_path: Path) -> None:
    _state(tmp_path, "0")
    out = U.previous_install(tmp_path, "0.1.11")
    assert out is not None and out.ok
    _state(tmp_path, None)
    out = U.previous_install(tmp_path, "0.1.11")
    assert out is not None and out.ok


def test_manager_offers_run_installer_after_a_partial_install(tmp_path: Path, caplog: Any) -> None:
    setup = _state(tmp_path, "5")
    notes: list[tuple[str, str]] = []
    opened: list[Path] = []
    m = U.UpdateManager(
        C.UpdatesConfig(), notify=lambda t, msg: notes.append((t, msg)), data_dir=tmp_path, current="0.1.11"
    )
    m.open_file = opened.append  # type: ignore[assignment]
    with caplog.at_level(logging.INFO, logger="away.updater"):
        res = m.report_previous()
    assert res is not None and res[0] is False
    assert notes and notes[0][0] == "Project Away update incomplete"
    assert "succeeded" not in caplog.text and "FAILED" in caplog.text
    assert m.run_installer_label() == U.RUN_INSTALLER_LABEL and m.failed_installer == setup


# ====================================================================== bug 3: mouse wheel
def test_wheel_steps_ignores_unparsable_tk_fields() -> None:
    assert wheel_steps("??", "??", platform="win32") == 0  # the reported crash: int('??')
    assert wheel_steps(120, "??", platform="win32") == -1
    assert wheel_steps("-240", "??", platform="win32") == 2
    assert wheel_steps("??", 4, platform="linux") == -1 and wheel_steps("??", "5", platform="linux") == 1
    assert wheel_steps(None, None, platform="win32") == 0 and wheel_steps("1.0", 0, platform="darwin") == -1
    assert wheel_steps(True, False, platform="win32") == 0


def test_settings_wheel_handler_survives_question_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import settings_gui as G

    handlers: dict[str, Any] = {}

    class Root:
        def bind_all(self, seq: str, fn: Any, add: str = "") -> None:
            handlers[seq] = fn

        def bind_class(self, *_a: Any) -> None:
            pass

        def winfo_containing(self, *_a: Any) -> Any:
            return types.SimpleNamespace(winfo_class=lambda: "TFrame", master=None)

    win = types.SimpleNamespace(root=Root(), _scrollers=[])
    G.SettingsWindow._bind_wheel(win)  # type: ignore[arg-type]
    ev = types.SimpleNamespace(delta="??", num="??", x_root=1, y_root=1)
    assert handlers["<MouseWheel>"](ev) is None  # no ValueError


# ====================================================================== bug 4: face gate
def _replay(
    frames: int,
    face_fraction: float,
    max_score: float | None,
    mouth_moving: int,
    *,
    frontal: bool | None = True,
    rest: float = 0.04,
    gate: SpeechGate | None = None,
    active: int | None = None,
    n_faces: int = 1,
    user_present: bool | None = None,
) -> dict[str, Any]:
    """Rebuild an utterance like one logged in away.log: ``frames`` face-tracking frames, a face in
    ``face_fraction`` of them, ``mouth_moving`` frames at ``max_score`` and the rest at a resting level.
    Ten seconds of the user sitting quietly at the screen come first (the adaptive baseline)."""
    g = gate or SpeechGate()
    if g.primary is None:
        g.enroll({0: 0.3})
    t = 0.0
    for i in range(200):
        g.observe(t, None, 1, rest + 0.004 * ((i % 5) - 2), True)
        t += 0.05
    t += 1.0
    t0 = t
    faces = round(frames * face_fraction)
    for i in range(frames):
        if i < faces:
            sc = max_score if (i < mouth_moving and max_score is not None) else rest
            if i == mouth_moving and max_score is not None and mouth_moving == 0:
                sc = max_score  # the one frame that set max_score in the log
            g.observe(t, active, n_faces, sc, frontal, user_present)
        else:
            g.observe(t, None, 0, None)
        t += 0.05
    return g.explain(t0, t - 0.05)


# gate dicts copied from the v0.1.11 session (frames, face_fraction, max_score, mouth_moving_frames)
LOGGED_REJECTIONS = [
    (21, 1.0, 0.06, 0),
    (32, 1.0, 0.11, 0),
    (59, 1.0, 0.09, 0),
    (36, 1.0, 0.08, 0),
    (31, 0.71, 0.07, 0),
    (34, 0.68, 0.09, 0),
    (14, 0.93, 0.09, 0),
    (43, 0.91, 0.1, 0),
    (56, 0.88, 0.1, 0),
    (53, 0.74, 0.12, 0),
    (100, 0.49, 0.15, 16),
]


@pytest.mark.parametrize(("frames", "face_fraction", "max_score", "moving"), LOGGED_REJECTIONS)
def test_logged_utterances_are_now_allowed(frames: int, face_fraction: float, max_score: float, moving: int) -> None:
    info = _replay(frames, face_fraction, max_score, moving)
    assert info["allowed"], info
    assert info["face_fraction"] == pytest.approx(face_fraction, abs=0.02)


def test_weak_mouth_motion_counts_relative_to_the_users_baseline() -> None:
    info = _replay(40, 1.0, 0.07, 10, frontal=None)  # pose unknown: only the relative threshold can allow it
    assert info["allowed"] and info["reason"] == "facing the screen and speaking"
    assert info["mouth_threshold"] < 0.07 and info["baseline"] == pytest.approx(0.04, abs=0.005)
    assert info["mouth_moving_frames"] == 10


def test_presence_rule_needs_a_frontal_enrolled_face() -> None:
    allowed = _replay(30, 1.0, 0.05, 0)
    assert allowed["allowed"] and "facing it (mouth motion weak)" in allowed["reason"]
    assert allowed["frontal_fraction"] == 1.0
    turned = _replay(30, 1.0, 0.05, 0, frontal=False)  # looking at another monitor, mouth still
    assert not turned["allowed"] and turned["reason"] == "face present but mouth not moving"
    unknown = _replay(30, 1.0, 0.05, 0, frontal=None)
    assert not unknown["allowed"]
    little = _replay(30, 0.5, 0.05, 0)  # frontal in only half the frames
    assert not little["allowed"]


def test_negative_no_face_and_other_speaker() -> None:
    assert _replay(40, 0.0, None, 0)["reason"] == "no face at the screen"
    g = SpeechGate()
    g.enroll({0: 0.3, 1: 0.1})
    info = _replay(40, 1.0, 0.3, 0, gate=g, active=1, n_faces=2)  # the other person is the active speaker
    assert not info["allowed"] and info["reason"] == "someone else was speaking"
    g2 = SpeechGate()
    g2.enroll({0: 0.3})
    away = _replay(40, 1.0, 0.04, 0, gate=g2, user_present=False)  # someone else sits there, quiet
    assert not away["allowed"] and away["frontal_fraction"] == 0.0


def test_baseline_threshold_is_clamped() -> None:
    g = SpeechGate()
    assert g.mouth_threshold(0.0) == (g.loose_score, None)  # no history yet
    for i in range(50):
        g.observe(i * 0.05, None, 1, 0.5)  # a user who rests with an open mouth
    thr, base = g.mouth_threshold(10.0)
    assert base == pytest.approx(0.5) and thr == g.loose_score
    g2 = SpeechGate()
    for i in range(50):
        g2.observe(i * 0.05, None, 1, 0.0)
    assert g2.mouth_threshold(10.0)[0] == g2.min_loose


def test_pipeline_passes_face_pose_to_the_gate() -> None:
    from away.app.runner import Pipeline

    p = Pipeline(C.AppConfig())
    box = [(0.5, 0.5, 0.3)]
    p.on_faces(0.0, box, [{}], [True])
    for i in range(1, 40):
        p.on_faces(i * 0.05, box, [{"jawOpen": 0.02}], [True])
    info = p.gate.explain(1.0, 1.9)
    assert info["allowed"] and info["frontal_fraction"] == 1.0
    p2 = Pipeline(C.AppConfig())
    for i in range(40):
        p2.on_faces(i * 0.05, box, [{"jawOpen": 0.02}])  # no pose (older callers): the presence rule is off
    assert not p2.gate.explain(1.0, 1.9)["allowed"]


def test_frontal_from_face_mesh_points() -> None:
    from away.face.landmarker import BROWS_MID, CHIN, EYE_OUTER_A, EYE_OUTER_B, NOSE_TIP, frontal_from_points

    xs, ys = np.full(478, 0.5), np.full(478, 0.5)
    xs[EYE_OUTER_A], xs[EYE_OUTER_B] = 0.4, 0.6
    ys[BROWS_MID], ys[CHIN], ys[NOSE_TIP] = 0.38, 0.70, 0.50
    assert frontal_from_points(xs, ys) is True
    xs[NOSE_TIP] = 0.585  # head turned ~45 degrees
    assert frontal_from_points(xs, ys) is False
    xs[NOSE_TIP], ys[NOSE_TIP] = 0.5, 0.68  # chin down a lot
    assert frontal_from_points(xs, ys) is False
    assert frontal_from_points(xs[:10], ys[:10]) is None


def test_gate_hint_fires_less_often() -> None:
    from away.voice.always_on import GateHint

    h = GateHint()
    rej = {"decision": "ignore", "reason": "face gate: you were not facing the screen and speaking"}
    fired = [h.observe(rej) for _ in range(12)]
    assert h.threshold == 10 and fired.index(True) == 9 and fired.count(True) == 1


def test_version_is_0_1_13() -> None:
    from away import __version__

    assert __version__ == "0.1.33"
    root = Path(__file__).parents[1]
    assert 'version = "0.1.33"' in (root / "pyproject.toml").read_text()
    assert '#define AppVersion "0.1.33"' in (root / "packaging" / "windows" / "away.iss").read_text()
    assert os.path.exists(root / "src" / "away" / "__init__.py")


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="real Media Foundation enumeration needs Windows")
def test_real_media_foundation_enumeration_runs_on_windows() -> None:
    """On the Windows CI runner (no camera): the ctypes MFEnumDeviceSources path returns a list (empty there)
    without raising or hanging; a server image without Media Foundation (no mf.dll) is tolerated."""
    import threading

    out: list[Any] = []

    def work() -> None:
        try:
            out.append(capture._msmf_enumerate())
        except OSError as exc:  # mf.dll / mfplat.dll missing on the server image
            out.append(("no-mf", str(exc)))

    th = threading.Thread(target=work)
    th.start()
    th.join(10)
    assert out, "Media Foundation enumeration hung"
    res = out[0]
    if isinstance(res, tuple):
        pytest.skip(f"Media Foundation not installed on this runner: {res[1]}")
    if res is None:
        pytest.skip("Media Foundation did not start on this runner (MFStartup / MFEnumDeviceSources failed)")
    assert all(isinstance(n, str) and isinstance(link, str) for n, link in res)
