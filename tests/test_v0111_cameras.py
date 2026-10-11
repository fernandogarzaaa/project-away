"""v0.1.11: Windows camera enumeration in true DirectShow order, cameras stored by name, virtual cameras.

The reported machine (Acer Predator, Windows 11): DirectShow lists ``Camera (NVIDIA Broadcast)`` (virtual) at 0,
the built-in ``HD User Facing`` at 1 and a USB webcam at 2, while the PnP/WMI list (used before) has only the two
real cameras, numbered 0 and 1. Index 0 therefore opened NVIDIA Broadcast (or failed) instead of the built-in.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from typing import Any

import pytest

from away import config as C
from away.app import camera_share as CS
from away.app.camera_loop import retry_labels
from away.events import EventBus
from away.multicam import capture
from away.multicam.capture import CameraDevice

BROADCAST = "Camera (NVIDIA Broadcast)"
BUILTIN = "HD User Facing"
EXTERNAL = "USB2.0 HD UVC WebCam"
PREDATOR = [
    (BROADCAST, ""),
    (BUILTIN, r"\\?\usb#vid_0408&pid_a061&mi_00#6&2b1e2e8&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"),
    (EXTERNAL, r"\\?\usb#vid_1bcf&pid_2284&mi_00#7&1c6b8e5&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"),
]
WMI_NAMES = f"{BUILTIN}\r\n{EXTERNAL}\r\n"  # what PnP/WMI reports: no virtual camera, numbered 0, 1


def _wmi(stdout: str = WMI_NAMES, rc: int = 0) -> Any:
    return lambda *a, **k: types.SimpleNamespace(returncode=rc, stdout=stdout)


def _win_list(entries: Any, **kw: Any) -> list[CameraDevice]:
    return capture.list_cameras(
        sysfs="/nonexistent", platform="win32", run=_wmi(), dshow=lambda: entries, in_use=kw.pop("in_use", set()), **kw
    )


# ====================================================================== enumeration
def test_dshow_enumeration_keeps_directshow_order_and_flags_virtual_cameras():
    devs = _win_list(PREDATOR)
    assert capture.last_source == "dshow"
    assert [(d.index, d.name, d.virtual) for d in devs] == [
        (0, BROADCAST, True),
        (1, BUILTIN, False),
        (2, EXTERNAL, False),
    ]
    assert devs[1].device_path == PREDATOR[1][1] and devs[0].device_path == ""
    assert [d.path for d in devs] == ["camera 0", "camera 1", "camera 2"]


def test_regression_predator_names_and_indices_come_from_the_same_list():
    """The built-in camera is DirectShow index 1, whatever WMI says; asking for it by name opens index 1."""
    devs = _win_list(PREDATOR)
    assert capture.resolve_device(BUILTIN, devs) == 1
    assert capture.resolve_device(EXTERNAL, devs) == 2
    assert capture.resolve_device(BROADCAST, devs) == 0  # a virtual camera can still be chosen explicitly
    # the old WMI-numbered list would have said 0 = HD User Facing: that is the bug
    wmi = capture.list_cameras(sysfs="/nonexistent", platform="win32", run=_wmi(), dshow=lambda: None, in_use=set())
    assert capture.last_source == "wmi" and [(d.index, d.name) for d in wmi] == [(0, BUILTIN), (1, EXTERNAL)]


def test_legacy_index_pointing_at_a_virtual_camera_uses_the_first_real_one(caplog):
    devs = _win_list(PREDATOR)
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        assert capture.resolve_device(0, devs) == 1  # old configs say 0 = the built-in camera
    assert "virtual camera" in caplog.text and BROADCAST in caplog.text
    assert capture.resolve_device(2, devs) == 2  # a real camera index is used as is
    assert capture.resolve_device("1", devs) == 1


def test_two_camera_legacy_set_is_pinned_to_two_real_cameras_by_name():
    """His two-camera config [0, 1]: 0 was Broadcast under DirectShow. Both specs end on distinct real cameras."""
    devs = _win_list(PREDATOR)
    specs = [C.CameraSpec(0, "primary"), C.CameraSpec(1, "secondary")]
    pinned = capture.pin_camera_set(specs, devs)
    assert [s.device for s in pinned] == [EXTERNAL, BUILTIN]
    assert [s.role for s in pinned] == ["primary", "secondary"] and specs[0].device == 0  # originals untouched
    assert all(not d.virtual for d in devs if d.name in {s.device for s in pinned})
    named = [C.CameraSpec(BROADCAST, "primary"), C.CameraSpec(1, "secondary")]  # explicit choice is kept
    assert [s.device for s in capture.pin_camera_set(named, devs)] == [BROADCAST, BUILTIN]
    assert capture.pin_camera_set(specs, []) == specs  # no list: nothing changes


def test_name_resolution_survives_hot_plug_reordering():
    before = _win_list(PREDATOR)
    assert capture.resolve_device(EXTERNAL, before) == 2
    # the user unplugs the webcam and plugs in another; DirectShow renumbers everything
    after = _win_list([(EXTERNAL, PREDATOR[2][1]), ("OBS Virtual Camera", ""), (BUILTIN, PREDATOR[1][1])])
    assert capture.resolve_device(EXTERNAL, after) == 0
    assert capture.resolve_device(BUILTIN, after) == 2
    gone = _win_list([(BROADCAST, ""), (BUILTIN, PREDATOR[1][1])])
    with pytest.raises(ValueError, match="no camera named"):
        capture.resolve_device(EXTERNAL, gone)  # never silently redirected to whatever is at its old index
    assert capture.resolve_device(EXTERNAL, [], hint=2) == 2  # the hint is used only when nothing is listed


def test_duplicate_names_use_device_path_or_occurrence_number():
    entries = [("USB Camera", r"\\?\usb#a"), ("USB Camera", r"\\?\usb#b")]
    devs = _win_list(entries)
    assert [capture.device_label(d, devs) for d in devs] == [r"\\?\usb#a", r"\\?\usb#b"]
    assert capture.resolve_device(r"\\?\usb#b", devs) == 1
    swapped = _win_list(list(reversed(entries)))
    assert capture.resolve_device(r"\\?\usb#b", swapped) == 0
    no_path = _win_list([("USB Camera", ""), ("USB Camera", "")])
    assert [capture.device_label(d, no_path) for d in no_path] == ["USB Camera #1", "USB Camera #2"]
    assert capture.resolve_device("USB Camera #2", no_path) == 1


def test_dshow_failure_falls_back_to_wmi_and_logs_it(caplog):
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        devs = capture.list_cameras(sysfs="/nonexistent", platform="win32", run=_wmi(), dshow=lambda: None)
    assert capture.last_source == "wmi" and [d.name for d in devs] == [BUILTIN, EXTERNAL]
    assert "DirectShow camera enumeration failed" in caplog.text and "cameras (wmi)" in caplog.text


def test_camera_list_is_logged_once_and_again_when_it_changes(caplog):
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        _win_list(PREDATOR)
        _win_list(PREDATOR)  # same list (e.g. every re-open): not logged again
        assert caplog.text.count("cameras (dshow)") == 1
        _win_list(PREDATOR[1:])  # hot-plug: logged
        _win_list(PREDATOR[1:], log_list=True)  # start-up / Settings 'Refresh list': always logged
    assert caplog.text.count("cameras (dshow)") == 3
    assert "0: Camera (NVIDIA Broadcast) (virtual)" in caplog.text and "1: HD User Facing" in caplog.text


def test_held_camera_is_matched_by_name_too():
    devs = _win_list(PREDATOR, in_use={BUILTIN})
    assert [d.in_use for d in devs] == [False, True, False]


@pytest.mark.parametrize(
    ("name", "virtual"),
    [
        (BROADCAST, True),
        ("OBS Virtual Camera", True),
        ("Iriun Webcam", True),
        ("DroidCam Source 3", True),
        ("Camo", True),
        ("Snap Camera", True),
        ("ManyCam Virtual Webcam", True),
        ("XSplit VCam", True),
        ("e2eSoft iVCam", True),
        (BUILTIN, False),
        ("Integrated Camera", False),
        ("Logitech BRIO", False),
        ("Camouflage Cam", False),
    ],
)
def test_virtual_camera_names(name, virtual):
    assert capture.is_virtual_camera(name) is virtual


@pytest.mark.skipif(sys.platform != "win32", reason="real DirectShow COM enumeration needs Windows")
def test_real_directshow_com_enumeration_on_windows():
    """Exercises the ctypes ICreateDevEnum/IEnumMoniker/IPropertyBag path on a CI runner (no camera there, so
    the always-populated DirectShow filter category is read; the video category must at least not fail)."""
    filters = capture._dshow_enumerate(capture.CLSID_LegacyAmFilterCategory)
    assert filters, "DirectShow filter category should never be empty on Windows"
    assert all(isinstance(n, str) and isinstance(p, str) for n, p in filters) and any(n for n, _ in filters)
    out: list[Any] = []
    th = threading.Thread(target=lambda: out.append(capture._dshow_enumerate(capture.CLSID_VideoInputDeviceCategory)))
    th.start()
    th.join(10)
    assert out and out[0] is not None


# ====================================================================== opening: DirectShow everywhere
class FakeCap:
    def __init__(self, src: Any, api: Any, ok: bool) -> None:
        self.src, self.api, self.ok, self.released = src, api, ok, False
        self.reads = 0

    def isOpened(self) -> bool:
        return self.ok

    def set(self, *_a: Any) -> bool:
        return True

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        return (self.reads <= 2, None if self.reads > 2 else "frame")

    def release(self) -> None:
        self.released = True


def fake_cv2(dshow_ok: bool = True) -> Any:
    opened: list[tuple[Any, Any]] = []

    def video_capture(src: Any, api: Any = None) -> FakeCap:
        opened.append((src, api))
        return FakeCap(src, api, ok=dshow_ok or api != 700)

    return types.SimpleNamespace(
        VideoCapture=video_capture,
        CAP_DSHOW=700,
        CAP_MSMF=1400,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        COLOR_BGR2RGB=4,
        flip=lambda f, _c: f,
        cvtColor=lambda f, _c: f,
        opened=opened,
    )


def test_single_camera_mode_opens_the_named_camera_with_directshow(monkeypatch):
    from away.hands.tracker import camera_frames, probe_camera

    monkeypatch.setattr(capture, "_windows", lambda: True)
    monkeypatch.setattr(capture, "list_cameras", lambda **kw: capture.dshow_devices(PREDATOR))
    cv2 = fake_cv2()
    frames = list(camera_frames(BUILTIN, cv2=cv2))
    assert len(frames) == 2 and cv2.opened == [(1, 700)]  # index 1, CAP_DSHOW: not the default backend (MSMF)
    cv2.opened.clear()
    list(camera_frames(0, cv2=cv2))  # legacy default index 0 = NVIDIA Broadcast -> the built-in camera
    assert cv2.opened == [(1, 700)]
    cv2.opened.clear()
    assert probe_camera(BUILTIN, cv2=cv2) == (True, f"camera {BUILTIN} OK") and cv2.opened == [(1, 700)]
    ok, msg = probe_camera("Logitech BRIO", cv2=cv2)
    assert not ok and "no camera named" in msg


def test_media_foundation_only_when_directshow_fails_and_it_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(capture, "_windows", lambda: True)
    cv2 = fake_cv2(dshow_ok=False)
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        cap = capture.open_video(1, cv2)
    assert cv2.opened == [(1, 700), (1, 1400)] and cap.isOpened()
    assert "DirectShow could not open" in caplog.text and "Media Foundation" in caplog.text
    cv2 = fake_cv2()
    capture.open_video(1, cv2)
    assert cv2.opened == [(1, 700)]  # DirectShow worked: no fallback
    capture.open_video("/dev/video0", cv2)
    assert cv2.opened[-1] == ("/dev/video0", None)  # paths/URLs: default backend
    monkeypatch.setattr(capture, "_windows", lambda: False)
    capture.open_video(1, cv2)
    assert cv2.opened[-1] == (1, None)  # Linux/macOS: default backend


# ====================================================================== config
def test_config_camera_index_accepts_int_and_name_backward_compatible(tmp_path):
    cfg, _ = C.from_dict({"hands": {"camera_index": 2}})  # v0.1.10 and older
    assert cfg.hands.camera_index == 2 and cfg.hands.camera_hint is None
    C.validate(cfg)
    cfg.hands.camera_index, cfg.hands.camera_hint = BUILTIN, 1
    cfg.hands.cameras = [C.CameraSpec(BUILTIN, "primary", index_hint=1), C.CameraSpec(EXTERNAL, "secondary")]
    p = C.save(cfg, tmp_path / "c.yaml")
    back, unknown = C.load(p)
    assert unknown == [] and back.hands.camera_index == BUILTIN and back.hands.camera_hint == 1
    assert back.hands.cameras[0].index_hint == 1 and back.hands.cameras[1].device == EXTERNAL
    for bad in (True, -1, "", "  ", 1.5):
        cfg.hands.camera_index = bad
        with pytest.raises(ValueError, match="camera_index"):
            C.validate(cfg)
    cfg.hands.camera_index, cfg.hands.camera_hint = 0, -2
    with pytest.raises(ValueError, match="camera_hint"):
        C.validate(cfg)


# ====================================================================== Settings
def test_settings_role_defaults_to_primary_for_the_only_camera(tmp_path):
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    assert m.role_default() == "primary" and m.role_default(0) == "primary"
    # Apply on the only camera with the default role now just works (it said "this is the only camera" before)
    assert m.set_camera(0, "role", m.role_default(0)) is None
    m.device_choices(capture.dshow_devices(PREDATOR))
    assert m.add_camera("2: " + EXTERNAL) is None
    assert m.role_default() == "secondary" and m.role_default(0) == "primary" and m.role_default(1) == "secondary"


def test_settings_calibrate_and_live_confidence_are_gated_in_single_mode(tmp_path):
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    ok, hint = m.multi_tools()
    assert not ok and "two cameras" in hint and len(hint) < 90
    m.cfg.hands.cameras = [C.CameraSpec(BUILTIN, "primary"), C.CameraSpec(EXTERNAL, "secondary")]
    assert m.multi_tools() == (True, "")


def test_settings_device_shows_true_names_and_saves_names(tmp_path):
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    m.camera_share = CS.CameraShare(tmp_path / "s", pid=2)
    choices = m.device_choices(capture.dshow_devices(PREDATOR))
    assert choices[:3] == [f"0: {BROADCAST} (virtual)", f"1: {BUILTIN}", f"2: {EXTERNAL}"]
    assert m.camera_rows()[0]["label"] == f"0: {BROADCAST} (virtual)"  # legacy index 0 shows what it really is
    assert m.set_camera(0, "device", choices[1]) is None
    assert m.cfg.hands.camera_index == BUILTIN and m.cfg.hands.camera_hint == 1
    assert m.camera_rows()[0]["label"] == f"1: {BUILTIN}"
    assert m.parse_device(choices[0]) == BROADCAST  # a virtual camera can be chosen explicitly
    m.listed = capture.dshow_devices(PREDATOR[:2])  # the webcam is unplugged
    assert m.device_display(EXTERNAL) == f"{EXTERNAL} (not connected)"
    # the app holds the camera by name: it is still listed as in use
    CS.CameraShare(tmp_path / "s", pid=1).publish_owner([BUILTIN])
    assert m.camera_rows()[0]["in_use"] is True
    assert f"1: {BUILTIN} (in use by Project Away)" in m.device_choices(capture.dshow_devices(PREDATOR))


def test_settings_remove_back_to_single_keeps_the_name(tmp_path):
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    m.device_choices(capture.dshow_devices(PREDATOR))
    assert m.set_camera(0, "device", f"1: {BUILTIN}") is None
    assert m.add_camera(f"2: {EXTERNAL}") is None
    assert [(c.device, c.index_hint) for c in m.cfg.hands.cameras] == [(BUILTIN, 1), (EXTERNAL, 2)]
    assert m.remove_camera(0) is None
    assert m.cfg.hands.cameras == [] and (m.cfg.hands.camera_index, m.cfg.hands.camera_hint) == (EXTERNAL, 2)


# ====================================================================== CLI
def test_cli_pick_camera_saves_the_name_with_index_hint():
    from away.cli import pick_camera

    devs = capture.dshow_devices(PREDATOR)
    assert pick_camera("1", devs) == (BUILTIN, 1)
    assert pick_camera("user facing", devs) == (BUILTIN, 1)
    assert pick_camera("Broadcast", devs) == (BROADCAST, 0)
    assert pick_camera("/dev/video2", devs) == ("/dev/video2", None)
    assert pick_camera("7", devs) == (7, None)
    assert pick_camera("1", [CameraDevice(1, "camera 1", "camera 1")]) == (1, None)  # no real name: index


def test_cli_cameras_use_and_add(tmp_path, monkeypatch, capsys):
    from away import cli

    monkeypatch.setattr(capture, "list_cameras", lambda **kw: capture.dshow_devices(PREDATOR))
    cfgp = tmp_path / "c.yaml"
    C.save(C.AppConfig(), cfgp)
    assert cli.main(["cameras", "use", "HD User", "--config", str(cfgp)]) == 0
    cfg, _ = C.load(cfgp)
    assert (cfg.hands.camera_index, cfg.hands.camera_hint) == (BUILTIN, 1)
    assert cli.main(["cameras", "add", "2", "--config", str(cfgp)]) == 0
    cfg, _ = C.load(cfgp)
    assert [(c.device, c.role, c.index_hint) for c in cfg.hands.cameras] == [
        (BUILTIN, "primary", 1),
        (EXTERNAL, "secondary", 2),
    ]
    assert cli.main(["cameras", "list", "--config", str(cfgp)]) == 0
    out = capsys.readouterr().out
    assert f"{BROADCAST} (virtual)" in out and "source:" in out


# ====================================================================== retry labelling
def test_retry_labels_name_the_retried_camera_not_its_position():
    specs = [C.CameraSpec(1, "secondary")]  # only camera 1 failed and is retried
    assert retry_labels(specs, {0: "open error: cannot open camera 1"}) == {"1": "open error: cannot open camera 1"}
    named = [C.CameraSpec(EXTERNAL, "secondary"), C.CameraSpec(BUILTIN, "primary")]
    assert retry_labels(named, {1: "x"}) == {BUILTIN: "x"}
    assert retry_labels(named, {5: "y"}) == {"5": "y"}  # out of range: left as is


def test_background_retry_event_is_labelled_with_the_failed_device(tmp_path):
    from away.app.runner import AwayApp

    app = AwayApp.__new__(AwayApp)
    app.cfg = C.AppConfig()
    app.stop_event = threading.Event()
    app.bus = EventBus()
    app.camera_share = CS.CameraShare(tmp_path / "share", pid=1)
    events: list[dict[str, Any]] = []
    app.bus.subscribe("cameras.retry", lambda e: events.append(e.data))
    app._probe_specs = lambda specs, probe_s: {0: f"open error: cannot open camera {specs[0].device}"}

    def run_single(device: Any, until: Any) -> None:
        t0 = time.monotonic()
        while not events and time.monotonic() - t0 < 5:
            time.sleep(0.01)

    app._single_until_retry(run_single, 0, [C.CameraSpec(1, "secondary")], 0.01, 0.1)
    assert events and events[0]["failed"] == {"1": "open error: cannot open camera 1"}
