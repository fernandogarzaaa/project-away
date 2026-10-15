"""Proposal #10: ``run_diagnose`` returns structured per-camera results for agents/MCP, not just prose.

The prose ``lines`` keep feeding the Settings UI and the CLI; the new ``result`` dict travels through the
``AWAY-DIAGNOSE-JSON:`` marker line into the Agent Link job ``result`` (``diagnostics`` action="job").
No real camera is touched: every device call goes through the fakes below.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import cli as cli_mod
from away import config as cfgmod
from away.agentlink.jobs import JobRunner, _diagnose_marker
from away.multicam import capture, diagnose
from away.multicam.capture import CameraDevice
from away.multicam.diagnose import RESULT_MARKER, DiagnoseReport
from away.multicam.winrt_capture import NativeFormat

DSHOW, MSMF = 700, 1400
FOURCC_PROP, W_PROP, H_PROP, FPS_PROP = 6, 3, 4, 5


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

    def isOpened(self) -> bool:
        return self.opens

    def set(self, prop: int, value: Any) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> Any:
        if prop == FOURCC_PROP:
            return float(self.props.get(FOURCC_PROP, capture.fourcc_code("YUY2")))
        return float(self.props.get(prop, 0) or 0)

    def read(self) -> tuple[bool, Any]:
        if not self.opens or self.rate <= 0:
            self.cv2.clock.t += 0.05
            return False, None
        self.cv2.clock.t += 1.0 / self.rate
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        pass


class FakeCV2:
    CAP_DSHOW, CAP_MSMF = DSHOW, MSMF
    CAP_PROP_FOURCC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = FOURCC_PROP, W_PROP, H_PROP, FPS_PROP
    CAP_PROP_EXPOSURE, CAP_PROP_AUTO_EXPOSURE, CAP_PROP_BUFFERSIZE = 15, 21, 38

    def __init__(self, devices: dict[Any, tuple[bool, float]], clock: Clock) -> None:
        self.devices = devices
        self.clock = clock

    def VideoCapture(self, src: Any, api: Any = None) -> FakeCap:
        return FakeCap(self, src, api)


class FakeWinrtCap:
    """Streams ``rate`` fps on the fake clock when ``opens``; otherwise ``isOpened()`` is False."""

    def __init__(self, cv2: FakeCV2, rate: float, opens: bool = True, error: str = "") -> None:
        self.cv2, self.rate, self.opens = cv2, rate, opens
        self.error = error if not opens else ""

    def isOpened(self) -> bool:
        return self.opens

    def read(self) -> tuple[bool, Any]:
        if self.rate <= 0 or not self.opens:
            self.cv2.clock.t += 0.05
            return False, None
        self.cv2.clock.t += 1.0 / self.rate
        return True, np.zeros((360, 640, 3), np.uint8)

    def get(self, prop: int) -> float:
        return {3: 640.0, 4: 360.0, 5: 30.0, 6: float(capture.fourcc_code("MJPG"))}.get(prop, 0.0)

    def release(self) -> None:
        pass


def nf(w: int, h: int, fps: float, sub: str = "MJPG", i: int = 0) -> NativeFormat:
    return NativeFormat(w, h, fps, sub, i)


@pytest.fixture(autouse=True)
def _windows_and_clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(capture, "_windows", lambda: True)
    capture.reset_session_modes()
    yield
    capture.reset_session_modes()


def _run(
    devs: list[CameraDevice],
    cv2: FakeCV2,
    clock: Clock,
    **kw: Any,
) -> DiagnoseReport:
    args: dict[str, Any] = {
        "cv2": cv2,
        "windows": True,
        "msmf_lister": lambda: [],
        "winrt_available": lambda: (True, ""),
        "winrt_formats": lambda n, o: ([nf(640, 480, 30)], ""),
        "winrt_open": lambda n, f, o: FakeWinrtCap(cv2, 30.0),
        "max_winrt_formats": 1,
        "measure_s": 0.2,
        "clock": clock,
        "sleep": clock.sleep,
        "enumerated_by": "dshow",
    }
    args.update(kw)
    return diagnose.run_diagnose(devs, **args)


def _camera_schema(cam: dict[str, Any]) -> None:
    assert set(cam) == {"name", "enumerated_by", "backends", "driver", "holder", "verdict"}
    assert set(cam["driver"]) == {"status", "config_error"}
    for b in cam["backends"].values():
        assert set(b) == {"opened", "fps", "error"}
        assert isinstance(b["opened"], bool)
        assert b["fps"] is None or isinstance(b["fps"], (int, float))
        assert b["error"] is None or isinstance(b["error"], str)
    assert cam["verdict"] in ("ok", "driver", "held", "privacy", "unknown")
    json.dumps(cam)  # JSON-serializable for REST/MCP


def test_structured_result_for_working_camera() -> None:
    clock = Clock()
    cv2 = FakeCV2({(0, DSHOW): (True, 30.0)}, clock)
    report = _run([CameraDevice(0, "camera 0", "JoyCam")], cv2, clock)
    assert isinstance(report, DiagnoseReport)
    assert report.result["stopped"] is False
    assert len(report.result["cameras"]) == 1
    cam = report.result["cameras"][0]
    _camera_schema(cam)
    assert cam["name"] == "JoyCam" and cam["enumerated_by"] == "dshow"
    assert cam["backends"]["dshow"] == {"opened": True, "fps": 30.0, "error": None}
    assert cam["backends"]["msmf"] == {"opened": False, "fps": None, "error": "not listed by Media Foundation"}
    assert cam["backends"]["winrt"]["opened"] is True and cam["backends"]["winrt"]["fps"] == 30.0
    assert cam["driver"] == {"status": "ok", "config_error": None}
    assert cam["holder"] is None and cam["verdict"] == "ok"
    # the prose lines are unchanged for the Settings UI
    assert any("640x480@30 MJPG: ok 30.0 fps" in ln for ln in report.lines)


def test_structured_result_driver_verdict_for_dead_camera() -> None:
    """The user's broken internal camera: nothing opens, Media Foundation does not list it, WinRT reports
    "Element not found" -> verdict "driver"."""
    clock = Clock()
    cv2 = FakeCV2({(1, DSHOW): (False, 0.0)}, clock)
    report = _run(
        [CameraDevice(1, "camera 1", "HD User Facing")],
        cv2,
        clock,
        winrt_formats=lambda n, o: ([], "WinRT could not initialise the camera (Element not found)"),
    )
    cam = report.result["cameras"][0]
    _camera_schema(cam)
    assert cam["verdict"] == "driver"
    assert cam["driver"]["status"] == "error"
    assert cam["driver"]["config_error"] is not None and "Element not found" in cam["driver"]["config_error"]
    assert cam["backends"]["dshow"]["error"] == "did not open"
    assert cam["backends"]["msmf"]["error"] == "not listed by Media Foundation"


def test_structured_result_held_verdict() -> None:
    clock = Clock()
    cv2 = FakeCV2({(1, DSHOW): (False, 0.0)}, clock)
    dev = CameraDevice(1, "camera 1", "Busy Cam")
    dev.in_use = True
    report = _run([dev], cv2, clock, winrt_available=lambda: (False, "not installed"))
    cam = report.result["cameras"][0]
    _camera_schema(cam)
    assert cam["holder"] == "Project Away" and cam["verdict"] == "held"


def test_structured_result_privacy_verdict() -> None:
    clock = Clock()
    cv2 = FakeCV2({(0, DSHOW): (False, 0.0)}, clock)

    def deny(name: str, f: Any, occ: int) -> Any:
        raise OSError("Access is denied")

    report = _run([CameraDevice(0, "camera 0", "Gated Cam")], cv2, clock, winrt_open=deny)
    cam = report.result["cameras"][0]
    _camera_schema(cam)
    assert cam["verdict"] == "privacy"
    assert cam["backends"]["winrt"]["error"] is not None


def test_structured_result_unknown_verdict() -> None:
    clock = Clock()
    report = _run(
        [CameraDevice(0, "camera 0", "Mystery Cam")],
        FakeCV2({(0, DSHOW): (False, 0.0)}, clock),
        clock,
        backends=("winrt",),
        winrt_available=lambda: (False, "not installed"),
    )
    cam = report.result["cameras"][0]
    _camera_schema(cam)
    assert cam["verdict"] == "unknown"
    assert cam["backends"]["winrt"]["error"] == "unavailable: not installed"


def test_no_cameras_structured() -> None:
    clock = Clock()
    report = diagnose.run_diagnose([], cv2=FakeCV2({}, clock), windows=False, clock=clock, sleep=clock.sleep)
    assert report.lines == ["no cameras found"]
    assert report.result == {"cameras": [], "stopped": False}


def test_enumerated_by_defaults_to_last_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "last_source", "wmi")
    clock = Clock()
    report = diagnose.run_diagnose(
        [CameraDevice(0, "camera 0", "JoyCam")],
        cv2=FakeCV2({(0, DSHOW): (True, 30.0)}, clock),
        windows=True,
        backends=("dshow",),
        clock=clock,
        sleep=clock.sleep,
    )
    assert report.result["cameras"][0]["enumerated_by"] == "wmi"


# ----------------------------------------------------------------------------- agentlink plumbing


def _proc_with(lines: list[str]) -> Any:
    class Proc:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.stdout = iter([ln + "\n" for ln in lines])

        def wait(self, timeout: float = 0) -> int:
            return 0

        def kill(self) -> None:
            pass

    return Proc


def _finished(runner: JobRunner) -> Any:
    job = runner.start("cameras")
    for _ in range(200):
        if job.state != "running":
            break
        import time

        time.sleep(0.02)
    assert job.state == "done"
    return job


def test_diagnose_marker_parsed_into_job_result(tmp_path: Path) -> None:
    structured = {
        "cameras": [
            {
                "name": "JoyCam",
                "enumerated_by": "dshow",
                "backends": {"dshow": {"opened": True, "fps": 30.0, "error": None}},
                "driver": {"status": "ok", "config_error": None},
                "holder": None,
                "verdict": "ok",
            }
        ],
        "stopped": False,
    }
    marker = RESULT_MARKER + " " + json.dumps(structured)
    r = JobRunner(tmp_path, command=lambda a: ["away", *a], runner=_proc_with(["camera 0: JoyCam", marker]))
    job = _finished(r)
    assert job.result["exit_code"] == 0
    assert job.result["cameras"] == structured["cameras"]
    assert job.result["stopped"] is False
    assert all(not ln.startswith(RESULT_MARKER) for ln in job.output)  # the marker is not agent prose
    assert "camera 0: JoyCam" in job.output
    # and it rides through the public job dict the REST/MCP layer serves
    assert job.public()["result"]["cameras"] == structured["cameras"]


def test_diagnose_marker_malformed_is_ignored(tmp_path: Path) -> None:
    r = JobRunner(
        tmp_path,
        command=lambda a: ["away", *a],
        runner=_proc_with(["camera 0: JoyCam", RESULT_MARKER + " not json{"]),
    )
    job = _finished(r)
    assert "cameras" not in job.result and "stopped" not in job.result
    assert any(RESULT_MARKER in ln for ln in job.output)  # unparseable marker stays visible, not lost


def test_diagnose_marker_helper() -> None:
    assert _diagnose_marker("camera 0: JoyCam") is None
    assert _diagnose_marker(RESULT_MARKER + " not json{") is None
    assert _diagnose_marker(RESULT_MARKER + " [1]") is None
    assert _diagnose_marker(RESULT_MARKER + ' {"cameras": [], "stopped": true}') == {
        "cameras": [],
        "stopped": True,
    }


# ----------------------------------------------------------------------------- CLI marker line


class _Lease:
    app_running = False
    handed_over = False


def test_cli_prints_marker_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfgmod, "load", lambda p: (cfgmod.AppConfig(), []))
    monkeypatch.setattr(capture, "configure", lambda hands: None)
    structured = {"cameras": [{"name": "JoyCam", "verdict": "ok"}], "stopped": False}

    def fake_runner(devs: Any, **kw: Any) -> DiagnoseReport:
        for d in devs:
            kw["out"](f"camera {d.index}: {d.name}")
        return DiagnoseReport(["camera 0: JoyCam"], structured)

    args = argparse.Namespace(device=None, seconds=0.3, budget=0, config=None)
    rc = cli_mod.cmd_cameras_diagnose(
        args,
        lease_factory=lambda: contextlib.nullcontext(_Lease()),
        runner=fake_runner,
        lister=lambda device: [CameraDevice(0, "camera 0", "JoyCam")],
    )
    assert rc == 0
    printed = capsys.readouterr().out.splitlines()
    markers = [ln for ln in printed if ln.startswith(RESULT_MARKER)]
    assert len(markers) == 1
    assert json.loads(markers[0][len(RESULT_MARKER) :].strip()) == structured


def test_cli_without_structured_runner_prints_no_marker(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old-style runner stubs that return plain line lists (no DiagnoseReport) stay working."""
    monkeypatch.setattr(cfgmod, "load", lambda p: (cfgmod.AppConfig(), []))
    monkeypatch.setattr(capture, "configure", lambda hands: None)

    args = argparse.Namespace(device=None, seconds=0.3, budget=0, config=None)
    rc = cli_mod.cmd_cameras_diagnose(
        args,
        lease_factory=lambda: contextlib.nullcontext(_Lease()),
        runner=lambda devs, **kw: ["camera 0: JoyCam"],
        lister=lambda device: [CameraDevice(0, "camera 0", "JoyCam")],
    )
    assert rc == 0
    assert not any(ln.startswith(RESULT_MARKER) for ln in capsys.readouterr().out.splitlines())
