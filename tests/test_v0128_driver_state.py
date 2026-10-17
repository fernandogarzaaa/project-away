"""Proposal #9 (v0.1.28): the driver-state step of ``away cameras diagnose``.

On Windows a camera that enumerates but never opens is usually a driver problem. ``diagnose._driver_state``
runs a PowerShell CIM query (Win32_PnPEntity: Name, Status, ConfigManagerErrorCode) and spells the codes out in
plain words, both as diagnose lines and as structured ``result["driver_state"]`` rows. Every device call goes
through the same injectable-``run`` pattern as ``capture.windows_camera_names``, so CI fakes the CIM output.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from away.multicam import diagnose
from away.multicam.capture import CameraDevice

FAKE_CIM_JSON = json.dumps(
    [
        {"Name": "HD User Facing", "Status": "Error", "ConfigManagerErrorCode": 22},
        {"Name": "JoyCam", "Status": "OK", "ConfigManagerErrorCode": 0},
        {"Name": "Mystery Cam", "Status": "Error", "ConfigManagerErrorCode": 99},
    ]
)


def make_run(stdout: str = FAKE_CIM_JSON, returncode: int = 0, exc: Exception | None = None) -> Any:
    """A fake subprocess.run: records its call, returns ``stdout``/``returncode`` or raises ``exc``."""
    calls: list[dict[str, Any]] = []

    def fake(cmd: Any, **kw: Any) -> Any:
        calls.append({"cmd": cmd, "kw": kw})
        if exc is not None:
            raise exc
        return SimpleNamespace(stdout=stdout, returncode=returncode)

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


class ClosedCV2:
    """A cv2 whose cameras never open: only the driver-state lines matter in these tests."""

    CAP_DSHOW = 700
    CAP_MSMF = 1400

    def VideoCapture(self, *args: Any) -> ClosedCV2:
        return self

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass


def _devs() -> list[CameraDevice]:
    return [CameraDevice(0, "camera 0", "JoyCam"), CameraDevice(1, "camera 1", "HD User Facing")]


def _run_diagnose(**kw: Any) -> tuple[list[str], dict[str, Any]]:
    report = diagnose.run_diagnose(
        _devs(),
        cv2=ClosedCV2(),
        windows=True,
        msmf_lister=lambda: [],
        winrt_available=lambda: (False, "no winrt here"),
        measure_s=0.1,
        **kw,
    )
    return report.lines, report.result


# ----------------------------------------------------------------------------- parsing and words
def test_parse_driver_state_list() -> None:
    rows = diagnose.parse_driver_state(FAKE_CIM_JSON)
    assert rows == [
        {"name": "HD User Facing", "status": "Error", "code": 22},
        {"name": "JoyCam", "status": "OK", "code": 0},
        {"name": "Mystery Cam", "status": "Error", "code": 99},
    ]


def test_parse_driver_state_single_object() -> None:
    rows = diagnose.parse_driver_state(json.dumps({"Name": "Solo Cam", "Status": "OK", "ConfigManagerErrorCode": 0}))
    assert rows == [{"name": "Solo Cam", "status": "OK", "code": 0}]


def test_parse_driver_state_garbage() -> None:
    assert diagnose.parse_driver_state("") == []
    assert diagnose.parse_driver_state("not json") == []
    assert diagnose.parse_driver_state("null") == []
    assert diagnose.parse_driver_state(json.dumps([{"Name": "", "Status": "OK", "ConfigManagerErrorCode": 0}])) == []
    assert diagnose.parse_driver_state(json.dumps([{"Name": "No Code Cam"}])) == [
        {"name": "No Code Cam", "status": "unknown", "code": None}
    ]


def test_driver_state_word() -> None:
    assert diagnose.driver_state_word(0) == "working, no error"
    assert diagnose.driver_state_word(10) == "cannot start"
    assert diagnose.driver_state_word(22) == "disabled"
    assert diagnose.driver_state_word(45) == "not connected to the computer"
    assert diagnose.driver_state_word(None) == "driver state unknown"
    assert diagnose.driver_state_word(99) == "problem code 99 (check Device Manager)"


# ----------------------------------------------------------------------------- the CIM query
def test_windows_driver_state_uses_injectable_run() -> None:
    run = make_run()
    rows = diagnose.windows_driver_state(run)
    assert rows is not None and [r["name"] for r in rows] == ["HD User Facing", "JoyCam", "Mystery Cam"]
    assert len(run.calls) == 1
    cmd, kw = run.calls[0]["cmd"], run.calls[0]["kw"]
    assert cmd[0] == "powershell" and "Get-CimInstance" in cmd[-1] and "Win32_PnPEntity" in cmd[-1]
    assert "ConfigManagerErrorCode" in cmd[-1]
    assert kw["timeout"] == diagnose.DRIVER_STATE_TIMEOUT_S


def test_windows_driver_state_query_failure() -> None:
    assert diagnose.windows_driver_state(make_run(exc=OSError("no powershell"))) is None
    assert diagnose.windows_driver_state(make_run(returncode=1, stdout="boom")) is None


# ----------------------------------------------------------------------------- the diagnose step
def test_run_diagnose_emits_driver_state_on_windows() -> None:
    lines, result = _run_diagnose(driver_state_run=make_run())
    text = "\n".join(lines)
    assert "driver state (from Windows):" in text
    assert "HD User Facing: disabled (code 22)" in text
    assert "JoyCam: working, no error (code 0)" in text
    assert "Mystery Cam: problem code 99 (check Device Manager) (code 99)" in text
    assert result["driver_state"] == [
        {"name": "HD User Facing", "status": "Error", "code": 22, "meaning": "disabled"},
        {"name": "JoyCam", "status": "OK", "code": 0, "meaning": "working, no error"},
        {"name": "Mystery Cam", "status": "Error", "code": 99, "meaning": "problem code 99 (check Device Manager)"},
    ]
    # the driver state comes before the backend probing, so it explains the failures below
    assert lines.index("driver state (from Windows):") < next(i for i, ln in enumerate(lines) if "did not open" in ln)


def test_run_diagnose_driver_state_query_failed() -> None:
    lines, result = _run_diagnose(driver_state_run=make_run(exc=OSError("no powershell")))
    assert any("driver state: could not read the Windows driver list" in ln for ln in lines)
    assert result["driver_state"] == []


def test_run_diagnose_driver_state_no_cameras_listed() -> None:
    lines, result = _run_diagnose(driver_state_run=make_run(stdout="[]"))
    assert any("driver state: Windows lists no cameras" in ln for ln in lines)
    assert result["driver_state"] == []


def test_run_diagnose_skips_driver_state_off_windows() -> None:
    run = make_run()
    report = diagnose.run_diagnose(
        [CameraDevice(None, "/dev/video0", "Integrated Camera")],
        cv2=ClosedCV2(),
        windows=False,
        driver_state_run=run,
    )
    lines = report.lines
    assert not any("driver state" in ln for ln in lines)
    assert report.result["driver_state"] == []
    assert run.calls == []  # the query never runs off Windows


def test_run_diagnose_no_cameras_never_queries_drivers() -> None:
    run = make_run()
    report = diagnose.run_diagnose([], cv2=ClosedCV2(), windows=True, driver_state_run=run)
    assert report.lines == ["no cameras found"]
    assert run.calls == []


def test_run_diagnose_budget_over_skips_driver_state() -> None:
    run = make_run()
    ticks = iter([0.0, 1000.0])  # the second clock() call is already past the budget
    report = diagnose.run_diagnose(
        _devs(),
        cv2=ClosedCV2(),
        windows=True,
        budget_s=60.0,
        clock=lambda: next(ticks, 1000.0),
        sleep=lambda s: None,
        driver_state_run=run,
    )
    lines = report.lines
    assert run.calls == []
    assert not any("driver state" in ln for ln in lines)
    assert any("stopped after 60 s" in ln for ln in lines)


def test_driver_state_survives_crlf_and_bom() -> None:
    rows = diagnose.parse_driver_state(
        '\ufeff[\r\n  {\r\n    "Name": "CRLF Cam",\r\n    "Status": "OK",\r\n    "ConfigManagerErrorCode": 0\r\n  }\r\n]'
    )
    assert rows == [{"name": "CRLF Cam", "status": "OK", "code": 0}]
