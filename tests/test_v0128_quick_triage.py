"""v0.1.28: quick-triage camera diagnose.

A full ``kind=cameras`` check sweeps every camera on every backend and mode (~1 minute): an agent doing
triage cannot wait for that for a fast answer. ``quick=true`` opens each camera on its remembered backend in
its remembered mode only (the mode file :func:`capture.saved_modes` / :func:`capture.saved_mode_info` records),
skipping the mode sweep. Every check's structured result always carries ``elapsed_s`` and ``budget_hit`` so a
triage agent can tell a fast failure from a timed-out one.

No real camera is touched: the OpenCV and WinRT pieces run behind the same fakes the v0.1.19 tests use, on a
fake clock.
"""

from __future__ import annotations

import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away.agentlink import groups, jobs
from away.multicam import capture, diagnose
from away.multicam.capture import CameraDevice, CaptureMode

DSHOW, MSMF = 700, 1400


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
        self.opens, self.rate = cv2.opens.get((src, api), (True, 30.0))
        cv2.log.append(("open", src, api))

    def isOpened(self) -> bool:
        return self.opens

    def set(self, prop: int, value: Any) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 0.0

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
    CAP_PROP_FOURCC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = 6, 3, 4, 5
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, opens: dict[Any, tuple[bool, float]], clock: Clock) -> None:
        self.opens = opens
        self.clock = clock
        self.log: list[tuple[Any, ...]] = []

    def VideoCapture(self, src: Any, api: Any = None) -> FakeCap:
        return FakeCap(self, src, api)


def _devs() -> list[CameraDevice]:
    return [CameraDevice(0, "usb#0", "JoyCam"), CameraDevice(1, "usb#1", "HD User Facing")]


def _remembered(monkeypatch: pytest.MonkeyPatch, modes: dict[str, tuple[str, CaptureMode]]) -> None:
    """The mode file remembers these (backend label, mode) pairs per camera name."""
    info = {k: {"fps": 30.0, "slow": False, "best": True} for k in modes}
    monkeypatch.setattr(capture, "saved_modes", lambda: dict(modes))
    monkeypatch.setattr(capture, "saved_mode_info", lambda: dict(info))


def _quick_kw(clock: Clock, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "quick": True,
        "windows": True,
        "backends": ("dshow", "msmf", "winrt"),
        "winrt_available": lambda: (False, "not installed"),
        "measure_s": 0.1,
        "clock": clock,
        "sleep": clock.sleep,
        "budget_s": 0,
    }
    base.update(kw)
    return base


# ===================================================================================== 1. quick triage
def test_quick_opens_only_the_remembered_backend_and_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2({(0, DSHOW): (True, 30.0), (0, MSMF): (False, 0.0)}, clock)
    _remembered(monkeypatch, {"JoyCam": ("DirectShow", CaptureMode(640, 480, 30, "MJPG"))})
    lines = diagnose.run_diagnose([_devs()[0]], cv2=cv2, out=lambda m: None, **_quick_kw(clock))
    text = "\n".join(lines)
    opens = [e for e in cv2.log if e[0] == "open"]
    assert len(opens) == 1 and opens[0][1:] == (0, DSHOW)  # no Media Foundation, no WinRT, no sweep
    assert "quick: remembered DirectShow 640x480@30 MJPG" in text
    assert "trying common modes" not in text
    assert "640x480@30 MJPG: ok" in text


def test_quick_with_remembered_media_foundation_uses_its_index(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2({(5, MSMF): (True, 30.0)}, clock)
    _remembered(monkeypatch, {"JoyCam": ("Media Foundation", CaptureMode(640, 480, 30, "YUY2"))})
    lines = diagnose.run_diagnose(
        [_devs()[0]],
        cv2=cv2,
        out=lambda m: None,
        msmf_lister=lambda: [("Other", "usb#x"), ("JoyCam", "usb#0")],
        **_quick_kw(clock),
    )
    opens = [e for e in cv2.log if e[0] == "open"]
    assert len(opens) == 1 and opens[0][1:] == (1, MSMF)  # Media Foundation's own index, one mode only
    assert "quick: remembered Media Foundation 640x480@30 YUY2" in "\n".join(lines)


def test_quick_without_a_remembered_backend_names_the_camera_and_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    cv2 = FakeCV2({}, clock)
    _remembered(monkeypatch, {})
    lines = diagnose.run_diagnose(_devs(), cv2=cv2, out=lambda m: None, **_quick_kw(clock))
    text = "\n".join(lines)
    assert text.count("quick: no remembered backend for this camera") == 2
    assert not [e for e in cv2.log if e[0] == "open"]


def test_quick_winrt_opens_only_the_remembered_native_format(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.multicam.winrt_capture import NativeFormat

    clock = Clock()
    opened: list[Any] = []

    class WinrtCap:
        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, Any]:
            clock.t += 1 / 30
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self) -> None:
            pass

    def winrt_open(name: str, f: Any, occ: int) -> Any:
        opened.append((name, f))
        return WinrtCap()

    fmts = [NativeFormat(640, 480, 30, "MJPG", 0), NativeFormat(1280, 720, 30, "MJPG", 1)]
    _remembered(monkeypatch, {"JoyCam": ("WinRT", CaptureMode(1280, 720, 30, "MJPG"))})
    lines = diagnose.run_diagnose(
        [_devs()[0]],
        cv2=FakeCV2({}, clock),
        out=lambda m: None,
        winrt_available=lambda: (True, ""),
        winrt_formats=lambda n, o: (fmts, ""),
        winrt_open=winrt_open,
        **{k: v for k, v in _quick_kw(clock).items() if k != "winrt_available"},
    )
    assert len(opened) == 1 and opened[0][1] is fmts[1]  # the remembered native format, not the best-first list
    assert "quick: remembered WinRT 1280x720@30 MJPG" in "\n".join(lines)


def test_quick_reports_a_remembered_camera_that_no_longer_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    cv2 = FakeCV2({(0, DSHOW): (False, 0.0)}, clock)
    _remembered(monkeypatch, {"JoyCam": ("DirectShow", CaptureMode(640, 480, 30, "MJPG"))})
    lines = diagnose.run_diagnose([_devs()[0]], cv2=cv2, out=lambda m: None, **_quick_kw(clock))
    text = "\n".join(lines)
    assert "640x480@30 MJPG: did not open" in text
    assert len([e for e in cv2.log if e[0] == "open"]) == 1  # tried once, no sweep after the failure


# ===================================================================================== 2. elapsed_s / budget_hit
class Proc:
    """A fake diagnose subprocess: the given stdout lines, then exit code 0."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.stdout = iter(line + "\n" for line in lines)

    def wait(self, timeout: float = 0) -> int:
        return 0

    def kill(self) -> None:
        pass


def _runner(tmp_path: Path, proc: Proc) -> tuple[jobs.JobRunner, list[list[str]]]:
    seen: list[list[str]] = []

    def command(a: list[str]) -> list[str]:
        seen.append(list(a))
        return ["away", *a]

    r = jobs.JobRunner(tmp_path / "diag", command=command, runner=lambda *a, **k: proc)
    return r, seen


def _wait_done(r: jobs.JobRunner, jid: str) -> jobs.Job:
    for _ in range(500):
        j = r.jobs[jid]
        if j.state != "running":
            return j
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_job_result_always_carries_elapsed_s_and_budget_hit(tmp_path: Path) -> None:
    r, _ = _runner(tmp_path, Proc(["camera 0: JoyCam", "  quick: remembered DirectShow 640x480@30 MJPG"]))
    job = r.start("cameras")
    _wait_done(r, job.id)
    res = r.lookup(job.id).public()["result"]
    assert res["exit_code"] == 0 and res["budget_hit"] is False
    assert isinstance(res["elapsed_s"], float) and res["elapsed_s"] >= 0.0


def test_job_result_marks_budget_hit_when_diagnose_hits_its_time_limit(tmp_path: Path) -> None:
    r, _ = _runner(tmp_path, Proc(["camera 0: JoyCam", "stopped after 60 s (time limit) - the rest"]))
    job = r.start("cameras")
    _wait_done(r, job.id)
    assert r.lookup(job.id).public()["result"]["budget_hit"] is True


def test_benchmark_result_carries_elapsed_s_and_budget_hit(tmp_path: Path) -> None:
    r = jobs.JobRunner(tmp_path / "diag", command=lambda a: ["away", *a], bench=lambda: {"x_ms": 1.0})
    job = r.start("benchmark")
    _wait_done(r, job.id)
    assert r.lookup(job.id).public()["result"] == {"x_ms": 1.0, "elapsed_s": 0.0, "budget_hit": False}


def test_start_quick_passes_quick_to_the_cli(tmp_path: Path) -> None:
    r, seen = _runner(tmp_path, Proc(["camera 0: JoyCam"]))
    _wait_done(r, r.start("cameras", quick=True).id)
    assert "--quick" in seen[-1]
    _wait_done(r, r.start("cameras").id)
    assert "--quick" not in seen[-1]


# ===================================================================================== 3. wiring
def test_groups_wires_quick_through_to_run_diagnose() -> None:
    method, kw = groups.resolve("diagnostics", {"action": "run", "kind": "cameras", "quick": True})
    assert (method, kw) == ("run_diagnose", {"kind": "cameras", "quick": True})
    method, kw = groups.resolve("diagnostics", {"action": "run", "kind": "cameras"})
    assert (method, kw) == ("run_diagnose", {"kind": "cameras"})  # unset quick stays out


def test_service_run_diagnose_passes_quick_to_the_job_runner() -> None:
    from away.agentlink import service as svcmod

    seen: dict[str, Any] = {}

    class FakeJobs:
        def start(self, kind: str, device: Any = None, seconds: Any = None, quick: bool = False) -> jobs.Job:
            seen.update(kind=kind, device=device, seconds=seconds, quick=quick)
            return jobs.Job("abc12345", kind, 0.0)

    svc = types.SimpleNamespace(require=lambda *a: None, jobs=FakeJobs())
    out = svcmod.Service.run_diagnose(svc, None, "cameras", quick=True)
    assert seen == {"kind": "cameras", "device": None, "seconds": None, "quick": True}
    assert out["kind"] == "cameras" and out["id"] == "abc12345"


def test_cli_diagnose_accepts_quick() -> None:
    from away import cli

    a = cli.build_parser().parse_args(["cameras", "diagnose", "--quick"])
    assert a.quick is True
    assert cli.build_parser().parse_args(["cameras", "diagnose"]).quick is False
