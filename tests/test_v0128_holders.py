"""Proposal #13 (v0.1.28): first-class "who is holding my camera" diagnose.

``diagnostics(action="run", kind="holders")`` answers with ``{"now": [...], "last_used": [{app, at}]}``:
registry holders (Windows says the app has a camera open now) + a live scan of known camera apps (a hint, not
proof) + the per-app last-used timestamps Windows stores. Every test is OS-free: the registry reader and the
process list are injected.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from away.agentlink import jobs
from away.multicam import camera_users as CU

UNIX_2026_10_16_NOON = 1792152000  # 2026-10-16T12:00:00+00:00 as a Unix timestamp
FT_2026_10_16_NOON = int(UNIX_2026_10_16_NOON * 10_000_000 + 116444736000000000)


def _entries() -> list[tuple[str, bool, dict[str, Any]]]:
    """One app holding now (registry), one that used it before, one never (start 0), Project Away itself."""
    return [
        (
            "C:#Program Files#NVIDIA Corporation#NVIDIA Broadcast#NVIDIA Broadcast.exe",
            False,
            {"LastUsedTimeStart": 133, "LastUsedTimeStop": 0},
        ),
        (
            "C:#Apps#Zoom.exe",
            False,
            {"LastUsedTimeStart": 130, "LastUsedTimeStop": FT_2026_10_16_NOON},
        ),
        ("C:#x#never.exe", False, {"LastUsedTimeStart": 0, "LastUsedTimeStop": 0}),
        ("C:#Program Files#Project Away#ProjectAway.exe", False, {"LastUsedTimeStart": 1, "LastUsedTimeStop": 0}),
    ]


class _Proc:
    def __init__(self, name: str, exe: str = "") -> None:
        self.info = {"name": name, "exe": exe}


class _FakePsutil:
    def __init__(self, procs: list[tuple[str, str]]) -> None:
        self._procs = procs

    def process_iter(self, attrs: list[str]) -> list[_Proc]:
        return [_Proc(n, e) for n, e in self._procs]


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------ filetime


def test_filetime_to_iso_converts_windows_ticks() -> None:
    assert CU.filetime_to_iso(FT_2026_10_16_NOON) == "2026-10-16T12:00:00+00:00"


def test_filetime_to_iso_rejects_junk() -> None:
    assert CU.filetime_to_iso(0) == ""  # before the Unix epoch: not a real timestamp
    assert CU.filetime_to_iso(-5) == ""
    assert CU.filetime_to_iso("not-a-number") == ""
    assert CU.filetime_to_iso(None) == ""


# ------------------------------------------------------------------ last_used


def test_last_used_reports_previous_apps_with_timestamps() -> None:
    got = CU.last_used(_entries())
    assert len(got) == 1
    row = got[0]
    assert row["name"] == "Zoom"
    assert row["path"] == r"C:\Apps\Zoom.exe"
    assert row["packaged"] is False
    assert row["at"] == "2026-10-16T12:00:00+00:00"


def test_last_used_most_recent_first_and_skips_away() -> None:
    rows = [
        ("C:#Apps#Old.exe", False, {"LastUsedTimeStart": 1, "LastUsedTimeStop": FT_2026_10_16_NOON - 10_000_000}),
        ("C:#Apps#New.exe", False, {"LastUsedTimeStart": 2, "LastUsedTimeStop": FT_2026_10_16_NOON}),
        (
            "C:#Program Files#Project Away#away.exe",
            False,
            {"LastUsedTimeStart": 3, "LastUsedTimeStop": FT_2026_10_16_NOON + 10_000_000},
        ),
    ]
    got = CU.last_used(rows)
    assert [r["name"] for r in got] == ["New", "Old"]  # Project Away excluded, newest first


# ------------------------------------------------------------------ process scan


def test_running_camera_apps_names_known_camera_programs_only() -> None:
    ps = _FakePsutil(
        [
            ("Zoom.exe", r"C:\Apps\Zoom.exe"),
            ("notepad.exe", r"C:\Windows\notepad.exe"),
            ("nvbroadcast.exe", r"C:\Program Files\NVIDIA Corporation\NVIDIA Broadcast\nvbroadcast.exe"),
        ]
    )
    got = CU.running_camera_apps(ps)
    assert got == [
        {"name": "NVIDIA Broadcast", "path": r"C:\Program Files\NVIDIA Corporation\NVIDIA Broadcast\nvbroadcast.exe"},
        {"name": "Zoom", "path": r"C:\Apps\Zoom.exe"},
    ]


def test_running_camera_apps_empty_without_psutil() -> None:
    class NoPsutil:
        def __getattr__(self, name: str) -> Any:
            raise ImportError("no psutil here")

    assert CU.running_camera_apps(NoPsutil()) == []


# ------------------------------------------------------------------ holders


def test_holders_combines_registry_and_process_scan() -> None:
    ps = _FakePsutil(
        [
            ("Zoom.exe", r"C:\Apps\Zoom.exe"),  # matches the last_used entry, but is not holding it now
            ("ms-teams.exe", r"C:\Apps\ms-teams.exe"),  # running now, no registry entry: a hint
            ("notepad.exe", r"C:\Windows\notepad.exe"),
        ]
    )
    rep = CU.holders(read=lambda: _entries(), psutil_mod=ps)
    now = rep["now"]
    assert {
        "name": "NVIDIA Broadcast",
        "path": r"C:\Program Files\NVIDIA Corporation\NVIDIA Broadcast\NVIDIA Broadcast.exe",
        "source": "registry",
    } in now
    assert {"name": "Microsoft Teams", "path": r"C:\Apps\ms-teams.exe", "source": "process"} in now
    # Zoom is running now (a hint: it may hold the camera) AND used one before (the registry timestamp)
    assert {"name": "Zoom", "path": r"C:\Apps\Zoom.exe", "source": "process"} in now
    assert not any("notepad" in u["name"].lower() for u in now)
    assert rep["last_used"] == [
        {"name": "Zoom", "path": r"C:\Apps\Zoom.exe", "packaged": False, "at": "2026-10-16T12:00:00+00:00"}
    ]


def test_holders_dedupes_registry_and_process_by_name() -> None:
    ps = _FakePsutil([("nvbroadcast.exe", r"C:\Program Files\NVIDIA Corporation\NVIDIA Broadcast\nvbroadcast.exe")])
    rep = CU.holders(read=lambda: _entries(), psutil_mod=ps)
    names = [u["name"] for u in rep["now"]]
    assert names.count("NVIDIA Broadcast") == 1
    assert [u for u in rep["now"] if u["name"] == "NVIDIA Broadcast"][0]["source"] == "registry"


def test_holders_empty_on_broken_registry() -> None:
    def boom() -> list[tuple[str, bool, dict[str, Any]]]:
        raise OSError("no registry here")

    assert CU.holders(read=boom, psutil_mod=_FakePsutil([])) == {"now": [], "last_used": []}


# ------------------------------------------------------------------ jobs wiring


def _runner(tmp_path: Path) -> jobs.JobRunner:
    return jobs.JobRunner(tmp_path / "diag", command=lambda a: ["away", *a], bench=lambda: {}, clock=_Clock())


def _wait_done(r: jobs.JobRunner, jid: str) -> jobs.Job:
    for _ in range(500):
        j = r.jobs[jid]
        if j.state != "running":
            return j
        time.sleep(0.01)
    raise AssertionError("holders job did not finish")


def test_holders_is_a_known_diagnose_kind(tmp_path: Path) -> None:
    r = _runner(tmp_path)
    job = r.start("holders")
    done = _wait_done(r, job.id)
    assert done.state == "done"
    res = done.public()["result"]
    assert set(res) == {"now", "last_used"}
    assert done.public()["output"]  # plain-words lines for the agent too
    with pytest.raises(ValueError, match="unknown check"):
        r.start("not-a-kind")


def test_holders_result_carries_registry_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = _runner(tmp_path)
    monkeypatch.setattr(
        CU, "holders", lambda: {"now": [{"name": "Zoom", "path": r"C:\Z", "source": "process"}], "last_used": []}
    )
    job = r.start("holders")
    done = _wait_done(r, job.id)
    res = done.public()["result"]
    assert res["now"] == [{"name": "Zoom", "path": r"C:\Z", "source": "process"}]
    assert done.public()["output"][0] == "holding a camera now: Zoom"


def test_holders_output_lines_in_plain_words() -> None:
    lines = jobs._holders_lines({"now": [], "last_used": [{"name": "Zoom", "at": "2026-10-16T12:00:00+00:00"}]})
    assert lines[0] == "no app is holding a camera now"
    assert lines[1] == "used a camera before: Zoom (last ended 2026-10-16 12:00)"


def test_now_timestamp_is_utc_iso() -> None:
    dt = datetime.fromisoformat(CU.filetime_to_iso(FT_2026_10_16_NOON))
    assert dt.tzinfo is not None and dt.utcoffset() == timezone.utc.utcoffset(None)
