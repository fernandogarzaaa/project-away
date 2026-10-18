"""Proposal #13 (v0.1.28): first-class "who is holding my camera" diagnose.

``diagnostics(action="run", kind="holders")`` answers with ``{"now": [...], "last_used": [{app, at}]}``:
registry holders (Windows says the app has a camera open now) + a live scan of known camera apps (a hint, not
proof) + the per-app last-used timestamps Windows stores. Every test is OS-free: the registry reader and the
process list are injected.
"""

from __future__ import annotations

import sys
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


EPOCH_FT = 116444736000000000  # 1970-01-01T00:00:00+00:00
UPPER_FT = 4102444800 * 10_000_000 + EPOCH_FT  # 2100-01-01T00:00:00+00:00


def test_filetime_to_iso_accepts_the_exact_bounds() -> None:
    assert CU.filetime_to_iso(EPOCH_FT) == "1970-01-01T00:00:00+00:00"
    assert CU.filetime_to_iso(UPPER_FT) == "2100-01-01T00:00:00+00:00"
    assert CU.filetime_to_iso(EPOCH_FT - 1) == ""  # one tick before the epoch
    assert CU.filetime_to_iso(UPPER_FT + 10_000_000) == ""  # one second past the upper bound
    assert CU.filetime_to_iso(UPPER_FT + 5_000_000) == ""  # half a second past it


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


def test_last_used_needs_both_a_start_and_a_stop() -> None:
    rows = [
        ("C:#Apps#NoStart.exe", False, {"LastUsedTimeStop": FT_2026_10_16_NOON}),  # no start: skipped
        ("C:#Apps#Early.exe", False, {"LastUsedTimeStart": 1, "LastUsedTimeStop": FT_2026_10_16_NOON}),  # kept
    ]
    assert [r["name"] for r in CU.last_used(rows)] == ["Early"]


def test_last_used_sorts_a_broken_timestamp_last() -> None:
    rows = [
        ("C:#Apps#Junk.exe", False, {"LastUsedTimeStart": 2, "LastUsedTimeStop": 1}),  # at == ""
        ("C:#Apps#New.exe", False, {"LastUsedTimeStart": 2, "LastUsedTimeStop": FT_2026_10_16_NOON}),
    ]
    got = CU.last_used(rows)
    assert [(r["name"], r["at"]) for r in got] == [("New", "2026-10-16T12:00:00+00:00"), ("Junk", "")]


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


def test_running_camera_apps_empty_when_psutil_cannot_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert CU.running_camera_apps() == []


def test_running_camera_apps_empty_when_the_process_list_fails() -> None:
    class BrokenPsutil:
        def process_iter(self, attrs: list[str]) -> Any:
            raise PermissionError("denied")

    assert CU.running_camera_apps(BrokenPsutil()) == []


def test_running_camera_apps_dedupes_by_program_name() -> None:
    ps = _FakePsutil(
        [
            ("nvbroadcast.exe", r"C:\a\nvbroadcast.exe"),
            ("NVBroadcast.EXE", r"C:\b\nvbroadcast.exe"),  # same program, different path/case
        ]
    )
    assert CU.running_camera_apps(ps) == [{"name": "NVIDIA Broadcast", "path": r"C:\a\nvbroadcast.exe"}]


def test_running_camera_apps_skips_processes_that_vanish() -> None:
    class Vanishing:
        @property
        def info(self) -> dict[str, Any]:
            raise RuntimeError("no such process")

    class FlakyPsutil:
        def process_iter(self, attrs: list[str]) -> list[Any]:
            return [Vanishing(), _Proc("zoom.exe", r"C:\Apps\Zoom.exe")]

    assert CU.running_camera_apps(FlakyPsutil()) == [{"name": "Zoom", "path": r"C:\Apps\Zoom.exe"}]


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
    assert set(res) == {"now", "last_used", "elapsed_s", "budget_hit"}
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
    # a process hint must never be worded as a confirmed holder
    assert done.public()["output"][0] == "may be holding a camera (running now, not confirmed): Zoom"


def test_holders_output_lines_in_plain_words() -> None:
    lines = jobs._holders_lines({"now": [], "last_used": [{"name": "Zoom", "at": "2026-10-16T12:00:00+00:00"}]})
    assert lines[0] == "no app is holding a camera now"
    assert lines[1] == "used a camera before: Zoom (last ended 2026-10-16 12:00)"


def test_holders_lines_distinguish_registry_from_process_hint() -> None:
    # Windows-confirmed holders are stated as fact; a running app is a hint only
    lines = jobs._holders_lines(
        {
            "now": [
                {"name": "OBS Studio", "path": r"C:\obs", "source": "registry"},
                {"name": "Zoom", "path": r"C:\Z", "source": "process"},
            ],
            "last_used": [],
        }
    )
    assert lines[0] == "holding a camera now (Windows reports): OBS Studio"
    assert lines[1] == "may be holding a camera (running now, not confirmed): Zoom"


def test_now_timestamp_is_utc_iso() -> None:
    dt = datetime.fromisoformat(CU.filetime_to_iso(FT_2026_10_16_NOON))
    assert dt.tzinfo is not None and dt.utcoffset() == timezone.utc.utcoffset(None)


def test_holders_survives_malformed_registry_data() -> None:
    # junk timestamps must not crash the holders job; they degrade to empty
    bad = [("WeirdApp", True, {"LastUsedTimeStart": "not-a-number", "LastUsedTimeStop": 0})]
    assert CU.holders(read=lambda: bad) == {"now": [], "last_used": []}
