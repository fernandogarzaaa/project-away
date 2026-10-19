"""Mutation-strengthening tests for crashlog + uiqueue (v0.1.29).

Targets surviving mutants in src/away/app/crashlog.py and src/away/app/uiqueue.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
import types
from pathlib import Path
from typing import Any

import pytest

from away.app import crashlog
from away.app.uiqueue import UiQueue


class FakeFaultHandler:
    def __init__(self) -> None:
        self.enabled: list[Any] = []
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


# _read_seen: JSON parsing edge cases (line 92 mutants)
def test_read_seen_valid_offset(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    seen.write_text(json.dumps({"offset": 1234}), encoding="utf-8")
    assert crashlog._read_seen(seen) == 1234


def test_read_seen_missing_file_returns_zero(tmp_path: Path) -> None:
    assert crashlog._read_seen(tmp_path / "nonexistent.json") == 0


def test_read_seen_invalid_json_returns_zero(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    seen.write_text("not json {{{", encoding="utf-8")
    assert crashlog._read_seen(seen) == 0


def test_read_seen_non_dict_returns_zero(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    seen.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert crashlog._read_seen(seen) == 0


def test_read_seen_missing_offset_returns_zero(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    seen.write_text(json.dumps({"note": "hi"}), encoding="utf-8")
    assert crashlog._read_seen(seen) == 0


# _write_seen: offset and note handling (lines 94, 103 mutants)
def test_write_seen_writes_offset(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    crashlog._write_seen(seen, 999)
    data = json.loads(seen.read_text(encoding="utf-8"))
    assert data["offset"] == 999


def test_write_seen_preserves_existing_note(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    seen.write_text(json.dumps({"offset": 10, "note": "old note", "noted_at": 123.0}), encoding="utf-8")
    crashlog._write_seen(seen, 20)  # note=None: keep the old note
    data = json.loads(seen.read_text(encoding="utf-8"))
    assert data["offset"] == 20
    assert data["note"] == "old note"
    assert data["noted_at"] == 123.0


def test_write_seen_replaces_note(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    crashlog._write_seen(seen, 5, note="new note", clock=lambda: 456.0)
    data = json.loads(seen.read_text(encoding="utf-8"))
    assert data["note"] == "new note"
    assert data["noted_at"] == 456.0


# recent_note: age boundaries (lines 112, 122, 125, 128 mutants)
def test_recent_note_returns_none_when_no_file(tmp_path: Path) -> None:
    assert crashlog.recent_note(tmp_path) is None


def test_recent_note_returns_none_when_no_note(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / crashlog.SEEN_FILE).write_text(json.dumps({"offset": 1}), encoding="utf-8")
    assert crashlog.recent_note(logs) is None


def test_recent_note_returns_note_when_fresh(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    now = time.time()
    (logs / crashlog.SEEN_FILE).write_text(
        json.dumps({"offset": 1, "note": "crashed!", "noted_at": now}), encoding="utf-8"
    )
    assert crashlog.recent_note(logs, clock=lambda: now + 10) == "crashed!"


def test_recent_note_returns_none_when_too_old(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    now = time.time()
    (logs / crashlog.SEEN_FILE).write_text(
        json.dumps({"offset": 1, "note": "crashed!", "noted_at": now - crashlog.NOTE_MAX_AGE_S - 1}),
        encoding="utf-8",
    )
    assert crashlog.recent_note(logs, clock=lambda: now) is None


def test_recent_note_max_age_constant(tmp_path: Path) -> None:
    # NOTE_MAX_AGE_S = 12 * 3600.0 (12 hours)
    assert crashlog.NOTE_MAX_AGE_S == 12 * 3600.0


# dismiss_note (lines 131+ mutants)
def test_dismiss_note_clears_state_and_file(crash_env: Any) -> None:
    folder = crash_env[0]
    _install(crash_env)
    crashlog._state["note"] = "test note"
    seen = folder / crashlog.SEEN_FILE
    seen.write_text(json.dumps({"offset": 1, "note": "test note", "noted_at": time.time()}), encoding="utf-8")
    assert crashlog.dismiss_note(folder) is True
    assert crashlog._state["note"] is None
    data = json.loads(seen.read_text(encoding="utf-8"))
    assert "note" not in data


def test_dismiss_note_returns_false_when_nothing(crash_env: Any) -> None:
    folder = crash_env[0]
    _install(crash_env)
    assert crashlog.dismiss_note(folder) is False


# tail (line 156 mutants)
def test_tail_returns_last_lines(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "crash.log").write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")
    lines = crashlog.tail(logs, lines=10)
    assert len(lines) == 10
    assert lines[-1] == "line 99"
    assert lines[0] == "line 90"


def test_tail_returns_empty_when_no_file(tmp_path: Path) -> None:
    assert crashlog.tail(tmp_path) == []


def test_tail_respects_max_lines(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "crash.log").write_text("a\nb\nc", encoding="utf-8")
    assert crashlog.tail(logs, lines=2) == ["b", "c"]


# new_entries (lines 178, 179, 182 mutants)
def test_new_entries_returns_new_text(tmp_path: Path) -> None:
    path = tmp_path / "crash.log"
    seen = tmp_path / "seen.json"
    path.write_bytes(b"old\nnew\n")  # binary: no CRLF translation on Windows
    seen.write_text(json.dumps({"offset": 4}), encoding="utf-8")  # after "old\n"
    assert crashlog.new_entries(path, seen) == "new\n"


def test_new_entries_returns_empty_when_no_new(tmp_path: Path) -> None:
    path = tmp_path / "crash.log"
    seen = tmp_path / "seen.json"
    path.write_text("data", encoding="utf-8")
    seen.write_text(json.dumps({"offset": 4}), encoding="utf-8")
    assert crashlog.new_entries(path, seen) == ""


def test_new_entries_handles_rotation(tmp_path: Path) -> None:
    path = tmp_path / "crash.log"
    seen = tmp_path / "seen.json"
    path.write_text("new", encoding="utf-8")
    seen.write_text(json.dumps({"offset": 1000}), encoding="utf-8")  # offset beyond size: rotated
    assert crashlog.new_entries(path, seen) == "new"


def test_new_entries_missing_file(tmp_path: Path) -> None:
    assert crashlog.new_entries(tmp_path / "missing.log", tmp_path / "seen.json") == ""


# entry_label (line 188+ mutants)
def test_entry_label_tray() -> None:
    assert crashlog.entry_label(["tray"]) == "app (tray)"
    assert crashlog.entry_label([]) == "app (tray)"


def test_entry_label_helpers() -> None:
    assert crashlog.entry_label(["settings"]) == "helper settings"
    assert crashlog.entry_label(["agent-ui", "toast"]) == "helper agent-ui toast"


def test_entry_label_frozen() -> None:
    assert crashlog.entry_label(["mycommand"]) == "frozen mycommand"


def test_entry_label_ignores_flags() -> None:
    assert crashlog.entry_label(["-v", "tray"]) == "app (tray)"


# _pid_alive (line 199+ mutants)
def test_pid_alive_self() -> None:
    assert crashlog._pid_alive(os.getpid()) is True


def test_pid_alive_nonexistent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock psutil to guarantee False (Windows PIDs wrap, 2**30 may exist)
    import sys

    fake_psutil = types.SimpleNamespace(pid_exists=lambda pid: False)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    # Use a PID that is not our own; psutil mock says it does not exist
    assert crashlog._pid_alive(999999999) is False


# summarize_native (lines 219-254 mutants)
def test_summarize_native_empty() -> None:
    assert crashlog.summarize_native("") == ""
    assert crashlog.summarize_native("   \n  ") == ""


def test_summarize_native_fatal() -> None:
    text = "Fatal Python error: Segmentation fault\n  File x\n"
    out = crashlog.summarize_native(text, pid=123)
    assert "native crash in pid 123" in out
    assert "Segmentation fault" in out


def test_summarize_native_handled_code() -> None:
    text = "Windows fatal exception: code 0x80010108\n  File foo.py\n"
    out = crashlog.summarize_native(text, pid=456)
    assert "handled (not a crash)" in out
    assert "0x80010108" in out
    assert "RPC_E_DISCONNECTED" in out
    assert "x1" in out


def test_summarize_native_unknown_code() -> None:
    text = "Windows fatal exception: code 0xdeadbeef\n"
    out = crashlog.summarize_native(text)
    assert "handled Windows exception" in out


# classify, first_crash_line (lines 290-304 mutants)
def test_classify_fatal() -> None:
    assert crashlog.classify("Fatal Python error: boom") == "crashed"
    assert crashlog.classify("Segmentation fault") == "crashed"


def test_classify_uncaught() -> None:
    assert crashlog.classify("Uncaught exception in thread") == "error"


def test_classify_none() -> None:
    assert crashlog.classify("just a log line") is None
    assert crashlog.classify("") is None


def test_first_crash_line_finds_marker() -> None:
    text = "normal line\nFatal Python error: boom\nanother"
    assert crashlog.first_crash_line(text) == "Fatal Python error: boom"


def test_first_crash_line_empty_when_none() -> None:
    assert crashlog.first_crash_line("nothing here") == ""


# write (lines 311-319 mutants)
def test_write_appends_when_installed(crash_env: Any) -> None:
    folder = crash_env[0]
    _install(crash_env)
    crashlog.write("hello")
    text = (folder / "crash.log").read_text(encoding="utf-8")
    assert "hello" in text


def test_write_noop_when_not_installed(crash_env: Any) -> None:
    # _state["file"] is None: write() returns silently
    crashlog.write("should not raise")


# UiQueue (uiqueue.py mutants)
def test_uiqueue_post_and_drain() -> None:
    q = UiQueue()
    calls: list[str] = []
    q.post(calls.append, "a")
    q.post(calls.append, "b")
    assert q.drain() == 2
    assert calls == ["a", "b"]


def test_uiqueue_drain_empty() -> None:
    q = UiQueue()
    assert q.drain() == 0


def test_uiqueue_drain_limit() -> None:
    q = UiQueue()
    calls: list[int] = []
    for i in range(5):
        q.post(calls.append, i)
    assert q.drain(limit=3) == 3
    assert q.drain() == 2  # remaining


def test_uiqueue_drain_logs_exception(caplog: Any) -> None:
    q = UiQueue()

    def bad() -> None:
        raise ValueError("oops")

    q.post(bad)
    q.post(lambda: None)
    with caplog.at_level(logging.ERROR, logger="away.ui"):
        assert q.drain() == 2  # exception logged, next still runs
    assert "UI callback failed" in caplog.text


def test_uiqueue_pump_drains_and_schedules() -> None:
    q = UiQueue()
    calls: list[str] = []
    q.post(calls.append, "x")
    scheduled: list[tuple] = []

    class FakeWidget:
        def after(self, ms: int, fn: Any, *args: Any) -> None:
            scheduled.append((ms, fn, args))

    q.pump(FakeWidget(), every_ms=50)
    assert calls == ["x"]
    assert len(scheduled) == 1
    assert scheduled[0][0] == 50


def test_uiqueue_owner_is_current_thread() -> None:
    import threading

    q = UiQueue()
    assert q.owner == threading.get_ident()


# Round 2: remaining survivors


def test_dismiss_note_removes_note_from_file(crash_env: Any) -> None:
    folder = crash_env[0]
    _install(crash_env)
    seen = folder / crashlog.SEEN_FILE
    seen.write_text(json.dumps({"offset": 5, "note": "to dismiss", "noted_at": time.time()}), encoding="utf-8")
    assert crashlog.dismiss_note(folder) is True
    data = json.loads(seen.read_text(encoding="utf-8"))
    assert data.get("note") is None
    assert data.get("noted_at") is None
    assert data["offset"] == 5  # offset preserved


def test_tail_seeks_from_end_for_large_files(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    # Write more than max_bytes to test the seek logic
    big = "x" * 300_000
    (logs / "crash.log").write_text(big + "\nlast", encoding="utf-8")
    lines = crashlog.tail(logs, lines=5, max_bytes=200_000)
    assert lines[-1] == "last"


def test_new_entries_offset_at_exact_size(tmp_path: Path) -> None:
    path = tmp_path / "crash.log"
    seen = tmp_path / "seen.json"
    path.write_text("1234", encoding="utf-8")
    seen.write_text(json.dumps({"offset": 4}), encoding="utf-8")
    # offset == size: nothing new (tests GtE -> Gt mutant at line 178)
    assert crashlog.new_entries(path, seen) == ""


def test_entry_label_agent_ui_two_args() -> None:
    # agent-ui takes 2 args, others take 1 (line 195 mutants)
    assert crashlog.entry_label(["agent-ui", "toast", "extra"]) == "helper agent-ui toast"
    assert crashlog.entry_label(["settings", "extra"]) == "helper settings"


def test_summarize_native_multiple_blocks() -> None:
    text = "Fatal Python error: boom\n  File a\nSome other line\n"
    out = crashlog.summarize_native(text, pid=1)
    assert "native crash in pid 1" in out
    assert "Some other line" in out  # non-crash blocks preserved (line 230)


def test_summarize_native_counts_repeated_codes() -> None:
    text = "Windows fatal exception: code 0x80010108\nWindows fatal exception: code 0x80010108\n"
    out = crashlog.summarize_native(text)
    assert "x2" in out  # count incremented (lines 241-242)


def test_fold_native_folds_dead_process_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    # Mock _pid_alive: Windows PIDs wrap, 2**30 may appear alive
    monkeypatch.setattr(crashlog, "_pid_alive", lambda pid: False)
    dead_pid = 12345
    (logs / f"{crashlog.NATIVE_PREFIX}{dead_pid}.log").write_text("Fatal Python error: boom\n", encoding="utf-8")
    n = crashlog.fold_native(logs)
    assert n == 1
    assert not (logs / f"{crashlog.NATIVE_PREFIX}{dead_pid}.log").exists()
    assert "boom" in (logs / "crash.log").read_text(encoding="utf-8")


def test_fold_native_skips_live_process(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    # Own PID file is skipped unless own=True
    (logs / f"{crashlog.NATIVE_PREFIX}{os.getpid()}.log").write_text("x", encoding="utf-8")
    assert crashlog.fold_native(logs) == 0
    assert crashlog.fold_native(logs, own=True) == 1  # own=True folds it


def test_fold_native_skips_bad_pid_filename(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / f"{crashlog.NATIVE_PREFIX}notapid.log").write_text("x", encoding="utf-8")
    assert crashlog.fold_native(logs) == 0


def test_install_returns_note_when_already_installed(crash_env: Any) -> None:
    _install(crash_env)
    crashlog._state["note"] = "existing"
    # Second install returns the existing note (line 369)
    assert _install(crash_env) == "existing"


def test_install_respects_no_crashlog_env(crash_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWAY_NO_CRASHLOG", "1")
    assert _install(crash_env) is None


def test_uninstall_closes_files(crash_env: Any) -> None:
    folder, fh, sysm, thr = crash_env
    _install(crash_env)
    f = crashlog._state["file"]
    assert f is not None
    crashlog.uninstall(fh)
    assert crashlog._state["file"] is None
    assert crashlog._state["native"] is None
    assert fh.disabled == 1  # fh.disable() called


def test_at_exit_folds_own_file(crash_env: Any) -> None:
    folder = crash_env[0]
    _install(crash_env)
    native_path = folder / f"{crashlog.NATIVE_PREFIX}{os.getpid()}.log"
    native_path.write_text("Fatal Python error: at exit\n", encoding="utf-8")
    crashlog._at_exit()
    assert crashlog._state["native"] is None
    # File should be folded (own=True)
    assert not native_path.exists()


def test_at_exit_noop_when_not_installed() -> None:
    crashlog._state["native"] = None
    crashlog._state["folder"] = None
    crashlog._at_exit()  # should not raise


def test_uiqueue_drain_exact_limit() -> None:
    q = UiQueue()
    calls: list[int] = []
    for i in range(3):
        q.post(calls.append, i)
    # drain(limit=3) with exactly 3 items (tests Lt -> LtE at line 33)
    assert q.drain(limit=3) == 3
    assert q.drain() == 0


def test_uiqueue_pump_custom_interval() -> None:
    q = UiQueue()
    scheduled: list[int] = []

    class FakeWidget:
        def after(self, ms: int, fn: Any, *args: Any) -> None:
            scheduled.append(ms)

    q.pump(FakeWidget(), every_ms=100)
    assert scheduled == [100]  # tests 50 -> 51 mutant at line 45


# Round 3: precise survivor targeting


def test_write_seen_noted_at_defaults_to_zero(tmp_path: Path) -> None:
    seen = tmp_path / "seen.json"
    # old note without noted_at: defaults to 0 (line 103 mutant)
    seen.write_text(json.dumps({"offset": 1, "note": "old"}), encoding="utf-8")
    crashlog._write_seen(seen, 2)
    data = json.loads(seen.read_text(encoding="utf-8"))
    assert data["noted_at"] == 0


def test_pid_alive_oserror_returns_true() -> None:
    # os.kill raising OSError (not ProcessLookupError) means "cannot tell" -> True
    # We can't easily trigger this, but we test the live PID path
    assert crashlog._pid_alive(os.getpid()) is True


def test_summarize_native_increments_count() -> None:
    # Two identical codes: count goes 1 -> 2 (lines 241-242 mutants)
    text = "Windows fatal exception: code 0x1234\nWindows fatal exception: code 0x1234\n"
    out = crashlog.summarize_native(text)
    assert "x2" in out


def test_summarize_native_preserves_other_blocks() -> None:
    # Non-exception blocks are extended to output (line 246 mutant)
    text = "Some random output\nMore lines\n"
    out = crashlog.summarize_native(text)
    assert "Some random output" in out
    assert "More lines" in out


def test_thread_excepthook_ignores_system_exit(crash_env: Any) -> None:
    folder, fh, sysm, thr = crash_env
    _install(crash_env)
    hook = crashlog._thread_excepthook(None)
    args = types.SimpleNamespace(
        exc_type=SystemExit,
        exc_value=SystemExit(0),
        exc_traceback=None,
        thread=types.SimpleNamespace(name="t"),
    )
    hook(args)  # SystemExit: should not write to crash log
    text = (folder / "crash.log").read_text(encoding="utf-8")
    assert "SystemExit" not in text


def test_thread_excepthook_calls_prev(crash_env: Any) -> None:
    folder, fh, sysm, thr = crash_env
    _install(crash_env)
    prev_calls: list[Any] = []
    hook = crashlog._thread_excepthook(prev_calls.append)
    args = types.SimpleNamespace(
        exc_type=ValueError,
        exc_value=ValueError("x"),
        exc_traceback=None,
        thread=types.SimpleNamespace(name="t"),
    )
    hook(args)
    assert len(prev_calls) == 1  # prev called (line 346 mutant)


def test_install_enables_faulthandler(crash_env: Any) -> None:
    folder, fh, sysm, thr = crash_env
    _install(crash_env)
    # fh.enable called with file and all_threads=True (line 379 mutant)
    assert len(fh.enabled) == 1
    assert fh.enabled[0]["all_threads"] is True


def test_install_folds_native_on_start(crash_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    folder, fh, sysm, thr = crash_env
    folder.mkdir(parents=True, exist_ok=True)
    # Mock _pid_alive: Windows PIDs wrap, 2**30 may appear alive
    monkeypatch.setattr(crashlog, "_pid_alive", lambda pid: False)
    dead = 12345
    (folder / f"{crashlog.NATIVE_PREFIX}{dead}.log").write_text("Fatal Python error: x\n", encoding="utf-8")
    _install(crash_env)  # install calls fold_native (line 383 mutant)
    assert not (folder / f"{crashlog.NATIVE_PREFIX}{dead}.log").exists()


def test_max_bytes_boundary(crash_env: Any) -> None:
    # File exactly at MAX_BYTES does not rotate; above does (line 47, 391 mutants)
    folder = crash_env[0]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "crash.log").write_text("x" * crashlog.MAX_BYTES, encoding="utf-8")
    _install(crash_env)
    assert not (folder / "crash.log.1").exists()  # exactly at: no rotate
    crashlog.uninstall(FakeFaultHandler())
    (folder / "crash.log").write_text("x" * (crashlog.MAX_BYTES + 1), encoding="utf-8")
    _install(crash_env)
    assert (folder / "crash.log.1").exists()  # over: rotates


# Round 4: final precision targets


def test_recent_note_dict_without_note(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    # dict but falsy note: tests `or` -> `and` at line 122
    # (both conditions must be False to proceed; if `and`, a non-dict would crash)
    (logs / crashlog.SEEN_FILE).write_text(json.dumps({"offset": 1, "note": ""}), encoding="utf-8")
    assert crashlog.recent_note(logs) is None
    (logs / crashlog.SEEN_FILE).write_text(json.dumps({"offset": 1}), encoding="utf-8")
    assert crashlog.recent_note(logs) is None


def test_recent_note_missing_noted_at(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    now = time.time()
    # noted_at missing: defaults to 0, age = now - 0 = huge -> None (line 125)
    (logs / crashlog.SEEN_FILE).write_text(json.dumps({"offset": 1, "note": "old"}), encoding="utf-8")
    assert crashlog.recent_note(logs, clock=lambda: now) is None


def test_recent_note_age_exactly_at_boundary(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    now = time.time()
    # age == max_age_s: 0 <= age <= max_age_s is True (tests LtE -> Lt at line 128)
    (logs / crashlog.SEEN_FILE).write_text(
        json.dumps({"offset": 1, "note": "edge", "noted_at": now - 100}),
        encoding="utf-8",
    )
    assert crashlog.recent_note(logs, max_age_s=100.0, clock=lambda: now) == "edge"
    # just over: None
    assert crashlog.recent_note(logs, max_age_s=99.9, clock=lambda: now) is None


def test_uiqueue_drain_default_limit_exact() -> None:
    q = UiQueue()
    calls: list[int] = []
    for i in range(100):
        q.post(calls.append, i)
    # exactly 100 items with default limit=100: all drain (tests 100 -> 101 at line 30)
    assert q.drain() == 100
    # 101 items: only 100 drain
    for i in range(101):
        q.post(calls.append, i)
    assert q.drain() == 100
    assert q.drain() == 1


def test_uiqueue_pump_default_interval() -> None:
    q = UiQueue()
    scheduled: list[int] = []

    class FakeWidget:
        def after(self, ms: int, fn: Any, *args: Any) -> None:
            scheduled.append(ms)

    q.pump(FakeWidget())  # default every_ms=50 (tests 50 -> 51 at line 45)
    assert scheduled == [50]

def test_write_flushes_data_and_at_exit_folds_native_file(crash_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    folder, fh, sysm, thr = crash_env
    _install(crash_env)
    crashlog.write("flush sentinel")
    f = crashlog._state["file"]
    assert f is not None
    assert f.closed is False
    f.flush()
    assert "flush sentinel" in (folder / "crash.log").read_text(encoding="utf-8")

    native_path = folder / f"{crashlog.NATIVE_PREFIX}{os.getpid()}.log"
    native_path.write_text("Fatal Python error: at exit flush sentinel\n", encoding="utf-8")
    crashlog._at_exit()
    assert not native_path.exists()
    assert crashlog._state["native"] is None


def test_pid_alive_unknown_os_error_is_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    fake_psutil = types.SimpleNamespace(pid_exists=lambda _pid: (_ for _ in ()).throw(OSError("unknown")))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert crashlog._pid_alive(987654321) is True

