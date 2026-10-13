"""v0.1.23 fixes that had no tests yet: tray COM guard + crash.log tagging (b), attention hysteresis (c), the big
cursor with two app copies, the single-instance lock and ``away cursor reset`` (f), the 120 Hz glide (g), busiest
programs + live camera removal (i), which app holds the camera (k), WinRT format fallback and no NVIDIA Broadcast
fallback for the built-in camera (l)."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app import bigcursor as BC
from away.app import crashlog, instance, sysload
from away.app.tray import TrayGuard, camera_menu_text, com_apartment
from away.comfort.attention import MIN_DWELL_S, AttentionGate
from away.comfort.settings import AttentionConfig
from away.hands.synthetic import pose
from away.multicam import camera_users as CU
from away.multicam import capture
from away.pointer.glide import Glide, GlideRunner


# ============================================================================ (b) tray guard, COM, crash.log
def test_tray_guard_serializes_swallows_errors_and_stops_once() -> None:
    g = TrayGuard()
    assert g.call(lambda a, b=0: a + b, 2, b=3) == 5

    def boom() -> None:
        raise OSError("RPC_E_DISCONNECTED")

    assert g.call(boom) is None and g.errors == 1  # logged, never raised into the camera/voice thread

    class Icon:
        visible = True
        stops = 0

        def stop(self) -> None:
            Icon.stops += 1
            raise RuntimeError("COM object already gone")  # stopping must not raise either

    icon = Icon()
    g.stop(icon)
    g.stop(icon)
    assert g.stopping and icon.visible is False and Icon.stops == 1
    called: list[int] = []
    assert g.call(called.append, 1) is None and called == []  # nothing touches the icon after stop


def test_tray_guard_calls_never_overlap() -> None:
    g = TrayGuard()
    inside, overlap = [0], [False]

    def work() -> None:
        inside[0] += 1
        if inside[0] > 1:
            overlap[0] = True
        threading.Event().wait(0.002)
        inside[0] -= 1

    ts = [threading.Thread(target=lambda: [g.call(work) for _ in range(20)]) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not overlap[0]


def test_com_apartment_results() -> None:
    class Ole:
        def __init__(self, hr: int) -> None:
            self.hr, self.args = hr, None

        def CoInitializeEx(self, *a: Any) -> int:
            self.args = a
            return self.hr

    assert com_apartment("linux") == "" and com_apartment("darwin") == ""
    ole = Ole(0)
    assert com_apartment("win32", ole) == "sta" and ole.args == (None, 0x2)  # COINIT_APARTMENTTHREADED
    assert com_apartment("win32", Ole(1)) == "sta"  # S_FALSE: already an STA
    assert com_apartment("win32", Ole(-2147417850)) == "already"  # RPC_E_CHANGED_MODE

    class Broken:
        def CoInitializeEx(self, *a: Any) -> int:
            raise OSError("no ole32")

    assert com_apartment("win32", Broken()) == ""


def test_camera_menu_text() -> None:
    assert camera_menu_text(types.SimpleNamespace()) == ""
    app = types.SimpleNamespace(camera_unavailable={"headline": "Internal camera unavailable", "devices": ["x"]})
    assert camera_menu_text(app) == "Internal camera unavailable"


@pytest.mark.parametrize(
    ("argv", "label"),
    [
        ([], "app (tray)"),
        (["tray"], "app (tray)"),
        (["agent-ui", "toast", "--id", "3"], "helper agent-ui toast"),
        (["settings"], "helper settings"),
        (["--debug", "mcp"], "helper mcp"),
        (["doctor"], "frozen doctor"),
    ],
)
def test_entry_labels_tell_helpers_from_restarts(argv: list[str], label: str) -> None:
    assert crashlog.entry_label(argv) == label


FAULT_TEXT = """Windows fatal exception: code 0x80010108

Thread 0x00001234 (most recent call first):
  File "pystray/_win32.py", line 210 in _hide
  File "away/app/tray.py", line 400 in on_quit
Windows fatal exception: code 0x80010108

Thread 0x00001234 (most recent call first):
  File "pystray/_win32.py", line 220 in _message
Windows fatal exception: code 0xe06d7363

Thread 0x00005678 (most recent call first):
  File "onnxruntime/capi/session.py", line 1 in run
"""


def test_handled_windows_exceptions_are_tagged_not_crashes() -> None:
    out = crashlog.summarize_native(FAULT_TEXT, 4242)
    assert "handled (not a crash): Windows exception 0x80010108 (RPC_E_DISCONNECTED) x2 in pid 4242" in out
    assert 'first at File "pystray/_win32.py", line 210 in _hide' in out
    assert "0xe06d7363 (C++ exception) x1" in out
    assert crashlog.classify(out) is None  # nothing here makes the next start say "the app crashed"
    assert crashlog.summarize_native("   \n") == ""


def test_a_real_native_crash_is_kept_in_full() -> None:
    text = 'Windows fatal exception: access violation\n\nCurrent thread 0x1 (most recent call first):\n  File "x.py"\n'
    out = crashlog.summarize_native(text, 7)
    assert out.startswith("native crash in pid 7:") and "access violation" in out and 'File "x.py"' in out
    assert crashlog.classify(out) == "crashed"


def test_fold_native_moves_finished_processes_into_crash_log(tmp_path: Path, monkeypatch: Any) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    dead, alive = 999_991, 999_992
    (logs / f"crash-native-{dead}.log").write_text(FAULT_TEXT, encoding="utf-8")
    (logs / f"crash-native-{alive}.log").write_text(FAULT_TEXT, encoding="utf-8")
    (logs / f"crash-native-{os.getpid()}.log").write_text("", encoding="utf-8")
    (logs / "crash-native-notapid.log").write_text("x", encoding="utf-8")
    monkeypatch.setattr(crashlog, "_pid_alive", lambda pid: pid == alive)
    assert crashlog.fold_native(logs) == 1
    text = (logs / "crash.log").read_text(encoding="utf-8")
    assert f"faulthandler output of pid {dead}:" in text and "handled (not a crash)" in text
    assert not (logs / f"crash-native-{dead}.log").exists()
    assert (logs / f"crash-native-{alive}.log").exists()  # still running: left alone
    assert (logs / f"crash-native-{os.getpid()}.log").exists()  # our own file only at exit
    assert crashlog.fold_native(logs, own=True) == 1  # empty own file: removed, nothing written
    assert not (logs / f"crash-native-{os.getpid()}.log").exists()


# ============================================================================ (c) attention hysteresis
def _gate(**kw: Any) -> AttentionGate:
    return AttentionGate(AttentionConfig(enabled=True, **kw))


def _hand(y: float) -> np.ndarray:
    return pose("point", center=(0.5, y), scale=0.12)


def test_arming_does_not_flicker_for_a_hand_hovering_at_the_line() -> None:
    g = _gate(onset_s=1.0, min_hand_height=0.45)
    g.update(0.0, _hand(0.5))  # height 0.5: clearly up
    assert g.state == "arming"
    states = []
    t = 0.0
    for i in range(30):  # hovering around the line: 0.43 / 0.47, below for < MIN_DWELL_S each time
        t += 1 / 30
        g.update(t, _hand(0.57 if i % 6 < 3 else 0.53))
        states.append(g.state)
    assert "idle" not in states  # 0.43 is within the lower (keep) threshold while arming


def test_arming_falls_back_to_idle_only_after_the_dwell() -> None:
    g = _gate(onset_s=1.0, min_hand_height=0.45)
    g.update(0.0, _hand(0.5))
    g.update(0.1, _hand(0.95))  # hand dropped well below
    assert g.state == "arming"
    g.update(0.1 + MIN_DWELL_S / 2, _hand(0.95))
    assert g.state == "arming"
    g.update(0.1 + MIN_DWELL_S + 0.01, _hand(0.95))
    assert g.state == "idle"


def test_a_short_dip_while_arming_keeps_the_onset_clock() -> None:
    g = _gate(onset_s=0.5, min_hand_height=0.45)
    g.update(0.0, _hand(0.5))
    g.update(0.2, _hand(0.95))  # a quick dip, under MIN_DWELL_S
    g.update(0.3, _hand(0.5))
    g.update(0.55, _hand(0.5))
    assert g.state == "engaged"  # onset counted from 0.0, not restarted by the dip


def test_idle_still_needs_the_full_threshold_to_start_arming() -> None:
    g = _gate(onset_s=1.0, min_hand_height=0.45)
    g.update(0.0, _hand(0.57))  # height 0.43: below the line, never armed
    assert g.state == "idle"


# ============================================================================ (f) big cursor + single instance
class FakeCursor:
    def __init__(self, size: int = 32, sticky: bool = False) -> None:
        self.size = size
        self.sets: list[int] = []
        self.sticky = sticky  # ignores restores (a stuck registry value)

    def get(self) -> int | None:
        return self.size

    def set(self, size: int) -> bool:
        self.sets.append(size)
        if not self.sticky or size > self.size:
            self.size = size
        return True


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_two_app_copies_sharing_a_marker_never_leave_the_cursor_big(tmp_path: Path) -> None:
    """The crash.log case: two app processes, one marker. The second copy must not take the enlarged size as the
    user's original, and both restore the user's real size."""
    marker = tmp_path / BC.MARKER_NAME
    sys_cursor = FakeCursor(40)  # the user's own size: 40 px
    clock = Clock()
    a = BC.BigCursor(sys_cursor, 2, marker, clock=clock)
    b = BC.BigCursor(sys_cursor, 3, marker, clock=clock)
    assert a.update(True) and sys_cursor.size == 64
    assert json.loads(marker.read_text())["original"] == 40
    clock.t += 5
    assert b.update(True) and sys_cursor.size == 96
    assert b.original == 40  # from the marker, not the 64 px it found
    assert json.loads(marker.read_text())["original"] == 40
    b.restore()
    assert sys_cursor.size == 40
    a.restore()
    assert sys_cursor.size == 40 and not marker.exists()


def test_a_leftover_big_size_is_never_adopted_as_the_original(tmp_path: Path) -> None:
    marker = tmp_path / BC.MARKER_NAME
    first = BC.BigCursor(FakeCursor(40), 2, marker, clock=Clock())
    first.update(True)  # remembers 40 as the user's own size
    first.restore()
    stuck = FakeCursor(64)  # a big cursor left behind with no marker (killed before v0.1.23)
    big = BC.BigCursor(stuck, 2, marker, clock=Clock())
    big.update(True)
    assert big.original == 40  # the last size known to be the user's
    big.restore()
    assert stuck.size == 40


def test_recover_replaces_a_marker_that_holds_one_of_our_sizes(tmp_path: Path) -> None:
    marker = tmp_path / BC.MARKER_NAME
    marker.write_text(json.dumps({"original": 96, "applied": 128, "pid": 1}))
    cur = FakeCursor(128)
    assert BC.BigCursor(cur, 2, marker).recover()
    assert cur.sets == [32] and not marker.exists()  # 96 is ours: back to 32 (no user size known)


def test_restore_is_immediate_but_re_enlarging_waits(tmp_path: Path) -> None:
    clock = Clock()
    cur = FakeCursor(32)
    big = BC.BigCursor(cur, 2, tmp_path / BC.MARKER_NAME, clock=clock)
    assert big.update(True)
    clock.t += 0.05
    assert big.update(False) and cur.size == 32  # pause: shrinks at once
    clock.t += 0.5
    assert not big.update(True) and cur.size == 32  # flapping auto-pause: no re-enlarge yet
    clock.t += BC.ENLARGE_DEBOUNCE_S
    assert big.update(True) and cur.size == 64


def test_a_restore_that_does_not_take_is_reported_and_reset_fixes_it(tmp_path: Path) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []
    cur = FakeCursor(32, sticky=True)
    big = BC.BigCursor(cur, 2, tmp_path / BC.MARKER_NAME, clock=Clock(), publish=lambda t, d: seen.append((t, d)))
    big.update(True)
    big.restore()
    stuck = [d for t, d in seen if t == "cursor.size_stuck"]
    assert big.stuck and stuck and "Reset cursor size" in stuck[0]["message"]
    assert seen[-1] == ("cursor.size", {"size": 32, "big": False, "applied": False})
    assert cur.sets.count(32) == 2  # tried once more before reporting
    cur.sticky = False
    assert big.reset() == 32 and cur.size == 32 and not big.stuck
    assert seen[-1] == ("cursor.size_reset", {"size": 32})
    assert not (tmp_path / BC.MARKER_NAME).exists()


def test_reset_uses_the_marked_original(tmp_path: Path) -> None:
    marker = tmp_path / BC.MARKER_NAME
    marker.write_text(json.dumps({"original": 48 + 2, "applied": 96, "pid": 1}))
    cur = FakeCursor(96)
    assert BC.BigCursor(cur, 0, marker).reset() == 50 and cur.size == 50


def test_cli_cursor_reset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any) -> None:
    from away import cli

    monkeypatch.setenv("AWAY_DATA_DIR", str(tmp_path))
    cur = FakeCursor(96)
    monkeypatch.setattr(BC, "default_backend", lambda platform=None: cur)
    assert cli.main(["cursor", "reset"]) == 0
    out = capsys.readouterr().out
    assert cur.size == 32
    assert ("reset to 32 px" in out) if sys.platform.startswith("win") else ("Nothing to reset" in out)


def test_single_instance_lock_with_a_fake_windows_mutex() -> None:
    class K32:
        def __init__(self) -> None:
            self.names: list[str] = []
            self.closed: list[int] = []

        def CreateMutexW(self, _sec: Any, _own: bool, name: str) -> int:
            self.names.append(name)
            return len(self.names)

        def GetLastError(self) -> int:
            return 183 if len(self.names) > 1 else 0  # ERROR_ALREADY_EXISTS for the second app

        def CloseHandle(self, h: int) -> None:
            self.closed.append(h)

    k = K32()
    try:
        assert instance.acquire("t1", platform="win32", kernel32=k)
        assert instance.owned("t1")
        instance._held.pop("t1")  # pretend this is another process
        assert not instance.acquire("t1", platform="win32", kernel32=k)
        assert k.names == ["Local\\ProjectAway.App.t1"] * 2 and k.closed == [2]
    finally:
        instance._held.pop("t1", None)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="flock: macOS / Linux")
def test_single_instance_lock_across_processes(tmp_path: Path) -> None:
    try:
        assert instance.acquire("t2", folder=tmp_path, platform="linux")
        assert instance.acquire("t2", folder=tmp_path, platform="linux")  # same process: already ours
        code = (
            "import sys; from pathlib import Path; from away.app import instance; "
            f"sys.exit(0 if instance.acquire('t2', folder=Path({str(tmp_path)!r}), platform='linux') else 3)"
        )
        assert subprocess.run([sys.executable, "-c", code], timeout=60).returncode == 3  # the second app
        instance.release("t2")
        assert subprocess.run([sys.executable, "-c", code], timeout=60).returncode == 0  # free again
    finally:
        instance.release("t2")


# ============================================================================ (g) 120 Hz glide
def test_glide_moves_smoothly_towards_each_new_frame() -> None:
    g = Glide()
    assert g.step(0.0) is None  # no target yet
    g.on_target(0.0, 100.0, 100.0)
    g.on_target(0.1, 110.0, 100.0)  # a slow camera: 10 fps, moving right at 100 px/s
    xs = []
    t = 0.1
    while t < 0.2:
        t += 1 / 120
        p = g.step(t)
        if p is not None:
            xs.append(p[0])
    assert xs == sorted(xs)  # never steps back
    assert max(b - a for a, b in itertools.pairwise(xs)) <= 6  # no jump
    assert 108 <= xs[-1] <= 110 + 100 * g.max_ahead_s  # closes in, at most one frame of prediction ahead


def test_glide_holds_still_when_the_frames_stop() -> None:
    g = Glide()
    g.on_target(0.0, 0.0, 0.0)
    g.on_target(1 / 30, 10.0, 0.0)  # 300 px/s
    late = 1 / 30 + g.stall_frames * g.frame_dt + 0.05
    assert g.predicted(late) == (10.0, 0.0)  # stalled: no extrapolation
    soon = g.predicted(1 / 30 + 0.01)
    assert soon is not None and soon[0] > 10.0  # between frames it keeps moving
    for i in range(200):
        g.step(late + i / 120)
    assert g.step(late + 3.0) is None  # settled on the target: nothing more to send


def test_glide_jump_and_stop() -> None:
    g = Glide()
    g.on_target(0.0, 0.0, 0.0)
    g.jump(500.0, 300.0, 0.01)
    assert g.out == (500.0, 300.0) and g.sent == (500, 300) and g.vel == (0.0, 0.0)
    assert g.step(0.02) is None  # already there
    g.stop()
    assert g.target is None and g.step(0.03) is None


def test_glide_runner_tick_only_moves_while_active() -> None:
    g = Glide()
    moves: list[tuple[int, int]] = []
    now = [0.0]
    on = [False]
    r = GlideRunner(g, lambda x, y: moves.append((x, y)), hz=120, clock=lambda: now[0], active=lambda: on[0])
    g.on_target(0.0, 0.0, 0.0)
    g.on_target(0.05, 50.0, 0.0)
    now[0] = 0.06
    assert not r.tick() and moves == []  # paused
    on[0] = True
    assert r.tick() and moves and moves[-1][0] > 0

    def broken(_x: int, _y: int) -> None:
        raise OSError("no input desktop")

    r2 = GlideRunner(g, broken, clock=lambda: now[0])
    now[0] = 0.2
    assert not r2.tick() and r2.errors == 1


def test_glide_runner_thread_runs_and_stops() -> None:
    g = Glide()
    g.on_target(0.0, 0.0, 0.0)
    r = GlideRunner(g, lambda x, y: None, hz=240).start()
    threading.Event().wait(0.1)
    r.stop()
    assert r.thread is not None
    r.thread.join(2)
    assert not r.thread.is_alive() and r.ticks > 5


# ============================================================================ (i) busiest programs, live removal
class FakeProc:
    def __init__(self, name: str, pct: float, pid: int) -> None:
        self.info = {"name": name, "pid": pid}
        self.pct = pct

    def cpu_percent(self, _i: Any) -> float:
        return self.pct


class FakePsutil:
    def __init__(self, procs: list[FakeProc], cores: int = 4) -> None:
        self.procs, self.cores = procs, cores

    def cpu_count(self) -> int:
        return self.cores

    def process_iter(self, _attrs: Any) -> list[FakeProc]:
        return self.procs


def test_top_processes_are_named_added_up_and_sorted() -> None:
    ps = FakePsutil(
        [
            FakeProc("NVIDIA Broadcast.exe", 120.0, 1),
            FakeProc("chrome.exe", 40.0, 2),
            FakeProc("chrome.exe", 40.0, 3),
            FakeProc("System Idle Process", 300.0, 0),
            FakeProc("python.exe", 0.0, 4),
            FakeProc("MsMpEng.exe", 8.0, 5),
            FakeProc("a.exe", 1.0, 6),
            FakeProc("b.exe", 1.0, 7),
        ]
    )
    rows = sysload.top_processes(5, psutil_mod=ps)
    assert [r["name"] for r in rows][:3] == ["NVIDIA Broadcast", "Google Chrome", "Windows Security scan"]
    assert rows[0]["cpu_pct"] == 30.0 and rows[1]["cpu_pct"] == 20.0  # percent of the whole PC (4 cores)
    assert len(rows) == 5 and all(r["name"] != "System Idle Process" for r in rows)
    assert sysload.describe_top(rows) == "NVIDIA Broadcast 30%, Google Chrome 20%"


def test_load_hints_need_a_sustained_reading() -> None:
    clk = Clock()
    h = sysload.LoadHints(clock=clk)
    top = [{"name": "NVIDIA Broadcast", "cpu_pct": 40.0}]
    assert h.update(95.0, 30.0, top) == ""
    clk.t += sysload.BUSY_S
    assert h.update(95.0, 30.0) == "Your PC is busy (NVIDIA Broadcast 40%), so gestures may be choppy"
    clk.t += 1
    assert h.update(50.0, 30.0) == ""  # one calm reading clears it
    h.update(50.0, 8.0, reason="a camera that keeps failing is being retried")
    clk.t += sysload.CHOPPY_S
    assert h.update(50.0, 8.0, reason="x") == "Hand tracking is choppy (8 fps): x"
    assert sysload.choppy_reason(types.SimpleNamespace(camera_unavailable={"x": 1})).startswith("a camera that")
    assert "slow" in sysload.choppy_reason(types.SimpleNamespace())


def test_health_reports_the_busiest_programs(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.agentlink import telemetry

    calls = []
    monkeypatch.setattr(sysload, "top_processes", lambda n=5: calls.append(n) or [{"name": "X", "cpu_pct": 9.0}])
    t = telemetry.Telemetry.__new__(telemetry.Telemetry)
    now = [0.0]
    t.clock = lambda: now[0]
    assert t.top_processes() == [{"name": "X", "cpu_pct": 9.0}]
    now[0] = 5.0
    t.top_processes()
    assert calls == [5]  # cached for 10 s
    now[0] = 11.0
    t.top_processes()
    assert calls == [5, 5]


def test_camera_removed_in_settings_applies_live() -> None:
    from away.app.runner import AwayApp

    app = AwayApp.__new__(AwayApp)
    cfg = C.AppConfig()
    cfg.hands.cameras = [C.CameraSpec(device="JoyCam", role="primary"), C.CameraSpec(device="HD User Facing")]
    app.cfg = cfg
    woke = threading.Event()
    app._retry_event = lambda: woke  # type: ignore[method-assign]
    new = C.AppConfig()
    new.hands.cameras = []
    new.hands.camera_index = "JoyCam"
    new.hands.disabled_cameras = [C.CameraSpec(device="HD User Facing")]
    app._apply_camera_switch(new)
    assert app._camera_set_dirty and woke.is_set()
    assert app.cfg.hands.cameras == [] and app.cfg.hands.camera_index == "JoyCam"
    assert [c.device for c in app.cfg.hands.disabled_cameras] == ["HD User Facing"]
    woke.clear()
    app._camera_set_dirty = False
    app._apply_camera_switch(new)  # nothing changed: no restart of the cameras
    assert not woke.is_set() and not app._camera_set_dirty


def test_turn_off_and_on_a_camera_keeps_its_settings() -> None:
    from away.app import camera_switch as CS

    cfg = C.AppConfig()
    cfg.hands.cameras = [
        C.CameraSpec(device="JoyCam", role="primary"),
        C.CameraSpec(device="HD User Facing", role="secondary", fourcc="NV12"),
    ]
    assert CS.turn_off(cfg, "hd user facing") == "Turned off HD User Facing"
    assert cfg.hands.cameras == [] and cfg.hands.camera_index == "JoyCam"
    assert CS.turn_off(cfg, "HD User Facing") == "HD User Facing is already off"
    with pytest.raises(CS.CameraSwitchError, match="only camera"):
        CS.turn_off(cfg, "JoyCam")
    assert [r["value"] for r in CS.listing(cfg)] == ["on", "off"]
    assert CS.turn_on(cfg, "HD User Facing") == "Turned on HD User Facing"
    assert [(c.device, c.role, c.fourcc) for c in cfg.hands.cameras] == [
        ("JoyCam", "primary", cfg.hands.cameras[0].fourcc),
        ("HD User Facing", "secondary", "NV12"),
    ]


# ============================================================================ (k) who holds the camera
def _entries() -> list[tuple[str, bool, dict[str, Any]]]:
    return [
        (
            "C:#Program Files#NVIDIA Corporation#NVIDIA Broadcast#NVIDIA Broadcast.exe",
            False,
            {"LastUsedTimeStart": 133, "LastUsedTimeStop": 0},
        ),
        ("Microsoft.WindowsCamera_8wekyb3d8bbwe", True, {"LastUsedTimeStart": 5, "LastUsedTimeStop": 9}),
        ("MSTeams_8wekyb3d8bbwe", True, {"LastUsedTimeStart": 7, "LastUsedTimeStop": 0}),
        ("C:#Program Files#Project Away#ProjectAway.exe", False, {"LastUsedTimeStart": 1, "LastUsedTimeStop": 0}),
        ("C:#Python312#python.exe", False, {"LastUsedTimeStart": 1, "LastUsedTimeStop": 0}),
        ("C:#x#never.exe", False, {"LastUsedTimeStart": 0, "LastUsedTimeStop": 0}),
    ]


def test_parse_finds_the_apps_using_a_camera_now() -> None:
    users = CU.parse(_entries())
    assert [(u.name, u.packaged) for u in users] == [("NVIDIA Broadcast", False), ("MSTeams", True)]
    assert users[0].path == r"C:\Program Files\NVIDIA Corporation\NVIDIA Broadcast\NVIDIA Broadcast.exe"
    assert CU.current_users(lambda: 1 / 0) == []  # type: ignore[arg-type, return-value]


class FakeWinreg:
    """The ConsentStore\\webcam tree as nested dicts: {subkey: {...}} with values in "__values__"."""

    HKEY_CURRENT_USER = "HKCU"

    def __init__(self, tree: dict[str, Any]) -> None:
        self.tree = tree

    class _K:
        def __init__(self, node: dict[str, Any]) -> None:
            self.node = node

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    def OpenKey(self, parent: Any, name: str) -> Any:
        node = self.tree if parent == "HKCU" else parent.node
        if parent == "HKCU":
            node = self.tree
            for part in name.split("\\"):
                node = node[part]
            return self._K(node)
        if name not in node:
            raise OSError(name)
        return self._K(node[name])

    def EnumKey(self, k: Any, i: int) -> str:
        keys = [n for n in k.node if n != "__values__"]
        if i >= len(keys):
            raise OSError("no more")
        return keys[i]

    def QueryValueEx(self, k: Any, name: str) -> tuple[Any, int]:
        vals = k.node.get("__values__", {})
        if name not in vals:
            raise OSError(name)
        return vals[name], 11


def test_read_registry_walks_packaged_and_desktop_apps() -> None:
    tree: dict[str, Any] = {}
    node = tree
    for part in CU.KEY.split("\\"):
        node = node.setdefault(part, {})
    node["Microsoft.WindowsCamera_8wekyb3d8bbwe"] = {"__values__": {"LastUsedTimeStart": 3, "LastUsedTimeStop": 0}}
    node["NonPackaged"] = {
        "C:#Apps#Zoom.exe": {"__values__": {"LastUsedTimeStart": 3, "LastUsedTimeStop": 0}},
        "C:#Apps#Old.exe": {"__values__": {"LastUsedTimeStart": 3, "LastUsedTimeStop": 4}},
    }
    rows = CU.read_registry(FakeWinreg(tree))
    users = CU.parse(rows)
    assert sorted(u.name for u in users) == ["Windows Camera", "Zoom"]


def test_broadcast_source_from_its_settings_file(tmp_path: Path) -> None:
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"effects": {"camera": {"sourceDevice": "HD User Facing", "blur": 1}}}))
    assert CU.broadcast_source(paths=[tmp_path / "missing.json", f]) == "HD User Facing"
    assert CU.broadcast_source(paths=[]) == ""
    assert CU.broadcast_settings_paths({"LOCALAPPDATA": "L"})[0].parts[-3:] == (
        "NVIDIA Corporation",
        "NVIDIA Broadcast",
        "settings.json",
    )


def test_explain_names_the_holder_in_plain_words() -> None:
    users = CU.parse(_entries())
    via = CU.explain("HD User Facing", users, broadcast_from="HD User Facing")
    assert via["broadcast_uses_it"] and via["message"].startswith("Your built-in camera is used by NVIDIA Broadcast")
    held = CU.explain("HD User Facing", users)
    assert (
        held["message"] == "Your built-in camera is being used by NVIDIA Broadcast and MSTeams. Close it, then Retry."
    )
    assert held["actions"][0].startswith("close:C:\\Program Files\\NVIDIA") and held["actions"][-1] == "retry"
    free = CU.explain("JoyCam", [])
    assert free["message"].startswith("JoyCam could not be opened") and free["actions"] == ["retry"]


def test_pc_context_lines_for_diagnose() -> None:
    from away import cli

    users = CU.parse(_entries())
    lines = cli.pc_context_lines(users, [{"name": "NVIDIA Broadcast", "cpu_pct": 31.0}], "HD User Facing")
    assert lines == [
        "apps using a camera now: NVIDIA Broadcast, MSTeams",
        "NVIDIA Broadcast takes its picture from: HD User Facing",
        "busiest programs: NVIDIA Broadcast 31%",
    ]
    assert cli.pc_context_lines([], [], "") == ["apps using a camera now: none"]


# ============================================================================ (l) WinRT formats, no Broadcast
@pytest.fixture
def no_configured_cameras() -> Any:
    capture.configure_cameras([])
    yield
    capture.configure_cameras([])


@pytest.mark.parametrize(
    ("name", "internal"),
    [
        ("HD User Facing", True),
        ("Integrated Camera", True),
        ("FaceTime HD Camera", True),
        ("Laptop Camera", True),
        ("JoyCam", False),
        ("Logitech BRIO", False),
        ("Camera (NVIDIA Broadcast)", False),
    ],
)
def test_internal_camera_names(name: str, internal: bool) -> None:
    assert capture.is_internal_camera(name) is internal


def test_no_broadcast_fallback_while_the_built_in_camera_is_configured(no_configured_cameras: Any) -> None:
    assert not capture.broadcast_fallback_allowed("HD User Facing")  # never for the built-in camera itself
    assert capture.broadcast_fallback_allowed("JoyCam")
    capture.configure_cameras(["JoyCam", "HD User Facing"])
    assert not capture.broadcast_fallback_allowed("JoyCam")  # Broadcast would take the built-in camera
    capture.configure_cameras(["JoyCam", 1])
    assert capture.broadcast_fallback_allowed("JoyCam")


def _native(w: int, h: int, fps: float, sub: str = "NV12") -> Any:
    from away.multicam import winrt_capture as W

    return W.NativeFormat(w, h, fps, sub)


def test_ranked_formats_closest_first_then_the_rest() -> None:
    from away.multicam import winrt_capture as W

    fmts = [_native(1920, 1080, 30), _native(640, 480, 30), _native(1280, 720, 30), _native(640, 480, 15)]
    ranked = W.ranked_formats(fmts, 640, 480, 30)
    assert ranked[0] == fmts[1] and len(ranked) == 4 and {id(f) for f in ranked} == {id(f) for f in fmts}
    assert W.ranked_formats([], 640, 480, 30) == []


def test_native_modes_turn_reported_formats_into_capture_modes() -> None:
    fmts = [_native(1280, 720, 30.0), _native(1280, 720, 29.97), _native(640, 360, 30), _native(320, 240, 15)]
    modes = capture.native_modes("HD User Facing", 640, 480, 30, lister=lambda name: (fmts, ""), limit=3)
    assert len(modes) == 3 and all(isinstance(m, capture.CaptureMode) for m in modes)
    assert len(set(modes)) == len(modes)  # 30 and 29.97 fps collapse into one mode
    assert capture.native_modes(None, 640, 480, 30, lister=lambda n: (fmts, "")) == []
    assert capture.native_modes("x", 640, 480, 30, lister=lambda n: 1 / 0) == []  # type: ignore[arg-type,return-value]


def test_winrt_tries_the_next_format_after_element_not_found() -> None:
    """The camera reports formats it then refuses ("Element not found"): each is tried, closest first."""
    from away.multicam import winrt_capture as W
    from test_v0119_winrt_camera import FakeApi, nf  # the fake WinRT camera of the v0.1.19 tests

    fmts = [nf(1280, 720, 30, "NV12", 0), nf(640, 480, 30, "MJPG", 1), nf(1920, 1080, 30, "NV12", 2)]

    class Refusing(FakeApi):
        async def set_format(self, source: Any, fmt: Any) -> None:
            self.calls.append(("set_format", fmt))
            if fmt.width != 1920:
                raise OSError("Element not found. (0x80070490)")
            source.current = fmt

    api = Refusing(formats=fmts, current=fmts[0])
    cap = W.WinrtCapture("HD User Facing", 640, 480, 30, "", api=api, open_timeout=5.0, out_size=(640, 480))
    try:
        assert cap.isOpened()
        sets = [c[1] for c in api.calls if c[0] == "set_format"]
        assert len(sets) == 3 and sets[-1].width == 1920  # every refused format tried before the one that worked
        assert sets == W.ranked_formats(fmts, 640, 480, 30)
        assert cap.native == fmts[2] and len(cap.tried) == 2 and "0x80070490" in cap.tried[0]
    finally:
        cap.release()

    api2 = Refusing(formats=fmts[:2], current=fmts[0])  # none can be set: the camera's current format is used
    cap2 = W.WinrtCapture("HD User Facing", 640, 480, 30, "", api=api2, open_timeout=5.0, out_size=(640, 480))
    try:
        assert cap2.isOpened() and cap2.native == fmts[0] and len(cap2.tried) == 2
    finally:
        cap2.release()
