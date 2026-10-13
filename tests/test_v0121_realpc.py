"""v0.1.21: the fixes for what the user's real PC showed (Acer Predator Helios 300, three monitors on a 5206x1080
desktop, JoyCam + HD User Facing + NVIDIA Broadcast).

1. Agent messages answered ``shown: true`` although no window appeared: the result is now truthful, with a tray
   notification as the fallback, and the message window starts like a normal window (no CREATE_NO_WINDOW, frozen
   environment reset).
2. ``notify(speak=true)`` always answered ``spoken: false`` (pyttsx3 was not bundled): a built-in voice chain
   (SAPI via comtypes, PowerShell System.Speech, pyttsx3, say, spd-say/espeak) and a ``speech:`` config section.
3. ``get_job?job_id=82557239`` said "unknown check": ids stay strings, the registry is shared, finished checks are
   kept for a while, ``id`` is an alias, and unknown (404) differs from expired (410).
4. After Settings borrowed the camera, gestures stayed paused: the lease gives the camera back when the preview
   stops, the app re-checks a vanished request once, and the status says who has the camera.
5. JoyCam asked for MJPG read back YUY2 and ran 720p at 10 fps: MJPG is retried size-first, a mode below 24 fps
   only counts when nothing faster works, and the best mode is remembered.
6. "uh" and "Dikontak" were typed with ``mouth_moving_frames: 0``: fillers are dropped and weak gate evidence needs
   a confident transcript; short voice commands keep working.
7. ``get_screen`` scaled 5206 px to 960 (unreadable): scale / monitor / region with a monitor list.
8. 37 MCP tools became 15 grouped tools; every old REST / rpc name still works.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.agentlink import groups, jobs, prompts, screenshot
from away.agentlink.about import INSTRUCTIONS, briefing
from away.agentlink.prompts import NullUI, PromptManager, question_speech, wait_for_ack, window_process_options
from away.agentlink.telemetry import Telemetry
from away.app import camera_share as CS
from away.app import selftest
from away.voice import speech, weak_speech
from away.voice.weak_speech import confidence_of, is_filler, should_drop, weak_evidence

GROUPED_TOOLS = {  # v0.1.22: + inbox (16 tools)
    "inbox",
    "get_status",
    "get_context",
    "events",
    "settings",
    "preferences",
    "diagnostics",
    "look",
    "get_screen",
    "ask_user",
    "notify_user",
    "control",
    "pointer",
    "keyboard",
    "open",
    "disconnect",
}


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(float(s), 1e-4)


# ===================================================================================== 1. truthful messages
def test_window_process_options_show_the_window_and_reset_the_frozen_environment(monkeypatch) -> None:
    import subprocess

    env = {"PATH": "/bin", "_PYI_APPLICATION_HOME_DIR": "x", "_PYI_ARCHIVE_FILE": "y", "_MEIPASS2": "z"}
    kw = window_process_options("win32", env)
    assert "creationflags" not in kw  # CREATE_NO_WINDOW hid the message window on the user's PC
    assert kw["env"] == {"PATH": "/bin"}
    assert kw["stdin"] is subprocess.DEVNULL and kw["stdout"] is subprocess.DEVNULL and kw["close_fds"]
    if sys.platform == "win32":  # pragma: no cover - only the Windows runner has STARTUPINFO
        assert kw["startupinfo"].wShowWindow == 1
    else:
        assert "startupinfo" not in kw
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert window_process_options("linux", env)["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    monkeypatch.delattr(sys, "frozen")
    assert "PYINSTALLER_RESET_ENVIRONMENT" not in window_process_options("linux", env)["env"]


class Proc:
    def __init__(self, code: int | None = None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


def test_wait_for_ack_reports_shown_problem_exit_and_timeout(tmp_path: Path) -> None:
    ack = tmp_path / "toast.ack.json"
    clock = Clock()
    ack.write_text(json.dumps({"shown": True}), encoding="utf-8")
    assert wait_for_ack(ack, Proc(), 1.0, clock.sleep, clock) == {"shown": True, "via": "window"}
    ack.write_text(json.dumps({"shown": False, "error": "TclError: no display"}), encoding="utf-8")
    res = wait_for_ack(ack, Proc(), 1.0, clock.sleep, clock)
    assert res["shown"] is False and "no display" in res["reason"]
    ack.unlink()
    res = wait_for_ack(ack, Proc(3), 1.0, clock.sleep, clock)
    assert res["shown"] is False and "exit code 3" in res["reason"]
    t0 = clock.t
    res = wait_for_ack(ack, Proc(None), 2.0, clock.sleep, clock)
    assert res["shown"] is False and "within 2 s" in res["reason"] and clock.t - t0 >= 2.0


class UI(NullUI):
    def __init__(self, result: Any = None, boom: bool = False) -> None:
        super().__init__()
        self.result, self.boom = result, boom

    def toast(self, text: str, level: str = "info", seconds: float = 6.0, title: str = "") -> Any:
        self.toasts.append((level, text))
        self.titles.append(title)
        if self.boom:
            raise OSError("spawn failed")
        return self.result


def test_notify_is_truthful_and_falls_back_to_the_tray() -> None:
    shown_tray: list[tuple[str, str]] = []
    ui = UI({"shown": False, "reason": "the message window did not appear within 12 s"})
    pm = PromptManager(ui=ui, fallback=lambda t, x: shown_tray.append((t, x)) or True)
    res = pm.notify("Build finished", agent="Claude")
    assert res["shown"] is True and res["via"] == "tray notification" and "12 s" in res["window_problem"]
    assert shown_tray == [("Message from Claude", "Build finished")] and ui.titles == ["Message from Claude"]
    assert res["spoken"] is False

    nothing = PromptManager(ui=UI({"shown": False, "reason": "no display"}), fallback=lambda t, x: False)
    res = nothing.notify("hi")
    assert res["shown"] is False and res["reason"] == "no display"

    broken = PromptManager(ui=UI(boom=True))
    res = broken.notify("hi")
    assert res["shown"] is False and "spawn failed" in res["reason"]

    old = PromptManager(ui=UI(None))  # an older UI returning nothing counts as shown
    assert old.notify("hi")["shown"] is True
    assert PromptManager(ui=NullUI()).notify("x", agent="")["shown"] is True


def test_notify_event_carries_the_truth() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []
    pm = PromptManager(ui=UI({"shown": False, "reason": "x"}), on_event=lambda t, d: seen.append((t, d)))
    pm.notify("hello", level="nonsense")
    assert seen == [("notify", {"text": "hello", "level": "info", "shown": False, "spoken": False})]


# ===================================================================================== 2. speaking
class FakeEngine:
    def __init__(self, name: str = "fake", fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.said: list[tuple[str, speech.Voice]] = []
        self.closed = False

    def speak(self, text: str, v: speech.Voice) -> None:
        if self.fail:
            raise RuntimeError("audio device gone")
        self.said.append((text, v))

    def close(self) -> None:
        self.closed = True


def _cfg(**kw: Any) -> C.SpeechConfig:
    return C.SpeechConfig(**kw)


def test_speaker_speaks_on_its_own_thread_with_the_settings() -> None:
    eng = FakeEngine()
    sp = speech.Speaker(lambda: _cfg(rate="fast", volume="loud", voice="Zira"), [("fake voice", lambda: eng)])
    res = sp.say("  Your   build \n finished  ")
    assert res == {"spoken": True, "engine": "fake voice", "queued": True}
    assert sp.wait(5)
    assert eng.said == [("Your build finished", speech.Voice("fast", "loud", "Zira"))]
    assert sp.spoken == ["Your build finished"] and sp.engine_name == "fake voice"
    assert sp.say("") == {"spoken": False, "reason": "nothing to say"}
    assert len(speech._clean("x" * 5000)) == speech.MAX_TEXT
    sp.stop()


def test_speaker_respects_let_agents_talk_and_reports_no_voice() -> None:
    eng = FakeEngine()
    off = speech.Speaker(lambda: _cfg(enabled=False), [("fake", lambda: eng)])
    res = off.say("hello")
    assert res["spoken"] is False and "turned off" in res["reason"]
    assert off.say("forced", force=True)["spoken"] is True and off.wait(5)
    assert eng.said[-1][0] == "forced"
    none = speech.Speaker(lambda: _cfg(), [])
    assert none.say("hello") == {"spoken": False, "reason": "no voice is available on this computer"}
    assert none.status()["available"] is False and none.status()["engine"] is None
    broken_settings = speech.Speaker(lambda: 1 / 0, [("fake", lambda: eng)])  # type: ignore[arg-type]
    assert broken_settings.allowed() and broken_settings.voice() == speech.Voice()


def test_speaker_falls_back_to_the_next_engine_and_reports_failures() -> None:
    errors: list[str] = []
    good = FakeEngine("good")

    def broken() -> Any:
        raise OSError("comtypes missing")

    sp = speech.Speaker(lambda: None, [("SAPI", broken), ("PowerShell", lambda: good)], on_error=errors.append)
    sp.say("one")
    assert sp.wait(5)
    assert good.said[0][0] == "one" and sp.engine_name == "PowerShell"
    assert errors and "SAPI could not start: comtypes missing" in errors[0]

    bad = FakeEngine("bad", fail=True)
    sp2 = speech.Speaker(lambda: None, [("bad", lambda: bad)], on_error=errors.append)
    sp2.say("two")
    assert sp2.wait(5)
    assert "bad failed: audio device gone" in sp2.last_error and bad.closed and sp2.spoken == []


def test_engine_order_per_platform() -> None:
    def which(found: set[str]) -> Any:
        return lambda exe: f"/usr/bin/{exe}" if exe in found else None

    win = speech.engine_factories("win32", which({"powershell"}), have=lambda m: m == "comtypes")
    assert [n for n, _ in win] == ["Windows voice (SAPI)", "Windows voice (PowerShell)"]
    win_no_comtypes = speech.engine_factories("win32", which({"powershell"}), have=lambda m: False)
    assert [n for n, _ in win_no_comtypes] == ["Windows voice (PowerShell)"]
    assert isinstance(win_no_comtypes[0][1](), speech.PowerShellEngine)
    mac = speech.engine_factories("darwin", which({"say"}), have=lambda m: False)
    assert [n for n, _ in mac] == ["say"]
    linux = speech.engine_factories("linux", which({"spd-say", "espeak"}), have=lambda m: m == "pyttsx3")
    assert [n for n, _ in linux] == ["pyttsx3", "spd-say"]
    eng = linux[1][1]()
    assert isinstance(eng, speech.CommandEngine) and eng.kind == "spd-say"
    only_espeak = speech.engine_factories("linux", which({"espeak"}), have=lambda m: False)
    assert [n for n, _ in only_espeak] == ["espeak"]
    assert speech.engine_factories("linux", which(set()), have=lambda m: False) == []


def test_powershell_engine_passes_text_on_stdin_and_never_injects_a_voice_name() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    eng = speech.PowerShellEngine("powershell.exe", run=lambda argv, **kw: calls.append((argv, kw)))
    eng.speak("Hello; Remove-Item C:\\ -Recurse", speech.Voice("slow", "quiet", "Zira"))
    argv, kw = calls[0]
    assert kw["input"] == "Hello; Remove-Item C:\\ -Recurse" and "Remove-Item" not in " ".join(argv)
    assert "$s.Rate = -3" in argv[-1] and "$s.Volume = 45" in argv[-1] and "*Zira*" in argv[-1]
    evil = eng.command(speech.Voice(voice="x'; Stop-Computer; '"))
    assert "Stop-Computer" not in evil[-1] and "SelectVoice" not in evil[-1]


def test_command_engines_put_the_text_after_double_dash() -> None:
    v = speech.Voice("fast", "loud", "Samantha")
    say = speech.CommandEngine("/usr/bin/say", "say").command("-v evil", v)
    assert say == ["/usr/bin/say", "-r", "225", "-v", "Samantha", "--", "-v evil"]
    spd = speech.CommandEngine("/usr/bin/spd-say", "spd-say").command("hi", v)
    assert spd == ["/usr/bin/spd-say", "-w", "-r", "35", "-i", "60", "--", "hi"]
    esp = speech.CommandEngine("/usr/bin/espeak", "espeak").command("hi", speech.Voice())
    assert esp == ["/usr/bin/espeak", "-s", "180", "-a", "110", "--", "hi"]
    ran: list[list[str]] = []
    speech.CommandEngine("/usr/bin/espeak", "espeak", run=lambda a, **k: ran.append(a)).speak("hi", speech.Voice())
    assert ran[0][-1] == "hi"


class FakeToken:
    def __init__(self, desc: str) -> None:
        self.desc = desc

    def GetDescription(self) -> str:
        return self.desc


class FakeTokens:
    def __init__(self, names: list[str]) -> None:
        self.items = [FakeToken(n) for n in names]
        self.Count = len(self.items)

    def Item(self, i: int) -> FakeToken:
        return self.items[i]


class FakeSpVoice:
    def __init__(self) -> None:
        self.Rate = 0
        self.Volume = 100
        self.Voice: Any = None
        self.spoken: list[str] = []

    def GetVoices(self) -> FakeTokens:
        return FakeTokens(["Microsoft David Desktop", "Microsoft Zira Desktop"])

    def Speak(self, text: str, flags: int) -> None:
        self.spoken.append(text)


def test_sapi_engine_sets_rate_volume_and_voice() -> None:
    created: list[str] = []
    voice = FakeSpVoice()
    inits: list[int] = []
    eng = speech.SapiEngine(create=lambda p: created.append(p) or voice, co_init=lambda: inits.append(1))
    assert created == ["SAPI.SpVoice"] and inits == [1]
    assert eng.voices() == ["Microsoft David Desktop", "Microsoft Zira Desktop"]
    eng.speak("Hello", speech.Voice("fast", "quiet", "zira"))
    assert voice.spoken == ["Hello"] and voice.Rate == 3 and voice.Volume == 45
    assert voice.Voice.GetDescription() == "Microsoft Zira Desktop"
    eng.speak("Again", speech.Voice("nonsense", "nonsense", "Nobody"))
    assert voice.Rate == 0 and voice.Volume == 80  # unknown values fall back to normal
    eng.close()
    assert eng.voice is None


def test_notify_and_ask_speak_through_the_speaker() -> None:
    eng = FakeEngine()
    sp = speech.Speaker(lambda: _cfg(), [("Windows voice (SAPI)", lambda: eng)])
    pm = PromptManager(ui=NullUI(), tts=sp, polite=lambda: True)
    res = pm.notify("Tests passed", speak=True, agent="Claude")
    assert res["shown"] and res["spoken"] and res["voice"] == "Windows voice (SAPI)"
    assert sp.wait(5) and eng.said[-1][0] == "Tests passed"
    assert pm.notify("quiet one")["spoken"] is False  # not asked, read-aloud off

    aloud = speech.Speaker(lambda: _cfg(read_messages_aloud=True), [("fake", lambda: eng)])
    pm2 = PromptManager(ui=NullUI(), tts=aloud)
    assert pm2.notify("every message")["spoken"] is True and aloud.wait(5)

    off = speech.Speaker(lambda: _cfg(enabled=False), [("fake", lambda: eng)])
    res = PromptManager(ui=NullUI(), tts=off).notify("x", speak=True)
    assert res["spoken"] is False and "turned off" in res["speech"]
    res = PromptManager(ui=NullUI(), tts=None).notify("x", speak=True)
    assert res["spoken"] is False and "no voice" in res["speech"]

    pm.create("Deploy to production?", ["Yes", "No"], agent="Claude", speak=True)
    pm.pump()
    assert sp.wait(5) and eng.said[-1][0] == "Claude asks: Deploy to production? Yes or No?"


def test_question_speech_wording() -> None:
    p = prompts.Prompt(id="p", title="", agent="", question="Pick one", options=["A", "B", "C"])
    assert question_speech(p) == "Pick one A, B or C?"
    p1 = prompts.Prompt(id="p", title="", agent="Cursor", question="Okay?", options=["Fine"])
    assert question_speech(p1) == "Cursor asks: Okay? Fine?"


def test_speech_config_defaults_validation_and_yaml(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    assert cfg.speech == C.SpeechConfig(True, False, "normal", "normal", "")
    for bad in (C.SpeechConfig(rate="ludicrous"), C.SpeechConfig(volume="max"), C.SpeechConfig(voice="x" * 61)):
        with pytest.raises(ValueError, match="speech"):
            C.validate_speech(bad)
    p = tmp_path / "config.yaml"
    p.write_text("speech:\n  rate: fast\n  voice: Zira\n  read_messages_aloud: true\n", encoding="utf-8")
    loaded = C.load(p)[0]
    assert loaded.speech.rate == "fast" and loaded.speech.voice == "Zira" and loaded.speech.read_messages_aloud


def test_selftest_has_a_speech_check_that_runs_here() -> None:
    names = [c.name for c in selftest.CHECKS]
    assert "speech" in names
    res = selftest.run([c for c in selftest.CHECKS if c.name == "speech"], display=False)
    assert res["speech"]["ok"] is True, res
    assert res["speech"]["detail"].startswith(("engines here:", "no voice engine here", "SAPI.SpVoice ok"))
    assert speech.selftest_check("linux").startswith(("engines here:", "no voice engine here"))


def test_release_workflow_runs_the_frozen_windows_selftest() -> None:
    yml = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert '"selftest", "--json"' in yml and "check_selftest.py selftest.json" in yml


# ===================================================================================== 3. job ids
def test_job_ids_are_normalized_strings() -> None:
    assert jobs.normalize_id(82557239) == "82557239"
    assert jobs.normalize_id(82557239.0) == "82557239"
    assert jobs.normalize_id(" 8255abCD ") == "8255abcd"
    assert jobs.normalize_id(None) == "" and jobs.normalize_id(True) == ""


def _runner(tmp_path: Path, clock: Clock) -> jobs.JobRunner:
    return jobs.JobRunner(tmp_path / "diag", command=lambda a: ["away", *a], bench=lambda: {"x_ms": 1.0}, clock=clock)


def _wait_done(r: jobs.JobRunner, jid: str) -> jobs.Job:
    for _ in range(500):
        j = r.jobs[jid]
        if j.state != "running":
            return j
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_job_round_trips_by_int_and_string_and_expires(tmp_path: Path) -> None:
    clock = Clock()
    r = _runner(tmp_path, clock)
    job = r.start("benchmark")
    assert isinstance(job.id, str) and len(job.id) == 8
    _wait_done(r, job.id)
    r.jobs["82557239"] = r.jobs.pop(job.id)
    r.jobs["82557239"].id = "82557239"
    assert r.lookup(82557239).public()["result"] == {"x_ms": 1.0}
    assert r.lookup("82557239").state == "done" and r.jobs["82557239"].fetched
    with pytest.raises(jobs.JobLookupError) as e:
        r.lookup("deadbeef")
    assert e.value.reason == "unknown" and "unknown check deadbeef" in str(e.value)
    clock.t += jobs.KEEP_S + 1  # read once: kept KEEP_S after it finished
    with pytest.raises(jobs.JobLookupError) as e:
        r.lookup(82557239)
    assert e.value.reason == "expired" and "expired" in str(e.value)


def test_unread_job_is_kept_longer_and_running_never_expires(tmp_path: Path) -> None:
    clock = Clock()
    r = _runner(tmp_path, clock)
    j = r.start("benchmark")
    _wait_done(r, j.id)
    clock.t += jobs.KEEP_S + 60  # never read: kept up to KEEP_UNREAD_S
    assert r.get(j.id) is not None
    running = jobs.Job("aaaa1111", "cameras", clock())
    r.jobs[running.id] = running
    clock.t += jobs.KEEP_UNREAD_S * 3
    assert r.get("aaaa1111") is running
    with pytest.raises(RuntimeError, match="another check"):
        r.start("benchmark")
    with pytest.raises(ValueError):
        r.start("rm -rf")


def test_shared_runner_is_one_per_folder(tmp_path: Path) -> None:
    a = jobs.shared_runner(tmp_path / "d")
    assert jobs.shared_runner(tmp_path / "d") is a
    assert jobs.shared_runner(tmp_path / "other") is not a


def test_api_error_status_404_unknown_410_expired() -> None:
    from away.agentlink import api

    assert api.error_status(jobs.JobLookupError("1", "unknown"))[0] == 404
    assert api.error_status(jobs.JobLookupError("1", "expired"))[0] == 410


# ===================================================================================== core fixture
HIS_DESKTOP = [  # left to right: 1920 + 1920 + 1366 = 5206 wide, 1080 high
    types.SimpleNamespace(left=-1920, top=0, width=1920, height=1080, primary=False),
    types.SimpleNamespace(left=0, top=0, width=1920, height=1080, primary=True),
    types.SimpleNamespace(left=1920, top=0, width=1366, height=768, primary=False),
]


@pytest.fixture
def core(tmp_path: Path):
    from away.agentlink.host import build_core

    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    c = build_core(
        tmp_path,
        cfg=lambda: cfg,
        settings=lambda: cfg.agent_link,
        config_path=lambda: cfgp,
        ui=NullUI(),
        screen_grab=lambda: None,
        monitors=lambda: HIS_DESKTOP,
    )
    c.prompts.polite = lambda: True
    return c


def _done_job(core: Any, jid: str = "82557239") -> None:
    j = jobs.Job(jid, "cameras", time.time(), state="done", output=["JoyCam ok"], finished=core.service.jobs.clock())
    core.service.jobs.jobs[jid] = j


def test_service_get_job_accepts_id_alias(core) -> None:
    a, _ = core.registry.issue("Agent", ["read", "diagnose"], 3600)
    _done_job(core)
    assert core.service.get_job(a, job_id="82557239")["id"] == "82557239"
    assert core.service.get_job(a, id=82557239)["output"] == ["JoyCam ok"]
    with pytest.raises(ValueError, match="job_id"):
        core.service.get_job(a)
    with pytest.raises(jobs.JobLookupError):
        core.service.get_job(a, id="00000000")


# ===================================================================================== 4. camera reclaim
def test_borrower_labels_in_plain_words() -> None:
    assert CS.borrower_label("settings-studio") == "Settings (Gesture Studio)"
    assert CS.borrower_label("settings-new-tab") == "Settings"
    assert CS.borrower_label(None) == "Another Project Away window"
    assert CS.in_use_status("settings-tuning") == "Settings (Tuning live view) is using the camera"


def test_app_reads_a_vanished_request_once_more(tmp_path: Path) -> None:
    clock = Clock(time.time())
    app = CS.CameraShare(tmp_path, clock=clock, pid=1)
    settings = CS.CameraShare(tmp_path, clock=clock, pid=2)
    waits: list[float] = []

    def rewrite_during_wait(s: float) -> None:  # the borrower rewrote the file while the app read it
        waits.append(s)
        settings.request("settings-studio")

    assert app.still_requested(rewrite_during_wait) is True and waits == [CS.RECLAIM_CONFIRM_S]
    assert app.requester() == "settings-studio"
    settings.release()
    assert app.still_requested(lambda s: None) is False and app.requester() is None


def test_lease_wants_the_camera_only_while_frames_arrive(tmp_path: Path) -> None:
    clock = Clock()
    stop = threading.Event()
    lease = CS.CameraLease(CS.CameraShare(tmp_path), idle_s=2.0, grace_s=10.0, stop=stop, clock=clock)
    assert lease.wants_camera()  # still opening
    clock.t += 9.0
    assert lease.wants_camera()
    clock.t += 2.0
    assert not lease.wants_camera()  # opened 11 s ago, never a frame
    lease.touch()
    assert lease.wants_camera()
    clock.t += 2.5
    assert not lease.wants_camera()  # frames stopped
    lease.touch()
    stop.set()
    assert not lease.wants_camera()  # the preview was stopped
    assert CS.CameraLease(CS.CameraShare(tmp_path)).wants_camera()  # no idle_s: always


def test_stopped_preview_gives_the_camera_back(tmp_path: Path) -> None:
    app = CS.CameraShare(tmp_path, pid=1)
    stop = threading.Event()
    lease = CS.CameraLease(CS.CameraShare(tmp_path, pid=2), timeout=0.2, who="settings-studio", stop=stop)
    lease.heartbeat_s = 0.05
    lease.open()
    try:
        assert app.requested() and app.requester() == "settings-studio"
        stop.set()
        for _ in range(100):
            if not app.requested():
                break
            time.sleep(0.02)
        assert not app.requested() and lease.gave_back == "stopped"
    finally:
        lease.close()


def test_telemetry_says_who_has_the_camera() -> None:
    clock = Clock()
    t = Telemetry(clock=clock)
    t.on_event("camera.handover", {"released": True, "by_label": "Settings (Gesture Studio)"})
    assert t.camera_in_use_by == "Settings (Gesture Studio)"
    h = t.health()
    assert h["camera_in_use_by"] == "Settings (Gesture Studio)"
    assert h["fps_note"].startswith("Settings (Gesture Studio) is using the camera")
    assert t.context()["camera_in_use_by"] == "Settings (Gesture Studio)"
    t.on_event("camera.handover", {"released": False})
    assert t.camera_in_use_by is None
    assert "no camera picture yet" in (t.fps_note() or "")
    t.on_event("camera.error", {"error": "JoyCam unplugged"})
    assert t.fps_note() == "the camera has a problem: JoyCam unplugged"
    for i in range(10):
        t.on_frame("0", 0.0, now=clock.t + i * 0.1)
    clock.t += 1.0
    assert t.fps_note() is None  # delivering


# ===================================================================================== 5. JoyCam modes
from away.multicam import capture  # noqa: E402
from away.multicam.capture import CaptureMode  # noqa: E402

DSHOW, MSMF = 700, 1400
FOURCC, W, H, FPS = 6, 3, 4, 5
JOYCAM_DEVS = [("JoyCam", r"\\?\usb#vid_aaaa&pid_0001"), ("HD User Facing", r"\\?\usb#vid_0408&pid_a061")]


class JoyCap:
    """The JoyCam driver: MJPG is accepted only when the size was set before the FOURCC on a fresh capture
    (``size_first``) or never (``mjpg=False``); otherwise it keeps YUY2. ``rate(fourcc, w, h)`` gives the fps."""

    def __init__(self, cv2: JoyCV2, src: Any, api: Any) -> None:
        self.cv2, self.src, self.api = cv2, src, api
        self.order: list[int] = []
        self.props: dict[int, Any] = {}
        self.fourcc = "YUY2"
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: Any) -> bool:
        self.order.append(prop)
        self.props[prop] = value
        if prop == FOURCC:
            want = capture.decode_fourcc(value)
            size_before = self.order[: len(self.order) - 1][:1] == [W]
            if want == "MJPG":
                self.fourcc = "MJPG" if (self.cv2.mjpg and size_before) else self.fourcc
            elif want:
                self.fourcc = "YUY2"
        return True

    def get(self, prop: int) -> Any:
        if prop == FOURCC:
            return float(capture.fourcc_code(self.fourcc))
        if prop == W:
            return float(self.props.get(W) or 640)
        if prop == H:
            return float(self.props.get(H) or 480)
        if prop == FPS:
            return float(self.props.get(FPS) or 30)
        return 0.0

    def read(self) -> tuple[bool, Any]:
        if self.released:
            return False, None
        r = self.cv2.rate(self.fourcc, int(self.get(W)), int(self.get(H)))
        self.cv2.clock.t += 1.0 / r
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        self.released = True


class JoyCV2:
    CAP_DSHOW, CAP_MSMF = DSHOW, MSMF
    CAP_PROP_FOURCC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = FOURCC, W, H, FPS
    CAP_PROP_EXPOSURE, CAP_PROP_AUTO_EXPOSURE, CAP_PROP_BUFFERSIZE = 15, 21, 38

    def __init__(self, clock: Clock, mjpg: bool, rate: Any) -> None:
        self.clock, self.mjpg, self.rate = clock, mjpg, rate
        self.caps: list[JoyCap] = []

    def VideoCapture(self, src: Any, api: Any = None) -> JoyCap:
        cap = JoyCap(self, src, api)
        self.caps.append(cap)
        return cap


def yuy2_slow_at_720p(fcc: str, w: int, h: int) -> float:
    if fcc == "MJPG":
        return 30.0
    return 12.0 if w * h > 640 * 480 else 30.0  # works (above the 10 fps minimum) but far below 24


@pytest.fixture
def _windows_cams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setattr(capture, "_windows", lambda: True)
    capture.reset_session_modes()
    capture._open_now.clear()
    capture.use_mode_store(tmp_path / "camera-modes.json")
    yield tmp_path / "camera-modes.json"
    capture.reset_session_modes()
    capture._open_now.clear()
    capture._abort_opens.clear()


def _open_joycam(cv2: JoyCV2, clock: Clock) -> capture.OpenedCamera:
    return capture.open_camera(
        0,
        1280,
        720,
        30,
        cv2=cv2,
        devices=capture.dshow_devices(JOYCAM_DEVS),
        msmf_lister=lambda: list(JOYCAM_DEVS),
        clock=clock,
        sleep=clock.sleep,
        min_fps=10.0,
        measure_s=1.5,
        first_frame_s=4.0,
        retry_frame_s=1.5,
    )


def test_joycam_mjpg_accepted_when_the_size_is_set_first(_windows_cams: Path, caplog: Any) -> None:
    clock = Clock()
    cv2 = JoyCV2(clock, mjpg=True, rate=yuy2_slow_at_720p)
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = _open_joycam(cv2, clock)
    assert got.mode == CaptureMode(1280, 720, 30, "MJPG") and got.backend == "DirectShow"
    assert got.fps is not None and got.fps >= capture.TRACKING_FPS
    assert any("retried MJPG with the size set first -> driver reports MJPG" in r.getMessage() for r in caplog.records)
    assert len(cv2.caps) == 2 and cv2.caps[0].released  # the first capture kept YUY2 and was reopened
    got.cap.release()


def test_joycam_prefers_a_smaller_mode_that_reaches_24fps_over_slow_720p_yuy2(_windows_cams: Path, caplog) -> None:
    clock = Clock()
    cv2 = JoyCV2(clock, mjpg=False, rate=yuy2_slow_at_720p)
    with caplog.at_level(logging.INFO, logger="away.cameras"):
        got = _open_joycam(cv2, clock)
    assert got.mode.fourcc != "MJPG" and got.mode.width * got.mode.height < 1280 * 720
    assert got.fps is not None and got.fps >= capture.TRACKING_FPS
    msgs = [r.getMessage() for r in caplog.records]
    assert any("will not send MJPG" in m and "driver keeps YUY2" in m for m in msgs)
    assert any("below 24 fps for smooth tracking" in m for m in msgs)
    assert sum(1 for c in cv2.caps if not c.released) == 1
    got.cap.release()


def test_joycam_uses_and_remembers_the_fastest_working_mode_when_none_reaches_24(_windows_cams: Path) -> None:
    clock = Clock()
    cv2 = JoyCV2(clock, mjpg=False, rate=lambda fcc, w, h: 12.0)
    got = _open_joycam(cv2, clock)
    assert got.fps is not None and 11.0 <= got.fps <= 13.0 and not got.slow
    saved = json.loads(_windows_cams.read_text())["JoyCam"]
    assert saved.get("best") is True
    got.cap.release()
    capture.reset_session_modes()
    capture.use_mode_store(_windows_cams)
    cv2.caps.clear()
    again = _open_joycam(cv2, clock)
    assert len(cv2.caps) == 1 and again.mode == got.mode  # the remembered best mode is trusted at once
    again.cap.release()


def test_speed_modes_only_for_large_sizes() -> None:
    assert capture.speed_modes(640, 480, 30) == []
    modes = capture.speed_modes(1280, 720, 30)
    assert [(m.width, m.height, m.fourcc) for m in modes][:2] == [(960, 540, "MJPG"), (960, 540, "YUY2")]
    assert {(m.width, m.height) for m in modes} == {(960, 540), (640, 360), (640, 480)}


# ===================================================================================== 6. weak speech
WEAK = {
    "allowed": True,
    "reason": "enrolled face at the screen, facing it (mouth motion weak)",
    "frames": 14,
    "mouth_moving_frames": 0,
    "speaker_frames": 0,
}
STRONG = {
    "allowed": True,
    "reason": "enrolled face speaking",
    "frames": 14,
    "mouth_moving_frames": 9,
    "speaker_frames": 7,
}


def test_the_exact_log_cases_are_dropped() -> None:
    # strict_ratio=0: the v0.1.21 rules on their own (v0.1.22 adds the barely-moving-mouth rule, tested elsewhere)
    assert should_drop("uh", WEAK, 0.9, strict_ratio=0) == "only a filler sound"
    assert should_drop("Uh.", STRONG, 0.99) == "only a filler sound"  # filler is never typed
    why = should_drop("Dikontak", WEAK, None, strict_ratio=0)
    assert "mouth not seen moving" in why and "unknown" in why
    assert "need 0.75" in should_drop("Dikontak", WEAK, 0.41, strict_ratio=0)


def test_real_speech_is_still_typed() -> None:
    assert should_drop("Dikontak", STRONG, None) == ""  # the mouth moved: trust the gate
    assert should_drop("Dikontak", WEAK, 0.9, strict_ratio=0) == ""  # weak evidence but a confident recognition
    assert should_drop("please send the report to Maria today", WEAK, None, strict_ratio=0) == ""
    assert "need 0.45" in should_drop("please send the report to Maria today", WEAK, 0.3, strict_ratio=0)
    assert should_drop("hello there", None, None) == ""  # no gate: unchanged


def test_filler_and_evidence_rules() -> None:
    assert is_filler("uh, hmm...") and is_filler("Mm-hmm") and is_filler("UM") and not is_filler("um, yes")
    assert not is_filler("") and not is_filler("...")
    assert weak_evidence(WEAK) and not weak_evidence(STRONG)
    assert weak_evidence({"allowed": True, "mouth_moving_frames": 0, "speaker_frames": 0})
    assert not weak_evidence({"allowed": False, "reason": "mouth motion weak"})
    assert not weak_evidence(None)
    assert weak_speech.words("Don't stop-now!") == ["don't", "stop-now"]


def test_confidence_from_whisper_segments() -> None:
    seg = types.SimpleNamespace
    assert confidence_of([]) is None and confidence_of([seg(text="x")]) is None
    assert confidence_of([seg(avg_logprob=0.0, no_speech_prob=0.0, text="hi")]) == 1.0
    c = confidence_of([seg(avg_logprob=-0.1, no_speech_prob=0.0, text="hello"), seg(avg_logprob=-2.0, text="x")])
    assert c is not None and 0.75 < c < 0.9
    assert confidence_of([seg(avg_logprob=0.0, no_speech_prob=1.0, text="hi")]) == 0.5


def _ambient(outputs: list[str], confidences: list[float | None], ran: list[Any], sink: Any, got: list[Any]) -> Any:
    from away.bindings import BindingsConfig
    from away.events import EventBus
    from away.voice.always_on import AmbientSession, StreamingEndpointer
    from away.voice.asr import FakeASR

    class Clean:
        def clean(self, text: str, context: Any = None) -> str:
            return text

    bus = EventBus()
    bus.subscribe("voice.utterance", lambda e: got.append(e.data))
    return AmbientSession(
        FakeASR(outputs, confidences),
        Clean(),
        sink,
        StreamingEndpointer(16000),
        commands=lambda: BindingsConfig().voice_phrases(),
        run_command=lambda a, p: ran.append((a, p)),
        bus=bus,
        gate=lambda t0, t1: True,
        gate_info=lambda: dict(WEAK),
    )


def test_always_on_drops_uh_and_dikontak_but_runs_short_commands() -> None:
    from away.voice.always_on import Utterance
    from away.voice.output import ListSink

    sink, ran, got = ListSink(), [], []
    s = _ambient(["uh", "Dikontak", "scroll down", "Dikontak"], [0.95, 0.41, 0.3, 0.92], ran, sink, got)
    audio = np.zeros(16000, np.float32)
    r1 = s.process(Utterance(audio, 0.0, 1.0))
    r2 = s.process(Utterance(audio, 2.0, 3.0))
    r3 = s.process(Utterance(audio, 4.0, 5.0))
    r4 = s.process(Utterance(audio, 6.0, 7.0))
    assert r1.decision.kind == "ignore" and "filler" in r1.decision.reason
    assert r2.decision.kind == "ignore" and "mouth" in r2.decision.reason  # v0.1.22: "mouth barely moved"
    assert r3.decision.kind == "command" and ran and ran[0][0].arg == "scroll_down"  # a command, unsure or not
    assert r4.decision.kind == "type"  # a confident recognition is typed
    assert sink.items and "Dikontak" in sink.items[-1] and "uh" not in " ".join(sink.items).lower().split()
    assert got[1]["confidence"] == 0.41 and got[1]["gate"]["mouth_moving_frames"] == 0


# ===================================================================================== 7. get_screen
def _displays() -> list[screenshot.Display]:
    return screenshot.displays(HIS_DESKTOP)


def test_his_desktop_bounds_and_wide_shot() -> None:
    ds = _displays()
    assert [d.index for d in ds] == [1, 2, 3] and ds[1].primary
    assert screenshot.desktop_bounds(ds) == (-1920, 0, 5206, 1080)
    p = screenshot.plan(ds, (5206, 1080))  # 0.5 of 5206 = 2603 px: just over the 2560 px cap
    assert p.what == "desktop" and p.requested_scale == 0.5 and p.capped and p.out == (2560, 531)
    assert p.scale == pytest.approx(2560 / 5206, abs=1e-6) and p.origin == (-1920, 0) and p.crop == (0, 0, 5206, 1080)
    assert p.to_screen(0, 0) == (-1920, 0)
    assert p.to_screen(1280, 265.5) == pytest.approx((683.0, 540.0), abs=0.5)  # the image centre (sub-pixel)
    assert p.to_image(*p.to_screen(100, 200)) == pytest.approx((100, 200))
    small = screenshot.plan(ds, (5206, 1080), scale=0.3)  # under the cap: exactly the requested scale
    assert small.scale == 0.3 and small.out == (1562, 324) and not small.capped
    assert small.to_screen(1562, 324) == pytest.approx((3286.67, 1080.0), abs=0.01)


def test_zoom_into_one_monitor_of_three() -> None:
    ds = _displays()
    left = screenshot.plan(ds, (5206, 1080), monitor=1, scale=1.0)
    assert left.crop == (0, 0, 1920, 1080) and left.origin == (-1920, 0) and left.out == (1920, 1080)
    assert left.to_screen(960, 540) == (-960.0, 540.0)
    right = screenshot.plan(ds, (5206, 1080), monitor="3", scale=0.5)
    assert right.what == "monitor 3" and right.crop == (3840, 0, 5206, 768) and right.out == (683, 384)
    assert right.to_screen(683, 384) == (3286.0, 768.0)


def test_region_with_negative_coordinates_and_clipping() -> None:
    ds = _displays()
    p = screenshot.plan(ds, (5206, 1080), region={"x": -500, "y": 100, "width": 400, "height": 300}, scale=1)
    assert p.what == "region" and p.crop == (1420, 100, 1820, 400) and p.to_screen(0, 0) == (-500, 100)
    clipped = screenshot.plan(ds, (5206, 1080), region="-2500,-50,1000,200", scale=0.5)
    assert clipped.origin == (-1920, 0) and clipped.area == (420, 150) and clipped.out == (210, 75)
    with pytest.raises(screenshot.ScreenRequestError, match="outside the screen"):
        screenshot.plan(ds, (5206, 1080), region=[4000, 0, 100, 100])


def test_negative_offset_layout_above_and_left() -> None:
    mons = [
        types.SimpleNamespace(left=-1280, top=-1024, width=1280, height=1024, primary=False),
        types.SimpleNamespace(left=0, top=0, width=2560, height=1440, primary=True),
    ]
    ds = screenshot.displays(mons)
    assert screenshot.desktop_bounds(ds) == (-1280, -1024, 3840, 2464)
    p = screenshot.plan(ds, (3840, 2464), monitor=2, scale=1.0)
    assert p.crop == (1280, 1024, 3840, 2464) and p.to_screen(10, 20) == (10, 20)
    q = screenshot.plan(ds, (3840, 2464), monitor=1, scale=0.5)
    assert q.crop == (0, 0, 1280, 1024) and q.to_screen(320, 256) == (-640.0, -512.0)


def test_scale_cap_dpi_mismatch_and_bad_requests() -> None:
    ds = _displays()
    big = screenshot.plan(ds, (5206, 1080), scale=1.0)
    assert big.capped and big.out[0] == 2560 and big.scale == pytest.approx(2560 / 5206, abs=1e-6)
    assert big.to_screen(2560, 0)[0] == pytest.approx(3286, abs=0.01)
    half = screenshot.plan(ds, (2603, 540), monitor=2, scale=1.0)  # the capture came back at half size
    assert half.crop == (960, 0, 1920, 540) and half.origin == (0, 0)
    for kw in ({"scale": 0.05}, {"scale": 1.5}, {"scale": True}, {"scale": "big"}, {"monitor": 4}, {"monitor": "x"}):
        with pytest.raises(screenshot.ScreenRequestError):
            screenshot.plan(ds, (5206, 1080), **kw)
    with pytest.raises(screenshot.ScreenRequestError, match="not both"):
        screenshot.plan(ds, (5206, 1080), monitor=1, region="0,0,10,10")
    assert screenshot.parse_region('{"x": 1, "y": 2, "w": 3, "h": 4}') == (1, 2, 3, 4)
    assert screenshot.parse_region("[1, 2, 3, 4]") == (1, 2, 3, 4)
    assert screenshot.parse_region(None) is None
    with pytest.raises(screenshot.ScreenRequestError):
        screenshot.parse_region({"x": 0, "y": 0, "width": 0, "height": 5})


def test_render_and_meta() -> None:
    from PIL import Image

    ds = _displays()
    p = screenshot.plan(ds, (5206, 1080), monitor=3, scale=0.5)
    jpeg = screenshot.render(Image.new("RGB", (5206, 1080), (20, 40, 60)), p)
    assert Image.open(io.BytesIO(jpeg)).size == (683, 384)
    arr = screenshot.render(np.zeros((1080, 5206, 3), np.uint8), p)
    assert arr[:2] == b"\xff\xd8"
    m = screenshot.meta(p, ds, len(jpeg))
    assert m["showing"] == "monitor 3" and m["desktop"] == {"x": -1920, "y": 0, "width": 5206, "height": 1080}
    assert len(m["monitors"]) == 3 and m["monitors"][2] == {
        "index": 3,
        "x": 1920,
        "y": 0,
        "w": 1366,
        "h": 768,
        "primary": False,
    }
    assert screenshot.jpeg_quality((683, 384), True) == 80 and screenshot.jpeg_quality((2603, 540), False) == 70


def test_service_get_screen_lists_monitors_and_maps_from_image(core) -> None:
    from PIL import Image

    core.service.screen_grab = lambda: Image.new("RGB", (5206, 1080))
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    jpeg, meta = core.service.get_screen(a)
    assert meta["showing"] == "desktop" and meta["image"]["width"] == 2560 and len(meta["monitors"]) == 3
    assert meta["scale_capped"] is True and meta["desktop"]["width"] == 5206
    assert "wide shot" in meta["note"]
    _, meta = core.service.get_screen(a, monitor=3, scale=1.0)
    assert meta["origin"] == {"x": 1920, "y": 0} and meta["image"]["width"] == 1366
    assert core.service.control.views[a.id] == (1920.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="no monitor 9"):
        core.service.get_screen(a, monitor=9)


def test_from_image_uses_the_zoomed_view_origin() -> None:
    from away.agentlink.control import ComputerControl, ControlHooks
    from away.control.mouse import RecordingBackend

    backend = RecordingBackend()
    hooks = ControlHooks(
        backend=backend,
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=lambda _s: None,
    )
    ctl = ComputerControl(hooks, lambda q, o, **k: {"answer": "Allow", "method": "click"}, clock=Clock())
    ctl.request("a1", "Claude")
    ctl.views["a1"] = (-1920.0, 0.0, 0.5)
    ctl.act("a1", "move", x=100, y=50, from_image=True)
    assert backend.calls[-1] == ("move", (-1720, 100))


# ===================================================================================== 8. grouped tools
def test_groups_resolve_to_the_old_methods() -> None:
    assert groups.resolve("settings", {"action": "change", "changes": [{"key": "a"}], "why": "x", "junk": 1}) == (
        "change_settings",
        {"ops": [{"key": "a"}], "why": "x"},
    )
    assert groups.resolve("pointer", {"action": "click", "x": 5, "y": 6, "element": None}) == (
        "act",
        {"action": "click", "x": 5, "y": 6},
    )
    assert groups.resolve("diagnostics", {"action": "job", "job_id": "82557239"}) == (
        "get_job",
        {"job_id": "82557239"},
    )
    assert groups.resolve("open", {"action": "url", "url": "https://x"}) == (
        "act",
        {"action": "open_url", "url": "https://x"},
    )
    with pytest.raises(ValueError, match="action must be one of"):
        groups.resolve("pointer", {"action": "explode"})
    with pytest.raises(KeyError):
        groups.resolve("nope", {"action": "x"})
    assert "settings(action=list|change|undo|history|config)" in groups.describe()


def test_every_group_action_names_a_real_service_method() -> None:
    from away.agentlink import api
    from away.agentlink.service import Service

    for g, acts in groups.GROUPS.items():
        for name, ga in acts.items():
            assert ga.method in api.TOOLS, (g, name)
            assert callable(getattr(Service, ga.method)), (g, name)


def test_old_names_and_groups_both_dispatch(core) -> None:
    from away.agentlink import api

    a, _ = core.registry.issue("Agent", ["read", "diagnose", "tune"], 3600)
    _done_job(core, "82557239")
    assert api.dispatch(core.service, a, "get_job", {"job_id": "82557239"})["id"] == "82557239"
    assert api.dispatch(core.service, a, "diagnostics", {"action": "job", "job_id": 82557239})["id"] == "82557239"
    assert api.dispatch(core.service, a, "preferences", {"action": "list"}) == api.dispatch(
        core.service, a, "get_preferences", {}
    )
    assert api.kind_of("get_screen", {}) == "image" and api.kind_of("diagnostics", {"action": "download"}) == "bytes"
    with pytest.raises(KeyError):
        api.dispatch(core.service, a, "rm", {})
    enum = api.openapi()["paths"]["/v1/rpc/{tool}"]["post"]["parameters"][0]["schema"]["enum"]
    assert "diagnostics" in enum and "change_settings" in enum


def test_mcp_lists_exactly_15_grouped_tools(core) -> None:
    pytest.importorskip("mcp")
    from away.agentlink import api, mcp_tools

    mcp = mcp_tools.build(api.LocalBackend(core.service))
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == GROUPED_TOOLS


def test_about_explains_wide_shot_then_zoom_and_the_grouped_tools(core) -> None:
    for name in GROUPED_TOOLS:
        assert name in INSTRUCTIONS
    assert "Tools (16)" in INSTRUCTIONS and "action" in INSTRUCTIONS
    assert "wide shot first" in INSTRUCTIONS.lower() and "monitor=N" in INSTRUCTIONS and "region=" in INSTRUCTIONS
    a, _ = core.registry.issue("Agent", ["read"], 3600)
    text = briefing(a, core.service.get_context(a), {"active": False}, "ask")
    assert "orient, then zoom" in text and "monitor=N" in text and "Tools" in text


# ===================================================================================== over HTTP
@pytest.fixture
def live(core):
    pytest.importorskip("mcp")
    pytest.importorskip("uvicorn")
    from away.agentlink import api, mcp_tools
    from away.agentlink.server import Server

    gate = api.Gate(core.service, core.registry, local_key="k" * 43, rate=(50.0, 60))
    srv = Server(api.create_app(core.service, gate, mcp_tools.build(api.LocalBackend(core.service))))
    port = srv.start()
    yield f"http://127.0.0.1:{port}"
    srv.stop()


def _http(base: str, method: str, path: str, token: str, body: Any = None) -> tuple[int, Any]:
    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method=method,
    )
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def test_job_82557239_round_trips_over_http(core, live, monkeypatch) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    _, tok = core.registry.issue("Script", ["read", "diagnose"], 3600)
    _done_job(core, "82557239")
    st, body = _http(live, "GET", "/v1/jobs?job_id=82557239", tok)
    assert st == 200 and body["id"] == "82557239" and body["output"] == ["JoyCam ok"]
    st, body = _http(live, "GET", "/v1/jobs?id=82557239", tok)
    assert st == 200 and body["id"] == "82557239"
    st, body = _http(live, "POST", "/v1/rpc/diagnostics", tok, {"action": "job", "job_id": 82557239})
    assert st == 200 and body["id"] == "82557239"
    st, body = _http(live, "POST", "/v1/rpc/get_job", tok, {"id": "82557239"})
    assert st == 200 and body["id"] == "82557239"
    st, body = _http(live, "GET", "/v1/jobs?job_id=12345678", tok)
    assert st == 404 and "unknown check 12345678" in body["error"]
    core.service.jobs._expired["87654321"] = 0.0
    st, body = _http(live, "GET", "/v1/jobs?job_id=87654321", tok)
    assert st == 410 and "expired" in body["error"]


# ===================================================================================== more coverage of the same fixes
ACK_SCRIPT = (
    "import json, sys; spec = json.load(open(sys.argv[1], encoding='utf-8')); "
    "open(spec['ack'], 'w', encoding='utf-8').write(json.dumps({'shown': True, 'title': spec['title']}))"
)


def test_message_window_process_acknowledges_and_is_reported_shown(tmp_path: Path) -> None:
    ui = prompts.SubprocessUI(tmp_path / "ui", command=lambda args: [sys.executable, "-c", ACK_SCRIPT, args[-1]])
    ui.ack_s = 30.0
    assert ui.toast("Build finished", title="Message from Claude") == {"shown": True, "via": "window"}
    assert not list((tmp_path / "ui").glob("toast-*.json"))  # spec and ack cleaned up


def test_message_window_that_exits_or_cannot_start_is_reported_not_shown(tmp_path: Path) -> None:
    ui = prompts.SubprocessUI(tmp_path / "ui", command=lambda args: [sys.executable, "-c", "raise SystemExit(2)"])
    ui.ack_s = 30.0
    res = ui.toast("hi")
    assert res["shown"] is False and "exit code 2" in res["reason"]
    missing = prompts.SubprocessUI(tmp_path / "ui2", command=lambda args: [str(tmp_path / "no-such-away.exe")])
    res = missing.toast("hi")
    assert res["shown"] is False and res["reason"].startswith("could not start the message window")


def test_service_notify_user_and_ask_user_report_speech(core) -> None:
    eng = FakeEngine()
    core.prompts.tts = speech.Speaker(lambda: _cfg(), [("fake", lambda: eng)])
    a, _ = core.registry.issue("Claude", ["read"], 3600)
    res = core.service.notify_user(a, "Tests passed", speak=True)
    assert res["shown"] is True and res["spoken"] is True and core.prompts.tts.wait(5)
    assert eng.said[-1][0] == "Tests passed" and core.prompts.ui.titles[-1] == "Message from Claude"
    core.prompts.tts = speech.Speaker(lambda: _cfg(enabled=False), [("fake", lambda: eng)])
    assert core.service._speech_allowed() is False


def test_speaker_drops_the_oldest_waiting_messages() -> None:
    gate = threading.Event()

    class Slow(FakeEngine):
        def speak(self, text: str, v: speech.Voice) -> None:
            gate.wait(5)
            super().speak(text, v)

    eng = Slow()
    sp = speech.Speaker(lambda: None, [("slow", lambda: eng)])
    sp.say("first")
    time.sleep(0.2)  # the worker is now speaking "first"
    for i in range(speech.MAX_QUEUE + 3):
        sp.say(f"m{i}")
    gate.set()
    assert sp.wait(10)
    said = [t for t, _ in eng.said]
    assert said[0] == "first" and len(said) == 1 + speech.MAX_QUEUE and said[-1] == f"m{speech.MAX_QUEUE + 2}"


def test_weak_speech_thresholds_are_adjustable() -> None:
    assert should_drop("open mail", WEAK, 0.6, min_confidence=0.5, strict_ratio=0) == ""
    assert should_drop("open mail", WEAK, 0.6, strict_ratio=0) != ""
    assert should_drop("open the mail app", WEAK, 0.6, max_words=4, strict_ratio=0) != ""
    assert should_drop("open the mail app", WEAK, 0.6, max_words=4, min_confidence=0.55, strict_ratio=0) == ""


def test_live_view_keeps_its_zoom_and_smaller_cap(core) -> None:
    from PIL import Image

    core.service.screen_grab = lambda: Image.new("RGB", (5206, 1080))
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    res = core.service.screen_watch(a, True, scale=0.8, monitor=2)
    assert res["watching"] and res["view"] == {"scale": 0.8, "monitor": 2}
    _, meta = core.service.get_screen(a, quiet=True)
    assert meta["showing"] == "monitor 2" and meta["image"]["width"] == 1536
    core.service.screen_watch(a, True)  # no options: the live view keeps what it had
    _, meta = core.service.get_screen(a, quiet=True)
    assert meta["showing"] == "monitor 2"
    core.service.screen_watch(a, False)
    _, meta = core.service.get_screen(a, quiet=True)
    assert meta["showing"] == "desktop" and max(meta["image"]["width"], meta["image"]["height"]) <= 1600


def test_size_first_sets_size_before_and_after_the_fourcc() -> None:
    clock = Clock()
    cv2 = JoyCV2(clock, mjpg=True, rate=yuy2_slow_at_720p)
    cap = cv2.VideoCapture(0, DSHOW)
    got = capture.apply_mode_size_first(cap, CaptureMode(1280, 720, 30, "MJPG"), cv2, "JoyCam")
    assert got == "MJPG" and cap.order[:6] == [W, H, FOURCC, W, H, FPS]
    plain = cv2.VideoCapture(0, DSHOW)
    assert capture.apply_mode(plain, CaptureMode(1280, 720, 30, "MJPG"), cv2, "JoyCam") == "YUY2"


def test_too_many_finished_jobs_become_expired_not_unknown(tmp_path: Path) -> None:
    clock = Clock()
    r = _runner(tmp_path, clock)
    for i in range(jobs.MAX_JOBS + 5):
        jid = f"{i:08d}"
        r.jobs[jid] = jobs.Job(jid, "benchmark", clock(), state="done", finished=clock())
    r.get("00000000")  # any read prunes
    assert len(r.jobs) == jobs.MAX_JOBS
    with pytest.raises(jobs.JobLookupError) as e:
        r.lookup("00000001")
    assert e.value.reason == "expired"


def test_grouped_mcp_tool_calls_reach_the_old_methods() -> None:
    pytest.importorskip("mcp")
    from away.agentlink import mcp_tools

    calls: list[tuple[str, dict[str, Any]]] = []

    class Recorder:
        def call(self, ctx: Any, method: str, **kw: Any) -> Any:
            calls.append((method, kw))
            return {"ok": True}

    mcp = mcp_tools.build(Recorder())
    asyncio.run(mcp.call_tool("pointer", {"action": "click", "x": 5, "y": 6}))
    asyncio.run(mcp.call_tool("diagnostics", {"action": "job", "job_id": "82557239"}))
    asyncio.run(mcp.call_tool("settings", {"action": "change", "changes": [{"key": "k", "value": 1}], "why": "w"}))
    assert calls[0] == ("act", {"action": "click", "x": 5, "y": 6})
    assert calls[1] == ("get_job", {"job_id": "82557239"})
    assert calls[2] == ("change_settings", {"ops": [{"key": "k", "value": 1}], "why": "w"})
