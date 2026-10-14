"""v0.1.25 Check my PC: each part's Pass / Fail / Skipped rule with fake devices (JoyCam plus a broken internal
camera, voices, tray, toast, secure desktop, pynput echo, link reconnect), the guided prompts, redaction (both
redactors), the saved JSON / HTML / text / verified.json, sharing only with agents allowed to receive it, the
in-app probe list, RemotePcEnv and the ``away check`` CLI."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from away.selftest import checkpc, pccheck, pcenv
from away.selftest import checks as ck
from away.selftest.model import FAIL, PASS, SKIP, WARN, Fix, Result


class FakePc:
    """A PC where everything works, unless a test changes one answer. Records prompts and questions."""

    def __init__(self, answers: Any = True, **over: Any) -> None:
        self.prompts: list[str] = []
        self.questions: list[str] = []
        self.answers = answers  # bool / None / callable(question)
        self.spoken: list[str] = []
        self.data: dict[str, Any] = {
            "cameras": [
                {"name": "JoyCam", "opened": True, "fps": 24.0, "format": "640x480@24 MJPG", "backend": "DirectShow"},
                {"name": "Integrated Camera", "opened": False, "fps": 0.0, "error": "no frames"},
            ],
            "camera_users": {"message": "Integrated Camera could not be opened.", "users": [], "actions": ["retry"]},
            "tracking": {"fps": 24.0, "latency_ms": 18.0, "backend": "graphics card", "source": "live"},
            "gpu_tracking": {"setting": "auto", "using": "graphics card", "provider": "DirectML", "models_ready": True},
            "glide": {"hz": 118.0, "target_hz": 120},
            "big_cursor": {"supported": True, "size": 32, "user_size": 32, "restored": True},
            "instances": {"lock_held": True, "count": 1, "pids": [100]},
            "mic": {"device": "Mic (USB)", "rms_db": -35.0, "peak_db": -12.0},
            "wait_gesture": {"seen": True, "after_ms": 800, "fps": 24.0, "latency_ms": 18.0},
            "dictation": {"heard": "Testing, one two three.", "method": "spoken", "ms": 1200},
            "wake_settings": {"word": "away", "enabled": True},
            "wake_word": {"heard": "Away, what time is it?", "matched": True},
            "voice_list": ["Natural (Kokoro)", "Light (Piper)", "Windows natural", "Windows voice (SAPI)"],
            "tray": {"visible": True, "tooltip": "Project Away - on"},
            "toast": {"shown": True, "via": "window"},
            "notification": {"shown": True, "via": "window"},
            "secure_desktop": {"active": False},
            "input_echo": {
                "mouse_ms": [8.0, 12.0, 9.5],
                "key_ms": [15.0, 20.0, 11.0],
                "mouse_window_s": 0.25,
                "key_window_s": 0.3,
            },
            "link": {
                "enabled": True,
                "server": {"running": True, "port": 5000, "ok": True, "ms": 3.0},
                "tunnel": {"ok": True, "ms": 240.0},
                "relays": [],
            },
            "link_reconnect": {"tries": 3, "ok": 3, "ms": 250.0, "reconnectable": 2},
        }
        self.speak_rows: dict[str, Any] = {}
        self.data.update(over)

    def prompt(self, text: str) -> None:
        self.prompts.append(text)

    def confirm(self, question: str) -> bool | None:
        self.questions.append(question)
        return self.answers(question) if callable(self.answers) else self.answers

    def speak(self, name: str) -> dict[str, Any]:
        self.spoken.append(name)
        row = self.speak_rows.get(name, {"name": name, "first_audio_ms": 400})
        return dict(row)

    def __getattr__(self, name: str) -> Any:
        if name not in self.data:
            raise AttributeError(name)
        val = self.data[name]

        def fn(*_a: Any) -> Any:
            if isinstance(val, Exception):
                raise val
            return val

        return fn


def by_id(results: list[Result]) -> dict[str, Result]:
    return {r.id: r for r in results}


# ------------------------------------------------------------------------------------------------ the run
def test_a_healthy_pc_passes_and_covers_every_part() -> None:
    env = FakePc(cameras=[{"name": "JoyCam", "opened": True, "fps": 24.0}])
    rep = pccheck.run(env, trigger="tray")
    pub = pccheck.with_verdicts(rep.public())
    assert pub["kind"] == "away.pccheck"
    assert pub["verdicts"]["fail"] == 0, pccheck.text(pub)
    ids = {r["id"] for r in pub["results"]}
    for want in (
        "camera:JoyCam",
        "tracking",
        "gpu",
        "gesture:hand",
        "gesture:pinch",
        "mic",
        "dictation",
        "wake",
        "voice:kokoro",
        "voice:piper",
        "voice:winrt",
        "voice:sapi",
        "tray",
        "big_cursor",
        "single",
        "glide",
        "toast",
        "notify",
        "secure_desktop",
        "echo:mouse",
        "echo:key",
        "link",
        "link:tunnel",
        "link:reconnect",
    ):
        assert want in ids, want
    assert {r["verdict"] for r in pub["results"]} <= {"pass", "fail", "skipped"}
    joined = " ".join(env.prompts)  # the guided steps, in plain words
    assert "Raise one hand" in joined and "pinch" in joined and "testing one two three" in joined
    assert "away" in joined
    assert any("hear" in q for q in env.questions)


def test_only_runs_the_named_parts_and_a_crashing_part_becomes_one_line() -> None:
    env = FakePc(secure_desktop=RuntimeError("boom"))
    rep = pccheck.run(env, only=["secure_desktop", "glide"])
    r = by_id(rep.results)
    assert set(r) == {"secure_desktop", "glide"}
    assert r["secure_desktop"].status == FAIL and "could not run" in r["secure_desktop"].summary


# ------------------------------------------------------------------------------------------------ cameras
def test_cameras_joycam_passes_and_the_broken_internal_camera_gets_a_reason_and_fix() -> None:
    r = by_id(pccheck.check_cameras(FakePc()))
    joy, internal = r["camera:JoyCam"], r["camera:Integrated Camera"]
    assert joy.status == PASS and "Nobody else is using it" in joy.summary and joy.details["held_by"] == []
    assert internal.status == FAIL
    assert "built-in camera" in internal.summary and "Device Manager" in internal.summary
    assert internal.fix is not None and internal.fix.action == "open:camera_privacy"
    assert internal.details["internal_suspect"] is True


def test_a_camera_held_by_another_app_names_it_and_offers_to_close_it() -> None:
    env = FakePc(
        camera_users={
            "message": "Zoom is using Integrated Camera.",
            "users": [{"name": "Zoom", "pid": 5}],
            "actions": ["close:C:/Zoom/zoom.exe"],
        }
    )
    r = by_id(pccheck.check_cameras(env))["camera:Integrated Camera"]
    assert r.status == FAIL and "Zoom" in r.summary and "built-in" not in r.summary
    assert r.fix is not None and r.fix.label == "Close Zoom" and r.details["held_by"] == ["Zoom"]


def test_a_working_camera_also_open_elsewhere_says_so() -> None:
    env = FakePc(
        cameras=[{"name": "JoyCam", "opened": True, "fps": 24.0}],
        camera_users={"users": [{"name": "OBS"}], "actions": []},
    )
    r = pccheck.check_cameras(env)[0]
    assert r.status == PASS and "Also open in: OBS" in r.summary


def test_a_camera_turned_off_is_skipped() -> None:
    env = FakePc(cameras=[{"name": "JoyCam", "off": True}])
    assert pccheck.check_cameras(env)[0].status == SKIP


# ------------------------------------------------------------------------------------------- gestures
@pytest.mark.parametrize(
    ("answer", "status"),
    [
        ({"seen": True, "after_ms": 600}, PASS),
        ({"seen": False}, FAIL),
        ({"error": "camera busy"}, FAIL),
        ({"skipped": "The hand model is not downloaded yet."}, SKIP),
    ],
)
def test_gesture_rules(answer: dict[str, Any], status: str) -> None:
    for kind in ("hand", "pinch"):
        env = FakePc(wait_gesture=answer)
        r = pccheck.check_gesture(env, kind)[0]
        assert r.status == status
        assert r.id == f"gesture:{kind}" and env.prompts
        if status == FAIL:
            assert r.fix is not None


def test_tracking_reports_speed_and_where_it_runs() -> None:
    r = by_id(pccheck.check_tracking(FakePc()))
    assert r["tracking"].status == PASS and "graphics card" in r["tracking"].summary
    assert r["gpu"].status == PASS and "DirectML" in r["gpu"].summary
    slow = by_id(pccheck.check_tracking(FakePc(tracking={"fps": 24.0, "latency_ms": 95.0, "backend": "processor"})))
    assert slow["tracking"].status == WARN  # a "needs a look" is still a Pass in Check my PC
    assert pccheck.VERDICT[slow["tracking"].status] == "pass"


# -------------------------------------------------------------------------------------- dictation, wake
def test_dictation_rules() -> None:
    assert pccheck.check_dictation(FakePc())[0].status == PASS
    wrong = pccheck.check_dictation(FakePc(dictation={"heard": "hello there", "method": "spoken", "ms": 900}))[0]
    assert wrong.status == FAIL and wrong.fix is not None and wrong.id == "dictation"
    assert pccheck.check_dictation(FakePc(dictation={"skipped": "no model"}))[0].status == SKIP
    assert pccheck.check_dictation(FakePc(dictation={"error": "bad"}))[0].status == FAIL


def test_wake_word_rules() -> None:
    env = FakePc()
    ok = pccheck.check_wake(env)[0]
    assert ok.status == PASS and "away" in env.prompts[0]
    off = pccheck.check_wake(FakePc(wake_settings={"word": "away", "enabled": False}))[0]
    assert off.status == PASS and "turned off" in off.summary
    miss = pccheck.check_wake(FakePc(wake_word={"heard": "a weigh", "matched": False}))[0]
    assert miss.status == FAIL and miss.fix is not None
    none = pccheck.check_wake(FakePc(wake_word={"heard": "", "matched": False}))[0]
    assert none.status == FAIL and "Heard nothing" in none.summary
    assert pccheck.check_wake(FakePc(wake_word={"skipped": "no model"}))[0].status == SKIP
    assert pccheck.check_wake(FakePc(wake_word={"error": "no mic"}))[0].status == FAIL


def test_wake_word_match_is_the_apps_own_matcher() -> None:
    from away.voice.always_on import match_wake

    assert match_wake("Away, what time is it?", "away")[0]
    assert not match_wake("what time is it", "away")[0]


# ---------------------------------------------------------------------------------------------- voices
def test_each_voice_engine_speaks_and_is_confirmed() -> None:
    env = FakePc()
    r = by_id(pccheck.check_voices(env, windows=True))
    assert [r[k].status for k in ("voice:kokoro", "voice:piper", "voice:winrt", "voice:sapi")] == [PASS] * 4
    assert env.spoken == ["Natural (Kokoro)", "Light (Piper)", "Windows natural", "Windows voice (SAPI)"]
    assert len(env.questions) == 4 and all("hear" in q for q in env.questions)


def test_voice_not_heard_fails_no_answer_skips_error_fails() -> None:
    env = FakePc(answers=lambda q: None if "Piper" in q else ("Kokoro" not in q))
    env.speak_rows["Windows voice (SAPI)"] = {"error": "COM error"}
    r = by_id(pccheck.check_voices(env, windows=True))
    assert r["voice:kokoro"].status == FAIL and r["voice:kokoro"].fix is not None
    assert r["voice:piper"].status == SKIP and "nobody answered" in r["voice:piper"].summary
    assert r["voice:winrt"].status == PASS
    assert r["voice:sapi"].status == FAIL and "COM error" in r["voice:sapi"].summary


def test_missing_engines_are_skipped_and_extra_voices_are_checked_too() -> None:
    env = FakePc(voice_list=["say"])
    r = by_id(pccheck.check_voices(env, windows=False))
    assert r["voice:kokoro"].status == SKIP and r["voice:kokoro"].fix is not None
    assert r["voice:winrt"].summary == "Windows only." and r["voice:sapi"].status == SKIP
    assert r["voice:say"].status == PASS
    win = by_id(pccheck.check_voices(FakePc(voice_list=[]), windows=True))
    assert win["voice:sapi"].summary == "Not installed on this PC."


def test_engine_names_map_to_the_four_readme_engines() -> None:
    assert pccheck.engine_of("Natural (Kokoro)") == "Kokoro"
    assert pccheck.engine_of("Light (Piper)") == "Piper"
    assert pccheck.engine_of("Windows natural") == "WinRT"
    assert pccheck.engine_of("Windows voice (SAPI)") == "SAPI"
    assert pccheck.engine_of("Windows voice (PowerShell)") is None


# ------------------------------------------------------------------------------------- tray and toast
def test_tray_rules() -> None:
    assert pccheck.check_tray(FakePc())[0].status == PASS
    hidden = pccheck.check_tray(FakePc(tray={"visible": False, "tooltip": "x"}))[0]
    assert hidden.status == FAIL and hidden.fix is not None
    long = pccheck.check_tray(FakePc(tray={"visible": True, "tooltip": "x" * 130}))[0]
    assert long.status == FAIL and "130" in long.summary
    unseen = pccheck.check_tray(FakePc(answers=False))[0]
    assert unseen.status == FAIL and unseen.fix is not None and unseen.fix.action == "open:taskbar_settings"
    assert pccheck.check_tray(FakePc(answers=None))[0].status == PASS  # Windows says it shows; no answer is fine
    assert pccheck.check_tray(FakePc(tray={"skipped": "not running"}))[0].status == SKIP


def test_tray_tooltip_with_emoji_counts_utf16_units() -> None:
    tip = "\U0001f44b" * 64  # 128 UTF-16 units
    assert pccheck.check_tray(FakePc(tray={"visible": True, "tooltip": tip}))[0].status == FAIL


def test_toast_rules() -> None:
    assert pccheck.check_toast(FakePc())[0].status == PASS
    no = pccheck.check_toast(FakePc(toast={"shown": False, "reason": "Tk failed"}))[0]
    assert no.status == FAIL and "Tk failed" in no.summary
    assert pccheck.check_toast(FakePc(answers=False))[0].status == FAIL
    assert pccheck.check_toast(FakePc(toast={"skipped": "not running"}))[0].status == SKIP


# ---------------------------------------------------------------------------------------------- safety
def test_secure_desktop_rules() -> None:
    assert pccheck.check_secure_desktop(FakePc())[0].status == PASS
    assert pccheck.check_secure_desktop(FakePc(secure_desktop={"active": True}))[0].status == FAIL
    err = pccheck.check_secure_desktop(FakePc(secure_desktop={"error": "OpenInputDesktop failed"}))[0]
    assert err.status == FAIL and "always stopped" in err.summary
    assert pccheck.check_secure_desktop(FakePc(secure_desktop={"skipped": "Windows only"}))[0].status == SKIP


def test_real_secure_desktop_probe_skips_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pcenv.sys, "platform", "linux")
    env = pcenv.RealPcEnv.__new__(pcenv.RealPcEnv)
    assert "skipped" in env.secure_desktop()


def test_real_secure_desktop_probe_reports_a_failing_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.agentlink import desktop

    def broken() -> bool:
        raise desktop.SecureDesktopUnknown("OpenInputDesktop failed")

    monkeypatch.setattr(pcenv.sys, "platform", "win32")
    monkeypatch.setattr(desktop, "secure_desktop_active", broken)
    env = pcenv.RealPcEnv.__new__(pcenv.RealPcEnv)
    assert env.secure_desktop() == {"error": "OpenInputDesktop failed"}
    monkeypatch.setattr(desktop, "secure_desktop_active", lambda: False)
    assert env.secure_desktop() == {"active": False}


def test_echo_timing_passes_inside_the_windows_and_calibrates_when_slower() -> None:
    r = by_id(pccheck.check_input_echo(FakePc()))
    assert r["echo:mouse"].status == PASS and r["echo:key"].status == PASS
    assert r["echo:mouse"].details["recommended_s"] == 0.25
    slow = by_id(
        pccheck.check_input_echo(
            FakePc(input_echo={"mouse_ms": [40.0, 310.0], "key_ms": [], "mouse_window_s": 0.25, "key_window_s": 0.3})
        )
    )
    assert slow["echo:mouse"].status == FAIL
    assert slow["echo:mouse"].details["recommended_s"] == 0.5  # 310 ms x 1.5 = 465 ms -> 0.50 s
    assert "0.50 s" in slow["echo:mouse"].summary
    assert slow["echo:key"].status == SKIP
    assert pccheck.check_input_echo(FakePc(input_echo={"skipped": "no pynput"}))[0].status == SKIP
    assert pccheck.check_input_echo(FakePc(input_echo={"error": "x"}))[0].status == FAIL


def test_recommended_window_never_shrinks() -> None:
    assert pccheck.recommended_window([], 0.3) == 0.3
    assert pccheck.recommended_window([10.0], 0.3) == 0.3
    assert pccheck.recommended_window([200.0], 0.25) == 0.3


class _FakeListener:
    instances: list[_FakeListener] = []

    def __init__(self, on_move: Any = None, on_press: Any = None) -> None:
        self.cb = on_move or on_press
        _FakeListener.instances.append(self)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def test_measure_echo_times_the_listener_with_fake_pynput() -> None:
    _FakeListener.instances = []

    def deliver_later(i: int) -> None:
        threading.Timer(0.01, lambda: _FakeListener.instances[i].cb()).start()

    mouse = SimpleNamespace(
        Listener=lambda on_move: _FakeListener(on_move=on_move),
        Controller=lambda: SimpleNamespace(move=lambda dx, dy: deliver_later(0)),
    )
    keyboard = SimpleNamespace(
        Listener=lambda on_press: _FakeListener(on_press=on_press),
        Controller=lambda: SimpleNamespace(press=lambda k: deliver_later(1), release=lambda k: None),
        Key=SimpleNamespace(shift="shift"),
    )
    out = pcenv.measure_echo(mouse, keyboard, 0.25, 0.3, samples=2, wait=1.0)
    assert len(out["mouse_ms"]) == 2 and len(out["key_ms"]) == 2
    assert all(5.0 <= x < 900.0 for x in out["mouse_ms"] + out["key_ms"])
    assert out["mouse_window_s"] == 0.25 and out["key_window_s"] == 0.3


def test_measure_echo_counts_missed_echoes() -> None:
    mouse = SimpleNamespace(
        Listener=lambda on_move: _FakeListener(on_move=on_move),
        Controller=lambda: SimpleNamespace(move=lambda dx, dy: None),
    )
    keyboard = SimpleNamespace(
        Listener=lambda on_press: _FakeListener(on_press=on_press),
        Controller=lambda: SimpleNamespace(press=lambda k: None, release=lambda k: None),
        Key=SimpleNamespace(shift="shift"),
    )
    out = pcenv.measure_echo(mouse, keyboard, 0.25, 0.3, samples=1, wait=0.05)
    assert out["mouse_ms"] == [] and out["mouse_missed"] == 1 and out["key_missed"] == 1


def test_pinch_geometry() -> None:
    import numpy as np

    lm = np.zeros((21, 3))
    lm[0] = (0.5, 0.9, 0)
    lm[9] = (0.5, 0.5, 0)  # palm length 0.4
    lm[4], lm[8] = (0.40, 0.4, 0), (0.42, 0.4, 0)
    assert pcenv.pinch_closed(lm)
    lm[8] = (0.60, 0.4, 0)
    assert not pcenv.pinch_closed(lm)


# ------------------------------------------------------------------------------------------------ link
def test_link_reconnect_rules() -> None:
    r = by_id(pccheck.check_link(FakePc()))
    assert r["link:reconnect"].status == PASS and "3 fresh connections" in r["link:reconnect"].summary
    bad = by_id(pccheck.check_link(FakePc(link_reconnect={"tries": 3, "ok": 1, "error": "timeout"})))
    assert bad["link:reconnect"].status == FAIL and bad["link:reconnect"].fix is not None
    skip = by_id(pccheck.check_link(FakePc(link_reconnect={"skipped": "not running"})))
    assert skip["link:reconnect"].status == SKIP
    off = pccheck.check_link(FakePc(link={"enabled": False}))
    assert [x.id for x in off] == ["link"] and off[0].status == SKIP


def test_real_link_reconnect_uses_fresh_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_http(url: str) -> float:
        calls.append(url)
        return 50.0

    monkeypatch.setattr(pcenv, "http_ms", fake_http)
    host = SimpleNamespace(
        tunnel=SimpleNamespace(url="https://x.trycloudflare.com"),
        relay=None,
        named=None,
        server=None,
        core=SimpleNamespace(registry=SimpleNamespace(reconnectable=lambda: [1, 2])),
    )
    env = pcenv.RealPcEnv.__new__(pcenv.RealPcEnv)
    env.host, env.app = host, None
    out = env.link_reconnect()
    assert out["ok"] == 3 and out["reconnectable"] == 2 and len(calls) == 3
    env.host = SimpleNamespace(tunnel=None, relay=None, named=None, server=None)
    assert "skipped" in env.link_reconnect()


# ------------------------------------------------------------------------------- verdicts, redaction, files
def _report(**over: Any) -> dict[str, Any]:
    rep = pccheck.run(FakePc(**over))
    return pccheck.with_verdicts(rep.public())


def test_verdicts_and_headline() -> None:
    pub = _report()
    c = pub["verdicts"]
    assert c["fail"] == 1  # the broken internal camera
    assert pub["headline"] == f"{c['pass']} passed, 1 failed, {c['skipped']} skipped"


def test_redaction_hides_tokens_names_emails_and_private_paths(tmp_path: Path) -> None:
    raw = {
        "id": "x",
        "results": [
            {
                "id": "camera:JoyCam",
                "title": "Camera",
                "status": "fail",
                "verdict": "fail",
                "summary": "fern's file C:\\Users\\fern\\Videos\\a.mp4; mail fern@example.com; Bearer abcdefghijklmnop",
                "details": {"token": "pa_secret123456", "log": f"{tmp_path}/logs/away.log", "url": "http://u:pw@h/x"},
                "fix": None,
            }
        ],
    }
    out = pccheck.redact(raw, keep_dirs=[tmp_path], users=["fern"])
    blob = json.dumps(out)
    for secret in ("fern", "example.com", "abcdefghijklmnop", "pa_secret123456", "Videos", "u:pw"):
        assert secret not in blob, secret
    assert "<data>/logs/away.log" in blob
    assert out["redacted"] is True


def test_save_writes_json_html_text_latest_and_verified(tmp_path: Path) -> None:
    pub = pccheck.redact(_report(), keep_dirs=[tmp_path])
    files = pccheck.save(pub, tmp_path)
    for k in ("json", "html", "text", "latest", "verified"):
        assert files[k].exists(), k
        assert files[k].parent == tmp_path / "check"
    assert json.loads(files["latest"].read_text(encoding="utf-8"))["kind"] == "away.pccheck"
    page = files["html"].read_text(encoding="utf-8")
    assert "<table>" in page and "Fail" in page and "Integrated Camera" in page
    txt = files["text"].read_text(encoding="utf-8")
    assert txt.startswith("Project Away - Check my PC") and "[Fail]" in txt and "[Pass]" in txt
    ver = json.loads(files["verified"].read_text(encoding="utf-8"))
    assert ver["kind"] == "away.verified"


def test_html_escapes_summaries() -> None:
    pub = {"results": [{"id": "x", "title": "<b>", "summary": "<script>", "verdict": "pass", "fix": None}]}
    page = pccheck.html_report(pub)
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_verified_maps_readme_items_to_pass_fail_untested() -> None:
    ver = pccheck.verified(_report())["items"]
    assert ver["speakers"]["status"] == "pass"
    assert ver["secure_desktop"]["status"] == "pass"
    assert ver["pynput_echo"]["status"] == "pass" and set(ver["pynput_echo"]["by"]) == {"echo:mouse", "echo:key"}
    assert ver["joycam"]["status"] == "pass" and ver["joycam"]["by"] == ["camera:JoyCam"]
    assert ver["two_webcams"]["status"] == "fail"  # the internal camera failed
    assert ver["link_tunnel"]["status"] == "pass"
    off = pccheck.verified(_report(secure_desktop={"skipped": "Windows only"}, tray={"skipped": "x"}))["items"]
    assert off["secure_desktop"]["status"] == "untested" and off["tray_tooltip"]["status"] == "untested"
    no_joy = pccheck.verified(_report(cameras=[{"name": "Other", "opened": True, "fps": 30}]))["items"]
    assert no_joy["joycam"]["status"] == "untested"
    assert set(ver) == {it["key"] for it in pccheck.UNTESTED}


def test_finish_redacts_then_saves(tmp_path: Path) -> None:
    rep = pccheck.run(FakePc(cameras=[{"name": "C:\\Users\\fern\\cam", "opened": True, "fps": 30}]))
    pub, files = pccheck.finish(rep, keep_dirs=[tmp_path], data_dir=tmp_path)
    assert pub["redacted"] is True and "fern" not in files["json"].read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------- sharing / perms
class _Inbox:
    def __init__(self) -> None:
        self.added: list[tuple[str, list[str]]] = []

    def add(self, text: str, targets: Any, source: str = "", wake: bool = True) -> Any:
        self.added.append((text, [t.name for t in targets]))
        return SimpleNamespace(id="m1")


def _agent(name: str, scopes: list[str], url: str = "") -> Any:
    return SimpleNamespace(name=name, scopes=scopes, wake_url=url)


def test_share_only_reaches_agents_with_share_logs_or_diagnose() -> None:
    pub = _report()
    inbox = _Inbox()
    posts: list[tuple[str, dict[str, Any]]] = []
    agents = [
        _agent("Claude", ["read", "share_logs"], "https://hook/a"),
        _agent("Codex", ["read", "diagnose"]),
        _agent("Spy", ["read", "screen"], "https://hook/spy"),
    ]
    out = pccheck.share(pub, agents, inbox=inbox, post=lambda u, p: posts.append((u, p)), background=False)
    assert [a["agent"] for a in out["agents"]] == ["Claude", "Codex"]
    assert out["skipped_agents"] == ["Spy"]
    assert inbox.added[0][1] == ["Claude", "Codex"] and "Check my PC" in inbox.added[0][0]
    assert [u for u, _p in posts] == ["https://hook/a"] and posts[0][1]["event"] == "pccheck.completed"
    assert "event_seq" not in out  # never on the shared event stream


def test_share_with_nobody_allowed_stays_local() -> None:
    inbox = _Inbox()
    out = pccheck.share(_report(), [_agent("Spy", ["read"])], inbox=inbox, background=False)
    assert out["local_only"] and inbox.added == [] and out["skipped_agents"] == ["Spy"]


def test_may_receive_uses_the_permission_names() -> None:
    from away.agentlink import perms

    assert pccheck.may_receive(_agent("a", [perms.SHARE_LOGS]))
    assert pccheck.may_receive(_agent("b", [perms.DIAGNOSE]))
    assert not pccheck.may_receive(_agent("c", [perms.READ, perms.TUNE, perms.CONTROL]))
    assert not pccheck.may_receive(SimpleNamespace(name="d"))


def test_sent_text_names_who_got_it_and_who_did_not() -> None:
    msg = checkpc.sent_text({"agents": [{"agent": "Claude"}], "skipped_agents": ["Spy"]})
    assert "Sent to Claude" in msg and "Not sent to Spy" in msg


# ------------------------------------------------------------------------------------ probes and remote env
def test_probe_runs_only_listed_steps() -> None:
    with pytest.raises(ValueError):
        pcenv.probe(None, None, "__init__")
    with pytest.raises(ValueError):
        pcenv.probe(None, None, "_record")


def test_live_gesture_reads_the_pipeline() -> None:
    gestures = SimpleNamespace(hand_present=True)
    pipe = SimpleNamespace(gestures=gestures, cursor_state=lambda: {"state": "pinch", "pinch": 1.0})
    env = pcenv.RealPcEnv.__new__(pcenv.RealPcEnv)
    env.app = SimpleNamespace(pipeline=pipe, _frame_times=[])
    env.host = None
    env.tracking = lambda: {"fps": 20.0, "latency_ms": 15.0, "backend": "processor"}  # type: ignore[method-assign]
    assert env.wait_gesture("pinch", 0.5)["seen"] is True
    pipe.cursor_state = lambda: {"state": "idle", "pinch": 0.2}
    t0 = time.monotonic()
    assert env.wait_gesture("pinch", 0.2)["seen"] is False and time.monotonic() - t0 < 2.0
    gestures.hand_present = False
    assert env.wait_gesture("hand", 0.1)["seen"] is False


def test_in_app_confirm_uses_the_prompt_window_answerable_by_voice_or_gesture() -> None:
    from away.agentlink.prompts import NullUI, PromptManager

    q = PromptManager(NullUI())
    env = pcenv.RealPcEnv.__new__(pcenv.RealPcEnv)
    env.confirm_fn, env.app = None, None
    env.host = SimpleNamespace(core=SimpleNamespace(prompts=q))

    def answer() -> None:  # a spoken "yes" once the prompt shows
        t_end = time.monotonic() + 5.0
        while time.monotonic() < t_end and not q.feed_voice("yes"):
            time.sleep(0.02)

    t = threading.Thread(target=answer)
    t.start()
    try:
        ans = env.confirm("Did you hear it?")
    finally:
        t.join()
    assert ans is True
    env.host = None
    with pytest.raises(RuntimeError):
        env.confirm("x")


def test_remote_env_sends_measurements_to_the_app_and_keeps_prompts_local() -> None:
    sent: list[tuple[str, dict[str, Any]]] = []

    def ask_host(action: str, timeout: float = 20.0, **data: Any) -> dict[str, Any] | None:
        sent.append((action, data))
        if data["probe"] == "voice_list":
            return {"value": ["Light (Piper)"]}
        if data["probe"] == "confirm":
            return None  # the app has no prompt window: fall back to the local popup
        return {"active": False}

    shown: list[str] = []
    env = pcenv.RemotePcEnv(ask_host, shown.append, lambda q: True)
    env.prompt("Raise your hand")
    assert shown == ["Raise your hand"] and sent == []
    assert env.secure_desktop() == {"active": False}
    assert env.voice_list() == ["Light (Piper)"]
    assert env.confirm("Did you hear it?") is True
    assert [d["probe"] for _a, d in sent] == ["secure_desktop", "voice_list", "confirm"]
    assert all(a == "pccheck_probe" for a, _d in sent)
    with pytest.raises(AttributeError):
        env.not_a_step  # noqa: B018


def test_remote_env_runs_the_whole_check() -> None:
    fake = FakePc(cameras=[{"name": "JoyCam", "opened": True, "fps": 24.0}])

    def ask_host(action: str, timeout: float = 20.0, **data: Any) -> dict[str, Any] | None:
        res = getattr(fake, data["probe"])(*data["args"])
        return res if isinstance(res, dict) else {"value": res}

    env = pcenv.RemotePcEnv(ask_host, fake.prompt)
    pub = pccheck.with_verdicts(pccheck.run(env).public())
    assert pub["verdicts"]["fail"] == 0, pccheck.text(pub)


# ------------------------------------------------------------------------------------------------ CLI
def test_cli_parser_has_away_check() -> None:
    from away import cli

    a = cli.build_parser().parse_args(["check", "--only", "voices", "--send"])
    assert a.only == "voices" and a.send is True and a.fn is cli.cmd_check


def test_cli_rejects_unknown_parts(capsys: pytest.CaptureFixture[str]) -> None:
    rc = checkpc.main(argparse.Namespace(only="cameras,nope", window=False, send=False, trigger=None))
    assert rc == 2 and "nope" in capsys.readouterr().err


def test_run_check_saves_redacted_files(tmp_path: Path) -> None:
    got = checkpc.run_check(None, None, ["glide", "secure_desktop"], "cli", data_dir=tmp_path, env=FakePc())
    assert got["report"]["redacted"] is True
    assert Path(got["files"]["verified"]).exists()


def test_cli_end_to_end_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from away import paths
    from away.agentlink import host

    good = FakePc(cameras=[{"name": "JoyCam", "opened": True, "fps": 24}])
    monkeypatch.setattr(checkpc, "make_env", lambda p, c: (good, False))
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(host, "host_alive", lambda: False)
    rc = checkpc.main(argparse.Namespace(only=None, window=False, send=True, trigger=None))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Check my PC" in out and "Verified list" in out and "not running" in out


def test_tray_and_settings_open_away_check_window() -> None:
    from away.app.tray import TrayController
    from away.comfort import actions

    assert TrayController.check_pc_command()[-4:] == ["check", "--window", "--trigger", "tray"]
    assert actions.check_pc_command()[-4:] == ["check", "--window", "--trigger", "Settings"]
    opened: list[list[str]] = []
    assert actions.launch_check_pc(opened.append) is None and opened

    def broken(_c: list[str]) -> None:
        raise OSError("x")

    assert "could not open" in str(actions.launch_check_pc(broken))


def test_self_test_still_runs_its_own_list() -> None:
    from away.selftest import run as runmod

    rep = runmod.run(FakePc(), only=["glide"])
    assert rep.public()["kind"] == "away.selftest" and [r.id for r in rep.results] == ["glide"]
    s = {"heard": "project away is ready", "method": "loopback", "ms": 5}
    assert ck.check_stt(FakePc(), s=s)[0].status == PASS


def test_fix_object_is_public() -> None:
    assert Fix("open:camera_privacy", "Open").public() == {"action": "open:camera_privacy", "label": "Open"}
