"""v0.1.22: fixes from the user's real PC (seen through Agent Link) and "Talk to your agent".

1. Background audio (a YouTube video) was typed: "It runs in the blood." (conf 0.474, mouth_moving_frames 0) and four
   more. A barely-moving mouth now needs confidence 0.7 at any length, and while another app plays sound the mouth
   must move.
2. The agent's spoken "Can you hear me now?" was transcribed and typed: the microphone ignores the computer's own
   voice while it speaks and 0.7 s after.
3. "no face frames during speech (CPU busy?)" let speech through while a camera check held the camera.
4. Camera diagnose output flooded with NVIDIA Broadcast [VCAMDS] lines and ran for minutes.
5. get_screen monitor=0 silently returned the whole desktop.
6. notify_user / ask_user accept `message`.
7. "New link": a fresh tunnel without disconnecting anyone.
8. Talk to your agent: per-agent wake words, per-agent inbox, default agent, webhooks.
"""

from __future__ import annotations

import asyncio
import json
import os
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.agentlink import inbox as ib
from away.agentlink import screenshot
from away.agentlink.prompts import NullUI
from away.face.gate import SpeechGate
from away.multicam import diagnose
from away.multicam.capture import CameraDevice
from away.voice import talk as T
from away.voice.always_on import AmbientSession, StreamingEndpointer, Utterance, VoiceController
from away.voice.asr import FakeASR
from away.voice.other_audio import OtherAudioMeter
from away.voice.speech import EchoGuard, Speaker
from away.voice.weak_speech import motion_ratio, should_drop

AUDIO = np.zeros(16000, np.float32)


def gate_info(moving: int, frames: int = 40, speaker: int = 0) -> dict[str, Any]:
    return {
        "allowed": True,
        "reason": "enrolled face at the screen, facing it (mouth motion weak)" if moving + speaker == 0 else "ok",
        "frames": frames,
        "mouth_moving_frames": moving,
        "speaker_frames": speaker,
    }


# ===================================================================================== 1. weak speech
YOUTUBE = [  # (text, confidence, mouth_moving_frames) exactly as logged on his PC
    ("It runs in the blood.", 0.474, 0),
    ("I've been able to update the ashran.", 0.475, 0),
    ("Lalingawan, garut malilingawan itu.", 0.523, 4),
    ("I'm- I know Rossy agent, ooooh.", 0.423, 7),
    ("I don't know what.", 0.369, 2),
]


@pytest.mark.parametrize(("text", "conf", "moving"), YOUTUBE)
def test_the_logged_youtube_lines_are_not_typed(text: str, conf: float, moving: int) -> None:
    why = should_drop(text, gate_info(moving), conf)
    assert why and "mouth" in why


def test_real_dictation_with_a_moving_mouth_is_typed() -> None:
    assert should_drop("Please send the report to Maria today.", gate_info(18), 0.52) == ""
    assert should_drop("Open the budget sheet.", gate_info(6, speaker=8), 0.41) == ""  # 14/40 = 35 %
    assert should_drop("Thanks", gate_info(12), 0.8) == ""
    # a barely moving mouth but a very confident recognition (an off-angle webcam) is still typed
    assert should_drop("Let's meet at three tomorrow.", gate_info(2), 0.86) == ""


def test_motion_ratio() -> None:
    assert motion_ratio(gate_info(10)) == 0.25
    assert motion_ratio(gate_info(0, frames=0)) == 0.0
    assert motion_ratio(None) is None and motion_ratio({"allowed": False, "frames": 9}) is None
    assert motion_ratio({"allowed": True}) is None  # no frame counts: rule not applied


def test_other_app_playing_sound_requires_mouth_motion() -> None:
    # a video can be recognised perfectly: confidence alone is not enough while another app plays sound
    why = should_drop("Welcome back to the channel.", gate_info(1), 0.95, other_audio=True)
    assert "another app was playing sound" in why
    assert should_drop("Welcome back to the channel.", gate_info(1), 0.95, other_audio=False) == ""
    assert should_drop("Reply that I agree.", gate_info(20), 0.6, other_audio=True) == ""  # you talked over it
    assert should_drop("Reply that I agree.", None, 0.6, other_audio=True) == ""  # face gate off: unchanged


def test_thresholds_are_adjustable() -> None:
    # with the strict confidence lowered, only the v0.1.21 rule is left (a longer line needs 0.45)
    assert should_drop("It runs in the blood.", gate_info(0), 0.474, strict_confidence=0.4) == ""
    assert should_drop("It runs in the blood.", gate_info(0), 0.44, strict_confidence=0.4) != ""
    assert should_drop("It runs in the blood.", gate_info(8), 0.474, strict_ratio=0.1) == ""
    cfg = C.AppConfig()
    assert cfg.voice.strict_motion_ratio == 0.25 and cfg.voice.strict_confidence == 0.7
    assert cfg.voice.other_audio_needs_mouth
    cfg.voice.strict_confidence = 1.5
    with pytest.raises(ValueError, match="strict_confidence"):
        C.validate(cfg)


class Clean:
    def clean(self, text: str, context: Any = None) -> str:
        return text


class Sink:
    def __init__(self) -> None:
        self.items: list[str] = []

    def emit(self, text: str) -> None:
        self.items.append(text)


def ambient(outputs: list[str], confs: list[float | None], gi: dict[str, Any], **kw: Any) -> tuple[Any, Sink]:
    from away.bindings import BindingsConfig

    sink = Sink()
    s = AmbientSession(
        FakeASR(outputs, confs),
        Clean(),
        sink,
        StreamingEndpointer(16000),
        commands=lambda: BindingsConfig().voice_phrases(),
        run_command=lambda a, p: None,
        gate=lambda t0, t1: True,
        gate_info=lambda: dict(gi),
        **kw,
    )
    return s, sink


def test_always_on_drops_youtube_but_types_you() -> None:
    s, sink = ambient(["It runs in the blood.", "Send the invoice now."], [0.474, 0.55], gate_info(0))
    assert s.process(Utterance(AUDIO, 0, 1)).decision.kind == "ignore"
    s.gate_info = lambda: gate_info(15)
    assert s.process(Utterance(AUDIO, 2, 3)).decision.kind == "type"
    assert sink.items == ["Send the invoice now. "]


def test_always_on_with_other_audio_playing() -> None:
    playing = {"on": True}
    s, sink = ambient(
        ["Smash that like button.", "Smash that like button."],
        [0.97, 0.97],
        gate_info(1),
        other_audio=lambda t0, t1: playing["on"],
    )
    r = s.process(Utterance(AUDIO, 0, 1))
    assert r.decision.kind == "ignore" and "another app" in r.decision.reason
    playing["on"] = False
    assert s.process(Utterance(AUDIO, 2, 3)).decision.kind == "type"
    assert len(sink.items) == 1


def test_other_audio_meter_ignores_this_process_and_system_sounds() -> None:
    t = {"now": 10.0}
    reads = [
        [(os.getpid(), "ProjectAway.exe", 0.9), (0, "", 0.8)],  # our own voice and system sounds
        [(4242, "chrome.exe", 0.4)],  # YouTube
        [(4242, "chrome.exe", 0.001)],  # paused
    ]
    m = OtherAudioMeter(lambda: reads.pop(0), clock=lambda: t["now"])
    assert m.available
    assert m.sample() is False
    t["now"] = 11.0
    assert m.sample() is True
    t["now"] = 20.0
    assert m.sample() is False
    assert m.playing_during(10.8, 11.2) and m.playing_app(10.8, 11.2) == "chrome.exe"
    assert not m.playing_during(15.0, 19.0)


def test_other_audio_meter_unavailable_is_harmless() -> None:
    m = OtherAudioMeter(None)
    assert not m.available and m.sample() is False and not m.playing_during(0, 100)
    assert m.start() is m

    def boom() -> list[tuple[int, str, float]]:
        raise OSError("no audio device")

    bad = OtherAudioMeter(boom)
    assert bad.sample() is False and "no audio device" in bad.error and not bad.available


# ===================================================================================== 2. echo
def test_echo_guard_spans_and_tail() -> None:
    t = {"now": 100.0}
    g = EchoGuard(tail_s=0.7, clock=lambda: t["now"])
    assert not g.active()
    g.begin()
    assert g.active() and g.overlaps(99.0, 100.5)
    t["now"] = 102.0
    g.end()
    assert g.active()  # within the 0.7 s tail
    t["now"] = 102.6
    assert g.active()
    t["now"] = 102.8
    assert not g.active()
    assert g.overlaps(101.0, 101.5) and g.overlaps(102.65, 103.0) and not g.overlaps(103.0, 104.0)
    g.end()  # unmatched end is ignored


def test_speaker_marks_the_echo_guard_while_it_speaks() -> None:
    seen: list[bool] = []
    g = EchoGuard(tail_s=0.0)

    class Eng:
        def speak(self, text: str, v: Any) -> None:
            seen.append(g.active())

        def close(self) -> None:
            pass

    sp = Speaker(factories=[("fake", Eng)], echo=g)
    assert sp.say("Can you hear me now?")["spoken"]
    assert sp.wait(10)
    assert seen == [True] and g.spans


def test_can_you_hear_me_now_is_never_typed() -> None:
    g = EchoGuard(tail_s=0.7)
    g.spans = [(5.0, 7.0)]
    s, sink = ambient(["Can you hear me now?"], [0.95], gate_info(20), echo=g)
    r = s.process(Utterance(AUDIO, 5.5, 6.6))
    assert r.decision.kind == "ignore" and "own voice" in r.decision.reason
    assert s.asr.calls == []  # not even transcribed
    assert sink.items == []


def test_voice_controller_drops_the_mic_while_the_voice_speaks() -> None:
    s, _sink = ambient([], [], gate_info(20))
    fed: list[int] = []
    resets: list[int] = []
    s.feed = lambda chunk: fed.append(len(chunk))  # type: ignore[method-assign]
    s.reset = lambda: resets.append(1)  # type: ignore[method-assign]
    vc = VoiceController("always_on", None, s)
    state = {"on": True}
    vc.echo = types.SimpleNamespace(active=lambda: state["on"])
    vc.feed(np.zeros(480, np.float32))
    vc.feed(np.zeros(480, np.float32))
    assert fed == [] and resets == [1]  # reset once when the voice started
    state["on"] = False
    vc.feed(np.zeros(480, np.float32))
    assert fed == [480]


# ===================================================================================== 3. face gate
def _starved_gate() -> SpeechGate:
    g = SpeechGate()
    g.primary = 1
    for i in range(10):
        g.observe(i * 0.1, None, 1, 0.01, True, True)  # face present until t=0.9
    return g


def test_starved_gate_needs_recent_camera_frames() -> None:
    g = _starved_gate()
    assert not g.explain(3.0, 4.0)["allowed"]  # the camera stopped delivering: not "CPU busy"
    assert g.last_explain["reason"].startswith("no camera frames")
    g.note_camera_frame(2.5)  # frames kept coming, face tracking was just behind
    info = g.explain(3.0, 4.0)
    assert info["allowed"] and "CPU busy" in info["reason"]


def test_starved_gate_blocks_while_a_camera_check_holds_the_camera() -> None:
    g = _starved_gate()
    g.note_camera_frame(3.5)
    g.camera_holder = lambda: "Settings (camera check)"
    info = g.explain(3.0, 4.0)
    assert not info["allowed"] and info["camera_busy"] == "Settings (camera check)"
    assert "camera busy" in info["reason"]
    g.when_camera_busy = "allow"
    assert g.explain(3.0, 4.0)["allowed"]
    cfg = C.AppConfig()
    assert cfg.face.gate_when_camera_busy == "block"
    cfg.face.gate_when_camera_busy = "maybe"
    with pytest.raises(ValueError, match="gate_when_camera_busy"):
        C.validate(cfg)


# ===================================================================================== 4. diagnose
VCAMDS = "[VCAMDS] CVCamDSFilter::GetMediaType iPosition=3 hr=0x0"


def test_noise_filter_collapses_vcamds() -> None:
    lines = ["camera 0: JoyCam", *([VCAMDS] * 500), "  DirectShow: ok 30.0 fps measured", VCAMDS]
    out = diagnose.filter_lines(lines)
    assert out[0] == "camera 0: JoyCam" and out[1].startswith("(camera driver) [VCAMDS]")
    assert "  DirectShow: ok 30.0 fps measured" in out
    assert out[-1] == "(hid 500 camera-driver messages: NVIDIA Broadcast [VCAMDS])"
    assert len(out) == 4
    assert diagnose.is_noise("[ WARN:0@12.345] global cap_msmf.cpp (476) anonymous-namespace")
    assert not diagnose.is_noise("camera 2: NVIDIA Broadcast (virtual)")
    assert diagnose.filter_lines(["a", "b"]) == ["a", "b"]


def test_job_output_hides_driver_noise(tmp_path: Path) -> None:
    from away.agentlink.jobs import JobRunner

    class Proc:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.stdout = iter(["diagnosing...\n", *[VCAMDS + "\n"] * 300, "camera 0: JoyCam\n"])

        def wait(self, timeout: float = 0) -> int:
            return 0

        def kill(self) -> None:
            pass

    r = JobRunner(tmp_path, command=lambda a: ["away", *a], runner=Proc)
    job = r.start("cameras")
    for _ in range(200):
        if job.state != "running":
            break
        import time

        time.sleep(0.02)
    assert job.state == "done"
    assert job.output[-1].startswith("(hid 299 camera-driver messages")
    assert "camera 0: JoyCam" in job.output and len(job.output) == 4
    assert job.result == {"exit_code": 0, "driver_messages_hidden": 299}


def test_diagnose_skips_virtual_cameras_and_stops_at_the_budget() -> None:
    devs = [
        CameraDevice(0, "camera 0", "JoyCam"),
        CameraDevice(2, "camera 2", "NVIDIA Broadcast", virtual=True),
        CameraDevice(1, "camera 1", "HD User Facing"),
    ]
    t = {"now": 0.0}

    def clock() -> float:
        t["now"] += 25.0
        return t["now"]

    report = diagnose.run_diagnose(
        devs,
        cv2=types.SimpleNamespace(),
        backends=("winrt",),
        windows=True,
        winrt_available=lambda: (False, "not installed"),
        clock=clock,
    )
    out = report.lines
    assert out[0] == "camera 0: JoyCam"
    assert any("NVIDIA Broadcast (virtual) - skipped" in x for x in out)
    assert not any(x == "camera 1: HD User Facing" for x in out)
    assert out[-1].startswith("stopped after 60 s (time limit)")
    assert report.result["stopped"] is True
    assert [c["name"] for c in report.result["cameras"]] == ["JoyCam"]


def test_diagnose_checks_a_virtual_camera_when_asked() -> None:
    devs = [CameraDevice(2, "camera 2", "NVIDIA Broadcast", virtual=True), CameraDevice(0, "camera 0", "JoyCam")]
    report = diagnose.run_diagnose(
        devs,
        cv2=types.SimpleNamespace(),
        backends=("winrt",),
        windows=True,
        winrt_available=lambda: (False, "x"),
        include_virtual=True,
        budget_s=0,
    )
    out = report.lines
    assert "camera 2: NVIDIA Broadcast (virtual)" in out and not any("time limit" in x for x in out)
    assert report.result["stopped"] is False


def test_cli_has_budget_and_include_virtual() -> None:
    from away import cli

    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    if p is None:
        pytest.skip("parser factory not exposed")
    a = p.parse_args(["cameras", "diagnose", "--budget", "30", "--include-virtual"])
    assert a.budget == 30.0 and a.include_virtual


# ===================================================================================== 5. monitor 0
HIS_DESKTOP = [
    types.SimpleNamespace(left=-1920, top=0, width=1920, height=1080, primary=False),
    types.SimpleNamespace(left=0, top=0, width=1920, height=1080, primary=True),
    types.SimpleNamespace(left=1920, top=0, width=1366, height=768, primary=False),
]


def test_monitor_zero_is_an_error_with_the_valid_list() -> None:
    ds = screenshot.displays(HIS_DESKTOP)
    with pytest.raises(screenshot.ScreenRequestError) as e:
        screenshot.plan(ds, (5206, 1080), monitor=0)
    msg = str(e.value)
    assert "there is no monitor 0" in msg and "1..3" in msg and "1: 1920x1080 at -1920,0" in msg
    assert "leave monitor out for the whole desktop" in msg
    with pytest.raises(screenshot.ScreenRequestError, match="no monitor 4"):
        screenshot.plan(ds, (5206, 1080), monitor="4")
    assert screenshot.plan(ds, (5206, 1080), monitor=None).what == "desktop"
    assert screenshot.plan(ds, (5206, 1080), monitor=2).what == "monitor 2"


# ===================================================================================== core fixture
@pytest.fixture
def cfg_core(tmp_path: Path):
    from away.agentlink.host import build_core

    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    ui = NullUI()
    c = build_core(
        tmp_path,
        cfg=lambda: cfg,
        settings=lambda: cfg.agent_link,
        config_path=lambda: cfgp,
        ui=ui,
        screen_grab=lambda: None,
        monitors=lambda: HIS_DESKTOP,
    )
    c.prompts.polite = lambda: True
    posts: list[tuple[str, dict[str, Any]]] = []
    c.service.inbox = ib.Inbox(post=lambda url, payload: posts.append((url, payload)), background=False)
    return c, cfg, ui, posts


def test_screen_watch_monitor_zero_is_refused(cfg_core) -> None:
    core, *_ = cfg_core
    a, _ = core.registry.issue("Claude", ["read", "screen"], 3600)
    with pytest.raises(ValueError, match="there is no monitor 0"):
        core.service.screen_watch(a, on=True, monitor=0)
    assert core.service.screen_watch(a, on=True, monitor=3)["view"] == {"monitor": 3}


# ===================================================================================== 6. message alias
def test_notify_and_ask_accept_message(cfg_core) -> None:
    core, _cfg, ui, _ = cfg_core
    a, _ = core.registry.issue("Claude", ["read"], 3600)
    res = core.service.notify_user(a, message="Build finished")
    assert res["shown"] and ui.toasts[-1] == ("info", "Build finished")
    with pytest.raises(ValueError, match="text"):
        core.service.notify_user(a)
    out = core.service.ask_user(a, message="Deploy now?", options=["Yes", "No"], timeout_s=0.2)
    assert ui.shown and ui.shown[-1].question == "Deploy now?" and out["answer"] is None


def test_message_alias_over_the_api_dispatch(cfg_core) -> None:
    from away.agentlink import api

    core, _cfg, ui, _ = cfg_core
    a, _ = core.registry.issue("Claude", ["read"], 3600)
    api.dispatch(core.service, a, "notify_user", {"message": "hi there"})
    assert ui.toasts[-1][1] == "hi there"


# ===================================================================================== 7. new link
def test_friendly_403_explains_new_link() -> None:
    from away.agentlink.host import friendly_error

    msg = friendly_error(RuntimeError("HTTP Error 403: Forbidden"))
    assert "New link" in msg and "stay connected" in msg


def test_new_link_restarts_the_tunnel_without_revoking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from away.agentlink.host import AgentLinkHost

    h = AgentLinkHost(None, data_dir=tmp_path, ui=NullUI(), headless=True)
    local, _ = h.core.registry.issue("Claude Desktop", ["read"], 3600, kind="local")
    remote, _ = h.core.registry.issue("ChatGPT", ["read"], 3600, kind="remote")
    stopped: list[str] = []

    class Tun:
        def __init__(self, url: str) -> None:
            self.url = url

        def alive(self) -> bool:
            return True

        def stop(self) -> None:
            stopped.append(self.url)

    h.tunnel = Tun("https://old-blocked.trycloudflare.com")
    monkeypatch.setattr(h, "ensure_server", lambda: 4321)

    def fresh(port: int, download: bool) -> str:
        h.tunnel = Tun("https://fresh-name.trycloudflare.com")
        return h.tunnel.url

    monkeypatch.setattr(h, "_ensure_tunnel", fresh)
    res = h.new_link()
    assert stopped == ["https://old-blocked.trycloudflare.com"]
    assert res["url"].startswith("https://fresh-name.trycloudflare.com/c/")
    assert res["previous_url"] == "https://old-blocked.trycloudflare.com"
    assert {a.id for a in h.core.registry.list()} == {local.id, remote.id}  # nobody was disconnected
    assert h.handle({"action": "new_link"})["url"].startswith("https://fresh-name")


# ===================================================================================== 8. talk to your agent
AGENTS = [
    {"id": "c1", "name": "Claude", "wake_word": ""},
    {"id": "x2", "name": "Cursor", "wake_word": "Jarvis"},
    {"id": "d3", "name": "Claude Desktop", "wake_word": ""},
]


def test_each_agent_answers_to_hey_its_name() -> None:
    r = T.match("Hey Claude, remind me to call mom.", AGENTS)
    assert r == T.Route("c1", "remind me to call mom.", "hey claude")
    assert T.match("Hey Cursor run the tests", AGENTS).target == "x2"
    assert T.match("Hey Claude Desktop, open notes", AGENTS).target == "d3"  # the longest name wins
    assert T.match("Okay, Claude. What's next?", AGENTS).body == "What's next?"
    assert T.match("Claude is great", AGENTS) is None  # no "hey": normal dictation
    assert T.match("I told Claude hey", AGENTS) is None
    assert T.match("please type this", AGENTS) is None


def test_custom_wake_word_with_or_without_hey() -> None:
    assert T.match("Jarvis, check the build.", AGENTS) == T.Route("x2", "check the build.", "jarvis")
    assert T.match("Hey Jarvis check the build", AGENTS).target == "x2"
    assert T.match("Hey Jarvis", AGENTS).body == ""
    assert T.phrases_for("Claude Desktop") == [("claude desktop", True), ("claude", True)]


def test_hey_agent_is_the_generic_route() -> None:
    r = T.match("Hey agent, what's on my calendar?", AGENTS)
    assert r is not None and r.target == T.DEFAULT and r.body == "what's on my calendar?"


def test_talk_state_window() -> None:
    t = {"now": 0.0}
    st = T.TalkState(8.0, clock=lambda: t["now"])
    assert st.armed() is None
    st.arm("x2", "voice")
    t["now"] = 7.9
    assert st.armed() == "x2"
    t["now"] = 8.1
    assert st.armed() is None


def two_agents(core: Any) -> tuple[Any, Any]:
    claude, _ = core.registry.issue("Claude", ["read"], 3600)
    cursor, _ = core.registry.issue("Cursor", ["read"], 3600)
    return claude, cursor


def test_routing_with_two_agents_only_the_named_agent_gets_it(cfg_core) -> None:
    core, _cfg, _ui, posts = cfg_core
    claude, cursor = two_agents(core)
    core.registry.set_talk(cursor.id, wake_url="https://hooks.example.com/cursor")
    core.registry.set_talk(claude.id, wake_url="https://hooks.example.com/claude")
    res = core.service.deliver_message(claude.id, "remind me to stretch at four")
    assert res["sent"] and res["label"] == "Sent to Claude" and res["to"] == ["Claude"]
    assert [m["text"] for m in core.service.inbox_list(claude)["messages"]] == ["remind me to stretch at four"]
    assert core.service.inbox_list(cursor)["messages"] == []
    # only Claude's webhook fired, with the message
    assert [u for u, _ in posts] == ["https://hooks.example.com/claude"]
    assert posts[0][1]["event"] == "user.message" and posts[0][1]["text"] == "remind me to stretch at four"
    # only Claude sees the user.message event
    seen_c = core.service.wait_for_event(claude, 0, 0, ["user.message"])["events"]
    seen_x = core.service.wait_for_event(cursor, 0, 0, ["user.message"])["events"]
    assert len(seen_c) == 1 and "_for" not in seen_c[0]["data"] and seen_c[0]["data"]["to"] == ["Claude"]
    assert seen_x == []


def test_generic_message_goes_to_the_default_agent(cfg_core) -> None:
    core, cfg, _ui, posts = cfg_core
    claude, cursor = two_agents(core)
    cfg.agent_link.default_agent = "cursor"  # case-insensitive name
    res = core.service.deliver_message("", "what's the weather?", "gesture")
    assert res["label"] == "Sent to Cursor"
    assert core.service.inbox_list(cursor)["messages"][0]["source"] == "gesture"
    assert core.service.inbox_list(claude)["messages"] == []
    assert posts == []  # no webhook configured: nothing is posted


def test_generic_message_without_default_goes_to_every_connected_agent(cfg_core) -> None:
    core, cfg, _ui, _posts = cfg_core
    claude, cursor = two_agents(core)
    cfg.agent_link.default_agent = ""
    res = core.service.deliver_message("", "I'm stepping out for lunch")
    assert res["label"] == "Sent to Claude and Cursor"
    for a in (claude, cursor):
        msgs = core.service.inbox_list(a)["messages"]
        assert len(msgs) == 1 and msgs[0]["shared"]
    cfg.agent_link.default_agent = "Gone Agent"  # a default that is not connected any more: everyone
    assert core.service.deliver_message("", "hello")["to"] == ["Claude", "Cursor"]


def test_nobody_connected_is_not_sent(cfg_core) -> None:
    core, *_ = cfg_core
    res = core.service.deliver_message("", "anyone there?")
    assert not res["sent"] and "No AI agent is connected" in res["reason"]


def test_read_and_reply(cfg_core) -> None:
    core, _cfg, ui, _ = cfg_core
    claude, cursor = two_agents(core)
    mid = core.service.deliver_message(claude.id, "summarise my inbox")["id"]
    assert core.service.inbox_list(claude, unread_only=True)["unread"] == 1
    with pytest.raises(KeyError):
        core.service.inbox_read(cursor, mid)  # not Cursor's message
    assert core.service.inbox_read(claude, mid)["read"]
    out = core.service.inbox_reply(claude, mid, "You have 3 new emails.", speak=False)
    assert out["replied"] and ui.toasts[-1] == ("info", "You have 3 new emails.")
    assert ui.titles[-1] == "Message from Claude"
    assert core.service.inbox_list(claude)["messages"][0]["replied"]
    with pytest.raises(ValueError):
        core.service.inbox_reply(claude, None, "x")


def test_inbox_group_resolves() -> None:
    from away.agentlink.groups import resolve

    assert resolve("inbox", {"action": "list", "unread_only": True}) == ("inbox_list", {"unread_only": True})
    assert resolve("inbox", {"action": "reply", "id": "a", "text": "ok", "speak": False}) == (
        "inbox_reply",
        {"id": "a", "text": "ok", "speak": False},
    )


def test_wake_words_are_saved_per_agent(cfg_core, tmp_path: Path) -> None:
    core, *_ = cfg_core
    claude, cursor = two_agents(core)
    core.registry.set_talk(cursor.id, wake_word="Jarvis")
    pub = {a.name: a.public() for a in core.registry.list()}
    assert pub["Cursor"]["wake_phrase"] == "Jarvis" and pub["Claude"]["wake_phrase"] == "Hey Claude"
    assert not pub["Cursor"]["wake_me"]
    from away.agentlink.auth import AgentRegistry

    again = AgentRegistry(core.registry.path)
    assert again.get(cursor.id).wake_word == "Jarvis"
    with pytest.raises(ValueError, match="https"):
        core.registry.set_talk(claude.id, wake_url="http://example.com/hook")


def test_check_wake_url() -> None:
    assert ib.check_wake_url("") == ""
    assert ib.check_wake_url("https://x.example/hook") == "https://x.example/hook"
    assert ib.check_wake_url("http://127.0.0.1:8080/wake") == "http://127.0.0.1:8080/wake"
    assert ib.check_wake_url("http://localhost/wake") == "http://localhost/wake"
    for bad in ("ftp://x", "http://10.0.0.5/x", "notaurl", "https://"):
        with pytest.raises(ValueError):
            ib.check_wake_url(bad)


def test_failed_webhook_is_logged_not_raised() -> None:
    def down(url: str, payload: dict[str, Any]) -> None:
        raise OSError("connection refused")

    box = ib.Inbox(post=down, background=False)
    box.add("hi", [{"id": "a", "name": "Claude", "wake_url": "https://x.example/h"}])
    assert box.wake_log[-1]["ok"] is False and "refused" in box.wake_log[-1]["error"]


def test_sent_label() -> None:
    assert ib.sent_label(["Claude"]) == "Sent to Claude"
    assert ib.sent_label(["A", "B", "C"]) == "Sent to A, B and C"
    assert ib.sent_label([]) == ""


def hook(agents: list[dict[str, Any]], sent: list[tuple[str, str, str]]) -> T.TalkHook:
    h = T.TalkHook()
    h.agents = lambda: agents
    names = {a["id"]: a["name"] for a in agents}

    def deliver(target: str, text: str, source: str) -> str:
        sent.append((target, text, source))
        return f"Sent to {names.get(target, 'everyone')}"

    h.deliver = deliver
    return h


def test_hey_claude_in_always_on_goes_to_claude_not_the_keyboard() -> None:
    sent: list[tuple[str, str, str]] = []
    s, sink = ambient(["Hey Claude, remind me to call mom."], [0.9], gate_info(20), talk=hook(AGENTS, sent))
    r = s.process(Utterance(AUDIO, 0, 1))
    assert r.decision.kind == "message" and r.decision.reason == "Sent to Claude"
    assert sent == [("c1", "remind me to call mom.", "voice")] and sink.items == []


def test_bare_wake_phrase_waits_for_the_message() -> None:
    sent: list[tuple[str, str, str]] = []
    s, sink = ambient(["Hey Jarvis.", "Check whether the build passed."], [0.9, 0.85], gate_info(20))
    s.talk = hook(AGENTS, sent)
    assert s.process(Utterance(AUDIO, 0, 1)).decision.kind == "listen"
    r = s.process(Utterance(AUDIO, 2, 4))
    assert r.decision.reason == "Sent to Cursor" and sent == [("x2", "Check whether the build passed.", "voice")]
    assert sink.items == []


def test_summon_gesture_then_speech_goes_to_the_default_route() -> None:
    sent: list[tuple[str, str, str]] = []
    h = hook(AGENTS, sent)
    s, sink = ambient(["What's on my calendar today?", "Back to typing."], [0.9, 0.9], gate_info(20), talk=h)
    h.arm(T.DEFAULT, "gesture")
    assert s.process(Utterance(AUDIO, 0, 1)).decision.kind == "message"
    assert sent == [("", "What's on my calendar today?", "gesture")]
    assert s.process(Utterance(AUDIO, 2, 3)).decision.kind == "type"  # the window is used up
    assert sink.items == ["Back to typing. "]


def test_message_from_background_audio_is_not_sent() -> None:
    sent: list[tuple[str, str, str]] = []
    h = hook(AGENTS, sent)
    s, _ = ambient(["Some video talking."], [0.95], gate_info(0), talk=h, other_audio=lambda a, b: True)
    h.arm(T.DEFAULT, "gesture")
    assert s.process(Utterance(AUDIO, 0, 1)).decision.kind == "ignore" and sent == []


def test_talk_turned_off_types_as_before() -> None:
    sent: list[tuple[str, str, str]] = []
    h = hook(AGENTS, sent)
    h.enabled = lambda: False
    s, sink = ambient(["Hey Claude, this is just text."], [0.9], gate_info(20), talk=h)
    assert s.process(Utterance(AUDIO, 0, 1)).decision.kind == "type" and sent == []


def test_host_routes_spoken_messages_and_shows_who_got_it(tmp_path: Path) -> None:
    from away.agentlink.host import AgentLinkHost

    h = AgentLinkHost(None, data_dir=tmp_path, ui=NullUI(), headless=True)
    flashes: list[str] = []
    h._flash = flashes.append  # type: ignore[method-assign]
    claude, _ = h.core.registry.issue("Claude", ["read"], 3600)
    h.core.registry.issue("Cursor", ["read"], 3600)
    assert {a["name"] for a in h._talk_agents()} == {"Claude", "Cursor"}
    assert h._deliver_message(claude.id, "ping", "voice") == "Sent to Claude"
    assert flashes[-1] == "Sent to Claude"
    h._talk_armed("")
    assert flashes[-1] == "Listening... say your message for Claude and Cursor"
    assert h.handle({"action": "set_talk", "agent_id": claude.id, "wake_word": "Hal"})["wake_phrase"] == "Hal"


def test_mcp_has_the_inbox_tool_and_resource(cfg_core) -> None:
    pytest.importorskip("mcp")
    from away.agentlink import api, mcp_tools
    from away.agentlink.about import INSTRUCTIONS, briefing

    core, *_ = cfg_core
    mcp = mcp_tools.build(api.LocalBackend(core.service))
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "inbox" in names and len(names) == 16
    uris = {str(r.uri) for r in asyncio.run(mcp.list_resources())}
    assert "away://inbox" in uris
    assert "inbox(action=list|read|reply)" in INSTRUCTIONS and "user.message" in INSTRUCTIONS
    a, _ = core.registry.issue("Claude", ["read"], 3600)
    assert "Hey Claude" in briefing(a, core.service.get_context(a), {"active": False}, "ask")


def test_rest_inbox_round_trip(cfg_core, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("starlette")
    pytest.importorskip("uvicorn")
    import urllib.error
    import urllib.request

    from away.agentlink import api
    from away.agentlink.server import Server

    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    core, *_ = cfg_core
    claude, tok = core.registry.issue("Claude", ["read"], 3600)
    mid = core.service.deliver_message(claude.id, "hello from the couch")["id"]
    gate = api.Gate(core.service, core.registry, local_key="k" * 43, rate=(50.0, 60))
    srv = Server(api.create_app(core.service, gate, None))
    port = srv.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(method: str, path: str, body: Any = None) -> tuple[int, Any]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"},
            method=method,
        )
        try:
            with opener.open(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    try:
        st, body = call("GET", "/v1/inbox")
        assert st == 200 and body["messages"][0]["text"] == "hello from the couch"
        st, body = call("GET", f"/v1/inbox/read?id={mid}")
        assert st == 200 and body["read"]
        st, body = call("POST", "/v1/inbox/reply", {"id": mid, "text": "On it", "speak": False})
        assert st == 200 and body["replied"]
        st, body = call("POST", "/v1/notify", {"message": "alias works"})
        assert st == 200 and body["shown"]
    finally:
        srv.stop()


# ===================================================================================== crash.log
def test_crash_notice_can_be_dismissed_and_is_in_the_logs(cfg_core, tmp_path: Path) -> None:
    from away.app import crashlog

    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / crashlog.CRASH_FILE).write_text("[x] Uncaught exception in thread y\nTraceback...\n", encoding="utf-8")
    crashlog._write_seen(logs / crashlog.SEEN_FILE, 0, crashlog.ERROR_MESSAGE)
    assert crashlog.recent_note(logs) == crashlog.ERROR_MESSAGE
    assert crashlog.dismiss_note(logs)
    assert crashlog.recent_note(logs) is None
    assert (logs / crashlog.CRASH_FILE).exists()  # the log itself is kept
    assert crashlog.tail(logs, 1) == ["Traceback..."]
    core, *_ = cfg_core
    a, _ = core.registry.issue("Claude", ["read", "share_logs"], 3600)
    assert core.service.get_logs(a, 5)["crash_log"][-1] == "Traceback..."
    assert crashlog.tail(tmp_path / "nowhere") == []


def test_gui_model_saves_talk_settings(tmp_path: Path) -> None:
    from away.agentlink.gui_model import EVERYONE, AgentLinkModel

    cfgp = tmp_path / "config.yaml"
    m = AgentLinkModel(data_dir=tmp_path, config_path=cfgp)
    assert m.talk_enabled() and m.default_agent() == ""
    m.set_default_agent("Claude")
    m.set_talk_enabled(False)
    cfg = C.load(cfgp)[0]
    assert cfg.agent_link.default_agent == "Claude" and not cfg.agent_link.talk_enabled
    assert m.default_choices([{"name": "Claude"}, {"name": "Cursor"}, {"name": "Claude"}]) == [
        EVERYONE,
        "Claude",
        "Cursor",
    ]
    assert m.set_talk("x", wake_url="http://evil.example/x").startswith("use https://")
