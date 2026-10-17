"""v0.1.24 one-click self-test: every check's pass/warn/fail rule (fake cameras, engines, tunnel...), the runner's
time limits, redaction, delivery to *every* connected agent (inbox, wake-up address, event), the Agent Link op
(diagnose permission), REST + MCP resource, fix buttons and the CLI (the installer bundle check still works)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pytest

from away import config as C
from away.selftest import checks as ck
from away.selftest import deliver as dl
from away.selftest import fixes
from away.selftest import run as runmod
from away.selftest.model import FAIL, PASS, SKIP, WARN, report_text
from away.selftest.redact import Redactor, redact_report
from away.selftest.service import SelftestService


class FakeEnv:
    """A PC where everything works, unless a test changes one answer."""

    def __init__(self, **over: Any) -> None:
        self.data: dict[str, Any] = {
            "cameras": [
                {"name": "JoyCam", "opened": True, "fps": 24.0, "format": "640x480@24 MJPG", "backend": "DirectShow"}
            ],
            "camera_users": {"message": "", "users": [], "actions": ["retry"]},
            "tracking": {"fps": 24.0, "latency_ms": 18.0, "backend": "processor", "source": "live"},
            "gpu_tracking": {"setting": "auto", "using": "processor", "provider": "", "reason": "no GPU runtime"},
            "glide": {"hz": 118.0, "target_hz": 120},
            "big_cursor": {"supported": True, "size": 32, "user_size": 32, "restored": True},
            "instances": {"lock_held": True, "count": 1, "pids": [100]},
            "mic": {"device": "Mic (USB)", "rms_db": -35.0, "peak_db": -12.0},
            "stt": {"heard": "Project Away is ready.", "method": "loopback", "ms": 900},
            "tts": [
                {"name": "Windows voice", "first_audio_ms": 300},
                {"name": "Natural (Kokoro)", "first_audio_ms": 600},
            ],
            "notification": {"shown": True, "via": "window"},
            "link": {
                "enabled": True,
                "server": {"running": True, "port": 5000, "ok": True, "ms": 3.0},
                "tunnel": {"ok": True, "ms": 240.0},
                "relays": [{"name": "Cloudflare", "ok": True, "ms": 20}, {"name": "ntfy.sh", "ok": True, "ms": 90}],
                "agents": {"paired": 2, "reconnectable": 2},
            },
            "load": {"cpu_pct": 30.0, "top": [{"name": "Project Away", "cpu_pct": 12.0, "pid": 1}]},
            "models": {"missing": [], "rows": [{"title": "Hand tracking", "current_title": "MediaPipe", "on": True}]},
            "winkey": {
                "supported": True,
                "opened": True,
                "ms": 320.0,
                "process": "StartMenuExperienceHost.exe",
                "title": "Start",
                "class": "Windows.UI.Core.CoreWindow",
            },
        }
        self.data.update(over)

    def __getattr__(self, name: str) -> Any:
        if name not in self.data:
            raise AttributeError(name)
        val = self.data[name]

        def fn(*_a: Any) -> Any:
            if isinstance(val, Exception):
                raise val
            return val

        return fn


def statuses(results: list[Any]) -> dict[str, str]:
    return {r.id: r.status for r in results}


def test_a_healthy_pc_passes_everything() -> None:
    rep = runmod.run(FakeEnv())
    assert rep.status == PASS, report_text(rep.public())
    assert rep.counts()[FAIL] == 0
    ids = {r.id for r in rep.results}
    for want in ("camera:JoyCam", "tracking", "gpu", "glide", "big_cursor", "single", "duplicates", "mic", "stt",
                 "tts:Windows voice", "tts:Natural (Kokoro)", "notify", "link", "link:tunnel", "relay:Cloudflare",
                 "link:agents", "cpu", "models", "winkey"):  # fmt: skip
        assert want in ids, want


def test_camera_held_by_another_app_names_it_and_offers_to_close_it() -> None:
    env = FakeEnv(
        cameras=[
            {"name": "HD User Facing", "opened": False, "fps": 0, "error": "busy"},
            {"name": "JoyCam", "opened": True, "fps": 12},
        ],
        camera_users={
            "message": "Your built-in camera is being used by Zoom. Close it, then Retry.",
            "users": [{"name": "Zoom", "path": "C:\\Zoom\\Zoom.exe", "packaged": False}],
            "actions": ["close:C:\\Zoom\\Zoom.exe", "retry"],
        },
    )
    res = ck.check_cameras(env)
    bad, slow = res
    assert bad.status == FAIL and "Zoom" in bad.summary
    assert bad.fix is not None and bad.fix.label == "Close Zoom" and bad.fix.action == "close_app:C:\\Zoom\\Zoom.exe"
    assert slow.status == WARN and "12 fps" in slow.summary


def test_camera_rules_too_slow_none_and_turned_off() -> None:
    assert ck.check_cameras(FakeEnv(cameras=[{"name": "0", "opened": True, "fps": 4}]))[0].status == FAIL
    none = ck.check_cameras(FakeEnv(cameras=[]))[0]
    assert none.status == FAIL and none.fix and none.fix.action == "open:camera_privacy"
    off = ck.check_cameras(FakeEnv(cameras=[{"name": "Old cam", "off": True}]))[0]
    assert off.status == SKIP and off.fix and off.fix.label == "Turn on"
    idx = ck.check_cameras(
        FakeEnv(cameras=[{"name": "0", "opened": False, "fps": 0}], camera_users={"message": "0 could not be opened."})
    )[0]
    assert idx.title == "Camera 0" and idx.summary.startswith("Camera 0 could")


@pytest.mark.parametrize(
    ("tracking", "want"),
    [
        ({"fps": 25, "latency_ms": 15}, PASS),
        ({"fps": 15, "latency_ms": 20}, WARN),
        ({"fps": 6, "latency_ms": 20}, FAIL),
        ({"fps": None, "latency_ms": 90, "gpu_possible": True}, WARN),
        ({"unavailable": "The hand model is missing."}, FAIL),
    ],
)
def test_tracking_rules(tracking: dict[str, Any], want: str) -> None:
    assert ck.check_tracking(FakeEnv(tracking=tracking))[0].status == want


def test_gpu_rules() -> None:
    offer = ck.check_gpu_tracking(
        FakeEnv(gpu_tracking={"setting": "auto", "using": "processor", "provider": "DirectML", "models_ready": False})
    )[0]
    assert offer.status == WARN and offer.fix and offer.fix.action == "download:gpu_tracking"
    on = ck.check_gpu_tracking(
        FakeEnv(
            gpu_tracking={"setting": "auto", "using": "graphics card", "provider": "DirectML", "models_ready": True}
        )
    )[0]
    assert on.status == PASS
    asked = ck.check_gpu_tracking(
        FakeEnv(gpu_tracking={"setting": "gpu", "using": "processor", "provider": "", "reason": "no GPU"})
    )[0]
    assert asked.status == WARN and asked.fix and asked.fix.action == "set:hands.compute=cpu"


def test_pointer_cursor_and_instance_rules() -> None:
    assert ck.check_glide(FakeEnv(glide={"hz": 64, "target_hz": 120}))[0].status == WARN  # Windows timer at 15.6 ms
    assert ck.check_glide(FakeEnv(glide={"hz": 30, "target_hz": 120}))[0].status == WARN
    assert ck.check_glide(FakeEnv(glide={"hz": 0, "target_hz": 120}))[0].status == FAIL
    stuck = ck.check_big_cursor(FakeEnv(big_cursor={"supported": True, "size": 96, "user_size": 32, "stuck": True}))[0]
    assert stuck.status == FAIL and stuck.fix and stuck.fix.action == "reset_cursor"
    assert ck.check_big_cursor(FakeEnv(big_cursor={"supported": False, "note": "Windows only"}))[0].status == SKIP
    dup = statuses(ck.check_instances(FakeEnv(instances={"lock_held": True, "count": 3, "pids": [1, 2, 3]})))
    assert dup == {"single": PASS, "duplicates": FAIL}


def test_sound_rules() -> None:
    assert ck.check_mic(FakeEnv(mic={"device": "Mic", "rms_db": -80.0, "peak_db": -70}))[0].status == FAIL
    assert ck.check_mic(FakeEnv(mic={"error": "no microphone found"}))[0].status == FAIL
    assert ck.check_mic(FakeEnv(mic={"device": "Mic", "rms_db": -10.0, "peak_db": 0.0}))[0].status == WARN
    assert ck.check_stt(FakeEnv(stt={"heard": "project way is ready", "method": "loopback", "ms": 1}))[0].status == PASS
    assert ck.check_stt(FakeEnv(stt={"heard": "banana", "method": "spoken", "ms": 1}))[0].status == WARN
    assert ck.check_stt(FakeEnv(stt={"skipped": "no model"}))[0].status == SKIP
    tts = ck.check_tts(
        FakeEnv(tts=[{"name": "Piper", "error": "not downloaded"}, {"name": "Slow", "first_audio_ms": 4000}])
    )
    assert [r.status for r in tts] == [FAIL, WARN]
    assert ck.check_tts(FakeEnv(tts=[]))[0].status == FAIL
    assert ck.check_notification(FakeEnv(notification={"shown": False, "reason": "Focus assist"}))[0].status == FAIL


def test_link_rules_with_a_fake_tunnel() -> None:
    link = {
        "enabled": True,
        "server": {"running": True, "ok": True, "ms": 2},
        "tunnel": {"ok": False, "error": "HTTP 530"},
        "backup": {"ok": True, "ms": 2500},
        "relays": [{"name": "Pinggy", "ok": False, "error": "timeout", "why": "try localhost.run"}],
    }
    got = statuses(ck.check_link(FakeEnv(link=link)))
    assert got == {"link": PASS, "link:tunnel": FAIL, "link:backup": WARN, "relay:Pinggy": WARN}
    assert ck.check_link(FakeEnv(link={"enabled": False}))[0].status == SKIP


def test_winkey_rules() -> None:
    """Proposal #4: the Win keypress must really open Start; skips gracefully where it cannot run."""
    ok = ck.check_winkey(FakeEnv())[0]
    assert ok.status == PASS and ok.id == "winkey" and "320 ms" in ok.summary
    shut = ck.check_winkey(FakeEnv(winkey={"supported": True, "opened": False, "ms": 2010.0}))[0]
    assert shut.status == FAIL and "did not open within 2 s" in shut.summary
    assert shut.fix is not None and shut.fix.action == "rerun:winkey" and shut.fix.label == "Try again"
    sec = ck.check_winkey(FakeEnv(winkey={"supported": True, "opened": False, "secure_desktop": True}))[0]
    assert sec.status == FAIL and "security prompt" in sec.summary
    broke = ck.check_winkey(FakeEnv(winkey={"supported": True, "opened": False, "reason": "the keypress failed"}))[0]
    assert broke.status == FAIL and "did not work" in broke.summary
    skip = ck.check_winkey(FakeEnv(winkey={"supported": False, "reason": "the Windows key check only runs"}))[0]
    assert skip.status == SKIP


def test_winkey_start_menu_host_matching() -> None:
    from away.selftest.env import _start_menu_host

    win11 = {"process": "StartMenuExperienceHost.exe", "title": "Start", "class": "Windows.UI.Core.CoreWindow"}
    win10 = {"process": "explorer.exe", "title": "Start", "class": "Windows.UI.Core.CoreWindow"}
    taskbar = {"process": "explorer.exe", "title": "", "class": "Shell_TrayWnd"}
    other = {"process": "notepad.exe", "title": "Untitled", "class": "Notepad"}
    assert _start_menu_host(win11) and _start_menu_host(win10)
    assert not _start_menu_host(taskbar) and not _start_menu_host(other) and not _start_menu_host({})


def test_winkey_env_skips_gracefully_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.selftest.env import RealEnv

    monkeypatch.setattr("sys.platform", "linux")
    out = RealEnv(cfg=C.AppConfig()).winkey()
    assert out["supported"] is False and "Windows" in str(out["reason"])


def test_load_and_models_rules() -> None:
    busy = ck.check_load(FakeEnv(load={"cpu_pct": 97, "gpu_pct": 95, "gpu_name": "RTX 2060", "top": [
        {"name": "Project Away", "cpu_pct": 10, "pid": 1}, {"name": "NVIDIA Broadcast", "cpu_pct": 40, "pid": 77}]}))  # fmt: skip
    assert [r.status for r in busy] == [WARN, WARN]
    assert busy[0].fix and busy[0].fix.action == "close_pid:77" and "NVIDIA Broadcast" in busy[0].fix.label
    miss = ck.check_models(FakeEnv(models={"missing": ["hand model"], "rows": []}))[0]
    assert miss.status == FAIL and miss.fix and miss.fix.action == "repair_models"


def test_a_crashing_or_stuck_check_never_breaks_the_report() -> None:
    env = FakeEnv(mic=RuntimeError("PortAudio exploded"))

    def slow_tts() -> list[Any]:
        time.sleep(2)
        return []

    env.data["tts"] = []
    env.tts = slow_tts  # type: ignore[attr-defined]
    rep = runmod.run(env, limits={"tts": 0.2})
    by = {r.id: r for r in rep.results}
    assert by["mic"].status == FAIL and "PortAudio exploded" in by["mic"].summary
    assert by["tts"].status == WARN and "longer than" in by["tts"].summary
    assert by["mic"].fix and by["mic"].fix.action == "rerun:mic"


def test_only_some_checks_and_one_at_a_time() -> None:
    rep = runmod.run(FakeEnv(), only=["mic", "glide"])
    assert {r.id for r in rep.results} == {"mic", "glide"}
    with runmod._lock, pytest.raises(runmod.Busy):
        runmod.run(FakeEnv())


# ------------------------------------------------------------------------------------------------ redaction
def test_redaction_removes_tokens_user_names_and_outside_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "ProjectAway"
    r = Redactor([data_dir], users=["fernando"])
    text = (
        f"log at {data_dir}/logs/away.log; token pa_abcdefghijklmnop and par_qrstuvwxyz123; Bearer abc.def.ghijkl; "
        "C:\\Users\\fernando\\Videos\\me.mp4 /home/fernando/secret.txt fernando's PC key=supersecretvalue"
    )
    out = r.text(text)
    assert "<data>/logs/away.log" in out
    for leak in (
        "pa_abcdefghijklmnop",
        "par_qrstuvwxyz123",
        "abc.def.ghijkl",
        "Videos",
        "secret.txt",
        "fernando",
        "supersecretvalue",
    ):
        assert leak not in out, leak
    rep = redact_report(
        {"results": [{"details": {"refresh_token": "par_x", "path": "D:\\Games\\x.exe"}}]}, [data_dir], ["fernando"]
    )
    assert rep["redacted"] is True
    assert rep["results"][0]["details"] == {"refresh_token": "***", "path": "<path>"}


# ------------------------------------------------------------------------------------------------- delivery
class Agent:
    def __init__(self, name: str, wake: str = "", scopes: tuple[str, ...] = ()) -> None:
        self.id, self.name, self.wake_url, self.kind = name.lower(), name, wake, "remote"
        self.scopes = list(scopes)


class Hub:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, data: dict[str, Any]) -> dict[str, Any]:
        self.events.append((topic, data))
        return {"seq": len(self.events)}


def test_report_only_reaches_share_logs_agents(tmp_path: Path) -> None:
    """Proposal #14: the self-test report and Check my PC share one audience policy (share_logs)."""
    from away.agentlink.inbox import Inbox

    agents = [
        Agent("Hark", "https://hark.example/hook", ("read", "share_logs")),
        Agent("Muse", scopes=("read", "share_logs")),
        Agent("Spy", "https://spy.example/w", ("read", "diagnose")),  # diagnose alone is not enough
    ]
    inbox = Inbox(background=False, path=tmp_path / "inbox.json")
    hub, posts = Hub(), []
    env = FakeEnv(mic={"device": "C:\\Users\\fernando\\mic", "rms_db": -90.0, "peak_db": -80.0})
    svc = SelftestService(
        tmp_path, lambda _i: env, agents=lambda: agents, inbox=inbox, events=hub,
        post=lambda u, p: posts.append((u, p)), background=False,
    )  # fmt: skip
    out = svc.run_now(trigger="tray")
    rep, delivery = out["report"], out["delivery"]
    assert rep["status"] == FAIL and rep["redacted"] and rep["delivered_to"] == ["Hark", "Muse"]
    assert "fernando" not in json.dumps(rep)
    assert {a: inbox.unread(a) for a in ("hark", "muse")} == {"hark": 1, "muse": 1}
    assert inbox.unread("spy") == 0
    note = inbox.for_agent("muse")[0]
    assert note.source == "system" and "Microphone" in note.text and "selftest/latest" in note.text
    assert [u for u, _p in posts] == ["https://hark.example/hook"]
    assert all(p["event"] == "selftest.completed" and p["report"]["id"] == rep["id"] for _, p in posts)
    assert hub.events == []  # the stream reaches every connected agent, so the share_logs audience skips it
    assert delivery["skipped_agents"] == ["Spy"]
    assert {r["agent"]: r["webhook"] for r in delivery["agents"]} == {"Hark": "ok", "Muse": "none"}
    saved = runmod.load_latest(tmp_path)
    assert saved is not None and saved["id"] == rep["id"] and svc.latest() == saved


def test_deliver_audience_share_logs_filters_targets_and_skips_the_event_stream() -> None:
    hub = Hub()
    posts: list[tuple[str, dict[str, Any]]] = []
    agents = [Agent("Ok", "https://ok.example", ("share_logs",)), Agent("No", "https://no.example", ("read",))]
    res = dl.deliver(
        {"id": "x", "results": []},
        agents,
        events=hub,
        post=lambda u, p: posts.append((u, p)),
        background=False,
        audience="share_logs",
    )
    assert [r["agent"] for r in res["agents"]] == ["Ok"]
    assert res["skipped_agents"] == ["No"]
    assert [u for u, _p in posts] == ["https://ok.example"]
    assert hub.events == []


def test_deliver_rejects_an_unknown_audience() -> None:
    with pytest.raises(ValueError, match="unknown audience"):
        dl.deliver({"id": "x"}, [], audience="everyone")


def test_no_agent_connected_keeps_the_report_local(tmp_path: Path) -> None:
    hub = Hub()
    svc = SelftestService(tmp_path, lambda _i: FakeEnv(), agents=lambda: [], events=hub, background=False)
    out = svc.run_now()
    assert out["delivery"]["delivered"] is False and out["delivery"]["local_only"] is True
    assert out["report"].get("delivered_to") == []
    text = report_text(out["report"])
    assert "Project Away self-test" in text and "Everything works" in text


def test_webhook_failure_is_reported_not_raised() -> None:
    def boom(_u: str, _p: dict[str, Any]) -> None:
        raise OSError("connection refused")

    res = dl.deliver({"id": "x", "results": []}, [Agent("Hark", "https://hark.example")], post=boom, background=False)
    assert res["agents"][0]["webhook"].startswith("failed:")


# --------------------------------------------------------------------------------------------- Agent Link op
def make_core(tmp_path: Path) -> Any:
    pytest.importorskip("starlette")
    from away.agentlink.host import build_core
    from away.agentlink.prompts import NullUI

    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    return build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=NullUI())


def test_agents_trigger_it_with_the_diagnose_permission(tmp_path: Path) -> None:
    from away.agentlink.service import Forbidden, NotAvailable

    core = make_core(tmp_path)
    reader, _ = core.registry.issue("Reader", ["read"], 3600)
    doctor, _ = core.registry.issue("Doctor", ["read", "diagnose", "share_logs"], 3600)
    with pytest.raises(NotAvailable):
        core.service.run_selftest(doctor)  # outside the app
    core.service.selftest = SelftestService(
        tmp_path, lambda _i: FakeEnv(), agents=core.registry.reconnectable, inbox=core.service.inbox,
        events=core.events, background=False,
    )  # fmt: skip
    with pytest.raises(Forbidden, match="Run camera and speed checks"):
        core.service.run_selftest(reader)
    got = core.service.run_selftest(doctor, wait=True)
    assert got["state"] == "done" and got["report"]["trigger"] == "Doctor"
    # the report is pushed only to the share_logs audience, not to every connected agent
    assert core.service.inbox.unread(reader.id, reader.name) == 0
    assert core.service.inbox.unread(doctor.id, doctor.name) == 1
    assert core.service.selftest_latest(reader)["id"] == got["report"]["id"]  # pull access is unchanged
    assert core.events.since(0, ("selftest",)) == []  # the stream reaches everyone, so it is skipped


def test_rest_and_mcp_resource(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from away.agentlink import api, groups, mcp_tools

    core = make_core(tmp_path)
    a, tok = core.registry.issue("Hark", ["read", "diagnose"], 3600)
    core.service.selftest = SelftestService(
        tmp_path, lambda _i: FakeEnv(), agents=core.registry.reconnectable, background=False
    )
    gate = api.Gate(core.service, core.registry, local_key="k" * 43, rate=(100.0, 100))
    mcp = mcp_tools.build(api.LocalBackend(core.service))
    client = TestClient(api.create_app(core.service, gate, mcp), base_url="http://127.0.0.1:8765")
    h = {"Authorization": f"Bearer {tok}"}
    assert client.get("/v1/selftest/latest", headers=h).status_code == 503  # none yet
    r = client.post("/v1/selftest", headers=h, json={"wait": True})
    assert r.status_code == 200 and r.json()["report"]["status"] == PASS
    latest = client.get("/v1/selftest/latest", headers=h).json()
    assert latest["kind"] == "away.selftest" and latest["redacted"]
    r = client.post("/v1/rpc/diagnostics", headers=h, json={"action": "selftest_latest"})
    assert r.status_code == 200 and r.json()["id"] == latest["id"]
    assert groups.resolve("diagnostics", {"action": "selftest", "wait": True}) == ("run_selftest", {"wait": True})
    import asyncio

    uris = {str(x.uri) for x in asyncio.run(mcp.list_resources())}
    assert "away://selftest/latest" in uris
    assert "/v1/selftest" in json.dumps(api.openapi())


# ------------------------------------------------------------------------------------------------- fixes
def test_fix_buttons_do_what_they_say(tmp_path: Path) -> None:
    class App:
        retried = 0

        def retry_cameras(self) -> None:
            self.retried += 1

    app = App()
    assert fixes.apply("retry_cameras", app=app)["ok"] and app.retried == 1
    assert not fixes.apply("retry_cameras")["ok"]

    class Proc:
        def __init__(self, pid: int, name: str, exe: str) -> None:
            self.info = {"pid": pid, "name": name, "exe": exe}
            self.killed = False

        def terminate(self) -> None:
            self.killed = True

    procs = [Proc(10, "Zoom.exe", "C:\\Zoom\\Zoom.exe"), Proc(11, "notepad.exe", "C:\\Windows\\notepad.exe")]

    class PS:
        @staticmethod
        def process_iter(_attrs: Any) -> list[Proc]:
            return procs

    res = fixes.apply("close_app:C:\\Zoom\\Zoom.exe", psutil_mod=PS)
    assert res["ok"] and "Zoom.exe" in res["message"] and procs[0].killed and not procs[1].killed
    assert not fixes.apply("close_app:C:\\Nope\\nope.exe", psutil_mod=PS)["ok"]
    launched: list[list[str]] = []
    assert fixes.apply("repair_models", launcher=launched.append)["ok"] and launched[0][-2:] == ["setup", "--yes"]
    assert fixes.apply("open:settings/models", launcher=launched.append)["ok"] and launched[-1][-1] == "models"
    assert not fixes.apply("set:hands.compute=turbo")["ok"]
    assert not fixes.apply("does_not_exist")["ok"]


def test_set_fix_writes_the_config_when_the_app_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    monkeypatch.setattr(C, "config_path", lambda: cfgp)
    assert fixes.apply("set:hands.compute=cpu")["ok"]
    assert C.load(cfgp)[0].hands.compute == "cpu"


# --------------------------------------------------------------------------------------------------- CLI
def test_cli_keeps_the_installer_bundle_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import selftest as bundle
    from away.selftest import cli as stcli

    seen: list[Any] = []
    monkeypatch.setattr(bundle, "main", lambda json_path=None: seen.append(json_path) or 0)
    p = argparse.ArgumentParser()
    stcli.add_arguments(p)
    assert stcli.main(p.parse_args(["--json", "out.json"])) == 0 and seen == ["out.json"]
    assert stcli.main(p.parse_args(["--only", "nonsense"])) == 2


def test_cli_runs_here_when_the_app_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    from away.selftest import cli as stcli

    monkeypatch.setattr(stcli, "_via_app", lambda only, trig: None)
    monkeypatch.setattr(
        stcli,
        "run_here",
        lambda only, trig, ask=None: SelftestService(tmp_path, lambda _i: FakeEnv(), background=False).run_now(
            trig, only
        ),
    )
    p = argparse.ArgumentParser()
    stcli.add_arguments(p)
    out_file = tmp_path / "r.json"
    assert stcli.main(p.parse_args(["--save", str(out_file)])) == 0
    text = capsys.readouterr().out
    assert "Everything works" in text and "nothing was sent to agents" in text
    assert json.loads(out_file.read_text())["kind"] == "away.selftest"


def test_glide_measurement_is_real_and_moves_no_pointer() -> None:
    from away.selftest.env import RealEnv

    env = RealEnv(cfg=C.AppConfig())
    g = env.glide()
    assert g["target_hz"] == C.AppConfig().hands.cursor.output_hz
    assert g["hz"] > 5  # the output loop really ran (a loaded macOS CI runner measured 16 Hz of 120)


def test_app_process_counting_skips_helpers() -> None:
    from away.selftest.env import app_processes, gpu_load

    class P:
        def __init__(self, pid: int, name: str, cmd: list[str]) -> None:
            self.info = {"pid": pid, "name": name, "cmdline": cmd}

    class PS:
        @staticmethod
        def process_iter(_a: Any) -> list[P]:
            return [
                P(1, "ProjectAway.exe", ["ProjectAway.exe"]),
                P(2, "ProjectAway.exe", ["ProjectAway.exe", "agent-ui", "toast", "x"]),
                P(3, "python.exe", ["python", "-m", "away.app.tray"]),
                P(4, "python.exe", ["python", "-m", "pytest"]),
                P(5, "ProjectAway.exe", ["ProjectAway.exe"]),
            ]

    assert app_processes(PS, me=999) == [1, 3, 5]
    got = gpu_load(lambda _cmd: "37, 1200, 6144, NVIDIA GeForce RTX 2060\n")
    assert got == {"gpu_pct": 37.0, "vram_used_mb": 1200, "vram_total_mb": 6144, "gpu_name": "NVIDIA GeForce RTX 2060"}
    assert gpu_load(lambda _cmd: None) == {}


# ------------------------------------------------------------------ optional model benchmark hook (feat/model-refresh)
def test_model_bench_hook_is_optional_and_does_nothing_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import away.app

    monkeypatch.setitem(sys.modules, "away.app.model_bench", None)  # import fails like a build without it
    monkeypatch.delattr(away.app, "model_bench", raising=False)  # now on main: drop the already-imported submodule
    assert ck.model_bench_hook() is None
    assert ck.check_model_bench(FakeEnv()) == []
    if ck._BENCH is None:
        assert "model_bench" not in ck.CHECK_IDS


def test_model_bench_hook_result_becomes_a_selftest_line(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    seen: list[Any] = []

    def check(cfg: Any, data_dir: Any = None, run: bool = False) -> dict[str, Any]:
        seen.append((cfg, data_dir, run))
        return {
            "id": "model_bench",
            "title": "Newer models",
            "status": "warn",
            "summary": "Speech to text is too slow for the next preset.",
            "details": {"rtf": 0.9},
            "fix": {"id": "model_bench:run", "label": "Test again"},
        }

    mod = types.ModuleType("away.app.model_bench")
    mod.SELFTEST_HOOK = {"id": "model_bench", "title": "Newer models", "check": check}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "away.app.model_bench", mod)
    import away.app

    monkeypatch.setattr(away.app, "model_bench", mod, raising=False)
    hook = ck.model_bench_hook()
    assert hook is not None
    env = FakeEnv()
    env.cfg, env.data_dir = "CFG", "DIR"  # type: ignore[attr-defined]
    (res,) = ck.check_model_bench(env, hook)
    assert (res.id, res.status, res.details) == ("model_bench", WARN, {"rtf": 0.9})
    assert res.fix is not None and (res.fix.action, res.fix.label) == ("model_bench:run", "Test again")
    assert seen == [("CFG", "DIR", False)]  # reads the last result; never starts a benchmark by itself
    mod.SELFTEST_HOOK["check"] = lambda *a, **k: {"status": "weird"}  # type: ignore[attr-defined]
    assert ck.check_model_bench(env, mod.SELFTEST_HOOK)[0].status == WARN  # type: ignore[attr-defined]
