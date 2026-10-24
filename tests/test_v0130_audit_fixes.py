"""Regression tests for the repo-wide audit fixes (config, bindings import, Agent Link, voice, tracking)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C

# ----------------------------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"hands": 5}, "hands must be a mapping"),
        ({"hands": {"roles": 3}}, "hands.roles must be a mapping"),
        ({"hands": {"cursor": {"sigmoid_low": "0.5"}}}, "sigmoid_low must be a number"),
        ({"hands": {"gestures": {"pinch_on": "0.3"}}}, "pinch_on must be a number"),
        ({"hands": {"cursor": {"active_region": [0, 0]}}}, "active_region"),
        ({"hands": {"cursor": {"active_region": [0.5, 0.5, 0.5, 0.9]}}}, "active_region"),
        ({"hands": {"cursor": {"active_region": 5}}}, "active_region"),
    ],
)
def test_bad_config_values_raise_value_error_not_a_crash(data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        C.from_dict(data)


def test_an_empty_section_keeps_its_defaults() -> None:
    cfg, _ = C.from_dict({"voice": None, "hands": {"cursor": None}})
    assert cfg.voice == C.AppConfig().voice and cfg.hands.cursor == C.AppConfig().hands.cursor
    cfg, _ = C.from_dict({"hands": {"cursor": {"active_region": [0.1, 0.2, 0.9, 0.8]}}})
    assert cfg.hands.cursor.active_region == (0.1, 0.2, 0.9, 0.8)


# ------------------------------------------------------------------------------------- bindings import
def test_an_imported_bindings_file_cannot_turn_shell_actions_on(tmp_path: Path) -> None:
    from away.bindings import BindingsConfig, BindingsError, load_bindings_file, merge_bindings

    f = tmp_path / "shared.yaml"
    f.write_text(
        'bindings: {allow_shell: true, profiles: {default: {voice: {hello: "shell:echo pwned"}}}}', encoding="utf-8"
    )
    with pytest.raises(BindingsError, match="shell actions are disabled"):
        load_bindings_file(f)
    f.write_text("bindings: {allow_shell: true, profiles: {default: {voice: {hello: 'key:f5'}}}}", encoding="utf-8")
    incoming, raw = load_bindings_file(f)
    assert incoming.allow_shell is False
    assert merge_bindings(BindingsConfig(), incoming, raw).allow_shell is False
    # a user who already allows shell commands keeps that switch (and can import them)
    on, raw = load_bindings_file(f, allow_shell=True)
    assert merge_bindings(BindingsConfig(allow_shell=True), on, raw).allow_shell is True


def test_cli_bindings_import_reports_an_unknown_tuning_key(tmp_path: Path, capsys: Any) -> None:
    from away import cli

    f = tmp_path / "t.yaml"
    f.write_text("bindings: {profiles: {chrome: {apps: [chrome], tuning: {hands.cursor.nope: 3}}}}", encoding="utf-8")
    rc = cli.main(["bindings", "--config", str(tmp_path / "c.yaml"), "import", str(f)])
    assert rc == 2 and "unknown setting" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------- actions
def test_launch_on_windows_drops_the_quotes_around_a_path() -> None:
    from away.actions import split_command

    argv = split_command(r'"C:\Program Files\Mozilla Firefox\firefox.exe" --new-window', nt=True)
    assert argv == [r"C:\Program Files\Mozilla Firefox\firefox.exe", "--new-window"]
    assert split_command('firefox --profile "a b"', nt=False) == ["firefox", "--profile", "a b"]


def test_every_held_key_is_released_even_when_one_release_fails() -> None:
    from away.control.macro_play import _release_all

    up: list[str] = []

    def key_up(k: str) -> None:
        up.append(k)
        if k == "shift":
            raise OSError("backend error")

    with pytest.raises(OSError):
        _release_all(key_up, ["ctrl", "shift", "alt"])
    assert up == ["alt", "shift", "ctrl"]


# -------------------------------------------------------------------------------------- live agent
def test_live_agent_never_picks_an_unrelated_element() -> None:
    from away.liveagent.jev import Decider

    d = Decider(None)
    assert d.pick_element("OK button", ["Delete account"]) is None
    assert d.pick_element("OK button", ["Delete account", "Cancel"]) is None
    pick = d.pick_element("the save button", ["File", "Save", "Save As..."])
    assert pick is not None and pick.value == "Save"


# -------------------------------------------------------------------------------------- Agent Link
def _core(tmp: Path) -> Any:
    from test_v0129_ingress_trust import _core as make

    return make(tmp)


def test_agents_cannot_send_undo_only_operations(tmp_path: Path) -> None:
    from away.agentlink.registry import ChangeError, Op

    ch = _core(tmp_path).service.changes
    for op in (
        Op("models", "restore", "stt", {"bindings.allow_shell": True, "agent_link.autofix": "auto_low_risk"}),
        Op("profiles", "restore", "evil", {"bindings": {"gestures": {"pinch": "launch:cmd.exe"}}}),
        Op("presets", "restore_file", "x", {"hands.cursor.gain": 1.0}),
    ):
        with pytest.raises(ChangeError, match="not a change an agent can make"):
            ch.preview([op])
        with pytest.raises(ChangeError, match="not a change an agent can make"):
            ch.apply([op], who="agent")
    assert C.load(tmp_path / "config.yaml")[0].bindings.allow_shell is False


def test_a_preview_writes_no_files_and_undo_still_restores(tmp_path: Path) -> None:
    from away.agentlink.registry import Op

    ch = _core(tmp_path).service.changes
    ch.presets.save("Mine", {"hands.cursor.gain": 1.5})
    ch.preview([Op("presets", "delete", "Mine")])
    assert ch.presets.names() == ["Mine"]  # still there after a preview (the person may say No)
    ch.preview([Op("presets", "create", "New")])
    assert "New" not in ch.presets.names()
    res = ch.apply([Op("presets", "delete", "Mine")], who="agent")
    assert ch.presets.names() == []
    ch.undo(res["changes"][0]["id"])  # undo may use the inverse (restore_file)
    assert ch.presets.names() == ["Mine"]


def test_macro_create_after_preview_succeeds(tmp_path: Path) -> None:
    from away.agentlink.registry import Op, apply_ops

    cfg = C.AppConfig()
    cfg.macros.store_dir = str(tmp_path / "macros")
    spec = {"title": "Hello", "steps": [{"type": "key", "keys": "ctrl+s"}]}
    apply_ops(cfg, [Op("macros", "create", "hello", spec)], dry_run=True)
    assert not list((tmp_path / "macros").glob("*.json")) if (tmp_path / "macros").exists() else True
    (done,) = apply_ops(cfg, [Op("macros", "create", "hello", spec)])  # used to fail: "already exists"
    assert done.after == "hello"


def test_a_junk_bearer_header_does_not_lift_the_pin_lockout(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from away.agentlink import api
    from test_v0129_ingress_trust import FakeClock, _gate

    core = _core(tmp_path)
    app = api.create_app(core.service, _gate(core, FakeClock(), {"port": None}))
    c = TestClient(app, base_url="http://127.0.0.1:8765")
    p = core.registry.new_pairing(["read"])
    wrong = "000000" if p.pin != "000000" else "111111"
    for _ in range(12):
        c.post("/v1/pair", json={"pin": wrong})
    assert c.post("/v1/pair", json={"pin": wrong}).status_code == 429
    r = c.post("/v1/pair", json={"pin": wrong}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 429


def test_malformed_initialize_on_the_pairing_link_is_not_a_500(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from away.agentlink import api
    from test_v0129_ingress_trust import FakeClock, _gate

    core = _core(tmp_path)
    app = api.create_app(core.service, _gate(core, FakeClock(), {"port": None}))
    c = TestClient(app, base_url="http://127.0.0.1:8765", raise_server_exceptions=False)
    p = core.registry.new_pairing(["read"])
    for params in ([], "x", {"clientInfo": "x"}, {"clientInfo": [1]}):
        r = c.post(f"/c/{p.code}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": params})
        assert r.status_code != 500, params


def test_a_session_extension_does_not_revive_an_expired_access_token() -> None:
    from away.agentlink.auth import Agent

    a = Agent("id", "n", "remote", [], "h", 0.0, expires=7200.0, token_expires=3600.0)
    assert a.active(5000.0)  # the session lives on (a link or persistent token extended it) ...

    from away.agentlink import auth

    reg = auth.AgentRegistry.__new__(auth.AgentRegistry)
    reg.agents, reg.clock, reg._lock = {"id": a}, lambda: 5000.0, threading.RLock()
    a.token_hash = auth.digest("tok")
    assert reg.authenticate("tok") is None  # ... but its one-hour access token stays expired
    reg.clock = lambda: 3000.0
    assert reg.authenticate("tok") is a


def test_rotated_logs_are_redacted_on_download(tmp_path: Path) -> None:
    core = _core(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    secret = "Authorization: Bearer paway_" + "A" * 43 + " email bob@example.com"
    (logs / "crash.log.1").write_text(secret + "\n", encoding="utf-8")
    p = core.registry.new_pairing(["read", "share_logs"])
    core.registry.mark_pending(p, "T")
    agent, _tok = core.registry.approve(p, 3600.0)
    out = core.service.read_file(agent, "logs/crash.log.1")
    assert b"paway_" not in out and b"bob@example.com" not in out


def test_jobs_refuse_flag_like_camera_names_and_run_one_check_at_a_time(tmp_path: Path) -> None:
    from away.agentlink import jobs as J

    gate = threading.Event()

    class Proc:
        def __init__(self) -> None:
            self.stdout = iter(())
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            gate.wait(5)
            return 0

        def kill(self) -> None:
            pass

    runner = J.JobRunner(tmp_path, command=lambda a: a, runner=lambda *a, **k: Proc())
    with pytest.raises(ValueError):
        runner.start("cameras", device="--json")
    results: list[Any] = []

    def go() -> None:
        try:
            results.append(runner.start("cameras"))
        except RuntimeError as exc:
            results.append(exc)

    ts = [threading.Thread(target=go) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    gate.set()
    assert sum(isinstance(r, RuntimeError) for r in results) == 3


def test_inbox_name_match_never_crosses_to_a_live_agent_with_the_same_name() -> None:
    from away.agentlink.inbox import Message

    m = Message.__new__(Message)
    m.to, m.to_names = ["local1"], ["Claude"]
    assert m.is_for("local1")
    assert not m.is_for("remote2", "Claude", live_ids={"local1", "remote2"})  # the local Claude still exists
    assert m.is_for("new3", "Claude", live_ids={"new3"})  # its old id is gone: the re-paired agent finds it


# -------------------------------------------------------------------------------------------- self-test
def test_selftest_redacts_persistent_agent_tokens() -> None:
    from away.selftest.redact import Redactor

    assert "paway_" not in Redactor([], users=[]).text("401 for paway_AbCdEfGhIjKlMnOpQrSt")


def test_settings_and_check_windows_are_not_extra_copies() -> None:
    from away.selftest.env import app_processes

    class P:
        def __init__(self, pid: int, cmd: list[str]) -> None:
            self.info = {"pid": pid, "name": "ProjectAway.exe", "cmdline": cmd}

    class PS:
        @staticmethod
        def process_iter(_attrs: Any) -> list[P]:
            return [
                P(100, ["ProjectAway.exe", "tray"]),
                P(200, ["ProjectAway.exe", "settings", "--tab", "Gesture Studio"]),
                P(300, ["ProjectAway.exe", "check"]),
            ]

    assert app_processes(PS, me=999) == [100]


# -------------------------------------------------------------------------------------------- voice
def test_vocabulary_replacements_are_literal() -> None:
    from away.voice.cleanup import Vocabulary

    v = Vocabulary.parse("my docs -> C:\\Users\\me\\Docs\ntemp dir -> C:\\temp\\new")
    assert v.apply("open my docs and temp dir") == "open C:\\Users\\me\\Docs and C:\\temp\\new"


def test_a_failing_player_does_not_leak_the_synth_thread() -> None:
    from away.voice import natural as N

    def play(_s: Any, _sr: int) -> None:
        raise OSError("device unplugged")

    before = {t for t in threading.enumerate() if t.name == "voice-synth"}
    with pytest.raises(OSError):
        N.play_chunks(lambda p: (np.zeros(10), 16000), ["a", "b", "c", "d", "e"], play)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and {t for t in threading.enumerate() if t.name == "voice-synth"} - before:
        time.sleep(0.05)
    assert not ({t for t in threading.enumerate() if t.name == "voice-synth"} - before)


def test_no_espeak_download_for_processors_it_is_not_built_for(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from away.voice import natural as N

    assert N.plat_key("darwin", "x86_64") != "darwin" and N.plat_key("linux", "armv7l") != "linux"
    assert N.plat_key("darwin", "arm64") == "darwin" and N.assets_for("kokoro", plat="darwin")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(N._pf, "machine", lambda: "x86_64")  # an Intel Mac
    assert N.assets_for("kokoro") == [] and not N.installed("kokoro")


def test_only_pinned_wheels_are_unpacked(tmp_path: Path) -> None:
    import hashlib
    import io
    import zipfile

    from away.voice import natural as N

    def wheel(name: str) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"{name}/__init__.py", "")
        return buf.getvalue()

    wheels = N.voices_dir(tmp_path) / "wheels"
    wheels.mkdir(parents=True)
    good = wheel("good")
    (wheels / "good-1-py3-none-any.whl").write_bytes(good)
    (wheels / "stale-1-py3-none-any.whl").write_bytes(wheel("stale"))
    a = N.Asset("good-1-py3-none-any.whl", "https://x/", len(good), hashlib.sha256(good).hexdigest())
    N.unpack_wheels(tmp_path, only=[a])
    lib = N.lib_dir(tmp_path)
    assert (lib / "good").exists() and not (lib / "stale").exists()
    bad = N.Asset(a.name, a.url, a.size, "0" * 64)
    with pytest.raises(N.VoiceDownloadError, match="damaged"):
        N.unpack_wheels(tmp_path, only=[bad])
    assert not (lib / ".unpacked").exists()


def test_steady_loud_noise_stops_producing_utterances() -> None:
    from away.voice.always_on import StreamingEndpointer

    ep = StreamingEndpointer(16000, max_utterance_s=2.0)
    rng = np.random.default_rng(0)
    out = ep.feed(rng.normal(0, 0.002, 32000).astype(np.float32))
    noise = rng.normal(0, 0.03, 16000 * 20).astype(np.float32)
    for i in range(0, len(noise), 1600):
        out += ep.feed(noise[i : i + 1600])
    assert len(out) <= 3  # not one every 2 s for 20 s


def test_a_message_queued_while_the_speaker_goes_idle_is_spoken() -> None:
    from away.voice.speech import EchoGuard, Speaker

    said: list[str] = []

    class Eng:
        name = "fake"

        def speak(self, t: str, _v: Any) -> None:
            said.append(t)

        def close(self) -> None:
            time.sleep(0.4)  # slow teardown (COM / ONNX)

    s = Speaker(factories=[("fake", Eng)], echo=EchoGuard())
    s.idle_s = 1.0
    s.say("first")
    time.sleep(1.2)  # the worker timed out and is closing its engine
    s.say("second")
    assert s.wait(4.0)
    assert said == ["first", "second"] and s._q.qsize() == 0


# ------------------------------------------------------------------------------------------ tracking
def test_secondary_camera_wrist_moves_at_the_primary_speed() -> None:
    from away.multicam.fusion import _Align

    base = np.random.default_rng(0).random((21, 3)) * 0.1 + 0.4
    prim, sec = base.copy(), base.copy()
    sec[:, :2] = (base[:, :2] - 0.5) * 2 + 0.5  # the secondary sees the hand twice as big
    a = _Align()
    for _ in range(50):
        a.learn(prim, sec, (640, 480), (640, 480))
    moved_p, moved_s = prim.copy(), sec.copy()
    moved_p[:, 0] += 0.05
    moved_s[:, 0] += 0.10
    assert a.apply(moved_s)[0, 0] == pytest.approx(moved_p[0, 0], abs=1e-6)


def test_a_pinch_held_over_a_pause_does_not_click_on_resume() -> None:
    from away.hands.gestures import GestureStateMachine
    from away.hands.synthetic import pose

    g = GestureStateMachine()
    t = 0.0
    for _ in range(10):
        t += 0.033
        g.update(t, pose("pinch_open"))
    for _ in range(5):
        t += 0.033
        g.update(t, pose("pinch"))
    g.paused = True
    for _ in range(60):
        t += 0.033
        g.update(t, pose("pinch_open"))
    g.paused = False
    t += 0.033
    kinds = [e.kind for e in g.update(t, pose("pinch_open"))]
    assert "click" not in kinds and "drag_end" not in kinds


def test_calibration_belongs_to_the_primary_camera() -> None:
    from away.hands.livecal import camera_key

    cfg = C.AppConfig()
    cfg.hands.cameras = [C.CameraSpec(1, "secondary"), C.CameraSpec(0, "primary")]
    assert camera_key(cfg) == "0"


def test_a_one_packet_csi_window_has_no_features() -> None:
    from away.wifi.features import window_features

    assert window_features(np.ones((1, 30), complex), None, 100.0) == {}
