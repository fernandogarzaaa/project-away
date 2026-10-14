"""Live agent: opencode engine (fake release feed, fake server program), turns, voice wiring, Agent Link, Settings."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from away.liveagent import catalog, keys, opencode
from away.liveagent import settings as st
from away.liveagent.agent import LiveAgent, Turn, load_log, parse_action
from away.liveagent.jev import Answer, Decider
from away.liveagent.ledger import CostLedger
from away.liveagent.providers import FakeProvider
from away.liveagent.router import Funnel, Router
from away.liveagent.speech import BargeIn, LiveSpeech, wrap_feed
from liveagent_fakes import FAKE_OPENCODE, fake_downloader, make_archive, release_feed


# ------------------------------------------------------------------------------------------------ release feed
def test_asset_choice_per_platform() -> None:
    assert opencode.asset_candidates("win32", "AMD64") == [
        "opencode-windows-x64.zip",
        "opencode-windows-x64-baseline.zip",
    ]
    assert opencode.asset_candidates("win32", "ARM64") == ["opencode-windows-arm64.zip"]
    assert opencode.asset_candidates("darwin", "arm64") == ["opencode-darwin-arm64.zip"]
    assert opencode.asset_candidates("linux", "aarch64", musl=False) == ["opencode-linux-arm64.tar.gz"]
    assert opencode.asset_candidates("linux", "x86_64", musl=True)[0] == "opencode-linux-x64-musl.tar.gz"


@pytest.mark.parametrize(
    "plat,name,kind", [("win32", "opencode-windows-x64.zip", "zip"), ("linux", "opencode-linux-x64.tar.gz", "tar")]
)
def test_install_from_a_fake_release_feed_verifies_the_checksum(tmp_path, plat, name, kind) -> None:
    exe = opencode.exe_name(plat)
    data = make_archive(kind, exe)
    feed = release_feed("1.19.0", {name: data, "latest.json": b"{}"})
    seen: list[tuple[int, int]] = []
    info = opencode.install(
        tmp_path,
        progress=lambda d, t: seen.append((d, t)),
        release=feed,
        downloader=fake_downloader({name: data}),
        platform=plat,
        machine="x86_64",
    )
    assert info["version"] == "1.19.0" and Path(info["path"]).is_file() and info["license"] == "MIT"
    assert seen[-1] == (len(data), len(data))
    assert opencode.installed_engine(tmp_path)["sha256"] == info["sha256"]
    cmd, where = opencode.resolve(tmp_path, which=lambda _n: None)
    assert cmd == [info["path"]] and "Away's copy" in where


def test_install_refuses_a_bad_checksum_or_missing_fingerprint(tmp_path) -> None:
    data = make_archive("tar")
    bad = release_feed("1.19.0", {"opencode-linux-x64.tar.gz": data}, bad_digest="0" * 64)
    with pytest.raises(opencode.EngineError, match="fingerprint"):
        opencode.install(
            tmp_path,
            release=bad,
            downloader=fake_downloader({"opencode-linux-x64.tar.gz": data}),
            platform="linux",
            machine="x86_64",
        )
    assert not list((tmp_path / "engines" / "opencode").glob("*.part"))
    assert opencode.installed_engine(tmp_path) is None
    nodigest = {"tag_name": "v1", "assets": [{"name": "opencode-linux-x64.tar.gz", "browser_download_url": "u"}]}
    with pytest.raises(opencode.EngineError, match="no fingerprint"):
        opencode.pick_asset(nodigest, ["opencode-linux-x64.tar.gz"])
    with pytest.raises(opencode.EngineError, match="no download for this computer"):
        opencode.pick_asset(release_feed("1", {"other.zip": b"x"}), ["opencode-linux-x64.tar.gz"])


def test_install_refuses_unsafe_archive_paths(tmp_path) -> None:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        info = tarfile.TarInfo("../../evil")
        info.size = 1
        t.addfile(info, io.BytesIO(b"x"))
    data = buf.getvalue()
    feed = release_feed("2.0.0", {"opencode-linux-x64.tar.gz": data})
    with pytest.raises(opencode.EngineError, match="unsafe"):
        opencode.install(
            tmp_path,
            release=feed,
            downloader=fake_downloader({"opencode-linux-x64.tar.gz": data}),
            platform="linux",
            machine="x86_64",
        )


def test_update_check_follows_releases_and_never_touches_your_own_cli(tmp_path, monkeypatch) -> None:
    data = make_archive("tar")
    name = "opencode-linux-x64.tar.gz"
    assert opencode.check_update(tmp_path)["updated"] is False  # no copy of Away's own: nothing to update
    opencode.install(
        tmp_path,
        release=release_feed("1.19.0", {name: data}),
        downloader=fake_downloader({name: data}),
        platform="linux",
        machine="x86_64",
    )
    newer = release_feed("1.20.1", {name: data})
    monkeypatch.setattr(opencode, "latest_release", lambda client=None: newer)
    assert opencode.check_update(tmp_path, apply=False)["available"] == "1.20.1"
    res = opencode.check_update(tmp_path, downloader=fake_downloader({name: data}), platform="linux", machine="x86_64")
    assert res == {"updated": True, "version": "1.20.1", "previous": "1.19.0"}
    assert not (tmp_path / "engines" / "opencode" / "1.19.0").exists()
    monkeypatch.setattr(opencode, "latest_release", lambda client=None: release_feed("1.20.1", {name: data}))
    assert opencode.check_update(tmp_path)["updated"] is False


def test_find_cli_and_private_environment(tmp_path) -> None:
    assert opencode.find_cli(lambda n: "C:/npm/opencode.cmd" if n == "opencode.cmd" else None, platform="win32") == [
        "cmd",
        "/c",
        "C:/npm/opencode.cmd",
    ]
    assert opencode.find_cli(lambda n: "/usr/bin/opencode", platform="linux") == ["/usr/bin/opencode"]
    home = tmp_path / "h"
    (home / ".opencode" / "bin").mkdir(parents=True)
    (home / ".opencode" / "bin" / "opencode").write_text("x")
    assert opencode.find_cli(lambda n: None, home=home, platform="linux") == [str(home / ".opencode/bin/opencode")]
    assert opencode.find_cli(lambda n: None, home=tmp_path / "none", platform="linux") is None
    env = opencode.private_env(
        tmp_path,
        "pw",
        {"OPENAI_API_KEY": "sk-x", "GITHUB_TOKEN": "g", "PATH": "/bin", "OPENCODE_CONFIG": "/u", "GROQ_KEY": "k"},
    )
    assert env["PATH"] == "/bin" and env["OPENCODE_SERVER_PASSWORD"] == "pw"
    assert not any(k in env for k in ("OPENAI_API_KEY", "GITHUB_TOKEN", "OPENCODE_CONFIG", "GROQ_KEY"))
    assert env["XDG_CONFIG_HOME"].startswith(str(tmp_path))


# ------------------------------------------------------------------------------------------------ fake server
@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    script = tmp_path / "fake_opencode.py"
    script.write_text(FAKE_OPENCODE, encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    srv = opencode.OpenCodeServer([sys.executable, str(script)], tmp_path / "engine" / "home", start_timeout=30)
    yield srv
    srv.stop()


def test_server_lifecycle_auth_free_models_chat_and_tool_rejection(fake_server, tmp_path) -> None:
    from away.liveagent.http import HttpError

    srv = fake_server
    srv.ensure()
    assert srv.running() and srv.port > 0 and len(srv.password) > 20
    with pytest.raises(HttpError) as exc:  # the password is required
        srv.client.get_json(f"{srv.url}/permission", {}, 5)
    assert exc.value.status == 401
    cfg = json.loads((srv.home / "config" / "opencode" / "opencode.json").read_text())
    assert cfg["permission"] == {"*": "ask"} and cfg["autoupdate"] is False
    free = catalog.opencode_free_models(served=srv.free_models_raw())
    assert [m.id for m in free] == ["big-pickle", "nemotron-3.5-lightning-free"]
    pieces = list(srv.chat_stream("big-pickle", "say hi", system="be brief", timeout=20))
    assert "".join(pieces) == "Hello there. All good."  # reasoning part left out, early delta kept
    assert srv.client.get_json(f"{srv.url}/rejected", srv._auth(), 5) == ["per_1"]  # the bash ask got "no"
    log = (srv.home / "state" / "fake.log").read_text()
    seen = json.loads(log.strip().splitlines()[-1])
    assert seen["env_has_secret"] is False and os.path.samefile(seen["cwd"], srv.home.parent / "work")
    with pytest.raises(HttpError) as limited:  # "Free usage exceeded": stop at once, the funnel tries another model
        list(srv.chat_stream("big-pickle-limited", "hi", timeout=20))
    assert limited.value.status == 429 and "free_tier_limit" in str(limited.value)
    srv.stop()
    assert not srv.running()
    srv.last_used = -1e9
    assert srv.idle_check() is False  # already stopped


def test_server_idle_stop_and_restart(fake_server) -> None:
    srv = fake_server
    srv.idle_stop_s = 0.0
    srv.ensure()
    srv.last_used = time.monotonic() - 10
    assert srv.idle_check() and not srv.running()
    srv.ensure()
    assert srv.running()


def test_server_reports_a_program_that_dies(tmp_path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("import sys; sys.exit(3)", encoding="utf-8")
    srv = opencode.OpenCodeServer([sys.executable, str(bad)], tmp_path / "home", start_timeout=10)
    with pytest.raises(opencode.EngineError, match="exit code 3"):
        srv.start()


def test_use_free_models_one_click_downloads_when_missing(tmp_path, monkeypatch) -> None:
    from away.liveagent import runtime

    script = tmp_path / "fake_opencode.py"
    script.write_text(FAKE_OPENCODE, encoding="utf-8")
    installed: list[Any] = []

    def installer(data_dir, progress=None, **_kw):
        progress(30_000_000, 60_000_000)
        installed.append(data_dir)
        return {}

    monkeypatch.setattr(opencode, "find_cli", lambda *a, **k: None)
    rt = runtime.Runtime(
        tmp_path,
        keystore=keys.KeyStore(keys.MemoryBackend()),
        server_factory=lambda: opencode.OpenCodeServer(
            [sys.executable, str(script)], tmp_path / "home", start_timeout=30
        ),
    )
    s = st.LiveSettings()
    steps: list[str] = []
    try:
        msg = runtime.use_free_models(rt, s, lambda m, d, t: steps.append(m), installer)
    finally:
        rt.stop()
    assert installed and "Downloading opencode" in steps[0]
    assert s.enabled and s.preferred == "opencode_free/nemotron-3.5-lightning-free" and s.provider("opencode_free")
    assert msg.startswith("Free models are on (Nemotron 3.5 Lightning")  # the fast free model answers voice


def test_fallback_options_offer_free_keys(tmp_path) -> None:
    from away.liveagent.runtime import fallback_options

    class NoOllama:
        def get_json(self, *a, **k):
            raise OSError("no")

    opts = fallback_options(tmp_path, NoOllama())  # type: ignore[arg-type]
    assert [o["kind"] for o in opts] == ["gemini", "groq"] and all(o["key_url"].startswith("https://") for o in opts)

    class Ollama:
        def get_json(self, *a, **k):
            return {"models": [{"name": "llama3.2"}]}

    assert fallback_options(tmp_path, Ollama())[-1]["kind"] == "ollama"  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------------ turns
class FakeSpeech:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.interrupted = 0

    def begin(self) -> None:
        pass

    def say(self, s: str) -> bool:
        self.said.append(s)
        return True

    def interrupt(self) -> None:
        self.interrupted += 1


class FakeTools:
    available = True

    def __init__(self, decider: Any = None) -> None:
        self.decider = decider
        self.acts: list[tuple[str, dict]] = []
        self.handed: list[str] = []
        self.finished = 0

    def context(self) -> str:
        return "<untrusted-content source=context>\n{}\n</untrusted-content>"

    def elements(self):
        return ["File", "Save"], "<untrusted-content source=screen>\nbutton: Save\n</untrusted-content>"

    def screenshot(self):
        return b"jpg", {}

    def act(self, action, why="", screen_text="", **kw):
        self.acts.append((action, kw))
        return {"detail": action}

    def change_settings(self, ops, why=""):
        return {"applied": True}

    def handoff(self, text: str) -> str:
        self.handed.append(text)
        return "Sent to Hark"

    def finish(self) -> None:
        self.finished += 1


def _agent(tmp_path, replies, route="chat", s: st.LiveSettings | None = None, tools: Any = None):
    s = s or st.LiveSettings(enabled=True, allow_control=True, allow_screen=True)
    p = FakeProvider(id="p", replies=list(replies), free=True)
    led = CostLedger()
    led.limit_usd = 1

    def funnel():
        return Funnel(Router({"p": p}, [catalog.guess("p", "m", free=True)], allow_screen=s.allow_screen), led)

    class RouteDecider(Decider):
        def route(self, text):
            return Answer(route, 1.0, {}, "test")

    speech = FakeSpeech()
    agent = LiveAgent(lambda: s, funnel, lambda: RouteDecider(None), tools, speech, log_path=tmp_path / "turns.jsonl")
    return agent, speech, p


def test_chat_turn_streams_sentences_to_the_voice_and_logs_redacted(tmp_path) -> None:
    agent, speech, p = _agent(
        tmp_path, ["<think>let me think</think>Sure. Paris is the capital of France. Anything else?"]
    )
    turn = agent.handle("what's the capital of France? my mail is me@x.com")
    assert speech.said == ["Sure. Paris is the capital of France.", "Anything else?"]
    assert turn.outcome == "answered" and turn.model and turn.first_token_s is not None
    log = load_log(tmp_path / "turns.jsonl")
    assert log[0]["text"].endswith("[email]") and "me@x.com" not in json.dumps(log)
    agent.handle("and Germany?")
    sent = p.calls[-1]["messages"]
    assert any(m["content"].startswith("Sure.") for m in sent if m["role"] == "assistant")  # follow-ups have context


def test_computer_turn_plans_json_steps_and_finishes(tmp_path) -> None:
    tools = FakeTools()
    replies = [
        '{"tool": "look"}',
        'ok {"tool": "act", "action": "click", "element": "Save"} done',
        '{"tool": "say", "text": "Saved your file."}',
    ]
    agent, speech, p = _agent(tmp_path, replies, route="computer", tools=tools)
    turn = agent.handle("save my file")
    assert tools.acts == [("click", {"element": "Save"})]
    assert speech.said == ["On it.", "Saved your file."] and turn.outcome == "done" and turn.steps == 3
    assert tools.finished >= 1
    assert all("image" not in json.dumps(c["messages"], default=str) for c in p.calls)  # free model: no pictures


def test_handoff_and_refused_steps(tmp_path) -> None:
    tools = FakeTools()
    agent, speech, _p = _agent(tmp_path, ["x"], route="handoff", tools=tools)
    turn = agent.handle("write a long report about solar")
    assert tools.handed == ["write a long report about solar"] and turn.outcome == "handed off"
    assert speech.said == ["Sent to Hark."]

    class Refusing(FakeTools):
        def act(self, action, why="", screen_text="", **kw):
            raise RuntimeError("you did not allow 'Send'")

    replies = [
        '{"tool": "act", "action": "click", "element": "Send"}',
        '{"tool": "say", "text": "Okay, I did not send it."}',
    ]
    agent2, speech2, p2 = _agent(tmp_path, replies, route="computer", tools=Refusing())
    agent2.handle("send the email")
    assert "refused: you did not allow 'Send'" in json.dumps(p2.calls[-1]["messages"])
    assert speech2.said[-1] == "Okay, I did not send it."


def test_without_permissions_a_computer_request_is_just_chat(tmp_path) -> None:
    s = st.LiveSettings(enabled=True, allow_control=False, allow_screen=False, allow_settings=False)
    agent, speech, _ = _agent(
        tmp_path, ["I can't use your computer yet, but here is how."], route="computer", s=s, tools=FakeTools()
    )
    assert agent.handle("open notepad").route == "computer" and speech.said[0].startswith("I can't")


def test_stop_word_cancels_and_parse_action() -> None:
    sp = FakeSpeech()
    a = LiveAgent(lambda: st.LiveSettings(), lambda: None, speech=sp)  # type: ignore[arg-type,return-value]
    assert a.submit("stop") and sp.interrupted == 1
    assert (
        parse_action('```json\n{"tool": "act", "action": "type_text", "text": "hi {there}"}\n```')["text"]
        == "hi {there}"
    )
    assert parse_action("I will just say hello.")["tool"] == "say"
    assert Turn(1, 0.0, "x").public()["time"]


# ------------------------------------------------------------------------------------------------ speech
def test_barge_in_fires_on_speech_louder_than_the_echo() -> None:
    clock = [0.0]
    fired: list[int] = []
    b = BargeIn(lambda: True, lambda: fired.append(1), sample_rate=16000, clock=lambda: clock[0])
    echo = np.full(1600, 0.02, dtype=np.float32)
    for _ in range(4):  # 0.4 s of the agent's own voice sets the echo level
        b.feed(echo)
        clock[0] += 0.1
    for _ in range(5):
        b.feed(echo)
        clock[0] += 0.1
    assert not fired
    loud = np.full(1600, 0.2, dtype=np.float32)
    for _ in range(3):
        b.feed(loud)
        clock[0] += 0.1
    assert fired == [1]
    quiet = BargeIn(lambda: False, lambda: fired.append(2))
    assert quiet.feed(loud) is False


def test_live_speech_interrupt_drops_waiting_sentences() -> None:
    import queue
    import threading

    class Spk:
        def __init__(self) -> None:
            self._q: queue.Queue = queue.Queue()
            self._lock = threading.Lock()
            self._idle = threading.Event()
            self.calls: list[str] = []

        def say(self, text, force=False):
            self.calls.append(text)
            self._q.put(text)
            self._idle.clear()
            return {"spoken": True}

    spk = Spk()
    stopped: list[int] = []
    ls = LiveSpeech(spk, stop_audio=lambda: stopped.append(1))
    ls.begin()
    ls.say("One.")
    ls.say("Two.")
    assert ls.speaking()
    ls.interrupt()
    assert spk._q.empty() and stopped == [1] and ls.cancelled
    assert ls.say("Three.") is False
    ls.begin()
    assert ls.say("Four.")

    class Voice:
        def __init__(self) -> None:
            self.got: list[Any] = []

        def feed(self, c):
            self.got.append(c)

    v = Voice()
    b = BargeIn(lambda: False, lambda: None)
    assert wrap_feed(v, b) and not wrap_feed(v, b)
    v.feed(np.zeros(10))
    assert len(v.got) == 1


# ------------------------------------------------------------------------------------------------ app wiring
def _fake_app() -> Any:
    from away.voice.talk import TalkHook

    talk = TalkHook()
    talk.agents = lambda: [{"id": "a1", "name": "Hark", "wake_word": ""}]
    sent: list[tuple] = []
    talk.deliver = lambda t, x, s: (sent.append((t, x)), "Sent to Hark")[1]
    bus = SimpleNamespace(publish=lambda *a, **k: None)
    return SimpleNamespace(
        talk=talk,
        bus=bus,
        agent_link=None,
        voice=None,
        cfg=SimpleNamespace(speech=None, voice=SimpleNamespace(sample_rate=16000)),
    ), sent


def test_hey_name_reaches_the_live_agent_through_the_existing_wake_words(tmp_path) -> None:
    from away.liveagent.integrate import LiveAgentHost

    app, sent = _fake_app()
    host = LiveAgentHost(app, tmp_path)
    host.hook_talk()
    got: list[str] = []
    host.agent.submit = lambda text, source="voice": got.append(text) or True  # type: ignore[method-assign]
    assert all(a["id"] != "live-agent" for a in app.talk.agents())  # off: not listening
    st.save(
        st.LiveSettings(enabled=True, name="Jarvis", providers=[st.ProviderConfig("opencode_free", "opencode_free")]),
        host.watcher.path,
    )
    host.watcher.poll()
    route = app.talk.route("Hey Jarvis, what time is it in Tokyo?")
    assert route is not None and route.target == "live-agent"
    assert app.talk.send(route.target, route.body, "voice") == "Jarvis is answering" and got == [
        "what time is it in Tokyo?"
    ]
    hark = app.talk.route("Hey Hark, write the report")
    assert app.talk.send(hark.target, hark.body, "voice") == "Sent to Hark" and sent == [("a1", "write the report")]
    assert host.tray_text().startswith("Live agent: on")
    assert host.toggle() and host.tray_text() == "Live agent: off"


def test_agent_link_domain_switch_and_off_never_on(tmp_path) -> None:
    from away.agentlink import registry as reg
    from away.liveagent.integrate import LiveAgentHost, register_domain

    app, _ = _fake_app()
    host = LiveAgentHost(app, tmp_path)
    st.save(
        st.LiveSettings(enabled=True, providers=[st.ProviderConfig("groq", "groq", "openai/gpt-oss-20b")]),
        host.watcher.path,
    )
    host.watcher.poll()
    register_domain(host)
    try:
        dom = reg.domains()["live_agent"]
        rows = {r["key"]: r for r in dom.list(None)}
        assert rows["enabled"]["value"] is True and "key" not in json.dumps(rows).replace('"key"', "")
        a = dom.apply(None, reg.Op("live_agent", "switch", "preferred", "groq/openai/gpt-oss-20b"))
        assert st.load(host.watcher.path).preferred == "groq/openai/gpt-oss-20b" and a.low_risk
        with pytest.raises(reg.ChangeError):
            dom.apply(None, reg.Op("live_agent", "switch", "preferred", "nobody/x"))
        with pytest.raises(reg.ChangeError):
            dom.apply(None, reg.Op("live_agent", "on", "enabled", True))
        off = dom.apply(None, reg.Op("live_agent", "off", "enabled"))
        assert st.load(host.watcher.path).enabled is False
        dom.apply(None, off.inverse[0])  # undo puts it back as it was
        assert st.load(host.watcher.path).enabled is True
        assert "live_agent" in json.dumps(host.public_status()) or host.public_status()["enabled"]
    finally:
        reg.unregister("live_agent")


def test_tray_item_and_update_hook(tmp_path) -> None:
    from away.liveagent.integrate import hook_updates, menu_items

    class Pystray:
        class MenuItem:
            def __init__(self, text, action):
                self.text, self.action = text, action

    opened: list[str] = []
    app = SimpleNamespace(live_agent=None)
    item = menu_items(Pystray, app, opened.append)[0]
    assert item.text(None) == "Live agent: off"
    item.action(None, None)
    assert opened == ["Live agent"]
    calls: list[int] = []
    upd = SimpleNamespace(_run_check=lambda: calls.append(1) or "result")
    hook_updates(upd, tmp_path)
    hook_updates(upd, tmp_path)  # only once
    assert upd._run_check() == "result" and calls == [1]


def test_attach_never_raises(tmp_path) -> None:
    from away.liveagent.integrate import attach

    class Broken:
        bus = SimpleNamespace(publish=lambda *a, **k: None)

        @property
        def talk(self):
            raise RuntimeError("boom")

    assert attach(Broken(), tmp_path) is None


# ------------------------------------------------------------------------------------------------ Settings logic
def test_settings_panel_keys_test_remove(tmp_path, monkeypatch) -> None:
    from away.liveagent.gui import LivePanel, plain_error
    from away.liveagent.http import Client, HttpError
    from liveagent_fakes import FakeTransport, openai_chunks

    def handler(method, url, headers, body):
        if url.endswith("/models"):
            return 200, {}, {"data": [{"id": "openai/gpt-oss-20b"}, {"id": "whisper-large-v3"}]}
        if headers.get("Authorization") != "Bearer gsk_good_key_123456":
            return 401, {}, {"error": {"message": "Invalid API Key"}}
        return 200, {}, openai_chunks("ready")

    ks = keys.KeyStore(keys.MemoryBackend())
    panel = LivePanel(tmp_path, ks)
    panel.runtime.client = Client(FakeTransport(handler))
    assert panel.set_enabled(True) == "Choose free models or add a key first"
    ok, msg = panel.add_key("groq", "gsk_good_key_123456")
    assert ok and msg.startswith("Works"), msg
    assert ks.get("groq") == "gsk_good_key_123456" and "gsk_good" not in panel.path.read_text()
    assert panel.s.preferred == "groq/openai/gpt-oss-20b"
    assert ("groq/openai/gpt-oss-20b", "Openai/gpt Oss 20b - Groq") in panel.model_choices() or len(
        panel.model_choices()
    ) == 2
    ks.set("groq", "gsk_bad_key_99999999")
    ok, msg = panel.test("groq")
    assert not ok and "refused" in msg
    assert panel.add_key("openai", "")[1] == "Paste the key first"
    assert panel.set_name("Jarvis") == 'Say "Hey Jarvis"' and panel.set_name("!!").startswith("Use a short")
    assert panel.remove("groq") == "Removed" and ks.get("groq") == "" and not panel.s.providers
    assert plain_error(HttpError(429, "x")).startswith("too many") and panel.recent_turns() == []
    assert "60 MB" in panel.engine_text() or "opencode" in panel.engine_text()


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and os.environ.get("DISPLAY")), reason="needs an X display (Xvfb)"
)
def test_settings_live_agent_tab_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tk = pytest.importorskip("tkinter")
    monkeypatch.setenv("AWAY_DATA_DIR", str(tmp_path / "data"))
    from away.app import settings_gui as G

    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - no Tk
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        w = G.SettingsWindow(root, G.SettingsModel(path=tmp_path / "config.yaml"))
        root.update()
        assert "Live agent" in [w.nb.tab(t, "text") for t in w.nb.tabs()]
        panel = getattr(w, "live_agent_panel", None)
        assert panel is not None and not panel.s.enabled  # off by default
        w.select_tab("Live agent")
        root.update()
        assert "Use free models" in panel.status_line()
    finally:
        root.destroy()


def test_voice_order_prefers_fast_models_that_do_not_think_first() -> None:
    from away.liveagent.runtime import voice_order

    def m(mid: str, fast: bool, reasoning: bool) -> catalog.ModelInfo:
        info = catalog.guess("opencode_free", mid, free=True, may_keep_prompts=True)
        info.fast, info.reasoning = fast, reasoning
        return info

    models = [m("big-pickle", False, True), m("lightning-think", True, True), m("cat-jam-free", True, False)]
    assert [x.id for x in voice_order(models)] == ["cat-jam-free", "lightning-think", "big-pickle"]


def test_cli_ask_reports_every_attempt_when_no_model_answers(tmp_path, monkeypatch, capsys) -> None:
    from away.liveagent import __main__ as cli
    from away.liveagent import runtime
    from away.liveagent.router import NoModel

    class DeadFunnel:
        def stream(self, *_a: Any, **_k: Any) -> Any:
            raise NoModel("every model is unavailable right now", [("opencode_free/x", "rate_limit: 429 Free usage")])
            yield ""  # pragma: no cover - makes this a generator

    monkeypatch.setattr(runtime.Runtime, "funnel", lambda self, s: DeadFunnel())
    assert cli.main(["--data-dir", str(tmp_path), "ask", "hi"]) == 1
    err = capsys.readouterr().err
    assert "No model answered" in err and "opencode_free/x: rate_limit: 429 Free usage" in err


def test_watcher_sees_a_second_save_with_the_same_time_stamp(tmp_path) -> None:
    p = st.path(tmp_path)
    st.save(st.LiveSettings(enabled=False), p)
    w = st.Watcher(p)
    stamp = p.stat().st_mtime_ns
    st.save(st.LiveSettings(enabled=True), p)
    os.utime(p, ns=(stamp, stamp))  # coarse clocks (Windows) can give two quick saves the same time
    assert w.poll() and w.current.enabled is True
    assert not w.poll()
