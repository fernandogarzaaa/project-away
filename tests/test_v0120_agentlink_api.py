"""Agent Link over the wire: REST + OpenAPI, MCP Streamable HTTP (bearer and pairing link), the stdio bridge
that Claude Desktop / Cursor launch (``away mcp``), resource subscriptions, origin check, lockout, rate limit."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

from away import config as C
from away.agentlink import api, mcp_tools
from away.agentlink.host import AgentLinkHost, build_core
from away.agentlink.prompts import NullUI
from away.agentlink.server import Server

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy to 127.0.0.1


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


class AutoAnswer:
    def __init__(self, prompts, ui: NullUI, answer: str = "Allow") -> None:
        self.answer = answer
        self.seen: list[str] = []
        self._stop = threading.Event()

        def run() -> None:
            while not self._stop.is_set():
                cur = prompts.current
                if cur is not None and cur.id not in ui.clicks:
                    self.seen.append(cur.question)
                    ui.clicks[cur.id] = self.answer if self.answer in cur.options else cur.options[-1]
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


@pytest.fixture
def live(tmp_path: Path):
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    ui = NullUI()
    core = build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=ui)
    core.prompts.polite = lambda: True
    core.prompts.start()
    gate = api.Gate(
        core.service,
        core.registry,
        local_key="k" * 43,
        rate=(50.0, 60),
        access_log=tmp_path / "logs" / "agent-access.log",
    )
    srv = Server(api.create_app(core.service, gate, mcp_tools.build(api.LocalBackend(core.service))))
    port = srv.start()
    auto = AutoAnswer(core.prompts, ui)
    yield {"core": core, "gate": gate, "base": f"http://127.0.0.1:{port}", "auto": auto, "tmp": tmp_path}
    auto.stop()
    core.prompts.stop()
    srv.stop()


def http(base: str, method: str, path: str, body=None, token: str | None = None, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode() if body is not None else None, headers=h, method=method
    )
    try:
        with OPENER.open(req, timeout=30) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_rest_requires_a_token_and_pairing_needs_approval(live) -> None:
    base, core = live["base"], live["core"]
    assert http(base, "GET", "/v1/status")[0] == 401
    assert http(base, "GET", "/v1/status", token="pa_wrong")[0] == 401
    p = core.registry.new_pairing(["read", "tune"])
    st, body, _ = http(base, "POST", "/v1/pair", {"pin": p.pin, "name": "My Script"})
    assert st == 200, body
    data = json.loads(body)
    assert data["agent"]["permissions"] == ["See my gestures", "Change my settings"]
    assert live["auto"].seen[-1] == "My Script wants to connect"
    tok = data["token"]
    assert http(base, "GET", "/v1/status", token=tok)[0] == 200
    assert http(base, "POST", "/v1/pair", {"pin": p.pin, "name": "again"})[0] == 404  # single use
    st, body, _ = http(base, "POST", "/v1/rpc/look", {}, token=tok)
    assert st == 403 and b"See my camera" in body
    log = (live["tmp"] / "logs" / "agent-access.log").read_text()
    assert "My Script" in log and tok not in log


def test_denied_pairing_gives_nothing(live) -> None:
    live["auto"].answer = "Deny"
    p = live["core"].registry.new_pairing(["read"])
    st, body, _ = http(live["base"], "POST", "/v1/pair", {"code": p.code, "name": "x"})
    assert st == 403 and live["core"].registry.list() == []


def test_browser_origin_is_rejected_and_lockout_after_bad_tokens(live) -> None:
    base = live["base"]
    p = live["core"].registry.new_pairing(["read"])
    st, _b, _ = http(base, "POST", "/v1/pair", {"pin": p.pin}, headers={"Origin": "https://evil.example"})
    assert st == 403
    for _ in range(10):
        http(base, "GET", "/v1/status", token="pa_nope")
    assert http(base, "GET", "/openapi.json")[0] == 429  # locked for a minute, even public pages


def test_rate_limit(live) -> None:
    gate = live["gate"]
    from away.agentlink.ratelimit import TokenBucket

    gate.bucket = TokenBucket(1.0, 3)
    codes = [http(live["base"], "GET", "/")[0] for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[-1] == 429


def test_openapi_lists_the_tools(live) -> None:
    st, body, _ = http(live["base"], "GET", "/openapi.json")
    spec = json.loads(body)
    assert st == 200 and spec["openapi"].startswith("3.")
    assert "/v1/ask" in spec["paths"] and "/v1/control/act" in spec["paths"] and "/v1/pair" in spec["paths"]


def test_rest_files_and_traversal(live) -> None:
    core, base = live["core"], live["base"]
    a, tok = core.registry.issue("Diag", ["read", "diagnose", "share_logs"], 3600)
    st, body, _ = http(base, "POST", "/v1/bundle", {}, token=tok)
    assert st == 200
    path = json.loads(body)["path"]
    st, body, h = http(base, "GET", f"/v1/file?rel={path}", token=tok)
    assert st == 200 and body[:2] == b"PK"
    for bad in ("../config.yaml", "diagnostics/../../config.yaml", "/etc/passwd"):
        st, _b, _ = http(base, "GET", f"/v1/file?rel={urllib.request.quote(bad)}", token=tok)
        assert st == 400


async def _mcp(url: str, headers: dict[str, str] | None, fn):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    notes: list = []

    async def handler(msg) -> None:
        notes.append(msg)

    async with (
        streamablehttp_client(url, headers=headers) as (r, w, _),
        ClientSession(r, w, message_handler=handler) as s,
    ):
        init = await s.initialize()
        return await fn(s, init, notes)


def test_mcp_over_http_with_bearer_token(live) -> None:
    core = live["core"]
    a, tok = core.registry.issue("Claude", ["read"], 3600)

    async def body(s, init, notes):
        assert "bridge" in (init.instructions or "")
        names = {t.name for t in (await s.list_tools()).tools}
        # v0.1.21: 15 grouped tools (was 37); the old names still work on REST and /v1/rpc. v0.1.22: + inbox;
        # v0.1.28: + ui_query, + shell (18)
        assert names == {
            "inbox",
            "get_status",
            "get_context",
            "events",
            "settings",
            "preferences",
            "diagnostics",
            "look",
            "get_screen",
            "ui_query",
            "ask_user",
            "notify_user",
            "control",
            "pointer",
            "keyboard",
            "open",
            "shell",
            "disconnect",
        }
        ctx = json.loads((await s.call_tool("get_context", {})).content[0].text)
        assert "attention" in ctx
        denied = await s.call_tool("look", {})
        assert denied.isError and "See my camera" in denied.content[0].text
        about = await s.read_resource("away://about")
        assert "Live context" in about.contents[0].text
        res = await s.call_tool("notify_user", {"text": "hello"})
        assert not res.isError
        return True

    assert asyncio.run(_mcp(f"{live['base']}/mcp", {"Authorization": f"Bearer {tok}"}, body))
    ui = live["core"].prompts.ui
    assert any(t == "hello" for _l, t in ui.toasts) and "Message from Claude" in ui.titles


def test_mcp_without_token_is_refused(live) -> None:
    async def body(s, init, notes):
        return True

    with pytest.raises(BaseException):
        asyncio.run(_mcp(f"{live['base']}/mcp", None, body))


def test_pairing_link_works_as_a_connector_url(live) -> None:
    core = live["core"]
    p = core.registry.new_pairing(["read"])
    url = f"{live['base']}/c/{p.code}"
    st, html, _ = http(live["base"], "GET", f"/c/{p.code}", headers={"Accept": "text/html"})
    assert st == 200 and b"Allow" in html and core.registry.find_pairing(code=p.code) is not None  # page != claim

    async def body(s, init, notes):
        return (await s.call_tool("get_status", {})).isError

    assert asyncio.run(_mcp(url, None, body)) is False
    assert live["auto"].seen[-1].endswith("wants to connect")
    assert [a.name for a in core.registry.list()] == ["AI agent"]
    assert asyncio.run(_mcp(url, None, body)) is False  # the same link keeps working for that agent
    core.registry.revoke(core.registry.list()[0].id)
    with pytest.raises(BaseException):
        asyncio.run(_mcp(url, None, body))


def test_events_subscription_pushes_updates(live) -> None:
    core = live["core"]
    a, tok = core.registry.issue("Claude", ["read"], 3600)

    async def body(s, init, notes):
        from pydantic import AnyUrl

        assert init.capabilities.resources and init.capabilities.resources.subscribe
        await s.subscribe_resource(AnyUrl("away://events"))
        await asyncio.sleep(0.5)
        core.events.publish("summon", {"source": "test"})
        for _ in range(50):
            if any("resources/updated" in str(getattr(getattr(n, "root", n), "method", "")) for n in notes):
                break
            await asyncio.sleep(0.1)
        res = await s.call_tool("events", {"action": "wait", "since": 0, "timeout_s": 1})
        return notes, json.loads(res.content[0].text)

    notes, events = asyncio.run(_mcp(f"{live['base']}/mcp", {"Authorization": f"Bearer {tok}"}, body))
    assert any("resources/updated" in str(getattr(getattr(n, "root", n), "method", "")) for n in notes)
    assert any(e["topic"] == "summon" for e in events["events"])


def test_stdio_bridge_like_claude_desktop(tmp_path: Path) -> None:
    """Launch `python -m away mcp` exactly as Claude Desktop would and talk MCP over its stdin/stdout."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    data = tmp_path / "data"
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    env_keep = {
        k: os.environ[k]
        for k in ("PATH", "SYSTEMROOT", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA")
        if k in os.environ
    }
    old = {k: os.environ.get(k) for k in ("AWAY_DATA_DIR", "AWAY_CONFIG")}
    os.environ["AWAY_DATA_DIR"], os.environ["AWAY_CONFIG"] = str(data), str(cfgp)
    ui = NullUI()
    host = AgentLinkHost(None, data_dir=data, ui=ui, headless=True)
    try:
        host.core.prompts.polite = lambda: True
        host.start()
        auto = AutoAnswer(host.core.prompts, ui)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "away", "mcp", "--scopes", "read,tune"],
            env={**env_keep, "AWAY_DATA_DIR": str(data), "AWAY_CONFIG": str(cfgp), "NO_PROXY": "127.0.0.1"},
        )

        async def run():
            async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
                init = await s.initialize()
                raw = (await s.call_tool("get_status", {})).content[0].text
                assert raw.startswith("{"), raw
                st = json.loads(raw)
                prefs = await s.call_tool("preferences", {"action": "add", "text": "slower cursor please"})
                return init, st, prefs

        init, st, prefs = asyncio.run(asyncio.wait_for(run(), 90))
        auto.stop()
        assert "bridge" in (init.instructions or "")
        assert st["agent"]["kind"] == "local" and st["agent"]["scopes"] == ["read", "tune"]
        assert any(q.endswith("wants to connect") for q in auto.seen)  # approved on the PC first
        assert not prefs.isError
        tokens = list((data / "agentlink" / "local-agents").glob("*.json"))
        assert len(tokens) == 1  # remembered for next time
    finally:
        host.stop()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_bridge_says_when_the_app_is_not_running(tmp_path: Path) -> None:
    from away.agentlink.bridge import NOT_RUNNING, RestBackend

    b = RestBackend("read", "Test", data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="isn't running"):
        b.call(None, "get_status")
    assert "Start it" in NOT_RUNNING


def test_cli_help_lists_agent_commands() -> None:
    out = subprocess.run([sys.executable, "-m", "away", "--help"], capture_output=True, text=True, timeout=60).stdout
    assert "agent" in out and "mcp" in out and "replay" in out and "agent-ui" not in out
