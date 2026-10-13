"""v0.1.23 security: the Agent Link HTTP server refuses what a web page in the person's browser could send.

* DNS rebinding: ``Host: evil.example`` (resolving to 127.0.0.1) is refused; only localhost / 127.0.0.1 / [::1]
  (any port) and the active secure-link / backup-link hostnames are answered.
* Cross-site requests: a state-changing request with ``Origin: null`` or a foreign origin is refused.
* Form posts: POST/PUT/PATCH must be ``application/json`` (no ``text/plain`` / form bodies, which need no preflight).
* The bearer token stays required.
Plus the ``User-Agent: ProjectAway/<version>`` sent to OpenAI-compatible LLM servers (Groq answers 403 without one).
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from away import __version__
from away import config as C
from away.agentlink import api
from away.agentlink.host import AgentLinkHost, build_core
from away.agentlink.prompts import NullUI


# ============================================================================ pure checks
@pytest.mark.parametrize(
    ("value", "host"),
    [
        ("127.0.0.1:8765", "127.0.0.1"),
        ("LOCALHOST", "localhost"),
        ("[::1]:9000", "::1"),
        ("::1", "::1"),
        ("abc.lhr.life", "abc.lhr.life"),
        ("", ""),
    ],
)
def test_host_without_port(value: str, host: str) -> None:
    assert api.host_without_port(value) == host


def test_host_allowed_only_for_this_pc_and_the_active_links() -> None:
    for ok in ("127.0.0.1", "127.0.0.1:51234", "localhost:1", "[::1]:80", "::1"):
        assert api.host_allowed(ok), ok
    for bad in ("evil.example", "evil.example:51234", "0.0.0.0:80", "192.168.1.10", "127.0.0.1.evil.example", ""):
        assert not api.host_allowed(bad), bad
    assert api.host_allowed("abc.trycloudflare.com", ["abc.trycloudflare.com"])
    assert api.host_allowed("ABC.trycloudflare.com:443", ["abc.trycloudflare.com"])
    assert not api.host_allowed("other.trycloudflare.com", ["abc.trycloudflare.com"])


def test_hostname_of_tunnel_urls() -> None:
    assert api.hostname_of("https://abc.lhr.life") == "abc.lhr.life"
    assert api.hostname_of("https://x.trycloudflare.com/c/123") == "x.trycloudflare.com"
    assert api.hostname_of("https://user@a.pinggy.link:443/") == "a.pinggy.link"
    assert api.hostname_of(None) == "" and api.hostname_of("") == ""


def test_origin_must_be_this_server_itself() -> None:
    host = "127.0.0.1:8765"
    assert api.origin_allowed("http://127.0.0.1:8765", host)
    assert not api.origin_allowed("null", host)  # sandboxed iframe / file: page
    assert not api.origin_allowed("https://evil.example", host)
    assert not api.origin_allowed("http://localhost:8765", host)  # a different origin than the Host it reached
    assert not api.origin_allowed("http://127.0.0.1:9999", host)  # another local web server
    assert not api.origin_allowed("http://user@127.0.0.1:8765", host)
    assert not api.origin_allowed("file://", host)
    assert api.origin_allowed("https://abc.lhr.life", "abc.lhr.life", ["abc.lhr.life"])
    assert not api.origin_allowed("https://abc.lhr.life", "abc.lhr.life")  # link no longer active


def test_browser_guard_verdicts() -> None:
    js = {"host": "127.0.0.1:1", "content-type": "application/json"}
    assert api.browser_guard("POST", js) is None
    assert api.browser_guard("POST", {**js, "host": "evil.example"}) == "host"
    assert api.browser_guard("GET", {"host": "evil.example"}) == "host"  # rebinding can read too
    assert api.browser_guard("POST", {**js, "origin": "null"}) == "origin"
    assert api.browser_guard("DELETE", {"host": "127.0.0.1:1", "origin": "https://evil.example"}) == "origin"
    assert api.browser_guard("GET", {"host": "127.0.0.1:1", "origin": "https://evil.example"}) is None
    assert api.browser_guard("POST", {**js, "content-type": "text/plain"}) == "content-type"
    assert api.browser_guard("POST", {**js, "content-type": "application/x-www-form-urlencoded"}) == "content-type"
    assert api.browser_guard("POST", {"host": "127.0.0.1:1"}) == "content-type"
    assert api.browser_guard("POST", {**js, "content-type": "application/json; charset=utf-8"}) is None
    assert api.browser_guard("DELETE", {"host": "127.0.0.1:1"}) is None  # MCP session close has no body
    assert api.browser_guard("POST", {"content-type": "application/json"}) is None  # no Host: not a browser


# ============================================================================ through the middleware
@pytest.fixture
def guarded(tmp_path: Path) -> Any:
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    core = build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=NullUI())
    links: list[str] = []
    gate = api.Gate(core.service, core.registry, local_key="k" * 43, rate=(100.0, 200), tunnel_hosts=lambda: links)

    async def ok(request: Any) -> Any:
        return JSONResponse({"ok": True, "path": request.url.path})

    inner = Starlette(routes=[Route("/v1/{rest:path}", ok, methods=["GET", "POST", "DELETE"])])
    client = TestClient(api.AuthMiddleware(inner, gate), base_url="http://127.0.0.1:8765")
    p = core.registry.new_pairing(["read", "tune"])
    _agent, token = core.registry.approve(p, 3600.0)
    return types.SimpleNamespace(client=client, token=token, links=links, gate=gate)


def _post(g: Any, path: str = "/v1/rpc/status", **headers: str) -> Any:
    h = {"Authorization": f"Bearer {g.token}", "Content-Type": "application/json", **headers}
    return g.client.post(path, content=json.dumps({}), headers=h)


def test_a_normal_local_agent_request_still_works(guarded: Any) -> None:
    assert _post(guarded).status_code == 200
    r = guarded.client.get("/v1/status", headers={"Authorization": f"Bearer {guarded.token}"})
    assert r.status_code == 200


def test_the_bearer_token_is_still_required(guarded: Any) -> None:
    assert guarded.client.get("/v1/status").status_code == 401
    r = guarded.client.post("/v1/rpc/status", content=b"{}", headers={"Content-Type": "application/json"})
    assert r.status_code == 401


def test_dns_rebinding_host_is_refused_even_with_a_token(guarded: Any) -> None:
    r = _post(guarded, Host="evil.example")
    assert r.status_code == 403 and "Host" in r.json()["error"]
    r = guarded.client.get("/v1/status", headers={"Host": "evil.example:8765"})
    assert r.status_code == 403


def test_the_active_tunnel_hostname_is_accepted_and_dropped_when_it_ends(guarded: Any) -> None:
    assert _post(guarded, Host="abc.lhr.life").status_code == 403
    guarded.links.append("abc.lhr.life")
    assert _post(guarded, Host="abc.lhr.life").status_code == 200
    guarded.links.clear()
    assert _post(guarded, Host="abc.lhr.life").status_code == 403


def test_origin_null_and_foreign_origins_are_refused(guarded: Any) -> None:
    assert _post(guarded, Origin="null").status_code == 403
    assert _post(guarded, Origin="https://evil.example").status_code == 403
    assert _post(guarded, Origin="http://127.0.0.1:8765").status_code == 200  # this server's own page


def test_text_plain_and_form_posts_are_refused(guarded: Any) -> None:
    for ct in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"):
        r = _post(guarded, **{"Content-Type": ct})
        assert r.status_code == 415, ct
    r = guarded.client.post("/v1/pair", content=b'{"pin": "123456"}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 415  # pairing is public, but not form-postable


def test_host_names_the_active_secure_and_backup_links(tmp_path: Path) -> None:
    host = AgentLinkHost.__new__(AgentLinkHost)
    host.tunnel = types.SimpleNamespace(url="https://abc.trycloudflare.com")
    host.relay = types.SimpleNamespace(url=None)
    assert host.tunnel_hosts() == ["abc.trycloudflare.com"]
    host.relay = types.SimpleNamespace(url="https://x1.lhr.life")
    assert host.tunnel_hosts() == ["abc.trycloudflare.com", "x1.lhr.life"]
    host.tunnel = host.relay = None
    assert host.tunnel_hosts() == []


# ============================================================================ LLM User-Agent
class _Resp:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: Any) -> None:
        return None


def test_openai_compatible_backend_sends_a_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.voice import llm_backends as L

    seen: list[Any] = []

    def fake_open(req: Any, timeout: float) -> Any:
        seen.append(req)
        return _Resp({"choices": [{"message": {"content": " ok "}}]})

    monkeypatch.setattr(L, "_open", fake_open)
    b = L.OpenAIBackend("https://api.groq.com/openai/v1", "llama", "gsk_x")
    assert b.complete("sys", "hi", 10, 5.0) == "ok"
    req = seen[0]
    assert req.get_header("User-agent") == f"ProjectAway/{__version__}"
    assert req.get_header("Authorization") == "Bearer gsk_x"
    assert L.user_agent() == f"ProjectAway/{__version__}"


def test_cleanup_and_edit_servers_get_the_user_agent_too() -> None:
    from away.voice import cleanup, edit

    for mod in (cleanup, edit):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert '"User-Agent": user_agent()' in src, mod.__name__
