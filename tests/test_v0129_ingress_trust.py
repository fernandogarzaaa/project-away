"""v0.1.29 request-specific ingress trust: a request's listener proves its ingress.

Trust rule: ingress is "tunnel" exactly when the request arrived on the dedicated Cloudflare listener, else
"shared". The cf-connecting-ip header is honored only on tunnel ingress, and only when it is one IP literal.
On shared ingress the header is ignored entirely, so a local process or relay client cannot pick its own
rate-limit identity. Rate buckets, per-source backoff, and lockout keys are qualified by ingress, so a lockout
on one ingress never blocks the other. Valid credentials pass during their own ingress's lockout and clear it.
Relay traffic arrives on the shared listener, so relay users share the shared key by design.

Flows run through the real middleware over TestClient. TestClient derives scope["server"] from the base_url
port, so the shared client (port 8765) and the tunnel client (port 9123) exercise the two listeners.
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

from away import config as C
from away.agentlink import api
from away.agentlink.auth import PERSIST_PREFIX
from away.agentlink.host import AgentLinkHost, build_core
from away.agentlink.prompts import NullUI
from away.agentlink.ratelimit import FailureLockout

MAIN_PORT = 8765
TUNNEL_PORT = 9123


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def _core(tmp_path: Path) -> Any:
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    return build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=NullUI())


def _tokens(core: Any) -> tuple[str, str]:
    p = core.registry.new_pairing(["read", "tune"])
    assert core.registry.mark_pending(p, "Test agent")
    agent, token = core.registry.approve(p, 3600.0)
    _rec, persistent = core.registry.issue_persistent(agent.id, "90d")
    return token, persistent


def _gate(core: Any, clk: FakeClock, tunnel: dict[str, Any]) -> Any:
    return api.Gate(
        core.service,
        core.registry,
        local_key="k" * 43,
        rate=(100.0, 200),
        clock=clk,
        tunnel_port=lambda: tunnel["port"],
    )


def _inner() -> Any:
    async def ok(request: Any) -> Any:
        state = request.scope.get("state", {})
        return JSONResponse(
            {"ok": True, "path": request.url.path, "ingress": state.get("ingress"), "key": state.get("ingress_key")}
        )

    return Starlette(
        routes=[
            Route("/", ok, methods=["GET"]),
            Route("/v1/{rest:path}", ok, methods=["GET", "POST", "DELETE"]),
        ]
    )


@pytest.fixture
def env(tmp_path: Path) -> Any:
    core = _core(tmp_path)
    clk = FakeClock()
    tunnel = {"port": None}
    gate = _gate(core, clk, tunnel)
    token, persistent = _tokens(core)
    app = _inner()
    shared = TestClient(api.AuthMiddleware(app, gate), base_url=f"http://127.0.0.1:{MAIN_PORT}")
    tunnel_client = TestClient(api.AuthMiddleware(app, gate), base_url=f"http://127.0.0.1:{TUNNEL_PORT}")
    return types.SimpleNamespace(
        core=core,
        clk=clk,
        tunnel=tunnel,
        gate=gate,
        token=token,
        persistent=persistent,
        shared=shared,
        tun=tunnel_client,
    )


@pytest.fixture
def full(tmp_path: Path) -> Any:
    core = _core(tmp_path)
    clk = FakeClock()
    tunnel = {"port": None}
    gate = _gate(core, clk, tunnel)
    app = api.create_app(core.service, gate, None)
    shared = TestClient(api.AuthMiddleware(app, gate), base_url=f"http://127.0.0.1:{MAIN_PORT}")
    tunnel_client = TestClient(api.AuthMiddleware(app, gate), base_url=f"http://127.0.0.1:{TUNNEL_PORT}")
    return types.SimpleNamespace(core=core, clk=clk, tunnel=tunnel, gate=gate, shared=shared, tun=tunnel_client)


def _status(client: Any, token: str, headers: dict[str, str] | None = None) -> Any:
    return client.get("/v1/status", headers={"Authorization": f"Bearer {token}", **(headers or {})})


def _last_ip(env: Any) -> str:
    return str(env.gate.recent[-1]["ip"])


def _tunnel_up(env: Any) -> None:
    env.tunnel["port"] = TUNNEL_PORT


# ============================================================================ ingress classification
def test_forged_header_ignored_when_tunnel_down(env: Any) -> None:
    r = _status(env.shared, env.token, {"cf-connecting-ip": "203.0.113.7"})
    assert r.status_code == 200
    assert _last_ip(env) == "shared:testclient"
    assert env.gate.ingress_source(
        {"client": ("testclient", 50000), "server": ["127.0.0.1", MAIN_PORT]}, {"cf-connecting-ip": "203.0.113.7"}
    ) == (
        "shared",
        "testclient",
        "shared:testclient",
    )


def test_forged_header_ignored_on_shared_listener_while_tunnel_up(env: Any) -> None:
    _tunnel_up(env)
    assert _status(env.shared, env.token, {"cf-connecting-ip": "203.0.113.7"}).status_code == 200
    assert _last_ip(env) == "shared:testclient"
    assert _status(env.tun, env.token, {"cf-connecting-ip": "203.0.113.7"}).status_code == 200
    assert _last_ip(env) == "tunnel:203.0.113.7"


def test_valid_header_honored_on_tunnel_listener(env: Any) -> None:
    _tunnel_up(env)
    r = _status(env.tun, env.token, {"cf-connecting-ip": "203.0.113.7"})
    assert r.status_code == 200 and r.json()["ingress"] == "tunnel" and r.json()["key"] == "tunnel:203.0.113.7"
    assert _last_ip(env) == "tunnel:203.0.113.7"


def test_each_ingress_classified_direct_tunnel_and_relay(env: Any) -> None:
    _tunnel_up(env)
    assert _status(env.shared, env.token).json()["ingress"] == "shared"  # a direct local request
    assert _status(env.tun, env.token).json()["ingress"] == "tunnel"  # Cloudflare traffic, no header
    # the relay path is the main port, so relay traffic is shared and its header is ignored
    assert _status(env.shared, env.token, {"cf-connecting-ip": "198.51.100.7"}).json()["ingress"] == "shared"
    assert _last_ip(env) == "shared:testclient"


def test_missing_malformed_and_conflicting_headers_on_both_ingresses(env: Any) -> None:
    _tunnel_up(env)
    cases = [
        (env.tun, {}, "tunnel:testclient"),  # missing header
        (env.shared, {}, "shared:testclient"),
        (env.tun, {"cf-connecting-ip": ""}, "tunnel:testclient"),  # empty header
        (env.tun, {"cf-connecting-ip": "not-an-ip"}, "tunnel:testclient"),  # garbage
        (env.tun, {"cf-connecting-ip": "203.0.113.7, 198.51.100.1"}, "tunnel:testclient"),  # a list, not one IP
        (env.tun, {"cf-connecting-ip": "  203.0.113.7  "}, "tunnel:203.0.113.7"),  # surrounding space is stripped
        # x-forwarded-for is never an identity, on either ingress
        (
            env.tun,
            {"cf-connecting-ip": "203.0.113.7", "x-forwarded-for": "198.51.100.1"},
            "tunnel:203.0.113.7",
        ),
        (
            env.shared,
            {"cf-connecting-ip": "203.0.113.7", "x-forwarded-for": "198.51.100.1"},
            "shared:testclient",
        ),
    ]
    for client, headers, want in cases:
        assert _status(client, env.token, headers).status_code == 200
        assert _last_ip(env) == want, (headers, want)


def test_duplicate_forwarding_header_uses_the_last_one_on_tunnel_only(env: Any) -> None:
    _tunnel_up(env)
    r = env.tun.get(
        "/v1/status",
        headers=[
            ("Authorization", f"Bearer {env.token}"),
            ("cf-connecting-ip", "198.51.100.1"),
            ("cf-connecting-ip", "203.0.113.7"),
        ],
    )
    assert r.status_code == 200 and _last_ip(env) == "tunnel:203.0.113.7"
    r = env.shared.get(
        "/v1/status",
        headers=[
            ("Authorization", f"Bearer {env.token}"),
            ("cf-connecting-ip", "198.51.100.1"),
            ("cf-connecting-ip", "203.0.113.7"),
        ],
    )
    assert r.status_code == 200 and _last_ip(env) == "shared:testclient"


def test_ipv4_and_ipv6_header_forms(env: Any) -> None:
    _tunnel_up(env)
    cases = [
        ("203.0.113.7", "tunnel:203.0.113.7"),
        ("2001:db8::1", "tunnel:2001:db8::1"),
        ("2001:0db8:0000:0000:0000:0000:0000:0001", "tunnel:2001:db8::1"),  # expanded form normalizes
        ("::1", "tunnel:::1"),
        ("192.168.001.001", "tunnel:testclient"),  # leading zeros are not an IP literal
    ]
    for header, want in cases:
        assert _status(env.tun, env.token, {"cf-connecting-ip": header}).status_code == 200
        assert _last_ip(env) == want, (header, want)
    # unit level: an IPv6 peer on shared ingress keeps its own key
    assert env.gate.ingress_source({"client": ("2001:db8::2", 50000), "server": ["127.0.0.1", MAIN_PORT]}, {}) == (
        "shared",
        "2001:db8::2",
        "shared:2001:db8::2",
    )


def test_ingress_source_unit_matrix(env: Any) -> None:
    g = env.gate
    assert g.ingress_source({"client": ("peer", 1), "server": ["127.0.0.1", MAIN_PORT]}, {}) == (
        "shared",
        "peer",
        "shared:peer",
    )
    _tunnel_up(env)
    assert g.ingress_source({"client": ("peer", 1), "server": ["127.0.0.1", TUNNEL_PORT]}, {}) == (
        "tunnel",
        "peer",
        "tunnel:peer",
    )
    assert g.ingress_source({"client": ("peer", 1)}, {"cf-connecting-ip": "203.0.113.7"}) == (
        "shared",
        "peer",
        "shared:peer",
    )  # no server entry: shared, never trusted
    g.tunnel_port = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # a failing callable fails closed
    assert g.ingress_source({"client": ("peer", 1), "server": ["127.0.0.1", TUNNEL_PORT]}, {})[0] == "shared"


# ============================================================================ lockout and backoff isolation
def _lock_shared(env: Any) -> None:
    codes = [_status(env.shared, "bad-token").status_code for _ in range(10)]
    assert codes == [401] * 10
    assert env.gate.lockout.locked("shared:testclient")


def test_shared_lockout_blocks_neither_tunnel_nor_valid_shared_credentials(env: Any) -> None:
    _tunnel_up(env)
    _lock_shared(env)
    assert env.shared.get("/").status_code == 429  # no credential: refused outright while locked
    assert _status(env.tun, env.token).status_code == 200  # the tunnel key was never locked
    assert _status(env.shared, env.token).status_code == 200  # a valid credential passes and clears its key
    assert env.shared.get("/").status_code == 200  # the lockout is gone after the valid credential
    assert _status(env.shared, "bad-token").status_code == 401  # failures start from zero again


def test_valid_credentials_pass_while_another_source_is_locked_out(env: Any) -> None:
    _tunnel_up(env)
    codes = [_status(env.tun, "bad-token", {"cf-connecting-ip": "198.51.100.7"}).status_code for _ in range(10)]
    assert codes == [401] * 10
    assert env.gate.lockout.locked("tunnel:198.51.100.7")
    assert _status(env.shared, env.token).status_code == 200
    assert _status(env.tun, env.token, {"cf-connecting-ip": "203.0.113.9"}).status_code == 200
    assert _status(env.tun, env.token, {"cf-connecting-ip": "198.51.100.7"}).status_code == 200


def test_lockout_expiry_and_recovery_with_fake_clock(env: Any) -> None:
    _lock_shared(env)
    assert env.shared.get("/").status_code == 429
    env.clk.t += 61  # past the 60 second lock, with no sleep
    assert env.shared.get("/").status_code == 200
    assert _status(env.shared, "bad-token").status_code == 401  # the old failures aged out of the window
    assert _status(env.shared, env.token).status_code == 200


def test_tunnel_source_backoff_is_per_source(env: Any) -> None:
    _tunnel_up(env)
    bad = PERSIST_PREFIX + "x" * 43
    attacker = {"cf-connecting-ip": "198.51.100.7"}
    codes = [_status(env.tun, bad, attacker).status_code for _ in range(6)]
    assert codes == [401] * 6
    st = _status(env.tun, bad, attacker)
    assert st.status_code == 429 and int(st.headers["retry-after"]) >= 1
    assert _status(env.tun, env.persistent, attacker).status_code == 429  # the right token waits on that source
    assert _status(env.tun, env.persistent, {"cf-connecting-ip": "203.0.113.9"}).status_code == 200
    assert _status(env.shared, env.persistent).status_code == 200  # shared was never backed off


def test_shared_backoff_ignores_the_header(env: Any) -> None:
    bad = PERSIST_PREFIX + "x" * 43
    codes = [_status(env.shared, bad, {"cf-connecting-ip": "198.51.100.7"}).status_code for _ in range(6)]
    assert codes == [401] * 6
    assert _status(env.shared, bad, {"cf-connecting-ip": "198.51.100.7"}).status_code == 429
    # a different forged header on the same shared listener is the same source, so it waits too
    assert _status(env.shared, env.persistent, {"cf-connecting-ip": "203.0.113.9"}).status_code == 429
    assert _status(env.shared, env.persistent).status_code == 429


def test_failure_lockout_keys_are_independent() -> None:
    clk = FakeClock()
    lock = FailureLockout(3, 60, 30, clk)
    for _ in range(3):
        lock.fail("a")
    assert lock.locked("a") and not lock.locked("b") and not lock.locked()
    lock.reset("a")
    assert not lock.locked("a")
    for _ in range(3):
        lock.fail("b")
    assert lock.locked("b")
    lock.reset()
    assert not lock.locked("b")
    for _ in range(3):
        lock.fail("c")
    assert lock.locked("c")
    clk.t += 31
    assert not lock.locked("c")  # the lock expired without a sleep
    assert not lock.locked("a")


# ============================================================================ handler-level keying (full app)
def _bad_pair(client: Any) -> int:
    return client.post(
        "/v1/pair",
        content=json.dumps({"code": "nope", "pin": "000000"}),
        headers={"Content-Type": "application/json"},
    ).status_code


def test_pair_failures_on_shared_do_not_lock_tunnel(full: Any) -> None:
    full.tunnel["port"] = TUNNEL_PORT
    assert [_bad_pair(full.shared) for _ in range(10)] == [404] * 10
    assert full.shared.get("/").status_code == 429
    assert full.tun.get("/").status_code == 200


def test_pair_failures_on_tunnel_do_not_lock_shared(full: Any) -> None:
    full.tunnel["port"] = TUNNEL_PORT
    assert [_bad_pair(full.tun) for _ in range(10)] == [404] * 10
    assert full.tun.get("/").status_code == 429
    assert full.shared.get("/").status_code == 200


# ============================================================================ tunnel listener lifecycle (host helper)
class _FakeServer:
    def __init__(self, app: Any, port: int = 0) -> None:
        self.app = app
        self.port = 45678
        self.started = False
        self.stopped = False

    @property
    def running(self) -> bool:
        return self.started and not self.stopped

    def start(self, timeout: float = 15.0) -> int:
        self.started = True
        return self.port

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True


def test_tunnel_server_starts_reuses_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import away.agentlink.server as server_mod

    monkeypatch.setattr(server_mod, "Server", _FakeServer)
    h = AgentLinkHost(None, data_dir=tmp_path, ui=NullUI(), headless=True)
    h.server = types.SimpleNamespace(app=object(), running=True)
    assert h.tunnel_server_port() is None
    assert h._ensure_tunnel_server() == 45678
    assert h._ensure_tunnel_server() == 45678  # reuse, no second listener
    assert h.tunnel_server_port() == 45678
    h.stop_tunnel()
    assert h.tunnel_server is None and h.tunnel_server_port() is None


def test_tunnel_server_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import away.agentlink.server as server_mod

    class _FailingServer(_FakeServer):
        def start(self, timeout: float = 15.0) -> int:
            raise RuntimeError("bind failed")

    monkeypatch.setattr(server_mod, "Server", _FailingServer)
    h = AgentLinkHost(None, data_dir=tmp_path, ui=NullUI(), headless=True)
    h.server = types.SimpleNamespace(app=object(), running=True)
    with pytest.raises(RuntimeError, match="bind failed"):
        h._ensure_tunnel_server()
    assert h.tunnel_server is None and h.tunnel_server_port() is None


def test_tunnel_server_refuses_without_a_main_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import away.agentlink.server as server_mod

    monkeypatch.setattr(server_mod, "Server", _FakeServer)
    h = AgentLinkHost(None, data_dir=tmp_path, ui=NullUI(), headless=True)
    with pytest.raises(RuntimeError, match="not running"):
        h._ensure_tunnel_server()
