"""Source identity trust: a request's ``cf-connecting-ip`` header counts only while the
Cloudflare tunnel is up.

The middleware keys three things on the request's source identity: the rate-limit bucket,
the per-source persistent-token backoff, and the source recorded in the access log and in
a token's ``last_source``. The server binds 127.0.0.1 only, so with no tunnel running the
header is whatever the caller claims: any local process, or any peer arriving over the SSH
backup relay (which forwards raw TCP), could mint a fresh bucket per request, reset a
tripped backoff by rotating the header, and poison the log. The header is honored only
when the app itself knows a Cloudflare tunnel is up (``Gate.cf_trusted``); Cloudflare's
edge sets it there, and the socket peer is the local cloudflared process for everyone, so
the header is the only per-source signal on that legitimate path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from away import config as C
from away.agentlink import api
from away.agentlink.host import AgentLinkHost, build_core
from away.agentlink.prompts import NullUI

PEER = "testclient"  # the socket peer Starlette's TestClient presents for every request
BAD = "paway_" + "A" * 43


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)


def make_core(tmp_path: Path):
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    ui = NullUI()
    return build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=ui)


def make_client(tmp_path: Path, rate: tuple[float, int] = (10.0, 40), cf_trusted: Any = None):
    core = make_core(tmp_path)
    gate = api.Gate(core.service, core.registry, local_key="k" * 43, rate=rate, access_log=tmp_path / "access.log")
    if cf_trusted is not None:
        gate.cf_trusted = cf_trusted
    app = api.create_app(core.service, gate, None)
    return core, gate, TestClient(app, base_url="http://127.0.0.1:8765", raise_server_exceptions=False)


# ------------------------------------------------------------------ unit: the identity rule
def test_source_ip_ignores_the_header_by_default(tmp_path: Path) -> None:
    core = make_core(tmp_path)
    gate = api.Gate(core.service, core.registry, local_key="k" * 43)
    scope = {"client": ("192.0.2.1", 5000)}
    assert gate.source_ip({"cf-connecting-ip": "203.0.113.9"}, scope) == "192.0.2.1"
    assert gate.source_ip({}, scope) == "192.0.2.1"


def test_source_ip_honors_the_header_only_when_cf_is_trusted(tmp_path: Path) -> None:
    core = make_core(tmp_path)
    scope = {"client": ("192.0.2.1", 5000)}
    gate = api.Gate(core.service, core.registry, local_key="k" * 43, cf_trusted=lambda: True)
    assert gate.source_ip({"cf-connecting-ip": "203.0.113.9"}, scope) == "203.0.113.9"
    assert gate.source_ip({}, scope) == "192.0.2.1"  # no header: the peer, as before

    def unknown() -> bool:
        raise RuntimeError("tunnel state unavailable")

    unsure = api.Gate(core.service, core.registry, local_key="k" * 43, cf_trusted=unknown)
    assert unsure.source_ip({"cf-connecting-ip": "203.0.113.9"}, scope) == "192.0.2.1"  # unknown: ignore


# ------------------------------------------------------------------ no tunnel: header is a claim, not an identity
def test_no_tunnel_rotating_header_shares_one_bucket(tmp_path: Path) -> None:
    _core, _gate, client = make_client(tmp_path, rate=(0.5, 3))
    codes = [client.get("/v1/ping", headers={"cf-connecting-ip": f"203.0.113.{i}"}).status_code for i in range(10)]
    # one shared bucket (burst 3): the first three pass, the rest are rate limited.
    # On the unfixed code every claimed address got its own bucket and all ten passed.
    assert codes[:3] == [200] * 3, codes
    assert all(c == 429 for c in codes[3:]), codes


def test_no_tunnel_backoff_is_shared_and_the_log_records_the_peer(tmp_path: Path) -> None:
    _core, gate, client = make_client(tmp_path)
    claimed = {"cf-connecting-ip": "198.51.100.7"}
    codes = [
        client.get("/v1/status", headers={"Authorization": f"Bearer {BAD}", **claimed}).status_code for _ in range(8)
    ]
    assert codes[-1] == 429, codes  # the source is in backoff after the free failures
    # claiming a fresh address does not reset the backoff: the identity is the socket peer
    fresh = client.get("/v1/status", headers={"Authorization": f"Bearer {BAD}", "cf-connecting-ip": "198.51.100.99"})
    assert fresh.status_code == 429, fresh.status_code
    # and neither the in-memory log nor the access log file ever records a claimed address
    assert gate.recent and all(item["ip"] == PEER for item in gate.recent), list(gate.recent)[-3:]
    text = (tmp_path / "access.log").read_text()
    assert "198.51.100" not in text and PEER in text


def test_no_tunnel_token_last_source_records_the_peer(tmp_path: Path) -> None:
    core, _gate, client = make_client(tmp_path)
    agent, _t = core.registry.issue("keeper", ["read"], 3600.0)
    rec, secret = core.registry.issue_persistent(agent.id, "90d")
    r = client.get("/v1/status", headers={"Authorization": f"Bearer {secret}", "cf-connecting-ip": "203.0.113.66"})
    assert r.status_code == 200, r.status_code
    assert core.registry.tokens[rec.id].last_source == PEER


# ------------------------------------------------------------------ tunnel up: the legitimate path still works
def test_tunnel_up_distinct_headers_get_distinct_buckets(tmp_path: Path) -> None:
    _core, _gate, client = make_client(tmp_path, rate=(0.5, 3), cf_trusted=lambda: True)
    codes = [client.get("/v1/ping", headers={"cf-connecting-ip": f"203.0.113.{i}"}).status_code for i in range(10)]
    assert codes == [200] * 10, codes  # each real remote source has its own allowance
    own = [client.get("/v1/ping").status_code for _ in range(5)]  # headerless requests share the peer bucket
    assert 429 in own, own


def test_tunnel_up_backoff_stays_per_claimed_source(tmp_path: Path) -> None:
    _core, gate, client = make_client(tmp_path, cf_trusted=lambda: True)
    claimed = {"cf-connecting-ip": "198.51.100.7"}
    codes = [
        client.get("/v1/status", headers={"Authorization": f"Bearer {BAD}", **claimed}).status_code for _ in range(8)
    ]
    assert codes[-1] == 429, codes
    # a different real source is not in backoff, and the log records the addresses Cloudflare reported
    other = client.get("/v1/status", headers={"Authorization": f"Bearer {BAD}", "cf-connecting-ip": "198.51.100.99"})
    assert other.status_code == 401, other.status_code
    assert any(item["ip"] == "198.51.100.99" for item in gate.recent), list(gate.recent)[-3:]


# ------------------------------------------------------------------ host wiring: what "tunnel up" means
class FakeLink:
    def __init__(self, url: str | None, alive: bool = True) -> None:
        self.url = url
        self._alive = alive

    def alive(self) -> bool:
        return self._alive


def bare_host() -> AgentLinkHost:
    h = object.__new__(AgentLinkHost)
    h.tunnel = None
    h.relay = None
    h.named = None
    return h


def test_host_cf_tunnel_up_counts_cloudflare_links_only() -> None:
    h = bare_host()
    assert h.cf_tunnel_up() is False
    h.tunnel = FakeLink("https://x.trycloudflare.com")
    assert h.cf_tunnel_up() is True  # the standard quick tunnel is Cloudflare
    h.tunnel = FakeLink("https://x.trycloudflare.com", alive=False)
    assert h.cf_tunnel_up() is False  # configured but down: nothing arrives through Cloudflare
    h.tunnel = None
    h.named = FakeLink("https://away.example.com")
    assert h.cf_tunnel_up() is True  # your own named Cloudflare tunnel too
    h.named = None
    h.relay = FakeLink("https://x.lhr.life")
    assert h.cf_tunnel_up() is False  # the SSH backup relay forwards raw TCP: no Cloudflare in the path
