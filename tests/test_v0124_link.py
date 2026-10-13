"""v0.1.24 "a link that survives restarts": hashed refresh tokens (rotation, grace, expiry, revocation, restart),
connector URLs that renew themselves, the address published to every paired agent (wake-up address, inbox, signed
rendezvous), the host bringing the link back, and scripts/agent_link_client.py following a moved address."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from away import config as C
from away.agentlink import rendezvous
from away.agentlink.auth import REFRESH_GRACE_S, AgentRegistry, RefreshError
from away.agentlink.inbox import Inbox

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def reg(tmp_path: Path, clock: Clock, days: float = 30) -> AgentRegistry:
    return AgentRegistry(tmp_path / "agents.json", clock=clock, refresh_ttl_s=days * 86400, link_access_s=3600)


# ------------------------------------------------------------------------------------------- refresh tokens
def test_refresh_rotates_and_survives_a_restart_with_only_hashes_on_disk(tmp_path: Path) -> None:
    clk = Clock()
    r = reg(tmp_path, clk)
    a, tok = r.issue("Hark", ["read", "diagnose"], 3600)
    rt = r.new_refresh(a.id)
    assert rt.startswith("par_") and a.rdv_id and a.rdv_key
    raw = (tmp_path / "agents.json").read_text()
    assert tok not in raw and rt not in raw  # only SHA-256 hashes are kept
    clk.t += 2 * 3600  # access expired (and Away restarted)
    r2 = reg(tmp_path, clk)
    assert r2.authenticate(tok) is None
    agent, access, fresh = r2.refresh(rt, 3600)
    assert agent.id == a.id and agent.scopes == ["read", "diagnose"]
    assert r2.authenticate(access) is not None and fresh != rt
    # the old refresh token still works briefly (a reply lost on the way), then never again
    r2.refresh(rt, 3600)
    clk.t += REFRESH_GRACE_S + 1
    with pytest.raises(RefreshError, match="already used"):
        r2.refresh(rt, 3600)


def test_refresh_expiry_window_moves_with_use_and_ends(tmp_path: Path) -> None:
    clk = Clock()
    r = reg(tmp_path, clk, days=30)
    a, _ = r.issue("Muse", ["read"], 3600)
    rt = r.new_refresh(a.id)
    clk.t += 29 * 86400
    _a, _tok, rt = r.refresh(rt, 3600)  # used in time: 30 more days
    clk.t += 29 * 86400
    _a, _tok, rt = r.refresh(rt, 3600)
    clk.t += 31 * 86400
    with pytest.raises(RefreshError, match="expired"):
        r.refresh(rt, 3600)


def test_revoking_in_settings_kills_the_refresh_token(tmp_path: Path) -> None:
    clk = Clock()
    r = reg(tmp_path, clk)
    a, _ = r.issue("Hark", ["read"], 3600)
    rt = r.new_refresh(a.id)
    assert r.revoke(a.id)
    with pytest.raises(RefreshError):
        r.refresh(rt, 3600)
    assert a not in r.reconnectable()
    assert '"refresh_hash": ""' in (tmp_path / "agents.json").read_text().replace("\n", "")


def test_garbage_refresh_tokens_are_refused(tmp_path: Path) -> None:
    r = reg(tmp_path, Clock())
    for bad in (None, "", "pa_notarefresh", "par_unknown"):
        with pytest.raises(RefreshError):
            r.refresh(bad, 3600)


def test_reconnectable_agents_are_remembered_and_kept(tmp_path: Path) -> None:
    clk = Clock()
    r = reg(tmp_path, clk)
    a, _ = r.issue("Hark", ["read"], 60)
    r.new_refresh(a.id)
    clk.t += 3600
    assert not a.active(clk.t) and a.can_refresh(clk.t)
    assert a in r.reconnectable() and a in r.known()
    pub = a.public(clk.t)
    assert pub["stays_connected_until"] > clk.t
    r.save()
    assert reg(tmp_path, clk).get(a.id).refresh_hash == a.refresh_hash


def test_connector_url_renews_itself_after_a_restart(tmp_path: Path) -> None:
    clk = Clock()
    r = reg(tmp_path, clk)
    p = r.new_pairing(["read"])
    r.mark_pending(p, "ChatGPT")
    a, _ = r.approve(p, 3600, bind_link=True)
    clk.t += 5 * 3600  # the 1-hour access ended while the PC was off
    r2 = reg(tmp_path, clk)
    got = r2.authenticate_link(p.code)
    assert got is not None and got.id == a.id and got.expires > clk.t
    r2.revoke(a.id)
    assert r2.authenticate_link(p.code) is None


# ------------------------------------------------------------------------------------------------ announcing
class FakeAgent:
    def __init__(self, name: str, kind: str = "remote", wake: str = "", rid: str = "", key: str = "") -> None:
        self.id, self.name, self.kind, self.wake_url = name.lower(), name, kind, wake
        self.rdv_id, self.rdv_key = rid or f"rid{name.lower()}", key or f"key-{name}"


def test_new_address_goes_to_every_paired_agent(tmp_path: Path) -> None:
    agents = [FakeAgent("Hark", wake="https://hark.example/hook"), FakeAgent("Muse"), FakeAgent("Desktop", "local")]
    posts: list[tuple[str, dict[str, Any]]] = []
    raws: list[tuple[str, bytes]] = []
    inbox = Inbox(background=False, path=tmp_path / "inbox.json")
    events: list[tuple[str, dict[str, Any]]] = []

    class Hub:
        def publish(self, topic: str, data: dict[str, Any]) -> dict[str, Any]:
            events.append((topic, data))
            return {"seq": len(events)}

    clk = Clock()
    ann = rendezvous.LinkAnnouncer(
        lambda: agents,
        inbox=inbox,
        events=Hub(),
        settings=lambda: C.AgentLinkConfig(),
        state_path=tmp_path / "link.json",
        post=lambda u, p: posts.append((u, p)),
        post_raw=lambda u, b: raws.append((u, b)),
        clock=clk,
        background=False,
    )
    res = ann.link_up("https://a-b-c.trycloudflare.com", "standard")
    assert [r["agent"] for r in res] == ["Hark", "Muse"]  # local agents find the app through host.json
    assert posts and posts[0][0] == "https://hark.example/hook" and posts[0][1]["event"] == "link.moved"
    assert {u for u, _ in raws} == {"https://ntfy.sh/away-ridhark", "https://ntfy.sh/away-ridmuse"}
    for u, body in raws:
        a = agents[0] if "hark" in u else agents[1]
        assert rendezvous.verify(json.loads(body), a.rdv_id, a.rdv_key, now=clk.t) == "https://a-b-c.trycloudflare.com"
    assert inbox.unread("hark") == 1 and inbox.unread("muse") == 1 and inbox.unread("desktop") == 0
    assert events[-1][0] == "link.moved"
    # same address again (e.g. a named tunnel after a restart): nothing new is sent
    assert ann.link_up("https://a-b-c.trycloudflare.com") == []
    # the relay copy is refreshed after REPUBLISH_S, without spamming the inbox or webhooks
    clk.t += rendezvous.REPUBLISH_S + 1
    n_posts, n_inbox = len(posts), inbox.unread("hark")
    assert ann.link_up("https://a-b-c.trycloudflare.com")
    assert len(posts) == n_posts and inbox.unread("hark") == n_inbox and len(raws) == 4
    # restart: a new random address is announced again (state survives in link.json)
    ann2 = rendezvous.LinkAnnouncer(
        lambda: agents, settings=lambda: C.AgentLinkConfig(), state_path=tmp_path / "link.json",
        post=lambda u, p: posts.append((u, p)), post_raw=lambda u, b: raws.append((u, b)), clock=clk, background=False,
    )  # fmt: skip
    assert ann2.last_url == "https://a-b-c.trycloudflare.com"
    ann2.link_up("https://x-y-z.trycloudflare.com")
    assert posts[-1][1]["previous"] == "https://a-b-c.trycloudflare.com"


def test_rendezvous_off_and_forged_messages() -> None:
    a = FakeAgent("Hark")
    msg = rendezvous.announcement(a, "https://new.trycloudflare.com", 1000)
    assert rendezvous.verify(msg, a.rdv_id, a.rdv_key, now=1000) == "https://new.trycloudflare.com"
    forged = dict(msg, url="https://evil.example")
    assert rendezvous.verify(forged, a.rdv_id, a.rdv_key, now=1000) is None
    assert rendezvous.verify(msg, a.rdv_id, "other-key", now=1000) is None
    assert rendezvous.verify(dict(msg, url="http://plain.example"), a.rdv_id, a.rdv_key, now=1000) is None
    assert rendezvous.verify(msg, a.rdv_id, a.rdv_key, now=1000 + 30 * 86400) is None  # too old
    raws: list[Any] = []
    s = C.AgentLinkConfig()
    s.rendezvous = "off"
    ann = rendezvous.LinkAnnouncer(
        lambda: [a], settings=lambda: s, post_raw=lambda u, b: raws.append(u), background=False
    )
    res = ann.link_up("https://q.trycloudflare.com")
    assert raws == [] and res[0]["rendezvous"] == "off"


def test_config_validation_for_the_new_link_settings() -> None:
    cfg = C.AppConfig()
    C.validate(cfg)
    for field, bad in (("rendezvous", "carrier-pigeon"), ("rendezvous_server", "http://x"), ("link_type", "ftp")):
        c = C.AppConfig()
        setattr(c.agent_link, field, bad)
        with pytest.raises(ValueError):
            C.validate(c)
    c = C.AppConfig()
    c.agent_link.named_tunnel_hostname = "not a host"
    with pytest.raises(ValueError):
        C.validate(c)
    c.agent_link.named_tunnel_hostname = "away.example.com"
    c.agent_link.link_type = "named"
    C.validate(c)


# --------------------------------------------------------------------------------------------- the host
def make_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("starlette")
    from away.agentlink.host import AgentLinkHost
    from away.agentlink.prompts import NullUI

    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    monkeypatch.setenv("AWAY_CONFIG", str(cfgp))
    h = AgentLinkHost(None, data_dir=tmp_path, ui=NullUI(), headless=False)
    h.config_path = cfgp
    h.announcer.post = lambda u, p: None
    h.announcer.post_raw = lambda u, b: None
    h.announcer.background = False
    return h


class FakeLink:
    def __init__(self, url: str) -> None:
        self.url = url
        self.up = True

    def alive(self) -> bool:
        return self.up

    def stop(self) -> None:
        self.up = False
        self.url = None


def test_host_brings_the_link_back_after_a_restart_and_tells_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = make_host(tmp_path, monkeypatch)
    a, _ = h.core.registry.issue("Hark", ["read"], 3600)
    h.core.registry.new_refresh(a.id)
    started: list[str] = []

    def fake_ensure(kind: str, port: int, download: bool) -> str:
        assert download is False  # never downloads by itself
        started.append(kind)
        h.tunnel = FakeLink("https://restart-1.trycloudflare.com")
        h._link_up(h.tunnel.url, kind)
        return h.tunnel.url

    monkeypatch.setattr(h, "ensure_server", lambda: 4321)
    monkeypatch.setattr(h, "_ensure_link", fake_ensure)
    h.restore_link("Project Away started")
    for _ in range(100):
        if started and not h._link_starting:
            break
        import time

        time.sleep(0.02)
    assert started == ["standard"]
    assert h.current_url() == "https://restart-1.trycloudflare.com"
    assert h.link_info()["url"] == "https://restart-1.trycloudflare.com"
    assert h.core.service.inbox.unread(a.id, a.name) == 1  # told through its inbox
    # the backup relay moving to a new address is announced the same way
    h.relay = FakeLink("https://new.lhr.life")
    h._relay_moved("https://old.lhr.life", "https://new.lhr.life")
    assert h.announcer.last_url == "https://new.lhr.life"
    h.stop()


def test_host_leaves_the_link_off_without_remote_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = make_host(tmp_path, monkeypatch)
    h.core.registry.issue("Claude Desktop", ["read"], 3600, kind="local")
    called: list[str] = []
    monkeypatch.setattr(h, "_ensure_link", lambda *a: called.append("x"))
    h.restore_link()
    assert called == [] and h._wants_link() == ""
    h.cfg().agent_link.keep_link = False
    assert h._wants_link() == ""
    h.stop()


# ------------------------------------------------------------------------------------------- over the wire
def test_pairing_hands_out_a_refresh_token_and_refresh_works_over_rest(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from away.agentlink import api, mcp_tools
    from away.agentlink.host import build_core
    from away.agentlink.prompts import NullUI

    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    ui = NullUI()
    core = build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=ui)
    core.prompts.polite = lambda: True
    gate = api.Gate(
        core.service, core.registry, local_key="k" * 43, rate=(100.0, 100),
        link_info=lambda: {"url": "https://now.trycloudflare.com", "rendezvous": "ntfy", "rendezvous_server": "https://ntfy.sh"},
    )  # fmt: skip
    gate.pair = lambda p, name, client, bind, kind="remote": core.registry.approve(  # type: ignore[method-assign]
        p if core.registry.mark_pending(p, name, client) else p, 3600
    )
    client = TestClient(api.create_app(core.service, gate, mcp_tools.build(api.LocalBackend(core.service))))
    assert client.get("/v1/ping").status_code == 200
    p = core.registry.new_pairing(["read", "diagnose"])
    data = client.post("/v1/pair", json={"pin": p.pin, "name": "Hark"}).json()
    assert data["refresh_token"].startswith("par_")
    assert data["link"]["url"] == "https://now.trycloudflare.com"
    rdv = data["rendezvous"]
    assert rdv["topic"].startswith("away-") and rdv["key"] and rdv["poll"].startswith("https://ntfy.sh/away-")
    assert client.get("/v1/link").status_code == 401
    assert client.get("/v1/link", headers={"Authorization": f"Bearer {data['token']}"}).json()["url"]
    r2 = client.post("/v1/token/refresh", json={"refresh_token": data["refresh_token"]})
    assert r2.status_code == 200 and r2.json()["token"] != data["token"]
    assert client.get("/v1/status", headers={"Authorization": f"Bearer {r2.json()['token']}"}).status_code == 200
    assert client.post("/v1/token/refresh", json={"refresh_token": "par_nope"}).status_code == 401


# --------------------------------------------------------------------------------------------- the client
def load_client() -> Any:
    spec = importlib.util.spec_from_file_location("agent_link_client", ROOT / "scripts" / "agent_link_client.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_link_client"] = mod
    spec.loader.exec_module(mod)
    return mod


class Resp(io.BytesIO):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(body if isinstance(body, bytes) else json.dumps(body).encode())
        self.status, self.headers = status, {}

    def __enter__(self) -> Resp:
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()


def test_client_follows_a_moved_address_and_refreshes(tmp_path: Path) -> None:
    mod = load_client()
    st = mod.State(tmp_path / "agent.json")
    key, rid = "k-123", "rid123"
    st.update(
        url="https://old.trycloudflare.com", token="pa_old", refresh_token="par_r1",
        rendezvous={"id": rid, "key": key, "poll": "https://ntfy.sh/away-rid123/json?poll=1&since=24h"},
    )  # fmt: skip
    signed = rendezvous.announcement(FakeAgent("x", rid=rid, key=key), "https://new.trycloudflare.com", 2_000_000)
    forged = dict(signed, url="https://evil.example", t=2_000_100)
    ndjson = "\n".join(json.dumps({"event": "message", "message": json.dumps(m)}) for m in (forged, signed))
    calls: list[str] = []

    def opener(req: Any, timeout: float) -> Resp:
        url = req.full_url
        calls.append(url)
        if url.startswith("https://old."):
            raise urllib.error.URLError("gone")
        if url.startswith("https://ntfy.sh"):
            return Resp(200, ndjson.encode())
        if url.endswith("/v1/token/refresh"):
            assert json.loads(req.data)["refresh_token"] == "par_r1"
            return Resp(200, {"token": "pa_new", "refresh_token": "par_r2", "agent": {"name": "x"}})
        if req.headers.get("Authorization") == "Bearer pa_old":
            raise urllib.error.HTTPError(url, 401, "no", {}, io.BytesIO(b'{"error": "expired"}'))  # type: ignore[arg-type]
        return Resp(200, {"ok": True})

    c = mod.Client(state=st, opener=opener, clock=lambda: 2_000_050)
    assert c.call("get_status") == {"ok": True}
    assert c.url == "https://new.trycloudflare.com"  # the forged (unsigned) newer message was ignored
    saved = json.loads((tmp_path / "agent.json").read_text())
    assert saved["token"] == "pa_new" and saved["refresh_token"] == "par_r2"
    assert saved["url"] == "https://new.trycloudflare.com"
    assert any("ntfy.sh" in u for u in calls)


def test_client_without_rendezvous_reports_plainly(tmp_path: Path) -> None:
    mod = load_client()
    st = mod.State(tmp_path / "agent.json")
    st.update(url="https://old.trycloudflare.com", token="pa_old")

    def opener(req: Any, timeout: float) -> Resp:
        raise urllib.error.URLError("gone")

    c = mod.Client(state=st, opener=opener)
    with pytest.raises(SystemExit, match="cannot reach Project Away"):
        c.call("get_status")
