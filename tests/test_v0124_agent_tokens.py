"""v0.1.24 "Keep this agent connected": persistent agent tokens (``paway_...``).

Acceptance: (1) pair once, take the token, restart Away -> the agent reconnects with the token and nobody is asked
anything; (2) revoke -> its live stream / MCP session drops at once and reconnecting fails; (3) rotate -> the new
token works, the old one works for an hour and then never again (injectable clock); (4) one-time pairing codes work
exactly as before. Plus: only SHA-256 hashes on disk, the token is shown once, failed tokens are rate limited per
source with backoff (429), the 30 days / 90 days / never expiry choices, Settings, the CLI and the example client.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from away import config as C
from away.agentlink.auth import (
    PERSIST_PREFIX,
    ROTATE_GRACE_S,
    TOKEN_SESSION_S,
    AgentRegistry,
    expiry_choice,
)
from away.agentlink.ratelimit import SourceBackoff

ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"^paway_[A-Za-z0-9_-]{43}$")  # 32 random bytes, base64url without padding
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy to 127.0.0.1
DAY = 86400.0


class Clock:
    """Real time plus an offset the test moves (the server's other parts keep reading the real clock)."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def __call__(self) -> float:
        return time.time() + self.offset


class FixedClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def http(
    base: str, method: str, path: str, body: Any = None, token: str | None = None, headers: Any = None
) -> tuple[int, bytes, dict[str, str]]:
    h = {"Content-Type": "application/json", **(headers or {})}
    if token:
        h["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=h, method=method)
    try:
        with OPENER.open(req, timeout=30) as r:
            return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, e.read(), {k.lower(): v for k, v in (e.headers or {}).items()}


class AutoAnswer:
    """Clicks an answer on every prompt (like the person would) and records what was asked and offered."""

    def __init__(self, prompts: Any, ui: Any, answer: str) -> None:
        self.answer = answer
        self.seen: list[tuple[str, list[str], list[str]]] = []
        self._stop = threading.Event()

        def run() -> None:
            while not self._stop.is_set():
                cur = prompts.current
                if cur is not None and cur.id not in ui.clicks:
                    self.seen.append((cur.question, list(cur.options), list(cur.details)))
                    ui.clicks[cur.id] = self.answer if self.answer in cur.options else cur.options[-1]
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


def start_host(tmp_path: Path, answer: str | None) -> tuple[Any, Any, str]:
    """A real AgentLinkHost (Away's Agent Link) on a real local port; ``answer`` is clicked on every prompt."""
    pytest.importorskip("mcp")
    pytest.importorskip("uvicorn")
    from away.agentlink.host import AgentLinkHost
    from away.agentlink.prompts import NullUI

    ui = NullUI()
    h = AgentLinkHost(None, data_dir=tmp_path, ui=ui, headless=True)
    h.core.prompts.polite = lambda: True
    h.start()
    port = h.ensure_server()
    auto = AutoAnswer(h.core.prompts, ui, answer) if answer else None
    return h, auto, f"http://127.0.0.1:{port}"


def stop_host(h: Any, auto: Any) -> None:
    if auto is not None:
        auto.stop()
    h.stop()


def pair_keep(h: Any, base: str, name: str = "Cookie", **extra: Any) -> dict[str, Any]:
    p = h.core.registry.new_pairing(["read", "diagnose"])
    st, body, _ = http(base, "POST", "/v1/pair", {"pin": p.pin, "name": name, "keep_connected": True, **extra})
    assert st == 200, body
    return dict(json.loads(body))


async def _mcp(url: str, headers: dict[str, str] | None, fn: Any) -> Any:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers=headers) as (r, w, _), ClientSession(r, w) as s:
        await s.initialize()
        return await fn(s)


def prompt_counter(core: Any) -> list[str]:
    asked: list[str] = []
    orig = core.prompts.create

    def create(question: str, *a: Any, **kw: Any) -> Any:
        asked.append(question)
        return orig(question, *a, **kw)

    core.prompts.create = create
    return asked


# ============================================================================ (1) pair once, survive a restart
def test_pair_once_then_restart_reconnects_with_the_token_and_nobody_is_asked(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        data = pair_keep(h, base)
        q, options, details = auto.seen[-1]
        assert q == "Cookie wants to connect and stay connected"
        assert options == ["Allow", KEEP_CONNECTED, "Deny"]
        assert any("for 90 days" in d for d in details)  # the default lifetime, said in the question itself
        tok, access = data["persistent_token"], data["token"]
        assert TOKEN_RE.match(tok)
        assert data["persistent_token_expires"] == pytest.approx(time.time() + 90 * DAY, abs=60)  # default: 90 days
        assert "Bearer" in data["how_to_stay_connected"]
        agent_id = data["agent"]["id"]
        assert http(base, "GET", "/v1/status", token=tok)[0] == 200
    finally:
        stop_host(h, auto)

    # Away restarts (or the PC reboots) two days later: the pairing's access token is long gone
    h2, _none, _base = start_host(tmp_path, None)
    try:
        h2.core.registry.clock = Clock(2 * DAY)
        asked = prompt_counter(h2.core)
        assert h2.server is not None and h2.server.running  # the server came up by itself for the kept agent
        base2 = f"http://127.0.0.1:{h2.server.port}"
        assert http(base2, "GET", "/v1/status", token=access)[0] == 401  # the old short access expired
        st, body, _ = http(base2, "GET", "/v1/status", token=tok)
        assert st == 200, body
        assert json.loads(body)["agent"]["id"] == agent_id  # the same agent record, same permissions
        assert json.loads(body)["agent"]["scopes"] == ["read", "diagnose"]

        async def mcp_call(s: Any) -> Any:
            return json.loads((await s.call_tool("get_status", {})).content[0].text)

        st_mcp = asyncio.run(_mcp(f"{base2}/mcp", {"Authorization": f"Bearer {tok}"}, mcp_call))
        assert st_mcp["agent"]["name"] == "Cookie"
        assert asked == [] and h2.core.prompts.pending() == []  # zero user action: no prompt at all
        a = h2.core.registry.get(agent_id)
        assert a is not None and a.active(h2.core.registry.clock())
    finally:
        h2.stop()


def test_token_session_slides_24h_on_activity_and_resumes_after_a_long_break(tmp_path: Path) -> None:
    clk = FixedClock()
    r = AgentRegistry(tmp_path / "agents.json", clock=clk)
    a, _access = r.issue("Cookie", ["read"], 3600)
    _t, tok = r.issue_persistent(a.id)
    for _ in range(5):  # used every 23 h: the session never lapses
        clk.t += 23 * 3600
        hit = r.authenticate_persistent(tok, "203.0.113.5", "abc.trycloudflare.com")
        assert hit is not None and hit[0].id == a.id
        assert a.expires == pytest.approx(clk.t + TOKEN_SESSION_S)
    clk.t += 3 * DAY  # silent for three days: the session lapsed ...
    assert not a.active(clk.t)
    assert r.authenticate_persistent(tok) is not None  # ... and the token simply opens a new one
    assert a.active(clk.t)


# ============================================================================ (2) revoke drops live sessions
def test_revoke_drops_the_live_stream_at_once_and_reconnecting_fails(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        data = pair_keep(h, base)
        tok = data["persistent_token"]
        ended: dict[str, float] = {}
        opened = threading.Event()

        def listen() -> None:
            req = urllib.request.Request(
                base + "/v1/events/stream", headers={"Authorization": f"Bearer {tok}", "Accept": "text/event-stream"}
            )
            with OPENER.open(req, timeout=60) as resp:
                opened.set()
                while resp.read(1):
                    pass
            ended["t"] = time.monotonic()

        th = threading.Thread(target=listen, daemon=True)
        th.start()
        assert opened.wait(10)
        time.sleep(0.3)
        assert th.is_alive()  # the stream is live (it would otherwise run for an hour)
        t0 = time.monotonic()
        res = h.handle({"action": "token_revoke", "token_id": data["persistent_token_id"]})  # Settings > Revoke
        assert res["revoked"] is True
        th.join(5)
        assert not th.is_alive() and ended["t"] - t0 < 2.0  # dropped at once (not after the 15 s event wait)
        assert http(base, "GET", "/v1/status", token=tok)[0] == 401
        assert http(base, "GET", "/v1/status", token=data["token"])[0] == 401  # the pairing token is gone too
        assert http(base, "POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token=tok)[0] == 401
        assert h.core.registry.tokens == {} and h.core.registry.get(data["agent"]["id"]).revoked
    finally:
        stop_host(h, auto)


def test_revoke_ends_a_live_mcp_call_and_session(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        data = pair_keep(h, base)
        tok = data["persistent_token"]

        ended: list[tuple[str, float]] = []
        orig_end = h.gate.end_mcp_session
        assert orig_end is not None

        async def end(sid: str) -> None:
            await orig_end(sid)
            ended.append((sid, time.monotonic()))

        h.gate.end_mcp_session = end

        async def long_wait(s: Any) -> float:
            call = asyncio.ensure_future(s.call_tool("events", {"action": "wait", "since": 10**9, "timeout_s": 60}))
            await asyncio.sleep(0.5)
            assert not call.done()  # the call is waiting on the PC
            t0 = time.monotonic()
            await asyncio.to_thread(h.core.registry.revoke_persistent, data["persistent_token_id"])
            while not ended and time.monotonic() - t0 < 15:
                await asyncio.sleep(0.05)
            call.cancel()
            return ended[0][1] - t0 if ended else 99.0

        with contextlib.suppress(BaseException):  # the client complains about its session going away; expected
            took = asyncio.run(_mcp(f"{base}/mcp", {"Authorization": f"Bearer {tok}"}, long_wait))
        assert ended  # the server ended the agent's MCP session ...
        assert took < 2.0  # ... at once, not after the 60 s the call asked for
        assert http(base, "GET", "/v1/status", token=tok)[0] == 401
    finally:
        stop_host(h, auto)


def test_disconnect_and_disconnect_all_also_end_agent_tokens(tmp_path: Path) -> None:
    r = AgentRegistry(tmp_path / "agents.json")
    a, _ = r.issue("Cookie", ["read"], 3600)
    b, _ = r.issue("Hark", ["read"], 3600)
    _ta, tok_a = r.issue_persistent(a.id)
    _tb, tok_b = r.issue_persistent(b.id)
    dropped: list[str] = []
    r.on_revoke.append(dropped.append)
    assert r.revoke(a.id)  # the existing Disconnect button
    assert r.authenticate_persistent(tok_a) is None and dropped == [a.id]
    assert r.revoke_all() == 1
    assert r.authenticate_persistent(tok_b) is None and b.id in dropped
    assert AgentRegistry(tmp_path / "agents.json").tokens == {}


# ============================================================================ (3) rotate with a one-hour grace
def test_rotate_new_works_old_works_within_the_hour_then_dies(tmp_path: Path) -> None:
    clk = FixedClock()
    r = AgentRegistry(tmp_path / "agents.json", clock=clk)
    a, _ = r.issue("Cookie", ["read"], 3600)
    t, old = r.issue_persistent(a.id)
    t2, new = r.rotate_persistent(t.id)
    assert t2.id == t.id and new != old and TOKEN_RE.match(new)
    assert r.authenticate_persistent(new) is not None
    clk.t += ROTATE_GRACE_S - 60  # 59 minutes later the old one still works (nothing breaks midway)
    assert r.authenticate_persistent(old) is not None
    r2 = AgentRegistry(tmp_path / "agents.json", clock=clk)  # ... also across a restart
    assert r2.authenticate_persistent(old) is not None
    clk.t += 120  # past the hour
    assert r2.authenticate_persistent(old) is None and r.authenticate_persistent(old) is None
    assert r2.authenticate_persistent(new) is not None
    r2.save()
    assert t.token_hash in (tmp_path / "agents.json").read_text()  # the new hash is kept ...
    assert hashlib.sha256(old.encode()).hexdigest() not in (tmp_path / "agents.json").read_text()  # the old is gone


def test_rotate_over_the_wire_and_an_old_token_stream_ends_with_its_grace(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        clk = Clock()
        h.core.registry.clock = clk
        data = pair_keep(h, base)
        old = data["persistent_token"]
        assert http(base, "POST", "/v1/token/rotate", {}, token=data["token"])[0] == 400  # needs the agent token
        st, body, _ = http(base, "POST", "/v1/token/rotate", {}, token=old)
        assert st == 200, body
        rot = json.loads(body)
        new = rot["persistent_token"]
        assert TOKEN_RE.match(new) and rot["old_token_grace_s"] == ROTATE_GRACE_S
        assert http(base, "GET", "/v1/status", token=new)[0] == 200
        assert http(base, "GET", "/v1/status", token=old)[0] == 200  # within the hour
        ended = threading.Event()

        def listen() -> None:
            req = urllib.request.Request(
                base + "/v1/events/stream", headers={"Authorization": f"Bearer {old}", "Accept": "text/event-stream"}
            )
            with OPENER.open(req, timeout=60) as resp:
                while resp.read(1):
                    pass
            ended.set()

        threading.Thread(target=listen, daemon=True).start()
        time.sleep(0.5)
        assert not ended.is_set()
        clk.offset = ROTATE_GRACE_S + 5  # the hour is over
        assert ended.wait(3)  # the stream opened with the old token ends with it
        assert http(base, "GET", "/v1/status", token=old)[0] == 401
        assert http(base, "GET", "/v1/status", token=new)[0] == 200  # the agent itself is still connected
    finally:
        stop_host(h, auto)


# ============================================================================ (4) one-time pairing unchanged
def test_one_time_pairing_works_exactly_as_before(tmp_path: Path) -> None:
    h, auto, base = start_host(tmp_path, "Allow")
    try:
        p = h.core.registry.new_pairing(["read", "tune"])
        st, body, _ = http(base, "POST", "/v1/pair", {"pin": p.pin, "name": "My Script"})
        assert st == 200, body
        data = json.loads(body)
        assert auto.seen[-1][0] == "My Script wants to connect"
        assert data["token"].startswith("pa_") and data["refresh_token"].startswith("par_")
        assert data["agent"]["permissions"] == ["See my gestures", "Change my settings"]
        assert not any(k.startswith("persistent") for k in data)  # nothing persistent unless chosen
        assert h.core.registry.tokens == {}
        assert http(base, "GET", "/v1/status", token=data["token"])[0] == 200
        assert http(base, "POST", "/v1/pair", {"pin": p.pin, "name": "again"})[0] == 404  # still single use
        auto.answer = "Deny"
        p2 = h.core.registry.new_pairing(["read"])
        assert http(base, "POST", "/v1/pair", {"code": p2.code, "name": "x", "keep_connected": True})[0] == 403
        assert h.core.registry.tokens == {}
    finally:
        stop_host(h, auto)


def test_connector_links_are_not_offered_a_token(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, "Allow")
    try:
        p = h.core.registry.new_pairing(["read"])

        async def ok(s: Any) -> bool:
            return not (await s.call_tool("get_status", {})).isError

        assert asyncio.run(_mcp(f"{base}/c/{p.code}", None, ok))  # a connector URL with no header, as before
        assert h.core.registry.tokens == {}
    finally:
        stop_host(h, auto)
    offered = [opts for _q, opts, _d in auto.seen]
    assert offered == [["Allow", "Deny"]]  # the link itself stays connected; a token it cannot send is not offered
    assert all(KEEP_CONNECTED not in o for o in offered)


# ============================================================================ storage, shown once
def test_only_the_sha256_is_stored_and_the_token_is_shown_once(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        data = pair_keep(h, base)
        tok = data["persistent_token"]
        h.write_state()
        disk = (tmp_path / "agentlink" / "agents.json").read_text()
        state = (tmp_path / "agentlink" / "state.json").read_text()
        assert tok not in disk and hashlib.sha256(tok.encode()).hexdigest() in disk
        assert tok not in state and hashlib.sha256(tok.encode()).hexdigest() not in state
        assert json.loads(state)["tokens"][0]["agent"] == "Cookie"
        rec = h.core.registry.tokens[data["persistent_token_id"]]
        assert tok not in json.dumps(vars(rec)) and h.gate.kept == {}  # nothing left that could show it again
        assert not any("hash" in k for k in h.core.registry.tokens_public()[0])
        # Settings > New token: the reply carries a one-time reference, never the token
        res = h.handle({"action": "token_rotate", "token_id": "Cookie", "id": "r1"})
        assert "secret_ref" in res and not any(str(v).startswith(PERSIST_PREFIX) for v in res.values())
        from away.agentlink.host import take_secret

        new = take_secret(res["secret_ref"], tmp_path)
        assert new is not None and TOKEN_RE.match(new)
        assert take_secret(res["secret_ref"], tmp_path) is None  # read once, then gone
        assert http(base, "GET", "/v1/status", token=new)[0] == 200
    finally:
        stop_host(h, auto)


def test_an_unread_rotated_token_is_deleted(tmp_path: Path) -> None:
    from away.agentlink.host import clear_secrets, put_secret, take_secret

    ref = put_secret(tmp_path / "agentlink", "paway_x")
    f = tmp_path / "agentlink" / "secrets" / f"{ref}.txt"
    if os.name != "nt":
        assert (f.stat().st_mode & 0o777) == 0o600
    os.utime(f, (time.time() - 600, time.time() - 600))
    clear_secrets(tmp_path / "agentlink")
    assert take_secret(ref, tmp_path) is None
    assert take_secret("../../config", tmp_path) is None  # only plain references are looked up


# ============================================================================ rate limiting
def test_wrong_tokens_are_rate_limited_per_source_with_backoff(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        tok = pair_keep(h, base)["persistent_token"]
        bad = PERSIST_PREFIX + "x" * 43
        attacker = {"cf-connecting-ip": "198.51.100.7"}
        codes = [http(base, "GET", "/v1/status", token=bad, headers=attacker)[0] for _ in range(6)]
        assert codes == [401] * 6
        st, body, hdrs = http(base, "GET", "/v1/status", token=bad, headers=attacker)
        assert st == 429 and int(hdrs["retry-after"]) >= 1 and b"wait" in body
        # even the right token is refused from that address while it waits ...
        assert http(base, "GET", "/v1/status", token=tok, headers=attacker)[0] == 429
        # ... and a different forged header on the same shared listener is the same source: the header is
        # ignored there, so it waits too. Per-source isolation on shared ingress is the relay's SSH
        # authentication, not the app's; tunnel-listener isolation is covered in test_v0129_ingress_trust.py.
        assert http(base, "GET", "/v1/status", token=tok, headers={"cf-connecting-ip": "203.0.113.9"})[0] == 429
        assert http(base, "GET", "/openapi.json")[0] == 200
    finally:
        stop_host(h, auto)


def test_backoff_doubles_caps_and_clears() -> None:
    clk = FixedClock(0.0)
    b = SourceBackoff(free=3, base_s=2.0, max_s=30.0, window_s=600.0, clock=clk)
    assert [b.fail("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert b.retry_after("a") == 0.0
    waits = [b.fail("a") for _ in range(6)]
    assert waits == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]  # doubles, capped
    assert b.retry_after("a") == 30.0 and b.retry_after("other") == 0.0
    clk.t += 31
    assert b.retry_after("a") == 0.0
    b.ok("a")
    assert b.fail("a") == 0.0  # a success starts it over


# ============================================================================ expiry choices
@pytest.mark.parametrize(("choice", "days"), [("30d", 30), ("90d", 90), (30, 30), ("90 days", 90)])
def test_expiry_choices_30_and_90_days(tmp_path: Path, choice: Any, days: int) -> None:
    clk = FixedClock()
    r = AgentRegistry(tmp_path / "agents.json", clock=clk)
    a, _ = r.issue("Cookie", ["read"], 3600)
    t, tok = r.issue_persistent(a.id, choice)
    assert t.expires == pytest.approx(clk.t + days * DAY)
    clk.t += days * DAY - 60
    assert r.authenticate_persistent(tok) is not None
    clk.t += 120
    assert r.authenticate_persistent(tok) is None
    assert r.tokens_public() == []
    r.save()
    assert AgentRegistry(tmp_path / "agents.json", clock=clk).tokens == {}  # an expired token is not kept


def test_90_days_is_the_default_never_is_still_a_choice_and_bad_choices_are_refused(tmp_path: Path) -> None:
    clk = FixedClock()
    r = AgentRegistry(tmp_path / "agents.json", clock=clk)
    a, _ = r.issue("Cookie", ["read"], 3600)
    t, tok = r.issue_persistent(a.id)
    assert t.expiry == "90d" and t.expires == pytest.approx(clk.t + 90 * DAY)
    assert expiry_choice(None) == expiry_choice("") == "90d"  # no choice given: the default, never "never"
    b, _ = r.issue("Hark", ["read"], 3600)
    tn, tok_n = r.issue_persistent(b.id, "never")  # until revoked: still selectable
    assert tn.expiry == "never" and tn.expires == 0.0
    assert expiry_choice("0") == expiry_choice("none") == "never"
    clk.t += 5 * 365 * DAY
    assert r.authenticate_persistent(tok) is None  # the 90-day token ended
    assert r.authenticate_persistent(tok_n) is not None
    for bad in ("7d", "forever", -1):
        with pytest.raises(ValueError, match="30 days, 90 days or never"):
            expiry_choice(bad)
    assert C.AppConfig().agent_link.token_expiry == "90d"
    cfg = C.AppConfig()
    cfg.agent_link.token_expiry = "7d"
    with pytest.raises(ValueError, match="token_expiry"):
        C.validate_agent_link(cfg.agent_link)


def test_the_agent_can_ask_for_an_expiry_and_a_bad_one_is_a_plain_400(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        data = pair_keep(h, base, token_expiry="30d")
        assert data["persistent_token_expires"] == pytest.approx(time.time() + 30 * DAY, abs=60)
        assert any("for 30 days" in d for d in auto.seen[-1][2])
        p = h.core.registry.new_pairing(["read"])
        st, body, _ = http(base, "POST", "/v1/pair", {"pin": p.pin, "keep_connected": True, "token_expiry": "7d"})
        assert st == 400 and b"30 days, 90 days or never" in body
        assert h.core.registry.find_pairing(code=p.code) is not None  # the code was not spent on a bad request
    finally:
        stop_host(h, auto)


# ============================================================================ Settings, CLI, client
def test_settings_rows_say_it_in_plain_words() -> None:
    from away.agentlink.gui_model import AgentLinkModel

    now = time.time()
    st = {
        "tokens": [
            {
                "id": "tok_1",
                "agent_id": "a1",
                "agent": "Cookie",
                "scopes": ["read", "diagnose"],
                "permissions": ["See my gestures", "Run checks"],
                "created": now - 3 * DAY,
                "expires": 0.0,
                "last_used": now - 300,
                "last_source": "203.0.113.5",
                "last_host": "abc.trycloudflare.com",
                "old_key_until": now + 1800,
                "connected": True,
            }
        ]
    }
    (row,) = AgentLinkModel.token_rows(st, now)
    assert row["name"] == "Cookie" and row["can"] == "Can: See my gestures, Run checks"
    assert "last used 5 minutes ago from 203.0.113.5 (through abc.trycloudflare.com)" in row["when"]
    assert row["ends"] == "Stays connected until you revoke it"
    assert row["old_key"].startswith("The old token still works until ")
    text = " ".join(str(row[k]) for k in ("name", "can", "when", "ends", "old_key")).lower()  # what is shown
    for jargon in ("scope", "hash", "bearer", "paway_", "tunnel"):
        assert jargon not in text


def test_settings_rotate_shows_the_new_token_once_and_revoke_says_what_happens(tmp_path: Path) -> None:
    from away.agentlink.gui_model import AgentLinkModel
    from away.agentlink.host import put_secret

    base = tmp_path / "agentlink"
    sent: list[tuple[str, dict[str, Any]]] = []
    results: dict[str, Any] = {}

    def send(action: str, **kw: Any) -> str:
        sent.append((action, kw))
        rid = f"r{len(sent)}"
        if action == "token_rotate":
            results[rid] = {"ok": True, "result": {"secret_ref": put_secret(base, "paway_" + "n" * 43)}}
        elif action == "token_revoke":
            results[rid] = {"ok": True, "result": {"revoked": True}}
        return rid

    m = AgentLinkModel(data_dir=tmp_path, send=send, state=lambda: {"results": results}, alive=lambda: True)
    problem, secret = m.rotate_token("tok_1")
    assert problem is None and secret == "paway_" + "n" * 43
    assert list((base / "secrets").glob("*.txt")) == []  # shown once: the hand-over file is gone
    assert m.revoke_token("tok_1").startswith("Revoked. It is disconnected now")
    m.set_token_scopes("tok_1", ["read", "tune"])
    assert sent[-1] == ("token_scopes", {"token_id": "tok_1", "scopes": ["read", "tune"]})
    off = AgentLinkModel(data_dir=tmp_path, send=send, state=dict, alive=lambda: None)
    assert off.rotate_token("tok_1")[0] is not None and off.revoke_token("x") == "Project Away isn't running."


def test_cli_lists_rotates_changes_and_revokes_tokens(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from away import paths
    from away.agentlink.host import base_dir
    from away.cli import main as cli_main

    path = base_dir(paths.data_dir()) / "agents.json"
    r = AgentRegistry(path)
    a, _ = r.issue("Cookie", ["read"], 3600)
    _t, tok = r.issue_persistent(a.id, "90d")
    assert cli_main(["agent", "tokens"]) == 0
    out = capsys.readouterr().out
    assert "Cookie" in out and "can: See my gestures" in out and "expires" in out and tok not in out
    assert cli_main(["agent", "tokens", "rotate", "cookie"]) == 0
    out = capsys.readouterr().out
    new = next(w for w in out.split() if w.startswith(PERSIST_PREFIX))
    assert TOKEN_RE.match(new) and "keeps working until" in out
    assert cli_main(["agent", "tokens", "scopes", "Cookie", "read,tune"]) == 0
    assert "Change my settings" in capsys.readouterr().out
    assert cli_main(["agent", "tokens", "revoke", "Cookie"]) == 0
    assert "revoked" in capsys.readouterr().out
    r2 = AgentRegistry(path)
    assert r2.authenticate_persistent(new) is None and r2.authenticate_persistent(tok) is None
    assert cli_main(["agent", "tokens", "rotate", "nobody"]) == 1
    assert cli_main(["agent", "tokens", "explode"]) == 2


def test_cli_talks_to_the_running_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from away import paths
    from away.agentlink import cli as acli
    from away.agentlink import host as hostmod
    from away.agentlink.prompts import NullUI

    h = hostmod.AgentLinkHost(None, data_dir=paths.data_dir(), ui=NullUI(), headless=True)
    a, _ = h.core.registry.issue("Cookie", ["read"], 3600)
    h.core.registry.issue_persistent(a.id)
    store: dict[str, Any] = {}

    def send_request(action: str, **kw: Any) -> str:
        store["r"] = {"ok": True, "result": h.handle({"action": action, **kw})}
        return "r"

    monkeypatch.setattr(hostmod, "send_request", send_request)
    monkeypatch.setattr(acli, "_wait_result", lambda rid, timeout=0: store[rid])
    ns = type("NS", (), {"value": "rotate", "rest": ["Cookie"], "scopes": "read"})()
    assert acli.cmd_tokens(ns, alive=True) == 0
    new = next(w for w in capsys.readouterr().out.split() if w.startswith(PERSIST_PREFIX))
    t = h.core.registry.find_token("Cookie")
    assert t is not None and t.prev_hash  # the app rotated it (the old one is in its grace hour) ...
    assert h.core.registry.authenticate_persistent(new) is not None  # ... and the CLI showed the new one
    ns = type("NS", (), {"value": "revoke", "rest": [t.id], "scopes": "read"})()
    assert acli.cmd_tokens(ns, alive=True) == 0
    assert h.core.registry.tokens == {} and "revoked" in capsys.readouterr().out


def load_client() -> Any:
    spec = importlib.util.spec_from_file_location("agent_link_client", ROOT / "scripts" / "agent_link_client.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_link_client"] = mod
    spec.loader.exec_module(mod)
    return mod


class Resp(io.BytesIO):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(json.dumps(body).encode())
        self.status, self.headers = status, {}

    def __enter__(self) -> Resp:
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()


def test_client_keeps_the_agent_token_and_reconnects_with_it(tmp_path: Path) -> None:
    mod = load_client()
    st = mod.State(tmp_path / "agent.json")
    tok = "paway_" + "k" * 43
    seen: list[tuple[str, str | None, Any]] = []

    def opener(req: Any, timeout: float) -> Resp:
        body = json.loads(req.data) if req.data else None
        seen.append((req.full_url, req.headers.get("Authorization"), body))
        if req.full_url.endswith("/v1/pair"):
            return Resp(200, {"token": "pa_1", "agent": {"name": "Cookie"}, "persistent_token": tok})
        return Resp(200, {"ok": True})

    c = mod.Client("https://pc.example/c/abcdef", state=st, opener=opener)
    res = c.pair("123456", None, "Cookie", keep=True, expiry="90d")
    assert seen[0][2]["keep_connected"] is True and seen[0][2]["token_expiry"] == "90d"
    assert res["persistent_token"] == tok
    assert json.loads((tmp_path / "agent.json").read_text())["persistent_token"] == tok
    c2 = mod.Client(state=mod.State(tmp_path / "agent.json"), opener=opener)  # a new run (after a reboot)
    assert c2.call("get_status") == {"ok": True}
    assert seen[-1][1] == f"Bearer {tok}"  # the agent token, not the short access token


def test_client_says_plainly_when_the_token_was_revoked(tmp_path: Path) -> None:
    mod = load_client()
    st = mod.State(tmp_path / "agent.json")
    st.update(url="https://pc.example", persistent_token="paway_" + "k" * 43)

    def opener(req: Any, timeout: float) -> Resp:
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b'{"error": "revoked"}'))  # type: ignore[arg-type]

    c = mod.Client(state=st, opener=opener)
    with pytest.raises(SystemExit, match="revoked or replaced it"):
        c.call("get_status")
    assert "persistent_token" not in json.loads((tmp_path / "agent.json").read_text())
    with pytest.raises(SystemExit, match="not an agent token"):
        c.use_token("pa_short")
    c.use_token("paway_" + "z" * 43)
    assert c.bearer() == "paway_" + "z" * 43


def _texts(widget: Any) -> list[str]:
    out = []
    for c in widget.winfo_children():
        with contextlib.suppress(Exception):
            out.append(str(c.cget("text")))
        out += _texts(c)
    return out


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and os.environ.get("DISPLAY")), reason="needs an X display (Xvfb)"
)
def test_agent_tokens_section_renders_under_a_display(tmp_path: Path) -> None:
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - no Tk
        pytest.skip(f"Tk unavailable: {exc}")
    from away.agentlink import gui as agent_gui
    from away.agentlink.gui_model import AgentLinkModel
    from away.app import settings_gui as G

    now = time.time()
    st = {
        "agents": [],
        "tokens": [
            {"id": "tok_1", "agent_id": "a1", "agent": "Cookie", "scopes": ["read"], "permissions": ["See my gestures"],
             "created": now, "expires": 0.0, "last_used": 0.0, "last_source": "", "last_host": "",
             "old_key_until": 0.0, "connected": False}
        ],
    }  # fmt: skip
    try:
        w = G.SettingsWindow(root, G.SettingsModel(path=tmp_path / "config.yaml"))
        nb = next(c for c in root.winfo_children() if c.winfo_class() == "TNotebook")
        m = AgentLinkModel(data_dir=tmp_path, send=lambda a, **k: "r", state=lambda: st, alive=lambda: True)
        tab = agent_gui.build_tab(w, nb, m)
        tab.refresh()
        root.update()
        texts = _texts(tab.tokens_box)
        assert any("Cookie" in t for t in texts)
        assert {"Revoke", "New token", "What it can do"} <= set(texts)
        st["tokens"] = []
        tab.refresh()
        root.update()
        assert any("Keep this agent connected" in t for t in _texts(tab.tokens_box))
    finally:
        root.destroy()


# ============================================================================ on top of the trust-and-safety hardening
def _safety() -> Any:
    """The hardening tests' helpers (a host with a live control grant and live view, the API test client)."""
    spec = importlib.util.spec_from_file_location("_safety_helpers", ROOT / "tests" / "test_safety_hardening.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ended(h: Any, a: Any, offs: list[int]) -> None:
    assert h.core.control.status() == {"active": False} and offs  # control ended, border off
    assert a.id not in h.core.service._watchers and a.id not in h.core.service._live_views
    assert h._watch_stop is None and a.id not in h.core.control.views  # the live view and its badge are gone


def test_revoking_a_token_ends_control_and_the_live_view(tmp_path: Path) -> None:
    h, a, offs = _safety()._host(tmp_path)
    t, tok = h.core.registry.issue_persistent(a.id)
    res = h.handle({"action": "token_revoke", "token_id": t.id})  # Settings > Agent tokens > Revoke
    assert res == {"revoked": True, "agent_id": a.id}
    _ended(h, a, offs)
    assert h.core.registry.authenticate_persistent(tok) is None


def test_rotating_a_token_out_ends_what_ran_under_it_but_keeps_the_agent(tmp_path: Path) -> None:
    h, a, offs = _safety()._host(tmp_path)
    t, old = h.core.registry.issue_persistent(a.id)
    res = h.handle({"action": "token_rotate", "token_id": t.id})  # Settings > Agent tokens > New token
    assert res["token_id"] == t.id and res["secret_ref"]
    _ended(h, a, offs)
    assert h.core.registry.get(a.id).active(time.time())  # still connected: the new token starts clean
    assert h.core.registry.authenticate_persistent(old) is not None  # the old one still signs in for the hour
    assert h.core.control.request(a.id, a.name)["active"]  # and it can ask for control again (asked each time)


def test_rotating_a_token_out_mid_typing_stops_and_lets_go_of_the_keys(tmp_path: Path) -> None:
    from away.agentlink.control import ControlRefused

    s = _safety()
    h, a, _ = s._host(tmp_path)
    t, _old = h.core.registry.issue_persistent(a.id)
    backend = s.BlockingBackend("type_text", 2)
    h.core.control.hooks.backend = backend
    h.core.control.hooks.uia = s.FakeUIA(focused=s.TEXT_BOX)
    th, out = s._run_in_thread(lambda: h.core.control.act(a.id, "type_text", text="abcdefgh" * 20))
    assert backend.reached.wait(5.0)
    h.handle({"action": "token_rotate", "token_id": t.id})
    backend.go.set()
    th.join(5.0)
    assert isinstance(out.get("error"), ControlRefused)
    assert [c[0] for c in backend.calls].count("type_text") == 2 and backend.calls[-1][0] == "release_modifiers"


def test_an_agent_rotating_its_own_token_runs_the_same_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _token, _calls, _spy, _gate, core = _safety()._api(tmp_path)
    a, _ = core.registry.issue("Cookie", ["read", "control"], 3600)
    _t, tok = core.registry.issue_persistent(a.id)
    cleaned: list[tuple[str, bool]] = []
    orig = core.service.reconcile

    def reconcile(agent_id: str, reason: str = "", end_sessions: bool = False) -> None:
        cleaned.append((agent_id, end_sessions))
        orig(agent_id, reason, end_sessions=end_sessions)

    monkeypatch.setattr(core.service, "reconcile", reconcile)
    h = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    r = client.post("/v1/token/rotate", content=b"{}", headers=h)
    assert r.status_code == 200, r.text
    assert cleaned == [(a.id, True)]


def test_the_agent_token_endpoints_use_the_body_cap(tmp_path: Path) -> None:
    from away.agentlink import api

    client, _token, _calls, _spy, _gate, core = _safety()._api(tmp_path)
    a, _ = core.registry.issue("Cookie", ["read"], 3600)
    t, tok = core.registry.issue_persistent(a.id)
    before = t.token_hash
    big = b'{"keep_connected": true, "name": "' + b"x" * (api.MAX_JSON_BODY + 1) + b'"}'
    h = {"Content-Type": "application/json"}
    for path in ("/v1/pair", "/v1/pair/local", "/v1/token/rotate"):  # no credentials: still refused before parsing
        r = client.post(path, content=big, headers=h)
        assert r.status_code == 413 and "too large" in r.json()["error"], path
    h["Authorization"] = f"Bearer {tok}"
    for path in ("/v1/token/rotate", "/v1/rpc/get_status", "/v1/pair"):
        assert client.post(path, content=big, headers=h).status_code == 413, path
    assert core.registry.tokens[t.id].token_hash == before and not core.registry.tokens[t.id].prev_hash  # not rotated
    chunks = iter([b'{"x": "', b"y" * (api.MAX_JSON_BODY + 1), b'"}'])  # chunked: no Content-Length to check
    assert client.post("/v1/token/rotate", content=chunks, headers=h).status_code == 413
    assert client.post("/v1/token/rotate", content=b"{}", headers=h).status_code == 200  # a normal one still works


def test_token_agents_get_share_logs_only_when_it_is_switched_on(tmp_path: Path) -> None:
    from away.agentlink import perms

    assert perms.SHARE_LOGS not in perms.DEFAULT
    client, _token, _calls, _spy, _gate, core = _safety()._api(tmp_path)
    a, _ = core.registry.issue("Cookie", perms.normalize(None), 3600)  # the default permissions
    t, tok = core.registry.issue_persistent(a.id, "30d")
    h = {"Authorization": f"Bearer {tok}"}
    assert client.get("/v1/status", headers=h).status_code == 200
    for path in ("/v1/logs", "/v1/files"):
        assert client.get(path, headers=h).status_code == 403, path  # diagnose alone does not cover logs
    _t, tok = core.service.rotate_token(t.id)  # a new token does not widen anything
    h = {"Authorization": f"Bearer {tok}"}
    assert client.get("/v1/logs", headers=h).status_code == 403
    assert core.registry.find_token(t.id).public(core.registry.get(a.id), time.time())["scopes"] == [
        "read",
        "diagnose",
    ]
    core.service.set_scopes(a.id, ["read", "diagnose", "share_logs"])  # the person switches it on for this agent
    assert client.get("/v1/logs", headers=h).status_code == 200
    core.service.set_scopes(a.id, ["read", "diagnose"])
    assert client.get("/v1/logs", headers=h).status_code == 403


def test_a_kept_agent_gets_only_the_permissions_it_was_allowed(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        data = pair_keep(h, base)  # the pairing allowed read + diagnose
        tok = data["persistent_token"]
        assert "share_logs" not in data["agent"]["scopes"]
        assert http(base, "GET", "/v1/logs", token=tok)[0] == 403
        h.handle({"action": "token_scopes", "token_id": data["persistent_token_id"], "scopes": ["read", "share_logs"]})
        assert http(base, "GET", "/v1/logs", token=tok)[0] == 200
    finally:
        stop_host(h, auto)


def test_the_default_expiry_can_be_changed_in_settings_and_on_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from away.agentlink.gui_model import AgentLinkModel
    from away.cli import main as cli_main

    cfgp = tmp_path / "config.yaml"
    m = AgentLinkModel(data_dir=tmp_path, config_path=cfgp)
    assert m.token_expiry() == "90d" and [k for k, _ in m.EXPIRY_CHOICES] == ["90d", "30d", "never"]
    m.set_token_expiry("30 days")  # the menu's label
    assert C.load(cfgp)[0].agent_link.token_expiry == "30d" and m.token_expiry() == "30d"
    m.set_token_expiry("never")
    assert m.token_expiry() == "never"
    with pytest.raises(ValueError):
        m.set_token_expiry("7d")
    assert cli_main(["agent", "tokens", "expiry"]) == 0
    assert "90 days" in capsys.readouterr().out
    assert cli_main(["agent", "tokens", "expiry", "30d"]) == 0
    assert "30 days" in capsys.readouterr().out
    assert C.load(C.config_path())[0].agent_link.token_expiry == "30d"
    assert cli_main(["agent", "tokens", "expiry", "never"]) == 0
    assert "until you revoke" in capsys.readouterr().out
    assert cli_main(["agent", "tokens", "expiry", "7d"]) == 2
    assert cli_main(["agent", "tokens", "expiry", "90d"]) == 0
    assert C.load(C.config_path())[0].agent_link.token_expiry == "90d"


def test_the_pc_setting_decides_the_lifetime_unless_the_agent_asks(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        h.cfg().agent_link.token_expiry = "30d"
        data = pair_keep(h, base)
        assert data["persistent_token_expires"] == pytest.approx(time.time() + 30 * DAY, abs=60)
        assert any("for 30 days" in d for d in auto.seen[-1][2])
        data = pair_keep(h, base, name="Hark", token_expiry="never")  # the agent asked; the dialog says so
        assert data["persistent_token_expires"] == 0
        assert any("until you revoke it (it asked for this)" in d for d in auto.seen[-1][2])
    finally:
        stop_host(h, auto)
