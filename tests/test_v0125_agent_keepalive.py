"""v0.1.25 Agent Link keepalive: agents stopped looking "disconnected" after a couple of minutes.

Root cause: an agent counted as connected only while ``now - last_seen < 90 s``, and ``last_seen`` moved only when
a request *started*. MCP clients keep one stream open for minutes (Streamable HTTP GET), and the example client keeps
``/v1/events/stream`` open, so a perfectly connected agent turned "not connected right now" after ~90 s: the tray,
Agent Link, voice messages ("Cookie isn't connected right now") all said it was gone. Over the Cloudflare quick tunnel,
long-polls of up to 120 s and the 120 s Allow prompt also hit Cloudflare's ~100 s limit (error 524).
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from test_v0124_agent_tokens import OPENER, _mcp, load_client, pair_keep, start_host, stop_host

CLOUDFLARE_IDLE_S = 100.0  # Cloudflare ends a proxied request that sends nothing for ~100 s (HTTP 524)


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def _age(h: Any, agent_id: str, seconds: float) -> None:
    """Pretend the agent's last request started ``seconds`` ago (its stream stays open)."""
    a = h.core.registry.get(agent_id)
    a.last_seen = time.time() - seconds


def _connected(h: Any, agent_id: str) -> bool:
    return bool(h.core.registry.get(agent_id).public(time.time())["connected"])


def test_an_open_event_stream_keeps_the_agent_connected_and_closing_it_lets_it_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from away.agentlink import api
    from away.agentlink.api import KEEP_CONNECTED

    monkeypatch.setattr(api, "LIVE_TOUCH_S", 0.3, raising=False)  # 20 s in the app; fast here
    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        data = pair_keep(h, base)
        aid, tok = data["agent"]["id"], data["persistent_token"]
        opened, stop = threading.Event(), threading.Event()

        def listen() -> None:
            req = urllib.request.Request(
                base + "/v1/events/stream", headers={"Authorization": f"Bearer {tok}", "Accept": "text/event-stream"}
            )
            with OPENER.open(req, timeout=60) as resp:
                opened.set()
                while not stop.is_set() and resp.read(1):
                    pass

        th = threading.Thread(target=listen, daemon=True)
        th.start()
        assert opened.wait(10)
        _age(h, aid, 200.0)  # the stream was opened 200 s ago and nothing else happened since
        assert not _connected(h, aid)
        time.sleep(1.0)
        assert _connected(h, aid)  # still on the line, so still connected
        assert h.core.service.reachable(h.core.registry.get(aid))  # "Hey Cookie" goes straight through
    finally:
        stop.set()
        stop_host(h, auto)


def test_a_live_mcp_session_keeps_the_agent_connected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from away.agentlink import api
    from away.agentlink.api import KEEP_CONNECTED

    monkeypatch.setattr(api, "LIVE_TOUCH_S", 0.3, raising=False)
    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        data = pair_keep(h, base)
        aid, tok = data["agent"]["id"], data["persistent_token"]
        seen: dict[str, bool] = {}

        async def idle(s: Any) -> None:
            await s.send_ping()
            await asyncio.sleep(0.5)  # the client's GET stream is open now
            _age(h, aid, 200.0)
            await asyncio.sleep(1.2)  # an idle agent between tasks, its MCP stream still open
            seen["connected"] = _connected(h, aid)

        asyncio.run(_mcp(base + "/mcp", {"Authorization": f"Bearer {tok}"}, idle))
        assert seen["connected"] is True
        # once the session is closed nothing touches it any more: it really is gone after the 90 s
        _age(h, aid, 200.0)
        time.sleep(0.8)
        assert not _connected(h, aid)
    finally:
        stop_host(h, auto)


def test_long_polls_answer_before_cloudflare_cuts_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.agentlink import service as S

    assert getattr(S, "LONG_POLL_MAX_S", 120.0) < CLOUDFLARE_IDLE_S
    monkeypatch.setattr(S, "LONG_POLL_MAX_S", 0.3, raising=False)
    hub = S.EventHub()
    done: dict[str, float] = {}

    def wait() -> None:
        t0 = time.monotonic()
        hub.wait(0, 120.0)  # the agent asked for 120 s
        done["s"] = time.monotonic() - t0

    th = threading.Thread(target=wait, daemon=True)
    th.start()
    th.join(5)
    assert "s" in done and done["s"] < 2.0


def test_the_allow_prompt_for_a_new_agent_fits_in_one_tunnel_request(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        waits: list[float] = []
        orig = h.core.prompts.create

        def create(question: str, *a: Any, **kw: Any) -> Any:
            waits.append(float(kw.get("timeout_s", 60.0)))
            return orig(question, *a, **kw)

        h.core.prompts.create = create
        pair_keep(h, base)
        assert waits and max(waits) < CLOUDFLARE_IDLE_S
    finally:
        stop_host(h, auto)


def test_event_stream_resumes_after_last_event_id(tmp_path: Path) -> None:
    from away.agentlink.api import KEEP_CONNECTED

    h, auto, base = start_host(tmp_path, KEEP_CONNECTED)
    try:
        tok = pair_keep(h, base)["persistent_token"]
        for i in range(3):
            h.core.events.publish("user.summon", {"n": i})
        last = h.core.events.seq
        h.core.events.publish("user.summon", {"n": "new"})
        req = urllib.request.Request(
            base + "/v1/events/stream",
            headers={"Authorization": f"Bearer {tok}", "Accept": "text/event-stream", "Last-Event-ID": str(last)},
        )
        with OPENER.open(req, timeout=30) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    break
        assert ev["seq"] == last + 1 and ev["data"]["n"] == "new"  # nothing seen twice, nothing missed
    finally:
        stop_host(h, auto)


class _Stream(io.BytesIO):
    def __init__(self, lines: list[str]) -> None:
        super().__init__("".join(lines).encode())
        self.status, self.headers = 200, {}

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()


def test_example_client_stream_reconnects_and_resumes(tmp_path: Path) -> None:
    mod = load_client()
    st = mod.State(tmp_path / "agent.json")
    st.update(url="https://pc.example", persistent_token="paway_" + "k" * 43)
    urls: list[tuple[str, str | None, float]] = []
    script: list[Any] = [
        [": keep-alive\n\n", 'id: 7\nevent: user.summon\ndata: {"seq": 7, "topic": "user.summon", "data": {}}\n\n'],
        TimeoutError("timed out"),  # the tunnel went silent: no keep-alive for 45 s
        ['data: {"seq": 8, "topic": "user.look", "data": {}}\n\n'],
    ]

    def opener(req: Any, timeout: float) -> Any:
        urls.append((req.full_url, req.headers.get("Last-event-id"), timeout))
        step = script.pop(0) if script else urllib.error.URLError("gone")
        if isinstance(step, Exception):
            raise step
        return _Stream(step)

    got: list[int] = []
    c = mod.Client(state=st, opener=opener)
    last = c.stream(0, on_event=lambda ev: got.append(ev["seq"]), sleep=lambda s: None, max_tries=2)
    assert got == [7, 8] and last == 8
    assert urls[1][0].endswith("since=7") and urls[1][1] == "7"  # resumed after the last event
    assert urls[2][0].endswith("since=7") and urls[3][0].endswith("since=8")
    assert all(t < 60 for _u, _h, t in urls)  # a dead link is noticed in under a minute, not after an hour


def test_example_client_stream_renews_access_on_401(tmp_path: Path) -> None:
    mod = load_client()
    st = mod.State(tmp_path / "agent.json")
    st.update(url="https://pc.example", token="pa_old", refresh_token="par_x")
    calls: list[str] = []

    def opener(req: Any, timeout: float) -> Any:
        calls.append(req.full_url)
        if req.full_url.endswith("/v1/token/refresh"):
            return _Stream([json.dumps({"token": "pa_new", "refresh_token": "par_y"})])
        if req.headers.get("Authorization") == "Bearer pa_old":
            raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b'{"error": "expired"}'))
        if len([u for u in calls if "/stream" in u]) == 2:
            return _Stream(['data: {"seq": 3, "topic": "x", "data": {}}\n\n'])
        raise urllib.error.URLError("gone")

    got: list[int] = []
    mod.Client(state=st, opener=opener).stream(
        0, on_event=lambda ev: got.append(ev["seq"]), sleep=lambda s: None, max_tries=1
    )
    assert got == [3] and st.get("token") == "pa_new"
