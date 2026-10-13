"""A link that survives restarts (v0.1.24): tell every paired agent where Project Away can be reached now.

The free Cloudflare quick tunnel (and the backup SSH relays) give a *new random address* every time the link
starts: after Away restarts, the PC reboots or the network drops. A paired agent keeps its refresh token (see
:mod:`away.agentlink.auth`), so all it needs is the new address. It is published three ways, to every agent that
can still reconnect (local agents on this PC are skipped: they find the app through ``host.json``):

1. the agent's **wake-up address** (webhook), when the person set one: a JSON POST ``{"event": "link.moved", ...}``;
2. the agent's **inbox** (a short system note it reads the next time it connects);
3. a **rendezvous** topic (``agent_link.rendezvous: ntfy``, on by default): a message on a free public relay
   (``https://ntfy.sh/<random topic>``) that the agent polls when the old address stops answering. Each agent has
   its own random 22-character topic and its own signing key, both handed over once at pairing, and every message
   is signed (HMAC-SHA256), so nobody else can send an agent to a fake address. ntfy.sh keeps messages about 12
   hours, so the current address is published again every :data:`REPUBLISH_S` while the link is up.

What a stranger who guessed a topic would learn: the current tunnel address, which on its own grants nothing (every
call needs a token; pairing needs a code and the person's Allow). ``rendezvous: off`` turns the relay off.

Trade-offs of the link types (README "A link that survives restarts" has the full table):

* standard (Cloudflare quick tunnel): free, no account, new address on every start -> rendezvous needed;
* backup (localhost.run / Pinggy over SSH): free, no account, new address on every start (Pinggy: every hour);
* named (your own Cloudflare tunnel): free Cloudflare account + a domain on Cloudflare, the address never changes.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REPUBLISH_S = 6 * 3600.0
TOPIC_PREFIX = "away-"
DEFAULT_SERVER = "https://ntfy.sh"
HTTP_TIMEOUT_S = 8.0


def topic(rdv_id: str) -> str:
    return TOPIC_PREFIX + str(rdv_id)


def signing_text(rdv_id: str, url: str, t: int) -> str:
    return f"v1|{rdv_id}|{url}|{int(t)}"


def sign(rdv_id: str, key: str, url: str, t: int) -> str:
    return hmac.new(str(key).encode("utf-8"), signing_text(rdv_id, url, t).encode("utf-8"), hashlib.sha256).hexdigest()


def announcement(agent: Any, url: str, t: float | None = None) -> dict[str, Any]:
    """The signed message for one agent: ``{"v": 1, "kind": "away.link", "url", "t", "sig"}``."""
    ts = int(time.time() if t is None else t)
    rid, key = str(getattr(agent, "rdv_id", "") or ""), str(getattr(agent, "rdv_key", "") or "")
    return {"v": 1, "kind": "away.link", "url": url, "t": ts, "sig": sign(rid, key, url, ts) if key else ""}


def verify(msg: Any, rdv_id: str, key: str, max_age_s: float = 14 * 86400.0, now: float | None = None) -> str | None:
    """The address in a signed announcement, or None when the signature, the shape or the age is wrong."""
    if not isinstance(msg, dict) or msg.get("kind") != "away.link":
        return None
    url, t, sig = str(msg.get("url") or ""), msg.get("t"), str(msg.get("sig") or "")
    if not url.startswith("https://") or not isinstance(t, int) or isinstance(t, bool) or not sig:
        return None
    now = time.time() if now is None else now
    if now - t > max_age_s or t - now > 600:
        return None
    if not hmac.compare_digest(sig, sign(rdv_id, key, url, t)):
        return None
    return url


def agent_info(agent: Any, server: str = "") -> dict[str, Any]:
    """What the agent keeps to find a moved address (handed over at pairing and with every refresh)."""
    rid = str(getattr(agent, "rdv_id", "") or "")
    if not rid:
        return {}
    srv = (server or DEFAULT_SERVER).rstrip("/")
    return {
        "type": "ntfy",
        "server": srv,
        "topic": topic(rid),
        "id": rid,
        "key": str(getattr(agent, "rdv_key", "") or ""),
        "poll": f"{srv}/{topic(rid)}/json?poll=1&since=24h",
        "verify": "HMAC-SHA256(key, 'v1|<id>|<url>|<t>') == sig",
    }


def post_raw(url: str, body: bytes, timeout: float = HTTP_TIMEOUT_S) -> int:  # pragma: no cover - network
    import urllib.request

    req = urllib.request.Request(
        url, data=body, method="POST", headers={"User-Agent": "ProjectAway-AgentLink", "Content-Type": "text/plain"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status)


def post_json(url: str, payload: dict[str, Any], timeout: float = HTTP_TIMEOUT_S) -> int:  # pragma: no cover - net
    from away.agentlink.inbox import post_json as pj

    return pj(url, payload, timeout)


class LinkAnnouncer:
    """Publishes the public address to every agent that can reconnect, once per new address (and again every
    :data:`REPUBLISH_S` on the rendezvous relay). Never raises; :attr:`log` keeps what happened per agent."""

    def __init__(
        self,
        agents: Callable[[], Iterable[Any]],
        *,
        inbox: Any = None,
        events: Any = None,
        settings: Callable[[], Any] = lambda: None,
        state_path: Path | None = None,
        post: Callable[[str, dict[str, Any]], Any] = post_json,
        post_raw: Callable[[str, bytes], Any] = post_raw,
        clock: Callable[[], float] = time.time,
        background: bool = True,
    ) -> None:
        self.agents = agents
        self.inbox = inbox
        self.events = events
        self.settings = settings
        self.state_path = state_path
        self.post = post
        self.post_raw = post_raw
        self.clock = clock
        self.background = background
        self.log: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.state = self._load()

    # ---------------------------------------------------------------- state
    def _load(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        if self.state_path is None:
            return
        with contextlib.suppress(OSError):
            from away.agentlink.host import write_private

            write_private(self.state_path, json.dumps(self.state))

    @property
    def last_url(self) -> str:
        return str(self.state.get("url") or "")

    @property
    def last_type(self) -> str:
        return str(self.state.get("link_type") or "")

    def _s(self, name: str, default: Any) -> Any:
        try:
            v = getattr(self.settings(), name)
            return default if v is None else v
        except Exception:
            return default

    def remote_agents(self) -> list[Any]:
        try:
            return [a for a in self.agents() if getattr(a, "kind", "remote") != "local"]
        except Exception:
            return []

    # -------------------------------------------------------------- announce
    def link_up(self, url: str, link_type: str = "standard") -> list[dict[str, Any]]:
        """The link is up at ``url``. Announces it when it differs from the last one announced (or the relay copy
        is due again). Returns the per-agent results (empty when nothing had to be sent)."""
        url = str(url or "").rstrip("/")
        if not url.startswith("https://"):
            return []
        now = self.clock()
        moved = url != self.last_url
        due = now - float(self.state.get("published") or 0) >= REPUBLISH_S
        if not moved and not due:
            return []
        prev = self.last_url
        self.state.update(url=url, link_type=link_type, published=now)
        if moved:
            self.state["moved_from"] = prev
        self._save()
        return self.announce(url, moved=moved, previous=prev)

    def announce(self, url: str, moved: bool = True, previous: str = "") -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        rdv_on = str(self._s("rendezvous", "ntfy")) == "ntfy"
        server = str(self._s("rendezvous_server", DEFAULT_SERVER) or DEFAULT_SERVER).rstrip("/")
        targets = self.remote_agents()
        jobs: list[Callable[[], None]] = []
        for a in targets:
            row: dict[str, Any] = {"agent": getattr(a, "name", "?"), "webhook": "none", "inbox": False}
            msg = announcement(a, url, self.clock())
            wake = str(getattr(a, "wake_url", "") or "")
            if moved and wake:
                payload = {
                    "event": "link.moved",
                    "agent": getattr(a, "name", ""),
                    "url": url,
                    "previous": previous,
                    "signed": msg,
                    "how_to_reconnect": "POST <url>/v1/token/refresh with your refresh token",
                }
                jobs.append(self._sender(row, "webhook", lambda w=wake, p=payload: self.post(w, p)))
            if moved and self.inbox is not None:
                with contextlib.suppress(Exception):
                    self.inbox.add(
                        f"Project Away has a new address: {url} (renew your access with your refresh token)",
                        [a],
                        source="system",
                        wake=False,
                    )
                    row["inbox"] = True
            if rdv_on and getattr(a, "rdv_id", ""):
                body = json.dumps(msg).encode("utf-8")
                dest = f"{server}/{topic(a.rdv_id)}"
                jobs.append(self._sender(row, "rendezvous", lambda d=dest, b=body: self.post_raw(d, b)))
            else:
                row["rendezvous"] = "off" if not rdv_on else "none"
            results.append(row)
        if moved and self.events is not None:
            with contextlib.suppress(Exception):
                self.events.publish("link.moved", {"url": url, "previous": previous, "agents": len(targets)})
        self._run(jobs)
        with self._lock:
            self.log = [*self.log, {"t": self.clock(), "url": url, "moved": moved, "agents": results}][-20:]
        return results

    def _sender(self, row: dict[str, Any], field: str, fn: Callable[[], Any]) -> Callable[[], None]:
        row[field] = "sending"

        def run() -> None:
            try:
                fn()
                row[field] = "ok"
            except Exception as exc:
                row[field] = f"failed: {str(exc)[:120]}"

        return run

    def _run(self, jobs: list[Callable[[], None]]) -> None:
        if not jobs:
            return
        if not self.background:
            for j in jobs:
                j()
            return

        def all_jobs() -> None:
            for j in jobs:
                j()

        threading.Thread(target=all_jobs, daemon=True, name="agentlink-announce").start()
