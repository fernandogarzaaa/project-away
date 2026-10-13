"""Agents, tokens and pairing codes.

* A token is 32 random bytes (``secrets``); only its SHA-256 is kept (in memory and in ``agents.json``), and a
  presented token is checked with :func:`hmac.compare_digest` against every stored hash (constant time per
  comparison, no early exit on the first match).
* A pairing link carries a random code (``/c/<code>``) plus a 6-digit PIN for typing. Neither grants anything:
  the first client that uses it creates a *pending* request that the user must approve on the PC. A pairing is
  single-use (approved or denied, it is spent) and expires after ``pairing_ttl_s`` (default 10 minutes).
* MCP connectors that cannot send a header (ChatGPT / Claude.ai "add custom connector" with a plain URL) keep
  using the pairing URL after approval: the code is then bound to that one agent (``link_hash``) until the agent
  expires or is revoked. Clients that can send headers exchange it for a bearer token instead.
"""

from __future__ import annotations

import builtins
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from away.agentlink import perms

TOKEN_PREFIX = "pa_"
CODE_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"  # no 0/o/1/l
NAME_RE = re.compile(r"[^\w .,'()+-]", re.UNICODE)


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def same(a: str, b: str) -> bool:
    """Constant-time string comparison."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def new_code(n: int = 10) -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(n))


def new_pin() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def clean_name(name: Any, fallback: str = "AI agent") -> str:
    """An agent's display name (from MCP clientInfo or a pairing request): printable, short, never empty."""
    text = NAME_RE.sub("", str(name or "")).strip()[:40]
    return text or fallback


@dataclass
class Agent:
    id: str
    name: str
    kind: str  # local (stdio bridge on this PC) | remote (through the pairing link)
    scopes: list[str]
    token_hash: str
    created: float
    expires: float
    last_seen: float = 0.0
    revoked: bool = False
    link_hash: str = ""  # the pairing code this connector keeps using as its URL (remote only)
    always_allow: list[str] = field(default_factory=list)  # risky-action categories the user always allows
    client: dict[str, Any] = field(default_factory=dict)  # MCP clientInfo (name, version)

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def active(self, now: float) -> bool:
        return not self.revoked and now < self.expires

    def public(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "scopes": perms.to_list(self.scopes),
            "permissions": perms.describe(self.scopes),
            "created": self.created,
            "expires": self.expires,
            "expires_in_s": max(0, int(self.expires - now)),
            "last_seen": self.last_seen,
            "connected": self.active(now) and now - self.last_seen < 90,
            "always_allow": list(self.always_allow),
            "client": dict(self.client),
        }


@dataclass
class Pairing:
    code: str
    pin: str
    scopes: list[str]
    created: float
    expires: float
    kind: str = "remote"
    used: bool = False
    pending: dict[str, Any] | None = None  # {"name":..., "client":..., "since":...} while waiting for approval
    agent_id: str = ""
    denied: bool = False

    def open(self, now: float) -> bool:
        return not self.used and now < self.expires


class AgentRegistry:
    """Thread-safe store of agents and pairings. Persists agents (hashes only) to ``path`` when given."""

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        self._lock = threading.RLock()
        self.agents: dict[str, Agent] = {}
        self.pairings: list[Pairing] = []
        self._load()

    # ------------------------------------------------------------- persistence
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data.get("agents", []):
                a = Agent(**{k: v for k, v in item.items() if k in Agent.__dataclass_fields__})
                self.agents[a.id] = a
        except Exception:
            self.agents = {}  # a corrupt file means everyone pairs again; never crash the app over it

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            now = self.clock()
            keep = [asdict(a) for a in self.agents.values() if a.active(now)]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"agents": keep}, indent=1), encoding="utf-8")
        if os.name != "nt":
            os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ agents
    def issue(
        self,
        name: str,
        scopes: Iterable[str],
        ttl_s: float,
        kind: str = "remote",
        client: dict[str, Any] | None = None,
        link_code: str = "",
    ) -> tuple[Agent, str]:
        token = new_token()
        now = self.clock()
        a = Agent(
            id=uuid.uuid4().hex[:12],
            name=clean_name(name),
            kind=kind,
            scopes=perms.to_list(perms.normalize(list(scopes))),
            token_hash=digest(token),
            created=now,
            expires=now + float(ttl_s),
            last_seen=now,
            link_hash=digest(link_code) if link_code else "",
            client=dict(client or {}),
        )
        with self._lock:
            self.agents[a.id] = a
        self.save()
        return a, token

    def authenticate(self, token: str | None) -> Agent | None:
        """The active agent holding ``token`` (constant-time comparison against every stored hash)."""
        if not token:
            return None
        h = digest(token)
        found: Agent | None = None
        now = self.clock()
        with self._lock:
            for a in self.agents.values():
                if same(a.token_hash, h) and found is None:  # no early exit: compare every hash
                    found = a
            if found is None or not found.active(now):
                return None
            found.last_seen = now
        return found

    def authenticate_link(self, code: str | None) -> Agent | None:
        """A URL-bound connector (the pairing code it was approved with)."""
        if not code:
            return None
        h = digest(code)
        found: Agent | None = None
        now = self.clock()
        with self._lock:
            for a in self.agents.values():
                if a.link_hash and same(a.link_hash, h) and found is None:
                    found = a
            if found is None or not found.active(now):
                return None
            found.last_seen = now
        return found

    def get(self, agent_id: str) -> Agent | None:
        return self.agents.get(agent_id)

    def list(self, include_inactive: bool = False) -> builtins.list[Agent]:
        now = self.clock()
        with self._lock:
            return [a for a in self.agents.values() if include_inactive or a.active(now)]

    def revoke(self, agent_id: str) -> bool:
        with self._lock:
            a = self.agents.get(agent_id)
            if a is None or a.revoked:
                return False
            a.revoked = True
        self.save()
        return True

    def revoke_all(self) -> int:
        with self._lock:
            n = 0
            for a in self.agents.values():
                if not a.revoked:
                    a.revoked = True
                    n += 1
            for p in self.pairings:
                p.used = True
        self.save()
        return n

    def set_scopes(self, agent_id: str, scopes: Iterable[str]) -> Agent:
        with self._lock:
            a = self.agents.get(agent_id)
            if a is None:
                raise KeyError(agent_id)
            a.scopes = perms.to_list(perms.normalize(list(scopes)))
        self.save()
        return a

    def allow_always(self, agent_id: str, category: str) -> None:
        with self._lock:
            a = self.agents.get(agent_id)
            if a is not None and category not in a.always_allow:
                a.always_allow.append(category)
        self.save()

    # ---------------------------------------------------------------- pairings
    def new_pairing(self, scopes: Iterable[str], ttl_s: float = 600.0, kind: str = "remote") -> Pairing:
        now = self.clock()
        p = Pairing(new_code(), new_pin(), perms.to_list(perms.normalize(list(scopes))), now, now + ttl_s, kind)
        with self._lock:
            self.pairings = [q for q in self.pairings if q.open(now)] + [p]
        return p

    def find_pairing(self, code: str | None = None, pin: str | None = None) -> Pairing | None:
        """An open pairing by its link code or its 6-digit PIN (constant-time comparisons)."""
        now = self.clock()
        found: Pairing | None = None
        with self._lock:
            for p in self.pairings:
                hit = (code is not None and same(p.code, str(code))) or (pin is not None and same(p.pin, str(pin)))
                if hit and found is None:
                    found = p
        return found if found is not None and found.open(now) else None

    def pairing_by_code_any(self, code: str) -> Pairing | None:
        with self._lock:
            for p in self.pairings:
                if same(p.code, code):
                    return p
        return None

    def mark_pending(self, p: Pairing, name: str, client: dict[str, Any] | None = None) -> bool:
        """First use of a pairing: remember who asked. False when someone else already claimed it."""
        with self._lock:
            if not p.open(self.clock()):
                return False
            if p.pending is not None:
                return clean_name(name) == p.pending["name"]
            p.pending = {"name": clean_name(name), "client": dict(client or {}), "since": self.clock()}
            return True

    def approve(self, p: Pairing, ttl_s: float, bind_link: bool = False) -> tuple[Agent, str]:
        with self._lock:
            if not p.open(self.clock()) or p.pending is None:
                raise ValueError("this pairing link was already used or has expired")
            p.used = True
            name, client = p.pending["name"], p.pending.get("client") or {}
        agent, token = self.issue(name, p.scopes, ttl_s, p.kind, client, link_code=p.code if bind_link else "")
        p.agent_id = agent.id
        return agent, token

    def deny(self, p: Pairing) -> None:
        with self._lock:
            p.used = True
            p.denied = True

    def open_pairings(self) -> builtins.list[Pairing]:
        now = self.clock()
        with self._lock:
            return [p for p in self.pairings if p.open(now)]
