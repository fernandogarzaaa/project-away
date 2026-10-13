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
# v0.1.23: an agent whose access ended (expired or disconnected) is remembered this long without any token, so
# pairing again under the same name keeps its wake word, default-agent choice and inbox instead of a duplicate
REMEMBER_S = 30 * 86400.0
REFRESH_PREFIX = "par_"
REFRESH_GRACE_S = 120.0  # the refresh token before the newest still works this long (a reply lost on the way)
DEFAULT_REFRESH_TTL_S = 30 * 86400.0


class RefreshError(PermissionError):
    """A refresh token that is unknown, expired or revoked; the message says what to do in plain words."""


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
    # v0.1.22 "Talk to your agent": the person's own wake word for this agent ("" = "Hey <name>"), and the optional
    # "wake me" address a message is POSTed to (set by the person in Settings > Agent Link; off by default)
    wake_word: str = ""
    wake_url: str = ""
    paired_again: int = 0  # v0.1.23: times this agent paired again under the same name (settings kept)
    # v0.1.24 "a link that survives restarts": a long-lived refresh token (only its SHA-256 is kept) renews the
    # short access token after Away restarts, the PC reboots or the access expires. It lasts refresh_ttl_days after
    # its last use and is rotated on every use; the one before stays valid for REFRESH_GRACE_S (a lost reply).
    refresh_hash: str = ""
    refresh_expires: float = 0.0
    prev_refresh_hash: str = ""
    prev_refresh_until: float = 0.0
    # where the agent looks up a moved address (rendezvous): a random topic and the key that signs announcements
    rdv_id: str = ""
    rdv_key: str = ""

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def active(self, now: float) -> bool:
        return not self.revoked and now < self.expires

    def can_refresh(self, now: float) -> bool:
        """v0.1.24: not revoked and its refresh token (or, for a connector URL, its link) is still valid."""
        return not self.revoked and bool(self.refresh_hash or self.link_hash) and now < self.refresh_expires

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
            "wake_word": self.wake_word,
            "wake_phrase": self.wake_phrase(),
            "wake_me": bool(self.wake_url),
            # v0.1.24: until when it can reconnect by itself (0 = it has to pair again once access ends)
            "stays_connected_until": self.refresh_expires if self.can_refresh(now) else 0.0,
        }

    def wake_phrase(self) -> str:
        """What the person says to talk to this agent: "Hey Claude" (or their own wake word)."""
        return self.wake_word.strip() or f"Hey {self.name}"


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

    def __init__(
        self,
        path: Path | None = None,
        clock: Callable[[], float] = time.time,
        refresh_ttl_s: float = DEFAULT_REFRESH_TTL_S,
        link_access_s: float = 3600.0,
    ) -> None:
        self.path = path
        self.clock = clock
        self.refresh_ttl_s = float(refresh_ttl_s)  # v0.1.24 (agent_link.refresh_ttl_days)
        self.link_access_s = float(link_access_s)  # access a connector URL gets when it renews itself
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

    @staticmethod
    def _ended(a: Agent) -> float:
        return a.expires  # revoke() moves expires to the moment it was disconnected

    def _forget_old(self, now: float) -> None:
        for aid in [
            i
            for i, a in self.agents.items()
            if not a.active(now)
            and not a.can_refresh(now)
            and now - max(self._ended(a), a.refresh_expires) > REMEMBER_S
        ]:
            del self.agents[aid]

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            now = self.clock()
            self._forget_old(now)
            keep = []
            for a in self.agents.values():
                d = asdict(a)
                if not a.active(now):  # remembered for its settings only: no access token survives
                    d["token_hash"] = ""
                    if not a.can_refresh(now):  # ... and nothing that could renew it either
                        d["link_hash"] = ""
                        d.update(refresh_hash="", prev_refresh_hash="", refresh_expires=0.0, prev_refresh_until=0.0)
                keep.append(d)
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
        cname = clean_name(name)
        with self._lock:
            old = self._same_name(cname, kind, now)
            if old is not None:  # v0.1.23: pairing again under the same name keeps its id and settings
                if old.revoked:
                    old.always_allow = []  # the person disconnected it: risky actions are asked again
                old.revoked = False
                old.kind = kind
                old.scopes = perms.to_list(perms.normalize(list(scopes)))
                old.token_hash = digest(token)
                old.expires = now + float(ttl_s)
                old.last_seen = now
                old.link_hash = digest(link_code) if link_code else ""
                old.client = dict(client or {})
                old.paired_again += 1
                _drop_refresh(old)  # a new pairing gets a new refresh token (new_refresh)
                if link_code:
                    old.refresh_expires = now + self.refresh_ttl_s
        if old is not None:
            self.save()
            return old, token
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
            refresh_expires=now + self.refresh_ttl_s if link_code else 0.0,  # v0.1.24: connector URLs renew
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
            if found is None:
                return None
            renewed = False
            if not found.active(now):
                # v0.1.24: a connector URL cannot send a refresh token; the link itself renews access while its
                # refresh window lasts (each use moves the window), so a restart does not break it
                if not found.can_refresh(now) or self.link_access_s <= 0:
                    return None
                found.expires = now + self.link_access_s
                found.refresh_expires = now + self.refresh_ttl_s
                renewed = True
            found.last_seen = now
        if renewed:
            self.save()
        return found

    def _same_name(self, name: str, kind: str, now: float) -> Agent | None:
        """The agent a new pairing under ``name`` continues: same name (any case) and kind, or the same name with
        no access right now (never an active agent of another kind: Claude Desktop and Claude.ai stay two)."""
        want = name.strip().lower()
        hits = [a for a in self.agents.values() if a.name.strip().lower() == want]
        for a in sorted(hits, key=lambda x: (x.kind != kind, not x.active(now), -x.last_seen)):
            if a.kind == kind or not a.active(now):
                return a
        return None

    def known(self) -> builtins.list[Agent]:
        """Every agent with access now plus the remembered ones (access ended within :data:`REMEMBER_S`, or it can
        still reconnect with its refresh token)."""
        now = self.clock()
        with self._lock:
            return [
                a
                for a in self.agents.values()
                if a.active(now) or a.can_refresh(now) or now - self._ended(a) <= REMEMBER_S
            ]

    def reconnectable(self) -> builtins.list[Agent]:
        """v0.1.24: agents that can reach this PC again without pairing (active, or a valid refresh token / link)."""
        now = self.clock()
        with self._lock:
            return [a for a in self.agents.values() if a.active(now) or a.can_refresh(now)]

    # ------------------------------------------------------------- refresh (v0.1.24)
    def new_refresh(self, agent_id: str) -> str:
        """A fresh refresh token for this agent (the old one stops working at once). Also gives the agent its
        rendezvous topic + signing key the first time."""
        token = REFRESH_PREFIX + secrets.token_urlsafe(32)
        now = self.clock()
        with self._lock:
            a = self.agents.get(agent_id)
            if a is None:
                raise KeyError(f"unknown agent {agent_id!r}")
            a.refresh_hash = digest(token)
            a.refresh_expires = now + self.refresh_ttl_s
            a.prev_refresh_hash, a.prev_refresh_until = "", 0.0
            if not a.rdv_id:
                a.rdv_id = new_code(22)
                a.rdv_key = secrets.token_urlsafe(24)
        self.save()
        return token

    def refresh(
        self, refresh_token: str | None, access_ttl_s: float | Callable[[Agent], float]
    ) -> tuple[Agent, str, str]:
        """Exchange a refresh token for a new access token and a new refresh token (rotation). Constant-time over
        every agent. Raises :class:`RefreshError` with a plain message."""
        if not refresh_token or not str(refresh_token).startswith(REFRESH_PREFIX):
            raise RefreshError("missing or malformed refresh token")
        h = digest(str(refresh_token))
        now = self.clock()
        found: Agent | None = None
        old = False
        with self._lock:
            for a in self.agents.values():
                hit_new = bool(a.refresh_hash) and same(a.refresh_hash, h)
                hit_old = bool(a.prev_refresh_hash) and same(a.prev_refresh_hash, h)
                if (hit_new or hit_old) and found is None:
                    found, old = a, (hit_old and not hit_new)
            if found is None:
                raise RefreshError("unknown refresh token; pair again in Project Away > Agent Link")
            if found.revoked:
                raise RefreshError("this agent was disconnected on the PC; pair again to reconnect")
            if old and now > found.prev_refresh_until:
                raise RefreshError("this refresh token was already used; use the newest one or pair again")
            if now >= found.refresh_expires:
                raise RefreshError("the refresh token expired; pair again in Project Away > Agent Link")
            access = new_token()
            fresh = REFRESH_PREFIX + secrets.token_urlsafe(32)
            ttl = access_ttl_s(found) if callable(access_ttl_s) else access_ttl_s
            found.token_hash = digest(access)
            found.expires = now + float(ttl)
            found.last_seen = now
            if not old:  # the token just used stays valid briefly, in case this reply never arrives
                found.prev_refresh_hash, found.prev_refresh_until = found.refresh_hash, now + REFRESH_GRACE_S
            found.refresh_hash = digest(fresh)
            found.refresh_expires = now + self.refresh_ttl_s
        self.save()
        return found, access, fresh

    def get(self, agent_id: str) -> Agent | None:
        return self.agents.get(agent_id)

    def list(self, include_inactive: bool = False) -> builtins.list[Agent]:
        now = self.clock()
        with self._lock:
            return [a for a in self.agents.values() if include_inactive or a.active(now)]

    def set_talk(self, agent_id: str, wake_word: str | None = None, wake_url: str | None = None) -> Agent:
        """v0.1.22: the person's wake word for this agent and its "wake me" address (None = unchanged, "" = off)."""
        from away.agentlink.inbox import check_wake_url

        with self._lock:
            a = self.agents.get(agent_id)
            if a is None:
                raise KeyError(f"unknown agent {agent_id!r}")
            if wake_word is not None:
                w = NAME_RE.sub("", str(wake_word)).strip()[:40]
                a.wake_word = w
            if wake_url is not None:
                a.wake_url = check_wake_url(wake_url)
        self.save()
        return a

    def revoke(self, agent_id: str) -> bool:
        with self._lock:
            a = self.agents.get(agent_id)
            if a is None or a.revoked:
                return False
            a.revoked = True
            a.expires = min(a.expires, self.clock())
            _drop_refresh(a)
        self.save()
        return True

    def revoke_all(self) -> int:
        with self._lock:
            n = 0
            for a in self.agents.values():
                if not a.revoked:
                    a.revoked = True
                    a.expires = min(a.expires, self.clock())
                    n += 1
                _drop_refresh(a)
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


def _drop_refresh(a: Agent) -> None:
    a.refresh_hash = ""
    a.prev_refresh_hash = ""
    a.refresh_expires = 0.0
    a.prev_refresh_until = 0.0
