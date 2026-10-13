"""Talk to your agent (v0.1.22): the agent-agnostic inbox.

The person says "Hey <agent name>, ..." (or uses the summon gesture, then speaks; see :mod:`away.voice.talk`).
The words become a message in *that agent's* inbox only, a ``user.message`` event only that agent can see, and,
when the person gave the agent a "wake me" address, a small JSON POST to it so agents that sleep between turns get
woken up. "Hey agent" and the summon gesture go to the default agent (Settings > Agent Link), or to every
connected agent when no default is set.

Agents read and answer through the MCP ``inbox`` tool (list / read / reply), ``GET /v1/inbox`` and the
``away://inbox`` resource; a reply is shown on screen and read aloud (same as ``notify_user`` with speak).

v0.1.23: the inbox is kept on disk (``<data>/agentlink/inbox.json``, readable only by you) so a message waits when
the agent is not connected, survives a restart or an expired link, and is delivered when the agent pairs again
under the same name. At most ``MAX_MESSAGES`` messages, none older than ``MAX_AGE_S`` (7 days). The audio itself is
never stored.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MAX_MESSAGES = 200
MAX_AGE_S = 7 * 86400.0
MAX_TEXT = 2000
WAKE_TIMEOUT_S = 5.0


@dataclass
class Message:
    id: str
    t: float
    text: str
    source: str  # voice | gesture
    to: list[str]  # agent ids
    to_names: list[str]
    read_by: list[str] = field(default_factory=list)
    replies: list[dict[str, Any]] = field(default_factory=list)

    def public(self, agent_id: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "t": self.t,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t)),
            "text": self.text,
            "source": self.source,
            "to": list(self.to_names),
            "shared": len(self.to) > 1,  # also sent to the person's other agents ("Hey agent" with no default)
        }
        if agent_id is not None:
            out["read"] = agent_id in self.read_by
            out["replied"] = any(r.get("agent_id") == agent_id for r in self.replies)
        return out

    def is_for(self, agent_id: str, name: str = "") -> bool:
        """Addressed to this agent: by id, or (v0.1.23) by name, so an agent that paired again under the same
        name finds what was said while it was away."""
        if agent_id in self.to:
            return True
        n = str(name or "").strip().lower()
        return bool(n) and any(str(x).strip().lower() == n for x in self.to_names)


def _attr(a: Any, name: str, default: Any = "") -> Any:
    return a.get(name, default) if isinstance(a, dict) else getattr(a, name, default)


def connected(a: Any, now: float) -> bool:
    if isinstance(a, dict):
        return bool(a.get("connected", False))
    try:
        return bool(a.public(now).get("connected"))
    except Exception:
        return False


def resolve_targets(target: str, agents: Sequence[Any], default_agent: str = "", now: float | None = None) -> list[Any]:
    """Who gets a message. ``target``: an agent id (from "Hey Claude") or "" (generic: "Hey agent", the summon
    gesture). Generic messages go to ``default_agent`` (a name or id, case-insensitive) when it is one of
    ``agents``; otherwise to every *connected* agent, or, when none is connected right now, to every agent (they
    find it in their inbox when they come back)."""
    now = time.time() if now is None else now
    if target:
        return [a for a in agents if _attr(a, "id") == target]
    want = str(default_agent or "").strip().lower()
    if want:
        hit = [a for a in agents if want in (str(_attr(a, "id")).lower(), str(_attr(a, "name")).lower())]
        if hit:
            return hit[:1]
    live = [a for a in agents if connected(a, now)]
    return live or list(agents)


def check_wake_url(url: str) -> str:
    """A "wake me" address: https anywhere, or http only on this computer (127.0.0.1 / localhost). "" turns it
    off. Raises ValueError with a plain message otherwise."""
    u = str(url or "").strip()
    if not u:
        return ""
    if len(u) > 500:
        raise ValueError("that address is too long")
    p = urlparse(u)
    if p.scheme not in ("https", "http") or not p.hostname:
        raise ValueError("the wake-up address must start with https://")
    if p.scheme == "http":
        host = p.hostname.lower()
        local = host == "localhost"
        with contextlib.suppress(ValueError):
            local = local or ipaddress.ip_address(host).is_loopback
        if not local:
            raise ValueError("use https:// (plain http:// only works for an address on this computer)")
    return u


def post_json(url: str, payload: dict[str, Any], timeout: float = WAKE_TIMEOUT_S) -> int:  # pragma: no cover - net
    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ProjectAway-AgentLink"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status)


class Inbox:
    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        post: Callable[[str, dict[str, Any]], Any] = post_json,
        background: bool = True,
        max_messages: int = MAX_MESSAGES,
        path: Path | None = None,
        max_age_s: float = MAX_AGE_S,
    ) -> None:
        self.clock = clock
        self.post = post
        self.background = background  # wake-up POSTs on a thread (tests: inline)
        self.messages: deque[Message] = deque(maxlen=max_messages)
        self.wake_log: deque[dict[str, Any]] = deque(maxlen=50)  # (agent, ok/error) - shown in Advanced
        self._lock = threading.Lock()
        self.path = Path(path) if path is not None else None
        self.max_age_s = float(max_age_s)
        self._load()

    # ------------------------------------------------------------ disk (v0.1.23)
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data.get("messages", []):
                m = Message(**{k: v for k, v in item.items() if k in Message.__dataclass_fields__})
                self.messages.append(m)
        except Exception:
            self.messages.clear()  # a damaged file is dropped, never a crash
        self._prune()

    def _prune(self) -> None:
        cutoff = self.clock() - self.max_age_s
        with self._lock:
            keep = [m for m in self.messages if float(m.t) >= cutoff]
            if len(keep) != len(self.messages):
                self.messages.clear()
                self.messages.extend(keep)

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {"version": 1, "messages": [asdict(m) for m in self.messages]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            if os.name != "nt":
                os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def add(self, text: str, targets: Iterable[Any], source: str = "voice") -> Message:
        ts = list(targets)
        m = Message(
            id=uuid.uuid4().hex[:10],
            t=self.clock(),
            text=str(text).strip()[:MAX_TEXT],
            source=source,
            to=[str(_attr(a, "id")) for a in ts],
            to_names=[str(_attr(a, "name")) for a in ts],
        )
        self._prune()
        with self._lock:
            self.messages.append(m)
        self.save()
        for a in ts:
            url = str(_attr(a, "wake_url", "") or "")
            if url:
                self.wake(a, url, m)
        return m

    def wake(self, agent: Any, url: str, m: Message) -> None:
        payload = {
            "event": "user.message",
            "id": m.id,
            "text": m.text,
            "agent": _attr(agent, "name"),
            "t": m.t,
            "source": m.source,
            "how_to_reply": "Agent Link inbox tool: action=reply, or notify_user with speak=true",
        }
        name = str(_attr(agent, "name"))

        def run() -> None:
            try:
                self.post(url, payload)
                self.wake_log.append({"agent": name, "t": self.clock(), "ok": True})
            except Exception as exc:
                self.wake_log.append({"agent": name, "t": self.clock(), "ok": False, "error": str(exc)[:200]})

        if self.background:
            threading.Thread(target=run, daemon=True, name="agentlink-wake").start()
        else:
            run()

    def for_agent(self, agent_id: str, unread_only: bool = False, name: str = "") -> list[Message]:
        with self._lock:
            ms = [m for m in self.messages if m.is_for(agent_id, name)]
        if unread_only:
            ms = [m for m in ms if agent_id not in m.read_by]
        return list(reversed(ms))

    def get(self, agent_id: str, msg_id: str, name: str = "") -> Message:
        with self._lock:
            for m in self.messages:
                if m.id == str(msg_id) and m.is_for(agent_id, name):
                    return m
        raise KeyError(f"no message {msg_id!r} in your inbox (inbox action=list shows them)")

    def mark_read(self, agent_id: str, m: Message) -> None:
        with self._lock:
            changed = agent_id not in m.read_by
            if changed:
                m.read_by.append(agent_id)
        if changed:
            self.save()

    def unread(self, agent_id: str, name: str = "") -> int:
        return len(self.for_agent(agent_id, unread_only=True, name=name))

    def unread_counts(self, agents: Iterable[Any]) -> dict[str, int]:
        """v0.1.23: unread messages per agent id (shown in Settings > Agent Link)."""
        return {str(_attr(a, "id")): self.unread(str(_attr(a, "id")), str(_attr(a, "name"))) for a in agents}

    def reply(self, agent_id: str, agent_name: str, msg_id: str, text: str) -> Message:
        m = self.get(agent_id, msg_id, agent_name)
        with self._lock:
            m.replies.append({"agent_id": agent_id, "agent": agent_name, "text": str(text)[:300], "t": self.clock()})
            if agent_id not in m.read_by:
                m.read_by.append(agent_id)
        self.save()
        return m


def sent_label(names: Sequence[str]) -> str:
    """ "Sent to Claude" / "Sent to Claude and Cursor" / "Sent to Claude, Cursor and ChatGPT"."""
    ns = [n for n in names if n]
    if not ns:
        return ""
    if len(ns) == 1:
        return f"Sent to {ns[0]}"
    return f"Sent to {', '.join(ns[:-1])} and {ns[-1]}"
