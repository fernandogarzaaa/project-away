"""HTTP side of Agent Link: one ASGI app on 127.0.0.1 serving

* ``/mcp`` - MCP over Streamable HTTP, ``Authorization: Bearer <token>``;
* ``/c/<code>`` - the pairing link. Opened in a browser it shows a short help page; used as an MCP connector URL
  (ChatGPT / Claude "add custom connector") the first ``initialize`` asks the person on the PC "<agent> wants to
  connect: Allow / Deny"; once allowed the link keeps working for that one agent until it expires or is revoked;
* ``/v1/...`` - a plain REST/JSON API with the same tools for agents that do not speak MCP (OpenAPI spec at
  ``/openapi.json``), ``POST /v1/rpc/<tool>`` for any tool, ``GET /v1/events/stream`` (server-sent events);
* ``POST /v1/pair`` (pairing code or 6-digit PIN -> token after approval) and ``POST /v1/pair/local`` (the stdio
  bridge on this PC, proven by the local key file only this user can read). v0.1.24: the approval offers "Keep this
  agent connected", which returns a persistent agent token (``paway_...``) once; it works as the bearer token on
  ``/mcp`` and REST with no further prompts until the person revokes it (``POST /v1/token/rotate`` replaces it).

Every request goes through :class:`AuthMiddleware`: Origin check (browsers cannot drive it cross-site), per-client
rate limit, lockout after repeated bad credentials, constant-time token checks, and an access log line.
Needs ``starlette``/``uvicorn``/``mcp`` (the ``[agent]`` extra).
"""

from __future__ import annotations

import base64
import contextlib
import functools
import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.agentlink import perms
from away.agentlink.auth import (
    DEFAULT_TOKEN_EXPIRY,
    PERSIST_PREFIX,
    ROTATE_GRACE_S,
    Agent,
    AgentRegistry,
    AgentToken,
    Pairing,
    RefreshError,
    clean_name,
    expiry_choice,
    same,
)
from away.agentlink.control import ControlRefused
from away.agentlink.groups import GROUPS
from away.agentlink.groups import describe as describe_groups
from away.agentlink.groups import resolve as resolve_group
from away.agentlink.jobs import JobLookupError
from away.agentlink.ratelimit import FailureLockout, SourceBackoff, TokenBucket
from away.agentlink.registry import ChangeError
from away.agentlink.safepaths import UnsafePath
from away.agentlink.service import Forbidden, NotAvailable, Service

FRIENDLY_CLIENTS = {
    "claude-ai": "Claude",
    "claude": "Claude",
    "claude-code": "Claude Code",
    "claude desktop": "Claude Desktop",
    "openai-mcp": "ChatGPT",
    "chatgpt": "ChatGPT",
    "cursor": "Cursor",
    "cursor-vscode": "Cursor",
    "gemini-cli-mcp-client": "Gemini CLI",
    "mcp": "AI agent",
}
CODE_PATH = re.compile(r"^/c/([a-z0-9]{6,32})/?(?:mcp/?)?$")
# v0.1.24 hardening: request bodies are capped before anything reads or parses them (Content-Length is checked first,
# then the stream is counted, so a chunked body or a wrong Content-Length cannot get past). Every REST/JSON tool
# argument fits easily in 64 KB (the largest, type_text, is 2000 characters). MCP JSON-RPC gets 256 KB: clients may
# batch several calls and send long capability lists in ``initialize``; it replaces the old 4 MB pairing-read cap.
MAX_JSON_BODY = 64 * 1024
MAX_MCP_BODY = 256 * 1024
TOO_LARGE = "the request is too large (at most {} KB)"


def body_limit(path: str) -> int:
    """The body size cap for a request path: MCP endpoints (``/mcp``, the pairing link ``/c/<code>``) or JSON."""
    return MAX_MCP_BODY if path == "/mcp" or path.startswith(("/mcp/", "/c/")) else MAX_JSON_BODY


class BodyTooLarge(ValueError):
    pass


KEEP_CONNECTED = "Keep this agent connected"  # v0.1.24: the approval choice that issues a persistent agent token
EXPIRY_WORDS = {"never": "until you revoke it", "30d": "for 30 days", "90d": "for 90 days"}
DROP_CHECK_S = 0.2  # how often a live request / stream checks whether its agent was disconnected
# v0.1.25 keepalive. An agent counts as connected while it was seen in the last 90 s, but "seen" only moved when a
# request started, so an agent sitting on a live MCP stream or event stream (open for minutes, as MCP clients do)
# showed as disconnected after ~90 s: the tray, Agent Link and "Hey <agent>" said it was gone. A live connection
# now marks its agent as seen every LIVE_TOUCH_S.
LIVE_TOUCH_S = 20.0
SSE_KEEPALIVE_S = 15.0  # /v1/events/stream sends a comment line at least this often (Cloudflare drops ~100 s idle)
APPROVE_WAIT_S = 90.0  # the Allow / Deny prompt for a new agent; under Cloudflare's ~100 s limit for one request

# tool name -> (result kind, http method for the convenience route, path, summary). "image" results are JPEG.
TOOLS: dict[str, tuple[str, str, str, str]] = {
    "get_status": ("json", "GET", "/v1/status", "State, permissions, health, context, control"),
    "get_context": ("json", "GET", "/v1/context", "The person's live context"),
    "about": ("text", "GET", "/v1/about", "Briefing for agents (markdown)"),
    "poll": ("json", "GET", "/v1/poll", "Events after ?since=, landmarks, health"),
    "health": ("json", "GET", "/v1/health", "fps per camera, latency, CPU/GPU, models"),
    "get_logs": ("json", "GET", "/v1/logs", "Log tail, secrets hidden (?lines=) (share_logs)"),
    "get_config": ("json", "GET", "/v1/config", "Redacted configuration"),
    "list_settings": ("json", "GET", "/v1/settings", "Changeable settings (?domain=)"),
    "change_settings": ("json", "POST", "/v1/settings/change", "Change settings {ops, why} (tune)"),
    "undo_changes": ("json", "POST", "/v1/settings/undo", "Undo {what: last|today|<id>} (tune)"),
    "change_history": ("json", "GET", "/v1/settings/history", "Change history"),
    "get_preferences": ("json", "GET", "/v1/preferences", "Preference notes"),
    "add_preference": ("json", "POST", "/v1/preferences", "Add a preference {text} (tune)"),
    "remove_preference": ("json", "POST", "/v1/preferences/remove", "Remove preference {number} (tune)"),
    "run_diagnose": ("json", "POST", "/v1/diagnose", "Start a check {kind, device, seconds} (diagnose)"),
    "get_job": ("json", "GET", "/v1/jobs", "A check's state (?job_id=) (diagnose)"),
    "record_session": ("json", "POST", "/v1/record", "Record landmarks {seconds, note} (share_logs)"),
    "run_selftest": (
        "json",
        "POST",
        "/v1/selftest",
        "Run the PC self-test {wait, only}; the report goes to every connected agent (diagnose)",
    ),
    "selftest_latest": ("json", "GET", "/v1/selftest/latest", "The newest self-test report (redacted)"),
    "get_bundle": ("json", "POST", "/v1/bundle", "Build a diagnostics bundle (share_logs)"),
    "list_files": ("json", "GET", "/v1/files", "Downloadable bundles/recordings/logs (share_logs)"),
    "read_file": ("bytes", "GET", "/v1/file", "Download ?rel=diagnostics/... (share_logs)"),
    "look": ("image", "GET", "/v1/look", "One camera still (snapshots)"),
    "get_screen": (
        "image",
        "GET",
        "/v1/screen",
        "Screenshot ?scale=0.1-1&monitor=N&region=x,y,w,h&elements= (screen); X-Away-Meta lists the monitors",
    ),
    "screen_watch": ("json", "POST", "/v1/screen/watch", "Live view {on, scale, monitor, region} (screen)"),
    "last_capture": ("image", "GET", "/v1/capture/latest", "The picture the person sent you"),
    "ask_user": ("json", "POST", "/v1/ask", "Ask the person {question, options, timeout_s, urgency}"),
    "notify_user": ("json", "POST", "/v1/notify", "Toast {text (or message), level, speak}"),
    "wait_for_event": ("json", "GET", "/v1/events", "Long-poll ?since=&timeout_s=&topics="),
    "inbox_list": ("json", "GET", "/v1/inbox", "Messages the person spoke to you (?unread_only=)"),
    "inbox_read": ("json", "GET", "/v1/inbox/read", "One message (?id=); marks it read"),
    "inbox_reply": ("json", "POST", "/v1/inbox/reply", "Answer a message {id, text, speak} (shown + read aloud)"),
    "request_control": ("json", "POST", "/v1/control/request", "Ask to use the computer {why, minutes} (control)"),
    "resume_control": ("json", "POST", "/v1/control/resume", "Ask to continue after a takeover (control)"),
    "stop_control": ("json", "POST", "/v1/control/stop", "Stop using the computer"),
    "act": ("json", "POST", "/v1/control/act", "{action: move|click|...|open_url, ...} (control)"),
    "undo_last_action": ("json", "POST", "/v1/control/undo", "Undo the last undoable action (control)"),
    "action_log": ("json", "GET", "/v1/control/log", "Your own computer actions"),
    "disconnect": ("json", "POST", "/v1/disconnect", "Disconnect this agent"),
}


def friendly_name(client_name: Any) -> str:
    raw = str(client_name or "").strip()
    return FRIENDLY_CLIENTS.get(raw.lower(), clean_name(raw))


def resolve_tool(tool: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """A service method name (``change_settings``) as is, or a v0.1.21 group (``settings`` + ``action``) resolved
    to its method; KeyError for neither."""
    if tool in TOOLS:
        return tool, kwargs
    if tool in GROUPS:
        return resolve_group(tool, kwargs)
    raise KeyError(f"unknown tool {tool!r}")


def kind_of(tool: str, kwargs: dict[str, Any]) -> str:
    try:
        return TOOLS[resolve_tool(tool, kwargs)[0]][0]
    except (KeyError, ValueError):
        return "json"


def dispatch(service: Service, agent: Agent, tool: str, kwargs: dict[str, Any]) -> Any:
    tool, kwargs = resolve_tool(tool, kwargs)
    fn = getattr(service, tool)
    clean = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    return fn(agent, **clean)


def error_status(exc: BaseException) -> tuple[int, str]:
    if isinstance(exc, Forbidden):
        return 403, str(exc)
    if isinstance(exc, ControlRefused):
        return 409, str(exc)
    if isinstance(exc, (ChangeError, UnsafePath, ValueError, TypeError)):
        return 400, str(exc)
    if isinstance(exc, NotAvailable):
        return 503, str(exc)
    if isinstance(exc, JobLookupError):
        return (410 if exc.reason == "expired" else 404), str(exc)
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return 404, str(exc).strip("'\"")
    if isinstance(exc, RuntimeError):
        return 409, str(exc)
    if isinstance(exc, AttributeError):  # v0.1.23: a wrongly shaped argument, never "internal error"
        return 400, "an argument had the wrong shape (see the tool's description for the expected form)"
    return 500, f"internal error: {type(exc).__name__}"


class Gate:
    """Authentication, pairing approvals, rate limits and the access log (shared by REST and MCP)."""

    def __init__(
        self,
        service: Service,
        registry: AgentRegistry,
        *,
        local_key: str,
        remote_ttl_s: Callable[[], float] = lambda: 3600.0,
        local_ttl_s: Callable[[], float] = lambda: 30 * 86400.0,
        rate: tuple[float, int] = (10.0, 40),
        access_log: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        tunnel_hosts: Callable[[], Any] = lambda: (),
        link_info: Callable[[], dict[str, Any]] = lambda: {},
        token_expiry: Callable[[], Any] = lambda: DEFAULT_TOKEN_EXPIRY,
    ) -> None:
        self.token_expiry = token_expiry  # v0.1.24: lifetime of a new persistent token unless the agent asks
        self.link_info = link_info  # v0.1.24: the current public addresses (GET /v1/link, refresh replies)
        self.service = service
        self.tunnel_hosts = tunnel_hosts  # v0.1.23: hostnames of the active secure / backup link (Host check)
        self.registry = registry
        self.local_key = local_key
        self.remote_ttl_s = remote_ttl_s
        self.local_ttl_s = local_ttl_s
        self.bucket = TokenBucket(rate[0], rate[1], clock)
        self.lockout = FailureLockout(10, 60.0, 60.0, clock)
        self.token_backoff = SourceBackoff(clock=clock)  # v0.1.24: failed bearer tokens, per source, with backoff
        self.kept: dict[str, tuple[AgentToken, str]] = {}  # v0.1.24: pairing code -> the token issued (taken once)
        self._drop_epoch: dict[str, int] = {}  # v0.1.24: agent id -> times it was disconnected (live requests watch)
        self.mcp_sessions: dict[str, set[str]] = {}  # v0.1.24: agent id -> its MCP session ids (ended on drop)
        self.end_mcp_session: Callable[[str], Any] | None = None  # set by create_app when MCP is served
        registry.on_revoke.append(self.drop_agent)
        self.access_log = access_log
        self.recent: deque[dict[str, Any]] = deque(maxlen=300)
        self._waiting: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.stopped = threading.Event()

    # ---------------------------------------------------------------- logging
    def log(self, ip: str, agent: str, method: str, path: str, status: int, ms: float) -> None:
        path = re.sub(r"^/c/[a-z0-9]+", "/c/***", path)
        item = {
            "t": time.time(),
            "ip": ip,
            "agent": agent,
            "method": method,
            "path": path,
            "status": status,
            "ms": round(ms, 1),
        }
        self.recent.append(item)
        if self.access_log is not None:
            try:
                self.access_log.parent.mkdir(parents=True, exist_ok=True)
                with self.access_log.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} {ip} {agent or '-'} {method} {path} {status} "
                        f"{ms:.0f}ms\n"
                    )
            except OSError:
                pass

    # ----------------------------------------------------------------- pairing
    def pair(
        self,
        p: Pairing,
        name: str,
        client: dict[str, Any] | None,
        bind_link: bool,
        kind: str = "remote",
        offer_keep: bool = False,
        wants_keep: bool = False,
        expiry: Any = None,
    ) -> tuple[Agent, str | None]:
        """Ask the person to approve this agent. Blocks until answered. Raises :class:`Forbidden` on deny.

        v0.1.24: with ``offer_keep`` (a one-time code redeemed by an agent that can store a token) the dialog also
        offers "Keep this agent connected"; choosing it issues a persistent agent token, left in :attr:`kept` under
        the pairing code for the caller to hand out exactly once."""
        with self._lock:
            waiter = self._waiting.get(p.code)
            mine = waiter is None
            if mine:
                waiter = self._waiting[p.code] = threading.Event()
        assert waiter is not None
        if not mine:  # a retry of the same request (connectors retry): wait for the first one's answer
            waiter.wait(130.0)
            if bind_link:
                a = self.registry.authenticate_link(p.code)
                if a is not None:
                    return a, None
            raise Forbidden("this pairing link was already used")
        try:
            if not self.registry.mark_pending(p, name, client):
                raise Forbidden("this pairing link was already used by another agent")
            pin = f"{p.pin[:3]} {p.pin[3:]}"
            options = ["Allow", "Deny"]
            # proposal #18: the prompt says what each permission means, in plain words, not just labels
            details = list(perms.describe(p.scopes, verbose=True))
            if perms.CONTROL in p.scopes:
                minutes = int(getattr(getattr(self.service, "control", None), "ttl_s", 1800.0) // 60)
                details.append(
                    f"Computer use: you approve it each time, up to {minutes} minutes, "
                    "and you can take over by moving the mouse."
                )
            details.append(f"Matching code: {pin}")
            choice = DEFAULT_TOKEN_EXPIRY
            if offer_keep:
                choice = expiry_choice(expiry if expiry not in (None, "") else self.token_expiry())
                options = ["Allow", KEEP_CONNECTED, "Deny"]
                details.append(
                    f"{KEEP_CONNECTED}: it reconnects by itself, even after a restart, {EXPIRY_WORDS[choice]}"
                    + (" (it asked for this)" if wants_keep else "")
                    + ". You can revoke it any time in Agent Link."
                )
            ans = self.service.prompts.ask(
                f"{clean_name(name)} wants to connect" + (" and stay connected" if offer_keep and wants_keep else ""),
                options,
                title="Connect an AI agent?",
                agent=clean_name(name),
                details=details,
                timeout_s=APPROVE_WAIT_S,
                urgency="high",
                kind="approve",
            )
            keep = offer_keep and ans.get("answer") == KEEP_CONNECTED
            if ans.get("answer") != "Allow" and not keep:
                self.registry.deny(p)
                self.service.events.publish("agent.denied", {"agent": clean_name(name)})
                raise Forbidden("the person did not allow this connection")
            ttl = self.local_ttl_s() if kind == "local" else self.remote_ttl_s()
            agent, token = self.registry.approve(p, ttl, bind_link=bind_link)
            if keep:
                self.kept[p.code] = self.registry.issue_persistent(agent.id, choice)
            self.service.events.publish(
                "agent.connected", {"agent": agent.name, "method": ans.get("method"), "stays_connected": keep}
            )
            self.service.prompts.notify(
                f"Connected to {agent.name} \u2713" + (" It stays connected until you revoke it." if keep else ""),
                "success",
            )
            return agent, token
        finally:
            waiter.set()
            with self._lock:
                self._waiting.pop(p.code, None)

    def reconnect_info(self, agent: Agent, refresh_token: str) -> dict[str, Any]:
        """v0.1.24: what a paired agent keeps so it survives restarts: the refresh token, until when it works, the
        current addresses and where to look up a moved one (rendezvous topic + the key its announcements are
        signed with)."""
        from away.agentlink import rendezvous

        info = dict(self.link_info() or {})
        rdv = (
            rendezvous.agent_info(agent, str(info.get("rendezvous_server") or ""))
            if info.get("rendezvous") != "off"
            else {}
        )
        return {
            "refresh_token": refresh_token,
            "refresh_expires": agent.refresh_expires,
            "token_expires": agent.expires,
            "link": info,
            "rendezvous": rdv,
            "how_to_reconnect": "POST /v1/token/refresh {refresh_token} on the newest address; if the address "
            "stopped answering, read the rendezvous topic for the new one (see scripts/agent_link_client.py)",
        }

    def pair_local(
        self, key: str, name: str, scopes: Any, wants_keep: bool = False, expiry: Any = None
    ) -> tuple[Agent, str, dict[str, Any]]:
        if not key or not same(key, self.local_key):
            raise Forbidden("local pairing needs the local key")
        p = self.registry.new_pairing(perms.normalize(scopes), 300.0, kind="local")
        agent, token = self.pair(
            p,
            name,
            {"name": name, "transport": "stdio"},
            bind_link=False,
            kind="local",
            offer_keep=True,
            wants_keep=wants_keep,
            expiry=expiry,
        )
        assert token is not None
        return agent, token, self.take_kept(p.code)

    # ------------------------------------------------- persistent agent tokens (v0.1.24)
    def take_kept(self, code: str) -> dict[str, Any]:
        """The persistent token issued for this pairing, as reply fields, exactly once ({} when none)."""
        got = self.kept.pop(code, None)
        if got is None:
            return {}
        t, secret = got
        return {
            "persistent_token": secret,
            "persistent_token_id": t.id,
            "persistent_token_expires": t.expires,
            "how_to_stay_connected": "store persistent_token safely (it is shown only this once) and send it as "
            "Authorization: Bearer <persistent_token> on /mcp or REST; it resumes your session after restarts with "
            "no prompt until the person revokes it",
        }

    def drop_agent(self, agent_id: str) -> None:
        """Disconnected or token revoked: every live request / stream of this agent ends within DROP_CHECK_S."""
        with self._lock:
            self._drop_epoch[agent_id] = self._drop_epoch.get(agent_id, 0) + 1

    def drop_epoch(self, agent_id: str) -> int:
        return self._drop_epoch.get(agent_id, 0)

    def note_mcp_session(self, agent_id: str, session_id: str) -> None:
        with self._lock:
            ids = self.mcp_sessions.setdefault(agent_id, set())
            ids.add(session_id)
            if len(ids) > 64:  # an agent that never closes its sessions: keep the newest few
                self.mcp_sessions[agent_id] = set(list(ids)[-32:])

    async def end_sessions(self, agent_id: str) -> None:
        """End this agent's MCP sessions on the server side too (best effort)."""
        with self._lock:
            ids = self.mcp_sessions.pop(agent_id, set())
        if self.end_mcp_session is None:
            return
        for sid in ids:
            with contextlib.suppress(Exception):
                await self.end_mcp_session(sid)


# ------------------------------------------------------------------ browser guard (v0.1.23)
# The local server only listens on 127.0.0.1, but a web page in the person's browser can still try to reach it:
# by DNS rebinding (evil.example resolves to 127.0.0.1, so the browser sends ``Host: evil.example``) or by a
# cross-site POST (``Origin: https://evil.example``, or the opaque ``Origin: null`` of sandboxed iframes and
# ``file:`` pages). Ported from the console guard of github.com/fernandogarzaaa/ghost-chimera-rs (MIT).
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
STATE_CHANGING = ("POST", "PUT", "PATCH", "DELETE")
JSON_BODY = ("POST", "PUT", "PATCH")


def host_without_port(value: str) -> str:
    """The host part of a ``Host`` header (``host``, ``host:port``, ``[v6]:port``, bare ``v6``), lower case."""
    v = str(value or "").strip().lower()
    if v.startswith("["):
        return v[1:].split("]", 1)[0]
    if v.count(":") == 1:
        return v.split(":", 1)[0]
    return v


def hostname_of(url: str | None) -> str:
    """``https://abc.lhr.life/x`` -> ``abc.lhr.life`` ("" when there is none)."""
    if not url:
        return ""
    rest = str(url).strip().split("://", 1)[-1]
    authority = rest.split("/", 1)[0].rsplit("@", 1)[-1]
    return host_without_port(authority)


def host_allowed(value: str, extra: Any = ()) -> bool:
    """A ``Host`` header naming this PC (localhost, 127.0.0.1, [::1], any port) or an active tunnel hostname."""
    h = host_without_port(value)
    if not h:
        return False
    return h in LOOPBACK_HOSTS or h in {str(x).strip().lower() for x in extra if x}


def origin_allowed(origin: str, host: str, extra: Any = ()) -> bool:
    """An ``Origin`` is accepted only when it is this server itself: ``http(s)://<the Host header>``, for an allowed
    host. ``null`` and every other site are refused."""
    o = str(origin or "").strip().lower()
    if not o.startswith(("http://", "https://")):
        return False  # "null", "file://", anything opaque
    authority = o.split("://", 1)[1].split("/", 1)[0]
    if "@" in authority or not host:
        return False
    return authority == str(host).strip().lower() and host_allowed(authority, extra)


def is_json(content_type: str) -> bool:
    ct = str(content_type or "").split(";", 1)[0].strip().lower()
    return ct == "application/json" or (ct.startswith("application/") and ct.endswith("+json"))


def browser_guard(method: str, headers: dict[str, str], extra_hosts: Any = ()) -> str | None:
    """Why a request must be refused ("host", "origin", "content-type"), or None. ``headers``: lower-case names."""
    extra = list(extra_hosts or ())
    host = headers.get("host", "")
    if "host" in headers and not host_allowed(host, extra):
        return "host"
    if method in STATE_CHANGING and "origin" in headers and not origin_allowed(headers["origin"], host, extra):
        return "origin"
    if method in JSON_BODY and not is_json(headers.get("content-type", "")):
        return "content-type"
    return None


def _json(data: Any, status: int = 200) -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse(data, status_code=status)


class AuthMiddleware:
    def __init__(self, app: Any, gate: Gate) -> None:
        self.app = app
        self.gate = gate

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        import anyio

        g = self.gate
        t0 = time.monotonic()
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        ip = headers.get("cf-connecting-ip") or (scope.get("client") or ("?", 0))[0]
        status_box = {"status": 0}
        agent_name = {"name": ""}
        started = {"start": False, "end": False}
        agent_id_box = {"id": ""}
        token_check: tuple[str, str] | None = None  # v0.1.24: (token id, matched hash) for agent-token requests

        async def send_logged(msg: dict[str, Any]) -> None:
            if msg["type"] == "http.response.start":
                status_box["status"] = int(msg["status"])
                started["start"] = True
                if agent_id_box["id"]:  # v0.1.24: remember the agent's MCP sessions (ended when it is dropped)
                    for k, v in msg.get("headers") or []:
                        if k.lower() == b"mcp-session-id":
                            g.note_mcp_session(agent_id_box["id"], v.decode("latin-1"))
            elif msg["type"] == "http.response.body" and not msg.get("more_body"):
                started["end"] = True
            await send(msg)

        async def reply(status: int, message: str) -> None:
            await _json({"error": message}, status)(scope, receive, send_logged)

        try:
            if g.stopped.is_set():
                await reply(503, "Agent Link is off")
                return
            extra: Any = ()
            with contextlib.suppress(Exception):
                extra = tuple(g.tunnel_hosts() or ())
            why = browser_guard(method, headers, extra)
            if why == "host":  # v0.1.23: DNS rebinding
                await reply(403, "unknown Host; Agent Link only answers on this PC or its own secure link")
                return
            if why == "origin":  # v0.1.23: cross-site and Origin: null
                await reply(403, "browser requests are not accepted")
                return
            if why == "content-type":  # v0.1.23: no form / text/plain posts
                await reply(415, "send JSON (Content-Type: application/json)")
                return
            limit = body_limit(path)
            try:
                declared = int(headers.get("content-length", "") or 0)
            except ValueError:
                declared = 0
            if declared > limit:  # v0.1.24: refused before a single byte is read
                await reply(413, TOO_LARGE.format(limit // 1024))
                return
            if method in JSON_BODY:  # read (and count) the whole body now; the app gets it replayed
                try:
                    buffered = await _read_body(receive, limit)
                except BodyTooLarge:
                    await reply(413, TOO_LARGE.format(limit // 1024))
                    return
                receive = _replay_body(buffered, receive)
            auth = headers.get("authorization", "")
            bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            persistent = bearer.startswith(PERSIST_PREFIX)
            # v0.1.24: an agent token is limited per source (with backoff) instead of by the global lockout, so a
            # stranger guessing tokens cannot lock the person's own agents out
            if g.lockout.locked() and not persistent:
                await reply(429, "too many failed attempts; wait a minute")
                return
            if not g.bucket.allow(ip):
                await reply(429, "too many requests; slow down")
                return
            public = (method == "GET" and path in ("/", "/openapi.json", "/favicon.ico", "/v1/ping")) or (
                method == "POST" and path in ("/v1/pair", "/v1/pair/local", "/v1/token/refresh")
            )
            m = CODE_PATH.match(path)
            agent: Agent | None = None
            if m:
                wants_mcp = method in ("POST", "DELETE") or "text/event-stream" in headers.get("accept", "")
                if not wants_mcp:
                    public = True  # the landing page; it reveals nothing about the code's state
                else:
                    code = m.group(1)
                    agent = g.registry.authenticate_link(code)
                    body: bytes | None = None
                    if agent is None:
                        p = g.registry.find_pairing(code=code)
                        if p is None:
                            g.lockout.fail()
                            await reply(404, "this link has expired or was already used; make a new one in Agent Link")
                            return
                        body = await _read_body(receive, MAX_MCP_BODY)
                        name, client = _client_info(body)
                        if name is None:
                            await reply(401, "connect first (MCP initialize)")
                            return
                        try:
                            agent, _tok = await anyio.to_thread.run_sync(
                                lambda: g.pair(p, friendly_name(name), client, True)
                            )
                        except Forbidden as exc:
                            await reply(403, str(exc))
                            return
                    scope = dict(scope, path="/mcp", raw_path=b"/mcp")
                    if body is not None:
                        receive = _replay_body(body, receive)
            elif not public:
                wait = g.token_backoff.retry_after(str(ip)) if persistent else 0.0
                if wait > 0:
                    await _json(
                        {"error": f"too many wrong tokens from this address; wait {int(wait) + 1} s"},
                        429,
                    )(scope, receive, _with_header(send_logged, "retry-after", str(int(wait) + 1)))
                    return
                if persistent:
                    where = host_without_port(headers.get("host", ""))
                    hit = g.registry.authenticate_persistent(bearer, str(ip), where)
                    if hit is not None:
                        agent, tok, used = hit
                        token_check = (tok.id, used)
                else:
                    agent = g.registry.authenticate(bearer)
                if agent is None:
                    if persistent:
                        g.token_backoff.fail(str(ip))
                    else:
                        g.lockout.fail()  # unchanged since v0.1.20 for access tokens
                    await reply(
                        401,
                        "this agent token was revoked, expired or replaced; pair again in Project Away > Agent Link"
                        if persistent
                        else "missing or invalid token (Authorization: Bearer ...)",
                    )
                    return
                if persistent:
                    g.token_backoff.ok(str(ip))
            if agent is not None:
                agent_name["name"] = agent.name
                agent_id_box["id"] = agent.id
                state = dict(scope.get("state") or {})
                state["agent"] = agent
                if token_check is not None:
                    state["agent_token"] = token_check
                scope = dict(scope, state=state)
                await _run_watched(g, agent.id, token_check, self.app, scope, receive, send_logged, started)
                return
            await self.app(scope, receive, send_logged)
        finally:
            g.log(str(ip), agent_name["name"], method, path, status_box["status"], (time.monotonic() - t0) * 1000.0)


def _with_header(send: Any, name: str, value: str) -> Any:
    async def snd(msg: dict[str, Any]) -> None:
        if msg["type"] == "http.response.start":
            msg = dict(msg, headers=[*msg.get("headers", []), (name.encode("latin-1"), value.encode("latin-1"))])
        await send(msg)

    return snd


async def _run_watched(
    g: Gate,
    agent_id: str,
    token_check: tuple[str, str] | None,
    app: Any,
    scope: dict[str, Any],
    receive: Any,
    send: Any,
    started: dict[str, bool],
) -> None:
    """v0.1.24: run an authenticated request while watching for its agent being disconnected (or, for an agent
    token, the token being revoked / expiring / its rotation grace ending). Then the request is cancelled at once:
    a live MCP stream or ``/v1/events/stream`` ends, a request still waiting gets a 401."""
    import anyio

    epoch = g.drop_epoch(agent_id)
    dropped = {"yes": False}

    def gone() -> bool:
        if g.drop_epoch(agent_id) != epoch:
            return True
        return token_check is not None and not g.registry.persistent_valid(*token_check)

    async with anyio.create_task_group() as tg:

        async def watch() -> None:
            touched = time.monotonic()
            while True:
                await anyio.sleep(DROP_CHECK_S)
                if gone():
                    dropped["yes"] = True
                    tg.cancel_scope.cancel()
                    return
                if time.monotonic() - touched >= LIVE_TOUCH_S:  # v0.1.25: still connected while the stream is open
                    touched = time.monotonic()
                    with contextlib.suppress(Exception):
                        g.registry.touch(agent_id)

        async def serve() -> None:
            await app(scope, receive, send)
            tg.cancel_scope.cancel()

        tg.start_soon(watch)
        tg.start_soon(serve)
    if not dropped["yes"]:
        return
    if g.drop_epoch(agent_id) != epoch:  # the agent was disconnected (not just an old token's hour running out)
        with anyio.CancelScope(shield=True):
            await g.end_sessions(agent_id)
    with contextlib.suppress(Exception):
        if not started["start"]:
            await _json({"error": "this agent was disconnected on the PC"}, 401)(scope, receive, send)
        elif not started["end"]:
            await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _read_body(receive: Any, limit: int = MAX_MCP_BODY) -> bytes:
    """The whole request body; :class:`BodyTooLarge` as soon as it passes ``limit`` (never truncated silently)."""
    chunks: list[bytes] = []
    size = 0
    while True:
        msg = await receive()
        if msg["type"] != "http.request":
            break
        chunk = msg.get("body", b"")
        size += len(chunk)
        if size > limit:
            raise BodyTooLarge(TOO_LARGE.format(limit // 1024))
        chunks.append(chunk)
        if not msg.get("more_body"):
            break
    return b"".join(chunks)


def _replay_body(body: bytes, receive: Any) -> Any:
    sent = {"done": False}

    async def rcv() -> dict[str, Any]:
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return dict(await receive())

    return rcv


def _client_info(body: bytes) -> tuple[str | None, dict[str, Any]]:
    """The MCP client's name from an ``initialize`` request (single or batched JSON-RPC)."""
    try:
        data = json.loads(body.decode("utf-8") or "null")
    except Exception:
        return None, {}
    items = data if isinstance(data, list) else [data]
    for it in items:
        if isinstance(it, dict) and it.get("method") == "initialize":
            info = (it.get("params") or {}).get("clientInfo") or {}
            return str(info.get("name") or "AI agent"), {"name": info.get("name"), "version": info.get("version")}
    return None, {}


# ------------------------------------------------------------------------------------------------- app
def _encode(kind: str, result: Any) -> Any:
    if kind == "image":
        raw, meta = result
        return {"image_b64": base64.b64encode(raw).decode("ascii"), "meta": meta}
    if kind == "bytes":
        return {"b64": base64.b64encode(result).decode("ascii")}
    return result


def openapi() -> dict[str, Any]:
    from away import __version__

    paths: dict[str, Any] = {
        "/v1/pair": {
            "post": {
                "summary": "Exchange a pairing code or 6-digit PIN for a token (the person approves on the PC)",
                "security": [],
                "description": "v0.1.24: add keep_connected: true (and token_expiry: 90d (default) | 30d | never) to "
                "ask to stay connected; if the person chooses Keep this agent connected the reply carries "
                "persistent_token (paway_..., shown only once)",
                "requestBody": _body(
                    {
                        "code": "string",
                        "pin": "string",
                        "name": "string",
                        "keep_connected": "boolean",
                        "token_expiry": "string",
                    }
                ),
            }
        },
        "/v1/rpc/{tool}": {
            "post": {
                "summary": "Call any tool with JSON arguments",
                "description": "Every service method below by name, or a tool group with an `action` (the MCP "
                "tools since v0.1.21): " + "; ".join(describe_groups()),
                "parameters": [
                    {
                        "name": "tool",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "enum": sorted([*TOOLS, *GROUPS])},
                    }
                ],
                "requestBody": _body({}),
            }
        },
        "/v1/token/refresh": {
            "post": {
                "summary": "Renew access with the refresh token from pairing (rotated on every use; v0.1.24)",
                "security": [],
                "requestBody": _body({"refresh_token": "string"}),
            }
        },
        "/v1/token/rotate": {
            "post": {
                "summary": "Replace your persistent agent token (send the current one as the bearer); the old one "
                "keeps working for an hour (v0.1.24)"
            }
        },
        "/v1/link": {"get": {"summary": "The current public addresses and the rendezvous option (v0.1.24)"}},
        "/v1/ping": {"get": {"summary": "Is Project Away answering here? (no sign-in)", "security": []}},
        "/v1/events/stream": {"get": {"summary": "Server-sent events (?since=)"}},
        "/mcp": {"post": {"summary": "MCP Streamable HTTP endpoint"}},
    }
    for tool, (kind, method, path, summary) in TOOLS.items():
        op: dict[str, Any] = {"summary": summary, "operationId": tool}
        if kind == "image":
            op["responses"] = {
                "200": {"description": "JPEG; metadata in the X-Away-Meta header", "content": {"image/jpeg": {}}}
            }
        elif kind == "bytes":
            op["responses"] = {"200": {"description": "the file", "content": {"application/octet-stream": {}}}}
        else:
            op["responses"] = {"200": {"description": "JSON", "content": {"application/json": {}}}}
        if method == "POST":
            op["requestBody"] = _body({})
        paths.setdefault(path, {})[method.lower()] = op
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Project Away Agent Link",
            "version": __version__,
            "description": "The bridge between a person and their AI agents. Same tools as the MCP server; "
            "same permissions. Never shell, never arbitrary files.",
        },
        "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}}},
        "security": [{"bearer": []}],
        "paths": paths,
    }


def _body(props: dict[str, str]) -> dict[str, Any]:
    return {
        "content": {
            "application/json": {"schema": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}}}
        }
    }


LANDING = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Project Away</title><body style="font:16px system-ui;max-width:34em;margin:2em auto;padding:0 1em">
<h2>Project Away</h2><p>This is a private link for connecting an AI agent to someone's computer.</p>
<p><b>To connect:</b> copy this page's address and paste it into your AI app as a custom connector
(ChatGPT: Settings &rarr; Connectors &rarr; Create; Claude: Settings &rarr; Connectors &rarr; Add custom connector).</p>
<p>The person at the computer is asked to <b>Allow</b> or <b>Deny</b> before anything is shared.</p></body>"""


def create_app(service: Service, gate: Gate, mcp: Any = None) -> Any:
    import contextlib

    import anyio
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, Response, StreamingResponse
    from starlette.routing import Route

    def agent_of(request: Request) -> Agent:
        a = request.scope.get("state", {}).get("agent")
        if a is None:
            raise Forbidden("not signed in")
        return a  # type: ignore[no-any-return]

    async def run(request: Request, tool: str, kwargs: dict[str, Any]) -> Any:
        try:
            agent = agent_of(request)
            result = await anyio.to_thread.run_sync(lambda: dispatch(service, agent, tool, kwargs))
        except Exception as exc:
            status, msg = error_status(exc)
            return _json({"error": msg}, status)
        kind = kind_of(tool, kwargs)
        if kind == "image" and request.url.path != f"/v1/rpc/{tool}":
            raw, meta = result
            return Response(raw, media_type="image/jpeg", headers={"X-Away-Meta": json.dumps(meta, default=str)[:4000]})
        if kind == "bytes" and request.url.path != f"/v1/rpc/{tool}":
            return Response(result, media_type="application/octet-stream")
        if kind == "text" and request.url.path != f"/v1/rpc/{tool}":
            return Response(str(result), media_type="text/markdown")
        return _json(_encode(kind, result) if request.url.path.startswith("/v1/rpc/") else result)

    def query_args(request: Request) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in request.query_params.items():
            if k == "topics":
                out[k] = [t for t in v.split(",") if t]
            elif k == "id" or k.endswith("_id"):
                out[k] = v  # v0.1.21: ids stay strings ("82557239" is a check id, not a number)
            elif v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            elif re.fullmatch(r"-?\d+", v):
                out[k] = int(v)
            elif re.fullmatch(r"-?\d+\.\d*", v):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    async def body_args(request: Request) -> dict[str, Any]:
        raw = b""
        async for chunk in request.stream():  # counted as it arrives (the middleware has capped it already)
            raw += chunk
            if len(raw) > MAX_JSON_BODY:
                raise BodyTooLarge(TOO_LARGE.format(MAX_JSON_BODY // 1024))
        if not raw:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("send a JSON object")
        return data

    def make_endpoint(tool: str, method: str) -> Callable[[Request], Any]:
        async def endpoint(request: Request) -> Any:
            try:
                kwargs = query_args(request) if method == "GET" else await body_args(request)
            except BodyTooLarge as exc:
                return _json({"error": str(exc)}, 413)
            except Exception as exc:
                return _json({"error": f"bad request: {exc}"}, 400)
            return await run(request, tool, kwargs)

        return endpoint

    async def rpc(request: Request) -> Any:
        tool = request.path_params["tool"]
        if tool not in TOOLS and tool not in GROUPS:
            return _json({"error": f"unknown tool {tool!r}"}, 404)
        try:
            kwargs = await body_args(request)
        except BodyTooLarge as exc:
            return _json({"error": str(exc)}, 413)
        except Exception as exc:
            return _json({"error": f"bad request: {exc}"}, 400)
        return await run(request, tool, kwargs)

    async def pair(request: Request) -> Any:
        try:
            data = await body_args(request)
        except BodyTooLarge as exc:
            return _json({"error": str(exc)}, 413)
        except Exception:
            return _json({"error": "send JSON"}, 400)
        wants_keep = bool(data.get("keep_connected"))
        try:  # v0.1.24: 30 days / 90 days / never (default: the person's setting, normally 90 days)
            expiry = expiry_choice(data["token_expiry"]) if data.get("token_expiry") not in (None, "") else None
        except ValueError as exc:
            return _json({"error": str(exc)}, 400)
        p = gate.registry.find_pairing(code=data.get("code"), pin=str(data["pin"]) if data.get("pin") else None)
        if p is None:
            gate.lockout.fail()
            return _json({"error": "unknown or expired code"}, 404)
        name = friendly_name(data.get("name") or "AI agent")
        try:
            agent, token = await anyio.to_thread.run_sync(
                lambda: gate.pair(
                    p, name, {"name": data.get("name")}, False, offer_keep=True, wants_keep=wants_keep, expiry=expiry
                )
            )
        except Forbidden as exc:
            return _json({"error": str(exc)}, 403)
        out: dict[str, Any] = {"token": token, "agent": agent.public(), "mcp_path": "/mcp"}
        out.update(gate.take_kept(p.code))  # v0.1.24: the persistent agent token, this one time only
        try:  # v0.1.24: a refresh token so the agent survives restarts without pairing again
            out.update(gate.reconnect_info(agent, gate.registry.new_refresh(agent.id)))
            out["agent"] = agent.public()
        except Exception:
            pass
        return _json(out)

    async def token_refresh(request: Request) -> Any:
        """v0.1.24: a new access token (and a rotated refresh token) for an agent that paired before."""
        try:
            data = await body_args(request)
        except BodyTooLarge as exc:
            return _json({"error": str(exc)}, 413)
        except Exception:
            return _json({"error": "send JSON"}, 400)
        try:
            agent, token, fresh = gate.registry.refresh(
                str(data.get("refresh_token") or ""),
                lambda a: gate.local_ttl_s() if a.kind == "local" else gate.remote_ttl_s(),
            )
        except RefreshError as exc:
            gate.lockout.fail()
            return _json({"error": str(exc)}, 401)
        service.events.publish("agent.reconnected", {"agent": agent.name})
        return _json({"token": token, "agent": agent.public(), **gate.reconnect_info(agent, fresh)})

    async def token_rotate(request: Request) -> Any:
        """v0.1.24: an agent replaces its own persistent token; the old one keeps working for an hour."""
        agent_of(request)
        held = request.scope.get("state", {}).get("agent_token")
        if not held:
            return _json({"error": "send your agent token (paway_...) to replace it"}, 400)
        try:
            t, secret = await anyio.to_thread.run_sync(lambda: service.rotate_token(held[0]))
        except KeyError:
            return _json({"error": "this agent token was revoked; pair again"}, 401)
        return _json(
            {
                "persistent_token": secret,
                "persistent_token_id": t.id,
                "persistent_token_expires": t.expires,
                "old_token_valid_until": t.prev_until,
                "old_token_grace_s": ROTATE_GRACE_S,
            }
        )

    async def link(request: Request) -> Any:
        agent_of(request)
        return _json(dict(gate.link_info() or {}))

    async def ping(request: Request) -> Any:
        from away import __version__

        return _json({"app": "Project Away", "version": __version__, "t": time.time()})

    async def pair_local(request: Request) -> Any:
        try:
            data = await body_args(request)
        except BodyTooLarge as exc:
            return _json({"error": str(exc)}, 413)
        except ValueError as exc:
            return _json({"error": str(exc)}, 400)
        try:
            key = request.headers.get("x-away-local-key", "")
            expiry = expiry_choice(data["token_expiry"]) if data.get("token_expiry") not in (None, "") else None
            agent, token, kept = await anyio.to_thread.run_sync(
                lambda: gate.pair_local(
                    key,
                    friendly_name(data.get("name") or "Local agent"),
                    data.get("scopes"),
                    wants_keep=bool(data.get("keep_connected")),
                    expiry=expiry,
                )
            )
        except Forbidden as exc:
            gate.lockout.fail()
            return _json({"error": str(exc)}, 403)
        except ValueError as exc:
            return _json({"error": str(exc)}, 400)
        return _json({"token": token, "agent": agent.public(), **kept})

    async def stream(request: Request) -> Any:
        agent = agent_of(request)
        since = int(request.query_params.get("since", "0") or 0)
        with contextlib.suppress(ValueError):  # v0.1.25: a reconnecting EventSource resumes where it stopped
            since = max(since, int(request.headers.get("last-event-id", "") or 0))

        async def gen() -> Any:
            seq = since
            while not gate.stopped.is_set() and agent.active(time.time()):
                res = await anyio.to_thread.run_sync(
                    functools.partial(service.wait_for_event, agent, seq, SSE_KEEPALIVE_S), abandon_on_cancel=True
                )
                for ev in res["events"]:
                    seq = max(seq, int(ev["seq"]))
                    yield f"id: {ev['seq']}\nevent: {ev['topic']}\ndata: {json.dumps(ev, default=str)}\n\n"
                if not res["events"]:
                    yield ": keep-alive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def landing(request: Request) -> Any:
        return HTMLResponse(LANDING)

    async def spec(request: Request) -> Any:
        return _json(openapi())

    routes: list[Any] = [
        Route("/", landing),
        Route("/c/{code}", landing),
        Route("/openapi.json", spec),
        Route("/v1/pair", pair, methods=["POST"]),
        Route("/v1/pair/local", pair_local, methods=["POST"]),
        Route("/v1/token/refresh", token_refresh, methods=["POST"]),
        Route("/v1/token/rotate", token_rotate, methods=["POST"]),
        Route("/v1/link", link),
        Route("/v1/ping", ping),
        Route("/v1/rpc/{tool}", rpc, methods=["POST"]),
        Route("/v1/events/stream", stream),
    ]
    for tool, (_kind, method, path, _summary) in TOOLS.items():
        routes.append(Route(path, make_endpoint(tool, method), methods=[method]))

    lifespan = None
    if mcp is not None:
        from mcp.server.fastmcp.server import StreamableHTTPASGIApp

        mcp.streamable_http_app()  # creates the session manager

        async def end_mcp_session(session_id: str) -> None:  # v0.1.24: a revoked agent's live MCP session ends
            transport = getattr(mcp.session_manager, "_server_instances", {}).get(session_id)
            if transport is not None:
                await transport.terminate()

        gate.end_mcp_session = end_mcp_session
        routes.append(Route("/mcp", StreamableHTTPASGIApp(mcp.session_manager), methods=["GET", "POST", "DELETE"]))

        @contextlib.asynccontextmanager
        async def lifespan(_app: Any) -> Any:
            async with mcp.session_manager.run():
                yield

    app = Starlette(routes=routes, lifespan=lifespan)
    return AuthMiddleware(app, gate)


class LocalBackend:
    """MCP over HTTP inside the app: the agent comes from the authenticated request."""

    def __init__(self, service: Service) -> None:
        self.service = service

    def call(self, ctx: Any, method: str, **kw: Any) -> Any:
        agent = None
        try:
            req = ctx.request_context.request
            agent = req.scope.get("state", {}).get("agent")
        except Exception:
            agent = None
        if agent is None:
            raise Forbidden("not signed in")
        return dispatch(self.service, agent, method, kw)
