"""HTTP side of Agent Link: one ASGI app on 127.0.0.1 serving

* ``/mcp`` - MCP over Streamable HTTP, ``Authorization: Bearer <token>``;
* ``/c/<code>`` - the pairing link. Opened in a browser it shows a short help page; used as an MCP connector URL
  (ChatGPT / Claude "add custom connector") the first ``initialize`` asks the person on the PC "<agent> wants to
  connect: Allow / Deny"; once allowed the link keeps working for that one agent until it expires or is revoked;
* ``/v1/...`` - a plain REST/JSON API with the same tools for agents that do not speak MCP (OpenAPI spec at
  ``/openapi.json``), ``POST /v1/rpc/<tool>`` for any tool, ``GET /v1/events/stream`` (server-sent events);
* ``POST /v1/pair`` (pairing code or 6-digit PIN -> token after approval) and ``POST /v1/pair/local`` (the stdio
  bridge on this PC, proven by the local key file only this user can read).

Every request goes through :class:`AuthMiddleware`: Origin check (browsers cannot drive it cross-site), per-client
rate limit, lockout after repeated bad credentials, constant-time token checks, and an access log line.
Needs ``starlette``/``uvicorn``/``mcp`` (the ``[agent]`` extra).
"""

from __future__ import annotations

import base64
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
from away.agentlink.auth import Agent, AgentRegistry, Pairing, clean_name, same
from away.agentlink.control import ControlRefused
from away.agentlink.groups import GROUPS
from away.agentlink.groups import describe as describe_groups
from away.agentlink.groups import resolve as resolve_group
from away.agentlink.jobs import JobLookupError
from away.agentlink.ratelimit import FailureLockout, TokenBucket
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

# tool name -> (result kind, http method for the convenience route, path, summary). "image" results are JPEG.
TOOLS: dict[str, tuple[str, str, str, str]] = {
    "get_status": ("json", "GET", "/v1/status", "State, permissions, health, context, control"),
    "get_context": ("json", "GET", "/v1/context", "The person's live context"),
    "about": ("text", "GET", "/v1/about", "Briefing for agents (markdown)"),
    "poll": ("json", "GET", "/v1/poll", "Events after ?since=, landmarks, health"),
    "health": ("json", "GET", "/v1/health", "fps per camera, latency, CPU/GPU, models"),
    "get_logs": ("json", "GET", "/v1/logs", "Log tail (?lines=)"),
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
    "record_session": ("json", "POST", "/v1/record", "Record landmarks {seconds, note} (diagnose)"),
    "get_bundle": ("json", "POST", "/v1/bundle", "Build a diagnostics bundle (diagnose)"),
    "list_files": ("json", "GET", "/v1/files", "Downloadable bundles/recordings/logs (diagnose)"),
    "read_file": ("bytes", "GET", "/v1/file", "Download ?rel=diagnostics/... (diagnose)"),
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
    "action_log": ("json", "GET", "/v1/control/log", "Computer actions log"),
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
    ) -> None:
        self.service = service
        self.registry = registry
        self.local_key = local_key
        self.remote_ttl_s = remote_ttl_s
        self.local_ttl_s = local_ttl_s
        self.bucket = TokenBucket(rate[0], rate[1], clock)
        self.lockout = FailureLockout(10, 60.0, 60.0, clock)
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
        self, p: Pairing, name: str, client: dict[str, Any] | None, bind_link: bool, kind: str = "remote"
    ) -> tuple[Agent, str | None]:
        """Ask the person to approve this agent. Blocks until answered. Raises :class:`Forbidden` on deny."""
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
            ans = self.service.prompts.ask(
                f"{clean_name(name)} wants to connect",
                ["Allow", "Deny"],
                title="Connect an AI agent?",
                agent=clean_name(name),
                details=[*(f"Can: {d}" for d in perms.describe(p.scopes)), f"Matching code: {pin}"],
                timeout_s=120.0,
                urgency="high",
                kind="approve",
            )
            if ans.get("answer") != "Allow":
                self.registry.deny(p)
                self.service.events.publish("agent.denied", {"agent": clean_name(name)})
                raise Forbidden("the person did not allow this connection")
            ttl = self.local_ttl_s() if kind == "local" else self.remote_ttl_s()
            agent, token = self.registry.approve(p, ttl, bind_link=bind_link)
            self.service.events.publish("agent.connected", {"agent": agent.name, "method": ans.get("method")})
            self.service.prompts.notify(f"Connected to {agent.name} \u2713", "success")
            return agent, token
        finally:
            waiter.set()
            with self._lock:
                self._waiting.pop(p.code, None)

    def pair_local(self, key: str, name: str, scopes: Any) -> tuple[Agent, str]:
        if not key or not same(key, self.local_key):
            raise Forbidden("local pairing needs the local key")
        p = self.registry.new_pairing(perms.normalize(scopes), 300.0, kind="local")
        agent, token = self.pair(p, name, {"name": name, "transport": "stdio"}, bind_link=False, kind="local")
        assert token is not None
        return agent, token


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

        async def send_logged(msg: dict[str, Any]) -> None:
            if msg["type"] == "http.response.start":
                status_box["status"] = int(msg["status"])
            await send(msg)

        async def reply(status: int, message: str) -> None:
            await _json({"error": message}, status)(scope, receive, send_logged)

        try:
            if g.stopped.is_set():
                await reply(503, "Agent Link is off")
                return
            if method != "GET" and headers.get("origin") and not headers.get("origin", "").startswith("null"):
                await reply(403, "browser requests are not accepted")
                return
            if g.lockout.locked():
                await reply(429, "too many failed attempts; wait a minute")
                return
            if not g.bucket.allow(ip):
                await reply(429, "too many requests; slow down")
                return
            public = (method == "GET" and path in ("/", "/openapi.json", "/favicon.ico")) or (
                method == "POST" and path in ("/v1/pair", "/v1/pair/local")
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
                        body = await _read_body(receive)
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
                auth = headers.get("authorization", "")
                token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
                agent = g.registry.authenticate(token)
                if agent is None:
                    g.lockout.fail()
                    await reply(401, "missing or invalid token (Authorization: Bearer ...)")
                    return
            if agent is not None:
                agent_name["name"] = agent.name
                state = dict(scope.get("state") or {})
                state["agent"] = agent
                scope = dict(scope, state=state)
            await self.app(scope, receive, send_logged)
        finally:
            g.log(str(ip), agent_name["name"], method, path, status_box["status"], (time.monotonic() - t0) * 1000.0)


async def _read_body(receive: Any, limit: int = 4_000_000) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        msg = await receive()
        if msg["type"] != "http.request":
            break
        chunk = msg.get("body", b"")
        size += len(chunk)
        if size > limit:
            break
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
                "requestBody": _body({"code": "string", "pin": "string", "name": "string"}),
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
        raw = await request.body()
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
        except Exception as exc:
            return _json({"error": f"bad request: {exc}"}, 400)
        return await run(request, tool, kwargs)

    async def pair(request: Request) -> Any:
        try:
            data = await body_args(request)
        except Exception:
            return _json({"error": "send JSON"}, 400)
        p = gate.registry.find_pairing(code=data.get("code"), pin=str(data["pin"]) if data.get("pin") else None)
        if p is None:
            gate.lockout.fail()
            return _json({"error": "unknown or expired code"}, 404)
        try:
            agent, token = await anyio.to_thread.run_sync(
                lambda: gate.pair(p, friendly_name(data.get("name") or "AI agent"), {"name": data.get("name")}, False)
            )
        except Forbidden as exc:
            return _json({"error": str(exc)}, 403)
        return _json({"token": token, "agent": agent.public(), "mcp_path": "/mcp"})

    async def pair_local(request: Request) -> Any:
        try:
            data = await body_args(request)
            key = request.headers.get("x-away-local-key", "")
            agent, token = await anyio.to_thread.run_sync(
                lambda: gate.pair_local(key, friendly_name(data.get("name") or "Local agent"), data.get("scopes"))
            )
        except Forbidden as exc:
            gate.lockout.fail()
            return _json({"error": str(exc)}, 403)
        except ValueError as exc:
            return _json({"error": str(exc)}, 400)
        return _json({"token": token, "agent": agent.public()})

    async def stream(request: Request) -> Any:
        agent = agent_of(request)
        since = int(request.query_params.get("since", "0") or 0)

        async def gen() -> Any:
            seq = since
            while not gate.stopped.is_set() and agent.active(time.time()):
                res = await anyio.to_thread.run_sync(functools.partial(service.wait_for_event, agent, seq, 15.0))
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
        Route("/v1/rpc/{tool}", rpc, methods=["POST"]),
        Route("/v1/events/stream", stream),
    ]
    for tool, (_kind, method, path, _summary) in TOOLS.items():
        routes.append(Route(path, make_endpoint(tool, method), methods=[method]))

    lifespan = None
    if mcp is not None:
        from mcp.server.fastmcp.server import StreamableHTTPASGIApp

        mcp.streamable_http_app()  # creates the session manager
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
