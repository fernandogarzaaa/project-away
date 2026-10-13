"""``away mcp``: the MCP server for agents on this PC (Claude Desktop, Cursor, Claude Code, Gemini CLI...), over
stdio. The agent app starts it; it forwards every tool call to the running Project Away app's local API
(``127.0.0.1``) with this agent's own token.

The first time an agent uses it, the person sees "<agent> wants to connect: Allow / Deny" on the PC (the agent's
name comes from the MCP client info; the permissions from ``--scopes``, which the Agent Link wizard writes into
the agent's config). The token is then kept in ``<data>/agentlink/local-agents/<name>.json`` (owner-only) until
it expires or is revoked. Proof that the request comes from this user on this PC: the ``local.key`` file only
this user can read.

Nothing is ever printed to stdout except MCP messages (stdout is the protocol channel).
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from away.agentlink import perms
from away.agentlink.host import base_dir, host_alive, send_request, write_private

NOT_RUNNING = "Project Away isn't running. Start it (Start menu or the tray icon), then try again."
LONG_TOOLS = {
    "ask_user": 330,
    "record_session": 140,
    "request_control": 90,
    "resume_control": 90,
    "change_settings": 120,
    "wait_for_event": 130,
    "act": 90,
    "get_bundle": 120,
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "agent"


class RestBackend:
    def __init__(self, scopes: Any = None, name: str | None = None, data_dir: Path | None = None) -> None:
        self.scopes = perms.to_list(perms.normalize(scopes))
        self.name_override = name
        self.data_dir = data_dir
        self.base = base_dir(data_dir)
        self.token: str | None = None
        self.name: str | None = None
        # local only: never through a proxy
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ------------------------------------------------------------------ host
    def _port(self, wait_s: float = 12.0) -> int:
        info = host_alive(self.data_dir)
        if info is None:
            raise RuntimeError(NOT_RUNNING)
        if not info.get("port"):
            send_request("serve", self.data_dir)  # the app starts its local server on request
            end = time.monotonic() + wait_s
            while time.monotonic() < end:
                time.sleep(0.3)
                info = host_alive(self.data_dir) or {}
                if info.get("port"):
                    break
            else:
                raise RuntimeError("Project Away did not open its local connection; restart it and try again.")
        return int(info["port"])

    def _token_file(self) -> Path:
        return self.base / "local-agents" / f"{_slug(self.name or 'agent')}.json"

    def _client_name(self, ctx: Any) -> str:
        if self.name_override:
            return self.name_override
        try:
            info = ctx.session.client_params.clientInfo
            from away.agentlink.api import friendly_name

            return friendly_name(info.name)
        except Exception:
            return "Local AI agent"

    def _http(
        self,
        method: str,
        path: str,
        body: Any = None,
        token: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        port = self._port()
        h = {"Content-Type": "application/json", **(headers or {})}
        if token:
            h["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers=h,
            method=method,
        )
        try:
            with self.opener.open(req, timeout=timeout) as resp:
                return int(resp.status), json.loads(resp.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            try:
                data = json.loads(exc.read().decode("utf-8") or "null")
            except Exception:
                data = {"error": str(exc)}
            return int(exc.code), data
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise RuntimeError(f"{NOT_RUNNING} ({exc})") from None

    def _ensure_token(self, ctx: Any) -> str:
        if self.token:
            return self.token
        self.name = self._client_name(ctx)
        f = self._token_file()
        try:
            saved = json.loads(f.read_text(encoding="utf-8"))
            if saved.get("token") and float(saved.get("expires", 0)) > time.time():
                self.token = str(saved["token"])
                return self.token
        except Exception:
            pass
        key = (
            (self.base / "local.key").read_text(encoding="utf-8").strip() if (self.base / "local.key").exists() else ""
        )
        status, data = self._http(
            "POST",
            "/v1/pair/local",
            {"name": self.name, "scopes": self.scopes},
            timeout=150.0,
            headers={"X-Away-Local-Key": key},
        )
        if status != 200:
            raise PermissionError(str((data or {}).get("error") or f"connection refused ({status})"))
        self.token = str(data["token"])
        write_private(
            f, json.dumps({"token": self.token, "name": self.name, "expires": float(data["agent"]["expires"])})
        )
        return self.token

    def call(self, ctx: Any, method: str, **kw: Any) -> Any:
        for attempt in (1, 2):
            token = self._ensure_token(ctx)
            status, data = self._http(
                "POST", f"/v1/rpc/{method}", kw, token=token, timeout=float(LONG_TOOLS.get(method, 60))
            )
            if status == 401 and attempt == 1:  # revoked or expired: pair again once
                self.token = None
                self._token_file().unlink(missing_ok=True)
                continue
            if status != 200:
                raise RuntimeError(str((data or {}).get("error") or f"error {status}"))
            return data
        raise RuntimeError("could not connect")


def _std_binary(fd: int) -> Any:  # pragma: no cover - exercised on Windows frozen builds only
    """The real stdin/stdout as binary streams, even in a windowed (no console) Windows exe whose sys.stdout was
    replaced by a log file: the pipes the agent app passed are still the process's standard handles."""
    stream = sys.stdin if fd == 0 else sys.__stdout__
    buf = getattr(stream, "buffer", None)
    if buf is not None and (fd == 0 or not str(getattr(stream, "name", "")).endswith(".log")):
        return buf
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        import os

        h = ctypes.windll.kernel32.GetStdHandle(-10 if fd == 0 else -11)
        osfd = msvcrt.open_osfhandle(h, os.O_RDONLY if fd == 0 else os.O_WRONLY)
        return os.fdopen(osfd, "rb" if fd == 0 else "wb", buffering=0)
    return open(fd, "rb" if fd == 0 else "wb", closefd=False)


def serve_stdio(scopes: Any = None, name: str | None = None) -> int:
    import anyio
    from mcp.server.stdio import stdio_server

    from away.agentlink import mcp_tools

    real_in, real_out = _std_binary(0), _std_binary(1)
    sys.stdout = sys.stderr  # anything that prints must not corrupt the protocol channel
    backend = RestBackend(scopes, name)
    mcp = mcp_tools.build(backend)

    async def run() -> None:
        stdin = anyio.wrap_file(io.TextIOWrapper(real_in, encoding="utf-8"))
        stdout = anyio.wrap_file(io.TextIOWrapper(real_out, encoding="utf-8", write_through=True))
        async with stdio_server(stdin=stdin, stdout=stdout) as (r, w):
            server = mcp._mcp_server
            await server.run(r, w, server.create_initialization_options())

    anyio.run(run)
    return 0
