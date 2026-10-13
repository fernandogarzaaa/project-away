"""Fakes for the live agent tests: an HTTP transport, a fake opencode server program, a fake release feed."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.liveagent.http import Response


class FakeTransport:
    """Routes requests to ``handler(method, url, headers, body) -> (status, headers, body-bytes | list[bytes])``."""

    def __init__(self, handler: Callable[..., Any]) -> None:
        self.handler = handler
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> Response:
        payload = json.loads(body) if body else None
        self.requests.append({"method": method, "url": url, "headers": dict(headers), "json": payload})
        status, hdrs, data = self.handler(method, url, headers, payload)
        lines = data if isinstance(data, list) else [data if isinstance(data, bytes) else json.dumps(data).encode()]
        return Response(status, hdrs or {}, iter(lines))


def sse(*events: Any) -> list[bytes]:
    out = []
    for e in events:
        out.append(f"data: {e if isinstance(e, str) else json.dumps(e)}\n".encode())
        out.append(b"\n")
    return out


def openai_chunks(text: str, reasoning: str = "") -> list[bytes]:
    evs: list[Any] = []
    if reasoning:
        evs.append({"choices": [{"delta": {"reasoning_content": reasoning}}]})
    for i in range(0, len(text), 5):
        evs.append({"choices": [{"delta": {"content": text[i : i + 5]}}]})
    evs.append({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}})
    evs.append("[DONE]")
    return sse(*evs)


# ------------------------------------------------------------------------------------------------ release feed
def make_archive(kind: str, exe_name: str = "opencode", content: bytes = b"#!/bin/sh\necho fake\n") -> bytes:
    buf = io.BytesIO()
    if kind == "zip":
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(exe_name, content)
    else:
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            info = tarfile.TarInfo(exe_name)
            info.size = len(content)
            info.mode = 0o755
            t.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def release_feed(version: str, assets: dict[str, bytes], bad_digest: str = "") -> dict[str, Any]:
    out = []
    for name, data in assets.items():
        digest = "sha256:" + (bad_digest or hashlib.sha256(data).hexdigest())
        out.append(
            {
                "name": name,
                "size": len(data),
                "digest": digest,
                "browser_download_url": f"https://example.invalid/{name}",
            }
        )
    return {"tag_name": f"v{version}", "assets": out}


def fake_downloader(files: dict[str, bytes]) -> Callable[..., str]:
    def dl(url: str, dest: Path, progress: Any = None, total: int = 0, **_kw: Any) -> str:
        data = files[url.rsplit("/", 1)[-1]]
        Path(dest).write_bytes(data)
        if progress is not None:
            progress(len(data) // 2, len(data))
            progress(len(data), len(data))
        return hashlib.sha256(data).hexdigest()

    return dl


# ------------------------------------------------------------------------------------------------ fake opencode
FAKE_OPENCODE = r"""
import base64, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
assert args[0] == "serve", args
port = int(args[args.index("--port") + 1])
host = args[args.index("--hostname") + 1]
pw = os.environ["OPENCODE_SERVER_PASSWORD"]
want = "Basic " + base64.b64encode(("opencode:" + pw).encode()).decode()
state = {"clients": [], "perms": [], "rejected": [], "sessions": set(), "lock": threading.Lock()}
log = open(os.path.join(os.environ["XDG_STATE_HOME"], "fake.log"), "a")
log.write(json.dumps({"env_has_secret": any("API_KEY" in k for k in os.environ), "cwd": os.getcwd()}) + "\n")
log.flush()

def emit(ev):
    data = ("data: " + json.dumps(ev) + "\n\n").encode()
    for w in list(state["clients"]):
        try:
            w.write(data); w.flush()
        except Exception:
            state["clients"].remove(w)

def run_turn(sid, text, model):
    time.sleep(0.05)
    if model == "big-pickle-limited":
        emit({"type": "session.status", "properties": {"sessionID": sid, "status": {"type": "retry", "attempt": 1,
              "message": "Free usage exceeded, subscribe to Go", "action": {"reason": "free_tier_limit"}}}})
        return
    state["perms"].append({"id": "per_1", "sessionID": sid, "permission": "bash"})
    emit({"type": "permission.asked", "properties": {"sessionID": sid, "id": "per_1"}})
    for _ in range(100):
        if "per_1" in state["rejected"]:
            break
        time.sleep(0.02)
    emit({"type": "message.part.updated", "properties": {"part": {"id": "prt_r", "type": "reasoning", "sessionID": sid}}})
    emit({"type": "message.part.delta", "properties": {"sessionID": sid, "partID": "prt_r", "field": "text", "delta": "thinking..."}})
    emit({"type": "message.part.delta", "properties": {"sessionID": sid, "partID": "prt_t", "field": "text", "delta": "Hello "}})
    emit({"type": "message.part.updated", "properties": {"part": {"id": "prt_t", "type": "text", "sessionID": sid}}})
    reply = "ready." if "ready" in text else "there. All good."
    emit({"type": "message.part.delta", "properties": {"sessionID": sid, "partID": "prt_t", "field": "text", "delta": reply}})
    emit({"type": "session.idle", "properties": {"sessionID": sid}})

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _auth(self):
        if self.headers.get("Authorization") != want:
            self.send_response(401); self.end_headers(); return False
        return True
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"null") if n else None
    def do_GET(self):
        if not self._auth(): return
        if self.path == "/event":
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            self.wfile.write(b'data: {"type": "server.connected"}\n\n'); self.wfile.flush()
            state["clients"].append(self.wfile)
            while self.wfile in state["clients"]:
                time.sleep(0.05)
            return
        if self.path == "/permission":
            return self._json([p for p in state["perms"] if p["id"] not in state["rejected"]])
        if self.path == "/question":
            return self._json([])
        if self.path == "/config/providers":
            return self._json({"providers": [{"id": "opencode", "models": {
                "big-pickle": {"name": "Big Pickle", "cost": {"input": 0, "output": 0}},
                "nemotron-3.5-lightning-free": {"name": "Nemotron 3.5 Lightning", "cost": {"input": 0, "output": 0}},
                "jev-1.13-free": {"cost": {"input": 0, "output": 0}},
                "claude-opus-5": {"cost": {"input": 5, "output": 25}}}}]})
        if self.path == "/rejected":
            return self._json(state["rejected"])
        self._json({"error": "nope"}, 404)
    def do_POST(self):
        if not self._auth(): return
        body = self._body() or {}
        if self.path == "/session":
            sid = "ses_%d" % (len(state["sessions"]) + 1)
            state["sessions"].add(sid)
            return self._json({"id": sid, "permission": body.get("permission")})
        if self.path.endswith("/prompt_async"):
            sid = self.path.split("/")[2]
            text = body["parts"][0]["text"]
            model = body["model"]["modelID"]
            threading.Thread(target=run_turn, args=(sid, text, model), daemon=True).start()
            self.send_response(204); self.end_headers(); return
        if self.path.endswith("/abort"):
            return self._json(True)
        if self.path.startswith("/permission/") and self.path.endswith("/reply"):
            state["rejected"].append(self.path.split("/")[2])
            return self._json(body.get("reply") == "reject")
        self._json({"error": "nope"}, 404)
    def do_DELETE(self):
        if not self._auth(): return
        self._json(True)

class Server(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind asks socket.getfqdn(), which can stall for many seconds on macOS runners
        import socketserver
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = host, self.server_address[1]

Server((host, port), H).serve_forever()
"""
