#!/usr/bin/env python3
"""A tiny, dependency-free Agent Link client: an example for agents that do not speak MCP (or for poking at it
from a terminal or a cloud sandbox).

    # once: exchange the pairing code (from Settings > Agent Link, or `away agent pair`) for a token. The person
    # at the PC is asked to Allow it.
    python scripts/agent_link_client.py --url https://<name>.trycloudflare.com pair --pin 123456 --name "My agent"

    # then (everything is kept in ~/.away-agent.json: address, token, refresh token, rendezvous)
    python scripts/agent_link_client.py status
    python scripts/agent_link_client.py stream                 # live events (server-sent events)
    python scripts/agent_link_client.py context
    python scripts/agent_link_client.py ask "Ready to start?" Yes No
    python scripts/agent_link_client.py notify "Done!"
    python scripts/agent_link_client.py look out.jpg           # needs "See my camera when I allow"
    python scripts/agent_link_client.py screen out.jpg         # needs "See my screen when I allow"
    python scripts/agent_link_client.py set hands.cursor.gain 1.8 --why "you said it is too fast"
    python scripts/agent_link_client.py bundle out.zip
    python scripts/agent_link_client.py selftest               # run the PC self-test (needs "Run checks")
    python scripts/agent_link_client.py selftest-latest        # the newest self-test report
    python scripts/agent_link_client.py reconnect              # find the current address + renew access now
    python scripts/agent_link_client.py call <tool> '{"json": "args"}'   # any tool (see /openapi.json)

v0.1.24 "a link that survives restarts": after Away restarts or the PC reboots the free secure link gets a new
address and the short access token may have expired. This client then:

1. renews access with the **refresh token** it got at pairing (``POST /v1/token/refresh``; a new refresh token
   comes back every time and is saved);
2. when the old address stops answering, reads the **rendezvous** topic it got at pairing (a signed message on
   ntfy.sh) for the new address, checks the signature with its own key, and continues there.

No re-pairing unless the person disconnected the agent in Settings or it was away longer than the refresh lifetime
(30 days by default). ``--url`` is only needed for ``pair`` (or to override the saved address).

MCP clients do not need this: point them at ``URL/mcp`` with ``Authorization: Bearer <token>``, or at the pairing
link itself (``URL/c/<code>``), which works as a connector URL with no header at all.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

STATE_FILE = Path(os.environ.get("AWAY_AGENT_STATE") or (Path.home() / ".away-agent.json"))
TOKEN_FILE = Path.home() / ".away-agent-token"  # v0.1.20-v0.1.23 clients kept only the token here
MOVED_STATUSES = (404, 502, 503, 521, 523, 530)  # Cloudflare / relays answer these when the old link is gone


class LinkError(SystemExit):
    pass


def verify_announcement(msg: Any, rdv_id: str, key: str, now: float | None = None) -> str | None:
    """The address in a signed rendezvous message (same rule as away.agentlink.rendezvous.verify)."""
    if not isinstance(msg, dict) or msg.get("kind") != "away.link":
        return None
    url, t, sig = str(msg.get("url") or ""), msg.get("t"), str(msg.get("sig") or "")
    if not url.startswith("https://") or not isinstance(t, int) or isinstance(t, bool) or not sig:
        return None
    now = time.time() if now is None else now
    if now - t > 14 * 86400 or t - now > 600:
        return None
    want = hmac.new(key.encode("utf-8"), f"v1|{rdv_id}|{url}|{t}".encode(), hashlib.sha256).hexdigest()
    return url if hmac.compare_digest(sig, want) else None


def newest_address(ndjson: str, rdv_id: str, key: str, now: float | None = None) -> tuple[str, int] | None:
    """The newest correctly signed address in an ntfy poll answer (one JSON event per line)."""
    best: tuple[str, int] | None = None
    for line in ndjson.splitlines():
        try:
            ev = json.loads(line)
            msg = json.loads(ev.get("message") or "null") if ev.get("event", "message") == "message" else None
        except (ValueError, AttributeError):
            continue
        url = verify_announcement(msg, rdv_id, key, now)
        if url and (best is None or int(msg["t"]) > best[1]):
            best = (url, int(msg["t"]))
    return best


class State:
    """What the client remembers between runs (owner-only file)."""

    def __init__(self, path: Path = STATE_FILE) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                self.data = {}
        elif TOKEN_FILE.exists() and path == STATE_FILE:
            self.data = {"token": TOKEN_FILE.read_text().strip()}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, **kw: Any) -> None:
        self.data.update({k: v for k, v in kw.items() if v is not None})
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        if os.name != "nt":
            tmp.chmod(0o600)
        os.replace(tmp, self.path)


Opener = Callable[[urllib.request.Request, float], Any]


def _default_open(req: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(req, timeout=timeout)


class Client:
    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        timeout: float = 330.0,
        state: State | None = None,
        opener: Opener = _default_open,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.state = state if state is not None else State()
        base = url or self.state.get("url") or ""
        self.url = self._base(base)
        self.token = token or self.state.get("token")
        self.timeout = timeout
        self.open = opener
        self.clock = clock
        self.notes: list[str] = []  # what happened while reconnecting (printed by the CLI)

    @staticmethod
    def _base(url: str) -> str:
        u = str(url or "").rstrip("/")
        return u.split("/c/")[0] if "/c/" in u else u

    # ------------------------------------------------------------------ raw HTTP
    def _send(self, method: str, path: str, body: Any = None, auth: bool = True) -> tuple[int, bytes, dict[str, str]]:
        if not self.url:
            raise LinkError("no address yet: run `pair --url <link>` first")
        headers = {"Content-Type": "application/json", "User-Agent": "away-agent-client"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with self.open(req, self.timeout) as resp:
                return int(resp.status), resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read() or b"", dict(exc.headers or {})

    @staticmethod
    def _error(status: int, payload: bytes) -> str:
        try:
            return str(json.loads(payload.decode("utf-8")).get("error"))
        except Exception:
            return payload.decode("utf-8", "replace")[:200] or f"HTTP {status}"

    # -------------------------------------------------------------- reconnecting
    def find_address(self) -> str | None:
        """The current address from the rendezvous topic (signed by Project Away), or None."""
        rdv = self.state.get("rendezvous") or {}
        poll, rid, key = rdv.get("poll"), rdv.get("id"), rdv.get("key")
        if not (poll and rid and key):
            return None
        req = urllib.request.Request(str(poll), headers={"User-Agent": "away-agent-client"})
        try:
            with self.open(req, 15.0) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as exc:
            self.notes.append(f"rendezvous unreachable: {exc}")
            return None
        got = newest_address(text, str(rid), str(key), self.clock())
        if got is None:
            return None
        url, t = got
        if t < int(self.state.get("address_t") or 0):
            return None  # never go back to an older address
        if url != self.url:
            self.notes.append(f"Project Away moved to {url}")
        self.url = url
        self.state.update(url=url, address_t=t)
        return url

    def refresh(self) -> bool:
        """Renew access with the refresh token (rotated: the new one is saved). False when there is none or the
        PC refused it (then the person has to pair this agent again)."""
        rt = self.state.get("refresh_token")
        if not rt:
            return False
        try:
            status, payload, _h = self._send("POST", "/v1/token/refresh", {"refresh_token": rt}, auth=False)
        except (urllib.error.URLError, OSError):
            if not self.find_address():
                return False
            status, payload, _h = self._send("POST", "/v1/token/refresh", {"refresh_token": rt}, auth=False)
        if status in MOVED_STATUSES and self.find_address():
            status, payload, _h = self._send("POST", "/v1/token/refresh", {"refresh_token": rt}, auth=False)
        if status != 200:
            self.notes.append(f"refresh refused: {self._error(status, payload)}")
            return False
        self._keep(json.loads(payload.decode("utf-8")))
        self.notes.append("access renewed")
        return True

    def _keep(self, res: dict[str, Any]) -> None:
        """Save a pairing / refresh answer."""
        self.token = res.get("token") or self.token
        link = res.get("link") or {}
        if link.get("url"):
            self.url = self._base(str(link["url"]))
        self.state.update(
            url=self.url,
            token=self.token,
            refresh_token=res.get("refresh_token"),
            refresh_expires=res.get("refresh_expires"),
            token_expires=res.get("token_expires"),
            rendezvous=res.get("rendezvous") or None,
            agent=(res.get("agent") or {}).get("name"),
        )

    # ------------------------------------------------------------------ requests
    def request(self, method: str, path: str, body: Any = None, raw: bool = False, _retry: bool = True) -> Any:
        try:
            status, payload, headers = self._send(method, path, body)
        except (urllib.error.URLError, OSError) as exc:
            if _retry and self.find_address():
                return self.request(method, path, body, raw, _retry=False)
            raise LinkError(f"cannot reach Project Away ({exc}); is the PC on and the app running?") from None
        if status == 401 and _retry and self.refresh():
            return self.request(method, path, body, raw, _retry=False)
        if status in MOVED_STATUSES and _retry and self.find_address():
            return self.request(method, path, body, raw, _retry=False)
        if status >= 400:
            raise LinkError(f"error {status}: {self._error(status, payload)}")
        return (payload, headers) if raw else json.loads(payload.decode("utf-8") or "null")

    def call(self, tool: str, **kwargs: Any) -> Any:
        return self.request("POST", f"/v1/rpc/{tool}", kwargs)

    def pair(self, pin: str | None, code: str | None, name: str) -> dict[str, Any]:
        status, payload, _h = self._send("POST", "/v1/pair", {"pin": pin, "code": code, "name": name}, auth=False)
        if status != 200:
            raise LinkError(f"error {status}: {self._error(status, payload)}")
        res = dict(json.loads(payload.decode("utf-8")))
        self._keep(res)
        return res

    def stream(self, since: int = 0) -> None:
        req = urllib.request.Request(
            f"{self.url}/v1/events/stream?since={since}",
            headers={"Authorization": f"Bearer {self.token}", "Accept": "text/event-stream"},
        )
        with self.open(req, 3600) as resp:
            for line in resp:
                text = line.decode("utf-8").rstrip()
                if text.startswith("data: "):
                    ev = json.loads(text[6:])
                    print(f"#{ev['seq']} {ev['topic']}: {json.dumps(ev['data'])}", flush=True)


def save_image(result: dict[str, Any], out: str) -> None:
    Path(out).write_bytes(base64.b64decode(result["image_b64"]))
    print(f"saved {out} {json.dumps(result.get('meta'))}")


COMMANDS = [
    "pair",
    "status",
    "context",
    "stream",
    "events",
    "ask",
    "notify",
    "look",
    "screen",
    "set",
    "undo",
    "bundle",
    "logs",
    "selftest",
    "selftest-latest",
    "reconnect",
    "call",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None, help="the Agent Link address (the pairing link works too)")
    p.add_argument("--token", default=os.environ.get("AWAY_TOKEN"))
    p.add_argument("cmd", choices=COMMANDS)
    p.add_argument("args", nargs="*")
    p.add_argument("--pin")
    p.add_argument("--code")
    p.add_argument("--name", default="Agent Link example client")
    p.add_argument("--why", default="")
    a = p.parse_args(argv)
    c = Client(a.url, a.token)
    if a.cmd == "pair":
        if not a.url:
            raise SystemExit("pair needs --url (the pairing link)")
        code = a.code or (a.url.split("/c/")[1].split("/")[0] if "/c/" in a.url else None)
        res = c.pair(a.pin, code, a.name)
        keep = " and a refresh token (survives restarts)" if res.get("refresh_token") else ""
        print(f"connected as {res['agent']['name']}: {', '.join(res['agent']['permissions'])} (token{keep} saved)")
        return 0
    if not c.token and not c.state.get("refresh_token"):
        raise SystemExit("no token: run `pair` first or pass --token")
    if a.cmd == "reconnect":
        moved = c.find_address()
        ok = c.refresh()
        for n in c.notes:
            print(n)
        print(f"address: {c.url}" + ("" if moved else " (unchanged)"))
        return 0 if ok else 1
    if a.cmd == "status":
        print(json.dumps(c.call("get_status"), indent=2))
    elif a.cmd == "context":
        print(json.dumps(c.call("get_context"), indent=2))
    elif a.cmd == "stream":
        c.stream(int(a.args[0]) if a.args else 0)
    elif a.cmd == "events":
        print(json.dumps(c.call("wait_for_event", since=int(a.args[0]) if a.args else 0, timeout_s=30), indent=2))
    elif a.cmd == "ask":
        q, *opts = a.args or ["Is this OK?"]
        print(json.dumps(c.call("ask_user", question=q, options=opts or None), indent=2))
    elif a.cmd == "notify":
        print(json.dumps(c.call("notify_user", text=" ".join(a.args) or "Hello from your agent")))
    elif a.cmd in ("look", "screen"):
        res = c.call("look") if a.cmd == "look" else c.call("get_screen", elements=False)
        save_image(res, a.args[0] if a.args else f"{a.cmd}.jpg")
    elif a.cmd == "set":
        key, value = a.args[0], json.loads(a.args[1])
        print(
            json.dumps(
                c.call(
                    "change_settings", ops=[{"domain": "settings", "op": "set", "key": key, "value": value}], why=a.why
                ),
                indent=2,
            )
        )
    elif a.cmd == "undo":
        print(json.dumps(c.call("undo_changes", what=a.args[0] if a.args else "last"), indent=2))
    elif a.cmd == "bundle":
        info = c.call("get_bundle")
        raw = base64.b64decode(c.call("read_file", rel=info["path"])["b64"])
        out = a.args[0] if a.args else Path(info["path"]).name
        Path(out).write_bytes(raw)
        print(f"saved {out} ({len(raw)} bytes)")
    elif a.cmd == "logs":
        print("\n".join(c.call("get_logs", lines=int(a.args[0]) if a.args else 100)["app_log"]))
    elif a.cmd == "selftest":
        print(json.dumps(c.call("run_selftest", wait=True), indent=2))
    elif a.cmd == "selftest-latest":
        print(json.dumps(c.request("GET", "/v1/selftest/latest"), indent=2))
    elif a.cmd == "call":
        print(
            json.dumps(c.call(a.args[0], **(json.loads(a.args[1]) if len(a.args) > 1 else {})), indent=2, default=str)
        )
    for n in c.notes:
        print(f"note: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
