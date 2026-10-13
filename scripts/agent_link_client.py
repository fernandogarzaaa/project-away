#!/usr/bin/env python3
"""A tiny, dependency-free Agent Link client: an example for agents that do not speak MCP (or for poking at it
from a terminal or a cloud sandbox).

    # once: exchange the pairing code (from Settings > Agent Link, or `away agent pair`) for a token. The person
    # at the PC is asked to Allow it.
    python scripts/agent_link_client.py --url https://<name>.trycloudflare.com pair --pin 123456 --name "My agent"

    # then (the token is kept in ~/.away-agent-token, or pass --token / AWAY_TOKEN)
    python scripts/agent_link_client.py --url URL status
    python scripts/agent_link_client.py --url URL stream                 # live events (server-sent events)
    python scripts/agent_link_client.py --url URL context
    python scripts/agent_link_client.py --url URL ask "Ready to start?" Yes No
    python scripts/agent_link_client.py --url URL notify "Done!"
    python scripts/agent_link_client.py --url URL look out.jpg           # needs "See my camera when I allow"
    python scripts/agent_link_client.py --url URL screen out.jpg         # needs "See my screen when I allow"
    python scripts/agent_link_client.py --url URL set hands.cursor.gain 1.8 --why "you said it is too fast"
    python scripts/agent_link_client.py --url URL bundle out.zip
    python scripts/agent_link_client.py --url URL call <tool> '{"json": "args"}'   # any tool (see /openapi.json)

MCP clients do not need this: point them at ``URL/mcp`` with ``Authorization: Bearer <token>``, or at the pairing
link itself (``URL/c/<code>``), which works as a connector URL with no header at all.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOKEN_FILE = Path.home() / ".away-agent-token"


class Client:
    def __init__(self, url: str, token: str | None = None, timeout: float = 330.0) -> None:
        self.url = url.rstrip("/")
        if "/c/" in self.url:
            self.url = self.url.split("/c/")[0]
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, body: Any = None, raw: bool = False) -> Any:
        headers = {"Content-Type": "application/json", "User-Agent": "away-agent-client"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                return (payload, dict(resp.headers)) if raw else json.loads(payload.decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            try:
                msg = json.loads(exc.read().decode("utf-8")).get("error")
            except Exception:
                msg = str(exc)
            raise SystemExit(f"error {exc.code}: {msg}") from None

    def call(self, tool: str, **kwargs: Any) -> Any:
        return self.request("POST", f"/v1/rpc/{tool}", kwargs)

    def pair(self, pin: str | None, code: str | None, name: str) -> dict[str, Any]:
        return dict(self.request("POST", "/v1/pair", {"pin": pin, "code": code, "name": name}))

    def stream(self, since: int = 0) -> None:
        req = urllib.request.Request(
            f"{self.url}/v1/events/stream?since={since}",
            headers={"Authorization": f"Bearer {self.token}", "Accept": "text/event-stream"},
        )
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for line in resp:
                text = line.decode("utf-8").rstrip()
                if text.startswith("data: "):
                    ev = json.loads(text[6:])
                    print(f"#{ev['seq']} {ev['topic']}: {json.dumps(ev['data'])}", flush=True)


def save_image(result: dict[str, Any], out: str) -> None:
    Path(out).write_bytes(base64.b64decode(result["image_b64"]))
    print(f"saved {out} {json.dumps(result.get('meta'))}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="the Agent Link address (the pairing link works too)")
    p.add_argument("--token", default=os.environ.get("AWAY_TOKEN"))
    p.add_argument(
        "cmd",
        choices=[
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
            "call",
        ],
    )
    p.add_argument("args", nargs="*")
    p.add_argument("--pin")
    p.add_argument("--code")
    p.add_argument("--name", default="Agent Link example client")
    p.add_argument("--why", default="")
    a = p.parse_args(argv)
    token = a.token or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None)
    c = Client(a.url, token)
    if a.cmd == "pair":
        code = a.code or (a.url.split("/c/")[1].split("/")[0] if "/c/" in a.url else None)
        res = c.pair(a.pin, code, a.name)
        TOKEN_FILE.write_text(res["token"])
        if os.name != "nt":
            TOKEN_FILE.chmod(0o600)
        print(f"connected as {res['agent']['name']}: {', '.join(res['agent']['permissions'])} (token saved)")
        return 0
    if not c.token:
        raise SystemExit("no token: run `pair` first or pass --token")
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
    elif a.cmd == "call":
        print(
            json.dumps(c.call(a.args[0], **(json.loads(a.args[1]) if len(a.args) > 1 else {})), indent=2, default=str)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
