"""One small HTTP client for every live-agent call (stdlib only, no new dependency).

* Every request sends ``User-Agent: ProjectAway/<version>`` (Groq answers 403 without a User-Agent).
* Calls to this computer (127.0.0.1 / localhost: opencode serve, Ollama, LM Studio) never go through a proxy;
  a system proxy would otherwise swallow them ("404 not found").
* Errors become :class:`HttpError` with the status, a short body and the ``Retry-After`` hint, so the funnel can
  tell "slow down" (429) from "broken" (5xx) from "wrong key" (401/403).
* :class:`Transport` is the seam tests replace (fake latency, 429s, failures) without a network.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import urlparse


def user_agent() -> str:
    try:
        from away import __version__

        return f"ProjectAway/{__version__}"
    except Exception:  # pragma: no cover - the package always has a version
        return "ProjectAway/0"


class HttpError(RuntimeError):
    def __init__(self, status: int, message: str, body: str = "", retry_after: float | None = None) -> None:
        super().__init__(f"{status} {message}".strip())
        self.status = int(status)
        self.body = body[:2000]
        self.retry_after = retry_after


def parse_retry_after(value: Any, now: float | None = None) -> float | None:
    """``Retry-After``: seconds or an HTTP date -> seconds from now (>= 0), None when missing/unreadable.
    (Ported from ape-mcp freepool ``parseRetryAfter``, MIT, Fernando Garza.)"""
    if value in (None, ""):
        return None
    s = str(value).strip()
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    try:
        t = parsedate_to_datetime(s).timestamp()
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, t - (time.time() if now is None else now))


def is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")


class Response:
    def __init__(self, status: int, headers: dict[str, str], lines: Any, close: Any = None) -> None:
        self.status = status
        self.headers = {k.lower(): v for k, v in headers.items()}
        self._lines = lines
        self._close = close

    def iter_lines(self) -> Iterator[str]:
        try:
            for raw in self._lines:
                yield raw.decode("utf-8", "replace").rstrip("\r\n")
        finally:
            self.close()

    def read(self) -> bytes:
        try:
            return b"".join(self._lines)
        finally:
            self.close()

    def json(self) -> Any:
        data = self.read()
        return json.loads(data) if data.strip() else None

    def close(self) -> None:
        if self._close is not None:
            try:
                self._close()
            except Exception:
                pass
            self._close = None


class Transport(Protocol):
    def request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> Response: ...


class UrllibTransport:
    """The real network (urllib). Proxies are skipped for this computer's own addresses."""

    def __init__(self) -> None:
        self._direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._default = urllib.request.build_opener()

    def request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> Response:  # pragma: no cover - network
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        opener = self._direct if is_local(url) else self._default
        try:
            resp = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            text = ""
            try:
                text = exc.read().decode("utf-8", "replace")
            except Exception:
                pass
            ra = parse_retry_after(exc.headers.get("Retry-After")) if exc.headers else None
            raise HttpError(exc.code, short_error(text) or str(exc.reason), text, ra) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise HttpError(0, f"could not connect: {reason}") from None
        return Response(int(resp.status), dict(resp.headers.items()), iter(resp), resp.close)


def short_error(text: str) -> str:
    """The provider's own error message out of a JSON error body (OpenAI / Anthropic / opencode shapes)."""
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return text.strip()[:200]
    if isinstance(d, dict):
        e = d.get("error", d)
        if isinstance(e, dict):
            return str(e.get("message") or e.get("type") or "")[:200]
        if isinstance(e, str):
            return e[:200]
        if isinstance(d.get("message"), str):
            return str(d["message"])[:200]
    return ""


class Client:
    def __init__(self, transport: Transport | None = None) -> None:
        self.transport = transport or UrllibTransport()

    def _headers(self, extra: dict[str, str] | None, has_body: bool) -> dict[str, str]:
        h = {"User-Agent": user_agent(), "Accept": "application/json"}
        if has_body:
            h["Content-Type"] = "application/json"
        if extra:
            h.update(extra)
        return h

    def send(
        self,
        method: str,
        url: str,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Response:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        resp = self.transport.request(method, url, self._headers(headers, body is not None), body, timeout)
        if resp.status >= 400:
            text = resp.read().decode("utf-8", "replace")
            raise HttpError(resp.status, short_error(text), text, parse_retry_after(resp.headers.get("retry-after")))
        return resp

    def get_json(self, url: str, headers: dict[str, str] | None = None, timeout: float = 20.0) -> Any:
        return self.send("GET", url, None, headers, timeout).json()

    def post_json(self, url: str, payload: Any, headers: dict[str, str] | None = None, timeout: float = 30.0) -> Any:
        return self.send("POST", url, payload, headers, timeout).json()

    def sse(
        self,
        method: str,
        url: str,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> Iterator[dict[str, Any]]:
        """Server-sent events: yields each ``data:`` JSON object (``[DONE]`` ends the stream)."""
        h = dict(headers or {})
        h["Accept"] = "text/event-stream"
        resp = self.send(method, url, payload, h, timeout)
        event = ""
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if isinstance(obj, dict):
                if event and "type" not in obj:
                    obj["type"] = event
                yield obj
            event = ""
