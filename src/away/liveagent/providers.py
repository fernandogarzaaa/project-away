"""Model providers behind one small interface: ``stream(model, messages, max_tokens=...)`` yields text pieces.

* :class:`OpenAICompatProvider` - OpenAI, Groq, Gemini (its OpenAI endpoint), OpenRouter, DeepSeek, Mistral,
  Together, OpenCode Zen with your key, Ollama, LM Studio, any other compatible server;
* :class:`AnthropicProvider` - Claude through Anthropic's own API;
* :class:`OpenCodeProvider` - opencode's free models through a private ``opencode serve`` (:mod:`.opencode`);
* :class:`FakeProvider` - scripted replies, latency, 429s and failures for tests.

Messages are ``{"role": "system"|"user"|"assistant", "content": str | [parts]}``; an image part is
``{"type": "image", "data": <jpeg bytes>, "mime": "image/jpeg"}``. Privacy filtering happens before this layer
(:func:`away.liveagent.privacy.prepare`); reasoning traces are stripped after it (:mod:`.textstream`).
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from away.liveagent import catalog
from away.liveagent.http import Client, HttpError


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider:
    id = ""
    title = ""
    may_keep_prompts = False
    local = False
    free = False

    def __init__(self) -> None:
        self.last_usage = Usage()

    def stream(
        self, model: str, messages: list[dict[str, Any]], *, max_tokens: int = 400, json_mode: bool = False,
        timeout: float = 60.0,
    ) -> Iterator[str]:  # pragma: no cover - interface
        raise NotImplementedError
        yield ""

    def complete(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> str:
        return "".join(self.stream(model, messages, **kw))

    def list_models(self) -> list[catalog.ModelInfo]:
        return []

    def test(self, model: str = "") -> str:
        """A tiny real call. Returns a plain "works" line; raises with a plain reason."""
        models = self.list_models()
        use = model or (models[0].id if models else "")
        t0 = time.monotonic()
        text = self.complete(use, [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=200)
        from away.liveagent.textstream import strip_reasoning

        if not strip_reasoning(text):
            raise RuntimeError("the model answered with nothing")
        return f"Works ({catalog.pretty(use)}, {time.monotonic() - t0:.1f} s)"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


def _b64(data: Any) -> str:
    return data if isinstance(data, str) else base64.b64encode(bytes(data)).decode("ascii")


# ------------------------------------------------------------------------------------------------ OpenAI-compatible
class OpenAICompatProvider(Provider):
    def __init__(
        self,
        pid: str,
        base_url: str,
        api_key: str = "",
        *,
        title: str = "",
        may_keep_prompts: bool = False,
        local: bool = False,
        free: bool = False,
        extra_headers: dict[str, str] | None = None,
        client: Client | None = None,
        kind: str = "",
    ) -> None:
        super().__init__()
        self.id = pid
        self.kind = kind or pid
        self.title = title or pid
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.may_keep_prompts = may_keep_prompts
        self.local = local
        self.free = free
        self.extra_headers = dict(extra_headers or {})
        self.client = client or Client()

    def _headers(self) -> dict[str, str]:
        h = dict(self.extra_headers)
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    @staticmethod
    def convert(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                parts: list[dict[str, Any]] = []
                for p in c:
                    if p.get("type") == "text":
                        parts.append({"type": "text", "text": str(p.get("text", ""))})
                    elif p.get("type") == "image":
                        url = f"data:{p.get('mime', 'image/jpeg')};base64,{_b64(p.get('data', b''))}"
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                out.append({"role": m["role"], "content": parts})
            else:
                out.append({"role": m["role"], "content": str(c or "")})
        return out

    def _uses_completion_tokens(self, model: str) -> bool:
        return self.kind == "openai" and (model.startswith(("o1", "o3", "o4", "gpt-5")))

    def stream(
        self, model: str, messages: list[dict[str, Any]], *, max_tokens: int = 400, json_mode: bool = False,
        timeout: float = 60.0,
    ) -> Iterator[str]:
        body: dict[str, Any] = {"model": model, "messages": self.convert(messages), "stream": True}
        body["max_completion_tokens" if self._uses_completion_tokens(model) else "max_tokens"] = int(max_tokens)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.kind in ("openai", "groq", "opencode_zen", "deepseek", "together", "openrouter"):
            body["stream_options"] = {"include_usage": True}
        self.last_usage = Usage()
        for ev in self.client.sse("POST", f"{self.base_url}/chat/completions", body, self._headers(), timeout):
            if "error" in ev:
                e = ev["error"]
                msg = e.get("message", "") if isinstance(e, dict) else str(e)
                code = int(e.get("code", 0) or 0) if isinstance(e, dict) and str(e.get("code", "")).isdigit() else 500
                raise HttpError(code, str(msg)[:200])
            u = ev.get("usage")
            if isinstance(u, dict):
                self.last_usage = Usage(u.get("prompt_tokens"), u.get("completion_tokens"))
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or ch.get("message") or {}
                piece = d.get("content")
                if isinstance(piece, str) and piece:
                    yield piece  # reasoning_content / reasoning are never spoken

    def list_models(self) -> list[catalog.ModelInfo]:
        data = self.client.get_json(f"{self.base_url}/models", self._headers(), timeout=15)
        ids = [str(m.get("id")) for m in (data or {}).get("data", []) if isinstance(m, dict) and m.get("id")]
        if not ids and isinstance(data, dict) and isinstance(data.get("models"), list):  # Ollama /api/tags shape
            ids = [str(m.get("name") or m.get("model")) for m in data["models"] if isinstance(m, dict)]
        return [
            catalog.guess(
                self.id,
                i,
                local=self.local,
                free=self.free or self.local or i.endswith(":free"),
                may_keep_prompts=self.may_keep_prompts or i.endswith(":free"),
            )
            for i in ids
            if catalog.chat_capable(i)
        ]


# ------------------------------------------------------------------------------------------------ Anthropic
class AnthropicProvider(Provider):
    VERSION = "2023-06-01"

    def __init__(self, pid: str, api_key: str, base_url: str = "https://api.anthropic.com/v1", *,
                 title: str = "Anthropic", client: Client | None = None) -> None:
        super().__init__()
        self.id = pid
        self.title = title
        self._key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = client or Client()

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._key, "anthropic-version": self.VERSION}

    @staticmethod
    def convert(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                continue
            c = m.get("content")
            if isinstance(c, list):
                blocks: list[dict[str, Any]] = []
                for p in c:
                    if p.get("type") == "text":
                        blocks.append({"type": "text", "text": str(p.get("text", ""))})
                    elif p.get("type") == "image":
                        blocks.append({"type": "image", "source": {"type": "base64",
                                       "media_type": p.get("mime", "image/jpeg"), "data": _b64(p.get("data", b""))}})
                out.append({"role": m["role"], "content": blocks})
            else:
                out.append({"role": m["role"], "content": str(c or "")})
        return system, out

    def stream(
        self, model: str, messages: list[dict[str, Any]], *, max_tokens: int = 400, json_mode: bool = False,
        timeout: float = 60.0,
    ) -> Iterator[str]:
        system, msgs = self.convert(messages)
        body: dict[str, Any] = {"model": model, "messages": msgs, "max_tokens": int(max_tokens), "stream": True}
        if system:
            body["system"] = system
        self.last_usage = Usage()
        for ev in self.client.sse("POST", f"{self.base_url}/messages", body, self._headers(), timeout):
            t = ev.get("type")
            if t == "error":
                e = ev.get("error") or {}
                raise HttpError(529 if e.get("type") == "overloaded_error" else 500, str(e.get("message", ""))[:200])
            if t == "message_start":
                u = (ev.get("message") or {}).get("usage") or {}
                self.last_usage.input_tokens = u.get("input_tokens")
            elif t == "message_delta":
                u = ev.get("usage") or {}
                self.last_usage.output_tokens = u.get("output_tokens")
            elif t == "content_block_delta":
                d = ev.get("delta") or {}
                if d.get("type") == "text_delta" and d.get("text"):
                    yield str(d["text"])  # thinking_delta is never spoken

    def list_models(self) -> list[catalog.ModelInfo]:
        data = self.client.get_json(f"{self.base_url}/models", self._headers(), timeout=15)
        return [catalog.guess(self.id, str(m["id"])) for m in (data or {}).get("data", []) if m.get("id")]


# ------------------------------------------------------------------------------------------------ opencode (free)
class OpenCodeProvider(Provider):
    """opencode's free models, through the private server (:class:`away.liveagent.opencode.OpenCodeServer`)."""

    def __init__(self, server_factory: Callable[[], Any], pid: str = "opencode_free",
                 catalog_fn: Callable[[], list[catalog.ModelInfo]] | None = None) -> None:
        super().__init__()
        self.id = pid
        self.title = "Free models (opencode)"
        self.may_keep_prompts = True
        self.free = True
        self.server_factory = server_factory
        self.catalog_fn = catalog_fn

    def stream(
        self, model: str, messages: list[dict[str, Any]], *, max_tokens: int = 400, json_mode: bool = False,
        timeout: float = 60.0,
    ) -> Iterator[str]:
        server = self.server_factory()
        system = "\n\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
        rest = [m for m in messages if m.get("role") != "system"]
        if len(rest) == 1:
            prompt = _text(rest[0].get("content"))
        else:
            lines = []
            for m in rest[:-1]:
                who = "User" if m.get("role") == "user" else "You"
                lines.append(f"{who}: {_text(m.get('content'))}")
            prompt = "Conversation so far:\n" + "\n".join(lines) + f"\n\nUser: {_text(rest[-1].get('content'))}"
        if json_mode:
            system += "\n\nAnswer with one JSON object only."
        yield from server.chat_stream(model, prompt, system=system, timeout=timeout)

    def list_models(self) -> list[catalog.ModelInfo]:
        if self.catalog_fn is not None:
            return self.catalog_fn()
        return []


# ------------------------------------------------------------------------------------------------ fake (tests)
@dataclass
class FakeProvider(Provider):
    """Scripted provider: ``replies`` (strings, or exceptions to raise), ``latency`` before the first piece."""

    id: str = "fake"
    replies: list[Any] = field(default_factory=lambda: ["Hello there. How can I help?"])
    latency: float = 0.0
    sleep: Callable[[float], None] = time.sleep
    may_keep_prompts: bool = False
    local: bool = False
    free: bool = False
    models: list[str] = field(default_factory=lambda: ["fake-1"])
    calls: list[dict[str, Any]] = field(default_factory=list)
    title: str = "Fake"

    def __post_init__(self) -> None:
        self.last_usage = Usage()

    def stream(
        self, model: str, messages: list[dict[str, Any]], *, max_tokens: int = 400, json_mode: bool = False,
        timeout: float = 60.0,
    ) -> Iterator[str]:
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens, "json": json_mode})
        if self.latency:
            if self.latency > timeout:
                self.sleep(timeout)
                raise HttpError(0, "timed out")
            self.sleep(self.latency)
        r = self.replies.pop(0) if len(self.replies) > 1 else (self.replies[0] if self.replies else "")
        if isinstance(r, BaseException):
            raise r
        text = str(r)
        for i in range(0, len(text), 7):
            yield text[i : i + 7]

    def list_models(self) -> list[catalog.ModelInfo]:
        return [
            catalog.guess(self.id, m, free=self.free, may_keep_prompts=self.may_keep_prompts, local=self.local)
            for m in self.models
        ]


# ------------------------------------------------------------------------------------------------ factory
def build(cfg: Any, key: str, client: Client | None = None, opencode_server: Callable[[], Any] | None = None,
          free_catalog: Callable[[], list[catalog.ModelInfo]] | None = None) -> Provider:
    """A provider from a :class:`away.liveagent.settings.ProviderConfig` and its key."""
    sp = catalog.spec(cfg.kind)
    keep = sp.may_keep_prompts if cfg.may_keep_prompts is None else bool(cfg.may_keep_prompts)
    if sp.kind == "opencode":
        if opencode_server is None:
            raise RuntimeError("the free models need the opencode engine")
        return OpenCodeProvider(opencode_server, cfg.id, free_catalog)
    if sp.kind == "anthropic":
        if not key:
            raise RuntimeError(f"{sp.title}: no API key saved")
        return AnthropicProvider(cfg.id, key, cfg.base_url or sp.base_url, title=cfg.title or sp.title, client=client)
    base = cfg.base_url or sp.base_url
    if not base:
        raise RuntimeError("give the server address (for example http://127.0.0.1:8000/v1)")
    if sp.needs_key and not key:
        raise RuntimeError(f"{sp.title}: no API key saved")
    return OpenAICompatProvider(
        cfg.id,
        base,
        key,
        title=cfg.title or sp.title,
        may_keep_prompts=keep,
        local=sp.local,
        free=sp.local,
        extra_headers=sp.extra_headers,
        client=client,
        kind=sp.id,
    )
