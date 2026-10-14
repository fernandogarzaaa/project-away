"""What may be sent to which model.

Two kinds of providers:

* **private enough** - your own paid keys (OpenAI, Anthropic, Groq...) and local models (Ollama, LM Studio);
* **may keep prompts** - free models that may store or train on what you send (opencode's free models such as
  Big Pickle; Nvidia's free Nemotron endpoints say not to send personal data; Gemini's free tier).

Rules, enforced in one place (:func:`prepare`) right before any call leaves the PC:

1. Providers that may keep prompts NEVER receive screenshots, camera frames, or text typed into password fields.
2. Text going to them is redacted first: e-mail addresses, phone and card-like numbers, API keys and tokens,
   one-time codes and reset links become ``[email]`` / ``[number]`` / ``[redacted]``.
3. Secrets (API keys, bearer tokens, private keys) are redacted for every provider, private or not.

The patterns are ported from GHOST-Chimera ``ghostchimera/redaction.py`` and
``ghostchimera/model_layer/sensitivity.py`` (MIT, Fernando Garza).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?:ghp|gho|github_pat|xoxb|xoxp|glpat)_[A-Za-z0-9_\-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.=]{12,}", re.IGNORECASE),
    re.compile(r"(?:gsk_|xai-|AIza)[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|pwd|token)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
    re.compile(r"\bpa_[A-Za-z0-9_-]{8,}"),  # Agent Link tokens
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
LINK_SECRET = re.compile(r"https?://\S*(?:reset|recover|verify-email|magic-link|one-time|token=|code=)[^\s]*", re.I)
OTP = re.compile(r"(?i)\b(code|otp|pin|passcode)(\s*(?:is|:)?\s*)\d{4,8}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w.])")
POISON = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"exfiltrat(?:e|ion)", re.IGNORECASE),
    re.compile(r"send\s+(?:the\s+)?(?:secret|token|password|credential)", re.IGNORECASE),
)


def redact_secrets(text: str) -> str:
    out = str(text or "")
    for p in SECRET_PATTERNS:
        out = p.sub("[redacted]", out)
    return out


def redact_personal(text: str) -> str:
    """Secrets + personal details (for providers that may keep prompts)."""
    out = redact_secrets(text)
    out = LINK_SECRET.sub("[link]", out)
    out = EMAIL.sub("[email]", out)
    out = OTP.sub(lambda m: f"{m.group(1)}{m.group(2)}[number]", out)
    out = SSN.sub("[number]", out)
    out = CARD.sub("[number]", out)
    out = PHONE.sub("[number]", out)
    return out


def looks_poisoned(text: str) -> bool:
    """A prompt-injection phrase (logged; the text is still fenced, never obeyed)."""
    return any(p.search(text or "") for p in POISON)


@dataclass
class Prepared:
    messages: list[dict[str, Any]]
    dropped_images: int = 0
    redacted: bool = False
    dropped_password_text: int = 0


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


_text_of = text_of  # older name


def prepare(messages: list[dict[str, Any]], may_keep_prompts: bool, allow_images: bool = True) -> Prepared:
    """A copy of ``messages`` safe to send. Messages are ``{"role", "content"}`` where content is text or a list of
    parts (``{"type": "text", "text"}`` / ``{"type": "image", "data": <jpeg bytes>, "mime"}``). A message marked
    ``"password": True`` (text that came from a password field) is dropped for providers that may keep prompts."""
    out: list[dict[str, Any]] = []
    res = Prepared(out)
    for m in messages:
        m2 = copy.copy(m)
        if m.get("password") and may_keep_prompts:
            res.dropped_password_text += 1
            continue
        m2.pop("password", None)
        content = m.get("content")
        fix = redact_personal if may_keep_prompts else redact_secrets
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            dropped_here = 0
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "image":
                    if may_keep_prompts or not allow_images:
                        dropped_here += 1
                        continue
                    parts.append(p)
                elif p.get("type") == "text":
                    t = fix(str(p.get("text", "")))
                    res.redacted = res.redacted or t != p.get("text")
                    parts.append({**p, "text": t})
            res.dropped_images += dropped_here
            if dropped_here:
                parts.append({"type": "text", "text": "(a picture was left out for privacy)"})
            m2["content"] = parts
        else:
            t = fix(str(content or ""))
            res.redacted = res.redacted or t != (content or "")
            m2["content"] = t
        out.append(m2)
    return res


def has_images(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") == "image" for p in m["content"])
        for m in messages
    )
