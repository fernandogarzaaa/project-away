"""Known model providers and plain facts about models.

Nothing here is the only option: every OpenAI-compatible endpoint works through ``custom``, the opencode free list
is discovered at run time (free models rotate), and an API provider's own model list is read when you press Test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ZEN_MODELS_URL = "https://opencode.ai/zen/v1/models"
MODELS_DEV_URL = "https://models.dev/api.json"


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    title: str
    kind: str  # opencode | openai | anthropic
    base_url: str = ""
    default_model: str = ""
    needs_key: bool = True
    key_url: str = ""
    free_tier: bool = False  # a free tier exists (with a key)
    may_keep_prompts: bool = False  # free tiers that may keep / train on prompts
    local: bool = False
    note: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        "opencode_free",
        "Free models (opencode)",
        "opencode",
        needs_key=False,
        free_tier=True,
        may_keep_prompts=True,
        note="No key needed. Free models may keep what you send, so they never get pictures of your screen.",
    ),
    ProviderSpec(
        "groq",
        "Groq",
        "openai",
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-20b",
        key_url="https://console.groq.com/keys",
        free_tier=True,
        note="Very fast. Free key, no card.",
    ),
    ProviderSpec(
        "gemini",
        "Google Gemini",
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-flash-latest",
        key_url="https://aistudio.google.com/apikey",
        free_tier=True,
        may_keep_prompts=True,
        note="Free key, no card. Google may use free-tier prompts to improve its products.",
    ),
    ProviderSpec(
        "openai", "OpenAI", "openai", "https://api.openai.com/v1", "gpt-5-mini", key_url="https://platform.openai.com/api-keys"
    ),
    ProviderSpec(
        "anthropic",
        "Anthropic (Claude)",
        "anthropic",
        "https://api.anthropic.com/v1",
        "claude-haiku-4-5",
        key_url="https://console.anthropic.com/settings/keys",
    ),
    ProviderSpec(
        "openrouter",
        "OpenRouter",
        "openai",
        "https://openrouter.ai/api/v1",
        "openrouter/auto",
        key_url="https://openrouter.ai/keys",
        free_tier=True,
        note="Many models with one key. Models ending in ':free' may keep prompts.",
        extra_headers={"HTTP-Referer": "https://github.com/fernandogarzaaa/project-away", "X-Title": "Project Away"},
    ),
    ProviderSpec(
        "opencode_zen",
        "OpenCode Zen (your key)",
        "openai",
        "https://opencode.ai/zen/v1",
        "claude-haiku-4-5",
        key_url="https://opencode.ai/auth",
    ),
    ProviderSpec(
        "deepseek", "DeepSeek", "openai", "https://api.deepseek.com/v1", "deepseek-chat",
        key_url="https://platform.deepseek.com/api_keys",
    ),
    ProviderSpec(
        "mistral", "Mistral", "openai", "https://api.mistral.ai/v1", "mistral-small-latest",
        key_url="https://console.mistral.ai/api-keys",
    ),
    ProviderSpec(
        "together",
        "Together AI",
        "openai",
        "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        key_url="https://api.together.ai/settings/api-keys",
    ),
    ProviderSpec("ollama", "Ollama (on this PC)", "openai", "http://127.0.0.1:11434/v1", "", needs_key=False, local=True),
    ProviderSpec("lmstudio", "LM Studio (on this PC)", "openai", "http://127.0.0.1:1234/v1", "", needs_key=False, local=True),
    ProviderSpec("custom", "Other (OpenAI-compatible)", "openai", "", "", needs_key=False),
)
BY_ID = {p.id: p for p in PROVIDERS}
FREE_KEY_PROVIDERS = ("gemini", "groq")  # the "Use a free key instead" fallback


def spec(provider_kind: str) -> ProviderSpec:
    return BY_ID.get(provider_kind, BY_ID["custom"])


# ------------------------------------------------------------------------------------------------ model facts
_VISION = re.compile(
    r"(vision|gpt-4o|gpt-4\.1|gpt-5|gemini|claude|llava|-vl|vl-|qwen.*vl|pixtral|mimo|omni|llama-4|grok-4|"
    r"step-5|longcat-2\.5|muse-spark|space-bunny|kimi-k2\.5|minimax-m3|moondream|gemma-3)",
    re.IGNORECASE,
)
_REASONING = re.compile(
    r"(nemotron|deepseek-r1|-r1|reason|think|qwq|gpt-oss|qwen3|\bo1\b|\bo3\b|\bo4\b|o\d-mini|magistral|"
    r"big-pickle|glm-4\.[5-9]|glm-5|kimi|minimax|ling-|ring-|longcat|step-)",
    re.IGNORECASE,
)
_FAST = re.compile(r"(flash|mini|nano|lite|instant|haiku|8b|small|lightning|tiny|turbo|20b|fast)", re.IGNORECASE)
_STRONG = re.compile(r"(pro|opus|sonnet|gpt-5(?!-nano)|large|70b|120b|max|ultra|big-pickle|plus|405b)", re.IGNORECASE)
_NOT_CHAT = re.compile(r"(whisper|embed|tts|orpheus|guard|moderation|rerank|dall-e|image|audio|transcribe)", re.I)


@dataclass
class ModelInfo:
    provider: str  # provider config id (e.g. "groq", "opencode_free")
    id: str
    title: str = ""
    vision: bool = False
    reasoning: bool = False
    fast: bool = False
    strong: bool = False
    tools: bool = True
    free: bool = False
    may_keep_prompts: bool = False
    local: bool = False
    price_in: float = 0.0  # USD per 1M input tokens
    price_out: float = 0.0

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.id}"

    def label(self) -> str:
        return self.title or pretty(self.id)

    def public(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "provider": self.provider,
            "model": self.id,
            "title": self.label(),
            "vision": self.vision,
            "free": self.free,
            "may_keep_prompts": self.may_keep_prompts,
            "local": self.local,
        }


def pretty(model_id: str) -> str:
    """``big-pickle`` -> "Big Pickle", ``nemotron-3.5-lightning-free`` -> "Nemotron 3.5 Lightning (free)"."""
    base = model_id.split("/")[-1]
    free = base.endswith("-free") or base.endswith(":free")
    base = re.sub(r"[-:]free$", "", base)
    words = [w if any(c.isdigit() for c in w) else w.capitalize() for w in re.split(r"[-_]", base) if w]
    out = " ".join(words) or model_id
    return f"{out} (free)" if free else out


def guess(provider: str, model_id: str, **kw: Any) -> ModelInfo:
    return ModelInfo(
        provider,
        model_id,
        vision=bool(_VISION.search(model_id)),
        reasoning=bool(_REASONING.search(model_id)),
        fast=bool(_FAST.search(model_id)),
        strong=bool(_STRONG.search(model_id)),
        **kw,
    )


def chat_capable(model_id: str) -> bool:
    return not _NOT_CHAT.search(model_id or "")


def pick_default(ids: list[str], wanted: str = "") -> str:
    """A sensible model from a provider's own list: the wanted one, else a fast general chat model."""
    chat = [m for m in ids if chat_capable(m)]
    if wanted and wanted in chat:
        return wanted
    for pat in (r"flash|mini|haiku|small|instant|lite|turbo", r"."):
        hits = [m for m in chat if re.search(pat, m, re.I) and "nano" not in m]
        if hits:
            return sorted(hits, key=len)[0]
    return chat[0] if chat else wanted


# ------------------------------------------------------------------------------------------------ opencode free
DECISION_MODELS = ("jev-",)  # Jev answers questions about a state; it does not chat
EXCLUDE = ("test",)


def is_free_id(model_id: str) -> bool:
    return model_id == "big-pickle" or model_id.endswith("-free")


def opencode_free_models(
    zen_ids: list[str] | None = None,
    models_dev: dict[str, Any] | None = None,
    served: dict[str, Any] | list[str] | None = None,
) -> list[ModelInfo]:
    """The free chat models opencode offers right now.

    ``served``: the running ``opencode serve``'s own ``opencode`` provider models (authoritative when known; a dict
    of models.dev-style entries or a list of ids); ``zen_ids``: https://opencode.ai/zen/v1/models; ``models_dev``:
    models.dev's data (cost 0 + capabilities). Jev (a decision model) and test entries are left out."""
    md: dict[str, Any] = {}
    if isinstance(models_dev, dict):
        md = dict((models_dev.get("opencode") or {}).get("models") or models_dev.get("models") or {})
    if isinstance(served, dict):
        md = {**md, **served}
        ids = list(served)
    elif served:
        ids = list(served)
    elif zen_ids:
        ids = list(zen_ids)
    else:
        ids = list(md)
    out: list[ModelInfo] = []
    for mid in ids:
        if mid in EXCLUDE or any(mid.startswith(p) for p in DECISION_MODELS):
            continue
        meta = md.get(mid) or {}
        cost = meta.get("cost") or {}
        try:
            zero = bool(cost) and float(cost.get("input", 1) or 0) == 0 and float(cost.get("output", 1) or 0) == 0
        except (TypeError, ValueError):
            zero = False
        if not (is_free_id(mid) or zero):
            continue
        info = guess("opencode_free", mid, free=True, may_keep_prompts=True)
        mods = (meta.get("modalities") or {}).get("input")
        if isinstance(mods, list):
            info.vision = "image" in mods
        caps = meta.get("capabilities") or {}
        if isinstance(caps, dict) and isinstance(caps.get("input"), dict):
            info.vision = bool(caps["input"].get("image", info.vision))
        if "tool_call" in meta:
            info.tools = bool(meta.get("tool_call"))
        if meta.get("reasoning") is not None:
            info.reasoning = info.reasoning or bool(meta.get("reasoning"))
        name = str(meta.get("name") or "")
        if name:
            info.title = name if ("free" in name.lower() or mid == "big-pickle") else f"{name} (free)"
        out.append(info)
    out.sort(key=lambda m: (m.id != "big-pickle", not m.fast, m.id))  # Big Pickle (opencode's default) first
    return out


def price_for(models_dev: dict[str, Any] | None, provider_hint: str, model_id: str) -> tuple[float, float]:
    """USD per 1M tokens (input, output) from models.dev, (0, 0) when unknown."""
    if not isinstance(models_dev, dict):
        return (0.0, 0.0)
    names = {"opencode_zen": "opencode", "gemini": "google", "together": "togetherai"}
    block = models_dev.get(names.get(provider_hint, provider_hint)) or {}
    meta = (block.get("models") or {}).get(model_id) or {}
    cost = meta.get("cost") or {}
    try:
        return (float(cost.get("input", 0) or 0), float(cost.get("output", 0) or 0))
    except (TypeError, ValueError):
        return (0.0, 0.0)
