"""Config for smart dictation cleanup (``voice.cleanup``) and the personal dictionary (``voice.dictionary``).

Kept in its own dependency-free module so :mod:`away.config` can import it without pulling in the voice stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CLEANUP_MODES = ("off", "rules", "llm")
CLEANUP_BACKENDS = ("llama_cpp", "ollama", "openai")
# auto = the downloaded cleanup model that suits this PC; a key from away.app.optional_models; a .gguf path;
# or, for the ollama/openai backends, the server's model name
CLEANUP_MODEL_KEYS = ("qwen2.5-1.5b", "qwen2.5-0.5b")
TONES = ("auto", "casual", "neutral", "formal", "verbatim", "code")


@dataclass
class CleanupConfig:
    # off: Whisper's text as heard | rules: fillers, spoken edits, punctuation (default) | llm: rules + a small
    # local language model that polishes punctuation and adapts tone to the focused app ("Smart cleanup")
    mode: str = "rules"
    backend: str = "llama_cpp"  # llama_cpp (in-process GGUF) | ollama (local server) | openai (AWAY_LLM_* endpoint)
    model: str = "auto"
    timeout_ms: int = 1500  # longer than this and the rule-based text is typed instead
    ollama_url: str = "http://localhost:11434"
    # per-app tone: process name (slack.exe, Code) or bindings profile name -> auto|casual|neutral|formal|verbatim|code
    tones: dict[str, str] = field(default_factory=dict)
    context: bool = True  # read the text before the cursor with UI Automation (no keystrokes, Windows only)
    context_clipboard: bool = False  # also copy the selection (clipboard is restored; never in terminals/editors)
    gpu_layers: int = -1  # llama_cpp: layers on the GPU (-1 = all when the engine has GPU support, 0 = CPU)


@dataclass
class DictionaryConfig:
    enabled: bool = True
    file: str | None = None  # None = dictionary.json in the config folder
    learn: bool = True  # learn from your corrections of just-dictated text
    learn_after: int = 2  # a fix learned from an edit counts after it was seen this many times
    watch_seconds: float = 20.0  # how long after typing to watch the text field for your corrections


def validate_cleanup(cleanup: Any, dictionary: Any) -> None:
    """Raises ValueError with a config path in the message (called from :func:`away.config.validate`)."""
    c = cleanup
    if c.mode is False:  # YAML 1.1 reads a bare `off` as false
        c.mode = "off"
    if c.mode not in CLEANUP_MODES:
        raise ValueError(f"voice.cleanup.mode must be {'|'.join(CLEANUP_MODES)}, got {c.mode!r}")
    if c.backend not in CLEANUP_BACKENDS:
        raise ValueError(f"voice.cleanup.backend must be {'|'.join(CLEANUP_BACKENDS)}, got {c.backend!r}")
    if not isinstance(c.model, str) or not c.model.strip():
        raise ValueError("voice.cleanup.model must be 'auto', a model key or a .gguf path")
    t = c.timeout_ms
    if isinstance(t, bool) or not isinstance(t, int) or not 100 <= t <= 30000:
        raise ValueError(f"voice.cleanup.timeout_ms must be an integer in [100, 30000], got {t!r}")
    if not isinstance(c.tones, dict):
        raise ValueError("voice.cleanup.tones must be a mapping of app or profile name -> tone")
    for app, tone in c.tones.items():
        if tone is False:
            tone = c.tones[app] = "verbatim"
        if tone not in TONES:
            raise ValueError(f"voice.cleanup.tones.{app} must be {'|'.join(TONES)}, got {tone!r}")
    for name in ("context", "context_clipboard"):
        if not isinstance(getattr(c, name), bool):
            raise ValueError(f"voice.cleanup.{name} must be true or false, got {getattr(c, name)!r}")
    g = c.gpu_layers
    if isinstance(g, bool) or not isinstance(g, int) or not -1 <= g <= 1000:
        raise ValueError(f"voice.cleanup.gpu_layers must be an integer in [-1, 1000], got {g!r}")
    d = dictionary
    for name in ("enabled", "learn"):
        if not isinstance(getattr(d, name), bool):
            raise ValueError(f"voice.dictionary.{name} must be true or false, got {getattr(d, name)!r}")
    la = d.learn_after
    if isinstance(la, bool) or not isinstance(la, int) or not 1 <= la <= 20:
        raise ValueError(f"voice.dictionary.learn_after must be an integer in [1, 20], got {la!r}")
    ws = d.watch_seconds
    if isinstance(ws, bool) or not isinstance(ws, (int, float)) or not 2 <= float(ws) <= 300:
        raise ValueError(f"voice.dictionary.watch_seconds must be seconds in [2, 300], got {ws!r}")
