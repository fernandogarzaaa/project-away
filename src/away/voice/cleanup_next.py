"""Smart cleanup with Qwen3.5 (v0.1.24 model refresh): non-thinking mode, a token limit, ``<think>`` removed.

Qwen3.5-2B and 0.8B (Apache-2.0, 2026-03) run in non-thinking mode by default (Qwen's model card); the card also
warns that 2B "is more prone to entering thinking loops" when thinking is on, so cleanup never turns it on, caps
``max_tokens`` (as every cleanup call already does) and strips any ``<think>...</think>`` block that still appears.

They need a llama.cpp build with the ``qwen35`` architecture. llama-cpp-python 0.3.36 (the pinned Smart cleanup
engine) loads Qwen3.5-0.8B Q4_K_M - checked on Linux (aarch64, CPU) while building this. If a Qwen3.5 file cannot
be loaded here (an older pip-installed llama-cpp-python, say), :class:`Qwen35Fallback` switches to the Qwen2.5
model when it is downloaded, and says so.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

QWEN35_KEYS = ("qwen3.5-2b", "qwen3.5-2b-q8", "qwen3.5-0.8b")
FALLBACK_KEYS = {"qwen3.5-2b": "qwen2.5-1.5b", "qwen3.5-2b-q8": "qwen2.5-1.5b", "qwen3.5-0.8b": "qwen2.5-0.5b"}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
_ARCH_WORDS = ("unknown model architecture", "qwen35", "unsupported model", "failed to load model")


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` (and an unterminated ``<think>`` tail: the model ran out of tokens while
    thinking, so nothing after it is an answer)."""
    s = _THINK_BLOCK.sub("", text or "")
    s = s.replace("</think>", "")
    s = _THINK_OPEN.sub("", s)
    return s.strip()


def is_qwen35(key: str) -> bool:
    return str(key).lower().startswith("qwen3.5")


def arch_unsupported(exc: BaseException) -> bool:
    low = f"{type(exc).__name__}: {exc}".lower()
    return any(w in low for w in _ARCH_WORDS)


class Qwen35Fallback:
    """A cleanup engine for a Qwen3.5 key: tries it, and when this llama.cpp cannot load Qwen3.5 uses Qwen2.5
    (if downloaded). ``make(path)`` builds the underlying llama.cpp backend."""

    def __init__(
        self,
        key: str,
        models_dir: Path,
        make: Callable[[Path], Any],
        path_of: Callable[[str, Path], Path],
        installed: Callable[[str, Path], bool],
        on_note: Callable[[str], None] | None = None,
    ) -> None:
        self.key = key
        self.models_dir = models_dir
        self.make = make
        self.path_of = path_of
        self.installed = installed
        self.on_note = on_note
        self.inner = make(path_of(key, models_dir))
        self.fell_back = False

    def check(self) -> None:
        fn = getattr(self.inner, "check", None)
        if fn is not None:
            fn()

    def _switch(self, exc: BaseException) -> bool:
        alt = FALLBACK_KEYS.get(self.key, "qwen2.5-0.5b")
        if self.fell_back or not arch_unsupported(exc) or not self.installed(alt, self.models_dir):
            return False
        self.inner = self.make(self.path_of(alt, self.models_dir))
        self.fell_back = True
        if self.on_note is not None:
            self.on_note(f"This copy of the cleanup engine cannot run Qwen3.5; using {alt} instead")
        return True

    def load(self) -> Any:
        try:
            return self.inner.load()
        except Exception as exc:
            if self._switch(exc):
                return self.inner.load()
            raise

    def complete(self, system: str, user: str, max_tokens: int, timeout: float) -> str:
        try:
            out = self.inner.complete(system, user, max_tokens, timeout)
        except Exception as exc:
            if not self._switch(exc):
                raise
            out = self.inner.complete(system, user, max_tokens, timeout)
        return strip_think(str(out))
