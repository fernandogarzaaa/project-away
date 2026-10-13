"""Local language-model engines for smart cleanup. Each one answers ``complete(system, user, max_tokens, timeout)``.

* :class:`LlamaCppBackend`: a GGUF file run in-process by ``llama-cpp-python`` (``pip install
  'project-away[cleanup]'``); loaded once, kept warm. Uses the GPU when the installed build has CUDA/Metal.
* :class:`OllamaBackend`: a local Ollama server (``/api/chat``), e.g. ``ollama pull qwen2.5:1.5b``.
* :class:`OpenAIBackend`: any OpenAI-compatible ``/chat/completions`` endpoint (the ``AWAY_LLM_*`` variables
  that ``voice.llm_cleanup`` used before v0.1.20).

:class:`EngineError` carries a friendly message and a ``fix`` id the Settings window turns into a button.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any, Protocol


class LLMBackend(Protocol):
    def complete(self, system: str, user: str, max_tokens: int, timeout: float) -> str: ...


class EngineError(RuntimeError):
    """``fix``: download | install_engine | start_ollama | pick_model | none."""

    def __init__(self, message: str, fix: str = "none") -> None:
        super().__init__(message)
        self.fix = fix


def llama_cpp_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        return False


class LlamaCppBackend:
    def __init__(self, model_path: Path, n_ctx: int = 2048, gpu_layers: int = -1, threads: int | None = None) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.gpu_layers = gpu_layers
        self.threads = threads
        self._llm: Any = None
        self._lock = threading.Lock()  # llama.cpp contexts are not re-entrant

    def check(self) -> None:
        if not self.model_path.is_file():
            raise EngineError("Smart cleanup needs its model: download it once (it stays on this PC).", "download")
        if not llama_cpp_available():
            raise EngineError(
                "Smart cleanup's engine (llama-cpp-python) is not installed in this copy of Project Away.",
                "install_engine",
            )

    def load(self) -> Any:  # pragma: no cover - needs llama-cpp-python and a model file
        with self._lock:
            if self._llm is None:
                self.check()
                from llama_cpp import Llama

                kw: dict[str, Any] = {
                    "model_path": str(self.model_path),
                    "n_ctx": self.n_ctx,
                    "n_gpu_layers": self.gpu_layers,
                    "verbose": False,
                }
                if self.threads:
                    kw["n_threads"] = self.threads
                self._llm = Llama(**kw)
            return self._llm

    def complete(self, system: str, user: str, max_tokens: int, timeout: float) -> str:  # pragma: no cover
        llm = self.load()
        with self._lock:
            r = llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        return str(r["choices"][0]["message"]["content"] or "").strip()


def _open(req: urllib.request.Request | str, timeout: float) -> Any:
    """urlopen, but a server on this PC is reached directly even when HTTP(S)_PROXY is set."""
    import urllib.parse

    url = req.full_url if isinstance(req, urllib.request.Request) else req
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _post_json(url: str, body: dict[str, Any], timeout: float, headers: dict[str, str] | None = None) -> Any:
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    with _open(req, timeout) as r:
        return json.loads(r.read())


class OllamaBackend:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:1.5b") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def check(self, timeout: float = 1.0) -> None:
        try:
            with _open(self.base_url + "/api/tags", timeout) as r:
                tags = json.loads(r.read())
        except Exception as exc:
            raise EngineError(f"Ollama is not running at {self.base_url}.", "start_ollama") from exc
        names = {str(m.get("name", "")) for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise EngineError(
                f"Ollama does not have the model {self.model!r} (ollama pull {self.model}).", "pick_model"
            )

    def complete(self, system: str, user: str, max_tokens: int, timeout: float) -> str:
        data = _post_json(
            self.base_url + "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout,
        )
        return str(data["message"]["content"] or "").strip()


class OpenAIBackend:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("AWAY_LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("AWAY_LLM_MODEL") or "llama3.2"
        self.api_key = api_key or os.environ.get("AWAY_LLM_API_KEY")

    def check(self) -> None:
        if not self.base_url:
            raise EngineError("No server is set: set AWAY_LLM_BASE_URL (e.g. http://localhost:1234/v1).", "pick_model")

    def complete(self, system: str, user: str, max_tokens: int, timeout: float) -> str:
        self.check()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_json(
            self.base_url + "/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout,
            headers,
        )
        return str(data["choices"][0]["message"]["content"] or "").strip()


def pick_model_key(models_dir: Path, gpu: bool, choice: str = "auto") -> str:
    """``voice.cleanup.model`` -> the cleanup model key to use (or the user's .gguf path / server model name).

    ``auto``: whichever cleanup model is downloaded (the standard one first when there is a GPU, the light one
    first otherwise); nothing downloaded -> the one the Download button would fetch for this PC."""
    from away.app import optional_models as om

    if choice != "auto":
        return choice
    order = ["qwen2.5-1.5b", "qwen2.5-0.5b"] if gpu else ["qwen2.5-0.5b", "qwen2.5-1.5b"]
    for key in order:
        if om.installed(key, models_dir):
            return key
    return order[0]


def make_backend(cleanup: Any, models_dir: Path, gpu: bool) -> LLMBackend:
    """The engine ``voice.cleanup`` asks for (nothing is loaded yet)."""
    from away.app import optional_models as om

    if cleanup.backend == "ollama":
        model = cleanup.model if cleanup.model != "auto" else ("qwen2.5:1.5b" if gpu else "qwen2.5:0.5b")
        return OllamaBackend(cleanup.ollama_url, model)
    if cleanup.backend == "openai":
        return OpenAIBackend(model=None if cleanup.model == "auto" else cleanup.model)
    key = pick_model_key(models_dir, gpu, cleanup.model)
    try:
        path = om.path_of(key, models_dir)
    except KeyError:
        path = Path(key).expanduser()  # a .gguf path of your own
    return LlamaCppBackend(path, gpu_layers=cleanup.gpu_layers)
