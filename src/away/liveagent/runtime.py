"""Everything the live agent needs, built from the settings: keys, providers, the model list, router, funnel, cost
ledger, Jev, the opencode engine. Used by the running app (:mod:`.integrate`), Settings (:mod:`.gui`) and the
``python -m away.liveagent`` command.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.liveagent import catalog, keys, opencode, providers
from away.liveagent import settings as st
from away.liveagent.http import Client
from away.liveagent.jev import ChatDecider, Decider, JevClient
from away.liveagent.ledger import CostLedger
from away.liveagent.router import Breakers, Funnel, Router

JEV_KEY_ID = "jev"  # an OpenCode key for paid Jev (optional); the OpenCode Zen key is used when present
FREE_CACHE_S = 3600.0


class Runtime:
    def __init__(
        self,
        data_dir: Path | None = None,
        settings: Callable[[], st.LiveSettings] | None = None,
        keystore: keys.KeyStore | None = None,
        client: Client | None = None,
        server_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if data_dir is None:
            from away import paths

            data_dir = paths.data_dir()
        self.data_dir = Path(data_dir)
        self._settings_fn = settings or (lambda: st.load(st.path(self.data_dir)))
        self.keys = keystore or keys.KeyStore()
        self.client = client or Client()
        self._server_factory = server_factory
        self.server: Any = None
        self.clock = clock
        self.ledger = CostLedger(self.data_dir / "liveagent" / "costs.json")
        self.breakers = Breakers(clock)
        self._lock = threading.RLock()
        self._free: list[catalog.ModelInfo] = []
        self._free_at = -1e9
        self._models_dev: dict[str, Any] | None = None
        self._listed: dict[str, list[catalog.ModelInfo]] = {}
        self.errors: dict[str, str] = {}

    def settings(self) -> st.LiveSettings:
        return self._settings_fn()

    # ------------------------------------------------------------ opencode
    def opencode_server(self) -> Any:
        with self._lock:
            if self.server is None:
                if self._server_factory is not None:
                    self.server = self._server_factory()
                else:
                    self.server = opencode.make_server(self.data_dir, client=self.client)
            return self.server.ensure()

    def stop(self) -> None:
        with self._lock:
            if self.server is not None:
                with contextlib.suppress(Exception):
                    self.server.stop()

    def models_dev(self) -> dict[str, Any] | None:
        if self._models_dev is None:
            self._models_dev = {}  # one try per run: a slow or blocked models.dev never delays a turn twice
            with contextlib.suppress(Exception):
                self._models_dev = self.client.get_json(catalog.MODELS_DEV_URL, timeout=10) or {}
        return self._models_dev

    def free_models(self, refresh: bool = False) -> list[catalog.ModelInfo]:
        """Free opencode models right now (from the running engine; else the public list). Cached for an hour."""
        with self._lock:
            if self._free and not refresh and self.clock() - self._free_at < FREE_CACHE_S:
                return list(self._free)
        served: Any = None
        with contextlib.suppress(Exception):
            if self.server is not None and self.server.running():
                served = self.server.free_models_raw()
        zen: list[str] | None = None
        if not served:
            with contextlib.suppress(Exception):
                data = self.client.get_json(catalog.ZEN_MODELS_URL, timeout=15)
                zen = [str(m["id"]) for m in (data or {}).get("data", []) if m.get("id")]
        found = catalog.opencode_free_models(zen, self.models_dev(), served)
        with self._lock:
            if found:
                self._free, self._free_at = found, self.clock()
            return list(self._free or found)

    # ------------------------------------------------------------ providers + models
    def build_providers(self, s: st.LiveSettings) -> dict[str, providers.Provider]:
        out: dict[str, providers.Provider] = {}
        for cfg in s.providers:
            if not cfg.enabled:
                continue
            try:
                key = self.keys.get(cfg.id) if catalog.spec(cfg.kind).kind != "opencode" else ""
                out[cfg.id] = providers.build(cfg, key, self.client, self.opencode_server, self.free_models)
                self.errors.pop(cfg.id, None)
            except Exception as exc:
                self.errors[cfg.id] = str(exc)
        return out

    def models(self, s: st.LiveSettings, provs: dict[str, providers.Provider]) -> list[catalog.ModelInfo]:
        out: list[catalog.ModelInfo] = []
        md = self.models_dev()
        for cfg in s.providers:
            p = provs.get(cfg.id)
            if p is None:
                continue
            sp = catalog.spec(cfg.kind)
            if sp.kind == "opencode":
                out.extend(m for m in self.free_models())
                continue
            if sp.local and not cfg.model:
                if cfg.id not in self._listed:
                    with contextlib.suppress(Exception):
                        self._listed[cfg.id] = p.list_models()
                out.extend(self._listed.get(cfg.id, [])[:3])
                continue
            mid = cfg.model or sp.default_model
            if not mid:
                continue
            keep = sp.may_keep_prompts if cfg.may_keep_prompts is None else bool(cfg.may_keep_prompts)
            info = catalog.guess(
                cfg.id,
                mid,
                free=sp.local or mid.endswith(":free"),
                local=sp.local,
                may_keep_prompts=keep or mid.endswith(":free"),
            )
            info.price_in, info.price_out = catalog.price_for(md, sp.id, mid)
            out.append(info)
        return out

    def funnel(self, s: st.LiveSettings | None = None) -> Funnel:
        s = s or self.settings()
        provs = self.build_providers(s)
        router = Router(provs, self.models(s, provs), s.preferred, s.routing, s.allow_screen)
        self.ledger.limit_usd = float(s.budget_usd_per_day)
        return Funnel(router, self.ledger, self.breakers, s.first_token_timeout_s, s.turn_timeout_s, clock=self.clock)

    def decider(self, s: st.LiveSettings | None = None) -> Decider:
        s = s or self.settings()
        jev = None
        if s.jev != "off":
            key = ""
            with contextlib.suppress(Exception):
                key = self.keys.get(JEV_KEY_ID) or self.keys.get("opencode_zen")
            jev = JevClient(key, self.client)

        def complete(system: str, user: str) -> str:
            f = self.funnel(s)
            return f.complete(
                "decide",
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=200,
                json_mode=True,
            )

        return Decider(jev, ChatDecider(complete), self.clock)

    # ------------------------------------------------------------ status
    def current_model(self, s: st.LiveSettings | None = None) -> str:
        """What answers quick questions right now, in plain words ("Big Pickle")."""
        s = s or self.settings()
        try:
            f = self.funnel(s)
            for c in f.router.candidates("chat"):
                if not self.breakers.is_open(c.info.provider):
                    return c.info.label()
        except Exception:
            pass
        return ""

    def status(self, s: st.LiveSettings | None = None) -> dict[str, Any]:
        """For the tray, Settings and Agent Link (never any key)."""
        s = s or self.settings()
        out: dict[str, Any] = {
            "enabled": s.enabled,
            "name": s.name,
            "wake_phrase": f"Hey {s.name}",
            "preferred": s.preferred,
            "providers": [
                {
                    "id": p.id,
                    "kind": p.kind,
                    "title": p.title or catalog.spec(p.kind).title,
                    "model": p.model,
                    "enabled": p.enabled,
                    "has_key": bool(p.has_key),
                    "error": self.errors.get(p.id, ""),
                }
                for p in s.providers
            ],
            "allow": {
                "screen": s.allow_screen,
                "camera": s.allow_camera,
                "control": s.allow_control,
                "settings": s.allow_settings,
            },
            "budget": self.ledger.summary(),
            "breakers": self.breakers.public(),
            "engine": engine_status(self.data_dir),
        }
        if s.enabled:
            try:
                f = self.funnel(s)
                out["routing"] = f.router.table()
                out["model"] = next((c.info.label() for c in f.router.candidates("chat")), "")
            except Exception as exc:
                out["error"] = str(exc)
        return out


def engine_status(data_dir: Path | None = None) -> dict[str, Any]:
    eng = opencode.installed_engine(data_dir)
    if eng is not None:
        return {"installed": True, "where": "away", "version": eng.get("version"), "license": "MIT"}
    cli = opencode.find_cli()
    if cli is not None:
        return {"installed": True, "where": "yours", "command": cli[-1]}
    return {"installed": False}


# ------------------------------------------------------------------------------------------------ one-click setups
def add_provider(
    s: st.LiveSettings, kind: str, model: str = "", base_url: str = "", has_key: bool = False, pid: str = ""
) -> st.ProviderConfig:
    sp = catalog.spec(kind)
    pid = pid or (kind if not s.provider(kind) or kind != "custom" else f"custom-{len(s.providers) + 1}")
    cfg = s.provider(pid)
    if cfg is None:
        cfg = st.ProviderConfig(pid, sp.id, model or sp.default_model, base_url, True, sp.title, None, has_key)
        s.providers.append(cfg)
    else:
        cfg.model = model or cfg.model or sp.default_model
        cfg.base_url = base_url or cfg.base_url
        cfg.enabled = True
        cfg.has_key = has_key or cfg.has_key
    return cfg


def use_free_models(
    rt: Runtime,
    s: st.LiveSettings,
    progress: Callable[[str, int, int], None] | None = None,
    installer: Callable[..., dict[str, Any]] = opencode.install,
) -> str:
    """The "Use free models" button: get opencode (download it when missing), check a free model answers, make it
    the preferred model, turn the live agent on. Returns a plain line; raises :class:`opencode.EngineError` with a
    plain reason (the button then offers "Use a free key instead")."""

    def tell(msg: str, done: int = 0, total: int = 0) -> None:
        if progress is not None:
            with contextlib.suppress(Exception):
                progress(msg, done, total)

    try:
        _cmd, where = opencode.resolve(rt.data_dir)
        tell(f"Using {where}")
    except opencode.EngineError:
        tell("Downloading opencode (free, MIT license)...")
        installer(rt.data_dir, progress=lambda d, t: tell("Downloading opencode", d, t))
        tell("Downloaded and checked")
    tell("Starting the free models...")
    rt.server = None
    server = rt.opencode_server()
    free = rt.free_models(refresh=True)
    if not free:
        raise opencode.EngineError("opencode is running but lists no free models right now")
    from away.liveagent.textstream import strip_reasoning

    tell("Checking a free model answers...")
    problems: list[str] = []
    for m in sorted(free, key=lambda m: (not m.fast, m.id != "big-pickle"))[:6]:
        t0 = time.monotonic()
        try:
            text = strip_reasoning("".join(server.chat_stream(m.id, "Reply with the single word: ready", timeout=30)))
        except Exception as exc:
            problems.append(f"{m.label()}: {str(exc)[:80]}")
            continue
        if not text:
            problems.append(f"{m.label()}: empty answer")
            continue
        add_provider(s, "opencode_free", m.id)
        if not s.preferred or s.preferred.startswith("opencode_free/"):
            s.preferred = m.ref
        s.enabled = True
        return f"Free models are on ({m.label()}, answered in {time.monotonic() - t0:.1f} s)"
    raise opencode.EngineError("no free model answered right now (" + "; ".join(problems[:3]) + ")")


def fallback_options(data_dir: Path | None = None, client: Client | None = None) -> list[dict[str, str]]:
    """When opencode cannot be downloaded or started: free keys (Gemini, Groq) and Ollama if it runs here."""
    out = [
        {"kind": k, "title": catalog.spec(k).title, "key_url": catalog.spec(k).key_url, "note": catalog.spec(k).note}
        for k in catalog.FREE_KEY_PROVIDERS
    ]
    try:
        c = client or Client()
        data = c.get_json("http://127.0.0.1:11434/api/tags", timeout=1.5)
        names = [str(m.get("name")) for m in (data or {}).get("models", []) if m.get("name")]
        if names:
            out.append(
                {"kind": "ollama", "title": "Ollama (on this PC)", "key_url": "", "note": f"{len(names)} model(s)"}
            )
    except Exception:
        pass
    return out
