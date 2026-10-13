"""What the person chose for the live agent, in ``<data>/liveagent/settings.json`` (readable only by them).

Kept outside the main config file on purpose (the live agent is a separate, optional feature; Settings and the
running app share it by file and the app re-reads it when it changes). API keys are NOT here (see :mod:`.keys`).
Defaults work: off, free models, no screen, no camera, no computer control until the person allows it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

VERSION = 1
TASKS = ("chat", "plan", "vision", "decide")
TASK_TITLES = {
    "chat": "Quick answers (voice)",
    "plan": "Doing things on the PC",
    "vision": "Looking at the screen",
    "decide": "Safety and routing decisions",
}
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 '-]{0,23}$")


@dataclass
class ProviderConfig:
    id: str  # unique: "groq", "openai", "custom-1"
    kind: str  # a catalog provider id (groq, openai, anthropic, ollama, custom, opencode_free...)
    model: str = ""
    base_url: str = ""  # "" = the catalog default
    enabled: bool = True
    title: str = ""
    may_keep_prompts: bool | None = None  # None = the catalog default
    has_key: bool = False  # a key is saved in the OS vault (never the key itself)


@dataclass
class LiveSettings:
    version: int = VERSION
    enabled: bool = False  # OFF by default
    name: str = "Away"  # "Hey Away, ..."
    preferred: str = ""  # "provider_id/model"; "" = automatic
    providers: list[ProviderConfig] = field(default_factory=list)
    routing: dict[str, str] = field(default_factory=dict)  # task -> "provider_id/model" ("" = automatic)
    budget_usd_per_day: float = 1.0  # paid keys stop for the day past this (0 = no paid use at all)
    allow_screen: bool = False  # may look at the screen (only private providers ever get the picture)
    allow_camera: bool = False
    allow_control: bool = False  # may use mouse/keyboard (approved on the PC each session, as for any agent)
    allow_settings: bool = True  # may change Away's own settings (undoable)
    jev: str = "auto"  # auto (free Jev when it answers) | on | off
    speak: bool = True
    barge_in: bool = True
    log_turns: bool = True
    handoff_agent: str = ""  # "" = the default agent of Settings > Agent Link
    first_token_timeout_s: float = 8.0
    turn_timeout_s: float = 60.0
    max_steps: int = 8  # computer-use steps per request

    def provider(self, pid: str) -> ProviderConfig | None:
        return next((p for p in self.providers if p.id == pid), None)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        for p in d["providers"]:
            p.pop("may_keep_prompts", None)
        return d


def path(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        from away import paths

        data_dir = paths.data_dir()
    return Path(data_dir) / "liveagent" / "settings.json"


def from_dict(d: Any) -> LiveSettings:
    s = LiveSettings()
    if not isinstance(d, dict):
        return s
    names = {f.name for f in fields(LiveSettings)}
    for k, v in d.items():
        if k not in names or k == "providers":
            continue
        cur = getattr(s, k)
        try:
            if isinstance(cur, bool):
                setattr(s, k, bool(v))
            elif isinstance(cur, int):
                setattr(s, k, int(v))
            elif isinstance(cur, float):
                setattr(s, k, float(v))
            elif isinstance(cur, dict):
                setattr(s, k, {str(a): str(b) for a, b in dict(v).items() if str(a) in TASKS})
            else:
                setattr(s, k, str(v))
        except (TypeError, ValueError):
            continue
    pnames = {f.name for f in fields(ProviderConfig)}
    for p in d.get("providers") or []:
        if isinstance(p, dict) and p.get("id") and p.get("kind"):
            try:
                s.providers.append(ProviderConfig(**{k: v for k, v in p.items() if k in pnames}))
            except TypeError:
                continue
    validate(s)
    return s


def validate(s: LiveSettings) -> None:
    """Clamp to safe values (a hand-edited file never breaks the app)."""
    s.name = s.name.strip() if NAME.match(s.name.strip() or "-") else "Away"
    s.budget_usd_per_day = max(0.0, min(1000.0, float(s.budget_usd_per_day)))
    s.first_token_timeout_s = max(2.0, min(60.0, float(s.first_token_timeout_s)))
    s.turn_timeout_s = max(10.0, min(600.0, float(s.turn_timeout_s)))
    s.max_steps = max(1, min(30, int(s.max_steps)))
    if s.jev not in ("auto", "on", "off"):
        s.jev = "auto"
    seen: set[str] = set()
    keep = []
    for p in s.providers:
        if p.id in seen:
            continue
        seen.add(p.id)
        keep.append(p)
    s.providers = keep


def load(p: Path | None = None) -> LiveSettings:
    p = p or path()
    try:
        return from_dict(json.loads(Path(p).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return LiveSettings()


KEY_LIKE = re.compile(r"(sk-|gsk_|AIza|sk-ant-)[A-Za-z0-9_-]{12,}")


def save(s: LiveSettings, p: Path | None = None) -> Path:
    p = Path(p or path())
    validate(s)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(asdict(s), indent=2)
    if KEY_LIKE.search(blob):  # belt and braces: never a key in this file
        raise ValueError("refusing to save: an API key ended up in the settings")
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(blob, encoding="utf-8")
    if os.name != "nt":
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    return p


class Watcher:
    """Re-reads the file when Settings (another process) saved it."""

    def __init__(self, p: Path | None = None) -> None:
        self.path = Path(p or path())
        self.current = load(self.path)
        self._mtime = self._stat()

    def _stat(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return -1.0

    def poll(self) -> bool:
        m = self._stat()
        if m != self._mtime:
            self._mtime = m
            self.current = load(self.path)
            return True
        return False
