"""The model switcher (v0.1.23): one place that knows every model Project Away can use, per job ("slot").

Slots: ``hands`` (hand tracking), ``face`` (face / attention), ``gaze``, ``stt`` (speech-to-text), ``cleanup``
(smart cleanup) and ``voice`` (how agents sound). Each model is a manifest entry::

    {"id": "whisper-small-cpu", "slot": "stt", "title": "...", "description": "plain words", "license": "MIT",
     "url": "https://...", "sha256": "...", "size": 484000000, "runtime": "whisper|onnx|gguf|mediapipe|...",
     "runs_on": "cpu|gpu|either", "cost": {"cpu_pct": 35, "vram_mb": 0}, "set": {"voice.asr_model": "small"}}

``set`` is what choosing the model writes to the settings, so a new model that an existing engine can run is added
by a manifest entry alone, no code. Sources, later ones winning per id:

1. ``away/data/models.json`` (shipped with the app);
2. the online catalog ``models/catalog.json`` on the repository's main branch (checked at most once a day, or
   with "Check now"); it is used only when its SHA-256 matches ``models/catalog.sha256`` next to it, every
   entry passes :func:`validate`, and every download address is on an allowed host;
3. ``<data>/models/local.json``: your own ONNX / GGUF files (``"path": "C:/models/my.gguf"``).

Restrictively licensed models (``permissive: false`` or a non-permissive license) are hidden unless you turn on
"Show models with restrictive licenses". "Try it" switches to a model, runs a short benchmark, and switches back
automatically when the new one fails or is slower.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SLOTS = ("hands", "face", "gaze", "stt", "cleanup", "voice")
SLOT_TITLES = {
    "hands": "Hand tracking",
    "face": "Face / attention",
    "gaze": "Eye tracking",
    "stt": "Speech to text",
    "cleanup": "Smart cleanup",
    "voice": "Voice (agents speaking)",
}
# the On/Off switch of each optional slot (hand tracking is always on)
TOGGLES = {"face": "face.enabled", "gaze": "gaze.enabled", "cleanup": "voice.cleanup.mode", "voice": "speech.enabled"}
MODES = ("light", "balanced", "best")
MODE_TITLES = {"light": "Light", "balanced": "Balanced", "best": "Best"}
PERMISSIVE = ("mit", "apache-2.0", "apache 2.0", "bsd", "bsd-2-clause", "bsd-3-clause", "isc", "zlib", "cc0", "mpl-2.0")
ALLOWED_HOSTS = (
    "github.com",
    "objects.githubusercontent.com",
    "raw.githubusercontent.com",
    "huggingface.co",
    "files.pythonhosted.org",
    "storage.googleapis.com",
)
CATALOG_URL = "https://raw.githubusercontent.com/fernandogarzaaa/project-away/main/models/catalog.json"
CATALOG_SHA_URL = "https://raw.githubusercontent.com/fernandogarzaaa/project-away/main/models/catalog.sha256"
CHECK_EVERY_S = 86400.0
RUNTIMES = ("mediapipe", "onnx", "gguf", "whisper", "rules", "ollama", "system")


@dataclass
class Model:
    id: str
    slot: str
    title: str
    description: str = ""
    license: str = ""
    runtime: str = ""
    runs_on: str = "cpu"
    size: int = 0
    url: str = ""
    sha256: str = ""
    cost: dict[str, Any] = field(default_factory=dict)
    set: dict[str, Any] = field(default_factory=dict)
    bundled: bool = False
    download: str = ""
    path: str = ""  # a local file the user added
    permissive: bool | None = None
    source: str = "built-in"

    @property
    def is_permissive(self) -> bool:
        if self.permissive is not None:
            return bool(self.permissive)
        low = self.license.strip().lower()
        return any(low.startswith(p) for p in PERMISSIVE) or low in ("system", "windows")

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "slot": self.slot,
            "title": self.title,
            "description": self.description,
            "license": self.license,
            "permissive": self.is_permissive,
            "runs_on": {"cpu": "CPU", "gpu": "GPU", "either": "CPU or GPU"}.get(self.runs_on, self.runs_on),
            "size_mb": round(self.size / 1e6) if self.size else 0,
            "cost": dict(self.cost),
            "source": self.source,
        }


def validate(item: Any) -> Model:
    """A manifest entry -> :class:`Model`; ValueError with a plain reason when it is not acceptable."""
    if not isinstance(item, dict):
        raise ValueError("a model entry must be an object")
    mid, slot = str(item.get("id") or ""), str(item.get("slot") or "")
    if not mid or len(mid) > 80 or not all(c.isalnum() or c in "-_." for c in mid):
        raise ValueError(f"bad model id {mid!r}")
    if slot not in SLOTS:
        raise ValueError(f"{mid}: unknown slot {slot!r}")
    sets = item.get("set") or {}
    if not isinstance(sets, dict) or not sets:
        raise ValueError(f"{mid}: 'set' must name the settings this model uses")
    for k in sets:
        if not str(k).startswith(("hands.", "face.", "gaze.", "voice.", "speech.")) or "agent_link" in str(k):
            raise ValueError(f"{mid}: a model may not change {k}")
    url = str(item.get("url") or "")
    if url:
        host = (urlparse(url).hostname or "").lower()
        if urlparse(url).scheme != "https" or not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
            raise ValueError(f"{mid}: downloads must come from an allowed https host, not {host or url!r}")
        sha = str(item.get("sha256") or "")
        if len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha.lower()):
            raise ValueError(f"{mid}: a download needs its sha256")
    runtime = str(item.get("runtime") or "")
    if runtime and runtime not in RUNTIMES:
        raise ValueError(f"{mid}: unknown runtime {runtime!r}")
    known = set(Model.__dataclass_fields__)
    m = Model(**{k: v for k, v in item.items() if k in known and k != "source"})
    m.size = int(m.size or 0)
    return m


# ------------------------------------------------------------------------------------------------ sources
def builtin_manifest() -> dict[str, Any]:
    p = Path(__file__).resolve().parent.parent / "data" / "models.json"
    return dict(json.loads(p.read_text(encoding="utf-8")))


def models_dir(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        from away import paths

        data_dir = Path(paths.data_dir())
    return Path(data_dir) / "models"


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Registry:
    def __init__(self, data_dir: Path | None = None, builtin: dict[str, Any] | None = None) -> None:
        self.data_dir = data_dir
        self.builtin = builtin if builtin is not None else builtin_manifest()
        self.errors: list[str] = []
        self.models: dict[str, Model] = {}
        self.modes: dict[str, dict[str, str]] = dict(self.builtin.get("modes") or {})
        self.reload()

    @property
    def folder(self) -> Path:
        return models_dir(self.data_dir)

    def _add(self, items: Any, source: str) -> None:
        for item in items or []:
            try:
                m = validate(item)
            except ValueError as exc:
                self.errors.append(f"{source}: {exc}")
                continue
            m.source = source
            self.models[m.id] = m

    def reload(self) -> None:
        self.models, self.errors = {}, []
        self._add(self.builtin.get("models"), "built-in")
        cat = _read_json(self.folder / "catalog.json")
        if isinstance(cat, dict):
            self._add(cat.get("models"), "catalog")
            for k, v in (cat.get("modes") or {}).items():
                if k in MODES and isinstance(v, dict):
                    self.modes[k] = {s: str(i) for s, i in v.items() if s in SLOTS}
        local = _read_json(self.folder / "local.json")
        if isinstance(local, dict):
            self._add(local.get("models"), "your file")

    # ----------------------------------------------------------------- reads
    def for_slot(self, slot: str, show_restrictive: bool = False) -> list[Model]:
        return [m for m in self.models.values() if m.slot == slot and (show_restrictive or m.is_permissive)]

    def get(self, mid: str) -> Model:
        if mid not in self.models:
            raise KeyError(f"there is no model {mid!r} (models list shows them)")
        return self.models[mid]

    def current(self, cfg: Any, slot: str) -> Model | None:
        """The model whose settings match the configuration now (the most specific match)."""
        best, score = None, -1
        for m in self.for_slot(slot, True):
            if all(_get(cfg, k) == v for k, v in m.set.items()) and len(m.set) > score:
                best, score = m, len(m.set)
        return best

    def enabled(self, cfg: Any, slot: str) -> bool:
        key = TOGGLES.get(slot)
        if key is None:
            return True
        v = _get(cfg, key)
        return v != "off" if key == "voice.cleanup.mode" else bool(v)

    # ----------------------------------------------------------------- changes (in place; return the undo)
    def switch(self, cfg: Any, slot: str, mid: str) -> dict[str, Any]:
        m = self.get(mid)
        if m.slot != slot:
            raise ValueError(f"{m.title} is a {SLOT_TITLES[m.slot]} model, not {SLOT_TITLES.get(slot, slot)}")
        before = {k: copy.deepcopy(_get(cfg, k)) for k in m.set}
        for k, v in m.set.items():
            _set(cfg, k, v)
        if m.path and m.runtime == "gguf":
            before.setdefault("voice.cleanup.model", _get(cfg, "voice.cleanup.model"))
            _set(cfg, "voice.cleanup.model", m.path)
        return before

    def set_enabled(self, cfg: Any, slot: str, on: bool) -> dict[str, Any]:
        key = TOGGLES.get(slot)
        if key is None:
            raise ValueError(f"{SLOT_TITLES.get(slot, slot)} cannot be turned off")
        before = {key: _get(cfg, key)}
        if key == "voice.cleanup.mode":
            _set(cfg, key, "rules" if on else "off")
        else:
            _set(cfg, key, bool(on))
        return before

    def apply_mode(self, cfg: Any, mode: str) -> dict[str, Any]:
        """Performance mode (Light / Balanced / Best): one model per slot. Returns the undo."""
        if mode not in self.modes:
            raise ValueError(f"mode must be one of {', '.join(self.modes)}")
        before: dict[str, Any] = {}
        for slot, mid in self.modes[mode].items():
            if mid in self.models:
                for k, v in self.switch(cfg, slot, mid).items():
                    before.setdefault(k, v)
        return before

    @staticmethod
    def restore(cfg: Any, before: dict[str, Any]) -> None:
        for k, v in before.items():
            _set(cfg, k, v)

    def rows(self, cfg: Any, show_restrictive: bool = False, seen: set[str] | None = None) -> list[dict[str, Any]]:
        """One row per slot for Settings > Models and the agents' ``models list``."""
        out = []
        for slot in SLOTS:
            cur = self.current(cfg, slot)
            out.append(
                {
                    "slot": slot,
                    "title": SLOT_TITLES[slot],
                    "current": cur.id if cur else None,
                    "current_title": cur.title if cur else "custom",
                    "runs_on": cur.public()["runs_on"] if cur else "",
                    "optional": slot in TOGGLES,
                    "on": self.enabled(cfg, slot),
                    "choices": [
                        {**m.public(), "new": seen is not None and m.id not in seen}
                        for m in self.for_slot(slot, show_restrictive)
                    ],
                }
            )
        return out


def _get(cfg: Any, path: str) -> Any:
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set(cfg: Any, path: str, value: Any) -> None:
    *head, last = path.split(".")
    obj = cfg
    for part in head:
        obj = getattr(obj, part)
    setattr(obj, last, value)


# ------------------------------------------------------------------------------------------------ catalog
def check_catalog(
    data_dir: Path | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.time,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch the online catalog (at most daily unless ``force``). ``{"checked", "new": [ids], "error"}``."""
    folder = models_dir(data_dir)
    stamp = folder / "catalog.checked"
    try:
        last = float(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        last = 0.0
    if not force and clock() - last < CHECK_EVERY_S:
        return {"checked": False, "new": [], "error": ""}
    try:
        body = _fetch(opener, CATALOG_URL)
        want = _fetch(opener, CATALOG_SHA_URL).decode("utf-8", "replace").split()[0].strip().lower()
    except Exception as exc:
        return {"checked": False, "new": [], "error": f"could not reach the model catalog ({exc})"}
    if hashlib.sha256(body).hexdigest() != want:
        return {"checked": True, "new": [], "error": "the model catalog did not match its checksum; it was ignored"}
    try:
        data = json.loads(body)
        items = data.get("models") or []
        for item in items:
            validate(item)
    except Exception as exc:
        return {"checked": True, "new": [], "error": f"the model catalog has a bad entry ({exc}); it was ignored"}
    old = _read_json(folder / "catalog.json") or {}
    old_ids = {str(m.get("id")) for m in (old.get("models") or []) if isinstance(m, dict)}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "catalog.json").write_bytes(body)
    stamp.write_text(str(clock()), encoding="utf-8")
    new = [str(m["id"]) for m in items if str(m.get("id")) not in old_ids]
    return {"checked": True, "new": new, "error": ""}


def _fetch(opener: Callable[..., Any], url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "project-away"})
    with opener(req, timeout=20) as resp:
        return bytes(resp.read(2_000_000))


def seen_ids(data_dir: Path | None = None) -> set[str]:
    v = _read_json(models_dir(data_dir) / "seen.json")
    return {str(x) for x in v} if isinstance(v, list) else set()


def mark_seen(ids: Any, data_dir: Path | None = None) -> None:
    folder = models_dir(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "seen.json").write_text(json.dumps(sorted(seen_ids(data_dir) | {str(i) for i in ids})), "utf-8")


def new_model_hint(reg: Registry, data_dir: Path | None = None) -> str:
    """ "A faster speech model is available - try it (Settings > Models)" for the first unseen permissive model."""
    seen = seen_ids(data_dir)
    for m in reg.models.values():
        if m.source == "catalog" and m.is_permissive and m.id not in seen:
            what = SLOT_TITLES[m.slot].lower()
            return f"A new {what} model is available: {m.title} - try it in Settings > Models"
    return ""


# ------------------------------------------------------------------------------------------------ try it
@dataclass
class TryResult:
    kept: bool
    message: str
    before: dict[str, Any]
    score_before: float | None = None
    score_after: float | None = None


def try_model(
    reg: Registry,
    cfg: Any,
    slot: str,
    mid: str,
    bench: Callable[[Any], float],
    tolerance: float = 0.1,
) -> TryResult:
    """Switch ``slot`` to ``mid``, benchmark (``bench(cfg)``: a score, higher is better, e.g. fps or 1/latency),
    and switch back when it fails or scores more than ``tolerance`` worse than the model before it."""
    try:
        old = bench(cfg)
    except Exception:
        old = None
    before = reg.switch(cfg, slot, mid)
    title = reg.get(mid).title
    try:
        new = bench(cfg)
    except Exception as exc:
        reg.restore(cfg, before)
        return TryResult(False, f"{title} did not work here ({str(exc)[:100]}); switched back", before, old, None)
    if old is not None and new < old * (1.0 - tolerance):
        reg.restore(cfg, before)
        return TryResult(False, f"{title} was slower on this PC; switched back", before, old, new)
    return TryResult(True, f"Now using {title}", before, old, new)
