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
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SLOTS = ("hands", "face", "gaze", "stt", "cleanup", "voice", "vad", "wake")
SLOT_TITLES = {
    "hands": "Hand tracking",
    "face": "Face / attention",
    "gaze": "Eye tracking",
    "stt": "Speech to text",
    "cleanup": "Smart cleanup",
    "voice": "Voice (agents speaking)",
    "vad": "Speech detection",  # v0.1.24
    "wake": "Wake word",  # v0.1.24
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
    "download.openmmlab.com",  # v0.1.24: RTMPose hands (OpenMMLab's official model downloads)
)
CATALOG_URL = "https://raw.githubusercontent.com/fernandogarzaaa/project-away/main/models/catalog.json"
CATALOG_SHA_URL = "https://raw.githubusercontent.com/fernandogarzaaa/project-away/main/models/catalog.sha256"
CHECK_EVERY_S = 86400.0
RUNTIMES = ("mediapipe", "onnx", "gguf", "whisper", "rules", "ollama", "system")
# v0.1.24 (models_next only): engines fetched at runtime by away.app.engines, or a pip extra (torch)
NEXT_RUNTIMES = ("moonshine", "nemo-speech", "livekit-wakeword", "torch")
SET_PREFIXES = ("hands.", "face.", "gaze.", "voice.", "speech.")
NEXT_SET_PREFIXES = (*SET_PREFIXES, "models_next.")
# fields a models_next entry may add to a model that already exists (its settings and title stay as they are)
INFO_FIELDS = ("notes", "version", "released", "sha256_note")
# v0.1.35: where a model runs, in this OS's words. A Mac has no NVIDIA card: its "GPU" models run on the Apple GPU /
# Neural Engine (Core ML) when that is faster, and their memory comes out of the unified memory shared with the CPU.
RUNS_ON_TITLES = {"cpu": "CPU", "gpu": "GPU", "either": "CPU or GPU"}
MAC_RUNS_ON_TITLES = {"cpu": "CPU", "gpu": "Apple GPU / Neural Engine", "either": "CPU or Apple GPU / Neural Engine"}
# only the tracking models run through Core ML; a "gpu" speech / voice model needs CUDA and runs on a Mac's CPU
COREML_SLOTS = frozenset({"hands", "face", "gaze"})
MAC_CUDA_RUNS_ON_TITLES = {"cpu": "CPU", "gpu": "NVIDIA GPU (CPU on a Mac)", "either": "CPU"}
# unified memory a Mac keeps for macOS, the camera / MediaPipe / Whisper and your own apps before the models' share:
# a quarter of it, at least 4 GB (an 8 GB Mac can spare ~4 GB, a 24 GB one ~18 GB)
UNIFIED_HEADROOM_MIN_MB = 4096
UNIFIED_HEADROOM_SHARE = 0.25
VRAM_HEADROOM_MB = 1024  # the same headroom the benchmark keeps on a graphics card (model_bench)


def is_mac(platform: str | None = None) -> bool:
    return (platform or sys.platform) == "darwin"


def runs_on_title(runs_on: str, platform: str | None = None, slot: str | None = None) -> str:
    """``cpu|gpu|either`` in plain words; v0.1.35: on macOS "Apple GPU / Neural Engine" for the tracking slots
    (Core ML), and the CPU for the others (``slot`` None: a tracking model)."""
    if not is_mac(platform):
        titles = RUNS_ON_TITLES
    else:
        titles = MAC_RUNS_ON_TITLES if slot is None or slot in COREML_SLOTS else MAC_CUDA_RUNS_ON_TITLES
    return titles.get(runs_on, runs_on)


def memory_budget(hw: Any = None, platform: str | None = None) -> dict[str, Any]:
    """How much memory the models' GPU share may use here: ``{"kind": "unified"|"vram", "total_mb", "budget_mb"}``
    (``None`` when unknown). v0.1.35: on macOS it is the unified memory (``hw.memsize``) minus
    :data:`UNIFIED_HEADROOM_SHARE` of it (at least :data:`UNIFIED_HEADROOM_MIN_MB`), not VRAM; elsewhere the NVIDIA
    card's VRAM minus :data:`VRAM_HEADROOM_MB`."""
    if is_mac(platform):
        total = getattr(hw, "ram_mb", None) if hw is not None else None
        if not total:
            return {"kind": "unified", "total_mb": None, "budget_mb": None}
        keep = max(UNIFIED_HEADROOM_MIN_MB, int(total * UNIFIED_HEADROOM_SHARE))
        return {"kind": "unified", "total_mb": int(total), "budget_mb": max(0, int(total) - keep)}
    gpu = getattr(hw, "nvidia", None) if hw is not None else None
    total = getattr(gpu, "vram_mb", None) if gpu is not None else None
    budget = max(0, int(total) - VRAM_HEADROOM_MB) if total else None
    return {"kind": "vram", "total_mb": int(total) if total else None, "budget_mb": budget}


def memory_fits(need_mb: int, hw: Any = None, platform: str | None = None) -> bool | None:
    """Does ``need_mb`` of model memory fit :func:`memory_budget`? ``None`` when the memory is unknown."""
    b = memory_budget(hw, platform)
    return None if b["budget_mb"] is None else int(need_mb) <= int(b["budget_mb"])


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
    # v0.1.24 (models_next)
    files: list[dict[str, Any]] = field(default_factory=list)  # a pinned multi-file bundle: path, url, sha256, size
    experimental: bool = False
    component: bool = False  # part of another model (downloaded with it), not a choice of its own
    engine: str = ""  # away.app.engines key the model runs in
    requires: list[str] = field(default_factory=list)  # e.g. wake_model (a trained file of yours)
    notes: str = ""
    version: str = ""
    released: str = ""
    from_next: bool = False  # came from the catalog's models_next (offered by the Newer models panel, not the hint)

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
            "runs_on": runs_on_title(self.runs_on, slot=self.slot),  # v0.1.35: in the Mac's words on macOS
            "size_mb": round(self.size / 1e6) if self.size else 0,
            "cost": dict(self.cost),
            "source": self.source,
            "experimental": self.experimental,
            "notes": self.notes,
        }


def validate(item: Any, next_entry: bool = False) -> Model:
    """A manifest entry -> :class:`Model`; ValueError with a plain reason when it is not acceptable.

    ``next_entry``: a v0.1.24 ``models_next`` entry, which may also use the new runtimes, a ``files`` bundle and
    ``models_next.*`` settings."""
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
    prefixes = NEXT_SET_PREFIXES if next_entry else SET_PREFIXES
    for k in sets:
        if not str(k).startswith(prefixes) or "agent_link" in str(k):
            raise ValueError(f"{mid}: a model may not change {k}")
    url = str(item.get("url") or "")
    if url:
        _check_download(mid, url, item.get("sha256"))
    files = item.get("files") or []
    if files and not next_entry:
        raise ValueError(f"{mid}: file bundles need Project Away 0.1.24 (models_next)")
    if not isinstance(files, list):
        raise ValueError(f"{mid}: 'files' must be a list")
    for f in files:
        if not isinstance(f, dict) or not str(f.get("path") or ""):
            raise ValueError(f"{mid}: every file needs a path")
        rel = str(f["path"])
        if rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/").split("/") or ":" in rel:
            raise ValueError(f"{mid}: unsafe file path {rel!r}")
        _check_download(mid, str(f.get("url") or ""), f.get("sha256"))
    runtime = str(item.get("runtime") or "")
    allowed = (*RUNTIMES, *NEXT_RUNTIMES) if next_entry else RUNTIMES
    if runtime and runtime not in allowed:
        raise ValueError(f"{mid}: unknown runtime {runtime!r}")
    known = set(Model.__dataclass_fields__)
    m = Model(**{k: v for k, v in item.items() if k in known and k not in ("source", "from_next")})
    m.size = int(m.size or 0)
    return m


def _check_download(mid: str, url: str, sha: Any) -> None:
    host = (urlparse(url).hostname or "").lower()
    if urlparse(url).scheme != "https" or not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise ValueError(f"{mid}: downloads must come from an allowed https host, not {host or url!r}")
    sha = str(sha or "")
    if len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha.lower()):
        raise ValueError(f"{mid}: a download needs its sha256")


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
        self.classic_modes: dict[str, dict[str, str]] = dict(self.modes)
        # v0.1.24: the 2026-10 presets (catalog "presets_next"); used only when models_next.presets is "next"
        self.presets_next: dict[str, Any] = {}
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
        if not isinstance(cat, dict) or "models_next" not in cat:
            cat = _merge_next(cat, shipped_catalog())  # until the online catalog has them: the copy in the app
        if isinstance(cat, dict):
            self._add(cat.get("models"), "catalog")
            for k, v in (cat.get("modes") or {}).items():
                if k in MODES and isinstance(v, dict):
                    self.modes[k] = {s: str(i) for s, i in v.items() if s in SLOTS}
            self.classic_modes = dict(self.modes)
            self._add_next(cat.get("models_next"))
            pn = cat.get("presets_next")
            self.presets_next = dict(pn) if isinstance(pn, dict) else {}
        local = _read_json(self.folder / "local.json")
        if isinstance(local, dict):
            self._add(local.get("models"), "your file")

    def _add_next(self, items: Any) -> None:
        """v0.1.24 ``models_next``: new models are added; for a model that already exists only its notes /
        version are filled in (its settings, title and download never change)."""
        for item in items or []:
            try:
                m = validate(item, next_entry=True)
            except ValueError as exc:
                self.errors.append(f"catalog (new models): {exc}")
                continue
            old = self.models.get(m.id)
            if old is not None:
                for f in INFO_FIELDS:
                    if getattr(m, f, "") and hasattr(old, f):
                        setattr(old, f, getattr(m, f))
                continue
            m.source = "catalog"
            m.from_next = True
            self.models[m.id] = m

    # ----------------------------------------------------------------- reads
    def for_slot(self, slot: str, show_restrictive: bool = False) -> list[Model]:
        return [
            m
            for m in self.models.values()
            if m.slot == slot and not m.component and (show_restrictive or m.is_permissive)
        ]

    def get(self, mid: str) -> Model:
        if mid not in self.models:
            raise KeyError(f"there is no model {mid!r} (models list shows them)")
        return self.models[mid]

    def current(self, cfg: Any, slot: str) -> Model | None:
        """The model whose settings match the configuration now (the most specific match)."""
        best, score = None, -1
        for m in self.for_slot(slot, True):
            if not all(_has(cfg, k) for k in m.set):
                continue
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

    def apply_mode(self, cfg: Any, mode: str, ready: Callable[[Model], bool] | None = None) -> dict[str, Any]:
        """Performance mode (Light / Balanced / Best): one model per slot. Returns the undo.

        v0.1.24: with ``models_next.presets: next`` (after a passing "Test my PC", or chosen by you) the modes pick
        the 2026-10 models of ``presets_next``; a model that is not ``ready`` yet (not downloaded, no wake-word
        model of yours) keeps that slot as it is. Otherwise the v0.1.23 modes are used unchanged."""
        if use_next(cfg) and self.presets_next.get("modes"):
            return self.apply_next_preset(cfg, mode, ready)
        if mode not in self.modes:
            raise ValueError(f"mode must be one of {', '.join(self.modes)}")
        before: dict[str, Any] = {}
        for slot, mid in self.modes[mode].items():
            if mid in self.models:
                for k, v in self.switch(cfg, slot, mid).items():
                    before.setdefault(k, v)
        return before

    # ----------------------------------------------------------------- v0.1.24 presets
    def next_modes(self) -> dict[str, dict[str, str]]:
        modes = self.presets_next.get("modes") or {}
        return {k: dict(v) for k, v in modes.items() if k in MODES and isinstance(v, dict)}

    def preset_plan(self, mode: str, ready: Callable[[Model], bool] | None = None) -> list[dict[str, Any]]:
        """One row per slot of a 2026-10 preset: the model, whether it is ready here, its estimated VRAM."""
        modes = self.next_modes()
        if mode not in modes:
            raise ValueError(f"mode must be one of {', '.join(modes) or 'light, balanced, best'}")
        rows = []
        for slot, mid in modes[mode].items():
            if mid == "off":
                rows.append({"slot": slot, "model": None, "title": "Off", "ready": True, "vram_mb": 0})
                continue
            m = self.models.get(mid)
            ok = m is not None and (ready(m) if ready is not None else True)
            rows.append(
                {
                    "slot": slot,
                    "model": mid,
                    "title": m.title if m else mid,
                    "ready": bool(ok),
                    "experimental": bool(m and m.experimental),
                    "vram_mb": int((m.cost or {}).get("vram_mb", 0)) if m else 0,
                    "size_mb": round(m.size / 1e6) if m and m.size else 0,
                }
            )
        return rows

    def preset_vram_mb(self, mode: str) -> int:
        """Estimated graphics memory of a 2026-10 preset (sum of its models' estimates; on a Mac the same estimate is
        taken out of the unified memory, see :meth:`preset_fits`)."""
        return sum(int(r["vram_mb"]) for r in self.preset_plan(mode))

    def preset_fits(self, mode: str, hw: Any = None, platform: str | None = None) -> bool | None:
        """v0.1.35: whether a preset's GPU memory fits this machine (:func:`memory_fits`: unified memory with
        headroom on macOS, VRAM elsewhere); ``None`` when the memory is unknown."""
        return memory_fits(self.preset_vram_mb(mode), hw, platform)

    def apply_next_preset(self, cfg: Any, mode: str, ready: Callable[[Model], bool] | None = None) -> dict[str, Any]:
        before: dict[str, Any] = {}
        for row in self.preset_plan(mode, ready):
            slot = str(row["slot"])
            if row["model"] is None:
                if slot in TOGGLES:
                    for k, v in self.set_enabled(cfg, slot, False).items():
                        before.setdefault(k, v)
                continue
            if not row["ready"]:
                continue
            for k, v in self.switch(cfg, slot, str(row["model"])).items():
                before.setdefault(k, v)
        for k, v in dict((self.presets_next.get("settings") or {}).get(mode) or {}).items():
            if str(k).startswith("models_next.") and _has(cfg, str(k)):
                before.setdefault(str(k), copy.deepcopy(_get(cfg, str(k))))
                _set(cfg, str(k), v)
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


def use_next(cfg: Any) -> bool:
    """Light / Balanced / Best pick the 2026-10 models (``models_next.presets: next``)."""
    mn = getattr(cfg, "models_next", None)
    return bool(mn is not None and getattr(mn, "presets", "classic") == "next")


def _has(cfg: Any, path: str) -> bool:
    try:
        _get(cfg, path)
    except AttributeError:
        return False
    return True


def shipped_catalog() -> dict[str, Any] | None:
    """The catalog that came with this app (``away/data/catalog_next.json``), used before the online one has the
    v0.1.24 entries."""
    v = _read_json(Path(__file__).resolve().parent.parent / "data" / "catalog_next.json")
    return v if isinstance(v, dict) else None


def _merge_next(cat: Any, shipped: dict[str, Any] | None) -> Any:
    if shipped is None:
        return cat
    out = dict(cat) if isinstance(cat, dict) else {"models": [], "modes": {}}
    out["models_next"] = shipped.get("models_next") or []
    out["presets_next"] = shipped.get("presets_next") or {}
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
        # v0.1.24: the 2026-10 models are offered by Settings > Newer models / Test my PC, never by this hint
        if m.source == "catalog" and m.is_permissive and m.id not in seen and not m.from_next:
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
