"""Studio gestures on disk: definitions, practice recordings (clips), per-gesture settings, gesture packs.

Layout under ``<data folder>/studio`` (never leaves the computer unless the user exports a pack):

* ``pack.json``: the gestures (name, ``static`` pose or ``motion``, one or two hands);
* ``clips/<label>/<label>_<time>.npz``: practice recordings saved by :func:`away.hands.recorder.save_clip`
  (``X`` pointer hand, ``X2`` + ``H2`` the second hand and where it was seen, ``session_id`` = one practice
  round; ``nothing`` holds the negatives);
* ``model/``: what :mod:`away.studio.trainer` wrote.

Per-gesture settings (on/off, how sure it must be, hold time, rest time, follow-up actions) live in the config
file under ``studio.gestures`` so they travel with the rest of the settings.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from away.bindings import GESTURE_TRIGGERS, LABEL_RE, parse_action

NOTHING = "nothing"  # the negative / background class
KINDS = ("static", "motion")
PACK_FORMAT = "away-gesture-pack"
PACK_VERSION = 1
MAX_CHAIN_STEPS = 10
MAX_WAIT_S = 10.0
MAX_PACK_BYTES = 200 * 1024 * 1024
DEFAULTS = {"static": {"threshold": 0.8, "hold_s": 0.3}, "motion": {"threshold": 0.8, "hold_s": 0.0}}
CONTINUOUS = ("move", "drag", "scroll", "autoscroll")


def studio_dir(base: Path | None = None) -> Path:
    from away import paths

    return Path(base) if base is not None else paths.data_dir() / "studio"


@dataclass
class GestureDef:
    name: str
    kind: str = "static"
    hands: int = 1
    created: str = ""

    def check(self) -> None:
        check_name(self.name)
        if self.kind not in KINDS:
            raise ValueError(f"gesture {self.name!r}: kind must be static or motion")
        if self.hands not in (1, 2):
            raise ValueError(f"gesture {self.name!r}: hands must be 1 or 2")


@dataclass
class GestureSettings:
    enabled: bool = True
    threshold: float = 0.8  # "How sure it must be"
    hold_s: float = 0.3  # "Hold for" before it counts
    cooldown_s: float = 1.0  # "Rest time" before it can fire again
    then: list[str] = field(default_factory=list)  # follow-up steps: actions or "wait:<seconds>"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_name(name: str) -> str:
    """Studio gesture names: lower_snake_case, not a built-in trigger and not 'nothing'."""
    if not isinstance(name, str) or not LABEL_RE.match(name):
        raise ValueError("use lowercase letters, numbers and _ only (for example thumbs_up), up to 32 characters")
    if name == NOTHING:
        raise ValueError("'nothing' is reserved for the 'no gesture' examples")
    if name in GESTURE_TRIGGERS:
        raise ValueError(f"{name!r} is already a built-in gesture; pick another name")
    return name


def friendly_name(text: str) -> str:
    """'Thumbs Up!' -> 'thumbs_up' (what the wizard does with whatever the user typed)."""
    out = "".join(ch if ch.isalnum() and ch.isascii() else "_" for ch in text.strip().lower())
    out = "_".join(p for p in out.split("_") if p)
    if out and not out[0].isalpha():
        out = "g_" + out
    return out[:32]


def parse_chain(steps: Any, allow_shell: bool = False, path: str = "then") -> list[str]:
    """Validate an automation chain: action strings (``key:ctrl+s``) and pauses (``wait:0.5``)."""
    if steps is None:
        return []
    if not isinstance(steps, list):
        raise ValueError(f"{path}: must be a list of steps")
    if len(steps) > MAX_CHAIN_STEPS:
        raise ValueError(f"{path}: at most {MAX_CHAIN_STEPS} steps")
    out = []
    for i, s in enumerate(steps):
        text = str(s).strip()
        if text.lower().startswith("wait:"):
            try:
                secs = float(text.split(":", 1)[1])
            except ValueError:
                raise ValueError(f"{path}[{i}]: wait needs a number of seconds, e.g. wait:0.5") from None
            if not 0.0 <= secs <= MAX_WAIT_S:
                raise ValueError(f"{path}[{i}]: wait must be 0..{MAX_WAIT_S:g} seconds")
            out.append(f"wait:{secs:g}")
            continue
        try:
            act = parse_action(text, allow_shell=allow_shell)
        except ValueError as exc:
            raise ValueError(f"{path}[{i}]: {exc}") from None
        if act.kind == "mouse" and act.arg in CONTINUOUS:
            raise ValueError(f"{path}[{i}]: mouse:{act.arg} is continuous and cannot be a step")
        if act.kind != "none":
            out.append(str(act))
    return out


def settings_from(raw: Any, kind: str = "static", allow_shell: bool = False, path: str = "") -> GestureSettings:
    d = DEFAULTS.get(kind, DEFAULTS["static"])
    s = GestureSettings(threshold=d["threshold"], hold_s=d["hold_s"])
    if raw is None:
        return s
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: must be a mapping")
    unknown = set(raw) - {"enabled", "threshold", "hold_s", "cooldown_s", "then"}
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {', '.join(sorted(unknown))}")
    if "enabled" in raw:
        if not isinstance(raw["enabled"], bool):
            raise ValueError(f"{path}.enabled: must be true or false")
        s.enabled = raw["enabled"]
    for key, lo, hi in (("threshold", 0.3, 0.99), ("hold_s", 0.0, 3.0), ("cooldown_s", 0.0, 30.0)):
        if key in raw:
            v = raw[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
                raise ValueError(f"{path}.{key}: must be a number between {lo} and {hi}")
            setattr(s, key, float(v))
    s.then = parse_chain(raw.get("then"), allow_shell, f"{path}.then")
    return s


def validate_studio(cfg: Any) -> None:
    """Called by :func:`away.config.validate`: the ``studio`` and ``zones`` sections."""
    from away.studio.zones import validate_zones

    validate_zones(cfg.zones)
    g = cfg.studio.gestures
    if not isinstance(g, dict):
        raise ValueError("studio.gestures must be a mapping of gesture name -> settings")
    for name, raw in g.items():
        if not isinstance(name, str) or not LABEL_RE.match(name):
            raise ValueError(f"studio.gestures.{name}: names are lower_snake_case")
        settings_from(raw, "static", cfg.bindings.allow_shell, f"studio.gestures.{name}")


def gesture_settings(cfg: Any, name: str, kind: str = "static") -> GestureSettings:
    """The settings of one Studio gesture (the defaults for its kind when none are saved)."""
    return settings_from(cfg.studio.gestures.get(name), kind, cfg.bindings.allow_shell, f"studio.gestures.{name}")


# ---------------------------------------------------------------------- the pack on disk
class StudioStore:
    """Read/write the Studio folder. All paths stay inside ``root``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = studio_dir(root)

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def pack_file(self) -> Path:
        return self.root / "pack.json"

    # ---- gesture definitions
    def gestures(self) -> list[GestureDef]:
        if not self.pack_file.exists():
            return []
        try:
            data = json.loads(self.pack_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for g in data.get("gestures", []) if isinstance(data, dict) else []:
            with contextlib.suppress(KeyError, TypeError, ValueError):
                gd = GestureDef(str(g["name"]), str(g.get("kind", "static")), int(g.get("hands", 1)))
                gd.created = str(g.get("created", ""))
                gd.check()
                out.append(gd)
        return out

    def get(self, name: str) -> GestureDef | None:
        return next((g for g in self.gestures() if g.name == name), None)

    def _write(self, gestures: list[GestureDef]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.pack_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": PACK_VERSION, "gestures": [asdict(g) for g in gestures]}, indent=2))
        tmp.replace(self.pack_file)

    def add(self, name: str, kind: str = "static", hands: int = 1) -> GestureDef:
        g = GestureDef(name, kind, hands, time.strftime("%Y-%m-%dT%H:%M:%S"))
        g.check()
        current = self.gestures()
        if any(x.name == name for x in current):
            raise ValueError(f"you already have a gesture called {name!r}")
        self._write([*current, g])
        return g

    def remove(self, name: str, delete_clips: bool = True) -> None:
        self._write([g for g in self.gestures() if g.name != name])
        if delete_clips:
            shutil.rmtree(self.clips_dir / name, ignore_errors=True)

    # ---- practice recordings
    def clip_paths(self, label: str | None = None) -> list[Path]:
        base = self.clips_dir / label if label else self.clips_dir
        return sorted(base.glob("**/*.npz")) if base.exists() else []

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.clip_paths():
            out[p.parent.name] = out.get(p.parent.name, 0) + 1
        return out

    def sessions(self, label: str) -> int:
        """Practice rounds recorded for ``label`` (distinct session ids)."""
        ids = set()
        for p in self.clip_paths(label):
            with contextlib.suppress(Exception), np.load(p, allow_pickle=False) as d:
                ids.add(str(d["session_id"].item()) if "session_id" in d.files else p.name)
        return len(ids)

    def save_take(
        self,
        label: str,
        frames: list[tuple[Any, Any]],
        handedness: str,
        fps: float,
        session_id: str,
    ) -> Path:
        """Save one practice recording: ``frames`` = [(pointer landmarks or None, helper or None)].
        Frames without the pointer hand are dropped; the helper is kept with a seen/unseen mask."""
        from away.hands.recorder import save_clip

        if label != NOTHING:
            check_name(label)
        kept = []
        for p, h in frames:
            if p is None:
                continue
            with contextlib.suppress(ValueError, TypeError):
                pa = np.asarray(p, dtype=np.float64).reshape(21, 3)
                ha = np.asarray(h, dtype=np.float64).reshape(21, 3) if h is not None else None
                kept.append((pa, ha))
        if len(kept) < 3:
            raise ValueError("your hand was not in view; try again with your hand in the picture")
        x = [p for p, _h in kept]
        x2 = np.stack([h if h is not None else np.zeros((21, 3)) for _p, h in kept])
        h2 = np.array([h is not None for _p, h in kept], dtype=bool)
        return save_clip(
            self.clips_dir,
            label,
            x,
            handedness,
            fps,
            allow_custom=True,
            session_id=session_id,
            extra={"X2": x2, "H2": h2},
        )

    def delete_all_clips(self) -> int:
        """The privacy button: removes every practice recording. Returns how many files were deleted."""
        n = len(self.clip_paths())
        shutil.rmtree(self.clips_dir, ignore_errors=True)
        return n

    def delete_clip(self, path: Path) -> None:
        p = Path(path).resolve()
        if self.clips_dir.resolve() not in p.parents:
            raise ValueError("not a Studio recording")
        p.unlink(missing_ok=True)


def new_session_id() -> str:
    return f"studio-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------- export / import
def export_pack(
    store: StudioStore,
    cfg: Any,
    out: Path,
    names: list[str] | None = None,
    include_clips: bool = False,
) -> Path:
    """Write a gesture pack zip: ``manifest.json`` + the trained model (+ the recordings only when
    ``include_clips`` is ticked; they are never added on their own)."""
    from away import __version__

    gestures = [g for g in store.gestures() if names is None or g.name in names]
    if not gestures:
        raise ValueError("there are no gestures to share yet")
    model_files = sorted(p for p in store.model_dir.glob("*") if p.is_file()) if store.model_dir.exists() else []
    manifest: dict[str, Any] = {
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "app_version": __version__,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "includes_clips": bool(include_clips),
        "gestures": [],
        "model_files": [p.name for p in model_files],
    }
    for g in gestures:
        act = cfg.bindings.resolve("gestures", g.name, "default")
        manifest["gestures"].append(
            {**asdict(g), "settings": gesture_settings(cfg, g.name, g.kind).to_dict(), "action": str(act)}
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for p in model_files:
            zf.write(p, f"model/{p.name}")
        if include_clips:
            for label in [*[g.name for g in gestures], NOTHING]:
                for p in store.clip_paths(label):
                    zf.write(p, f"clips/{label}/{p.name}")
    return out


def _safe_member(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"the pack contains an unsafe file path: {name!r}")
    return p


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as zf:
            try:
                m = json.loads(zf.read("manifest.json"))
            except KeyError:
                raise ValueError("this file is not a gesture pack (no manifest.json)") from None
    except zipfile.BadZipFile:
        raise ValueError("this file is not a gesture pack (not a zip file)") from None
    if not isinstance(m, dict) or m.get("format") != PACK_FORMAT:
        raise ValueError("this file is not a Project Away gesture pack")
    if int(m.get("version", 0)) > PACK_VERSION:
        raise ValueError("this pack was made by a newer Project Away; update the app first")
    return m


def import_pack(store: StudioStore, cfg: Any, path: Path) -> list[str]:
    """Bring a pack in: its gestures, model and (if it has them) recordings; bindings and settings go into
    ``cfg`` (saved by the caller). Shell actions are skipped unless ``bindings.allow_shell`` is on.
    The pack's model replaces the current one (teach again afterwards to combine it with your own gestures).
    Returns notes for the user."""
    manifest = read_manifest(path)
    notes: list[str] = []
    with zipfile.ZipFile(path) as zf:
        total = 0
        for info in zf.infolist():
            _safe_member(info.filename)
            total += info.file_size
        if total > MAX_PACK_BYTES:
            raise ValueError("this pack is too large (over 200 MB unpacked)")
        incoming = []
        for g in manifest.get("gestures", []):
            gd = GestureDef(str(g.get("name")), str(g.get("kind", "static")), int(g.get("hands", 1)))
            gd.check()
            incoming.append((gd, g))
        if not incoming:
            raise ValueError("this pack has no gestures in it")
        existing = {g.name for g in store.gestures()}
        for gd, _raw in incoming:
            if gd.name not in existing:
                store.add(gd.name, gd.kind, gd.hands)
        model_names = {str(n) for n in manifest.get("model_files", [])}
        if model_names:
            shutil.rmtree(store.model_dir, ignore_errors=True)
            store.model_dir.mkdir(parents=True, exist_ok=True)
        for info in zf.infolist():
            p = _safe_member(info.filename)
            if len(p.parts) == 2 and p.parts[0] == "model" and p.name in model_names:
                (store.model_dir / p.name).write_bytes(zf.read(info))
            elif len(p.parts) == 3 and p.parts[0] == "clips" and p.suffix == ".npz":
                label = p.parts[1]
                if label != NOTHING:
                    check_name(label)
                dest = store.clips_dir / label / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info))
    b = cfg.bindings
    for gd, raw in incoming:
        if gd.name not in b.custom_gestures:
            b.custom_gestures.append(gd.name)
        try:
            s = settings_from(raw.get("settings"), gd.kind, b.allow_shell, gd.name)
        except ValueError as exc:
            s = settings_from(None, gd.kind)
            notes.append(f"{gd.name}: kept the default settings ({exc})")
        cfg.studio.gestures[gd.name] = s.to_dict()
        action = str(raw.get("action") or "none")
        try:
            b.set(f"gesture:{gd.name}", action)
        except ValueError as exc:
            notes.append(f"{gd.name}: its action was not added ({exc})")
    extra = " with recordings" if manifest.get("includes_clips") else ""
    notes.insert(0, f"added {len(incoming)} gesture(s){extra}")
    return notes
