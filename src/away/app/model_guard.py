"""Automatic rollback for the 2026-10 models (v0.1.24; ``models_next.auto_rollback``).

Whenever a slot is switched to a new model (a preset after "Test my PC", "Try it", or an agent), what it replaced is
remembered per slot in ``<data>/models/undo.json``. Then:

* **it fails** (cannot start, crashes mid-dictation, CUDA error) -> :func:`rollback` puts that slot back as it was
  and returns a plain sentence for the tray / Agent Link;
* **it is slower** -> :func:`try_next` benchmarks before/after and keeps the old model when the new one scores more
  than ``models_next.rollback_tolerance`` worse (the v0.1.23 "Try it" logic, :func:`model_registry.try_model`).

A slot's undo is dropped once the new model has worked (:func:`confirm`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

UNDO = "undo.json"


def _path(data_dir: Path | None) -> Path:
    from away.app.model_registry import models_dir

    return models_dir(data_dir) / UNDO


def _read(data_dir: Path | None) -> dict[str, Any]:
    try:
        v = json.loads(_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _write(data: dict[str, Any], data_dir: Path | None) -> None:
    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")


def remember(slot: str, before: dict[str, Any], data_dir: Path | None = None) -> None:
    """Keep the oldest undo of a slot (several switches in a row still go back to the last model that worked)."""
    data = _read(data_dir)
    if slot not in data:
        data[slot] = dict(before)
        _write(data, data_dir)


def pending(data_dir: Path | None = None) -> dict[str, Any]:
    return _read(data_dir)


def confirm(slot: str, data_dir: Path | None = None) -> None:
    data = _read(data_dir)
    if data.pop(slot, None) is not None:
        _write(data, data_dir)


def rollback(cfg: Any, slot: str, reason: str, data_dir: Path | None = None) -> str | None:
    """Put ``slot`` back (in ``cfg``; the caller saves it). None when there is nothing to undo or rollback is off."""
    from away.app.model_registry import SLOT_TITLES, Registry

    if not bool(getattr(getattr(cfg, "models_next", None), "auto_rollback", True)):
        return None
    data = _read(data_dir)
    before = data.pop(slot, None)
    if not isinstance(before, dict) or not before:
        return None
    Registry.restore(cfg, before)
    _write(data, data_dir)
    return f"{SLOT_TITLES.get(slot, slot)}: switched back to the previous model ({reason[:160]})"


_SLOT_KEYS = (
    ("models_next.stt", "stt"),
    ("models_next.moonshine_model", "stt"),
    ("models_next.nemotron", "stt"),
    ("voice.asr", "stt"),
    ("voice.device", "stt"),
    ("voice.cleanup", "cleanup"),
    ("speech.", "voice"),
    ("models_next.kokoro", "voice"),
    ("models_next.instant_voice", "voice"),
    ("models_next.vad", "vad"),
    ("models_next.wake", "wake"),
    ("gaze.", "gaze"),
    ("face.", "face"),
    ("hands.", "hands"),
    ("models_next.hands", "hands"),
)


def slot_of_key(key: str) -> str:
    for prefix, slot in _SLOT_KEYS:
        if key.startswith(prefix):
            return slot
    return "preset"


def remember_all(before: dict[str, Any], data_dir: Path | None = None) -> None:
    """Split a preset's undo per slot so one failing model rolls back only its own slot."""
    per: dict[str, dict[str, Any]] = {}
    for k, v in before.items():
        per.setdefault(slot_of_key(k), {})[k] = v
    for slot, undo in per.items():
        remember(slot, undo, data_dir)


def try_next(
    reg: Any, cfg: Any, slot: str, mid: str, bench: Callable[[Any], float], data_dir: Path | None = None
) -> Any:
    """"Try it" for a new model with this PC's tolerance; a kept model's undo is remembered for later failures."""
    from away.app.model_registry import try_model

    tol = float(getattr(getattr(cfg, "models_next", None), "rollback_tolerance", 0.1))
    res = try_model(reg, cfg, slot, mid, bench, tolerance=tol)
    if res.kept:
        remember(slot, res.before, data_dir)
    return res
