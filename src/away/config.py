"""YAML configuration with typed defaults.

Unknown keys are ignored with a warning list so an old config never crashes a
newer app; missing keys fall back to defaults.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from away import paths


@dataclass
class CursorConfig:
    mode: str = "relative"  # "relative" (trackpad) or "absolute"
    dead_zone: float = 0.004  # normalized hand units below which motion is ignored
    gain: float = 2.2  # relative-mode base speed (screen widths per hand unit)
    acceleration: float = 1.6  # exponent >1 accelerates fast motion
    min_cutoff: float = 1.0  # One Euro filter
    beta: float = 0.02
    d_cutoff: float = 1.0
    active_region: tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85)  # absolute-mode box


@dataclass
class GestureConfig:
    pinch_on: float = 0.32  # thumb-index distance / hand scale to close a pinch
    pinch_off: float = 0.45  # hysteresis to release
    double_pinch_window: float = 0.35  # seconds
    drag_hold: float = 0.30  # pinch held this long (or moved) becomes a drag
    drag_move: float = 0.03
    scroll_gain: float = 40.0
    swipe_min_dx: float = 0.25  # normalized image width
    swipe_max_time: float = 0.45
    clutch_hold: float = 0.4  # open palm held this long toggles pause
    use_model: bool = False  # gate discrete gestures with the temporal model (enable after `away calibrate`)
    model_min_conf: float = 0.7


@dataclass
class HandsConfig:
    enabled: bool = True
    camera_index: int = 0
    max_hands: int = 1
    mirror: bool = True
    cursor: CursorConfig = field(default_factory=CursorConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)


@dataclass
class VoiceConfig:
    enabled: bool = True
    ptt_mode: str = "toggle"  # toggle: press hotkey to start/stop | hold: hold `hold_key` while speaking
    hotkey: str = "<ctrl>+<alt>+space"
    hold_key: str = "ctrl_r"
    asr_backend: str = "faster-whisper"  # faster-whisper | parakeet
    asr_model: str = "large-v3-turbo"  # or "small" for weak CPUs
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    vad: str = "silero"  # silero | energy
    output: str = "paste"  # paste | type
    vocabulary_file: str | None = None
    llm_cleanup: bool = False  # needs AWAY_LLM_BASE_URL (+ AWAY_LLM_API_KEY, AWAY_LLM_MODEL)
    sample_rate: int = 16000


@dataclass
class FaceConfig:
    enabled: bool = True
    gate_dictation: bool = True
    speaking_threshold: float = 0.25
    correlation_window: float = 1.5  # seconds
    min_correlation: float = 0.2
    brow_raise_toggle: bool = False
    brow_threshold: float = 0.55
    brow_hold: float = 0.6


@dataclass
class WifiConfig:
    enabled: bool = False
    source: str = "rssi"  # rssi | picoscenes | iax | synthetic
    path: str | None = None  # file or interface
    interface: str | None = None
    sample_rate: float = 100.0
    fusion_weight: float = 0.25


@dataclass
class AppConfig:
    hands: HandsConfig = field(default_factory=HandsConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    wifi: WifiConfig = field(default_factory=WifiConfig)
    overlay: bool = False
    telemetry: bool = False  # never sends anything; kept explicit for the privacy doc
    models_dir: str | None = None


def _merge(obj: Any, data: dict[str, Any], prefix: str, unknown: list[str]) -> Any:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            unknown.append(prefix + key)
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, prefix + key + ".", unknown)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
    return obj


def from_dict(data: dict[str, Any] | None) -> tuple[AppConfig, list[str]]:
    unknown: list[str] = []
    cfg = AppConfig()
    if data:
        _merge(cfg, data, "", unknown)
    validate(cfg)
    return cfg, unknown


def validate(cfg: AppConfig) -> None:
    if cfg.hands.cursor.mode not in ("relative", "absolute"):
        raise ValueError(f"hands.cursor.mode must be relative|absolute, got {cfg.hands.cursor.mode!r}")
    if cfg.voice.output not in ("paste", "type"):
        raise ValueError(f"voice.output must be paste|type, got {cfg.voice.output!r}")
    if cfg.voice.ptt_mode not in ("toggle", "hold"):
        raise ValueError(f"voice.ptt_mode must be toggle|hold, got {cfg.voice.ptt_mode!r}")
    if cfg.voice.asr_backend not in ("faster-whisper", "parakeet"):
        raise ValueError(f"voice.asr_backend must be faster-whisper|parakeet, got {cfg.voice.asr_backend!r}")
    if cfg.wifi.source not in ("rssi", "picoscenes", "iax", "synthetic"):
        raise ValueError(f"wifi.source invalid: {cfg.wifi.source!r}")
    g = cfg.hands.gestures
    if not g.pinch_on < g.pinch_off:
        raise ValueError("hands.gestures.pinch_on must be < pinch_off (hysteresis)")


def to_dict(cfg: AppConfig) -> dict[str, Any]:
    d = asdict(cfg)

    def fix(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, tuple):
            return list(x)
        return x

    result: dict[str, Any] = fix(d)
    return result


def load(path: Path | None = None) -> tuple[AppConfig, list[str]]:
    p = path or Path(os.environ.get("AWAY_CONFIG") or paths.config_file())
    if not p.exists():
        return from_dict(None)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return from_dict(data)


def save(cfg: AppConfig, path: Path | None = None) -> Path:
    p = path or paths.config_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
    return p
