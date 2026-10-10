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
from away.bindings import BindingsConfig, BindingsError, parse_bindings
from away.control.keys import KeyComboError, canonical


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


VOICE_MODES = ("always_on", "push_to_talk", "toggle")


@dataclass
class VoiceConfig:
    enabled: bool = True
    # always_on: ambient listening with VAD end-pointing (default) | push_to_talk: hold `hold_key`
    # | toggle: press `hotkey` to start/stop
    mode: str = "always_on"
    ptt_mode: str = "toggle"  # deprecated: `ptt_mode: hold` in an old config maps to mode: push_to_talk
    hotkey: str = "<ctrl>+<alt>+space"  # hotkey:ptt trigger (toggle mode; in always_on it toggles listening)
    hold_key: str = "ctrl_r"
    mute_hotkey: str = "<ctrl>+<alt>+m"  # hotkey:mute trigger (default action mic:toggle)
    wake_word_enabled: bool = False  # always_on: only act on utterances that start with the wake word
    wake_word: str = "away"
    wake_window: float = 5.0  # seconds after a bare wake word during which the next utterance needs no wake word
    intent: str = "dictation"  # always_on: dictation (type speech + run commands) | command (commands only)
    ambient_fallback: str = "commands_only"  # always_on, no wake word and no face gate: commands_only | type
    vad_sensitivity: float = 0.5  # 0 = only loud, close speech .. 1 = picks up quiet speech
    end_silence_ms: int = 700  # silence that ends an utterance
    min_utterance_ms: int = 250
    max_utterance_s: float = 15.0
    commands_enabled: bool = True  # an utterance that exactly matches a bound phrase runs it instead of typing
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
    mouth_open_trigger: bool = False  # face:mouth_open trigger (bind it under bindings ... face)
    mouth_threshold: float = 0.5
    mouth_hold: float = 0.8


@dataclass
class WifiConfig:
    enabled: bool = False
    source: str = "rssi"  # rssi | picoscenes | iax | synthetic
    path: str | None = None  # file or interface
    interface: str | None = None
    sample_rate: float = 100.0
    fusion_weight: float = 0.25


@dataclass
class AppSection:
    always_on: bool = True  # gestures + voice active as soon as the app launches (False = start paused, mic muted)
    start_minimized: bool = True  # tray only at launch; False also opens the settings window


@dataclass
class AppConfig:
    app: AppSection = field(default_factory=AppSection)
    hands: HandsConfig = field(default_factory=HandsConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    wifi: WifiConfig = field(default_factory=WifiConfig)
    overlay: bool = False
    telemetry: bool = False  # never sends anything; kept explicit for the privacy doc
    models_dir: str | None = None
    bindings: BindingsConfig = field(default_factory=BindingsConfig)


def _merge(obj: Any, data: dict[str, Any], prefix: str, unknown: list[str]) -> Any:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            unknown.append(prefix + key)
            continue
        current = getattr(obj, key)
        if isinstance(current, BindingsConfig):
            setattr(obj, key, parse_bindings(value, prefix + key))
            continue
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
        v = data.get("voice")
        if isinstance(v, dict) and "mode" not in v and v.get("ptt_mode") == "hold":
            cfg.voice.mode = "push_to_talk"  # legacy configs
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
    v = cfg.voice
    if v.mode not in VOICE_MODES:
        raise ValueError(f"voice.mode must be {'|'.join(VOICE_MODES)}, got {v.mode!r}")
    if v.intent not in ("dictation", "command"):
        raise ValueError(f"voice.intent must be dictation|command, got {v.intent!r}")
    if v.ambient_fallback not in ("commands_only", "type"):
        raise ValueError(f"voice.ambient_fallback must be commands_only|type, got {v.ambient_fallback!r}")
    if v.wake_word_enabled and not str(v.wake_word or "").strip():
        raise ValueError("voice.wake_word must not be empty when voice.wake_word_enabled is true")
    for name in ("hotkey", "mute_hotkey", "hold_key"):
        try:
            keys = canonical(str(getattr(v, name)))
        except KeyComboError as exc:
            raise ValueError(f"voice.{name}: {exc}") from None
        if name == "hold_key" and "+" in keys.strip("+"):
            raise ValueError(f"voice.hold_key must be a single key (e.g. ctrl_r, f9), got {v.hold_key!r}")
    for path, value, lo, hi in tunable_ranges(cfg):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= float(value) <= hi:
            raise ValueError(f"{path} must be a number in [{lo}, {hi}], got {value!r}")
    if not isinstance(cfg.bindings, BindingsConfig):
        raise BindingsError(["bindings: invalid"])


def tunable_ranges(cfg: AppConfig) -> list[tuple[str, Any, float, float]]:
    """(yaml path, current value, min, max) for every user tunable; the GUI sliders use the same table."""
    c, g, v = cfg.hands.cursor, cfg.hands.gestures, cfg.voice
    return [
        ("hands.cursor.gain", c.gain, 0.1, 10.0),
        ("hands.cursor.dead_zone", c.dead_zone, 0.0, 0.05),
        ("hands.cursor.acceleration", c.acceleration, 1.0, 3.0),
        ("hands.cursor.min_cutoff", c.min_cutoff, 0.01, 10.0),
        ("hands.cursor.beta", c.beta, 0.0, 1.0),
        ("hands.cursor.d_cutoff", c.d_cutoff, 0.01, 10.0),
        ("hands.gestures.pinch_on", g.pinch_on, 0.05, 1.0),
        ("hands.gestures.pinch_off", g.pinch_off, 0.05, 1.5),
        ("hands.gestures.scroll_gain", g.scroll_gain, 1.0, 200.0),
        ("hands.gestures.swipe_min_dx", g.swipe_min_dx, 0.05, 0.9),
        ("hands.gestures.swipe_max_time", g.swipe_max_time, 0.1, 2.0),
        ("hands.gestures.double_pinch_window", g.double_pinch_window, 0.1, 1.5),
        ("hands.gestures.drag_hold", g.drag_hold, 0.05, 2.0),
        ("hands.gestures.clutch_hold", g.clutch_hold, 0.1, 3.0),
        ("voice.vad_sensitivity", v.vad_sensitivity, 0.0, 1.0),
        ("voice.end_silence_ms", v.end_silence_ms, 150, 3000),
        ("voice.min_utterance_ms", v.min_utterance_ms, 50, 2000),
        ("voice.max_utterance_s", v.max_utterance_s, 2.0, 60.0),
        ("voice.wake_window", v.wake_window, 0.0, 30.0),
        ("face.mouth_threshold", cfg.face.mouth_threshold, 0.05, 1.0),
        ("face.mouth_hold", cfg.face.mouth_hold, 0.1, 5.0),
    ]


def to_dict(cfg: AppConfig) -> dict[str, Any]:
    d = {
        f.name: asdict(getattr(cfg, f.name)) if is_dataclass(getattr(cfg, f.name)) else getattr(cfg, f.name)
        for f in fields(cfg)
        if f.name != "bindings"
    }

    def fix(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, tuple):
            return list(x)
        return x

    result: dict[str, Any] = fix(d)
    result["bindings"] = cfg.bindings.to_dict()
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


def config_path() -> Path:
    """The config file in use (``AWAY_CONFIG`` overrides the per-OS default)."""
    return Path(os.environ.get("AWAY_CONFIG") or paths.config_file())


def save(cfg: AppConfig, path: Path | None = None) -> Path:
    validate(cfg)
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
    return p
