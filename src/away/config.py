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
from away.voice.cleanup_config import CleanupConfig, DictionaryConfig, validate_cleanup


@dataclass
class AutoCenterConfig:
    """Keep the hand inside the camera frame without thinking about it (v0.1.19, ``hands.cursor.autocenter``)."""

    enabled: bool = True  # master switch for the three helpers below
    # relative mode: moves away from the target get a little more gain and moves back a little less, so over
    # many strokes the hand's resting spot drifts toward the target without the cursor ever jumping
    soft_recenter: bool = True
    soft_strength: float = 0.12  # 0..0.5: +-12% gain at the frame edge, scaled down toward the target
    target: str = "frame"  # frame (camera centre) | comfort (the learned resting spot, kept inside the frame)
    # learned comfort zone: an EMA of where the hand rests; absolute mode centres its active region there
    comfort_zone: bool = True
    comfort_tau_s: float = 20.0  # seconds of resting for the EMA to move ~63% of the way
    rest_speed: float = 0.08  # hand slower than this (frame widths/s) counts as resting
    # edge re-anchor: a quick swing back from the frame edge does not drag the cursor back (like lifting a mouse)
    edge_reanchor: bool = True
    edge_margin: float = 0.08  # edge band, fraction of the frame
    edge_return_speed: float = 0.5  # inward speed (frame widths/s) that counts as a swing back


@dataclass
class HeadConfig:
    """Head tracking from the Face Landmarker's facial transformation matrix (``hands.cursor.head``)."""

    min_cutoff: float = 1.0  # One Euro filter on yaw/pitch/roll (degrees)
    beta: float = 0.05
    pointer_gain: float = 0.012  # pointer mode: primary-monitor widths per degree of head turn
    pointer_dead_zone: float = 0.15  # degrees per frame below which head motion is ignored (tremor)
    pointer_acceleration: float = 1.4  # power curve exponent on the per-frame head delta
    coarse_cols: int = 3  # coarse mode: zones per monitor (columns x rows)
    coarse_rows: int = 3
    coarse_yaw_span: float = 30.0  # coarse mode without a screen map: +- degrees across the whole desktop
    coarse_pitch_span: float = 15.0
    coarse_hysteresis: float = 0.25  # fraction of a zone the head must go past the border to change zone
    coarse_dwell_s: float = 0.35  # the new zone must be held this long before the cursor jumps
    # head-turn monitor switch: needs the screen map (`away gaze calibrate`), off by default
    monitor_switch: bool = False
    switch_margin: float = 4.0  # degrees closer to the other monitor's direction than to the current one
    switch_dwell_s: float = 0.6  # held this long (a glance does not switch)
    gate_tolerance: float = 25.0  # face gate: degrees from the usual screen-facing pose that still count


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
    # transfer function: "power" (|d|^acceleration, the original curve) or "sigmoid" (precise low-speed plateau,
    # capped high-speed plateau; idea from Google Project Gameface, Apache-2.0, re-implemented)
    curve: str = "power"
    sigmoid_low: float = 0.6  # gain multiplier for very slow moves
    sigmoid_high: float = 3.0  # gain multiplier for very fast moves
    sigmoid_mid: float = 1.5  # speed (in units of the neutral per-frame delta) where the curve is halfway
    sigmoid_slope: float = 2.5
    speed_up: float = 1.0  # per-direction multipliers (asymmetry for limited reach / camera angle)
    speed_down: float = 1.0
    speed_left: float = 1.0
    speed_right: float = 1.0
    click_freeze: bool = True  # hold the cursor still while a pinch is closing
    click_rewind_ms: int = 100  # click where the cursor was this long before the pinch started closing
    freeze_margin: float = 0.35  # start freezing when pinch ratio < pinch_on * (1 + margin)
    # multi-monitor: all | primary | [1, 3] (numbers from `away monitors`, left to right)
    monitors: str | list[int] = "all"
    # extra reach for fast moves on a desktop wider/taller than the primary monitor:
    # factor = (desktop span / primary span) ** monitor_reach; 0 = none, 1 = scales fully with the desktop
    monitor_reach: float = 0.5
    monitor_refresh_s: float = 3.0  # re-detect plugged/unplugged monitors this often (0 = only at start)
    # Enlarge the system cursor while gesture control is active (users sit far from the screen); it shrinks back
    # on pause, so the size also shows the state. 0 = off, 1..5 = 1.5x, 2x, 3x, 4x, 6x (Windows only)
    big_cursor_size: int = 2
    autocenter: AutoCenterConfig = field(default_factory=AutoCenterConfig)
    # off | coarse (the head moves the cursor in big steps between zones, the hand refines) | pointer (head-only
    # pointer while no hand points, for when the hands are busy; clicks with the face triggers)
    head_assist: str = "off"
    head: HeadConfig = field(default_factory=HeadConfig)


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
    # four fingers up (thumb folded) = auto-scroll: hand above/below where the gesture started scrolls that way,
    # faster the further it is, and keeps scrolling while the hand holds still; a fist stops it
    autoscroll_enabled: bool = True
    autoscroll_hold: float = 0.2  # four-finger pose held this long before auto-scroll starts
    # thumb_fold_ratio below this = thumb folded (four-finger auto-scroll, not the open palm); released again
    # above it + 0.1 (hysteresis). v0.1.16: 0.95 (was a fixed 0.75, too strict for a real hand: it read as the
    # open palm and toggled pause). Open palms measure about 1.1-1.5.
    autoscroll_thumb_fold: float = 0.95
    autoscroll_dead_zone: float = 0.15  # offset from the neutral point (palm lengths) that does not scroll
    autoscroll_gain: float = 20.0  # wheel notches/s per palm length beyond the dead zone (proportional)
    autoscroll_max_speed: float = 30.0  # wheel notches per second
    autoscroll_stop_hold: float = 0.2  # fist held this long stops auto-scroll
    autoscroll_horizontal: bool = True  # hand left/right of the start scrolls sideways (the larger offset wins)


@dataclass
class DwellConfig:
    enabled: bool = False  # gesture:dwell fires when the cursor rests inside radius_px for time_s
    radius_px: float = 30.0
    time_s: float = 0.8


CAMERA_ROLES = ("primary", "secondary")
CAMERA_BACKENDS = ("auto", "dshow", "msmf", "winrt")  # hands.camera_backend (Windows; elsewhere always auto)


@dataclass
class CameraSpec:
    device: int | str = 0  # camera name ("HD User Facing"), OpenCV index, or device path (/dev/video2)
    role: str = "primary"  # primary (cursor frame of reference) | secondary
    width: int = 640
    height: int = 480
    fps: int = 30
    hfov_deg: float = 70.0  # horizontal field of view, used to approximate intrinsics before calibration
    mirror: bool = True
    index_hint: int | None = None  # index the named camera had when it was chosen (used only with no camera list)
    # pixel format requested before the size (v0.1.12): MJPG keeps two USB webcams within the USB bandwidth;
    # "" = do not set one (the driver's default, usually uncompressed YUY2). Virtual cameras ignore it.
    fourcc: str = "MJPG"


@dataclass
class MultiCamConfig:
    fusion: str = "weighted"  # weighted (confidence-weighted average) | best_view
    pair_tolerance_ms: float = 40.0  # frames further apart than this are not fused
    dropout_ms: float = 300.0  # a camera silent for longer is treated as dropped
    handover_decay_s: float = 1.0  # how fast the offset that hides a view switch fades out
    min_confidence: float = 0.2  # views below this are ignored
    use_3d_pinch: bool = True  # pinch distance from triangulated 3D points when calibrated
    push_distance_m: float = 0.06  # push gesture: hand moves this much toward the screen...
    push_time_s: float = 0.35  # ...within this time
    push_cooldown_s: float = 0.6
    push_monocular: bool = False  # also detect push from hand-size growth with one camera (rougher)
    palm_length_m: float = 0.09  # wrist -> middle MCP, used to give hand-motion calibration a metric scale


ROLE_MODES = ("auto", "right", "left", "any")


@dataclass
class RolesConfig:
    """Which visible hand is the pointer (cursor, clicks, gestures) and which is the helper (v0.1.14)."""

    # auto: the first hand that points (or the only hand in view) | right / left: that real hand | any: the first
    # hand seen, whatever its pose. Hands are told apart by position and size, not MediaPipe's flipping label.
    mode: str = "auto"
    release_s: float = 1.0  # the pointer keeps its role until it has been gone this long (or left the frame)


@dataclass
class PalmScrollConfig:
    """Palm-glide scroll (v0.1.14): glide the pointer's index fingertip along the helper's flat palm."""

    enabled: bool = True
    engage_s: float = 0.15  # fingertip on the palm this long before scrolling starts
    depth_threshold: float = 0.6  # fingertip at most this far off the palm plane (palm lengths; 0 = no depth check)
    margin: float = 0.25  # the palm outline is padded by this much (palm lengths)
    gain: float = 12.0  # wheel notches per palm length of glide
    dead_zone: float = 0.05  # glide below this (palm lengths) does nothing; also the most a tap may move
    invert: bool = False  # True: glide toward the fingers scrolls down
    horizontal: bool = True  # index + middle finger on the palm scroll sideways
    momentum: bool = True  # keep scrolling after the fingertip lifts, slowing down
    friction: float = 4.0  # momentum decay rate (1/s): speed * exp(-friction * t)


@dataclass
class HandsConfig:
    enabled: bool = True
    # single-camera mode (used when `cameras` is empty): a camera name (stable; what Settings saves) or an index
    camera_index: int | str = 0
    camera_hint: int | None = None  # index the named camera had when it was chosen (used only with no camera list)
    camera_fourcc: str = "MJPG"  # single-camera mode pixel format (see CameraSpec.fourcc); "" = do not set
    # a physical camera that is busy (NVIDIA Broadcast keeps its source camera open) is replaced by the one
    # NVIDIA Broadcast virtual camera when exactly one is listed and free; logged when it happens
    virtual_fallback: bool = True
    # v0.1.16: a camera measured below this frame rate is FAILED ("too slow"), not ok, in `away cameras test`,
    # Settings and the app log; calibration refuses to run with it; a too-slow camera is replaced by a free
    # NVIDIA Broadcast camera (when virtual_fallback is on and Broadcast itself reaches this rate)
    min_camera_fps: float = 10.0
    # v0.1.19: the Windows capture backend: auto (DirectShow, then Media Foundation, then WinRT MediaCapture, the
    # Windows Camera app's API) | dshow | msmf | winrt (that backend alone). Ignored on macOS / Linux.
    camera_backend: str = "auto"
    # v0.1.19: a camera that failed while another one tracks is re-probed after camera_retry_s, then after twice
    # as long each time (capped at camera_retry_max_s); after camera_retry_limit failed probes in a row the app
    # stops retrying and says so (tray / Settings > Cameras > Retry cameras starts again). 0 = never stop.
    camera_retry_s: float = 45.0
    camera_retry_max_s: float = 900.0
    camera_retry_limit: int = 8
    max_hands: int = 2  # hands MediaPipe looks for; 2 = pointer + helper hand (1 = lighter on CPU, no palm scroll)
    # hand roles (v0.1.14; replaces hands.prefer: right/left -> mode right/left, any -> auto)
    roles: RolesConfig = field(default_factory=RolesConfig)
    palm_scroll: PalmScrollConfig = field(default_factory=PalmScrollConfig)
    mirror: bool = True
    cursor: CursorConfig = field(default_factory=lambda: CursorConfig(curve="sigmoid"))
    gestures: GestureConfig = field(default_factory=GestureConfig)
    dwell: DwellConfig = field(default_factory=DwellConfig)
    cameras: list[CameraSpec] = field(default_factory=list)  # 2+ entries turn on multi-camera fusion
    multicam: MultiCamConfig = field(default_factory=MultiCamConfig)


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
    asr_model: str = "auto"  # auto = picked from your hardware (`away hardware`); or large-v3-turbo, small, base...
    device: str = "auto"  # auto | cuda | cpu (explicit wins over performance.mode)
    compute_type: str = "auto"  # auto | float16 | int8_float16 | int8 | float32
    language: str | None = None
    vad: str = "silero"  # silero | energy
    output: str = "paste"  # paste | type
    vocabulary_file: str | None = None
    llm_cleanup: bool = False  # needs AWAY_LLM_BASE_URL (+ AWAY_LLM_API_KEY, AWAY_LLM_MODEL)
    sample_rate: int = 16000
    hotwords: bool = True  # also pass the vocabulary to faster-whisper `hotwords` (in addition to initial_prompt)
    numbers_as_digits: bool = True  # "twenty five" -> 25
    numbers_min_value: int = 10  # smaller numbers stay words ("one of them")
    edit_hotkey: str = "<ctrl>+<alt>+e"  # speak an instruction that edits the selected text ("" = off)
    # clap control (uses the voice microphone): one clap = resume gesture control, two claps = pause it
    clap_enabled: bool = True
    clap_sensitivity: float = 0.5  # 0 = only sharp, loud claps .. 1 = softer claps (more false triggers)
    clap_double_window: float = 0.6  # second clap within this many seconds = double clap (a single waits this long)
    clap_require_hands: bool = False  # only accept claps while the camera sees both hands close together
    # v0.1.20: smart cleanup (rules / optional local LLM, per-app tone) and the personal dictionary
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)


@dataclass
class IPhoneFaceConfig:
    """Live Link Face (iOS) receiver; used when ``face.source`` is ``iphone`` or ``auto``. See docs/iphone.md."""

    host: str = "0.0.0.0"  # interface to listen on (0.0.0.0 = all)
    port: int = 11111  # the Live Link Face app's default target port (UDP)
    stale_after: float = 0.5  # seconds without a face packet before the iPhone counts as gone (auto -> webcam)
    max_rate: float = 30.0  # feed the face pipeline at most this many frames/s (the phone sends ~60)
    swap_sides: bool = False  # flip Left/Right blendshapes if winks come out reversed vs the webcam


FACE_SOURCES = ("webcam", "iphone", "auto")


@dataclass
class FaceConfig:
    enabled: bool = True
    source: str = "webcam"  # webcam | iphone | auto (experimental iPhone via Live Link Face; see docs/iphone.md)
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
    wink_trigger: bool = False  # face:wink_left / face:wink_right triggers
    wink_threshold: float = 0.5  # eyeBlink score of the closed eye
    wink_open_max: float = 0.3  # the other eye must stay below this (a blink closes both)
    wink_hold: float = 0.25
    iphone: IPhoneFaceConfig = field(default_factory=IPhoneFaceConfig)


GAZE_MODELS = ("auto", "iris", "l2cs", "mobilegaze")


@dataclass
class GazeConfig:
    """Eye tracking + the spatial screen map (``away gaze calibrate``). Nothing here acts until calibrated."""

    enabled: bool = True  # master switch (still needs a calibration for the current camera)
    # iris: iris landmarks + head pose only (no extra model) | l2cs / mobilegaze: also a learned gaze network
    # (optional download, research-only weights) | auto: l2cs when downloaded and a GPU is usable, else
    # mobilegaze when downloaded, else iris
    model: str = "auto"
    model_rate_hz: float = 10.0  # the learned gaze network runs at most this often
    min_cutoff: float = 0.4  # One Euro filter on the predicted gaze point (pixels)
    beta: float = 0.004
    ridge_alpha: float = 1.0  # calibration regression regularisation
    points_per_monitor: int = 9  # calibration dots per monitor (9 = 3x3; 5 = corners + centre)
    # look-to-jump (MAGIC pointing): the hand starts moving after resting -> warp to where you look
    look_to_jump: bool = True
    jump_idle_s: float = 0.6
    jump_min_px: float = 250.0  # only when the gaze point is at least this far from the cursor
    jump_across_monitors_only: bool = False  # only warp when you look at another monitor
    # drift correction: each click nudges the model toward where you clicked (you look where you click)
    drift_correction: bool = True
    drift_rate: float = 0.1
    drift_max_px: float = 150.0  # the learned offset never exceeds this
    drift_max_err_px: float = 350.0  # clicks further from the gaze point are ignored (not looking there)


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
    auto_pause: bool = True  # physical mouse/keyboard use pauses gesture control...
    auto_pause_idle_ms: int = 1500  # ...until it has been idle this long
    toggle_hotkey: str = "f9"  # global on/off for gesture control ("" = off)


PERFORMANCE_MODES = ("auto", "cpu", "gpu")


@dataclass
class PerformanceConfig:
    """Hardware-dependent settings. ``auto`` detects CPU/RAM/GPU and picks them (`away hardware` shows what and
    why); any number set here (non-zero) overrides the automatic choice."""

    mode: str = "auto"  # auto (GPU when an NVIDIA GPU + CUDA libraries are usable) | cpu | gpu
    cpu_threads: int = 0  # Whisper CPU threads (0 = auto)
    beam_size: int = 0  # Whisper beam size (0 = auto; 1 = fastest)
    camera_width: int = 0  # single-camera capture size / rate (0 = auto)
    camera_height: int = 0
    camera_fps: int = 0
    hands_every: int = 0  # run hand tracking on 1 in N frames (0 = auto)
    face_every: int = 0  # run face tracking on 1 in N frames (0 = auto)


UPDATE_CHANNELS = ("stable", "beta")


@dataclass
class UpdatesConfig:
    # The only network call is an anonymous GET to the public GitHub Releases API; nothing is sent about you.
    check: bool = True  # check when the tray app starts and every 24 h (background thread)
    auto_install: bool = False  # False: notify + tray item "Update to vX.Y.Z"; True: install when idle/paused
    channel: str = "stable"  # stable | beta (beta also offers GitHub pre-releases)
    allow_unverified: bool = False  # install a download that has no published SHA-256 (not recommended)


CONFIG_VERSION = 4  # 1 (implicit) = written by v0.1.5 or older; 2 = v0.1.6..v0.1.8; 3 = v0.1.9..v0.1.13; 4 = v0.1.14


@dataclass
class AppConfig:
    config_version: int = CONFIG_VERSION
    app: AppSection = field(default_factory=AppSection)
    hands: HandsConfig = field(default_factory=HandsConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    wifi: WifiConfig = field(default_factory=WifiConfig)
    updates: UpdatesConfig = field(default_factory=UpdatesConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    overlay: bool = False
    cursor_overlay: bool = False  # ring at the cursor: pinch progress, drag, paused, dwell
    telemetry: bool = False  # never sends anything; kept explicit for the privacy doc
    models_dir: str | None = None
    bindings: BindingsConfig = field(default_factory=BindingsConfig)
    # what :func:`migrate` changed while loading (not saved); the app re-saves the file when non-empty
    migrations: list[str] = field(default_factory=list, compare=False, repr=False)


# Values older versions wrote into every config file as plain defaults (setup/Settings save the whole
# config), so finding them there says nothing about the user's choice.
OLD_WRITTEN_DEFAULTS = {("voice", "asr_model"): ("large-v3-turbo",)}  # v0.1.0-v0.1.4 default
HAND_PREFERENCES = ("any", "right", "left")  # the old hands.prefer values (read only by the migration)
PREFER_TO_MODE = {"right": "right", "left": "left", "any": "auto"}


def migrate(data: dict[str, Any]) -> list[str]:
    """Upgrade a raw config mapping written by an older version, in place. Returns what changed.

    v1 -> v2: ``voice.asr_model: large-v3-turbo`` was the hard-coded default up to v0.1.4 and setup wrote it
    into the file, so on a PC without a working GPU it pinned the slowest model to the CPU. It becomes
    ``auto`` (the hardware profile picks large-v3-turbo on a usable GPU, a CPU-sized model otherwise).
    A model the user picked in v0.1.6+ is saved with ``config_version: 2`` and never touched.
    """
    notes: list[str] = []
    ver = data.get("config_version", 1)
    if not isinstance(ver, int) or isinstance(ver, bool):
        ver = 1
    if ver < 2:
        for (section, key), olds in OLD_WRITTEN_DEFAULTS.items():
            sec = data.get(section)
            if isinstance(sec, dict) and sec.get(key) in olds:
                notes.append(f"{section}.{key}: {sec[key]} (old default written by setup) -> auto")
                sec[key] = "auto"
    if ver < 3:
        hands = data.get("hands")
        if isinstance(hands, dict) and hands.get("max_hands") == 1:
            # v0.1.8 and older wrote the old default max_hands: 1 into every file; v0.1.9 needs 2 to tell the
            # cursor hand from the other one (hands.prefer)
            notes.append("hands.max_hands: 1 (old default written by setup) -> 2")
            hands["max_hands"] = 2
    hands = data.get("hands")
    if isinstance(hands, dict) and "prefer" in hands:
        # v0.1.9..v0.1.13 hands.prefer -> v0.1.14 hands.roles.mode (any -> auto: the new pose-based roles)
        old = hands.pop("prefer")
        if old not in PREFER_TO_MODE:
            raise ValueError(f"hands.prefer must be {'|'.join(HAND_PREFERENCES)}, got {old!r}")
        roles = hands.get("roles")
        roles = dict(roles) if isinstance(roles, dict) else {}
        if "mode" not in roles:
            roles["mode"] = PREFER_TO_MODE[old]
            notes.append(f"hands.prefer: {old} -> hands.roles.mode: {roles['mode']}")
        hands["roles"] = roles
    if ver < 4:
        # the default voice command "use either hand" was hands:any up to v0.1.13 (saved in every file); it now
        # restores automatic roles
        voice = _get_path(data, ("bindings", "profiles", "default", "voice"))
        if isinstance(voice, dict) and str(voice.get("use either hand", "")).strip().lower() == "hands:any":
            voice["use either hand"] = "hands:auto"
            notes.append("bindings voice 'use either hand': hands:any -> hands:auto")
    data["config_version"] = CONFIG_VERSION
    return notes


def _get_path(data: Any, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(k)
    return data


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
        if key == "cameras" and isinstance(obj, HandsConfig):
            setattr(obj, key, _parse_cameras(value, prefix + key, unknown))
            continue
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, prefix + key + ".", unknown)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
    return obj


def _parse_cameras(value: Any, path: str, unknown: list[str]) -> list[CameraSpec]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list of cameras")
    out = []
    for i, item in enumerate(value):
        if isinstance(item, (int, str)):
            item = {"device": item, "role": "primary" if i == 0 else "secondary"}
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{i}] must be a mapping (device, role, width, height, fps)")
        out.append(_merge(CameraSpec(role="primary" if i == 0 else "secondary"), item, f"{path}[{i}].", unknown))
    return out


def from_dict(data: dict[str, Any] | None) -> tuple[AppConfig, list[str]]:
    unknown: list[str] = []
    cfg = AppConfig()
    if data:
        data = dict(data)
        for sec in ("voice", "hands"):
            if isinstance(data.get(sec), dict):
                data[sec] = dict(data[sec])
        data["bindings"] = _copy_voice_bindings(data.get("bindings"))
        if data["bindings"] is None:
            data.pop("bindings")
        notes = migrate(data)
        data.pop("migrations", None)
        _merge(cfg, data, "", unknown)
        cfg.migrations = notes
        v = data.get("voice")
        if isinstance(v, dict) and "mode" not in v and v.get("ptt_mode") == "hold":
            cfg.voice.mode = "push_to_talk"  # legacy configs
    validate(cfg)
    return cfg, unknown


def _copy_voice_bindings(b: Any) -> Any:
    """Copy the path the migration may edit (bindings.profiles.default.voice) so the caller's dict is untouched."""
    if not isinstance(b, dict):
        return b
    b = dict(b)
    profs = b.get("profiles")
    if isinstance(profs, dict):
        profs = dict(profs)
        d = profs.get("default")
        if isinstance(d, dict):
            d = dict(d)
            if isinstance(d.get("voice"), dict):
                d["voice"] = dict(d["voice"])
            profs["default"] = d
        b["profiles"] = profs
    return b


def validate(cfg: AppConfig) -> None:
    if cfg.hands.cursor.mode not in ("relative", "absolute"):
        raise ValueError(f"hands.cursor.mode must be relative|absolute, got {cfg.hands.cursor.mode!r}")
    if cfg.voice.output not in ("paste", "type"):
        raise ValueError(f"voice.output must be paste|type, got {cfg.voice.output!r}")
    if cfg.voice.ptt_mode not in ("toggle", "hold"):
        raise ValueError(f"voice.ptt_mode must be toggle|hold, got {cfg.voice.ptt_mode!r}")
    if cfg.voice.asr_backend not in ("faster-whisper", "parakeet"):
        raise ValueError(f"voice.asr_backend must be faster-whisper|parakeet, got {cfg.voice.asr_backend!r}")
    perf = cfg.performance
    if perf.mode not in PERFORMANCE_MODES:
        raise ValueError(f"performance.mode must be {'|'.join(PERFORMANCE_MODES)}, got {perf.mode!r}")
    for name, top in (
        ("cpu_threads", 256),
        ("beam_size", 10),
        ("camera_width", 7680),
        ("camera_height", 4320),
        ("camera_fps", 240),
        ("hands_every", 10),
        ("face_every", 30),
    ):
        val = getattr(perf, name)
        if isinstance(val, bool) or not isinstance(val, int) or not 0 <= val <= top:
            raise ValueError(f"performance.{name} must be an integer in [0, {top}] (0 = auto), got {val!r}")
    if cfg.voice.device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"voice.device must be auto|cuda|cpu, got {cfg.voice.device!r}")
    if not str(cfg.voice.asr_model or "").strip():
        raise ValueError("voice.asr_model must not be empty (use 'auto')")
    if cfg.updates.channel not in UPDATE_CHANNELS:
        raise ValueError(f"updates.channel must be {'|'.join(UPDATE_CHANNELS)}, got {cfg.updates.channel!r}")
    if cfg.face.source not in FACE_SOURCES:
        raise ValueError(f"face.source must be {'|'.join(FACE_SOURCES)}, got {cfg.face.source!r}")
    ip = cfg.face.iphone
    if isinstance(ip.port, bool) or not isinstance(ip.port, int) or not 1 <= ip.port <= 65535:
        raise ValueError(f"face.iphone.port must be an integer in [1, 65535], got {ip.port!r}")
    if not isinstance(ip.host, str) or not ip.host.strip():
        raise ValueError("face.iphone.host must be an IP address or hostname (0.0.0.0 = all interfaces)")
    if not isinstance(ip.swap_sides, bool):
        raise ValueError(f"face.iphone.swap_sides must be true or false, got {ip.swap_sides!r}")
    for name, lo, hi in (("stale_after", 0.05, 10.0), ("max_rate", 1.0, 120.0)):
        val = getattr(ip, name)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not lo <= float(val) <= hi:
            raise ValueError(f"face.iphone.{name} must be a number in [{lo}, {hi}], got {val!r}")
    if cfg.wifi.source not in ("rssi", "picoscenes", "iax", "synthetic"):
        raise ValueError(f"wifi.source invalid: {cfg.wifi.source!r}")
    c = cfg.hands.cursor
    if c.curve not in ("power", "sigmoid"):
        raise ValueError(f"hands.cursor.curve must be power|sigmoid, got {c.curve!r}")
    if not c.sigmoid_low <= c.sigmoid_high:
        raise ValueError("hands.cursor.sigmoid_low must be <= sigmoid_high")
    from away.control.monitors import parse_monitor_spec

    c.monitors = parse_monitor_spec(c.monitors)
    r = c.monitor_refresh_s
    if isinstance(r, bool) or not isinstance(r, (int, float)) or not 0 <= float(r) <= 300:
        raise ValueError(f"hands.cursor.monitor_refresh_s must be seconds in [0, 300] (0 = off), got {r!r}")
    validate_tracking(cfg)
    if cfg.hands.roles.mode not in ROLE_MODES:
        raise ValueError(f"hands.roles.mode must be {'|'.join(ROLE_MODES)}, got {cfg.hands.roles.mode!r}")
    ps = cfg.hands.palm_scroll
    for name in ("enabled", "invert", "horizontal", "momentum"):
        if not isinstance(getattr(ps, name), bool):
            raise ValueError(f"hands.palm_scroll.{name} must be true or false, got {getattr(ps, name)!r}")
    mh = cfg.hands.max_hands
    if isinstance(mh, bool) or not isinstance(mh, int) or not 1 <= mh <= 4:
        raise ValueError(f"hands.max_hands must be an integer in [1, 4], got {mh!r}")
    bc = c.big_cursor_size
    if isinstance(bc, bool) or not isinstance(bc, int) or not 0 <= bc <= 5:
        raise ValueError(f"hands.cursor.big_cursor_size must be an integer in [0, 5] (0 = off), got {bc!r}")
    validate_camera_device("hands.camera_index", cfg.hands.camera_index)
    validate_hint("hands.camera_hint", cfg.hands.camera_hint)
    validate_cameras(cfg.hands.cameras)
    cfg.hands.camera_fourcc = validate_fourcc("hands.camera_fourcc", cfg.hands.camera_fourcc)
    h = cfg.hands
    if h.camera_backend not in CAMERA_BACKENDS:
        raise ValueError(f"hands.camera_backend must be {'|'.join(CAMERA_BACKENDS)}, got {h.camera_backend!r}")
    for name, lo, hi in (("camera_retry_s", 5.0, 3600.0), ("camera_retry_max_s", 5.0, 86400.0)):
        val = getattr(h, name)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not lo <= float(val) <= hi:
            raise ValueError(f"hands.{name} must be seconds in [{lo:g}, {hi:g}], got {val!r}")
    if h.camera_retry_max_s < h.camera_retry_s:
        raise ValueError("hands.camera_retry_max_s must be >= hands.camera_retry_s")
    lim = h.camera_retry_limit
    if isinstance(lim, bool) or not isinstance(lim, int) or not 0 <= lim <= 1000:
        raise ValueError(f"hands.camera_retry_limit must be an integer in [0, 1000] (0 = never stop), got {lim!r}")
    if not isinstance(cfg.hands.virtual_fallback, bool):
        raise ValueError(f"hands.virtual_fallback must be true or false, got {cfg.hands.virtual_fallback!r}")
    if cfg.hands.multicam.fusion not in ("weighted", "best_view"):
        raise ValueError(f"hands.multicam.fusion must be weighted|best_view, got {cfg.hands.multicam.fusion!r}")
    for name, combo in (("app.toggle_hotkey", cfg.app.toggle_hotkey), ("voice.edit_hotkey", cfg.voice.edit_hotkey)):
        if combo:
            try:
                canonical(str(combo))
            except KeyComboError as exc:
                raise ValueError(f"{name}: {exc}") from None
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
    validate_profile_tuning(cfg)
    validate_cleanup(cfg.voice.cleanup, cfg.voice.dictionary)


def validate_profile_tuning(cfg: AppConfig) -> None:
    """Per-app profile ``tuning:`` entries name real tunables and stay inside their slider range."""
    ranges = {p: (lo, hi) for p, _v, lo, hi in tunable_ranges(cfg)}
    for name, prof in cfg.bindings.profiles.items():
        for path, value in getattr(prof, "tuning", {}).items():
            where = f"bindings.profiles.{name}.tuning.{path}"
            if path not in ranges:
                raise ValueError(f"{where}: unknown setting")
            lo, hi = ranges[path]
            if not lo <= float(value) <= hi:
                raise ValueError(f"{where} must be a number in [{lo}, {hi}], got {value!r}")


HEAD_ASSIST_MODES = ("off", "coarse", "pointer")
AUTOCENTER_TARGETS = ("frame", "comfort")


def validate_tracking(cfg: AppConfig) -> None:
    """hands.cursor.autocenter / head_assist / head and gaze: the enums and switches (numbers are in
    :func:`tunable_ranges`)."""
    c = cfg.hands.cursor
    ac, hd, gz = c.autocenter, c.head, cfg.gaze
    if c.head_assist is False:  # YAML 1.1 reads a bare `off` as false
        c.head_assist = "off"
    if c.head_assist not in HEAD_ASSIST_MODES:
        raise ValueError(f"hands.cursor.head_assist must be {'|'.join(HEAD_ASSIST_MODES)}, got {c.head_assist!r}")
    if ac.target not in AUTOCENTER_TARGETS:
        raise ValueError(f"hands.cursor.autocenter.target must be {'|'.join(AUTOCENTER_TARGETS)}, got {ac.target!r}")
    if gz.model not in GAZE_MODELS:
        raise ValueError(f"gaze.model must be {'|'.join(GAZE_MODELS)}, got {gz.model!r}")
    for path, obj, names in (
        ("hands.cursor.autocenter", ac, ("enabled", "soft_recenter", "comfort_zone", "edge_reanchor")),
        ("hands.cursor.head", hd, ("monitor_switch",)),
        ("gaze", gz, ("enabled", "look_to_jump", "jump_across_monitors_only", "drift_correction")),
    ):
        for name in names:
            if not isinstance(getattr(obj, name), bool):
                raise ValueError(f"{path}.{name} must be true or false, got {getattr(obj, name)!r}")
    for path, val in (
        ("hands.cursor.head.coarse_cols", hd.coarse_cols),
        ("hands.cursor.head.coarse_rows", hd.coarse_rows),
        ("gaze.points_per_monitor", gz.points_per_monitor),
    ):
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{path} must be an integer, got {val!r}")
    if gz.points_per_monitor not in (5, 9, 16):
        raise ValueError(f"gaze.points_per_monitor must be 5, 9 or 16, got {gz.points_per_monitor!r}")


def validate_camera_device(path: str, device: Any) -> None:
    """A camera is a name (``"HD User Facing"``), an index (0, 1, ...) or a device path."""
    if isinstance(device, bool) or not isinstance(device, (int, str)) or str(device).strip() == "":
        raise ValueError(f"{path} must be a camera name, an index (0, 1, ...) or a device path, got {device!r}")
    if isinstance(device, int) and device < 0:
        raise ValueError(f"{path} must not be negative, got {device!r}")


def validate_hint(path: str, hint: Any) -> None:
    if hint is not None and (isinstance(hint, bool) or not isinstance(hint, int) or hint < 0):
        raise ValueError(f"{path} must be a camera index (0, 1, ...) or empty, got {hint!r}")


def validate_fourcc(path: str, value: Any) -> str:
    """A FOURCC is ``""`` (do not set) or 4 printable ASCII characters (``MJPG``, ``YUY2``); returned upper-cased."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a 4-character pixel format such as MJPG, or empty, got {value!r}")
    v = value.strip()
    if v and (len(v) != 4 or not v.isascii() or not v.isprintable()):
        raise ValueError(f"{path} must be a 4-character pixel format such as MJPG, or empty, got {value!r}")
    return v.upper()


def validate_cameras(cams: list[CameraSpec]) -> None:
    for i, cam in enumerate(cams):
        p = f"hands.cameras[{i}]"
        cam.fourcc = validate_fourcc(f"{p}.fourcc", cam.fourcc)
        if cam.role not in CAMERA_ROLES:
            raise ValueError(f"{p}.role must be primary|secondary, got {cam.role!r}")
        validate_camera_device(f"{p}.device", cam.device)
        validate_hint(f"{p}.index_hint", cam.index_hint)
        for k, lo, hi in (("width", 160, 7680), ("height", 120, 4320), ("fps", 5, 240), ("hfov_deg", 20.0, 170.0)):
            v = getattr(cam, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError(f"{p}.{k} must be in [{lo}, {hi}], got {v!r}")
    if cams:
        prim = [c for c in cams if c.role == "primary"]
        if len(prim) != 1:
            raise ValueError(f"hands.cameras needs exactly one primary camera, got {len(prim)}")
        devs = [str(c.device) for c in cams]
        if len(set(devs)) != len(devs):
            raise ValueError("hands.cameras lists the same device twice")


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
        ("hands.cursor.big_cursor_size", c.big_cursor_size, 0, 5),
        ("hands.gestures.autoscroll_hold", g.autoscroll_hold, 0.05, 1.0),
        ("hands.gestures.autoscroll_thumb_fold", g.autoscroll_thumb_fold, 0.3, 1.4),
        ("hands.gestures.autoscroll_dead_zone", g.autoscroll_dead_zone, 0.0, 2.0),
        ("hands.gestures.autoscroll_gain", g.autoscroll_gain, 0.5, 100.0),
        ("hands.gestures.autoscroll_max_speed", g.autoscroll_max_speed, 1.0, 200.0),
        ("hands.gestures.autoscroll_stop_hold", g.autoscroll_stop_hold, 0.05, 1.5),
        ("voice.vad_sensitivity", v.vad_sensitivity, 0.0, 1.0),
        ("voice.end_silence_ms", v.end_silence_ms, 150, 3000),
        ("voice.min_utterance_ms", v.min_utterance_ms, 50, 2000),
        ("voice.max_utterance_s", v.max_utterance_s, 2.0, 60.0),
        ("voice.wake_window", v.wake_window, 0.0, 30.0),
        ("face.mouth_threshold", cfg.face.mouth_threshold, 0.05, 1.0),
        ("face.mouth_hold", cfg.face.mouth_hold, 0.1, 5.0),
        ("hands.cursor.sigmoid_low", c.sigmoid_low, 0.05, 5.0),
        ("hands.cursor.sigmoid_high", c.sigmoid_high, 0.1, 10.0),
        ("hands.cursor.sigmoid_mid", c.sigmoid_mid, 0.1, 10.0),
        ("hands.cursor.sigmoid_slope", c.sigmoid_slope, 0.1, 20.0),
        ("hands.cursor.speed_up", c.speed_up, 0.1, 5.0),
        ("hands.cursor.speed_down", c.speed_down, 0.1, 5.0),
        ("hands.cursor.speed_left", c.speed_left, 0.1, 5.0),
        ("hands.cursor.speed_right", c.speed_right, 0.1, 5.0),
        ("hands.cursor.click_rewind_ms", c.click_rewind_ms, 0, 500),
        ("hands.cursor.freeze_margin", c.freeze_margin, 0.0, 1.0),
        ("hands.cursor.monitor_reach", c.monitor_reach, 0.0, 1.0),
        ("hands.dwell.radius_px", cfg.hands.dwell.radius_px, 2.0, 200.0),
        ("hands.dwell.time_s", cfg.hands.dwell.time_s, 0.2, 5.0),
        ("hands.multicam.pair_tolerance_ms", cfg.hands.multicam.pair_tolerance_ms, 1.0, 500.0),
        ("hands.multicam.push_distance_m", cfg.hands.multicam.push_distance_m, 0.01, 0.5),
        ("face.brow_threshold", cfg.face.brow_threshold, 0.05, 1.0),
        ("face.wink_threshold", cfg.face.wink_threshold, 0.05, 1.0),
        ("face.wink_open_max", cfg.face.wink_open_max, 0.0, 1.0),
        ("face.wink_hold", cfg.face.wink_hold, 0.05, 3.0),
        ("app.auto_pause_idle_ms", cfg.app.auto_pause_idle_ms, 100, 10000),
        ("voice.numbers_min_value", cfg.voice.numbers_min_value, 0, 1000000),
        ("voice.clap_sensitivity", cfg.voice.clap_sensitivity, 0.0, 1.0),
        ("voice.clap_double_window", cfg.voice.clap_double_window, 0.2, 1.5),
        ("hands.roles.release_s", cfg.hands.roles.release_s, 0.1, 5.0),
        ("hands.palm_scroll.engage_s", cfg.hands.palm_scroll.engage_s, 0.0, 1.0),
        ("hands.palm_scroll.depth_threshold", cfg.hands.palm_scroll.depth_threshold, 0.0, 3.0),
        ("hands.palm_scroll.margin", cfg.hands.palm_scroll.margin, 0.0, 1.0),
        ("hands.palm_scroll.gain", cfg.hands.palm_scroll.gain, 0.5, 100.0),
        ("hands.palm_scroll.dead_zone", cfg.hands.palm_scroll.dead_zone, 0.0, 0.5),
        ("hands.palm_scroll.friction", cfg.hands.palm_scroll.friction, 0.5, 20.0),
        ("hands.min_camera_fps", cfg.hands.min_camera_fps, 1.0, 60.0),
        *tracking_tunables(cfg),
    ]


# Tuning-tab labels for :func:`tracking_tunables` (merged into the Settings window's label table)
TRACKING_TUNABLE_LABELS = {
    "hands.cursor.autocenter.soft_strength": "Auto-center: soft recenter strength",
    "hands.cursor.autocenter.comfort_tau_s": "Auto-center: comfort zone learning time (s)",
    "hands.cursor.autocenter.rest_speed": "Auto-center: resting hand speed",
    "hands.cursor.autocenter.edge_margin": "Auto-center: camera edge band",
    "hands.cursor.autocenter.edge_return_speed": "Auto-center: swing-back speed at the edge",
    "hands.cursor.head.min_cutoff": "Head: smoothing (min cutoff)",
    "hands.cursor.head.beta": "Head: smoothing (speed response)",
    "hands.cursor.head.pointer_gain": "Head pointer: speed",
    "hands.cursor.head.pointer_dead_zone": "Head pointer: dead zone (degrees)",
    "hands.cursor.head.pointer_acceleration": "Head pointer: acceleration",
    "hands.cursor.head.coarse_cols": "Head coarse: zone columns per monitor",
    "hands.cursor.head.coarse_rows": "Head coarse: zone rows per monitor",
    "hands.cursor.head.coarse_yaw_span": "Head coarse: head turn across the desktop (deg)",
    "hands.cursor.head.coarse_pitch_span": "Head coarse: head tilt across the desktop (deg)",
    "hands.cursor.head.coarse_hysteresis": "Head coarse: zone hysteresis",
    "hands.cursor.head.coarse_dwell_s": "Head coarse: dwell before a jump (s)",
    "hands.cursor.head.switch_margin": "Head monitor switch: margin (degrees)",
    "hands.cursor.head.switch_dwell_s": "Head monitor switch: dwell (s)",
    "hands.cursor.head.gate_tolerance": "Face gate: head turn still facing the screen (deg)",
    "gaze.model_rate_hz": "Gaze: learned model rate (Hz)",
    "gaze.min_cutoff": "Gaze: smoothing (min cutoff)",
    "gaze.beta": "Gaze: smoothing (speed response)",
    "gaze.ridge_alpha": "Gaze: calibration regularisation",
    "gaze.jump_idle_s": "Look-to-jump: hand rest before a jump (s)",
    "gaze.jump_min_px": "Look-to-jump: minimum distance (px)",
    "gaze.drift_rate": "Gaze drift correction: rate per click",
    "gaze.drift_max_px": "Gaze drift correction: maximum offset (px)",
    "gaze.drift_max_err_px": "Gaze drift correction: ignore clicks further than (px)",
}
TRACKING_INT_TUNABLES = frozenset({"hands.cursor.head.coarse_cols", "hands.cursor.head.coarse_rows"})


def tracking_tunables(cfg: AppConfig) -> list[tuple[str, Any, float, float]]:
    """Auto-centering, head tracking and gaze (v0.1.19): appended to :func:`tunable_ranges`."""
    ac, hd, gz = cfg.hands.cursor.autocenter, cfg.hands.cursor.head, cfg.gaze
    return [
        ("hands.cursor.autocenter.soft_strength", ac.soft_strength, 0.0, 0.5),
        ("hands.cursor.autocenter.comfort_tau_s", ac.comfort_tau_s, 1.0, 600.0),
        ("hands.cursor.autocenter.rest_speed", ac.rest_speed, 0.005, 1.0),
        ("hands.cursor.autocenter.edge_margin", ac.edge_margin, 0.0, 0.3),
        ("hands.cursor.autocenter.edge_return_speed", ac.edge_return_speed, 0.05, 5.0),
        ("hands.cursor.head.min_cutoff", hd.min_cutoff, 0.05, 10.0),
        ("hands.cursor.head.beta", hd.beta, 0.0, 2.0),
        ("hands.cursor.head.pointer_gain", hd.pointer_gain, 0.001, 0.1),
        ("hands.cursor.head.pointer_dead_zone", hd.pointer_dead_zone, 0.0, 3.0),
        ("hands.cursor.head.pointer_acceleration", hd.pointer_acceleration, 1.0, 3.0),
        ("hands.cursor.head.coarse_cols", hd.coarse_cols, 1, 8),
        ("hands.cursor.head.coarse_rows", hd.coarse_rows, 1, 8),
        ("hands.cursor.head.coarse_yaw_span", hd.coarse_yaw_span, 5.0, 90.0),
        ("hands.cursor.head.coarse_pitch_span", hd.coarse_pitch_span, 3.0, 60.0),
        ("hands.cursor.head.coarse_hysteresis", hd.coarse_hysteresis, 0.0, 0.49),
        ("hands.cursor.head.coarse_dwell_s", hd.coarse_dwell_s, 0.0, 3.0),
        ("hands.cursor.head.switch_margin", hd.switch_margin, 0.0, 30.0),
        ("hands.cursor.head.switch_dwell_s", hd.switch_dwell_s, 0.0, 5.0),
        ("hands.cursor.head.gate_tolerance", hd.gate_tolerance, 5.0, 90.0),
        ("gaze.model_rate_hz", gz.model_rate_hz, 1.0, 30.0),
        ("gaze.min_cutoff", gz.min_cutoff, 0.01, 10.0),
        ("gaze.beta", gz.beta, 0.0, 1.0),
        ("gaze.ridge_alpha", gz.ridge_alpha, 0.0001, 1000.0),
        ("gaze.jump_idle_s", gz.jump_idle_s, 0.1, 5.0),
        ("gaze.jump_min_px", gz.jump_min_px, 0.0, 5000.0),
        ("gaze.drift_rate", gz.drift_rate, 0.0, 1.0),
        ("gaze.drift_max_px", gz.drift_max_px, 0.0, 1000.0),
        ("gaze.drift_max_err_px", gz.drift_max_err_px, 10.0, 5000.0),
    ]


def to_dict(cfg: AppConfig) -> dict[str, Any]:
    d = {
        f.name: asdict(getattr(cfg, f.name)) if is_dataclass(getattr(cfg, f.name)) else getattr(cfg, f.name)
        for f in fields(cfg)
        if f.name not in ("bindings", "migrations")
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
