"""``away setup``: first-run wizard.

1. create the user data/config folders
2. download MediaPipe .task models (sha256-verified)
3. detect the hardware; with a usable NVIDIA GPU, offer the GPU pack (cuBLAS, frozen apps only)
4. download the ASR model the hardware profile picks (or ``--asr-model``)
5. probe camera and microphone (and explain OS permission dialogs)
6. write config.yaml
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from away import config as cfgmod
from away import paths
from away.app import models


@dataclass
class SetupReport:
    models: dict[str, str] = field(default_factory=dict)
    probes: dict[str, str] = field(default_factory=dict)
    config_path: str = ""
    warnings: list[str] = field(default_factory=list)


PERMISSION_HELP = {
    "darwin": (
        "macOS asks for permission the first time each sensor is used. If a prompt was dismissed:\n"
        "  System Settings > Privacy & Security > Camera / Microphone: enable Terminal (or Project Away).\n"
        "  System Settings > Privacy & Security > Accessibility AND Input Monitoring: enable it too,\n"
        "  otherwise mouse control, hotkeys and typing are silently ignored."
    ),
    "win32": (
        "Windows: Settings > Privacy & security > Camera / Microphone > allow desktop apps.\n"
        "Typing into elevated (admin) windows requires running Project Away elevated too."
    ),
    "linux": (
        "Linux: your user needs access to /dev/video* (group 'video') and a PulseAudio/PipeWire mic.\n"
        "Mouse/keyboard control works on X11. On Wayland, pynput cannot move the pointer or read global\n"
        "hotkeys for most compositors: log into an X11 session (e.g. 'Ubuntu on Xorg')."
    ),
}


def ask(prompt: str, default: bool, assume: bool | None) -> bool:
    """Yes/no question. Never blocks without a user: ``assume`` (``--yes``), no stdin (a windowed frozen exe
    has ``sys.stdin is None``), a closed stdin or EOF all return the default."""
    if assume is not None:
        return assume
    if sys.stdin is None or getattr(sys.stdin, "closed", False):
        return default
    try:
        ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    except (EOFError, RuntimeError, OSError):  # RuntimeError: "input(): lost sys.stdin"
        return default
    return default if not ans else ans in ("y", "yes")


OPTIONAL_GAZE_NOTICE = (
    "  Optional, not downloaded: learned gaze networks (L2CS-Net / MobileGaze) for eye tracking. Their weights are\n"
    "  research-only (trained on Gaze360), so they are an explicit opt-in: Settings > Performance > Models, or\n"
    "  `away gaze model install mobilegaze --accept-license`. Eye tracking also works without them (iris landmarks)."
)


def run(
    yes: bool = False,
    asr_model: str | None = None,
    skip_asr: bool = False,
    skip_probe: bool = False,
    out: Callable[[str], None] = print,
    downloader: Callable[..., Path] = models.download,
    connections: int = 8,
    gpu: bool | None = None,
    gpu_installer: Callable[..., Path] | None = None,
    profiler: Callable[[cfgmod.AppConfig], object] | None = None,
) -> SetupReport:
    rep = SetupReport()
    assume = True if yes else None
    cfg, unknown = cfgmod.load()
    if unknown:
        rep.warnings.append("ignored unknown config keys: " + ", ".join(unknown))
    mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
    out(
        "Project Away setup: the bridge between you and your AI agents, controlled by your hands, face and voice.\n"
        "  (Connect your AI later in Settings > Agent Link.)\n"
        f"  data:   {paths.data_dir()}\n  config: {paths.config_file()}"
    )

    out("\n[1/4] MediaPipe models")
    for mf in models.MEDIAPIPE_MODELS:
        try:
            downloader(mf, mdir, out)
            rep.models[mf.name] = "ok"
        except Exception as exc:
            rep.models[mf.name] = f"failed: {exc}"
            rep.warnings.append(f"{mf.name} download failed: {exc}")

    out(OPTIONAL_GAZE_NOTICE)

    out("\n[2/4] Hardware and speech recognition model")
    prof = _profile(cfg, profiler, out)
    if prof is not None and not skip_asr and gpu is not False:
        _offer_gpu_pack(cfg, prof, rep, out, gpu, assume, connections, gpu_installer)
        if rep.models.get("gpu_pack") == "ok":
            prof = _profile(cfg, profiler, out, quiet=True) or prof
    if skip_asr:
        rep.models["asr"] = "skipped"
    else:
        auto_model = getattr(prof, "asr_model", None) or "small"
        model = asr_model or (cfg.voice.asr_model if cfg.voice.asr_model != "auto" else auto_model)
        size = models.ASR_SIZES_MB.get(model, 0)
        # v0.1.35: on Apple Silicon with mlx-whisper, the MLX weights (Whisper on the Apple GPU) instead
        from away.app.frozen import asr_engine

        engine = "mlx-whisper" if asr_engine(cfg, model) == "mlx-whisper" else "faster-whisper"
        if engine == "mlx-whisper":
            size = models.MLX_MODELS[model].size_mb
        hint = "" if model in ("small", "base", "tiny") else " 'small' is ~485 MB for slower PCs or connections."
        if ask(f"  Download Whisper '{model}' (~{size} MB)?{hint}", True, assume):
            try:
                models.download_asr(model, mdir, out, connections=max(1, connections), engine=engine)
                if asr_model:  # an explicit choice is remembered; otherwise the config stays "auto"
                    cfg.voice.asr_model = model
                rep.models["asr"] = "ok"
            except Exception as exc:
                rep.models["asr"] = f"failed: {exc}"
                rep.warnings.append(f"ASR download failed ({exc}); it will download on first dictation")
        else:
            rep.models["asr"] = "deferred (downloads on first use)"

    out("\n[3/4] Camera and microphone")
    if skip_probe:
        rep.probes["camera"] = rep.probes["microphone"] = "skipped"
    else:
        from away.hands.tracker import probe_camera
        from away.voice.dictation import probe_microphone

        ok, msg = probe_camera(cfg.hands.camera_index, hint=cfg.hands.camera_hint)
        rep.probes["camera"] = msg
        out(f"  {msg}")
        ok2, msg2 = probe_microphone()
        rep.probes["microphone"] = msg2
        out(f"  {msg2}")
        if not (ok and ok2):
            rep.warnings.append("camera or microphone check failed; see permission help above")
    plat = "linux" if sys.platform.startswith("linux") else sys.platform
    out("  " + PERMISSION_HELP.get(plat, "").replace("\n", "\n  "))

    out("\n[4/4] Config")
    cfg.models_dir = str(mdir)
    rep.config_path = str(cfgmod.save(cfg))
    out(f"  wrote {rep.config_path}")
    for w in rep.warnings:
        out(f"  WARNING: {w}")
    out("\nDone. Start with:  away run   (or the tray app: away tray)")
    out("New here? `away tour` walks you through a 2-minute gesture tutorial.")
    return rep


def _profile(
    cfg: cfgmod.AppConfig,
    profiler: Callable[[cfgmod.AppConfig], object] | None,
    out: Callable[[str], None],
    quiet: bool = False,
) -> object | None:
    try:
        if profiler is not None:
            prof = profiler(cfg)
        else:
            from away.app.performance import current_profile

            prof = current_profile(cfg)
    except Exception as exc:  # detection must never break setup
        out(f"  hardware detection failed ({exc}); using defaults")
        return None
    if not quiet:
        summary = getattr(prof, "summary", None)
        if callable(summary):
            out(f"  detected profile: {summary()}")
        if getattr(prof, "gpu", None):
            out(f"  GPU: {getattr(prof, 'gpu', '')}")
    return prof


def _offer_gpu_pack(
    cfg: cfgmod.AppConfig,
    prof: object,
    rep: SetupReport,
    out: Callable[[str], None],
    gpu: bool | None,
    assume: bool | None,
    connections: int,
    installer: Callable[..., Path] | None,
) -> None:
    """With a usable NVIDIA GPU, mode auto/gpu and no cuBLAS: frozen apps offer the GPU pack, pip installs
    get the pip line."""
    from away.app import gpupack
    from away.app.frozen import is_frozen

    if cfg.performance.mode == "cpu" or cfg.voice.device == "cpu":
        return
    if not getattr(prof, "gpu_capable", False) or getattr(prof, "gpu_pack", "") != "recommended":
        return
    if not is_frozen() and installer is None:
        out("  NVIDIA GPU found, but the CUDA library is not installed. " + gpupack.pip_instructions())
        rep.models["gpu_pack"] = "pip instructions shown"
        return
    question = (
        f"  Use the NVIDIA GPU ({getattr(prof, 'gpu', '')}) for speech recognition? "
        "Downloads the GPU pack (cuBLAS, ~570 MB, once)."
    )
    if not ask(question, True, True if gpu else assume):
        rep.models["gpu_pack"] = "declined"
        return
    try:
        (installer or gpupack.install)(progress=out, connections=max(1, connections))
        rep.models["gpu_pack"] = "ok"
    except Exception as exc:
        rep.models["gpu_pack"] = f"not installed ({exc})"
        rep.warnings.append(f"GPU pack download failed ({exc}); speech runs on the CPU. Retry: away gpu install")
