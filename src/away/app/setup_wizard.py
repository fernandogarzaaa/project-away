"""``away setup``: first-run wizard.

1. create the user data/config folders
2. download MediaPipe .task models (sha256-verified)
3. download the ASR model (optional, large)
4. probe camera and microphone (and explain OS permission dialogs)
5. write config.yaml
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


def run(
    yes: bool = False,
    asr_model: str | None = None,
    skip_asr: bool = False,
    skip_probe: bool = False,
    out: Callable[[str], None] = print,
    downloader: Callable[..., Path] = models.download,
    connections: int = 8,
) -> SetupReport:
    rep = SetupReport()
    assume = True if yes else None
    cfg, unknown = cfgmod.load()
    if unknown:
        rep.warnings.append("ignored unknown config keys: " + ", ".join(unknown))
    mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
    out(f"Project Away setup\n  data:   {paths.data_dir()}\n  config: {paths.config_file()}")

    out("\n[1/4] MediaPipe models")
    for mf in models.MEDIAPIPE_MODELS:
        try:
            downloader(mf, mdir, out)
            rep.models[mf.name] = "ok"
        except Exception as exc:
            rep.models[mf.name] = f"failed: {exc}"
            rep.warnings.append(f"{mf.name} download failed: {exc}")

    out("\n[2/4] Speech recognition model")
    if skip_asr:
        rep.models["asr"] = "skipped"
    else:
        model = asr_model or cfg.voice.asr_model
        size = models.ASR_SIZES_MB.get(model, 0)
        if ask(f"  Download Whisper '{model}' (~{size} MB)? 'small' is ~485 MB for slower PCs.", True, assume):
            try:
                models.download_asr(model, mdir, out, connections=max(1, connections))
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

        ok, msg = probe_camera(cfg.hands.camera_index)
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
    return rep
