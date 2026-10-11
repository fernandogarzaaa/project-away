"""``away selftest`` (hidden): prove every bundled feature loads inside the installed app.

Used by the release workflow on each OS runner against the frozen PyInstaller build, so a release
fails when a feature (torch, tkinter, MediaPipe, faster-whisper, PortAudio...) silently drops out of
the bundle. Each check imports the module *and* touches its native library or data files, because a
module can import fine while its ``.so``/``.dll`` or model asset is missing.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    feature: str
    fn: Callable[[], str]
    needs_display: bool = False  # pynput/pystray backends connect to the display server on import


def _numpy() -> str:
    import numpy as np
    import scipy.signal  # noqa: F401
    import yaml

    return f"numpy {np.__version__}, PyYAML {yaml.__version__}"


def _torch() -> str:
    import torch

    x = torch.ones(3, 4) @ torch.ones(4, 2)
    assert float(x.sum()) == 24.0
    cuda = torch.backends.cuda.is_built()
    return f"torch {torch.__version__} (CUDA built: {cuda})"


def _onnx() -> str:
    import onnx

    return f"onnx {onnx.__version__}"


def _onnxruntime() -> str:
    import numpy as np
    import onnxruntime as ort

    from away.hands.runtime import bundled_model

    path = bundled_model("gesture_tcn.onnx")
    assert path is not None, "bundled gesture_tcn.onnx missing"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    sess.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})
    return f"onnxruntime {ort.__version__}, gesture_tcn.onnx runs"


def _mediapipe() -> str:
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    assert hasattr(vision, "HandLandmarker") and hasattr(vision, "FaceLandmarker")
    return f"mediapipe {mp.__version__}"


def _opencv() -> str:
    import cv2
    import numpy as np

    cv2.GaussianBlur(np.zeros((8, 8), np.uint8), (3, 3), 0)
    return f"opencv {cv2.__version__}"


def _faster_whisper() -> str:
    import ctranslate2
    import faster_whisper
    import numpy as np
    from faster_whisper.utils import get_assets_path
    from faster_whisper.vad import get_speech_timestamps

    assets = sorted(p.name for p in Path(get_assets_path()).glob("*.onnx"))
    assert any("silero" in a for a in assets), f"Silero VAD asset missing (assets: {assets})"
    get_speech_timestamps(np.zeros(16000, dtype=np.float32))  # loads the Silero ONNX via onnxruntime
    ctranslate2.get_supported_compute_types("cpu")  # loads the native CTranslate2 library
    return f"faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}, VAD {assets}"


def _hf_xet() -> str:
    import hf_xet
    import hf_xet.hf_xet  # noqa: F401  (the native Rust extension .so/.pyd)
    from huggingface_hub.utils import _runtime

    # huggingface_hub finds hf_xet through importlib.metadata: the dist-info must be bundled too
    assert _runtime.is_xet_available(), "huggingface_hub does not see hf_xet (missing dist-info?)"
    return f"hf_xet {_runtime.get_xet_version()}"


def _sounddevice() -> str:
    import sounddevice as sd

    return f"sounddevice {sd.__version__}, {sd.get_portaudio_version()[1]}"


def _tkinter() -> str:
    import tkinter
    import tkinter.ttk

    tcl = tkinter.Tcl()  # needs the bundled Tcl library files, not a display
    return f"Tk {tkinter.TkVersion}, Tcl {tcl.eval('info patchlevel')}"


def _pynput() -> str:
    import pynput.keyboard
    import pynput.mouse  # noqa: F401

    return "pynput ok"


def _pystray() -> str:
    import pystray  # noqa: F401
    from PIL import Image

    Image.new("RGBA", (16, 16))
    return "pystray + Pillow ok"


def _pyperclip() -> str:
    import pyperclip  # noqa: F401

    return "pyperclip ok"


def _away_data() -> str:
    from away.hands.runtime import bundled_model

    missing = [n for n in ("static_pose.npz", "gesture_tcn.onnx") if bundled_model(n) is None]
    assert not missing, f"bundled away data missing: {missing}"
    return "static_pose.npz + gesture_tcn.onnx present"


def _hardware() -> str:
    """Detection + the device speech recognition will use (no model is loaded)."""
    from away import config as cfgmod
    from away.app import gpupack, hardware, performance

    cfg, _ = cfgmod.load()
    hw = hardware.detect()
    prof = performance.current_profile(cfg, hw)
    gpus = ", ".join(g.name for g in hw.gpus) or "none"
    pack = "installed" if gpupack.installed() else "not installed"
    return (
        f"asr device={prof.device} ({prof.compute_type}, {prof.asr_model}); "
        f"{hw.cores_physical or '?'} cores, {(hw.ram_mb or 0) / 1024:.1f} GB RAM; GPUs: {gpus}; GPU pack {pack}"
    )


CHECKS: tuple[Check, ...] = (
    Check("hardware", "hardware detection + speech device", _hardware),
    Check("core", "numpy/scipy/yaml", _numpy),
    Check("away-data", "bundled gesture models", _away_data),
    Check("onnxruntime", "gesture model runtime", _onnxruntime),
    Check("mediapipe", "hand + face tracking", _mediapipe),
    Check("opencv", "camera capture", _opencv),
    Check("faster-whisper", "dictation (Whisper + Silero VAD)", _faster_whisper),
    Check("hf-xet", "fast Whisper download fallback (Xet)", _hf_xet),
    Check("sounddevice", "microphone (PortAudio)", _sounddevice),
    Check("tkinter", "settings window", _tkinter),
    Check("torch", "calibrate / custom gestures / train", _torch),
    Check("onnx", "model export (train)", _onnx),
    Check("pynput", "mouse + keyboard control", _pynput, needs_display=True),
    Check("pystray", "tray icon", _pystray, needs_display=True),
    Check("pyperclip", "paste dictation", _pyperclip),
)


def has_display(platform: str | None = None) -> bool:
    plat = platform or sys.platform
    if not plat.startswith("linux"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run(checks: Sequence[Check] | None = None, display: bool | None = None) -> dict[str, dict[str, object]]:
    checks = CHECKS if checks is None else checks
    display = has_display() if display is None else display
    results: dict[str, dict[str, object]] = {}
    for c in checks:
        if c.needs_display and not display:
            results[c.name] = {"ok": None, "feature": c.feature, "detail": "skipped: no display"}
            continue
        try:
            results[c.name] = {"ok": True, "feature": c.feature, "detail": c.fn()}
        except BaseException as exc:  # a broken native lib can raise anything, even SystemExit
            if isinstance(exc, KeyboardInterrupt):
                raise
            results[c.name] = {"ok": False, "feature": c.feature, "detail": f"{type(exc).__name__}: {exc}"}
    return results


def main(json_path: str | None = None, checks: Sequence[Check] | None = None, display: bool | None = None) -> int:
    from away import __version__

    res = run(checks, display)
    device = None
    hw_detail = str(res.get("hardware", {}).get("detail", ""))
    if hw_detail.startswith("asr device="):
        device = hw_detail.split("=", 1)[1].split(" ", 1)[0]
    report = {
        "version": __version__,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": sys.platform,
        "asr_device": device,
        "ok": all(r["ok"] is not False for r in res.values()),
        "checks": res,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if json_path:
        Path(json_path).write_text(text, encoding="utf-8")
    return 0 if report["ok"] else 1
