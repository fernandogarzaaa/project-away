# PyInstaller spec for Project Away (onedir; used by all three OS installers).
# Build:  pyinstaller packaging/pyinstaller/away.spec --noconfirm
#
# The installer bundles EVERY runtime feature: hand/face tracking (MediaPipe, OpenCV), dictation
# (faster-whisper + CTranslate2 + Silero VAD, PortAudio), mouse/keyboard control (pynput), tray
# (pystray), the settings window (tkinter) and on-device training (CPU torch + onnx) for
# `calibrate`, custom gestures and `train`. NVIDIA NeMo (Parakeet ASR) is deliberately excluded:
# see docs/INSTALL.md. Set AWAY_SPEC_STRICT=1 (the release workflow does) to fail the build when
# any of these packages is not installed instead of silently shipping without it.
import ctypes.util
import importlib.util
import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parents[1]  # noqa: F821  (SPECPATH is injected by PyInstaller)
APP = "ProjectAway"
STRICT = os.environ.get("AWAY_SPEC_STRICT") == "1"
HERE = ROOT / "packaging" / "pyinstaller"

datas = collect_data_files("away")  # bundled static_pose.npz / gesture_tcn.onnx
binaries = []
hidden = collect_submodules("away")
missing = []


def optional(pkg, data=True, libs=False, subs=False, hidden_mods=(), meta=False):
    # find_spec, not import: pynput/pystray fail to *import* on a headless build machine
    if importlib.util.find_spec(pkg) is None:
        print(f"[away.spec] package {pkg!r} not installed, skipping")
        missing.append(pkg)
        return
    global datas, binaries, hidden
    if data:
        datas += collect_data_files(pkg)
    if libs:
        binaries += collect_dynamic_libs(pkg)
    if subs:
        hidden += collect_submodules(pkg)
    if meta:
        datas += copy_metadata(pkg)
    hidden += list(hidden_mods)


optional("mediapipe", libs=True)  # .task graphs, C library; tasks.vision imports matplotlib (do not exclude it)
optional("cv2")
optional("faster_whisper")  # ships the Silero VAD ONNX in faster_whisper/assets/
# hf_xet: huggingface_hub imports it lazily, so PyInstaller never sees it. Without it the fallback Whisper
# download (when away.app.fastdl's parallel path fails) is a single slow HTTP stream. meta: huggingface_hub
# detects it via importlib.metadata, so the dist-info must ship too.
optional("hf_xet", libs=True, subs=True, meta=True)
optional("ctranslate2", libs=True)
optional("onnxruntime", libs=True)
optional("onnx")
optional("torch")  # hooks-contrib's torch hook collects the libs; CPU wheel only (see release.yml)
optional("sounddevice", data=False, hidden_mods=["_sounddevice"])
if sys.platform in ("win32", "darwin"):
    optional("_sounddevice_data", libs=True)  # the wheel bundles the PortAudio DLL/dylib on Windows/macOS
# pynput/pystray pick a backend at import time and fail to import without a display, so
# collect_submodules() returns nothing on a headless build machine: name the backends explicitly.
if sys.platform == "win32":
    _pynput = ["keyboard._win32", "mouse._win32", "_util.win32", "_util.win32_vks"]
    _pystray = ["_win32"]
elif sys.platform == "darwin":
    _pynput = ["keyboard._darwin", "mouse._darwin", "_util.darwin", "_util.darwin_vks"]
    _pystray = ["_darwin"]
else:
    _pynput = ["keyboard._xorg", "mouse._xorg", "_util.xorg", "_util.xorg_keysyms", "keyboard._uinput", "_util.uinput"]
    _pystray = ["_xorg", "_appindicator", "_gtk"]
optional("pynput", data=False, subs=True, hidden_mods=[f"pynput.{m}" for m in ["_info", *_pynput]])
optional("pystray", data=False, subs=True, hidden_mods=[f"pystray.{m}" for m in ["_base", "_util", *_pystray]])
optional("PIL", data=False)
optional("pyperclip", data=False)
optional("tkinter", data=False, hidden_mods=["tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox"])

if sys.platform.startswith("linux"):
    # Linux sounddevice uses the system PortAudio; bundle one too (used when the system has none,
    # e.g. the AppImage on a fresh distro) via rthook_portaudio.py
    lib = ctypes.util.find_library("portaudio")
    for cand in ("/usr/lib/x86_64-linux-gnu/libportaudio.so.2", "/usr/lib/aarch64-linux-gnu/libportaudio.so.2"):
        if os.path.exists(cand):
            binaries.append((cand, "."))
            break
    else:
        print(f"[away.spec] libportaudio.so.2 not found (find_library: {lib}); relying on the system copy")
        missing.append("libportaudio.so.2")

if STRICT and missing:
    raise SystemExit(f"[away.spec] AWAY_SPEC_STRICT=1 and these bundled features are missing: {missing}")

a = Analysis(  # noqa: F821
    [str(HERE / "away_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    runtime_hooks=[str(HERE / "rthook_portaudio.py")],
    # nemo: Parakeet is not bundled. torchvision/torchaudio/triton/tensorflow: unused and huge.
    excludes=[
        "nemo",
        "nemo_toolkit",
        "torchvision",
        "torchaudio",
        "triton",
        "tensorflow",
        "tensorboard",
        "IPython",
        "pytest",
    ],
    noarchive=False,
)

# Never ship GPU runtimes (CPU torch is installed, but a CUDA build or nvidia-* wheel can slip in and
# push a single release asset past GitHub's 2 GB limit).
CUDA_RE = re.compile(r"(^|[\\/])(nvidia[\\/]|lib(cudnn|cublas|cufft|curand|cusolver|cusparse|nccl|nvrtc|nvjitlink|"
                     r"cupti|nvToolsExt|cudart|torch_cuda|c10_cuda)|(cudnn|cublas|cufft|curand|cusolver|cusparse|"
                     r"nvrtc|cudart|nvJitLink)\d*_?\d*\.dll|torch_cuda\.dll|c10_cuda\.dll)", re.I)
before = len(a.binaries)


def _is_gpu_runtime(dest, src):
    if "ctranslate2" in (dest + src).lower():  # its CPU path may link small CUDA stubs; selftest verifies it loads
        return False
    return bool(CUDA_RE.search(dest) or CUDA_RE.search(src))


a.binaries = [b for b in a.binaries if not _is_gpu_runtime(b[0], b[1])]
print(f"[away.spec] stripped {before - len(a.binaries)} CUDA/NVIDIA binaries")

pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP,
    # Windows/macOS: windowed tray app (CLI commands open a console window on Windows, see away.app.frozen)
    console=sys.platform.startswith("linux"),
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name=APP)  # noqa: F821

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{APP}.app",
        bundle_identifier="com.fernandogarzaaa.projectaway",
        info_plist={
            "CFBundleDisplayName": "Project Away",
            "CFBundleShortVersionString": os.environ.get("AWAY_VERSION", "0.1.16"),
            "LSUIElement": True,  # menu-bar (tray) app, no Dock icon
            "NSCameraUsageDescription": "Project Away tracks your hands and face locally to control the pointer.",
            "NSMicrophoneUsageDescription": "Project Away transcribes push-to-talk dictation locally.",
            "NSAppleEventsUsageDescription": "Project Away reads the name of the active app to tune dictation.",
        },
    )
