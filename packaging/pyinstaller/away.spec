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

# v0.1.23: never bundled (GPL-3.0); downloaded on request at run time only (away.voice.natural)
NON_PERMISSIVE = ["espeakng_loader", "phonemizer", "piper", "piper_phonemize", "piper_tts", "espeak"]
NON_PERMISSIVE_FILES = ("espeak", "phonemizer", "piper_phonemize", "libpiper")
datas = collect_data_files("away")  # bundled static_pose.npz / gesture_tcn.onnx
binaries = []
hidden = collect_submodules("away")
missing = []


def optional(pkg, data=True, libs=False, subs=False, hidden_mods=(), meta=False, skip_subs=()):
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
        # skip_subs: subpackages that must not even be imported while collecting (they exit or need extras)
        hidden += collect_submodules(
            pkg, filter=lambda name: not any(name == s or name.startswith(s + ".") for s in skip_subs)
        )
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
# v0.1.20 Smart cleanup: Settings' Download fetches a pinned llama-cpp-python wheel (away.voice.engine) into the
# data folder; the pure-Python packages it imports must already be in the bundle (selftest: cleanup-runtime).
optional("diskcache", data=False, hidden_mods=["diskcache"])
optional("jinja2", data=False, subs=True)
optional("markupsafe", data=False, subs=True)
optional("typing_extensions", data=False)
# v0.1.20 Agent Link: the MCP SDK + its HTTP stack (uvicorn picks protocol/loop modules by name at run time),
# QR codes, the OS keyring, and UI Automation on Windows. All pure Python (no new DLLs).
# mcp.cli (the `mcp` developer CLI) calls sys.exit(1) on import without typer (the mcp[cli] extra), which killed
# PyInstaller's submodule scan; Project Away never uses it.
optional("mcp", data=False, subs=True, meta=True, skip_subs=("mcp.cli",))
optional("uvicorn", data=False, subs=True)
optional("starlette", data=False, subs=True)
optional("sse_starlette", data=False, subs=True)
optional("pydantic", data=False, subs=True, meta=True)
optional("pydantic_settings", data=False, subs=True, meta=True)
optional("qrcode", data=False, subs=True)
optional("keyring", data=False, subs=True, meta=True)
if sys.platform == "win32":
    optional("uiautomation", data=True, subs=True)
if sys.platform == "win32":
    # v0.1.19: the WinRT MediaCapture camera backend (away.multicam.winrt_capture imports pywinrt lazily, so
    # PyInstaller never sees it). "winrt" is a namespace package: name every module and bundle its .pyd/.dll.
    _winrt_ns = [
        "winrt.system",
        "winrt.runtime",
        "winrt.windows.foundation",
        "winrt.windows.foundation.collections",
        "winrt.windows.devices.enumeration",
        "winrt.windows.media",
        "winrt.windows.media.capture",
        "winrt.windows.media.capture.frames",
        "winrt.windows.media.mediaproperties",
        "winrt.windows.graphics.imaging",
        "winrt.windows.storage.streams",
    ]
    optional("winrt", data=False, libs=True, subs=True, hidden_mods=_winrt_ns + ["winrt._winrt", "typing_extensions"])
    # v0.1.20 smart pointer: UI Automation through comtypes (imported lazily by away.pointer.uia). comtypes writes
    # its generated UIAutomationClient wrapper to a temp folder when frozen, so only the package itself ships.
    optional("comtypes", data=False, subs=True, hidden_mods=["comtypes.client", "comtypes.automation"])
    # v0.1.22: other apps' audio meters (away.voice.other_audio imports pycaw lazily); psutil names the processes
    optional("pycaw", data=False, subs=True)
    optional("psutil", data=False, subs=True)
    _winrt_spec = importlib.util.find_spec("winrt._winrt")
    if _winrt_spec is not None and _winrt_spec.origin:
        _old_msvcp = Path(_winrt_spec.origin).parent / "msvcp140.dll"
        if _old_msvcp.exists():
            # PyInstaller imports every collected package in one subprocess to find DLLs; with pywinrt's old
            # msvcp140.dll loaded first, importing onnxruntime there crashes (0xC0000005). release.yml deletes it.
            raise SystemExit(
                f"[away.spec] delete {_old_msvcp} first (pywinrt's old msvcp140.dll crashes onnxruntime; the system "
                "VC++ 2022 copy is used instead)"
            )

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
        # v0.1.23 licensing: GPL voice pieces are never bundled. espeak-ng (GPL-3.0, via espeakng_loader) and
        # phonemizer (GPL-3.0) are an opt-in runtime download (Settings > Voice & input > Voice > Download);
        # piper-tts / piper_phonemize (GPL) are not used at all (Piper voices run directly on onnxruntime).
        *NON_PERMISSIVE,
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


def _is_winrt_msvcp(dest):
    # v0.1.19: winrt-runtime ships an old msvcp140.dll (14.29) in winrt/. Windows keeps the first msvcp140.dll a
    # process loads, and onnxruntime needs >= 14.40, so pywinrt's copy loaded first crashes onnxruntime (access
    # violation). Ship only the top-level (system, VC++ 2022) copy; away.multicam.winrt_capture also preloads it.
    parts = re.split(r"[\\/]", dest.lower())
    return len(parts) >= 2 and parts[-2] == "winrt" and parts[-1] == "msvcp140.dll"


before = len(a.binaries)
a.binaries = [b for b in a.binaries if not _is_winrt_msvcp(b[0])]
print(f"[away.spec] dropped {before - len(a.binaries)} pywinrt msvcp140.dll copies")

def _is_non_permissive(dest, src):
    low = (dest + " " + src).lower()
    return any(w in low for w in NON_PERMISSIVE_FILES)


before = len(a.binaries) + len(a.datas)
a.binaries = [b for b in a.binaries if not _is_non_permissive(b[0], b[1])]
a.datas = [d for d in a.datas if not _is_non_permissive(d[0], d[1])]
print(f"[away.spec] dropped {before - len(a.binaries) - len(a.datas)} GPL voice files (espeak-ng / phonemizer)")


def _windows_metadata():
    """Windows exe version resource + icon. The version resource (product name "Project Away", product version =
    the release version) is what SignPath's metadata restrictions check before signing (README.md, "Code signing policy");
    the icon is the tray icon's glyph. Neither exists on macOS/Linux builds."""
    if sys.platform != "win32":
        return None, None
    import tempfile

    from PyInstaller.utils.win32 import versioninfo as vi

    raw = os.environ.get("AWAY_VERSION") or re.search(
        r'(?m)^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ).group(1)
    nums = tuple(([int(n) for n in re.findall(r"\d+", raw.split("-")[0])[:3]] + [0, 0, 0, 0])[:4])
    strings = [
        vi.StringStruct("CompanyName", "Fernando Garza"),
        vi.StringStruct("FileDescription", "Project Away"),
        vi.StringStruct("FileVersion", raw),
        vi.StringStruct("InternalName", APP),
        vi.StringStruct("LegalCopyright", "Copyright (c) Fernando Garza. MIT License."),
        vi.StringStruct("OriginalFilename", f"{APP}.exe"),
        vi.StringStruct("ProductName", "Project Away"),
        vi.StringStruct("ProductVersion", raw),
    ]
    version = vi.VSVersionInfo(
        ffi=vi.FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[vi.StringFileInfo([vi.StringTable("040904B0", strings)]), vi.VarFileInfo([vi.VarStruct("Translation", [1033, 1200])])],
    )
    icon = None
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_msix import _icon_master  # the same drawing as the MSIX tiles

        icon = os.path.join(tempfile.mkdtemp(prefix="away-icon-"), "ProjectAway.ico")
        _icon_master(256).save(icon, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    except Exception as e:  # Pillow missing: ship without an icon rather than fail the build
        print(f"[away.spec] no exe icon: {e}")
        icon = None
    return version, icon


_win_version, _win_icon = _windows_metadata()

pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP,
    # Windows/macOS: windowed tray app (CLI commands open a console window on Windows, see away.app.frozen)
    console=sys.platform.startswith("linux"),
    icon=_win_icon,
    version=_win_version,
)
coll = COLLECT(exe, a.binaries, a.datas, name=APP)  # noqa: F821

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{APP}.app",
        bundle_identifier="com.fernandogarzaaa.projectaway",
        info_plist={
            "CFBundleDisplayName": "Project Away",
            "CFBundleShortVersionString": os.environ.get("AWAY_VERSION", "0.1.29"),
            "LSUIElement": True,  # menu-bar (tray) app, no Dock icon
            "NSCameraUsageDescription": "Project Away tracks your hands and face locally to control the pointer.",
            "NSMicrophoneUsageDescription": "Project Away transcribes push-to-talk dictation locally.",
            "NSAppleEventsUsageDescription": "Project Away reads the name of the active app to tune dictation.",
        },
    )
