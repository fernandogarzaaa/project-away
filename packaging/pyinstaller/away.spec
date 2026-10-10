# PyInstaller spec for Project Away (onedir; used by all three OS installers).
# Build:  pyinstaller packaging/pyinstaller/away.spec --noconfirm
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parents[1]  # noqa: F821  (SPECPATH is injected by PyInstaller)
APP = "ProjectAway"

datas = collect_data_files("away")  # bundled static_pose.npz / gesture_tcn.onnx
binaries = []
hidden = collect_submodules("away")


def optional(pkg, data=True, libs=False, subs=False):
    # find_spec, not import: pynput/pystray fail to *import* on a headless build machine
    if importlib.util.find_spec(pkg) is None:
        print(f"[away.spec] optional package {pkg!r} not installed, skipping")
        return
    global datas, binaries, hidden
    if data:
        datas += collect_data_files(pkg)
    if libs:
        binaries += collect_dynamic_libs(pkg)
    if subs:
        hidden += collect_submodules(pkg)


optional("mediapipe", libs=True)
optional("faster_whisper")  # ships the Silero VAD ONNX in assets/
optional("ctranslate2", libs=True)
optional("onnxruntime", libs=True)
optional("pynput", data=False, subs=True)  # backends are imported dynamically per OS
optional("pystray", data=False, subs=True)

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "pyinstaller" / "away_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["torch", "torchvision", "torchaudio", "tensorflow", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP,
    console=sys.platform.startswith("linux"),  # Windows/macOS: windowed tray app; CLI via `away` from pipx
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
            "CFBundleShortVersionString": "0.1.0",
            "LSUIElement": True,  # menu-bar (tray) app, no Dock icon
            "NSCameraUsageDescription": "Project Away tracks your hands and face locally to control the pointer.",
            "NSMicrophoneUsageDescription": "Project Away transcribes push-to-talk dictation locally.",
            "NSAppleEventsUsageDescription": "Project Away reads the name of the active app to tune dictation.",
        },
    )
