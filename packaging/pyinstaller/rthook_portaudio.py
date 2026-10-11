# PyInstaller runtime hook (Linux): sounddevice finds PortAudio with ctypes.util.find_library("portaudio"),
# which only searches system paths. Prefer the system library (matches the user's ALSA/PulseAudio setup)
# and fall back to the copy bundled in the app, so the AppImage works without `apt install libportaudio2`.
import ctypes.util
import os
import sys

if sys.platform.startswith("linux"):
    _orig_find_library = ctypes.util.find_library
    _bundled = os.path.join(getattr(sys, "_MEIPASS", ""), "libportaudio.so.2")

    def _find_library(name):
        found = _orig_find_library(name)
        if found is None and name == "portaudio" and os.path.exists(_bundled):
            return _bundled
        return found

    ctypes.util.find_library = _find_library
