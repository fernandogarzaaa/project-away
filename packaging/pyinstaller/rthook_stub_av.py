"""macOS only: stand in for PyAV ("av"), which away.spec leaves out of the bundle.

Why it is left out: on macOS two copies of FFmpeg's ``libavdevice`` end up in the app, one inside ``av`` (PyAV)
and one inside ``cv2`` (OpenCV). Both define the Objective-C classes ``AVFFrameReceiver`` and ``AVFAudioReceiver``,
and the Objective-C runtime warns ("may cause spurious casting failures and mysterious crashes") when the second
one loads. Neither copy can simply be deleted: every PyAV module and ``cv2.abi3.so`` is linked against its own, so
the import would fail. PyAV is the one the app can do without.

faster-whisper imports ``av`` at import time (``faster_whisper/audio.py``) but only uses it in ``decode_audio``,
which reads audio *files*. Project Away hands it audio as numpy arrays, so that function is never called.
This hook installs an empty ``av`` so the import succeeds; using it for real (``decode_audio("file.wav")``)
raises an error that says what is missing. A build that does contain PyAV is left alone.
"""

import importlib.machinery
import sys
import types


def install() -> None:
    try:
        import av  # noqa: F401  (a real PyAV in the bundle or on the path wins)

        return
    except ImportError:
        pass

    stub = types.ModuleType("av")
    stub.__doc__ = "Stand-in for PyAV: not included in the macOS build of Project Away (see rthook_stub_av.py)."
    stub.__version__ = "0+not-bundled"  # type: ignore[attr-defined]
    # a spec, so a library that probes with importlib.util.find_spec("av") gets an answer instead of a ValueError
    stub.__spec__ = importlib.machinery.ModuleSpec("av", None)

    def __getattr__(name: str):  # PEP 562 module attribute hook
        # AttributeError (not RuntimeError) so hasattr()/getattr(..., default) probes behave
        raise AttributeError(
            f"PyAV ('av.{name}') is not included in the macOS build of Project Away, so audio files cannot be "
            "decoded. Project Away passes audio to Whisper as arrays and never needs this."
        )

    stub.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules["av"] = stub


install()
