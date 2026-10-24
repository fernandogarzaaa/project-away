"""macOS bundle: PyAV is left out and replaced by an empty ``av`` (packaging/pyinstaller/rthook_stub_av.py).

Two copies of FFmpeg's libavdevice (one in PyAV, one in OpenCV) define the same Objective-C classes, which macOS
warns "may cause spurious casting failures and mysterious crashes". Neither copy can be deleted (every PyAV module
and cv2.abi3.so is linked against its own), so PyAV, which faster-whisper only uses to decode audio files, is dropped.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "packaging" / "pyinstaller" / "rthook_stub_av.py"


def _run_hook() -> None:
    runpy.run_path(str(HOOK), run_name="rthook_stub_av")


def test_the_hook_installs_an_empty_av_when_pyav_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "av", None)  # `import av` raises ImportError, as in the bundle
    _run_hook()
    av = sys.modules["av"]
    assert av is not None and av.__version__ == "0+not-bundled"
    import av as imported  # the import statement now succeeds, which is all faster-whisper needs at import time

    assert imported is av
    assert importlib.util.find_spec("av") is not None  # a probe gets an answer, not a ValueError
    assert not hasattr(av, "open") and getattr(av, "open", None) is None  # probes behave
    with pytest.raises(AttributeError, match="not included in the macOS build"):
        av.open("some.wav")  # real use says what is missing instead of a vague failure
    with pytest.raises(AttributeError, match=r"av\.audio"):
        av.audio.resampler.AudioResampler  # noqa: B018  (the chain faster_whisper.audio.decode_audio uses)


def test_a_real_pyav_is_never_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    real = types.ModuleType("av")
    real.marker = "the real one"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "av", real)
    _run_hook()
    assert sys.modules["av"] is real


def test_the_spec_leaves_pyav_out_on_macos_only_and_uses_the_hook() -> None:
    spec = (ROOT / "packaging" / "pyinstaller" / "away.spec").read_text(encoding="utf-8")
    assert '_mac = sys.platform == "darwin"' in spec
    assert '*(["av"] if _mac else [])' in spec  # excluded on macOS only: Windows and Linux are unchanged
    assert 'str(HERE / "rthook_stub_av.py")' in spec and "if _mac else []" in spec
    assert HOOK.is_file()


def test_the_release_workflow_checks_for_exactly_one_libavdevice() -> None:
    yml = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    mac = yml[yml.index("\n  macos:") : yml.index("\n  publish:")]
    step = mac.index("Exactly one libavdevice")
    assert mac.index("pyinstaller packaging/pyinstaller/away.spec") < step < mac.index("Smoke-test the frozen app")
    assert "-type f -name 'libavdevice*'" in mac and "Contents/Frameworks/av" in mac
