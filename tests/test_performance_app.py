"""v0.1.5 app wiring: CUDA -> CPU fallback, setup + GPU pack, `away hardware`, selftest device, QuickEdit."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from away import config as C
from away.app import frozen, gpupack, hardware, models, selftest, setup_wizard, tuning
from away.cli import main
from away.voice import asr as asrmod


# ---------------------------------------------------------------- ASR device fallback
class FakeWhisper:
    built: ClassVar[list[tuple[str, str, int, int]]] = []

    def __init__(self, model, device, compute_type, language, root, hotwords, cpu_threads, beam_size, fail=None):
        FakeWhisper.built.append((device, compute_type, cpu_threads, beam_size))
        self.device = device
        self.fail = fail

    def transcribe(self, audio, prompt=None):
        if self.device == "cuda" and self.fail:
            raise RuntimeError(self.fail)
        return ""


def factory(fail: str | None):
    def build(*a: Any) -> FakeWhisper:
        return FakeWhisper(*a, fail=fail)

    return build


def test_cuda_load_failure_falls_back_to_cpu_with_reason(monkeypatch: pytest.MonkeyPatch):
    FakeWhisper.built = []
    activated: list[int] = []
    monkeypatch.setattr(gpupack, "activate", lambda *a, **k: activated.append(1) or [])
    reasons: list[str] = []
    a = asrmod.make_asr(
        "faster-whisper",
        "large-v3-turbo",
        "cuda",
        "float16",
        cpu_threads=4,
        beam_size=5,
        on_fallback=reasons.append,
        factory=factory("Library cublas64_12.dll is not found or cannot be loaded"),
    )
    assert activated == [1]
    assert FakeWhisper.built == [("cuda", "float16", 4, 5), ("cpu", "int8", 4, 5)]
    assert a.device == "cpu"
    assert "away gpu install" in reasons[0] and "cublas64_12.dll" in reasons[0]


def test_cuda_success_warms_up_once(monkeypatch: pytest.MonkeyPatch):
    FakeWhisper.built = []
    monkeypatch.setattr(gpupack, "activate", lambda *a, **k: [])
    reasons: list[str] = []
    a = asrmod.make_asr("faster-whisper", "small", "cuda", "int8", on_fallback=reasons.append, factory=factory(None))
    assert a.device == "cuda" and reasons == [] and FakeWhisper.built == [("cuda", "int8", 0, 5)]


def test_cpu_device_never_touches_cuda(monkeypatch: pytest.MonkeyPatch):
    FakeWhisper.built = []
    monkeypatch.setattr(gpupack, "activate", lambda *a, **k: pytest.fail("no CUDA on cpu"))
    asrmod.make_asr("faster-whisper", "base", "cpu", "int8", cpu_threads=3, beam_size=1, factory=factory(None))
    assert FakeWhisper.built == [("cpu", "int8", 3, 1)]


@pytest.mark.parametrize(
    "msg,needle",
    [
        ("Library cublas64_12.dll is not found", "away gpu install"),
        ("CUDA failed with error out of memory", "smaller model"),
        ("CUDA driver version is insufficient for CUDA runtime version", "update it"),
        ("no CUDA-capable device is detected", "no usable CUDA device"),
        ("something else", "CUDA failed to start"),
    ],
)
def test_cuda_failure_reasons(msg: str, needle: str):
    r = asrmod.cuda_failure_reason(RuntimeError(msg))
    assert needle in r and r.startswith("GPU unavailable") and "using the CPU" in r


def test_whisper_transcribe_passes_beam_size():
    seen: dict[str, Any] = {}

    class M:
        def transcribe(
            self,
            audio,
            language=None,
            initial_prompt=None,
            vad_filter=False,
            beam_size=5,
            condition_on_previous_text=False,
        ):
            seen["beam"] = beam_size
            return [], None

    asrmod.whisper_transcribe(M(), np.zeros(10, dtype=np.float32), None, None, None, beam_size=2)
    assert seen["beam"] == 2


# ---------------------------------------------------------------- setup wizard + GPU pack
def rtx_profile(pack: str = "recommended") -> tuning.Profile:
    return tuning.Profile(
        device="cpu", asr_model="small", gpu="NVIDIA GeForce RTX 2060", gpu_capable=True, gpu_pack=pack
    )


def _dl(mf, d, out):
    d.mkdir(parents=True, exist_ok=True)
    (d / mf.name).write_bytes(b"x")
    return d / mf.name


def test_setup_installs_gpu_pack_then_uses_the_gpu_profile(monkeypatch: pytest.MonkeyPatch):
    got: list[str] = []
    monkeypatch.setattr(models, "download_asr", lambda m, d, out, **k: got.append(m))
    profiles = iter([rtx_profile(), tuning.Profile(device="cuda", compute_type="float16", asr_model="large-v3-turbo")])
    installs: list[int] = []
    rep = setup_wizard.run(
        yes=True,
        skip_probe=True,
        out=lambda s: None,
        downloader=_dl,
        profiler=lambda cfg: next(profiles),
        gpu_installer=lambda **k: installs.append(k["connections"]) or Path("x"),
    )
    assert installs == [8] and rep.models["gpu_pack"] == "ok"
    assert got == ["large-v3-turbo"]  # the model picked after the GPU became usable
    assert C.load()[0].voice.asr_model == "auto"


def test_setup_no_gpu_flag_and_cpu_mode_skip_the_pack(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(models, "download_asr", lambda m, d, out, **k: None)
    boom = lambda **k: pytest.fail("must not install")  # noqa: E731
    rep = setup_wizard.run(
        yes=True,
        skip_probe=True,
        out=lambda s: None,
        downloader=_dl,
        profiler=lambda c: rtx_profile(),
        gpu=False,
        gpu_installer=boom,
    )
    assert "gpu_pack" not in rep.models
    cfg = C.AppConfig()
    cfg.performance.mode = "cpu"
    C.save(cfg)
    rep = setup_wizard.run(
        yes=True,
        skip_probe=True,
        out=lambda s: None,
        downloader=_dl,
        profiler=lambda c: rtx_profile(),
        gpu_installer=boom,
    )
    assert "gpu_pack" not in rep.models


def test_setup_gpu_pack_failure_does_not_fail_setup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(models, "download_asr", lambda m, d, out, **k: None)

    def bad(**k):
        raise OSError("pypi down")

    rep = setup_wizard.run(
        yes=True,
        skip_probe=True,
        out=lambda s: None,
        downloader=_dl,
        profiler=lambda c: rtx_profile(),
        gpu_installer=bad,
    )
    assert rep.models["gpu_pack"].startswith("not installed") and rep.models["asr"] == "ok"
    assert any("away gpu install" in w for w in rep.warnings)
    assert not any(v.startswith("failed") for v in rep.models.values())


def test_setup_on_pip_prints_instructions_instead_of_downloading(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(models, "download_asr", lambda m, d, out, **k: None)
    lines: list[str] = []
    rep = setup_wizard.run(
        yes=True, skip_probe=True, out=lines.append, downloader=_dl, profiler=lambda c: rtx_profile()
    )
    assert rep.models["gpu_pack"] == "pip instructions shown"
    assert any("pip install nvidia-cublas-cu12" in s for s in lines)


def test_setup_survives_detection_failure(monkeypatch: pytest.MonkeyPatch):
    got: list[str] = []
    monkeypatch.setattr(models, "download_asr", lambda m, d, out, **k: got.append(m))

    def broken(cfg):
        raise RuntimeError("wmi exploded")

    lines: list[str] = []
    setup_wizard.run(yes=True, skip_probe=True, out=lines.append, downloader=_dl, profiler=broken)
    assert got == ["small"] and any("hardware detection failed" in s for s in lines)


def test_setup_explicit_model_is_saved(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(models, "download_asr", lambda m, d, out, **k: None)
    setup_wizard.run(yes=True, asr_model="tiny", skip_probe=True, out=lambda s: None, downloader=_dl)
    assert C.load()[0].voice.asr_model == "tiny"


def test_first_launch_passes_explicit_model(monkeypatch: pytest.MonkeyPatch):
    cfg = C.AppConfig()
    cfg.voice.asr_model = "base"
    calls: list[list[str]] = []
    frozen.first_launch(lambda argv: calls.append(argv) or 0, cfg, notifier=lambda t, m: None)
    assert calls == [["setup", "--yes", "--skip-probe", "--asr-model", "base"]]


# ---------------------------------------------------------------- CLI / selftest
def test_cli_hardware_text_and_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, fixed_hardware
):
    monkeypatch.setattr(hardware, "detect", lambda: fixed_hardware)
    assert main(["hardware"]) == 0
    out = capsys.readouterr().out
    assert "Profile (cpu-mid): small on CPU (int8)" in out and "Why:" in out
    assert main(["hardware", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["profile"]["asr_model"] == "small" and data["hardware"]["cores_physical"] == 8
    assert data["gpu_pack"]["installed"] is False


def test_cli_hardware_suggests_gpu_pack(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    g = hardware.GPU("nvidia", "NVIDIA GeForce RTX 2060", 6144, 5000, (7, 5), "560.94", "12.6", "nvml")
    hw = hardware.Hardware(os="win32", arch="amd64", cores_physical=6, ram_mb=16384, gpus=[g])
    monkeypatch.setattr(hardware, "detect", lambda: hw)
    assert main(["hardware"]) == 0
    out = capsys.readouterr().out
    assert "GPU pack: recommended" in out and "pip install nvidia-cublas-cu12" in out


def test_selftest_reports_the_speech_device(monkeypatch: pytest.MonkeyPatch, fixed_hardware, tmp_path: Path):
    monkeypatch.setattr(hardware, "detect", lambda: fixed_hardware)
    check = next(c for c in selftest.CHECKS if c.name == "hardware")
    out = tmp_path / "s.json"
    assert selftest.main(str(out), checks=[check]) == 0
    rep = json.loads(out.read_text())
    assert rep["asr_device"] == "cpu" and rep["checks"]["hardware"]["detail"].startswith("asr device=cpu (int8, small)")


def test_console_commands_include_new_ones():
    assert frozen.wants_console(["hardware"]) and frozen.wants_console(["gpu", "status"])


# ---------------------------------------------------------------- QuickEdit
class FakeKernel32:
    def __init__(self, mode: int, handle: int = 44, get_ok: bool = True, set_ok: bool = True) -> None:
        self.mode, self.handle, self.get_ok, self.set_ok = mode, handle, get_ok, set_ok
        self.set_calls: list[int] = []
        self.CreateFileW = self._create  # attribute so restype can be assigned like on a ctypes function

    def _create(self, name, access, share, sec, disp, flags, tmpl):
        assert name == "CONIN$" and disp == frozen.OPEN_EXISTING
        return self.handle

    def GetConsoleMode(self, handle, ref):
        ref._obj.value = self.mode
        return self.get_ok

    def SetConsoleMode(self, handle, mode):
        value = mode.value if isinstance(mode, ctypes.c_uint32) else int(mode)
        self.set_calls.append(value)
        return self.set_ok


def test_quick_edit_bits():
    assert frozen.quick_edit_off(0x01F7) == (0x01F7 & ~0x40) | 0x80
    assert frozen.quick_edit_off(0x0040) == 0x0080
    assert frozen.quick_edit_off(0x0080) == 0x0080


def test_disable_quick_edit_sets_mode_and_restores_at_exit():
    k = FakeKernel32(0x01F7)  # typical cmd.exe default: QuickEdit + insert + extended flags ...
    assert frozen.disable_quick_edit(k, platform="win32") is True
    assert k.set_calls == [0x01B7]  # QuickEdit cleared, ENABLE_EXTENDED_FLAGS kept
    frozen._restore_console_mode()
    assert k.set_calls[-1] == 0x01F7


def test_disable_quick_edit_noops_and_failures():
    assert frozen.disable_quick_edit(FakeKernel32(0x01F7), platform="linux") is False
    assert frozen.disable_quick_edit(FakeKernel32(0x01F7, handle=-1), platform="win32") is False
    assert frozen.disable_quick_edit(FakeKernel32(0x01F7, get_ok=False), platform="win32") is False
    assert frozen.disable_quick_edit(FakeKernel32(0x01F7, set_ok=False), platform="win32") is False
    already = FakeKernel32(0x0080)
    assert frozen.disable_quick_edit(already, platform="win32") is True and already.set_calls == []

    class Exploding:
        def __getattr__(self, name):
            raise OSError("no console")

    assert frozen.disable_quick_edit(Exploding(), platform="win32") is False


def test_ensure_streams_disables_quick_edit_for_frozen_console_commands(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    monkeypatch.setattr(frozen, "is_frozen", lambda: True)
    monkeypatch.setattr(
        frozen,
        "disable_quick_edit",
        lambda kernel32=None, platform=None, restore_at_exit=True: calls.append(platform) or True,
    )
    assert frozen.ensure_streams(["setup", "--yes"], platform="win32") == "tty"
    assert calls == ["win32"]
    frozen.ensure_streams(["tray"], platform="win32")
    frozen.ensure_streams(["setup"], platform="linux")
    assert calls == ["win32"]
