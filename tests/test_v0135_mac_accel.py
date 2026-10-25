"""v0.1.35 Mac-native acceleration and hardware text: hand / face / gaze tracking may use Core ML (Apple Neural Engine
/ GPU) on macOS only, and keep it per model only when it measures faster than the processor; the hardware lines and
Settings > Models speak of unified memory on Apple Silicon; the NVIDIA GPU pack is never offered on a Mac. Every
platform is injected (``platform=``), so these run the same on Windows, Linux and macOS CI."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from away import config as C
from away.accel import compute
from away.accel.settings_ui import MAC_GPU_LABEL, label_to_value, labels, status_line
from away.app import gpupack, performance, tuning
from away.app import hardware as H
from away.app import model_registry as R
from away.face import gaze_net as gn

ALL = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
MAC = ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture(autouse=True)
def _fresh_measurements() -> Any:
    compute._measured.clear()
    compute._last.clear()
    yield
    compute._measured.clear()
    compute._last.clear()


# ------------------------------------------------------------------------------------------ provider lists
def test_coreml_is_tried_on_macos_only_and_after_cuda_and_directml() -> None:
    assert compute.gpu_providers("darwin") == (*compute.GPU_PROVIDERS, "CoreMLExecutionProvider")
    for plat in ("win32", "linux"):
        assert compute.gpu_providers(plat) == ("CUDAExecutionProvider", "DmlExecutionProvider")
        assert compute.gpu_provider(MAC, plat) == ""  # a Core ML-only list means nothing off a Mac
    assert compute.gpu_provider(MAC, "darwin") == "CoreMLExecutionProvider"
    assert compute.gpu_provider(ALL, "darwin") == "CUDAExecutionProvider"
    assert compute.PROVIDER_TITLES["CoreMLExecutionProvider"] == "Apple Neural Engine / GPU (Core ML)"


def test_gaze_network_providers_follow_the_same_per_os_list() -> None:
    assert gn.providers(MAC, platform="darwin") == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert gn.providers(MAC, platform="win32") == ["CPUExecutionProvider"]
    assert gn.providers(ALL, platform="linux") == [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert gn.gpu_available(MAC, platform="darwin") and not gn.gpu_available(MAC, platform="win32")


@pytest.mark.parametrize(
    ("setting", "ready", "backend", "provider"),
    [
        ("auto", False, "mediapipe-cpu", ""),  # auto never downloads: MediaPipe until the models are there
        ("auto", True, "onnx-gpu", "CoreMLExecutionProvider"),
        ("gpu", False, "onnx-gpu", "CoreMLExecutionProvider"),  # gpu downloads the ONNX models first
        ("cpu", True, "mediapipe-cpu", ""),
    ],
)
def test_mac_choice_uses_coreml_like_the_other_gpu_providers(setting: str, ready: bool, backend: str, provider: str):
    c = compute.choose(setting, platform="darwin", providers=MAC, models_ready=ready)
    assert (c.backend, c.provider) == (backend, provider), c.reason


def test_coreml_options_ask_for_every_compute_unit_and_a_cache_under_the_data_dir(tmp_path: Path) -> None:
    opts = compute.provider_options("CoreMLExecutionProvider", tmp_path)
    assert opts["MLComputeUnits"] == "ALL"
    assert Path(opts["ModelCacheDirectory"]).is_relative_to(tmp_path)
    assert Path(opts["ModelCacheDirectory"]).is_dir()
    assert "ModelFormat" not in opts  # MLProgram fails on the palm / face models: keep the default format
    assert compute.provider_options("CUDAExecutionProvider", tmp_path) == {}
    assert compute.provider_options("CPUExecutionProvider", tmp_path) == {}


# ------------------------------------------------------------------------------------- measured sessions
class FakeSession:
    def __init__(self, providers: list[str], ms: float) -> None:
        self._providers = [p for p in providers if p != "AzureExecutionProvider"]
        self.ms = ms

    def get_providers(self) -> list[str]:
        return list(self._providers)


def fake_maker(times: dict[str, float], made: list[list[str]] | None = None, broken: bool = False) -> Any:
    """Opens a session whose first provider decides its (fake) speed."""

    def make(path: Any, providers: list[str], data_dir: Any = None, threads: int = 0) -> FakeSession:
        if made is not None:
            made.append(list(providers))
        if broken and providers[0] == "CoreMLExecutionProvider":
            raise RuntimeError("Error compiling model")
        return FakeSession(providers, times[providers[0]])

    return make


def timer(sess: FakeSession) -> float:
    return sess.ms


def test_benchmark_keeps_coreml_when_it_measures_faster(tmp_path: Path) -> None:
    make = fake_maker({"CoreMLExecutionProvider": 0.5, "CPUExecutionProvider": 3.4})
    s = compute.measured_session(tmp_path / "hand_landmark.onnx", ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                                 make=make, timer=timer)  # fmt: skip
    assert s.get_providers()[0] == "CoreMLExecutionProvider"
    (m,) = compute.measurements()
    assert m["kept"] and (m["ms"], m["cpu_ms"]) == (0.5, 3.4) and m["model"] == "hand_landmark.onnx"


def test_benchmark_drops_coreml_when_the_processor_is_faster(tmp_path: Path) -> None:
    make = fake_maker({"CoreMLExecutionProvider": 7.7, "CPUExecutionProvider": 4.3})
    s = compute.measured_session(tmp_path / "palm_detector.onnx", ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                                 make=make, timer=timer)  # fmt: skip
    assert s.get_providers() == ["CPUExecutionProvider"]
    assert compute.measurements()[0]["why"] == "slower than the processor"


def test_benchmark_decision_is_remembered_per_model(tmp_path: Path) -> None:
    made: list[list[str]] = []
    make = fake_maker({"CoreMLExecutionProvider": 9.0, "CPUExecutionProvider": 1.0}, made)
    order = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    compute.measured_session(tmp_path / "m.onnx", order, make=make, timer=timer)
    assert len(made) == 2  # Core ML and the processor, both timed
    s = compute.measured_session(tmp_path / "m.onnx", order, make=make, timer=timer)
    assert made[2:] == [["CPUExecutionProvider"]] and s.get_providers() == ["CPUExecutionProvider"]


def test_coreml_that_fails_to_compile_falls_back_to_the_processor(tmp_path: Path) -> None:
    make = fake_maker({"CPUExecutionProvider": 2.0}, broken=True)
    s = compute.measured_session(tmp_path / "m.onnx", ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                                 make=make, timer=timer)  # fmt: skip
    assert s.get_providers() == ["CPUExecutionProvider"]
    assert not compute.measurements()[0]["kept"] and "did not start" in compute.measurements()[0]["why"]


def test_cuda_and_directml_are_not_benchmarked(tmp_path: Path) -> None:
    made: list[list[str]] = []
    make = fake_maker({"CUDAExecutionProvider": 50.0, "CPUExecutionProvider": 1.0}, made)
    s = compute.measured_session(tmp_path / "m.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"],
                                 make=make, timer=timer)  # fmt: skip
    assert (
        made == [["CUDAExecutionProvider", "CPUExecutionProvider"]] and s.get_providers()[0] == "CUDAExecutionProvider"
    )
    assert compute.measurements() == []


def test_a_session_that_cannot_run_measures_as_infinitely_slow() -> None:
    class Broken:
        def get_inputs(self) -> list[Any]:
            return [SimpleNamespace(name="x", shape=[1, 3], type="tensor(float)")]

        def run(self, *_a: Any) -> Any:
            raise RuntimeError("no")

    assert math.isinf(compute.time_session(Broken()))


def _hand_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, times: dict[str, dict[str, float]]) -> None:
    """Pretend the ONNX hand models are downloaded and open fake sessions with per-model speeds."""
    from away.accel import models as gm

    monkeypatch.setattr(gm, "ready", lambda *_a, **_k: True)

    def make(path: Any, providers: list[str], data_dir: Any = None, threads: int = 0) -> Any:
        sess = FakeSession(providers, times[Path(str(path)).stem][providers[0]])
        sess.get_inputs = lambda: [SimpleNamespace(name="x", shape=[1, 8, 8, 3], type="tensor(float)")]  # type: ignore[attr-defined]
        return sess

    monkeypatch.setattr(compute, "ort_session", make)
    monkeypatch.setattr(compute, "time_session", timer)


def test_hand_tracking_keeps_coreml_for_the_faster_model_only(tmp_path: Path, monkeypatch) -> None:
    _hand_models(tmp_path, monkeypatch, {
        "palm_detector": {"CoreMLExecutionProvider": 7.7, "CPUExecutionProvider": 4.3},
        "hand_landmark": {"CoreMLExecutionProvider": 0.5, "CPUExecutionProvider": 3.4},
    })  # fmt: skip
    monkeypatch.setattr(compute.sys, "platform", "darwin")
    cfg = C.AppConfig()
    got = compute.hand_factory(cfg, tmp_path / "hand.task", tmp_path, providers=MAC)
    assert got is not None
    lm = got[0](2)
    assert lm.lmk.get_providers()[0] == "CoreMLExecutionProvider" and lm.palm.get_providers() == [
        "CPUExecutionProvider"
    ]
    st = compute.status(cfg, tmp_path, providers=MAC)
    assert st["using"] == "graphics card" and "Core ML" in st["provider"]
    assert "hand landmarks" in st["reason"] and "palm detector stays on the processor" in st["reason"]
    assert "Apple Neural Engine" in status_line(st)


def test_hand_tracking_goes_back_to_mediapipe_when_coreml_is_slower_everywhere(tmp_path: Path, monkeypatch) -> None:
    _hand_models(tmp_path, monkeypatch, {
        "palm_detector": {"CoreMLExecutionProvider": 9.0, "CPUExecutionProvider": 4.0},
        "hand_landmark": {"CoreMLExecutionProvider": 5.0, "CPUExecutionProvider": 3.0},
    })  # fmt: skip
    monkeypatch.setattr(compute.sys, "platform", "darwin")
    notes: list[str] = []
    cfg = C.AppConfig()
    assert compute.hand_factory(cfg, tmp_path / "hand.task", tmp_path, on_note=notes.append, providers=MAC) is None
    assert notes and "measured slower than the processor" in notes[0] and "5.0 vs 3.0 ms" in notes[0]
    st = compute.status(cfg, tmp_path, providers=MAC)
    assert st["using"] == "processor" and "slower" in st["reason"]


def test_mac_settings_labels_name_the_apple_gpu_and_keep_windows_text() -> None:
    assert labels("darwin")["gpu"] == MAC_GPU_LABEL == "Apple GPU / Neural Engine"
    assert labels("win32")["gpu"] == "Graphics card" and labels("linux")["gpu"] == "Graphics card"
    assert label_to_value(MAC_GPU_LABEL) == "gpu" and label_to_value("Graphics card") == "gpu"


# ------------------------------------------------------------------------------------------ hardware text
SP_DISPLAYS_M5 = """Graphics/Displays:

    Apple M5:

      Chipset Model: Apple M5
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 10
      Vendor: Apple (0x106b)
      Metal Support: Metal 4
"""


def _mac_detect(sp: str | None = SP_DISPLAYS_M5) -> H.Hardware:
    table = {
        "machdep.cpu.brand_string": "Apple M5\n",
        "hw.physicalcpu": "10\n",
        "hw.memsize": str(24 * 1024**3) + "\n",
        "SPDisplaysDataType": sp,
    }

    def run(cmd: Any, timeout: float) -> str | None:
        line = " ".join(cmd)
        return next((out for key, out in table.items() if key in line), None)

    return H.detect(os_name="darwin", arch="arm64", run=run, read=lambda _p: None, disk=lambda: 100_000)


def test_apple_silicon_reports_chip_gpu_cores_and_unified_memory() -> None:
    hw = _mac_detect()
    (g,) = hw.gpus
    assert g.vendor == "apple" and g.name == "Apple M5 GPU (10-core)" and g.gpu_cores == 10
    assert g.unified_memory_mb == 24 * 1024 == hw.unified_memory_mb and g.vram_mb is None
    assert "not used" not in g.name
    p = tuning.choose_profile(hw)
    line = next(ln for ln in performance.describe(hw, p) if ln.startswith("GPU:"))
    assert "24 GB unified memory (shared with the CPU)" in line and "VRAM" not in line
    assert "Core ML" in line and ("speech too (MLX)" in line or "speech runs on the CPU" in line)


def test_apple_silicon_without_system_profiler_still_names_the_chip() -> None:
    (g,) = _mac_detect(sp=None).gpus
    assert g.name == "Apple M5 GPU" and g.gpu_cores is None and g.unified_memory_mb == 24 * 1024


def test_parse_sp_displays_handles_intel_macs_without_core_counts() -> None:
    assert H.parse_sp_displays(SP_DISPLAYS_M5) == ("Apple M5", 10)
    assert H.parse_sp_displays("Chipset Model: Intel Iris Plus Graphics 655\n") == (
        "Intel Iris Plus Graphics 655",
        None,
    )
    assert H.parse_sp_displays("") == ("", None)


def test_unified_memory_is_only_reported_on_apple_silicon() -> None:
    assert H.Hardware(os="win32", arch="amd64", ram_mb=16384).unified_memory_mb is None
    assert H.Hardware(os="darwin", arch="x86_64", ram_mb=16384).unified_memory_mb is None


# ------------------------------------------------------------------------------- models: memory and labels
def test_runs_on_reads_apple_gpu_on_a_mac_and_gpu_elsewhere() -> None:
    assert R.runs_on_title("gpu", "darwin") == "Apple GPU / Neural Engine"
    assert R.runs_on_title("either", "darwin") == "CPU or Apple GPU / Neural Engine"
    assert R.runs_on_title("cpu", "darwin") == "CPU"
    assert (R.runs_on_title("gpu", "win32"), R.runs_on_title("either", "linux")) == ("GPU", "CPU or GPU")
    # a CUDA speech model must not claim the Apple GPU: only tracking goes through Core ML
    assert R.runs_on_title("gpu", "darwin", slot="stt") == "NVIDIA GPU (CPU on a Mac)"
    assert R.runs_on_title("either", "darwin", slot="voice") == "CPU"
    assert R.runs_on_title("gpu", "darwin", slot="gaze") == "Apple GPU / Neural Engine"
    assert R.runs_on_title("gpu", "win32", slot="stt") == "GPU"


@pytest.mark.parametrize(
    ("ram_gb", "need_mb", "fits"),
    [
        (24, 3900, True),  # Best on a 24 GB Mac: 18 GB can be spared
        (8, 3900, True),  # 8 GB keeps 4 GB for macOS: exactly 4 GB spare
        (8, 4200, False),
        (16, 12_500, False),  # 16 GB keeps 4 GB (a quarter)
        (16, 12_000, True),
    ],
)
def test_unified_memory_fit_keeps_headroom_for_macos(ram_gb: int, need_mb: int, fits: bool) -> None:
    hw = H.Hardware(os="darwin", arch="arm64", ram_mb=ram_gb * 1024)
    assert R.memory_budget(hw, "darwin")["kind"] == "unified"
    assert R.memory_fits(need_mb, hw, "darwin") is fits


def test_vram_fit_elsewhere_uses_the_nvidia_card_not_system_memory() -> None:
    g = H.GPU("nvidia", "NVIDIA GeForce RTX 2060", 6144)
    hw = H.Hardware(os="win32", arch="amd64", ram_mb=65536, gpus=[g])
    assert R.memory_budget(hw, "win32") == {"kind": "vram", "total_mb": 6144, "budget_mb": 5120}
    assert R.memory_fits(3900, hw, "win32") and not R.memory_fits(5200, hw, "win32")
    assert R.memory_fits(100, H.Hardware(os="win32", ram_mb=65536), "win32") is None  # no card: unknown


def _panel(tmp_path: Path, plat: str, ram_gb: int = 24) -> Any:
    from away.app.models_next_panel import NextModelsPanel

    hw = H.Hardware(os=plat, arch="arm64" if plat == "darwin" else "amd64", ram_mb=ram_gb * 1024)
    return NextModelsPanel(C.AppConfig(), data_dir=tmp_path, ready=lambda m: False, platform=plat, hw=hw)


def test_mac_presets_show_shared_memory_and_whether_it_fits(tmp_path: Path) -> None:
    rows = {r["mode"]: r for r in _panel(tmp_path, "darwin").presets()}
    assert rows["best"]["vram"] == "about 3.9 GB of shared (unified) memory; fits this Mac's 24 GB"
    assert "graphics" not in rows["light"]["vram"]
    small = {r["mode"]: r for r in _panel(tmp_path, "darwin", ram_gb=4).presets()}
    assert "too much for this Mac's 4 GB" in small["best"]["vram"]


def test_windows_presets_text_is_unchanged(tmp_path: Path) -> None:
    rows = {r["mode"]: r for r in _panel(tmp_path, "win32").presets()}
    assert rows["best"]["vram"] == "about 3.9 GB of graphics memory"


def test_registry_preset_fit_uses_unified_memory_on_a_mac(tmp_path: Path) -> None:
    reg = R.Registry(tmp_path)
    mac8 = H.Hardware(os="darwin", arch="arm64", ram_mb=8 * 1024)
    assert reg.preset_fits("light", mac8, "darwin") is True
    assert reg.preset_fits("best", H.Hardware(os="darwin", arch="arm64", ram_mb=2048), "darwin") is False


# ------------------------------------------------------------------------------------------- the GPU pack
def _rtx2060() -> H.Hardware:
    g = H.GPU("nvidia", "NVIDIA GeForce RTX 2060", 6144, 5480, (7, 5), "560.94", "12.6")
    return H.Hardware(os="win32", arch="amd64", cores_physical=6, cores_logical=12, ram_mb=16384, gpus=[g])


def test_gpu_pack_never_applies_on_macos() -> None:
    assert not gpupack.applies("darwin")
    assert gpupack.applies("win32") and gpupack.applies("linux")


def test_gpu_pack_offer_is_hidden_on_macos_but_kept_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import hardware

    monkeypatch.setattr(hardware, "cached", _rtx2060)
    cfg = C.AppConfig()
    prof = tuning.choose_profile(_rtx2060(), cuda_runtime=False, pack_installed=False)
    assert performance.gpu_pack_offer(cfg, prof, frozen=True, platform="win32")
    assert performance.gpu_pack_offer(cfg, prof, frozen=True, platform="darwin") is None


def test_tray_menu_item_and_notification_are_hidden_on_a_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    from away.app import frozen, hardware, tray

    monkeypatch.setattr(frozen, "is_frozen", lambda: True)  # the installed app, where the offer is live
    prof = tuning.choose_profile(_rtx2060(), cuda_runtime=False, pack_installed=False)  # even a capable profile
    ctl = SimpleNamespace(app=SimpleNamespace(cfg=C.AppConfig(), profile=prof))
    monkeypatch.setattr(hardware, "cached", _rtx2060)
    assert tray.TrayController.gpu_pack_offer(ctl)  # type: ignore[arg-type]  # control: Windows offers it
    mac = H.Hardware(os="darwin", arch="arm64", cpu_name="Apple M5", ram_mb=24 * 1024)
    monkeypatch.setattr(hardware, "cached", lambda: mac)
    assert tray.TrayController.gpu_pack_offer(ctl) is None  # type: ignore[arg-type]
