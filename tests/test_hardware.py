"""Hardware detection (recorded tool outputs, no real GPUs) and the tuning profile chooser."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Sequence

import pytest

from away import config as C
from away.app import hardware as H
from away.app import performance
from away.app import tuning as T

SMI_RTX2060 = "NVIDIA GeForce RTX 2060, 6144, 5480, 560.94, 7.5\n"
SMI_HEADER_126 = "| NVIDIA-SMI 560.94     Driver Version: 560.94     CUDA Version: 12.6     |\n"
SMI_GTX1050 = "NVIDIA GeForce GTX 1050, 2048, 1790, 537.13, 6.1\n"
SMI_OLD_4COL = "NVIDIA GeForce RTX 2060, 6144, 5480, 472.12\n"
SMI_HEADER_114 = "| NVIDIA-SMI 472.12     Driver Version: 472.12     CUDA Version: 11.4     |\n"
CPUINFO_4C = "".join(
    f"processor\t: {i}\nmodel name\t: Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz\nphysical id\t: 0\ncore id\t\t: {i}\n\n"
    for i in range(4)
)
MEMINFO_8G = "MemTotal:        8048576 kB\nMemFree:  100 kB\n"


def fake_runner(table: dict[str, str | None], log: list[list[str]] | None = None):
    """Answer a probe command by the first table key contained in the joined command line."""

    def run(cmd: Sequence[str], timeout: float) -> str | None:
        line = " ".join(cmd)
        if log is not None:
            log.append(list(cmd))
        for key, out in table.items():
            if key in line:
                return out
        return None

    return run


def no_nvml(_os: str) -> list[H.GPU]:
    return []


def detect(os_name: str = "linux", arch: str = "x86_64", table: dict[str, str | None] | None = None, **kw):
    files = kw.pop("files", {"/proc/cpuinfo": CPUINFO_4C, "/proc/meminfo": MEMINFO_8G})
    return H.detect(
        os_name=os_name,
        arch=arch,
        run=fake_runner(table or {}),
        read=lambda p: files.get(p),
        nvml_probe=kw.pop("nvml_probe", no_nvml),
        disk=kw.pop("disk", lambda: 200_000),
    )


# ---------------------------------------------------------------- parsing
def test_parse_nvidia_smi_rtx2060_with_cuda_version():
    (g,) = H.parse_nvidia_smi(SMI_RTX2060, SMI_HEADER_126)
    assert (g.vendor, g.name, g.vram_mb, g.vram_free_mb) == ("nvidia", "NVIDIA GeForce RTX 2060", 6144, 5480)
    assert g.compute_capability == (7, 5) and g.driver_version == "560.94" and g.cuda_driver_version == "12.6"
    assert g.source == "nvidia-smi" and g.cc_str == "7.5"


def test_parse_nvidia_smi_old_driver_without_compute_cap_and_junk_lines():
    gpus = H.parse_nvidia_smi("garbage\n\n" + SMI_OLD_4COL, None)
    assert len(gpus) == 1 and gpus[0].compute_capability is None and gpus[0].cuda_driver_version is None


def test_nvidia_smi_retries_without_compute_cap_on_old_drivers():
    log: list[list[str]] = []
    table = {"compute_cap": None, "--query-gpu": SMI_OLD_4COL, "nvidia-smi": SMI_HEADER_114}
    gpus = H.nvidia_smi(fake_runner(table, log))
    assert gpus[0].driver_version == "472.12" and gpus[0].cuda_driver_version == "11.4"
    assert any("compute_cap" in " ".join(c) for c in log) and len(log) == 3


def test_parse_cim_and_windows_driver_version():
    text = "NVIDIA GeForce RTX 2060|4293918720|31.0.15.6094\nIntel(R) UHD Graphics 630|1073741824|27.20.100.8681\n"
    nv, intel = H.parse_cim_video(text)
    assert nv.vendor == "nvidia" and nv.driver_version == "560.94" and nv.compute_capability == (7, 5)
    assert nv.vram_mb == 4095  # AdapterRAM is a uint32: capped near 4 GB, only a hint
    assert intel.vendor == "intel" and intel.compute_capability is None
    assert H.nvidia_from_windows_driver("31.0.15.3623") == "536.23"
    assert H.nvidia_from_windows_driver("1.2") is None


def test_parse_wmic_csv():
    text = "Node,AdapterRAM,DriverVersion,Name\nPC,2147483648,31.0.15.3713,NVIDIA GeForce GTX 1050\n"
    (g,) = H.parse_wmic(text)
    assert g.name == "NVIDIA GeForce GTX 1050" and g.vram_mb == 2048 and g.driver_version == "537.13"
    assert g.compute_capability == (6, 1) and g.source == "wmic"
    assert H.parse_wmic("") == []


def test_parse_lspci():
    text = (
        "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 630\n"
        "01:00.0 VGA compatible controller: NVIDIA Corporation TU106 [GeForce RTX 2060] (rev a1)\n"
        "00:1f.3 Audio device: Intel Corporation Cannon Lake PCH cAVS\n"
    )
    intel, nv = H.parse_lspci(text)
    assert intel.vendor == "intel" and nv.vendor == "nvidia" and nv.compute_capability == (7, 5)


def test_parse_cpuinfo_and_meminfo():
    name, phys, logical = H.parse_cpuinfo(CPUINFO_4C)
    assert name.startswith("Intel") and phys == 4 and logical == 4
    ht = "".join(f"processor\t: {i}\nphysical id\t: 0\ncore id\t\t: {i % 2}\n\n" for i in range(4))
    assert H.parse_cpuinfo(ht)[1:] == (2, 4)  # hyper-threading: 2 cores, 4 threads
    assert H.parse_cpuinfo("processor : 0\n\nprocessor : 1\n")[1:] == (None, 2)  # ARM: no core ids
    assert H.parse_meminfo(MEMINFO_8G) == 7859 and H.parse_meminfo("nothing") is None


@pytest.mark.parametrize(
    "name,cc",
    [
        ("NVIDIA GeForce RTX 2060", (7, 5)),
        ("GeForce GTX 1660 Ti", (7, 5)),
        ("NVIDIA GeForce RTX 3060 Laptop GPU", (8, 6)),
        ("NVIDIA GeForce RTX 4090", (8, 9)),
        ("NVIDIA GeForce GTX 1050", (6, 1)),
        ("GeForce GTX 970", (5, 2)),
        ("Some Unknown Board", None),
    ],
)
def test_guess_cc(name, cc):
    assert H.guess_cc(name) == cc


def test_parse_version():
    assert H.parse_version("536.23") == (536, 23)
    assert H.parse_version("525.60.13") == (525, 60, 13)
    assert H.parse_version(None) == () and H.parse_version("n/a") == ()


# ---------------------------------------------------------------- detect()
def test_detect_linux_rtx2060_via_nvidia_smi():
    hw = detect(table={"--query-gpu": SMI_RTX2060, "nvidia-smi": SMI_HEADER_126, "lspci": None})
    assert hw.cores_physical == 4 and hw.ram_mb is not None and hw.disk_free_mb == 200_000
    assert hw.nvidia is not None and hw.nvidia.vram_mb == 6144 and hw.nvidia.source == "nvidia-smi"


def test_detect_prefers_nvml_and_dedupes_os_list():
    nv = H.GPU("nvidia", "NVIDIA GeForce RTX 2060", 6144, 5000, (7, 5), "560.94", "12.6", "nvml")
    lspci = "01:00.0 VGA compatible controller: NVIDIA Corporation TU106 [GeForce RTX 2060]\n"
    log: list[list[str]] = []
    hw = H.detect(
        os_name="linux",
        arch="x86_64",
        run=fake_runner({"lspci": lspci}, log),
        read=lambda p: None,
        nvml_probe=lambda o: [nv],
        disk=lambda: None,
    )
    assert hw.gpus == [nv]  # the lspci duplicate of the NVIDIA card is dropped
    assert not any("nvidia-smi" in c[0] for c in log)  # NVML answered: no subprocess


def test_detect_nvidia_smi_missing_or_timed_out_never_raises():
    hw = detect(table={})  # every command "missing"
    assert hw.gpus == [] and any("no NVIDIA driver" in n for n in hw.notes)
    p = T.choose_profile(hw)
    assert p.device == "cpu"


def test_detect_survives_probes_that_raise():
    def boom(*a, **k):
        raise RuntimeError("probe exploded")

    hw = H.detect(os_name="linux", arch="x86_64", run=boom, read=boom, nvml_probe=boom, disk=boom)
    assert isinstance(hw, H.Hardware) and len(hw.notes) >= 3


def test_detect_windows_cim_fallback():
    cim = "NVIDIA GeForce GTX 1050|2147483648|31.0.15.3713\n"
    cpu = "Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz|4\n"
    hw = detect(os_name="win32", table={"Win32_VideoController": cim, "Win32_Processor": cpu})
    assert hw.cpu_name.startswith("Intel") and hw.cores_physical == 4
    assert hw.nvidia is not None and hw.nvidia.source == "cim" and hw.nvidia.driver_version == "537.13"


def test_detect_apple_m1():
    table = {
        "machdep.cpu.brand_string": "Apple M1\n",
        "hw.physicalcpu": "8\n",
        "hw.memsize": str(16 * 1024**3) + "\n",
    }
    hw = detect(os_name="darwin", arch="arm64", table=table)
    assert hw.apple_silicon and hw.cpu_name == "Apple M1" and hw.cores_physical == 8 and hw.ram_mb == 16384
    assert hw.nvidia is None and hw.gpus[0].vendor == "apple"


def test_run_cmd_timeout_and_missing_tool():
    assert H.run_cmd(["definitely-not-a-real-tool-xyz"]) is None
    assert H.run_cmd([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.3) is None
    assert H.run_cmd([sys.executable, "-c", "print('hi')"]).strip() == "hi"
    assert H.run_cmd([sys.executable, "-c", "raise SystemExit(3)"]) is None


def test_nvml_missing_library_and_failing_init():
    assert H.nvml("linux", loader=lambda o: None) == []

    class BadInit:
        def nvmlInit_v2(self):
            return 9  # NVML_ERROR_DRIVER_NOT_LOADED

        def nvmlShutdown(self):
            return 0

    assert H.nvml("linux", loader=lambda o: BadInit()) == []


def test_nvml_reads_device_info_through_ctypes_calls():
    def put(ref, value):
        ref._obj.value = value

    class FakeNVML:
        def nvmlInit_v2(self):
            return 0

        def nvmlShutdown(self):
            return 0

        def nvmlSystemGetDriverVersion(self, buf, n):
            buf.value = b"560.94"
            return 0

        def nvmlSystemGetCudaDriverVersion_v2(self, ref):
            put(ref, 12060)
            return 0

        def nvmlDeviceGetCount_v2(self, ref):
            put(ref, 1)
            return 0

        def nvmlDeviceGetHandleByIndex_v2(self, i, ref):
            return 0

        def nvmlDeviceGetName(self, h, buf, n):
            buf.value = b"NVIDIA GeForce RTX 2060"
            return 0

        def nvmlDeviceGetMemoryInfo(self, h, ref):
            mem = ref._obj
            mem.total, mem.free = 6144 * 1024**2, 5000 * 1024**2
            return 0

        def nvmlDeviceGetCudaComputeCapability(self, h, a, b):
            put(a, 7)
            put(b, 5)
            return 0

    (g,) = H.nvml("linux", loader=lambda o: FakeNVML())
    assert (g.name, g.vram_mb, g.vram_free_mb, g.compute_capability) == ("NVIDIA GeForce RTX 2060", 6144, 5000, (7, 5))
    assert (g.driver_version, g.cuda_driver_version, g.source) == ("560.94", "12.6", "nvml")
    assert isinstance(ctypes.c_int(0), ctypes.c_int)


def test_driver_check():
    g = H.GPU("nvidia", "x", driver_version="560.94", cuda_driver_version="12.6")
    assert H.driver_ok_for_cuda12(g, "win32")[0]
    old = H.GPU("nvidia", "x", driver_version="472.12", cuda_driver_version="11.4")
    ok, why = H.driver_ok_for_cuda12(old, "win32")
    assert not ok and "CUDA 11.4" in why
    assert not H.driver_ok_for_cuda12(H.GPU("nvidia", "x", driver_version="520.10"), "win32")[0]
    assert H.driver_ok_for_cuda12(H.GPU("nvidia", "x", driver_version="527.41"), "win32")[0]
    assert not H.driver_ok_for_cuda12(H.GPU("nvidia", "x", driver_version="525.60.12"), "linux")[0]
    assert H.driver_ok_for_cuda12(H.GPU("nvidia", "x", driver_version="525.60.13"), "linux")[0]
    assert H.driver_ok_for_cuda12(H.GPU("nvidia", "x"), "linux")[0]  # unknown: try, fall back at runtime


def test_hardware_to_dict_and_cores_fallback():
    hw = H.Hardware(cores_logical=6, gpus=[H.GPU("nvidia", "g", compute_capability=(7, 5))])
    assert hw.cores == 6 and H.Hardware().cores == 2
    assert hw.to_dict()["gpus"][0]["compute_capability"] == "7.5"


# ---------------------------------------------------------------- profiles for the reference machines
def rtx2060(**kw) -> H.Hardware:
    g = H.GPU(
        "nvidia", "NVIDIA GeForce RTX 2060", 6144, 5480, (7, 5), kw.pop("driver", "560.94"), kw.pop("cuda", "12.6")
    )
    return H.Hardware(os="win32", arch="amd64", cores_physical=6, cores_logical=12, ram_mb=16384, gpus=[g], **kw)


def test_profile_rtx2060_with_cuda_runtime():
    p = T.choose_profile(rtx2060(), cuda_runtime=True)
    assert (p.device, p.compute_type, p.asr_model, p.beam_size, p.cpu_threads) == (
        "cuda",
        "float16",
        "large-v3-turbo",
        5,
        4,
    )
    assert p.tier == "gpu" and p.gpu_capable and p.gpu_pack == "not-needed"
    assert (p.camera_width, p.camera_height, p.camera_fps, p.hands_every, p.face_every) == (640, 480, 30, 1, 2)
    assert any("RTX 2060" in r for r in p.reasons)


def test_profile_rtx2060_without_cuda_libraries_recommends_gpu_pack():
    p = T.choose_profile(rtx2060(), cuda_runtime=False)
    assert p.device == "cpu" and p.compute_type == "int8" and p.gpu_pack == "recommended" and p.gpu_capable
    assert p.asr_model == "small"  # 6 cores / 16 GB CPU tier
    assert any("away gpu install" in r for r in p.reasons)
    assert T.choose_profile(rtx2060(), cuda_runtime=True, pack_installed=True).gpu_pack == "installed"


def test_profile_rtx2060_old_driver_falls_back_to_cpu_with_reason():
    p = T.choose_profile(rtx2060(driver="472.12", cuda="11.4"), cuda_runtime=True)
    assert p.device == "cpu" and not p.gpu_capable and p.gpu_pack == "unavailable"
    assert any("CUDA 11.4" in r for r in p.reasons)


def test_profile_gtx1050_2gb():
    g = H.GPU("nvidia", "NVIDIA GeForce GTX 1050", 2048, 1790, (6, 1), "537.13", "12.2")
    hw = H.Hardware(os="win32", arch="amd64", cores_physical=4, cores_logical=8, ram_mb=8192, gpus=[g])
    p = T.choose_profile(hw, cuda_runtime=True)
    assert (p.device, p.compute_type, p.asr_model) == ("cuda", "int8", "small")  # turbo does not fit in 2 GB
    assert any("needs" in r for r in p.reasons)


def test_profile_turing_with_3gb_uses_int8_float16_and_still_fits_turbo():
    g = H.GPU("nvidia", "NVIDIA GeForce GTX 1660", 3072, 2800, (7, 5), "560.94", "12.6")
    hw = H.Hardware(os="linux", arch="x86_64", cores_physical=6, ram_mb=16384, gpus=[g])
    p = T.choose_profile(hw, cuda_runtime=True)
    assert (p.compute_type, p.asr_model) == ("int8_float16", "large-v3-turbo")


def test_profile_no_gpu_4_cores_8gb():
    hw = H.Hardware(os="linux", arch="x86_64", cores_physical=4, cores_logical=4, ram_mb=8000)
    p = T.choose_profile(hw)
    assert (p.device, p.compute_type, p.asr_model, p.beam_size, p.cpu_threads) == ("cpu", "int8", "small", 3, 3)
    assert (p.camera_width, p.camera_fps, p.face_every) == (640, 30, 3)  # Whisper shares 4 cores with MediaPipe
    assert p.gpu_pack == "not-needed" and not p.gpu_capable


def test_profile_apple_m1():
    hw = H.Hardware(os="darwin", arch="arm64", cpu_name="Apple M1", cores_physical=8, ram_mb=16384)
    p = T.choose_profile(hw, cuda_runtime=True, ov=T.Overrides(mode="gpu"))
    assert p.device == "cuda"  # explicit gpu mode is honoured (and falls back at runtime) ...
    assert any("Metal" in r for r in p.reasons)  # ... with the reason it cannot work
    p = T.choose_profile(hw)
    assert (p.device, p.compute_type, p.asr_model) == ("cpu", "int8", "small")
    assert any("Apple Silicon" in r for r in p.reasons)


@pytest.mark.parametrize(
    "cores,ram,model,beam,tier",
    [
        (16, 32768, "large-v3-turbo", 2, "cpu-high"),
        (8, 16384, "small", 3, "cpu-mid"),
        (2, 4096, "base", 1, "cpu-low"),
        (1, 2048, "tiny", 1, "cpu-min"),
        (8, 4096, "base", 1, "cpu-low"),  # many cores, little RAM
    ],
)
def test_cpu_tiers(cores, ram, model, beam, tier):
    p = T.choose_profile(H.Hardware(os="linux", arch="x86_64", cores_physical=cores, ram_mb=ram))
    assert (p.asr_model, p.beam_size, p.tier) == (model, beam, tier)
    assert 1 <= p.cpu_threads <= 8


def test_low_disk_downsizes_model():
    hw = rtx2060(disk_free_mb=1200)
    p = T.choose_profile(hw, cuda_runtime=True)
    assert p.asr_model == "small" and any("free disk" in r for r in p.reasons)


def test_tiny_vram_and_arm_linux_and_lspci_only_gpus_are_not_used():
    tiny = H.Hardware(
        os="linux",
        arch="x86_64",
        cores_physical=4,
        ram_mb=8192,
        gpus=[H.GPU("nvidia", "GT 710", 1024, None, (3, 5), "470.1")],
    )
    assert T.choose_profile(tiny, cuda_runtime=True).device == "cpu"
    jetson = H.Hardware(
        os="linux",
        arch="aarch64",
        cores_physical=6,
        ram_mb=8192,
        gpus=[H.GPU("nvidia", "Orin", 8192, None, (8, 7), "540.0")],
    )
    p = T.choose_profile(jetson, cuda_runtime=True)
    assert p.device == "cpu" and any("aarch64" in r for r in p.reasons)
    nodrv = H.Hardware(
        os="linux",
        arch="x86_64",
        cores_physical=8,
        ram_mb=16384,
        gpus=[H.GPU("nvidia", "RTX 2060", source="lspci", compute_capability=(7, 5))],
    )
    p = T.choose_profile(nodrv, cuda_runtime=True)
    assert p.device == "cpu" and any("driver" in r for r in p.reasons)
    assert T.choose_profile(rtx2060(), cuda_runtime=True, ct2_cuda=False).device == "cpu"
    maxwell = H.Hardware(os="linux", arch="x86_64", gpus=[H.GPU("nvidia", "GTX 970", 4096, None, (5, 2), "560.1")])
    assert T.choose_profile(maxwell, cuda_runtime=True).compute_type == "float32"


def test_overrides_win():
    ov = T.Overrides(mode="cpu")
    p = T.choose_profile(rtx2060(), ov, cuda_runtime=True)
    assert p.device == "cpu" and p.compute_type == "int8" and p.gpu_capable
    ov = T.Overrides(
        device="cpu",
        asr_model="large-v3-turbo",
        compute_type="int8",
        cpu_threads=6,
        beam_size=1,
        camera_fps=15,
        face_every=5,
    )
    p = T.choose_profile(rtx2060(), ov, cuda_runtime=True)
    assert (p.device, p.asr_model, p.compute_type, p.cpu_threads, p.beam_size, p.camera_fps, p.face_every) == (
        "cpu",
        "large-v3-turbo",
        "int8",
        6,
        1,
        15,
        5,
    )
    assert any("explicit" in r for r in p.reasons)
    p = T.choose_profile(
        H.Hardware(os="linux", arch="x86_64", cores_physical=4, ram_mb=8000), T.Overrides(device="cuda")
    )
    assert p.device == "cuda"  # explicit device is tried (the loader falls back to the CPU if it fails)
    p = T.choose_profile(rtx2060(), T.Overrides(mode="gpu"), cuda_runtime=False)
    assert p.device == "cuda" and any("away gpu install" in r for r in p.reasons)


def test_overrides_from_config_and_validation():
    cfg = C.AppConfig()
    assert cfg.performance.mode == "auto" and cfg.voice.asr_model == "auto"
    cfg, unknown = C.from_dict({"performance": {"mode": "gpu", "beam_size": 2}, "voice": {"device": "cuda"}})
    ov = T.Overrides.from_config(cfg)
    assert (ov.mode, ov.beam_size, ov.device, ov.asr_model) == ("gpu", 2, "cuda", "auto") and not unknown
    for bad in (
        {"performance": {"mode": "turbo"}},
        {"performance": {"beam_size": 99}},
        {"performance": {"cpu_threads": True}},
        {"voice": {"device": "metal"}},
        {"voice": {"asr_model": ""}},
    ):
        with pytest.raises(ValueError):
            C.from_dict(bad)
    # round trip keeps the section
    assert C.from_dict(C.to_dict(cfg))[0].performance.beam_size == 2


def test_compute_type_rules():
    assert T.gpu_compute_type((7, 5), 6144)[0] == "float16"
    assert T.gpu_compute_type((8, 6), 2048)[0] == "int8_float16"
    assert T.gpu_compute_type((6, 1), 4096)[0] == "int8"
    assert T.gpu_compute_type((5, 2), 4096)[0] == "float32"
    assert T.gpu_compute_type(None, None)[0] == "float16"
    assert T.model_fits_vram("large-v3-turbo", "float16", None)
    assert not T.model_fits_vram("large-v3-turbo", "float16", 2048)


# ---------------------------------------------------------------- glue
def test_performance_glue_uses_cached_hardware_and_explicit_model(fixed_hardware):
    cfg = C.AppConfig()
    p = performance.current_profile(cfg)
    assert p.asr_model == "small" and p.device == "cpu"
    assert performance.effective_asr_model(cfg, p) == "small"
    cfg.voice.asr_model = "tiny"
    assert performance.effective_asr_model(cfg) == "tiny"
    lines = performance.describe(fixed_hardware, p)
    assert any(line.startswith("Profile (cpu-mid)") for line in lines) and "GPU:   none detected" in lines


def test_gpu_pack_offer_only_for_frozen_capable_machines(monkeypatch):
    from away.app import hardware

    monkeypatch.setattr(hardware, "cached", lambda: rtx2060())
    cfg = C.AppConfig()
    assert performance.gpu_pack_offer(cfg, frozen=True) and "RTX 2060" in performance.gpu_pack_offer(cfg, frozen=True)
    assert performance.gpu_pack_offer(cfg, frozen=False) is None
    cfg.performance.mode = "cpu"
    assert performance.gpu_pack_offer(cfg, frozen=True) is None
