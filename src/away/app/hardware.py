"""Hardware auto-detection (stdlib only; every probe is optional, bounded by a timeout, and never raises).

What it finds:

* CPU: name, physical and logical cores (``/proc/cpuinfo``, ``sysctl``, WMI/``Get-CimInstance``),
* RAM total (``/proc/meminfo``, ``GlobalMemoryStatusEx``, ``sysctl hw.memsize``),
* free disk space where the models are stored,
* OS / architecture,
* GPUs: NVIDIA through NVML (``nvml.dll`` / ``libnvidia-ml.so.1`` via ``ctypes``, no subprocess), then
  ``nvidia-smi``; any vendor through ``Get-CimInstance Win32_VideoController`` / ``wmic`` on Windows or
  ``lspci`` on Linux (name only); Apple Silicon from ``platform`` + ``sysctl`` (v0.1.35: its unified memory from
  ``hw.memsize`` and the GPU core count from ``system_profiler SPDisplaysDataType``).

:func:`detect` takes injectable probes so the tests can feed recorded outputs (RTX 2060, GTX 1050,
Apple M1, an old driver, ``nvidia-smi`` missing or hanging...) without the real tools.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TIMEOUT_S = 4.0

# CUDA 12.x minor-version compatibility needs at least this driver (CUDA 12.0 release notes, table 3).
MIN_DRIVER_LINUX = (525, 60, 13)
MIN_DRIVER_WINDOWS = (527, 41)

Runner = Callable[[Sequence[str], float], "str | None"]


@dataclass
class GPU:
    vendor: str  # nvidia | amd | intel | apple | other
    name: str
    vram_mb: int | None = None  # total memory (MiB); unified memory on Apple Silicon is in unified_memory_mb
    vram_free_mb: int | None = None
    compute_capability: tuple[int, int] | None = None  # NVIDIA only, e.g. (7, 5)
    driver_version: str | None = None
    cuda_driver_version: str | None = None  # highest CUDA the driver supports, e.g. "12.4"
    source: str = ""  # nvml | nvidia-smi | cim | wmic | lspci | platform
    # v0.1.35: Apple Silicon has no VRAM: the GPU and the Neural Engine share the machine's memory (hw.memsize)
    unified_memory_mb: int | None = None
    gpu_cores: int | None = None  # Apple Silicon: "Total Number of Cores" from system_profiler

    @property
    def cc_str(self) -> str:
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}" if self.compute_capability else "?"


@dataclass
class Hardware:
    os: str = ""  # win32 | darwin | linux
    os_release: str = ""
    arch: str = ""  # x86_64 | arm64 | aarch64 ...
    cpu_name: str = ""
    cores_physical: int | None = None
    cores_logical: int | None = None
    ram_mb: int | None = None
    disk_free_mb: int | None = None
    gpus: list[GPU] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # what failed / timed out (for `away hardware`)

    @property
    def nvidia(self) -> GPU | None:
        """The NVIDIA GPU with the most memory (the one CTranslate2 uses as device 0 is usually that one)."""
        nv = [g for g in self.gpus if g.vendor == "nvidia"]
        return max(nv, key=lambda g: g.vram_mb or 0) if nv else None

    @property
    def apple_silicon(self) -> bool:
        return self.os == "darwin" and self.arch in ("arm64", "aarch64")

    @property
    def unified_memory_mb(self) -> int | None:
        """v0.1.35: the memory the Apple Silicon GPU / Neural Engine share with the CPU (None elsewhere)."""
        return self.ram_mb if self.apple_silicon else None

    @property
    def cores(self) -> int:
        """Physical cores when known, else logical, else 2 (a safe guess)."""
        return self.cores_physical or self.cores_logical or 2

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gpus"] = [{**asdict(g), "compute_capability": g.cc_str if g.compute_capability else None} for g in self.gpus]
        return d


# ---------------------------------------------------------------- helpers
def run_cmd(cmd: Sequence[str], timeout: float = TIMEOUT_S) -> str | None:
    """Run a probe command; ``None`` when it is missing, fails, or exceeds ``timeout``."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    kw: dict[str, Any] = {}
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash from a GUI app
    try:
        p = subprocess.run(
            [exe, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
            **kw,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return p.stdout if p.returncode == 0 else None


def parse_version(text: str | None) -> tuple[int, ...]:
    """``"536.23"`` -> ``(536, 23)``; junk -> ``()``."""
    if not text:
        return ()
    m = re.match(r"\s*(\d+(?:\.\d+)*)", str(text))
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def _int(text: Any) -> int | None:
    try:
        return int(float(str(text).strip()))
    except (TypeError, ValueError):
        return None


def _cc(text: Any) -> tuple[int, int] | None:
    m = re.match(r"\s*(\d+)\.(\d+)", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "tesla" in n or "rtx" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n or "ati " in n:
        return "amd"
    if "intel" in n:
        return "intel"
    if "apple" in n:
        return "apple"
    return "other"


# ---------------------------------------------------------------- NVIDIA
def parse_nvidia_smi(csv_text: str, driver_header: str | None = None) -> list[GPU]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap
    --format=csv,noheader,nounits`` (``compute_cap`` may be missing on old drivers: 4 columns)."""
    cuda = None
    if driver_header:
        m = re.search(r"CUDA Version:\s*([\d.]+)", driver_header)
        cuda = m.group(1) if m else None
    gpus = []
    for line in csv_text.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 4 or not cols[0]:
            continue
        gpus.append(
            GPU(
                "nvidia",
                cols[0],
                vram_mb=_int(cols[1]),
                vram_free_mb=_int(cols[2]),
                driver_version=cols[3] or None,
                compute_capability=_cc(cols[4]) if len(cols) > 4 else None,
                cuda_driver_version=cuda,
                source="nvidia-smi",
            )
        )
    return gpus


NVSMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
    "--format=csv,noheader,nounits",
]
NVSMI_QUERY_OLD = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.free,driver_version",
    "--format=csv,noheader,nounits",
]


def nvidia_smi(run: Runner = run_cmd) -> list[GPU]:
    out = run(NVSMI_QUERY, TIMEOUT_S)
    if out is None:  # drivers older than ~510 reject the compute_cap field
        out = run(NVSMI_QUERY_OLD, TIMEOUT_S)
    if not out:
        return []
    return parse_nvidia_smi(out, run(["nvidia-smi"], TIMEOUT_S))


# GPUs whose name is enough to know the compute capability when the driver cannot report it
_CC_BY_NAME = (
    (r"RTX 50\d\d", (12, 0)),
    (r"RTX 40\d\d|RTX \d+ Ada|L4\b|L40", (8, 9)),
    (r"RTX 30\d\d|A\d{2,4}\b|RTX A\d", (8, 6)),
    (r"A100|A30\b", (8, 0)),
    (r"RTX 20\d\d|GTX 16\d\d|TITAN RTX|Quadro RTX|T4\b|T\d{3,4}\b", (7, 5)),
    (r"V100|TITAN V", (7, 0)),
    (r"GTX 10\d\d|TITAN X[p ]|P\d{2,4}\b|MX[1-4]\d\d", (6, 1)),
    (r"GTX 9\d\d|GTX 750", (5, 2)),
)


def guess_cc(name: str) -> tuple[int, int] | None:
    for pat, cc in _CC_BY_NAME:
        if re.search(pat, name, re.IGNORECASE):
            return cc
    return None


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _load_nvml(os_name: str) -> Any:
    candidates: list[str] = []
    if os_name == "win32":
        sysroot = os.environ.get("SYSTEMROOT", r"C:\Windows")  # Windows env vars are case-insensitive
        pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        candidates = [
            "nvml.dll",
            str(Path(sysroot) / "System32" / "nvml.dll"),
            str(Path(pf) / "NVIDIA Corporation" / "NVSMI" / "nvml.dll"),
        ]
    elif os_name.startswith("linux"):
        candidates = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    for c in candidates:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    return None


def nvml(os_name: str | None = None, loader: Callable[[str], Any] | None = None) -> list[GPU]:
    """Query NVML directly (fast, no subprocess). ``[]`` when the library is missing or anything fails."""
    os_name = os_name or sys.platform
    lib = (loader or _load_nvml)(os_name)
    if lib is None:
        return []
    gpus: list[GPU] = []
    try:
        init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
        if init() != 0:
            return []
        try:
            buf = ctypes.create_string_buffer(96)
            driver = buf.value.decode() if lib.nvmlSystemGetDriverVersion(buf, 96) == 0 else None
            cuda = None
            cv = ctypes.c_int(0)
            fn = getattr(lib, "nvmlSystemGetCudaDriverVersion_v2", None) or getattr(
                lib, "nvmlSystemGetCudaDriverVersion", None
            )
            if fn is not None and fn(ctypes.byref(cv)) == 0 and cv.value:
                cuda = f"{cv.value // 1000}.{cv.value % 1000 // 10}"
            count = ctypes.c_uint(0)
            get_count = getattr(lib, "nvmlDeviceGetCount_v2", None) or lib.nvmlDeviceGetCount
            if get_count(ctypes.byref(count)) != 0:
                return []
            get_handle = getattr(lib, "nvmlDeviceGetHandleByIndex_v2", None) or lib.nvmlDeviceGetHandleByIndex
            for i in range(count.value):
                h = ctypes.c_void_p()
                if get_handle(ctypes.c_uint(i), ctypes.byref(h)) != 0:
                    continue
                name = buf.value.decode(errors="replace") if lib.nvmlDeviceGetName(h, buf, 96) == 0 else "NVIDIA GPU"
                mem = _NvmlMemory()
                total = free = None
                if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem)) == 0:
                    total, free = int(mem.total // (1024 * 1024)), int(mem.free // (1024 * 1024))
                major, minor = ctypes.c_int(0), ctypes.c_int(0)
                cc = None
                ccfn = getattr(lib, "nvmlDeviceGetCudaComputeCapability", None)
                if ccfn is not None and ccfn(h, ctypes.byref(major), ctypes.byref(minor)) == 0:
                    cc = (major.value, minor.value)
                gpus.append(GPU("nvidia", name, total, free, cc, driver, cuda, "nvml"))
        finally:
            with contextlib.suppress(Exception):
                lib.nvmlShutdown()
    except Exception:  # missing symbol, ABI surprise: NVML is a best-effort fast path
        return []
    return gpus


# ---------------------------------------------------------------- other GPUs
def parse_cim_video(text: str) -> list[GPU]:
    """``Get-CimInstance Win32_VideoController | ForEach-Object { "$($_.Name)|$($_.AdapterRAM)|$($_.DriverVersion)" }``

    ``AdapterRAM`` is a uint32 (caps at 4 GB) so it is only a hint; Windows driver versions look like
    ``31.0.15.3623`` where the NVIDIA version is the last 5 digits (``536.23``)."""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if not parts or not parts[0]:
            continue
        name = parts[0]
        ram = _int(parts[1]) if len(parts) > 1 else None
        drv = parts[2] if len(parts) > 2 and parts[2] else None
        vendor = _vendor(name)
        if vendor == "nvidia" and drv:
            drv = nvidia_from_windows_driver(drv) or drv
        gpus.append(
            GPU(
                vendor,
                name,
                vram_mb=(ram // (1024 * 1024)) if ram and ram > 0 else None,
                driver_version=drv,
                compute_capability=guess_cc(name) if vendor == "nvidia" else None,
                source="cim",
            )
        )
    return gpus


def nvidia_from_windows_driver(ver: str) -> str | None:
    """``31.0.15.3623`` -> ``536.23`` (NVIDIA encodes its version in the last two fields)."""
    digits = "".join(ver.split(".")[-2:])
    if len(digits) < 5 or not digits.isdigit():
        return None
    d = digits[-5:]
    return f"{int(d[:3])}.{d[3:]}"


def parse_wmic(text: str) -> list[GPU]:
    """``wmic path win32_VideoController get Name,AdapterRAM,DriverVersion /format:csv``."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = [h.strip().lower() for h in lines[0].split(",")]
    out = []
    for ln in lines[1:]:
        row = dict(zip(header, [c.strip() for c in ln.split(",")]))
        if row.get("name"):
            out.append(f"{row['name']}|{row.get('adapterram', '')}|{row.get('driverversion', '')}")
    gpus = parse_cim_video("\n".join(out))
    for g in gpus:
        g.source = "wmic"
    return gpus


def parse_lspci(text: str) -> list[GPU]:
    gpus = []
    for line in text.splitlines():
        if not re.search(r"VGA compatible controller|3D controller|Display controller", line):
            continue
        name = line.split(":", 2)[-1].strip()
        vendor = _vendor(name)
        gpus.append(
            GPU(vendor, name, compute_capability=guess_cc(name) if vendor == "nvidia" else None, source="lspci")
        )
    return gpus


CIM_CMD = [
    "powershell",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    'Get-CimInstance Win32_VideoController | ForEach-Object { "$($_.Name)|$($_.AdapterRAM)|$($_.DriverVersion)" }',
]
WMIC_CMD = ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM,DriverVersion", "/format:csv"]


def generic_gpus(os_name: str, run: Runner = run_cmd) -> list[GPU]:
    if os_name == "win32":
        out = run(CIM_CMD, TIMEOUT_S + 4)  # PowerShell start-up is slow on some PCs
        if out:
            return parse_cim_video(out)
        out = run(WMIC_CMD, TIMEOUT_S)
        return parse_wmic(out) if out else []
    if os_name.startswith("linux"):
        out = run(["lspci"], TIMEOUT_S)
        return parse_lspci(out) if out else []
    return []


# ---------------------------------------------------------------- Apple Silicon (v0.1.35)
SP_DISPLAYS_CMD = ["system_profiler", "-detailLevel", "mini", "SPDisplaysDataType"]  # ~0.2 s


def parse_sp_displays(text: str) -> tuple[str, int | None]:
    """``(chip, GPU cores)`` from ``system_profiler SPDisplaysDataType`` ("Chipset Model: Apple M5", "Total Number
    of Cores: 10"); ``("", None)`` when they are missing (Intel Macs list a Radeon / Intel GPU with no core count)."""
    chip = re.search(r"Chipset Model:\s*(.+)", text or "")
    cores = re.search(r"Total Number of Cores:\s*(\d+)", text or "")
    return (chip.group(1).strip() if chip else ""), (int(cores.group(1)) if cores else None)


def apple_gpu(cpu_name: str, ram_mb: int | None, run: Runner = run_cmd) -> GPU:
    """The Apple Silicon GPU: its chip, GPU cores and the unified memory it shares with the CPU. Used by hand / face
    / eye tracking through Core ML (:mod:`away.accel.compute`, kept only where it measures faster than the CPU)
    and by speech recognition through MLX (:mod:`away.voice.asr_mlx`) when mlx-whisper is installed."""
    chip, cores = "", None
    with contextlib.suppress(Exception):
        chip, cores = parse_sp_displays(run(SP_DISPLAYS_CMD, TIMEOUT_S) or "")
    name = f"{chip or cpu_name or 'Apple Silicon'} GPU" + (f" ({cores}-core)" if cores else "")
    return GPU("apple", name, unified_memory_mb=ram_mb, gpu_cores=cores, source="platform")


# ---------------------------------------------------------------- CPU / RAM / disk
def parse_cpuinfo(text: str) -> tuple[str, int | None, int | None]:
    """(model name, physical cores, logical cores) from ``/proc/cpuinfo``."""
    name = ""
    logical = 0
    phys: set[tuple[str, str]] = set()
    cur_pkg = cur_core = None
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            if cur_core is not None:
                phys.add((cur_pkg or "0", cur_core))
            cur_pkg = cur_core = None
            continue
        k, _, v = line.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k == "processor":
            logical += 1
        elif k in ("model name", "hardware", "cpu model") and not name:
            name = v
        elif k == "physical id":
            cur_pkg = v
        elif k == "core id":
            cur_core = v
    return name, (len(phys) or None), (logical or None)


def parse_meminfo(text: str) -> int | None:
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    return int(m.group(1)) // 1024 if m else None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _windows_ram_mb() -> int | None:  # pragma: no cover - Windows only
    class MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    st = MemStatus()
    st.dwLength = ctypes.sizeof(MemStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
        return int(st.ullTotalPhys // (1024 * 1024))
    return None


def cpu_ram(os_name: str, run: Runner = run_cmd, read: Callable[[str], str | None] = _read) -> dict[str, Any]:
    info: dict[str, Any] = {"cpu_name": "", "cores_physical": None, "cores_logical": None, "ram_mb": None}
    if os_name.startswith("linux"):
        txt = read("/proc/cpuinfo")
        if txt:
            name, phys, logical = parse_cpuinfo(txt)
            info["cpu_name"] = name
            info["cores_physical"] = info["cores_physical"] or phys or logical  # ARM cpuinfo has no core ids
            info["cores_logical"] = info["cores_logical"] or logical
        mem = read("/proc/meminfo")
        if mem and not info["ram_mb"]:
            info["ram_mb"] = parse_meminfo(mem)
    elif os_name == "darwin":
        info["cpu_name"] = (run(["sysctl", "-n", "machdep.cpu.brand_string"], TIMEOUT_S) or "").strip()
        info["cores_physical"] = info["cores_physical"] or _int(run(["sysctl", "-n", "hw.physicalcpu"], TIMEOUT_S))
        if not info["ram_mb"]:
            mb = _int(run(["sysctl", "-n", "hw.memsize"], TIMEOUT_S))
            info["ram_mb"] = mb // (1024 * 1024) if mb else None
    elif os_name == "win32":
        out = run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                'Get-CimInstance Win32_Processor | ForEach-Object { "$($_.Name)|$($_.NumberOfCores)" }',
            ],
            TIMEOUT_S + 4,
        )
        if out:
            names, cores = [], 0
            for ln in out.splitlines():
                if "|" in ln:
                    n, c = ln.rsplit("|", 1)
                    names.append(n.strip())
                    cores += _int(c) or 0
            info["cpu_name"] = names[0] if names else ""
            info["cores_physical"] = info["cores_physical"] or cores or None
        if not info["cpu_name"]:
            info["cpu_name"] = _platform.processor()
        if not info["ram_mb"]:
            with contextlib.suppress(Exception):
                info["ram_mb"] = _windows_ram_mb()
    if not (info["cores_physical"] and info["ram_mb"]):
        with contextlib.suppress(Exception):
            import psutil  # optional, not a dependency: only fills what the OS probes missed

            info["cores_physical"] = info["cores_physical"] or psutil.cpu_count(logical=False)
            info["ram_mb"] = info["ram_mb"] or int(psutil.virtual_memory().total // (1024 * 1024))
    info["cores_logical"] = info["cores_logical"] or os.cpu_count()
    if not info["ram_mb"]:
        with contextlib.suppress(ValueError, OSError, AttributeError):
            info["ram_mb"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    return info


def disk_free_mb(path: Path | None = None) -> int | None:
    p = path
    if p is None:
        from away import paths

        p = paths.data_dir()
    while not p.exists() and p.parent != p:  # the data dir may not exist yet: ask about its drive
        p = p.parent
    try:
        return int(shutil.disk_usage(p).free // (1024 * 1024))
    except OSError:
        return None


# ---------------------------------------------------------------- top level
def detect(
    os_name: str | None = None,
    arch: str | None = None,
    run: Runner = run_cmd,
    read: Callable[[str], str | None] = _read,
    nvml_probe: Callable[[str], list[GPU]] | None = None,
    disk: Callable[[], int | None] | None = None,
) -> Hardware:
    """Detect the machine. Never raises: anything that fails is left ``None`` and noted."""
    os_name = os_name or ("linux" if sys.platform.startswith("linux") else sys.platform)
    arch = arch or _platform.machine().lower()
    hw = Hardware(os=os_name, os_release=_platform.release(), arch=arch)
    try:
        info = cpu_ram(os_name, run, read)
        hw.cpu_name = str(info.get("cpu_name") or "")
        hw.cores_physical = info.get("cores_physical")
        hw.cores_logical = info.get("cores_logical")
        hw.ram_mb = info.get("ram_mb")
    except Exception as exc:
        hw.notes.append(f"cpu/ram probe failed: {type(exc).__name__}: {exc}")
    try:
        hw.disk_free_mb = (disk or disk_free_mb)()
    except Exception as exc:
        hw.notes.append(f"disk probe failed: {exc}")

    nvidia: list[GPU] = []
    if os_name in ("win32", "linux"):
        try:
            nvidia = (nvml_probe or nvml)(os_name)
        except Exception as exc:
            hw.notes.append(f"NVML failed: {exc}")
        if not nvidia:
            try:
                nvidia = nvidia_smi(run)
            except Exception as exc:
                hw.notes.append(f"nvidia-smi failed: {exc}")
            if not nvidia:
                hw.notes.append("no NVIDIA driver answered (NVML / nvidia-smi missing, failed or timed out)")
        for g in nvidia:
            if g.compute_capability is None:
                g.compute_capability = guess_cc(g.name)
    others: list[GPU] = []
    try:
        others = generic_gpus(os_name, run)
    except Exception as exc:
        hw.notes.append(f"GPU list failed: {exc}")
    if nvidia:  # NVML/nvidia-smi have the real numbers; keep only the non-NVIDIA adapters from the OS list
        others = [g for g in others if g.vendor != "nvidia"]
    hw.gpus = nvidia + others
    if os_name == "darwin" and arch in ("arm64", "aarch64"):
        hw.gpus.append(apple_gpu(hw.cpu_name, hw.ram_mb, run))
    return hw


@functools.lru_cache(maxsize=1)
def cached() -> Hardware:
    """Detect once per process (``nvidia-smi``/PowerShell cost up to a second or two)."""
    return detect()


def driver_ok_for_cuda12(gpu: GPU, os_name: str) -> tuple[bool, str]:
    """Is the NVIDIA driver new enough for CUDA 12 (what the bundled CTranslate2 is built with)?"""
    if gpu.cuda_driver_version:
        cv = parse_version(gpu.cuda_driver_version)
        if cv and cv[0] >= 12:
            return True, f"driver supports CUDA {gpu.cuda_driver_version}"
        if cv:
            return False, f"driver only supports CUDA {gpu.cuda_driver_version}; CUDA 12 is needed"
    dv = parse_version(gpu.driver_version)
    if not dv:
        return True, "driver version unknown (assuming CUDA 12 works; falls back to CPU if not)"
    need = MIN_DRIVER_WINDOWS if os_name == "win32" else MIN_DRIVER_LINUX
    if dv >= need:
        return True, f"driver {gpu.driver_version} >= {'.'.join(map(str, need))}"
    return False, (
        f"NVIDIA driver {gpu.driver_version} is too old for CUDA 12 (needs >= {'.'.join(map(str, need))}); "
        "update the driver from nvidia.com to use the GPU"
    )
