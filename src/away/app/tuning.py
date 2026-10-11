"""Hardware -> settings: the automatic performance profile (pure functions, no I/O).

:func:`choose_profile` maps a detected :class:`~away.app.hardware.Hardware` plus the user's explicit
overrides to everything that depends on the machine:

* speech recognition: device (``cuda``/``cpu``), CTranslate2 ``compute_type``, Whisper model,
  ``cpu_threads`` and beam size,
* camera capture: resolution and frame rate, and how often hand / face landmarks are computed
  (``hands_every`` / ``face_every``: 1 = every frame, 2 = every other frame...).

Every decision appends a plain-English reason so ``away hardware`` and the Settings window can show
*why*. Explicit settings (``voice.device``, ``voice.compute_type``, ``voice.asr_model``,
``performance.*`` numbers, ``performance.mode: cpu|gpu``) always win over the automatic choice.

Facts the rules rely on (checked against CTranslate2 4.8.2's build scripts):

* the PyPI wheels for Windows and Linux x86-64 are built with CUDA 12.8, ``WITH_CUDNN=OFF`` and
  ``CUDA_DYNAMIC_LOADING=ON``: the GPU needs only cuBLAS 12 (``cublas64_12.dll`` /
  ``libcublas.so.12``), loaded at runtime, and an NVIDIA driver that supports CUDA 12;
* the macOS and Linux aarch64 wheels have no CUDA at all;
* ``float16`` needs compute capability >= 5.3 and is fast from 7.0 (Volta/Turing tensor cores); ``int8``
  on the GPU needs >= 6.1; ``int8_float16`` needs >= 7.0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from away.app.hardware import GPU, Hardware, driver_ok_for_cuda12

MODES = ("auto", "cpu", "gpu")

# fp16 weight sizes on disk (MB) = roughly what the model takes in VRAM at float16
WEIGHTS_MB = {
    "large-v3-turbo": 1620,
    "large-v3": 3100,
    "distil-large-v3": 1510,
    "medium": 1530,
    "small": 485,
    "base": 150,
    "tiny": 75,
}
WEIGHT_FACTOR = {"float16": 1.0, "int8_float16": 0.55, "int8": 0.55, "int8_float32": 0.55, "float32": 2.0}
VRAM_HEADROOM_MB = 1200  # CUDA context + activations + what the desktop already uses
MIN_VRAM_MB = 1536
CUDA_ARCHES = ("x86_64", "amd64")  # CTranslate2's CUDA wheels exist only for these
MIN_CC = (5, 0)


@dataclass
class Overrides:
    """The user's explicit settings ("auto"/0 = let the profile decide)."""

    mode: str = "auto"  # performance.mode
    device: str = "auto"  # voice.device
    compute_type: str = "auto"  # voice.compute_type
    asr_model: str = "auto"  # voice.asr_model
    cpu_threads: int = 0
    beam_size: int = 0
    camera_width: int = 0
    camera_height: int = 0
    camera_fps: int = 0
    hands_every: int = 0
    face_every: int = 0

    @classmethod
    def from_config(cls, cfg: Any) -> Overrides:
        p, v = cfg.performance, cfg.voice
        return cls(
            mode=p.mode,
            device=v.device,
            compute_type=v.compute_type,
            asr_model=v.asr_model,
            cpu_threads=p.cpu_threads,
            beam_size=p.beam_size,
            camera_width=p.camera_width,
            camera_height=p.camera_height,
            camera_fps=p.camera_fps,
            hands_every=p.hands_every,
            face_every=p.face_every,
        )


@dataclass
class Profile:
    device: str = "cpu"
    compute_type: str = "int8"
    asr_model: str = "small"
    cpu_threads: int = 4
    beam_size: int = 5
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    hands_every: int = 1
    face_every: int = 1
    tier: str = "cpu-mid"  # gpu | cpu-high | cpu-mid | cpu-low | cpu-min
    gpu: str | None = None  # name of the GPU that is (or could be) used
    gpu_capable: bool = False  # an NVIDIA GPU that CUDA 12 can use is present
    gpu_pack: str = "not-needed"  # not-needed | recommended | installed | unavailable
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return f"{self.asr_model} on {self.device.upper()} ({self.compute_type}), beam {self.beam_size}"


def gpu_compute_type(cc: tuple[int, int] | None, vram_mb: int | None) -> tuple[str, str]:
    if cc is None:
        return "float16", "compute capability unknown: float16"
    if cc >= (7, 0):
        if vram_mb is not None and vram_mb < 3800:
            return "int8_float16", f"compute {cc[0]}.{cc[1]} with <4 GB VRAM: int8 weights, float16 math"
        return "float16", f"compute {cc[0]}.{cc[1]} (tensor cores): float16"
    if cc >= (6, 1):
        return "int8", f"compute {cc[0]}.{cc[1]} (no fast float16): int8"
    return "float32", f"compute {cc[0]}.{cc[1]}: float32 (int8/float16 unsupported or slow)"


def model_fits_vram(model: str, compute_type: str, vram_mb: int | None) -> bool:
    if vram_mb is None:
        return True
    need = WEIGHTS_MB.get(model, 1620) * WEIGHT_FACTOR.get(compute_type, 1.0) + VRAM_HEADROOM_MB
    return need <= vram_mb


def gpu_usable(hw: Hardware, gpu: GPU | None, ct2_cuda: bool | None = None) -> tuple[bool, str]:
    """Can the bundled CTranslate2 run Whisper on this GPU at all (ignoring whether cuBLAS is present)?"""
    if gpu is None:
        if hw.apple_silicon:
            return False, "Apple Silicon: CTranslate2 has no Metal backend, Whisper runs on the CPU (int8)"
        if hw.os == "darwin":
            return False, "macOS: no CUDA, Whisper runs on the CPU"
        return False, "no NVIDIA GPU detected"
    if hw.os not in ("win32", "linux"):
        return False, f"{gpu.name}: CUDA is only used on Windows and Linux"
    if hw.arch not in CUDA_ARCHES:
        return False, f"{gpu.name}: CTranslate2 has no CUDA build for {hw.arch}"
    if gpu.source == "lspci" and not gpu.driver_version:
        return False, f"{gpu.name}: no NVIDIA driver answered (nvidia-smi); install the NVIDIA driver to use it"
    if ct2_cuda is False:
        return False, f"{gpu.name}: the installed CTranslate2 was built without CUDA"
    if gpu.compute_capability is not None and gpu.compute_capability < MIN_CC:
        return False, f"{gpu.name}: compute capability {gpu.cc_str} is below 5.0 (unsupported by CUDA 12)"
    ok, why = driver_ok_for_cuda12(gpu, hw.os)
    if not ok:
        return False, why
    if gpu.vram_mb is not None and gpu.vram_mb < MIN_VRAM_MB:
        return False, f"{gpu.name}: only {gpu.vram_mb} MB VRAM (needs >= {MIN_VRAM_MB} MB)"
    return True, f"{gpu.name} ({_gb(gpu.vram_mb)} VRAM, compute {gpu.cc_str}; {why})"


def _gb(mb: int | None) -> str:
    return f"{mb / 1024:.1f} GB" if mb else "? GB"


def cpu_tier(hw: Hardware) -> str:
    cores, ram = hw.cores, hw.ram_mb or 8192
    if cores >= 12 and ram >= 15000:
        return "cpu-high"
    if cores >= 4 and ram >= 7000:
        return "cpu-mid"
    if cores >= 2 and ram >= 3500:
        return "cpu-low"
    return "cpu-min"


# tier -> (model, beam)
CPU_ASR = {
    "cpu-high": ("large-v3-turbo", 2),
    "cpu-mid": ("small", 3),
    "cpu-low": ("base", 1),
    "cpu-min": ("tiny", 1),
}
# camera load by CPU strength: (width, height, fps, hands_every, face_every)
CAMERA = {
    "cpu-high": (640, 480, 30, 1, 1),
    "cpu-mid": (640, 480, 30, 1, 2),
    "cpu-low": (480, 360, 20, 1, 3),
    "cpu-min": (320, 240, 15, 1, 4),
}
DOWNSIZE = ("large-v3-turbo", "small", "base", "tiny")


def _fits_disk(model: str, hw: Hardware) -> bool:
    return hw.disk_free_mb is None or WEIGHTS_MB.get(model, 0) + 500 <= hw.disk_free_mb


def choose_profile(
    hw: Hardware,
    ov: Overrides | None = None,
    cuda_runtime: bool = False,
    ct2_cuda: bool | None = None,
    pack_installed: bool = False,
) -> Profile:
    """Pick the settings for ``hw``.

    ``cuda_runtime``: cuBLAS 12 can be loaded (GPU pack installed, pip ``nvidia-cublas-cu12``, or a CUDA
    toolkit on PATH). ``ct2_cuda``: the installed CTranslate2 has CUDA (``None`` = unknown, assume yes on
    Windows/Linux x86-64). ``pack_installed``: the GPU pack is in the data dir (for the status line).
    """
    ov = ov or Overrides()
    p = Profile(reasons=[])
    r = p.reasons
    tier = cpu_tier(hw)
    ram = f"{(hw.ram_mb or 0) / 1024:.0f} GB RAM" if hw.ram_mb else "RAM unknown"
    r.append(f"CPU: {hw.cores} cores ({hw.cores_logical or '?'} threads), {ram} -> {tier}")

    gpu = hw.nvidia
    usable, why = gpu_usable(hw, gpu, ct2_cuda)
    p.gpu_capable = usable
    p.gpu = gpu.name if gpu else None
    if usable:
        p.gpu_pack = "installed" if pack_installed else ("not-needed" if cuda_runtime else "recommended")
    elif gpu is not None:
        p.gpu_pack = "unavailable"

    mode = ov.mode if ov.mode in MODES else "auto"
    want_gpu = False
    if ov.device not in ("auto", ""):
        want_gpu = ov.device == "cuda"
        r.append(f"voice.device: {ov.device} (explicit)")
    elif mode == "cpu":
        r.append("performance.mode: cpu (explicit)")
    elif mode == "gpu":
        want_gpu = True
        r.append("performance.mode: gpu (explicit)")
        if not usable:
            r.append(f"warning: {why}; dictation falls back to the CPU if CUDA cannot start")
        elif not cuda_runtime:
            r.append("warning: CUDA libraries (cuBLAS 12) not found: run `away gpu install`; CPU fallback until then")
    elif usable and cuda_runtime:
        want_gpu = True
        r.append(f"GPU: {why}")
    elif usable:
        r.append(f"GPU: {why}, but the CUDA libraries (cuBLAS 12) are missing -> CPU for now")
        r.append("run `away gpu install` (or Settings > Performance) to enable the GPU")
    else:
        r.append(f"GPU: {why}")

    if want_gpu:
        p.device, p.tier = "cuda", "gpu"
        cc = gpu.compute_capability if gpu else None
        vram = gpu.vram_mb if gpu else None
        p.compute_type, why_ct = gpu_compute_type(cc, vram)
        r.append(why_ct)
        model = next((m for m in DOWNSIZE if model_fits_vram(m, p.compute_type, vram)), "tiny")
        if model != DOWNSIZE[0]:
            r.append(
                f"{_gb(vram)} VRAM: {model} (large-v3-turbo needs ~{_vram_need('large-v3-turbo', p.compute_type)} MB)"
            )
        else:
            r.append(f"{_gb(vram)} VRAM: large-v3-turbo fits (~{_vram_need(model, p.compute_type)} MB)")
        p.asr_model, p.beam_size = model, 5
        p.cpu_threads = max(1, min(4, hw.cores))
    else:
        p.device, p.tier, p.compute_type = "cpu", tier, "int8"
        p.asr_model, p.beam_size = CPU_ASR[tier]
        p.cpu_threads = max(1, min(hw.cores - 1, 8)) if hw.cores > 1 else 1
        r.append(
            f"CPU speech: {p.asr_model} int8, beam {p.beam_size}, "
            f"{p.cpu_threads} thread{'s' if p.cpu_threads != 1 else ''}"
        )

    if not _fits_disk(p.asr_model, hw):
        smaller = next((m for m in DOWNSIZE[DOWNSIZE.index(p.asr_model) + 1 :] if _fits_disk(m, hw)), "tiny")
        r.append(f"only {hw.disk_free_mb} MB free disk: {smaller} instead of {p.asr_model}")
        p.asr_model = smaller

    w, h, fps, he, fe = CAMERA[tier]
    if p.device == "cpu" and tier == "cpu-mid" and hw.cores < 6:
        fe = 3  # Whisper competes with MediaPipe for the same few cores
    p.camera_width, p.camera_height, p.camera_fps, p.hands_every, p.face_every = w, h, fps, he, fe
    face = "every frame" if fe == 1 else f"1 in {fe} frames"
    r.append(f"camera: {w}x{h} @ {fps} fps, hand tracking every frame, face tracking {face}")

    _apply_overrides(p, ov)
    return p


def _vram_need(model: str, compute_type: str) -> int:
    return round(WEIGHTS_MB.get(model, 1620) * WEIGHT_FACTOR.get(compute_type, 1.0) + VRAM_HEADROOM_MB)


def _apply_overrides(p: Profile, ov: Overrides) -> None:
    r = p.reasons
    if ov.device not in ("auto", ""):
        p.device = ov.device
        if p.device == "cpu" and p.compute_type not in ("int8", "float32", "int8_float32"):
            p.compute_type = "int8"
    if ov.compute_type not in ("auto", "", "default"):
        p.compute_type = ov.compute_type
        r.append(f"voice.compute_type: {ov.compute_type} (explicit)")
    if ov.asr_model not in ("auto", ""):
        if ov.asr_model != p.asr_model:
            r.append(f"voice.asr_model: {ov.asr_model} (explicit; auto would pick {p.asr_model})")
        p.asr_model = ov.asr_model
    for name in (
        "cpu_threads",
        "beam_size",
        "camera_width",
        "camera_height",
        "camera_fps",
        "hands_every",
        "face_every",
    ):
        val = getattr(ov, name)
        if isinstance(val, int) and val > 0:
            setattr(p, name, val)
            r.append(f"performance.{name}: {val} (explicit)")
