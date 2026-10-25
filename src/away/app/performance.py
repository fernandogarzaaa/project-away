"""Glue between the config, hardware detection, the GPU pack and the tuning profile."""

from __future__ import annotations

from typing import Any

from away.app import gpupack, hardware, tuning


def current_profile(cfg: Any, hw: hardware.Hardware | None = None, cuda_runtime: bool | None = None) -> tuning.Profile:
    """The profile for this machine and config (detection is cached per process)."""
    hw = hw or hardware.cached()
    ov = tuning.Overrides.from_config(cfg)
    if cuda_runtime is None:
        # only try to load cuBLAS when there is a GPU worth using (it is a large library)
        usable, _ = tuning.gpu_usable(hw, hw.nvidia)
        wants = ov.mode != "cpu" and ov.device != "cpu"
        cuda_runtime = bool(usable and wants and gpupack.cublas_available())
    return tuning.choose_profile(
        hw, ov, cuda_runtime=cuda_runtime, pack_installed=gpupack.installed(), apple_gpu=apple_gpu(cfg, hw)
    )


def asr_backend(cfg: Any) -> str:
    """``voice.asr_backend`` as :func:`away.voice.asr.make_asr` should get it (v0.1.35): ``auto`` with
    ``performance.mode: cpu`` or ``voice.device: cpu`` set explicitly means faster-whisper on the CPU, not MLX."""
    backend = str(cfg.voice.asr_backend or "auto")
    if backend == "auto" and (cfg.performance.mode == "cpu" or cfg.voice.device == "cpu"):
        return "faster-whisper"
    return backend


def apple_gpu(cfg: Any, hw: hardware.Hardware | None = None) -> bool:
    """Speech recognition will run on the Apple GPU (MLX): Apple Silicon, mlx-whisper installed, and
    ``voice.asr_backend`` auto or mlx-whisper (v0.1.35)."""
    from away.voice import asr_mlx

    try:
        if hw is not None and not hw.apple_silicon:
            return False
        return (
            asr_mlx.resolve_backend(asr_backend(cfg)) == asr_mlx.MLX_BACKEND
            and asr_mlx.apple_silicon()
            and asr_mlx.mlx_installed()
        )
    except Exception:
        return False


def effective_asr_model(cfg: Any, profile: tuning.Profile | None = None) -> str:
    """``voice.asr_model`` unless it is ``auto``, else the profile's pick."""
    m = str(cfg.voice.asr_model or "auto")
    if m != "auto":
        return m
    return (profile or current_profile(cfg)).asr_model


# rough cost order (lower = heavier) to compare a chosen model with what the CPU can sustain
MODEL_WEIGHT = {
    "large-v3": 0,
    "large-v3-turbo": 0,
    "distil-large-v3": 0,
    "medium": 0.5,
    "small": 1,
    "base": 2,
    "tiny": 3,
}


def cpu_profile(cfg: Any, hw: hardware.Hardware | None = None) -> tuning.Profile:
    """The profile this machine gets when speech runs on the CPU (what a CUDA fallback should use)."""
    ov = tuning.Overrides.from_config(cfg)
    ov.mode, ov.device, ov.asr_model = "cpu", "cpu", "auto"
    return tuning.choose_profile(hw or hardware.cached(), ov, cuda_runtime=False, pack_installed=gpupack.installed())


def heavier(model: str, than: str) -> bool:
    return MODEL_WEIGHT.get(model, 1) < MODEL_WEIGHT.get(than, 1)


def asr_plan(cfg: Any, profile: tuning.Profile, cpu: tuning.Profile | None = None) -> tuple[str, str, str | None]:
    """``(model, cpu_fallback_model, notice)`` for starting speech recognition.

    * GPU profile: the chosen model, and the CPU pick for a CUDA fallback.
    * CPU profile with a model heavier than the CPU can sustain (e.g. large-v3-turbo carried over from an
      old config, or chosen while the GPU pack is missing): the CPU pick instead, with a notice saying how
      to get the big model back (install the GPU pack). ``voice.device: cpu`` set explicitly keeps the
      user's model: they asked for exactly that.
    """
    model = effective_asr_model(cfg, profile)
    cpu = cpu or (profile if profile.device == "cpu" and cfg.voice.asr_model == "auto" else cpu_profile(cfg))
    cpu_model = cpu.asr_model
    if profile.device != "cpu" or cfg.voice.device == "cpu" or not heavier(model, cpu_model):
        return model, cpu_model, None
    hint = (
        "install the GPU pack (Settings > Performance) to use it on your NVIDIA GPU"
        if profile.gpu_capable
        else "set voice.device: cpu to force it anyway"
    )
    note = f"{model} is too slow for this CPU, so speech recognition uses {cpu_model}; {hint}"
    return cpu_model, cpu_model, note


def gpu_pack_offer(cfg: Any, profile: tuning.Profile | None = None, frozen: bool | None = None) -> str | None:
    """Why to offer the GPU pack now, or ``None``: an NVIDIA GPU that can run CUDA 12, mode auto/gpu, the
    pack is not installed and cuBLAS cannot be loaded otherwise. Only frozen apps download it (pip installs
    get the ``pip install`` line)."""
    from away.app.frozen import is_frozen

    p = profile or current_profile(cfg)
    if cfg.performance.mode == "cpu" or cfg.voice.device == "cpu":
        return None
    if not p.gpu_capable or p.gpu_pack != "recommended":
        return None
    if not (is_frozen() if frozen is None else frozen):
        return None
    return f"{p.gpu}: download the NVIDIA GPU pack (cuBLAS, ~570 MB once) to run speech recognition on the GPU"


def describe(hw: hardware.Hardware, p: tuning.Profile) -> list[str]:
    """Human-readable lines for `away hardware` and the Settings window."""
    lines = [
        f"OS:    {hw.os} {hw.os_release} ({hw.arch})",
        f"CPU:   {hw.cpu_name or 'unknown'}: {hw.cores_physical or '?'} cores / {hw.cores_logical or '?'} threads",
        f"RAM:   {(hw.ram_mb or 0) / 1024:.1f} GB" if hw.ram_mb else "RAM:   unknown",
        f"Disk:  {(hw.disk_free_mb or 0) / 1024:.1f} GB free" if hw.disk_free_mb is not None else "Disk:  unknown",
    ]
    if not hw.gpus:
        lines.append("GPU:   none detected")
    for g in hw.gpus:
        bits = [g.name]
        if g.vram_mb:
            bits.append(f"{g.vram_mb / 1024:.1f} GB VRAM")
        if g.compute_capability:
            bits.append(f"compute {g.cc_str}")
        if g.driver_version:
            bits.append(f"driver {g.driver_version}")
        if g.cuda_driver_version:
            bits.append(f"CUDA {g.cuda_driver_version}")
        lines.append("GPU:   " + ", ".join(bits) + (f"  [{g.source}]" if g.source else ""))
    lines.append("")
    lines.append(f"Profile ({p.tier}): {p.summary()}")
    if p.asr_backend == "mlx-whisper":  # v0.1.35
        lines.append(f"  engine=mlx-whisper (Apple GPU, float16) model={p.asr_model}")
        lines.append(f"  CPU fallback: device={p.device} compute_type={p.compute_type}")
    else:
        lines.append(f"  device={p.device} compute_type={p.compute_type} model={p.asr_model}")
    lines.append(f"  cpu_threads={p.cpu_threads} beam_size={p.beam_size}")
    lines.append(
        f"  camera={p.camera_width}x{p.camera_height}@{p.camera_fps} "
        f"hands_every={p.hands_every} face_every={p.face_every}"
    )
    lines.append(f"  GPU pack: {p.gpu_pack}")
    lines.append("Why:")
    lines += [f"  - {r}" for r in p.reasons]
    return lines
