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
    return tuning.choose_profile(hw, ov, cuda_runtime=cuda_runtime, pack_installed=gpupack.installed())


def effective_asr_model(cfg: Any, profile: tuning.Profile | None = None) -> str:
    """``voice.asr_model`` unless it is ``auto``, else the profile's pick."""
    m = str(cfg.voice.asr_model or "auto")
    if m != "auto":
        return m
    return (profile or current_profile(cfg)).asr_model


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
