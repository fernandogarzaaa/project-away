"""Where hand and face tracking run (v0.1.24, ``hands.compute: auto | gpu | cpu``).

What can run where:

* **MediaPipe on the processor** (the default everywhere, always available as the fallback);
* **MediaPipe's own GPU delegate** on Linux and macOS (MediaPipe has no GPU support on Windows: "GPU support is
  currently limited to Ubuntu platforms", and its Windows wheels raise "GPU Delegate is not yet supported");
* **the same models as ONNX on ONNX Runtime** with a GPU execution provider: DirectML on Windows (any DirectX 12
  graphics card, NVIDIA / AMD / Intel; ``onnxruntime-directml``) or CUDA (``onnxruntime-gpu``). The models are
  Google's MediaPipe hand / face models converted to ONNX (Apache-2.0, like the originals), a one-time download of
  about 23 MB (:mod:`away.accel.models`), checked against pinned SHA-256 sums.

``auto`` uses the graphics card only when a GPU path is available *and* its models are already downloaded, so it
never downloads by itself; ``gpu`` downloads them when needed; ``cpu`` never touches the graphics card. Whatever
fails to start falls back to MediaPipe on the processor, and :func:`status` says why in plain words (shown in
Settings > Models and in the self-test).
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away.accel import models as gm

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
PROVIDER_TITLES = {"CUDAExecutionProvider": "NVIDIA CUDA", "DmlExecutionProvider": "DirectML"}
_last: dict[str, Any] = {}  # what the trackers last started with (for status / the self-test)


@dataclass(frozen=True)
class Choice:
    backend: str  # mediapipe-cpu | mediapipe-gpu | onnx-gpu
    provider: str  # the ONNX Runtime provider ("" for MediaPipe)
    reason: str  # plain words

    @property
    def using(self) -> str:
        return "processor" if self.backend == "mediapipe-cpu" else "graphics card"


def ort_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def gpu_provider(providers: list[str]) -> str:
    for p in GPU_PROVIDERS:
        if p in providers:
            return p
    return ""


def mediapipe_gpu_platform(platform: str | None = None) -> bool:
    plat = platform or sys.platform
    return plat.startswith("linux") or plat == "darwin"


def choose(
    setting: str,
    *,
    platform: str | None = None,
    providers: list[str] | None = None,
    models_ready: bool = False,
    mediapipe_gpu: bool | None = None,
) -> Choice:
    """The pure decision (tests feed every combination)."""
    plat = platform or sys.platform
    if setting == "cpu":
        return Choice("mediapipe-cpu", "", "set to use the processor")
    prov = gpu_provider(list(providers or []))
    mp_gpu = mediapipe_gpu_platform(plat) if mediapipe_gpu is None else bool(mediapipe_gpu)
    if prov and models_ready:
        return Choice("onnx-gpu", prov, f"{PROVIDER_TITLES.get(prov, prov)} with the ONNX models")
    if prov and setting == "gpu":
        return Choice("onnx-gpu", prov, "needs the graphics-card models (download)")
    if mp_gpu and setting == "gpu":
        return Choice("mediapipe-gpu", "", "MediaPipe's GPU delegate")
    if not prov:
        why = (
            "no graphics-card runtime found (install onnxruntime-directml, or use the installer)"
            if plat.startswith("win")
            else "no ONNX Runtime GPU provider found"
        )
        return Choice("mediapipe-cpu", "", why)
    return Choice("mediapipe-cpu", "", "the graphics-card models are not downloaded yet")


def current_choice(cfg: Any, data_dir: Path | None = None, providers: list[str] | None = None) -> Choice:
    setting = str(getattr(getattr(cfg, "hands", None), "compute", "auto") or "auto")
    prov = ort_providers() if providers is None else providers
    return choose(setting, providers=prov, models_ready=gm.ready("hands", data_dir))


def status(cfg: Any, data_dir: Path | None = None, providers: list[str] | None = None) -> dict[str, Any]:
    """For Settings > Models and the self-test: ``{"setting", "using", "provider", "reason", "models_ready",
    "running"}``; ``running`` is what the trackers really started with (after any fallback)."""
    prov = ort_providers() if providers is None else providers
    c = current_choice(cfg, data_dir, prov)
    run = dict(_last)
    using = run.get("using") or c.using
    reason = run.get("reason") or c.reason
    return {
        "setting": str(getattr(cfg.hands, "compute", "auto")),
        "using": using,
        "backend": run.get("backend") or c.backend,
        "provider": PROVIDER_TITLES.get(gpu_provider(prov), "") if gpu_provider(prov) else "",
        "reason": reason,
        "models_ready": gm.ready("hands", data_dir),
        "download_mb": round(gm.download_size("hands") / 1e6),
        "running": bool(run),
        "cpu_saving_pct": run.get("cpu_saving_pct"),
    }


def note_running(choice: Choice, reason: str = "") -> None:
    _last.clear()
    _last.update(backend=choice.backend, using=choice.using, reason=reason or choice.reason, provider=choice.provider)


# ------------------------------------------------------------------------------------------- factories
def hand_factory(
    cfg: Any,
    model_path: Path,
    data_dir: Path | None = None,
    min_conf: float = 0.5,
    on_note: Callable[[str], Any] | None = None,
    providers: list[str] | None = None,
) -> tuple[Callable[[int], Any], Callable[[Any], Any]] | None:
    """``(factory, image_fn)`` for :class:`away.hands.tracker.HandTracker` when the graphics card should run hand
    tracking, or None for MediaPipe on the processor (the tracker's default). Downloads the ONNX models first when
    ``hands.compute`` is ``gpu``. Never raises: anything that fails falls back to None with a note."""
    say = on_note or (lambda _m: None)
    prov = ort_providers() if providers is None else providers
    try:
        c = current_choice(cfg, data_dir, prov)
        if c.backend == "onnx-gpu":
            if not gm.ready("hands", data_dir):
                gm.download("hands", data_dir)
            from away.accel.onnx_hands import OnnxHandLandmarker

            paths = gm.paths("hands", data_dir)
            order = [c.provider, "CPUExecutionProvider"]
            test = OnnxHandLandmarker(paths["palm_detector"], paths["hand_landmark"], 1, order)
            if c.provider not in test.providers:
                raise RuntimeError(f"{PROVIDER_TITLES.get(c.provider, c.provider)} did not start")

            def factory(n: int) -> Any:
                return OnnxHandLandmarker(paths["palm_detector"], paths["hand_landmark"], n, order, min_conf, min_conf)

            note_running(c)
            return factory, lambda rgb: rgb
        if c.backend == "mediapipe-gpu":
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            def mp_factory(n: int) -> Any:
                opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.GPU),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=n,
                    min_hand_detection_confidence=min_conf,
                    min_hand_presence_confidence=min_conf,
                    min_tracking_confidence=min_conf,
                )
                return vision.HandLandmarker.create_from_options(opts)

            mp_factory(1).close()  # fails here (not mid-tracking) when the GPU delegate cannot start

            def image_fn(rgb: Any) -> Any:
                import numpy as np

                return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

            note_running(c)
            return mp_factory, image_fn
        note_running(c)
        return None
    except Exception as exc:
        why = f"the graphics card could not start ({str(exc)[:120]}); using the processor"
        say(why)
        note_running(Choice("mediapipe-cpu", "", why))
        return None


def face_landmarker(
    cfg: Any,
    model_path: Path,
    data_dir: Path | None = None,
    max_faces: int = 4,
    providers: list[str] | None = None,
) -> Any:
    """A face landmarker on the graphics card (ONNX) when hand tracking uses one and the face models are there;
    otherwise MediaPipe's :class:`away.face.landmarker.FaceLandmarker` on the processor."""
    prov = ort_providers() if providers is None else providers
    c = current_choice(cfg, data_dir, prov)
    if c.backend == "onnx-gpu":
        with contextlib.suppress(Exception):
            if not gm.ready("face", data_dir) and str(getattr(cfg.hands, "compute", "auto")) == "gpu":
                gm.download("face", data_dir)
            if gm.ready("face", data_dir):
                from away.accel.onnx_face import OnnxFaceLandmarker

                p = gm.paths("face", data_dir)
                return OnnxFaceLandmarker(
                    p["face_detector"], p["face_landmark"], p["face_blendshape"], max_faces, [c.provider, "CPUExecutionProvider"]
                )
    from away.face.landmarker import FaceLandmarker

    return FaceLandmarker(model_path, max_faces)
