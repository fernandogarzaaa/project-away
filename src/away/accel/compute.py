"""Where hand and face tracking run (v0.1.24, ``hands.compute: auto | gpu | cpu``).

What can run where:

* **MediaPipe on the processor** (the default everywhere, always available as the fallback);
* **MediaPipe's own GPU delegate** on Linux and macOS (MediaPipe has no GPU support on Windows: "GPU support is
  currently limited to Ubuntu platforms", and its Windows wheels raise "GPU Delegate is not yet supported");
* **the same models as ONNX on ONNX Runtime** with a GPU execution provider: DirectML on Windows (any DirectX 12
  graphics card, NVIDIA / AMD / Intel; ``onnxruntime-directml``) or CUDA (``onnxruntime-gpu``). The models are
  Google's MediaPipe hand / face models converted to ONNX (Apache-2.0, like the originals), a one-time download of
  about 23 MB (:mod:`away.accel.models`), checked against pinned SHA-256 sums;
* v0.1.35: **the same ONNX models on Core ML** on macOS (``CoreMLExecutionProvider``, part of the standard macOS
  ``onnxruntime`` wheel): Apple's Neural Engine and GPU. Core ML is not faster for every model (measured on an
  Apple M5: the hand landmark model 0.5 ms vs 3.4 ms on the processor, the gaze network 1.1 ms vs 17 ms, but the
  palm detector and all three face models slower), so each model is timed on Core ML and on the processor when its
  session is created and keeps whichever was faster (:func:`measured_session`); when no model kept Core ML,
  tracking goes back to MediaPipe on the processor and :func:`status` says it was slower.

``auto`` uses the graphics card only when a GPU path is available *and* its models are already downloaded, so it
never downloads by itself; ``gpu`` downloads them when needed; ``cpu`` never touches the graphics card. Whatever
fails to start falls back to MediaPipe on the processor, and :func:`status` says why in plain words (shown in
Settings > Models and in the self-test).
"""

from __future__ import annotations

import contextlib
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away.accel import models as gm

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
COREML_PROVIDER = "CoreMLExecutionProvider"  # v0.1.35: macOS only (Apple Neural Engine / GPU)
CPU_PROVIDER = "CPUExecutionProvider"
PROVIDER_TITLES = {
    "CUDAExecutionProvider": "NVIDIA CUDA",
    "DmlExecutionProvider": "DirectML",
    COREML_PROVIDER: "Apple Neural Engine / GPU (Core ML)",
}
# v0.1.35: providers that must measure faster than the processor, per model, before they are kept. CUDA / DirectML
# are used as soon as they start (as before); Core ML runs some of these models slower than the CPU (above).
MEASURED_PROVIDERS = (COREML_PROVIDER,)
BENCH_WARMUP = 3  # Core ML's first runs include its own compilation / Neural Engine set-up
BENCH_RUNS = 10
_last: dict[str, Any] = {}  # what the trackers last started with (for status / the self-test)
_measured: dict[tuple[str, str], dict[str, Any]] = {}  # (model path, provider) -> what the timing decided


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


def gpu_providers(platform: str | None = None) -> tuple[str, ...]:
    """The accelerated providers tried on this OS, best first (v0.1.35: Core ML on macOS only; the Windows and Linux
    list is unchanged)."""
    plat = platform or sys.platform
    return (*GPU_PROVIDERS, COREML_PROVIDER) if plat == "darwin" else GPU_PROVIDERS


def gpu_provider(providers: list[str], platform: str | None = None) -> str:
    for p in gpu_providers(platform):
        if p in providers:
            return p
    return ""


# ------------------------------------------------------------------------------- sessions (v0.1.35)
def coreml_cache_dir(data_dir: Path | None = None) -> Path:
    """Where Core ML keeps the compiled models (a second start loads in ~20 ms instead of ~0.3-1 s). The cache is
    keyed by the model file; the files here are pinned by SHA-256, so a stale entry cannot meet a changed model."""
    return gm.folder(data_dir) / "coreml-cache"


def provider_options(provider: str, data_dir: Path | None = None) -> dict[str, str]:
    """ONNX Runtime options for ``provider`` ({} for the ones that need none).

    Core ML: ``MLComputeUnits=ALL`` lets Core ML spread the model over the Neural Engine, the GPU and the CPU, and
    ``ModelCacheDirectory`` keeps the compiled model. ``ModelFormat`` stays at its default (NeuralNetwork): the
    newer MLProgram format fails to compile the palm detector and all three face models ("Unable to parse ML
    Program", onnxruntime 1.31), and was slower for the hand landmark and gaze models that it did compile."""
    if provider != COREML_PROVIDER:
        return {}
    opts = {"MLComputeUnits": "ALL"}
    with contextlib.suppress(Exception):  # no cache is only slower to start
        d = coreml_cache_dir(data_dir)
        d.mkdir(parents=True, exist_ok=True)
        opts["ModelCacheDirectory"] = str(d)
    return opts


def ort_session(path: Path | str, providers: list[str], data_dir: Path | None = None, threads: int = 0) -> Any:
    """An ONNX Runtime session with :func:`provider_options`; an onnxruntime too old for an option gets the plain
    provider names instead (it then runs with the provider's defaults)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if threads:
        opts.intra_op_num_threads = int(threads)
    full: list[Any] = [(p, o) if (o := provider_options(p, data_dir)) else p for p in providers]
    try:
        return ort.InferenceSession(str(path), sess_options=opts, providers=full)
    except Exception:
        if full == list(providers):
            raise
        return ort.InferenceSession(str(path), sess_options=opts, providers=list(providers))


_DTYPES = {"tensor(float)": "float32", "tensor(float16)": "float16", "tensor(int64)": "int64", "tensor(int32)": "int32"}


def time_session(sess: Any, runs: int = BENCH_RUNS, warmup: int = BENCH_WARMUP) -> float:
    """Median milliseconds of one run on a dummy input of the model's own shape (``inf`` when it fails). These are
    convolutional networks whose speed does not depend on the pixel values, so zeros measure what a frame costs."""
    import numpy as np

    try:
        feed = {
            i.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in (i.shape or [])], _DTYPES.get(i.type, "float32")
            )
            for i in sess.get_inputs()
        }
        for _ in range(warmup):
            sess.run(None, feed)
        times = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            sess.run(None, feed)
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(statistics.median(times))
    except Exception:
        return math.inf


def session_providers(sess: Any) -> list[str]:
    fn = getattr(sess, "get_providers", None)
    return list(fn()) if callable(fn) else []


def measured_session(
    path: Path | str,
    providers: list[str],
    data_dir: Path | None = None,
    threads: int = 0,
    make: Callable[..., Any] | None = None,
    timer: Callable[[Any], float] | None = None,
) -> Any:
    """A session for ``providers`` (best first). When the first one is in :data:`MEASURED_PROVIDERS` (Core ML), the
    model is also opened on the remaining providers (the processor), both are timed with :func:`time_session`, and
    the faster session is returned; a provider that fails to start or is not faster is dropped. The decision is
    remembered per model file for this run, so restarting a tracker does not measure again."""
    make = make or ort_session
    timer = timer or time_session
    first = providers[0] if providers else ""
    if first not in MEASURED_PROVIDERS:
        return make(path, providers, data_dir, threads)
    rest = [p for p in providers if p != first] or [CPU_PROVIDER]
    key = (str(path), first)
    known = _measured.get(key)
    if known is not None:
        return make(path, known["providers"], data_dir, threads)
    name = Path(str(path)).name
    try:
        fast = make(path, providers, data_dir, threads)
    except Exception as exc:
        _measured[key] = {"model": name, "providers": rest, "kept": False, "why": f"did not start ({exc})"[:160]}
        return make(path, rest, data_dir, threads)
    if first not in session_providers(fast):  # onnxruntime quietly left it out
        _measured[key] = {"model": name, "providers": rest, "kept": False, "why": "did not start"}
        return fast
    slow = make(path, rest, data_dir, threads)
    t_fast, t_cpu = timer(fast), timer(slow)
    kept = t_fast < t_cpu
    _measured[key] = {
        "model": name,
        "providers": list(providers) if kept else rest,
        "kept": kept,
        "ms": round(t_fast, 2) if math.isfinite(t_fast) else None,
        "cpu_ms": round(t_cpu, 2) if math.isfinite(t_cpu) else None,
        "why": "faster than the processor" if kept else "slower than the processor",
    }
    return fast if kept else slow


def measurements() -> list[dict[str, Any]]:
    """What :func:`measured_session` decided so far (Settings, the self-test)."""
    return [dict(v) for v in _measured.values()]


def _uses(sessions: list[Any], provider: str) -> bool:
    return any(provider in session_providers(s) for s in sessions if s is not None)


def _slower_note(provider: str, models: list[Path]) -> str | None:
    """Plain words when ``provider`` started for ``models`` but every one measured slower than the processor
    (None when it was not measured, e.g. it never started)."""
    found = [_measured.get((str(p), provider)) or {} for p in models]
    rows = [m for m in found if m.get("ms") is not None and m.get("cpu_ms") is not None]
    if not rows:
        return None
    times = ", ".join(f"{m['model']} {m['ms']} vs {m['cpu_ms']} ms" for m in rows)
    return (
        f"{PROVIDER_TITLES.get(provider, provider)} measured slower than the processor ({times}); using the processor"
    )


def _split_note(c: Choice, sessions: list[Any], names: list[str]) -> str:
    """For a measured provider: which models it kept and which run on the processor because they were faster there
    ("" otherwise, so the choice's own reason is shown)."""
    if c.provider not in MEASURED_PROVIDERS:
        return ""
    on = [n for s, n in zip(sessions, names) if c.provider in session_providers(s)]
    off = [n for s, n in zip(sessions, names) if c.provider not in session_providers(s)]
    note = f"{PROVIDER_TITLES.get(c.provider, c.provider)} runs the {', '.join(on)}"
    verb = "stays" if len(off) == 1 else "stay"
    return note + (f"; the {', '.join(off)} {verb} on the processor (faster there)" if off else "")


class SlowerThanProcessor(RuntimeError):
    """v0.1.35: the accelerated provider started, but no model ran faster on it than on the processor."""


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
    prov = gpu_provider(list(providers or []), plat)
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
        "measured": measurements(),  # v0.1.35: per model, Core ML vs the processor
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
            test = OnnxHandLandmarker(paths["palm_detector"], paths["hand_landmark"], 1, order, data_dir=data_dir)
            if not _uses([test.palm, test.lmk], c.provider):
                # v0.1.35: Core ML started but lost the timing on both hand models (measured, not broken)
                slower = _slower_note(c.provider, [paths["palm_detector"], paths["hand_landmark"]])
                if slower:
                    raise SlowerThanProcessor(slower)
                raise RuntimeError(f"{PROVIDER_TITLES.get(c.provider, c.provider)} did not start")

            def factory(n: int) -> Any:
                return OnnxHandLandmarker(
                    paths["palm_detector"], paths["hand_landmark"], n, order, min_conf, min_conf, data_dir=data_dir
                )

            note_running(c, _split_note(c, [test.palm, test.lmk], ["palm detector", "hand landmarks"]))
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
    except SlowerThanProcessor as exc:
        why = str(exc)
        say(why)
        note_running(Choice("mediapipe-cpu", "", why))
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
                lm = OnnxFaceLandmarker(
                    p["face_detector"],
                    p["face_landmark"],
                    p["face_blendshape"],
                    max_faces,
                    [c.provider, "CPUExecutionProvider"],
                    data_dir=data_dir,
                )
                # v0.1.35: when Core ML lost the timing on every face model (all three do on an Apple M5),
                # MediaPipe's own processor path is the better-tested one
                if c.provider not in MEASURED_PROVIDERS or _uses([lm.det, lm.mesh, lm.bs], c.provider):
                    return lm
    from away.face.landmarker import FaceLandmarker

    return FaceLandmarker(model_path, max_faces)
