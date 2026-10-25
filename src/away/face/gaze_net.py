"""Optional learned gaze network (L2CS-Net formulation) on the Face Landmarker's face crop.

The network predicts the gaze direction (yaw, pitch) of a face crop. Its two outputs become two extra
features of the same per-user calibration regression (:mod:`away.face.gaze`), so the screen map and
look-to-jump work unchanged; it only makes the features richer. No face detector is added: the crop comes
from the face box the Face Landmarker already computed.

Models (optional downloads, :mod:`away.app.optional_models`): the ONNX exports published by
github.com/yakhyo/gaze-estimation ("MobileGaze", MIT code; an L2CS-Net style 90-bin classifier per angle,
448x448 input, ImageNet normalisation). ``l2cs`` = the ResNet-18 backbone (the L2CS-Net architecture family),
``mobilegaze`` = MobileOne-S0 (about 5x cheaper). **The weights were trained on Gaze360, whose license is
research-only**, so they are opt-in downloads and never bundled.

:class:`GazeNet` is rate-limited (``max_hz``) and runs onnxruntime on CUDA when it is available, else on the
CPU. v0.1.35: on macOS it also tries Core ML (Neural Engine / GPU) and keeps it only when it measures faster than the
CPU (an Apple M5: L2CS ResNet-18 1.1 ms on Core ML vs 17 ms on the CPU). The session is injected, so the tests use a
fake one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

BINS = 90
BIN_WIDTH = 4.0  # degrees
ANGLE_OFFSET = 180.0
INPUT_SIZE = 448
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_SCALE = 1.25  # the face box (landmark extent) padded a little: the networks were trained on face crops


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def decode(logits: np.ndarray) -> float:
    """90-bin logits -> angle in radians (softmax expectation, 4 degree bins from -180 degrees)."""
    probs = softmax(np.asarray(logits, dtype=np.float64).reshape(-1)[:BINS])
    deg = float(np.sum(probs * np.arange(BINS))) * BIN_WIDTH - ANGLE_OFFSET
    return math.radians(deg)


def crop_box(box: tuple[float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Square pixel crop (x0, y0, x1, y1) around a normalised face box (cx, cy, size), clamped to the image."""
    cx, cy, size = box
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(size)) or size <= 0:
        return None
    half = size * CROP_SCALE * max(width, height) / 2.0
    x0, x1 = round(cx * width - half), round(cx * width + half)
    y0, y1 = round(cy * height - half), round(cy * height + half)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize with OpenCV when present, else nearest-neighbour numpy (tests, minimal installs)."""
    try:
        import cv2

        return np.asarray(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR))
    except ImportError:  # pragma: no cover - OpenCV is installed with the hands/face extras
        h, w = img.shape[:2]
        ys = (np.arange(size) * h / size).astype(int)
        xs = (np.arange(size) * w / size).astype(int)
        return img[ys][:, xs]


def preprocess(rgb: np.ndarray, box: tuple[float, float, float], size: int = INPUT_SIZE) -> np.ndarray | None:
    """RGB frame + face box -> (1, 3, size, size) float32 tensor, or None when the crop is unusable."""
    h, w = rgb.shape[:2]
    cb = crop_box(box, w, h)
    if cb is None:
        return None
    x0, y0, x1, y1 = cb
    crop = resize(np.ascontiguousarray(rgb[y0:y1, x0:x1, :3]), size).astype(np.float32) / 255.0
    crop = (crop - MEAN) / STD
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def providers(available: list[str] | None = None, platform: str | None = None) -> list[str]:
    """CUDA first when onnxruntime offers it (the GPU build), then DirectML, then (v0.1.35, macOS only) Core ML,
    then the CPU; the same list as hand tracking (:func:`away.accel.compute.gpu_providers`)."""
    from away.accel.compute import gpu_providers

    if available is None:
        try:
            import onnxruntime as ort

            available = list(ort.get_available_providers())
        except Exception:  # pragma: no cover - onnxruntime missing
            available = []
    out = [p for p in gpu_providers(platform) if p in available]
    return [*out, "CPUExecutionProvider"]


def gpu_available(available: list[str] | None = None, platform: str | None = None) -> bool:
    return len(providers(available, platform)) > 1


def default_session(path: Path) -> Any:  # pragma: no cover - needs onnxruntime + the model file
    from away.accel import compute

    # 2 threads keep CPU use bounded next to MediaPipe and Whisper; v0.1.35: Core ML is kept only when it measures
    # faster than this CPU session (CUDA / DirectML open as before)
    return compute.measured_session(path, providers(), threads=2)


class GazeNet:
    """Rate-limited gaze network. ``update(t, rgb, box)`` returns the latest (yaw, pitch) in radians; it only
    runs the network when ``1/max_hz`` seconds passed since the last run (otherwise the previous answer)."""

    def __init__(
        self,
        path: Path,
        name: str = "l2cs",
        max_hz: float = 10.0,
        session_factory: Callable[[Path], Any] = default_session,
    ) -> None:
        self.path = path
        self.name = name
        self.max_hz = max_hz
        self.session = session_factory(path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = list(getattr(inp, "shape", []) or [])
        self.size = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else INPUT_SIZE
        outs = [o.name for o in self.session.get_outputs()]
        # the exports name their outputs; fall back to the documented order (yaw, pitch)
        self.yaw_out = outs.index("yaw") if "yaw" in outs else 0
        self.pitch_out = outs.index("pitch") if "pitch" in outs else 1
        self.last: tuple[float, float] | None = None
        self.last_t: float | None = None
        self.runs = 0
        self.last_ms = 0.0

    def due(self, t: float) -> bool:
        return self.last_t is None or t - self.last_t >= 1.0 / max(self.max_hz, 1e-6)

    def update(self, t: float, rgb: Any, box: tuple[float, float, float] | None) -> tuple[float, float] | None:
        if box is None or rgb is None:
            return self.last
        if not self.due(t):
            return self.last
        x = preprocess(np.asarray(rgb), box, self.size)
        self.last_t = t
        if x is None:
            return self.last
        import time

        t0 = time.perf_counter()
        outs = self.session.run(None, {self.input_name: x})
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.runs += 1
        self.last = (decode(outs[self.yaw_out]), decode(outs[self.pitch_out]))
        return self.last

    def fresh(self, t: float, max_age: float = 0.5) -> tuple[float, float] | None:
        if self.last is None or self.last_t is None or t - self.last_t > max_age:
            return None
        return self.last
