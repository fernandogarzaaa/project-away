"""Gesture classifiers.

* :class:`StaticPoseMLP` – single-frame pose (point/palm/fist/two/ok/none),
  trained on real HaGRID landmark annotations.
* :class:`GestureTCN` – temporal convolutional network over landmark
  sequences for the dynamic gestures in :data:`away.hands.synthetic.GESTURES`.

Torch is only needed for training/export (``pip install project-away[train]``).
At runtime the app uses ONNX Runtime (:class:`OnnxClassifier`) or, for the MLP,
a pure-numpy forward pass (:class:`NumpyMLP`), so installers stay small.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

STATIC_CLASSES = ("point", "palm", "fist", "two", "ok", "none")
SEQ_FEATURES = 21 * 2 + 2
STATIC_FEATURES = 21 * 2


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: NDArray[np.float64] = e / e.sum(axis=-1, keepdims=True)
    return out


def build_torch_models() -> tuple[Any, Any]:
    """Return (StaticPoseMLP, GestureTCN) classes; imports torch lazily."""
    import torch
    from torch import nn

    class StaticPoseMLP(nn.Module):
        def __init__(self, n_in: int = STATIC_FEATURES, n_out: int = len(STATIC_CLASSES), hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_out),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TemporalBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, dilation: int):
            super().__init__()
            self.pad = 2 * dilation  # causal: left pad only
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=3, dilation=dilation)
            self.act = nn.ReLU()
            self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.conv(nn.functional.pad(x, (self.pad, 0)))
            return self.act(y + self.res(x))

    class GestureTCN(nn.Module):
        def __init__(self, n_in: int = SEQ_FEATURES, n_out: int = 9, channels: int = 64):
            super().__init__()
            self.blocks = nn.Sequential(
                TemporalBlock(n_in, channels, 1),
                TemporalBlock(channels, channels, 2),
                TemporalBlock(channels, channels, 4),
            )
            self.head = nn.Linear(channels, n_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
            h = self.blocks(x.transpose(1, 2))
            return self.head(h.mean(dim=2))

    return StaticPoseMLP, GestureTCN


def export_onnx(model: Any, example: Any, path: Path, meta: dict[str, Any]) -> Path:
    import torch

    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dyn = {"x": {0: "batch"}, "logits": {0: "batch"}}
    if example.dim() == 3:
        dyn["x"][1] = "time"
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes=dyn,
        opset_version=17,
        dynamo=False,
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


class OnnxClassifier:
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta_path = Path(path).with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.classes: list[str] = list(self.meta.get("classes", []))

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        logits = self.sess.run(["logits"], {"x": np.asarray(x, dtype=np.float32)})[0]
        return softmax(np.asarray(logits, dtype=np.float64))


class NumpyMLP:
    """Dependency-free forward pass for the exported static MLP (Linear/ReLU stack)."""

    def __init__(self, weights: dict[str, NDArray[np.float64]], classes: list[str]) -> None:
        self.layers: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        i = 0
        while f"w{i}" in weights:
            self.layers.append((np.asarray(weights[f"w{i}"]), np.asarray(weights[f"b{i}"])))
            i += 1
        if not self.layers:
            raise ValueError("no layers in weights")
        self.classes = classes

    @classmethod
    def load(cls, path: Path) -> NumpyMLP:
        data = np.load(path)
        classes = [str(c) for c in data["classes"]]
        return cls({k: data[k] for k in data.files if k != "classes"}, classes)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        h = np.atleast_2d(np.asarray(x, dtype=np.float64))
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return softmax(h)


def mlp_state_to_numpy(model: Any) -> dict[str, NDArray[np.float64]]:
    """Extract Linear weights in order from a torch Sequential MLP."""
    from torch import nn

    out: dict[str, NDArray[np.float64]] = {}
    i = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            out[f"w{i}"] = m.weight.detach().cpu().numpy().astype(np.float64)
            out[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float64)
            i += 1
    return out
