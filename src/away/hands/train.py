"""Training / evaluation / calibration entry points (need the [train] extra)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.hands.dataset import fixed_length, load_hagrid_dir, load_recordings, subject_split
from away.hands.landmarks import sequence_features
from away.hands.model import (
    SEQ_FEATURES,
    STATIC_CLASSES,
    STATIC_FEATURES,
    build_torch_models,
    export_onnx,
    mlp_state_to_numpy,
)
from away.hands.synthetic import GESTURES, make_dataset

SEQ_FRAMES = 24


def confusion(y_true: NDArray[np.int64], y_pred: NDArray[np.int64], n: int) -> NDArray[np.int64]:
    m = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m


def macro_f1(cm: NDArray[np.int64]) -> float:
    f1s = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def _fit(model: Any, Xtr: Any, ytr: Any, Xva: Any, yva: Any, epochs: int, lr: float, bs: int = 256) -> Any:
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item() if len(Xva) else 0.0
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _augment_static(X: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Small in-plane rotation + scale + jitter on (N, 42) normalized landmarks."""
    pts = X.reshape(len(X), 21, 2)
    ang = rng.uniform(-0.3, 0.3, len(X))
    c, s = np.cos(ang), np.sin(ang)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    pts = np.einsum("nij,nkj->nki", rot, pts) * rng.uniform(0.9, 1.1, (len(X), 1, 1))
    pts = pts + rng.normal(0, 0.02, pts.shape)
    out: NDArray[np.float64] = pts.reshape(len(X), -1)
    return out


def train_static(
    hagrid_dir: Path, out_dir: Path, max_per_class: int = 3000, epochs: int = 40, seed: int = 0
) -> dict[str, Any]:
    """Train the static pose MLP on real HaGRID landmarks; evaluate on held-out *subjects*."""
    import torch

    torch.manual_seed(seed)
    X, y, users = load_hagrid_dir(hagrid_dir, max_per_class=max_per_class, seed=seed)
    tr, va, te = subject_split(users)
    rng = np.random.default_rng(seed)
    Xtr = np.concatenate([X[tr], _augment_static(X[tr], rng)])
    ytr = np.concatenate([y[tr], y[tr]])
    StaticPoseMLP, _ = build_torch_models()
    model = StaticPoseMLP()
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)  # noqa: E731
    model = _fit(model, t(Xtr), t(ytr, torch.long), t(X[va]), t(y[va], torch.long), epochs, 2e-3)
    with torch.no_grad():
        pred = model(t(X[te])).argmax(1).numpy()
    cm = confusion(y[te], pred, len(STATIC_CLASSES))
    present = sorted(set(int(v) for v in y))
    metrics = {
        "model": "static_pose_mlp",
        "data": "HaGRID v2 annotations_with_landmarks (test split JSONs), subject-disjoint split by user_id",
        "classes": list(STATIC_CLASSES),
        "classes_present": [STATIC_CLASSES[i] for i in present],
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "n_test_subjects": len({u for u, m in zip(users, te) if m}),
        "test_accuracy": float((pred == y[te]).mean()),
        "test_macro_f1": macro_f1(cm[np.ix_(present, present)]),
        "confusion": cm.tolist(),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = mlp_state_to_numpy(model)
    np.savez(out_dir / "static_pose.npz", classes=np.array(STATIC_CLASSES), **weights)  # type: ignore[arg-type]
    export_onnx(
        model,
        torch.zeros(1, STATIC_FEATURES),
        out_dir / "static_pose.onnx",
        {"classes": list(STATIC_CLASSES), "input": "static_features (42,)"},
    )
    (out_dir / "static_pose.metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def seqs_to_features(
    seqs: list[NDArray[np.float64]], hands: list[str], frames: int = SEQ_FRAMES
) -> NDArray[np.float64]:
    return np.stack([sequence_features(fixed_length(s, frames), h) for s, h in zip(seqs, hands)])


def train_temporal(
    out_dir: Path,
    recordings: Path | None = None,
    n_synthetic: int = 150,
    epochs: int = 25,
    seed: int = 0,
    init_from: Path | None = None,
    lr: float = 2e-3,
) -> dict[str, Any]:
    """Train (or fine-tune) the temporal TCN.

    Data = synthetic sequences (+ your ``away record`` clips if given). Reported
    accuracy is split by source: ``synthetic_holdout_accuracy`` is NOT a
    real-world number; ``recorded_holdout_accuracy`` is, when recordings exist.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _, GestureTCN = build_torch_models()
    model = GestureTCN(SEQ_FEATURES, len(GESTURES))
    if init_from is not None and init_from.exists():
        model.load_state_dict(torch.load(init_from, map_location="cpu"))
    Xs: list[NDArray[np.float64]] = []
    ys: list[int] = []
    src: list[str] = []
    if n_synthetic:
        S, Sy, Sh = make_dataset(n_synthetic, SEQ_FRAMES, seed)
        Xs.extend(seqs_to_features(list(S), Sh))
        ys.extend(int(v) for v in Sy)
        src.extend(["synthetic"] * len(Sy))
    if recordings is not None and Path(recordings).exists():
        R, Rl, Rh = load_recordings(Path(recordings))
        keep = [i for i, lab in enumerate(Rl) if lab in GESTURES]
        if keep:
            Xs.extend(seqs_to_features([R[i] for i in keep], [Rh[i] for i in keep]))
            ys.extend(GESTURES.index(Rl[i]) for i in keep)
            src.extend(["recorded"] * len(keep))
    if not Xs:
        raise ValueError("no training data")
    X = np.stack(Xs)
    y = np.array(ys, dtype=np.int64)
    source = np.array(src)
    test = rng.random(len(y)) < 0.2
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)  # noqa: E731
    model = _fit(model, t(X[~test]), t(y[~test], torch.long), t(X[test]), t(y[test], torch.long), epochs, lr, bs=64)
    with torch.no_grad():
        pred = model(t(X[test])).argmax(1).numpy() if test.any() else np.array([], dtype=np.int64)
    metrics: dict[str, Any] = {"model": "gesture_tcn", "classes": list(GESTURES), "frames": SEQ_FRAMES}
    for name in ("synthetic", "recorded"):
        m = test & (source == name)
        if m.any():
            metrics[f"{name}_holdout_accuracy"] = float((pred[m[test]] == y[m]).mean())
            metrics[f"{name}_holdout_n"] = int(m.sum())
    metrics["note"] = "synthetic_* numbers measure fit to procedural data only, not real-world accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "gesture_tcn.pt")
    export_onnx(
        model,
        torch.zeros(1, SEQ_FRAMES, SEQ_FEATURES),
        out_dir / "gesture_tcn.onnx",
        {"classes": list(GESTURES), "frames": SEQ_FRAMES, "input": "sequence_features (T, 44)"},
    )
    (out_dir / "gesture_tcn.metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def calibrate(models_dir: Path, recordings: Path, epochs: int = 15) -> dict[str, Any]:
    """Short personal fine-tune: start from the shipped TCN, mix your clips with a little synthetic data."""
    base = models_dir / "gesture_tcn.pt"
    return train_temporal(models_dir, recordings=recordings, n_synthetic=30, epochs=epochs, init_from=base, lr=5e-4)
