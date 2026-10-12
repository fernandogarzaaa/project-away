"""Training / evaluation / calibration entry points (need the [train] extra)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.hands.dataset import fixed_length, load_hagrid_dir, load_recordings_with_groups, subject_split
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



def temporal_split(
    source: NDArray[np.str_], groups: list[str], rng: np.random.Generator,
    test_frac: float = 0.2, val_frac: float = 0.1,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Split by sample for synthetic data and by recording/session group for real clips."""
    train = np.zeros(len(source), dtype=bool)
    val = np.zeros(len(source), dtype=bool)
    test = np.zeros(len(source), dtype=bool)
    synthetic_idx = np.flatnonzero(source == "synthetic")
    if len(synthetic_idx):
        shuffled = rng.permutation(synthetic_idx)
        n_test = int(round(len(shuffled) * test_frac))
        n_val = int(round(len(shuffled) * val_frac))
        if len(shuffled) >= 3:
            n_test, n_val = max(1, n_test), max(1, n_val)
        n_test = min(n_test, max(0, len(shuffled) - 1))
        n_val = min(n_val, max(0, len(shuffled) - n_test - 1))
        test[shuffled[:n_test]] = True
        val[shuffled[n_test:n_test + n_val]] = True
        train[shuffled[n_test + n_val:]] = True
    recorded_idx = np.flatnonzero(source == "recorded")
    if len(recorded_idx):
        group_to_indices: dict[str, list[int]] = {}
        for idx in recorded_idx:
            group_to_indices.setdefault(groups[int(idx)], []).append(int(idx))
        unique_groups = np.array(sorted(group_to_indices), dtype=object)
        # Fewer than three groups cannot support distinct train/validation/test partitions.
        if len(unique_groups) >= 3:
            shuffled_groups = rng.permutation(unique_groups).tolist()
            n_test = max(1, int(round(len(shuffled_groups) * test_frac)))
            n_test = min(n_test, len(shuffled_groups) - 2)
            n_val = max(1, int(round(len(shuffled_groups) * val_frac)))
            n_val = min(n_val, len(shuffled_groups) - n_test - 1)
            test_groups = set(shuffled_groups[:n_test])
            val_groups = set(shuffled_groups[n_test:n_test + n_val])
            for group, indices in group_to_indices.items():
                mask = test if group in test_groups else val if group in val_groups else train
                mask[indices] = True
        else:
            train[recorded_idx] = True
    return train, val, test


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

    Data = synthetic sequences (+ your recorded clips if given). Validation selects
    the checkpoint; the final test partition is evaluated once after training. Synthetic
    accuracy is procedural-data fit, not a real-world number. Recorded test scores are
    group-disjoint only when recording metadata identifies independent sessions/subjects.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _, GestureTCN = build_torch_models()
    rec_data = load_recordings_with_groups(Path(recordings)) if recordings is not None and Path(recordings).exists() else None
    classes = temporal_classes(rec_data[1] if rec_data else [])
    model = GestureTCN(SEQ_FEATURES, len(classes))
    if init_from is not None and init_from.exists():
        state = torch.load(init_from, map_location="cpu")
        own = model.state_dict()
        # a new custom gesture changes the output layer: keep every weight whose shape still matches
        model.load_state_dict({k: v for k, v in state.items() if k in own and own[k].shape == v.shape}, strict=False)
    Xs: list[NDArray[np.float64]] = []
    ys: list[int] = []
    src: list[str] = []
    groups: list[str] = []
    if n_synthetic:
        S, Sy, Sh = make_dataset(n_synthetic, SEQ_FRAMES, seed)
        Xs.extend(seqs_to_features(list(S), Sh))
        ys.extend(int(v) for v in Sy)
        src.extend(["synthetic"] * len(Sy))
        groups.extend(f"synthetic-{i}" for i in range(len(Sy)))
    if rec_data is not None:
        R, Rl, Rh, Rg = rec_data
        keep = [i for i, lab in enumerate(Rl) if lab in classes]
        if keep:
            Xs.extend(seqs_to_features([R[i] for i in keep], [Rh[i] for i in keep]))
            ys.extend(classes.index(Rl[i]) for i in keep)
            src.extend(["recorded"] * len(keep))
            groups.extend(Rg[i] for i in keep)
    if not Xs:
        raise ValueError("no training data")
    X = np.stack(Xs)
    y = np.array(ys, dtype=np.int64)
    source = np.array(src)
    train_mask, val_mask, test_mask = temporal_split(source, groups, rng)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)  # noqa: E731
    model = _fit(
        model, t(X[train_mask]), t(y[train_mask], torch.long),
        t(X[val_mask]), t(y[val_mask], torch.long), epochs, lr, bs=64,
    )
    # Test data is evaluated only after validation has selected the model checkpoint.
    with torch.no_grad():
        pred = model(t(X[test_mask])).argmax(1).numpy() if test_mask.any() else np.array([], dtype=np.int64)
    metrics: dict[str, Any] = {
        "model": "gesture_tcn", "classes": list(classes), "frames": SEQ_FRAMES,
        "split_counts": {"train": int(train_mask.sum()), "validation": int(val_mask.sum()), "test": int(test_mask.sum())},
        "recorded_groups": len({groups[i] for i in range(len(groups)) if source[i] == "recorded"}),
    }
    for name in ("synthetic", "recorded"):
        m = test_mask & (source == name)
        if m.any():
            metrics[f"{name}_holdout_accuracy"] = float((pred[m[test_mask]] == y[m]).mean())
            metrics[f"{name}_holdout_n"] = int(m.sum())
    if np.any(source == "recorded") and not np.any(test_mask & (source == "recorded")):
        metrics["recorded_holdout_warning"] = (
            "No recorded holdout score: at least three distinct recording/session groups are required. "
            "Record more independent sessions and set session_id or subject_id in recording metadata."
        )
    metrics["note"] = (
        "Synthetic holdout measures procedural-data fit, not real-world accuracy. Recorded holdout is group-disjoint "
        "only to the extent that recording session/subject metadata correctly identifies independent groups."
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "gesture_tcn.pt")
    export_onnx(
        model,
        torch.zeros(1, SEQ_FRAMES, SEQ_FEATURES),
        out_dir / "gesture_tcn.onnx",
        {"classes": list(classes), "frames": SEQ_FRAMES, "input": "sequence_features (T, 44)"},
    )
    (out_dir / "gesture_tcn.metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def temporal_classes(recorded_labels: list[str]) -> list[str]:
    """Built-in gesture classes followed by any custom gestures found in your recordings (sorted)."""
    from away.bindings import LABEL_RE

    custom = sorted({lab for lab in recorded_labels if lab not in GESTURES and LABEL_RE.match(lab)})
    return [*GESTURES, *custom]


def calibrate(models_dir: Path, recordings: Path, epochs: int = 15) -> dict[str, Any]:
    """Short personal fine-tune: start from the shipped TCN, mix your clips with a little synthetic data."""
    base = models_dir / "gesture_tcn.pt"
    return train_temporal(models_dir, recordings=recordings, n_synthetic=30, epochs=epochs, init_from=base, lr=5e-4)
