"""Train Studio gestures locally, in seconds, starting from a blank model.

* Static poses: the static pose model's MLP shape (:class:`~away.hands.model.NumpyMLP` weights). Trained with
  torch (:func:`away.hands.model.build_torch_models` ``StaticPoseMLP`` + :func:`away.hands.train._fit`) when the
  [train] extra is installed, otherwise with the small numpy trainer below; the runtime is numpy either way.
* Motions: the temporal TCN (``GestureTCN``, exported to ONNX) with torch; without torch an MLP over the same
  window features (numpy), so the installed app can train with nothing extra.

Every group gets a ``nothing`` class (relax rounds + "That was wrong" examples). The held-out score is
group-disjoint by ``session_id`` (one practice round = one group), the rule PR #6 introduced: a round is never
in both the training and the checking part of a fold. With fewer than 3 rounds there is no honest score and the
result says so.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.studio.features import MOTION_FRAMES, group_key, motion_window, parse_group, static_vector
from away.studio.pack import NOTHING, GestureDef, StudioStore

Array = NDArray[np.float64]
MIN_ROUNDS = 3  # practice rounds (sessions) a gesture needs before it can be checked on unseen rounds
MIN_EXAMPLES = 8
SIMILAR_COSINE = 0.92
CONFUSED_RATE = 0.3
MAX_FOLDS = 5


# ---------------------------------------------------------------------- numpy MLP
def train_mlp(
    X: Array,
    y: NDArray[np.int64],
    n_classes: int,
    hidden: tuple[int, ...] = (128, 128),
    epochs: int = 120,
    lr: float = 3e-3,
    l2: float = 1e-4,
    batch: int = 128,
    seed: int = 0,
) -> dict[str, Array]:
    """Adam + softmax cross-entropy on standardised inputs. Returns NumpyMLP weights (w0, b0, ...) with the
    standardisation folded into the first layer, so the runtime feeds raw features."""
    rng = np.random.default_rng(seed)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    sizes = [X.shape[1], *hidden, n_classes]
    W = [rng.normal(0, np.sqrt(2.0 / a), (b, a)) for a, b in zip(sizes, sizes[1:])]
    B = [np.zeros(b) for b in sizes[1:]]
    params = [*W, *B]
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    step = 0
    Y = np.eye(n_classes)[y]
    nL = len(W)
    for _ in range(epochs):
        perm = rng.permutation(len(Z))
        for i in range(0, len(Z), batch):
            idx = perm[i : i + batch]
            acts = [Z[idx]]
            for k in range(nL):
                h = acts[-1] @ W[k].T + B[k]
                acts.append(np.maximum(h, 0.0) if k < nL - 1 else h)
            logits = acts[-1] - acts[-1].max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            g = (p - Y[idx]) / len(idx)
            gW: list[Array] = [np.zeros(1)] * nL
            gB: list[Array] = [np.zeros(1)] * nL
            for k in range(nL - 1, -1, -1):
                gW[k] = g.T @ acts[k] + l2 * W[k]
                gB[k] = g.sum(axis=0)
                if k:
                    g = (g @ W[k]) * (acts[k] > 0)
            step += 1
            grads = [*gW, *gB]
            for j, (prm, gr) in enumerate(zip(params, grads)):
                m1[j] = 0.9 * m1[j] + 0.1 * gr
                m2[j] = 0.999 * m2[j] + 0.001 * gr * gr
                prm -= lr * (m1[j] / (1 - 0.9**step)) / (np.sqrt(m2[j] / (1 - 0.999**step)) + 1e-8)
    weights = {f"w{k}": W[k] for k in range(nL)} | {f"b{k}": B[k] for k in range(nL)}
    return fold_standardisation(weights, mu, sd)


def fold_standardisation(weights: dict[str, Array], mu: Array, sd: Array) -> dict[str, Array]:
    """``W0 (x - mu) / sd + b0`` == ``(W0 / sd) x + (b0 - (W0 / sd) mu)``."""
    w0 = weights["w0"] / sd
    out = dict(weights)
    out["w0"] = w0
    out["b0"] = weights["b0"] - w0 @ mu
    return out


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------- data
@dataclass
class Clip:
    label: str
    session: str
    X: Array
    X2: Array | None
    H2: NDArray[np.bool_] | None
    handedness: str
    fps: float
    number: int  # 1-based position among this label's recordings ("practice example #7")
    path: str = ""


def load_clips(store: StudioStore) -> list[Clip]:
    out: list[Clip] = []
    numbers: dict[str, int] = {}
    for p in store.clip_paths():
        try:
            with np.load(p, allow_pickle=False) as d:
                label = str(d["label"])
                numbers[label] = numbers.get(label, 0) + 1
                out.append(
                    Clip(
                        label=label,
                        session=str(d["session_id"].item()) if "session_id" in d.files else p.name,
                        X=np.asarray(d["X"], dtype=np.float64),
                        X2=np.asarray(d["X2"], dtype=np.float64) if "X2" in d.files else None,
                        H2=np.asarray(d["H2"], dtype=bool) if "H2" in d.files else None,
                        handedness=str(d["handedness"]) if "handedness" in d.files else "Right",
                        fps=float(d["fps"]) if "fps" in d.files else 30.0,
                        number=numbers[label],
                        path=str(p),
                    )
                )
        except Exception:
            continue  # a damaged file must not stop training
    return out


def clip_samples(
    clip: Clip, kind: str, hands: int, rng: np.random.Generator, augment: int = 0, window_s: float = 1.5
) -> list[Array]:
    """Model inputs from one clip: static = every (second) frame; motion = the whole clip (+ ``augment``
    random crops; a long 'nothing' clip is cut into gesture-length windows). Degenerate frames are skipped."""
    out: list[Array] = []
    if kind == "static":
        for i in range(0, len(clip.X), 2 if augment else 1):
            h = clip.X2[i] if clip.X2 is not None and clip.H2 is not None and clip.H2[i] else None
            with contextlib.suppress(ValueError):
                out.append(static_vector(clip.X[i], h, hands))
        return out
    n = len(clip.X)
    spans = [(0, n)]
    if clip.label == NOTHING and clip.fps > 0:
        w = max(8, int(round(window_s * clip.fps)))
        spans = [(s, min(n, s + w)) for s in range(0, max(1, n - w + 1), max(1, w // 2))]
    for s, e in spans:
        crops = [(s, e)]
        for _ in range(augment):
            ln = e - s
            a = s + int(rng.integers(0, max(1, int(ln * 0.15)) + 1))
            b = e - int(rng.integers(0, max(1, int(ln * 0.15)) + 1))
            if b - a >= 4:
                crops.append((a, b))
        for a, b in crops:
            if b - a < 4:
                continue
            with contextlib.suppress(ValueError):
                out.append(
                    motion_window(
                        clip.X[a:b],
                        clip.handedness,
                        clip.X2[a:b] if clip.X2 is not None else None,
                        clip.H2[a:b] if clip.H2 is not None else None,
                        hands,
                    ).reshape(-1)
                )
    return out


def _jitter(X: Array, rng: np.random.Generator) -> Array:
    out: Array = X * rng.uniform(0.92, 1.08, (len(X), 1)) + rng.normal(0, 0.03, X.shape)
    return out


# ---------------------------------------------------------------------- one group
@dataclass
class GroupModel:
    key: str
    classes: list[str]
    backend: str  # "mlp" (numpy weights) | "tcn" (torch, exported to onnx)
    weights: dict[str, Array] = field(default_factory=dict)
    torch_model: Any = None
    mean: Array = field(default_factory=lambda: np.zeros(0))
    sd: Array = field(default_factory=lambda: np.ones(0))

    def predict(self, X: Array) -> Array:
        from away.hands.model import NumpyMLP, softmax

        if self.backend == "tcn":
            import torch

            Z = (X - self.mean) / self.sd
            with torch.no_grad():
                t = torch.tensor(Z.reshape(len(X), MOTION_FRAMES, -1), dtype=torch.float32)
                logits = self.torch_model(t).numpy().astype(np.float64)
            return softmax(logits)
        return NumpyMLP(self.weights, self.classes).predict_proba(X)


def fit_group(
    key: str,
    X: Array,
    y: NDArray[np.int64],
    classes: list[str],
    backend: str,
    seed: int = 0,
    epochs: int | None = None,
) -> GroupModel:
    kind, _hands = parse_group(key)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    if backend == "torch":
        import torch

        from away.hands.model import build_torch_models, mlp_state_to_numpy
        from away.hands.train import _fit

        torch.manual_seed(seed)
        StaticPoseMLP, GestureTCN = build_torch_models()
        Z = (X - mu) / sd
        if kind == "static":
            model = StaticPoseMLP(n_in=X.shape[1], n_out=len(classes))
            tx = torch.tensor(Z, dtype=torch.float32)
        else:
            model = GestureTCN(n_in=X.shape[1] // MOTION_FRAMES, n_out=len(classes), channels=48)
            tx = torch.tensor(Z.reshape(len(Z), MOTION_FRAMES, -1), dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.long)
        model = _fit(model, tx, ty, tx, ty, epochs or (40 if kind == "static" else 30), 3e-3, bs=64)
        model.eval()
        if kind == "static":
            return GroupModel(key, classes, "mlp", fold_standardisation(mlp_state_to_numpy(model), mu, sd), mean=mu)
        return GroupModel(key, classes, "tcn", torch_model=model, mean=mu, sd=sd)
    hidden = (128, 128) if kind == "static" else (64,)
    w = train_mlp(X, y, len(classes), hidden=hidden, epochs=epochs or (80 if kind == "static" else 150), seed=seed)
    return GroupModel(key, classes, "mlp", w, mean=mu)


def group_folds(groups: list[str]) -> list[set[str]]:
    """Deterministic group k-fold: every round lands in exactly one checking fold (none below MIN_ROUNDS)."""
    uniq = sorted(set(groups), key=lambda g: hashlib.sha1(g.encode()).hexdigest())
    k = min(MAX_FOLDS, len(uniq))
    return [set(uniq[i::k]) for i in range(k)] if k >= MIN_ROUNDS else []


@dataclass
class TrainResult:
    ok: bool
    message: str
    score: float | None = None  # clip-level accuracy on practice rounds the model never saw
    per_gesture: dict[str, float] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    backend: str = "numpy"
    seconds: float = 0.0

    @property
    def n_checked(self) -> int:
        return int(sum(sum(r) for r in self.confusion))

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _group_data(
    clips: list[Clip], kind: str, hands: int, names: list[str], rng: np.random.Generator, window_s: float, aug: int
) -> tuple[list[Array], list[int]]:
    classes = [NOTHING, *names]
    X: list[Array] = []
    y: list[int] = []
    for c in clips:
        if c.label not in classes:
            continue
        for v in clip_samples(c, kind, hands, rng, aug, window_s):
            X.append(v)
            y.append(classes.index(c.label))
    return X, y


def _fit_data(
    clips: list[Clip], kind: str, hands: int, names: list[str], rng: np.random.Generator, window_s: float
) -> tuple[Array, NDArray[np.int64]] | None:
    X, y = _group_data(clips, kind, hands, names, rng, window_s, aug=6 if kind == "motion" else 1)
    if len(set(y)) < 2:
        return None
    Xa, ya = np.stack(X), np.array(y, dtype=np.int64)
    if kind == "static":
        return np.concatenate([Xa, _jitter(Xa, rng)]), np.concatenate([ya, ya])
    return Xa, ya


def evaluate_group(
    clips: list[Clip], kind: str, hands: int, names: list[str], backend: str, window_s: float, seed: int = 0
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Group k-fold over practice rounds: ``[(true, predicted)]`` per held-out clip + a split description."""
    rng = np.random.default_rng(seed)
    classes = [NOTHING, *names]
    use = [c for c in clips if c.label in classes]
    folds = group_folds([c.session for c in use])
    split: dict[str, Any] = {
        "rounds": len({c.session for c in use}),
        "folds": len(folds),
        "rounds_per_gesture": {n: len({c.session for c in use if c.label == n}) for n in classes},
    }
    pairs: list[tuple[int, int]] = []
    for fold in folds:
        train_c = [c for c in use if c.session not in fold]
        test_c = [c for c in use if c.session in fold]
        if {c.session for c in train_c} & {c.session for c in test_c}:  # pragma: no cover - by construction
            raise AssertionError("a practice round leaked into both sides of a fold")
        data = _fit_data(train_c, kind, hands, names, rng, window_s)
        if data is None or not test_c:
            continue
        gm = fit_group(group_key(kind, hands), data[0], data[1], classes, backend, seed)
        for c in test_c:
            vs = clip_samples(c, kind, hands, rng, 0, window_s)
            if vs:
                p = gm.predict(np.stack(vs)).mean(axis=0)
                pairs.append((classes.index(c.label), int(np.argmax(p))))
    return pairs, split


def similar_pairs(X: Array, y: NDArray[np.int64], classes: list[str]) -> list[tuple[str, str, float]]:
    """Gesture pairs whose average (standardised) inputs point the same way: likely to be mixed up."""
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    cents = {i: Z[y == i].mean(axis=0) for i in sorted(set(int(v) for v in y)) if classes[i] != NOTHING}
    out = []
    keys = sorted(cents)
    for a_i, a in enumerate(keys):
        for b in keys[a_i + 1 :]:
            va, vb = cents[a], cents[b]
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
            if cos >= SIMILAR_COSINE:
                out.append((classes[a], classes[b], cos))
    return out


def train_studio(
    store: StudioStore,
    gestures: list[GestureDef] | None = None,
    backend: str = "auto",
    seed: int = 0,
    progress: Any = None,
) -> TrainResult:
    """Train every Studio gesture that has recordings and write ``store.model_dir``. Never raises for a data
    problem: the result carries a plain-words message instead."""
    t0 = time.monotonic()
    say = progress or (lambda _text: None)
    gestures = gestures if gestures is not None else store.gestures()
    if backend == "auto":
        backend = "torch" if torch_available() else "numpy"
    clips = load_clips(store)
    have = {c.label for c in clips}
    if not gestures:
        return TrainResult(False, "Teach a gesture first: press 'Teach a new gesture'.")
    if NOTHING not in have:
        return TrainResult(False, "The app also needs to see your hand doing nothing special. Record a relax round.")
    ready = [g for g in gestures if g.name in have]
    if not ready:
        return TrainResult(False, "Record some practice examples first.")
    warnings = [f"'{g.name}' has no practice examples yet, so it is left out." for g in gestures if g.name not in have]
    rng = np.random.default_rng(seed)
    groups: dict[str, list[GestureDef]] = {}
    for g in ready:
        groups.setdefault(group_key(g.kind, g.hands), []).append(g)
    motion_names = {g.name for g in ready if g.kind == "motion"}
    lengths = [len(c.X) / c.fps for c in clips if c.fps > 0 and c.label in motion_names]
    window_s = float(np.clip(np.median(lengths), 0.6, 3.0)) if lengths else 1.5
    all_pairs: list[tuple[str, str]] = []
    trained: list[tuple[GroupModel, dict[str, Any]]] = []
    split_all: dict[str, Any] = {"rounds_per_gesture": {}, "folds": 0}
    examples: dict[str, Array] = {}
    for key, defs in sorted(groups.items()):
        kind, hands = parse_group(key)
        names = [g.name for g in defs]
        classes = [NOTHING, *names]
        say(f"Checking {', '.join(names)} on practice rounds it has not seen...")
        pairs, split = evaluate_group(clips, kind, hands, names, backend, window_s, seed)
        all_pairs += [(classes[t], classes[p]) for t, p in pairs]
        split_all["folds"] = max(split_all["folds"], split["folds"])
        split_all["rounds_per_gesture"].update(split["rounds_per_gesture"])
        say(f"Learning {', '.join(names)}...")
        data = _fit_data(clips, kind, hands, names, rng, window_s)
        if data is None:
            warnings.append(f"Not enough different examples to learn {', '.join(names)}.")
            continue
        gm = fit_group(key, data[0], data[1], classes, backend, seed)
        trained.append((gm, {"kind": kind, "hands": hands, "window_s": window_s}))
        for a, b, _cos in similar_pairs(data[0], data[1], classes):
            warnings.append(f"'{a}' and '{b}' look very alike, so the app may mix them up. Make them more different.")
        emb, lab, num = [], [], []
        for c in clips:
            if c.label in names:
                vs = clip_samples(c, kind, hands, rng, 0, window_s)
                if vs:
                    emb.append(np.mean(vs, axis=0) if kind == "static" else vs[0])
                    lab.append(c.label)
                    num.append(c.number)
        if emb:
            examples[f"emb_{key}"] = np.stack(emb)
            examples[f"lab_{key}"] = np.array(lab)
            examples[f"num_{key}"] = np.array(num)
    if not trained:
        return TrainResult(False, "Nothing could be learned yet; record more practice examples.", warnings=warnings)
    for g in ready:
        n = sum(1 for c in clips if c.label == g.name)
        rounds = int(split_all["rounds_per_gesture"].get(g.name, 0))
        if rounds < MIN_ROUNDS:
            warnings.append(
                f"'{g.name}' was practised in {rounds} round(s). Do {MIN_ROUNDS - rounds} more so the app can check "
                "itself on rounds it has not seen."
            )
        if n < MIN_EXAMPLES:
            warnings.append(f"'{g.name}' has {n} practice examples; {MIN_EXAMPLES} or more works better.")
    labels = [NOTHING, *[g.name for g in ready]]
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in all_pairs:
        cm[labels.index(t), labels.index(p)] += 1
    per = {lab: float(cm[i, i] / cm[i].sum()) for i, lab in enumerate(labels) if cm[i].sum()}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j and NOTHING not in (a, b) and cm[i].sum() and cm[i, j] / cm[i].sum() >= CONFUSED_RATE:
                warnings.append(f"When you did '{a}' the app often thought it was '{b}'.")
    score = float(np.trace(cm) / cm.sum()) if cm.sum() else None
    res = TrainResult(
        True,
        "All set! Your gestures are ready." if score is not None else "Learned. Do more practice rounds to see a score.",
        score=score,
        per_gesture=per,
        labels=labels,
        confusion=cm.tolist(),
        split=split_all,
        warnings=list(dict.fromkeys(warnings)),
        backend=backend,
    )
    save_models(store, trained, examples, res)
    res.seconds = round(time.monotonic() - t0, 2)
    (store.model_dir / "metrics.json").write_text(json.dumps(res.to_dict(), indent=2))
    return res


def save_models(
    store: StudioStore, trained: list[tuple[GroupModel, dict[str, Any]]], examples: dict[str, Array], res: TrainResult
) -> None:
    tmp = store.model_dir.with_name("model.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    groups = []
    for gm, info in trained:
        entry: dict[str, Any] = {"key": gm.key, "classes": gm.classes, "backend": gm.backend, **info}
        if gm.backend == "tcn":
            import torch

            from away.hands.model import export_onnx

            dim = len(gm.mean) // MOTION_FRAMES
            meta = {"classes": gm.classes, "frames": MOTION_FRAMES, "mean": gm.mean.tolist(), "sd": gm.sd.tolist()}
            export_onnx(gm.torch_model, torch.zeros(1, MOTION_FRAMES, dim), tmp / f"{gm.key}.onnx", meta)
            entry["file"] = f"{gm.key}.onnx"
        else:
            np.savez(tmp / f"{gm.key}.npz", classes=np.array(gm.classes), mean=gm.mean, **gm.weights)  # type: ignore[arg-type]
            entry["file"] = f"{gm.key}.npz"
        groups.append(entry)
    if examples:
        np.savez(tmp / "examples.npz", **examples)  # type: ignore[arg-type]
    meta_all = {"version": 1, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": res.backend, "groups": groups}
    (tmp / "studio.json").write_text(json.dumps(meta_all, indent=2))
    shutil.rmtree(store.model_dir, ignore_errors=True)
    tmp.replace(store.model_dir)


def retrain_builtin(models_dir: Path, recordings: Path) -> dict[str, Any]:
    """'Improve the built-in gestures with my examples': the existing personal fine-tune of the bundled TCN
    (:func:`away.hands.train.calibrate`); needs the [train] extra (torch)."""
    if not torch_available():
        raise RuntimeError("needs the training add-on")
    from away.hands.train import calibrate

    return calibrate(models_dir, recordings)
