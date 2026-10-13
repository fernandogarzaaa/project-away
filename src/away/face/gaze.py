"""Eye tracking and the spatial screen map (pure logic: numpy only, no devices).

"Spatial awareness" here means knowing where the user's screens are and where on them the user is looking.
A webcam cannot measure that directly, so the *screen map is the calibration*: the user looks at dots on
every selected monitor (``away gaze calibrate``), and a regularised regression learns the mapping from what
the camera sees to virtual-desktop pixels.

Features (:func:`gaze_features`), all from the Face Landmarker result the app already computes:

* the two iris centres (mesh indices 468 and 473; 468-477 are the iris points) as offsets from the eye
  centre, normalised by the eye-corner span and expressed along/across the eye line (so head roll and face
  size cancel out), plus eyelid opening (looking down closes the lids a little);
* head yaw / pitch / roll (:mod:`away.face.headpose`);
* face position and size in the frame (where the head is, how far);
* optionally the yaw / pitch predicted by a learned gaze network (:mod:`away.face.gaze_net`).

The regression (:func:`fit`) is ridge regression on standardised features with quadratic terms for the eye
and head-angle features. Its leave-one-target-out error is stored with the calibration, so the app can say
honestly how well it works: expect roughly 2-4 degrees of visual angle with a webcam, i.e. "which monitor
and roughly where", not "which button".

Runtime helpers: :class:`GazeEstimator` (predict + smooth + drift bias), :class:`LookToJump` (MAGIC pointing
trigger), drift correction (:meth:`GazeEstimator.correct`), and :class:`CalibrationSession` (the state
machine behind the calibration screen).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from away.hands.filters import OneEuro2D

IRIS_A, IRIS_B = 468, 473  # iris centres (468-472 one eye, 473-477 the other)
EYE_A = (33, 133, 159, 145)  # outer corner, inner corner, upper lid, lower lid
EYE_B = (263, 362, 386, 374)
N_LANDMARKS_WITH_IRIS = 478

# feature vector layout (see gaze_features)
EYE_FEATURES = ("iris_a_h", "iris_a_v", "iris_b_h", "iris_b_v", "lid_open")
HEAD_FEATURES = ("yaw", "pitch", "roll")
FACE_FEATURES = ("face_x", "face_y", "face_size")
NET_FEATURES = ("net_yaw", "net_pitch")
BASE_FEATURES = EYE_FEATURES + HEAD_FEATURES + FACE_FEATURES
FILE_NAME = "gaze-calibration.json"
FILE_VERSION = 1


def gaze_features(
    xs: Any,
    ys: Any,
    pose: tuple[float, float, float] | None,
    box: tuple[float, float, float] | None = None,
    net: tuple[float, float] | None = None,
) -> np.ndarray | None:
    """The feature vector for one face, or None when the mesh has no iris points or is degenerate.

    ``xs``/``ys``: normalised landmark coordinates (478 with iris); ``pose``: (yaw, pitch, roll) degrees;
    ``box``: (cx, cy, size) of the face; ``net``: learned-model (yaw, pitch) in radians."""
    try:
        if len(xs) < N_LANDMARKS_WITH_IRIS or len(ys) < N_LANDMARKS_WITH_IRIS:
            return None
        p = np.stack([np.asarray(xs[:N_LANDMARKS_WITH_IRIS], float), np.asarray(ys[:N_LANDMARKS_WITH_IRIS], float)], 1)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(p)):
        return None
    axis = p[EYE_B[0]] - p[EYE_A[0]]  # across the face, outer corner to outer corner
    span_face = float(np.hypot(*axis))
    if span_face < 1e-6:
        return None
    e = axis / span_face
    n = np.array([-e[1], e[0]])  # perpendicular, pointing down the face
    feats: list[float] = []
    opens = []
    for iris, (outer, inner, up, low) in ((IRIS_A, EYE_A), (IRIS_B, EYE_B)):
        span = float(np.hypot(*(p[inner] - p[outer])))
        if span < 1e-6:
            return None
        c = (p[inner] + p[outer]) / 2.0
        d = p[iris] - c
        feats += [float(d @ e) / span, float(d @ n) / span]
        opens.append(float((p[low] - p[up]) @ n) / span)
    feats.append(sum(opens) / 2.0)
    yaw, pitch, roll = pose if pose is not None else (0.0, 0.0, 0.0)
    feats += [yaw / 30.0, pitch / 30.0, roll / 30.0]
    if box is None:
        box = (float(p[:, 0].mean()), float(p[:, 1].mean()), float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))))
    feats += [float(box[0]), float(box[1]), float(box[2])]
    if net is not None:
        feats += [float(net[0]), float(net[1])]
    return np.asarray(feats, dtype=float)


def quad_indices(n_features: int) -> list[int]:
    """Features that get quadratic terms: the mean iris offsets are formed inside :func:`expand`; here the
    head yaw/pitch and, when present, the network yaw/pitch."""
    idx = [BASE_FEATURES.index("yaw"), BASE_FEATURES.index("pitch")]
    if n_features >= len(BASE_FEATURES) + 2:
        idx += [len(BASE_FEATURES), len(BASE_FEATURES) + 1]
    return idx


def expand(z: np.ndarray, quad: Sequence[int]) -> np.ndarray:
    """Design matrix from standardised features ``z`` (n x d): [1, z, mean-iris h/v, quadratic terms]."""
    z = np.atleast_2d(z)
    h = (z[:, 0] + z[:, 2]) / 2.0
    v = (z[:, 1] + z[:, 3]) / 2.0
    cols = [np.ones(len(z)), *z.T, h * h, v * v, h * v]
    q = [h, v, *(z[:, i] for i in quad)]
    for i in range(len(q)):
        for j in range(i, len(q)):
            if i < 2 and j < 2:
                continue  # h*h, v*v, h*v already added
            cols.append(q[i] * q[j])
    return np.stack(cols, 1)


@dataclass
class GazeFit:
    """A fitted feature -> desktop mapping (ridge regression)."""

    mean: list[float]
    std: list[float]
    weights: list[list[float]]  # (terms x 2), predicting desktop x, y normalised to the bounds
    bounds: tuple[float, float, float, float]  # x0, y0, width, height of the virtual desktop it maps onto
    quad: list[int] = field(default_factory=list)
    rms_px: float = 0.0  # training error
    loo_px: float | None = None  # leave-one-target-out error (honest estimate)

    @property
    def n_features(self) -> int:
        return len(self.mean)

    def predict(self, feats: np.ndarray) -> tuple[float, float]:
        z = (np.asarray(feats, float) - np.asarray(self.mean)) / np.asarray(self.std)
        uv = expand(z[None, :], self.quad) @ np.asarray(self.weights)
        x0, y0, w, h = self.bounds
        return float(x0 + uv[0, 0] * w), float(y0 + uv[0, 1] * h)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "bounds": list(self.bounds),
            "quad": self.quad,
            "rms_px": self.rms_px,
            "loo_px": self.loo_px,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GazeFit:
        b = d["bounds"]
        return cls(
            [float(v) for v in d["mean"]],
            [float(v) for v in d["std"]],
            [[float(a), float(b_)] for a, b_ in d["weights"]],
            (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            [int(i) for i in d.get("quad", [])],
            float(d.get("rms_px", 0.0)),
            None if d.get("loo_px") is None else float(d["loo_px"]),
        )


def _solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    reg = alpha * np.eye(x.shape[1])
    reg[0, 0] = 0.0  # never penalise the intercept
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def fit(
    feats: np.ndarray,
    targets: np.ndarray,
    bounds: tuple[float, float, float, float],
    alpha: float = 1.0,
    groups: Sequence[int] | None = None,
) -> GazeFit:
    """Ridge regression from ``feats`` (n x d) to desktop pixels ``targets`` (n x 2).

    ``groups`` (the calibration dot each sample belongs to) enables the leave-one-target-out error."""
    f = np.asarray(feats, float)
    t = np.asarray(targets, float)
    if f.ndim != 2 or t.shape != (len(f), 2) or len(f) < 3:
        raise ValueError("need at least 3 samples with matching targets")
    x0, y0, w, h = bounds
    if w <= 0 or h <= 0:
        raise ValueError("bounds must have a positive size")
    uv = np.stack([(t[:, 0] - x0) / w, (t[:, 1] - y0) / h], 1)
    mean = f.mean(0)
    std = f.std(0)
    std[std < 1e-6] = 1.0
    quad = quad_indices(f.shape[1])

    def train(rows: np.ndarray) -> np.ndarray:
        return _solve(expand((f[rows] - mean) / std, quad), uv[rows], alpha)

    all_rows = np.arange(len(f))
    wts = train(all_rows)
    pred = expand((f - mean) / std, quad) @ wts
    scale = np.array([w, h])
    rms = float(np.sqrt(np.mean(np.sum(((pred - uv) * scale) ** 2, 1))))
    loo: float | None = None
    if groups is not None:
        g = np.asarray(groups)
        errs = []
        for k in np.unique(g):
            test = all_rows[g == k]
            tr = all_rows[g != k]
            if len(tr) < 3:
                continue
            pk = expand((f[test] - mean) / std, quad) @ train(tr)
            errs.append(np.sum(((pk - uv[test]) * scale) ** 2, 1))
        if errs:
            loo = float(np.sqrt(np.mean(np.concatenate(errs))))
    return GazeFit(
        mean.tolist(), std.tolist(), wts.tolist(), (float(x0), float(y0), float(w), float(h)), quad, rms, loo
    )


# --------------------------------------------------------------------------- calibration targets
def grid_fractions(n: int) -> list[tuple[float, float]]:
    """Dot positions as fractions of a monitor: 5 (corners + centre), 9 (3x3) or 16 (4x4)."""
    if n == 5:
        return [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    k = 4 if n == 16 else 3
    vals = [0.1 + 0.8 * i / (k - 1) for i in range(k)]
    return [(x, y) for y in vals for x in vals]


def calibration_targets(monitors: Sequence[Any], per_monitor: int = 9) -> list[tuple[int, float, float]]:
    """``(monitor index, x, y)`` desktop pixels of every calibration dot, monitor by monitor."""
    out = []
    for i, m in enumerate(monitors):
        for fx, fy in grid_fractions(per_monitor):
            out.append((i, m.left + fx * (m.width - 1), m.top + fy * (m.height - 1)))
    return out


def desktop_bounds(monitors: Sequence[Any]) -> tuple[float, float, float, float]:
    x0 = min(m.left for m in monitors)
    y0 = min(m.top for m in monitors)
    return (
        float(x0),
        float(y0),
        float(max(m.left + m.width for m in monitors) - x0),
        float(max(m.top + m.height for m in monitors) - y0),
    )


@dataclass
class Sample:
    target: int  # index into the targets
    monitor: int
    x: float
    y: float
    feats: np.ndarray
    pose: tuple[float, float, float] | None


class CalibrationSession:
    """Drives the calibration dots: each dot waits ``settle_s`` (the eyes jump and settle), then collects
    face frames for ``collect_s`` (at least ``min_frames``). Frames without a usable face are skipped; a dot
    that gets no frames for ``timeout_s`` is skipped. Pure state machine; the Tk screen calls :meth:`feed`."""

    def __init__(
        self,
        targets: Sequence[tuple[int, float, float]],
        settle_s: float = 0.7,
        collect_s: float = 1.0,
        min_frames: int = 8,
        timeout_s: float = 6.0,
    ) -> None:
        self.targets = list(targets)
        self.settle_s, self.collect_s, self.min_frames, self.timeout_s = settle_s, collect_s, min_frames, timeout_s
        self.index = 0
        self.samples: list[Sample] = []
        self.skipped: list[int] = []
        self._t0: float | None = None
        self._n = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.targets)

    @property
    def current(self) -> tuple[int, float, float] | None:
        return None if self.done else self.targets[self.index]

    def start(self, t: float) -> None:
        self._t0, self._n = t, 0

    def progress(self, t: float) -> float:
        """0..1 within the current dot (for the shrinking-dot animation)."""
        if self._t0 is None:
            return 0.0
        return min(1.0, max(0.0, (t - self._t0) / (self.settle_s + self.collect_s)))

    def feed(self, t: float, feats: np.ndarray | None, pose: tuple[float, float, float] | None = None) -> bool:
        """One camera frame. Returns True when the session moved on to the next dot (or finished)."""
        if self.done:
            return False
        if self._t0 is None:
            self.start(t)
        assert self._t0 is not None
        el = t - self._t0
        if feats is not None and el >= self.settle_s:
            mon, x, y = self.targets[self.index]
            self.samples.append(Sample(self.index, mon, x, y, np.asarray(feats, float), pose))
            self._n += 1
        enough = el >= self.settle_s + self.collect_s and self._n >= self.min_frames
        if enough or el >= self.timeout_s:
            if not enough:
                self.skipped.append(self.index)
            self.index += 1
            self._t0, self._n = t, 0
            return True
        return False

    def result(
        self, monitors: Sequence[Any], alpha: float = 1.0, camera: str = "", net_model: str | None = None
    ) -> Calibration:
        """Fit the screen map from what was collected (raises ValueError when too few dots worked)."""
        return calibration_from_samples(self.samples, monitors, alpha, camera, net_model)


def calibration_from_samples(
    samples: Sequence[Sample],
    monitors: Sequence[Any],
    alpha: float = 1.0,
    camera: str = "",
    net_model: str | None = None,
) -> Calibration:
    dots = {s.target for s in samples}
    if len(dots) < 4:
        raise ValueError(f"only {len(dots)} calibration dots got face frames; need at least 4 (is the face visible?)")
    bounds = desktop_bounds(monitors)
    f = np.stack([s.feats for s in samples])
    t = np.array([[s.x, s.y] for s in samples])
    groups = [s.target for s in samples]
    fits: dict[str, GazeFit] = {}
    n_base = len(BASE_FEATURES)
    fits["iris"] = fit(f[:, :n_base], t, bounds, alpha, groups)
    if net_model is not None and f.shape[1] >= n_base + 2:
        fits["net"] = fit(f[:, : n_base + 2], t, bounds, alpha, groups)
    poses = [s.pose for s in samples if s.pose is not None]
    mon_poses: list[tuple[float, float] | None] = []
    for i in range(len(monitors)):
        ps = [s.pose for s in samples if s.monitor == i and s.pose is not None]
        mon_poses.append((float(np.median([p[0] for p in ps])), float(np.median([p[1] for p in ps]))) if ps else None)
    box = None
    if poses:
        ys = [p[0] for p in poses]
        pt = [p[1] for p in poses]
        box = (float(min(ys)), float(max(ys)), float(min(pt)), float(max(pt)))
    return Calibration(
        camera=camera,
        created=time.time(),
        monitors=[(int(m.left), int(m.top), int(m.width), int(m.height)) for m in monitors],
        fits=fits,
        net_model=net_model if "net" in fits else None,
        monitor_poses=mon_poses,
        pose_box=box,
        n_samples=len(samples),
        n_dots=len(dots),
    )


@dataclass
class Calibration:
    camera: str
    created: float
    monitors: list[tuple[int, int, int, int]]
    fits: dict[str, GazeFit]
    net_model: str | None = None  # learned gaze model whose angles the "net" fit uses
    monitor_poses: list[tuple[float, float] | None] = field(default_factory=list)
    pose_box: tuple[float, float, float, float] | None = None  # yaw min/max, pitch min/max while calibrating
    n_samples: int = 0
    n_dots: int = 0
    bias: tuple[float, float] = (0.0, 0.0)  # drift correction (pixels), saved with the calibration

    def best_fit(self, net_available: bool) -> tuple[str, GazeFit]:
        if net_available and "net" in self.fits:
            return "net", self.fits["net"]
        return "iris", self.fits["iris"]

    def accuracy_px(self) -> float | None:
        f = self.fits.get("net") or self.fits.get("iris")
        return None if f is None else (f.loo_px if f.loo_px is not None else f.rms_px)

    def matches_layout(self, monitors: Sequence[Any]) -> bool:
        """Calibrated for this exact monitor layout? (A changed layout makes the screen map wrong.)"""
        now = sorted((int(m.left), int(m.top), int(m.width), int(m.height)) for m in monitors)
        return sorted(tuple(r) for r in self.monitors) == now

    def monitor_poses_for(self, monitors: Sequence[Any]) -> list[tuple[float, float]] | None:
        """Per-monitor head poses in the order of ``monitors`` (None unless every monitor has one)."""
        rects = [tuple(r) for r in self.monitors]
        out = []
        for m in monitors:
            key = (int(m.left), int(m.top), int(m.width), int(m.height))
            if key not in rects:
                return None
            p = self.monitor_poses[rects.index(key)] if rects.index(key) < len(self.monitor_poses) else None
            if p is None:
                return None
            out.append(p)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "created": self.created,
            "monitors": [list(r) for r in self.monitors],
            "fits": {k: v.to_dict() for k, v in self.fits.items()},
            "net_model": self.net_model,
            "monitor_poses": [list(p) if p is not None else None for p in self.monitor_poses],
            "pose_box": list(self.pose_box) if self.pose_box is not None else None,
            "n_samples": self.n_samples,
            "n_dots": self.n_dots,
            "bias": list(self.bias),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        box = d.get("pose_box")
        bias = d.get("bias") or [0.0, 0.0]
        return cls(
            camera=str(d.get("camera", "")),
            created=float(d.get("created", 0.0)),
            monitors=[(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in d.get("monitors", [])],
            fits={k: GazeFit.from_dict(v) for k, v in d["fits"].items()},
            net_model=d.get("net_model"),
            monitor_poses=[(float(p[0]), float(p[1])) if p else None for p in d.get("monitor_poses", [])],
            pose_box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])) if box else None,
            n_samples=int(d.get("n_samples", 0)),
            n_dots=int(d.get("n_dots", 0)),
            bias=(float(bias[0]), float(bias[1])),
        )


# --------------------------------------------------------------------------- storage (per camera name)
def calibration_file(config_dir: Path | None = None) -> Path:
    if config_dir is None:
        from away import paths

        config_dir = paths.config_dir()
    return config_dir / FILE_NAME


def _read_all(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("cameras"), dict) else {}


def load_calibration(camera: str, path: Path | None = None) -> Calibration | None:
    data = _read_all(path or calibration_file())
    entry = data.get("cameras", {}).get(str(camera))
    if not isinstance(entry, dict):
        return None
    try:
        return Calibration.from_dict(entry)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None


def save_calibration(cal: Calibration, path: Path | None = None) -> Path:
    p = path or calibration_file()
    data = _read_all(p) or {"version": FILE_VERSION, "cameras": {}}
    data["version"] = FILE_VERSION
    data["cameras"][str(cal.camera)] = cal.to_dict()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(p)
    return p


def clear_calibration(camera: str | None = None, path: Path | None = None) -> bool:
    """Remove one camera's calibration (or all of them with ``camera=None``). True when something went."""
    p = path or calibration_file()
    if camera is None:
        existed = p.exists()
        p.unlink(missing_ok=True)
        return existed
    data = _read_all(p)
    if str(camera) not in data.get("cameras", {}):
        return False
    del data["cameras"][str(camera)]
    if data["cameras"]:
        p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    else:
        p.unlink(missing_ok=True)
    return True


def list_calibrations(path: Path | None = None) -> dict[str, Calibration]:
    out = {}
    for cam in _read_all(path or calibration_file()).get("cameras", {}):
        c = load_calibration(cam, path)
        if c is not None:
            out[cam] = c
    return out


# --------------------------------------------------------------------------- runtime
class GazeEstimator:
    """Calibration + smoothing + drift bias: features in, a desktop point out."""

    def __init__(self, cal: Calibration, min_cutoff: float = 0.4, beta: float = 0.004) -> None:
        self.cal = cal
        self.filter = OneEuro2D(min_cutoff, beta, 1.0)
        self.bias = cal.bias
        self.point: tuple[float, float] | None = None
        self.raw: tuple[float, float] | None = None
        self.t: float | None = None
        self.kind = "iris"

    def configure(self, min_cutoff: float, beta: float) -> None:
        for f in (self.filter.fx, self.filter.fy):
            f.min_cutoff, f.beta = min_cutoff, beta

    def update(self, t: float, feats: np.ndarray, net_available: bool = False) -> tuple[float, float] | None:
        kind, f = self.cal.best_fit(net_available and len(feats) >= len(BASE_FEATURES) + 2)
        if kind != self.kind:
            self.filter.reset()
            self.kind = kind
        x = np.asarray(feats, float)[: f.n_features]
        if len(x) < f.n_features or not np.all(np.isfinite(x)):
            return self.point
        px, py = f.predict(x)
        px, py = px + self.bias[0], py + self.bias[1]
        x0, y0, w, h = f.bounds
        px, py = min(max(px, x0), x0 + w - 1), min(max(py, y0), y0 + h - 1)
        self.raw = (px, py)
        self.point = self.filter(t, px, py)
        self.t = t
        return self.point

    def fresh(self, t: float, max_age: float = 0.5) -> tuple[float, float] | None:
        if self.point is None or self.t is None or t - self.t > max_age:
            return None
        return self.point

    def correct(self, click: tuple[float, float], rate: float, max_px: float, max_err: float) -> bool:
        """Drift correction: nudge the bias toward a click (the user looks where they click). Ignored when the
        click is further than ``max_err`` from the raw prediction; the bias never exceeds ``max_px``."""
        if self.raw is None or rate <= 0.0:
            return False
        ex, ey = click[0] - self.raw[0], click[1] - self.raw[1]
        if math.hypot(ex, ey) > max_err:
            return False
        bx, by = self.bias[0] + rate * ex, self.bias[1] + rate * ey
        mag = math.hypot(bx, by)
        if mag > max_px > 0:
            bx, by = bx * max_px / mag, by * max_px / mag
        elif max_px <= 0:
            bx, by = 0.0, 0.0
        self.bias = (bx, by)
        self.cal.bias = self.bias
        return True


class LookToJump:
    """MAGIC pointing trigger: when the hand starts moving after resting ``idle_s`` seconds, warp the cursor to
    the gaze point if it is at least ``min_px`` away (or only when it is on another monitor). One jump per
    movement onset; the hand then refines from there."""

    def __init__(self, idle_s: float = 0.6, min_px: float = 250.0, motion: float = 0.006) -> None:
        self.idle_s, self.min_px, self.motion = idle_s, min_px, motion
        self._last: tuple[float, float] | None = None
        self._still_since: float | None = None

    def reset(self) -> None:
        """Hand lost: the next appearance counts as resting from then on."""
        self._last = None
        self._still_since = None

    def update(
        self,
        t: float,
        hand: tuple[float, float],
        cursor: tuple[float, float],
        gaze: tuple[float, float] | None,
        monitor_of: Any = None,
        across_only: bool = False,
    ) -> tuple[float, float] | None:
        """One pointing frame (normalised hand position). Returns the warp target or None."""
        last, self._last = self._last, hand
        if last is None:
            self._still_since = t
            return None
        moved = math.hypot(hand[0] - last[0], hand[1] - last[1]) >= self.motion
        if not moved:
            if self._still_since is None:
                self._still_since = t
            return None
        rested = self._still_since is not None and t - self._still_since >= self.idle_s
        self._still_since = None
        if not rested or gaze is None:
            return None
        if math.hypot(gaze[0] - cursor[0], gaze[1] - cursor[1]) < self.min_px:
            return None
        if across_only and monitor_of is not None and monitor_of(*gaze) == monitor_of(*cursor):
            return None
        return gaze
