"""Fuse hand observations from several cameras into one hand for the gesture pipeline.

Per frame set (see :class:`~away.multicam.capture.FramePairer`):

1. Each view gets a confidence from the MediaPipe handedness score, how many
   landmarks fall inside the image, the hand's size in pixels and a sharpness
   (motion-blur) score.
2. When the cameras are calibrated and the primary plus a calibrated secondary
   both see the hand, the 21 landmarks are triangulated (``cv2.triangulatePoints``).
   That gives true depth (used by the ``push`` gesture), a 3D pinch distance that
   does not collapse when the hand is seen edge-on, and 2D landmarks for the
   cursor by reprojecting into the primary view.
3. Otherwise each secondary view is mapped into the primary image with a running
   translation + scale alignment learned while both cameras saw the hand, and the
   views are either confidence-weighted (``fusion: weighted``) or the best one is
   used (``fusion: best_view``).
4. When the set of contributing views changes (occlusion handover), the jump between
   the old and new estimate is absorbed by an offset that fades out over
   ``handover_decay_s``, so the cursor's One Euro filter sees a continuous signal and
   nothing re-anchors.

Output is always in the primary camera's normalised (mirrored) image coordinates,
the same frame single-camera mode uses, so the gesture state machine and cursor
code are unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.config import CameraSpec, MultiCamConfig
from away.hands.landmarks import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST
from away.multicam.geometry import (
    CameraCalib,
    Extrinsics,
    from_pixels,
    intrinsics_from_fov,
    monocular_depth,
    project,
    to_pixels,
    triangulate,
)

Array = NDArray[np.float64]


@dataclass
class ViewObs:
    """One camera's hand at one moment (landmarks normalised to that camera's possibly mirrored frame)."""

    cam: int
    t: float
    landmarks: Array  # (21, 3)
    handedness: str = "Right"
    score: float = 1.0  # MediaPipe handedness confidence
    size: tuple[int, int] = (640, 480)
    sharpness: float | None = None  # variance of the Laplacian over the hand crop (None = unknown)


@dataclass
class FusedHand:
    t: float
    landmarks: Array  # (21, 3) in the primary view's normalised coordinates
    handedness: str
    source: str  # "3d", "cam0", "cam0+cam1", ...
    confidences: dict[int, float] = field(default_factory=dict)
    points3d: Array | None = None  # (21, 3) metres in the primary camera frame
    pinch: float | None = None  # 3D thumb-index / palm ratio when triangulated
    depth: float | None = None  # hand distance from the primary camera (m)


def visibility(lm: Any, margin: float = 0.0) -> float:
    """Fraction of landmarks inside the image (MediaPipe extrapolates off-frame points)."""
    a = np.asarray(lm, dtype=np.float64)[:, :2]
    inside = (a >= -margin) & (a <= 1.0 + margin)
    return float(np.mean(inside.all(axis=1)))


def palm_px(lm: Any, size: tuple[int, int]) -> float:
    a = np.asarray(lm, dtype=np.float64)
    return math.hypot((a[MIDDLE_MCP, 0] - a[WRIST, 0]) * size[0], (a[MIDDLE_MCP, 1] - a[WRIST, 1]) * size[1])


def sharpness_score(gray_crop: Any) -> float:
    """Variance of a 4-neighbour Laplacian (numpy only): low for motion-blurred crops."""
    g = np.asarray(gray_crop, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) < 3:
        return 0.0
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def hand_crop(gray: Any, lm: Any, pad: float = 0.15) -> Any:
    g = np.asarray(gray)
    h, w = g.shape[:2]
    a = np.asarray(lm, dtype=np.float64)
    x0, y0 = a[:, 0].min(), a[:, 1].min()
    x1, y1 = a[:, 0].max(), a[:, 1].max()
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    c0, c1 = int(max(0, (x0 - px) * w)), int(min(w, (x1 + px) * w))
    r0, r1 = int(max(0, (y0 - py) * h)), int(min(h, (y1 + py) * h))
    return g[r0:r1, c0:c1]


def view_confidence(
    obs: ViewObs, min_palm_px: float = 12.0, good_palm_px: float = 60.0, sharp_ref: float = 80.0
) -> float:
    """0..1 trust in one view."""
    s = float(np.clip(obs.score, 0.0, 1.0))
    v = visibility(obs.landmarks)
    size = float(np.clip((palm_px(obs.landmarks, obs.size) - min_palm_px) / (good_palm_px - min_palm_px), 0.0, 1.0))
    b = 1.0 if obs.sharpness is None else float(np.clip(obs.sharpness / sharp_ref, 0.0, 1.0))
    return s * v * (0.25 + 0.75 * size) * (0.5 + 0.5 * b)


def pinch_ratio_3d(points: Array) -> float:
    palm = float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST]))
    if palm <= 1e-9:
        return float("inf")
    return float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP]) / palm)


class PushDetector:
    """Hand moves toward the screen (depth from the primary camera, mounted at the screen, drops) by
    ``distance_m`` within ``time_s``. Fires once, then waits ``cooldown_s``."""

    def __init__(self, distance_m: float = 0.06, time_s: float = 0.35, cooldown_s: float = 0.6) -> None:
        self.distance_m = distance_m
        self.time_s = time_s
        self.cooldown_s = cooldown_s
        self._hist: deque[tuple[float, float]] = deque()
        self._last = -1e9

    def reset(self) -> None:
        self._hist.clear()

    def update(self, t: float, depth: float | None) -> bool:
        if depth is None or not math.isfinite(depth):
            self.reset()
            return False
        self._hist.append((t, depth))
        while self._hist and t - self._hist[0][0] > self.time_s:
            self._hist.popleft()
        if t - self._last < self.cooldown_s:
            return False
        far = max(d for _, d in self._hist)
        if far - depth >= self.distance_m:
            self._last = t
            self._hist.clear()
            return True
        return False


class _Align:
    """Running map from a secondary view's normalised coords into the primary's: one similarity for every point
    (per-axis scale, then shift), so the wrist moves at the primary camera's speed too once only the secondary
    sees the hand. Per axis because the two cameras' frames may have different aspect ratios."""

    def __init__(self, rate: float = 0.2) -> None:
        self.rate = rate
        self.offset: Array | None = None
        self.scale: Array = np.ones(2)

    def learn(self, prim: Array, sec: Array, size_p: tuple[int, int], size_s: tuple[int, int]) -> None:
        sp, ss = palm_px(prim, size_p), palm_px(sec, size_s)
        if ss > 1e-6:
            scale = np.array([(sp / size_p[0]) / (ss / size_s[0]), (sp / size_p[1]) / (ss / size_s[1])])
        else:
            scale = np.ones(2)
        if self.offset is None:
            self.scale = scale
        else:
            self.scale = (1 - self.rate) * self.scale + self.rate * scale
        off = prim[WRIST, :2] - self.scale * sec[WRIST, :2]
        self.offset = off if self.offset is None else (1 - self.rate) * self.offset + self.rate * off

    def apply(self, sec: Array) -> Array:
        out = sec.copy()
        out[:, :2] = sec[:, :2] * self.scale + (self.offset if self.offset is not None else 0.0)
        return out


class MultiViewFuser:
    def __init__(
        self, cameras: list[CameraSpec], cfg: MultiCamConfig | None = None, extrinsics: Extrinsics | None = None
    ) -> None:
        if not cameras:
            raise ValueError("MultiViewFuser needs at least one camera")
        self.cameras = cameras
        self.cfg = cfg or MultiCamConfig()
        self.primary = next((i for i, c in enumerate(cameras) if c.role == "primary"), 0)
        self.calib: dict[int, CameraCalib] = {}
        self._align: dict[int, _Align] = {i: _Align() for i in range(len(cameras))}
        self._source: str | None = None
        self._handover: Array | None = None
        self._handover_t = 0.0
        self._last_out: Array | None = None
        self.last_confidences: dict[int, float] = {}
        self.set_extrinsics(extrinsics)

    @property
    def calibrated(self) -> bool:
        return len(self.calib) >= 2

    def set_extrinsics(self, ex: Extrinsics | None) -> None:
        """Attach a saved calibration (matched to the configured cameras by device)."""
        self.calib = {}
        if ex is None:
            return
        p = self.cameras[self.primary]
        if str(ex.primary.device) != str(p.device):
            return
        self.calib[self.primary] = ex.primary
        for i, cam in enumerate(self.cameras):
            if i != self.primary:
                c = ex.for_device(str(cam.device))
                if c is not None:
                    self.calib[i] = c

    def _K(self, i: int, size: tuple[int, int]) -> Array:
        c = self.calib.get(i)
        if c is not None:
            K = np.asarray(c.K, dtype=np.float64).copy()
            if (c.width, c.height) != tuple(size):  # resolution changed since calibration: rescale
                K[0] *= size[0] / c.width
                K[1] *= size[1] / c.height
            return K
        return intrinsics_from_fov(size[0], size[1], self.cameras[i].hfov_deg)

    # --------------------------------------------------------------------------- step
    def update(self, t: float, views: dict[int, ViewObs | None] | list[ViewObs | None]) -> FusedHand | None:
        items = views.items() if isinstance(views, dict) else enumerate(views)
        obs = {i: o for i, o in items if o is not None}
        confs = {i: view_confidence(o) for i, o in obs.items()}
        self.last_confidences = confs
        good = {i: o for i, o in obs.items() if confs[i] >= self.cfg.min_confidence}
        if not good:
            self._source = None
            self._handover = None
            self._last_out = None
            return None
        p = self.primary
        # keep the secondary->primary alignment fresh whenever both see the hand
        if p in good:
            for i, o in good.items():
                if i != p:
                    self._align[i].learn(good[p].landmarks, o.landmarks, good[p].size, o.size)

        points3d = None
        pair = self._best_calibrated_pair(good, confs)
        if pair is not None:
            j = pair
            P, S = good[p], good[j]
            cp, cs = self.calib[p], self.calib[j]
            pts = triangulate(
                to_pixels(P.landmarks, *P.size, mirrored=self.cameras[p].mirror),
                to_pixels(S.landmarks, *S.size, mirrored=self.cameras[j].mirror),
                self._K(p, P.size),
                self._K(j, S.size),
                cs.R,
                cs.t,
                cp.dist,
                cs.dist,
            )
            if np.all(np.isfinite(pts)) and np.all(pts[:, 2] > 0):
                points3d = pts
        if points3d is not None:
            P = good[p]
            uv = from_pixels(project(points3d, self._K(p, P.size)), *P.size, mirrored=self.cameras[p].mirror)
            lm = np.column_stack([uv, P.landmarks[:, 2]])
            source = "3d"
            hand = P.handedness
        else:
            lm, source, hand = self._fuse_2d(good, confs)
        lm = self._continuity(t, source, lm)
        pinch = pinch_ratio_3d(points3d) if points3d is not None and self.cfg.use_3d_pinch else None
        depth = None
        if points3d is not None:
            depth = float(np.mean(points3d[[WRIST, 5, MIDDLE_MCP, 13, 17], 2]))
        elif self.cfg.push_monocular and p in good:
            P = good[p]
            depth = monocular_depth(P.landmarks, P.size[0], P.size[1], self.cameras[p].hfov_deg, self.cfg.palm_length_m)
        return FusedHand(t, lm, hand, source, confs, points3d, pinch, depth)

    def _best_calibrated_pair(self, good: dict[int, ViewObs], confs: dict[int, float]) -> int | None:
        p = self.primary
        if p not in good or p not in self.calib:
            return None
        cands = [i for i in good if i != p and i in self.calib]
        return max(cands, key=lambda i: confs[i]) if cands else None

    def _fuse_2d(self, good: dict[int, ViewObs], confs: dict[int, float]) -> tuple[Array, str, str]:
        p = self.primary
        mapped: dict[int, Array] = {}
        for i, o in good.items():
            if i == p:
                mapped[i] = np.asarray(o.landmarks, dtype=np.float64).copy()
            elif self._align[i].offset is not None:
                mapped[i] = self._align[i].apply(np.asarray(o.landmarks, dtype=np.float64))
            elif p not in good:
                mapped[i] = np.asarray(o.landmarks, dtype=np.float64).copy()  # never seen together: raw view
        best = max(mapped, key=lambda i: confs[i])
        hand = good[best].handedness
        if self.cfg.fusion == "best_view" or len(mapped) == 1:
            return mapped[best], f"cam{best}", hand
        w = np.array([confs[i] for i in mapped])
        stack = np.stack([mapped[i] for i in mapped])
        lm = np.tensordot(w / w.sum(), stack, axes=1)
        return lm, "+".join(f"cam{i}" for i in sorted(mapped)), hand

    def _continuity(self, t: float, source: str, lm: Array) -> Array:
        if self._source is not None and source != self._source and self._last_out is not None:
            # _last_out already includes any offset still fading from an earlier switch
            self._handover = self._last_out[WRIST, :2] - lm[WRIST, :2]
            self._handover_t = t
        self._source = source
        off = self._current_offset(t)
        out = lm.copy()
        out[:, :2] += off
        self._last_out = out
        return out

    def _current_offset(self, t: float) -> Array:
        if self._handover is None:
            return np.zeros(2)
        k = math.exp(-max(0.0, t - self._handover_t) / max(self.cfg.handover_decay_s, 1e-6))
        return self._handover * k
