"""Collect correspondences and solve camera extrinsics (``away calibrate-cameras``).

* :class:`HandMotionCalibrator` (default, nothing to print): the user waves an open
  hand around for ~20 s; every frame where the primary and a secondary both see the
  hand adds 21 landmark correspondences. Intrinsics are approximated from each
  camera's ``hfov_deg``; the translation is scaled with an average palm length, so
  depth is approximately metric.
* :class:`CheckerboardCalibrator` (accurate): a printed checkerboard shown to both
  cameras; per-camera intrinsics + ``cv2.stereoCalibrate`` with the real square size.

Both produce an :class:`~away.multicam.geometry.Extrinsics` saved to the user data dir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from away.config import CameraSpec
from away.multicam.fusion import ViewObs, view_confidence
from away.multicam.geometry import (
    CameraCalib,
    Extrinsics,
    checkerboard_points,
    estimate_pose_from_hands,
    intrinsics_from_fov,
    stereo_calibrate,
    to_pixels,
)


@dataclass
class CalibReport:
    ok: bool
    per_camera: dict[str, dict[str, Any]] = field(default_factory=dict)
    message: str = ""


def _primary_index(cams: list[CameraSpec]) -> int:
    return next((i for i, c in enumerate(cams) if c.role == "primary"), 0)


class HandMotionCalibrator:
    def __init__(
        self, cams: list[CameraSpec], min_frames: int = 40, max_frames: int = 400, min_conf: float = 0.4
    ) -> None:
        if len(cams) < 2:
            raise ValueError("calibration needs at least two cameras in hands.cameras")
        self.cams = cams
        self.primary = _primary_index(cams)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.min_conf = min_conf
        self.samples: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            i: [] for i in range(len(cams)) if i != self.primary
        }
        self.sizes: dict[int, tuple[int, int]] = {}

    def add(self, views: list[ViewObs | None]) -> None:
        p = views[self.primary] if self.primary < len(views) else None
        if p is None or view_confidence(p) < self.min_conf:
            return
        self.sizes[self.primary] = p.size
        pp = to_pixels(p.landmarks, *p.size, mirrored=self.cams[self.primary].mirror)
        for j in self.samples:
            o = views[j] if j < len(views) else None
            if o is None or view_confidence(o) < self.min_conf or len(self.samples[j]) >= self.max_frames:
                continue
            self.sizes[j] = o.size
            self.samples[j].append((pp, to_pixels(o.landmarks, *o.size, mirrored=self.cams[j].mirror)))

    def progress(self) -> float:
        """0..1: the slowest secondary's share of ``min_frames``."""
        return min(min(1.0, len(s) / self.min_frames) for s in self.samples.values())

    def solve(self, palm_length_m: float = 0.09) -> tuple[Extrinsics | None, CalibReport]:
        p = self.primary
        size_p = self.sizes.get(p, (self.cams[p].width, self.cams[p].height))
        Kp = intrinsics_from_fov(size_p[0], size_p[1], self.cams[p].hfov_deg)
        rep = CalibReport(True)
        secs: list[CameraCalib] = []
        errs = []
        for j, samples in self.samples.items():
            name = str(self.cams[j].device)
            if len(samples) < self.min_frames:
                rep.ok = False
                rep.per_camera[name] = {"error": f"only {len(samples)} frames with the hand in both views"}
                continue
            size_j = self.sizes[j]
            Kj = intrinsics_from_fov(size_j[0], size_j[1], self.cams[j].hfov_deg)
            p1 = np.concatenate([a for a, _ in samples])
            p2 = np.concatenate([b for _, b in samples])
            try:
                est = estimate_pose_from_hands(p1, p2, Kp, Kj, palm_length_m, frames=len(samples))
            except ValueError as exc:
                rep.ok = False
                rep.per_camera[name] = {"error": str(exc)}
                continue
            secs.append(CameraCalib(name, size_j[0], size_j[1], Kj, np.zeros(5), est.R, est.t, self.cams[j].mirror))
            errs.append(est.reprojection_error_px)
            rep.per_camera[name] = {
                "frames": len(samples),
                "reprojection_error_px": round(est.reprojection_error_px, 2),
                "inliers": f"{est.inliers}/{est.total}",
                "baseline_m": round(float(np.linalg.norm(est.t)), 3),
            }
        if not secs:
            rep.ok = False
            rep.message = "no secondary camera could be calibrated; move your open hand through both views"
            return None, rep
        prim = CameraCalib(
            str(self.cams[p].device), size_p[0], size_p[1], Kp, np.zeros(5), mirrored=self.cams[p].mirror
        )
        ex = Extrinsics(prim, secs, "hand", float(np.mean(errs)), True)
        rep.message = (
            f"mean reprojection error {ex.reprojection_error_px:.2f} px (hand-motion calibration, approximate)"
        )
        return ex, rep


class CheckerboardCalibrator:
    """``cols x rows`` inner corners, ``square_m`` square size. Frames are RGB as delivered (mirrored or not)."""

    def __init__(self, cams: list[CameraSpec], cols: int = 9, rows: int = 6, square_m: float = 0.025) -> None:
        if len(cams) < 2:
            raise ValueError("calibration needs at least two cameras in hands.cameras")
        self.cams, self.cols, self.rows, self.square_m = cams, cols, rows, square_m
        self.primary = _primary_index(cams)
        self.views: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            i: [] for i in range(len(cams)) if i != self.primary
        }
        self.sizes: dict[int, tuple[int, int]] = {}

    def detect(self, rgb: Any, mirrored: bool) -> np.ndarray | None:
        import cv2

        img = np.asarray(rgb)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        if mirrored:
            gray = cv2.flip(gray, 1)  # geometry wants the real sensor image
        ok, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows), None)
        if not ok:
            return None
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
        return np.asarray(corners, np.float64).reshape(-1, 2)

    def add(self, frames: list[Any]) -> list[int]:
        """Detect the board in a synchronised frame set. Returns the secondaries that got a new view."""
        p = self.primary
        if frames[p] is None:
            return []
        cp = self.detect(frames[p], self.cams[p].mirror)
        if cp is None:
            return []
        self.sizes[p] = (np.asarray(frames[p]).shape[1], np.asarray(frames[p]).shape[0])
        added = []
        for j in self.views:
            if frames[j] is None:
                continue
            cj = self.detect(frames[j], self.cams[j].mirror)
            if cj is not None:
                self.sizes[j] = (np.asarray(frames[j]).shape[1], np.asarray(frames[j]).shape[0])
                self.views[j].append((cp, cj))
                added.append(j)
        return added

    def add_corners(
        self, j: int, corners_p: Any, corners_j: Any, size_p: tuple[int, int], size_j: tuple[int, int]
    ) -> None:
        self.sizes[self.primary], self.sizes[j] = size_p, size_j
        self.views[j].append((np.asarray(corners_p, np.float64), np.asarray(corners_j, np.float64)))

    def solve(self, min_views: int = 8) -> tuple[Extrinsics | None, CalibReport]:
        p = self.primary
        obj = checkerboard_points(self.cols, self.rows, self.square_m)
        rep = CalibReport(True)
        secs, errs = [], []
        prim: CameraCalib | None = None
        for j, views in self.views.items():
            name = str(self.cams[j].device)
            if len(views) < min_views:
                rep.ok = False
                rep.per_camera[name] = {
                    "error": f"board seen by both cameras in only {len(views)} views (need {min_views})"
                }
                continue
            K1, d1, K2, d2, R, t, rms = stereo_calibrate(
                [obj] * len(views), [a for a, _ in views], [b for _, b in views], self.sizes[p], self.sizes[j]
            )
            if prim is None:
                prim = CameraCalib(str(self.cams[p].device), *self.sizes[p], K1, d1, mirrored=self.cams[p].mirror)
            secs.append(CameraCalib(name, *self.sizes[j], K2, d2, R, t, self.cams[j].mirror))
            errs.append(rms)
            rep.per_camera[name] = {
                "views": len(views),
                "reprojection_error_px": round(rms, 3),
                "baseline_m": round(float(np.linalg.norm(t)), 3),
            }
        if prim is None or not secs:
            rep.ok = False
            rep.message = "not enough checkerboard views; show the board to both cameras at different angles"
            return None, rep
        ex = Extrinsics(prim, secs, "checkerboard", float(np.mean(errs)), True)
        rep.message = f"stereo RMS reprojection error {ex.reprojection_error_px:.3f} px (checkerboard)"
        return ex, rep
