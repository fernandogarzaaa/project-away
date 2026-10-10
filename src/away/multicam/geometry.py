"""Two-view geometry for hand landmarks (OpenCV).

Conventions: the *primary* camera frame is the world frame. A camera with
rotation ``R`` and translation ``t`` maps a world point ``X`` to ``R @ X + t``
in its own frame; pixels are ``K @ X_cam / z``. Landmarks from MediaPipe are
normalised image coordinates of the (possibly mirrored) frame; :func:`to_pixels`
undoes the mirror so the geometry sees the real optics.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.hands.landmarks import MIDDLE_MCP, WRIST

Array = NDArray[np.float64]
EXTRINSICS_FILE = "camera_extrinsics.json"


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> Array:
    """Pinhole K with square pixels and the principal point at the centre."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def to_pixels(lm: Any, width: int, height: int, mirrored: bool = False) -> Array:
    """(21, 2|3) normalised landmarks -> (21, 2) pixel coordinates of the un-mirrored sensor image."""
    a = np.asarray(lm, dtype=np.float64)[:, :2].copy()
    if mirrored:
        a[:, 0] = 1.0 - a[:, 0]
    a[:, 0] *= width
    a[:, 1] *= height
    return a


def from_pixels(px: Any, width: int, height: int, mirrored: bool = False) -> Array:
    a = np.asarray(px, dtype=np.float64)[:, :2].copy()
    a[:, 0] /= width
    a[:, 1] /= height
    if mirrored:
        a[:, 0] = 1.0 - a[:, 0]
    return a


def project(points: Any, K: Array, R: Array | None = None, t: Array | None = None) -> Array:
    """(N, 3) world points -> (N, 2) pixels."""
    X = np.asarray(points, dtype=np.float64)
    if R is not None:
        X = X @ np.asarray(R).T + (np.zeros(3) if t is None else np.asarray(t).reshape(3))
    uvw = X @ np.asarray(K, dtype=np.float64).T
    out: Array = np.asarray(uvw[:, :2] / uvw[:, 2:3], dtype=np.float64)
    return out


def normalize_points(px: Any, K: Array, dist: Any = None) -> Array:
    """Pixels -> normalised camera coordinates (x/z, y/z), undistorting when ``dist`` is non-zero."""
    p = np.asarray(px, dtype=np.float64)
    if dist is not None and np.any(np.asarray(dist) != 0):
        import cv2

        und = cv2.undistortPoints(p.reshape(-1, 1, 2), K, np.asarray(dist, np.float64))
        return np.asarray(und, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([(p[:, 0] - K[0, 2]) / K[0, 0], (p[:, 1] - K[1, 2]) / K[1, 1]])


def triangulate(
    px1: Any, px2: Any, K1: Array, K2: Array, R: Array, t: Array, dist1: Any = None, dist2: Any = None
) -> Array:
    """Pixels in the primary (1) and secondary (2) view -> (N, 3) points in the primary frame."""
    import cv2

    n1 = normalize_points(px1, K1, dist1)
    n2 = normalize_points(px2, K2, dist2)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, n1.T.astype(np.float64), n2.T.astype(np.float64))
    out: Array = (Xh[:3] / Xh[3]).T
    return out


def reprojection_error(points: Array, px1: Any, px2: Any, K1: Array, K2: Array, R: Array, t: Array) -> float:
    """RMS pixel error of ``points`` reprojected into both views."""
    e1 = project(points, K1) - np.asarray(px1)
    e2 = project(points, K2, R, t) - np.asarray(px2)
    return float(np.sqrt(np.mean(np.concatenate([np.sum(e1**2, 1), np.sum(e2**2, 1)]))))


def rotation_angle_deg(R_a: Array, R_b: Array) -> float:
    """Angle of the rotation that takes ``R_a`` to ``R_b``."""
    c = (np.trace(np.asarray(R_a).T @ np.asarray(R_b)) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


@dataclass
class PoseEstimate:
    R: Array
    t: Array  # metric when a scale reference was available (palm length or board squares)
    reprojection_error_px: float
    inliers: int
    total: int
    scale_m: float  # factor applied to the unit-length translation


def estimate_pose_from_hands(
    px1: Any,
    px2: Any,
    K1: Array,
    K2: Array,
    palm_length_m: float | None = 0.09,
    frames: int | None = None,
    ransac_px: float = 3.0,
) -> PoseEstimate:
    """Relative pose of camera 2 from corresponding hand landmarks (N*21 points each).

    ``findFundamentalMat`` (RANSAC) -> ``E = K2^T F K1`` -> ``recoverPose`` on normalised inliers. The
    translation comes out unit-length; when ``frames`` (number of 21-point hands stacked in ``px1``) and
    ``palm_length_m`` are given it is rescaled so the median triangulated palm (wrist -> middle MCP) has
    that length, which gives depth in metres (approximately: real palms differ by ~10-15%).
    """
    import cv2

    p1 = np.asarray(px1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(px2, dtype=np.float64).reshape(-1, 2)
    if len(p1) < 8 or len(p1) != len(p2):
        raise ValueError(f"need >= 8 matching points in both views, got {len(p1)} and {len(p2)}")
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, ransac_px, 0.999)
    if F is None or F.shape != (3, 3):
        raise ValueError("fundamental matrix estimation failed (too little hand motion / depth variation?)")
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(p1), bool)
    if inl.sum() >= 8:  # refit on all inliers (least squares beats the RANSAC minimal sample under noise)
        F8, _ = cv2.findFundamentalMat(p1[inl], p2[inl], cv2.FM_8POINT)
        if F8 is not None and F8.shape == (3, 3):
            F = F8
    E = K2.T @ F @ K1
    n1 = normalize_points(p1[inl], K1)
    n2 = normalize_points(p2[inl], K2)
    _n, R_cv, t_cv, _m = cv2.recoverPose(E, n1, n2, np.eye(3))
    R: Array = np.asarray(R_cv, dtype=np.float64)
    t: Array = np.asarray(t_cv, dtype=np.float64).reshape(3)
    t = t / max(float(np.linalg.norm(t)), 1e-12)
    scale = 1.0
    if palm_length_m and frames:
        pts = triangulate(p1, p2, K1, K2, R, t).reshape(frames, -1, 3)
        palms = np.linalg.norm(pts[:, MIDDLE_MCP] - pts[:, WRIST], axis=1)
        palms = palms[np.isfinite(palms) & (palms > 0)]
        if len(palms):
            scale = float(palm_length_m / np.median(palms))
    t = t * scale
    X = triangulate(p1[inl], p2[inl], K1, K2, R, t)
    err = reprojection_error(X, p1[inl], p2[inl], K1, K2, R, t)
    return PoseEstimate(np.asarray(R, dtype=np.float64), t, err, int(inl.sum()), len(p1), scale)


def checkerboard_points(cols: int, rows: int, square_m: float) -> Array:
    """Inner-corner grid (cols x rows) on the z=0 plane, in metres."""
    g = np.zeros((rows * cols, 3), np.float64)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return g


def stereo_calibrate(
    obj_points: list[Array],
    img1: list[Array],
    img2: list[Array],
    size1: tuple[int, int],
    size2: tuple[int, int],
) -> tuple[Array, Array, Array, Array, Array, Array, float]:
    """Checkerboard stereo calibration: per-camera ``calibrateCamera`` then ``stereoCalibrate`` with fixed intrinsics.

    Returns ``(K1, d1, K2, d2, R, t, rms_px)``.
    """
    import cv2

    obj = [np.asarray(o, np.float32) for o in obj_points]
    i1 = [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in img1]
    i2 = [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in img2]
    _e1, K1, d1, _r1, _t1 = cv2.calibrateCamera(obj, i1, size1, None, None)
    _e2, K2, d2, _r2, _t2 = cv2.calibrateCamera(obj, i2, size2, None, None)
    rms, K1, d1, K2, d2, R, t, _E, _F = cv2.stereoCalibrate(
        obj, i1, i2, K1, d1, K2, d2, size1, flags=cv2.CALIB_FIX_INTRINSIC
    )
    f = np.float64
    return (
        np.asarray(K1, f),
        np.asarray(d1, f).ravel(),
        np.asarray(K2, f),
        np.asarray(d2, f).ravel(),
        np.asarray(R, f),
        np.asarray(t, f).reshape(3),
        float(rms),
    )


def make_checkerboard_image(cols: int, rows: int, square_px: int = 80, margin_px: int = 40) -> NDArray[np.uint8]:
    """A printable board with ``cols x rows`` *inner* corners (so (cols+1) x (rows+1) squares)."""
    w = (cols + 1) * square_px + 2 * margin_px
    h = (rows + 1) * square_px + 2 * margin_px
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = margin_px + r * square_px, margin_px + c * square_px
                img[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    return img


# ---------------------------------------------------------------------------- persistence
@dataclass
class CameraCalib:
    device: str
    width: int
    height: int
    K: Array
    dist: Array = field(default_factory=lambda: np.zeros(5))
    R: Array = field(default_factory=lambda: np.eye(3))
    t: Array = field(default_factory=lambda: np.zeros(3))
    mirrored: bool = True


@dataclass
class Extrinsics:
    primary: CameraCalib
    secondaries: list[CameraCalib]
    method: str  # hand | checkerboard
    reprojection_error_px: float
    metric: bool  # translation in metres (board, or palm-length scaled)
    created: float = field(default_factory=time.time)

    def for_device(self, device: str) -> CameraCalib | None:
        return next((c for c in self.secondaries if c.device == str(device)), None)

    def to_dict(self) -> dict[str, Any]:
        def cam(c: CameraCalib) -> dict[str, Any]:
            return {
                "device": c.device,
                "width": c.width,
                "height": c.height,
                "K": np.asarray(c.K).tolist(),
                "dist": np.asarray(c.dist).ravel().tolist(),
                "R": np.asarray(c.R).tolist(),
                "t": np.asarray(c.t).ravel().tolist(),
                "mirrored": c.mirrored,
            }

        return {
            "version": 1,
            "method": self.method,
            "reprojection_error_px": self.reprojection_error_px,
            "metric": self.metric,
            "created": self.created,
            "primary": cam(self.primary),
            "secondaries": [cam(c) for c in self.secondaries],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Extrinsics:
        def cam(c: dict[str, Any]) -> CameraCalib:
            return CameraCalib(
                str(c["device"]),
                int(c["width"]),
                int(c["height"]),
                np.array(c["K"], np.float64),
                np.array(c.get("dist", [0] * 5), np.float64),
                np.array(c.get("R", np.eye(3).tolist()), np.float64),
                np.array(c.get("t", [0, 0, 0]), np.float64),
                bool(c.get("mirrored", True)),
            )

        return cls(
            cam(d["primary"]),
            [cam(c) for c in d.get("secondaries", [])],
            str(d.get("method", "hand")),
            float(d.get("reprojection_error_px", float("nan"))),
            bool(d.get("metric", False)),
            float(d.get("created", 0.0)),
        )


def extrinsics_path() -> Path:
    from away import paths

    return paths.data_dir() / EXTRINSICS_FILE


def save_extrinsics(ex: Extrinsics, path: Path | None = None) -> Path:
    p = path or extrinsics_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ex.to_dict(), indent=2), encoding="utf-8")
    return p


def load_extrinsics(path: Path | None = None) -> Extrinsics | None:
    p = path or extrinsics_path()
    if not p.exists():
        return None
    try:
        return Extrinsics.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError):
        return None


def monocular_depth(lm: Any, width: int, height: int, hfov_deg: float, palm_length_m: float = 0.09) -> float | None:
    """Rough single-camera depth (m) from apparent palm size: z ~ f * palm / palm_px."""
    a = np.asarray(lm, dtype=np.float64)
    d = math.hypot((a[MIDDLE_MCP, 0] - a[WRIST, 0]) * width, (a[MIDDLE_MCP, 1] - a[WRIST, 1]) * height)
    if d < 1e-6:
        return None
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return float(f * palm_length_m / d)
