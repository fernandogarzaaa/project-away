"""Synthetic face meshes and a synthetic "user looking at the screens" for tests and the demo.

:func:`face_mesh` places the landmarks the app reads (eye corners, lids, iris centres, nose, brows, chin)
for a given head turn, iris offset, roll, position and size; every other landmark sits on an ellipse around
the face so the face box is right. :class:`SyntheticViewer` turns a desktop gaze point into the mesh a user
would plausibly produce: the head turns part of the way toward the point, the eyes do the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from away.face.gaze import EYE_A, EYE_B, IRIS_A, IRIS_B, N_LANDMARKS_WITH_IRIS
from away.face.headpose import BROWS_MID, CHIN, NOSE_TIP


def face_mesh(
    yaw: float = 0.0,
    pitch: float = 0.0,
    iris_h: float = 0.0,
    iris_v: float = 0.0,
    roll: float = 0.0,
    center: tuple[float, float] = (0.5, 0.5),
    size: float = 0.3,
    lid_open: float = 0.25,
    n: int = N_LANDMARKS_WITH_IRIS,
) -> tuple[np.ndarray, np.ndarray]:
    """``(xs, ys)`` normalised image coordinates. ``yaw``/``pitch`` in degrees (+ = toward image right / down),
    ``iris_h``/``iris_v`` the iris offset as a fraction of the eye width (+ = right / down)."""
    pts = np.zeros((n, 2))
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts[:, 0] = 0.5 * np.cos(ang)
    pts[:, 1] = 0.5 * np.sin(ang)
    sy = math.sin(math.radians(yaw))
    sp = math.sin(math.radians(pitch))

    def put(i: int, x: float, y: float) -> None:
        pts[i] = (x, y)

    eye_w = 0.25
    for (outer, inner, up, low), iris, side in ((EYE_A, IRIS_A, -1.0), (EYE_B, IRIS_B, 1.0)):
        ex_o, ex_i = side * 0.4, side * 0.15
        cx = (ex_o + ex_i) / 2.0
        put(outer, ex_o, -0.1)
        put(inner, ex_i, -0.1)
        put(up, cx, -0.1 - lid_open * eye_w / 2.0)
        put(low, cx, -0.1 + lid_open * eye_w / 2.0)
        put(iris, cx + iris_h * eye_w, -0.1 + iris_v * eye_w)
    put(BROWS_MID, 0.0, -0.2)
    put(CHIN, 0.0, 0.5)
    put(NOSE_TIP, 0.5 * sy * 0.9, -0.2 + 0.7 * (0.5 + 0.4 * sp))
    r = math.radians(roll)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    pts = pts @ rot.T * size + np.asarray(center)
    return pts[:, 0].copy(), pts[:, 1].copy()


@dataclass
class SyntheticViewer:
    """A user at ``distance`` (in desktop widths) in front of a desktop of ``bounds``, who turns the head by
    ``head_share`` of the gaze angle; :meth:`look` returns the mesh, the head pose (degrees) and the box."""

    bounds: tuple[float, float, float, float] = (0.0, 0.0, 3840.0, 1080.0)
    distance: float = 0.7
    head_share: float = 0.4
    camera_yaw: float = 20.0  # an off-axis camera sees the head turned by this much when facing the screens
    noise: float = 0.0

    def angles(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, w, h = self.bounds
        u = (x - x0) / w - 0.5
        v = (y - y0) / w - 0.5 * h / w
        return math.degrees(math.atan2(u, self.distance)), math.degrees(math.atan2(v, self.distance))

    def look(
        self, x: float, y: float, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
        gyaw, gpitch = self.angles(x, y)
        hyaw, hpitch = gyaw * self.head_share, gpitch * self.head_share
        ih = math.sin(math.radians(gyaw - hyaw)) * 0.45
        iv = math.sin(math.radians(gpitch - hpitch)) * 0.35
        if rng is not None and self.noise > 0:
            ih += float(rng.normal(0, self.noise))
            iv += float(rng.normal(0, self.noise))
        cam_yaw = hyaw + self.camera_yaw
        xs, ys = face_mesh(cam_yaw, hpitch, ih, iv, lid_open=0.25 - 0.2 * iv)
        box = (float(xs.mean()), float(ys.mean()), float(max(np.ptp(xs), np.ptp(ys))))
        return xs, ys, (cam_yaw, hpitch, 0.0), box
