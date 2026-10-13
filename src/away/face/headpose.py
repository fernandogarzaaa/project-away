"""Head pose from the Face Landmarker, and what the app does with it (pure logic, no devices).

* :func:`pose_from_matrix`: yaw / pitch / roll (degrees) from MediaPipe's 4x4 facial transformation matrix
  (``output_facial_transformation_matrixes=True``, already computed by the landmarker, so it costs nothing).
  :func:`pose_from_points` is a rougher geometric fallback (nose vs eyes / brows / chin) for when there is no
  matrix (e.g. an older model or the iPhone source).
* :class:`SignAligner`: the matrix angles' sign convention depends on the camera mirroring. The aligner
  correlates them with the image-space nose offset (which is screen-aligned because frames are mirrored) and
  flips them so that **positive yaw = the user turns toward screen right, positive pitch = looks down**.
* :class:`HeadPoseFilter`: One Euro filter per angle.
* :class:`PoseBaseline`: slow EMA of the usual pose (where "looking at the screens" is for this user/desk).
* :class:`HeadPointer` (``head_assist: pointer``): head-turn deltas -> cursor deltas, Gameface-style.
* :class:`HeadCoarse` (``head_assist: coarse``): the head picks a zone of a monitor (with hysteresis and a
  dwell), the cursor jumps there, the hand refines.
* :class:`MonitorSwitch`: a sustained head turn toward another monitor (from the screen map) moves the cursor
  there; hysteresis + dwell so glances do not.
* :func:`facing_screen`: for the speech gate, is the head turned toward the screens (not merely toward the
  camera, which sits off-axis on many desks)?
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from away.hands.filters import OneEuroFilter

NOSE_TIP, EYE_OUTER_A, EYE_OUTER_B, CHIN, BROWS_MID = 1, 33, 263, 152, 168


@dataclass(frozen=True)
class HeadPose:
    yaw: float  # degrees, + = toward screen right (after :class:`SignAligner`)
    pitch: float  # degrees, + = looking down
    roll: float  # degrees


def pose_from_matrix(m: Any) -> HeadPose | None:
    """Euler angles of the rotation part of a 4x4 (or 3x3) matrix, as R = Ry(yaw) Rx(pitch) Rz(roll)."""
    try:
        r = [[float(m[i][j]) for j in range(3)] for i in range(3)]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for row in r for v in row):
        return None
    # normalise columns: the matrix may carry the face scale
    for j in range(3):
        n = math.sqrt(sum(r[i][j] ** 2 for i in range(3)))
        if n < 1e-9:
            return None
        for i in range(3):
            r[i][j] /= n
    pitch = math.asin(max(-1.0, min(1.0, -r[1][2])))
    yaw = math.atan2(r[0][2], r[2][2])
    roll = math.atan2(r[1][0], r[1][1])
    return HeadPose(math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def rotation_matrix(yaw: float, pitch: float, roll: float) -> list[list[float]]:
    """Inverse of :func:`pose_from_matrix` (degrees in, 4x4 out); used by the tests and synthetic faces."""
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]]
    rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]

    def mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    rot = mul(mul(ry, rx), rz)
    return [[*rot[0], 0.0], [*rot[1], 0.0], [*rot[2], 0.0], [0.0, 0.0, 0.0, 1.0]]


def nose_offsets(xs: Any, ys: Any) -> tuple[float, float] | None:
    """Screen-aligned head-turn proxies from the mesh: nose offset from the eye midpoint / eye span (+ = right)
    and nose height between brows (0) and chin (1), minus the frontal 0.5 (+ = down)."""
    try:
        if len(xs) <= max(NOSE_TIP, EYE_OUTER_A, EYE_OUTER_B, CHIN, BROWS_MID):
            return None
        span = abs(float(xs[EYE_OUTER_B]) - float(xs[EYE_OUTER_A]))
        height = float(ys[CHIN]) - float(ys[BROWS_MID])
        if span < 1e-6 or abs(height) < 1e-6:
            return None
        mid = (float(xs[EYE_OUTER_A]) + float(xs[EYE_OUTER_B])) / 2.0
        h = (float(xs[NOSE_TIP]) - mid) / span
        v = (float(ys[NOSE_TIP]) - float(ys[BROWS_MID])) / height - 0.5
    except (TypeError, ValueError, IndexError):
        return None
    return h, v


def pose_from_points(xs: Any, ys: Any) -> HeadPose | None:
    """Rough yaw/pitch from the mesh geometry alone (no roll): a fallback when there is no matrix."""
    off = nose_offsets(xs, ys)
    if off is None:
        return None
    h, v = off
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * h))))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.5 * v))))
    try:
        dx = float(xs[EYE_OUTER_B]) - float(xs[EYE_OUTER_A])
        dy = float(ys[EYE_OUTER_B]) - float(ys[EYE_OUTER_A])
        roll = math.degrees(math.atan2(dy, dx)) if dx >= 0 else math.degrees(math.atan2(-dy, -dx))
    except (TypeError, ValueError, IndexError):
        roll = 0.0
    return HeadPose(yaw, pitch, roll)


class SignAligner:
    """Flip the matrix yaw/pitch so they agree with the screen-aligned geometric proxies.

    The proxies carry a per-face offset (a real nose is not exactly halfway between brows and chin), so this
    correlates *deviations* from slow running means: ``acc += (angle - mean) * (proxy - mean)`` (decayed) and
    uses the sign of ``acc``. Until there is enough evidence (``min_evidence``) the sign is +1 (MediaPipe's
    matrix yaw already grows toward image right; checked on a real portrait, mirrored and not)."""

    def __init__(self, decay: float = 0.995, min_evidence: float = 2.0, mean_rate: float = 0.05) -> None:
        self.decay = decay
        self.min_evidence = min_evidence
        self.mean_rate = mean_rate
        self.yaw_acc = 0.0
        self.pitch_acc = 0.0
        self._means: list[float] | None = None  # yaw, pitch, h, v

    def update(self, pose: HeadPose, proxy: tuple[float, float] | None) -> HeadPose:
        if proxy is not None:
            vals = [pose.yaw, pose.pitch, proxy[0], proxy[1]]
            if self._means is None:
                self._means = list(vals)
            m = self._means
            dy, dp, dh, dv = (v - mu for v, mu in zip(vals, m))
            self.yaw_acc = self.yaw_acc * self.decay + dy * dh
            self.pitch_acc = self.pitch_acc * self.decay + dp * dv
            for i, v in enumerate(vals):
                m[i] += self.mean_rate * (v - m[i])
        return HeadPose(pose.yaw * self.yaw_sign, pose.pitch * self.pitch_sign, pose.roll)

    @property
    def yaw_sign(self) -> float:
        return -1.0 if self.yaw_acc < -self.min_evidence else 1.0

    @property
    def pitch_sign(self) -> float:
        return -1.0 if self.pitch_acc < -self.min_evidence else 1.0


class HeadPoseFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05) -> None:
        self.f = [OneEuroFilter(min_cutoff, beta) for _ in range(3)]

    def configure(self, min_cutoff: float, beta: float) -> None:
        for f in self.f:
            f.min_cutoff, f.beta = min_cutoff, beta

    def reset(self) -> None:
        for f in self.f:
            f.reset()

    def __call__(self, t: float, pose: HeadPose) -> HeadPose:
        return HeadPose(self.f[0](t, pose.yaw), self.f[1](t, pose.pitch), self.f[2](t, pose.roll))


class PoseBaseline:
    """Slow EMA of the head pose: the user's usual "looking at the screens" direction. Learns fast at first
    (``warm_tau_s`` for the first ``warm_s`` seconds of samples), then slowly (``tau_s``)."""

    def __init__(self, tau_s: float = 60.0, warm_tau_s: float = 1.0, warm_s: float = 3.0) -> None:
        self.tau_s, self.warm_tau_s, self.warm_s = tau_s, warm_tau_s, warm_s
        self.yaw: float | None = None
        self.pitch = 0.0
        self.seen_s = 0.0
        self._t: float | None = None

    def update(self, t: float, pose: HeadPose) -> None:
        if self.yaw is None or self._t is None:
            self.yaw, self.pitch, self._t = pose.yaw, pose.pitch, t
            return
        dt = t - self._t
        self._t = t
        if dt <= 0.0 or dt > 1.0:
            return
        self.seen_s += dt
        tau = self.warm_tau_s if self.seen_s < self.warm_s else self.tau_s
        a = 1.0 - math.exp(-dt / tau)
        self.yaw += a * (pose.yaw - self.yaw)
        self.pitch += a * (pose.pitch - self.pitch)

    @property
    def ready(self) -> bool:
        return self.yaw is not None and self.seen_s >= self.warm_s


def facing_screen(
    pose: HeadPose,
    baseline: PoseBaseline | None,
    tolerance: float,
    pose_box: tuple[float, float, float, float] | None = None,
) -> bool | None:
    """Is the head turned toward the screens? With a screen map (``pose_box`` = yaw/pitch min/max seen while
    looking at the calibration dots) the pose must be inside that box widened by ``tolerance``; otherwise it
    must be within ``tolerance`` degrees of the learned usual pose. None = cannot tell yet."""
    if pose_box is not None:
        y0, y1, p0, p1 = pose_box
        return bool(y0 - tolerance <= pose.yaw <= y1 + tolerance and p0 - tolerance <= pose.pitch <= p1 + tolerance)
    if baseline is None or not baseline.ready or baseline.yaw is None:
        return None
    return bool(abs(pose.yaw - baseline.yaw) <= tolerance and abs(pose.pitch - baseline.pitch) <= tolerance)


class HeadPointer:
    """Head-only pointer: per-frame head-turn delta (degrees) -> cursor delta (pixels).

    ``gain`` is in reference-monitor widths per degree; a power curve (``acceleration``) makes slow turns
    precise and quick turns cross the screen; deltas below ``dead_zone`` degrees are tremor."""

    def __init__(self, gain: float = 0.012, dead_zone: float = 0.15, acceleration: float = 1.4) -> None:
        self.gain, self.dead_zone, self.acceleration = gain, dead_zone, acceleration
        self._prev: HeadPose | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, pose: HeadPose, ref_width: float) -> tuple[float, float]:
        prev, self._prev = self._prev, pose
        if prev is None:
            return 0.0, 0.0
        dyaw, dpitch = pose.yaw - prev.yaw, pose.pitch - prev.pitch
        mag = math.hypot(dyaw, dpitch)
        if mag < self.dead_zone or mag == 0.0:
            self._prev = prev  # tremor: keep the anchor so slow, steady turns still add up
            return 0.0, 0.0
        k = max(1.0, mag) ** (self.acceleration - 1.0)
        scale = self.gain * ref_width * k
        return dyaw * scale, dpitch * scale


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height


def _rects(monitors: Sequence[Any]) -> list[Rect]:
    return [Rect(float(m.left), float(m.top), float(m.width), float(m.height)) for m in monitors]


class HeadCoarse:
    """Head-coarse / hand-fine: the head picks a zone, the cursor jumps to the zone centre.

    The head offset from the usual pose maps linearly onto the desktop (``yaw_span`` degrees from the centre to
    the left/right edge, ``pitch_span`` to the top/bottom). Each monitor is split into ``cols`` x ``rows``
    zones. The zone only changes once the mapped point is ``hysteresis`` of a zone past the border and the new
    zone was held ``dwell_s`` seconds; then :meth:`update` returns that zone's centre (else None)."""

    def __init__(
        self,
        cols: int = 3,
        rows: int = 3,
        yaw_span: float = 30.0,
        pitch_span: float = 15.0,
        hysteresis: float = 0.25,
        dwell_s: float = 0.35,
    ) -> None:
        self.cols, self.rows = cols, rows
        self.yaw_span, self.pitch_span = yaw_span, pitch_span
        self.hysteresis, self.dwell_s = hysteresis, dwell_s
        self.zone: tuple[int, int, int] | None = None  # (monitor, col, row)
        self._cand: tuple[int, int, int] | None = None
        self._cand_t = 0.0

    def reset(self) -> None:
        self.zone = None
        self._cand = None

    def point(self, offset: tuple[float, float], monitors: Sequence[Any]) -> tuple[float, float]:
        """Desktop point for a (yaw, pitch) offset from the usual pose."""
        rs = _rects(monitors)
        x0, y0 = min(r.left for r in rs), min(r.top for r in rs)
        x1, y1 = max(r.right for r in rs), max(r.bottom for r in rs)
        u = min(max(0.5 + offset[0] / (2.0 * self.yaw_span), 0.0), 1.0)
        v = min(max(0.5 + offset[1] / (2.0 * self.pitch_span), 0.0), 1.0)
        return x0 + u * (x1 - x0 - 1), y0 + v * (y1 - y0 - 1)

    def zone_of(self, x: float, y: float, monitors: Sequence[Any]) -> tuple[int, int, int]:
        rs = _rects(monitors)
        best = min(
            range(len(rs)),
            key=lambda i: math.hypot(
                max(rs[i].left - x, 0.0, x - rs[i].right + 1), max(rs[i].top - y, 0.0, y - rs[i].bottom + 1)
            ),
        )
        r = rs[best]
        c = int(min(max((x - r.left) / r.width * self.cols, 0.0), self.cols - 1e-9))
        w = int(min(max((y - r.top) / r.height * self.rows, 0.0), self.rows - 1e-9))
        return best, c, w

    def center_of(self, zone: tuple[int, int, int], monitors: Sequence[Any]) -> tuple[float, float]:
        r = _rects(monitors)[zone[0]]
        return r.left + (zone[1] + 0.5) * r.width / self.cols, r.top + (zone[2] + 0.5) * r.height / self.rows

    def _inside_with_margin(self, zone: tuple[int, int, int], x: float, y: float, monitors: Sequence[Any]) -> bool:
        r = _rects(monitors)[zone[0]]
        zw, zh = r.width / self.cols, r.height / self.rows
        left, top = r.left + zone[1] * zw, r.top + zone[2] * zh
        mx, my = self.hysteresis * zw, self.hysteresis * zh
        return left - mx <= x <= left + zw + mx and top - my <= y <= top + zh + my

    def update(self, t: float, offset: tuple[float, float], monitors: Sequence[Any]) -> tuple[float, float] | None:
        x, y = self.point(offset, monitors)
        if self.zone is not None and self._inside_with_margin(self.zone, x, y, monitors):
            self._cand = None
            return None
        z = self.zone_of(x, y, monitors)
        if z == self.zone:
            self._cand = None
            return None
        if z != self._cand:
            self._cand, self._cand_t = z, t
        if t - self._cand_t < self.dwell_s:
            return None
        first = self.zone is None
        self.zone, self._cand = z, None
        return None if first else self.center_of(z, monitors)


class MonitorSwitch:
    """Head-turn monitor switch with hysteresis and dwell.

    ``poses[i]`` is the (yaw, pitch) the head had while looking at monitor ``i`` during calibration. A switch
    to monitor ``j`` needs the pose to be ``margin`` degrees closer to ``poses[j]`` than to the current
    monitor's pose, continuously for ``dwell_s`` seconds."""

    def __init__(self, poses: Sequence[tuple[float, float]], margin: float = 4.0, dwell_s: float = 0.6) -> None:
        self.poses = list(poses)
        self.margin, self.dwell_s = margin, dwell_s
        self._cand: int | None = None
        self._cand_t = 0.0

    def reset(self) -> None:
        self._cand = None

    def update(self, t: float, pose: HeadPose, current: int) -> int | None:
        if len(self.poses) < 2 or not 0 <= current < len(self.poses):
            self._cand = None
            return None

        def dist(i: int) -> float:
            return math.hypot(pose.yaw - self.poses[i][0], pose.pitch - self.poses[i][1])

        best = min(range(len(self.poses)), key=dist)
        if best == current or dist(current) - dist(best) < self.margin:
            self._cand = None
            return None
        if best != self._cand:
            self._cand, self._cand_t = best, t
        if t - self._cand_t < self.dwell_s:
            return None
        self._cand = None
        return best
