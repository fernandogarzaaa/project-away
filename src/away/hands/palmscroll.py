"""Palm-glide scroll: the helper hand is a trackpad, the pointer's index fingertip glides on it. Pure logic.

Hold one hand flat (the *helper*, :mod:`away.hands.roles`) and glide the other hand's index fingertip along
its palm:

* toward the fingers = scroll up, toward the wrist = scroll down (``invert`` swaps them);
* index + middle finger together glide sideways = horizontal scroll (``horizontal``);
* lift the fingertip while gliding and the page keeps going, slowing down (``momentum``, ``friction``);
* a quick tap on the palm (touch and lift within :data:`TAP_MAX_S`, moving less than ``dead_zone``) fires the
  bindable ``gesture:palm_tap`` trigger (unbound by default).

Geometry
--------
Both hands are put in one camera frame measured in **palm lengths** (wrist -> middle MCP):

* with multi-camera fusion, the triangulated 3D points (metres / ``palm_length_m``);
* with one camera, from MediaPipe's image landmarks through a pinhole (``hfov``): each landmark's relative
  ``z`` (MediaPipe gives it roughly in image-width units, wrist = 0) sets its depth relative to the wrist, and
  the hand's distance is chosen so its palm is one palm length long (both hands are assumed to have the same
  palm length, so the apparent size difference gives the depth between them).

The helper's palm frame: origin = wrist (0), ``v`` = wrist -> middle MCP (9), normal ``n`` from the index and
pinky MCPs, ``u`` = ``v x n`` (signed so it points to the user's right in the mirrored image, and kept
continuous between frames). So a rotated or tilted palm gives the same local coordinates.

It engages when, for ``engage_s``: the helper is flat (four fingers out), the pointer's index fingertip (8)
projects inside the palm outline (0, 1, 5, 9, 13, 17) padded by ``margin``, and the fingertip is within
``depth_threshold`` palm lengths of the palm plane (0 = no depth check). While the fingertip touches the
palm (engaging or engaged), :attr:`PalmScroller.active` is True and the pipeline suppresses the pointer's
cursor movement, clicks, auto-scroll and open-palm pause.

Glide in palm lengths -> wheel notches with ``gain``; a ratcheting ``dead_zone`` swallows jitter. Notches are
fractional and go through the mouse backend's accumulating scroll path. When the helper's landmarks drop out
or jump while the fingertip covers it, the last good palm frame is held for :data:`OCCLUSION_HOLD_S`. The
fingertip is smoothed with the One Euro filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.config import PalmScrollConfig
from away.hands.filters import OneEuroFilter
from away.hands.gestures import GestureEvent
from away.hands.landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    RING_MCP,
    WRIST,
    as_array,
    extended_fingers,
)

Array = NDArray[np.float64]

TAP_MAX_S = 0.25  # touch and lift within this = a tap
TAP_MIN_FRAMES = 2  # a one-frame touch is noise
OCCLUSION_HOLD_S = 0.3  # keep the last good palm frame this long while the fingertip hides the helper
RELEASE_GRACE_S = 0.1  # an engaged glide survives a contact dropout this short
JUMP_PALMS = 0.6  # the helper's wrist moving more than this (palm lengths) in one frame is a bad detection
JUMP_SCALE = 0.35  # ...or its palm length changing by more than this fraction
MOMENTUM_MIN = 2.0  # notches/s: a slower glide stops dead when the fingertip lifts
MOMENTUM_STOP = 0.5  # notches/s: momentum below this ends
MAX_FRAME_DT = 0.1  # a stalled frame never produces a burst
VEL_ALPHA = 0.5  # EMA step of the glide speed used for momentum
DEFAULT_ASPECT = 0.75  # image height / width (640x480)
DEFAULT_HFOV = 70.0
PALM_OUTLINE = (WRIST, 1, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FILTER = (1.5, 0.3, 1.0)  # One Euro min_cutoff (Hz), beta, d_cutoff for the fingertip (palm-length units)


# ------------------------------------------------------------------------------------------- geometry
def hand_points(
    lm: Any,
    points3d: Any = None,
    metric_scale: float | None = None,
    aspect: float = DEFAULT_ASPECT,
    hfov_deg: float = DEFAULT_HFOV,
    mirrored: bool = True,
) -> Array:
    """(21, 3) camera-frame points in palm lengths (see the module docstring). Triangulated ``points3d`` are
    in the real camera frame; with ``mirrored`` their x is flipped to match the mirrored image (user's right =
    +x), like the image landmarks."""
    if points3d is not None:
        p = np.asarray(points3d, dtype=np.float64) / float(metric_scale or 0.09)
        if mirrored:
            p = p * np.array([-1.0, 1.0, 1.0])
        return p
    a = as_array(lm)
    f = 1.0 / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    # MediaPipe z is relative to the wrist in image-width units: depth_i = D * (1 + z_i / f) for the wrist's
    # distance D. Every reconstructed point scales with D, so build them for D = 1 and pick D so the palm is
    # one palm length long (index -> pinky MCP / 0.8 as well, so a palm turned edge-on still gets its size).
    k = 1.0 + (a[:, 2] - a[WRIST, 2]) / f
    p = np.column_stack([(a[:, 0] - 0.5) * k / f, (a[:, 1] - 0.5) * aspect * k / f, k])
    length = float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]))
    width = float(np.linalg.norm(p[INDEX_MCP] - p[PINKY_MCP])) / 0.8
    size = max(length, width, 1e-9)
    out: Array = p / size
    return out


@dataclass
class PalmFrame:
    origin: Array
    u: Array
    v: Array
    n: Array
    length: float
    outline: Array  # (k, 2) convex outline of the palm in local (u, v)

    def local(self, p: Any) -> tuple[float, float, float]:
        d = np.asarray(p, dtype=np.float64) - self.origin
        L = self.length
        return float(d @ self.u) / L, float(d @ self.v) / L, float(d @ self.n) / L


def _hull(pts: Array) -> Array:
    """Convex hull (monotone chain), counter-clockwise."""
    p = sorted({(float(x), float(y)) for x, y in pts})
    if len(p) <= 2:
        return np.array(p)

    def cross(o: Any, a: Any, b: Any) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Any] = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[Any] = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def palm_frame(points: Array, prev_u: Array | None = None) -> PalmFrame | None:
    """The helper palm's local frame from its (21, 3) points, or None when degenerate."""
    o = points[WRIST]
    vv = points[MIDDLE_MCP] - o
    L = float(np.linalg.norm(vv))
    if L < 1e-6:
        return None
    v = vv / L
    n = np.cross(points[INDEX_MCP] - o, points[PINKY_MCP] - o)
    n = n - (n @ v) * v
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n = n / nn
    u = np.cross(v, n)
    u = u / float(np.linalg.norm(u))
    flip = (float(u @ prev_u) < 0) if prev_u is not None else (u[0] < 0)
    if flip:
        u, n = -u, -n
    fr = PalmFrame(o, u, v, n, L, np.zeros((0, 2)))
    loc = np.array([fr.local(points[i])[:2] for i in PALM_OUTLINE])
    fr.outline = _hull(loc)
    return fr


def outline_distance(outline: Array, uv: tuple[float, float]) -> float:
    """Signed distance from ``uv`` to the outline: <= 0 inside, > 0 outside (palm lengths)."""
    if len(outline) < 3:
        return math.inf
    x, y = uv
    inside = True
    best = math.inf
    k = len(outline)
    for i in range(k):
        ax, ay = outline[i]
        bx, by = outline[(i + 1) % k]
        ex, ey = bx - ax, by - ay
        if ex * (y - ay) - ey * (x - ax) < 0:  # right of a CCW edge: outside
            inside = False
        seg = ex * ex + ey * ey
        tt = 0.0 if seg <= 0 else max(0.0, min(1.0, ((x - ax) * ex + (y - ay) * ey) / seg))
        best = min(best, math.hypot(x - (ax + tt * ex), y - (ay + tt * ey)))
    return -best if inside else best


def helper_flat(lm: Any, strict: bool = True) -> bool:
    """Four fingers extended (``strict``); three are enough to stay engaged (the fingertip hides one)."""
    n = sum(extended_fingers(as_array(lm)).values())
    return n >= (4 if strict else 3)


def pointer_fingers(lm: Any) -> str:
    """``index`` (index out), ``two`` (index + middle out, ring + pinky in) or ``none``."""
    ext = extended_fingers(as_array(lm))
    if not ext["index"]:
        return "none"
    if ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "two"
    return "index"


def direction(dx: float, dy: float) -> str | None:
    if dx == 0 and dy == 0:
        return None
    if abs(dy) >= abs(dx):
        return "up" if dy > 0 else "down"
    return "right" if dx > 0 else "left"


# ------------------------------------------------------------------------------------------- state machine
class PalmScroller:
    def __init__(
        self, cfg: PalmScrollConfig | None = None, aspect: float = DEFAULT_ASPECT, hfov_deg: float = DEFAULT_HFOV
    ):
        self.cfg = cfg or PalmScrollConfig()
        self.aspect = aspect
        self.hfov_deg = hfov_deg
        self.mirrored = True  # the primary camera's mirror flag (only matters for triangulated 3D points)
        self._f = [OneEuroFilter(*FILTER) for _ in range(3)]
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        self.touching = False
        self.engaged = False
        self.momentum = False
        self.vx = self.vy = 0.0  # notches/s (momentum / glide speed)
        self.last_dir: str | None = None
        self._contact_t0 = 0.0
        self._contact_last = -1e9
        self._contact_frames = 0
        self._start: tuple[float, float] | None = None
        self._anchor: tuple[float, float] | None = None
        self._axis: str | None = None
        self._max_disp = 0.0
        self._last_t: float | None = None
        self._good: PalmFrame | None = None
        self._good_t = -1e9
        self._prev_u: Array | None = None
        for f in self._f:
            f.reset()

    @property
    def active(self) -> bool:
        """The fingertip is on the palm (engaging or engaged): suppress the pointer's other gestures."""
        return self.touching or self.engaged

    @property
    def direction(self) -> str | None:
        return self.last_dir if (self.engaged or self.momentum) else None

    def stop_momentum(self) -> bool:
        """A pointer gesture (click, drag, scroll...) ends the coast. True if momentum was running."""
        was = self.momentum
        if was:
            self.momentum = False
            self.vx = self.vy = 0.0
        return was

    # ---------------------------------------------------------------- helpers
    def _helper_frame(self, t: float, helper: Any, helper3d: Any, metric: float | None) -> PalmFrame | None:
        """This frame's palm frame, or the held one while the helper is occluded / jumped (OCCLUSION_HOLD_S)."""
        fr: PalmFrame | None = None
        strict = not self.active
        if helper is not None and helper_flat(helper, strict=strict):
            pts = hand_points(helper, helper3d, metric, self.aspect, self.hfov_deg, self.mirrored)
            fr = palm_frame(pts, self._prev_u)
            g = self._good
            if fr is not None and g is not None and self.active and t - self._good_t <= OCCLUSION_HOLD_S:
                jump = float(np.linalg.norm(fr.origin - g.origin)) / g.length
                if jump > JUMP_PALMS or abs(fr.length / g.length - 1.0) > JUMP_SCALE:
                    fr = None  # a bad detection while the fingertip covers the palm
        if fr is not None:
            self._good, self._good_t, self._prev_u = fr, t, fr.u
            return fr
        if self._good is not None and self.active and t - self._good_t <= OCCLUSION_HOLD_S:
            return self._good
        return None

    def _tip(self, t: float, pointer: Any, pointer3d: Any, metric: float | None) -> Array:
        p = hand_points(pointer, pointer3d, metric, self.aspect, self.hfov_deg, self.mirrored)[INDEX_TIP]
        return np.array([self._f[i](t, float(p[i])) for i in range(3)])

    def _end(self, t: float, reason: str) -> list[GestureEvent]:
        out: list[GestureEvent] = []
        dur = self._contact_last - self._contact_t0
        if (
            self._contact_frames >= TAP_MIN_FRAMES
            and dur <= TAP_MAX_S
            and self._max_disp < max(self.cfg.dead_zone, 1e-9)
            and reason == "lift"
        ):
            out.append(GestureEvent("palm_tap", t, {"duration": dur}))
        if self.engaged:
            out.append(GestureEvent("palm_scroll_stop", t, {"reason": reason}))
            speed = max(abs(self.vx), abs(self.vy))
            if self.cfg.momentum and speed >= MOMENTUM_MIN and reason in ("lift", "pointer_lost"):
                self.momentum = True
            else:
                self.vx = self.vy = 0.0
        self.touching = self.engaged = False
        self._anchor = self._start = self._axis = None
        self._contact_frames = 0
        self._max_disp = 0.0
        for f in self._f:
            f.reset()
        return out

    def _momentum_step(self, t: float, dt: float) -> list[GestureEvent]:
        k = math.exp(-self.cfg.friction * dt)
        self.vx *= k
        self.vy *= k
        if max(abs(self.vx), abs(self.vy)) < MOMENTUM_STOP:
            self.momentum = False
            self.vx = self.vy = 0.0
            return [GestureEvent("palm_scroll_stop", t, {"reason": "momentum_end"})]
        dx, dy = self.vx * dt, self.vy * dt
        return [GestureEvent("palm_scroll", t, {"dx": dx, "dy": dy, "momentum": True, "direction": self.last_dir})]

    # ---------------------------------------------------------------- step
    def update(
        self,
        t: float,
        pointer: Any,
        helper: Any,
        pointer3d: Any = None,
        helper3d: Any = None,
        metric_scale: float | None = None,
    ) -> list[GestureEvent]:
        """Feed one frame: the pointer's and the helper's landmarks (None when not seen). Returns events:
        ``palm_scroll_start``, ``palm_scroll`` (``dx``/``dy`` notches, + = right / up), ``palm_scroll_stop``,
        ``palm_tap``."""
        cfg = self.cfg
        dt = 0.0 if self._last_t is None else min(MAX_FRAME_DT, max(0.0, t - self._last_t))
        self._last_t = t
        if not cfg.enabled:
            out = self._end(t, "disabled") if self.active else []
            self.momentum = False
            self.vx = self.vy = 0.0
            return out
        if pointer3d is None or helper3d is None:  # both 3D or both monocular: one consistent frame
            pointer3d = helper3d = None

        contact = False
        uvw: tuple[float, float, float] | None = None
        fingers = "none"
        if pointer is not None:
            fingers = pointer_fingers(pointer)
            frame = self._helper_frame(t, helper, helper3d, metric_scale) if fingers != "none" else None
            if frame is not None:
                tip = self._tip(t, pointer, pointer3d, metric_scale)
                uvw = frame.local(tip)
                inside = outline_distance(frame.outline, (uvw[0], uvw[1])) <= cfg.margin
                near = cfg.depth_threshold <= 0 or abs(uvw[2]) <= cfg.depth_threshold
                contact = inside and near
        elif helper is not None:
            self._helper_frame(t, helper, helper3d, metric_scale)  # keep the palm frame fresh

        events: list[GestureEvent] = []
        if not contact:
            if self.active:
                if self.engaged and t - self._contact_last <= RELEASE_GRACE_S:
                    return events  # a short contact / landmark dropout: hold, do not scroll
                events += self._end(t, "lift" if pointer is not None else "pointer_lost")
                return events
            if self.momentum:
                events += self._momentum_step(t, dt)
            return events

        assert uvw is not None
        pos = (uvw[0], uvw[1])
        if not self.touching:  # first contact
            self.touching = True
            self.momentum = False
            self.vx = self.vy = 0.0
            self._contact_t0 = t
            self._contact_frames = 0
            self._start = self._anchor = pos
            self._max_disp = 0.0
        self._contact_last = t
        self._contact_frames += 1
        assert self._start is not None
        self._max_disp = max(self._max_disp, math.hypot(pos[0] - self._start[0], pos[1] - self._start[1]))
        if not self.engaged:
            if t - self._contact_t0 < cfg.engage_s:
                return events
            self.engaged = True
            events.append(GestureEvent("palm_scroll_start", t, {"u": pos[0], "v": pos[1]}))
        axis = "u" if (fingers == "two" and cfg.horizontal) else "v"
        if axis != self._axis:
            self._axis = axis
            self._anchor = pos  # switching between vertical and horizontal: start from here
        assert self._anchor is not None
        k = 0 if axis == "u" else 1
        delta = pos[k] - self._anchor[k]
        dz = cfg.dead_zone
        moved = 0.0
        if abs(delta) > dz:
            moved = delta - math.copysign(dz, delta)
            a = list(self._anchor)
            a[k] += moved
            self._anchor = (a[0], a[1])
        notches = moved * cfg.gain
        dx = notches if axis == "u" else 0.0
        dy = notches if axis == "v" else 0.0
        if cfg.invert:
            dx, dy = -dx, -dy
        if dt > 0:
            self.vx = (1 - VEL_ALPHA) * self.vx + VEL_ALPHA * (dx / dt)
            self.vy = (1 - VEL_ALPHA) * self.vy + VEL_ALPHA * (dy / dt)
            if axis == "u":
                self.vy = 0.0
            else:
                self.vx = 0.0
        if dx or dy:
            self.last_dir = direction(dx, dy)
            events.append(GestureEvent("palm_scroll", t, {"dx": dx, "dy": dy, "direction": self.last_dir}))
        return events
