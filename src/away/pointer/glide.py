"""Smooth cursor output at a fixed rate (v0.1.23, ``hands.cursor.output_hz``, default 120).

The cursor used to move once per hand frame. At 30 fps that is fine; when the camera dropped to 4-10 fps (a
failing second camera being re-probed, USB bandwidth) every move became a visible jump: "too choppy". Now each
hand frame only sets a *target* (already One Euro / predictive filtered by the cursor mapper), and a small output
thread moves the pointer ``output_hz`` times a second:

* it glides toward the newest target with a short time constant (a new frame never teleports the pointer);
* between frames it keeps moving along the hand's last velocity, but at most about one frame interval ahead
  (``max_ahead_s``), so a slow camera still gives a moving cursor without overshooting a stop;
* when no frame arrives for ``stall_frames`` intervals (the hand stopped being tracked) it holds still.

:class:`Glide` is pure logic (time is passed in); :class:`GlideRunner` is the thread. Nothing else runs on that
thread: camera checks, smart-pointer UI scans and agent messages all run on their own threads.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any

DEFAULT_HZ = 120
TAU_S = 0.018  # how fast the output closes in on the predicted point
MAX_AHEAD_S = 0.12  # never predict further than this (about one frame at 8 fps)
STALL_FRAMES = 2.5  # no frame for this many frame intervals: hold still


class Glide:
    def __init__(
        self, tau_s: float = TAU_S, max_ahead_s: float = MAX_AHEAD_S, stall_frames: float = STALL_FRAMES
    ) -> None:
        self.tau_s = float(tau_s)
        self.max_ahead_s = float(max_ahead_s)
        self.stall_frames = float(stall_frames)
        self._lock = threading.Lock()
        self.target: tuple[float, float] | None = None
        self.t_target = 0.0
        self.vel = (0.0, 0.0)
        self.frame_dt = 1.0 / 30.0
        self.out: tuple[float, float] | None = None
        self.t_out = 0.0
        self.sent: tuple[int, int] | None = None

    def _clear(self) -> None:
        self.target = None
        self.vel = (0.0, 0.0)
        self.out = None
        self.sent = None

    def on_target(self, t: float, x: float, y: float) -> None:
        """A hand frame's cursor position (screen px) at time ``t`` (monotonic seconds)."""
        with self._lock:
            if self.target is not None and t > self.t_target:
                dt = t - self.t_target
                if dt < 1.0:
                    self.frame_dt = 0.7 * self.frame_dt + 0.3 * dt  # the camera's real frame interval
                    vx, vy = (x - self.target[0]) / dt, (y - self.target[1]) / dt
                    self.vel = (0.5 * self.vel[0] + 0.5 * vx, 0.5 * self.vel[1] + 0.5 * vy)
                else:
                    self.vel = (0.0, 0.0)
            if self.out is None:
                self.out, self.t_out = (x, y), t
            self.target, self.t_target = (x, y), t

    def jump(self, x: float, y: float, t: float) -> None:
        """The cursor was placed directly (steady click, monitor jump): continue from there, no glide."""
        with self._lock:
            self.target, self.t_target, self.out, self.t_out = (x, y), t, (x, y), t
            self.vel = (0.0, 0.0)
            self.sent = (round(x), round(y))

    def stop(self) -> None:
        """Hand lost / paused: hold still and forget the motion."""
        with self._lock:
            self._clear()

    def predicted(self, now: float) -> tuple[float, float] | None:
        if self.target is None:
            return None
        gap = now - self.t_target
        if gap > self.stall_frames * self.frame_dt:
            return self.target  # stalled: no more extrapolation
        ahead = max(0.0, min(gap, self.frame_dt, self.max_ahead_s))
        return (self.target[0] + self.vel[0] * ahead, self.target[1] + self.vel[1] * ahead)

    def step(self, now: float) -> tuple[int, int] | None:
        """The position to send now, or None when the pointer should not move (no target / under 1 px)."""
        with self._lock:
            goal = self.predicted(now)
            if goal is None or self.out is None:
                return None
            dt = max(0.0, now - self.t_out)
            a = 1.0 - math.exp(-dt / self.tau_s) if self.tau_s > 0 else 1.0
            self.out = (self.out[0] + (goal[0] - self.out[0]) * a, self.out[1] + (goal[1] - self.out[1]) * a)
            self.t_out = now
            px = (round(self.out[0]), round(self.out[1]))
            if px == self.sent:
                return None
            self.sent = px
            return px


class GlideRunner:
    """The fixed-rate output thread. ``move`` is the (guarded) mouse backend's move."""

    def __init__(
        self,
        glide: Glide,
        move: Callable[[int, int], Any],
        hz: float = DEFAULT_HZ,
        clock: Callable[[], float] = time.monotonic,
        active: Callable[[], bool] = lambda: True,
    ) -> None:
        self.glide = glide
        self.move = move
        self.period = 1.0 / max(1.0, float(hz))
        self.clock = clock
        self.active = active
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.ticks = 0
        self.errors = 0

    def tick(self) -> bool:
        if not self.active():
            return False
        p = self.glide.step(self.clock())
        self.ticks += 1
        if p is None:
            return False
        try:
            self.move(p[0], p[1])
        except Exception:
            self.errors += 1
            return False
        return True

    def run(self) -> None:
        nxt = self.clock()
        while not self.stop_event.is_set():
            self.tick()
            nxt += self.period
            delay = nxt - self.clock()
            if delay < -0.25:  # fell far behind (sleeping PC): start over instead of catching up
                nxt, delay = self.clock(), 0.0
            self.stop_event.wait(max(0.0, delay))

    def start(self) -> GlideRunner:
        self.thread = threading.Thread(target=self.run, daemon=True, name="cursor-output")
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
