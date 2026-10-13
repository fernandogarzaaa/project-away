"""``away gaze calibrate`` / ``away gaze test``: the camera side and the Tk screens.

* :func:`face_stream` turns camera frames into ``(t, features, raw pose)`` with the Face Landmarker (and the
  optional learned gaze network), in a background thread feeding a queue.
* :func:`drive` is the pure loop between that stream, a :class:`~away.face.gaze.CalibrationSession` and a
  display callback; the tests run it with synthetic faces.
* :class:`DotScreens` (Tk) shows one borderless full-screen window per selected monitor with the current dot,
  a shrinking ring while it records; Esc cancels. :class:`GazeDot` (Tk) is the small always-on-top dot that
  follows the live gaze point for ``away gaze test``.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from away.face import gaze as gz


@dataclass
class FaceSample:
    t: float
    feats: np.ndarray | None
    pose: tuple[float, float, float] | None


def drive(
    session: gz.CalibrationSession,
    samples: Iterable[FaceSample],
    show: Callable[[tuple[int, float, float] | None, float], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> bool:
    """Feed samples into the session until it is done (True) or cancelled / the stream ends (False)."""
    if show is not None:
        show(session.current, 0.0)
    for s in samples:
        if cancelled():
            return False
        session.feed(s.t, s.feats, s.pose)
        if show is not None:
            show(session.current, session.progress(s.t))
        if session.done:
            return True
    return session.done


def camera_source(cfg: Any) -> tuple[Any, bool, int | None, str]:
    """(device, mirror, index hint, fourcc) of the camera the gaze calibration belongs to."""
    prim = next((c for c in cfg.hands.cameras if c.role == "primary"), None)
    if prim is not None:
        return prim.device, bool(prim.mirror), prim.index_hint, prim.fourcc
    return cfg.hands.camera_index, bool(cfg.hands.mirror), cfg.hands.camera_hint, cfg.hands.camera_fourcc


def face_stream(
    cfg: Any, models_dir: Path, stop: threading.Event, net_name: str | None = None
) -> Iterator[FaceSample]:  # pragma: no cover - needs a camera + mediapipe
    """Camera -> FaceSample, one per frame (feats None when no face)."""
    from away.face.landmarker import FaceLandmarker
    from away.hands.tracker import camera_frames

    device, mirror, hint, fourcc = camera_source(cfg)
    lm = FaceLandmarker(models_dir / "face_landmarker.task", max_faces=1)
    net = None
    if net_name:
        from away.app.optional_models import path_of
        from away.face.gaze_net import GazeNet

        net = GazeNet(path_of(net_name, models_dir), net_name, max_hz=30.0)
    try:
        for t, rgb in camera_frames(device, mirror, 640, 480, 30, hint=hint, fourcc=fourcc):
            if stop.is_set():
                return
            obs = lm.process(rgb, t)
            if not obs or obs[0].gaze is None:
                yield FaceSample(t, None, None)
                continue
            o = obs[0]
            f = o.gaze
            if net is not None:
                ang = net.update(t, rgb, o.box)
                if ang is None:
                    yield FaceSample(t, None, o.pose)
                    continue
                f = np.concatenate([f, np.asarray(ang, float)])
            yield FaceSample(t, f, o.pose)
    finally:
        lm.close()


def threaded(stream: Iterator[FaceSample], maxsize: int = 4) -> tuple[queue.Queue[FaceSample | None], threading.Thread]:
    """Run a blocking stream in a thread; the queue drops old frames when the UI is slower (keeps the newest)."""
    q: queue.Queue[FaceSample | None] = queue.Queue(maxsize=maxsize)

    def run() -> None:
        try:
            for s in stream:
                while True:
                    try:
                        q.put_nowait(s)
                        break
                    except queue.Full:
                        with contextlib.suppress(queue.Empty):
                            q.get_nowait()
        finally:
            q.put(None)

    th = threading.Thread(target=run, daemon=True, name="gaze-camera")
    th.start()
    return q, th


class DotScreens:  # pragma: no cover - needs a display
    """One borderless window per monitor; the current dot is drawn on its monitor."""

    def __init__(self, monitors: Sequence[Any], title: str = "Project Away gaze calibration") -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.monitors = list(monitors)
        self.cancelled = False
        self.windows = []
        self.canvases = []
        for m in self.monitors:
            w = tk.Toplevel(self.root)
            w.title(title)
            w.overrideredirect(True)
            w.geometry(f"{m.width}x{m.height}+{m.left}+{m.top}")
            w.configure(background="#101010")
            w.attributes("-topmost", True)
            c = tk.Canvas(w, width=m.width, height=m.height, background="#101010", highlightthickness=0)
            c.pack(fill="both", expand=True)
            w.bind("<Escape>", self._cancel)
            self.windows.append(w)
            self.canvases.append(c)
        self.windows[0].focus_force()
        self.text(0, "Look at each dot until it disappears. Keep your head natural. Esc cancels.")

    def _cancel(self, _e: Any = None) -> None:
        self.cancelled = True

    def text(self, mon: int, msg: str) -> None:
        c = self.canvases[mon]
        c.delete("msg")
        m = self.monitors[mon]
        c.create_text(m.width // 2, 40, text=msg, fill="#cccccc", font=("TkDefaultFont", 16), tags="msg")

    def show(self, target: tuple[int, float, float] | None, progress: float) -> None:
        for c in self.canvases:
            c.delete("dot")
        if target is not None:
            mon, x, y = target
            m = self.monitors[mon]
            cx, cy = x - m.left, y - m.top
            r = 22 * (1.0 - 0.7 * progress) + 6
            c = self.canvases[mon]
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#ffcc00", width=3, tags="dot")
            c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#ff3030", outline="", tags="dot")
        self.root.update()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.root.destroy()


class GazeDot:  # pragma: no cover - needs a display
    """A small always-on-top dot at the live gaze point (``away gaze test``)."""

    SIZE = 36

    def __init__(self) -> None:
        import tkinter as tk

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        key = "#ff00ff"
        self.root.configure(background=key)
        with contextlib.suppress(Exception):
            self.root.attributes("-transparentcolor", key)  # Windows: only the dot is visible
        c = tk.Canvas(self.root, width=self.SIZE, height=self.SIZE, background=key, highlightthickness=0)
        c.pack()
        c.create_oval(4, 4, self.SIZE - 4, self.SIZE - 4, outline="#ff3030", width=4)
        c.create_oval(self.SIZE / 2 - 3, self.SIZE / 2 - 3, self.SIZE / 2 + 3, self.SIZE / 2 + 3, fill="#ff3030")
        self.closed = False
        self.root.bind("<Escape>", lambda _e: self.close())

    def move(self, x: float, y: float) -> None:
        if self.closed:
            return
        h = self.SIZE // 2
        self.root.geometry(f"{self.SIZE}x{self.SIZE}+{round(x) - h}+{round(y) - h}")
        self.root.update()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with contextlib.suppress(Exception):
                self.root.destroy()


def queue_samples(
    q: queue.Queue[FaceSample | None], cancelled: Callable[[], bool], idle: Callable[[], None]
) -> Iterator[FaceSample]:
    """Samples from the camera thread's queue; calls ``idle`` (Tk update) while waiting."""
    while not cancelled():
        try:
            s = q.get(timeout=0.03)
        except queue.Empty:
            idle()
            continue
        if s is None:
            return
        yield s


def calibrate(
    cfg: Any, models_dir: Path, monitors: Sequence[Any], camera: str, say: Callable[[str], None] = print
) -> gz.Calibration | None:  # pragma: no cover - needs a camera, mediapipe and a display
    from away.app.optional_models import resolve_gaze_model

    net_name = resolve_gaze_model(cfg.gaze.model, models_dir)
    net = None if net_name == "iris" else net_name
    if net:
        say(f"using the learned gaze model '{net}' as extra features")
    stop = threading.Event()
    q, _th = threaded(face_stream(cfg, models_dir, stop, net))
    screens = DotScreens(monitors)
    session = gz.CalibrationSession(gz.calibration_targets(monitors, int(cfg.gaze.points_per_monitor)))
    try:
        t_end = time.monotonic() + 1.5  # read the instructions
        while time.monotonic() < t_end and not screens.cancelled:
            screens.root.update()
            time.sleep(0.02)
        ok = drive(
            session,
            queue_samples(q, lambda: screens.cancelled, screens.root.update),
            screens.show,
            lambda: screens.cancelled,
        )
    finally:
        stop.set()
        screens.close()
    if not ok:
        say("calibration cancelled" if screens.cancelled else "the camera stopped")
        return None
    if session.skipped:
        say(f"{len(session.skipped)} dot(s) got no face frames and were skipped")
    return session.result(monitors, float(cfg.gaze.ridge_alpha), camera, net)


def live_test(
    cfg: Any, models_dir: Path, cal: gz.Calibration, seconds: float, say: Callable[[str], None] = print
) -> int:  # pragma: no cover - needs a camera, mediapipe and a display
    from away.app.optional_models import resolve_gaze_model

    net_name = resolve_gaze_model(cfg.gaze.model, models_dir)
    net = net_name if net_name != "iris" and cal.net_model == net_name else None
    est = gz.GazeEstimator(cal, cfg.gaze.min_cutoff, cfg.gaze.beta)
    stop = threading.Event()
    q, _th = threaded(face_stream(cfg, models_dir, stop, net))
    dot = GazeDot()
    t_end = time.monotonic() + seconds
    n = 0
    try:
        for s in queue_samples(q, lambda: dot.closed or time.monotonic() > t_end, dot.root.update):
            if s.feats is None:
                continue
            pt = est.update(s.t, s.feats, net_available=net is not None)
            if pt is not None:
                dot.move(*pt)
                n += 1
    finally:
        stop.set()
        dot.close()
    say(f"{n} gaze points shown")
    return 0
