"""Camera capture + MediaPipe Hand Landmarker (Tasks API). Needs the [hands] extra.

Untested in the build sandbox (no camera); the landmarker itself was smoke-tested
on a still image there.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.hands.select import HandObs, HandSelector


@dataclass
class HandFrame:
    t: float
    landmarks: NDArray[np.float64] | None  # (21, 3) image coords of the selected hand (hands.prefer)
    handedness: str  # MediaPipe's label for this image (see away.hands.select for what it means)
    image: Any = None
    score: float = 1.0  # MediaPipe handedness confidence (used by multi-camera fusion)
    hand: str = ""  # the user's real hand: right | left ("" = no hand)
    n_hands: int = 0  # hands MediaPipe saw in this frame (the other one is ignored)
    hands_gap: float | None = None  # with 2+ hands: closest distance between palm centres (image units)


# Two hands cost more: with num_hands=2 and only one hand in view, MediaPipe re-runs palm detection on every
# frame looking for the second one (about 2x the CPU of num_hands=1 measured on one CPU). So a one-hand
# landmarker tracks normally and a two-hand one probes every PROBE_EVERY frames; once it sees two hands it takes
# over (both are then tracked, no palm detection) until it has seen one or none for DUAL_EXIT_FRAMES frames.
PROBE_EVERY = 6
DUAL_EXIT_FRAMES = 15


def _parse(res: Any) -> list[HandObs]:
    hands = []
    for i, pts in enumerate(res.hand_landmarks or []):
        lm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
        cats = res.handedness[i] if res.handedness and i < len(res.handedness) else []
        label = cats[0].category_name if cats else "Right"
        score = float(cats[0].score) if cats else 1.0
        hands.append(HandObs(lm, label, score))
    return hands


def hands_gap(hands: list[HandObs]) -> float | None:
    """Closest distance between two hands' palm centres (middle-finger MCP), None with fewer than 2 hands."""
    if len(hands) < 2:
        return None
    pts = [np.asarray(h.landmarks, dtype=np.float64)[9, :2] for h in hands]
    return min(float(np.linalg.norm(a - b)) for i, a in enumerate(pts) for b in pts[i + 1 :])


class HandTracker:
    def __init__(
        self,
        model_path: Path | None,
        max_hands: int = 1,
        min_conf: float = 0.5,
        *,
        prefer: str | Callable[[], str] = "any",
        mirrored: bool = True,
        factory: Callable[[int], Any] | None = None,
        image_fn: Callable[[Any], Any] | None = None,
        probe_every: int = PROBE_EVERY,
    ) -> None:
        """``factory(num_hands)`` builds a landmarker with ``detect_for_video(image, ms)`` (tests inject fakes)."""
        if factory is None:  # pragma: no cover - real MediaPipe
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            def factory(n: int) -> Any:
                opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=n,
                    min_hand_detection_confidence=min_conf,
                    min_hand_presence_confidence=min_conf,
                    min_tracking_confidence=min_conf,
                )
                return vision.HandLandmarker.create_from_options(opts)

            def image_fn(rgb: Any) -> Any:
                return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        self._image = image_fn or (lambda rgb: rgb)
        self.max_hands = max(1, int(max_hands))
        self.selector = HandSelector(prefer, mirrored)
        self.probe_every = max(1, int(probe_every))
        self._one = factory(1)
        self._multi = factory(self.max_hands) if self.max_hands > 1 else None
        self.landmarker = self._one  # kept for callers that poke at it
        self._last_ms = {id(self._one): -1, id(self._multi): -1}
        self.dual = False
        self._n = 0
        self._few = 0
        self._wrong = False
        self.multi_runs = 0  # frames the two-hand landmarker ran (tests / diagnostics)

    def _run(self, lmk: Any, image: Any, ms: int) -> list[HandObs]:
        key = id(lmk)
        if ms <= self._last_ms[key]:  # VIDEO mode needs strictly increasing timestamps per landmarker
            ms = self._last_ms[key] + 1
        self._last_ms[key] = ms
        if lmk is self._multi:
            self.multi_runs += 1
        return _parse(lmk.detect_for_video(image, ms))

    def detect(self, rgb: Any, t: float) -> list[HandObs]:
        """Every hand in the frame (one landmarker run, picked as described at PROBE_EVERY)."""
        ms = int(t * 1000)
        image = self._image(rgb)
        self._n += 1
        if self._multi is None:
            return self._run(self._one, image, ms)
        if self.dual:
            hands = self._run(self._multi, image, ms)
            self._few = self._few + 1 if len(hands) < 2 else 0
            if self._few >= DUAL_EXIT_FRAMES:
                self.dual = False
            return hands
        probe = self._n % self.probe_every == 0 or (self._wrong and self._n % 2 == 0)
        if probe:
            hands = self._run(self._multi, image, ms)
            if len(hands) >= 2:
                self.dual, self._few = True, 0
            return hands
        return self._run(self._one, image, ms)

    def process(self, rgb: NDArray[np.uint8], t: float) -> HandFrame:
        hands = self.detect(rgb, t)
        gap = hands_gap(hands)
        self._wrong = self.selector.wrong_side(hands)
        pick = self.selector.select(t, hands)
        if pick is None:
            return HandFrame(t, None, "Right", n_hands=len(hands), hands_gap=gap)
        return HandFrame(
            t,
            pick.landmarks,
            pick.label,
            score=pick.score,
            hand=self.selector.side(pick),
            n_hands=len(hands),
            hands_gap=gap,
        )

    def close(self) -> None:
        for lmk in (self._one, self._multi):
            if lmk is not None:
                lmk.close()


def camera_frames(
    index: int | str = 0,
    mirror: bool = True,
    width: int = 640,
    height: int = 480,
    fps: int = 0,
    hint: int | None = None,
    cv2: Any = None,
) -> Iterator[tuple[float, Any]]:
    """Yield (monotonic time, RGB frame). Mirrored by default so motion matches the screen.

    ``index`` is a camera name or an index, resolved to the camera's current index when the generator starts
    (so every retry re-resolves it and a hot-plugged camera cannot redirect it). On Windows the camera is
    opened with DirectShow, the numbering ``away cameras list`` shows (see :func:`open_video`)."""
    from away.multicam.capture import load_cv2, open_video, resolve_device

    if cv2 is None:  # pragma: no cover - real OpenCV
        cv2 = load_cv2()

    src = resolve_device(index, hint=hint)
    cap = open_video(src, cv2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open camera {index} (in use by another app, no device, or permission denied)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            yield time.monotonic(), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def probe_camera(index: int | str = 0, hint: int | None = None, cv2: Any = None) -> tuple[bool, str]:
    from away.multicam.capture import load_cv2, open_video, resolve_device

    if cv2 is None:  # pragma: no cover - real OpenCV
        try:
            cv2 = load_cv2()
        except ImportError:
            return False, "opencv not installed (pip install 'project-away[hands]')"

    try:
        src = resolve_device(index, hint=hint)
    except ValueError as exc:
        return False, str(exc)
    cap = open_video(src, cv2)
    try:
        if not cap.isOpened():
            return False, f"camera {index} could not be opened (no device or permission denied)"
        ok, _ = cap.read()
        return (True, f"camera {index} OK") if ok else (False, f"camera {index} opened but returned no frame")
    finally:
        cap.release()
