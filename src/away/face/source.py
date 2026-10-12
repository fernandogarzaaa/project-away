"""Which face source drives the face pipeline: the webcam (MediaPipe), an iPhone (Live Link Face), or auto.

``face.source``:

* ``webcam`` (default): MediaPipe Face Landmarker on the camera frames, exactly as before.
* ``iphone``: blendshapes from the iPhone only; the webcam face model is not run at all.
* ``auto``: the iPhone while fresh packets with a face arrive, the webcam when they go stale.

The iPhone sees exactly one face (the person in front of the phone), so it is fed to the
pipeline as a single face at a fixed box; the active-speaker detector then runs in its
single-face mode (mouth motion gates dictation) and the enrolled user is that face.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

from away.config import FACE_SOURCES
from away.face.livelink import LiveLinkReceiver, to_mediapipe

__all__ = ["FACE_SOURCES", "IPHONE_BOX", "FacePipeline", "FaceSourceSelector", "IPhoneFaceFeed"]
IPHONE_BOX: tuple[float, float, float] = (0.5, 0.5, 0.6)  # cx, cy, size: one large centred face


class FacePipeline(Protocol):
    def on_faces(
        self, t: float, boxes: list[tuple[float, float, float]], blendshapes: list[dict[str, float]]
    ) -> int | None: ...

    def re_enroll(self) -> None: ...


class FaceSourceSelector:
    """Decides, per moment, whether webcam or iPhone face data is used. Reports switches once."""

    def __init__(self, mode: str, receiver: LiveLinkReceiver | None) -> None:
        if mode not in FACE_SOURCES:
            raise ValueError(f"face.source must be {'|'.join(FACE_SOURCES)}, got {mode!r}")
        if mode != "webcam" and receiver is None:
            raise ValueError(f"face.source={mode} needs an iPhone receiver")
        self.mode = mode
        self.receiver = receiver
        self.current: str | None = None

    def active(self, now: float) -> str:
        if self.mode == "webcam":
            return "webcam"
        if self.mode == "iphone":
            return "iphone"
        assert self.receiver is not None
        return "iphone" if self.receiver.is_fresh(now) else "webcam"

    def update(self, now: float) -> tuple[str, bool]:
        """(active source, switched since the last call)."""
        src = self.active(now)
        switched = self.current is not None and src != self.current
        self.current = src
        return src, switched

    def use_webcam(self, now: float) -> bool:
        return self.active(now) == "webcam"

    @property
    def needs_webcam_model(self) -> bool:
        return self.mode in ("webcam", "auto")

    @property
    def needs_receiver(self) -> bool:
        return self.mode in ("iphone", "auto")


class IPhoneFaceFeed:
    """Pushes iPhone frames into the face pipeline at up to ``max_rate`` Hz.

    Call :meth:`step` often (e.g. after every received packet). While the iPhone is the active
    source it feeds one face per step, or "no face" when the phone stops seeing one or the stream
    goes stale (so the speech gate knows nobody is at the screen). A switch between webcam and
    iPhone re-enrolls the primary face, since face ids do not carry across sources.
    """

    def __init__(
        self,
        receiver: LiveLinkReceiver,
        pipeline: FacePipeline,
        selector: FaceSourceSelector,
        max_rate: float = 30.0,
        swap_sides: bool = False,
        lock: threading.Lock | None = None,
        on_switch: Callable[[str], object] | None = None,
    ) -> None:
        self.receiver = receiver
        self.pipeline = pipeline
        self.selector = selector
        self.min_dt = 1.0 / max(max_rate, 1e-6)
        self.swap_sides = swap_sides
        self.lock = lock  # shared with the webcam face path so the two never call the pipeline at once
        self.on_switch = on_switch
        self._last_feed = -1e18
        self.fed = 0
        self.fed_empty = 0
        self.switches = 0

    def step(self, now: float) -> bool:
        """Returns True if a frame (face or empty) was passed to the pipeline."""
        src, switched = self.selector.update(now)
        if switched:
            self.switches += 1
            self.pipeline.re_enroll()
            if self.on_switch is not None:
                self.on_switch(src)
        if src != "iphone" or now - self._last_feed < self.min_dt:
            return False
        self._last_feed = now
        face = self.receiver.latest_face(now)
        boxes, shapes = ([], []) if face is None else ([IPHONE_BOX], [to_mediapipe(face[1], self.swap_sides)])
        if self.lock is not None:
            with self.lock:
                self.pipeline.on_faces(now, boxes, shapes)
        else:
            self.pipeline.on_faces(now, boxes, shapes)
        if face is None:
            self.fed_empty += 1
        else:
            self.fed += 1
        return True
