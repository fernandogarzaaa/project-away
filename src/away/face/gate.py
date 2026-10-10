"""Gate dictation so only the enrolled (primary) user's speech is typed.

"Enrolled" here means the tracked primary face (the largest face when you
enroll, re-acquired by the tracker). There is no face-recognition model in
the MVP; an identity check can be added behind :class:`IdentityCheck`.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol


class IdentityCheck(Protocol):
    def is_user(self, face_id: int) -> bool: ...


class SpeechGate:
    def __init__(self, allow_without_face: bool = True, min_fraction: float = 0.4, horizon: int = 600) -> None:
        self.primary: int | None = None
        self.allow_without_face = allow_without_face
        self.min_fraction = min_fraction
        self._log: deque[tuple[float, int | None, int]] = deque(maxlen=horizon)  # (t, active, n_faces)
        self.identity: IdentityCheck | None = None

    def enroll(self, face_sizes: dict[int, float]) -> int | None:
        """Pick the largest visible face as the primary user."""
        self.primary = max(face_sizes, key=lambda k: face_sizes[k]) if face_sizes else None
        return self.primary

    def observe(self, t: float, active_speaker: int | None, n_faces: int) -> None:
        self._log.append((t, active_speaker, n_faces))

    def allow(self, t0: float, t1: float) -> bool:
        window = [(a, n) for t, a, n in self._log if t0 <= t <= t1]
        if not window or all(n == 0 for _, n in window):
            return self.allow_without_face
        if self.primary is None:
            return True  # not enrolled: gate is a no-op
        if self.identity is not None and not self.identity.is_user(self.primary):
            return False
        hits = sum(1 for a, _ in window if a == self.primary)
        others = sum(1 for a, _ in window if a is not None and a != self.primary)
        if others > hits:
            return False
        return hits / len(window) >= self.min_fraction
