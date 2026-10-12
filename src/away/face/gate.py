"""Gate dictation so only the enrolled (primary) user's speech is typed.

"Enrolled" here means the tracked primary face (the largest face when you
enroll, re-acquired by the tracker). There is no face-recognition model in
the MVP; an identity check can be added behind :class:`IdentityCheck`.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol


class IdentityCheck(Protocol):
    def is_user(self, face_id: int) -> bool: ...


class SpeechGate:
    """Who is talking, from what face tracking saw while an utterance was spoken.

    v0.1.12 (the user's webcam sits off-angle at arm's length: mouth-motion scores of 0.05-0.16 while
    talking, so almost everything was ignored):

    * the mouth-motion threshold is relative to the user's own resting level: the median of their speaking
      score before the utterance plus a few median absolute deviations, between ``min_loose`` and
      ``loose_score`` (``loose_score`` alone when there is no history yet);
    * a lower speaking fraction (``ambient_fraction``) is enough;
    * when the enrolled face is at the screen and seen *frontal* in at least ``presence_fraction`` of the
      utterance's frames, dictation is allowed even with weak mouth motion, unless another face is clearly
      the speaker. Frames whose pose is unknown never count as frontal.
    """

    def __init__(
        self,
        allow_without_face: bool = True,
        min_fraction: float = 0.4,
        horizon: int = 600,
        ambient_fraction: float = 0.12,
        loose_score: float = 0.08,
        pad_s: float = 0.3,
        starved_s: float = 10.0,
        presence_fraction: float = 0.6,
        min_loose: float = 0.03,
        baseline_frames: int = 300,
        baseline_min_frames: int = 15,
        baseline_mads: float = 3.0,
        baseline_margin: float = 0.015,
    ) -> None:
        self.primary: int | None = None
        self.allow_without_face = allow_without_face
        self.min_fraction = min_fraction
        self._log: deque[tuple[float, int | None, int]] = deque(maxlen=horizon)  # (t, active, n_faces)
        self._scores: deque[tuple[float, float | None]] = deque(maxlen=horizon)  # (t, best speaking score)
        self.identity: IdentityCheck | None = None
        # always-on gate tolerances (v0.1.6): a visible face whose mouth moves counts even when the
        # active-speaker match fails (head turned, off-axis webcam on a multi-monitor desk)
        self.ambient_fraction = ambient_fraction
        self.loose_score = loose_score
        self.pad_s = pad_s  # face frames just before/after the utterance count too
        self.starved_s = starved_s  # no face frames in the window (CPU busy): trust the last face seen
        self.last_explain: dict[str, Any] = {}
        self.presence_fraction = presence_fraction
        self.min_loose = min_loose
        self.baseline_frames = baseline_frames
        self.baseline_min_frames = baseline_min_frames
        self.baseline_mads = baseline_mads
        self.baseline_margin = baseline_margin
        # per frame: (the user's face is at the screen, it faces the camera: True/False, None = unknown)
        self._pose: deque[tuple[bool, bool | None]] = deque(maxlen=horizon)

    def enroll(self, face_sizes: dict[int, float]) -> int | None:
        """Pick the largest visible face as the primary user."""
        self.primary = max(face_sizes, key=lambda k: face_sizes[k]) if face_sizes else None
        return self.primary

    def observe(
        self,
        t: float,
        active_speaker: int | None,
        n_faces: int,
        score: float | None = None,
        frontal: bool | None = None,
        user_present: bool | None = None,
    ) -> None:
        """One face-tracking frame: the active speaker (or None), how many faces, the speaking score of the
        user's face (the primary, or the best one when not enrolled), whether that face looks at the camera
        (``frontal``, None = unknown) and whether it is in the frame at all (``user_present``; default: any
        face is)."""
        self._log.append((t, active_speaker, n_faces))
        self._scores.append((t, score))
        present = n_faces > 0 if user_present is None else bool(user_present and n_faces > 0)
        self._pose.append((present, frontal if present else None))

    def mouth_threshold(self, before: float) -> tuple[float, float | None]:
        """``(threshold, baseline)``: the speaking score that counts as a moving mouth, relative to the user's
        resting level in the frames before ``before`` (``baseline`` None when there is too little history)."""
        known = [s for t, s in self._scores if t < before and s is not None][-self.baseline_frames :]
        if len(known) < self.baseline_min_frames:
            return self.loose_score, None
        ordered = sorted(known)
        base = _median(ordered)
        mad = _median(sorted(abs(s - base) for s in known))
        thr = base + max(self.baseline_mads * mad, self.baseline_margin)
        return min(self.loose_score, max(self.min_loose, thr)), base

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

    def allow_ambient(self, t0: float, t1: float) -> bool:
        """Stricter gate for always-on listening: a face must be at the screen *and* be the one talking.

        Unlike :meth:`allow`, no visible face means "not the user" (background talk, TV). See
        :meth:`explain` for the numbers behind the decision.
        """
        return bool(self.explain(t0, t1)["allowed"])

    def explain(self, t0: float, t1: float) -> dict[str, Any]:
        """The always-on gate decision for the utterance ``[t0, t1]`` with its evidence (for the log)."""
        lo, hi = t0 - self.pad_s, t1 + self.pad_s
        idx = [i for i, (t, _a, _n) in enumerate(self._log) if lo <= t <= hi]
        log = list(self._log)
        scores = list(self._scores)
        info: dict[str, Any] = {"frames": len(idx), "enrolled": self.primary is not None}

        def done(allowed: bool, reason: str) -> dict[str, Any]:
            info["allowed"], info["reason"] = allowed, reason
            self.last_explain = info
            return info

        if self.primary is not None and self.identity is not None and not self.identity.is_user(self.primary):
            return done(False, "identity check: not the enrolled user")
        if not idx:
            before = [(t, n) for t, _a, n in log if t <= hi]
            if before and t1 - before[-1][0] <= self.starved_s and before[-1][1] > 0:
                # face tracking delivered nothing during the utterance (the CPU was busy, e.g. with speech
                # recognition) but the user was at the screen just before: give them the benefit of the doubt
                return done(True, "no face frames during speech (CPU busy?); a face was present just before")
            return done(False, "no face frames" if not before else "no face at the screen")
        window = [log[i] for i in idx]
        win_scores = [scores[i][1] if i < len(scores) else None for i in idx]
        poses = list(self._pose)
        win_pose = [poses[i] if i < len(poses) else (False, None) for i in idx]
        faces = sum(1 for _t, _a, n in window if n > 0)
        info["face_fraction"] = round(faces / len(window), 2)
        known = [s for s in win_scores if s is not None]
        info["max_score"] = round(max(known), 2) if known else None
        if faces == 0:
            return done(False, "no face at the screen")
        if self.primary is None:
            hits = sum(1 for _t, a, _n in window if a is not None)
            others = 0
        else:
            hits = sum(1 for _t, a, _n in window if a == self.primary)
            others = sum(1 for _t, a, _n in window if a is not None and a != self.primary)
        thr, base = self.mouth_threshold(lo)
        loose = sum(
            1 for (_t, a, n), sc in zip(window, win_scores) if a is None and n > 0 and sc is not None and sc >= thr
        )
        frontal = sum(1 for present, fr in win_pose if present and fr is True)
        presence = frontal / len(window)
        info.update(speaker_frames=hits, other_speaker_frames=others, mouth_moving_frames=loose)
        info["mouth_threshold"] = round(thr, 3)
        if base is not None:
            info["baseline"] = round(base, 3)
        info["frontal_fraction"] = round(presence, 2)
        if others > hits + loose:
            return done(False, "someone else was speaking")
        frac = (hits + loose) / len(window)
        info["speaking_fraction"] = round(frac, 2)
        if frac >= min(self.ambient_fraction, self.min_fraction):
            return done(True, "facing the screen and speaking")
        if presence >= self.presence_fraction:
            return done(True, "enrolled face at the screen, facing it (mouth motion weak)")
        return done(False, "face present but mouth not moving")


def _median(ordered: list[float]) -> float:
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
