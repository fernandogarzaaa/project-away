"""Greedy nearest-centroid multi-face tracker with stable integer IDs."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    size: float
    missing: int = 0


class FaceTracker:
    def __init__(self, max_dist: float = 0.15, max_missing: int = 15) -> None:
        self.max_dist = max_dist
        self.max_missing = max_missing
        self.tracks: dict[int, Track] = {}
        self._next = 0

    def update(self, boxes: list[tuple[float, float, float]]) -> list[int]:
        """boxes = [(cx, cy, size)] in normalized coords. Returns one id per box (same order)."""
        ids: list[int | None] = [None] * len(boxes)
        pairs = []
        for bi, (cx, cy, _s) in enumerate(boxes):
            for tid, tr in self.tracks.items():
                d = math.hypot(cx - tr.cx, cy - tr.cy)
                if d <= self.max_dist:
                    pairs.append((d, bi, tid))
        used_t: set[int] = set()
        for _d, bi, tid in sorted(pairs):
            if ids[bi] is None and tid not in used_t:
                ids[bi] = tid
                used_t.add(tid)
        for bi, (cx, cy, s) in enumerate(boxes):
            assigned = ids[bi]
            if assigned is None:
                assigned = self._next
                self._next += 1
                ids[bi] = assigned
            self.tracks[assigned] = Track(assigned, cx, cy, s, 0)
            used_t.add(assigned)
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_missing:
                    del self.tracks[tid]
        return [int(i) for i in ids if i is not None]
