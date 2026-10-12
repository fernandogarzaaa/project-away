"""Automatic hand roles: which visible hand is the *pointer* and which is the *helper*. Pure logic, no devices.

The pointer drives the cursor, clicks and every one-hand gesture; the helper only feeds two-hand features
(palm-glide scroll, :mod:`away.hands.palmscroll`). This replaces the left/right bias of ``hands.prefer``.

Identity, not labels
--------------------
MediaPipe's handedness label flips (a hand seen side-on, motion blur, two hands crossing), so hands are
tracked across frames by **where they are and how big they are**: every frame the detections are matched to
the existing tracks by palm-centre distance (in palm sizes) plus a size-ratio term, with an exhaustive
(Hungarian-equivalent for two or three hands) assignment and a gate. The label is only a weak prior: each
track keeps an exponentially smoothed ``side`` score in [-1, 1] (+1 = the user's real right hand, see
:func:`away.hands.select.real_side`) that adds a small cost when a detection's label disagrees, and that
``mode: right | left`` reads. One flipped label moves the score by ``SIDE_ALPHA`` only, so it never swaps roles.

Roles (``hands.roles.mode``)
----------------------------
* ``auto`` (default): the pointer is the first hand that shows the pointing pose (index finger out), or the
  only hand in view. The other hand is the helper. A hand that got the role only as the lone hand and has
  only held a flat palm since (the user raised the "trackpad" hand first) hands it to the first hand that
  points; once the pointer has shown any other pose it is confirmed.
* ``right`` / ``left``: the pointer is the track whose smoothed side matches; the other hand is the helper.
* ``any``: the first hand seen, whatever its pose (the v0.1.9 ``any`` lock).

The pointer role is sticky: it stays on its track while that track is visible, and while it is lost for up to
``release_s`` (a dropout, the fingertip hidden behind the other hand). It is released early when the track was
last seen at the image border (the hand left the frame). A lost track that reappears is re-attached when it is
the only unexplained hand.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from away.hands.landmarks import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, RING_MCP, WRIST, as_array, classify_pose
from away.hands.select import HandObs, real_side

RELEASE_S = 1.0
GATE = 2.5  # a detection may move this many palm sizes from its track between two frames
REACQUIRE_GATE = 6.0  # palm sizes: a lost track may be picked up again this far from where it was last seen
SIZE_WEIGHT = 1.0  # cost per unit of |log(size ratio)|
SIDE_WEIGHT = 0.3  # cost when a detection's label disagrees with the track's smoothed side (weak prior)
SIDE_ALPHA = 0.15  # EMA step of the side score per frame
SIDE_KEEP = 0.5  # right/left mode: the pointer keeps its role until its smoothed side drops below -SIDE_KEEP
EDGE_MARGIN = 0.04  # normalised image units: a track last seen this close to the border left the frame...
EDGE_RELEASE_S = 0.15  # ...once it has been gone this long
POINTER_POSES = ("point",)
FLAT_POSES = ("palm",)  # a pointer that has only shown these got the role as the only hand: not yet confirmed
MODES = ("auto", "right", "left", "any")


def palm_center(lm: Any) -> tuple[float, float]:
    a = np.asarray(lm, dtype=np.float64)
    pts = a[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2]
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1])


def hand_size(lm: Any) -> float:
    """Palm size in image units, robust to the hand turning: max(wrist->middle MCP, index->pinky MCP / 0.8)."""
    a = np.asarray(lm, dtype=np.float64)
    length = float(np.linalg.norm(a[MIDDLE_MCP, :2] - a[WRIST, :2]))
    width = float(np.linalg.norm(a[INDEX_MCP, :2] - a[PINKY_MCP, :2])) / 0.8
    return max(length, width, 1e-4)


def normalize_mode(mode: str | None) -> str:
    """Mode from config / voice / the old ``hands.prefer`` value (``any`` given as a prefer value means auto)."""
    m = str(mode or "auto").strip().lower()
    return m if m in MODES else "auto"


@dataclass
class Track:
    id: int
    obs: HandObs
    center: tuple[float, float]
    size: float
    side: float  # smoothed real side, +1 = right .. -1 = left
    first_seen: float
    last_seen: float
    visible: bool = True

    @property
    def real(self) -> str:
        return "right" if self.side >= 0 else "left"


@dataclass
class RoleResult:
    pointer: HandObs | None = None
    helper: HandObs | None = None
    pointer_id: int | None = None
    helper_id: int | None = None
    pointer_side: str = ""  # the pointer's real hand (smoothed), "" with no pointer
    promoted: bool = False  # the pointer this frame was the helper until now (another hand took over)
    tracks: list[int] = field(default_factory=list)


class HandRoles:
    def __init__(
        self,
        mode: str | Callable[[], str] = "auto",
        mirrored: bool = True,
        release_s: float | Callable[[], float] = RELEASE_S,
        pose_fn: Callable[[Any], str] | None = None,
        max_tracks: int = 2,
    ) -> None:
        self._mode = mode
        self.mirrored = mirrored
        self._release = release_s
        self.pose_fn = pose_fn or (lambda lm: classify_pose(as_array(lm)))
        self.max_tracks = max(1, int(max_tracks))
        self.reset()

    # ------------------------------------------------------------------ config
    @property
    def mode(self) -> str:
        return normalize_mode(self._mode() if callable(self._mode) else self._mode)

    @mode.setter
    def mode(self, value: str | Callable[[], str]) -> None:
        self._mode = value
        self.reset_roles()

    @property
    def release_s(self) -> float:
        r = self._release() if callable(self._release) else self._release
        return max(0.0, float(r))

    def reset(self) -> None:
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.reset_roles()

    def reset_roles(self) -> None:
        self.pointer_id: int | None = None
        self.confirmed = False  # auto: the pointer has pointed (else it only got the role as the only hand)
        self._last_mode: str | None = None

    # ------------------------------------------------------------------ helpers
    def _side_of(self, h: HandObs) -> float:
        return 1.0 if real_side(h.label, self.mirrored) == "right" else -1.0

    def wants_more(self, hands: list[HandObs]) -> bool:
        """Hands are visible but none can be the pointer (``right``/``left`` with only the other hand): the
        tracker should look for a second hand more often."""
        m = self.mode
        if m not in ("right", "left") or not hands:
            return False
        want = 1.0 if m == "right" else -1.0
        vis = [t for t in self.tracks.values() if t.visible]
        return bool(vis) and all(t.side * want <= 0 for t in vis)

    def _cost(self, tr: Track, h: HandObs, gate: float) -> float:
        c = palm_center(h.landmarks)
        s = hand_size(h.landmarks)
        ref = max(tr.size, s)
        d = math.dist(c, tr.center) / ref
        if d > gate:
            return math.inf
        cost = d + SIZE_WEIGHT * abs(math.log(s / tr.size))
        if self._side_of(h) * tr.side < 0:
            cost += SIDE_WEIGHT * min(1.0, abs(tr.side))
        return cost

    def _assign(self, track_ids: list[int], hands: list[HandObs], gate: float) -> dict[int, int]:
        """Minimum-cost assignment track -> detection index (exhaustive: at most a handful of hands)."""
        if not track_ids or not hands:
            return {}
        costs = {(ti, hi): self._cost(self.tracks[ti], hands[hi], gate) for ti in track_ids for hi in range(len(hands))}
        best: dict[int, int] = {}
        best_cost = math.inf
        best_n = 0
        k = min(len(track_ids), len(hands))
        for n in range(k, 0, -1):
            for tsub in itertools.combinations(track_ids, n):
                for hperm in itertools.permutations(range(len(hands)), n):
                    total = sum(costs[(ti, hi)] for ti, hi in zip(tsub, hperm))
                    if total < best_cost:
                        best_cost, best, best_n = total, dict(zip(tsub, hperm)), n
            if best_n:  # prefer matching as many tracks as possible
                break
        return best

    def _update_track(self, tr: Track, h: HandObs, t: float) -> None:
        tr.obs = h
        tr.center = palm_center(h.landmarks)
        tr.size = hand_size(h.landmarks)
        w = SIDE_ALPHA * max(0.2, min(1.0, float(h.score)))
        tr.side = (1 - w) * tr.side + w * self._side_of(h)
        tr.last_seen = t
        tr.visible = True

    @staticmethod
    def _at_edge(c: tuple[float, float]) -> bool:
        return min(c[0], c[1], 1.0 - c[0], 1.0 - c[1]) < EDGE_MARGIN

    def _expired(self, tr: Track, t: float) -> bool:
        gone = t - tr.last_seen
        if gone > self.release_s:
            return True
        return self._at_edge(tr.center) and gone > EDGE_RELEASE_S

    # ------------------------------------------------------------------ step
    def update(self, t: float, hands: list[HandObs]) -> RoleResult:
        mode = self.mode
        if mode != self._last_mode:  # Settings / voice changed the mode: pick the pointer again
            self.pointer_id, self.confirmed = None, False
            self._last_mode = mode
        prev_pointer = self.pointer_id
        prev_ids = set(self.tracks)

        # 1. match the visible tracks, then the lost ones (wider gate) to this frame's detections
        visible_ids = [i for i, tr in self.tracks.items() if tr.visible]
        match = self._assign(visible_ids, hands, GATE)
        left = [i for i in range(len(hands)) if i not in match.values()]
        lost_ids = [i for i, tr in self.tracks.items() if i not in match and not self._expired(tr, t)]
        if left and lost_ids:
            sub = [hands[i] for i in left]
            again = self._assign(lost_ids, sub, REACQUIRE_GATE)
            for ti, k in again.items():
                match[ti] = left[k]
            left = [i for i in range(len(hands)) if i not in match.values()]
            # the only unexplained hand while the pointer is lost: it is the pointer coming back (fast move)
            pid = self.pointer_id
            if left and len(left) == 1 and pid is not None and pid in lost_ids and pid not in match:
                others_matched = all(ti in match for ti in self.tracks if ti != pid)
                if others_matched:
                    match[pid] = left[0]
                    left = []
        for ti, hi in match.items():
            self._update_track(self.tracks[ti], hands[hi], t)
        for ti, tr in self.tracks.items():
            if ti not in match:
                tr.visible = False
        # 2. forget expired tracks; start new ones for unmatched detections (best score first)
        for ti in [i for i, tr in self.tracks.items() if not tr.visible and self._expired(tr, t)]:
            del self.tracks[ti]
            if ti == self.pointer_id:
                self.pointer_id, self.confirmed = None, False
        for hi in sorted(left, key=lambda i: -hands[i].score):
            if len(self.tracks) >= self.max_tracks:
                # room only by dropping a lost (not visible) non-pointer track
                stale = [i for i, tr in self.tracks.items() if not tr.visible and i != self.pointer_id]
                if not stale:
                    break
                del self.tracks[min(stale, key=lambda i: self.tracks[i].last_seen)]
            h = hands[hi]
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(tid, h, palm_center(h.landmarks), hand_size(h.landmarks), self._side_of(h), t, t)

        # 3. roles
        if self.pointer_id is not None and self.pointer_id not in self.tracks:
            self.pointer_id, self.confirmed = None, False
        self._choose_pointer(mode)
        pid = self.pointer_id
        res = RoleResult(tracks=sorted(self.tracks))
        ptr = self.tracks.get(pid) if pid is not None else None
        if ptr is not None and ptr.visible:
            res.pointer, res.pointer_id, res.pointer_side = ptr.obs, ptr.id, ptr.real
            res.promoted = pid != prev_pointer and pid in prev_ids
        helpers = [tr for tr in self.tracks.values() if tr.visible and tr.id != pid]
        if helpers:
            hp = max(helpers, key=lambda tr: (tr.obs.score, -tr.first_seen))
            res.helper, res.helper_id = hp.obs, hp.id
        return res

    def _choose_pointer(self, mode: str) -> None:
        tracks = self.tracks
        pid = self.pointer_id
        if mode in ("right", "left"):
            want = 1.0 if mode == "right" else -1.0
            if pid is not None and tracks[pid].side * want > -SIDE_KEEP:
                return  # hysteresis: a few flipped labels do not hand the role over
            cands = [tr for tr in tracks.values() if tr.visible and tr.side * want > 0]
            self.pointer_id = max(cands, key=lambda tr: tr.side * want).id if cands else None
            return
        vis = [tr for tr in tracks.values() if tr.visible]
        if mode == "any":
            if pid is None and vis:
                self.pointer_id = min(vis, key=lambda tr: (tr.first_seen, -tr.obs.score)).id
            return
        if pid is not None:
            ptr = tracks[pid]
            if not self.confirmed and ptr.visible and self._pose(ptr) not in FLAT_POSES:
                self.confirmed = True  # it did something other than hold a flat palm: it is the pointer
            if self.confirmed:
                return  # sticky until released
            # the only-hand pointer has only held a flat palm so far: a hand that points takes over (the user
            # raised the "trackpad" hand first)
            pointing = [tr for tr in vis if tr.id != pid and self._pose(tr) in POINTER_POSES]
            if pointing:
                self.pointer_id = min(pointing, key=lambda tr: tr.first_seen).id
                self.confirmed = True
            return
        if not vis:
            return
        pointing = [tr for tr in vis if self._pose(tr) in POINTER_POSES]
        if pointing:
            self.pointer_id = min(pointing, key=lambda tr: tr.first_seen).id
            self.confirmed = True
        elif len(vis) == 1 and len(tracks) == 1:
            self.pointer_id = vis[0].id  # the only hand in view
            self.confirmed = self._pose(vis[0]) not in FLAT_POSES

    def _pose(self, tr: Track) -> str:
        try:
            return str(self.pose_fn(tr.obs.landmarks))
        except Exception:
            return "other"
