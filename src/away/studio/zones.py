"""Zone maps: where in the camera picture gestures may fire.

A zone map is a grid (default 6 x 4, boundaries draggable) whose cells are ``active`` or ``ignore`` ("Ignore
here" in the window), free-form rectangles drawn on top (later ones win), and an optional activation line (the
hand must be on one side of it). Coordinates are normalised (0..1) in the MIRRORED image, i.e. what the user
sees in the preview, whatever ``hands.mirror`` says.

ONE map applies to every gesture. By default it gates activations only (clicks, swipes, custom gestures...);
``zones.block_cursor`` also stops the cursor. :class:`ZoneGate` adds hysteresis so a hand resting on a boundary
does not flicker between the two sides.
"""

from __future__ import annotations

import bisect
import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from away.hands.landmarks import INDEX_TIP, WRIST, as_array

ACTIVE, IGNORE = "active", "ignore"
CELL_MODES = (ACTIVE, IGNORE)
DEFAULT_COLS, DEFAULT_ROWS = 6, 4
MAX_CELLS_PER_AXIS = 24
MIN_EDGE_GAP = 0.02  # two grid boundaries never come closer than this
PALM_POINTS = (0, 5, 9, 13, 17)
ANY_CAMERA = "*"
DEFAULT_PROFILE = "default"


@dataclass
class ZoneRect:
    x0: float
    y0: float
    x1: float
    y1: float
    mode: str = IGNORE

    def contains(self, x: float, y: float) -> bool:
        return min(self.x0, self.x1) <= x <= max(self.x0, self.x1) and min(self.y0, self.y1) <= y <= max(
            self.y0, self.y1
        )


@dataclass
class ActivationLine:
    """Gestures count only on one side of the line a -> b. ``keep`` = -1 keeps the side where
    ``cross(b - a, p - a) < 0``; for a left-to-right horizontal line that is ABOVE it (image y grows down)."""

    x0: float = 0.0
    y0: float = 0.7
    x1: float = 1.0
    y1: float = 0.7
    keep: int = -1

    def side(self, x: float, y: float) -> float:
        return (self.x1 - self.x0) * (y - self.y0) - (self.y1 - self.y0) * (x - self.x0)

    def keeps(self, x: float, y: float) -> bool:
        s = self.side(x, y)
        return s * self.keep > 0 or s == 0


def even_edges(n: int) -> list[float]:
    return [i / n for i in range(n + 1)]


@dataclass
class ZoneMap:
    xs: list[float] = field(default_factory=lambda: even_edges(DEFAULT_COLS))
    ys: list[float] = field(default_factory=lambda: even_edges(DEFAULT_ROWS))
    cells: list[list[str]] = field(default_factory=lambda: [[ACTIVE] * DEFAULT_COLS for _ in range(DEFAULT_ROWS)])
    rects: list[ZoneRect] = field(default_factory=list)
    line: ActivationLine | None = None

    @property
    def cols(self) -> int:
        return len(self.xs) - 1

    @property
    def rows(self) -> int:
        return len(self.ys) - 1

    # ------------------------------------------------------------------ queries
    def cell_at(self, x: float, y: float) -> tuple[int, int]:
        """``(row, col)`` of the cell holding the point (clamped into the picture)."""
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        c = min(max(bisect.bisect_right(self.xs, x) - 1, 0), self.cols - 1)
        r = min(max(bisect.bisect_right(self.ys, y) - 1, 0), self.rows - 1)
        return r, c

    def allowed(self, x: float, y: float) -> bool:
        """Raw answer (no hysteresis): may a gesture fire with the hand at ``(x, y)``?"""
        r, c = self.cell_at(x, y)
        ok = self.cells[r][c] == ACTIVE
        for rect in self.rects:
            if rect.contains(x, y):
                ok = rect.mode == ACTIVE
        if self.line is not None and not self.line.keeps(x, y):
            ok = False
        return ok

    def is_open(self) -> bool:
        """True when the map allows everything (nothing to enforce)."""
        return (
            all(m == ACTIVE for row in self.cells for m in row)
            and all(r.mode == ACTIVE for r in self.rects)
            and self.line is None
        )

    def active_fraction(self, n: int = 40) -> float:
        """Share of the picture where gestures may fire (sampled on an n x n grid), for the summary text."""
        pts = (np.arange(n) + 0.5) / n
        return float(np.mean([[self.allowed(float(x), float(y)) for x in pts] for y in pts]))

    # ------------------------------------------------------------------ edits
    def toggle_cell(self, r: int, c: int) -> str:
        self.cells[r][c] = IGNORE if self.cells[r][c] == ACTIVE else ACTIVE
        return self.cells[r][c]

    def set_cell(self, r: int, c: int, mode: str) -> None:
        if mode not in CELL_MODES:
            raise ValueError(f"cell mode must be {' or '.join(CELL_MODES)}, got {mode!r}")
        self.cells[r][c] = mode

    def set_all(self, mode: str) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                self.set_cell(r, c, mode)

    def resize(self, cols: int, rows: int) -> None:
        """New even grid. A cell keeps the mode of the old cell under its centre."""
        if not (1 <= cols <= MAX_CELLS_PER_AXIS and 1 <= rows <= MAX_CELLS_PER_AXIS):
            raise ValueError(f"the grid must be 1..{MAX_CELLS_PER_AXIS} cells each way")
        old = copy.deepcopy(self)
        self.xs, self.ys = even_edges(cols), even_edges(rows)
        self.cells = []
        for r in range(rows):
            row = []
            for c in range(cols):
                cx = (self.xs[c] + self.xs[c + 1]) / 2
                cy = (self.ys[r] + self.ys[r + 1]) / 2
                orow, ocol = old.cell_at(cx, cy)
                row.append(old.cells[orow][ocol])
            self.cells.append(row)

    def nearest_edge(self, axis: str, value: float, tol: float) -> int | None:
        """Index of the INTERIOR boundary within ``tol`` of ``value`` (``axis`` 'x' or 'y'), for dragging."""
        edges = self.xs if axis == "x" else self.ys
        best, best_d = None, tol
        for i in range(1, len(edges) - 1):
            d = abs(edges[i] - value)
            if d <= best_d:
                best, best_d = i, d
        return best

    def move_edge(self, axis: str, i: int, value: float) -> float:
        """Drag interior boundary ``i`` to ``value``, kept MIN_EDGE_GAP away from its neighbours."""
        edges = self.xs if axis == "x" else self.ys
        if not 0 < i < len(edges) - 1:
            raise ValueError("only the inner grid lines can move")
        lo, hi = edges[i - 1] + MIN_EDGE_GAP, edges[i + 1] - MIN_EDGE_GAP
        edges[i] = float(min(max(value, lo), hi))
        return edges[i]

    # ------------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "grid": {
                "cols": self.cols,
                "rows": self.rows,
                "xs": [round(v, 4) for v in self.xs],
                "ys": [round(v, 4) for v in self.ys],
                "cells": ["".join("a" if m == ACTIVE else "i" for m in row) for row in self.cells],
            },
            "rects": [{"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1, "mode": r.mode} for r in self.rects],
            "line": None,
        }
        if self.line is not None:
            ln = self.line
            d["line"] = {"x0": ln.x0, "y0": ln.y0, "x1": ln.x1, "y1": ln.y1, "keep": ln.keep}
        return d


def _num(v: Any, path: str, lo: float = 0.0, hi: float = 1.0) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"{path}: must be a number, got {v!r}")
    if not lo <= float(v) <= hi:
        raise ValueError(f"{path}: must be between {lo} and {hi}, got {v}")
    return float(v)


def _edges(v: Any, n: int, path: str) -> list[float]:
    if v is None:
        return even_edges(n)
    if not isinstance(v, list) or len(v) != n + 1:
        raise ValueError(f"{path}: must list {n + 1} boundaries")
    out = [_num(e, f"{path}[{i}]") for i, e in enumerate(v)]
    if out[0] != 0.0 or out[-1] != 1.0 or any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(f"{path}: must start at 0, end at 1 and increase")
    return out


def zone_map_from_dict(d: Any, path: str = "zone map") -> ZoneMap:
    """Parse (and validate) what :meth:`ZoneMap.to_dict` wrote. Raises ValueError naming the bad field."""
    if not isinstance(d, dict):
        raise ValueError(f"{path}: must be a mapping")
    g = d.get("grid") or {}
    if not isinstance(g, dict):
        raise ValueError(f"{path}.grid: must be a mapping")
    cols, rows = g.get("cols", DEFAULT_COLS), g.get("rows", DEFAULT_ROWS)
    for name, v in (("cols", cols), ("rows", rows)):
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_CELLS_PER_AXIS:
            raise ValueError(f"{path}.grid.{name}: must be a whole number 1..{MAX_CELLS_PER_AXIS}")
    zm = ZoneMap(_edges(g.get("xs"), cols, f"{path}.grid.xs"), _edges(g.get("ys"), rows, f"{path}.grid.ys"))
    raw = g.get("cells")
    if raw is None:
        zm.cells = [[ACTIVE] * cols for _ in range(rows)]
    else:
        if not isinstance(raw, list) or len(raw) != rows:
            raise ValueError(f"{path}.grid.cells: must have {rows} rows")
        zm.cells = []
        for i, row in enumerate(raw):
            if not isinstance(row, str) or len(row) != cols or set(row) - {"a", "i"}:
                raise ValueError(f"{path}.grid.cells[{i}]: must be {cols} letters, a (active) or i (ignore)")
            zm.cells.append([ACTIVE if ch == "a" else IGNORE for ch in row])
    rects = d.get("rects") or []
    if not isinstance(rects, list):
        raise ValueError(f"{path}.rects: must be a list")
    for i, r in enumerate(rects):
        rp = f"{path}.rects[{i}]"
        if not isinstance(r, dict):
            raise ValueError(f"{rp}: must be a mapping")
        mode = r.get("mode", IGNORE)
        if mode not in CELL_MODES:
            raise ValueError(f"{rp}.mode: must be active or ignore")
        x0, y0, x1, y1 = (_num(r.get(k), f"{rp}.{k}") for k in ("x0", "y0", "x1", "y1"))
        zm.rects.append(ZoneRect(x0, y0, x1, y1, mode=mode))
    ln = d.get("line")
    if ln is not None:
        if not isinstance(ln, dict):
            raise ValueError(f"{path}.line: must be a mapping or null")
        keep = ln.get("keep", -1)
        if isinstance(keep, bool) or keep not in (-1, 1):
            raise ValueError(f"{path}.line.keep: must be -1 or 1")
        lx0, ly0, lx1, ly1 = (_num(ln.get(k), f"{path}.line.{k}") for k in ("x0", "y0", "x1", "y1"))
        if lx0 == lx1 and ly0 == ly1:
            raise ValueError(f"{path}.line: the two ends must differ")
        zm.line = ActivationLine(lx0, ly0, lx1, ly1, keep=int(keep))
    return zm


# ---------------------------------------------------------------------- per camera / per profile
def camera_key(cfg: Any) -> str:
    """The camera a zone map belongs to: the primary camera's name/index (what Settings saved)."""
    hands = cfg.hands
    prim = next((c for c in hands.cameras if c.role == "primary"), None)
    return str(prim.device if prim is not None else hands.camera_index)


def lookup_map(maps: dict[str, Any], camera: str, profile: str) -> tuple[ZoneMap | None, tuple[str, str] | None]:
    """The zone map for ``camera`` + ``profile`` and where it came from. Fallbacks, in order: this camera +
    this profile, this camera + default, any camera ("*") + this profile, any camera + default."""
    for cam in (camera, ANY_CAMERA):
        per = maps.get(cam)
        if not isinstance(per, dict):
            continue
        for prof in (profile, DEFAULT_PROFILE):
            if prof in per:
                return zone_map_from_dict(per[prof], f"zones.maps.{cam}.{prof}"), (cam, prof)
    return None, None


def store_map(maps: dict[str, Any], camera: str, profile: str, zm: ZoneMap | None) -> None:
    """Save (or with ``None`` remove) the map for ``camera`` + ``profile``."""
    per = maps.setdefault(camera, {})
    if zm is None:
        per.pop(profile, None)
        if not per:
            maps.pop(camera, None)
    else:
        per[profile] = zm.to_dict()


def validate_zones(z: Any) -> None:
    from away.studio.settings import ZONE_ANCHORS

    if not isinstance(z.enabled, bool) or not isinstance(z.block_cursor, bool):
        raise ValueError("zones.enabled and zones.block_cursor must be true or false")
    _num(z.hysteresis, "zones.hysteresis", 0.0, 0.2)
    if z.anchor not in ZONE_ANCHORS:
        raise ValueError(f"zones.anchor must be {'|'.join(ZONE_ANCHORS)}, got {z.anchor!r}")
    if not isinstance(z.maps, dict):
        raise ValueError("zones.maps must be a mapping of camera -> profile -> zone map")
    for cam, per in z.maps.items():
        if not isinstance(per, dict):
            raise ValueError(f"zones.maps.{cam}: must be a mapping of profile -> zone map")
        for prof, d in per.items():
            zone_map_from_dict(d, f"zones.maps.{cam}.{prof}")


# ---------------------------------------------------------------------- the hand's point + hysteresis
def anchor_point(landmarks: Any, anchor: str = "palm", mirrored: bool = True) -> tuple[float, float]:
    """The hand's point in mirrored image space (landmarks of an un-mirrored camera are flipped)."""
    lm = as_array(landmarks)
    if anchor == "index_tip":
        x, y = float(lm[INDEX_TIP, 0]), float(lm[INDEX_TIP, 1])
    elif anchor == "wrist":
        x, y = float(lm[WRIST, 0]), float(lm[WRIST, 1])
    else:
        p = lm[list(PALM_POINTS), :2].mean(axis=0)
        x, y = float(p[0]), float(p[1])
    return (x if mirrored else 1.0 - x), y


_PROBES = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]


class ZoneGate:
    """Stateful zone answer with a hysteresis band: the side only changes once the point is at least
    ``hysteresis`` away from every boundary (all probes on a circle of that radius agree with the new side)."""

    def __init__(self, zone_map: ZoneMap | None, hysteresis: float = 0.03) -> None:
        self.map = zone_map
        self.h = float(hysteresis)
        self.state: bool | None = None
        self.point: tuple[float, float] | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, point: tuple[float, float] | None) -> bool:
        """True = gestures may fire. No map (or an open one) always allows; no hand resets the state."""
        self.point = point
        if self.map is None or point is None:
            self.state = None
            return True
        x, y = point
        raw = self.map.allowed(x, y)
        if self.state is None or self.h <= 0:
            self.state = raw
        elif raw != self.state and all(
            self.map.allowed(x + self.h * dx, y + self.h * dy) == raw for dx, dy in _PROBES
        ):
            self.state = raw
        return self.state
