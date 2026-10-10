"""Coarse Wi-Fi sensing classifier: empty / presence / motion / push / pull / swipe.

Threshold rules over :func:`away.wifi.features.window_features`, with a
calibration step that learns thresholds (and the push/pull sign, which
depends on room geometry) from a few labelled windows. RSSI-only input can
only give empty / presence / motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LABELS = ("empty", "presence", "motion", "push", "pull", "swipe")


@dataclass
class CSIThresholds:
    presence: float = 0.015  # CSI motion energy (synthetic-tuned default; run calibration)
    motion: float = 0.08
    rssi_presence: float = 0.3  # dB mean |diff|
    rssi_motion: float = 1.5
    min_net_rotations: float = 1.0
    directional_ratio: float = 0.6  # |net| / total above which motion is push/pull
    swipe_ratio: float = 0.3  # |net| / total below which motion is a swipe
    min_swipe_rotations: float = 2.0
    direction_sign: float = 1.0  # +1: positive net rotation = push (calibrate per room)


class CoarseCSIClassifier:
    def __init__(self, thresholds: CSIThresholds | None = None) -> None:
        self.th = thresholds or CSIThresholds()

    def classify(self, feats: dict[str, float]) -> tuple[str, float]:
        """Return (label, confidence in [0, 1])."""
        th = self.th
        if "energy" in feats:
            e = feats["energy"]
            if e < th.presence:
                return "empty", _margin(th.presence - e, th.presence)
            if e < th.motion:
                return "presence", _margin(min(e - th.presence, th.motion - e), th.motion - th.presence)
            net = feats.get("rotation_net")
            total = feats.get("rotation_total")
            if net is None or total is None or total <= 0:
                return "motion", _margin(e - th.motion, th.motion)
            frac = abs(net) / total
            if abs(net) >= th.min_net_rotations and frac >= th.directional_ratio:
                label = "push" if net * th.direction_sign > 0 else "pull"
                return label, min(1.0, frac)
            if total >= th.min_swipe_rotations and frac <= th.swipe_ratio:
                return "swipe", min(1.0, 1.0 - frac)
            return "motion", 0.5
        if "rssi_energy" in feats:
            r = feats["rssi_energy"]
            if r < th.rssi_presence:
                return "empty", _margin(th.rssi_presence - r, th.rssi_presence)
            if r < th.rssi_motion:
                return "presence", 0.5
            return "motion", _margin(r - th.rssi_motion, th.rssi_motion)
        return "empty", 0.0

    def calibrate(self, samples: list[tuple[dict[str, float], str]]) -> CSIThresholds:
        """Set thresholds at the geometric midpoint between labelled groups; learn the push sign."""
        by: dict[str, list[dict[str, float]]] = {}
        for f, lab in samples:
            by.setdefault(lab, []).append(f)

        def med(lab: str, key: str) -> float | None:
            vals = [f[key] for f in by.get(lab, []) if key in f]
            return float(np.median(vals)) if vals else None

        moving = [lab for lab in ("motion", "push", "pull", "swipe") if lab in by]
        e_empty, e_pres = med("empty", "energy"), med("presence", "energy")
        e_move = [m for m in (med(lab, "energy") for lab in moving) if m is not None]
        if e_empty is not None and e_pres is not None and e_empty < e_pres:
            self.th.presence = math.sqrt(e_empty * e_pres) if e_empty > 0 else e_pres / 2
        if e_pres is not None and e_move and e_pres < min(e_move):
            self.th.motion = math.sqrt(e_pres * min(e_move))
        push = med("push", "rotation_net")
        if push is not None and push != 0:
            self.th.direction_sign = 1.0 if push > 0 else -1.0
        return self.th


def _margin(dist: float, scale: float) -> float:
    if scale <= 0:
        return 0.5
    return float(min(1.0, 0.5 + 0.5 * dist / scale))
