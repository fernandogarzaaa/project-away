"""Facial command gestures (optional): a held brow raise toggles dictation."""

from __future__ import annotations


def brow_raise(blendshapes: dict[str, float]) -> float:
    inner = float(blendshapes.get("browInnerUp", 0.0))
    outer = (float(blendshapes.get("browOuterUpLeft", 0.0)) + float(blendshapes.get("browOuterUpRight", 0.0))) / 2
    return max(inner, outer)


class BrowRaiseToggle:
    """Fires once when the brows stay above ``threshold`` for ``hold`` seconds; re-arms below ``threshold*0.6``."""

    def __init__(self, threshold: float = 0.55, hold: float = 0.6, cooldown: float = 1.0) -> None:
        self.threshold = threshold
        self.release = threshold * 0.6
        self.hold = hold
        self.cooldown = cooldown
        self._t_up: float | None = None
        self._armed = True
        self._last_fire = -1e9

    def update(self, t: float, blendshapes: dict[str, float]) -> bool:
        v = brow_raise(blendshapes)
        if v < self.release:
            self._armed = True
            self._t_up = None
            return False
        if v < self.threshold:
            self._t_up = None
            return False
        if self._t_up is None:
            self._t_up = t
        if self._armed and t - self._t_up >= self.hold and t - self._last_fire >= self.cooldown:
            self._armed = False
            self._last_fire = t
            return True
        return False


def mouth_open(blendshapes: dict[str, float]) -> float:
    return float(blendshapes.get("jawOpen", 0.0))


class MouthOpenTrigger(BrowRaiseToggle):
    """Same hold/re-arm/cooldown logic as the brow raise, on the ``jawOpen`` blendshape.

    Use a long ``hold`` (default 0.8 s) so talking does not fire it.
    """

    def __init__(self, threshold: float = 0.5, hold: float = 0.8, cooldown: float = 1.0) -> None:
        super().__init__(threshold, hold, cooldown)

    def update(self, t: float, blendshapes: dict[str, float]) -> bool:
        return super().update(t, {"browInnerUp": mouth_open(blendshapes)})
