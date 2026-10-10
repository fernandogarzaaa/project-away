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


class WinkTrigger:
    """One eye closed (``eyeBlinkLeft``/``eyeBlinkRight`` >= ``threshold``) while the other stays open
    (<= ``open_max``) for ``hold`` seconds. A normal blink closes both eyes, so it never fires.

    Sides are as MediaPipe reports them for the (mirrored) camera image. Re-arms when the eye opens again.
    """

    def __init__(
        self, threshold: float = 0.5, open_max: float = 0.3, hold: float = 0.25, cooldown: float = 0.8
    ) -> None:
        self.threshold = threshold
        self.open_max = open_max
        self.hold = hold
        self.cooldown = cooldown
        self._t0: dict[str, float | None] = {"left": None, "right": None}
        self._armed = {"left": True, "right": True}
        self._last = -1e9

    def update(self, t: float, blendshapes: dict[str, float]) -> str | None:
        """Returns ``"wink_left"`` / ``"wink_right"`` when one fires, else None."""
        lv = float(blendshapes.get("eyeBlinkLeft", 0.0))
        rv = float(blendshapes.get("eyeBlinkRight", 0.0))
        fired = None
        for side, closed, other in (("left", lv, rv), ("right", rv, lv)):
            if closed < self.threshold * 0.6:
                self._armed[side] = True
            if closed >= self.threshold and other <= self.open_max:
                if self._t0[side] is None:
                    self._t0[side] = t
                t0 = self._t0[side]
                if (
                    fired is None
                    and self._armed[side]
                    and t0 is not None
                    and t - t0 >= self.hold
                    and t - self._last >= self.cooldown
                ):
                    self._armed[side] = False
                    self._last = t
                    fired = f"wink_{side}"
            else:
                self._t0[side] = None
        return fired
