"""A short, soft sound when an agent starts listening (v0.1.26, ``voice.listen_sound``).

"listen" (two rising notes, ~0.14 s): it heard "Hey <name>" and waits for your words. "heard" (one note, ~0.08 s):
your message was taken and the answer is on its way, so the wait for the first word does not feel like nothing
happened. Generated in memory (no files), played without blocking; never played while the computer is speaking.
Too short to count as speech for the microphone (shorter than ``voice.min_utterance_ms``).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

SR = 24000
NOTES = {"listen": ((660.0, 0.06), (880.0, 0.08)), "heard": ((784.0, 0.08),)}
VOLUME = 0.18


def tone(kind: str = "listen", sr: int = SR, volume: float = VOLUME) -> np.ndarray:
    parts = []
    for freq, dur in NOTES.get(kind, NOTES["listen"]):
        n = int(sr * dur)
        t = np.arange(n, dtype=np.float32) / sr
        env = np.minimum(1.0, np.minimum(t, dur - t) / 0.012).astype(np.float32)  # 12 ms fades: no click
        parts.append(np.sin(2 * np.pi * freq * t).astype(np.float32) * env * float(volume))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _sd_play(samples: Any, sr: int) -> None:  # pragma: no cover - audio device
    import sounddevice as sd

    sd.play(samples, sr)  # returns at once


class Earcon:
    """``Earcon(enabled, speaking)(kind)``: plays ``kind`` unless turned off or the computer is speaking."""

    def __init__(
        self,
        enabled: Callable[[], bool] = lambda: True,
        speaking: Callable[[], bool] = lambda: False,
        play: Callable[[Any, int], None] | None = None,
    ) -> None:
        self.enabled = enabled
        self.speaking = speaking
        self.play = play or _sd_play
        self.played: list[str] = []
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, kind: str = "listen") -> bool:
        try:
            if not self.enabled() or self.speaking():
                return False
        except Exception:
            return False
        if kind not in self._cache:
            self._cache[kind] = tone(kind)
        samples = self._cache[kind]
        self.played = [*self.played[-9:], kind]

        def run() -> None:
            with contextlib.suppress(Exception):
                self.play(samples, SR)

        threading.Thread(target=run, daemon=True, name="away-earcon").start()
        return True
