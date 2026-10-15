"""Speaking the live agent's replies with the app's current voice, sentence by sentence, and stopping when you talk.

* Each finished sentence goes to the app's :class:`away.voice.speech.Speaker` right away, so the first sentence is
  heard while the model is still writing the rest.
* **Barge-in**: :class:`BargeIn` watches the microphone level while the agent speaks. Sustained speech clearly
  louder than the agent's own voice coming back through the speakers (measured at the start of each reply) stops it:
  the waiting sentences are dropped at once, and the natural voices (Kokoro / Piper, played through sounddevice)
  stop mid-word. The classic Windows voice (SAPI) finishes the word-group it is saying (it cannot be interrupted
  from another thread safely), then stops.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np


class LiveSpeech:
    def __init__(
        self, speaker: Any, enabled: Callable[[], bool] = lambda: True, stop_audio: Callable[[], None] | None = None
    ) -> None:
        self.speaker = speaker  # away.voice.speech.Speaker (or anything with say(text, force=...))
        self.enabled = enabled
        self.stop_audio = stop_audio or _stop_sounddevice
        self._cancel = threading.Event()
        # Serialize enqueue and interrupt: otherwise barge-in can drain the queue immediately before say()
        # enqueues, leaving a sentence that starts after the user interrupted.
        self._state_lock = threading.RLock()
        self.interrupted = 0
        self.spoken: list[str] = []
        self.on_interrupt: Callable[[], None] | None = None

    def begin(self) -> None:
        with self._state_lock:
            self._cancel.clear()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def say(self, sentence: str) -> bool:
        with self._state_lock:
            if self._cancel.is_set() or not sentence.strip() or self.speaker is None:
                return False
            try:
                if not self.enabled():
                    return False
            except Exception:
                return False
            try:
                res = self.speaker.say(sentence, force=True)
            except TypeError:
                try:
                    res = self.speaker.say(sentence)
                except Exception:
                    return False
            except Exception:
                return False
            accepted = bool(res.get("spoken", True)) if isinstance(res, dict) else True
            if accepted:
                self.spoken = [*self.spoken[-19:], sentence]
            return accepted

    def speaking(self) -> bool:
        # Speaker._idle also covers model loading and queued text; only arm acoustic barge-in while audio
        # is actually being rendered. Keep the old fake/third-party speaker contract as a fallback.
        active = getattr(self.speaker, "_speaking", None)
        if active is not None:
            return bool(active.is_set())
        idle = getattr(self.speaker, "_idle", None)
        return idle is not None and not idle.is_set()

    def interrupt(self) -> None:
        """Atomically stop accepting speech, drain queued sentences, and stop current playback."""
        with self._state_lock:
            self._cancel.set()
            self.interrupted += 1
            q: Any = getattr(self.speaker, "_q", None)
            lock = getattr(self.speaker, "_lock", None)
            if q is not None:
                with lock if lock is not None else contextlib.nullcontext():
                    while True:
                        try:
                            item = q.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:  # preserve a shutdown request
                            q.put(None)
                            break
        with contextlib.suppress(Exception):
            self.stop_audio()
        if self.on_interrupt is not None:
            with contextlib.suppress(Exception):
                self.on_interrupt()


def _stop_sounddevice() -> None:  # pragma: no cover - audio device
    import sys

    sd = sys.modules.get("sounddevice")  # only when a natural voice already loaded it
    if sd is not None:
        sd.stop()


class BargeIn:
    """Decides from microphone chunks (float32, mono) that the person started talking over the agent.

    While the agent speaks, the first ``calib_s`` seconds set the echo level (the agent's own voice coming back
    through the speakers). After that, ``hold_s`` of sound louder than ``ratio`` x that level (and above
    ``floor``) fires ``on_barge``. Nothing happens while the agent is silent."""

    def __init__(
        self,
        speaking: Callable[[], bool],
        on_barge: Callable[[], None],
        sample_rate: int = 16000,
        ratio: float = 2.5,
        floor: float = 0.02,
        hold_s: float = 0.25,
        calib_s: float = 0.35,
        clock: Callable[[], float] = time.monotonic,
        enabled: Callable[[], bool] = lambda: True,
    ) -> None:
        self.speaking = speaking
        self.on_barge = on_barge
        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if hold_s <= 0 or calib_s < 0 or ratio <= 0 or floor < 0:
            raise ValueError("invalid barge-in thresholds")
        self.sr = int(sample_rate)
        self.ratio, self.floor, self.hold_s, self.calib_s = float(ratio), float(floor), float(hold_s), float(calib_s)
        self.clock = clock
        self.enabled = enabled
        self._since: float | None = None  # agent started speaking
        self._echo: list[float] = []
        self._loud_s = 0.0
        self.fired = 0

    def feed(self, chunk: Any) -> bool:
        try:
            if not self.enabled() or not self.speaking():
                self._since, self._echo, self._loud_s = None, [], 0.0
                return False
        except Exception:
            return False
        try:
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return False
        if arr.size == 0 or not np.isfinite(arr).all():
            return False
        rms = float(np.sqrt(np.mean(arr * arr)))
        dur = arr.size / float(self.sr)
        now = self.clock()
        if self._since is None:
            self._since = now
        if now - self._since < self.calib_s:
            # A clock that stalls or a broken audio callback must not grow calibration memory without bound.
            if len(self._echo) < 64:
                self._echo.append(rms)
            return False
        echo = float(np.median(self._echo)) if self._echo else 0.0
        if rms > max(self.floor, echo * self.ratio):
            self._loud_s += dur
        else:
            self._loud_s = max(0.0, self._loud_s - dur / 2)
        if self._loud_s >= self.hold_s:
            self._loud_s = 0.0
            self._since = None
            self.fired += 1
            with contextlib.suppress(Exception):
                self.on_barge()
            return True
        return False


def wrap_feed(voice: Any, barge: BargeIn) -> bool:
    """Adds the barge-in check in front of the app's microphone feed (``VoiceController.feed``). True when
    wrapped; never breaks the voice input."""
    orig = getattr(voice, "feed", None)
    if orig is None or getattr(orig, "_liveagent_barge", False):
        return False

    def feed(chunk: Any) -> Any:
        with contextlib.suppress(Exception):
            barge.feed(chunk)
        return orig(chunk)

    feed._liveagent_barge = True  # type: ignore[attr-defined]
    voice.feed = feed
    return True
