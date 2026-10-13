"""Is another app playing sound right now? (v0.1.22)

On the user's PC a YouTube video in the background was transcribed and typed. Windows can tell which apps are
playing sound: every audio session has a peak meter (``IAudioMeterInformation``), read here through ``pycaw`` /
``comtypes`` when they are installed. :class:`OtherAudioMeter` samples the sessions of *other* processes a few
times a second on its own thread and remembers when they were last loud, so the dictation filter can ask "was
another app playing during this utterance?" (:meth:`OtherAudioMeter.playing_during`).

Everywhere else (macOS, Linux, or Windows without pycaw) the meter is simply unavailable and nothing changes.
Project Away's own voice is never counted: its sessions belong to this process (the agent's speech is handled by
the echo guard in :mod:`away.voice.speech`).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

log = logging.getLogger("away.voice")

PEAK_THRESHOLD = 0.02  # a session's peak meter above this counts as "playing" (0..1)
POLL_S = 0.25
HISTORY = 600  # samples kept (about 2.5 minutes at 4 per second)


def pycaw_reader() -> Callable[[], list[tuple[int, str, float]]] | None:  # pragma: no cover - Windows audio
    """``() -> [(pid, process name, peak 0..1), ...]`` for every audio session, or None without pycaw."""
    if sys.platform != "win32":
        return None
    try:
        from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
    except Exception:
        return None

    def read() -> list[tuple[int, str, float]]:
        out: list[tuple[int, str, float]] = []
        for s in AudioUtilities.GetAllSessions():
            try:
                meter = s._ctl.QueryInterface(IAudioMeterInformation)
                peak = float(meter.GetPeakValue())
            except Exception:
                continue
            pid = int(getattr(s, "ProcessId", 0) or 0)
            proc = getattr(s, "Process", None)
            name = ""
            with contextlib.suppress(Exception):
                name = str(proc.name()) if proc is not None else ""
            out.append((pid, name, peak))
        return out

    return read


class OtherAudioMeter:
    """Samples other apps' audio peaks (``reader``) every ``poll_s`` on a daemon thread. ``own_pids``: process ids
    whose sound is never counted (this process by default)."""

    def __init__(
        self,
        reader: Callable[[], list[tuple[int, str, float]]] | None = None,
        *,
        threshold: float = PEAK_THRESHOLD,
        poll_s: float = POLL_S,
        clock: Callable[[], float] = time.monotonic,
        own_pids: set[int] | None = None,
        co_init: Callable[[], Any] | None = None,
    ) -> None:
        self.reader = reader
        self.threshold = float(threshold)
        self.poll_s = float(poll_s)
        self.clock = clock
        self.own_pids = set(own_pids) if own_pids is not None else {os.getpid()}
        self.co_init = co_init
        self.samples: deque[tuple[float, bool, str]] = deque(maxlen=HISTORY)  # (t, loud, which app)
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self.reader is not None and not self.error

    def sample(self) -> bool:
        """Read once; True when another app is playing."""
        if self.reader is None:
            return False
        loud, who = False, ""
        try:
            for pid, name, peak in self.reader():
                if pid in self.own_pids or pid == 0:
                    continue  # our own voice, and the system sounds session
                if peak >= self.threshold:
                    loud, who = True, name
                    break
        except Exception as exc:
            self.error = str(exc)[:200]
            return False
        self.samples.append((self.clock(), loud, who))
        return loud

    def playing_during(self, t0: float, t1: float) -> bool:
        """True when any sample in ``[t0 - poll, t1 + poll]`` heard another app."""
        pad = self.poll_s * 1.5
        return any(loud and t0 - pad <= t <= t1 + pad for t, loud, _who in list(self.samples))

    def playing_app(self, t0: float, t1: float) -> str:
        pad = self.poll_s * 1.5
        for t, loud, who in list(self.samples):
            if loud and t0 - pad <= t <= t1 + pad:
                return who
        return ""

    def start(self) -> OtherAudioMeter:
        if self.reader is None or (self._thread is not None and self._thread.is_alive()):
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="away-other-audio")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:  # pragma: no cover - timing loop; sample() is unit-tested
        if self.co_init is not None:
            with contextlib.suppress(Exception):
                self.co_init()
        else:
            with contextlib.suppress(Exception):
                import comtypes

                comtypes.CoInitialize()
        while not self._stop.wait(self.poll_s):
            self.sample()
            if self.error:
                log.info("other-app audio meter stopped: %s", self.error)
                return


def make_meter() -> OtherAudioMeter:
    """The meter for this computer (unavailable when pycaw/comtypes are missing or not on Windows)."""
    return OtherAudioMeter(pycaw_reader())
