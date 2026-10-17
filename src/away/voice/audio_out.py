"""Serialized access to the sounddevice default output stream (v0.1.29).

PortAudio stream handles are not thread safe: the barge-in interrupt path
(``liveagent/speech.py``), the earcon thread (``voice/earcon.py``) and the TTS
playback path (``voice/natural.py``) can otherwise call ``sd.play`` / ``sd.stop``
at the same moment. That crashed the app twice in 0.1.27 with a native access
violation inside PortAudio, and a native access violation cannot be caught from
Python, so every output-stream call goes through this module's one process-wide
lock. ``sounddevice`` stays an optional dependency: it is imported lazily, only
when one of these functions actually runs.
"""

from __future__ import annotations

import threading
from typing import Any

_sd_lock = threading.Lock()


def _sd() -> Any:
    import sounddevice as sd  # lazy: sounddevice stays optional

    return sd


def play(samples: Any, sr: int) -> None:
    """Start ``samples`` on the default output stream. Serialized with every other
    output-stream call in this process. Returns at once; does not wait."""
    with _sd_lock:
        _sd().play(samples, int(sr))


def play_blocking(samples: Any, sr: int) -> None:
    """Like :func:`play`, then wait for the stream to finish. The wait happens
    OUTSIDE the lock, so a barge-in :func:`stop` lands at once instead of queueing
    behind the whole utterance."""
    play(samples, sr)
    _sd().wait()


def stop() -> None:
    """Stop the default output stream. Serialized with every other output-stream
    call in this process."""
    with _sd_lock:
        _sd().stop()
