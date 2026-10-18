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
import time
from typing import Any

_sd_lock = threading.Lock()
# Streams that sd.stop() already closed. play_blocking's teardown must skip
# its close() for these: Stream.close() has no double-close guard and
# sd.stop() closes the stream itself, so closing again would re-enter
# Pa_CloseStream on a dead handle. At most one entry: sounddevice keeps a
# single current output stream (play() stops the previous one first).
_closed_by_stop: list[Any] = []


def _note_stopped(stream: Any) -> None:
    """Remember that the stream was closed via sd.stop(); caller holds _sd_lock."""
    _closed_by_stop.clear()
    if stream is not None:
        _closed_by_stop.append(stream)


def _current_stream(sd: Any) -> Any | None:
    try:
        return sd.get_stream()
    except Exception:
        return None  # play()/rec()/playrec() was not called yet


def _sd() -> Any:
    import sounddevice as sd  # lazy: sounddevice stays optional

    return sd


def play(samples: Any, sr: int) -> None:
    """Start ``samples`` on the default output stream. Serialized with every other
    output-stream call in this process. Returns at once; does not wait."""
    with _sd_lock:
        sd = _sd()
        prev = _current_stream(sd)
        sd.play(samples, int(sr))
        # sd.play() stops (and closes) any current stream first.
        _note_stopped(prev)


def play_blocking(samples: Any, sr: int) -> None:
    """Like :func:`play`, then wait for the stream to finish.

    The wait happens OUTSIDE the lock so a barge-in :func:`stop` lands at once
    instead of queueing behind the whole utterance. The stream teardown
    (``close()``) happens UNDER the lock: ``sounddevice.wait()`` closes the
    stream in a ``finally`` block and ``sounddevice.stop()`` also closes it, so
    leaving the teardown unlocked would let the two ``Pa_CloseStream`` calls
    race on the same handle (``Stream.close()`` is not idempotent). We therefore
    poll for completion ourselves and close under the lock instead of calling
    ``sounddevice.wait()``. When :func:`stop` (or another :func:`play`) already
    closed this stream, the teardown skips its ``close()`` instead of closing
    the dead handle a second time.
    """
    sd = _sd()
    with _sd_lock:
        prev = _current_stream(sd)
        sd.play(samples, int(sr))
        _note_stopped(prev)
        stream = sd.get_stream()
    try:
        while stream.active:
            time.sleep(0.005)
    finally:
        with _sd_lock:
            if _closed_by_stop and _closed_by_stop[0] is stream:
                _closed_by_stop.clear()
            else:
                stream.close()


def stop() -> None:
    """Stop the default output stream. Serialized with every other output-stream
    call in this process. ``sd.stop()`` closes the stream itself; the closure is
    recorded so a concurrent :func:`play_blocking` teardown does not close it
    again."""
    with _sd_lock:
        sd = _sd()
        stream = _current_stream(sd)
        sd.stop()
        _note_stopped(stream)
