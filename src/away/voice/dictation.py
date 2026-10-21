"""Push-to-talk dictation pipeline.

mic chunks -> buffer (while the hotkey is active) -> VAD trim -> speaker gate
-> ASR (with vocabulary/app prompt) -> cleanup -> paste/type.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from away.events import EventBus
from away.voice.asr import ASRBackend
from away.voice.context import whisper_prompt
from away.voice.hallucination import clean_transcript
from away.voice.output import TextSink


class Cleaner(Protocol):
    def clean(self, text: str, context: str | None = None) -> str: ...


class Trimmer(Protocol):
    def trim(self, audio: Any, pad_ms: int = 150) -> Any: ...


@dataclass
class DictationResult:
    status: str  # ok | silence | gated | empty | disabled | command
    raw: str = ""
    text: str = ""
    seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class DictationSession:
    def __init__(
        self,
        asr: ASRBackend,
        cleaner: Cleaner,
        sink: TextSink,
        vad: Trimmer,
        bus: EventBus | None = None,
        sample_rate: int = 16000,
        gate: Callable[[float, float], bool] | None = None,
        context: Callable[[], str | None] | None = None,
        vocabulary: list[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        command_handler: Callable[[str], bool] | None = None,
    ) -> None:
        self.asr, self.cleaner, self.sink, self.vad = asr, cleaner, sink, vad
        self.bus = bus
        self.sr = sample_rate
        self.gate = gate
        self.context = context
        self.vocabulary = vocabulary or []
        self.clock = clock
        # called with the raw transcript; returning True means it was a voice command (nothing is typed)
        self.command_handler = command_handler
        self.enabled = True
        self.recording = False
        self._chunks: list[NDArray[np.float32]] = []
        self._t_start = 0.0
        self._lock = threading.Lock()

    def _publish(self, topic: str, data: dict[str, Any]) -> None:
        if self.bus is not None:
            self.bus.publish(topic, data)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled and self.recording:
            with self._lock:
                self.recording = False
                self._chunks.clear()
        self._publish("dictation.enabled", {"enabled": enabled})

    def start(self) -> bool:
        if not self.enabled or self.recording:
            return False
        with self._lock:
            self._chunks = []
            self.recording = True
            self._t_start = self.clock()
        self._publish("dictation.start", {})
        return True

    def feed(self, chunk: Any) -> None:
        if self.recording:
            with self._lock:
                self._chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))

    def stop(self) -> DictationResult:
        if not self.recording:
            return DictationResult("disabled" if not self.enabled else "empty")
        with self._lock:
            self.recording = False
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
            self._chunks = []
        t_end = self.clock()
        result = self._process(audio, self._t_start, t_end)
        self._publish("dictation.result", {"status": result.status, "text": result.text})
        return result

    def cancel(self) -> None:
        """Stop recording and drop the audio without transcribing it (mute)."""
        with self._lock:
            self.recording = False
            self._chunks = []

    def toggle(self) -> DictationResult | None:
        if self.recording:
            return self.stop()
        self.start()
        return None

    def _process(self, audio: NDArray[np.float32], t0: float, t1: float) -> DictationResult:
        seconds = len(audio) / self.sr
        speech = self.vad.trim(audio)
        if len(speech) == 0:
            return DictationResult("silence", seconds=seconds)
        if self.gate is not None and not self.gate(t0, t1):
            return DictationResult("gated", seconds=seconds, meta={"reason": "speaker is not the enrolled user"})
        app = self.context() if self.context else None
        prompt = whisper_prompt(app, self.vocabulary)
        heard = self.asr.transcribe(np.asarray(speech, dtype=np.float32), prompt)
        raw, why = clean_transcript(heard, prompt, self.vocabulary, len(speech) / self.sr)
        if not raw:
            reason = why or "empty transcript"
            return DictationResult("empty", raw=heard, seconds=seconds, meta={"app": app, "reason": reason})
        if self.command_handler is not None and raw.strip() and self.command_handler(raw):
            return DictationResult("command", raw=raw, seconds=seconds, meta={"app": app})
        text = self.cleaner.clean(raw, app)
        if not text:
            return DictationResult("empty", raw=raw, seconds=seconds)
        self.sink.emit(text + " ")
        return DictationResult("ok", raw=raw, text=text, seconds=seconds, meta={"app": app})


class MicStream:  # pragma: no cover - needs a microphone
    """sounddevice input stream feeding a callback with float32 mono chunks."""

    def __init__(
        self, on_chunk: Callable[[NDArray[np.float32]], None], sample_rate: int = 16000, block_ms: int = 30
    ) -> None:
        import sounddevice as sd

        def cb(indata: Any, frames: int, t: Any, status: Any) -> None:
            on_chunk(indata[:, 0].copy())

        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(sample_rate * block_ms / 1000),
            callback=cb,
        )

    def __enter__(self) -> MicStream:
        self.stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stream.stop()
        self.stream.close()


class MicSupervisor:
    """Supervised lifecycle for the runner's microphone stream.

    The mic thread used to be ``with MicStream(...): stop_event.wait()``:
    if the stream failed to open (device unplugged, driver hiccup,
    permission denied), the exception left the thread and dictation was
    silently dead until the app restarted. The supervisor owns the stream
    instead. An open failure is recorded in :attr:`error` and handed to
    ``on_error`` (so the voice state surface can show it), and the stream
    is retried with bounded exponential backoff until the stop event
    fires. A stream that opens clears the error. A stream that opens but
    stops delivering chunks for ``stall_timeout_s`` is treated as failed
    and reopened the same way.
    """

    _POLL_S = 0.25  # stop-event responsiveness while a stream is live

    def __init__(
        self,
        on_chunk: Callable[[NDArray[np.float32]], None],
        sample_rate: int = 16000,
        *,
        stop_event: threading.Event | None = None,
        stream_factory: Callable[[Callable[[NDArray[np.float32]], None], int], Any] = MicStream,
        bus: EventBus | None = None,
        on_error: Callable[[str | None], None] | None = None,
        base_delay_s: float = 1.0,
        max_delay_s: float = 30.0,
        stall_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        self.on_chunk = on_chunk
        self.sample_rate = sample_rate
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._factory = stream_factory
        self.bus = bus
        self.on_error = on_error
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.stall_timeout_s = stall_timeout_s
        self.clock = clock
        self._wait_fn = wait
        self.error: str | None = None
        self.attempts = 0  # consecutive failures since the last live stream
        self.live = False
        self._last_chunk = 0.0

    # ------------------------------------------------------------------ state
    def _set_error(self, error: str | None) -> None:
        if error == self.error:
            return
        self.error = error
        if self.bus is not None:
            self.bus.publish("voice.mic_error", {"error": error, "attempts": self.attempts})
        if self.on_error is not None:
            self.on_error(error)

    def _failed(self, message: str) -> None:
        self.attempts += 1
        self._set_error(message)

    def _opened(self) -> None:
        self.live = True
        self.attempts = 0
        self._last_chunk = self.clock()
        self._set_error(None)

    # ------------------------------------------------------------------- loop
    def _wait(self, seconds: float) -> bool:
        """Wait ``seconds``; True means the supervisor should stop."""
        if self._wait_fn is not None:
            return self._wait_fn(seconds)
        return self.stop_event.wait(seconds)

    def _feed(self, chunk: NDArray[np.float32]) -> None:
        self._last_chunk = self.clock()
        self.on_chunk(chunk)

    def _wait_live(self) -> bool:
        """Block while the stream is live; True when it stopped delivering audio."""
        while not self.stop_event.is_set():
            if self._wait(self._POLL_S):
                return False
            if self.clock() - self._last_chunk > self.stall_timeout_s:
                return True
        return False

    def _backoff(self) -> float:
        return min(self.base_delay_s * 2 ** max(0, self.attempts - 1), self.max_delay_s)

    def run(self) -> None:
        """Thread body: keep a stream open until the stop event fires."""
        while not self.stop_event.is_set():
            stalled = False
            try:
                with self._factory(self._feed, self.sample_rate):
                    self._opened()
                    stalled = self._wait_live()
            except Exception as exc:
                if not self.stop_event.is_set():
                    self._failed(f"microphone error: {exc}")
            self.live = False
            if self.stop_event.is_set():
                return
            if stalled:
                self._failed("microphone stopped delivering audio")
            if self._wait(self._backoff()):
                return


def probe_microphone(seconds: float = 0.5, sample_rate: int = 16000) -> tuple[bool, str]:  # pragma: no cover
    try:
        import sounddevice as sd
    except Exception as exc:
        return False, f"sounddevice unavailable ({exc}); pip install 'project-away[voice]'"
    try:
        rec = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        level = float(np.sqrt(np.mean(rec**2)))
        return True, f"microphone OK (rms {level:.4f}; 0.0000 may mean muted or permission denied)"
    except Exception as exc:
        return False, f"microphone error: {exc}"


def hotkey_listener(mode: str, hotkey: str, hold_key: str, session: DictationSession) -> Any:  # pragma: no cover
    """Start a pynput listener; returns it (call .stop())."""
    from pynput import keyboard

    if mode == "toggle":
        listener = keyboard.GlobalHotKeys(
            {hotkey: lambda: threading.Thread(target=session.toggle, daemon=True).start()}
        )
    else:
        target = getattr(keyboard.Key, hold_key, None) or keyboard.KeyCode.from_char(hold_key)

        def on_press(k: Any) -> None:
            if k == target and not session.recording:
                session.start()

        def on_release(k: Any) -> None:
            if k == target and session.recording:
                threading.Thread(target=session.stop, daemon=True).start()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener
