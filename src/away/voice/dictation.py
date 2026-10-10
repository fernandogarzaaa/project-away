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
from away.voice.output import TextSink


class Cleaner(Protocol):
    def clean(self, text: str, context: str | None = None) -> str: ...


class Trimmer(Protocol):
    def trim(self, audio: Any, pad_ms: int = 150) -> Any: ...


@dataclass
class DictationResult:
    status: str  # ok | silence | gated | empty | disabled
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
    ) -> None:
        self.asr, self.cleaner, self.sink, self.vad = asr, cleaner, sink, vad
        self.bus = bus
        self.sr = sample_rate
        self.gate = gate
        self.context = context
        self.vocabulary = vocabulary or []
        self.clock = clock
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
        raw = self.asr.transcribe(np.asarray(speech, dtype=np.float32), whisper_prompt(app, self.vocabulary))
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
