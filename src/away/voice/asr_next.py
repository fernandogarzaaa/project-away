"""Streaming speech-to-text backends from the 2026-10 model refresh (v0.1.24). Optional; Whisper stays the default.

* :class:`MoonshineASR` - Moonshine v2 Small / Medium Streaming (MIT) on the CPU through ``moonshine-voice``.
* :class:`NemotronASR` - NVIDIA Nemotron 3.5 ASR Streaming 0.6B (OpenMDW-1.1) through NeMo-Speech.cpp's local
  server (``nemo-speech serve``, started on 127.0.0.1 with a random key). **Experimental**: the Windows CUDA build
  has not been verified on a real PC yet, so any start-up or CUDA failure falls back to Whisper and says why.

Both answer the same ``transcribe(audio, prompt) -> str`` the dictation pipeline already uses, so they feed the
existing pipeline unchanged. While you speak, :meth:`push` streams the microphone to the engine and ``on_partial``
receives the words so far (``voice.partial`` on the event bus); when the utterance ends, Moonshine's streamed
result is used directly if it covered the utterance, otherwise the utterance is transcribed in one go.

Nothing is imported or started until a backend is built: the engines come from :mod:`away.app.engines`, the model
files from the catalog (:mod:`away.app.model_fetch`).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

NEXT_BACKENDS = ("moonshine", "nemotron")
MOONSHINE_IDS = {"small": "stt-moonshine-v2-small-streaming-en", "medium": "stt-moonshine-v2-medium-streaming-en"}
NEMOTRON_ID = "stt-nemotron-3.5-streaming-0.6b-q8"
NEMOTRON_FILE = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
# cache-aware streaming: chunk = (right context + 1) x 80 ms (Nemotron 3.5 model card: 80/160/320/560/1120 ms)
CHUNK_MS = (80, 160, 320, 560, 1120)
EXPERIMENTAL_NOTE = (
    "Nemotron streaming speech is experimental: its Windows graphics-card build has not been verified on a real PC"
)

PartialFn = Callable[[str], None]


def chunk_to_right_context(chunk_ms: int) -> int:
    ms = min(CHUNK_MS, key=lambda c: abs(c - int(chunk_ms)))
    return ms // 80 - 1


def pcm16_wav(audio: NDArray[np.float32], sr: int = 16000) -> bytes:
    a = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


def multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = "away" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    out.write(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
    )
    out.write(data)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def _local_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through HTTP(S)_PROXY


# ----------------------------------------------------------------------------------------------- Moonshine
class MoonshineASR:
    """Moonshine v2 streaming on the CPU. ``transcriber`` is injectable for tests (a moonshine ``Transcriber``)."""

    name = "moonshine"

    def __init__(
        self,
        model_dir: Path | str,
        size: str = "small",
        update_interval: float = 0.3,
        transcriber: Any = None,
        sample_rate: int = 16000,
    ) -> None:
        self.sr = sample_rate
        self.size = size
        self.update_interval = update_interval
        self.on_partial: PartialFn | None = None
        self.last_confidence: float | None = None
        self.device = "cpu"
        self._lock = threading.Lock()
        self._stream: Any = None
        self._streamed = 0
        self._lines: dict[int, str] = {}
        if transcriber is None:
            from away.app import engines

            engines.activate("moonshine")
            from moonshine_voice import ModelArch
            from moonshine_voice.transcriber import Transcriber

            arch = ModelArch.MEDIUM_STREAMING if size == "medium" else ModelArch.SMALL_STREAMING
            transcriber = Transcriber(str(model_dir), arch, update_interval)
        self.tr = transcriber

    # live partials
    def _on_event(self, ev: Any) -> None:
        line = getattr(ev, "line", None)
        if line is None:
            return
        key = int(getattr(line, "line_id", id(line)) or 0)
        self._lines[key] = str(getattr(line, "text", "") or "").strip()
        if self.on_partial is not None:
            text = " ".join(t for t in self._lines.values() if t)
            with contextlib.suppress(Exception):
                self.on_partial(text)

    def push(self, chunk: Any) -> None:
        a = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            if self._stream is None:
                self._stream = self.tr.create_stream(self.update_interval)
                self._stream.add_listener(self._on_event)
                self._stream.start()
                self._streamed, self._lines = 0, {}
            self._stream.add_audio(a.tolist(), self.sr)
            self._streamed += len(a)

    def reset(self) -> None:
        with self._lock:
            self._close_stream()

    def _close_stream(self) -> str:
        s, self._stream = self._stream, None
        text = ""
        if s is not None:
            try:
                res = s.stop()
                lines = getattr(res, "lines", None) or []
                text = " ".join(str(getattr(ln, "text", "")).strip() for ln in lines if getattr(ln, "text", ""))
            finally:
                with contextlib.suppress(Exception):
                    s.close()
        if not text:
            text = " ".join(t for t in self._lines.values() if t)
        self._streamed, self._lines = 0, {}
        return text.strip()

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        with self._lock:
            covered = self._stream is not None and self._streamed >= 0.8 * len(a) > 0
            streamed = self._close_stream() if self._stream is not None else ""
            if covered and streamed:
                return streamed
            res = self.tr.transcribe_without_streaming(a.tolist(), self.sr)
        lines = getattr(res, "lines", None) or []
        return " ".join(str(getattr(ln, "text", "")).strip() for ln in lines if getattr(ln, "text", "")).strip()

    def close(self) -> None:
        self.reset()
        with contextlib.suppress(Exception):
            self.tr.close()


# ----------------------------------------------------------------------------------------------- Nemotron
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class NemoServer:
    """``nemo-speech serve`` for one ASR model on 127.0.0.1 (random port, random API key)."""

    def __init__(
        self,
        exe: Path | str,
        model: Path | str,
        device: str = "cuda",
        chunk_ms: int = 560,
        threads: int = 2,
        popen: Callable[..., Any] = subprocess.Popen,
        opener: Any = None,
    ) -> None:
        self.exe, self.model, self.device = str(exe), str(model), device
        self.right_context = chunk_to_right_context(chunk_ms)
        self.threads = threads
        self.port = 0
        self.key = secrets.token_urlsafe(24)
        self.popen = popen
        self.opener = opener or _local_opener()
        self.proc: Any = None
        self.log: list[str] = []

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [
            self.exe,
            "serve",
            "--asr-model",
            self.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--api-key",
            self.key,
            "--no-ui",
            "--threads",
            str(self.threads),
            "--device",
            self.device,
            "--asr.streaming.rnnt_right_context",
            str(self.right_context),
        ]

    def start(self, timeout: float = 90.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.port = self.port or free_port()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
        self.proc = self.popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        threading.Thread(target=self._drain, daemon=True).start()
        end = clock() + timeout
        while clock() < end:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the speech engine stopped while starting: {self.tail()}")
            if self.ready():
                return
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"the speech engine did not get ready in {timeout:.0f} s: {self.tail()}")

    def _drain(self) -> None:
        out = getattr(self.proc, "stdout", None)
        if out is None:
            return
        for raw in iter(out.readline, b""):
            self.log.append(raw.decode("utf-8", "replace").rstrip() if isinstance(raw, bytes) else str(raw).rstrip())
            del self.log[:-40]

    def tail(self, n: int = 3) -> str:
        return " | ".join(self.log[-n:])[:400] or "no output"

    def ready(self) -> bool:
        try:
            with self.opener.open(self.base + "/ready", timeout=1.0) as r:
                return bool(json.loads(r.read() or b"{}").get("ready", r.status == 200))
        except Exception:
            return False

    def transcribe(self, audio: NDArray[np.float32], sr: int = 16000, prompt: str | None = None) -> str:
        fields = {"response_format": "json"}
        if prompt:
            fields["prompt"] = prompt[:200]
        body, ctype = multipart(fields, "file", "speech.wav", pcm16_wav(audio, sr))
        req = urllib.request.Request(
            self.base + "/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": ctype, "Authorization": f"Bearer {self.key}"},
        )
        with self.opener.open(req, timeout=60.0) as r:
            return str(json.loads(r.read()).get("text", "")).strip()

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        with contextlib.suppress(Exception):
            p.terminate()
            p.wait(timeout=5)
        with contextlib.suppress(Exception):
            p.kill()


class RealtimeStream:
    """One live transcription over ``/v1/audio/transcriptions/realtime``: PCM16 in, partial text out."""

    def __init__(self, server: NemoServer, on_partial: PartialFn | None, sr: int = 16000, ws: Any = None) -> None:
        from away.voice.ws_client import WebSocket

        self.ws = ws or WebSocket(f"ws://127.0.0.1:{server.port}/v1/audio/transcriptions/realtime?api_key={server.key}")
        self.on_partial = on_partial
        self.finals: list[str] = []
        self.partial = ""
        self.closed = False
        self.done = threading.Event()  # set when the server sent the final text of a committed stream
        self.ws.send_text(json.dumps({"type": "session.update", "session": {"sample_rate": sr}}))
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        while not self.closed:
            try:
                msg = self.ws.recv(timeout=30.0)
            except Exception:
                break
            if msg is None:
                break
            if not isinstance(msg, str):
                continue
            with contextlib.suppress(ValueError):
                self.handle(json.loads(msg))
        self.closed = True

    def handle(self, ev: dict[str, Any]) -> None:
        kind = str(ev.get("type", ""))
        if kind.endswith("input_audio_transcription.delta"):
            delta = str(ev.get("delta", ev.get("text", "")))
            if not delta:
                return
            self.partial = delta if ev.get("replace") or not self.partial else self.partial + delta
        elif kind.endswith("input_audio_transcription.completed"):
            text = str(ev.get("transcript", ev.get("text", ""))).strip()
            if text:
                self.finals.append(text)
            self.partial = ""
            self.done.set()
        else:
            return
        if self.on_partial is not None:
            with contextlib.suppress(Exception):
                self.on_partial(" ".join([*self.finals, self.partial]).strip())

    def send(self, audio: NDArray[np.float32]) -> None:
        a = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self.ws.send_binary((a * 32767.0).astype("<i2").tobytes())

    def commit(self) -> None:
        self.ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))

    def close(self) -> None:
        if not self.closed:
            with contextlib.suppress(Exception):
                self.commit()
        self.closed = True
        with contextlib.suppress(Exception):
            self.ws.close()


class NemotronASR:
    """Nemotron 3.5 streaming through a local NeMo-Speech.cpp server. Final text = one request per utterance
    (exact and simple); partials = a realtime stream while you talk (best effort, never blocks the microphone)."""

    name = "nemotron"

    def __init__(self, server: NemoServer, sample_rate: int = 16000, partials: bool = True) -> None:
        self.server = server
        self.sr = sample_rate
        self.partials = partials
        self.on_partial: PartialFn | None = None
        self.last_confidence: float | None = None
        self.device = server.device.split(":", 1)[0]
        self._q: queue.Queue[Any] = queue.Queue(maxsize=200)
        self._stream: RealtimeStream | None = None
        self._pushed = 0
        self._handoff = threading.Lock()  # a committed stream goes to its waiter or, if it gave up, gets closed
        self.final_wait_s = 3.0
        self._worker = threading.Thread(target=self._pump, daemon=True)
        self._worker.start()

    def _pump(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                if isinstance(item, tuple):  # ("commit", holder, gave_up): finish the live stream, hand it back
                    st, self._stream = self._stream, None
                    if st is not None and not st.closed:
                        st.commit()
                        holder, gave_up = item[1], item[2]
                        with self._handoff:
                            late = gave_up.is_set()
                            if not late:
                                holder.append(st)
                        if late:  # the waiter already returned: nobody else will close it
                            st.close()
                    continue
                if isinstance(item, str):  # "reset"
                    if self._stream is not None:
                        self._stream.close()
                    self._stream = None
                    continue
                if self._stream is None or self._stream.closed:
                    self._stream = RealtimeStream(self.server, self.on_partial, self.sr)
                self._stream.send(item)
            except Exception:
                self._stream = None  # partials are best effort

    def push(self, chunk: Any) -> None:
        if not self.partials or self.on_partial is None:
            return
        a = np.asarray(chunk, dtype=np.float32).reshape(-1).copy()
        try:
            self._q.put_nowait(a)
            self._pushed += len(a)
        except queue.Full:
            pass

    def reset(self) -> None:
        self._pushed = 0
        with contextlib.suppress(queue.Full):
            self._q.put_nowait("reset")

    def _streamed_final(self, n: int) -> str:
        """The live stream's own final text when it saw (almost) the whole utterance; "" otherwise."""
        if self._pushed < 0.8 * n or n <= 0:
            return ""
        holder: list[RealtimeStream] = []
        gave_up = threading.Event()
        try:
            self._q.put(("commit", holder, gave_up), timeout=0.5)
        except queue.Full:
            return ""
        end = time.monotonic() + self.final_wait_s
        while not holder and time.monotonic() < end:
            time.sleep(0.01)
        done = bool(holder) and holder[0].done.wait(max(0.0, end - time.monotonic()))
        with self._handoff:
            gave_up.set()
            st = holder[0] if holder else None
        if st is not None:
            st.close()  # on every path: a slow final must not leave its socket and server session open
        return " ".join(st.finals).strip() if st is not None and done else ""

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        n = len(np.asarray(audio).reshape(-1))
        text = self._streamed_final(n)
        self.reset()
        return text or self.server.transcribe(audio, self.sr, prompt)

    def close(self) -> None:
        with contextlib.suppress(queue.Full):
            self._q.put_nowait(None)
        self.server.stop()


# ----------------------------------------------------------------------------------------------- fallback
class NextFallbackASR:
    """A new backend that switches (once, for this session) to Whisper when it fails, saying why."""

    def __init__(
        self,
        primary: Any,
        build_fallback: Callable[[], Any],
        on_fallback: Callable[[str], object] | None = None,
        label: str = "",
    ) -> None:
        self.primary = primary
        self.build_fallback = build_fallback
        self.on_fallback = on_fallback
        self.label = label or getattr(primary, "name", "new speech model")
        self.fallback: Any = None
        self.name = getattr(primary, "name", "next")

    @property
    def active(self) -> Any:
        return self.fallback if self.fallback is not None else self.primary

    @property
    def device(self) -> str:
        return str(getattr(self.active, "device", "cpu"))

    @property
    def last_confidence(self) -> float | None:
        v = getattr(self.active, "last_confidence", None)
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def on_partial(self) -> PartialFn | None:
        return getattr(self.primary, "on_partial", None)

    @on_partial.setter
    def on_partial(self, fn: PartialFn | None) -> None:
        if hasattr(self.primary, "on_partial"):
            self.primary.on_partial = fn

    def push(self, chunk: Any) -> None:
        if self.fallback is None:
            with contextlib.suppress(Exception):
                self.primary.push(chunk)

    def reset(self) -> None:
        if self.fallback is None:
            with contextlib.suppress(Exception):
                self.primary.reset()

    def transcribe(self, audio: NDArray[np.float32], prompt: str | None = None) -> str:
        if self.fallback is not None:
            return str(self.fallback.transcribe(audio, prompt))
        try:
            return str(self.primary.transcribe(audio, prompt))
        except Exception as exc:
            self.fallback = self.build_fallback()
            with contextlib.suppress(Exception):
                self.primary.close()
            if self.on_fallback is not None:
                self.on_fallback(f"{self.label} stopped working ({str(exc)[:160]}); using Whisper instead")
            return str(self.fallback.transcribe(audio, prompt))


# ----------------------------------------------------------------------------------------------- build
def nvidia_gpu(which: Callable[[str], str | None] = shutil.which) -> bool:
    return bool(which("nvidia-smi"))


def nemotron_plan(
    device: str = "auto", data_dir: Path | None = None, which: Callable[[str], str | None] = shutil.which
) -> dict[str, Any]:
    """What the Nemotron backend would use here: engine variant, device, and whether it is ready."""
    from away.app import engines

    gpu = nvidia_gpu(which)
    want = device if device in ("cuda", "vulkan", "cpu") else ("cuda" if gpu else "cpu")
    avail = engines.variants("nemo-speech")
    variant = want if want in avail else ("vulkan" if "vulkan" in avail and want != "cpu" else "cpu")
    a = engines.archive_for("nemo-speech", variant)
    return {
        "variant": variant,
        "device": {"cuda": "cuda:0", "vulkan": "vulkan:0"}.get(variant, "cpu"),
        "nvidia_gpu": gpu,
        "engine_installed": engines.installed(a, data_dir),
        "engine_size_mb": round(a.size / 1e6) if a else 0,
        "experimental": True,
        "note": EXPERIMENTAL_NOTE,
    }


def build_next(
    backend: str,
    opts: Any,
    sample_rate: int = 16000,
    data_dir: Path | None = None,
    reg: Any = None,
) -> Any:
    """Build the Moonshine / Nemotron backend from ``opts`` (``cfg.models_next``). Raises with a plain reason."""
    from away.app import engines, model_fetch
    from away.app.model_registry import Registry

    reg = reg or Registry(data_dir)
    if backend == "moonshine":
        size = str(getattr(opts, "moonshine_model", "small") or "small")
        m = reg.get(MOONSHINE_IDS.get(size, MOONSHINE_IDS["small"]))
        if not model_fetch.have(m, data_dir):
            raise RuntimeError(f"{m.title} is not downloaded yet (Settings > Models)")
        if not engines.activate("moonshine", data_dir):
            raise RuntimeError("the Moonshine engine is not installed (Settings > Models > Download)")
        folder = model_fetch.model_dir(m.id, data_dir)
        cfg = model_fetch.find(folder, "streaming_config.json")
        return MoonshineASR(cfg.parent if cfg else folder, size, sample_rate=sample_rate)
    if backend == "nemotron":
        m = reg.get(NEMOTRON_ID)
        if not model_fetch.have(m, data_dir):
            raise RuntimeError(f"{m.title} is not downloaded yet (Settings > Models)")
        plan = nemotron_plan(str(getattr(opts, "nemotron_device", "auto")), data_dir)
        a = engines.archive_for("nemo-speech", plan["variant"])
        exe = engines.binary(a, data_dir) if a is not None and engines.installed(a, data_dir) else None
        if exe is None:
            raise RuntimeError("the NeMo-Speech engine is not installed (Settings > Models > Download)")
        gguf = model_fetch.find(model_fetch.model_dir(m.id, data_dir), NEMOTRON_FILE)
        if gguf is None:
            raise RuntimeError(f"{m.title} is incomplete; download it again")
        srv = NemoServer(exe, gguf, plan["device"], int(getattr(opts, "nemotron_chunk_ms", 560) or 560))
        srv.start()
        try:
            asr = NemotronASR(srv, sample_rate, bool(getattr(opts, "stream_partials", True)))
        except BaseException:
            srv.stop()
            raise
        try:
            asr.transcribe(np.zeros(16000, dtype=np.float32))  # warm-up: a CUDA problem shows here, not mid-dictation
        except BaseException:
            asr.close()  # else the server keeps its VRAM while the Whisper fallback starts
            raise
        return asr
    raise ValueError(f"unknown speech backend {backend!r}")


def make_next_asr(
    backend: str,
    opts: Any,
    build_whisper: Callable[[], Any],
    on_fallback: Callable[[str], object] | None = None,
    data_dir: Path | None = None,
    builder: Callable[..., Any] = build_next,
    sample_rate: int = 16000,
) -> Any:
    """The new backend wrapped in :class:`NextFallbackASR`; Whisper (with the reason) when it cannot start."""
    label = {"moonshine": "Moonshine speech", "nemotron": "Nemotron speech (experimental)"}.get(backend, backend)
    try:
        primary = builder(backend, opts, sample_rate, data_dir)
    except Exception as exc:
        if on_fallback is not None:
            extra = f" {EXPERIMENTAL_NOTE}." if backend == "nemotron" else ""
            on_fallback(f"{label} could not start ({str(exc)[:200]}); using Whisper instead.{extra}")
        return build_whisper()
    return NextFallbackASR(primary, build_whisper, on_fallback, label)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
