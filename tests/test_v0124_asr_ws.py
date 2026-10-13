"""v0.1.24 model refresh: streaming speech-to-text (away.voice.asr_next) and its stdlib WebSocket client
(away.voice.ws_client), against fakes: a fake moonshine transcriber, a fake ``nemo-speech serve`` (a real HTTP
server on 127.0.0.1 in a thread, started through a fake Popen) and a tiny WebSocket server.
"""

from __future__ import annotations

import contextlib
import io
import json
import queue
import socket
import struct
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from away.voice import asr_next, ws_client
from away.voice.ws_client import OP_BIN, OP_CLOSE, OP_PING, OP_PONG, OP_TEXT


# ============================================================================================ ws_client
def test_accept_key_matches_rfc6455_example():
    assert ws_client.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


@pytest.mark.parametrize("n", [0, 5, 125, 126, 300, 65535, 70000])
def test_frames_round_trip_with_every_length_encoding(n):
    payload = bytes(i % 251 for i in range(n))
    raw = ws_client.encode_frame(OP_BIN, payload, mask=b"\x01\x02\x03\x04")
    assert raw[0] == 0x80 | OP_BIN and raw[1] & 0x80  # final + masked (clients always mask)
    buf = io.BytesIO(raw)
    fin, op, data = ws_client.decode_frame(buf.read)
    assert (fin, op, data) == (True, OP_BIN, payload)
    assert buf.read() == b""


def test_client_frames_are_actually_masked():
    raw = ws_client.encode_frame(OP_TEXT, b"hello", mask=b"\xff\xff\xff\xff")
    assert b"hello" not in raw


def _server_frame(op: int, payload: bytes, fin: bool = True) -> bytes:
    """Server frames are not masked."""
    n = len(payload)
    head = bytes([(0x80 if fin else 0) | op])
    if n < 126:
        head += bytes([n])
    elif n < 1 << 16:
        head += bytes([126]) + struct.pack("!H", n)
    else:
        head += bytes([127]) + struct.pack("!Q", n)
    return head + payload


class FakeWSServer:
    """One-connection WebSocket server. ``script(conn, read_frame)`` runs after the handshake."""

    def __init__(self, script, accept_override: str | None = None, status: str = "101 Switching Protocols"):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.script = script
        self.accept_override = accept_override
        self.status = status
        self.request = b""
        self.received: list[tuple[int, bytes]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        conn, _ = self.sock.accept()
        conn.settimeout(5)
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += conn.recv(4096)
        head, rest = buf.split(b"\r\n\r\n", 1)
        self.request = head
        key = next(
            ln.split(b":", 1)[1].strip().decode()
            for ln in head.split(b"\r\n")
            if ln.lower().startswith(b"sec-websocket-key")
        )
        acc = self.accept_override or ws_client.accept_key(key)
        conn.sendall(
            f"HTTP/1.1 {self.status}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {acc}\r\n\r\n".encode()
        )
        pending = [rest]

        def rd(n):
            data = pending[0]
            while len(data) < n:
                chunk = conn.recv(65536)
                if not chunk:
                    raise ConnectionError
                data += chunk
            pending[0] = data[n:]
            return data[:n]

        def read_frame():
            fin, op, data = ws_client.decode_frame(rd)
            self.received.append((op, data))
            return op, data

        with contextlib.suppress(Exception):
            self.script(conn, read_frame)
        with contextlib.suppress(Exception):
            conn.close()
        self.sock.close()


def test_websocket_handshake_echo_ping_and_close():
    def script(conn, read_frame):
        op, data = read_frame()  # text from the client
        conn.sendall(_server_frame(OP_PING, b"p"))  # the client must answer and keep going
        conn.sendall(_server_frame(OP_TEXT, b"echo:" + data))
        assert read_frame() == (OP_PONG, b"p")
        op, data = read_frame()  # binary from the client
        conn.sendall(_server_frame(OP_BIN, data[::-1]))
        conn.sendall(_server_frame(OP_TEXT, b"frag-", fin=False) + _server_frame(0x0, b"ment"))
        conn.sendall(_server_frame(OP_CLOSE, b""))
        read_frame()  # the client's close

    srv = FakeWSServer(script)
    ws = ws_client.WebSocket(f"ws://127.0.0.1:{srv.port}/v1/audio/transcriptions/realtime?api_key=k", timeout=5)
    ws.send_text("hi")
    assert ws.recv(timeout=5) == "echo:hi"
    ws.send_binary(b"\x01\x02\x03")
    assert ws.recv(timeout=5) == b"\x03\x02\x01"
    assert ws.recv(timeout=5) == "frag-ment"
    assert ws.recv(timeout=5) is None  # server closed the stream
    ws.close()
    srv.thread.join(5)
    assert b"GET /v1/audio/transcriptions/realtime?api_key=k HTTP/1.1" in srv.request
    assert srv.received[0] == (OP_TEXT, b"hi")


def test_websocket_rejects_a_bad_handshake_and_a_refusal():
    srv = FakeWSServer(lambda c, r: None, accept_override="bogus")
    with pytest.raises(ws_client.WebSocketError, match="bad handshake"):
        ws_client.WebSocket(f"ws://127.0.0.1:{srv.port}/", timeout=5)
    srv2 = FakeWSServer(lambda c, r: None, status="403 Forbidden")
    with pytest.raises(ws_client.WebSocketError, match="refused"):
        ws_client.WebSocket(f"ws://127.0.0.1:{srv2.port}/", timeout=5)


def test_websocket_only_plain_ws():
    with pytest.raises(ws_client.WebSocketError):
        ws_client.WebSocket("wss://example.com/")


# ============================================================================================ helpers
def test_chunk_to_right_context_snaps_to_the_model_card_sizes():
    assert asr_next.chunk_to_right_context(80) == 0
    assert asr_next.chunk_to_right_context(560) == 6
    assert asr_next.chunk_to_right_context(1120) == 13
    assert asr_next.chunk_to_right_context(500) == 6  # nearest: 560
    assert asr_next.chunk_to_right_context(99999) == 13


def test_pcm16_wav_is_16k_mono_and_clipped():
    audio = np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32)
    with wave.open(io.BytesIO(asr_next.pcm16_wav(audio))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        pcm = np.frombuffer(w.readframes(5), dtype="<i2")
    assert list(pcm) == [0, 16383, -16383, 32767, -32767]


def test_multipart_body_carries_fields_and_file():
    body, ctype = asr_next.multipart({"response_format": "json"}, "file", "speech.wav", b"RIFFdata")
    boundary = ctype.split("boundary=", 1)[1]
    assert ctype.startswith("multipart/form-data")
    assert body.startswith(f"--{boundary}\r\n".encode()) and body.endswith(f"--{boundary}--\r\n".encode())
    assert b'name="response_format"\r\n\r\njson' in body
    assert b'filename="speech.wav"' in body and b"RIFFdata" in body


# ============================================================================================ Moonshine
class FakeStream:
    def __init__(self, final: str):
        self.final, self.listeners, self.audio, self.closed = final, [], [], False

    def add_listener(self, fn):
        self.listeners.append(fn)

    def start(self):
        pass

    def add_audio(self, samples, sr):
        self.audio.extend(samples)
        for fn in self.listeners:  # one growing line, like moonshine's LineTextChanged
            fn(SimpleNamespace(line=SimpleNamespace(line_id=1, text=f"words {len(self.audio)}")))

    def stop(self):
        return SimpleNamespace(lines=[SimpleNamespace(text=self.final)] if self.final else [])

    def close(self):
        self.closed = True


class FakeTranscriber:
    def __init__(self, final="streamed text", batch="batch text"):
        self.final, self.batch, self.streams, self.batch_calls, self.closed = final, batch, [], 0, False

    def create_stream(self, interval):
        s = FakeStream(self.final)
        self.streams.append(s)
        return s

    def transcribe_without_streaming(self, samples, sr):
        self.batch_calls += 1
        return SimpleNamespace(lines=[SimpleNamespace(text=self.batch)])

    def close(self):
        self.closed = True


def test_moonshine_streams_partials_and_uses_the_streamed_final():
    tr = FakeTranscriber()
    asr = asr_next.MoonshineASR("unused", transcriber=tr)
    seen = []
    asr.on_partial = seen.append
    audio = np.zeros(16000, dtype=np.float32)
    for chunk in np.split(audio, 4):
        asr.push(chunk)
    assert seen[-1] == "words 16000" and len(seen) == 4
    assert asr.transcribe(audio) == "streamed text"
    assert tr.batch_calls == 0 and tr.streams[0].closed


def test_moonshine_transcribes_in_one_go_when_the_stream_missed_the_utterance():
    tr = FakeTranscriber()
    asr = asr_next.MoonshineASR("unused", transcriber=tr)
    asr.push(np.zeros(1000, dtype=np.float32))  # only a sliver was streamed
    assert asr.transcribe(np.zeros(16000, dtype=np.float32)) == "batch text"
    assert tr.batch_calls == 1
    # nothing streamed at all
    assert asr.transcribe(np.zeros(8000, dtype=np.float32)) == "batch text"
    asr.close()
    assert tr.closed


def test_moonshine_falls_back_to_partial_lines_when_stop_has_none():
    tr = FakeTranscriber(final="")
    asr = asr_next.MoonshineASR("unused", transcriber=tr)
    asr.push(np.zeros(16000, dtype=np.float32))
    assert asr.transcribe(np.zeros(16000, dtype=np.float32)) == "words 16000"


# ============================================================================================ fake nemo-speech serve
class FakeNemo:
    """A fake ``nemo-speech serve``: /ready and /v1/audio/transcriptions with the bearer key."""

    def __init__(self, text="hello from nemotron", fail_with: int | None = None, ready_after: int = 0):
        self.text, self.fail_with, self.ready_after = text, fail_with, ready_after
        self.ready_calls = 0
        self.requests: list[dict] = []
        self.key = ""
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/ready":
                    outer.ready_calls += 1
                    self._json(200, {"ready": outer.ready_calls > outer.ready_after})
                else:
                    self._json(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                outer.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "ctype": self.headers.get("Content-Type"),
                        "body": body,
                    }
                )
                if self.headers.get("Authorization") != f"Bearer {outer.key}":
                    self._json(401, {"error": "bad key"})
                elif outer.fail_with:
                    self._json(outer.fail_with, {"error": "CUDA error: out of memory"})
                else:
                    self._json(200, {"text": f"  {outer.text}  "})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeProc:
    def __init__(self, exit_code=None, lines=(b"loading model\n", b"listening\n")):
        self.exit_code = exit_code
        self.stdout = io.BytesIO(b"".join(lines))
        self.terminated = self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


@pytest.fixture
def nemo():
    srv = FakeNemo()
    yield srv
    srv.close()


def _server(fake, proc=None, **kw):
    launched = []

    def popen(cmd, **k):
        launched.append((cmd, k))
        return proc or FakeProc()

    s = asr_next.NemoServer("/engines/nemo-speech", "/models/n.gguf", popen=popen, **kw)
    s.port = fake.port
    fake.key = s.key
    return s, launched


def test_nemo_server_command_is_local_with_a_random_key(nemo):
    s, _ = _server(nemo, device="cuda:0", chunk_ms=1120)
    cmd = s.command()
    assert cmd[:2] == ["/engines/nemo-speech", "serve"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--api-key") + 1] == s.key and len(s.key) >= 24
    assert cmd[cmd.index("--asr.streaming.rnnt_right_context") + 1] == "13"
    assert "--no-ui" in cmd
    other, _ = _server(nemo)
    assert other.key != s.key


def test_nemo_server_starts_waits_for_ready_and_transcribes(nemo):
    nemo.ready_after = 1  # not ready on the first poll
    s, launched = _server(nemo)
    s.start(timeout=10)
    assert launched and launched[0][0][1] == "serve" and nemo.ready_calls >= 2
    text = s.transcribe(np.zeros(1600, dtype=np.float32), prompt="Away")
    assert text == "hello from nemotron"
    req = nemo.requests[-1]
    assert req["path"] == "/v1/audio/transcriptions" and req["auth"] == f"Bearer {s.key}"
    assert b'name="prompt"\r\n\r\nAway' in req["body"] and b"RIFF" in req["body"]
    time.sleep(0.05)
    assert "listening" in s.tail()
    proc = s.proc
    s.stop()
    assert proc.terminated and s.proc is None
    s.stop()  # twice is fine


def test_nemo_server_start_fails_plainly_when_the_engine_exits(nemo):
    s, _ = _server(nemo, proc=FakeProc(exit_code=3, lines=(b"CUDA driver version is insufficient\n",)))
    with pytest.raises(RuntimeError, match="stopped while starting"):
        s.start(timeout=5)


def test_nemo_server_start_times_out(nemo):
    nemo.ready_after = 10**9
    s, _ = _server(nemo)
    t = iter([0.0, 0.0, 100.0, 100.0])
    with pytest.raises(RuntimeError, match="did not get ready"):
        s.start(timeout=1, clock=lambda: next(t))
    assert s.proc is None  # stopped again


def test_nemotron_asr_without_partials_posts_one_request(nemo):
    s, _ = _server(nemo)
    asr = asr_next.NemotronASR(s, partials=False)
    asr.on_partial = lambda t: None
    asr.push(np.zeros(1600, dtype=np.float32))  # ignored: partials off
    assert asr.transcribe(np.zeros(16000, dtype=np.float32)) == "hello from nemotron"
    assert len(nemo.requests) == 1
    assert asr.device == "cuda"
    asr.close()


def test_nemotron_server_error_falls_back_to_whisper(nemo):
    """A CUDA failure mid-dictation: the fallback wrapper switches to Whisper once and says why."""
    nemo.fail_with = 500
    s, _ = _server(nemo)
    primary = asr_next.NemotronASR(s, partials=False)
    whisper = SimpleNamespace(transcribe=lambda a, p=None: "whisper text", device="cpu", last_confidence=0.9)
    built, notes = [], []

    def build():
        built.append(1)
        return whisper

    asr = asr_next.NextFallbackASR(primary, build, notes.append, "Nemotron speech (experimental)")
    assert asr.transcribe(np.zeros(16000, dtype=np.float32)) == "whisper text"
    assert "Nemotron speech (experimental) stopped working" in notes[0] and "using Whisper instead" in notes[0]
    assert asr.transcribe(np.zeros(16000, dtype=np.float32)) == "whisper text"
    assert built == [1] and len(nemo.requests) == 1  # Nemotron is not asked again this session
    assert asr.device == "cpu" and asr.last_confidence == 0.9


# ============================================================================================ realtime partials
class FakeRealtimeWS:
    """Stands in for ws_client.WebSocket: answers a commit with the server's completed event."""

    instances: ClassVar[list[FakeRealtimeWS]] = []

    def __init__(self, url, *a, **k):
        self.url, self.sent, self.binary, self.closed = url, [], 0, False
        self.inbox: queue.Queue = queue.Queue()
        FakeRealtimeWS.instances.append(self)

    def send_text(self, text):
        ev = json.loads(text)
        self.sent.append(ev)
        if ev["type"] == "input_audio_buffer.commit":
            self.inbox.put(
                json.dumps(
                    {"type": "conversation.item.input_audio_transcription.completed", "transcript": "streamed final"}
                )
            )

    def send_binary(self, data):
        self.binary += len(data)
        self.inbox.put(json.dumps({"type": "conversation.item.input_audio_transcription.delta", "delta": "word "}))

    def recv(self, timeout=None):
        try:
            return self.inbox.get(timeout=timeout or 5)
        except queue.Empty:
            return None

    def close(self):
        self.closed = True
        self.inbox.put(None)


def test_realtime_stream_events_build_partials():
    seen = []
    ws = FakeRealtimeWS("x")
    st = asr_next.RealtimeStream(SimpleNamespace(port=1, key="k"), seen.append, ws=ws)
    assert ws.sent[0] == {"type": "session.update", "session": {"sample_rate": 16000}}
    st.handle({"type": "x.input_audio_transcription.delta", "delta": "hel"})
    st.handle({"type": "x.input_audio_transcription.delta", "delta": "lo"})
    assert seen[-1] == "hello"
    st.handle({"type": "x.input_audio_transcription.delta", "delta": "Hello", "replace": True})
    assert st.partial == "Hello"
    st.handle({"type": "x.input_audio_transcription.completed", "transcript": "Hello there."})
    assert st.finals == ["Hello there."] and st.partial == "" and st.done.is_set()
    st.handle({"type": "unrelated"})
    assert seen[-1] == "Hello there."
    st.close()
    assert ws.closed and ws.sent[-1] == {"type": "input_audio_buffer.commit"}


def test_nemotron_uses_the_live_streams_final_when_it_covered_the_utterance(nemo, monkeypatch):
    FakeRealtimeWS.instances.clear()
    monkeypatch.setattr(ws_client, "WebSocket", FakeRealtimeWS)
    s, _ = _server(nemo)
    asr = asr_next.NemotronASR(s, partials=True)
    seen = []
    asr.on_partial = seen.append
    audio = np.zeros(16000, dtype=np.float32)
    for c in np.split(audio, 4):
        asr.push(c)
    assert asr.transcribe(audio) == "streamed final"
    assert nemo.requests == []  # no second, one-shot request was needed
    assert seen and "word" in seen[0]
    ws = FakeRealtimeWS.instances[0]
    assert f"api_key={s.key}" in ws.url and ws.binary == 16000 * 2
    asr.close()


def test_nemotron_short_stream_uses_one_request(nemo, monkeypatch):
    monkeypatch.setattr(ws_client, "WebSocket", FakeRealtimeWS)
    s, _ = _server(nemo)
    asr = asr_next.NemotronASR(s, partials=True)
    asr.on_partial = lambda t: None
    asr.push(np.zeros(1000, dtype=np.float32))  # far less than the utterance
    assert asr.transcribe(np.zeros(16000, dtype=np.float32)) == "hello from nemotron"
    assert len(nemo.requests) == 1
    asr.close()


# ============================================================================================ fallback + build
def test_fallback_wrapper_passes_partials_and_push_to_the_primary():
    pushed = []
    primary = SimpleNamespace(
        name="moonshine",
        on_partial=None,
        push=pushed.append,
        reset=lambda: pushed.append("r"),
        transcribe=lambda a, p=None: "ok",
        device="cpu",
        last_confidence=None,
    )
    asr = asr_next.NextFallbackASR(primary, lambda: None)
    fn = print
    asr.on_partial = fn
    assert primary.on_partial is fn and asr.on_partial is fn
    asr.push(1)
    asr.reset()
    assert pushed == [1, "r"] and asr.transcribe(np.zeros(4)) == "ok" and asr.name == "moonshine"


def test_make_next_asr_uses_whisper_with_the_reason_when_the_engine_cannot_start():
    notes = []
    whisper = object()

    def builder(*a):
        raise RuntimeError("the NeMo-Speech engine is not installed")

    got = asr_next.make_next_asr("nemotron", SimpleNamespace(), lambda: whisper, notes.append, builder=builder)
    assert got is whisper
    assert "could not start (the NeMo-Speech engine is not installed); using Whisper instead" in notes[0]
    assert asr_next.EXPERIMENTAL_NOTE in notes[0]

    primary = SimpleNamespace(name="moonshine")
    wrapped = asr_next.make_next_asr("moonshine", SimpleNamespace(), lambda: whisper, builder=lambda *a: primary)
    assert isinstance(wrapped, asr_next.NextFallbackASR) and wrapped.primary is primary
    assert wrapped.label == "Moonshine speech"


def test_nemotron_plan_picks_the_variant_for_this_pc(tmp_path, monkeypatch):
    from away.app import engines

    monkeypatch.setattr(engines, "platform_key", lambda system=None, machine=None: ("win32", "amd64"))
    gpu = asr_next.nemotron_plan("auto", tmp_path, which=lambda n: "C:/nvidia-smi.exe")
    assert (gpu["variant"], gpu["device"], gpu["nvidia_gpu"]) == ("cuda", "cuda:0", True)
    assert gpu["experimental"] is True and gpu["engine_installed"] is False and gpu["engine_size_mb"] > 50
    cpu = asr_next.nemotron_plan("auto", tmp_path, which=lambda n: None)
    assert (cpu["variant"], cpu["device"]) == ("cpu", "cpu")
    vk = asr_next.nemotron_plan("vulkan", tmp_path, which=lambda n: None)
    assert (vk["variant"], vk["device"]) == ("vulkan", "vulkan:0")
    # Linux has no vulkan build: asking for it on an NVIDIA-less Linux PC gives the CPU build
    monkeypatch.setattr(engines, "platform_key", lambda system=None, machine=None: ("linux", "x86_64"))
    assert asr_next.nemotron_plan("vulkan", tmp_path, which=lambda n: None)["variant"] == "cpu"


def test_build_next_says_plainly_what_is_missing(tmp_path):
    from away.config import ModelsNextConfig

    with pytest.raises(RuntimeError, match="not downloaded yet"):
        asr_next.build_next("moonshine", ModelsNextConfig(stt="moonshine"), data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not downloaded yet"):
        asr_next.build_next("nemotron", ModelsNextConfig(stt="nemotron"), data_dir=tmp_path)
    with pytest.raises(ValueError):
        asr_next.build_next("parakeet", ModelsNextConfig(), data_dir=tmp_path)


def test_next_wiring_keeps_whisper_by_default_and_rolls_back_a_failing_model(tmp_path, monkeypatch):
    from away.app import model_guard, next_wiring
    from away.config import AppConfig

    cfg = AppConfig()
    whisper = object()
    assert next_wiring.next_asr(cfg, lambda: whisper) is whisper  # defaults unchanged: Whisper as before

    monkeypatch.setenv("AWAY_DATA_DIR", str(tmp_path))
    model_guard.remember("stt", {"models_next.stt": "whisper"})
    cfg.models_next.stt = "moonshine"  # chosen, but not downloaded -> cannot start
    notices, saved = [], []
    got = next_wiring.next_asr(
        cfg, lambda: whisper, lambda title, msg: notices.append(msg), save=lambda: saved.append(1)
    )
    assert got is whisper
    assert cfg.models_next.stt == "whisper" and saved == [1]  # rolled back and saved
    assert "Moonshine speech could not start" in notices[0] and "switched back" in notices[0]
