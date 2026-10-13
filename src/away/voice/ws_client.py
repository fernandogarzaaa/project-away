"""A tiny blocking WebSocket client (RFC 6455) for talking to a speech server on this PC (v0.1.24).

Only what the NeMo-Speech.cpp realtime transcription endpoint needs: ``ws://127.0.0.1`` (no TLS), text and binary
frames out (masked, as the RFC requires of clients), text frames in, ping/pong and close. Kept in the standard
library so the streaming speech backend needs no extra package.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import struct
from collections.abc import Callable
from urllib.parse import urlsplit

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketError(ConnectionError):
    pass


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def encode_frame(opcode: int, payload: bytes, mask: bytes | None = None) -> bytes:
    """One final frame. Client frames are always masked (``mask`` is 4 random bytes unless given)."""
    m = mask if mask is not None else os.urandom(4)
    n = len(payload)
    head = bytes([0x80 | (opcode & 0x0F)])
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 1 << 16:
        head += bytes([0x80 | 126]) + struct.pack("!H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack("!Q", n)
    body = bytes(b ^ m[i % 4] for i, b in enumerate(payload))
    return head + m + body


def decode_frame(rd: Callable[[int], bytes]) -> tuple[bool, int, bytes]:
    """Read one frame with ``rd(n) -> bytes`` (exactly n bytes). Returns (fin, opcode, payload)."""
    b0, b1 = rd(2)
    fin, op = bool(b0 & 0x80), b0 & 0x0F
    masked, n = bool(b1 & 0x80), b1 & 0x7F
    if n == 126:
        n = struct.unpack("!H", rd(2))[0]
    elif n == 127:
        n = struct.unpack("!Q", rd(8))[0]
    m = rd(4) if masked else b""
    data = rd(n) if n else b""
    if masked:
        data = bytes(b ^ m[i % 4] for i, b in enumerate(data))
    return fin, op, data


class WebSocket:
    def __init__(self, url: str, timeout: float = 10.0, headers: dict[str, str] | None = None) -> None:
        u = urlsplit(url)
        if u.scheme != "ws":
            raise WebSocketError("only ws:// on this computer is supported")
        host, port = u.hostname or "127.0.0.1", u.port or 80
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._buf = b""
        key = base64.b64encode(os.urandom(16)).decode()
        extra = "".join(f"{k}: {v}\r\n" for k, v in (headers or {}).items())
        req = (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n{extra}\r\n"
        )
        self.sock.sendall(req.encode())
        head = self._read_until(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        if not lines or " 101 " not in lines[0] + " ":
            self.close()
            raise WebSocketError(f"the speech server refused the stream ({lines[0] if lines else 'no answer'})")
        hdrs = {ln.split(":", 1)[0].strip().lower(): ln.split(":", 1)[1].strip() for ln in lines[1:] if ":" in ln}
        if hdrs.get("sec-websocket-accept") != accept_key(key):
            self.close()
            raise WebSocketError("the speech server answered with a bad handshake")

    def _read_until(self, marker: bytes) -> bytes:
        while marker not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WebSocketError("connection closed")
            self._buf += chunk
        head, self._buf = self._buf.split(marker, 1)
        return head

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise WebSocketError("connection closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def send_text(self, text: str) -> None:
        self.sock.sendall(encode_frame(OP_TEXT, text.encode("utf-8")))

    def send_binary(self, data: bytes) -> None:
        self.sock.sendall(encode_frame(OP_BIN, data))

    def recv(self, timeout: float | None = None) -> str | bytes | None:
        """The next text/binary message; None when the server closed the stream."""
        self.sock.settimeout(timeout)
        parts: list[bytes] = []
        kind = OP_TEXT
        while True:
            fin, op, data = decode_frame(self._read)
            if op == OP_PING:
                self.sock.sendall(encode_frame(OP_PONG, data))
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                return None
            if op in (OP_TEXT, OP_BIN):
                kind, parts = op, [data]
            else:
                parts.append(data)
            if fin:
                body = b"".join(parts)
                return body.decode("utf-8", "replace") if kind == OP_TEXT else body

    def close(self) -> None:
        try:
            self.sock.sendall(encode_frame(OP_CLOSE, b""))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
