"""iPhone TrueDepth face data from Epic's free *Live Link Face* iOS app (experimental).

The app's **Live Link (ARKit)** capture mode streams one UDP datagram per ARKit frame
(~60 Hz) to every target in *Settings > Live Link* (default port 11111). Wire format,
packet version 6, all multi-byte values **big-endian** (from Unreal's
``AppleARKitLiveLinkSource.cpp``, as decoded by PyLiveLinkFace and others)::

    uint8    packet version (6)
    uint32   device id length,  bytes  device id (utf-8)
    uint32   subject name length, bytes subject name (utf-8)
    int32    frame number
    float32  sub-frame
    uint32   frame rate numerator
    uint32   frame rate denominator
    uint8    value count (61)              } absent when the phone sees no face
    float32  x count: 52 ARKit blendshapes, } (header-only packet)
             head yaw/pitch/roll, left eye yaw/pitch/roll, right eye yaw/pitch/roll

The app's OSC support is remote control only (start/stop recording, slate/take) and carries
no blendshapes, so there is no OSC data path here. The newer *MetaHuman Animator* /
*Live Link (MetaHuman)* modes use a different, undocumented protocol and are not supported:
the phone must be in ARKit mode.

Everything here is standard library + numpy-free; a bad packet is counted and dropped,
never raised out of the listener thread.
"""

from __future__ import annotations

import contextlib
import math
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PACKET_VERSION = 6
DEFAULT_PORT = 11111
MAX_STRING = 1024  # device id / subject name sanity cap (real ones are < 100 bytes)
MAX_DATAGRAM = 4096

# ARKit blendshape order used by Live Link (Unreal's EARFaceBlendShape), camelCase as in
# Apple's ARFaceAnchor.BlendShapeLocation and in MediaPipe's Face Landmarker categories.
ARKIT_BLENDSHAPES: tuple[str, ...] = (
    "eyeBlinkLeft",
    "eyeLookDownLeft",
    "eyeLookInLeft",
    "eyeLookOutLeft",
    "eyeLookUpLeft",
    "eyeSquintLeft",
    "eyeWideLeft",
    "eyeBlinkRight",
    "eyeLookDownRight",
    "eyeLookInRight",
    "eyeLookOutRight",
    "eyeLookUpRight",
    "eyeSquintRight",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
)
POSE_VALUES: tuple[str, ...] = (
    "headYaw",
    "headPitch",
    "headRoll",
    "leftEyeYaw",
    "leftEyePitch",
    "leftEyeRoll",
    "rightEyeYaw",
    "rightEyePitch",
    "rightEyeRoll",
)
VALUE_COUNT = len(ARKIT_BLENDSHAPES) + len(POSE_VALUES)  # 61

# ARKit -> MediaPipe Face Landmarker category names. They are the same camelCase names
# except: ARKit has ``tongueOut`` (MediaPipe has no tongue), MediaPipe has ``_neutral``
# (no ARKit equivalent; nothing in Project Away reads it, so it is not synthesized).
ARKIT_TO_MEDIAPIPE: dict[str, str | None] = {n: n for n in ARKIT_BLENDSHAPES}
ARKIT_TO_MEDIAPIPE["tongueOut"] = None
MEDIAPIPE_ONLY: frozenset[str] = frozenset({"_neutral"})


class LiveLinkError(ValueError):
    """A datagram that is not a usable Live Link Face packet."""


@dataclass(frozen=True)
class LiveLinkFrame:
    version: int
    device_id: str
    subject: str
    frame: int
    sub_frame: float
    fps: int
    denominator: int
    blendshapes: dict[str, float] | None  # ARKit names, clipped to [0, 1]; None = no face this frame
    pose: dict[str, float] = field(default_factory=dict)  # head / eye angles as sent (radians-ish)

    @property
    def has_face(self) -> bool:
        return self.blendshapes is not None


def _clean(v: float, lo: float | None = None, hi: float | None = None) -> float:
    if not math.isfinite(v):
        return 0.0
    if lo is not None and v < lo:
        return lo
    if hi is not None and v > hi:
        return hi
    return float(v)


def parse_packet(data: bytes) -> LiveLinkFrame:
    """Decode one Live Link Face datagram. Raises :class:`LiveLinkError` on anything malformed."""
    n = len(data)
    if n < 1:
        raise LiveLinkError("empty packet")
    version = data[0]
    if version != PACKET_VERSION:
        raise LiveLinkError(f"unsupported Live Link packet version {version} (expected {PACKET_VERSION})")
    off = 1
    strings: list[str] = []
    for what in ("device id", "subject name"):
        if off + 4 > n:
            raise LiveLinkError(f"truncated before {what} length")
        (ln,) = struct.unpack_from(">I", data, off)
        off += 4
        if ln > MAX_STRING:
            raise LiveLinkError(f"{what} length {ln} is implausible")
        if off + ln > n:
            raise LiveLinkError(f"truncated {what}")
        strings.append(data[off : off + ln].decode("utf-8", errors="replace"))
        off += ln
    if off + 16 > n:
        raise LiveLinkError("truncated frame time")
    frame, sub_frame, fps, den = struct.unpack_from(">ifII", data, off)
    off += 16
    head = (version, strings[0], strings[1], int(frame), _clean(sub_frame), int(fps), int(den))
    if off == n:  # header only: the phone is tracking no face right now
        return LiveLinkFrame(*head, blendshapes=None)
    count = data[off]
    off += 1
    if count < len(ARKIT_BLENDSHAPES) or count > VALUE_COUNT:
        raise LiveLinkError(f"value count {count} (expected {VALUE_COUNT})")
    if off + 4 * count > n:
        raise LiveLinkError(f"truncated blendshapes ({(n - off) // 4} of {count} values)")
    vals = struct.unpack_from(f">{count}f", data, off)
    bs = {name: _clean(v, 0.0, 1.0) for name, v in zip(ARKIT_BLENDSHAPES, vals)}
    pose = {name: _clean(v) for name, v in zip(POSE_VALUES, vals[len(ARKIT_BLENDSHAPES) :])}
    return LiveLinkFrame(*head, blendshapes=bs, pose=pose)


def encode_packet(
    blendshapes: dict[str, float] | None,
    pose: dict[str, float] | None = None,
    *,
    device_id: str = "test-device",
    subject: str = "iPhone",
    frame: int = 0,
    sub_frame: float = 0.0,
    fps: int = 60,
    denominator: int = 1,
    version: int = PACKET_VERSION,
) -> bytes:
    """Build a packet in the app's format (tests, simulators). ``None`` blendshapes = header-only/no face."""
    dev, sub = device_id.encode(), subject.encode()
    out = struct.pack(">BI", version, len(dev)) + dev + struct.pack(">I", len(sub)) + sub
    out += struct.pack(">ifII", frame, sub_frame, fps, denominator)
    if blendshapes is None:
        return out
    bs, ps = blendshapes, pose or {}
    vals = [float(bs.get(k, 0.0)) for k in ARKIT_BLENDSHAPES] + [float(ps.get(k, 0.0)) for k in POSE_VALUES]
    return out + struct.pack(f">B{VALUE_COUNT}f", VALUE_COUNT, *vals)


def _swap_side(name: str) -> str:
    if name.endswith("Left"):
        return name[:-4] + "Right"
    if name.endswith("Right"):
        return name[:-5] + "Left"
    return name


def to_mediapipe(blendshapes: dict[str, float], swap_sides: bool = False) -> dict[str, float]:
    """ARKit blendshapes -> the MediaPipe-named dict the face pipeline consumes.

    ARKit's Left/Right are the *person's* left/right. ``swap_sides`` flips every Left/Right pair
    in case they come out mirrored relative to the webcam path (check with ``away iphone compare``).
    """
    out: dict[str, float] = {}
    for name, v in blendshapes.items():
        mp = ARKIT_TO_MEDIAPIPE.get(name, name)
        if mp is None:
            continue
        out[_swap_side(mp) if swap_sides else mp] = float(v)
    return out


def _new_socket() -> socket.socket:  # patched in tests
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class LiveLinkReceiver:
    """Background UDP listener keeping the latest face frame, packet rate and staleness.

    ``handle(data, addr, t)`` is the single entry point for datagrams, so it can be driven
    directly in tests. The thread never raises: bind/socket failures land in ``error``.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        stale_after: float = 0.5,
        timeout: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.host = host
        self.port = port
        self.stale_after = stale_after
        self.timeout = timeout
        self.clock = clock
        self.packets = 0
        self.bad_packets = 0
        self.no_face_packets = 0
        self.last_error: str | None = None  # last malformed-packet reason
        self.error: str | None = None  # fatal listener error (e.g. port in use)
        self.last_sender: str | None = None
        self.subject: str | None = None
        self.seq = 0  # bumps on every good packet
        self._latest: LiveLinkFrame | None = None
        self._latest_t: float | None = None
        self._face_t: float | None = None  # time of the last packet that had a face
        self._times: deque[float] = deque(maxlen=512)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: Any = None

    # -- datagrams -------------------------------------------------------------------
    def handle(self, data: bytes, addr: Any = None, t: float | None = None) -> LiveLinkFrame | None:
        now = self.clock() if t is None else t
        try:
            fr = parse_packet(data)
        except Exception as exc:  # malformed / foreign traffic on the port
            with self._lock:
                self.bad_packets += 1
                self.last_error = str(exc)
            return None
        with self._cond:
            self.packets += 1
            self.seq += 1
            self._times.append(now)
            self._latest, self._latest_t = fr, now
            self.subject = fr.subject
            if addr is not None:
                self.last_sender = str(addr[0]) if isinstance(addr, tuple) and addr else str(addr)
            if fr.has_face:
                self._face_t = now
            else:
                self.no_face_packets += 1
            self._cond.notify_all()
        return fr

    # -- state -----------------------------------------------------------------------
    def latest(self) -> tuple[float, LiveLinkFrame] | None:
        with self._lock:
            if self._latest is None or self._latest_t is None:
                return None
            return self._latest_t, self._latest

    def latest_face(self, now: float | None = None) -> tuple[float, dict[str, float]] | None:
        """(time, ARKit blendshapes) of the newest frame if it has a face and is fresh, else None."""
        now = self.clock() if now is None else now
        with self._lock:
            fr, t = self._latest, self._latest_t
        if fr is None or t is None or fr.blendshapes is None or now - t > self.stale_after:
            return None
        return t, dict(fr.blendshapes)

    def is_fresh(self, now: float | None = None) -> bool:
        """A face frame arrived within ``stale_after`` seconds (and the newest packet still has a face)."""
        return self.latest_face(now) is not None

    def age(self, now: float | None = None) -> float | None:
        now = self.clock() if now is None else now
        with self._lock:
            return None if self._latest_t is None else now - self._latest_t

    def rate(self, now: float | None = None, window: float = 1.0) -> float:
        """Packets per second over the last ``window`` seconds."""
        now = self.clock() if now is None else now
        with self._lock:
            n = sum(1 for x in self._times if now - window < x <= now)
        return n / window

    def wait(self, after_seq: int, timeout: float) -> int:
        """Block until a packet newer than ``after_seq`` arrives (or ``timeout``). Returns the current seq."""
        with self._cond:
            if self.seq <= after_seq:
                self._cond.wait(timeout)
            return self.seq

    # -- thread ----------------------------------------------------------------------
    def open(self) -> None:
        sock = _new_socket()
        with contextlib.suppress(OSError, AttributeError):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(self.timeout)
        self._sock = sock

    def poll_once(self) -> bool:
        """Receive at most one datagram. Returns False only when the socket is gone."""
        sock = self._sock
        if sock is None:
            return False
        try:
            data, addr = sock.recvfrom(MAX_DATAGRAM)
        except (TimeoutError, BlockingIOError, InterruptedError):
            return True
        except ConnectionResetError:  # Windows WSAECONNRESET on UDP after an ICMP port-unreachable
            return True
        except OSError as exc:
            if self._stop.is_set():
                return False
            with self._lock:
                self.last_error = f"socket: {exc}"
            return True
        self.handle(data, addr)
        return True

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not self.poll_once():
                    break
        except Exception as exc:  # never take the app down
            self.error = f"listener stopped: {exc}"

    def start(self) -> bool:
        """Bind and start the background thread. Returns False (and sets ``error``) if the port can't be bound."""
        if self._thread is not None:
            return self.error is None
        try:
            self.open()
        except Exception as exc:
            self.error = f"cannot listen on UDP {self.host}:{self.port}: {exc}"
            return False
        self._thread = threading.Thread(target=self._run, daemon=True, name="iphone-livelink")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> LiveLinkReceiver:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def lan_ips() -> list[str]:
    """This PC's IPv4 addresses that a phone on the same Wi-Fi could reach (best effort, no traffic sent)."""
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))  # UDP connect sends nothing; it only picks the outgoing interface
            ips.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.append(str(info[4][0]))
    except OSError:
        pass
    out: list[str] = []
    for ip in ips:
        if ip.startswith(("127.", "169.254.", "0.")) or ip in out:
            continue
        out.append(ip)
    return out
