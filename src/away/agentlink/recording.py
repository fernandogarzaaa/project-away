"""Session recordings: hand landmarks and app events, never video or audio.

A ``.awayrec`` file is gzip-compressed JSON lines:

* line 1, the header: ``{"format": "away-rec", "version": 1, "app": "0.1.20", "created": ..., "config": {...},
  "screen": [w, h], "paused": false, "note": "..."}`` (the config is what the replay runs with);
* then one record per line: ``{"k": "h", "t": 12.345, "lm": [[x, y, z] x21] | null, "hd": "Right", ...}`` for
  every frame the gesture pipeline saw (``Pipeline.on_hand`` arguments), and ``{"k": "e", "t": ..., "topic":
  ..., "data": {...}}`` for app events.

Coordinates are rounded to 5 decimals (MediaPipe's own precision is lower), which with gzip makes 30 s at 30 fps
about 100-200 KB. :class:`RingRecorder` keeps the last N seconds in memory (for bundles and auto-detectors) and
can record forward into a file.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

FORMAT = "away-rec"
VERSION = 1
SUFFIX = ".awayrec"
HAND_KEYS = ("pinch", "depth", "pointer_id", "helper_id", "promoted")


def _pts(a: Any) -> list[list[float]] | None:
    if a is None:
        return None
    try:
        return [[round(float(v), 5) for v in p[:3]] for p in a]
    except Exception:
        return None


def hand_record(t: float, landmarks: Any, handedness: str = "Right", **kw: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {"k": "h", "t": round(float(t), 5), "lm": _pts(landmarks), "hd": str(handedness)}
    if kw.get("helper") is not None:
        rec["helper"] = _pts(kw["helper"])
    for k in ("points3d", "helper_points3d"):
        if kw.get(k) is not None:
            rec[k] = _pts(kw[k])
    for k in HAND_KEYS:
        v = kw.get(k)
        if v is not None and v is not False:
            rec[k] = round(float(v), 5) if isinstance(v, float) else v
    return rec


def event_record(t: float, topic: str, data: dict[str, Any]) -> dict[str, Any]:
    from away.agentlink.telemetry import _jsonable

    return {"k": "e", "t": round(float(t), 5), "topic": topic, "data": _jsonable(data)}


def make_header(cfg: Any = None, screen: tuple[int, int] = (1920, 1080), paused: bool = False, note: str = "") -> dict:
    from away import __version__
    from away import config as cfgmod

    conf = cfgmod.to_dict(cfg) if cfg is not None else cfgmod.to_dict(cfgmod.AppConfig())
    return {
        "format": FORMAT,
        "version": VERSION,
        "app": __version__,
        "created": time.time(),
        "config": conf,
        "screen": [int(screen[0]), int(screen[1])],
        "paused": bool(paused),
        "note": note,
    }


def write(path: Path, header: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: the same recording always produces the same bytes
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write((json.dumps(header, separators=(",", ":")) + "\n").encode("utf-8"))
        for r in records:
            gz.write((json.dumps(r, separators=(",", ":")) + "\n").encode("utf-8"))
    return path


def read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        first = fh.readline()
        header = json.loads(first) if first else {}
        if header.get("format") != FORMAT:
            raise ValueError(f"{path} is not a Project Away recording")
        if int(header.get("version", 0)) > VERSION:
            raise ValueError(f"{path} was written by a newer version (format {header.get('version')})")
        records = [json.loads(line) for line in fh if line.strip()]
    return header, records


class RingRecorder:
    """The last ``seconds`` of hand frames + events in memory; optionally also recording forward to a file."""

    def __init__(self, seconds: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = float(seconds)
        self.clock = clock
        self._buf: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._forward: list[dict[str, Any]] | None = None
        self._forward_until = 0.0
        self.enabled = True

    def add(self, rec: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._buf.append(rec)
            cutoff = float(rec["t"]) - self.seconds
            while self._buf and float(self._buf[0]["t"]) < cutoff:
                self._buf.popleft()
            if self._forward is not None:
                self._forward.append(rec)
                if len(self._forward) > 200_000:  # ~2 h at 30 fps: a hard ceiling
                    self._forward_until = 0.0

    def on_hand(self, t: float, landmarks: Any, handedness: str = "Right", **kw: Any) -> None:
        self.add(hand_record(t, landmarks, handedness, **kw))

    def on_event(self, t: float, topic: str, data: dict[str, Any]) -> None:
        if topic in ("gesture.move", "gesture.scroll", "gesture.autoscroll", "gesture.palm_scroll", "wifi.state"):
            return  # re-derived by the replay; too frequent to store
        self.add(event_record(t, topic, data))

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buf)

    def start_forward(self, seconds: float) -> None:
        with self._lock:
            self._forward = []
            self._forward_until = self.clock() + float(seconds)

    def forward_done(self) -> bool:
        return self._forward is not None and self.clock() >= self._forward_until

    def take_forward(self) -> list[dict[str, Any]]:
        with self._lock:
            out, self._forward = self._forward or [], None
        return out


def iter_hands(records: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    return (r for r in records if r.get("k") == "h")
