"""Live numbers for agents and the Settings health panel: fps per camera, end-to-end latency, CPU/GPU, the
latest landmarks and gestures, a log tail, and the user's live context (present, facing the screen, typing,
dictating, active app).

Everything is fed by the app (``on_frame`` / ``on_faces`` / ``on_event`` / ``note_key``) and read by the
service; nothing here touches a device, so it is fully testable with an injected clock.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

QUIET_TOPICS = ("gesture.move", "gesture.scroll", "gesture.autoscroll", "gesture.palm_scroll", "wifi.state")
PRESENT_HAND_S = 3.0
PRESENT_FACE_S = 5.0
AWAY_AFTER_S = 10.0  # nothing seen this long: the user left
FACING_S = 2.0
TYPING_S = 2.0


def _round(x: float, n: int = 4) -> float:
    return float(round(float(x), n))


class CpuSampler:
    """Process CPU % (all platforms, from ``time.process_time``) and system CPU % (Linux /proc/stat, Windows
    GetSystemTimes; None elsewhere)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._last = (clock(), time.process_time())
        self._sys_last = self._system_times()
        self.process_pct: float | None = None
        self.system_pct: float | None = None

    @staticmethod
    def _system_times() -> tuple[float, float] | None:
        """(idle, total) since boot, or None."""
        try:
            if sys.platform.startswith("linux"):
                with open("/proc/stat", encoding="ascii") as fh:
                    vals = [float(v) for v in fh.readline().split()[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
                return idle, sum(vals)
            if sys.platform == "win32":  # pragma: no cover - Windows only
                import ctypes
                from ctypes import wintypes

                idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
                if not ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)
                ):
                    return None

                def v(ft: Any) -> float:
                    return float((ft.dwHighDateTime << 32) | ft.dwLowDateTime)

                return v(idle), v(kern) + v(user)  # kernel time includes idle time
        except Exception:
            return None
        return None

    def sample(self) -> tuple[float | None, float | None]:
        now, cpu = self.clock(), time.process_time()
        dt = now - self._last[0]
        if dt > 0.2:
            self.process_pct = _round(100.0 * (cpu - self._last[1]) / dt / max(1, os.cpu_count() or 1), 1)
            self._last = (now, cpu)
            cur = self._system_times()
            if cur is not None and self._sys_last is not None and cur[1] > self._sys_last[1]:
                d_idle, d_total = cur[0] - self._sys_last[0], cur[1] - self._sys_last[1]
                self.system_pct = _round(100.0 * (1.0 - d_idle / d_total), 1)
            self._sys_last = cur
        return self.process_pct, self.system_pct


class GpuSampler:
    """NVIDIA GPU utilisation / memory from ``nvidia-smi`` (fixed arguments, no shell), at most every 5 s.
    None when there is no NVIDIA driver."""

    def __init__(self, every_s: float = 5.0, clock: Callable[[], float] = time.monotonic, runner: Any = None) -> None:
        self.every_s, self.clock = every_s, clock
        self.exe = shutil.which("nvidia-smi")
        self.runner = runner or subprocess.run
        self._next = 0.0
        self.last: dict[str, Any] | None = None

    def sample(self) -> dict[str, Any] | None:
        if self.exe is None and self.runner is subprocess.run:
            return None
        now = self.clock()
        if now < self._next:
            return self.last
        self._next = now + self.every_s
        try:
            kw: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 3}
            if sys.platform == "win32":  # pragma: no cover
                kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            out = self.runner(
                [
                    self.exe or "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                **kw,
            )
            name, util, used, total = [s.strip() for s in str(out.stdout).splitlines()[0].split(",")[:4]]
            self.last = {
                "name": name,
                "util_pct": float(util),
                "mem_used_mb": float(used),
                "mem_total_mb": float(total),
            }
        except Exception:
            self.last = None
        return self.last


class Telemetry:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        cpu: CpuSampler | None = None,
        gpu: GpuSampler | None = None,
        log_lines: int = 400,
    ) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self.frames: dict[str, deque[float]] = {}
        self.latency_ms: deque[float] = deque(maxlen=300)
        self.landmarks: dict[str, Any] = {}
        self.gestures: deque[dict[str, Any]] = deque(maxlen=200)
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self.log: deque[str] = deque(maxlen=log_lines)
        self.seq = 0
        self.cpu = cpu if cpu is not None else CpuSampler(clock)
        self.gpu = gpu if gpu is not None else GpuSampler(clock=clock)
        self.models: dict[str, Any] = {}
        self.last_hand_t = -1e9
        self.last_face_t = -1e9
        self.last_frontal_t = -1e9
        self.face_seen_ever = False
        self.last_key_t = -1e9
        self.last_dictation_t = -1e9
        self.voice_state: Callable[[], str] = lambda: "off"
        self.active_app: Callable[[], str | None] = lambda: None
        self.paused: Callable[[], bool] = lambda: False
        self.extensions: dict[str, Callable[[], Any]] = {}  # e.g. "fatigue" from the attention branch
        self._present = False
        self._presence_changed = clock()
        self._attention = "unknown"
        self.on_change: Callable[[str, dict[str, Any]], None] | None = None  # presence / attention transitions

    # ------------------------------------------------------------------ inputs
    def on_frame(self, cam: Any, t: float, landmarks: Any = None, n_hands: int = 0, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        key = str(cam)
        with self._lock:
            dq = self.frames.setdefault(key, deque(maxlen=120))
            dq.append(now)
            if t:
                self.latency_ms.append(max(0.0, (now - t) * 1000.0))
            if landmarks is not None:
                try:
                    pts = [[_round(v) for v in p[:3]] for p in landmarks]
                except Exception:
                    pts = None
                self.landmarks[key] = {"t": now, "points": pts, "n_hands": n_hands}
                self.last_hand_t = now
            elif key in self.landmarks:
                self.landmarks[key] = {"t": now, "points": None, "n_hands": n_hands}

    def on_faces(self, n_faces: int, frontal: bool | None, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        if n_faces > 0:
            self.last_face_t = now
            self.face_seen_ever = True
            if frontal:
                self.last_frontal_t = now
        self._update_context(now)

    def note_key(self, now: float | None = None) -> None:
        """The user pressed a physical key (no key value is ever recorded)."""
        self.last_key_t = self.clock() if now is None else now

    def on_event(self, topic: str, data: dict[str, Any], now: float | None = None) -> None:
        now = self.clock() if now is None else now
        with self._lock:
            self.seq += 1
            item = {"seq": self.seq, "t": now, "topic": topic, "data": _jsonable(data)}
            if topic.startswith("gesture.") and topic not in QUIET_TOPICS:
                self.gestures.append(item)
            if topic not in QUIET_TOPICS:
                self.events.append(item)
                self.log.append(f"{time.strftime('%H:%M:%S')} [{topic}] {item['data']}")
        if topic.startswith(("dictation.", "voice.utterance")):
            self.last_dictation_t = now

    def add_log(self, line: str) -> None:
        with self._lock:
            self.log.append(line.rstrip())

    # ----------------------------------------------------------------- outputs
    def fps(self, now: float | None = None) -> dict[str, float]:
        now = self.clock() if now is None else now
        out = {}
        with self._lock:
            for k, dq in self.frames.items():
                recent = [t for t in dq if now - t <= 2.0]
                out[k] = _round(len(recent) / 2.0, 1) if len(recent) >= 2 else 0.0
        return out

    def latency(self) -> dict[str, float | None]:
        with self._lock:
            vals = sorted(self.latency_ms)
        if not vals:
            return {"p50_ms": None, "p95_ms": None, "n": 0}
        return {
            "p50_ms": _round(vals[len(vals) // 2], 1),
            "p95_ms": _round(vals[min(len(vals) - 1, int(len(vals) * 0.95))], 1),
            "n": len(vals),
        }

    def health(self) -> dict[str, Any]:
        proc, system = self.cpu.sample()
        return {
            "fps": self.fps(),
            "latency": self.latency(),
            "cpu": {"process_pct": proc, "system_pct": system},
            "gpu": self.gpu.sample(),
            "models": dict(self.models),
        }

    def _update_context(self, now: float) -> None:
        seen = max(self.last_hand_t, self.last_face_t)
        present = (now - self.last_hand_t <= PRESENT_HAND_S) or (now - self.last_face_t <= PRESENT_FACE_S)
        if self._present and not present and now - seen < AWAY_AFTER_S:
            present = True  # debounce: "left" only after AWAY_AFTER_S without seeing anything
        if present != self._present:
            self._present = present
            self._presence_changed = now
            if self.on_change:
                self.on_change("presence", {"present": present, "event": "sat_down" if present else "left"})
        if not present:
            att = "away"
        elif not self.face_seen_ever:
            att = "unknown"
        elif now - self.last_frontal_t <= FACING_S:
            att = "facing_screen"
        else:
            att = "looking_away"
        if att != self._attention:
            old, self._attention = self._attention, att
            if self.on_change:
                self.on_change("attention", {"attention": att, "was": old})

    def context(self, now: float | None = None) -> dict[str, Any]:
        now = self.clock() if now is None else now
        self._update_context(now)
        vs = "off"
        with contextlib.suppress(Exception):
            vs = str(self.voice_state())
        app = None
        with contextlib.suppress(Exception):
            app = self.active_app()
        ext: dict[str, Any] = {}
        for name, fn in self.extensions.items():
            try:
                ext[name] = fn()
            except Exception as exc:
                ext[name] = f"unavailable: {exc}"
        return {
            "present": self._present,
            "attention": self._attention,
            "typing": now - self.last_key_t <= TYPING_S,
            "dictating": vs == "recording" or now - self.last_dictation_t <= 1.5,
            "voice": vs,
            "active_app": app,
            "gestures_paused": bool(self._safe(self.paused)),
            "hand_visible": now - self.last_hand_t <= 1.0,
            "fatigue_mode": ext.pop("fatigue", None),  # None until the attention/comfort feature is merged
            "extensions": ext,
            "since_presence_change_s": _round(now - self._presence_changed, 1),
        }

    @staticmethod
    def _safe(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception:
            return None

    def polite_moment(self, now: float | None = None) -> bool:
        """A good time to show a prompt: present, facing the screen (or face tracking off) and not typing."""
        c = self.context(now)
        return bool(c["present"]) and c["attention"] in ("facing_screen", "unknown") and not c["typing"]

    def poll(self, since: int = 0, limit: int = 200) -> dict[str, Any]:
        with self._lock:
            evs = [e for e in self.events if e["seq"] > since][-limit:]
            seq = self.seq
            lms = dict(self.landmarks)
        return {"seq": seq, "events": evs, "landmarks": lms}

    def log_tail(self, lines: int = 100) -> list[str]:
        with self._lock:
            return list(self.log)[-max(1, min(int(lines), len(self.log) or 1)) :]


def _jsonable(x: Any, depth: int = 0) -> Any:
    if depth > 4:
        return str(x)
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return _round(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(x.items())[:50]}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in list(x)[:100]]
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist(), depth + 1)
    return str(x)
