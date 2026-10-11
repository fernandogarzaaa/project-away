"""Monitor enumeration and multi-monitor cursor geometry.

Every rectangle is in **OS mouse coordinates**: the same space ``pynput`` moves the pointer in.
The virtual desktop is the bounding box of all monitors; it can start at a negative x or y when a
monitor sits left of or above the primary one (Windows and macOS put the primary at ``(0, 0)``).

* :func:`enumerate_monitors` asks the OS (Windows ``EnumDisplayMonitors``; macOS Quartz, then AppKit;
  Linux ``screeninfo``, then ``xrandr --listmonitors``; then tkinter, primary only) and never raises:
  when everything fails it returns the old primary-only behaviour and says why in ``Layout.note``.
* :func:`enable_dpi_awareness` (Windows) makes the process per-monitor DPI aware *before* pynput or
  tkinter touch the screen, so monitor rectangles and ``SetCursorPos`` (what pynput calls) both use
  physical pixels. pynput 1.8 does not set DPI awareness itself, so this is the only place it is set.
* The rest is pure geometry (union clamp, nearest monitor, selection) and is unit-tested headless.

Monitors are numbered 1..N left to right, then top to bottom (ties): ``away monitors`` prints that
numbering, and ``hands.cursor.monitors: [1, 3]`` and ``cursor:monitor 2`` use it.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("away.monitors")

FALLBACK_SIZE = (1920, 1080)


@dataclass(frozen=True)
class Monitor:
    left: int
    top: int
    width: int
    height: int
    primary: bool = False
    name: str = ""

    @property
    def right(self) -> int:  # exclusive
        return self.left + self.width

    @property
    def bottom(self) -> int:  # exclusive
        return self.top + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width / 2.0, self.top + self.height / 2.0

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right - 1 and self.top <= y <= self.bottom - 1

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, float(self.left)), float(self.right - 1)),
            min(max(y, float(self.top)), float(self.bottom - 1)),
        )

    def describe(self) -> str:
        flag = " (primary)" if self.primary else ""
        name = f" {self.name}" if self.name else ""
        return f"{self.width}x{self.height} at ({self.left}, {self.top}){flag}{name}"


@dataclass(frozen=True)
class Layout:
    """The detected monitors (sorted and numbered 1..N) plus where the list came from."""

    monitors: tuple[Monitor, ...]
    source: str = "fallback"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.monitors:
            raise ValueError("a layout needs at least one monitor")

    @property
    def primary(self) -> Monitor:
        return next((m for m in self.monitors if m.primary), self.monitors[0])

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return virtual_bounds(self.monitors)

    def key(self) -> tuple[tuple[int, int, int, int, bool], ...]:
        """Geometry only (names can change spelling between backends); used to detect a change."""
        return tuple((m.left, m.top, m.width, m.height, m.primary) for m in self.monitors)

    def lines(self) -> list[str]:
        n = len(self.monitors)
        out = [f"{n} monitor{'s' if n != 1 else ''} (source: {self.source})"]
        for i, m in enumerate(self.monitors, 1):
            out.append(f"  {i}: {m.describe()}")
        x, y, w, h = self.bounds
        out.append(f"  virtual desktop: {w}x{h} at ({x}, {y})")
        if self.note:
            out.append(f"  note: {self.note}")
        return out

    def summary(self) -> str:
        return f"{len(self.monitors)} via {self.source}: " + "; ".join(
            f"{i}: {m.describe()}" for i, m in enumerate(self.monitors, 1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "note": self.note,
            "bounds": list(self.bounds),
            "monitors": [
                {
                    "index": i,
                    "left": m.left,
                    "top": m.top,
                    "width": m.width,
                    "height": m.height,
                    "primary": m.primary,
                    "name": m.name,
                }
                for i, m in enumerate(self.monitors, 1)
            ],
        }


# ----------------------------------------------------------------------------------------- geometry
def sort_monitors(mons: Iterable[Monitor]) -> tuple[Monitor, ...]:
    """Left to right, then top to bottom: the numbering users see. Drops empty and duplicate rects
    (a mirrored display reports the same rectangle twice)."""
    seen: set[tuple[int, int, int, int]] = set()
    out: list[Monitor] = []
    for m in sorted(mons, key=lambda m: (m.left, m.top, not m.primary)):
        r = (m.left, m.top, m.width, m.height)
        if m.width <= 0 or m.height <= 0 or r in seen:
            continue
        seen.add(r)
        out.append(m)
    if out and not any(m.primary for m in out):
        # the monitor holding the origin is the primary on Windows/macOS; else the first one
        idx = next((i for i, m in enumerate(out) if m.contains(0, 0)), 0)
        o = out[idx]
        out[idx] = Monitor(o.left, o.top, o.width, o.height, True, o.name)
    return tuple(out)


def virtual_bounds(mons: Sequence[Monitor]) -> tuple[int, int, int, int]:
    """(left, top, width, height) of the bounding box of every monitor; origin may be negative."""
    x0 = min(m.left for m in mons)
    y0 = min(m.top for m in mons)
    x1 = max(m.right for m in mons)
    y1 = max(m.bottom for m in mons)
    return x0, y0, x1 - x0, y1 - y0


def nearest_monitor(x: float, y: float, mons: Sequence[Monitor]) -> int:
    """Index (0-based) of the monitor containing the point, else the closest one."""
    best, best_d = 0, math.inf
    for i, m in enumerate(mons):
        cx, cy = m.clamp(x, y)
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best, best_d = i, d
            if d == 0.0:
                break
    return best


def clamp_to_monitors(x: float, y: float, mons: Sequence[Monitor]) -> tuple[float, float]:
    """The nearest point that lies on some monitor, so the cursor never sits in a dead zone between
    monitors of different sizes or offsets (which the OS would refuse anyway)."""
    return mons[nearest_monitor(x, y, mons)].clamp(x, y)


def select_monitors(layout: Layout, spec: Any) -> tuple[Monitor, ...]:
    """Apply ``hands.cursor.monitors`` (``all`` | ``primary`` | list of 1-based numbers). Numbers that
    are not connected right now are skipped; if none is left, all monitors are used (and logged)."""
    mons = layout.monitors
    if spec in (None, "all", ""):
        return mons
    if spec == "primary":
        return (layout.primary,)
    if isinstance(spec, (list, tuple)):
        picked = tuple(mons[i - 1] for i in spec if isinstance(i, int) and 1 <= i <= len(mons))
        if picked:
            return picked
        log.warning("hands.cursor.monitors %r: none of those monitors is connected; using all", spec)
    return mons


def parse_monitor_spec(value: Any, path: str = "hands.cursor.monitors") -> str | list[int]:
    """Validate the config value; returns ``"all"``, ``"primary"`` or a list of 1-based numbers."""
    if value is None:
        return "all"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("all", "primary"):
            return v
        parts = [p for p in re.split(r"[,\s]+", v) if p]
        if parts and all(p.isdigit() for p in parts):
            value = [int(p) for p in parts]
        else:
            raise ValueError(f"{path} must be all, primary or a list of monitor numbers (1, 2, ...), got {value!r}")
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for i in value:
            if isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= 16:
                raise ValueError(f"{path}: monitor numbers start at 1 (see `away monitors`), got {i!r}")
            if i not in out:
                out.append(i)
        if not out:
            raise ValueError(f"{path}: the list must name at least one monitor (or use all)")
        return out
    raise ValueError(f"{path} must be all, primary or a list of monitor numbers, got {value!r}")


def single(width: int, height: int, source: str = "fallback", note: str = "") -> Layout:
    return Layout((Monitor(0, 0, int(width), int(height), True, ""),), source, note)


# ------------------------------------------------------------------------------------ DPI awareness
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
PROCESS_PER_MONITOR_DPI_AWARE = 2
E_ACCESSDENIED = -2147024891  # 0x80070005 as a signed HRESULT: awareness was already set
_dpi_mode: str | None = None


def enable_dpi_awareness(
    user32: Any = None, shcore: Any = None, platform: str | None = None, force: bool = False
) -> str:
    """Make this process per-monitor DPI aware (Windows only; once per process; never raises).

    Tries ``SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)`` (Windows 10 1703+), then
    ``SetProcessDpiAwareness(PER_MONITOR)`` (8.1+), then ``SetProcessDPIAware`` (Vista+, system DPI).
    Returns ``per_monitor_v2`` | ``per_monitor`` | ``system`` | ``already`` (set earlier, e.g. by an exe
    manifest) | ``disabled`` (``AWAY_DPI_AWARE=0``) | ``unsupported`` (not Windows) | ``failed``.
    """
    global _dpi_mode
    plat = platform or sys.platform
    if _dpi_mode is not None and not force:
        return _dpi_mode
    if plat != "win32":
        _dpi_mode = "unsupported"
        return _dpi_mode
    if os.environ.get("AWAY_DPI_AWARE", "1").strip().lower() in ("0", "false", "no", "off"):
        _dpi_mode = "disabled"
        return _dpi_mode
    mode = "failed"
    try:
        import ctypes

        u32 = user32
        if u32 is None:  # pragma: no cover - Windows only
            u32 = ctypes.windll.user32  # type: ignore[attr-defined]
        fn = getattr(u32, "SetProcessDpiAwarenessContext", None)
        # fails (ERROR_ACCESS_DENIED) when awareness was already set; shcore below then reports that
        if fn is not None and fn(ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)):
            mode = "per_monitor_v2"
        if mode == "failed":
            sh = shcore
            if sh is None:  # pragma: no cover - Windows 8.1+
                try:
                    sh = ctypes.windll.shcore  # type: ignore[attr-defined]
                except (AttributeError, OSError):
                    sh = None
            if sh is not None and hasattr(sh, "SetProcessDpiAwareness"):
                hr = int(sh.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE))
                if hr == 0:
                    mode = "per_monitor"
                elif hr in (E_ACCESSDENIED, E_ACCESSDENIED & 0xFFFFFFFF):
                    mode = "already"
        if mode == "failed" and hasattr(u32, "SetProcessDPIAware") and u32.SetProcessDPIAware():
            mode = "system"
    except Exception as exc:  # never stop the app over DPI
        log.warning("DPI awareness could not be set: %s", exc)
        mode = "failed"
    _dpi_mode = mode
    return mode


# ------------------------------------------------------------------------------------- enumeration
def _win32(user32: Any = None) -> list[Monitor]:
    """``EnumDisplayMonitors`` + ``GetMonitorInfoW``. ``user32`` is injectable for tests."""
    import ctypes
    from ctypes import wintypes

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    u32 = user32
    if u32 is None:  # pragma: no cover - Windows only
        u32 = ctypes.windll.user32  # type: ignore[attr-defined]
    functype = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
    proc_t = functype(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(wintypes.RECT), ctypes.c_ssize_t)
    handles: list[Any] = []

    def _cb(hmon: Any, _hdc: Any, _rect: Any, _data: Any) -> int:
        handles.append(hmon)
        return 1

    cb = proc_t(_cb)
    if not u32.EnumDisplayMonitors(None, None, cb, 0):
        raise OSError("EnumDisplayMonitors failed")
    out: list[Monitor] = []
    for h in handles:
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if not u32.GetMonitorInfoW(h, ctypes.pointer(info)):
            continue
        r = info.rcMonitor
        out.append(
            Monitor(r.left, r.top, r.right - r.left, r.bottom - r.top, bool(info.dwFlags & 1), str(info.szDevice))
        )
    return out


def _quartz() -> list[Monitor]:  # pragma: no cover - macOS only
    import Quartz

    err, ids, count = Quartz.CGGetActiveDisplayList(32, None, None)
    if err:
        raise OSError(f"CGGetActiveDisplayList error {err}")
    main = Quartz.CGMainDisplayID()
    out = []
    for did in list(ids)[:count]:
        b = Quartz.CGDisplayBounds(did)  # global display coordinates: top-left origin, y down (= pynput)
        out.append(
            Monitor(
                int(b.origin.x), int(b.origin.y), int(b.size.width), int(b.size.height), did == main, f"display {did}"
            )
        )
    return out


def appkit_to_quartz(frames: Sequence[tuple[float, float, float, float]]) -> list[Monitor]:
    """NSScreen frames (bottom-left origin, y up; the first screen is the primary) -> top-left, y down."""
    if not frames:
        return []
    primary_h = frames[0][3]
    return [
        Monitor(int(x), int(primary_h - (y + h)), int(w), int(h), i == 0, f"screen {i + 1}")
        for i, (x, y, w, h) in enumerate(frames)
    ]


def _appkit() -> list[Monitor]:  # pragma: no cover - macOS only
    from AppKit import NSScreen

    frames = []
    for s in NSScreen.screens():
        f = s.frame()
        frames.append((f.origin.x, f.origin.y, f.size.width, f.size.height))
    return appkit_to_quartz(frames)


def _screeninfo() -> list[Monitor]:
    from screeninfo import get_monitors

    return [
        Monitor(int(m.x), int(m.y), int(m.width), int(m.height), bool(getattr(m, "is_primary", False)), m.name or "")
        for m in get_monitors()
    ]


_XRANDR_RE = re.compile(r"^\s*\d+:\s+([+*]*)(\S+)\s+(\d+)/\d+x(\d+)/\d+([+-]\d+)([+-]\d+)(?:\s+(\S+))?")


def parse_xrandr(text: str) -> list[Monitor]:
    """``xrandr --listmonitors`` -> monitors (``0: +*eDP-1 1920/344x1080/194+0+0  eDP-1``)."""
    out = []
    for line in text.splitlines():
        m = _XRANDR_RE.match(line)
        if m:
            flags, name, w, h, x, y, out_name = m.groups()
            out.append(Monitor(int(x), int(y), int(w), int(h), "*" in flags, out_name or name))
    return out


def _xrandr(run: Callable[..., Any] = subprocess.run) -> list[Monitor]:
    if not (os.environ.get("DISPLAY") or "").strip():
        raise OSError("no X display")
    r = run(["xrandr", "--listmonitors"], capture_output=True, text=True, timeout=2, check=False)
    if r.returncode != 0:
        raise OSError(f"xrandr exited {r.returncode}")
    return parse_xrandr(r.stdout)


def _tkinter() -> list[Monitor]:  # pragma: no cover - needs a display
    import tkinter

    root = tkinter.Tk()
    try:
        root.withdraw()
        return [Monitor(0, 0, int(root.winfo_screenwidth()), int(root.winfo_screenheight()), True, "primary")]
    finally:
        root.destroy()


Backend = tuple[str, Callable[[], list[Monitor]]]


def default_backends(platform: str | None = None) -> list[Backend]:
    plat = platform or sys.platform
    if plat == "win32":
        return [("win32", _win32), ("tkinter", _tkinter)]
    if plat == "darwin":
        return [("quartz", _quartz), ("appkit", _appkit), ("tkinter", _tkinter)]
    return [("screeninfo", _screeninfo), ("xrandr", _xrandr), ("tkinter", _tkinter)]


# backends that only ever see the primary monitor: not worth polling for hot-plug
PRIMARY_ONLY_SOURCES = ("tkinter", "fallback")


def enumerate_monitors(backends: Sequence[Backend] | None = None, platform: str | None = None) -> Layout:
    """Detected monitors. Never raises: falls back to primary-only (and logs why)."""
    problems: list[str] = []
    for name, fn in backends if backends is not None else default_backends(platform):
        try:
            mons = sort_monitors(fn())
        except Exception as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if mons:
            note = ""
            if name in PRIMARY_ONLY_SOURCES and problems:
                note = "only the primary monitor is visible (" + "; ".join(problems) + ")"
                log.warning("monitor enumeration fell back to %s: %s", name, "; ".join(problems))
            return Layout(mons, name, note)
        problems.append(f"{name}: no monitors reported")
    note = (
        f"monitor detection failed, assuming one {FALLBACK_SIZE[0]}x{FALLBACK_SIZE[1]} screen ("
        + ("; ".join(problems) or "no backend")
        + ")"
    )
    log.warning(note)
    return single(*FALLBACK_SIZE, source="fallback", note=note)


class MonitorWatcher:
    """Re-detects monitors every ``interval`` seconds (cheap) so hot-plugging needs no restart.

    :meth:`poll` returns the new layout when the geometry changed, else ``None``. A failed detection
    keeps the last good layout instead of shrinking to the fallback. Primary-only sources (tkinter,
    fallback) are not polled unless ``force`` is given.
    """

    def __init__(
        self,
        layout: Layout,
        interval: float = 3.0,
        detect: Callable[[], Layout] = enumerate_monitors,
    ) -> None:
        self.layout = layout
        self.interval = interval
        self.detect = detect
        self._next = 0.0

    @property
    def enabled(self) -> bool:
        return self.interval > 0 and self.layout.source not in PRIMARY_ONLY_SOURCES

    def poll(self, now: float, force: bool = False) -> Layout | None:
        if not force and (not self.enabled or now < self._next):
            return None
        self._next = now + max(self.interval, 0.5)
        try:
            new = self.detect()
        except Exception as exc:  # detect() should not raise, but never let hot-plug polling crash
            log.warning("monitor re-detection failed: %s", exc)
            return None
        if new.source == "fallback" and self.layout.source != "fallback":
            return None  # a transient failure: keep what worked
        if new.key() == self.layout.key():
            return None
        self.layout = new
        return new
