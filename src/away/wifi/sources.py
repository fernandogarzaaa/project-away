"""Wi-Fi sensing sources.

* :class:`PicoScenesSource` – ``.csi`` files from PicoScenes (Intel AX200/AX210
  in monitor mode on Linux) via the PicoScenes Python Toolbox, if installed.
* :class:`IAXSource` – a simple, documented JSONL/CSV interchange for CSI
  produced by other Intel AX tools (e.g. FeitCSI) after conversion; see
  docs/wifi-setup.md. The native binary formats are *not* parsed here.
* :class:`RSSISource` – signal level only, works with any card on any OS.
* :func:`synthetic_csi` – physically-motivated multipath simulator for tests.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CSIFrame:
    t: float
    rssi: float | None
    csi: NDArray[np.complex128] | None  # (n_rx, n_sub) or None for RSSI-only


# ----------------------------------------------------------------- PicoScenes
class PicoScenesSource:  # pragma: no cover - needs the PicoScenes toolbox + a capture file
    def __init__(self, path: str | Path) -> None:
        try:
            from picoscenes import Picoscenes  # PicoScenes-Python-Toolbox (build from source)
        except ImportError as exc:
            raise RuntimeError(
                "PicoScenes Python Toolbox not installed. See docs/wifi-setup.md "
                "(git clone https://github.com/wifisensing/PicoScenes-Python-Toolbox && build)."
            ) from exc
        self.frames = Picoscenes(str(path))

    def __iter__(self) -> Iterator[CSIFrame]:
        for raw in self.frames.raw:
            yield picoscenes_frame(raw)


def picoscenes_frame(raw: dict[str, Any]) -> CSIFrame:
    """Convert one PicoScenes toolbox frame dict to a :class:`CSIFrame` (defensive key access)."""
    basic = raw.get("RxSBasic") or {}
    t = float(basic.get("systemns", basic.get("timestamp", 0.0))) / 1e9
    rssi = basic.get("rssi")
    csi_block = raw.get("CSI") or {}
    csi = csi_block.get("CSI")
    arr = None
    if csi is not None:
        arr = np.asarray(csi, dtype=np.complex128)
        n_tones = int(csi_block.get("numTones", arr.shape[0] if arr.ndim == 1 else arr.shape[-1]))
        arr = arr.reshape(-1, n_tones) if arr.size % max(n_tones, 1) == 0 else arr.reshape(1, -1)
    return CSIFrame(t, float(rssi) if rssi is not None else None, arr)


# ---------------------------------------------------------------- IAX (JSONL/CSV)
def parse_iax_line(line: str) -> CSIFrame | None:
    """JSONL: {"t": s, "rssi": dBm, "re": [[...]] or [...], "im": same shape}."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    d = json.loads(line)
    re_ = np.asarray(d["re"], dtype=np.float64)
    im_ = np.asarray(d["im"], dtype=np.float64)
    if re_.shape != im_.shape:
        raise ValueError("re/im shape mismatch")
    csi = (re_ + 1j * im_).reshape(-1, re_.shape[-1])
    return CSIFrame(float(d["t"]), float(d["rssi"]) if d.get("rssi") is not None else None, csi)


def parse_iax_csv_row(row: list[str]) -> CSIFrame:
    """CSV: t, rssi, re_0, im_0, re_1, im_1, ... (single RX chain)."""
    if len(row) < 4 or (len(row) - 2) % 2:
        raise ValueError("CSV row must be t, rssi, then re/im pairs")
    vals = np.asarray(row[2:], dtype=np.float64)
    csi = (vals[0::2] + 1j * vals[1::2])[None, :]
    return CSIFrame(float(row[0]), float(row[1]) if row[1] else None, csi)


class IAXSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[CSIFrame]:
        with self.path.open(encoding="utf-8") as fh:
            if self.path.suffix.lower() == ".csv":
                for row in csv.reader(fh):
                    if row and not row[0].startswith("#") and row[0] != "t":
                        yield parse_iax_csv_row(row)
            else:
                for line in fh:
                    fr = parse_iax_line(line)
                    if fr is not None:
                        yield fr


# ---------------------------------------------------------------------- RSSI
def parse_proc_net_wireless(text: str, interface: str | None = None) -> float | None:
    """Linux /proc/net/wireless -> signal level (dBm)."""
    for line in text.splitlines()[2:]:
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        if interface and name.strip() != interface:
            continue
        parts = rest.split()
        if len(parts) >= 3:
            return float(parts[2].rstrip("."))
    return None


def parse_netsh(text: str) -> float | None:
    """Windows ``netsh wlan show interfaces`` -> approximate dBm from Signal %."""
    m = re.search(r"^\s*Signal\s*:\s*(\d+)%", text, re.MULTILINE)
    if not m:
        return None
    return float(m.group(1)) / 2.0 - 100.0


def parse_wdutil(text: str) -> float | None:
    """macOS ``wdutil info`` / ``airport -I`` -> RSSI dBm."""
    m = re.search(r"\b(?:RSSI|agrCtlRSSI)\s*:\s*(-?\d+)", text)
    return float(m.group(1)) if m else None


class RSSISource:  # pragma: no cover - OS specific
    def __init__(self, interface: str | None = None, rate: float = 10.0) -> None:
        self.interface = interface
        self.period = 1.0 / rate

    def read(self) -> float | None:
        try:
            if sys.platform.startswith("linux"):
                return parse_proc_net_wireless(Path("/proc/net/wireless").read_text(), self.interface)
            if sys.platform.startswith("win"):
                out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=2)
                return parse_netsh(out.stdout)
            if sys.platform == "darwin" and shutil.which("wdutil"):
                out = subprocess.run(["wdutil", "info"], capture_output=True, text=True, timeout=2)
                return parse_wdutil(out.stdout)
        except Exception:
            return None
        return None

    def __iter__(self) -> Iterator[CSIFrame]:
        while True:
            yield CSIFrame(time.monotonic(), self.read(), None)
            time.sleep(self.period)


# ----------------------------------------------------------------- synthetic
C = 3e8


def synthetic_csi(
    kind: str,
    seconds: float = 2.0,
    fs: float = 100.0,
    n_sub: int = 56,
    n_rx: int = 2,
    fc: float = 5.18e9,
    bw: float = 20e6,
    seed: int = 0,
    noise: float = 0.01,
    phase_garbage: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Simulate CSI for ``kind`` in {empty, presence, push, pull, swipe}.

    Static multipath + one dynamic reflection whose path length d(t) changes:
    push = shorter (hand toward the link), pull = longer, swipe = shorter then
    longer. Random per-packet common phase + timing slope (CFO/STO) is added
    when ``phase_garbage`` so sanitisation is exercised.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    f = fc + (np.arange(n_sub) - n_sub / 2) * (bw / n_sub)
    static = np.zeros((n_rx, n_sub), dtype=np.complex128)
    for _ in range(4):
        d = rng.uniform(2, 8)
        a = rng.uniform(0.2, 1.0)
        for r in range(n_rx):
            static[r] += a * np.exp(-2j * np.pi * f * (d + 0.03 * r) / C)
    d0 = 3.0
    amp_dyn = 0.0
    dt = np.zeros(n)
    mid = (t - t.mean()) / max(seconds / 2, 1e-9)
    if kind == "presence":
        amp_dyn, dt = 0.3, 0.006 * np.sin(2 * np.pi * 0.25 * t)  # breathing-scale motion
    elif kind == "push":
        amp_dyn, dt = 0.35, -0.3 * (1 / (1 + np.exp(-6 * mid)))
    elif kind == "pull":
        amp_dyn, dt = 0.35, 0.3 * (1 / (1 + np.exp(-6 * mid)))
    elif kind == "swipe":
        amp_dyn, dt = 0.35, -0.3 * np.exp(-((mid / 0.35) ** 2))
    elif kind != "empty":
        raise ValueError(f"unknown kind {kind!r}")
    H = np.empty((n, n_rx, n_sub), dtype=np.complex128)
    for i in range(n):
        dyn = amp_dyn * np.exp(-2j * np.pi * f * (d0 + dt[i]) / C)
        for r in range(n_rx):
            H[i, r] = static[r] + dyn * np.exp(-2j * np.pi * f * 0.02 * r / C)
    H += noise * (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape))
    if phase_garbage:
        k = np.arange(n_sub) - n_sub / 2
        common = rng.uniform(-np.pi, np.pi, n)
        slope = rng.normal(0, 0.05, n)
        H *= np.exp(1j * (common[:, None, None] + slope[:, None, None] * k[None, None, :]))
    return t, H


def frames_from_array(t: NDArray[np.float64], H: NDArray[np.complex128]) -> list[CSIFrame]:
    return [CSIFrame(float(ti), float(20 * math.log10(np.abs(h).mean() + 1e-12)), h) for ti, h in zip(t, H)]
