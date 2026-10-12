"""``away iphone listen | compare``: check the Live Link Face stream and compare it with the webcam."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from away.face.livelink import ARKIT_BLENDSHAPES, ARKIT_TO_MEDIAPIPE, LiveLinkReceiver, lan_ips, to_mediapipe

# what `listen` shows: the blendshapes behind the mouth/brow/wink clicks and the lip-motion gate
LISTEN_KEYS = ("jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "browInnerUp", "mouthSmile")
COMPARE_KEYS = (
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
)
SHARED_KEYS = tuple(n for n in ARKIT_BLENDSHAPES if ARKIT_TO_MEDIAPIPE[n] is not None)  # 51 in both models


def setup_steps(ips: list[str], port: int) -> list[str]:
    ip_txt = " or ".join(ips) if ips else "this PC's IPv4 address (ipconfig on Windows)"
    return [
        "On the iPhone (Face ID / TrueDepth camera needed; free):",
        "  1. Install 'Live Link Face' by Unreal Engine from the App Store.",
        "  2. Open it, tap the gear (Settings) and set the capture mode to Live Link (ARKit)",
        "     (not MetaHuman Animator / MetaHuman: those use a different protocol).",
        f"  3. Settings > Streaming / Live Link > Add Target: IP {ip_txt}, port {port}.",
        "  4. Keep the phone and this PC on the same Wi-Fi network (no guest network / client isolation).",
        "  5. Back on the main screen make sure the Live button is on (green), and face the phone.",
        f"On this PC: when Windows Firewall asks, allow Python / Project Away on Private networks (UDP {port}).",
        f'  No prompt? Run as admin: netsh advfirewall firewall add rule name="Project Away iPhone" '
        f"dir=in action=allow protocol=UDP localport={port} profile=private",
    ]


def _fmt_values(bs: dict[str, float]) -> str:
    parts = []
    for k in LISTEN_KEYS:
        if k == "mouthSmile":
            v = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2
        else:
            v = bs.get(k, 0.0)
        parts.append(f"{k} {v:4.2f}")
    return "  ".join(parts)


def status_line(rx: LiveLinkReceiver, now: float) -> str:
    rate = rx.rate(now)
    latest = rx.latest()
    if latest is None:
        msg = f"waiting for packets on UDP {rx.port}..."
        if rx.bad_packets:
            msg += f" ({rx.bad_packets} unreadable: {rx.last_error}; is the app in Live Link (ARKit) mode?)"
        return msg
    face = rx.latest_face(now)
    age = rx.age(now) or 0.0
    who = f" from {rx.last_sender}" if rx.last_sender else ""
    if face is None:
        state = "STALE" if age > rx.stale_after else "no face"
        return f"{rate:5.1f} pkt/s  [{state}, last packet {age:.1f}s ago]{who}"
    return f"{rate:5.1f} pkt/s  {_fmt_values(face[1])}{who}"


def run_listen(
    rx: LiveLinkReceiver,
    seconds: float | None = None,
    out: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    interval: float = 0.5,
    ips: list[str] | None = None,
) -> int:
    """Print setup steps, then a status line every ``interval`` s. Returns 0 if any packet arrived."""
    ips = lan_ips() if ips is None else ips
    out("This PC's LAN IP: " + (", ".join(ips) if ips else "(not found; run ipconfig)") + f"   UDP port: {rx.port}")
    for line in setup_steps(ips, rx.port):
        out(line)
    if not rx.start():
        out(f"error: {rx.error}")
        return 2
    out("Listening (Ctrl+C to stop)...")
    t0 = clock()
    try:
        while seconds is None or clock() - t0 < seconds:
            sleep(interval)
            out(status_line(rx, clock()))
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
    out(f"packets: {rx.packets} ok, {rx.no_face_packets} without a face, {rx.bad_packets} unreadable")
    if rx.packets == 0:
        out("Nothing received: check the target IP/port in the app, the same Wi-Fi, ARKit mode and the firewall.")
        return 1
    return 0


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_compare(
    rx: LiveLinkReceiver,
    webcam: Iterable[tuple[float, dict[str, float] | None]],
    seconds: float,
    out_path: Path,
    keys: tuple[str, ...] = COMPARE_KEYS,
    swap_sides: bool = False,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Log webcam vs iPhone blendshapes per webcam frame to ``out_path`` (CSV) for ``seconds``.

    ``webcam`` yields (monotonic time, MediaPipe blendshapes of the main face or None). The iPhone value
    on each row is the newest fresh iPhone frame at that moment. Returns a per-key summary.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["t", "webcam_face", "iphone_face", "iphone_age_ms"]
    for k in keys:
        header += [f"webcam_{k}", f"iphone_{k}"]
    paired: dict[str, tuple[list[float], list[float]]] = {k: ([], []) for k in keys}
    rows = webcam_faces = iphone_faces = 0
    t0: float | None = None
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for t, wbs in webcam:
            if t0 is None:
                t0 = t
            if t - t0 > seconds:
                break
            face = rx.latest_face(t)
            ibs = to_mediapipe(face[1], swap_sides) if face is not None else None
            age = "" if face is None else f"{max(0.0, t - face[0]) * 1000:.0f}"
            row: list[Any] = [f"{t - t0:.3f}", int(wbs is not None), int(ibs is not None), age]
            for k in keys:
                wv = None if wbs is None else float(wbs.get(k, 0.0))
                iv = None if ibs is None else float(ibs.get(k, 0.0))
                row += ["" if wv is None else f"{wv:.4f}", "" if iv is None else f"{iv:.4f}"]
                if wv is not None and iv is not None:
                    paired[k][0].append(wv)
                    paired[k][1].append(iv)
            w.writerow(row)
            rows += 1
            webcam_faces += wbs is not None
            iphone_faces += ibs is not None
    summary: dict[str, Any] = {
        "csv": str(out_path),
        "rows": rows,
        "webcam_face_rows": webcam_faces,
        "iphone_face_rows": iphone_faces,
        "keys": {},
    }
    out(f"wrote {rows} rows to {out_path}  (webcam face in {webcam_faces}, iPhone face in {iphone_faces})")
    out(f"{'blendshape':<22}{'pairs':>6}{'webcam':>9}{'iphone':>9}{'|diff|':>9}{'corr':>7}")
    for k in keys:
        a, b = (np.asarray(x, dtype=float) for x in paired[k])
        if len(a) == 0:
            summary["keys"][k] = {"pairs": 0}
            continue
        corr = _pearson(a, b)
        stats = {
            "pairs": len(a),
            "webcam_mean": float(a.mean()),
            "iphone_mean": float(b.mean()),
            "mean_abs_diff": float(np.abs(a - b).mean()),
            "corr": corr,
        }
        summary["keys"][k] = stats
        c = "   n/a" if corr is None else f"{corr:7.2f}"
        out(f"{k:<22}{len(a):>6}{a.mean():>9.3f}{b.mean():>9.3f}{stats['mean_abs_diff']:>9.3f}{c}")
    if not any(v.get("pairs") for v in summary["keys"].values()):
        out("No moments with both a webcam face and a fresh iPhone face; run `away iphone listen` first.")
    return summary


def default_compare_path(data_dir: Path) -> Path:
    return data_dir / "iphone" / f"compare-{datetime.now():%Y%m%d-%H%M%S}.csv"


def webcam_blendshapes(
    model: Path, camera: int | str, mirror: bool
) -> Iterable[tuple[float, dict[str, float] | None]]:  # pragma: no cover - needs a camera + mediapipe
    """(t, blendshapes of the largest face) per webcam frame, using the same model as the app."""
    from away.face.landmarker import FaceLandmarker
    from away.hands.tracker import camera_frames

    lm = FaceLandmarker(model, max_faces=1)
    try:
        for t, rgb in camera_frames(camera, mirror):
            obs = lm.process(rgb, t)
            main = max(obs, key=lambda o: o.box[2]) if obs else None
            yield t, (main.blendshapes if main is not None else None)
    finally:
        lm.close()
