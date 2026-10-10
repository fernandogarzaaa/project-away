"""``away record``: capture your own labelled gesture clips for calibration."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from away.hands.synthetic import GESTURES


def save_clip(folder: Path, label: str, frames: list[NDArray[np.float64]], handedness: str, fps: float) -> Path:
    if label not in GESTURES:
        raise ValueError(f"label must be one of {GESTURES}")
    if not frames:
        raise ValueError("no hand frames captured")
    folder = Path(folder) / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label}_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.npz"
    np.savez_compressed(path, X=np.stack(frames), label=label, handedness=handedness, fps=fps)
    return path


def record(  # pragma: no cover - needs a camera
    label: str, count: int, seconds: float, out: Path, model_path: Path, camera: int = 0, mirror: bool = True
) -> list[Path]:
    from away.hands.tracker import HandTracker, camera_frames

    tracker = HandTracker(model_path)
    saved: list[Path] = []
    frames_iter = camera_frames(camera, mirror)
    try:
        for i in range(count):
            print(f"[{i + 1}/{count}] '{label}' in 3...", flush=True)
            t_ready = time.monotonic() + 3.0
            for t, _rgb in frames_iter:  # keep the camera warm during the countdown
                if t >= t_ready:
                    break
            print("  GO", flush=True)
            clip: list[NDArray[np.float64]] = []
            hand = "Right"
            t0 = time.monotonic()
            for t, rgb in frames_iter:
                fr = tracker.process(rgb, t)
                if fr.landmarks is not None:
                    clip.append(fr.landmarks)
                    hand = fr.handedness
                if t - t0 >= seconds:
                    break
            if len(clip) < 5:
                print("  no hand seen, skipping", flush=True)
                continue
            saved.append(save_clip(out, label, clip, hand, len(clip) / seconds))
            print(f"  saved {saved[-1].name} ({len(clip)} frames)", flush=True)
    finally:
        tracker.close()
    return saved
