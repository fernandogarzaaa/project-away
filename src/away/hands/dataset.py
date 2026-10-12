"""Training data sources.

1. **HaGRID v2** real landmark annotations (static poses). The full
   ``annotations.zip`` is ~720 MB, so :func:`fetch_hagrid_subset` reads the
   zip's central directory with HTTP range requests and pulls only the
   per-class JSON files you ask for (~3 MB compressed each from the *test*
   split). Each JSON has ``hand_landmarks`` (21 x [x, y]), ``labels`` and an
   anonymised ``user_id`` that we use for a subject-disjoint split.
2. **Your own recordings** from ``away record`` (``.npz`` with ``X`` (T,21,3),
   ``label``, ``handedness``).
3. **Synthetic** sequences (:mod:`away.hands.synthetic`) for tests/pre-training.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.hands.landmarks import infer_side, static_features
from away.hands.model import STATIC_CLASSES

HAGRID_V2_ANNOTATIONS = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
    "annotations_with_landmarks/annotations.zip"
)
# HaGRID class -> our static pose class
HAGRID_MAP = {
    "point": "point",
    "palm": "palm",
    "fist": "fist",
    "peace": "two",
    "two_up": "two",
    "ok": "ok",
    "no_gesture": "none",
}


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (enough for zipfile)."""

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url, self.timeout, self.pos = url, timeout, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b: Any) -> int:
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_hagrid_subset(
    dest: Path, classes: list[str] | None = None, split: str = "test", url: str = HAGRID_V2_ANNOTATIONS
) -> list[Path]:
    """Download only the named per-class annotation JSONs. Skips files already present."""
    classes = classes or list(HAGRID_MAP)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    missing: list[Path] = []
    for c in classes:
        p = dest / f"{split}_{c}.json"
        (out if p.exists() else missing).append(p)
    if missing:
        zf = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=1 << 20))
        for p in missing:
            c = p.stem.split("_", 1)[1]
            p.write_bytes(zf.read(f"annotations/{split}/{c}.json"))
            out.append(p)
    return out


def load_hagrid_json(path: Path, cls: str) -> list[tuple[NDArray[np.float64], str, str]]:
    """Return (features(42,), our_label, user_id) for every hand labelled ``cls``."""
    data = json.loads(Path(path).read_text())
    target = HAGRID_MAP[cls]
    rows: list[tuple[NDArray[np.float64], str, str]] = []
    for ann in data.values():
        lms = ann.get("hand_landmarks") or []
        labels = ann.get("labels") or []
        for lm, lab in zip(lms, labels):
            if lab != cls or not lm or len(lm) != 21:
                continue
            arr = np.asarray(lm, dtype=np.float64)
            # HaGRID gives no handedness; mirror by thumb side so left hands look right-handed
            hand = infer_side(arr)
            try:
                rows.append((static_features(arr, hand), target, str(ann.get("user_id", ""))))
            except ValueError:
                continue
    return rows


def subject_split(
    users: list[str], test_frac: float = 0.2, val_frac: float = 0.1
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Deterministic subject-disjoint split by hashing user ids."""
    h = np.array([int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for u in users])
    test = h < test_frac
    val = (h >= test_frac) & (h < test_frac + val_frac)
    return ~(test | val), val, test


def load_hagrid_dir(
    folder: Path, max_per_class: int | None = None, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    rng = np.random.default_rng(seed)
    X, y, users = [], [], []
    for p in sorted(Path(folder).glob("*_*.json")):
        cls = p.stem.split("_", 1)[1]
        if cls not in HAGRID_MAP:
            continue
        rows = load_hagrid_json(p, cls)
        if max_per_class and len(rows) > max_per_class:
            idx = rng.choice(len(rows), max_per_class, replace=False)
            rows = [rows[i] for i in idx]
        for f, lab, u in rows:
            X.append(f)
            y.append(STATIC_CLASSES.index(lab))
            users.append(u)
    if not X:
        raise FileNotFoundError(f"no HaGRID annotation JSONs in {folder}")
    return np.stack(X), np.array(y, dtype=np.int64), users


def load_recordings_with_groups(
    folder: Path,
) -> tuple[list[NDArray[np.float64]], list[str], list[str], list[str]]:
    """Load clips and optional session/subject grouping metadata for evaluation."""
    seqs, labels, hands, groups = [], [], [], []
    for p in sorted(Path(folder).glob("**/*.npz")):
        with np.load(p, allow_pickle=False) as d:
            seqs.append(np.asarray(d["X"], dtype=np.float64))
            labels.append(str(d["label"]))
            hands.append(str(d["handedness"]) if "handedness" in d.files else "Right")
            if "session_id" in d.files:
                group = str(d["session_id"].item())
            elif "subject_id" in d.files:
                group = str(d["subject_id"].item())
            else:
                # File-level grouping is a fallback, not proof of subject independence.
                group = str(p.relative_to(folder))
            groups.append(group)
    return seqs, labels, hands, groups


def load_recordings(folder: Path) -> tuple[list[NDArray[np.float64]], list[str], list[str]]:
    """Backward-compatible loader returning (sequences, labels, handedness)."""
    seqs, labels, hands, _groups = load_recordings_with_groups(folder)
    return seqs, labels, hands


def fixed_length(seq: NDArray[np.float64], frames: int) -> NDArray[np.float64]:
    """Resample a (T, ...) sequence to ``frames`` by nearest-index sampling."""
    if len(seq) == 0:
        raise ValueError("empty sequence")
    idx = np.linspace(0, len(seq) - 1, frames).round().astype(int)
    return seq[idx]
