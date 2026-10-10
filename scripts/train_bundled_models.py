"""Reproduce the models shipped in src/away/data.

* static_pose.npz  - MLP trained on REAL HaGRID v2 landmark annotations
  (test-split JSONs only, ~20 MB fetched via HTTP range requests), evaluated
  on held-out subjects. Metrics -> static_pose.metrics.json.
* gesture_tcn.onnx - TCN trained on SYNTHETIC sequences only. Its metrics are
  fit-to-synthetic numbers, not real-world accuracy. Disabled by default in
  config (hands.gestures.use_model: false) until you run `away calibrate`.

Usage: python scripts/train_bundled_models.py [--hagrid-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from away.hands.dataset import fetch_hagrid_subset
from away.hands.train import train_static, train_temporal

DATA = Path(__file__).resolve().parents[1] / "src" / "away" / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hagrid-dir", default=str(Path.home() / ".cache" / "project-away" / "hagrid"))
    ap.add_argument("--epochs-static", type=int, default=30)
    ap.add_argument("--epochs-temporal", type=int, default=30)
    a = ap.parse_args()
    hdir = Path(a.hagrid_dir)
    fetch_hagrid_subset(hdir)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        ms = train_static(hdir, out, max_per_class=3000, epochs=a.epochs_static)
        mt = train_temporal(out, n_synthetic=250, epochs=a.epochs_temporal)
        for name in (
            "static_pose.npz",
            "static_pose.metrics.json",
            "gesture_tcn.onnx",
            "gesture_tcn.json",
            "gesture_tcn.metrics.json",
        ):
            shutil.copy(out / name, DATA / name)
    print(json.dumps({"static": {k: v for k, v in ms.items() if k != "confusion"}, "temporal": mt}, indent=2))


if __name__ == "__main__":
    main()
