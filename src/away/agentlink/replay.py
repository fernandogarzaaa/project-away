"""``away replay FILE``: run a recording through the gesture pipeline headless (no camera, no mouse).

The replay builds a fresh :class:`~away.app.runner.Pipeline` from the configuration stored in the recording,
with a :class:`~away.control.mouse.RecordingBackend` instead of the real mouse, and feeds it every recorded frame
with its original timestamps. The pipeline is clocked only by those timestamps, so the result (gesture events
and mouse calls) is the same on every run and every machine: that is what turns a recording into a regression
test (see :func:`away.agentlink.replay.assert_replay` and ``tests/replays/``).

Not replayed, on purpose: the learned temporal model (``--model`` turns it on; its ONNX output can differ in the
last bits between CPUs), head/gaze tracking (needs face data), Wi-Fi, voice, and the physical-input auto-pause.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from away.agentlink import recording

DROP_TOPICS = ("gesture.move",)  # very frequent; summarised by the mouse calls instead


@dataclass
class ReplayResult:
    frames: int
    events: list[tuple[float, str, dict[str, Any]]] = field(default_factory=list)
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    recorded_events: list[str] = field(default_factory=list)

    def kinds(self) -> list[str]:
        return [topic for _t, topic, _d in self.events]

    def summary(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "events": dict(Counter(self.kinds())),
            "mouse": dict(Counter(name for name, _a in self.calls)),
            "digest": self.digest(),
        }

    def digest(self) -> str:
        """A short fingerprint of everything the pipeline did (events + mouse calls)."""
        payload = json.dumps(
            {"e": [(round(t, 4), k, d) for t, k, d in self.events], "c": [(n, list(a)) for n, a in self.calls]},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def expectation(self) -> dict[str, Any]:
        """What a regression test pins: the event sequence (no timestamps) and mouse call counts."""
        return {"events": self.kinds(), "mouse": dict(Counter(name for name, _a in self.calls))}


def _arr(x: Any) -> Any:
    return None if x is None else np.asarray(x, dtype=np.float64)


def replay(
    header: dict[str, Any], records: list[dict[str, Any]], *, use_model: bool = False, cfg: Any = None
) -> ReplayResult:
    from away import config as cfgmod
    from away.app.runner import Pipeline
    from away.control.mouse import RecordingBackend
    from away.events import EventBus

    if cfg is None:
        cfg, _unknown = cfgmod.from_dict(header.get("config") or {})
    cfg.hands.cursor.head_assist = "off"
    cfg.hands.cursor.head.monitor_switch = False
    cfg.app.auto_pause = False
    cfg.wifi.enabled = False
    temporal = None
    if use_model and cfg.hands.gestures.use_model:
        from away.hands.runtime import TemporalGestureClassifier, find_model

        p = find_model("gesture_tcn.onnx", None)
        temporal = TemporalGestureClassifier(p) if p else None
    bus = EventBus(history=16)
    backend = RecordingBackend()
    screen = tuple(header.get("screen") or (1920, 1080))
    pipe = Pipeline(cfg, backend, (int(screen[0]), int(screen[1])), bus, temporal)
    pipe.headgaze.estimator = None
    if header.get("paused"):
        pipe.set_paused(True)
    result = ReplayResult(frames=0)

    def collect(ev: Any) -> None:
        if ev.topic.startswith("gesture.") and ev.topic not in DROP_TOPICS:
            data = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.data.items()}
            result.events.append((float(ev.ts), ev.topic, data))

    bus.subscribe("*", collect)
    for r in records:
        if r.get("k") == "e":
            if str(r.get("topic", "")).startswith("gesture."):
                result.recorded_events.append(str(r["topic"]))
            continue
        if r.get("k") != "h":
            continue
        result.frames += 1
        kw: dict[str, Any] = {}
        for k in ("pinch", "depth", "pointer_id", "helper_id"):
            if k in r:
                kw[k] = r[k]
        if r.get("promoted"):
            kw["promoted"] = True
        pipe.on_hand(
            float(r["t"]),
            _arr(r.get("lm")),
            str(r.get("hd") or "Right"),
            helper=_arr(r.get("helper")),
            points3d=_arr(r.get("points3d")),
            helper_points3d=_arr(r.get("helper_points3d")),
            **kw,
        )
    result.calls = list(backend.calls)
    return result


def replay_file(path: Path, use_model: bool = False) -> ReplayResult:
    header, records = recording.read(path)
    return replay(header, records, use_model=use_model)


def expectation_path(path: Path) -> Path:
    return Path(str(path) + ".expected.json")


def assert_replay(path: Path, update: bool = False) -> ReplayResult:
    """Pytest helper: replay ``path`` and compare with ``<path>.expected.json`` (written on first run or with
    ``update=True``, e.g. ``AWAY_UPDATE_REPLAYS=1``). Raises AssertionError with a readable diff."""
    res = replay_file(Path(path))
    exp_path = expectation_path(Path(path))
    got = res.expectation()
    if update or not exp_path.exists():
        exp_path.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
        return res
    want = json.loads(exp_path.read_text(encoding="utf-8"))
    if got != want:
        raise AssertionError(
            f"replay of {Path(path).name} changed:\n  expected events {want.get('events')}\n"
            f"  got      events {got['events']}\n  expected mouse {want.get('mouse')}\n  got      mouse {got['mouse']}"
        )
    return res


def synthetic_recording(script: list[str] | None = None, seed: int = 1, fps: float = 30.0) -> tuple[dict, list]:
    """A recording made from the synthetic hand generator (``away demo``'s gestures), for tests and demos."""
    from away.hands.synthetic import gesture_sequence

    rng = np.random.default_rng(seed)
    t = 0.0
    recs: list[dict[str, Any]] = []
    for label in script or ["move", "pinch_click", "idle", "pinch_drag", "scroll", "palm_clutch"]:
        for lm in gesture_sequence(label, 24, rng):
            t += 1 / fps
            recs.append(recording.hand_record(t, lm, "Right"))
        t += 1 / fps
        recs.append(recording.hand_record(t, None, "Right"))
        t += 0.5
    return recording.make_header(note="synthetic"), recs
