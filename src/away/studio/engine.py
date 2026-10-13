"""Runtime side of Gesture Studio: recognise Studio gestures, explain the model, enforce zones.

* :class:`StudioRecognizer`: the trained groups (see :mod:`away.studio.trainer`) -> one probability per class,
  ``nothing`` included; plus a landmark importance map (occlusion) and the closest practice example.
* :class:`GestureArbiter`: per gesture ``idle -> arming -> firing -> cooldown`` (``recognized`` = seen but not
  allowed to fire: switched off or outside the zones), with how-sure threshold, hold-to-arm and rest time.
* :class:`StudioRuntime`: what :class:`away.app.runner.Pipeline` uses: the zone gate for every activation, the
  Studio gestures, and their follow-up action chains.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.studio.features import (
    MOTION_FRAMES,
    motion_window,
    occlude_motion,
    occlude_static,
    parse_group,
    static_vector,
)
from away.studio.pack import NOTHING, GestureSettings, StudioStore, gesture_settings
from away.studio.zones import ZoneGate, anchor_point, camera_key, lookup_map

Array = NDArray[np.float64]
STATES = ("idle", "arming", "recognized", "firing", "cooldown")
FIRING_SHOW_S = 0.35  # how long "firing" stays visible after the action ran
RELEASE_MARGIN = 0.15  # probability must drop this far under the threshold to re-arm
TIMELINE_S = 5.0
# events that START something are gated by zones; ongoing/ending ones always pass (a drag must end)
ZONE_GATED_EVENTS = frozenset(
    {
        "click",
        "double_click",
        "drag_start",
        "scroll",
        "autoscroll_start",
        "clutch",
        "swipe_left",
        "swipe_right",
        "push",
        "dwell",
        "palm_tap",
        "custom",
    }
)


# ---------------------------------------------------------------------- model groups
class _Group:
    def __init__(self, entry: dict[str, Any], folder: Path) -> None:
        self.key = str(entry["key"])
        self.kind, self.hands = parse_group(self.key)
        self.classes: list[str] = [str(c) for c in entry["classes"]]
        self.window_s = float(entry.get("window_s", 1.5))
        self.backend = str(entry.get("backend", "mlp"))
        path = folder / str(entry["file"])
        if self.backend == "tcn":
            from away.hands.model import OnnxClassifier

            self._onnx = OnnxClassifier(path)
            self.mean = np.asarray(self._onnx.meta.get("mean"), dtype=np.float64)
            self.sd = np.asarray(self._onnx.meta.get("sd"), dtype=np.float64)
        else:
            from away.hands.model import NumpyMLP

            with np.load(path, allow_pickle=False) as data:
                self.mean = np.asarray(data["mean"], dtype=np.float64)
                weights = {k: data[k] for k in data.files if k not in ("classes", "mean")}
            self._mlp = NumpyMLP(weights, self.classes)

    def predict(self, X: Array) -> Array:
        if self.backend == "tcn":
            Z = ((X - self.mean) / self.sd).reshape(len(X), MOTION_FRAMES, -1)
            return self._onnx.predict_proba(Z)
        return self._mlp.predict_proba(X)


@dataclass
class Frame:
    t: float
    pointer: Array
    helper: Array | None
    hand: str


class StudioRecognizer:
    """Loads ``<studio>/model`` and turns hand frames into class probabilities."""

    def __init__(self, model_dir: Path) -> None:
        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "studio.json").read_text())
        self.groups: list[_Group] = []
        self.notes: list[str] = []
        for entry in meta.get("groups", []):
            try:
                self.groups.append(_Group(entry, self.dir))
            except Exception as exc:  # e.g. onnxruntime missing for a motion group
                self.notes.append(f"{entry.get('key')}: {exc}")
        if not self.groups:
            raise ValueError("no usable gesture model" + (f" ({'; '.join(self.notes)})" if self.notes else ""))
        self.classes = [NOTHING] + [c for g in self.groups for c in g.classes if c != NOTHING]
        self.examples: dict[str, Any] = {}
        ex = self.dir / "examples.npz"
        if ex.exists():
            with np.load(ex, allow_pickle=False) as d:
                self.examples = {k: d[k] for k in d.files}
        self.keep_s = max([g.window_s for g in self.groups if g.kind == "motion"] or [0.0]) + 0.5
        self.buf: deque[Frame] = deque()
        self.last_inputs: dict[str, Array] = {}
        self.last_group_probs: dict[str, Array] = {}

    @classmethod
    def load(cls, model_dir: Path) -> StudioRecognizer | None:
        try:
            return cls(model_dir) if (Path(model_dir) / "studio.json").exists() else None
        except Exception:
            return None

    def reset(self) -> None:
        self.buf.clear()

    def _inputs(self, g: _Group) -> Array | None:
        last = self.buf[-1]
        if g.hands == 2 and last.helper is None:
            return None
        if g.kind == "static":
            try:
                return static_vector(last.pointer, last.helper, g.hands)
            except ValueError:
                return None
        frames = [f for f in self.buf if f.t >= last.t - g.window_s]
        if len(frames) < 8 or frames[-1].t - frames[0].t < 0.6 * g.window_s:
            return None
        helpers = [f.helper for f in frames]
        hs = np.stack([h if h is not None else np.zeros((21, 3)) for h in helpers])
        seen = np.array([h is not None for h in helpers])
        try:
            w = motion_window(np.stack([f.pointer for f in frames]), last.hand, hs, seen, g.hands)
        except ValueError:
            return None
        return w.reshape(-1)

    def push(self, t: float, pointer: Any, hand: str = "Right", helper: Any = None) -> dict[str, float] | None:
        """One frame. Returns ``{class: probability}`` (``nothing`` first, sums to 1), or None without a hand."""
        if pointer is None:
            self.buf.clear()
            return None
        try:
            p = np.asarray(pointer, dtype=np.float64).reshape(21, -1)[:, :3]
            h = np.asarray(helper, dtype=np.float64).reshape(21, -1)[:, :3] if helper is not None else None
        except ValueError:
            return None
        if p.shape[1] < 3:
            p = np.concatenate([p, np.zeros((21, 3 - p.shape[1]))], axis=1)
        self.buf.append(Frame(t, p, h if h is not None and h.shape[1] == 3 else None, hand))
        while self.buf and self.buf[0].t < t - self.keep_s:
            self.buf.popleft()
        probs = dict.fromkeys(self.classes, 0.0)
        nothing = 1.0
        self.last_inputs, self.last_group_probs = {}, {}
        for g in self.groups:
            x = self._inputs(g)
            if x is None:
                continue
            pr = g.predict(x[None])[0]
            self.last_inputs[g.key] = x
            self.last_group_probs[g.key] = pr
            for c, v in zip(g.classes, pr):
                if c == NOTHING:
                    nothing *= float(v)
                else:
                    probs[c] = float(v)
        probs[NOTHING] = nothing
        total = sum(probs.values()) or 1.0
        return {c: v / total for c, v in probs.items()}

    def _group_of(self, name: str) -> _Group | None:
        return next((g for g in self.groups if name in g.classes and g.key in self.last_inputs), None)

    def saliency(self, name: str) -> list[float] | None:
        """How much each landmark matters for ``name`` right now (0..1, 21 per hand): the drop in its
        probability when that landmark is replaced by its average position (occlusion)."""
        g = self._group_of(name)
        if g is None or name == NOTHING:
            return None
        x = self.last_inputs[g.key]
        ci = g.classes.index(name)
        base = float(self.last_group_probs[g.key][ci])
        if g.kind == "static":
            batch = occlude_static(x, g.mean, g.hands)
        else:
            batch = occlude_motion(x.reshape(MOTION_FRAMES, -1), g.mean, g.hands).reshape(21 * g.hands, -1)
        drops = np.maximum(0.0, base - g.predict(batch)[:, ci])
        top = float(drops.max())
        return [float(v / top) if top > 1e-9 else 0.0 for v in drops]

    def nearest(self, name: str) -> tuple[str, int, float] | None:
        """``(gesture, example number, distance)`` of the closest practice example of ``name``."""
        g = self._group_of(name)
        if g is None or f"emb_{g.key}" not in self.examples:
            return None
        emb, lab, num = (self.examples[f"{p}_{g.key}"] for p in ("emb", "lab", "num"))
        mask = lab == name
        if not mask.any():
            return None
        sd = emb.std(axis=0) + 1e-6
        d = np.linalg.norm((emb[mask] - self.last_inputs[g.key]) / sd, axis=1) / np.sqrt(emb.shape[1])
        i = int(np.argmin(d))
        return name, int(num[mask][i]), float(d[i])


# ---------------------------------------------------------------------- per gesture states
class GestureArbiter:
    """idle -> arming (sure enough) -> firing (held ``hold_s``, allowed) -> cooldown (``cooldown_s`` passed and
    the probability dropped) -> idle. ``recognized``: held long enough but not allowed (off, or outside zones)."""

    def __init__(self) -> None:
        self.state: dict[str, str] = {}
        self._since: dict[str, float] = {}
        self._fired: dict[str, float] = {}

    def reset(self) -> None:
        self.state.clear()
        self._since.clear()

    def _fire_or_hold(self, name: str, t: float, s: GestureSettings, allowed: bool, out: list[str]) -> str:
        if s.enabled and allowed:
            self._fired[name] = t
            out.append(name)
            return "firing"
        return "recognized"

    def update(
        self, t: float, probs: dict[str, float] | None, settings: dict[str, GestureSettings], allowed: bool = True
    ) -> list[str]:
        out: list[str] = []
        for name, s in settings.items():
            p = float((probs or {}).get(name, 0.0))
            st = self.state.get(name, "idle")
            release = s.threshold - RELEASE_MARGIN
            if st == "firing" and t - self._fired.get(name, -1e9) >= FIRING_SHOW_S:
                st = "cooldown"
            if st == "cooldown":
                if t - self._fired.get(name, -1e9) >= s.cooldown_s and p < release:
                    st = "idle"
            elif st in ("arming", "recognized"):
                if p < release:
                    st = "idle"
                elif st == "arming" and t - self._since.get(name, t) >= s.hold_s:
                    st = self._fire_or_hold(name, t, s, allowed, out)
                elif st == "recognized" and s.enabled and allowed and p >= s.threshold:
                    st = "arming"
                    self._since[name] = t
            if st == "idle" and p >= s.threshold:
                st = "arming"
                self._since[name] = t
                if s.hold_s <= 0:
                    st = self._fire_or_hold(name, t, s, allowed, out)
            self.state[name] = st
        return out


def top_candidate(probs: dict[str, float] | None, threshold: float) -> tuple[str, float, bool]:
    """``(class, probability, unknown)``; ``unknown`` when the best class is under ``threshold``."""
    if not probs:
        return NOTHING, 0.0, True
    name = max(probs, key=lambda k: probs[k])
    p = probs[name]
    return name, p, p < threshold


class ProbTimeline:
    """The last :data:`TIMELINE_S` seconds of probabilities and fired actions (for the detail charts)."""

    def __init__(self, seconds: float = TIMELINE_S) -> None:
        self.seconds = seconds
        self.points: deque[tuple[float, dict[str, float]]] = deque()
        self.events: deque[tuple[float, str]] = deque(maxlen=50)

    def add(self, t: float, probs: dict[str, float] | None) -> None:
        self.points.append((t, dict(probs or {})))
        while self.points and self.points[0][0] < t - self.seconds:
            self.points.popleft()

    def fired(self, t: float, text: str) -> None:
        self.events.append((t, text))

    def series(self, name: str) -> list[tuple[float, float]]:
        return [(t, p.get(name, 0.0)) for t, p in self.points]

    def recent_events(self, now: float) -> list[tuple[float, str]]:
        return [(t, e) for t, e in self.events if t >= now - self.seconds]


# ---------------------------------------------------------------------- chains
class ChainRunner:
    """Runs a gesture's follow-up steps (``wait:0.5`` pauses) after its bound action, off the camera thread."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep, threaded: bool = True) -> None:
        self.sleep = sleep
        self.threaded = threaded

    def run(self, steps: list[str], execute: Callable[[str], Any]) -> threading.Thread | None:
        if not steps:
            return None

        def go() -> None:
            for step in steps:
                if step.startswith("wait:"):
                    self.sleep(float(step.split(":", 1)[1]))
                else:
                    with contextlib.suppress(Exception):
                        execute(step)

        if not self.threaded:
            go()
            return None
        th = threading.Thread(target=go, daemon=True, name="away-chain")
        th.start()
        return th


# ---------------------------------------------------------------------- what the app runs
class StudioRuntime:
    """Zones + Studio gestures inside :class:`~away.app.runner.Pipeline`."""

    def __init__(
        self,
        cfg: Any,
        profile: str = "default",
        store: StudioStore | None = None,
        recognizer: StudioRecognizer | None = None,
        chains: ChainRunner | None = None,
    ) -> None:
        self.store = store or StudioStore()
        self.recognizer = recognizer if recognizer is not None else StudioRecognizer.load(self.store.model_dir)
        self.arbiter = GestureArbiter()
        self.chains = chains or ChainRunner()
        self.profile = profile
        self.zone_ok = True
        self.last_probs: dict[str, float] | None = None
        self.apply_config(cfg)

    def apply_config(self, cfg: Any) -> None:
        self.cfg = cfg
        self.kinds = {g.name: g.kind for g in self.store.gestures()}
        self._settings_cache: dict[str, GestureSettings] | None = None
        self._load_map()

    def _load_map(self) -> None:
        z = self.cfg.zones
        zm = None
        if z.enabled:
            with contextlib.suppress(ValueError):
                zm, _src = lookup_map(z.maps, camera_key(self.cfg), self.profile)
        self.gate = ZoneGate(zm if zm is not None and not zm.is_open() else None, z.hysteresis)

    def set_profile(self, name: str) -> None:
        if name != self.profile:
            self.profile = name
            self._load_map()

    @property
    def settings(self) -> dict[str, GestureSettings]:
        if self._settings_cache is None:
            names = self.recognizer.classes[1:] if self.recognizer is not None else []
            out = {}
            for n in names:
                try:
                    out[n] = gesture_settings(self.cfg, n, self.kinds.get(n, "static"))
                except ValueError:
                    out[n] = GestureSettings(enabled=False)
            self._settings_cache = out
        return self._settings_cache

    def check_zone(self, landmarks: Any) -> bool:
        """Update the zone gate with this frame's pointer hand. True = activations may fire."""
        if landmarks is None:
            self.gate.update(None)
            self.zone_ok = True
            return True
        try:
            pt = anchor_point(landmarks, self.cfg.zones.anchor, bool(self.cfg.hands.mirror))
        except ValueError:
            return self.zone_ok
        self.zone_ok = self.gate.update(pt)
        return self.zone_ok

    def event_allowed(self, kind: str, data: dict[str, Any] | None = None) -> bool:
        """Zones gate activations; the cursor only with ``zones.block_cursor`` (a drag keeps moving)."""
        if self.zone_ok:
            return True
        if kind == "move":
            return not self.cfg.zones.block_cursor or (data or {}).get("anchor") == "pinch"
        return kind not in ZONE_GATED_EVENTS

    def update(self, t: float, landmarks: Any, handedness: str = "Right", helper: Any = None) -> list[str]:
        """Studio gestures that fire on this frame (already zone-gated and per-gesture filtered)."""
        if self.recognizer is None:
            return []
        self.last_probs = self.recognizer.push(t, landmarks, handedness, helper)
        return self.arbiter.update(t, self.last_probs, self.settings, allowed=self.zone_ok)

    def run_chain(self, name: str, execute: Callable[[str], Any]) -> None:
        s = self.settings.get(name)
        if s is not None and s.then:
            self.chains.run(list(s.then), execute)


def fire_studio_gesture(pipeline: Any, name: str, t: float) -> list[str]:
    """A Studio gesture fired inside the app: through the bindings (``gesture:<name>``) like any custom gesture,
    then its follow-up chain (only when the bound action actually ran: not while paused or unbound)."""
    from away.bindings import parse_action
    from away.hands.gestures import GestureEvent

    pipeline.bus.publish("gesture.custom", {"name": name, "studio": True}, t)
    act = pipeline.dispatcher.on_gesture(GestureEvent("custom", t, {"name": name}))
    if act.kind != "none":
        allow = bool(pipeline.cfg.bindings.allow_shell)
        pipeline.studio.run_chain(name, lambda step: pipeline.executor.run(parse_action(step, allow_shell=allow)))
    return [f"custom:{name}"]
