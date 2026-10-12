"""Wire the subsystems together on the event bus.

``Pipeline`` is pure logic (no devices) so it can be driven by synthetic
frames in tests and by ``away demo``. ``AwayApp`` adds the camera, mic,
hotkeys and Wi-Fi threads around it.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from away import paths
from away.actions import ActionExecutor, CustomGestureDetector, Dispatcher, hotkey_map
from away.bindings import Action
from away.config import AppConfig
from away.control.mouse import MouseBackend, MouseController, RecordingBackend
from away.events import EventBus
from away.face.commands import BrowRaiseToggle, MouthOpenTrigger, WinkTrigger
from away.face.gate import SpeechGate
from away.face.speaker import ActiveSpeakerDetector
from away.face.speaking import SpeakingScorer
from away.face.tracking import FaceTracker
from away.fusion import GestureFusion
from away.hands.cursor import CursorMapper, Screen
from away.hands.filters import OneEuro2D
from away.hands.gestures import GestureEvent, GestureStateMachine
from away.hands.landmarks import as_array, pinch_ratio
from away.hands.stabilize import ClickStabilizer, DwellDetector, pinch_progress
from away.multicam.fusion import FusedHand, PushDetector
from away.voice.always_on import GateHint
from away.voice.clap import camera_view
from away.wifi.classifier import CoarseCSIClassifier
from away.wifi.features import window_features

# Multi-camera start-up. Windows MSMF/DirectShow can take several seconds to open a USB webcam (more when
# two open at once), so a camera only counts as failed when it errors or sends nothing for MULTICAM_PROBE_S.
MULTICAM_PROBE_S = 12.0
MULTICAM_RETRY_S = 45.0  # after falling back to one camera: re-probe the failed ones this often
MULTICAM_RETRY_MAX_S = 600.0  # back-off ceiling when the full set keeps failing right after a good probe
MULTICAM_NONE_RETRY_S = 10.0  # no camera works at all: retry the whole set this often


def effective_ambient_fallback(cfg: AppConfig) -> str:
    """Always-on without a wake word: ``type`` when the user turned the face gate off (that must still type;
    ``commands_only`` there silently dropped all dictation), else ``voice.ambient_fallback``."""
    if cfg.voice.ambient_fallback == "type" or not cfg.face.gate_dictation:
        return "type"
    return cfg.voice.ambient_fallback


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        backend: MouseBackend | None = None,
        screen: Screen = (1920, 1080),
        bus: EventBus | None = None,
        temporal: Any = None,
        active_app: Any = None,
        guard: Any = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.gestures = GestureStateMachine(cfg.hands.gestures)
        c = cfg.hands.cursor
        stab = ClickStabilizer(c.click_rewind_ms / 1000.0, c.freeze_margin) if c.click_freeze else None
        self.mouse = MouseController(backend or RecordingBackend(), CursorMapper(screen, c), stabilizer=stab)
        self.guard = guard  # ManualInputGuard: physical mouse/keyboard use pauses gesture output
        self._auto_paused = False
        self.dwell = DwellDetector(cfg.hands.dwell.radius_px, cfg.hands.dwell.time_s)
        mc = cfg.hands.multicam
        self.push = PushDetector(mc.push_distance_m, mc.push_time_s, mc.push_cooldown_s)
        self.wink = WinkTrigger(cfg.face.wink_threshold, cfg.face.wink_open_max, cfg.face.wink_hold)
        self.last_ratio: float | None = None
        self._hand_views: deque[tuple[float, int, float | None]] = deque(maxlen=90)
        self.last_fused: FusedHand | None = None
        self.fusion = GestureFusion(
            cfg.wifi.fusion_weight if cfg.wifi.enabled else 0.0,
            threshold=cfg.hands.gestures.model_min_conf if temporal else 0.5,
        )
        self.temporal = temporal if cfg.hands.gestures.use_model else None
        self.faces = FaceTracker()
        self.speaking = SpeakingScorer()
        self.asd = ActiveSpeakerDetector(
            min_correlation=cfg.face.min_correlation, speaking_threshold=cfg.face.speaking_threshold
        )
        self.gate = SpeechGate()
        self.brow = BrowRaiseToggle(cfg.face.brow_threshold, cfg.face.brow_hold)
        self.mouth = MouthOpenTrigger(cfg.face.mouth_threshold, cfg.face.mouth_hold)
        self.csi = CoarseCSIClassifier()
        self.paused = False
        self.mic_energy = 0.0  # updated by the mic thread
        self._enroll_pending = True
        # detection (state machine) -> mapping (bindings) -> execution (backend / hooks)
        self.hand_prefer = cfg.hands.prefer  # hands:right|left|any (voice) changes it until the next restart
        self.executor = ActionExecutor(
            self.mouse, cfg.bindings.allow_shell, hooks={"pause": self._pause_hook, "hands": self._hands_hook}
        )
        self.dispatcher = Dispatcher(
            cfg.bindings, self.executor, self.gestures, active_app=active_app, paused=self.output_paused
        )
        self.dispatcher.on_fire = self._on_fire
        self.custom = CustomGestureDetector(cfg.bindings.custom_gestures, cfg.hands.gestures.model_min_conf)

    def _on_fire(self, trig: str, act: Action) -> None:
        self.bus.publish("action.fired", {"trigger": trig, "action": str(act)})

    # pause -------------------------------------------------------------------
    def output_paused(self) -> bool:
        """Paused by the user, or auto-paused because the physical mouse/keyboard is in use."""
        return self.paused or self._auto_paused

    def _check_auto_pause(self, t: float) -> None:
        auto = bool(self.guard is not None and self.guard.active)
        if auto == self._auto_paused:
            return
        self._auto_paused = auto
        if auto:
            self.mouse._release_if_down()
            self.gestures.autoscroll.stop()
        else:
            self.mouse.mapper.reset()  # resume from wherever the real mouse left the cursor, no jump
        self.bus.publish("app.auto_paused", {"paused": auto}, t)

    def gestures_active(self) -> bool:
        """Gesture control is on and not paused (manual pause, F9, clutch, auto-pause): the big cursor shows."""
        return bool(self.cfg.hands.enabled) and not self.output_paused()

    def set_monitors(self, layout: Screen) -> None:
        """New monitor layout (hot-plug): the cursor stays put if it is still on a monitor."""
        self.mouse.mapper.set_monitors(layout, keep_position=True)

    def cursor_state(self) -> dict[str, Any]:
        """Snapshot for the at-cursor overlay (polled; nothing is published per frame)."""
        g = self.cfg.hands.gestures
        if self.output_paused():
            state = "paused"
        elif not self.gestures.hand_present:
            state = "hand_lost"
        elif self.gestures.autoscroll.active:
            state = "autoscroll"
        elif self.gestures.dragging:
            state = "drag"
        elif self.gestures.pinched:
            state = "pinch"
        elif self.cfg.hands.dwell.enabled and self.dwell.progress > 0:
            state = "dwell"
        else:
            state = "idle"
        x, y = self.mouse.mapper.pos
        return {
            "x": round(x),
            "y": round(y),
            "state": state,
            "pinch": pinch_progress(self.last_ratio, g.pinch_on, g.pinch_off) if self.gestures.hand_present else 0.0,
            "dwell": self.dwell.progress if self.cfg.hands.dwell.enabled else 0.0,
            "frozen": bool(self.mouse.stabilizer is not None and self.mouse.stabilizer.frozen),
            "direction": self.gestures.autoscroll.direction,
            "size": int(self.cfg.hands.cursor.big_cursor_size),
        }

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.gestures.paused = paused
        self.mouse.enabled = not paused
        if paused:
            self.mouse._release_if_down()
            self.gestures.autoscroll.stop()

    def _pause_hook(self, arg: str, data: dict[str, Any]) -> None:
        if arg == "toggle" and "paused" in data:  # open palm: the detector already flipped its own state
            self.set_paused(bool(data["paused"]))
        else:
            self.set_paused({"toggle": not self.paused, "on": True, "off": False}[arg])
        self.bus.publish("app.paused", {"paused": self.paused})

    def _hands_hook(self, arg: str, _data: dict[str, Any]) -> None:
        """hands:right | left | any (e.g. the voice command "use left hand"): which hand drives the cursor."""
        if arg not in ("right", "left", "any"):
            return
        self.hand_prefer = arg
        self.bus.publish("hands.prefer", {"prefer": arg})

    # live reload of bindings + tunables (settings GUI / CLI edits) ------------
    def apply_config(self, cfg: AppConfig) -> None:
        for src, dst in ((cfg.hands.gestures, self.cfg.hands.gestures), (cfg.hands.cursor, self.cfg.hands.cursor)):
            for k, v in vars(src).items():
                setattr(dst, k, v)
        for src, dst in ((cfg.hands.dwell, self.cfg.hands.dwell), (cfg.face, self.cfg.face)):  # type: ignore[assignment]
            for k, v in vars(src).items():
                setattr(dst, k, v)
        c = self.cfg.hands.cursor
        self.mouse.mapper.filter = OneEuro2D(c.min_cutoff, c.beta, c.d_cutoff)
        self.mouse.mapper.refresh_selection()  # hands.cursor.monitors / monitor_reach
        self.mouse.stabilizer = ClickStabilizer(c.click_rewind_ms / 1000.0, c.freeze_margin) if c.click_freeze else None
        self.dwell = DwellDetector(self.cfg.hands.dwell.radius_px, self.cfg.hands.dwell.time_s)
        f = self.cfg.face  # per-user face-click thresholds
        self.brow = BrowRaiseToggle(f.brow_threshold, f.brow_hold)
        self.mouth = MouthOpenTrigger(f.mouth_threshold, f.mouth_hold)
        self.wink = WinkTrigger(f.wink_threshold, f.wink_open_max, f.wink_hold)
        if cfg.hands.prefer != self.cfg.hands.prefer:  # a Settings change wins over an earlier voice command
            self.hand_prefer = cfg.hands.prefer
        self.cfg.hands.prefer = cfg.hands.prefer
        self.cfg.hands.enabled = cfg.hands.enabled
        self.cfg.bindings = cfg.bindings
        self.dispatcher.set_bindings(cfg.bindings)
        self.custom = CustomGestureDetector(cfg.bindings.custom_gestures, cfg.hands.gestures.model_min_conf)

    def hotkeys(self) -> dict[str, str]:
        return hotkey_map(self.cfg.bindings, self.cfg.voice, self.dispatcher.profile)

    def fire(self, category: str, name: str, data: dict[str, Any] | None = None) -> Action:
        """Fire a voice/face/hotkey trigger through the bindings."""
        return self.dispatcher.trigger(category, name, data)

    # hands -------------------------------------------------------------------
    def on_fused(self, fused: FusedHand | None, t: float | None = None) -> list[str]:
        """Feed a multi-camera fused hand (``None`` = no camera sees a hand)."""
        self.last_fused = fused
        if fused is None:
            return self.on_hand(t if t is not None else 0.0, None)
        return self.on_hand(fused.t, fused.landmarks, fused.handedness, pinch=fused.pinch, depth=fused.depth)

    def on_hand(
        self,
        t: float,
        landmarks: Any,
        handedness: str = "Right",
        *,
        pinch: float | None = None,
        depth: float | None = None,
    ) -> list[str]:
        accepted = []
        self._check_auto_pause(t)
        ratio = None
        if landmarks is not None:
            ratio = pinch if pinch is not None else pinch_ratio(as_array(landmarks))
            g = self.gestures
            self.mouse.observe_pinch(t, ratio, self.cfg.hands.gestures.pinch_on, g.pinched, g.dragging)
        self.last_ratio = ratio
        if self.temporal is not None and landmarks is not None:
            probs = self.temporal.push(landmarks, handedness)
            self.fusion.update_model(probs)
            for name in self.custom.update(t, probs):
                self.bus.publish("gesture.custom", {"name": name}, t)
                self.dispatcher.on_gesture(GestureEvent("custom", t, {"name": name}))
                accepted.append(f"custom:{name}")
        pointed = False
        for ev in self.gestures.update(t, landmarks, pinch=pinch):
            if not self.fusion.accept(ev):
                self.bus.publish("gesture.rejected", {"kind": ev.kind}, t)
                continue
            self.bus.publish(f"gesture.{ev.kind}", dict(ev.data), t)
            if not self.paused or ev.kind in ("clutch", "hand_lost", "drag_end"):
                self.dispatcher.on_gesture(ev)
            if ev.kind == "move" and ev.data.get("anchor") != "pinch":
                pointed = True
            accepted.append(ev.kind)
        active = not self.output_paused()
        if self.cfg.hands.dwell.enabled:
            pos = self.mouse.mapper.pos if pointed and active and not self.gestures.pinched else None
            if self.dwell.update(t, pos):
                accepted += self._synthetic_gesture(GestureEvent("dwell", t, {"x": pos[0], "y": pos[1]} if pos else {}))
        if landmarks is not None and depth is not None and active and not self.gestures.pinched:
            if self.push.update(t, depth):
                accepted += self._synthetic_gesture(GestureEvent("push", t, {"depth": depth}))
        elif landmarks is None or depth is None:
            self.push.reset()
        return accepted

    # claps (microphone) ---------------------------------------------------------
    def note_hands(self, t: float, n_hands: int, gap: float | None) -> None:
        """What the camera saw this frame, for clap confirmation (two hands close together)."""
        self._hand_views.append((t, int(n_hands), gap))

    def hands_view(self, t0: float) -> str:
        """``together`` (two hands close around ``t0``), ``seen`` (camera running), ``none`` (no camera frames)."""
        return camera_view(self._hand_views, t0)

    def on_clap(self, kind: str, t: float) -> list[str]:
        """``single`` -> gesture ``clap`` (default pause:off), ``double`` -> ``double_clap`` (pause:on)."""
        name = "clap" if kind == "single" else "double_clap"
        return self._synthetic_gesture(GestureEvent(name, t, {"view": self.hands_view(t)}))

    def _synthetic_gesture(self, ev: GestureEvent) -> list[str]:
        """Gestures detected outside the state machine (dwell, push) go through the same bindings."""
        self.bus.publish(f"gesture.{ev.kind}", dict(ev.data), ev.t)
        self.dispatcher.on_gesture(ev)
        return [ev.kind]

    # face --------------------------------------------------------------------
    def on_faces(
        self,
        t: float,
        boxes: list[tuple[float, float, float]],
        blendshapes: list[dict[str, float]],
        frontal: list[bool | None] | None = None,
    ) -> int | None:
        ids = self.faces.update(boxes)
        if self._enroll_pending and ids:
            self.gate.enroll({fid: b[2] for fid, b in zip(ids, boxes)})
            self._enroll_pending = False
            self.bus.publish("face.enrolled", {"face_id": self.gate.primary}, t)
        elif len(ids) == 1 and self.gate.primary not in ids:
            # one face at the screen but the tracker gave it a new id (looked away, left and came back):
            # it is the user; otherwise the gate would count them as "someone else speaking"
            self.gate.primary = ids[0]
            self.bus.publish("face.enrolled", {"face_id": ids[0], "reacquired": True}, t)
        scores = {fid: self.speaking.update(fid, bs) for fid, bs in zip(ids, blendshapes)}
        active = self.asd.update(scores, self.mic_energy)
        user_score = (
            scores.get(self.gate.primary) if self.gate.primary in scores else max(scores.values(), default=None)
        )
        user = self.gate.primary if self.gate.primary in ids else None
        user_frontal: bool | None = None
        if frontal is not None and ids:
            k = ids.index(user) if user is not None else 0
            if self.gate.primary is None and scores:  # not enrolled: the face the score came from
                k = ids.index(max(scores, key=lambda f: scores[f]))
            user_frontal = frontal[k] if k < len(frontal) else None
        present = (user is not None) if self.gate.primary is not None else bool(ids)
        self.gate.observe(t, active, len(ids), user_score, user_frontal, present)
        if self.cfg.face.brow_raise_toggle and self.gate.primary in ids:
            bs = blendshapes[ids.index(self.gate.primary)]  # type: ignore[arg-type]
            if self.brow.update(t, bs):
                self.bus.publish("face.brow_toggle", {}, t)
                self.fire("face", "brow_raise")
        if self.cfg.face.mouth_open_trigger and self.gate.primary in ids:
            bs = blendshapes[ids.index(self.gate.primary)]  # type: ignore[arg-type]
            if self.mouth.update(t, bs):
                self.bus.publish("face.mouth_open", {}, t)
                self.fire("face", "mouth_open")
        if self.cfg.face.wink_trigger and self.gate.primary in ids:
            bs = blendshapes[ids.index(self.gate.primary)]  # type: ignore[arg-type]
            side = self.wink.update(t, bs)
            if side:
                self.bus.publish(f"face.{side}", {}, t)
                self.fire("face", side)
        return active

    def re_enroll(self) -> None:
        self._enroll_pending = True

    # wifi --------------------------------------------------------------------
    def on_csi_window(self, t: float, csi: Any, rssi: Any, fs: float) -> tuple[str, float]:
        feats = window_features(csi, rssi, fs)
        label, conf = self.csi.classify(feats)
        self.fusion.update_csi(label, conf, t)
        self.bus.publish("wifi.state", {"label": label, "confidence": conf, **feats}, t)
        return label, conf


class AwayApp:  # pragma: no cover - device threads; exercised manually on real hardware
    def __init__(self, cfg: AppConfig, dry_run: bool = False, config_path: Path | None = None) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.config_path = config_path
        self.stop_event = threading.Event()
        self.bus = EventBus()
        self.models_dir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
        try:
            from away.app.performance import current_profile

            self.profile = current_profile(cfg)
        except Exception as exc:  # detection must never stop the app
            from away.app.tuning import Profile

            self.profile = Profile(reasons=[f"hardware detection failed: {exc}"])
        self.asr_device = self.profile.device
        from away.app.camera_share import CameraShare
        from away.control import monitors

        self.camera_share = CameraShare()
        backend: MouseBackend
        if dry_run:
            backend = RecordingBackend()
            self.monitor_layout = monitors.single(1920, 1080, source="dry-run")
        else:
            from away.control.mouse import PynputBackend

            monitors.enable_dpi_awareness()  # no-op when cli/tray main already did it (before pynput/tk)
            self.monitor_layout = monitors.enumerate_monitors()
            backend = PynputBackend()
        screen = self.monitor_layout
        self.monitor_watcher = monitors.MonitorWatcher(self.monitor_layout, float(cfg.hands.cursor.monitor_refresh_s))
        temporal = None
        if cfg.hands.gestures.use_model:
            from away.hands.runtime import TemporalGestureClassifier, find_model

            p = find_model("gesture_tcn.onnx", self.models_dir)
            temporal = TemporalGestureClassifier(p) if p else None
        from away.voice.context import active_process

        self.guard: Any = None
        if cfg.app.auto_pause and not dry_run:
            from away.app.autopause import GuardedBackend, ManualInputGuard

            self.guard = ManualInputGuard(cfg.app.auto_pause_idle_ms)
            backend = GuardedBackend(backend, self.guard)  # type: ignore[assignment]
        self.pipeline = Pipeline(cfg, backend, screen, self.bus, temporal, active_app=active_process, guard=self.guard)
        self.camera_confidences: dict[int, float] = {}  # multi-camera: latest per-view confidence
        self.voice: Any = None  # VoiceController
        self.editor: Any = None  # SelectionEditor (voice.edit_hotkey)
        self.threads: list[threading.Thread] = []
        self._listeners: list[Any] = []
        self.asr_model = ""
        self._notices: set[str] = set()
        self.gate_hint = GateHint()
        self.bus.subscribe("*", self._log)
        self.bus.subscribe("voice.utterance", self._on_utterance)
        from away.app.bigcursor import BigCursor, NullCursorBackend, default_backend

        self.big_cursor = BigCursor(
            NullCursorBackend() if dry_run else default_backend(),
            cfg.hands.cursor.big_cursor_size,
            publish=self._publish_quiet,
        )
        try:
            self.big_cursor.recover()  # a crash last time may have left the cursor big
        except Exception as exc:
            self.bus.publish("cursor.size_error", {"error": str(exc)})
        import atexit

        atexit.register(self._restore_cursor)
        self._camera_episode = 0  # bumps when the cameras recover, so a new failure is notified again
        self.bus.subscribe("cameras.degraded", self._on_cameras_degraded)
        self.bus.subscribe("cameras.restored", self._on_cameras_restored)
        self.bus.subscribe("camera.error", self._on_camera_error)
        self._save_migrations()
        self.bus.publish(
            "monitors.detected", {"layout": self.monitor_layout.summary(), "note": self.monitor_layout.note}
        )
        if not cfg.app.always_on:  # start paused and muted until the user resumes from the tray
            self.pipeline.set_paused(True)

    # notices -------------------------------------------------------------------
    def notice(self, title: str, message: str, key: str | None = None) -> bool:
        """A user-facing heads-up (the tray shows it as a notification). ``key``: at most once per run."""
        if key is not None:
            if key in self._notices:
                return False
            self._notices.add(key)
        self.bus.publish("app.notice", {"title": title, "message": message, "key": key})
        return True

    def _publish_quiet(self, topic: str, data: dict[str, Any]) -> None:
        self.bus.publish(topic, data)

    def _restore_cursor(self) -> None:
        with contextlib.suppress(Exception):
            self.big_cursor.restore()

    def _cursor_size_loop(self, every_s: float = 0.1) -> None:
        """Big cursor while gestures are active; back to normal on pause / auto-pause / disable (debounced)."""
        try:
            while not self.stop_event.wait(every_s):
                try:
                    self.big_cursor.set_level(int(self.cfg.hands.cursor.big_cursor_size))
                    self.big_cursor.update(self.pipeline.gestures_active())
                except Exception as exc:
                    self.bus.publish("cursor.size_error", {"error": str(exc)})
                    return
        finally:
            self._restore_cursor()

    def _release_hand(self) -> None:
        self.pipeline.on_hand(time.monotonic(), None)

    def _on_cameras_degraded(self, ev: Any) -> None:
        msg = str(ev.data.get("message") or "a camera could not be opened and was skipped")
        mode = str(ev.data.get("mode") or "single")
        failed = ",".join(sorted(str(k) for k in (ev.data.get("failed") or {})))
        print(f"warning: {msg}", flush=True)
        title = "Project Away: no camera" if mode == "none" else "Project Away: camera skipped"
        # once per problem: a retry that fails the same way stays quiet until the cameras come back
        self.notice(title, msg, key=f"cameras-degraded:{self._camera_episode}:{mode}:{failed}")
        self._camera_degraded_shown = True

    def _on_camera_error(self, ev: Any) -> None:
        """Single-camera mode: a camera that will not open is retried quietly, but the NVIDIA Broadcast hint
        (Broadcast holds the physical camera as its source) is worth a notification, once."""
        msg = str(ev.data.get("error") or "")
        if "NVIDIA Broadcast is probably using" in msg:
            self.notice("Project Away: camera busy", msg, key=f"camera-broadcast:{self._camera_episode}")

    def _on_cameras_restored(self, ev: Any) -> None:
        msg = str(ev.data.get("message") or "all cameras are working again")
        print(f"info: {msg}", flush=True)
        if not getattr(self, "_camera_degraded_shown", False):
            return  # nothing was reported broken: nothing to announce
        self._camera_degraded_shown = False
        self.notice("Project Away: cameras back", msg, key=f"cameras-restored:{self._camera_episode}")
        self._camera_episode += 1

    def _on_utterance(self, ev: Any) -> None:
        if self.gate_hint.observe(ev.data):
            self.bus.publish(
                "voice.gate_hint",
                {"count": self.gate_hint.count, "gate": ev.data.get("gate")},
            )
            self.notice(
                "Project Away: speech ignored by the face gate",
                f"{self.gate_hint.count} things you said were ignored because the camera did not see you facing "
                "it and talking. Turn off face gating from the tray menu (or Settings > Voice) to always type.",
                key="gate-hint",
            )

    def _save_migrations(self) -> None:
        """Re-save a config an older version wrote, so its migration (e.g. asr_model -> auto) sticks."""
        if not self.cfg.migrations or self.dry_run:
            return
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        try:
            if path.exists():
                cfgmod.save(self.cfg, path)
            self.bus.publish("config.migrated", {"changes": list(self.cfg.migrations), "path": str(path)})
        except Exception as exc:
            self.bus.publish("config.error", {"error": f"could not save the migrated config: {exc}"})

    @property
    def dictation(self) -> Any:  # tray compatibility
        return self.voice.dictation if self.voice is not None else None

    def _log(self, ev: Any) -> None:
        if ev.topic not in ("gesture.move", "gesture.scroll", "gesture.autoscroll", "wifi.state", "app.auto_paused"):
            print(f"[{ev.topic}] {ev.data}", flush=True)

    # public controls used by the tray --------------------------------------
    def set_paused(self, paused: bool) -> None:
        self.pipeline.set_paused(paused)
        self.bus.publish("app.paused", {"paused": paused})

    def toggle_dictation(self) -> None:
        if self.voice is not None:
            self.voice.handle_action("dictation", "toggle")

    def set_mic(self, on: bool) -> None:
        if self.voice is not None:
            self.voice.set_muted(not on)

    def voice_state(self) -> str:
        return self.voice.state if self.voice is not None else "off"

    # monitors ----------------------------------------------------------------
    def _watch_monitors(self) -> None:
        """Re-detect monitors every hands.cursor.monitor_refresh_s (hot-plug without a restart)."""
        w = self.monitor_watcher
        while not self.stop_event.wait(max(0.5, float(w.interval) or 3.0)):
            w.interval = float(self.cfg.hands.cursor.monitor_refresh_s)
            new = w.poll(time.monotonic())
            if new is not None:
                self.monitor_layout = new
                self.pipeline.set_monitors(new)
                self.bus.publish("monitors.changed", {"layout": new.summary()})

    # config hot-reload (settings GUI / CLI write the file) --------------------
    def _watch_config(self) -> None:
        from away import config as cfgmod

        path = self.config_path or paths.config_file()
        last = path.stat().st_mtime if path.exists() else 0.0
        while not self.stop_event.wait(1.0):
            m = path.stat().st_mtime if path.exists() else 0.0
            if m == last:
                continue
            last = m
            try:
                new, _ = cfgmod.load(path)
            except Exception as exc:
                self.bus.publish("config.error", {"error": str(exc)})
                continue
            self.pipeline.apply_config(new)
            if self.voice is not None:
                self._apply_voice(new)
            self._restart_hotkeys()
            self.bus.publish("config.reloaded", {"path": str(path)})

    def _apply_voice(self, new: AppConfig) -> None:
        v = new.voice
        amb = self.voice.ambient
        if amb is not None:
            amb.wake_word = v.wake_word if v.wake_word_enabled else None
            amb.wake_window = v.wake_window
            amb.intent = v.intent
            amb.fallback = effective_ambient_fallback(new)
            face_gate = new.face.enabled and new.face.gate_dictation
            amb.gate = self.pipeline.gate.allow_ambient if face_gate else None
            amb.commands_enabled = v.commands_enabled
            amb.ep.set_sensitivity(v.vad_sensitivity)
        for k, val in vars(v).items():
            setattr(self.cfg.voice, k, val)
        if v.mode != self.voice.mode:
            self.voice.set_mode(v.mode)

    # voice -------------------------------------------------------------------
    def _run_voice_command(self, phrase: str) -> bool:
        act = self.pipeline.fire("voice", phrase)
        return act.kind != "none"

    def _voice_hook(self, kind: str, arg: str, _data: dict[str, Any]) -> None:
        self.voice.handle_action(kind, arg)

    def _ambient_command(self, act: Action, phrase: str) -> None:
        self.pipeline.fire("voice", phrase)

    def _start_voice(self) -> None:
        from away.app.performance import asr_plan
        from away.voice.always_on import AmbientSession, ASRQueue, StreamingEndpointer, VoiceController
        from away.voice.asr import make_asr
        from away.voice.cleanup import LLMCleaner, RuleCleaner, Vocabulary
        from away.voice.context import active_app
        from away.voice.dictation import DictationSession, MicStream
        from away.voice.output import ListSink, make_sink
        from away.voice.vad import make_vad

        v = self.cfg.voice
        vocab = Vocabulary.load(v.vocabulary_file)
        rules = RuleCleaner(vocab, numbers_as_digits=v.numbers_as_digits, numbers_min_value=v.numbers_min_value)
        cleaner = LLMCleaner(rules) if v.llm_cleanup else rules
        prof = self.profile
        try:
            model, cpu_model, slow_note = asr_plan(self.cfg, prof)
        except Exception as exc:  # hardware detection problems must not stop voice
            from away.app.performance import effective_asr_model

            model, cpu_model, slow_note = effective_asr_model(self.cfg, prof), None, None
            self.bus.publish("voice.plan_error", {"error": str(exc)})
        if slow_note:
            self.notice("Project Away: smaller speech model", slow_note, key="asr-downsized")
        self.asr_model = model

        def gpu_fallback(reason: str) -> None:
            self.asr_device = "cpu"
            if cpu_model:
                self.asr_model = cpu_model
            self.bus.publish("voice.gpu_fallback", {"reason": reason, "model": self.asr_model})
            self.notice("Project Away: GPU not used", reason, key="gpu-fallback")

        self.asr_device = prof.device
        asr = make_asr(
            v.asr_backend,
            model,
            prof.device,
            prof.compute_type,
            v.language,
            str(self.models_dir / f"whisper-{model}"),
            vocab.hotwords() if v.hotwords else None,
            cpu_threads=prof.cpu_threads,
            beam_size=prof.beam_size,
            on_fallback=gpu_fallback,
            cpu_model=cpu_model,
            cpu_download_root=str(self.models_dir / f"whisper-{cpu_model}") if cpu_model else None,
        )
        self.bus.publish(
            "voice.asr_ready",
            {
                "model": self.asr_model,
                "device": self.asr_device,
                "compute_type": prof.compute_type if self.asr_device == prof.device else "int8",
            },
        )
        face_gate = self.cfg.face.enabled and self.cfg.face.gate_dictation
        fallback = effective_ambient_fallback(self.cfg)
        gate = self.pipeline.gate.allow if face_gate else None
        sink = ListSink() if self.dry_run else make_sink(v.output)
        commands = self._run_voice_command if v.commands_enabled else None
        dictation = DictationSession(
            asr,
            cleaner,
            sink,
            make_vad(v.vad, v.sample_rate),
            self.bus,
            v.sample_rate,
            gate,
            active_app,
            vocab.terms(),
            command_handler=commands,
        )

        def overloaded() -> None:
            dev = self.asr_device
            fix = (
                "install the GPU pack (Settings > Performance)"
                if dev == "cpu" and prof.gpu_capable
                else "pick a smaller model (voice.asr_model: small or base)"
            )
            self.notice(
                "Project Away: speech model too slow",
                f"Speech recognition ({self.asr_model} on {dev.upper()}) cannot keep up, so older speech was "
                f"skipped; {fix}.",
                key="asr-overload",
            )

        # ASR runs off the mic callback thread; a short bounded backlog, oldest dropped when it overflows
        work = ASRQueue(
            3,
            on_drop=lambda n: self._publish_quiet("voice.dropped", {"reason": "ASR busy", "dropped_total": n}),
            on_overload=overloaded,
        )

        def worker() -> None:
            while not self.stop_event.is_set():
                job = work.get(timeout=0.5)
                if job is None:
                    continue
                try:
                    job()
                except Exception as exc:
                    self.bus.publish("voice.error", {"error": str(exc)})

        submit = work.put

        ambient = AmbientSession(
            asr,
            cleaner,
            sink,
            StreamingEndpointer(
                v.sample_rate,
                sensitivity=v.vad_sensitivity,
                end_silence_ms=v.end_silence_ms,
                min_utterance_ms=v.min_utterance_ms,
                max_utterance_s=v.max_utterance_s,
            ),
            commands=lambda: self.cfg.bindings.voice_phrases(self.pipeline.dispatcher.profile),
            run_command=self._ambient_command,
            bus=self.bus,
            gate=self.pipeline.gate.allow_ambient if face_gate else None,
            context=active_app,
            vocabulary=vocab.terms(),
            wake_word=v.wake_word if v.wake_word_enabled else None,
            wake_window=v.wake_window,
            intent=v.intent,
            fallback=fallback,
            commands_enabled=v.commands_enabled,
            submit=submit,
            gate_info=lambda: self.pipeline.gate.last_explain,
        )
        self.voice = VoiceController(v.mode, dictation, ambient, self.bus)
        if not self.cfg.app.always_on:
            self.voice.set_muted(True)
        for kind in ("dictation", "mic", "voice"):
            self.pipeline.executor.hooks[kind] = functools.partial(self._voice_hook, kind)

        if v.edit_hotkey and not self.dry_run:
            try:
                from away.control.keys import parse_combo
                from away.voice.edit import LLMEditor, SelectionEditor, SystemClipboard

                mod = "cmd" if sys.platform == "darwin" else "ctrl"
                b = self.pipeline.executor.backend
                ed_llm = LLMEditor()
                self.editor = SelectionEditor(
                    asr,
                    SystemClipboard(),
                    lambda: b.hotkey(parse_combo(f"{mod}+c")),
                    lambda: b.hotkey(parse_combo(f"{mod}+v")),
                    ed_llm if ed_llm.configured else None,
                    self.bus,
                    v.sample_rate,
                )  # while it records, on_chunk routes the mic to the editor instead of ambient listening
            except Exception as exc:
                self.bus.publish("edit.unavailable", {"error": str(exc)})

        from away.voice.clap import ClapControl

        def dictation_busy() -> bool:
            vc = self.voice
            if vc is None:
                return False
            if vc.dictation is not None and vc.dictation.recording:
                return True
            amb = vc.ambient
            return bool(amb is not None and vc.listening and amb.ep.in_speech)

        self.claps = ClapControl(
            self.cfg.voice,
            v.sample_rate,
            busy=dictation_busy,
            hands=self.pipeline.hands_view,
            on_candidate=lambda c, ok, view: self._publish_quiet(
                "clap.candidate", {"score": round(c.score, 2), "accepted": ok, "camera": view}
            ),
        )

        def on_chunk(chunk: np.ndarray) -> None:
            self.pipeline.mic_energy = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            if not self.voice.muted:  # a muted mic hears nothing, claps included
                try:
                    for kind, t in self.claps.feed(chunk, time.monotonic()):
                        self.pipeline.on_clap(kind, t)
                except Exception as exc:
                    self._publish_quiet("clap.error", {"error": str(exc)})
            if self.editor is not None and self.editor.recording:
                self.editor.feed(chunk)
                return
            self.voice.feed(chunk)

        def mic_loop() -> None:
            with MicStream(on_chunk, v.sample_rate):
                self.stop_event.wait()

        self.threads.append(threading.Thread(target=mic_loop, daemon=True, name="mic"))
        self.threads.append(threading.Thread(target=worker, daemon=True, name="asr"))

    # hotkeys -----------------------------------------------------------------
    def _restart_hotkeys(self) -> None:
        for lst in self._listeners:
            lst.stop()
        self._listeners = []
        try:
            from pynput import keyboard
        except Exception as exc:
            self.bus.publish("hotkeys.unavailable", {"error": str(exc)})
            return

        def fire(name: str) -> None:
            if (
                name == "ptt"
                and self.voice is not None
                and self.cfg.bindings.resolve("hotkeys", "ptt", self.pipeline.dispatcher.profile)
                == Action("dictation", "toggle")
            ):
                threading.Thread(target=self.voice.toggle, daemon=True).start()
                return
            threading.Thread(target=self.pipeline.fire, args=("hotkeys", name), daemon=True).start()

        combos: dict[str, Any] = {
            combo: functools.partial(fire, name) for combo, name in self.pipeline.hotkeys().items()
        }
        from away.control.keys import canonical, to_pynput_hotkey

        if self.cfg.app.toggle_hotkey:  # global on/off for gesture control
            combos[to_pynput_hotkey(canonical(self.cfg.app.toggle_hotkey))] = lambda: self.set_paused(
                not self.pipeline.paused
            )
        if self.cfg.voice.edit_hotkey and self.editor is not None:
            ed = self.editor
            combos[to_pynput_hotkey(canonical(self.cfg.voice.edit_hotkey))] = lambda: threading.Thread(
                target=ed.toggle, daemon=True
            ).start()
        if combos:
            gh = keyboard.GlobalHotKeys(combos)
            gh.start()
            self._listeners.append(gh)
        v = self.cfg.voice
        if v.mode == "push_to_talk" and self.voice is not None:
            hk = v.hold_key
            target = getattr(keyboard.Key, hk, None) or keyboard.KeyCode.from_char(hk)

            def on_press(k: Any) -> None:
                if k == target:
                    self.voice.ptt_press()

            def on_release(k: Any) -> None:
                if k == target:
                    threading.Thread(target=self.voice.ptt_release, daemon=True).start()

            lst = keyboard.Listener(on_press=on_press, on_release=on_release)
            lst.start()
            self._listeners.append(lst)

    def _camera_loop(self) -> None:
        from away.app.camera_loop import CameraLoop
        from away.hands.tracker import HandTracker, camera_frames

        hand_model = self.models_dir / "hand_landmarker.task"
        face_model = self.models_dir / "face_landmarker.task"
        hands = (
            HandTracker(
                hand_model,
                self.cfg.hands.max_hands,
                prefer=lambda: self.pipeline.hand_prefer,
                mirrored=self.cfg.hands.mirror,
            )
            if self.cfg.hands.enabled
            else None
        )
        faces = None
        if self.cfg.face.enabled and face_model.exists():
            from away.face.landmarker import FaceLandmarker

            faces = FaceLandmarker(face_model)
        prof = self.profile

        def process(n: int, t: float, rgb: Any) -> None:
            if hands is not None and n % max(1, prof.hands_every) == 0:
                fr = hands.process(rgb, t)
                self.pipeline.note_hands(t, fr.n_hands, fr.hands_gap)
                self.pipeline.on_hand(t, fr.landmarks, fr.handedness)
            if faces is not None and n % max(1, prof.face_every) == 0:
                obs = faces.process(rgb, t)
                self.pipeline.on_faces(
                    t, [o.box for o in obs], [o.blendshapes for o in obs], [getattr(o, "frontal", None) for o in obs]
                )

        def run_single(idx: Any, until: Any = None) -> None:
            """Track on one camera (retried while it will not open) until ``until()`` or stop. ``idx`` is a camera
            name or index; it is resolved to the camera's current DirectShow index at every (re)open."""
            hint = self._hint_for(idx)

            def open_frames() -> Any:
                return camera_frames(
                    idx,
                    self.cfg.hands.mirror,
                    prof.camera_width,
                    prof.camera_height,
                    prof.camera_fps,
                    hint=hint,
                    fourcc=self._fourcc_for(idx),
                    virtual_fallback=bool(self.cfg.hands.virtual_fallback),
                )

            CameraLoop(
                open_frames,
                process,
                self.stop_event,
                cameras=[idx],
                share=self.camera_share,
                publish=self._publish_quiet,
                on_release=self._release_hand,
                until=until,
            ).run()

        self._camera_list(log=True)  # the enumerated cameras (source, index, name, virtual) -> away.log
        try:
            if len(self.cfg.hands.cameras) >= 2 and hands is not None:
                self._multicam_loop(hands, faces, run_single)
                return
            run_single(self.cfg.hands.camera_index)
        finally:
            # close the landmarkers only after this thread stopped calling them (shutdown race)
            for tr in (hands, faces):
                close = getattr(tr, "close", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        close()

    def _fourcc_for(self, device: Any) -> str:
        """The pixel format to request for a camera: its own ``fourcc`` in a multi-camera set (also when it
        tracks alone after the others failed), else ``hands.camera_fourcc``."""
        h = self.cfg.hands
        spec = next((c for c in h.cameras if c.device == device), None)
        return str(spec.fourcc if spec is not None else h.camera_fourcc)

    def _hint_for(self, device: Any) -> int | None:
        """The index saved alongside a camera name (used only when no camera list is available)."""
        h = self.cfg.hands
        if not h.cameras and device == h.camera_index:
            return h.camera_hint
        return next((c.index_hint for c in h.cameras if c.device == device), None)

    def _camera_list(self, log: bool = False) -> list[Any] | None:
        """The cameras the OS lists now (Windows: DirectShow order), logged when it changed (or ``log``);
        None when listing fails. Overridden in tests."""
        try:
            from away.multicam.capture import list_cameras

            return list_cameras(log_list=True if log else None)
        except Exception as exc:
            logging.getLogger("away.cameras").warning("camera listing failed: %s", exc)
            return None

    # ------------------------------------------------------------------------------ multi-camera
    def _start_cameras(self, specs: list[Any], pairer: Any, session: threading.Event, open_lock: Any) -> list[Any]:
        """Start one capture thread per camera (overridden in tests with simulated cameras)."""
        import functools

        from away.multicam.capture import CameraThread, open_capture

        opener = functools.partial(open_capture, virtual_fallback=bool(self.cfg.hands.virtual_fallback))
        threads = [CameraThread(i, c, pairer, session, open_lock=open_lock, opener=opener) for i, c in enumerate(specs)]
        for th in threads:
            th.start()
        return threads

    @staticmethod
    def _end_cameras(session: threading.Event, threads: list[Any]) -> None:
        session.set()
        for th in threads:
            th.join(timeout=2.0)

    @staticmethod
    def _serial_open() -> bool:
        """Open cameras one after another (Windows: two webcams opening at once is slow and flaky)."""
        return sys.platform.startswith("win")

    def _probe_timeout(self, n: int, probe_s: float) -> float:
        # opening one after another: give every extra camera half a probe more
        return probe_s + (probe_s / 2.0) * (n - 1) if self._serial_open() else probe_s

    def _probe_specs(self, specs: list[Any], probe_s: float) -> dict[int, str]:
        """Open ``specs`` briefly (while the working camera keeps tracking): ``{}`` when all send frames."""
        from away.app.camera_loop import probe_cameras
        from away.multicam.capture import FramePairer

        session = threading.Event()
        pairer = FramePairer(len(specs), 0, 1.0, 1.0)
        lock = threading.Lock() if self._serial_open() else None
        threads = self._start_cameras(list(specs), pairer, session, lock)
        try:
            return probe_cameras(threads, pairer.seen, self._probe_timeout(len(specs), probe_s), self.stop_event)
        finally:
            self._end_cameras(session, threads)

    def _single_until_retry(
        self, run_single: Any, device: Any, failed_specs: list[Any], every_s: float, probe_s: float
    ) -> None:
        """Track on ``device`` alone; every ``every_s`` probe the failed cameras in the background and return
        (to retry the full multi-camera set) once they all deliver frames. Tracking is never interrupted by a
        probe that still fails."""
        from away.app.camera_loop import retry_labels

        ready = threading.Event()
        done = threading.Event()

        def prober() -> None:
            while not done.wait(every_s):
                if self.stop_event.is_set():
                    return
                if self.camera_share.requested():  # Settings is using the cameras: do not grab them
                    continue
                try:
                    bad = self._probe_specs(failed_specs, probe_s)
                except Exception as exc:
                    bad = {0: f"probe error: {exc}"}
                self.bus.publish("cameras.retry", {"ok": not bad, "failed": retry_labels(failed_specs, bad)})
                if not bad and not done.is_set():
                    ready.set()
                    return

        th = threading.Thread(target=prober, name="camera-retry", daemon=True)
        th.start()
        try:
            run_single(device, ready.is_set)
        finally:
            done.set()

    def _multicam_loop(
        self,
        first_tracker: Any,
        faces: Any,
        run_single: Any = None,
        probe_s: float = MULTICAM_PROBE_S,
        retry_s: float = MULTICAM_RETRY_S,
        none_retry_s: float = MULTICAM_NONE_RETRY_S,
        trackers: list[Any] | None = None,
    ) -> None:
        """Two or more cameras: threaded capture -> timestamp pairing -> per-view hands -> fusion -> pipeline.

        * Start-up probe: a camera fails only when its thread dies with an error, or it sends no frame within
          ``probe_s`` (Windows webcams are slow to open; on Windows they open one after another).
        * One camera left: track on it alone (``run_single``) and re-probe the failed cameras every
          ``retry_s`` without touching the working one; when they deliver again, go back to fusion (backing off
          up to MULTICAM_RETRY_MAX_S if the set keeps failing right after a good probe).
        * No camera works: nothing is claimed; the whole set is retried every ``none_retry_s``.
        * The captures restart after a hand-over to Settings (camera_share) without restarting the app.
        """
        from away.app.camera_loop import degrade_plan, is_shutdown_error, probe_cameras
        from away.multicam.capture import FramePairer, pin_camera_set
        from away.multicam.fusion import MultiViewFuser, ViewObs, hand_crop, sharpness_score
        from away.multicam.geometry import load_extrinsics

        cams, mc = self.cfg.hands.cameras, self.cfg.hands.multicam
        primary = next(i for i, c in enumerate(cams) if c.role == "primary")
        own_trackers = trackers is None
        if trackers is None:
            from away.hands.tracker import HandTracker

            model = self.models_dir / "hand_landmarker.task"
            first_tracker.selector.mirrored = cams[0].mirror  # each camera's own frame orientation
            trackers = [first_tracker] + [
                HandTracker(
                    model, self.cfg.hands.max_hands, prefer=lambda: self.pipeline.hand_prefer, mirrored=c.mirror
                )
                for c in cams[1:]
            ]
        ex = load_extrinsics()
        fuser = MultiViewFuser(cams, mc, ex)
        share = self.camera_share
        timeout = self._probe_timeout(len(cams), probe_s)
        degraded = False
        backoff = retry_s
        after_retry = False
        try:
            while not self.stop_event.is_set():
                # every attempt: bare indices from older versions are pinned to camera names (a virtual camera at
                # that index is swapped for a real one: the auto-skip), so each open resolves by name (hot-plug safe)
                listed = self._camera_list()
                run_cams = pin_camera_set(cams, listed) if listed else list(cams)
                devices = [c.device for c in run_cams]
                session = threading.Event()
                pairer = FramePairer(len(cams), primary, mc.pair_tolerance_ms / 1000.0, mc.dropout_ms / 1000.0)
                lock = threading.Lock() if self._serial_open() else None
                threads = self._start_cameras(run_cams, pairer, session, lock)
                failed = probe_cameras(threads, pairer.seen, timeout, self.stop_event)
                if self.stop_event.is_set():
                    self._end_cameras(session, threads)
                    break
                working = len(cams) - len(failed)
                if working < 2 and after_retry:  # the set failed again right after a good probe: back off
                    backoff = min(backoff * 2.0, MULTICAM_RETRY_MAX_S)
                after_retry = False
                retry_every = none_retry_s if working == 0 else (backoff if working == 1 else None)
                mode, device, why = degrade_plan(devices, primary, failed, retry_every)
                if failed:
                    errs = {str(devices[i]): failed[i] for i in sorted(failed)}
                    self.bus.publish("cameras.degraded", {"failed": errs, "message": why, "mode": mode})
                if mode != "multi":
                    self._end_cameras(session, threads)
                    share.clear_owner()
                    degraded = True
                    if mode == "none" or run_single is None:
                        self.stop_event.wait(none_retry_s)
                        continue
                    self._single_until_retry(
                        run_single, device, [run_cams[i] for i in sorted(failed)], backoff, probe_s
                    )
                    after_retry = True
                    continue
                backoff = retry_s
                if degraded:
                    degraded = False
                    self.bus.publish(
                        "cameras.restored",
                        {
                            "cameras": devices,
                            "message": f"cameras {', '.join(str(d) for d in devices)} are working again; "
                            "multi-camera tracking is back on.",
                        },
                    )
                self.bus.publish("cameras.started", {"cameras": len(cams), "calibrated": fuser.calibrated})
                dropped: set[int] = set()
                next_check = 0.0
                handover = False
                try:
                    while not self.stop_event.is_set():
                        now = time.monotonic()
                        if now >= next_check:
                            next_check = now + 1.0
                            share.publish_owner(devices)
                            if share.requested():
                                handover = True
                                break
                        s = pairer.pop_set()
                        if s is None:
                            time.sleep(0.002)
                            continue
                        t_ref, frames = s
                        alive = pairer.alive()
                        for i, ok in enumerate(alive):
                            if not ok and i not in dropped:
                                dropped.add(i)
                                self.bus.publish("cameras.dropped", {"camera": i, "error": threads[i].error})
                            elif ok and i in dropped:
                                dropped.discard(i)
                                self.bus.publish("cameras.recovered", {"camera": i})
                        views: list[Any] = []
                        try:
                            for i, item in enumerate(frames):
                                if item is None:
                                    views.append(None)
                                    continue
                                t_i, rgb = item
                                fr = trackers[i].process(rgb, t_i)
                                note = getattr(self.pipeline, "note_hands", None)
                                if i == primary and note is not None:
                                    note(t_i, getattr(fr, "n_hands", 1), getattr(fr, "hands_gap", None))
                                if fr.landmarks is None:
                                    views.append(None)
                                    continue
                                h, w = rgb.shape[:2]
                                crop = hand_crop(rgb, fr.landmarks)
                                sharp = sharpness_score(crop.mean(axis=2)) if crop.size else None
                                views.append(ViewObs(i, t_i, fr.landmarks, fr.handedness, fr.score, (w, h), sharp))
                            fused = fuser.update(t_ref, views)
                            self.camera_confidences = dict(fuser.last_confidences)
                            self.pipeline.on_fused(fused, t_ref)
                            prim_frame = frames[primary]
                            if faces is not None and prim_frame is not None:
                                t_p, rgb_p = prim_frame
                                obs = faces.process(rgb_p, t_p)
                                self.pipeline.on_faces(
                                    t_p,
                                    [o.box for o in obs],
                                    [o.blendshapes for o in obs],
                                    [getattr(o, "frontal", None) for o in obs],
                                )
                        except Exception as exc:
                            if self.stop_event.is_set() or is_shutdown_error(exc):
                                self.stop_event.set()
                                return
                            raise
                finally:
                    self._end_cameras(session, threads)
                if handover:
                    self.pipeline.on_fused(None, time.monotonic())
                    share.clear_owner()
                    share.ack_released(devices)
                    self.bus.publish("camera.handover", {"released": True, "cameras": devices})
                    while share.requested() and not self.stop_event.is_set():
                        self.stop_event.wait(0.5)
                    share.clear_released()
                    self.bus.publish("camera.handover", {"released": False, "cameras": devices})
        finally:
            share.clear_owner()
            if own_trackers:
                for tr in trackers[1:]:
                    with contextlib.suppress(Exception):
                        tr.close()

    def _wifi_loop(self) -> None:
        from away.wifi.sources import IAXSource, PicoScenesSource, RSSISource, stack_csi

        w = self.cfg.wifi
        src: Any
        if w.source == "picoscenes":
            src = PicoScenesSource(w.path or "")
        elif w.source == "iax":
            src = IAXSource(w.path or "")
        else:
            src = RSSISource(w.interface)
        win = max(int(w.sample_rate), 10)
        buf_csi: list[Any] = []
        buf_rssi: list[float] = []
        for fr in src:
            if self.stop_event.is_set():
                break
            if fr.csi is not None:
                buf_csi.append(fr.csi)
            if fr.rssi is not None:
                buf_rssi.append(fr.rssi)
            n = len(buf_csi) if buf_csi else len(buf_rssi)
            if n >= (win if buf_csi else 10):
                csi = stack_csi(buf_csi)
                self.pipeline.on_csi_window(
                    time.monotonic(), csi, np.array(buf_rssi) if buf_rssi else None, w.sample_rate if buf_csi else 10.0
                )
                buf_csi = buf_csi[len(buf_csi) // 2 :]
                buf_rssi = buf_rssi[len(buf_rssi) // 2 :]

    def start(self) -> None:
        if self.cfg.hands.enabled or self.cfg.face.enabled:
            self.threads.append(threading.Thread(target=self._camera_loop, daemon=True, name="camera"))
        if self.cfg.voice.enabled:
            self._start_voice()
        if self.cfg.wifi.enabled:
            self.threads.append(threading.Thread(target=self._wifi_loop, daemon=True, name="wifi"))
        self.threads.append(threading.Thread(target=self._watch_config, daemon=True, name="config-watch"))
        if self.cfg.hands.enabled:
            self.threads.append(threading.Thread(target=self._cursor_size_loop, daemon=True, name="cursor-size"))
        if self.monitor_watcher.enabled:
            self.threads.append(threading.Thread(target=self._watch_monitors, daemon=True, name="monitors"))
        for th in self.threads:
            th.start()
        if self.guard is not None:
            try:
                from away.app.autopause import start_listeners

                self._input_listeners = start_listeners(self.guard)
            except Exception as exc:
                self.bus.publish("autopause.unavailable", {"error": str(exc)})
        self._restart_hotkeys()

    def stop(self, join_timeout: float = 3.0) -> None:
        # Stop producing new events first, then make a best-effort release of
        # any synthetic mouse button still held by an in-progress drag.
        self.stop_event.set()
        # Let the camera thread finish its current frame and close MediaPipe itself before the
        # interpreter shuts the executors down ("cannot schedule new futures after shutdown").
        for th in getattr(self, "threads", []):
            if th.name == "camera" and th.is_alive() and th is not threading.current_thread():
                th.join(timeout=join_timeout)
        try:
            self.pipeline.set_paused(True)
        except Exception as exc:
            # Shutdown must continue even if the OS input backend has failed.
            self.bus.publish("mouse.release_error", {"error": str(exc)})
        self._restore_cursor()  # the user's own cursor size, right away (no debounce)
        for lst in [*self._listeners, *getattr(self, "_input_listeners", [])]:
            try:
                lst.stop()
            except Exception as exc:
                self.bus.publish("listener.stop_error", {"error": str(exc)})

    def run_forever(self) -> None:
        self.start()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
