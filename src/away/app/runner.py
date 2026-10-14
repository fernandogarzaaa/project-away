"""Wire the subsystems together on the event bus.

``Pipeline`` is pure logic (no devices) so it can be driven by synthetic
frames in tests and by ``away demo``. ``AwayApp`` adds the camera, mic,
hotkeys and Wi-Fi threads around it.
"""

from __future__ import annotations

import contextlib
import copy
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
from away.app.tunables import profile_tuned
from away.bindings import Action
from away.comfort.hub import ComfortHub
from away.config import AppConfig
from away.control.mouse import MouseBackend, MouseController, RecordingBackend
from away.events import EventBus
from away.face.commands import BrowRaiseToggle, MouthOpenTrigger, WinkTrigger
from away.face.gate import SpeechGate
from away.face.headgaze import HeadGaze, calibration_camera
from away.face.speaker import ActiveSpeakerDetector
from away.face.speaking import SpeakingScorer
from away.face.tracking import FaceTracker
from away.fusion import GestureFusion
from away.hands.cursor import CursorMapper, Screen
from away.hands.gestures import GestureEvent, GestureStateMachine
from away.hands.landmarks import as_array, pinch_ratio
from away.hands.palmscroll import PalmScroller
from away.hands.stabilize import ClickStabilizer, DwellDetector, pinch_progress
from away.multicam.fusion import FusedHand, PushDetector
from away.pointer.smart import SmartPointer
from away.pointer.smoothing import make_cursor_filter
from away.pointer.targets import NullBackend
from away.studio.engine import StudioRuntime, fire_studio_gesture
from away.voice.always_on import GateHint
from away.voice.clap import camera_view
from away.wifi.classifier import CoarseCSIClassifier
from away.wifi.features import window_features

# Multi-camera start-up. Windows MSMF/DirectShow can take several seconds to open a USB webcam (more when
# two open at once), so a camera only counts as failed when it errors or sends nothing for MULTICAM_PROBE_S.
# v0.1.13: room for every DirectShow mode plus Media Foundation; v0.1.16: every mode is measured for 1.5 s and a
# too-slow camera sweeps every format and size once (capture.SWEEP_BUDGET_S), so more room
MULTICAM_PROBE_S = 45.0
HANDOVER_FREE_S = 15.0  # a hand-over waits at most this long for this process's cameras to be closed
SPEED_CHECK_S = 5.0  # after this long streaming, each camera's measured frame rate is logged (ok / too slow)
MULTICAM_RETRY_S = 45.0  # after falling back to one camera: re-probe the failed ones (default hands.camera_retry_s)
AUTO_PAUSE_DWELL_S = 0.5  # v0.1.23: auto-pause ends only after the mouse/keyboard stayed quiet this long
HAND_QUIET_S = 20.0  # v0.1.23: no background camera probe while a hand was tracked this recently
MAX_DEFER_S = 1800.0  # ...but a probe is not put off longer than this
DEVICE_CHECK_S = 30.0  # v0.1.23: how often a waiting camera retry checks whether the camera list changed
MULTICAM_RETRY_MAX_S = 900.0  # back-off ceiling (default hands.camera_retry_max_s; v0.1.19: 15 min)
MULTICAM_NONE_RETRY_S = 10.0  # no camera works at all: retry the whole set this often
# v0.1.23: with no camera working there is nothing a probe could disturb, so the "no camera" retry keeps the
# v0.1.19 schedule (twice as long each time up to 15 min, 8 tries) instead of the slow 5/15/60 min one
NONE_RETRY_MAX_S = 900.0
NONE_RETRY_LIMIT = 8


def effective_ambient_fallback(cfg: AppConfig) -> str:
    """Always-on without a wake word: ``type`` when the user turned the face gate off (that must still type;
    ``commands_only`` there silently dropped all dictation), else ``voice.ambient_fallback``."""
    if cfg.voice.ambient_fallback == "type" or not cfg.face.gate_dictation:
        return "type"
    return cfg.voice.ambient_fallback


CLICK_ACTIONS = ("click", "double_click", "right_click", "middle_click")


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
        pointer_backend: Any = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.gestures = GestureStateMachine(cfg.hands.gestures)
        self.palm = PalmScroller(cfg.hands.palm_scroll)  # palm-glide scroll (pointer fingertip on the helper palm)
        self._pointer_id: int | None = None
        prim = next((c for c in cfg.hands.cameras if c.role == "primary"), None)
        self.palm.mirrored = bool(prim.mirror if prim is not None else cfg.hands.mirror)
        c = cfg.hands.cursor
        stab = ClickStabilizer(c.click_rewind_ms / 1000.0, c.freeze_margin) if c.click_freeze else None
        self.mouse = MouseController(backend or RecordingBackend(), CursorMapper(screen, c), stabilizer=stab)
        self.guard = guard  # ManualInputGuard: physical mouse/keyboard use pauses gesture output
        self._auto_paused = False
        self._auto_clear_since: float | None = None
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
        self.hand_mode = cfg.hands.roles.mode  # hands:auto|right|left|any (voice) changes it until the next restart
        self.executor = ActionExecutor(
            self.mouse, cfg.bindings.allow_shell, hooks={"pause": self._pause_hook, "hands": self._hands_hook}
        )
        self.dispatcher = Dispatcher(
            cfg.bindings, self.executor, self.gestures, active_app=active_app, paused=self.output_paused
        )
        self.dispatcher.on_fire = self._on_fire
        self.studio = StudioRuntime(cfg, profile=self.dispatcher._profile)  # zones + Studio gestures (v0.1.20)
        self._base_cfg = copy.deepcopy(cfg)  # the global tunables, before any per-app profile tuning
        self.dispatcher.on_profile = self._on_profile
        first = self.dispatcher._profile
        if cfg.bindings.profiles[first].tuning:  # a pinned profile with its own tuning
            self.apply_config(profile_tuned(self._base_cfg, first), base=False)
        self.custom = CustomGestureDetector(cfg.bindings.custom_gestures, cfg.hands.gestures.model_min_conf)
        # head tracking + gaze (v0.1.19): acts only when hands.cursor.head_assist / monitor_switch are on or a
        # gaze calibration exists for this camera
        self.headgaze = HeadGaze(
            cfg,
            self.mouse,
            calibration_camera(cfg),
            paused=self.output_paused,
            hand_active=lambda: bool(self.gestures.hand_present),
            busy=lambda: bool(self.gestures.pinched or self.gestures.dragging),
            publish=lambda topic, data: self.bus.publish(topic, data),
        )
        self.mouse.gaze_point = lambda: self.headgaze.estimator.point if self.headgaze.estimator else None
        # v0.1.20 smart pointer: magnetic targets + "click <name>" (Windows UI Automation; a no-op elsewhere).
        # AwayApp starts its background element cache; tests / demo never do, so the magnet has no targets there.
        self.smart = SmartPointer(
            cfg,
            self.mouse,
            backend=pointer_backend if pointer_backend is not None else NullBackend("pipeline"),
            publish=lambda topic, data: self.bus.publish(topic, data),
            is_bound=lambda p: self.dispatcher.resolve("voice", p).kind != "none",
            paused=self.output_paused,
        )
        self.mouse.refiner = self.smart
        # v0.1.20 attention gate + arm fatigue + break reminders (away.comfort)
        self.comfort = ComfortHub(cfg, self.headgaze, publish=lambda topic, data: self.bus.publish(topic, data))

    @property
    def hand_prefer(self) -> str:
        """The hand-roles mode (kept under its v0.1.9 name for callers that read it)."""
        return self.hand_mode

    @hand_prefer.setter
    def hand_prefer(self, value: str) -> None:
        self.hand_mode = value

    def _on_profile(self, name: str) -> None:
        """The focused app picked another bindings profile: apply its tuning (or go back to the global values)."""
        self.apply_config(profile_tuned(self._base_cfg, name), base=False)
        self.studio.set_profile(name)

    def _on_fire(self, trig: str, act: Action) -> None:
        self.bus.publish("action.fired", {"trigger": trig, "action": str(act)})
        if act.kind == "mouse" and act.arg in CLICK_ACTIONS:
            self.headgaze.on_click(self.mouse.mapper.pos)  # gaze drift correction: you look where you click

    # pause -------------------------------------------------------------------
    def output_paused(self) -> bool:
        """Paused by the user, or auto-paused because the physical mouse/keyboard is in use."""
        return self.paused or self._auto_paused

    def _check_auto_pause(self, t: float) -> None:
        auto = bool(self.guard is not None and self.guard.active)
        if auto == self._auto_paused:
            self._auto_clear_since = None
            return
        if not auto:  # v0.1.23: resume only after the mouse/keyboard stayed quiet AUTO_PAUSE_DWELL_S (no flapping)
            since = getattr(self, "_auto_clear_since", None)
            if since is None:
                self._auto_clear_since = t
                return
            if t - since < AUTO_PAUSE_DWELL_S:
                return
        self._auto_clear_since = None
        self._auto_paused = auto
        if auto:
            self.mouse._release_if_down()
            self.gestures.autoscroll.stop()
            self.palm.stop_momentum()
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
        elif self.palm.engaged or self.palm.momentum:
            state = "palmscroll"
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
            "direction": self.palm.direction if state == "palmscroll" else self.gestures.autoscroll.direction,
            "size": int(self.cfg.hands.cursor.big_cursor_size),
            "attention": self.comfort.gate.state if self.cfg.attention.indicator else "off",
        }

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.gestures.paused = paused
        self.mouse.enabled = not paused
        if paused:
            self.mouse._release_if_down()
            self.gestures.autoscroll.stop()
            self.palm.stop_momentum()

    def _pause_hook(self, arg: str, data: dict[str, Any]) -> None:
        if arg == "toggle" and "paused" in data:  # open palm: the detector already flipped its own state
            self.set_paused(bool(data["paused"]))
        else:
            self.set_paused({"toggle": not self.paused, "on": True, "off": False}[arg])
        self.bus.publish("app.paused", {"paused": self.paused})

    def _hands_hook(self, arg: str, _data: dict[str, Any]) -> None:
        """hands:auto | right | left | any (voice: "use left hand", "use either hand" = auto): the pointer hand."""
        if arg not in ("auto", "right", "left", "any"):
            return
        self.hand_mode = arg
        self.bus.publish("hands.roles", {"mode": arg})

    # live reload of bindings + tunables (settings GUI / CLI edits) ------------
    def apply_config(self, cfg: AppConfig, *, base: bool = True) -> None:
        """Take over a (re)loaded config. ``base``: it is the global config (the file); the focused app's per-app
        profile ``tuning`` is layered on top. ``base=False``: an already layered copy (a profile switch)."""
        if base:
            self._base_cfg = copy.deepcopy(cfg)
            prof = cfg.bindings.profile_for_app(self.dispatcher.last_app)
            if cfg.bindings.profiles[prof].tuning:
                cfg = profile_tuned(cfg, prof)
        for src, dst in ((cfg.hands.gestures, self.cfg.hands.gestures), (cfg.hands.cursor, self.cfg.hands.cursor)):
            for k, v in vars(src).items():
                setattr(dst, k, v)
        for src, dst in (  # type: ignore[assignment]
            (cfg.hands.dwell, self.cfg.hands.dwell),
            (cfg.face, self.cfg.face),
            (cfg.gaze, self.cfg.gaze),
        ):
            for k, v in vars(src).items():
                setattr(dst, k, v)
        c = self.cfg.hands.cursor
        self.mouse.mapper.filter = make_cursor_filter(c)  # hands.cursor.smoothing (v0.1.20)
        self.mouse.mapper.refresh_selection()  # hands.cursor.monitors / monitor_reach
        self.mouse.stabilizer = ClickStabilizer(c.click_rewind_ms / 1000.0, c.freeze_margin) if c.click_freeze else None
        self.dwell = DwellDetector(self.cfg.hands.dwell.radius_px, self.cfg.hands.dwell.time_s)
        f = self.cfg.face  # per-user face-click thresholds
        self.brow = BrowRaiseToggle(f.brow_threshold, f.brow_hold)
        self.mouth = MouthOpenTrigger(f.mouth_threshold, f.mouth_hold)
        self.wink = WinkTrigger(f.wink_threshold, f.wink_open_max, f.wink_hold)
        if cfg.hands.roles.mode != self.cfg.hands.roles.mode:  # a Settings change wins over an earlier voice command
            self.hand_mode = cfg.hands.roles.mode
        for src, dst in ((cfg.hands.roles, self.cfg.hands.roles), (cfg.hands.palm_scroll, self.cfg.hands.palm_scroll)):  # type: ignore[assignment]
            for k, v in vars(src).items():
                setattr(dst, k, v)
        self.cfg.hands.enabled = cfg.hands.enabled
        self.cfg.bindings = cfg.bindings
        self.dispatcher.set_bindings(cfg.bindings)
        self.custom = CustomGestureDetector(cfg.bindings.custom_gestures, cfg.hands.gestures.model_min_conf)
        self.cfg.zones, self.cfg.studio = cfg.zones, cfg.studio
        self.studio.apply_config(self.cfg)
        if hasattr(self, "comfort"):  # a pinned profile's tuning is applied in __init__, before the hub exists
            self.comfort.apply_config(cfg)

    def attention_state(self) -> dict[str, Any]:
        """Attention gate + arm-fatigue snapshot for the UI (v0.1.20)."""
        return self.comfort.attention_state()

    def hotkeys(self) -> dict[str, str]:
        return hotkey_map(self.cfg.bindings, self.cfg.voice, self.dispatcher.profile)

    def fire(self, category: str, name: str, data: dict[str, Any] | None = None) -> Action:
        """Fire a voice/face/hotkey trigger through the bindings (the action engine logs a voice trigger's input
        as the ``voice`` source)."""
        if category == "voice":
            from away.actions_engine import VOICE, attributed

            with attributed(VOICE):
                return self._fire(category, name, data)
        return self._fire(category, name, data)

    def _fire(self, category: str, name: str, data: dict[str, Any] | None = None) -> Action:
        if category == "voice":
            act = self.smart.voice(name)  # "click <name>" / "click that" ... (None: not one, or a bound phrase)
            if act is not None:
                return act
        return self.dispatcher.trigger(category, name, data)

    def voice_phrases(self) -> dict[str, Action]:
        """The bound voice phrases for always-on listening, plus the smart pointer commands (v0.1.20)."""
        return self.smart.phrases(self.cfg.bindings.voice_phrases(self.dispatcher.profile))

    # hands -------------------------------------------------------------------
    def on_fused(
        self, fused: FusedHand | None, t: float | None = None, helper: FusedHand | None = None, **ids: Any
    ) -> list[str]:
        """Feed a multi-camera fused pointer hand (``None`` = no camera sees it) and the fused helper hand.
        ``ids``: ``pointer_id`` / ``helper_id`` / ``promoted`` from the primary camera's roles."""
        self.last_fused = fused
        h_lm = helper.landmarks if helper is not None else None
        h_3d = helper.points3d if helper is not None else None
        if fused is None:
            return self.on_hand(t if t is not None else 0.0, None, helper=h_lm, **ids)
        return self.on_hand(
            fused.t,
            fused.landmarks,
            fused.handedness,
            pinch=fused.pinch,
            depth=fused.depth,
            helper=h_lm,
            points3d=fused.points3d,
            helper_points3d=h_3d,
            **ids,
        )

    def on_hand(
        self,
        t: float,
        landmarks: Any,
        handedness: str = "Right",
        *,
        pinch: float | None = None,
        depth: float | None = None,
        helper: Any = None,
        points3d: Any = None,
        helper_points3d: Any = None,
        pointer_id: int | None = None,
        helper_id: int | None = None,
        promoted: bool = False,
    ) -> list[str]:
        """One frame: the pointer hand (``landmarks``, drives everything) and the helper hand (``helper``, only
        palm-glide scroll). ``promoted``: the pointer was the helper until now (its flat palm must not pause)."""
        accepted = []
        self._check_auto_pause(t)
        if not self.comfort.allow_hand(t, landmarks):  # attention gate / resting arm: the hand does not count
            landmarks = pinch = depth = points3d = helper = helper_points3d = None
        if pointer_id is not None and self._pointer_id is not None and pointer_id != self._pointer_id:
            # the pointer role moved to the other hand: end the old hand's drag/scroll and re-anchor the cursor
            # on the new one (no jump)
            for ev in self.gestures.update(t, None):
                self.bus.publish(f"gesture.{ev.kind}", dict(ev.data), t)
                self.dispatcher.on_gesture(ev)
                accepted.append(ev.kind)
        if pointer_id is not None or landmarks is None:
            self._pointer_id = pointer_id
        if promoted:
            self.gestures.block_clutch()
        palm_events: list[GestureEvent] = []
        if self.cfg.hands.enabled:
            palm_events = self.palm.update(
                t,
                landmarks,
                helper,
                points3d,
                helper_points3d,
                self.cfg.hands.multicam.palm_length_m,
            )
        suppress = self.palm.active
        zone_ok = self.studio.check_zone(landmarks)  # Gesture Studio zones gate activations (v0.1.20)
        ratio = None
        if landmarks is not None:
            ratio = pinch if pinch is not None else pinch_ratio(as_array(landmarks))
            g = self.gestures
            self.mouse.observe_pinch(t, ratio, self.cfg.hands.gestures.pinch_on, g.pinched, g.dragging)
        self.last_ratio = ratio
        if self.temporal is not None and landmarks is not None:
            probs = self.temporal.push(landmarks, handedness)
            self.fusion.update_model(probs)
            for name in self.custom.update(t, probs) if not suppress and zone_ok else []:
                self.bus.publish("gesture.custom", {"name": name}, t)
                self.dispatcher.on_gesture(GestureEvent("custom", t, {"name": name}))
                accepted.append(f"custom:{name}")
        if landmarks is not None and not suppress:
            for name in self.studio.update(t, landmarks, handedness, helper):
                accepted += fire_studio_gesture(self, name, t)
        pointed = False
        for ev in self.gestures.update(t, landmarks, pinch=pinch, suppress=suppress):
            if not self.studio.event_allowed(ev.kind, ev.data):
                self.bus.publish("gesture.zoned", {"kind": ev.kind}, t)
                continue
            if ev.kind not in ("move", "hand_lost", "drag_end", "autoscroll", "autoscroll_stop"):
                self.palm.stop_momentum()  # a pointer gesture ends the palm-scroll coast
            if not self.fusion.accept(ev):
                self.bus.publish("gesture.rejected", {"kind": ev.kind}, t)
                continue
            self.bus.publish(f"gesture.{ev.kind}", dict(ev.data), t)
            if ev.kind == "move" and ev.data.get("anchor") != "pinch" and "x" in ev.data:
                self.headgaze.on_hand_move(t, (float(ev.data["x"]), float(ev.data["y"])))  # look-to-jump
            elif ev.kind == "hand_lost":
                self.headgaze.on_hand_lost()
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
        if landmarks is not None and depth is not None and active and not self.gestures.pinched and not suppress:
            if self.push.update(t, depth):
                accepted += self._synthetic_gesture(GestureEvent("push", t, {"depth": depth}))
        elif landmarks is None or depth is None or suppress:
            self.push.reset()
        accepted += self._palm_events(palm_events)
        return accepted

    def _palm_events(self, events: list[GestureEvent]) -> list[str]:
        """Palm-glide scroll output: scrolling goes straight to the wheel (not while paused); the tap goes
        through the bindings (``gesture:palm_tap``, so a pause action still works while paused)."""
        out = []
        for ev in events:
            if not self.studio.event_allowed(ev.kind, ev.data):
                continue
            if ev.kind == "palm_tap":
                self.bus.publish("gesture.palm_tap", dict(ev.data), ev.t)
                self.dispatcher.on_gesture(ev)
            elif self.output_paused():
                self.palm.stop_momentum()
                continue
            else:
                self.bus.publish(f"gesture.{ev.kind}", dict(ev.data), ev.t)
                if ev.kind == "palm_scroll":
                    self.mouse.handle(ev)
            out.append(ev.kind)
        return out

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
        if not self.studio.event_allowed(ev.kind, ev.data):
            return []
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
        extras: list[Any] | None = None,
        net: tuple[float, float] | None = None,
    ) -> int | None:
        """``extras``: per face, an object with ``pose`` / ``proxy`` / ``gaze`` (a FaceObs) for head tracking and
        gaze (v0.1.19); ``net``: the learned gaze network's (yaw, pitch) for the user's face, if it ran."""
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
        if extras is not None and ids:
            k = ids.index(user) if user is not None else 0
            ex = extras[k] if k < len(extras) else None
            raw_pose = getattr(ex, "pose", None)
            self.headgaze.on_face(t, raw_pose, getattr(ex, "proxy", None), getattr(ex, "gaze", None), net)
            self.comfort.on_face(t, raw_pose is not None)
            if self.headgaze.facing(raw_pose) and user_frontal is not True:
                # an off-axis webcam sees the head turned even while the user looks at the screens: the head pose
                # (relative to the usual / calibrated screen-facing pose) says "facing the screen"
                user_frontal = True
        elif extras is not None:
            self.comfort.on_face(t, False)
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
        self.tray_icon: Any = None  # v0.1.24: the pystray icon (set by the tray; Check my PC reads it)
        self.bus = EventBus()
        with contextlib.suppress(Exception):  # v0.1.24: the 2026-10 voices read models_next (only when chosen)
            from away.app.next_wiring import configure_voices

            configure_voices(lambda: self.cfg)
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
        self.camera_in_use_by: str | None = None  # v0.1.21: "Settings (Gesture Studio)" while it borrows the camera
        backend: MouseBackend
        if dry_run:
            backend = RecordingBackend()
            self.monitor_layout = monitors.single(1920, 1080, source="dry-run")
        else:
            from away.actions_engine import get_engine

            monitors.enable_dpi_awareness()  # no-op when cli/tray main already did it (before pynput/tk)
            self.monitor_layout = monitors.enumerate_monitors()
            from away.control.macro_record import MarkingBackend

            # every synthetic event Away sends is marked, so a shortcut recording leaves it out (v0.1.20)
            backend = MarkingBackend(get_engine().os_backend())  # type: ignore[assignment]
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
        if not dry_run:
            # audit Phase 1b: the engine owns the real backend; gestures (and the voice commands that run the same
            # bindings, attributed "voice") reach it through the engine as the "gesture" source
            from away.actions_engine import GESTURE, get_engine

            get_engine().set_backend(backend)
            backend = get_engine().client(GESTURE)  # type: ignore[assignment]
        from away.pointer.uia import make_backend as pointer_backend

        self.pipeline = Pipeline(
            cfg,
            backend,
            screen,
            self.bus,
            temporal,
            active_app=active_process,
            guard=self.guard,
            pointer_backend=pointer_backend(),  # Windows UI Automation for the smart pointer (v0.1.20)
        )
        self.pipeline.smart.notify = self.notice
        # recorded shortcuts (v0.1.20): macro:<name> / macro_control:... run through the bindings like any action
        from away.app.macro_service import DryRunBackend, MacroService

        self.macros = MacroService(
            cfg.macros,
            notify=self._shortcut_notice,
            on_replay=self._on_shortcut_replay,
            backend_factory=(lambda _mark: DryRunBackend()) if dry_run else None,
        )
        self._paused_before_shortcut = True
        self.pipeline.executor.hooks["macro"] = self.macros.hook_macro
        self.pipeline.executor.hooks["macro_control"] = self.macros.hook_control
        self.camera_confidences: dict[int, float] = {}  # multi-camera: latest per-view confidence
        self.voice: Any = None  # VoiceController
        self.editor: Any = None  # SelectionEditor (voice.edit_hotkey)
        self.threads: list[threading.Thread] = []
        self._listeners: list[Any] = []
        self.asr_model = ""
        self._notices: set[str] = set()
        self.gate_hint = GateHint()
        # v0.1.22: the face gate knows who borrowed the camera and when the camera last delivered a frame
        self.pipeline.gate.camera_holder = lambda: self.camera_in_use_by
        self.pipeline.gate.when_camera_busy = cfg.face.gate_when_camera_busy
        from away.voice.talk import TalkHook

        # v0.1.22 "Talk to your agent": Agent Link fills in the agents and where messages go
        self.talk = TalkHook(enabled=lambda: bool(getattr(self.cfg.agent_link, "talk_enabled", True)))
        self.other_audio: Any = None  # away.voice.other_audio.OtherAudioMeter (Windows, while voice runs)
        self._face_lock = threading.Lock()
        self.gaze_net: Any = None  # NetRunner (optional learned gaze model), created with the first face
        self.iphone: Any = None  # LiveLinkReceiver when face.source is iphone|auto
        from away.face.source import FaceSourceSelector

        if cfg.face.enabled and cfg.face.source != "webcam":
            from away.face.livelink import LiveLinkReceiver

            ip = cfg.face.iphone
            self.iphone = LiveLinkReceiver(ip.host, ip.port, ip.stale_after)
        self.face_source = FaceSourceSelector(cfg.face.source if self.iphone is not None else "webcam", self.iphone)
        self.bus.subscribe("*", self._log)
        self.bus.subscribe("voice.utterance", self._on_utterance)
        self.bus.subscribe("camera.handover", self._on_handover)
        from away.app import instance
        from away.app.bigcursor import BigCursor, NullCursorBackend, default_backend

        # v0.1.23: only the one app holding the single-instance lock changes the system cursor size
        owns_cursor = not dry_run and instance.owned()
        self.big_cursor = BigCursor(
            default_backend() if owns_cursor else NullCursorBackend(),
            cfg.hands.cursor.big_cursor_size,
            publish=self._publish_quiet,
        )
        self.bus.subscribe("cursor.size_stuck", self._cursor_stuck)
        try:
            self.big_cursor.recover()  # a crash last time may have left the cursor big
        except Exception as exc:
            self.bus.publish("cursor.size_error", {"error": str(exc)})
        import atexit

        atexit.register(self._restore_cursor)
        self._camera_episode = 0  # bumps when the cameras recover, so a new failure is notified again
        # v0.1.19: camera retries back off and stop after hands.camera_retry_limit failures; "Retry cameras"
        # (tray, or Settings through a request file) sets this to probe again at once
        self._camera_retry_now = threading.Event()
        self.camera_retry_stopped = False
        self.camera_unavailable: dict[str, Any] | None = None  # v0.1.23: "Internal camera unavailable ..." for the tray
        self._frame_times: deque[float] = deque(maxlen=240)
        self.health_hint = ""  # v0.1.23: "Your PC is busy (...)" / "Hand tracking is choppy (...)" for the tray
        self._camera_set_changed_since: float | None = None
        with contextlib.suppress(Exception):
            from away.multicam import capture as _capture

            _capture.configure(cfg.hands)
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

    def _shortcut_notice(self, title: str, message: str) -> None:
        self.notice(title, message)

    def _on_shortcut_replay(self, started: bool) -> None:
        """A shortcut replays: gesture output pauses so the hand cannot fight the replay, then resumes."""
        if started:
            self._paused_before_shortcut = self.pipeline.paused
            self.pipeline.set_paused(True)
        elif not self._paused_before_shortcut:
            self.pipeline.set_paused(False)

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
        (the camera could not be opened directly but Broadcast's virtual camera shows it) is worth a
        notification, once."""
        msg = str(ev.data.get("error") or "")
        if "NVIDIA Broadcast's camera shows the same picture" in msg:
            self.notice("Project Away: camera could not be opened", msg, key=f"camera-broadcast:{self._camera_episode}")

    def _on_cameras_restored(self, ev: Any) -> None:
        msg = str(ev.data.get("message") or "all cameras are working again")
        print(f"info: {msg}", flush=True)
        if not getattr(self, "_camera_degraded_shown", False):
            return  # nothing was reported broken: nothing to announce
        self._camera_degraded_shown = False
        self.notice("Project Away: cameras back", msg, key=f"cameras-restored:{self._camera_episode}")
        self._camera_episode += 1

    def _on_utterance(self, ev: Any) -> None:
        data = ev.data or {}
        gi: dict[str, Any] = data["gate"] if isinstance(data.get("gate"), dict) else {}
        if data.get("decision") == "ignore" and gi.get("camera_busy") and not gi.get("allowed", False):
            self.notice(
                "Project Away: speech not typed",
                "The camera is being used by a camera check or Settings, so Project Away can't see whether you are "
                "the one talking. What you say is not typed until the camera comes back. (Settings > Voice can "
                "type anyway.)",
                key="camera-busy-dictation",
            )
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
        if ev.topic not in (
            "gesture.move",
            "gesture.scroll",
            "gesture.autoscroll",
            "gesture.palm_scroll",
            "wifi.state",
            "app.auto_paused",
        ):
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
            self._apply_camera_switch(new)
            self.macros.set_config(new.macros)
            if self.voice is not None:
                self._apply_voice(new)
            self._restart_hotkeys()
            self.bus.publish("config.reloaded", {"path": str(path)})

    def tracking_fps(self, now: float | None = None, window_s: float = 4.0) -> float | None:
        """v0.1.23: camera frames processed per second over the last ``window_s`` (None before any frame)."""
        t = time.monotonic() if now is None else now
        ts = [x for x in list(getattr(self, "_frame_times", [])) if t - x <= window_s]
        if not ts:
            return None
        return len(ts) / window_s

    def _health_loop(self, every_s: float = 5.0) -> None:
        """v0.1.23: the busy-PC / choppy-tracking tray hint (one sentence, changes published as app.health_hint)."""
        from away.agentlink.telemetry import CpuSampler
        from away.app.sysload import LoadHints, choppy_reason, top_processes

        hints, cpu, last = LoadHints(), CpuSampler(), ""
        n = 0
        while not self.stop_event.wait(every_s):
            try:
                _proc, system = cpu.sample()
                top = top_processes(5) if (system or 0) >= 80 and n % 2 == 0 else None
                n += 1
                active = self.pipeline.gestures_active()
                fps = self.tracking_fps() if active else None
                hint = hints.update(system, fps, top, choppy_reason(self))
            except Exception:
                continue
            if hint != last:
                last = hint
                self.health_hint = hint
                self.bus.publish("app.health_hint", {"hint": hint})
                if hint:
                    with contextlib.suppress(Exception):
                        self.notice("Project Away", hint, key=f"health:{hint.split('(')[0]}")

    def _start_glide(self) -> None:
        """v0.1.23: the fixed-rate cursor output (hands.cursor.output_hz). Only pointer moves run on it."""
        hz = int(getattr(self.cfg.hands.cursor, "output_hz", 0) or 0)
        if self.dry_run or hz <= 0 or not self.cfg.hands.enabled:
            return
        from away.pointer.glide import Glide, GlideRunner

        mouse = self.pipeline.mouse
        mouse.glide = Glide()
        self.glide_runner = GlideRunner(
            mouse.glide,
            mouse.backend.move,
            hz,
            active=lambda: bool(mouse.enabled) and not self.pipeline.output_paused(),
        ).start()

    def _cursor_stuck(self, ev: Any) -> None:
        """v0.1.23: a restore did not take: one friendly notice (the tray menu has "Reset cursor size")."""
        with contextlib.suppress(Exception):
            self.notice("Project Away: mouse pointer", str(ev.data.get("message")), key="cursor-size-stuck")

    def reset_cursor_size(self) -> int:
        """v0.1.23 "Reset cursor size to normal" (tray menu)."""
        return int(self.big_cursor.reset())

    def _apply_camera_switch(self, new: AppConfig) -> None:
        """v0.1.23: a camera turned off or on in Settings takes effect without restarting the app (the rest of
        the camera setup still applies at the next start, as before)."""
        h, nh = self.cfg.hands, new.hands
        same_off = [str(c.device) for c in h.disabled_cameras] == [str(c.device) for c in nh.disabled_cameras]
        same_on = [str(c.device) for c in h.cameras] == [str(c.device) for c in nh.cameras]
        if same_off and same_on and h.camera_index == nh.camera_index:
            return
        # v0.1.23: a camera removed (or turned off) in Settings > Cameras applies at once, without a restart
        self._camera_set_dirty = True
        h.disabled_cameras = nh.disabled_cameras
        h.cameras = nh.cameras
        h.camera_index, h.camera_hint, h.camera_fourcc = nh.camera_index, nh.camera_hint, nh.camera_fourcc
        self._retry_event().set()

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
            amb.strict_ratio, amb.strict_confidence = float(v.strict_motion_ratio), float(v.strict_confidence)
            amb.other_audio_needs_mouth = bool(v.other_audio_needs_mouth)
        self.pipeline.gate.when_camera_busy = new.face.gate_when_camera_busy
        for k, val in vars(v).items():
            setattr(self.cfg.voice, k, val)
        if v.mode != self.voice.mode:
            self.voice.set_mode(v.mode)

    # voice -------------------------------------------------------------------
    def _run_voice_command(self, phrase: str) -> bool:
        if not self.pipeline.comfort.voice_allowed():  # attention.gate_voice: not looking at the screen
            return False
        act = self.pipeline.fire("voice", phrase)
        return act.kind != "none"

    def _voice_hook(self, kind: str, arg: str, _data: dict[str, Any]) -> None:
        self.voice.handle_action(kind, arg)

    def _ambient_command(self, act: Action, phrase: str) -> None:
        if not self.pipeline.comfort.voice_allowed():
            return
        self.pipeline.fire("voice", phrase)

    def _start_voice(self) -> None:
        from away.app.performance import asr_plan
        from away.voice.always_on import AmbientSession, ASRQueue, StreamingEndpointer, VoiceController
        from away.voice.asr import make_asr
        from away.voice.cleanup import RuleCleaner, Vocabulary
        from away.voice.context import active_app
        from away.voice.dictation import DictationSession, MicStream
        from away.voice.output import ListSink, make_sink
        from away.voice.speech import ECHO
        from away.voice.vad import make_vad

        v = self.cfg.voice
        vocab = Vocabulary.load(v.vocabulary_file)
        rules = RuleCleaner(vocab, numbers_as_digits=v.numbers_as_digits, numbers_min_value=v.numbers_min_value)
        prof = self.profile
        # v0.1.20: smart cleanup (per-app tone, optional local LLM) + personal dictionary (away.voice.voice_setup)
        from away.voice.voice_setup import setup_voice_context

        vctx = setup_voice_context(self, rules, vocab)
        cleaner = vctx.cleaner
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
        from away.app import next_wiring  # v0.1.24: optional 2026-10 models (nothing changes until one is chosen)

        def build_whisper() -> Any:
            return make_asr(
                v.asr_backend,
                model,
                prof.device,
                prof.compute_type,
                v.language,
                str(self.models_dir / f"whisper-{model}"),
                vctx.hotwords() if v.hotwords else None,
                cpu_threads=prof.cpu_threads,
                beam_size=prof.beam_size,
                on_fallback=gpu_fallback,
                cpu_model=cpu_model,
                cpu_download_root=str(self.models_dir / f"whisper-{cpu_model}") if cpu_model else None,
            )

        asr = next_wiring.next_asr(
            self.cfg,
            build_whisper,
            lambda title, msg: self.notice(title, msg, key="next-model"),
            v.sample_rate,
            self._save_cfg,
        )
        if hasattr(asr, "push"):  # a streaming backend (Moonshine / Nemotron) was chosen
            self.asr_model = str(getattr(asr, "name", self.asr_model))
            self.asr_device = str(getattr(asr, "device", self.asr_device))
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
        if self.dry_run:
            sink: Any = ListSink()
        else:
            from away.control.macro_record import MarkingBackend

            sink = MarkingBackend(make_sink(v.output))  # dictated text is Away's own input: never recorded (v0.1.20)
        commands = self._run_voice_command if v.commands_enabled else None
        dictation = DictationSession(
            asr,
            cleaner,
            sink,
            make_vad(next_wiring.vad_kind(self.cfg, v.vad), v.sample_rate),
            self.bus,
            v.sample_rate,
            gate,
            active_app,
            vctx.terms,
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
            commands=self.pipeline.voice_phrases,
            run_command=self._ambient_command,
            bus=self.bus,
            gate=self.pipeline.gate.allow_ambient if face_gate else None,
            context=active_app,
            vocabulary=vctx.terms,
            wake_word=v.wake_word if v.wake_word_enabled else None,
            wake_window=v.wake_window,
            intent=v.intent,
            fallback=fallback,
            commands_enabled=v.commands_enabled,
            submit=submit,
            gate_info=lambda: self.pipeline.gate.last_explain,
            echo=ECHO,
            talk=self.talk,
        )
        ambient.strict_ratio, ambient.strict_confidence = float(v.strict_motion_ratio), float(v.strict_confidence)
        ambient.other_audio_needs_mouth = bool(v.other_audio_needs_mouth)
        try:  # v0.1.22: is another app (a video) playing sound? Windows only, needs pycaw
            from away.voice.other_audio import make_meter

            meter = make_meter()
            if meter.reader is not None and not self.dry_run:
                self.other_audio = meter.start()
                ambient.other_audio = meter.playing_during
        except Exception as exc:
            self._publish_quiet("voice.other_audio_unavailable", {"error": str(exc)})
        # v0.1.24: words while you speak (streaming backends), and the wake-word model in front of always-on
        self._partials = next_wiring.partial_tap(asr, ambient, self._publish_quiet)
        gated = next_wiring.wrap_ambient(self.cfg, ambient, lambda title, msg: self.notice(title, msg, key="wake"))
        self.voice = VoiceController(v.mode, dictation, gated, self.bus)
        self.voice.echo = ECHO  # v0.1.22: the microphone ignores the computer's own voice
        vctx.attach(asr, dictation, ambient)
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
            tap = getattr(self, "_partials", None)
            if tap is not None and not self.voice.muted:
                with contextlib.suppress(Exception):
                    tap(chunk)

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
        accel = self._tracking_accel(hand_model)  # v0.1.24: hands.compute (graphics card or processor)
        nxt = self._next_hands() if self.cfg.hands.enabled else {}
        hands = (
            HandTracker(
                hand_model,
                self.cfg.hands.max_hands,
                prefer=lambda: self.pipeline.hand_mode,
                mirrored=self.cfg.hands.mirror,
                release_s=lambda: self.cfg.hands.roles.release_s,
                # v0.1.25: RTMPose (experimental, models_next) wins only when chosen and downloaded; else the
                # v0.1.24 hands.compute choice (graphics card or processor)
                **(nxt or {"factory": accel[0] if accel else None, "image_fn": accel[1] if accel else None}),
            )
            if self.cfg.hands.enabled
            else None
        )
        faces = None
        if self.cfg.face.enabled and self.face_source.needs_webcam_model and face_model.exists():
            from away.accel.compute import face_landmarker

            faces = face_landmarker(self.cfg, face_model, paths.data_dir())
        prof = self.profile

        def process(n: int, t: float, rgb: Any) -> None:
            self.pipeline.gate.note_camera_frame(t)  # v0.1.22: the camera is delivering (for the face gate)
            self._frame_times.append(time.monotonic())  # v0.1.23: hand-tracking rate for the "choppy" hint
            if hands is not None and n % max(1, prof.hands_every) == 0:
                fr = hands.process(rgb, t)
                self.pipeline.note_hands(t, fr.n_hands, fr.hands_gap)
                if fr.n_hands:
                    self._last_hand_t = time.monotonic()  # v0.1.23: probes wait while a hand is tracked
                self.pipeline.on_hand(
                    t,
                    fr.landmarks,
                    fr.handedness,
                    helper=getattr(fr, "helper", None),
                    pointer_id=getattr(fr, "pointer_id", None),
                    helper_id=getattr(fr, "helper_id", None),
                    promoted=bool(getattr(fr, "promoted", False)),
                )
                self._agent_frame(self.cfg.hands.camera_index, t, rgb, fr)
            if faces is not None and n % max(1, prof.face_every) == 0 and self.face_source.use_webcam(t):
                obs = faces.process(rgb, t)
                net = self._gaze_net(t, rgb, obs)
                with self._face_lock:
                    self.pipeline.on_faces(
                        t,
                        [o.box for o in obs],
                        [o.blendshapes for o in obs],
                        [getattr(o, "frontal", None) for o in obs],
                        extras=list(obs),
                        net=net,
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
                wait_free=self._wait_devices_free,
            ).run()

        self._camera_list(log=True)  # the enumerated cameras (source, index, name, virtual) -> away.log
        with contextlib.suppress(Exception):  # v0.1.23: no NVIDIA Broadcast fallback while the built-in camera is set
            from away.multicam.capture import configure_cameras

            h = self.cfg.hands
            configure_cameras([c.device for c in h.cameras] if h.cameras else [h.camera_index])
        try:
            while not self.stop_event.is_set():
                if len(self.cfg.hands.cameras) >= 2 and hands is not None:
                    self._multicam_loop(hands, faces, run_single)
                    if (
                        getattr(self, "_camera_set_changed_since", None) is not None
                    ):  # v0.1.23: a camera was turned off / on
                        self._camera_set_changed_since = None
                        continue
                    return
                run_single(self.cfg.hands.camera_index)
                return
        finally:
            # close the landmarkers only after this thread stopped calling them (shutdown race)
            for tr in (hands, faces):
                close = getattr(tr, "close", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        close()

    def _agent_frame(self, cam: Any, t: float, rgb: Any, fr: Any) -> None:
        """Agent Link (v0.1.20): fps / latency / landmarks per camera, and the still for an allowed look()."""
        al = getattr(self, "agent_link", None)
        if al is not None:
            with contextlib.suppress(Exception):
                al.on_frame(cam, t, rgb, fr)

    def _gaze_net(self, t: float, rgb: Any, obs: list[Any]) -> tuple[float, float] | None:
        """The optional learned gaze network on the largest face (only when the calibration uses it)."""
        if not obs:
            return None
        if self.gaze_net is None:
            from away.face.headgaze import NetRunner

            self.gaze_net = NetRunner(self.cfg, self.models_dir)
        box = max(obs, key=lambda o: o.box[2]).box
        try:
            return self.gaze_net.update(t, rgb, box, self.pipeline.headgaze.cal)  # type: ignore[no-any-return]
        except Exception as exc:  # never let the optional network stop face tracking
            self._publish_quiet("gaze.net_error", {"error": f"{type(exc).__name__}: {exc}"})
            return None

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
    def _start_cameras(
        self, specs: list[Any], pairer: Any, session: threading.Event, open_lock: Any, skip_held: bool = False
    ) -> list[Any]:
        """Start one capture thread per camera (overridden in tests with simulated cameras): the shared
        :func:`~away.multicam.capture.start_camera_threads` (also used by Settings and the CLI). ``skip_held``
        (background re-probes): never reopen a camera this process already has open."""
        from away.multicam.capture import start_camera_threads

        return start_camera_threads(
            list(specs),
            pairer,
            session,
            open_lock=open_lock,
            serial=open_lock is not None,
            # v0.1.23: a background re-probe never opens NVIDIA Broadcast's virtual camera: Broadcast reads the
            # working camera, so the probe dragged it from 30 down to 4-10 fps and looked like a success at 1 fps
            virtual_fallback=bool(self.cfg.hands.virtual_fallback) and not skip_held,
            min_fps=float(self.cfg.hands.min_camera_fps),
            skip_held=skip_held,
        )

    def _tracking_accel(self, model: Any) -> Any:
        """v0.1.24 ``hands.compute``: (factory, image_fn) for the hand trackers when the graphics card runs them,
        None for MediaPipe on the processor. Never raises (falls back to the processor and says why)."""
        try:
            from away.accel.compute import hand_factory

            return hand_factory(
                self.cfg,
                model,
                paths.data_dir(),
                on_note=lambda why: self.notice(
                    "Project Away: hand tracking on the processor", why, key="gpu-tracking"
                ),
            )
        except Exception:
            return None

    def _use_mode_store(self) -> None:
        """Remember per camera name a fallback mode it only works in (v0.1.15), so the next start opens it in
        that mode right away."""
        from away.multicam import capture

        if capture.mode_store() is None:
            with contextlib.suppress(Exception):
                capture.use_mode_store(capture.default_mode_file())

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
        import inspect

        # v0.1.19: a re-probe never reopens (or re-sweeps) the camera that is tracking right now
        extra = {"skip_held": True} if "skip_held" in inspect.signature(self._start_cameras).parameters else {}
        threads = self._start_cameras(list(specs), pairer, session, lock, **extra)
        try:
            # a lease from Settings / calibration ends the probe at once (the device is let go right away)
            stop = _ProbeStop(self.stop_event, self.camera_share, None)
            return probe_cameras(threads, pairer.seen, self._probe_timeout(len(specs), probe_s), stop)
        finally:
            self._end_cameras(session, threads)

    # camera retries (v0.1.19) ---------------------------------------------------
    def retry_cameras(self) -> None:
        """ "Retry cameras" (tray menu): probe the failed cameras again now and restart the back-off."""
        self.camera_retry_stopped = False
        self._retry_event().set()
        self.bus.publish("cameras.retry_requested", {})

    def _retry_event(self) -> threading.Event:
        ev = getattr(self, "_camera_retry_now", None)
        if ev is None:  # an app built without __init__ (tests)
            ev = self._camera_retry_now = threading.Event()
        return ev

    def _manual_retry(self) -> bool:
        """True (once) when "Retry cameras" was pressed here (tray) or in Settings (a request file)."""
        from away.app.camera_loop import take_retry_request

        ev = self._retry_event()
        if ev.is_set():
            ev.clear()
            return True
        folder = self._retry_folder()
        if folder is None:
            return False
        try:
            return take_retry_request(folder)
        except Exception:
            return False

    def _retry_folder(self) -> Any:
        """Where the retry request / status files live (the camera-share folder; tests use a temp one)."""
        return getattr(self.camera_share, "folder", None)

    def _retry_backoff(self, base_s: float | None = None) -> Any:
        from away.app.camera_loop import RetryBackoff

        h = self.cfg.hands
        base = float(h.camera_retry_s if base_s is None else base_s)
        # v0.1.23: 5 min, 15 min, 60 min, then stop until Retry (or the camera list changes)
        return RetryBackoff(
            base, max(base, float(h.camera_retry_max_s)), int(h.camera_retry_limit), factor=3.0, last_max=True
        )

    def _none_backoff(self, base_s: float) -> Any:
        """v0.1.23: the back-off while no camera works at all (see NONE_RETRY_MAX_S)."""
        from away.app.camera_loop import RetryBackoff

        base = float(base_s)
        return RetryBackoff(base, max(base, NONE_RETRY_MAX_S), NONE_RETRY_LIMIT)

    def _camera_backoff(self, specs: list[Any], base_s: float | None = None) -> Any:
        """v0.1.23: one back-off per set of failing cameras that lasts across camera-session restarts (it used to
        start over every time, so a camera that never works was re-tried every few minutes)."""
        store: dict[tuple[str, ...], Any] | None = getattr(self, "_camera_backoffs", None)
        if store is None:
            store = {}
            self._camera_backoffs = store
        key = tuple(sorted(str(getattr(s, "device", s)) for s in specs))
        if key not in store:
            store[key] = self._retry_backoff(base_s)
        return store[key]

    def _probe_would_disturb(self, now: float | None = None) -> bool:
        """v0.1.23: opening another camera takes USB bandwidth from the one that tracks; while your hand is being
        tracked (seen in the last HAND_QUIET_S seconds) a background probe waits."""
        last = getattr(self, "_last_hand_t", None)
        if last is None:
            return False
        t = time.monotonic() if now is None else now
        return t - float(last) < HAND_QUIET_S

    def _devices_changed(self, now: float | None = None) -> bool:
        """v0.1.23: True (once) when the OS camera list changed (plugged in / unplugged / a driver came back):
        a stopped retry starts again by itself. Checked at most every DEVICE_CHECK_S seconds."""
        t = time.monotonic() if now is None else now
        last = getattr(self, "_dev_check_t", None)
        if last is None:
            self._dev_check_t = t
            return False
        if t - float(last) < DEVICE_CHECK_S:
            return False
        self._dev_check_t = t
        listed = self._camera_list()
        if listed is None:
            return False
        sig = tuple(sorted(str(getattr(d, "name", d)) for d in listed))
        old = getattr(self, "_dev_sig", None)
        self._dev_sig = sig
        return old is not None and old != sig

    def _turned_off(self, specs: list[Any]) -> bool:
        """v0.1.23: every failing camera was turned off (Turn off / an agent): stop probing them."""
        off = {str(c.device) for c in getattr(self.cfg.hands, "disabled_cameras", [])}
        return bool(off) and all(str(getattr(s, "device", s)) in off for s in specs)

    def _save_cfg(self) -> None:
        """Save the settings file now (v0.1.24: after an automatic model rollback)."""
        from away import config as cfgmod

        with contextlib.suppress(Exception):
            cfgmod.save(self.cfg, getattr(self, "config_path", None) or cfgmod.config_path())

    def _next_hands(self) -> dict[str, Any]:
        from away.app import next_wiring

        return next_wiring.hand_kwargs(
            self.cfg, lambda title, msg: self.notice(title, msg, key="next-hands"), self._save_cfg
        )

    def turn_off_camera(self, device: Any) -> str:
        """ "Turn off" next to a failing camera (tray / Settings > Cameras): saved to the config file."""
        from away import config as cfgmod
        from away.app.camera_switch import turn_off

        msg = turn_off(self.cfg, str(device))
        with contextlib.suppress(Exception):
            cfgmod.save(self.cfg, getattr(self, "config_path", None) or cfgmod.config_path())
        self._retry_status(None)
        self.camera_retry_stopped = False
        self._retry_event().set()  # wake the prober so it notices
        self.bus.publish("cameras.turned_off", {"device": str(device), "message": msg})
        return msg

    def _headline(self, labels: dict[str, str]) -> str:
        from away.app.camera_loop import unavailable_text

        listed = None
        with contextlib.suppress(Exception):
            listed = self._camera_list() if hasattr(self, "_camera_list") else None
        broadcast = any("nvidia broadcast" in str(getattr(d, "name", d)).lower() for d in (listed or []))
        name, why = next(iter(labels.items()), ("", ""))
        with contextlib.suppress(Exception):  # v0.1.23: name the app that holds the camera (Windows records it)
            from away.multicam import camera_users as cu

            users = cu.current_users()
            info = cu.explain(name, users, cu.broadcast_source())
            self.camera_users = info
            if users or info.get("broadcast_uses_it"):
                return str(info["message"]).rstrip(".")
        return unavailable_text(name, why, broadcast)

    def _retry_status(self, status: dict[str, Any] | None) -> None:
        from away.app.camera_loop import write_retry_status

        folder = self._retry_folder()
        if folder is not None:
            with contextlib.suppress(Exception):
                write_retry_status(status, folder)

    def _retries_stopped(self, labels: dict[str, str], attempts: int) -> None:
        """The back-off ran out: tell the user once (tray notification + Settings) and wait for Retry cameras."""
        self.camera_retry_stopped = True
        names = ", ".join(labels) or "a camera"
        headline = self._headline(labels)
        msg = (
            f"{headline}; stopped retrying {names} after {attempts} failed attempts. Retry / Turn off: tray menu "
            "or Settings > Cameras (Retry cameras)."
        )
        logging.getLogger("away.cameras").warning("%s", msg)
        self.camera_unavailable = {"headline": headline, "devices": list(labels)}
        self._retry_status({"stopped": True, "attempts": attempts, "failed": labels, "headline": headline})
        self.bus.publish(
            "cameras.retry_stopped",
            {"failed": labels, "attempts": attempts, "message": msg, "headline": headline},
        )
        episode = getattr(self, "_camera_episode", 0)
        if hasattr(self, "_notices"):
            self.notice("Project Away: camera retries stopped", msg, key=f"cameras-retry-stopped:{episode}")
        else:  # an app built without __init__ (tests)
            self.bus.publish("app.notice", {"title": "Project Away: camera retries stopped", "message": msg})

    def _wait_none_retry(self, backoff: Any, labels: dict[str, str]) -> None:
        """No camera works: wait before the next try of the whole set (backing off; after the limit until
        "Retry cameras")."""
        from away.app.camera_loop import wait_for_retry

        backoff.failed()
        if backoff.exhausted and not getattr(self, "camera_retry_stopped", False):
            self._retries_stopped(labels, backoff.failures)
        delay = float("inf") if backoff.exhausted else backoff.delay()
        if not backoff.exhausted:
            self._retry_status({"stopped": False, "attempts": backoff.failures, "failed": labels, "next_s": delay})
        never = threading.Event()
        if wait_for_retry(delay, never, self.stop_event, self._manual_retry) == "manual":
            backoff.reset()
            self.camera_retry_stopped = False

    def _single_until_retry(
        self, run_single: Any, device: Any, failed_specs: list[Any], every_s: float, probe_s: float
    ) -> None:
        """Track on ``device`` alone and probe the failed cameras in the background: first after ``every_s``,
        then after twice as long each time (capped at hands.camera_retry_max_s); return (to retry the full
        multi-camera set) once they all deliver frames. After hands.camera_retry_limit failed probes it stops
        probing and says so until "Retry cameras" is pressed (v0.1.19). Tracking is never interrupted by a probe
        that still fails, and the working camera is never reopened by a probe (``_probe_specs``)."""
        from away.app.camera_loop import retry_labels, wait_for_retry

        ready = threading.Event()
        done = threading.Event()
        backoff = self._camera_backoff(failed_specs, every_s)
        changed = {"devices": False}

        def asked() -> bool:
            if self._turned_off(failed_specs):
                return True
            if self._manual_retry():
                return True
            if self._devices_changed():
                changed["devices"] = True
                return True
            return False

        def prober() -> None:
            labels = {str(getattr(s, "device", s)): "" for s in failed_specs}
            while True:
                delay = float("inf") if backoff.exhausted else backoff.delay()
                if not backoff.exhausted:
                    self._retry_status(
                        {"stopped": False, "attempts": backoff.failures, "failed": labels, "next_s": delay}
                    )
                why = wait_for_retry(delay, done, self.stop_event, asked)
                if why == "done":
                    return
                if self._turned_off(failed_specs) or getattr(self, "_camera_set_dirty", False):
                    # v0.1.23: turned off / removed: back to the outer loop, which starts the new camera set
                    self._retry_status(None)
                    self.camera_unavailable = None
                    ready.set()
                    return
                if why == "manual":
                    backoff.reset()
                    self.camera_retry_stopped = False
                    self.camera_unavailable = None
                    reason = "the camera list changed" if changed["devices"] else "Retry cameras"
                    changed["devices"] = False
                    logging.getLogger("away.cameras").info("cameras: retrying now (%s)", reason)
                if self.camera_share.requested():  # Settings is using the cameras: do not grab them
                    continue
                waited = 0.0
                while self._probe_would_disturb() and not done.is_set() and waited < MAX_DEFER_S:
                    done.wait(5.0)  # v0.1.23: your hand is being tracked: probe when it is quiet
                    waited += 5.0
                if done.is_set():
                    return
                try:
                    bad = self._probe_specs(failed_specs, probe_s)
                except Exception as exc:
                    bad = {0: f"probe error: {exc}"}
                labels = retry_labels(failed_specs, bad) or labels
                if bad:
                    backoff.failed()
                self.bus.publish(
                    "cameras.retry",
                    {
                        "ok": not bad,
                        "failed": retry_labels(failed_specs, bad),
                        "attempt": backoff.failures,
                        "next_s": None if (not bad or backoff.exhausted) else backoff.delay(),
                    },
                )
                if not bad and not done.is_set():
                    self._retry_status(None)
                    ready.set()
                    return
                if backoff.exhausted:
                    self._retries_stopped(labels, backoff.failures)

        th = threading.Thread(target=prober, name="camera-retry", daemon=True)
        th.start()
        try:
            run_single(device, ready.is_set)
        finally:
            done.set()

    def _log_camera_speed(self, threads: list[Any]) -> list[tuple[bool, str]]:
        """away.log: each camera ok or FAILED (too slow) with its measured rate (v0.1.16); a too-slow camera is
        also shown to the user once."""
        from away.app.camera_loop import camera_speed_lines

        lines = camera_speed_lines(threads, float(self.cfg.hands.min_camera_fps))
        lg = logging.getLogger("away.cameras")
        for failed, text in lines:
            (lg.warning if failed else lg.info)("%s", text)
        slow = [text for failed, text in lines if failed and "too slow" in text]
        if slow:
            self.bus.publish("cameras.slow", {"message": "; ".join(slow)})
            self.bus.publish(
                "app.notice",
                {
                    "title": "Project Away: camera too slow",
                    "message": slow[0].split(";")[0] + ". Settings > Cameras > Test both cameras tells you more.",
                },
            )
        return lines

    @staticmethod
    def _wait_devices_free(timeout: float = HANDOVER_FREE_S) -> bool:
        """v0.1.16: hand the cameras over only once no camera of this process is open or still opening (a
        capture thread that was in the middle of opening, or a background re-probe, is told to give up); the
        borrower would otherwise open a device this process still holds. Overridden in tests."""
        from away.multicam import capture

        free = capture.wait_devices_free(timeout)
        if not free:
            logging.getLogger("away.cameras").warning(
                "handing the cameras over although this process may still hold one: %s",
                ", ".join(sorted(capture.held_cameras())) or "a camera is still opening",
            )
        return free

    def _on_handover(self, ev: Any) -> None:
        """Who borrowed the camera (both camera loops publish ``camera.handover``): the tray says it in plain words."""
        data = getattr(ev, "data", {}) or {}
        self.camera_in_use_by = str(data.get("by_label") or "Settings") if data.get("released") else None

    def camera_status(self) -> str:
        """ "Settings (Gesture Studio) is using the camera" while it borrows it, else ""."""
        return f"{self.camera_in_use_by} is using the camera" if self.camera_in_use_by else ""

    def _wait_handover(self, devices: list[Any]) -> None:
        """The cameras are closed: make sure none is still held or opening, tell the borrower (Settings,
        calibration) and wait until it is done."""
        from away.app.camera_share import borrower_label, in_use_status

        share = self.camera_share
        self._wait_devices_free()
        share.clear_owner()
        share.ack_released(devices)
        who = _requester(share)
        self.camera_in_use_by = borrower_label(who)  # v0.1.21: tray + Agent Link say who has the camera
        self.bus.publish(
            "camera.handover",
            {
                "released": True,
                "cameras": devices,
                "by": who,
                "by_label": self.camera_in_use_by,
                "status": in_use_status(who),
            },
        )
        while not self.stop_event.is_set() and _still_requested(share, self.stop_event.wait):
            self.stop_event.wait(0.5)
            share.ack_released(devices)  # keep the acknowledgement fresh (it goes stale after 5 s)
            now_who = _requester(share, "")
            if now_who and now_who != who:  # another Settings tool took over the lease
                who = now_who
                self.camera_in_use_by = borrower_label(who)
                self.bus.publish(
                    "camera.handover",
                    {
                        "released": True,
                        "cameras": devices,
                        "by": who,
                        "by_label": self.camera_in_use_by,
                        "status": in_use_status(who),
                    },
                )
        share.clear_released()
        self.camera_in_use_by = None
        self.bus.publish("camera.handover", {"released": False, "cameras": devices, "by": who})

    def _multicam_loop(
        self,
        first_tracker: Any,
        faces: Any,
        run_single: Any = None,
        probe_s: float = MULTICAM_PROBE_S,
        retry_s: float | None = None,
        none_retry_s: float = MULTICAM_NONE_RETRY_S,
        trackers: list[Any] | None = None,
    ) -> None:
        """Two or more cameras: threaded capture -> timestamp pairing -> per-view hands -> fusion -> pipeline.

        * Start-up probe: a camera fails only when its thread dies with an error, or it sends no frame within
          ``probe_s`` (Windows webcams are slow to open; on Windows they open one after another).
        * One camera left: track on it alone (``run_single``) and re-probe the failed cameras without touching
          the working one, after ``retry_s`` (default hands.camera_retry_s) and then twice as long each time up
          to hands.camera_retry_max_s, stopping after hands.camera_retry_limit failures until "Retry cameras"
          (v0.1.19); when they deliver again, go back to fusion (the first wait also doubles if the set keeps
          failing right after a good probe).
        * No camera works: nothing is claimed; the whole set is retried after ``none_retry_s``, backing off and
          stopping the same way.
        * The captures restart after a hand-over to Settings (camera_share) without restarting the app.
        """
        from away.app.camera_loop import degrade_plan, is_shutdown_error, probe_cameras
        from away.multicam.capture import FramePairer, pin_camera_set
        from away.multicam.fusion import MultiViewFuser, ViewObs, hand_crop, sharpness_score
        from away.multicam.geometry import load_extrinsics

        cams, mc = self.cfg.hands.cameras, self.cfg.hands.multicam
        primary = next(i for i, c in enumerate(cams) if c.role == "primary")
        if trackers is None:  # the real app (tests pass trackers and keep the store off)
            self._use_mode_store()
        own_trackers = trackers is None
        if trackers is None:
            from away.hands.tracker import HandTracker

            model = self.models_dir / "hand_landmarker.task"
            first_tracker.roles.mirrored = cams[0].mirror  # each camera's own frame orientation
            accel = self._tracking_accel(model)  # v0.1.24
            trackers = [first_tracker] + [
                HandTracker(
                    model,
                    self.cfg.hands.max_hands,
                    prefer=lambda: self.pipeline.hand_mode,
                    mirrored=c.mirror,
                    release_s=lambda: self.cfg.hands.roles.release_s,
                    factory=accel[0] if accel else None,
                    image_fn=accel[1] if accel else None,
                )
                for c in cams[1:]
            ]
        ex = load_extrinsics()
        fuser = MultiViewFuser(cams, mc, ex)
        helper_fuser = MultiViewFuser(cams, mc, ex)  # the helper hand (palm-glide scroll), fused the same way
        share = self.camera_share
        timeout = self._probe_timeout(len(cams), probe_s)
        degraded = False
        if retry_s is None:
            retry_s = float(self.cfg.hands.camera_retry_s)
        retry_max = max(float(retry_s), float(self.cfg.hands.camera_retry_max_s))
        backoff = retry_s
        after_retry = False
        none_backoff = self._none_backoff(none_retry_s)
        try:
            while not self.stop_event.is_set():
                if [str(c.device) for c in self.cfg.hands.cameras] != [str(c.device) for c in cams]:
                    self._camera_set_dirty = False
                    # v0.1.23: a camera was turned off (tray / Settings / an agent): start over with the new set
                    self._camera_set_changed_since = time.monotonic()
                    break
                # every attempt: bare indices from older versions are pinned to camera names (a virtual camera at
                # that index is swapped for a real one: the auto-skip), so each open resolves by name (hot-plug safe)
                listed = self._camera_list()
                run_cams = pin_camera_set(cams, listed) if listed else list(cams)
                devices = [c.device for c in run_cams]
                session = threading.Event()
                pairer = FramePairer(len(cams), primary, mc.pair_tolerance_ms / 1000.0, mc.dropout_ms / 1000.0)
                if share.requested():  # Settings / calibration is using the cameras: wait until it is done
                    self._wait_handover(devices)
                    continue
                lock = threading.Lock() if self._serial_open() else None
                share.publish_owner(devices)  # visible while probing too, so a lease asks us to let go
                threads = self._start_cameras(run_cams, pairer, session, lock)
                failed = probe_cameras(threads, pairer.seen, timeout, _ProbeStop(self.stop_event, share, devices))
                if self.stop_event.is_set():
                    self._end_cameras(session, threads)
                    break
                if share.requested():  # asked for the cameras while probing: hand them over now
                    self._end_cameras(session, threads)
                    self._wait_handover(devices)
                    continue
                working = len(cams) - len(failed)
                if working < 2 and after_retry:  # the set failed again right after a good probe: back off
                    backoff = min(backoff * 2.0, retry_max)
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
                        self._wait_none_retry(none_backoff, {str(d): "" for d in devices})
                        continue
                    self._single_until_retry(
                        run_single, device, [run_cams[i] for i in sorted(failed)], backoff, probe_s
                    )
                    after_retry = True
                    continue
                backoff = retry_s
                none_backoff.reset()
                with contextlib.suppress(Exception):
                    self._camera_backoffs.clear()  # v0.1.23: every camera works again: a new failure starts over
                self._retry_status(None)
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
                speed_at = time.monotonic() + SPEED_CHECK_S  # log each camera's measured rate once per session
                try:
                    while not self.stop_event.is_set():
                        now = time.monotonic()
                        if now >= next_check:
                            next_check = now + 1.0
                            if speed_at and now >= speed_at:
                                speed_at = 0.0
                                self._log_camera_speed(threads)
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
                        helpers: list[Any] = []
                        prim_ids: dict[str, Any] = {}
                        try:
                            for i, item in enumerate(frames):
                                if item is None:
                                    views.append(None)
                                    helpers.append(None)
                                    continue
                                t_i, rgb = item
                                fr = trackers[i].process(rgb, t_i)
                                self._agent_frame(i, t_i, rgb, fr)
                                note = getattr(self.pipeline, "note_hands", None)
                                if i == primary and note is not None:
                                    note(t_i, getattr(fr, "n_hands", 1), getattr(fr, "hands_gap", None))
                                h, w = rgb.shape[:2]
                                hlm = getattr(fr, "helper", None)
                                helpers.append(
                                    None
                                    if hlm is None
                                    else ViewObs(
                                        i,
                                        t_i,
                                        hlm,
                                        getattr(fr, "helper_handedness", "Right"),
                                        getattr(fr, "helper_score", 1.0),
                                        (w, h),
                                    )
                                )
                                if i == primary:
                                    prim_ids = {
                                        "pointer_id": getattr(fr, "pointer_id", None),
                                        "helper_id": getattr(fr, "helper_id", None),
                                        "promoted": bool(getattr(fr, "promoted", False)),
                                    }
                                if fr.landmarks is None:
                                    views.append(None)
                                    continue
                                crop = hand_crop(rgb, fr.landmarks)
                                sharp = sharpness_score(crop.mean(axis=2)) if crop.size else None
                                views.append(ViewObs(i, t_i, fr.landmarks, fr.handedness, fr.score, (w, h), sharp))
                            fused = fuser.update(t_ref, views)
                            self.camera_confidences = dict(fuser.last_confidences)
                            fused_helper = helper_fuser.update(t_ref, helpers)
                            extra = {k: v for k, v in prim_ids.items() if v not in (None, False)}
                            if fused_helper is not None:
                                extra["helper"] = fused_helper
                            self.pipeline.on_fused(fused, t_ref, **extra)
                            prim_frame = frames[primary]
                            if (
                                faces is not None
                                and prim_frame is not None
                                and self.face_source.use_webcam(prim_frame[0])
                            ):
                                t_p, rgb_p = prim_frame
                                obs = faces.process(rgb_p, t_p)
                                net = self._gaze_net(t_p, rgb_p, obs)
                                with self._face_lock:
                                    self.pipeline.on_faces(
                                        t_p,
                                        [o.box for o in obs],
                                        [o.blendshapes for o in obs],
                                        [getattr(o, "frontal", None) for o in obs],
                                        extras=list(obs),
                                        net=net,
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
                    self._wait_handover(devices)
        finally:
            share.clear_owner()
            if own_trackers:
                for tr in trackers[1:]:
                    with contextlib.suppress(Exception):
                        tr.close()

    def _iphone_loop(self) -> None:
        """Live Link Face (iPhone) -> face pipeline, while it is the active face source."""
        from away.face.source import IPhoneFaceFeed

        rx = self.iphone
        if not rx.start():
            self.bus.publish("iphone.error", {"error": rx.error})
            return
        ip = self.cfg.face.iphone
        self.bus.publish("iphone.listening", {"host": ip.host, "port": ip.port, "source": self.cfg.face.source})
        feed = IPhoneFaceFeed(
            rx,
            self.pipeline,
            self.face_source,
            ip.max_rate,
            ip.swap_sides,
            lock=self._face_lock,
            on_switch=lambda src: self.bus.publish("face.source", {"source": src}),
        )
        seq = 0
        while not self.stop_event.is_set():
            seq = rx.wait(seq, 0.1)
            try:
                feed.step(time.monotonic())
            except Exception as exc:  # a face-pipeline error must not kill the listener
                self.bus.publish("iphone.error", {"error": str(exc)})
        rx.stop()

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
        webcam_face = self.cfg.face.enabled and self.face_source.needs_webcam_model
        if self.cfg.hands.enabled or webcam_face:
            self.threads.append(threading.Thread(target=self._camera_loop, daemon=True, name="camera"))
        if self.iphone is not None:
            self.threads.append(threading.Thread(target=self._iphone_loop, daemon=True, name="iphone"))
        if self.cfg.voice.enabled:
            self._start_voice()
        if self.cfg.wifi.enabled:
            self.threads.append(threading.Thread(target=self._wifi_loop, daemon=True, name="wifi"))
        self.threads.append(threading.Thread(target=self._watch_config, daemon=True, name="config-watch"))
        if self.cfg.hands.enabled:
            self.threads.append(threading.Thread(target=self._cursor_size_loop, daemon=True, name="cursor-size"))
        if self.monitor_watcher.enabled:
            self.threads.append(threading.Thread(target=self._watch_monitors, daemon=True, name="monitors"))
        if not self.dry_run:
            self.threads.append(threading.Thread(target=self._health_loop, daemon=True, name="health-hints"))
        for th in self.threads:
            th.start()
        self._start_glide()
        with contextlib.suppress(Exception):
            self.pipeline.smart.start()  # low-rate UI Automation element cache (Windows; no-op elsewhere)
        if self.guard is not None:
            try:
                from away.app.autopause import start_listeners

                self._input_listeners = start_listeners(self.guard)
            except Exception as exc:
                self.bus.publish("autopause.unavailable", {"error": str(exc)})
        self._restart_hotkeys()
        from away.agentlink.host import attach  # Agent Link (v0.1.20): never raises

        self.agent_link = attach(self)

    def stop(self, join_timeout: float = 3.0) -> None:
        # Stop producing new events first, then make a best-effort release of
        # any synthetic mouse button still held by an in-progress drag.
        self.stop_event.set()
        al = getattr(self, "agent_link", None)
        if al is not None:  # agents lose control and the border/banner disappear first
            with contextlib.suppress(Exception):
                al.stop()
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
        with contextlib.suppress(Exception):
            self.macros.stop_playback()
            self.macros.stop_recording()
        if not getattr(self, "dry_run", True):
            from away.actions_engine import get_engine

            with contextlib.suppress(Exception):  # every source's in-flight action, and anything still held
                get_engine().stop_all("Project Away is closing")
        self._restore_cursor()  # the user's own cursor size, right away (no debounce)
        with contextlib.suppress(Exception):
            self.pipeline.headgaze.save_bias()  # keep the gaze drift correction for the next start
        with contextlib.suppress(Exception):
            self.pipeline.smart.stop()  # the UI Automation element cache thread + number badges
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


class _ProbeStop:
    """``stop`` for :func:`~away.app.camera_loop.probe_cameras`: set when the app stops or another process
    (Settings, ``away calibrate-cameras``) asks for the cameras. With ``devices``, the owner record is kept
    fresh while probing (a probe can take 30 s; the record goes stale after 5 s), so a lease sees the app
    holds the cameras and asks for them instead of failing to open a busy device (v0.1.15)."""

    def __init__(self, stop: threading.Event, share: Any, devices: list[Any] | None, every_s: float = 1.0) -> None:
        self.stop, self.share, self.devices, self.every_s = stop, share, devices, every_s
        self._next = 0.0

    def is_set(self) -> bool:
        if self.stop.is_set():
            return True
        now = time.monotonic()
        if now >= self._next:
            self._next = now + self.every_s
            if self.devices is not None:
                self.share.publish_owner(self.devices)
            return bool(self.share.requested())
        return False


def _requester(share: Any, default: str = "settings") -> str:
    fn = getattr(share, "requester", None)
    try:
        return str(fn() or default) if callable(fn) else default
    except Exception:
        return default


def _still_requested(share: Any, wait: Any) -> bool:
    """``share.still_requested`` (v0.1.21: a vanished request is read once more), or plain ``requested``."""
    fn = getattr(share, "still_requested", None)
    return bool(fn(wait)) if callable(fn) else bool(share.requested())
