"""Wire the subsystems together on the event bus.

``Pipeline`` is pure logic (no devices) so it can be driven by synthetic
frames in tests and by ``away demo``. ``AwayApp`` adds the camera, mic,
hotkeys and Wi-Fi threads around it.
"""

from __future__ import annotations

import functools
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from away import paths
from away.actions import ActionExecutor, CustomGestureDetector, Dispatcher, hotkey_map
from away.bindings import Action
from away.config import AppConfig
from away.control.mouse import MouseBackend, MouseController, RecordingBackend
from away.events import EventBus
from away.face.commands import BrowRaiseToggle, MouthOpenTrigger
from away.face.gate import SpeechGate
from away.face.speaker import ActiveSpeakerDetector
from away.face.speaking import SpeakingScorer
from away.face.tracking import FaceTracker
from away.fusion import GestureFusion
from away.hands.cursor import CursorMapper
from away.hands.filters import OneEuro2D
from away.hands.gestures import GestureEvent, GestureStateMachine
from away.wifi.classifier import CoarseCSIClassifier
from away.wifi.features import window_features


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        backend: MouseBackend | None = None,
        screen: tuple[int, int] = (1920, 1080),
        bus: EventBus | None = None,
        temporal: Any = None,
        active_app: Any = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.gestures = GestureStateMachine(cfg.hands.gestures)
        self.mouse = MouseController(backend or RecordingBackend(), CursorMapper(screen, cfg.hands.cursor))
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
        self.executor = ActionExecutor(self.mouse, cfg.bindings.allow_shell, hooks={"pause": self._pause_hook})
        self.dispatcher = Dispatcher(
            cfg.bindings, self.executor, self.gestures, active_app=active_app, paused=lambda: self.paused
        )
        self.dispatcher.on_fire = self._on_fire
        self.custom = CustomGestureDetector(cfg.bindings.custom_gestures, cfg.hands.gestures.model_min_conf)

    def _on_fire(self, trig: str, act: Action) -> None:
        self.bus.publish("action.fired", {"trigger": trig, "action": str(act)})

    # pause -------------------------------------------------------------------
    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.gestures.paused = paused
        self.mouse.enabled = not paused
        if paused:
            self.mouse._release_if_down()

    def _pause_hook(self, arg: str, data: dict[str, Any]) -> None:
        if arg == "toggle" and "paused" in data:  # open palm: the detector already flipped its own state
            self.set_paused(bool(data["paused"]))
        else:
            self.set_paused({"toggle": not self.paused, "on": True, "off": False}[arg])
        self.bus.publish("app.paused", {"paused": self.paused})

    # live reload of bindings + tunables (settings GUI / CLI edits) ------------
    def apply_config(self, cfg: AppConfig) -> None:
        for src, dst in ((cfg.hands.gestures, self.cfg.hands.gestures), (cfg.hands.cursor, self.cfg.hands.cursor)):
            for k, v in vars(src).items():
                setattr(dst, k, v)
        c = self.cfg.hands.cursor
        self.mouse.mapper.filter = OneEuro2D(c.min_cutoff, c.beta, c.d_cutoff)
        self.cfg.bindings = cfg.bindings
        self.dispatcher.set_bindings(cfg.bindings)
        self.custom = CustomGestureDetector(cfg.bindings.custom_gestures, cfg.hands.gestures.model_min_conf)

    def hotkeys(self) -> dict[str, str]:
        return hotkey_map(self.cfg.bindings, self.cfg.voice, self.dispatcher.profile)

    def fire(self, category: str, name: str, data: dict[str, Any] | None = None) -> Action:
        """Fire a voice/face/hotkey trigger through the bindings."""
        return self.dispatcher.trigger(category, name, data)

    # hands -------------------------------------------------------------------
    def on_hand(self, t: float, landmarks: Any, handedness: str = "Right") -> list[str]:
        accepted = []
        if self.temporal is not None and landmarks is not None:
            probs = self.temporal.push(landmarks, handedness)
            self.fusion.update_model(probs)
            for name in self.custom.update(t, probs):
                self.bus.publish("gesture.custom", {"name": name}, t)
                self.dispatcher.on_gesture(GestureEvent("custom", t, {"name": name}))
                accepted.append(f"custom:{name}")
        for ev in self.gestures.update(t, landmarks):
            if not self.fusion.accept(ev):
                self.bus.publish("gesture.rejected", {"kind": ev.kind}, t)
                continue
            self.bus.publish(f"gesture.{ev.kind}", dict(ev.data), t)
            if not self.paused or ev.kind in ("clutch", "hand_lost", "drag_end"):
                self.dispatcher.on_gesture(ev)
            accepted.append(ev.kind)
        return accepted

    # face --------------------------------------------------------------------
    def on_faces(
        self, t: float, boxes: list[tuple[float, float, float]], blendshapes: list[dict[str, float]]
    ) -> int | None:
        ids = self.faces.update(boxes)
        if self._enroll_pending and ids:
            self.gate.enroll({fid: b[2] for fid, b in zip(ids, boxes)})
            self._enroll_pending = False
            self.bus.publish("face.enrolled", {"face_id": self.gate.primary}, t)
        scores = {fid: self.speaking.update(fid, bs) for fid, bs in zip(ids, blendshapes)}
        active = self.asd.update(scores, self.mic_energy)
        self.gate.observe(t, active, len(ids))
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
        backend: MouseBackend
        if dry_run:
            backend = RecordingBackend()
            screen = (1920, 1080)
        else:
            from away.control.mouse import PynputBackend, screen_size

            backend, screen = PynputBackend(), screen_size()
        temporal = None
        if cfg.hands.gestures.use_model:
            from away.hands.runtime import TemporalGestureClassifier, find_model

            p = find_model("gesture_tcn.onnx", self.models_dir)
            temporal = TemporalGestureClassifier(p) if p else None
        from away.voice.context import active_process

        self.pipeline = Pipeline(cfg, backend, screen, self.bus, temporal, active_app=active_process)
        self.voice: Any = None  # VoiceController
        self.threads: list[threading.Thread] = []
        self._listeners: list[Any] = []
        self.bus.subscribe("*", self._log)
        if not cfg.app.always_on:  # start paused and muted until the user resumes from the tray
            self.pipeline.set_paused(True)

    @property
    def dictation(self) -> Any:  # tray compatibility
        return self.voice.dictation if self.voice is not None else None

    def _log(self, ev: Any) -> None:
        if ev.topic not in ("gesture.move", "gesture.scroll", "wifi.state"):
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
            amb.fallback = v.ambient_fallback
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
        import queue

        from away.voice.always_on import AmbientSession, StreamingEndpointer, VoiceController
        from away.voice.asr import make_asr
        from away.voice.cleanup import LLMCleaner, RuleCleaner, Vocabulary
        from away.voice.context import active_app
        from away.voice.dictation import DictationSession, MicStream
        from away.voice.output import ListSink, make_sink
        from away.voice.vad import make_vad

        v = self.cfg.voice
        vocab = Vocabulary.load(v.vocabulary_file)
        rules = RuleCleaner(vocab)
        cleaner = LLMCleaner(rules) if v.llm_cleanup else rules
        asr = make_asr(
            v.asr_backend,
            v.asr_model,
            v.device,
            v.compute_type,
            v.language,
            str(self.models_dir / f"whisper-{v.asr_model}"),
        )
        face_gate = self.cfg.face.enabled and self.cfg.face.gate_dictation
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
        work: queue.Queue[Any] = queue.Queue(maxsize=8)  # ASR runs off the mic callback thread

        def worker() -> None:
            while not self.stop_event.is_set():
                try:
                    job = work.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    job()
                except Exception as exc:
                    self.bus.publish("voice.error", {"error": str(exc)})

        def submit(job: Any) -> None:
            try:
                work.put_nowait(job)
            except queue.Full:
                self.bus.publish("voice.dropped", {"reason": "ASR busy"})

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
            fallback=v.ambient_fallback,
            commands_enabled=v.commands_enabled,
            submit=submit,
        )
        self.voice = VoiceController(v.mode, dictation, ambient, self.bus)
        if not self.cfg.app.always_on:
            self.voice.set_muted(True)
        for kind in ("dictation", "mic", "voice"):
            self.pipeline.executor.hooks[kind] = functools.partial(self._voice_hook, kind)

        def on_chunk(chunk: np.ndarray) -> None:
            self.pipeline.mic_energy = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
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

        combos = {combo: (lambda n=name: fire(n)) for combo, name in self.pipeline.hotkeys().items()}
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
        from away.hands.tracker import HandTracker, camera_frames

        hand_model = self.models_dir / "hand_landmarker.task"
        face_model = self.models_dir / "face_landmarker.task"
        hands = HandTracker(hand_model, self.cfg.hands.max_hands) if self.cfg.hands.enabled else None
        faces = None
        if self.cfg.face.enabled and face_model.exists():
            from away.face.landmarker import FaceLandmarker

            faces = FaceLandmarker(face_model)
        for t, rgb in camera_frames(self.cfg.hands.camera_index, self.cfg.hands.mirror):
            if self.stop_event.is_set():
                break
            if hands is not None:
                fr = hands.process(rgb, t)
                self.pipeline.on_hand(t, fr.landmarks, fr.handedness)
            if faces is not None:
                obs = faces.process(rgb, t)
                self.pipeline.on_faces(t, [o.box for o in obs], [o.blendshapes for o in obs])

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
        for th in self.threads:
            th.start()
        self._restart_hotkeys()

    def stop(self) -> None:
        self.stop_event.set()
        for lst in self._listeners:
            lst.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
