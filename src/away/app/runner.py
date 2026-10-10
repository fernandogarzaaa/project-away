"""Wire the subsystems together on the event bus.

``Pipeline`` is pure logic (no devices) so it can be driven by synthetic
frames in tests and by ``away demo``. ``AwayApp`` adds the camera, mic,
hotkeys and Wi-Fi threads around it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from away import paths
from away.config import AppConfig
from away.control.mouse import MouseBackend, MouseController, RecordingBackend
from away.events import EventBus
from away.face.commands import BrowRaiseToggle
from away.face.gate import SpeechGate
from away.face.speaker import ActiveSpeakerDetector
from away.face.speaking import SpeakingScorer
from away.face.tracking import FaceTracker
from away.fusion import GestureFusion
from away.hands.cursor import CursorMapper
from away.hands.gestures import GestureStateMachine
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
        self.csi = CoarseCSIClassifier()
        self.paused = False
        self.mic_energy = 0.0  # updated by the mic thread
        self._enroll_pending = True

    # hands -------------------------------------------------------------------
    def on_hand(self, t: float, landmarks: Any, handedness: str = "Right") -> list[str]:
        if self.temporal is not None and landmarks is not None:
            self.fusion.update_model(self.temporal.push(landmarks, handedness))
        accepted = []
        for ev in self.gestures.update(t, landmarks):
            if not self.fusion.accept(ev):
                self.bus.publish("gesture.rejected", {"kind": ev.kind}, t)
                continue
            self.bus.publish(f"gesture.{ev.kind}", dict(ev.data), t)
            if ev.kind == "clutch":
                self.paused = bool(ev.data.get("paused"))
            if not self.paused or ev.kind == "clutch":
                self.mouse.handle(ev)
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
    def __init__(self, cfg: AppConfig, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
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
        self.pipeline = Pipeline(cfg, backend, screen, self.bus, temporal)
        self.dictation: Any = None
        self.threads: list[threading.Thread] = []
        self._listener: Any = None
        self.bus.subscribe("*", self._log)

    def _log(self, ev: Any) -> None:
        if ev.topic not in ("gesture.move", "gesture.scroll", "wifi.state"):
            print(f"[{ev.topic}] {ev.data}", flush=True)

    # public controls used by the tray --------------------------------------
    def set_paused(self, paused: bool) -> None:
        self.pipeline.paused = paused
        self.pipeline.mouse.enabled = not paused
        self.bus.publish("app.paused", {"paused": paused})

    def toggle_dictation(self) -> None:
        if self.dictation is not None:
            self.dictation.set_enabled(not self.dictation.enabled)

    # threads -----------------------------------------------------------------
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

    def _start_voice(self) -> None:
        from away.voice.asr import make_asr
        from away.voice.cleanup import LLMCleaner, RuleCleaner, Vocabulary
        from away.voice.context import active_app
        from away.voice.dictation import DictationSession, MicStream, hotkey_listener
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
        gate = self.pipeline.gate.allow if (self.cfg.face.enabled and self.cfg.face.gate_dictation) else None
        sink = ListSink() if self.dry_run else make_sink(v.output)
        self.dictation = DictationSession(
            asr, cleaner, sink, make_vad(v.vad, v.sample_rate), self.bus, v.sample_rate, gate, active_app, vocab.terms()
        )
        self.bus.subscribe("face.brow_toggle", lambda _e: self.dictation.toggle())

        def on_chunk(chunk: np.ndarray) -> None:
            self.pipeline.mic_energy = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            self.dictation.feed(chunk)

        def mic_loop() -> None:
            with MicStream(on_chunk, v.sample_rate):
                self.stop_event.wait()

        self._listener = hotkey_listener(v.ptt_mode, v.hotkey, v.hold_key, self.dictation)
        self.threads.append(threading.Thread(target=mic_loop, daemon=True, name="mic"))

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
        for th in self.threads:
            th.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._listener is not None:
            self._listener.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
