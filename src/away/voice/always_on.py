"""Always-on (ambient) voice input and the voice-mode state machine.

* :class:`StreamingEndpointer` segments a continuous mic stream into
  utterances (adaptive energy threshold, pre-roll, end-of-utterance silence,
  max length). Audio lives only in memory and is dropped after transcription:
  nothing is ever written to disk.
* :func:`decide` is the pure policy that turns a transcript into "run this
  command", "type this", "wake" or "ignore" (wake word, face/active-speaker
  gate, listening on/off, command-vs-dictation intent).
* :class:`AmbientSession` glues endpointer -> ASR -> :func:`decide` -> sink/commands.
* :class:`VoiceController` is the mode state machine over the three input
  modes (``always_on``, ``push_to_talk``, ``toggle``), mute, and listening.
"""

from __future__ import annotations

import contextlib
import functools
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from away.bindings import Action, normalize_phrase
from away.events import EventBus
from away.voice import latency
from away.voice.asr import ASRBackend
from away.voice.context import whisper_prompt
from away.voice.dictation import Cleaner, DictationSession
from away.voice.hallucination import clean_transcript
from away.voice.output import TextSink
from away.voice.talk import is_stop
from away.voice.weak_speech import STRICT_CONFIDENCE, STRICT_MOTION_RATIO, should_drop

WAKE_LEADS = ("hey", "ok", "okay", "hi")
START_ACTIONS = (Action("dictation", "start"), Action("dictation", "toggle"))


@dataclass
class Utterance:
    audio: NDArray[np.float32]
    t0: float  # stream seconds (plus the session's clock origin)
    t1: float
    closed_at: float | None = None  # v0.1.26 latency log: when the end of speech was decided (session clock)
    end_wait_s: float = 0.0  # the silence waited for before deciding it


def sensitivity_params(sensitivity: float) -> tuple[float, float]:
    """Map a 0..1 sensitivity to (ratio over noise floor, absolute RMS floor)."""
    s = min(max(float(sensitivity), 0.0), 1.0)
    return 1.5 + 4.5 * (1.0 - s), 0.002 + 0.02 * (1.0 - s)


class StreamingEndpointer:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        sensitivity: float = 0.5,
        end_silence_ms: int = 700,
        min_utterance_ms: int = 250,
        max_utterance_s: float = 15.0,
        preroll_ms: int = 300,
        start_frames: int = 2,
        noise_alpha: float = 0.05,
        quick_end_silence_ms: int | None = None,
    ) -> None:
        self.sr = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.frame_ms = frame_ms
        self.set_sensitivity(sensitivity)
        self.end_frames = max(1, round(end_silence_ms / frame_ms))
        # v0.1.26: while you talk to an agent (``quick()`` True) a shorter pause ends the utterance: the agent
        # starts answering sooner. Dictation keeps the longer pause (you may stop to think mid-sentence).
        self.quick: Callable[[], bool] | None = None
        self.set_quick_end_silence(quick_end_silence_ms)
        self.last_end_wait_s = 0.0  # the silence waited before the last utterance was closed (latency log)
        self.min_frames = max(1, round(min_utterance_ms / frame_ms))
        self.max_frames = max(self.min_frames + 1, int(max_utterance_s * 1000 / frame_ms))
        self.preroll = max(0, round(preroll_ms / frame_ms))
        self.start_frames = start_frames
        self.noise_alpha = noise_alpha
        self.origin = 0.0
        self.reset()

    def set_quick_end_silence(self, ms: int | None) -> None:
        self.quick_end_frames = self.end_frames if ms is None else max(1, round(int(ms) / self.frame_ms))

    def _end_frames_now(self) -> int:
        q = self.quick
        if q is not None:
            try:
                if q():
                    return min(self.end_frames, self.quick_end_frames)
            except Exception:
                pass
        return self.end_frames

    def set_sensitivity(self, sensitivity: float) -> None:
        self.ratio, self.min_rms = sensitivity_params(sensitivity)

    def reset(self) -> None:
        """Drop every buffered sample (privacy: called on mute and after each utterance)."""
        self._rest = np.zeros(0, dtype=np.float32)
        self._pre: deque[NDArray[np.float32]] = deque(maxlen=max(self.preroll, 1))
        self._utt: list[NDArray[np.float32]] = []
        self._in_speech = False
        self._run = 0  # consecutive speech frames while idle
        self._silence = 0
        self._speech_frames = 0
        self._utt_start = 0
        self._frames_seen = 0
        self.noise: float | None = None

    @property
    def threshold(self) -> float:
        return max(self.min_rms, (self.noise or 0.0) * self.ratio)

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, chunk: Any) -> list[Utterance]:
        x = np.concatenate([self._rest, np.asarray(chunk, dtype=np.float32).reshape(-1)])
        n = len(x) // self.frame
        self._rest = x[n * self.frame :].copy()
        out: list[Utterance] = []
        for i in range(n):
            fr = x[i * self.frame : (i + 1) * self.frame]
            u = self._step(fr)
            if u is not None:
                out.append(u)
        return out

    def _t(self, frame_idx: int) -> float:
        return self.origin + frame_idx * self.frame / self.sr

    def _step(self, fr: NDArray[np.float32]) -> Utterance | None:
        idx = self._frames_seen
        self._frames_seen += 1
        rms = float(np.sqrt(np.mean(fr.astype(np.float64) ** 2)))
        if self.noise is None:
            self.noise = rms
        speech = rms > self.threshold
        if not self._in_speech:
            if not speech:  # adapt the noise floor only on non-speech
                self.noise = (1 - self.noise_alpha) * self.noise + self.noise_alpha * rms
            self._run = self._run + 1 if speech else 0
            self._pre.append(fr)
            if self._run >= self.start_frames:
                self._in_speech = True
                self._utt = list(self._pre) if self.preroll else list(self._pre)[-self._run :]
                self._utt_start = idx + 1 - len(self._utt)
                self._pre.clear()
                self._speech_frames = self._run
                self._silence = 0
            return None
        self._utt.append(fr)
        if speech:
            self._speech_frames += 1
            self._silence = 0
        else:
            self._silence += 1
        end_frames = self._end_frames_now()
        if self._silence >= end_frames or len(self._utt) >= self.max_frames:
            self.last_end_wait_s = self._silence * self.frame / self.sr
            return self._close(idx + 1)
        return None

    def _close(self, end_idx: int) -> Utterance | None:
        utt, speech = self._utt, self._speech_frames
        keep_tail = max(0, self._silence - max(1, int(150 / self.frame_ms)))  # leave ~150 ms of trailing silence
        if keep_tail and len(utt) > keep_tail:
            utt = utt[: len(utt) - keep_tail]
        start = self._utt_start
        self._utt, self._in_speech, self._run, self._silence, self._speech_frames = [], False, 0, 0, 0
        if speech < self.min_frames:
            return None
        audio = np.concatenate(utt).astype(np.float32)
        return Utterance(audio, self._t(start), self._t(start + len(utt)))

    def flush(self) -> Utterance | None:
        """End an in-progress utterance now (e.g. when listening stops)."""
        if not self._in_speech:
            return None
        return self._close(self._frames_seen)


# --------------------------------------------------------------------------- policy
def match_wake(text: str, wake_word: str) -> tuple[bool, str]:
    """Does ``text`` start with the wake word? Returns (matched, rest-with-original-casing).

    Tolerates a leading "hey/ok", punctuation, and ASR splitting the word ("a way" for "away").
    """
    wake = normalize_phrase(wake_word).replace(" ", "")
    if not wake:
        return False, text
    raw = text.split()
    norm = [normalize_phrase(w).replace(" ", "") for w in raw]
    start = 0
    while start < len(norm) and not norm[start]:
        start += 1
    if start < len(norm) - 1 and norm[start] in WAKE_LEADS:
        start += 1
    joined = ""
    for k in range(start, min(len(norm), start + 4)):
        joined += norm[k]
        if joined == wake:
            rest = " ".join(raw[k + 1 :]).lstrip(" ,.:;!?-")
            return True, rest
        if not wake.startswith(joined):
            break
    return False, text


@dataclass
class Decision:
    kind: str  # command | type | wake | ignore | listen / message (v0.1.22: talking to an agent)
    reason: str = ""
    action: Action | None = None
    text: str = ""
    phrase: str = ""


def decide(
    text: str,
    *,
    commands: dict[str, Action],
    wake_word: str | None = None,
    in_wake_window: bool = False,
    gate: bool | None = None,
    listening: bool = True,
    intent: str = "dictation",
    fallback: str = "commands_only",
    commands_enabled: bool = True,
) -> Decision:
    """What to do with one always-on transcript.

    ``gate``: result of the face/active-speaker gate for the utterance window, or
    None when no camera gate is available. Only consulted when the wake word is off.
    """
    if not normalize_phrase(text):
        return Decision("ignore", "empty transcript")
    body = text
    if wake_word:
        matched, rest = match_wake(text, wake_word)
        if matched and not normalize_phrase(rest):
            return Decision("wake", "wake word heard")
        if not matched and not in_wake_window:
            return Decision("ignore", "no wake word")
        body = rest if matched else text
        may_type = True
    else:
        if gate is False:
            return Decision("ignore", "face gate: you were not facing the screen and speaking")
        may_type = gate is True or fallback == "type"
    phrase = normalize_phrase(body)
    act = commands.get(phrase) if commands_enabled else None
    if act is not None and act.kind != "none":
        if not listening and act not in START_ACTIONS:
            return Decision("ignore", "stopped listening (say the start command)", phrase=phrase)
        return Decision("command", "matched voice command", act, phrase=phrase)
    if not listening:
        return Decision("ignore", "stopped listening", phrase=phrase)
    if intent == "command":
        return Decision("ignore", "command mode: not a bound phrase", phrase=phrase)
    if not may_type:
        return Decision(
            "ignore",
            "no face gate available: commands only (turn the face gate off in Settings, or set "
            "voice.ambient_fallback: type, to type without a wake word)",
        )
    return Decision("type", "dictation", text=body.strip(), phrase=phrase)


@dataclass
class AmbientResult:
    decision: Decision
    raw: str = ""
    typed: str = ""
    seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class AmbientSession:
    """Always-on listening: endpoint -> ASR -> decide -> run command / type text."""

    def __init__(
        self,
        asr: ASRBackend,
        cleaner: Cleaner,
        sink: TextSink,
        endpointer: StreamingEndpointer,
        *,
        commands: Callable[[], dict[str, Action]],
        run_command: Callable[[Action, str], None],
        bus: EventBus | None = None,
        gate: Callable[[float, float], bool] | None = None,
        context: Callable[[], str | None] | None = None,
        vocabulary: list[str] | None = None,
        wake_word: str | None = None,
        wake_window: float = 5.0,
        intent: str = "dictation",
        fallback: str = "commands_only",
        commands_enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        submit: Callable[[Callable[[], None]], None] | None = None,
        gate_info: Callable[[], dict[str, Any]] | None = None,
        echo: Any = None,
        other_audio: Callable[[float, float], bool] | None = None,
        talk: Any = None,
    ) -> None:
        self.asr, self.cleaner, self.sink, self.ep = asr, cleaner, sink, endpointer
        # v0.1.22: the computer's own voice (away.voice.speech.EchoGuard: .overlaps(t0, t1)) is never transcribed
        self.echo = echo
        # v0.1.22: another app played sound during [t0, t1] (away.voice.other_audio); then the mouth must move
        self.other_audio = other_audio
        self.other_audio_needs_mouth = True
        self.strict_ratio = STRICT_MOTION_RATIO
        self.strict_confidence = STRICT_CONFIDENCE
        # v0.1.22: "Hey <agent>, ..." goes to that agent's inbox (away.voice.talk.TalkHook)
        self.talk = talk
        if talk is not None and getattr(endpointer, "quick", None) is None:
            endpointer.quick = self.talking  # v0.1.26: a shorter end-of-speech pause while you talk to an agent
        self.gate_info = gate_info  # the face gate's evidence for its last decision (logged with the utterance)
        self.commands = commands
        self.run_command = run_command
        self.bus = bus
        self.gate = gate
        self.context = context
        self.vocabulary = vocabulary or []
        self.wake_word = wake_word
        self.wake_window = wake_window
        self.intent = intent
        self.fallback = fallback
        self.commands_enabled = commands_enabled
        self.clock = clock
        self.submit = submit  # None = process inline (tests); the app passes a worker-thread submit
        self.listening = True
        self._origin_set = False
        self._wake_until = -1e9
        self.results: deque[AmbientResult] = deque(maxlen=50)
        self._lock = threading.Lock()

    def talking(self) -> bool:
        """v0.1.26: you are talking to an agent right now: its listening window is open (after "Hey <name>", or a
        reply in conversational mode), or the wake-word model just fired with agents to talk to."""
        talk = self.talk
        if talk is None:
            return False
        try:
            if not talk.ready():
                return False
            if talk.state.armed() is not None:
                return True
            return self.clock() <= self._wake_until
        except Exception:
            return False

    def _same_clock(self) -> bool:
        talk = self.talk
        return talk is not None and getattr(getattr(talk, "state", None), "clock", None) is self.clock

    def _onset(self) -> float | None:
        """v0.1.27: when the utterance being processed STARTED (session clock), for the listening window. From the
        moment the endpointer closed it minus its silence and length (the stream clock can lag while muted)."""
        u = getattr(self, "_utt", None)
        if u is None or not self._same_clock():
            return None
        if u.closed_at is not None:
            return float(u.closed_at) - float(u.end_wait_s) - max(0.0, float(u.t1) - float(u.t0))
        return float(u.t0)

    def _speech_end(self) -> float | None:
        """v0.1.27: when the wake phrase ended: the window starts there, not after speech-to-text finished."""
        u = getattr(self, "_utt", None)
        if u is None or not self._same_clock() or u.closed_at is None:
            return None
        return float(u.closed_at) - float(u.end_wait_s)

    def _publish(self, topic: str, data: dict[str, Any]) -> None:
        if self.bus is not None:
            self.bus.publish(topic, data)

    def reset(self) -> None:
        with self._lock:
            self.ep.reset()
            self._origin_set = False

    def feed(self, chunk: Any) -> list[AmbientResult]:
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            if not self._origin_set:
                self.ep.origin = self.clock() - len(arr) / self.ep.sr
                self._origin_set = True
            utts = self.ep.feed(arr)
            if utts:
                now = self.clock()
                for u in utts:
                    u.closed_at, u.end_wait_s = now, float(self.ep.last_end_wait_s)
        out: list[AmbientResult] = []
        for u in utts:
            if self.submit is None:
                out.append(self.process(u))
            else:
                self.submit(functools.partial(self._process_quiet, u))
        return out

    def _process_quiet(self, u: Utterance) -> None:
        self.process(u)

    def process(self, u: Utterance) -> AmbientResult:
        seconds = len(u.audio) / self.ep.sr
        if self.echo is not None and _safe(lambda: self.echo.overlaps(u.t0, u.t1)):
            # v0.1.22: the agent's spoken "Can you hear me now?" was transcribed and typed
            u.audio = np.zeros(0, dtype=np.float32)
            d0 = Decision("ignore", "dropped: the computer's own voice was speaking")
            res0 = AmbientResult(d0, seconds=seconds)
            self.results.append(res0)
            self._publish("voice.utterance", {"decision": "ignore", "reason": d0.reason, "text": ""})
            return res0
        app = self.context() if self.context else None
        prompt = whisper_prompt(app, self.vocabulary)
        heard = self.asr.transcribe(u.audio, prompt)
        self._stt_at = self.clock()
        self._utt = u
        u.audio = np.zeros(0, dtype=np.float32)  # privacy: the audio is gone once transcribed
        commands = self.commands()
        if normalize_phrase(heard) in commands:  # a bound phrase is never a hallucination ("thank you")
            raw, dropped = heard, ""
        else:
            raw, why = clean_transcript(heard, prompt, self.vocabulary, seconds)
            dropped = why if not raw and heard.strip() else ""
        gate = self.gate(u.t0, u.t1) if (self.gate is not None and not self.wake_word) else None
        now = self.clock()
        conf = _confidence(self.asr)
        gi: dict[str, Any] | None = None
        if gate is not None and self.gate_info is not None:
            with contextlib.suppress(Exception):
                gi = dict(self.gate_info())  # frames, face/speaking fractions, max mouth score
        other = False
        oa = self.other_audio
        if oa is not None and self.other_audio_needs_mouth:
            other = _safe(lambda: oa(u.t0, u.t1))
        if not dropped and self.talk is not None and self.listening:
            msg = self._talk(raw, gate, gi, conf, other, seconds, app)
            if msg is not None:
                return msg
        d = decide(
            raw,
            commands=commands,
            wake_word=self.wake_word,
            in_wake_window=now <= self._wake_until,
            gate=gate,
            listening=self.listening,
            intent=self.intent,
            fallback=self.fallback,
            commands_enabled=self.commands_enabled,
        )
        if dropped:
            d = Decision("ignore", f"dropped: {dropped}")
        if d.kind == "type":
            # v0.1.21: "uh" / "Dikontak" were typed with the gate's "mouth motion weak" allowance (0 moving frames)
            # v0.1.22: + barely-moving mouth needs confidence 0.7; other apps playing sound need mouth motion
            why = should_drop(
                d.text,
                gi,
                conf,
                other_audio=other,
                strict_ratio=self.strict_ratio,
                strict_confidence=self.strict_confidence,
            )
            if why:
                d = Decision("ignore", f"dropped: {why}", phrase=d.phrase)
        res = AmbientResult(d, raw=raw or heard, seconds=seconds, meta={"app": app, "gate": gate, "confidence": conf})
        if d.kind == "wake":
            self._wake_until = now + self.wake_window
        elif d.kind == "command" and d.action is not None:
            self._wake_until = now + self.wake_window if self.wake_word else -1e9
            self.run_command(d.action, d.phrase)
        elif d.kind == "type":
            text = self.cleaner.clean(d.text, app)
            if text:
                self.sink.emit(text + " ")
                res.typed = text
            self._wake_until = -1e9
        self.results.append(res)
        data: dict[str, Any] = {"decision": d.kind, "reason": d.reason, "text": res.typed or d.phrase}
        if conf is not None:
            data["confidence"] = conf
        if gi is not None:
            data["gate"] = gi
        if other:
            data["other_audio"] = True
        self._publish("voice.utterance", data)
        return res

    def _talk(
        self,
        raw: str,
        gate: bool | None,
        gi: dict[str, Any] | None,
        conf: float | None,
        other: bool,
        seconds: float,
        app: str | None,
    ) -> AmbientResult | None:
        """v0.1.22 "Talk to your agent": "Hey <agent>, <message>" (or anything said while a summon is waiting)
        goes to the agent's inbox instead of being typed. None: not a message, carry on as dictation."""
        talk = self.talk
        if not raw.strip():
            return None
        onset = self._onset()
        try:
            route = talk.route(raw)
            waiting = talk.state.armed(at=onset) if talk.ready() else None
        except Exception:
            return None
        if route is None and waiting is None:
            with contextlib.suppress(Exception):
                late = talk.state.expired(at=onset) if onset is not None else talk.state.expired()
                if late is not None:  # v0.1.27 feedback: said just after the window closed
                    talk.report("expired", str(late), "the listening window had closed, so it was treated as dictation")
            return None
        if route is None and waiting == getattr(talk, "conversation", None) and is_stop(raw):
            # v0.1.26 conversational mode: "stop" / "that's all" ends the conversation (and stops the agent)
            target0 = str(waiting)
            talk.stop_conversation(target0)
            d0 = Decision("message", "conversation ended")
            res0 = AmbientResult(d0, raw=raw, seconds=seconds, meta={"app": app, "gate": gate, "confidence": conf})
            self.results.append(res0)
            self._publish("voice.utterance", {"text": "", "decision": d0.kind, "reason": d0.reason})
            return res0
        target = route.target if route is not None else str(waiting)
        body = route.body if route is not None else raw.strip()
        source = "voice" if route is not None else str(getattr(talk.state, "source", "voice"))
        data: dict[str, Any] = {"text": ""}  # the message itself is not repeated on the event bus
        if conf is not None:
            data["confidence"] = conf
        if gi is not None:
            data["gate"] = gi
        # v0.1.27: the face gate is for stray dictation typing. A message addressed to an agent by name (or said
        # inside its listening window / conversation) is never blocked by it: with the camera not seeing a face
        # (face_fraction 0) "Hey <name>" used to be dropped and the agent never heard anything.
        if gate is False:
            data["face_gate_bypassed"] = True
        if route is not None and not body:
            end = self._speech_end()
            if end is not None:
                talk.arm_at(target, "voice", at=end)
            else:
                talk.arm(target, "voice")
            d = Decision("listen", f"waiting for your message ({route.phrase})", phrase=route.phrase)
        else:
            # mouth-motion evidence comes from the face gate, which does not apply to agent messages; only fillers
            # are dropped. Exception (v0.1.22 rule kept): another app playing sound needs the mouth seen moving.
            why = should_drop(
                body,
                gi if other else None,
                conf,
                other_audio=other,
                strict_ratio=self.strict_ratio,
                strict_confidence=self.strict_confidence,
            )
            if why:
                d = Decision("ignore", f"dropped: {why}")
                talk.report("dropped", target, why)
            else:
                talk.state.disarm()
                u = getattr(self, "_utt", None)
                if u is not None and u.closed_at is not None:  # v0.1.26 latency log
                    latency.start(u.closed_at, u.end_wait_s, self.clock).mark("stt", getattr(self, "_stt_at", None))
                label = talk.send(target, body, source)
                d = Decision("message", label or "could not send it to your agent")
                data["sent_to"] = label
        res = AmbientResult(d, raw=raw, seconds=seconds, meta={"app": app, "gate": gate, "confidence": conf})
        self.results.append(res)
        data.update(decision=d.kind, reason=d.reason)
        self._publish("voice.utterance", data)
        return res


def _safe(fn: Callable[[], Any]) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


def _confidence(asr: Any) -> float | None:
    """The recognizer's confidence for its last transcript (0..1), None when it does not say."""
    try:
        v = getattr(asr, "last_confidence", None)
    except Exception:
        return None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


HELD_SAMPLES = 16000  # ~1 s of microphone audio (16 kHz) kept while the computer speaks, for a barge-in

VOICE_STATES = ("muted", "stopped", "listening", "recording", "idle", "off")


class VoiceController:
    """Mode state machine: always_on | push_to_talk | toggle, plus mic mute and listening on/off.

    ``state``:
      * ``off``       voice disabled
      * ``muted``     mic muted (audio is discarded before any processing)
      * ``stopped``   always_on, listening paused by "stop listening" (only the start command is honoured)
      * ``listening`` always_on, ambient listening
      * ``recording`` push_to_talk/toggle while capturing
      * ``idle``      push_to_talk/toggle waiting for the hotkey
    """

    def __init__(
        self,
        mode: str,
        dictation: DictationSession | None = None,
        ambient: AmbientSession | None = None,
        bus: EventBus | None = None,
        enabled: bool = True,
    ) -> None:
        if mode not in ("always_on", "push_to_talk", "toggle"):
            raise ValueError(f"voice mode must be always_on|push_to_talk|toggle, got {mode!r}")
        self.mode = mode
        self.dictation = dictation
        self.ambient = ambient
        self.bus = bus
        self.enabled = enabled
        self.muted = False
        self.echo: Any = None  # v0.1.22: away.voice.speech.EchoGuard (the app sets it)
        self._echo_dropping = False
        # v0.1.26: the last ~1 s heard while the voice spoke (replayed only after a barge-in, else dropped)
        self._held: deque[NDArray[np.float32]] = deque()
        self._last_state: str | None = None
        self._emit()

    # ----------------------------------------------------------------- state
    @property
    def listening(self) -> bool:
        return self.ambient.listening if self.ambient is not None else True

    @property
    def intent(self) -> str:
        return self.ambient.intent if self.ambient is not None else "dictation"

    @property
    def state(self) -> str:
        if not self.enabled:
            return "off"
        if self.muted:
            return "muted"
        if self.mode == "always_on":
            return "listening" if self.listening else "stopped"
        return "recording" if (self.dictation is not None and self.dictation.recording) else "idle"

    def _emit(self) -> None:
        s = self.state
        if s != self._last_state:
            self._last_state = s
            if self.bus is not None:
                self.bus.publish("voice.state", {"state": s, "mode": self.mode, "intent": self.intent})

    # ----------------------------------------------------------------- audio
    def feed(self, chunk: Any) -> Any:
        if not self.enabled or self.muted:
            return None  # dropped: never buffered, never transcribed
        if self.echo is not None and _safe(self.echo.active):
            # v0.1.22: the computer's own voice is speaking (and 0.7 s after): the microphone hears nothing
            if not self._echo_dropping and self.ambient is not None:
                self.ambient.reset()  # an utterance cut by the voice is thrown away, not half-transcribed
            self._echo_dropping = True
            self._held.append(np.asarray(chunk, dtype=np.float32).reshape(-1))  # v0.1.26: for a barge-in
            while len(self._held) > 1 and sum(len(c) for c in self._held) - len(self._held[0]) >= HELD_SAMPLES:
                self._held.popleft()
            return None
        replay = None
        if self._echo_dropping:
            self._echo_dropping = False
            if self._held and _safe(getattr(self.echo, "take_barge", lambda: False)):
                # v0.1.26 barge-in: you talked over the voice; your first words were heard while it spoke
                replay = np.concatenate([*self._held, np.asarray(chunk, dtype=np.float32).reshape(-1)])
            self._held.clear()
        if replay is not None:
            chunk = replay
        if self.mode == "always_on":
            return self.ambient.feed(chunk) if self.ambient is not None else None
        if self.dictation is not None:
            self.dictation.feed(chunk)
        return None

    # --------------------------------------------------------------- controls
    def set_mode(self, mode: str) -> None:
        if mode not in ("always_on", "push_to_talk", "toggle"):
            raise ValueError(f"voice mode must be always_on|push_to_talk|toggle, got {mode!r}")
        if self.dictation is not None:
            self.dictation.cancel()
        if self.ambient is not None:
            self.ambient.reset()
        self.mode = mode
        self._emit()

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            if self.dictation is not None:
                self.dictation.cancel()
            if self.ambient is not None:
                self.ambient.reset()
        self._emit()

    def set_listening(self, on: bool) -> None:
        if self.ambient is not None:
            self.ambient.listening = on
        self._emit()

    def ptt_press(self) -> None:
        if self.mode == "push_to_talk" and not self.muted and self.dictation is not None:
            self.dictation.start()
            self._emit()

    def ptt_release(self) -> Any:
        if self.mode == "push_to_talk" and self.dictation is not None and self.dictation.recording:
            res = self.dictation.stop()
            self._emit()
            return res
        return None

    def toggle(self) -> Any:
        """The ptt/toggle hotkey: start/stop a recording, or (always_on) stop/start listening."""
        if self.mode == "always_on":
            self.set_listening(not self.listening)
            return None
        if self.muted or self.dictation is None:
            return None
        res = self.dictation.toggle()
        self._emit()
        return res

    def handle_action(self, kind: str, arg: str) -> None:
        """Hook for ``dictation:*``, ``mic:*`` and ``voice:*`` actions."""
        if kind == "mic":
            self.set_muted({"toggle": not self.muted, "on": False, "off": True}[arg])
        elif kind == "dictation":
            if self.mode == "always_on":
                self.set_listening({"toggle": not self.listening, "start": True, "stop": False}[arg])
            elif self.dictation is not None and not self.muted:
                if arg == "toggle":
                    self.dictation.toggle()
                elif arg == "start":
                    self.dictation.start()
                elif self.dictation.recording:
                    self.dictation.stop()
                self._emit()
        elif kind == "voice" and self.ambient is not None:
            cur = self.ambient.intent
            self.ambient.intent = {
                "command_mode": "command",
                "dictation_mode": "dictation",
                "toggle_mode": "dictation" if cur == "command" else "command",
            }[arg]
            if self.bus is not None:
                self.bus.publish("voice.state", {"state": self.state, "mode": self.mode, "intent": self.intent})


class ASRQueue:
    """Bounded hand-off from the mic thread to the speech-recognition worker.

    When recognition cannot keep up, the *oldest* waiting utterance is dropped (the newest speech is the
    one you are waiting for), ``on_drop`` is called for each, and ``on_overload`` once per session so the
    user hears about it instead of speech vanishing silently.
    """

    def __init__(
        self,
        maxsize: int = 3,
        on_drop: Callable[[int], None] | None = None,
        on_overload: Callable[[], None] | None = None,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self.on_drop = on_drop
        self.on_overload = on_overload
        self.dropped = 0
        self.warned = False
        self._q: deque[Callable[[], None]] = deque()
        self._cv = threading.Condition()

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)

    def put(self, job: Callable[[], None]) -> None:
        overload = did_drop = False
        with self._cv:
            if len(self._q) >= self.maxsize:
                self._q.popleft()
                self.dropped += 1
                did_drop = True
                overload = not self.warned
                self.warned = True
            self._q.append(job)
            self._cv.notify()
            dropped = self.dropped
        if did_drop and self.on_drop is not None:
            self.on_drop(dropped)
        if overload and self.on_overload is not None:
            self.on_overload()

    def get(self, timeout: float = 0.5) -> Callable[[], None] | None:
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            return self._q.popleft() if self._q else None


class GateHint:
    """Counts consecutive utterances the face gate rejected; after ``threshold`` of them, fires once per
    session (the tray then suggests turning the face gate off)."""

    def __init__(self, threshold: int = 10) -> None:  # v0.1.12: 10 (was 5), now that the gate is looser
        self.threshold = threshold
        self.count = 0
        self.fired = False

    def observe(self, data: dict[str, Any]) -> bool:
        reason = str(data.get("reason", ""))
        if data.get("decision") == "ignore" and reason.startswith("face gate"):
            self.count += 1
        elif data.get("decision") in ("type", "command"):
            self.count = 0
        if self.count >= self.threshold and not self.fired:
            self.fired = True
            return True
        return False
