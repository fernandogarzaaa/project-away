"""A small wake-word model for "Hey <name>" (v0.1.24 model refresh; ``models_next.wake_engine: livekit``).

livekit-wakeword (Apache-2.0) runs a tiny conv-attention classifier on top of a shared mel-spectrogram and Google
speech-embedding front end (both Apache-2.0 ONNX files inside the library). It needs only numpy and onnxruntime.
openWakeWord's own pre-trained models are CC BY-NC-SA and are never shipped or downloaded.

How it replaces the always-on Whisper wake loop: :class:`WakeGate` sits in front of the always-on session. While
nobody has said the wake word, microphone audio goes only to the wake-word model (a few % of one CPU core) and
speech-to-text does not run at all. When the model fires:

* **verify off** (Light / Balanced): the session's wake window opens at once, and the next ``wake_window`` seconds of
  speech go to speech-to-text as if you had said the wake word;
* **verify on** (Best): the last two seconds, wake word included, go to speech-to-text first, and the session only
  acts when speech-to-text also hears "Hey <name>" (the second stage), which removes most false wakes.

A model is trained per name (synthetic speech, a few hours on a GPU; Linux or WSL with espeak-ng/ffmpeg - training
time only, nothing of that is shipped). :func:`training_plan` writes the config and says how; a trained ``.onnx``
is then picked in Settings > Models > Wake word (``models_next.wake_model``).
"""

from __future__ import annotations

import collections
import contextlib
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SR = 16000
WINDOW_S = 2.0  # the classifier sees 16 embeddings = ~2 s of audio
HOP_S = 0.25  # how often the model runs
REFRACTORY_S = 2.0  # one wake per 2 s


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return s or "away"


class LiveKitDetector:
    """Runs the classifier every ``hop_s`` on the last ``window_s`` of audio. ``model``: anything with
    ``predict(audio) -> {name: score}`` (a ``livekit.wakeword.WakeWordModel``; injectable for tests)."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float = 0.6,
        model: Any = None,
        hop_s: float = HOP_S,
        window_s: float = WINDOW_S,
        refractory_s: float = REFRACTORY_S,
    ) -> None:
        if model is None:
            if not model_path or not Path(model_path).is_file():
                raise FileNotFoundError("no wake-word model for your name yet (Settings > Models > Wake word)")
            from away.app import engines

            engines.activate("livekit-wakeword")
            from livekit.wakeword import WakeWordModel

            model = WakeWordModel(models=[str(model_path)])
        self.model = model
        self.threshold = float(threshold)
        self.hop = int(SR * hop_s)
        self.size = int(SR * window_s)
        self.refractory = int(SR * refractory_s)
        self._buf = np.zeros(0, dtype=np.float32)
        self._since = 0
        self._quiet_left = 0
        self.last_score = 0.0

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._since = 0
        self._quiet_left = 0

    def feed(self, chunk: Any) -> bool:
        """True when the wake word was just heard."""
        a = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, a])[-self.size :]
        self._since += len(a)
        self._quiet_left = max(0, self._quiet_left - len(a))
        if self._since < self.hop or len(self._buf) < self.size:
            return False
        self._since = 0
        scores = self.model.predict(self._buf)
        self.last_score = max((float(v) for v in scores.values()), default=0.0)
        if self.last_score >= self.threshold and self._quiet_left == 0:
            self._quiet_left = self.refractory
            return True
        return False


@dataclass
class WakeStats:
    wakes: int = 0
    verified: int = 0
    asr_seconds: float = 0.0  # audio that reached speech-to-text
    heard_seconds: float = 0.0  # all audio
    events: list[str] = field(default_factory=list)


class WakeGate:
    """In front of an :class:`~away.voice.always_on.AmbientSession`: audio reaches the session (and speech-to-text)
    only after the wake-word model fired, and for as long as the session's wake window stays open."""

    def __init__(
        self,
        session: Any,
        detector: Any,
        verify: bool = False,
        wake_window: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        preroll_s: float = WINDOW_S,
        on_wake: Callable[[float], None] | None = None,
    ) -> None:
        self.session = session
        self.detector = detector
        self.verify = verify
        self.wake_window = float(wake_window if wake_window is not None else getattr(session, "wake_window", 5.0))
        self.clock = clock
        self.on_wake = on_wake
        self._pre: collections.deque[NDArray[np.float32]] = collections.deque()
        self._pre_len = 0
        self.preroll = int(SR * preroll_s)
        self._awake_until = -1e9
        self.stats = WakeStats()

    @property
    def awake(self) -> bool:
        if self.clock() <= self._awake_until or bool(getattr(getattr(self.session, "ep", None), "in_speech", False)):
            return True
        # v0.1.26: an agent is listening for you ("Hey <name>" on its own, or a conversational follow-up)
        talk = getattr(self.session, "talk", None)
        try:
            return talk is not None and talk.state.armed() is not None
        except Exception:
            return False

    def _remember(self, a: NDArray[np.float32]) -> None:
        self._pre.append(a)
        self._pre_len += len(a)
        while self._pre and self._pre_len - len(self._pre[0]) >= self.preroll:
            self._pre_len -= len(self._pre.popleft())

    def _preroll(self, seconds: float) -> NDArray[np.float32]:
        if not self._pre:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._pre))[-int(SR * seconds) :]

    def feed(self, chunk: Any) -> list[Any]:
        a = np.asarray(chunk, dtype=np.float32).reshape(-1)
        now = self.clock()
        self.stats.heard_seconds += len(a) / SR
        if self.awake:
            self._awake_until = max(self._awake_until, float(getattr(self.session, "_wake_until", -1e9)))
            self.stats.asr_seconds += len(a) / SR
            self._remember(a)
            return list(self.session.feed(a) or [])
        self._remember(a)
        if not self.detector.feed(a):
            return []
        self.stats.wakes += 1
        score = float(getattr(self.detector, "last_score", 1.0))
        if self.on_wake is not None:
            with contextlib.suppress(Exception):
                self.on_wake(score)
        with contextlib.suppress(Exception):
            self.session.reset()  # a fresh clock origin: the audio below is "now"
        if self.verify:
            # stage 2: speech-to-text must also hear "Hey <name>" in the last two seconds (session.wake_word)
            audio = self._preroll(WINDOW_S)
            self._awake_until = now + 3.0
            self.stats.events.append(f"wake {score:.2f} (verifying)")
        else:
            self.session._wake_until = float(getattr(self.session, "clock", self.clock)()) + self.wake_window
            audio = self._preroll(0.5)
            self._awake_until = now + self.wake_window
            self.stats.events.append(f"wake {score:.2f}")
        self.stats.asr_seconds += len(audio) / SR
        return list(self.session.feed(audio) or [])


def wrap_session(session: Any, opts: Any, preset_best: bool = False, detector: Any = None) -> tuple[Any, str]:
    """``(session behind a WakeGate, "")`` when ``models_next.wake_engine`` is livekit and a model is set;
    otherwise ``(session, reason)`` - speech-to-text keeps listening for the wake word as before."""
    if getattr(opts, "wake_engine", "asr") != "livekit":
        return session, ""
    if not getattr(session, "wake_word", None):
        return session, "turn on the wake word (voice.wake_word_enabled) to use the wake-word model"
    try:
        det = detector or LiveKitDetector(str(getattr(opts, "wake_model", "")), float(opts.wake_threshold))
    except Exception as exc:
        return session, f"wake-word model unavailable ({exc}); speech-to-text listens for the wake word"
    verify = {"on": True, "off": False}.get(str(getattr(opts, "wake_verify", "auto")), preset_best)
    return WakeGate(session, det, verify=verify), ""


# ----------------------------------------------------------------------------------------------- training
TRAIN_STEPS = (
    "livekit-wakeword setup --config {config}",
    "livekit-wakeword run {config}",
)


def training_config(name: str, size: str = "small", n_samples: int = 10000, steps: int = 50000) -> str:
    """The livekit-wakeword YAML for "Hey <name>" (configs/prod.yaml's minimum keys)."""
    phrase = f"hey {str(name).strip().lower()}"
    return (
        f"model_name: hey_{slug(name)}\n"
        f"target_phrases:\n  - {phrase!r}\n"
        f"n_samples: {int(n_samples)}\n"
        "model:\n  model_type: conv_attention\n"
        f"  model_size: {size}\n"
        f"steps: {int(steps)}\n"
        "target_fp_per_hour: 0.2\n"
    )


def can_train_here(platform: str | None = None, which: Callable[[str], str | None] = shutil.which) -> tuple[bool, str]:
    plat = platform or sys.platform
    if plat.startswith("win"):
        if which("wsl") or which("wsl.exe"):
            return False, "train inside WSL (Ubuntu): the commands below run there; then pick the .onnx it makes"
        return False, "training needs Linux or WSL (Windows Subsystem for Linux); then pick the .onnx it makes"
    missing = [t for t in ("livekit-wakeword", "espeak-ng", "ffmpeg") if not which(t)]
    if missing:
        return False, "install first: pip install 'livekit-wakeword[train,eval,export]' and " + ", ".join(
            m for m in missing if m != "livekit-wakeword"
        )
    return True, "ready to train on this computer (a GPU makes it much faster)"


def training_plan(name: str, folder: Path, size: str = "small") -> dict[str, Any]:
    """Write ``hey_<name>.yaml`` into ``folder`` and return what to run. Nothing is started here."""
    folder.mkdir(parents=True, exist_ok=True)
    cfg = folder / f"hey_{slug(name)}.yaml"
    cfg.write_text(training_config(name, size), encoding="utf-8")
    ok, why = can_train_here()
    return {
        "config": str(cfg),
        "commands": [c.format(config=cfg.as_posix()) for c in TRAIN_STEPS],
        "can_train_here": ok,
        "how": why,
        "output": f"output/hey_{slug(name)}/hey_{slug(name)}.onnx (then: Settings > Models > Wake word > Use a file)",
    }
