"""Clap control: a single clap resumes gesture control, a double clap pauses it. Pure logic, no devices.

Audio (:class:`ClapDetector`) runs on the microphone chunks the voice pipeline already captures, in 10 ms frames:

* **onset**: the frame is far louder than the running background level and than the 30 ms before it (a
  sharp rise, not a word that swells);
* **short**: 40-60 ms after the onset the level has dropped well below the peak (a clap is a ~5-30 ms burst;
  speech, even a plosive, carries on into a vowel);
* **noisy**: the onset window's spectrum is flat (broadband like a clap, unlike voiced speech or a beep), and
  its peak/RMS (crest) is high.

These give a score in 0..1. :class:`ClapControl` accepts a clap when the score clears a threshold set by
``voice.clap_sensitivity``, adjusted by what the camera saw around the onset (:func:`hand_threshold`): two hands
close together lower the bar, no camera raises it, ``voice.clap_require_hands`` makes the camera mandatory.

:class:`DictationGate` drops claps while the user is mid-utterance (speech that started before the clap and was
not itself started by a clap), so plosives in dictation do not pause anything.

:class:`ClapTiming` keeps single and double claps apart: a second clap within ``clap_double_window`` is a
double (pause, fired at once); a single (resume) fires only after the window has passed with no second clap.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FRAME_MS = 10
LOOKAHEAD_FRAMES = 6  # decide 60 ms after the onset
PRE_FRAMES = 3  # level before the onset (30 ms)
REFRACTORY_S = 0.1  # one clap's echo is not a second clap
MIN_LEVEL = 0.02  # RMS (full scale = 1.0) below which nothing counts as a clap
BACKGROUND_RATIO = 6.0  # onset frame vs running background RMS
RISE_RATIO = 4.0  # onset frame vs the 30 ms before it
DECAY_RATIO = 0.4  # 40-60 ms after the onset the level must be below this fraction of the peak
FLAT_BAND_HZ = (300.0, 6000.0)
MIN_FLATNESS = 0.15  # tonal clicks / beeps / knocks on resonant objects are not claps


def spectral_flatness(x: NDArray[np.float64], sample_rate: int, band: tuple[float, float] = FLAT_BAND_HZ) -> float:
    """Geometric / arithmetic mean of the power spectrum in ``band`` (1 = white noise, ~0 = a pure tone)."""
    if len(x) < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    sel = spec[(freqs >= band[0]) & (freqs <= min(band[1], sample_rate / 2))] + 1e-12
    if sel.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(sel))) / np.mean(sel))


@dataclass
class ClapCandidate:
    t: float  # onset time (same clock as the chunk timestamps)
    score: float  # 0..1
    flatness: float
    crest: float
    rise: float
    decay: float  # level 40-60 ms after the onset / peak level


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


class ClapDetector:
    """Feed float32 mono chunks with the time of their *last* sample; returns clap candidates (scored)."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sr = sample_rate
        self.n = max(1, sample_rate * FRAME_MS // 1000)
        self._buf = np.zeros(0, dtype=np.float64)  # samples not yet framed
        self._frames: deque[NDArray[np.float64]] = deque(maxlen=PRE_FRAMES + LOOKAHEAD_FRAMES + 1)
        self._rms: deque[float] = deque(maxlen=PRE_FRAMES + LOOKAHEAD_FRAMES + 1)
        self._times: deque[float] = deque(maxlen=PRE_FRAMES + LOOKAHEAD_FRAMES + 1)
        self.background = 1e-3
        self._refractory_until = -1e9

    def feed(self, chunk: Any, t_end: float) -> list[ClapCandidate]:
        x = np.asarray(chunk, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return []
        start_t = t_end - len(x) / self.sr
        t0 = start_t - len(self._buf) / self.sr
        self._buf = np.concatenate([self._buf, x])
        out = []
        k = 0
        while len(self._buf) - k >= self.n:
            fr = self._buf[k : k + self.n]
            k += self.n
            c = self._push(fr, t0 + k / self.sr - self.n / self.sr)
            if c is not None:
                out.append(c)
        self._buf = self._buf[k:]
        return out

    def _push(self, fr: NDArray[np.float64], t: float) -> ClapCandidate | None:
        rms = float(np.sqrt(np.mean(fr * fr)))
        self._frames.append(fr)
        self._rms.append(rms)
        self._times.append(t)
        cand = None
        if len(self._rms) == self._rms.maxlen:
            cand = self._evaluate()
        # slow background estimate from quiet-ish frames (not the bursts themselves)
        if rms < self.background * BACKGROUND_RATIO:
            self.background = max(1e-4, 0.98 * self.background + 0.02 * rms)
        return cand

    def _evaluate(self) -> ClapCandidate | None:
        rms = list(self._rms)
        i = PRE_FRAMES  # the frame under test; LOOKAHEAD_FRAMES follow it
        t = self._times[i]
        r = rms[i]
        if t < self._refractory_until or r < MIN_LEVEL or r < BACKGROUND_RATIO * self.background:
            return None
        pre = float(np.mean(rms[:i])) + 1e-6
        rise = r / pre
        if rise < RISE_RATIO:
            return None
        peak = max(rms[i : i + 3])
        decay = max(rms[i + 4 : i + 7]) / (peak + 1e-12)
        if decay > DECAY_RATIO:
            return None
        frames = list(self._frames)
        win = np.concatenate(frames[i : i + 2])  # 20 ms from the onset
        flat = spectral_flatness(win, self.sr)
        if flat < MIN_FLATNESS:
            return None
        burst = np.concatenate(frames[i : i + 5])  # 50 ms
        crest = float(np.max(np.abs(burst)) / (np.sqrt(np.mean(burst * burst)) + 1e-12))
        self._refractory_until = t + REFRACTORY_S
        score = (
            0.45 * _clip01((flat - 0.08) / 0.3)
            + 0.2 * _clip01((crest - 2.5) / 4.0)
            + 0.2 * _clip01((rise - RISE_RATIO) / 16.0)
            + 0.15 * _clip01((DECAY_RATIO - decay) / DECAY_RATIO)
        )
        return ClapCandidate(t, float(score), flat, crest, rise, decay)


VIEW_BEFORE_S = 0.6  # camera frames this long before a clap count as its confirmation
HANDS_GAP = 0.3  # palm centres closer than this (normalised image units) = hands together


def camera_view(views: Any, t0: float, before_s: float = VIEW_BEFORE_S, gap: float = HANDS_GAP) -> str:
    """``views``: iterable of ``(t, n_hands, hands_gap)`` per camera frame. Returns ``together`` (two hands
    close around ``t0``), ``seen`` (the camera was running) or ``none`` (no frames near ``t0``)."""
    near = [(t, n, g) for t, n, g in list(views) if t0 - before_s <= t <= t0 + 0.15]
    if not near:
        return "none"
    if any(n >= 2 and g is not None and g <= gap for _t, n, g in near):
        return "together"
    return "seen"


def audio_threshold(sensitivity: float) -> float:
    """Score a clap needs from audio alone with the camera running (sensitivity 0..1, higher = easier)."""
    return 0.75 - 0.4 * _clip01(sensitivity)


def hand_threshold(sensitivity: float, hands: str, require_hands: bool = False) -> float | None:
    """Threshold given the camera's view near the onset: ``together`` (two hands close), ``seen`` (camera on,
    hands not together), ``none`` (no camera frames). None = reject (``require_hands`` without hands together)."""
    base = audio_threshold(sensitivity)
    if hands == "together":
        return base - 0.15
    if require_hands:
        return None
    return base + 0.1 if hands == "none" else base


class DictationGate:
    """Ignore claps while the user is mid-utterance.

    ``observe(t, busy)`` is called with every chunk (``busy`` = the voice endpointer is inside speech, or a
    push-to-talk / toggle recording is running). A clap's own sound also starts "speech", so a speech segment
    that began within ``clap_slack_s`` of a clap onset is the clap, not dictation."""

    def __init__(self, lead_s: float = 0.1, clap_slack_s: float = 0.3) -> None:
        self.lead_s = lead_s
        self.clap_slack_s = clap_slack_s
        self._since: float | None = None
        self._onsets: deque[float] = deque(maxlen=8)
        self._clap_caused = False

    def note_onset(self, t: float) -> None:
        self._onsets.append(t)
        if self._since is not None and abs(self._since - t) <= self.clap_slack_s:
            self._clap_caused = True

    def observe(self, t: float, busy: bool) -> None:
        if not busy:
            self._since, self._clap_caused = None, False
            return
        if self._since is None:
            self._since = t
            self._clap_caused = any(abs(t - o) <= self.clap_slack_s for o in self._onsets)

    def allows(self, onset: float) -> bool:
        """False when speech that is not a clap was already going on ``lead_s`` before the onset."""
        if self._since is None or self._clap_caused:
            return True
        return not onset - self._since >= self.lead_s


class ClapTiming:
    """Single vs double. ``on_clap`` returns ``double`` at once; ``tick`` returns ``single`` once the window passed."""

    def __init__(self, double_window: float = 0.6) -> None:
        self.double_window = double_window
        self._pending: float | None = None
        self._cooldown_until = -1e9

    def on_clap(self, t: float) -> str | None:
        if self._pending is not None and t - self._pending <= self.double_window:
            self._pending = None
            self._cooldown_until = t + self.double_window  # a third clap does not start a new single
            return "double"
        if t < self._cooldown_until:
            return None
        self._pending = t
        return None

    def tick(self, t: float) -> str | None:
        if self._pending is not None and t - self._pending > self.double_window:
            self._pending = None
            return "single"
        return None

    def reset(self) -> None:
        self._pending = None


class ClapControl:
    """Detector + dictation gate + camera confirmation + single/double timing.

    ``cfg`` is the live :class:`~away.config.VoiceConfig` (clap_enabled, clap_sensitivity, clap_double_window,
    clap_require_hands are read on every chunk, so Settings changes apply at once). ``busy()`` says whether the
    user is mid-utterance; ``hands(t)`` returns ``together`` | ``seen`` | ``none`` for the camera's view around
    the onset. :meth:`feed` returns ``[(kind, t)]`` with kind ``single`` | ``double``.
    """

    def __init__(
        self,
        cfg: Any,
        sample_rate: int = 16000,
        busy: Callable[[], bool] = lambda: False,
        hands: Callable[[float], str] = lambda _t: "none",
        on_candidate: Callable[[ClapCandidate, bool, str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.detector = ClapDetector(sample_rate)
        self.gate = DictationGate()
        self.timing = ClapTiming(float(cfg.clap_double_window))
        self.busy = busy
        self.hands = hands
        self.on_candidate = on_candidate
        self.accepted: list[float] = []

    def feed(self, chunk: Any, t_end: float) -> list[tuple[str, float]]:
        if not self.cfg.clap_enabled:
            self.timing.reset()
            return []
        self.timing.double_window = float(self.cfg.clap_double_window)
        out: list[tuple[str, float]] = []
        for c in self.detector.feed(chunk, t_end):
            self.gate.note_onset(c.t)
            view = self.hands(c.t)
            thr = hand_threshold(float(self.cfg.clap_sensitivity), view, bool(self.cfg.clap_require_hands))
            ok = thr is not None and c.score >= thr and self.gate.allows(c.t)
            if self.on_candidate is not None:
                self.on_candidate(c, ok, view)
            if not ok:
                continue
            self.accepted.append(c.t)
            del self.accepted[:-20]
            kind = self.timing.on_clap(c.t)
            if kind:
                out.append((kind, c.t))
        # busy is sampled after the detector saw this chunk, so a clap's own "speech" is recognised as the clap
        self.gate.observe(t_end, bool(self.busy()))
        single = self.timing.tick(t_end)
        if single:
            out.append((single, t_end))
        return out
