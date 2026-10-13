"""v0.1.21: don't type stray sounds when the face gate let an utterance through on weak evidence.

Seen on the user's PC (always-on dictation, face gate on): "uh" and "Dikontak" were typed with the reason
"enrolled face at the screen, facing it (mouth motion weak)" and ``mouth_moving_frames: 0``. The gate allows that
case on purpose (an off-angle webcam barely sees the mouth move), so the transcript itself has to carry the
evidence instead:

* a transcript that is only filler sounds ("uh", "um", "hmm"...) is never typed, whatever the gate said;
* when the mouth did not visibly move (weak gate evidence), a very short transcript (``max_words`` or fewer)
  needs a confident recognition (``min_confidence``; an unknown confidence counts as not confident), and a
  longer one needs at least ``min_confidence_long``.

v0.1.22 (background audio from a YouTube video was typed: "It runs in the blood." at confidence 0.47 with no mouth
motion, "I'm- I know Rossy agent, ooooh." at 0.42 with 7 moving frames out of about 40):

* when the mouth barely moved for the length of the utterance (mouth-moving plus active-speaker frames below
  ``STRICT_MOTION_RATIO`` of the utterance's face frames), the transcript must be recognised with at least
  ``STRICT_CONFIDENCE`` at any length;
* when another app was playing sound during the utterance (Windows audio meter, :mod:`away.voice.other_audio`), the
  mouth must have visibly moved (at least ``STRICT_MOTION_RATIO``), however confident the recognition was: a video
  can be recognised perfectly.

Voice commands are matched before this runs (a bound phrase such as "scroll down" is a command, not typing), so
short commands keep working.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

FILLERS = frozenset(
    {
        "uh",
        "uhh",
        "uhm",
        "um",
        "umm",
        "uhmm",
        "ah",
        "ahh",
        "aah",
        "eh",
        "ehh",
        "er",
        "err",
        "erm",
        "hm",
        "hmm",
        "hmmm",
        "mm",
        "mmm",
        "mhm",
        "mm hmm",
        "uh huh",
        "uh-huh",
        "huh",
        "oh",
        "ooh",
        "uh oh",
        "ha",
        "heh",
        "psst",
        "shh",
        "tsk",
    }
)
MAX_WORDS = 2
MIN_CONFIDENCE = 0.75  # weak mouth evidence + a 1-2 word transcript: Whisper must be this sure
MIN_CONFIDENCE_LONG = 0.45  # weak mouth evidence + a longer transcript
WEAK_REASON = "mouth motion weak"
STRICT_MOTION_RATIO = 0.25  # v0.1.22: below this share of moving-mouth frames the utterance counts as "barely moved"
STRICT_CONFIDENCE = 0.7  # v0.1.22: ... and then needs this confidence, whatever its length


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", str(text).lower())


def is_filler(text: str) -> bool:
    """Only filler sounds: "uh", "Um.", "uh, hmm", "Hmm..." (punctuation and case ignored)."""
    ws = words(text)
    if not ws:
        return False
    joined = " ".join(ws)
    if joined in FILLERS or joined.replace("-", " ") in FILLERS:
        return True
    return all(w in FILLERS or w.replace("-", " ") in FILLERS for w in ws)


def weak_evidence(gate_info: Mapping[str, Any] | None) -> bool:
    """The face gate allowed the utterance without seeing the mouth move (its "facing it, mouth motion weak"
    rule): no active-speaker frames and no mouth-moving frames."""
    if not gate_info or not gate_info.get("allowed", True):
        return False
    reason = str(gate_info.get("reason") or "")
    if WEAK_REASON in reason:
        return True
    moving = gate_info.get("mouth_moving_frames")
    speaker = gate_info.get("speaker_frames")
    return moving == 0 and speaker == 0


def motion_ratio(gate_info: Mapping[str, Any] | None) -> float | None:
    """v0.1.22: the share of the utterance's face frames in which the user's mouth moved (active-speaker frames plus
    mouth-moving frames, over all frames in the gate window). None when the gate gave no frame counts (no face gate,
    or a denied utterance); 0.0 when it saw no frames at all."""
    if not gate_info or not gate_info.get("allowed", True):
        return None
    frames = gate_info.get("frames")
    if not isinstance(frames, (int, float)) or isinstance(frames, bool):
        return None
    moving = gate_info.get("mouth_moving_frames") or 0
    speaker = gate_info.get("speaker_frames") or 0
    try:
        moved = float(moving) + float(speaker)
    except (TypeError, ValueError):
        return None
    if frames <= 0:
        return 0.0
    return max(0.0, min(1.0, moved / float(frames)))


def confidence_of(segments: Iterable[Any]) -> float | None:
    """Recognition confidence 0..1 from Whisper segments: exp(average log-probability), weighted by segment
    length, lowered by the segment's no-speech probability. None when the backend gives neither."""
    total = 0.0
    weight = 0.0
    for s in segments:
        alp = getattr(s, "avg_logprob", None)
        if alp is None:
            continue
        try:
            p = math.exp(min(0.0, float(alp)))
            nsp = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        p *= 1.0 - max(0.0, min(1.0, nsp)) * 0.5
        w = max(1, len(str(getattr(s, "text", "")).strip()))
        total += p * w
        weight += w
    return None if weight == 0 else round(total / weight, 3)


def should_drop(
    text: str,
    gate_info: Mapping[str, Any] | None,
    confidence: float | None,
    *,
    max_words: int = MAX_WORDS,
    min_confidence: float = MIN_CONFIDENCE,
    min_confidence_long: float = MIN_CONFIDENCE_LONG,
    other_audio: bool = False,
    strict_ratio: float = STRICT_MOTION_RATIO,
    strict_confidence: float = STRICT_CONFIDENCE,
) -> str:
    """Why this dictation should not be typed ("" = type it). ``other_audio``: another app was playing sound
    during the utterance (v0.1.22)."""
    if is_filler(text):
        return "only a filler sound"
    conf = "unknown" if confidence is None else f"{confidence:.2f}"
    ratio = motion_ratio(gate_info)
    if ratio is not None:
        if other_audio and ratio < strict_ratio:
            return (
                f"another app was playing sound and your mouth was not seen moving enough "
                f"(moving in {ratio:.0%} of the frames, need {strict_ratio:.0%})"
            )
        if ratio < strict_ratio and (confidence is None or confidence < strict_confidence):
            return (
                f"mouth barely moved (in {ratio:.0%} of the frames) and the transcript was not sure enough "
                f"(confidence {conf}, need {strict_confidence:g})"
            )
    if not weak_evidence(gate_info):
        return ""
    n = len(words(text))
    if n <= max_words:
        if confidence is None or confidence < min_confidence:
            return f"mouth not seen moving and a short, unsure transcript (confidence {conf}, need {min_confidence:g})"
        return ""
    if confidence is not None and confidence < min_confidence_long:
        return f"mouth not seen moving and an unsure transcript (confidence {conf}, need {min_confidence_long:g})"
    return ""
