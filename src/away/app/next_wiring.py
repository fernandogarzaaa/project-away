"""Where the running app plugs in the 2026-10 models (v0.1.24). Each helper returns the v0.1.23 object unchanged
when nothing new is chosen, so the runner's behaviour only changes after a model or preset is picked."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

Notice = Callable[[str, str], None]


def opts(cfg: Any) -> Any:
    return getattr(cfg, "models_next", None)


def next_asr(
    cfg: Any,
    build_whisper: Callable[[], Any],
    notice: Notice | None = None,
    sample_rate: int = 16000,
    save: Callable[[], None] | None = None,
) -> Any:
    """Whisper as before, or Moonshine / Nemotron (``models_next.stt``) with Whisper as the fallback. A failure
    rolls the speech slot back (``models_next.auto_rollback``) and says so."""
    mn = opts(cfg)
    if mn is None or getattr(mn, "stt", "whisper") == "whisper":
        return build_whisper()
    from away.app import model_guard
    from away.voice.asr_next import make_next_asr

    def fell_back(reason: str) -> None:
        msg = reason
        with contextlib.suppress(Exception):
            back = model_guard.rollback(cfg, "stt", reason)
            if back:
                msg = f"{reason}. {back}"
                if save is not None:
                    save()
        if notice is not None:
            notice("Project Away: speech model", msg)

    return make_next_asr(str(mn.stt), mn, build_whisper, fell_back, sample_rate=sample_rate)


def vad_kind(cfg: Any, default: str) -> str:
    mn = opts(cfg)
    return "silero6" if mn is not None and getattr(mn, "vad", "bundled") == "silero6" else default


def partial_tap(asr: Any, ambient: Any, publish: Callable[[str, dict[str, Any]], None]) -> Callable[[Any], None] | None:
    """Stream the microphone to a streaming backend while the endpointer hears speech; the words so far go out as
    ``voice.partial``. None for Whisper (no partials)."""
    if not hasattr(asr, "push"):
        return None
    asr.on_partial = lambda text: publish("voice.partial", {"text": text})
    ep = getattr(ambient, "ep", None)
    state = {"speech": False}

    def tap(chunk: Any) -> None:
        speaking = bool(getattr(ep, "in_speech", False))
        if speaking:
            asr.push(chunk)
        elif state["speech"]:
            asr.reset()
        state["speech"] = speaking

    return tap


class GatedAmbient:
    """The always-on session behind the wake-word gate; everything but ``feed`` goes to the session itself."""

    def __init__(self, gate: Any, session: Any) -> None:
        self._gate = gate
        self._session = session

    def feed(self, chunk: Any) -> Any:
        return self._gate.feed(chunk)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


def wrap_ambient(cfg: Any, ambient: Any, notice: Notice | None = None) -> Any:
    mn = opts(cfg)
    if mn is None or getattr(mn, "wake_engine", "asr") != "livekit":
        return ambient
    from away.voice.wakeword import wrap_session

    best = str(getattr(getattr(cfg, "performance", None), "mode", "")) == "best"
    gated, reason = wrap_session(ambient, mn, preset_best=best)
    if gated is ambient:
        if reason and notice is not None:
            notice("Project Away: wake word", reason)
        return ambient
    return GatedAmbient(gated, ambient)


def hand_kwargs(cfg: Any, notice: Notice | None = None, save: Callable[[], None] | None = None) -> dict[str, Any]:
    """``factory=`` for HandTracker when RTMPose is chosen and downloaded (experimental), else nothing."""
    mn = opts(cfg)
    if mn is None or getattr(mn, "hands_landmarker", "mediapipe") != "rtmpose":
        return {}
    try:
        from away.app.model_fetch import model_dir
        from away.hands.rtmpose import MODEL_ID, find_models, tracker_factory

        find_models(model_dir(MODEL_ID))
        return {"factory": tracker_factory()}
    except Exception as exc:
        with contextlib.suppress(Exception):
            from away.app import model_guard

            if model_guard.rollback(cfg, "hands", str(exc)) and save is not None:
                save()
        if notice is not None:
            notice("Project Away: hand tracking", f"RTMPose hands unavailable ({exc}); using MediaPipe")
        return {}


def configure_voices(cfg_getter: Callable[[], Any]) -> None:
    from away.voice import tts_next

    tts_next.configure(lambda: opts(cfg_getter()))
