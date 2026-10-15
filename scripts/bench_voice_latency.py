"""Where the time goes between "...what's the weather?" and the first sound of the answer (v0.1.26).

Runs the app's real voice path - the always-on endpointer, the "Hey <name>" routing (TalkHook), the live agent host
and turn loop, the sentence splitter and the Speaker - with fakes for the slow outside parts (speech-to-text, the
routing call, the model, the voice engine), and prints the timing log (:mod:`away.voice.latency`) for:

* ``before`` - the four fixes switched off, which is how v0.1.25 behaved for these stages: the 700 ms end-of-speech
  pause for everything, a routing call (Jev / a model) before every answer, the voice engine loaded again for an
  answer (v0.1.25 unloaded it 30 s after the last sentence), and the providers rebuilt (keys read from the OS vault)
  for every model call;
* ``after`` - this version's defaults: a 450 ms pause while you talk to an agent, plain questions go straight to the
  answer model, the voice is loaded while you speak and kept loaded, the providers are reused.

The fake costs (seconds) are in :data:`COSTS`; ``--scale`` shrinks them all (the tests run it at 0.2). Real
microphone numbers are different on every PC: run ``away`` with the live agent and read the ``latency:`` lines in
the log instead.

    python scripts/bench_voice_latency.py            # both, 3 turns each, medians
    python scripts/bench_voice_latency.py --scale 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

COSTS = {
    "stt": 0.25,  # speech-to-text of a short question
    "route_call": 0.9,  # asking Jev / a model "what kind of request is this?"
    "funnel_build": 0.08,  # building the providers (reading keys from the OS vault)
    "first_token": 0.6,  # the model's first word
    "token_gap": 0.03,  # each following piece
    "engine_load": 1.2,  # loading the natural voice (Kokoro on the processor)
    "synth_first": 0.25,  # synthesizing the first short piece
}
REPLY = "It is sunny and twenty two degrees in Manila right now. Tonight it cools down to about twenty five. Enjoy!"
SR = 16000


def _speech_then_silence(seconds: float = 1.0, silence: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 0.001, int(SR * 0.6)).astype(np.float32)
    talk = rng.normal(0, 0.12, int(SR * seconds)).astype(np.float32)
    tail = rng.normal(0, 0.001, int(SR * silence)).astype(np.float32)
    return np.concatenate([quiet, talk, tail])


class FakeASR:
    def __init__(self, text: str, cost: float) -> None:
        self.text, self.cost = text, cost
        self.last_confidence = 0.95

    def transcribe(self, audio: Any, prompt: Any = None) -> str:
        time.sleep(self.cost)
        return self.text


class FakeFunnel:
    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.router = SimpleNamespace(candidates=lambda task: [])

    def stream(self, task: str, messages: Any, rec: Any = None, **_k: Any) -> Any:
        time.sleep(COSTS["first_token"] * self.scale)
        words = REPLY.split(" ")
        for i, w in enumerate(words):
            if i:
                time.sleep(COSTS["token_gap"] * self.scale)
            yield w + (" " if i < len(words) - 1 else "")

    def complete(self, *a: Any, **k: Any) -> str:
        return '{"tool": "say", "text": "ok"}'


class SlowDecider:
    def __init__(self, cost: float) -> None:
        self.cost = cost
        self.calls = 0

    def route(self, text: str) -> Any:
        self.calls += 1
        time.sleep(self.cost)
        return SimpleNamespace(value="chat", source="jev")


class FakeEngine:
    name = "fake natural voice"

    def __init__(self, scale: float) -> None:
        time.sleep(COSTS["engine_load"] * scale)
        self.scale = scale

    def speak(self, text: str, v: Any) -> None:
        from away.voice import natural

        natural.play_chunks(
            lambda p: (time.sleep(COSTS["synth_first"] * self.scale), (np.zeros(10, dtype=np.float32), 24000))[1],
            natural.chunks(text),
            lambda s, sr: None,
        )

    def close(self) -> None:
        pass


def run(mode: str = "after", turns: int = 3, scale: float = 1.0) -> list[dict[str, float]]:
    """``turns`` questions through the real path; the timing log of each."""
    from away.liveagent import settings as st
    from away.liveagent.integrate import LiveAgentHost
    from away.voice import latency
    from away.voice.always_on import AmbientSession, StreamingEndpointer
    from away.voice.output import ListSink
    from away.voice.speech import EchoGuard, Speaker
    from away.voice.talk import TalkHook

    before = mode == "before"
    latency.reset()
    tmp = Path(tempfile.mkdtemp(prefix="away-bench-"))
    talk = TalkHook(window_s=8.0, conversational=lambda: False)
    talk.agents = list
    talk.deliver = lambda t, x, s: ""
    app = SimpleNamespace(
        talk=talk,
        bus=SimpleNamespace(publish=lambda *a, **k: None),
        agent_link=None,
        voice=None,
        cfg=SimpleNamespace(speech=None, voice=SimpleNamespace(sample_rate=SR)),
    )
    host = LiveAgentHost(app, tmp)
    st.save(
        st.LiveSettings(
            enabled=True,
            name="Cookie",
            quick_route=not before,
            jev="off",
            providers=[st.ProviderConfig("groq", "groq", "m")],
        ),
        host.watcher.path,
    )
    host.watcher.poll()
    decider = SlowDecider(COSTS["route_call"] * scale)

    def build_funnel(_s: Any = None) -> FakeFunnel:
        time.sleep(COSTS["funnel_build"] * scale)
        return FakeFunnel(scale)

    host.runtime.funnel = build_funnel  # type: ignore[method-assign]
    host.decider = lambda: decider  # type: ignore[method-assign]
    host.agent.decider = lambda: decider
    if before:
        host.agent.funnel = lambda: build_funnel()  # rebuilt for every model call

    echo = EchoGuard(tail_s=0.0)

    def new_speaker() -> Speaker:
        sp = Speaker(lambda: None, factories=[("fake", lambda: FakeEngine(scale))], echo=echo)
        if before:
            sp.warm = lambda: False  # type: ignore[method-assign]
        return sp

    host.speech.speaker = new_speaker()
    host.hook_talk()
    rows: list[dict[str, float]] = []
    for _ in range(turns):
        if before:
            host.speech.speaker = new_speaker()  # unloaded since the last answer (v0.1.25: after 30 s idle)
        ep = StreamingEndpointer(SR, end_silence_ms=700, quick_end_silence_ms=700 if before else 450)
        amb = AmbientSession(
            FakeASR("Hey Cookie, what's the weather like?", COSTS["stt"] * scale),
            SimpleNamespace(clean=lambda t, app=None: t),  # type: ignore[arg-type]
            ListSink(),
            ep,
            commands=dict,
            run_command=lambda a, p: None,
            talk=talk,
        )
        amb._wake_until = amb.clock() + 5.0  # the wake-word model heard "Hey Cookie" (the quick pause applies)
        n0 = len(latency.LOG)
        audio = _speech_then_silence()
        for i in range(0, len(audio), 480):
            amb.feed(audio[i : i + 480])
        deadline = time.monotonic() + 30 * max(scale, 0.2)
        while len(latency.LOG) == n0 and time.monotonic() < deadline:
            time.sleep(0.005)
        if len(latency.LOG) > n0:
            rows.append(dict(latency.LOG[-1]))
        host.speech.speaker.wait(10)
        w = host.agent._worker
        if w is not None:
            w.join(5)
    host.stop()
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from away.voice import latency

    out = {m: latency.summary(run(m, a.turns, a.scale)) for m in ("before", "after")}
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    stages = (*latency.STAGES, "total")
    print(f"{'stage':<16}{'before ms':>11}{'after ms':>11}")
    for k in stages:
        b, f = out["before"].get(k), out["after"].get(k)
        print(f"{k:<16}{b if b is not None else '-':>11}{f if f is not None else '-':>11}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
