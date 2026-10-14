"""v0.1.24 model refresh: Silero VAD v6.2 (vad_next), the livekit wake-word gate (wakeword), Qwen3.5 cleanup
(cleanup_next) and the new voices (tts_next) - all with fakes, no models downloaded."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from away.voice import cleanup_next, tts_next, vad_next, wakeword
from away.voice.vad import Segment

SR = 16000


# ============================================================================================ Silero v6.2
class FakeSilero:
    """``run(None, feeds)`` like the ONNX graph: probability from the window's loudness; state counts calls."""

    def __init__(self):
        self.feeds: list[dict] = []

    def run(self, _outputs, feeds):
        self.feeds.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        x = feeds["input"]
        loud = float(np.mean(np.abs(x[:, -512:] if x.shape[1] > 300 else x[:, -256:])))
        prob = 0.95 if loud > 0.1 else 0.02
        return np.array([[prob]], dtype=np.float32), feeds["state"] + 1.0


def _speech(seconds_silence_before=0.5, seconds_speech=1.0, seconds_after=0.5):
    a = np.zeros(int(SR * seconds_silence_before), dtype=np.float32)
    b = (0.5 * np.sin(2 * np.pi * 200 * np.arange(int(SR * seconds_speech)) / SR)).astype(np.float32)
    c = np.zeros(int(SR * seconds_after), dtype=np.float32)
    return np.concatenate([a, b, c]), len(a), len(a) + len(b)


def test_silero6_window_feeds_context_state_and_rate():
    sess = FakeSilero()
    vad = vad_next.SileroV6VAD(session=sess)
    w1 = np.full(512, 0.5, dtype=np.float32)
    assert vad.window_prob(w1) == pytest.approx(0.95)
    f = sess.feeds[0]
    assert f["input"].shape == (1, 64 + 512) and f["state"].shape == (2, 1, 128)
    assert int(f["sr"]) == 16000 and f["sr"].dtype == np.int64
    assert np.all(f["input"][:, :64] == 0)  # no context before the first window
    vad.window_prob(np.zeros(512, dtype=np.float32))
    f2 = sess.feeds[1]
    assert np.allclose(f2["input"][:, :64], 0.5)  # the last 64 samples of the previous window
    assert np.all(f2["state"] == 1.0)  # the state the model returned is fed back
    with pytest.raises(ValueError):
        vad.window_prob(np.zeros(100, dtype=np.float32))


def test_silero6_8k_uses_smaller_windows_and_rejects_other_rates():
    sess = FakeSilero()
    vad = vad_next.SileroV6VAD(sample_rate=8000, session=sess)
    assert (vad.window, vad.context_size) == (256, 32)
    vad.window_prob(np.zeros(256, dtype=np.float32))
    assert sess.feeds[0]["input"].shape == (1, 288)
    with pytest.raises(ValueError):
        vad_next.SileroV6VAD(sample_rate=44100, session=sess)


def test_silero6_stream_keeps_the_remainder_for_the_next_chunk():
    vad = vad_next.SileroV6VAD(session=FakeSilero())
    assert vad.stream(np.zeros(700, dtype=np.float32)) == [pytest.approx(0.02)]  # 1 window, 188 left over
    assert len(vad.stream(np.zeros(400, dtype=np.float32))) == 1  # 188 + 400 = 588 -> one more window
    assert len(vad.stream(np.zeros(10, dtype=np.float32))) == 0


def test_silero6_segments_trim_and_is_speech():
    audio, start, end = _speech()
    vad = vad_next.SileroV6VAD(session=FakeSilero())
    segs = vad.segments(audio)
    assert len(segs) == 1
    assert abs(segs[0].start - start) <= 512 and abs(segs[0].end - end) <= 1024
    assert vad.is_speech(audio) and not vad.is_speech(np.zeros(SR, dtype=np.float32))
    trimmed = vad.trim(audio, pad_ms=0)
    assert 0.9 * SR <= len(trimmed) <= 1.15 * SR
    assert len(vad.trim(np.zeros(SR, dtype=np.float32))) == 0
    # probabilities() starts from a fresh state every time
    p1, p2 = vad.probabilities(audio), vad.probabilities(audio)
    assert p1 == p2 and len(p1) == -(-len(audio) // 512)


def test_segments_from_probs_hysteresis_and_minimums():
    w = 512
    probs = [0.0, 0.9, 0.9, 0.9, 0.4, 0.9, 0.0, 0.0, 0.0, 0.0]
    # 0.4 is between neg (0.35) and threshold: speech continues through it
    segs = vad_next.segments_from_probs(probs, w, 0.5, 0.35, min_speech=w, min_silence=2 * w, total=10 * w)
    assert segs == [Segment(1 * w, 6 * w)]
    # too short to count
    assert vad_next.segments_from_probs([0.9, 0.0, 0.0, 0.0], w, 0.5, 0.35, 4 * w, w, 4 * w) == []
    # speech running to the end is closed at the end of the audio
    assert vad_next.segments_from_probs([0.0, 0.9, 0.9], w, 0.5, 0.35, w, w, 3 * w - 100) == [Segment(w, 3 * w - 100)]


def test_make_silero6_falls_back_to_the_v0123_detector_until_downloaded(tmp_path):
    fallback = object()
    assert vad_next.make_silero6(data_dir=tmp_path, fallback=lambda: fallback) is fallback
    with pytest.raises(FileNotFoundError, match="not downloaded yet"):
        vad_next.make_silero6(data_dir=tmp_path)
    assert vad_next.model_path(tmp_path) == tmp_path / "models" / "vad-silero-v6.2" / "silero_vad.onnx"


def test_make_vad_silero6_kind_uses_the_old_detector_when_missing():
    from away.voice import vad

    got = vad.make_vad("silero6")
    assert not isinstance(got, vad_next.SileroV6VAD)


# ============================================================================================ wake word
class FakeWakeModel:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0
        self.lengths: list[int] = []

    def predict(self, audio):
        self.calls += 1
        self.lengths.append(len(audio))
        s = self.scores.pop(0) if self.scores else 0.0
        return {"hey_away": s}


def _chunks(seconds, size=1600):
    return [np.zeros(size, dtype=np.float32) for _ in range(int(SR * seconds) // size)]


def test_livekit_detector_runs_every_hop_on_a_two_second_window_with_refractory():
    model = FakeWakeModel([0.1] * 3 + [0.9, 0.9, 0.9] + [0.1] * 4 + [0.95])
    det = wakeword.LiveKitDetector(model=model, threshold=0.6)
    fired = []
    for i, c in enumerate(_chunks(8.0)):
        if det.feed(c):
            fired.append(i)
    # nothing runs until 2 s of audio are buffered (chunk 19); then once the 0.25 s hop has passed (every 3rd
    # 0.1 s chunk), always on exactly the last 2 s of audio
    assert set(model.lengths) == {32000}
    assert model.calls == len(range(19, 80, 3))
    # the 5th/6th high scores fall inside the 2 s refractory period; the 11th call (chunk 49) is past it
    assert fired == [28, 49]
    assert det.last_score >= 0.0
    det.reset()
    assert not det.feed(np.zeros(1600, dtype=np.float32))


def test_livekit_detector_needs_a_model_file():
    with pytest.raises(FileNotFoundError, match="no wake-word model"):
        wakeword.LiveKitDetector("")
    with pytest.raises(FileNotFoundError):
        wakeword.LiveKitDetector("/nope/hey_away.onnx")


class FakeSession:
    def __init__(self):
        self.wake_word = "hey away"
        self.wake_window = 5.0
        self._wake_until = -1e9
        self.fed: list[int] = []
        self.resets = 0
        self.now = 0.0
        self.clock = lambda: self.now
        self.ep = SimpleNamespace(in_speech=False)

    def feed(self, a):
        self.fed.append(len(a))
        return [f"heard {len(a)}"]

    def reset(self):
        self.resets += 1


class FakeDetector:
    def __init__(self):
        self.fire_next = False
        self.last_score = 0.88
        self.fed = 0

    def feed(self, a):
        self.fed += len(a)
        fire, self.fire_next = self.fire_next, False
        return fire


def test_wake_gate_keeps_speech_to_text_off_until_the_wake_word():
    sess, det = FakeSession(), FakeDetector()
    t = [0.0]
    gate = wakeword.WakeGate(sess, det, verify=False, clock=lambda: t[0])
    wakes = []
    gate.on_wake = wakes.append
    for _ in range(20):  # 2 s of room noise
        assert gate.feed(np.zeros(1600, dtype=np.float32)) == []
    assert sess.fed == [] and det.fed == 32000
    det.fire_next = True
    out = gate.feed(np.zeros(1600, dtype=np.float32))
    assert out == ["heard 8000"]  # the last 0.5 s goes to the session at once
    assert wakes == [0.88] and sess.resets == 1
    assert sess._wake_until == pytest.approx(5.0)  # the session's wake window opened
    t[0] = 1.0
    gate.feed(np.zeros(1600, dtype=np.float32))
    assert sess.fed[-1] == 1600  # awake: audio flows to speech-to-text
    t[0] = 6.0
    gate.feed(np.zeros(1600, dtype=np.float32))
    assert sess.fed[-1] == 1600 and len(sess.fed) == 2  # window closed: back to the wake-word model only
    st = gate.stats
    assert st.wakes == 1 and st.asr_seconds == pytest.approx(0.6) and st.heard_seconds == pytest.approx(2.3)
    assert st.events == ["wake 0.88"]


def test_wake_gate_verify_sends_the_wake_word_itself_to_speech_to_text():
    sess, det = FakeSession(), FakeDetector()
    t = [0.0]
    gate = wakeword.WakeGate(sess, det, verify=True, clock=lambda: t[0])
    for _ in range(30):
        gate.feed(np.zeros(1600, dtype=np.float32))
    det.fire_next = True
    out = gate.feed(np.zeros(1600, dtype=np.float32))
    assert out == ["heard 32000"]  # the last 2 s, wake word included: speech-to-text must confirm it
    assert sess._wake_until == -1e9  # the session decides (it only opens when it hears "hey away")
    assert gate.stats.events == ["wake 0.88 (verifying)"]
    t[0] = 2.0
    gate.feed(np.zeros(1600, dtype=np.float32))
    assert sess.fed[-1] == 1600  # a 3 s verification window
    t[0] = 3.5
    gate.feed(np.zeros(1600, dtype=np.float32))
    assert len(sess.fed) == 2


def test_wake_gate_stays_open_while_the_endpointer_hears_speech():
    sess, det = FakeSession(), FakeDetector()
    gate = wakeword.WakeGate(sess, det, clock=lambda: 100.0)
    sess.ep.in_speech = True
    gate.feed(np.zeros(1600, dtype=np.float32))
    assert sess.fed == [1600] and det.fed == 0


def test_wrap_session_only_with_livekit_a_wake_word_and_a_model():
    from away.config import ModelsNextConfig

    sess = FakeSession()
    assert wakeword.wrap_session(sess, ModelsNextConfig()) == (sess, "")  # default: speech-to-text listens
    on = ModelsNextConfig(wake_engine="livekit")
    got, why = wakeword.wrap_session(sess, on)
    assert got is sess and "wake-word model unavailable" in why
    nowake = FakeSession()
    nowake.wake_word = None
    assert "turn on the wake word" in wakeword.wrap_session(nowake, on, detector=FakeDetector())[1]
    gate, why = wakeword.wrap_session(sess, on, preset_best=True, detector=FakeDetector())
    assert isinstance(gate, wakeword.WakeGate) and why == "" and gate.verify is True  # auto: on in Best
    gate, _ = wakeword.wrap_session(sess, on, preset_best=False, detector=FakeDetector())
    assert gate.verify is False
    forced = ModelsNextConfig(wake_engine="livekit", wake_verify="on")
    assert wakeword.wrap_session(sess, forced, detector=FakeDetector())[0].verify is True


def test_next_wiring_wraps_the_ambient_session_only_when_chosen():
    from away.app import next_wiring
    from away.config import AppConfig

    cfg = AppConfig()
    sess = FakeSession()
    assert next_wiring.wrap_ambient(cfg, sess) is sess
    cfg.models_next.wake_engine = "livekit"
    notes = []
    assert next_wiring.wrap_ambient(cfg, sess, lambda t, m: notes.append(m)) is sess  # no model file: unchanged
    assert notes and "wake-word model unavailable" in notes[0]


def test_training_plan_writes_a_config_and_starts_nothing(tmp_path):
    assert wakeword.slug("José  Bot!") == "jos_bot" and wakeword.slug("!!!") == "away"
    cfg = wakeword.training_config("Away", size="medium", n_samples=5, steps=7)
    assert "model_name: hey_away" in cfg and "'hey away'" in cfg and "model_size: medium" in cfg
    assert "steps: 7" in cfg
    plan = wakeword.training_plan("Away", tmp_path / "train")
    assert Path(plan["config"]).read_text() == wakeword.training_config("Away")
    assert plan["commands"][0].startswith("livekit-wakeword setup --config ")
    assert plan["output"].startswith("output/hey_away/hey_away.onnx")


def test_can_train_here_explains_what_is_missing():
    ok, why = wakeword.can_train_here("win32", which=lambda n: None)
    assert not ok and "WSL" in why
    ok, why = wakeword.can_train_here("win32", which=lambda n: "wsl.exe" if n.startswith("wsl") else None)
    assert not ok and "inside WSL" in why
    ok, why = wakeword.can_train_here("linux", which=lambda n: None if n == "espeak-ng" else "/usr/bin/" + n)
    assert not ok and "espeak-ng" in why
    assert wakeword.can_train_here("linux", which=lambda n: "/usr/bin/" + n)[0]


# ============================================================================================ Qwen3.5 cleanup
@pytest.mark.parametrize(
    "raw, clean",
    [
        ("<think>\nthe user said um\n</think>\n\nMeet at 3 pm tomorrow.", "Meet at 3 pm tomorrow."),
        ("<THINK>x</THINK>Hello.", "Hello."),
        ("<think></think>Hi there.", "Hi there."),
        ("Answer.</think>", "Answer."),
        ("Start. <think>ran out of tokens while thinking", "Start."),
        ("<think>a</think>One. <think>b</think>Two.", "One. Two."),
        ("No tags at all.", "No tags at all."),
        ("", ""),
    ],
)
def test_strip_think(raw, clean):
    assert cleanup_next.strip_think(raw) == clean


class FakeLLM:
    def __init__(self, path, fail=None, out="<think>hmm</think> Clean text."):
        self.path, self.fail, self.out = path, fail, out
        self.loads = 0

    def load(self):
        self.loads += 1
        if self.fail:
            raise self.fail
        return self

    def complete(self, system, user, max_tokens, timeout):
        if self.fail:
            raise self.fail
        return self.out


def _qwen(key, fail_for=(), installed=("qwen2.5-1.5b", "qwen2.5-0.5b"), err=None):
    made = []

    def make(path):
        llm = FakeLLM(
            path,
            fail=(err or ValueError("unknown model architecture: 'qwen35'"))
            if path.name.startswith(fail_for)
            else None,
        )
        made.append(llm)
        return llm

    notes = []
    q = cleanup_next.Qwen35Fallback(
        key, Path("/m"), make, lambda k, d: d / f"{k}.gguf", lambda k, d: k in installed, notes.append
    )
    return q, made, notes


def test_qwen35_complete_strips_think():
    q, made, notes = _qwen("qwen3.5-2b")
    assert q.complete("sys", "um hi", 64, 5.0) == "Clean text."
    assert not q.fell_back and notes == [] and made[0].path == Path("/m/qwen3.5-2b.gguf")


def test_qwen35_falls_back_to_qwen25_when_llama_cpp_cannot_load_it():
    q, made, notes = _qwen("qwen3.5-2b", fail_for=("qwen3.5",))
    q.load()
    assert q.fell_back and made[-1].path == Path("/m/qwen2.5-1.5b.gguf")
    assert "cannot run Qwen3.5; using qwen2.5-1.5b" in notes[0]
    assert q.complete("s", "u", 8, 1.0) == "Clean text."
    q2, made2, _ = _qwen("qwen3.5-0.8b", fail_for=("qwen3.5",))
    assert q2.complete("s", "u", 8, 1.0) == "Clean text." and made2[-1].path.name == "qwen2.5-0.5b.gguf"


def test_qwen35_other_errors_and_missing_fallback_are_raised():
    q, _, notes = _qwen("qwen3.5-2b", fail_for=("qwen3.5",), err=TimeoutError("took too long"))
    with pytest.raises(TimeoutError):
        q.complete("s", "u", 8, 1.0)
    assert not q.fell_back and notes == []
    q2, _, _ = _qwen("qwen3.5-2b", fail_for=("qwen3.5",), installed=())
    with pytest.raises(ValueError, match="architecture"):
        q2.load()


def test_qwen35_helpers():
    assert cleanup_next.is_qwen35("Qwen3.5-2b") and not cleanup_next.is_qwen35("qwen2.5-1.5b")
    assert cleanup_next.arch_unsupported(RuntimeError("llama_model_load: unknown model architecture"))
    assert not cleanup_next.arch_unsupported(RuntimeError("out of memory"))
    assert set(cleanup_next.FALLBACK_KEYS) == set(cleanup_next.QWEN35_KEYS)


# ============================================================================================ voices
def test_kokoro_precision_and_next_voice_factories_default_empty(tmp_path):
    from away.config import ModelsNextConfig

    assert tts_next.kokoro_precision(ModelsNextConfig(), cuda=lambda: True) == "fp16"
    assert tts_next.kokoro_precision(ModelsNextConfig(), cuda=lambda: False) == "int8"
    assert tts_next.kokoro_precision(ModelsNextConfig(kokoro_precision="int8"), cuda=lambda: True) == "int8"
    settings = lambda: SimpleNamespace(engine="auto")  # noqa: E731
    # nothing chosen and nothing downloaded: the v0.1.23 chain is untouched
    assert tts_next.next_factories(settings, tmp_path, ModelsNextConfig(), cuda=lambda: False) == []
    assert tts_next.next_factories(settings, tmp_path, None) == []
    got = tts_next.next_factories(
        settings,
        tmp_path,
        ModelsNextConfig(instant_voice=True, kokoro_precision="fp16"),
        have_fp16=lambda: True,
        have_pocket=lambda: True,
    )
    assert [n for n, _ in got] == ["Instant (pocket-tts)", "Natural (Kokoro, graphics card)"]
    # Windows voice chosen: no Kokoro in front of it
    got = tts_next.next_factories(
        lambda: SimpleNamespace(engine="windows"),
        tmp_path,
        ModelsNextConfig(kokoro_precision="fp16"),
        have_fp16=lambda: True,
    )
    assert got == []


def test_pocket_engine_hands_long_text_to_the_next_voice():
    class M:
        sample_rate = 24000

        def get_state_for_audio_prompt(self, voice):
            return voice

        def generate_audio(self, state, piece):
            return np.zeros(240, dtype=np.float32)

    played = []
    eng = tts_next.PocketEngine(model=M(), play=lambda s, sr, vol: played.append(sr))
    eng.speak("Done, sent it.", SimpleNamespace(volume="normal"))
    assert played and set(played) == {24000}
    with pytest.raises(RuntimeError, match="long message"):
        eng.speak("word " * 40, SimpleNamespace(volume="normal"))
