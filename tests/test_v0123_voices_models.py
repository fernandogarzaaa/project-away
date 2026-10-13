"""v0.1.23 natural voices (h) and the model registry (j), on fake engines, a fake downloader and a fake catalog.

Nothing here downloads a voice or a model; the pinned sizes / checksums are checked against fake files built to
match, and the licence rule (GPL espeak-ng / phonemizer never bundled) is checked against the PyInstaller spec.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from away import config as C
from away.agentlink import registry as AR
from away.app import model_registry as MR
from away.app import voice_picker as VP
from away.voice import natural as N
from away.voice import speech

ROOT = Path(__file__).resolve().parents[1]


# ============================================================================ (h) text and playback
def test_chunks_start_short_and_split_long_sentences() -> None:
    text = (
        "Your build finished. It took four minutes, and every test passed, including the slow ones that "
        "talk to the camera, the microphone, and the network, which is nice! Done?"
    )
    pieces = N.chunks(text)
    assert len(pieces[0].split()) <= N.FIRST_CHUNK_WORDS
    assert all(len(p.split()) <= N.CHUNK_WORDS for p in pieces)
    assert " ".join(pieces) == " ".join(text.split())  # nothing lost or reordered
    assert N.chunks("   ") == [] and N.chunks("Hi.") == ["Hi."]


def test_play_chunks_overlaps_synthesis_and_reports_errors() -> None:
    order: list[str] = []
    lock = threading.Lock()

    def synth(p: str) -> tuple[Any, int]:
        with lock:
            order.append(f"synth {p}")
        return p, 16000

    def play(samples: Any, sr: int) -> None:
        with lock:
            order.append(f"play {samples}")
        time.sleep(0.01)

    assert N.play_chunks(synth, ["a", "b", "c"], play) == 3
    assert [o for o in order if o.startswith("play")] == ["play a", "play b", "play c"]
    assert order.index("synth b") < order.index("play b")  # the next piece is ready before it is needed

    def bad(_p: str) -> tuple[Any, int]:
        raise RuntimeError("model broke")

    with pytest.raises(RuntimeError, match="model broke"):
        N.play_chunks(bad, ["a"], play)


def test_piper_phoneme_ids() -> None:
    ids = N.piper_ids("hə", {"_": [0], "^": [1], "$": [2], "h": [20], "ə": [59]})
    assert ids == [1, 0, 20, 0, 59, 0, 2]
    assert N.piper_ids("?", {"_": [0], "^": [1], "$": [2]}) == [1, 0, 2]  # unknown phonemes are skipped


class FakeKokoro:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[list[float], int]:
        self.calls.append((text, voice, speed))
        return [0.0] * 10, 24000


def test_kokoro_engine_uses_the_chosen_voice_and_speed() -> None:
    model = FakeKokoro()
    played: list[tuple[int, float]] = []
    eng = N.KokoroEngine(voice=lambda: "bm_george", play=lambda s, sr, v: played.append((sr, v)), model=model)
    eng.speak("Hello there. All done.", speech.Voice("fast", "loud", ""))
    assert {c[1] for c in model.calls} == {"bm_george"} and {c[2] for c in model.calls} == {1.2}
    assert played and all(p == (24000, 1.0) for p in played) and eng.used_voice == "bm_george"
    eng2 = N.KokoroEngine(voice=lambda: "nonsense", play=lambda *a: None, model=model)
    eng2.speak("x", speech.Voice())
    assert eng2.used_voice == N.KOKORO_VOICES[0]  # an unknown name falls back to the default voice


def test_piper_engine_runs_the_onnx_session() -> None:
    np = pytest.importorskip("numpy")

    class Session:
        def __init__(self) -> None:
            self.feeds: list[dict[str, Any]] = []

        def run(self, _out: Any, feeds: dict[str, Any]) -> list[Any]:
            self.feeds.append(feeds)
            return [np.zeros((1, 1, 100), dtype=np.float32)]

    sess = Session()
    config = {"phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [5]}, "audio": {"sample_rate": 16000}}
    eng = N.PiperEngine(play=lambda *a: None, session=sess, config=config, phonemize=lambda t: "aa")
    samples, sr = eng.synth("hi", speed=1.25)
    assert sr == 16000 and samples.shape == (100,)
    f = sess.feeds[0]
    assert f["input"].tolist() == [[1, 0, 5, 0, 5, 0, 2]] and f["scales"][1] == pytest.approx(0.8)


def test_windows_natural_command_and_injection_guard() -> None:
    runs: list[Any] = []

    class Res:
        returncode = 0
        stderr = ""

    eng = N.WindowsNaturalEngine("powershell.exe", run=lambda cmd, **kw: runs.append((cmd, kw)) or Res())
    eng.speak("Hello 'world'", speech.Voice("slow", "quiet", "Aria"))
    cmd, kw = runs[0]
    assert cmd[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    assert "*Aria*" in cmd[4] and "SpeakingRate=0.85" in cmd[4] and "AudioVolume=0.55" in cmd[4]
    assert kw["input"] == "Hello 'world'"  # the text goes through stdin, never into the script
    evil = eng.command(speech.Voice("normal", "normal", "x'; Remove-Item C:\\ -Recurse; '"))
    assert "Remove-Item C:" not in evil[4] and "AllVoices" not in evil[4]  # an unsafe voice name is ignored
    Res.returncode = 1
    Res.stderr = "no voice"
    with pytest.raises(RuntimeError, match="Windows natural voice failed"):
        eng.speak("x", speech.Voice())


# ============================================================================ (h) engine order and fallback
class S:
    def __init__(self, engine: str = "auto") -> None:
        self.engine, self.kokoro_voice, self.piper_voice = engine, "af_heart", "lessac"


def _names(fs: list[tuple[str, Any]]) -> list[str]:
    return [n for n, _ in fs]


def test_natural_factories_order() -> None:
    def which(_exe: str) -> str:
        return "C:\\powershell.exe"

    both = {"have_kokoro": lambda: True, "have_piper": lambda: True}
    auto = N.natural_factories(lambda: S("auto"), platform="win32", which=which, **both)
    assert _names(auto) == ["Natural (Kokoro)", "Light (Piper)", "Windows natural"]
    piper = N.natural_factories(lambda: S("piper"), platform="win32", which=which, **both)
    assert _names(piper) == ["Light (Piper)", "Natural (Kokoro)", "Windows natural"]  # chosen first, then fallbacks
    win = N.natural_factories(lambda: S("windows"), platform="win32", which=which, **both)
    assert _names(win)[0] == "Windows natural"
    assert N.natural_factories(lambda: S("classic"), platform="win32", which=which, **both) == []
    none = N.natural_factories(lambda: S("kokoro"), platform="linux", which=which, have_kokoro=lambda: False)
    assert none == [] or _names(none) == ["Light (Piper)"]  # not downloaded: left out, never an error


def test_speaker_falls_back_silently_through_the_chain() -> None:
    class Eng:
        def __init__(self, name: str, fail: bool) -> None:
            self.name, self.fail, self.said = name, fail, []

        def speak(self, text: str, v: Any) -> None:
            if self.fail:
                raise RuntimeError(f"{self.name} has no audio")
            self.said.append(text)

        def close(self) -> None:
            pass

    kokoro, piper, classic = Eng("kokoro", True), Eng("piper", True), Eng("classic", False)

    def broken() -> Any:
        raise RuntimeError("not downloaded")

    sp = speech.Speaker(
        lambda: None,
        [
            ("Windows natural", broken),
            ("Natural (Kokoro)", lambda: kokoro),
            ("Light (Piper)", lambda: piper),
            ("Classic", lambda: classic),
        ],
    )
    sp.say("Build finished")
    assert sp.wait(5)
    assert classic.said == ["Build finished"] and sp.engine_name == "Classic"
    sp.say("again")
    assert sp.wait(5) and classic.said == ["Build finished", "again"]  # the failed engines are not retried
    sp.stop()


def test_voice_label_reports_the_voice() -> None:
    sp = speech.Speaker(lambda: C.SpeechConfig(kokoro_voice="bf_emma"), [("Natural (Kokoro)", lambda: None)])
    assert sp.voice_label() == "Natural (Kokoro): bf_emma"


# ============================================================================ (h) downloads (fake opener)
def _wheel(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    return buf.getvalue()


def _assets(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes], bad: str = "") -> None:
    """Replace the pinned assets with small fake files whose size and sha256 match (``bad``: one that won't)."""

    def asset(name: str) -> N.Asset:
        body = payloads[name]
        sha = hashlib.sha256(body).hexdigest() if name != bad else "0" * 64
        return N.Asset(name, "https://example.invalid/" + name, len(body), sha)

    monkeypatch.setattr(N, "PIPER_MODELS", {"lessac": [asset("v.onnx"), asset("v.onnx.json")]})
    monkeypatch.setattr(N, "WHEELS_ANY", [asset("tiny-1.0-py3-none-any.whl")])
    monkeypatch.setattr(N, "ESPEAK_WHEELS", {})


class Opener:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads, self.urls = payloads, []

    def __call__(self, req: Any, timeout: float) -> Any:
        self.urls.append(req.full_url)
        return io.BytesIO(self.payloads[req.full_url.rsplit("/", 1)[1]])


def test_download_verifies_and_unpacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payloads = {
        "v.onnx": b"onnx" * 100,
        "v.onnx.json": b"{}",
        "tiny-1.0-py3-none-any.whl": _wheel({"tiny/__init__.py": b"X = 1\n"}),
    }
    _assets(monkeypatch, payloads)
    assert not N.installed("piper", "lessac", data_dir=tmp_path)
    assert N.download_size("piper", "lessac", data_dir=tmp_path) == sum(len(b) for b in payloads.values())
    seen: list[tuple[int, int]] = []
    op = Opener(payloads)
    N.download("piper", "lessac", data_dir=tmp_path, progress=lambda d, t: seen.append((d, t)), opener=op)
    assert N.installed("piper", "lessac", data_dir=tmp_path) and N.download_size("piper", data_dir=tmp_path) == 0
    assert (N.lib_dir(tmp_path) / "tiny" / "__init__.py").exists() and seen[-1][0] == seen[-1][1]
    n = len(op.urls)
    N.download("piper", "lessac", data_dir=tmp_path, opener=op)
    assert len(op.urls) == n  # nothing fetched twice
    assert N.remove(tmp_path) and not N.installed("piper", data_dir=tmp_path) and not N.remove(tmp_path)


def test_download_with_a_bad_checksum_installs_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payloads = {"v.onnx": b"x" * 10, "v.onnx.json": b"{}", "tiny-1.0-py3-none-any.whl": _wheel({"t.py": b""})}
    _assets(monkeypatch, payloads, bad="v.onnx")
    with pytest.raises(N.VoiceDownloadError, match="checksum"):
        N.download("piper", "lessac", data_dir=tmp_path, opener=Opener(payloads))
    d = N.voices_dir(tmp_path)
    assert not (d / "v.onnx").exists() and not list(d.glob("*.part"))


def test_a_wheel_with_an_unsafe_path_is_refused(tmp_path: Path) -> None:
    wheels = N.voices_dir(tmp_path) / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "evil-1-py3-none-any.whl").write_bytes(_wheel({"../../escape.py": b"boom"}))
    with pytest.raises(N.VoiceDownloadError, match="unsafe path"):
        N.unpack_wheels(tmp_path)
    assert not (tmp_path / "escape.py").exists()


def test_pinned_assets_are_complete() -> None:
    for engine in ("kokoro", "piper"):
        for plat in ("win32", "linux", "darwin"):
            for a in N.assets_for(engine, plat=plat):
                assert a.url.startswith("https://") and a.size > 0 and len(a.sha256) == 64, a.name
    piper = [a.name for a in N.assets_for("piper", "amy", plat="win32")]
    assert "en_US-amy-medium.onnx" in piper and not any(n.startswith("kokoro_onnx") for n in piper)
    assert N.plat_key("linux", "aarch64") == "linux-aarch64" and N.plat_key("win32") == "win32"


# ============================================================================ (h) picker + licences
def test_voice_picker_labels_and_status(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    assert VP.label_for(cfg) == "Best available"
    assert VP.apply_label(cfg, "Light (Piper) - amy") == "piper" and cfg.speech.piper_voice == "amy"
    assert VP.label_for(cfg) == "Light (Piper) - amy"
    assert VP.apply_label(cfg, "Natural (Kokoro) - am_michael") == "kokoro"
    assert cfg.speech.kokoro_voice == "am_michael"
    assert VP.needs_download(cfg, tmp_path) > 50_000_000
    assert "one-time download" in VP.status_text(cfg, tmp_path)
    with pytest.raises(ValueError):
        VP.apply_label(cfg, "Robot")
    VP.apply_label(cfg, "Classic system voice")
    assert VP.needs_download(cfg, tmp_path) == 0 and VP.status_text(cfg, tmp_path) == "Ready."


def test_voice_picker_shows_the_gpl_licences_before_the_download() -> None:
    cfg = C.AppConfig()
    for label in ("Natural (Kokoro) - af_heart", "Light (Piper) - lessac"):
        VP.apply_label(cfg, label)
        text = VP.license_text(cfg)
        assert "espeak-ng" in text and "phonemizer" in text and "GPL-3.0" in text
        assert "not part of Project Away" in text
    VP.apply_label(cfg, "Windows natural")
    assert VP.license_text(cfg) == ""


def test_preview_goes_through_a_speaker() -> None:
    said: list[str] = []

    class Sp:
        def say(self, text: str, force: bool = False) -> dict[str, Any]:
            said.append(text)
            return {"spoken": True, "engine": "x"}

    cfg = C.AppConfig()
    assert VP.preview(cfg, lambda _c: Sp())["spoken"] and "voice" in said[0]


def test_gpl_voice_parts_are_never_bundled() -> None:
    spec = (ROOT / "packaging" / "pyinstaller" / "away.spec").read_text(encoding="utf-8")
    for pkg in ("espeakng_loader", "phonemizer", "piper_phonemize"):
        assert f'"{pkg}"' in spec.split("NON_PERMISSIVE = [", 1)[1].split("]", 1)[0], pkg
    assert "*NON_PERMISSIVE," in spec and "_is_non_permissive" in spec
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for pkg in ("espeak", "phonemizer", "piper", "kokoro"):
        assert pkg not in pyproject, pkg  # never a dependency, so pip / the installers never pull it in
    iss = (ROOT / "packaging" / "windows" / "away.iss").read_text(encoding="utf-8").lower()
    assert "espeak" not in iss and "phonemizer" not in iss


# ============================================================================ (j) model registry
def test_builtin_manifest_is_valid_and_modes_point_at_real_models(tmp_path: Path) -> None:
    reg = MR.Registry(tmp_path)
    assert reg.errors == []
    for mode, slots in reg.modes.items():
        assert mode in MR.MODES
        for slot, mid in slots.items():
            assert reg.get(mid).slot == slot
            assert reg.get(mid).is_permissive, f"{mode}/{slot}: {mid} is not permissively licensed"


@pytest.mark.parametrize(
    ("patch", "why"),
    [
        ({"slot": "toaster"}, "unknown slot"),
        ({"id": "../x"}, "bad model id"),
        ({"set": {}}, "'set'"),
        ({"set": {"agent_link.port": 1}}, "may not change"),
        ({"set": {"app.toggle_hotkey": "x"}}, "may not change"),
        ({"url": "http://huggingface.co/x.onnx", "sha256": "a" * 64}, "allowed https host"),
        ({"url": "https://evil.example/x.onnx", "sha256": "a" * 64}, "allowed https host"),
        ({"url": "https://huggingface.co/x.onnx"}, "sha256"),
        ({"runtime": "exe"}, "unknown runtime"),
    ],
)
def test_validate_refuses_unsafe_entries(patch: dict[str, Any], why: str) -> None:
    item = {"id": "m1", "slot": "stt", "title": "M", "set": {"voice.asr_model": "small"}, **patch}
    with pytest.raises(ValueError, match=why):
        MR.validate(item)


def test_validate_accepts_an_allowed_download() -> None:
    m = MR.validate(
        {
            "id": "whisper-x",
            "slot": "stt",
            "title": "X",
            "license": "MIT",
            "url": "https://huggingface.co/org/x/resolve/main/model.bin",
            "sha256": "b" * 64,
            "set": {"voice.asr_model": "x"},
        }
    )
    assert m.is_permissive and m.public()["size_mb"] == 0


def _catalog(tmp_path: Path, models: list[dict[str, Any]], modes: dict[str, Any] | None = None) -> None:
    folder = MR.models_dir(tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "catalog.json").write_text(json.dumps({"version": 1, "models": models, "modes": modes or {}}))


def test_catalog_and_local_models_join_the_list_and_restrictive_ones_hide(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        [
            {"id": "stt-new", "slot": "stt", "title": "New", "license": "Apache-2.0", "set": {"voice.asr_model": "n"}},
            {"id": "stt-nc", "slot": "stt", "title": "NC", "license": "CC-BY-NC-4.0", "set": {"voice.asr_model": "c"}},
            {"id": "broken", "slot": "stt"},
        ],
    )
    (MR.models_dir(tmp_path) / "local.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "my-gguf",
                        "slot": "cleanup",
                        "title": "Mine",
                        "license": "MIT",
                        "runtime": "gguf",
                        "path": "C:/m.gguf",
                        "set": {"voice.cleanup.mode": "llm"},
                    }
                ]
            }
        )
    )
    reg = MR.Registry(tmp_path)
    ids = [m.id for m in reg.for_slot("stt")]
    assert "stt-new" in ids and "stt-nc" not in ids and "stt-nc" in [m.id for m in reg.for_slot("stt", True)]
    assert reg.get("stt-new").source == "catalog" and reg.get("my-gguf").source == "your file"
    assert len(reg.errors) == 1 and reg.errors[0].startswith("catalog:")
    assert "New" in MR.new_model_hint(reg, tmp_path)
    MR.mark_seen(["stt-new"], tmp_path)
    assert MR.new_model_hint(reg, tmp_path) == ""
    cfg = C.AppConfig()
    before = reg.switch(cfg, "cleanup", "my-gguf")
    assert cfg.voice.cleanup.model == "C:/m.gguf" and "voice.cleanup.model" in before
    reg.restore(cfg, before)
    assert cfg.voice.cleanup.model == C.AppConfig().voice.cleanup.model


def test_switch_enable_mode_and_undo(tmp_path: Path) -> None:
    reg = MR.Registry(tmp_path)
    cfg = C.AppConfig()
    original = C.to_dict(cfg)
    before = reg.switch(cfg, "stt", "whisper-small-cpu")
    assert (cfg.voice.asr_model, cfg.voice.device) == ("small", "cpu")
    assert reg.current(cfg, "stt").id == "whisper-small-cpu"  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="not"):
        reg.switch(cfg, "gaze", "whisper-small-cpu")
    reg.restore(cfg, before)
    off = reg.set_enabled(cfg, "cleanup", False)
    assert cfg.voice.cleanup.mode == "off" and not reg.enabled(cfg, "cleanup")
    reg.restore(cfg, off)
    with pytest.raises(ValueError, match="cannot be turned off"):
        reg.set_enabled(cfg, "hands", False)
    undo = reg.apply_mode(cfg, "light")
    assert cfg.voice.asr_model == "base" and cfg.speech.engine == "classic"
    reg.restore(cfg, undo)
    assert C.to_dict(cfg) == original
    rows = reg.rows(cfg)
    assert [r["slot"] for r in rows] == list(MR.SLOTS) and rows[0]["choices"]


def test_try_it_keeps_a_faster_model_and_rolls_back_a_slower_or_broken_one(tmp_path: Path) -> None:
    reg = MR.Registry(tmp_path)
    cfg = C.AppConfig()
    start = (cfg.voice.asr_model, cfg.voice.device)
    speed = {"small": 10.0, "base": 20.0}

    def bench(c: Any) -> float:
        if c.voice.asr_model == "distil-large-v3":
            raise RuntimeError("CUDA not available")
        return speed.get(c.voice.asr_model, 15.0)

    slower = MR.try_model(reg, cfg, "stt", "whisper-small-cpu", bench)
    assert not slower.kept and "slower" in slower.message and (cfg.voice.asr_model, cfg.voice.device) == start
    broken = MR.try_model(reg, cfg, "stt", "whisper-distil-large-v3-gpu", bench)
    assert not broken.kept and "did not work" in broken.message and cfg.voice.asr_model == start[0]
    faster = MR.try_model(reg, cfg, "stt", "whisper-base-cpu", bench)
    assert faster.kept and cfg.voice.asr_model == "base" and faster.score_after == 20.0


class Net:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files, self.calls = files, 0

    def __call__(self, req: Any, timeout: float) -> Any:
        self.calls += 1
        return io.BytesIO(self.files[req.full_url])


def test_catalog_check_verifies_the_sha256(tmp_path: Path) -> None:
    body = json.dumps(
        {"models": [{"id": "c1", "slot": "stt", "title": "C1", "license": "MIT", "set": {"voice.asr_model": "c"}}]}
    ).encode()
    good = Net({MR.CATALOG_URL: body, MR.CATALOG_SHA_URL: hashlib.sha256(body).hexdigest().encode() + b"  x\n"})
    res = MR.check_catalog(tmp_path, opener=good, clock=lambda: 1e9)
    assert res == {"checked": True, "new": ["c1"], "error": ""}
    assert MR.Registry(tmp_path).get("c1").source == "catalog"
    assert MR.check_catalog(tmp_path, opener=good, clock=lambda: 1e9 + 60)["checked"] is False  # once a day
    again = MR.check_catalog(tmp_path, opener=good, clock=lambda: 1e9 + 2 * MR.CHECK_EVERY_S)
    assert again["new"] == []  # nothing new the second time

    tampered = Net({MR.CATALOG_URL: body + b" ", MR.CATALOG_SHA_URL: hashlib.sha256(body).hexdigest().encode()})
    res = MR.check_catalog(tmp_path / "t", opener=tampered, force=True)
    assert "checksum" in res["error"] and not (MR.models_dir(tmp_path / "t") / "catalog.json").exists()

    bad = json.dumps({"models": [{"id": "x", "slot": "stt", "set": {"agent_link.port": 1}}]}).encode()
    evil = Net({MR.CATALOG_URL: bad, MR.CATALOG_SHA_URL: hashlib.sha256(bad).hexdigest().encode()})
    res = MR.check_catalog(tmp_path / "e", opener=evil, force=True)
    assert "bad entry" in res["error"] and not (MR.models_dir(tmp_path / "e") / "catalog.json").exists()

    def offline(req: Any, timeout: float) -> Any:
        raise OSError("no network")

    assert "could not reach" in MR.check_catalog(tmp_path / "o", opener=offline, force=True)["error"]


def test_the_repository_catalog_matches_its_sha256_file() -> None:
    body = (ROOT / "models" / "catalog.json").read_bytes()
    want = (ROOT / "models" / "catalog.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(body).hexdigest() == want, "update models/catalog.sha256 (see README > Models)"
    for item in json.loads(body).get("models") or []:
        MR.validate(item)


def test_models_agent_op_is_undoable() -> None:
    cfg = C.AppConfig()
    a = AR.models_apply(cfg, AR.Op("models", "switch", "stt", "whisper-small-cpu"))
    assert (
        a.low_risk and cfg.voice.asr_model == "small" and a.text == "Speech to text: now Whisper small on the processor"
    )
    AR.models_apply(cfg, a.inverse[0])
    assert cfg.voice.asr_model == C.AppConfig().voice.asr_model
    m = AR.models_apply(cfg, AR.Op("models", "mode", "light"))
    assert cfg.speech.engine == "classic" and m.text == "Performance mode: light"
    AR.models_apply(cfg, m.inverse[0])
    assert C.to_dict(cfg) == C.to_dict(C.AppConfig())
    d = AR.models_apply(cfg, AR.Op("models", "disable", "face"))
    assert cfg.face.enabled is False
    AR.models_apply(cfg, d.inverse[0])
    assert cfg.face.enabled is C.AppConfig().face.enabled
    with pytest.raises(AR.ChangeError):
        AR.models_apply(cfg, AR.Op("models", "switch", "stt", "no-such-model"))
    with pytest.raises(AR.ChangeError):
        AR.models_apply(cfg, AR.Op("models", "explode", "stt"))
    assert "models" in AR.domains()


# ============================================================================ Settings > Models tab
def test_models_tab_logic(tmp_path: Path) -> None:
    from away.app import models_tab as MT

    reg = MR.Registry(tmp_path)
    cfg = C.AppConfig()
    rows = MT.rows(reg, cfg)
    assert [r["slot"] for r in rows] == list(MR.SLOTS)
    stt = next(r for r in rows if r["slot"] == "stt")
    assert "Whisper small on the processor" in stt["labels"]
    gaze = next(r for r in rows if r["slot"] == "gaze")
    assert "L2CS (best gaze)" not in gaze["labels"]  # research-only weights are not offered
    assert MT.choose(reg, cfg, "stt", "Whisper small on the processor").startswith("Speech to text: Whisper small")
    assert cfg.voice.asr_model == "small"
    with pytest.raises(ValueError):
        MT.choose(reg, cfg, "stt", "L2CS (best gaze)")
    assert "off" in MT.toggle(reg, cfg, "face", False) and cfg.face.enabled is False
    assert MT.apply_mode(reg, cfg, "Light").startswith("Performance mode: Light") and cfg.voice.asr_model == "base"
    assert MT.voice_engine(reg.get("voice-kokoro")) == "kokoro" and MT.voice_engine(reg.get("voice-classic")) == ""
    assert MT.download_hint(reg.get("cleanup-qwen-0.5b")).startswith("Download: Settings")
    assert MT.download_hint(reg.get("mediapipe-hands")) == ""
    fresh = next(r for r in MT.rows(reg, C.AppConfig()) if r["slot"] == "stt")
    assert fresh["label"] == MT.AUTO_LABEL and fresh["labels"][0] == MT.AUTO_LABEL  # the defaults: automatic
    assert next(r for r in MT.rows(reg, C.AppConfig()) if r["slot"] == "voice")["label"] == "Best available"
    cfg.voice.asr_model = "tiny.en"  # a setting no listed model matches
    stt = next(r for r in MT.rows(reg, cfg) if r["slot"] == "stt")
    assert stt["label"] == "custom" and stt["labels"][0] == "custom"


def test_models_tab_builds_and_reacts_under_a_display(tmp_path: Path) -> None:
    tk = pytest.importorskip("tkinter")
    from tkinter import ttk

    from away.app import models_tab as MT

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    try:
        saved: list[int] = []
        cfg = C.AppConfig()
        win = type(
            "Win",
            (),
            {
                "tk": tk,
                "ttk": ttk,
                "m": type("M", (), {"cfg": cfg, "save": lambda self: saved.append(1)})(),
                "_ui": type("U", (), {"post": lambda self, fn: fn()})(),
            },
        )()
        nb = ttk.Notebook(root)
        nb.pack()
        frame = MT.build(win, nb, data_dir=tmp_path)
        root.update()
        assert [nb.tab(t, "text") for t in nb.tabs()] == ["Models"]
        combos = [w for w in frame.winfo_children()[1].winfo_children() if isinstance(w, ttk.Combobox)]
        assert len(combos) == len(MR.SLOTS)
        mode = next(w for w in frame.winfo_children()[0].winfo_children() if isinstance(w, ttk.Combobox))
        mode.set("Light")
        mode.event_generate("<<ComboboxSelected>>")
        root.update()
        assert saved and cfg.voice.asr_model == "base"
        stt = combos[MR.SLOTS.index("stt")]
        assert stt.get() == "Whisper base on the processor (lightest)"  # the rows follow the mode
    finally:
        root.destroy()
