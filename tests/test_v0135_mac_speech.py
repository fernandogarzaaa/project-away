"""v0.1.35: Whisper on the Apple Silicon GPU (mlx-whisper): backend choice, fallback to faster-whisper on the CPU,
config, first-run downloads and the hardware text. Fakes only: no real model, no MLX, no network."""

from __future__ import annotations

import io
import json
import logging
import sys
import threading
import types
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app import fastdl, frozen, hardware, models, performance, tuning
from away.voice import asr as asrmod
from away.voice import asr_mlx
from away.voice.asr import FakeASR, FallbackASR, make_asr

AUDIO = np.zeros(16000, dtype=np.float32)


def apple_hw() -> hardware.Hardware:
    return hardware.Hardware(
        os="darwin", arch="arm64", cpu_name="Apple M5", cores_physical=10, ram_mb=24576, disk_free_mb=100_000
    )


@pytest.fixture
def on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> hardware.Hardware:
    """An M-series Mac with mlx-whisper installed (both faked)."""
    hw = apple_hw()
    monkeypatch.setattr(hardware, "cached", lambda: hw)
    monkeypatch.setattr(asr_mlx, "mlx_installed", lambda: True)
    return hw


class CPUFactory:
    """Stands in for FasterWhisperASR: records how it was built."""

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.built: list[tuple[Any, ...]] = []
        self.outputs = outputs or ["typed on the cpu"]

    def __call__(self, *args: Any) -> FakeASR:
        self.built.append(args)
        a = FakeASR(list(self.outputs))
        a.language = args[3]  # type: ignore[attr-defined]
        return a


class FakeMLX:
    name = "mlx-whisper"
    device = "metal"
    compute_type = "float16"

    def __init__(self, text: str = "typed on the apple gpu", fail: BaseException | None = None) -> None:
        self.text, self.fail, self.calls, self.language = text, fail, 0, None
        self.last_confidence = 0.9

    def transcribe(self, audio: Any, prompt: str | None = None) -> str:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return self.text


# ---------------------------------------------------------------- backend choice
def test_auto_backend_is_mlx_only_on_apple_silicon_with_mlx_whisper_installed() -> None:
    r = asr_mlx.resolve_backend
    assert r("auto", "darwin", "arm64", installed=True) == "mlx-whisper"
    assert r("auto", "darwin", "arm64", installed=False) == "faster-whisper"
    assert r("auto", "darwin", "x86_64", installed=True) == "faster-whisper"  # Intel Mac: no MLX
    assert r("auto", "linux", "aarch64", installed=True) == "faster-whisper"
    assert r("auto", "win32", "AMD64", installed=True) == "faster-whisper"
    assert r("", "darwin", "arm64", installed=True) == "mlx-whisper"


def test_explicit_backend_is_kept_whatever_the_machine() -> None:
    for b in ("faster-whisper", "mlx-whisper", "parakeet"):
        assert asr_mlx.resolve_backend(b, "linux", "x86_64", installed=False) == b
        assert asr_mlx.resolve_backend(b, "darwin", "arm64", installed=True) == b


def test_apple_silicon_follows_the_detected_hardware(on_apple_silicon: hardware.Hardware) -> None:
    assert asr_mlx.apple_silicon() is True
    assert asr_mlx.apple_silicon("linux", "x86_64") is False


def test_tests_never_see_apple_silicon_by_default() -> None:
    # tests/conftest.py pins a Linux machine: the real MLX can never start from an unrelated test
    assert asr_mlx.apple_silicon() is False
    assert asr_mlx.resolve_backend("auto", installed=True) == "faster-whisper"


def test_make_asr_auto_on_apple_silicon_builds_the_mlx_backend(on_apple_silicon: hardware.Hardware) -> None:
    cpu, got = CPUFactory(), []

    def mlx_factory(model: str, language: str | None, root: str | None, beam: int) -> FakeMLX:
        got.append((model, language, root, beam))
        return FakeMLX()

    asr = make_asr("auto", "small", "cpu", "int8", "en", "/m/whisper-small", factory=cpu, mlx_factory=mlx_factory)
    assert isinstance(asr, FallbackASR) and asr.name == "mlx-whisper" and asr.device == "metal"
    assert asr.compute_type == "float16"
    assert got == [("small", "en", "/m/whisper-small", 5)]
    assert asr.transcribe(AUDIO) == "typed on the apple gpu"
    assert asr.last_confidence == 0.9
    assert cpu.built == []  # faster-whisper is not even loaded while MLX works


def test_make_asr_auto_off_apple_silicon_never_touches_mlx() -> None:
    cpu = CPUFactory()

    def mlx_factory(*a: Any) -> FakeMLX:
        raise AssertionError("MLX must not be built on Linux/Windows")

    asr = make_asr("auto", "small", "cpu", "int8", None, factory=cpu, mlx_factory=mlx_factory)
    assert isinstance(asr, FakeASR) and len(cpu.built) == 1


def test_make_asr_faster_whisper_explicit_on_a_mac_stays_on_the_cpu(on_apple_silicon: hardware.Hardware) -> None:
    cpu = CPUFactory()
    asr = make_asr("faster-whisper", "small", "cpu", "int8", factory=cpu, mlx_factory=lambda *a: FakeMLX())
    assert isinstance(asr, FakeASR) and cpu.built[0][:3] == ("small", "cpu", "int8")


def test_mlx_failing_to_load_falls_back_to_faster_whisper_with_a_log_line(
    on_apple_silicon: hardware.Hardware, caplog: pytest.LogCaptureFixture
) -> None:
    cpu, reasons = CPUFactory(), []

    def broken(*a: Any) -> FakeMLX:
        raise RuntimeError("[metal::Device] Unable to load the default metallib")

    with caplog.at_level(logging.WARNING, logger="away.voice.asr"):
        asr = make_asr(
            "auto",
            "small",
            "cpu",
            "int8",
            "en",
            "/m/whisper-small",
            factory=cpu,
            mlx_factory=broken,
            on_fallback=reasons.append,
        )
    assert isinstance(asr, FakeASR) and asr.transcribe(AUDIO) == "typed on the cpu"
    assert cpu.built[0][:5] == ("small", "cpu", "int8", "en", "/m/whisper-small")
    assert len(reasons) == 1 and "Apple GPU" in reasons[0] and "metallib" in reasons[0] and "CPU" in reasons[0]
    assert any("Apple GPU speech recognition unavailable" in r.getMessage() for r in caplog.records)


def test_mlx_import_error_falls_back_too(on_apple_silicon: hardware.Hardware) -> None:
    cpu, reasons = CPUFactory(), []

    def missing(*a: Any) -> FakeMLX:
        raise ImportError("No module named 'mlx'")

    asr = make_asr("mlx-whisper", "small", "cpu", "int8", factory=cpu, mlx_factory=missing, on_fallback=reasons.append)
    assert isinstance(asr, FakeASR)
    assert "mlx-whisper is not installed" in reasons[0]


def test_explicit_mlx_whisper_off_apple_silicon_falls_back_and_says_why() -> None:
    cpu, reasons = CPUFactory(), []
    asr = make_asr("mlx-whisper", "base", "cpu", "int8", factory=cpu, on_fallback=reasons.append)
    assert isinstance(asr, FakeASR) and cpu.built[0][0] == "base"
    assert "Apple Silicon" in reasons[0]


def test_mlx_error_during_a_transcription_switches_to_the_cpu_once(on_apple_silicon: hardware.Hardware) -> None:
    cpu, reasons = CPUFactory(["first", "second"]), []
    gpu = FakeMLX(fail=RuntimeError("[METAL] Command buffer execution failed"))
    asr = make_asr(
        "auto", "small", "cpu", "int8", None, factory=cpu, mlx_factory=lambda *a: gpu, on_fallback=reasons.append
    )
    asr.language = "es"  # a language picked after start-up must survive the switch
    assert asr.transcribe(AUDIO) == "first"  # the utterance is not lost
    assert asr.transcribe(AUDIO) == "second"
    assert gpu.calls == 1 and len(cpu.built) == 1 and asr.device == "cpu"
    assert asr.language == "es" and cpu.built[0][1:3] == ("cpu", "int8")
    assert len(reasons) == 1 and "MLX failed" in reasons[0]


def test_cuda_fallback_is_unchanged_and_language_now_reaches_the_active_backend() -> None:
    # FallbackASR keeps its CUDA defaults: only CUDA errors switch
    gpu = FakeMLX(fail=ValueError("bad audio"))
    f = FallbackASR(gpu, lambda: FakeASR(["x"]), None, "small")
    assert f.device == "cuda"
    with pytest.raises(ValueError):
        f.transcribe(AUDIO)
    f.language = "fr"
    assert gpu.language == "fr" and f.language == "fr"


def test_set_asr_language_reaches_the_mlx_backend_through_the_wrapper(on_apple_silicon: hardware.Hardware) -> None:
    from away.app.runner import set_asr_language

    gpu = FakeMLX()
    asr = make_asr("auto", "small", "cpu", "int8", None, factory=CPUFactory(), mlx_factory=lambda *a: gpu)
    set_asr_language(asr, "de")
    assert gpu.language == "de"


# ---------------------------------------------------------------- the MLX backend itself (fake mlx_whisper)
class FakeMLXWhisperAPI:
    """``mlx_whisper.transcribe`` stand-in: returns its dict shape."""

    def __init__(self, segments: list[dict[str, Any]], language: str = "en") -> None:
        self.segments, self.language, self.calls = segments, language, []

    def transcribe(self, audio: Any, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {
            "text": "".join(s["text"] for s in self.segments),
            "segments": self.segments,
            "language": kw.get("language") or self.language,
        }


SEGS = [
    {
        "text": " Hello world.",
        "start": 0.0,
        "end": 1.0,
        "avg_logprob": -0.1,
        "no_speech_prob": 0.01,
        "compression_ratio": 1.2,
    },
    {
        "text": " Thanks for watching!",
        "start": 1.0,
        "end": 2.0,
        "avg_logprob": -1.5,
        "no_speech_prob": 0.9,
        "compression_ratio": 1.0,
    },  # Whisper guessing over silence: dropped like with faster-whisper
]


def test_adapter_answers_like_faster_whisper_so_whisper_transcribe_works_unchanged() -> None:
    api = FakeMLXWhisperAPI(SEGS)
    adapter = asr_mlx.MLXModelAdapter("/m/whisper-mlx-small", api=api)
    stats: dict[str, object] = {}
    text = asrmod.whisper_transcribe(adapter, AUDIO, "en", "Vocabulary: Inan.", "Inan", 5, stats)
    assert text == "Hello world."
    kw = api.calls[0]
    assert kw["path_or_hf_repo"] == "/m/whisper-mlx-small" and kw["language"] == "en"
    assert kw["initial_prompt"] == "Vocabulary: Inan." and kw["condition_on_previous_text"] is False
    assert "beam_size" not in kw and "hotwords" not in kw and "vad_filter" not in kw  # mlx-whisper has none
    assert kw["fp16"] is True
    assert stats["language"] == "en" and isinstance(stats["confidence"], float)


def test_adapter_reports_the_detected_language_probability_for_the_safer_auto() -> None:
    class Detecting(asr_mlx.MLXModelAdapter):
        def detect_language(self, audio: Any) -> tuple[str | None, float | None]:
            return "es", 0.42

    api = FakeMLXWhisperAPI(SEGS[:1])
    segs, info = Detecting("/m", api=api).transcribe(AUDIO)
    assert info.language == "es" and info.language_probability == 0.42
    assert api.calls[0]["language"] == "es"
    assert segs[0].text == " Hello world." and segs[0].avg_logprob == -0.1


def test_adapter_language_detection_failure_leaves_it_to_mlx_whisper() -> None:
    class Broken(asr_mlx.MLXModelAdapter):
        def detect_language(self, audio: Any) -> tuple[str | None, float | None]:
            raise RuntimeError("no GPU")

    api = FakeMLXWhisperAPI(SEGS[:1], language="pt")
    _, info = Broken("/m", api=api).transcribe(AUDIO)
    assert api.calls[0]["language"] is None and info.language == "pt" and info.language_probability is None


class FakeAdapter:
    def __init__(self) -> None:
        self.threads: list[str] = []
        self.loaded = 0

    def load(self) -> None:
        self.loaded += 1
        self.threads.append(threading.current_thread().name)

    def transcribe(self, audio: Any, **kw: Any) -> tuple[list[Any], Any]:
        self.threads.append(threading.current_thread().name)
        seg = types.SimpleNamespace(text=" Ship it.", avg_logprob=-0.2, no_speech_prob=0.0)
        return [seg], types.SimpleNamespace(language=kw.get("language") or "en", language_probability=0.99)


def test_mlx_backend_warms_up_and_runs_every_call_on_its_own_thread() -> None:
    fake = FakeAdapter()
    asr = asr_mlx.MLXWhisperASR("/m/whisper-mlx-small", "en", adapter=fake)
    assert fake.loaded == 1 and len(fake.threads) == 2  # load + a silent warm-up transcription
    out: list[str] = []
    t = threading.Thread(target=lambda: out.append(asr.transcribe(AUDIO, "Vocabulary: Inan.")), name="asr")
    t.start()
    t.join()
    assert out == ["Ship it."] and asr.last_confidence is not None
    assert {n.split("_")[0] for n in fake.threads} == {"away-mlx"}
    assert (asr.name, asr.device, asr.compute_type) == ("mlx-whisper", "metal", "float16")


def test_numba_stand_in_when_the_bundle_leaves_numba_out(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_numba(name: str, *a: Any, **k: Any) -> Any:
        if name == "numba":
            raise ImportError("excluded from the app bundle")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "numba", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_numba)
    asr_mlx.ensure_numba()
    stub = sys.modules["numba"]
    monkeypatch.delitem(sys.modules, "numba")

    def f(x: int) -> int:
        return x + 1

    assert stub.jit(nopython=True)(f) is f and stub.jit(f) is f
    assert stub.jit(nopython=True, parallel=True)(f)(1) == 2


def test_tqdm_gets_a_thread_lock_so_the_frozen_app_never_spawns_a_resource_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tqdm = pytest.importorskip("tqdm")
    monkeypatch.setattr(tqdm.tqdm, "_lock", None, raising=False)
    asr_mlx.quiet_tqdm()
    assert type(tqdm.tqdm._lock) is type(threading.RLock())
    lock = tqdm.tqdm._lock
    asr_mlx.quiet_tqdm()  # an existing lock is left alone
    assert tqdm.tqdm._lock is lock


# ---------------------------------------------------------------- config
def test_asr_backend_defaults_to_auto_and_accepts_mlx_whisper() -> None:
    cfg = C.AppConfig()
    assert cfg.voice.asr_backend == "auto"
    for b in ("auto", "faster-whisper", "mlx-whisper", "parakeet"):
        cfg.voice.asr_backend = b
        C.validate(cfg)
    assert C.ASR_BACKENDS == asr_mlx.ASR_BACKENDS


def test_asr_backend_rejects_unknown_values_with_the_choices() -> None:
    with pytest.raises(ValueError, match=r"auto\|faster-whisper\|mlx-whisper\|parakeet"):
        C.from_dict({"config_version": C.CONFIG_VERSION, "voice": {"asr_backend": "metal"}})


def test_old_written_faster_whisper_default_migrates_to_auto() -> None:
    cfg, _ = C.from_dict({"config_version": 4, "voice": {"asr_backend": "faster-whisper"}})
    assert cfg.voice.asr_backend == "auto" and cfg.config_version == C.CONFIG_VERSION == 5
    assert any("asr_backend" in n for n in cfg.migrations)
    cfg, _ = C.from_dict({"voice": {"asr_backend": "faster-whisper"}})  # v0.1.5-era file, no version
    assert cfg.voice.asr_backend == "auto"


def test_faster_whisper_chosen_from_v0135_on_and_parakeet_are_kept() -> None:
    cfg, _ = C.from_dict({"config_version": 5, "voice": {"asr_backend": "faster-whisper"}})
    assert cfg.voice.asr_backend == "faster-whisper" and not cfg.migrations
    cfg, _ = C.from_dict({"config_version": 4, "voice": {"asr_backend": "parakeet"}})
    assert cfg.voice.asr_backend == "parakeet" and not any("asr_backend" in n for n in cfg.migrations)


def test_cpu_mode_or_cpu_device_keeps_auto_off_the_apple_gpu(on_apple_silicon: hardware.Hardware) -> None:
    cfg = C.AppConfig()
    assert performance.asr_backend(cfg) == "auto" and performance.apple_gpu(cfg)
    cfg.performance.mode = "cpu"
    assert performance.asr_backend(cfg) == "faster-whisper" and not performance.apple_gpu(cfg)
    cfg.performance.mode, cfg.voice.device = "auto", "cpu"
    assert performance.asr_backend(cfg) == "faster-whisper" and not performance.apple_gpu(cfg)
    cfg.voice.device, cfg.voice.asr_backend = "auto", "faster-whisper"
    assert not performance.apple_gpu(cfg)


# ---------------------------------------------------------------- hardware text
def test_profile_on_apple_silicon_with_mlx_says_apple_gpu_not_cpu() -> None:
    p = tuning.choose_profile(apple_hw(), apple_gpu=True)
    assert p.asr_backend == "mlx-whisper"
    assert p.summary() == f"{p.asr_model} on the Apple GPU (MLX, float16)"
    text = " ".join(p.reasons)
    assert "Apple GPU" in text and "Whisper runs on the CPU" not in text
    lines = "\n".join(performance.describe(apple_hw(), p))
    assert "engine=mlx-whisper" in lines and "CPU fallback" in lines


def test_profile_on_apple_silicon_without_mlx_still_says_cpu_and_how_to_fix_it() -> None:
    p = tuning.choose_profile(apple_hw(), apple_gpu=False)
    assert p.asr_backend == "faster-whisper" and "on CPU" in p.summary()
    text = " ".join(p.reasons)
    assert "Whisper runs on the CPU" in text and "mlx-whisper" in text


def test_apple_gpu_flag_is_ignored_off_apple_silicon() -> None:
    hw = hardware.Hardware(os="linux", arch="x86_64", cores_physical=8, ram_mb=16384)
    assert tuning.choose_profile(hw, apple_gpu=True).asr_backend == "faster-whisper"


def test_current_profile_picks_up_mlx_on_an_m_series_mac(on_apple_silicon: hardware.Hardware) -> None:
    p = performance.current_profile(C.AppConfig(), on_apple_silicon)
    assert p.asr_backend == "mlx-whisper" and p.device == "cpu"  # CPU = the faster-whisper fallback


# ---------------------------------------------------------------- first-run downloads
def test_mlx_models_cover_the_profile_picks_and_are_pinned() -> None:
    for m in tuning.DOWNSIZE:
        spec = models.MLX_MODELS[m]
        assert spec.repo.startswith("mlx-community/") and len(spec.revision) == 40
        assert len(spec.sha256) == 64 and spec.weights in ("weights.npz", "weights.safetensors")


def test_missing_models_asks_for_the_mlx_weights_on_apple_silicon(
    on_apple_silicon: hardware.Hardware, tmp_path: Path
) -> None:
    cfg = C.AppConfig()
    cfg.voice.asr_model = "small"
    missing = frozen.missing_models(cfg, tmp_path)
    assert "whisper-mlx-small" in missing and "whisper-small" not in missing
    assert frozen.setup_download_mb(cfg, missing) >= models.MLX_MODELS["small"].size_mb
    d = models.mlx_dir(tmp_path, "small")
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert "whisper-mlx-small" in frozen.missing_models(cfg, tmp_path)  # weights still missing
    (d / "weights.npz").write_bytes(b"x")
    assert not any(m.startswith("whisper") for m in frozen.missing_models(cfg, tmp_path))


def test_missing_models_keeps_the_ctranslate2_weights_elsewhere_or_when_forced(
    on_apple_silicon: hardware.Hardware, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = C.AppConfig()
    cfg.voice.asr_model = "small"
    cfg.voice.asr_backend = "faster-whisper"
    assert "whisper-small" in frozen.missing_models(cfg, tmp_path)
    cfg.voice.asr_backend, cfg.performance.mode = "auto", "cpu"
    assert "whisper-small" in frozen.missing_models(cfg, tmp_path)
    cfg.performance.mode = "auto"
    monkeypatch.setattr(asr_mlx, "mlx_installed", lambda: False)  # pip install without the [mlx] extra
    assert "whisper-small" in frozen.missing_models(cfg, tmp_path)


def test_setup_backend_needs_an_mlx_build_of_the_model(on_apple_silicon: hardware.Hardware) -> None:
    assert asr_mlx.setup_backend("auto", "small") == "mlx-whisper"
    assert asr_mlx.setup_backend("auto", "distil-large-v2") == "faster-whisper"
    assert asr_mlx.setup_backend("parakeet", "small") == "parakeet"


def test_download_asr_routes_the_mlx_engine_to_the_pinned_repo(tmp_path: Path) -> None:
    got: dict[str, Any] = {}

    def fast(model: str, out: Path, write: Any, **kw: Any) -> Path:
        got.update(kw, model=model, out=out)
        return out

    p = models.download_asr(
        "small", tmp_path, lambda s: None, connections=3, write=lambda s: None, fast=fast, engine="mlx-whisper"
    )
    spec = models.MLX_MODELS["small"]
    assert p == tmp_path / "whisper-mlx-small" == got["out"]
    assert got["repo"] == spec.repo and got["revision"] == spec.revision and got["connections"] == 3
    assert got["expect"] == {spec.weights: spec.sha256} and spec.weights in got["patterns"]


def test_download_mlx_skips_weights_already_on_disk(tmp_path: Path) -> None:
    d = models.mlx_dir(tmp_path, "tiny")
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "weights.npz").write_bytes(b"x")
    msgs: list[str] = []
    assert models.download_mlx("tiny", tmp_path, msgs.append, fast=lambda *a, **k: 1 / 0) == d
    assert "already present" in msgs[0]


def _hub(listing: list[dict[str, Any]]) -> Any:
    def opener(req: urllib.request.Request, timeout: float) -> Any:
        assert "/api/models/mlx-community/whisper-tiny-mlx/tree/" in req.full_url
        return io.BytesIO(json.dumps(listing).encode())

    return opener


def test_fastdl_refuses_a_hub_file_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    spec = models.MLX_MODELS["tiny"]
    listing = [
        {"type": "file", "path": "config.json", "size": 2, "oid": "abc"},
        {"type": "file", "path": "weights.npz", "size": 5, "lfs": {"oid": "f" * 64, "size": 5}},
    ]
    with pytest.raises(fastdl.ChecksumError, match="pinned"):
        fastdl.download_model(
            "tiny",
            tmp_path,
            lambda s: None,
            opener=_hub(listing),
            repo=spec.repo,
            revision=spec.revision,
            patterns=("config.json", spec.weights),
            required=(spec.weights,),
            expect={spec.weights: spec.sha256},
            retries=1,
            backoff=0,
        )


def test_fastdl_listing_needs_the_mlx_weights_not_model_bin() -> None:
    listing = [{"type": "file", "path": "config.json", "size": 2, "oid": "abc"}]
    with pytest.raises(RuntimeError, match=r"weights\.npz"):
        fastdl.list_files(
            "mlx-community/whisper-tiny-mlx",
            _hub(listing),
            "main",
            None,
            ("config.json", "weights.npz"),
            ("weights.npz",),
        )
    ok = [*listing, {"type": "file", "path": "weights.npz", "size": 5, "lfs": {"oid": "a" * 64, "size": 5}}]
    files = fastdl.list_files(
        "mlx-community/whisper-tiny-mlx", _hub(ok), "main", None, ("config.json", "weights.npz"), ("weights.npz",)
    )
    assert [f.path for f in files] == ["config.json", "weights.npz"]


def test_mlx_model_path_downloads_missing_weights_next_to_the_whisper_folder(tmp_path: Path) -> None:
    fetched: list[tuple[str, Path]] = []
    root = str(tmp_path / "whisper-small")
    assert asr_mlx.model_path("small", root, fetch=lambda m, d, p: fetched.append((m, d))) == str(
        tmp_path / "whisper-mlx-small"
    )
    assert fetched == [("small", tmp_path)]
    assert asr_mlx.model_path("small", None) == models.MLX_MODELS["small"].repo
    with pytest.raises(KeyError):
        asr_mlx.model_path("large-v1", root)
