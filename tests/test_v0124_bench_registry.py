"""v0.1.24 model refresh: "Test my PC" (model_bench: measure, recommend, report, self-test hook, Agent Link domain),
automatic rollback (model_guard), the registry's 2026-10 presets and their gating, the models_next settings, and
the "Newer models (optional)" panel logic - with fakes, no downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from away import config as cfgmod
from away.app import model_bench, model_guard, model_registry
from away.app.model_bench import Measure
from away.app.model_registry import Registry
from away.config import AppConfig, ModelsNextConfig

ROOT = Path(__file__).resolve().parent.parent
NEM = "stt-nemotron-3.5-streaming-0.6b-q8"
MOON = "stt-moonshine-v2-small-streaming-en"
VAD = "vad-silero-v6.2"
Q8 = "cleanup-qwen3.5-2b-q8"
Q4 = "cleanup-qwen3.5-2b-q4km"


def _ok(mid, slot="stt", **kw):
    return Measure(mid, slot, mid, ok=True, **kw)


RTX2060 = {"nvidia": True, "name": "NVIDIA GeForce RTX 2060", "total_mb": 6144, "used_mb": 500}
NO_GPU = {"nvidia": False, "name": "", "total_mb": 0, "used_mb": 0}


# ============================================================================================ measure / probes
def test_measure_times_load_and_steps_and_never_raises():
    t = [0.0]

    def clock():
        return t[0]

    def load():
        t[0] += 1.5
        return SimpleNamespace(close=lambda: None)

    def step(obj):
        t[0] += 0.25

    vr = iter([1000, 2100])
    m = model_bench.measure(
        "x",
        "stt",
        "X",
        load,
        step,
        runs=3,
        audio_s=5.0,
        vram=lambda: next(vr),
        clock=clock,
        cpu_clock=lambda: t[0] / 2,
        ncpu=2,
    )
    assert m.ok and m.load_s == 1.5 and m.latency_ms == 250.0 and m.rtf == 0.05 and m.vram_mb == 1100
    assert m.cpu_pct == 25.0  # half a core of two
    bad = model_bench.measure("y", "stt", "Y", lambda: (_ for _ in ()).throw(OSError("CUDA error 35")), step)
    assert not bad.ok and bad.error.startswith("OSError: CUDA error 35")


def test_gpu_info_reads_nvidia_smi_and_copes_without_one():
    run = lambda *a, **k: SimpleNamespace(stdout="NVIDIA GeForce RTX 2060, 6144, 812\n")  # noqa: E731
    assert model_bench.gpu_info(run, lambda n: "/usr/bin/nvidia-smi") == {
        "nvidia": True,
        "name": "NVIDIA GeForce RTX 2060",
        "total_mb": 6144,
        "used_mb": 812,
    }
    assert model_bench.gpu_info(run, lambda n: None)["nvidia"] is False
    broken = lambda *a, **k: SimpleNamespace(stdout="")  # noqa: E731
    assert model_bench.gpu_info(broken, lambda n: "x") == {"nvidia": True, "name": "", "total_mb": 0, "used_mb": 0}


def test_test_audio_is_deterministic_speech_shaped():
    a, b = model_bench.test_audio(1.0), model_bench.test_audio(1.0)
    assert len(a) == 16000 and (a == b).all() and 0.05 < float(abs(a).max()) <= 0.4
    assert (a == 0).mean() > 0.3  # pauses


# ============================================================================================ recommend
def test_recommend_needs_silero_first():
    rec = model_bench.recommend([_ok(MOON, rtf=0.1)], NO_GPU)
    assert rec["preset"] == "none" and not rec["passed"] and "Silero VAD v6.2" in rec["why"]


def test_recommend_best_on_a_free_6gb_nvidia_card():
    ms = [
        _ok(VAD, "vad"),
        _ok(NEM, rtf=0.1),
        _ok(Q8, "cleanup", latency_ms=900.0),
        _ok(Q4, "cleanup", latency_ms=600.0),
    ]
    rec = model_bench.recommend(ms, RTX2060)
    assert rec == {
        "preset": "best",
        "passed": True,
        "why": "the graphics card runs streaming speech and the 8-bit cleanup fast",
    }


def test_recommend_balanced_when_vram_is_tight_or_8bit_cleanup_is_slow():
    ms = [_ok(VAD, "vad"), _ok(NEM, rtf=0.1), _ok(Q8, "cleanup", latency_ms=900.0)]
    busy = {**RTX2060, "used_mb": 1500}  # 4644 MB free < 3900 + 1024 headroom, >= 2700 + 1024
    assert model_bench.recommend(ms, busy)["preset"] == "balanced"
    slow = [
        _ok(VAD, "vad"),
        _ok(NEM, rtf=0.1),
        _ok(Q8, "cleanup", latency_ms=4000.0),
        _ok(Q4, "cleanup", latency_ms=3000.0),
    ]
    rec = model_bench.recommend(slow, RTX2060)
    assert rec["preset"] == "balanced" and "cleanup was slow" in rec["why"]


def test_recommend_light_on_the_processor_and_none_with_reasons():
    assert model_bench.recommend([_ok(VAD, "vad"), _ok(MOON, rtf=0.2)], NO_GPU)["preset"] == "light"
    # a fast GPU result is ignored without an NVIDIA card; Moonshine too slow -> nothing changes
    rec = model_bench.recommend([_ok(VAD, "vad"), _ok(NEM, rtf=0.1), _ok(MOON, rtf=0.9)], NO_GPU)
    assert rec["preset"] == "none" and "Moonshine was too slow here (RTF 0.9)" in rec["why"]
    assert rec["why"].endswith("Nothing was changed.")
    failed = Measure(NEM, "stt", "N", ok=False, error="RuntimeError: CUDA driver too old")
    rec = model_bench.recommend([_ok(VAD, "vad"), failed], RTX2060)
    assert "Nemotron did not run (RuntimeError: CUDA driver too old)" in rec["why"]
    rec = model_bench.recommend([_ok(VAD, "vad")], NO_GPU)
    assert "download Moonshine" in rec["why"]
    # a slow Nemotron on a GPU
    rec = model_bench.recommend([_ok(VAD, "vad"), _ok(NEM, rtf=0.8)], RTX2060)
    assert rec["preset"] == "none" and "too slow" in rec["why"]


def test_recommend_uses_the_registry_vram_estimates():
    ms = [_ok(VAD, "vad"), _ok(NEM, rtf=0.1), _ok(Q8, "cleanup", latency_ms=900.0)]
    est = {"light": 700, "balanced": 2700, "best": 9000}
    assert model_bench.recommend(ms, RTX2060, est)["preset"] == "balanced"


# ============================================================================================ report + apply
def _bench_light(tmp_path, cfg=None):
    benches = [(VAD, lambda: _ok(VAD, "vad")), (MOON, lambda: _ok(MOON, rtf=0.2))]
    seen = []
    rep = model_bench.run_bench(cfg or AppConfig(), tmp_path, benches=benches, progress=seen.append, gpu=NO_GPU)
    return rep, seen


def test_run_bench_saves_a_report_and_summarises_it(tmp_path):
    assert model_bench.last_report(tmp_path) is None
    assert model_bench.summary(None) == "Not tested yet: press Test my PC"
    rep, seen = _bench_light(tmp_path)
    assert seen == ["Testing Silero VAD v6.2...", f"Testing {Registry(tmp_path).models[MOON].title}..."]
    saved = model_bench.last_report(tmp_path)
    assert saved == json.loads(json.dumps(rep.to_dict()))
    assert model_bench.passed(saved) and saved["recommendation"]["preset"] == "light"
    assert model_bench.summary(saved).startswith("Recommended: Light - Moonshine keeps up")
    assert model_bench.summary({"recommendation": {"passed": False, "why": "x"}}) == "Keep the current models - x"
    model_bench.report_path(tmp_path).write_text("not json")
    assert model_bench.last_report(tmp_path) is None


def test_apply_recommendation_only_after_a_pass_and_remembers_undo_per_slot(tmp_path):
    cfg = AppConfig()
    with pytest.raises(ValueError, match="did not pass"):
        model_bench.apply_recommendation(cfg, {"recommendation": {"passed": False}}, tmp_path)
    rep, _ = _bench_light(tmp_path)
    before = model_bench.apply_recommendation(cfg, rep.to_dict(), tmp_path)
    assert cfg.models_next.presets == "next" and before["models_next.presets"] == "classic"
    # nothing is downloaded, so the new speech model is not switched on (its slot stays as it was)
    assert cfg.models_next.stt == "whisper" and cfg.models_next.vad == "bundled"
    pend = model_guard.pending(tmp_path)
    assert pend["preset"] == {"models_next.presets": "classic"}
    Registry.restore(cfg, before)
    assert cfg.models_next.presets == "classic"


def test_selftest_hook_skip_pass_warn(tmp_path):
    assert model_bench.SELFTEST_HOOK["id"] == "model_bench" and callable(model_bench.SELFTEST_HOOK["check"])
    cfg = AppConfig()
    r = model_bench.selftest_check(cfg, tmp_path)
    assert r["status"] == "skip" and r["fix"] == {"id": "model_bench:run", "label": "Test my PC"}
    _bench_light(tmp_path)
    r = model_bench.selftest_check(cfg, tmp_path)
    assert r["status"] == "pass" and r["fix"] == {"id": "model_bench:apply", "label": "Use Light"}
    assert r["details"]["recommendation"]["preset"] == "light"
    cfg.models_next.presets = "next"
    assert model_bench.selftest_check(cfg, tmp_path)["fix"] is None  # already using it
    model_bench.run_bench(cfg, tmp_path, benches=[], gpu=NO_GPU)
    r = model_bench.selftest_check(cfg, tmp_path)
    assert r["status"] == "warn" and r["fix"]["label"] == "Test again"
    assert set(r) == {"id", "title", "status", "summary", "details", "fix"}


def test_model_bench_agent_link_domain(tmp_path, monkeypatch):
    from away.agentlink import registry as al

    model_bench.register_agent_link(tmp_path)
    try:
        cfg = AppConfig()
        assert "model_bench" in al.domains()
        rows = model_bench.agent_rows(cfg, tmp_path)
        assert rows[0]["key"] == "status" and rows[0]["passed"] is False and rows[0]["presets"] == "classic"
        with pytest.raises(al.ChangeError, match="did not pass"):
            al.apply_ops(cfg, [al.Op("model_bench", "apply", "")])
        _bench_light(tmp_path)
        (done,) = al.apply_ops(cfg, [al.Op("model_bench", "apply", "")])
        assert cfg.models_next.presets == "next" and done.after == "light" and done.low_risk
        assert done.text == "Models: light preset (2026-10)"
        (inv,) = done.inverse
        assert inv.op == "restore"
        with pytest.raises(al.ChangeError, match="not a change an agent can make"):
            al.apply_ops(cfg, [inv])  # inverse ops are undo-only
        al.apply_ops(cfg, [inv], internal=True)
        assert cfg.models_next.presets == "classic"
        rows = model_bench.agent_rows(cfg, tmp_path)
        assert {r["key"] for r in rows[1:]} == {VAD, MOON}
        with pytest.raises(al.ChangeError, match="run, apply and restore"):
            al.apply_ops(cfg, [al.Op("model_bench", "delete", "")])
        # run starts in the background and returns at once
        ran = []
        monkeypatch.setattr(model_bench, "run_bench", lambda c, d=None: ran.append(c))
        (r,) = al.apply_ops(cfg, [al.Op("model_bench", "run", "")])
        assert "started" in r.text or "already running" in r.text
        for _ in range(100):
            if not model_bench._running.is_set():
                break
            import time

            time.sleep(0.01)
        assert ran and ran[0] is not cfg  # a snapshot, never the live settings
    finally:
        al.unregister("model_bench")


# ============================================================================================ model_guard
def test_guard_keeps_the_oldest_undo_and_rolls_back_one_slot(tmp_path):
    cfg = AppConfig()
    assert model_guard.rollback(cfg, "stt", "x", tmp_path) is None  # nothing to undo
    model_guard.remember("stt", {"models_next.stt": "whisper"}, tmp_path)
    model_guard.remember("stt", {"models_next.stt": "moonshine"}, tmp_path)  # second switch: oldest kept
    model_guard.remember("vad", {"models_next.vad": "bundled"}, tmp_path)
    cfg.models_next.stt, cfg.models_next.vad = "nemotron", "silero6"
    msg = model_guard.rollback(cfg, "stt", "CUDA error: no kernel image is available", tmp_path)
    assert msg == "Speech to text: switched back to the previous model (CUDA error: no kernel image is available)"
    assert cfg.models_next.stt == "whisper" and cfg.models_next.vad == "silero6"  # only its own slot
    assert set(model_guard.pending(tmp_path)) == {"vad"}
    model_guard.confirm("vad", tmp_path)  # it worked: the undo is dropped
    assert model_guard.pending(tmp_path) == {}
    assert model_guard.rollback(cfg, "vad", "x", tmp_path) is None


def test_guard_respects_auto_rollback_off(tmp_path):
    cfg = AppConfig()
    cfg.models_next.auto_rollback = False
    model_guard.remember("stt", {"models_next.stt": "whisper"}, tmp_path)
    cfg.models_next.stt = "moonshine"
    assert model_guard.rollback(cfg, "stt", "boom", tmp_path) is None
    assert cfg.models_next.stt == "moonshine" and "stt" in model_guard.pending(tmp_path)


def test_guard_splits_a_preset_undo_per_slot(tmp_path):
    assert model_guard.slot_of_key("models_next.moonshine_model") == "stt"
    assert model_guard.slot_of_key("voice.cleanup.model") == "cleanup"
    assert model_guard.slot_of_key("speech.engine") == "voice"
    assert model_guard.slot_of_key("models_next.kokoro_precision") == "voice"
    assert model_guard.slot_of_key("models_next.wake_verify") == "wake"
    assert model_guard.slot_of_key("models_next.hands_landmarker") == "hands"
    assert model_guard.slot_of_key("models_next.presets") == "preset"
    model_guard.remember_all(
        {"models_next.stt": "whisper", "voice.asr_model": "auto", "speech.engine": "auto", "gaze.enabled": True},
        tmp_path,
    )
    p = model_guard.pending(tmp_path)
    assert p == {
        "stt": {"models_next.stt": "whisper", "voice.asr_model": "auto"},
        "voice": {"speech.engine": "auto"},
        "gaze": {"gaze.enabled": True},
    }
    (tmp_path / "models" / "undo.json").write_text("[1, 2]")
    assert model_guard.pending(tmp_path) == {}  # a damaged file is ignored


def test_try_next_keeps_a_faster_model_and_switches_back_a_slower_one(tmp_path):
    reg = Registry(tmp_path)
    cfg = AppConfig()
    score = {"whisper": 10.0, "moonshine": 12.0, "nemotron": 5.0}
    bench = lambda c: score[c.models_next.stt]  # noqa: E731
    res = model_guard.try_next(reg, cfg, "stt", MOON, bench, tmp_path)
    assert res.kept and cfg.models_next.stt == "moonshine"
    assert model_guard.pending(tmp_path)["stt"]["models_next.stt"] == "whisper"
    res = model_guard.try_next(reg, cfg, "stt", NEM, bench, tmp_path)
    assert not res.kept and "slower" in res.message and cfg.models_next.stt == "moonshine"
    # this PC's tolerance: 15 % by default
    cfg.models_next.rollback_tolerance = 0.6
    assert model_guard.try_next(reg, cfg, "stt", NEM, bench, tmp_path).kept


# ============================================================================================ registry + gating
def test_old_catalog_keys_stay_empty_so_older_apps_see_no_change():
    """v0.1.23 apps read only "models" and "modes" from the online catalog; the 2026-10 entries live under
    models_next / presets_next, and the published sha256 still matches."""
    raw = (ROOT / "models" / "catalog.json").read_bytes()
    cat = json.loads(raw)
    assert cat["models"] == [] and cat["modes"] == {} and cat["version"] == 1
    assert {"models_next", "presets_next"} <= set(cat)
    sha = (ROOT / "models" / "catalog.sha256").read_text().split()[0]
    # the published file has LF line ends (a Windows checkout may turn them into CRLF)
    assert sha == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    shipped = json.loads((ROOT / "src" / "away" / "data" / "catalog_next.json").read_text())
    assert shipped["models_next"] == cat["models_next"] and shipped["presets_next"] == cat["presets_next"]
    # the built-in manifest stays the v0.1.23 one: nothing in it needs 0.1.24
    for m in model_registry.builtin_manifest()["models"]:
        model_registry.validate(m)  # the old rules
        assert not m.get("files") and not any(k.startswith("models_next.") for k in m["set"])


def test_every_models_next_entry_is_rejected_by_the_old_rules_or_is_an_info_update():
    """An old validator (no next_entry) must never accept a new bundle, runtime or models_next.* key."""
    cat = json.loads((ROOT / "models" / "catalog.json").read_text())
    builtin_ids = {m["id"] for m in model_registry.builtin_manifest()["models"]}
    for item in cat["models_next"]:
        model_registry.validate(item, next_entry=True)  # all valid for 0.1.24
        if item["id"] in builtin_ids:
            continue
        uses_new = (
            bool(item.get("files"))
            or item.get("runtime") in model_registry.NEXT_RUNTIMES
            or any(k.startswith("models_next.") for k in item["set"])
        )
        if uses_new:
            with pytest.raises(ValueError):
                model_registry.validate(item)


def test_registry_loads_the_shipped_2026_10_entries(tmp_path):
    reg = Registry(tmp_path)
    assert reg.errors == []
    for mid in (NEM, MOON, VAD, Q8, Q4, "wake-livekit-custom", "hands-rtmpose-m-hand5", "voice-pocket-tts"):
        assert reg.models[mid].source == "catalog"
    assert reg.models[NEM].experimental and reg.models["hands-rtmpose-m-hand5"].experimental
    assert "hands-rtmdet-nano" in reg.models  # a component of RTMPose...
    assert "hands-rtmdet-nano" not in {m.id for m in reg.for_slot("hands", True)}  # ...not a choice of its own
    # a models_next entry for an existing model only fills in its notes; settings and title stay
    turbo = reg.models["whisper-large-v3-turbo-gpu"]
    assert (
        turbo.source == "built-in"
        and turbo.notes
        and turbo.set == {"voice.asr_model": "large-v3-turbo", "voice.device": "cuda"}
    )
    assert set(model_registry.SLOTS) >= {"vad", "wake"}


def test_online_catalog_with_models_next_wins_over_the_shipped_copy(tmp_path):
    d = tmp_path / "models"
    d.mkdir(parents=True)
    online = {
        "version": 1,
        "models": [],
        "modes": {},
        "models_next": [
            {
                "id": VAD,
                "slot": "vad",
                "title": "Silero VAD v6.2 (online)",
                "license": "MIT",
                "set": {"models_next.vad": "silero6"},
                "runtime": "onnx",
            },
            {
                "id": "bad-host",
                "slot": "vad",
                "title": "x",
                "license": "MIT",
                "set": {"models_next.vad": "silero6"},
                "url": "https://evil.example.com/x.onnx",
                "sha256": "a" * 64,
            },
        ],
        "presets_next": {},
    }
    (d / "catalog.json").write_text(json.dumps(online))
    reg = Registry(tmp_path)
    assert reg.models[VAD].title == "Silero VAD v6.2 (online)"
    assert NEM not in reg.models  # the shipped copy is not merged in
    assert any("bad-host" in e and "allowed https host" in e for e in reg.errors)
    assert reg.next_modes() == {}
    # an online catalog that predates 0.1.24 gets the shipped entries
    (d / "catalog.json").write_text(json.dumps({"version": 1, "models": [], "modes": {}}))
    assert NEM in Registry(tmp_path).models


def test_validate_next_entries():
    base = {"id": "x", "slot": "vad", "title": "X", "license": "MIT", "set": {"models_next.vad": "silero6"}}
    bundle = {**base, "files": [{"path": "a/b.onnx", "url": "https://huggingface.co/x/b", "sha256": "a" * 64}]}
    assert model_registry.validate(bundle, next_entry=True).files[0]["path"] == "a/b.onnx"
    with pytest.raises(ValueError, match=r"need Project Away 0\.1\.24"):
        model_registry.validate({**bundle, "set": {"voice.vad": "silero"}})
    with pytest.raises(ValueError, match=r"may not change models_next\.vad"):
        model_registry.validate(base)
    for bad in ("../x", "/abs", "C:/x"):
        with pytest.raises(ValueError, match="unsafe file path"):
            model_registry.validate(
                {**base, "files": [{"path": bad, "url": "https://huggingface.co/x", "sha256": "a" * 64}]},
                next_entry=True,
            )
    with pytest.raises(ValueError, match="sha256"):
        model_registry.validate({**base, "files": [{"path": "a", "url": "https://huggingface.co/x"}]}, next_entry=True)
    with pytest.raises(ValueError, match="unknown runtime"):
        model_registry.validate({**base, "runtime": "tensorrt"}, next_entry=True)
    with pytest.raises(ValueError):
        model_registry.validate({**base, "set": {"agent_link.enabled": True}}, next_entry=True)


def test_preset_vram_estimates_match_the_readme():
    reg = Registry()
    assert (reg.preset_vram_mb("light"), reg.preset_vram_mb("balanced"), reg.preset_vram_mb("best")) == (
        700,
        2700,
        3900,
    )
    with pytest.raises(ValueError):
        reg.preset_plan("ultra")


def test_classic_presets_are_unchanged_by_default():
    reg = Registry()
    cfg = AppConfig()
    assert not model_registry.use_next(cfg)
    reg.apply_mode(cfg, "light")
    # exactly the v0.1.23 Light mode of the built-in manifest (whatever it picks), nothing from presets_next
    for slot, mid in reg.classic_modes["light"].items():
        if mid in reg.models:
            for k, v in reg.models[mid].set.items():
                assert model_registry._get(cfg, k) == v, (slot, k)
    assert cfg.voice.asr_model == "base" and cfg.voice.device == "cpu"
    assert cfg.models_next == ModelsNextConfig()  # nothing new was switched on
    assert not model_registry.use_next(SimpleNamespace())  # an older settings object


def test_next_presets_switch_only_ready_models_and_apply_their_settings():
    reg = Registry()
    cfg = AppConfig()
    cfg.models_next.presets = "next"
    before = reg.apply_mode(cfg, "light", ready=lambda m: False)
    assert cfg.models_next.stt == "whisper" and cfg.models_next.vad == "bundled"  # nothing ready: slots kept
    assert cfg.gaze.enabled is False and before["gaze.enabled"] is True  # "off" slots still turn off
    assert cfg.models_next.wake_verify == "off"
    Registry.restore(cfg, before)
    assert cfg.gaze.enabled is True and cfg.models_next.wake_verify == "auto"

    reg.apply_mode(cfg, "light", ready=lambda m: True)
    assert (cfg.models_next.stt, cfg.models_next.moonshine_model) == ("moonshine", "small")
    assert cfg.models_next.vad == "silero6" and cfg.models_next.wake_engine == "livekit"
    assert cfg.voice.cleanup.model == "qwen3.5-0.8b" and cfg.speech.engine == "windows"

    cfg2 = AppConfig()
    cfg2.models_next.presets = "next"
    reg.apply_mode(cfg2, "best", ready=lambda m: True)
    assert cfg2.models_next.stt == "nemotron" and cfg2.models_next.nemotron_chunk_ms == 1120
    assert cfg2.models_next.wake_verify == "on" and cfg2.models_next.kokoro_precision == "fp16"
    assert cfg2.voice.cleanup.model == "qwen3.5-2b-q8"
    assert reg.current(cfg2, "stt").id == NEM
    cfgmod.validate(cfg2)


def test_current_ignores_models_whose_settings_an_old_config_lacks():
    reg = Registry()
    old = SimpleNamespace(**{k: getattr(AppConfig(), k) for k in ("hands", "face", "gaze", "voice", "speech")})
    reg.current(old, "stt")  # no AttributeError for the models_next.* models
    assert reg.current(old, "vad") is None  # models_next.* does not exist there


# ============================================================================================ settings
def test_models_next_defaults_keep_v0123_behaviour_and_round_trip(tmp_path):
    m = ModelsNextConfig()
    assert (m.presets, m.stt, m.vad, m.wake_engine, m.hands_landmarker, m.instant_voice) == (
        "classic",
        "whisper",
        "bundled",
        "asr",
        "mediapipe",
        False,
    )
    cfg = AppConfig()
    cfg.models_next.stt = "nemotron"
    cfg.models_next.nemotron_chunk_ms = 160
    p = tmp_path / "c.yaml"
    cfgmod.save(cfg, p)
    back, unknown = cfgmod.load(p)
    assert back.models_next == cfg.models_next and unknown == []
    # a settings file written by v0.1.23 (no models_next) loads with the defaults
    d = cfgmod.to_dict(AppConfig())
    d.pop("models_next")
    old, _ = cfgmod.from_dict(d)
    assert old.models_next == ModelsNextConfig()


@pytest.mark.parametrize(
    "field, value",
    [
        ("presets", "turbo"),
        ("stt", "parakeet"),
        ("nemotron_chunk_ms", 500),
        ("rollback_tolerance", 2.0),
        ("wake_threshold", True),
        ("instant_voice", "yes"),
        ("wake_model", 5),
        ("hands_landmarker", "yolo"),
    ],
)
def test_models_next_validation_rejects_bad_values(field, value):
    m = ModelsNextConfig()
    setattr(m, field, value)
    with pytest.raises(ValueError, match=f"models_next.{field}"):
        cfgmod.validate_models_next(m)


# ============================================================================================ panel logic
def _panel(tmp_path, ready=lambda m: False, **kw):
    from away.app.models_next_panel import NextModelsPanel

    saved = []
    cfg = AppConfig()
    p = NextModelsPanel(cfg, lambda: saved.append(1), tmp_path, ready=ready, **kw)
    return p, cfg, saved


def test_panel_presets_show_vram_and_what_to_download(tmp_path):
    p, _, _ = _panel(tmp_path)
    rows = {r["mode"]: r for r in p.presets()}
    assert rows["light"]["vram"] == "about 0.7 GB of graphics memory"
    assert rows["balanced"]["vram"] == "about 2.7 GB of graphics memory"
    assert rows["best"]["vram"] == "about 3.9 GB of graphics memory"
    assert rows["best"]["download_mb"] > 1000 and rows["best"]["missing"]
    assert any("Nemotron" in t for t in rows["balanced"]["experimental"])
    ready, _, _ = _panel(tmp_path, ready=lambda m: True)
    assert all(r["missing"] == [] and r["download_mb"] == 0 for r in ready.presets())


def test_panel_licence_notes(tmp_path):
    from away.app.models_next_panel import GPL_NOTE

    p, _, _ = _panel(tmp_path)
    notes = p.license_notes("balanced")
    assert notes[-1] == GPL_NOTE and "never" in GPL_NOTE and "espeak-ng" in GPL_NOTE and "Piper" in GPL_NOTE
    assert any("OpenMDW" in n for n in notes)
    assert GPL_NOTE in p.license_notes("light")


def test_panel_headline_test_and_one_click(tmp_path):
    rep = {"recommendation": {"passed": True, "preset": "light", "why": "Moonshine keeps up on the processor"}}
    p, cfg, saved = _panel(tmp_path, ready=lambda m: True, bench=lambda: rep)
    assert p.headline().startswith("Using the standard models. Press Test my PC")
    res = p.test_my_pc()
    assert res == {
        "passed": True,
        "preset": "light",
        "text": "Recommended: Light - Moonshine keeps up on the processor",
    }
    msg = p.use_preset("light")
    assert "newer Light models" in msg and saved == [1]
    assert cfg.models_next.presets == "next" and cfg.models_next.stt == "moonshine"
    assert p.headline().startswith("Using the newer models.")
    assert model_guard.pending(tmp_path)  # every changed job can roll back by itself
    assert p.standard() == "Back to the standard models."
    assert cfg.models_next.presets == "classic" and cfg.models_next.stt == "whisper"
    assert cfg.models_next.vad == "bundled" and cfg.gaze.enabled is True and model_guard.pending(tmp_path) == {}


def test_panel_download_missing_skips_trained_and_reports_errors(tmp_path):
    got = []

    def dl(m):
        got.append(m.id)
        if m.id == NEM:
            raise RuntimeError("disk full")

    p, _, _ = _panel(tmp_path, downloader=dl)
    errs = p.download_missing("balanced", progress=lambda s: None)
    assert "wake-livekit-custom" not in got  # trained per name: never a one-click download
    assert NEM in got and Q4 in got
    assert errs == [f"{Registry(tmp_path).models[NEM].title}: disk full"]


def test_panel_advanced_lists_new_models_per_job(tmp_path):
    p, _, _ = _panel(tmp_path)
    rows = {r["slot"]: r for r in p.advanced()}
    assert {"stt", "vad", "wake", "cleanup", "voice"} <= set(rows)
    stt_ids = {c["id"] for c in rows["stt"]["choices"]}
    assert {NEM, MOON} <= stt_ids
    assert rows["vad"]["current"] == "vad-bundled"


def test_panel_states_in_plain_words(tmp_path, monkeypatch):
    from away.app.models_next_panel import state_text, title_with_flag
    from away.voice import tts_next

    reg = Registry(tmp_path)
    assert state_text(reg.models[VAD], True) == "ready"
    assert state_text(reg.models[VAD], False).endswith("MB to download")
    assert state_text(reg.models["wake-livekit-custom"], False).startswith("train a model for your name")
    assert state_text(reg.models["voice-pocket-tts"], False) == "optional install: pip install 'project-away[pocket]'"
    assert title_with_flag({"title": "RTMPose hands (experimental)", "experimental": True}).count("experimental") == 1
    assert title_with_flag({"title": "Nemotron", "experimental": True}) == "Nemotron (experimental)"
    # pocket-tts is ready only once the [pocket] extra is installed
    monkeypatch.setattr(tts_next, "pocket_available", lambda: False)
    p, _, _ = _panel(tmp_path, ready=None)
    assert not p.is_ready(reg.models["voice-pocket-tts"])
    monkeypatch.setattr(tts_next, "pocket_available", lambda: True)
    assert p.is_ready(reg.models["voice-pocket-tts"])
    assert p.is_ready(reg.models["voice-windows"])  # built in
    assert not p.is_ready(reg.models["wake-livekit-custom"])  # no trained file yet


def test_new_model_hint_ignores_the_2026_10_models(tmp_path):
    """The v0.1.23 tray hint keeps talking only about the v0.1.23 catalog; the newer models wait in their panel."""
    reg = Registry(tmp_path)
    assert reg.models[NEM].from_next and not reg.models["whisper-large-v3-turbo-gpu"].from_next
    assert model_registry.new_model_hint(reg, tmp_path) == ""
    item = {
        "id": VAD,
        "slot": "vad",
        "title": "x",
        "license": "MIT",
        "set": {"models_next.vad": "silero6"},
        "from_next": False,
        "source": "built-in",
    }
    m = model_registry.validate(item, next_entry=True)
    assert m.from_next is False and m.source == "built-in"  # a catalog cannot set either field
