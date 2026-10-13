""""Test my PC" for the 2026-10 models (v0.1.24): measure each model here, recommend Light / Balanced / Best, and
switch only when the benchmark passes. Until then Light / Balanced / Best keep picking the v0.1.23 models.

For every model that is downloaded (the rest are listed as "not downloaded", never fetched by the benchmark):

* **load** - seconds to load it;
* **latency** - median time of a short, fixed job (5 s of audio for speech-to-text, a sentence for cleanup and the
  voices, 30 ms windows for speech detection, a 2 s window for the wake word, a 640x480 frame for hands);
* **RTF** - for speech: processing time / audio length (below 1 = faster than real time);
* **CPU %** - this process's CPU time during the job as a share of the whole computer (so a busy PC shows up);
* **VRAM** - graphics memory used after loading minus before (``nvidia-smi``; empty without an NVIDIA card).

:func:`recommend` turns that into a preset; :func:`passed` is the gate (`models_next.presets: next`).
Run it from Settings > Models ("Test my PC"), ``python -m away.app.model_bench``, the Agent Link
``model_bench`` domain (``run`` / ``apply`` / ``restore``), or the self-test hook :data:`SELFTEST_HOOK`.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SR = 16000
REPORT = "bench.json"
SENTENCE = "um so i think we should uh meet at three pm tomorrow to go over the the report"
RTF_OK = 0.5  # speech-to-text must run at least 2x faster than real time on this PC
CLEANUP_OK_MS = 1500.0
CLEANUP_LIGHT_OK_MS = 2500.0
VRAM_HEADROOM_MB = 1024  # Windows, the browser and the desktop keep using the card


@dataclass
class Measure:
    id: str
    slot: str
    title: str
    ok: bool = False
    skipped: bool = False
    load_s: float | None = None
    latency_ms: float | None = None
    rtf: float | None = None
    cpu_pct: float | None = None
    vram_mb: int | None = None
    error: str = ""


@dataclass
class Report:
    when: float
    gpu: dict[str, Any]
    cpu_busy_pct: float | None
    measures: list[Measure] = field(default_factory=list)
    recommendation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["measures"] = [asdict(m) for m in self.measures]
        return d


# ------------------------------------------------------------------------------------------------ probes
def gpu_info(run: Callable[..., Any] = subprocess.run, which: Callable[[str], str | None] = shutil.which) -> dict:
    """``{"nvidia": bool, "name", "total_mb", "used_mb"}`` from nvidia-smi (all empty without one)."""
    if not which("nvidia-smi"):
        return {"nvidia": False, "name": "", "total_mb": 0, "used_mb": 0}
    try:
        cp = run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        name, total, used = (p.strip() for p in str(cp.stdout).strip().splitlines()[0].split(","))
        return {"nvidia": True, "name": name, "total_mb": int(float(total)), "used_mb": int(float(used))}
    except Exception:
        return {"nvidia": True, "name": "", "total_mb": 0, "used_mb": 0}


def cpu_busy_pct() -> float | None:
    """How busy the whole computer is right now (psutil when present)."""
    try:
        import psutil

        return float(psutil.cpu_percent(interval=0.5))
    except Exception:
        return None


def measure(
    mid: str,
    slot: str,
    title: str,
    load: Callable[[], Any],
    step: Callable[[Any], Any],
    runs: int = 3,
    audio_s: float | None = None,
    vram: Callable[[], int | None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
    ncpu: int | None = None,
) -> Measure:
    """Load once, run ``step`` once to warm up and ``runs`` times measured. Never raises."""
    m = Measure(mid, slot, title)
    n = max(1, ncpu or os.cpu_count() or 1)
    before = vram() if vram else None
    try:
        t = clock()
        obj = load()
        m.load_s = round(clock() - t, 2)
        after = vram() if vram else None
        if before is not None and after is not None:
            m.vram_mb = max(0, int(after) - int(before))
        step(obj)
        times = []
        c0, w0 = cpu_clock(), clock()
        for _ in range(max(1, runs)):
            t = clock()
            step(obj)
            times.append(clock() - t)
        wall = max(1e-6, clock() - w0)
        m.cpu_pct = round(100.0 * (cpu_clock() - c0) / (wall * n), 1)
        med = statistics.median(times)
        m.latency_ms = round(med * 1000, 1)
        if audio_s:
            m.rtf = round(med / audio_s, 3)
        m.ok = True
        with contextlib.suppress(Exception):
            getattr(obj, "close", lambda: None)()
    except Exception as exc:
        m.error = f"{type(exc).__name__}: {str(exc)[:200]}"
    return m


def test_audio(seconds: float = 5.0, seed: int = 7) -> np.ndarray:
    """Speech-shaped test sound (voiced bursts + pauses). Fine for timing; not for accuracy."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    env = (np.sin(2 * np.pi * 2.5 * t) > 0).astype(np.float32)
    voice = 0.2 * np.sin(2 * np.pi * 140 * t) + 0.1 * np.sin(2 * np.pi * 280 * t) + 0.02 * rng.standard_normal(len(t))
    return (voice * env).astype(np.float32)


# ------------------------------------------------------------------------------------------------ benches
def default_benches(cfg: Any, data_dir: Path | None = None, reg: Any = None) -> list[tuple[str, Callable[[], Measure]]]:
    """One measuring function per 2026-10 model that is present here (in preset order)."""
    from away.app import engines, model_fetch
    from away.app.model_registry import Registry

    reg = reg or Registry(data_dir)
    gpu = gpu_info()
    vram = (lambda: gpu_info()["used_mb"]) if gpu["nvidia"] else None
    audio = test_audio(5.0)
    out: list[tuple[str, Callable[[], Measure]]] = []

    def have(mid: str) -> bool:
        return mid in reg.models and model_fetch.have(reg.models[mid], data_dir)

    def skip(mid: str, why: str) -> Callable[[], Measure]:
        m = reg.models.get(mid)
        return lambda: Measure(mid, m.slot if m else "", m.title if m else mid, skipped=True, error=why)

    # speech detection
    from away.voice import vad_next

    mid = vad_next.MODEL_ID
    out.append(
        (
            mid,
            (
                lambda: measure(
                    mid,
                    "vad",
                    "Silero VAD v6.2",
                    lambda: vad_next.SileroV6VAD(SR, path=vad_next.model_path(data_dir)),
                    lambda v: v.probabilities(audio),
                    audio_s=5.0,
                )
            )
            if have(mid)
            else skip(mid, "not downloaded"),
        )
    )
    # speech to text
    from away.voice import asr_next

    for size, mid in asr_next.MOONSHINE_IDS.items():
        if have(mid) and engines.activate("moonshine", data_dir):
            opts = copy.copy(cfg.models_next)
            opts.moonshine_model = size
            out.append(
                (
                    mid,
                    lambda mid=mid, opts=opts: measure(
                        mid,
                        "stt",
                        reg.models[mid].title,
                        lambda: asr_next.build_next("moonshine", opts, SR, data_dir, reg),
                        lambda a: a.transcribe(audio),
                        audio_s=5.0,
                    ),
                )
            )
        else:
            out.append((mid, skip(mid, "not downloaded (model or engine)")))
    mid = asr_next.NEMOTRON_ID
    if have(mid):
        out.append(
            (
                mid,
                lambda: measure(
                    mid,
                    "stt",
                    reg.models[mid].title,
                    lambda: asr_next.build_next("nemotron", cfg.models_next, SR, data_dir, reg),
                    lambda a: a.transcribe(audio),
                    audio_s=5.0,
                    vram=vram,
                ),
            )
        )
    else:
        out.append((mid, skip(mid, "not downloaded")))
    # smart cleanup
    from away.app import optional_models as om
    from away.voice.llm_backends import LlamaCppBackend, llama_cpp_available

    for key, mid in (
        ("qwen3.5-0.8b", "cleanup-qwen3.5-0.8b-q4km"),
        ("qwen3.5-2b", "cleanup-qwen3.5-2b-q4km"),
        ("qwen3.5-2b-q8", "cleanup-qwen3.5-2b-q8"),
    ):
        md = Path(data_dir) / "models" if data_dir else _models_dir()
        if om.installed(key, md) and llama_cpp_available():
            out.append(
                (
                    mid,
                    lambda key=key, mid=mid, md=md: measure(
                        mid,
                        "cleanup",
                        reg.models[mid].title if mid in reg.models else key,
                        lambda: _loaded(LlamaCppBackend(om.path_of(key, md), gpu_layers=-1 if gpu["nvidia"] else 0)),
                        lambda b: b.complete("Fix punctuation. Return only the text.", SENTENCE, 48, 30),
                        runs=2,
                        vram=vram,
                    ),
                )
            )
        else:
            out.append((mid, skip(mid, "not downloaded (model or cleanup engine)")))
    return out


def _models_dir() -> Path:
    from away.app.model_registry import models_dir

    return models_dir()


def _loaded(b: Any) -> Any:
    b.load()
    return b


# ------------------------------------------------------------------------------------------------ verdict
def _get(ms: list[Measure], mid: str) -> Measure | None:
    return next((m for m in ms if m.id == mid), None)


def recommend(ms: list[Measure], gpu: dict[str, Any], vram_est: dict[str, int] | None = None) -> dict[str, Any]:
    """``{"preset": light|balanced|best|none, "passed": bool, "why": str}``."""
    est = vram_est or {"light": 700, "balanced": 2700, "best": 3900}
    free = int(gpu.get("total_mb", 0)) - int(gpu.get("used_mb", 0))
    vad = _get(ms, "vad-silero-v6.2")
    if vad is None or not vad.ok:
        return {"preset": "none", "passed": False, "why": "Silero VAD v6.2 is needed by every new preset: download it"}
    nem = _get(ms, "stt-nemotron-3.5-streaming-0.6b-q8")
    moon = _get(ms, "stt-moonshine-v2-small-streaming-en") or _get(ms, "stt-moonshine-v2-medium-streaming-en")
    q2 = _get(ms, "cleanup-qwen3.5-2b-q4km")
    q8 = _get(ms, "cleanup-qwen3.5-2b-q8")
    gpu_ok = bool(gpu.get("nvidia")) and nem is not None and nem.ok and (nem.rtf or 9) <= RTF_OK
    if gpu_ok and q8 is not None and q8.ok and (q8.latency_ms or 1e9) <= CLEANUP_OK_MS and free >= est["best"] + VRAM_HEADROOM_MB:
        return {"preset": "best", "passed": True, "why": "the graphics card runs streaming speech and the 8-bit cleanup fast"}
    if gpu_ok and free >= est["balanced"] + VRAM_HEADROOM_MB:
        why = "the graphics card runs streaming speech fast"
        if q2 is not None and q2.ok and (q2.latency_ms or 1e9) > CLEANUP_OK_MS:
            why += " (cleanup was slow: it keeps the current model)"
        return {"preset": "balanced", "passed": True, "why": why}
    if moon is not None and moon.ok and (moon.rtf or 9) <= RTF_OK:
        return {"preset": "light", "passed": True, "why": "Moonshine keeps up on the processor"}
    reasons = []
    if nem is not None and not nem.ok and not nem.skipped:
        reasons.append(f"Nemotron did not run ({nem.error[:80]})")
    elif nem is not None and nem.ok:
        reasons.append(f"Nemotron was too slow here (RTF {nem.rtf})")
    if moon is not None and moon.ok:
        reasons.append(f"Moonshine was too slow here (RTF {moon.rtf})")
    if not reasons:
        reasons.append("download Moonshine (processor) or Nemotron (graphics card) first")
    return {"preset": "none", "passed": False, "why": "; ".join(reasons) + ". Nothing was changed."}


def report_path(data_dir: Path | None = None) -> Path:
    from away.app.model_registry import models_dir

    return models_dir(data_dir) / REPORT


def run_bench(
    cfg: Any,
    data_dir: Path | None = None,
    benches: list[tuple[str, Callable[[], Measure]]] | None = None,
    progress: Callable[[str], None] | None = None,
    gpu: dict[str, Any] | None = None,
    save: bool = True,
) -> Report:
    from away.app.model_registry import Registry

    reg = Registry(data_dir)
    g = gpu if gpu is not None else gpu_info()
    rep = Report(time.time(), g, cpu_busy_pct())
    for mid, fn in benches if benches is not None else default_benches(cfg, data_dir, reg):
        if progress is not None:
            with contextlib.suppress(Exception):
                progress(f"Testing {reg.models[mid].title if mid in reg.models else mid}...")
        rep.measures.append(fn())
    est = {}
    with contextlib.suppress(Exception):
        est = {m: reg.preset_vram_mb(m) for m in ("light", "balanced", "best")}
    rep.recommendation = recommend(rep.measures, g, est or None)
    if save:
        p = report_path(data_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    return rep


def last_report(data_dir: Path | None = None) -> dict[str, Any] | None:
    try:
        v = json.loads(report_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def passed(report: dict[str, Any] | None) -> bool:
    return bool(report and (report.get("recommendation") or {}).get("passed"))


def apply_recommendation(cfg: Any, report: dict[str, Any], data_dir: Path | None = None) -> dict[str, Any]:
    """Switch to the recommended preset (``models_next.presets: next`` + Light/Balanced/Best). Returns the undo."""
    from away.app import model_fetch
    from away.app.model_registry import Registry

    if not passed(report):
        raise ValueError("the benchmark did not pass, so nothing is switched (run Test my PC again)")
    preset = str(report["recommendation"]["preset"])
    reg = Registry(data_dir)
    before = {"models_next.presets": cfg.models_next.presets}
    cfg.models_next.presets = "next"
    for k, v in reg.apply_mode(cfg, preset, ready=lambda m: _ready(m, cfg, data_dir, model_fetch)).items():
        before.setdefault(k, v)
    from away.app import model_guard

    model_guard.remember_all(before, data_dir)  # a model that later fails rolls back its own slot
    return before


def _ready(m: Any, cfg: Any, data_dir: Path | None, model_fetch: Any) -> bool:
    if "wake_model" in (m.requires or []):
        p = str(getattr(cfg.models_next, "wake_model", "") or "")
        return bool(p) and Path(p).is_file()
    if m.bundled or m.download:
        return True
    return bool(model_fetch.have(m, data_dir))


def summary(report: dict[str, Any] | None) -> str:
    if not report:
        return "Not tested yet: press Test my PC"
    rec = report.get("recommendation") or {}
    if rec.get("passed"):
        return f"Recommended: {str(rec.get('preset', '')).title()} - {rec.get('why', '')}"
    return f"Keep the current models - {rec.get('why', '')}"


# ------------------------------------------------------------------------------------------------ Agent Link
_running = threading.Event()


def start_background(cfg: Any, data_dir: Path | None = None) -> bool:
    """Run the benchmark on a thread (an agent's call returns at once). False when one is already running."""
    if _running.is_set():
        return False
    _running.set()
    snapshot = copy.deepcopy(cfg)

    def go() -> None:
        try:
            run_bench(snapshot, data_dir)
        finally:
            _running.clear()

    threading.Thread(target=go, daemon=True, name="away-model-bench").start()
    return True


def agent_rows(cfg: Any, data_dir: Path | None = None) -> list[dict[str, Any]]:
    rep = last_report(data_dir)
    rows: list[dict[str, Any]] = [
        {
            "key": "status",
            "running": _running.is_set(),
            "summary": summary(rep),
            "presets": cfg.models_next.presets,
            "passed": passed(rep),
        }
    ]
    for m in (rep or {}).get("measures") or []:
        rows.append({"key": m.get("id"), **{k: m.get(k) for k in ("slot", "title", "ok", "skipped", "load_s", "latency_ms", "rtf", "cpu_pct", "vram_mb", "error")}})
    return rows


def register_agent_link(data_dir: Path | None = None) -> None:
    """The ``model_bench`` domain: ``run`` (starts "Test my PC" in the background), ``apply`` (switch to the
    recommended preset when it passed; undoable), ``restore``."""
    from away.agentlink import registry as al

    def apply(cfg: Any, op: Any) -> Any:
        if op.op == "run":
            started = start_background(cfg, data_dir)
            text = "Test my PC started; read model_bench for the results" if started else "Test my PC is already running"
            return al.Applied(op, None, None, [], text, True)
        if op.op == "apply":
            rep = last_report(data_dir)
            try:
                before = apply_recommendation(cfg, rep or {}, data_dir)
            except ValueError as exc:
                raise al.ChangeError(str(exc)) from None
            preset = (rep or {}).get("recommendation", {}).get("preset", "")
            return al.Applied(op, before, preset, [al.Op("model_bench", "restore", "", before)], f"Models: {preset} preset (2026-10)", True)
        if op.op == "restore":
            before = {k: al._get(cfg, k) for k in dict(op.value or {})}
            for k, v in dict(op.value or {}).items():
                al._set(cfg, k, v)
            return al.Applied(op, before, op.value, [al.Op("model_bench", "restore", "", before)], "Models: put back as before", True)
        raise al.ChangeError("model_bench supports run, apply and restore")

    al.register(
        al.Domain(
            "model_bench",
            "Test my PC (newer models: benchmark and recommended preset)",
            lambda cfg: agent_rows(cfg, data_dir),
            apply,
            ops=("run", "apply", "restore"),
        )
    )


# ------------------------------------------------------------------------------------------------ self-test hook
def selftest_check(cfg: Any, data_dir: Path | None = None, run: bool = False) -> dict[str, Any]:
    """For the v0.1.24 self-test (kept free of its types): ``{"id", "title", "status": pass|warn|skip,
    "summary", "details", "fix": {"id", "label"} | None}``. ``run``: measure now instead of reading the last result."""
    rep = run_bench(cfg, data_dir).to_dict() if run else last_report(data_dir)
    base = {"id": "model_bench", "title": "Newer models"}
    if not rep:
        return {**base, "status": "skip", "summary": summary(None), "details": {}, "fix": {"id": "model_bench:run", "label": "Test my PC"}}
    rec = rep.get("recommendation") or {}
    details = {"recommendation": rec, "measures": rep.get("measures"), "gpu": rep.get("gpu")}
    if rec.get("passed"):
        fix = None if cfg.models_next.presets == "next" else {"id": "model_bench:apply", "label": f"Use {str(rec.get('preset')).title()}"}
        return {**base, "status": "pass", "summary": summary(rep), "details": details, "fix": fix}
    return {**base, "status": "warn", "summary": summary(rep), "details": details, "fix": {"id": "model_bench:run", "label": "Test again"}}


SELFTEST_HOOK = {"id": "model_bench", "title": "Newer models", "check": selftest_check}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - prints a real benchmark
    from away import config as cfgmod

    cfg, _unknown = cfgmod.load()
    rep = run_bench(cfg, progress=print)
    print(json.dumps(rep.to_dict(), indent=1))
    print(summary(rep.to_dict()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
