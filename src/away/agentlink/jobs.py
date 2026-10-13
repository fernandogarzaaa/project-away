"""Diagnose jobs an agent can run (one at a time): camera diagnosis, camera mode sweep, model benchmark.

Camera checks run the existing CLI commands in their own process with fixed arguments (``away cameras diagnose``,
``away cameras test``): they already know how to borrow the cameras from the running app and give them back.
The agent can only pick the job and, for the diagnosis, one camera name (passed as a single argument, never
through a shell). Output is kept in ``<data>/diagnostics/cameras-<kind>-<time>.txt`` for the bundle.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("cameras", "sweep", "benchmark")
DEVICE_RE = re.compile(r"^[\w .:#()\-]{1,80}$", re.UNICODE)


@dataclass
class Job:
    id: str
    kind: str
    started: float
    state: str = "running"  # running | done | failed
    output: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    file: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "started": self.started,
            "output": self.output[-200:],
            "result": self.result,
            "file": self.file,
        }


def benchmark(iterations: int = 30) -> dict[str, Any]:
    """Milliseconds per call for the gesture pipeline, the static pose model, the temporal model (when
    onnxruntime is installed) and the hand landmarker (when MediaPipe and its model are installed)."""
    import numpy as np

    from away import config as cfgmod
    from away.app.runner import Pipeline
    from away.hands.runtime import StaticPoseClassifier, TemporalGestureClassifier, bundled_model
    from away.hands.synthetic import gesture_sequence, pose

    out: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    pipe = Pipeline(cfgmod.AppConfig())
    seq = [lm for g in ("move", "pinch_click", "scroll") for lm in gesture_sequence(g, 24, rng)]
    t0 = time.perf_counter()
    for i, lm in enumerate(seq):
        pipe.on_hand(i / 30.0, lm)
    out["pipeline_ms_per_frame"] = round((time.perf_counter() - t0) * 1000.0 / len(seq), 3)
    p = bundled_model("static_pose.npz")
    if p:
        clf = StaticPoseClassifier(p)
        hand = pose("palm")
        t0 = time.perf_counter()
        for _ in range(iterations):
            clf.predict(hand)
        out["static_pose_ms"] = round((time.perf_counter() - t0) * 1000.0 / iterations, 3)
    p = bundled_model("gesture_tcn.onnx")
    if p:
        try:
            tcn = TemporalGestureClassifier(p)
            for lm in seq[: tcn.frames]:
                tcn.push(lm)
            t0 = time.perf_counter()
            for lm in seq[:iterations]:
                tcn.push(lm)
            out["temporal_model_ms"] = round((time.perf_counter() - t0) * 1000.0 / iterations, 3)
        except Exception as exc:
            out["temporal_model_ms"] = f"unavailable: {exc}"
    try:
        from away import paths
        from away.hands.tracker import HandTracker

        model = paths.models_dir() / "hand_landmarker.task"
        if model.exists():
            tr = HandTracker(model, 2)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            for i in range(10):
                tr.process(img, i / 30.0)
            out["hand_landmarker_ms_640x480"] = round((time.perf_counter() - t0) * 100.0, 2)
            close = getattr(tr, "close", None)
            if close:
                close()
        else:
            out["hand_landmarker_ms_640x480"] = "model not downloaded"
    except Exception as exc:
        out["hand_landmarker_ms_640x480"] = f"unavailable: {exc}"
    return out


class JobRunner:
    def __init__(
        self,
        out_dir: Path,
        command: Callable[[list[str]], list[str]] | None = None,
        runner: Callable[..., Any] | None = None,
        bench: Callable[[], dict[str, Any]] = benchmark,
    ) -> None:
        self.out_dir = Path(out_dir)
        if command is None:
            from away.app.frozen import away_command

            command = away_command
        self.command = command
        self.runner = runner or subprocess.Popen
        self.bench = bench
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def running(self) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs.values() if j.state == "running"), None)

    def start(self, kind: str, device: str | None = None, seconds: float | None = None) -> Job:
        if kind not in KINDS:
            raise ValueError(f"unknown check {kind!r}; use {', '.join(KINDS)}")
        if self.running() is not None:
            raise RuntimeError("another check is still running; wait for it (get_job)")
        args: list[str] = []
        if kind == "cameras":
            args = ["cameras", "diagnose"]
            if device:
                if not DEVICE_RE.match(str(device)):
                    raise ValueError("camera name has characters that are not allowed")
                args.append(str(device))
            args += ["--seconds", f"{max(0.3, min(float(seconds or 1.0), 5.0)):g}"]
        elif kind == "sweep":
            args = ["cameras", "test", "--seconds", f"{max(2.0, min(float(seconds or 10.0), 30.0)):g}"]
        job = Job(uuid.uuid4().hex[:8], kind, time.time())
        with self._lock:
            self.jobs[job.id] = job
            if len(self.jobs) > 20:
                for k in list(self.jobs)[:-20]:
                    self.jobs.pop(k, None)
        threading.Thread(target=self._run, args=(job, args), daemon=True, name=f"agentlink-job-{kind}").start()
        return job

    def _run(self, job: Job, args: list[str]) -> None:
        try:
            if job.kind == "benchmark":
                job.result = self.bench()
                job.output = [f"{k}: {v}" for k, v in job.result.items()]
            else:
                kw: dict[str, Any] = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
                if sys.platform == "win32":  # pragma: no cover
                    kw["creationflags"] = 0x08000000
                proc = self.runner(self.command(args), **kw)
                deadline = time.time() + 600
                assert proc.stdout is not None
                for line in proc.stdout:
                    job.output.append(line.rstrip())
                    if time.time() > deadline:
                        proc.kill()
                        job.output.append("(stopped after 10 minutes)")
                        break
                rc = proc.wait(timeout=30)
                job.result = {"exit_code": rc}
            self.out_dir.mkdir(parents=True, exist_ok=True)
            f = self.out_dir / f"cameras-{job.kind}-{time.strftime('%Y%m%d-%H%M%S')}.txt"
            f.write_text("\n".join(job.output) + "\n", encoding="utf-8")
            job.file = f"diagnostics/{f.name}"
            job.state = "done"
        except Exception as exc:
            job.output.append(f"failed: {exc}")
            job.state = "failed"

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)
