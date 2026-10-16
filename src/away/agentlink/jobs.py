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
# v0.1.21: a finished check is kept this long (and at least until an agent has read its result once), so an agent
# that polls a little late still gets it; then it is "expired" (not "unknown")
KEEP_S = 600.0
KEEP_UNREAD_S = 1800.0
MAX_JOBS = 50


class JobLookupError(KeyError):
    """``get_job`` for an id that never existed here (``unknown``) or whose result was dropped (``expired``)."""

    def __init__(self, job_id: str, reason: str) -> None:
        self.job_id, self.reason = job_id, reason
        if reason == "expired":
            msg = (
                f"check {job_id} expired: results are kept {KEEP_S / 60:.0f} minutes after it finished; "
                "start it again with run_diagnose"
            )
        else:
            msg = f"unknown check {job_id}: no check with that id was started (run_diagnose returns the id)"
        super().__init__(msg)

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass
class Job:
    id: str
    kind: str
    started: float
    state: str = "running"  # running | done | failed
    output: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    file: str = ""
    finished: float | None = None  # time.time() when it ended (None while running)
    fetched: bool = False  # an agent has read it at least once since it finished

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "started": self.started,
            "output": self.output[-200:],
            "result": self.result,
            "file": self.file,
            "finished": self.finished,
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
        clock: Callable[[], float] = time.time,
        keep_s: float = KEEP_S,
        keep_unread_s: float = KEEP_UNREAD_S,
    ) -> None:
        self.out_dir = Path(out_dir)
        if command is None:
            from away.app.frozen import away_command

            command = away_command
        self.command = command
        self.runner = runner or subprocess.Popen
        self.bench = bench
        self.clock = clock
        self.keep_s, self.keep_unread_s = float(keep_s), float(keep_unread_s)
        self.jobs: dict[str, Job] = {}
        self._expired: dict[str, float] = {}  # ids dropped after keep_s: "expired", not "unknown"
        self._lock = threading.Lock()

    def running(self) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs.values() if j.state == "running"), None)

    def start(self, kind: str, device: str | None = None, seconds: float | None = None, quick: bool = False) -> Job:
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
            if quick:
                args.append("--quick")  # v0.1.28: quick triage - the remembered backend only, no mode sweep
        elif kind == "sweep":
            args = ["cameras", "test", "--seconds", f"{max(2.0, min(float(seconds or 10.0), 30.0)):g}"]
        job = Job(_new_id(), kind, self.clock())
        with self._lock:
            self._prune()
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, args), daemon=True, name=f"agentlink-job-{kind}").start()
        return job

    def _run(self, job: Job, args: list[str]) -> None:
        # v0.1.28: every check's structured result carries elapsed_s and budget_hit (whether a time limit
        # stopped the check), so a triage agent can tell a fast failure from a timed-out one.
        t0 = self.clock()
        budget_hit = False
        try:
            if job.kind == "benchmark":
                job.result = self.bench()
                job.output = [f"{k}: {v}" for k, v in job.result.items()]
            else:
                kw: dict[str, Any] = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
                if sys.platform == "win32":  # pragma: no cover
                    kw["creationflags"] = 0x08000000
                from away.multicam.diagnose import NoiseFilter

                noise = NoiseFilter()  # v0.1.22: NVIDIA Broadcast's [VCAMDS] stderr lines flooded the output
                proc = self.runner(self.command(args), **kw)
                deadline = time.time() + 600
                assert proc.stdout is not None
                for line in proc.stdout:
                    shown = noise.feed(line.rstrip())
                    if shown is not None:
                        job.output.append(shown)
                    if time.time() > deadline:
                        proc.kill()
                        budget_hit = True
                        job.output.append("(stopped after 10 minutes)")
                        break
                rc = proc.wait(timeout=30)
                if noise.summary():
                    job.output.append(noise.summary())
                job.result = {"exit_code": rc}
                if noise.hidden():
                    job.result["driver_messages_hidden"] = noise.hidden()
                budget_hit = budget_hit or any("stopped after" in ln and "(time limit)" in ln for ln in job.output)
            job.result["elapsed_s"] = round(self.clock() - t0, 1)
            job.result["budget_hit"] = budget_hit
            self.out_dir.mkdir(parents=True, exist_ok=True)
            f = self.out_dir / f"cameras-{job.kind}-{time.strftime('%Y%m%d-%H%M%S')}.txt"
            f.write_text("\n".join(job.output) + "\n", encoding="utf-8")
            job.file = f"diagnostics/{f.name}"
            job.finished = self.clock()
            job.state = "done"
        except Exception as exc:
            job.output.append(f"failed: {exc}")
            # v0.1.28: the result contract holds on the failure path too (CodeRabbit)
            job.result["elapsed_s"] = round(self.clock() - t0, 1)
            job.result["budget_hit"] = budget_hit
            job.finished = self.clock()
            job.state = "failed"

    def _prune(self) -> None:
        """Drop finished checks older than ``keep_s`` (``keep_unread_s`` when no agent has read them yet); never a
        running one. Called with the lock held."""
        now = self.clock()
        for k, j in list(self.jobs.items()):
            if j.state == "running" or j.finished is None:
                continue
            age = now - j.finished
            if age > (self.keep_s if j.fetched else max(self.keep_s, self.keep_unread_s)):
                self.jobs.pop(k, None)
                self._expired[k] = now
        while len(self.jobs) > MAX_JOBS:
            old = next((k for k, j in self.jobs.items() if j.state != "running"), None)
            if old is None:
                break
            self.jobs.pop(old, None)
            self._expired[old] = now
        for k in list(self._expired)[:-500]:
            self._expired.pop(k, None)

    def get(self, job_id: Any) -> Job | None:
        """The check ``job_id`` (an int is accepted: ``?job_id=82557239`` used to arrive as a number and never
        matched the string id), or None. Reading a finished check marks it fetched."""
        key = normalize_id(job_id)
        with self._lock:
            self._prune()
            j = self.jobs.get(key)
            if j is not None and j.state != "running":
                j.fetched = True
            return j

    def lookup(self, job_id: Any) -> Job:
        """Like :meth:`get` but raises :class:`JobLookupError` saying ``unknown`` or ``expired``."""
        j = self.get(job_id)
        if j is not None:
            return j
        key = normalize_id(job_id)
        with self._lock:
            reason = "expired" if key in self._expired else "unknown"
        raise JobLookupError(key, reason)


def normalize_id(job_id: Any) -> str:
    """A check id as given by an agent: ``"82557239"``, ``82557239`` (a query string number), ``" 8255 "``."""
    if isinstance(job_id, bool) or job_id is None:
        return ""
    if isinstance(job_id, float) and job_id.is_integer():
        job_id = int(job_id)
    return str(job_id).strip().lower()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


_shared: dict[str, JobRunner] = {}
_shared_lock = threading.Lock()


def shared_runner(out_dir: Path, command: Callable[[list[str]], list[str]] | None = None) -> JobRunner:
    """One :class:`JobRunner` per diagnostics folder in this process (v0.1.21): every REST/MCP request and every
    core built for the same data folder sees the same checks, so a check started by one request is found by the
    next one."""
    key = str(Path(out_dir).resolve())
    with _shared_lock:
        r = _shared.get(key)
        if r is None or (command is not None and r.command is not command):
            r = JobRunner(out_dir, command=command)
            _shared[key] = r
        return r
