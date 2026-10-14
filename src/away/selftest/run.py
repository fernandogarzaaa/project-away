"""Run the self-test (v0.1.24): every check with a time limit, one report, saved as the latest.

``run(env)`` never raises: a check that crashes or takes longer than its limit becomes a plain "could not run"
line. The latest report is kept at ``<data>/selftest/latest.json`` (already redacted: it is what agents read
through ``away://selftest/latest`` and ``GET /v1/selftest/latest``).
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from away.selftest import checks as ck
from away.selftest.model import FAIL, WARN, Fix, Report, Result

LIMITS_S = {"cameras": 60.0, "stt": 45.0, "tts": 60.0, "link": 30.0, "load": 15.0}
DEFAULT_LIMIT_S = 20.0
_lock = threading.Lock()  # one self-test at a time per process


class Busy(RuntimeError):
    pass


def folder(data_dir: Path | None = None) -> Path:
    from away import paths

    return Path(data_dir or paths.data_dir()) / "selftest"


def _call_with_limit(fn: Callable[[], list[Result]], limit: float) -> list[Result]:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["out"] = fn()
        except BaseException as exc:  # a broken native library can raise anything
            box["err"] = exc

    th = threading.Thread(target=target, daemon=True, name="selftest-check")
    th.start()
    th.join(limit)
    if th.is_alive():
        raise TimeoutError(f"took longer than {limit:.0f} s")
    if "err" in box:
        raise box["err"]
    return list(box.get("out") or [])


def run(
    env: Any,
    only: Iterable[str] | None = None,
    trigger: str = "you",
    progress: Callable[[str, int, int], Any] | None = None,
    limits: dict[str, float] | None = None,
    clock: Callable[[], float] = time.time,
    checks: tuple[tuple[str, str, Callable[[Any], list[Result]]], ...] | None = None,
    kind: str = "away.selftest",
) -> Report:
    from away import __version__

    if not _lock.acquire(blocking=False):
        raise Busy("a self-test is already running")
    try:
        lim = {**LIMITS_S, **(limits or {})}
        table = ck.CHECKS if checks is None else checks  # v0.1.24: "Check my PC" runs its own list through here
        want = set(only or [c[0] for c in table])
        todo = [c for c in table if c[0] in want]
        rep = Report(__version__, _platform(), clock(), trigger=trigger, id=uuid.uuid4().hex[:10], kind=kind)
        for i, (cid, label, fn) in enumerate(todo):
            if progress is not None:
                with contextlib.suppress(Exception):
                    progress(label, i, len(todo))
            t0 = time.monotonic()
            try:
                res = _call_with_limit(functools.partial(fn, env), float(lim.get(cid, DEFAULT_LIMIT_S)))
            except Exception as exc:
                status = WARN if isinstance(exc, TimeoutError) else FAIL
                res = [
                    Result(
                        cid,
                        label,
                        status,
                        f"This check could not run ({type(exc).__name__}: {str(exc)[:160]}).",
                        {},
                        Fix(f"rerun:{cid}", "Try this check again"),
                    )
                ]
            ms = (time.monotonic() - t0) * 1000.0
            for r in res:
                r.ms = ms / max(1, len(res))
            rep.results.extend(res)
        rep.finished = clock()
        if progress is not None:
            with contextlib.suppress(Exception):
                progress("done", len(todo), len(todo))
        return rep
    finally:
        _lock.release()


def _platform() -> str:
    import platform

    try:
        return f"{platform.system()} {platform.release()}".strip() or sys.platform
    except Exception:
        return sys.platform


def save_latest(report: dict[str, Any], data_dir: Path | None = None) -> Path:
    d = folder(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "latest.json"
    tmp = d / f"latest.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    return p


def load_latest(data_dir: Path | None = None) -> dict[str, Any] | None:
    p = folder(data_dir) / "latest.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None
