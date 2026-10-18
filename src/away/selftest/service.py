"""One place that runs the self-test and hands the report out (v0.1.27).

Used by the tray ("Run self-test"), Settings (through the running app), ``away selftest`` and agents (the
``selftest`` op, "Run checks" permission). It runs the checks, redacts the report, saves it as the latest and
delivers it to the audience the policy allows (:mod:`away.selftest.deliver`; ``audience="share_logs"``: only
agents the person let "Share my logs and recordings", the same audience as Check my PC). One self-test at a time.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from away.selftest import deliver as dl
from away.selftest import run as runmod
from away.selftest.redact import redact_report


class SelftestService:
    def __init__(
        self,
        data_dir: Path,
        env_factory: Callable[[bool], Any],
        *,
        agents: Callable[[], list[Any]] = lambda: [],
        inbox: Any = None,
        events: Any = None,
        post: Callable[[str, dict[str, Any]], Any] | None = None,
        keep_dirs: list[Path] | None = None,
        on_done: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
        background: bool = True,
        audience: str = "share_logs",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.env_factory = env_factory  # env_factory(interactive) -> Env
        self.agents = agents
        self.inbox = inbox
        self.events = events
        self.post = post
        self.keep_dirs = keep_dirs if keep_dirs is not None else [self.data_dir]
        self.on_done = on_done
        self.background = background
        self.audience = audience
        self.state: dict[str, Any] = {"running": False, "step": "", "done": 0, "total": 0, "id": ""}
        self.last_delivery: dict[str, Any] = {}
        self._done = threading.Event()
        self._done.set()

    def _progress(self, label: str, i: int, n: int) -> None:
        self.state.update(step=label, done=i, total=n)

    def run_now(self, trigger: str = "you", only: Any = None, interactive: bool = False) -> dict[str, Any]:
        """Run every check (blocking), save, deliver. Returns ``{"report": <redacted>, "delivery": {...}}``."""
        self._done.clear()
        self.state.update(running=True, step="starting", done=0, total=0, started=time.time())
        try:
            env = self.env_factory(interactive)
            rep = runmod.run(env, only=only, trigger=trigger, progress=self._progress)
            public = redact_report(rep.public(), self.keep_dirs)
            runmod.save_latest(public, self.data_dir)
            try:
                agents = list(self.agents())
            except Exception:
                agents = []
            delivery = dl.deliver(
                public,
                agents,
                inbox=self.inbox,
                events=self.events,
                post=self.post,
                background=self.background,
                audience=self.audience,
            )
            public["delivered_to"] = [r["agent"] for r in delivery.get("agents", [])]
            runmod.save_latest(public, self.data_dir)
            self.last_delivery = delivery
            self.state.update(id=public.get("id", ""), status=public.get("status"))
            if self.on_done is not None:
                with contextlib.suppress(Exception):
                    self.on_done(public, delivery)
            return {"report": public, "delivery": delivery}
        finally:
            self.state.update(running=False, step="done", finished=time.time())
            self._done.set()

    def start(self, trigger: str = "you", only: Any = None, interactive: bool = False) -> dict[str, Any]:
        """Start in the background; ``{"state": "running", ...}``. Raises :class:`runmod.Busy` when one runs."""
        if self.state.get("running"):
            raise runmod.Busy("a self-test is already running; its report will be sent when it finishes")
        self.state["running"] = True
        self._done.clear()

        def go() -> None:
            with contextlib.suppress(Exception):
                self.run_now(trigger, only, interactive)
            self.state["running"] = False
            self._done.set()

        threading.Thread(target=go, daemon=True, name="selftest").start()
        return {
            "state": "running",
            "how_to_get_it": "away://selftest/latest, GET /v1/selftest/latest or the event selftest.completed",
        }

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    def latest(self) -> dict[str, Any] | None:
        return runmod.load_latest(self.data_dir)
