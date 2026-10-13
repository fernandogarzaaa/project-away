"""The transport-agnostic core of Agent Link. REST (:mod:`api`), MCP over HTTP and the stdio bridge all call
these methods with the authenticated :class:`~away.agentlink.auth.Agent`; the permission check happens here, once,
so every transport enforces exactly the same rules.

What no method can ever do: run a shell command, read or write an arbitrary file (only the bundles / recordings /
logs listed by :mod:`safepaths`), inject input without the "use my computer" permission and an approval on the PC,
or change ``allow_shell`` / the agents' own permissions.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from away.agentlink import about, perms, safepaths
from away.agentlink.auth import Agent, AgentRegistry
from away.agentlink.changes import ChangeManager
from away.agentlink.control import ComputerControl, ControlRefused
from away.agentlink.jobs import JobRunner
from away.agentlink.prefs import Preferences
from away.agentlink.prompts import PromptManager
from away.agentlink.recording import RingRecorder
from away.agentlink.registry import ChangeError, Op, preset_ops
from away.agentlink.telemetry import Telemetry

EVENT_TOPICS = (
    "presence",
    "attention",
    "summon",
    "user.look",
    "user.screen",
    "wrong",
    "detector",
    "misfire",
    "control",
    "prompt",
    "change",
    "notify",
)
MAX_RECORD_S = 120.0


class Forbidden(PermissionError):
    """The agent lacks a permission; the message names it in plain words."""


class NotAvailable(RuntimeError):
    pass


class EventHub:
    """Events agents can wait for / subscribe to. Waiters block on a condition (long-poll)."""

    def __init__(self, size: int = 500) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=size)
        self._seq = 0
        self._cond = threading.Condition()
        self.listeners: list[Callable[[dict[str, Any]], None]] = []

    def publish(self, topic: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._cond:
            self._seq += 1
            ev = {"seq": self._seq, "t": time.time(), "topic": topic, "data": dict(data or {})}
            self._events.append(ev)
            self._cond.notify_all()
        for fn in list(self.listeners):
            with contextlib.suppress(Exception):
                fn(ev)
        return ev

    @property
    def seq(self) -> int:
        return self._seq

    def since(self, seq: int, topics: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        with self._cond:
            evs = [e for e in self._events if e["seq"] > seq]
        if topics:
            evs = [e for e in evs if any(e["topic"] == t or e["topic"].startswith(t + ".") for t in topics)]
        return evs

    def wait(self, seq: int, timeout: float, topics: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, min(float(timeout), 120.0))
        while True:
            evs = self.since(seq, topics)
            left = deadline - time.monotonic()
            if evs or left <= 0:
                return evs
            with self._cond:
                self._cond.wait(min(left, 1.0))


@dataclass
class Capture:
    id: str
    t: float
    kind: str  # camera | screen
    jpeg: bytes
    meta: dict[str, Any] = field(default_factory=dict)


class Service:
    def __init__(
        self,
        *,
        data_dir: Path,
        registry: AgentRegistry,
        telemetry: Telemetry,
        prompts: PromptManager,
        changes: ChangeManager,
        prefs: Preferences,
        control: ComputerControl,
        jobs: JobRunner,
        recorder: RingRecorder,
        cfg: Callable[[], Any],
        settings: Callable[[], Any],
        camera_frame: Callable[[str | None], Any] = lambda _cam: None,
        screen_grab: Callable[[], Any] | None = None,
        flash: Callable[[str], None] = lambda _text: None,
        foreground: Callable[[], dict[str, Any] | None] = lambda: None,
        uia: Any = None,
        record_header: Callable[[], dict[str, Any]] | None = None,
        events: EventHub | None = None,
        watching: Callable[[str, bool], None] = lambda _name, _on: None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.registry = registry
        self.telemetry = telemetry
        self.prompts = prompts
        self.changes = changes
        self.prefs = prefs
        self.control = control
        self.jobs = jobs
        self.recorder = recorder
        self.cfg = cfg
        self.settings = settings
        self.camera_frame = camera_frame
        if screen_grab is None:
            from away.agentlink.desktop import grab_screen

            screen_grab = grab_screen
        self.screen_grab = screen_grab
        self.flash = flash
        self.foreground = foreground
        if uia is None:
            from away.agentlink.desktop import NullUIA

            uia = NullUIA()
        self.uia = uia
        self.record_header = record_header
        self.events = events or EventHub()
        self.started = time.time()
        self.captures: deque[Capture] = deque(maxlen=4)
        self._record_lock = threading.Lock()
        self._watchers: set[str] = set()
        self.watching = watching

    # ------------------------------------------------------------- helpers
    @staticmethod
    def require(agent: Agent, scope: str) -> None:
        if not agent.has(scope):
            label = perms.BY_SCOPE[scope].label if scope in perms.BY_SCOPE else scope
            raise Forbidden(f"this needs the permission '{label}'; ask the person to turn it on in Agent Link")

    def _s(self, name: str, default: Any) -> Any:
        try:
            return getattr(self.settings(), name)
        except Exception:
            return default

    # ---------------------------------------------------------------- read
    def get_status(self, agent: Agent) -> dict[str, Any]:
        from away import __version__

        return {
            "app": "Project Away",
            "version": __version__,
            "uptime_s": int(time.time() - self.started),
            "agent": agent.public(),
            "health": self.telemetry.health(),
            "context": self.telemetry.context(),
            "control": self.control.status(),
            "autofix_mode": self._s("autofix", "ask"),
            "pending_prompts": len(self.prompts.pending()),
            "event_seq": self.events.seq,
        }

    def get_context(self, agent: Agent) -> dict[str, Any]:
        ctx = self.telemetry.context()
        ctx["control"] = self.control.status()
        ctx["polite_moment"] = self.telemetry.polite_moment()
        return ctx

    def about(self, agent: Agent) -> str:
        return about.briefing(agent, self.telemetry.context(), self.control.status(), str(self._s("autofix", "ask")))

    def poll(self, agent: Agent, since: int = 0, landmarks: bool = True) -> dict[str, Any]:
        data = self.telemetry.poll(since)
        if not landmarks:
            data.pop("landmarks", None)
        data["health"] = self.telemetry.health()
        return data

    def health(self, agent: Agent) -> dict[str, Any]:
        return self.telemetry.health()

    def get_logs(self, agent: Agent, lines: int = 100) -> dict[str, Any]:
        lines = max(1, min(int(lines), 2000))
        mem = self.telemetry.log_tail(lines)
        file_lines: list[str] = []
        p = self.data_dir / "logs" / "away.log"
        if p.is_file():
            from away.agentlink.bundle import _tail

            file_lines = _tail(p, 400_000).decode("utf-8", "replace").splitlines()[-lines:]
        return {"app_log": file_lines, "events": mem}

    def get_config(self, agent: Agent) -> dict[str, Any]:
        from away import config as cfgmod
        from away.agentlink.bundle import redact

        cfg = self.cfg()
        d = redact(cfgmod.to_dict(cfg))
        d.pop("agent_link", None)
        return {"config": d, "active_profile": cfg.bindings.active_profile}

    def list_settings(self, agent: Agent, domain: str | None = None) -> dict[str, Any]:
        from away.agentlink.registry import domains

        doms = domains(self.changes.presets)
        cfg = self.cfg()
        if domain:
            d = doms.get(domain)
            if d is None:
                raise ValueError(f"unknown kind {domain!r}; use {', '.join(doms)}")
            return {
                "domain": d.name,
                "title": d.title,
                "available": d.available,
                "note": d.note,
                "ops": list(d.ops),
                "items": d.list(cfg) if d.available else [],
            }
        return {
            "domains": [
                {"name": d.name, "title": d.title, "available": d.available, "ops": list(d.ops), "note": d.note}
                for d in doms.values()
            ]
        }

    # ---------------------------------------------------------------- tune
    def change_settings(self, agent: Agent, ops: list[dict[str, Any]], why: str = "") -> dict[str, Any]:
        """Apply changes with the user's policy: ask first (default), or low-risk changes at once and tell them."""
        self.require(agent, perms.TUNE)
        specs: list[Op] = []
        for o in ops:
            op = Op.from_dict(o)
            if op.domain == "presets" and op.op == "apply":
                specs.extend(preset_ops(self.cfg(), op.key, self.changes.presets))
            else:
                specs.append(op)
        if not specs:
            raise ChangeError("nothing to change")
        preview = self.changes.preview(specs)
        mode = str(self._s("autofix", "ask"))
        if mode == "auto_low_risk" and preview["low_risk"]:
            res = self.changes.apply(specs, who=agent.name, agent_id=agent.id, why=why)
            self.prompts.notify(
                f"{agent.name} changed: " + "; ".join(preview["changes"])[:200] + " (undo in Agent Link)"
            )
            self.events.publish("change.applied", {"agent": agent.name, "changes": preview["changes"], "asked": False})
            return {"applied": True, "asked": False, **res}
        details = list(preview["changes"])[:8] + ([f"Why: {why[:150]}"] if why else [])
        ans = self.prompts.ask(
            f"{agent.name} wants to change {len(preview['changes'])} setting(s)",
            ["Yes", "No"],
            title="Change settings?",
            agent=agent.name,
            details=details,
            timeout_s=90.0,
            urgency="normal",
            kind="confirm",
        )
        if ans.get("answer") != "Yes":
            self.events.publish("change.declined", {"agent": agent.name, "changes": preview["changes"]})
            return {
                "applied": False,
                "asked": True,
                "answer": ans.get("answer"),
                "method": ans.get("method"),
                "changes": preview["changes"],
            }
        res = self.changes.apply(specs, who=agent.name, agent_id=agent.id, why=why)
        self.events.publish("change.applied", {"agent": agent.name, "changes": preview["changes"], "asked": True})
        return {"applied": True, "asked": True, "method": ans.get("method"), **res}

    def undo_changes(self, agent: Agent, what: str = "last") -> dict[str, Any]:
        self.require(agent, perms.TUNE)
        if what == "last":
            return self.changes.undo_last(who=f"undo by {agent.name}")
        if what == "today":
            return self.changes.undo_agent_today(agent.id, who=f"undo by {agent.name}")
        try:
            return self.changes.undo(what, who=f"undo by {agent.name}")
        except ChangeError:
            return self.changes.undo_batch(what, who=f"undo by {agent.name}")

    def change_history(self, agent: Agent, limit: int = 30) -> list[dict[str, Any]]:
        return self.changes.history(limit)

    def get_preferences(self, agent: Agent) -> dict[str, Any]:
        return {"text": self.prefs.text(), "items": self.prefs.items()}

    def add_preference(self, agent: Agent, text: str) -> dict[str, Any]:
        self.require(agent, perms.TUNE)
        return {"items": self.prefs.add(text, who=agent.name)}

    def remove_preference(self, agent: Agent, number: int) -> dict[str, Any]:
        self.require(agent, perms.TUNE)
        return {"items": self.prefs.remove(int(number))}

    # ------------------------------------------------------------ diagnose
    def run_diagnose(
        self, agent: Agent, kind: str = "cameras", device: str | None = None, seconds: float | None = None
    ) -> dict[str, Any]:
        self.require(agent, perms.DIAGNOSE)
        return self.jobs.start(kind, device, seconds).public()

    def get_job(self, agent: Agent, job_id: str) -> dict[str, Any]:
        self.require(agent, perms.DIAGNOSE)
        j = self.jobs.get(job_id)
        if j is None:
            raise KeyError(f"no check {job_id!r}")
        return j.public()

    def record_session(self, agent: Agent, seconds: float = 10.0, note: str = "") -> dict[str, Any]:
        """Record the next ``seconds`` of landmarks + events (no video) to ``recordings/agent/<time>.awayrec``."""
        self.require(agent, perms.DIAGNOSE)
        from away.agentlink import recording as rec

        seconds = max(1.0, min(float(seconds), MAX_RECORD_S))
        if not self._record_lock.acquire(blocking=False):
            raise RuntimeError("a recording is already running")
        try:
            self.recorder.start_forward(seconds)
            time.sleep(seconds)
            records = self.recorder.take_forward()
        finally:
            self._record_lock.release()
        header = self.record_header() if self.record_header else rec.make_header()
        header["note"] = f"recorded by {agent.name}" + (f": {note[:200]}" if note else "")
        name = f"{time.strftime('%Y%m%d-%H%M%S')}{rec.SUFFIX}"
        path = rec.write(self.data_dir / "recordings" / "agent" / name, header, records)
        frames = sum(1 for r in records if r.get("k") == "h")
        return {"path": f"recordings/agent/{name}", "frames": frames, "bytes": path.stat().st_size, "seconds": seconds}

    def get_bundle(self, agent: Agent, note: str = "") -> dict[str, Any]:
        self.require(agent, perms.DIAGNOSE)
        from away.agentlink import bundle

        stats = {
            "health": self.telemetry.health(),
            "context": self.telemetry.context(),
            "events": self.telemetry.poll(0, 100)["events"],
        }
        header = self.record_header() if self.record_header else None
        snap = self.recorder.snapshot()
        path = bundle.build(
            self.data_dir,
            self.cfg(),
            stats,
            (header or {}, snap) if header else None,
            note=f"requested by {agent.name}" + (f": {note}" if note else ""),
        )
        return {"path": f"diagnostics/{path.name}", "bytes": path.stat().st_size}

    def list_files(self, agent: Agent) -> list[dict[str, Any]]:
        self.require(agent, perms.DIAGNOSE)
        return safepaths.listing(self.data_dir)

    def read_file(self, agent: Agent, rel: str, max_bytes: int = 25_000_000) -> bytes:
        self.require(agent, perms.DIAGNOSE)
        p = safepaths.resolve(self.data_dir, rel)
        if p.stat().st_size > max_bytes:
            raise ValueError("that file is too large to download")
        return p.read_bytes()

    # ------------------------------------------------------------ see
    def look(self, agent: Agent, camera: str | None = None) -> tuple[bytes, dict[str, Any]]:
        """One low-resolution camera still. Needs "See my camera when I allow"; flashes an indicator each time."""
        self.require(agent, perms.SNAPSHOTS)
        frame = self.camera_frame(camera)
        if frame is None:
            raise NotAvailable("no camera picture right now (is the camera running and not paused?)")
        from away.agentlink.desktop import to_jpeg

        self._flash(f"{agent.name} looked through your camera")
        jpeg, scale = to_jpeg(frame, int(self._s("snapshot_max_width", 320)), 60)
        self.events.publish("look.taken", {"agent": agent.name})
        return jpeg, {"camera": camera or "primary", "scale": scale}

    def get_screen(self, agent: Agent, elements: bool = False, quiet: bool = False) -> tuple[bytes, dict[str, Any]]:
        """One low-resolution screenshot (+ focused window and optionally the UI element tree). Needs "See my screen
        when I allow"; flashes an indicator each time (``quiet`` is only used by the live view, which keeps a
        permanent indicator on screen instead)."""
        self.require(agent, perms.SCREEN)
        img = self.screen_grab()
        if img is None:
            raise NotAvailable("could not take a screenshot on this system")
        from away.agentlink.desktop import to_jpeg

        if not quiet:
            self._flash(f"{agent.name} looked at your screen")
        jpeg, scale = to_jpeg(img, int(self._s("screen_max_width", 960)), 55)
        self.control.scales[agent.id] = scale
        meta: dict[str, Any] = {"scale": scale, "size": list(getattr(img, "size", (0, 0))), "window": None}
        try:
            meta["window"] = self.foreground()
        except Exception:
            meta["window"] = None
        if elements:
            if getattr(self.uia, "available", False):
                tree = self.uia.tree(3, 150)
                meta["elements"] = tree.to_dict(3) if tree is not None else None
            else:
                meta["elements"] = None
                meta["elements_note"] = getattr(self.uia, "reason", "not available")
        meta["note"] = "image coordinates / scale = screen coordinates (or pass from_image=true to actions)"
        return jpeg, meta

    def screen_watch(self, agent: Agent, on: bool = True) -> dict[str, Any]:
        """The live screen view started/stopped: a permanent on-screen badge shows while anyone is watching."""
        self.require(agent, perms.SCREEN)
        if on:
            self._watchers.add(agent.id)
        else:
            self._watchers.discard(agent.id)
        with contextlib.suppress(Exception):
            self.watching(agent.name if self._watchers else "", bool(self._watchers))
        return {"watching": bool(self._watchers)}

    def _flash(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self.flash(text)

    def user_capture(self, kind: str, jpeg: bytes, meta: dict[str, Any]) -> Capture:
        """The user pushed a capture to their agents ("hey agent, look" / the summon gesture)."""
        import uuid

        c = Capture(uuid.uuid4().hex[:8], time.time(), kind, jpeg, meta)
        self.captures.append(c)
        return c

    def last_capture(self, agent: Agent) -> tuple[bytes, dict[str, Any]]:
        for c in reversed(self.captures):
            if (c.kind == "camera" and agent.has(perms.SNAPSHOTS)) or (c.kind == "screen" and agent.has(perms.SCREEN)):
                return c.jpeg, dict(c.meta, kind=c.kind, id=c.id, t=c.t)
        raise NotAvailable("the person has not sent you a picture (or you lack the camera/screen permission)")

    # ----------------------------------------------------------- human
    def ask_user(
        self,
        agent: Agent,
        question: str,
        options: list[str] | None = None,
        timeout_s: float = 60.0,
        urgency: str = "normal",
    ) -> dict[str, Any]:
        return self.prompts.ask(question, options, agent=agent.name, timeout_s=timeout_s, urgency=urgency)

    def notify_user(self, agent: Agent, text: str, level: str = "info", speak: bool = False) -> dict[str, Any]:
        return self.prompts.notify(f"{agent.name}: {text}", level, speak)

    def wait_for_event(
        self, agent: Agent, since: int = 0, timeout_s: float = 30.0, topics: list[str] | None = None
    ) -> dict[str, Any]:
        evs = self.events.wait(int(since), float(timeout_s), tuple(topics) if topics else None)
        return {"seq": self.events.seq, "events": evs}

    # --------------------------------------------------------- control
    def request_control(self, agent: Agent, why: str = "", minutes: float | None = None) -> dict[str, Any]:
        self.require(agent, perms.CONTROL)
        return self.control.request(agent.id, agent.name, why, minutes)

    def resume_control(self, agent: Agent) -> dict[str, Any]:
        self.require(agent, perms.CONTROL)
        return self.control.resume(agent.id)

    def act(self, agent: Agent, action: str, **kw: Any) -> dict[str, Any]:
        self.require(agent, perms.CONTROL)
        # TODO(merge feat/smart-pointer): resolve element names with the smart pointer's UIA matcher when merged
        return self.control.act(agent.id, action, **kw)

    def stop_control(self, agent: Agent) -> dict[str, Any]:
        if self.control.grant is not None and self.control.grant.agent_id == agent.id:
            self.control.stop("the agent finished")
        return self.control.status()

    def undo_last_action(self, agent: Agent) -> dict[str, Any]:
        self.require(agent, perms.CONTROL)
        return self.control.undo_last(agent.id)

    def action_log(self, agent: Agent, limit: int = 30) -> list[dict[str, Any]]:
        return self.control.actions(limit)

    # --------------------------------------------------------- session
    def disconnect(self, agent: Agent) -> dict[str, Any]:
        self.control.revoke_agent(agent.id)
        self.registry.revoke(agent.id)
        return {"disconnected": True}


__all__ = ["ControlRefused", "EventHub", "Forbidden", "NotAvailable", "Service"]
