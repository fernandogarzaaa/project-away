"""The transport-agnostic core of Agent Link. REST (:mod:`api`), MCP over HTTP and the stdio bridge all call
these methods with the authenticated :class:`~away.agentlink.auth.Agent`; the permission check happens here, once,
so every transport enforces exactly the same rules.

What no method can ever do: read or write an arbitrary file (only the bundles / recordings /
logs listed by :mod:`safepaths`), inject input without the "use my computer" permission and an approval on the PC,
run a shell outside the staged :meth:`run_shell` (tier-1 allowlist or per-call approval, never credential
stores), or change ``allow_shell`` / the agents' own permissions.
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
from away.agentlink import shell as sh
from away.agentlink.auth import Agent, AgentRegistry
from away.agentlink.bundle import redact_text
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
    "user.message",  # v0.1.22: the person spoke to you (only the agents it was sent to see it)
    "inbox",
    "selftest",  # v0.1.24: selftest.completed (the report reached every connected agent)
    "link",  # v0.1.24: link.moved (the public address changed)
)
MAX_RECORD_S = 120.0
# v0.1.25: a long-poll (``/v1/events``, the ``wait_for_event`` tool) answers within this time, even when the agent
# asks for longer. Cloudflare's quick tunnel ends a request that sends nothing for ~100 s (error 524), which the agent
# saw as being disconnected; an empty answer before that just means "ask again".
LONG_POLL_MAX_S = 55.0


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
        deadline = time.monotonic() + max(0.0, min(float(timeout), LONG_POLL_MAX_S))
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
        monitors: Callable[[], Any] | None = None,
        inbox: Any = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if inbox is None:
            from away.agentlink.inbox import Inbox

            inbox = Inbox()
        self.inbox = inbox  # v0.1.22 "Talk to your agent"
        # v0.1.21: the monitor layout (away.control.monitors Monitor objects), for get_screen's monitor/region; None:
        # the system's (one display the size of the screenshot when it cannot be read)
        self.monitors = monitors
        self._live_views: dict[str, dict[str, Any]] = {}  # agent id -> live screen view options
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
        # v0.1.28 "Recently": plain-language history of who looked through the camera or at the screen
        self.capture_history: deque[dict[str, Any]] = deque(maxlen=20)
        self._record_lock = threading.Lock()
        self._watchers: set[str] = set()
        self.watching = watching
        self._listening: dict[str, int] = {}  # v0.1.23: agent id -> open event waits (a live connection)
        self.selftest: Any = None  # v0.1.24: away.selftest.service.SelftestService (set by the app's host)
        self._listen_lock = threading.Lock()

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
            "inbox_unread": self.inbox.unread(agent.id, agent.name),  # v0.1.23
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
        """Log tails, with tokens, keys, passwords, e-mail addresses and the home folder hidden. Needs "Share my
        logs and recordings" (v0.1.24)."""
        from away.agentlink.bundle import redact, redact_text

        self.require(agent, perms.SHARE_LOGS)
        lines = max(1, min(int(lines), 2000))
        mem = redact(self.telemetry.log_tail(lines))
        file_lines: list[str] = []
        p = self.data_dir / "logs" / "away.log"
        if p.is_file():
            from away.agentlink.bundle import _tail

            file_lines = redact_text(_tail(p, 400_000).decode("utf-8", "replace")).splitlines()[-lines:]
        from away.app import crashlog

        # v0.1.22: crash.log too (crashes and uncaught errors; the Settings notice points at it)
        crash = [redact_text(x) for x in crashlog.tail(self.data_dir / "logs", min(lines, 400))]
        return {"app_log": file_lines, "crash_log": crash, "events": mem}

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
    def change_settings(self, agent: Agent, ops: Any = None, why: str = "", changes: Any = None) -> dict[str, Any]:
        """Apply changes with the user's policy: ask first (default), or low-risk changes at once and tell them.
        v0.1.23: ``ops`` (alias ``changes``) is a list of changes, one change, or ``{key: value}``."""
        self.require(agent, perms.TUNE)
        specs: list[Op] = []
        for o in normalize_ops(ops if ops is not None else changes):
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
        self,
        agent: Agent,
        kind: str = "cameras",
        device: str | None = None,
        seconds: float | None = None,
        quick: bool = False,
    ) -> dict[str, Any]:
        """v0.1.28: ``quick`` (kind="cameras" only) opens each camera on its remembered backend only, skipping
        the mode sweep - a fast triage answer. Every check's result carries ``elapsed_s`` and ``budget_hit``."""
        self.require(agent, perms.DIAGNOSE)
        return self.jobs.start(kind, device, seconds, quick).public()

    def get_job(self, agent: Agent, job_id: str | int | None = None, id: str | int | None = None) -> dict[str, Any]:
        """A check's state. ``id`` is accepted as an alias of ``job_id`` (v0.1.21); the error says whether the id
        is unknown or its result expired."""
        self.require(agent, perms.DIAGNOSE)
        key = job_id if job_id not in (None, "") else id
        if key in (None, ""):
            raise ValueError("give the check's id: job_id (as returned by run_diagnose)")
        return self.jobs.lookup(key).public()

    def run_selftest(self, agent: Agent, wait: bool = False, only: Any = None) -> dict[str, Any]:
        """v0.1.24: run the PC self-test (cameras, tracking, pointer, microphone, voices, link, load, models).
        The report goes to *every* connected agent (inbox, wake-up address, event selftest.completed) and is
        readable as away://selftest/latest. ``wait``: block until it is done (at most 3 minutes) and return it."""
        self.require(agent, perms.DIAGNOSE)
        st = self.selftest
        if st is None:
            raise NotAvailable("the self-test runs inside the Project Away app; start the app and try again")
        checks = [str(c) for c in only] if isinstance(only, (list, tuple)) else None
        from away.selftest.run import Busy

        try:
            started = st.start(trigger=agent.name, only=checks)
        except Busy:
            started = {"state": "running", "note": "a self-test was already running; its report will reach you too"}
        if not wait:
            return started
        st.wait(180.0)
        return {"state": "done", "report": st.latest()}

    def selftest_latest(self, agent: Agent) -> dict[str, Any]:
        """v0.1.24: the newest self-test report (redacted), or a hint to run one."""
        st = self.selftest
        rep = st.latest() if st is not None else None
        if rep is None:
            from away.selftest.run import load_latest

            rep = load_latest(self.data_dir)
        if rep is None:
            raise NotAvailable("no self-test has run yet; run one with diagnostics action=selftest")
        running = bool(st is not None and st.state.get("running"))
        return dict(rep, running_now=running)

    def record_session(self, agent: Agent, seconds: float = 10.0, note: str = "") -> dict[str, Any]:
        """Record the next ``seconds`` of landmarks + events (no video) to ``recordings/agent/<time>.awayrec``."""
        self.require(agent, perms.SHARE_LOGS)
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
        self.require(agent, perms.SHARE_LOGS)
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
        self.require(agent, perms.SHARE_LOGS)
        return safepaths.listing(self.data_dir)

    def read_file(self, agent: Agent, rel: str, max_bytes: int = 25_000_000) -> bytes:
        self.require(agent, perms.SHARE_LOGS)
        p = safepaths.resolve(self.data_dir, rel)
        if p.stat().st_size > max_bytes:
            raise ValueError("that file is too large to download")
        if p.suffix.lower() in (".log", ".txt"):  # v0.1.24: raw logs leave the PC with secrets hidden
            from away.agentlink.bundle import redact_text

            return redact_text(p.read_bytes().decode("utf-8", "replace")).encode("utf-8")
        return p.read_bytes()

    # ------------------------------------------------------------ see
    def look(self, agent: Agent, camera: str | None = None) -> tuple[bytes, dict[str, Any]]:
        """One low-resolution camera still. Needs "See my camera when I allow"; flashes an indicator each time."""
        self.require(agent, perms.SNAPSHOTS)
        frame = self.camera_frame(camera)
        if frame is None:
            raise NotAvailable("no camera picture right now (is the camera running and not paused?)")
        from away.agentlink.desktop import to_jpeg

        notice = f"{agent.name} looked through your camera"
        self._flash(notice, "camera picture")
        jpeg, scale = to_jpeg(frame, int(self._s("snapshot_max_width", 320)), 60)
        self._log_capture(notice)  # v0.1.28: only after the picture exists
        self.events.publish("look.taken", {"agent": agent.name})
        return jpeg, {"camera": camera or "primary", "scale": scale}

    def _displays(self, image_size: tuple[int, int]) -> list[Any]:
        from away.agentlink import screenshot as shot

        mons: Any = None
        try:
            if self.monitors is not None:
                mons = self.monitors()
            else:
                from away.control.monitors import enumerate_monitors

                mons = enumerate_monitors().monitors
        except Exception:
            mons = None
        ds = shot.displays(list(mons or []))
        if not ds:
            ds = [shot.Display(1, 0, 0, int(image_size[0]), int(image_size[1]), True)]
        return ds

    def get_screen(
        self,
        agent: Agent,
        elements: bool = False,
        quiet: bool = False,
        scale: float | None = None,
        monitor: int | None = None,
        region: Any = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """A screenshot (+ the focused window and optionally the UI element tree). Needs "See my screen when I
        allow"; flashes an indicator each time (``quiet`` is only used by the live view, which keeps a permanent
        indicator on screen instead).

        v0.1.21: ``scale`` 0.1-1.0 (default 0.5), ``monitor`` 1..N (one display) or ``region`` {x, y, width,
        height} in screen coordinates (zoom in); the longest side is capped at 2560 px (1600 for the live view).
        The metadata lists the monitors and the area/scale used: orient with a wide shot, then zoom."""
        from away.agentlink import screenshot as shot

        self.require(agent, perms.SCREEN)
        live = quiet and scale is None and monitor is None and region in (None, "", {})
        if live:
            opts = self._live_views.get(agent.id, {})
            scale, monitor, region = opts.get("scale"), opts.get("monitor"), opts.get("region")
        img = self.screen_grab()
        if img is None:
            raise NotAvailable("could not take a screenshot on this system")
        size = tuple(getattr(img, "size", None) or (img.shape[1], img.shape[0]))
        ds = self._displays((int(size[0]), int(size[1])))
        try:
            p = shot.plan(
                ds,
                (int(size[0]), int(size[1])),
                scale=scale,
                monitor=monitor,
                region=region,
                max_side=shot.LIVE_MAX_SIDE if quiet else shot.MAX_SIDE,
                default_scale=shot.LIVE_SCALE if quiet else shot.DEFAULT_SCALE,
            )
        except shot.ScreenRequestError as exc:
            raise ValueError(str(exc)) from None
        notice = f"{agent.name} looked at your screen"
        if not quiet or agent.id not in self._watchers:
            # v0.1.24 hardening: "quiet" only skips the notice while this agent's live view badge is on screen
            self._flash(notice, "screenshot")
        jpeg = shot.render(img, p)
        self._log_capture(notice)  # v0.1.28: after the picture exists; the badge is the indicator for quiet views
        self.control.scales[agent.id] = p.scale
        self.control.views[agent.id] = (float(p.origin[0]), float(p.origin[1]), float(p.scale))
        meta: dict[str, Any] = shot.meta(p, ds, len(jpeg))
        meta["size"] = [int(size[0]), int(size[1])]
        meta["window"] = None
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
        meta["note"] = (
            "Orient with a wide shot (scale 0.3-0.5), then zoom with monitor=N or region={x,y,width,height} "
            "(screen coordinates). " + meta.pop("to_screen")
        )
        return jpeg, meta

    def screen_watch(
        self,
        agent: Agent,
        on: bool = True,
        scale: float | None = None,
        monitor: int | None = None,
        region: Any = None,
    ) -> dict[str, Any]:
        """The live screen view started/stopped: a permanent on-screen badge shows while anyone is watching.
        v0.1.21: ``scale`` / ``monitor`` / ``region`` choose what the live view shows (same as get_screen)."""
        from away.agentlink import screenshot as shot

        self.require(agent, perms.SCREEN)
        if on:
            opts: dict[str, Any] = {}
            if scale is not None:
                opts["scale"] = shot.parse_scale(scale)
            if monitor not in (None, ""):
                n = int(monitor)  # type: ignore[arg-type]
                ds = self._displays((1, 1))
                if not any(d.index == n for d in ds):
                    raise ValueError(shot.monitor_error(n, ds))  # v0.1.22: monitors are numbered from 1
                opts["monitor"] = n
            if region not in (None, "", {}):
                r = shot.parse_region(region)
                opts["region"] = list(r) if r else None
            if opts or agent.id not in self._live_views:
                self._live_views[agent.id] = opts
            self._watchers.add(agent.id)
            try:
                shown = self.watching(agent.name, True)
            except Exception:
                shown = False
            if shown is False:
                # v0.1.24 hardening: no live view without its "watching your screen" badge on screen
                self.stop_watching(agent.id)
                raise NotAvailable(
                    "the live screen view was not started because its 'watching your screen' badge could not be "
                    "shown on the screen"
                )
        else:
            self.stop_watching(agent.id)
        return {"watching": bool(self._watchers), "view": self._live_views.get(agent.id, {})}

    def stop_watching(self, agent_id: str) -> None:
        """End this agent's live screen view (and hide the badge when nobody else is watching)."""
        self._watchers.discard(agent_id)
        self._live_views.pop(agent_id, None)
        with contextlib.suppress(Exception):
            self.watching(self._watcher_name(), bool(self._watchers))

    def _watcher_name(self) -> str:
        for aid in self._watchers:
            a = self.registry.get(aid)
            if a is not None:
                return a.name
        return ""

    def _flash(self, text: str, what: str = "picture") -> None:
        """The on-screen notice every camera still and screenshot shows. v0.1.24 hardening: it is a precondition,
        so when it cannot be shown the capture is refused (``flash`` raising or returning False)."""
        try:
            shown = self.flash(text)
        except Exception:
            shown = False
        if shown is False:
            raise NotAvailable(
                f"the {what} was not taken because the on-screen notice that tells the person could not be shown"
            )

    def _log_capture(self, text: str) -> None:
        """v0.1.28 "Recently": the history entry for a capture. Called only after the image is made, so the
        history can never claim a capture that was refused (no notice, so no capture) or that failed before a
        picture existed. The live-view badge counts as the visible indicator for badge-backed captures."""
        self.capture_history.append({"text": text, "t": time.time()})

    def user_capture(self, kind: str, jpeg: bytes, meta: dict[str, Any]) -> Capture:
        """The user pushed a capture to their agents ("hey agent, look" / the summon gesture)."""
        import uuid

        c = Capture(uuid.uuid4().hex[:8], time.time(), kind, jpeg, meta)
        self.captures.append(c)
        self._log_capture(f"You sent a {kind} picture to your agents")  # v0.1.28 "Recently"
        return c

    def capture_feed(self, limit: int = 10) -> list[dict[str, Any]]:
        """v0.1.28 "Recently": who looked through the camera or at the screen, newest first."""
        return list(reversed(self.capture_history))[: max(0, limit)]

    def last_capture(self, agent: Agent) -> tuple[bytes, dict[str, Any]]:
        for c in reversed(self.captures):
            if (c.kind == "camera" and agent.has(perms.SNAPSHOTS)) or (c.kind == "screen" and agent.has(perms.SCREEN)):
                return c.jpeg, dict(c.meta, kind=c.kind, id=c.id, t=c.t)
        raise NotAvailable("the person has not sent you a picture (or you lack the camera/screen permission)")

    # ----------------------------------------------------------- human
    def ask_user(
        self,
        agent: Agent,
        question: str | None = None,
        options: list[str] | None = None,
        timeout_s: float = 60.0,
        urgency: str = "normal",
        speak: bool = False,
        message: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        """``speak``: also read the question and its choices aloud (v0.1.21; "Let agents talk to me" must be on).
        ``message`` / ``text`` are accepted as other names for ``question`` (v0.1.22)."""
        question = _text_arg(question, message if message not in (None, "") else text, "ask_user", "question")
        res = self.prompts.ask(
            question, options, agent=agent.name, timeout_s=timeout_s, urgency=urgency, speak=bool(speak)
        )
        if speak:
            res["spoken"] = bool(self._speech_allowed())
        return res

    def notify_user(
        self,
        agent: Agent,
        text: str | None = None,
        level: str = "info",
        speak: bool = False,
        message: str | None = None,
    ) -> dict[str, Any]:
        """A message on screen ("Message from <agent>", bottom-right, ~6 s), optionally read aloud. ``shown`` and
        ``spoken`` are only true when that really happened (v0.1.21). ``message`` is accepted as another name for
        ``text`` (v0.1.22)."""
        body = _text_arg(text, message, "notify_user")
        return self.prompts.notify(body, level, speak, agent=agent.name)

    def _speech_allowed(self) -> bool:
        tts = self.prompts.tts
        fn = getattr(tts, "allowed", None)
        try:
            return tts is not None and (bool(fn()) if callable(fn) else True)
        except Exception:
            return False

    def wait_for_event(
        self, agent: Agent, since: int = 0, timeout_s: float = 30.0, topics: list[str] | None = None
    ) -> dict[str, Any]:
        """Events after ``since``. v0.1.22: an event addressed to some agents (a ``user.message``) is only seen by
        them; the wait goes on while only other agents' events arrive."""
        deadline = time.monotonic() + max(0.0, min(float(timeout_s), LONG_POLL_MAX_S))
        seq = int(since)
        tp = tuple(topics) if topics else None
        with self._listen_lock:
            self._listening[agent.id] = self._listening.get(agent.id, 0) + 1
        try:
            while True:
                evs = self.events.wait(seq, max(0.0, deadline - time.monotonic()), tp)
                mine = [_for_agent(e, agent.id) for e in evs]
                out = [e for e in mine if e is not None]
                if evs:
                    seq = max(seq, max(int(e["seq"]) for e in evs))
                if out or time.monotonic() >= deadline:
                    return {"seq": self.events.seq, "events": out}
        finally:
            with self._listen_lock:
                n = self._listening.get(agent.id, 1) - 1
                if n > 0:
                    self._listening[agent.id] = n
                else:
                    self._listening.pop(agent.id, None)

    def reachable(self, agent: Any, now: float | None = None) -> bool:
        """v0.1.23: will this agent see a message soon? It has access and contacted the PC in the last
        :data:`RECENT_S` seconds, has a live connection open (an event wait / stream), or has a wake-up address."""
        now = time.time() if now is None else now
        if not agent.active(now):
            return False
        with self._listen_lock:
            live = self._listening.get(agent.id, 0) > 0
        return live or now - float(agent.last_seen) < RECENT_S or bool(getattr(agent, "wake_url", ""))

    # ----------------------------------------------------------- inbox (v0.1.22)
    def deliver_message(self, target: str, text: str, source: str = "voice") -> dict[str, Any]:
        """The person spoke to an agent ("Hey Claude, ..." -> ``target`` = its id; "" = the default agent, or
        every connected agent when none is set). Returns ``{"sent", "label" ("Sent to Claude"), "to", "id"}``."""
        from away.agentlink import inbox as ib

        text = str(text or "").strip()
        if not text:
            return {"sent": False, "label": "", "reason": "nothing was said"}
        # v0.1.23: an agent whose link expired is still someone you can talk to; its messages wait for it
        active = self.registry.list()
        known = self.registry.known() if hasattr(self.registry, "known") else active
        pool = known if target else (active or known)
        ts = ib.resolve_targets(str(target or ""), pool, str(self._s("default_agent", "") or ""))
        if not ts:
            return {"sent": False, "label": "", "reason": "No AI agent is connected, so the message was not sent."}
        m = self.inbox.add(text, ts, source)
        self.events.publish(
            "user.message",
            {"id": m.id, "text": m.text, "source": source, "to": list(m.to_names), "_for": list(m.to)},
        )
        away = [a.name for a in ts if not self.reachable(a)]
        note = waiting_note(away)
        return {
            "sent": True,
            "label": ib.sent_label(m.to_names),
            "to": list(m.to_names),
            "id": m.id,
            "waiting": away,
            "note": note,
        }

    def inbox_list(self, agent: Agent, unread_only: bool = False) -> dict[str, Any]:
        ms = self.inbox.for_agent(agent.id, bool(unread_only), agent.name)
        return {
            "messages": [m.public(agent.id) for m in ms],
            "unread": self.inbox.unread(agent.id, agent.name),
            "wake_phrase": agent.wake_phrase(),
            "how_to_reply": "inbox action=reply with id and text (shown and read aloud)",
        }

    def inbox_read(self, agent: Agent, id: str | None = None) -> dict[str, Any]:
        if id in (None, ""):
            raise ValueError("give the message id (inbox action=list shows them)")
        m = self.inbox.get(agent.id, str(id), agent.name)
        self.inbox.mark_read(agent.id, m)
        return m.public(agent.id)

    def inbox_reply(
        self,
        agent: Agent,
        id: str | None = None,
        text: str | None = None,
        speak: bool = True,
        message: str | None = None,
    ) -> dict[str, Any]:
        """Answer a message: shown on screen ("Message from <agent>") and read aloud unless ``speak`` is false."""
        if id in (None, ""):
            raise ValueError("give the id of the message you are answering")
        body = _text_arg(text, message, "inbox reply")
        m = self.inbox.reply(agent.id, agent.name, str(id), body)
        res = self.prompts.notify(body, "info", bool(speak), agent=agent.name)
        self.events.publish("inbox.reply", {"id": m.id, "agent": agent.name, "_for": [agent.id]})
        return {"replied": True, "id": m.id, **res}

    # --------------------------------------------------------- control
    def request_control(self, agent: Agent, why: str = "", minutes: float | None = None) -> dict[str, Any]:
        self.require(agent, perms.CONTROL)
        return self.control.request(agent.id, agent.name, why, minutes)

    def resume_control(self, agent: Agent) -> dict[str, Any]:
        self.require(agent, perms.CONTROL)
        return self.control.resume(agent.id)

    def act(self, agent: Agent, action: str, **kw: Any) -> dict[str, Any]:
        self.require(agent, perms.CONTROL)
        # element names resolve through hooks.uia: the smart pointer's cache + fuzzy matcher when the app runs
        return self.control.act(agent.id, action, **kw)

    def act_sequence(self, agent: Agent, steps: Any = None) -> dict[str, Any]:
        """v0.1.27+: several computer actions in one call (``control(action="sequence", steps=[...])``): one grant
        check, one log entry, stops at the first refusal, cancel or takeover."""
        self.require(agent, perms.CONTROL)
        return self.control.act_sequence(agent.id, steps)

    def stop_control(self, agent: Agent) -> dict[str, Any]:
        if self.control.grant is not None and self.control.grant.agent_id == agent.id:
            self.control.stop("the agent finished")
        return self.control.status()

    def undo_last_action(self, agent: Agent) -> dict[str, Any]:
        self.require(agent, perms.CONTROL)
        return self.control.undo_last(agent.id)

    def action_log(self, agent: Agent, limit: int = 30) -> list[dict[str, Any]]:
        """This agent's own computer actions (v0.1.24), or everyone's when the person turned on
        ``agent_link.share_control_history``."""
        if bool(self._s("share_control_history", False)):
            return self.control.actions(limit)
        return self.control.actions(limit, agent_id=agent.id)

    # --------------------------------------------------------- staged shell (proposal #26)
    @property
    def _shell_audit(self) -> sh.ShellAudit:
        audit = getattr(self, "_shell_audit_obj", None)
        if audit is None:
            audit = sh.ShellAudit(self.data_dir / "agentlink" / "shell-audit.jsonl")
            self._shell_audit_obj = audit
        return audit

    def _shell_log_tail(self, kb: int = 32) -> str:
        p = self.data_dir / "logs" / "away.log"
        try:
            with p.open("rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - kb * 1024))
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return "(no away.log yet)"

    def run_shell(
        self,
        agent: Agent,
        action: str = "run",
        command: str | None = None,
        argv: list[str] | None = None,
        why: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Safety-gated shell with staged access (proposal #26), under "use my computer".

        - ``action="list"``: the tier-1 read-only commands (no session grant needed to look).
        - ``action="log"``: every shell invocation so far (command, tier, result summary).
        - ``action="run"`` with ``command`` = a tier-1 name: fixed, read-only, runs at once inside an
          approved computer session (no per-call approval).
        - ``action="run"`` with ``argv``: tier 2, anything else. The person is asked first, every time,
          through the same prompt path as risky clicks; only "Allow" runs it.

        Never a shell (``shell=False``), never credential stores or password material (refused outright),
        64 KB output cap, secrets redacted, every invocation audited.
        """
        if action == "list":
            self.require(agent, perms.CONTROL)
            return {
                "tier1": sh.describe(),
                "note": "tier 1 runs at once inside an approved computer session (control action=request); "
                "anything else asks the person first, every time.",
            }
        if action == "log":
            self.require(agent, perms.CONTROL)
            return {"entries": self._shell_audit.tail(limit or 30)}
        if action != "run":
            raise ValueError(f"shell: action must be run|list|log (got {action!r})")
        self.require(agent, perms.CONTROL)
        # the approved session: same per-session grant as computer control (expires, takeover pauses it)
        self.control.require_grant(agent.id)

        if command and argv:
            raise ValueError("shell: pass either command (tier 1) or argv (tier 2), not both")
        if command:
            return self._run_shell_tier1(agent, command, why)
        if argv:
            return self._run_shell_argv(agent, [str(a) for a in argv], why)
        raise ValueError("shell: action=run needs command (a tier-1 name) or argv (a command list)")

    def _run_shell_tier1(self, agent: Agent, command: str, why: str | None) -> dict[str, Any]:
        kind = sh.tier1_kind(command)  # KeyError -> unknown command (REST 404)
        audit = self._shell_audit
        if kind == "unavailable":
            audit.append(
                {
                    "agent": agent.name,
                    "agent_id": agent.id,
                    "tier": 1,
                    "command": command,
                    "argv": [],
                    "approved": False,
                    "refused": "not available on this PC",
                }
            )
            raise NotAvailable(f"shell: {command!r} is not available on this PC")
        if kind == "in_process":
            text = sh.serve_in_process(command, log_tail=self._shell_log_tail)
            text = redact_text(text)
            audit.append(
                {
                    "agent": agent.name,
                    "agent_id": agent.id,
                    "tier": 1,
                    "command": command,
                    "argv": [],
                    "approved": False,
                    "rc": 0,
                    "out_bytes": len(text.encode("utf-8")),
                    "truncated": False,
                    "summary": sh.summarize(text, 0),
                }
            )
            return {
                "tier": 1,
                "command": command,
                "approved": False,
                "rc": 0,
                "timed_out": False,
                "truncated": False,
                "stdout": text,
                "stderr": "",
            }
        fixed = sh.tier1_argv(command)
        assert fixed is not None
        return self._run_shell_exec(agent, fixed, tier=1, name=command, why=why, approved=False)

    def _run_shell_argv(self, agent: Agent, argv: list[str], why: str | None) -> dict[str, Any]:
        try:
            tier, name = sh.classify(argv)  # ShellRefused -> REST 400, before anything runs
        except sh.ShellRefused as e:
            self._shell_audit.append(
                {
                    "agent": agent.name,
                    "agent_id": agent.id,
                    "tier": 2,
                    "command": argv[0],
                    "argv": [redact_text(a) for a in argv],
                    "approved": False,
                    "refused": str(e),
                }
            )
            raise
        if tier == 2:
            details = [
                "Tier 2: anything outside the read-only tier-1 list. It runs once, only with your approval.",
                "No shell is involved: the words below go to the program exactly as written.",
            ]
            if sh.has_metachars(argv):
                details.append(
                    "It contains shell-looking characters (; | $ ...): they are passed to the program "
                    "as-is and do nothing special, but double-check the command."
                )
            if why:
                details.append(f"The agent's reason: {why[:300]}")
            res = self.prompts.ask(
                f"{agent.name} wants to run this command on your PC:\n{' '.join(argv)}",
                ["Allow", "Deny"],
                title="Run this command?",
                agent=agent.name,
                details=details,
                timeout_s=60.0,
                urgency="high",
                kind="confirm",
            )
            if res.get("answer") != "Allow":
                self._shell_audit.append(
                    {
                        "agent": agent.name,
                        "agent_id": agent.id,
                        "tier": 2,
                        "command": argv[0],
                        "argv": [redact_text(a) for a in argv],
                        "approved": False,
                        "refused": f"the user answered {res.get('answer')!r}",
                    }
                )
                raise ControlRefused(f"the user did not allow the command ({argv[0][:60]})")
            # the grant can expire, pause or be revoked while the prompt waited: re-check it now, so a
            # stale "Allow" cannot start the command without a live grant
            try:
                self.control.require_grant(agent.id)
            except ControlRefused as e:
                self._shell_audit.append(
                    {
                        "agent": agent.name,
                        "agent_id": agent.id,
                        "tier": 2,
                        "command": argv[0],
                        "argv": [redact_text(a) for a in argv],
                        "approved": True,
                        "refused": f"the computer session ended while awaiting approval ({e})",
                    }
                )
                raise
            return self._run_shell_exec(agent, argv, tier=2, name=argv[0], why=why, approved=True)
        return self._run_shell_exec(agent, argv, tier=1, name=name or argv[0], why=why, approved=False)

    def _run_shell_exec(
        self, agent: Agent, argv: list[str], *, tier: int, name: str, why: str | None, approved: bool
    ) -> dict[str, Any]:
        try:
            res = sh.run(argv)
        except sh.ShellNotAvailable as e:
            self._shell_audit.append(
                {
                    "agent": agent.name,
                    "agent_id": agent.id,
                    "tier": tier,
                    "command": name,
                    "argv": [redact_text(a) for a in argv],
                    "approved": approved,
                    "refused": f"not available on this PC ({e})",
                }
            )
            raise NotAvailable(str(e)) from None
        stdout, stderr = redact_text(res["stdout"]), redact_text(res["stderr"])
        self._shell_audit.append(
            {
                "agent": agent.name,
                "agent_id": agent.id,
                "tier": tier,
                "command": name,
                "argv": [redact_text(a) for a in argv],
                "approved": approved,
                "rc": res["rc"],
                "timed_out": res["timed_out"],
                "truncated": res["truncated"],
                "out_bytes": res["out_bytes"],
                "summary": sh.summarize(stdout, res["rc"]),
            }
        )
        return {
            "tier": tier,
            "command": name,
            "approved": approved,
            "rc": res["rc"],
            "timed_out": res["timed_out"],
            "truncated": res["truncated"],
            "stdout": stdout,
            "stderr": stderr,
        }

    # --------------------------------------------------------- session
    def disconnect(self, agent: Agent) -> dict[str, Any]:
        self.registry.revoke(agent.id)
        self.reconcile(agent.id, "the agent disconnected")
        return {"disconnected": True}

    def set_scopes(self, agent_id: str, scopes: Any) -> Agent:
        """The person changed this agent's permissions (Settings > Agent Link): applied to what is running now."""
        a = self.registry.set_scopes(agent_id, scopes)
        self.reconcile(agent_id, "you turned off its permission to use your computer")
        return a

    def revoke(self, agent_id: str) -> bool:
        done = self.registry.revoke(agent_id)
        self.reconcile(agent_id, "the agent was disconnected")
        return done

    def revoke_token(self, token_id: str) -> bool:
        """v0.1.24: the person revoked a persistent agent token. The agent is disconnected and everything it had
        running ends at once (control grant, held buttons and keys, live views), as for any disconnect."""
        t = self.registry.tokens.get(token_id)
        agent_id = t.agent_id if t is not None else ""
        done = self.registry.revoke_persistent(token_id)
        if agent_id:
            self.reconcile(agent_id, "its agent token was revoked")
        return done

    def rotate_token(self, token_id: str) -> tuple[Any, str]:
        """v0.1.24: a persistent agent token was replaced (by the person, or by the agent itself). The old token
        keeps signing in for the grace hour, but nothing it started carries over: the control grant ends (input is
        released) and live views stop, so the new token starts clean and asks again."""
        t, secret = self.registry.rotate_persistent(token_id)
        self.reconcile(t.agent_id, "its agent token was replaced", end_sessions=True)
        return t, secret

    def reconcile(self, agent_id: str, reason: str = "its access changed", end_sessions: bool = False) -> None:
        """v0.1.24 hardening: live state follows the agent's permissions at once. Without "Let it use my computer"
        (or without access at all) its control session ends (border off, any action in progress stops and lets go
        of buttons and keys); without "See my screen" its live view stops and its view state is cleared."""
        a = self.registry.get(agent_id)
        active = a is not None and a.active(time.time())
        can_control = a is not None and active and a.has(perms.CONTROL)
        can_see = a is not None and active and a.has(perms.SCREEN)
        if end_sessions:  # v0.1.24: a token rotated out ends what was running under it, whatever the permissions
            can_control = can_see = False
        if not can_control:
            self.control.revoke_agent(agent_id, reason if active else "the agent was disconnected")
        if not can_see:
            if agent_id in self._watchers or agent_id in self._live_views:
                self.stop_watching(agent_id)
            self.control.views.pop(agent_id, None)
            self.control.scales.pop(agent_id, None)


def normalize_ops(ops: Any) -> list[dict[str, Any]]:
    """v0.1.23: the forms an agent may send to settings change: ``[{"key": ..., "value": ...}, ...]``, one
    ``{"key": ..., "value": ...}``, or ``{"hands.cursor.gain": 1.8, ...}``. Anything else is a plain ChangeError."""
    hint = 'send changes as a list like [{"key": "hands.cursor.gain", "value": 1.8}] or as {"hands.cursor.gain": 1.8}'
    if ops is None or ops == [] or ops == {}:
        raise ChangeError("nothing to change; " + hint)
    if isinstance(ops, str):
        import json

        try:
            ops = json.loads(ops)
        except ValueError:
            raise ChangeError(hint) from None
    if isinstance(ops, dict):
        if "key" in ops or "domain" in ops or "op" in ops:
            return [ops]
        return [{"domain": "settings", "op": "set", "key": str(k), "value": v} for k, v in ops.items()]
    if isinstance(ops, (list, tuple)):
        out = []
        for i, o in enumerate(ops):
            if not isinstance(o, dict):
                raise ChangeError(f"change {i + 1} is not an object; " + hint)
            out.append(o)
        return out
    raise ChangeError(hint)


RECENT_S = 120.0  # v0.1.23: an agent that contacted the PC this recently counts as connected for messages


def waiting_note(names: list[str]) -> str:
    """ "Hark isn't connected right now; I'll keep your message until it checks in." ("" when all are here)."""
    ns = [n for n in names if n]
    if not ns:
        return ""
    if len(ns) == 1:
        return f"{ns[0]} isn't connected right now; I'll keep your message until it checks in."
    who = ", ".join(ns[:-1]) + " and " + ns[-1]
    return f"{who} aren't connected right now; I'll keep your message until they check in."


def _for_agent(ev: dict[str, Any], agent_id: str) -> dict[str, Any] | None:
    """The event as this agent may see it (None when it is addressed to other agents only)."""
    data = ev.get("data") or {}
    targets = data.get("_for")
    if targets is None:
        return ev
    if agent_id not in targets:
        return None
    return dict(ev, data={k: v for k, v in data.items() if k != "_for"})


def _text_arg(primary: Any, alias: Any, tool: str, name: str = "text") -> str:
    """v0.1.22: agents send ``message`` instead of ``text`` (or ``question``); accept either, refuse neither."""
    val = primary if primary not in (None, "") else alias
    if val in (None, ""):
        raise ValueError(f"{tool} needs the words to show: {name} (or message)")
    return str(val)


__all__ = ["ControlRefused", "EventHub", "Forbidden", "NotAvailable", "Service"]
