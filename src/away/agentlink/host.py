"""Agent Link inside the running app (and the standalone ``away agent start`` host).

:func:`build_core` assembles the parts (registry, telemetry, prompts, changes, control, service) without devices,
so tests and the CLI use the same wiring as the app. :class:`AgentLinkHost` attaches that core to an
:class:`~away.app.runner.AwayApp`: it watches the event bus, wraps ``Pipeline.on_hand`` / ``on_faces`` (recording,
hand-sign answers, the open-palm stop), registers the ``agent:`` binding hook, runs the local server and the
tunnel on demand, and talks to Settings / the CLI / ``away mcp`` through small files in ``<data>/agentlink``:

* ``requests/*.json`` - what Settings or the CLI asks for (pair, disconnect, revoke, answer a prompt, undo...)
* ``state.json`` - what Settings shows (agents, pairing link, prompts, changes, actions, health), owner-only
* ``host.json`` - port + heartbeat for ``away mcp``; ``local.key`` - the secret that proves "same user, same PC"
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away import paths
from away.agentlink import handsigns, perms
from away.agentlink.auth import AgentRegistry
from away.agentlink.changes import ChangeManager
from away.agentlink.control import ComputerControl, ControlHooks
from away.agentlink.detectors import Detectors
from away.agentlink.jobs import JobRunner, shared_runner
from away.agentlink.prefs import Preferences
from away.agentlink.prompts import NullUI, PromptManager, normalize
from away.agentlink.recording import RingRecorder
from away.agentlink.service import EventHub, Service
from away.agentlink.telemetry import Telemetry

STOP_WORDS = ("stop", "stop it", "stop now", "agent stop", "stop agent")
STATE_EVERY_S = 1.0
HEALTH_WANT_S = 5.0


def base_dir(data_dir: Path | None = None) -> Path:
    return Path(data_dir or paths.data_dir()) / "agentlink"


def write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def local_key(base: Path) -> str:
    p = base / "local.key"
    try:
        k = p.read_text(encoding="utf-8").strip()
        if len(k) >= 32:
            return k
    except OSError:
        pass
    k = secrets.token_urlsafe(32)
    write_private(p, k)
    return k


def send_request(action: str, data_dir: Path | None = None, **data: Any) -> str:
    """Settings / CLI -> the running app. Returns the request id (its result appears in state.json)."""
    rid = uuid.uuid4().hex[:10]
    write_private(
        base_dir(data_dir) / "requests" / f"{time.time():.3f}-{rid}.json",
        json.dumps({"id": rid, "action": action, **data}),
    )
    return rid


def read_state(data_dir: Path | None = None) -> dict[str, Any]:
    try:
        return dict(json.loads((base_dir(data_dir) / "state.json").read_text(encoding="utf-8")))
    except Exception:
        return {}


def host_alive(data_dir: Path | None = None, max_age_s: float = 6.0) -> dict[str, Any] | None:
    try:
        info = json.loads((base_dir(data_dir) / "host.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return info if time.time() - float(info.get("ts", 0)) <= max_age_s else None


@dataclass
class Core:
    base: Path
    registry: AgentRegistry
    telemetry: Telemetry
    recorder: RingRecorder
    prompts: PromptManager
    changes: ChangeManager
    prefs: Preferences
    control: ComputerControl
    jobs: JobRunner
    events: EventHub
    service: Service


def build_core(
    data_dir: Path,
    *,
    cfg: Callable[[], Any],
    settings: Callable[[], Any],
    config_path: Callable[[], Path] | None = None,
    ui: Any = None,
    control_hooks: ControlHooks | None = None,
    on_config: Callable[[Any], None] | None = None,
    camera_frame: Callable[[str | None], Any] = lambda _c: None,
    screen_grab: Callable[[], Any] | None = None,
    flash: Callable[[str], None] = lambda _t: None,
    watching: Callable[[str, bool], None] = lambda _n, _o: None,
    foreground: Callable[[], Any] = lambda: None,
    uia: Any = None,
    tts: Any = None,
    notify_fallback: Callable[[str, str], bool] | None = None,
    record_header: Callable[[], dict[str, Any]] | None = None,
    job_command: Callable[[list[str]], list[str]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Core:
    from away import config as cfgmod

    base = base_dir(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    s = settings()
    events = EventHub()
    telemetry = Telemetry(clock=clock)

    def emit(topic: str, data: dict[str, Any]) -> None:
        events.publish(topic, data)

    telemetry.on_change = emit
    recorder = RingRecorder(float(getattr(s, "record_buffer_s", 30.0)), clock)
    registry = AgentRegistry(base / "agents.json")
    prompts = PromptManager(
        ui if ui is not None else NullUI(),
        polite=telemetry.polite_moment,
        clock=clock,
        tts=tts,
        on_event=emit,
        fallback=notify_fallback,
    )
    changes = ChangeManager(base, config_path or cfgmod.config_path, on_applied=on_config)
    prefs = Preferences(base / "preferences.md")

    def always(agent_id: str, cat: str) -> bool:
        a = registry.get(agent_id)
        return bool(a and cat in a.always_allow)

    if control_hooks is None:
        from away.control.mouse import RecordingBackend

        control_hooks = ControlHooks(backend=RecordingBackend())
    control = ComputerControl(
        control_hooks,
        prompts.ask,
        ttl_s=float(getattr(s, "control_ttl_min", 30)) * 60.0,
        rate_per_s=float(getattr(s, "action_rate_per_s", 4.0)),
        log_folder=base / "actions",
        clock=clock,
        on_event=emit,
        always_allowed=always,
        allow_always=registry.allow_always,
    )
    jobs = shared_runner(Path(data_dir) / "diagnostics", command=job_command)
    service = Service(
        data_dir=Path(data_dir),
        registry=registry,
        telemetry=telemetry,
        prompts=prompts,
        changes=changes,
        prefs=prefs,
        control=control,
        jobs=jobs,
        recorder=recorder,
        cfg=cfg,
        settings=settings,
        camera_frame=camera_frame,
        screen_grab=screen_grab,
        flash=flash,
        foreground=foreground,
        uia=uia,
        record_header=record_header,
        events=events,
        watching=watching,
    )
    return Core(base, registry, telemetry, recorder, prompts, changes, prefs, control, jobs, events, service)


class AgentLinkHost:
    """One per process. ``app`` is an AwayApp (or None for the standalone CLI host)."""

    def __init__(self, app: Any = None, data_dir: Path | None = None, ui: Any = None, headless: bool = False) -> None:
        from away import config as cfgmod

        self.app = app
        self.data_dir = Path(data_dir or paths.data_dir())
        self.base = base_dir(self.data_dir)
        self.stop_event = threading.Event()
        self._cfg_fallback: Any = None
        self.config_path = (getattr(app, "config_path", None) if app is not None else None) or cfgmod.config_path()
        self.headless = headless
        self.frames: dict[str, Any] = {}
        self.server: Any = None
        self.gate: Any = None
        self.tunnel: Any = None
        self.pairing: Any = None
        self.errors: list[str] = []
        self._overlay_stop: Path | None = None
        self._watch_stop: Path | None = None
        self._listeners: list[Any] = []
        self._palm = handsigns.SignHold(hold_s=1.0)
        self._lock = threading.RLock()
        self._last_state = 0.0
        self._results: dict[str, Any] = {}
        self._health_until = 0.0
        self.local_key = local_key(self.base)
        if ui is None and not headless:
            try:
                from away.agentlink.prompts import SubprocessUI

                ui = SubprocessUI(self.base / "ui")
            except Exception:
                ui = None
        self.ui = ui
        hooks = self._control_hooks()
        self.core = build_core(
            self.data_dir,
            cfg=self.cfg,
            settings=lambda: self.cfg().agent_link,
            config_path=lambda: Path(self.config_path),
            ui=ui,
            control_hooks=hooks,
            on_config=self._apply_config,
            camera_frame=self._camera_frame,
            flash=self._flash,
            watching=self._watching,
            foreground=hooks.foreground,
            uia=hooks.uia,
            tts=_tts(lambda: getattr(self.cfg(), "speech", None)),
            notify_fallback=self._tray_notice,
            record_header=self._record_header,
        )
        s = self.cfg().agent_link
        self.detectors = Detectors(
            self._on_trip,
            fps_drop_ratio=float(s.fps_drop_ratio),
            misfire_threshold=int(s.misfire_threshold),
            cooldown_s=float(s.detector_cooldown_s),
        )
        self.core.events.listeners.append(self._on_hub_event)

    # --------------------------------------------------------------- config
    def cfg(self) -> Any:
        if self.app is not None:
            return self.app.cfg
        if self._cfg_fallback is None:
            from away import config as cfgmod

            self._cfg_fallback, _ = (
                cfgmod.load(Path(self.config_path)) if Path(self.config_path).exists() else (cfgmod.AppConfig(), [])
            )
        return self._cfg_fallback

    def _apply_config(self, cfg: Any) -> None:
        if self.app is None:
            self._cfg_fallback = cfg
            return
        with contextlib.suppress(Exception):
            self.app.pipeline.apply_config(cfg)
        with contextlib.suppress(Exception):
            # v0.1.20: + attention/comfort, Gesture Studio (zones, per-gesture settings) and shortcuts
            for f in ("hands", "face", "gaze", "bindings", "attention", "comfort", "zones", "studio", "macros"):
                setattr(self.app.cfg, f, getattr(cfg, f))
            self.app.cfg.voice.click_by_name = cfg.voice.click_by_name  # smart pointer (the rest of voice: Settings)
        macros = getattr(self.app, "macros", None)
        if macros is not None:
            with contextlib.suppress(Exception):
                macros.set_config(cfg.macros)

    def _record_header(self) -> dict[str, Any]:
        from away.agentlink.recording import make_header

        screen = (1920, 1080)
        paused = False
        if self.app is not None:
            with contextlib.suppress(Exception):
                lay = self.app.monitor_layout
                screen = (
                    (int(lay[0]), int(lay[1]))
                    if isinstance(lay, tuple)
                    else tuple(self.app.pipeline.mouse.mapper.screen)[:2]
                )  # type: ignore[assignment]
            with contextlib.suppress(Exception):
                paused = bool(self.app.pipeline.paused)
        return make_header(self.cfg(), screen, paused)

    # ------------------------------------------------------------ platform
    def _control_hooks(self) -> ControlHooks:
        from away.agentlink import desktop

        backend: Any
        if self.app is not None:
            backend = self.app.pipeline.mouse.backend
        else:
            from away.control.mouse import RecordingBackend

            backend = RecordingBackend()
        if self.app is not None and not getattr(self.app, "dry_run", True):
            from away.app.autopause import GuardedBackend, ManualInputGuard

            self.input_guard: Any = ManualInputGuard(500.0)
            backend = GuardedBackend(backend, self.input_guard)
        else:
            self.input_guard = None

        def pos() -> tuple[int, int] | None:
            try:
                from pynput.mouse import Controller

                x, y = Controller().position
                return int(x), int(y)
            except Exception:
                return None

        uia: Any = desktop.make_uia()
        smart = getattr(getattr(self.app, "pipeline", None), "smart", None) if self.app is not None else None
        if smart is not None:  # v0.1.20: find elements by name with the smart pointer's fast cache + fuzzy names
            uia = desktop.SmartPointerUIA(smart, uia)
        return ControlHooks(
            backend=backend,
            uia=uia,
            cursor_pos=pos,
            overlay_on=self._overlay_on,
            overlay_off=self._overlay_off,
        )

    def _overlay_on(self, name: str) -> None:
        self._overlay_off()
        self._start_input_watch()
        if self.ui is not None and hasattr(self.ui, "overlay"):
            self._overlay_stop = self.base / "ui" / f"overlay-{uuid.uuid4().hex[:6]}.stop"
            self.ui.overlay(name, self._overlay_stop)
        self._notice("Project Away", f"{name} is using your computer. Move your hand or mouse to take over.")

    def _overlay_off(self) -> None:
        if self._overlay_stop is not None:
            with contextlib.suppress(OSError):
                self._overlay_stop.write_text("stop", encoding="utf-8")
            self._overlay_stop = None
        self._stop_input_watch()

    def _watching(self, name: str, on: bool) -> None:
        if on and self._watch_stop is None and self.ui is not None and hasattr(self.ui, "overlay"):
            self._watch_stop = self.base / "ui" / f"watch-{uuid.uuid4().hex[:6]}.stop"
            self.ui.overlay(name, self._watch_stop, mode="watch")
        elif not on and self._watch_stop is not None:
            with contextlib.suppress(OSError):
                self._watch_stop.write_text("stop", encoding="utf-8")
            self._watch_stop = None

    def _flash(self, text: str) -> None:
        if self.ui is not None and hasattr(self.ui, "flash"):
            with contextlib.suppress(Exception):
                self.ui.flash(text)
        self._notice("Project Away", text)

    def _tray_notice(self, title: str, text: str) -> bool:
        """v0.1.21: when the message window cannot show, the tray's notification balloon (Windows) / desktop
        notification shows the agent's message instead. True when a tray app took it."""
        if self.app is None or self.headless:
            return False
        try:
            subs = getattr(self.app.bus, "has_subscribers", None)
            if callable(subs) and not subs("app.notice"):
                return False
            self.app.bus.publish("app.notice", {"title": title, "message": text, "key": None})
            return True
        except Exception:
            return False

    def _notice(self, title: str, text: str) -> None:
        if self.app is not None:
            with contextlib.suppress(Exception):
                self.app.bus.publish("app.notice", {"title": title, "message": text, "key": None})

    def _start_input_watch(self) -> None:  # pragma: no cover - needs a desktop session
        """While an agent is in control: physical mouse/keyboard -> takeover pause; Ctrl+Alt+Esc -> stop."""
        if self._listeners or self.app is None or getattr(self.app, "dry_run", True):
            return
        try:
            from pynput import keyboard, mouse

            from away.control.keys import to_pynput_hotkey

            g = self.input_guard
            ctl = self.core.control

            def moved(x: float, y: float) -> None:
                if g is not None and g.on_mouse_move(x, y):
                    ctl.user_input("mouse")

            def clicked(*_a: Any) -> None:
                if g is not None and g.on_click():
                    ctl.user_input("mouse")

            def pressed(_k: Any) -> None:
                self.core.telemetry.note_key()
                if g is not None and g.on_key():
                    ctl.user_input("keyboard")

            ml = mouse.Listener(on_move=moved, on_click=clicked, on_scroll=lambda *a: clicked())
            kl = keyboard.Listener(on_press=pressed)
            hk = keyboard.GlobalHotKeys(
                {to_pynput_hotkey(self.cfg().agent_link.stop_hotkey): lambda: ctl.stop("you pressed the stop keys")}
            )
            for lst in (ml, kl, hk):
                lst.start()
            self._listeners = [ml, kl, hk]
        except Exception as exc:
            self.errors.append(f"could not watch the mouse/keyboard for takeover: {exc}")

    def _stop_input_watch(self) -> None:
        for lst in self._listeners:
            with contextlib.suppress(Exception):
                lst.stop()
        self._listeners = []

    def _camera_frame(self, cam: str | None) -> Any:
        with self._lock:
            if cam is not None and str(cam) in self.frames:
                return self.frames[str(cam)]
            return next(iter(self.frames.values()), None)

    def _any_agent_has(self, scope: str) -> bool:
        return any(a.has(scope) for a in self.core.registry.list())

    # --------------------------------------------------------- app wiring
    def attach(self) -> AgentLinkHost:
        app = self.app
        if app is None:
            return self
        app.bus.subscribe("*", self._on_bus)
        pipe = app.pipeline
        orig_hand = pipe.on_hand
        orig_faces = pipe.on_faces

        def on_hand(t: float, landmarks: Any, handedness: str = "Right", **kw: Any) -> Any:
            with contextlib.suppress(Exception):
                self._on_hand(t, landmarks, handedness, kw)
            return orig_hand(t, landmarks, handedness, **kw)

        def on_faces(t: float, boxes: Any, blendshapes: Any, frontal: Any = None, **kw: Any) -> Any:
            res = orig_faces(t, boxes, blendshapes, frontal, **kw)
            try:
                user_frontal = (
                    bool(pipe.gate.last_frontal)
                    if hasattr(pipe.gate, "last_frontal")
                    else any(bool(f) for f in (frontal or []))
                )
                self.core.telemetry.on_faces(len(boxes or []), user_frontal)
            except Exception:
                pass
            return res

        pipe.on_hand = on_hand
        pipe.on_faces = on_faces
        pipe.executor.hooks["agent"] = self._agent_action
        tel = self.core.telemetry
        tel.voice_state = lambda: app.voice_state()
        tel.active_app = lambda: pipe.dispatcher.last_app
        tel.paused = lambda: bool(pipe.paused)
        tel.models = self._models()
        with contextlib.suppress(Exception):  # v0.1.20: fatigue mode, attention gate, smart pointer, shortcuts...
            from away.agentlink.integrations import context_extensions

            tel.extensions.update(context_extensions(app))
        guard = getattr(app, "guard", None)
        if guard is not None:  # typing context from the auto-pause listener (timestamps only, never keys)
            orig_key = guard.on_key

            def on_key() -> bool:
                tel.note_key()
                return bool(orig_key())

            guard.on_key = on_key
        return self

    def _models(self) -> dict[str, Any]:
        app = self.app
        out: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            md = Path(app.models_dir)
            out["hand_landmarker"] = (md / "hand_landmarker.task").exists()
            out["face_landmarker"] = (md / "face_landmarker.task").exists()
            out["gesture_model"] = bool(app.pipeline.temporal)
            out["speech_model"] = app.asr_model or app.cfg.voice.asr_model
            out["speech_device"] = app.asr_device
            out["gaze_model"] = app.cfg.gaze.model
        return out

    def on_frame(self, cam: Any, t: float, rgb: Any, fr: Any) -> None:
        """Called by the camera loop for every processed frame (per camera)."""
        lms = getattr(fr, "landmarks", None) if fr is not None else None
        self.core.telemetry.on_frame(cam, t, lms, int(getattr(fr, "n_hands", 0) or 0))
        if rgb is not None and self._snapshots_wanted():
            with self._lock:
                self.frames[str(cam)] = rgb

    def _snapshots_wanted(self) -> bool:
        now = time.monotonic()
        if now - getattr(self, "_snap_checked", 0.0) > 2.0:
            self._snap_checked = now
            self._snap_want = self._any_agent_has(perms.SNAPSHOTS)
            if not self._snap_want:
                with self._lock:
                    self.frames.clear()
        return bool(getattr(self, "_snap_want", False))

    def _on_hand(self, t: float, landmarks: Any, handedness: str, kw: dict[str, Any]) -> None:
        self.core.recorder.on_hand(t, landmarks, handedness, **kw)
        sign = handsigns.classify(landmarks) if landmarks is not None else None
        if self.core.prompts.current is not None:
            self.core.prompts.feed_sign(t, sign)
        held = self._palm.update(t, sign if sign == handsigns.OPEN_PALM else None)
        if self.core.control.grant is not None and held:
            self.core.control.stop("you held up an open palm")

    def _on_bus(self, ev: Any) -> None:
        topic, data = ev.topic, ev.data
        tel = self.core.telemetry
        tel.on_event(topic, data)
        self.core.recorder.on_event(float(ev.ts), topic, data)
        self.detectors.on_event(topic, data)
        active = self.app is not None and not self.app.pipeline.paused
        if topic.startswith("gesture.") and topic not in ("gesture.rejected", "gesture.hand_lost") and active:
            self.core.control.user_input("hand")
        if topic == "gesture.custom":
            self.core.prompts.feed_gesture(str(data.get("name", "")))
        if topic == "voice.utterance":
            text = str(data.get("text") or "")
            if self.core.prompts.current is not None and self.core.prompts.feed_voice(text):
                return
            if normalize(text) in STOP_WORDS and self.core.control.grant is not None:
                self.core.control.stop("you said stop")
        if topic in ("action.cancelled", "action.undone", "feedback.wrong"):
            self.core.events.publish("misfire", {"topic": topic, **{k: str(v)[:100] for k, v in data.items()}})

    def _agent_action(self, arg: str, data: dict[str, Any]) -> None:
        """``agent:summon|look|screen|wrong|stop`` bindings (gestures or voice phrases the user binds)."""
        ev = self.core.events
        if arg == "summon":
            ev.publish("summon", {"source": "gesture or voice"})
            threading.Thread(
                target=self.core.prompts.notify, args=("Your agents were called", "info"), daemon=True
            ).start()
        elif arg in ("look", "screen"):
            self._user_capture(arg)
        elif arg == "wrong":
            ev.publish(
                "wrong", {"note": "the person said that was wrong", "recent": self.core.telemetry.poll(0, 10)["events"]}
            )
            if self.app is not None:
                with contextlib.suppress(Exception):
                    self.app.bus.publish("feedback.wrong", {})
        elif arg == "stop":
            self.core.control.stop("you asked to stop")

    def _user_capture(self, kind: str) -> None:
        from away.agentlink.desktop import grab_screen, to_jpeg

        svc = self.core.service
        if kind == "look" and self._any_agent_has(perms.SNAPSHOTS):
            frame = self._camera_frame(None)
            if frame is not None:
                jpeg, scale = to_jpeg(frame, int(self.cfg().agent_link.snapshot_max_width), 60)
                c = svc.user_capture("camera", jpeg, {"scale": scale})
                self.core.events.publish("user.look", {"capture": c.id, "kind": "camera"})
                self._flash("Sent a camera picture to your agent")
                return
        if self._any_agent_has(perms.SCREEN):
            img = grab_screen()
            if img is not None:
                jpeg, scale = to_jpeg(img, int(self.cfg().agent_link.screen_max_width), 55)
                c = svc.user_capture("screen", jpeg, {"scale": scale})
                self.core.events.publish("user.screen", {"capture": c.id, "kind": "screen"})
                self._flash("Sent your screen to your agent")
                return
        self.core.events.publish("summon", {"source": kind, "note": "no agent may see the camera/screen"})

    def _on_hub_event(self, ev: dict[str, Any]) -> None:
        if ev["topic"].startswith(("agent.", "control.")) and self.app is not None:
            with contextlib.suppress(Exception):
                self.app.bus.publish("agentlink.changed", {"topic": ev["topic"]})  # the tray refreshes its badge
        if ev["topic"] == "control.paused":
            self._notice("Project Away", f"{ev['data'].get('agent')} paused: you took over.")

    def _on_trip(self, name: str, detail: dict[str, Any]) -> None:
        from away.agentlink import bundle

        path = None
        try:
            snap = self.core.recorder.snapshot()
            path = bundle.build(
                self.data_dir,
                self.cfg(),
                {"detector": name, "detail": detail, "health": self.core.telemetry.health()},
                (self._record_header(), snap),
                note=f"auto-detector: {name}",
            )
        except Exception as exc:
            self.errors.append(f"could not save a diagnostics bundle: {exc}")
        info = {"detector": name, "detail": detail, "bundle": f"diagnostics/{path.name}" if path else None}
        self.core.events.publish("detector", info)
        s = self.cfg().agent_link
        if s.github_issues:
            threading.Thread(target=self._file_issue, args=(name, detail), daemon=True).start()

    def _file_issue(self, name: str, detail: dict[str, Any]) -> None:  # pragma: no cover - network
        from away import __version__
        from away.agentlink.detectors import file_github_issue, github_token

        token = github_token()
        if not token:
            return
        import platform as pf

        body = (
            f"Auto-detector **{name}** tripped.\n\n- version: {__version__}\n- os: {pf.platform()}\n"
            f"- detail: `{json.dumps(detail, default=str)[:1500]}`\n\nA diagnostics bundle was saved locally "
            "(not uploaded)."
        )
        with contextlib.suppress(Exception):
            file_github_issue(self.cfg().agent_link.github_repo, f"[auto] {name} on v{__version__}", body, token)

    # -------------------------------------------------------------- server
    def ensure_server(self) -> int:
        with self._lock:
            if self.server is not None and self.server.running:
                return int(self.server.port)
            from away.agentlink import api, mcp_tools
            from away.agentlink.server import Server

            s = self.cfg().agent_link
            self.gate = api.Gate(
                self.core.service,
                self.core.registry,
                local_key=self.local_key,
                remote_ttl_s=lambda: float(self.cfg().agent_link.remote_ttl_min) * 60.0,
                local_ttl_s=lambda: float(self.cfg().agent_link.local_ttl_days) * 86400.0,
                rate=(float(s.rate_per_s), int(s.rate_burst)),
                access_log=self.data_dir / "logs" / "agent-access.log",
            )
            mcp = mcp_tools.build(
                api.LocalBackend(self.core.service), screen_fps=lambda: float(self.cfg().agent_link.screen_fps)
            )
            app = api.create_app(self.core.service, self.gate, mcp)
            self.server = Server(app, int(s.port))
            port = self.server.start()
            self.core.prompts.start()
            self._write_host()
            return port

    def _write_host(self) -> None:
        port = int(self.server.port) if self.server is not None and self.server.running else 0
        write_private(self.base / "host.json", json.dumps({"pid": os.getpid(), "port": port, "ts": time.time()}))

    def start_pairing(self, scopes: Any, remote: bool = True, download: bool = True) -> dict[str, Any]:
        """A new single-use pairing link (+ QR + 6-digit code); starts the tunnel for remote agents."""
        port = self.ensure_server()
        s = self.cfg().agent_link
        url = f"http://127.0.0.1:{port}"
        if remote:
            url = self._ensure_tunnel(port, download)
        p = self.core.registry.new_pairing(perms.normalize(scopes), float(s.pairing_ttl_s))
        self.pairing = {
            "url": f"{url}/c/{p.code}",
            "pin": p.pin,
            "expires": time.time() + float(s.pairing_ttl_s),
            "scopes": perms.to_list(p.scopes),
            "remote": remote,
            "code": p.code,
        }
        return dict(self.pairing)

    def _ensure_tunnel(self, port: int, download: bool) -> str:
        from away.agentlink import tunnel

        if self.tunnel is not None and self.tunnel.alive() and self.tunnel.url:
            return str(self.tunnel.url)
        s = self.cfg().agent_link
        exe = tunnel.find(self.data_dir, s.cloudflared_path)
        if exe is None:
            if not (download and s.cloudflared_download):
                raise tunnel.TunnelError("the secure link helper (cloudflared) is not installed")
            exe = tunnel.download(self.data_dir)
        self.tunnel = tunnel.Tunnel(exe)
        return self.tunnel.start(port)

    def stop_tunnel(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()
            self.tunnel = None
        self.pairing = None

    def disconnect_all(self) -> int:
        n = self.core.registry.revoke_all()
        self.core.control.stop("Disconnect all")
        self.core.service._watchers.clear()
        self._watching("", False)
        self.stop_tunnel()
        self.core.events.publish("agent.disconnected", {"all": True})
        return n

    # --------------------------------------------------------- control loop
    def start(self) -> AgentLinkHost:
        self.core.prompts.start()
        if any(a.kind == "local" for a in self.core.registry.list()):
            with contextlib.suppress(Exception):
                self.ensure_server()
        self._write_host()
        threading.Thread(target=self._loop, daemon=True, name="agentlink-host").start()
        return self

    def _loop(self) -> None:
        while not self.stop_event.wait(0.3):
            try:
                self._take_requests()
                self.core.control.status()  # expiry
                now = time.monotonic()
                if now - self._last_state >= STATE_EVERY_S:
                    self._last_state = now
                    self._tick()
            except Exception as exc:
                self.errors.append(str(exc)[:200])
                self.errors = self.errors[-20:]

    def _tick(self) -> None:
        crash = self.data_dir / "logs" / "crash.log"
        size = crash.stat().st_size if crash.exists() else None
        bus_errors = len(getattr(getattr(self.app, "bus", None), "errors", []) or [])
        self.detectors.tick(self.core.telemetry.fps(), size, bus_errors)
        self._write_host()  # heartbeat for `away mcp` (port 0 until an agent needs the server)
        if self.tunnel is not None and not self.tunnel.alive():
            self.errors.append("the secure link stopped; press Connect again")
            self.tunnel = None
        self.write_state()

    def _take_requests(self) -> None:
        folder = self.base / "requests"
        if not folder.is_dir():
            return
        for f in sorted(folder.glob("*.json")):
            try:
                req = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                req = None
            f.unlink(missing_ok=True)
            if not isinstance(req, dict):
                continue
            try:
                result = self.handle(req)
                self._results[str(req.get("id"))] = {"ok": True, "result": result}
            except Exception as exc:
                self._results[str(req.get("id"))] = {"ok": False, "error": friendly_error(exc)}
            self._results = dict(list(self._results.items())[-20:])
            self.write_state()

    def handle(self, req: dict[str, Any]) -> Any:
        a = req.get("action")
        core = self.core
        if a == "serve":
            return {"port": self.ensure_server()}
        if a == "pair":
            return self.start_pairing(req.get("scopes"), bool(req.get("remote", True)), bool(req.get("download", True)))
        if a == "stop_link":
            self.stop_tunnel()
            return {}
        if a == "revoke":
            core.control.revoke_agent(str(req.get("agent_id")))
            return {"revoked": core.registry.revoke(str(req.get("agent_id")))}
        if a == "set_scopes":
            return core.registry.set_scopes(str(req.get("agent_id")), req.get("scopes") or []).public()
        if a == "disconnect_all":
            return {"disconnected": self.disconnect_all()}
        if a == "answer":
            return {"answered": core.prompts.click(str(req.get("prompt_id")), str(req.get("answer")))}
        if a == "undo_change":
            return (
                core.changes.undo_batch(str(req["batch"]), who="you")
                if req.get("batch")
                else core.changes.undo(str(req.get("change_id")), who="you")
            )
        if a == "undo_agent_today":
            return core.changes.undo_agent_today(str(req.get("agent_id")), who="you")
        if a == "control_stop":
            return {"stopped": core.control.stop("you pressed Stop")}
        if a == "control_undo":
            return core.control.undo_last()
        if a == "add_preference":
            return {"items": core.prefs.add(str(req.get("text")), who="you")}
        if a == "remove_preference":
            return {"items": core.prefs.remove(int(req.get("number", 0)))}
        if a == "bundle":
            from away.agentlink import bundle

            path = bundle.build(
                self.data_dir,
                self.cfg(),
                {"health": core.telemetry.health(), "context": core.telemetry.context()},
                (self._record_header(), core.recorder.snapshot()),
                note="requested by you",
            )
            return {"path": f"diagnostics/{path.name}"}
        if a == "want_health":
            self._health_until = time.monotonic() + HEALTH_WANT_S
            return {}
        raise ValueError(f"unknown request {a!r}")

    def write_state(self) -> None:
        core = self.core
        p = self.pairing
        if p is not None and time.time() >= float(p["expires"]):
            self.pairing = p = None
        pending = core.registry.open_pairings()
        if p is not None and not any(q.code == p["code"] for q in pending):
            self.pairing = p = None  # used (approved or denied)
        state = {
            "ts": time.time(),
            "server": {
                "running": bool(self.server and self.server.running),
                "port": int(self.server.port) if self.server else 0,
            },
            "link": {"url": self.tunnel.url if self.tunnel else None},
            "pairing": {k: v for k, v in (p or {}).items() if k != "code"} or None,
            "agents": [a.public() for a in core.registry.list()],
            "prompts": core.prompts.pending(),
            "control": core.control.status(),
            "actions": core.control.actions(20),
            "changes": core.changes.history(20),
            "access": list(self.gate.recent)[-20:] if self.gate else [],
            "results": self._results,
            "errors": self.errors[-5:],
            "autofix": self.cfg().agent_link.autofix,
            "health": core.telemetry.health() if time.monotonic() < self._health_until else None,
            "context": core.telemetry.context(),
        }
        write_private(self.base / "state.json", json.dumps(state, default=str))

    def stop(self) -> None:
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.core.control.stop("Project Away is closing")
        with contextlib.suppress(Exception):
            self._watching("", False)
        self.core.prompts.stop()
        self.stop_tunnel()
        if self.server is not None:
            if self.gate is not None:
                self.gate.stopped.set()
            self.server.stop()
            self.server = None
        with contextlib.suppress(OSError):
            (self.base / "host.json").unlink()


def friendly_error(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    low = text.lower()
    if isinstance(exc, ModuleNotFoundError):
        return 'Agent Link needs its extra parts: pip install "project-away[agent]" (the installer includes them).'
    if "cloudflared" in low or "secure link" in low:
        return "Couldn't start the secure link. Check your internet connection and try again."
    if "urlopen" in low or "getaddrinfo" in low or "timed out" in low:
        return "No internet connection right now. Connect and try again."
    return text[:200]


def _tts(settings: Callable[[], Any] | None = None) -> Any:
    """v0.1.21: the built-in voice (Windows SAPI, pyttsx3, macOS say, Linux spd-say/espeak); see
    :mod:`away.voice.speech`. ``settings`` returns the config's ``speech`` section."""
    try:
        from away.voice.speech import Speaker

        return Speaker(settings)
    except Exception:
        return None


def attach(app: Any) -> AgentLinkHost | None:
    """Called by the app at start. Never raises: Agent Link must not stop the app from running."""
    try:
        if not bool(getattr(app.cfg.agent_link, "enabled", True)):
            return None
        return (
            AgentLinkHost(app, headless=bool(getattr(app, "dry_run", False)) and "pytest" in sys.modules)
            .attach()
            .start()
        )
    except Exception as exc:
        with contextlib.suppress(Exception):
            app.bus.publish("agentlink.error", {"error": str(exc)})
        return None


def badge(h: AgentLinkHost | None, now: float | None = None) -> str:
    """Tray text: "<agent> is using the computer", "<agents> connected", or ""."""
    if h is None:
        return ""
    try:
        ctl = h.core.control.status()
        if ctl.get("active"):
            return f"{ctl['agent']} is using the computer" + (" (paused)" if ctl.get("paused") else "")
        t = time.time() if now is None else now
        names = sorted({a.name for a in h.core.registry.list() if t - a.last_seen < 90})
        return f"{', '.join(names)} connected" if names else ""
    except Exception:
        return ""


def mask(text: str) -> str:
    return re.sub(r"pa_[A-Za-z0-9_-]{8,}", "pa_***", text)
