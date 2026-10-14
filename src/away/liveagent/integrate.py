"""Wiring the live agent into the running app. The app calls :func:`attach` once at start (after Agent Link);
the tray adds :func:`menu_items` and :func:`hook_updates`; nothing else in the app knows about the live agent.

* **Voice** - "Hey <name>, ..." uses the app's existing wake-word routing (``away.voice.talk``): the live agent is
  listed as one more agent, and its messages are answered here instead of going to an inbox. "Hey agent" also
  reaches it when no other agent is connected and no default agent is set.
* **Barge-in** - the microphone feed gets a level check in front (:class:`away.liveagent.speech.BargeIn`).
* **Agent Link** - external agents see ``live_agent`` in ``get_context`` and can switch its model or turn it off
  through ``change_settings`` (domain ``live_agent``; you confirm on the PC, it can be undone). They never see keys.
* **Settings changes** - Settings is another process; the app re-reads ``liveagent/settings.json`` every 2 s.
* **Load** - nothing runs while it is off; when on, the opencode engine starts on the first request and stops
  after 30 idle minutes.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Any

from away.liveagent import settings as st
from away.liveagent.tools import AGENT_ID

POLL_S = 2.0


class LiveAgentHost:
    def __init__(self, app: Any, data_dir: Path | None = None) -> None:
        from away import paths
        from away.liveagent.agent import LiveAgent
        from away.liveagent.runtime import Runtime
        from away.liveagent.speech import LiveSpeech
        from away.liveagent.tools import ToolBox

        self.app = app
        self.data_dir = Path(data_dir or paths.data_dir())
        self.watcher = st.Watcher(st.path(self.data_dir))
        self.runtime = Runtime(self.data_dir, settings=lambda: self.watcher.current)
        al = getattr(app, "agent_link", None)
        self.service = getattr(getattr(al, "core", None), "service", None)
        speaker = getattr(getattr(self.service, "prompts", None), "tts", None) if self.service is not None else None
        if speaker is None:
            with contextlib.suppress(Exception):
                from away.voice.speech import Speaker

                speaker = Speaker(lambda: getattr(app.cfg, "speech", None))
        self.speech = LiveSpeech(speaker, enabled=lambda: self.watcher.current.speak)
        self._decider: Any = None
        self._decider_key: Any = None
        self.tools = ToolBox(self.service, lambda: self.watcher.current, None)
        self.agent = LiveAgent(
            lambda: self.watcher.current,
            lambda: self.runtime.funnel(self.watcher.current),
            self.decider,
            self.tools,
            self.speech,
            self._show,
            self.data_dir / "liveagent" / "turns.jsonl",
        )
        self.stop_event = threading.Event()
        self.barge: Any = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ pieces
    def settings(self) -> st.LiveSettings:
        return self.watcher.current

    def decider(self) -> Any:
        s = self.watcher.current
        key = (s.jev, s.preferred, tuple((p.id, p.model, p.enabled) for p in s.providers))
        if self._decider is None or key != self._decider_key:
            self._decider = self.runtime.decider(s)
            self._decider_key = key
            self.tools.decider = self._decider
        return self._decider

    def _show(self, text: str) -> None:
        if not text:
            return
        name = self.watcher.current.name
        al = getattr(self.app, "agent_link", None)
        if al is not None and hasattr(al, "_flash"):
            with contextlib.suppress(Exception):
                al._flash(f"{name}: {text[:200]}")
                return
        with contextlib.suppress(Exception):
            self.app.bus.publish("app.notice", {"title": name, "message": text[:200], "key": None})

    def enabled(self) -> bool:
        return bool(self.watcher.current.enabled)

    # ------------------------------------------------------------ voice
    def hook_talk(self) -> None:
        talk = getattr(self.app, "talk", None)
        if talk is None:
            return
        orig_agents, orig_deliver, orig_enabled = talk.agents, talk.deliver, talk.enabled

        def agents() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            with contextlib.suppress(Exception):
                out = list(orig_agents())
            if self.enabled():
                out.append({"id": AGENT_ID, "name": self.watcher.current.name, "wake_word": ""})
            return out

        def deliver(target: str, text: str, source: str) -> str:
            s = self.watcher.current
            if s.enabled and (target == AGENT_ID or (target == "" and self._generic_is_mine())):
                self.agent.submit(text, source)
                return f"{s.name} is answering"
            if orig_deliver is None:
                return ""
            return str(orig_deliver(target, text, source) or "")

        def enabled() -> bool:
            try:
                if orig_enabled():
                    return True
            except Exception:
                pass
            return self.enabled()

        talk.agents = agents
        talk.deliver = deliver
        talk.enabled = enabled

    def _generic_is_mine(self) -> bool:
        """ "Hey agent" goes to the live agent when nobody else would get it."""
        svc = self.service
        if svc is None:
            return True
        try:
            if str(getattr(svc.settings(), "default_agent", "") or ""):
                return False
            now = time.time()
            return not any(svc.reachable(a, now) for a in svc.registry.list())
        except Exception:
            return False

    def hook_barge_in(self) -> bool:
        from away.liveagent.speech import BargeIn, wrap_feed

        voice = getattr(self.app, "voice", None)
        if voice is None:
            return False
        if self.barge is None:
            sr = 16000
            with contextlib.suppress(Exception):
                sr = int(self.app.cfg.voice.sample_rate)
            self.barge = BargeIn(
                self.speech.speaking,
                self.agent.cancel,
                sample_rate=sr,
                enabled=lambda: self.enabled() and self.watcher.current.barge_in and self.agent.busy,
            )
        return wrap_feed(voice, self.barge)

    # ------------------------------------------------------------ Agent Link
    def hook_agent_link(self) -> None:
        al = getattr(self.app, "agent_link", None)
        tel = getattr(getattr(al, "core", None), "telemetry", None)
        if tel is not None:
            tel.extensions["live_agent"] = self.public_status
        register_domain(self)

    def public_status(self) -> dict[str, Any]:
        s = self.watcher.current
        out: dict[str, Any] = {"enabled": s.enabled, "wake_phrase": f"Hey {s.name}", "busy": self.agent.busy}
        if s.enabled:
            with contextlib.suppress(Exception):
                out["model"] = self.runtime.current_model(s)
        return out

    def tray_text(self) -> str:
        s = self.watcher.current
        if not s.enabled:
            return "Live agent: off"
        model = ""
        with contextlib.suppress(Exception):
            model = self.runtime.current_model(s)
        return f"Live agent: on ({model})" if model else "Live agent: on"

    def toggle(self) -> bool:
        """Tray: one click on/off. Returns False when it still needs setting up (Settings opens instead)."""
        s = st.load(self.watcher.path)
        if not s.enabled and not s.providers:
            return False
        s.enabled = not s.enabled
        st.save(s, self.watcher.path)
        self.watcher.poll()
        if not s.enabled:
            self.agent.cancel()
            self.runtime.stop()
        return True

    # ------------------------------------------------------------ lifecycle
    def start(self) -> LiveAgentHost:
        self.hook_talk()
        self.hook_agent_link()
        self.hook_barge_in()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="live-agent-watch")
        self._thread.start()
        return self

    def _loop(self) -> None:
        was = self.enabled()
        while not self.stop_event.wait(POLL_S):
            with contextlib.suppress(Exception):
                if self.watcher.poll():
                    now = self.enabled()
                    if was and not now:
                        self.agent.cancel()
                        self.runtime.stop()
                    was = now
                if self.barge is None:
                    self.hook_barge_in()
                srv = self.runtime.server
                if srv is not None:
                    srv.idle_check()

    def stop(self) -> None:
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.agent.cancel()
        self.runtime.stop()


def attach(app: Any, data_dir: Path | None = None) -> LiveAgentHost | None:
    """Called by the app at start. Never raises: the live agent must not stop the app from running."""
    try:
        host = LiveAgentHost(app, data_dir).start()
        import atexit

        atexit.register(host.stop)
        return host
    except Exception as exc:
        with contextlib.suppress(Exception):
            app.bus.publish("liveagent.error", {"error": str(exc)})
        return None


# ------------------------------------------------------------------------------------------------ Agent Link domain
def register_domain(host: Any) -> None:
    """``live_agent`` in Agent Link's change_settings / list_settings: status rows; ``switch`` the model
    (key = ``preferred`` or a job: chat / plan / vision / decide; value = "provider/model", "" = automatic) and
    ``off``. Turning it ON is the person's own switch (Settings or tray). Keys are never listed."""
    from away.agentlink import registry as reg

    def path() -> Path:
        return Path(host.watcher.path) if host is not None else st.path()

    def lst(_cfg: Any) -> list[dict[str, Any]]:
        s = st.load(path())
        rows: list[dict[str, Any]] = [
            {"key": "enabled", "value": s.enabled, "note": "only the person can turn it on (Settings > Live agent)"},
            {"key": "preferred", "value": s.preferred or "automatic"},
        ]
        rows += [{"key": t, "value": s.routing.get(t, "") or "automatic", "title": st.TASK_TITLES[t]} for t in st.TASKS]
        if host is not None:
            with contextlib.suppress(Exception):
                f = host.runtime.funnel(s)
                rows.append({"key": "models", "value": [m.public() for m in f.router.models]})
        return rows

    def apply(_cfg: Any, op: Any) -> Any:
        p = path()
        s = st.load(p)
        key = str(op.key or "preferred")
        before: Any
        inverse: list[Any]
        if op.op == "off":
            before = s.enabled
            s.enabled = False
            text = "Live agent: off"
            inverse = [reg.Op("live_agent", "restore", "enabled", before)]
        elif op.op == "switch":
            if key != "preferred" and key not in st.TASKS:
                raise reg.ChangeError(f"switch key must be preferred or one of {', '.join(st.TASKS)}")
            value = str(op.value or "")
            if value and "/" not in value:
                raise reg.ChangeError('give the model as "provider/model" (list_settings live_agent shows them)')
            if value and s.provider(value.split("/", 1)[0]) is None:
                raise reg.ChangeError(f"{value.split('/', 1)[0]} is not set up in Settings > Live agent")
            before = s.preferred if key == "preferred" else s.routing.get(key, "")
            if key == "preferred":
                s.preferred = value
            elif value:
                s.routing[key] = value
            else:
                s.routing.pop(key, None)
            text = f"Live agent {key}: {value or 'automatic'}"
            inverse = [reg.Op("live_agent", "switch", key, before)]
        elif op.op == "restore" and key == "enabled":
            before = s.enabled
            s.enabled = bool(op.value)  # "undo" of an off: back to how it was before the agent turned it off
            text = "Live agent: put back"
            inverse = [reg.Op("live_agent", "off", "enabled", None)]
        else:
            raise reg.ChangeError("live_agent supports switch and off")
        st.save(s, p)
        if host is not None:
            with contextlib.suppress(Exception):
                host.watcher.poll()
        return reg.Applied(op, before, op.value, inverse, text, True)

    reg.register(reg.Domain("live_agent", "Live agent (model, on/off)", lst, apply, ops=("switch", "off")))


# ------------------------------------------------------------------------------------------------ tray
def menu_items(pystray: Any, app: Any, open_settings: Any = None) -> list[Any]:
    """ "Live agent: on (Big Pickle)" - click to turn it on or off (opens Settings > Live agent when it has never
    been set up)."""

    def text(_i: Any) -> str:
        h = getattr(app, "live_agent", None)
        return h.tray_text() if h is not None else "Live agent: off"

    def click(_icon: Any, _item: Any) -> None:
        h = getattr(app, "live_agent", None)
        if h is not None and h.toggle():
            return
        if open_settings is not None:
            with contextlib.suppress(Exception):
                open_settings("Live agent")

    return [pystray.MenuItem(text, click)]


def hook_updates(updates: Any, data_dir: Path | None = None) -> None:
    """When Away checks for updates, also update Away's own copy of opencode (in the background; never the
    person's own install). Never breaks the update check."""
    run = getattr(updates, "_run_check", None)
    if run is None or getattr(run, "_liveagent", False):
        return

    def wrapped(*a: Any, **kw: Any) -> Any:
        res = run(*a, **kw)

        def bg() -> None:
            with contextlib.suppress(Exception):
                from away.liveagent import opencode

                opencode.check_update(data_dir)

        threading.Thread(target=bg, daemon=True, name="opencode-update").start()
        return res

    wrapped._liveagent = True  # type: ignore[attr-defined]
    updates._run_check = wrapped
