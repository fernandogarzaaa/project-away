""" "Let it use my computer": an agent moves the mouse, clicks, types and opens apps, while the user watches and
can take over at any moment.

Safety rules, all enforced here (the service only adds the per-agent permission check):

1. **Approval every session.** :meth:`ComputerControl.request` shows "<agent> wants to use your computer" on the
   PC; nothing happens until the user allows it (click, thumbs up or voice). The grant expires after
   ``agent_link.control_ttl_min`` (default 30 min), on Disconnect, or on a global stop.
2. **Visible.** While a grant is active a bright border and the banner "<agent> is using your computer - move
   your hand or mouse to take over" are on screen (``away agent-ui control``).
3. **Takeover.** Any real input (physical mouse/keyboard - told apart from the agent's own input by
   :class:`~away.app.autopause.ManualInputGuard` and short echo windows: typing never hides a real mouse movement,
   and each typed chunk hides key presses for a moment only - or a hand gesture) pauses the agent instantly and
   interrupts the action in progress. While paused a slim banner stays on screen ("Paused - you took over from
   <agent>") with "Let <agent> continue" and "Stop" buttons whose click IS the answer: no second approval
   round-trip through the agent. Resuming through the agent still needs a new approval. Global stop:
   Ctrl+Alt+Esc, saying "stop", or holding an open palm ~1 s (never delayed by the short grace right after
   approval).
4. **No secrets, no shell.** Typing is refused into password fields (UI Automation ``IsPassword``; on Windows
   typing is refused altogether when UI Automation is unavailable) and into terminals; nothing works while the
   secure desktop (UAC / Ctrl+Alt+Del) is up; open_app refuses terminals and system tools, except the
   hardcoded ``SAFE_SYSTEM_TOOLS`` allowlist launched straight through the shell (no keystrokes, no command
   line) - opening one is navigation only, anything consequential inside it still asks first; Win+R / Win+X are
   blocked.
5. **Risky clicks ask first.** Clicking an element named like Send / Submit / Pay / Buy / Purchase / Delete /
   Confirm / Install / Uninstall..., dragging onto or from one (the Recycle Bin, the Trash), or pressing Enter while
   one is focused, asks the user first (with "Always allow for this agent"). A click, drag, Enter, Delete or
   shortcut whose target it cannot identify (no UI Automation name, coordinates only) asks too ("Allow for this
   session" stops asking until the session ends). Moving, scrolling and typing into a named text box never ask.
6. **Rate limited and logged.** At most ``action_rate_per_s`` actions per second; every action is logged with a
   thumbnail of the screen; "Undo last" reverses what can be reversed (cursor moves, scrolls, typed text).
7. **Step budget.** Each approved session gets ``agent_link.control_step_budget`` actions (default 40, fresh
   every grant): every action result tells the agent how many steps it has left, and a session that used them
   all refuses new actions in plain words until the person approves again.
8. **Visual audit trail.** Every control step is kept per session (``agent_link.control_audit_trail``, on by
   default): timestamp, action, screenshot thumbnail and result, persisted to ``actions/session-<id>.jsonl`` and
   replayable step by step in the app (Settings > Agent Link).

Since audit Phase 1b every input goes through the one action engine (:mod:`away.actions_engine`): it owns the
cancel tokens, the secure desktop check, the rate limit, the risk classification, held-input cleanup, "Stop
everything" and the audit log (source ``agent``). This module keeps what is specific to agents: grants, takeover,
the "Are you sure?" prompts, screenshots and undo.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from away.actions_engine import (
    AGENT,
    CONSEQUENTIAL_KEYS,
    RISKY,
    SHORTCUT_MODIFIERS,
    USER,
    Action,
    ActionCancelled,
    ActionEngine,
    ActionRefused,
    classify,
)
from away.agentlink import desktop

__all__ = [
    "CONSEQUENTIAL_KEYS",
    "RISKY",
    "SAFE_SYSTEM_TOOLS",
    "SHORTCUT_MODIFIERS",
    "ComputerControl",
    "ControlHooks",
    "ControlRefused",
    "Grant",
    "LogEntry",
    "classify",
    "describe_action",
    "plain_end_reason",
]

UNKNOWN_NOTE = "Project Away can't tell what is there, so it is asking first."
ALLOW_SESSION = "Allow for this session"


BLOCKED_COMBOS = ({"cmd", "r"}, {"cmd", "x"}, {"ctrl", "alt", "delete"}, {"ctrl", "shift", "esc"})
BLOCKED_APPS = desktop.TERMINAL_PROCESSES
# Allowlisted Windows system tools (proposal #2): the only system tools an agent may open, selected by a
# hardcoded key - the launch target is hardcoded too, so no command line is ever built and the name cannot
# smuggle one in. Opening one is navigation only: anything consequential done inside it still goes through
# the normal ask-first rules. Terminals, PowerShell, regedit and mmc-for-other-snapins stay BLOCKED:
# devmgmt.msc runs under mmc.exe, so it is launched by its .msc file path explicitly, not by unblocking mmc.
# Task Manager is deliberately NOT allowlisted: its File > Run new task dialog executes arbitrary commands,
# which would bypass the terminal block through an allowlisted tool.
SAFE_SYSTEM_TOOLS: dict[str, str] = {
    "device-manager": r"C:\Windows\System32\devmgmt.msc",
    "windows-update": "ms-settings:windowsupdate",
    "network": "ms-settings:network",
    "bluetooth": "ms-settings:bluetooth",
    "display": "ms-settings:display",
    "sound": "ms-settings:sound",
    "apps": "ms-settings:appsfeatures",
    "storage": "ms-settings:storagesense",
    "about": "ms-settings:about",
}
APP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+&'-]{0,39}$")
ENTER_KEYS = ("enter", "return")
# v0.1.24 hardening: the moment after approval (the click or thumbs up that allowed it) is not a takeover, but only
# briefly; a deliberate stop (hotkey, "stop", open palm) never waits for it (:meth:`ComputerControl.stop`).
GRACE_S = 0.75
# The agent's own input echoes back through the OS listeners. Mouse and keyboard echoes are told apart: typing never
# hides a real mouse movement, and each typed chunk hides key presses for a short moment only.
MOUSE_ECHO_S = 0.25
KEY_ECHO_S = 0.3
KEY_ECHO_PER_CHAR_S = 0.02
TYPE_CHUNK = 8  # characters typed at a time (each chunk can be interrupted)
# open_app verification: after the launcher keystrokes (fixed 0.6/0.8 s sleeps stay as the base timing), the
# foreground window is polled for up to this long for a title/process match against the requested app name.
# A poll miss does not mean the app failed to launch; it only means it could not be confirmed, so the result
# carries a warning instead of a silent ok.
OPEN_APP_VERIFY_S = 3.0
OPEN_APP_POLL_S = 0.25
ACTIONS = (
    "move",
    "click",
    "double_click",
    "right_click",
    "drag",
    "scroll",
    "type_text",
    "press_keys",
    "open_app",
    "open_url",
)
# v0.1.28: the step budget and the visual audit trail
DEFAULT_STEP_BUDGET = 40  # actions per approved control session when nothing is configured
TRAIL_CAP = 300  # audit-trail steps kept per session (matches the action log's bound)
MAX_SESSIONS = 12  # session trails kept in memory and on disk; older ones are dropped
BUDGET_EXHAUSTED = (
    "you have used all {budget} of this session's steps. Tell the person what is done and what is left, "
    "then ask them to let you continue: approving again starts a fresh session."
)
# An action sequence is one MCP call, not one grant: it is bounded so a single call cannot queue an unbounded
# batch of input.
SEQUENCE_MAX_STEPS = 25


def _target_label(detail: str) -> str:
    """The quoted element name in an action detail like "(120, 300) 'OK'", or "" when there was none."""
    m = re.search(r"'([^']{1,60})'", detail or "")
    return m.group(1) if m else ""


def describe_action(action: str, detail: str) -> str:
    """Plain words for what the agent did (no jargon): the "what did the agent just do" receipt lists one of
    these per action. ``detail`` is the raw detail string stored on the :class:`LogEntry`."""
    detail = str(detail or "")
    label = _target_label(detail)
    if action == "move":
        return "Moved the mouse"
    if action == "click":
        return f"Clicked on '{label}'" if label else f"Clicked at {detail.strip() or 'a spot on the screen'}"
    if action == "double_click":
        return f"Double-clicked '{label}'" if label else f"Double-clicked at {detail.strip() or 'a spot on the screen'}"
    if action == "right_click":
        return f"Right-clicked '{label}'" if label else f"Right-clicked at {detail.strip() or 'a spot on the screen'}"
    if action == "drag":
        return "Dragged something across the screen"
    if action == "scroll":
        return "Scrolled"
    if action == "type_text":
        n = detail.split(" ")[0] if detail else "0"
        return f"Typed {n} characters"
    if action == "press_keys":
        return f"Pressed {detail}" if detail else "Pressed some keys"
    if action == "open_app":
        return f"Opened {detail}" if detail else "Opened an app"
    if action == "open_url":
        return f"Opened {detail}" if detail else "Opened a link"
    return detail or action


PLAIN_END_REASONS = {
    "expired": "its time ran out",
    "stopped": "you stopped it",
    "you pressed Stop": "you pressed Stop",
    "the user chose to stop": "you chose to stop it",
    "the agent was disconnected": "the agent disconnected",
}


def plain_end_reason(reason: str) -> str:
    """Plain words for why a control grant ended (proposal #17 receipts)."""
    return PLAIN_END_REASONS.get(reason, str(reason or ""))


def purge_trail_folder(folder: Path | None) -> dict[str, int]:
    """Delete every persisted audit-trail artifact in ``folder``: the ``session-*.jsonl`` files and
    the action thumbnails (every PNG there is one; the folder holds nothing else). Returns how many
    of each were removed. Used when the audit trail is switched off and no host is running to do it."""
    removed = {"files": 0, "thumbnails": 0}
    if folder is None:
        return removed
    folder = Path(folder)
    if not folder.is_dir():
        return removed
    for pattern, key in (("session-*.jsonl", "files"), ("*.png", "thumbnails")):
        for path in folder.glob(pattern):
            with contextlib.suppress(Exception):
                path.unlink(missing_ok=True)
                removed[key] += 1
    return removed


class ControlRefused(ActionRefused):
    """An action that was not run; ``str(exc)`` is a plain-language reason the agent can relay."""


# v0.1.28: grant ends now tell the user in plain words ("time ran out", "you stopped it") so a silently
# expired grant is not a mystery. Maps the internal end reasons to non-technical phrasing.
END_CAUSE = {
    "expired": "time ran out",
    "stopped": "you stopped it",
    "the user chose to stop": "you stopped it",
    "the agent was disconnected": "it was disconnected",
}


@dataclass
class Grant:
    agent_id: str
    agent_name: str
    started: float
    expires: float
    paused: str = ""  # why it is paused ("" = running)
    # v0.1.24 hardening: set the moment the grant is paused or ended; an action in progress checks it before every
    # step (each drag step, each typed chunk, each key) and stops at once. A resume gets a fresh one.
    cancel: threading.Event = field(default_factory=threading.Event)
    ended: str = ""  # why it ended ("" = still active)
    trust_unknown: bool = False  # the person chose "Allow for this session" for actions it cannot identify
    # proposal #17: id of the newest log entry when the grant started; the receipt lists what came after it
    last_log_id: str = ""
    # v0.1.28: the control session's id (its audit trail), step budget and steps used so far (budget resets per grant)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    step_budget: int = 40
    steps_used: int = 0
    # proposal #21: the person picked "Yes, for the rest of this session". resume() then skips the prompt for the rest
    # of the session. A takeover still pauses at once; nothing auto-resumes.
    no_reprompt: bool = False


# The grant was paused or ended while an action was running (the engine's cancellation).
_Interrupted = ActionCancelled
# how a user source names itself in "paused because you used the ..."
_TAKEOVER_NAMES = {"gesture": "hand", "voice": "voice", "dictation": "keyboard", "macro": "shortcut", "user": "mouse"}

# proposal #21: the same takeover sources in plain words for the resume prompt.
_RESUME_REASON_WORDS = {
    "mouse": "you moved the mouse",
    "keyboard": "you used the keyboard",
    "hand": "you moved your hand",
    "voice": "you spoke",
    "shortcut": "you used a shortcut",
}


def _pause_in_plain_words(paused: str) -> str:
    """Turn the grant's "you used the {source}" pause reason into plain words (proposal #21)."""
    if paused.startswith("you used the "):
        src = paused[len("you used the ") :]
        return _RESUME_REASON_WORDS.get(src, paused)
    return paused


@dataclass
class LogEntry:
    id: str
    ts: float
    agent: str
    action: str
    detail: str
    result: str
    thumb: str = ""  # file name in the action log folder (PNG), "" when no screenshot was possible
    undo: dict[str, Any] | None = None
    agent_id: str = ""  # who did it (undo and the action log are per agent)
    source: str = AGENT  # the action engine source it ran as
    session: str = ""  # v0.1.28: the control session (grant id) this step belongs to


@dataclass
class _ElementQuery:
    """One agent's last ui_query: the foreground window fingerprint at query time plus the id -> Element
    snapshot. An element id resolves only while the same window is still in the foreground."""

    window: tuple[str, str, str, tuple[int, ...] | None] | None  # (title, process, class, rect)
    elements: dict[str, desktop.Element]
    generation: int = 0  # increments per agent per query; binds an id to one specific query attempt


@dataclass
class ControlHooks:
    """Platform access, injectable for tests."""

    backend: Any  # mouse/keyboard backend (RecordingBackend in tests; GuardedBackend(PynputBackend) in the app)
    uia: Any = field(default_factory=desktop.NullUIA)
    foreground: Callable[[], dict[str, Any] | None] = desktop.foreground_window
    secure_desktop: Callable[[], bool] = desktop.secure_desktop_active
    grab: Callable[[], Any] = desktop.grab_screen
    open_url: Callable[[str], Any] | None = None
    system_launcher: Callable[[str], Any] | None = None  # launch an allowlisted system tool (os.startfile)
    cursor_pos: Callable[[], tuple[int, int] | None] = lambda: None
    platform: str = ""
    overlay_on: Callable[[str], None] = lambda _name: None
    overlay_off: Callable[[], None] = lambda: None
    # the takeover banner (proposal #16): shown instead of overlay_off() while the grant is paused
    overlay_paused: Callable[[str], None] = lambda _name: None
    sleep: Callable[[float], None] = time.sleep
    engine: Any = None  # the shared ActionEngine in the app; None = a private one (tests, headless)


class ComputerControl:
    def __init__(
        self,
        hooks: ControlHooks,
        ask: Callable[..., dict[str, Any]],
        *,
        ttl_s: float = 1800.0,
        rate_per_s: float = 4.0,
        log_folder: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        always_allowed: Callable[[str, str], bool] = lambda _agent_id, _cat: False,
        allow_always: Callable[[str, str], None] = lambda _agent_id, _cat: None,
        step_budget: int = DEFAULT_STEP_BUDGET,
        audit_trail: bool = True,
        settings: Callable[[], Any] | None = None,
        notify: Callable[..., Any] | None = None,
    ) -> None:
        import sys

        self.hooks = hooks
        self.ask = ask
        self.notify = notify  # plain-words toast when a grant ends; None = silent (tests, headless)
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self.on_event = on_event
        self.always_allowed = always_allowed
        self.allow_always = allow_always
        # v0.1.28: the step budget and the audit trail follow the live settings when given (the person can
        # change them in Settings without restarting); otherwise the constructor values hold.
        self.step_budget = max(1, int(step_budget))
        self.audit_trail = bool(audit_trail)
        self._settings = settings
        self.platform = hooks.platform or sys.platform
        self.engine: ActionEngine = hooks.engine or ActionEngine(clock=clock, platform=self.platform)
        self.engine.set_agent_rate(rate_per_s, clock)
        self.engine.add_stop_listener(self._on_stop_all)
        self.engine.add_takeover_listener(self._on_user_action)
        self.log_folder = log_folder
        self.log: deque[LogEntry] = deque(maxlen=300)
        # v0.1.28: per-session audit trails (session id -> steps in order, oldest first), bounded like the log
        self.sessions: dict[str, list[LogEntry]] = {}
        self._session_order: list[str] = []
        self._session_agents: dict[str, str] = {}  # session id -> agent name (for the trail view)
        self.grant: Grant | None = None
        self._lock = threading.RLock()
        self._own_mouse_until = 0.0  # the agent's own mouse input may echo back until then
        self._own_keys_until = 0.0  # the agent's own key presses may echo back until then
        self._grace_until = 0.0  # just after the user approved (maybe with a thumbs up): that is not a takeover
        self.scales: dict[str, float] = {}  # agent -> last screenshot scale (for from_image coordinates)
        # v0.1.21: agent -> (origin x, origin y, scale) of its last screenshot (a monitor or a zoomed region)
        self.views: dict[str, tuple[float, float, float]] = {}
        # v0.1.27: agent -> {mark id: (screen x, screen y)} of the numbered boxes from its last
        # get_screen(annotate=true) (set-of-mark prompting for empty UIA trees)
        self.marks: dict[str, dict[int, tuple[int, int]]] = {}
        # v0.1.28 proposal #24 (hybrid grounding): each agent's last ui_query result: the foreground
        # window fingerprint plus id -> Element. A pointer or keyboard action may pass element_id=...
        # instead of coordinates; it resolves here, then acts as usual. Ids are per-agent: one agent's
        # query never replaces another's, and an id stops resolving when the foreground window changes.
        self._element_queries: dict[str, _ElementQuery] = {}
        self._query_generations: dict[str, int] = {}  # agent -> last query generation number

    def _budget(self) -> int:
        """The configured step budget (live when the app's settings are wired in)."""
        if self._settings is not None:
            with contextlib.suppress(Exception):
                return max(1, int(getattr(self._settings(), "control_step_budget", self.step_budget)))
        return self.step_budget

    def _trail_on(self) -> bool:
        """The visual audit trail switch (live when the app's settings are wired in)."""
        if self._settings is not None:
            with contextlib.suppress(Exception):
                return bool(getattr(self._settings(), "control_audit_trail", self.audit_trail))
        return self.audit_trail

    @property
    def bucket(self) -> Any:
        """The agents' rate limit (owned by the action engine)."""
        return self.engine.bucket

    def _on_stop_all(self, reason: str) -> None:
        """ "Stop everything" from any trigger ends the grant too."""
        if self.grant is not None:
            self._end(reason)

    def _on_user_action(self, source: str) -> bool:
        """The user's own gesture / voice / shortcut acted while an agent action was in flight: take over."""
        return self.user_input(_TAKEOVER_NAMES.get(source, source))

    # ---------------------------------------------------------------- events
    def _emit(self, topic: str, data: dict[str, Any]) -> None:
        if self.on_event:
            with contextlib.suppress(Exception):
                self.on_event(topic, data)

    # --------------------------------------------------------------- session
    def status(self) -> dict[str, Any]:
        g = self._check_expiry()
        if g is None:
            return {"active": False}
        return {
            "active": True,
            "agent": g.agent_name,
            "agent_id": g.agent_id,
            "paused": bool(g.paused),
            "paused_reason": g.paused,
            "expires_in_s": max(0, int(g.expires - self.clock())),
            "session": g.id,
            "step_budget": g.step_budget,
            "steps_used": g.steps_used,
            "steps_left": max(0, g.step_budget - g.steps_used),
        }

    def _check_expiry(self) -> Grant | None:
        with self._lock:
            g = self.grant
            if g is not None and self.clock() >= g.expires:
                self._end("expired")
                return None
            return g

    def request(self, agent_id: str, agent_name: str, why: str = "", minutes: float | None = None) -> dict[str, Any]:
        """Ask the user (on the PC) to let this agent use the computer. Blocks until answered or timed out."""
        g = self._check_expiry()
        if g is not None and g.agent_id == agent_id and not g.paused:
            return self.status()
        if g is not None and g.agent_id != agent_id:
            raise ControlRefused(f"{g.agent_name} is using the computer right now")
        ttl = self.ttl_s if minutes is None else max(1.0, min(float(minutes) * 60.0, self.ttl_s))
        details = [
            f"For up to {int(ttl // 60)} minutes.",
            "Move your hand or mouse to take over at any time.",
            "It will ask before sending, paying, deleting or installing anything,",
            "and before clicking something it can't identify.",
        ]
        if why:
            details.insert(0, f"Why: {str(why)[:150]}")
        res = self.ask(
            f"{agent_name} wants to use your computer",
            ["Allow", "Deny"],
            title="Use your computer?",
            agent=agent_name,
            details=details,
            timeout_s=60.0,
            urgency="high",
            kind="approve",
        )
        if res.get("answer") != "Allow":
            self._emit("control.denied", {"agent": agent_name, "method": res.get("method")})
            raise ControlRefused("the user did not allow it" if res.get("answer") else "no answer from the user")
        with self._lock:
            now = self.clock()
            g = Grant(agent_id, agent_name, now, now + ttl)
            g.step_budget = self._budget()  # v0.1.28: the budget resets with every new grant
            g.last_log_id = self.log[-1].id if self.log else ""  # proposal #17: the receipt's start
            self.grant = g
            self._start_session(g)
            self._grace_until = now + GRACE_S
        with contextlib.suppress(Exception):
            self.hooks.overlay_on(agent_name)
        self._emit("control.started", {"agent": agent_name, "minutes": int(ttl // 60), "method": res.get("method")})
        return self.status()

    def resume(self, agent_id: str) -> dict[str, Any]:
        g = self._check_expiry()
        if g is None or g.agent_id != agent_id:
            raise ControlRefused("this agent is not allowed to use the computer right now; ask with request_control")
        if not g.paused:
            return self.status()
        answer = "Continue"
        if not g.no_reprompt:  # the person picked "Yes, for the rest of this session" earlier this session
            reason = _pause_in_plain_words(g.paused)
            res = self.ask(
                f"{g.agent_name} stopped because {reason}. Let it keep going?",
                ["Continue", "Yes, for the rest of this session", "Stop"],
                title="Resume?",
                agent=g.agent_name,
                details=[
                    "It still stops the moment you take over again.",
                    "'Yes, for the rest of this session' skips this question until the grant ends.",
                ],
                timeout_s=60.0,
                urgency="high",
                kind="approve",
            )
            answer = res.get("answer") or ""
            if answer == "Yes, for the rest of this session":
                with self._lock:
                    g.no_reprompt = True
        if answer != "Continue" and answer != "Yes, for the rest of this session":
            self._end("the user chose to stop")
            raise ControlRefused("the user did not let it continue")
        with self._lock:
            if self.grant is not None:
                self.grant.paused = ""
                self.grant.cancel = threading.Event()  # the old takeover does not cancel new actions
                self._grace_until = self.clock() + GRACE_S
        with contextlib.suppress(Exception):
            self.hooks.overlay_on(g.agent_name)
        self._emit("control.resumed", {"agent": g.agent_name, "no_reprompt": g.no_reprompt})
        return self.status()

    def _end(self, reason: str) -> None:
        with self._lock:
            g, self.grant = self.grant, None
            if g is not None:
                g.ended = reason
                g.cancel.set()  # an action in progress stops before its next step
                # The grant is over: its agent's element ids must not outlive the session.
                if g is not None:
                    self._element_queries.pop(g.agent_id, None)
        receipt = self.receipt(g) if g is not None else None
        if g is not None:
            self._trail_event(g, {"event": "ended", "reason": reason, "steps_used": g.steps_used})
            with contextlib.suppress(Exception):
                self.hooks.overlay_off()
            self._emit("control.ended", {"agent": g.agent_name, "reason": reason})
            if receipt is not None:  # proposal #17: "what did the agent just do"
                self._emit("control.receipt", receipt)

            if self.notify is not None:
                cause = END_CAUSE.get(reason, reason)
                with contextlib.suppress(Exception):
                    # a toast failure must never break grant teardown
                    self.notify(
                        f"{g.agent_name}'s turn with your computer just ended ({cause}).",
                        level="info",
                        agent=g.agent_name,
                    )

    # ------------------------------------------------------- v0.1.28 sessions
    def _start_session(self, g: Grant) -> None:
        """A new grant starts a fresh audit trail (bounded) and a fresh step budget."""
        self.sessions[g.id] = []
        self._session_agents[g.id] = g.agent_name
        self._session_order.append(g.id)
        self._prune_sessions()
        self._trail_event(
            g,
            {
                "event": "started",
                "agent": g.agent_name,
                "agent_id": g.agent_id,
                "step_budget": g.step_budget,
                "started": g.started,
            },
        )

    def _prune_sessions(self) -> None:
        """Keep only the newest sessions, in memory and on disk (the PNG thumbnails are pruned with the log)."""
        while len(self._session_order) > MAX_SESSIONS:
            old = self._session_order.pop(0)
            trail = self.sessions.pop(old, None) or []
            self._session_agents.pop(old, None)
            if self.log_folder is not None:
                thumbs = {e.thumb for e in trail if e.thumb}
                thumbs |= {e.thumb for e in self.log if e.session == old and e.thumb}
                for name in thumbs:
                    with contextlib.suppress(Exception):
                        (self.log_folder / name).unlink(missing_ok=True)
                for e in self.log:  # the names now dangle; the files are gone
                    if e.session == old and e.thumb:
                        e.thumb = ""
                with contextlib.suppress(Exception):
                    (self.log_folder / f"session-{old}.jsonl").unlink(missing_ok=True)

    def clear_trail_data(self) -> dict[str, int]:
        """Forget the whole audit trail: the in-memory sessions plus every persisted session file and
        thumbnail. Used when the user turns the audit trail off (off means gone). The in-memory safety
        action log itself stays (undo still needs it); its thumbnail names are blanked with the files.
        Returns how much was removed: {"sessions": n, "files": n, "thumbnails": n}."""
        with self._lock:
            sessions = len(self._session_order)
            self.sessions.clear()
            self._session_agents.clear()
            self._session_order.clear()
        removed = purge_trail_folder(self.log_folder)
        with self._lock:
            for e in self.log:
                e.thumb = ""
        return {"sessions": sessions, **removed}

    def _session_file(self, session_id: str) -> Path | None:
        if self.log_folder is None:
            return None
        return self.log_folder / f"session-{session_id}.jsonl"

    def _trail_event(self, g: Grant, event: dict[str, Any]) -> None:
        """One line in the session's persisted trail (the audit trail switch gates this)."""
        if not self._trail_on():
            return
        path = self._session_file(g.id)
        if path is None:
            return
        try:
            import json

            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "session": g.id, **event}, default=str) + "\n")
        except Exception:
            pass

    def receipt(self, g: Grant) -> dict[str, Any]:
        """Build the "what did the agent just do" receipt for a grant that just ended: one plain-language line
        per action this grant ran (the existing ``actions()`` log with the existing thumbnails), plus whether
        anything can still be put back. The app shows it as a toast plus one dialog with an "Undo all" button."""
        with self._lock:
            rows = [e for e in self.log if e.agent_id == g.agent_id]
        if g.last_log_id:  # only what this grant did (the log keeps the agent's older actions too)
            idx = next((i + 1 for i, e in enumerate(rows) if e.id == g.last_log_id), 0)
            rows = rows[idx:]
        items = []
        for e in rows:
            text = describe_action(e.action, e.detail)
            if e.result == "stopped":
                text += " (stopped part-way)"
            elif e.result == "undone":
                text += " (you already put this back)"
            items.append(
                {
                    "id": e.id,
                    "ts": e.ts,
                    "text": text,
                    "thumb": e.thumb,  # file name in the action log folder (PNG)
                    "undoable": e.undo is not None and e.result == "done",
                }
            )
        n = len(items)
        headline = (
            f"{g.agent_name} did {n} thing{'s' if n != 1 else ''}."
            if n
            else (f"{g.agent_name} did not do anything this time.")
        )
        return {
            "agent": g.agent_name,
            "agent_id": g.agent_id,
            "reason": g.ended,
            "ended": plain_end_reason(g.ended),  # plain words for "why it ended"
            "title": f"{g.agent_name} finished using your computer",
            "headline": headline,
            "count": n,
            "items": items,
            "undoable": any(i["undoable"] for i in items),
            "log_folder": str(self.log_folder) if self.log_folder else "",
        }

    def stop(self, reason: str = "stopped") -> bool:
        """Global stop (hotkey, "stop", open palm, Disconnect, revoke): ends the grant at once."""
        had = self.grant is not None
        self._end(reason)
        return had

    def revoke_agent(self, agent_id: str, reason: str = "the agent was disconnected") -> None:
        """End this agent's grant now (an action in progress stops and lets go of buttons and keys) and forget
        its screenshot geometry."""
        if self.grant is not None and self.grant.agent_id == agent_id:
            self._end(reason)
        self.views.pop(agent_id, None)
        self.scales.pop(agent_id, None)
        self.marks.pop(agent_id, None)
        # Revocation invalidates the agent's element ids too: they must not survive disconnect.
        with self._lock:
            self._element_queries.pop(str(agent_id), None)

    # -------------------------------------------------------------- takeover
    def note_own_mouse(self, seconds: float = MOUSE_ECHO_S) -> None:
        self._own_mouse_until = self.clock() + seconds

    def note_own_keys(self, seconds: float = KEY_ECHO_S) -> None:
        self._own_keys_until = self.clock() + seconds

    def user_input(self, source: str) -> bool:
        """Real user input (mouse, keyboard, gesture). Pauses an active grant. True when it paused."""
        with self._lock:
            g = self.grant
            if g is None or g.paused:
                return False
            now = self.clock()
            if now < self._grace_until:
                return False  # the approval itself (a click, a thumbs up) is not a takeover
            if source == "mouse" and now < self._own_mouse_until:
                return False  # our own mouse input echoing back through the OS listener
            if source == "keyboard" and now < self._own_keys_until:
                return False  # our own key presses echoing back (one short window per typed chunk)
            g.paused = f"you used the {source}"
            g.cancel.set()  # stop the action in progress (typing, a drag) before its next step
        # the banner stays up while paused ("Paused - you took over from <agent>", with buttons to continue or
        # stop), instead of the border vanishing with no explanation
        with contextlib.suppress(Exception):
            self.hooks.overlay_paused(g.agent_name)
        self._emit("control.paused", {"agent": g.agent_name, "reason": g.paused})
        return True

    def paused_banner_answer(self, answer: str) -> bool:
        """The user answered the paused banner directly: the click IS the answer (no prompt round-trip).

        "continue" resumes the grant without asking again (the banner click itself is covered by the same short
        grace as an approval click, so it is not a takeover); anything else stops the grant. True when it acted
        on a live grant; False when the grant had already ended another way (the banner closes with its stop
        file, and a click on a stale banner does nothing).
        """
        with self._lock:
            g = self.grant
            if g is None:
                return False
            name = g.agent_name
            stop = answer != "continue"
            resumed = answer == "continue" and bool(g.paused)
            if resumed:
                g.paused = ""
                g.cancel = threading.Event()  # the old takeover does not cancel new actions
                self._grace_until = self.clock() + GRACE_S  # the banner click itself is not a takeover
        if stop:
            self._end("the user chose to stop")
            return True
        with contextlib.suppress(Exception):
            self.hooks.overlay_on(name)  # back to the control border/banner (this clears the paused one)
        if resumed:
            self._emit("control.resumed", {"agent": name, "via": "paused banner"})
        return True

    # ---------------------------------------------------------------- acting
    def require_grant(self, agent_id: str) -> Grant:
        """The live, unpaused grant for this agent (raises ControlRefused otherwise): the approved-session
        gate for tools that ride the computer-control session, like the staged shell."""
        return self._require(agent_id)

    def _require_grant(self, agent_id: str) -> Grant:
        """The live, unpaused grant for this agent, without consuming a budget step."""
        g = self._check_expiry()
        if g is None:
            raise ControlRefused("not allowed to use the computer right now; call request_control first")
        if g.agent_id != agent_id:
            raise ControlRefused(f"{g.agent_name} is using the computer, not this agent")
        if g.paused:
            raise ControlRefused(f"paused because {g.paused}; call resume_control to ask the user to continue")
        return g

    def _consume_step(self, g: Grant) -> None:
        """Spend one step of the session's budget; refuses in plain words when the budget is exhausted."""
        # Each attempted agent action, including undo, consumes one step. An exhausted session
        # refuses politely, in plain words, until the person approves again (a new grant = a fresh budget).
        with self._lock:
            if g.steps_used >= g.step_budget:
                raise ControlRefused(BUDGET_EXHAUSTED.format(budget=g.step_budget))
            g.steps_used += 1

    def _require(self, agent_id: str) -> Grant:
        g = self._require_grant(agent_id)
        self._policy_check(agent_id)
        self._consume_step(g)
        return g

    def _policy_check(self, agent_id: str) -> None:
        """The engine's per-action policy (secure desktop, rate limit) without repeating the grant approval.

        Called once per single action by :meth:`_require`, and once per sub-step by :meth:`act_sequence`, so a
        batch cannot spend the whole rate-limit budget on one check."""
        try:  # the engine's agent policy: secure desktop (fail closed on Windows) and the rate limit
            self.engine.check(AGENT, agent_id, secure_desktop=self.hooks.secure_desktop, platform=self.platform)
        except ControlRefused:
            raise
        except ActionRefused as exc:
            raise ControlRefused(str(exc)) from None

    def _confirm_risky(self, g: Grant, label: str, what: str) -> None:
        m = RISKY.search(label or "")
        if not m:
            return
        cat = m.group(1).lower()
        if self.always_allowed(g.agent_id, cat):
            return
        res = self.ask(
            f"{g.agent_name} wants to {what}",
            ["Allow", "Deny", "Always allow for this agent"],
            title="Are you sure?",
            agent=g.agent_name,
            details=[f"'{label[:80]}' can't always be undone."],
            timeout_s=60.0,
            urgency="high",
            kind="confirm",
        )
        ans = res.get("answer")
        if ans == "Always allow for this agent":
            self.allow_always(g.agent_id, cat)
            return
        if ans != "Allow":
            self._emit("control.risky_denied", {"agent": g.agent_name, "target": label})
            raise ControlRefused(f"the user did not allow '{label[:60]}'")

    def _confirm_unknown(self, g: Grant, what: str) -> None:
        """A consequential action whose target has no name (no UI Automation, nothing found there, coordinates
        only): the person decides, once or for the rest of this session."""
        if g.trust_unknown:
            return
        res = self.ask(
            f"{g.agent_name} wants to {what}",
            ["Allow", "Deny", ALLOW_SESSION],
            title="Are you sure?",
            agent=g.agent_name,
            details=[UNKNOWN_NOTE],
            timeout_s=60.0,
            urgency="high",
            kind="confirm",
        )
        ans = res.get("answer")
        if ans == ALLOW_SESSION:
            g.trust_unknown = True
            return
        if ans != "Allow":
            self._emit("control.unknown_denied", {"agent": g.agent_name, "what": what})
            raise ControlRefused(f"the user did not allow it ({what[:80]})")

    def _confirm_targets(self, g: Grant, what: str, labels: list[str]) -> None:
        """Clicks and drags: a risky target (either end of a drag) asks as before; a target it cannot identify
        asks too; named, harmless targets go ahead."""
        kinds = [classify(label) for label in labels]
        if "risky" in kinds:
            for label, kind in zip(labels, kinds):
                if kind == "risky":
                    self._confirm_risky(g, label, what)
            return
        if "unknown" in kinds:
            self._confirm_unknown(g, what)

    def _confirm_keys(self, g: Grant, what: str) -> None:
        """Enter, Delete or a shortcut acts on whatever has the focus: risky focus asks as before; with no known
        focus (no UI Automation, nothing focused) it asks too. A named text box goes ahead."""
        label = self._focused_label()
        context = label or self._window_title()
        if classify(context) == "risky":
            self._confirm_risky(g, context, what)
        elif not label.strip():
            self._confirm_unknown(g, what)

    @staticmethod
    def _where(label: str, x: int, y: int) -> str:
        return f"'{label[:60]}'" if label.strip() else f"the spot at ({x}, {y})"

    # v0.1.28 proposal #24 (hybrid grounding): UI Automation gives exact element bounds for free; raw pixel
    # grounding is the weakest link. ui_query lists the foreground window's interactable elements with short
    # ids; pointer and keyboard accept an element_id and resolve it to bounds, then act as with coordinates.
    # v0.1.27: a numbered box from get_screen(annotate=true): "mark:14", "mark 14", "#14"
    MARK_REF = re.compile(r"(?i)^(?:mark[:#]?\s*|#)(\d{1,4})$")
    UI_QUERY_DEPTH = 4
    UI_QUERY_NODES = 400
    UI_QUERY_LIMIT = 200

    def _window_fingerprint(self) -> tuple[str, str, str, tuple[int, ...] | None] | None:
        """(title, process, class, rect) of the foreground window, or None when it cannot be told."""
        try:
            fg = self.hooks.foreground()
        except Exception:
            return None
        if not fg:
            return None
        rect = fg.get("rect")
        return (
            str(fg.get("title", "")),
            str(fg.get("process", "")),
            str(fg.get("class", "")),
            tuple(int(v) for v in rect) if rect else None,
        )

    def ui_query(self, agent_id: str) -> dict[str, Any]:
        """The foreground window's interactable elements: ``[{id, name, role, bounds}]`` with screen-pixel
        bounds. Ids belong to the querying agent only and stay valid until that agent's next query; an id
        stops resolving when the foreground window changes, so a stale id can never click into a new window."""
        aid = str(agent_id)
        # Invalidate the prior registry at the START: a failed or incomplete query must not leave the old
        # ids usable. The new registry is installed only on success below.
        with self._lock:
            self._element_queries.pop(aid, None)
        uia = self.hooks.uia
        if not getattr(uia, "available", False):
            return {
                "available": False,
                "reason": str(getattr(uia, "reason", "UI Automation is not available on this system")),
            }
        try:
            root = uia.tree(self.UI_QUERY_DEPTH, self.UI_QUERY_NODES)
        except Exception as exc:  # the tree walk itself failed
            return {"available": False, "reason": f"could not read the UI tree: {exc}"}
        found: list[desktop.Element] = []
        stack = [root] if root is not None else []
        while stack and len(found) < self.UI_QUERY_LIMIT:
            el = stack.pop()
            if el.enabled and el.rect is not None:
                found.append(el)
            stack.extend(reversed(el.children or []))
        registry: dict[str, desktop.Element] = {}
        items: list[dict[str, Any]] = []
        for i, el in enumerate(found, 1):
            rect = el.rect
            assert rect is not None  # filtered above
            eid = f"e{i}"
            registry[eid] = el
            items.append({"id": eid, "name": el.name, "role": el.control_type, "bounds": list(rect)})
        with self._lock:
            gen = self._query_generations.get(aid, 0) + 1
            self._query_generations[aid] = gen
            self._element_queries[aid] = _ElementQuery(self._window_fingerprint(), registry, gen)
        window: dict[str, str] | None = None
        try:
            fg = self.hooks.foreground()
            if fg:
                window = {"title": str(fg.get("title", "")), "process": str(fg.get("process", ""))}
        except Exception:
            window = None
        self._emit("control.ui_query", {"count": len(items)})
        return {"available": True, "window": window, "count": len(items), "elements": items}

    def _resolve_element_id(self, agent_id: str, element_id: str) -> desktop.Element:
        """A ui_query id back to its element. Refuses when the id is unknown, belongs to another agent's
        query, or the foreground window changed since the query: a stale id never clicks."""
        key = str(element_id).strip()
        with self._lock:
            query = self._element_queries.get(str(agent_id))
        if query is None:
            raise ControlRefused(f"unknown element id {element_id!r}; call ui_query first and use one of its ids")
        current = self._window_fingerprint()
        # A missing fingerprint must never compare equal to another missing fingerprint: without a valid
        # foreground identity on both sides, the id cannot be trusted.
        if query.window is None or current is None:
            raise ControlRefused("could not verify the foreground window; run ui_query again and use a fresh id")
        if query.window != current:
            raise ControlRefused(
                "the foreground window changed since your ui_query; run ui_query again and use a fresh id"
            )
        el = query.elements.get(key)
        if el is None or el.center is None:
            raise ControlRefused(f"unknown element id {element_id!r}; call ui_query first and use one of its ids")
        self._revalidate_element(el)
        return el

    def _revalidate_element(self, el: desktop.Element) -> None:
        """Refuse when the live UI no longer shows this element where the snapshot recorded it. Fail closed:
        a missing, failed, or inconclusive live lookup refuses rather than falling back to the cached element."""
        center = el.center
        if center is None:
            return
        try:
            live = self.hooks.uia.element_at(center[0], center[1])
        except Exception as exc:
            raise ControlRefused(
                f"could not revalidate the element before acting ({exc}); run ui_query again and use a fresh id"
            ) from exc
        if live is None:
            raise ControlRefused("that element is gone or has moved; run ui_query again and use a fresh id")
        if el.automation_id and live.automation_id:
            if live.automation_id != el.automation_id:
                raise ControlRefused("that element is gone or has moved; run ui_query again and use a fresh id")
            return
        if el.name and (live.name != el.name or live.control_type != el.control_type):
            raise ControlRefused("that element is gone or has moved; run ui_query again and use a fresh id")

    def _focus_element_id(self, g: Grant, agent_id: str, step: Callable[[], Any], element_id: str) -> str:
        """Click the center of a ui_query element so keyboard input lands in it; returns the element's name."""
        x, y, label = self._point(agent_id, None, None, None, element_id, False)
        self._confirm_targets(g, f"click {self._where(label, x, y)}", [label])
        self.note_own_mouse()
        step().move(x, y)
        step().click(1)
        return label

    def _point(
        self, agent_id: str, x: Any, y: Any, element: str | None, element_id: str | None, from_image: bool
    ) -> tuple[int, int, str]:
        if element:
            mark = self.MARK_REF.match(str(element).strip())
            if mark is not None:  # a set-of-mark reference: needs no UI Automation; wins over element_id
                mid = int(mark.group(1))
                pt = self.marks.get(agent_id, {}).get(mid)
                if pt is None:
                    raise ControlRefused(
                        f"no annotation {mid} on your last screenshot; call get_screen(annotate=true) first"
                    )
                return int(pt[0]), int(pt[1]), f"mark {mid}"
        if element_id:  # exact bounds from the agent's last ui_query; beats fuzzy name matching and coordinates
            el = self._resolve_element_id(agent_id, element_id)
            center = el.center
            assert center is not None  # _resolve_element_id refuses bound-less elements
            return int(center[0]), int(center[1]), el.name
        if element:
            if not getattr(self.hooks.uia, "available", False):
                raise ControlRefused("acting by element name needs UI Automation (Windows); use coordinates")
            el = self.hooks.uia.find(str(element))
            if el is None or el.center is None:
                raise ControlRefused(f"no visible element named like '{element}'")
            cx, cy = el.center
            return int(cx), int(cy), el.name
        if x is None or y is None:
            raise ControlRefused("give x and y, or an element name")
        fx, fy = float(x), float(y)
        if from_image:
            view = self.views.get(agent_id)
            if view is not None:
                ox, oy, s = view
                fx, fy = ox + fx / (s or 1.0), oy + fy / (s or 1.0)
            else:
                s = self.scales.get(agent_id) or 1.0
                fx, fy = fx / s, fy / s
        xi, yi = round(fx), round(fy)
        label = ""
        try:
            el = self.hooks.uia.element_at(xi, yi)
            label = el.name if el is not None else ""
        except Exception:
            label = ""
        return xi, yi, label

    def _typing_target_ok(self) -> None:
        fg = None
        try:
            fg = self.hooks.foreground()
        except Exception:
            fg = None
        if fg and desktop.is_terminal(fg.get("process")):
            raise ControlRefused("agents cannot type into terminals or system tools")
        uia = self.hooks.uia
        if getattr(uia, "available", False):
            el = uia.focused()
            if el is not None and el.is_password:
                raise ControlRefused("that is a password field; agents never type passwords")
            return
        if self.platform == "win32":
            raise ControlRefused("typing needs UI Automation to rule out password fields, and it is not available")
        if fg and desktop.PASSWORD_TITLE.search(str(fg.get("title", ""))):
            raise ControlRefused("this window looks like a sign-in or password prompt; agents never type there")

    def _focused_label(self) -> str:
        uia = self.hooks.uia
        try:
            el = uia.focused() if getattr(uia, "available", False) else None
        except Exception:
            el = None
        return el.name if el is not None else ""

    def act(self, agent_id: str, action: str, **kw: Any) -> dict[str, Any]:
        """Run one action. Raises :class:`ControlRefused` with a plain reason when it is not allowed."""
        if action not in ACTIONS:
            raise ControlRefused(f"unknown action {action!r}; use {', '.join(ACTIONS)}")
        g = self._require(agent_id)
        b = self.hooks.backend
        try:
            with self.engine.begin(
                AGENT, action, actor=agent_id, token=g.cancel, backend=b, sleep=self.hooks.sleep, audit=False
            ) as a:
                return self._act(g, agent_id, action, kw, b, a)
        except ActionCancelled:
            reason = g.paused or g.ended or "it was stopped"
            self._log(g, action, "stopped part-way", "stopped", None)
            self._emit("control.interrupted", {"agent": g.agent_name, "action": action, "reason": reason})
            raise ControlRefused(f"stopped part-way because {reason}") from None

    def _act(
        self,
        g: Grant,
        agent_id: str,
        action: str,
        kw: dict[str, Any],
        b: Any,
        a: Action,
        *,
        record: bool = True,
    ) -> dict[str, Any]:
        """One action inside the engine's :meth:`~away.actions_engine.ActionEngine.begin`: ``a.step()`` before
        every input (stops at once on a pause, takeover or stop), held buttons released if it does not finish.

        ``record=False`` is for :meth:`act_sequence`: the sub-step still gets its engine audit entry and all the
        safety checks, but it does not append its own :class:`LogEntry` - the sequence records one entry with all
        the sub-step detail.
        """
        step, wait = a.step, a.wait
        undo: dict[str, Any] | None = None
        extra: dict[str, Any] = {}  # per-action result fields beyond ok/action/detail/log_id
        before = None
        try:
            before = self.hooks.cursor_pos()
        except Exception:
            before = None
        detail = ""
        warning: str | None = None  # set by actions that verify after the fact (open_app)
        if action in ("move", "click", "double_click", "right_click"):
            x, y, label = self._point(
                agent_id, kw.get("x"), kw.get("y"), kw.get("element"), kw.get("element_id"), bool(kw.get("from_image"))
            )
            if action == "right_click":  # it only opens a menu: asks for a risky target only
                self._confirm_risky(g, label, f"right-click '{label}'")
            elif action != "move":
                verb = "double-click" if action == "double_click" else "click"
                self._confirm_targets(g, f"{verb} {self._where(label, x, y)}", [label])
            if action == "right_click" and not hasattr(b, "click_button"):
                raise ControlRefused("right click is not supported on this system")
            self.note_own_mouse()
            step().move(x, y)
            if action == "click":
                step().click(1)
            elif action == "double_click":
                step().click(2)
            elif action == "right_click":
                step().click_button("right", 1)
            detail = f"({x}, {y})" + (f" '{label}'" if label else "")
            # proposal #6: the resolved pre-click label reaches the agent in the result, so it can see
            # e.g. "clicked 'Desktop', not 'Start'" and self-correct instead of taking another screenshot
            extra["target_label"] = label
            if action == "move" and before is not None:
                undo = {"move": list(before)}
        elif action == "drag":
            x1, y1, l1 = self._point(
                agent_id, kw.get("x"), kw.get("y"), kw.get("element"), kw.get("element_id"), bool(kw.get("from_image"))
            )
            x2, y2, l2 = self._point(
                agent_id,
                kw.get("x2"),
                kw.get("y2"),
                kw.get("to_element"),
                kw.get("to_element_id"),
                bool(kw.get("from_image")),
            )
            # both ends count: dropping a file on the Recycle Bin is as risky as clicking Delete
            self._confirm_targets(g, f"drag {self._where(l1, x1, y1)} to {self._where(l2, x2, y2)}", [l1, l2])
            self.note_own_mouse()
            step().move(x1, y1)
            a.press_button()
            steps = 8
            for i in range(1, steps + 1):
                self.note_own_mouse()
                step().move(int(x1 + (x2 - x1) * i / steps), int(y1 + (y2 - y1) * i / steps))
                wait(0.02)
            a.release_button()
            detail = f"({x1}, {y1}) -> ({x2}, {y2})"
        elif action == "scroll":
            dy = max(-50.0, min(50.0, float(kw.get("dy") or 0.0)))
            dx = max(-50.0, min(50.0, float(kw.get("dx") or 0.0)))
            if kw.get("x") is not None or kw.get("element") or kw.get("element_id"):
                x, y, _l = self._point(
                    agent_id,
                    kw.get("x"),
                    kw.get("y"),
                    kw.get("element"),
                    kw.get("element_id"),
                    bool(kw.get("from_image")),
                )
                self.note_own_mouse()
                step().move(x, y)
            self.note_own_mouse()
            if dy:
                step().scroll(dy)
            if dx and hasattr(b, "hscroll"):
                step().hscroll(dx)
            detail = f"dy={dy:g} dx={dx:g}"
            undo = {"scroll": [-dy, -dx]}
        elif action == "type_text":
            text = str(kw.get("text") or "")
            if not text or len(text) > 2000:
                raise ControlRefused("type between 1 and 2000 characters at a time")
            detail = f"{len(text)} characters"
            if kw.get("element_id"):  # click the field first so the text lands there
                tlabel = self._focus_element_id(g, agent_id, step, str(kw["element_id"]))
                detail = f"{len(text)} characters into '{tlabel}'" if tlabel.strip() else detail
            self._typing_target_ok()
            label = self._focused_label()
            title = self._window_title()
            extra["target"] = {"window": title, "element": label}
            expect = str(kw.get("expect") or "").strip()
            if expect and expect.lower() not in title.lower():
                if title.strip():
                    raise ControlRefused(
                        f"the focused window ({title!r}) does not contain {expect!r}: "
                        "refusing to type into the wrong window"
                    )
                raise ControlRefused(
                    f"I cannot see a focused window title to check {expect!r} against: "
                    "refusing to type into the wrong window"
                )
            if "\n" in text:
                self._confirm_keys(g, "press Enter")
                # Re-validate after the user prompt: the focused window may have
                # changed while the prompt was up. Fail closed if it did.
                if expect:
                    title = self._window_title()
                    if expect.lower() not in title.lower():
                        raise ControlRefused(
                            f"the focused window changed while confirming ({title!r} does not contain "
                            f"{expect!r}): refusing to type into the wrong window"
                        )
            if not hasattr(b, "type_text"):
                raise ControlRefused("typing is not supported on this system")
            with self._lock:
                self._own_mouse_until = 0.0  # while it types, any real mouse movement is the person
            typed = 0
            for i in range(0, len(text), TYPE_CHUNK):
                # Re-verify the target window before each chunk: focus may have changed
                # during the previous chunk. Abort before typing into the wrong app.
                if expect:
                    title = self._window_title()
                    if expect.lower() not in title.lower():
                        # partial completion is reported honestly via the exception message
                        raise ControlRefused(
                            f"the focused window changed while typing ({title!r} does not contain "
                            f"{expect!r}): stopped after {typed} of {len(text)} characters"
                        )
                chunk = text[i : i + TYPE_CHUNK]
                self.note_own_keys(KEY_ECHO_S + KEY_ECHO_PER_CHAR_S * len(chunk))
                step().type_text(chunk)
                typed += len(chunk)
            detail = f"{len(text)} characters"
            undo = {"keys": ["cmd" if self.platform == "darwin" else "ctrl", "z"]}
        elif action == "press_keys":
            from away.control.keys import parse_combo

            combo = str(kw.get("keys") or "").strip()
            try:
                keys = parse_combo(combo)
            except ValueError as exc:
                raise ControlRefused(str(exc)) from None
            norm = {("cmd" if k in ("win", "super", "cmd", "windows") else k) for k in keys}
            if any(bad <= norm for bad in BLOCKED_COMBOS):
                raise ControlRefused(f"{combo} is blocked for agents")
            if not keys:
                raise ControlRefused("no keys given")
            if kw.get("element_id"):  # click the field first so the keys land there
                self._focus_element_id(g, agent_id, step, str(kw["element_id"]))
            if any(k in ENTER_KEYS for k in keys) and len(keys) == 1:
                self._confirm_keys(g, "press Enter")
            elif any(k in CONSEQUENTIAL_KEYS for k in keys) or any(k in SHORTCUT_MODIFIERS for k in keys):
                self._confirm_keys(g, f"press {combo}")
            printable = len(keys) == 1 and len(keys[0]) == 1
            if printable:
                self._typing_target_ok()
            if len(keys) > 1 and not hasattr(b, "hotkey"):
                raise ControlRefused("key combinations are not supported on this system")
            self.note_own_keys()
            if len(keys) == 1:
                step().key(keys[0])
            else:
                step().hotkey(keys)
            detail = combo
        elif action == "open_url":
            url = str(kw.get("url") or "").strip()
            if not re.match(r"^https?://[^\s]{1,2000}$", url):
                raise ControlRefused("only http(s) links can be opened")
            opener = self.hooks.open_url
            if opener is None:
                import webbrowser

                opener = webbrowser.open
            opener(url)
            detail = url[:200]
        elif action == "open_app":
            name = str(kw.get("name") or "").strip()
            if kw.get("system"):
                # Proposal #2: an allowlisted system tool, launched through the shell - no synthetic
                # keystrokes, no command line built. The blocklist above stays untouched.
                key = name.lower().replace(" ", "-")
                target = SAFE_SYSTEM_TOOLS.get(key)
                if target is None:
                    raise ControlRefused("unknown system tool; use one of: " + ", ".join(sorted(SAFE_SYSTEM_TOOLS)))
                self._open_system_tool(target)
                detail = f"system tool '{key}'"
            else:
                if not APP_NAME.match(name):
                    raise ControlRefused("give a plain app name, like 'Notepad' or 'Spotify'")
                if BLOCKED_APPS.match(name.replace(" ", "")) or name.lower() in (
                    "command prompt",
                    "terminal",
                    "powershell",
                    "registry editor",
                ):
                    raise ControlRefused("agents cannot open terminals or system tools")
                if not self._open_app(name, step, wait):
                    warning = (
                        f"could not confirm '{name}' opened within {OPEN_APP_VERIFY_S:g} s (the foreground "
                        "window did not match its name); the launcher keystrokes may have done nothing. "
                        "Ask the person whether it opened before typing into it."
                    )
                detail = name
        a.ok = True
        # the feature's click-verification extras (e.g. target_label) ride along in `extra`; the
        # observed effect fields stay explicit for a stable result shape.
        foreground = self._foreground_state()
        focused = self._focused_label()
        if not record:
            self._audit(g, action, detail, "done")
            return {
                "ok": True,
                "action": action,
                "detail": detail,
                "log_id": "",
                "undo": undo,
                # the observed effect: what the desktop looks like right after the input landed, so the
                # agent can see e.g. "clicked 'Desktop', not 'Start'" and self-correct
                "foreground": foreground,
                "focused": focused,
                **extra,
            }
        entry = self._log(g, action, detail, "warned" if warning else "done", undo)
        self._emit("control.action", {"agent": g.agent_name, "action": action, "detail": detail, "id": entry.id})
        # v0.1.28: the agent is told its remaining budget with every action result
        out: dict[str, Any] = {
            "ok": True,
            "action": action,
            "detail": detail,
            "log_id": entry.id,
            "steps_left": max(0, g.step_budget - g.steps_used),
            "undo": undo,
            # the observed effect: what the desktop looks like right after the input landed, so the
            # agent can see e.g. "clicked 'Desktop', not 'Start'" and self-correct
            "foreground": foreground,
            "focused": focused,
            **extra,
        }
        if warning:
            out["warning"] = warning
        return out

    def act_sequence(self, agent_id: str, steps: Any) -> dict[str, Any]:
        """Run several actions in one call: one grant check, one log entry, one engine Action. Every sub-step
        still runs through :meth:`_act`, so the risky/unknown "Are you sure?" prompts, the password and terminal
        refusals, the per-step engine audit and the per-step rate limit apply exactly as they do for single
        actions - a batch is not a rubber stamp. Stops at the first refusal, cancel or takeover and reports how
        far it got; a part-way stop or refusal still records one combined log entry with the completed steps."""
        if not isinstance(steps, list) or not steps:
            raise ControlRefused("sequence needs at least one step, e.g. [{'action': 'move', 'x': 10, 'y': 10}]")
        if len(steps) > SEQUENCE_MAX_STEPS:
            raise ControlRefused(f"a sequence is limited to {SEQUENCE_MAX_STEPS} steps at a time")
        plan: list[tuple[str, dict[str, Any]]] = []
        for i, s in enumerate(steps, 1):
            if not isinstance(s, dict):
                raise ControlRefused(f"step {i} must be an object like {{'action': 'move', ...}}")
            action = s.get("action")
            if action not in ACTIONS:
                raise ControlRefused(f"step {i}: unknown action {action!r}; use {', '.join(ACTIONS)}")
            plan.append((str(action), {k: v for k, v in s.items() if k != "action"}))
        g = self._require_grant(agent_id)
        self._policy_check(agent_id)
        # v0.1.28: a sequence spends one budget step per sub-step, like one per single action. It must fit
        # in the remaining budget before it starts: a batch cannot spend steps it does not have.
        with self._lock:
            remaining = g.step_budget - g.steps_used
        if len(plan) > remaining:
            raise ControlRefused(
                f"that sequence needs {len(plan)} steps but only {max(0, remaining)} are left in this "
                "session; use a shorter sequence or ask the person to approve again for a fresh budget"
            )
        b = self.hooks.backend
        done: list[dict[str, Any]] = []
        try:
            with self.engine.begin(
                AGENT, "sequence", actor=agent_id, token=g.cancel, backend=b, sleep=self.hooks.sleep, audit=False
            ) as a:
                for action, kw in plan:
                    a.step()  # a pause, takeover or stop ends the batch before its next sub-step
                    # one rate-limit token per sub-step, like one per single action: a batch must not spend
                    # the whole budget on the single grant-time check
                    self._policy_check(agent_id)
                    self._consume_step(g)  # one budget step per executed sub-step
                    done.append(self._act(g, agent_id, action, kw, b, a, record=False))
        except ActionCancelled:
            reason = g.paused or g.ended or "it was stopped"
            self._log(
                g,
                "sequence",
                f"stopped part-way after {len(done)} of {len(plan)} steps ({reason}): {self._seq_detail(done)}",
                "stopped",
                None,
            )
            self._emit("control.interrupted", {"agent": g.agent_name, "action": "sequence", "reason": reason})
            raise ControlRefused(f"stopped part-way because {reason}") from None
        except ControlRefused as exc:
            # A refusal mid-batch (a denied "Are you sure?", a password field, a terminal, the rate limit):
            # the earlier sub-steps already ran, so the action log still gets one combined entry - no silent gap.
            self._log(
                g,
                "sequence",
                f"refused after {len(done)} of {len(plan)} steps ({exc}): {self._seq_detail(done)}",
                "refused",
                None,
            )
            self._emit("control.interrupted", {"agent": g.agent_name, "action": "sequence", "reason": str(exc)})
            raise
        detail = self._seq_detail(done)
        undos = [r["undo"] for r in done if r.get("undo")]
        undo: dict[str, Any] | None = {"sequence": undos} if undos else None
        entry = self._log(g, "sequence", detail, "done", undo)
        self._emit("control.action", {"agent": g.agent_name, "action": "sequence", "detail": detail, "id": entry.id})
        return {"ok": True, "action": "sequence", "detail": detail, "log_id": entry.id, "steps": done}

    @staticmethod
    def _seq_detail(done: list[dict[str, Any]]) -> str:
        """One line per completed sub-step, shared by the done / stopped / refused sequence log entries."""
        return "; ".join(f"{n}. {r['action']}: {r['detail']}" for n, r in enumerate(done, 1))

    def _window_title(self) -> str:
        try:
            fg = self.hooks.foreground() or {}
        except Exception:
            fg = {}
        return str(fg.get("title", ""))

    def _foreground_state(self) -> dict[str, str]:
        """Best-effort post-action foreground window (title + process name). Never raises: if the hook
        fails, the action already succeeded and the agent still gets its result, just without the
        observed window state."""
        try:
            fg = self.hooks.foreground() or {}
        except Exception:
            fg = {}
        return {"title": str(fg.get("title", "")), "process": str(fg.get("process", ""))}

    def _open_app(self, name: str, step: Callable[[], Any], wait: Callable[[float], None]) -> bool:
        """Through the OS launcher (Start menu search / Spotlight): no command line is ever built. Each step checks
        that the person has not stopped or taken over. After Enter, the foreground window is polled for up to
        ``OPEN_APP_VERIFY_S`` for a title/process match against ``name``; returns True when the app visibly
        opened. The fixed 0.6/0.8 s sleeps stay as the base timing; the poll is the verification."""
        b = self.hooks.backend
        if not (self.platform == "win32" or (self.platform == "darwin" and hasattr(b, "hotkey"))):
            raise ControlRefused("open_app works on Windows and macOS; on Linux, open it yourself or use open_url")
        self.note_own_keys()
        if self.platform == "win32":
            step().key("cmd")
        else:
            step().hotkey(["cmd", "space"])
        wait(0.6)
        self.note_own_keys(KEY_ECHO_S + KEY_ECHO_PER_CHAR_S * len(name))
        step().type_text(name)
        wait(0.8)
        self.note_own_keys()
        step().key("enter")
        return self._app_visible(name, wait)

    @staticmethod
    def _name_matches(want: str, title: str, process: str) -> bool:
        """The requested app name against a foreground window's title and process name (case-insensitive,
        punctuation-insensitive: "Notepad" matches "Untitled - Notepad" and "notepad.exe").

        Short names (under 3 alphanumeric characters) use stricter matching: they must equal the
        process stem or appear as a whole word in the title. This avoids "a" matching "Calculator"
        or "it" matching "Settings".
        """

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())

        want = norm(want)
        if not want:
            return False
        proc = norm(process[:-4]) if process.lower().endswith(".exe") else norm(process)
        title_n = norm(title)
        if len(want) < 3:
            # strict mode for short names: exact process stem, or whole-word in title
            if proc and want == proc:
                return True
            words = re.sub(r"[^a-z0-9]+", " ", title.lower()).split()
            return want in [re.sub(r"[^a-z0-9]", "", w) for w in words]
        return want in title_n or (bool(proc) and want in proc)

    def _app_visible(self, name: str, wait: Callable[[float], None]) -> bool:
        """Poll the foreground window until its title or process matches ``name`` (or the ~3 s budget runs out).
        ``wait`` still checks for a stop/takeover between polls, so the person can interrupt the verification."""
        deadline = self.clock() + OPEN_APP_VERIFY_S
        for _ in range(int(OPEN_APP_VERIFY_S / OPEN_APP_POLL_S) + 1):
            try:
                fg = self.hooks.foreground() or {}
            except Exception:
                fg = {}
            if self._name_matches(name, str(fg.get("title", "")), str(fg.get("process", ""))):
                return True
            if self.clock() >= deadline:
                return False
            wait(OPEN_APP_POLL_S)
        return False

    def _open_system_tool(self, target: str) -> None:
        """Launch an allowlisted ``SAFE_SYSTEM_TOOLS`` target through the shell: no synthetic keystrokes,
        no command line is ever built (``target`` is hardcoded, never from agent input). Windows only."""
        if self.platform != "win32":
            raise ControlRefused("system tools are only available on Windows")
        launcher = self.hooks.system_launcher
        if launcher is not None:
            launcher(target)
            return
        startfile = getattr(os, "startfile", None)
        if startfile is None:  # not Windows (platform check above already covers this)
            raise ControlRefused("system tools are only available on Windows")
        startfile(target)

    def _audit(self, g: Grant, action: str, detail: str, result: str) -> None:
        """The engine's audit trail: one entry per action (or per sub-step of a sequence), never batched."""
        self.engine.audit(AGENT, g.agent_name, action, detail, result)

    def _log(self, g: Grant, action: str, detail: str, result: str, undo: dict[str, Any] | None) -> LogEntry:
        e = LogEntry(
            uuid.uuid4().hex[:10],
            time.time(),
            g.agent_name,
            action,
            detail,
            result,
            "",
            undo,
            g.agent_id,
            session=g.id,
        )
        self._audit(g, action, detail, result)
        # v0.1.28: the visual audit trail (on by default): a screenshot thumbnail per step, persisted with the
        # session's trail. Off: the step is still in the in-memory action log (undo, the safety log), but no
        # picture is kept and nothing is written to the session file.
        if self.log_folder is not None and self._trail_on():
            try:
                img = self.hooks.grab()
                if img is not None:
                    self.log_folder.mkdir(parents=True, exist_ok=True)
                    (self.log_folder / f"{e.id}.png").write_bytes(desktop.thumbnail_png(img))
                    e.thumb = f"{e.id}.png"
                    keep = sorted(self.log_folder.glob("*.png"), key=lambda p: p.stat().st_mtime)
                    for old in keep[:-200]:
                        old.unlink(missing_ok=True)
            except Exception:
                e.thumb = ""
        with self._lock:
            self.log.append(e)
            if self._trail_on():  # off: no per-session trail at all (the safety action log above still runs)
                trail = self.sessions.setdefault(g.id, [])
                if len(trail) >= TRAIL_CAP:
                    trail.pop(0)
                trail.append(e)
        if self._trail_on():
            self._trail_event(
                g,
                {
                    "event": "step",
                    "id": e.id,
                    "agent": e.agent,
                    "agent_id": e.agent_id,
                    "action": e.action,
                    "detail": e.detail,
                    "result": e.result,
                    "thumb": e.thumb,
                },
            )
        return e

    def actions(
        self, limit: int = 50, agent_id: str | None = None, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """The newest actions; only ``agent_id``'s own when given (an agent's view of the log), and only one
        session's when ``session_id`` is given (the visual audit trail)."""
        with self._lock:
            items = [
                e
                for e in self.log
                if (agent_id is None or e.agent_id == agent_id) and (session_id is None or e.session == session_id)
            ][-limit:]
        return [self._entry_dict(e) for e in reversed(items)]

    @staticmethod
    def _entry_dict(e: LogEntry) -> dict[str, Any]:
        return {
            "id": e.id,
            "ts": e.ts,
            "agent": e.agent,
            "action": e.action,
            "detail": e.detail,
            "result": e.result,
            "thumb": e.thumb,
            "undoable": e.undo is not None,
            "source": e.source,
            "session": e.session,
        }

    def session_trail(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """v0.1.28: one control session's audit trail, oldest first (the live session when ``session_id`` is not
        given): timestamp, action, detail, result and screenshot thumbnail per step - the replay view."""
        with self._lock:
            sid = session_id or (self.grant.id if self.grant is not None else None)
            if sid is None and self._session_order:
                sid = self._session_order[-1]
            entries = list(self.sessions.get(sid, []) if sid else [])[-limit:]
        return [self._entry_dict(e) for e in entries]

    def sessions_summary(self) -> list[dict[str, Any]]:
        """v0.1.28: the recent control sessions (newest last): who, how many steps, the budget, still active."""
        with self._lock:
            active = self.grant.id if self.grant is not None else None
            out = []
            for sid in self._session_order:
                steps = len(self.sessions.get(sid, []))
                out.append(
                    {
                        "session": sid,
                        "agent": self._session_agents.get(sid, ""),
                        "steps": steps,
                        "active": sid == active,
                    }
                )
        return out

    def undo_last(self, agent_id: str) -> dict[str, Any]:
        """Undo the agent's newest undoable action. Undo is state-changing and spends one budget step."""
        g = self._require_grant(agent_id)
        self._policy_check(agent_id)
        return self._undo(agent_id, budget_grant=g)

    def undo_last_by_user(self) -> dict[str, Any]:
        """The person pressed Undo in Agent Link (no grant needed: it is their own input)."""
        return self._undo(None)

    def undo_all_by_user(self, item_ids: list[str]) -> dict[str, Any]:
        """The person pressed "Undo all" on a consent receipt (proposal #17): put back every undoable action
        listed on that receipt (cursor moves, scrolls, typed text). Only the receipt's own items are touched:
        never another agent's actions, never an unrelated prior session. No grant needed: it is their own input.
        Clicks and key presses can't be put back."""
        ids = set(item_ids)
        undone: list[str] = []
        failed: list[str] = []
        while True:
            try:
                r = self._undo(None, only_ids=ids)
            except ControlRefused as exc:
                # "nothing that can be undone" means we are done (all undoable items
                # handled). Any other refusal is a real failure; report it.
                if "nothing that can be undone" not in str(exc):
                    failed.append(str(exc))
                break
            undone.append(describe_action(r["undone"], r["detail"]))
        out: dict[str, Any] = {"undone": undone, "count": len(undone)}
        if failed:
            out["failed"] = failed
            out["partial"] = True
        return out

    def _undo(
        self, agent_id: str | None, only_ids: set[str] | None = None, budget_grant: Grant | None = None
    ) -> dict[str, Any]:
        with self._lock:
            e = next(
                (
                    x
                    for x in reversed(self.log)
                    if x.undo is not None
                    and x.result == "done"
                    and (agent_id is None or x.agent_id == agent_id)
                    and (only_ids is None or x.id in only_ids)
                ),
                None,
            )
            if e is None:
                raise ControlRefused("nothing that can be undone (clicks and key presses can't be)")
            # Select the entry, mark it in-flight, and reserve budget atomically. Concurrent undo calls
            # cannot select the same action or race past the session's limit.
            if budget_grant is not None:
                if budget_grant.steps_used >= budget_grant.step_budget:
                    raise ControlRefused(BUDGET_EXHAUSTED.format(budget=budget_grant.step_budget))
                budget_grant.steps_used += 1
            e.result = "undoing"
        b = self.hooks.backend
        u = e.undo or {}
        sub = list(reversed(u["sequence"])) if "sequence" in u else [u]
        source = USER if agent_id is None else AGENT
        try:
            if any("keys" in s for s in sub):
                self.note_own_keys()
            else:
                self.note_own_mouse()
            with self.engine.begin(source, "undo", actor=agent_id or "", backend=b) as a:
                for s in sub:
                    self._apply_undo_step(a, b, s)
                a.ok = True
        except ActionCancelled as exc:
            with self._lock:
                e.result = "done"
            raise ControlRefused(f"stopped part-way because {exc.reason}") from None
        except Exception:
            with self._lock:
                e.result = "done"
            raise
        with self._lock:
            e.result = "undone"
        if budget_grant is not None:
            # The undo is a distinct agent action in the session's visual audit trail as well as the budget.
            self._log(budget_grant, "undo", f"Undid {e.action}", "done", None)
        self._emit("control.undo", {"id": e.id, "action": e.action})
        return {"undone": e.action, "detail": e.detail}

    @staticmethod
    def _apply_undo_step(a: Action, b: Any, u: dict[str, Any]) -> None:
        """Undo one undoable step (a single action's undo record, or one element of a sequence's)."""
        if "move" in u:
            a.step().move(int(u["move"][0]), int(u["move"][1]))
        elif "scroll" in u:
            dy, dx = u["scroll"]
            if dy:
                a.step().scroll(dy)
            if dx and hasattr(b, "hscroll"):
                a.step().hscroll(dx)
        elif "keys" in u and hasattr(b, "hotkey"):
            a.step().hotkey(list(u["keys"]))
