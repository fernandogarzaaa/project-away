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
   interrupts the action in progress; resuming needs a new approval. Global stop: Ctrl+Alt+Esc, saying "stop", or
   holding an open palm ~1 s (never delayed by the short grace right after approval).
4. **No secrets, no shell.** Typing is refused into password fields (UI Automation ``IsPassword``; on Windows
   typing is refused altogether when UI Automation is unavailable) and into terminals; nothing works while the
   secure desktop (UAC / Ctrl+Alt+Del) is up; open_app refuses terminals and system tools; Win+R / Win+X are
   blocked.
5. **Risky clicks ask first.** Clicking an element named like Send / Submit / Pay / Buy / Purchase / Delete /
   Confirm / Install / Uninstall..., or pressing Enter while one is focused, asks the user first (with "Always
   allow for this agent").
6. **Rate limited and logged.** At most ``action_rate_per_s`` actions per second; every action is logged with a
   thumbnail of the screen; "Undo last" reverses what can be reversed (cursor moves, scrolls, typed text).
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from away.agentlink import desktop
from away.agentlink.ratelimit import TokenBucket

RISKY = re.compile(
    r"\b(send|submit|pay|buy|purchase|delete|remove|confirm|install|uninstall|checkout|check out|place order|"
    r"transfer|erase|format|reset|sign out|log out|publish|post)\b",
    re.IGNORECASE,
)
BLOCKED_COMBOS = ({"cmd", "r"}, {"cmd", "x"}, {"ctrl", "alt", "delete"}, {"ctrl", "shift", "esc"})
BLOCKED_APPS = desktop.TERMINAL_PROCESSES
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


class ControlRefused(Exception):
    """An action that was not run; ``str(exc)`` is a plain-language reason the agent can relay."""


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


class _Interrupted(Exception):
    """The grant was paused or ended while an action was running."""


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


@dataclass
class ControlHooks:
    """Platform access, injectable for tests."""

    backend: Any  # mouse/keyboard backend (RecordingBackend in tests; GuardedBackend(PynputBackend) in the app)
    uia: Any = field(default_factory=desktop.NullUIA)
    foreground: Callable[[], dict[str, Any] | None] = desktop.foreground_window
    secure_desktop: Callable[[], bool] = desktop.secure_desktop_active
    grab: Callable[[], Any] = desktop.grab_screen
    open_url: Callable[[str], Any] | None = None
    cursor_pos: Callable[[], tuple[int, int] | None] = lambda: None
    platform: str = ""
    overlay_on: Callable[[str], None] = lambda _name: None
    overlay_off: Callable[[], None] = lambda: None
    sleep: Callable[[float], None] = time.sleep


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
    ) -> None:
        import sys

        self.hooks = hooks
        self.ask = ask
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self.on_event = on_event
        self.always_allowed = always_allowed
        self.allow_always = allow_always
        self.bucket = TokenBucket(rate_per_s, max(2, int(rate_per_s * 2)), clock)
        self.log_folder = log_folder
        self.log: deque[LogEntry] = deque(maxlen=300)
        self.grant: Grant | None = None
        self._lock = threading.RLock()
        self._own_mouse_until = 0.0  # the agent's own mouse input may echo back until then
        self._own_keys_until = 0.0  # the agent's own key presses may echo back until then
        self._grace_until = 0.0  # just after the user approved (maybe with a thumbs up): that is not a takeover
        self.platform = hooks.platform or sys.platform
        self.scales: dict[str, float] = {}  # agent -> last screenshot scale (for from_image coordinates)
        # v0.1.21: agent -> (origin x, origin y, scale) of its last screenshot (a monitor or a zoomed region)
        self.views: dict[str, tuple[float, float, float]] = {}

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
            "It will ask before sending, paying, deleting or installing anything.",
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
            self.grant = Grant(agent_id, agent_name, now, now + ttl)
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
        res = self.ask(
            f"Let {g.agent_name} continue using your computer?",
            ["Continue", "Stop"],
            title="Resume?",
            agent=g.agent_name,
            details=[f"It was paused because {g.paused}."],
            timeout_s=60.0,
            urgency="high",
            kind="approve",
        )
        if res.get("answer") != "Continue":
            self._end("the user chose to stop")
            raise ControlRefused("the user did not let it continue")
        with self._lock:
            if self.grant is not None:
                self.grant.paused = ""
                self.grant.cancel = threading.Event()  # the old takeover does not cancel new actions
                self._grace_until = self.clock() + GRACE_S
        with contextlib.suppress(Exception):
            self.hooks.overlay_on(g.agent_name)
        self._emit("control.resumed", {"agent": g.agent_name})
        return self.status()

    def _end(self, reason: str) -> None:
        with self._lock:
            g, self.grant = self.grant, None
            if g is not None:
                g.ended = reason
                g.cancel.set()  # an action in progress stops before its next step
        if g is not None:
            with contextlib.suppress(Exception):
                self.hooks.overlay_off()
            self._emit("control.ended", {"agent": g.agent_name, "reason": reason})

    def stop(self, reason: str = "stopped") -> bool:
        """Global stop (hotkey, "stop", open palm, Disconnect, revoke): ends the grant at once."""
        had = self.grant is not None
        self._end(reason)
        return had

    def revoke_agent(self, agent_id: str) -> None:
        if self.grant is not None and self.grant.agent_id == agent_id:
            self._end("the agent was disconnected")

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
        with contextlib.suppress(Exception):
            self.hooks.overlay_off()
        self._emit("control.paused", {"agent": g.agent_name, "reason": g.paused})
        return True

    # ---------------------------------------------------------------- acting
    def _require(self, agent_id: str) -> Grant:
        g = self._check_expiry()
        if g is None:
            raise ControlRefused("not allowed to use the computer right now; call request_control first")
        if g.agent_id != agent_id:
            raise ControlRefused(f"{g.agent_name} is using the computer, not this agent")
        if g.paused:
            raise ControlRefused(f"paused because {g.paused}; call resume_control to ask the user to continue")
        try:
            if self.hooks.secure_desktop():
                raise ControlRefused("a Windows security prompt is on screen; agents can never touch it")
        except ControlRefused:
            raise
        except Exception:
            pass
        if not self.bucket.allow(agent_id):
            raise ControlRefused("too many actions; slow down (max a few per second)")
        return g

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

    def _point(self, agent_id: str, x: Any, y: Any, element: str | None, from_image: bool) -> tuple[int, int, str]:
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
        tok = g.cancel
        st = {"sent": False, "held": False, "ok": False}

        def step() -> Any:
            """The backend, after checking that the grant was not paused or ended (call before every input)."""
            if tok.is_set():
                raise _Interrupted
            st["sent"] = True
            return b

        def wait(seconds: float) -> None:
            if self.hooks.sleep is time.sleep:
                tok.wait(seconds)  # wakes at once on a stop
            else:
                self.hooks.sleep(seconds)
            if tok.is_set():
                raise _Interrupted

        try:
            return self._act(g, agent_id, action, kw, b, step, wait, st)
        except _Interrupted:
            reason = g.paused or g.ended or "it was stopped"
            self._log(g, action, "stopped part-way", "stopped", None)
            self._emit("control.interrupted", {"agent": g.agent_name, "action": action, "reason": reason})
            raise ControlRefused(f"stopped part-way because {reason}") from None
        finally:
            if not st["ok"] and st["sent"]:  # never leave a mouse button or a modifier key held down
                if st["held"]:
                    with contextlib.suppress(Exception):
                        b.release()
                rm = getattr(b, "release_modifiers", None)
                if callable(rm):
                    with contextlib.suppress(Exception):
                        rm()

    def _act(
        self,
        g: Grant,
        agent_id: str,
        action: str,
        kw: dict[str, Any],
        b: Any,
        step: Callable[[], Any],
        wait: Callable[[float], None],
        st: dict[str, bool],
    ) -> dict[str, Any]:
        undo: dict[str, Any] | None = None
        before = None
        try:
            before = self.hooks.cursor_pos()
        except Exception:
            before = None
        detail = ""
        if action in ("move", "click", "double_click", "right_click"):
            x, y, label = self._point(agent_id, kw.get("x"), kw.get("y"), kw.get("element"), bool(kw.get("from_image")))
            if action != "move":
                self._confirm_risky(g, label, f"click '{label}'")
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
            if action == "move" and before is not None:
                undo = {"move": list(before)}
        elif action == "drag":
            x1, y1, l1 = self._point(agent_id, kw.get("x"), kw.get("y"), kw.get("element"), bool(kw.get("from_image")))
            x2, y2, _l2 = self._point(
                agent_id, kw.get("x2"), kw.get("y2"), kw.get("to_element"), bool(kw.get("from_image"))
            )
            self._confirm_risky(g, l1, f"drag '{l1}'")
            self.note_own_mouse()
            step().move(x1, y1)
            step().press()
            st["held"] = True
            steps = 8
            for i in range(1, steps + 1):
                self.note_own_mouse()
                step().move(int(x1 + (x2 - x1) * i / steps), int(y1 + (y2 - y1) * i / steps))
                wait(0.02)
            b.release()
            st["held"] = False
            detail = f"({x1}, {y1}) -> ({x2}, {y2})"
        elif action == "scroll":
            dy = max(-50.0, min(50.0, float(kw.get("dy") or 0.0)))
            dx = max(-50.0, min(50.0, float(kw.get("dx") or 0.0)))
            if kw.get("x") is not None or kw.get("element"):
                x, y, _l = self._point(
                    agent_id, kw.get("x"), kw.get("y"), kw.get("element"), bool(kw.get("from_image"))
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
            self._typing_target_ok()
            if "\n" in text:
                self._confirm_risky(g, self._focused_label() or self._window_title(), "press Enter")
            if not hasattr(b, "type_text"):
                raise ControlRefused("typing is not supported on this system")
            with self._lock:
                self._own_mouse_until = 0.0  # while it types, any real mouse movement is the person
            for i in range(0, len(text), TYPE_CHUNK):
                chunk = text[i : i + TYPE_CHUNK]
                self.note_own_keys(KEY_ECHO_S + KEY_ECHO_PER_CHAR_S * len(chunk))
                step().type_text(chunk)
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
            if any(k in ENTER_KEYS for k in keys):
                self._confirm_risky(g, self._focused_label() or self._window_title(), "press Enter")
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
            if not APP_NAME.match(name):
                raise ControlRefused("give a plain app name, like 'Notepad' or 'Spotify'")
            if BLOCKED_APPS.match(name.replace(" ", "")) or name.lower() in (
                "command prompt",
                "terminal",
                "powershell",
                "registry editor",
            ):
                raise ControlRefused("agents cannot open terminals or system tools")
            self._open_app(name, step, wait)
            detail = name
        st["ok"] = True
        entry = self._log(g, action, detail, "done", undo)
        self._emit("control.action", {"agent": g.agent_name, "action": action, "detail": detail, "id": entry.id})
        return {"ok": True, "action": action, "detail": detail, "log_id": entry.id}

    def _window_title(self) -> str:
        try:
            fg = self.hooks.foreground() or {}
        except Exception:
            fg = {}
        return str(fg.get("title", ""))

    def _open_app(self, name: str, step: Callable[[], Any], wait: Callable[[float], None]) -> None:
        """Through the OS launcher (Start menu search / Spotlight): no command line is ever built. Each step checks
        that the person has not stopped or taken over."""
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

    def _log(self, g: Grant, action: str, detail: str, result: str, undo: dict[str, Any] | None) -> LogEntry:
        e = LogEntry(uuid.uuid4().hex[:10], time.time(), g.agent_name, action, detail, result, "", undo, g.agent_id)
        if self.log_folder is not None:
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
        return e

    def actions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self.log)[-limit:]
        return [
            {
                "id": e.id,
                "ts": e.ts,
                "agent": e.agent,
                "action": e.action,
                "detail": e.detail,
                "result": e.result,
                "thumb": e.thumb,
                "undoable": e.undo is not None,
            }
            for e in reversed(items)
        ]

    def undo_last(self, agent_id: str) -> dict[str, Any]:
        """An agent undoes its own newest undoable action (cursor move, scroll, typed text). It needs the same live,
        unexpired, unpaused grant as :meth:`act`: undo is input too. Clicks and keys cannot be undone."""
        self._require(agent_id)
        return self._undo(agent_id)

    def undo_last_by_user(self) -> dict[str, Any]:
        """The person pressed Undo in Agent Link (no grant needed: it is their own input)."""
        return self._undo(None)

    def _undo(self, agent_id: str | None) -> dict[str, Any]:
        with self._lock:
            e = next(
                (
                    x
                    for x in reversed(self.log)
                    if x.undo is not None and x.result == "done" and (agent_id is None or x.agent_id == agent_id)
                ),
                None,
            )
        if e is None:
            raise ControlRefused("nothing that can be undone (clicks and key presses can't be)")
        b = self.hooks.backend
        u = e.undo or {}
        if "keys" in u:
            self.note_own_keys()
        else:
            self.note_own_mouse()
        if "move" in u:
            b.move(int(u["move"][0]), int(u["move"][1]))
        elif "scroll" in u:
            dy, dx = u["scroll"]
            if dy:
                b.scroll(dy)
            if dx and hasattr(b, "hscroll"):
                b.hscroll(dx)
        elif "keys" in u and hasattr(b, "hotkey"):
            b.hotkey(list(u["keys"]))
        e.result = "undone"
        self._emit("control.undo", {"id": e.id, "action": e.action})
        return {"undone": e.action, "detail": e.detail}
