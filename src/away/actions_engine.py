"""One action engine for every computer-side effect (audit Phase 1b).

Every synthetic mouse move, click, key press and typed character that Project Away sends goes through
:class:`ActionEngine`. The engine is the **only** module allowed to talk to the OS input backend
(``tests/test_actions_engine.py`` enforces it). Gestures, voice commands, dictation, shortcut (macro) playback,
Agent Link computer control and, later, the live agent are clients of it, each tagged with a *source*.

What it owns:

* **The input backend.** :class:`OSInputBackend` (pynput) is built only here (:meth:`ActionEngine.os_backend`);
  :meth:`ActionEngine.cursor_position` reads the pointer.
* **Safety checks, per source.** :data:`POLICIES` says which checks apply. The user's own sources (gestures,
  voice, dictation, macros, the user pressing Undo) are never gated: no grants, no risk prompts, no rate limit.
  Agent sources keep every rule of the v0.1.24 hardening: the secure desktop check fails closed on Windows,
  a per-agent rate limit, and the risk classification (:func:`classify`) that the agent's control layer
  (:mod:`away.agentlink.control`) turns into "Are you sure?" prompts.
* **Cancellation.** Each in-flight action has a generation token (a :class:`threading.Event`), checked before
  every sub-step (:meth:`Action.step`) and during waits (:meth:`Action.wait`). Buttons and keys an action holds
  are released in ``finally``. :meth:`ActionEngine.stop_all` ("Stop everything": hotkey, open palm, voice, tray)
  cancels every source's in-flight action at once and lets go of everything still held.
* **One state machine**: ``running`` -> ``paused`` -> ``running`` and, at exit, ``shutdown`` (terminal).
* **The audit log**: every discrete action (not each pointer move) with its source, actor and result. Typed
  text is never logged, only its length.
* **Cleanup**: :meth:`ActionEngine.shutdown` cancels everything, releases what is held and refuses new input.

When a user source acts while an agent action is in flight, the user wins: the takeover listeners (the agent
control layer's "you used the ..." pause) run and the agent's token is cancelled. The user is never blocked.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("away.actions")

# ------------------------------------------------------------------------------------------------ sources
GESTURE = "gesture"
VOICE = "voice"
DICTATION = "dictation"
MACRO = "macro"
USER = "user"  # the person pressed a button in Project Away (Undo in Agent Link)
AGENT = "agent"
LIVE_AGENT = "live_agent"  # future: the live (streaming) agent
SOURCES = (GESTURE, VOICE, DICTATION, MACRO, USER, AGENT, LIVE_AGENT)
USER_SOURCES = frozenset({GESTURE, VOICE, DICTATION, MACRO, USER})
AGENT_SOURCES = frozenset({AGENT, LIVE_AGENT})

# --------------------------------------------------------------------------------- risk classification
RISKY = re.compile(
    r"\b(send|submit|pay|buy|purchase|delete|remove|confirm|install|uninstall|checkout|check out|place order|"
    r"transfer|erase|format|reset|sign out|log out|publish|post|trash|recycle bin)\b",
    re.IGNORECASE,
)
# keys that act on whatever has the focus (Enter submits, Delete deletes a selected file)
CONSEQUENTIAL_KEYS = ("enter", "return", "delete", "del")
SHORTCUT_MODIFIERS = ("ctrl", "alt", "cmd", "win", "super", "windows", "option")
MODIFIERS = ("shift", "shift_r", "ctrl", "ctrl_r", "alt", "alt_r", "alt_gr", "cmd", "cmd_r")

SECURE_DESKTOP_UNKNOWN = (
    "Project Away could not check whether a Windows security prompt is on screen, so it stopped to be safe; try "
    "again in a moment"
)
SECURE_DESKTOP_UP = "a Windows security prompt is on screen; agents can never touch it"
RATE_LIMITED = "too many actions; slow down (max a few per second)"


def classify(label: str | None) -> str:
    """What an action's target is, from its UI Automation name: ``risky`` (Send, Pay, Delete, Recycle Bin...),
    ``safe`` (any other name) or ``unknown`` (no name: no UI Automation, nothing found there, coordinates only)."""
    text = str(label or "").strip()
    if not text:
        return "unknown"
    return "risky" if RISKY.search(text) else "safe"


@dataclass(frozen=True)
class Policy:
    """Which checks the engine applies to a source before it acts."""

    secure_desktop: bool = False  # refuse while UAC / Ctrl+Alt+Del is up (fail closed on Windows)
    rate_limited: bool = False  # per-actor token bucket
    risk_prompts: bool = False  # risky / unknown targets ask the user first (done by the agent control layer)
    takeover: bool = False  # paused/cancelled when the user acts at the same time


USER_POLICY = Policy()
AGENT_POLICY = Policy(secure_desktop=True, rate_limited=True, risk_prompts=True, takeover=True)
POLICIES: dict[str, Policy] = {s: USER_POLICY for s in USER_SOURCES} | {s: AGENT_POLICY for s in AGENT_SOURCES}


def policy_for(source: str) -> Policy:
    """Unknown sources get the strict agent policy (fail closed)."""
    return POLICIES.get(source, AGENT_POLICY)


# ----------------------------------------------------------------------------------------------- errors
class ActionRefused(Exception):
    """An action that was not run; ``str(exc)`` is a plain-language reason."""


class ActionCancelled(Exception):
    """The action's token was cancelled (stop, takeover, pause, shutdown) while it ran."""

    def __init__(self, reason: str = "it was stopped") -> None:
        super().__init__(reason)
        self.reason = reason


# ------------------------------------------------------------------------------------- source attribution
_attr = threading.local()


@contextlib.contextmanager
def attributed(source: str) -> Iterator[None]:
    """Attribute every engine call made on this thread inside the block to ``source`` (a voice command runs the
    same bindings as a gesture, through the same client)."""
    prev = getattr(_attr, "source", None)
    _attr.source = source
    try:
        yield
    finally:
        _attr.source = prev


def current_attribution() -> str | None:
    return getattr(_attr, "source", None)


# ------------------------------------------------ Windows system-key taps (proposal #4)
# A lone Win keypress must be HELD, not tapped. pynput's Controller.press()/release() sends two back-to-back
# SendInput events with no dwell in between; the Windows shell discards the zero-dwell tap (down and up land in
# the same input batch with identical timestamps), so `press_keys keys="win"` returned "ok" while Start never
# opened. A human-like hold (80-120 ms) through native SendInput makes it through. Windows only: every other
# platform keeps the existing pynput path.
_WIN_TAP_HOLD_S = 0.10  # 100 ms: inside the 80-120 ms band the shell answers to
_WIN_TAP_VKS = {"cmd": 0x5B, "cmd_l": 0x5B, "cmd_r": 0x5C}  # VK_LWIN / VK_RWIN


def _sendinput_key(vk: int, down: bool) -> None:
    """One keyboard event through the native Windows input queue.

    Raises :class:`OSError` when nothing was injected: a silent no-op here is exactly the class of bug this
    fixes (the old path always answered "ok").
    """
    if sys.platform != "win32":
        raise RuntimeError("_sendinput_key is Windows only")
    import ctypes
    from ctypes import wintypes

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_void_p),  # ULONG_PTR: pointer-sized, like the real field
        ]

    class _Ki(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _Ki)]

    INPUT_KEYBOARD, KEYEVENTF_KEYUP, MAPVK_VK_TO_VSC = 1, 0x0002, 0
    user32 = ctypes.windll.user32
    ki = KEYBDINPUT(
        wVk=vk,
        wScan=user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC),
        dwFlags=0 if down else KEYEVENTF_KEYUP,
        time=0,
        dwExtraInfo=0,
    )
    sent = user32.SendInput(1, ctypes.byref(INPUT(INPUT_KEYBOARD, _Ki(ki=ki))), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput injected {sent}/1 keyboard events (vk=0x{vk:02X})")


def _win_tap(key: str, hold_s: float = _WIN_TAP_HOLD_S) -> None:
    """Press and release one Windows system key with a real hold; the key is always released."""
    vk = _WIN_TAP_VKS[key]
    _sendinput_key(vk, True)
    try:
        time.sleep(hold_s)
    finally:
        _sendinput_key(vk, False)


# --------------------------------------------------------------------------------------------- the backend
class OSInputBackend:  # pragma: no cover - needs a desktop session
    """The real mouse and keyboard (pynput). Built only by :meth:`ActionEngine.os_backend`."""

    def __init__(self) -> None:
        from pynput import keyboard, mouse

        self._m = mouse.Controller()
        self._k = keyboard.Controller()
        self._buttons = mouse.Button
        self._button = mouse.Button.left
        self._keys = keyboard.Key
        self._scroll_acc = 0.0
        self._hscroll_acc = 0.0

    def _key(self, k: str) -> Any:
        return getattr(self._keys, k, k)

    def _btn(self, b: str) -> Any:
        return getattr(self._buttons, b, self._buttons.left)

    @property
    def position(self) -> tuple[int, int]:
        x, y = self._m.position
        return int(x), int(y)

    def move(self, x: int, y: int) -> None:
        self._m.position = (x, y)

    def click(self, count: int = 1) -> None:
        self._m.click(self._button, count)

    def press(self, button: str = "left") -> None:
        self._m.press(self._btn(button))

    def release(self, button: str = "left") -> None:
        self._m.release(self._btn(button))

    def scroll(self, dy: float) -> None:
        self._scroll_acc += dy
        steps = int(self._scroll_acc)
        if steps:
            self._m.scroll(0, steps)
            self._scroll_acc -= steps

    def hscroll(self, dx: float) -> None:
        """Horizontal wheel notches (+ = right); fractions carry over like :meth:`scroll`."""
        self._hscroll_acc += dx
        steps = int(self._hscroll_acc)
        if steps:
            self._m.scroll(steps, 0)
            self._hscroll_acc -= steps

    def scroll_xy(self, dx: int, dy: int) -> None:
        """Raw wheel notches (shortcut replay)."""
        self._m.scroll(dx, dy)

    def key(self, name: str) -> None:
        # the same guaranteed cleanup path as multi-key chords
        self.hotkey([name])

    def click_button(self, button: str, count: int = 1) -> None:
        self._m.click(self._btn(button), count)

    def key_down(self, k: str) -> None:
        self._k.press(self._key(k))

    def key_up(self, k: str) -> None:
        self._k.release(self._key(k))

    def hotkey(self, keys: list[str]) -> None:
        """Press a chord and always attempt to release every key touched (a failure part-way must never leave a
        modifier logically held). An original press error wins over a cleanup error. A lone Windows system key
        (the Win key) goes through the native held tap instead: pynput's instant down/up is swallowed by the
        shell, so Start would never open."""
        if len(keys) == 1 and sys.platform == "win32" and str(keys[0]).lower() in _WIN_TAP_VKS:
            _win_tap(str(keys[0]).lower())
            return
        objs = [self._key(k) for k in keys]
        attempted: list[Any] = []
        try:
            for key in objs:
                attempted.append(key)  # before press(): the OS may have accepted the key-down before raising
                self._k.press(key)
        finally:
            active_error = sys.exc_info()[0] is not None
            release_error: BaseException | None = None
            for key in reversed(attempted):
                try:
                    self._k.release(key)
                except BaseException as exc:
                    if release_error is None:
                        release_error = exc
            if release_error is not None and not active_error:
                raise release_error

    def type_text(self, text: str) -> None:
        self._k.type(text)

    def release_modifiers(self) -> None:
        """Let go of Shift / Ctrl / Alt / Win-Cmd (both sides). Releasing a key that is not down does nothing."""
        for name in MODIFIERS:
            k = getattr(self._keys, name, None)
            if k is not None:
                with contextlib.suppress(Exception):
                    self._k.release(k)


# ------------------------------------------------------------------------------------------- audit + actions
@dataclass
class AuditEntry:
    ts: float
    source: str
    actor: str
    action: str
    detail: str
    result: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "source": self.source,
            "actor": self.actor,
            "action": self.action,
            "detail": self.detail,
            "result": self.result,
        }


@dataclass(eq=False)
class Action:
    """One in-flight action. Use :meth:`step` before every input and :meth:`wait` for pauses."""

    engine: ActionEngine
    source: str
    actor: str
    name: str
    token: threading.Event
    backend: Any
    sleep: Callable[[float], None] | None = None
    sent: bool = False
    ok: bool = False
    held_buttons: list[str] = field(default_factory=list)
    held_keys: list[str] = field(default_factory=list)
    cancel_reason: str = ""

    def cancel(self, reason: str) -> None:
        if not self.cancel_reason:
            self.cancel_reason = reason
        self.token.set()

    @property
    def cancelled(self) -> bool:
        return self.token.is_set()

    def check(self) -> None:
        if self.token.is_set():
            raise ActionCancelled(self.cancel_reason or "it was stopped")
        st = self.engine.state
        if st != RUNNING and not (st == PAUSED and self.source in self.engine._exempt_from_pause):
            raise ActionCancelled(self.engine.state_reason or f"input is {st}")

    def step(self) -> Any:
        """The backend, after checking that this action was not cancelled (call before every input)."""
        self.check()
        self.sent = True
        return self.backend

    def wait(self, seconds: float) -> None:
        if self.sleep is None or self.sleep is time.sleep:
            self.token.wait(seconds)  # wakes at once on a stop
        else:
            self.sleep(seconds)
        self.check()

    # held input, released in ``finally`` when the action does not complete
    def press_button(self, button: str = "left") -> None:
        b = self.step()
        self.held_buttons.append(button)
        if button == "left":
            b.press()
        else:
            b.press(button)

    def release_button(self, button: str = "left") -> None:
        if button in self.held_buttons:
            self.held_buttons.remove(button)
        if button == "left":
            self.backend.release()
        else:
            self.backend.release(button)

    def key_down(self, k: str) -> None:
        b = self.step()
        self.held_keys.append(k)
        b.key_down(k)

    def key_up(self, k: str) -> None:
        if k in self.held_keys:
            self.held_keys.remove(k)
        self.backend.key_up(k)

    def release_all(self, modifiers: bool = True) -> None:
        """Never leave a mouse button or a key held down."""
        for btn in reversed(self.held_buttons):
            with contextlib.suppress(Exception):
                if btn == "left":
                    self.backend.release()
                else:
                    self.backend.release(btn)
        self.held_buttons.clear()
        for k in reversed(self.held_keys):
            with contextlib.suppress(Exception):
                self.backend.key_up(k)
        self.held_keys.clear()
        if modifiers:
            rm = getattr(self.backend, "release_modifiers", None)
            if callable(rm):
                with contextlib.suppress(Exception):
                    rm()


RUNNING = "running"
PAUSED = "paused"
SHUTDOWN = "shutdown"


class ActionEngine:
    def __init__(
        self,
        backend: Any = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        secure_desktop: Callable[[], bool] | None = None,
        platform: str = "",
        agent_rate_per_s: float = 4.0,
        audit_size: int = 1000,
    ) -> None:
        from away.agentlink.ratelimit import TokenBucket

        self.backend = backend
        self.clock = clock
        self._secure_desktop = secure_desktop
        self.platform = platform or sys.platform
        self.bucket = TokenBucket(agent_rate_per_s, max(2, int(agent_rate_per_s * 2)), clock)
        self.audit_log: deque[AuditEntry] = deque(maxlen=audit_size)
        self.state = RUNNING
        self.state_reason = ""
        self.generation = 0
        self._exempt_from_pause: frozenset[str] = frozenset()
        self._lock = threading.RLock()
        self._io = threading.RLock()  # one OS call at a time (a chord is never interleaved with another source)
        self._inflight: set[Action] = set()
        self._tokens: dict[threading.Event, tuple[str, str]] = {}  # tracked tokens (macro replays) -> source
        # client holds: (backend id, "source|kind", name) -> (backend, the press arguments)
        self._held: dict[tuple[int, str, str], tuple[Any, tuple[Any, ...]]] = {}
        self._stop_listeners: list[Callable[[str], Any]] = []
        self._takeover_listeners: list[Callable[[str], Any]] = []

    # ---------------------------------------------------------------- backend
    @staticmethod
    def os_backend() -> OSInputBackend:  # pragma: no cover - needs a desktop session
        """A new real mouse/keyboard backend. The only constructor of OS input in Project Away."""
        return OSInputBackend()

    def set_backend(self, backend: Any) -> None:
        self.backend = backend

    def cursor_position(self) -> tuple[int, int] | None:
        """Where the pointer is (None when it cannot be read)."""
        try:
            b = self.backend
            pos = getattr(b, "position", None) if b is not None else None
            if pos is None:
                from pynput.mouse import Controller  # pragma: no cover - desktop

                pos = Controller().position  # pragma: no cover
            return int(pos[0]), int(pos[1])
        except Exception:
            return None

    def client(self, source: str, backend: Any = None) -> SourceBackend:
        """A backend-shaped client for ``source``: every call goes through the engine."""
        return SourceBackend(self, source, backend)

    # ---------------------------------------------------------------- state
    def pause(self, reason: str = "paused", *, except_sources: frozenset[str] | set[str] = frozenset()) -> None:
        """Refuse new input (and cancel in-flight actions) from every source except ``except_sources``."""
        with self._lock:
            if self.state == SHUTDOWN:
                return
            self.state, self.state_reason = PAUSED, reason
            self._exempt_from_pause = frozenset(except_sources)
        self._cancel(lambda s: s not in self._exempt_from_pause, reason, release=True)
        self.audit("engine", "", "pause", reason, "done")

    def resume(self) -> None:
        with self._lock:
            if self.state != PAUSED:
                return
            self.state, self.state_reason = RUNNING, ""
            self._exempt_from_pause = frozenset()
        self.audit("engine", "", "resume", "", "done")

    def shutdown(self, reason: str = "Project Away is closing") -> None:
        """Terminal: cancel everything, let go of everything held, refuse all new input."""
        with self._lock:
            if self.state == SHUTDOWN:
                return
            self.state, self.state_reason = SHUTDOWN, reason
        self.stop_all(reason)

    def accepting(self, source: str) -> bool:
        return self.state == RUNNING or (self.state == PAUSED and source in self._exempt_from_pause)

    # ---------------------------------------------------------------- listeners
    def add_stop_listener(self, fn: Callable[[str], Any]) -> None:
        """Called with the reason on every :meth:`stop_all` (the agent control layer ends its grant)."""
        self._stop_listeners.append(fn)

    def remove_stop_listener(self, fn: Callable[[str], Any]) -> None:
        with contextlib.suppress(ValueError):
            self._stop_listeners.remove(fn)

    def add_takeover_listener(self, fn: Callable[[str], Any]) -> None:
        """Called with the user source when the user acts while an agent action is in flight."""
        self._takeover_listeners.append(fn)

    def remove_takeover_listener(self, fn: Callable[[str], Any]) -> None:
        with contextlib.suppress(ValueError):
            self._takeover_listeners.remove(fn)

    # ---------------------------------------------------------------- stop
    def _cancel(self, which: Callable[[str], bool], reason: str, release: bool) -> int:
        with self._lock:
            acts = [a for a in self._inflight if which(a.source)]
            toks = [t for t, (src, _a) in self._tokens.items() if which(src)]
            held = [(k, v) for k, v in self._held.items() if which(k[1].split("|", 1)[0])] if release else []
            for k, _v in held:
                self._held.pop(k, None)
        for a in acts:
            a.cancel(reason)
        for t in toks:
            t.set()
        for (_id, key, _name), (backend, args) in held:
            kind = key.split("|", 1)[1]
            with contextlib.suppress(Exception), self._io:
                # released exactly the way it was pressed (``press()`` or ``press("right")``)
                (backend.release if kind == "button" else backend.key_up)(*args)
        return len(acts) + len(toks)

    def stop_all(self, reason: str = "you asked to stop everything") -> int:
        """ "Stop everything": cancel every source's in-flight action at once, let go of every held button and key,
        and tell the stop listeners (an agent's grant ends). Returns how many in-flight actions were cancelled."""
        with self._lock:
            self.generation += 1
        n = self._cancel(lambda _s: True, reason, release=True)
        for fn in list(self._stop_listeners):
            with contextlib.suppress(Exception):
                fn(reason)
        self.audit("engine", "", "stop_all", reason, f"cancelled {n}")
        log.info("stop everything (%s): %d in-flight action(s) cancelled", reason, n)
        return n

    def busy(self) -> bool:
        """True while any action (or tracked replay) is in flight."""
        with self._lock:
            return bool(self._inflight or self._tokens)

    def inflight(self) -> list[tuple[str, str, str]]:
        with self._lock:
            return [(a.source, a.actor, a.name) for a in self._inflight]

    # ---------------------------------------------------------------- safety
    def check(
        self,
        source: str,
        actor: str = "",
        *,
        secure_desktop: Callable[[], bool] | None = None,
        platform: str | None = None,
    ) -> None:
        """The checks ``source``'s policy asks for, before it acts. Raises :class:`ActionRefused`."""
        if not self.accepting(source):
            raise ActionRefused(self.state_reason or f"input is {self.state}")
        pol = policy_for(source)
        if pol.secure_desktop:
            probe = secure_desktop or self._secure_desktop
            if probe is None:
                from away.agentlink import desktop

                probe = desktop.secure_desktop_active
            try:
                secure = bool(probe())
            except Exception:
                if (platform or self.platform) == "win32":  # fail closed when Windows cannot be checked
                    raise ActionRefused(SECURE_DESKTOP_UNKNOWN) from None
                secure = False  # macOS and Linux have no secure desktop to protect
            if secure:
                raise ActionRefused(SECURE_DESKTOP_UP)
        if pol.rate_limited and not self.bucket.allow(f"{source}:{actor}"):
            raise ActionRefused(RATE_LIMITED)

    def set_agent_rate(self, rate_per_s: float, clock: Callable[[], float] | None = None) -> None:
        from away.agentlink.ratelimit import TokenBucket

        self.bucket = TokenBucket(rate_per_s, max(2, int(rate_per_s * 2)), clock or self.clock)

    # ---------------------------------------------------------------- multi-step actions
    @contextlib.contextmanager
    def begin(
        self,
        source: str,
        name: str,
        *,
        actor: str = "",
        token: threading.Event | None = None,
        backend: Any = None,
        sleep: Callable[[float], None] | None = None,
        audit: bool = True,
    ) -> Iterator[Action]:
        """Run one (multi-step) action. Raises :class:`ActionCancelled` from :meth:`Action.step` /
        :meth:`Action.wait` when it is cancelled; held input is released on any exit that did not complete
        (set ``action.ok = True`` when done). Policy checks are separate (:meth:`check`)."""
        a = Action(self, source, actor, name, token or threading.Event(), backend or self.backend, sleep)
        if not self.accepting(source):
            a.cancel(self.state_reason or f"input is {self.state}")
        self._user_acts(source)
        with self._lock:
            self._inflight.add(a)
        result = "done"
        try:
            yield a
            if a.token.is_set() and not a.ok:
                result = "stopped"
        except ActionCancelled:
            result = "stopped"
            raise
        except BaseException:
            result = "failed"
            raise
        finally:
            with self._lock:
                self._inflight.discard(a)
            if not a.ok and a.sent:
                a.release_all(modifiers=True)
            elif a.held_buttons or a.held_keys:
                a.release_all(modifiers=False)
            if audit:
                self.audit(source, actor, name, "", result)

    @contextlib.contextmanager
    def track(self, source: str, token: threading.Event, name: str = "") -> Iterator[threading.Event]:
        """Register an externally owned cancel token (a shortcut replay's abort event) so :meth:`stop_all`
        reaches it."""
        if not self.accepting(source):
            token.set()
        self._user_acts(source)
        with self._lock:
            self._tokens[token] = (source, name)
        try:
            yield token
        finally:
            with self._lock:
                self._tokens.pop(token, None)

    def _user_acts(self, source: str) -> None:
        """The user wins: when a user source acts while an agent action is in flight, the agent is taken over."""
        if source not in USER_SOURCES:
            return
        with self._lock:
            agents = [a for a in self._inflight if policy_for(a.source).takeover]
        if not agents:
            return
        listeners = list(self._takeover_listeners)
        for fn in listeners:
            with contextlib.suppress(Exception):
                fn(source)  # the agent layer pauses its grant (with its own grace and echo rules) and cancels
        if not listeners:  # nobody owns the agent's session: cancel its in-flight action directly
            for a in agents:
                a.cancel(f"you used {source}")

    # ---------------------------------------------------------------- single calls (clients)
    def call(self, source: str, backend: Any, op: str, *args: Any) -> Any:
        """One backend call from a client. Dropped (returns None) while the engine does not accept ``source``;
        releases always go through so nothing stays held. Holds are tracked for :meth:`stop_all`."""
        releasing = op in ("release", "key_up", "release_modifiers")
        if not releasing and not self.accepting(source):
            return None
        self._user_acts(source)
        bid = id(backend)
        if op == "press":
            with self._lock:
                self._held[(bid, f"{source}|button", str(args[0]) if args else "left")] = (backend, args)
        elif op == "key_down":
            with self._lock:
                self._held[(bid, f"{source}|key", str(args[0]))] = (backend, args)
        elif op == "release":
            with self._lock:
                self._held.pop((bid, f"{source}|button", str(args[0]) if args else "left"), None)
        elif op == "key_up":
            with self._lock:
                self._held.pop((bid, f"{source}|key", str(args[0])), None)
        fn = getattr(backend, op)
        with self._io:
            out = fn(*args)
        if op in AUDITED_OPS:
            self.audit(source, "", op, _detail(op, args), "done")
        return out

    # ---------------------------------------------------------------- audit
    def audit(self, source: str, actor: str, action: str, detail: str = "", result: str = "done") -> AuditEntry:
        e = AuditEntry(time.time(), source, actor, action, detail[:200], result)
        with self._lock:
            self.audit_log.append(e)
        log.debug("%s %s %s %s -> %s", source, actor, action, detail[:80], result)
        return e

    def recent(self, limit: int = 50, source: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self.audit_log if source is None or e.source == source][-limit:]
        return [e.to_dict() for e in reversed(items)]


AUDITED_OPS = frozenset(
    {"click", "click_button", "press", "release", "key", "hotkey", "key_down", "key_up", "type_text", "scroll_xy"}
)


def _detail(op: str, args: tuple[Any, ...]) -> str:
    if op == "type_text":
        return f"{len(str(args[0])) if args else 0} characters"  # never the text itself
    if op == "hotkey" and args:
        return "+".join(str(k) for k in args[0])
    return " ".join(str(a) for a in args)


class SourceBackend:
    """A backend-shaped client: the same methods as the mouse/keyboard backends, routed through the engine and
    attributed to ``source`` (or to the thread's :func:`attributed` source)."""

    _OPS = (
        "move",
        "click",
        "press",
        "release",
        "scroll",
        "hscroll",
        "scroll_xy",
        "key",
        "click_button",
        "hotkey",
        "key_down",
        "key_up",
        "type_text",
        "release_modifiers",
    )

    def __init__(self, engine: ActionEngine, source: str, backend: Any = None) -> None:
        self._engine = engine
        self._source = source
        self._own = backend

    @property
    def source(self) -> str:
        return current_attribution() or self._source

    @property
    def target(self) -> Any:
        """The backend the calls reach (the engine's, unless this client was given its own)."""
        return self._own if self._own is not None else self._engine.backend

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        target = self.target
        attr = getattr(target, name)
        if name not in self._OPS or not callable(attr):
            return attr

        def routed(*a: Any) -> Any:
            return self._engine.call(self.source, target, name, *a)

        return routed


# ------------------------------------------------------------------------------------------ the shared engine
_ENGINE: ActionEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> ActionEngine:
    """The process-wide engine (the app, the tray, Agent Link and Settings share it)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = ActionEngine()
            atexit.register(_ENGINE.shutdown)
        return _ENGINE


def stop_everything(reason: str = "you asked to stop everything") -> int:
    """The global "Stop everything" (hotkey, open palm, voice, tray)."""
    return get_engine().stop_all(reason)
