"""The smart pointer: the one object :class:`~away.app.runner.Pipeline` talks to.

* :meth:`SmartPointer.refine` runs once per cursor frame (from :class:`~away.control.mouse.MouseController`): the
  magnet, reading the latest cached targets. Never blocks, never calls UI Automation.
* :meth:`SmartPointer.voice` handles "click <name>" / "click that" / "double click <name>" / "focus <name>" and
  the numbered choice that follows an ambiguous name. Bound voice phrases always win.
* :meth:`SmartPointer.phrases` wraps the bound phrase table so always-on listening recognises the commands too.

Off Windows (or without comtypes) the backend is a :class:`~away.pointer.targets.NullBackend`: ``refine`` returns
its input unchanged and ``voice`` returns None, so nothing changes for anyone else.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

from away.bindings import Action
from away.pointer.badges import BadgeOverlay, badge_specs
from away.pointer.looksay import ALWAYS_COMMAND, Chooser, Command, match_names, nearest_to, parse
from away.pointer.magnet import Magnet
from away.pointer.targets import Snapshot, Target, TargetBackend, TargetCache
from away.pointer.uia import make_backend

log = logging.getLogger(__name__)

POINTER_ACTION = "pointer"  # Action.kind of a smart pointer voice command
ACTION_WORDS = {"click": "Clicked", "double_click": "Double-clicked", "focus": "Focused"}
FRIENDLY_REFRESH_ERROR = (
    "Snap to buttons can't read this window right now (some apps hide their buttons). It keeps trying, and the "
    "cursor works normally meanwhile."
)


class PointerPhrases(dict):  # type: ignore[type-arg]
    """The bound voice phrases, plus "click ..." style commands recognised on the fly (always-on listening asks
    ``phrase in table`` and ``table.get(phrase)``)."""

    def __init__(self, bound: dict[str, Action], smart: SmartPointer) -> None:
        super().__init__(bound)
        self._smart = smart

    def get(self, key: Any, default: Any = None) -> Any:
        hit = super().get(key)
        if hit is not None:
            return hit
        act = self._smart.phrase_action(str(key))
        return act if act is not None else default

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key) or self._smart.phrase_action(str(key)) is not None


class SmartPointer:
    def __init__(
        self,
        cfg: Any,
        mouse: Any,
        *,
        backend: TargetBackend | None = None,
        publish: Callable[[str, dict[str, Any]], None] | None = None,
        notify: Callable[[str, str, str], Any] | None = None,
        is_bound: Callable[[str], bool] | None = None,
        paused: Callable[[], bool] | None = None,
        overlay: BadgeOverlay | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg  # the AppConfig, read live (Settings replaces sub-sections in place)
        self.mouse = mouse
        self.backend = backend if backend is not None else make_backend()
        self.publish = publish
        self.notify = notify
        self.is_bound = is_bound or (lambda _p: False)
        self.paused = paused or (lambda: False)
        self.clock = clock
        self.overlay = overlay if overlay is not None else BadgeOverlay()
        self.cache = TargetCache(self.backend, clock=clock, on_error=self._on_error, enabled=self._wanted)
        self.magnet = Magnet(lambda: self.cfg.hands.cursor.magnet, lambda: self.cache.snapshot)
        self.chooser = Chooser()
        self.last_message = ""

    # ------------------------------------------------------------------ state
    @property
    def available(self) -> bool:
        return bool(self.backend.available())

    def magnet_on(self) -> bool:
        return self.available and bool(self.cfg.hands.cursor.magnet.enabled)

    def voice_on(self) -> bool:
        return self.available and bool(getattr(self.cfg.voice, "click_by_name", False))

    def _wanted(self) -> bool:
        return self.magnet_on() or self.voice_on()

    def start(self) -> None:
        if self.available:
            self.cache.start()

    def stop(self) -> None:
        self.cache.stop()
        with contextlib.suppress(Exception):
            self.overlay.hide()

    def _emit(self, topic: str, data: dict[str, Any]) -> None:
        if self.publish is not None:
            with contextlib.suppress(Exception):
                self.publish(topic, data)

    def _say(self, text: str) -> None:
        self.last_message = text
        self._emit("pointer.say", {"text": text})

    def _on_error(self, msg: str) -> None:
        log.warning("smart pointer: UI Automation refresh failed: %s", msg)
        self._emit("pointer.status", {"ok": False, "text": FRIENDLY_REFRESH_ERROR, "detail": msg})
        if self.notify is not None:
            with contextlib.suppress(Exception):
                self.notify("Project Away: snap to buttons", FRIENDLY_REFRESH_ERROR, "pointer-uia-error")

    # ------------------------------------------------------------------ cursor
    def refine(self, x: int, y: int) -> tuple[int, int]:
        """Magnet on the mapped cursor position (desktop px)."""
        if not self.magnet_on():
            return x, y
        rx, ry = self.magnet.apply(float(x), float(y))
        return round(rx), round(ry)

    def resync(self, shown: tuple[float, float]) -> None:
        if self.magnet_on():
            self.magnet.resync(shown)

    # ------------------------------------------------------------------ voice
    def phrase_action(self, phrase: str) -> Action | None:
        """Is ``phrase`` a smart pointer command right now? Cheap: parses and, for "focus ...", looks at the cached
        targets; never refreshes. Only asked about phrases the bindings do not know."""
        if not self.voice_on():
            return None
        cmd = parse(phrase, choosing=self.chooser.active(self.clock()))
        if cmd is None:
            return None
        if cmd.action in ("choose", "cancel") or cmd.that or cmd.action in ALWAYS_COMMAND:
            return Action(POINTER_ACTION, phrase)
        if match_names(cmd.name, self.cache.snapshot.targets):  # "focus <name>" only when something matches
            return Action(POINTER_ACTION, phrase)
        return None

    def phrases(self, bound: dict[str, Action]) -> dict[str, Action]:
        return PointerPhrases(bound, self) if self.voice_on() else bound

    def voice(self, phrase: str) -> Action | None:
        """Run a smart pointer voice command. None: not one (or a bound phrase, or switched off / paused), so the
        bindings / dictation handle it as before."""
        if not self.voice_on() or self.paused():
            return None
        if self.is_bound(phrase):
            return None
        now = self.clock()
        cmd = parse(phrase, choosing=self.chooser.active(now))
        if cmd is None:
            return None
        done = Action(POINTER_ACTION, phrase)
        if cmd.action == "cancel":
            self.chooser.cancel()
            self.overlay.hide()
            self._say("Okay, cancelled.")
            return done
        if cmd.action == "choose":
            pending = self.chooser.pending
            picked = self.chooser.choose(cmd.number, now)
            if picked is None:
                n = len(pending.targets) if pending is not None else 0
                self._say(f"Say a number from 1 to {n}, or “cancel”.")
                return done
            self.overlay.hide()
            self._execute(picked[0], picked[1])
            return done
        snap = self.cache.refresh_now()  # the freshest tree for a spoken command (voice thread, not the cursor)
        if cmd.that:
            self._that(cmd, snap)
            return done
        found = match_names(cmd.name, snap.targets)
        if not found:
            if cmd.action not in ALWAYS_COMMAND:
                return None  # "focus on the budget" was dictation after all
            self._say(f"I couldn't find “{cmd.name}” in this window.")
            return done
        if len(found) == 1:
            self.chooser.cancel()
            self.overlay.hide()
            self._execute(cmd.action, found[0].target)
            return done
        targets = [m.target for m in found]
        self.chooser.offer(cmd.action, targets, now)
        self.overlay.show(badge_specs(targets))
        self._say(f"Which one? Say a number from 1 to {len(targets)}.")
        self._emit("pointer.choices", {"action": cmd.action, "names": [t.name for t in targets], "query": cmd.name})
        return done

    def look_point(self) -> tuple[float, float]:
        """Where "that" is: the gaze point when gaze is calibrated and a face is seen, else the cursor."""
        gp = getattr(self.mouse, "gaze_point", None)
        if gp is not None:
            with contextlib.suppress(Exception):
                pt = gp()
                if pt is not None:
                    return float(pt[0]), float(pt[1])
        x, y = self.mouse.mapper.pos
        return float(x), float(y)

    def _that(self, cmd: Command, snap: Snapshot) -> None:
        target = nearest_to(snap, self.look_point())
        if target is None:
            self._say("There's nothing to click right there.")
            return
        self.chooser.cancel()
        self.overlay.hide()
        self._execute(cmd.action, target)

    def _execute(self, action: str, target: Target) -> None:
        label = target.name or target.kind
        if action == "focus":
            try:
                if self.backend.focus(target):
                    self._say(f"Focused {label}.")
                    return
            except Exception as exc:  # some elements refuse focus: a click focuses them instead
                log.info("smart pointer: SetFocus failed (%s), clicking instead", exc)
        x, y = target.center
        mouse = self.mouse
        pos = mouse.mapper.warp(x, y)  # re-anchors the hand at the target (no jump back on the next move)
        self.magnet.reset()
        if getattr(mouse, "stabilizer", None) is not None:
            mouse.stabilizer.reset()
        mouse.backend.move(*pos)
        mouse.backend.click(2 if action == "double_click" else 1)
        self._say(f"{ACTION_WORDS.get(action, 'Clicked')} {label}.")
