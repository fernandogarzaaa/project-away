"""What the live agent can do on this PC: Agent Link's own tools, called in-process.

The live agent acts as a built-in Agent Link agent (it is not listed among your connected agents and has no
token). Its permissions come from Settings > Live agent, and every Agent Link rule still applies:

* using the mouse/keyboard needs your approval on the PC each session, shows the border and banner, pauses the
  moment you move, never types into password fields or terminals, and asks before risky clicks;
* on top of that, the live agent asks Jev (or the chat model) "is this a send / pay / delete / irreversible
  action?" before each action and asks you first when the answer is yes (:meth:`ToolBox.act`);
* looking at the screen needs "Let it look at my screen", and only private models ever see the picture;
* settings changes go through Agent Link's change flow (shown, confirmed, undoable).

Anything read from the screen is fenced as untrusted data before it reaches a model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from away.liveagent import untrusted

AGENT_ID = "live-agent"
ACTIONS = ("click", "double_click", "right_click", "move", "scroll", "type_text", "press_keys", "open_app", "open_url")


class ToolError(RuntimeError):
    pass


def identity(name: str, s: Any) -> Any:
    """The live agent as an Agent Link :class:`~away.agentlink.auth.Agent` (in memory only)."""
    from away.agentlink import perms
    from away.agentlink.auth import Agent

    scopes = [perms.READ]
    if s.allow_screen:
        scopes.append(perms.SCREEN)
    if s.allow_camera:
        scopes.append(perms.SNAPSHOTS)
    if s.allow_control:
        scopes.append(perms.CONTROL)
    if s.allow_settings:
        scopes.append(perms.TUNE)
    now = time.time()
    return Agent(AGENT_ID, name, "local", scopes, "", now, now + 10 * 365 * 86400.0, last_seen=now)


class ToolBox:
    def __init__(
        self,
        service: Any,
        settings: Callable[[], Any],
        decider: Any = None,
        ask: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.service = service  # away.agentlink.service.Service, or None when Agent Link is off
        self.settings = settings
        self.decider = decider
        self.ask = ask or (getattr(getattr(service, "prompts", None), "ask", None) if service is not None else None)
        self.log: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self.service is not None

    def agent(self) -> Any:
        s = self.settings()
        return identity(s.name, s)

    def _need(self) -> Any:
        if self.service is None:
            raise ToolError("Agent Link is turned off, so I can't use the computer (Settings > Agent Link)")
        return self.service

    # ------------------------------------------------------------ read
    def context(self) -> str:
        svc = self._need()
        ctx = svc.get_context(self.agent())
        keep = {k: ctx.get(k) for k in ("active_app", "window", "voice", "paused", "control", "presence") if k in ctx}
        return untrusted.fence(keep, "context", 1500)

    def elements(self, limit: int = 120) -> tuple[list[str], str]:
        """UI element names on screen (needs "look at my screen"). Returns (names, fenced text)."""
        svc = self._need()
        s = self.settings()
        if not s.allow_screen:
            raise ToolError("I'm not allowed to look at the screen (Settings > Live agent)")
        uia = getattr(svc, "uia", None)
        if not getattr(uia, "available", False):
            return [], untrusted.fence("(this system cannot list on-screen elements)", "screen")
        tree = uia.tree(3, limit)
        names: list[str] = []
        lines: list[str] = []

        def walk(el: Any, depth: int) -> None:
            if el is None or len(names) >= limit:
                return
            if el.name and not el.is_password:
                names.append(el.name)
                lines.append(f"{'  ' * depth}{el.control_type or 'element'}: {el.name}")
            for c in el.children:
                walk(c, depth + 1)

        walk(tree, 0)
        return names, untrusted.fence("\n".join(lines), "screen", 4000)

    def screenshot(self) -> tuple[bytes, dict[str, Any]]:
        svc = self._need()
        if not self.settings().allow_screen:
            raise ToolError("I'm not allowed to look at the screen (Settings > Live agent)")
        jpeg, meta = svc.get_screen(self.agent(), elements=False, scale=0.5)
        return jpeg, meta

    def focused_password(self) -> bool:
        svc = self.service
        uia = getattr(svc, "uia", None) if svc is not None else None
        try:
            el = uia.focused() if getattr(uia, "available", False) else None
            return bool(el is not None and el.is_password)
        except Exception:
            return False

    # ------------------------------------------------------------ act
    def act(self, action: str, why: str = "", screen_text: str = "", **kw: Any) -> dict[str, Any]:
        svc = self._need()
        s = self.settings()
        if not s.allow_control:
            raise ToolError("I'm not allowed to use the mouse and keyboard (Settings > Live agent)")
        if action not in ACTIONS:
            raise ToolError(f"unknown action {action!r}")
        agent = self.agent()
        target = str(kw.get("element") or kw.get("text") or kw.get("keys") or kw.get("name") or kw.get("url") or "")
        if self.decider is not None and action not in ("move", "scroll"):
            risk = self.decider.risk(f"{action} {target}".strip(), target, screen_text)
            self.log.append({"t": time.time(), "check": "risk", "risky": bool(risk.value), "by": risk.source})
            if risk.value:
                if self.ask is None:
                    raise ToolError("that action may not be undoable and I can't ask you on screen, so I stopped")
                res = self.ask(
                    f"{agent.name} wants to {action.replace('_', ' ')} '{target[:60]}'",
                    ["Allow", "Deny"],
                    title="Are you sure?",
                    agent=agent.name,
                    details=["This may not be undoable (send, pay, delete...)."] + ([f"Why: {why[:150]}"] if why else []),
                    timeout_s=60.0,
                    urgency="high",
                    kind="confirm",
                )
                if res.get("answer") != "Allow":
                    raise ToolError(f"you did not allow '{target[:60]}'")
        if svc.control.grant is None or svc.control.grant.agent_id != AGENT_ID:
            svc.request_control(agent, why=why or "a request you just made", minutes=5)
        if action in ("click", "double_click", "right_click", "move") and kw.get("element") and self.decider is not None:
            try:
                names, _ = self.elements()
            except ToolError:
                names = []
            if names:
                pick = self.decider.pick_element(str(kw["element"]), names)
                if pick is not None:
                    kw["element"] = pick.value
        args = {k: v for k, v in kw.items() if k in ("x", "y", "element", "text", "keys", "name", "url", "dx", "dy",
                                                       "amount", "direction")}
        res: dict[str, Any] = svc.act(agent, action, **args)
        self.log.append({"t": time.time(), "action": action, "target": target[:80]})
        return res

    def finish(self) -> None:
        svc = self.service
        if svc is not None:
            try:
                svc.stop_control(self.agent())
            except Exception:
                pass

    # ------------------------------------------------------------ settings
    def change_settings(self, ops: Any, why: str = "") -> dict[str, Any]:
        svc = self._need()
        if not self.settings().allow_settings:
            raise ToolError("I'm not allowed to change settings (Settings > Live agent)")
        res: dict[str, Any] = svc.change_settings(self.agent(), ops, why=why)
        return res

    # ------------------------------------------------------------ hand-off
    def handoff(self, text: str) -> str:
        """A bigger job for another agent: into its inbox (it is woken up if it gave a wake-up address)."""
        svc = self._need()
        target = ""
        want = str(self.settings().handoff_agent or "").strip().lower()
        if want:
            for a in svc.registry.known() if hasattr(svc.registry, "known") else svc.registry.list():
                if want in (a.id.lower(), a.name.lower()):
                    target = a.id
        res = svc.deliver_message(target, text, "live-agent")
        if not res.get("sent"):
            raise ToolError(str(res.get("reason") or "no other agent is connected"))
        return str(res.get("label") or "sent")
