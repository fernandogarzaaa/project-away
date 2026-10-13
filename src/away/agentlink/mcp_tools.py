"""The MCP surface (tools + resources), defined once and served two ways:

* in the app, over Streamable HTTP (``/mcp`` with a bearer token, or the pairing URL ``/c/<code>/mcp``): the
  backend calls :class:`~away.agentlink.service.Service` directly with the agent the request authenticated as;
* by ``away mcp`` over stdio for local agents (Claude Desktop, Cursor...): the backend forwards every call to the
  running app's local API with that agent's token (:mod:`away.agentlink.bridge`).

Needs the ``mcp`` package (the ``[agent]`` extra). Tools are thin: names and parameters an LLM can use, plain
results; all rules live in the service.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import Callable
from typing import Any, Protocol

from mcp import types
from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings

from away.agentlink.about import INSTRUCTIONS

LIVE_SCREEN = "away://screen/live"
EVENTS = "away://events"
SUBSCRIBABLE = (LIVE_SCREEN, EVENTS)


class Backend(Protocol):
    def call(self, ctx: Any, method: str, **kwargs: Any) -> Any: ...


def _image(data: Any) -> tuple[bytes, dict[str, Any]]:
    """A service image result: (bytes, meta) in-process, {"image_b64", "meta"} over the bridge."""
    if isinstance(data, tuple):
        return data[0], data[1]
    return base64.b64decode(data["image_b64"]), dict(data.get("meta") or {})


def build(backend: Backend, name: str = "Project Away", screen_fps: Callable[[], float] = lambda: 1.0) -> Any:
    mcp = FastMCP(
        name,
        instructions=INSTRUCTIONS,
        stateless_http=False,
        # the bearer token / pairing code is the access control; the tunnel hostname changes every session
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        streamable_http_path="/mcp",
        log_level="WARNING",
    )

    async def call(ctx: Context | None, method: str, **kw: Any) -> Any:
        import anyio

        return await anyio.to_thread.run_sync(lambda: backend.call(ctx, method, **kw))

    # ------------------------------------------------------------------ read
    @mcp.tool()
    async def get_status(ctx: Context) -> dict:
        """Project Away's state: version, your permissions, health (fps, latency, CPU/GPU), the person's context,
        computer-control state and the latest event number."""
        return await call(ctx, "get_status")

    @mcp.tool()
    async def get_context(ctx: Context) -> dict:
        """The person's live context: present or away, attention (facing_screen / looking_away / away), typing,
        dictating, active app, fatigue mode, and whether now is a polite moment to interrupt."""
        return await call(ctx, "get_context")

    @mcp.tool()
    async def poll_events(ctx: Context, since: int = 0, landmarks: bool = False) -> dict:
        """Recent gestures and app events after event number `since`, plus health; with landmarks=true also the
        latest 21 hand landmarks per camera. Call again with the returned `seq` to stream."""
        return await call(ctx, "poll", since=since, landmarks=landmarks)

    @mcp.tool()
    async def wait_for_event(
        ctx: Context, since: int = 0, timeout_s: float = 30.0, topics: list[str] | None = None
    ) -> dict:
        """Wait (long-poll, up to 120 s) for user-initiated or app events after `since`: presence, attention,
        summon, user.look, user.screen, wrong ("that was wrong"), detector, misfire, control, prompt, change."""
        return await call(ctx, "wait_for_event", since=since, timeout_s=timeout_s, topics=topics)

    @mcp.tool()
    async def get_logs(ctx: Context, lines: int = 100) -> dict:
        """The tail of the app log and recent events."""
        return await call(ctx, "get_logs", lines=lines)

    @mcp.tool()
    async def get_config(ctx: Context) -> dict:
        """The current configuration (secrets redacted) and the active profile."""
        return await call(ctx, "get_config")

    @mcp.tool()
    async def list_settings(ctx: Context, domain: str | None = None) -> dict:
        """What can be changed. Without `domain`: the kinds (settings, zones, bindings, custom_gestures,
        profiles, presets, macros...). With a domain: its items with current values and allowed ranges."""
        return await call(ctx, "list_settings", domain=domain)

    # ------------------------------------------------------------------ tune
    @mcp.tool()
    async def set_config(ctx: Context, changes: list[dict], why: str) -> dict:
        """Change settings, zones, bindings, custom gestures, profiles or presets (needs "Change my settings").
        `changes`: [{"domain": "settings", "op": "set", "key": "hands.cursor.gain", "value": 1.8}, ...];
        domains/ops from list_settings (bindings: key "gesture:pinch", value "mouse:click", optional "profile";
        presets: {"domain":"presets","op":"apply","key":"Precise"}). `why`: the reason, citing the person's
        preference when there is one. By default the person approves a before/after first. Undoable."""
        return await call(ctx, "change_settings", ops=changes, why=why)

    @mcp.tool()
    async def undo_change(ctx: Context, what: str = "last") -> dict:
        """Undo settings changes: "last" (the newest batch), "today" (everything you changed today), or a change
        / batch id from change_history."""
        return await call(ctx, "undo_changes", what=what)

    @mcp.tool()
    async def change_history(ctx: Context, limit: int = 30) -> list:
        """Recent settings changes: who, why, what, undone or not."""
        return await call(ctx, "change_history", limit=limit)

    @mcp.tool()
    async def get_preferences(ctx: Context) -> dict:
        """The person's preference notes (plain language). Read before changing anything."""
        return await call(ctx, "get_preferences")

    @mcp.tool()
    async def add_preference(ctx: Context, text: str) -> dict:
        """Remember a preference the person told you (needs "Change my settings")."""
        return await call(ctx, "add_preference", text=text)

    @mcp.tool()
    async def remove_preference(ctx: Context, number: int) -> dict:
        """Remove preference line `number` (1-based, as listed)."""
        return await call(ctx, "remove_preference", number=number)

    # -------------------------------------------------------------- diagnose
    @mcp.tool()
    async def run_diagnose(
        ctx: Context, kind: str = "cameras", device: str | None = None, seconds: float | None = None
    ) -> dict:
        """Start a check: "cameras" (every camera/backend/mode, optionally one `device`), "sweep" (stream the
        configured cameras together) or "benchmark" (model speeds). Returns a job; poll get_job."""
        return await call(ctx, "run_diagnose", kind=kind, device=device, seconds=seconds)

    @mcp.tool()
    async def get_job(ctx: Context, job_id: str) -> dict:
        """State and output of a check started with run_diagnose."""
        return await call(ctx, "get_job", job_id=job_id)

    @mcp.tool()
    async def record_session(ctx: Context, seconds: float = 10.0, note: str = "") -> dict:
        """Record the next `seconds` (max 120) of hand landmarks and events (never video) to a file that
        `away replay` re-runs headless. Returns its path for download_file."""
        return await call(ctx, "record_session", seconds=seconds, note=note)

    @mcp.tool()
    async def get_bundle(ctx: Context, note: str = "") -> dict:
        """Build a diagnostics bundle (logs, redacted config, camera diagnosis, stats, last 30 s of landmarks)."""
        return await call(ctx, "get_bundle", note=note)

    @mcp.tool()
    async def download_file(ctx: Context, path: str) -> dict:
        """Download a bundle, recording or log by the path another tool returned (base64)."""
        data = await call(ctx, "read_file", rel=path)
        raw = data if isinstance(data, bytes) else base64.b64decode(data["b64"])
        return {"path": path, "bytes": len(raw), "base64": base64.b64encode(raw).decode("ascii")}

    # ------------------------------------------------------------------- see
    @mcp.tool()
    async def look(ctx: Context, camera: str | None = None) -> list:
        """One low-resolution still from the camera (needs "See my camera when I allow"; the person sees a flash
        each time)."""
        raw, meta = _image(await call(ctx, "look", camera=camera))
        return [Image(data=raw, format="jpeg"), json.dumps(meta)]

    @mcp.tool()
    async def get_screen(ctx: Context, elements: bool = False) -> list:
        """One low-resolution screenshot plus the focused window (needs "See my screen when I allow"; the person
        sees a flash). elements=true adds the UI element tree (names you can pass to click etc.). Coordinates in
        the image divided by `scale` are screen coordinates (or pass from_image=true to actions)."""
        raw, meta = _image(await call(ctx, "get_screen", elements=elements))
        return [Image(data=raw, format="jpeg"), json.dumps(meta, default=str)]

    # ------------------------------------------------------------------ human
    @mcp.tool()
    async def ask_user(
        ctx: Context, question: str, options: list[str] | None = None, timeout_s: float = 60.0, urgency: str = "normal"
    ) -> dict:
        """Ask the person on their screen. They answer with thumbs up (first/yes option), thumbs down (last/no),
        a finger count (option N), a custom gesture, a click or by voice. urgency low|normal waits for a polite
        moment; high shows at once. Returns {answer, method, waited_s}; answer is null on timeout."""
        return await call(ctx, "ask_user", question=question, options=options, timeout_s=timeout_s, urgency=urgency)

    @mcp.tool()
    async def notify_user(ctx: Context, text: str, level: str = "info", speak: bool = False) -> dict:
        """Show a short toast on the person's screen (level info|success|warning|error), optionally spoken."""
        return await call(ctx, "notify_user", text=text, level=level, speak=speak)

    # ---------------------------------------------------------------- control
    @mcp.tool()
    async def request_control(ctx: Context, why: str, minutes: float | None = None) -> dict:
        """Ask the person to let you use their computer (needs "Let it use my computer"). They approve on the PC;
        it expires (default 30 min). Any move of their hand, mouse or keyboard pauses you."""
        return await call(ctx, "request_control", why=why, minutes=minutes)

    @mcp.tool()
    async def resume_control(ctx: Context) -> dict:
        """After the person took over (paused), ask them to let you continue."""
        return await call(ctx, "resume_control")

    @mcp.tool()
    async def stop_control(ctx: Context) -> dict:
        """You are done using the computer: removes the border and banner."""
        return await call(ctx, "stop_control")

    async def act(ctx: Context, action: str, **kw: Any) -> dict:
        return await call(ctx, "act", action=action, **{k: v for k, v in kw.items() if v is not None})

    @mcp.tool()
    async def move(
        ctx: Context,
        x: float | None = None,
        y: float | None = None,
        element: str | None = None,
        from_image: bool = False,
    ) -> dict:
        """Move the pointer to screen x,y or to the element named `element`."""
        return await act(ctx, "move", x=x, y=y, element=element, from_image=from_image)

    @mcp.tool()
    async def click(
        ctx: Context,
        x: float | None = None,
        y: float | None = None,
        element: str | None = None,
        from_image: bool = False,
    ) -> dict:
        """Left click at x,y or on the element named `element` (risky buttons ask the person first)."""
        return await act(ctx, "click", x=x, y=y, element=element, from_image=from_image)

    @mcp.tool()
    async def double_click(
        ctx: Context,
        x: float | None = None,
        y: float | None = None,
        element: str | None = None,
        from_image: bool = False,
    ) -> dict:
        """Double click at x,y or on an element."""
        return await act(ctx, "double_click", x=x, y=y, element=element, from_image=from_image)

    @mcp.tool()
    async def right_click(
        ctx: Context,
        x: float | None = None,
        y: float | None = None,
        element: str | None = None,
        from_image: bool = False,
    ) -> dict:
        """Right click at x,y or on an element."""
        return await act(ctx, "right_click", x=x, y=y, element=element, from_image=from_image)

    @mcp.tool()
    async def drag(
        ctx: Context,
        x: float | None = None,
        y: float | None = None,
        x2: float | None = None,
        y2: float | None = None,
        element: str | None = None,
        to_element: str | None = None,
        from_image: bool = False,
    ) -> dict:
        """Drag from x,y (or `element`) to x2,y2 (or `to_element`)."""
        return await act(
            ctx, "drag", x=x, y=y, x2=x2, y2=y2, element=element, to_element=to_element, from_image=from_image
        )

    @mcp.tool()
    async def scroll(
        ctx: Context,
        dy: float = -3.0,
        dx: float = 0.0,
        x: float | None = None,
        y: float | None = None,
        element: str | None = None,
    ) -> dict:
        """Scroll (dy > 0 up, dy < 0 down; dx sideways), optionally over x,y or an element first."""
        return await act(ctx, "scroll", dy=dy, dx=dx, x=x, y=y, element=element)

    @mcp.tool()
    async def type_text(ctx: Context, text: str) -> dict:
        """Type text into the focused field (never into password fields or terminals; a newline counts as Enter
        and asks first in risky contexts)."""
        return await act(ctx, "type_text", text=text)

    @mcp.tool()
    async def press_keys(ctx: Context, keys: str) -> dict:
        """Press a key or a combination, e.g. "enter", "ctrl+s", "alt+tab"."""
        return await act(ctx, "press_keys", keys=keys)

    @mcp.tool()
    async def open_app(ctx: Context, name: str) -> dict:
        """Open an app by name through the Start menu / Spotlight (not terminals or system tools)."""
        return await act(ctx, "open_app", name=name)

    @mcp.tool()
    async def open_url(ctx: Context, url: str) -> dict:
        """Open an http(s) link in the default browser."""
        return await act(ctx, "open_url", url=url)

    @mcp.tool()
    async def undo_last_action(ctx: Context) -> dict:
        """Undo your last undoable computer action (pointer move, scroll, typed text)."""
        return await call(ctx, "undo_last_action")

    @mcp.tool()
    async def disconnect(ctx: Context) -> dict:
        """Disconnect yourself from Project Away (the person can reconnect you later)."""
        return await call(ctx, "disconnect")

    # ------------------------------------------------------------- resources
    @mcp.resource("away://about", mime_type="text/markdown")
    async def about_res() -> str:
        """Briefing: what Project Away is, your permissions, the person's live context, how to talk to them."""
        return str(await call(_ctx(mcp), "about"))

    @mcp.resource("away://preferences", mime_type="text/markdown")
    async def prefs_res() -> str:
        """The person's preference notes."""
        return str((await call(_ctx(mcp), "get_preferences"))["text"])

    @mcp.resource("away://context", mime_type="application/json")
    async def context_res() -> str:
        """Live context (presence, attention, typing, dictating, active app)."""
        return json.dumps(await call(_ctx(mcp), "get_context"), default=str)

    @mcp.resource(EVENTS, mime_type="application/json")
    async def events_res() -> str:
        """The latest events (subscribe to be told when a new one arrives)."""
        return json.dumps(await call(_ctx(mcp), "wait_for_event", since=0, timeout_s=0), default=str)

    @mcp.resource(LIVE_SCREEN, mime_type="image/jpeg")
    async def screen_res() -> bytes:
        """The live screen view (subscribe for updates at ~1-2 fps; needs "See my screen when I allow")."""
        raw, _meta = _image(await call(_ctx(mcp), "get_screen", elements=False, quiet=True))
        return raw

    @mcp.resource("away://user/last-capture", mime_type="image/jpeg")
    async def capture_res() -> bytes:
        """The picture the person sent you with "hey agent, look" or the summon gesture."""
        raw, _meta = _image(await call(_ctx(mcp), "last_capture"))
        return raw

    _add_subscriptions(mcp, backend, call, screen_fps, types)
    return mcp


def _ctx(mcp: Any) -> Any:
    try:
        return mcp.get_context()
    except Exception:
        return None


def _add_subscriptions(mcp: Any, backend: Backend, call: Any, screen_fps: Callable[[], float], types: Any) -> None:
    """resources/subscribe for away://events (pushed when an event arrives) and away://screen/live (pushed at the
    live-view rate). FastMCP 1.x has no subscription support of its own, so the low-level handlers are used and
    the capability is advertised by wrapping ``get_capabilities``."""
    server = mcp._mcp_server
    tasks: dict[tuple[int, str], asyncio.Task[Any]] = {}

    async def pump(session: Any, uri: str, ctx: Any) -> None:
        from pydantic import AnyUrl

        seq = 0
        try:
            if uri == EVENTS:
                first = await call(ctx, "wait_for_event", since=0, timeout_s=0)
                seq = int(first.get("seq", 0))
            else:
                await call(ctx, "screen_watch", on=True)
            while True:
                if uri == EVENTS:
                    res = await call(ctx, "wait_for_event", since=seq, timeout_s=25)
                    if res.get("events"):
                        seq = int(res["seq"])
                        await session.send_resource_updated(AnyUrl(uri))
                else:
                    await asyncio.sleep(1.0 / max(0.2, min(2.0, float(screen_fps()))))
                    await session.send_resource_updated(AnyUrl(uri))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            if uri == LIVE_SCREEN:
                with contextlib.suppress(Exception):
                    await call(ctx, "screen_watch", on=False)

    @server.subscribe_resource()
    async def on_subscribe(uri: Any) -> None:
        u = str(uri)
        if u not in SUBSCRIBABLE:
            return
        rc = server.request_context
        key = (id(rc.session), u)
        if key in tasks and not tasks[key].done():
            return
        if u == LIVE_SCREEN:
            await call(_ctx(mcp), "get_screen", elements=False, quiet=True)  # permission check up front
        tasks[key] = asyncio.create_task(pump(rc.session, u, _ctx(mcp)))

    @server.unsubscribe_resource()
    async def on_unsubscribe(uri: Any) -> None:
        rc = server.request_context
        t = tasks.pop((id(rc.session), str(uri)), None)
        if t is not None:
            t.cancel()

    orig = server.get_capabilities

    def caps(*a: Any, **k: Any) -> Any:
        c = orig(*a, **k)
        if c.resources is not None:
            c.resources.subscribe = True
        return c

    server.get_capabilities = caps  # type: ignore[method-assign]
