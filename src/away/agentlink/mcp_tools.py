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
from away.agentlink.groups import resolve

LIVE_SCREEN = "away://screen/live"
EVENTS = "away://events"
INBOX = "away://inbox"
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

    async def group(ctx: Context | None, _group: str, **kw: Any) -> Any:
        method, args = resolve(_group, kw)
        return await call(ctx, method, **args)

    # v0.1.21: 15 tools (was 37), v0.1.22: 16 (inbox), v0.1.28: 18 (shell, ui_query). Related tools are one
    # tool with an `action`
    # (away.agentlink.groups); every old name still works on the REST API and /v1/rpc/<name>.
    # ------------------------------------------------------------------ read
    @mcp.tool()
    async def get_status(ctx: Context) -> dict:
        """Project Away's state: version, your permissions, health (fps per camera and why it is 0, latency,
        CPU/GPU), the person's context, computer-control state and the latest event number."""
        return await call(ctx, "get_status")

    @mcp.tool()
    async def get_context(ctx: Context) -> dict:
        """The person's live context: present or away, attention (facing_screen / looking_away / away), typing,
        dictating, active app, fatigue mode, who is using the camera, and whether now is a polite moment."""
        return await call(ctx, "get_context")

    @mcp.tool()
    async def events(
        ctx: Context,
        action: str = "wait",
        since: int = 0,
        timeout_s: float = 30.0,
        topics: list[str] | None = None,
        landmarks: bool = False,
    ) -> dict:
        """Events after event number `since`. action="wait": long-poll (up to 55 s; call again) for user/app events
        (presence, attention, summon, user.look, user.screen, wrong, detector, misfire, control, prompt, change;
        filter with `topics`). action="poll": return recent gestures/events at once plus health (landmarks=true
        adds the latest hand landmarks). Call again with the returned `seq`."""
        if action == "poll":
            return await group(ctx, "events", action="poll", since=since, landmarks=landmarks)
        return await group(ctx, "events", action="wait", since=since, timeout_s=timeout_s, topics=topics)

    @mcp.tool()
    async def settings(
        ctx: Context,
        action: str = "list",
        domain: str | None = None,
        changes: list[dict] | dict | None = None,
        why: str | None = None,
        what: str | None = None,
        limit: int | None = None,
        ops: list[dict] | dict | None = None,
    ) -> Any:
        """Read and change the person's settings (changing needs "Change my settings").
        - action="list": what can be changed; with `domain` (settings, zones, bindings, custom_gestures,
          profiles, presets, macros, gestures, cameras) its items, current values and allowed ranges.
        - action="change": `changes` (or `ops`) = [{"domain": "settings", "op": "set", "key": "hands.cursor.gain",
          "value": 1.8}, ...] or simply {"hands.cursor.gain": 1.8} (bindings: key "gesture:pinch", value
          "mouse:click"; presets: op "apply"; a failing camera: {"domain": "cameras", "op": "disable", "key":
          "<camera name>"}, undoable; models: {"domain": "models", "op": "switch", "key": "stt", "value":
          "whisper-small-cpu"} or op "mode" with key light|balanced|best), `why` = the reason, citing the
          person's preference. By default the person approves a before/after first.
        - action="undo": `what` = "last" | "today" | a change id. action="history": recent changes (`limit`).
        - action="config": the whole configuration (secrets redacted)."""
        return await group(
            ctx, "settings", action=action, domain=domain, changes=changes, why=why, what=what, limit=limit, ops=ops
        )

    @mcp.tool()
    async def preferences(
        ctx: Context, action: str = "list", text: str | None = None, number: int | None = None
    ) -> dict:
        """The person's preference notes (plain language). action="list" (read them before changing anything),
        "add" with `text` (something they told you; needs "Change my settings"), "remove" with `number`
        (1-based, as listed)."""
        return await group(ctx, "preferences", action=action, text=text, number=number)

    # -------------------------------------------------------------- diagnose
    @mcp.tool()
    async def diagnostics(
        ctx: Context,
        action: str,
        kind: str | None = None,
        device: str | None = None,
        seconds: float | None = None,
        quick: bool | None = None,
        job_id: str | None = None,
        note: str | None = None,
        path: str | None = None,
        lines: int | None = None,
        wait: bool | None = None,
        only: list[str] | None = None,
    ) -> Any:
        """Find out what is wrong. run/job need "Run camera and speed checks"; record, bundle, files, download
        and logs need "Share my logs and recordings" (off unless the person turns it on; secrets are hidden).
        - action="run": start a check, `kind` = "cameras" (every camera/backend/mode with measured fps; optional
          `device`; `quick`=true opens each camera on its remembered backend only and skips the mode sweep, a
          fast triage answer), "sweep" (the configured cameras together) or "benchmark" (model speeds). Returns
          a job id. Every check's result carries `elapsed_s` and `budget_hit` (whether a time limit stopped it).
        - action="job": that check's state and output (`job_id`); results are kept 10 min after it finished.
          A finished "cameras" check also carries `result.cameras`, one structured dict per camera: `id`
          (per-device identifier; names can collide), `name`,
          `enumerated_by` (v4l2|dshow|wmi|system_profiler|probe), `backends` (per backend id: `opened` bool,
          `fps` measured best or null, `error` text or null), `driver` (`status` ok|error|unknown and
          `config_error`; minimal until proposal #9's driver-state helper lands), `holder` (Project Away
          itself when it holds the camera, or null; other apps are not detected yet, so null does
          not mean the camera is free), and `verdict` (ok = frames flowed; driver = driver/config
          problem, e.g. "Element not
          found"; held = another app holds it; privacy = OS privacy blocked it; unknown = could not tell).
          `result.stopped` is true when the time budget cut the check short. Prefer this over parsing the
          prose `output`.
        - action="record": record the next `seconds` (max 120) of hand landmarks and events (never video).
        - action="bundle": build a diagnostics bundle (logs, redacted config, camera checks, last 30 s).
        - action="files": downloadable bundles, recordings and logs. action="download": one of them by `path`
          (base64). action="logs": the app log tail (`lines`).
        - action="selftest": run the one-click PC self-test (cameras, hand tracking, pointer, microphone, speech,
          voices, pop-ups, link, PC load, models); `wait`=true returns the report, otherwise it arrives as the event
          selftest.completed, in your inbox and as away://selftest/latest. action="selftest_latest": the newest
          report. Every result is pass/warn/fail in plain words with the fix the person can press."""
        if action == "download":
            data = await group(ctx, "diagnostics", action="download", path=path)
            raw = data if isinstance(data, bytes) else base64.b64decode(data["b64"])
            return {"path": path, "bytes": len(raw), "base64": base64.b64encode(raw).decode("ascii")}
        return await group(
            ctx,
            "diagnostics",
            action=action,
            kind=kind,
            device=device,
            seconds=seconds,
            quick=quick,
            job_id=job_id,
            note=note,
            lines=lines,
            wait=wait,
            only=only,
        )

    # ------------------------------------------------------------------- see
    @mcp.tool()
    async def look(ctx: Context, camera: str | None = None) -> list:
        """One low-resolution still from the camera (needs "See my camera when I allow"; the person sees a flash
        each time)."""
        raw, meta = _image(await call(ctx, "look", camera=camera))
        return [Image(data=raw, format="jpeg"), json.dumps(meta)]

    @mcp.tool()
    async def get_screen(
        ctx: Context,
        scale: float | None = None,
        monitor: int | None = None,
        region: dict | None = None,
        elements: bool = False,
    ) -> list:
        """A screenshot plus the focused window (needs "See my screen when I allow"; the person sees a flash).
        Orient, then zoom: first a wide shot (default: the whole desktop at scale 0.5; the metadata lists every
        monitor with index, x, y, w, h), then `monitor`=N for one display or `region`={x, y, width, height} in
        screen coordinates to read small text (scale up to 1.0). The longest side is capped at 2560 px.
        Screen point = origin + image point / scale (or pass from_image=true to pointer actions).
        elements=true adds the UI element tree (names you can pass to pointer actions)."""
        raw, meta = _image(
            await call(ctx, "get_screen", elements=elements, scale=scale, monitor=monitor, region=region)
        )
        return [Image(data=raw, format="jpeg"), json.dumps(meta, default=str)]

    # ------------------------------------------------------------------ human
    @mcp.tool()
    async def ask_user(
        ctx: Context,
        question: str | None = None,
        options: list[str] | None = None,
        timeout_s: float = 60.0,
        urgency: str = "normal",
        speak: bool = False,
        message: str | None = None,
    ) -> dict:
        """Ask the person on their screen. They answer with thumbs up (first/yes option), thumbs down (last/no),
        a finger count (option N), a custom gesture, a click or by voice. urgency low|normal waits for a polite
        moment; high shows at once. speak=true also reads the question aloud. Returns {answer, method,
        waited_s}; answer is null on timeout. (`message` works as another name for `question`.)"""
        return await call(
            ctx,
            "ask_user",
            question=question,
            options=options,
            timeout_s=timeout_s,
            urgency=urgency,
            speak=speak,
            message=message,
        )

    @mcp.tool()
    async def notify_user(
        ctx: Context, text: str | None = None, level: str = "info", speak: bool = False, message: str | None = None
    ) -> dict:
        """Show a short message on the person's screen ("Message from <you>", bottom-right, ~6 s; level
        info|success|warning|error). speak=true also reads it aloud with the computer's voice (when the person
        allows agents to talk). Returns {shown, spoken} - each false with a reason when it did not happen.
        Use it to answer a message from your inbox out loud. (`message` works as another name for `text`.)"""
        return await call(ctx, "notify_user", text=text, level=level, speak=speak, message=message)

    @mcp.tool()
    async def inbox(
        ctx: Context,
        action: str = "list",
        id: str | None = None,
        text: str | None = None,
        unread_only: bool = False,
        speak: bool = True,
    ) -> dict:
        """Messages the person spoke to you ("Hey <your name>, ..." or the summon gesture, then speaking). Each
        also arrives as a `user.message` event (events action="wait", topics=["user.message"]).
        - action="list": your messages, newest first (`unread_only`).
        - action="read": one message by `id` (marks it read).
        - action="reply": answer message `id` with `text`; it is shown on their screen and read aloud
          (speak=false to only show it). Same as notify_user with speak=true."""
        return await group(ctx, "inbox", action=action, id=id, text=text, unread_only=unread_only, speak=speak)

    # ---------------------------------------------------------------- control
    @mcp.tool()
    async def ui_query(ctx: Context) -> dict:
        """The foreground window's interactable UI elements with exact screen bounds (after control
        action="request" was approved; Windows UI Automation, no screenshot taken): [{id, name, role,
        bounds}]. Pass an `id` back as element_id= to pointer or keyboard actions to act on it exactly -
        no pixel guessing. Ids are yours only, expire with your next ui_query call, and stop resolving
        if the foreground window changes."""
        return await call(ctx, "ui_query")

    @mcp.tool()
    async def control(
        ctx: Context,
        action: str,
        why: str | None = None,
        minutes: float | None = None,
        steps: list[dict] | None = None,
    ) -> Any:
        """Using the person's computer (needs "Let it use my computer").
        - action="request": ask to use it (`why`; expires after `minutes`, default 30). Wait for their approval.
        - action="resume": continue after they took over (any move of their hand, mouse or keyboard pauses you).
        - action="stop": you are done (removes the border and banner). action="undo": undo your last undoable
          action. action="log": your recent actions. action="sequence": run several actions in one call
          (`steps`: a list like [{"action": "move", "x": 10, "y": 10}, {"action": "click"}]; each step is one
          pointer/keyboard/open action with its usual arguments). One approval covers the whole sequence, but
          risky steps still ask the person one by one, and the sequence stops at the first refusal or takeover.
          Each step counts against the action rate limit like a single action, and the whole sequence is undone
          with action="undo". Keep sequences short and related."""
        return await group(ctx, "control", action=action, why=why, minutes=minutes, steps=steps)

    @mcp.tool()
    async def pointer(
        ctx: Context,
        action: str,
        x: float | None = None,
        y: float | None = None,
        element: str | None = None,
        element_id: str | None = None,
        from_image: bool = False,
        x2: float | None = None,
        y2: float | None = None,
        to_element: str | None = None,
        to_element_id: str | None = None,
        dy: float | None = None,
        dx: float | None = None,
    ) -> dict:
        """Mouse actions (after control action="request" was approved): action = move | click | double_click |
        right_click | drag (to x2,y2 or `to_element`/`to_element_id`) | scroll (dy > 0 up, dy < 0 down, dx
        sideways). Target screen x,y, an element name from get_screen(elements=true), an element id from
        ui_query (exact bounds, no guessing), or image x,y with from_image=true (your last screenshot's origin
        and scale are applied). An id beats a name beats coordinates. Risky buttons (send, pay, delete...) ask
        the person first. The result reports the observed effect: `foreground` (the foreground window's title
        and process name right after the action) and `focused` (the focused UI element's label, when UI
        Automation has one). Check these: they tell you whether the click actually landed where you intended."""
        if action == "scroll" and dy is None and dx is None:
            dy = -3.0
        return await group(
            ctx,
            "pointer",
            action=action,
            x=x,
            y=y,
            element=element,
            element_id=element_id,
            from_image=from_image or None,
            x2=x2,
            y2=y2,
            to_element=to_element,
            to_element_id=to_element_id,
            dy=dy,
            dx=dx,
        )

    @mcp.tool()
    async def keyboard(
        ctx: Context,
        action: str,
        text: str | None = None,
        keys: str | None = None,
        expect: str | None = None,
        element_id: str | None = None,
    ) -> dict:
        """Keyboard actions (after control was approved): action="type" types `text` into the focused field
        (never into password fields or terminals; a newline counts as Enter and asks first in risky contexts).
        Pass `expect` to type only when the focused window's title contains that text (case-insensitive): a
        mismatch refuses before anything is typed, so text never lands in the wrong window. Pass element_id=
        (from ui_query) to click that field first so the input lands there. The result reports the observed
        effect: `foreground` (the foreground window's title and process name right after the action) and
        `focused` (the focused UI element's label, when UI Automation has one). action="press" presses `keys`,
        e.g. "enter", "ctrl+s", "alt+tab"."""
        return await group(ctx, "keyboard", action=action, text=text, keys=keys, expect=expect, element_id=element_id)

    @mcp.tool()
    async def open(ctx: Context, action: str, name: str | None = None, url: str | None = None) -> dict:
        """Open things (after control was approved): action="app" opens an app by `name` through the Start menu
        / Spotlight (not terminals or system tools); action="system" opens an allowlisted Windows system tool by
        `name` (device-manager or a Windows Settings page such as windows-update: use the name with
        spaces or dashes, e.g. "Device Manager"): it is launched straight through the shell, with no keystrokes
        and no command line, so opening it is navigation only and anything consequential inside it still asks
        first; action="url" opens an http(s) `url` in the default browser.
        The result reports the observed effect: `foreground` (the foreground window's title and process name
        right after the action) and `focused` (the focused UI element's label, when UI Automation has one)."""
        return await group(ctx, "open", action=action, name=name, url=url)

    @mcp.tool()
    async def shell(
        ctx: Context,
        action: str = "run",
        command: str | None = None,
        argv: list[str] | None = None,
        why: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Run a command on the person's PC, in two safety tiers (needs "Let it use my computer", inside an
        approved computer session: control action="request" first).
        - action="list": the tier-1 read-only commands (device/camera/process lists, the away CLI, the app log).
        - action="run": tier 1 = `command` set to one of those names runs at once (fixed, read-only, no shell
          is ever involved); tier 2 = anything else via `argv` asks the person first, every time, showing the
          exact command. `why` is your reason, shown with the approval question. Output is capped at 64 KB
          and secrets are hidden; password fields and credential stores are never touched.
        - action="log": every shell call so far (command, tier, result)."""
        return await group(ctx, "shell", action=action, command=command, argv=argv, why=why, limit=limit)

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

    @mcp.resource("away://inbox", mime_type="application/json")
    async def inbox_res() -> str:
        """Messages the person spoke to you (newest first)."""
        return json.dumps(await call(_ctx(mcp), "inbox_list"), default=str)

    @mcp.resource("away://selftest/latest", mime_type="application/json")
    async def selftest_res() -> str:
        """v0.1.24: the newest self-test report (pass/warn/fail per check, plain words, fix buttons)."""
        return json.dumps(await call(_ctx(mcp), "selftest_latest"), default=str)

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
