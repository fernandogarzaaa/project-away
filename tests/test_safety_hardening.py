"""Trust and safety hardening (external audit, nine findings + two small fixes).

Every test here failed on the code before its fix (checked by running it against the unfixed commit).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element, best_match
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeUIA:
    available = True

    def __init__(self, focused: Element | None = None, at: Any = None, elements: Any = ()) -> None:
        self._focused, self._at, self._els = focused, at, list(elements)

    def focused(self) -> Element | None:
        return self._focused

    def element_at(self, x: int, y: int) -> Element | None:
        if callable(self._at):
            return self._at(x, y)  # type: ignore[no-any-return]
        return self._at

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return best_match(self._els, name, control_type)

    def tree(self, *a: Any) -> None:
        return None


TEXT_BOX = Element("Message", "EditControl", rect=(0, 0, 9, 9))


def _control(
    clock: Clock,
    answers: list[str] | None = None,
    uia: Any = None,
    backend: Any = None,
    secure: Any = False,
    platform: str = "linux",
    sleep: Any = None,
) -> tuple[ComputerControl, Any, list[dict[str, Any]]]:
    backend = backend if backend is not None else RecordingBackend()
    asked: list[dict[str, Any]] = []
    queue = list(answers or ["Allow"])

    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        asked.append({"question": question, "options": options, **kw})
        return {"answer": queue.pop(0) if queue else None, "method": "click"}

    hooks = ControlHooks(
        backend=backend,
        platform=platform,
        secure_desktop=secure if callable(secure) else (lambda: secure),
        foreground=lambda: None,
        grab=lambda: None,
        sleep=sleep or (lambda _s: None),
        uia=uia if uia is not None else FakeUIA(focused=TEXT_BOX, at=Element("Canvas", "PaneControl")),
    )
    ctl = ComputerControl(hooks, ask, ttl_s=1800, rate_per_s=50, clock=clock)
    return ctl, backend, asked


# ------------------------------------------------------------------------- 1. undo needs a live grant
def test_undo_needs_a_live_unpaused_grant_owned_by_the_calling_agent() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow", "Allow"])
    ctl.hooks.cursor_pos = lambda: (50, 60)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=300, y=300)
    n = len(backend.calls)
    with pytest.raises(ControlRefused):
        ctl.undo_last("a2")  # another agent cannot undo (or even trigger) anything
    ctl.stop("you pressed Stop")
    with pytest.raises(ControlRefused):
        ctl.undo_last("a1")  # the grant ended: no more input from this agent
    assert len(backend.calls) == n
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=310, y=310)
    clock.advance(5)
    assert ctl.user_input("mouse")  # the person took over
    n = len(backend.calls)
    with pytest.raises(ControlRefused, match="paused"):
        ctl.undo_last("a1")
    assert len(backend.calls) == n


def test_undo_only_reverses_the_agents_own_actions_and_expires() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow", "Allow"])
    ctl.hooks.cursor_pos = lambda: (1, 2)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=300, y=300)
    ctl.stop()
    ctl.request("a2", "Gemini")
    with pytest.raises(ControlRefused, match="nothing"):
        ctl.undo_last("a2")  # a1's move is not a2's to undo
    clock.advance(1801)
    with pytest.raises(ControlRefused):
        ctl.undo_last("a2")
    assert ctl.undo_last_by_user()["undone"] == "move"  # the person's own Undo button still works


def test_service_undo_passes_the_agent(tmp_path: Path) -> None:
    from away import config as C
    from away.agentlink.host import build_core

    cfg = C.AppConfig()
    core = build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: tmp_path / "c")
    seen: list[Any] = []
    core.control.undo_last = lambda *a, **k: seen.append((a, k)) or {}  # type: ignore[method-assign]
    a, _ = core.registry.issue("Claude", ["read", "control"], 3600)
    core.service.undo_last_action(a)
    assert seen == [((a.id,), {})]


# --------------------------------------------------------------- 2. takeover windows (mouse vs keyboard echo)
def test_real_mouse_movement_pauses_the_agent_while_it_types_a_long_text() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock)
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    ctl.act("a1", "type_text", text="x" * 2000)  # used to hide every mouse movement for 40.5 s
    clock.advance(0.05)
    assert ctl.user_input("mouse")
    assert ctl.status()["paused"]


def test_keyboard_echo_is_ignored_only_briefly_after_each_typed_chunk() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock)
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    ctl.act("a1", "type_text", text="y" * 2000)
    assert not ctl.user_input("keyboard")  # the agent's own keys echoing back
    typed = [c for c in backend.calls if c[0] == "type_text"]
    assert len(typed) > 1 and all(len(c[1][0]) <= 16 for c in typed)  # small chunks, one short window each
    clock.advance(1.0)
    assert ctl.user_input("keyboard")  # one second later a key press is the person


def test_mouse_echo_of_the_agents_own_move_is_ignored_but_not_for_long() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock)
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    ctl.act("a1", "move", x=5, y=5)
    assert not ctl.user_input("mouse")
    assert ctl.user_input("keyboard")  # a key press during a mouse move is never the agent's echo


def test_approval_grace_is_short_and_never_blocks_a_deliberate_stop() -> None:
    clock = Clock()
    ctl, _, _ = _control(clock)
    ctl.request("a1", "Claude")
    assert not ctl.user_input("hand")  # the thumbs up that approved it
    clock.advance(1.0)
    assert ctl.user_input("mouse")  # a second later, moving the mouse takes over
    ctl2, backend2, _ = _control(clock)
    ctl2.request("a1", "Claude")
    assert ctl2.stop("you held up an open palm")  # inside the grace window: the stop still works at once
    with pytest.raises(ControlRefused):
        ctl2.act("a1", "move", x=1, y=1)
    assert backend2.calls == []


# ------------------------------------------------------------- 3. stopping an action that is already running
class BlockingBackend(RecordingBackend):
    """Records like RecordingBackend, and blocks inside the ``n``-th call of ``method`` until ``go`` is set (so a
    test can stop or take over while the action is in the middle of a drag or of typing)."""

    def __init__(self, method: str, n: int) -> None:
        super().__init__()
        self.method, self.n, self.seen = method, n, 0
        self.reached, self.go = threading.Event(), threading.Event()

    def _maybe_block(self, name: str) -> None:
        if name == self.method:
            self.seen += 1
            if self.seen == self.n:
                self.reached.set()
                assert self.go.wait(5.0)

    def move(self, x: int, y: int) -> None:
        super().move(x, y)
        self._maybe_block("move")

    def type_text(self, text: str) -> None:
        super().type_text(text)
        self._maybe_block("type_text")

    def release_modifiers(self) -> None:
        self.calls.append(("release_modifiers", ()))


def _run_in_thread(fn: Any) -> tuple[threading.Thread, dict[str, Any]]:
    out: dict[str, Any] = {}

    def run() -> None:
        try:
            out["result"] = fn()
        except Exception as exc:
            out["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_stop_interrupts_a_drag_in_the_middle_and_lets_go_of_the_button() -> None:
    clock = Clock()
    backend = BlockingBackend("move", 4)  # start point + 2 steps done, blocked in the 3rd step
    ctl, _, _ = _control(clock, backend=backend)
    ctl.request("a1", "Claude")
    t, out = _run_in_thread(lambda: ctl.act("a1", "drag", x=0, y=0, x2=800, y2=0))
    assert backend.reached.wait(5.0)
    assert ctl.stop("you pressed the stop keys")
    backend.go.set()
    t.join(5.0)
    assert isinstance(out.get("error"), ControlRefused) and "stop" in str(out["error"])
    after = backend.calls[[c[0] for c in backend.calls].index("press") :]
    assert [c[0] for c in after].count("move") == 3  # no step after the stop
    assert ("release", ()) in after  # the mouse button is never left held down
    assert ("release_modifiers", ()) in after
    assert ctl.log[-1].result == "stopped"


def test_moving_the_mouse_interrupts_typing_between_chunks() -> None:
    clock = Clock()
    backend = BlockingBackend("type_text", 2)
    ctl, _, _ = _control(clock, backend=backend)
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    t, out = _run_in_thread(lambda: ctl.act("a1", "type_text", text="abcdefgh" * 50))
    assert backend.reached.wait(5.0)
    assert ctl.user_input("mouse")  # the person grabs the mouse while it types
    backend.go.set()
    t.join(5.0)
    assert isinstance(out.get("error"), ControlRefused) and "mouse" in str(out["error"])
    assert [c[0] for c in backend.calls].count("type_text") == 2  # the rest of the text is never typed
    assert backend.calls[-1] == ("release_modifiers", ())


def test_after_resume_a_new_action_is_not_cancelled_by_the_old_takeover() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, ["Allow", "Continue"])
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    assert ctl.user_input("mouse")
    ctl.resume("a1")
    ctl.act("a1", "type_text", text="hello world, again")
    assert "".join(c[1][0] for c in backend.calls if c[0] == "type_text") == "hello world, again"


def test_open_app_checks_for_a_stop_before_each_step() -> None:
    clock = Clock()
    stopped: dict[str, Any] = {}
    ctl, backend, _ = _control(clock, platform="win32", uia=FakeUIA(focused=TEXT_BOX))

    def sleep(_s: float) -> None:
        if not stopped:
            stopped["x"] = ctl.stop("you said stop")

    ctl.hooks.sleep = sleep
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="stop"):
        ctl.act("a1", "open_app", name="Notepad")
    assert backend.calls == [("key", ("cmd",)), ("release_modifiers", ())]  # never typed the name or pressed Enter


# ------------------------------------------------------------- 4. clicks, drags and keys it cannot classify
class NoUIA:
    available = False
    reason = "UI Automation is not available"

    def focused(self) -> None:
        return None

    def element_at(self, x: int, y: int) -> None:
        return None


class BrokenUIA(FakeUIA):
    def element_at(self, x: int, y: int) -> Element | None:
        raise RuntimeError("COM error")

    def focused(self) -> Element | None:
        raise RuntimeError("COM error")


@pytest.mark.parametrize("uia", [NoUIA(), FakeUIA(focused=TEXT_BOX, at=None), BrokenUIA()])
def test_a_click_on_something_it_cannot_identify_asks_first(uia: Any) -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=uia)
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act("a1", "click", x=640, y=300)
    assert asked[-1]["kind"] == "confirm" and "(640, 300)" in asked[-1]["question"]
    assert any("can't tell" in d for d in asked[-1]["details"])
    assert backend.calls == []


def test_allow_for_this_session_covers_unlabelled_clicks_until_the_grant_ends() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Allow for this session", "Allow"], uia=NoUIA())
    ctl.request("a1", "Claude")
    ctl.act("a1", "click", x=1, y=1)
    n = len(asked)
    ctl.act("a1", "double_click", x=2, y=2)
    assert len(asked) == n and backend.calls[-1] == ("click", (2,))
    ctl.stop()
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused):
        ctl.act("a1", "click", x=1, y=1)  # a new session asks again (and nobody answers)


def test_a_drag_ending_on_a_risky_target_asks_with_the_destination() -> None:
    clock = Clock()
    file_el = Element("report.docx", "ListItemControl", rect=(0, 0, 40, 40))
    bin_el = Element("Recycle Bin", "ListItemControl", rect=(400, 0, 440, 40))
    uia = FakeUIA(at=lambda x, y: file_el if x < 100 else bin_el, elements=[file_el, bin_el])
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=uia)
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act("a1", "drag", x=20, y=20, x2=420, y2=20)
    assert "Recycle Bin" in asked[-1]["question"] and "press" not in [c[0] for c in backend.calls]


def test_a_drag_to_an_unidentified_place_asks_and_a_labelled_drag_does_not() -> None:
    clock = Clock()
    a = Element("Layer 1", "ListItemControl", rect=(0, 0, 40, 40))
    b = Element("Layer 2", "ListItemControl", rect=(0, 50, 40, 90))
    uia = FakeUIA(at=lambda x, y: a if y < 45 else (b if y < 100 else None))
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=uia)
    ctl.request("a1", "Claude")
    ctl.act("a1", "drag", x=20, y=20, x2=20, y2=70)  # both ends have a name: no question
    assert len(asked) == 1
    with pytest.raises(ControlRefused):
        ctl.act("a1", "drag", x=20, y=20, x2=20, y2=500)
    assert "(20, 500)" in asked[-1]["question"]


def test_enter_and_shortcuts_without_a_known_focus_ask_but_typing_and_moving_do_not() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Deny", "Deny"], uia=NoUIA())
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=5, y=5)
    ctl.act("a1", "scroll", dy=-3)
    ctl.act("a1", "type_text", text="hello")  # Linux: no UI Automation, typing itself is allowed
    assert len(asked) == 1
    with pytest.raises(ControlRefused):
        ctl.act("a1", "press_keys", keys="enter")
    assert asked[-1]["question"] == "Claude wants to press Enter"
    with pytest.raises(ControlRefused):
        ctl.act("a1", "press_keys", keys="ctrl+s")
    assert "ctrl+s" in asked[-1]["question"]
    ctl2, _, asked2 = _control(clock, ["Allow"])  # focus on a labelled text box: no questions
    ctl2.request("a1", "Claude")
    ctl2.act("a1", "press_keys", keys="ctrl+a")
    ctl2.act("a1", "type_text", text="hi\nthere")
    ctl2.act("a1", "click", x=3, y=3)  # on "Canvas"
    assert len(asked2) == 1


# ------------------------------------------------------------- 5. no picture without a visible notice
def _core(tmp_path: Path, **kw: Any) -> Any:
    from away import config as C
    from away.agentlink.host import build_core
    from away.agentlink.prompts import NullUI

    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    c = build_core(
        tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=NullUI(), **kw
    )
    c.prompts.polite = lambda: True
    c.cfg_obj = cfg  # type: ignore[attr-defined]
    return c


def _broken(_text: Any = None, _on: Any = None) -> None:
    raise RuntimeError("the notice window could not start")


def test_no_camera_still_when_the_notice_cannot_be_shown(tmp_path: Path) -> None:
    import numpy as np

    from away.agentlink.service import NotAvailable

    core = _core(tmp_path, flash=_broken, camera_frame=lambda _c: np.zeros((480, 640, 3), dtype=np.uint8))
    a, _ = core.registry.issue("Agent", ["read", "snapshots"], 3600)
    seq = core.events.seq
    with pytest.raises(NotAvailable, match="notice"):
        core.service.look(a)
    assert not [e for e in core.events.since(seq) if e["topic"] == "look.taken"]


def test_no_screenshot_when_the_notice_cannot_be_shown(tmp_path: Path) -> None:
    from PIL import Image

    from away.agentlink.service import NotAvailable

    core = _core(tmp_path, flash=_broken, screen_grab=lambda: Image.new("RGB", (800, 600)))
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    with pytest.raises(NotAvailable, match="notice"):
        core.service.get_screen(a)
    with pytest.raises(NotAvailable, match="notice"):
        core.service.get_screen(a, quiet=True)  # "quiet" is not a way around the notice


def test_quiet_screenshots_need_the_live_view_badge(tmp_path: Path) -> None:
    from PIL import Image

    flashes: list[str] = []
    core = _core(tmp_path, flash=flashes.append, screen_grab=lambda: Image.new("RGB", (800, 600)))
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    core.service.get_screen(a, quiet=True)
    assert len(flashes) == 1  # not watching: an ordinary screenshot with its notice
    core.service.screen_watch(a, True)
    core.service.get_screen(a, quiet=True)
    assert len(flashes) == 1  # the live view's permanent badge is the notice


def test_live_view_is_refused_and_unregistered_when_its_badge_fails(tmp_path: Path) -> None:
    from away.agentlink.service import NotAvailable

    core = _core(tmp_path, watching=_broken)
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    with pytest.raises(NotAvailable, match="badge"):
        core.service.screen_watch(a, True, scale=0.5)
    assert a.id not in core.service._watchers and a.id not in core.service._live_views


def test_host_reports_a_notice_that_could_not_be_shown(tmp_path: Path) -> None:
    from away.agentlink.host import AgentLinkHost

    class UI:
        def __init__(self, ok: bool) -> None:
            self.ok, self.shown = ok, []

        def flash(self, text: str) -> None:
            if not self.ok:
                raise RuntimeError("no display")
            self.shown.append(text)

    good = UI(True)
    AgentLinkHost(None, data_dir=tmp_path / "a", ui=good, headless=True)._flash("Claude looked at your screen")
    assert good.shown
    with pytest.raises(RuntimeError):
        AgentLinkHost(None, data_dir=tmp_path / "b", ui=UI(False), headless=True)._flash("x")
    with pytest.raises(RuntimeError):
        AgentLinkHost(None, data_dir=tmp_path / "c", ui=object(), headless=True)._watching("Claude", True)


def test_the_notice_window_launcher_reports_a_failed_start(tmp_path: Path) -> None:
    from away.agentlink.prompts import SubprocessUI

    ui = SubprocessUI(tmp_path / "ui")
    ui._spawn_argv = lambda argv: None  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        ui.flash("Claude looked at your screen")
    with pytest.raises(RuntimeError):
        ui.overlay("Claude", tmp_path / "stop", mode="watch")


# ------------------------------------------------------------- 6. request body size limit
def _api(tmp_path: Path) -> Any:
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    from away.agentlink import api

    core = _core(tmp_path)
    gate = api.Gate(core.service, core.registry, local_key="k" * 43, rate=(100.0, 200))
    calls: list[str] = []
    orig = api.dispatch

    def spy(service: Any, agent: Any, tool: str, kwargs: dict[str, Any]) -> Any:
        calls.append(tool)
        return orig(service, agent, tool, kwargs)

    app = api.create_app(core.service, gate)
    a, token = core.registry.issue("Agent", ["read", "tune"], 3600)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    return client, token, calls, spy, gate, core


def test_an_oversized_json_body_is_refused_with_413_before_parsing(tmp_path: Path, monkeypatch: Any) -> None:
    from away.agentlink import api

    client, token, calls, spy, _gate, _core_ = _api(tmp_path)
    monkeypatch.setattr(api, "dispatch", spy)
    parsed: list[int] = []
    real_loads = api.json.loads

    def loads(raw: Any, *a: Any, **k: Any) -> Any:
        parsed.append(len(raw))
        return real_loads(raw, *a, **k)

    monkeypatch.setattr(api.json, "loads", loads)
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    big = b'{"text": "' + b"x" * (api.MAX_JSON_BODY + 10) + b'"}'
    for path in ("/v1/notify", "/v1/rpc/notify_user", "/v1/pair"):
        r = client.post(path, content=big, headers=h)
        assert r.status_code == 413, path
        assert not [n for n in parsed if n > api.MAX_JSON_BODY]  # never parsed
        assert "too large" in r.json()["error"]
    assert calls == []


def test_a_chunked_body_without_content_length_cannot_get_past_the_limit(tmp_path: Path, monkeypatch: Any) -> None:
    from away.agentlink import api

    client, token, calls, spy, _gate, _core_ = _api(tmp_path)
    monkeypatch.setattr(api, "dispatch", spy)

    def chunks() -> Any:
        yield b'{"text": "'
        for _ in range(40):
            yield b"y" * 4096
        yield b'"}'

    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = client.post("/v1/rpc/notify_user", content=chunks(), headers=h)
    assert r.status_code == 413 and calls == []


def test_a_body_bigger_than_its_content_length_is_cut_off_at_the_asgi_level(tmp_path: Path) -> None:
    import asyncio

    from away.agentlink import api

    core = _core(tmp_path)
    gate = api.Gate(core.service, core.registry, local_key="k" * 43, rate=(100.0, 200))
    reached: list[int] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached.append(1)

    a, token = core.registry.issue("Agent", ["read"], 3600)
    sent: list[dict[str, Any]] = []
    pieces = [b"{" + b" " * 30_000] * 4 + [b"}"]

    async def receive() -> dict[str, Any]:
        body = pieces.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(pieces)}

    async def send(msg: dict[str, Any]) -> None:
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/rpc/notify_user",
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"content-type", b"application/json"),
            (b"content-length", b"10"),  # a lie
            (b"authorization", f"Bearer {token}".encode()),
        ],
        "client": ("127.0.0.1", 5000),
    }
    asyncio.run(api.AuthMiddleware(inner, gate)(scope, receive, send))
    assert sent[0]["status"] == 413 and reached == []


def test_normal_bodies_and_mcp_sized_bodies_still_work(tmp_path: Path) -> None:
    from away.agentlink import api

    client, token, *_ = _api(tmp_path)
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = client.post("/v1/preferences", content=b'{"text": "' + b"z" * 500 + b'"}', headers=h)
    assert r.status_code == 200
    assert api.MAX_JSON_BODY == 64 * 1024 and api.MAX_JSON_BODY <= api.MAX_MCP_BODY <= 1024 * 1024
    assert api.body_limit("/mcp") == api.MAX_MCP_BODY and api.body_limit("/c/abcdefgh") == api.MAX_MCP_BODY
    assert api.body_limit("/v1/rpc/act") == api.MAX_JSON_BODY


# ------------------------------------------------------------- 7. sharing logs is its own permission; secrets hidden
def test_sharing_logs_and_recordings_is_a_separate_permission_off_by_default() -> None:
    from away.agentlink import perms

    p = perms.BY_SCOPE[perms.SHARE_LOGS]
    assert p.label == "Share my logs and recordings" and p.default is False
    assert perms.SHARE_LOGS not in perms.DEFAULT
    assert perms.BY_SCOPE[perms.DIAGNOSE].label == "Run camera and speed checks"
    assert perms.SHARE_LOGS in perms.normalize("read,share_logs")


def test_camera_checks_do_not_let_an_agent_take_logs_or_recordings(tmp_path: Path) -> None:
    from away.agentlink.service import Forbidden

    core = _core(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "away.log").write_text("hello\n", encoding="utf-8")
    a, _ = core.registry.issue("Agent", ["read", "diagnose"], 3600)
    svc = core.service
    for call in (
        lambda: svc.get_logs(a),
        lambda: svc.get_bundle(a),
        lambda: svc.list_files(a),
        lambda: svc.read_file(a, "logs/away.log"),
        lambda: svc.record_session(a, seconds=1),
    ):
        with pytest.raises(Forbidden, match="Share my logs and recordings"):
            call()
    with pytest.raises(ValueError):
        svc.get_job(a)  # camera checks still work (here: a missing id, not a permission error)


def test_existing_agents_are_not_silently_given_the_new_permission(tmp_path: Path) -> None:
    import json

    from away.agentlink import auth, perms

    path = tmp_path / "agents.json"
    old = {
        "id": "abc",
        "name": "Claude",
        "kind": "local",
        "scopes": ["read", "diagnose"],
        "token_hash": "0" * 64,
        "created": time.time(),
        "expires": time.time() + 3600,
    }
    path.write_text(json.dumps({"agents": [old]}), encoding="utf-8")
    reg = auth.AgentRegistry(path)
    a = reg.get("abc")
    assert a is not None and a.has(perms.DIAGNOSE) and not a.has(perms.SHARE_LOGS)


SECRETS = (
    "paway_Q2hlY2tUaGlzVG9rZW5JdElzTG9uZ0Vub3VnaDEyMw",
    "pa_dGhpcy1pcy1hLWxlZ2FjeS10b2tlbi0xMjM0NTY3OA",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXNpZ25hdHVyZQ",
    "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
    "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    "hunter2-correct-horse",
    "s3cr3t-api-value-123",
    "fernando@example.com",
)


def _secret_log(home: str) -> str:
    return "\n".join(
        [
            f"2026-10-14 agent connected with token {SECRETS[0]}",
            f"legacy token {SECRETS[1]}",
            f"GET /v1/status Authorization: Bearer {SECRETS[2]}",
            f"openai api_key={SECRETS[3]}",
            f"github said: bad credentials {SECRETS[4]}",
            f"login password: {SECRETS[5]}",
            f'{{"apiKey": "{SECRETS[6]}"}}',
            f"signed in as {SECRETS[7]}",
            f"loaded {home}/AppData/Roaming/away/config.yaml",
            "camera 0 opened at 30 fps",
        ]
    )


def test_bundle_and_get_logs_hide_tokens_keys_passwords_emails_and_the_home_folder(tmp_path: Path) -> None:
    import zipfile

    from away.agentlink import bundle

    home = str(Path.home())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "away.log").write_text(_secret_log(home), encoding="utf-8")
    (logs / "crash.log").write_text(_secret_log(home), encoding="utf-8")
    core = _core(tmp_path)
    a, _ = core.registry.issue("Agent", ["read", "share_logs"], 3600)
    got = core.service.get_logs(a, lines=50)
    text = "\n".join(got["app_log"] + got["crash_log"])
    assert "camera 0 opened at 30 fps" in text  # still useful
    for s in (*SECRETS, home):
        assert s not in text, s
    path = bundle.build(tmp_path, core.cfg_obj, {}, None, note="test")
    with zipfile.ZipFile(path) as z:
        blob = "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.startswith("logs/"))
    assert "camera 0 opened at 30 fps" in blob
    for s in (*SECRETS, home):
        assert s not in blob, s


def test_action_log_shows_an_agent_only_its_own_actions(tmp_path: Path) -> None:
    core = _core(tmp_path)
    ctl = core.control
    ctl.ask = lambda *a, **k: {"answer": "Allow"}
    a, _ = core.registry.issue("Claude", ["read", "control"], 3600)
    b, _ = core.registry.issue("Gemini", ["read", "control"], 3600)
    ctl.request(a.id, a.name)
    ctl.act(a.id, "move", x=1, y=1)
    ctl.stop()
    ctl.request(b.id, b.name)
    ctl.act(b.id, "scroll", dy=-1)
    assert [e["agent"] for e in core.service.action_log(a)] == ["Claude"]
    assert [e["agent"] for e in core.service.action_log(b)] == ["Gemini"]
    core.cfg_obj.agent_link.share_control_history = True  # the person allowed agents to see the whole history
    assert {e["agent"] for e in core.service.action_log(a)} == {"Claude", "Gemini"}


# ------------------------------------------------------------- 8. taking a permission away ends what it allowed
def _host(tmp_path: Path) -> Any:
    from away.agentlink.host import AgentLinkHost
    from away.agentlink.prompts import NullUI

    class UI(NullUI):
        def __init__(self) -> None:
            super().__init__()
            self.overlays: list[tuple[str, str]] = []

        def flash(self, text: str) -> None:
            pass

        def overlay(self, name: str, stop_file: Path, mode: str = "control") -> None:
            self.overlays.append((name, mode))

    ui = UI()
    h = AgentLinkHost(None, data_dir=tmp_path, ui=ui, headless=True)
    h.core.control.ask = lambda *a, **k: {"answer": "Allow"}
    offs: list[int] = []
    orig_off = h.core.control.hooks.overlay_off

    def off() -> None:
        offs.append(1)
        orig_off()

    h.core.control.hooks.overlay_off = off
    a, _ = h.core.registry.issue("Claude", ["read", "control", "screen"], 3600)
    h.core.control.request(a.id, a.name)
    h.core.service.screen_watch(a, True, scale=0.5)
    h.core.control.views[a.id] = (0.0, 0.0, 0.5)
    assert h.core.control.status()["active"] and h._watch_stop is not None
    return h, a, offs


def test_removing_control_ends_the_session_and_turns_the_border_off(tmp_path: Path) -> None:
    h, a, offs = _host(tmp_path)
    h.handle({"action": "set_scopes", "agent_id": a.id, "scopes": ["read", "screen"]})
    assert h.core.control.status() == {"active": False} and offs
    with pytest.raises(ControlRefused):
        h.core.control.act(a.id, "move", x=1, y=1)
    assert a.id in h.core.service._watchers  # the screen permission is still on: its live view stays


def test_removing_screen_stops_the_live_view_and_forgets_the_view(tmp_path: Path) -> None:
    h, a, _ = _host(tmp_path)
    h.handle({"action": "set_scopes", "agent_id": a.id, "scopes": ["read", "control"]})
    assert a.id not in h.core.service._watchers and a.id not in h.core.service._live_views
    assert h._watch_stop is None  # the badge is gone
    assert a.id not in h.core.control.views
    assert h.core.control.status()["active"]  # control is still allowed


@pytest.mark.parametrize("how", ["revoke", "disconnect", "disconnect_all"])
def test_revoke_and_disconnect_end_control_and_the_live_view(tmp_path: Path, how: str) -> None:
    h, a, offs = _host(tmp_path)
    if how == "revoke":
        h.handle({"action": "revoke", "agent_id": a.id})
    elif how == "disconnect":
        h.core.service.disconnect(a)
    else:
        h.handle({"action": "disconnect_all"})
    assert h.core.control.status() == {"active": False} and offs
    assert a.id not in h.core.service._watchers and h._watch_stop is None
    assert a.id not in h.core.control.views


def test_removing_control_mid_typing_stops_and_lets_go_of_the_keys(tmp_path: Path) -> None:
    h, a, _ = _host(tmp_path)
    backend = BlockingBackend("type_text", 2)
    h.core.control.hooks.backend = backend
    h.core.control.hooks.uia = FakeUIA(focused=TEXT_BOX)
    t, out = _run_in_thread(lambda: h.core.control.act(a.id, "type_text", text="abcdefgh" * 20))
    assert backend.reached.wait(5.0)
    h.handle({"action": "set_scopes", "agent_id": a.id, "scopes": ["read"]})
    backend.go.set()
    t.join(5.0)
    assert isinstance(out.get("error"), ControlRefused)
    assert [c[0] for c in backend.calls].count("type_text") == 2 and backend.calls[-1][0] == "release_modifiers"


def test_the_settings_window_sends_the_change_to_the_host(tmp_path: Path) -> None:
    from away.agentlink.gui_model import AgentLinkModel

    m = AgentLinkModel(data_dir=tmp_path, config_path=tmp_path / "config.yaml")
    sent: list[tuple[str, dict[str, Any]]] = []
    m.send = lambda action, **kw: sent.append((action, kw))  # type: ignore[method-assign]
    m.set_scopes("abc", ["read", "screen"])
    assert sent == [("set_scopes", {"agent_id": "abc", "scopes": ["read", "screen"]})]


# ------------------------------------------------------------- 9. the secure-desktop check fails closed on Windows
def _raises() -> bool:
    raise OSError("OpenInputDesktop failed")


def test_windows_refuses_to_act_when_the_security_prompt_check_fails() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, secure=_raises, platform="win32")
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="security prompt"):
        ctl.act("a1", "move", x=1, y=1)
    with pytest.raises(ControlRefused, match="security prompt"):
        ctl.undo_last("a1")
    assert backend.calls == []


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_other_systems_have_no_secure_desktop_to_check(platform: str) -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, secure=_raises, platform=platform)
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=1, y=1)  # explicit: no secure desktop exists there, so a failing check is not a stop
    assert backend.calls == [("move", (1, 1))]


def test_the_windows_detector_reports_its_own_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    from away.agentlink import desktop

    class Broken:
        @property
        def user32(self) -> Any:
            raise OSError("access denied")

    monkeypatch.setattr(desktop.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", Broken(), raising=False)
    with pytest.raises(desktop.SecureDesktopUnknown):
        desktop.secure_desktop_active()
