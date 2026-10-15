"""One action engine for every input (audit Phase 1b).

Enforcement (no module but the engine touches OS input), per-source policy, concurrency between the user and an
agent, "Stop everything" mid-macro / mid-drag / mid-dictation, the state machine and source attribution.
"""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from away.actions_engine import (
    AGENT,
    DICTATION,
    GESTURE,
    LIVE_AGENT,
    MACRO,
    PAUSED,
    RUNNING,
    SHUTDOWN,
    USER_SOURCES,
    VOICE,
    ActionCancelled,
    ActionEngine,
    ActionRefused,
    attributed,
    classify,
    policy_for,
)
from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element
from away.app.macro_service import DryRunBackend
from away.control.macro_platform import PynputMacroBackend
from away.control.macro_play import MacroPlayer
from away.control.macros import Macro, Step, Target
from away.control.mouse import MouseController, RecordingBackend
from away.hands.cursor import CursorMapper
from away.hands.gestures import GestureEvent
from away.voice.output import emit_paste, emit_typed

SRC = Path(__file__).resolve().parents[1] / "src" / "away"
ENGINE = SRC / "actions_engine.py"

# ------------------------------------------------------------------------------------------- enforcement
# Anything that sends synthetic input to the OS. Only the engine may call these.
FORBIDDEN_CALLS = {
    "Controller",  # pynput.mouse.Controller / pynput.keyboard.Controller
    "OSInputBackend",
    "PynputBackend",
    "SendInput",
    "keybd_event",
    "mouse_event",
    "SetCursorPos",
    "CGEventPost",
    "CGEventCreateMouseEvent",
    "CGEventCreateKeyboardEvent",
    "XTestFakeKeyEvent",
    "XTestFakeButtonEvent",
    "XTestFakeMotionEvent",
}
FORBIDDEN_MODULES = ("pyautogui", "pydirectinput", "keyboard", "mouse")  # top-level input-injection packages
XDOTOOL_INPUT = {"key", "keydown", "keyup", "type", "mousemove", "click", "mousedown", "mouseup"}


def input_violations(source: str, filename: str = "<src>") -> list[str]:
    """OS input calls in ``source`` (empty = clean)."""
    out: list[str] = []
    tree = ast.parse(source, filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in FORBIDDEN_CALLS:
                out.append(f"{filename}:{node.lineno} calls {name}()")
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module.split(".")[0]
            if mod in FORBIDDEN_MODULES or (
                node.module.startswith("pynput") and any(a.name == "Controller" for a in node.names)
            ):
                out.append(f"{filename}:{node.lineno} imports from {node.module}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_MODULES:
                    out.append(f"{filename}:{node.lineno} imports {a.name}")
        elif isinstance(node, ast.List | ast.Tuple):
            vals = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if "xdotool" in vals and XDOTOOL_INPUT & set(vals):
                out.append(f"{filename}:{node.lineno} runs xdotool input")
    return out


def test_no_module_but_the_engine_sends_os_input() -> None:
    bad: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path == ENGINE:
            continue
        bad += input_violations(path.read_text(encoding="utf-8"), str(path.relative_to(SRC)))
    assert not bad, "only away.actions_engine may send OS input:\n" + "\n".join(bad)


def test_the_enforcement_check_catches_violations_and_the_engine_is_the_input_owner() -> None:
    assert input_violations("from pynput.mouse import Controller\nController().position = (1, 2)\n")
    assert input_violations("from pynput import keyboard\nkeyboard.Controller().type('x')\n")
    assert input_violations("import ctypes\nctypes.windll.user32.SendInput(1, 2, 3)\n")
    assert input_violations("import pyautogui\n")
    assert input_violations("import subprocess\nsubprocess.run(['xdotool', 'type', 'hi'])\n")
    assert input_violations("from away.control.mouse import PynputBackend\nb = PynputBackend()\n")
    assert not input_violations("from pynput import keyboard\nkeyboard.Listener(on_press=print)\n")  # reading
    assert not input_violations("subprocess.run(['xdotool', 'getactivewindow', 'getwindowname'])\n")
    assert input_violations(ENGINE.read_text(encoding="utf-8"))  # the engine is where input lives


# --------------------------------------------------------------------------------------------- policy
def test_the_users_own_sources_are_never_gated_and_agents_keep_every_rule() -> None:
    def broken() -> bool:
        raise OSError("no desktop")

    eng = ActionEngine(secure_desktop=broken, platform="win32", agent_rate_per_s=1.0)
    for src in sorted(USER_SOURCES):
        for _ in range(20):  # no rate limit, no secure-desktop probe, no prompts
            eng.check(src)
        assert not policy_for(src).risk_prompts
    with pytest.raises(ActionRefused, match="could not check"):
        eng.check(AGENT, "a1")  # Windows fails closed
    ok = ActionEngine(secure_desktop=lambda: False, platform="win32", agent_rate_per_s=1.0)
    ok.check(LIVE_AGENT, "x")
    ok.check(LIVE_AGENT, "x")
    with pytest.raises(ActionRefused, match="too many"):
        ok.check(LIVE_AGENT, "x")  # the future live agent is rate limited like any agent
    up = ActionEngine(secure_desktop=lambda: True)
    with pytest.raises(ActionRefused, match="security prompt"):
        up.check(AGENT, "a1")
    up.check(GESTURE)  # the user's gestures are not blocked by the agent rule
    assert policy_for("something-new") == policy_for(AGENT)  # unknown sources fail closed
    assert classify("Send") == "risky" and classify("Notes") == "safe" and classify("") == "unknown"


# --------------------------------------------------------------------------------------------- helpers
class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class UIA:
    available = True

    def focused(self) -> Element:
        return Element("Message", "EditControl", rect=(0, 0, 9, 9))

    def element_at(self, x: int, y: int) -> Element:
        return Element("Canvas", "PaneControl")

    def find(self, *_a: Any) -> None:
        return None


def _agent(eng: ActionEngine, backend: Any, clock: Clock, sleep: Any = None) -> ComputerControl:
    hooks = ControlHooks(
        backend=backend,
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: None,
        sleep=sleep or (lambda _s: None),
        uia=UIA(),
        engine=eng,
    )
    ctl = ComputerControl(hooks, lambda *_a, **_k: {"answer": "Allow"}, rate_per_s=50, clock=clock)
    ctl.request("a1", "Claude")
    clock.advance(5)  # past the approval grace
    return ctl


# ---------------------------------------------------------------------------- user vs agent concurrency
def test_a_user_gesture_during_an_agent_drag_takes_over_and_the_gesture_is_never_blocked() -> None:
    clock = Clock()
    eng = ActionEngine(clock=clock)
    rec = RecordingBackend()
    gesture = eng.client(GESTURE, rec)
    steps = {"n": 0}

    def sleep(_s: float) -> None:
        steps["n"] += 1
        if steps["n"] == 3:
            gesture.move(5, 5)  # the person moves their hand mid-drag

    ctl = _agent(eng, rec, clock, sleep)
    with pytest.raises(ControlRefused, match="stopped part-way because you used the hand"):
        ctl.act("a1", "drag", x=0, y=0, x2=800, y2=800)
    assert ctl.status()["paused"]
    names = [c[0] for c in rec.calls]
    assert ("move", (5, 5)) in rec.calls  # the user's input went out
    assert names.count("press") == 1 and "release" in names[names.index("press") :]  # button let go
    assert names[-1] == "release_modifiers"
    assert not eng.busy()
    srcs = [e["source"] for e in eng.recent()]
    assert AGENT in srcs  # the interrupted agent action is in the one audit log


def test_threaded_user_gesture_and_agent_action_user_wins() -> None:
    clock = Clock()
    eng = ActionEngine(clock=clock)
    rec = RecordingBackend()
    started = threading.Event()

    class SlowBackend(RecordingBackend):
        def move(self, x: int, y: int) -> None:
            super().move(x, y)
            started.set()

    slow = SlowBackend()
    ctl = _agent(eng, slow, clock, sleep=time.sleep)  # waits are real (token.wait), so a stop wakes them
    err: list[BaseException] = []

    def agent() -> None:
        try:
            ctl.act("a1", "drag", x=0, y=0, x2=500, y2=500)
        except BaseException as exc:
            err.append(exc)

    th = threading.Thread(target=agent, daemon=True)
    th.start()
    assert started.wait(2.0)
    eng.client(GESTURE, rec).click(1)  # the user clicks from the gesture thread
    th.join(2.0)
    assert not th.is_alive()
    assert err and isinstance(err[0], ControlRefused)
    assert rec.calls == [("click", (1,))]
    assert "release" in [c[0] for c in slow.calls]


# ---------------------------------------------------------------------------------------- stop everything
def test_stop_everything_mid_agent_drag_ends_the_grant_and_releases_the_button() -> None:
    clock = Clock()
    eng = ActionEngine(clock=clock)
    rec = RecordingBackend()
    steps = {"n": 0}

    def sleep(_s: float) -> None:
        steps["n"] += 1
        if steps["n"] == 2:
            assert eng.inflight() == [(AGENT, "a1", "drag")]
            eng.stop_all("you held up an open palm")

    ctl = _agent(eng, rec, clock, sleep)
    with pytest.raises(ControlRefused, match="open palm"):
        ctl.act("a1", "drag", x=0, y=0, x2=100, y2=100)
    assert ctl.status() == {"active": False}  # the grant ended with it
    names = [c[0] for c in rec.calls]
    assert names.count("move") == 3 and "release" in names  # stopped after two steps, button released


def test_stop_everything_mid_gesture_drag_lets_go_of_the_button() -> None:
    eng = ActionEngine()
    rec = RecordingBackend()
    mouse = MouseController(eng.client(GESTURE, rec), CursorMapper((1920, 1080)))
    mouse.handle(GestureEvent("drag_start", 0.0, {}))
    assert rec.calls == [("press", ())]
    eng.stop_all("you pressed the stop keys")
    assert rec.calls[-1] == ("release", ())
    mouse.handle(GestureEvent("drag_end", 0.1, {}))  # the gesture's own release later is harmless
    mouse.handle(GestureEvent("click", 0.2, {}))  # and gestures keep working: stop is not a pause
    assert rec.calls[-1] == ("click", (1,))


def test_stop_everything_mid_macro_aborts_the_replay_and_releases_held_input() -> None:
    eng = ActionEngine()
    out = DryRunBackend()
    backend = PynputMacroBackend(engine=eng, backend=out)
    waits = {"n": 0}

    def sleep(ev: threading.Event, _s: float) -> bool:
        waits["n"] += 1
        if waits["n"] == 3:
            eng.stop_all("you said stop")
        return ev.is_set()

    drag = Step("drag", target=Target(10, 10), path=[(5, 5), (10, 10), (20, 20), (30, 30)], duration=1.0)
    macro = Macro("m", steps=[Step("key", keys="ctrl+c"), drag, Step("text", text="never typed")])
    player = MacroPlayer(backend, sleep=sleep)
    with eng.track(MACRO, player.abort, "m"):
        res = player.run(macro)
    assert res.stopped == "aborted" and res.steps_run == 1
    names = [c[0] for c in out.calls]
    assert names.count("press") == 1 and "release" in names[names.index("press") :]
    assert "type_text" not in names
    assert names.count("key_down") == names.count("key_up") == 2  # ctrl+c fully released
    assert {e["source"] for e in eng.recent() if e["action"] != "stop_all"} == {MACRO}


def test_stop_everything_releases_a_key_a_replay_still_holds() -> None:
    eng = ActionEngine()
    out = DryRunBackend()
    b = PynputMacroBackend(engine=eng, backend=out)
    b.key_down("shift")
    b.press("left")
    eng.stop_all("tray")
    assert ("key_up", ("shift",)) in out.calls and ("release", ("left",)) in out.calls


def test_stop_everything_mid_dictation_drops_the_rest() -> None:
    eng = ActionEngine()

    class Keys(RecordingBackend):
        def type_text(self, text: str) -> None:
            super().type_text(text)
            eng.stop_all("you said stop")  # said while the first chunk was going out

    kb = Keys()
    text = "a long dictated sentence that would take a while to type out"
    sent = emit_typed(eng, kb, text, chunk=10)
    assert sent == 10
    assert [c for c in kb.calls if c[0] == "type_text"] == [("type_text", (text[:10],))]
    plain = RecordingBackend()
    assert emit_typed(eng, plain, "next one") == len("next one")  # the next dictation types normally
    entry = next(e for e in eng.recent() if e["source"] == DICTATION)
    assert "dictated" not in entry["detail"]  # the audit log never holds the text


def test_paste_dictation_is_cancelled_before_the_keys_and_the_clipboard_comes_back() -> None:
    eng = ActionEngine()

    class Clip:
        value = "mine"

        def paste(self) -> str:
            return self.value

        def copy(self, v: str) -> None:
            self.value = v

    clip, kb = Clip(), RecordingBackend()
    assert emit_paste(eng, kb, clip, "ctrl", "hello", 0.0)
    assert kb.calls == [("hotkey", ("ctrl", "v"))] and clip.value == "mine"
    eng.pause("busy")
    assert not emit_paste(eng, kb, clip, "ctrl", "again", 0.0)
    assert len(kb.calls) == 1 and clip.value == "mine"


# ------------------------------------------------------------------------------------------ state machine
def test_pause_resume_shutdown() -> None:
    eng = ActionEngine()
    rec = RecordingBackend()
    g = eng.client(GESTURE, rec)
    g.press()
    eng.pause("you paused Project Away")
    assert eng.state == PAUSED
    assert rec.calls == [("press", ()), ("release", ())]  # pausing lets go of held input
    g.click(1)  # dropped while paused
    g.release()  # releases always go through
    assert rec.calls[-1] == ("release", ())
    with pytest.raises(ActionRefused, match="paused"):
        eng.check(GESTURE)
    eng.resume()
    assert eng.state == RUNNING
    g.click(1)
    assert rec.calls[-1] == ("click", (1,))
    eng.pause("agents only", except_sources={GESTURE})
    g.click(2)
    assert rec.calls[-1] == ("click", (2,))
    with pytest.raises(ActionCancelled), eng.begin(AGENT, "move", backend=rec) as a:
        a.step()
    eng.resume()
    keys = DryRunBackend()
    k = eng.client(MACRO, keys)
    k.key_down("shift")
    eng.shutdown()
    assert eng.state == SHUTDOWN and keys.calls[-1] == ("key_up", ("shift",))
    eng.resume()  # terminal
    k.key_down("a")
    g.click(1)
    assert eng.state == SHUTDOWN and keys.calls[-1] == ("key_up", ("shift",))
    assert rec.calls[-1] == ("click", (2,))


# ------------------------------------------------------------------------------------------- attribution
def test_every_source_is_attributed_in_the_one_audit_log() -> None:
    clock = Clock()
    eng = ActionEngine(clock=clock)
    rec = RecordingBackend()
    g = eng.client(GESTURE, rec)
    g.click(1)
    with attributed(VOICE):  # a voice command runs the same bindings through the gesture client
        g.hotkey(["ctrl", "c"])
    PynputMacroBackend(engine=eng, backend=DryRunBackend()).click("left", 1)
    emit_typed(eng, rec, "hi")
    ctl = _agent(eng, rec, clock)
    ctl.act("a1", "click", x=3, y=4)
    log = list(reversed(eng.recent()))
    pairs = [(e["source"], e["action"]) for e in log]
    assert pairs[:4] == [(GESTURE, "click"), (VOICE, "hotkey"), (MACRO, "click_button"), (DICTATION, "type_text")]
    agent = [e for e in log if e["source"] == AGENT]
    assert agent and agent[-1]["actor"] == "Claude" and agent[-1]["action"] == "click"
    assert ctl.actions()[0]["source"] == AGENT
    assert eng.recent(source=VOICE)[0]["detail"] == "ctrl+c"
    g.move(1, 1)  # pointer moves are not audited (one per frame)
    assert eng.recent()[0]["action"] != "move"


def test_a_private_engine_per_control_keeps_old_callers_working() -> None:
    hooks = ControlHooks(backend=RecordingBackend(), platform="linux", secure_desktop=lambda: False)
    ctl = ComputerControl(hooks, lambda *_a, **_k: {"answer": "Allow"})
    assert isinstance(ctl.engine, ActionEngine)
    ctl.request("a1", "A")
    assert ctl.engine.stop_all("tray") == 0 and ctl.status() == {"active": False}


# ------------------------------------------------- proposal #4: the swallowed Win key
def test_win_key_is_a_held_native_tap_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proposal #4: `press_keys keys="win"` returned ok while Start never opened. pynput's instant
    press()/release() is two SendInput events in one batch (zero dwell), which the shell discards; the
    fix holds the key natively for 80-120 ms on Windows."""
    import away.actions_engine as engmod

    monkeypatch.setattr("sys.platform", "win32")
    sent: list[tuple[int, bool]] = []
    monkeypatch.setattr(engmod, "_sendinput_key", lambda vk, down: sent.append((vk, down)))
    held: list[float] = []
    monkeypatch.setattr(engmod.time, "sleep", lambda s: held.append(s))

    b = object.__new__(engmod.OSInputBackend)
    b.key("cmd")  # what step().key("cmd") reaches for a lone Win press

    assert sent == [(0x5B, True), (0x5B, False)]  # VK_LWIN down, then up
    assert len(held) == 1 and 0.08 <= held[0] <= 0.12


def test_win_key_is_released_even_when_the_hold_is_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hold must never leave the Win key logically down (a stuck Win key hijacks the PC)."""
    import away.actions_engine as engmod

    monkeypatch.setattr("sys.platform", "win32")
    sent: list[tuple[int, bool]] = []
    monkeypatch.setattr(engmod, "_sendinput_key", lambda vk, down: sent.append((vk, down)))

    def boom(_s: float) -> None:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(engmod.time, "sleep", boom)
    b = object.__new__(engmod.OSInputBackend)
    with pytest.raises(RuntimeError, match="interrupted"):
        b.key("cmd")
    assert sent == [(0x5B, True), (0x5B, False)]


def test_win_key_keeps_the_pynput_path_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Everywhere but Windows the tap still goes through pynput's press/release (no behavior change)."""
    import away.actions_engine as engmod

    monkeypatch.setattr("sys.platform", "linux")
    pressed: list[Any] = []
    released: list[Any] = []

    class FakeKeys:
        cmd = "Key.cmd"

    class FakeK:
        def press(self, k: Any) -> None:
            pressed.append(k)

        def release(self, k: Any) -> None:
            released.append(k)

    b = object.__new__(engmod.OSInputBackend)
    b._k = FakeK()  # type: ignore[attr-defined]
    b._keys = FakeKeys()  # type: ignore[attr-defined]
    b.key("cmd")
    assert pressed == ["Key.cmd"] and released == ["Key.cmd"]


def test_win_chords_still_press_and_release_through_pynput(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a LONE Win key takes the native path: chords (Win+R) hold Win while the other key is down."""
    import away.actions_engine as engmod

    monkeypatch.setattr("sys.platform", "win32")
    pressed: list[Any] = []
    released: list[Any] = []

    class FakeKeys:
        cmd = "Key.cmd"

    class FakeK:
        def press(self, k: Any) -> None:
            pressed.append(k)

        def release(self, k: Any) -> None:
            released.append(k)

    b = object.__new__(engmod.OSInputBackend)
    b._k = FakeK()  # type: ignore[attr-defined]
    b._keys = FakeKeys()  # type: ignore[attr-defined]
    b.hotkey(["cmd", "r"])
    assert pressed == ["Key.cmd", "r"] and released == ["r", "Key.cmd"]


def test_sendinput_input_struct_matches_the_windows_abi() -> None:
    """CodeRabbit: SendInput's cbSize must be sizeof(INPUT) (40 on x64); an undersized struct makes it fail
    with ERROR_INVALID_PARAMETER, turning every Win tap into an OSError instead of opening Start."""
    import ctypes

    import away.actions_engine as engmod

    ptr = ctypes.sizeof(ctypes.c_void_p)

    def aligned(n: int, a: int) -> int:
        return (n + a - 1) // a * a

    mouse = aligned(4 * 5 + ptr, 8)
    kbd = aligned(2 + 2 + 4 + 4 + ptr, 8)
    hw = aligned(4 + 2 + 2, 4)
    union = aligned(max(mouse, kbd, hw), 8)
    assert ctypes.sizeof(engmod._INPUT_UNION) == union
    assert ctypes.sizeof(engmod._INPUT) == 4 + 4 + union  # DWORD type + alignment padding + union
    if ptr == 8:  # the 64-bit Windows cbSize SendInput checks
        assert ctypes.sizeof(engmod._INPUT) == 40
