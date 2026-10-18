"""Agent Link (v0.1.20): auth, pairing, permissions, safety rules, prompts, changes, replay, detectors."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import ClassVar

import pytest

from away import config as C
from away.agentlink import auth, handsigns, perms, safepaths
from away.agentlink.changes import ChangeManager
from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element, best_match
from away.agentlink.host import build_core
from away.agentlink.prompts import NullUI, PromptManager, voice_answer
from away.agentlink.ratelimit import FailureLockout, TokenBucket
from away.agentlink.registry import ChangeError, Op
from away.agentlink.service import Forbidden, NotAvailable
from away.control.mouse import RecordingBackend
from away.hands.synthetic import make_hand, pose


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ------------------------------------------------------------------------------------------------ auth
def test_token_is_hashed_compared_in_constant_time_and_expires(tmp_path: Path) -> None:
    clock = Clock()
    reg = auth.AgentRegistry(tmp_path / "agents.json", clock=clock)
    agent, token = reg.issue("Claude", ["read", "tune"], ttl_s=60)
    assert token.startswith("pa_") and len(token) > 40
    saved = (tmp_path / "agents.json").read_text()
    assert token not in saved and auth.digest(token) in saved  # only the hash is stored
    assert reg.authenticate(token) is agent
    assert reg.authenticate(token + "x") is None
    assert reg.authenticate("") is None and reg.authenticate(None) is None
    clock.advance(61)
    assert reg.authenticate(token) is None  # expired


def test_revoke_and_reload(tmp_path: Path) -> None:
    reg = auth.AgentRegistry(tmp_path / "agents.json")
    a, tok = reg.issue("Cursor", ["read"], 3600, kind="local")
    b, tok2 = reg.issue("ChatGPT", ["read"], 3600)
    again = auth.AgentRegistry(tmp_path / "agents.json")
    assert again.authenticate(tok) is not None
    assert reg.revoke(a.id) and reg.authenticate(tok) is None and reg.authenticate(tok2) is not None
    assert reg.revoke_all() == 1 and reg.authenticate(tok2) is None


def test_same_uses_compare_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    import hmac

    real = hmac.compare_digest
    monkeypatch.setattr(auth.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    assert auth.same("abc", "abc") and not auth.same("abc", "abd")
    assert len(calls) == 2


def test_pairing_is_single_use_expires_and_needs_approval(tmp_path: Path) -> None:
    clock = Clock()
    reg = auth.AgentRegistry(None, clock=clock)
    p = reg.new_pairing(["read", "screen"], ttl_s=600)
    assert len(p.pin) == 6 and p.pin.isdigit() and len(p.code) >= 10
    assert reg.find_pairing(pin=p.pin) is p and reg.find_pairing(code=p.code) is p
    with pytest.raises(ValueError):
        reg.approve(p, 60)  # nobody asked yet
    assert reg.mark_pending(p, "ChatGPT")
    assert not reg.mark_pending(p, "Someone else")  # one claimant only
    agent, tok = reg.approve(p, 3600, bind_link=True)
    assert agent.scopes == ["read", "screen"]
    assert reg.find_pairing(code=p.code) is None  # used
    assert reg.authenticate_link(p.code) is agent  # the connector keeps using its URL
    with pytest.raises(ValueError):
        reg.approve(p, 60)
    p2 = reg.new_pairing(["read"], ttl_s=600)
    clock.advance(601)
    assert reg.find_pairing(code=p2.code) is None  # 10 minutes
    p3 = reg.new_pairing(["read"])
    reg.mark_pending(p3, "x")
    reg.deny(p3)
    assert reg.find_pairing(code=p3.code) is None  # a denial spends it too


def test_permissions_normalize() -> None:
    assert perms.normalize("tune, screen") == frozenset({"read", "tune", "screen"})
    assert perms.READ in perms.normalize([])
    with pytest.raises(ValueError):
        perms.normalize("shell")
    labels = perms.describe(["read", "control"])
    assert labels == ["See my gestures", "Let it use my computer"]
    assert not any(
        p.default for p in perms.PERMISSIONS if p.scope in (perms.SCREEN, perms.SNAPSHOTS, perms.CONTROL, perms.TUNE)
    )


def test_rate_limit_and_lockout() -> None:
    clock = Clock()
    b = TokenBucket(2.0, 3, clock)
    assert [b.allow("ip") for _ in range(4)] == [True, True, True, False]
    assert b.allow("other")
    clock.advance(0.5)
    assert b.allow("ip") and not b.allow("ip")
    lock = FailureLockout(3, 60, 30, clock)
    for _ in range(3):
        lock.fail()
    assert lock.locked()
    clock.advance(31)
    assert not lock.locked()


# ------------------------------------------------------------------------------------------- safepaths
def test_path_traversal_is_refused(tmp_path: Path) -> None:
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "diagnostics" / "bundle.zip").write_bytes(b"z")
    (tmp_path / "secret.txt").write_text("no")
    assert safepaths.resolve(tmp_path, "diagnostics/bundle.zip").read_bytes() == b"z"
    for bad in (
        "../secret.txt",
        "diagnostics/../secret.txt",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "diagnostics\\x",
        "secret.txt",
        "models/x",
        "",
        "diagnostics//bundle.zip",
        "diagnostics/./bundle.zip",
        "logs/\x00",
    ):
        with pytest.raises(safepaths.UnsafePath):
            safepaths.resolve(tmp_path, bad)
    with pytest.raises(FileNotFoundError):
        safepaths.resolve(tmp_path, "diagnostics/missing.zip")


def test_symlink_escape_is_refused(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "outside.txt").write_text("x")
    try:
        (tmp_path / "logs" / "link.log").symlink_to(tmp_path / "outside.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    with pytest.raises(safepaths.UnsafePath):
        safepaths.resolve(tmp_path, "logs/link.log")


# -------------------------------------------------------------------------------------------- handsigns
def test_hand_signs() -> None:
    assert handsigns.classify(pose("fist")) == handsigns.THUMBS_UP
    assert handsigns.classify(pose("fist", rotation=math.pi)) == handsigns.THUMBS_DOWN
    assert handsigns.classify(pose("palm")) == handsigns.OPEN_PALM
    assert handsigns.classify(pose("point")) == "1"
    assert handsigns.classify(pose("two")) == "2"
    assert handsigns.classify(make_hand({"pinky": 1.0}, thumb_fold=1.0)) == "3"
    assert handsigns.classify(pose("pinch")) is None  # a pinch is not a thumbs up
    assert handsigns.classify(None) is None
    assert handsigns.answer_for("thumbs_up", ["Allow", "Deny"]) == "Allow"
    assert handsigns.answer_for("thumbs_down", ["Allow", "Deny"]) == "Deny"
    assert handsigns.answer_for("2", ["a", "b", "c"]) == "b"
    assert handsigns.answer_for("4", ["a", "b"]) is None


def test_sign_hold_needs_a_steady_sign() -> None:
    h = handsigns.SignHold(0.6)
    assert [h.update(i * 0.1, "thumbs_up") for i in range(6)] == [None] * 6
    assert h.update(0.61, "thumbs_up") == "thumbs_up"
    assert h.update(0.7, "thumbs_up") is None  # once
    assert h.update(0.8, "2") is None
    assert h.update(1.5, "2") is None  # a gap in the frames restarts the hold


# ---------------------------------------------------------------------------------------------- prompts
def _pm(clock: Clock, polite: bool = True) -> tuple[PromptManager, NullUI]:
    ui = NullUI()
    return PromptManager(ui, polite=lambda: polite, clock=clock), ui


def test_prompt_answered_by_gesture_voice_click_and_timeout() -> None:
    clock = Clock(0.0)
    pm, ui = _pm(clock)
    p = pm.create("Send it?", ["Yes", "No"])
    pm.pump()
    assert ui.shown == [p]
    for i in range(8):
        pm.feed_sign(i * 0.1, handsigns.THUMBS_UP)
    assert p.answer == "Yes" and p.method == "gesture:thumbs_up"
    p2 = pm.create("Which?", ["Red", "Green", "Blue"])
    pm.pump()
    assert pm.feed_voice("option two") and p2.answer == "Green" and p2.method == "voice"
    p3 = pm.create("Allow?", ["Allow", "Deny"])
    pm.pump()
    ui.clicks[p3.id] = "Deny"
    pm.pump()
    assert p3.answer == "Deny" and p3.method == "click"
    p4 = pm.create("Late?", timeout_s=5)
    pm.pump()
    clock.advance(6)
    pm.pump()
    assert p4.answer is None and p4.method == "timeout"


def test_prompt_waits_for_a_polite_moment_unless_urgent() -> None:
    clock = Clock(0.0)
    polite = {"v": False}
    pm = PromptManager(NullUI(), polite=lambda: polite["v"], clock=clock)
    low = pm.create("later", urgency="normal", timeout_s=60)
    pm.pump()
    assert pm.current is None  # the user is typing / away
    urgent = pm.create("now!", urgency="high")
    pm.pump()
    assert pm.current is urgent
    pm.feed_voice("yes")
    clock.advance(31)  # normal waits at most half its timeout
    pm.pump()
    assert pm.current is low


def test_voice_answers() -> None:
    assert voice_answer("Allow", ["Allow", "Deny"]) == "Allow"
    assert voice_answer("no thanks", ["Allow", "Deny"]) == "Deny"
    assert voice_answer("the blue one", ["red", "blue one"]) == "blue one"
    assert voice_answer("hmm", ["a", "b"]) is None


# ------------------------------------------------------------------------------------- core + service
@pytest.fixture
def core(tmp_path: Path):
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    ui = NullUI()
    flashes: list[str] = []
    frame = {"img": None}
    c = build_core(
        tmp_path,
        cfg=lambda: cfg,
        settings=lambda: cfg.agent_link,
        config_path=lambda: cfgp,
        ui=ui,
        flash=flashes.append,
        camera_frame=lambda _cam: frame["img"],
        screen_grab=lambda: None,
    )
    c.prompts.polite = lambda: True
    c.ui, c.flashes, c.frame, c.cfgp, c.cfg_obj = ui, flashes, frame, cfgp, cfg  # type: ignore[attr-defined]
    return c


def answer_with(core, answer: str):
    """Answer every prompt that is shown (simulates the person clicking)."""
    import threading

    stop = threading.Event()

    def run() -> None:
        while not stop.is_set():
            cur = core.prompts.current
            if cur is not None:
                core.ui.clicks[cur.id] = answer if answer in cur.options else cur.options[0]
            time.sleep(0.01)

    threading.Thread(target=run, daemon=True).start()
    return stop


def test_scope_enforcement(core) -> None:
    a, _ = core.registry.issue("Agent", ["read"], 3600)
    svc = core.service
    assert svc.get_status(a)["agent"]["name"] == "Agent"
    assert "present" in svc.get_context(a)
    for call in (
        lambda: svc.change_settings(a, [{"key": "hands.cursor.gain", "value": 1.0}]),
        lambda: svc.look(a),
        lambda: svc.get_screen(a),
        lambda: svc.run_diagnose(a, "benchmark"),
        lambda: svc.get_bundle(a),
        lambda: svc.read_file(a, "logs/away.log"),
        lambda: svc.request_control(a),
        lambda: svc.act(a, "click", x=1, y=1),
        lambda: svc.add_preference(a, "x"),
    ):
        with pytest.raises(Forbidden):
            call()


def test_snapshot_gating_and_flash(core) -> None:
    import numpy as np

    a, _ = core.registry.issue("Agent", ["read", "snapshots"], 3600)
    with pytest.raises(NotAvailable):
        core.service.look(a)  # no camera picture
    core.frame["img"] = np.zeros((480, 640, 3), dtype=np.uint8)
    jpeg, meta = core.service.look(a)
    assert jpeg[:2] == b"\xff\xd8" and meta["scale"] == pytest.approx(0.5)
    assert core.flashes and "camera" in core.flashes[-1]  # visible every time
    import io

    from PIL import Image

    assert Image.open(io.BytesIO(jpeg)).size == (320, 240)  # low resolution only
    b, _ = core.registry.issue("Other", ["read", "screen"], 3600)
    with pytest.raises(Forbidden):
        core.service.look(b)


def test_screen_needs_its_own_scope_and_flashes(core) -> None:
    from PIL import Image

    core.service.screen_grab = lambda: Image.new("RGB", (1920, 1080))
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    n = len(core.flashes)
    jpeg, meta = core.service.get_screen(a)
    assert meta["scale"] == pytest.approx(0.5) and len(core.flashes) == n + 1
    snap_only, _ = core.registry.issue("Cam", ["read", "snapshots"], 3600)
    with pytest.raises(Forbidden):
        core.service.get_screen(snap_only)


def test_change_settings_asks_first_and_is_undoable(core) -> None:
    a, _ = core.registry.issue("Claude", ["read", "tune"], 3600)
    stop = answer_with(core, "No")
    res = core.service.change_settings(
        a,
        [{"domain": "settings", "op": "set", "key": "hands.cursor.gain", "value": 1.5}],
        why="you said the cursor is too fast",
    )
    assert res["applied"] is False and res["answer"] == "No"
    assert C.load(core.cfgp)[0].hands.cursor.gain == 2.2
    stop.set()
    stop = answer_with(core, "Yes")
    res = core.service.change_settings(
        a, [{"domain": "settings", "op": "set", "key": "hands.cursor.gain", "value": 1.5}], why="pref 1"
    )
    stop.set()
    assert res["applied"] and C.load(core.cfgp)[0].hands.cursor.gain == 1.5
    hist = core.service.change_history(a)
    assert hist[0]["who"] == "Claude" and hist[0]["why"] == "pref 1"
    core.service.undo_changes(a, "today")
    assert C.load(core.cfgp)[0].hands.cursor.gain == 2.2


def test_auto_mode_applies_only_low_risk_without_asking(core) -> None:
    core.cfg_obj.agent_link.autofix = "auto_low_risk"
    a, _ = core.registry.issue("Claude", ["read", "tune"], 3600)
    res = core.service.change_settings(a, [{"key": "hands.gestures.pinch_on", "value": 0.25}], why="misfires")
    assert res["applied"] and res["asked"] is False
    assert any("Claude changed" in t for _lvl, t in core.ui.toasts)  # and it tells the person
    stop = answer_with(core, "No")
    res = core.service.change_settings(
        a, [{"domain": "bindings", "op": "set", "key": "gesture:swipe_left", "value": "key:ctrl+z"}], why="x"
    )
    stop.set()
    assert res["asked"] is True and res["applied"] is False  # a binding is not low risk


@pytest.mark.parametrize(
    "op",
    [
        Op("settings", "set", "bindings.allow_shell", True),
        Op("settings", "set", "agent_link.autofix", "auto_low_risk"),
        Op("settings", "set", "updates.auto_install", True),
        Op("settings", "set", "models_dir", "/tmp"),
        Op("settings", "set", "hands.cursor.gain", 50),
        Op("settings", "set", "hands.cursor.mode", "teleport"),
        Op("bindings", "set", "gesture:pinch", "shell:calc"),
        Op("bindings", "set", "gesture:pinch", "launch:cmd.exe"),
        Op("bindings", "set", "hotkey:ptt", "key:f1"),
        Op("zones", "set", "hands.cursor.active_region", [0.5, 0.5, 0.52, 0.52]),
        Op("macros", "create", "x"),
        Op("profiles", "delete", "default"),
    ],
)
def test_forbidden_or_invalid_changes(tmp_path: Path, op: Op) -> None:
    cm = ChangeManager(tmp_path / "al", lambda: tmp_path / "config.yaml")
    with pytest.raises(ChangeError):
        cm.apply([op], "agent")
    assert not (tmp_path / "config.yaml").exists()  # nothing written


def test_batch_snapshot_history_and_undo(tmp_path: Path) -> None:
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cm = ChangeManager(tmp_path / "al", lambda: cfgp)
    r = cm.apply(
        [
            Op("settings", "set", "hands.cursor.gain", 1.5),
            Op("profiles", "create", "ps", {"apps": ["Photoshop.exe"]}),
            Op("profiles", "set", "ps.tuning.hands.cursor.gain", 1.0),
            Op("bindings", "set", "voice:make it so", "key:ctrl+s"),
            Op("custom_gestures", "create", "wave"),
        ],
        "Claude",
        "a1",
        "test",
    )
    assert r["snapshot"] and (tmp_path / "al" / "snapshots" / r["snapshot"]).exists()
    cfg = C.load(cfgp)[0]
    assert cfg.bindings.profiles["ps"].tuning == {"hands.cursor.gain": 1.0}
    cm.undo_batch(r["batch"])
    cfg = C.load(cfgp)[0]
    assert (
        cfg.hands.cursor.gain == 2.2
        and "ps" not in cfg.bindings.profiles
        and "wave" not in cfg.bindings.custom_gestures
    )
    assert "make it so" not in cfg.bindings.default.voice
    with pytest.raises(ChangeError):
        cm.undo_batch(r["batch"])  # nothing left


def test_preset_apply_is_undoable(core) -> None:
    core.cfg_obj.agent_link.autofix = "auto_low_risk"
    a, _ = core.registry.issue("Claude", ["read", "tune"], 3600)
    res = core.service.change_settings(a, [{"domain": "presets", "op": "apply", "key": "Precise"}], why="slower")
    assert res["applied"] and C.load(core.cfgp)[0].hands.cursor.gain < 2.2
    core.service.undo_changes(a, "last")
    assert C.load(core.cfgp)[0].hands.cursor.gain == pytest.approx(2.2)


def test_preferences(core) -> None:
    a, _ = core.registry.issue("Claude", ["read", "tune"], 3600)
    core.service.add_preference(a, "I hate accidental right-clicks")
    items = core.service.get_preferences(a)["items"]
    assert items[0].startswith("I hate accidental right-clicks") and "Claude" in items[0]
    core.service.remove_preference(a, 1)
    assert core.service.get_preferences(a)["items"] == []


def test_record_session_writes_a_replayable_file(core) -> None:
    from away.agentlink import recording, replay

    a, _ = core.registry.issue("Claude", ["read", "diagnose", "share_logs"], 3600)
    import threading

    def feed() -> None:
        time.sleep(0.2)
        for i, lm in enumerate([pose("point")] * 20):
            core.recorder.on_hand(i / 30.0, lm, "Right")

    threading.Thread(target=feed).start()
    res = core.service.record_session(a, seconds=1.0)
    assert res["frames"] == 20 and res["path"].startswith("recordings/agent/")
    header, recs = recording.read(core.service.data_dir / res["path"])
    assert header["format"] == "away-rec" and len(recs) == 20
    replay.replay(header, recs)
    raw = core.service.read_file(a, res["path"])
    assert raw[:2] == b"\x1f\x8b"


# ------------------------------------------------------------------------------------------ control
def _control(clock: Clock, answers: list[str] | None = None, uia=None, fg=None, secure=False, platform="linux"):
    backend = RecordingBackend()
    asked: list[dict] = []
    queue = list(answers or ["Allow"])

    def ask(question: str, options: list[str], **kw):
        asked.append({"question": question, "options": options, **kw})
        return {"answer": queue.pop(0) if queue else None, "method": "click"}

    allowed: set[tuple[str, str]] = set()
    hooks = ControlHooks(
        backend=backend,
        platform=platform,
        secure_desktop=lambda: secure,
        foreground=lambda: fg,
        grab=lambda: None,
        sleep=lambda _s: None,
        **({"uia": uia} if uia is not None else {}),
    )
    ctl = ComputerControl(
        hooks,
        ask,
        ttl_s=1800,
        rate_per_s=4,
        clock=clock,
        always_allowed=lambda a, c: (a, c) in allowed,
        allow_always=lambda a, c: allowed.add((a, c)),
    )
    return ctl, backend, asked


class FakeUIA:
    available = True

    def __init__(self, focused: Element | None = None, at: Element | None = None, elements=()) -> None:
        self._focused, self._at, self._els = focused, at, list(elements)

    def focused(self):
        return self._focused

    def element_at(self, x, y):
        return self._at

    def find(self, name, control_type=None):
        return best_match(self._els, name, control_type)

    def tree(self, *a):
        return None


def test_control_needs_approval_and_expires() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Deny"])
    with pytest.raises(ControlRefused):
        ctl.act("a1", "click", x=10, y=10)  # no grant
    with pytest.raises(ControlRefused):
        ctl.request("a1", "Claude")  # denied
    assert backend.calls == []
    canvas = FakeUIA(at=Element("Canvas", "PaneControl", rect=(0, 0, 99, 99)))  # a named target: no question
    ctl2, backend2, asked2 = _control(clock, ["Allow"], uia=canvas)
    st = ctl2.request("a1", "Claude", why="fill the form")
    assert st["active"] and asked2[0]["kind"] == "approve" and asked2[0]["urgency"] == "high"
    ctl2.act("a1", "click", x=10, y=20)
    assert backend2.calls == [("move", (10, 20)), ("click", (1,))]
    with pytest.raises(ControlRefused):
        ctl2.act("other", "click", x=1, y=1)  # a different agent
    clock.advance(1801)
    with pytest.raises(ControlRefused):
        ctl2.act("a1", "click", x=10, y=20)
    assert ctl2.status() == {"active": False}


def test_takeover_pauses_and_resume_needs_approval() -> None:
    clock = Clock()
    canvas = FakeUIA(at=Element("Canvas", "PaneControl", rect=(0, 0, 99, 99)))
    ctl, backend, asked = _control(clock, ["Allow", "Continue"], uia=canvas)
    ctl.request("a1", "Claude")
    assert not ctl.user_input("hand")  # the thumbs up that approved it is not a takeover (2 s grace)
    clock.advance(2.5)
    ctl.act("a1", "move", x=5, y=5)
    assert not ctl.user_input("mouse")  # the agent's own move echoing back
    clock.advance(1.0)
    assert ctl.user_input("mouse")
    with pytest.raises(ControlRefused, match="paused"):
        ctl.act("a1", "click", x=5, y=5)
    assert ctl.user_input("hand") is False  # already paused
    ctl.resume("a1")
    assert asked[-1]["question"] == "Claude stopped because you moved the mouse. Let it keep going?"
    assert asked[-1]["options"] == ["Continue", "Yes, for the rest of this session", "Stop"]
    ctl.act("a1", "click", x=5, y=5)
    clock.advance(2.5)
    assert ctl.user_input("hand")  # a hand gesture also takes over
    assert ctl.stop("you said stop") and ctl.status()["active"] is False


# ------------------------------------------------------- proposal #21: friendly resume wording


def _takeover_and_prompt(clock: Clock, source: str) -> tuple:
    """Request a grant, let the person take over with `source`, return (ctl, asked)."""
    ctl, backend, asked = _control(clock, ["Allow"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    assert ctl.user_input(source)  # the takeover pauses at once
    return ctl, asked


def test_resume_prompt_names_each_takeover_source_in_plain_words() -> None:
    expected = {
        "mouse": "Claude stopped because you moved the mouse. Let it keep going?",
        "keyboard": "Claude stopped because you used the keyboard. Let it keep going?",
        "hand": "Claude stopped because you moved your hand. Let it keep going?",
        "voice": "Claude stopped because you spoke. Let it keep going?",
        "shortcut": "Claude stopped because you used a shortcut. Let it keep going?",
    }
    for source, question in expected.items():
        clock = Clock()
        ctl, asked = _takeover_and_prompt(clock, source)
        with pytest.raises(ControlRefused, match="did not let it continue"):
            ctl.resume("a1")  # the queued answers ran out: no answer ends the grant
        assert asked[-1]["question"] == question
        assert asked[-1]["options"] == ["Continue", "Yes, for the rest of this session", "Stop"]


def test_resume_until_takeover_suppresses_reprompts_but_takeover_still_pauses() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Yes, for the rest of this session"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    assert ctl.user_input("mouse")
    prompts = len(asked)
    ctl.resume("a1")  # the person picks "Yes, for the rest of this session"
    assert len(asked) == prompts + 1
    assert ctl.grant is not None and ctl.grant.no_reprompt
    clock.advance(1.0)
    assert ctl.user_input("mouse")  # a takeover still pauses at once
    with pytest.raises(ControlRefused, match="paused"):
        ctl.act("a1", "move", x=5, y=5)  # nothing runs while it is paused
    n_before = len(asked)
    ctl.resume("a1")  # no re-prompt this session: the agent just resumes
    assert len(asked) == n_before
    ctl.act("a1", "move", x=5, y=5)
    assert ("move", (5, 5)) in backend.calls


def test_resume_continue_still_prompts_next_time() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Continue", "Continue"], uia=FakeUIA())
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    assert ctl.user_input("mouse")
    ctl.resume("a1")
    assert ctl.grant is not None and not ctl.grant.no_reprompt
    clock.advance(1.0)
    assert ctl.user_input("mouse")
    ctl.resume("a1")  # "Continue" is one-time: it prompts again
    assert asked[-1]["question"] == "Claude stopped because you moved the mouse. Let it keep going?"


def test_resume_until_takeover_is_per_session() -> None:
    clock = Clock()
    ctl, backend, asked = _control(
        clock, ["Allow", "Yes, for the rest of this session", "Allow", "Continue"], uia=FakeUIA()
    )
    ctl.request("a1", "Claude")
    clock.advance(1.0)
    assert ctl.user_input("mouse")
    ctl.resume("a1")
    clock.advance(1.0)
    ctl.stop("the person ended it")
    ctl.request("a1", "Claude")  # a new grant: the opt-in does not carry over
    assert ctl.grant is not None and not ctl.grant.no_reprompt
    clock.advance(1.0)
    assert ctl.user_input("mouse")
    prompts = len(asked)
    ctl.resume("a1")
    assert len(asked) == prompts + 1  # it prompts again on the new grant

    clock = Clock()
    pwd = FakeUIA(focused=Element("Password", "EditControl", is_password=True, rect=(0, 0, 10, 10)))
    ctl, backend, _ = _control(clock, uia=pwd)
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="password"):
        ctl.act("a1", "type_text", text="hunter2")
    with pytest.raises(ControlRefused, match="password"):
        ctl.act("a1", "press_keys", keys="a")
    assert not any(c[0] == "type_text" for c in backend.calls)
    term, b2, _ = _control(clock, uia=FakeUIA(focused=Element("x")), fg={"process": "powershell.exe", "title": "PS"})
    term.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="terminal"):
        term.act("a1", "type_text", text="dir")
    win, b3, _ = _control(clock, platform="win32")  # Windows without UI Automation: no typing at all
    win.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="UI Automation"):
        win.act("a1", "type_text", text="hello")
    nix, b4, _ = _control(clock, fg={"title": "Sign in to your account", "process": "firefox"})
    nix.request("a1", "Claude")
    with pytest.raises(ControlRefused):
        nix.act("a1", "type_text", text="me@example.com")
    sec, b5, _ = _control(clock, secure=True)
    sec.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="security prompt"):
        sec.act("a1", "click", x=1, y=1)
    assert b5.calls == []


def test_risky_click_asks_and_always_allow_sticks() -> None:
    clock = Clock()
    uia = FakeUIA(
        at=Element("Send", "ButtonControl", rect=(100, 100, 140, 120)),
        elements=[
            Element("Send", "ButtonControl", rect=(100, 100, 140, 120)),
            Element("Subject", "EditControl", rect=(0, 0, 50, 20)),
        ],
    )
    ctl, backend, asked = _control(clock, ["Allow", "Deny", "Always allow for this agent"], uia=uia)
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act("a1", "click", element="Send")
    assert asked[-1]["kind"] == "confirm" and "Always allow for this agent" in asked[-1]["options"]
    assert backend.calls == []
    ctl.act("a1", "click", element="send")  # "Always allow for this agent"
    assert backend.calls[-1] == ("click", (1,)) and backend.calls[-2] == ("move", (120, 110))
    n = len(asked)
    ctl.act("a1", "click", x=120, y=110)  # coordinates on the same risky button: remembered
    assert len(asked) == n


def test_enter_in_a_risky_context_asks() -> None:
    clock = Clock()
    uia = FakeUIA(focused=Element("Confirm purchase", "ButtonControl", rect=(0, 0, 9, 9)))
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=uia)
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused):
        ctl.act("a1", "press_keys", keys="enter")
    assert asked[-1]["question"] == "Claude wants to press Enter"


def test_blocked_combos_apps_urls_and_rate_limit() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, uia=FakeUIA(focused=Element("Body", rect=(0, 0, 9, 9))))
    ctl.request("a1", "Claude")
    for combo in ("win+r", "ctrl+alt+delete", "ctrl+shift+esc", "win+x"):
        with pytest.raises(ControlRefused, match="blocked"):
            ctl.act("a1", "press_keys", keys=combo)
    with pytest.raises(ControlRefused):
        ctl.act("a1", "open_app", name="PowerShell")
    with pytest.raises(ControlRefused):
        ctl.act("a1", "open_app", name="cmd; rm -rf /")
    with pytest.raises(ControlRefused):
        ctl.act("a1", "open_url", url="file:///etc/passwd")
    opened: list[str] = []
    ctl.hooks.open_url = opened.append
    ctl.act("a1", "open_url", url="https://example.com")
    assert opened == ["https://example.com"]
    clock.advance(10)
    results = []
    for _ in range(12):
        try:
            ctl.act("a1", "scroll", dy=-1)
            results.append(True)
        except ControlRefused:
            results.append(False)
    assert results.count(True) == 8 and not results[-1]  # burst of 2 s at 4 actions/s


def test_undo_last_action() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock, uia=FakeUIA(focused=Element("Body", rect=(0, 0, 9, 9))))
    ctl.hooks.cursor_pos = lambda: (50, 60)
    ctl.request("a1", "Claude")
    ctl.act("a1", "type_text", text="hello")
    ctl.act("a1", "move", x=300, y=300)
    ctl.undo_last("a1")
    assert backend.calls[-1] == ("move", (50, 60))
    ctl.undo_last("a1")
    assert backend.calls[-1] == ("hotkey", ("ctrl", "z"))
    with pytest.raises(ControlRefused):
        ctl.undo_last("a1")


def test_from_image_coordinates_use_the_last_screenshot_scale() -> None:
    clock = Clock()
    ctl, backend, _ = _control(clock)
    ctl.request("a1", "Claude")
    ctl.scales["a1"] = 0.5
    ctl.act("a1", "move", x=100, y=50, from_image=True)
    assert backend.calls[-1] == ("move", (200, 100))


def test_act_result_reports_the_observed_effect() -> None:
    """Proposal #3: every action result names the foreground window (title + process) and the focused
    element, so the agent sees where its input actually landed and can self-correct."""
    clock = Clock()
    uia = FakeUIA(focused=Element("OK", "ButtonControl", rect=(0, 0, 10, 10)))
    ctl, _, _ = _control(clock, uia=uia, fg={"title": "Untitled - Notepad", "process": "notepad.exe"})
    ctl.request("a1", "Claude")
    res = ctl.act("a1", "move", x=10, y=10)
    assert res["ok"] is True and res["action"] == "move"
    assert res["foreground"] == {"title": "Untitled - Notepad", "process": "notepad.exe"}
    assert res["focused"] == "OK"
    assert "log_id" in res and "detail" in res


def test_act_result_when_nothing_is_focused_or_foreground_unknown() -> None:
    clock = Clock()
    ctl, _, _ = _control(clock, fg=None)  # foreground hook returns None; no UIA element
    ctl.request("a1", "Claude")
    res = ctl.act("a1", "move", x=10, y=10)
    assert res["foreground"] == {"title": "", "process": ""}
    assert res["focused"] == ""


def test_act_result_survives_a_broken_foreground_hook() -> None:
    """Post-state capture is best effort: a failing hook must not fail an action that succeeded."""
    clock = Clock()
    uia = FakeUIA(focused=Element("OK", "ButtonControl", rect=(0, 0, 10, 10)))
    ctl, backend, _ = _control(clock, uia=uia)
    ctl.request("a1", "Claude")

    def boom() -> None:
        raise RuntimeError("no window info")

    ctl.hooks.foreground = boom  # type: ignore[method-assign]
    res = ctl.act("a1", "move", x=10, y=10)
    assert res["ok"] is True
    assert backend.calls == [("move", (10, 10))]
    assert res["foreground"] == {"title": "", "process": ""}
    assert res["focused"] == "OK"


# ---------------------------------------------------------------------------------------- detectors
def test_detectors() -> None:
    from away.agentlink.detectors import Detectors

    clock = Clock(0.0)
    trips: list[str] = []
    d = Detectors(lambda n, _d: trips.append(n), misfire_threshold=3, cooldown_s=100, clock=clock)
    for _ in range(12):
        d.tick({"0": 30.0})
        clock.advance(1.0)
    for _ in range(6):
        d.tick({"0": 8.0})
        clock.advance(1.0)
    assert trips == ["fps_drop"]
    d.on_event("camera.error", {"error": "x"})
    d.on_event("camera.error", {})
    d.on_event("cameras.degraded", {})
    assert trips[1:] == ["open_failure", "retry_loop"]
    for _ in range(3):
        d.on_event("agent.wrong", {})
    assert trips[-1] == "misfire"
    d.on_event("camera.error", {})
    assert trips.count("retry_loop") == 1  # cooldown
    d.tick({}, crash_size=10)
    d.tick({}, crash_size=50)
    assert trips[-1] == "crash"


def test_bundle_redacts_and_includes_recording(tmp_path: Path) -> None:
    import zipfile

    from away.agentlink import bundle, replay

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "away.log").write_text("hello log\n")
    cfg = C.AppConfig()
    header, recs = replay.synthetic_recording(["move"])
    p = bundle.build(
        tmp_path, cfg, {"api_key": "sk-123", "nested": {"token": "abc"}, "path": str(Path.home() / "x")}, (header, recs)
    )
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        stats = z.read("stats.json").decode()
    assert {"logs/away.log", "config-redacted.yaml", "stats.json", "last-seconds.awayrec", "README.txt"} <= set(names)
    assert "sk-123" not in stats and "abc" not in stats and str(Path.home()) not in stats


# -------------------------------------------------------------------------------------------- replay
def test_replay_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    from away.agentlink import recording, replay

    header, recs = replay.synthetic_recording()
    a, b = replay.replay(header, recs), replay.replay(header, recs)
    assert a.digest() == b.digest() and a.frames == len(recs)
    assert "gesture.click" in a.kinds() and "gesture.drag_start" in a.kinds()
    p = recording.write(tmp_path / "s.awayrec", header, recs)
    recording.write(tmp_path / "s2.awayrec", header, recs)
    assert p.read_bytes() == (tmp_path / "s2.awayrec").read_bytes()  # byte-identical files
    assert replay.replay_file(p).digest() == a.digest()


def test_recording_rejects_other_files(tmp_path: Path) -> None:
    import gzip

    from away.agentlink import recording

    p = tmp_path / "x.awayrec"
    with gzip.open(p, "wt") as fh:
        fh.write('{"format": "nope"}\n')
    with pytest.raises(ValueError):
        recording.read(p)


def test_ring_recorder_keeps_the_last_seconds() -> None:
    from away.agentlink.recording import RingRecorder

    r = RingRecorder(seconds=2.0)
    for i in range(100):
        r.on_hand(i * 0.1, pose("point"))
    snap = r.snapshot()
    assert snap[0]["t"] >= 9.9 - 2.0 - 1e-9 and len(snap) <= 21


# ---------------------------------------------------------------------------------------- config / cli
def test_agent_link_config_defaults_and_validation() -> None:
    cfg = C.AppConfig()
    assert cfg.agent_link.autofix == "ask" and cfg.agent_link.control_ttl_min == 30
    assert cfg.agent_link.remote_ttl_min == 60 and cfg.agent_link.pairing_ttl_s == 600
    for key, bad in (("autofix", "yolo"), ("screen_fps", 30.0), ("port", 70000), ("stop_hotkey", "ctrl+nonsense+")):
        d = C.to_dict(C.AppConfig())
        d["agent_link"][key] = bad
        with pytest.raises(ValueError):
            C.from_dict(d)


def test_agent_binding_kind() -> None:
    from away.bindings import parse_action

    assert str(parse_action("agent:summon")) == "agent:summon"
    with pytest.raises(ValueError):
        parse_action("agent:shell")
    assert str(C.AppConfig().bindings.resolve("voice", "hey agent look")) == "agent:look"


def test_cli_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from away import cli
    from away.agentlink import recording, replay

    header, recs = replay.synthetic_recording(["move", "pinch_click"])
    p = recording.write(tmp_path / "r.awayrec", header, recs)
    assert cli.main(["replay", str(p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["frames"] == len(recs) and out["events"].get("gesture.click") == 1


def test_clients_config_write_keeps_a_backup(tmp_path: Path) -> None:
    from away.agentlink import clients

    p = tmp_path / "claude_desktop_config.json"
    p.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"}))
    entry = clients.server_entry(["read", "tune"], command=["/opt/away/ProjectAway", "mcp"])
    backup = clients.write(p, entry, clock=lambda: 0)
    data = json.loads(p.read_text())
    assert data["theme"] == "dark" and "other" in data["mcpServers"]
    assert data["mcpServers"]["project-away"]["args"] == ["mcp", "--scopes", "read,tune"]
    assert backup is not None and json.loads(backup.read_text())["mcpServers"] == {"other": {"command": "x"}}
    assert clients.remove(p) and "project-away" not in json.loads(p.read_text())["mcpServers"]
    env = {"APPDATA": str(tmp_path / "Roaming")}
    assert (
        clients.config_path("claude_desktop", "win32", env)
        == tmp_path / "Roaming" / "Claude" / "claude_desktop_config.json"
    )
    bad = tmp_path / "bad.json"
    bad.write_text("[1]")
    with pytest.raises(ValueError):
        clients.plan(bad, entry)


def test_tunnel_url_parsing_and_pinned_checksum(tmp_path: Path) -> None:
    import io

    from away.agentlink import tunnel

    class FakeProc:
        def __init__(self, *_a, **_k) -> None:
            self.stderr = io.StringIO(
                "INF Requesting new quick Tunnel\nINF |  https://brave-cat-1234.trycloudflare.com  |\n"
            )

        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None) -> int:
            return 0

    t = tunnel.Tunnel(tmp_path / "cloudflared", popen=FakeProc)
    assert t.start(1234, timeout=5) == "https://brave-cat-1234.trycloudflare.com"

    class Resp(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {"Content-Length": "5"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with pytest.raises(tunnel.TunnelError, match="checksum"):
        tunnel.download(tmp_path, opener=lambda *_a, **_k: Resp(b"evil!"), key=("linux", "amd64"))
    assert not (tmp_path / "bin" / "cloudflared").exists()
    assert set(tunnel.ASSETS) >= {("win32", "amd64"), ("darwin", "arm64"), ("linux", "amd64")}
    assert tunnel.machine_key("win32", "AMD64") == ("win32", "amd64")


def test_gui_model_plain_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from away.agentlink import gui_model

    m = gui_model.AgentLinkModel(
        tmp_path,
        send=lambda *a, **k: "r1",
        state=lambda: {
            "agents": [{"name": "Claude", "connected": True}],
            "results": {"r1": {"ok": True, "result": {"disconnected": 1}}},
        },
        alive=lambda: {"port": 1},
    )
    assert m.status_line() == ("Connected to Claude \u2713", True)
    assert m.disconnect_all() == "Disconnected 1 agent."
    monkeypatch.setattr(gui_model.clients, "installed", lambda key, *a, **k: False)
    problem, _text, _p = m.local_plan("claude_desktop", ["read"])
    assert problem is not None and problem.text == "Claude isn't installed." and problem.fix_label == "Get it"
    off = gui_model.AgentLinkModel(tmp_path, alive=lambda: None)
    assert off.status_line()[0] == "Project Away isn't running"
    assert off.start_remote(["read"])[0].fix == "start_app"
    words = " ".join(p.label for p in perms.PERMISSIONS) + " ".join(m.health_lines({"fps": {"0": 30}}))
    for jargon in ("MCP", "tunnel", "token", "scope"):
        assert jargon.lower() not in words.lower()


def test_about_briefing(core) -> None:
    from away.agentlink.about import INSTRUCTIONS

    a, _ = core.registry.issue("Claude", ["read", "screen"], 3600)
    text = core.service.about(a)
    assert "bridge" in text and "See my screen when I allow" in text and "(not allowed) use the computer" in text
    for word in ("ask_user", "notify_user", "get_context", "takeover pause", "polite", "away://preferences"):
        assert word in INSTRUCTIONS


def test_events_hub_wait() -> None:
    import threading

    from away.agentlink.service import EventHub

    hub = EventHub()
    threading.Timer(0.1, lambda: hub.publish("summon", {"x": 1})).start()
    res = hub.wait(0, 3.0, ("summon",))
    assert res and res[0]["topic"] == "summon"
    assert hub.wait(hub.seq, 0.05) == []
