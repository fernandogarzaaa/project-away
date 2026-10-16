"""Proposal #22 "who looked at me": a "Recently" capture history.

Camera/screen captures flash a transient on-screen notice, but there was no history: the person could not
later check who looked through their camera. Now every capture that actually showed its notice is logged
("{name} looked through your camera") and rendered in the Agent Link tab's "Recently" feed.

Each test here FAILS on the unfixed code (proved with `git stash`): the service has no capture_history,
the model has no capture_lines, and state.json has no "captures" key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from away import config as C
from away.agentlink.gui_model import AgentLinkModel
from away.agentlink.host import build_core
from away.agentlink.prompts import NullUI
from away.agentlink.service import NotAvailable


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


def test_look_logs_who_looked_through_your_camera(core) -> None:
    import numpy as np

    a, _ = core.registry.issue("Claude", ["read", "snapshots"], 3600)
    core.frame["img"] = np.zeros((480, 640, 3), dtype=np.uint8)
    core.service.look(a)
    feed = core.service.capture_feed()
    assert len(feed) == 1
    assert feed[0]["text"] == "Claude looked through your camera"
    assert abs(feed[0]["t"] - time.time()) < 60


def test_get_screen_logs_who_looked_at_your_screen(core) -> None:
    from PIL import Image

    core.service.screen_grab = lambda: Image.new("RGB", (1920, 1080))
    a, _ = core.registry.issue("Claude", ["read", "screen"], 3600)
    core.service.get_screen(a)
    feed = core.service.capture_feed()
    assert len(feed) == 1
    assert feed[0]["text"] == "Claude looked at your screen"


def test_refused_capture_is_not_logged(core) -> None:
    """A capture whose notice could not be shown is refused by the safety rule - and must never
    appear in the history either (no indicator means no capture)."""
    import numpy as np

    a, _ = core.registry.issue("Claude", ["read", "snapshots"], 3600)
    core.frame["img"] = np.zeros((480, 640, 3), dtype=np.uint8)
    core.service.flash = lambda _t: False  # the notice could not be shown
    with pytest.raises(NotAvailable):
        core.service.look(a)
    assert core.service.capture_feed() == []


def test_failed_image_creation_is_not_logged(core, monkeypatch) -> None:
    """CodeRabbit: _flash runs before to_jpeg; if the image operation fails the history must not claim the
    agent looked, even though the transient notice already showed."""
    import numpy as np

    from away.agentlink import desktop

    a, _ = core.registry.issue("Claude", ["read", "snapshots"], 3600)
    core.frame["img"] = np.zeros((480, 640, 3), dtype=np.uint8)

    def boom(_frame: object, _width: int, _quality: int) -> object:
        raise RuntimeError("jpeg exploded")

    monkeypatch.setattr(desktop, "to_jpeg", boom)
    with pytest.raises(RuntimeError):
        core.service.look(a)
    assert core.flashes and "camera" in core.flashes[-1]  # the notice still showed first (the precondition)
    assert core.service.capture_feed() == []


def test_badge_backed_screen_capture_is_logged_without_a_second_notice(core) -> None:
    """CodeRabbit: get_screen(quiet=True) for an agent with the live-view badge skips _flash; the screenshot
    must still reach Recently (the badge is the visible indicator), without a second notice."""
    from PIL import Image

    core.service.screen_grab = lambda: Image.new("RGB", (1920, 1080))
    a, _ = core.registry.issue("Claude", ["read", "screen"], 3600)
    core.service.screen_watch(a, on=True)
    n = len(core.flashes)
    core.service.get_screen(a, quiet=True)
    assert len(core.flashes) == n
    feed = core.service.capture_feed()
    assert len(feed) == 1 and feed[0]["text"] == "Claude looked at your screen"


def test_user_capture_is_logged(core) -> None:
    core.service.user_capture("camera", b"jpeg", {})
    feed = core.service.capture_feed()
    assert len(feed) == 1
    assert feed[0]["text"] == "You sent a camera picture to your agents"


def test_feed_is_newest_first_and_bounded(core) -> None:
    for _ in range(25):
        core.service.user_capture("screen", b"jpeg", {})
    feed = core.service.capture_feed(10)
    assert len(feed) == 10
    assert feed[0]["t"] >= feed[-1]["t"]  # newest first
    assert len(core.service.capture_history) == 20  # the history itself is capped


def test_capture_lines_plain_words() -> None:
    t = time.mktime(time.strptime("2026-10-16 00:33", "%Y-%m-%d %H:%M"))
    lines = AgentLinkModel.capture_lines({"captures": [{"text": "Claude looked through your camera", "t": t}]})
    assert lines == ["Claude looked through your camera \u2014 12:33 AM"]


def test_capture_lines_no_leading_zero() -> None:
    t = time.mktime(time.strptime("2026-10-16 01:05", "%Y-%m-%d %H:%M"))
    lines = AgentLinkModel.capture_lines({"captures": [{"text": "Claude looked at your screen", "t": t}]})
    assert lines == ["Claude looked at your screen \u2014 1:05 AM"]


def test_capture_lines_empty_state() -> None:
    lines = AgentLinkModel.capture_lines({})
    assert len(lines) == 1
    assert "Nothing yet" in lines[0] and "look" in lines[0]


def test_host_state_carries_capture_feed(tmp_path: Path) -> None:
    """Settings > Agent Link reads state.json: it must carry the feed for the "Recently" section."""
    from away.agentlink.host import AgentLinkHost

    h = AgentLinkHost(None, data_dir=tmp_path, ui=NullUI(), headless=True)
    h.core.service.user_capture("camera", b"jpeg", {})
    h.write_state()
    state = json.loads((tmp_path / "agentlink" / "state.json").read_text())
    assert len(state["captures"]) == 1
    assert state["captures"][0]["text"] == "You sent a camera picture to your agents"
    h.stop()
