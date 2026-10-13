"""v0.1.20 guided first run: steps, skipping, progress dots, plug-in steps, framing tips, the live tutorial coach,
the camera session (fake camera), the text tour, the CLI and the Settings "Getting started" rows."""

from __future__ import annotations

import queue
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away import cli
from away import config as C
from away.app import first_run as fr
from away.app.first_run_gui import dot_style, format_left
from away.app.settings_gui import SettingsModel
from away.comfort.settings_tab import advanced_rows, tab_rows
from away.hands.gestures import GestureStateMachine
from away.hands.synthetic import pose

FPS = 30.0


@pytest.fixture(autouse=True)
def no_plugins():
    saved = list(fr._PLUGIN_STEPS)
    fr._PLUGIN_STEPS.clear()
    yield
    fr._PLUGIN_STEPS[:] = saved


def keys(steps: list[fr.Step]) -> list[str]:
    return [s.key for s in steps]


# ============================================================================== steps + plug-ins
def test_default_steps_without_gesture_studio() -> None:
    steps = fr.build_steps(zones_available=lambda: False)
    assert keys(steps) == ["welcome", "camera", "framing", "tutorial", "comfort", "gaze", "done"]
    for s in steps:
        assert s.title and s.body and s.button
        assert "config" not in s.body.lower() and "yaml" not in s.body.lower()  # plain words only


def test_zone_placeholder_appears_when_gesture_studio_is_installed() -> None:
    steps = fr.build_steps(zones_available=lambda: True)
    assert keys(steps)[3] == "zones"
    assert steps[3].run is not None and "Gesture Studio" in steps[3].button


def test_a_plugin_replaces_the_placeholder_and_adds_a_custom_gesture_offer() -> None:
    opened: list[str] = []
    fr.register_step(fr.Step("zones", "Zones", "Draw zones.", "Open", run=lambda: opened.append("z")), "framing")
    fr.register_step(fr.Step("custom_gesture", "Your own gesture", "Record one.", "Record"), "gaze")
    fr.register_step(fr.Step("custom_gesture", "Make your own gesture", "Record one.", "Record"), "gaze")
    steps = fr.build_steps(zones_available=lambda: False)
    assert keys(steps) == [
        "welcome",
        "camera",
        "framing",
        "zones",
        "tutorial",
        "comfort",
        "gaze",
        "custom_gesture",
        "done",
    ]
    assert [s.title for s in steps if s.key == "custom_gesture"] == ["Make your own gesture"]
    steps[3].run()  # type: ignore[misc]
    assert opened == ["z"]
    fr.unregister_step("custom_gesture")
    assert "custom_gesture" not in keys(fr.build_steps(zones_available=lambda: False))
    fr.register_step(fr.Step("x", "X", "x", shown=lambda: 1 / 0 > 0), "nowhere")  # a broken plug-in is dropped
    assert "x" not in keys(fr.build_steps(zones_available=lambda: False))
    assert keys(fr.build_steps(plugins=False, zones_available=lambda: False))[3] == "tutorial"


def test_discover_plugins_skips_missing_modules() -> None:
    assert fr.discover_plugins(("away.does_not_exist", "away.comfort")) == ["away.comfort"]


# ============================================================================== navigation + state
def test_every_step_is_skippable_and_the_dots_show_it(tmp_path: Path) -> None:
    flow = fr.FirstRun(fr.build_steps(zones_available=lambda: False), tmp_path / "fr.json")
    assert flow.dots()[0] == ("welcome", fr.CURRENT) and flow.position() == (1, 7)
    flow.next()
    flow.skip()  # camera
    flow.next(fr.step_result("framing", True))
    assert [s for _k, s in flow.dots()][:4] == [fr.DONE, fr.SKIPPED, fr.DONE, fr.CURRENT]
    for _ in range(3):
        flow.skip()
    assert flow.at_end and flow.step.key == "done"
    flow.skip()  # stays on the last step
    assert flow.at_end
    assert flow.summary() == "1 of 5 steps done, 4 skipped (you can do them later)"
    flow.back()
    assert flow.step.key == "gaze"
    flow.go("camera")
    flow.next(fr.DONE)  # going back and doing a skipped step counts
    assert flow.results["camera"] == fr.DONE
    flow.go("done")
    flow.finish()
    st = fr.load_state(tmp_path / "fr.json")
    assert st["seen"] and st["completed"] and st["results"]["camera"] == fr.DONE
    flow.back()
    flow.back()
    flow.save()
    assert fr.load_state(tmp_path / "fr.json")["completed"]  # a later save keeps "completed"


def test_a_later_skip_does_not_undo_a_done_step(tmp_path: Path) -> None:
    flow = fr.FirstRun(fr.build_steps(zones_available=lambda: False), tmp_path / "fr.json")
    flow.go("camera")
    flow.mark(fr.DONE)
    flow.skip()
    assert flow.results["camera"] == fr.DONE


def test_step_results() -> None:
    assert fr.step_result("camera", False) == fr.SKIPPED and fr.step_result("camera", True) == fr.DONE
    assert fr.step_result("gaze", False) == fr.SKIPPED
    assert fr.step_result("tutorial", False, 0) == fr.SKIPPED and fr.step_result("tutorial", False, 2) == fr.DONE
    assert fr.step_result("comfort", False) == fr.DONE


def test_auto_open_only_once(tmp_path: Path) -> None:
    p = tmp_path / "first_run.json"
    assert fr.should_auto_open(p)
    fr.mark_seen(p)
    assert not fr.should_auto_open(p)
    p.write_text("not json")
    assert fr.should_auto_open(p)  # a broken file never blocks the guide
    assert fr.state_file().name == "first_run.json"


def test_progress_dot_styles_and_timer_text() -> None:
    assert dot_style(fr.DONE)[0] != dot_style(fr.SKIPPED)[0]
    assert dot_style(fr.SKIPPED)[0] == "#FFFFFF"  # hollow
    assert dot_style(fr.CURRENT) != dot_style(fr.TODO)
    assert format_left(125.4) == "2:05 left" and format_left(-3) == "0:00 left"


# ============================================================================== framing
def test_framing_tips_in_plain_words() -> None:
    good_face, lm = (0.5, 0.4, 0.3), pose("palm", center=(0.5, 0.6))
    assert fr.framing_tips(good_face, lm, 120.0) == []
    assert fr.framing_tips(good_face, lm, 20.0)[0].startswith("It's a bit dark")
    assert "very bright" in fr.framing_tips(good_face, lm, 240.0)[0]
    assert fr.framing_tips((0.5, 0.4, 0.7), lm, 120.0) == ["You're a bit close. Lean back a little."]
    assert fr.framing_tips((0.5, 0.4, 0.1), lm, 120.0) == ["You're a bit far away. Come a little closer."]
    assert "middle" in fr.framing_tips((0.9, 0.4, 0.3), lm, 120.0)[0]
    assert "face" in fr.framing_tips(None, lm, 120.0)[0]
    assert fr.framing_tips(None, lm, None, face_tracking=False) == []
    assert fr.framing_tips(good_face, None, 120.0) == ["Now raise one hand so the camera can see it."]
    edge = pose("palm", center=(0.98, 0.6))
    assert fr.framing_tips(good_face, edge, 120.0) == ["Keep your whole hand inside the picture."]


def test_framing_check_needs_a_steady_good_picture() -> None:
    chk = fr.FramingCheck()
    lm = pose("palm", center=(0.5, 0.6))
    res = chk.update(0.0, (0.5, 0.4, 0.3), lm, 120.0)
    assert res.ok and not res.stable
    res = chk.update(0.3, None, lm, None, face_checked=False)  # the face is measured every few frames: kept
    assert res.ok
    res = chk.update(1.0, None, lm, None, face_checked=False)  # ...but only for FACE_KEEP_S
    assert not res.ok
    chk2 = fr.FramingCheck()
    for i in range(50):
        res = chk2.update(i / FPS, (0.5, 0.4, 0.3), lm, 120.0 if i == 0 else None, face_checked=i % 3 == 0)
    assert res.stable
    res = chk2.update(2.0, (0.5, 0.4, 0.3), None, None)
    assert not res.ok and not res.stable


def test_brightness_of() -> None:
    assert fr.brightness_of(np.full((48, 64, 3), 100, np.uint8)) == pytest.approx(100.0)
    assert fr.brightness_of(object()) is None


# ============================================================================== tutorial
def feed_sm(coach: fr.TutorialCoach, sm: GestureStateMachine, frames: list[Any], t0: float) -> float:
    """Run frames through the real gesture state machine and feed its events to the coach."""
    t = t0
    for lm in frames:
        for ev in sm.update(t, lm):
            coach.feed(ev.kind, dict(ev.data), t)
        t += 1 / FPS
    return t


def path_to(a: tuple[float, float], b: tuple[float, float], n: int = 20) -> list[np.ndarray]:
    return [pose("point", center=(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)) for i in range(n + 1)]


def target_cam(tx: float, ty: float) -> tuple[float, float]:
    """Where to put the synthetic hand so its index fingertip lands on practice target (tx, ty)."""
    x0, y0, x1, y1 = fr.REGION
    lm = pose("point", center=(0.5, 0.5))
    dx, dy = lm[8, 0] - 0.5, lm[8, 1] - 0.5
    return x0 + tx * (x1 - x0) - dx, y0 + ty * (y1 - y0) - dy


def test_full_tutorial_with_the_real_gesture_state_machine() -> None:
    coach = fr.TutorialCoach()
    sm = GestureStateMachine(C.AppConfig().hands.gestures)
    assert coach.current is not None and coach.current.key == "move"
    t, cur = 0.0, (0.5, 0.5)
    for tx, ty in fr.MOVE_TARGETS:
        nxt = target_cam(tx, ty)
        t = feed_sm(coach, sm, path_to(cur, nxt), t)
        cur = nxt
    assert coach.tasks[0].done and coach.flash == "Nice pointing!"
    assert coach.current.key == "click"  # type: ignore[union-attr]
    t = feed_sm(coach, sm, [pose("pinch_open")] * 5 + [pose("pinch")] * 3 + [pose("pinch_open")] * 5, t)
    assert coach.tasks[1].done
    t = feed_sm(coach, sm, [pose("two", center=(0.5, 0.7 - 0.02 * i)) for i in range(25)], t)
    assert coach.tasks[2].done, coach.scrolled
    t = feed_sm(coach, sm, [None] * 3 + [pose("palm")] * 25 + [pose("point")] * 10 + [pose("palm")] * 25, t)
    assert coach.tasks[3].done and coach.complete
    assert coach.score() == (4, 4) and coach.remaining(t) > 0


def test_tutorial_skip_hint_and_any_order() -> None:
    coach = fr.TutorialCoach(budget_s=10.0)
    coach.feed("click", {}, 0.0)  # clicking first still counts
    assert coach.tasks[1].done and coach.current.key == "move"  # type: ignore[union-attr]
    assert not coach.show_hint(5.0) and coach.show_hint(25.0)
    coach.skip(26.0)
    assert coach.tasks[0].skipped and coach.current.key == "scroll"  # type: ignore[union-attr]
    for _ in range(31):
        coach.feed("autoscroll", {"direction": "down", "vy": 5.0}, 27.0)
    assert coach.tasks[2].done
    coach.feed("autoscroll", {}, 28.0)  # the hold-still report is not scrolling
    coach.feed("clutch", {"paused": False}, 29.0)  # a resume without a pause first is not the task
    assert not coach.tasks[3].done
    coach.feed("clutch", {"paused": True}, 30.0)
    assert coach.tasks[3].progress == 0.5 and "again" in coach.flash
    coach.feed("palm_scroll", {"dy": 3.0}, 31.0)
    coach.feed("clutch", {"paused": False}, 32.0)
    assert coach.complete and coach.remaining(32.0) == 0.0
    assert coach.score() == (3, 4)


def test_practice_point_maps_and_clamps() -> None:
    assert fr.practice_point(0.5, 0.5) == pytest.approx((0.5, 0.5))
    assert fr.practice_point(0.0, 1.0) == (0.0, 1.0)


# ============================================================================== camera
def test_camera_check_wraps_the_probe() -> None:
    cfg = C.AppConfig()
    calls: list[Any] = []

    def probe(dev: Any, hint: Any = None) -> tuple[bool, str]:
        calls.append((dev, hint))
        return True, "camera 0 OK"

    assert fr.camera_check(cfg, probe) == (True, "camera 0 OK") and calls == [(0, None)]
    cfg.hands.cameras = [C.CameraSpec(device="HD User Facing")]

    def boom(dev: Any, hint: Any = None) -> tuple[bool, str]:
        raise RuntimeError(f"{dev} did not open")

    assert fr.camera_check(cfg, boom) == (False, "HD User Facing did not open")


class FakeTracker:
    def __init__(self, frames: list[Any]) -> None:
        self.frames = iter(frames)
        self.closed = False

    def process(self, _rgb: Any, _t: float) -> Any:
        return types.SimpleNamespace(landmarks=next(self.frames))

    def close(self) -> None:
        self.closed = True


class FakeFaces:
    def process(self, _rgb: Any, _t: float) -> list[Any]:
        return [types.SimpleNamespace(box=(0.5, 0.4, 0.3))]

    def close(self) -> None:
        pass


class FakeShare:
    def __init__(self) -> None:
        self.requests = 0
        self.released = False

    def owner(self) -> Any:
        return None

    def requested(self) -> bool:
        return False

    def request(self, _who: str) -> None:
        self.requests += 1

    def release(self) -> None:
        self.released = True


def test_camera_session_posts_frames_events_and_gives_the_camera_back() -> None:
    lms = [pose("pinch_open")] * 4 + [pose("pinch")] * 3 + [pose("pinch_open")] * 4
    tracker, share = FakeTracker(lms), FakeShare()
    out: queue.Queue[Any] = queue.Queue()

    def frames(_idx: Any, _mirror: bool) -> Any:
        for i in range(len(lms)):
            yield i / FPS, np.full((480, 640, 3), 90, np.uint8)

    fr.camera_session(
        C.AppConfig(),
        out,
        threading.Event(),
        share=share,
        frames=frames,
        tracker_factory=lambda: tracker,
        face_factory=FakeFaces,
    )
    msgs = [out.get_nowait() for _ in range(out.qsize())]
    assert len(msgs) == len(lms)
    assert msgs[0]["face_box"] == (0.5, 0.4, 0.3) and msgs[0]["face_checked"] and msgs[0]["brightness"] == 90.0
    assert msgs[0]["image"].shape == (120, 160, 3) and "image" not in msgs[1]
    assert not msgs[1]["face_checked"] and msgs[1]["brightness"] is None
    assert any(kind == "click" for m in msgs for kind, _ in m["events"])
    assert tracker.closed and share.released and share.requests >= 1


def test_camera_session_reports_a_failure_as_a_message() -> None:
    out: queue.Queue[Any] = queue.Queue()

    def frames(_idx: Any, _mirror: bool) -> Any:
        raise RuntimeError("camera 0 did not open")
        yield  # pragma: no cover

    def no_faces() -> Any:
        raise RuntimeError("no face model")

    fr.camera_session(
        C.AppConfig(),
        out,
        threading.Event(),
        share=FakeShare(),
        frames=frames,
        tracker_factory=lambda: FakeTracker([]),
        face_factory=no_faces,
    )
    msg = out.get_nowait()
    assert "error" in msg and "did not open" in msg["error"]


def test_camera_session_stops_when_asked() -> None:
    stop = threading.Event()
    out: queue.Queue[Any] = queue.Queue()

    def frames(_idx: Any, _mirror: bool) -> Any:
        i = 0
        while True:
            if i == 3:
                stop.set()
            yield i / FPS, np.zeros((8, 8, 3), np.uint8)
            i += 1

    fr.camera_session(
        C.AppConfig(),
        out,
        stop,
        share=FakeShare(),
        frames=frames,
        tracker_factory=lambda: FakeTracker([None] * 10),
        face_factory=lambda: None,
    )
    assert out.qsize() == 3
    assert out.get_nowait()["face_tracking"] is False


# ============================================================================== readiness, download, text tour
def test_tracking_ready_and_friendly_download(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    cfg.models_dir = str(tmp_path)
    assert fr.tracking_ready(cfg, find_spec=lambda m: None)[1].startswith("cv2 not installed")
    ok, msg = fr.tracking_ready(cfg, find_spec=lambda m: object())
    assert not ok and "hand model missing" in msg
    got: list[str] = []

    def dl(mf: Any, dest: Path, _out: Any) -> Path:
        got.append(mf.name)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / mf.name).write_bytes(b"x")
        return dest / mf.name

    assert fr.download_models(cfg, downloader=dl) is None and "hand_landmarker.task" in got
    assert fr.tracking_ready(cfg, find_spec=lambda m: object()) == (True, "ready")

    def bad(*_a: Any) -> Any:
        raise OSError("offline")

    assert fr.download_models(cfg, downloader=bad) == "download failed: offline"


def test_text_tour(tmp_path: Path) -> None:
    cfg = C.AppConfig()
    cfg.models_dir = str(tmp_path)
    lines: list[str] = []
    rc = fr.text_tour(
        cfg, lines.append, prober=lambda d, hint=None: (False, "camera 0 did not open"), find_spec=lambda m: object()
    )
    text = "\n".join(lines)
    assert rc == 1
    assert "We can't see your camera" in text and "A small download is missing" in text
    assert "Pinch to click" in text and "[ ] Only respond when I'm looking at the screen" in text
    assert not fr.should_auto_open()
    (tmp_path / "hand_landmarker.task").write_bytes(b"x")
    lines.clear()
    assert fr.text_tour(cfg, lines.append, prober=lambda d, hint=None: (True, "ok"), find_spec=lambda m: object()) == 0
    assert "  [ok] Your camera works!" in lines


def test_cli_tour_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(fr, "camera_check", lambda cfg, prober=None: (True, "camera 0 OK"))
    monkeypatch.setattr(fr, "tracking_ready", lambda cfg, find_spec=None: (True, "ready"))
    assert cli.main(["tour", "--text"]) == 0
    assert "Your camera works!" in capsys.readouterr().out


def test_setup_points_to_the_tour() -> None:
    from away.app import setup_wizard

    lines: list[str] = []
    setup_wizard.run(
        yes=True,
        skip_asr=True,
        skip_probe=True,
        out=lines.append,
        downloader=lambda *a, **k: Path("x"),
        profiler=lambda cfg: None,
    )
    assert any("away tour" in line for line in lines)


# ============================================================================== Settings > Getting started
def test_getting_started_rows(tmp_path: Path) -> None:
    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    rows = tab_rows(m.cfg)
    assert [r[1] for r in rows][:2] == [
        "Only respond when I'm looking at the screen",
        "Switch to head control when my arm is tired",
    ]
    assert [r[3] for r in rows] == [False, False, True, True]
    adv = advanced_rows(m)
    assert [t.path for t in adv][0] == "attention.min_hand_height" and len(adv) == 15
    assert all(t.title and "Higher" in t.help for t in adv)
    assert m.set_tunable("comfort.break_every_min", 45.0) is None and m.cfg.comfort.break_every_min == 45.0
    assert m.set_tunable("comfort.break_every_min", 1.0) is not None  # outside the slider range
