"""Gesture Studio tab logic (no Tk): saving, actions, settings, wizard plan, plain words, Settings wiring."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from away import config as C
from away.app.settings_gui import SettingsModel
from away.control.mouse import RecordingBackend
from away.hands.synthetic import pose
from away.studio import words as W
from away.studio.controller import StudioController, WizardPlan, action_runner
from away.studio.engine import ChainRunner
from away.studio.pack import NOTHING, StudioStore
from away.studio.zones import IGNORE, ZoneMap
from studio_helpers import record_nothing, record_static


@pytest.fixture
def ctl(tmp_path: Path) -> StudioController:
    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    m.save()
    return StudioController(m, StudioStore(tmp_path / "studio"))


def test_create_gesture_saves_only_studio_parts(ctl: StudioController) -> None:
    ctl.m.cfg.hands.cursor.gain = 7.5  # an unsaved edit in another tab
    name, err = ctl.create_gesture("Thumbs Up!", "static", 1)
    assert (name, err) == ("thumbs_up", None)
    on_disk, _ = C.load(ctl.m.path)
    assert on_disk.bindings.custom_gestures == ["thumbs_up"]
    assert on_disk.studio.gestures["thumbs_up"]["hold_s"] == 0.3
    assert str(on_disk.bindings.resolve("gestures", "thumbs_up")) == "none"
    assert on_disk.hands.cursor.gain != 7.5 and ctl.m.dirty  # the other edit stays unsaved
    assert [r.status for r in ctl.rows()] == ["Needs practice"]
    assert ctl.create_gesture("thumbs up")[1] == "You already have a gesture called 'thumbs_up'"
    assert ctl.create_gesture("  ")[1] == "Type a name for your gesture first."
    assert "built-in" in (ctl.create_gesture("pinch")[1] or "")


def test_actions_from_the_plain_picker(ctl: StudioController) -> None:
    ctl.create_gesture("wave", "motion")
    assert ctl.set_action("wave", "url", "example.com") is None
    assert str(ctl.cfg.bindings.resolve("gestures", "wave")) == "launch:https://example.com"
    assert ctl.current_action("wave") == ("url", "https://example.com")
    assert ctl.set_action("wave", "key", "ctrl+shift+t") is None and ctl.current_action("wave") == (
        "key",
        "ctrl+shift+t",
    )
    assert ctl.set_action("wave", "mouse", "right_click") is None
    assert ctl.set_action("wave", "type", "") == "Fill in the box next to the action first."
    assert ctl.set_action("wave", "url", "") == "Type the website address, like example.com"
    assert "disabled for safety" in (ctl.set_action("wave", "shell", "ls") or "")
    assert ctl.set_action("wave", "none") is None and ctl.current_action("wave") == ("none", "")
    on_disk, _ = C.load(ctl.m.path)
    assert str(on_disk.bindings.resolve("gestures", "wave")) == "none"
    assert ctl.choice_args("mouse") and ctl.choice_args("key") is None and ctl.choice_args("none") == ()


def test_settings_from_the_dialog(ctl: StudioController) -> None:
    ctl.create_gesture("wave", "motion")
    assert ctl.set_settings("wave", False, 0.9, 0.0, 2.0, "wait:0.5\n\nkey:enter\n") is None
    on_disk, _ = C.load(ctl.m.path)
    assert on_disk.studio.gestures["wave"] == {
        "enabled": False, "threshold": 0.9, "hold_s": 0.0, "cooldown_s": 2.0, "then": ["wait:0.5", "key:enter"],
    }  # fmt: skip
    assert ctl.rows()[0].status == "Switched off"
    err = ctl.set_settings("wave", True, 0.9, 0.0, 2.0, "wait:forever")
    assert err is not None and err.startswith("Wait needs a number")
    assert "disabled for safety" in (ctl.set_settings("wave", True, 0.9, 0.0, 2.0, "shell:ls") or "")


def test_remove_gesture_cleans_everything(ctl: StudioController) -> None:
    ctl.create_gesture("wave", "motion")
    ctl.set_action("wave", "key", "f5")
    ctl.store.save_take("wave", [(pose("palm"), None)] * 4, "Right", 30, "s")
    assert ctl.remove_gesture("wave") is None
    on_disk, _ = C.load(ctl.m.path)
    assert on_disk.bindings.custom_gestures == [] and "wave" not in on_disk.studio.gestures
    assert "wave" not in on_disk.bindings.default.gestures and ctl.store.clip_paths() == []


def test_rows_follow_practice_and_learning(ctl: StudioController) -> None:
    rng = np.random.default_rng(0)
    ctl.create_gesture("fist_hold")
    record_static(ctl.store, "fist_hold", "fist", 3, 3, rng)
    assert ctl.rows()[0].status == "Ready to learn" and ctl.rows()[0].rounds == 3 and ctl.rows()[0].examples == 9
    assert not ctl.learn(backend="numpy").ok  # no relax round yet
    record_nothing(ctl.store, 3, rng)
    res = ctl.learn(backend="numpy")
    assert res.ok and ctl.rows()[0].status == "Ready" and ctl.last_result()["labels"] == [NOTHING, "fist_hold"]
    p, err = ctl.save_take({"label": "fist_hold", "frames": [(pose("fist"), None)] * 5, "fps": 30, "session": "x"})
    assert p is not None and err is None
    import os

    os.utime(p, (ctl.model_time() + 5, ctl.model_time() + 5))
    assert ctl.rows()[0].status == "New examples: learn again"
    assert ctl.save_take({"label": "fist_hold", "frames": [(None, None)] * 5})[1] == (
        "Your hand was not in view; try again with your hand in the picture"
    )
    assert ctl.save_negative([(pose("palm"), None)] * 5) is None and ctl.store.sessions(NOTHING) == 4
    assert ctl.delete_all_recordings() > 0 and ctl.store.clip_paths() == []


def test_zones_saved_per_camera_and_profile(ctl: StudioController) -> None:
    ctl.cfg.hands.camera_index = "HD User Facing"
    ctl.cfg.bindings.new_profile("chrome", ["chrome"])
    zm = ctl.zone_map("chrome")
    assert zm.is_open() and (zm.cols, zm.rows) == (6, 4)
    zm.set_cell(0, 0, IGNORE)
    assert ctl.save_zone_map(zm, "chrome") is None
    on_disk, _ = C.load(ctl.m.path)
    assert list(on_disk.zones.maps) == ["HD User Facing"] and list(on_disk.zones.maps["HD User Facing"]) == ["chrome"]
    assert not ctl.zone_map("chrome").is_open() and ctl.zone_map("default").is_open()
    assert ctl.save_zone_map(ZoneMap(), "chrome") is None  # all 'works here' = no entry
    assert C.load(ctl.m.path)[0].zones.maps == {}
    assert ctl.set_block_cursor(True) is None and C.load(ctl.m.path)[0].zones.block_cursor
    assert ctl.zone_profiles() == ["default", "chrome"]


def test_share_and_add_file(ctl: StudioController, tmp_path: Path) -> None:
    assert ctl.export(tmp_path / "x.zip") == "There are no gestures to share yet"
    ctl.create_gesture("wave", "motion")
    ctl.set_action("wave", "key", "f6")
    assert ctl.export(tmp_path / "p.zip") is None
    m2 = SettingsModel(C.AppConfig(), tmp_path / "c2.yaml")
    m2.save()
    other = StudioController(m2, StudioStore(tmp_path / "s2"))
    notes, err = other.import_(tmp_path / "p.zip")
    assert err is None and notes[0] == "added 1 gesture(s)"
    assert str(C.load(m2.path)[0].bindings.resolve("gestures", "wave")) == "key:f6"
    (tmp_path / "junk.zip").write_text("x")
    assert other.import_(tmp_path / "junk.zip")[1] == "This file is not a gesture pack (not a zip file)"


def test_action_runner_runs_the_action_then_the_chain(ctl: StudioController, monkeypatch) -> None:
    monkeypatch.setattr("away.studio.controller.StudioStore", lambda: ctl.store)
    ctl.create_gesture("wave", "motion")
    ctl.set_action("wave", "type", "hello")
    ctl.set_settings("wave", True, 0.8, 0.0, 1.0, "wait:0.1\nkey:enter")
    b = RecordingBackend()
    slept: list[float] = []
    run = action_runner(ctl.m, backend=b, chains=ChainRunner(sleep=slept.append, threaded=False))
    assert run("wave") is True
    assert b.calls[0] == ("type_text", ("hello",)) and b.calls[1][0] == "key" and slept == [0.1]
    ctl.set_action("wave", "none")
    assert run("wave") is False


def test_wizard_plan_rounds_relax_and_redo(tmp_path: Path) -> None:
    plan = WizardPlan("wave", "motion", rounds=3, per_round=5)
    assert len(plan.steps) == 18 and plan.total_takes == 15
    kinds = [s.kind for s in plan.steps]
    assert kinds == (["take"] * 5 + ["relax"]) * 3
    take_sessions = {s.session for s in plan.steps if s.kind == "take"}
    relax_sessions = {s.session for s in plan.steps if s.kind == "relax"}
    assert len(take_sessions) == 3 and len(relax_sessions) == 3 and not take_sessions & relax_sessions
    assert plan.steps[0].countdown == 3.0 and plan.steps[1].countdown == 1.5 and plan.steps[0].seconds == 1.5
    assert WizardPlan("x", "static").steps[0].seconds == 1.0
    assert plan.progress_text() == "Practice example 1 of 15 (round 1 of 3)"
    cmd = plan.command()
    assert cmd is not None and cmd["label"] == "wave" and cmd["tag"] == ("wizard", 0)
    for i in range(5):
        plan.accept(tmp_path / f"{i}.npz")
    assert plan.progress_text() == "Relax round 1 of 3" and plan.command()["label"] == NOTHING  # type: ignore[index]
    plan.accept(tmp_path / "relax.npz")
    assert plan.redo_last() == [tmp_path / "relax.npz", tmp_path / "4.npz"]
    assert plan.pos == 4 and plan.takes_done() == 4 and len(plan.saved) == 4
    while not plan.done:
        plan.accept(tmp_path / "x.npz")
    assert plan.command() is None and plan.progress_text() == "Done!"
    assert WizardPlan("a").redo_last() == []


# ---------------------------------------------------------------------- plain words
JARGON = re.compile(
    r"\b(confidence|model|classifier|inference|dataset|epochs?|neural|saliency|logits?|takes?|blocked|"
    r"threshold|hysteresis|tensor|softmax|onnx|tcn|mlp|train(ing)?|accuracy|holdout|held-out|fold)\b",
    re.IGNORECASE,
)


def test_user_facing_words_have_no_jargon() -> None:
    texts = [
        *W.LABELS.values(), *W.TOOLTIPS.values(), *W.WIZARD.values(), *W.STATE_WORDS.values(),
        *W.KIND_WORDS.values(), *W.HANDS_WORDS.values(), *(c[0] for c in W.ACTION_CHOICES), *W.FIXES.values(),
        *(e[2] for e in W._EXPLAIN),
    ]  # fmt: skip
    bad = [t for t in texts if JARGON.search(t)]
    assert bad == []
    assert W.LABELS["how_sure"] == "How sure it is" and W.LABELS["ignore_here"] == "Ignore here"
    assert W.LABELS["examples"] == "Practice examples"


def test_ui_strings_in_the_widget_module_have_no_jargon() -> None:
    import away.studio.ui as ui

    src = Path(ui.__file__).read_text(encoding="utf-8")
    literals = re.findall(r'(?<![\w.])"([A-Z][^"\n]{3,})"', src)  # capitalised string literals = visible text
    assert literals and [s for s in literals if JARGON.search(s)] == []


def test_every_tooltip_key_used_by_the_tab_exists() -> None:
    import away.studio.ui as ui

    src = Path(ui.__file__).read_text(encoding="utf-8")
    for key in re.findall(r'tip\([^()]*(?:\([^()]*\))*[^()]*,\s*"([a-z_]+)"\)', src):
        assert key in W.TOOLTIPS, key
    for key in re.findall(r'LABELS\["([a-z_]+)"\]', src):
        assert key in W.LABELS, key
    for key in re.findall(r'WIZARD\["([a-z0-9_]+)"\]', src):
        assert key in W.WIZARD, key


def test_friendly_errors_have_a_fix() -> None:
    assert W.explain("camera 0 did not open (index 0)") == (
        W.CAMERA_DID_NOT_START,
        "open_cameras",
        "Open camera settings",
    )
    msg, fix, label = W.explain("The app also needs to see your hand doing nothing special. Record a relax round.")  # type: ignore[misc]
    assert fix == "record_relax" and label == "Record a relax round"
    assert W.explain("'wave' was practised in 1 round(s). Do 2 more")[1] == "practise_more"  # type: ignore[index]
    assert W.explain("needs the learning add-on")[1] == "install_training"  # type: ignore[index]
    assert W.explain("Waiting for Project Away to let go of the camera...")[1] == ""  # type: ignore[index]
    assert W.explain("Camera on") is None
    for _a, _b, _msg, fix_id in W._EXPLAIN:
        assert fix_id == "" or fix_id in W.FIXES
    for fix_id in ("install_training", "setup", "allow_shell"):
        assert fix_id in W.HELP


def test_plain_word_helpers() -> None:
    assert W.name_for("thumbs_up") == "Thumbs up" and W.name_for(NOTHING) == "No gesture"
    assert W.state_text("firing") == "Doing it!" and W.state_text("odd") == "odd"
    assert W.sure_text(0.876) == "88%" and W.sure_text(1.7) == "100%"
    assert W.score_text(0.9, 10) == "When checking itself on practice rounds it had not seen, it got 9 of 10 right."
    assert W.score_text(None, 0).startswith("Practise in 3")
    assert W.summary_split({"rounds_per_gesture": {"nothing": 3, "wave": 1}}) == (
        "Practice rounds: No gesture: 3 rounds, Wave: 1 round"
    )
    assert W.action_text("key:ctrl+c") == "press ctrl+c" and W.action_text("none").startswith("nothing")
    assert W.action_text("launch:https://x.org") == "open https://x.org" and W.action_text("odd:thing") == "odd:thing"


def test_settings_window_has_the_studio_tab_not_the_old_test_tab() -> None:
    import away.app.settings_gui as sg

    src = Path(sg.__file__).read_text(encoding="utf-8")
    assert "self._build_studio(nb)" in src and "self._build_test(nb)" not in src
    assert "self.studio.stop.set()" in src
