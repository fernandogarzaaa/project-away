"""Gesture Studio (v0.1.20): recordings, local learning from a blank model, recogniser, states, packs, worker."""

from __future__ import annotations

import json
import queue
import threading
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from away import config as C
from away.app import camera_share as CS
from away.control.mouse import RecordingBackend
from away.hands.synthetic import gesture_sequence, pose
from away.studio import pack as P
from away.studio import trainer as T
from away.studio.engine import ChainRunner, GestureArbiter, ProbTimeline, StudioRecognizer, StudioRuntime, top_candidate
from away.studio.pack import NOTHING, GestureSettings, StudioStore
from studio_helpers import motion_store, record_nothing, record_static


@pytest.fixture
def trained_store(tmp_path: Path) -> StudioStore:
    rng = np.random.default_rng(1)
    store = StudioStore(tmp_path / "studio")
    store.add("fist_hold", "static", 1)
    record_static(store, "fist_hold", "fist", rounds=3, per_round=5, rng=rng)
    record_nothing(store, rounds=3, rng=rng)
    res = T.train_studio(store, backend="numpy")
    assert res.ok, res.message
    return store


# ---------------------------------------------------------------------- storage
def test_names_are_checked_and_made_friendly() -> None:
    assert P.friendly_name("Thumbs Up!") == "thumbs_up"
    assert P.friendly_name("  2 fingers ") == "g_2_fingers"
    assert P.friendly_name("Ünï") == "n"
    for bad in ("nothing", "pinch", "Has Caps", "", "x" * 33):
        with pytest.raises(ValueError):
            P.check_name(bad)
    assert P.check_name("wave_hi") == "wave_hi"


def test_save_take_keeps_the_second_hand_and_session(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.add("clap_pose", "static", 2)
    frames = [(pose("palm"), pose("fist")), (None, None), (pose("palm"), None), (pose("palm"), pose("fist"))]
    p = store.save_take("clap_pose", frames, "Left", 25.0, "sess-1")
    with np.load(p) as d:
        assert d["X"].shape == (3, 21, 3) and d["X2"].shape == (3, 21, 3)
        assert d["H2"].tolist() == [True, False, True]
        assert str(d["session_id"]) == "sess-1" and str(d["label"]) == "clap_pose" and str(d["handedness"]) == "Left"
    assert store.counts() == {"clap_pose": 1} and store.sessions("clap_pose") == 1
    with pytest.raises(ValueError, match="not in view"):
        store.save_take("clap_pose", [(None, None)] * 5 + [(pose("palm"), None)] * 2, "Right", 30, "s")
    with pytest.raises(ValueError):
        store.save_take("pinch", [(pose("palm"), None)] * 5, "Right", 30, "s")
    assert store.delete_all_clips() == 1 and store.clip_paths() == []


def test_store_add_remove_and_damaged_pack_file(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.add("a_one", "motion", 1)
    with pytest.raises(ValueError, match="already"):
        store.add("a_one")
    with pytest.raises(ValueError):
        store.add("b_two", "dance")
    with pytest.raises(ValueError):
        store.add("b_two", "static", 3)
    store.save_take("a_one", [(pose("palm"), None)] * 4, "Right", 30, "s")
    store.remove("a_one")
    assert store.gestures() == [] and store.clip_paths("a_one") == []
    store.pack_file.write_text("{not json")
    assert store.gestures() == []
    with pytest.raises(ValueError):
        store.delete_clip(tmp_path / "elsewhere.npz")


def test_chains_and_settings_validation() -> None:
    assert P.parse_chain(["wait:0.25", "key:ctrl+s", " type:hi ", "none"]) == ["wait:0.25", "key:ctrl+s", "type:hi"]
    for bad in (["wait:x"], ["wait:11"], ["mouse:move"], ["shell:ls"], "key:a", ["bogus:1"], ["key:a"] * 11):
        with pytest.raises(ValueError):
            P.parse_chain(bad)
    assert P.parse_chain(["shell:ls"], allow_shell=True) == ["shell:ls"]
    s = P.settings_from(None, "motion")
    assert s.hold_s == 0.0 and s.threshold == 0.8 and s.enabled
    assert P.settings_from(None, "static").hold_s == 0.3
    for bad in ({"threshold": 1.5}, {"enabled": "yes"}, {"hold_s": -1}, {"cooldown_s": True}, {"colour": 1}, []):
        with pytest.raises(ValueError):
            P.settings_from(bad, "static", path="x")
    cfg, _ = C.from_dict({"studio": {"gestures": {"wave": {"threshold": 0.9, "then": ["key:enter"]}}}})
    assert P.gesture_settings(cfg, "wave").threshold == 0.9 and P.gesture_settings(cfg, "other").threshold == 0.8
    for bad_cfg in (
        {"studio": {"gestures": {"wave": {"then": ["shell:rm -rf /"]}}}},
        {"studio": {"gestures": {"Bad Name": {}}}},
        {"studio": {"gestures": []}},
    ):
        with pytest.raises(ValueError):
            C.from_dict(bad_cfg)


# ---------------------------------------------------------------------- learning
def test_numpy_mlp_learns_and_folds_standardisation() -> None:
    from away.hands.model import NumpyMLP

    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(5, 1, (100, 6)), rng.normal(8, 1, (100, 6))]) * np.array([1, 10, 100, 1, 1, 1])
    y = np.array([0] * 100 + [1] * 100)
    w = T.train_mlp(X, y, 2, hidden=(16,), epochs=60)
    acc = (NumpyMLP(w, ["a", "b"]).predict_proba(X).argmax(1) == y).mean()
    assert acc > 0.97
    mu, sd = X.mean(0), X.std(0) + 1e-6
    raw = {"w0": rng.normal(size=(3, 6)), "b0": rng.normal(size=3)}
    folded = T.fold_standardisation(raw, mu, sd)
    x = X[:5]
    assert np.allclose(x @ folded["w0"].T + folded["b0"], ((x - mu) / sd) @ raw["w0"].T + raw["b0"])


def test_group_folds_are_disjoint_and_cover_each_round_once() -> None:
    groups = [f"s{i}" for i in range(9) for _ in range(3)]
    folds = T.group_folds(groups)
    assert len(folds) == T.MAX_FOLDS
    seen = [g for f in folds for g in f]
    assert sorted(seen) == sorted(set(groups)) and len(seen) == len(set(seen))
    assert T.group_folds(["a", "a", "b"]) == []
    assert len(T.group_folds(["a", "b", "c"])) == 3


def test_held_out_rounds_are_never_trained_on(monkeypatch, tmp_path: Path) -> None:
    """The score is group-disjoint: every clip checked in a fold comes from a round absent from its training."""
    rng = np.random.default_rng(2)
    store = StudioStore(tmp_path)
    store.add("fist_hold")
    record_static(store, "fist_hold", "fist", 3, 3, rng)
    record_nothing(store, 3, rng)
    clips = T.load_clips(store)
    seen_train: list[set[str]] = []
    real = T._fit_data

    def spy(cs, *a, **k):
        seen_train.append({c.session for c in cs})
        return real(cs, *a, **k)

    monkeypatch.setattr(T, "_fit_data", spy)
    pairs, split = T.evaluate_group(clips, "static", 1, ["fist_hold"], "numpy", 1.5)
    folds = T.group_folds([c.session for c in clips])
    assert split["folds"] == len(folds) == 5 and len(pairs) == len(clips)
    for train_sessions, fold in zip(seen_train, folds):
        assert not train_sessions & fold


def test_learning_a_pose_from_a_blank_model(trained_store: StudioStore) -> None:
    res = json.loads((trained_store.model_dir / "metrics.json").read_text())
    assert res["score"] is not None and res["score"] >= 0.8
    assert res["labels"] == [NOTHING, "fist_hold"]
    assert res["split"]["rounds_per_gesture"] == {NOTHING: 3, "fist_hold": 3}
    assert sum(map(sum, res["confusion"])) == 18  # 15 practice examples + 3 relax clips, each checked once
    rec = StudioRecognizer(trained_store.model_dir)
    assert rec.classes == [NOTHING, "fist_hold"]
    p_fist = rec.push(0.0, pose("fist", center=(0.55, 0.45)))
    assert p_fist is not None and p_fist["fist_hold"] > 0.8 and sum(p_fist.values()) == pytest.approx(1.0)
    p_palm = rec.push(0.1, pose("palm", center=(0.55, 0.45)))
    assert p_palm is not None and p_palm[NOTHING] > 0.8
    assert rec.push(0.2, None) is None


def test_saliency_and_nearest_example(trained_store: StudioStore) -> None:
    rec = StudioRecognizer(trained_store.model_dir)
    rec.push(0.0, pose("fist"))
    sal = rec.saliency("fist_hold")
    assert sal is not None and len(sal) == 21 and max(sal) == pytest.approx(1.0) and min(sal) >= 0.0
    assert rec.saliency(NOTHING) is None and rec.saliency("unknown") is None
    near = rec.nearest("fist_hold")
    assert near is not None and near[0] == "fist_hold" and 1 <= near[1] <= 15 and near[2] >= 0
    assert rec.nearest("other") is None


def test_broken_model_folder_is_not_loaded(tmp_path: Path) -> None:
    assert StudioRecognizer.load(tmp_path) is None
    (tmp_path / "studio.json").write_text(json.dumps({"groups": [{"key": "static1", "classes": [], "file": "x.npz"}]}))
    assert StudioRecognizer.load(tmp_path) is None


def test_learning_without_relax_round_or_gestures_explains_itself(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    assert "Teach a gesture first" in T.train_studio(store).message
    store.add("fist_hold")
    record_static(store, "fist_hold", "fist", 1, 3, np.random.default_rng(0))
    res = T.train_studio(store, backend="numpy")
    assert not res.ok and "relax round" in res.message
    assert not store.model_dir.exists()


def test_too_few_rounds_gives_no_score_and_a_warning(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    store = StudioStore(tmp_path)
    store.add("fist_hold")
    store.add("never_done")
    record_static(store, "fist_hold", "fist", rounds=1, per_round=5, rng=rng)
    record_nothing(store, rounds=1, rng=rng)
    res = T.train_studio(store, backend="numpy")
    assert res.ok and res.score is None and res.n_checked == 0
    text = " ".join(res.warnings)
    assert "practised in 1 round(s)" in text and "'never_done' has no practice examples" in text
    assert "5 practice examples" in text
    assert (store.model_dir / "studio.json").exists()


def test_similar_gestures_are_flagged(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    store = StudioStore(tmp_path)
    store.add("fist_a")
    store.add("fist_b")
    record_static(store, "fist_a", "fist", 3, 4, rng)
    record_static(store, "fist_b", "fist", 3, 4, rng)
    record_nothing(store, 3, rng)
    res = T.train_studio(store, backend="numpy")
    assert any("look very alike" in w for w in res.warnings)
    assert any("often thought it was" in w for w in res.warnings)


def test_different_gestures_are_not_flagged(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    store = StudioStore(tmp_path)
    store.add("fist_a")
    store.add("ok_b")
    record_static(store, "fist_a", "fist", 3, 4, rng)
    record_static(store, "ok_b", "pinch", 3, 4, rng)
    record_nothing(store, 3, rng)
    res = T.train_studio(store, backend="numpy")
    assert not any("look very alike" in w for w in res.warnings)
    assert res.score is not None and res.score >= 0.8


def test_learning_a_movement_numpy(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    store = motion_store(tmp_path, rng)
    res = T.train_studio(store, backend="numpy")
    assert res.ok and res.score is not None and res.score >= 0.7, res
    meta = json.loads((store.model_dir / "studio.json").read_text())
    assert meta["groups"][0]["key"] == "motion1" and meta["groups"][0]["window_s"] == pytest.approx(1.2)
    rec = StudioRecognizer(store.model_dir)
    swipe = gesture_sequence("swipe_right", frames=36, rng=rng)
    probs = None
    for i, f in enumerate(swipe):
        probs = rec.push(i / 30.0, f) or probs
    assert probs is not None and probs["flick"] > probs[NOTHING]
    sal = rec.saliency("flick")
    assert sal is not None and len(sal) == 21
    rec.reset()
    assert rec.push(0.0, swipe[0]) == {NOTHING: 1.0, "flick": 0.0}  # not enough motion yet


# ---------------------------------------------------------------------- states
def test_arbiter_hold_fire_cooldown() -> None:
    a = GestureArbiter()
    s = {"g": GestureSettings(threshold=0.8, hold_s=0.2, cooldown_s=1.0)}
    assert a.update(0.0, {"g": 0.9}, s) == [] and a.state["g"] == "arming"
    assert a.update(0.1, {"g": 0.9}, s) == []
    assert a.update(0.2, {"g": 0.9}, s) == ["g"] and a.state["g"] == "firing"
    assert a.update(0.3, {"g": 0.9}, s) == []
    assert a.update(0.6, {"g": 0.9}, s) == [] and a.state["g"] == "cooldown"
    assert a.update(1.5, {"g": 0.9}, s) == [] and a.state["g"] == "cooldown"  # still held: no repeat
    assert a.update(1.6, {"g": 0.1}, s) == [] and a.state["g"] == "idle"
    assert a.update(1.7, {"g": 0.7}, s) == [] and a.state["g"] == "idle"  # under the threshold
    a.update(1.8, {"g": 0.85}, s)
    assert a.update(1.9, {"g": 0.66}, s) == [] and a.state["g"] == "arming"  # inside the release margin
    assert a.update(2.0, {"g": 0.6}, s) == [] and a.state["g"] == "idle"


def test_arbiter_rest_time_blocks_a_quick_repeat() -> None:
    a = GestureArbiter()
    s = {"g": GestureSettings(hold_s=0.0, cooldown_s=1.0)}
    assert a.update(0.0, {"g": 0.9}, s) == ["g"]
    a.update(0.5, {"g": 0.1}, s)  # released, but the rest time is not over
    assert a.state["g"] == "cooldown" and a.update(0.6, {"g": 0.9}, s) == []
    a.update(1.1, {"g": 0.1}, s)
    assert a.state["g"] == "idle" and a.update(1.2, {"g": 0.9}, s) == ["g"]


def test_arbiter_recognized_when_off_or_outside_zones() -> None:
    a = GestureArbiter()
    s = {"g": GestureSettings(hold_s=0.0)}
    assert a.update(0.0, {"g": 0.95}, s, allowed=False) == [] and a.state["g"] == "recognized"
    assert a.update(0.1, {"g": 0.95}, s, allowed=True) == [] and a.state["g"] == "arming"
    assert a.update(0.2, {"g": 0.95}, s, allowed=True) == ["g"]
    off = {"g": GestureSettings(enabled=False, hold_s=0.0)}
    b = GestureArbiter()
    assert b.update(0.0, {"g": 0.99}, off) == [] and b.state["g"] == "recognized"
    assert b.update(0.1, None, off) == [] and b.state["g"] == "idle"


def test_top_candidate_and_timeline() -> None:
    assert top_candidate(None, 0.8) == (NOTHING, 0.0, True)
    assert top_candidate({NOTHING: 0.3, "g": 0.7}, 0.8) == ("g", 0.7, True)
    assert top_candidate({NOTHING: 0.1, "g": 0.9}, 0.8) == ("g", 0.9, False)
    tl = ProbTimeline(seconds=1.0)
    for i in range(30):
        tl.add(i * 0.1, {"g": i / 30})
    assert len(tl.points) in (10, 11) and tl.series("g")[-1] == (pytest.approx(2.9), pytest.approx(29 / 30))
    tl.fired(2.0, "g")
    tl.fired(2.95, "h")
    assert [e for _t, e in tl.recent_events(2.5)] == ["g", "h"] and tl.recent_events(5.0) == []


def test_chain_runner_runs_steps_in_order_with_waits() -> None:
    slept: list[float] = []
    ran: list[str] = []
    ChainRunner(sleep=slept.append, threaded=False).run(["key:a", "wait:0.5", "type:b"], ran.append)
    assert ran == ["key:a", "type:b"] and slept == [0.5]
    th = ChainRunner(sleep=lambda s: None).run(["key:c"], ran.append)
    assert th is not None
    th.join(2)
    assert ran[-1] == "key:c" and ChainRunner().run([], ran.append) is None


# ---------------------------------------------------------------------- the app runs Studio gestures
def test_pipeline_fires_a_studio_gesture_with_its_chain(trained_store: StudioStore) -> None:
    from away.app.runner import Pipeline

    cfg = C.AppConfig()
    cfg.bindings.custom_gestures.append("fist_hold")
    cfg.bindings.set("gesture:fist_hold", "key:ctrl+s")
    cfg.studio.gestures["fist_hold"] = {"hold_s": 0.1, "then": ["wait:0.2", "type:done"]}
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    slept: list[float] = []
    p.studio = StudioRuntime(p.cfg, store=trained_store, chains=ChainRunner(sleep=slept.append, threaded=False))
    out: list[str] = []
    for i in range(6):
        out += p.on_hand(i * 0.05, pose("fist"))
    assert out.count("custom:fist_hold") == 1
    assert ("hotkey", (["ctrl", "s"],)) in b.calls or any(c[0] == "hotkey" for c in b.calls)
    assert ("type_text", ("done",)) in b.calls and slept == [0.2]
    assert p.bus.history("gesture.custom")[-1].data == {"name": "fist_hold", "studio": True}


def test_pipeline_studio_gesture_respects_zones_and_unbound(trained_store: StudioStore) -> None:
    from away.app.runner import Pipeline
    from away.studio import zones as Z

    cfg = C.AppConfig()
    cfg.bindings.custom_gestures.append("fist_hold")
    cfg.studio.gestures["fist_hold"] = {"hold_s": 0.0, "then": ["type:x"]}
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    p.studio = StudioRuntime(p.cfg, store=trained_store, chains=ChainRunner(threaded=False))
    out: list[str] = []
    for i in range(3):
        out += p.on_hand(i * 0.05, pose("fist"))
    assert "custom:fist_hold" in out and not any(c[0] == "type_text" for c in b.calls)  # unbound: no chain
    zm = Z.ZoneMap()
    zm.set_all(Z.IGNORE)
    Z.store_map(cfg.zones.maps, "*", "default", zm)
    p2 = Pipeline(cfg, RecordingBackend())
    p2.studio = StudioRuntime(p2.cfg, store=trained_store)
    out2: list[str] = []
    for i in range(5):
        out2 += p2.on_hand(i * 0.05, pose("fist"))
    assert "custom:fist_hold" not in out2 and p2.studio.arbiter.state["fist_hold"] == "recognized"


def test_pipeline_studio_gesture_switched_off(trained_store: StudioStore) -> None:
    from away.app.runner import Pipeline

    cfg = C.AppConfig()
    cfg.bindings.custom_gestures.append("fist_hold")
    cfg.bindings.set("gesture:fist_hold", "key:f1")
    cfg.studio.gestures["fist_hold"] = {"enabled": False, "hold_s": 0.0}
    b = RecordingBackend()
    p = Pipeline(cfg, b)
    p.studio = StudioRuntime(p.cfg, store=trained_store)
    for i in range(4):
        assert "custom:fist_hold" not in p.on_hand(i * 0.05, pose("fist"))
    assert not any(c[0] == "key" for c in b.calls)


# ---------------------------------------------------------------------- packs
def test_export_and_import_a_pack(trained_store: StudioStore, tmp_path: Path) -> None:
    cfg = C.AppConfig()
    cfg.bindings.custom_gestures.append("fist_hold")
    cfg.bindings.set("gesture:fist_hold", "key:ctrl+z")
    cfg.studio.gestures["fist_hold"] = {"threshold": 0.9}
    lean = P.export_pack(trained_store, cfg, tmp_path / "lean.zip")
    with zipfile.ZipFile(lean) as zf:
        names = zf.namelist()
        man = json.loads(zf.read("manifest.json"))
    assert not any(n.startswith("clips/") for n in names) and "model/studio.json" in names
    assert man["includes_clips"] is False and man["gestures"][0]["action"] == "key:ctrl+z"
    assert man["gestures"][0]["settings"]["threshold"] == 0.9
    full = P.export_pack(trained_store, cfg, tmp_path / "full.zip", include_clips=True)
    with zipfile.ZipFile(full) as zf:
        clips = [n for n in zf.namelist() if n.startswith("clips/")]
    assert any(n.startswith("clips/fist_hold/") for n in clips) and any(n.startswith("clips/nothing/") for n in clips)
    other = StudioStore(tmp_path / "other")
    cfg2 = C.AppConfig()
    notes = P.import_pack(other, cfg2, lean)
    assert notes[0] == "added 1 gesture(s)" and other.clip_paths() == []
    assert [g.name for g in other.gestures()] == ["fist_hold"]
    assert str(cfg2.bindings.resolve("gestures", "fist_hold")) == "key:ctrl+z"
    assert cfg2.studio.gestures["fist_hold"]["threshold"] == 0.9
    C.validate(cfg2)
    rec = StudioRecognizer(other.model_dir)
    assert rec.classes == [NOTHING, "fist_hold"]
    third = StudioStore(tmp_path / "third")
    assert "with recordings" in P.import_pack(third, C.AppConfig(), full)[0] and third.clip_paths()


def test_import_refuses_unsafe_or_foreign_files_and_shell_actions(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "s")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        man = {"format": P.PACK_FORMAT, "version": 1, "gestures": [{"name": "evil"}], "model_files": []}
        zf.writestr("manifest.json", json.dumps(man))
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        P.import_pack(store, C.AppConfig(), bad)
    assert not (tmp_path / "escape.txt").exists() and store.gestures() == []
    foreign = tmp_path / "foreign.zip"
    with zipfile.ZipFile(foreign, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError, match="not a Project Away"):
        P.read_manifest(foreign)
    (tmp_path / "plain.zip").write_text("not a zip")
    with pytest.raises(ValueError, match="not a zip"):
        P.read_manifest(tmp_path / "plain.zip")
    newer = tmp_path / "newer.zip"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": P.PACK_FORMAT, "version": 99}))
    with pytest.raises(ValueError, match="newer"):
        P.read_manifest(newer)
    shelly = tmp_path / "shell.zip"
    with zipfile.ZipFile(shelly, "w") as zf:
        g = {"name": "runner", "action": "shell:rm -rf ~", "settings": {"then": ["shell:ls"]}}
        zf.writestr("manifest.json", json.dumps({"format": P.PACK_FORMAT, "version": 1, "gestures": [g]}))
    cfg = C.AppConfig()
    notes = P.import_pack(store, cfg, shelly)
    assert str(cfg.bindings.resolve("gestures", "runner")) == "none"
    assert any("action was not added" in n for n in notes) and any("default settings" in n for n in notes)
    with pytest.raises(ValueError, match="no gestures"):
        P.export_pack(StudioStore(tmp_path / "empty"), cfg, tmp_path / "x.zip")


# ---------------------------------------------------------------------- the camera worker
class FakeTracker:
    def __init__(self, hands):
        self.hands = list(hands)
        self.closed = False

    def process(self, rgb, t):
        lm, helper = self.hands.pop(0) if self.hands else (None, None)
        return types.SimpleNamespace(landmarks=lm, helper=helper, handedness="Right")

    def close(self):
        self.closed = True


def _drain(q: queue.Queue) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _run_worker(tmp_path, store, hands, cmds_list, **kw):
    from away.app.settings_gui import SettingsModel
    from away.studio.live import studio_camera_worker

    cfg = kw.pop("cfg", None) or C.AppConfig()
    inject = kw.pop("inject", {})
    m = SettingsModel(cfg, tmp_path / "c.yaml")
    out: queue.Queue = queue.Queue()
    cmds: queue.Queue = queue.Queue()
    for c in cmds_list:
        cmds.put(c)
    n = len(hands)
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)

    def frames(idx, mirror):
        for i in range(n):
            if i in inject:
                cmds.put(inject[i])
            yield i / 30.0, rgb

    tracker = FakeTracker(hands)
    studio_camera_worker(
        m, out, threading.Event(), cmds, store=store, share=CS.CameraShare(tmp_path / "share", pid=2),
        frames=frames, tracker_factory=lambda: tracker, every=1, **kw,
    )  # fmt: skip
    return _drain(out), tracker


def test_worker_shows_probabilities_states_and_fires_in_practice_mode(tmp_path, trained_store) -> None:
    fired_for_real: list[str] = []
    cfg = C.AppConfig()
    cfg.bindings.custom_gestures.append("fist_hold")
    cfg.bindings.set("gesture:fist_hold", "key:f5")
    hands = [(pose("palm"), None)] * 3 + [(pose("fist"), None)] * 20
    msgs, tracker = _run_worker(tmp_path, trained_store, hands, [], executor=fired_for_real.append, cfg=cfg)
    assert msgs[-1]["type"] == "stopped" and tracker.closed
    frames = [m for m in msgs if m["type"] == "frame"]
    assert frames[0]["top"] == NOTHING and set(frames[0]["probs"]) == {NOTHING, "fist_hold"}
    assert frames[0]["preview"][:2] == (64, 48) and frames[0]["preview"][2].startswith(b"P6 64 48 255\n")
    last = frames[-1]
    assert last["top"] == "fist_hold" and not last["unknown"] and last["states"]["fist_hold"] in ("firing", "cooldown")
    assert last["nearest"][0] == "fist_hold" and any(f["saliency"] for f in frames)
    fired = [m for m in msgs if m["type"] == "fired"]
    assert len(fired) == 1 and fired[0]["action"] == "key:f5" and fired[0]["dry_run"] is True
    assert fired_for_real == []  # practice mode: nothing ran


def test_worker_runs_actions_when_practice_mode_is_off(tmp_path, trained_store) -> None:
    ran: list[str] = []
    hands = [(pose("fist"), None)] * 20
    msgs, _ = _run_worker(tmp_path, trained_store, hands, [{"cmd": "dry_run", "on": False}], executor=ran.append)
    assert ran == ["fist_hold"] and [m["dry_run"] for m in msgs if m["type"] == "fired"] == [False]


def test_worker_zones_block_firing_and_report_the_point(tmp_path, trained_store) -> None:
    from away.studio import zones as Z

    zm = Z.ZoneMap()
    zm.set_all(Z.IGNORE)
    hands = [(pose("fist"), None)] * 15
    msgs, _ = _run_worker(tmp_path, trained_store, hands, [{"cmd": "zones", "map": zm.to_dict()}])
    frames = [m for m in msgs if m["type"] == "frame"]
    assert all(f["zone_ok"] is False for f in frames) and frames[-1]["point"] is not None
    assert not [m for m in msgs if m["type"] == "fired"] and frames[-1]["states"]["fist_hold"] == "recognized"
    msgs2, _ = _run_worker(tmp_path, trained_store, hands, [{"cmd": "zones", "map": {"grid": "bad"}}])
    assert any("could not be used" in m.get("text", "") for m in msgs2)


def test_worker_records_a_take_and_that_was_wrong(tmp_path) -> None:
    store = StudioStore(tmp_path / "empty")
    hands = [(pose("fist"), None)] * 40 + [(None, None)] * 50 + [(pose("palm"), None)] * 30
    cmds = [{"cmd": "record", "label": "fist_hold", "session": "s1", "seconds": 0.3, "countdown": 0.1, "tag": ("w", 0)}]
    msgs, _ = _run_worker(tmp_path, store, hands, cmds, inject={119: {"cmd": "wrong"}})
    assert any("No gestures taught yet" in m.get("text", "") for m in msgs)
    take = next(m for m in msgs if m["type"] == "take")
    assert take["label"] == "fist_hold" and take["session"] == "s1" and take["tag"] == ("w", 0)
    assert 8 <= len(take["frames"]) <= 11
    neg = next(m for m in msgs if m["type"] == "negative")
    # 'wrong' at frame 119 (t = 3.97 s): the frames of the last 2 s (frame 59 on) come back, newest last
    assert 59 <= len(neg["frames"]) <= 61 and neg["frames"][-1][0] is not None and neg["frames"][0][0] is None
    kinds = [m["type"] for m in msgs]
    assert kinds.index("countdown") < kinds.index("recording") < kinds.index("take")


def test_worker_reports_the_old_test_tab_events(tmp_path) -> None:
    hands = [(pose("pinch_open"), None), (pose("pinch"), None), (pose("pinch_open"), None)] + [(None, None)] * 3
    msgs, _ = _run_worker(tmp_path, StudioStore(tmp_path / "e"), hands, [])
    texts = [m["builtin"] for m in msgs if m["type"] == "frame" and m["builtin"]]
    assert any(t.startswith("pinch") and "mouse:click" in t for t in texts) and "no hand" in texts


def test_worker_builtin_source(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    hands = [(f, None) for f in gesture_sequence("swipe_right", frames=30, rng=np.random.default_rng(0))]
    msgs, _ = _run_worker(tmp_path, StudioStore(tmp_path / "e"), hands, [{"cmd": "source", "which": "builtin"}])
    probs = [m["probs"] for m in msgs if m["type"] == "frame" and m["probs"]]
    assert probs and NOTHING in probs[-1] and "swipe_right" in probs[-1] and "idle" not in probs[-1]


def test_worker_tracker_failure_is_reported(tmp_path) -> None:
    from away.app.settings_gui import SettingsModel
    from away.studio.live import studio_camera_worker

    out: queue.Queue = queue.Queue()

    def boom():
        raise RuntimeError("no model file")

    studio_camera_worker(
        SettingsModel(C.AppConfig(), tmp_path / "c.yaml"), out, threading.Event(), queue.Queue(),
        store=StudioStore(tmp_path / "e"), share=CS.CameraShare(tmp_path / "share", pid=2),
        frames=lambda i, m: iter([]), tracker_factory=boom,
    )  # fmt: skip
    msgs = _drain(out)
    assert any("Hand tracking could not start" in m.get("text", "") for m in msgs) and msgs[-1]["type"] == "stopped"


def test_ppm_preview_rejects_non_images() -> None:
    from away.studio.live import ppm_preview

    assert ppm_preview(None) is None and ppm_preview(np.zeros((4, 4))) is None
    w, h, data = ppm_preview(np.full((480, 640, 3), 7, dtype=np.uint8))  # type: ignore[misc]
    assert (w, h) == (320, 240) and len(data) == len(b"P6 320 240 255\n") + 320 * 240 * 3
