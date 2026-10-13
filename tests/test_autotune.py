"""Tuning tab live view + auto-tune: measurements, gauges, threshold selection, the guided session, clip saving
(away record format with session_id) and the camera worker with fake frames."""

from __future__ import annotations

import queue
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from away import config as C
from away.app import camera_share as CS
from away.app import tuning_live as TL
from away.hands import autotune as A
from away.hands import synthetic as S
from away.hands.dataset import load_recordings_with_groups
from away.hands.recorder import save_clip

PALM = S.pose("palm")
FIST = S.pose("fist")
FOUR = S.pose("four")


def pinch_hand(p: float) -> np.ndarray:
    return S.make_hand(curls={"index": 0.35, "middle": 0.6, "ring": 0.8, "pinky": 0.8}, pinch=p)


# ===================================================================================== measurements
def test_measurements_without_a_hand() -> None:
    assert A.pinch_distance(None) is None and A.thumb_fold(None) is None
    assert A.palm_depth(None, PALM) is None and A.palm_depth(PALM, None) is None
    assert A.measure("index_extension", None) is None
    assert A.measure("pinch", np.zeros((3, 3))) is None  # malformed landmarks
    with pytest.raises(ValueError, match="unknown measurement"):
        A.measure("nope", PALM)


def test_pinch_distance_follows_the_pinch() -> None:
    d = [A.pinch_distance(pinch_hand(p)) for p in (0.0, 0.5, 1.0)]
    assert d[0] > d[1] > d[2] and d[2] == pytest.approx(0.0, abs=1e-9)
    assert A.measure("pinch", pinch_hand(1.0)) == pytest.approx(0.0, abs=1e-9)
    degenerate = np.zeros((21, 3))
    assert A.pinch_distance(degenerate) is None


def test_thumb_fold_open_vs_tucked() -> None:
    assert A.thumb_fold(PALM) > 1.0 > 0.95 > A.thumb_fold(FOUR)
    assert A.thumb_fold(np.zeros((21, 3))) is None


def test_finger_extension_open_vs_fist() -> None:
    open_ = A.finger_extension(PALM)
    closed = A.finger_extension(FIST)
    assert set(open_) == {"index", "middle", "ring", "pinky"}
    assert all(v > A.EXTENDED_RATIO for v in open_.values())
    assert all(v < A.EXTENDED_RATIO for v in closed.values())
    assert A.finger_extension(np.zeros((21, 3))) == {"index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}


def test_palm_depth_on_vs_off_the_palm() -> None:
    assert A.palm_depth(PALM, PALM) == pytest.approx(0.0, abs=1e-6)  # the fingertip lies in the palm plane
    far = S.make_hand(scale=0.06)  # half the apparent size: twice as far from the camera
    assert A.palm_depth(far, PALM) > 0.5
    assert A.measure("palm_depth", far, PALM) == pytest.approx(A.palm_depth(far, PALM))
    assert A.palm_depth(PALM, np.zeros((21, 3))) is None  # degenerate palm


# ===================================================================================== gauges
def test_gauge_active_fraction_text() -> None:
    g = A.Gauge("p", "Pinch", 0.2, 0.32, 0.0, 1.0)
    assert g.active and g.fraction == pytest.approx(0.2) and g.threshold_fraction == pytest.approx(0.32)
    assert g.text() == "Pinch: 0.20 < 0.32  ON"
    g2 = A.Gauge("p", "Pinch", 0.32, 0.32, 0.0, 1.0)
    assert not g2.active and g2.text() == "Pinch: 0.32 >= 0.32  off"
    up = A.Gauge("c", "Curl", 1.2, 1.1, 0.5, 2.0, "above")
    assert up.active and up.text() == "Curl: 1.20 > 1.10  ON"
    down = A.Gauge("c", "Curl", 1.1, 1.1, 0.5, 2.0, "above")
    assert not down.active and down.text() == "Curl: 1.10 <= 1.10  off"
    none = A.Gauge("x", "X", None, 1.0, 0.0, 2.0, hint="no hand")
    assert not none.active and none.fraction is None and none.text() == "X: (not measured)  no hand"
    assert A.Gauge("x", "X", None, 1.0, 0.0, 2.0).text() == "X: (not measured)"


def test_gauge_fraction_clamps() -> None:
    assert A.gauge_fraction(0.5, 0.0, 1.0) == 0.5
    assert A.gauge_fraction(-1.0, 0.0, 1.0) == 0.0
    assert A.gauge_fraction(3.0, 0.0, 1.0) == 1.0
    assert A.gauge_fraction(1.0, 1.0, 1.0) == 0.0
    assert A.gauge_fraction(1.5, 1.0, 2.0) == 0.5


def test_live_gauges_follow_the_settings() -> None:
    cfg = C.AppConfig()
    gs = A.live_gauges(pinch_hand(1.0), None, cfg)
    assert [g.key for g in gs] == ["pinch", "thumb_fold", "curl", "palm_depth"]
    by = {g.key: g for g in gs}
    assert by["pinch"].active and by["pinch"].threshold == 0.32
    assert by["thumb_fold"].threshold == 0.95 and by["palm_depth"].value is None
    assert by["palm_depth"].hint == "needs both hands"
    cfg.hands.gestures.pinch_on = 0.0001
    assert not {g.key: g for g in A.live_gauges(pinch_hand(0.9), None, cfg)}["pinch"].active
    palm = {g.key: g for g in A.live_gauges(PALM, PALM, cfg)}
    assert palm["curl"].label == "Index finger out (4/4 fingers up)" and palm["curl"].active
    assert palm["palm_depth"].active and palm["palm_depth"].threshold == 0.6
    cfg.hands.palm_scroll.depth_threshold = 0.0
    off = {g.key: g for g in A.live_gauges(PALM, PALM, cfg)}["palm_depth"]
    assert "no depth check" in off.label and off.threshold == 3.0
    fist = {g.key: g for g in A.live_gauges(FIST, None, cfg)}
    assert fist["curl"].label.endswith("(0/4 fingers up)") and not fist["curl"].active
    empty = A.live_gauges(None, None, cfg)
    assert all(g.value is None for g in empty) and "(0/4" in empty[2].label


def test_skeleton_points_fit_the_canvas() -> None:
    assert A.skeleton_points([], 200, 100) == []
    assert A.skeleton_points([None], 200, 100) == []
    (pts,) = A.skeleton_points([PALM], 200, 100, margin=10)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert min(xs) >= 10 - 1e-9 and max(xs) <= 190 + 1e-9 and min(ys) >= 10 - 1e-9 and max(ys) <= 90 + 1e-9
    assert max(ys) - min(ys) == pytest.approx(80) or max(xs) - min(xs) == pytest.approx(180)
    # centred
    assert (min(xs) + max(xs)) / 2 == pytest.approx(100) and (min(ys) + max(ys)) / 2 == pytest.approx(50)
    two = A.skeleton_points([PALM, S.make_hand(center=(0.8, 0.5))], 300, 150)
    assert len(two) == 2 and all(len(h) == 21 for h in two)
    assert min(p[0] for p in two[1]) > min(p[0] for p in two[0])  # the second hand is to the right
    assert len(A.HAND_CONNECTIONS) == 21 and all(0 <= a < 21 and 0 <= b < 21 for a, b in A.HAND_CONNECTIONS)


def test_skeleton_uniform_scale() -> None:
    hand = np.zeros((21, 3))
    hand[:, 0] = np.linspace(0.0, 0.1, 21)  # a flat line: x span only
    (pts,) = A.skeleton_points([hand], 100, 100, margin=0)
    assert pts[0] == pytest.approx((0.0, 50.0), abs=1e-3) and pts[-1] == pytest.approx((100.0, 50.0), abs=1e-3)


# ===================================================================================== thresholds
def test_clip_statistic() -> None:
    assert A.clip_statistic([0.1, None, float("nan")]) is None
    assert A.clip_statistic([1.0, 2.0]) is None
    vals = [float(v) for v in range(11)]  # 0..10
    assert A.clip_statistic(vals) == pytest.approx(1.0)
    assert A.clip_statistic(vals, "above") == pytest.approx(9.0)
    assert A.clip_statistic([5.0, float("inf"), 5.0, None, 5.0]) == 5.0


def test_balanced_accuracy() -> None:
    assert A.balanced_accuracy([0.1, 0.2], [0.5, 0.6], 0.3) == 1.0
    assert A.balanced_accuracy([0.1, 0.2], [0.5, 0.6], 0.15) == 0.75
    assert A.balanced_accuracy([0.1, 0.2], [0.5, 0.6], 0.0) == 0.5
    assert A.balanced_accuracy([0.1, 0.2], [0.5, 0.6], 0.2) == 0.75  # strict: 0.2 < 0.2 does not fire
    assert A.balanced_accuracy([0.9, 0.8], [0.1, 0.2], 0.5, "above") == 1.0
    assert A.balanced_accuracy([0.9, 0.8], [0.1, 0.5], 0.5, "above") == 1.0  # 0.5 <= 0.5 stays quiet
    assert A.balanced_accuracy([0.9, 0.5], [0.1, 0.2], 0.5, "above") == 0.75


def test_choose_threshold_separable_midpoint() -> None:
    c = A.choose_threshold([0.1, 0.15, 0.2], [0.5, 0.6, 0.7])
    assert c.threshold == pytest.approx(0.35) and c.separable and c.balanced_accuracy == 1.0
    assert c.margin == pytest.approx(0.15) and (c.n_pos, c.n_neg) == (3, 3) and not c.clamped
    up = A.choose_threshold([0.9, 1.0, 1.1], [0.1, 0.2, 0.3], "above")
    assert up.threshold == pytest.approx(0.6) and up.separable


def test_choose_threshold_overlap_and_ties() -> None:
    c = A.choose_threshold([0.1, 0.2, 0.55], [0.5, 0.6, 0.7])
    assert not c.separable and c.margin == 0.0
    assert c.balanced_accuracy == pytest.approx(5 / 6)
    assert c.threshold == pytest.approx(0.35)  # widest gap among the best splits (0.2..0.5)
    # two equally good gaps of equal width: the lower threshold wins
    t = A.choose_threshold([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert t.threshold == pytest.approx(0.5)
    same = A.choose_threshold([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    assert same.balanced_accuracy == 0.5 and not same.separable


def test_choose_threshold_clamps_and_filters() -> None:
    c = A.choose_threshold([0.1, 0.15, 0.2], [0.5, 0.6, 0.7], lo=0.4, hi=1.0)
    assert c.threshold == 0.4 and c.clamped
    c2 = A.choose_threshold([0.1, 0.15, 0.2], [0.5, 0.6, 0.7], lo=0.0, hi=0.3)
    assert c2.threshold == 0.3 and c2.clamped
    c3 = A.choose_threshold([0.1, 0.15, 0.2, float("nan")], [0.5, 0.6, 0.7, float("inf")])
    assert c3.n_pos == 3 and c3.n_neg == 3
    with pytest.raises(ValueError, match="at least 3"):
        A.choose_threshold([0.1, 0.2], [0.5, 0.6, 0.7])
    with pytest.raises(ValueError, match="at least 3"):
        A.choose_threshold([0.1, 0.2, 0.3], [0.5, float("nan"), 0.7])
    with pytest.raises(ValueError, match="fires"):
        A.choose_threshold([0.1, 0.2, 0.3], [0.5, 0.6, 0.7], "sideways")


def test_choose_threshold_all_pos_below_everything() -> None:
    c = A.choose_threshold([0.1, 0.1, 0.1], [0.1, 0.1, 0.9])
    assert c.balanced_accuracy == pytest.approx(2 / 3)
    assert c.threshold == pytest.approx(0.5)  # just above the 0.1 cluster: the 0.9 trial stays quiet


# ===================================================================================== session
def _trial(hand: np.ndarray, n: int = 10, helper: np.ndarray | None = None) -> list[tuple]:
    return [(hand, helper) for _ in range(n)]


def test_session_steps_prompts_and_result() -> None:
    spec = A.AUTOTUNE_SPECS["pinch"]
    s = A.AutoTuneSession(spec, count=3, session_id="sid")
    assert s.next_step() == ("positive", 1) and s.prompt().endswith("(1/3)")
    assert spec.positive_prompt in s.prompt()
    for p in (1.0, 0.95, 0.9):
        assert s.add_trial(_trial(pinch_hand(p))) is None
    assert s.next_step() == ("negative", 1) and spec.negative_prompt in s.prompt()
    assert "could not measure" in s.add_trial(_trial(None))  # type: ignore[arg-type]
    assert s.next_step() == ("negative", 1)
    with pytest.raises(ValueError, match="record all trials"):
        s.result({}, {})
    for p in (0.0, 0.1, 0.2):
        assert s.add_trial(_trial(pinch_hand(p))) is None
    assert s.next_step() is None and s.prompt().startswith("Done")
    assert s.add_trial(_trial(pinch_hand(1.0))) == "all trials are recorded"
    cur = {"hands.gestures.pinch_on": 0.32, "hands.gestures.pinch_off": 0.45}
    bounds = {"hands.gestures.pinch_on": (0.05, 1.0), "hands.gestures.pinch_off": (0.05, 1.5)}
    r = s.result(cur, bounds)
    hi_pos = max(r.pos_stats)
    lo_neg = min(r.neg_stats)
    assert hi_pos < r.after["hands.gestures.pinch_on"] < lo_neg
    assert r.after["hands.gestures.pinch_on"] == pytest.approx((hi_pos + lo_neg) / 2)
    gap = 0.45 - 0.32
    assert r.after["hands.gestures.pinch_off"] == pytest.approx(r.after["hands.gestures.pinch_on"] + gap)
    assert r.before == cur
    assert r.summary().startswith("Pinch click: 0.32 -> ") and "separates all 3 + 3 trials" in r.summary()


def test_session_result_clamps_linked_and_reports_overlap() -> None:
    s = A.AutoTuneSession(A.AUTOTUNE_SPECS["pinch"], count=3)
    for p in (0.0, 0.1, 0.2):
        s.add_trial(_trial(pinch_hand(p)))
    for p in (0.05, 0.15, 0.25):
        s.add_trial(_trial(pinch_hand(p)))
    cur = {"hands.gestures.pinch_on": 0.32, "hands.gestures.pinch_off": 0.45}
    r = s.result(cur, {"hands.gestures.pinch_off": (0.05, 0.5), "hands.gestures.pinch_on": (0.9, 1.0)})
    assert r.after["hands.gestures.pinch_on"] == 0.9 and r.choice.clamped
    assert r.after["hands.gestures.pinch_off"] == 0.5  # 0.45 + 0.58 clamped
    assert "overlap" in r.summary() and "(limited by the slider range)" in r.summary()
    r2 = s.result({"hands.gestures.pinch_on": 0.32}, {})  # linked value unknown: left alone
    assert set(r2.after) == {"hands.gestures.pinch_on"}


def test_thumb_fold_and_palm_depth_sessions() -> None:
    s = A.AutoTuneSession(A.AUTOTUNE_SPECS["thumb_fold"], count=3)
    for _ in range(3):
        s.add_trial(_trial(FOUR))
    for _ in range(3):
        s.add_trial(_trial(PALM))
    r = s.result({"hands.gestures.autoscroll_thumb_fold": 0.95}, {"hands.gestures.autoscroll_thumb_fold": (0.3, 1.4)})
    assert A.thumb_fold(FOUR) < r.after["hands.gestures.autoscroll_thumb_fold"] < A.thumb_fold(PALM)

    d = A.AutoTuneSession(A.AUTOTUNE_SPECS["palm_depth"], count=3)
    assert "both hands" in d.add_trial(_trial(PALM))  # no helper: not measurable
    far = S.make_hand(scale=0.06)
    for _ in range(3):
        d.add_trial(_trial(PALM, helper=PALM))
    for _ in range(3):
        d.add_trial(_trial(far, helper=PALM))
    r = d.result({"hands.palm_scroll.depth_threshold": 0.6}, {"hands.palm_scroll.depth_threshold": (0.0, 3.0)})
    assert 0.0 < r.after["hands.palm_scroll.depth_threshold"] < A.palm_depth(far, PALM)


def test_session_rejects_tiny_counts_and_makes_ids() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        A.AutoTuneSession(A.AUTOTUNE_SPECS["pinch"], count=2)
    s = A.AutoTuneSession(A.AUTOTUNE_SPECS["pinch"])
    assert s.count == 5 and s.session_id.startswith("autotune-pinch-")
    sid = A.new_session_id("x", now=0.0)
    parts = sid.split("-")
    assert parts[0] == "autotune" and parts[1] == "x" and len(parts[-1]) == 4
    assert A.new_session_id("x") != A.new_session_id("x")


def test_specs_cover_the_required_gestures() -> None:
    assert set(A.AUTOTUNE_SPECS) >= {"pinch", "thumb_fold", "palm_depth"}
    paths = {s.path for s in A.AUTOTUNE_SPECS.values()}
    assert paths >= {
        "hands.gestures.pinch_on",
        "hands.gestures.autoscroll_thumb_fold",
        "hands.palm_scroll.depth_threshold",
    }
    tunables = {p for p, *_ in C.tunable_ranges(C.AppConfig())}
    for s in A.AUTOTUNE_SPECS.values():
        assert s.path in tunables and set(s.linked) <= tunables and s.measure in A.MEASURES
        assert s.fires in ("below", "above")


def test_save_clips_writes_labelled_sessions(tmp_path: Path) -> None:
    s = A.AutoTuneSession(A.AUTOTUNE_SPECS["pinch"], count=3, session_id="autotune-pinch-test")
    for p in (1.0, 0.95, 0.9):
        s.add_trial(_trial(pinch_hand(p), n=8))
    for p in (0.0, 0.1, 0.2):
        s.add_trial(_trial(pinch_hand(p), n=8))
    s.positives[0] = s.positives[0][:4] + [(None, None)] * 4  # 4 hand frames: skipped (< 5)
    paths = s.save_clips(tmp_path, "Left", 25.0)
    assert len(paths) == 5
    assert {p.parent.name for p in paths} == {"pinch_click", "move"}
    with np.load(paths[0], allow_pickle=False) as d:
        assert d["X"].shape == (8, 21, 3) and str(d["label"]) == "pinch_click"
        assert str(d["handedness"]) == "Left" and float(d["fps"]) == 25.0
        assert str(d["session_id"]) == "autotune-pinch-test" and str(d["source"]) == "autotune"
        assert str(d["trial"]) == "positive" and str(d["measure"]) == "pinch" and str(d["autotune"]) == "pinch"
        assert d["values"].shape == (8,) and np.all(np.isfinite(d["values"]))
    seqs, labels, hands, groups = load_recordings_with_groups(tmp_path)
    assert len(seqs) == 5 and set(groups) == {"autotune-pinch-test"}
    assert sorted(labels) == ["move", "move", "move", "pinch_click", "pinch_click"]


def test_save_clips_custom_labels_and_nan_values(tmp_path: Path) -> None:
    s = A.AutoTuneSession(A.AUTOTUNE_SPECS["palm_depth"], count=3)
    far = S.make_hand(scale=0.06)
    for _ in range(3):
        s.add_trial([(PALM, PALM)] * 6 + [(PALM, None)])
    for _ in range(3):
        s.add_trial(_trial(far, n=6, helper=PALM))
    paths = s.save_clips(tmp_path)
    assert {p.parent.name for p in paths} == {"palm_scroll", "move"}
    with np.load(next(p for p in paths if p.parent.name == "palm_scroll")) as d:
        assert np.isnan(d["values"][-1]) and d["X"].shape[0] == 7


# ===================================================================================== recorder
def test_save_clip_session_id_extra_and_unique_names(tmp_path: Path, monkeypatch) -> None:
    import away.hands.recorder as R

    monkeypatch.setattr(R.time, "time", lambda: 1_700_000_000.123)
    frames = [PALM] * 5
    a = save_clip(tmp_path, "idle", frames, "Right", 30.0, session_id="s1", extra={"note": "x"})
    b = save_clip(tmp_path, "idle", frames, "Right", 30.0)
    c = save_clip(tmp_path, "idle", frames, "Right", 30.0)
    assert len({a, b, c}) == 3 and b.name.endswith("-1.npz") and c.name.endswith("-2.npz")
    with np.load(a) as d:
        assert str(d["session_id"]) == "s1" and str(d["note"]) == "x"
    with np.load(b) as d:
        assert "session_id" not in d.files
    with pytest.raises(ValueError, match="no hand frames"):
        save_clip(tmp_path, "idle", [], "Right", 30.0)


# ===================================================================================== camera worker
def test_handle_command() -> None:
    assert TL.handle_command(None, "junk", 0.0) is None
    rec = TL.handle_command(None, {"cmd": "record", "tag": "a", "seconds": 2.0, "countdown": 1.0}, 10.0)
    assert rec["start"] == 11.0 and rec["end"] == 13.0 and rec["tag"] == "a" and rec["frames"] == []
    assert TL.handle_command(rec, {"cmd": "cancel"}, 10.5) is None
    assert TL.handle_command(rec, {"cmd": "other"}, 10.5) is rec
    tiny = TL.handle_command(None, {"cmd": "record", "seconds": 0.0, "countdown": -5}, 1.0)
    assert tiny["start"] == 1.0 and tiny["seconds"] == 0.1
    dflt = TL.handle_command(None, {"cmd": "record"}, 0.0)
    assert dflt["start"] == 1.5 and dflt["end"] == 3.5


def test_step_recording_countdown_then_trial() -> None:
    rec = TL.handle_command(None, {"cmd": "record", "tag": "t", "seconds": 1.0, "countdown": 2.0}, 0.0)
    msgs, done = TL.step_recording(rec, 0.1, PALM, None, "Right")
    assert msgs == [{"type": "countdown", "tag": "t", "left": 2}] and not done
    assert TL.step_recording(rec, 0.5, PALM, None, "Right") == ([], False)  # same second: no repeat
    msgs, _ = TL.step_recording(rec, 1.2, PALM, None, "Right")
    assert msgs == [{"type": "countdown", "tag": "t", "left": 1}]
    msgs, done = TL.step_recording(rec, 2.0, PALM, None, "Left")
    assert msgs == [{"type": "recording", "tag": "t"}] and not done
    msgs, done = TL.step_recording(rec, 2.5, None, None, "")
    assert msgs == [] and not done
    msgs, done = TL.step_recording(rec, 3.0, PALM, PALM, "Left")
    assert done and msgs[-1]["type"] == "trial" and len(msgs[-1]["frames"]) == 3
    assert msgs[-1]["handedness"] == "Left" and msgs[-1]["fps"] == pytest.approx(2.0)


def test_step_recording_frame_past_the_end_is_not_kept() -> None:
    rec = TL.handle_command(None, {"cmd": "record", "tag": 1, "seconds": 1.0, "countdown": 0.0}, 0.0)
    TL.step_recording(rec, 0.0, PALM, None, "Right")
    msgs, done = TL.step_recording(rec, 1.5, PALM, None, "Right")
    assert done and len(msgs[-1]["frames"]) == 1


class FakeTracker:
    closed = False

    def __init__(self, hands):
        self.hands = list(hands)

    def process(self, rgb, t):
        lm, helper = self.hands.pop(0) if self.hands else (None, None)
        return types.SimpleNamespace(landmarks=lm, helper=helper, handedness="Right")

    def close(self):
        FakeTracker.closed = True


def _drain(q: queue.Queue) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_worker_streams_frames_and_records_a_trial(tmp_path: Path) -> None:
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    out: queue.Queue = queue.Queue()
    cmds: queue.Queue = queue.Queue()
    cmds.put({"cmd": "record", "tag": ("s", ("positive", 1)), "seconds": 0.3, "countdown": 0.2})
    n = 20
    frames = lambda idx, mirror: iter([(i * 0.05, None) for i in range(n)])  # noqa: E731
    tracker = FakeTracker([(pinch_hand(1.0), None)] * n)
    share = CS.CameraShare(tmp_path / "share", pid=2)
    TL.tuning_camera_worker(
        m, out, threading.Event(), cmds, share=share, frames=frames, tracker_factory=lambda: tracker, every=5
    )
    msgs = _drain(out)
    kinds = [x["type"] for x in msgs]
    assert kinds[0] == "status" and kinds[-1] == "stopped" and FakeTracker.closed
    frames_msgs = [x for x in msgs if x["type"] == "frame"]
    assert len(frames_msgs) == 4  # every 5th of 20
    assert frames_msgs[0]["gauges"][0].key == "pinch" and frames_msgs[0]["gauges"][0].active
    trial = next(x for x in msgs if x["type"] == "trial")
    assert trial["tag"] == ("s", ("positive", 1)) and 6 <= len(trial["frames"]) <= 8
    assert kinds.index("countdown") < kinds.index("recording") < kinds.index("trial")


def test_worker_stop_cancel_and_errors(tmp_path: Path) -> None:
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    share = CS.CameraShare(tmp_path / "share", pid=2)

    # stop before the first frame
    out: queue.Queue = queue.Queue()
    stop = threading.Event()
    stop.set()
    frames = lambda idx, mirror: iter([(0.0, None)])  # noqa: E731
    TL.tuning_camera_worker(
        m, out, stop, queue.Queue(), share=share, frames=frames, tracker_factory=lambda: FakeTracker([])
    )
    assert not [x for x in _drain(out) if x["type"] == "frame"]

    # a cancel drops the recording
    out = queue.Queue()
    cmds: queue.Queue = queue.Queue()
    cmds.put({"cmd": "record", "tag": 1, "seconds": 0.1, "countdown": 0.0})
    cmds.put({"cmd": "cancel"})
    frames = lambda idx, mirror: iter([(i * 0.1, None) for i in range(5)])  # noqa: E731
    TL.tuning_camera_worker(
        m, out, threading.Event(), cmds, share=share, frames=frames, tracker_factory=lambda: FakeTracker([])
    )
    assert not [x for x in _drain(out) if x["type"] in ("trial", "recording")]

    # the camera fails
    def broken(idx, mirror):
        raise RuntimeError(f"cannot open camera {idx} (in use by another app, no device, or permission denied)")
        yield  # pragma: no cover

    out = queue.Queue()
    TL.tuning_camera_worker(
        m, out, threading.Event(), queue.Queue(), share=share, frames=broken, tracker_factory=lambda: FakeTracker([])
    )
    texts = [x.get("text", "") for x in _drain(out)]
    assert any("in use by Project Away or another app" in t for t in texts)

    # the tracker cannot start
    def no_tracker():
        raise FileNotFoundError("hand_landmarker.task")

    out = queue.Queue()
    TL.tuning_camera_worker(
        m, out, threading.Event(), queue.Queue(), share=share, frames=frames, tracker_factory=no_tracker
    )
    msgs = _drain(out)
    assert any("hand tracker unavailable" in x.get("text", "") for x in msgs) and msgs[-1]["type"] == "stopped"


def test_worker_survives_malformed_landmarks(tmp_path: Path) -> None:
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    out: queue.Queue = queue.Queue()
    frames = lambda idx, mirror: iter([(0.0, None)])  # noqa: E731
    TL.tuning_camera_worker(
        m,
        out,
        threading.Event(),
        queue.Queue(),
        share=CS.CameraShare(tmp_path / "share", pid=2),
        frames=frames,
        tracker_factory=lambda: FakeTracker([(np.zeros((5, 3)), None)]),
        every=1,
    )
    fr = [x for x in _drain(out) if x["type"] == "frame"]
    assert len(fr) == 1 and fr[0]["gauges"] == []


# ===================================================================================== edge cases (mutation pass)
def test_handle_command_keeps_the_recording_on_junk() -> None:
    rec = TL.handle_command(None, {"cmd": "record"}, 0.0)
    assert TL.handle_command(rec, "junk", 1.0) is rec
    assert TL.handle_command(rec, None, 1.0) is rec


def test_step_recording_fps_uses_the_seconds() -> None:
    rec = TL.handle_command(None, {"cmd": "record", "tag": 1, "seconds": 0.5, "countdown": 0.0}, 0.0)
    for t in (0.0, 0.25):
        TL.step_recording(rec, t, PALM, None, "Right")
    msgs, done = TL.step_recording(rec, 0.5, PALM, None, "Right")
    assert done and msgs[-1]["fps"] == pytest.approx(6.0)  # 3 hand frames in 0.5 s


def test_worker_frame_cadence(tmp_path: Path) -> None:
    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    share = CS.CameraShare(tmp_path / "share", pid=2)
    frames = lambda idx, mirror: iter([(i * 0.1, None) for i in range(20)])  # noqa: E731
    for every, want in ((4, 5), (None, 10)):
        out: queue.Queue = queue.Queue()
        kw = {} if every is None else {"every": every}
        TL.tuning_camera_worker(
            m, out, threading.Event(), queue.Queue(), share=share, frames=frames,
            tracker_factory=lambda: FakeTracker([]), **kw,
        )  # fmt: skip
        fr = [x for x in _drain(out) if x["type"] == "frame"]
        assert len(fr) == want and fr[0]["t"] == 0.0


def test_worker_borrows_the_camera_from_the_running_app(tmp_path: Path) -> None:
    import time

    from away.app.settings_gui import SettingsModel

    m = SettingsModel(C.AppConfig(), tmp_path / "c.yaml")
    app = CS.CameraShare(tmp_path / "share", pid=1)
    app.publish_owner([0])

    def app_side():
        for _ in range(300):
            if app.requested():
                app.ack_released([0])
                return
            time.sleep(0.01)

    threading.Thread(target=app_side, daemon=True).start()
    out: queue.Queue = queue.Queue()
    frames = lambda idx, mirror: iter([(0.0, None)])  # noqa: E731
    TL.tuning_camera_worker(
        m, out, threading.Event(), queue.Queue(), share=CS.CameraShare(tmp_path / "share", pid=2),
        frames=frames, tracker_factory=lambda: FakeTracker([]),
    )  # fmt: skip
    texts = [x.get("text", "") for x in _drain(out)]
    assert any("paused hand tracking for the live view" in t for t in texts)
    assert not app.requested()  # handed back


def test_gauge_active_is_a_bool() -> None:
    assert A.Gauge("x", "X", None, 1.0, 0.0, 2.0).active is False
    assert A.gauge_fraction(1.0, 0.0, 2.0) == 0.5
    assert A.gauge_fraction(0.5, 0.0, 4.0) == 0.125


def test_live_gauge_ranges_are_fixed() -> None:
    gs = A.live_gauges(PALM, PALM, C.AppConfig())
    assert [(g.lo, g.hi, g.fires) for g in gs] == [
        (0.0, 1.0, "below"),
        (0.0, 1.6, "below"),
        (0.5, 2.0, "above"),
        (0.0, 2.0, "below"),
    ]
    assert gs[3].label == "Fingertip to palm"


def test_skeleton_margin_on_a_wide_hand() -> None:
    hand = np.zeros((21, 3))
    hand[:, 0] = np.linspace(0.0, 0.1, 21)
    (pts,) = A.skeleton_points([hand], 100, 100, margin=10)
    assert pts[0] == pytest.approx((10.0, 50.0), abs=1e-3) and pts[-1] == pytest.approx((90.0, 50.0), abs=1e-3)
    (dflt,) = A.skeleton_points([hand], 100, 100)
    assert dflt[0][0] == pytest.approx(10.0, abs=1e-3)
    (tiny,) = A.skeleton_points([hand], 4, 4, margin=10)  # canvas smaller than the margins: still 1 px of room
    assert tiny[-1][0] - tiny[0][0] == pytest.approx(1.0, abs=1e-3)


def test_balanced_accuracy_negative_on_the_threshold_stays_quiet() -> None:
    assert A.balanced_accuracy([0.1, 0.2], [0.3, 0.6], 0.3) == 1.0


def test_choose_threshold_edge_candidates() -> None:
    same = A.choose_threshold([0.1, 0.1, 0.1], [0.1, 0.1, 0.1])
    assert same.threshold == pytest.approx(0.099)  # one value: a 0.001 step below it
    spread = A.choose_threshold([0.3, 0.3, 0.3], [0.1, 0.1, 0.3])
    assert spread.threshold == pytest.approx(0.09)  # 5 % of the 0.2 span below the smallest value


def test_new_session_id_uses_the_given_time() -> None:
    import time

    sid = A.new_session_id("pinch", now=86400.0 * 365)
    assert time.strftime("%Y%m%d-%H%M%S", time.localtime(86400.0 * 365)) in sid


def test_save_clips_defaults_and_five_frame_minimum(tmp_path: Path) -> None:
    s = A.AutoTuneSession(A.AUTOTUNE_SPECS["pinch"], count=3)
    for p in (1.0, 0.95, 0.9):
        s.add_trial(_trial(pinch_hand(p), n=5))
    for p in (0.0, 0.1, 0.2):
        s.add_trial(_trial(pinch_hand(p), n=5))
    paths = s.save_clips(tmp_path)
    assert len(paths) == 6
    with np.load(paths[0]) as d:
        assert float(d["fps"]) == 30.0 and str(d["handedness"]) == "Right"
