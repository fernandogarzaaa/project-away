"""v0.1.20 smart pointer: magnetic targets, look + say clicking, predictive smoothing (away.pointer).

UI Automation is mocked everywhere except the Windows-only smoke test at the bottom, which talks to the real
UIAutomationCore on a Windows CI runner.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import pytest

from away import config as C
from away.app import tunables as T
from away.app.overlay import status_text
from away.app.runner import Pipeline
from away.app.settings_gui import SettingsModel
from away.bindings import NONE, Action
from away.control.mouse import RecordingBackend
from away.events import Event
from away.hands.cursor import CursorMapper
from away.hands.filters import OneEuro2D
from away.hands.gestures import GestureEvent
from away.pointer import bench as B
from away.pointer import looksay as L
from away.pointer import settings_ui as S
from away.pointer import uia as U
from away.pointer.badges import BadgeOverlay, badge_specs
from away.pointer.magnet import Magnet, pull_vector
from away.pointer.smart import FRIENDLY_REFRESH_ERROR, POINTER_ACTION, PointerPhrases
from away.pointer.smoothing import PredictiveFilter2D, make_cursor_filter
from away.pointer.targets import NullBackend, Snapshot, Target, TargetCache, clean_targets
from away.voice.always_on import decide

POINTER_PATHS = {p for p, *_ in C.pointer_tunables(C.AppConfig())}


# ====================================================================================== fixtures
class FakeBackend:
    """Stands in for UI Automation: a fixed list of targets, focus recorded."""

    def __init__(self, targets: list[Target] | None = None, fail: Exception | None = None) -> None:
        self.targets = list(targets or [])
        self.fail = fail
        self.calls = 0
        self.focused: list[str] = []
        self.focus_error: Exception | None = None

    def available(self) -> bool:
        return True

    def foreground_targets(self) -> list[Target]:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return list(self.targets)

    def focus(self, target: Target) -> bool:
        if self.focus_error is not None:
            raise self.focus_error
        self.focused.append(target.name)
        return True


def btn(name: str, left: float, top: float, w: float = 80, h: float = 30, kind: str = "button") -> Target:
    return Target(name, kind, (left, top, left + w, top + h))


WINDOW = [
    btn("Save", 100, 100),
    btn("Save as", 300, 100),
    btn("Cancel", 500, 100),
    btn("OK", 100, 400),
    btn("OK", 700, 400),
    btn("Photos", 900, 600, kind="listitem"),
    btn("Search", 100, 700, w=400, kind="edit"),
]


def magnet_cfg(**kw: Any) -> C.MagnetConfig:
    return C.MagnetConfig(**kw)


def pipeline(targets: list[Target] | None = None, **cfg_kw: Any) -> tuple[Pipeline, RecordingBackend, FakeBackend]:
    cfg = C.AppConfig()
    for k, v in cfg_kw.items():
        obj = cfg
        *head, last = k.split("__")
        for part in head:
            obj = getattr(obj, part)
        setattr(obj, last, v)
    rec = RecordingBackend()
    fake = FakeBackend(WINDOW if targets is None else targets)
    p = Pipeline(cfg, rec, (1920, 1080), pointer_backend=fake)
    p.smart.overlay = BadgeOverlay(platform="linux")  # records, never opens a window
    return p, rec, fake


def clicks(rec: RecordingBackend) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in rec.calls if c[0] in ("move", "click")]


# ====================================================================================== config
def test_defaults_and_yaml_round_trip() -> None:
    cfg = C.AppConfig()
    c = cfg.hands.cursor
    assert c.smoothing == "adaptive" and c.magnet.enabled and cfg.voice.click_by_name
    assert (c.magnet.radius_px, c.magnet.strength, c.magnet.sticky) == (40.0, 0.35, 0.5)
    assert c.predictive.lookahead_ms == 33.0
    d = C.to_dict(cfg)
    assert d["hands"]["cursor"]["magnet"]["radius_px"] == 40.0 and d["hands"]["cursor"]["smoothing"] == "adaptive"
    new, unknown = C.from_dict(
        {
            "hands": {
                "cursor": {
                    "smoothing": "predictive",
                    "magnet": {"enabled": False, "radius_px": 60, "strength": 0.5, "sticky": 0.2},
                    "predictive": {"lookahead_ms": 50},
                }
            },
            "voice": {"click_by_name": False},
        }
    )
    assert unknown == []
    assert new.hands.cursor.smoothing == "predictive" and not new.hands.cursor.magnet.enabled
    assert new.hands.cursor.magnet.radius_px == 60 and new.hands.cursor.predictive.lookahead_ms == 50
    assert new.voice.click_by_name is False


@pytest.mark.parametrize(
    ("data", "msg"),
    [
        ({"hands": {"cursor": {"smoothing": "kalman"}}}, "hands.cursor.smoothing"),
        ({"hands": {"cursor": {"magnet": {"enabled": "yes"}}}}, "magnet.enabled"),
        ({"hands": {"cursor": {"magnet": {"radius_px": 500}}}}, "magnet.radius_px"),
        ({"hands": {"cursor": {"magnet": {"sticky": 1.0}}}}, "magnet.sticky"),
        ({"hands": {"cursor": {"predictive": {"lookahead_ms": -1}}}}, "lookahead_ms"),
        ({"voice": {"click_by_name": 1}}, "click_by_name"),
    ],
)
def test_invalid_values_are_rejected(data: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        C.from_dict(data)


def test_numbers_are_advanced_tuning_sliders_with_plain_titles(tmp_path: Any) -> None:
    assert len(POINTER_PATHS) == 7
    assert {p for p, *_ in C.tunable_ranges(C.AppConfig())} >= POINTER_PATHS
    model = SettingsModel(C.AppConfig(), tmp_path / "config.yaml")
    by_path = {t.path: t for t in model.tunables()}
    for p in POINTER_PATHS:
        t = by_path[p]
        assert t.group == "Advanced" and not t.title.startswith("hands."), p
        assert t.title.startswith(("Snap to buttons", "Smoother cursor")), p
        assert round(t.default / t.resolution) * t.resolution == pytest.approx(t.default), p  # not snapped
    advanced = {t.path for t in model.tunables() if t.group == "Advanced"}
    assert advanced == POINTER_PATHS  # nothing else moved under Advanced
    for name in T.PRESETS:  # presets and the sensitivity slider leave them alone
        assert not set(T.preset_values(name, T.default_values())) & POINTER_PATHS
    assert not set(T.SENSITIVITY_PATHS) & POINTER_PATHS


def test_per_app_profile_tuning_accepts_the_new_sliders() -> None:
    cfg, _ = C.from_dict(
        {"bindings": {"profiles": {"games": {"apps": ["game.exe"], "tuning": {"hands.cursor.magnet.strength": 0}}}}}
    )
    assert cfg.bindings.profiles["games"].tuning["hands.cursor.magnet.strength"] == 0


# ====================================================================================== smoothing
def test_make_cursor_filter_follows_the_setting() -> None:
    from away.hands.steady import AdaptiveFilter2D

    c = C.CursorConfig()
    assert isinstance(make_cursor_filter(c), AdaptiveFilter2D)  # v0.1.25 default
    c.smoothing = "one_euro"
    assert isinstance(make_cursor_filter(c), OneEuro2D)
    c.smoothing = "predictive"
    c.predictive.lookahead_ms = 50
    f = make_cursor_filter(c)
    assert isinstance(f, PredictiveFilter2D) and f.lookahead_s == pytest.approx(0.05)
    assert isinstance(CursorMapper((1920, 1080), c).filter, PredictiveFilter2D)


def test_settings_change_switches_the_filter_live() -> None:
    from away.hands.steady import AdaptiveFilter2D

    p, _, _ = pipeline()
    assert isinstance(p.mouse.mapper.filter, AdaptiveFilter2D)
    new = C.AppConfig()
    new.hands.cursor.smoothing = "predictive"
    p.apply_config(new)
    assert isinstance(p.mouse.mapper.filter, PredictiveFilter2D)
    new2 = C.AppConfig()
    new2.hands.cursor.smoothing = "one_euro"
    p.apply_config(new2)
    assert isinstance(p.mouse.mapper.filter, OneEuro2D)


def test_predictive_filter_leads_a_constant_velocity_move_and_rests_without_lead() -> None:
    f = PredictiveFilter2D(lookahead_s=0.033, measurement_noise=0.001)
    v = 0.6  # hand units / s, a brisk move
    out = (0.0, 0.0)
    for i in range(60):
        t = i / 30
        out = f(t, 0.2 + v * t, 0.5)
    true_now = 0.2 + v * (59 / 30)
    assert f.velocity[0] == pytest.approx(v, rel=0.05)
    assert out[0] == pytest.approx(true_now + v * 0.033, abs=0.003)  # predicted one frame ahead
    assert out[1] == pytest.approx(0.5, abs=1e-3)
    still = PredictiveFilter2D()
    for i in range(30):
        o = still(i / 30, 0.4, 0.4)
    assert o == pytest.approx((0.4, 0.4), abs=1e-9)  # no velocity, no lead


def test_predictive_filter_edge_cases() -> None:
    f = PredictiveFilter2D(max_lead=0.01, lookahead_s=0.1)
    assert f(0.0, 0.3, 0.3) == (0.3, 0.3)
    a = f(1 / 30, 0.35, 0.3)
    assert f(1 / 30, 0.9, 0.9) == a  # duplicate timestamp: hold
    assert f(0.0, 0.9, 0.9) == a  # out of order: hold
    for i in range(2, 40):
        out = f(i / 30, 0.3 + 2.0 * i / 30, 0.3)
    assert out[0] - f._ax.x <= 0.01 + 1e-9  # type: ignore[union-attr]  # lead capped
    f(100.0, 0.5, 0.5)  # a long gap does not blow up
    assert all(math.isfinite(v) for v in f(100.04, 0.5, 0.5))
    f.reset()
    assert f.velocity == (0.0, 0.0) and f(5.0, 0.1, 0.2) == (0.1, 0.2)
    with pytest.raises(ValueError):
        PredictiveFilter2D(lookahead_s=-0.1)
    with pytest.raises(ValueError):
        PredictiveFilter2D(measurement_noise=0)


def test_benchmark_predictive_beats_one_euro_on_lag_and_keeps_a_resting_cursor_still() -> None:
    """The synthetic jitter / latency benchmark (1 frame latency, 0.002 landmark jitter, 5 seeds)."""
    res = B.compare(B.default_filters(), seeds=range(5))
    oe, pr = res["one_euro"], res["predictive"]
    assert pr.lag_rmse < 0.3 * oe.lag_rmse  # measured: ~0.0125 vs ~0.103 hand units
    assert pr.still_jitter < 0.002  # small enough for the hold-still lock / dead zone to keep a resting cursor still
    assert oe.still_jitter < 0.002
    ad = res["adaptive"]  # v0.1.25 default: much less lag than One Euro, still steady at rest
    assert ad.lag_rmse < 0.5 * oe.lag_rmse and ad.still_jitter < 0.002 and ad.overshoot < 0.01
    assert pr.overshoot < 0.015  # bounded overshoot at the end of a reach (~1% of the frame)
    assert B.compare(B.default_filters(), seeds=[3]) == B.compare(B.default_filters(), seeds=[3])  # deterministic


def test_benchmark_metrics_measure_what_they_say() -> None:
    traj = B.synthetic(seed=1, jitter=0.0, latency_frames=0, reaches=5)
    perfect = traj.truth.copy()
    assert B.lag_rmse(perfect, traj) == pytest.approx(0.0, abs=1e-12)
    assert B.overshoot(perfect, traj) <= 1e-12
    late = B.synthetic(seed=1, jitter=0.0, latency_frames=1, reaches=5)
    assert B.lag_rmse(late.measured, late) > 0.005  # one frame late is a measurable error
    raw = OneEuro2D(1000.0, 0.0, 1.0)  # effectively unfiltered
    assert B.still_jitter(raw, jitter=0.002) == pytest.approx(0.002 * math.sqrt(2) * math.sqrt(2), rel=0.25)


# ====================================================================================== targets
def test_target_geometry_and_snapshot_ordering() -> None:
    t = btn("A", 100, 100, 50, 20)
    assert t.center == (125, 110) and t.contains(100, 100) and not t.contains(99, 100)
    assert (
        t.distance(125, 110) == 0
        and t.distance(90, 110) == 10
        and t.distance(160, 130) == pytest.approx(math.hypot(10, 10))
    )
    big = btn("List", 0, 0, 500, 500, kind="listitem")
    small = btn("Row", 90, 90, 100, 30, kind="listitem")
    snap = Snapshot((big, small))
    assert snap.nearest(100, 100) is small  # both contain the point: the smaller wins
    assert snap.nearest(1000, 1000, radius=50) is None
    assert [h[1].name for h in snap.near(600, 100, 200)] == ["List"]


def test_clean_targets_drops_degenerate_and_caps() -> None:
    items = [btn("ok", 0, 0), btn("tiny", 0, 0, 2, 2), Target("nan", "button", (math.nan, 0, 10, 10))]
    assert [t.name for t in clean_targets(items)] == ["ok"]
    assert len(clean_targets([btn(str(i), i, 0) for i in range(50)], limit=10)) == 10


def test_cache_refresh_stale_and_errors_reported_once() -> None:
    now = [0.0]
    fake = FakeBackend([btn("Save", 0, 0)])
    errors: list[str] = []
    cache = TargetCache(fake, stale_s=2.0, clock=lambda: now[0], on_error=errors.append)
    assert cache.snapshot.targets == ()
    assert [t.name for t in cache.refresh_now().targets] == ["Save"]
    now[0] = 1.0
    assert cache.snapshot.targets  # fresh
    fake.fail = RuntimeError("COM error 0x80040201")
    snap = cache.refresh_now()
    assert snap.error.startswith("COM error") and snap.targets  # the last good targets are kept for a while
    cache.refresh_now()
    assert errors == ["COM error 0x80040201"]  # reported once, not every refresh
    now[0] = 3.5
    assert cache.snapshot.targets == ()  # stale: never pull toward a window that went away
    fake.fail = None
    cache.refresh_now()
    fake.fail = RuntimeError("again")
    cache.refresh_now()
    assert len(errors) == 2  # a new failure after a recovery is reported again


def test_cache_thread_refreshes_in_the_background_and_stops() -> None:
    fake = FakeBackend([btn("Save", 0, 0)])
    cache = TargetCache(fake, period_s=0.01)
    cache.start()
    try:
        deadline = time.monotonic() + 5
        while cache.refreshes < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cache.refreshes >= 3 and cache.running
    finally:
        cache.stop()
    assert not cache.running
    n = cache.refreshes
    time.sleep(0.05)
    assert cache.refreshes == n
    off = TargetCache(NullBackend())
    off.start()
    assert not off.running  # nothing to read: no thread at all


def test_cache_thread_skips_work_while_both_features_are_off() -> None:
    fake = FakeBackend([btn("Save", 0, 0)])
    cache = TargetCache(fake, period_s=0.01, enabled=lambda: False)
    cache.start()
    time.sleep(0.08)
    cache.stop()
    assert fake.calls == 0


def test_cursor_reads_never_wait_for_a_slow_refresh() -> None:
    gate = threading.Event()

    class Slow(FakeBackend):
        def foreground_targets(self) -> list[Target]:
            gate.wait(5)
            return [btn("Save", 0, 0)]

    cache = TargetCache(Slow(), period_s=0.01)
    cache.start()
    try:
        time.sleep(0.05)  # the thread is now stuck inside the "COM call"
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = cache.snapshot
        assert time.perf_counter() - t0 < 0.5
    finally:
        gate.set()
        cache.stop()


# ====================================================================================== magnet
def test_pull_is_bounded_continuous_and_fades_with_distance() -> None:
    t = btn("OK", 100, 100, 40, 20)  # centre (120, 110)
    r, s = 40.0, 0.35
    assert pull_vector(300, 110, t, r, s) == (0.0, 0.0)  # beyond the radius
    near = pull_vector(95, 110, t, r, s)
    far = pull_vector(75, 110, t, r, s)
    assert near[0] > far[0] > 0 and near[1] == pytest.approx(0)
    edge_out = pull_vector(99.999, 110, t, r, s)
    edge_in = pull_vector(100.001, 110, t, r, s)
    assert edge_out[0] == pytest.approx(edge_in[0], abs=0.05)  # no jump at the edge
    assert math.hypot(*pull_vector(60.5, 50.5, t, 200, 1.0)) <= 200 * 1.0 + 1e-9
    box = btn("Search", 0, 0, 800, 40, kind="edit")
    assert pull_vector(300, 20, box, r, s) == (0.0, 0.0)  # along the middle of a wide text box: no pull
    assert pull_vector(300, 3, box, r, s)[0] == 0.0 < pull_vector(300, 3, box, r, s)[1]  # only steadied across it
    assert pull_vector(95, 110, t, r, 0.0) == (0.0, 0.0)


def make_magnet(targets: list[Target], **kw: Any) -> tuple[Magnet, C.MagnetConfig]:
    cfg = magnet_cfg(**kw)
    snap = Snapshot(tuple(targets))
    return Magnet(lambda: cfg, lambda: snap), cfg


def test_magnet_is_identity_without_targets_or_when_off() -> None:
    m, cfg = make_magnet([])
    assert m.apply(500, 500) == (500, 500)
    m2, cfg2 = make_magnet([btn("OK", 480, 480)], enabled=False)
    assert m2.apply(490, 490) == (490, 490)


def test_magnet_pulls_toward_a_nearby_button() -> None:
    m, _ = make_magnet([btn("OK", 100, 100, 40, 20)])
    x, y = m.apply(90, 110)  # 10 px left of the button
    assert 90 < x < 120 and y == pytest.approx(110)
    assert m.last_target is not None and m.last_target.name == "OK"


def test_sticky_friction_cancels_jitter_over_a_button() -> None:
    rng = np.random.default_rng(0)
    target = btn("Send", 400, 400, 60, 30)  # centre (430, 415)
    m, _ = make_magnet([target], sticky=0.6, strength=0.0)  # friction only
    raw = [(430 + dx, 415 + dy) for dx, dy in rng.normal(0, 4, size=(300, 2))]
    out = np.array([m.apply(x, y) for x, y in raw])
    raw_a = np.array(raw)
    jit_raw = np.std(np.diff(raw_a, axis=0))
    jit_out = np.std(np.diff(out, axis=0))
    assert jit_out < 0.6 * jit_raw  # frame-to-frame shake over the button drops sharply
    assert max(math.hypot(*o) for o in [(a - b) for a, b in zip(out, raw_a)]) <= 60 + 1e-6  # offset capped at radius


def test_magnet_keeps_a_jittery_hover_on_the_button() -> None:
    """Webcam jitter while hovering at the edge of a small button: more frames land on it with the magnet."""
    rng = np.random.default_rng(1)
    target = btn("x", 500, 500, 24, 24)
    m, _ = make_magnet([target])
    raw = [(498 + dx, 512 + dy) for dx, dy in rng.normal(0, 5, size=(400, 2))]  # hovering at the left edge
    on_raw = sum(target.contains(x, y) for x, y in raw)
    on_mag = sum(target.contains(*m.apply(x, y)) for x, y in raw)
    assert on_mag > on_raw * 1.5


def test_deliberate_moves_leave_freely_and_the_offset_bleeds_away() -> None:
    target = btn("OK", 100, 100, 40, 20)
    m, _ = make_magnet([target], sticky=0.9, strength=0.0)
    for i in range(20):
        m.apply(120 + (i % 2), 110)  # small wiggles over the button build a little offset
    x = 120.0
    for _ in range(10):
        x += 50  # fast moves (>= radius per frame) are not slowed at all
        out = m.apply(x, 110)
    assert out[0] == pytest.approx(x, abs=1.0) and m.offset == pytest.approx((0.0, 0.0), abs=1.0)
    m.apply(5000, 110)  # a warp (monitor jump) resets everything
    assert m.offset == (0.0, 0.0)


def test_resync_after_a_direct_placement_does_not_shift_the_cursor() -> None:
    target = btn("OK", 100, 100, 40, 20)
    m, _ = make_magnet([target])
    m.resync((105.0, 110.0))
    out = m.apply(105.0, 110.0)
    assert out == pytest.approx((105.0, 110.0), abs=0.5)


def test_mouse_controller_runs_the_magnet_and_records_the_shown_position() -> None:
    p, rec, _ = pipeline([btn("OK", 900, 500, 40, 20)])
    p.smart.cache.refresh_now()
    mapper = p.mouse.mapper
    mapper.pos = (895.0, 510.0)
    raw = p.smart.refine(895, 510)
    assert raw[0] > 895  # pulled right toward the button
    p.mouse.handle(GestureEvent("move", 0.0, {"x": 0.5, "y": 0.5}))  # first frame anchors (relative mode)
    p.mouse.handle(GestureEvent("move", 1 / 30, {"x": 0.5, "y": 0.5}))
    moves = [c for c in rec.calls if c[0] == "move"]
    assert moves and moves[-1][1][0] > round(mapper.pos[0])  # the OS cursor shows the pulled position
    st = p.mouse.stabilizer
    assert st is not None and st._hist[-1][1] == pytest.approx(moves[-1][1])  # rewind replays what was shown


def test_without_ui_automation_the_cursor_is_untouched() -> None:
    cfg = C.AppConfig()
    rec = RecordingBackend()
    p = Pipeline(cfg, rec)  # no pointer backend: NullBackend
    assert not p.smart.available and p.smart.refine(10, 20) == (10, 20)
    assert p.fire("voice", "click save") is NONE
    assert p.voice_phrases() == cfg.bindings.voice_phrases("default")


# ====================================================================================== look + say: parsing
@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("click save", L.Command("click", "save")),
        ("Click Save!", L.Command("click", "save")),
        ("click on the send button", L.Command("click", "send button")),
        ("double click photos", L.Command("double_click", "photos")),
        ("double-click photos", L.Command("double_click", "photos")),
        ("focus search", L.Command("focus", "search")),
        ("click that", L.Command("click", that=True)),
        ("click this one", L.Command("click", that=True)),
        ("double click that", L.Command("double_click", that=True)),
        ("click", None),
        ("click the", None),
        ("scroll down", None),
        ("two", None),
    ],
)
def test_parse(phrase: str, expected: L.Command | None) -> None:
    assert L.parse(phrase) == expected


@pytest.mark.parametrize(
    ("phrase", "n"),
    [
        ("2", 2),
        ("two", 2),
        ("to", 2),
        ("number three", 3),
        ("pick 4", 4),
        ("the second one", 2),
        ("one", 1),
        ("click 5", 5),
        ("10", 0),
        ("zero", 0),
        ("hello", 0),
    ],
)
def test_parse_number(phrase: str, n: int) -> None:
    assert L.parse_number(phrase) == n


def test_numbers_and_cancel_only_count_while_choosing() -> None:
    assert L.parse("two", choosing=True) == L.Command("choose", number=2)
    assert L.parse("cancel", choosing=True) == L.Command("cancel")
    assert L.parse("cancel") is None
    assert L.parse("click save", choosing=True) == L.Command("click", "save")


def test_name_scores() -> None:
    assert L.name_score("save", "Save") == 1.0
    assert L.name_score("send", "Send message") == 0.92
    assert L.name_score("message", "Send message") == 0.85
    assert L.name_score("sand", "Send message") < L.name_score("send", "Send message")
    assert L.name_score("setings", "Settings") > L.MATCH_THRESHOLD  # recogniser typos still match
    assert L.name_score("photos", "Cancel") < L.MATCH_THRESHOLD
    assert L.name_score("", "x") == 0.0


def test_match_names_unique_ambiguous_and_missing() -> None:
    assert [m.target.name for m in L.match_names("cancel", WINDOW)] == ["Cancel"]
    assert [m.target.name for m in L.match_names("save", WINDOW)] == ["Save"]  # exact beats "Save as"
    oks = L.match_names("ok", WINDOW)
    assert [m.target.rect[0] for m in oks] == [100, 700]  # two OK buttons: ambiguous, top-left first
    assert L.match_names("printer", WINDOW) == []
    twins = [btn("Next", 10, 10), Target("Next", "button", (11, 10, 91, 40))]  # same element reported twice
    assert len(L.match_names("next", twins)) == 1
    many = [btn("Item", i * 100, 0) for i in range(20)]
    assert len(L.match_names("item", many)) == L.MAX_CHOICES


def test_chooser_times_out() -> None:
    ch = L.Chooser(timeout_s=10)
    ch.offer("click", WINDOW[:3], now=0.0)
    assert ch.active(5.0) and ch.choose(4, 5.0) is None and ch.active(5.0)  # out of range keeps it open
    assert ch.choose(2, 6.0) == ("click", WINDOW[1]) and not ch.active(6.0)
    ch.offer("click", WINDOW[:3], now=0.0)
    assert not ch.active(11.0) and ch.choose(1, 11.0) is None


def test_badges_do_not_overlap() -> None:
    two = [btn("OK", 100, 100), btn("OK", 105, 102)]
    specs = badge_specs(two)
    assert [b.number for b in specs] == [1, 2]
    assert abs(specs[0].x - specs[1].x) >= 26 or abs(specs[0].y - specs[1].y) >= 26
    assert badge_specs([btn("A very long button name that goes on and on", 0, 0)])[0].label.endswith("…")


# ====================================================================================== look + say: pipeline
def test_click_by_name_moves_there_and_clicks() -> None:
    p, rec, _ = pipeline()
    act = p.fire("voice", "click cancel")
    assert act.kind == POINTER_ACTION
    assert clicks(rec) == [("move", (540, 115)), ("click", (1,))]
    assert p.mouse.mapper.pos == (540.0, 115.0)  # the hand is re-anchored there
    assert p.smart.last_message == "Clicked Cancel."


def test_double_click_and_focus() -> None:
    p, rec, fake = pipeline()
    p.fire("voice", "double click photos")
    assert clicks(rec)[-1] == ("click", (2,))
    rec.calls.clear()
    p.fire("voice", "focus search")
    assert fake.focused == ["Search"] and clicks(rec) == []  # UIA focus, no mouse click
    fake.focus_error = RuntimeError("element refuses focus")
    p.fire("voice", "focus search")
    assert clicks(rec) == [("move", (300, 715)), ("click", (1,))]  # falls back to a click


def test_ambiguous_name_shows_numbered_badges_then_a_number_picks() -> None:
    p, rec, _ = pipeline()
    events: list[Event] = []
    p.bus.subscribe("pointer.choices", events.append)
    p.fire("voice", "click ok")
    assert clicks(rec) == []
    assert [b.number for b in p.smart.overlay.shown] == [1, 2]
    assert p.smart.last_message == "Which one? Say a number from 1 to 2."
    assert any(e.topic == "pointer.choices" and e.data["names"] == ["OK", "OK"] for e in events)
    assert "two" in p.voice_phrases()  # always-on listening now hears a bare number
    p.fire("voice", "number seven")
    assert clicks(rec) == [] and "1 to 2" in p.smart.last_message
    p.fire("voice", "two")
    assert clicks(rec) == [("move", (740, 415)), ("click", (1,))]
    assert p.smart.overlay.shown == [] and "two" not in p.voice_phrases()


def test_cancel_clears_the_badges() -> None:
    p, rec, _ = pipeline()
    p.fire("voice", "click ok")
    p.fire("voice", "never mind")
    assert p.smart.overlay.shown == [] and clicks(rec) == [] and p.smart.last_message == "Okay, cancelled."


def test_click_that_uses_the_gaze_point_or_the_cursor() -> None:
    p, rec, _ = pipeline()
    p.mouse.gaze_point = lambda: (950.0, 650.0)  # looking at Photos
    p.fire("voice", "click that")
    assert clicks(rec)[-1] == ("click", (1,)) and clicks(rec)[0] == ("move", (940, 615))
    rec.calls.clear()
    p.mouse.gaze_point = lambda: None  # not calibrated / no face: the cursor position
    p.mouse.mapper.pos = (150.0, 130.0)  # just below Save
    p.fire("voice", "click that")
    assert clicks(rec)[0] == ("move", (140, 115))
    rec.calls.clear()
    p.mouse.mapper.pos = (1800.0, 50.0)  # nothing near
    act = p.fire("voice", "click that")
    assert act.kind == POINTER_ACTION and clicks(rec) == []
    assert p.smart.last_message == "There's nothing to click right there."


def test_unknown_names_are_explained_and_focus_falls_back_to_dictation() -> None:
    p, rec, _ = pipeline()
    act = p.fire("voice", "click printer settings")
    assert act.kind == POINTER_ACTION and clicks(rec) == []  # consumed (never typed) with a friendly message
    assert p.smart.last_message == "I couldn't find “printer settings” in this window."
    assert p.fire("voice", "focus on the budget") is NONE  # not a command after all: dictation types it


def test_bound_phrases_win_and_paused_or_switched_off_does_nothing() -> None:
    p, rec, _ = pipeline()
    assert p.fire("voice", "click").kind == "mouse"  # the existing bare "click" binding
    p.cfg.bindings = p.cfg.bindings  # bind "click save" to a key: the binding wins over the smart pointer
    from away.bindings import parse_bindings

    b = parse_bindings({"profiles": {"default": {"voice": {"click save": "key:ctrl+s"}}}}, "bindings")
    p.dispatcher.set_bindings(b)
    p.cfg.bindings = b
    rec.calls.clear()
    assert p.fire("voice", "click save").kind == "key"
    assert [c for c in rec.calls if c[0] == "click"] == []
    q, rec2, _ = pipeline()
    q.set_paused(True)
    assert q.fire("voice", "click cancel") is NONE and clicks(rec2) == []
    r, rec3, _ = pipeline(voice__click_by_name=False)
    assert r.fire("voice", "click cancel") is NONE and clicks(rec3) == []
    assert not isinstance(r.voice_phrases(), PointerPhrases)


def test_always_on_listening_recognises_the_commands() -> None:
    p, _, _ = pipeline()
    table = p.voice_phrases()
    assert isinstance(table, PointerPhrases)
    assert "scroll down" in table and table.get("scroll down") == p.cfg.bindings.voice_phrases()["scroll down"]
    assert "click save" in table and table.get("click save") == Action(POINTER_ACTION, "click save")
    assert "focus search" in table or p.smart.cache.snapshot.targets == ()  # focus needs a match in the cache
    p.smart.cache.refresh_now()
    assert "focus search" in table and "focus on the budget" not in table
    assert table.get("hello there") is None
    d = decide("Click Save.", commands=table)
    assert d.kind == "command" and d.action is not None and d.action.kind == POINTER_ACTION
    d2 = decide("focus on the budget please", commands=table, gate=True)
    assert d2.kind == "type"  # ordinary dictation is still typed


def test_refresh_errors_are_friendly() -> None:
    p, _, fake = pipeline()
    events: list[Event] = []
    notes: list[tuple[str, str, str]] = []
    p.bus.subscribe("pointer.status", events.append)
    p.smart.notify = lambda title, msg, key: notes.append((title, msg, key))
    fake.fail = OSError("(-2147220991, 'An event was unable to invoke any of the subscribers')")
    act = p.fire("voice", "click save")
    assert act.kind == POINTER_ACTION and p.smart.last_message.startswith("I couldn't find")
    assert events and events[0].data["text"] == FRIENDLY_REFRESH_ERROR and "subscribers" in events[0].data["detail"]
    assert notes == [("Project Away: snap to buttons", FRIENDLY_REFRESH_ERROR, "pointer-uia-error")]
    assert "0x" not in FRIENDLY_REFRESH_ERROR and "COM" not in FRIENDLY_REFRESH_ERROR


def test_overlay_shows_the_spoken_feedback() -> None:
    assert status_text(Event("pointer.say", {"text": "Clicked Save."}, 0.0)) == "Clicked Save."


# ====================================================================================== settings (plain words)
def test_three_plain_toggles() -> None:
    assert [t.label for t in S.TOGGLES] == [
        "Snap to buttons",
        "Click things by saying their name",
        "Smoother cursor",
    ]
    for t in S.TOGGLES:
        assert "." not in t.label and "_" not in t.label and t.help
        assert not any(w in t.help.lower() for w in ("uia", "kalman", "comtypes", "px"))


def test_toggles_drive_the_config_and_validate() -> None:
    cfg = C.AppConfig()
    assert S.toggle_on(cfg, "snap") and S.toggle_on(cfg, "say") and not S.toggle_on(cfg, "smooth")
    S.set_toggle(cfg, "smooth", True)
    S.set_toggle(cfg, "snap", False)
    S.set_toggle(cfg, "say", False)
    C.validate(cfg)
    assert cfg.hands.cursor.smoothing == "predictive" and not cfg.hands.cursor.magnet.enabled
    assert not cfg.voice.click_by_name
    S.set_toggle(cfg, "smooth", False)
    assert cfg.hands.cursor.smoothing == "adaptive"
    var = S._ToggleVar(
        types.SimpleNamespace(v=None, set=lambda x: setattr(var_store, "v", x), get=lambda: True), "smooth"
    )
    var.set("predictive")
    assert var_store.v is True
    var.set("adaptive")
    assert var_store.v is False


var_store = types.SimpleNamespace(v=None)


def test_health_messages_and_fix_buttons() -> None:
    ok = S.health("ok")
    assert ok.ok and not ok.fix
    mac = S.health("not_windows")
    assert mac.ok and "Windows" in mac.message and not mac.fix
    pip = S.health("missing_comtypes", frozen=False)
    assert not pip.ok and pip.fix == "install_comtypes" and pip.fix_label == "Install"
    frozen = S.health("missing_comtypes", frozen=True)
    assert not frozen.ok and frozen.fix == "open_download" and "Reinstall" in frozen.message


@pytest.mark.parametrize(("code", "ok"), [(0, True), (1, False)])
def test_install_fix_reports_in_plain_words(code: int, ok: bool) -> None:
    got: list[tuple[bool, str]] = []
    ran: list[list[str]] = []
    done = threading.Event()

    def runner(cmd: list[str], **_kw: Any) -> Any:
        ran.append(cmd)
        return types.SimpleNamespace(returncode=code)

    S.run_fix("install_comtypes", lambda a, b: (got.append((a, b)), done.set()), runner=runner)
    assert done.wait(5)
    assert ran[0][-1].startswith("comtypes") and "pip" in ran[0]
    assert got[0][0] is ok and ("Restart" in got[0][1] if ok else "didn't work" in got[0][1])


# ====================================================================================== UI Automation (mocked COM)
def test_availability() -> None:
    assert U.availability("linux") == "not_windows" and U.availability("darwin") == "not_windows"
    assert isinstance(U.make_backend("linux"), NullBackend)


class FakeRect:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self.left, self.top, self.right, self.bottom = left, top, right, bottom


class FakeElement:
    def __init__(self, name: Any, ct: int, rect: Any) -> None:
        self.CachedName, self.CachedControlType, self.CachedBoundingRectangle = name, ct, rect
        self.focused = False

    def SetFocus(self) -> None:
        self.focused = True


class FakeArray:
    def __init__(self, items: list[Any]) -> None:
        self.items = items
        self.Length = len(items)

    def GetElement(self, i: int) -> Any:
        return self.items[i]


class Vanishing:
    @property
    def CachedControlType(self) -> int:
        raise OSError("element gone")


def test_targets_from_array_reads_cached_properties() -> None:
    arr = FakeArray(
        [
            FakeElement(" Save ", 50000, FakeRect(10, 20, 90, 50)),
            FakeElement(None, 50005, (0, 0, 40, 10)),  # a link without a name, rect as a tuple
            FakeElement("Pane", 50033, FakeRect(0, 0, 999, 999)),  # not clickable: skipped
            Vanishing(),
        ]
    )
    got = U.targets_from_array(arr)
    assert [(t.name, t.kind, t.rect) for t in got] == [
        ("Save", "button", (10.0, 20.0, 90.0, 50.0)),
        ("", "hyperlink", (0.0, 0.0, 40.0, 10.0)),
    ]
    assert len(U.targets_from_array(FakeArray([FakeElement("x", 50000, (0, 0, 9, 9))] * 10), limit=3)) == 3


def install_fake_comtypes(monkeypatch: pytest.MonkeyPatch, root: Any) -> dict[str, Any]:
    log: dict[str, Any] = {"init": 0, "conds": [], "props": []}

    class FakeUIA:
        def CreatePropertyCondition(self, prop: int, value: Any) -> Any:
            log["conds"].append((prop, value))
            return ("cond", prop, value)

        def CreateOrConditionFromArray(self, conds: list[Any]) -> Any:
            return ("or", len(conds))

        def CreateAndCondition(self, a: Any, b: Any) -> Any:
            return ("and", a, b)

        def CreateCacheRequest(self) -> Any:
            return types.SimpleNamespace(AddProperty=lambda p: log["props"].append(p))

        def ElementFromHandle(self, hwnd: int) -> Any:
            log["hwnd"] = hwnd
            return root

    comtypes = types.ModuleType("comtypes")
    comtypes.COINIT_MULTITHREADED = 0  # type: ignore[attr-defined]

    def co_init(_flag: int) -> None:
        log["init"] += 1

    comtypes.CoInitializeEx = co_init  # type: ignore[attr-defined]
    client = types.ModuleType("comtypes.client")
    mod = types.SimpleNamespace(CUIAutomation="CLSID", IUIAutomation="IID")
    client.GetModule = lambda name: (log.__setitem__("module", name), mod)[1]  # type: ignore[attr-defined]
    client.CreateObject = lambda clsid, interface=None: FakeUIA()  # type: ignore[attr-defined]
    comtypes.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comtypes", comtypes)
    monkeypatch.setitem(sys.modules, "comtypes.client", client)
    return log


def test_uia_backend_with_mocked_comtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    save = FakeElement("Save", 50000, FakeRect(10, 20, 90, 50))
    calls: list[Any] = []

    class Root:
        def FindAllBuildCache(self, scope: int, cond: Any, cache: Any) -> Any:
            calls.append((scope, cond))
            return FakeArray([save])

    log = install_fake_comtypes(monkeypatch, Root())
    be = U.UIABackend(foreground=lambda: 0x1234)
    got = be.foreground_targets()
    assert [(t.name, t.kind) for t in got] == [("Save", "button")]
    assert log["module"] == "UIAutomationCore.dll" and log["hwnd"] == 0x1234
    assert calls[0][0] == U.TREE_SCOPE_DESCENDANTS and calls[0][1][0] == "and"
    assert {ct for p, ct in log["conds"] if p == U.UIA_CONTROL_TYPE} == set(U.CONTROL_TYPES)
    assert (U.UIA_IS_OFFSCREEN, False) in log["conds"]
    assert log["props"] == [U.UIA_NAME, U.UIA_CONTROL_TYPE, U.UIA_BOUNDING_RECTANGLE]
    be.foreground_targets()
    assert log["init"] == 1  # COM joined once per thread, the automation object created once
    assert be.focus(got[0]) and save.focused
    assert not be.focus(btn("no ref", 0, 0))
    assert U.UIABackend(foreground=lambda: 0).foreground_targets() == []  # no foreground window


def test_uia_backend_threads_each_join_com(monkeypatch: pytest.MonkeyPatch) -> None:
    class Root:
        def FindAllBuildCache(self, *_a: Any) -> Any:
            return FakeArray([])

    log = install_fake_comtypes(monkeypatch, Root())
    be = U.UIABackend(foreground=lambda: 1)
    be.foreground_targets()
    th = threading.Thread(target=be.foreground_targets)
    th.start()
    th.join()
    assert log["init"] == 2


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="real UI Automation needs Windows")
def test_real_ui_automation_smoke() -> None:  # pragma: no cover - Windows CI only
    """Talks to the real UIAutomationCore: creates the automation object and walks the foreground window."""
    pytest.importorskip("comtypes")
    be = U.UIABackend()
    assert be.available()
    be.thread_init()  # CoInitializeEx + CUIAutomation + conditions + cache request
    try:
        got = be.foreground_targets()
    except OSError as exc:  # a service session without a desktop has no foreground window to walk
        pytest.skip(f"no interactive desktop on this runner: {exc}")
    assert isinstance(got, list) and all(isinstance(t, Target) for t in got)


def test_spatial_index_keeps_the_per_frame_magnet_cheap_on_a_huge_page() -> None:
    rng = np.random.default_rng(2)
    many = tuple(btn(str(i), float(x), float(y)) for i, (x, y) in enumerate(rng.uniform(0, 1800, size=(2000, 2))))
    giant = Target("Document", "dataitem", (-1e9, -1e9, 1e9, 1e9))  # absurd rectangles do not explode the index
    snap = Snapshot((*many, giant))
    cfg = magnet_cfg()
    m = Magnet(lambda: cfg, lambda: snap)
    t0 = time.perf_counter()
    for i in range(200):
        m.apply(500.0 + i % 7, 500.0)
    per_frame = (time.perf_counter() - t0) / 200
    assert per_frame < 0.002  # measured ~0.08 ms; the full scan it replaced took ~7 ms
    # the index finds exactly what a full scan finds
    for x, y in rng.uniform(0, 1900, size=(50, 2)):
        full = sorted((t.distance(x, y), t.width * t.height, t.name) for t in snap.targets if t.distance(x, y) <= 40)
        fast = [(d, t.width * t.height, t.name) for d, t in snap.near(float(x), float(y), 40)]
        assert sorted(fast) == full


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and __import__("os").environ.get("DISPLAY")),
    reason="needs an X display (Xvfb)",
)
def test_settings_window_shows_the_three_switches_and_they_work(tmp_path: Any) -> None:
    tk = pytest.importorskip("tkinter")
    from away.app import settings_gui as G

    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - no Tk
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        w = G.SettingsWindow(root, G.SettingsModel(path=tmp_path / "config.yaml"))
        root.update()

        def walk(widget: Any) -> list[Any]:
            out = [widget]
            for c in widget.winfo_children():
                out += walk(c)
            return out

        frames = [x for x in walk(root) if x.winfo_class() == "TLabelframe" and x.cget("text") == "Smart pointer"]
        assert len(frames) == 1
        checks = {x.cget("text"): x for x in walk(frames[0]) if x.winfo_class() == "TCheckbutton"}
        assert list(checks) == ["Snap to buttons", "Click things by saying their name", "Smoother cursor"]
        labels = [x for x in walk(frames[0]) if x.winfo_class() == "TLabel"]
        # the status line is a textvariable label, so read the variable too (cget("text") is empty there)
        texts = " ".join(
            str(root.getvar(str(x.cget("textvariable")))) if str(x.cget("textvariable")) else str(x.cget("text"))
            for x in labels
        )
        assert "need Windows" in texts and "Tuning tab > Advanced" in texts  # Linux: a calm, plain explanation
        assert checks["Snap to buttons"].instate(["disabled"])  # Windows-only switches are greyed out here
        smooth = checks["Smoother cursor"]
        smooth.invoke()
        assert w.m.cfg.hands.cursor.smoothing == "predictive" and w.m.dirty
        w.m.revert()
        w.refresh()  # Settings' own refresh drives the switch through the adapter
        root.update()
        assert w.m.cfg.hands.cursor.smoothing == "adaptive" and not smooth.instate(["selected"])
    finally:
        root.destroy()
