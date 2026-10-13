"""Settings > Tuning redesign: grouping, plain-language metadata, presets, master sensitivity, undo, per-app profile
tuning (bindings + config + live runtime) and a headless-safe smoke test of the tab."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

from away import config as C
from away.app import tunables as T
from away.app.settings_gui import SettingsModel, Tunable
from away.bindings import PROFILE_TUNING_PREFIXES, BindingsError, merge_bindings, parse_bindings

DEFAULTS = T.default_values()
BOUNDS = T.tunable_bounds()


@pytest.fixture
def model(tmp_path: Path) -> SettingsModel:
    return SettingsModel(C.AppConfig(), tmp_path / "config.yaml")


@pytest.fixture
def ctl(model: SettingsModel) -> T.TuningController:
    clock = iter(float(i) * 10 for i in range(10_000))  # 10 s apart: nothing coalesces unless a test wants it
    return T.TuningController(model, clock=lambda: next(clock))


# ===================================================================================== grouping
def test_every_described_path_is_a_real_tunable_in_a_known_group() -> None:
    paths = {p for p, *_ in C.tunable_ranges(C.AppConfig())}
    assert set(T.TUNABLE_META) <= paths
    for path, meta in T.TUNABLE_META.items():
        assert meta.group in T.GROUPS, path
        assert meta.title and not meta.title.startswith(("hands.", "face.", "voice.")), path
        assert "Higher" in meta.help or "Lower" in meta.help or "Rarely" in meta.help, path
        assert T.group_for(path) == meta.group


def test_all_current_tunables_are_described() -> None:
    paths = [p for p, *_ in C.tunable_ranges(C.AppConfig())]
    missing = [p for p in paths if p not in T.TUNABLE_META]
    assert missing == []


@pytest.mark.parametrize(
    ("path", "group"),
    [
        ("hands.cursor.new_thing", "Cursor"),
        ("hands.roles.something", "Cursor"),
        ("hands.dwell.extra", "Clicks"),
        ("hands.gestures.pinch_extra", "Clicks"),
        ("hands.gestures.autoscroll_new", "Scroll"),
        ("hands.gestures.scroll_inertia", "Scroll"),
        ("hands.palm_scroll.new", "Scroll"),
        ("hands.gestures.wave_hold", "Gestures"),
        ("face.smile_threshold", "Face"),
        ("voice.new_knob", "Voice"),
        ("face.head.dead_zone", "Head & eyes"),
        ("face.gaze_gain", "Head & eyes"),
        ("hands.cursor.auto_center_speed", "Head & eyes"),  # keyword beats the hands.cursor. prefix
        ("hands.cursor.autocenter_s", "Head & eyes"),
        ("head.yaw_gain", "Head & eyes"),
        ("eyes.blink_hold", "Head & eyes"),
        ("app.something", "Other"),
        ("performance.whatever", "Other"),
        ("weird", "Other"),
    ],
)
def test_unknown_tunables_land_in_the_right_group(path: str, group: str) -> None:
    assert T.group_for(path) == group


def test_declared_group_is_used_for_unknown_paths_only() -> None:
    assert T.group_for("app.new_knob", "Voice") == "Voice"
    assert T.group_for("app.new_knob", "Not a group") == "Other"
    assert T.group_for("app.new_knob", None) == "Other"
    assert T.group_for("hands.cursor.gain", "Voice") == "Cursor"  # an explicit meta entry wins
    assert T.group_for("face.head.x", "Clicks") == "Clicks"  # declared beats the keyword


def test_longest_prefix_wins() -> None:
    assert T.group_for("hands.gestures.pinch_new") == "Clicks"  # not the shorter "hands.gestures."
    assert T.group_for("hands.gestures.autoscroll_x") == "Scroll"
    assert T.group_for("hands.gesturesX") == "Other"


@pytest.mark.parametrize(
    ("path", "title"),
    [
        ("face.head.dead_zone", "Head dead zone"),
        ("voice.new_knob", "New knob"),
        ("x", "X"),
        ("", ""),
        ("a.b", "B"),
    ],
)
def test_humanize(path: str, title: str) -> None:
    assert T.humanize(path) == title


@pytest.mark.parametrize(
    ("path", "unit"),
    [
        ("face.head.delay_ms", "ms"),
        ("a.radius_px", "px"),
        ("a.min_fps", "fps"),
        ("a.fov_deg", "°"),
        ("a.reach_m", "m"),
        ("a.time_s", "s"),
        ("a.wave_hold", "s"),
        ("a.tap_window", "s"),
        ("a.gain", ""),
        ("hands.cursor.gain", "×"),
        ("hands.dwell.radius_px", "px"),
        ("voice.vad_sensitivity", ""),
    ],
)
def test_unit_for(path: str, unit: str) -> None:
    assert T.unit_for(path) == unit


def test_describe_known_and_unknown() -> None:
    d = T.describe("hands.gestures.pinch_on", "Pinch threshold (close)")
    assert d == T.TUNABLE_META["hands.gestures.pinch_on"]
    u = T.describe("face.head.delay_ms", "Head delay")
    assert (u.group, u.title, u.unit) == ("Head & eyes", "Head delay", "ms")
    assert u.help == "Advanced setting (face.head.delay_ms)."
    assert T.describe("face.head.delay_ms", "face.head.delay_ms").title == "Head delay ms"  # label == path: humanize
    assert T.describe("face.head.delay_ms").title == "Head delay ms"


@pytest.mark.parametrize(
    ("args", "text"),
    [
        ((0.32, "palm"), "0.32 palm"),
        ((800, "ms", True), "800 ms"),
        ((812.6, "ms", True), "813 ms"),
        ((2.2, "×"), "2.2×"),
        ((45.0, "°"), "45°"),
        ((0.5, ""), "0.5"),
        ((0.0, "s"), "0 s"),
        ((-0.0001, ""), "0"),
        ((1.23456, "", False, 0.01), "1.23"),
        ((1.0, "", False, 0.01), "1"),
        ((1.23456, "", False, 0.00001), "1.2346"),
        ((12.5, "", False, 10.0), "12"),
    ],
)
def test_format_value(args: tuple, text: str) -> None:
    assert T.format_value(*args) == text


def test_group_items_order_unknown_and_empty() -> None:
    items = [("a", "Voice"), ("b", "Cursor"), ("c", "Nope"), ("d", "Cursor"), ("e", "Other")]
    out = T.group_items(items, key=lambda it: it[1])
    assert [g for g, _ in out] == ["Cursor", "Voice", "Other"]
    assert [i[0] for i in out[0][1]] == ["b", "d"]
    assert [i[0] for i in out[2][1]] == ["c", "e"]  # unknown -> Other, original order kept
    assert T.group_items([], key=str) == []


def test_model_tunables_carry_group_title_help_unit_default(model: SettingsModel) -> None:
    tun = {t.path: t for t in model.tunables()}
    g = tun["hands.cursor.gain"]
    assert (g.group, g.title, g.unit, g.default) == ("Cursor", "Cursor speed", "×", 2.2)
    assert g.label == "Cursor sensitivity (gain)"  # the old label stays (errors, Voice tab)
    assert tun["hands.palm_scroll.depth_threshold"].group == "Scroll"
    assert tun["voice.end_silence_ms"].is_int and not tun["hands.cursor.gain"].is_int
    assert Tunable("x", "x", 0, 1, 0.5, False).group == "Other"


def test_new_tunable_declared_in_config_appears_automatically(model: SettingsModel, monkeypatch) -> None:
    orig = C.tunable_ranges

    def more(cfg):
        return [
            *orig(cfg),
            ("hands.cursor.monitor_refresh_s", cfg.hands.cursor.monitor_refresh_s, 0.0, 300.0),
            ("hands.multicam.push_time_s", cfg.hands.multicam.push_time_s, 0.05, 2.0),
        ]

    monkeypatch.setattr(C, "tunable_ranges", more)
    T._defaults_cache.cache_clear()
    try:
        ctl = T.TuningController(model)
        groups = dict(ctl.groups())
        cursor = [t.path for t in groups["Cursor"]]
        other = [t.path for t in groups["Other"]]
        assert "hands.cursor.monitor_refresh_s" in cursor and "hands.multicam.push_time_s" in other
        t = next(t for t in groups["Other"] if t.path == "hands.multicam.push_time_s")
        assert t.unit == "s" and t.help.startswith("Advanced setting") and t.default is not None
        assert ctl.set("hands.multicam.push_time_s", 0.5) is None
    finally:
        T._defaults_cache.cache_clear()


def test_short_label() -> None:
    assert T.short_label("abc") == "abc"
    assert T.short_label("x" * 24) == "x" * 24
    assert T.short_label("x" * 25) == "x" * 23 + "…"
    assert T.short_label("  a   b  ") == "a b"
    assert T.short_label("abcdef", 1) == "a…"


def test_cached_defaults_is_a_copy() -> None:
    d = T.cached_defaults()
    d["hands.cursor.gain"] = 99.0
    assert T.cached_defaults()["hands.cursor.gain"] == 2.2


# ===================================================================================== sensitivity
def test_sensitivity_zero_is_exactly_the_defaults() -> None:
    vals = T.sensitivity_values(0.0, DEFAULTS, BOUNDS)
    assert set(vals) == set(T.SENSITIVITY_PATHS)
    for p, v in vals.items():
        assert v == pytest.approx(DEFAULTS[p], abs=1e-12)


def test_sensitivity_extremes_exact() -> None:
    hi = T.sensitivity_values(1.0, DEFAULTS, BOUNDS)
    lo = T.sensitivity_values(-1.0, DEFAULTS, BOUNDS)
    assert hi["hands.cursor.gain"] == pytest.approx(4.4)
    assert lo["hands.cursor.gain"] == pytest.approx(1.1)
    assert hi["hands.cursor.acceleration"] == pytest.approx(1.0 + 0.6 * 1.5)
    assert lo["hands.cursor.acceleration"] == pytest.approx(1.0 + 0.6 * 0.5)
    assert hi["hands.cursor.dead_zone"] == pytest.approx(0.002)
    assert lo["hands.cursor.dead_zone"] == pytest.approx(0.008)
    assert hi["hands.gestures.pinch_on"] == pytest.approx(0.4)
    assert lo["hands.gestures.pinch_on"] == pytest.approx(0.256)
    assert hi["hands.gestures.pinch_off"] == pytest.approx(0.5625)
    half = T.sensitivity_values(0.5, DEFAULTS)
    assert half["hands.cursor.gain"] == pytest.approx(2.2 * math.sqrt(2))
    assert T.sensitivity_values(-0.5, DEFAULTS)["hands.cursor.gain"] == pytest.approx(2.2 * math.sqrt(0.5))


def test_sensitivity_clamps_input_and_bounds() -> None:
    assert T.sensitivity_values(5.0, DEFAULTS) == T.sensitivity_values(1.0, DEFAULTS)
    assert T.sensitivity_values(-5.0, DEFAULTS) == T.sensitivity_values(-1.0, DEFAULTS)
    tight = dict(BOUNDS, **{"hands.cursor.gain": (2.0, 3.0)})
    assert T.sensitivity_values(1.0, DEFAULTS, tight)["hands.cursor.gain"] == 3.0
    assert T.sensitivity_values(-1.0, DEFAULTS, tight)["hands.cursor.gain"] == 2.0
    partial = {"hands.cursor.gain": 1.0}
    assert T.sensitivity_values(1.0, partial) == {"hands.cursor.gain": 2.0}


def test_sensitivity_monotonic_and_keeps_pinch_hysteresis() -> None:
    prev = None
    for i in range(-10, 11):
        v = T.sensitivity_values(i / 10, DEFAULTS, BOUNDS)
        assert v["hands.gestures.pinch_on"] < v["hands.gestures.pinch_off"]
        if prev is not None:
            assert v["hands.cursor.gain"] > prev["hands.cursor.gain"]
            assert v["hands.cursor.dead_zone"] < prev["hands.cursor.dead_zone"]
            assert v["hands.cursor.acceleration"] > prev["hands.cursor.acceleration"]
        prev = v


def test_rule_factor() -> None:
    r = T.SensitivityRule("x", 4.0, 0.25)
    assert T.rule_factor(r, 0.0) == 1.0
    assert T.rule_factor(r, 0.5) == pytest.approx(2.0)
    assert T.rule_factor(r, -0.5) == pytest.approx(0.5)
    assert T.rule_factor(r, 2.0) == pytest.approx(4.0)
    assert T.rule_factor(r, -2.0) == pytest.approx(0.25)


@pytest.mark.parametrize("s", [-1.0, -0.6, -0.25, 0.0, 0.3, 0.75, 1.0])
def test_estimate_sensitivity_inverts_the_mapping(s: float) -> None:
    vals = T.sensitivity_values(s, DEFAULTS, BOUNDS)
    assert T.estimate_sensitivity(vals, DEFAULTS) == pytest.approx(s, abs=1e-9)


def test_estimate_sensitivity_edges() -> None:
    assert T.estimate_sensitivity({}, DEFAULTS) == 0.0
    assert T.estimate_sensitivity({"hands.cursor.gain": 0.0}, DEFAULTS) == 0.0
    assert T.estimate_sensitivity({"hands.cursor.gain": 2.0}, {"hands.cursor.gain": 0.0}) == 0.0
    assert T.estimate_sensitivity({"hands.cursor.gain": 100.0}, DEFAULTS) == 1.0
    assert T.estimate_sensitivity({"hands.cursor.gain": 0.01}, DEFAULTS) == -1.0
    assert T.estimate_sensitivity({"hands.cursor.gain": 2.2}, DEFAULTS) == 0.0


def test_slider_conversion() -> None:
    assert T.slider_to_sensitivity(50) == 0.0
    assert T.slider_to_sensitivity(100) == 1.0
    assert T.slider_to_sensitivity(0) == -1.0
    assert T.slider_to_sensitivity(75) == 0.5
    assert T.slider_to_sensitivity(150) == 1.0 and T.slider_to_sensitivity(-20) == -1.0
    assert T.sensitivity_to_slider(0.0) == 50.0 and T.sensitivity_to_slider(-1.0) == 0.0
    assert T.sensitivity_to_slider(0.5) == 75.0 and T.sensitivity_to_slider(9) == 100.0


# ===================================================================================== presets
def test_balanced_is_the_defaults() -> None:
    vals = T.preset_values("Balanced", DEFAULTS, BOUNDS)
    assert set(vals) == set(T.SENSITIVITY_PATHS) | set(T.PRESET_SMOOTHING_PATHS)
    for p, v in vals.items():
        assert v == pytest.approx(DEFAULTS[p])


def test_precise_and_fast_values() -> None:
    p = T.preset_values("Precise", DEFAULTS, BOUNDS)
    f = T.preset_values("Fast", DEFAULTS, BOUNDS)
    assert p["hands.cursor.gain"] == pytest.approx(2.2 * 0.5**0.5)
    assert f["hands.cursor.gain"] == pytest.approx(2.2 * 2**0.5)
    assert p["hands.cursor.min_cutoff"] == pytest.approx(DEFAULTS["hands.cursor.min_cutoff"] * 0.6)
    assert p["hands.cursor.beta"] == pytest.approx(DEFAULTS["hands.cursor.beta"] * 0.7)
    assert f["hands.cursor.min_cutoff"] == pytest.approx(DEFAULTS["hands.cursor.min_cutoff"] * 1.6)
    assert f["hands.cursor.beta"] == pytest.approx(DEFAULTS["hands.cursor.beta"] * 1.5)
    assert p["hands.gestures.pinch_on"] < DEFAULTS["hands.gestures.pinch_on"] < f["hands.gestures.pinch_on"]
    for vals in (p, f):
        for path, v in vals.items():
            lo, hi = BOUNDS[path]
            assert lo <= v <= hi


def test_preset_bounds_and_unknown() -> None:
    tight = dict(BOUNDS, **{"hands.cursor.min_cutoff": (0.0, 1.1)})
    assert T.preset_values("Fast", DEFAULTS, tight)["hands.cursor.min_cutoff"] == 1.1
    with pytest.raises(ValueError, match="unknown preset"):
        T.preset_values("Turbo", DEFAULTS)
    assert "hands.cursor.beta" not in T.preset_values("Fast", {"hands.cursor.gain": 1.0})


def test_matching_preset() -> None:
    for name in T.PRESETS:
        vals = dict(DEFAULTS, **T.preset_values(name, DEFAULTS, BOUNDS))
        assert T.matching_preset(vals, DEFAULTS) == name
    vals = dict(DEFAULTS, **{"hands.cursor.gain": 3.0})
    assert T.matching_preset(vals, DEFAULTS) is None
    assert T.matching_preset({}, DEFAULTS) is None


# ===================================================================================== undo
def test_undo_drops_noops() -> None:
    u = T.UndoStack()
    assert not u.push("x", [T.Change("a", 1.0, 1.0)])
    assert len(u) == 0 and not u.can_undo and u.pop() is None and u.peek() is None
    assert u.push("x", [T.Change("a", 1.0, 1.0), T.Change("b", 1.0, 2.0)])
    assert u.peek().changes == [T.Change("b", 1.0, 2.0)]


def test_undo_coalesces_a_slider_drag() -> None:
    u = T.UndoStack(coalesce_s=1.0)
    u.push("gain", [T.Change("g", 1.0, 1.1)], t=0.0, coalesce=True)
    u.push("gain", [T.Change("g", 1.1, 1.2)], t=0.5, coalesce=True)
    u.push("gain", [T.Change("g", 1.2, 1.3)], t=1.5, coalesce=True)  # within 1 s of the previous move
    assert len(u) == 1 and u.peek().changes == [T.Change("g", 1.0, 1.3)] and u.peek().t == 1.5
    u.push("gain", [T.Change("g", 1.3, 1.4)], t=2.6, coalesce=True)  # 1.1 s later: a new step
    assert len(u) == 2
    u.push("gain", [T.Change("g", 1.4, 1.5)], t=2.7, coalesce=False)
    assert len(u) == 3
    u.push("dz", [T.Change("d", 0.0, 0.1)], t=2.8, coalesce=True)  # another slider
    assert len(u) == 4
    u.push("dz", [T.Change("d", 0.1, 0.2)], t=2.7, coalesce=True)  # time went backwards: not merged
    assert len(u) == 5


def test_undo_coalesce_boundary_and_batches() -> None:
    u = T.UndoStack(coalesce_s=1.0)
    u.push("g", [T.Change("g", 1.0, 2.0)], t=0.0, coalesce=True)
    u.push("g", [T.Change("g", 2.0, 3.0)], t=1.0, coalesce=True)  # exactly the window: merged
    assert len(u) == 1
    u.push("p", [T.Change("g", 3.0, 4.0), T.Change("h", 0.0, 1.0)], t=1.1, coalesce=True)
    u.push("g", [T.Change("g", 4.0, 5.0)], t=1.2, coalesce=True)  # last entry is a batch: not merged
    assert len(u) == 3


def test_undo_drag_back_to_start_removes_the_entry() -> None:
    u = T.UndoStack()
    u.push("g", [T.Change("g", 1.0, 1.5)], t=0.0, coalesce=True)
    u.push("g", [T.Change("g", 1.5, 1.0)], t=0.2, coalesce=True)
    assert len(u) == 0


def test_undo_max_depth_and_clear() -> None:
    u = T.UndoStack(max_depth=3)
    for i in range(5):
        u.push(f"s{i}", [T.Change("a", float(i), float(i + 1))])
    assert [e.label for e in u.entries] == ["s2", "s3", "s4"]
    assert u.pop().label == "s4" and len(u) == 2
    u.clear()
    assert not u.can_undo


# ===================================================================================== controller
def test_set_and_undo(ctl: T.TuningController, model: SettingsModel) -> None:
    assert ctl.set("hands.cursor.gain", 3.5) is None
    assert model.cfg.hands.cursor.gain == 3.5 and not ctl.is_default("hands.cursor.gain")
    assert ctl.undo_stack.peek().label == "Cursor speed"
    assert ctl.undo() == (None, "undid Cursor speed")
    assert model.cfg.hands.cursor.gain == 2.2 and ctl.is_default("hands.cursor.gain")
    assert ctl.undo() == ("nothing to undo", None)


def test_set_errors_are_not_recorded(ctl: T.TuningController, model: SettingsModel) -> None:
    assert "between" in ctl.set("hands.cursor.gain", 99)
    assert "unknown setting" in ctl.set("hands.nope", 1)
    assert len(ctl.undo_stack) == 0 and model.cfg.hands.cursor.gain == 2.2


def test_apply_is_order_independent_for_pinch_hysteresis(ctl: T.TuningController, model: SettingsModel) -> None:
    g = model.cfg.hands.gestures
    # raising both above the old pinch_off: pinch_on first would fail (0.6 >= 0.45) without the retry
    assert ctl.apply({"hands.gestures.pinch_on": 0.6, "hands.gestures.pinch_off": 0.8}) is None
    assert (g.pinch_on, g.pinch_off) == (0.6, 0.8)
    # lowering both below the old pinch_on: pinch_off first would fail
    assert ctl.apply({"hands.gestures.pinch_off": 0.3, "hands.gestures.pinch_on": 0.2}) is None
    assert (g.pinch_on, g.pinch_off) == (0.2, 0.3)
    assert ctl.undo()[0] is None and (g.pinch_on, g.pinch_off) == (0.6, 0.8)
    assert ctl.undo()[0] is None and (g.pinch_on, g.pinch_off) == (0.32, 0.45)


def test_apply_rolls_back_everything_on_failure(ctl: T.TuningController, model: SettingsModel) -> None:
    err = ctl.apply({"hands.cursor.gain": 3.0, "hands.gestures.pinch_on": 0.9})  # pinch_on >= pinch_off
    assert err and "pinch_on must be < pinch_off" in err
    assert model.cfg.hands.cursor.gain == 2.2 and model.cfg.hands.gestures.pinch_on == 0.32
    assert len(ctl.undo_stack) == 0
    assert ctl.apply({"hands.nope": 1.0, "hands.cursor.gain": 3.0}) == "unknown setting 'hands.nope'"
    assert model.cfg.hands.cursor.gain == 2.2


def test_reset_and_reset_group(ctl: T.TuningController, model: SettingsModel) -> None:
    ctl.set("hands.cursor.gain", 4.0)
    ctl.set("hands.cursor.dead_zone", 0.02)
    ctl.set("voice.end_silence_ms", 1500)
    assert ctl.reset("hands.cursor.gain") is None and model.cfg.hands.cursor.gain == 2.2
    assert ctl.undo_stack.peek().label == "reset Cursor speed"
    assert ctl.reset_group("Cursor") is None
    assert model.cfg.hands.cursor.dead_zone == DEFAULTS["hands.cursor.dead_zone"]
    assert model.cfg.voice.end_silence_ms == 1500  # another section is untouched
    assert ctl.reset_group("Nope") == "nothing to reset in Nope"
    assert ctl.undo()[0] is None and model.cfg.hands.cursor.dead_zone == 0.02
    assert "has no default" in ctl.reset("hands.nope")


def test_presets_through_the_controller(ctl: T.TuningController, model: SettingsModel) -> None:
    assert ctl.preset() == "Balanced"
    assert ctl.apply_preset("Fast") is None and ctl.preset() == "Fast"
    assert model.cfg.hands.cursor.gain == pytest.approx(2.2 * 2**0.5)
    assert ctl.sensitivity() == pytest.approx(0.5)
    assert ctl.apply_preset("Precise") is None and ctl.preset() == "Precise"
    assert ctl.sensitivity() == pytest.approx(-0.5)
    assert "unknown preset" in ctl.apply_preset("Nope")
    ctl.set("hands.cursor.gain", 3.0)
    assert ctl.preset() is None
    assert ctl.undo()[0] is None and ctl.preset() == "Precise"
    assert ctl.undo()[0] is None and ctl.preset() == "Fast"


def test_master_sensitivity_coalesces_and_matches(model: SettingsModel) -> None:
    t = iter([0.0, 0.2, 0.4, 5.0])
    ctl = T.TuningController(model, clock=lambda: next(t))
    assert ctl.set_sensitivity(0.2) is None
    assert ctl.set_sensitivity(0.6) is None
    assert ctl.set_sensitivity(1.0) is None
    assert len(ctl.undo_stack) == 1  # one drag, one undo step
    assert ctl.sensitivity() == pytest.approx(1.0) and ctl.sensitivity_matches()
    assert model.cfg.hands.cursor.gain == pytest.approx(4.4)
    ctl.set("hands.cursor.dead_zone", 0.03)
    assert not ctl.sensitivity_matches()
    assert ctl.undo()[0] is None and ctl.undo()[0] is None
    assert model.cfg.hands.cursor.gain == 2.2 and ctl.sensitivity() == 0.0


def test_undo_survives_a_model_that_refuses(ctl: T.TuningController, model: SettingsModel) -> None:
    ctl.set("hands.cursor.gain", 3.0)
    model.set_tunable = lambda path, v: "nope"  # type: ignore[method-assign]
    err, ok = ctl.undo()
    assert err == "nope" and ok is None and len(ctl.undo_stack) == 1  # kept for a later try


def test_undo_restores_pinch_pair_in_any_order(ctl: T.TuningController, model: SettingsModel) -> None:
    ctl.apply({"hands.gestures.pinch_on": 0.6, "hands.gestures.pinch_off": 0.8})
    ctl.apply({"hands.gestures.pinch_on": 0.1, "hands.gestures.pinch_off": 0.15})
    assert ctl.undo()[0] is None
    assert (model.cfg.hands.gestures.pinch_on, model.cfg.hands.gestures.pinch_off) == (0.6, 0.8)


def test_values_value_default(ctl: T.TuningController) -> None:
    vals = ctl.values()
    assert vals["hands.cursor.gain"] == 2.2 and ctl.value("hands.cursor.gain") == 2.2
    assert ctl.default("hands.cursor.gain") == 2.2 and ctl.default("nope") is None
    assert not ctl.is_default("nope")
    assert set(dict(ctl.groups())) <= set(T.GROUPS)


# ===================================================================================== per-app profiles
def test_snapshot_keeps_only_changed_hand_and_face_values() -> None:
    vals = dict(DEFAULTS)
    vals["hands.cursor.gain"] = 3.0
    vals["face.wink_hold"] = 0.5
    vals["voice.end_silence_ms"] = 999.0  # voice stays global
    vals["hands.min_camera_fps"] = 5.0  # camera stays global
    vals["hands.cursor.brand_new"] = 1.0  # no default known: kept
    snap = T.profile_tuning_snapshot(vals, DEFAULTS)
    assert snap == {"face.wink_hold": 0.5, "hands.cursor.brand_new": 1.0, "hands.cursor.gain": 3.0}
    assert list(snap) == sorted(snap)
    assert T.profile_tuning_snapshot(DEFAULTS, DEFAULTS) == {}


def test_profile_tuned_applies_without_touching_the_original() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("chrome", ["chrome"])
    cfg.bindings.profiles["chrome"].tuning = {
        "hands.cursor.gain": 3.0,
        "hands.cursor.big_cursor_size": 3.4,  # int field: rounded
        "hands.nope.x": 1.0,  # unknown: skipped
    }
    out = T.profile_tuned(cfg, "chrome")
    assert out.hands.cursor.gain == 3.0 and out.hands.cursor.big_cursor_size == 3
    assert isinstance(out.hands.cursor.big_cursor_size, int)
    assert cfg.hands.cursor.gain == 2.2  # untouched
    assert T.profile_tuned(cfg, "default").hands.cursor.gain == 2.2
    assert T.profile_tuned(cfg, "missing").hands.cursor.gain == 2.2


def test_profile_tuned_skips_bools_and_guards_pinch() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("x", ["x"])
    cfg.bindings.profiles["x"].tuning = {"hands.cursor.click_freeze": 0.0, "hands.gestures.pinch_on": 0.9}
    out = T.profile_tuned(cfg, "x")
    assert out.hands.cursor.click_freeze is cfg.hands.cursor.click_freeze
    assert (out.hands.gestures.pinch_on, out.hands.gestures.pinch_off) == (0.32, 0.45)


def test_save_as_profile_new_and_existing(ctl: T.TuningController, model: SettingsModel) -> None:
    ctl.set("hands.cursor.gain", 3.0)
    ctl.set("voice.end_silence_ms", 1200)
    err, ok = ctl.save_as_profile("chrome", "chrome, firefox.exe")
    assert err is None and "1 tuned setting;" in ok and "chrome, firefox.exe" in ok
    prof = model.bindings.profiles["chrome"]
    assert prof.tuning == {"hands.cursor.gain": 3.0} and prof.apps == ["chrome", "firefox.exe"]
    ctl.set("hands.cursor.dead_zone", 0.01)
    err, ok = ctl.save_as_profile("chrome")
    assert err is None and "2 tuned settings" in ok
    assert prof.tuning == {"hands.cursor.dead_zone": 0.01, "hands.cursor.gain": 3.0}
    assert prof.apps == ["chrome", "firefox.exe"]  # unchanged without apps text
    err, ok = ctl.save_as_profile("chrome", "edge")
    assert err is None and model.bindings.profiles["chrome"].apps == ["edge"]


def test_save_as_profile_errors_and_empty(ctl: T.TuningController, model: SettingsModel) -> None:
    assert ctl.save_as_profile("") == ("type a profile name", None)
    assert "global tuning" in ctl.save_as_profile("default")[0]
    assert ctl.save_as_profile("bad name!")[0]
    err, ok = ctl.save_as_profile("vlc")
    assert err is None and "no differences" in ok and "no apps yet" in ok
    assert model.bindings.profiles["vlc"].tuning == {}
    ctl.save_as_profile("chrome", "chrome")
    err, _ok = ctl.save_as_profile("vlc", "chrome")  # chrome is claimed by another profile
    assert err and "already claimed" in err


def test_profile_tuning_round_trips_through_the_config_file(ctl: T.TuningController, model: SettingsModel) -> None:
    ctl.set("hands.palm_scroll.depth_threshold", 0.9)
    ctl.save_as_profile("chrome", "chrome")
    model.save()
    cfg, _ = C.load(model.path)
    assert cfg.bindings.profiles["chrome"].tuning == {"hands.palm_scroll.depth_threshold": 0.9}
    assert "tuning" not in cfg.bindings.to_dict()["profiles"]["default"]


def test_validate_checks_profile_tuning() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("x", ["x"])
    cfg.bindings.profiles["x"].tuning = {"hands.cursor.nope": 1.0}
    with pytest.raises(ValueError, match="unknown setting"):
        C.validate(cfg)
    cfg.bindings.profiles["x"].tuning = {"hands.cursor.gain": 50.0}
    with pytest.raises(ValueError, match=r"must be a number in \[0.1, 10.0\]"):
        C.validate(cfg)
    cfg.bindings.profiles["x"].tuning = {"hands.cursor.gain": 10.0}
    C.validate(cfg)


@pytest.mark.parametrize(
    ("tuning", "msg"),
    [
        ([1, 2], "must be a mapping"),
        ({"voice.vad_sensitivity": 0.5}, "only hand and face settings"),
        ({"hands.cursor.gain": True}, "must be a number"),
        ({"hands.cursor.gain": "3"}, "must be a number"),
        ({"hands.cursor.gain": float("nan")}, "must be a number"),
    ],
)
def test_bindings_reject_bad_tuning(tuning, msg) -> None:
    with pytest.raises(BindingsError) as exc:
        parse_bindings({"profiles": {"x": {"apps": ["x"], "tuning": tuning}}})
    assert any(msg in e for e in exc.value.errors)


def test_bindings_tuning_parse_dump_merge_reset_copy() -> None:
    b = parse_bindings({"profiles": {"x": {"apps": ["x"], "tuning": {"hands.cursor.gain": 3, "face.wink_hold": 0.4}}}})
    assert b.profiles["x"].tuning == {"hands.cursor.gain": 3.0, "face.wink_hold": 0.4}
    assert isinstance(b.profiles["x"].tuning["hands.cursor.gain"], float)
    assert b.to_dict()["profiles"]["x"]["tuning"] == {"hands.cursor.gain": 3.0, "face.wink_hold": 0.4}
    assert parse_bindings({"profiles": {"x": {"apps": ["x"], "tuning": None}}}).profiles["x"].tuning == {}
    assert "tuning" not in parse_bindings({"profiles": {"x": {"apps": ["x"]}}}).to_dict()["profiles"]["x"]
    with pytest.raises(BindingsError, match="default profile uses the global settings"):
        parse_bindings({"profiles": {"default": {"tuning": {"hands.cursor.gain": 3}}}})
    with pytest.raises(BindingsError, match="did you mean 'tuning'"):
        parse_bindings({"profiles": {"x": {"apps": ["x"], "tunin": {}}}})
    # merge: only an explicit tuning: is taken over
    raw = {"profiles": {"x": {"tuning": {"hands.cursor.gain": 5.0}}}}
    merged = merge_bindings(b, parse_bindings(raw), raw)
    assert merged.profiles["x"].tuning == {"hands.cursor.gain": 5.0} and merged.profiles["x"].apps == ["x"]
    raw2 = {"profiles": {"x": {"gestures": {"pinch": "mouse:right_click"}}}}
    assert merge_bindings(b, parse_bindings(raw2), raw2).profiles["x"].tuning == b.profiles["x"].tuning
    # resetting the bindings keeps the tuning; copying a profile copies it
    b.reset("x")
    assert b.profiles["x"].tuning == {"hands.cursor.gain": 3.0, "face.wink_hold": 0.4}
    b.reset()
    assert b.profiles["x"].tuning == {"hands.cursor.gain": 3.0, "face.wink_hold": 0.4}
    b.new_profile("y", ["y"], copy_from="x")
    assert b.profiles["y"].tuning == b.profiles["x"].tuning and b.profiles["y"].tuning is not b.profiles["x"].tuning
    assert all(p.endswith(".") for p in PROFILE_TUNING_PREFIXES)


# ===================================================================================== runtime
def _pipeline(cfg, app):
    from away.app.runner import Pipeline

    return Pipeline(cfg, active_app=lambda: app["name"])


def test_pipeline_switches_tuning_with_the_focused_app() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("chrome", ["chrome"])
    cfg.bindings.profiles["chrome"].tuning = {"hands.cursor.gain": 3.0, "hands.gestures.pinch_on": 0.4}
    app = {"name": "notepad"}
    pipe = _pipeline(cfg, app)
    pipe.dispatcher.app_poll = 0.0
    assert pipe.dispatcher.profile == "default" and pipe.cfg.hands.cursor.gain == 2.2
    app["name"] = "chrome.exe"
    assert pipe.dispatcher.profile == "chrome"
    assert pipe.cfg.hands.cursor.gain == 3.0 and pipe.gestures.cfg.pinch_on == 0.4
    assert pipe.mouse.mapper.cfg.gain == 3.0
    app["name"] = "notepad"
    assert pipe.dispatcher.profile == "default"
    assert pipe.cfg.hands.cursor.gain == 2.2 and pipe.gestures.cfg.pinch_on == 0.32


def test_pipeline_reload_keeps_the_app_overlay() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("chrome", ["chrome"])
    cfg.bindings.profiles["chrome"].tuning = {"hands.cursor.gain": 3.0}
    app = {"name": "chrome"}
    pipe = _pipeline(cfg, app)
    pipe.dispatcher.app_poll = 0.0
    assert pipe.dispatcher.profile == "chrome" and pipe.cfg.hands.cursor.gain == 3.0
    new = C.AppConfig()
    new.hands.cursor.dead_zone = 0.01
    new.bindings.new_profile("chrome", ["chrome"])
    new.bindings.profiles["chrome"].tuning = {"hands.cursor.gain": 4.0}
    pipe.apply_config(new)
    assert pipe.cfg.hands.cursor.gain == 4.0 and pipe.cfg.hands.cursor.dead_zone == 0.01
    assert new.hands.cursor.gain == 2.2  # the loaded config itself is not changed
    app["name"] = "other"
    assert pipe.dispatcher.profile == "default" and pipe.cfg.hands.cursor.gain == 2.2
    assert pipe.cfg.hands.cursor.dead_zone == 0.01


def test_pipeline_pinned_profile_applies_at_start() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("game", [])
    cfg.bindings.profiles["game"].tuning = {"hands.cursor.gain": 5.0}
    cfg.bindings.active_profile = "game"
    from away.app.runner import Pipeline

    pipe = Pipeline(cfg)
    assert pipe.cfg.hands.cursor.gain == 5.0
    assert pipe._base_cfg.hands.cursor.gain == 2.2


def test_dispatcher_reports_profile_switches() -> None:
    from away.actions import ActionExecutor, Dispatcher
    from away.control.mouse import MouseController, RecordingBackend
    from away.hands.cursor import CursorMapper

    cfg = C.AppConfig()
    cfg.bindings.new_profile("chrome", ["chrome"])
    app = {"name": "x"}
    ex = ActionExecutor(MouseController(RecordingBackend(), CursorMapper((100, 100), cfg.hands.cursor)))
    d = Dispatcher(cfg.bindings, ex, active_app=lambda: app["name"], app_poll=0.0)
    seen: list[str] = []
    d.on_profile = seen.append
    assert d.profile == "default" and seen == []
    app["name"] = "chrome"
    assert d.profile == "chrome" and seen == ["chrome"]
    assert d.profile == "chrome" and seen == ["chrome"]  # no repeat
    app["name"] = "y"
    d.profile  # noqa: B018
    assert seen == ["chrome", "default"]
    d2 = Dispatcher(cfg.bindings, ex)
    cfg.bindings.active_profile = "chrome"
    d2.on_profile = seen.append
    assert d2.profile == "chrome" and seen[-1] == "chrome"


# ===================================================================================== the tab (Xvfb only)
@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and os.environ.get("DISPLAY")), reason="needs an X display (Xvfb)"
)
def test_tuning_tab_smoke(tmp_path: Path) -> None:  # pragma: no cover - needs a display
    tk = pytest.importorskip("tkinter")
    from away.app import settings_gui as G

    try:
        root = tk.Tk()
    except Exception as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        w = G.SettingsWindow(root, G.SettingsModel(path=tmp_path / "config.yaml"))
        root.geometry("900x640")
        root.update()
        assert len(w._scrollers) == 7  # v0.1.20: + Agent Link
        titles = list(w._tune_sections)
        assert titles[:2] == ["Live view: why did a gesture (not) fire?", "Auto-tune a gesture"]
        assert [t for t in titles[2:]] == [g for g, _ in w.tuning.groups()]
        assert len(w._tune_rows) == len(w.m.tunables())
        w._tune_preset("Fast")
        root.update()
        assert w.preset_note.get().startswith("Preset: Fast") and str(w.undo_btn.cget("state")) == "normal"
        w._tune_undo()
        root.update()
        assert w.m.cfg.hands.cursor.gain == 2.2 and str(w.undo_btn.cget("state")) == "disabled"
        w._on_scale("hands.cursor.gain", 4.0)
        root.update()
        val, rb, _t = w._tune_rows["hands.cursor.gain"]
        assert val.cget("text") == "4×" and str(rb.cget("state")) == "normal"
        w._tune_reset("hands.cursor.gain")
        root.update()
        assert w.m.cfg.hands.cursor.gain == 2.2
        body, state, btn = w._tune_sections["Cursor"]
        btn.invoke()
        assert state["open"] is False
        w._live_draw(None)
        from away.hands import synthetic
        from away.hands.autotune import live_gauges

        lm = synthetic.pose("palm")
        w._live_draw({"lm": lm, "helper": lm, "gauges": live_gauges(lm, lm, w.m.cfg)})
        root.update()
    finally:
        root.destroy()


# ===================================================================================== edge cases (mutation pass)
def test_sensitivity_pinch_off_floor_exact() -> None:
    assert T.sensitivity_values(-1.0, DEFAULTS, BOUNDS)["hands.gestures.pinch_off"] == pytest.approx(0.36)


def test_format_value_zero_step_falls_back_to_three_decimals() -> None:
    assert T.format_value(1.23456, "", False, 0.0) == "1.235"
    assert T.format_value(1.23456, "", False, None) == "1.235"


def test_estimate_sensitivity_small_default() -> None:
    assert T.estimate_sensitivity({"hands.cursor.gain": 1.0}, {"hands.cursor.gain": 0.5}) == pytest.approx(1.0)
    assert T.estimate_sensitivity({"hands.cursor.gain": 0.5}, {"hands.cursor.gain": 0.5}) == 0.0


def test_preset_bounds_without_the_smoothing_paths() -> None:
    only_gain = {"hands.cursor.gain": (0.1, 10.0)}
    vals = T.preset_values("Fast", DEFAULTS, only_gain)
    assert vals["hands.cursor.min_cutoff"] == pytest.approx(DEFAULTS["hands.cursor.min_cutoff"] * 1.6)


def test_matching_preset_tolerance() -> None:
    fast = dict(DEFAULTS, **T.preset_values("Fast", DEFAULTS, BOUNDS))
    v = fast["hands.cursor.min_cutoff"]
    assert 1.0 < v < 2.0
    assert T.matching_preset(dict(fast, **{"hands.cursor.min_cutoff": v + 0.5e-6 * v}), DEFAULTS) == "Fast"
    assert T.matching_preset(dict(fast, **{"hands.cursor.min_cutoff": v + 1.8e-6}), DEFAULTS) is None


def test_undo_defaults_and_return_values() -> None:
    u = T.UndoStack()
    assert u.max_depth == 100 and u.coalesce_s == 1.0
    assert T.UndoEntry("x", []).t == 0.0
    assert u.can_undo is False
    assert u.push("g", [T.Change("g", 1.0, 1.0)]) is False
    assert u.push("g", [T.Change("g", 1.0, 2.0)]) is True
    assert u.peek().t == 0.0 and u.can_undo is True
    assert u.push("g", [T.Change("g", 2.0, 3.0)]) is True  # default: no coalescing even at the same time
    assert len(u) == 2
    assert u.push("g", [T.Change("g", 3.0, 4.0)], coalesce=True) is True  # same instant: merged
    assert len(u) == 2 and u.peek().changes == [T.Change("g", 2.0, 4.0)]
    assert u.push("h", [T.Change("g", 4.0, 5.0)], coalesce=True) is True  # another label: not merged
    assert len(u) == 3


def test_snapshot_tolerance_is_strict() -> None:
    d = {"hands.cursor.gain": 1.0}
    assert T.profile_tuning_snapshot({"hands.cursor.gain": 1.5}, d, tol=0.5) == {}
    assert T.profile_tuning_snapshot({"hands.cursor.gain": 1.75}, d, tol=0.5) == {"hands.cursor.gain": 1.75}


def test_profile_tuned_equal_pinch_values_are_reset() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("x", ["x"])
    cfg.bindings.profiles["x"].tuning = {"hands.gestures.pinch_on": 0.45}  # == the global pinch_off
    out = T.profile_tuned(cfg, "x")
    assert (out.hands.gestures.pinch_on, out.hands.gestures.pinch_off) == (0.32, 0.45)
    cfg.bindings.profiles["x"].tuning = {"hands.gestures.pinch_on": 0.44}
    assert T.profile_tuned(cfg, "x").hands.gestures.pinch_on == 0.44


def test_controller_bounds_are_cached(ctl: T.TuningController) -> None:
    b = ctl.bounds
    assert b["hands.cursor.gain"] == (0.1, 10.0) and ctl.bounds is b
    assert ctl.defaults is ctl.defaults


def test_is_default_and_sensitivity_matches_tolerance(ctl: T.TuningController, model: SettingsModel) -> None:
    d = DEFAULTS["hands.cursor.dead_zone"]
    assert d < 1.0
    model.cfg.hands.cursor.dead_zone = d + 1e-12
    assert ctl.is_default("hands.cursor.dead_zone")
    model.cfg.hands.cursor.dead_zone = d + 1.5e-9
    assert not ctl.is_default("hands.cursor.dead_zone")
    model.cfg.hands.cursor.dead_zone = d + 0.5e-6
    assert ctl.sensitivity_matches()
    model.cfg.hands.cursor.dead_zone = d + 1.5e-6
    assert not ctl.sensitivity_matches()


def test_slider_moves_coalesce_but_apply_does_not(model: SettingsModel) -> None:
    ctl = T.TuningController(model, clock=lambda: 0.0)
    ctl.set("hands.cursor.gain", 3.0)
    ctl.set("hands.cursor.gain", 3.5)
    assert len(ctl.undo_stack) == 1
    ctl.apply({"hands.cursor.beta": 0.5}, label="x")
    ctl.apply({"hands.cursor.beta": 0.6}, label="x")
    assert len(ctl.undo_stack) == 3


def test_failed_undo_rolls_back_the_part_that_worked(ctl: T.TuningController, model: SettingsModel) -> None:
    ctl.apply({"hands.cursor.gain": 3.0, "hands.cursor.dead_zone": 0.01}, label="batch")
    real = model.set_tunable

    def picky(path: str, v: float) -> str | None:
        if path == "hands.cursor.dead_zone" and v == DEFAULTS["hands.cursor.dead_zone"]:
            return "refused"
        return real(path, v)

    model.set_tunable = picky  # type: ignore[method-assign]
    err, ok = ctl.undo()
    assert err == "refused" and ok is None
    assert model.cfg.hands.cursor.gain == 3.0 and model.cfg.hands.cursor.dead_zone == 0.01
    assert ctl.undo_stack.peek().label == "batch"


def test_tuning_sources_decode_with_the_windows_code_page() -> None:
    """Some tests read sources with the default encoding (cp1252 on Windows CI); keep the Tuning glyphs escaped."""
    root = Path(__file__).resolve().parents[1] / "src" / "away"
    for rel in ("app/settings_gui.py", "app/tunables.py", "app/tuning_live.py", "hands/autotune.py"):
        (root / rel).read_bytes().decode("cp1252")
