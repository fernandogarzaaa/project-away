"""v0.1.19 integration: the auto-centering / head / gaze tunables (PR #10) inside the grouped Tuning tab, presets,
master sensitivity and per-app profile tuning (PR #11)."""

from __future__ import annotations

import pytest

from away import config as C
from away.app import tunables as T
from away.app.settings_gui import SettingsModel
from away.bindings import BindingsError, parse_bindings
from away.hands.cursor import CursorMapper

TRACKING_PREFIXES = ("hands.cursor.autocenter.", "hands.cursor.head.", "gaze.")
V0120_PREFIXES = ("attention.", "comfort.")
DEFAULTS = T.default_values()


def tracking_paths() -> list[str]:
    return [p for p, *_ in C.tunable_ranges(C.AppConfig()) if p.startswith(TRACKING_PREFIXES)]


@pytest.fixture
def model(tmp_path):  # type: ignore[no-untyped-def]
    return SettingsModel(C.AppConfig(), tmp_path / "config.yaml")


@pytest.fixture
def ctl(model: SettingsModel) -> T.TuningController:
    clock = iter(float(i) * 10 for i in range(10_000))
    return T.TuningController(model, clock=lambda: next(clock))


def test_tracking_tunables_land_in_head_and_eyes(model: SettingsModel) -> None:
    paths = tracking_paths()
    assert len(paths) == 28
    by_path = {t.path: t for t in model.tunables()}
    for p in paths:
        t = by_path[p]
        assert t.group == "Head & eyes", p
        assert t.title and not t.title.startswith(("hands.", "gaze.")), p
        assert t.help and not t.help.startswith("Advanced setting"), p
    assert by_path["hands.cursor.head.coarse_cols"].is_int and by_path["hands.cursor.head.coarse_rows"].is_int
    # nothing else moved into Head & eyes, and the old Cursor section keeps its own sliders
    # (v0.1.20 adds the attention-gate / arm-fatigue sliders to Head & eyes on purpose)
    head = {t.path for t in model.tunables() if t.group == "Head & eyes" and not t.path.startswith(V0120_PREFIXES)}
    assert head == set(paths)
    assert by_path["hands.cursor.gain"].group == "Cursor" and by_path["hands.cursor.min_cutoff"].group == "Cursor"


def test_tracking_defaults_are_not_snapped_by_the_slider_step(model: SettingsModel) -> None:
    for t in model.tunables():
        if t.path.startswith(TRACKING_PREFIXES) and t.default is not None:
            assert round(t.default / t.resolution) * t.resolution == pytest.approx(t.default), t.path


def test_presets_and_sensitivity_leave_tracking_alone(ctl: T.TuningController, model: SettingsModel) -> None:
    for name in T.PRESETS:
        assert not any(p.startswith(TRACKING_PREFIXES) for p in T.preset_values(name, DEFAULTS))
    assert not any(p.startswith(TRACKING_PREFIXES) for p in T.SENSITIVITY_PATHS)
    ctl.set("hands.cursor.autocenter.soft_strength", 0.3)
    ctl.set("hands.cursor.head.min_cutoff", 3.0)  # same leaf name as hands.cursor.min_cutoff (a preset path)
    assert ctl.preset() == "Balanced" and ctl.sensitivity() == pytest.approx(0.0)
    assert ctl.apply_preset("Fast") is None and ctl.preset() == "Fast"
    assert model.cfg.hands.cursor.autocenter.soft_strength == 0.3
    assert model.cfg.hands.cursor.head.min_cutoff == 3.0
    assert model.cfg.hands.cursor.min_cutoff == pytest.approx(1.6)
    assert ctl.set_sensitivity(-1.0) is None
    assert model.cfg.hands.cursor.autocenter.soft_strength == 0.3
    assert ctl.reset_group("Head & eyes") is None
    assert model.cfg.hands.cursor.autocenter.soft_strength == DEFAULTS["hands.cursor.autocenter.soft_strength"]
    assert ctl.sensitivity() == pytest.approx(-1.0)  # resetting Head & eyes did not touch the master slider


def _stroke_distance(sensitivity: float, autocenter: bool) -> float:
    """Cursor travel for one stroke from the frame centre toward its edge (relative mode, filter off)."""
    cfg = C.CursorConfig(curve="power", min_cutoff=1e6, d_cutoff=1e6, beta=0.0)
    vals = T.sensitivity_values(sensitivity, DEFAULTS, T.tunable_bounds())
    cfg.gain = vals["hands.cursor.gain"]
    cfg.acceleration = vals["hands.cursor.acceleration"]
    cfg.dead_zone = vals["hands.cursor.dead_zone"]
    cfg.autocenter = C.AutoCenterConfig(enabled=autocenter, edge_reanchor=False, comfort_zone=False)
    m = CursorMapper((4000, 4000), cfg)
    m.set_position(2000, 2000)
    x0 = m.pos[0]
    for i in range(11):
        m.update(i / 30, 0.5 + i * 0.01, 0.5)
    return m.pos[0] - x0


def test_master_sensitivity_still_orders_speed_with_soft_recenter_on() -> None:
    slow, mid, fast = (_stroke_distance(s, True) for s in (-0.5, 0.0, 0.5))
    assert 0 < slow < mid < fast
    # soft recenter changes the gain by at most +-soft_strength, so it never undoes a sensitivity step
    strength = C.AutoCenterConfig().soft_strength
    for s in (-0.5, 0.0, 0.5):
        on, off = _stroke_distance(s, True), _stroke_distance(s, False)
        assert off * (1 - strength) - 1 <= on <= off * (1 + strength) + 1


def test_profile_tuning_accepts_tracking_hand_keys_but_not_gaze() -> None:
    data = {
        "profiles": {
            "game": {"apps": ["game"], "tuning": {"hands.cursor.autocenter.soft_strength": 0.3}},
        }
    }
    b = parse_bindings(data)
    assert b.profiles["game"].tuning == {"hands.cursor.autocenter.soft_strength": 0.3}
    with pytest.raises(BindingsError, match="only hand and face settings"):
        parse_bindings({"profiles": {"game": {"apps": ["game"], "tuning": {"gaze.min_cutoff": 1.0}}}})


def test_profile_tuning_with_tracking_keys_validates_and_applies() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("game", ["game"])
    cfg.bindings.profiles["game"].tuning = {
        "hands.cursor.autocenter.soft_strength": 0.3,
        "hands.cursor.head.coarse_cols": 5.0,
        "hands.cursor.gain": 3.0,
    }
    C.validate(cfg)
    out = T.profile_tuned(cfg, "game")
    assert out.hands.cursor.autocenter.soft_strength == 0.3
    assert out.hands.cursor.head.coarse_cols == 5 and isinstance(out.hands.cursor.head.coarse_cols, int)
    assert out.hands.cursor.gain == 3.0
    assert cfg.hands.cursor.autocenter.soft_strength == C.AutoCenterConfig().soft_strength  # original untouched
    cfg.bindings.profiles["game"].tuning = {"hands.cursor.autocenter.soft_strength": 0.9}
    with pytest.raises(ValueError, match=r"must be a number in \[0.0, 0.5\]"):
        C.validate(cfg)


def test_save_as_profile_keeps_tracking_hand_tuning_and_round_trips(
    ctl: T.TuningController, model: SettingsModel
) -> None:
    ctl.set("hands.cursor.autocenter.soft_strength", 0.25)
    ctl.set("hands.cursor.head.pointer_gain", 0.02)
    ctl.set("gaze.jump_min_px", 400.0)  # global only: not kept in the profile
    err, _ok = ctl.save_as_profile("game", "game")
    assert err is None
    assert model.bindings.profiles["game"].tuning == {
        "hands.cursor.autocenter.soft_strength": 0.25,
        "hands.cursor.head.pointer_gain": 0.02,
    }
    model.save()
    cfg, _ = C.load(model.path)
    assert cfg.bindings.profiles["game"].tuning["hands.cursor.autocenter.soft_strength"] == 0.25
    assert cfg.gaze.jump_min_px == 400.0


def test_pipeline_profile_switch_carries_tracking_tuning() -> None:
    cfg = C.AppConfig()
    cfg.bindings.new_profile("game", [])
    cfg.bindings.profiles["game"].tuning = {
        "hands.cursor.autocenter.soft_strength": 0.4,
        "hands.cursor.head.coarse_rows": 4.0,
    }
    cfg.bindings.active_profile = "game"
    from away.app.runner import Pipeline

    pipe = Pipeline(cfg)
    assert pipe.headgaze is not None
    assert pipe.cfg.hands.cursor.autocenter.soft_strength == 0.4
    assert pipe.cfg.hands.cursor.head.coarse_rows == 4
    assert pipe._base_cfg.hands.cursor.autocenter.soft_strength == C.AutoCenterConfig().soft_strength
    # the mapper follows the live config object (Settings / profile switches replace it)
    ac = pipe.mouse.mapper._sync_autocenter()
    assert ac is not None and ac.cfg.soft_strength == 0.4
    pipe._on_profile("default")
    assert pipe.cfg.hands.cursor.autocenter.soft_strength == C.AutoCenterConfig().soft_strength
    assert pipe.mouse.mapper._sync_autocenter().cfg.soft_strength == C.AutoCenterConfig().soft_strength  # type: ignore[union-attr]
