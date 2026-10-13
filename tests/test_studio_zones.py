"""Gesture Studio zones (v0.1.20): grid / boxes / line, hysteresis, per camera + profile, config, runtime gate."""

from __future__ import annotations

import numpy as np
import pytest

from away import config as C
from away.control.mouse import RecordingBackend
from away.hands.synthetic import pose
from away.studio import zones as Z


def test_default_map_is_6_by_4_and_open() -> None:
    zm = Z.ZoneMap()
    assert (zm.cols, zm.rows) == (6, 4)
    assert zm.is_open() and zm.allowed(0.01, 0.99) and zm.active_fraction() == 1.0


def test_ignored_cell_only_blocks_inside_it() -> None:
    zm = Z.ZoneMap()
    r, c = zm.cell_at(0.1, 0.1)
    assert (r, c) == (0, 0)
    assert zm.toggle_cell(r, c) == Z.IGNORE
    assert not zm.allowed(0.1, 0.1)
    assert zm.allowed(0.2, 0.1)  # next column (1/6 = 0.1667 boundary)
    assert zm.allowed(0.1, 0.3)  # next row (0.25 boundary)
    assert not zm.is_open()
    assert zm.active_fraction() == pytest.approx(1 - 1 / 24, abs=0.01)
    assert zm.toggle_cell(r, c) == Z.ACTIVE and zm.is_open()


def test_cell_at_clamps_outside_points() -> None:
    zm = Z.ZoneMap()
    assert zm.cell_at(-1, -1) == (0, 0) and zm.cell_at(2, 2) == (3, 5) and zm.cell_at(1.0, 1.0) == (3, 5)


def test_later_rectangles_win_over_grid_and_earlier_rects() -> None:
    zm = Z.ZoneMap()
    zm.set_all(Z.IGNORE)
    assert not zm.allowed(0.5, 0.5)
    zm.rects.append(Z.ZoneRect(0.4, 0.4, 0.6, 0.6, Z.ACTIVE))
    assert zm.allowed(0.5, 0.5) and not zm.allowed(0.3, 0.5)
    zm.rects.append(Z.ZoneRect(0.6, 0.6, 0.45, 0.45, Z.IGNORE))  # corners in any order
    assert not zm.allowed(0.5, 0.5) and zm.allowed(0.42, 0.42)
    with pytest.raises(ValueError):
        zm.set_cell(0, 0, "maybe")


def test_activation_line_keeps_only_the_side_above() -> None:
    zm = Z.ZoneMap(line=Z.ActivationLine(0.0, 0.7, 1.0, 0.7, keep=-1))
    assert zm.allowed(0.5, 0.5) and not zm.allowed(0.5, 0.8)
    assert zm.allowed(0.5, 0.7)  # on the line counts
    zm.line = Z.ActivationLine(0.0, 0.7, 1.0, 0.7, keep=1)
    assert not zm.allowed(0.5, 0.5) and zm.allowed(0.5, 0.8)
    assert not zm.is_open()


def test_resize_keeps_the_mode_under_each_new_cell_centre() -> None:
    zm = Z.ZoneMap()
    for r in range(4):
        for c in range(3):
            zm.set_cell(r, c, Z.IGNORE)  # left half ignored
    zm.resize(2, 2)
    assert zm.cells == [[Z.IGNORE, Z.ACTIVE], [Z.IGNORE, Z.ACTIVE]]
    with pytest.raises(ValueError):
        zm.resize(0, 2)
    with pytest.raises(ValueError):
        zm.resize(25, 2)


def test_move_edge_keeps_a_gap_and_outer_edges_are_fixed() -> None:
    zm = Z.ZoneMap()
    assert zm.move_edge("x", 1, 0.25) == pytest.approx(0.25)
    assert zm.move_edge("x", 1, 0.9) == pytest.approx(2 / 6 - Z.MIN_EDGE_GAP)
    assert zm.move_edge("y", 1, -1.0) == pytest.approx(Z.MIN_EDGE_GAP)
    with pytest.raises(ValueError):
        zm.move_edge("x", 0, 0.5)
    with pytest.raises(ValueError):
        zm.move_edge("x", 6, 0.5)
    assert zm.cell_at(0.32, 0.01) == (0, 1) and zm.cell_at(0.3, 0.1) == (1, 0)


def test_nearest_edge_finds_inner_lines_only() -> None:
    zm = Z.ZoneMap()
    assert zm.nearest_edge("x", 0.34, 0.02) == 2
    assert zm.nearest_edge("x", 0.01, 0.05) is None  # the outer edge cannot be dragged
    assert zm.nearest_edge("y", 0.4, 0.02) is None
    assert zm.nearest_edge("y", 0.51, 0.02) == 2


def test_round_trip_and_validation() -> None:
    zm = Z.ZoneMap()
    zm.toggle_cell(1, 2)
    zm.move_edge("x", 3, 0.55)
    zm.rects.append(Z.ZoneRect(0.1, 0.1, 0.2, 0.3, Z.IGNORE))
    zm.line = Z.ActivationLine(0.0, 0.6, 1.0, 0.65, keep=-1)
    back = Z.zone_map_from_dict(zm.to_dict())
    assert back.to_dict() == zm.to_dict()
    for x in np.linspace(0, 1, 11):
        for y in np.linspace(0, 1, 11):
            assert back.allowed(float(x), float(y)) == zm.allowed(float(x), float(y))
    bad = [
        {"grid": {"cols": 2, "rows": 1, "cells": ["ax"]}},
        {"grid": {"cols": 2, "rows": 1, "cells": ["a"]}},
        {"grid": {"cols": 2, "rows": 1, "xs": [0, 0.7, 0.5]}},
        {"grid": {"cols": 2, "rows": 1, "xs": [0.1, 0.5, 1]}},
        {"grid": {"cols": True, "rows": 1}},
        {"grid": {"cols": 0, "rows": 1}},
        {"rects": [{"x0": 0, "y0": 0, "x1": 2, "y1": 1}]},
        {"rects": [{"x0": 0, "y0": 0, "x1": 1, "y1": 1, "mode": "maybe"}]},
        {"line": {"x0": 0, "y0": 0.5, "x1": 1, "y1": 0.5, "keep": 0}},
        {"line": {"x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}},
        {"line": {"x0": 0, "y0": 0.5, "x1": 1, "y1": 0.5, "keep": True}},
        "nope",
    ]
    for d in bad:
        with pytest.raises(ValueError):
            Z.zone_map_from_dict(d)


def test_lookup_falls_back_camera_then_any_camera_then_default_profile() -> None:
    maps: dict = {}
    a, b, c, d = (Z.ZoneMap() for _ in range(4))
    a.toggle_cell(0, 0)
    b.toggle_cell(0, 1)
    c.toggle_cell(0, 2)
    d.toggle_cell(0, 3)
    Z.store_map(maps, "*", "default", a)
    assert Z.lookup_map(maps, "cam", "chrome")[1] == ("*", "default")
    Z.store_map(maps, "*", "chrome", b)
    assert Z.lookup_map(maps, "cam", "chrome")[1] == ("*", "chrome")
    Z.store_map(maps, "cam", "default", c)
    assert Z.lookup_map(maps, "cam", "chrome")[1] == ("cam", "default")  # this camera beats any camera
    Z.store_map(maps, "cam", "chrome", d)
    zm, src = Z.lookup_map(maps, "cam", "chrome")
    assert src == ("cam", "chrome") and zm is not None and zm.cells[0][3] == Z.IGNORE
    assert Z.lookup_map(maps, "other", "word")[1] == ("*", "default")
    Z.store_map(maps, "cam", "chrome", None)
    Z.store_map(maps, "cam", "default", None)
    assert "cam" not in maps
    assert Z.lookup_map({}, "cam", "default") == (None, None)


def test_camera_key_uses_the_primary_camera() -> None:
    cfg = C.AppConfig()
    cfg.hands.camera_index = "HD User Facing"
    assert Z.camera_key(cfg) == "HD User Facing"
    cfg.hands.cameras = [C.CameraSpec(device="JoyCam", role="secondary"), C.CameraSpec(device=1, role="primary")]
    assert Z.camera_key(cfg) == "1"


def test_anchor_point_is_in_mirrored_space() -> None:
    lm = pose("palm", center=(0.2, 0.4))
    x, y = Z.anchor_point(lm, "palm", mirrored=True)
    xm, ym = Z.anchor_point(lm, "palm", mirrored=False)
    assert xm == pytest.approx(1 - x) and ym == pytest.approx(y)
    assert Z.anchor_point(lm, "index_tip")[1] < y  # fingertip above the palm
    assert Z.anchor_point(lm, "wrist")[1] > y


def test_gate_hysteresis_band() -> None:
    zm = Z.ZoneMap(xs=[0.0, 0.5, 1.0], ys=[0.0, 1.0], cells=[[Z.ACTIVE, Z.IGNORE]])
    g = Z.ZoneGate(zm, hysteresis=0.05)
    assert g.update((0.3, 0.5)) is True
    assert g.update((0.52, 0.5)) is True  # just past the boundary: still inside the band
    assert g.update((0.56, 0.5)) is False  # far enough
    assert g.update((0.48, 0.5)) is False  # back across, inside the band: stays ignored
    assert g.update((0.44, 0.5)) is True
    assert g.update(None) is True and g.state is None
    assert g.update((0.52, 0.5)) is False  # a fresh hand takes the raw side at once
    nog = Z.ZoneGate(zm, hysteresis=0.0)
    nog.update((0.3, 0.5))
    assert nog.update((0.51, 0.5)) is False
    assert Z.ZoneGate(None).update((0.9, 0.5)) is True


def test_config_round_trip_and_validation(tmp_path) -> None:
    zm = Z.ZoneMap()
    zm.toggle_cell(2, 2)
    data = {"zones": {"block_cursor": True, "anchor": "index_tip", "maps": {"cam": {"default": zm.to_dict()}}}}
    cfg, unknown = C.from_dict(data)
    assert not unknown and cfg.zones.block_cursor and cfg.zones.anchor == "index_tip"
    p = C.save(cfg, tmp_path / "c.yaml")
    back, _ = C.load(p)
    assert back.zones.maps == cfg.zones.maps and back.zones.block_cursor
    assert C.AppConfig().zones.block_cursor is False and C.AppConfig().zones.hysteresis == 0.03
    for bad in (
        {"zones": {"anchor": "elbow"}},
        {"zones": {"hysteresis": 0.5}},
        {"zones": {"block_cursor": "yes"}},
        {"zones": {"maps": {"cam": {"default": {"grid": {"cols": 99}}}}}},
        {"zones": {"maps": {"cam": "x"}}},
        {"zones": {"maps": []}},
    ):
        with pytest.raises(ValueError):
            C.from_dict(bad)


# ---------------------------------------------------------------------- the pipeline enforces zones
def _ignore_all_cfg(block_cursor: bool = False) -> C.AppConfig:
    cfg = C.AppConfig()
    zm = Z.ZoneMap()
    zm.set_all(Z.IGNORE)
    Z.store_map(cfg.zones.maps, "*", "default", zm)
    cfg.zones.block_cursor = block_cursor
    return cfg


def _click(p) -> list[str]:
    out: list[str] = []
    t = 0.0
    for lm in [pose("pinch_open"), pose("pinch"), pose("pinch_open")]:
        t += 0.05
        out += p.on_hand(t, lm)
    return out


def test_pipeline_ignores_clicks_in_ignored_areas_but_moves_the_cursor() -> None:
    from away.app.runner import Pipeline

    b = RecordingBackend()
    p = Pipeline(_ignore_all_cfg(), b)
    out = _click(p)
    assert "click" not in out and not any(c[0] == "click" for c in b.calls)
    assert p.bus.history("gesture.zoned") and p.bus.history("gesture.zoned")[0].data == {"kind": "click"}
    for i in range(6):
        p.on_hand(1.0 + i * 0.05, pose("point", center=(0.5 + 0.02 * i, 0.5)))
    assert p.bus.history("gesture.move")  # the cursor still moves (zones.block_cursor is off)


def test_pipeline_clicks_when_zones_are_open_or_disabled() -> None:
    from away.app.runner import Pipeline

    b = RecordingBackend()
    assert "click" in _click(Pipeline(C.AppConfig(), b)) and ("click", (1,)) in b.calls
    cfg = _ignore_all_cfg()
    cfg.zones.enabled = False
    assert "click" in _click(Pipeline(cfg, RecordingBackend()))


def test_block_cursor_also_stops_moves() -> None:
    from away.app.runner import Pipeline

    p = Pipeline(_ignore_all_cfg(block_cursor=True), RecordingBackend())
    for i in range(6):
        p.on_hand(i * 0.05, pose("point", center=(0.5 + 0.02 * i, 0.5)))
    zoned = [e.data["kind"] for e in p.bus.history("gesture.zoned")]
    assert "move" in zoned and not p.bus.history("gesture.move")


def test_zone_map_follows_the_profile_and_live_reload() -> None:
    from away.app.runner import Pipeline

    cfg = C.AppConfig()
    cfg.bindings.new_profile("chrome", ["chrome"])
    zm = Z.ZoneMap()
    zm.set_all(Z.IGNORE)
    Z.store_map(cfg.zones.maps, "*", "chrome", zm)
    p = Pipeline(cfg, RecordingBackend())
    assert p.studio.gate.map is None  # default profile: no map
    p.studio.set_profile("chrome")
    assert p.studio.gate.map is not None
    p._on_profile("default")
    assert p.studio.gate.map is None
    new = C.AppConfig()
    Z.store_map(new.zones.maps, "*", "default", zm)
    p.apply_config(new)
    assert p.studio.gate.map is not None and p.cfg.zones.maps == new.zones.maps


def test_ending_events_always_pass_the_gate() -> None:
    from away.studio.engine import StudioRuntime

    rt = StudioRuntime(_ignore_all_cfg(), recognizer=None)
    assert rt.check_zone(pose("palm")) is False
    for kind in ("drag_end", "hand_lost", "autoscroll", "autoscroll_stop", "palm_scroll", "clap", "move"):
        assert rt.event_allowed(kind), kind
    for kind in ("click", "swipe_left", "clutch", "custom", "dwell", "push", "palm_tap", "drag_start"):
        assert not rt.event_allowed(kind), kind
    assert rt.check_zone(None) is True and rt.event_allowed("click")
