"""v0.1.14: automatic hand roles (hands.roles) and palm-glide scroll (hands.palm_scroll).

Hands are built in a camera frame measured in palm lengths and projected through a pinhole into
MediaPipe-style landmarks (x, y normalised; z relative to the wrist in image-width units), so rotated and
tilted palms are exercised the way the monocular reconstruction sees them.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import yaml

from away import config as C
from away.app.cursor_overlay import COLORS, ring_spec
from away.app.runner import Pipeline
from away.app.settings_gui import PALM_SCROLL_SWITCHES, TUNABLE_LABELS, SettingsModel
from away.bindings import GESTURE_TRIGGERS, Action, BindingsConfig, default_profile
from away.config import AppConfig, PalmScrollConfig
from away.control.mouse import RecordingBackend
from away.hands import palmscroll as PS
from away.hands import roles as R
from away.hands import tracker as T
from away.hands.gestures import GestureStateMachine
from away.hands.palmscroll import PalmScroller, hand_points, outline_distance, palm_frame
from away.hands.roles import HandRoles, hand_size, normalize_mode, palm_center
from away.hands.select import HandObs
from away.hands.synthetic import pose

DT = 1 / 30
F = 1.0 / (2.0 * math.tan(math.radians(PS.DEFAULT_HFOV) / 2.0))
ASPECT = PS.DEFAULT_ASPECT


# ============================================================================ synthetic two-hand scenes
def rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project(P: np.ndarray) -> np.ndarray:
    """Camera points (palm lengths) -> MediaPipe-style landmarks, the inverse of palmscroll.hand_points."""
    P = np.asarray(P, dtype=np.float64)
    zw = P[0, 2]
    x = 0.5 + P[:, 0] * F / P[:, 2]
    y = 0.5 + P[:, 1] * F / (P[:, 2] * ASPECT)
    z = (P[:, 2] - zw) * F / zw
    return np.column_stack([x, y, z])


FLAT = pose("palm", center=(0.0, 0.0), scale=1.0)  # palm-length units, fingers toward -y, z = 0 (flat)


class Helper:
    """A flat helper hand: palm frame placed at ``origin`` (wrist, camera units) with rotation ``R``."""

    def __init__(self, R_: np.ndarray | None = None, origin=(0.0, 0.0, 5.0), template: np.ndarray | None = None):
        self.R = np.eye(3) if R_ is None else R_
        self.origin = np.array(origin, dtype=np.float64)
        self.template = FLAT if template is None else template

    def points(self) -> np.ndarray:
        local = self.template.copy()
        local[:, 2] = 0.0 if self.template is FLAT else local[:, 2]
        return local @ self.R.T + self.origin

    def landmarks(self) -> np.ndarray:
        return project(self.points())

    def at(self, u: float, v: float, w: float = 0.0) -> np.ndarray:
        """Camera point at palm-local (u, v, w): v toward the fingers, u to the user's right (mirrored image)."""
        e_u = self.R @ np.array([1.0, 0.0, 0.0])
        e_v = self.R @ np.array([0.0, -1.0, 0.0])  # wrist -> middle MCP is -y in the template
        e_n = np.cross(e_u, e_v)
        return self.origin + u * e_u + v * e_v + w * e_n


POINT = pose("point", center=(0.0, 0.0), scale=1.0)
TWO = pose("two", center=(0.0, 0.0), scale=1.0)
FIST = pose("fist", center=(0.0, 0.0), scale=1.0)


def pointer_at(tip: np.ndarray, template: np.ndarray = POINT) -> np.ndarray:
    """A pointing hand (facing the camera) with its index fingertip at the camera point ``tip``."""
    P = template.copy() - template[8] + np.asarray(tip)
    return project(P)


def obs(lm: np.ndarray, label: str = "Left", score: float = 0.95) -> HandObs:
    return HandObs(lm, label, score)


def cfg(**kw) -> PalmScrollConfig:
    c = PalmScrollConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def run(sc: PalmScroller, helper: Helper | None, path, t0=0.0, template=POINT, helper_lm=None):
    """Feed fingertip positions (palm-local (u, v, w) tuples, or None = pointer gone). Returns (events, t)."""
    evs = []
    t = t0
    for p in path:
        h = helper_lm if helper_lm is not None else (helper.landmarks() if helper is not None else None)
        ptr = None if p is None else pointer_at(helper.at(*p), template) if helper is not None else None
        evs += sc.update(t, ptr, h)
        t += DT
    return evs, t


def kinds(evs):
    return [e.kind for e in evs]


def total(evs, key):
    return sum(e.data.get(key, 0.0) for e in evs if e.kind == "palm_scroll")


def glide(v0, v1, n, u=0.0, w=0.0):
    return [(u, v, w) for v in np.linspace(v0, v1, n)]


def still(p, n):
    return [p] * n


ORIENTATIONS = {
    "upright": np.eye(3),
    "rolled_35": rot("z", 35),
    "sideways_90": rot("z", 90),
    "upside_down": rot("z", 180),
    "tilted_back_55": rot("x", 55),  # top of the palm leans away from the camera
    "turned_40": rot("y", 40),
    "tilted_and_rolled": rot("x", -40) @ rot("z", -30),
}


# ============================================================================ geometry
def test_hand_points_reconstructs_the_camera_frame():
    h = Helper(rot("x", 30) @ rot("z", 20), origin=(0.3, -0.2, 6.0))
    rec = hand_points(h.landmarks())
    true = h.points()
    # palm geometry (not absolute position) is what matters: distances between landmarks agree
    for i, j in ((0, 9), (5, 17), (0, 12), (4, 20)):
        assert np.linalg.norm(rec[i] - rec[j]) == pytest.approx(np.linalg.norm(true[i] - true[j]), rel=0.08)
    assert rec[0, 2] == pytest.approx(6.0, rel=0.05)  # distance from the apparent size


def test_triangulated_points_are_scaled_to_palm_lengths_and_mirrored():
    pts = np.zeros((21, 3))
    pts[9] = (0.09, 0.0, 0.5)
    out = hand_points(None, pts, 0.09, mirrored=True)
    assert out[9].tolist() == pytest.approx([-1.0, 0.0, 0.5 / 0.09])
    assert hand_points(None, pts, 0.09, mirrored=False)[9, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("name", list(ORIENTATIONS))
def test_palm_frame_axes_follow_the_palm(name):
    h = Helper(ORIENTATIONS[name])
    fr = palm_frame(hand_points(h.landmarks()))
    assert fr is not None
    for local in ((0.0, 0.5, 0.0), (0.2, 0.3, 0.0), (-0.25, 0.7, 0.0)):
        u, v, w = fr.local(hand_points(pointer_at(h.at(*local)))[8])
        assert v == pytest.approx(local[1], abs=0.06)
        assert abs(w) < 0.12
        assert abs(u) == pytest.approx(abs(local[0]), abs=0.06)
    assert outline_distance(fr.outline, (0.0, 0.5)) < 0  # palm centre is inside
    assert outline_distance(fr.outline, (0.0, 3.0)) > 1.0  # far above the fingers is outside


def test_u_points_to_the_users_right_and_stays_continuous():
    fr = palm_frame(hand_points(Helper().landmarks()))
    assert fr is not None and fr.u[0] > 0.9
    rolled = palm_frame(hand_points(Helper(rot("z", 170)).landmarks()), prev_u=fr.u)
    # upside down: no flip back to image-right because the previous frame's u wins
    assert rolled is not None and float(rolled.u @ fr.u) > 0


def test_degenerate_hands_have_no_frame():
    assert palm_frame(np.zeros((21, 3))) is None
    line = np.zeros((21, 3))
    line[:, 1] = np.linspace(0, -2, 21)  # every point on one line: no normal
    assert palm_frame(line) is None
    assert outline_distance(np.zeros((2, 2)), (0, 0)) == math.inf


def test_outline_distance_is_signed():
    sq = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    assert outline_distance(sq, (0.5, 0.5)) == pytest.approx(-0.5)
    assert outline_distance(sq, (1.5, 0.5)) == pytest.approx(0.5)
    assert outline_distance(sq, (1.0, 0.5)) == pytest.approx(0.0)


def test_pointer_fingers_and_helper_flat():
    assert PS.pointer_fingers(POINT) == "index"
    assert PS.pointer_fingers(TWO) == "two"
    assert PS.pointer_fingers(FIST) == "none"
    assert PS.helper_flat(FLAT) and not PS.helper_flat(FIST)
    three = pose("palm", curls={"pinky": 1.0}, center=(0, 0), scale=1.0)
    assert not PS.helper_flat(three) and PS.helper_flat(three, strict=False)


def test_scroll_direction_names():
    assert PS.direction(0, 1) == "up" and PS.direction(0, -1) == "down"
    assert PS.direction(2, 1) == "right" and PS.direction(-2, 1) == "left"
    assert PS.direction(0, 0) is None


# ============================================================================ engage / disengage
def test_defaults():
    c = PalmScrollConfig()
    assert (c.enabled, c.engage_s, c.depth_threshold, c.margin, c.gain, c.dead_zone) == (
        True,
        0.15,
        0.6,
        0.25,
        12.0,
        0.05,
    )
    assert (c.invert, c.horizontal, c.momentum, c.friction) == (False, True, True, 4.0)
    assert AppConfig().hands.roles.mode == "auto" and AppConfig().hands.roles.release_s == 1.0
    assert (PS.TAP_MAX_S, PS.OCCLUSION_HOLD_S) == (0.25, 0.3)


@pytest.mark.parametrize("name", list(ORIENTATIONS))
def test_engages_on_rotated_and_tilted_palms_and_scrolls_toward_the_fingers(name):
    h = Helper(ORIENTATIONS[name])
    sc = PalmScroller(cfg(momentum=False))
    evs, t = run(sc, h, still((0.0, 0.3, 0.0), 8))
    assert "palm_scroll_start" in kinds(evs) and sc.engaged
    up, t = run(sc, h, glide(0.3, 0.8, 12) + still((0.0, 0.8, 0.0), 20), t)
    assert total(up, "dy") > 3.0 and total(up, "dx") == 0.0
    down, t = run(sc, h, glide(0.8, 0.2, 12) + still((0.0, 0.2, 0.0), 20), t)
    assert total(down, "dy") < -4.0


def test_engage_needs_engage_s():
    h = Helper()
    sc = PalmScroller(cfg(engage_s=0.15))
    evs, t = run(sc, h, still((0.0, 0.4, 0.0), 4))  # 0.1 s of contact
    assert "palm_scroll_start" not in kinds(evs) and sc.touching and sc.active and not sc.engaged
    evs, t = run(sc, h, still((0.0, 0.4, 0.0), 2), t)  # reaches 0.15 s
    assert kinds(evs) == ["palm_scroll_start"]


def test_fingertip_outside_the_palm_does_not_engage():
    h = Helper()
    sc = PalmScroller()
    evs, _ = run(sc, h, still((1.6, 0.4, 0.0), 20))  # beside the palm
    assert evs == [] and not sc.active
    sc2 = PalmScroller()
    evs, _ = run(sc2, h, still((0.0, 2.4, 0.0), 20))  # far above the fingers
    assert evs == [] and not sc2.active


def test_margin_pads_the_palm_outline():
    h = Helper()
    edge = (0.62, 0.4, 0.0)  # just beyond the index side of the palm
    assert not any(e.kind == "palm_scroll_start" for e in run(PalmScroller(cfg(margin=0.0)), h, still(edge, 10))[0])
    assert any(e.kind == "palm_scroll_start" for e in run(PalmScroller(cfg(margin=0.4)), h, still(edge, 10))[0])


def test_hovering_fingertip_does_not_engage_unless_the_depth_check_is_off():
    h = Helper(rot("x", 30))
    hover = (0.0, 0.4, 1.5)  # 1.5 palm lengths in front of the palm
    assert run(PalmScroller(), h, still(hover, 15))[0] == []
    evs, _ = run(PalmScroller(cfg(depth_threshold=0.0)), h, still(hover, 15))
    assert "palm_scroll_start" in kinds(evs)
    evs, _ = run(PalmScroller(cfg(depth_threshold=2.0)), h, still(hover, 15))
    assert "palm_scroll_start" in kinds(evs)


def test_helper_must_be_flat():
    fist = Helper(template=FIST)
    sc = PalmScroller()
    evs, _ = run(sc, fist, still((0.0, 0.3, 0.0), 20))
    assert evs == [] and not sc.active


def test_pointer_without_the_index_finger_out_does_not_engage():
    h = Helper()
    sc = PalmScroller()
    evs, _ = run(sc, h, still((0.0, 0.4, 0.0), 20), template=FIST)
    assert evs == [] and not sc.active


def test_lifting_the_fingertip_disengages():
    h = Helper()
    sc = PalmScroller(cfg(momentum=False))
    evs, t = run(sc, h, still((0.0, 0.4, 0.0), 10))  # 0.3 s: longer than a tap
    assert sc.engaged
    evs, t = run(sc, h, still((0.0, 0.4, 2.0), 6), t)  # lifted off the palm
    assert kinds(evs) == ["palm_scroll_stop"] and evs[0].data["reason"] == "lift"
    assert not sc.active and not sc.engaged


def test_short_contact_dropout_does_not_disengage():
    h = Helper()
    sc = PalmScroller(cfg(momentum=False))
    _, t = run(sc, h, still((0.0, 0.4, 0.0), 10))
    evs, t = run(sc, h, [None, None], t)  # the pointer's landmarks drop out for two frames (< RELEASE_GRACE_S)
    assert evs == [] and sc.engaged and sc.active
    evs, t = run(sc, h, still((0.0, 0.4, 0.0), 2), t)
    assert sc.engaged and "palm_scroll_start" not in kinds(evs)
    evs, t = run(sc, h, [(0.0, 0.4, 0.75)], t)  # one frame just off the palm (depth noise)
    assert evs == [] and sc.engaged


def test_pointer_lost_disengages_with_its_reason():
    h = Helper()
    sc = PalmScroller(cfg(momentum=False))
    _, t = run(sc, h, still((0.0, 0.4, 0.0), 10))
    evs, _ = run(sc, h, [None] * 5, t)
    assert kinds(evs) == ["palm_scroll_stop"] and evs[0].data["reason"] == "pointer_lost"


def test_disabled_does_nothing_and_ends_a_running_glide():
    h = Helper()
    assert run(PalmScroller(cfg(enabled=False)), h, still((0.0, 0.4, 0.0), 20))[0] == []
    c = cfg()
    sc = PalmScroller(c)
    _, t = run(sc, h, still((0.0, 0.4, 0.0), 8))
    c.enabled = False  # Settings switched it off (live config object)
    evs = sc.update(t, pointer_at(h.at(0.0, 0.4)), h.landmarks())
    assert kinds(evs) == ["palm_scroll_stop"] and evs[0].data["reason"] == "disabled" and not sc.active


# ============================================================================ local-frame scrolling
def test_invert_swaps_the_direction():
    h = Helper()
    sc = PalmScroller(cfg(invert=True, momentum=False))
    evs, _ = run(sc, h, still((0.0, 0.3, 0.0), 6) + glide(0.3, 0.8, 12) + still((0.0, 0.8, 0.0), 20))
    assert total(evs, "dy") < -3.0


def test_two_fingers_scroll_horizontally_along_u():
    h = Helper(rot("z", 25))
    sc = PalmScroller(cfg(momentum=False))
    path = still((-0.2, 0.4, 0.0), 6) + [(u, 0.4, 0.0) for u in np.linspace(-0.2, 0.25, 12)] + still((0.25, 0.4, 0), 20)
    evs, t = run(sc, h, path, template=TWO)
    assert total(evs, "dx") > 2.0 and total(evs, "dy") == 0.0
    assert sc.direction == "right"
    left, _ = run(sc, h, [(u, 0.4, 0.0) for u in np.linspace(0.25, -0.2, 12)] + still((-0.2, 0.4, 0), 20), t, TWO)
    assert total(left, "dx") < -2.0 and sc.direction == "left"


def test_two_fingers_scroll_vertically_when_horizontal_is_off():
    h = Helper()
    sc = PalmScroller(cfg(horizontal=False, momentum=False))
    evs, _ = run(sc, h, still((0.0, 0.3, 0.0), 6) + glide(0.3, 0.8, 12) + still((0, 0.8, 0), 20), template=TWO)
    assert total(evs, "dy") > 3.0 and total(evs, "dx") == 0.0


def test_gain_is_notches_per_palm_length_beyond_the_dead_zone():
    h = Helper()
    sc = PalmScroller(cfg(gain=10.0, dead_zone=0.05, momentum=False, engage_s=0.0))
    evs, _ = run(sc, h, still((0.0, 0.2, 0.0), 10) + glide(0.2, 0.7, 15) + still((0.0, 0.7, 0.0), 60))
    # 0.5 palm lengths of glide, minus the 0.05 dead zone, at 10 notches per palm length
    assert total(evs, "dy") == pytest.approx(4.5, abs=0.35)


def test_dead_zone_swallows_jitter():
    h = Helper()
    sc = PalmScroller(cfg(dead_zone=0.05, momentum=False))
    rng = np.random.default_rng(3)
    jitter = [(float(rng.uniform(-0.02, 0.02)), 0.4 + float(rng.uniform(-0.02, 0.02)), 0.0) for _ in range(60)]
    evs, _ = run(sc, h, still((0.0, 0.4, 0.0), 6) + jitter)
    assert sc.engaged and not any(e.kind == "palm_scroll" for e in evs)


def test_moving_palm_under_a_still_fingertip_scrolls_too():
    """Relative glide: the helper sliding down under the fingertip = the fingertip moving toward the fingers."""
    sc = PalmScroller(cfg(momentum=False))
    tip_cam = Helper().at(0.0, 0.3)
    evs = []
    t = 0.0
    for k in range(40):
        dy = 0.0 if k < 8 else min(0.5, (k - 8) * 0.04)
        h = Helper(origin=(0.0, dy, 5.0))  # palm moves down in the image (+y)
        evs += sc.update(t, pointer_at(tip_cam), h.landmarks())
        t += DT
    assert total(evs, "dy") > 3.0


# ============================================================================ momentum
def fast_glide_then_lift(sc, h):
    evs, t = run(sc, h, still((0.0, 0.2, 0.0), 6) + glide(0.2, 0.9, 6))
    lift, t = run(sc, h, still((0.0, 0.9, 2.5), 4), t)
    return evs, lift, t


def test_momentum_decays_exponentially_after_a_lift():
    h = Helper()
    c = cfg(friction=4.0)
    sc = PalmScroller(c)
    _, lift, t = fast_glide_then_lift(sc, h)
    assert "palm_scroll_stop" in kinds(lift) and sc.momentum
    coast, t = run(sc, h, [None] * 3 + still((0.0, 0.9, 2.5), 2), t)
    steps = [e.data["dy"] for e in coast if e.kind == "palm_scroll"]
    assert len(steps) >= 4 and all(e.data.get("momentum") for e in coast if e.kind == "palm_scroll")
    assert all(s > 0 for s in steps)
    for a, b in itertools.pairwise(steps):
        assert b / a == pytest.approx(math.exp(-4.0 * DT), rel=1e-6)
    # it runs out on its own
    rest, _ = run(sc, h, still((0.0, 0.9, 2.5), 90), t)
    assert kinds(rest)[-1] == "palm_scroll_stop" and rest[-1].data["reason"] == "momentum_end"
    assert not sc.momentum


def test_higher_friction_stops_sooner():
    def coast_frames(friction):
        h = Helper()
        sc = PalmScroller(cfg(friction=friction))
        _, _, t = fast_glide_then_lift(sc, h)
        n = 0
        while sc.momentum and n < 400:
            sc.update(t, None, h.landmarks())
            t += DT
            n += 1
        return n

    assert coast_frames(8.0) < coast_frames(2.0)


def test_momentum_off_stops_dead():
    h = Helper()
    sc = PalmScroller(cfg(momentum=False))
    _, lift, t = fast_glide_then_lift(sc, h)
    assert not sc.momentum
    assert not any(e.kind == "palm_scroll" for e in run(sc, h, still((0.0, 0.9, 2.5), 10), t)[0])


def test_slow_glide_has_no_momentum():
    h = Helper()
    sc = PalmScroller()
    _, t = run(sc, h, still((0.0, 0.3, 0.0), 6) + glide(0.3, 0.35, 30) + still((0.0, 0.35, 0.0), 30))
    run(sc, h, still((0.0, 0.35, 2.5), 4), t)
    assert not sc.momentum


def test_momentum_is_stopped_by_a_pointer_gesture_and_by_re_engaging():
    h = Helper()
    sc = PalmScroller()
    _, _, t = fast_glide_then_lift(sc, h)
    assert sc.stop_momentum() and not sc.momentum and sc.vy == 0.0
    assert not sc.stop_momentum()  # nothing to stop
    sc2 = PalmScroller()
    _, _, t = fast_glide_then_lift(sc2, h)
    assert sc2.momentum
    evs, _ = run(sc2, h, still((0.0, 0.5, 0.0), 2), t)  # fingertip back on the palm
    assert not sc2.momentum and sc2.touching
    assert not any(e.data.get("momentum") for e in evs)


def test_frame_stall_never_bursts():
    h = Helper()
    sc = PalmScroller()
    _, _, t = fast_glide_then_lift(sc, h)
    evs = sc.update(t + 5.0, None, h.landmarks())  # 5 s later
    vy_before = None
    for e in evs:
        if e.kind == "palm_scroll":
            vy_before = e.data["dy"]
    assert vy_before is None or abs(vy_before) < 3.0


# ============================================================================ tap
def test_quick_tap_fires_palm_tap():
    h = Helper()
    sc = PalmScroller()
    evs, t = run(sc, h, still((0.0, 0.4, 2.0), 3) + still((0.0, 0.4, 0.0), 4) + still((0.0, 0.4, 2.0), 3))
    assert kinds(evs) == ["palm_tap"]
    assert 0 < evs[0].data["duration"] <= PS.TAP_MAX_S


def test_tap_after_engage_but_within_the_tap_time_also_counts():
    h = Helper()
    sc = PalmScroller(cfg(momentum=False))
    evs, _ = run(sc, h, still((0.0, 0.4, 0.0), 7) + still((0.0, 0.4, 2.0), 4))  # 0.2 s touch
    assert kinds(evs) == ["palm_scroll_start", "palm_tap", "palm_scroll_stop"]


def test_long_touch_is_not_a_tap():
    h = Helper()
    sc = PalmScroller()
    evs, _ = run(sc, h, still((0.0, 0.4, 0.0), 12) + still((0.0, 0.4, 2.0), 4))  # 0.37 s
    assert "palm_tap" not in kinds(evs)


def test_moving_touch_is_not_a_tap():
    h = Helper()
    sc = PalmScroller(cfg(engage_s=0.5))
    evs, _ = run(sc, h, glide(0.2, 0.45, 6) + still((0.0, 0.45, 2.0), 4))
    assert "palm_tap" not in kinds(evs)


def test_single_frame_touch_is_noise():
    h = Helper()
    sc = PalmScroller()
    evs, _ = run(sc, h, still((0.0, 0.4, 2.0), 3) + [(0.0, 0.4, 0.0)] + still((0.0, 0.4, 2.0), 4))
    assert "palm_tap" not in kinds(evs)


def test_tap_trigger_is_bindable_and_unbound_by_default():
    assert "palm_tap" in GESTURE_TRIGGERS
    assert default_profile().gestures["palm_tap"] == Action("none")
    p = Pipeline(AppConfig(), RecordingBackend())
    b = p.cfg.bindings
    b.set("gesture:palm_tap", "key:space")
    p.dispatcher.set_bindings(b)
    h = Helper()
    t = 0.0
    for p_ in still((0.0, 0.4, 2.0), 3) + still((0.0, 0.4, 0.0), 4) + still((0.0, 0.4, 2.0), 3):
        p.on_hand(t, pointer_at(h.at(*p_)), "Left", helper=h.landmarks())
        t += DT
    assert ("key", ("space",)) in p.mouse.backend.calls
    assert ("gesture:palm_tap", "key:space") in p.dispatcher.fired


# ============================================================================ occlusion
def test_helper_dropout_is_held_briefly():
    h = Helper()
    sc = PalmScroller(cfg(momentum=False))
    _, t = run(sc, h, still((0.0, 0.3, 0.0), 8))
    assert sc.engaged
    tip = pointer_at(h.at(0.0, 0.5))
    evs = []
    for _ in range(6):  # 0.2 s with no helper landmarks (the fingertip covers it)
        evs += sc.update(t, tip, None)
        t += DT
    assert sc.engaged and "palm_scroll_stop" not in kinds(evs)
    assert total(evs, "dy") > 0  # the glide toward the fingers is measured in the held palm frame
    for _ in range(12):  # past OCCLUSION_HOLD_S + the release grace
        evs += sc.update(t, tip, None)
        t += DT
    assert not sc.engaged and "palm_scroll_stop" in kinds(evs)


def test_helper_jump_while_covered_uses_the_last_good_frame():
    h = Helper()
    sc = PalmScroller(cfg(momentum=False))
    _, t = run(sc, h, still((0.0, 0.4, 0.0), 8))
    tip = pointer_at(h.at(0.0, 0.4))
    bad = Helper(origin=(2.5, 1.0, 5.0)).landmarks()  # a misdetection two palm lengths away
    evs = []
    for _ in range(3):
        evs += sc.update(t, tip, bad)
        t += DT
    assert sc.engaged and total(evs, "dy") == 0.0 and total(evs, "dx") == 0.0


def test_helper_finger_hidden_by_the_fingertip_keeps_it_flat_enough():
    three = Helper(template=pose("palm", curls={"ring": 1.0}, center=(0, 0), scale=1.0))
    sc = PalmScroller(cfg(momentum=False))
    _, t = run(sc, Helper(), still((0.0, 0.3, 0.0), 8))
    evs, _ = run(sc, three, still((0.0, 0.3, 0.0), 20), t)
    assert sc.engaged and "palm_scroll_stop" not in kinds(evs)
    # but three fingers are not enough to start
    assert run(PalmScroller(), three, still((0.0, 0.3, 0.0), 20))[0] == []


def test_fingertip_is_smoothed():
    h = Helper()
    sc = PalmScroller(cfg(momentum=False, dead_zone=0.0))
    _, t = run(sc, h, still((0.0, 0.3, 0.0), 8))
    evs, _ = run(sc, h, [(0.0, 0.6, 0.0)], t)  # a 0.3 palm-length step in one frame
    assert 0 < total(evs, "dy") < 0.3 * 12.0  # filtered: less than the raw step


def test_uses_3d_points_when_both_hands_have_them():
    h = Helper(rot("x", 60))
    sc = PalmScroller(cfg(momentum=False))
    sc.mirrored = False
    t = 0.0
    evs = []
    for p in still((0.0, 0.3, 0.0), 8) + glide(0.3, 0.8, 12) + still((0.0, 0.8, 0.0), 20):
        hp = h.points() * 0.09
        pp = (POINT - POINT[8] + h.at(*p)) * 0.09
        # 2D landmarks deliberately useless: the 3D path must be what engages
        evs += sc.update(t, pointer_at(h.at(*p)), h.landmarks(), pp, hp, 0.09)
        t += DT
    assert "palm_scroll_start" in kinds(evs) and total(evs, "dy") > 3.0


# ============================================================================ suppression (pipeline + state machine)
def test_suppressed_hand_produces_no_clutch_autoscroll_click_or_move():
    sm = GestureStateMachine()
    out = []
    t = 0.0
    for name in ("palm", "four", "pinch", "point", "two"):
        for _ in range(40):
            out += sm.update(t, pose(name), suppress=True)
            t += DT
    assert out == [] and not sm.paused and not sm.autoscroll.active


def test_suppress_ends_a_running_drag_and_autoscroll():
    sm = GestureStateMachine()
    t = 0.0
    for _ in range(20):
        sm.update(t, pose("pinch"))
        t += DT
    assert sm.dragging
    evs = sm.update(t, pose("pinch"), suppress=True)
    assert [e.kind for e in evs] == ["drag_end"] and not sm.dragging
    sm2 = GestureStateMachine()
    for _ in range(15):
        sm2.update(t, pose("four"))
        t += DT
    assert sm2.autoscroll.active
    assert [e.kind for e in sm2.update(t, pose("four"), suppress=True)] == ["autoscroll_stop"]


def test_block_clutch_until_the_hand_changes_pose():
    sm = GestureStateMachine()
    sm.block_clutch()
    t = 0.0
    for _ in range(40):
        assert not any(e.kind == "clutch" for e in sm.update(t, pose("palm")))
        t += DT
    sm.update(t, pose("point"))
    t += DT
    got = []
    for _ in range(20):
        got += sm.update(t, pose("palm"))
        t += DT
    assert any(e.kind == "clutch" for e in got)


def scene(p: Pipeline, helper: Helper, path, t=0.0, template=POINT):
    acc = []
    for q in path:
        acc += p.on_hand(t, pointer_at(helper.at(*q), template), "Left", helper=helper.landmarks())
        t += DT
    return acc, t


def test_pipeline_palm_scroll_suppresses_cursor_and_scrolls_the_wheel():
    p = Pipeline(AppConfig(), RecordingBackend())
    h = Helper()
    p.cfg.hands.palm_scroll.momentum = False
    acc, t = scene(p, h, still((0.0, 0.3, 0.0), 8) + glide(0.3, 0.8, 12) + still((0.0, 0.8, 0.0), 10))
    calls = p.mouse.backend.calls
    assert "palm_scroll_start" in acc and "palm_scroll" in acc
    assert "move" not in acc[acc.index("palm_scroll_start") :]
    assert sum(c[1][0] for c in calls if c[0] == "scroll") > 3.0
    assert not any(c[0] in ("click", "press") for c in calls)
    assert p.cursor_state()["state"] == "palmscroll" and p.cursor_state()["direction"] == "up"


def test_pipeline_helper_flat_palm_never_pauses_and_two_hands_do_not_autoscroll():
    p = Pipeline(AppConfig(), RecordingBackend())
    h = Helper(origin=(1.5, 0.0, 5.0))  # helper flat beside the pointer, not touched
    t = 0.0
    acc = []
    for _ in range(90):  # 3 s: far longer than clutch_hold and autoscroll_hold
        acc += p.on_hand(t, pointer_at(np.array([-1.5, -1.0, 5.0])), "Left", helper=h.landmarks())
        t += DT
    assert not p.paused and "clutch" not in acc and "autoscroll_start" not in acc
    # and while engaged, even a four-finger / open-palm pointer does nothing
    acc2, t = scene(p, Helper(), still((0.0, 0.4, 0.0), 8), t)
    for _ in range(40):
        acc2 += p.on_hand(t, pointer_at(Helper().at(0.0, 0.4)), "Left", helper=Helper().landmarks())
        t += DT
    assert not p.paused and "clutch" not in acc2 and "autoscroll_start" not in acc2


def test_pipeline_pointer_gesture_stops_momentum():
    p = Pipeline(AppConfig(), RecordingBackend())
    h = Helper()
    _, t = scene(p, h, still((0.0, 0.2, 0.0), 6) + glide(0.2, 0.9, 6) + still((0.0, 0.9, 2.5), 5))
    assert p.palm.momentum
    for _ in range(12):  # pinch somewhere else: a click/drag ends the coast
        p.on_hand(t, pose("pinch", center=(0.2, 0.5)), "Left")
        t += DT
    assert not p.palm.momentum


def test_pipeline_paused_does_not_scroll_and_promoted_pointer_does_not_pause():
    p = Pipeline(AppConfig(), RecordingBackend())
    p.set_paused(True)
    scene(p, Helper(), still((0.0, 0.3, 0.0), 8) + glide(0.3, 0.8, 12))
    assert not any(c[0] == "scroll" for c in p.mouse.backend.calls)
    q = Pipeline(AppConfig(), RecordingBackend())
    t = 0.0
    acc = []
    for i in range(40):  # the former helper (flat palm) is now the only hand and the pointer
        acc += q.on_hand(t, pose("palm"), "Left", promoted=i == 0)
        t += DT
    assert "clutch" not in acc and not q.paused


def test_pipeline_palm_scroll_disabled():
    c = AppConfig()
    c.hands.palm_scroll.enabled = False
    p = Pipeline(c, RecordingBackend())
    acc, _ = scene(p, Helper(), still((0.0, 0.3, 0.0), 8) + glide(0.3, 0.8, 12))
    assert "palm_scroll" not in acc and "move" in acc


def test_pipeline_fused_helper_reaches_palm_scroll():
    from away.multicam.fusion import FusedHand

    p = Pipeline(AppConfig(), RecordingBackend())
    h = Helper()
    t = 0.0
    acc = []
    for q in still((0.0, 0.3, 0.0), 8) + glide(0.3, 0.8, 12) + still((0.0, 0.8, 0.0), 10):
        ptr = FusedHand(t, pointer_at(h.at(*q)), "Left", "cam0")
        hlp = FusedHand(t, h.landmarks(), "Right", "cam0")
        acc += p.on_fused(ptr, t, helper=hlp, pointer_id=1, helper_id=2)
        t += DT
    assert "palm_scroll" in acc


def test_palm_scroll_events_reach_the_horizontal_wheel():
    p = Pipeline(AppConfig(), RecordingBackend())
    p.cfg.hands.palm_scroll.momentum = False
    h = Helper()
    path = still((-0.2, 0.4, 0.0), 6) + [(u, 0.4, 0.0) for u in np.linspace(-0.2, 0.3, 12)] + still((0.3, 0.4, 0), 10)
    scene(p, h, path, template=TWO)
    assert sum(c[1][0] for c in p.mouse.backend.calls if c[0] == "hscroll") > 2.0


def test_overlay_shows_a_palm_scroll_indicator():
    spec = ring_spec({"state": "palmscroll", "direction": "down"})
    assert spec["color"] == COLORS["palmscroll"] and spec["arrow"] == "down" and spec["visible"]
    assert ring_spec({"state": "palmscroll", "direction": "sideways"})["arrow"] is None


# ============================================================================ roles
def hand(x, y=0.5, name="point", scale=0.12):
    return pose(name, center=(x, y), scale=scale)


def test_role_helpers():
    lm = hand(0.4, 0.6)
    cx, cy = palm_center(lm)
    assert abs(cx - 0.4) < 0.1 and abs(cy - 0.6) < 0.15
    assert hand_size(lm) == pytest.approx(0.12, rel=0.05)
    assert normalize_mode("RIGHT") == "right" and normalize_mode(None) == "auto" and normalize_mode("x") == "auto"
    assert R.RELEASE_S == 1.0 and R.MODES == ("auto", "right", "left", "any")


def test_only_hand_in_view_is_the_pointer_whatever_its_pose():
    r = HandRoles("auto")
    res = r.update(0.0, [obs(hand(0.5, name="palm"))])
    assert res.pointer is not None and res.helper is None and not res.promoted


def test_auto_first_pointing_hand_gets_the_pointer_and_keeps_it():
    r = HandRoles("auto")
    a, b = hand(0.3, name="palm"), hand(0.7, name="fist")
    res = r.update(0.0, [obs(a, "Right"), obs(b, "Left")])
    assert res.pointer is None and res.helper is not None  # two hands, nobody points yet
    res = r.update(DT, [obs(hand(0.3, name="palm"), "Right"), obs(hand(0.7, name="point"), "Left")])
    assert res.pointer is not None and palm_center(res.pointer.landmarks)[0] > 0.6
    pid = res.pointer_id
    # now the other hand points as well, and the pointer opens its palm: the role stays put (sticky)
    for i in range(30):
        t = 2 * DT + i * DT
        res = r.update(t, [obs(hand(0.3, name="point"), "Right"), obs(hand(0.7, name="palm"), "Left")])
        assert res.pointer_id == pid and palm_center(res.helper.landmarks)[0] < 0.4


def test_handedness_flips_never_swap_roles():
    r = HandRoles("auto")
    t = 0.0
    res = r.update(t, [obs(hand(0.65), "Left")])  # pointer first, alone
    pid = res.pointer_id
    for i in range(60):
        t += DT
        flip = i % 2 == 0
        hands = [
            obs(hand(0.65 + 0.001 * i), "Right" if flip else "Left"),
            obs(hand(0.3, name="palm"), "Left" if flip else "Right"),
        ]
        res = r.update(t, hands[::-1] if i % 3 == 0 else hands)
        assert res.pointer_id == pid
        assert palm_center(res.pointer.landmarks)[0] > 0.6


def test_identity_follows_position_when_both_labels_say_the_same_side():
    r = HandRoles("auto")
    res = r.update(0.0, [obs(hand(0.7), "Left")])
    pid = res.pointer_id
    for i in range(20):  # MediaPipe calls both hands "Left"; the hands drift toward each other
        res = r.update(
            (i + 1) * DT, [obs(hand(0.3 + 0.005 * i, name="palm"), "Left"), obs(hand(0.7 - 0.005 * i), "Left")]
        )
        assert res.pointer_id == pid and res.helper_id not in (None, pid)


def test_pointer_survives_a_short_loss_and_is_reacquired():
    r = HandRoles("auto", release_s=1.0)
    res = r.update(0.0, [obs(hand(0.7), "Left")])
    pid = res.pointer_id
    r.update(DT, [obs(hand(0.7), "Left"), obs(hand(0.3, name="palm"), "Right")])
    t = 2 * DT
    for _ in range(15):  # 0.5 s: only the helper is seen
        res = r.update(t, [obs(hand(0.3, name="palm"), "Right")])
        assert res.pointer is None and res.helper is not None  # helper does not take over
        t += DT
    res = r.update(t, [obs(hand(0.55), "Left"), obs(hand(0.3, name="palm"), "Right")])  # back, moved a lot
    assert res.pointer_id == pid


def test_pointer_released_after_release_s_and_the_helper_is_promoted():
    r = HandRoles("auto", release_s=0.5)
    r.update(0.0, [obs(hand(0.7), "Left")])
    res = r.update(DT, [obs(hand(0.7), "Left"), obs(hand(0.3, name="palm"), "Right")])
    helper_id = res.helper_id
    t = 2 * DT
    res = r.update(t, [obs(hand(0.3, name="palm"), "Right")])
    while t < 0.5:
        t += DT
        res = r.update(t, [obs(hand(0.3, name="palm"), "Right")])
        assert res.pointer is None or t > 0.5
    t += 2 * DT
    res = r.update(t, [obs(hand(0.3, name="palm"), "Right")])
    assert res.pointer_id == helper_id and res.promoted
    res = r.update(t + DT, [obs(hand(0.3, name="palm"), "Right")])
    assert not res.promoted


def test_release_s_boundary():
    r = HandRoles("auto", release_s=lambda: 0.3)
    r.update(0.0, [obs(hand(0.7), "Left")])
    r.update(0.1, [obs(hand(0.7), "Left"), obs(hand(0.3, name="palm"), "Right")])
    assert r.update(0.39, [obs(hand(0.3, name="palm"), "Right")]).pointer_id is None
    assert r.pointer_id is not None  # still held at 0.29 s gone
    r.update(0.41, [obs(hand(0.3, name="palm"), "Right")])
    assert r.pointer_id is not None and r.tracks[r.pointer_id].visible  # released: the helper is now the pointer


def test_pointer_that_left_the_frame_is_released_early():
    r = HandRoles("auto", release_s=5.0)
    r.update(0.0, [obs(hand(0.985), "Left"), obs(hand(0.3, name="palm"), "Right")])
    r.update(DT, [obs(hand(0.985), "Left"), obs(hand(0.3, name="palm"), "Right")])
    assert r.pointer_id is not None
    first = r.pointer_id
    t = 2 * DT
    for _ in range(8):  # 0.27 s gone, last seen at the right border
        res = r.update(t, [obs(hand(0.3, name="palm"), "Right")])
        t += DT
    assert res.pointer_id not in (None, first)


def test_mode_right_and_left_use_the_smoothed_real_side():
    r = HandRoles("right", mirrored=True)
    right_hand, left_hand = hand(0.7), hand(0.3)
    res = r.update(0.0, [obs(left_hand, "Right"), obs(right_hand, "Left")])  # mirrored: real right = "Left"
    assert palm_center(res.pointer.landmarks)[0] > 0.6 and res.pointer_side == "right"
    pid = res.pointer_id
    for i in range(6):  # both labels flip for a few frames: no swap
        res = r.update((i + 1) * DT, [obs(left_hand, "Left"), obs(right_hand, "Right")])
        assert res.pointer_id == pid
    l_ = HandRoles("left", mirrored=True)
    res = l_.update(0.0, [obs(left_hand, "Right"), obs(right_hand, "Left")])
    assert palm_center(res.pointer.landmarks)[0] < 0.4
    assert HandRoles("right", mirrored=False).update(0.0, [obs(left_hand, "Right")]).pointer is not None


def test_mode_right_ignores_the_left_hand_alone_and_asks_for_more():
    r = HandRoles("right", mirrored=True)
    hands = [obs(hand(0.3), "Right")]  # the real left hand
    res = r.update(0.0, hands)
    assert res.pointer is None and r.wants_more(hands)
    assert not HandRoles("auto").wants_more(hands) and not r.wants_more([])


def test_mode_any_takes_the_first_hand_whatever_its_pose():
    r = HandRoles("any")
    res = r.update(0.0, [obs(hand(0.3, name="palm"), "Right"), obs(hand(0.7, name="fist"), "Left")])
    assert res.pointer is not None and res.helper is not None


def test_changing_the_mode_picks_again():
    state = {"m": "right"}
    r = HandRoles(lambda: state["m"], mirrored=True)
    hands = [obs(hand(0.3), "Right"), obs(hand(0.7), "Left")]
    assert palm_center(r.update(0.0, hands).pointer.landmarks)[0] > 0.6
    state["m"] = "left"
    assert palm_center(r.update(DT, hands).pointer.landmarks)[0] < 0.4
    r.mode = "auto"
    assert r.pointer_id is None


def test_size_term_separates_a_near_and_a_far_hand():
    r = HandRoles("auto")
    r.update(0.0, [obs(hand(0.5, 0.5, scale=0.2), "Left")])
    pid = r.pointer_id
    # a small far hand appears right next to the pointer's centre; the big one moved a little
    res = r.update(DT, [obs(hand(0.52, 0.5, name="palm", scale=0.07), "Left"), obs(hand(0.47, 0.5, scale=0.2), "Left")])
    assert res.pointer_id == pid and hand_size(res.pointer.landmarks) > 0.15


def test_tracker_reports_pointer_and_helper():
    class Fake:
        def __init__(self, n):
            self.n = n

        def detect_for_video(self, image, ms):
            class P:
                def __init__(self, x, y, z):
                    self.x, self.y, self.z = x, y, z

            class Cat:
                def __init__(self, name):
                    self.category_name, self.score = name, 0.9

            class Res:
                pass

            hands = [(hand(0.7), "Left"), (hand(0.3, name="palm"), "Right")][: self.n]
            res = Res()
            res.hand_landmarks = [[P(*row) for row in lm] for lm, _ in hands]
            res.handedness = [[Cat(lab)] for _, lab in hands]
            return res

        def close(self):
            pass

    tr = T.HandTracker(None, 2, factory=Fake, probe_every=1)
    fr = tr.process(None, 0.0)
    assert fr.landmarks is not None and fr.helper is not None
    assert fr.pointer_id is not None and fr.helper_id not in (None, fr.pointer_id)
    assert palm_center(fr.landmarks)[0] > 0.6 and fr.hand == "right" and fr.helper_handedness == "Right"


# ============================================================================ config, voice, settings
def test_prefer_migrates_to_roles_mode():
    for old, new in (("right", "right"), ("left", "left"), ("any", "auto")):
        c, unknown = C.from_dict({"config_version": 3, "hands": {"prefer": old}})
        assert c.hands.roles.mode == new and unknown == []
        assert any("hands.prefer" in n for n in c.migrations)
    with pytest.raises(ValueError, match=r"hands\.prefer"):
        C.from_dict({"hands": {"prefer": "both"}})
    c, _ = C.from_dict({"hands": {"prefer": "right", "roles": {"mode": "left"}}})
    assert c.hands.roles.mode == "left"  # an explicit roles.mode wins
    assert "prefer" not in C.to_dict(c)["hands"] and C.to_dict(c)["config_version"] == C.CONFIG_VERSION == 4


def test_migration_does_not_touch_the_callers_dict():
    raw = {
        "config_version": 3,
        "hands": {"prefer": "right"},
        "bindings": {"profiles": {"default": {"voice": {"use either hand": "hands:any"}}}},
    }
    C.from_dict(raw)
    assert raw["hands"] == {"prefer": "right"}
    assert raw["bindings"]["profiles"]["default"]["voice"]["use either hand"] == "hands:any"


def test_old_saved_either_hand_binding_restores_auto(tmp_path):
    old = AppConfig()
    d = C.to_dict(old)
    d["config_version"] = 3
    d["hands"]["prefer"] = "any"
    d["bindings"]["profiles"]["default"]["voice"]["use either hand"] = "hands:any"
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(d), "utf-8")
    c, unknown = C.load(p)
    assert unknown == [] and c.hands.roles.mode == "auto"
    assert c.bindings.resolve("voice", "use either hand") == Action("hands", "auto")
    d4 = C.to_dict(c)
    d4["bindings"]["profiles"]["default"]["voice"]["use either hand"] = "hands:any"  # chosen in v0.1.14: kept
    c4, _ = C.from_dict(d4)
    assert c4.bindings.resolve("voice", "use either hand") == Action("hands", "any")


def test_roles_and_palm_scroll_validation():
    with pytest.raises(ValueError, match=r"hands\.roles\.mode"):
        C.from_dict({"hands": {"roles": {"mode": "both"}}})
    with pytest.raises(ValueError, match=r"hands\.palm_scroll\.invert"):
        C.from_dict({"hands": {"palm_scroll": {"invert": "yes"}}})
    with pytest.raises(ValueError, match=r"hands\.palm_scroll\.gain"):
        C.from_dict({"hands": {"palm_scroll": {"gain": 0.0}}})
    with pytest.raises(ValueError, match=r"hands\.roles\.release_s"):
        C.from_dict({"hands": {"roles": {"release_s": 9.0}}})
    c, _ = C.from_dict({"hands": {"roles": {"mode": "any", "release_s": 2.0}, "palm_scroll": {"friction": 6}}})
    assert c.hands.roles.mode == "any" and c.hands.palm_scroll.friction == 6


def test_voice_commands_set_and_restore_the_roles():
    p = Pipeline(AppConfig(), RecordingBackend())
    assert p.hand_mode == "auto"
    assert p.fire("voice", "use right hand") == Action("hands", "right") and p.hand_mode == "right"
    assert p.fire("voice", "use left hand") == Action("hands", "left") and p.hand_mode == "left"
    assert p.fire("voice", "use either hand") == Action("hands", "auto") and p.hand_mode == "auto"
    p.fire("voice", "use right hand")
    p.apply_config(AppConfig())  # Settings unchanged (auto == auto): the voice override stays
    assert p.hand_mode == "right"
    p.executor.run(Action("hands", "any"))
    assert p.hand_mode == "any"
    p.executor.run(Action("hands", "both"))  # not a mode: ignored
    assert p.hand_mode == "any"
    assert BindingsConfig().resolve("voice", "use either hand") == Action("hands", "auto")


def test_apply_config_updates_palm_scroll_live():
    p = Pipeline(AppConfig(), RecordingBackend())
    new = AppConfig()
    new.hands.palm_scroll.gain = 30.0
    new.hands.palm_scroll.invert = True
    new.hands.roles.release_s = 2.5
    p.apply_config(new)
    assert p.palm.cfg.gain == 30.0 and p.palm.cfg.invert and p.cfg.hands.roles.release_s == 2.5


def test_settings_model_fields(tmp_path):
    m = SettingsModel(AppConfig(), tmp_path / "c.yaml")
    tun = {t.path: t for t in m.tunables()}
    for path in (
        "hands.roles.release_s",
        "hands.palm_scroll.engage_s",
        "hands.palm_scroll.depth_threshold",
        "hands.palm_scroll.margin",
        "hands.palm_scroll.gain",
        "hands.palm_scroll.dead_zone",
        "hands.palm_scroll.friction",
    ):
        assert path in tun and tun[path].label == TUNABLE_LABELS[path] and not tun[path].label.startswith("hands.")
        assert tun[path].lo <= tun[path].value <= tun[path].hi
    assert m.set_tunable("hands.palm_scroll.gain", 20.0) is None and m.cfg.hands.palm_scroll.gain == 20.0
    assert m.set_tunable("hands.palm_scroll.gain", 500.0) is not None
    for path, label in PALM_SCROLL_SWITCHES:
        assert (
            label
            and m.set_voice(path, False) is None
            and getattr(m.cfg.hands.palm_scroll, path.rsplit(".", 1)[1]) is False
        )
    assert m.role_choices() == ["Auto", "Right", "Left"]
    assert m.role_label() == "Auto"
    assert m.set_role_label("Left") is None and m.cfg.hands.roles.mode == "left" and m.role_label() == "Left"
    assert m.set_role_label("right") is None and m.cfg.hands.roles.mode == "right"
    assert m.set_role_label("both") is not None and m.cfg.hands.roles.mode == "right"
    m.cfg.hands.roles.mode = "any"
    assert m.role_label().startswith("Any")
    assert m.set_voice("hands.prefer", "left") is not None  # the old key is gone from Settings
    m.save()
    saved = yaml.safe_load((tmp_path / "c.yaml").read_text("utf-8"))
    assert "prefer" not in saved["hands"] and saved["hands"]["palm_scroll"]["gain"] == 20.0


def test_flat_trackpad_hand_raised_first_hands_the_pointer_to_the_pointing_hand():
    r = HandRoles("auto")
    res = r.update(0.0, [obs(hand(0.3, name="palm"), "Right")])  # the "trackpad" hand comes up first
    flat_id = res.pointer_id
    assert res.pointer is not None and not r.confirmed
    res = r.update(DT, [obs(hand(0.3, name="palm"), "Right"), obs(hand(0.7, name="point"), "Left")])
    assert res.pointer_id not in (None, flat_id) and res.helper_id == flat_id and r.confirmed
    assert not res.promoted  # a brand-new hand, not a former helper


def test_confirmed_only_hand_keeps_the_pointer():
    r = HandRoles("auto")
    res = r.update(0.0, [obs(hand(0.3, name="fist"), "Right")])  # did something other than a flat palm
    pid = res.pointer_id
    assert r.confirmed
    res = r.update(DT, [obs(hand(0.3, name="palm"), "Right"), obs(hand(0.7, name="point"), "Left")])
    assert res.pointer_id == pid
    r2 = HandRoles("auto")
    r2.update(0.0, [obs(hand(0.3, name="palm"), "Right")])
    r2.update(DT, [obs(hand(0.3, name="point"), "Right")])  # the lone hand points: confirmed
    res = r2.update(2 * DT, [obs(hand(0.3, name="palm"), "Right"), obs(hand(0.7, name="point"), "Left")])
    assert palm_center(res.pointer.landmarks)[0] < 0.4


def test_pipeline_pointer_switch_ends_the_old_hands_drag():
    p = Pipeline(AppConfig(), RecordingBackend())
    t = 0.0
    for _ in range(15):
        p.on_hand(t, pose("pinch", center=(0.3, 0.5)), "Left", pointer_id=1)
        t += DT
    assert p.gestures.dragging and p.mouse.button_down
    acc = p.on_hand(t, pose("point", center=(0.7, 0.5)), "Left", pointer_id=2)
    assert "drag_end" in acc and "hand_lost" in acc and not p.mouse.button_down
    assert "hand_lost" not in p.on_hand(t + DT, pose("point", center=(0.7, 0.5)), "Left", pointer_id=2)
