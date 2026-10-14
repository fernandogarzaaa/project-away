"""v0.1.25 live tracking calibration and a steadier cursor, tested with fake landmark streams.

The jitter sources this fixes (see :mod:`away.hands.steady`): the dead zone that piled small moves up and released
them as 15-20 px hops, a One Euro ``beta`` too small to ever speed the filter up, and the 120 Hz glide extrapolating
jitter past a stop. Calibration measures the user's own noise / reach / pinch per camera.
"""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path
from typing import Any

import pytest

from away import config as C
from away.hands import livecal
from away.hands.steady import AdaptiveFilter2D, HoldStill, deadband, jitter_px
from away.pointer.glide import Glide

FPS = 30.0
SCREEN = (1920, 1080)


def still_hand(
    seconds: float, sigma: float, x: float = 0.5, y: float = 0.5, t0: float = 0.0, seed: int = 1, fps: float = FPS
) -> list[tuple[float, float, float]]:
    rnd = random.Random(seed)
    return [(t0 + i / fps, x + rnd.gauss(0, sigma), y + rnd.gauss(0, sigma)) for i in range(int(seconds * fps))]


def hand_points(x: float, y: float) -> list[list[float]]:
    """21 fake landmarks with the index fingertip at (x, y)."""
    pts = [[x, y + 0.15] for _ in range(21)]
    pts[8] = [x, y]
    return pts


def steps(points: list[tuple[int, int]]) -> list[float]:
    return [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(points)]


# ------------------------------------------------------------------ jitter sources (these fail on v0.1.24)
def test_a_slow_move_is_smooth_not_a_staircase_of_hops() -> None:
    """A careful slow move (0.03 frame widths/s, no noise) with the default cursor: every frame moves a little.
    v0.1.24's dead zone dropped each 0.001 step until 0.004 had piled up, then jumped ~15-20 px at once."""
    c = C.HandsConfig().cursor
    c.hold_still = False  # this is about the dead zone itself
    samples = [(i / FPS, 0.4 + 0.001 * i, 0.5) for i in range(90)]
    pos = livecal.simulate(c, samples, SCREEN)
    st = steps(pos[30:])  # after the filter has settled
    assert max(st) <= 4.0  # no hops
    assert sum(1 for s in st if s == 0) <= len(st) // 4  # it keeps moving instead of stop-go
    assert pos[-1][0] - pos[30][0] > 20  # and it really follows the move


def test_the_filter_speeds_up_for_fast_moves() -> None:
    """At a fast move the default filter follows closely (v0.1.24's beta never raised the cutoff)."""
    from away.pointer.smoothing import make_cursor_filter

    f = make_cursor_filter(C.HandsConfig().cursor)
    f(0.0, 0.2, 0.5)
    x, out = 0.2, (0.2, 0.5)
    for i in range(1, 16):  # 1.2 frame widths / s for half a second
        x = 0.2 + 0.04 * i
        out = f(i / FPS, x, 0.5)
    assert x - out[0] < 0.05  # under 5% of the frame behind (a fixed 1 Hz low-pass lags ~0.19)


def test_the_glide_does_not_overshoot_a_stop() -> None:
    """The hand moves fast, then stops dead: the 120 Hz output must not run past the stop and come back."""
    g = Glide()
    xs: list[float] = []
    t = 0.0
    for i in range(10):  # 300 px/s ...
        g.on_target(i / FPS, 100.0 + 10.0 * i, 100.0)
    for i in range(10, 20):  # ... then held still at 190
        g.on_target(i / FPS, 190.0, 100.0)
        while t < (i + 1) / FPS:
            t += 1 / 120
            p = g.step(t)
            if p is not None:
                xs.append(p[0])
    assert xs and max(xs) <= 190.5


def test_deadband_is_slack_and_frame_rate_free() -> None:
    p = (0.5, 0.5)
    assert deadband(p, (0.502, 0.5), 0.003) == p  # inside the slack: no move
    assert deadband(p, (0.51, 0.5), 0.003) == pytest.approx((0.507, 0.5))  # dragged along at the slack distance
    a = b = p  # the same path at 30 and 15 fps ends at the same place (it is positional, not per frame)
    for i in range(1, 31):
        a = deadband(a, (0.5 + 0.001 * i, 0.5), 0.003)
    for i in range(1, 16):
        b = deadband(b, (0.5 + 0.002 * i, 0.5), 0.003)
    assert a == pytest.approx(b)


def test_hold_still_locks_and_unlocks_without_a_jump() -> None:
    h = HoldStill(radius=0.01, hold_s=0.2)
    out = [h(t, x, y) for t, x, y in still_hand(1.0, 0.001)]
    assert h.locked and len(set(out[-10:])) == 1  # locked: not a single twitch
    last = out[-1]
    moved = h(1.0, 0.53, 0.5)  # a real move: unlocks
    assert not h.locked and moved[0] - last[0] < 0.03  # continues from the locked point, no catch-up jump


def test_adaptive_filter_is_steady_at_any_frame_rate() -> None:
    """The same still noise filtered at 15 and 30 fps: both far below the raw noise (the factor uses the real
    frame time, and noise does not fool the speed estimate into opening the filter)."""
    raw = 0.003 * 640 * math.sqrt(2)
    for fps in (15.0, 30.0):
        f = AdaptiveFilter2D()
        out = [f(t, x, y) for t, x, y in still_hand(4.0, 0.003, fps=fps)]
        assert jitter_px([(x * 640, y * 640) for x, y in out[int(fps) :]]) < raw / 2, fps


# ------------------------------------------------------------------ calibration maths
def test_noise_measurement_ignores_slow_drift() -> None:
    pts = [(x, y) for _t, x, y in still_hand(3.0, 0.002)]
    n = livecal.measure_noise(pts)
    assert 0.0024 < n < 0.0033  # sqrt(2) * 0.002
    drift = [(x + 0.0005 * i, y) for i, (x, y) in enumerate(pts)]  # the arm sags 0.045 over 3 s
    assert livecal.measure_noise(drift) == pytest.approx(n, rel=0.05)


def test_derived_settings_follow_the_measurements() -> None:
    q = livecal.derive(livecal.Measurements(noise=0.001, fps=30.0))["cursor"]
    n = livecal.derive(livecal.Measurements(noise=0.004, fps=15.0))["cursor"]
    assert n["min_cutoff"] < q["min_cutoff"]  # noisier: steadier smoothing
    assert n["dead_zone"] > q["dead_zone"] and n["hold_radius"] > q["hold_radius"]
    assert n["fast_cutoff"] < q["fast_cutoff"]  # slower camera: no point following faster than frames arrive
    small = livecal.derive(livecal.Measurements(noise=0.002, reach=(0.35, 0.3, 0.65, 0.6)))["cursor"]
    big = livecal.derive(livecal.Measurements(noise=0.002, reach=(0.1, 0.1, 0.9, 0.8)))["cursor"]
    assert small["gain"] > big["gain"]  # a small reach still covers the screen
    assert small["active_region"] == [0.32, 0.27, 0.68, 0.63]
    g = livecal.derive(livecal.Measurements(pinch_closed=0.1, pinch_open=0.9))["gestures"]
    assert g == {"pinch_on": pytest.approx(0.42), "pinch_off": pytest.approx(0.66)}
    for m in (  # every derived set is a valid config, even at the extremes
        livecal.Measurements(noise=1e-5, fps=60, reach=(0.0, 0.0, 1.0, 1.0), pinch_closed=0.0, pinch_open=2.0),
        livecal.Measurements(noise=0.05, fps=5, reach=(0.49, 0.49, 0.51, 0.51), pinch_closed=0.5, pinch_open=0.6),
    ):
        C.validate(livecal.apply_entry(C.AppConfig(), livecal.derive(m)))


def run_session(noise: float = 0.003, seed: int = 3) -> livecal.Session:
    rnd = random.Random(seed)
    s = livecal.Session()
    t = 0.0
    for _ in range(int(3.5 * FPS)):  # 1. hold still
        s.feed(t, hand_points(0.5 + rnd.gauss(0, noise), 0.5 + rnd.gauss(0, noise)), 1.0)
        t += 1 / FPS
    assert s.step == "corners"
    corners = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.65), (0.3, 0.65)]
    for i in range(int(8 * FPS)):  # 2. sweep the four corners
        a, b = corners[(i // 20) % 4], corners[(i // 20 + 1) % 4]
        k = (i % 20) / 20
        s.feed(t, hand_points(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k), 1.0)
        t += 1 / FPS
        if s.step != "corners":
            break
    assert s.step == "pinch"
    for _ in range(3):  # 3. pinch three times: closes to 0.12, opens to 0.8
        for r in [0.8, 0.6, 0.35, 0.2, 0.12, 0.12, 0.2, 0.4, 0.6, 0.8, 0.8]:
            s.feed(t, hand_points(0.5, 0.5), r)
            t += 1 / FPS
    return s


def test_a_full_session_with_a_fake_landmark_stream() -> None:
    s = run_session()
    assert s.step == "done" and s.progress() == 1.0
    m = s.measurements()
    assert 0.0035 < m.noise < 0.005  # sqrt(2) * 0.003
    assert m.reach is not None and m.reach[0] < 0.33 and m.reach[2] > 0.67
    assert m.pinch_closed == pytest.approx(0.12) and m.pinch_open == pytest.approx(0.8)
    assert 25 < m.fps < 35
    e = livecal.entry_for(m, now=1.0)
    assert e["noise_px"] == pytest.approx(m.noise * 640, abs=0.01) and "cursor" in e and "gestures" in e


def test_a_shaky_start_restarts_the_still_step_and_a_missing_hand_is_said_plainly() -> None:
    s = livecal.Session()
    for i in range(20):
        s.feed(i / FPS, hand_points(0.3 + 0.08 * (i % 2), 0.5), 1.0)  # waving, not still
    assert s.step == "still" and "movement" in s.message
    s.feed(1.0, None)
    assert "can't see your hand" in s.message


def test_after_calibration_a_noisy_still_hand_is_steady_and_fast_moves_still_follow() -> None:
    """The acceptance check: a 640x480 webcam with ~2.5 px of landmark noise. After calibrating, the cursor of a
    still hand shakes under a pixel, and a quick move still arrives within a few frames of the hand stopping."""
    s = run_session(noise=0.004, seed=5)
    entry = livecal.entry_for(s.measurements(), now=1.0)
    after = livecal.apply_entry(C.AppConfig(), entry).hands.cursor
    still = still_hand(4.0, 0.004, seed=9)
    pos = livecal.simulate(after, still, SCREEN)
    assert jitter_px([(float(x), float(y)) for x, y in pos[30:]]) < 1.0
    t0 = still[-1][0]
    move = [(t0 + i / FPS, 0.5 + min(1.0, i / 6) * 0.25, 0.5) for i in range(1, 30)]  # 0.25 in 0.2 s, then hold
    pos2 = livecal.simulate(after, still + move, SCREEN)
    start = pos2[len(still) - 1][0]
    travelled = pos2[-1][0] - start
    assert travelled > 300  # it really follows the move across the screen
    assert pos2[len(still) + 6 + 5][0] >= start + 0.9 * travelled  # 5 frames (~170 ms) after the hand stopped


def test_before_after_trails_show_the_improvement() -> None:
    s = run_session(noise=0.004, seed=7)
    entry = livecal.entry_for(s.measurements(), now=1.0)
    cfg = C.AppConfig()
    cfg.hands.cursor.smoothing = "one_euro"  # a v0.1.24-like "before"
    cfg.hands.cursor.hold_still = False
    cfg.hands.cursor.dead_zone = 0.004
    before, after, jb, ja = livecal.before_after(cfg, entry, s.trail, SCREEN)
    assert len(before) == len(after) == len(s.trail)
    assert ja < jb


# ------------------------------------------------------------------ saved per camera
def test_saved_per_camera_applied_only_to_that_camera_and_reset(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    cfg = C.AppConfig()
    cfg.hands.camera_index = "JoyCam"
    C.save(cfg, p)
    entry = {
        "cursor": {"min_cutoff": 0.5, "gain": 3.1, "active_region": [0.2, 0.2, 0.8, 0.8]},
        "gestures": {"pinch_on": 0.25, "pinch_off": 0.5},
        "noise_px": 2.1,
    }
    livecal.save(p, "JoyCam", entry)
    loaded = C.load(p)[0]
    assert loaded.hands.calibration["JoyCam"]["cursor"]["gain"] == 3.1
    on = livecal.calibrated(loaded)
    assert on.hands.cursor.gain == 3.1 and on.hands.gestures.pinch_on == 0.25
    assert on.hands.cursor.active_region == (0.2, 0.2, 0.8, 0.8)
    assert loaded.hands.cursor.gain == C.AppConfig().hands.cursor.gain  # the file's own settings are untouched
    loaded.hands.camera_index = "HD Webcam"  # another camera has no calibration: defaults
    assert livecal.calibrated(loaded) is loaded
    livecal.save(p, "JoyCam", None)  # Reset to defaults
    assert C.load(p)[0].hands.calibration == {}


def test_the_running_app_uses_the_cameras_calibration() -> None:
    from away.app.runner import Pipeline
    from away.control.mouse import RecordingBackend

    cfg = C.AppConfig()
    cfg.hands.calibration = {"0": {"cursor": {"gain": 3.3}, "gestures": {"pinch_on": 0.22, "pinch_off": 0.4}}}
    p = Pipeline(cfg, RecordingBackend())
    assert p.cfg.hands.cursor.gain == 3.3 and p.cfg.hands.gestures.pinch_on == 0.22
    new = C.AppConfig()
    new.hands.calibration = {"0": {"cursor": {"gain": 2.9}}}
    p.apply_config(new)  # Save in the window rewrites the file; the app reloads it
    assert p.cfg.hands.cursor.gain == 2.9


def test_a_broken_hand_edited_entry_is_ignored_not_fatal() -> None:
    cfg = C.AppConfig()
    cfg.hands.calibration = {"0": {"gestures": {"pinch_on": 0.9, "pinch_off": 0.1}}}  # on > off: invalid
    assert livecal.calibrated(cfg) is cfg
    with pytest.raises(ValueError):
        C.from_dict({"hands": {"calibration": {"0": "nope"}}})


# ------------------------------------------------------------------ where it is opened
def test_tray_settings_and_command_line_open_it(tmp_path: Path) -> None:
    from away import cli
    from away.comfort import actions

    seen: list[list[str]] = []
    assert actions.launch_calibrate(lambda argv: seen.append(list(argv))) is None
    assert seen[-1][-1] == "calibrate"

    def boom(argv: Any) -> None:
        raise OSError("no")

    assert "Calibrate tracking" in str(actions.launch_calibrate(boom))
    p = tmp_path / "config.yaml"
    cfg = C.AppConfig()
    cfg.hands.calibration = {"0": {"cursor": {"gain": 3.0}}}
    C.save(cfg, p)
    assert cli.main(["calibrate", "--config", str(p), "--reset"]) == 0
    assert C.load(p)[0].hands.calibration == {}


def test_window_texts_are_plain() -> None:
    from away.app import calibrate_gui as G

    assert G.status_line(1.24, 29.6, 45.2) == "Shake 1.2 px  |  30 fps  |  delay 45 ms"
    assert "steadier" in G.result_line(3.0, 0.6) and "80%" in G.result_line(3.0, 0.6)
    assert G.trail_points([(960, 540)], (1920, 1080)) == [160.0, 90.0]
    for text in livecal.INSTRUCTIONS.values():
        assert len(text) < 120 and "landmark" not in text.lower()
