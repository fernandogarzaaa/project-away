"""Smart pointer settings (v0.1.20). Pure dataclasses + validation; imported by :mod:`away.config`, so this module
must not import it back.

YAML::

    hands:
      cursor:
        smoothing: one_euro        # one_euro | predictive  (Settings: "Smoother cursor")
        magnet:                    # Settings: "Snap to buttons" (Windows; a no-op elsewhere)
          enabled: true
          radius_px: 40
          strength: 0.35
          sticky: 0.5
        predictive:                # only used when smoothing: predictive
          lookahead_ms: 33
          process_noise: 5.0
          measurement_noise: 0.004
          speed_gate: 0.3
    voice:
      click_by_name: true          # Settings: "Click things by saying their name"

The numbers are Tuning-tab sliders (section "Advanced"); the three switches are plain toggles in Settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SMOOTHING_MODES = ("one_euro", "predictive")


@dataclass
class MagnetConfig:
    """Magnetic targets: the cursor is gently pulled toward clickable things nearby and slowed down over them."""

    enabled: bool = True  # Windows only (UI Automation); ignored elsewhere
    radius_px: float = 40.0  # how far from a target the pull starts
    strength: float = 0.35  # 0..1: how much of the way to the target's centre the cursor is pulled (at the edge)
    sticky: float = 0.5  # 0..0.95: how much slower small moves are over a target (cancels webcam jitter)


@dataclass
class PredictiveConfig:
    """``hands.cursor.smoothing: predictive``: speed-adaptive Kalman filter with lookahead."""

    lookahead_ms: float = 33.0  # how far ahead to predict (about one camera frame at 30 fps)
    process_noise: float = 5.0  # higher: follows fast direction changes sooner, a bit less smooth
    measurement_noise: float = 0.004  # expected hand-landmark jitter (hand units); higher: smoother, more lag
    speed_gate: float = 0.3  # hand speed (frame widths / s) at which the full prediction applies


def pointer_tunables(cfg: Any) -> list[tuple[str, Any, float, float]]:
    """(path, value, min, max) for :func:`away.config.tunable_ranges` (the Tuning tab's "Advanced" section)."""
    m, p = cfg.hands.cursor.magnet, cfg.hands.cursor.predictive
    return [
        ("hands.cursor.magnet.radius_px", m.radius_px, 5.0, 200.0),
        ("hands.cursor.magnet.strength", m.strength, 0.0, 1.0),
        ("hands.cursor.magnet.sticky", m.sticky, 0.0, 0.95),
        ("hands.cursor.predictive.lookahead_ms", p.lookahead_ms, 0.0, 100.0),
        ("hands.cursor.predictive.process_noise", p.process_noise, 0.1, 50.0),
        ("hands.cursor.predictive.measurement_noise", p.measurement_noise, 0.0005, 0.02),
        ("hands.cursor.predictive.speed_gate", p.speed_gate, 0.01, 2.0),
    ]


def validate_pointer(cfg: Any) -> None:
    """The enum and the switches (the numbers are range-checked with every other tunable)."""
    c = cfg.hands.cursor
    if c.smoothing not in SMOOTHING_MODES:
        raise ValueError(f"hands.cursor.smoothing must be {'|'.join(SMOOTHING_MODES)}, got {c.smoothing!r}")
    if not isinstance(c.magnet.enabled, bool):
        raise ValueError(f"hands.cursor.magnet.enabled must be true or false, got {c.magnet.enabled!r}")
    if not isinstance(cfg.voice.click_by_name, bool):
        raise ValueError(f"voice.click_by_name must be true or false, got {cfg.voice.click_by_name!r}")


# Tuning tab descriptions: (title, help, unit); merged into away.app.tunables.TUNABLE_META under "Advanced"
POINTER_TUNABLE_TEXT: dict[str, tuple[str, str, str]] = {
    "hands.cursor.magnet.radius_px": (
        "Snap to buttons: reach",
        "Higher: buttons start pulling the pointer from further away.",
        "px",
    ),
    "hands.cursor.magnet.strength": (
        "Snap to buttons: pull",
        "Higher: the pointer is drawn more firmly toward a nearby button (0 = no pull).",
        "",
    ),
    "hands.cursor.magnet.sticky": (
        "Snap to buttons: hold still over a button",
        "Higher: small shakes move the pointer less while it is over a button (0 = off).",
        "",
    ),
    "hands.cursor.predictive.lookahead_ms": (
        "Smoother cursor: look ahead",
        "Higher: the pointer runs further ahead of your hand (less lag, may overshoot a little).",
        "ms",
    ),
    "hands.cursor.predictive.process_noise": (
        "Smoother cursor: follow quick turns",
        "Higher: follows sudden changes of direction sooner; lower: smoother.",
        "",
    ),
    "hands.cursor.predictive.measurement_noise": (
        "Smoother cursor: shake filter",
        "Higher: more smoothing of a shaky camera image, with a little more lag.",
        "",
    ),
    "hands.cursor.predictive.speed_gate": (
        "Smoother cursor: prediction speed",
        "Higher: the look-ahead only kicks in on faster moves.",
        "frame/s",
    ),
}
