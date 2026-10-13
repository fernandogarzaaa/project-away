"""The ``zones:`` and ``studio:`` config sections (dataclasses only; no imports from :mod:`away.config`).

``zones`` holds ONE zone map per camera and per app profile; the same map applies to every gesture.

.. code-block:: yaml

    zones:
      enabled: true          # false = zones are kept but never enforced
      block_cursor: false    # true = the cursor also stops in "Ignore here" areas (default: only activations)
      hysteresis: 0.03       # the hand must be this far past a boundary (image widths) before its side changes
      anchor: palm           # which point of the hand is tested: palm | index_tip | wrist
      maps:                  # camera name (or "*" = any camera) -> profile name -> zone map
        "*":
          default: {grid: {cols: 6, rows: 4, ...}, rects: [...], line: null}

    studio:
      gestures:              # per Studio gesture (trained in Settings > Gesture Studio)
        thumbs_up: {enabled: true, threshold: 0.8, hold_s: 0.15, cooldown_s: 1.0, then: ["wait:0.3", "key:enter"]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ZONE_ANCHORS = ("palm", "index_tip", "wrist")


@dataclass
class ZonesConfig:
    enabled: bool = True
    block_cursor: bool = False
    hysteresis: float = 0.03
    anchor: str = "palm"
    maps: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudioConfig:
    gestures: dict[str, Any] = field(default_factory=dict)
