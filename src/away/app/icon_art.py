"""v0.1.35: the Project Away glyph (an open hand with a voice waveform in its palm), drawn from code.

One drawing serves two places so they share a design language:

* :func:`template_icon` - the macOS menu bar icon. A *template* image (black + alpha only): macOS tints it for a
  light or dark menu bar and for the highlighted (menu open) state, as every native status item does. The state is
  shown by shape, never colour: paused is an outlined hand, a muted mic is a slash, "not listening" is a dimmed
  hand, recording adds a dot.
* ``packaging/icon/make_icons.py`` - the app icon (the glyph on a rounded-square tile), rendered to the .icns.

Everything is drawn supersampled (4x) and scaled down with a Lanczos filter, so edges stay smooth from 16 px up.
Coordinates are in a 1000-unit square that holds the glyph."""

from __future__ import annotations

from typing import Any

#: supersampling factor for every drawing here
SS = 4

#: fingers as (centre x, top y, bottom y, width) in glyph units: index, middle, ring, little
FINGERS = (
    (322.0, 168.0, 560.0, 112.0),
    (452.0, 104.0, 560.0, 112.0),
    (582.0, 140.0, 560.0, 112.0),
    (704.0, 252.0, 600.0, 104.0),
)
#: the palm: a rounded rectangle with a big radius at the heel and a squarer top under the fingers
PALM = (266.0, 440.0, 756.0, 880.0)
PALM_RADIUS = 190.0
PALM_TOP = (266.0, 440.0, 756.0, 700.0)
PALM_TOP_RADIUS = 56.0
#: the thumb: a capsule from its base (in the palm) to its tip, and its width
THUMB = ((380.0, 716.0), (162.0, 452.0), 116.0)
#: fillet radii where the thumb meets the index finger (the web) and the heel of the palm
WEB_RADIUS = 34.0
HEEL_RADIUS = 56.0


def _waveform(bars: int) -> list[tuple[float, float, float, float]]:
    """The voice waveform cut out of the palm: ``bars`` rounded bars, tallest in the middle (glyph units)."""
    cx, cy = 511.0, 662.0
    if bars <= 3:
        heights: tuple[float, ...] = (96.0, 176.0, 96.0)
        width, step = 50.0, 84.0
    else:
        heights = (64.0, 124.0, 184.0, 124.0, 64.0)
        width, step = 34.0, 62.0
    n = len(heights)
    out = []
    for i, h in enumerate(heights):
        x = cx + (i - (n - 1) / 2) * step
        out.append((x - width / 2, cy - h / 2, x + width / 2, cy + h / 2))
    return out


def _fillet(
    d: Any, k: float, corner: tuple[float, float], a: tuple[float, float], b: tuple[float, float], radius: float
) -> None:
    """Round the concave corner at ``corner`` between two outside edges leaving it along the unit vectors ``a`` and
    ``b``: fill the wedge up to the tangent points and carve the circle tangent to both edges back out."""
    import math

    cos = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    half = math.acos(cos) / 2
    if half <= 0.01:
        return
    t = radius / math.tan(half)
    ta = (corner[0] + a[0] * t, corner[1] + a[1] * t)
    tb = (corner[0] + b[0] * t, corner[1] + b[1] * t)
    bis = (a[0] + b[0], a[1] + b[1])
    n = math.hypot(*bis)
    dist = radius / math.sin(half)
    c = (corner[0] + bis[0] / n * dist, corner[1] + bis[1] / n * dist)
    d.polygon([(corner[0] * k, corner[1] * k), (ta[0] * k, ta[1] * k), (tb[0] * k, tb[1] * k)], fill=255)
    d.ellipse(((c[0] - radius) * k, (c[1] - radius) * k, (c[0] + radius) * k, (c[1] + radius) * k), fill=0)


def _draw_hand(d: Any, k: float, inset: float = 0.0) -> None:
    """Every part of the hand, each shrunk by ``inset`` glyph units (the inside of the paused outline)."""
    import math

    i = inset
    for cx, top, bottom, w in FINGERS:
        if w / 2 - i < 1:
            continue
        d.rounded_rectangle(
            ((cx - w / 2 + i) * k, (top + i) * k, (cx + w / 2 - i) * k, bottom * k), radius=(w / 2 - i) * k, fill=255
        )
    x0, y0, x1, y1 = PALM
    d.rounded_rectangle(
        ((x0 + i) * k, (y0 + i) * k, (x1 - i) * k, (y1 - i) * k), radius=(PALM_RADIUS - i) * k, fill=255
    )
    x0, y0, x1, y1 = PALM_TOP
    d.rounded_rectangle(
        ((x0 + i) * k, (y0 + i) * k, (x1 - i) * k, y1 * k), radius=max(1.0, PALM_TOP_RADIUS - i) * k, fill=255
    )
    (bx, by), (tx, ty), tw = THUMB
    r = tw / 2 - i
    _capsule(d, (bx * k, by * k), (tx * k, ty * k), 2 * r * k, 255)
    # where the thumb's two edges cross the hand's left side (x = left): the web above, the heel below
    length = math.hypot(tx - bx, ty - by)
    u = ((tx - bx) / length, (ty - by) / length)  # towards the tip
    left = PALM[0] + i
    for sign, out, radius in ((1, (0.0, -1.0), WEB_RADIUS), (-1, (0.0, 1.0), HEEL_RADIUS)):
        nx, ny = u[1] * -sign, u[0] * sign  # the edge's outward normal: up-right for the top edge
        px, py = bx + nx * r, by + ny * r
        t = (left - px) / u[0]
        _fillet(d, k, (left, py + u[1] * t), out, u, radius)


def _capsule(d: Any, p0: tuple[float, float], p1: tuple[float, float], width: float, fill: int) -> None:
    d.line((p0, p1), fill=fill, width=round(width))
    r = width / 2
    for x, y in (p0, p1):
        d.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def hand_mask(px: int, bars: int = 5, outline: float = 0.0) -> Any:
    """The glyph as an ``L`` mask, ``px`` pixels square (255 = hand). ``bars``: waveform bars cut out of the palm
    (0 for none). ``outline`` > 0 draws only the hand's outline, that many glyph units thick (no waveform)."""
    from PIL import Image, ImageChops, ImageDraw

    big = px * SS
    k = big / 1000.0
    img = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(img)

    def s(*v: float) -> tuple[float, ...]:
        return tuple(x * k for x in v)

    _draw_hand(d, k)
    if outline > 0:
        # the same hand shrunk by the stroke width, carved out: an even stroke that follows every finger
        inner = Image.new("L", (big, big), 0)
        _draw_hand(ImageDraw.Draw(inner), k, inset=outline)
        img = ImageChops.subtract(img, inner)
    elif bars:
        for x0, y0, x1, y1 in _waveform(bars):
            d.rounded_rectangle(s(x0, y0, x1, y1), radius=(x1 - x0) * k / 2, fill=0)
    return img.resize((px, px), Image.Resampling.LANCZOS)


def template_icon(state: str = "active", size: int = 64, agent_dot: bool = False) -> Any:
    """v0.1.35: the macOS menu bar icon, a template image (black + alpha) ``size`` pixels square.

    active: the solid hand with its waveform; paused: the hand's outline; muted: the hand crossed by a slash;
    stopped ("stop listening" was said): the hand at half strength; recording: a solid dot by the hand. The agent
    dot (an AI agent may be watching) is a dot at the top right, cut free of the hand."""
    from PIL import Image, ImageChops, ImageDraw

    # the hand is ~2/3 of the bar's height, as SF Symbols are in a 22 pt menu bar; centred on its own outline (the
    # thumb makes the glyph's box lopsided)
    g = round(size * 0.86)
    glyph = hand_mask(g, outline=40.0) if state == "paused" else hand_mask(g, bars=3)
    box = glyph.getbbox() or (0, 0, g, g)
    off = ((size - (box[2] - box[0])) // 2 - box[0], (size - (box[3] - box[1])) // 2 - box[1])
    alpha = Image.new("L", (size, size), 0)
    alpha.paste(glyph, off)

    big = size * SS
    k = big / 1000.0

    def mark(draw_fn: Any) -> Any:
        m = Image.new("L", (big, big), 0)
        draw_fn(ImageDraw.Draw(m))
        return m.resize((size, size), Image.Resampling.LANCZOS)

    if state == "muted":
        # a slash from top left to bottom right, with a clear gap either side so it reads at 16 px
        def slash(d: Any, width: float, fill: int) -> None:
            _capsule(d, (200 * k, 150 * k), (820 * k, 870 * k), width * k, fill)

        alpha = ImageChops.subtract(alpha, mark(lambda d: slash(d, 190, 255)))
        alpha = ImageChops.lighter(alpha, mark(lambda d: slash(d, 84, 255)))
    elif state == "stopped":
        alpha = alpha.point(lambda v: v * 45 // 100)
    elif state == "recording":
        r = 125.0

        def dot(d: Any, rr: float) -> None:
            d.ellipse(((850 - rr) * k, (840 - rr) * k, (850 + rr) * k, (840 + rr) * k), fill=255)

        alpha = ImageChops.subtract(alpha, mark(lambda d: dot(d, r + 60)))
        alpha = ImageChops.lighter(alpha, mark(lambda d: dot(d, r)))
    if agent_dot:
        r = 115.0

        def adot(d: Any, rr: float) -> None:
            d.ellipse(((860 - rr) * k, (150 - rr) * k, (860 + rr) * k, (150 + rr) * k), fill=255)

        alpha = ImageChops.subtract(alpha, mark(lambda d: adot(d, r + 60)))
        alpha = ImageChops.lighter(alpha, mark(lambda d: adot(d, r)))
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.putalpha(alpha)
    return out
