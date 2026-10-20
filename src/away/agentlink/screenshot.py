"""What part of the screen an agent sees, and at what size (v0.1.21).

On the user's PC (three monitors, a 5206x1080 virtual desktop) ``get_screen`` scaled the whole desktop to 960 px
wide (scale 0.184): text was unreadable. An agent now orients with a wide shot and then zooms:

* ``scale`` 0.1-1.0 (default 0.5) of the captured area's real pixels;
* ``monitor`` 1..N (as ``away monitors`` numbers them, left to right): one display instead of the whole desktop;
* ``region`` ``{x, y, width, height}`` in virtual-desktop coordinates (the coordinates actions use): zoom into an
  area;
* the picture's longest side is capped (:data:`MAX_SIDE`; 1600 for the live view) so the secure link stays fast.

The metadata lists the monitors and says exactly which area and scale were used, so image coordinates map back to
the screen: ``screen_x = origin_x + image_x / scale`` (what ``from_image=true`` on actions does).

Everything here is pure geometry (unit-tested with negative-offset and multi-monitor layouts); the capture itself is
:func:`away.agentlink.desktop.grab_screen`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_SCALE = 0.5
MIN_SCALE = 0.1
MAX_SCALE = 1.0
MAX_SIDE = 2560  # longest side of a screenshot sent to an agent
LIVE_MAX_SIDE = 1600  # the live screen view (sent every second or so)
LIVE_SCALE = 0.35


class ScreenRequestError(ValueError):
    """A bad ``scale`` / ``monitor`` / ``region`` (answered with HTTP 400 and a plain message)."""


@dataclass(frozen=True)
class Display:
    index: int  # 1..N, left to right (as `away monitors` and "monitor two" number them)
    x: int
    y: int
    w: int
    h: int
    primary: bool = False

    def public(self) -> dict[str, Any]:
        return {"index": self.index, "x": self.x, "y": self.y, "w": self.w, "h": self.h, "primary": self.primary}


@dataclass(frozen=True)
class Plan:
    crop: tuple[int, int, int, int]  # (left, top, right, bottom) in the captured image's pixels
    origin: tuple[int, int]  # the area's top-left corner in virtual-desktop (screen) coordinates
    area: tuple[int, int]  # the area's size in screen pixels
    out: tuple[int, int]  # the picture's size
    scale: float  # picture pixels per screen pixel (after the size cap)
    requested_scale: float
    capped: bool  # the size cap lowered the scale
    what: str  # "desktop" | "monitor N" | "region"

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        """Picture coordinates -> screen (virtual-desktop) coordinates."""
        return self.origin[0] + ix / self.scale, self.origin[1] + iy / self.scale

    def to_image(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.origin[0]) * self.scale, (sy - self.origin[1]) * self.scale


def displays(monitors: Sequence[Any]) -> list[Display]:
    """``away.control.monitors.Monitor`` objects (already sorted left to right) -> numbered displays."""
    out = []
    for i, m in enumerate(monitors, 1):
        out.append(Display(i, int(m.left), int(m.top), int(m.width), int(m.height), bool(getattr(m, "primary", False))))
    return out


def desktop_bounds(ds: Sequence[Display]) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the virtual desktop that holds every display (x/y can be negative)."""
    if not ds:
        raise ScreenRequestError("no monitors")
    x0 = min(d.x for d in ds)
    y0 = min(d.y for d in ds)
    x1 = max(d.x + d.w for d in ds)
    y1 = max(d.y + d.h for d in ds)
    return x0, y0, x1 - x0, y1 - y0


def parse_scale(value: Any, default: float = DEFAULT_SCALE) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ScreenRequestError("scale must be a number from 0.1 to 1.0")
    try:
        s = float(value)
    except (TypeError, ValueError):
        raise ScreenRequestError(f"scale must be a number from 0.1 to 1.0, got {value!r}") from None
    if not MIN_SCALE <= s <= MAX_SCALE:
        raise ScreenRequestError(f"scale must be from {MIN_SCALE:g} to {MAX_SCALE:g}, got {s:g}")
    return s


def parse_region(value: Any) -> tuple[int, int, int, int] | None:
    """``{x, y, width, height}`` (``w``/``h`` accepted), a list ``[x, y, w, h]`` or ``"x,y,w,h"``; None for none."""
    if value is None or value == "" or value == {}:
        return None
    try:
        if isinstance(value, str) and value.strip().startswith(("{", "[")):
            import json

            return parse_region(json.loads(value))
        if isinstance(value, str):
            parts = [float(p) for p in value.replace(" ", "").split(",")]
            x, y, w, h = parts
        elif isinstance(value, Mapping):
            x = float(value["x"])
            y = float(value["y"])
            w = float(value.get("width", value.get("w")))  # type: ignore[arg-type]
            h = float(value.get("height", value.get("h")))  # type: ignore[arg-type]
        else:
            x, y, w, h = (float(v) for v in value)
    except (KeyError, TypeError, ValueError):
        raise ScreenRequestError("region must be {x, y, width, height} in screen coordinates") from None
    if w < 1 or h < 1:
        raise ScreenRequestError("region width and height must be at least 1 pixel")
    return round(x), round(y), round(w), round(h)


def monitor_error(n: int, ds: Sequence[Display]) -> str:
    """ "there is no monitor 0; monitors are numbered 1..3 (1: 1920x1080 at 0,0; ...); leave monitor out for the
    whole desktop"."""
    listing = "; ".join(f"{d.index}: {d.w}x{d.h} at {d.x},{d.y}" for d in ds)
    return (
        f"there is no monitor {n}; monitors are numbered 1..{len(ds)} ({listing}); "
        "leave monitor out for the whole desktop"
    )


def plan(
    ds: Sequence[Display],
    image_size: tuple[int, int],
    *,
    scale: Any = None,
    monitor: Any = None,
    region: Any = None,
    max_side: int = MAX_SIDE,
    default_scale: float = DEFAULT_SCALE,
) -> Plan:
    """Which pixels of the captured desktop image to send and at what size.

    ``image_size`` is the captured image (normally the whole virtual desktop); when it differs from the monitors'
    bounding box (a capture that only got the primary monitor, DPI mismatch) coordinates are scaled to fit it."""
    s = parse_scale(scale, default_scale)
    vx, vy, vw, vh = desktop_bounds(ds)
    iw, ih = int(image_size[0]), int(image_size[1])
    if iw < 1 or ih < 1:
        raise ScreenRequestError("empty screenshot")
    kx, ky = iw / float(vw), ih / float(vh)  # image pixels per screen pixel (1.0 normally)
    if monitor not in (None, "") and region not in (None, "", {}):
        raise ScreenRequestError("give either monitor or region, not both")
    what = "desktop"
    ax, ay, aw, ah = vx, vy, vw, vh
    if monitor not in (None, ""):
        try:
            n = int(monitor)
        except (TypeError, ValueError):
            raise ScreenRequestError(f"monitor must be a number 1..{len(ds)}") from None
        d = next((d for d in ds if d.index == n), None)
        if d is None:
            # v0.1.22: monitor=0 used to return the whole desktop silently; monitors are numbered from 1
            raise ScreenRequestError(monitor_error(n, ds))
        ax, ay, aw, ah = d.x, d.y, d.w, d.h
        what = f"monitor {n}"
    reg = parse_region(region)
    if reg is not None:
        rx, ry, rw, rh = reg
        x0, y0 = max(rx, vx), max(ry, vy)
        x1, y1 = min(rx + rw, vx + vw), min(ry + rh, vy + vh)
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise ScreenRequestError(
                f"region is outside the screen (the desktop spans x {vx}..{vx + vw - 1}, y {vy}..{vy + vh - 1})"
            )
        ax, ay, aw, ah = x0, y0, x1 - x0, y1 - y0
        what = "region"
    crop = (
        round((ax - vx) * kx),
        round((ay - vy) * ky),
        round((ax - vx + aw) * kx),
        round((ay - vy + ah) * ky),
    )
    crop = (max(0, crop[0]), max(0, crop[1]), min(iw, crop[2]), min(ih, crop[3]))
    eff = s
    longest = max(aw, ah) * s
    capped = longest > max_side
    if capped:
        eff = max_side / float(max(aw, ah))
    out = (max(1, round(aw * eff)), max(1, round(ah * eff)))
    return Plan(crop, (ax, ay), (aw, ah), out, round(eff, 6), s, capped, what)


def jpeg_quality(out: tuple[int, int], zoomed: bool) -> int:
    """Readable text when zoomed in; smaller files for big overview shots."""
    px = out[0] * out[1]
    if zoomed:
        return 80 if px <= 1_000_000 else 72
    return 70 if px <= 1_500_000 else 60


def to_image(img: Any, p: Plan) -> Any:
    """Crop + resize a PIL image (or an RGB numpy array) per ``p`` as a PIL RGB image: exactly the picture the
    agent sees. ``render`` is this plus :func:`to_jpeg`."""
    from PIL import Image

    if not hasattr(img, "getbbox"):
        img = Image.fromarray(img)
    part = img.crop(p.crop)
    if part.size != p.out:
        part = part.resize(p.out, Image.Resampling.LANCZOS if p.scale < 1.0 else Image.Resampling.BICUBIC)
    if part.mode != "RGB":
        part = part.convert("RGB")
    return part


def to_jpeg(pil: Any, p: Plan, quality: int | None = None) -> bytes:
    """Encode an already cropped/resized PIL image as JPEG (used after set-of-mark annotation is drawn on it)."""
    import io

    buf = io.BytesIO()
    q = quality if quality is not None else jpeg_quality(p.out, p.what != "desktop")
    pil.save(buf, format="JPEG", quality=int(q), optimize=True)
    return buf.getvalue()


def render(img: Any, p: Plan, quality: int | None = None) -> bytes:
    """Crop + resize a PIL image (or an RGB numpy array) per ``p`` and encode it as JPEG."""
    import io

    from PIL import Image

    if not hasattr(img, "getbbox"):
        img = Image.fromarray(img)
    part = img.crop(p.crop)
    if part.size != p.out:
        part = part.resize(p.out, Image.Resampling.LANCZOS if p.scale < 1.0 else Image.Resampling.BICUBIC)
    if part.mode != "RGB":
        part = part.convert("RGB")
    buf = io.BytesIO()
    q = quality if quality is not None else jpeg_quality(p.out, p.what != "desktop")
    part.save(buf, format="JPEG", quality=int(q), optimize=True)
    return buf.getvalue()


def meta(p: Plan, ds: Sequence[Display], image_bytes: int) -> dict[str, Any]:
    """What the agent needs to orient and to click: the monitors, the area and the scale used."""
    vx, vy, vw, vh = desktop_bounds(ds)
    return {
        "showing": p.what,
        "scale": p.scale,
        "requested_scale": p.requested_scale,
        "scale_capped": p.capped,
        "origin": {"x": p.origin[0], "y": p.origin[1]},
        "area": {"x": p.origin[0], "y": p.origin[1], "width": p.area[0], "height": p.area[1]},
        "image": {"width": p.out[0], "height": p.out[1], "bytes": image_bytes},
        "desktop": {"x": vx, "y": vy, "width": vw, "height": vh},
        "monitors": [d.public() for d in ds],
        "to_screen": "screen_x = origin.x + image_x / scale; screen_y = origin.y + image_y / scale "
        "(or pass from_image=true to actions)",
    }
