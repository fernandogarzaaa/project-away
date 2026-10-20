"""Set-of-mark prompting for screens where the UI Automation tree is empty (proposal #25).

Canvas apps, games and remote desktops expose no named UI elements, so an agent must guess pixels from the
screenshot. With ``get_screen(annotate=True)`` the server instead detects plausible interactable regions from the
picture itself, draws numbered red boxes on it, and lists the boxes with screen coordinates in the metadata. The
agent then clicks "element 14" (``pointer(action="click", element="mark:14")``) instead of estimating pixels.

The detector is deliberately dependency-free (numpy/scipy/PIL, all already core deps): no ML model, no download,
no service, nothing paid. It is a fallback for when UIA is empty, not a replacement for ``elements=true`` on
native apps.

Honest limits, stated plainly: this finds *visual regions* (boxes, buttons, panels, bars), not semantic elements.
A numbered box can be a button, a scrollbar or a plain panel border, and small text links without visible borders
will be missed. The detector is tuned for recall of boxed regions on clean screenshots; noisy gradients and
anti-aliased text edges can split or merge boxes. ``confidence`` is edge-density based, not a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_MARKS = 100  # never flood the model with boxes; the biggest/densest regions win
WORK_MAX_SIDE = 768  # detection runs at this size, boxes are scaled back up (fast enough on one core)


@dataclass(frozen=True)
class Region:
    id: int  # 1..N, reading order
    box: tuple[int, int, int, int]  # x, y, width, height in the annotated image's pixels
    confidence: float  # 0..1, edge density of the region (not a probability)


def detect_regions(img: Any) -> list[Region]:
    """Plausible interactable regions in the picture (PIL image or RGB numpy array), numbered in reading order."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    if not hasattr(img, "getbbox"):
        img = Image.fromarray(np.asarray(img))
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w < 1 or h < 1:
        return []
    k = min(1.0, WORK_MAX_SIDE / max(w, h))
    work = rgb.resize((max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.BILINEAR) if k < 1.0 else rgb
    gray = np.asarray(work.convert("L"), dtype=np.float32)
    mag = np.hypot(ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0))
    top = float(mag.max())
    if top <= 0.0:
        return []  # a flat image has no regions to mark
    # edge pixels: clearly above the background gradient (percentile so it adapts to the picture)
    thr = max(top * 0.15, float(np.percentile(mag, 88)))
    edges = mag > thr
    closed = ndimage.binary_closing(edges, structure=np.ones((5, 5)))
    labeled, _ = ndimage.label(closed)
    objects = ndimage.find_objects(labeled)
    H, W = gray.shape
    area = float(H * W)
    found: list[tuple[int, int, int, int, float]] = []  # x, y, w, h, density
    for i, sl in enumerate(objects):
        if sl is None:
            continue
        ys, xs = sl
        bw, bh = xs.stop - xs.start, ys.stop - ys.start
        if bw < 12 or bh < 12:
            continue  # smaller than a plausible control at the work size
        box_area = bw * bh
        if box_area < 0.0004 * area or box_area > 0.30 * area:
            continue  # specks, or the whole background as one "region"
        if max(bw, bh) / max(1, min(bw, bh)) > 12:
            continue  # hairlines and window-wide rules
        density = float(np.count_nonzero(edges[sl] & (labeled[sl] == i + 1))) / box_area
        if not 0.015 <= density <= 0.85:
            continue  # too faint to be a region, or a solid text blob
        found.append((xs.start, ys.start, bw, bh, density))
    kept = _nms(found)
    kept.sort(key=_reading_key)
    out: list[Region] = []
    for n, (x, y, bw, bh, density) in enumerate(kept[:MAX_MARKS], 1):
        sx, sy = x / k, y / k
        out.append(Region(n, (round(sx), round(sy), round(bw / k), round(bh / k)), round(density, 3)))
    return out


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    if ix == 0 or iy == 0:
        return 0.0
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def _nms(found: list[tuple[int, int, int, int, float]]) -> list[tuple[int, int, int, int, float]]:
    """Keep the densest box of every overlapping cluster (IoU > 0.5)."""
    ordered = sorted(found, key=lambda f: -f[4])
    kept: list[tuple[int, int, int, int, float]] = []
    for f in ordered:
        if all(_iou(f[:4], g[:4]) <= 0.5 for g in kept):
            kept.append(f)
    return kept


def _reading_key(f: tuple[int, int, int, int, float]) -> tuple[int, int]:
    x, y, _bw, bh = f[:4]
    band = max(1, int(bh * 0.6))  # boxes on roughly the same row share a band
    return (y // band, x)


def draw_marks(img: Any, regions: list[Region]) -> Any:
    """A copy of the PIL image with numbered red boxes and label chips drawn over it."""
    from PIL import ImageDraw, ImageFont

    if not hasattr(img, "getbbox"):
        raise TypeError("draw_marks needs a PIL image")
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    lw = max(2, round(min(w, h) / 300))
    try:
        font = ImageFont.load_default(size=max(12, round(min(w, h) / 60)))
    except TypeError:  # very old Pillow: load_default takes no size
        font = ImageFont.load_default()
    red, chip, white = (255, 45, 45), (200, 0, 0), (255, 255, 255)
    for r in regions:
        x, y, bw, bh = r.box
        d.rectangle([x, y, x + bw, y + bh], outline=red, width=lw)
        label = str(r.id)
        box = d.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        pad = max(2, lw)
        cx, cy = x, max(0, y - th - 2 * pad)  # the chip sits above the box when there is room
        d.rectangle([cx, cy, cx + tw + 2 * pad, cy + th + 2 * pad], fill=chip)
        d.text((cx + pad, cy + pad - box[1]), label, font=font, fill=white)
    return out
