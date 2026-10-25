"""v0.1.35: render the Project Away app icon from code. Re-run after changing the drawing:

    python packaging/icon/make_icons.py            # writes the files below next to this script
    python packaging/icon/make_icons.py --preview  # also writes preview.png (Dock sizes on light and dark)

Outputs (committed, so a build needs neither Pillow nor iconutil):

* ``ProjectAway-1024.png`` - the master: macOS icon grid (a 1024 canvas, an 824 px rounded-square body inset 100 px,
  continuous "squircle" corners and the soft drop shadow macOS icons carry), a deep blue-to-violet gradient with a
  glassy top light, and the open-hand-with-waveform glyph from :mod:`away.app.icon_art` (the menu bar icon draws the
  same hand as a template image).
* ``ProjectAway.icns`` - every size macOS asks for (16 to 512 at 1x and 2x) via ``iconutil`` (macOS only; elsewhere
  the old .icns is kept). away.spec puts it in the .app bundle.
* ``menubar-template.png`` - the menu bar glyph's states side by side (for review; the app draws them at run time).

Pillow only; drawn 4x supersampled and scaled down with Lanczos, like icon_art."""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))  # away.app.icon_art without installing the package

from away.app.icon_art import SS, hand_mask, template_icon  # noqa: E402

CANVAS = 1024
BODY = 824  # Apple's macOS app icon grid: the tile is 824 px of the 1024 canvas, centred
INSET = (CANVAS - BODY) // 2
#: gradient stops, top left -> bottom right: a bright azure into a deep indigo-violet; saturated enough to stand
#: out in a light Dock, dark enough at the bottom that the white glyph reads in a dark one
TOP = (64, 156, 255)
MID = (84, 96, 240)
BOTTOM = (98, 52, 196)


def squircle(px: int, inset: float, body: float, n: float = 5.0) -> Image.Image:
    """The tile's mask: a superellipse |x|^n + |y|^n = 1 (n=5 matches macOS's continuous corners closely)."""
    m = Image.new("L", (px, px), 0)
    a = body / 2
    cx = cy = inset + a
    pts = []
    for i in range(720):
        t = 2 * math.pi * i / 720
        c, s = math.cos(t), math.sin(t)
        pts.append((cx + a * math.copysign(abs(c) ** (2 / n), c), cy + a * math.copysign(abs(s) ** (2 / n), s)))
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m


def gradient(px: int) -> Image.Image:
    """The diagonal three-stop gradient (top left to bottom right)."""
    small = 256
    g = Image.new("RGB", (small, small))
    pix = g.load()
    for y in range(small):
        for x in range(small):
            t = (x * 0.35 + y * 0.65) / (small - 1)
            a, b, u = (TOP, MID, t / 0.5) if t < 0.5 else (MID, BOTTOM, (t - 0.5) / 0.5)
            pix[x, y] = tuple(round(a[i] + (b[i] - a[i]) * u) for i in range(3))
    return g.resize((px, px), Image.Resampling.BICUBIC)


def app_icon(size: int = CANVAS) -> Image.Image:
    big = CANVAS * SS
    k = SS
    tile = squircle(big, INSET * k, BODY * k)

    # the drop shadow macOS icons sit on: soft and offset down a little
    shadow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    sh_mask = Image.new("L", (big, big), 0)
    sh_mask.paste(tile.point(lambda v: v * 46 // 100), (0, 12 * k))
    shadow.putalpha(sh_mask.filter(ImageFilter.GaussianBlur(14 * k)))

    body = gradient(big).convert("RGBA")
    # glassy light: a soft white glow in the top half and a fine bright rim along the top edge
    glow = Image.linear_gradient("L").rotate(180).resize((big, big)).point(lambda v: max(0, v - 120) * 52 // 135)
    body = Image.composite(Image.new("RGBA", (big, big), (255, 255, 255, 255)), body, glow)
    rim = ImageChops.subtract(tile, ImageChops.offset(tile, 0, 5 * k)).filter(ImageFilter.GaussianBlur(1.5 * k))
    body = Image.composite(
        Image.new("RGBA", (big, big), (255, 255, 255, 255)), body, rim.point(lambda v: v * 55 // 100)
    )
    edge = ImageChops.subtract(tile, ImageChops.offset(tile, 0, -4 * k)).filter(ImageFilter.GaussianBlur(2 * k))
    body = Image.composite(Image.new("RGBA", (big, big), (30, 10, 90, 255)), body, edge.point(lambda v: v * 40 // 100))

    out = Image.alpha_composite(Image.new("RGBA", (big, big), (0, 0, 0, 0)), shadow)
    tile_img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    tile_img.paste(body, (0, 0), tile)
    out = Image.alpha_composite(out, tile_img)

    # the glyph: white with a faint cool tint at the bottom, a soft shadow under it for depth
    gpx = 600 * k
    glyph = hand_mask(gpx, bars=5)
    # centre the hand on its own outline (the thumb makes the glyph box lopsided), a hair above the middle
    x0, y0, x1, y1 = glyph.getbbox()
    gx = (big - (x1 - x0)) // 2 - x0
    gy = (big - (y1 - y0)) // 2 - y0 - 4 * k
    gmask = Image.new("L", (big, big), 0)
    gmask.paste(glyph, (gx, gy))
    gshadow = Image.new("RGBA", (big, big), (28, 12, 92, 0))
    gshadow.putalpha(
        ImageChops.offset(gmask, 0, 14 * k).filter(ImageFilter.GaussianBlur(18 * k)).point(lambda v: v * 38 // 100)
    )
    out = Image.alpha_composite(out, gshadow)
    fill = Image.new("RGBA", (big, big))
    ImageDraw.Draw(fill).rectangle((0, 0, big, big), fill=(255, 255, 255, 255))
    tint = Image.linear_gradient("L").resize((big, big)).point(lambda v: v * 22 // 100)
    fill = Image.composite(Image.new("RGBA", (big, big), (214, 222, 255, 255)), fill, tint)
    fill.putalpha(gmask)
    out = Image.alpha_composite(out, fill)
    return out.resize((size, size), Image.Resampling.LANCZOS)


def write_icns(master: Image.Image, dest: Path) -> bool:
    """Every iconset size macOS wants, packed by iconutil. False (nothing written) off macOS."""
    if shutil.which("iconutil") is None:
        print("iconutil not found (macOS only): ProjectAway.icns not rebuilt")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "ProjectAway.iconset"
        iconset.mkdir()
        for pt in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = pt * scale
                name = f"icon_{pt}x{pt}{'@2x' if scale == 2 else ''}.png"
                # small sizes are drawn from the master, sharpened a touch so the fingers stay apart
                img = master.resize((px, px), Image.Resampling.LANCZOS)
                if px <= 64:
                    img = img.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=0))
                img.save(iconset / name)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dest)], check=True)
    return True


def menubar_sheet() -> Image.Image:
    """The menu bar glyph's states on a light and a dark bar, at 2x (44 px) and 1x (22 px)."""
    states = [
        ("active", False),
        ("paused", False),
        ("muted", False),
        ("stopped", False),
        ("recording", False),
        ("active", True),
    ]
    cell = 84
    sheet = Image.new("RGBA", (cell * len(states), 2 * 60), (0, 0, 0, 0))
    for row, (bg, fg) in enumerate((((236, 236, 240), (0, 0, 0)), ((40, 40, 46), (255, 255, 255)))):
        y = row * 60
        ImageDraw.Draw(sheet).rectangle((0, y, sheet.width, y + 59), fill=(*bg, 255))
        for i, (state, dot) in enumerate(states):
            for px, ox, oy in ((44, 6, 8), (22, 56, 19)):
                t = template_icon(state, px, agent_dot=dot)
                tinted = Image.new("RGBA", t.size, (*fg, 255))
                tinted.putalpha(t.getchannel("A"))
                sheet.alpha_composite(tinted, (i * cell + ox, y + oy))
    return sheet


def preview(master: Image.Image) -> Image.Image:
    """The icon at Dock and Finder sizes on a light and a dark background."""
    sizes = (512, 256, 128, 64, 32, 16)
    w = sum(sizes) + 40 * (len(sizes) + 1)
    h = 512 + 80
    sheet = Image.new("RGBA", (w, h * 2), (0, 0, 0, 255))
    for row, bg in enumerate(((242, 242, 247), (28, 28, 32))):
        ImageDraw.Draw(sheet).rectangle((0, row * h, w, row * h + h), fill=(*bg, 255))
        x = 40
        for s in sizes:
            img = master.resize((s, s), Image.Resampling.LANCZOS)
            sheet.alpha_composite(img, (x, row * h + 40 + (512 - s) // 2))
            x += s + 40
    return sheet


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    master = app_icon()
    master.save(HERE / "ProjectAway-1024.png")
    write_icns(master, HERE / "ProjectAway.icns")
    menubar_sheet().save(HERE / "menubar-template.png")
    if "--preview" in args:
        preview(master).save(HERE / "preview.png")
    print(f"wrote {HERE / 'ProjectAway-1024.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
