"""Build the Windows MSIX packages from the PyInstaller output (release.yml, windows-latest).

    python scripts/build_msix.py                       # dist/ProjectAway -> dist/*.msixupload (+ sideload .msix)
    python scripts/build_msix.py --version 0.1.23 --input dist/ProjectAway --out dist
    python scripts/build_msix.py --render-manifest out.xml   # just the manifest (any OS)
    python scripts/build_msix.py --assets-only DIR            # just the tile/logo PNGs (any OS, needs Pillow)
    python scripts/build_msix.py --dry-run                    # print the makepri/makeappx commands, run nothing

Two packages, from the same files:

* Store upload (always): ``ProjectAway-<v>-windows-x64-store.msixupload``, a zip holding the unsigned .msix.
  Partner Center signs it. Its identity comes from MSIX_IDENTITY_NAME / MSIX_PUBLISHER /
  MSIX_PUBLISHER_DISPLAY_NAME (repo variables, from Partner Center > Product identity).
* Sideload (only when ``--sideload-publisher`` / MSIX_SIDELOAD_PUBLISHER is set): ``ProjectAway-<v>-windows-x64.msix``,
  unsigned here; release.yml has SignPath sign it. Its Publisher must equal the signing certificate's subject.

Needs makeappx.exe and makepri.exe from the Windows 10/11 SDK (preinstalled on GitHub's windows-latest runners)
for a real build; the manifest and the assets work on any OS. Stdlib only, except Pillow for the assets.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "packaging" / "msix" / "AppxManifest.xml.in"
APP_EXE = "ProjectAway.exe"

DEFAULT_IDENTITY_NAME = "FernandoGarza.ProjectAway"
UNCONFIGURED_PUBLISHER = "CN=ProjectAway-Unconfigured"
DEFAULT_PUBLISHER_DISPLAY_NAME = "Fernando Garza"

SCALES = (100, 125, 150, 200, 400)
TARGET_SIZES = (16, 24, 32, 48, 256)
# logo name -> base (width, height) at scale-100. Names match AppxManifest.xml.in.
LOGOS: dict[str, tuple[int, int]] = {
    "Square44x44Logo": (44, 44),
    "Square150x150Logo": (150, 150),
    "Wide310x150Logo": (310, 150),
    "SmallTile": (71, 71),
    "LargeTile": (310, 310),
    "StoreLogo": (50, 50),
    "SplashScreen": (620, 300),
}

_NAME_RE = re.compile(r"^[A-Za-z0-9.\-]{3,50}$")


class BuildError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ version
def pyproject_version(pyproject: Path = ROOT / "pyproject.toml") -> str:
    """The [project] version (regex: tomllib is 3.11+ and this runs on 3.10 too)."""
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if not m:
        raise BuildError(f"no version in {pyproject}")
    return m.group(1)


def msix_version(version: str) -> str:
    """'0.1.22' -> '0.1.22.0'. MSIX wants four numbers (each 0-65535) and the Store wants the last one 0, so a
    pre-release suffix is dropped ('1.2.0-rc.1' -> '1.2.0.0')."""
    v = version.strip().lstrip("vV")
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", v)
    if not m:
        raise BuildError(f"not a version: {version!r}")
    nums = [int(x) if x is not None else 0 for x in m.groups()]
    if any(n > 65535 for n in nums):
        raise BuildError(f"MSIX version parts must be 0-65535: {version!r}")
    return ".".join(str(n) for n in [*nums, 0])


# ----------------------------------------------------------------------------------------------- manifest
def validate_identity(name: str, publisher: str) -> None:
    if not _NAME_RE.match(name):
        raise BuildError(f"identity name must be 3-50 letters, digits, '.' or '-': {name!r}")
    if not publisher.startswith("CN="):
        raise BuildError(f"publisher must be a certificate subject starting with 'CN=': {publisher!r}")


def render_manifest(
    version: str,
    identity_name: str = DEFAULT_IDENTITY_NAME,
    publisher: str = UNCONFIGURED_PUBLISHER,
    publisher_display_name: str = DEFAULT_PUBLISHER_DISPLAY_NAME,
    template: Path = TEMPLATE,
) -> str:
    validate_identity(identity_name, publisher)
    attr = {'"': "&quot;"}
    values = {
        "IDENTITY_NAME": escape(identity_name, attr),
        "PUBLISHER": escape(publisher, attr),
        "PUBLISHER_DISPLAY_NAME": escape(publisher_display_name.strip() or DEFAULT_PUBLISHER_DISPLAY_NAME),
        "VERSION": msix_version(version),
    }
    text = template.read_text(encoding="utf-8")
    body_start = text.index("-->") + 3  # the header comment documents the placeholders; leave it as is
    head, body = text[:body_start], text[body_start:]
    for key, val in values.items():
        body = body.replace("{{" + key + "}}", val)
    left = re.findall(r"\{\{[A-Z_]+\}\}", body)
    if left:
        raise BuildError(f"unfilled placeholders in the manifest: {sorted(set(left))}")
    return head + body


# ------------------------------------------------------------------------------------------------- assets
def asset_files() -> dict[str, tuple[int, int]]:
    """Every PNG under Assets/ -> its pixel size: scale-qualified tiles, target-size icons (plated and unplated)
    for the taskbar/Start, and an unqualified copy of each logo (the name the manifest uses)."""
    out: dict[str, tuple[int, int]] = {}
    for name, (w, h) in LOGOS.items():
        for s in SCALES:
            out[f"{name}.scale-{s}.png"] = (int(w * s / 100 + 0.5), int(h * s / 100 + 0.5))
        out[f"{name}.png"] = (w * 2, h * 2)  # scale-200 pixels for the unqualified fallback
    for t in TARGET_SIZES:
        out[f"Square44x44Logo.targetsize-{t}.png"] = (t, t)
        out[f"Square44x44Logo.altform-unplated_targetsize-{t}.png"] = (t, t)
    return out


APP_ICON_MASTER = Path(__file__).resolve().parent.parent / "packaging" / "icon" / "ProjectAway-1024.png"
APP_ICON_BODY = (100, 100, 924, 924)  # the rounded square inside the macOS icon grid (packaging/icon/make_icons.py)


def _icon_master(size: int = 1024) -> object:
    """The app icon at ``size`` pixels. v0.1.35: the same hand icon as the Mac app (packaging/icon/ProjectAway-1024.png,
    regenerated by packaging/icon/make_icons.py), cut to its rounded square: the Mac grid's empty margin and drop
    shadow would only shrink a Windows icon or tile. Falls back to the old green disc if the master is missing."""
    from PIL import Image

    if APP_ICON_MASTER.exists():
        with Image.open(APP_ICON_MASTER) as im:
            body = im.convert("RGBA").crop(APP_ICON_BODY)
        return body.resize((size, size), Image.Resampling.LANCZOS)
    return _old_icon_master(size)


def _old_icon_master(size: int = 1024) -> object:
    """The pre-v0.1.35 icon (green disc, white open hand), drawn at ``size`` pixels.

    away.app.tray.make_icon_image uses fixed pixel gaps that suit 16-256 px; here every coordinate of its 256 px
    drawing is scaled instead, so a 1240 px tile looks like the 256 px icon rather than a hand with fused fingers.
    """

    from PIL import Image, ImageDraw

    try:
        from away.app.tray import STATE_COLORS

        color = STATE_COLORS["active"]
    except Exception:  # the tray module (or its colors) changed: keep the icon's green
        color = (46, 160, 67, 255)
    k = size / 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
        return (round(x0 * k), round(y0 * k), round(x1 * k), round(y1 * k))

    d.ellipse(box(4, 4, 252, 252), fill=color)
    w = 25  # 256 // 10, as in make_icon_image(size=256)
    for i in range(4):
        x = 64 + i * (w + 2)
        d.rounded_rectangle(box(x, 51, x + w, 132), radius=round(w / 2 * k), fill="white")
    d.rounded_rectangle(box(62, 128, 194, 204), radius=round(6 * k), fill="white")
    return img


def make_assets(dest: Path, icon: Callable[[], object] | None = None) -> list[Path]:
    """Write every tile/logo PNG into ``dest`` (the icon centered on a transparent canvas). Needs Pillow."""
    from PIL import Image

    master = (icon or _icon_master)()
    assert isinstance(master, Image.Image)
    master = master.convert("RGBA")
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, (w, h) in asset_files().items():
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # square icons fill the tile (small ones edge to edge); wide tiles and the splash keep a margin
        side = min(w, h) if (w == h and w <= 64) else int(min(w, h) * (0.8 if w == h else 0.66))
        glyph = master.resize((side, side), Image.Resampling.LANCZOS)
        canvas.paste(glyph, ((w - side) // 2, (h - side) // 2), glyph)
        path = dest / fname
        canvas.save(path, "PNG")
        written.append(path)
    return written


# -------------------------------------------------------------------------------------------- SDK + build
def find_sdk_tool(name: str, roots: Iterable[Path] | None = None) -> Path:
    """Newest x64 ``name`` (makeappx.exe, makepri.exe) in the Windows SDK, or on PATH."""
    found = shutil.which(name)
    if found:
        return Path(found)
    if roots is None:
        pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        roots = [Path(pf86) / "Windows Kits" / "10" / "bin"]
    cands: list[tuple[tuple[int, ...], Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            exe = d / "x64" / name
            if exe.is_file() and re.match(r"^\d+(\.\d+)*$", d.name):
                cands.append((tuple(int(x) for x in d.name.split(".")), exe))
    if not cands:
        raise BuildError(f"{name} not found (install the Windows 10/11 SDK; GitHub's windows-latest has it)")
    return max(cands)[1]


def write_mapping(app_dir: Path, stage: Path, mapping: Path) -> int:
    """makeappx /f mapping file: every file of the PyInstaller folder plus the staged manifest, assets and
    resources.pri, without copying gigabytes into a layout folder. Returns the number of files."""
    app_dir, stage = app_dir.resolve(), stage.resolve()
    if not (app_dir / APP_EXE).is_file():
        raise BuildError(f"{app_dir / APP_EXE} not found (run PyInstaller first)")
    lines = ["[Files]"]
    staged = {p.relative_to(stage).as_posix().lower() for p in stage.rglob("*") if p.is_file()}
    for p in sorted(app_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(app_dir)
            if rel.as_posix().lower() in staged:
                raise BuildError(f"the app folder already has {rel}, which the package needs for itself")
            lines.append(f'"{p}" "{rel.as_posix().replace("/", chr(92))}"')
    for p in sorted(stage.rglob("*")):
        if p.is_file() and p.name != "priconfig.xml":
            lines.append(f'"{p}" "{p.relative_to(stage).as_posix().replace("/", chr(92))}"')
    mapping.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - 1


def package_names(version: str) -> dict[str, str]:
    v = version.strip().lstrip("vV")
    return {
        "store_msix": f"ProjectAway-{v}-windows-x64-store.msix",
        "store_upload": f"ProjectAway-{v}-windows-x64-store.msixupload",
        "sideload_msix": f"ProjectAway-{v}-windows-x64.msix",
    }


Runner = Callable[[list[str]], None]


def _run(cmd: list[str]) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build(
    version: str,
    app_dir: Path,
    out_dir: Path,
    identity_name: str,
    publisher: str,
    publisher_display_name: str,
    sideload_publisher: str | None,
    run: Runner = _run,
    tools: dict[str, Path] | None = None,
    work: Path | None = None,
) -> list[Path]:
    names = package_names(version)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = work or out_dir / "msix-work"
    if work.exists():
        shutil.rmtree(work)
    stage = work / "stage"
    make_assets(stage / "Assets")
    tools = tools or {"makeappx": find_sdk_tool("makeappx.exe"), "makepri": find_sdk_tool("makepri.exe")}
    makeappx, makepri = str(tools["makeappx"]), str(tools["makepri"])
    manifest = stage / "AppxManifest.xml"

    def stage_manifest(pub: str) -> None:
        manifest.write_text(
            render_manifest(version, identity_name, pub, publisher_display_name), encoding="utf-8", newline="\r\n"
        )

    # resources.pri: index only the staged manifest + Assets (not the PyInstaller folder, whose locale folders
    # would be read as language qualifiers). The identity name is the same for both packages, so one PRI fits both.
    stage_manifest(publisher)
    priconfig = work / "priconfig.xml"
    run([makepri, "createconfig", "/cf", str(priconfig), "/dq", "en-US", "/pv", "10.0.0", "/o"])
    run(
        [
            makepri,
            "new",
            "/pr",
            str(stage),
            "/cf",
            str(priconfig),
            "/mn",
            str(manifest),
            "/of",
            str(stage / "resources.pri"),
            "/o",
        ]
    )

    mapping = work / "mapping.txt"
    produced = []

    store_msix = work / names["store_msix"]
    write_mapping(app_dir, stage, mapping)
    run([makeappx, "pack", "/o", "/h", "SHA256", "/f", str(mapping), "/p", str(store_msix)])
    upload = out_dir / names["store_upload"]
    with zipfile.ZipFile(upload, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        if store_msix.exists():  # (absent in --dry-run)
            z.write(store_msix, store_msix.name)
    produced.append(upload)
    if publisher == UNCONFIGURED_PUBLISHER:
        print(
            "WARNING: MSIX_PUBLISHER is not set; the Store package has a placeholder identity and Partner Center "
            "will refuse it. See README.md, 'Code signing policy'.",
            flush=True,
        )

    if sideload_publisher:
        stage_manifest(sideload_publisher)
        write_mapping(app_dir, stage, mapping)
        sideload = out_dir / names["sideload_msix"]
        run([makeappx, "pack", "/o", "/h", "SHA256", "/f", str(mapping), "/p", str(sideload)])
        produced.append(sideload)
    else:
        print("No MSIX_SIDELOAD_PUBLISHER: skipping the sideload .msix (it is only useful once signed).", flush=True)
    return produced


# -------------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    env = os.environ.get
    p = argparse.ArgumentParser(description="Build Project Away's MSIX packages from the PyInstaller output.")
    p.add_argument("--version", help="app version (default: pyproject.toml); a leading 'v' is fine")
    p.add_argument("--input", type=Path, default=ROOT / "dist" / "ProjectAway", help="PyInstaller onedir folder")
    p.add_argument("--out", type=Path, default=ROOT / "dist", help="where the packages go")
    p.add_argument("--identity-name", default=env("MSIX_IDENTITY_NAME") or DEFAULT_IDENTITY_NAME)
    p.add_argument("--publisher", default=env("MSIX_PUBLISHER") or UNCONFIGURED_PUBLISHER)
    p.add_argument(
        "--publisher-display-name", default=env("MSIX_PUBLISHER_DISPLAY_NAME") or DEFAULT_PUBLISHER_DISPLAY_NAME
    )
    p.add_argument(
        "--sideload-publisher",
        default=env("MSIX_SIDELOAD_PUBLISHER") or None,
        help="signing certificate subject; when set, also build the sideload .msix",
    )
    p.add_argument("--render-manifest", type=Path, metavar="FILE", help="only write the rendered manifest")
    p.add_argument("--assets-only", type=Path, metavar="DIR", help="only write the tile/logo PNGs")
    p.add_argument("--dry-run", action="store_true", help="print the SDK commands instead of running them")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        version = args.version or pyproject_version()
        if args.render_manifest:
            text = render_manifest(version, args.identity_name, args.publisher, args.publisher_display_name)
            args.render_manifest.write_text(text, encoding="utf-8")
            print(f"wrote {args.render_manifest} (version {msix_version(version)})")
            return 0
        if args.assets_only:
            n = len(make_assets(args.assets_only))
            print(f"wrote {n} PNGs to {args.assets_only}")
            return 0
        validate_identity(args.identity_name, args.publisher)
        if args.sideload_publisher:
            validate_identity(args.identity_name, args.sideload_publisher)
        run: Runner = _run
        tools = None
        if args.dry_run:

            def run(cmd: list[str]) -> None:
                print("would run:", subprocess.list2cmdline(cmd))

            tools = {"makeappx": Path("makeappx.exe"), "makepri": Path("makepri.exe")}
        produced = build(
            version,
            args.input,
            args.out,
            args.identity_name,
            args.publisher,
            args.publisher_display_name,
            args.sideload_publisher,
            run=run,
            tools=tools,
        )
    except (BuildError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for p in produced:
        print(f"built {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
