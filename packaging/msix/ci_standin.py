"""CI only (ci.yml "msix" job, Windows): build a tiny stand-in for dist/ProjectAway with PyInstaller, using the
same version resource and icon as the real release exe (``_windows_metadata`` from away.spec), so every pull
request proves that the spec's Windows metadata, the MSIX manifest, makepri and makeappx all work, without the
40-minute full build.

    python packaging/msix/ci_standin.py OUT_DIR    # -> OUT_DIR/ProjectAway/ProjectAway.exe (+ _internal/)
"""

import os
import re
import sys
import tempfile
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]).resolve()

spec = (ROOT / "packaging" / "pyinstaller" / "away.spec").read_text(encoding="utf-8")
src = spec[spec.index("def _windows_metadata") : spec.index("\n_win_version, _win_icon = _windows_metadata()")]
ns = {"os": os, "re": re, "sys": sys, "ROOT": ROOT, "APP": "ProjectAway"}
exec(src, ns)  # our own spec file
version, icon = ns["_windows_metadata"]()
assert version is not None, "no version resource (not Windows?)"
assert icon and Path(icon).is_file(), f"no icon: {icon!r}"

work = Path(tempfile.mkdtemp(prefix="away-standin-"))
(work / "hello.py").write_text("import sys\nprint('Project Away stand-in', sys.argv[1:])\n", encoding="utf-8")
(work / "version.txt").write_text(str(version), encoding="utf-8")
PyInstaller.__main__.run(
    [
        str(work / "hello.py"),
        "--name",
        "ProjectAway",
        "--onedir",
        "--windowed",
        "--noconfirm",
        "--icon",
        icon,
        "--version-file",
        str(work / "version.txt"),
        "--distpath",
        str(out),
        "--workpath",
        str(work / "build"),
        "--specpath",
        str(work),
    ]
)
exe = out / "ProjectAway" / "ProjectAway.exe"
assert exe.is_file(), exe
print(f"stand-in app: {exe}")
