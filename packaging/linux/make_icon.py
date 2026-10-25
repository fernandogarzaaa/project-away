"""Write the 256 px app icon PNG. Usage: make_icon.py OUT.png

v0.1.35: the same hand icon as the Mac and Windows apps (scripts/build_msix.py _icon_master, from
packaging/icon/ProjectAway-1024.png); before, the tray's green disc."""

import sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from build_msix import _icon_master

    _icon_master(256).save(sys.argv[1])
except Exception:  # Pillow missing: write a 1x1 PNG so packaging still succeeds
    import base64

    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    with open(sys.argv[1], "wb") as fh:
        fh.write(base64.b64decode(png))
