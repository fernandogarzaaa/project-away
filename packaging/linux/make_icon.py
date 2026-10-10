"""Write the app icon PNG (same glyph as the tray icon). Usage: make_icon.py OUT.png"""

import sys

try:
    from away.app.tray import make_icon_image

    make_icon_image(False, 256).save(sys.argv[1])
except Exception:  # Pillow missing: write a 1x1 PNG so packaging still succeeds
    import base64

    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    open(sys.argv[1], "wb").write(base64.b64decode(png))
