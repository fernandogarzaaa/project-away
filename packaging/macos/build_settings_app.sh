#!/usr/bin/env bash
# v0.1.35: build the native macOS Settings app (SwiftUI, macos/SettingsApp) into
#   macos/SettingsApp/build/Project Away Settings.app
# away.spec copies it into ProjectAway.app/Contents/Helpers/ (build_dmg.sh checks it is there); `away settings`
# opens it instead of the Tk window when it is present (away.app.settings_native). Needs Xcode (Liquid Glass needs
# Xcode 26+; an older Xcode builds the same app with standard materials).
#   bash packaging/macos/build_settings_app.sh [VERSION]
set -euo pipefail
VERSION="${1:-${AWAY_VERSION:-0.1.34}}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="$ROOT/macos/SettingsApp"
OUT="$PKG/build/Project Away Settings.app"
EXE=ProjectAwaySettings

swift --version
swift build -c release --package-path "$PKG"
BIN="$(swift build -c release --package-path "$PKG" --show-bin-path)/$EXE"

rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
cp "$BIN" "$OUT/Contents/MacOS/$EXE"
cat > "$OUT/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>com.fernandogarzaaa.projectaway.settings</string>
  <key>CFBundleName</key><string>Project Away Settings</string>
  <key>CFBundleDisplayName</key><string>Project Away Settings</string>
  <key>CFBundleExecutable</key><string>$EXE</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.utilities</string>
</dict>
</plist>
PLIST

# The app icon: the tray icon's drawing (scripts/build_msix.py), when Python with Pillow is around
PY="${PYTHON:-python3}"
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
if PYTHONPATH="$ROOT/src:$ROOT/scripts" "$PY" - "$ICONSET" <<'PYCODE'
import sys
from build_msix import _icon_master

out = sys.argv[1]
for size in (16, 32, 128, 256, 512):
    _icon_master(size).save(f"{out}/icon_{size}x{size}.png")
    _icon_master(size * 2).save(f"{out}/icon_{size}x{size}@2x.png")
PYCODE
then
  iconutil -c icns "$ICONSET" -o "$OUT/Contents/Resources/AppIcon.icns" || echo "no app icon (iconutil failed)"
else
  echo "no app icon (Python with Pillow not found)"
fi

# ad-hoc signature (free), like build_dmg.sh: lets the arm64 binary run
codesign --force --sign - "$OUT" || true
echo "built $OUT"
