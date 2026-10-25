#!/usr/bin/env bash
# Build an UNSIGNED Project Away .dmg from the PyInstaller .app (macOS only; free, no Apple Developer ID).
set -euo pipefail
VERSION="${1:-0.1.34}"
ARCH="$(uname -m)"
APP="dist/ProjectAway.app"
[ -d "$APP" ] || { echo "missing $APP (run pyinstaller packaging/pyinstaller/away.spec first)"; exit 1; }
# v0.1.35: the native Settings app (away.spec copies it; this covers an app built by an older spec)
HELPER="Project Away Settings.app"
if [ ! -d "$APP/Contents/Helpers/$HELPER" ] && [ -d "macos/SettingsApp/build/$HELPER" ]; then
  mkdir -p "$APP/Contents/Helpers"
  cp -R "macos/SettingsApp/build/$HELPER" "$APP/Contents/Helpers/"
fi
[ -d "$APP/Contents/Helpers/$HELPER" ] && echo "native Settings: $APP/Contents/Helpers/$HELPER" \
  || echo "warning: no native Settings app in $APP (macOS Settings will be the Tk window)"
# ad-hoc signature (free): lets the arm64 binary run locally; Gatekeeper still needs right-click > Open
codesign --force --deep --sign - "$APP" || true
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/Project Away.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Project Away" -srcfolder "$STAGE" -ov -format UDZO "dist/ProjectAway-${VERSION}-macos-${ARCH}.dmg"
rm -rf "$STAGE"
echo "built dist/ProjectAway-${VERSION}-macos-${ARCH}.dmg"
