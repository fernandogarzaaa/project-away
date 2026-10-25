#!/usr/bin/env bash
# Build the Project Away .dmg from the PyInstaller .app (macOS only; free, no Apple Developer ID).
# MAC_SIGN_IDENTITY set (v0.1.35): signed with that fixed certificate (see import_signing_cert.sh) so macOS keeps
# the app's permissions across updates; unset: ad-hoc, as before (a local build).
set -euo pipefail
VERSION="${1:-0.1.34}"
ARCH="$(uname -m)"
APP="dist/ProjectAway.app"
[ -d "$APP" ] || { echo "missing $APP (run pyinstaller packaging/pyinstaller/away.spec first)"; exit 1; }
if [ -n "${MAC_SIGN_IDENTITY:-}" ]; then
  codesign --force --deep --sign "$MAC_SIGN_IDENTITY" ${MAC_SIGN_KEYCHAIN:+--keychain "$MAC_SIGN_KEYCHAIN"} "$APP"
  codesign --verify --deep --strict "$APP"
  codesign -d -r- "$APP" 2>&1 | grep designated  # the identity macOS ties permissions to; must not change
else
  # ad-hoc signature (free): lets the arm64 binary run locally; Gatekeeper still needs right-click > Open
  codesign --force --deep --sign - "$APP" || true
fi
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/Project Away.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Project Away" -srcfolder "$STAGE" -ov -format UDZO "dist/ProjectAway-${VERSION}-macos-${ARCH}.dmg"
rm -rf "$STAGE"
echo "built dist/ProjectAway-${VERSION}-macos-${ARCH}.dmg"
