#!/usr/bin/env bash
# Package the PyInstaller onedir (dist/ProjectAway) as an AppImage. Needs appimagetool (downloaded if absent).
set -euo pipefail
VERSION="${1:-0.1.21}"
ARCH="$(uname -m)"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$ROOT/dist/ProjectAway"
APPDIR="$ROOT/packaging/linux/AppDir"
[ -d "$DIST" ] || { echo "missing $DIST (run pyinstaller first)"; exit 1; }
rm -rf "$APPDIR" && mkdir -p "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -a "$DIST" "$APPDIR/usr/lib/ProjectAway"
cp "$ROOT/packaging/linux/project-away.desktop" "$APPDIR/project-away.desktop"
cp "$ROOT/packaging/linux/project-away.desktop" "$APPDIR/usr/share/applications/"
python3 "$ROOT/packaging/linux/make_icon.py" "$APPDIR/project-away.png"
cp "$APPDIR/project-away.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/project-away.png"
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/ProjectAway/ProjectAway" "$@"
RUN
chmod +x "$APPDIR/AppRun"
TOOL="${APPIMAGETOOL:-$ROOT/packaging/linux/appimagetool-$ARCH.AppImage}"
if [ ! -x "$TOOL" ]; then
  curl -fsSL -o "$TOOL" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
  chmod +x "$TOOL"
fi
# fetch the runtime with curl ourselves (appimagetool's built-in download ignores some proxy setups)
RUNTIME="${APPIMAGE_RUNTIME:-$ROOT/packaging/linux/runtime-$ARCH}"
if [ ! -s "$RUNTIME" ]; then
  curl -fsSL -o "$RUNTIME" "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-$ARCH"
fi
# --appimage-extract-and-run: works without FUSE (CI containers)
ARCH="$ARCH" "$TOOL" --appimage-extract-and-run --runtime-file "$RUNTIME" "$APPDIR" \
  "$ROOT/dist/ProjectAway-${VERSION}-linux-${ARCH}.AppImage"
echo "built dist/ProjectAway-${VERSION}-linux-${ARCH}.AppImage"
