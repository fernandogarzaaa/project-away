#!/usr/bin/env bash
# Package the PyInstaller onedir as a .deb (installs to /opt/project-away, adds `away` + menu entry).
set -euo pipefail
VERSION="${1:-0.1.6}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$ROOT/dist/ProjectAway"
case "$(uname -m)" in x86_64) DEBARCH=amd64 ;; aarch64) DEBARCH=arm64 ;; *) DEBARCH="$(uname -m)" ;; esac
PKG="$(mktemp -d)/project-away_${VERSION}_${DEBARCH}"
mkdir -p "$PKG/DEBIAN" "$PKG/opt/project-away" "$PKG/usr/bin" "$PKG/usr/share/applications" "$PKG/usr/share/icons/hicolor/256x256/apps"
cp -a "$DIST/." "$PKG/opt/project-away/"
ln -s /opt/project-away/ProjectAway "$PKG/usr/bin/away"
sed 's#^Exec=ProjectAway#Exec=/opt/project-away/ProjectAway#' "$ROOT/packaging/linux/project-away.desktop" > "$PKG/usr/share/applications/project-away.desktop"
python3 "$ROOT/packaging/linux/make_icon.py" "$PKG/usr/share/icons/hicolor/256x256/apps/project-away.png"
SIZE_KB="$(du -sk "$PKG/opt" | cut -f1)"
cat > "$PKG/DEBIAN/control" <<CTL
Package: project-away
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${DEBARCH}
Installed-Size: ${SIZE_KB}
Depends: libc6, libegl1, libgl1, libportaudio2, libxtst6
Recommends: xdotool
Maintainer: Fernando Garza <fernandogarzaaa@gmail.com>
Description: Hands-free PC control (gestures, dictation, speaker gating)
 Local-first: camera hand gestures move the pointer, push-to-talk dictation types text.
CTL
dpkg-deb --root-owner-group --build "$PKG" "$ROOT/dist/project-away_${VERSION}_${DEBARCH}.deb"
echo "built dist/project-away_${VERSION}_${DEBARCH}.deb"
