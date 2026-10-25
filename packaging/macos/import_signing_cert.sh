#!/usr/bin/env bash
# Put the Mac signing certificate in a temporary keychain so build_dmg.sh can sign with it (v0.1.35).
#
# Every release used to be ad-hoc signed, so macOS saw each build as a new app and dropped its Accessibility,
# Input Monitoring and Camera permissions on every update. One fixed (self-signed) certificate keeps the app's
# identity the same across releases. Gatekeeper still asks on first open; that needs a paid Developer ID.
#
# usage: import_signing_cert.sh P12_FILE P12_PASSWORD [KEYCHAIN]
# Prints the keychain path. CI (release.yml) passes the p12 from the MAC_SIGN_P12 secret.
set -euo pipefail
P12="$1"
PW="$2"
KC="${3:-$(mktemp -d)/away-signing.keychain-db}"
KC_PW="$(openssl rand -hex 16)"
security create-keychain -p "$KC_PW" "$KC"
security set-keychain-settings -lut 3600 "$KC"
security unlock-keychain -p "$KC_PW" "$KC"
security import "$P12" -k "$KC" -P "$PW" -T /usr/bin/codesign >/dev/null
security set-key-partition-list -S apple-tool:,apple: -s -k "$KC_PW" "$KC" >/dev/null
# codesign only finds identities in the search list
security list-keychains -d user -s "$KC" $(security list-keychains -d user | tr -d '"')
echo "$KC"
