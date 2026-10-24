
---

### How to install

**Windows:** download `ProjectAway-<version>-windows-x64-setup.exe` and run it. If Windows SmartScreen says
"Windows protected your PC", click **More info → Run anyway**.

**Mac (Apple Silicon):** download `ProjectAway-<version>-macos-arm64.dmg`, open it, and drag **Project Away**
onto **Applications**. The Mac app is not signed by Apple, so macOS blocks it the first time. Open **Terminal**
(⌘ + Space, type `Terminal`, Return), paste this line and press Return:

```
xattr -dr com.apple.quarantine "/Applications/Project Away.app"
```

Then open Project Away. The first time, a setup window downloads the models (~1.6 GB); after that the app lives
in the **menu bar** at the top right of the screen (no window, no Dock icon). Allow Camera, Microphone,
Accessibility and Input Monitoring when macOS asks.

**Linux:** the `.AppImage` (make it executable and run it) or the `.deb` (`sudo apt install ./project-away_<version>_amd64.deb`).

**With Python** (any system, including Intel Macs): `pipx install "project-away[all]"`, then `away tray`.

`SHA256SUMS.txt` lists the checksum of every file. Full guide: [docs/INSTALL.md](https://github.com/fernandogarzaaa/project-away/blob/main/docs/INSTALL.md).
