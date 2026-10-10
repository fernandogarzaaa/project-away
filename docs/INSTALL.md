# Installing Project Away

Project Away is free and local-first. Pick **one** of the install methods below.

| Method | Best for | Status |
|---|---|---|
| `pipx install "project-away[all]"` | anyone with Python 3.10–3.12 | wheel install verified in a clean venv (Linux) |
| Windows installer (`ProjectAway-x.y.z-windows-x64-setup.exe`) | Windows 10/11 | **built by CI, untested on real hardware** |
| macOS disk image (`ProjectAway-x.y.z-macos-arm64.dmg`) | Apple Silicon Macs | **built by CI, untested on real hardware** |
| Linux AppImage / `.deb` | Ubuntu/Debian/Fedora… | built and launched (`--version`, `demo`) in the sandbox on aarch64; x86_64 built by CI |

The native installers appear on the GitHub **Releases** page after a version tag is pushed
(`git tag v0.1.0 && git push origin v0.1.0`); the release is created as a **draft** so you can check it first.

Nothing is code-signed (signing certificates cost money). That means one extra click the first time
on Windows and macOS — see below.

---

## Option A — pipx (all OSes)

```bash
# 1. Python 3.10, 3.11 or 3.12 and pipx
python -m pip install --user pipx && python -m pipx ensurepath   # then open a new terminal

# 2. Install (the [all] extra = camera, face, voice, mouse/keyboard control, tray)
pipx install "project-away[all]"          # from PyPI, once it is published
# or, from a clone of the repo:
pipx install ".[all]"

# 3. First-run wizard: downloads models, checks camera + mic, writes config
away setup                                 # ~1.6 GB Whisper large-v3-turbo; use --asr-model small on slower PCs

# 4. Run
away run          # foreground, Ctrl+C to quit
away tray         # system-tray app (Pause / Dictation / Settings / Quit)
away autostart on # start the tray app at login
```

Extras if you want less: `[hands]`, `[face]`, `[voice]`, `[control]`, `[tray]`, `[train]` (PyTorch, only for
`away calibrate` / `away train`), `[parakeet]` (NVIDIA NeMo ASR, large).

`away doctor` prints which extras, models and folders it can see.

## Option B — Windows installer

1. Download `ProjectAway-<version>-windows-x64-setup.exe` from Releases.
2. Windows SmartScreen will say *"Windows protected your PC"* because the installer is unsigned:
   click **More info → Run anyway**.
3. The installer is per-user (no admin). Options: **Start when I sign in**, **Desktop shortcut**.
4. Start Menu → **Project Away**. On first launch it downloads the MediaPipe models (~12 MB) and sits in the tray.
   Start Menu → **Project Away setup (download models)** runs the full wizard (speech model, camera/mic check).
5. Privacy: *Settings → Privacy & security → Camera* and *Microphone* → allow **desktop apps**.
6. Typing into apps running as Administrator only works if Project Away also runs as Administrator (Windows UIPI rule).

## Option C — macOS disk image (Apple Silicon)

1. Download `ProjectAway-<version>-macos-arm64.dmg`, open it, drag **Project Away** to **Applications**.
2. Gatekeeper blocks unsigned apps on first open. **Right-click (or Control-click) the app → Open → Open.**
   On macOS 15+, if there is no Open button: *System Settings → Privacy & Security* → scroll down →
   **Open Anyway** next to "Project Away was blocked".
3. Grant permissions when prompted, or later in *System Settings → Privacy & Security*:
   - **Camera** — hand and face tracking
   - **Microphone** — dictation
   - **Accessibility** — moving the pointer, clicking, pasting text
   - **Input Monitoring** — the push-to-talk hotkey
   Without Accessibility/Input Monitoring macOS silently ignores synthetic input; quit and reopen the app after
   granting them.
4. It runs as a menu-bar app (no Dock icon). Intel Macs: use Option A (pipx).

## Option D — Linux AppImage or .deb

```bash
# AppImage (any distro)
chmod +x ProjectAway-<version>-linux-x86_64.AppImage
./ProjectAway-<version>-linux-x86_64.AppImage            # tray app
./ProjectAway-<version>-linux-x86_64.AppImage setup      # wizard
./ProjectAway-<version>-linux-x86_64.AppImage run        # foreground
# no FUSE (containers, some minimal distros)? add --appimage-extract-and-run

# .deb (Ubuntu/Debian)
sudo apt install ./project-away_<version>_amd64.deb      # installs /opt/project-away + `away` command
away setup
```

Linux notes:

- Mouse/keyboard control uses X11. On **Wayland** most compositors block synthetic input and global hotkeys:
  choose an *Xorg/X11 session* at the login screen.
- Your user needs camera access (`groups` should list `video`) and a PipeWire/PulseAudio microphone.
- `xdotool` (optional) lets dictation know which window is focused and lets per-app control profiles switch.
- The Settings window uses tkinter. With pipx on a distro Python install it first:
  `sudo apt install python3-tk` (Fedora: `sudo dnf install python3-tkinter`). The AppImage/.deb bundle it.

---

## After installing: choose how it listens and remap controls

The installed app starts **active and listening** (`app.always_on: true`, `voice.mode: always_on`). To change that:

- Tray icon → **Settings...** (or `away settings`): bindings table with dropdowns and key capture, sliders,
  profiles, voice mode, wake word, sensitivity, and a camera *Test gesture* tab. Press **Save**; a running app
  reloads within a second.
- Tray icon → **Mic on** toggles the microphone; **Pause** stops gestures. The default mute hotkey is Ctrl+Alt+M.
- Terminal equivalents:

```bash
away voice mode push_to_talk     # or toggle / always_on
away voice wake-word away        # only act on "away, ..."; `off` to disable
away bindings set gesture:swipe_left key:alt+left
away profile new chrome --apps chrome chrome.exe
away bindings export controls.yaml   # back up / share; `away bindings import controls.yaml`
```

No display (SSH, containers)? `away settings` prints these commands instead of opening a window.
Shell-command actions are disabled until you set `bindings.allow_shell: true` yourself.
See the README's *Customizing controls* section for every trigger, action and setting.

## Where things are stored

| | Windows | macOS | Linux |
|---|---|---|---|
| Config (`config.yaml`) | `%APPDATA%\ProjectAway` | `~/Library/Application Support/ProjectAway` | `~/.config/project-away` |
| Models, recordings | `%LOCALAPPDATA%\ProjectAway` | `~/Library/Application Support/ProjectAway` | `~/.local/share/project-away` |

Override with `AWAY_CONFIG_DIR` / `AWAY_DATA_DIR`. Nothing leaves your machine unless you turn on the optional
LLM cleanup (`voice.llm_cleanup: true` + `AWAY_LLM_BASE_URL`), which sends the dictated text to the endpoint you
configure (a local Ollama/LM Studio server keeps it on-device). Always-on listening keeps audio in memory only;
no recordings are saved unless you run `away record` (hand landmarks, not audio).

## Uninstalling

Always start with the app's own cleanup, which deletes models, recordings, config and the start-at-login entry:

```bash
away uninstall            # asks first; --yes to skip, --keep-config to keep settings, --dry-run to preview
```

Then remove the program:

| Installed with | Remove with |
|---|---|
| pipx | `pipx uninstall project-away` |
| pip | `pip uninstall project-away` |
| Windows installer | *Settings → Apps → Installed apps → Project Away → Uninstall* (also removes the autostart entry). Run `ProjectAway.exe uninstall` first to delete downloaded models. |
| macOS .dmg | Quit from the menu bar, drag `/Applications/Project Away.app` to the Trash. Remove it from *Privacy & Security* lists if you like. |
| AppImage | delete the `.AppImage` file |
| .deb | `sudo apt remove project-away` |
