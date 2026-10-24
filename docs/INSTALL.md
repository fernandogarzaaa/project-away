# Installing Project Away

Project Away is free and local-first. Pick **one** of the install methods below.

| Method | Best for | Status |
|---|---|---|
| **Windows installer** (`ProjectAway-x.y.z-windows-x64-setup.exe`) | Windows 10/11 | one click; **CI-verified on tag only, untested on real hardware** |
| **macOS disk image** (`ProjectAway-x.y.z-macos-arm64.dmg`) | Apple Silicon Macs | one drag; **CI-verified on tag only, untested on real hardware** |
| **Linux AppImage / `.deb`** | Ubuntu/Debian/Fedora… | one file; **CI-verified on tag only** (an earlier lean aarch64 build was launched in the sandbox) |
| `pipx install "project-away[all,train]"` | anyone with Python 3.10–3.12, Intel Macs, Parakeet users | wheel install verified in a clean venv (Linux) |

The native installers appear on the GitHub **Releases** page after a version tag is pushed
(`git tag v0.1.0 && git push origin v0.1.0`); the release is published as soon as every installer job passes (a tag with a `-`, such as `v0.2.0-rc1`, is
marked as a pre-release), so `away update` can see it.

## One click: what you get

Every native installer contains **every feature**. Nothing else needs to be installed:

| Feature | Bundled |
|---|---|
| Hand + face tracking | MediaPipe (with its data files), OpenCV, ONNX Runtime, the bundled gesture models |
| Dictation | faster-whisper + CTranslate2, the Silero VAD model, PortAudio (microphone) |
| Mouse/keyboard control, paste | pynput, pyperclip |
| Tray icon, Settings window | pystray + Pillow, tkinter (Tcl/Tk) |
| `away calibrate`, custom gestures, `away train` | PyTorch (**CPU build**), onnx |
| Parakeet (NVIDIA NeMo) speech recognition | **Not bundled** (see below) |

The models are downloaded **during installation** (Windows) or **on first launch** (macOS, Linux), not
by a separate step:

| Download | Size | When |
|---|---|---|
| MediaPipe hand + face models | ~12 MB | always |
| Whisper model picked for your hardware (`away hardware` shows which and why) | ~0.08–1.6 GB | default |
| Whisper `small` (for slower PCs or connections) | ~0.5 GB | Windows: tick *Use the smaller speech model*; elsewhere `setup --yes --asr-model small` |
| NVIDIA GPU pack (cuBLAS 12, only with a usable NVIDIA GPU) | ~570 MB download, ~790 MB on disk | offered by setup; or `away gpu install` |

The Whisper download uses 8 parallel connections, verifies each file's checksum, and resumes where it stopped if interrupted (run `setup` again, even after an older version was interrupted); tune with `setup --connections N` (1 = single stream). Hugging Face's download links expire after an hour; the downloader fetches a fresh link and carries on instead of starting over. On a very slow connection (under 0.5 MB/s) it prints the estimated time and suggests `--asr-model small`.

### Hardware auto-tuning and the NVIDIA GPU

On first run Project Away detects your CPU cores, RAM, free disk and GPU and picks the speech model, CPU or
GPU, precision, threads, beam size, camera resolution/frame rate and how often face tracking runs. Run
`away hardware` (or open *Settings > Performance*) to see what it found and why.

| Machine | Speech recognition |
|---|---|
| NVIDIA GPU with >= 4 GB VRAM (e.g. RTX 2060 6 GB) + GPU pack | `large-v3-turbo` on CUDA, float16 (int8 on Pascal cards), beam 5 |
| NVIDIA GPU with 2–3 GB (e.g. GTX 1050 2 GB) + GPU pack | `small` on CUDA, int8 |
| No usable GPU, 12+ cores and 16 GB | `large-v3-turbo` on the CPU, int8 |
| No usable GPU, 4+ cores and 8 GB | `small` on the CPU, int8 |
| Smaller PCs | `base` / `tiny` on the CPU, int8; lighter camera settings |
| Mac (Apple Silicon or Intel) | CPU, int8 (CTranslate2 has no Metal backend) |

**GPU (Windows / Linux, NVIDIA):** the installers stay small and do not include CUDA. When setup finds an
NVIDIA GPU it offers the **GPU pack**: a one-time download of NVIDIA's official `nvidia-cublas-cu12` 12.8
wheel from PyPI (checksum-verified), from which only the two cuBLAS libraries are kept, in your data folder.
That is all Whisper's engine (CTranslate2 4.8) needs; cuDNN is not required. You need an NVIDIA driver that
supports CUDA 12 (Windows 527.41+, Linux 525.60.13+); with an older driver it stays on the CPU and says why.

```bash
away gpu install      # download the GPU pack (installed app); resumes if interrupted
away gpu status       # what is installed and whether cuBLAS loads
away gpu remove
```

From pip/pipx, install the library into the same environment instead:
`pip install nvidia-cublas-cu12==12.8.4.1` (Project Away finds it automatically, no `LD_LIBRARY_PATH`).

Choose explicitly in `config.yaml` (or *Settings > Performance*); explicit settings always win:

```yaml
performance:
  mode: auto        # auto | cpu | gpu
  cpu_threads: 0    # 0 = auto; also beam_size, camera_width/height/fps, hands_every, face_every
voice:
  asr_model: auto   # or large-v3-turbo, small, base, tiny
  device: auto      # auto | cuda | cpu
  compute_type: auto
```

If the GPU fails to start at runtime (missing library, driver, out of memory), dictation falls back to
the CPU automatically and shows a notification with the reason.

Approximate installer sizes (CPU PyTorch is most of it; GitHub allows 2 GB per file and the release
workflow fails above 1.9 GB):

| Installer | Download | Installed |
|---|---|---|
| Windows `setup.exe` | ~0.4–0.6 GB | ~1.3–1.6 GB |
| macOS `.dmg` | ~0.4–0.6 GB | ~1.0–1.4 GB |
| Linux AppImage | ~0.5–0.8 GB | runs from the file |
| Linux `.deb` | ~0.4–0.7 GB | ~1.3–1.6 GB |

Sizes are estimates from the package sizes; the release workflow prints the real sizes of each build.
Add the model download on top.

Nothing is code-signed (signing certificates cost money). That means one extra click the first time
on Windows and macOS — see below.

### Why Parakeet is not in the installers

NVIDIA NeMo (needed for `voice.asr_backend: parakeet`) pulls in several GB of dependencies, is aimed at
Linux with NVIDIA GPUs, and cannot be added to a frozen app after installation (there is no Python
interpreter to `pip install` into). An installer-time download is therefore not feasible. Whisper
`large-v3-turbo` is the default and is bundled. To use Parakeet, install from Python:
`pipx install "project-away[all,parakeet]"`. If the installed app's config selects Parakeet, it stops with
a message that says exactly this.

---

## Option A — pipx (all OSes)

```bash
# 1. Python 3.10, 3.11 or 3.12 and pipx
python -m pip install --user pipx && python -m pipx ensurepath   # then open a new terminal

# 2. Install (the [all] extra = camera, face, voice, mouse/keyboard control, tray; [train] = PyTorch for
#    calibrate / custom gestures). CPU-only PyTorch is much smaller than the default CUDA build on Linux:
pipx install "project-away[all,train]" --pip-args="--extra-index-url https://download.pytorch.org/whl/cpu"
# or, from a clone of the repo:
pipx install ".[all,train]"

# 3. First-run wizard: downloads models, checks camera + mic, writes config
away setup                                 # picks the Whisper model for your hardware; --asr-model small on slow PCs/links
away hardware                              # what was detected and the settings picked from it

# 4. Run
away run          # foreground, Ctrl+C to quit
away tray         # system-tray app (Pause / Dictation / Settings / Quit)
away autostart on # start the tray app at login
```

Extras if you want less: `[hands]`, `[face]`, `[voice]`, `[control]`, `[tray]`, `[train]` (PyTorch, only for
`away calibrate` / `away train`), `[parakeet]` (NVIDIA NeMo ASR, large), `[winrt]` (Windows only: the WinRT
MediaCapture camera backend, the API the Windows Camera app uses; included in `[all]`, installs nothing elsewhere).

`away doctor` prints which extras, models and folders it can see.

## Option B — Windows installer (one click)

1. Download `ProjectAway-<version>-windows-x64-setup.exe` from Releases and run it.
2. Windows SmartScreen will say *"Windows protected your PC"* because the installer is unsigned:
   click **More info → Run anyway**. (Your browser may also warn about an uncommon download: choose **Keep**.)
3. Click **Next / Install**. The installer is per-user (no admin rights). These are ticked by default:
   - **Download hand-tracking and speech models now** (~1.6 GB). A console window shows the download progress.
     Tick **Use the smaller speech model** too on slower PCs (~0.5 GB).
   - **Start Project Away when I sign in**
   - **Create a desktop shortcut**
4. On the last page **Launch Project Away now** is ticked: the app starts in the tray (notification area).
5. Privacy: *Settings → Privacy & security → Camera* and *Microphone* → allow **desktop apps**.
6. Typing into apps running as Administrator only works if Project Away also runs as Administrator (Windows UIPI rule).

If a download failed (offline, proxy), the app retries it the next time it starts. Start Menu →
**Project Away: re-download models** runs it again by hand. `ProjectAway.exe doctor`, `setup`, `demo`
and the other `away` commands work from a terminal and open their own console window.

## Option C — macOS disk image (Apple Silicon)

1. Download `ProjectAway-<version>-macos-arm64.dmg` and double-click it. In the window that opens, drag
   **Project Away** onto the **Applications** folder next to it, then close the window.
2. **Let macOS open it (once).** The Mac app is not signed by Apple (that needs a paid Apple developer
   account), so macOS refuses to open it the first time. The reliable way past that: open **Terminal**
   (⌘ + Space, type `Terminal`, Return), paste this line and press Return:

   ```
   xattr -dr com.apple.quarantine "/Applications/Project Away.app"
   ```

   Nothing is printed; that is normal. Without Terminal: try to open the app, then *System Settings →
   Privacy & Security*, scroll down and click **Open Anyway** next to the Project Away message. (The old
   right-click → Open trick no longer works on recent macOS versions.)
3. **First launch** shows a *first-time setup* window with a progress bar while it downloads the models
   (~1.6 GB, once). When it closes, Project Away runs in the **menu bar** at the top right of the screen; it
   has no window and no Dock icon. The log is in `~/Library/Application Support/ProjectAway/logs/away.log`.
4. Grant permissions when prompted, or later in *System Settings → Privacy & Security*:
   - **Camera** — hand and face tracking
   - **Microphone** — dictation
   - **Accessibility** — moving the pointer, clicking, pasting text
   - **Input Monitoring** — the push-to-talk hotkey
   Without Accessibility/Input Monitoring macOS silently ignores synthetic input; quit and reopen the app after
   granting them.
5. It runs as a menu-bar app (no Dock icon). Start at login: `"/Applications/Project Away.app/Contents/MacOS/ProjectAway" autostart on`.
   Intel Macs: use Option A (pipx).

## Option D — Linux AppImage or .deb

```bash
# AppImage (any distro): everything is inside the file, including PortAudio and Tk
chmod +x ProjectAway-<version>-linux-x86_64.AppImage
./ProjectAway-<version>-linux-x86_64.AppImage            # first launch downloads the models, then the tray app
./ProjectAway-<version>-linux-x86_64.AppImage run        # foreground
./ProjectAway-<version>-linux-x86_64.AppImage autostart on
# no FUSE (containers, some minimal distros)? add --appimage-extract-and-run

# .deb (Ubuntu/Debian)
sudo apt install ./project-away_<version>_amd64.deb      # installs /opt/project-away + `away` command
away tray                                                # or the "Project Away" menu entry
```

**First launch** (tray, menu entry or `away tray`) downloads the models (~1.6 GB) with a desktop notification
(`notify-send`) and progress in the terminal, then starts the tray app. The `.deb` deliberately does **not**
download in its install script: models are per-user (`~/.local/share/project-away`), and a root-run
package script cannot safely write into a user's home or know which user will run the app.
To pick the smaller speech model instead: `away setup --yes --asr-model small` before the first launch.

Linux notes:

- Mouse/keyboard control uses X11. On **Wayland** most compositors block synthetic input and global hotkeys:
  choose an *Xorg/X11 session* at the login screen.
- Your user needs camera access (`groups` should list `video`) and a PipeWire/PulseAudio microphone.
- `xdotool` (optional) lets dictation know which window is focused and lets per-app control profiles switch.
- The Settings window uses tkinter. With pipx on a distro Python install it first:
  `sudo apt install python3-tk` (Fedora: `sudo dnf install python3-tkinter`). The AppImage/.deb bundle it.
- The AppImage uses the system PortAudio when present and its own copy otherwise.

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

## Optional: a second camera (multi-camera fusion)

A laptop webcam plus an external USB camera, ideally 30–60 cm apart and both pointed at where your
hand moves, track more reliably than one camera. Calibrating them also gives depth, which enables
the push-to-click gesture and a 3D pinch that stops false clicks when one camera sees the hand edge-on.

```bash
away cameras list                 # find the device index or name
away cameras add 2                # your current camera becomes the primary (cursor reference)
away calibrate-cameras            # move an OPEN hand slowly around both views for ~20 s
away run
```

- The primary camera should face you from the screen. Push is detected as the hand getting closer
  to it.
- **Accurate calibration:** `away calibrate-cameras --print-board board.png --checkerboard 9x6`, print
  the board flat, measure one square, then run
  `away calibrate-cameras --method checkerboard --checkerboard 9x6 --square-mm 25` and tilt the board
  in front of both cameras.
- Re-run calibration after you move either camera. Calibration data is stored in the data folder below
  as `camera_extrinsics.json`.
- **Settings → Cameras** does the same with pickers, a calibrate button and live per-camera confidence.
  While Project Away is running, *Test gesture*, *Live confidence* and *Calibrate* ask it to hand the
  cameras over and take them back afterwards. Calibration progress and the result ("calibrated,
  reprojection error X px") appear in the status line; the full output is in `calibration.log` in the
  data folder.
- **Windows:** some webcams can be opened by only one app at a time. Project Away hands its own cameras
  over to Settings automatically; close Zoom or Teams first. A secondary camera that cannot be opened is
  skipped (one notification) and tracking continues on the other.
- **Linux:** each USB camera exposes extra `/dev/video*` metadata nodes. `away cameras list` hides them.

### Choosing a camera by name, and virtual cameras (NVIDIA Broadcast, OBS...)

Project Away saves the camera you pick by its **name** (for example `HD User Facing`), so plugging in
another webcam or reordering USB ports can never switch it to a different camera. On Windows the list
comes from DirectShow, the same list OpenCV opens, so the number shown next to a name is the number that
really opens it.

```bash
away cameras list                 # index, name, and "(virtual)" for software cameras
away cameras use "HD User Facing" # single camera: pick it by name (part of the name works too)
away cameras add "USB2.0"         # add a second camera by name
```

- **Virtual cameras** such as *Camera (NVIDIA Broadcast)*, *OBS Virtual Camera*, Iriun, DroidCam, Camo,
  Snap Camera, ManyCam or XSplit appear in the list marked **(virtual)**. They usually take index 0, ahead
  of the real webcam, and Windows' own device list does not show them. Project Away never picks one on
  its own; choose it by name if you really want its (processed) picture.
- Settings written by v0.1.10 or older store a number. If that number now points at a virtual camera,
  the first real camera is used instead (the log says so). Pick your camera once in **Settings → Cameras**
  (or `away cameras use NAME`) to store it by name.
- Two identical webcams share a name; they are told apart by their Windows device path (or `NAME #2`).
- The camera list (source, index, name, virtual) is written to `away.log` at start-up and whenever it changes.
- Remove the second camera with `away cameras remove 2`. With one camera left, Away returns to
  single-camera mode.

## Other new options worth knowing

- **F9** toggles gesture control on and off from anywhere (`app.toggle_hotkey`).
- Touching the real mouse or keyboard pauses gestures for 1.5 s (`app.auto_pause`).
- `away run --cursor-overlay` shows a ring at the pointer that fills as your pinch closes.
- Dwell click (`hands.dwell.enabled`) and wink clicks (`face.wink_trigger`) are off by default. Turn
  them on in **Settings → Voice & input**.
- **Ctrl+Alt+E** edits the selected text by voice (`voice.edit_hotkey`). Clipboard access needs the
  `control` extra (pyperclip; on Linux also `xclip` or `xsel`).

## Updates

Installed apps check for a new release when the tray starts and once a day, then show a notification and an
**Update to vX.Y.Z** item in the tray menu. Click it to download, verify (SHA-256) and install the update;
the app restarts by itself.

- **Windows:** the new setup.exe runs silently, closes the app, keeps your models and relaunches it. If an
  update does not take, the next start tells you why; then **right-click the tray icon → Run update
  installer** to run the downloaded setup normally. Copies on v0.1.9 or older may need the v0.1.10 setup
  run by hand once (download it from the releases page).
- **macOS:** the app quits, the new `Project Away.app` from the `.dmg` replaces the old one (quarantine flag
  removed, so no second right-click > Open), and it reopens. If the app is not in a folder you can write to,
  you get the `.dmg` path to drag it yourself.
- **Linux AppImage:** the file is replaced in place (same name) and restarted.
- **Linux .deb:** asks for your admin password via `pkexec apt install`; nothing happens without your OK.
- **pipx / pip:** never changed automatically. Run `pipx upgrade project-away`.

From a terminal: `away update --check` (exit code 10 means an update is available) and `away update --yes`.
Turn checks off with `updates.check: false`, or install without asking when idle with
`updates.auto_install: true` (both also in **Settings → Voice & input**). The check is one anonymous request
to the GitHub Releases API; nothing about you is sent.

## Where things are stored

| | Windows | macOS | Linux |
|---|---|---|---|
| Config (`config.yaml`) | `%APPDATA%\ProjectAway` | `~/Library/Application Support/ProjectAway` | `~/.config/project-away` |
| Models, recordings | `%LOCALAPPDATA%\ProjectAway` | `~/Library/Application Support/ProjectAway` | `~/.local/share/project-away` |

Override with `AWAY_CONFIG_DIR` / `AWAY_DATA_DIR`. Apart from the anonymous update check (see **Updates**;
`updates.check: false` turns it off), nothing leaves your machine unless you turn on the optional
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
