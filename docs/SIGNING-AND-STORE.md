# Code signing and the Microsoft Store

This guide makes Project Away a "real" Windows app in two ways, both free:

1. **Code signing (SignPath Foundation).** Windows stops showing "Windows protected your PC / unknown publisher"
   for the installer, and people can install the `.msix` package directly.
2. **Microsoft Store.** People can install Project Away from the Store, which also handles updates.

They are independent: do either one first. Everything in the repository is ready. What is left are the steps only
you can do (accounts, ID checks, pasting values into GitHub). Each step says where to click.

**What the release already does today, before you do anything:** every `v*` tag still builds and publishes the
installers exactly as before, plus one new file, `ProjectAway-X.Y.Z-windows-x64-store.msixupload` (the Store
upload). Signing steps are skipped until the SignPath secrets exist, so nothing breaks while you wait for approval.

---

## Part 1: free code signing with SignPath Foundation

### What you get

SignPath Foundation is a non-profit that signs open-source apps for free. The certificate is issued **to SignPath
Foundation, not to you**, so Windows will show the publisher as **"SignPath Foundation"**. That is normal for
this program and is what removes the "unknown publisher" warning. In return, they check that every signed file
was built by GitHub Actions from this public repository, and you approve each release by hand.

Note: a brand-new certificate still needs some downloads before Windows SmartScreen fully trusts it, so a
warning may show for the first few releases. It fades as people install it.

### Before you apply (requirements)

- [ ] **Two-factor login (MFA) on GitHub.** github.com → your photo → Settings → Password and authentication →
      Enable two-factor authentication. (SignPath requires MFA for everyone who can change the code or sign.)
- [ ] **An OSI open-source license with no paid "dual license".** Project Away is MIT. Done.
- [ ] **No closed-source parts.** Everything bundled is open source. Done.
- [ ] **Already released.** Project Away has public releases on GitHub. Done.
- [ ] **Functionality described on the download page.** The README and the release notes do this. Done.
- [ ] **Code signing policy and privacy policy on the home page.** Added to the README in this change
      ("Code signing policy" section and [PRIVACY.md](../PRIVACY.md)). Done once this PR is merged.
- [ ] **"Code signing policy" on the download page too.** Add this line to each GitHub release's notes (Releases →
      Edit): `Code signing policy: https://github.com/fernandogarzaaa/project-away#code-signing-policy`

### Step 1: apply

1. Open **https://signpath.org/apply** and fill in the form.
2. Paste the text below where it fits (project description, links, build details).

> **Ready-to-paste application text**
>
> - **Project name:** Project Away
> - **Repository:** https://github.com/fernandogarzaaa/project-away
> - **Home page:** https://github.com/fernandogarzaaa/project-away
> - **Download page:** https://github.com/fernandogarzaaa/project-away/releases (GitHub Releases)
> - **License:** MIT (OSI-approved, no dual licensing). All bundled components are open source.
> - **Maintainer:** Fernando Garza (GitHub: fernandogarzaaa), the only committer and approver.
> - **What to sign:** the Windows installer (`ProjectAway-X.Y.Z-windows-x64-setup.exe`, Inno Setup), the
>   application executable inside it (`ProjectAway.exe`, PyInstaller) and a sideload MSIX package
>   (`ProjectAway-X.Y.Z-windows-x64.msix`).
> - **Build system:** GitHub Actions on GitHub-hosted runners only (`.github/workflows/release.yml`). A version tag
>   builds every installer from source; signing uses the official `signpath/github-action-submit-signing-request`
>   action, already wired into the workflow and switched off until the subscription exists.
> - **Description:** Project Away is a free, local-first accessibility and productivity app for Windows, macOS and
>   Linux. It uses the webcam to track hands, face and head so people can move the pointer, click and scroll
>   hands-free, and the microphone for on-device dictation (Whisper). It can also connect the user's own AI agent
>   (Claude, ChatGPT, Cursor, via the open Model Context Protocol) so the agent can ask quick questions the user
>   answers with a gesture. Camera and audio are processed locally; nothing is sent anywhere unless the user
>   connects an agent and approves it on screen. Privacy policy:
>   https://github.com/fernandogarzaaa/project-away/blob/main/PRIVACY.md
> - **Releases so far:** regular tagged releases on GitHub since 2026, also published to PyPI as `project-away`.

3. Wait for their email. They check the project by hand, so it can take from days to a few weeks.

### Step 2: after approval, set up the SignPath project

SignPath will give you an account on **https://app.signpath.io**. In it:

1. **Turn on MFA** for your SignPath login (required).
2. Install the **SignPath GitHub App** when asked (https://github.com/apps/signpath), and give it access to
   `project-away`.
3. Open the project SignPath created for you (or create one) and note its **slug** (the short name in its URL),
   for example `project-away`.
4. Open its **signing policy** (usually `release-signing`) and note that slug too.
5. Under **Artifact configurations**, add two configurations, pasting the files from this repo:
   - slug **`app-exe`** ← contents of `packaging/signpath/app-exe.xml`
   - slug **`release`** ← contents of `packaging/signpath/release.xml`

   (Other slugs are fine if you also set the variables `SIGNPATH_APP_CONFIG_SLUG` / `SIGNPATH_RELEASE_CONFIG_SLUG`.)
6. Create an **API token** for a user that has the "submitter" role on this project: your profile → API tokens →
   Add. Copy it (you only see it once).
7. Note your **Organization ID** (Settings / Organization, a long ID like `c2099ac1-...`).
8. Open the certificate SignPath gave the project and copy its **Subject** exactly as shown. It looks like
   `CN=SignPath Foundation, O=SignPath Foundation, L=Lewes, S=Delaware, C=US`, but use the real one: the MSIX
   publisher must match it character for character.

### Step 3: add the secrets and variables in GitHub

Go to **github.com/fernandogarzaaa/project-away → Settings → Secrets and variables → Actions**.

**Secrets** tab → *New repository secret* (these are hidden; never paste them anywhere else):

| Name | Value |
|---|---|
| `SIGNPATH_API_TOKEN` | the API token from step 2.6 |
| `SIGNPATH_ORGANIZATION_ID` | the organization ID from step 2.7 |

**Variables** tab → *New repository variable*:

| Name | Value |
|---|---|
| `SIGNPATH_PROJECT_SLUG` | the project slug from step 2.3 |
| `SIGNPATH_POLICY_SLUG` | the signing policy slug from step 2.4 |
| `MSIX_SIDELOAD_PUBLISHER` | the certificate Subject from step 2.8 (turns on the signed `.msix`) |
| `SIGNPATH_WAIT_SECONDS` | optional; how long a release waits for your approval (default `3600` = 1 hour) |

That's it. Signing switches itself on for the next version tag.

### Step 4: what a signed release looks like

When you push a tag (for example `v0.1.24`), the Windows job:

1. builds and tests the app as before;
2. sends `ProjectAway.exe` to SignPath → **you get an email: open it and click Approve**;
3. builds the installer and the MSIX packages around the signed exe;
4. sends the installer and the sideload `.msix` to SignPath → **approve again**;
5. puts the signed files on the release. `SHA256SUMS.txt` is made from these signed files, so the in-app updater's
   checks match.

Approve within the wait time (1 hour by default). If you miss it, re-run the failed **windows** job from the
Actions tab.

**Unsigned leftovers:** the files sent to SignPath are kept as short-lived workflow artifacts named
`signpath-unsigned-*`. They are never put on the release.

**If SignPath rejects a file** because of "product version": the installer's version comes from
`packaging/windows/away.iss` (AppVersion) and the exe's from `packaging/pyinstaller/away.spec`; both equal the tag
without the `v`. Open the signing request on signpath.io; it says which value it saw.

---

## Part 2: the Microsoft Store

### Step 1: free developer account

1. Go to **https://storedeveloper.microsoft.com** → **Get started for free** → **Individual developer**.
   (Start here; other paths show the old paid sign-up.)
2. Sign in with your **personal Microsoft account** (Outlook/Hotmail/Live), or create one.
3. **Verify your identity:** photograph your **government ID** and take a **selfie** on your phone, in good light,
   with the original document.
4. Check the details it fills in, finish, and click **Go to Partner Center dashboard**.

### Step 2: reserve the name

Partner Center → **Apps and games** → **New product** → **MSIX or PWA app** → name it **Project Away** →
**Reserve product name**.

### Step 3: copy the identity into GitHub

In the new app: **Product management → Product identity**. Copy three values into
**GitHub → Settings → Secrets and variables → Actions → Variables**:

| GitHub variable | Copy from Product identity |
|---|---|
| `MSIX_IDENTITY_NAME` | **Package/Identity/Name** (looks like `12345FernandoGarza.ProjectAway`) |
| `MSIX_PUBLISHER` | **Package/Identity/Publisher** (looks like `CN=1A2B3C4D-....`) |
| `MSIX_PUBLISHER_DISPLAY_NAME` | **Package/Properties/PublisherDisplayName** (your publisher name) |

These are not secrets. Then push a new version tag (or re-run the release). The release gets a
`ProjectAway-X.Y.Z-windows-x64-store.msixupload` built with your real identity. Without these variables the file
is still built, but with a placeholder publisher that Partner Center refuses.

### Step 4: create the submission

In Partner Center → your app → **Start your submission**:

1. **Pricing and availability:** Free; all markets (or the ones you want).
2. **Properties:** Category **Productivity** (sub: none) or **Utilities & tools**. Tick that the app
   **accesses the webcam and microphone**. Privacy policy URL:
   `https://github.com/fernandogarzaaa/project-away/blob/main/PRIVACY.md`. Website:
   `https://github.com/fernandogarzaaa/project-away`. Support: `https://github.com/fernandogarzaaa/project-away/issues`.
   System requirements: webcam recommended, microphone recommended, 8 GB RAM recommended, ~4 GB free disk.
3. **Age ratings:** answer the questionnaire (answers below).
4. **Packages:** download `ProjectAway-X.Y.Z-windows-x64-store.msixupload` from the GitHub release and drag it in.
   Device family: Windows 10/11 Desktop.
5. **Store listings → English:** paste the listing text below; add at least one screenshot (1366×768 or larger)
   of the Settings window and the tray menu.
6. **Submission options → Restricted capabilities:** paste the runFullTrust justification below.
7. **Submit to the Store.** Certification usually takes a few days. Each later version: new submission, new
   `.msixupload`, same steps 4–7 (the listing is remembered).

### Ready-to-paste Store listing

**Product name:** Project Away

**Short description** (shown at the top):

> Control your PC and your AI agents with your hands, face and voice. Point to move, pinch to click, talk to type.
> Free, open source, and processed on your PC.

**Description:**

> Project Away turns your webcam and microphone into a hands-free way to use your computer.
>
> Point at the screen to move the pointer, pinch to click, pinch and hold to drag, and raise two fingers to scroll. Talk to
> type anywhere: your words are turned into text by a speech model that runs on your own PC. Look away and gestures
> pause by themselves; look back and they continue.
>
> New: Agent Link connects the AI agent you already use (Claude, ChatGPT, Cursor and others) to Project Away. Your
> agent can ask you a quick question you answer with a thumbs up, tell you when a job is done, or, only if you allow
> it, look at your screen or use your computer while you watch.
>
> A friendly guide walks you through the first five minutes with big buttons, and every step can be skipped.
>
> Private by design: video and audio are processed on your PC and never uploaded. There is no account, no ads and no
> tracking. Nothing is shared unless you connect an agent and press Allow on your own screen.
>
> Free and open source (MIT): https://github.com/fernandogarzaaa/project-away
>
> Needs a webcam for gestures and a microphone for voice. The first start downloads the tracking and speech models
> (about 0.5–1.6 GB).

**Features** (one per line):

> Move the pointer by pointing, click by pinching, scroll and drag with gestures
> Voice typing that runs on your PC, in any app
> Teach your own gestures in Gesture Studio, no coding
> Shortcuts: show an action once, then repeat it with a gesture or a word
> Head and eye tracking for people who prefer not to use their hands
> Only responds when you are looking at the screen (optional)
> Arm-rest mode and break reminders for long sessions
> Agent Link: connect Claude, ChatGPT, Cursor and other AI agents
> Works offline after setup; no account, no ads, no tracking
> Free and open source

**Keywords** (up to 7): `hands-free`, `gesture control`, `accessibility`, `voice typing`, `dictation`,
`head tracking`, `AI agent`

**Copyright:** © 2026 Fernando Garza. MIT License.

**What the app does with the camera and microphone** (for reviewers; paste into "Notes for certification" too):

> Project Away is an accessibility app. The webcam is used to track the user's hands, face and head in real time so
> they can control the mouse pointer and trigger actions without touching the mouse. The microphone is used for
> dictation and voice commands. Both are processed entirely on the device (Google MediaPipe for tracking, OpenAI
> Whisper via faster-whisper for speech); no video or audio is recorded or uploaded. To test: start the app (the
> first start downloads the models), allow camera and microphone, point with the index finger to move the pointer,
> tap thumb and index finger together to click, then open Notepad and speak a sentence: it is typed. Ctrl+Alt+Space
> turns listening on and off.

### Ready-to-paste runFullTrust justification

> Project Away is a full-trust desktop application (a Python app packaged with PyInstaller) and needs runFullTrust
> to work at all. Its core function is hands-free control of the whole PC for accessibility: it moves the system
> pointer, sends clicks and key presses to whatever app is in front (SendInput), shows a notification-area (tray)
> icon, reads the focused control through Windows UI Automation to snap the pointer to buttons, and runs a
> localhost-only server (127.0.0.1) so the user's own AI agent can talk to it. None of this is possible inside the
> app container. The camera and microphone are processed on the device and never uploaded; see the privacy policy.

### Age rating answers (IARC questionnaire)

Pick **"Utility, productivity, communication or other app"** (not a game), then:

| Question (wording varies slightly) | Answer |
|---|---|
| Violence, blood, fear, sexuality, nudity, crude humor, bad language | No |
| Drugs, alcohol, tobacco references | No |
| Gambling or simulated gambling | No |
| Can users interact with or send content to other users? | No (Agent Link talks to the user's own AI agent, not to other people) |
| Does it share the user's location with other users? | No |
| Digital purchases? | No |
| Unrestricted internet/web browsing? | No |
| Does it collect personal information? | No (nothing is sent to the developer) |

Expected result: **3+ / Everyone**.

### Things the Store review may flag (known, honest)

- **In-app updates.** The installed app checks GitHub for updates and can run the Inno Setup installer. A Store copy
  should be updated by the Store only, so the updater needs to recognise an MSIX install and stand down. That is a
  small change in `src/away/app/updater.py` (`detect_install`: an exe under `\WindowsApps\` → no in-app updates),
  **not included in this change** because `src/` was being rewritten for v0.1.23 at the same time. Until it lands,
  turn updates off in the Store build or explain in the certification notes that `updates.check` is user-controlled.
- **Downloads of extra components.** The optional Smart cleanup engine, the GPU pack and the Cloudflare tunnel
  program are downloaded on request. Describe them in the listing if a reviewer asks (policy 10.2.2), or hide those
  buttons in the Store build later.
- **Start at sign-in.** In the MSIX the switch is a Windows startup task (off by default; Settings → Apps →
  Startup → Project Away). The app's own "start at sign-in" option writes a registry key that packaged apps can't
  use, so it does nothing in the MSIX until the app calls the StartupTask API (another small `src/` follow-up).
- **Where data lives.** Windows keeps an MSIX app's `%LOCALAPPDATA%` files in the app's private storage, so the Store
  copy downloads its own models (the installer copy's models are not shared) and Windows removes them on uninstall.

---

## Part 3 (optional): winget

winget lets people run `winget install FernandoGarza.ProjectAway`. Templates are in `packaging/winget/`.

1. After a release, copy the three files, replace `{{VERSION}}` with the version (e.g. `0.1.24`) and `{{SHA256}}`
   with the installer's line from that release's `SHA256SUMS.txt` (upper or lower case both work).
2. Check them: `winget validate --manifest <folder>` and test `winget install --manifest <folder>` on a Windows PC.
3. Fork https://github.com/microsoft/winget-pkgs, add the files under
   `manifests/f/FernandoGarza/ProjectAway/<version>/`, and open a pull request. A bot tests it; a moderator merges.
   (`wingetcreate update FernandoGarza.ProjectAway --version X.Y.Z --urls <installer url> --submit` automates later
   versions.)

Wait until the installer is signed before submitting: winget's checks flag unsigned installers more often.

---

## Reference: every secret and variable

| Name | Kind | Needed for | Where it comes from |
|---|---|---|---|
| `SIGNPATH_API_TOKEN` | secret | signing | SignPath → profile → API tokens |
| `SIGNPATH_ORGANIZATION_ID` | secret | signing | SignPath → organization settings |
| `SIGNPATH_PROJECT_SLUG` | variable | signing | SignPath project URL |
| `SIGNPATH_POLICY_SLUG` | variable | signing | SignPath signing policy |
| `SIGNPATH_APP_CONFIG_SLUG` | variable, optional | signing | default `app-exe` |
| `SIGNPATH_RELEASE_CONFIG_SLUG` | variable, optional | signing | default `release` |
| `SIGNPATH_WAIT_SECONDS` | variable, optional | signing | default `3600` |
| `MSIX_SIDELOAD_PUBLISHER` | variable | signed `.msix` | SignPath certificate Subject |
| `MSIX_IDENTITY_NAME` | variable | Store upload | Partner Center → Product identity |
| `MSIX_PUBLISHER` | variable | Store upload | Partner Center → Product identity |
| `MSIX_PUBLISHER_DISPLAY_NAME` | variable | Store upload | Partner Center → Product identity |

Signing runs only on version tags and only when the first four are all set. Build it locally on Windows with
`python scripts/build_msix.py --help` (needs the Windows SDK); the manifest template is
`packaging/msix/AppxManifest.xml.in`.
