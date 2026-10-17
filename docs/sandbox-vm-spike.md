# Spike: the agent's own desktop (sandbox VM) with its own cursor

Status: research spike, not a feature. Nothing here ships without the on-Acer validation checklist passing.
Branch: `spike/agent-vm-desktop`. Prototype skeleton: `src/away/sandbox/`.

## The problem

Project Away drives the real system cursor today (`SendInput` family via `away/control/mouse.py`).
Windows gives each desktop session exactly one cursor, so a second, independent cursor on the same
desktop is impossible. When the agent works, the user's cursor is hijacked: the cursor jumps, the
user's typing can land in the agent's window, and the machine is unusable in parallel.

Requirement: the computer-use agent gets its own desktop with its own cursor, shown to the user in
a viewer window. The user's cursor never moves and never interferes.

## The key architectural rule

Input must be injected at the **guest session** level, never through the host input queue. Anything
that goes through host `SendInput` moves the host cursor and violates the requirement. The guest
session has its own cursor; the host cursor stays free. Every design below is judged against this rule.

## Recommended path: Windows Sandbox first, Hyper-V as fallback

### Windows Sandbox (recommended first)

Why first:

- Built into Windows 11 Pro (confirmed on the Acer: 11 Pro 26H2, build 26300.9711). No image to
  build or maintain.
- Launches in seconds: the dynamic base image shares immutable host files, so there is no boot
  of a full OS.
- Disposable by design: closing the Sandbox discards everything. That matches an agent task's
  lifecycle (spin up, do the task, throw away).
- The `.wsb` config file gives us every hook we need:
  - `MappedFolders`: ship the guest bridge binaries/scripts into the Sandbox from the host.
  - `LogonCommand`: start the guest bridge automatically at logon.
  - `Networking`: can be disabled for an air-gapped agent desktop.
  - `VGpu`, `AudioInput`, `VideoInput`: toggles as needed.
  - `ProtectedClient`, `ClipboardRedirection`, `PrinterRedirection`: tighten the boundary.
- Requirement: Pro/Enterprise/Education (not Home), virtualization enabled in BIOS, ~4 GB RAM.

What is NOT available: there is no official host API to capture the Sandbox display or inject
input from the host. The viable bridge is an **in-guest agent**: `LogonCommand` starts a small
bridge service inside the Sandbox at logon; the host talks to it over the Sandbox's virtual
network. The bridge reuses away's own screen-capture and input code (it already has both), so
the guest side is thin.

How the requirement is met: the bridge injects input into the guest session's input queue, so
the *guest* cursor moves inside the Sandbox window while the host cursor never moves. Screenshots
for the agent's vision come from the bridge (primary) or from capturing the Sandbox window rect
on the host (fallback; degrades if the window is covered).

### Hyper-V VM (fallback)

When Sandbox cannot do the job (needs persistence, needs a custom OS image, Sandbox networking
proves unworkable), fall back to a full Hyper-V VM:

- Standard, mature path: enable RDP in the VM (or install a VNC server), connect from the host.
- RDP gives framebuffer plus input events at the protocol level: the host cursor is untouched,
  the VM has its own cursor. This is the same architecture as the classic computer-use demos.
- Costs: image provisioning and servicing, RDP/VNC setup, heavier RAM/CPU, slower startup
  (tens of seconds, not seconds).

### Alternatives considered and set aside

- **Background control without a VM** (OpenAmer, Apache-2.0, Windows-first): drives target
  windows with synthetic input without touching the physical pointer. Real and respected, but the
  user explicitly asked for a VM with its own cursor, and a VM additionally gives disposal and
  isolation. Its hard-won lessons still apply to our input bridge: DPI as a first-class part of
  every coordinate, some controls only accept input when they believe they have focus, wait on
  state never on fixed sleeps, and verify against the world (outcome ledger), not the model's
  self-report.
- **Host `SendInput` while the user watches**: rejected outright. It moves the host cursor and
  violates the requirement.
- **Perplexity SPACE**: open-sourced 2026-10-15 (Apache-2.0, Rust, Firecracker microVMs,
  `perplexityai/SPACE`). It is a *code execution* sandbox on Linux/KVM hosts, not a Windows
  desktop VM. It cannot host a Windows desktop session and cannot run on the Acer. Not applicable
  here; noted only because it motivated the request.

## Module design: `src/away/sandbox/`

```
src/away/sandbox/
    __init__.py      # re-exports
    detection.py     # runtime capability detection (pure, unit-tested)
    session.py       # sandbox session lifecycle (launch / alive / terminate)
    bridge.py        # screenshot + input bridges (guest-session level)
```

Future (not in the skeleton): `viewer.py` (embed the guest desktop; Clicky-style overlay),
`guest/` (the in-guest bridge payload shipped via MappedFolders).

### detection.py

Pure functions with injected runners so they unit-test on any OS:

- `detect_edition(run) -> EditionInfo`: parses Windows edition (Pro vs Home) from a command
  runner; on non-Windows returns `unknown`.
- `sandbox_available(info, sandbox_exe_present, virtualization_on) -> bool`: Pro/Enterprise/
  Education plus the Sandbox feature present plus virtualization.
- `hyperv_available(info, hyperv_present) -> bool`: fallback gate.

The app calls this at runtime and enables the agent-desktop feature only when a backend is
available; otherwise it degrades to host control with the step budget (no silent fallback).

### session.py

- `SessionConfig`: memory_mb, networking on/off, vgpu on/off, mapped folders
  (host path, sandbox path, readonly), logon command, audio/video input toggles.
- `SessionBackend` (Protocol): `launch(config) -> SessionHandle`, `is_alive(handle) -> bool`,
  `terminate(handle) -> None`, `wait_ready(handle, timeout) -> bool`.
- `SandboxSession`: orchestrates config -> `.wsb` file -> launch -> ready -> terminate.
  The `.wsb` writer is pure (unit-tested): given a `SessionConfig`, produce the XML.
- `HyperVBackend`: stub interface for the fallback (not implemented in the spike).
- `FakeSessionBackend`: scripted lifecycle for tests.

### bridge.py

- `Frame`: width, height, pixel bytes (RGB), plus the guest cursor position `(cx, cy)` when
  known. Coordinates are always **guest pixels**; DPI scaling is applied once, at the bridge
  boundary, never ad hoc (OpenAmer scar #1).
- `ScreenshotBridge` (Protocol): `capture() -> Frame`.
- `InputBridge` (Protocol): `move(x, y)`, `click(button, count)`, `press/release`,
  `scroll(dx, dy)`, `key(name)`, `type_text(text)`, all in guest pixels.
- `GuestBridge`: the recommended implementation. Transport: HTTP/TCP over the Sandbox virtual
  network to the in-guest agent started by `LogonCommand`. Endpoints: `GET /frame`,
  `POST /input` with the action batch. The guest agent reuses away's capture/input code.
- `VncBridge`: fallback using a VNC server in the guest (TightVNC) and a Python VNC client on
  the host. Mature but heavier; kept as the escape hatch.
- `FakeScreenshotBridge` / `FakeInputBridge`: record calls, serve canned frames; for tests.

### Wiring into the existing action pipeline

The seams already exist, which is why this is a spike and not a rewrite:

- `away.control.mouse.MouseBackend` is a Protocol (`move/click/press/release/scroll/key`).
  A `SandboxMouseBackend` implements it against `InputBridge`. The action engine
  (`away.actions_engine`, the single choke point since audit Phase 1b) does not change.
- Screen source for the agent: `ScreenshotBridge.capture()` replaces the host monitor capture
  for sandboxed sessions. `away.control.monitors.Monitor/Layout` describes the *guest* layout.
- Step budget and audit trail (proposal #27, branch `feature/away-27-step-budget`): attach per
  sandbox session exactly like agent sessions. Budget default 40, `steps_left` in every action
  result, trail with screenshot thumbnails persisted per session and replayable. The audit trail
  is also the data source for the visible-cursor overlay below.
- UIA (`away/pointer/uia.py`, plus the `away-24` element-query branch): actions that UIA can do
  need no cursor at all. Inside the guest, the bridge prefers UIA for standard controls and
  falls back to coordinate input, same policy as the host.

### The visible-cursor layer (Clicky overlay reuse)

tekram/clicky-windows (MIT) draws the AI's cursor on a transparent, click-through overlay window
(`WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST`), with a pulsing blue dot, smooth 0.4s
transitions between points, a label under the cursor ("Click the File menu"), and a `[POINT]` tag
protocol: the model emits coordinates, the overlay animates to them.

Reuse plan:

- In the viewer window: draw the agent's guest-cursor position (we know it from the bridge) as
  the Clicky-style dot, so the user sees exactly where the agent is working. The guest cursor is
  already in the framebuffer; the overlay adds the label and the trail.
- On the host desktop: the same overlay shows what the agent did via UIA cursorless actions
  (the audit trail replays as animated points). This is the visual audit trail made visible.
- Implementation note: Project Away's UI is tkinter, not Electron. A borderless topmost tkinter
  window made click-through via `ctypes` `SetWindowLong` with `WS_EX_LAYERED|WS_EX_TRANSPARENT`
  achieves the same effect. Multi-monitor: one overlay window per monitor, per-monitor DPI.

## Open questions

1. Host-to-guest network reachability: the Sandbox sits behind NAT on the default Hyper-V
   switch. Can the host open a TCP connection to a port the guest bridge listens on? Must be
   measured on the Acer; vsock or a host-side port forward are fallbacks.
2. `LogonCommand` privilege: the bridge needs to capture the screen and inject input in the
   guest session. Same-session `SendInput`/screen capture needs no special privilege, but this
   must be confirmed on the Acer.
3. Frame rate and latency of bridge screenshots vs host window capture; the agent's vision loop
   needs to stay usable (target: interactive, not slideshow).
4. Sandbox startup time on the Acer (target: seconds) and behavior on close while a session is
   mid-task.
5. Whether the Sandbox window must stay visible for the guest session to keep rendering, or
   whether the bridge keeps working headless.

## On-Acer validation checklist

Run these in order; stop at the first failure and report it.

1. **Sandbox feature present.**
   Open an elevated PowerShell and run:
   `Get-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM`
   Expect `State : Enabled`. If not:
   `Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All`
   then reboot and re-check.
2. **LogonCommand works.**
   Save as `C:\Temp\away-probe.wsb`:
   `<Configuration><LogonCommand><Command>cmd.exe /c echo away-probe-ok ^> C:\Users\WDAGUtilityAccount\Desktop\probe.txt</Command></LogonCommand></Configuration>`
   Double-click it, wait for the Sandbox desktop, confirm `probe.txt` exists on the Sandbox
   desktop containing `away-probe-ok`. Close the Sandbox.
3. **MappedFolders works.**
   Add to the `.wsb`:
   `<MappedFolders><MappedFolder><HostFolder>C:\Temp\away-share</HostFolder><SandboxFolder>C:\AwayShare</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder></MappedFolders>`
   (create `C:\Temp\away-share` with a test file first). Confirm the file is visible at
   `C:\AwayShare` inside the Sandbox.
4. **Guest network reachability.**
   Inside the Sandbox, run `ipconfig` and note the IPv4 address. On the host, run
   `Test-NetConnection -ComputerName <that IP> -Port 8000` after starting
   `python -m http.server 8000` inside the Sandbox. Record whether the host can reach the
   guest. This decides the bridge transport.
5. **Bridge round trip (the critical test).**
   Build the guest bridge per `src/away/sandbox/bridge.py` (or a minimal prototype: an HTTP
   server in the guest exposing `/frame` and `/input`). Launch via `.wsb` with `MappedFolders`
   + `LogonCommand`. From the host: fetch a frame, then POST a mouse move. Confirm (a) the
   guest cursor moved inside the Sandbox window, and (b) the **host cursor did not move at
   all**. If (b) fails, the architecture is wrong; stop and report.
6. **Timing.** Measure Sandbox cold-start time (double-click to interactive desktop) and a
   screenshot round-trip time. Record both.
7. **Hyper-V fallback (only if steps 4 or 5 fail).**
   Create a Windows 11 VM in Hyper-V Manager, enable Remote Desktop in the VM, connect with
   `mstsc /v:<vm-ip>`, and confirm protocol-level input (type in the RDP window while moving
   the host mouse elsewhere; host cursor must stay free).

## References

- Windows Sandbox .wsb schema: MappedFolders, LogonCommand, Networking, VGpu, AudioInput,
  VideoInput, ProtectedClient, ClipboardRedirection, PrinterRedirection. Requires
  Pro/Enterprise/Education, virtualization in BIOS.
- tekram/clicky-windows (MIT): transparent click-through overlay, animated cursor,
  `[POINT]` tag protocol. DeepWiki: overlay window page.
- OpenAmer (Apache-2.0, Windows-first): background desktop control without VM; scars that
  apply here: DPI-first coordinates, focus borrowing, wait-on-state, outcome ledger, verify
  against the world.
- Perplexity SPACE (Apache-2.0, 2026-10-15): Linux/KVM code-execution microVMs. Not a Windows
  desktop VM; not applicable to the Acer. Listed only as a considered alternative.
- In-repo: `away/control/mouse.py` (`MouseBackend` protocol), `away/control/monitors.py`,
  `away/actions_engine.py` (single action choke point), `away/pointer/uia.py`,
  branch `feature/away-27-step-budget` (per-session step budget default 40, visual audit trail
  with thumbnails in `actions/session-<id>.jsonl`, replayable).
