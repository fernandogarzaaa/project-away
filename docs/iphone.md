# iPhone face tracking (experimental, $0)

> **Status: experimental, untested with a real iPhone.** The packet parser, listener, source switching and
> CLI are unit-tested with synthetic packets built to the documented layout; no packet from a real phone has
> been received yet. Treat this as a validation experiment: run `away iphone compare` and decide from the
> numbers whether it beats your webcam.

## What it is for

The face features in Project Away (wink / mouth-open / brow-raise clicks and the lip-motion gate that only
types speech when *your* mouth is moving) read 52 face **blendshapes**. By default they come from MediaPipe
Face Landmarker on the webcam image. An iPhone or iPad with a **TrueDepth (Face ID) camera** computes the
same ARKit blendshapes from a depth sensor, usually more stable in low light and for small motions like a
wink or a lip twitch. Epic Games' free **Live Link Face** app streams them over Wi-Fi; Project Away can now
listen for that stream and use it instead of, or as a fallback with, the webcam.

Cost: **$0**. Live Link Face is free, nothing is installed on the phone besides it, and nothing leaves your
local network (the phone sends UDP packets straight to this PC).

Hand tracking still uses the webcam; only the face features change source.

## Phone setup

1. Install **Live Link Face** (by Unreal Engine / Epic Games) from the App Store. Needs Face ID
   (TrueDepth front camera).
2. Open it, tap the **gear** (Settings) and set the capture mode to **Live Link (ARKit)**. The
   *MetaHuman Animator* / *Live Link (MetaHuman)* modes use a different protocol and will not work.
3. **Settings > Streaming / Live Link > Add Target**: enter this PC's LAN IP address and port **11111**.
   `away iphone listen` prints the IP to use (on Windows `ipconfig` also shows it: the IPv4 address of the
   Wi-Fi or Ethernet adapter).
4. Phone and PC on the **same Wi-Fi network** (not a guest network or one with client/AP isolation).
5. On the app's main screen make sure **Live** is on (green) and look at the phone.

## PC setup

```bash
away iphone listen          # prints this PC's IP + the steps above, then live packet rate and values
```

The first time, **Windows Firewall** asks whether Python / Project Away may use the network: allow it on
**Private networks** (your home Wi-Fi should be set to *Private* in Windows network settings). If no prompt
appears and nothing arrives, add the rule yourself in an administrator terminal:

```bat
netsh advfirewall firewall add rule name="Project Away iPhone" dir=in action=allow protocol=UDP localport=11111 profile=private
```

`listen` shows e.g. `59.8 pkt/s  jawOpen 0.12  eyeBlinkLeft 0.03  eyeBlinkRight 0.02  browInnerUp 0.10  mouthSmile 0.05`.
`waiting for packets` means nothing reached the PC (IP, port, Wi-Fi or firewall); `unreadable ... version`
means the app is not in Live Link (ARKit) mode; `no face` means the phone can't see your face.

## Compare it with the webcam

```bash
away iphone compare --seconds 30          # key blendshapes;  --all logs all 51 shared ones
```

For 30 s (talk, blink, wink each eye, raise your brows, open your mouth) it logs webcam and iPhone values
side by side, one row per webcam frame, to `<data dir>/iphone/compare-<time>.csv` (Windows:
`%LOCALAPPDATA%\ProjectAway\iphone\`), and prints per blendshape the mean of each, the mean absolute
difference and the correlation. High correlation with a cleaner iPhone signal (wink without the other eye
moving, jaw without jitter) is the case for using it.

If the iPhone's `eyeBlinkLeft` follows your webcam's `eyeBlinkRight` (winks reversed), set
`face.iphone.swap_sides: true`.

## Use it in the app

```yaml
face:
  source: auto          # webcam (default) | iphone | auto
  iphone:
    port: 11111
    stale_after: 0.5    # s without a face packet before auto falls back to the webcam
    max_rate: 30        # frames/s fed to the face pipeline (the phone sends ~60)
    swap_sides: false
```

- `webcam`: unchanged behaviour (default).
- `iphone`: the webcam face model is not run at all (saves CPU); when the phone stops sending, the app sees
  "no face", so always-on dictation (which requires your face) pauses until it is back.
- `auto`: the iPhone while fresh packets with a face arrive, the webcam otherwise. Each switch re-enrolls
  the primary face. Changing `face.source` needs an app restart.

## Limits

- The iPhone sees one face. With `iphone`, the "who is talking" correlation between several people cannot
  run; the gate becomes "is the person at the phone moving their lips while the mic hears speech".
- Wi-Fi adds a few ms of latency and can drop packets; `stale_after` covers short gaps.
- The OSC feature of Live Link Face is remote control only (start/stop recording); it carries no face data,
  so there is nothing to receive over OSC.
- Protocol: Live Link packet version 6 (UE5-era app). If a future app version changes it, `listen` will
  report the unsupported version instead of misreading data.
