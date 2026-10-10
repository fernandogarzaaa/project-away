# Wi-Fi sensing setup (experimental)

> **Status: untested on real hardware.** The CSI parsers, feature extraction and classifier are unit-tested on
> synthetic multipath data, and the PicoScenes adapter was run on the PicoScenes toolbox's own sample capture
> (`rx_by_usrpN210.csi`, USRP N210 — not an Intel card). Nothing here has been run with an AX200/AX210 yet.
> Commands below follow the PicoScenes documentation; check each against <https://ps.zpj.io> before running.

Wi-Fi sensing is optional. Project Away is fully usable with the camera alone; CSI is only a *confidence signal*
in fusion (e.g. it can veto a camera "click" when the room is empty, or support a swipe).

## What you can get, from cheapest to best

| Level | Hardware | What it senses | Cost |
|---|---|---|---|
| RSSI only | any Wi-Fi card, any OS | coarse empty / presence / motion | $0 |
| CSI (this guide) | Intel **AX200** or **AX210** on **Ubuntu 22.04** (associated or monitor mode) | presence, motion bursts, push / pull / swipe (coarse) | $0 if you already have the card; an M.2 card is otherwise a ~US$15–25 part |

## Why the router doesn't matter (much)

Your **Mercusys mesh router runs closed firmware**: we cannot install CSI tools on it. It doesn't need to: it is
simply the **transmitter**. Every frame it sends (beacons ~10/s, plus any traffic) is received by the PC's Intel
card, and the card reports the channel state for each frame. More traffic from the router = more CSI samples per
second, so keep something streaming or ping the router (step 4).

The PC needs:

- an **Intel AX200 or AX210** Wi-Fi card (M.2 2230 E-key; desktops can use a cheap PCIe adapter),
- **Ubuntu 22.04 LTS** (the release PicoScenes currently packages for; a dual-boot or spare SSD is fine).

PicoScenes documents two ways to get CSI from a router you don't control; both work with the Mercusys:

- **A. Associated mode (simplest):** the AX2xx stays connected to your Wi-Fi as normal and measures CSI on the
  frames the router sends it. Generate traffic (ping) so there are enough frames.
- **B. Fully-passive monitor mode:** the card leaves your network and overhears every frame on the router's
  channel. More samples and no association, but the PC then needs Ethernet or a second adapter for internet.

## Steps (PicoScenes)

Source: PicoScenes manual — installation <https://ps.zpj.io/manual/installation.html>, CSI scenarios
<https://ps.zpj.io/manual/scenarios.html>, channel strings <https://ps.zpj.io/manual/channels.html>.

1. **Install Ubuntu 22.04** on the PC with the AX200/AX210.
2. **Install PicoScenes.** Download the *PicoScenes Source Updater (Ubuntu 22.04 version)* `.deb` from the
   installation page, then:
   ```bash
   sudo apt install virt-what            # dependency some users report missing
   sudo apt install ./picoscenes-source-updater.deb
   MaintainPicoScenes                    # choose "1 Update/Install PicoScenes", accept the EULA
   sudo reboot
   ```
   CSI measurement on AX200/AX210 (associated and monitor mode) is in the **free** tier. Each PicoScenes build
   expires after ~60 days; re-run `MaintainPicoScenes` (or `sudo apt update && sudo apt upgrade`) to renew.
3. **Find the card's PhyPath ID**: `array_status` (examples below assume `3`).
4. **A. Associated mode** — stay connected to your Mercusys Wi-Fi, then:
   ```bash
   ping -i 0.01 <router-ip> &              # ~100 frames/s from the router (sudo for intervals < 0.2 s)
   PicoScenes "-d debug -i 3 --mode logger" # add --plot for a live view; Ctrl+C to stop
   ```
   **B. Monitor mode** — find the router's channel first (`nmcli -f SSID,CHAN,FREQ dev wifi`; mesh nodes may use
   different channels, pick the node nearest the desk), then:
   ```bash
   array_prepare_for_picoscenes 3 "5180 HT20"   # channel string format: see the channels page
   PicoScenes "-d debug -i 3 --mode logger"
   ```
   Either way the capture is written to `rx_3_<time>.csi` in the current folder.
5. **Install the PicoScenes Python Toolbox** in the same Python environment as Project Away (needs `g++`). This
   step was run in the build sandbox; the adapter parsed the toolbox's sample capture (74 frames).
   ```bash
   sudo apt install g++ python3-dev
   git clone --recursive https://github.com/wifisensing/PicoScenes-Python-Toolbox
   pip install ./PicoScenes-Python-Toolbox
   ```
6. **Point Project Away at the capture** (offline replay of a file; live streaming from PicoScenes is on the
   roadmap):
   ```yaml
   # config.yaml
   wifi:
     enabled: true
     source: picoscenes
     path: /home/you/rx_3_xxxx.csi
     sample_rate: 100        # packets per second you actually got
     fusion_weight: 0.25
   ```
   then `away run --wifi`.

## Alternatives

- **FeitCSI** (<https://github.com/KuskoSoft/FeitCSI>, open source) also extracts CSI from AX200/AX210. Its
  output is not parsed natively; convert it to the **IAX JSONL** format below and use `source: iax`.
- **RSSI only** (no special hardware): `wifi: {enabled: true, source: rssi}`. Reads `/proc/net/wireless`
  (Linux), `netsh wlan show interfaces` (Windows) or `wdutil info` (macOS, may need `sudo`).

### IAX interchange format (`source: iax`)

One JSON object per line (`.jsonl`), or CSV:

```text
{"t": 12.345, "rssi": -48, "re": [[...56 floats...], [...]], "im": [[...], [...]]}   # re/im: [n_rx][n_subcarriers]
t,rssi,re0,im0,re1,im1,...                                                       # CSV: single RX chain
```

## Calibrating the classifier

Thresholds depend on the room, distance and antennas. Push vs pull **direction** depends on geometry, so it must
be learned once: record ~10 one-second windows of each of *empty room*, *sitting still*, *push*, *pull*,
*swipe*, compute `away.wifi.features.window_features` per window and call
`CoarseCSIClassifier().calibrate([(features, label), ...])`. (A `away wifi-calibrate` command is on the roadmap.)

Push/pull/swipe need **two or more receive chains** (the AX200/AX210 have two): the antenna ratio cancels the
random per-packet phase offsets that otherwise make phase useless.
