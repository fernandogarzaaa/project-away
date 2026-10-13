# Privacy policy

*Project Away, by Fernando Garza. Applies to every version: the Windows installer, the Microsoft Store app, the
macOS/Linux packages and the `pip` package. Last updated: October 2026.*

**Short version:** your camera and microphone are processed on your own computer. Project Away has no account, no
ads, no analytics and no telemetry, and the author runs no server that receives your data. Something leaves your
computer only in the cases listed below, and each of them is either something you switch on yourself or an
anonymous download.

## Camera

Project Away uses your webcam to follow your hands, face and head so you can move the pointer, click and use
gestures. The video is processed **on your computer**, frame by frame, and then thrown away. It is never recorded
to a file and never uploaded by Project Away.

What is kept on your computer: the hand/face landmark positions (dots, not pictures) for a short time while the app
runs, the gestures you teach in Gesture Studio, and, when something goes wrong, a diagnostics file with the last
30 seconds of landmark positions (no video, no audio, no screenshots). These stay in Project Away's own folder
until you delete them.

## Microphone

Project Away uses your microphone for dictation (talk to type) and voice commands. Speech is turned into text by a
speech model that runs **on your computer**. Audio is not saved and not uploaded. The words you add under "My
words" are saved in your settings file.

The one exception is a feature you set up yourself: if you point "Smart cleanup" at an online AI service (by
entering its web address, or the `AWAY_LLM_BASE_URL` setting), the text of your dictation (not the audio) is sent
to that service to tidy it up, under that service's own privacy policy. By default, cleanup uses simple rules on your
computer, and the optional Smart cleanup model also runs on your computer.

## Agent Link (connecting your AI agent)

Agent Link lets an AI agent you already use (for example Claude, ChatGPT or Cursor) talk to Project Away. It only
shares anything **after you connect an agent and press Allow on your own screen**, and only the permissions you
approved (for example: asking you a question you answer with a gesture, a screenshot or camera picture when you
allow it, or using your computer while you watch). What you share goes to **that agent's provider** (for example
Anthropic or OpenAI) under their privacy policy, not to the author of Project Away.

* The Agent Link server listens on your own computer only (127.0.0.1). It does not open your computer to the network.
* An agent that is not on your computer reaches it through a temporary Cloudflare quick tunnel that exists only while
  that agent is connected. Cloudflare can see that traffic.
* You can disconnect every agent at once from the tray menu or Settings, or switch Agent Link off completely.
* An optional, off-by-default feature can open a GitHub issue with performance numbers (never images, audio or
  text you typed) using a GitHub token you provide.

## Network requests Project Away makes

| What | When | What is sent | Turn it off |
|---|---|---|---|
| Update check (GitHub Releases) | when the tray app starts and every 24 hours | an anonymous request for the latest version number; nothing about you | `updates.check: false`, or Settings, Voice & input |
| Model downloads (Google's MediaPipe storage, Hugging Face, GitHub, PyPI) | during setup, on first launch if models are missing, or when you choose an optional model, the GPU pack or the Smart cleanup engine | ordinary anonymous file downloads | don't install that model |
| Cloudflare tunnel program (from Cloudflare's GitHub releases) | only when you connect an agent that is not on this computer | an anonymous download, checked against a fixed fingerprint | don't use remote agents |
| Your AI agent | only after you connect it and press Allow | what you allowed | disconnect it |
| An online Smart cleanup service | only if you set one up | the dictated text | remove the address |
| iPhone face tracking (Live Link Face) | only if you set it up | your phone sends face data to this computer over your own Wi-Fi; nothing goes to the internet | stop the app on the phone |

These downloads come from the services named above, which may log your IP address like any web server.

## What Project Away does not do

* No telemetry, usage statistics, crash reporting to a server, advertising or tracking.
* No account or sign-in.
* No selling or sharing of data: the author never receives your data in the first place.
* No reading of your files outside Project Away's own folder.

## Where your data is stored

On Windows: settings in `%APPDATA%\ProjectAway`, models, logs and diagnostics in `%LOCALAPPDATA%\ProjectAway` (the
Microsoft Store version keeps these inside its own app storage, which Windows deletes when you uninstall it). On
macOS and Linux: the matching folders in your home folder. To delete everything, run **Project Away: uninstall**
(`away uninstall`), or delete those folders.

## Children

Project Away does not collect personal information from anyone, including children.

## Changes and contact

Changes to this policy are made in this file, with the history visible on GitHub. Questions:
[open an issue](https://github.com/fernandogarzaaa/project-away/issues) or email fernandogarzaaa@gmail.com.
