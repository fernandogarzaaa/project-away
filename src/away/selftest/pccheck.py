"""Check my PC (v0.1.25): one click turns the README's "untested" list into facts about *this* PC.

It is the self-test (:mod:`away.selftest.checks`, :mod:`away.selftest.run`) plus a short guided part where the
person helps: "raise your hand", "pinch", "say: testing one two three", "did you hear the voice?". Every item ends
as **Pass**, **Fail** or **Skipped** with one plain sentence and, where the app can do it, a fix button.

The checks only read measurements from an environment (:class:`PcEnv`); the real one is
:class:`away.selftest.pcenv.RealPcEnv`, tests use fakes. The report is redacted twice (the self-test's
:func:`~away.selftest.redact.redact_report` and Agent Link's ``redact_text``), saved as JSON, HTML and text, and
``verified.json`` maps the README's untested items to pass / fail. It is sent to connected agents only when the
person asks for it, and then only to agents allowed to "Share my logs and recordings" or "Run camera and speed
checks" (v0.1.24 hardening rules: no event-stream broadcast, so other agents never see it).
"""

from __future__ import annotations

import contextlib
import html
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from away.selftest import checks as ck
from away.selftest.model import FAIL, PASS, SKIP, WARN, Fix, Result, report_text

DICTATION_PHRASE = "testing one two three"
GESTURE_WAIT_S = 8.0
WAKE_WAIT_S = 6.0
ECHO_MARGIN = 1.5  # the takeover echo window should be this much longer than the slowest echo measured
# The four voice engines the README lists, and how to spot each one in an engine name.
VOICE_ENGINES = (
    ("Kokoro", ("kokoro",)),
    ("Piper", ("piper",)),
    ("WinRT", ("windows natural", "winrt")),
    ("SAPI", ("sapi",)),
)
VERDICT = {PASS: "pass", WARN: "pass", FAIL: "fail", SKIP: "skipped"}
VERDICT_WORDS = {"pass": "Pass", "fail": "Fail", "skipped": "Skipped"}


class PcEnv(ck.Env, Protocol):  # every method may raise (the run turns that into a "could not run" line)
    def prompt(self, text: str) -> None: ...  # show / print an instruction and wait until the person is ready
    def confirm(self, question: str) -> bool | None: ...  # Yes / No (popup or thumbs up/down); None = no answer
    def wait_gesture(self, kind: str, seconds: float) -> dict[str, Any]: ...
    def dictation(self, phrase: str) -> dict[str, Any]: ...
    def wake_settings(self) -> dict[str, Any]: ...  # {"word": "away", "enabled": bool}
    def wake_word(self, seconds: float) -> dict[str, Any]: ...  # records after the prompt
    def voice_list(self) -> list[str]: ...
    def speak(self, name: str) -> dict[str, Any]: ...
    def tray(self) -> dict[str, Any]: ...
    def toast(self) -> dict[str, Any]: ...
    def secure_desktop(self) -> dict[str, Any]: ...
    def input_echo(self) -> dict[str, Any]: ...
    def link_reconnect(self) -> dict[str, Any]: ...


def _r(id: str, title: str, status: str, summary: str, fix: Fix | None = None, /, **details: Any) -> Result:
    return Result(id, title, status, summary, {k: v for k, v in details.items() if v is not None}, fix)


def _confirm(env: PcEnv, question: str) -> bool | None:
    try:
        return env.confirm(question)
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ cameras
INTERNAL_HINT = (
    " If this is the laptop's built-in camera: check its privacy switch or the camera key (often F8/F10), that "
    "Windows camera privacy lets desktop apps use it, and its driver in Device Manager (Cameras > right-click > "
    "Update driver, or Uninstall device and restart)."
)


def check_cameras(env: PcEnv) -> list[Result]:
    """The self-test's camera check, plus which app holds each camera and, for a camera that fails with nobody
    holding it (the user's broken internal camera), the likely reasons and where to fix them."""
    out = ck.check_cameras(env)
    for r in out:
        if not r.id.startswith("camera:") or r.status == SKIP:
            continue
        name = r.id.split(":", 1)[1]
        held = r.details.get("held_by")
        if held is None:
            with contextlib.suppress(Exception):
                who = env.camera_users(name) or {}
                held = [str(u.get("name")) for u in who.get("users") or [] if isinstance(u, dict)] or None
        r.details["held_by"] = held or []
        if r.status == FAIL and not held:
            r.summary = r.summary.rstrip() + INTERNAL_HINT
            r.details["internal_suspect"] = True
            if r.fix is None or r.fix.action == "retry_cameras":
                r.fix = Fix("open:camera_privacy", "Open camera privacy settings")
        elif r.status != FAIL:
            r.summary += " Nobody else is using it." if not held else f" Also open in: {', '.join(held)}."
    return out


# ------------------------------------------------------------------------------------------ guided gestures
GESTURES = {
    "hand": ("gesture:hand", "Raise your hand", "Raise one hand in front of the camera, palm towards it."),
    "pinch": ("gesture:pinch", "Pinch", "Now pinch: touch your thumb and index finger together, then open them."),
}


def check_gesture(env: PcEnv, kind: str) -> list[Result]:
    rid, title, text = GESTURES[kind]
    env.prompt(text)
    g = env.wait_gesture(kind, GESTURE_WAIT_S) or {}
    d = {k: g.get(k) for k in ("after_ms", "fps", "latency_ms", "backend", "camera")}
    if g.get("skipped"):
        return [_r(rid, title, SKIP, str(g["skipped"]))]
    if g.get("error"):
        msg = f"Hand tracking could not watch the camera: {g['error']}"
        return [_r(rid, title, FAIL, msg, Fix("retry_cameras", "Try again"), **d)]
    if g.get("seen"):
        secs = float(g.get("after_ms") or 0) / 1000.0
        return [_r(rid, title, PASS, f"Seen after {secs:.1f} s.", **d)]
    msg = (
        f"Not seen within {GESTURE_WAIT_S:.0f} s. More light on your hand, the hand fully in the picture, or "
        "sitting a little further back usually helps."
    )
    return [_r(rid, title, FAIL, msg, Fix(f"rerun:gesture_{kind}", "Try again"), **d)]


def check_hand(env: PcEnv) -> list[Result]:
    return check_gesture(env, "hand")


def check_pinch(env: PcEnv) -> list[Result]:
    return check_gesture(env, "pinch")


def check_tracking(env: PcEnv) -> list[Result]:
    """Hand-tracking speed and where it runs (processor vs graphics card / DirectML): the self-test's two checks."""
    return ck.check_tracking(env) + ck.check_gpu_tracking(env)


# ------------------------------------------------------------------------------------------- mic + speech
def check_dictation(env: PcEnv) -> list[Result]:
    env.prompt(f"Say: \u201c{DICTATION_PHRASE}\u201d (press OK first, then speak).")
    s = env.dictation(DICTATION_PHRASE)
    out = ck.check_stt(env, DICTATION_PHRASE, s)
    for r in out:
        r.id, r.title = "dictation", "Test dictation"
        if r.status == WARN:  # a guided test that did not understand the phrase is a failure, not a note
            r.status = FAIL
            r.fix = r.fix or Fix("rerun:dictation", "Try again")
    return out


def check_wake(env: PcEnv) -> list[Result]:
    title = "Wake word"
    ws = env.wake_settings() or {}
    env.prompt(f"Say your wake word: \u201c{ws.get('word') or 'away'}, what time is it?\u201d (press OK first).")
    w = {**ws, **(env.wake_word(WAKE_WAIT_S) or {})}
    if w.get("skipped"):
        return [_r("wake", title, SKIP, str(w["skipped"]))]
    word = str(w.get("word") or "away")
    if w.get("error"):
        return [
            _r(
                "wake",
                title,
                FAIL,
                f"Could not listen for it: {w['error']}",
                Fix("open:sound_settings", "Open sound settings"),
            )
        ]
    d = {"word": word, "heard": w.get("heard"), "enabled": w.get("enabled")}
    if w.get("matched"):
        on = "" if w.get("enabled") else " (it is turned off in Settings > Voice, so turn it on to use it)"
        return [_r("wake", title, PASS, f"Heard \u201c{word}\u201d and recognised it{on}.", **d)]
    heard = str(w.get("heard") or "")
    msg = (
        f"Heard \u201c{heard[:60]}\u201d, which does not start with \u201c{word}\u201d."
        if heard
        else f"Heard nothing when you said \u201c{word}\u201d."
    ) + " Speak a little closer, or pick a longer wake word in Settings > Voice."
    return [_r("wake", title, FAIL, msg, Fix("open:settings/voice", "Open voice settings"), **d)]


def engine_of(name: str) -> str | None:
    low = name.lower()
    for label, keys in VOICE_ENGINES:
        if any(k in low for k in keys):
            return label
    return None


def check_voices(env: PcEnv, windows: bool | None = None) -> list[Result]:
    """Kokoro, Piper, WinRT and SAPI each speak once and the person says whether they heard it. Other voices this
    PC has (PowerShell, pyttsx3, say...) are checked the same way."""
    import sys

    win = sys.platform.startswith("win") if windows is None else windows
    names = list(env.voice_list() or [])
    by_label: dict[str, str] = {}
    extra: list[str] = []
    for n in names:
        lab = engine_of(n)
        if lab and lab not in by_label:
            by_label[lab] = n
        else:
            extra.append(n)
    out = []
    for label, _keys in VOICE_ENGINES:
        rid, title = f"voice:{label.lower()}", f"Voice: {label}"
        name = by_label.get(label)
        if name is None:
            if label in ("WinRT", "SAPI") and not win:
                out.append(_r(rid, title, SKIP, "Windows only."))
            else:
                fix = Fix("open:settings/voice", "Download it") if label in ("Kokoro", "Piper") else None
                out.append(_r(rid, title, SKIP, "Not installed on this PC.", fix))
            continue
        out.append(_voice(env, rid, title, name))
    for n in extra:
        out.append(_voice(env, f"voice:{n}", f"Voice: {n}", n))
    return out


def _voice(env: PcEnv, rid: str, title: str, name: str) -> Result:
    row = env.speak(name) or {}
    d = {"engine": name, "first_audio_ms": row.get("first_audio_ms")}
    if row.get("error"):
        return _r(
            rid,
            title,
            FAIL,
            f"Could not speak: {row['error']}",
            Fix("open:settings/voice", "Choose another voice"),
            **d,
        )
    heard = _confirm(env, f"Did you hear the {name} voice say \u201cTest\u201d?")
    d["heard"] = heard
    ms = row.get("first_audio_ms")
    when = f" (first sound after {float(ms):.0f} ms)" if ms is not None else ""
    if heard is True:
        return _r(rid, title, PASS, f"Spoke and you heard it{when}.", **d)
    if heard is False:
        msg = "It ran, but you did not hear it. Check the volume and which speakers are the default."
        return _r(rid, title, FAIL, msg, Fix("open:sound_settings", "Open sound settings"), **d)
    return _r(rid, title, SKIP, f"Spoke{when}, but nobody answered whether it was heard.", **d)


# --------------------------------------------------------------------------------------- windows on screen
def check_tray(env: PcEnv) -> list[Result]:
    from away.app.tray import TOOLTIP_MAX

    t = env.tray() or {}
    title = "Tray icon and tooltip"
    if t.get("skipped"):
        return [_r("tray", title, SKIP, str(t["skipped"]))]
    tip = str(t.get("tooltip") or "")
    units = len(tip.encode("utf-16-le")) // 2
    d = {"visible": t.get("visible"), "tooltip_units": units, "limit": TOOLTIP_MAX, "error": t.get("error")}
    if t.get("error") or not t.get("visible"):
        msg = f"The tray icon is not showing ({t.get('error') or 'hidden'})."
        return [_r("tray", title, FAIL, msg, Fix("restart_app", "Restart Project Away"), **d)]
    if units > TOOLTIP_MAX:
        msg = f"The tooltip is {units} characters; Windows allows {TOOLTIP_MAX}, so the icon may stop updating."
        return [_r("tray", title, FAIL, msg, Fix("restart_app", "Restart Project Away"), **d)]
    seen = _confirm(env, "Look at the taskbar corner: do you see the Project Away icon (it may be under the ^ arrow)?")
    d["seen"] = seen
    if seen is False:
        msg = "The icon is there for Windows but you could not see it. Taskbar settings > Other system tray icons."
        return [_r("tray", title, FAIL, msg, Fix("open:taskbar_settings", "Open taskbar settings"), **d)]
    return [_r("tray", title, PASS, f"Showing, tooltip {units}/{TOOLTIP_MAX} characters.", **d)]


def check_toast(env: PcEnv) -> list[Result]:
    t = env.toast() or {}
    title = "Small message window (toast)"
    if t.get("skipped"):
        return [_r("toast", title, SKIP, str(t["skipped"]))]
    d = {"via": t.get("via"), "reason": t.get("reason")}
    fix = Fix("restart_app", "Restart Project Away")
    if not t.get("shown"):
        return [_r("toast", title, FAIL, f"It could not start: {t.get('reason') or 'no window'}.", fix, **d)]
    seen = _confirm(env, "Did a small Project Away message just appear on the screen?")
    d["seen"] = seen
    if seen is False:
        msg = "Windows says it was shown, but you did not see it. Focus / Do not disturb may hide it."
        return [_r("toast", title, FAIL, msg, Fix("open:notification_settings", "Open notification settings"), **d)]
    return [_r("toast", title, PASS, "Shows" + (f" ({t['via']})." if t.get("via") else "."), **d)]


# ----------------------------------------------------------------------------------------------- safety
def check_secure_desktop(env: PcEnv) -> list[Result]:
    s = env.secure_desktop() or {}
    title = "Lock-screen / admin-prompt detector"
    if s.get("skipped"):
        return [_r("secure_desktop", title, SKIP, str(s["skipped"]))]
    if s.get("error"):
        msg = (
            f"The check itself fails ({s['error']}), so agent control is always stopped. Safe, but agents cannot "
            "use the mouse or keyboard."
        )
        return [_r("secure_desktop", title, FAIL, msg, None, error=s["error"])]
    if s.get("active"):
        msg = "It thinks a lock screen or admin prompt is open while you are at your desktop, so agents are blocked."
        return [_r("secure_desktop", title, FAIL, msg, None, active=True)]
    return [_r("secure_desktop", title, PASS, "Works: your normal desktop is recognised.", active=False)]


def recommended_window(samples_ms: list[float], current_s: float) -> float:
    """The echo window (seconds, rounded up to 0.05) that covers the slowest echo with :data:`ECHO_MARGIN`."""
    if not samples_ms:
        return current_s
    need = max(samples_ms) / 1000.0 * ECHO_MARGIN
    return max(current_s, math.ceil(need * 20.0 - 1e-9) / 20.0)


def check_input_echo(env: PcEnv) -> list[Result]:
    """How long Windows takes to hand our own mouse / key presses back to the pynput takeover listener. If that is
    longer than the echo window, the app's own input looks like the person taking over and pauses the agent."""
    e = env.input_echo() or {}
    title = "Takeover timing (pynput echo)"
    if e.get("skipped"):
        return [_r("echo", title, SKIP, str(e["skipped"]))]
    if e.get("error"):
        return [_r("echo", title, FAIL, f"Could not measure: {e['error']}", None)]
    out = []
    for kind, label in (("mouse", "mouse"), ("key", "keyboard")):
        samples = [float(x) for x in e.get(f"{kind}_ms") or []]
        window = float(e.get(f"{kind}_window_s") or 0.0)
        missed = int(e.get(f"{kind}_missed") or 0)
        rid = f"echo:{kind}"
        if not samples:
            out.append(_r(rid, f"Takeover timing ({label})", SKIP, f"No {label} echo arrived to time.", missed=missed))
            continue
        worst = max(samples)
        rec = recommended_window(samples, window)
        d = {
            "samples_ms": [round(x, 1) for x in samples],
            "worst_ms": round(worst, 1),
            "window_s": window,
            "recommended_s": rec,
            "missed": missed,
        }
        if worst / 1000.0 > window:
            msg = (
                f"Windows hands our own {label} input back after up to {worst:.0f} ms, longer than the "
                f"{window * 1000:.0f} ms window, so an agent may pause by itself. Recommended window: {rec:.2f} s."
            )
            out.append(_r(rid, f"Takeover timing ({label})", FAIL, msg, None, **d))
        else:
            msg = f"Echo arrives within {worst:.0f} ms (window {window * 1000:.0f} ms)."
            out.append(_r(rid, f"Takeover timing ({label})", PASS, msg, **d))
    return out


# -------------------------------------------------------------------------------------------- agent link
def check_link(env: PcEnv) -> list[Result]:
    out = ck.check_link(env)
    if any(r.status == SKIP and r.id == "link" for r in out):
        return out
    r = env.link_reconnect() or {}
    title = "Agent Link reconnect"
    if r.get("skipped"):
        out.append(_r("link:reconnect", title, SKIP, str(r["skipped"])))
        return out
    d = {k: r.get(k) for k in ("tries", "ok", "ms", "reconnectable", "error")}
    tries, ok = int(r.get("tries") or 0), int(r.get("ok") or 0)
    if tries and ok == tries:
        msg = f"{ok} fresh connections in a row worked"
        if r.get("ms") is not None:
            msg += f" (slowest {float(r['ms']):.0f} ms)"
        out.append(_r("link:reconnect", title, PASS, msg + ".", **d))
    else:
        msg = f"Only {ok} of {tries} fresh connections worked ({r.get('error') or 'no answer'})."
        out.append(_r("link:reconnect", title, FAIL, msg, Fix("restart_link", "Restart Agent Link"), **d))
    return out


# --------------------------------------------------------------------------------------------- the list
PC_CHECKS: tuple[tuple[str, str, Callable[[Any], list[Result]]], ...] = (
    ("cameras", "Cameras", check_cameras),
    ("tracking", "Hand tracking speed", check_tracking),
    ("gesture_hand", "Raise your hand", check_hand),
    ("gesture_pinch", "Pinch", check_pinch),
    ("mic", "Microphone", ck.check_mic),
    ("dictation", "Test dictation", check_dictation),
    ("wake", "Wake word", check_wake),
    ("voices", "Voices", check_voices),
    ("tray", "Tray icon", check_tray),
    ("big_cursor", "Big pointer", ck.check_big_cursor),
    ("instances", "One copy running", ck.check_instances),
    ("glide", "Smooth pointer (120 Hz)", ck.check_glide),
    ("toast", "Message windows", check_toast),
    ("notify", "Agent messages", ck.check_notification),
    ("secure_desktop", "Lock-screen detector", check_secure_desktop),
    ("echo", "Takeover timing", check_input_echo),
    ("link", "Agent Link", check_link),
)
PC_CHECK_IDS = tuple(c[0] for c in PC_CHECKS)
PC_LIMITS_S = {
    "cameras": 60.0,
    "gesture_hand": 90.0,
    "gesture_pinch": 90.0,
    "dictation": 120.0,
    "wake": 120.0,
    "voices": 240.0,
    "tray": 120.0,
    "toast": 120.0,
    "link": 45.0,
}


def run(env: PcEnv, only: Iterable[str] | None = None, trigger: str = "you", progress: Any = None) -> Any:
    """Run Check my PC through the self-test's runner (time limits, one at a time, crash-proof)."""
    from away.selftest import run as runmod

    return runmod.run(
        env, only=only, trigger=trigger, progress=progress, limits=PC_LIMITS_S, checks=PC_CHECKS, kind="away.pccheck"
    )


# --------------------------------------------------------------------------------------- verified.json
# README "Verified vs untested" rows this check can settle on a real PC, and the result ids that settle them.
# ``match``: result id prefixes; ``name``: also require this text in the result id (the JoyCam rows).
UNTESTED: tuple[dict[str, Any], ...] = (
    {
        "key": "joycam",
        "readme": "JoyCam MJPG size-first retry, 24 fps preference",
        "match": ["camera:"],
        "name": "joycam",
    },
    {
        "key": "camera_reclaim",
        "readme": "Camera reclaim after Settings, who-has-the-camera status",
        "match": ["camera:"],
    },
    {"key": "two_webcams", "readme": "Two-webcam start-up, fallback and retry", "match": ["camera:"]},
    {
        "key": "hand_signs",
        "readme": "Prompts answered by thumbs up/down / fingers / voice (real landmarks)",
        "match": ["gesture:"],
    },
    {"key": "tracking_numbers", "readme": "fps/latency/CPU numbers on real hardware", "match": ["tracking", "gpu"]},
    {"key": "dictation", "readme": "Speech to text on the real microphone", "match": ["mic", "dictation"]},
    {"key": "wake_word", "readme": "Talk to your agent: wake word with real speech", "match": ["wake"]},
    {"key": "speakers", "readme": "Speaking (v0.1.21): heard on real speakers", "match": ["voice:"]},
    {"key": "tray_tooltip", "readme": "Tray tooltip limit in the real Windows tray", "match": ["tray"]},
    {
        "key": "big_cursor",
        "readme": "Big cursor: registry + SystemParametersInfo resize on real Windows",
        "match": ["big_cursor"],
    },
    {
        "key": "single_instance",
        "readme": "Only one Project Away (single-instance lock)",
        "match": ["single", "duplicates"],
    },
    {"key": "glide", "readme": "120 Hz pointer glide", "match": ["glide"]},
    {
        "key": "notice_windows",
        "readme": "Notice/badge/toast windows starting on a real PC",
        "match": ["toast", "notify"],
    },
    {"key": "secure_desktop", "readme": "Windows secure-desktop detector", "match": ["secure_desktop"]},
    {"key": "pynput_echo", "readme": "Real pynput echo timing against the 0.25 s / 0.3 s windows", "match": ["echo:"]},
    {
        "key": "link_tunnel",
        "readme": "Agent Link through a real tunnel, and reconnect",
        "match": ["link:tunnel", "link:backup", "link:named", "link:reconnect"],
    },
)


def _matches(rid: str, item: dict[str, Any]) -> bool:
    if item.get("name") and str(item["name"]).lower() not in rid.lower():
        return False
    return any(rid.startswith(m) if m.endswith(":") else (rid == m or rid.startswith(m + ":")) for m in item["match"])


def verified(report: dict[str, Any]) -> dict[str, Any]:
    """``verified.json``: each README untested item as pass / fail / untested on this PC, with the result ids."""
    items: dict[str, Any] = {}
    for it in UNTESTED:
        rows = [r for r in report.get("results") or [] if _matches(str(r.get("id", "")), it)]
        verdicts = [VERDICT.get(str(r.get("status")), "skipped") for r in rows]
        if "fail" in verdicts:
            v = "fail"
        elif "pass" in verdicts:
            v = "pass"
        else:
            v = "untested"
        items[it["key"]] = {"readme": it["readme"], "status": v, "by": [str(r.get("id")) for r in rows]}
    return {
        "kind": "away.verified",
        "version": report.get("version"),
        "platform": report.get("platform"),
        "time": report.get("time"),
        "report_id": report.get("id"),
        "items": items,
    }


# ------------------------------------------------------------------------------------- report + saving
def with_verdicts(report: dict[str, Any]) -> dict[str, Any]:
    """Pass / Fail / Skipped on every result (a self-test "needs a look" still works, so it is a Pass)."""
    out = dict(report)
    out["results"] = [
        dict(r, verdict=VERDICT.get(str(r.get("status")), "skipped")) for r in report.get("results") or []
    ]
    c = {"pass": 0, "fail": 0, "skipped": 0}
    for r in out["results"]:
        c[r["verdict"]] += 1
    out["verdicts"] = c
    out["headline"] = f"{c['pass']} passed, {c['fail']} failed, {c['skipped']} skipped"
    return out


def redact(
    report: dict[str, Any], keep_dirs: list[Path] | None = None, users: list[str] | None = None
) -> dict[str, Any]:
    """The self-test redaction (tokens, user name, paths outside the app's folders) and then Agent Link's
    ``redact_text`` on every string (e-mail addresses, home folder, pairing links, secret ``key=value``)."""
    from away.agentlink.bundle import redact as al_redact
    from away.selftest.redact import redact_report

    out: dict[str, Any] = al_redact(redact_report(report, keep_dirs, users))
    out["redacted"] = True
    return out


def text(report: dict[str, Any]) -> str:
    """Plain text: the self-test's layout with Pass / Fail / Skipped words."""
    from away.agentlink.bundle import redact_text

    lines = report_text(report).replace("Project Away self-test", "Project Away - Check my PC", 1).splitlines()
    by_title = {f"{r.get('title')}: {r.get('summary')}": r for r in report.get("results") or []}
    out = []
    for ln in lines:
        for key, r in by_title.items():
            if key in ln:
                ln = f"[{VERDICT_WORDS.get(str(r.get('verdict')), 'Skipped')}] " + ln
                break
        out.append(ln)
    return redact_text("\n".join(out))


def html_report(report: dict[str, Any]) -> str:
    from away.agentlink.bundle import redact_text

    colour = {"pass": "#1b7f3b", "fail": "#b3261e", "skipped": "#6b6b6b"}
    rows = []
    order = {"fail": 0, "pass": 1, "skipped": 2}
    for r in sorted(report.get("results") or [], key=lambda r: order.get(str(r.get("verdict")), 9)):
        v = str(r.get("verdict") or "skipped")
        fix = f"<td>{html.escape(r['fix']['label'])}</td>" if r.get("fix") else "<td></td>"
        rows.append(
            f"<tr><td style='color:{colour.get(v, '#000')};font-weight:bold'>{VERDICT_WORDS.get(v, v)}</td>"
            f"<td>{html.escape(str(r.get('title')))}</td><td>{html.escape(str(r.get('summary')))}</td>{fix}</tr>"
        )
    head = html.escape(
        f"Check my PC - {report.get('time', '')} (v{report.get('version', '?')}, {report.get('platform', '')})"
    )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Project Away - Check my PC</title>"
        "<style>body{font-family:Segoe UI,sans-serif;margin:24px}td{padding:4px 10px;vertical-align:top}"
        "tr:nth-child(even){background:#f4f4f4}</style>"
        f"<h1>{head}</h1><p>{html.escape(str(report.get('headline', '')))}</p>"
        "<table><tr><th>Result</th><th>What</th><th>Details</th><th>Fix</th></tr>" + "".join(rows) + "</table>"
    )
    return redact_text(page)


def folder(data_dir: Path | None = None) -> Path:
    from away import paths

    return Path(data_dir or paths.data_dir()) / "check"


def _write(p: Path, data: str) -> None:
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def save(report: dict[str, Any], data_dir: Path | None = None) -> dict[str, Path]:
    """Save the (redacted) report as ``check-<time>.json/.html/.txt``, ``latest.json`` and ``verified.json``."""
    d = folder(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(float(report.get("finished") or time.time())))
    base = d / f"check-{stamp}"
    paths = {
        "json": base.with_suffix(".json"),
        "html": base.with_suffix(".html"),
        "text": base.with_suffix(".txt"),
        "latest": d / "latest.json",
        "verified": d / "verified.json",
    }
    body = json.dumps(report, indent=1, default=str)
    _write(paths["json"], body)
    _write(paths["latest"], body)
    _write(paths["html"], html_report(report))
    _write(paths["text"], text(report))
    _write(paths["verified"], json.dumps(verified(report), indent=1, default=str))
    return paths


def finish(
    rep: Any, keep_dirs: list[Path] | None = None, data_dir: Path | None = None
) -> tuple[dict[str, Any], dict[str, Path]]:
    """A finished run's Report: verdicts, redaction, saved files. Returns (redacted report, paths)."""
    public = redact(with_verdicts(rep.public()), keep_dirs)
    return public, save(public, data_dir)


# ---------------------------------------------------------------------------------------------- sharing
def may_receive(agent: Any) -> bool:
    """Only agents allowed to "Share my logs and recordings" or "Run camera and speed checks" get the report."""
    from away.agentlink import perms

    scopes = set(getattr(agent, "scopes", None) or [])
    return perms.SHARE_LOGS in scopes or perms.DIAGNOSE in scopes


def note_text(report: dict[str, Any]) -> str:
    lines = [f"Project Away - Check my PC: {report.get('headline', '')}"]
    bad = [r for r in report.get("results") or [] if r.get("verdict") == "fail"]
    for r in bad[:8]:
        fix = f" (fix: {r['fix']['label']})" if r.get("fix") else ""
        lines.append(f"- FAIL: {r.get('title')}: {r.get('summary')}{fix}")
    if len(bad) > 8:
        lines.append(f"- and {len(bad) - 8} more")
    return "\n".join(lines)


def share(
    report: dict[str, Any],
    agents: Iterable[Any],
    *,
    inbox: Any = None,
    post: Callable[..., Any] | None = None,
    background: bool = True,
) -> dict[str, Any]:
    """Send the report to the allowed agents (an inbox note + their wake-up address). No event-stream broadcast:
    every connected agent can read that, including ones without either permission."""
    from away.selftest import deliver as dl

    every = list(agents)
    allowed = [a for a in every if may_receive(a)]
    out = dl.deliver(
        report, allowed, inbox=None, events=None, post=post, background=background, event="pccheck.completed"
    )
    if inbox is not None and allowed:
        with contextlib.suppress(Exception):
            out["inbox_id"] = inbox.add(note_text(report), allowed, source="system", wake=False).id
    out["skipped_agents"] = [str(getattr(a, "name", "?")) for a in every if not may_receive(a)]
    return out
