"""The self-test's checks (v0.1.24): each reads measurements from an *environment* and turns them into pass / warn /
fail with plain words and a fix button.

The environment (:class:`Env`) is where the real PC is touched: :class:`away.selftest.env.RealEnv` opens cameras,
records the microphone, speaks, pings the link... Tests use a fake one, so every rule below runs in CI without a
camera, a microphone, a GPU or a network. Each env method returns plain dicts and may raise; a check that raises
becomes a "this check could not run" line, never a crashed self-test.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from away.selftest.model import FAIL, PASS, SKIP, WARN, Fix, Result

CAMERA_GOOD_FPS = 20.0
CAMERA_MIN_FPS = 10.0
TRACKING_GOOD_FPS = 20.0
TRACKING_MIN_FPS = 10.0
TRACKING_SLOW_MS = 60.0  # one hand-tracking run slower than this cannot keep up with a 15+ fps camera
GLIDE_GOOD_RATIO = 0.8  # measured pointer updates / the configured rate (hands.cursor.output_hz)
GLIDE_MIN_HZ = 50.0
MIC_SILENT_DB = -60.0  # quieter than this on average = nothing reaches the microphone
MIC_LOUD_DB = -1.0  # peaks above this = clipping
TTS_SLOW_MS = 1500.0
LINK_SLOW_MS = 1500.0
CPU_BUSY_PCT = 85.0
GPU_BUSY_PCT = 90.0
STT_MATCH = 0.6  # share of the phrase's words heard back

PHRASE = "project away is ready"


class Env(Protocol):  # every method may raise; see away.selftest.env.RealEnv for what they return
    def cameras(self) -> list[dict[str, Any]]: ...
    def camera_users(self, camera: str) -> dict[str, Any]: ...
    def tracking(self) -> dict[str, Any]: ...
    def gpu_tracking(self) -> dict[str, Any]: ...
    def glide(self) -> dict[str, Any]: ...
    def big_cursor(self) -> dict[str, Any]: ...
    def instances(self) -> dict[str, Any]: ...
    def mic(self) -> dict[str, Any]: ...
    def stt(self, phrase: str) -> dict[str, Any]: ...
    def tts(self) -> list[dict[str, Any]]: ...
    def notification(self) -> dict[str, Any]: ...
    def link(self) -> dict[str, Any]: ...
    def load(self) -> dict[str, Any]: ...
    def models(self) -> dict[str, Any]: ...


def _r(id: str, title: str, status: str, summary: str, fix: Fix | None = None, /, **details: Any) -> Result:
    return Result(id, title, status, summary, {k: v for k, v in details.items() if v is not None}, fix)


# ------------------------------------------------------------------------------------------------ cameras
def _camera_fix(env: Env, name: str) -> tuple[str, Fix, list[str]]:
    who: dict[str, Any] = {}
    try:
        who = env.camera_users(name) or {}
    except Exception:
        who = {}
    msg = str(who.get("message") or f"{name} could not be opened.")
    users = [str(u.get("name")) for u in who.get("users") or [] if isinstance(u, dict)]
    fix = Fix("retry_cameras", "Try again")
    for a in who.get("actions") or []:
        if str(a).startswith("close:") and users:
            fix = Fix("close_app:" + str(a)[len("close:") :], f"Close {users[0]}")
            break
        if a == "use_broadcast":
            fix = Fix("use_broadcast", "Use it through NVIDIA Broadcast")
            break
    return msg, fix, users


def check_cameras(env: Env) -> list[Result]:
    cams = env.cameras()
    if not cams:
        return [
            _r(
                "camera",
                "Camera",
                FAIL,
                "No camera was found. Plug one in, or check that Windows lets apps use the camera.",
                Fix("open:camera_privacy", "Open camera privacy settings"),
            )
        ]
    out = []
    for c in cams:
        name = str(c.get("name") or "camera")
        shown = f"camera {name}" if name.isdigit() else name  # a camera index, not a name
        cid, title = f"camera:{name}", (f"Camera {name}" if name.isdigit() else f"Camera \u201c{name}\u201d")
        if c.get("off"):
            out.append(_r(cid, title, SKIP, "Turned off in Settings.", Fix(f"camera_on:{name}", "Turn on")))
            continue
        fps = float(c.get("fps") or 0)
        fmt = str(c.get("format") or "")
        how = fmt + (f" via {c['backend']}" if c.get("backend") else "")
        d = {"fps": round(fps, 1), "format": fmt or None, "backend": c.get("backend"), "source": c.get("source")}
        if not c.get("opened") or fps <= 0:
            msg, fix, users = _camera_fix(env, name)
            if name.isdigit():
                msg = msg.replace(f"{name} could", f"{shown.capitalize()} could", 1)
            out.append(_r(cid, title, FAIL, msg, fix, error=c.get("error") or None, held_by=users or None, **d))
        elif fps < CAMERA_MIN_FPS:
            out.append(
                _r(
                    cid,
                    title,
                    FAIL,
                    f"Too slow: {fps:.0f} pictures a second (needs {CAMERA_MIN_FPS:.0f}+). More light, another USB "
                    "port, or closing other camera apps usually helps.",
                    Fix("retry_cameras", "Try again"),
                    **d,
                )
            )
        elif fps < CAMERA_GOOD_FPS:
            out.append(
                _r(
                    cid,
                    title,
                    WARN,
                    f"Works, but slowly ({fps:.0f} fps{', ' + how if how else ''}). Gestures may feel a little choppy.",
                    Fix("retry_cameras", "Try again"),
                    **d,
                )
            )
        else:
            out.append(_r(cid, title, PASS, f"Works: {fps:.0f} fps{', ' + how if how else ''}.", **d))
    return out


# ----------------------------------------------------------------------------------------------- tracking
def check_tracking(env: Env) -> list[Result]:
    t = env.tracking()
    if not t or t.get("unavailable"):
        return [
            _r(
                "tracking",
                "Hand tracking",
                FAIL,
                str((t or {}).get("unavailable") or "Hand tracking could not start (the hand model may be missing)."),
                Fix("repair_models", "Download the hand model again"),
            )
        ]
    fps, ms = t.get("fps"), t.get("latency_ms")
    where = str(t.get("backend") or "")
    d = {"fps": fps, "latency_ms": ms, "backend": where or None, "source": t.get("source")}
    on = f" on the {where}" if where in ("processor", "graphics card") else (f" ({where})" if where else "")
    if ms is not None and float(ms) > TRACKING_SLOW_MS:
        gpu_fix = Fix("set:hands.compute=gpu", "Use the graphics card") if t.get("gpu_possible") else None
        return [
            _r(
                "tracking",
                "Hand tracking",
                WARN,
                f"Slow: {float(ms):.0f} ms per picture{on}. Try 'Light' in Models, or the graphics card.",
                gpu_fix,
                **d,
            )
        ]
    if fps is not None and 0 < float(fps) < TRACKING_MIN_FPS:
        return [
            _r(
                "tracking",
                "Hand tracking",
                FAIL,
                f"Very choppy: {float(fps):.0f} updates a second{on}. The camera or the PC is too busy.",
                Fix("retry_cameras", "Try again"),
                **d,
            )
        ]
    if fps is not None and 0 < float(fps) < TRACKING_GOOD_FPS:
        return [_r("tracking", "Hand tracking", WARN, f"A little choppy: {float(fps):.0f} updates a second{on}.", **d)]
    parts = []
    if fps:
        parts.append(f"{float(fps):.0f} updates a second")
    if ms is not None:
        parts.append(f"{float(ms):.0f} ms each")
    return [
        _r("tracking", "Hand tracking", PASS, "Smooth" + (f": {', '.join(parts)}" if parts else "") + f"{on}.", **d)
    ]


def check_gpu_tracking(env: Env) -> list[Result]:
    g = env.gpu_tracking()
    using = str(g.get("using") or "processor")
    setting = str(g.get("setting") or "auto")
    title = "Graphics card for tracking"
    d = {k: g.get(k) for k in ("setting", "using", "provider", "reason", "models_ready", "cpu_saving_pct")}
    if setting == "gpu" and using != "graphics card":
        fix = (
            Fix("download:gpu_tracking", "Download the graphics-card models")
            if g.get("provider") and not g.get("models_ready")
            else Fix("set:hands.compute=cpu", "Use the processor instead")
        )
        why = g.get("reason") or "not available"
        return [_r("gpu", title, WARN, f"Set to the graphics card, but it runs on the processor: {why}.", fix, **d)]
    if using == "graphics card":
        return [_r("gpu", title, PASS, f"Hand tracking runs on the graphics card ({g.get('provider')}).", **d)]
    if g.get("provider") and not g.get("models_ready") and setting != "cpu":
        return [
            _r(
                "gpu",
                title,
                WARN,
                "Your graphics card could take hand tracking off the processor (a one-time ~23 MB download).",
                Fix("download:gpu_tracking", "Download and use it"),
                **d,
            )
        ]
    return [_r("gpu", title, PASS, f"Runs on the processor ({g.get('reason') or 'your choice'}).", **d)]


# ------------------------------------------------------------------------------------------------- cursor
def check_glide(env: Env) -> list[Result]:
    g = env.glide()
    hz, want = float(g.get("hz") or 0), float(g.get("target_hz") or 120)
    d = {"hz": round(hz, 1), "target_hz": want, "source": g.get("source")}
    if hz <= 0:
        return [
            _r(
                "glide",
                "Smooth pointer",
                FAIL,
                "The pointer output did not run.",
                Fix("restart_app", "Restart Project Away"),
                **d,
            )
        ]
    if hz < GLIDE_MIN_HZ:
        return [
            _r(
                "glide",
                "Smooth pointer",
                WARN,
                f"The pointer moves {hz:.0f} times a second (aim: {want:.0f}). It may look a little jumpy; a busy PC "
                "or power saving causes this.",
                **d,
            )
        ]
    status = PASS if hz >= want * GLIDE_GOOD_RATIO else WARN
    return [_r("glide", "Smooth pointer", status, f"The pointer moves {hz:.0f} times a second (aim: {want:.0f}).", **d)]


def check_big_cursor(env: Env) -> list[Result]:
    b = env.big_cursor()
    if not b.get("supported"):
        return [_r("big_cursor", "Big pointer", SKIP, str(b.get("note") or "Not used on this system."))]
    d = {k: b.get(k) for k in ("size", "user_size", "restored", "stuck")}
    fix = Fix("reset_cursor", "Put my pointer size back")
    if b.get("stuck"):
        msg = f"The pointer is stuck at the big size ({b.get('size')} px; yours is {b.get('user_size')} px)."
        return [_r("big_cursor", "Big pointer", FAIL, msg, fix, **d)]
    if b.get("restored") is False:
        return [
            _r("big_cursor", "Big pointer", FAIL, "The pointer grew but did not shrink back to your size.", fix, **d)
        ]
    return [_r("big_cursor", "Big pointer", PASS, "Grows while gestures are on and shrinks back to your size.", **d)]


def check_instances(env: Env) -> list[Result]:
    i = env.instances()
    count = int(i.get("count") or 0)
    d = {"count": count, "lock_held": i.get("lock_held"), "pids": i.get("pids")}
    out = []
    if i.get("lock_held") is False and count > 0:
        msg = "Project Away is running without its one-copy lock, so a second copy could start."
        out.append(_r("single", "Only one Project Away", WARN, msg, Fix("restart_app", "Restart Project Away"), **d))
    else:
        out.append(_r("single", "Only one Project Away", PASS, "Only one copy can run at a time.", **d))
    if count > 1:
        msg = f"{count} copies of Project Away are running. They fight over the camera and the pointer."
        out.append(
            _r("duplicates", "Extra copies running", FAIL, msg, Fix("close_duplicates", "Close the extra copies"), **d)
        )
    else:
        out.append(_r("duplicates", "Extra copies running", PASS, "None.", **d))
    return out


# -------------------------------------------------------------------------------------------- mic + speech
def check_mic(env: Env) -> list[Result]:
    m = env.mic()
    sound = Fix("open:sound_settings", "Open sound settings")
    if m.get("error"):
        return [
            _r(
                "mic",
                "Microphone",
                FAIL,
                f"The microphone could not be opened: {m['error']}",
                sound,
                device=m.get("device"),
            )
        ]
    rms, peak = m.get("rms_db"), m.get("peak_db")
    d = {"device": m.get("device"), "rms_db": rms, "peak_db": peak}
    dev = f" ({m['device']})" if m.get("device") else ""
    if rms is None or float(rms) < MIC_SILENT_DB:
        msg = f"Nothing reaches the microphone{dev}. Is it muted, or is another microphone the default?"
        return [_r("mic", "Microphone", FAIL, msg, sound, **d)]
    if peak is not None and float(peak) > MIC_LOUD_DB:
        return [
            _r(
                "mic",
                "Microphone",
                WARN,
                "Very loud: words may get cut off. Turn the input level down a little.",
                sound,
                **d,
            )
        ]
    return [_r("mic", "Microphone", PASS, f"Hears sound{dev or ' (default microphone)'}.", **d)]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def match(phrase: str, heard: str) -> float:
    want, got = words(phrase), set(words(heard))
    return sum(1 for w in want if w in got) / len(want) if want else 0.0


def check_stt(env: Env) -> list[Result]:
    s = env.stt(PHRASE)
    if s.get("skipped"):
        return [_r("stt", "Speech to text", SKIP, str(s["skipped"]))]
    if s.get("error"):
        msg = f"Speech recognition did not run: {s['error']}"
        return [_r("stt", "Speech to text", FAIL, msg, Fix("repair_models", "Download the speech model again"))]
    heard = str(s.get("heard") or "")
    score = match(PHRASE, heard)
    how = {"loopback": "spoken by the computer", "spoken": "said by you"}.get(str(s.get("method")), "")
    d = {"heard": heard, "expected": PHRASE, "match": round(score, 2), "method": s.get("method"), "ms": s.get("ms")}
    if score >= STT_MATCH:
        secs = float(s.get("ms") or 0) / 1000
        return [_r("stt", "Speech to text", PASS, f"Understood the test phrase ({how}) in {secs:.1f} s.", **d)]
    if heard:
        msg = f"Heard \u201c{heard[:60]}\u201d instead of \u201c{PHRASE}\u201d. Speak closer to the microphone."
        return [_r("stt", "Speech to text", WARN, msg, **d)]
    msg = "Heard nothing back. Run the self-test at the PC and say the phrase when asked."
    return [_r("stt", "Speech to text", WARN, msg, Fix("open:sound_settings", "Open sound settings"), **d)]


def check_tts(env: Env) -> list[Result]:
    engines = env.tts()
    if not engines:
        return [
            _r(
                "tts",
                "Speaking voice",
                FAIL,
                "No voice found to speak with.",
                Fix("open:settings/voice", "Choose a voice"),
            )
        ]
    out = []
    for e in engines:
        name = str(e.get("name") or "voice")
        ms = e.get("first_audio_ms")
        d = {"first_audio_ms": ms, "engine": name, "measured": e.get("measured")}
        tid, title = f"tts:{name}", f"Voice \u201c{name}\u201d"
        if e.get("error"):
            fix = Fix("open:settings/voice", "Choose another voice")
            out.append(_r(tid, title, FAIL, f"Could not speak: {e['error']}", fix, **d))
        elif ms is not None and float(ms) > TTS_SLOW_MS:
            out.append(_r(tid, title, WARN, f"Speaks, but starts slowly ({float(ms) / 1000:.1f} s).", **d))
        else:
            when = f" (first sound after {float(ms):.0f} ms)" if ms is not None else ""
            out.append(_r(tid, title, PASS, f"Speaks{when}.", **d))
    return out


def check_notification(env: Env) -> list[Result]:
    n = env.notification()
    if n.get("skipped"):
        return [_r("notify", "Pop-up messages", SKIP, str(n["skipped"]))]
    if n.get("shown"):
        via = f" ({n['via']})" if n.get("via") else ""
        return [_r("notify", "Pop-up messages", PASS, f"A test message popped up{via}.", via=n.get("via"))]
    msg = f"Messages could not pop up: {n.get('reason') or 'unknown reason'}. Check Windows Focus / Do not disturb."
    fix = Fix("open:notification_settings", "Open notification settings")
    return [_r("notify", "Pop-up messages", FAIL, msg, fix, reason=n.get("reason"))]


# -------------------------------------------------------------------------------------------------- link
LINKS = (
    ("tunnel", "Secure link (Cloudflare)", Fix("new_link", "Get a new link")),
    ("backup", "Backup link", Fix("backup_link", "Restart the backup link")),
    ("named", "Your own Cloudflare tunnel", Fix("restart_link", "Restart it")),
)


def check_link(env: Env) -> list[Result]:
    k = env.link()
    if not k.get("enabled", True):
        return [_r("link", "Agent Link", SKIP, "Agent Link is turned off.")]
    out = []
    srv = k.get("server") or {}
    if srv.get("running"):
        ms, ok = srv.get("ms"), bool(srv.get("ok", True))
        msg = (
            "Answers on this PC" + (f" ({float(ms):.0f} ms)." if ms is not None else ".") if ok else "Does not answer."
        )
        out.append(
            _r(
                "link",
                "Agent Link",
                PASS if ok else FAIL,
                msg,
                None if ok else Fix("restart_link", "Restart Agent Link"),
                port=srv.get("port"),
                ms=ms,
            )
        )
    else:
        out.append(_r("link", "Agent Link", PASS, "Ready (it starts when an agent connects)."))
    for key, title, fix in LINKS:
        t = k.get(key)
        if not t:
            continue
        ms = t.get("ms")
        if t.get("ok"):
            st = WARN if ms is not None and float(ms) > LINK_SLOW_MS else PASS
            msg = f"Reachable from the internet ({float(ms):.0f} ms)." if ms is not None else "Reachable."
            out.append(_r(f"link:{key}", title, st, msg, ms=ms))
        else:
            msg = f"Not reachable from the internet: {t.get('error') or 'no answer'}."
            out.append(_r(f"link:{key}", title, FAIL, msg, fix, ms=ms, error=t.get("error")))
    for r in k.get("relays") or []:
        name = str(r.get("name"))
        if r.get("ok"):
            out.append(
                _r(
                    f"relay:{name}",
                    f"Can reach {name}",
                    PASS,
                    f"Yes ({float(r.get('ms') or 0):.0f} ms).",
                    ms=r.get("ms"),
                )
            )
        else:
            msg = f"No ({r.get('error') or 'blocked'}). {r.get('why') or ''}".strip()
            out.append(_r(f"relay:{name}", f"Can reach {name}", WARN, msg, error=r.get("error")))
    a = k.get("agents")
    if a:
        n, back = int(a.get("paired") or 0), int(a.get("reconnectable") or 0)
        msg = f"{n} agent{'s' if n != 1 else ''} paired; {back} can reconnect by themselves after a restart."
        out.append(_r("link:agents", "Paired agents", PASS, msg, paired=n, reconnectable=back))
    return out


# -------------------------------------------------------------------------------------------- the PC itself
def check_load(env: Env) -> list[Result]:
    ld = env.load()
    cpu, gpu = ld.get("cpu_pct"), ld.get("gpu_pct")
    top = list(ld.get("top") or [])
    names = ", ".join(f"{p.get('name')} {float(p.get('cpu_pct') or 0):.0f}%" for p in top[:3])
    d = {
        "cpu_pct": cpu,
        "gpu_pct": gpu,
        "gpu": ld.get("gpu_name"),
        "top": top[:5],
        "vram_used_mb": ld.get("vram_used_mb"),
    }
    out = []
    if cpu is not None and float(cpu) >= CPU_BUSY_PCT:
        first = next((p for p in top if p.get("name") != "Project Away"), None)
        fix = None
        if first and first.get("pid"):
            fix = Fix(f"close_pid:{first['pid']}", f"Close {first.get('name')}")
        out.append(_r("cpu", "Processor load", WARN, f"Very busy ({float(cpu):.0f}%): {names}.", fix, **d))
    else:
        msg = f"{float(cpu or 0):.0f}% busy" + (f" (top: {names})." if names else ".")
        out.append(_r("cpu", "Processor load", PASS, msg, **d))
    if gpu is not None:
        st = WARN if float(gpu) >= GPU_BUSY_PCT else PASS
        out.append(
            _r(
                "gpu_load",
                "Graphics card load",
                st,
                f"{float(gpu):.0f}% busy ({ld.get('gpu_name') or 'graphics card'}).",
                **d,
            )
        )
    return out


def check_models(env: Env) -> list[Result]:
    m = env.models()
    missing = list(m.get("missing") or [])
    rows = list(m.get("rows") or [])
    d = {"rows": rows, "missing": missing}
    if missing:
        msg = f"Missing: {', '.join(missing)}. The features that need them are off until they are downloaded."
        return [_r("models", "Models", FAIL, msg, Fix("repair_models", "Download what's missing"), **d)]
    if m.get("update"):
        return [_r("models", "Models", WARN, str(m["update"]), Fix("open:settings/models", "Open Models"), **d)]
    using = ", ".join(f"{r.get('title')}: {r.get('current_title')}" for r in rows[:6] if r.get("on", True))
    return [_r("models", "Models", PASS, "All present" + (f" ({using})." if using else "."), **d)]


CHECKS: tuple[tuple[str, str, Callable[[Env], list[Result]]], ...] = (
    ("cameras", "Cameras", check_cameras),
    ("tracking", "Hand tracking", check_tracking),
    ("gpu", "Graphics card", check_gpu_tracking),
    ("glide", "Smooth pointer", check_glide),
    ("big_cursor", "Big pointer", check_big_cursor),
    ("instances", "One copy running", check_instances),
    ("mic", "Microphone", check_mic),
    ("stt", "Speech to text", check_stt),
    ("tts", "Speaking voices", check_tts),
    ("notify", "Pop-up messages", check_notification),
    ("link", "Agent Link", check_link),
    ("load", "PC load", check_load),
    ("models", "Models", check_models),
)
CHECK_IDS = tuple(c[0] for c in CHECKS)
