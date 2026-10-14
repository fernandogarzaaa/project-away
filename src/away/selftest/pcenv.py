"""Check my PC on the real PC (v0.1.24): the measurements behind :mod:`away.selftest.pccheck`.

:class:`RealPcEnv` extends the self-test's :class:`~away.selftest.env.RealEnv` (cameras, tracking, glide, big
pointer, one-copy lock, microphone, link: unchanged) with the guided and desktop parts. When Project Away is running,
the check window uses :class:`RemotePcEnv`: every measurement runs inside the app (which holds the cameras, the tray
icon, the message windows and Agent Link) through the ``pccheck_probe`` request, and only the instructions are shown
by the window itself. Yes / No questions then appear as the app's own prompt, answerable with a click or a thumbs
up / down.
"""

from __future__ import annotations

import contextlib
import math
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from away.selftest.env import RealEnv, http_ms

ECHO_SAMPLES = 5
# What the app may run on request (a fixed list: the request comes from another process on this PC).
PROBES = (
    "cameras",
    "camera_users",
    "tracking",
    "gpu_tracking",
    "glide",
    "big_cursor",
    "instances",
    "mic",
    "notification",
    "link",
    "confirm",
    "wait_gesture",
    "dictation",
    "wake_settings",
    "wake_word",
    "voice_list",
    "speak",
    "tray",
    "toast",
    "secure_desktop",
    "input_echo",
    "link_reconnect",
)
PROBE_TIMEOUT_S = {"cameras": 60.0, "wait_gesture": 30.0, "confirm": 75.0, "speak": 60.0, "input_echo": 30.0}


class RealPcEnv(RealEnv):
    def __init__(self, *a: Any, confirm: Callable[[str], bool | None] | None = None, **k: Any) -> None:
        super().__init__(*a, **k)
        self.confirm_fn = confirm

    # ---------------------------------------------------------------- talking to the person
    def prompt(self, text: str) -> None:
        if self.ask is not None:
            self.ask(text)

    def _prompts(self) -> Any:
        h = self.host or getattr(self.app, "agent_link", None)
        return getattr(getattr(h, "core", None), "prompts", None)

    def confirm(self, question: str) -> bool | None:
        if self.confirm_fn is not None:
            return self.confirm_fn(question)
        pr = self._prompts()
        if pr is None:
            raise RuntimeError("no prompt window to ask with")
        res = pr.ask(question, ["Yes", "No"], title="Check my PC", agent="Project Away", timeout_s=60.0)
        ans = str(res.get("answer") or "").lower()
        return True if ans == "yes" else (False if ans == "no" else None)

    # ---------------------------------------------------------------- gestures
    def wait_gesture(self, kind: str, seconds: float) -> dict[str, Any]:
        if self.app is not None and getattr(self.app, "pipeline", None) is not None:
            return self._gesture_live(kind, seconds)
        return self._gesture_camera(kind, seconds)

    def _gesture_live(self, kind: str, seconds: float) -> dict[str, Any]:
        p = self.app.pipeline
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            hand = bool(getattr(p.gestures, "hand_present", False))
            st: dict[str, Any] = {}
            with contextlib.suppress(Exception):
                st = p.cursor_state()
            seen = (
                hand if kind == "hand" else hand and (st.get("state") == "pinch" or float(st.get("pinch") or 0) >= 0.99)
            )
            if seen:
                tr = self.tracking()
                return {
                    "seen": True,
                    "after_ms": round((time.monotonic() - t0) * 1000.0),
                    "fps": tr.get("fps"),
                    "latency_ms": tr.get("latency_ms"),
                    "backend": tr.get("backend"),
                }
            time.sleep(0.05)
        return {"seen": False}

    def _gesture_camera(self, kind: str, seconds: float) -> dict[str, Any]:  # pragma: no cover - real camera
        model = self._models_dir() / "hand_landmarker.task"
        if not model.exists():
            return {"skipped": "The hand model is not downloaded yet."}
        from away.hands.tracker import HandTracker
        from away.multicam.capture import open_video

        spec = self._specs()[0]
        src: Any = spec.device
        with contextlib.suppress(Exception):
            src = int(src)
        cap = open_video(src)
        if cap is None or not cap.isOpened():
            return {"error": f"camera {spec.device} did not open"}
        tr = HandTracker(model, 1)
        lat: list[float] = []
        try:
            t0 = time.monotonic()
            while time.monotonic() - t0 < seconds:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rgb = frame[:, :, ::-1].copy()
                a = time.perf_counter()
                hf = tr.process(rgb, time.monotonic() - t0)
                lat.append((time.perf_counter() - a) * 1000.0)
                lm = hf.landmarks
                if lm is None:
                    continue
                if kind == "hand" or pinch_closed(lm):
                    el = time.monotonic() - t0
                    return {
                        "seen": True,
                        "after_ms": round(el * 1000.0),
                        "fps": round(len(lat) / max(el, 1e-3), 1),
                        "latency_ms": round(sorted(lat)[len(lat) // 2], 1),
                        "backend": "processor",
                        "camera": str(spec.device),
                    }
            return {"seen": False, "fps": round(len(lat) / max(seconds, 1e-3), 1)}
        finally:
            with contextlib.suppress(Exception):
                tr.close()
            with contextlib.suppress(Exception):
                cap.release()

    # ---------------------------------------------------------------- speech
    def dictation(self, phrase: str) -> dict[str, Any]:
        try:
            asr = self._asr()
        except Exception as exc:
            return {"error": str(exc)[:200]}
        if asr is None:
            return {"skipped": "No speech model is downloaded yet, so dictation was not tested."}
        t0 = time.perf_counter()
        heard = str(asr.transcribe(self._record(4.0)) or "")
        return {"heard": heard.strip(), "method": "spoken", "ms": round((time.perf_counter() - t0) * 1000.0)}

    def wake_settings(self) -> dict[str, Any]:
        v = getattr(self.cfg, "voice", None)
        return {
            "word": str(getattr(v, "wake_word", "away") or "away"),
            "enabled": bool(getattr(v, "wake_word_enabled", False)),
        }

    def wake_word(self, seconds: float) -> dict[str, Any]:
        from away.voice.always_on import match_wake

        ws = self.wake_settings()
        got = self.dictation(ws["word"])
        if got.get("skipped") or got.get("error"):
            return {**ws, **got}
        heard = str(got.get("heard") or "")
        return {**ws, "heard": heard, "matched": match_wake(heard, ws["word"])[0]}

    def voice_list(self) -> list[str]:
        return [n for n, _m in self.tts_factories()]

    def speak(self, name: str) -> dict[str, Any]:
        for n, make in self.tts_factories():
            if n == name:
                return self.speak_one(n, make, "Test. This is Project Away.")
        return {"name": name, "error": "this voice is no longer available"}

    # ---------------------------------------------------------------- windows on screen
    def tray(self) -> dict[str, Any]:
        icon = getattr(self.app, "tray_icon", None)
        if icon is None:
            return {"skipped": "Project Away is not running, so there is no tray icon to check."}
        return {"visible": bool(getattr(icon, "visible", False)), "tooltip": str(getattr(icon, "title", "") or "")}

    def toast(self) -> dict[str, Any]:
        pr = self._prompts()
        ui = getattr(pr, "ui", None)
        if ui is None:
            return {"skipped": "Project Away is not running, so no message window was shown."}
        res = ui.toast("Check my PC: this is a Project Away message window.", "info", 6.0, "Check my PC")
        res = res if isinstance(res, dict) else {"shown": True}
        return {"shown": bool(res.get("shown", True)), "via": res.get("via"), "reason": res.get("reason")}

    # ---------------------------------------------------------------- safety
    def secure_desktop(self) -> dict[str, Any]:
        if sys.platform != "win32":
            return {"skipped": "Windows only (other systems have no secure desktop)."}
        from away.agentlink import desktop

        try:
            return {"active": bool(desktop.secure_desktop_active())}
        except Exception as exc:
            return {"error": str(exc)[:160] or type(exc).__name__}

    def input_echo(self) -> dict[str, Any]:  # pragma: no cover - real input devices
        from away.agentlink.control import KEY_ECHO_S, MOUSE_ECHO_S

        try:
            from pynput import keyboard, mouse
        except Exception as exc:
            return {"skipped": f"pynput is not available ({type(exc).__name__})."}
        return measure_echo(mouse, keyboard, MOUSE_ECHO_S, KEY_ECHO_S)

    def link_reconnect(self) -> dict[str, Any]:
        h = self.host or getattr(self.app, "agent_link", None)
        if h is None:
            return {"skipped": "Project Away is not running, so Agent Link was not tested."}
        url = ""
        for attr in ("tunnel", "relay", "named"):
            t = getattr(h, attr, None)
            if t is not None and getattr(t, "url", None):
                url = str(t.url)
                break
        srv = getattr(h, "server", None)
        if not url and srv is not None and getattr(srv, "running", False):
            url = f"http://127.0.0.1:{int(srv.port)}"
        if not url:
            return {"skipped": "Agent Link is not running (it starts when an agent connects)."}
        ok, worst, err = 0, 0.0, ""
        for _ in range(3):  # three brand-new connections, like an agent coming back after a drop
            try:
                worst = max(worst, http_ms(f"{url}/v1/ping"))
                ok += 1
            except Exception as exc:
                err = str(exc)[:120]
        back = None
        with contextlib.suppress(Exception):
            back = len(h.core.registry.reconnectable())
        return {"tries": 3, "ok": ok, "ms": round(worst, 1), "error": err or None, "reconnectable": back}


def pinch_closed(lm: Any) -> bool:
    """Thumb tip (4) and index tip (8) closer than a third of the palm length (wrist 0 to middle base 9)."""
    d = math.dist(lm[4][:2], lm[8][:2])
    size = math.dist(lm[0][:2], lm[9][:2])
    return size > 0 and d < 0.35 * size


def engine_input() -> Any:  # pragma: no cover - real input devices
    """The echo probe's input: a client of the one action engine (``user`` source: the person asked for the check),
    so the 1 px move and the Shift tap are audited, obey Stop everything and never bypass the engine."""
    from away.actions_engine import USER, get_engine

    eng = get_engine()
    return eng.client(USER, eng.backend if eng.backend is not None else eng.os_backend())


def measure_echo(
    mouse: Any,
    keyboard: Any,
    mouse_window: float,
    key_window: float,
    samples: int = ECHO_SAMPLES,
    wait: float = 1.0,
    inject: Any = None,
) -> dict[str, Any]:
    """Move the pointer by one pixel and back, and tap Shift, ``samples`` times each; time how long each takes to
    reach a pynput listener (the same path the takeover detection uses). ``mouse`` / ``keyboard`` (pynput) are
    only listened to; the input itself goes through ``inject`` (default: the action engine, :func:`engine_input`),
    which needs ``position``, ``move(x, y)`` and ``key(name)``."""
    got: dict[str, threading.Event] = {"mouse": threading.Event(), "key": threading.Event()}
    at: dict[str, float] = {}

    def seen(kind: str) -> None:
        if not got[kind].is_set():
            at[kind] = time.perf_counter()
            got[kind].set()

    ml = mouse.Listener(on_move=lambda *_a: seen("mouse"))
    kl = keyboard.Listener(on_press=lambda *_a: seen("key"))
    ml.start()
    kl.start()
    out: dict[str, Any] = {"mouse_ms": [], "key_ms": [], "mouse_missed": 0, "key_missed": 0}
    try:
        time.sleep(0.3)
        src = inject if inject is not None else engine_input()
        for i in range(samples):
            for kind in ("mouse", "key"):
                got[kind].clear()
                t0 = time.perf_counter()
                if kind == "mouse":
                    x, y = src.position
                    src.move(int(x) + (1 if i % 2 == 0 else -1), int(y))
                else:
                    src.key("shift")
                if got[kind].wait(wait):
                    out[f"{kind}_ms"].append(round((at[kind] - t0) * 1000.0, 1))
                else:
                    out[f"{kind}_missed"] += 1
                time.sleep(0.15)
    finally:
        with contextlib.suppress(Exception):
            ml.stop()
        with contextlib.suppress(Exception):
            kl.stop()
    out["mouse_window_s"], out["key_window_s"] = mouse_window, key_window
    return out


def probe(app: Any, host: Any, name: str, args: list[Any] | None = None) -> dict[str, Any]:
    """Run one measurement inside the app (the ``pccheck_probe`` request). Only names in :data:`PROBES`."""
    if name not in PROBES:
        raise ValueError(f"unknown check step {name!r}")
    env = RealPcEnv(app=app, host=host)
    res = getattr(env, name)(*(args or []))
    return res if isinstance(res, dict) else {"value": res}


class RemotePcEnv:
    """The check window's environment while Project Away runs: instructions here, measurements in the app."""

    def __init__(
        self,
        ask_host: Callable[..., dict[str, Any] | None],
        prompt: Callable[[str], Any] | None = None,
        confirm: Callable[[str], bool | None] | None = None,
    ) -> None:
        self.ask_host = ask_host  # ask_host(action, timeout=..., **data) -> result dict or None
        self._prompt = prompt
        self._confirm = confirm  # used when the app has no prompt window to ask with

    def confirm(self, question: str) -> bool | None:
        try:
            ans = self._call("confirm", question)  # the app's prompt: a click or thumbs up / down
        except Exception:
            return self._confirm(question) if self._confirm is not None else None
        return ans if isinstance(ans, bool) else None

    def prompt(self, text: str) -> None:
        if self._prompt is not None:
            self._prompt(text)

    def _call(self, name: str, *args: Any) -> Any:
        res = self.ask_host("pccheck_probe", timeout=PROBE_TIMEOUT_S.get(name, 20.0), probe=name, args=list(args))
        if res is None:
            raise RuntimeError("Project Away did not answer")
        return res["value"] if set(res) == {"value"} else res

    def __getattr__(self, name: str) -> Any:
        if name in PROBES and not name.startswith("_"):
            return lambda *a: self._call(name, *a)
        raise AttributeError(name)

    def stt(self, phrase: str) -> dict[str, Any]:  # the self-test's speech check (unused by Check my PC)
        return dict(self._call("dictation", phrase))

    def tts(self) -> list[dict[str, Any]]:
        return [self._call("speak", n) for n in self._call("voice_list")]
