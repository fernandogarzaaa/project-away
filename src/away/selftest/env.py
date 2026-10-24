"""The self-test on the real PC (v0.1.24): the measurements behind each check in :mod:`away.selftest.checks`.

:class:`RealEnv` works in two places:

* **inside the running app** (tray "Run self-test", Settings, agents, ``away selftest`` while the app runs): it reads
  what the app already measures (live camera rate and format, hand-tracking rate and latency, the pointer output
  thread, the big pointer, Agent Link) instead of grabbing the cameras away from it;
* **on its own** (``away selftest`` with the app closed): it opens each camera for a few seconds, times the hand
  model on a test picture and runs the pointer output loop for a second.

Every method returns plain dicts and may raise; the check turns an exception into a plain "could not run" line.
Speaking, listening and the pop-up really happen (that is the point of a self-test); each takes a few seconds.
"""

from __future__ import annotations

import contextlib
import math
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

APP_NAMES = ("projectaway", "project away", "away.exe", "away-tray")
# command lines of Away's own short-lived helpers and windows (Settings, Check my PC): not "extra copies"
HELPER_WORDS = (
    "agent-ui",
    "selftest",
    " mcp",
    "cameras ",
    "ui_helper",
    "--helper",
    "update-install",
    " settings",
    " check",
)
RELAYS = (
    ("Cloudflare", "region1.v2.argotunnel.com", 7844, "The standard link may not start; try the backup link."),
    ("localhost.run", "localhost.run", 22, "The backup link (localhost.run) may not start; try Pinggy."),
    ("Pinggy", "a.pinggy.io", 443, "The backup link (Pinggy) may not start; try localhost.run."),
    ("ntfy.sh", "ntfy.sh", 443, "Agents may not find the new address after a restart (rendezvous)."),
)


def tcp_ms(host: str, port: int, timeout: float = 4.0) -> float:
    t0 = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout):
        pass
    return (time.perf_counter() - t0) * 1000.0


def http_ms(url: str, timeout: float = 8.0) -> float:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "ProjectAway-SelfTest"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read(2000)
        if int(resp.status) >= 400:
            raise RuntimeError(f"HTTP {resp.status}")
    return (time.perf_counter() - t0) * 1000.0


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-9))


def _start_menu_host(fg: dict[str, Any]) -> bool:
    """True when the foreground window is the Start menu (proposal #4: the Win keypress worked).

    Windows 11 hosts Start in StartMenuExperienceHost.exe; Windows 10 hosts it inside explorer.exe, where the
    taskbar (also explorer.exe) is told apart by the window class.
    """
    proc = str(fg.get("process") or "").lower()
    if proc == "startmenuexperiencehost.exe":
        return True
    return (
        proc == "explorer.exe"
        and str(fg.get("class") or "") == "Windows.UI.Core.CoreWindow"
        and str(fg.get("title") or "").lower() == "start"
    )


class RealEnv:
    def __init__(
        self,
        app: Any = None,
        host: Any = None,
        cfg: Any = None,
        data_dir: Path | None = None,
        interactive: bool = False,
        ask: Callable[[str], Any] | None = None,
        speak_test: bool = True,
    ) -> None:
        from away import config as cfgmod
        from away import paths

        self.app = app
        self.host = host
        if cfg is None:
            cfg = getattr(app, "cfg", None)
        if cfg is None:
            p = cfgmod.config_path()
            cfg = cfgmod.load(p)[0] if p.exists() else cfgmod.AppConfig()
        self.cfg = cfg
        self.data_dir = Path(data_dir or paths.data_dir())
        self.interactive = interactive
        self.ask = ask  # ask("Now say: ...") before recording the person's own voice (window / CLI)
        self.speak_test = speak_test

    # ----------------------------------------------------------------- helpers
    def _telemetry(self) -> Any:
        h = self.host or getattr(self.app, "agent_link", None)
        return getattr(getattr(h, "core", None), "telemetry", None)

    def _models_dir(self) -> Path:
        md = getattr(self.app, "models_dir", None)
        return Path(md) if md else self.data_dir / "models"

    # ----------------------------------------------------------------- cameras
    def _specs(self) -> list[Any]:
        from away.config import CameraSpec

        h = self.cfg.hands
        if h.cameras:
            return list(h.cameras)
        return [CameraSpec(device=h.camera_index, role="primary", fourcc=h.camera_fourcc)]

    def cameras(self) -> list[dict[str, Any]]:
        off = [{"name": str(c.device), "off": True} for c in getattr(self.cfg.hands, "disabled_cameras", []) or []]
        tel = self._telemetry()
        if self.app is not None and tel is not None:
            fps = tel.fps()
            out = []
            modes = self._live_modes()
            for spec in self._specs():
                name = str(spec.device)
                rate = fps.get(name, fps.get(str(getattr(spec, "role", "")), 0.0))
                if not rate and len(fps) == 1 and len(self._specs()) == 1:
                    rate = next(iter(fps.values()))
                m = modes.get(name, {})
                out.append(
                    {
                        "name": name,
                        "opened": rate > 0,
                        "fps": rate,
                        "format": m.get("mode") or "",
                        "backend": m.get("backend") or "",
                        "error": m.get("error") or (tel.fps_note(fps) if not rate else ""),
                        "source": "live",
                    }
                )
            return out + off
        from away.multicam.camtest import run_camera_test

        rep = run_camera_test(self._specs(), seconds=4.0, min_fps=float(self.cfg.hands.min_camera_fps))
        res = rep.phases[0].results if rep.phases else []
        return [
            {
                "name": r.opened or r.device,
                "opened": r.frames > 0 and not r.error,
                "fps": r.fps,
                "format": r.mode,
                "backend": r.backend,
                "error": r.error,
                "source": "opened for the test",
            }
            for r in res
        ] + off

    def _live_modes(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        with contextlib.suppress(Exception):
            from away.multicam import capture

            for name, (backend, m) in dict(getattr(capture, "_session_modes", {}) or {}).items():
                desc = getattr(m, "describe", None)
                out[str(name)] = {"mode": str(desc() if callable(desc) else m), "backend": str(backend)}
        return out

    def camera_users(self, camera: str) -> dict[str, Any]:
        from away.multicam import camera_users as cu

        return cu.explain(camera, cu.current_users(), cu.broadcast_source())

    # ----------------------------------------------------------------- tracking
    def gpu_tracking(self) -> dict[str, Any]:
        from away.accel import compute

        return compute.status(self.cfg, self.data_dir)

    def tracking(self) -> dict[str, Any]:
        tel = self._telemetry()
        g: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            g = self.gpu_tracking()
        where = str(g.get("using") or "processor")
        possible = bool(g.get("provider"))
        if self.app is not None:
            times = list(getattr(self.app, "_frame_times", []) or [])
            now = time.monotonic()
            recent = [t for t in times if now - t <= 3.0]
            fps = round(len(recent) / 3.0, 1) if len(recent) >= 2 else 0.0
            lat = tel.latency() if tel is not None else {}
            return {
                "fps": fps,
                "latency_ms": lat.get("p50_ms"),
                "backend": where,
                "source": "live",
                "gpu_possible": possible,
            }
        return self._bench_tracking(where, possible)

    def _bench_tracking(self, where: str, possible: bool) -> dict[str, Any]:
        import numpy as np

        model = self._models_dir() / "hand_landmarker.task"
        if not model.exists():
            return {"unavailable": "The hand model is not downloaded yet (run setup, or Models > Download)."}
        from away.accel import compute
        from away.hands.tracker import HandTracker

        factory = compute.hand_factory(self.cfg, model, data_dir=self.data_dir)
        tr = HandTracker(model, 1, factory=factory[0], image_fn=factory[1]) if factory else HandTracker(model, 1)
        try:
            img = np.zeros((480, 640, 3), np.uint8)
            img[:, :, 1] = 90
            tr.process(img, 0.0)  # warm up (model load / first run)
            n, t0 = 20, time.perf_counter()
            for i in range(n):
                tr.process(img, (i + 1) / 30.0)
            ms = (time.perf_counter() - t0) * 1000.0 / n
        finally:
            with contextlib.suppress(Exception):
                tr.close()
        return {
            "fps": None,
            "latency_ms": round(ms, 1),
            "backend": where if factory else "processor",
            "source": "timed on a test picture",
            "gpu_possible": possible,
        }

    # ------------------------------------------------------------------ pointer
    def glide(self) -> dict[str, Any]:
        want = float(getattr(self.cfg.hands.cursor, "output_hz", 120) or 120)
        from away.pointer.glide import Glide, GlideRunner

        moves: list[float] = []
        g = Glide()
        runner = GlideRunner(g, lambda _x, _y: moves.append(time.monotonic()), hz=want)
        t_end = time.monotonic() + 1.2
        runner.start()
        k = 0
        while time.monotonic() < t_end:  # a hand moving at 30 fps
            g.on_target(time.monotonic(), 100.0 + k, 100.0)
            k += 3
            time.sleep(1 / 30)
        runner.stop()
        n = len([t for t in moves if t >= t_end - 1.0])
        return {"hz": float(n), "target_hz": want, "source": "pointer output loop timed for 1 s (no pointer moved)"}

    def big_cursor(self) -> dict[str, Any]:
        if not sys.platform.startswith("win"):
            return {"supported": False, "note": "Windows only (other systems show a bigger ring instead)."}
        from away.app import bigcursor as bc

        ctl = getattr(self.app, "big_cursor", None)
        if ctl is None:
            ctl = bc.BigCursor(bc.default_backend())
        size = ctl.backend.get()
        user = ctl.user_size()
        active = bool(getattr(ctl, "enlarged", False))
        stuck = bool(getattr(ctl, "stuck", False)) or bool(
            size and user and size > user and size in bc.our_sizes() and not active
        )
        if stuck:
            return {"supported": True, "size": size, "user_size": user, "stuck": True}
        if active:
            return {"supported": True, "size": size, "user_size": user, "restored": None, "note": "big right now"}
        big = bc.target_size(max(1, int(getattr(self.cfg.hands.cursor, "big_cursor_size", 0) or 2)))
        ok = ctl.backend.set(big)
        time.sleep(0.2)
        ctl.backend.set(user)
        time.sleep(0.2)
        after = ctl.backend.get()
        return {"supported": True, "size": after, "user_size": user, "restored": bool(ok) and after == user}

    def instances(self) -> dict[str, Any]:
        from away.app import instance

        held = instance.owned("app") if self.app is not None else None
        pids = app_processes()
        if self.app is None and not pids and instance.acquire("app"):
            instance.release("app")  # nobody holds the one-copy lock: no copy is running
        return {"lock_held": held, "count": len(pids), "pids": pids}

    # ---------------------------------------------------------------- sound
    def _record(self, seconds: float, sr: int = 16000) -> Any:
        import numpy as np
        import sounddevice as sd

        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def mic(self) -> dict[str, Any]:
        import numpy as np

        try:
            import sounddevice as sd

            dev = sd.query_devices(kind="input")
            name = str(dev.get("name", "")) if isinstance(dev, dict) else ""
        except Exception as exc:
            return {"error": f"no microphone found ({exc})"}
        try:
            a = self._record(1.5)
        except Exception as exc:
            return {"error": str(exc)[:200], "device": name}
        rms = float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        return {"device": name, "rms_db": round(db(rms), 1), "peak_db": round(db(peak), 1)}

    def _asr(self) -> Any:
        for path in ("voice.dictation.asr", "dictation.asr", "voice.ambient.dictation.asr", "asr"):
            obj: Any = self.app
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "transcribe"):
                return obj
        from away.voice.asr import make_asr

        md = self._models_dir()
        for model in ("tiny", "base", "small"):
            if (md / f"whisper-{model}" / "model.bin").exists():
                return make_asr("faster-whisper", model, "cpu", "int8", "en", str(md / f"whisper-{model}"))
        return None

    def stt(self, phrase: str) -> dict[str, Any]:
        try:
            asr = self._asr()
        except Exception as exc:
            return {"error": str(exc)[:200]}
        if asr is None:
            return {"skipped": "No speech model is downloaded yet, so speech to text was not tested."}
        heard, method = "", ""
        t0 = time.perf_counter()
        if self.speak_test:  # the computer says the phrase while the microphone listens (speakers -> mic)
            box: dict[str, Any] = {}
            th = threading.Thread(target=lambda: box.update(a=self._record(3.5)), daemon=True)
            th.start()
            time.sleep(0.4)
            with contextlib.suppress(Exception):
                self._say(phrase)
            th.join(6.0)
            if box.get("a") is not None:
                heard, method = str(asr.transcribe(box["a"]) or ""), "loopback"
        from away.selftest.checks import STT_MATCH, match

        if match(phrase, heard) < STT_MATCH and self.interactive and self.ask is not None:
            with contextlib.suppress(Exception):
                self.ask(f"Please say: \u201c{phrase}\u201d")
            t0 = time.perf_counter()
            heard, method = str(asr.transcribe(self._record(4.0)) or ""), "spoken"
        if not method:
            return {"skipped": "The microphone could not record for the speech test."}
        return {"heard": heard.strip(), "method": method, "ms": round((time.perf_counter() - t0) * 1000.0)}

    def _say(self, text: str) -> None:
        from away.voice import speech

        for _name, make in speech.engine_factories():
            eng = make()
            try:
                eng.speak(text, speech.Voice("normal", "normal", ""))
                return
            finally:
                with contextlib.suppress(Exception):
                    eng.close()

    def tts_factories(self) -> list[tuple[str, Callable[[], Any]]]:
        """Every voice engine this PC has (natural voices first, then the classic ones), each name once."""
        from away.voice import natural, speech

        fac: list[tuple[str, Callable[[], Any]]] = []
        with contextlib.suppress(Exception):
            fac += natural.natural_factories(lambda: getattr(self.cfg, "speech", None), self.data_dir)
        fac += speech.engine_factories()
        seen: set[str] = set()
        out = []
        for name, make in fac:
            if name not in seen:
                seen.add(name)
                out.append((name, make))
        return out

    def speak_one(self, name: str, make: Callable[[], Any], text: str = "Test.") -> dict[str, Any]:
        """Speak ``text`` with one engine; time to its first sound (natural voices) or until it finished."""
        from away.voice import speech

        row: dict[str, Any] = {"name": name}
        try:
            eng = make()
            first: dict[str, float] = {}
            t0 = time.perf_counter()
            play = getattr(eng, "play", None)
            if callable(play):  # natural voices: time to the first audio chunk exactly

                def timed(*a: Any, _p: Callable[..., Any] = play, _first: dict[str, float] = first, **k: Any) -> Any:
                    _first.setdefault("t", time.perf_counter())
                    return _p(*a, **k)

                eng.play = timed
            eng.speak(text, speech.Voice("normal", "quiet", ""))
            end = first.get("t", time.perf_counter())
            row["first_audio_ms"] = round((end - t0) * 1000.0)
            row["measured"] = "first sound" if "t" in first else "until it finished speaking"
            with contextlib.suppress(Exception):
                eng.close()
        except Exception as exc:
            row["error"] = str(exc)[:200]
        return row

    def tts(self) -> list[dict[str, Any]]:
        return [self.speak_one(name, make) for name, make in self.tts_factories()]

    def notification(self) -> dict[str, Any]:
        h = self.host or getattr(self.app, "agent_link", None)
        prompts = getattr(getattr(h, "core", None), "prompts", None)
        if prompts is None:
            return {"skipped": "Project Away is not running, so no pop-up was shown."}
        res = prompts.notify(
            "Self-test: this is how messages from your agents look.", "info", False, agent="Project Away"
        )
        return {"shown": bool(res.get("shown")), "via": res.get("via"), "reason": res.get("reason")}

    # ---------------------------------------------------------------- Windows Start key (proposal #4)
    def winkey(self) -> dict[str, Any]:
        """Press the Win key through the action engine and watch for the Start menu.

        The press takes the same path agents use (:meth:`away.actions_engine.ActionEngine.call`), so this
        check proves the proposal #4 fix on the real PC. Like the pop-up test above, the keypress really
        happens: the person asked for a self-test. Returns ``supported: False`` (with a reason) where the
        check cannot run: anywhere but Windows, or with no keyboard backend.
        """
        if sys.platform != "win32":
            return {"supported": False, "reason": "the Windows key check only runs on Windows"}
        from away import actions_engine as eng
        from away.agentlink import desktop

        try:
            secure = bool(desktop.secure_desktop_active())
        except Exception:
            secure = False
        engine = eng.get_engine()
        target = engine.backend if getattr(engine, "backend", None) is not None else None
        if target is None:
            try:
                target = engine.os_backend()
            except Exception as exc:
                return {"supported": False, "reason": f"no keyboard backend ({exc})"}
        t0 = time.monotonic()
        try:
            engine.client(eng.USER, target).key("cmd")
        except Exception as exc:
            ms = (time.monotonic() - t0) * 1000.0
            return {"supported": True, "opened": False, "ms": ms, "reason": f"the keypress failed ({exc})"}
        seen: dict[str, Any] | None = None
        while time.monotonic() - t0 < 2.0:
            fg = desktop.foreground_window() or {}
            if _start_menu_host(fg):
                seen = fg
                break
            time.sleep(0.05)
        ms = (time.monotonic() - t0) * 1000.0
        out: dict[str, Any] = {"supported": True, "opened": seen is not None, "ms": ms}
        if seen is not None:
            out["process"] = seen.get("process")
            out["title"] = seen.get("title")
            out["class"] = seen.get("class")
        else:
            fg = desktop.foreground_window() or {}
            out["foreground"] = {k: fg.get(k) for k in ("process", "title", "class")}
        if secure:
            out["secure_desktop"] = True
        return out

    # ---------------------------------------------------------------- link
    def link(self) -> dict[str, Any]:
        s = getattr(self.cfg, "agent_link", None)
        if s is not None and not bool(getattr(s, "enabled", True)):
            return {"enabled": False}
        h = self.host or getattr(self.app, "agent_link", None)
        out: dict[str, Any] = {"enabled": True, "server": {"running": False}}
        if h is not None:
            srv = getattr(h, "server", None)
            if srv is not None and getattr(srv, "running", False):
                port = int(srv.port)
                try:
                    out["server"] = {
                        "running": True,
                        "port": port,
                        "ok": True,
                        "ms": round(http_ms(f"http://127.0.0.1:{port}/v1/ping"), 1),
                    }
                except Exception as exc:
                    out["server"] = {"running": True, "port": port, "ok": False, "error": str(exc)[:120]}
            for key, attr in (("tunnel", "tunnel"), ("backup", "relay"), ("named", "named")):
                t = getattr(h, attr, None)
                url = getattr(t, "url", None) if t is not None else None
                if url:
                    try:
                        out[key] = {"ok": True, "ms": round(http_ms(f"{url}/v1/ping"), 1)}
                    except Exception as exc:
                        out[key] = {"ok": False, "error": str(exc)[:120]}
            with contextlib.suppress(Exception):
                reg = h.core.registry
                out["agents"] = {"paired": len(reg.list()), "reconnectable": len(reg.reconnectable())}
        relays = []
        for name, hostn, port, why in RELAYS:
            try:
                relays.append({"name": name, "ok": True, "ms": round(tcp_ms(hostn, port), 1)})
            except Exception as exc:
                relays.append({"name": name, "ok": False, "error": type(exc).__name__, "why": why})
        out["relays"] = relays
        return out

    # ---------------------------------------------------------------- PC load
    def load(self) -> dict[str, Any]:
        from away.app import sysload

        out: dict[str, Any] = {"top": sysload.top_processes(5, interval=1.0)}
        with contextlib.suppress(Exception):
            import psutil

            out["cpu_pct"] = float(psutil.cpu_percent(interval=0.5))
        out.update(gpu_load())
        return out

    def models(self) -> dict[str, Any]:
        md = self._models_dir()
        missing = []
        if self.cfg.hands.enabled and not (md / "hand_landmarker.task").exists():
            missing.append("hand model")
        if self.cfg.face.enabled and not (md / "face_landmarker.task").exists():
            missing.append("face model")
        rows: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            from away.app.model_registry import Registry

            rows = [
                {k: r.get(k) for k in ("slot", "title", "current", "current_title", "runs_on", "on")}
                for r in Registry(self.data_dir).rows(self.cfg)
            ]
        return {"missing": missing, "rows": rows}


def app_processes(psutil_mod: Any = None, me: int | None = None) -> list[int]:
    """Process ids of running Project Away apps (helpers such as the agent prompt window are not counted)."""
    try:
        ps = psutil_mod
        if ps is None:
            import psutil as ps  # type: ignore[no-redef]
    except Exception:
        return []
    mine = os.getpid() if me is None else me
    out = []
    for p in ps.process_iter(["pid", "name", "cmdline"]):
        try:
            name = str(p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
        except Exception:
            continue
        is_app = any(n in name for n in APP_NAMES) or "away.app.tray" in cmd or "away-tray" in cmd
        if name.startswith("python") and "away.app.tray" not in cmd and "away-tray" not in cmd:
            is_app = False
        if not is_app or any(w in cmd for w in HELPER_WORDS):
            continue
        if int(p.info.get("pid") or 0) == mine and "tray" not in cmd and "projectaway" not in name:
            continue  # this CLI run itself
        out.append(int(p.info["pid"]))
    return sorted(out)


def gpu_load(run: Callable[..., str | None] | None = None) -> dict[str, Any]:
    """NVIDIA load through nvidia-smi ({} when there is none)."""
    from away.app.hardware import run_cmd

    runner = run or run_cmd
    txt = runner(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name", "--format=csv,noheader,nounits"]
    )
    if not txt:
        return {}
    try:
        util, used, total, name = [x.strip() for x in txt.strip().splitlines()[0].split(",", 3)]
        return {
            "gpu_pct": float(util),
            "vram_used_mb": int(float(used)),
            "vram_total_mb": int(float(total)),
            "gpu_name": name,
        }
    except Exception:
        return {}
