"""System tray app (pystray): listening state, pause, mic on/off, dictation, settings, quit."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from away import config as cfgmod
from away import paths

STATE_COLORS = {
    "active": (46, 160, 67, 255),  # gestures on, mic listening (or push-to-talk idle)
    "recording": (33, 118, 214, 255),  # push-to-talk / toggle capturing
    "stopped": (214, 160, 33, 255),  # always-on: "stop listening" was said
    "muted": (200, 60, 60, 255),  # mic off
    "paused": (140, 140, 140, 255),  # gestures paused
}


# Windows keeps the tray tooltip in a 128-character UTF-16 buffer (the terminating zero included); a longer one makes
# pystray fail ("string too long (136, maximum length 128)"). Other systems take the same, shorter text.
TOOLTIP_MAX = 127


def tooltip_text(text: str, limit: int = TOOLTIP_MAX) -> str:
    """``text`` cut to ``limit`` UTF-16 units with an ellipsis, never splitting a character in two."""

    def units(t: str) -> int:
        return len(t.encode("utf-16-le")) // 2

    if units(text) <= limit:
        return text
    out = ""
    for ch in text:
        if units(out + ch) + 1 > limit:  # + 1 for the ellipsis
            break
        out += ch
    return out.rstrip(" -,") + "\u2026"


def icon_state(paused: bool, voice_state: str) -> str:
    """Which colour the tray icon shows: paused wins, then mic muted/stopped/recording, else active."""
    if paused:
        return "paused"
    if voice_state in ("muted", "stopped", "recording"):
        return voice_state
    return "active"


def make_icon_image(paused: bool = False, size: int = 64, state: str | None = None) -> Any:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = STATE_COLORS[state or ("paused" if paused else "active")]
    d.ellipse((4, 4, size - 4, size - 4), fill=color)
    # a simple open-hand glyph: palm + four fingers
    w = size // 10
    for i in range(4):
        x = size // 4 + i * (w + 2)
        d.rounded_rectangle((x, size // 5, x + w, size // 2 + 4), radius=w // 2, fill="white")
    d.rounded_rectangle((size // 4 - 2, size // 2, size * 3 // 4 + 2, size * 4 // 5), radius=6, fill="white")
    return img


def open_path(path: str) -> None:  # pragma: no cover - launches a GUI app
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class TrayController:
    """Menu state, separated from pystray so it can be unit-tested."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.paused = False
        self.last_activity = float("-inf")
        self.crash_note: str | None = None  # "the app crashed last time; details in crash.log" (v0.1.16)

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        self.app.set_paused(self.paused)
        return self.paused

    def toggle_dictation(self) -> bool:
        self.app.toggle_dictation()
        d = getattr(self.app, "dictation", None)
        return bool(d.enabled) if d is not None else False

    def dictation_enabled(self) -> bool:
        d = getattr(self.app, "dictation", None)
        return bool(d.enabled) if d is not None else False

    def settings_path(self) -> str:
        p = paths.config_file()
        if not p.exists():
            cfgmod.save(self.app.cfg, p)
        return str(p)

    # mic / listening ------------------------------------------------------
    def voice_state(self) -> str:
        fn = getattr(self.app, "voice_state", None)
        return str(fn()) if callable(fn) else "off"

    def mic_on(self) -> bool:
        return self.voice_state() not in ("muted", "off")

    def toggle_mic(self) -> bool:
        """Mic on/off (mute). Returns the new 'mic on' state."""
        on = not self.mic_on()
        fn = getattr(self.app, "set_mic", None)
        if callable(fn):
            fn(on)
        return self.mic_on()

    def title(self) -> str:
        state = self.voice_state()
        voice = {
            "listening": "listening",
            "stopped": "not listening",
            "muted": "mic off",
            "recording": "recording",
            "idle": "push-to-talk ready",
            "off": "voice off",
        }.get(state, state)
        text = f"Project Away - {'paused' if self.paused else 'active'}, {voice}"
        text += self.attention_suffix()
        cam = self.camera_status()
        text += f" - {cam}" if cam else ""
        badge = self.agent_badge()
        text += f" - {badge}" if badge else ""
        hint = str(getattr(self.app, "health_hint", "") or "")  # v0.1.23: busy PC / choppy tracking
        text += f" - {hint}" if hint else ""
        return text + (" - crashed last time, see crash.log" if self.crash_note else "")

    def tooltip(self) -> str:
        """:meth:`title`, short enough for the Windows tray tooltip (the menu shows it in full)."""
        return tooltip_text(self.title())

    def camera_status(self) -> str:
        """v0.1.21: "Settings is using the camera" while Settings (or a camera check) borrows it."""
        fn = getattr(self.app, "camera_status", None)
        try:
            return str(fn()) if callable(fn) else ""
        except Exception:
            return ""

    def attention_suffix(self) -> str:
        """v0.1.20: " - waiting for you (raise your hand...)" while the attention gate is on."""
        from away.comfort.ui import tray_suffix

        fn = getattr(getattr(self.app, "pipeline", None), "attention_state", None)
        return tray_suffix(fn()) if callable(fn) else ""

    def agent_badge(self) -> str:
        """ "Claude connected" / "Claude is using the computer" (Agent Link, v0.1.20), or ""."""
        from away.agentlink.host import badge

        return badge(getattr(self.app, "agent_link", None))

    def disconnect_agents(self) -> None:
        al = getattr(self.app, "agent_link", None)
        if al is not None:
            al.disconnect_all()

    def icon_state(self) -> str:
        return icon_state(self.paused, self.voice_state())

    @staticmethod
    def selftest_command() -> list[str]:
        """v0.1.24 "Run self-test": the report window in its own process (it runs the checks inside this app)."""
        from away.app.frozen import away_command

        return away_command(["selftest", "--window", "--trigger", "tray"])

    @staticmethod
    def check_pc_command() -> list[str]:
        """v0.1.24 "Check my PC": the guided real-hardware check window (measurements run inside this app)."""
        from away.app.frozen import away_command

        return away_command(["check", "--window", "--trigger", "tray"])

    def run_check_pc(self) -> None:  # pragma: no cover - launches a GUI process
        import subprocess

        try:
            subprocess.Popen(self.check_pc_command())
        except Exception:
            import logging

            logging.getLogger("away.tray").warning("could not open Check my PC", exc_info=True)

    def run_selftest(self) -> None:  # pragma: no cover - launches a GUI process
        import subprocess

        try:
            subprocess.Popen(self.selftest_command())
        except Exception:
            al = getattr(self.app, "agent_link", None)
            st = getattr(al, "selftest", None)
            if st is not None:
                st.start(trigger="tray")

    @staticmethod
    def settings_command() -> list[str]:
        """argv that opens the settings window in its own process (Tk and pystray each want the main thread)."""
        from away.app.frozen import away_command  # frozen: the app exe is the `away` CLI (no `-m away`)

        return away_command(["settings"])

    def is_idle(self, idle_s: float = 600.0, now: float | None = None) -> bool:
        """Safe moment for a silent auto-update: gestures paused, or no action fired for ``idle_s``."""
        import time

        if self.paused or bool(getattr(getattr(self.app, "pipeline", None), "paused", False)):
            return True
        t = time.monotonic() if now is None else now
        return t - self.last_activity >= idle_s

    def mark_activity(self, now: float | None = None) -> None:
        import time

        self.last_activity = time.monotonic() if now is None else now

    # face gate ------------------------------------------------------------
    def face_gate_on(self) -> bool:
        cfg = self.app.cfg
        return bool(cfg.face.enabled and cfg.face.gate_dictation)

    def set_face_gate(self, on: bool, path: Any = None) -> bool:
        """Turn the face gate on/off in the config file (the app hot-reloads it). Off also sets
        ``voice.ambient_fallback: type`` so speech is typed without a wake word. Returns the new state."""
        from pathlib import Path

        p = Path(path) if path is not None else paths.config_file()
        cfg, _ = cfgmod.load(p) if p.exists() else (self.app.cfg, [])
        cfg.face.gate_dictation = bool(on)
        if not on:
            cfg.voice.ambient_fallback = "type"
        cfgmod.save(cfg, p)
        self.app.cfg.face.gate_dictation = bool(on)
        if not on:
            self.app.cfg.voice.ambient_fallback = "type"
        return bool(on)

    # GPU pack ---------------------------------------------------------------
    def camera_retry_stopped(self) -> bool:
        """The app gave up re-probing a failed camera (v0.1.19): the tray then offers "Retry cameras"."""
        return getattr(self.app, "camera_retry_stopped", False) is True

    def retry_cameras(self) -> None:
        fn = getattr(self.app, "retry_cameras", None)
        if fn is not None:
            fn()

    def gpu_pack_offer(self) -> str | None:
        """Why to offer the NVIDIA GPU pack (GPU present, pack missing), or None."""
        try:
            from away.app.performance import gpu_pack_offer

            return gpu_pack_offer(self.app.cfg, getattr(self.app, "profile", None))
        except Exception:
            return None

    # shortcuts (v0.1.20) -----------------------------------------------------
    def shortcut_recording(self) -> bool:
        m = getattr(self.app, "macros", None)
        return bool(m is not None and m.recording)

    def shortcut_playing(self) -> bool:
        m = getattr(self.app, "macros", None)
        return bool(m is not None and m.playing)

    def toggle_shortcut_recording(self) -> None:
        """Tray "Record a shortcut" / "Stop recording": a tray recording is saved as "Shortcut N"."""
        m = getattr(self.app, "macros", None)
        if m is not None:
            m.hook_control("toggle_recording", {})

    def stop_shortcut(self) -> None:
        m = getattr(self.app, "macros", None)
        if m is not None:
            m.stop_playback()

    def stop_everything(self) -> int:
        """Tray "Stop everything" (audit Phase 1b): cancels every source's in-flight action at once (a shortcut,
        a drag, a dictation, an agent's action, which also loses its grant) and lets go of held keys/buttons."""
        from away.actions_engine import get_engine

        return get_engine().stop_all("you pressed Stop everything in the tray")

    def quit(self) -> None:
        self.app.stop()


class TrayGuard:
    """v0.1.23: every call into the tray icon goes through here. The crash log showed Windows COM errors
    (0x80010108 RPC_E_DISCONNECTED) in pystray's _hide/_message and one access violation: icon updates came from
    camera / voice threads while the icon was being stopped. Calls are serialized, never made once stopping began,
    and an error in one is logged once instead of escaping."""

    def __init__(self) -> None:
        import threading

        self.stopping = False
        self.errors = 0
        self._lock = threading.RLock()

    def call(self, fn: Any, *args: Any, **kw: Any) -> Any:
        if self.stopping:
            return None
        with self._lock:
            if self.stopping:
                return None
            try:
                return fn(*args, **kw)
            except Exception as exc:
                self.errors += 1
                if self.errors <= 3:
                    import logging

                    logging.getLogger("away.tray").warning("tray icon update failed (ignored): %s", exc)
                return None

    def stop(self, icon: Any) -> None:
        """Hide and stop the icon once; a COM object that is already gone is not touched again."""
        with self._lock:
            if self.stopping:
                return
            self.stopping = True
            for step in ("hide", "stop"):
                try:
                    if step == "hide":
                        if getattr(icon, "visible", False):
                            icon.visible = False
                    else:
                        icon.stop()
                except Exception as exc:
                    import logging

                    logging.getLogger("away.tray").info("tray icon %s failed (ignored): %s", step, exc)


def com_apartment(platform: str | None = None, ole32: Any = None) -> str:
    """v0.1.23: the tray's main thread is a single-threaded COM apartment (what Shell_NotifyIcon and the tray's
    window expect), set before anything else touches COM. Returns "sta", "already" (set earlier, maybe as MTA by
    another library), or "" off Windows."""
    plat = platform or sys.platform
    if not plat.startswith("win"):
        return ""
    try:
        if ole32 is None:  # pragma: no cover - Windows only
            import ctypes

            ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]
        hr = int(ole32.CoInitializeEx(None, 0x2)) & 0xFFFFFFFF  # COINIT_APARTMENTTHREADED
    except Exception:
        return ""
    if hr in (0, 1):  # S_OK, S_FALSE (already an STA)
        return "sta"
    return "already"  # RPC_E_CHANGED_MODE: another library made it MTA first


def camera_menu_text(app: Any) -> str:
    """v0.1.23: "Internal camera unavailable (probably in use by NVIDIA Broadcast)" while a camera is given up on."""
    info = getattr(app, "camera_unavailable", None)
    return str(info.get("headline") or "") if isinstance(info, dict) else ""


def comfort_menu_items(pystray: Any, ctl: TrayController, notify: Any) -> list[Any]:
    """Tray menu (v0.1.20): "Only respond when I'm looking at the screen" (a check box) and "Guided first run"."""
    from away.comfort import actions

    def on_gate(_icon: Any, _item: Any) -> None:
        on = not actions.switch_value(ctl.app.cfg, "attention.enabled")
        err = actions.set_switch_in_file("attention.enabled", on, live_cfg=ctl.app.cfg)
        notify("Project Away", err or ("Only responding when you look at the screen" if on else "Always responding"))

    return [
        pystray.MenuItem(
            "Only respond when I'm looking at the screen",
            on_gate,
            checked=lambda _i: actions.switch_value(ctl.app.cfg, "attention.enabled"),
        ),
        pystray.MenuItem("Guided first run...", lambda _icon, _item: actions.launch_first_run()),
    ]


def open_settings(ctl: TrayController) -> None:  # pragma: no cover - launches a GUI process
    try:
        subprocess.Popen(ctl.settings_command())
    except Exception:
        open_path(ctl.settings_path())  # fall back to the YAML file in the default editor


def update_menu_items(
    pystray: Any, updates: Any, on_update: Any, on_run_installer: Any, on_check_updates: Any
) -> list[Any]:
    """The update part of the tray menu: "Update to vX" while an update is available, "Run update installer"
    after a failed update whose installer is still on disk (opens it visibly, like a double-click), and
    "Check for updates"."""
    return [
        pystray.MenuItem(
            lambda _i: updates.menu_label() or "Update",
            on_update,
            visible=lambda _i: updates.menu_label() is not None,
            enabled=lambda _i: not updates.installing,
        ),
        pystray.MenuItem(
            lambda _i: updates.run_installer_label() or "Run update installer",
            on_run_installer,
            visible=lambda _i: updates.run_installer_label() is not None,
        ),
        pystray.MenuItem(
            "Check for updates", on_check_updates, enabled=lambda _i: not (updates.installing or updates.checking)
        ),
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a desktop session
    import contextlib
    import logging
    import threading

    from away.control.monitors import enable_dpi_awareness

    enable_dpi_awareness()  # before pystray/pynput/tkinter (Windows; no-op elsewhere)
    com_apartment()  # v0.1.23: STA on the tray thread before pystray / SAPI / pycaw touch COM
    import pystray

    from away.app.frozen import setup_logging
    from away.app.runner import AwayApp
    from away.app.updater import UpdateManager, exit_after_update

    setup_logging()  # away.* INFO -> away.log (windowed app) or the terminal
    from away.app import crashlog

    crash_note = crashlog.install("tray")  # crash.log (v0.1.16); None unless the last run crashed
    from away.app import instance

    if not instance.acquire():  # v0.1.23: already running: show its Settings instead of a second app
        logging.getLogger("away.tray").info("Project Away is already running: opening its Settings")
        with contextlib.suppress(Exception):
            subprocess.Popen(TrayController.settings_command())
        return 0
    had_config = cfgmod.config_path().exists()  # v0.1.20: an existing user is not shown the first-run guide
    cfg, _ = cfgmod.load()
    app = AwayApp(cfg)
    ctl = TrayController(app)
    ctl.paused = not cfg.app.always_on  # app.always_on: false starts paused (and muted)
    app.start()
    if not cfg.app.start_minimized:
        open_settings(ctl)
    from away.app import first_run

    first_start = first_run.should_auto_open()  # v0.1.20: once ever; a new user gets the guide, others a hint
    if first_start:
        first_run.mark_seen()

    guard = TrayGuard()

    def _set_icon(ic: Any) -> None:
        ic.icon = make_icon_image(ctl.paused, state=ctl.icon_state())
        ic.title = ctl.tooltip()

    def refresh(icon: Any) -> None:
        guard.call(_set_icon, icon)

    def on_pause(icon: Any, _item: Any) -> None:
        ctl.toggle_pause()
        refresh(icon)

    def on_dictation(icon: Any, _item: Any) -> None:
        ctl.toggle_dictation()
        refresh(icon)

    def on_mic(icon: Any, _item: Any) -> None:
        ctl.toggle_mic()
        refresh(icon)

    def on_settings(_icon: Any, _item: Any) -> None:
        open_settings(ctl)

    def on_edit_yaml(_icon: Any, _item: Any) -> None:
        open_path(ctl.settings_path())

    def on_quit(icon: Any, _item: Any) -> None:
        updates.stop()
        ctl.quit()
        guard.stop(icon)

    def notify(title: str, message: str) -> None:
        if guard.stopping:
            return
        try:
            with guard._lock:
                icon.notify(message, title)
        except Exception:
            from away.app.frozen import notify as desktop_notify

            desktop_notify(title, message)

    def _safe_update_menu(ic: Any) -> None:
        # some pystray backends cannot rebuild the menu from another thread; the next open does
        guard.call(ic.update_menu)

    def quit_for_update() -> None:
        # Runs on the update worker thread. pystray's icon.stop() only posts a message to the tray loop, but
        # the app shutdown (camera/voice threads) can hang; the installer waits for this process to be gone,
        # so after a short grace the process exits no matter what.
        exit_after_update(lambda: on_quit(icon, None))

    updates = UpdateManager(
        cfg.updates,
        notify=notify,
        on_change=lambda: _safe_update_menu(icon),
        quit_app=quit_for_update,
        is_idle=ctl.is_idle,
    )

    def on_update(_icon: Any, _item: Any) -> None:
        threading.Thread(target=updates.install_now, name="away-update-install", daemon=True).start()

    def on_check_updates(_icon: Any, _item: Any) -> None:
        threading.Thread(target=updates.check_now, name="away-update-check", daemon=True).start()

    def on_run_installer(_icon: Any, _item: Any) -> None:
        updates.run_installer()

    gpu_offer = {"text": ctl.gpu_pack_offer(), "installing": False}
    gate_hint = {"shown": False}

    def on_face_gate(icon: Any, _item: Any) -> None:
        try:
            on = ctl.set_face_gate(not ctl.face_gate_on())
            notify("Project Away", "Face gate on" if on else "Face gate off: speech is typed without a wake word")
        except Exception as exc:
            notify("Project Away", f"could not change the face gate: {exc}")
        _safe_update_menu(icon)

    def on_gpu_install(icon: Any, _item: Any) -> None:
        from away.app import gpupack

        def run() -> None:
            gpu_offer["installing"] = True
            notify("Project Away", "Downloading the GPU pack (~570 MB)...")
            try:
                gpupack.install()
                gpu_offer["text"] = None
                notify("Project Away", "GPU pack installed: restart Project Away to use the GPU for speech")
            except Exception as exc:
                notify("Project Away: GPU pack", f"download failed: {exc} (try again to resume)")
            finally:
                gpu_offer["installing"] = False
                _safe_update_menu(icon)

        threading.Thread(target=run, name="gpu-pack", daemon=True).start()

    menu = pystray.Menu(
        pystray.MenuItem(lambda _i: ctl.title(), None, enabled=False),
        pystray.MenuItem(lambda _i: "Resume" if ctl.paused else "Pause", on_pause, default=True),
        pystray.MenuItem("Mic on", on_mic, checked=lambda _i: ctl.mic_on()),
        pystray.MenuItem(
            lambda _i: "Start listening" if ctl.voice_state() == "stopped" else "Stop listening / dictation",
            on_dictation,
        ),
        pystray.MenuItem(
            "Face gate (type only when facing the camera)",
            on_face_gate,
            checked=lambda _i: ctl.face_gate_on(),
            visible=lambda _i: gate_hint["shown"] or not ctl.face_gate_on(),
        ),
        pystray.MenuItem(
            "Install GPU pack (faster speech)",
            on_gpu_install,
            visible=lambda _i: bool(gpu_offer["text"]),
            enabled=lambda _i: not gpu_offer["installing"],
        ),
        pystray.MenuItem(  # v0.1.23: "Internal camera unavailable (probably in use by NVIDIA Broadcast)"
            lambda _i: camera_menu_text(app) or "Camera unavailable",
            None,
            enabled=False,
            visible=lambda _i: bool(camera_menu_text(app)),
        ),
        pystray.MenuItem(
            "Retry cameras",
            lambda _icon, _item: ctl.retry_cameras(),
            visible=lambda _i: ctl.camera_retry_stopped(),
        ),
        pystray.MenuItem(
            lambda _i: "Turn off " + ", ".join((getattr(app, "camera_unavailable", None) or {}).get("devices") or []),
            lambda _icon, _item: [app.turn_off_camera(d) for d in (app.camera_unavailable or {}).get("devices", [])],
            visible=lambda _i: bool(camera_menu_text(app)),
        ),
        *comfort_menu_items(pystray, ctl, notify),  # v0.1.20: attention gate switch + guided first run
        pystray.MenuItem(
            lambda _i: "Stop recording shortcut" if ctl.shortcut_recording() else "Record a shortcut",
            lambda _icon, _item: ctl.toggle_shortcut_recording(),
        ),
        pystray.MenuItem(
            "Stop shortcut", lambda _icon, _item: ctl.stop_shortcut(), visible=lambda _i: ctl.shortcut_playing()
        ),
        pystray.MenuItem("Stop everything", lambda _icon, _item: ctl.stop_everything()),
        pystray.MenuItem(
            lambda _i: f"Disconnect AI agents ({ctl.agent_badge()})",
            lambda _icon, _item: ctl.disconnect_agents(),
            visible=lambda _i: bool(ctl.agent_badge()),
        ),
        pystray.MenuItem(  # v0.1.23
            "Reset cursor size to normal", lambda _icon, _item: app.reset_cursor_size()
        ),
        pystray.MenuItem("Check my PC", lambda _icon, _item: ctl.run_check_pc()),  # v0.1.24
        pystray.MenuItem("Run self-test", lambda _icon, _item: ctl.run_selftest()),  # v0.1.24
        pystray.MenuItem("Settings...", on_settings),
        pystray.MenuItem("Edit config file", on_edit_yaml),
        *update_menu_items(pystray, updates, on_update, on_run_installer, on_check_updates),
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon("project-away", make_icon_image(ctl.paused, state=ctl.icon_state()), ctl.tooltip(), menu)
    app.tray_icon = icon  # v0.1.24: Check my PC reads whether it shows and its tooltip
    topics = (
        "voice.state",
        "app.paused",
        "gesture.clutch",
        "cameras.retry_stopped",
        "cameras.retry_requested",
        "cameras.turned_off",
        "app.health_hint",
    )
    # v0.1.20: the attention gate and connected AI agents change the tray title
    for topic in (*topics, "attention.state", "agentlink.changed", "camera.handover"):
        app.bus.subscribe(topic, lambda _e: _sync(icon))
    app.bus.subscribe("action.fired", lambda _e: ctl.mark_activity())
    app.bus.subscribe("app.notice", lambda e: notify(str(e.data.get("title", "Project Away")), str(e.data["message"])))

    def _gate_hint(_e: Any) -> None:
        gate_hint["shown"] = True
        _safe_update_menu(icon)

    app.bus.subscribe("voice.gate_hint", _gate_hint)
    if gpu_offer["text"]:
        from away.app.notices import once

        if once("gpu-pack-offer"):  # one notification ever; the menu item stays until it is installed
            notify(
                "Project Away: faster speech available", f"{gpu_offer['text']}. Use the tray menu: Install GPU pack."
            )
    ctl.mark_activity()  # the idle clock starts at launch
    if first_start:
        if had_config:
            notify("Project Away", "New: a guided tour and comfort helpers. Tray menu > Guided first run...")
        else:
            from away.comfort.actions import launch_first_run

            launch_first_run()
    if crash_note:
        ctl.crash_note = f"{crash_note} ({crashlog.crash_log_path()})"
        notify("Project Away", ctl.crash_note)

    def _sync(ic: Any) -> None:
        if guard.stopping:
            return
        ctl.paused = bool(app.pipeline.paused)
        refresh(ic)
        guard.call(ic.update_menu)

    updates.start()  # reports the last install; first check right away, then every 24 h (unless updates.check: false)
    icon.run()
    if updates.exiting:  # quitting for an update: do not wait on any leftover thread at interpreter exit
        logging.getLogger("away.updater").info("update: tray closed, exiting for the installer")
        logging.shutdown()
        os._exit(0)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
