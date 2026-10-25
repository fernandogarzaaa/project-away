"""System tray app (pystray): listening state, pause, mic on/off, dictation, settings, quit."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable
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

#: v0.1.28: "an AI agent may be watching" dot in the tray icon's corner (drawn with a white ring so it reads
#: on every state colour)
AGENT_DOT_COLOR = (33, 118, 214, 255)


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


def make_icon_image(paused: bool = False, size: int = 64, state: str | None = None, agent_dot: bool = False) -> Any:
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
    if agent_dot:  # v0.1.28: a dot in the corner while an agent holds snapshots/screen/control and was active
        r = max(3, size // 8)
        cx, cy = size - r - 2, r + 2
        d.ellipse((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2), fill="white")
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=AGENT_DOT_COLOR)
    return img


def agent_popup_lines(host: Any, now: float | None = None) -> list[str]:
    """One plain-words line per agent for the tray popup: "Claude — can see your screen · used your computer
    2 min ago". Tk-free and pure so it is unit-testable; :meth:`TrayController.show_agent_popup` shows it."""
    from away.agentlink.host import watchers

    lines = []
    for w in watchers(host, now=now):
        caps = ", ".join(w["capabilities"]) or "connected"
        tail = "using your computer right now" if w["controlling"] else f"used your computer {w['active']}"
        lines.append(f"{w['name']} — {caps} · {tail}")
    return lines


def open_path(path: str) -> None:  # pragma: no cover - launches a GUI app
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def menu_safe(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a pystray menu callback: it runs on the tray's window-message thread, where an
    escaping exception (a COM error deep in the app, a torn-down icon, ...) would break the
    message loop and take the tray down with it. Log it and keep the menu alive instead."""
    import functools
    import logging

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(*args: Any) -> Any:
            try:
                return fn(*args)
            except Exception as exc:
                logging.getLogger("away.tray").warning("tray menu %r failed (ignored): %s", name, exc)
                return None

        # pystray decides how to call a menu action from ``action.__code__.co_argcount`` (minus one for a bound
        # method) and rejects anything that is not 0, 1 or 2 arguments. A bare ``*args`` wrapper reports 0, so
        # the tray refused its own menu at start-up (ValueError) and plain functions were called with no
        # arguments. The wrapper therefore keeps the wrapped function's argument count.
        n = fn.__code__.co_argcount
        wrapper: Callable[..., Any]
        if n == 3:

            def wrapper(a: Any, b: Any, c: Any) -> Any:
                return guarded(a, b, c)

        elif n == 2:

            def wrapper(a: Any, b: Any) -> Any:
                return guarded(a, b)

        elif n == 1:

            def wrapper(a: Any) -> Any:
                return guarded(a)

        elif n == 0:

            def wrapper() -> Any:
                return guarded()

        else:
            raise TypeError(f"menu_safe: {fn!r} takes {n} arguments; a pystray menu action takes 0 to 2")
        return functools.wraps(fn)(wrapper)

    return deco


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
        if not p.exists():  # the defaults (the live config has this camera's calibration layered in)
            cfgmod.save(cfgmod.AppConfig(), p)
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
            "mic_error": "mic problem",
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

    def agent_dot(self) -> bool:
        """v0.1.28: the tray icon's state dot (an agent holds snapshots/screen/control and was recently active)."""
        from away.agentlink.host import watcher_dot

        try:
            return watcher_dot(getattr(self.app, "agent_link", None))
        except Exception:
            return False

    def popup_lines(self) -> list[str]:
        """v0.1.28: the "who can see what" lines (one per agent); [] when Agent Link is off."""
        try:
            return agent_popup_lines(getattr(self.app, "agent_link", None))
        except Exception:
            return []

    @staticmethod
    def agent_link_command() -> list[str]:
        """argv that opens Settings on the Agent Link tab (the popup's "Open Agent Link" button)."""
        from away.app.frozen import away_command

        return away_command(["settings", "--tab", "Agent Link"])

    def open_agent_link(self) -> None:  # pragma: no cover - launches a GUI process
        import subprocess

        try:
            subprocess.Popen(self.agent_link_command())
        except Exception:
            import logging

            logging.getLogger("away.tray").warning("could not open Agent Link settings", exc_info=True)

    def show_agent_popup(self) -> None:  # pragma: no cover - Tk GUI
        """v0.1.28: the "who can see what" popup: one line per agent, each with an Open Agent Link button."""
        import tkinter as tk
        from tkinter import ttk

        lines = self.popup_lines()
        root = tk.Tk()
        root.title("Who can see what")
        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="These AI agents are connected to Project Away:", wraplength=400).pack(
            anchor="w", pady=(0, 8)
        )
        for line in lines:
            row = ttk.Frame(frm)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=line, wraplength=320).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Open Agent Link", command=self.open_agent_link).pack(side="right", padx=(8, 0))
        ttk.Button(frm, text="Close", command=root.destroy).pack(pady=(10, 0))
        root.mainloop()

    def disconnect_agents(self) -> None:
        al = getattr(self.app, "agent_link", None)
        if al is not None:
            al.disconnect_all()

    def open_recent_activity(self) -> None:  # pragma: no cover - launches a GUI process
        """Open Settings directly on the Agent Link tab for the recent activity and consent receipts."""
        _open_settings_tab(self, "Agent Link")

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


class TrayMenuCallbacks:
    """The pystray menu callbacks, kept at module level (instead of closures in :func:`main`)
    so tests can drive them directly with fakes.

    ``icon`` is assigned after the pystray Icon exists; callbacks only run on user clicks,
    which happen after that. Every callback that runs on pystray's window-message thread is
    wrapped in :func:`menu_safe`: an escaping exception there would break the message loop
    and take the tray down with it.
    """

    def __init__(self, ctl: TrayController, app: Any, updates: Any, guard: TrayGuard) -> None:
        self.ctl = ctl
        self.app = app
        self.updates = updates  # set after UpdateManager is built (it needs our notify/quit_app)
        self.guard = guard
        self.icon: Any = None
        self.gpu_offer = {"text": ctl.gpu_pack_offer(), "installing": False}
        self.gate_hint = {"shown": False}
        # v0.1.32: pystray's macOS notifications run osascript, credited to Script Editor (a click opens it); there
        # the app's own message window is used instead (see notify)
        self.icon_notifications = sys.platform != "darwin"

    # -- icon / menu plumbing -------------------------------------------------
    def _set_icon(self, icon: Any) -> None:
        icon.icon = make_icon_image(self.ctl.paused, state=self.ctl.icon_state())
        icon.title = self.ctl.tooltip()

    def refresh(self, icon: Any) -> None:
        self._on_icon(self._set_icon, icon)

    def _on_icon(self, fn: Any, *args: Any) -> None:
        # v0.1.34: the change runs on the main thread on a Mac (see on_main_thread); the guard is checked when it runs
        on_main_thread(lambda: self.guard.call(fn, *args))

    def notify(self, title: str, message: str) -> None:
        if self.guard.stopping:
            return
        from away.app.frozen import notify as desktop_notify

        if not self.icon_notifications:  # macOS: Project Away's own message window (frozen.notify)
            desktop_notify(title, message)
            return
        try:
            with self.guard._lock:
                self.icon.notify(message, title)
        except Exception:
            desktop_notify(title, message)

    def safe_update_menu(self, icon: Any = None) -> None:
        # some pystray backends cannot rebuild the menu from another thread; the next open does
        ic = icon if icon is not None else self.icon
        if ic is not None:
            self._on_icon(ic.update_menu)

    def quit_for_update(self) -> None:
        # Runs on the update worker thread. pystray's icon.stop() only posts a message to the tray
        # loop, but the app shutdown (camera/voice threads) can hang; the installer waits for this
        # process to be gone, so after a short grace the process exits no matter what.
        from away.app.updater import exit_after_update

        on_quit = self.on_quit
        icon = self.icon
        exit_after_update(lambda: on_quit(icon, None))

    # -- menu callbacks ---------------------------------------------------------
    @menu_safe("pause")
    def on_pause(self, icon: Any, _item: Any) -> None:
        self.ctl.toggle_pause()
        self.refresh(icon)

    @menu_safe("dictation")
    def on_dictation(self, icon: Any, _item: Any) -> None:
        self.ctl.toggle_dictation()
        self.refresh(icon)

    @menu_safe("mic")
    def on_mic(self, icon: Any, _item: Any) -> None:
        self.ctl.toggle_mic()
        self.refresh(icon)

    @menu_safe("settings")
    def on_settings(self, _icon: Any, _item: Any) -> None:
        open_settings(self.ctl)

    @menu_safe("edit config")
    def on_edit_yaml(self, _icon: Any, _item: Any) -> None:
        open_path(self.ctl.settings_path())

    @menu_safe("quit")
    def on_quit(self, icon: Any, _item: Any) -> None:
        self.updates.stop()
        self.ctl.quit()
        on_main_thread(lambda: self.guard.stop(icon))  # an update quits from a worker thread; hiding is AppKit too

    @menu_safe("install update")
    def on_update(self, _icon: Any, _item: Any) -> None:
        threading.Thread(target=self.updates.install_now, name="away-update-install", daemon=True).start()

    @menu_safe("check updates")
    def on_check_updates(self, _icon: Any, _item: Any) -> None:
        threading.Thread(target=self.updates.check_now, name="away-update-check", daemon=True).start()

    @menu_safe("run update installer")
    def on_run_installer(self, _icon: Any, _item: Any) -> None:
        self.updates.run_installer()

    @menu_safe("face gate")
    def on_face_gate(self, icon: Any, _item: Any) -> None:
        try:
            on = self.ctl.set_face_gate(not self.ctl.face_gate_on())
            self.notify("Project Away", "Face gate on" if on else "Face gate off: speech is typed without a wake word")
        except Exception as exc:
            self.notify("Project Away", f"could not change the face gate: {exc}")
        self.safe_update_menu(icon)

    @menu_safe("gpu pack install")
    def on_gpu_install(self, icon: Any, _item: Any) -> None:
        from away.app import gpupack

        def run() -> None:
            self.gpu_offer["installing"] = True
            self.notify("Project Away", "Downloading the GPU pack (~570 MB)...")
            try:
                gpupack.install()
                self.gpu_offer["text"] = None
                self.notify("Project Away", "GPU pack installed: restart Project Away to use the GPU for speech")
            except Exception as exc:
                self.notify("Project Away: GPU pack", f"download failed: {exc} (try again to resume)")
            finally:
                self.gpu_offer["installing"] = False
                self.safe_update_menu(icon)

        threading.Thread(target=run, name="gpu-pack", daemon=True).start()

    @menu_safe("retry cameras")
    def on_retry_cameras(self, _icon: Any, _item: Any) -> None:
        self.ctl.retry_cameras()

    @menu_safe("turn off camera")
    def on_turn_off_camera(self, _icon: Any, _item: Any) -> None:
        app = self.app
        [app.turn_off_camera(d) for d in (app.camera_unavailable or {}).get("devices", [])]

    @menu_safe("record shortcut")
    def on_record_shortcut(self, _icon: Any, _item: Any) -> None:
        self.ctl.toggle_shortcut_recording()

    @menu_safe("stop shortcut")
    def on_stop_shortcut(self, _icon: Any, _item: Any) -> None:
        self.ctl.stop_shortcut()

    @menu_safe("stop everything")
    def on_stop_everything(self, _icon: Any, _item: Any) -> None:
        self.ctl.stop_everything()

    @menu_safe("disconnect agents")
    def on_disconnect_agents(self, _icon: Any, _item: Any) -> None:
        self.ctl.disconnect_agents()

    @menu_safe("reset cursor size")
    def on_reset_cursor(self, _icon: Any, _item: Any) -> None:
        self.app.reset_cursor_size()

    @menu_safe("mac permissions")
    def on_mac_permissions(self, _icon: Any, _item: Any) -> None:
        from away.app import macperms

        state = {"accessibility": macperms.accessibility(), "input_monitoring": macperms.input_monitoring()}
        if all(v is True for v in state.values()):
            self.notify(
                "Project Away", "Pointer and keyboard control are allowed (Accessibility and Input Monitoring)."
            )
            return
        missing = [k for k, v in state.items() if v is False]
        macperms.open_settings(missing[0] if missing else "accessibility")

    @menu_safe("check my PC")
    def on_check_pc(self, _icon: Any, _item: Any) -> None:
        self.ctl.run_check_pc()

    @menu_safe("run self-test")
    def on_run_selftest(self, _icon: Any, _item: Any) -> None:
        self.ctl.run_selftest()

    def on_gate_hint(self, _event: Any) -> None:
        """Bus handler for "voice.gate_hint": show the face-gate menu item once."""
        self.gate_hint["shown"] = True
        self.safe_update_menu()

    def sync_icon(self, icon: Any) -> None:
        """Bus handler: refresh the tray icon after state changes (pause, voice, cameras...)."""
        if self.guard.stopping:
            return
        self.ctl.paused = bool(self.app.pipeline.paused)
        self.refresh(icon)
        self._on_icon(icon.update_menu)


def on_main_thread(fn: Callable[[], Any], platform: str | None = None) -> None:
    """v0.1.34 (macOS): AppKit may only be touched from the main thread, and macOS 27 kills the app when it is not
    (SIGTRAP in -[NSStatusItem setMenu:], eight crash reports from 0.1.32 and 0.1.33). pystray's Mac backend changes
    the status item on whichever thread sets ``icon.icon`` / ``icon.title`` or calls ``update_menu``, and the bus
    handlers that do so run on worker threads. On a Mac, off the main thread, ``fn`` is queued to the main run loop
    (pystray's NSApplication runs it); everywhere else it runs now, as before."""
    plat = platform or sys.platform
    if plat != "darwin" or threading.current_thread() is threading.main_thread():
        fn()
        return
    from Foundation import NSOperationQueue

    NSOperationQueue.mainQueue().addOperationWithBlock_(fn)


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


#: v0.1.28: the tray menu shows at most this many per-agent lines (pystray cannot add menu items after the menu
#: is built, so the slots are fixed and the extras live in the popup)
POPUP_MENU_LINES = 5


def agent_popup_menu_items(pystray: Any, ctl: TrayController) -> list[Any]:
    """v0.1.28: one disabled line per connected agent (what it can see, when it was last active), then
    "Who can see what..." which opens the popup with an Open Agent Link button per agent. Clicking the tray
    icon shows the menu, so the lines are right there; the popup has the buttons."""

    def on_popup(_icon: Any, _item: Any) -> None:
        ctl.show_agent_popup()

    def line(i: int) -> str:
        lines = ctl.popup_lines()
        return lines[i] if i < len(lines) else ""

    items: list[Any] = [
        pystray.MenuItem(
            lambda _i, i=i: line(i),
            None,
            enabled=False,
            visible=lambda _i, i=i: i < len(ctl.popup_lines()),
        )
        for i in range(POPUP_MENU_LINES)
    ]
    items.append(
        pystray.MenuItem(
            lambda _i: f"Who can see what ({len(ctl.popup_lines())})...",
            on_popup,
            visible=lambda _i: bool(ctl.popup_lines()),
        )
    )
    return items


def comfort_menu_items(pystray: Any, ctl: TrayController, notify: Any) -> list[Any]:
    """Tray menu (v0.1.20): "Only respond when I'm looking at the screen" (a check box) and "Guided first run"."""
    from away.comfort import actions

    @menu_safe("attention gate")
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
        pystray.MenuItem(
            "Guided first run...", menu_safe("guided first run")(lambda _icon, _item: actions.launch_first_run())
        ),
        pystray.MenuItem(
            "Calibrate tracking...",
            menu_safe("calibrate tracking")(lambda _icon, _item: actions.launch_calibrate()),
        ),  # v0.1.25
    ]


def open_settings(ctl: TrayController) -> None:  # pragma: no cover - launches a GUI process
    try:
        subprocess.Popen(ctl.settings_command())
    except Exception:
        open_path(ctl.settings_path())  # fall back to the YAML file in the default editor


def _open_settings_tab(ctl: TrayController, tab: str) -> None:  # pragma: no cover - launches a GUI process
    try:
        subprocess.Popen([*ctl.settings_command(), "--tab", tab])
    except Exception:
        open_settings(ctl)


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


def build_tray_menu(
    pystray: Any,
    ctl: TrayController,
    app: Any,
    callbacks: TrayMenuCallbacks,
    updates: Any,
    live_menu_items: Any,
    open_settings_tab: Callable[[str], None],
) -> Any:
    """Build the pystray menu. Extracted from :func:`main` so tests can drive it with a fake
    ``pystray`` module and verify the wiring."""
    notify = callbacks.notify
    gpu_offer = callbacks.gpu_offer
    gate_hint = callbacks.gate_hint
    return pystray.Menu(
        pystray.MenuItem(lambda _i: ctl.title(), None, enabled=False),
        *agent_popup_menu_items(pystray, ctl),  # v0.1.28: one line per connected agent, then the popup
        pystray.MenuItem(lambda _i: "Resume" if ctl.paused else "Pause", callbacks.on_pause, default=True),
        pystray.MenuItem("Mic on", callbacks.on_mic, checked=lambda _i: ctl.mic_on()),
        pystray.MenuItem(
            lambda _i: "Start listening" if ctl.voice_state() == "stopped" else "Stop listening / dictation",
            callbacks.on_dictation,
        ),
        pystray.MenuItem(
            "Face gate (type only when facing the camera)",
            callbacks.on_face_gate,
            checked=lambda _i: ctl.face_gate_on(),
            visible=lambda _i: gate_hint["shown"] or not ctl.face_gate_on(),
        ),
        pystray.MenuItem(
            "Install GPU pack (faster speech)",
            callbacks.on_gpu_install,
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
            callbacks.on_retry_cameras,
            visible=lambda _i: ctl.camera_retry_stopped(),
        ),
        pystray.MenuItem(
            lambda _i: "Turn off " + ", ".join((getattr(app, "camera_unavailable", None) or {}).get("devices") or []),
            callbacks.on_turn_off_camera,
            visible=lambda _i: bool(camera_menu_text(app)),
        ),
        *comfort_menu_items(pystray, ctl, notify),  # v0.1.20: attention gate switch + guided first run
        pystray.MenuItem(
            lambda _i: "Stop recording shortcut" if ctl.shortcut_recording() else "Record a shortcut",
            callbacks.on_record_shortcut,
        ),
        pystray.MenuItem(
            "Stop shortcut",
            callbacks.on_stop_shortcut,
            visible=lambda _i: ctl.shortcut_playing(),
        ),
        pystray.MenuItem("Stop everything", callbacks.on_stop_everything),
        pystray.MenuItem(
            lambda _i: f"Disconnect AI agents ({ctl.agent_badge()})",
            callbacks.on_disconnect_agents,
            visible=lambda _i: bool(ctl.agent_badge()),
        ),
        *live_menu_items(pystray, app, open_settings_tab),  # "Live agent: on (Big Pickle)"
        pystray.MenuItem("Recent activity...", lambda _i, _item: ctl.open_recent_activity()),
        pystray.MenuItem(  # v0.1.23
            "Reset cursor size to normal", callbacks.on_reset_cursor
        ),
        pystray.MenuItem("Check my PC", callbacks.on_check_pc),  # v0.1.24
        pystray.MenuItem(  # v0.1.32: the macOS permissions without which control silently does nothing
            "Allow pointer and keyboard control...", callbacks.on_mac_permissions, visible=sys.platform == "darwin"
        ),
        pystray.MenuItem("Run self-test", callbacks.on_run_selftest),  # v0.1.24
        pystray.MenuItem("Settings...", callbacks.on_settings),
        pystray.MenuItem("Edit config file", callbacks.on_edit_yaml),
        *update_menu_items(
            pystray, updates, callbacks.on_update, callbacks.on_run_installer, callbacks.on_check_updates
        ),
        pystray.MenuItem("Quit", callbacks.on_quit),
    )


def ask_mac_permissions(callbacks: Any) -> Any:
    """v0.1.32 (macOS): without Accessibility and Input Monitoring the pointer, clicks, typing and hotkeys silently
    do nothing. Check both once the icon exists; macOS asks for a missing one itself and one message says what to
    allow. On a thread: the icon never waits for it."""
    import threading

    from away.app import macperms

    t = threading.Thread(
        target=lambda: macperms.check_and_ask(callbacks.notify), name="away-mac-permissions", daemon=True
    )
    t.start()
    return t


def acquire_single_instance() -> bool:
    """True when this process owns the single app instance; False when another copy is running."""
    from away.app import instance

    return instance.acquire()


def open_running_settings() -> None:
    """Ask the already-running copy to open Settings. Best effort; never raises."""
    import contextlib
    import logging

    logging.getLogger("away.tray").info("Project Away is already running: opening its Settings")
    with contextlib.suppress(Exception):
        subprocess.Popen(TrayController.settings_command())


def load_tray_config() -> tuple[Any, bool]:
    """Load the app config. Returns (cfg, had_config); had_config tells first-run apart."""
    had_config = cfgmod.config_path().exists()  # v0.1.20: an existing user is not shown the first-run guide
    cfg, _ = cfgmod.load()
    return cfg, had_config


def create_tray_app(cfg: Any) -> tuple[Any, TrayController]:
    """Build the AwayApp and its TrayController, apply the startup pause state, start the app."""
    from away.app.runner import AwayApp

    app = AwayApp(cfg)
    ctl = TrayController(app)
    ctl.paused = not cfg.app.always_on  # app.always_on: false starts paused (and muted)
    app.start()
    return app, ctl


def subscribe_tray_bus(app: Any, ctl: TrayController, callbacks: Any, icon: Any) -> None:
    """Subscribe the icon-refresh and activity handlers to the app bus."""
    notify = callbacks.notify
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
        app.bus.subscribe(topic, lambda _e: callbacks.sync_icon(icon))
    app.bus.subscribe("action.fired", lambda _e: ctl.mark_activity())
    app.bus.subscribe("app.notice", lambda e: notify(str(e.data.get("title", "Project Away")), str(e.data["message"])))
    app.bus.subscribe("voice.gate_hint", callbacks.on_gate_hint)


def setup_tray_runtime(app: Any, ctl: TrayController, cfg: Any, pystray: Any) -> tuple[Any, Any, Any]:
    """Wire the tray runtime: callbacks, update manager, menu, icon, bus subscriptions.

    Returns (callbacks, updates, icon).
    """
    from away.app.updater import UpdateManager

    guard = TrayGuard()
    callbacks = TrayMenuCallbacks(ctl, app, None, guard)  # updates wired in below
    updates = UpdateManager(
        cfg.updates,
        notify=callbacks.notify,
        on_change=callbacks.safe_update_menu,
        quit_app=callbacks.quit_for_update,
        is_idle=ctl.is_idle,
    )
    callbacks.updates = updates
    from away.liveagent.integrate import hook_updates
    from away.liveagent.integrate import menu_items as live_menu_items

    hook_updates(updates)  # Away's own copy of opencode is updated when Away checks for updates
    menu = build_tray_menu(
        pystray,
        ctl,
        app,
        callbacks,
        updates,
        live_menu_items,
        lambda tab: _open_settings_tab(ctl, tab),
    )
    icon = pystray.Icon("project-away", make_icon_image(ctl.paused, state=ctl.icon_state()), ctl.tooltip(), menu)
    callbacks.icon = icon
    app.tray_icon = icon  # v0.1.24: Check my PC reads whether it shows and its tooltip
    subscribe_tray_bus(app, ctl, callbacks, icon)
    return callbacks, updates, icon


def maybe_offer_gpu_pack(callbacks: Any) -> None:
    """One-time notification about the GPU pack while the offer is live."""
    gpu_offer = callbacks.gpu_offer
    if gpu_offer["text"]:
        from away.app.notices import once

        if once("gpu-pack-offer"):  # one notification ever; the menu item stays until it is installed
            callbacks.notify(
                "Project Away: faster speech available", f"{gpu_offer['text']}. Use the tray menu: Install GPU pack."
            )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a desktop session
    import logging

    from away.control.monitors import enable_dpi_awareness

    enable_dpi_awareness()  # before pystray/pynput/tkinter (Windows; no-op elsewhere)
    com_apartment()  # v0.1.23: STA on the tray thread before pystray / SAPI / pycaw touch COM
    import pystray

    from away.app.frozen import setup_logging

    setup_logging()  # away.* INFO -> away.log (windowed app) or the terminal
    from away.app import crashlog

    crash_note = crashlog.install("tray")  # crash.log (v0.1.16); None unless the last run crashed
    if sys.platform == "darwin":
        from away.app import macperms

        macperms.warm_up()  # before any listener thread: PyObjC's lazy HIServices is not thread-safe
    if not acquire_single_instance():  # v0.1.23: already running: show its Settings instead of a second app
        open_running_settings()
        return 0
    cfg, had_config = load_tray_config()
    app, ctl = create_tray_app(cfg)
    if not cfg.app.start_minimized:
        open_settings(ctl)
    from away.app import first_run

    first_start = first_run.should_auto_open()  # v0.1.20: once ever; a new user gets the guide, others a hint
    if first_start:
        first_run.mark_seen()
    callbacks, updates, icon = setup_tray_runtime(app, ctl, cfg, pystray)
    maybe_offer_gpu_pack(callbacks)
    ctl.mark_activity()  # the idle clock starts at launch
    if first_start:
        if had_config:
            callbacks.notify("Project Away", "New: a guided tour and comfort helpers. Tray menu > Guided first run...")
        else:
            from away.comfort.actions import launch_first_run

            launch_first_run()
    if crash_note:
        ctl.crash_note = f"{crash_note} ({crashlog.crash_log_path()})"
        callbacks.notify("Project Away", ctl.crash_note)
    if sys.platform == "darwin":
        ask_mac_permissions(callbacks)

    updates.start()  # reports the last install; first check right away, then every 24 h (unless updates.check: false)
    icon.run()
    if updates.exiting:  # quitting for an update: do not wait on any leftover thread at interpreter exit
        logging.getLogger("away.updater").info("update: tray closed, exiting for the installer")
        logging.shutdown()
        os._exit(0)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
