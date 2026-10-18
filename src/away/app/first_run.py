"""Guided first run (v0.1.20): ``away tour``, the tray's "Guided first run..." and Settings > Getting started.

Goal: a new user is productive in about five minutes, every step is skippable and everything works with the
defaults. This module is the pure part (no tkinter; the camera only through injectable pieces), unit-tested:

* :class:`FirstRun`: the step list, navigation (next / back / skip), the progress dots and the saved state.
* :class:`FramingCheck`: "sit comfortably" advice from the face box, the hand and the picture brightness.
* :class:`TutorialCoach`: the 2-minute interactive tutorial (point, pinch click, scroll, pause) with live success
  detection from the gesture state machine's events.
* :func:`camera_session`: camera -> hand/face landmarks -> gesture events for the window (nothing is executed).
* :func:`register_step` / :func:`discover_plugins`: other features plug in their own steps. Gesture Studio can
  provide the "zones" step and a "make your own gesture" offer (key ``custom_gesture``); without it the zones
  step is a placeholder shown only when Gesture Studio is installed, and there is no custom-gesture step.

:mod:`away.app.first_run_gui` is the window on top of it.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATE_VERSION = 1
TUTORIAL_BUDGET_S = 120.0
HINT_AFTER_S = 20.0  # a task not done after this long shows its extra hint

# ----------------------------------------------------------------------------------------------- steps


@dataclass
class Step:
    key: str
    title: str  # friendly, short
    body: str  # what to do, in plain words
    button: str = "Next"  # the big main button
    # plug-in steps: ``run()`` opens what the step is about (Gesture Studio, the gesture recorder...)
    run: Callable[[], Any] | None = None
    shown: Callable[[], bool] | None = None  # False = the step is left out (e.g. its feature is not installed)


def default_steps() -> list[Step]:
    return [
        Step(
            "welcome",
            "Welcome to Project Away!",
            "Control your computer with your hands, your head and your voice.\n\nThis takes about five minutes. "
            "You can skip any step: everything already works with the standard settings.",
            "Let's go",
        ),
        Step(
            "camera",
            "Let's find your camera",
            "We'll check that your camera turns on. Make sure nothing covers it.",
            "Check my camera",
        ),
        Step(
            "framing",
            "Sit comfortably",
            "Sit where you normally sit, about an arm's length from the screen. Then raise one hand so the "
            "camera can see it.",
            "Looks good",
        ),
        Step(
            "tutorial",
            "Try it out",
            "Four quick moves: point, pinch to click, scroll and pause. Each one turns green when we see it.",
            "Next",
        ),
        Step(
            "comfort",
            "Make it comfortable",
            "Optional helpers. You can change them any time in Settings > Getting started.",
            "Next",
        ),
        Step(
            "gaze",
            "Teach it where you look (optional)",
            "Look at a few dots on your screen so Project Away knows where your screens are. It takes about "
            "a minute and lets the pointer jump to where you look.",
            "Start eye setup",
        ),
        Step(
            "agent",
            "Want to connect an AI agent?",
            "Project Away is the bridge between you and your AI agents. Connect one now and it can see your "
            "screen, answer questions and run errands for you.\n\nOnly when you allow it: nothing is shared "
            "until you press Allow. You can always do this later from the tray menu.",
            "Connect one now",
        ),
        Step(
            "done",
            "You're all set!",
            "Project Away lives in the tray, next to the clock.\n\n"
            "  - Show an open palm to pause or resume.\n"
            "  - Press F9 to turn gestures on or off.\n"
            "  - Run this guide again from the tray menu or Settings > Getting started.",
            "Finish",
        ),
    ]


# Plug-in steps, as (step, after_key). Gesture Studio: "zones" after "framing", "custom_gesture" after "gaze".
_PLUGIN_STEPS: list[tuple[Step, str]] = []
PLUGIN_MODULES = ("away.studio.first_run",)  # imported if present; each may call register_step()


def register_step(step: Step, after: str) -> None:
    """Add a step after the step ``after`` (replacing an earlier registration with the same key)."""
    _PLUGIN_STEPS[:] = [(s, a) for s, a in _PLUGIN_STEPS if s.key != step.key]
    _PLUGIN_STEPS.append((step, after))


def unregister_step(key: str) -> None:
    _PLUGIN_STEPS[:] = [(s, a) for s, a in _PLUGIN_STEPS if s.key != key]


def discover_plugins(modules: Sequence[str] = PLUGIN_MODULES) -> list[str]:
    """Import the optional plug-in modules that exist (they register their steps). Returns the ones loaded."""
    loaded = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:  # not in this version, or broken: the guide works without it
            continue
        loaded.append(name)
    return loaded


def studio_installed() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("away.studio") is not None
    except (ImportError, ValueError):
        return False


def zone_placeholder(available: Callable[[], bool] = studio_installed) -> Step:
    """The "zones" step until Gesture Studio registers its own: shown only when Gesture Studio is installed;
    its button opens Settings (where Gesture Studio lives)."""

    def open_settings() -> Any:
        import subprocess

        from away.app.frozen import away_command

        return subprocess.Popen(away_command(["settings"]))

    return Step(
        "zones",
        "Choose where gestures work (optional)",
        "In Gesture Studio you can mark parts of the camera picture where gestures are ignored, for example "
        "where your coffee cup or keyboard is.",
        "Open Gesture Studio",
        run=open_settings,
        shown=available,
    )


def build_steps(plugins: bool = True, zones_available: Callable[[], bool] = studio_installed) -> list[Step]:
    steps = default_steps()
    registered = list(_PLUGIN_STEPS) if plugins else []
    extra: list[tuple[Step, str]] = []
    if not any(s.key == "zones" for s, _ in registered):
        extra.append((zone_placeholder(zones_available), "framing"))
    for step, after in extra + registered:
        keys = [s.key for s in steps]
        steps.insert(keys.index(after) + 1 if after in keys else len(steps) - 1, step)
    return [s for s in steps if s.shown is None or _safe(s.shown)]


def _safe(fn: Callable[[], bool]) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


DONE, SKIPPED, FAILED, TODO, CURRENT = "done", "skipped", "failed", "todo", "current"


def step_result(key: str, step_ok: bool, tutorial_done: int = 0) -> str:
    """What pressing the big button on a step means: the camera and eye setup count as done only once they
    worked, the tutorial once at least one move was seen; anything else the user moved on from is done."""
    if key in ("camera", "gaze"):
        return DONE if step_ok else SKIPPED
    if key == "tutorial":
        return DONE if tutorial_done > 0 else SKIPPED
    return DONE


def state_file(folder: Path | None = None) -> Path:
    if folder is None:
        from away import paths

        folder = paths.config_dir()
    return Path(folder) / "first_run.json"


def load_state(path: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads((path or state_file()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def should_auto_open(path: Path | None = None) -> bool:
    """Open the guide by itself on the very first start only (never again once it was opened)."""
    return not load_state(path).get("seen")


def mark_seen(path: Path | None = None) -> None:
    """Remember the guide was offered, so the tray opens it by itself only once."""
    p = path or state_file()
    data = load_state(p)
    data.update({"version": STATE_VERSION, "seen": True})
    with contextlib.suppress(OSError):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


class FirstRun:
    def __init__(self, steps: list[Step] | None = None, state_path: Path | None = None) -> None:
        self.steps = steps if steps is not None else build_steps()
        self.state_path = state_path
        self.index = 0
        self.results: dict[str, str] = {s.key: TODO for s in self.steps}

    @property
    def step(self) -> Step:
        return self.steps[self.index]

    @property
    def at_end(self) -> bool:
        return self.index == len(self.steps) - 1

    def dots(self) -> list[tuple[str, str]]:
        """Progress dots: ``(key, status)``, status done | skipped | failed | todo | current."""
        return [(s.key, CURRENT if i == self.index else self.results[s.key]) for i, s in enumerate(self.steps)]

    def position(self) -> tuple[int, int]:
        return self.index + 1, len(self.steps)

    def mark(self, result: str) -> None:
        self.results[self.step.key] = result

    def next(self, result: str = DONE) -> Step:
        """Finish the current step with ``result`` and move on (stays on the last step)."""
        if result == DONE or self.results[self.step.key] in (TODO, FAILED):
            self.mark(result)
        if not self.at_end:
            self.index += 1
        self.save()
        return self.step

    def skip(self) -> Step:
        return self.next(SKIPPED)

    def back(self) -> Step:
        if self.index > 0:
            self.index -= 1
        return self.step

    def go(self, key: str) -> Step:
        self.index = [s.key for s in self.steps].index(key)
        return self.step

    def summary(self) -> str:
        middle = [k for k in self.results if k not in ("welcome", "done")]
        done = sum(1 for k in middle if self.results[k] == DONE)
        skipped = sum(1 for k in middle if self.results[k] == SKIPPED)
        text = f"{done} of {len(middle)} steps done"
        return text + (f", {skipped} skipped (you can do them later)" if skipped else "")

    def finish(self) -> None:
        self.mark(DONE)
        self.save(completed=True)

    def save(self, completed: bool | None = None) -> None:
        data = {
            "version": STATE_VERSION,
            "seen": True,
            "completed": bool(completed if completed is not None else load_state(self.state_path).get("completed")),
            "results": dict(self.results),
        }
        p = self.state_path or state_file()
        with contextlib.suppress(OSError):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- framing

FACE_CLOSE = 0.60  # face landmarks span more than this share of the picture = too close
FACE_FAR = 0.15  # less than this = too far (about 1.3 m with a 70° webcam)
OFF_CENTER = 0.25
DARK = 50.0  # mean picture brightness (0..255)
BRIGHT = 225.0
EDGE = 0.02  # a hand landmark this close to the picture edge = the hand is cut off
STABLE_S = 1.5


@dataclass
class FramingResult:
    tips: list[str] = field(default_factory=list)
    ok: bool = False  # nothing to fix and a hand is in view
    stable: bool = False  # ok for STABLE_S seconds in a row


def framing_tips(
    face_box: tuple[float, float, float] | None,
    hand: Any,
    brightness: float | None,
    face_tracking: bool = True,
) -> list[str]:
    """Friendly tips, most important first ([] = all good). ``face_box`` = (cx, cy, size) normalized."""
    tips: list[str] = []
    if brightness is not None and brightness < DARK:
        tips.append("It's a bit dark. Turn on a light or face a window.")
    elif brightness is not None and brightness > BRIGHT:
        tips.append("It's very bright behind you. Try closing a curtain or turning a little.")
    if face_tracking:
        if face_box is None:
            tips.append("We can't see your face yet. Sit in front of the camera.")
        else:
            cx, _cy, size = face_box
            if size > FACE_CLOSE:
                tips.append("You're a bit close. Lean back a little.")
            elif size < FACE_FAR:
                tips.append("You're a bit far away. Come a little closer.")
            if abs(cx - 0.5) > OFF_CENTER:
                tips.append("Move a little toward the middle of the picture.")
    if hand is None:
        tips.append("Now raise one hand so the camera can see it.")
    else:
        import numpy as np

        lm = np.asarray(hand, dtype=float)
        xs, ys = lm[:, 0], lm[:, 1]
        if xs.min() < EDGE or xs.max() > 1 - EDGE or ys.min() < EDGE or ys.max() > 1 - EDGE:
            tips.append("Keep your whole hand inside the picture.")
    return tips


class FramingCheck:
    """Tips per frame, plus ``stable`` once everything has been fine for :data:`STABLE_S` seconds. The face box
    is only measured every few frames, so the last one is kept for :data:`FACE_KEEP_S`."""

    FACE_KEEP_S = 0.5

    def __init__(self, face_tracking: bool = True) -> None:
        self.face_tracking = face_tracking
        self._ok_since: float | None = None
        self._face: Any = None
        self._face_t: float | None = None
        self._bright: float | None = None

    def update(
        self, t: float, face_box: Any, hand: Any, brightness: float | None, face_checked: bool = True
    ) -> FramingResult:
        if face_checked:
            self._face, self._face_t = face_box, t
        elif self._face_t is not None and t - self._face_t > self.FACE_KEEP_S:
            self._face = None
        if brightness is not None:
            self._bright = brightness
        tips = framing_tips(self._face, hand, self._bright, self.face_tracking)
        ok = not tips
        if ok:
            if self._ok_since is None:
                self._ok_since = t
        else:
            self._ok_since = None
        stable = ok and self._ok_since is not None and t - self._ok_since >= STABLE_S
        return FramingResult(tips, ok, stable)


def brightness_of(rgb: Any) -> float | None:
    """Mean brightness of an RGB frame (subsampled; cheap)."""
    try:
        import numpy as np

        return float(np.asarray(rgb)[::8, ::8].mean())
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------- tutorial


@dataclass
class Task:
    key: str
    title: str
    how: str
    hint: str
    done: bool = False
    skipped: bool = False
    progress: float = 0.0  # 0..1


def tutorial_tasks() -> list[Task]:
    return [
        Task(
            "move",
            "Point",
            "Point your index finger and move your hand. Touch the three circles with the dot.",
            "Move slowly. If the dot is shaky, come a little closer to the camera or add light.",
        ),
        Task(
            "click",
            "Pinch to click",
            "Touch your thumb and index finger together, then let go. That's a click!",
            "Make the pinch quick and clear, like picking up a grain of rice.",
        ),
        Task(
            "scroll",
            "Scroll",
            "Raise four fingers (tuck your thumb in) and move your hand up or down.",
            "Or hold up two fingers (like a peace sign) and move them up and down. A fist stops scrolling.",
        ),
        Task(
            "pause",
            "Pause and resume",
            "Show an open palm and hold it still for a moment to pause. Do it again to resume.",
            "Spread all five fingers and keep your hand still for about half a second.",
        ),
    ]


MOVE_TARGETS = ((0.2, 0.3), (0.8, 0.3), (0.5, 0.75))
TARGET_RADIUS = 0.12
REGION = (0.15, 0.15, 0.85, 0.85)  # the part of the camera picture mapped onto the practice area
SCROLL_GOAL = 6.0  # wheel notches of scrolling


def practice_point(x: float, y: float, region: tuple[float, float, float, float] = REGION) -> tuple[float, float]:
    """Camera fingertip (normalized) -> position in the practice area (0..1, clamped)."""
    x0, y0, x1, y1 = region
    px = (x - x0) / max(1e-6, x1 - x0)
    py = (y - y0) / max(1e-6, y1 - y0)
    return min(1.0, max(0.0, px)), min(1.0, max(0.0, py))


class TutorialCoach:
    """Feed it gesture events (``kind``, ``data``, ``t``); it ticks off tasks. Tasks are shown in order but any
    task seen early still counts (someone who scrolls first is not made to do it again)."""

    def __init__(self, budget_s: float = TUTORIAL_BUDGET_S) -> None:
        self.tasks = tutorial_tasks()
        self.budget_s = budget_s
        self.started: float | None = None
        self.task_started: float | None = None
        self.dot: tuple[float, float] | None = None
        self.targets_hit: list[bool] = [False] * len(MOVE_TARGETS)
        self.scrolled = 0.0
        self.paused_once = False
        self.flash = ""  # a short "Nice!" message after a success

    @property
    def current(self) -> Task | None:
        return next((t for t in self.tasks if not (t.done or t.skipped)), None)

    @property
    def complete(self) -> bool:
        return self.current is None

    def start(self, t: float) -> None:
        self.started = self.task_started = t

    def remaining(self, t: float) -> float:
        return self.budget_s if self.started is None else max(0.0, self.budget_s - (t - self.started))

    def show_hint(self, t: float) -> bool:
        return self.task_started is not None and t - self.task_started >= HINT_AFTER_S

    def skip(self, t: float) -> None:
        cur = self.current
        if cur is not None:
            cur.skipped = True
            self.task_started = t

    def _task(self, key: str) -> Task:
        return next(x for x in self.tasks if x.key == key)

    def _finish(self, key: str, t: float, msg: str) -> None:
        task = self._task(key)
        if task.done or task.skipped:
            return
        task.done, task.progress = True, 1.0
        self.flash = msg
        self.task_started = t

    def feed(self, kind: str, data: dict[str, Any], t: float) -> None:
        if self.started is None:
            self.start(t)
        if kind == "move" and data.get("anchor") != "pinch" and "x" in data:
            self.dot = practice_point(float(data["x"]), float(data["y"]))
            for i, (tx, ty) in enumerate(MOVE_TARGETS):
                if not self.targets_hit[i] and math.hypot(self.dot[0] - tx, self.dot[1] - ty) <= TARGET_RADIUS:
                    self.targets_hit[i] = True
            self._task("move").progress = sum(self.targets_hit) / len(MOVE_TARGETS)
            if all(self.targets_hit):
                self._finish("move", t, "Nice pointing!")
        elif kind in ("click", "double_click"):
            self._finish("click", t, "Click! Well done.")
        elif kind == "scroll":
            self._add_scroll(abs(float(data.get("dy", 0.0) or 0.0)), t)  # wheel notches (scroll_gain applied)
        elif kind == "palm_scroll":
            self._add_scroll(abs(float(data.get("dy", 0.0) or 0.0)) + abs(float(data.get("dx", 0.0) or 0.0)), t)
        elif kind == "autoscroll" and data.get("direction"):
            self._add_scroll(0.2, t)  # reported every frame while auto-scrolling: about a second is enough
        elif kind == "clutch":
            if data.get("paused"):
                self.paused_once = True
                pause = self._task("pause")
                if not pause.done:
                    pause.progress = 0.5
                    self.flash = "Paused. Now do it again to resume."
            elif self.paused_once:
                self._finish("pause", t, "Resumed. You've got it!")

    def _add_scroll(self, amount: float, t: float) -> None:
        self.scrolled += amount
        self._task("scroll").progress = min(1.0, self.scrolled / SCROLL_GOAL)
        if self.scrolled >= SCROLL_GOAL:
            self._finish("scroll", t, "Smooth scrolling!")

    def score(self) -> tuple[int, int]:
        return sum(1 for x in self.tasks if x.done), len(self.tasks)


# ----------------------------------------------------------------------------------------------- camera


def camera_check(cfg: Any, prober: Callable[..., tuple[bool, str]] | None = None) -> tuple[bool, str]:
    """Quick "does the camera turn on" check (the probe ``away setup`` uses). ``(ok, raw message)``."""
    if prober is None:  # pragma: no cover - real camera
        from away.hands.tracker import probe_camera

        prober = probe_camera
    h = cfg.hands
    device = h.cameras[0].device if h.cameras else h.camera_index
    try:
        return prober(device, hint=h.camera_hint)
    except Exception as exc:
        return False, str(exc)


PREVIEW_EVERY = 2  # send every 2nd frame's picture to the window
PREVIEW_SHRINK = 4  # 640x480 -> 160x120
FACE_EVERY = 3  # the face is measured on every 3rd frame (framing tips only)


def camera_session(
    cfg: Any,
    out: Any,
    stop: Any,
    *,
    share: Any = None,
    frames: Any = None,
    tracker_factory: Callable[[], Any] | None = None,
    face_factory: Callable[[], Any] | None = None,
    lease_timeout: float = 4.0,
) -> None:
    """Camera -> hand (+ face) landmarks -> gesture events, posted to ``out`` as dicts for the guide window:
    ``{"t", "hand", "face_box", "face_checked", "face_tracking", "brightness", "events": [(kind, data)],
    "image"}``; on failure ``{"error": text}``. Nothing is executed: the practice area only shows what would
    happen. Like Settings > Test gesture, the camera is borrowed from the running app and given back after."""
    from away.app.camera_share import LEASE_IDLE_S, CameraLease, CameraShare, camera_error_message
    from away.hands.gestures import GestureStateMachine

    h = cfg.hands
    idx = h.cameras[0].device if h.cameras else h.camera_index
    if frames is None:  # pragma: no cover - real camera
        from away.hands.tracker import camera_frames as frames
    if tracker_factory is None:  # pragma: no cover - real MediaPipe

        def tracker_factory() -> Any:
            from away import paths
            from away.hands.tracker import HandTracker

            mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
            return HandTracker(mdir / "hand_landmarker.task", h.max_hands, prefer=h.roles.mode, mirrored=h.mirror)

    if face_factory is None:  # pragma: no cover - real MediaPipe

        def face_factory() -> Any:
            from away import paths
            from away.face.landmarker import FaceLandmarker

            mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
            model = mdir / "face_landmarker.task"
            return FaceLandmarker(model, max_faces=1) if cfg.face.enabled and model.exists() else None

    lease = CameraLease(share or CameraShare(), timeout=lease_timeout, who="first-run", idle_s=LEASE_IDLE_S, stop=stop)
    tracker = faces = None
    try:
        lease.open()
        tracker = tracker_factory()
        try:
            faces = face_factory()
        except Exception:
            faces = None  # the framing tips about the face are then left out
        sm = GestureStateMachine(h.gestures)
        for n, (t, rgb) in enumerate(frames(idx, h.mirror)):
            if stop.is_set():
                break
            lease.touch()  # v0.1.21: frames are flowing (a stuck preview gives the camera back)
            fr = tracker.process(rgb, t)
            events = [(ev.kind, dict(ev.data)) for ev in sm.update(t, fr.landmarks)]
            checked = faces is not None and n % FACE_EVERY == 0
            box = None
            if checked and faces is not None:
                obs = faces.process(rgb, t)
                box = tuple(obs[0].box) if obs else None
            msg: dict[str, Any] = {
                "t": t,
                "hand": None if fr.landmarks is None else [[float(v) for v in p[:2]] for p in fr.landmarks],
                "face_box": box,
                "face_checked": checked,
                "face_tracking": faces is not None,
                "brightness": brightness_of(rgb) if n % 5 == 0 else None,
                "events": events,
            }
            if n % PREVIEW_EVERY == 0:
                with contextlib.suppress(Exception):
                    msg["image"] = rgb[::PREVIEW_SHRINK, ::PREVIEW_SHRINK].copy()
            out.put(msg)
    except Exception as exc:
        out.put({"error": camera_error_message(exc, idx, lease.app_running, lease.handed_over)})
    finally:
        for obj in (tracker, faces):
            if obj is not None:
                with contextlib.suppress(Exception):
                    obj.close()
        lease.close()


def tracking_ready(cfg: Any, find_spec: Callable[[str], Any] | None = None) -> tuple[bool, str]:
    """Can the live steps run? (OpenCV + MediaPipe installed, hand model downloaded). ``(ok, raw message)``;
    the raw message goes through :func:`away.comfort.ui.friendly_error` for the user."""
    import importlib.util

    spec = find_spec or importlib.util.find_spec
    for mod in ("cv2", "mediapipe"):
        try:
            if spec(mod) is None:
                return False, f"{mod} not installed (pip install 'project-away[hands]')"
        except (ImportError, ValueError):
            return False, f"{mod} not installed"
    from away import paths

    mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
    if not (mdir / "hand_landmarker.task").exists():
        return False, "hand model missing (run `away setup`)"
    return True, "ready"


def download_models(
    cfg: Any, out: Callable[[str], None] = print, downloader: Callable[..., Any] | None = None
) -> str | None:
    """The friendly "Download now" fix: fetch the MediaPipe models ``away setup`` downloads. Error text or None."""
    from away import paths
    from away.app import models

    mdir = Path(cfg.models_dir) if cfg.models_dir else paths.models_dir()
    fetch = downloader or models.download
    for mf in models.MEDIAPIPE_MODELS:
        try:
            fetch(mf, mdir, out)
        except Exception as exc:
            return f"download failed: {exc}"
    return None


def text_tour(
    cfg: Any,
    out: Callable[[str], None] = print,
    prober: Callable[..., tuple[bool, str]] | None = None,
    find_spec: Callable[[str], Any] | None = None,
) -> int:
    """``away tour --text``: the parts of the guide that need no window (camera check, readiness, tips).
    Returns 0 when the camera and hand tracking are ready, 1 otherwise."""
    from away.comfort import ui
    from away.comfort.settings import PLAIN_SWITCHES, get_path

    out("Welcome to Project Away! Let's check your setup.\n")
    ok, raw = camera_check(cfg, prober)
    ready, raw2 = tracking_ready(cfg, find_spec)
    for good, text, done in ((ok, raw, "Your camera works!"), (ready, raw2, "Hand tracking is ready.")):
        if good:
            out(f"  [ok] {done}")
        else:
            err = ui.friendly_error(text)
            out(f"  [!!] {err.title}\n       {err.message}\n       ({text})")
    out("\nHow to use it:")
    for task in tutorial_tasks():
        out(f"  - {task.title}: {task.how}")
    out("\nComfort helpers (change them in Settings > Getting started):")
    for path, label, _help in PLAIN_SWITCHES:
        out(f"  [{'x' if get_path(cfg, path) else ' '}] {label}")
    out("\nStart with `away tray`. For the full interactive tutorial, run `away tour` on your desktop.")
    mark_seen()
    return 0 if ok and ready else 1
