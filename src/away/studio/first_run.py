"""Gesture Studio's steps in the guided first run (``away tour``, v0.1.20).

:mod:`away.app.first_run` imports this module when it exists (``PLUGIN_MODULES``); importing it registers:

* ``zones`` after ``framing``: "Choose where gestures work" opens Settings on the Gesture Studio tab, where the
  zone map is drawn on the live camera picture (replaces the first run's placeholder).
* ``custom_gesture`` after ``gaze``: "Make your own gesture" opens Gesture Studio's teach-a-gesture guide.

Both are optional and skippable like every step; nothing here runs at import time except the registration.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from away.app.first_run import Step, register_step


def settings_command(teach: bool = False) -> list[str]:
    from away.app.frozen import away_command

    args = ["settings", "--tab", "Gesture Studio"]
    return away_command([*args, "--teach-gesture"] if teach else args)


def _opener(teach: bool, popen: Callable[[Sequence[str]], Any] = subprocess.Popen) -> Callable[[], Any]:
    def run() -> Any:
        return popen(settings_command(teach))

    return run


def zones_step(popen: Callable[[Sequence[str]], Any] = subprocess.Popen) -> Step:
    return Step(
        "zones",
        "Choose where gestures work (optional)",
        "Mark the parts of the camera picture where gestures should be ignored, for example where your coffee "
        "cup or keyboard is. Gesture Studio opens with your camera picture: drag over an area and pick "
        '"Ignore here".',
        "Open Gesture Studio",
        run=_opener(False, popen),
    )


def custom_gesture_step(popen: Callable[[Sequence[str]], Any] = subprocess.Popen) -> Step:
    return Step(
        "custom_gesture",
        "Make your own gesture (optional)",
        "Teach Project Away a gesture of your own, like a thumbs up or a peace sign, and choose what it does. "
        "Show it a few times and Gesture Studio learns it in about a minute.",
        "Teach a gesture",
        run=_opener(True, popen),
    )


def register(popen: Callable[[Sequence[str]], Any] = subprocess.Popen) -> None:
    register_step(zones_step(popen), after="framing")
    register_step(custom_gesture_step(popen), after="gaze")


register()
