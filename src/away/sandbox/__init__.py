"""The agent's own desktop: a sandboxed VM session with its own cursor.

Spike (see ``docs/sandbox-vm-spike.md``). The rule this package exists to enforce: input is
injected at the *guest session* level, never through the host input queue, so the host cursor
never moves while the agent works.

Layout:

- :mod:`away.sandbox.detection`: runtime capability detection (pure, unit-tested).
- :mod:`away.sandbox.session`: sandbox session lifecycle (launch / alive / terminate).
- :mod:`away.sandbox.bridge`: screenshot and input bridges at guest-pixel coordinates.
"""

from __future__ import annotations

from away.sandbox.bridge import (
    FakeInputBridge,
    FakeScreenshotBridge,
    Frame,
    InputBridge,
    ScreenshotBridge,
)
from away.sandbox.detection import EditionInfo, detect_edition, hyperv_available, sandbox_available
from away.sandbox.session import FakeSessionBackend, SandboxSession, SessionBackend, SessionConfig, SessionHandle

__all__ = [
    "EditionInfo",
    "FakeInputBridge",
    "FakeScreenshotBridge",
    "FakeSessionBackend",
    "Frame",
    "InputBridge",
    "SandboxSession",
    "ScreenshotBridge",
    "SessionBackend",
    "SessionConfig",
    "SessionHandle",
    "detect_edition",
    "hyperv_available",
    "sandbox_available",
]
