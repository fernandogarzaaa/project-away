"""Screenshot and input bridges to the agent desktop, at guest-pixel coordinates.

The rule: input is injected at the *guest session* level (by the in-guest bridge started via
``LogonCommand``, or a VNC server in the guest). Nothing here may go through the host input
queue: that would move the host cursor and violate the spike's whole purpose.

Coordinates are always **guest pixels**. DPI scaling is applied once, at the bridge boundary,
never ad hoc. Tests drive both bridges through fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Frame:
    """One captured guest desktop frame."""

    width: int
    height: int
    pixels: bytes  # RGB, row-major, width*height*3 bytes
    cursor_x: int = -1  # guest cursor position when known, else -1
    cursor_y: int = -1

    def valid(self) -> bool:
        return self.width > 0 and self.height > 0 and len(self.pixels) == self.width * self.height * 3


class ScreenshotBridge(Protocol):
    """Capture the guest desktop. The guest cursor is part of the frame or reported."""

    def capture(self) -> Frame: ...


class InputBridge(Protocol):
    """Inject input into the guest session. All coordinates are guest pixels."""

    def move(self, x: int, y: int) -> None: ...
    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> None: ...
    def press(self, x: int, y: int, button: str = "left") -> None: ...
    def release(self, x: int, y: int, button: str = "left") -> None: ...
    def scroll(self, x: int, y: int, dx: float, dy: float) -> None: ...
    def key(self, name: str) -> None: ...
    def type_text(self, text: str) -> None: ...


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Clamp a guest-pixel point into the frame. Pure; unit-tested."""
    return (max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1)))


@dataclass
class FakeScreenshotBridge:
    """Canned frames for tests. ``frames`` cycles; the cursor marches diagonally."""

    width: int = 320
    height: int = 200
    frames: int = 3
    calls: int = 0

    def capture(self) -> Frame:
        n = self.calls
        self.calls += 1
        pixels = bytes((n * 40 + i) % 256 for i in range(self.width * self.height * 3))
        return Frame(
            width=self.width,
            height=self.height,
            pixels=pixels,
            cursor_x=(n * 17) % self.width,
            cursor_y=(n * 11) % self.height,
        )


@dataclass
class FakeInputBridge:
    """Records every input call for tests. The host cursor is never touched (there is none)."""

    width: int = 320
    height: int = 200
    calls: list[tuple[str, tuple[int, ...] | tuple[str, ...]]] = field(default_factory=list)

    def _at(self, x: int, y: int) -> tuple[int, int]:
        return clamp_point(x, y, self.width, self.height)

    def move(self, x: int, y: int) -> None:
        self.calls.append(("move", self._at(x, y)))

    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> None:
        self.calls.append(("click", (*self._at(x, y), button, count)))

    def press(self, x: int, y: int, button: str = "left") -> None:
        self.calls.append(("press", (*self._at(x, y), button)))

    def release(self, x: int, y: int, button: str = "left") -> None:
        self.calls.append(("release", (*self._at(x, y), button)))

    def scroll(self, x: int, y: int, dx: float, dy: float) -> None:
        self.calls.append(("scroll", (*self._at(x, y), dx, dy)))

    def key(self, name: str) -> None:
        self.calls.append(("key", (name,)))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", (text,)))


@dataclass(frozen=True)
class GuestBridgeConfig:
    """How the host reaches the in-guest bridge started by LogonCommand."""

    host: str  # guest IP on the sandbox virtual network (measured on the Acer)
    port: int = 8787
    timeout_s: float = 5.0


class GuestBridge:
    """The recommended bridge (spike interface; transport validated on the Acer).

    The guest runs a small HTTP service (started by ``LogonCommand``):
    ``GET /frame`` returns a Frame, ``POST /input`` takes an action batch.
    The guest service reuses away's own screen-capture and input code inside the
    guest session, so the *guest* cursor moves and the host cursor never does.
    """

    def __init__(self, config: GuestBridgeConfig) -> None:
        self.config = config

    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    # The transport methods are intentionally unimplemented: step 4/5 of the on-Acer
    # checklist decides the transport (direct TCP, vsock, or port forward) before code lands.
    def capture(self) -> Frame:  # pragma: no cover - needs the Acer
        raise NotImplementedError("GuestBridge transport is validated on the Acer first")

    def move(self, x: int, y: int) -> None:  # pragma: no cover - needs the Acer
        raise NotImplementedError("GuestBridge transport is validated on the Acer first")


class VncBridge:
    """Fallback bridge: a VNC server in the guest, Python VNC client on the host.

    Mature and well understood (framebuffer + input at protocol level, host cursor
    untouched), but heavier than the guest agent. Kept as the escape hatch if the
    custom bridge transport proves unworkable.
    """

    def __init__(self, host: str, port: int = 5900) -> None:
        self.host = host
        self.port = port

    def capture(self) -> Frame:  # pragma: no cover - needs the Acer
        raise NotImplementedError("VncBridge is a spike stub; validate on the Acer first")

    def move(self, x: int, y: int) -> None:  # pragma: no cover - needs the Acer
        raise NotImplementedError("VncBridge is a spike stub; validate on the Acer first")


__all__ = [
    "FakeInputBridge",
    "FakeScreenshotBridge",
    "Frame",
    "GuestBridge",
    "GuestBridgeConfig",
    "InputBridge",
    "ScreenshotBridge",
    "VncBridge",
    "clamp_point",
]
