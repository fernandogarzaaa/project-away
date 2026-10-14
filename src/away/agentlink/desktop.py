"""Desktop access for agents: screenshots, the focused window, UI elements (Windows UI Automation), and the
secure-desktop check. Every function degrades to "not available" instead of raising when the platform or an
optional package is missing.

* Screenshots: Pillow's ``ImageGrab`` (Windows, macOS; Linux under X11).
* UI elements: the optional ``uiautomation`` package on Windows (installed with the app's ``[agent]`` extra).
  Without it, agents act by coordinates only and typing is refused on Windows (password fields cannot be
  detected).
* Secure desktop: on Windows, the input desktop is not ``Default`` while a UAC prompt / Ctrl+Alt+Del screen is
  up; agents are blocked then (Windows also refuses synthetic input there).
"""

from __future__ import annotations

import io
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

TERMINAL_PROCESSES = re.compile(
    r"^(cmd|powershell|pwsh|windowsterminal|wt|conhost|mintty|bash|zsh|sh|fish|terminal|iterm2|alacritty|"
    r"wezterm|kitty|xterm|gnome-terminal(-server)?|konsole|regedit|mmc|taskmgr)(\.exe)?$",
    re.IGNORECASE,
)
SECURE_CLASSES = ("Credential Dialog Xaml Host", "$$$Secure UAP Dummy Window Class For Interim Dialog")
PASSWORD_TITLE = re.compile(r"password|passcode|\bpin\b|sign[ -]?in|log[ -]?in|unlock|credential", re.IGNORECASE)


@dataclass
class Element:
    name: str
    control_type: str = ""
    automation_id: str = ""
    is_password: bool = False
    enabled: bool = True
    rect: tuple[int, int, int, int] | None = None  # left, top, right, bottom (screen pixels)
    children: list[Element] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int] | None:
        if self.rect is None:
            return None
        left, top, right, bottom = self.rect
        return (left + right) // 2, (top + bottom) // 2

    def to_dict(self, depth: int = 0) -> dict[str, Any]:
        d = asdict(self)
        d.pop("children")
        d["center"] = self.center
        if depth > 0 and self.children:
            d["children"] = [c.to_dict(depth - 1) for c in self.children]
        return d


# ------------------------------------------------------------------------------------------------ screen
def grab_screen() -> Any:  # pragma: no cover - needs a display
    """A PIL image of the whole desktop, or None."""
    try:
        from PIL import ImageGrab

        if sys.platform == "win32":
            return ImageGrab.grab(all_screens=True)
        return ImageGrab.grab()
    except Exception:
        return None


def to_jpeg(img: Any, max_width: int = 960, quality: int = 60) -> tuple[bytes, float]:
    """Downscale a PIL image or an RGB numpy array to ``max_width`` and encode JPEG. Returns (bytes, scale)."""
    from PIL import Image

    if not hasattr(img, "getbbox"):  # a numpy RGB array, not a PIL image
        img = Image.fromarray(img)
    w, h = img.size
    scale = min(1.0, float(max_width) / float(w)) if w else 1.0
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue(), scale


def thumbnail_png(img: Any, max_width: int = 200) -> bytes:
    from PIL import Image

    if not hasattr(img, "getbbox"):  # a numpy RGB array, not a PIL image
        img = Image.fromarray(img)
    img = img.copy()
    img.thumbnail((max_width, max_width))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ windows
def foreground_window() -> dict[str, Any] | None:  # pragma: no cover - platform specific
    """Title, process name and window class of the focused window (Windows); None elsewhere."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        proc = ""
        h = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            name = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, name, ctypes.byref(size)):
                proc = name.value.replace("\\", "/").rsplit("/", 1)[-1]
            kernel32.CloseHandle(h)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return {
            "title": buf.value,
            "process": proc,
            "class": cls.value,
            "rect": [rect.left, rect.top, rect.right, rect.bottom],
        }
    except Exception:
        return None


class SecureDesktopUnknown(RuntimeError):
    """The secure-desktop check itself failed (v0.1.24: agents are then stopped, never let through)."""


def secure_desktop_active() -> bool:
    """True while Windows shows the secure desktop (UAC, Ctrl+Alt+Del, lock screen) or a credential dialog.
    Raises :class:`SecureDesktopUnknown` when it cannot tell (the caller fails closed)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hdesk = user32.OpenInputDesktop(0, False, 0x0001)  # DESKTOP_READOBJECTS
        if not hdesk:
            return True  # cannot even open it: the secure desktop
        try:
            name = ctypes.create_unicode_buffer(256)
            needed = ctypes.c_ulong()
            user32.GetUserObjectInformationW(hdesk, 2, name, 512, ctypes.byref(needed))  # UOI_NAME
            if name.value.lower() != "default":
                return True
        finally:
            user32.CloseDesktop(hdesk)
        fg = foreground_window() or {}
        return fg.get("class") in SECURE_CLASSES or str(fg.get("process", "")).lower() == "consent.exe"
    except Exception as exc:
        raise SecureDesktopUnknown(str(exc)) from exc


def is_terminal(process: str | None) -> bool:
    return bool(process) and bool(TERMINAL_PROCESSES.match(str(process).strip()))


# ------------------------------------------------------------------------------------------------ UIA
class NullUIA:
    available = False
    reason = "reading on-screen elements needs Windows and the 'uiautomation' package"

    def focused(self) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        return None

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None

    def tree(self, max_depth: int = 3, max_nodes: int = 200) -> Element | None:
        return None


class WindowsUIA:  # pragma: no cover - needs Windows + uiautomation
    available = True
    reason = ""

    def __init__(self) -> None:
        import uiautomation

        self.ua = uiautomation

    def _el(self, c: Any) -> Element | None:
        if c is None:
            return None
        try:
            r = c.BoundingRectangle
            rect = (int(r.left), int(r.top), int(r.right), int(r.bottom)) if r and r.width() > 0 else None
            pwd = False
            try:
                pwd = bool(c.GetPropertyValue(self.ua.PropertyId.IsPasswordProperty))
            except Exception:
                pwd = False
            return Element(
                name=str(c.Name or "")[:200],
                control_type=str(c.ControlTypeName or ""),
                automation_id=str(c.AutomationId or "")[:100],
                is_password=pwd,
                enabled=bool(c.IsEnabled),
                rect=rect,
            )
        except Exception:
            return None

    def focused(self) -> Element | None:
        return self._el(self.ua.GetFocusedControl())

    def element_at(self, x: int, y: int) -> Element | None:
        return self._el(self.ua.ControlFromPoint(int(x), int(y)))

    def _walk(self, root: Any, max_depth: int, max_nodes: int) -> list[tuple[Any, int]]:
        out: list[tuple[Any, int]] = []
        stack = [(root, 0)]
        while stack and len(out) < max_nodes:
            c, d = stack.pop()
            out.append((c, d))
            if d < max_depth:
                try:
                    kids = c.GetChildren()
                except Exception:
                    kids = []
                stack.extend((k, d + 1) for k in reversed(kids))
        return out

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        root = self.ua.GetForegroundControl()
        if root is None:
            return None
        cands = [self._el(c) for c, _d in self._walk(root, 12, 3000)]
        return best_match([e for e in cands if e is not None], name, control_type)

    def tree(self, max_depth: int = 3, max_nodes: int = 200) -> Element | None:
        root = self.ua.GetForegroundControl()
        if root is None:
            return None
        top = self._el(root)
        if top is None:
            return None
        built: dict[int, Element] = {}
        for c, d in self._walk(root, max_depth, max_nodes):
            e = top if d == 0 else self._el(c)
            if e is None:
                continue
            built[d] = e
            if d > 0 and d - 1 in built:
                built[d - 1].children.append(e)
        return top


class SmartPointerUIA:
    """UI Automation for agents that reuses the smart pointer (v0.1.20): ``find`` first looks in the smart
    pointer's element cache (one fast ``FindAllBuildCache`` walk of the foreground window, refreshed on demand)
    with its fuzzy spoken-name matcher, and falls back to ``base`` (the full ``uiautomation`` walk). Everything
    else - the focused element, the element at a point, the tree, password detection - comes from ``base``."""

    def __init__(self, smart: Any, base: Any) -> None:
        self.smart = smart
        self.base = base

    @property
    def available(self) -> bool:
        return bool(getattr(self.base, "available", False)) or self._smart_ok()

    @property
    def reason(self) -> str:
        return str(getattr(self.base, "reason", ""))

    def _smart_ok(self) -> bool:
        try:
            return bool(self.smart.available)
        except Exception:
            return False

    def focused(self) -> Element | None:
        return self.base.focused()  # type: ignore[no-any-return]

    def element_at(self, x: int, y: int) -> Element | None:
        return self.base.element_at(x, y)  # type: ignore[no-any-return]

    def tree(self, max_depth: int = 3, max_nodes: int = 200) -> Element | None:
        return self.base.tree(max_depth, max_nodes)  # type: ignore[no-any-return]

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        if self._smart_ok():
            try:
                from away.pointer.looksay import match_names

                snap = self.smart.cache.refresh_now()
                targets = [t for t in snap.targets if not control_type or _same_type(t.kind, control_type)]
                matches = match_names(name, targets)
                if matches:  # ties are listed top-left first: an agent gets the first, a person would get badges
                    t = matches[0].target
                    left, top, right, bottom = (int(v) for v in t.rect)
                    return Element(name=t.name, control_type=t.kind, rect=(left, top, right, bottom))
            except Exception:
                pass
        return self.base.find(name, control_type)  # type: ignore[no-any-return]


def _same_type(kind: str, control_type: str) -> bool:
    return control_type.lower().replace("control", "") in str(kind).lower()


def best_match(elements: list[Element], name: str, control_type: str | None = None) -> Element | None:
    """Exact name (case-insensitive) beats prefix beats substring; enabled and on-screen elements only. With no
    such match, the smart pointer's fuzzy spoken-name score (v0.1.20) picks a close one ("Sned" -> "Send")."""
    want = name.strip().lower()
    if not want:
        return None
    ct = (control_type or "").lower().replace("control", "")
    best: tuple[int, int, Element] | None = None
    for e in elements:
        if not e.enabled or e.rect is None or not e.name:
            continue
        if ct and ct not in e.control_type.lower():
            continue
        n = e.name.strip().lower()
        score = 3 if n == want else 2 if n.startswith(want) else 1 if want in n else 0
        if score == 0:
            continue
        cand = (score, -len(n), e)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is not None:
        return best[2]
    from away.pointer.looksay import MATCH_THRESHOLD, name_score

    fuzzy: tuple[float, Element] | None = None
    for e in elements:
        if not e.enabled or e.rect is None or not e.name:
            continue
        if ct and ct not in e.control_type.lower():
            continue
        sc = name_score(name, e.name)
        if sc >= MATCH_THRESHOLD and (fuzzy is None or sc > fuzzy[0]):
            fuzzy = (sc, e)
    return fuzzy[1] if fuzzy else None


def make_uia() -> Any:
    if sys.platform == "win32":  # pragma: no cover
        try:
            return WindowsUIA()
        except Exception:
            pass
    return NullUIA()
