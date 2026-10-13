"""Windows UI Automation backend for :mod:`away.pointer.targets` (comtypes; MIT, pure Python, PyInstaller hooks).

One ``FindAllBuildCache`` call per refresh walks the foreground window for the clickable control types and
fetches name, control type and bounding rectangle in the same cross-process round trip (a cache request), so a
refresh is one COM call plus a cheap loop. It runs on the :class:`~away.pointer.targets.TargetCache` thread,
never on the cursor loop. Each thread that touches UIA joins the multithreaded COM apartment first.

Nothing is imported until the backend is used, so importing this module is safe on every platform.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from typing import Any

from away.pointer.targets import MAX_TARGETS, NullBackend, Target, TargetBackend

# UIA property ids
UIA_BOUNDING_RECTANGLE = 30001
UIA_CONTROL_TYPE = 30003
UIA_NAME = 30005
UIA_IS_OFFSCREEN = 30022
TREE_SCOPE_DESCENDANTS = 4
# UIA control type ids -> our kind names (away.pointer.targets.CLICKABLE_TYPES)
CONTROL_TYPES = {
    50000: "button",
    50002: "checkbox",
    50003: "combobox",
    50004: "edit",
    50005: "hyperlink",
    50007: "listitem",
    50011: "menuitem",
    50013: "radiobutton",
    50019: "tabitem",
    50024: "treeitem",
    50029: "dataitem",
    50031: "splitbutton",
}


def availability(platform: str | None = None) -> str:
    """``ok`` | ``not_windows`` | ``missing_comtypes`` (what :mod:`away.pointer.settings_ui` explains)."""
    if not (platform or sys.platform).startswith("win"):
        return "not_windows"
    if importlib.util.find_spec("comtypes") is None:
        return "missing_comtypes"
    return "ok"


def rect_of(r: Any) -> tuple[float, float, float, float]:
    """A comtypes ``tagRECT`` (left/top/right/bottom) or a 4-sequence -> a float tuple."""
    if hasattr(r, "left"):
        return float(r.left), float(r.top), float(r.right), float(r.bottom)
    left, top, right, bottom = r
    return float(left), float(top), float(right), float(bottom)


def targets_from_array(arr: Any, limit: int = MAX_TARGETS) -> list[Target]:
    """Read cached properties out of an ``IUIAutomationElementArray`` (duck-typed: ``Length`` + ``GetElement``)."""
    out: list[Target] = []
    for i in range(int(arr.Length)):
        if len(out) >= limit:
            break
        e = arr.GetElement(i)
        try:
            kind = CONTROL_TYPES.get(int(e.CachedControlType))
            if kind is None:
                continue
            name = str(e.CachedName or "").strip()
            rect = rect_of(e.CachedBoundingRectangle)
        except Exception:  # an element that vanished mid-walk
            continue
        out.append(Target(name, kind, rect, e))
    return out


class UIABackend:
    """Real UI Automation. Tests drive it with a fake ``comtypes`` module; the real thing only runs on Windows."""

    def __init__(self, limit: int = MAX_TARGETS, foreground: Any = None) -> None:
        self.limit = limit
        self._foreground = foreground  # () -> HWND; default: user32.GetForegroundWindow
        self._local = threading.local()
        self._lock = threading.Lock()
        self._uia: Any = None
        self._cond: Any = None
        self._cache: Any = None

    def available(self) -> bool:
        return availability() == "ok"

    def thread_init(self) -> None:
        self._ensure()

    def _com_init(self) -> None:
        if getattr(self._local, "com", False):
            return
        import comtypes

        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        except OSError:
            pass  # this thread already joined an apartment (e.g. a GUI thread): use it as is
        self._local.com = True

    def _ensure(self) -> Any:
        self._com_init()
        with self._lock:
            if self._uia is None:
                import comtypes.client

                mod = comtypes.client.GetModule("UIAutomationCore.dll")
                uia = comtypes.client.CreateObject(mod.CUIAutomation, interface=mod.IUIAutomation)
                types = [uia.CreatePropertyCondition(UIA_CONTROL_TYPE, ct) for ct in CONTROL_TYPES]
                clickable = uia.CreateOrConditionFromArray(types)
                onscreen = uia.CreatePropertyCondition(UIA_IS_OFFSCREEN, False)
                self._cond = uia.CreateAndCondition(clickable, onscreen)
                cache = uia.CreateCacheRequest()
                for prop in (UIA_NAME, UIA_CONTROL_TYPE, UIA_BOUNDING_RECTANGLE):
                    cache.AddProperty(prop)
                self._cache = cache
                self._uia = uia
            return self._uia

    def _foreground_hwnd(self) -> int:
        if self._foreground is not None:
            return int(self._foreground() or 0)
        import ctypes

        return int(ctypes.windll.user32.GetForegroundWindow() or 0)  # type: ignore[attr-defined]

    def foreground_targets(self) -> list[Target]:
        uia = self._ensure()
        hwnd = self._foreground_hwnd()
        if not hwnd:
            return []
        root = uia.ElementFromHandle(hwnd)
        if root is None:
            return []
        arr = root.FindAllBuildCache(TREE_SCOPE_DESCENDANTS, self._cond, self._cache)
        if arr is None:
            return []
        return targets_from_array(arr, self.limit)

    def focus(self, target: Target) -> bool:
        if target.ref is None:
            return False
        self._com_init()
        target.ref.SetFocus()
        return True


def make_backend(platform: str | None = None) -> TargetBackend:
    """The UIA backend on Windows with comtypes, else a :class:`~away.pointer.targets.NullBackend`."""
    state = availability(platform)
    if state != "ok":
        return NullBackend(state)
    return UIABackend()
