"""OS glue for shortcuts: windows under the pointer, UI Automation, the pynput listener and the replay backend.

Everything here talks to the OS and is exercised only on a real desktop; the logic that uses it
(:mod:`away.control.macro_record`, :mod:`away.control.macro_play`) takes these objects injected, so its tests run
with fakes. Windows gets the full feature set through ctypes (no extra packages, nothing for PyInstaller to miss):
window lookup by point/process and UI Automation (control name under a click, finding that control again, and
"is the focused control a password box?"). Elsewhere only the focused app's name is known, so replay uses screen
coordinates and typed text is not recorded unless ``macros.record_text_without_password_check`` is on.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from away.control.macros import app_key


@dataclass
class WindowInfo:
    handle: int  # HWND on Windows, 0 = not known
    pid: int
    process: str  # chrome.exe
    title: str
    cls: str  # window class (Shell_TrayWnd for the taskbar)
    rect: tuple[int, int, int, int]  # left, top, right, bottom (screen pixels)


@dataclass
class ElementInfo:
    name: str
    control_type: int
    rect: tuple[int, int, int, int]


# window classes of the taskbar / tray / notification overflow: a click there that ends a recording is not a step
SHELL_TRAY_CLASSES = frozenset(
    {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "NotifyIconOverflowWindow", "TopLevelWindowForOverflowXamlIsland"}
)


class NullWindowApi:
    """No window information: replay falls back to screen coordinates and the focused-app check.

    The interface every window API offers (duck-typed): ``window_at``, ``foreground``, ``find_window``,
    ``element_at``, ``find_element``, ``focused_is_password`` and the flags ``can_locate`` (window rectangles
    are known, so clicks are stored relative to their window) and ``can_check_password``."""

    can_locate = False
    can_check_password = False

    def __init__(self, active_process: Callable[[], str | None] | None = None) -> None:
        self._active = active_process

    def window_at(self, x: int, y: int) -> WindowInfo | None:
        return None

    def foreground(self) -> WindowInfo | None:
        if self._active is None:
            return None
        try:
            p = self._active()
        except Exception:
            return None
        return WindowInfo(0, 0, p, "", "", (0, 0, 0, 0)) if p else None

    def find_window(self, process: str, title: str = "") -> WindowInfo | None:
        fg = self.foreground()
        return fg if fg is not None and app_key(fg.process) == app_key(process) else None

    def element_at(self, x: int, y: int) -> ElementInfo | None:
        return None

    def find_element(self, window: WindowInfo, name: str, control_type: int = 0) -> ElementInfo | None:
        return None

    def focused_is_password(self) -> bool | None:
        return None


def default_window_api(use_uia: bool = True) -> Any:  # pragma: no cover - OS specific
    if sys.platform.startswith("win"):
        with contextlib.suppress(Exception):
            return Win32WindowApi(use_uia=use_uia)
    from away.voice.context import active_process

    return NullWindowApi(active_process)


# --------------------------------------------------------------------------- Windows (ctypes)
class Win32WindowApi:  # pragma: no cover - Windows only, verified by hand
    """Window lookup with user32 and UI Automation through raw COM vtable calls (ctypes only).

    COM is initialised per thread (``CoInitializeEx``), so each thread that asks gets its own UIA object.
    Every failure degrades to "unknown" rather than raising.
    """

    can_locate = True
    can_check_password = True

    # IUIAutomation vtable slots (UIAutomationClient.h order; IUnknown takes 0-2)
    _ELEMENT_FROM_HANDLE = 6
    _ELEMENT_FROM_POINT = 7
    _GET_FOCUSED = 8
    _CREATE_PROPERTY_CONDITION = 23
    # IUIAutomationElement slots
    _FIND_FIRST = 5
    _CUR_CONTROL_TYPE = 21
    _CUR_NAME = 23
    _CUR_IS_PASSWORD = 35
    _CUR_BOUNDING = 43
    _NAME_PROPERTY_ID = 30005
    _TREE_SCOPE_DESCENDANTS = 4

    def __init__(self, use_uia: bool = True) -> None:
        import ctypes
        from ctypes import wintypes

        self.ct = ctypes
        self.wt = wintypes
        self.user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        self.kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        self.user32.WindowFromPoint.restype = wintypes.HWND
        self.user32.WindowFromPoint.argtypes = [wintypes.POINT]
        self.user32.GetAncestor.restype = wintypes.HWND
        self.user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.use_uia = use_uia
        self._local = threading.local()
        self.own_pid = os.getpid()

    # ---------------------------------------------------------------- windows
    def _info(self, hwnd: Any) -> WindowInfo | None:
        ct, wt, u = self.ct, self.wt, self.user32
        if not hwnd:
            return None
        pid = wt.DWORD()
        u.GetWindowThreadProcessId(hwnd, ct.byref(pid))
        n = u.GetWindowTextLengthW(hwnd)
        buf = ct.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        cls = ct.create_unicode_buffer(256)
        u.GetClassNameW(hwnd, cls, 256)
        r = wt.RECT()
        u.GetWindowRect(hwnd, ct.byref(r))
        return WindowInfo(
            int(ct.cast(hwnd, ct.c_void_p).value or 0),
            int(pid.value),
            self._process_name(int(pid.value)),
            buf.value,
            cls.value,
            (r.left, r.top, r.right, r.bottom),
        )

    def _process_name(self, pid: int) -> str:
        ct, wt, k = self.ct, self.wt, self.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        try:
            buf = ct.create_unicode_buffer(1024)
            size = wt.DWORD(1024)
            if k.QueryFullProcessImageNameW(h, 0, buf, ct.byref(size)):
                return str(buf.value).replace("\\", "/").rsplit("/", 1)[-1]
        finally:
            k.CloseHandle(h)
        return ""

    def window_at(self, x: int, y: int) -> WindowInfo | None:
        hwnd = self.user32.WindowFromPoint(self.wt.POINT(int(x), int(y)))
        root = self.user32.GetAncestor(hwnd, 2) if hwnd else None  # GA_ROOT
        return self._info(root or hwnd)

    def foreground(self) -> WindowInfo | None:
        return self._info(self.user32.GetForegroundWindow())

    def find_window(self, process: str, title: str = "") -> WindowInfo | None:
        fg = self.foreground()
        want = app_key(process)
        if fg is not None and app_key(fg.process) == want and (not title or fg.title == title):
            return fg
        ct, wt, u = self.ct, self.wt, self.user32
        found: list[WindowInfo] = []
        proto = ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

        def cb(hwnd: Any, _lp: Any) -> bool:
            if u.IsWindowVisible(hwnd) and not u.IsIconic(hwnd) and u.GetWindowTextLengthW(hwnd) > 0:
                info = self._info(hwnd)
                if info is not None and app_key(info.process) == want:
                    found.append(info)
            return True

        u.EnumWindows(proto(cb), 0)
        if not found:
            return fg if fg is not None and app_key(fg.process) == want else None
        exact = [w for w in found if title and w.title == title]
        return (exact or found)[0]  # EnumWindows lists in z-order: the topmost first

    # ---------------------------------------------------------------- UI Automation
    def _uia(self) -> Any:
        if not self.use_uia:
            return None
        loc = self._local
        if getattr(loc, "tried", False):
            return getattr(loc, "uia", None)
        loc.tried = True
        ct = self.ct
        try:
            ole32 = ct.windll.ole32  # type: ignore[attr-defined]
            ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED; harmless if already initialised
            clsid = self._guid("{ff48dba4-60ef-4201-aa87-54103eef594e}")  # CUIAutomation
            iid = self._guid("{30cbe57d-d9d0-452a-ab13-7ac5ac4825ee}")  # IUIAutomation
            ptr = ct.c_void_p()
            hr = ole32.CoCreateInstance(ct.byref(clsid), None, 1, ct.byref(iid), ct.byref(ptr))  # CLSCTX_INPROC
            loc.uia = ptr if hr == 0 and ptr.value else None
        except Exception:
            loc.uia = None
        return loc.uia

    def _guid(self, text: str) -> Any:
        ct = self.ct

        class GUID(ct.Structure):
            _fields_ = [("d1", ct.c_ulong), ("d2", ct.c_ushort), ("d3", ct.c_ushort), ("d4", ct.c_ubyte * 8)]

        g = GUID()
        ct.oledll.ole32.CLSIDFromString(ct.c_wchar_p(text), ct.byref(g))  # type: ignore[attr-defined]
        return g

    def _method(self, obj: Any, slot: int, argtypes: list[Any]) -> Any:
        ct = self.ct
        vtbl = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
        return ct.WINFUNCTYPE(ct.c_long, ct.c_void_p, *argtypes)(vtbl[slot])

    def _release(self, obj: Any) -> None:
        with contextlib.suppress(Exception):
            if obj and obj.value:
                self._method(obj, 2, [])(obj)

    def _element_info(self, el: Any) -> ElementInfo | None:
        ct, wt = self.ct, self.wt
        name = ct.c_void_p()  # BSTR
        ctype = ct.c_int()
        rect = wt.RECT()
        if self._method(el, self._CUR_NAME, [ct.POINTER(ct.c_void_p)])(el, ct.byref(name)) != 0:
            return None
        self._method(el, self._CUR_CONTROL_TYPE, [ct.POINTER(ct.c_int)])(el, ct.byref(ctype))
        self._method(el, self._CUR_BOUNDING, [ct.POINTER(wt.RECT)])(el, ct.byref(rect))
        text = ct.wstring_at(name.value) if name.value else ""
        if name.value:
            with contextlib.suppress(Exception):
                ct.windll.oleaut32.SysFreeString(name)  # type: ignore[attr-defined]
        return ElementInfo(text, int(ctype.value), (rect.left, rect.top, rect.right, rect.bottom))

    def element_at(self, x: int, y: int) -> ElementInfo | None:
        uia = self._uia()
        if uia is None:
            return None
        ct, wt = self.ct, self.wt
        el = ct.c_void_p()
        try:
            hr = self._method(uia, self._ELEMENT_FROM_POINT, [wt.POINT, ct.POINTER(ct.c_void_p)])(
                uia, wt.POINT(int(x), int(y)), ct.byref(el)
            )
            if hr != 0 or not el.value:
                return None
            return self._element_info(el)
        except Exception:
            return None
        finally:
            self._release(el)

    def find_element(self, window: WindowInfo, name: str, control_type: int = 0) -> ElementInfo | None:
        uia = self._uia()
        if uia is None or not name or not window.handle:
            return None
        ct = self.ct

        class VARIANT(ct.Structure):
            _fields_ = [
                ("vt", ct.c_ushort),
                ("r1", ct.c_ushort),
                ("r2", ct.c_ushort),
                ("r3", ct.c_ushort),
                ("val", ct.c_void_p),
                ("pad", ct.c_void_p),
            ]

        root = ct.c_void_p()
        cond = ct.c_void_p()
        found = ct.c_void_p()
        bstr = None
        try:
            if self._method(uia, self._ELEMENT_FROM_HANDLE, [ct.c_void_p, ct.POINTER(ct.c_void_p)])(
                uia, ct.c_void_p(window.handle), ct.byref(root)
            ):
                return None
            oleaut = ct.windll.oleaut32  # type: ignore[attr-defined]
            oleaut.SysAllocString.restype = ct.c_void_p
            bstr = oleaut.SysAllocString(ct.c_wchar_p(name))
            var = VARIANT(8, 0, 0, 0, bstr, None)  # VT_BSTR
            if self._method(uia, self._CREATE_PROPERTY_CONDITION, [ct.c_int, VARIANT, ct.POINTER(ct.c_void_p)])(
                uia, self._NAME_PROPERTY_ID, var, ct.byref(cond)
            ):
                return None
            if self._method(root, self._FIND_FIRST, [ct.c_int, ct.c_void_p, ct.POINTER(ct.c_void_p)])(
                root, self._TREE_SCOPE_DESCENDANTS, cond, ct.byref(found)
            ):
                return None
            if not found.value:
                return None
            info = self._element_info(found)
            if info is not None and control_type and info.control_type != control_type:
                return None
            return info
        except Exception:
            return None
        finally:
            for o in (found, cond, root):
                self._release(o)
            if bstr:
                with contextlib.suppress(Exception):
                    ct.windll.oleaut32.SysFreeString(ct.c_void_p(bstr))  # type: ignore[attr-defined]

    def focused_is_password(self) -> bool | None:
        uia = self._uia()
        if uia is not None:
            ct = self.ct
            el = ct.c_void_p()
            try:
                if self._method(uia, self._GET_FOCUSED, [ct.POINTER(ct.c_void_p)])(uia, ct.byref(el)) == 0 and el.value:
                    flag = ct.c_int()
                    if self._method(el, self._CUR_IS_PASSWORD, [ct.POINTER(ct.c_int)])(el, ct.byref(flag)) == 0:
                        return bool(flag.value)
            except Exception:
                pass
            finally:
                self._release(el)
        return self._win32_password()

    def _win32_password(self) -> bool | None:
        """Fallback without UIA: a classic Edit control with the ES_PASSWORD style has focus."""
        ct, wt, u = self.ct, self.wt, self.user32

        class GUITHREADINFO(ct.Structure):
            _fields_ = [
                ("cbSize", wt.DWORD),
                ("flags", wt.DWORD),
                ("hwndActive", wt.HWND),
                ("hwndFocus", wt.HWND),
                ("hwndCapture", wt.HWND),
                ("hwndMenuOwner", wt.HWND),
                ("hwndMoveSize", wt.HWND),
                ("hwndCaret", wt.HWND),
                ("rcCaret", wt.RECT),
            ]

        gti = GUITHREADINFO(cbSize=ct.sizeof(GUITHREADINFO))
        try:
            if not u.GetGUIThreadInfo(0, ct.byref(gti)) or not gti.hwndFocus:
                return None
            style = u.GetWindowLongW(gti.hwndFocus, -16)  # GWL_STYLE
            cls = ct.create_unicode_buffer(64)
            u.GetClassNameW(gti.hwndFocus, cls, 64)
            if cls.value.lower() == "edit":
                return bool(style & 0x20)  # ES_PASSWORD
        except Exception:
            return None
        return None


# --------------------------------------------------------------------------- pynput listener + replay backend
class PynputListener:  # pragma: no cover - needs a desktop session
    """Feeds a recorder (or the abort watcher) from global pynput hooks.

    ``sink`` gets ``(kind, payload, injected)``; ``injected`` is True for events Windows marks as synthetic
    (SendInput from any program, including Project Away), False/unknown elsewhere.
    """

    def __init__(self, sink: Callable[[str, dict[str, Any], bool], None], mouse: bool = True) -> None:
        from pynput import keyboard
        from pynput import mouse as pmouse

        self._sink = sink
        self._injected = False
        kw: dict[str, Any] = {}
        mkw: dict[str, Any] = {}
        if sys.platform.startswith("win"):

            def kfilter(_msg: Any, data: Any) -> bool:
                self._injected = bool(getattr(data, "flags", 0) & 0x10)  # LLKHF_INJECTED
                return True

            def mfilter(_msg: Any, data: Any) -> bool:
                self._injected = bool(getattr(data, "flags", 0) & 0x01)  # LLMHF_INJECTED
                return True

            kw["win32_event_filter"] = kfilter
            mkw["win32_event_filter"] = mfilter
        self._k = keyboard.Listener(on_press=self._press, on_release=self._release, **kw)
        self._m = (
            pmouse.Listener(on_move=self._move, on_click=self._click, on_scroll=self._scroll, **mkw) if mouse else None
        )

    def start(self) -> None:
        self._k.start()
        if self._m is not None:
            self._m.start()

    def stop(self) -> None:
        for lst in (self._k, self._m):
            with contextlib.suppress(Exception):
                if lst is not None:
                    lst.stop()

    @staticmethod
    def key_name(k: Any) -> tuple[str, str]:
        """pynput key -> (canonical name, printable char or "")."""
        name = getattr(k, "name", None)
        if name:
            return str(name), ""
        ch = getattr(k, "char", None) or ""
        vk = getattr(k, "vk", None)
        if ch and ch.isprintable():
            return ch.lower(), ch
        if vk is not None and (0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A):  # ctrl+letter arrives as a control char
            return chr(vk).lower(), ""
        return "", ""

    def _press(self, k: Any) -> None:
        name, ch = self.key_name(k)
        self._sink("key_down", {"key": name, "char": ch}, self._injected)

    def _release(self, k: Any) -> None:
        name, ch = self.key_name(k)
        self._sink("key_up", {"key": name, "char": ch}, self._injected)

    def _move(self, x: int, y: int) -> None:
        self._sink("move", {"x": int(x), "y": int(y)}, self._injected)

    def _click(self, x: int, y: int, button: Any, pressed: bool) -> None:
        b = getattr(button, "name", "left")
        self._sink("down" if pressed else "up", {"x": int(x), "y": int(y), "button": b}, self._injected)

    def _scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._sink("scroll", {"x": int(x), "y": int(y), "dx": int(dx), "dy": int(dy)}, self._injected)


class PynputMacroBackend:  # pragma: no cover - needs a desktop session
    """Replay output. Every event is reported to ``mark`` first so recorders/abort watchers ignore it."""

    def __init__(self, mark: Callable[[], None] = lambda: None) -> None:
        from pynput import keyboard, mouse

        self._m = mouse.Controller()
        self._k = keyboard.Controller()
        self._buttons = mouse.Button
        self._keys = keyboard.Key
        self._mark = mark

    def _key(self, k: str) -> Any:
        return getattr(self._keys, k, k)

    def _button(self, b: str) -> Any:
        return getattr(self._buttons, b, self._buttons.left)

    def move(self, x: int, y: int) -> None:
        self._mark()
        self._m.position = (x, y)

    def press(self, button: str = "left") -> None:
        self._mark()
        self._m.press(self._button(button))

    def release(self, button: str = "left") -> None:
        self._mark()
        self._m.release(self._button(button))

    def click(self, button: str = "left", count: int = 1) -> None:
        self._mark()
        self._m.click(self._button(button), count)

    def scroll(self, dx: int, dy: int) -> None:
        self._mark()
        self._m.scroll(dx, dy)

    def key_down(self, k: str) -> None:
        self._mark()
        self._k.press(self._key(k))

    def key_up(self, k: str) -> None:
        self._mark()
        self._k.release(self._key(k))

    def type_text(self, text: str) -> None:
        self._mark()
        self._k.type(text)
