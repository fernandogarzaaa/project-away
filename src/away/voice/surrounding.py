"""Read the text around the cursor in the focused app, for smart cleanup and the personal dictionary.

* :func:`uia_text_before_caret` / :func:`uia_field_text`: Windows UI Automation (TextPattern) through
  ``comtypes`` when it is installed. Read-only: no keystrokes, the clipboard is never touched. Apps that do not
  expose TextPattern (many Electron/Chromium editors, terminals) return None.
* :class:`ClipboardSelection`: the selected text via a marker + Ctrl/Cmd+C, with the user's clipboard put back
  afterwards in every case. Opt-in (``voice.cleanup.context_clipboard``) and never used in terminals or code
  editors, where Ctrl+C means something else.

Everything here fails soft: any error -> None.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable
from typing import Any, Protocol

UIA_TEXT_PATTERN_ID = 10014
UIA_VALUE_PATTERN_ID = 10002
TEXT_UNIT_CHARACTER = 0
ENDPOINT_START = 0


def _uia() -> Any:  # pragma: no cover - Windows + comtypes
    import comtypes.client  # type: ignore[import-not-found]

    comtypes.client.GetModule("UIAutomationCore.dll")
    from comtypes.gen import UIAutomationClient as UIA  # type: ignore[import-not-found]

    return UIA, comtypes.client.CreateObject(UIA.CUIAutomation, interface=UIA.IUIAutomation)


def _focused_text_pattern() -> Any:  # pragma: no cover - Windows + comtypes
    UIA, auto = _uia()
    el = auto.GetFocusedElement()
    if el is None:
        return None, None, UIA
    pat = el.GetCurrentPattern(UIA_TEXT_PATTERN_ID)
    if pat is None:
        return el, None, UIA
    return el, pat.QueryInterface(UIA.IUIAutomationTextPattern), UIA


def uia_text_before_caret(max_chars: int = 400) -> str | None:  # pragma: no cover - Windows + comtypes
    if not sys.platform.startswith("win"):
        return None
    try:
        _el, tp, _UIA = _focused_text_pattern()
        if tp is None:
            return None
        sel = tp.GetSelection()
        if sel is None or sel.Length == 0:
            return None
        rng = sel.GetElement(0).Clone()
        # collapse to the caret, then extend backwards
        rng.MoveEndpointByRange(1, rng, ENDPOINT_START)  # End <- Start
        rng.MoveEndpointByUnit(ENDPOINT_START, TEXT_UNIT_CHARACTER, -int(max_chars))
        return str(rng.GetText(int(max_chars)) or "") or None
    except Exception:
        return None


def uia_field_text(max_chars: int = 4000) -> str | None:  # pragma: no cover - Windows + comtypes
    """The whole text of the focused field (TextPattern document, else the Value pattern)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        el, tp, UIA = _focused_text_pattern()
        if tp is not None:
            return str(tp.DocumentRange.GetText(int(max_chars)) or "") or None
        if el is not None:
            vp = el.GetCurrentPattern(UIA_VALUE_PATTERN_ID)
            if vp is not None:
                val = vp.QueryInterface(UIA.IUIAutomationValuePattern).CurrentValue
                return str(val or "")[-max_chars:] or None
    except Exception:
        return None
    return None


class Clipboard(Protocol):
    def get(self) -> str: ...
    def set(self, text: str) -> None: ...


class ClipboardSelection:
    """The current selection via the clipboard, restoring what was there before (even on errors)."""

    def __init__(
        self,
        clipboard: Clipboard,
        copy_keys: Callable[[], None],
        sleep: Callable[[float], None] = time.sleep,
        settle_s: float = 0.12,
    ) -> None:
        self.clip = clipboard
        self.copy_keys = copy_keys
        self.sleep = sleep
        self.settle_s = settle_s

    def read(self) -> str | None:
        try:
            saved: str | None = self.clip.get()
        except Exception:
            saved = None
        marker = f"\u2063away-ctx-{time.monotonic_ns()}"
        try:
            self.clip.set(marker)
            self.copy_keys()
            self.sleep(self.settle_s)
            got = self.clip.get()
            return None if not got or got == marker else got
        except Exception:
            return None
        finally:
            with contextlib.suppress(Exception):
                self.clip.set(saved if saved is not None else "")
