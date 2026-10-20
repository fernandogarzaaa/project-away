"""Prompt coverage for agent actions (audit Phase 4, lane B item 1).

Four gaps, each reproduced against the unfixed code before this file landed:

1. A set-of-mark click or drag carried the synthetic label "mark N", which classifies
   safe, so a mark sitting on a risky element acted with no prompt while the same spot
   clicked by raw coordinates prompted. A mark now inherits the classification of the
   element actually under it, exactly like the coordinate path.
2. Space is the standard keyboard activation key for a focused button, but it was in
   no prompt list, so Space on a focused "Pay now" button activated it silently while
   Enter on the same focus asked first. Space now goes through the same focus check
   as Enter.
3. The typing target (terminal process, password field) was verified once before
   typing; later chunks were re-checked only against the caller's ``expect`` string.
   A mid-type focus flip to a terminal received the rest of the text. The target class
   is now re-verified before every chunk, and again after the Enter prompt.
4. Delete was judged by the focused element's name, which names the container, not the
   thing Delete acts on: Delete on a selected file called report.docx ran silently.
   Delete now asks unless the focused element is a text-editing element, the one case
   where Delete provably deletes a character rather than the selected item.
"""

from __future__ import annotations

from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element, best_match
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeUIA:
    available = True

    def __init__(self, focused: Any = None, at: Any = None, elements: Any = ()) -> None:
        self._focused, self._at, self._els = focused, at, list(elements)

    def focused(self) -> Any:
        return self._focused() if callable(self._focused) else self._focused

    def element_at(self, x: int, y: int) -> Any:
        return self._at(x, y) if callable(self._at) else self._at

    def find(self, name: str, control_type: str | None = None) -> Any:
        return best_match(self._els, name, control_type)

    def tree(self, *a: Any) -> None:
        return None


TEXT_BOX = Element("Message", "EditControl", rect=(0, 0, 9, 9))


def _control(
    clock: Clock,
    answers: list[str] | None = None,
    uia: Any = None,
    backend: Any = None,
    platform: str = "linux",
    foreground: Any = None,
    on_ask: Any = None,
) -> tuple[ComputerControl, Any, list[dict[str, Any]]]:
    backend = backend if backend is not None else RecordingBackend()
    asked: list[dict[str, Any]] = []
    queue = list(answers or ["Allow"])

    def ask(question: str, options: list[str], **kw: Any) -> dict[str, Any]:
        asked.append({"question": question, "options": options, **kw})
        if on_ask is not None:
            on_ask(len(asked))
        return {"answer": queue.pop(0) if queue else "Allow", "method": "click"}

    hooks = ControlHooks(
        backend=backend,
        platform=platform,
        secure_desktop=lambda: False,
        foreground=foreground or (lambda: None),
        grab=lambda: None,
        sleep=lambda _s: None,
        uia=uia if uia is not None else FakeUIA(focused=TEXT_BOX, at=Element("Canvas", "PaneControl")),
    )
    ctl = ComputerControl(hooks, ask, ttl_s=1800, rate_per_s=1000, clock=clock)
    return ctl, backend, asked


DELETE_BTN = Element("Delete", "ButtonControl", rect=(40, 10, 60, 30))


# --------------------------------------------------------------------- 1. marks inherit the target's risk


def test_mark_click_on_a_risky_element_asks_first_and_denial_blocks_it() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=FakeUIA(focused=TEXT_BOX, at=DELETE_BTN))
    ctl.request("a1", "Claude")
    ctl.marks["a1"] = {1: (50, 20)}
    asked.clear()
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act("a1", "click", element="mark:1")
    assert len(asked) == 1, "the mark click must ask, like the coordinate click on the same spot"
    assert "click" not in [c[0] for c in backend.calls]


def test_mark_click_on_a_risky_element_runs_when_allowed_and_reports_the_element() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Allow"], uia=FakeUIA(focused=TEXT_BOX, at=DELETE_BTN))
    ctl.request("a1", "Claude")
    ctl.marks["a1"] = {1: (50, 20)}
    asked.clear()
    out = ctl.act("a1", "click", element="mark:1")
    assert out["ok"] and len(asked) == 1
    assert ("click", (1,)) in backend.calls
    assert out["target_label"] == "Delete"  # the agent is told what the mark actually sat on


def test_mark_click_with_nothing_identifiable_under_it_asks_first() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=FakeUIA(focused=TEXT_BOX, at=None))
    ctl.request("a1", "Claude")
    ctl.marks["a1"] = {1: (50, 20)}
    asked.clear()
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act("a1", "click", element="mark:1")
    assert len(asked) == 1, "no element under the mark is unknown, and unknown asks (coordinate parity)"
    assert "click" not in [c[0] for c in backend.calls]


def test_mark_drag_onto_a_risky_element_asks_first_and_denial_blocks_it() -> None:
    clock = Clock()
    file_el = Element("report.docx", "ListItemControl", rect=(0, 0, 40, 40))
    bin_el = Element("Recycle Bin", "ListItemControl", rect=(400, 0, 440, 40))
    uia = FakeUIA(at=lambda x, y: file_el if x < 100 else bin_el)
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=uia)
    ctl.request("a1", "Claude")
    ctl.marks["a1"] = {1: (420, 20)}
    asked.clear()
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act("a1", "drag", x=20, y=20, to_element="mark:1")
    assert len(asked) == 1 and "Recycle Bin" in asked[-1]["question"]
    assert "press" not in [c[0] for c in backend.calls]


# --------------------------------------------------------------------- 2. Space activates the focused control


def test_space_on_a_focused_risky_button_asks_first_and_denial_blocks_it() -> None:
    clock = Clock()
    pay = Element("Pay now", "ButtonControl", rect=(0, 0, 9, 9))
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=FakeUIA(focused=pay))
    ctl.request("a1", "Claude")
    asked.clear()
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act("a1", "press_keys", keys="space")
    assert len(asked) == 1, "Space activates the focused button, so it asks like Enter does"
    assert "key" not in [c[0] for c in backend.calls]


def test_space_on_a_safe_focused_button_does_not_ask() -> None:
    clock = Clock()
    play = Element("Play", "ButtonControl", rect=(0, 0, 9, 9))
    ctl, backend, asked = _control(clock, uia=FakeUIA(focused=play))
    ctl.request("a1", "Claude")
    asked.clear()
    out = ctl.act("a1", "press_keys", keys="space")
    assert out["ok"] and asked == []  # parity with Enter on a harmless focus: no question
    assert ("key", ("space",)) in backend.calls


# --------------------------------------------------------------------- 3. typing target, re-verified per chunk


class FlipBackend(RecordingBackend):
    """Flips the foreground (to a terminal) after every typed chunk."""

    def __init__(self, flip: Any) -> None:
        super().__init__()
        self._flip = flip
        self.typed: list[str] = []

    def type_text(self, text: str) -> None:
        self.typed.append(text)
        super().type_text(text)
        self._flip()


def test_typing_stops_when_the_foreground_becomes_a_terminal_mid_type() -> None:
    clock = Clock()
    state = {"fg": {"title": "Notes", "process": "notes.exe", "class": "", "rect": (0, 0, 9, 9)}}

    def flip() -> None:
        state["fg"] = {"title": "Windows Terminal", "process": "windowsterminal.exe", "class": "", "rect": (0, 0, 9, 9)}

    backend = FlipBackend(flip)
    ctl, _b, _asked = _control(clock, backend=backend, foreground=lambda: state["fg"])
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="terminal"):
        ctl.act("a1", "type_text", text="x" * 40)  # 5 chunks of 8, no `expect` given
    joined = "".join(backend.typed)
    assert len(joined) == 8, f"only the chunk typed before the flip may land, got {len(joined)} characters"


def test_typing_stops_after_the_enter_prompt_when_the_target_turned_terminal_without_expect() -> None:
    clock = Clock()
    state = {"fg": {"title": "Notes", "process": "notes.exe", "class": "", "rect": (0, 0, 9, 9)}}

    def flip(n: int) -> None:
        if n >= 2:  # ask 1 is the grant request; ask 2 is the Enter prompt
            state["fg"] = {
                "title": "Windows Terminal",
                "process": "windowsterminal.exe",
                "class": "",
                "rect": (0, 0, 9, 9),
            }

    ctl, backend, asked = _control(clock, foreground=lambda: state["fg"], on_ask=flip, uia=FakeUIA(focused=None))
    ctl.request("a1", "Claude")
    n_asked = len(asked)  # not cleared: the flip counts asks from the start, grant included
    with pytest.raises(ControlRefused, match="terminal"):
        ctl.act("a1", "type_text", text="hello\nworld")
    assert len(asked) == n_asked + 1  # the Enter prompt did show, and the user allowed it
    assert "type_text" not in [c[0] for c in backend.calls], "nothing may be typed into the terminal"


# --------------------------------------------------------------------- 4. Delete acts on the selected item


def test_delete_on_a_file_item_asks_first_and_denial_blocks_it() -> None:
    clock = Clock()
    fname = Element("report.docx", "ListItemControl", rect=(0, 0, 9, 9))
    ctl, backend, asked = _control(clock, ["Allow", "Deny"], uia=FakeUIA(focused=fname))
    ctl.request("a1", "Claude")
    asked.clear()
    with pytest.raises(ControlRefused, match="did not allow"):
        ctl.act("a1", "press_keys", keys="delete")
    assert len(asked) == 1, "the label names the file, and Delete removes the file: it must ask"
    assert "key" not in [c[0] for c in backend.calls]


def test_delete_on_a_file_item_runs_when_allowed() -> None:
    clock = Clock()
    fname = Element("report.docx", "ListItemControl", rect=(0, 0, 9, 9))
    ctl, backend, asked = _control(clock, ["Allow", "Allow"], uia=FakeUIA(focused=fname))
    ctl.request("a1", "Claude")
    asked.clear()
    out = ctl.act("a1", "press_keys", keys="delete")
    assert out["ok"] and len(asked) == 1
    assert ("key", ("delete",)) in backend.calls


def test_delete_in_a_text_field_does_not_ask() -> None:
    clock = Clock()
    ctl, backend, asked = _control(clock, uia=FakeUIA(focused=TEXT_BOX))
    ctl.request("a1", "Claude")
    asked.clear()
    out = ctl.act("a1", "press_keys", keys="delete")
    assert out["ok"] and asked == [], "Delete inside a text editor deletes a character: no question"
    assert ("key", ("delete",)) in backend.calls
