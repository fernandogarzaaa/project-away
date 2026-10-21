"""Agent Link UIA identity and session boundaries (audit fix, 2026-10-20).

Regression tests for the Phase 2 verification findings D1-D6 in
``docs/audits`` (phase2-verification.md): the element registry must be bound to one
verified foreground window, one query generation, and one live control session.

* D1: the foreground fingerprint is captured before AND after the tree walk; the
  registry is installed only when both are non-null and equal.
* D2: element ids are never reissued, so an id from an older query generation
  cannot silently resolve to a different element.
* D3: click and drag revalidate the target after an approval prompt.
* D4: an automation id on the snapshot is strong identity (the live element must
  carry the same one); bounds must match exactly, so duplicate-named controls
  without automation ids are not confused and an empty name skips no check.
* D5: ui_query requires the querying agent's own live, unpaused control grant,
  like act and undo (the CONTROL permission alone is not enough).
* D6: every grant end path clears set-of-mark coordinates and screenshot
  geometry along with the element ids; a new session needs a fresh screenshot.
"""

from __future__ import annotations

from typing import Any

import pytest

from away.agentlink.control import ComputerControl, ControlHooks, ControlRefused
from away.agentlink.desktop import Element
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def fg_dict(title: str, process: str = "app.exe", cls: str = "WinClass", rect=(0, 0, 800, 600)) -> dict[str, Any]:
    return {"title": title, "process": process, "class": cls, "rect": rect}


class FakeUIA:
    """A UIA source whose tree and hit-test results the test can swap mid-flight."""

    available = True
    reason = ""

    def __init__(self, root: Element | None = None, on_tree: Any = None) -> None:
        self._root = root
        self._on_tree = on_tree

    def tree(self, max_depth: int = 3, max_nodes: int = 200) -> Element | None:
        if self._on_tree is not None:
            self._on_tree()
        return self._root

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        def visit(el: Element) -> Element | None:
            for child in el.children or []:
                found = visit(child)
                if found is not None:
                    return found
            if el.rect is not None:
                left, top, right, bottom = el.rect
                if left <= x <= right and top <= y <= bottom:
                    return el
            return None

        return visit(self._root) if self._root is not None else None

    def focused(self) -> Element | None:
        return None


def _control(
    clock: Clock,
    *,
    uia: Any = None,
    fg: Any = None,
    ask: Any = None,
) -> tuple[ComputerControl, RecordingBackend]:
    backend = RecordingBackend()
    hooks = ControlHooks(
        backend=backend,
        platform="linux",
        secure_desktop=lambda: False,
        foreground=fg if fg is not None else (lambda: fg_dict("Mail", "mail.exe")),
        grab=lambda: None,
        sleep=lambda _s: None,
        **({"uia": uia} if uia is not None else {}),
    )
    ctl = ComputerControl(
        hooks,
        ask or (lambda _q, _o, **kw: {"answer": "Allow", "method": "click"}),
        ttl_s=1800,
        rate_per_s=10000.0,
        clock=clock,
    )
    return ctl, backend


def _mail_tree() -> Element:
    send = Element("Send", "ButtonControl", rect=(100, 100, 140, 120))
    subject = Element("Subject", "EditControl", rect=(0, 0, 50, 20))
    root = Element("Mail", "WindowControl", rect=(0, 0, 800, 600))
    root.children = [send, subject]
    return root


# ---------------------------------------------------------------------- D5: grant gate


def test_ui_query_without_any_grant_is_refused() -> None:
    ctl, _ = _control(Clock(), uia=FakeUIA(_mail_tree()))
    assert ctl.grant is None  # nobody ever requested control
    with pytest.raises(ControlRefused, match="request_control first"):
        ctl.ui_query("a1")


def test_ui_query_with_another_agents_grant_is_refused() -> None:
    ctl, _ = _control(Clock(), uia=FakeUIA(_mail_tree()))
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="is using the computer"):
        ctl.ui_query("a2")


def test_ui_query_with_a_paused_grant_is_refused() -> None:
    ctl, _ = _control(Clock(), uia=FakeUIA(_mail_tree()))
    ctl.request("a1", "Claude")
    assert ctl.grant is not None
    ctl.grant.paused = "the user is using the mouse"
    with pytest.raises(ControlRefused, match="paused"):
        ctl.ui_query("a1")


def test_ui_query_with_own_live_grant_works() -> None:
    ctl, _ = _control(Clock(), uia=FakeUIA(_mail_tree()))
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    assert res["available"] is True and res["count"] == 3


# ------------------------------------------- D1: fingerprint before and after the walk


def test_foreground_change_during_tree_walk_fails_closed() -> None:
    state = {"fg": fg_dict("WindowA")}
    el_a = Element("AlphaButton", "ButtonControl", rect=(0, 0, 100, 40))
    tree = Element("RootA", "WindowControl", rect=(0, 0, 800, 600), children=[el_a])
    uia = FakeUIA(tree, on_tree=lambda: state.update(fg=fg_dict("WindowB")))
    ctl, _ = _control(Clock(), uia=uia, fg=lambda: state["fg"])
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    assert res["available"] is False  # the walk crossed a foreground change: no registry
    with pytest.raises(ControlRefused, match="unknown element id"):
        ctl.act("a1", "click", element_id="e1")


def test_unverifiable_foreground_fails_closed() -> None:
    ctl, _ = _control(Clock(), uia=FakeUIA(_mail_tree()), fg=lambda: None)
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    assert res["available"] is False and "could not verify the foreground window" in res["reason"]


# ------------------------------------------------- D2: ids bound to one query generation


def test_stale_id_from_a_previous_generation_is_refused() -> None:
    alpha = Element("Alpha", "ButtonControl", rect=(0, 0, 100, 40))
    beta = Element("Beta", "ButtonControl", rect=(200, 200, 300, 240))
    state = {"root": Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[alpha])}
    uia = FakeUIA(on_tree=None)
    uia._root = state["root"]
    ctl, _ = _control(Clock(), uia=uia)
    ctl.request("a1", "Claude")
    first = ctl.ui_query("a1")
    alpha_id = next(e["id"] for e in first["elements"] if e["name"] == "Alpha")
    state["root"] = Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[beta])
    uia._root = state["root"]
    second = ctl.ui_query("a1")
    beta_id = next(e["id"] for e in second["elements"] if e["name"] == "Beta")
    assert beta_id != alpha_id  # ids are never reissued across generations
    with pytest.raises(ControlRefused, match="unknown element id"):
        ctl.act("a1", "click", element_id=alpha_id)  # the stale id must not land on Beta
    ctl.act("a1", "click", element_id=beta_id)  # the fresh id works


# ------------------------------------------- D4: identity strength at revalidation time


def test_live_element_without_the_snapshots_automation_id_is_refused() -> None:
    cached = Element("OK", "ButtonControl", automation_id="okButton", rect=(0, 0, 100, 40))
    tree = Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[cached])
    uia = FakeUIA(tree)
    ctl, backend = _control(Clock(), uia=uia)
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    ok_id = next(e["id"] for e in res["elements"] if e["name"] == "OK")
    # A different element with the same name and type but NO automation id takes the spot.
    impostor = Element("OK", "ButtonControl", automation_id="", rect=(0, 0, 100, 40))
    uia._root = Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[impostor])
    with pytest.raises(ControlRefused, match="gone or has moved"):
        ctl.act("a1", "click", element_id=ok_id)
    assert backend.calls == []


def test_duplicate_named_controls_are_not_confused() -> None:
    ok1 = Element("OK", "ButtonControl", rect=(0, 0, 100, 40))
    ok2 = Element("OK", "ButtonControl", rect=(200, 0, 300, 40))
    tree = Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[ok1, ok2])
    uia = FakeUIA(tree)
    ctl, backend = _control(Clock(), uia=uia)
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    ids = {tuple(e["bounds"]): e["id"] for e in res["elements"] if e["name"] == "OK"}
    ctl.act("a1", "click", element_id=ids[(200, 0, 300, 40)])
    assert backend.calls[-2:] == [("move", (250, 20)), ("click", (1,))]  # the second OK, not the first
    # The second OK moves away and another OK (same name, same type, different bounds) now covers the
    # old center: name and type alone must not wave it through; the bounds no longer match.
    moved = Element("OK", "ButtonControl", rect=(500, 300, 600, 340))
    impostor = Element("OK", "ButtonControl", rect=(200, 0, 320, 40))
    uia._root = Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[ok1, moved, impostor])
    with pytest.raises(ControlRefused, match="gone or has moved"):
        ctl.act("a1", "click", element_id=ids[(200, 0, 300, 40)])


def test_empty_name_element_gets_no_free_pass() -> None:
    icon = Element("", "ButtonControl", rect=(0, 0, 100, 40))  # no name, no automation id
    tree = Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[icon])
    uia = FakeUIA(tree)
    ctl, backend = _control(Clock(), uia=uia)
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    icon_id = next(e["id"] for e in res["elements"] if e["name"] == "")
    ctl.act("a1", "click", element_id=icon_id)  # unchanged: bounds are its identity, click works
    assert backend.calls[-2:] == [("move", (50, 20)), ("click", (1,))]
    # The icon moves: with no name and no automation id, nothing may wave it through.
    shifted = Element("", "ButtonControl", rect=(300, 300, 400, 340))
    other = Element("", "ButtonControl", rect=(0, 0, 90, 30))
    uia._root = Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[shifted, other])
    with pytest.raises(ControlRefused, match="gone or has moved"):
        ctl.act("a1", "click", element_id=icon_id)


# ------------------------------------------- D3: revalidation after an approval prompt


def _ask_flipping(changes: Any) -> Any:
    def ask(_q: str, _o: Any, **kw: Any) -> dict[str, Any]:
        if kw.get("title") == "Are you sure?":
            changes()
        return {"answer": "Allow", "method": "click"}

    return ask


def _delete_tree() -> Element:
    delete = Element("Delete", "ButtonControl", rect=(0, 0, 100, 40))
    return Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[delete])


def test_foreground_change_during_risky_prompt_refuses_the_click() -> None:
    state = {"fg": fg_dict("WindowA")}
    uia = FakeUIA(_delete_tree())
    ask = _ask_flipping(lambda: state.update(fg=fg_dict("WindowB")))
    ctl, backend = _control(Clock(), uia=uia, fg=lambda: state["fg"], ask=ask)
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    delete_id = next(e["id"] for e in res["elements"] if e["name"] == "Delete")
    with pytest.raises(ControlRefused):
        ctl.act("a1", "click", element_id=delete_id)  # approved for WindowA's Delete only
    assert backend.calls == []


def test_target_vanishing_during_risky_prompt_refuses_the_drag() -> None:
    state = {"root": _delete_tree()}
    file_el = Element("File", "ListItemControl", rect=(300, 300, 360, 340))
    state["root"].children.append(file_el)
    uia = FakeUIA(state["root"])

    def empty_the_tree() -> None:
        uia._root = Element("Root", "WindowControl", rect=(0, 0, 800, 600), children=[])

    ask = _ask_flipping(empty_the_tree)
    ctl, backend = _control(Clock(), uia=uia, ask=ask)
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    file_id = next(e["id"] for e in res["elements"] if e["name"] == "File")
    delete_id = next(e["id"] for e in res["elements"] if e["name"] == "Delete")
    with pytest.raises(ControlRefused):
        ctl.act("a1", "drag", element_id=file_id, to_element_id=delete_id)
    assert ("press", ()) not in backend.calls  # the drag never started


def test_risky_click_still_lands_when_nothing_changes_during_the_prompt() -> None:
    uia = FakeUIA(_delete_tree())
    ask = _ask_flipping(lambda: None)
    ctl, backend = _control(Clock(), uia=uia, ask=ask)
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    delete_id = next(e["id"] for e in res["elements"] if e["name"] == "Delete")
    ctl.act("a1", "click", element_id=delete_id)
    assert backend.calls[-2:] == [("move", (50, 20)), ("click", (1,))]


# ------------------------------------------- D6: grant end clears marks and ids together


def test_stop_clears_marks_and_ids_for_the_next_session() -> None:
    ctl, _ = _control(Clock(), uia=FakeUIA(_mail_tree()))
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    old_id = res["elements"][0]["id"]
    ctl.marks["a1"] = {3: (111, 222)}  # what an annotated screenshot stored this session
    ctl.stop("stopped")
    assert ctl.marks.get("a1") is None  # set-of-mark coordinates die with the grant
    ctl.request("a1", "Claude")  # a brand-new session, no fresh screenshot, no fresh query
    with pytest.raises(ControlRefused, match="no annotation 3"):
        ctl._point("a1", None, None, "mark:3", None, False)
    with pytest.raises(ControlRefused, match="unknown element id"):
        ctl.act("a1", "click", element_id=old_id)


def test_expiry_clears_marks() -> None:
    clock = Clock()
    ctl, _ = _control(clock, uia=FakeUIA(_mail_tree()))
    ctl.request("a1", "Claude")
    ctl.marks["a1"] = {1: (10, 10)}
    clock.advance(1801)  # past the grant TTL
    with pytest.raises(ControlRefused):
        ctl.ui_query("a1")  # the expiry check ends the grant on the way in
    assert ctl.marks.get("a1") is None


def test_revoke_clears_marks_and_ids() -> None:
    ctl, _ = _control(Clock(), uia=FakeUIA(_mail_tree()))
    ctl.request("a1", "Claude")
    res = ctl.ui_query("a1")
    old_id = res["elements"][0]["id"]
    ctl.marks["a1"] = {2: (50, 60)}
    ctl.revoke_agent("a1")
    assert ctl.marks.get("a1") is None
    ctl.request("a1", "Claude")
    with pytest.raises(ControlRefused, match="no annotation 2"):
        ctl._point("a1", None, None, "mark:2", None, False)
    with pytest.raises(ControlRefused, match="unknown element id"):
        ctl.act("a1", "click", element_id=old_id)
