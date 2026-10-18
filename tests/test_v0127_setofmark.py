"""Agent Link (v0.1.27): set-of-mark prompting, proposal #25.

get_screen(annotate=true) draws numbered boxes over detected interactable regions for canvas apps, games and
remote desktops where the UIA tree is empty; pointer actions can click them as element="mark:14".
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from away import config as C
from away.agentlink.control import ControlRefused
from away.agentlink.host import build_core
from away.agentlink.prompts import NullUI


def _screen_with_boxes() -> object:
    """A synthetic 1920x1080 screen with three outlined button-like boxes."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1920, 1080), (245, 245, 245))
    d = ImageDraw.Draw(img)
    for x, y, w, h in ((200, 150, 300, 120), (700, 150, 300, 120), (200, 420, 220, 220)):
        d.rectangle([x, y, x + w, y + h], outline=(20, 20, 20), width=4)
        d.rectangle([x + 16, y + 16, x + w - 16, y + h - 16], fill=(255, 255, 255))
    return img


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / (aw * ah + bw * bh - ix * iy)


@pytest.fixture
def core(tmp_path: Path):
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    ui = NullUI()
    flashes: list[str] = []
    c = build_core(
        tmp_path,
        cfg=lambda: cfg,
        settings=lambda: cfg.agent_link,
        config_path=lambda: cfgp,
        ui=ui,
        flash=flashes.append,
        camera_frame=lambda _cam: None,
        screen_grab=lambda: None,
    )
    c.prompts.polite = lambda: True
    c.ui, c.flashes, c.cfgp, c.cfg_obj = ui, flashes, cfgp, cfg  # type: ignore[attr-defined]
    return c


# --------------------------------------------------------------------------- the detector


def test_detector_finds_drawn_boxes() -> None:
    from away.agentlink import setofmark as som

    img = _screen_with_boxes()
    regions = som.detect_regions(img)
    truth = [(200, 150, 300, 120), (700, 150, 300, 120), (200, 420, 220, 220)]
    for t in truth:
        best = max((_iou(t, r.box) for r in regions), default=0.0)
        assert best > 0.5, f"no detected region matches the drawn box {t} (found {len(regions)})"
    assert [r.id for r in regions] == sorted(r.id for r in regions)  # ids are sequential
    assert all(0.0 <= r.confidence <= 1.0 for r in regions)


def test_detector_handles_flat_and_empty_images() -> None:
    from PIL import Image

    from away.agentlink import setofmark as som

    assert som.detect_regions(Image.new("RGB", (640, 480), "white")) == []
    assert som.detect_regions(Image.new("RGB", (0, 1), "white")) == []


# --------------------------------------------------------------------------- get_screen(annotate=true)


def test_get_screen_annotate_marks_boxes_and_lists_them(core) -> None:
    from PIL import Image

    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    jpeg, meta = core.service.get_screen(a, annotate=True)
    anns = meta.get("annotations")
    assert isinstance(anns, list) and len(anns) >= 3, f"expected >= 3 annotations, got {anns}"
    assert [e["id"] for e in anns] == list(range(1, len(anns) + 1))
    assert all(e["label"] == f"region {e['id']}" for e in anns)
    for e in anns:  # every box and center is inside the 1920x1080 screen
        s = e["screen"]
        assert s["x"] >= 0 and s["y"] >= 0 and s["x"] + s["width"] <= 1920 and s["y"] + s["height"] <= 1080
        c = e["center"]
        assert s["x"] <= c["x"] <= s["x"] + s["width"] and s["y"] <= c["y"] <= s["y"] + s["height"]
    assert "annotate_note" in meta and "mark:14" in meta["annotate_note"]
    # the numbered boxes are really drawn on the returned picture (red pixels exist)
    px = Image.open(io.BytesIO(jpeg)).convert("RGB").load()
    w, h = Image.open(io.BytesIO(jpeg)).size
    red = sum(
        1 for x in range(0, w, 4) for y in range(0, h, 4) if px[x, y][0] > 200 and px[x, y][1] < 90 and px[x, y][2] < 90
    )
    assert red > 0, "no red annotation boxes drawn on the returned screenshot"
    # the click points are cached for this agent's pointer actions
    marks = core.service.control.marks.get(a.id, {})
    assert len(marks) == len(anns)
    assert marks[anns[0]["id"]] == (anns[0]["center"]["x"], anns[0]["center"]["y"])


def test_get_screen_without_annotate_is_unchanged(core) -> None:
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    _jpeg, meta = core.service.get_screen(a)
    assert "annotations" not in meta and "annotate_note" not in meta
    assert a.id not in core.service.control.marks


# --------------------------------------------------------------------------- element="mark:14" references


def test_pointer_accepts_mark_reference_without_uia(core) -> None:
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    _jpeg, meta = core.service.get_screen(a, annotate=True)
    first = meta["annotations"][0]
    x, y, label = core.service.control._point(a.id, None, None, "mark:1", False)
    assert (x, y) == (first["center"]["x"], first["center"]["y"])
    assert label == "mark 1"
    # the other spellings work too
    assert core.service.control._point(a.id, None, None, "#2", False)[:2] == (
        meta["annotations"][1]["center"]["x"],
        meta["annotations"][1]["center"]["y"],
    )
    assert core.service.control._point(a.id, None, None, "MARK 3", False)[2] == "mark 3"


def test_pointer_rejects_unknown_mark(core) -> None:
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    core.service.get_screen(a, annotate=True)
    with pytest.raises(ControlRefused, match="no annotation 999"):
        core.service.control._point(a.id, None, None, "mark:999", False)
    b, _ = core.registry.issue("Other", ["read", "screen"], 3600)  # never annotated for this agent
    with pytest.raises(ControlRefused, match="annotate=true"):
        core.service.control._point(b.id, None, None, "mark:1", False)


def test_revoke_clears_marks(core) -> None:
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    core.service.get_screen(a, annotate=True)
    assert core.service.control.marks.get(a.id)
    core.service.control.revoke_agent(a.id, "test")
    assert a.id not in core.service.control.marks


def test_fresh_screenshot_clears_marks(core) -> None:
    """Marks belong to the annotated screenshot: a newer deliberate screenshot retires them."""
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    core.service.get_screen(a, annotate=True)
    assert core.service.control.marks.get(a.id)
    core.service.get_screen(a, annotate=False)  # the agent looks at a fresh screenshot
    assert a.id not in core.service.control.marks
    with pytest.raises(ControlRefused, match="no annotation 1"):
        core.service.control._point(a.id, None, None, "mark:1", False)


def test_quiet_live_view_keeps_marks(core) -> None:
    """Background live-view frames (quiet) are not the agent's working screenshot: they keep marks."""
    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    _jpeg, meta = core.service.get_screen(a, annotate=True)
    first = meta["annotations"][0]
    core.service.get_screen(a, annotate=False, quiet=True)  # a live-view frame, not a new look
    x, y, _label = core.service.control._point(a.id, None, None, "mark:1", False)
    assert (x, y) == (first["center"]["x"], first["center"]["y"])


def test_failed_annotation_clears_marks(core, monkeypatch) -> None:
    """A failed annotation must not leave the previous screenshot's marks behind."""
    from away.agentlink import setofmark as som

    core.service.screen_grab = lambda: _screen_with_boxes()
    a, _ = core.registry.issue("Agent", ["read", "screen"], 3600)
    core.service.get_screen(a, annotate=True)
    assert core.service.control.marks.get(a.id)

    def _boom(_pil):
        raise RuntimeError("detector blew up")

    monkeypatch.setattr(som, "detect_regions", _boom)
    with pytest.raises(RuntimeError, match="detector blew up"):
        core.service.get_screen(a, annotate=True)
    assert a.id not in core.service.control.marks
