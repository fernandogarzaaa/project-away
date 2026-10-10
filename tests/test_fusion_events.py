import pytest

from away.events import EventBus
from away.fusion import EVENT_TO_CLASS, GestureFusion
from away.hands.gestures import GestureEvent


def ev(kind, t=10.0):
    return GestureEvent(kind, t)


def test_pass_through_events_always_accepted():
    f = GestureFusion(csi_weight=1.0, threshold=0.99)
    f.update_csi("empty", 1.0, 10.0)
    for k in ("move", "scroll", "drag_end", "hand_lost", "unknown_kind"):
        assert f.score(ev(k)) == 1.0
        assert f.accept(ev(k))


def test_no_model_no_csi_scores():
    f = GestureFusion(csi_weight=0.25)
    assert f.camera_conf("pinch_click") == 1.0
    assert f.csi_support("click", 10.0) == 0.5
    assert f.score(ev("click")) == pytest.approx(0.75 + 0.125)


def test_model_probability_uses_recent_max():
    f = GestureFusion(csi_weight=0.0, model_window=2)
    f.update_model({"pinch_click": 0.2})
    f.update_model(None)
    f.update_model({})
    assert f.camera_conf("pinch_click") == 0.2
    f.update_model({"pinch_click": 0.9})
    f.update_model({"pinch_click": 0.1})
    assert f.camera_conf("pinch_click") == 0.9
    f.update_model({"pinch_click": 0.3})
    assert f.camera_conf("pinch_click") == 0.3  # window of 2 dropped the 0.9
    assert f.camera_conf("swipe_left") == 0.0


def test_csi_empty_room_suppresses():
    f = GestureFusion(csi_weight=0.5, threshold=0.6)
    f.update_csi("empty", 1.0, 10.0)
    assert f.csi_support("click", 10.0) == 0.0
    assert not f.accept(ev("click"))
    assert f.rejected[0][0].kind == "click"
    assert f.rejected[0][1] == pytest.approx(0.5)
    f.update_csi("empty", 0.4, 10.0)
    assert f.csi_support("click", 10.0) == pytest.approx(0.3)


def test_csi_agreement_boosts_swipes_only():
    f = GestureFusion(csi_weight=0.5)
    f.update_csi("swipe", 0.8, 10.0)
    assert f.csi_support("swipe_left", 10.0) == pytest.approx(0.9)
    assert f.csi_support("swipe_right", 10.0) == pytest.approx(0.9)
    assert f.csi_support("click", 10.0) == 0.5
    f.update_csi("motion", 0.0, 10.0)
    assert f.csi_support("swipe_left", 10.0) == 0.5
    f.update_csi("push", 1.0, 10.0)
    assert f.csi_support("swipe_left", 10.0) == 0.5


def test_csi_stale_is_neutral():
    f = GestureFusion(csi_weight=0.5, csi_max_age=1.0)
    f.update_csi("empty", 1.0, 0.0)
    assert f.csi_support("click", 1.0) == 0.0
    assert f.csi_support("click", 1.01) == 0.5


def test_threshold_boundary_and_validation():
    f = GestureFusion(csi_weight=0.0, threshold=0.5)
    f.update_model({"pinch_click": 0.5})
    assert f.accept(ev("click"))
    f2 = GestureFusion(csi_weight=0.0, threshold=0.5)
    f2.update_model({"pinch_click": 0.49})
    assert not f2.accept(ev("click"))
    for w in (-0.1, 1.1):
        with pytest.raises(ValueError):
            GestureFusion(csi_weight=w)


def test_event_class_map():
    assert EVENT_TO_CLASS["double_click"] == "double_pinch"
    assert EVENT_TO_CLASS["clutch"] == "palm_clutch"


def test_bus_publish_subscribe_and_unsubscribe():
    bus = EventBus()
    got, star = [], []
    unsub = bus.subscribe("a", lambda e: got.append(e.data["v"]))
    bus.subscribe("*", lambda e: star.append(e.topic))
    bus.publish("a", {"v": 1}, ts=5.0)
    bus.publish("b", {"v": 2})
    unsub()
    unsub()  # idempotent
    bus.publish("a", {"v": 3})
    assert got == [1]
    assert star == ["a", "b", "a"]
    assert bus.history("a")[0].ts == 5.0
    assert [e.topic for e in bus.history()] == ["a", "b", "a"]


def test_bus_publish_copies_data_and_defaults():
    bus = EventBus()
    d = {"x": 1}
    e = bus.publish("t", d)
    d["x"] = 2
    assert e.data == {"x": 1}
    assert bus.publish("t").data == {}
    assert e.ts > 0


def test_bus_isolates_failing_handlers():
    bus = EventBus()
    ok = []

    def bad(e):
        raise RuntimeError("boom")

    bus.subscribe("t", bad)
    bus.subscribe("t", lambda e: ok.append(1))
    bus.publish("t")
    assert ok == [1]
    assert bus.errors[0][0] == "t"
    assert isinstance(bus.errors[0][1], RuntimeError)


def test_bus_history_is_bounded():
    bus = EventBus(history=3)
    for i in range(5):
        bus.publish("t", {"i": i})
    assert [e.data["i"] for e in bus.history()] == [2, 3, 4]
    bus2 = EventBus(history=3)
    for i in range(3):
        bus2.publish("t", {"i": i})
    assert len(bus2.history()) == 3
