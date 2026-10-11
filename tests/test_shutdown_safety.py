"""Shutdown safety tests that do not require a desktop session."""

import threading

from away.app.runner import AwayApp
from away.events import EventBus


class _Pipeline:
    def __init__(self, error: Exception | None = None) -> None:
        self.pauses: list[bool] = []
        self.error = error

    def set_paused(self, paused: bool) -> None:
        self.pauses.append(paused)
        if self.error is not None:
            raise self.error


class _Listener:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    def stop(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


def _app(pipeline: _Pipeline, *listeners: _Listener) -> AwayApp:
    app = object.__new__(AwayApp)
    app.stop_event = threading.Event()
    app.pipeline = pipeline
    app.bus = EventBus()
    app._listeners = list(listeners)
    app._input_listeners = []
    return app


def test_stop_pauses_pipeline_and_stops_listeners() -> None:
    pipeline = _Pipeline()
    listener = _Listener()
    app = _app(pipeline, listener)

    app.stop()

    assert app.stop_event.is_set()
    assert pipeline.pauses == [True]
    assert listener.calls == 1


def test_stop_continues_when_input_release_fails() -> None:
    pipeline = _Pipeline(RuntimeError("release failed"))
    listener = _Listener()
    app = _app(pipeline, listener)
    events = []
    app.bus.subscribe("*", events.append)

    app.stop()

    assert app.stop_event.is_set()
    assert listener.calls == 1
    assert any(event.topic == "mouse.release_error" for event in events)


def test_stop_continues_when_one_listener_stop_fails() -> None:
    pipeline = _Pipeline()
    broken = _Listener(RuntimeError("listener failed"))
    later = _Listener()
    app = _app(pipeline, broken, later)
    events = []
    app.bus.subscribe("*", events.append)

    app.stop()

    assert broken.calls == later.calls == 1
    assert any(event.topic == "listener.stop_error" for event in events)
