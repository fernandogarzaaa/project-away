"""v0.1.28 (proposal #11): a driver-broken camera gets fix buttons.

``open:device_manager`` opens Device Manager (``devmgmt.msc``) through the injectable launcher pattern, and
``guide:uninstall_driver`` is a Yes/No-confirmed guided repair (uninstall the camera, restart the PC). ``_camera_fix``
offers them when the verdict is ``driver``: from an explicit verdict, from PR #46's ``driver_state`` rows
(``{"name", "status", "code", "meaning"}``) when the environment carries them, or from the prose message as a
fallback. Everything is unit-tested with fake envs and injected launchers/prompts: no camera, no Windows."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from away.selftest import checks as ck
from away.selftest import fixes
from away.selftest.model import FAIL


class FakeEnv:
    def __init__(self, **over: Any) -> None:
        self.data = over

    def __getattr__(self, name: str) -> Any:
        if name not in self.data:
            raise AttributeError(name)
        val = self.data[name]

        def fn(*_a: Any) -> Any:
            if isinstance(val, Exception):
                raise val
            return val

        return fn


DRIVER_ROWS = [
    {"name": "HD User Facing", "status": "Error", "code": 22, "meaning": "disabled"},
    {"name": "JoyCam", "status": "OK", "code": 0, "meaning": "working, no error"},
]


def driver_env(camera_users: dict[str, Any], name: str = "HD User Facing") -> FakeEnv:
    return FakeEnv(
        cameras=[{"name": name, "opened": False, "fps": 0, "error": "Element not found"}],
        camera_users=camera_users,
    )


def test_driver_verdict_offers_the_guided_repair() -> None:
    res = ck.check_cameras(
        driver_env({"message": "The camera will not start.", "users": [], "actions": ["retry"], "verdict": "driver"})
    )[0]
    assert res.status == FAIL
    assert res.fix is not None
    assert res.fix.action == "guide:uninstall_driver"
    assert res.fix.label == "Fix the camera driver"
    assert "driver" in res.summary.lower()


def test_driver_state_rows_detect_a_broken_driver() -> None:
    res = ck.check_cameras(
        driver_env(
            {
                "message": "HD User Facing could not be opened.",
                "users": [],
                "actions": ["retry"],
                "driver_state": DRIVER_ROWS,
            }
        )
    )[0]
    assert res.fix is not None and res.fix.action == "guide:uninstall_driver"
    assert "disabled" in res.summary  # the PR #46 "meaning" reaches the person in plain words


def test_driver_state_rows_leave_a_working_driver_alone() -> None:
    res = ck.check_cameras(
        driver_env(
            {
                "message": "JoyCam could not be opened.",
                "users": [],
                "actions": ["retry"],
                "driver_state": DRIVER_ROWS,
            },
            name="JoyCam",
        )
    )[0]
    assert res.fix is not None and res.fix.action == "retry_cameras"


def test_prose_verdict_falls_back_without_structured_rows() -> None:
    res = ck.check_cameras(
        driver_env(
            {
                "message": "HD User Facing could not be opened: its drivers need to be reinstalled.",
                "users": [],
                "actions": ["retry"],
            }
        )
    )[0]
    assert res.fix is not None and res.fix.action == "guide:uninstall_driver"


def test_plain_unknown_failure_keeps_try_again() -> None:
    res = ck.check_cameras(
        driver_env({"message": "HD User Facing could not be opened.", "users": [], "actions": ["retry"]})
    )[0]
    assert res.fix is not None and res.fix.action == "retry_cameras"


def test_held_camera_still_offers_to_close_the_app() -> None:
    res = ck.check_cameras(
        driver_env(
            {
                "message": "Your built-in camera is being used by Zoom. Close it, then Retry.",
                "users": [{"name": "Zoom", "path": "C:\\Zoom\\Zoom.exe", "packaged": False}],
                "actions": ["close:C:\\Zoom\\Zoom.exe", "retry"],
            }
        )
    )[0]
    assert res.fix is not None and res.fix.action == "close_app:C:\\Zoom\\Zoom.exe"


@pytest.fixture()
def win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")


def test_open_device_manager_launches_devmgmt(win: None) -> None:
    launched: list[list[str]] = []
    res = fixes.apply("open:device_manager", launcher=launched.append)
    assert res["ok"] and launched == [["mmc.exe", "devmgmt.msc"]]
    assert "Device Manager" in res["message"]


def test_open_device_manager_off_windows_is_honest(win: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    launched: list[list[str]] = []
    res = fixes.apply("open:device_manager", launcher=launched.append)
    assert res["ok"] and not launched
    assert "Device Manager" in res["message"]


def test_guided_repair_yes_opens_device_manager_and_hands_over_steps(win: None) -> None:
    launched: list[list[str]] = []
    res = fixes.apply("guide:uninstall_driver", launcher=launched.append, confirm=lambda _q: True)
    assert res["ok"] and launched == [["mmc.exe", "devmgmt.msc"]]
    assert "Uninstall device" in res["message"] and "restart the PC" in res["message"]


def test_guided_repair_no_changes_nothing(win: None) -> None:
    launched: list[list[str]] = []
    res = fixes.apply("guide:uninstall_driver", launcher=launched.append, confirm=lambda _q: False)
    assert res["ok"] and not launched
    assert "nothing changed" in res["message"].lower()


def test_guided_repair_without_a_prompt_gives_the_steps(win: None) -> None:
    launched: list[list[str]] = []
    res = fixes.apply("guide:uninstall_driver", launcher=launched.append)
    assert res["ok"]
    assert "Uninstall device" in res["message"] and "no answer came back" in res["message"]


class FakePrompts:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def ask(self, question: str, options: list[str], **_kw: Any) -> dict[str, Any]:
        self.asked.append(question)
        assert options == ["Yes", "No"]
        return {"answer": self.answer}


def host_with(answer: str) -> Any:
    return SimpleNamespace(core=SimpleNamespace(prompts=FakePrompts(answer)))


def test_guided_repair_uses_the_app_prompt_window(win: None) -> None:
    launched: list[list[str]] = []
    host = host_with("Yes")
    res = fixes.apply("guide:uninstall_driver", host=host, launcher=launched.append)
    assert res["ok"] and launched == [["mmc.exe", "devmgmt.msc"]]
    assert host.core.prompts.asked and "driver" in host.core.prompts.asked[0].lower()


def test_guided_repair_no_in_the_prompt_window_changes_nothing(win: None) -> None:
    launched: list[list[str]] = []
    res = fixes.apply("guide:uninstall_driver", host=host_with("No"), launcher=launched.append)
    assert res["ok"] and not launched


def test_unknown_guide_action_is_not_ok() -> None:
    assert not fixes.apply("guide:something_else")["ok"]
