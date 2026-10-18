"""Proposal #12 (v0.1.28): the camera diagnose reports the Windows camera privacy consent in plain
words, points at the lens cover / kill key when a camera lists but never opens, and the privacy-denied
verdict carries the ``open:camera_privacy`` fix marker so Settings > Cameras shows the fix button."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from away.multicam import diagnose
from away.multicam.capture import CameraDevice


class FakeKey:
    """A fake winreg key handle: a context manager holding a name -> value mapping."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def __enter__(self) -> FakeKey:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class FakeWinreg:
    """A fake ``winreg`` module: ``keys`` maps (root, subkey-path) -> {value-name: value}."""

    HKEY_CURRENT_USER = "HKCU"
    HKEY_LOCAL_MACHINE = "HKLM"

    def __init__(self, keys: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.keys = keys

    def OpenKey(self, root: str, path: str) -> FakeKey:
        try:
            return FakeKey(self.keys[(root, path)])
        except KeyError:
            raise OSError(f"no such key: {root}\\{path}") from None

    def QueryValueEx(self, key: FakeKey, name: str) -> tuple[Any, int]:
        return key.values[name], 1


def consent_keys(value: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {("HKCU", diagnose.CONSENT_KEY): {"Value": value}}


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeCap:
    def __init__(self, opens: bool, clock: Clock) -> None:
        self._opens = opens
        self._clock = clock

    def isOpened(self) -> bool:
        return self._opens

    def set(self, _prop: int, _value: Any) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 0.0

    def read(self) -> tuple[bool, Any]:
        if not self._opens:
            return False, None
        self._clock.t += 0.05
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        pass


class FakeCV2:
    CAP_DSHOW, CAP_MSMF = 700, 1400
    CAP_PROP_FOURCC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = 6, 3, 4, 5
    CAP_PROP_EXPOSURE, CAP_PROP_AUTO_EXPOSURE, CAP_PROP_BUFFERSIZE = 15, 21, 38

    def __init__(self, opens: bool, clock: Clock) -> None:
        self._opens = opens
        self._clock = clock

    def VideoCapture(self, _src: Any, _api: Any = None) -> FakeCap:
        return FakeCap(self._opens, self._clock)


def one_camera() -> list[CameraDevice]:
    return [CameraDevice(0, r"\\?\usb#vid_aaaa&pid_0001", "JoyCam")]


def run(devices: list[CameraDevice], opens: bool, **kw: Any) -> list[str]:
    clock = Clock()
    return diagnose.run_diagnose(
        devices,
        cv2=FakeCV2(opens, clock),
        windows=True,
        backends=["dshow"],
        measure_s=0.2,
        clock=clock,
        sleep=clock.sleep,
        budget_s=None,
        **kw,
    ).lines


# ----------------------------------------------------------------------------- the consent store
def test_read_consent_off_windows_never_touches_winreg() -> None:
    assert diagnose.read_consent(platform="linux") is None
    assert diagnose.read_consent(platform="darwin") is None


def test_read_consent_reports_deny_and_allow() -> None:
    assert diagnose.read_consent(FakeWinreg(consent_keys("Deny"))) == "Deny"
    assert diagnose.read_consent(FakeWinreg(consent_keys("Allow"))) == "Allow"


def test_read_consent_falls_back_to_hklm() -> None:
    wr = FakeWinreg({("HKLM", diagnose.CONSENT_KEY): {"Value": "Deny"}})
    assert diagnose.read_consent(wr) == "Deny"


def test_read_consent_empty_hkcu_value_falls_back_to_hklm() -> None:
    # an empty Value is not a verdict; the HKLM fallback must still be tried
    wr = FakeWinreg(
        {
            ("HKCU", diagnose.CONSENT_KEY): {"Value": "  "},
            ("HKLM", diagnose.CONSENT_KEY): {"Value": "Deny"},
        }
    )
    assert diagnose.read_consent(wr) == "Deny"


def test_read_consent_empty_everywhere_means_could_not_check() -> None:
    wr = FakeWinreg(
        {
            ("HKCU", diagnose.CONSENT_KEY): {"Value": ""},
            ("HKLM", diagnose.CONSENT_KEY): {"Value": "  "},
        }
    )
    assert diagnose.read_consent(wr) is None


def test_read_consent_none_value_falls_back_to_hklm() -> None:
    wr = FakeWinreg(
        {
            ("HKCU", diagnose.CONSENT_KEY): {"Value": None},
            ("HKLM", diagnose.CONSENT_KEY): {"Value": "Allow"},
        }
    )
    assert diagnose.read_consent(wr) == "Allow"


def test_read_consent_missing_key_means_could_not_check() -> None:
    assert diagnose.read_consent(FakeWinreg({})) is None


def test_read_consent_graceful_when_winreg_unimportable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "winreg", None)  # `import winreg` raises ImportError everywhere
    assert diagnose.read_consent(platform="win32") is None


# --------------------------------------------------------------------------------- the plain words
def test_consent_line_deny_is_plain_words_with_fix_marker() -> None:
    line = diagnose.consent_line(lambda: "Deny")
    assert line is not None
    assert line.startswith("[fix:open:camera_privacy]")
    assert "blocking" in line and "privacy" in line.lower()
    assert "\u2014" not in line  # plain words: no em dashes


def test_consent_line_allow_is_plain_words_without_marker() -> None:
    line = diagnose.consent_line(lambda: "Allow")
    assert line is not None
    assert "lets apps use the camera" in line
    assert "[fix:" not in line


def test_consent_line_unreadable_is_skipped() -> None:
    assert diagnose.consent_line(lambda: None) is None
    assert diagnose.consent_line(lambda: "something-unexpected") is None


def test_split_fix_marker() -> None:
    action, text = diagnose.split_fix_marker("[fix:open:camera_privacy] some words")
    assert action == "open:camera_privacy" and text == "some words"
    action, text = diagnose.split_fix_marker("a plain line")
    assert action is None and text == "a plain line"


def test_wired_fix_target_exists_in_open_targets() -> None:
    from away.selftest import fixes

    assert diagnose.PRIVACY_FIX == "open:camera_privacy"
    kind, _, target = diagnose.PRIVACY_FIX.partition(":")
    assert kind == "open" and target in fixes.OPEN_TARGETS  # the marker's button resolves to a real fix
    res = fixes.apply(diagnose.PRIVACY_FIX)
    assert res["ok"] and "privacy" in str(res["message"]).lower()


# ---------------------------------------------------------------------------- wired into diagnose
def test_diagnose_reports_privacy_first() -> None:
    lines = run(one_camera(), True, consent=lambda: "Deny")
    assert lines[0].startswith("[fix:open:camera_privacy] Camera privacy:")
    assert "blocking" in lines[0]


def test_diagnose_skips_privacy_line_when_unreadable() -> None:
    lines = run(one_camera(), True, consent=lambda: None)
    assert not any("Camera privacy" in ln for ln in lines)


def test_diagnose_skips_privacy_line_by_default() -> None:
    lines = run(one_camera(), True)
    assert not any("Camera privacy" in ln for ln in lines)


def test_diagnose_reports_privacy_even_with_no_cameras() -> None:
    lines = run([], True, consent=lambda: "Deny")
    assert lines[0].startswith("[fix:open:camera_privacy]")
    assert lines[1] == "no cameras found"


def test_never_opens_gets_shutter_guidance() -> None:
    lines = run(one_camera(), False, consent=lambda: "Allow")
    assert any("did not open" in ln for ln in lines)
    assert any("lens cover" in ln and "kill key" in ln for ln in lines)


def test_opens_gets_no_shutter_guidance() -> None:
    lines = run(one_camera(), True, consent=lambda: "Allow")
    assert any("ok" in ln for ln in lines)
    assert not any("lens cover" in ln for ln in lines)


def test_stopped_run_emits_no_shutter_guidance() -> None:
    class LaterClock(Clock):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 0.0 if self.calls == 1 else 999.0  # the budget is spent before the first camera

    clock = LaterClock()
    lines = diagnose.run_diagnose(
        one_camera(),
        cv2=FakeCV2(False, clock),
        windows=True,
        backends=["dshow"],
        clock=clock,
        sleep=clock.sleep,
        budget_s=60.0,
        consent=lambda: "Allow",
    ).lines
    assert any("Camera privacy" in ln for ln in lines)
    assert not any("lens cover" in ln for ln in lines)
