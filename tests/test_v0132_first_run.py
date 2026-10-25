"""v0.1.32: a smoother first run, mostly on macOS.

* the release self-test builds the real tray menu (0.1.29/0.1.30 crashed there at start-up, unnoticed by CI)
"""

from __future__ import annotations

from typing import Any

import pytest

from away.app import selftest, tray

TRAY_MENU = [c for c in selftest.CHECKS if c.name == "tray-menu"]


def _real_pystray(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("PYSTRAY_BACKEND", "dummy")  # the menu code is backend independent; no display needed
    return pytest.importorskip("pystray")


def test_the_release_self_test_builds_the_real_tray_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    _real_pystray(monkeypatch)
    assert TRAY_MENU and TRAY_MENU[0].needs_display  # runs under the release jobs' display (xvfb on Linux)
    res = selftest.run(TRAY_MENU, display=True)["tray-menu"]
    assert res["ok"] is True, res
    assert "menu items accepted by pystray" in str(res["detail"])


def test_the_tray_menu_check_fails_on_the_0130_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    _real_pystray(monkeypatch)

    def wrapper(*args: Any, **kwargs: Any) -> None:  # the bare wrapper menu_safe used to return
        return None

    monkeypatch.setattr(tray.TrayMenuCallbacks, "on_pause", wrapper)
    res = selftest.run(TRAY_MENU, display=True)["tray-menu"]
    assert res["ok"] is False and "ValueError" in str(res["detail"])  # the release build would now fail


# ------------------------------------------------------------------------------------- release page install notes
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_every_release_page_gets_how_to_install_notes() -> None:
    notes = (ROOT / "packaging" / "release-install-notes.md").read_text(encoding="utf-8")
    assert 'xattr -dr com.apple.quarantine "/Applications/Project Away.app"' in notes  # the step that works
    assert "ProjectAway-<version>-macos-arm64.dmg" in notes and "windows-x64-setup.exe" in notes
    yml = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    publish = yml[yml.index("\n  publish:") :]
    assert "sparse-checkout: packaging/release-install-notes.md" in publish  # the job had no checkout before
    assert 'sed "s/<version>/${GITHUB_REF_NAME#v}/g" packaging/release-install-notes.md > install-notes.md' in publish
    assert "body_path: install-notes.md" in publish and "append_body: true" in publish  # under the release's notes


# --------------------------------------------------------------------------- macOS: no Script Editor notifications
def test_mac_messages_use_project_aways_own_window(tmp_path: Any) -> None:
    import json
    import os

    from away.app import frozen

    old = tmp_path / "message-old.json"
    old.write_text("{}", encoding="utf-8")
    os.utime(old, (0, 0))
    spawned: list[list[str]] = []
    ok = frozen.mac_message(
        "Project Away", "Models ready.", folder=tmp_path, spawn=lambda argv, **kw: spawned.append(argv)
    )
    assert ok and len(spawned) == 1
    assert spawned[0][-3:-1] == ["agent-ui", "toast"]  # the app's own message window, not osascript
    spec = json.loads(__import__("pathlib").Path(spawned[0][-1]).read_text(encoding="utf-8"))
    assert spec["title"] == "Project Away" and spec["text"] == "Models ready."
    assert not old.exists()  # old message files are tidied away

    def fail(argv: list[str], **kw: Any) -> None:
        raise OSError("no exe")

    assert frozen.mac_message("t", "m", folder=tmp_path, spawn=fail) is False  # notify() then falls back


def test_the_tray_does_not_use_pystray_notifications_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from away.app import frozen

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(frozen, "notify", lambda t, m, platform=None: shown.append((t, m)) or True)
    monkeypatch.setattr(tray.sys, "platform", "darwin")
    app = MagicMock()
    cb = tray.TrayMenuCallbacks(tray.TrayController(app), app, MagicMock(), tray.TrayGuard())
    cb.icon = MagicMock()
    assert cb.icon_notifications is False
    cb.notify("Project Away", "hello")
    assert shown == [("Project Away", "hello")] and not cb.icon.notify.called  # pystray's runs osascript
    monkeypatch.setattr(tray.sys, "platform", "win32")
    other = tray.TrayMenuCallbacks(tray.TrayController(app), app, MagicMock(), tray.TrayGuard())
    other.icon = MagicMock()
    assert other.icon_notifications is True
    other.notify("Project Away", "hi")
    other.icon.notify.assert_called_once_with("hi", "Project Away")


# --------------------------------------------------------------------------------------- setup progress window
def test_progress_reads_the_setup_output() -> None:
    from away.app.setup_window import Progress

    p = Progress()
    assert p.snapshot().percent is None and p.snapshot().line == ""
    p.feed("  hand_landmarker.task: downloading 7.8 MB ...\n")
    assert p.snapshot().line == "hand_landmarker.task: downloading 7.8 MB ..." and p.snapshot().percent is None
    p.feed("\r[####......]  41%  664/1620 MB  5.2 MB/s  ETA 3m04s")
    assert p.snapshot().percent == 41 and p.snapshot().line.startswith("[####")
    p.feed("\r[##########] 100%  1620/1620 MB\n\n")
    assert p.snapshot().percent == 100
    p.feed("  face_landmarker.task: downloading 3.8 MB ...\n")  # the next file: no percentage for it yet
    assert p.snapshot().percent is None
    p.feed("   \n")  # blank output changes nothing
    assert p.snapshot().line.startswith("face_landmarker")
    p.feed("999%")
    assert p.snapshot().percent == 100
    assert not p.snapshot().done
    p.finish(0)
    assert p.snapshot().done and p.snapshot().rc == 0


def test_setup_runs_while_the_window_shows_its_progress() -> None:
    import io
    import sys

    from away.app import setup_window

    def cli_main(argv: list[str]) -> int:
        assert argv == ["setup", "--yes"]
        print("  hand_landmarker.task: downloading 7.8 MB ...")
        sys.stdout.write("\r[####......]  41%  3/7 MB")
        return 0

    seen: list[Any] = []

    def window(progress: Any, intro: str) -> None:
        while not progress.snapshot().done:
            pass
        seen.append((intro, progress.snapshot()))

    log = io.StringIO()
    out_before = sys.stdout
    assert setup_window.run(["setup", "--yes"], cli_main, window=window, log=log, intro="hi") == 0
    assert sys.stdout is out_before  # restored
    intro, snap = seen[0]
    assert intro == "hi" and snap.percent == 41 and snap.rc == 0
    assert "hand_landmarker.task: downloading" in log.getvalue()  # the log still gets everything


def test_setup_still_runs_when_no_window_can_open() -> None:
    from away.app import setup_window

    ran: list[list[str]] = []

    def no_display(progress: Any, intro: str) -> None:
        raise RuntimeError("no display name and no $DISPLAY")

    assert setup_window.run(["setup"], lambda a: ran.append(a) or 3, window=no_display) == 3
    assert ran == [["setup"]]  # once, without the window


def test_a_setup_that_crashes_or_exits_still_ends_the_window() -> None:
    from away.app import setup_window

    def boom(argv: list[str]) -> int:
        raise ValueError("disk full")

    def leave(argv: list[str]) -> int:
        raise SystemExit(4)

    def wait(progress: Any, intro: str) -> None:
        while not progress.snapshot().done:
            pass

    assert setup_window.run(["setup"], boom, window=wait) == 1
    assert setup_window.run(["setup"], leave, window=wait) == 4


def test_intro_and_tray_place_texts() -> None:
    from away.app import setup_window

    assert "about 1.6 GB" in setup_window.intro_text(1600) and "only once" in setup_window.intro_text(1600)
    assert "a few MB" in setup_window.intro_text(11)
    assert "menu bar" in setup_window.tray_place("darwin") and "tray" in setup_window.tray_place("linux")


def test_first_launch_uses_the_window_and_falls_back_without_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from away import config as cfgmod
    from away.app import frozen

    cfg = cfgmod.AppConfig()
    monkeypatch.setattr(frozen, "missing_models", lambda c, models_dir=None: ["hand_landmarker.task"])
    calls: list[Any] = []
    notes: list[str] = []
    cli = lambda argv: calls.append(("cli", argv)) or 0  # noqa: E731
    note = lambda t, m: notes.append(t)  # noqa: E731
    rc = frozen.first_launch(cli, cfg, notifier=note, window=lambda argv: calls.append(("window", argv)) or 0)
    assert rc == 0 and calls == [("window", ["setup", "--yes", "--skip-probe"])] and notes == []  # no notifications
    calls.clear()
    rc = frozen.first_launch(cli, cfg, notifier=note, window=lambda argv: None)  # the window process did not start
    assert rc == 0 and calls == [("cli", ["setup", "--yes", "--skip-probe"])] and len(notes) == 2


def test_the_setup_window_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from away.app import frozen

    got: list[list[str]] = []
    rc = frozen.setup_window_process(
        ["setup", "--yes"], run=lambda argv, **kw: got.append(argv) or SimpleNamespace(returncode=2)
    )
    assert rc == 2 and got[0][-3:] == ["setup-window", "setup", "--yes"]

    def fail(argv: list[str], **kw: Any) -> None:
        raise OSError("exec format error")

    assert frozen.setup_window_process(["setup"], run=fail) is None


def test_the_setup_window_command_and_its_helper_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from away import cli
    from away.app import crashlog, setup_window
    from away.selftest import env

    got: list[Any] = []
    monkeypatch.setattr(setup_window, "main", lambda args, cli_main: got.append(args) or 0)
    assert cli.main(["setup-window", "setup", "--yes", "--skip-probe", "--asr-model", "tiny"]) == 0
    assert got == [["setup", "--yes", "--skip-probe", "--asr-model", "tiny"]]
    assert "setup-window" not in cli.build_parser().format_help()  # hidden
    assert crashlog.entry_label(["setup-window", "setup", "--yes"]) == "helper setup-window"  # not a restart
    assert "setup-window" in env.HELPER_WORDS  # not counted as a second copy of the app


# ---------------------------------------------------------------------------------- macOS permissions
class FakeHIServices:
    kAXTrustedCheckOptionPrompt = "AXTrustedCheckOptionPrompt"

    def __init__(self, trusted: bool) -> None:
        self.trusted = trusted
        self.asked: list[Any] = []

    def AXIsProcessTrusted(self) -> bool:
        return self.trusted

    def AXIsProcessTrustedWithOptions(self, options: Any) -> bool:
        self.asked.append(options)
        return self.trusted


class FakeIOKit:
    def __init__(self, access: int) -> None:
        self.access = access
        self.requested: list[int] = []

    def IOHIDCheckAccess(self, kind: int) -> int:
        return self.access

    def IOHIDRequestAccess(self, kind: int) -> bool:
        self.requested.append(kind)
        return False


def test_permissions_are_only_checked_on_macos() -> None:
    from away.app import macperms

    assert macperms.accessibility(platform="win32", hiservices=FakeHIServices(False)) is None
    assert macperms.input_monitoring(platform="linux", iokit=FakeIOKit(1)) is None
    assert macperms.check_and_ask(lambda t, m: pytest.fail("no message"), platform="win32") == {
        "accessibility": None,
        "input_monitoring": None,
    }


def test_missing_permissions_make_macos_ask_and_one_message_says_what_to_allow() -> None:
    from away.app import macperms

    hs, io = FakeHIServices(False), FakeIOKit(2)  # Input Monitoring not decided yet
    msgs: list[tuple[str, str]] = []
    state = macperms.check_and_ask(lambda t, m: msgs.append((t, m)), platform="darwin", hiservices=hs, iokit=io)
    assert state == {"accessibility": False, "input_monitoring": False}
    assert hs.asked == [{"AXTrustedCheckOptionPrompt": True}]  # macOS shows its own request
    assert io.requested == [macperms.LISTEN_EVENT]
    assert len(msgs) == 1 and "Accessibility and Input Monitoring" in msgs[0][1] and "open it again" in msgs[0][1]


def test_granted_permissions_stay_quiet_and_errors_are_unknown() -> None:
    from away.app import macperms

    msgs: list[Any] = []
    state = macperms.check_and_ask(
        lambda t, m: msgs.append(m), platform="darwin", hiservices=FakeHIServices(True), iokit=FakeIOKit(0)
    )
    assert state == {"accessibility": True, "input_monitoring": True} and msgs == []

    class Broken:
        def __getattr__(self, name: str) -> Any:
            raise OSError("no IOKit")

    assert macperms.input_monitoring(platform="darwin", iokit=Broken()) is None  # unknown, never a crash
    assert macperms.accessibility(platform="darwin", hiservices=Broken()) is None
    assert macperms.missing_names({"accessibility": None, "input_monitoring": False}) == ["Input Monitoring"]


def test_the_permissions_menu_item_opens_the_right_settings_page(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from away.app import macperms

    opened: list[list[str]] = []
    assert macperms.open_settings("input_monitoring", run=lambda argv, **kw: opened.append(argv))
    assert opened == [["open", macperms.SETTINGS["input_monitoring"]]]
    assert "Privacy_Accessibility" in macperms.SETTINGS["accessibility"]

    app = MagicMock()
    cb = tray.TrayMenuCallbacks(tray.TrayController(app), app, MagicMock(), tray.TrayGuard())
    shown: list[str] = []
    monkeypatch.setattr(cb, "notify", lambda t, m: shown.append(m))
    pages: list[str] = []
    monkeypatch.setattr(macperms, "open_settings", lambda which="accessibility", run=None: pages.append(which) or True)
    monkeypatch.setattr(macperms, "accessibility", lambda *a, **k: True)
    monkeypatch.setattr(macperms, "input_monitoring", lambda *a, **k: False)
    cb.on_mac_permissions(None, None)
    assert pages == ["input_monitoring"] and shown == []
    monkeypatch.setattr(macperms, "input_monitoring", lambda *a, **k: True)
    cb.on_mac_permissions(None, None)
    assert pages == ["input_monitoring"] and len(shown) == 1  # all allowed: say so, open nothing


def test_the_permissions_check_runs_in_the_background(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from away.app import macperms

    got: list[Any] = []
    monkeypatch.setattr(macperms, "check_and_ask", lambda notify: got.append(notify))
    cb = SimpleNamespace(notify=lambda t, m: None)
    t = tray.ask_mac_permissions(cb)
    t.join(5)
    assert got == [cb.notify] and t.daemon


def test_the_permissions_menu_item_shows_on_macos_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    pystray = _real_pystray(monkeypatch)
    from away.liveagent.integrate import menu_items

    for plat, visible in (("darwin", True), ("win32", False)):
        monkeypatch.setattr(tray.sys, "platform", plat)
        app = MagicMock()
        ctl = tray.TrayController(app)
        cb = tray.TrayMenuCallbacks(ctl, app, MagicMock(), tray.TrayGuard())
        menu = tray.build_tray_menu(pystray, ctl, app, cb, MagicMock(), menu_items, lambda _t: None)
        item = next(i for i in menu.items if i.text == "Allow pointer and keyboard control...")
        assert item.visible is visible


# --------------------------------------------------------------------------- voice downloads: checked by hash
def test_a_voice_file_of_the_right_size_but_wrong_content_is_not_installed(tmp_path: Any) -> None:
    import hashlib

    from away.voice import natural as N

    good = b"model" * 100
    a = N.Asset("v.onnx", "https://x/v.onnx", len(good), hashlib.sha256(good).hexdigest())
    p = N._path(tmp_path, a)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * len(good))  # same size, different bytes (damaged or replaced)
    assert N._have(tmp_path, a) is False
    p.write_bytes(good)
    assert N._have(tmp_path, a) is True
    assert N._stamp(p).read_text(encoding="utf-8") == a.sha256  # hashed once, then remembered
    assert N._have(tmp_path, a) is True


def test_warm_up_touches_the_hiservices_names_on_macos_only() -> None:
    """v0.1.33: PyObjC fills HIServices in lazily, not thread-safely; pynput's two listener threads raced on it
    (KeyError: 'AXIsProcessTrusted'). The main thread resolves the names first."""
    from away.app import macperms

    seen: list[str] = []

    class Spy:
        def __getattr__(self, name: str) -> Any:
            seen.append(name)
            return object()

    assert macperms.warm_up(platform="win32", hiservices=Spy()) is False and seen == []
    assert macperms.warm_up(platform="darwin", hiservices=Spy()) is True
    assert seen == ["AXIsProcessTrusted", "AXIsProcessTrustedWithOptions", "kAXTrustedCheckOptionPrompt"]


def test_warm_up_never_raises() -> None:
    from away.app import macperms

    class Broken:
        def __getattr__(self, name: str) -> Any:
            raise KeyError(name)

    assert macperms.warm_up(platform="darwin", hiservices=Broken()) is False
