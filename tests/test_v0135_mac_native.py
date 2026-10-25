"""v0.1.35: a native-feeling Mac app. The menu bar icon is a template image (black + alpha, tinted by macOS for a light
or dark bar; state shown by shape), notifications go through UserNotifications with Project Away's window as the
fallback, and the app icon is generated from code (packaging/icon/make_icons.py)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PIL")

from away.app import icon_art, macnotify, tray

ROOT = Path(__file__).resolve().parents[1]


# -- the template menu bar icon ------------------------------------------------------------------------------------
def test_template_icon_is_black_plus_alpha_only_for_every_state() -> None:
    for state in ("active", "paused", "muted", "stopped", "recording"):
        img = icon_art.template_icon(state, 44, agent_dot=state == "active")
        assert img.mode == "RGBA" and img.size == (44, 44)
        r, g, b, a = img.split()
        assert r.getextrema() == g.getextrema() == b.getextrema() == (0, 0)  # macOS tints only the alpha
        assert a.getextrema()[1] > (100 if state == "stopped" else 200)  # something is drawn (stopped: dimmed)


def test_template_states_differ_by_shape_not_colour() -> None:
    alphas = {s: icon_art.template_icon(s, 44).getchannel("A").tobytes() for s in ("active", "paused", "muted")}
    assert len(set(alphas.values())) == 3
    ink = {s: sum(icon_art.template_icon(s, 44).getchannel("A").tobytes()) for s in ("active", "paused", "stopped")}
    assert ink["paused"] < ink["active"] * 0.7  # an outline, not a filled hand
    assert ink["stopped"] < ink["active"] * 0.6  # dimmed


def test_the_hand_is_centred_in_the_menu_bar_icon() -> None:
    box = icon_art.template_icon("active", 88).getchannel("A").getbbox()
    assert box is not None
    assert abs((box[0] + box[2]) / 2 - 44) <= 2 and abs((box[1] + box[3]) / 2 - 44) <= 2


def test_tray_icon_image_is_the_template_on_a_mac_and_the_coloured_disc_elsewhere() -> None:
    mac = tray.tray_icon_image(True, platform="darwin")
    assert mac.size == (tray.MAC_ICON_PX, tray.MAC_ICON_PX) and mac.getchannel("R").getextrema() == (0, 0)
    win = tray.tray_icon_image(True, platform="win32")
    assert win.getpixel((32, 6))[:3] == tray.STATE_COLORS["paused"][:3]
    assert tray.tray_icon_image(False, state="muted", platform="linux").getpixel((32, 6)) == tray.STATE_COLORS["muted"]


def test_tray_icon_image_falls_back_to_the_coloured_icon_if_the_template_cannot_be_drawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_a: Any, **_k: Any) -> Any:
        raise AttributeError("old Pillow")

    monkeypatch.setattr(icon_art, "template_icon", broken)
    assert tray.tray_icon_image(False, platform="darwin").getpixel((32, 6)) == tray.STATE_COLORS["active"]


# -- pystray's Mac Icon, made native -------------------------------------------------------------------------------
class _NSImage:
    def __init__(self, data: bytes) -> None:
        self.data, self.template, self._size = data, False, None

    @classmethod
    def alloc(cls) -> Any:
        return types.SimpleNamespace(initWithData_=lambda d: cls(d))

    def setSize_(self, size: Any) -> None:
        self._size = tuple(size)

    def size(self) -> Any:
        return self._size

    def setTemplate_(self, flag: bool) -> None:
        self.template = flag


class _Button:
    def __init__(self) -> None:
        self.images: list[Any] = []

    def setImage_(self, img: Any) -> None:
        self.images.append(img)


class _FakeDarwinIcon:
    """The parts of pystray._darwin.Icon the subclass relies on."""

    def __init__(self, image: Any) -> None:
        self._icon = image
        self._icon_image = None
        self._status_bar = types.SimpleNamespace(thickness=lambda: 22.0)
        self._button = _Button()
        self._status_item = types.SimpleNamespace(button=lambda: self._button)
        self.plain_calls = 0
        self.run_setup: Any = "unset"

    def _assert_image(self) -> None:
        self.plain_calls += 1

    def run(self, setup: Any = None) -> None:
        self.run_setup = setup


@pytest.fixture
def fake_appkit(monkeypatch: pytest.MonkeyPatch) -> Any:
    appkit = types.ModuleType("AppKit")
    appkit.NSImage = _NSImage  # type: ignore[attr-defined]
    appkit.NSScreen = types.SimpleNamespace(  # type: ignore[attr-defined]
        screens=lambda: [types.SimpleNamespace(backingScaleFactor=lambda: 1.0)]
    )
    foundation = types.ModuleType("Foundation")
    foundation.NSData = lambda b: b  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setitem(sys.modules, "Foundation", foundation)
    return appkit


def test_status_item_image_is_a_retina_template_sized_in_points(fake_appkit: Any) -> None:
    from PIL import Image

    cls = tray.mac_template_icon_class(_FakeDarwinIcon)
    icon = cls(tray.tray_icon_image(platform="darwin"))
    icon._assert_image()
    (img,) = icon._button.images
    assert img.template is True and img.size() == (22, 22) and icon.plain_calls == 0
    import io

    assert Image.open(io.BytesIO(img.data)).size == (44, 44)  # at least 2x, even when every screen is 1x
    icon._assert_image()  # cached: same bar thickness, nothing rebuilt
    assert len(icon._button.images) == 1


def test_a_failing_template_image_falls_back_to_pystrays_plain_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "AppKit", None)  # import fails
    icon = tray.mac_template_icon_class(_FakeDarwinIcon)(tray.tray_icon_image(platform="darwin"))
    icon._assert_image()
    assert icon.plain_calls == 1


def test_the_first_show_goes_through_the_main_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    queued: list[Any] = []
    monkeypatch.setattr(tray, "on_main_thread", lambda fn: queued.append(fn))
    icon = tray.mac_template_icon_class(_FakeDarwinIcon)(object())
    icon.visible = False
    icon.run()
    icon.run_setup(icon)  # what pystray's setup thread does
    assert icon.visible is False and len(queued) == 1
    queued[0]()
    assert icon.visible is True
    custom = object()
    icon.run(custom)
    assert icon.run_setup is custom


def test_only_pystrays_mac_backend_is_subclassed() -> None:
    def fake_icon(*_a: Any) -> Any:
        return None

    assert tray.mac_template_icon_class(fake_icon) is fake_icon  # type: ignore[arg-type]
    assert tray.mac_template_icon_class(dict) is dict


# -- UserNotifications ----------------------------------------------------------------------------------------------
class _Content:
    def __init__(self) -> None:
        self.title = self.body = None

    @classmethod
    def alloc(cls) -> Any:
        return types.SimpleNamespace(init=cls)

    def setTitle_(self, t: str) -> None:
        self.title = t

    def setBody_(self, b: str) -> None:
        self.body = b


class _Center:
    def __init__(self) -> None:
        self.auth: list[Any] = []
        self.posted: list[Any] = []
        self.delegate: Any = None
        self.post_error: Any = None

    def setDelegate_(self, d: Any) -> None:
        self.delegate = d

    def requestAuthorizationWithOptions_completionHandler_(self, opts: int, handler: Any) -> None:
        self.auth.append((opts, handler))

    def addNotificationRequest_withCompletionHandler_(self, request: Any, handler: Any) -> None:
        self.posted.append(request)
        handler(self.post_error)


def _framework(center: _Center) -> Any:
    return types.SimpleNamespace(
        UNUserNotificationCenter=types.SimpleNamespace(currentNotificationCenter=lambda: center),
        UNMutableNotificationContent=_Content,
        UNNotificationRequest=types.SimpleNamespace(
            requestWithIdentifier_content_trigger_=lambda ident, content, trigger: (ident, content, trigger)
        ),
    )


@pytest.fixture
def no_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macnotify, "_presenting_delegate", lambda: "DELEGATE")


def test_notifications_wait_for_permission_then_post(no_delegate: None) -> None:
    center = _Center()
    n = macnotify.MacNotifier(_framework(center), bundle_id=lambda: macnotify.BUNDLE_ID)
    fell_back: list[Any] = []
    assert n.notify("Project Away", "one", fallback=lambda *a: fell_back.append(a)) is True
    assert n.notify("Project Away", "two") is True
    assert len(center.auth) == 1 and n.state == "asking" and not center.posted  # asked once, both queued
    assert center.auth[0][0] == macnotify.AUTH_OPTIONS and center.delegate == "DELEGATE"
    center.auth[0][1](True, None)  # the user allowed it (a background queue)
    assert n.state == "granted" and [r[1].body for r in center.posted] == ["one", "two"]
    assert center.posted[0][1].title == "Project Away" and center.posted[0][2] is None  # delivered now
    assert n.notify("Project Away", "three") is True and center.posted[-1][1].body == "three"
    assert len(center.auth) == 1 and not fell_back


def test_denied_permission_hands_queued_messages_to_the_fallback_and_later_ones_back(no_delegate: None) -> None:
    center = _Center()
    n = macnotify.MacNotifier(_framework(center), bundle_id=lambda: macnotify.BUNDLE_ID)
    fell_back: list[Any] = []
    n.notify("Project Away", "queued", fallback=lambda *a: fell_back.append(a))
    center.auth[0][1](False, "denied")
    assert n.state == "denied" and fell_back == [("Project Away", "queued")] and not center.posted
    assert n.notify("Project Away", "later") is False  # the caller shows its window right away


def test_a_failed_post_uses_the_fallback(no_delegate: None) -> None:
    center = _Center()
    n = macnotify.MacNotifier(_framework(center), bundle_id=lambda: macnotify.BUNDLE_ID)
    n.state = "granted"
    center.post_error = "UNErrorDomain 1"
    fell_back: list[Any] = []
    assert n.notify("T", "M", fallback=lambda *a: fell_back.append(a)) is True
    assert fell_back == [("T", "M")]


def test_outside_the_app_bundle_the_framework_is_never_touched() -> None:
    def explode() -> Any:
        raise AssertionError("currentNotificationCenter() must not be called from a bare python")

    fw = types.SimpleNamespace(UNUserNotificationCenter=types.SimpleNamespace(currentNotificationCenter=explode))
    for ident in (None, "org.python.python", "com.example.other"):
        n = macnotify.MacNotifier(fw, bundle_id=lambda ident=ident: ident)
        assert n.notify("T", "M") is False and n.state == "unavailable"


def test_a_missing_framework_or_an_objc_exception_means_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "UserNotifications", None)
    n = macnotify.MacNotifier(bundle_id=lambda: macnotify.BUNDLE_ID)
    assert n.notify("T", "M") is False and n.state == "unavailable"

    def raises() -> Any:
        raise RuntimeError("NSInternalInconsistencyException: bundleProxyForCurrentProcess is nil")

    fw = types.SimpleNamespace(UNUserNotificationCenter=types.SimpleNamespace(currentNotificationCenter=raises))
    n = macnotify.MacNotifier(fw, bundle_id=lambda: macnotify.BUNDLE_ID)
    assert n.notify("T", "M") is False and n.state == "unavailable"


def test_a_failing_permission_request_falls_back(no_delegate: None) -> None:
    center = _Center()

    def broken(*_a: Any) -> None:
        raise RuntimeError("no")

    center.requestAuthorizationWithOptions_completionHandler_ = broken  # type: ignore[method-assign]
    n = macnotify.MacNotifier(_framework(center), bundle_id=lambda: macnotify.BUNDLE_ID)
    fell_back: list[Any] = []
    assert n.notify("T", "M", fallback=lambda *a: fell_back.append(a)) is True
    assert n.state == "denied" and fell_back == [("T", "M")]


def test_frozen_notify_on_a_mac_prefers_the_system_and_falls_back_to_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from away.app import frozen

    window: list[Any] = []
    monkeypatch.setattr(frozen, "mac_window_notify", lambda t, m: window.append((t, m)) or True)
    posted: list[Any] = []
    monkeypatch.setattr(macnotify, "notify", lambda t, m, fallback=None: posted.append((t, m, fallback)) or True)
    assert frozen.notify("Project Away", "hi", platform="darwin") is True
    assert posted == [("Project Away", "hi", frozen.mac_window_notify)] and not window
    monkeypatch.setattr(macnotify, "notify", lambda t, m, fallback=None: False)
    assert frozen.notify("Project Away", "again", platform="darwin") is True
    assert window == [("Project Away", "again")]


@pytest.mark.skipif(sys.platform != "darwin", reason="real PyObjC")
def test_real_bare_python_is_not_the_app_bundle_so_no_system_notification() -> None:
    assert macnotify.bundle_identifier() != macnotify.BUNDLE_ID
    assert macnotify.MacNotifier(bundle_id=macnotify.bundle_identifier).notify("T", "M") is False


# -- the app icon -----------------------------------------------------------------------------------------------------
def test_app_icon_assets_are_committed_and_wired_into_the_mac_bundle() -> None:
    from PIL import Image

    icon_dir = ROOT / "packaging" / "icon"
    master = Image.open(icon_dir / "ProjectAway-1024.png")
    assert master.size == (1024, 1024) and master.mode == "RGBA"
    assert master.getpixel((4, 4))[3] == 0 and master.getpixel((512, 160))[3] == 255  # margin, then the tile
    assert (icon_dir / "ProjectAway.icns").read_bytes()[:4] == b"icns"
    spec = (ROOT / "packaging" / "pyinstaller" / "away.spec").read_text(encoding="utf-8")
    assert "ProjectAway.icns" in spec and "UserNotifications" in spec
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pyobjc-framework-UserNotifications" in pyproject


# ------------------------------------------------------------------ stale permission entries (v0.1.35)
class _HS:
    kAXTrustedCheckOptionPrompt = "AXTrustedCheckOptionPrompt"

    def __init__(self, trusted: bool) -> None:
        self.trusted = trusted

    def AXIsProcessTrusted(self) -> bool:
        return self.trusted


def _runner(identity: str, calls: list[list[str]]):
    from types import SimpleNamespace

    def run(cmd, **_kw):
        calls.append(list(cmd))
        if cmd[0] == "codesign":
            return SimpleNamespace(stdout="", stderr=f"Executable=/x\ndesignated => {identity}\n")
        return SimpleNamespace(stdout="", stderr="")

    return run


EXE = "/Applications/Project Away.app/Contents/MacOS/ProjectAway"


def test_a_differently_signed_build_clears_its_stale_permissions_once(tmp_path) -> None:
    # an ad-hoc 0.1.34 left "on" switches that belong to the old build; the signed build was asked forever
    from away.app import macperms

    calls: list[list[str]] = []
    run = _runner('identifier "com.fernandogarzaaa.projectaway" and certificate root = H"23a8"', calls)
    assert macperms.clear_stale(tmp_path, "darwin", _HS(False), EXE, run) is True
    resets = [c for c in calls if c[0] == "tccutil"]
    assert resets == [
        ["tccutil", "reset", "Accessibility", "com.fernandogarzaaa.projectaway"],
        ["tccutil", "reset", "ListenEvent", "com.fernandogarzaaa.projectaway"],
    ]
    calls.clear()
    # same identity again (you turned it off on purpose, or have not allowed it yet): never reset twice
    assert macperms.clear_stale(tmp_path, "darwin", _HS(False), EXE, run) is False
    assert not [c for c in calls if c[0] == "tccutil"]
    # a new certificate later: once more
    run2 = _runner('identifier "com.fernandogarzaaa.projectaway" and certificate root = H"ffff"', calls)
    assert macperms.clear_stale(tmp_path, "darwin", _HS(False), EXE, run2) is True


def test_stale_permissions_are_left_alone_when_trusted_off_a_mac_or_outside_the_app(tmp_path) -> None:
    from away.app import macperms

    calls: list[list[str]] = []
    run = _runner('cdhash H"abc"', calls)
    assert macperms.clear_stale(tmp_path, "darwin", _HS(True), EXE, run) is False  # already works
    assert macperms.clear_stale(tmp_path, "win32", _HS(False), EXE, run) is False
    assert macperms.clear_stale(tmp_path, "darwin", _HS(False), "/usr/bin/python3", run) is False  # not a .app
    assert calls == []
    assert macperms.app_bundle(EXE).name == "Project Away.app"
