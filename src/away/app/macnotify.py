"""v0.1.35 (macOS): real system notifications through the UserNotifications framework (UNUserNotificationCenter).

They show Project Away's name and icon in Notification Center and follow the user's notification settings (Focus,
banners or alerts, sound), unlike ``osascript`` notifications, which are credited to Script Editor (a click opens
it), and unlike Project Away's own message window (:func:`away.app.frozen.mac_message`), which is the fallback.

The framework only works inside an app bundle with a bundle identifier: from a bare ``python`` (a source checkout,
the tests) ``UNUserNotificationCenter.currentNotificationCenter()`` raises (or aborts in some macOS versions), so
it is never called unless the running bundle is the Project Away .app. Permission is asked once per process, the
first time a notification is sent (macOS remembers the answer and only shows its prompt the first time ever);
messages sent while the answer is pending wait for it. When the framework is missing, the bundle is wrong, the user
said no, or posting fails, :func:`notify` returns False (or hands the message to ``fallback`` when the failure is
only known later) so the caller shows its own window instead.

Every call here is thread-safe: UNUserNotificationCenter may be used from any thread and its completion handlers
run on a background queue (no AppKit, so none of this needs the main thread)."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any

BUNDLE_ID = "com.fernandogarzaaa.projectaway"
#: UNAuthorizationOptionBadge = 1, Sound = 2, Alert = 4
AUTH_OPTIONS = 2 | 4
#: UNNotificationPresentationOptionList = 1 << 3, Banner = 1 << 4 (shown even while Project Away is the active app)
PRESENT_OPTIONS = (1 << 3) | (1 << 4)

log = logging.getLogger("away.notify")


def bundle_identifier() -> str | None:
    """The running app bundle's identifier, or None outside an .app (a bare python, a test run)."""
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        path = str(bundle.bundlePath() or "")
        ident = bundle.bundleIdentifier()
    except Exception:
        return None
    return str(ident) if ident and path.endswith(".app") else None


class MacNotifier:
    """Posts notifications through UNUserNotificationCenter. ``framework`` is the ``UserNotifications`` module (a
    fake in tests) and ``bundle_id`` the running bundle's identifier; both are looked up on first use when None."""

    def __init__(
        self,
        framework: Any = None,
        bundle_id: Callable[[], str | None] = bundle_identifier,
        expected_bundle: str = BUNDLE_ID,
    ) -> None:
        self._framework = framework
        self._bundle_id = bundle_id
        self._expected = expected_bundle
        self._lock = threading.Lock()
        self._center: Any = None
        self._delegate: Any = None  # the center holds its delegate weakly; keep it alive here
        #: "unknown" (not asked yet), "asking", "granted", "denied", or "unavailable" (no framework / bundle)
        self.state = "unknown"
        self._pending: list[tuple[str, str, Callable[[str, str], Any] | None]] = []

    # -- setup ------------------------------------------------------------------
    def _load(self) -> Any:
        """The notification center, or None (and state "unavailable") when it cannot be used here."""
        if self._center is not None:
            return self._center
        if self._bundle_id() != self._expected:
            # never touch the framework outside the .app: currentNotificationCenter() raises (or aborts) there
            self.state = "unavailable"
            return None
        try:
            un = self._framework
            if un is None:
                import UserNotifications as un  # type: ignore[import-not-found,no-redef]

                self._framework = un
            self._center = un.UNUserNotificationCenter.currentNotificationCenter()
        except Exception as exc:  # ImportError (pyobjc-framework-UserNotifications missing) or an ObjC exception
            log.info("system notifications unavailable (%s); using Project Away's window", exc)
            self.state = "unavailable"
            return None
        self._set_delegate()
        return self._center

    def _set_delegate(self) -> None:
        """Banners also show while Project Away is the active app (its menu is open): without a delegate saying so,
        macOS silently drops a notification from the frontmost app."""
        try:
            self._delegate = _presenting_delegate()
            if self._delegate is not None:
                self._center.setDelegate_(self._delegate)
        except Exception as exc:
            log.debug("notification delegate not set: %s", exc)

    # -- posting ----------------------------------------------------------------
    def notify(self, title: str, message: str, fallback: Callable[[str, str], Any] | None = None) -> bool:
        """Post ``title``/``message``. False when the system cannot show it (the caller shows its own window now);
        True when it was posted, or queued until the user answers the permission prompt; if that answer is no or
        posting fails later, ``fallback(title, message)`` is called then."""
        with self._lock:
            if self.state in ("denied", "unavailable"):
                return False
            center = self._load()
            if center is None:
                return False
            if self.state == "granted":
                self._post(title, message, fallback)
                return True
            self._pending.append((title, message, fallback))
            if self.state == "asking":
                return True
            self.state = "asking"
        try:
            center.requestAuthorizationWithOptions_completionHandler_(AUTH_OPTIONS, self._on_authorized)
        except Exception as exc:
            log.info("notification permission request failed: %s", exc)
            self._on_authorized(False, None)
        return True

    def _on_authorized(self, granted: Any, error: Any) -> None:
        """The permission answer (a background queue): post what waited, or hand it to the fallback."""
        with self._lock:
            self.state = "granted" if granted else "denied"
            pending, self._pending = self._pending, []
        if not granted:
            log.info("notifications not allowed (%s); using Project Away's window", error or "denied in Settings")
        for title, message, fallback in pending:
            if granted:
                self._post(title, message, fallback)
            elif fallback is not None:
                fallback(title, message)

    def _post(self, title: str, message: str, fallback: Callable[[str, str], Any] | None) -> None:
        un = self._framework
        try:
            content = un.UNMutableNotificationContent.alloc().init()
            content.setTitle_(title)
            content.setBody_(message)
            request = un.UNNotificationRequest.requestWithIdentifier_content_trigger_(
                f"away-{uuid.uuid4().hex[:12]}", content, None
            )
        except Exception as exc:
            log.info("could not build the notification: %s", exc)
            if fallback is not None:
                fallback(title, message)
            return

        def added(error: Any) -> None:
            if error is not None:
                log.info("notification not shown: %s", error)
                if fallback is not None:
                    fallback(title, message)

        try:
            self._center.addNotificationRequest_withCompletionHandler_(request, added)
        except Exception as exc:
            added(exc)


_DELEGATE_CLASS: Any = None


def _presenting_delegate() -> Any:
    """An NSObject implementing ``userNotificationCenter:willPresentNotification:withCompletionHandler:`` (banner +
    list). The Objective-C class is registered once per process (a second definition with the same name fails)."""
    global _DELEGATE_CLASS
    if _DELEGATE_CLASS is None:
        import objc  # type: ignore[import-not-found]
        import UserNotifications  # type: ignore[import-not-found]  # noqa: F401  (registers the protocol)
        from Foundation import NSObject

        protocol = objc.protocolNamed("UNUserNotificationCenterDelegate")

        class AwayNotificationDelegate(NSObject, protocols=[protocol]):  # type: ignore[misc,call-arg]
            def userNotificationCenter_willPresentNotification_withCompletionHandler_(
                self, _center: Any, _note: Any, handler: Any
            ) -> None:
                handler(PRESENT_OPTIONS)

        _DELEGATE_CLASS = AwayNotificationDelegate
    return _DELEGATE_CLASS.alloc().init()


_NOTIFIER: MacNotifier | None = None
_NOTIFIER_LOCK = threading.Lock()


def notifier() -> MacNotifier:
    """The process-wide notifier (permission is asked once per process)."""
    global _NOTIFIER
    with _NOTIFIER_LOCK:
        if _NOTIFIER is None:
            _NOTIFIER = MacNotifier()
        return _NOTIFIER


def notify(title: str, message: str, fallback: Callable[[str, str], Any] | None = None) -> bool:
    """Post a system notification; False when the caller should show its own window instead."""
    try:
        return notifier().notify(title, message, fallback)
    except Exception as exc:  # never let a notification take the caller down
        log.info("system notification failed: %s", exc)
        return False
