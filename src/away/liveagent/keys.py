"""API keys for the live agent, kept in the operating system's own vault: Windows Credential Manager, the macOS
Keychain or the Secret Service on Linux (the ``keyring`` package, already installed with Agent Link).

A key is never written to the settings file, the config file, a log, the turn log, or Agent Link answers. Settings
only remembers *that* a provider has a key (``has_key``). The backend is injectable so tests use a fake vault.
"""

from __future__ import annotations

import contextlib
from typing import Any, Protocol

SERVICE = "ProjectAway-LiveAgent"


class Backend(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...


class KeyStoreError(RuntimeError):
    pass


class MemoryBackend:
    """A fake vault (tests)."""

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.data.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.data[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.data.pop((service, username), None)


def _system_backend() -> Any:
    try:
        import keyring

        return keyring
    except Exception as exc:  # pragma: no cover - keyring ships with the [agent] extra
        raise KeyStoreError(
            "this computer has no safe place for API keys ('keyring' is not installed: pip install keyring)"
        ) from exc


class KeyStore:
    def __init__(self, backend: Backend | None = None, service: str = SERVICE) -> None:
        self._backend = backend
        self.service = service

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = _system_backend()
        return self._backend

    def get(self, provider_id: str) -> str:
        try:
            return str(self.backend.get_password(self.service, str(provider_id)) or "")
        except KeyStoreError:
            raise
        except Exception:
            return ""

    def has(self, provider_id: str) -> bool:
        try:
            return bool(self.get(provider_id))
        except KeyStoreError:
            return False

    def set(self, provider_id: str, key: str) -> None:
        key = str(key or "").strip()
        if not key:
            raise KeyStoreError("the key is empty")
        if any(c in key for c in "\r\n\t ") or len(key) > 4096:
            raise KeyStoreError("that does not look like an API key (it has spaces or line breaks)")
        try:
            self.backend.set_password(self.service, str(provider_id), key)
        except KeyStoreError:
            raise
        except Exception as exc:
            raise KeyStoreError(f"could not save the key in the system vault: {type(exc).__name__}") from None

    def delete(self, provider_id: str) -> None:
        with contextlib.suppress(Exception):
            self.backend.delete_password(self.service, str(provider_id))


def mask(key: str) -> str:
    """For display only: ``sk-...9f3a``."""
    k = str(key or "")
    if len(k) <= 8:
        return "\u2022" * len(k)
    return f"{k[:3]}...{k[-4:]}"
