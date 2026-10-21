"""v0.1.29 (audit Phase 4, lane A item 6): three privacy residuals.

1. ``bundle.redact()`` only hid values under key names it already knew, so a secret stored under an
   unrecognized name (``private_key``, ``passphrase``) left the PC untouched in an exported bundle.
   Redaction now fails closed on secret-shaped key names, with word boundaries so innocent names
   (``keyboard``, ``monkey``, ``passage``, ``hotkey``) survive.
2. Turning the audit trail off only flipped a config flag; the session files and thumbnails already
   on disk stayed there. Off now means gone, and the setter reports what it removed.
3. Pruning a session deleted its JSONL but not its thumbnails; they left only via the global cap.
   Pruning now takes the session's PNGs with it.

Every test here fails on the unfixed code.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from away.agentlink.bundle import redact, redact_text
from away.agentlink.control import MAX_SESSIONS, ComputerControl, ControlHooks
from away.agentlink.desktop import Element
from away.agentlink.gui_model import AgentLinkModel
from away.control.mouse import RecordingBackend


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeUIA:
    available = True

    def focused(self) -> Element | None:
        return None

    def element_at(self, x: int, y: int) -> Element | None:
        return Element("Canvas", "PaneControl")

    def find(self, name: str, control_type: str | None = None) -> Element | None:
        return None


def _control(clock: Clock, tmp_path: Path, settings: Any = None) -> ComputerControl:
    hooks = ControlHooks(
        backend=RecordingBackend(),
        platform="linux",
        secure_desktop=lambda: False,
        foreground=lambda: None,
        grab=lambda: np.zeros((12, 12, 3), dtype=np.uint8),  # a real image, so thumbnails are written
        sleep=lambda _s: None,
        uia=FakeUIA(),
    )
    hooks.cursor_pos = lambda: (7, 8)
    return ComputerControl(
        hooks,
        lambda _q, _o, **kw: {"answer": "Allow", "method": "click"},
        ttl_s=1800,
        rate_per_s=10000.0,
        log_folder=tmp_path / "actions",
        clock=clock,
        step_budget=40,
        audit_trail=True,
        settings=settings,
    )


# ---------------------------------------------------------------------- 1. redaction


def test_redact_hides_values_under_unrecognized_secret_key_names() -> None:
    out = redact(
        {
            "private_key": "-----BEGIN PRIVATE KEY----- MIIB...",
            "passphrase": "correct horse battery staple",
            "seed_phrase": "legal winner thank error wave sausage worth useful legal winner thank yellow",
            "signing_key": "ed25519:AAAA",
            "signingKey": "ed25519:BBBB",
            "passcode": "918273",
            "nested": {"vault_token": "paway_should_not_matter", "mnemonic": "zoo zoo zoo"},
        }
    )
    assert out["private_key"] == "***"
    assert out["passphrase"] == "***"
    assert out["seed_phrase"] == "***"
    assert out["signing_key"] == "***"
    assert out["signingKey"] == "***"
    assert out["passcode"] == "***"
    assert out["nested"]["vault_token"] == "***"
    assert out["nested"]["mnemonic"] == "***"


def test_redact_keeps_innocent_keys_and_prose() -> None:
    data = {
        "keyboard_layout": "us",
        "monkey": "banana",
        "passage": "a short passage of prose",
        "hotkey": "ctrl+alt+k",
        "monkey_count": 3,
        "note": "the token bucket refills at 10 per second",
        "author": "Jane Doe",
    }
    assert redact(data) == data


def test_redact_hides_auth_derived_key_names() -> None:
    # Codex P1 on #96: "authorization" split into one token that matched neither
    # the substring list (no auth) nor the token set (only whole "auth"), so the
    # credential passed through. Whole-word coverage, not the auth substring,
    # so "author" stays innocent (see the test above).
    out = redact(
        {
            "authorization": "Basic dXNlcjpwYXNz",
            "authentication": "hunter2hunter2",
            "author": "Jane Doe",
        }
    )
    assert out["authorization"] == "***"
    assert out["authentication"] == "***"
    assert out["author"] == "Jane Doe"


def test_redact_text_hides_new_secret_key_value_shapes() -> None:
    out = redact_text("loaded private_key=abc123def and passphrase: hunter2hunter2 from the vault")
    assert "abc123def" not in out
    assert "hunter2hunter2" not in out


def test_bundle_export_redacts_unrecognized_secret_keys(tmp_path: Path) -> None:
    from away.agentlink import bundle

    path = bundle.build(
        tmp_path,
        stats={
            "private_key": "-----BEGIN PRIVATE KEY----- MIIB...",
            "passphrase": "correct horse battery staple",
            "keyboard_layout": "us",
        },
        note="check",
    )
    with zipfile.ZipFile(path) as z:
        stats = json.loads(z.read("stats.json"))
    assert stats["stats"]["private_key"] == "***"
    assert stats["stats"]["passphrase"] == "***"
    assert stats["stats"]["keyboard_layout"] == "us"


# ---------------------------------------------------------------------- 2. trail off means gone


def _live_settings(cfg_path: Path) -> Any:
    from away import config as cfgmod

    def read() -> Any:
        cfg, _ = cfgmod.load(cfg_path) if cfg_path.exists() else (cfgmod.AppConfig(), [])
        return cfg.agent_link

    return read


def _model_wired_to(ctl: ComputerControl, tmp_path: Path, cfg_path: Path) -> AgentLinkModel:
    results: dict[str, Any] = {}

    def send(action: str, **kw: Any) -> str:
        assert action == "control_clear_trail"
        results["r1"] = {"ok": True, "result": ctl.clear_trail_data()}
        return "r1"

    return AgentLinkModel(
        data_dir=tmp_path,
        send=send,
        state=lambda: {"results": results},
        alive=lambda: {"pid": 1},
        config_path=cfg_path,
    )


def test_trail_off_deletes_existing_files_and_thumbnails(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    ctl = _control(Clock(), tmp_path, settings=_live_settings(cfg_path))
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=1, y=1)
    ctl.act("a1", "move", x=2, y=2)
    folder = tmp_path / "actions"
    assert len(list(folder.glob("session-*.jsonl"))) == 1
    assert len(list(folder.glob("*.png"))) == 2

    m = _model_wired_to(ctl, tmp_path, cfg_path)
    removed = m.set_audit_trail(False)
    assert m.audit_trail_on() is False
    assert removed["sessions"] == 1
    assert removed["files"] == 1
    assert removed["thumbnails"] == 2
    assert list(folder.glob("session-*.jsonl")) == []
    assert list(folder.glob("*.png")) == []
    assert ctl.sessions == {}  # the in-memory trail is dropped too

    # ...and nothing new is written while the trail is off
    ctl.stop("done")
    ctl.request("a1", "Claude")
    ctl.act("a1", "move", x=3, y=3)
    assert list(folder.glob("session-*.jsonl")) == []
    assert list(folder.glob("*.png")) == []


def test_trail_off_without_a_running_host_still_deletes(tmp_path: Path) -> None:
    folder = tmp_path / "agentlink" / "actions"
    folder.mkdir(parents=True)
    (folder / "session-deadbeef.jsonl").write_text('{"event": "started"}\n', encoding="utf-8")
    (folder / "0123abcd.png").write_bytes(b"\x89PNG\r\n")
    m = AgentLinkModel(data_dir=tmp_path, config_path=tmp_path / "config.yaml", alive=lambda: None)
    removed = m.set_audit_trail(False)
    assert removed["files"] == 1
    assert removed["thumbnails"] == 1
    assert list(folder.iterdir()) == []


def test_trail_off_falls_back_to_local_purge_when_the_host_clear_fails(tmp_path: Path) -> None:
    # Codex P2 on #96: when the host was alive but its clear request failed, the
    # path returned zeros with no local purge, so files stayed on disk forever
    # even though the setting was already saved as off (startup with the trail
    # off only skips restoring, it never deletes).
    folder = tmp_path / "agentlink" / "actions"
    folder.mkdir(parents=True)
    (folder / "session-deadbeef.jsonl").write_text('{"event": "started"}\n', encoding="utf-8")
    (folder / "0123abcd.png").write_bytes(b"\x89PNG\r\n")
    results: dict[str, Any] = {"r1": {"ok": False, "error": "boom"}}

    def send(action: str, **kw: Any) -> str:
        assert action == "control_clear_trail"
        return "r1"

    m = AgentLinkModel(
        data_dir=tmp_path,
        send=send,
        state=lambda: {"results": results},
        alive=lambda: {"pid": 1},
        config_path=tmp_path / "config.yaml",
    )
    removed = m.set_audit_trail(False)
    assert m.audit_trail_on() is False
    assert removed["files"] == 1
    assert removed["thumbnails"] == 1
    assert list(folder.iterdir()) == []


def test_trail_on_removes_nothing(tmp_path: Path) -> None:
    folder = tmp_path / "agentlink" / "actions"
    folder.mkdir(parents=True)
    (folder / "session-deadbeef.jsonl").write_text('{"event": "started"}\n', encoding="utf-8")
    (folder / "0123abcd.png").write_bytes(b"\x89PNG\r\n")
    m = AgentLinkModel(data_dir=tmp_path, config_path=tmp_path / "config.yaml", alive=lambda: None)
    removed = m.set_audit_trail(True)
    assert removed == {"sessions": 0, "files": 0, "thumbnails": 0}
    assert (folder / "session-deadbeef.jsonl").exists()
    assert (folder / "0123abcd.png").exists()


# ---------------------------------------------------------------------- 3. pruning takes thumbnails


def test_pruning_a_session_deletes_its_thumbnails(tmp_path: Path) -> None:
    ctl = _control(Clock(), tmp_path)
    folder = tmp_path / "actions"
    thumbs: dict[str, str] = {}
    ids: list[str] = []
    for _ in range(MAX_SESSIONS + 3):
        ctl.request("a1", "Claude")
        ctl.act("a1", "move", x=1, y=1)
        assert ctl.grant is not None
        ids.append(ctl.grant.id)
        thumbs[ctl.grant.id] = ctl.log[-1].thumb
        ctl.stop("done")
    oldest, newest = ids[0], ids[-1]
    assert thumbs[oldest]  # the oldest session really had a thumbnail
    assert not (folder / f"session-{oldest}.jsonl").exists()  # pruned, as before
    assert not (folder / thumbs[oldest]).exists()  # its thumbnail goes with it now
    assert (folder / f"session-{newest}.jsonl").exists()
    assert (folder / thumbs[newest]).exists()  # the newest session keeps its own


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
