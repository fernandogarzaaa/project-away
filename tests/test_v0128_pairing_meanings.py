"""Proposal #18: the Allow/Deny pairing prompt says what each permission means, in plain words.

The prompt details come from ``perms.describe(..., verbose=True)`` (label + help
text) instead of labels only, plus a session-limits line when the agent asks for
computer control.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from away import config as C
from away.agentlink import api, perms
from away.agentlink.host import build_core
from away.agentlink.perms import BY_SCOPE, PERMISSIONS
from away.agentlink.prompts import NullUI


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


class PromptSpy:
    """Clicks Allow on every prompt (like the person would) and records what was shown."""

    def __init__(self, prompts, ui: NullUI, answer: str = "Allow") -> None:
        self.answer = answer
        self.seen: list[tuple[str, list[str], list[str]]] = []
        self._stop = threading.Event()

        def run() -> None:
            while not self._stop.is_set():
                cur = prompts.current
                if cur is not None and cur.id not in ui.clicks:
                    self.seen.append((cur.question, list(cur.options), list(cur.details)))
                    ui.clicks[cur.id] = self.answer if self.answer in cur.options else cur.options[-1]
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


@pytest.fixture
def live(tmp_path: Path):
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    ui = NullUI()
    core = build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=ui)
    core.prompts.polite = lambda: True
    core.prompts.start()
    gate = api.Gate(
        core.service,
        core.registry,
        local_key="k" * 43,
        rate=(50.0, 60),
        access_log=tmp_path / "logs" / "agent-access.log",
    )
    spy = PromptSpy(core.prompts, ui)
    yield {"core": core, "gate": gate, "spy": spy}
    spy.stop()
    core.prompts.stop()


def pair_and_capture_details(live: dict, scopes: list[str]) -> list[str]:
    core, gate, spy = live["core"], live["gate"], live["spy"]
    p = core.registry.new_pairing(scopes)
    agent, _token = gate.pair(p, "Meaning Test", {"name": "Meaning Test"}, False)
    assert agent is not None
    assert spy.seen, "the pairing prompt was never shown"
    question, _options, details = spy.seen[-1]
    assert question == "Meaning Test wants to connect"
    return details


# ------------------------------------------------------------- perms.describe(verbose=True)


def test_describe_labels_only_by_default() -> None:
    assert perms.describe(["read", "control"]) == ["See my gestures", "Let it use my computer"]


def test_describe_verbose_adds_each_permissions_plain_language_meaning() -> None:
    scopes = ["read", "snapshots", "screen", "tune", "control", "diagnose", "share_logs"]
    lines = perms.describe(scopes, verbose=True)
    have = set(scopes)
    expected = [f"{p.label}: {p.help}" if p.help else p.label for p in PERMISSIONS if p.scope in have]
    assert lines == expected
    for p in PERMISSIONS:
        if p.scope in have:
            assert p.help, f"{p.scope} has no plain-language meaning"
            assert any(p.label in line and p.help in line for line in lines), p.scope
    assert "scope" not in " ".join(lines).lower()  # the UI never says "scope"


# ------------------------------------------------------------- the pairing prompt


def test_pairing_prompt_shows_what_each_permission_means(live: dict) -> None:
    details = pair_and_capture_details(live, ["read", "control", "screen"])
    for scope in ("read", "control", "screen"):
        p = BY_SCOPE[scope]
        assert any(p.label in d and p.help in d for d in details), (scope, details)
    assert any(
        d == "Computer use: you approve it each time, up to 30 minutes, and you can take over by moving the mouse."
        for d in details
    ), details
    assert any(d.startswith("Matching code:") for d in details)


def test_pairing_prompt_omits_the_computer_use_line_without_control(live: dict) -> None:
    details = pair_and_capture_details(live, ["read", "screen"])
    assert not any(d.startswith("Computer use:") for d in details), details
    p = BY_SCOPE["screen"]
    assert any(p.label in d and p.help in d for d in details), details
    assert any(d.startswith("Matching code:") for d in details)
