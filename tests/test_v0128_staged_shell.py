"""v0.1.28 staged shell (proposal #26): a `shell` tool in two safety tiers.

* Tier 1 (read-only allowlist: devices, cameras, processes, away_version, logs): fixed argv built here,
  never a shell, runs without per-call approval inside an approved computer session.
* Tier 2 (anything else): the person is asked first, every time, through the same prompt path as risky
  clicks; only "Allow" runs it.
* Hard rules: no shell metacharacters can escalate to tier 1 (exact-match classification), credential
  stores / password material are refused outright (even with approval), 64 KB output cap, secrets redacted,
  every invocation audited.

The service-level tests below FAIL on the unfixed code (no ``Service.run_shell`` / no ``shell`` module),
which is the regression proof required by the workflow (verified with ``git stash``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from away import config as C
from away.agentlink import api, groups
from away.agentlink import shell as sh
from away.agentlink.auth import Agent
from away.agentlink.control import ControlRefused
from away.agentlink.host import build_core
from away.agentlink.prompts import NullUI
from away.agentlink.service import Forbidden, NotAvailable


# ============================================================================ pure: classify / run / audit
def test_tier1_is_exact_match_only() -> None:
    ps = sh.tier1_argv("processes")
    assert ps is not None
    assert sh.classify(ps) == (1, "processes")
    # any deviation at all drops to tier 2: no smuggling arguments into a tier-1 command
    assert sh.classify([*ps, "--extra"]) == (2, None)
    assert sh.classify(ps[:-1]) == (2, None)
    assert sh.classify(["ps", "-eo", "pid,comm,args"]) == (2, None)
    assert sh.classify(["rm", "-rf", "/"]) == (2, None)
    assert sh.classify(["not-a-real-command"]) == (2, None)


def test_classify_needs_no_shell_metacharacters_for_tier1() -> None:
    # a tier-1-shaped argv with metacharacters is not on the allowlist -> tier 2 (asks the person)
    assert sh.classify(["ps", "-eo", "pid,comm;id"]) == (2, None)
    assert sh.classify(["echo", "$(whoami)"]) == (2, None)
    assert sh.has_metachars(["echo", "a|b"])
    assert not sh.has_metachars(["ps", "-eo", "pid,comm"])


@pytest.mark.parametrize(
    "argv",
    [
        ["cat", str(Path.home() / ".ssh" / "id_rsa")],
        ["type", "C:\\Users\\x\\.aws\\credentials"],
        ["powershell", "-Command", "Get-Credential | Export-Clixml out.xml"],
        ["mimikatz", "sekurlsa::logonpasswords"],
        ["security", "dump-keychain"],
        ["printenv", "MY_API_SECRET"],
    ],
)
def test_classify_refuses_credential_stores(argv: list[str]) -> None:
    with pytest.raises(sh.ShellRefused):
        sh.classify(argv)


def _proc(stdout: bytes = b"out", stderr: bytes = b"", rc: int = 0) -> Any:
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=rc)


def test_run_never_uses_a_shell() -> None:
    seen: list[tuple[Any, dict[str, Any]]] = []

    def fake(argv: Any, **kw: Any) -> Any:
        seen.append((argv, kw))
        return _proc()

    res = sh.run(["echo", "hi"], runner=fake)
    assert res["rc"] == 0 and res["stdout"] == "out" and not res["timed_out"]
    ((argv, kw),) = seen
    assert argv == ["echo", "hi"] and kw["shell"] is False


def test_run_rejects_bad_argv() -> None:
    with pytest.raises(ValueError):
        sh.run([])
    with pytest.raises(ValueError):
        sh.run(["echo", ""])


def test_run_output_cap_64k() -> None:
    big = b"x" * 200_000
    res = sh.run(["whatever"], runner=lambda argv, **kw: _proc(stdout=big, stderr=b"e" * 100))
    assert res["truncated"] is True
    assert len(res["stdout"].encode()) + len(res["stderr"].encode()) <= sh.OUTPUT_CAP


def test_run_missing_program_is_not_available() -> None:
    def fake(argv: Any, **kw: Any) -> Any:
        raise FileNotFoundError(2, "no such file")

    with pytest.raises(sh.ShellNotAvailable):
        sh.run(["definitely-not-a-program-xyz"], runner=fake)


def test_run_timeout_reports_and_keeps_partial_output() -> None:
    def fake(argv: Any, **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(argv, 30.0, output=b"partial")

    res = sh.run(["sleepy"], timeout_s=0.01, runner=fake)
    assert res["timed_out"] is True and res["rc"] is None and res["stdout"] == "partial"


def test_audit_appends_jsonl_and_tails_newest_first(tmp_path: Path) -> None:
    audit = sh.ShellAudit(tmp_path / "audit.jsonl")
    audit.append({"tier": 1, "command": "processes"})
    audit.append({"tier": 2, "command": "echo", "approved": True})
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and all(json.loads(ln)["t"] for ln in lines)
    tail = audit.tail(2)
    assert [e["command"] for e in tail] == ["echo", "processes"]


def test_run_bounds_output_retention() -> None:
    # a chatty child (5 MB) must not grow memory without bound: retention is capped, the child still
    # completes normally (it is not killed for being chatty)
    res = sh.run([sys.executable, "-c", "import sys; sys.stdout.write('x' * 5_000_000)"])
    assert res["rc"] == 0 and not res["timed_out"]
    assert res["truncated"] is True
    assert len(res["stdout"].encode()) + len(res["stderr"].encode()) <= sh.OUTPUT_CAP


def test_summarize_is_short() -> None:
    assert sh.summarize("line one\nline two", 0) == "rc=0 line one"
    assert len(sh.summarize("y" * 500, 1)) <= 170


# ============================================================================ service wiring
@pytest.fixture
def core(tmp_path: Path) -> Any:
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    return build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=NullUI())


def _agent_with(core: Any, *scopes: str) -> Agent:
    p = core.registry.new_pairing(["read", *scopes])
    assert core.registry.mark_pending(p, "Test agent")
    agent, _token = core.registry.approve(p, 3600.0)
    return agent


def _grant(core: Any, agent: Agent) -> None:
    core.control.ask = lambda *a, **k: {"answer": "Allow", "method": "test"}  # type: ignore[method-assign]
    core.control.request(agent.id, agent.name, "test")


@pytest.fixture
def granted(tmp_path: Path, core: Any) -> Any:
    agent = _agent_with(core, "control")
    _grant(core, agent)
    return types.SimpleNamespace(core=core, agent=agent)


def _no_subprocess(monkeypatch: pytest.MonkeyPatch, out: bytes = b"fake out") -> list[Any]:
    calls: list[Any] = []

    def fake(args: Any, **kw: Any) -> Any:
        calls.append(args)
        proc = types.SimpleNamespace(args=args, returncode=0, stdout=out, stderr=b"")
        return proc

    monkeypatch.setattr(sh, "_bounded_popen_run", fake)
    return calls


def test_shell_requires_control_permission(core: Any) -> None:
    agent = _agent_with(core)  # read-only agent
    with pytest.raises(Forbidden):
        core.service.run_shell(agent, action="run", command="processes")
    with pytest.raises(Forbidden):
        core.service.run_shell(agent, action="list")


def test_shell_requires_an_approved_session(core: Any) -> None:
    agent = _agent_with(core, "control")  # permission, but no grant: nobody approved this session
    with pytest.raises(ControlRefused):
        core.service.run_shell(agent, action="run", command="processes")


def test_shell_list_describes_tier1(granted: Any) -> None:
    res = granted.core.service.run_shell(granted.agent, action="list")
    names = [c["name"] for c in res["tier1"]]
    assert names == ["devices", "cameras", "processes", "away_version", "logs"]
    assert all(c["help"] for c in res["tier1"])


def test_tier1_runs_without_per_call_approval(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_subprocess(monkeypatch)
    asked: list[Any] = []

    def ask(*a: Any, **k: Any) -> dict[str, Any]:
        asked.append((a, k))
        return {"answer": "Deny"}

    granted.core.service.prompts.ask = ask  # type: ignore[method-assign]
    res = granted.core.service.run_shell(granted.agent, action="run", command="processes")
    assert res["tier"] == 1 and res["approved"] is False and res["rc"] == 0
    assert res["stdout"] == "fake out"
    assert asked == [], "tier 1 must not prompt"
    assert calls and calls[0][0] in ("ps", "tasklist")


def test_tier1_exact_argv_is_also_tier1(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_subprocess(monkeypatch)
    argv = sh.tier1_argv("processes")
    assert argv is not None
    res = granted.core.service.run_shell(granted.agent, action="run", argv=argv)
    assert res["tier"] == 1 and res["command"] == "processes"


def test_tier1_logs_reads_the_app_log(granted: Any, tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    logdir.mkdir()
    (logdir / "away.log").write_text("hello log line\ntoken=paway_0123456789abcdef\n", encoding="utf-8")
    res = granted.core.service.run_shell(granted.agent, action="run", command="logs")
    assert res["tier"] == 1 and "hello log line" in res["stdout"]
    assert "paway_0123456789abcdef" not in res["stdout"], "secrets must be redacted from output"


def test_tier2_asks_and_runs_on_allow(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_subprocess(monkeypatch, out=b"hi\n")
    asked: list[Any] = []

    def ask(*a: Any, **k: Any) -> dict[str, Any]:
        asked.append((a, k))
        return {"answer": "Allow", "method": "click"}

    granted.core.service.prompts.ask = ask  # type: ignore[method-assign]
    res = granted.core.service.run_shell(granted.agent, action="run", argv=["echo", "hi"], why="a test")
    assert res["tier"] == 2 and res["approved"] is True and res["stdout"] == "hi\n"
    assert len(asked) == 1 and "echo hi" in asked[0][0][0]
    assert calls == [["echo", "hi"]]


def test_tier2_denied_runs_nothing_and_is_audited(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_subprocess(monkeypatch)
    granted.core.service.prompts.ask = lambda *a, **k: {"answer": "Deny", "method": "click"}  # type: ignore[method-assign]
    with pytest.raises(ControlRefused):
        granted.core.service.run_shell(granted.agent, action="run", argv=["echo", "hi"])
    assert calls == [], "a denied command must never start"
    entries = granted.core.service.run_shell(granted.agent, action="log")["entries"]
    assert entries and entries[0]["tier"] == 2 and entries[0]["approved"] is False


def test_tier2_timeout_denied_runs_nothing(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_subprocess(monkeypatch)
    granted.core.service.prompts.ask = lambda *a, **k: {"answer": None, "method": "timeout"}  # type: ignore[method-assign]
    with pytest.raises(ControlRefused):
        granted.core.service.run_shell(granted.agent, action="run", argv=["echo", "hi"])
    assert calls == []


def test_tier2_credential_argv_refused_before_any_prompt(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_subprocess(monkeypatch)
    asked: list[Any] = []

    def ask(*a: Any, **k: Any) -> dict[str, Any]:
        asked.append((a, k))
        return {"answer": "Allow"}

    granted.core.service.prompts.ask = ask  # type: ignore[method-assign]
    with pytest.raises(sh.ShellRefused):
        granted.core.service.run_shell(granted.agent, action="run", argv=["cat", "~/.ssh/id_rsa"])
    assert asked == [] and calls == [], "credential touching is refused even when the user would allow it"


def test_tier2_grant_rechecked_after_approval(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # TOCTOU: the grant ends while the approval prompt waits -> the stale "Allow" must not run anything
    calls = _no_subprocess(monkeypatch)

    def ask(*a: Any, **k: Any) -> dict[str, Any]:
        granted.core.control.stop("the person pressed Stop")
        return {"answer": "Allow", "method": "click"}

    granted.core.service.prompts.ask = ask  # type: ignore[method-assign]
    with pytest.raises(ControlRefused):
        granted.core.service.run_shell(granted.agent, action="run", argv=["echo", "hi"])
    assert calls == []
    entries = granted.core.service.run_shell(granted.agent, action="log")["entries"]
    assert entries and entries[0]["approved"] is True and "session ended" in entries[0]["refused"]


def test_shellrefused_credential_argv_is_audited(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_subprocess(monkeypatch)
    granted.core.service.prompts.ask = lambda *a, **k: {"answer": "Allow"}  # type: ignore[method-assign]
    with pytest.raises(sh.ShellRefused):
        granted.core.service.run_shell(granted.agent, action="run", argv=["cat", "~/.ssh/id_rsa"])
    entries = granted.core.service.run_shell(granted.agent, action="log")["entries"]
    assert entries and entries[0]["command"] == "cat" and entries[0]["approved"] is False
    assert "credential" in entries[0]["refused"]


def test_shell_not_available_is_audited(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(argv: Any, **kw: Any) -> Any:
        raise FileNotFoundError(2, "no such file")

    monkeypatch.setattr(sh, "_bounded_popen_run", fake)
    with pytest.raises(NotAvailable, match="not installed"):
        granted.core.service.run_shell(granted.agent, action="run", command="devices")
    entries = granted.core.service.run_shell(granted.agent, action="log")["entries"]
    assert entries and entries[0]["command"] == "devices" and "not available" in entries[0]["refused"]


def test_shell_rejects_command_and_argv_together(granted: Any) -> None:
    with pytest.raises(ValueError):
        granted.core.service.run_shell(granted.agent, action="run", command="processes", argv=["ps"])


def test_shell_unknown_command_is_404(granted: Any) -> None:
    with pytest.raises(KeyError):
        granted.core.service.run_shell(granted.agent, action="run", command="nope")


def test_shell_log_lists_every_invocation(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_subprocess(monkeypatch)
    granted.core.service.run_shell(granted.agent, action="run", command="processes")
    entries = granted.core.service.run_shell(granted.agent, action="log")["entries"]
    assert entries and entries[0]["command"] == "processes" and entries[0]["tier"] == 1
    assert "summary" in entries[0] and entries[0]["agent"] == "Test agent"


def test_shell_output_secrets_redacted(granted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_subprocess(monkeypatch, out=b"key=sk-abcdefghijklmnopqrst\n")
    res = granted.core.service.run_shell(granted.agent, action="run", command="processes")
    assert "sk-abcdefghijklmnopqrst" not in res["stdout"]


# ============================================================================ group + REST wiring
def test_shell_group_resolves() -> None:
    method, args = groups.resolve("shell", {"action": "run", "command": "processes"})
    assert (method, args) == ("run_shell", {"command": "processes"})
    method, args = groups.resolve("shell", {"action": "log", "limit": 5})
    assert (method, args) == ("run_shell", {"action": "log", "limit": 5})
    method, args = groups.resolve("shell", {"action": "list"})
    assert (method, args) == ("run_shell", {"action": "list"})
    with pytest.raises(ValueError):
        groups.resolve("shell", {"action": "nuke"})


def test_api_knows_run_shell() -> None:
    assert "run_shell" in api.TOOLS
    tool, kwargs = api.resolve_tool("shell", {"action": "run", "command": "devices"})
    assert tool == "run_shell" and kwargs["command"] == "devices"


def test_shell_tool_appears_in_mcp_server() -> None:
    pytest.importorskip("mcp")
    import asyncio

    from away.agentlink import mcp_tools

    mcp = mcp_tools.build(types.SimpleNamespace())
    names = sorted(t.name for t in asyncio.run(mcp.list_tools()))
    assert "shell" in names and len(names) == 18
