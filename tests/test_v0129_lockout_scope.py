"""v0.1.29 (audit fix): the failed sign-in lockout is per source and never refuses valid credentials.

Before the fix, ``FailureLockout`` kept one global failure list: ten bogus pair attempts from any
peer locked every non-persistent caller (even the public ping) for a minute, and the lock could be
re-armed at will. Now failures are counted per source (the identity the middleware computes),
a successful sign-in clears that source's failures and any active lock, and the lock only ever
refuses attempts that carry no credentials. Public paths carry no credentials, so they stay
refused for a locked source, but only for that source.
"""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from away import config as C
from away.agentlink import api
from away.agentlink.host import build_core
from away.agentlink.prompts import NullUI
from away.agentlink.ratelimit import FailureLockout

A = {"cf-connecting-ip": "198.51.100.1"}  # one source, as the middleware computes it
B = {"cf-connecting-ip": "198.51.100.2"}  # another source


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def make_client(tmp_path: Path):
    cfgp = tmp_path / "config.yaml"
    C.save(C.AppConfig(), cfgp)
    cfg = C.load(cfgp)[0]
    core = build_core(tmp_path, cfg=lambda: cfg, settings=lambda: cfg.agent_link, config_path=lambda: cfgp, ui=NullUI())
    gate = api.Gate(core.service, core.registry, local_key="k" * 43, access_log=tmp_path / "access.log")
    app = api.create_app(core.service, gate, None)
    return core, gate, TestClient(app, base_url="http://127.0.0.1:8765", raise_server_exceptions=False)


def bad_pair(client: TestClient, headers: dict[str, str]) -> int:
    return client.post("/v1/pair", json={"code": "zzzzzzzzzz", "pin": "000000"}, headers=headers).status_code


def status(client: TestClient, headers: dict[str, str], token: str | None = None) -> int:
    h = dict(headers)
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    return client.get("/v1/status", headers=h).status_code


# ------------------------------------------------------------------------------------- unit level


def test_lockout_counts_failures_per_source() -> None:
    clock = Clock()
    lock = FailureLockout(3, 60, 30, clock)
    for _ in range(3):
        lock.fail("a")
    assert lock.locked("a")
    assert not lock.locked("b")  # one source's failures never lock another source
    assert not lock.locked("")
    for _ in range(3):
        lock.fail("b")
    assert lock.locked("b")
    clock.advance(31)
    assert not lock.locked("a") and not lock.locked("b")


def test_lockout_success_clears_the_sources_failures_and_lock() -> None:
    clock = Clock()
    lock = FailureLockout(3, 60, 30, clock)
    lock.fail("a")
    lock.fail("a")
    lock.ok("a")  # a successful sign-in forgives the earlier failures
    lock.fail("a")
    lock.fail("a")
    assert not lock.locked("a")
    lock.fail("a")
    assert lock.locked("a")
    lock.ok("a")  # ...and clears an active lock for that source
    assert not lock.locked("a")


# ------------------------------------------------------------------------------------- HTTP level


def test_bad_attempts_from_one_source_do_not_lock_out_a_valid_token(tmp_path: Path) -> None:
    core, _gate, client = make_client(tmp_path)
    _agent, tok = core.registry.issue("victim", ["read"], 3600.0)
    assert status(client, A, tok) == 200
    assert [bad_pair(client, A) for _ in range(10)] == [404] * 10
    # the source is locked now, but a valid token from that same source still passes
    assert status(client, A, tok) == 200
    # and no other source was ever affected
    assert status(client, B, tok) == 200
    assert status(client, {}, tok) == 200


def test_a_locked_source_stays_throttled_without_credentials(tmp_path: Path) -> None:
    _core, _gate, client = make_client(tmp_path)
    assert [bad_pair(client, A) for _ in range(10)] == [404] * 10
    # brute-force protection is intact: the locked source's unauthenticated attempts are refused
    assert bad_pair(client, A) == 429
    assert status(client, A, "not-a-real-token") == 429
    assert status(client, A) == 429
    # public paths carry no credentials, so they stay refused for the locked source only
    assert client.get("/v1/ping", headers=A).status_code == 429
    # another source is not throttled at all: its bad attempts get the ordinary answers
    assert bad_pair(client, B) == 404
    assert status(client, B, "not-a-real-token") == 401
    assert client.get("/v1/ping", headers=B).status_code == 200


def test_successful_auth_clears_the_sources_failure_count(tmp_path: Path) -> None:
    core, _gate, client = make_client(tmp_path)
    _agent, tok = core.registry.issue("victim", ["read"], 3600.0)
    assert [bad_pair(client, A) for _ in range(9)] == [404] * 9  # one short of the lock
    assert status(client, A, tok) == 200  # the success clears the count
    # nine more failures are answered normally again: the count restarted at zero
    assert [bad_pair(client, A) for _ in range(9)] == [404] * 9
    assert bad_pair(client, A) == 404  # the tenth engages the lock but still answers normally
    assert bad_pair(client, A) == 429  # and only now is the source throttled


def test_refresh_with_a_valid_token_works_while_the_source_is_locked(tmp_path: Path) -> None:
    core, _gate, client = make_client(tmp_path)
    agent, _tok = core.registry.issue("victim", ["read"], 3600.0)
    refresh = core.registry.new_refresh(agent.id)
    assert [bad_pair(client, A) for _ in range(10)] == [404] * 10
    r = client.post("/v1/token/refresh", json={"refresh_token": refresh}, headers=A)
    assert r.status_code == 200, r.text  # a valid credential is judged on its merits, lock or not
    # the success cleared the lock for this source, so even public paths answer again
    assert client.get("/v1/ping", headers=A).status_code == 200
    fresh = r.json()["token"]  # the refresh rotated the access token; the old one is dead by design
    assert status(client, A, fresh) == 200


def test_persistent_token_still_works_while_its_source_is_locked(tmp_path: Path) -> None:
    core, _gate, client = make_client(tmp_path)
    keeper, _t = core.registry.issue("keeper", ["read"], 3600.0)
    _rec, secret = core.registry.issue_persistent(keeper.id)
    assert [bad_pair(client, A) for _ in range(10)] == [404] * 10
    assert status(client, A, secret) == 200  # the long-standing exemption is preserved
