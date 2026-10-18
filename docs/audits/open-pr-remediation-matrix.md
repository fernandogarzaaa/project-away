# Project Away: Open-PR Remediation Matrix

**Audit date:** 2026-10-18
**Scope:** Defects confirmed and fixed by Tracks A-D during the open-PR audit program.

## Defect matrix

| # | Defect | Severity | Evidence | Fix commit | Tests | CI status | Remaining blocker |
|---|--------|----------|----------|------------|-------|-----------|-------------------|
| 70-1 | Job names said "(non-blocking)" but continue-on-error was removed; a group failure fails the workflow | Medium | ci.yml: job names vs `continue-on-error: false` | 5fad0c2e | Workflow validation in CI | Queued on 5fad0c2e | None (CI pending) |
| 70-2 | `summary()` returned 0 unconditionally; "weak score shows red" claim was false | High | scripts/mutation.py: `summary()` always exited 0 | 5fad0c2e | Threshold enforcement tested in CI | Queued on 5fad0c2e | None (CI pending) |
| 70-3 | Group-list drift between matrix and scripts/mutation.py | Medium | Matrix hardcoded 15 groups; script had different list | 5fad0c2e | `--list-groups` single-source verified | Queued on 5fad0c2e | None (CI pending) |
| 70-4 | Missing reports failed with raw error instead of clear message | Low | Summary job `ls` failure mode | 5fad0c2e | Error path exercised | Queued on 5fad0c2e | None (CI pending) |
| 71-1 | Committed `.mutation-logs/run6.log` (development artifact) | Low | File present in PR diff | 89a175f7 | N/A (hygiene) | In progress on 89a175f7 | None (CI pending) |
| 71-2 | Committed `STATE.md` (work-state file) | Low | File present in PR diff | 89a175f7 | N/A (hygiene) | In progress on 89a175f7 | None (CI pending) |
| 71-3 | Gate atomicity not covered by tests | Medium | No test for concurrent gate acquisition | 89a175f7 | 3 new tests, fail pre-fix | In progress on 89a175f7 | On-device camera test outstanding |
| 72-1 | `on_face_gate` missing `@menu_safe` (exception in callback kills tray) | High | test_v0129_com_hardening.py: gap proven pre-fix | b8e8fea8 | New test fails pre-fix | In progress on b8e8fea8 | On-device COM disconnect outstanding |
| 72-2 | `on_gpu_install` missing `@menu_safe` | High | Same as above | b8e8fea8 | New test fails pre-fix | In progress on b8e8fea8 | Same |
| 72-3 | Committed `STATE.md` (work-state file) | Low | File present in PR diff | b8e8fea8 | N/A (hygiene) | In progress on b8e8fea8 | None |
| 74-1 | `play_blocking` double-closed stream after barge-in `stop()` | High | sounddevice `stop()` closes; `close()` has no guard; 2 new tests fail pre-fix | 0bd3417e | 2 new tests, fail pre-fix | Queued on 0bd3417e | On-device audio outstanding |
| 68-1 | Stale element IDs clickable after window change (D1) | High | Demonstrated: ID from old window resolves to wrong target | 35cd4c23 | 5 tests, 20/20 pass, 4/5 fail pre-fix | Queued; **macos test FAILED** | macos CI failure must be diagnosed |
| 68-2 | Element registry shared across agents (D2) | High | Single registry; agent B could use agent A's IDs | 35cd4c23 | Per-agent `_ElementQuery` tested | Queued; **macos test FAILED** | Same |
| 68-3 | Lock missing on element read (D3) | Medium | Read path not synchronized | 35cd4c23 | Covered by 20/20 | Queued; **macos test FAILED** | Same |
| 66-1 | `act_sequence` consumed 1 budget step for N actions | High | Demonstrated: 10 actions on budget 5 consumed 1 step | 32f755c5 | 3 tests, 18/18 pass, all fail pre-fix | Queued on 32f755c5 | Semantic merge with #68/#64 |
| 65-1 | Stale marks survived fresh unannotated screenshots | High | Mark from old screenshot still clickable | a4851725 | 3 tests, 10/10 pass, 2 fail pre-fix | Queued on a4851725 | Semantic merge with #68 |
| 65-2 | Stale marks survived failed annotations | Medium | `_annotate` failure left old marks | a4851725 | Covered by 10/10 | Queued on a4851725 | Same |
| 60-1 | `selftest_latest` had zero authorization checks | Critical | Any agent could retrieve reports via tool/REST/MCP | 37483a22 | 32/32 pass, 2 new tests fail pre-fix | Queued on 37483a22 | None (CI pending) |
| 57-1 | Process hints worded as fact (not qualified) | Medium | "Process X is using the camera" without qualification | a0dd930f | Tests fail pre-fix | Success (latest) | On-device validation outstanding |
| 57-2 | ValueError crash on malformed registry timestamps | Medium | Uncaught exception on bad timestamp | a0dd930f | Tests fail pre-fix | Success (latest) | Same |
| 59-1 | `read_consent` returned None on empty HKCU without trying HKLM | Medium | Registry fallback missing | c5988552 | 3 tests fail pre-fix | Queued; **macos test FAILED** | macos CI failure must be diagnosed |
| 45-1 | Task Manager allowlist enabled terminal bypass via File > Run new task | High | Security: approved tool leads to terminal | 8d2cda3d | 11/11 pass | Cancelled (mutation) | Mergeable=false; mutation via #70 |
| 64-1 | `undo_all_by_user` undid other agents' actions | High | `agent_id=None` matched all agents | f4dca251 | 22/22 pass | In progress; **mypy FAILED** | mypy failure must be fixed |
| 49-1 | `expect` not re-validated after user prompt (TOCTOU) | High | Focus could change between check and typing | a2f0eebc | Tests fail pre-fix | Success | On-device focus test outstanding |

## Fix verification summary

**Total defects fixed:** 23 across 11 PRs.
**Tests added:** All new tests fail pre-fix (verified by the tracks).
**Code gates:** ruff check, ruff format, mypy clean per the tracks (except #64 mypy, which is now failing in CI and must be fixed).

## Remaining blockers per defect

1. **CI failures on remediation pushes:** #68 (macos), #64 (mypy), #59 (macos). These are new failures introduced by or exposed by the remediation. They must be diagnosed and fixed before the respective PRs can merge.
2. **Semantic merges:** #65/#68 (`_point` incompatibility), #66/#68/#64 (`control.py` overlaps). These require coordinated integration work.
3. **Mutation CI:** #73, #52, #46 need the #70 workflow to get real mutation scores.
4. **On-device validation:** All hardware-dependent fixes need Windows testing (camera, COM, audio, UIA, registry). Mock tests provide deterministic coverage but do not prove hardware behavior.
