# Project Away: Open-PR Audit Review

**Audit date:** 2026-10-18
**Auditor:** Phase 7/8 validation agent (program: open-PR audit, remediation, release readiness)
**Scope:** All 23 open PRs on fernandogarzaaa/project-away as of 2026-10-18
**Base:** origin/main at 347d51f8 (Release v0.1.28)
**Method:** Head SHAs and CI states refreshed via GitHub API on 2026-10-18. Remediation was performed by Tracks A-D (see track-reports.md); this document validates and records dispositions. CodeRabbit quota is exhausted repo-wide; no bot reviews were available during this audit.

## Disposition summary

| PR | Title | Head | Mergeable | CI (head) | Disposition |
|----|-------|------|-----------|-----------|-------------|
| #70 | Mutation-testing CI matrix | 5fad0c2e | Yes | Queued | Ready for human review (CI pending) |
| #71 | Serialize camera opens | 89a175f7 | Yes | In progress | Ready for human review (CI pending) |
| #74 | Serialize audio output calls | 0bd3417e | Yes | Queued | Ready for human review (CI pending) |
| #73 | Combine agent settings tabs | 655003b9 | Yes | Cancelled (mutation) | Needs more work (mutation CI) |
| #72 | COM hardening | b8e8fea8 | Yes | In progress | Ready for human review (CI pending) |
| #68 | UI Automation grounding | 35cd4c23 | No | Queued; macos test FAILED | Needs more work (CI failure + conflicts) |
| #64 | Consent receipts and Undo all | f4dca251 | No | In progress; lint FAILED (mypy) | Needs more work (CI failure + conflicts) |
| #57 | Camera-holder diagnosis | a0dd930f | Yes | Success (latest) | Ready for human review |
| #65 | Set-of-mark annotations | a4851725 | No | Queued | Blocked by dependency (conflicts with #68) |
| #56 | Click verification via UIA | 3c1a3081 | Yes | Success | Ready for human review |
| #46 | Camera driver-state diagnosis | 6ca02f8c | No | Cancelled | Needs more work (conflicts) |
| #52 | Computer-access end notifications | 0a6e56bf | Yes | Cancelled (mutation) | Needs more work (mutation CI) |
| #66 | Step budget and visual audit trail | 32f755c5 | No | Queued | Blocked by dependency (conflicts with #68/#64) |
| #59 | Camera privacy and shutter diagnosis | c5988552 | No | Queued; macos test FAILED | Needs more work (CI failure + conflicts) |
| #45 | Allowlisted system-tool opening | 8d2cda3d | No | Cancelled | Needs more work (conflicts; base now main) |
| #63 | First-run agent-connection wizard | 11fbfecc | Yes | Success | Ready for human review |
| #61 | Tray privacy/activity popup | cc02f136 | Yes | Success | Ready for human review |
| #53 | Friendlier resume wording | df4fbd51 | Yes | Success | Ready for human review |
| #60 | Unify self-test report audiences | 37483a22 | Yes | Queued | Ready for human review (CI pending) |
| #55 | Driver-broken camera fix buttons | e5cb7ed7 | Yes | Success | Ready for human review |
| #54 | Pairing permission explanations | 7e8250a7 | Yes | Success | Ready for human review |
| #50 | Verify application opening | 29de2693 | Yes | Success | Ready for human review |
| #49 | Focus-verified typing | a2f0eebc | Yes | Success | Ready for human review |

**Counts:** Ready for human review: 13 (4 with CI pending). Needs more work: 7. Blocked by dependency: 2. Unsafe to merge: 0.

---

## PR #70: CI: run mutation groups as a parallel matrix

**Link:** https://github.com/fernandogarzaaa/project-away/pull/70
**Head:** 5fad0c2e (refreshed 2026-10-18; was ed9bc928 in inventory)
**Base:** main (347d51f8)
**Files:** .github/workflows/ci.yml, scripts/mutation.py

**Inspected:** The workflow defines a 15-group parallel mutation matrix with per-group artifacts and a summary job. scripts/mutation.py supports --list-groups and --summary with threshold enforcement.

**Confirmed findings (Track A remediation):**
1. Job names said "(non-blocking)" but continue-on-error was removed: names corrected in 5fad0c2e.
2. summary() returned 0 unconditionally: now enforces the threshold (fail-under 80).
3. Group-list drift between matrix and scripts/mutation.py: now single-sourced via --list-groups.
4. Missing reports failed with raw error: now a clear error message.

**Risks investigated:** The gh-CLI download loop in the summary job depends on `gh` availability on the runner; verified present via the passing run on the prior head.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Run 39094288943 queued on head 5fad0c2e at refresh time. Prior head ed9bc928: 26/26 success (verified 2026-10-18).

**Fix commit:** 5fad0c2e (pushed by Track A).

**Tests:** CI workflow validation; scripts/mutation.py --list-groups and --summary exercised in CI.

**Remaining verification:** None platform-specific; workflow runs on GitHub-hosted runners.

**Disposition:** Ready for human review (CI pending on the new head; prior head was fully green).

---

## PR #71: Fix: serialize camera opens with a process-wide gate

**Link:** https://github.com/fernandogarzaaa/project-away/pull/71
**Head:** 89a175f7 (refreshed; was 33caeebf)
**Base:** main (347d51f8)
**Files:** src/away/multicam/capture.py, tests/test_v0129_camera_open_gate.py (STATE.md and .mutation-logs/run6.log removed by Track A)

**Inspected:** The `_open_gate` module-level lock in capture.py, `open_camera()` gate acquisition with timeout and generation hand-over, and 16 regression tests.

**Confirmed findings (Track A):** Hygiene only. Removed committed `.mutation-logs/run6.log` and `STATE.md` (development artifacts that do not belong in source control). Added 3 gate-atomicity regression tests. Gate logic traced sound: the lock serializes `_open_camera`, `wait_devices_free()` acquires the gate after setting `_abort_opens`, and the timeout loop re-checks the hand-over generation.

**Risks investigated but not reproduced:** Post-read cancellation (a native open blocked in a read while hand-over completes) was attempted and reverted (mutation score cratered to 63.4%). Residual risk requires `measure_s=0` AND hand-over finishing during a blocking read; documented in the PR. Native Windows camera behavior cannot be verified in CI or this sandbox.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Run 39094329271 in progress on head 89a175f7 at refresh time. Prior head 33caeebf: 10/10 real jobs success; mutation job in progress (old workflow).

**Fix commit:** 89a175f7 (Track A hygiene + tests).

**Tests:** 16 tests in test_v0129_camera_open_gate.py pass locally; 3 new gate-atomicity tests fail pre-fix. Mutation gate 95.8% on the feature (reported by the implementation crew).

**Remaining verification:** On-device Windows camera test (concurrent opens, cancellation during native open). Marked outstanding.

**Disposition:** Ready for human review (CI pending on the new head).

---

## PR #74: Fix B: serialize sounddevice output-stream calls through one lock

**Link:** https://github.com/fernandogarzaaa/project-away/pull/74
**Head:** 0bd3417e (refreshed; was 1fcde643)
**Base:** main (347d51f8)
**Files:** src/away/voice/audio_out.py, src/away/voice/earcon.py, src/away/voice/natural.py, src/away/liveagent/speech.py, tests/test_v0129_audio_out_serial.py

**Inspected:** The `_sd_lock` in audio_out.py, `play()`, `play_blocking()` (waits outside the lock), `stop()`, and the double-close guard for streams closed by `stop()`.

**Confirmed findings (Track A):** Real defect fixed. `play_blocking` double-closed the stream after barge-in `stop()`: sounddevice `stop()` closes the stream, and `close()` has no double-close guard. The fix tracks stop-closed streams; all teardown paths now close exactly once. Added 2 regression tests.

**Risks investigated:** Concurrent play/stop/interrupt stress; stop() remains prompt because `play_blocking` waits outside the lock. sounddevice remains optional and lazily imported.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Run 39094327691 queued on head 0bd3417e at refresh time. Prior head 1fcde643: 10/11 green; mutation job cancelled (old workflow).

**Fix commit:** 0bd3417e (Track A).

**Tests:** 13 tests pass; 2 new double-close tests fail pre-fix. Mutation gate 92.3% (reported by implementation crew).

**Remaining verification:** On-device audio (concurrent playback, barge-in interruption, earcons during speech). Marked outstanding.

**Disposition:** Ready for human review (CI pending on the new head).

---

## PR #73: Merge Live agent and Agent Link tabs into one Agents tab

**Link:** https://github.com/fernandogarzaaa/project-away/pull/73
**Head:** 655003b9 (unchanged)
**Base:** main (347d51f8)
**Files:** 9 files including src/away/app/agents_tab.py (new), settings_gui.py, agentlink/gui.py, liveagent/gui.py, tests/test_v0129_agents_tab.py

**Inspected:** The Agents tab composition, TAB_ORDER merge, `resolve_tab_alias()` for backward compatibility, and panel refactoring.

**Confirmed findings:** No defects found by the audit tracks. The implementation composes both panels without modifying models.

**Risks investigated but not reproduced:** Widget lifecycle and background refresh duplication; both panels retain existing functionality per the test suite.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Completed, cancelled. The only non-green job is the old single mutation job (40-min timeout). This PR was not remediated by the tracks; it needs the #70 workflow to get a real mutation score.

**Fix commit:** None (no changes made; nothing defective found).

**Tests:** 106 focused tests pass (agents_tab + release integration + liveagent app + check-my-pc).

**Remaining verification:** On-device settings UI (widget lifecycle, scroll behavior). Marked outstanding.

**Disposition:** Needs more work (mutation CI must complete via the #70 workflow before merge).

---

## PR #72: Fix: harden COM usage (UIA lock, re-create on disconnect, guarded audio sessions)

**Link:** https://github.com/fernandogarzaaa/project-away/pull/72
**Head:** b8e8fea8 (refreshed; was 12d8ac1a)
**Base:** main (347d51f8)
**Files:** src/away/app/tray.py, src/away/pointer/uia.py, src/away/voice/other_audio.py, tests/test_v0129_com_hardening.py (STATE.md removed by Track A)

**Inspected:** The UIA lock in `foreground_targets()`/`focus()`, COM object re-creation on disconnect, `other_audio.py` guarded `GetAllSessions()`, the `menu_safe` decorator on tray callbacks, and the extracted `TrayMenuCallbacks` class.

**Confirmed findings (Track A):** Real defect. `on_face_gate` and `on_gpu_install` were missing `@menu_safe` (test proves the gap pre-fix). Hygiene: removed STATE.md.

**Risks investigated but not reproduced:** Lock ownership across COM apartments; disconnect during `foreground_targets` iteration. The fix holds the lock for the whole method body and resets `_uia`/`_cond`/`_cache` on COMError/OSError. True native SEH faults cannot be caught from Python (documented non-goal).

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Run 39094334656 in progress on head b8e8fea8 at refresh time.

**Fix commit:** b8e8fea8 (Track A).

**Tests:** 75 tests pass; the menu_safe gap test fails pre-fix. Mutation gate 85.6% (reported by implementation crew).

**Remaining verification:** On-device Windows COM disconnect (real UIA provider failure). Marked outstanding.

**Disposition:** Ready for human review (CI pending on the new head).

---

## PR #68: Agent Link: UI Automation element query and id-based actions

**Link:** https://github.com/fernandogarzaaa/project-away/pull/68
**Head:** 35cd4c23 (refreshed; was da91216e)
**Base:** 3b4063e4 (28 commits behind main)
**Files:** 47 files

**Inspected:** `_ElementQuery` (per-agent query with foreground fingerprint), `_revalidate_element`, `_resolve_element_id`, and the 5 regression tests.

**Confirmed findings (Track B):**
1. D1: Stale element IDs could be clicked after window change (demonstrated). Fixed with foreground fingerprint + `_revalidate_element`.
2. D2: Element registry shared across agents. Fixed with per-agent `_ElementQuery`.
3. D3: Lock missing on read. Fixed.
4. 5 tests, 20/20 pass, 4/5 fail pre-fix. testteeth 39.8% (survivors are edge branches).

**Risks investigated:** Disconnected COM objects during revalidation; the fix handles them via the uia.py hardening (see #72).

**Conflicts/dependencies:** Mergeable: false. Base is 28 commits behind main. **INTEGRATION FINDING (Phase 7):** #68's `_point` signature `(agent_id, x, y, element, element_id, from_image)` is incompatible with #65's `_point` `(agent_id, x, y, element, from_image)`. #68 handles `element_id` but ignores mark references ("mark:14"); #65 handles marks but lacks `element_id`. A semantic merge is required before both can land. Additionally #68's screenshot pipeline (`shot.render`) differs from #65's (`shot.to_image`/`_annotate`/`shot.to_jpeg`).

**CI:** Run 39095125979 queued on head 35cd4c23. **FAILING CHECK on this head:** `test (macos-latest, py3.12)` failed in the "Core tests" step. This is a real failure introduced by or exposed by the remediation push; logs were not retrievable (API 401).

**Fix commit:** 35cd4c23 (Track B).

**Tests:** 20/20 pass locally (test_v0128_uia_grounding.py).

**Remaining verification:** On-device Windows UIA (real element discovery, foreground changes). Marked outstanding.

**Disposition:** Needs more work (macos CI failure must be diagnosed; merge conflicts with main and semantic conflict with #65 must be resolved).

---

## PR #64: Agent Link: consent receipts

**Link:** https://github.com/fernandogarzaaa/project-away/pull/64
**Head:** f4dca251 (refreshed; was 3ea444f1)
**Base:** 3b4063e4 (28 commits behind main)
**Files:** 49 files

**Inspected:** `undo_all_by_user(item_ids)`, `_undo(agent_id, only_ids)`, receipt generation, and the 22 regression tests.

**Confirmed findings (Track D):** Real defect. `undo_all_by_user` undid other agents' actions (`agent_id=None` matched all). Now scoped to receipt item IDs. 22/22 tests pass.

**Risks investigated:** Partial undo results are now explicit; the audit trail reflects them.

**Conflicts/dependencies:** Mergeable: false. Base is 28 commits behind main. Overlaps with #66 (both refactor `act_sequence`, `_policy_check`, `_audit`, `require_grant`); a semantic merge is needed if both land.

**CI:** Run in progress on head f4dca251. **FAILING CHECK on this head:** `lint` failed in the "Run mypy" step. This is a real failure on the remediation push.

**Fix commit:** f4dca251 (Track D).

**Tests:** 22/22 pass locally.

**Remaining verification:** None platform-specific for the undo logic.

**Disposition:** Needs more work (mypy failure must be fixed; merge conflicts must be resolved).

---

## PR #57: Diagnose: who is holding my camera

**Link:** https://github.com/fernandogarzaaa/project-away/pull/57
**Head:** a0dd930f (refreshed; was 8771dd6a)
**Base:** main (347d51f8)
**Files:** 5 files (camera_users.py, groups.py, jobs.py, mcp_tools.py, tests)

**Inspected:** Process-hint wording, registry timestamp parsing, and the camera-holder diagnosis.

**Confirmed findings (Track C):** 2 fixes. (1) Process hints were worded as fact; now qualified. (2) ValueError crash on malformed registry timestamps; now handled. Tests fail pre-fix.

**Risks investigated but not reproduced:** Distinguishing current usage from historical last-use; name matching across processes.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Two runs on this head: one failure, one success. The latest run is success. The failure included `test (windows-latest, py3.12)` and the cancelled mutation job; the success run is the current state.

**Fix commit:** a0dd930f (Track C).

**Tests:** Pass per the success run.

**Remaining verification:** On-device Windows (real camera holder detection via registry/CIM). Marked outstanding.

**Disposition:** Ready for human review.

---

## PR #65: Agent Link: set-of-mark annotation for empty UIA trees

**Link:** https://github.com/fernandogarzaaa/project-away/pull/65
**Head:** a4851725 (refreshed; was eb07de4d)
**Base:** 3b4063e4 (28 commits behind main)
**Files:** 49 files

**Inspected:** `setofmark.py`, the marks cache (`control.marks`), `_point` mark resolution, and the 3 regression tests.

**Confirmed findings (Track B):** Real defect. Stale marks survived fresh unannotated screenshots and failed annotations. Fix: clear on deliberate unannotated `get_screen` (quiet live-view exempt) and on `_annotate` failure. 3 tests, 10/10 pass, 2 fail pre-fix.

**Risks investigated:** Multi-agent isolation (marks are per-agent); revoke/reconnect clearing.

**Conflicts/dependencies:** Mergeable: false. **BLOCKED by #68:** the `_point` signatures are incompatible (see #68). #65 must be semantically merged with #68 before either lands.

**CI:** Run 39095133143 queued on head a4851725.

**Fix commit:** a4851725 (Track B).

**Tests:** 10/10 pass locally.

**Remaining verification:** On-device (annotated screenshots, real UIA trees). Marked outstanding.

**Disposition:** Blocked by dependency (semantic merge with #68 required).

---

## PR #56: Agent Link: click verification via UIA

**Link:** https://github.com/fernandogarzaaa/project-away/pull/56
**Head:** 3c1a3081 (unchanged)
**Base:** main (347d51f8)
**Files:** 3 files (control.py, mcp_tools.py, tests)

**Inspected:** Click verification logic and the `target_label` semantics.

**Confirmed findings:** No defects found by the audit tracks. The audit verified `target_label` represents the pre-click requested target with post-click verification evidence (not merely "click was issued").

**Risks investigated but not reproduced:** False "verified" labels when the click misses; the implementation checks post-click state.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Success (latest run on head 3c1a3081).

**Fix commit:** None (no changes made).

**Tests:** Pass per CI.

**Remaining verification:** On-device Windows (real click verification via UIA). Marked outstanding.

**Disposition:** Ready for human review.

---

## PR #46: Diagnose: driver-state check for cameras that enumerate but won't open

**Link:** https://github.com/fernandogarzaaa/project-away/pull/46
**Head:** 6ca02f8c (unchanged)
**Base:** 3b4063e4 (28 commits behind main)
**Files:** 47 files

**Inspected:** PowerShell/CIM calls, JSON parsing, device matching, driver codes, fix recommendations.

**Confirmed findings (Track C):** No defects. The diagnosis gives accurate, not overconfident, fix recommendations.

**Risks investigated but not reproduced:** Unknown or contradictory driver data; the implementation does not trigger unsafe recommendations.

**Conflicts/dependencies:** Mergeable: false. Base is 28 commits behind main.

**CI:** Cancelled (old mutation job; no remediation push).

**Fix commit:** None (no changes made).

**Tests:** Pass per earlier CI (non-mutation jobs).

**Remaining verification:** On-device Windows (real driver-state via CIM). Marked outstanding.

**Disposition:** Needs more work (merge conflicts must be resolved; 28 commits behind).

---

## PR #52: Agent Link: toast when computer access ends

**Link:** https://github.com/fernandogarzaaa/project-away/pull/52
**Head:** 0a6e56bf (unchanged)
**Base:** 3b4063e4 (28 commits behind main)
**Files:** 3 files (control.py, host.py, tests/test_grant_end_toast.py)

**Inspected:** Grant expiry, user stop, disconnection, revocation, duplicate teardown, notification failures.

**Confirmed findings:** No defects found by the audit tracks. A toast failure does not prevent cleanup (verified by inspection).

**Risks investigated but not reproduced:** Duplicate teardown notifications; the implementation guards against them.

**Conflicts/dependencies:** None. Mergeable: true. (Base is old but the files merge cleanly.)

**CI:** Cancelled (old mutation job; no remediation push).

**Fix commit:** None (no changes made).

**Tests:** Pass per earlier CI (non-mutation jobs).

**Remaining verification:** On-device (real toast notifications on grant end). Marked outstanding.

**Disposition:** Needs more work (mutation CI must complete via the #70 workflow before merge).

---

## PR #66: Agent Link: step budget and visual audit trail

**Link:** https://github.com/fernandogarzaaa/project-away/pull/66
**Head:** 32f755c5 (refreshed; was eb35b435)
**Base:** 3b4063e4 (28 commits behind main)
**Files:** 48 files

**Inspected:** `_require_grant`, `_consume_step`, `act_sequence` per-sub-step budget, fail-fast pre-check, session trail persistence, and the 3 regression tests.

**Confirmed findings (Track B):** Real defect. `act_sequence` consumed 1 step for N actions (demonstrated: 3 actions to 1 step; 10 actions on budget 5 to 1 step). Fix: `_require_grant` + `_consume_step` per sub-step, fail-fast pre-check. 3 tests, 18/18 pass, all fail pre-fix. testteeth 77.1% (survivors are telemetry only).

**Risks investigated:** Concurrent requests overspending via races (lock-protected); exhausted budget blocks input; partial completion logged truthfully.

**Conflicts/dependencies:** Mergeable: false. **BLOCKED by #68/#64:** overlaps with both in `control.py` (`act_sequence`, `_policy_check`, `_audit`, `require_grant`). A three-way semantic merge is needed. **INTEGRATION NOTE (Phase 7):** #66's `_undo` does not call `_consume_step`, so undo does not consume budget despite the code comment saying "Each action (and undo) counts one step." #64's `undo_all_by_user` explicitly needs no grant ("it is their own input"). The combined behavior (user-initiated undo bypasses budget) is defensible but the comment should be corrected.

**CI:** Run 39095129628 queued on head 32f755c5.

**Fix commit:** 32f755c5 (Track B).

**Tests:** 18/18 pass locally.

**Remaining verification:** None platform-specific for budget logic.

**Disposition:** Blocked by dependency (semantic merge with #68 and #64 required).

---

## PR #59: Diagnose: camera privacy and shutter check

**Link:** https://github.com/fernandogarzaaa/project-away/pull/59
**Head:** c5988552 (refreshed; was 63096286)
**Base:** 3b4063e4 (28 commits behind main)
**Files:** 48 files

**Inspected:** `read_consent` (HKCU/HKLM fallback), registry read failures, shutter guidance, fix-button route.

**Confirmed findings (Track C):** Real defect. `read_consent` returned None on empty HKCU value without trying HKLM. Fixed. 3 regression tests fail pre-fix.

**Risks investigated but not reproduced:** Unknown consent state must not be reported as granted or denied; the fix preserves the unknown state.

**Conflicts/dependencies:** Mergeable: false. Base is 28 commits behind main.

**CI:** Run 39093562919 queued. **FAILING CHECK on this head:** `test (macos-latest, py3.12)` failed in the "Core tests" step. Real failure on the remediation push.

**Fix commit:** c5988552 (Track C).

**Tests:** 3 regression tests pass locally.

**Remaining verification:** On-device Windows (real registry reads, shutter detection). Marked outstanding.

**Disposition:** Needs more work (macos CI failure must be diagnosed; merge conflicts must be resolved).

---

## PR #45: Agent Link: allowlisted system-tool opening

**Link:** https://github.com/fernandogarzaaa/project-away/pull/45
**Head:** 8d2cda3d (refreshed; was 6f226cc0)
**Base:** main (retargeted from 3b4063e4 by Track D)
**Files:** 47 files

**Inspected:** The `SAFE_SYSTEM_TOOLS` allowlist, platform checks, launcher behavior, terminal restrictions.

**Confirmed findings (Track D):** Real security defect. Task Manager allowlist enabled terminal-block bypass via File > Run new task. Removed from `SAFE_SYSTEM_TOOLS`. 11/11 tests pass.

**Risks investigated:** Ways to navigate from an approved tool to consequential system operations; the removal closes the Task Manager vector.

**Conflicts/dependencies:** Mergeable: false (mechanical: linear history vs old merge-base; trees correct per Track D). Base is now main.

**CI:** Run 39094275655 completed, cancelled (old mutation job).

**Fix commit:** 8d2cda3d (Track D).

**Tests:** 11/11 pass.

**Remaining verification:** On-device Windows (real tool launching, bypass attempts). Marked outstanding.

**Disposition:** Needs more work (mergeable=false must be resolved; mutation CI via #70 workflow).

---

## PR #63: Onboarding: first-run wizard page for agent connection

**Link:** https://github.com/fernandogarzaaa/project-away/pull/63
**Head:** 11fbfecc (unchanged)
**Base:** 3b4063e4
**Files:** 4 files (agentlink/gui.py, app/first_run.py, app/first_run_gui.py, tests)

**Inspected:** Wizard cancellation, "Later" path, repeated opening, thread boundaries, window lifecycle.

**Confirmed findings:** No defects found by the audit tracks.

**Risks investigated but not reproduced:** Background thread/UI-thread violations; the implementation respects the boundary.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Success.

**Fix commit:** None (no changes made).

**Tests:** Pass per CI.

**Remaining verification:** On-device (real first-run flow). Marked outstanding.

**Disposition:** Ready for human review.

---

## PR #61: Tray: live who-can-see-what popup

**Link:** https://github.com/fernandogarzaaa/project-away/pull/61
**Head:** cc02f136 (unchanged)
**Base:** 3b4063e4
**Files:** 3 files (agentlink/host.py, app/tray.py, tests/test_tray_popup.py)

**Inspected:** Live capability state, stale/disconnected agents, last-activity timestamps, UI lifecycle.

**Confirmed findings:** No defects found. The popup does not falsely imply an agent is actively watching (verified by inspection: it shows capabilities and last activity, not live surveillance).

**Risks investigated but not reproduced:** Stale agent entries; the implementation prunes disconnected agents.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Success.

**Fix commit:** None (no changes made).

**Tests:** Pass per CI.

**Remaining verification:** On-device (real tray popup with live agents). Marked outstanding.

**Disposition:** Ready for human review.

---

## PR #53: Agent Link: friendlier resume wording

**Link:** https://github.com/fernandogarzaaa/project-away/pull/53
**Head:** df4fbd51 (unchanged)
**Base:** 3b4063e4
**Files:** 2 files (control.py, tests)

**Inspected:** The "until takeover" resume option, pause-on-input behavior, session scoping.

**Confirmed findings:** No defects found. The option persists only for the intended grant/session, pauses immediately on user input, and cannot silently resume after revocation or a fresh session (verified by inspection and tests).

**Risks investigated but not reproduced:** Silent resume after unrelated revocation; the implementation ties the option to the grant lifecycle.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Success.

**Fix commit:** None (no changes made).

**Tests:** Pass per CI.

**Remaining verification:** None platform-specific.

**Disposition:** Ready for human review.

---

## PR #60: Reports: one audience policy for self-test and Check my PC

**Link:** https://github.com/fernandogarzaaa/project-away/pull/60
**Head:** 37483a22 (refreshed; was 689f70a2)
**Base:** 3b4063e4
**Files:** 9 files

**Inspected:** Every route exposing self-test reports: `selftest_latest` (service), MCP tool, REST `/v1/selftest/latest`, cached retrieval paths.

**Confirmed findings (Track B):** Real security defect. `selftest_latest` had ZERO authorization checks (tool, REST, MCP resource). Fix: gate on DIAGNOSE or SHARE_LOGS. Behavior change vs PR author intent, flagged in the track report. 32/32 tests pass, both new tests fail pre-fix.

**Phase 7 validation:** Integration tests confirm the fix holds across the service method (the single choke point for MCP, REST, and direct calls). An agent without DIAGNOSE or SHARE_LOGS is refused with Forbidden. Agents with either scope can read. The authorization check precedes report loading; no cached path bypasses it.

**Risks investigated:** Cached results from another agent's authorized session; the check applies to every read, so no leakage.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Run 39095002146 queued on head 37483a22.

**Fix commit:** 37483a22 (Track B).

**Tests:** 32/32 pass locally. Phase 7 integration: 4/4 pass.

**Remaining verification:** None platform-specific.

**Disposition:** Ready for human review (CI pending on the new head).

---

## PR #55: Self-test: fix buttons for driver-broken cameras

**Link:** https://github.com/fernandogarzaaa/project-away/pull/55
**Head:** e5cb7ed7 (unchanged)
**Base:** 3b4063e4
**Files:** 3 files (selftest/checks.py, selftest/fixes.py, tests)

**Inspected:** Fix-button evidence requirements, driver data validation.

**Confirmed findings (Track C):** No defects. Suggested fixes are tied to reliable, camera-specific evidence; unknown or contradictory driver data does not trigger unsafe recommendations.

**Risks investigated but not reproduced:** Misleading recommendations from partial driver data; the implementation requires specific evidence.

**Conflicts/dependencies:** None. Mergeable: true. (28 commits behind main per Track C, but merges cleanly; out of scope for rebasing per the track.)

**CI:** Success.

**Fix commit:** None (no changes made).

**Tests:** Pass per CI.

**Remaining verification:** On-device Windows (real driver-broken camera). Marked outstanding.

**Disposition:** Ready for human review.

---

## PR #54: Agent Link: pairing approval shows what permissions mean

**Link:** https://github.com/fernandogarzaaa/project-away/pull/54
**Head:** 7e8250a7 (unchanged)
**Base:** 3b4063e4
**Files:** 3 files (agentlink/api.py, agentlink/perms.py, tests)

**Inspected:** Permission explanations vs actual granted scopes, expiry, computer-control behavior.

**Confirmed findings:** No defects found. The displayed explanations match the actual scopes; no permission is implied narrower or safer than it is.

**Risks investigated but not reproduced:** Expiry misrepresentation; the explanations include the actual expiry.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Success.

**Fix commit:** None (no changes made).

**Tests:** Pass per CI.

**Remaining verification:** On-device (real pairing flow). Marked outstanding.

**Disposition:** Ready for human review.

---

## PR #50: Agent Link: verify _open_app actually opened the app

**Link:** https://github.com/fernandogarzaaa/project-away/pull/50
**Head:** 29de2693 (unchanged)
**Base:** main (347d51f8)
**Files:** 3 files (control.py, mcp_tools.py, tests/test_safety_hardening.py)

**Inspected:** Title/process-name matching, timeout semantics, foreground-window polling, launch-vs-confirm distinction.

**Confirmed findings:** No defects found by the audit tracks.

**Risks investigated but not reproduced:** False "opened" when the launch cannot be confirmed; the implementation distinguishes unsuccessful launch from unconfirmed launch.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Success (two runs).

**Fix commit:** None (no changes made).

**Tests:** Pass per CI.

**Remaining verification:** On-device Windows (real app launching). Marked outstanding.

**Disposition:** Ready for human review.

---

## PR #49: Agent Link: focus-verified typing

**Link:** https://github.com/fernandogarzaaa/project-away/pull/49
**Head:** a2f0eebc (refreshed; was 8bcbb64a)
**Base:** main (347d51f8)
**Files:** 4 files (control.py, groups.py, mcp_tools.py, tests)

**Inspected:** Focus validation before typing, TOCTOU window, fail-closed behavior.

**Confirmed findings (Track D):** Real defect. `expect` was not re-validated after the user prompt (time-of-check/time-of-use). Fix: re-validate after `_confirm_keys`, fail closed. Tests fail pre-fix.

**Risks investigated:** Focus changing between validation and typing; the re-validation closes the window.

**Conflicts/dependencies:** None. Mergeable: true.

**CI:** Success (two runs on head a2f0eebc).

**Fix commit:** a2f0eebc (Track D).

**Tests:** Pass per CI.

**Remaining verification:** On-device Windows (real focus changes during typing). Marked outstanding.

**Disposition:** Ready for human review.

---

## Confirmed blockers

1. **#68 macos CI failure** (`test (macos-latest, py3.12)`, "Core tests" step) on head 35cd4c23. Real failure on the remediation push. Must be diagnosed before merge.
2. **#64 mypy failure** (`lint`, "Run mypy" step) on head f4dca251. Real failure on the remediation push. Must be fixed before merge.
3. **#59 macos CI failure** (`test (macos-latest, py3.12)`, "Core tests" step) on head c5988552. Real failure on the remediation push. Must be diagnosed before merge.
4. **#65 blocked by #68** (incompatible `_point` signatures; semantic merge required).
5. **#66 blocked by #68 and #64** (overlapping refactors in `control.py`; three-way semantic merge required).
6. **CodeRabbit quota exhausted** repo-wide; no bot reviews available. Manual review required for all PRs.

## Security and privacy findings

- **#60 (FIXED):** `selftest_latest` had zero authorization checks. Any agent could retrieve self-test reports. Now gated on DIAGNOSE or SHARE_LOGS. Phase 7 integration tests confirm the fix.
- **#45 (FIXED):** Task Manager in the allowlist enabled terminal-block bypass via File > Run new task. Removed.
- **#64 (FIXED):** `undo_all_by_user` undid other agents' actions. Now scoped to receipt item IDs.
- **#66 (FIXED):** `act_sequence` consumed 1 budget step for N actions, allowing budget bypass. Now per-sub-step.
- **#68 (FIXED):** Stale UIA element IDs could be clicked after window change. Now revalidated with foreground fingerprint.
- **#65 (FIXED):** Stale set-of-mark coordinates survived fresh screenshots. Now cleared at lifecycle boundaries.
- **#49 (FIXED):** Focus not re-validated after user prompt (TOCTOU). Now re-validated, fail-closed.

## Test gaps

- **On-device Windows validation** is outstanding for all hardware-dependent PRs (#71 camera, #72 COM, #74 audio, #68 UIA, #65 screenshots, #57/#59/#46/#55 camera diagnostics, #56 click verify, #49 focus typing, #50 app open, #52 toasts, #63 wizard, #61 tray popup, #54 pairing). Mock-based tests provide deterministic coverage; CI does not prove hardware behavior.
- **#68 testteeth 39.8%** (edge branches uncovered). The fix logic is tested; edge branches need more tests if the PR is to meet the 80 bar.
- **#73, #52, #46** have no completed mutation score (old workflow timeout). They need the #70 matrix workflow.

## Dependency order

The PRs fall into three groups:

**Group 1 (independent, main-based, no conflicts):** #70, #71, #74, #73, #72, #57, #56, #50, #49, #63, #61, #53, #60, #55, #54, #52. These can merge in any order once their individual gates pass. #70 should go first (it fixes the mutation CI that #73 and #52 need).

**Group 2 (old-base, need rebase):** #46, #59, #45. These need their base updated to main and conflicts resolved. #59 also has a macos CI failure to fix.

**Group 3 (semantic merge required):** #68, #65, #66, #64. These conflict with each other and cannot merge independently. They need a coordinated integration:
1. Merge #68 (UIA grounding) first as the foundation.
2. Semantically merge #65 (set-of-mark) onto #68, resolving the `_point` signature and screenshot pipeline.
3. Semantically merge #66 (step budget) onto the result, resolving the `control.py` overlaps.
4. Semantically merge #64 (consent receipts) onto the result, resolving the `control.py` overlaps.
5. Fix the #68 macos and #64 mypy CI failures during the merge.

## Safest integration sequence

1. **#70** (mutation CI). Unblocks mutation scores for #73, #52, #46.
2. **#71, #74, #72** (crash fixes). Independent, well-tested, high value.
3. **#60** (report authz). Security fix, independent.
4. **#49, #50, #56** (agent actions). Independent, CI green.
5. **#57, #55, #54, #63, #61, #53** (diagnostics/UI). Independent, CI green.
6. **#73, #52** (need #70 for mutation CI).
7. **Group 2** (#46, #59, #45): rebase, fix CI, then merge.
8. **Group 3** (#68, #65, #66, #64): coordinated semantic merge, then merge as a unit or in dependency order.

Each merge must be verified individually: CI green on the head, CodeRabbit (or human) review, one at a time per the standing orders.
