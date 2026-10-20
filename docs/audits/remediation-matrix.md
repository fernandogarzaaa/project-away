# Project Away: Remediation Matrix, Security Audit 2026-10-20

**Audit date:** 2026-10-20
**Base:** main at `9347132afc5e5a9837a3e7525a099493b986e498`
**Companion documents:** `docs/audits/main-branch-security-audit.md` (finding entries with root causes and reproductions) and `docs/audits/open-pr-review.md` (state and PR inventory)

This matrix covers the 2026-10-20 main-branch security audit only. The earlier file `docs/audits/open-pr-remediation-matrix.md` covers the 2026-10-18 open-PR program and is left as the record of that program.

Fix commit SHA means the PR head SHA at writing. Each head contains its fix and its regression tests. Nothing in this matrix is merged.

## Finding matrix

| Finding | Severity | Status | Regression test file | Fix commit SHA |
| --- | --- | --- | --- | --- |
| A. Grant admission race | High | Fixed in PR #79 | `tests/test_v0129_grant_admission.py` | `c399ce03cf3a797be59079c2c666ea633ddf13de` |
| A2. Re-request while paused discards the session | Medium | Fixed in PR #79 | `tests/test_v0129_grant_admission.py` | `c399ce03cf3a797be59079c2c666ea633ddf13de` |
| B. Undo retry replays partial work | High | Fixed in PR #80 | `tests/test_v0129_undo_progress.py` | `d7114c4ffc895c1c2214d5b0d7cab3136a4f0855` |
| C. Sequence preflight without reservation | Medium | Fixed in PR #75 | `tests/test_v0129_budget_reservation.py` | `4645dcd3b27b5e7f79aebc1b1d8720ef476d757a` |
| C2. Whole-operation interleaving | Low | Fixed in PR #75 | `tests/test_v0129_budget_reservation.py` | `4645dcd3b27b5e7f79aebc1b1d8720ef476d757a` |
| D1. Tree before fingerprint | High | Fixed in PR #76 | `tests/test_v0129_uia_identity.py` | `26bba6b2bdf6083cf375e778015abda270f33b16` |
| D2. Element ids reused across queries | Medium | Fixed in PR #76 | `tests/test_v0129_uia_identity.py` | `26bba6b2bdf6083cf375e778015abda270f33b16` |
| D3. No post-prompt revalidation (click, drag) | High | Fixed in PR #76 | `tests/test_v0129_uia_identity.py` | `26bba6b2bdf6083cf375e778015abda270f33b16` |
| D4. Name plus type identity fallback | Medium | Fixed in PR #76 | `tests/test_v0129_uia_identity.py` | `26bba6b2bdf6083cf375e778015abda270f33b16` |
| D5. ui_query grant gate | Medium | Fixed in PR #76 | `tests/test_v0129_uia_identity.py` | `26bba6b2bdf6083cf375e778015abda270f33b16` |
| D6. Grant end clears ids and marks | Medium | Fixed in PR #76 | `tests/test_v0129_uia_identity.py` | `26bba6b2bdf6083cf375e778015abda270f33b16` |
| E1. Trail undercounts sequences | Low | Fixed in PR #77 | `tests/test_v0129_audit_trail.py` | `84f9eedfe4bde6c1c1fee63595cb4520dd81556f` |
| E2. No durable undo transition | Low | Fixed in PR #77 (progress mechanics in PR #80) | `tests/test_v0129_audit_trail.py` | `84f9eedfe4bde6c1c1fee63595cb4520dd81556f` |
| E3. JSONL never reloaded | Low | Fixed in PR #77 | `tests/test_v0129_audit_trail.py` | `84f9eedfe4bde6c1c1fee63595cb4520dd81556f` |
| E4. Zero and negative limits | Low | Fixed in PR #77 | `tests/test_v0129_audit_trail.py` | `84f9eedfe4bde6c1c1fee63595cb4520dd81556f` |
| F. Mutation summary denominators | Medium | Fixed in PR #78 | `tests/test_mutation_summary_gate.py` | `2aa3e8ec1dc302570c06fc2f44a2f2caff5ff5f6` |
| F2. Matrix/GROUPS drift unchecked | Low | Fixed in PR #78 | `tests/test_mutation_matrix_sync.py` | `2aa3e8ec1dc302570c06fc2f44a2f2caff5ff5f6` |
| A-1. cf-connecting-ip trust | Medium | Fixed in PR #82 | `tests/test_v0129_source_identity.py` | `78860f34f1159f73868f7e1984ca45be7836834e` |
| A-2. Global lockout denial of service | Medium-low | Fixed in PR #84 | `tests/test_v0129_lockout_scope.py` | `de39f2974c2b1d602817d8edf79a76567a18676a` |
| A-3. DNS-rebinding coverage | None | Refuted | None (audit repro only, uncommitted) | None |
| A-4. Token lifecycle | None | No defect found | None | None |
| A-5. Permission-negative sweep | None | Refuted | None (audit repro only, uncommitted) | None |
| A-6a. Redaction key-name residual | Low | Fixed in PR #85 | `tests/test_v0129_redaction_trailoff.py` | `7ce20692f83212387b992743a2d2fa822061625a` |
| A-6b. Pruning leaves a session's thumbnails | Low | Fixed in PR #85 | `tests/test_v0129_redaction_trailoff.py` | `7ce20692f83212387b992743a2d2fa822061625a` |
| A-6c. Trail-off disk residue | Low | Fixed in PR #85 | `tests/test_v0129_redaction_trailoff.py` | `7ce20692f83212387b992743a2d2fa822061625a` |
| A-6d. safepaths confinement | None | Refuted | None (audit repro only, uncommitted) | None |
| A-6e. Capture indicators | None | Refuted | None | None |
| B-1.1. Mark clicks skip prompts | Medium | Fixed in PR #86 | `tests/test_v0129_prompt_coverage.py` | `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388` |
| B-1.2. Space activates risky control unprompted | Medium | Fixed in PR #86 | `tests/test_v0129_prompt_coverage.py` | `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388` |
| B-1.3. Delete with a safe focused label | Low to medium | Fixed in PR #86 (narrow rule) | `tests/test_v0129_prompt_coverage.py` | `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388` |
| B-1.4. Typing target checked once | Medium | Fixed in PR #86 | `tests/test_v0129_prompt_coverage.py` | `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388` |
| B-1.5. Password protection off Windows | Low | Accepted limitation (platform); wording fixed in PR #81 | `tests/test_v0129_docs_contract.py` (wording only) | `3350a14f0ef841eb338a8612f3b1440837d2112b` (wording only) |
| B-1.6. open_app keystrokes precede verification | Low to medium | Fixed in PR #83 | `tests/test_v0129_open_app_safety.py` | `8b7b7567529cb6e933b9b0bee2af1dad5319f7b9` |
| B-1.7. Other injection paths | None | Refuted | None | None |
| B-2.1. Threads, owners, shutdown, buffers | None | Refuted | None | None |
| B-2.2. Lock ordering | None | Unconfirmed (experiment named in the audit doc) | None | None |
| B-3.1. Camera holder state and release | None | Refuted | None | None |
| B-3.2. Actual versus inferred diagnose states | None | Refuted | None | None |
| B-3.3. Face gate versus its contract | None | Refuted | None | None |
| B-3.4. Mic failure kills dictation silently | Low to medium | Fixed in PR #88 | `tests/test_v0129_mic_supervision.py` | `8e7665de12f2ccaeaa50d5d74cf5ea829d6db42b` |
| B-4.1. Non-atomic config save | Medium-low | Fixed in PR #87 | `tests/test_v0129_config_atomic.py` | `1a24aff78ec8327d95a756a7a29f545694808de7` |
| B-4.2. Updater verification | None | Refuted | None (audit repro only, uncommitted) | None |
| B-4.3. Path safety on user-supplied names | None | Refuted | None | None |
| B-4.4. Pinning posture | Low | Fixed in PR #81 (actions half); lockfile not implemented, see recommendations | `tests/test_v0129_docs_contract.py` | `3350a14f0ef841eb338a8612f3b1440837d2112b` |
| B-4.5. MSIX correctness | None | Unconfirmed (experiment named in the audit doc) | None | None |
| B-5.1. README tool count | Low | Fixed in PR #81 | `tests/test_v0129_docs_contract.py` | `3350a14f0ef841eb338a8612f3b1440837d2112b` |
| B-5.2. steps_left coverage | Low | Fixed in PR #81 | `tests/test_v0129_docs_contract.py` | `3350a14f0ef841eb338a8612f3b1440837d2112b` |
| B-5.3. Password-field wording | Low | Fixed in PR #81 | `tests/test_v0129_docs_contract.py` | `3350a14f0ef841eb338a8612f3b1440837d2112b` |
| B-5.4. Other contract spot-checks | None | Refuted | None | None |

## CI evidence

One row per fix PR. Every pull_request run in this table was verified by the parent by reading its job list directly, not by trusting a reported verdict: each recorded success means every job in the run concluded success, including every mutation group. A cancelled dispatch run recorded below is one the parent cancelled itself as a duplicate, to drain a saturated queue; the canonical pull_request run is the evidence for that head.

| PR | Head SHA | pull_request run | workflow_dispatch run | Status at writing |
| --- | --- | --- | --- | --- |
| #75 | `4645dcd3` | 39437897086 | 39438050682 | Both success (parent verified) |
| #76 | `26bba6b2` | 39439650397 | 39439651402 | Both success (parent verified) |
| #77 | `84f9eedf` | 39440434359 | 39440478195 | Both success. PR run 26/26 jobs. Dispatch run attempt 2 success after the attempt 1 failure explained below. |
| #78 | `2aa3e8ec` | 39449467654 | 39449599761 | PR run success, 27 jobs including the new mutation-groups-check job. Dispatch run cancelled by the parent as a duplicate. |
| #79 | `c399ce03` | 39440968691 | 39440969048 | Both success. PR run 26/26 jobs. Dispatch attempt 2 success after the macOS glide self-test flake (assert hz > 5, measured 4.0 on a loaded runner) was re-run. |
| #80 | `d7114c4f` | 39441810533 | 39441809492 | Both success (parent verified) |
| #81 | `3350a14f` | 39452426584 | 39452462697 | PR run success, 26/26 jobs, attempt 2. Attempt 1 failed only the macOS job on a voice latency ratio flake (1541.5 ms against a 1536.4 ms line); re-run passed. Dispatch run cancelled by the parent as a duplicate. See the history note below. |
| #82 | `78860f34` | 39446969989 | 39446969692 | Both success (parent verified) |
| #83 | `8b7b7567` | 39447553917 | 39447568421 | Both success (parent verified) |
| #84 | `de39f297` | 39447653938 | 39447687181 | PR run success, 26/26 jobs. Dispatch run cancelled by the parent as a duplicate. |
| #85 | `7ce20692` | 39450418632 | 39450420629 | PR run success, 26/26 jobs. Dispatch run cancelled by the parent as a duplicate. |
| #86 | `95a59577` | 39452121988 | 39452154791 | PR run success, 26/26 jobs. Dispatch run cancelled by the parent as a duplicate. |
| #87 | `1a24aff7` | 39452200251 | 39452239489 | PR run success, 26/26 jobs. Dispatch run cancelled by the parent as a duplicate. |
| #88 | `8e7665de` | 39453788206 | 39453810843 | PR run success, 26/26 jobs. Dispatch run cancelled by the parent as a duplicate. |

**Note on PR #77's dispatch run.** The mutation summary job failed in its "download mutation reports" step because downloading the filters group artifact returned HTTP 503 (GitHub artifact storage egress over the account limit). The summary step was skipped and no score was computed in that run. The pull_request run 39440434359 on the same head passed the full mutation gate. The failed job has been re-run by the parent and completed success on attempt 2.

**History note on PR #81.** The branch's first head, `3f31aaab`, failed all platform test jobs on one stale test expectation: `tests/test_msix_packaging.py` expected the SignPath action at `@v3` after the branch itself had SHA-pinned that action. The fix updated the expectation to the pin. The head in the table above, `3350a14f`, carries that fix. Its attempt 1 run failed only the macOS job on a voice latency ratio flake (after total 1541.5 ms against a 1536.4 ms line on a loaded runner); the re-run passed all 26 jobs on attempt 2.

## Recommended merge order

Rationale: land the independent, low-overlap fixes first, then the `control.py` cluster in dependency order. Several cluster PRs touch the same functions, so each landing should be followed by fresh CI on the resulting main head before the next one lands. PR #77 and PR #80 both modify `_undo` behavior in `control.py` (#77 the outcome reporting, #80 the per-step progress), and PR #79 touches the admission paths in the same file. Expect textual conflicts at sequencing in that cluster, not semantic ones.

1. **#87** (config atomic save). Isolated: `config.py` only.
2. **#82** (source identity). Touches the api.py auth area.
3. **#84** (lockout scope). Also touches the api.py auth area. Land adjacently after #82 and re-check the combined auth surface.
4. **#85** (redaction and trail-off).
5. **#88** (mic supervision). Note that it also touches the `scripts/mutation.py` dictation group.
6. **#78** (mutation summary and ci.yml).
7. **#81** (docs and API contract). Touches the ci.yml action pins and the `control.py` `steps_left` paths. Land after #78 so the ci.yml changes stack once.
8. **#75** (budget reservation and engine serialization). First in the control.py cluster; later cluster fixes assume its budget accounting.
9. **#79** (grant admission).
10. **#83** (open_app keystroke safety).
11. **#80** (undo progress).
12. **#77** (audit trail). Touches `_undo` reporting; land after #80.
13. **#76** (UIA identity).
14. **#86** (prompt coverage). Builds on the post-prompt recheck pattern from #76; land last in the cluster.

Nothing in this program merges without Inan's explicit call. This order is a recommendation only.

## Recommendations not implemented

1. **Python dependency lockfile** (from finding B-4.4). Python dependencies are version ranges in `pyproject.toml` with no lockfile. PR #81 pinned the GitHub Actions to commit SHAs. The dependency half is unchanged: adopt a lockfile (for example a compiled requirements file or the project's chosen resolver lock) so installs are reproducible and auditable. Recorded as a recommendation; no PR in this program implements it.
2. **Permission census** (from finding A-5). The permission-negative sweep exercised the named boundaries and both image scopes, not all roughly 30 service methods. A mechanical census that matches each service method's require call against the tool table would close the coverage gap permanently, ideally as a test.
3. **DNS-rebinding structural guard** (from finding A-3). The rebinding defense is one middleware wrapper around one application. A test that fails when a route or mount exists outside that wrapper would turn the fragility note into a checked invariant.
4. **Real-hardware passes.** PR #76's exact-bounds identity check and PR #88's mic supervision are proven by deterministic tests. Both still need a pass on real Windows hardware: UIA hit-testing on the Acer for #76, and a device-less dictation run for #88.
