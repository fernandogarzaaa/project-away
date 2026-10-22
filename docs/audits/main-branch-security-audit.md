# Project Away: Main-Branch Security Audit, 2026-10-20

**Audit date:** 2026-10-20
**Repository:** fernandogarzaaa/project-away
**Audited code:** main at `9347132afc5e5a9837a3e7525a099493b986e498` (verified live through the GitHub API before each phase)
**Companion documents:** `docs/audits/open-pr-review.md` (state and PR inventory) and `docs/audits/remediation-matrix.md` (finding-to-fix mapping, CI evidence, merge order)

## Methods

The audit ran in four phases. Phase 1 mapped the module inventory and the trust boundaries. Phase 2 verified the external review's claims about the computer-control code (the code that merged as PR #66, audited here as main code) claim by claim. Phase 4 swept the rest of the surface in two lanes: lane A covered auth, network, and privacy; lane B covered action safety, platform lifecycle, packaging, and the docs contract.

Every verdict rests on a running reproduction, not on code reading alone. The Phase 2 reproductions are 15 scratch tests in `test_audit_repro.py`, all passing against the audited main, where a passing test means the claimed defect reproduced. Lane A ran 7 scratch reproduction tests and lane B ran 9. Scratch reproductions were never committed. Each fix PR carries its own committed regression tests, and each of those tests was shown to fail on the unfixed code before the fix landed.

Evidence discipline note. One fix agent's first full-suite figure was retracted by that agent after its own recheck. The retracted figure appears nowhere in this audit. Every CI state quoted in these documents was re-read from the GitHub API by the parent agent after the fix agent reported.

Severity labels for Phase 4 findings are the lane reports' own guesses, kept as guesses. Severity labels for Phase 2 findings were assigned during this audit. Neither set was re-graded after the fixes.

## Trust boundary summary

Principals, strongest first: the person at the PC; local processes running as the same user and holding `local.key`; paired agents with a bearer token and per-agent scopes; unauthenticated network callers, who reach only the landing page, `/v1/ping`, `/openapi.json`, and the pairing endpoints.

Seven permission scopes live in `src/away/agentlink/perms.py`. `read` is locked on. `diagnose` defaults on. `snapshots`, `screen`, `tune`, `share_logs`, and `control` default off. Enforcement funnels through `Service.require()` at the top of each service method, and every transport (REST, MCP over HTTP, the stdio bridge) dispatches into the same service methods.

Every HTTP request passes `AuthMiddleware` in `src/away/agentlink/api.py`: a Host allowlist and strict Origin match, JSON-only bodies with size caps, a per-source token bucket (10 per second, burst 40), a failed-auth lockout, then authentication. Pairing uses a link code of about 50 bits plus a 6-digit PIN, single-use, 10-minute TTL, approved on the PC. Tokens are 32 random bytes stored as SHA-256 only, compared in constant time. Refresh rotates with a 120-second grace. Persistent `paway_` tokens reopen 24-hour sliding sessions. Revocation bumps a gate drop epoch that aborts in-flight requests.

There is exactly one TCP listener, bound to 127.0.0.1 (`src/away/agentlink/server.py`). Remote access is outbound-only: a cloudflared tunnel with a SHA-256-pinned binary, or SSH reverse relays. The one non-loopback inbound surface is the iPhone Live Link Face UDP listener. Downloads by agents are confined by `src/away/agentlink/safepaths.py`. Logs and config pass through the redaction in `src/away/agentlink/bundle.py` before export.

The control path is the highest-stakes surface. The `control` scope plus a per-session grant (user-approved, time-limited, step-budgeted, paused on takeover) gates `ComputerControl`, which drives `ActionEngine`, which injects OS input. Most Phase 2 findings live on this path.

---

## Phase 2 findings: computer control

### A. Grant admission race overwrites a live grant

**Severity:** High
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl.request()` (line 431)

**Root cause.** `request()` checks the current grant, then blocks on the approval prompt for up to 60 seconds while holding no lock and no reservation. After approval it assigns `self.grant` under the lock with no recheck of what the grant became in the meantime. Two concurrent requests can both pass the check and both be approved. The second assignment silently replaces a live grant. The overwritten grant is never ended: its cancel event never fires, no `control.ended` event or receipt is emitted, and its session trail never gets an ended line.

**Reproduction.** A barrier inside the fake prompt held two requests in flight at once. Both returned active, two `control.started` events fired, and zero `control.ended` events fired.

**Fix.** PR #79, head `c399ce03cf3a797be59079c2c666ea633ddf13de`. Admission is decided atomically under the control lock before any prompt, through a pending-admission reservation. At most one approval prompt is in flight. A grant installs only while the caller's reservation token is still its own. Stop, stop-all, and revoke cancel the reservation, so a late approval installs nothing. Regression tests in `tests/test_v0129_grant_admission.py`.

**Remaining limitations.** A concurrent second `request_control` now returns an immediate refusal instead of queueing. A client retrying in a tight loop spins on refusals, bounded by the 60-second prompt timeout.

### A2. Re-request while paused silently discards the paused session

**Severity:** Medium
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl.request()` (lines 433-437)

**Root cause.** The same agent calling `request()` while its grant is paused falls through both early returns and reaches a fresh prompt. On approval a brand-new grant overwrites the paused one. The old session ends with no end event, no receipt, and no trail line. Its remaining budget and undo state vanish with it.

**Reproduction.** After a re-request during pause, the grant id differed, the old grant's ended field stayed empty, and no `control.ended` or `control.receipt` event was emitted.

**Fix.** PR #79, head `c399ce03cf3a797be59079c2c666ea633ddf13de`. Re-request while paused now refuses with a message pointing at `resume_control`. The paused session is kept and ends only through stop, revoke, or expiry. Ending the old session on re-request was rejected during the fix because a denied replacement prompt would have destroyed a resumable session.

**Remaining limitations.** An agent whose paused session it cannot resume depends on the user stopping it. No agent-initiated end-session call exists.

### B. Undo retry replays an inverse sequence that already partially ran

**Severity:** High
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl._undo()` (line 1620)

**Root cause.** `_undo()` runs the inverse sub-steps in a plain loop and records no per-step progress. Both failure exits (cancellation and the generic exception path) reset the entry to `done`, which makes the entry fully undoable again. A retry re-runs the whole inverse sequence from the first sub-step, including sub-steps that already executed.

**Reproduction.** A two-scroll sequence was undone with the second inverse scroll made to raise. The entry returned to `done`. A second undo applied the first inverse scroll again, so that inverse ran twice in total, and the entry ended `undone`.

**Fix.** PR #80, head `d7114c4ffc895c1c2214d5b0d7cab3136a4f0855`. `LogEntry.undo_progress` records each inverse step's outcome. Interruption before any inverse input leaves the entry fully retryable. Any other interruption settles to a partially-undone state, never back to `done`. Retries run only the steps never attempted. Regression tests in `tests/test_v0129_undo_progress.py`.

**Remaining limitations.** A partially-undone entry whose remaining steps all failed or were cancelled is terminal; no force-retry affordance exists. Per-step progress is written at settle points only, so a process crash mid-undo loses it. That crash exposure matches the pre-existing in-memory log.

### C. Sequence budget preflight reserves nothing

**Severity:** Medium
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl.act_sequence()` (line 1304)

**Root cause.** `act_sequence()` preflights the plan count against the remaining budget under the lock, then releases the lock and reserves nothing. Each sub-step consumes budget as it executes. A concurrent action from the same agent that lands between preflight and a later sub-step consumes budget, and the sequence refuses mid-run, after earlier sub-steps already produced side effects.

**Reproduction.** With budget 3, a 3-step sequence blocked inside its first sub-step while a concurrent `act()` spent one step. The sequence refused after 2 of 3 sub-steps executed.

**Fix.** PR #75, head `4645dcd3b27b5e7f79aebc1b1d8720ef476d757a`. `act_sequence` reserves its whole plan atomically at admission. `_consume_step` converts reserved steps as sub-steps run. A `finally` releases reservations for sub-steps that never ran. Regression tests in `tests/test_v0129_budget_reservation.py`.

**Remaining limitations.** None recorded beyond the fix scope.

### C2. Whole-operation interleaving protection

**Severity:** Low
**Verdict:** PARTIALLY TRUE
**Location:** `src/away/actions_engine.py`, `ActionEngine._io` (line 477)

**What is true.** The engine serializes individual OS calls only. One click or one chord cannot be split, but nothing holds a lock across a whole operation, so the steps of two operations can interleave between engine steps. Cross-agent interleaving is normally prevented by the single-grant design. The exposure is same-agent concurrent calls, plus any second grant admitted through the claim A race.

**What is not true.** Single OS calls are atomic. The claim overstated the gap if read as per-call interleaving.

**Reproduction.** Established by reading the lock scope together with the claim C reproduction, where a concurrent call landed mid-sequence.

**Fix.** PR #75, head `4645dcd3b27b5e7f79aebc1b1d8720ef476d757a`. The engine gained a whole-operation lock for agent-policy sources, with a cancellation-aware acquire that re-checks the cancellation token, engine state, and the stop-all generation every 20 ms, so takeover, stop-all, pause, and shutdown never deadlock behind a queued operation. User sources (gesture, voice, dictation, macro) keep per-call atomicity by design.

**Remaining limitations.** Queued-operation cancellation latency is bounded by the 20 ms poll.

### D1. UIA tree is built before the foreground fingerprint is captured

**Severity:** High
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl.ui_query()` (line 851)

**Root cause.** `ui_query()` walks `uia.tree()` first and captures `_window_fingerprint()` only when installing the element registry. A foreground change during enumeration pairs window A's elements with window B's fingerprint. Later resolution accepts those ids while B is foreground, because the stored fingerprint genuinely equals B's.

**Reproduction.** The fake tree walk flipped the foreground from window A to window B mid-query. Element `e1` from A's tree still resolved afterwards.

**Fix.** PR #76, head `26bba6b2bdf6083cf375e778015abda270f33b16`. The fingerprint is captured before and after the tree walk. The registry installs only when both captures are non-null and equal; otherwise the query reports unavailable. Regression tests in `tests/test_v0129_uia_identity.py`.

**Remaining limitations.** Real-Windows UIA behavior is unverified on hardware. See the platform note under D4.

### D2. Element ids are reused across queries

**Severity:** Medium
**Verdict:** PARTIALLY TRUE
**Location:** `src/away/agentlink/control.py`, `_ElementQuery` and `ComputerControl._resolve_element_id()` (line 899)

**What is true.** Ids are plain `e1`, `e2`, assigned per query, so the same string names a different element after every new query. The registry's generation counter increments per query, but resolution never checks it. A stale id lands within the same window, on a different element, with no refusal.

**What mitigates it.** The registry is per agent. The prior registry is dropped at the start of each query. Resolution requires the stored window fingerprint to still match the live foreground. `_revalidate_element` re-checks the live element at the resolved center. The stale id cannot jump windows.

**Reproduction.** Two queries returned different elements as `e1`, with generations 1 then 2. The id from the first query resolved to the second query's element.

**Fix.** PR #76, head `26bba6b2bdf6083cf375e778015abda270f33b16`. Element ids now come from a per-agent monotonic counter and are never reissued. A stale id stays an unknown element id.

**Remaining limitations.** None recorded beyond the fix scope.

### D3. No revalidation after an approval prompt before click or drag

**Severity:** High
**Verdict:** CONFIRMED for click and drag
**Location:** `src/away/agentlink/control.py`, `ComputerControl._act()` (about lines 1040-1090)

**Root cause.** `_act()` resolves and revalidates the element first, then `_confirm_targets()` shows the risky-action prompt for up to 60 seconds, then the code clicks at the pre-prompt coordinates without re-resolving the element or re-checking the foreground fingerprint. A foreground or target change during the prompt does not stop the click. `type_text` had a partial exception: with a newline in the text and an `expect` string given, the window title is re-checked after the prompt and before each chunk. Click, drag, and `press_keys` had no equivalent.

**Reproduction.** Clicking a Delete element by id, the fake prompt flipped the foreground to another window and emptied the element lookup while the prompt was up. The click still executed at the pre-prompt coordinates.

**Fix.** PR #76, head `26bba6b2bdf6083cf375e778015abda270f33b16`. A post-prompt recheck re-resolves the target after any approval prompt and refuses on a foreground, element, or move change. It is wired into click, double-click, right-click, both drag ends, and the element-focus click used by `type_text` and `press_keys`. Regression tests in `tests/test_v0129_uia_identity.py`.

**Remaining limitations.** The recheck covers element-id and element-name targets. Pure coordinate targets and mark targets carry no UIA identity to recheck. Marks are bounded by the D6 fix.

### D4. Name plus control type treated as identity when the live element lacks an automation id

**Severity:** Medium
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl._revalidate_element()` (line 922)

**Root cause.** `_revalidate_element()` compares automation ids only when both the cached and live elements have one. When the cached element has an automation id and the live one does not, execution falls through to the name check: same name and same control type is accepted as the same element. A second gap in the same function: when the cached element's name is empty, no identity check runs at all, and any element found at the center passes.

**Reproduction.** The cached element had `automation_id` set. The live element at the center was a different element with an empty automation id but the same name and control type. Resolution succeeded.

**Fix.** PR #76, head `26bba6b2bdf6083cf375e778015abda270f33b16`. Revalidation requires exact bounds in all cases. A snapshot automation id must be matched live. Name plus control type at the recorded bounds counts as identity only when the snapshot has no automation id. An empty name skips no check.

**Remaining limitations.** The exact-bounds requirement is a fail-closed trade-off. A real `element_at` hit test on Windows can return a child whose bounds differ from the snapshot, which now refuses. That behavior needs a pass on real Windows UIA before the trade-off is fully priced.

### D5. ui_query requires the CONTROL permission but no live grant

**Severity:** Medium
**Verdict:** PARTIALLY TRUE
**Location:** `src/away/agentlink/service.py`, `Service.ui_query()` (lines 920-924); `src/away/agentlink/control.py`, `ComputerControl.ui_query()` (line 851)

**What is true.** The CONTROL permission half holds on every path. All transports funnel through the service, and `Service.ui_query()` calls `require(agent, perms.CONTROL)` first. The grant half held nowhere. `ComputerControl.ui_query` never called `_require_grant`, unlike `act`, `act_sequence`, and `undo_last`. An agent holding the CONTROL permission with no approved session could enumerate the foreground window's element names, roles, and bounds at any time, including while another agent's grant was paused or after its own grant ended.

**Reproduction.** With no grant in existence and no request ever made, `ui_query` at the control layer returned the element list.

**Fix.** PR #76, head `26bba6b2bdf6083cf375e778015abda270f33b16`. `ComputerControl.ui_query` now requires a live, unpaused grant owned by the caller, the same gate as act and undo. It consumes no budget step. The service layer still checks the CONTROL permission first.

**Remaining limitations.** Callers outside this repo that ran `ui_query` without a grant now get a refusal. No in-repo caller depended on the old behavior.

### D6. Grant end clears element ids but not set-of-mark coordinates

**Severity:** Medium
**Verdict:** PARTIALLY TRUE
**Location:** `src/away/agentlink/control.py`, `ComputerControl._end()` (line 512) and `revoke_agent()` (line 646)

**What is true.** `_end()` pops the ending agent's element registry on expiry, stop, and disconnect alike, so element ids die with the grant. It did not touch `self.marks`. Marks, views, and scales were cleared only in `revoke_agent()`. After an ordinary grant end, a later grant for the same agent could act on mark coordinates captured in the earlier session, with no fresh screenshot.

**Reproduction.** Marks set during grant 1 survived `stop()`. In grant 2, a mark reference resolved to the old coordinates (111, 222). The same test confirmed the element registry was cleared, which is the contrast.

**Fix.** PR #76, head `26bba6b2bdf6083cf375e778015abda270f33b16`. `_end()` now clears marks, views, and scales alongside element ids on every end path.

**Remaining limitations.** None recorded beyond the fix scope.

### E1. Sequence budget counts sub-steps, the trail records one row

**Severity:** Low
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl.act_sequence()`, `_log()`, `sessions_summary()`

**Root cause.** `act_sequence()` spends one budget step per sub-step, while each sub-step runs with recording off and writes no log entry. The sequence writes exactly one combined entry. `sessions_summary()` counts trail rows, so its steps figure disagrees with the grant's `steps_used` by design. Per-sub-step timestamps, results, and thumbnails did not exist in the trail. The separate engine audit log did receive one entry per sub-step; the session trail and the Settings replay did not.

**Reproduction.** A 3-step sequence left `steps_used` at 3, the summary steps figure at 1, one trail entry, and one step event in the session JSONL.

**Fix.** PR #77, head `84f9eedfe4bde6c1c1fee63595cb4520dd81556f`. A sequence now files one trail event per executed sub-step, each linked to a grouping entry, on the done, stopped, and refused paths alike. `sessions_summary()` reports a steps figure that agrees with `steps_used`, plus a new calls figure. Regression tests in `tests/test_v0129_audit_trail.py`.

**Remaining limitations.** Session files written before the fix reload as one step and one call, because per-sub-step data was never persisted for them.

### E2. Undo appends no durable status transition linked to the original action

**Severity:** Low
**Verdict:** PARTIALLY TRUE
**Location:** `src/away/agentlink/control.py`, `ComputerControl._undo()` (about line 1670)

**What is true.** An agent-initiated undo that succeeds appends a new undo entry, but that entry carries its own id, not the original action's. The persisted original still reads `done` forever. A user-initiated undo appends nothing at all. A failed or cancelled undo appends nothing either. Only the in-memory entry and the transient event stream carried the transition.

**Reproduction.** After a user undo, the session file was byte-identical to before, its step event still said `done`, while the in-memory entry read `undone`.

**Fix.** PR #77, head `84f9eedfe4bde6c1c1fee63595cb4520dd81556f`. `_undo()` counts the applied inverse steps and appends a status event linked to the original action id, with the outcome undone, partially undone, or undo failed. Agent, user, and receipt-scoped undos all use this path. The per-step progress mechanics underneath are PR #80.

**Remaining limitations.** None recorded beyond the fix scope.

### E3. Persisted trail history is never reloaded on startup

**Severity:** Low
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl._trail_event()` (about line 576); `src/away/agentlink/host.py` (lines 1452-1453)

**Root cause.** The session JSONL files were write-only. No reader existed anywhere in `src/away`; the only other touch was pruning, which deletes. The Settings replay reads in-memory session data only, so after a restart, earlier sessions were invisible in the app even though their files and thumbnails remained on disk. The module docstring and the config description promised replay in Settings without stating the same-process limit.

**Reproduction.** A second `ComputerControl` over the same folder reported no sessions and an empty trail while the session file with step events existed on disk.

**Fix.** PR #77, head `84f9eedfe4bde6c1c1fee63595cb4520dd81556f`. Session restore runs at startup: newest sessions up to the existing caps, malformed or truncated lines skipped per line, missing thumbnails cleared while the records are kept, status transitions applied, session ids sanitized against path escapes. Restored sessions participate in pruning. With the trail switched off, nothing persists and nothing reloads.

**Remaining limitations.** Legacy files carry the E1 limitation above.

### E4. Trail query limits of zero or negative return more than requested

**Severity:** Low
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl.actions()` (line 1531) and `session_trail()` (line 1559); `src/away/agentlink/service.py`, `Service.action_log()` (lines 941-946)

**Root cause.** Both methods sliced with `[-limit:]` and no validation. Python slice semantics turn a limit of 0 into the whole list and a negative limit into all but the first entries. The agent-facing call site forwarded the agent-supplied limit unclamped.

**Reproduction.** With 3 logged actions, a limit of 0 returned 3, a limit of -1 returned 2, and the session trail with a limit of 0 returned all 3.

**Fix.** PR #77, head `84f9eedfe4bde6c1c1fee63595cb4520dd81556f`. The documented behavior is now enforced: a zero or negative limit returns no entries. The clamp sits in both control methods and again at the service boundary.

**Remaining limitations.** None recorded beyond the fix scope.

### F. Mutation summary prints a percentage and a ratio with different denominators

**Severity:** Medium
**Verdict:** CONFIRMED
**Location:** `scripts/mutation.py`, `summary()` (lines 86-116)

**Root cause.** The summary printed each group's percentage from the report's own score, whose denominator counts every mutant including the no-coverage ones. The ratio printed beside it used a denominator of killed plus survived plus timeouts, omitting no-coverage. The OVERALL line used the same wrong denominator. One concrete instance: a dictation report printed `63.8% 60/61`, where 63.8 percent is 60 of 94 and 60 of 61 would be 98.4 percent. The gate itself used the score consistently, so gating was sound; the printed evidence could not be reproduced from the printed numbers. The gate tests never caught it because their synthetic reports carried no no-coverage field.

**Reproduction.** Feeding the dictation report shape to `summary()` produced a line containing both `63.8%` and `60/61`.

**Fix.** PR #78, head `2aa3e8ec1dc302570c06fc2f44a2f2caff5ff5f6`. One denominator everywhere, matching the testteeth score definition: killed plus timeouts over killed plus survived plus no-coverage plus timeouts. Each group row and the OVERALL line also print the four counts separately. Regression tests in `tests/test_mutation_summary_gate.py`.

**Remaining limitations.** If testteeth changes its score definition, the summary's definition must follow by hand.

### F2. Mutation matrix and GROUPS can drift with no direct check

**Severity:** Low
**Verdict:** PARTIALLY TRUE
**Location:** `.github/workflows/ci.yml` (lines 181-198); `scripts/mutation.py`, `GROUPS`

**What is true.** The CI matrix hardcoded the same 15 groups as `GROUPS`, tied together only by a keep-in-sync comment. No test or CI step compared the two sets. The sets matched at audit time, 15 groups each.

**What mitigates it.** Drift failed only incidentally: a group missing from the matrix failed later in the summary job's download step, and a group missing from `GROUPS` failed its own job with an unknown-group error. Neither failure names the drift.

**Fix.** PR #78, head `2aa3e8ec1dc302570c06fc2f44a2f2caff5ff5f6`. `tests/test_mutation_matrix_sync.py` compares the full matrix set against `GROUPS` and names both drift directions. A lightweight `mutation-groups-check` CI job runs that test, and the mutation matrix job now needs it.

**Remaining limitations.** The matrix stays hand-maintained, now guarded by the sync test. The sync test's parser fails loudly on a wholesale ci.yml restructure rather than passing silently.

---

## Phase 4 lane A findings: auth, network, privacy

### A-1. Source identity trusts the client-supplied cf-connecting-ip header

**Severity:** Medium (lane guess)
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/api.py`, `AuthMiddleware.__call__` (line 541)

**Root cause.** The middleware computed the request's source identity from the `cf-connecting-ip` header whenever present, ahead of the actual socket peer, unconditionally. That one value keys three consumers: the rate-limit bucket, the per-source persistent-token backoff, and the access log plus each token's recorded last-used source. The server binds loopback only, so any local process could set the header freely, and peers arriving over the SSH reverse relays could too, because those relays forward raw TCP with inner headers untouched.

**Reproduction.** With a small test bucket, five rapid headerless pings from one peer tripped a 429, while ten pings with rotating header values all returned 200. Eight bad persistent-token attempts under one claimed source ended in the backoff 429; the same bad token under a fresh claimed source returned a plain 401, and the access log recorded the spoofed address.

**Fix.** PR #82, head `78860f34f1159f73868f7e1984ca45be7836834e`. One function computes source identity. It returns the socket peer unless the host itself reports a Cloudflare tunnel up. The default is distrust: an empty header, a failing state check, or an unknown state all fall back to the peer. The SSH relay is deliberately excluded, because no Cloudflare edge sits in its path. While a tunnel is up, distinct header values still get distinct buckets and backoffs, which is the legitimate path. Regression tests in `tests/test_v0129_source_identity.py`.

**Remaining limitations.** While a tunnel is up, the header is trusted for all requests, including direct loopback arrivals in that window. That is strictly narrower than the old always-trust behavior. Whether a client-supplied header can survive Cloudflare's own edge is unconfirmed; the fix does not depend on that answer. The experiment that would settle it is in the unconfirmed section below.

### A-2. The failed-auth lockout is one global counter

**Severity:** Medium-low (lane guess)
**Verdict:** CONFIRMED, with a scoped exemption
**Location:** `src/away/agentlink/ratelimit.py`, `FailureLockout`; constructed in `src/away/agentlink/api.py` (line 247)

**Root cause.** The lockout kept one global failure list and one global lock deadline: 10 failures inside 60 seconds locked everyone for 60 seconds. Failures are recorded from unauthenticated paths (bad pairing code, bad access token, bad refresh, forbidden pair or local pair). The lock check ran in the middleware before routing. The lockout's reset was never called anywhere, so a lock ended only when its timer expired, and each new cycle needed only 10 fresh cheap failures.

**Reproduction.** After ten bogus pair attempts, a valid access-token holder got 429 on `/v1/status` and even on the public `/v1/ping`. A persistent `paway_` token returned 200 during the same lock, which confirms the exemption is exactly the persistent-token class.

**Fix.** PR #84, head `de39f2974c2b1d602817d8edf79a76567a18676a`. The lockout is per source. A request presenting valid credentials passes even while its source has failures. A successful authentication clears that source's failure count. Unauthenticated brute force from one source still locks that source's unauthenticated attempts. Regression tests in `tests/test_v0129_lockout_scope.py`.

**Remaining limitations.** Per-source scoping inherits the source-identity question of finding A-1; the two fixes land adjacently in the merge order for that reason.

### A-3. DNS-rebinding coverage of the MCP transport

**Verdict:** REFUTED as a gap
**Location:** `src/away/agentlink/mcp_tools.py` (about line 51); `src/away/agentlink/api.py` (lines 1159-1160)

The transport's DNS-rebinding protection is disabled, and the audit checked whether anything reachable sits outside the compensating control. Exactly one Starlette application exists in `src/`. It is returned wrapped in `AuthMiddleware`, and its only production caller is the Agent Link host. A search found no second application, no mount, and no websocket route anywhere in `src/`. Every route, including `/mcp` and the SSE stream, sits behind the middleware's Host and Origin guard. The one adjacent server, the private opencode child process in `liveagent`, serves no Agent Link routes and is not reachable through the tunnel.

The boundary held. One fragility note stands: the defense is a single wrapper, and nothing fails if a future mount bypasses it. That is a watch item for future refactors, not a hole today.

### A-4. Token lifecycle

**Verdict:** No defect found. The lifecycle matches its documented design on every point checked.

Pairing codes are 10 characters from a 32-character alphabet, about 50 bits, drawn from `secrets`. PINs are 6 digits. The TTL is 600 seconds. A pairing is single-use on approval or on denial, and the first claimant's name is bound. Guessing is throttled by the lockout of finding A-2.

Refresh rotation uses a deterministic 120-second grace (`REFRESH_GRACE_S`). The old access token dies instantly at rotation. Redeeming the old refresh token inside grace silently retires the intermediate refresh token; the lane recorded this behavior because it is easy to misread, and judged it consistent with the design.

Revocation deletes the agent's persistent tokens, drops its refresh material, bumps the gate drop epoch, and terminates noted MCP sessions. In-flight requests abort through the watched dispatch, which polls the epoch every 0.2 seconds. That drop timing was verified by code path only. The timed end-to-end experiment is in the unconfirmed section below.

Scope changes mutate the shared agent object, and `Service.require` re-reads scopes on every call, so a removal takes effect on the next call even inside a live session. Persistent tokens are `paway_` plus 32 random bytes, stored as SHA-256 only, with 24-hour sliding sessions, 30-day, 90-day, or never lifetimes, and a 1-hour rotation grace enforced in both authentication and the live-stream validity check.

### A-5. Permission-negative sweep

**Verdict:** REFUTED as a gap
**Location:** `src/away/agentlink/api.py`, `dispatch`; `src/away/agentlink/service.py`, `Service.require()` (line 193)

All transports funnel through `api.dispatch` into `Service.require`. The sweep exercised the boundary directly: a read-only agent was refused through dispatch for log reading and bundle export (`share_logs`), diagnose (`diagnose`), control requests (`control`), camera looks (`snapshots`), screen capture (`screen`), and settings changes (`tune`), and passed the `read` method. Over real HTTP the same agent got 403 on `/v1/logs` and 200 on `/v1/status`. A second agent holding read, diagnose, and share_logs passed the log gate and was still refused control and tune.

The boundary held. One follow-up stands: not all roughly 30 service methods were individually exercised. A full census (matching each method's require call against the tool table) is a mechanical follow-up, recorded here so the gap in coverage is explicit.

### A-6a. Bundle redaction misses secrets under unrecognized key names

**Severity:** Low (lane guess)
**Verdict:** CONFIRMED residual in an otherwise strong control
**Location:** `src/away/agentlink/bundle.py`, `redact()`

**Root cause.** `redact()` is key-name based with a value-shape pass for known token forms. The reproduction confirmed the coverage is strong for known shapes: a synthetic log line carrying a `paway_` token, a Bearer token, a GitHub token, an `api_key` value, a pairing URL, an email, and a home path came out with none of those strings present. The residual: values stored under key names the matcher does not recognize, such as `private_key` or `passphrase`, pass through untouched when the value also matches no known token shape. No current config field is known to hold such a value.

**Fix.** PR #85, head `7ce20692f83212387b992743a2d2fa822061625a`. `redact()` now fails closed on secret-shaped key names. Unambiguous substrings match anywhere in a name (token, password, passwd, passcode, passphrase, secret, credential, credentials, mnemonic, api_key). Short words match only as whole words after splitting on separators and camelCase humps (key, keys, pass, phrase, auth, oauth, bearer, cookie, cookies, session, seed, seeds, cert, certs, cred, creds, pwd, private), so prose and innocent keys survive. Regression tests in `tests/test_v0129_redaction_trailoff.py`.

**Remaining limitations.** Key-name redaction stays a heuristic. A secret under a name with no secret-shaped word in it remains a value-shape question.

### A-6b. Retention bounds

**Verdict:** Bounded everywhere checked, with one CONFIRMED residue
**Location:** `src/away/agentlink/bundle.py`; `src/away/agentlink/control.py`, `_prune_sessions()`

Bundles keep the newest 20. Trail sessions cap at 12 (`MAX_SESSIONS`). Trail rows cap at 300 per session. Action thumbnails cap at the newest 200. The residue: session pruning deleted the session's JSONL but not that session's thumbnails. Orphaned thumbnails left only through the global 200 cap.

**Fix.** PR #85, head `7ce20692f83212387b992743a2d2fa822061625a`. Pruning now deletes the pruned session's thumbnails with its JSONL.

### A-6c. Switching the audit trail off deletes nothing already recorded

**Severity:** Low (lane guess)
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/gui_model.py`, `set_audit_trail()` (line 432); `src/away/agentlink/control.py`, `_trail_event()`

**Root cause.** Switching the trail off stops new writes, because the writers are gated on the trail switch. The setter only flipped the config flag and saved. No code path deleted the existing session files or thumbnails, so everything recorded while the trail was on stayed on disk until pruning or the thumbnail cap eventually removed it. A person reading "off" as "gone" would be wrong.

**Fix.** PR #85, head `7ce20692f83212387b992743a2d2fa822061625a`. Switching the trail off now deletes the existing session JSONL files and their thumbnails in the same operation and reports what was removed.

**Remaining limitations.** None recorded beyond the fix scope.

### A-6d. Download path confinement (safepaths)

**Verdict:** REFUTED as a gap
**Location:** `src/away/agentlink/safepaths.py`, `resolve()`

The resolver refuses empty, absolute, drive-letter, backslash, and NUL forms and any dot segments. It restricts the first path part to the diagnostics, recordings, or logs roots. It then resolves both the root and the target with symlinks followed and requires the resolved target to sit inside the resolved root and to be a file. The reproduction confirmed a symlink inside the logs root pointing outside raised `UnsafePath`, while a symlink whose target stays inside its root resolved, which is the intended behavior. One theoretical residual: resolution and the later file read are two steps, so a local process able to write the data folder could swap a symlink between check and read. That attacker is already inside the trust boundary. Severity of the residual: negligible.

### A-6e. Camera and screen indicator accuracy

**Verdict:** REFUTED as a gap
**Location:** `src/away/agentlink/service.py`, `Service._flash()` (line 723), `look()` (line 504)

The flash is a precondition: if the flash callback raises or returns false, the capture is refused. The capture history is written only after a picture exists, so the history cannot claim a capture that did not happen. The live screen view refuses to start when its watching badge cannot be shown, and quiet captures skip the flash only while that same agent's badge is on screen. One precision the lane recorded: in `look()` the frame is grabbed before the flash, so on a flash failure the sensor was read but no image leaves the service and nothing is recorded. The indicator never under-reports a delivered capture.

---

## Phase 4 lane B findings: action safety, platform, packaging, docs

### B-1.1. Set-of-mark clicks and drags skip the risky and unknown prompts

**Severity:** Medium (lane guess)
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl._point()` (line 952) and `_confirm_targets()`

**Root cause.** A mark reference gets the synthetic label `mark N`, which classifies as safe. Neither the risky prompt nor the unknown-target prompt runs. The same spot clicked by raw coordinates gets its label from the element under the point and prompts when that element is risky or unnamed. Drag takes both end labels through the same helper, so a drag onto a mark over the Recycle Bin had the same gap. The prompt promise in the agent briefing has no mark exception.

**Reproduction.** Marks were set on a fake Delete button. The mark click executed with zero prompts. The coordinate click on the same spot prompted once.

**Fix.** PR #86, head `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388`. A mark click or drag end now inherits the risk classification of the element actually under the mark, resolved the way the coordinate path resolves it, and falls back to the unknown-target prompt when no element is found there. Regression tests in `tests/test_v0129_prompt_coverage.py`.

**Remaining limitations.** None recorded beyond the fix scope.

### B-1.2. Space activates a focused risky control with no prompt

**Severity:** Medium (lane guess)
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl.press_keys()` (line 1206) and `_confirm_keys()`

**Root cause.** The key prompt fired for a single Enter, for keys in the consequential set, or for combos with a shortcut modifier. Space was in none of those sets, and because space is not a single printable character, the typing-target check did not run either. Space is the standard keyboard activation key for a focused button.

**Reproduction.** With a Pay now button focused, `press_keys` with `space` executed with zero prompts. `enter` on the same focus prompted once.

**Fix.** PR #86, head `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388`. Activation keys, space and its equivalents, now go through the same focused-control judgment as Enter. Regression tests in `tests/test_v0129_prompt_coverage.py`.

**Remaining limitations.** Reaching the gap required the risky control to be focused first, usually through a prompted click or unprompted Tab navigation. The fix covers the activation step itself.

### B-1.3. Delete with a safe focused label runs unprompted

**Severity:** Low to medium (lane guess)
**Verdict:** CONFIRMED as a mechanism; whether it is a defect is a design call
**Location:** `src/away/agentlink/control.py`, `ComputerControl._confirm_keys()` (line 812)

**Root cause.** The key prompt judges the focused element's name. A selected file named `report.docx` classifies safe, so Delete runs silently even though Delete is listed as consequential. The label describes the container, not the thing Delete acts on. The design intent (a named text box goes ahead) is documented; the gap is that the judgment reads the wrong noun.

**Reproduction.** With the focused label `report.docx`, Delete executed with zero prompts.

**Fix.** PR #86, head `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388`. The narrow rule shipped: Delete prompts in file and list-item contexts, and stays unprompted inside text-editing elements, where it deletes a character. Where the UIA data cannot draw that line, Delete prompts whenever the focused element is not a text-editing element (`EditControl` or `DocumentControl`).

**Remaining limitations.** The rule depends on the focused element's reported control type. A file manager that reports a text-editing type for its list would land on the unprompted side.

### B-1.4. Typing target is verified once, not per chunk

**Severity:** Medium (lane guess)
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl.type_text()` (lines 1145-1210)

**Root cause.** `type_text` checked the typing target once before typing. Text is typed in chunks. The per-chunk re-verification re-checked only the window title against the optional `expect` argument. Without `expect`, nothing re-ran the terminal-process or password-field check between chunks. Focus can change without user input when an app raises itself or a sign-in window appears. A user takeover pauses the grant; an external focus change does not. The same gap existed after the Enter prompt when `expect` was absent. This finding is distinct from Phase 2 claim D3, which covered click and drag.

**Reproduction.** The foreground flipped to a terminal process after the first chunk of a 40-character text with no `expect`, and all 40 characters were typed. A second reproduction flipped the foreground to a terminal while the newline prompt was up, with the same result.

**Fix.** PR #86, head `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388`. The typing target class is re-verified before every chunk, and typing stops on a change. The check also runs after the Enter prompt when `expect` is absent. Regression tests in `tests/test_v0129_prompt_coverage.py`.

**Evidence caveat.** The audit's own scratch reproduction for this item was weaker than intended: its focus flip may never have fired. The mechanism was confirmed by code reading of the chunk loop, and the PR #86 regression test fires the flip properly and fails on the unfixed code. The verdict rests on those two legs, not on the scratch repro.

**Remaining limitations.** The first chunk is typed under the pre-typing check alone, as before. Protection between chunks depends on the same target classification used by the initial check, including the off-Windows limit in finding B-1.5.

### B-1.5. Off Windows, password-field protection degrades to a window-title check

**Severity:** Low (lane guess)
**Verdict:** CONFIRMED as a platform limit; the code states it
**Location:** `src/away/agentlink/control.py`, `ComputerControl._typing_target_ok()` (line 998); `src/away/agentlink/desktop.py`, `make_uia()` (line 355)

On Windows the check fails closed when UI Automation is unavailable. On macOS and Linux the UIA layer is `NullUIA`, and the remaining checks are the terminal-process match and a window-title regex. A password field in a window whose title does not match the regex is not detected off Windows. Typing into a password field types the agent's intended text and reads nothing back, which bounds the impact.

**Fix.** No code fix. The wording half of this finding is fixed in PR #81 (see B-5.3). The platform limit itself is recorded as an accepted limitation.

### B-1.6. open_app delivers its launcher keystrokes before any verification

**Severity:** Low to medium (lane guess)
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/control.py`, `ComputerControl._open_app()` (line 1398)

**Root cause.** `_open_app()` pressed the launcher key, typed the app name, and pressed Enter into whatever window had focus, and only then polled the foreground for a match. No check confirmed the launcher (Start search or Spotlight) actually opened before the name was typed. When the launcher fails to open, the app name and Enter land in the foreground app: into a chat window, which sends the name as a message; into a document; or into a terminal, where the typed name plus Enter is a command line. The result was honest about the miss, carrying a warning when verification failed, but the keystrokes were already delivered.

**Reproduction.** The foreground stayed a chat window on win32. The backend recorded the typing and key calls, and the result carried the warning.

**Fix.** PR #83, head `8b7b7567529cb6e933b9b0bee2af1dad5319f7b9`. No keystroke is sent until the launcher is verified foreground. The fix verifies, then types, then keeps the existing post-launch app verification. It refuses without typing when the launcher cannot be verified. Regression tests in `tests/test_v0129_open_app_safety.py`.

**Remaining limitations.** The typed string stays constrained by the app-name pattern (letters, digits, spaces, and a few punctuation marks, 40 characters), which bounded the original impact and still bounds what verification failure can deliver: nothing, since refusal now precedes typing.

### B-1.7. Other injection paths (gesture, macro, dictation)

**Verdict:** REFUTED as a gap

Non-agent injection funnels through the engine as user sources by design: gestures, macros, and dictation output all enter through the engine's user-source policies. Macros carry their own protections (same-app rules, a password-box skip, an abort watcher). Blocked key combos (Win+R, Win+X, Ctrl+Alt+Delete, Ctrl+Shift+Esc) are refused before any prompt logic. `open_url` accepts only http and https URLs matching a strict regex; no prompt is promised for navigation. The boundary held at every spot checked.

### B-2.1. Threads, owners, shutdown ordering, bounded buffers

**Verdict:** REFUTED as a gap

Every long-lived thread the lane found is a daemon with a named owner and a stop path. The server stop joins its thread. The host stop order is deliberate: control ends first, then prompts, tunnel, gate, server. `PromptManager.stop()` cancels all pending prompts, and prompt waits are bounded. Tunnel and relay stops terminate their subprocesses with a kill fallback. Jobs reach terminal states on every path, including exceptions. The buffers checked are bounded: the event ring, the engine audit log, the prompt history, the tunnel log buffer. No leak or hang path was found.

### B-2.2. Engine and control lock ordering

**Verdict:** UNCONFIRMED
**Location:** `src/away/agentlink/control.py` (line 354); `src/away/actions_engine.py` (line 476)

Both locks are re-entrant locks. The engine fires its stop and takeover listeners outside its own lock, so the classic two-lock cycle was not found by reading: the stop-all path takes the control lock without the engine lock held, and the control end path does its file I/O and hook callbacks after dropping the control lock, except when reached from the expiry check, which is re-entrant on the same thread. No deadlock cycle was found and no reproduction was produced. Reading alone cannot close this. The experiment that would settle it is in the unconfirmed section below.

### B-3.1. Camera holder state and release

**Verdict:** REFUTED as a gap
**Location:** `src/away/app/camera_share.py`; `src/away/multicam/camera_users.py`; `src/away/multicam/capture.py`

The camera lease uses heartbeats with a 5-second staleness bound, so a crashed holder never blocks the camera forever. Registry-based holders count only while their last-used stop time is zero. Process-based holders are labeled hints, not proof, in both code and wording. Captures are released on every failure path the lane checked, including the app camera loop's finally block. The boundary held.

### B-3.2. Actual versus inferred camera and driver states

**Verdict:** REFUTED as a gap
**Location:** `src/away/multicam/diagnose.py`

Diagnosis labels its verdicts as best guesses in its own docstrings. The driver status defaults to unknown, never to a guessed state. The real driver state comes from a PnP/CIM query whose failure surfaces as unknown. The one inference shown to users, which app is using the camera, is fed only by the registry holder list, not by the process hint list. The boundary held.

### B-3.3. Face gate behavior against its contract

**Verdict:** REFUTED as a gap
**Location:** `src/away/face/gate.py`

The speech gate behaves exactly as its docstrings state, including the pre-enrollment no-op and the stated absence of a recognition model. The push-to-talk path returns the configured allow-without-face default when no face was seen and blocks when another face speaks more than the primary. The strict ambient path requires a face at the screen and credits frontal presence only from frames whose pose is known. History is bounded. One behavior is worth naming, and it is documented rather than defective: before enrollment, the gate types anyone's speech, which is the stated MVP state.

### B-3.4. A microphone failure kills dictation silently for the session

**Severity:** Low to medium (lane guess)
**Verdict:** CONFIRMED as a mechanism; the live run is named below
**Location:** `src/away/app/runner.py` (line 1319); `src/away/voice/dictation.py`, `MicStream` (line 168)

**Root cause.** The runner's mic thread body opens the stream and waits. If the device is missing, denied, or unplugged before the stream opens, the exception leaves the thread. The thread is a daemon with no supervisor, no error handling, and no restart. The health loop publishes performance hints only and never checks thread aliveness. Dictation is then dead until the app restarts, with no user-facing explanation from this path.

**Reproduction.** Established by code path: the thread body leaves no other outcome. The sandbox has no audio device, so the live run (start the app with no input device, attempt push-to-talk dictation, observe that nothing is typed and no notice names the microphone) was not done in the audit.

**Fix.** PR #88, head `8e7665de12f2ccaeaa50d5d74cf5ea829d6db42b`. The mic lifecycle is supervised: open and read failures are caught, recorded in dictation state, and retried with bounded backoff, so there is no hot loop. Recovery resumes dictation without a restart. The failure is visible on the existing status surface. Shutdown is clean, with no leaked threads. Regression tests in `tests/test_v0129_mic_supervision.py`, using a fake sounddevice layer.

**Remaining limitations.** Live confirmation on a device-less Windows run is still outstanding. The sandbox tests prove the supervision logic, not the hardware path.

### B-4.1. Config save is not atomic, and a truncated config has no fallback

**Severity:** Medium-low (lane guess)
**Verdict:** CONFIRMED
**Location:** `src/away/config.py`, `save()` (line 1278) and `load()`; `src/away/app/tray.py`, `load_tray_config()` (line 855)

**Root cause.** `save()` truncated and wrote `config.yaml` in place: no temp file, no `os.replace`, no backup copy. The same codebase writes `agents.json` atomically, so the pattern existed and the config path did not use it. Saves are frequent: every Settings change and every agent settings change goes through this path. A truncated file makes `load()` raise, the tray loader does not catch it, and the app cannot start with no surviving copy of the user's settings. Config migration itself is clean: old mappings upgrade in place, and unknown keys are collected and reported rather than silently dropped.

**Reproduction.** A valid config was saved, then truncated to a third of its length, which is what an interrupted write leaves. `config.load` raised a YAML error.

**Fix.** PR #87, head `1a24aff78ec8327d95a756a7a29f545694808de7`. Save writes a temp file in the same directory, flushes and fsyncs it, and replaces the target atomically, matching the repo's existing atomic writer pattern. The previous valid file is kept as `config.yaml.bak`. `load()` falls back to the backup when the main file is missing or unparseable, visibly in the log, and raises only when both are unusable. Regression tests in `tests/test_v0129_config_atomic.py`.

**Remaining limitations.** None recorded beyond the fix scope.

### B-4.2. Update verification and release sums

**Verdict:** REFUTED as a gap
**Location:** `src/away/app/updater.py`, `verify_download()` (line 465)

The updater fails closed: without a published SHA-256 from the API digest or the sums file, it deletes the download and refuses the install, unless the user explicitly allowed unverified updates. The release workflow builds the sums file from the final signed assets. The docs state plainly that nothing is code-signed. No overclaim was found.

### B-4.3. Path safety on user-supplied names

**Verdict:** REFUTED as a gap at the spots checked

Macro names are validated against a strict pattern before a file path is built, and macro files write atomically. Gesture pack imports reject zip members with absolute paths, dot-dot parts, backslashes, or drive colons before extraction. Macro export refuses sensitive macros unless explicitly overridden. Agent downloads go through `safepaths`, settled in finding A-6d.

### B-4.4. Dependency and action pinning posture

**Severity:** Low (lane guess)
**Verdict:** CONFIRMED as a posture, recorded as a hardening note
**Location:** `pyproject.toml`; `.github/workflows/ci.yml`; `.github/workflows/release.yml`

At audit time, GitHub Actions were pinned to version tags, not commit SHAs. Python dependencies are version ranges with no lockfile. Downloaded binaries and models are the strongly pinned part of the repo, verified by SHA-256. This is the repo's stated posture, not a discovered defect.

**Fix.** PR #81, head `3350a14f0ef841eb338a8612f3b1440837d2112b`, pins every third-party GitHub Action in the workflows to a full commit SHA with the version kept in a comment. The SHAs were resolved through the GitHub API, never guessed.

**Remaining limitations.** The Python dependency policy is unchanged. A lockfile is recommended, not implemented; see the recommendations section of `docs/audits/remediation-matrix.md`.

### B-4.5. MSIX package correctness

**Verdict:** UNCONFIRMED
**Location:** `packaging/msix/AppxManifest.xml.in`; `scripts/build_msix.py`

The manifest is templated from Partner Center variables, and CI builds and test-installs the package on Windows. The sandbox cannot run that build. The experiment that would settle it is in the unconfirmed section below.

### B-5.1. README counts 17 tools and omits ui_query

**Severity:** Low (lane guess)
**Verdict:** CONFIRMED
**Location:** `README.md` (line 367)

The README said 17 tools and its table listed 17 rows. The actual MCP surface is 18 tools, matching the agent briefing in `about.py`. The missing row was `ui_query`. The same table listed the `open` tool's actions as app and url only; the code and the briefing also include a system action. Measured counts for the record: 18 MCP tools, 42 group actions, 42 REST tool entries.

**Fix.** PR #81, head `3350a14f0ef841eb338a8612f3b1440837d2112b`. The README now states 18 tools with the missing entries added. A tool-count test in `tests/test_v0129_docs_contract.py` fails on registry or README drift.

### B-5.2. steps_left promised on every action result, omitted by sequences and undo

**Severity:** Low (lane guess)
**Verdict:** CONFIRMED
**Location:** `src/away/agentlink/about.py`; `src/away/agentlink/control.py` (lines 1293, 1374, 1672)

The briefing tells agents that every action result carries `steps_left`. Single actions do. The sequence result did not, and neither did its per-step results or the undo result. An agent spending its budget through sequences got no remaining-budget figure from any result until it made a single action. Budget enforcement itself was correct; only the reporting promise was wider than the implementation.

**Fix.** PR #81, head `3350a14f0ef841eb338a8612f3b1440837d2112b`. Sequence, sub-step, and undo results now carry `steps_left`, using the same accounting as the single-action path on main. The promise was kept and the code was fixed, not the other way round. Regression tests in `tests/test_v0129_docs_contract.py`.

### B-5.3. "Never into password fields" stated without qualification

**Severity:** Low (lane guess)
**Verdict:** CONFIRMED as a wording defect; it is the docs half of B-1.5
**Location:** `src/away/agentlink/mcp_tools.py` (line 370); `src/away/agentlink/about.py` (line 61)

Agent-facing text stated the password-field guarantee without qualification. Per B-1.5, that guarantee is enforced by UI Automation on Windows and degrades to a title heuristic on macOS and Linux.

**Fix.** PR #81, head `3350a14f0ef841eb338a8612f3b1440837d2112b`. Both texts now carry the off-Windows caveat.

### B-5.4. Remaining contract spot-checks

**Verdict:** REFUTED as gaps

Job retention numbers in the README match the jobs module. Screenshot scale and size claims in the briefing match the screenshot module (default scale 0.5, longest side at most 2560 px). Unknown tool names over REST return a clean 404 JSON error, not a crash. Permission defaults match their documentation: read and diagnose on, everything that shows the camera or screen or touches the computer off.

---

## Unconfirmed items and the experiments that settle them

Four items could not be settled from the sandbox. Each names the experiment that would settle it. None is counted as a defect, and none is counted as clean.

1. **Engine and control lock ordering (B-2.2).** Experiment: a threaded stress harness that runs stop-all, status (which triggers the expiry check), and act concurrently for some thousands of iterations, under a watchdog that dumps thread stacks on a hang, ideally against the real app where the overlay and notify hooks do real work.
2. **cf-connecting-ip survival through the Cloudflare edge (residual of A-1).** Experiment: send a request through a live trycloudflare URL with a forged `cf-connecting-ip` header and read the source the access log records. The PR #82 fix does not depend on the answer; the experiment prices the tunnel-up window only.
3. **Revocation stream-drop timing, end to end (A-4).** Experiment: hold an SSE stream open against the real server, revoke the agent from Settings, and measure the time to close. The code path bounds it at about 0.2 seconds through the watched dispatch; the timed run was not done.
4. **MSIX correctness (B-4.5).** Experiment: a Windows run of `scripts/build_msix.py` plus the CI msix job against the current manifest.

## Verdict summary

| Finding | Verdict |
| --- | --- |
| A. Grant admission race | CONFIRMED |
| A2. Re-request while paused discards the session | CONFIRMED |
| B. Undo retry replays partial work | CONFIRMED |
| C. Sequence preflight without reservation | CONFIRMED |
| C2. Whole-operation interleaving | PARTIALLY TRUE |
| D1. Tree before fingerprint | CONFIRMED |
| D2. Element ids reused across queries | PARTIALLY TRUE |
| D3. No post-prompt revalidation (click, drag) | CONFIRMED |
| D4. Name plus type identity fallback | CONFIRMED |
| D5. ui_query grant gate | PARTIALLY TRUE |
| D6. Grant end clears ids and marks | PARTIALLY TRUE |
| E1. Trail undercounts sequences | CONFIRMED |
| E2. No durable undo transition | PARTIALLY TRUE |
| E3. JSONL never reloaded | CONFIRMED |
| E4. Zero and negative limits | CONFIRMED |
| F. Mutation summary denominators | CONFIRMED |
| F2. Matrix/GROUPS drift unchecked | PARTIALLY TRUE |
| A-1. cf-connecting-ip trust | CONFIRMED |
| A-2. Global lockout denial of service | CONFIRMED |
| A-3. DNS-rebinding coverage | REFUTED |
| A-4. Token lifecycle | No defect |
| A-5. Permission-negative sweep | REFUTED |
| A-6a. Redaction key-name residual | CONFIRMED |
| A-6b. Retention bounds | Bounded; thumbnail residue CONFIRMED |
| A-6c. Trail-off disk residue | CONFIRMED |
| A-6d. safepaths confinement | REFUTED |
| A-6e. Capture indicators | REFUTED |
| B-1.1. Mark clicks skip prompts | CONFIRMED |
| B-1.2. Space activates risky control unprompted | CONFIRMED |
| B-1.3. Delete with a safe focused label | CONFIRMED (mechanism; design call, narrow rule shipped) |
| B-1.4. Typing target checked once | CONFIRMED (see the evidence caveat in the entry) |
| B-1.5. Password protection off Windows | CONFIRMED (platform limit; accepted) |
| B-1.6. open_app keystrokes precede verification | CONFIRMED |
| B-1.7. Other injection paths | REFUTED |
| B-2.1. Threads, owners, shutdown, buffers | REFUTED |
| B-2.2. Lock ordering | UNCONFIRMED |
| B-3.1. Camera holder state and release | REFUTED |
| B-3.2. Actual versus inferred diagnose states | REFUTED |
| B-3.3. Face gate versus its contract | REFUTED |
| B-3.4. Mic failure kills dictation silently | CONFIRMED (mechanism) |
| B-4.1. Non-atomic config save | CONFIRMED |
| B-4.2. Updater verification | REFUTED |
| B-4.3. Path safety on user-supplied names | REFUTED |
| B-4.4. Pinning posture | CONFIRMED (posture; actions half fixed) |
| B-4.5. MSIX correctness | UNCONFIRMED |
| B-5.1. README tool count | CONFIRMED |
| B-5.2. steps_left coverage | CONFIRMED |
| B-5.3. Password-field wording | CONFIRMED (wording) |
| B-5.4. Other contract spot-checks | REFUTED |
