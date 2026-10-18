# Project Away: Open PR Re-audit Review

**Review date:** 2026-10-18  
**Repository:** fernandogarzaaa/project-away  
**Main branch:** 234a2ab2 (after #71, #45, #73 merges; merge queue PAUSED per audit rules)  
**Auditor:** Cookie (4 crew subagents + 1 mutation crew)

## Executive Verdict

**20 open PRs reviewed.**  
- **Ready (pending CI):** 19  
- **Blocked:** 1 (#70, mutation scores unverified for 6 groups)  
- **On hold:** 0  
- **Needs human review:** 0  

**Fixes made:** 14 PRs received code fixes with regression tests.  
**Verified clean:** 6 PRs needed no changes.

## Inventory

| PR | Title | Head SHA | Branch |
|----|-------|----------|--------|
| #46 | Driver-state diagnostics | 77394e72 | feature/away-09-driver-state |
| #49 | Focus-verified typing | 897ae6bb | feature/away-05-focus-type |
| #50 | App-open verification | 9970600f | feature/away-07-openapp-verify |
| #52 | Grant-end toast | 0a6e56bf | feature/away-20-grantend-toast |
| #53 | Friendlier resume wording | 26a935cb | feature/away-21-resume-wording |
| #54 | Pairing permission explanations | 7e8250a7 | feature/away-18-pairing-meanings |
| #55 | Driver fix buttons | 9ab759e2 | feature/away-11-driver-fixbtn |
| #56 | Click verification via UIA | 3c1a3081 | feature/away-06-click-verify |
| #57 | Camera-holder diagnosis | a0dd930f | feature/away-13-cam-holders |
| #59 | Camera privacy/shutter | 7251a538 | feature/away-12-privacy-shutter |
| #60 | Self-test report audiences | 52ca9ac3 | feature/away-14-audience-fix |
| #61 | Tray privacy popup | cc02f136 | feature/away-19-tray-popup |
| #63 | First-run wizard | fa4962c4 | feature/away-23-first-run-wizard |
| #64 | Consent receipts | 3d56d63d | feature/away-17-consent-receipts |
| #65 | Set-of-mark annotations | d00b7675 | feature/away-25-setofmark |
| #66 | Step budget + audit trail | fed4a2ab | feature/away-27-step-budget |
| #68 | UIA grounding + element IDs | 83a1feb1 | feature/away-24-uia-grounding |
| #70 | Mutation CI matrix | 6b4d7fba | fix/mutation-ci-timeout |
| #72 | COM hardening | 4ec90841 | fix/com-hardening |
| #74 | Audio output serialization | dc41b703 | fix/audio-out-serialize |

## Confirmed Defects and Fixes

### HIGH Severity

**#68: Stale element IDs usable after failed query**  
`ui_query()` returned early on UIA failure without clearing the agent's registry. Old IDs stayed resolvable. Fixed: invalidate registry at start of query; install only on success. 9 regression tests added.

**#68: Missing fingerprints compared equal**  
`None != None` is False, so two "unknown" windows passed the check. Fixed: require non-None fingerprints.

**#68: Inconclusive revalidation fell back to cached**  
`_revalidate_element()` silently returned on exception/None, allowing stale element. Fixed: fail closed with ControlRefused.

**#74: Double-close on dead stream handle**  
`_closed_by_stop` (single-element list) was overwritten when multiple play_blocking calls unwound. Caused double `Pa_CloseStream` on already-closed handle. Fixed: set-based per-stream tracking. Reproduced and verified.

**#64: Concurrent undo double-undo race**  
Entry selected under lock, but `result="undone"` set outside lock. Two threads could select same entry. Fixed: atomic claim with "undoing" state.

### MEDIUM Severity

**#68: Element IDs survived revocation**  
`revoke_agent()` and `_end()` didn't clear `_element_queries`. Fixed.

**#60: Cross-run report attribution**  
`run_selftest(wait=True)` returned another agent's report without indicating it wasn't the caller's. Fixed: response now marks `joined=True` with trigger name.

**#64: Partial undo failures swallowed**  
`undo_all_by_user` broke silently on cancellation, returning only success count. Fixed: includes `failed=[...]` and `partial=True`.

### LOW Severity

**#49: Focus check not per-chunk**  
Window verified before typing but not between chunks. Could type into wrong app after focus change. Fixed: per-chunk verification.

**#50: Substring matching false positives**  
Short names ("a", "it") matched unrelated apps. Fixed: names <3 chars require exact match.

**#55: Substring device matching**  
"HD Webcam" could match "HD Webcam Pro". Fixed: exact match only.

**#53: Ambiguous resume wording**  
"Until I take over again" implied expiry on takeover. Fixed: "for the rest of this session".

**#63: Wizard crash on close**  
Background thread called `root.after()` after window destroyed. Fixed: `winfo_exists()` check.

**#66: Ruff F841**  
Unused variable `g` in `undo_last()`. Fixed.

## Security Findings

**#60 Policy:** DIAGNOSE is intentionally permitted to read full self-test reports (per `perms.py` line 14: "run them, read their results"). Not an implementation inference.

**#64:** Receipts properly scoped via `only_ids`. Forged IDs fail closed. No sensitive data in receipts (filenames not images, "Typed 7 characters" not content).

**#59:** Consent interpretation correct. `read_consent` returns None (not "Allow") for missing stores. Never infers positive permission.

**#57:** Three-way distinction preserved (confirmed/maybe/historical). No changes needed.

## Remaining Blockers

**#70:** 6 mutation groups have new tests but scores unverified (cursor, face, dictation, autoscroll, cleanup, vad). Crashlog verified at 80.2%. CI script now includes new test files (was the root cause of stagnant scores).

## Limitations

- On-device Windows/COM/UIA validation outstanding (mock tests only).
- CI results pending for all new heads at report time.
- #65/#66 carry #68's pre-hardening code; need forward-merge after #68 lands.
