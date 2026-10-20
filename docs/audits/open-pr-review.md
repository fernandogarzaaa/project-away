# Project Away: Open-PR Review, Security Audit 2026-10-20

**Audit date:** 2026-10-20
**Repository:** fernandogarzaaa/project-away
**Base:** main at `9347132afc5e5a9837a3e7525a099493b986e498` (verified live through the GitHub API at audit start and again before every fix branch was cut)
**Companion documents:** `docs/audits/main-branch-security-audit.md` (the findings) and `docs/audits/remediation-matrix.md` (finding-to-fix mapping, CI evidence, merge order)

This document supersedes the 2026-10-18 open-PR re-audit review that previously lived at this path. That program closed on 2026-10-20 with all 23 of its PRs merged.

## State at audit start

The audit mission assumed PR #66 was still open at head `7d7ce017`. That premise was stale. PR #66 had merged minutes before the audit began, as main commit `9347132a`. Zero PRs were open when the audit started.

Consequence: the code the mission targeted for review is main-branch code. Phase 2 therefore verified the external review's claims against main itself, and every fix landed as a new branch and a new PR off main. The audit mission's rules bound the whole program: nothing merged, nothing pushed to main directly, no PR closed, no force-push, one branch per coherent fix.

## What the audit produced

Fourteen fix PRs, #75 through #88. All fourteen are open. None is merged. Merging any of them needs Inan's explicit call.

| PR | Branch | Head SHA | Purpose |
|----|--------|----------|---------|
| #75 | `fix/budget-reservation` | `4645dcd3b27b5e7f79aebc1b1d8720ef476d757a` | Sequence budget reservation plus whole-operation serialization in the action engine (Phase 2 claims C and C2) |
| #76 | `fix/uia-identity` | `26bba6b2bdf6083cf375e778015abda270f33b16` | UIA element identity hardening (Phase 2 claims D1 to D6) |
| #77 | `fix/audit-trail-integrity` | `84f9eedfe4bde6c1c1fee63595cb4520dd81556f` | Audit trail integrity (Phase 2 claims E1 to E4) |
| #78 | `fix/mutation-summary-integrity` | `2aa3e8ec1dc302570c06fc2f44a2f2caff5ff5f6` | Mutation summary denominators plus a matrix/GROUPS sync check (Phase 2 claims F and F2) |
| #79 | `fix/grant-admission-race` | `c399ce03cf3a797be59079c2c666ea633ddf13de` | Control grant admission race and re-request while paused (Phase 2 claims A and A2) |
| #80 | `fix/undo-progress-tracking` | `d7114c4ffc895c1c2214d5b0d7cab3136a4f0855` | Undo per-step progress tracking (Phase 2 claim B) |
| #81 | `fix/docs-api-contract` | `3350a14f0ef841eb338a8612f3b1440837d2112b` | Docs and API contract: README tool count, `steps_left` coverage, password-field wording, GitHub Action SHA pinning (Phase 4 lane B items 5.1, 5.2, 5.3, 4.4 in part) |
| #82 | `fix/source-identity-trust` | `78860f34f1159f73868f7e1984ca45be7836834e` | Source identity trust: `cf-connecting-ip` honored only while a Cloudflare tunnel is up (Phase 4 lane A item 1) |
| #83 | `fix/open-app-keystroke-safety` | `8b7b7567529cb6e933b9b0bee2af1dad5319f7b9` | open_app verifies the launcher is foreground before any keystroke (Phase 4 lane B item 1.6) |
| #84 | `fix/auth-lockout-scope` | `de39f2974c2b1d602817d8edf79a76567a18676a` | Failed-auth lockout scoped per source; valid credentials pass during a lock (Phase 4 lane A item 2) |
| #85 | `fix/redaction-and-trail-off` | `7ce20692f83212387b992743a2d2fa822061625a` | Redaction fails closed on secret-shaped key names; switching the audit trail off deletes stored trail files; pruning removes a session's thumbnails (Phase 4 lane A items 6a, 6c, 6b residue) |
| #86 | `fix/action-prompt-coverage` | `95a59577fc2c3acf59ad9e88f84bc3e1e6eac388` | Prompt coverage for mark clicks and drags, activation keys, Delete, and per-chunk typing re-verification (Phase 4 lane B items 1.1, 1.2, 1.3, 1.4) |
| #87 | `fix/config-atomic-save` | `1a24aff78ec8327d95a756a7a29f545694808de7` | Atomic config save with a backup fallback on load (Phase 4 lane B item 4.1) |
| #88 | `fix/mic-thread-supervision` | `8e7665de12f2ccaeaa50d5d74cf5ea829d6db42b` | Supervised microphone lifecycle for dictation (Phase 4 lane B item 3.4) |

## Notes

- CI evidence for every head above lives in `docs/audits/remediation-matrix.md`, one row per PR with both run IDs. A queued run is recorded as queued. Nothing in this program counts a queued, cancelled, or missing check as a pass.
- This documentation set ships on its own docs branch. Its PR is opened separately from the fourteen fix PRs.
- The recommended merge order is in `docs/audits/remediation-matrix.md`. It is a recommendation only.
