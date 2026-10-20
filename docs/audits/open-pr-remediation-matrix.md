# Project Away: Open PR Remediation Matrix

**Date:** 2026-10-18  
**Main:** 234a2ab2

| PR | Head Before | Head After | Finding | Severity | Fix Commit | Regression Test | CI | Mergeable | Blocker | Disposition |
|----|-------------|------------|---------|----------|------------|-----------------|----|-----------|---------|-------------|
| #46 | 77394e72 | 77394e72 | None | - | - | - | Triggered | Yes | None | Ready |
| #49 | a2f0eebc | 897ae6bb | Focus not checked between chunks | Low | 897ae6bb | test_type_text_aborts_when_window_changes_mid_typing | Triggered | Yes | None | Ready |
| #50 | 29de2693 | 9970600f | Substring false positives | Low | 9970600f | test_open_app_short_names_need_exact_match | Triggered | Yes | None | Ready |
| #52 | 0a6e56bf | 0a6e56bf | None | - | - | - | Triggered | Yes | None | Ready |
| #53 | df4fbd51 | 26a935cb | Ambiguous wording | Low | 26a935cb | Updated 5 assertions | Triggered | Yes | None | Ready |
| #54 | 7e8250a7 | 7e8250a7 | None | - | - | - | Triggered | Yes | None | Ready |
| #55 | e5cb7ed7 | 9ab759e2 | Substring device match | Low | 9ab759e2 | test_driver_state_row_rejects_similar_names | Triggered | Yes | None | Ready |
| #56 | 3c1a3081 | 3c1a3081 | None | - | - | - | Triggered | Yes | None | Ready |
| #57 | a0dd930f | a0dd930f | None | - | - | - | Green (prior) | Yes | None | Ready |
| #59 | 545f9c12 | 7251a538 | Need malformed-data tests | Low | 7251a538 | 3 consent tests | Triggered | Yes | None | Ready |
| #60 | 37483a22 | 52ca9ac3 | Cross-run attribution | Medium | 52ca9ac3 | test_run_selftest_wait_joins_existing_run | Triggered | Yes | None | Ready |
| #61 | cc02f136 | cc02f136 | None | - | - | - | Triggered | Yes | None | Ready |
| #63 | 11fbfecc | fa4962c4 | Wizard crash on close | Low | fa4962c4 | Existing 26 tests | Triggered | Yes | None | Ready |
| #64 | 2846e4a8 | 3d56d63d | Undo race + partial failures | High/Med | 3d56d63d | 2 new tests | Triggered | Yes | None | Ready |
| #65 | 46f8528b | d00b7675 | Stale tool counts | Low | d00b7675 | - | Triggered | Yes | Needs #68 first | Ready |
| #66 | b9e73357 | fed4a2ab | F841 + stale counts | Low | fed4a2ab | - | Triggered | Yes | Needs #68 first | Ready |
| #68 | ffb294c7 | 83a1feb1 | 3 HIGH stale-ID bugs | High | 83a1feb1 | 9 new tests | Triggered | Yes | None | Ready |
| #70 | 5fad0c2e | 6b4d7fba | 7 groups <80%, Windows fails | High | Multiple | 189 new tests | Triggered | Yes | 6 groups unverified | Blocked |
| #72 | b8e8fea8 | 4ec90841 | Merge conflict | Low | 4ec90841 | - | Triggered | Yes | None | Ready |
| #74 | fdc4110d | dc41b703 | Double-close hazard | High | dc41b703 | 2 new tests | Triggered | Yes | None | Ready |

## Merge Order (dependency-aware)

1. **#70** (mutation CI) - unblocks CI for others, but currently blocked on verification
2. **#68** (UIA grounding) - must precede #65/#66 (they carry its code)
3. **#72** (COM hardening) - independent
4. **#74** (audio) - independent
5. **#60, #64** (security) - independent
6. **#65, #66** - after #68 (need forward-merge for hardening)
7. **Remaining** (#46, #49, #50, #52, #53, #54, #55, #56, #57, #59, #61, #63) - independent, any order

**Note:** Merge queue is PAUSED per audit rules. Resume requires Inan's explicit go.
