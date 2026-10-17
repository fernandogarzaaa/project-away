# Fix A: camera open gate - state (RESUMED 2026-10-17 ~18:30 UTC, budget away-fixA-resume-01)

## Resume progress
- Mutation gate RE-RUN: 100.0% PASS (32/32 killed, 0 survived, 0 uncovered, 4 timeouts;
  wait_devices_free 18/18, open_camera 14/14). Log: /tmp/fixA_mutation.log. First resume attempt
  was killed by a runtime restart; second attempt completed in ~715s.
- origin/main verified still at 347d51f8 (no merge needed).
- Branch pushed via replicate_branch.py: fix/camera-open-gate -> 3a3a966b035176616447d1ae89e2d84f0b11a6fc
- PR #71 opened: https://github.com/fernandogarzaaa/project-away/pull/71 (head 3a3a966b)
- Posted @coderabbitai review (comment 6238813886).
- CI run 39059323979 in progress (background monitor: /tmp/watch_pr71.py -> /tmp/watch_pr71.log).
- CodeRabbit quota: EXHAUSTED as of 18:43 UTC ("Next included review available in 51 minutes",
  i.e. ~19:34 UTC). Auto-review on PR open was rate-limited; manual trigger also rate-limited.
  Codex bot review also failed on usage limits (not a code finding).
- MERGE AUTHORIZATION from Inan: squash-merge when CI fully green on head AND CodeRabbit has
  reviewed that exact head SHA. Next: wait for CI, re-trigger CodeRabbit after ~19:34 UTC.

## CodeRabbit review + finding fix (2026-10-17 ~19:35-20:00 UTC)
- CodeRabbit reviewed head 3a3a966b (review 5559712088, COMMENTED): 1 actionable Major finding.
  Finding (valid, confirmed against code + runner.py:1933): an open queued on _open_gate while
  wait_devices_free timed out would acquire the gate after _abort_opens was cleared and open a
  camera after hand-over. Fix: _handover_gen counter, bumped in wait_devices_free's finally;
  open_camera records it before queuing and raises OpenCancelled if it changed on acquire.
  (A hand-over cannot return True while an open is queued: the queued open holds _opening > 0,
  so a changed generation always means a timed-out hand-over and a stale open.)
- New tests: test_queued_open_is_cancelled_if_handover_finished_while_queued (PROVEN to fail
  without the fix), test_queued_open_proceeds_when_no_handover_ran,
  test_handover_generation_increments_by_exactly_one.
- Mutation re-runs: 91.9% (34/37, 3 counter mutants survived) -> added the increment test ->
  re-running for 100%. Commits: dcfadf1 (fix), +1 (test).
- NOTE: venv python resolves `away` to away-b23's src via .pth; manual pytest runs MUST use
  PYTHONPATH=src. (testteeth inserts the target's src itself, which is why its runs were fine.)

## CodeRabbit follow-up finding (2026-10-17 ~20:45 UTC, review 5560000072 on head 13e08241)
- Finding (Minor, valid): the gate-acquisition generation check runs once; a hand-over finishing
  after the open takes the gate but before _open_camera's first _check() goes unseen (event
  cleared, generation not re-checked). Fix: thread queued_gen through _open_camera into _Search;
  cancelled() is now an instance method firing when _abort_opens is set OR _handover_gen changed.
  _open_broadcast inherits via parent.handover_gen; measure_fps callback keeps working.
- 3 new tests, all proven to fail without the fix. ruff/mypy clean. Mutation re-run in progress.
- NOTE: head changed -> CI re-runs and CodeRabbit must re-review the new head before merge.

# Fix A: camera open gate - state (HOLD as of 2026-10-17 ~20:36 PST)

Budget: run_id=away-fixA-camera-gate-01 (preserved).

## Branch and commit
- Branch: `fix/camera-open-gate` (worktree: ~/workspace/builders/away-fixA, tracks origin/main)
- Base: origin/main @ 347d51f82e8f5976360741889395640d0a5564c7 (Release v0.1.28)
- HEAD: 709d935f9d6db212bad7b637708b08158a2f30d0 ("test: strengthen camera gate tests to the mutation bar")
- Commits: a06ede2 (implementation + 2 regression tests), 709d935 (7 more tests for the mutation bar)
- Worktree clean, nothing uncommitted.

## What is done
1. Implementation in src/away/multicam/capture.py (only file touched besides the new test):
   - Module-level `_open_gate = threading.Lock()`.
   - `open_camera()` holds the gate for the whole `_open_camera()` call (counter `_opening` kept as is).
   - `wait_devices_free()` sets `_abort_opens`, then acquires the gate with the remaining
     timeout and releases it right away before the drain loop (no hold during drain:
     holding it would stall a concurrent open that already incremented `_opening`,
     and new opens during hand-over must keep failing fast on `_abort_opens`).
   - Verified no re-entrancy: `_open_camera` never calls `open_camera`;
     `wait_devices_free` is only called from runner threads, never while the gate is held.
2. Regression tests in tests/test_v0129_camera_open_gate.py (9 tests, all pass):
   - `test_concurrent_opens_never_overlap_inside_open_camera`: PROVEN to fail on unfixed
     code (assert 2 == 1 before the fix).
   - `test_wait_devices_free_aborts_an_in_progress_open`: abort-contract test (passes
     on unfixed code too; pins the hand-over interaction).
   - 7 more: open_camera default forwarding, wait_devices_free defaults (15.0/0.05),
     gate-wait timeout gives up on a stuck open, remaining timeout is used, no gate
     touch when no time left, drain loop stops at deadline, held-camera log format.
3. Gates run locally:
   - `ruff check`: clean. `ruff format`: applied (write mode) to both files.
   - `mypy src/away/multicam/capture.py`: "Success: no issues found" (via ~/.venvs/away-b23).
   - Targeted camera tests (new + test_v0115/v0116/v0119/test_capture_history): 152 passed, 1 skipped.
   - Full suite: all pass EXCEPT 2 pre-existing/environmental failures, both proven
     unrelated by running them on base code (347d51f8):
     * tests/test_iphone.py::test_receiver_real_udp_loopback: PermissionError on UDP
       in the Linux sandbox (same failure documented as environmental in the 0.1.28 release).
     * tests/test_safety_hardening.py::test_a_body_bigger_than_its_content_length_is_cut_off_at_the_asgi_level:
       ModuleNotFoundError: No module named 'anyio' (missing dep in this venv).
     * tests/test_paused_banner.py flaked once under full-suite load; passes 3/3 in
       isolation with the fix and does not touch capture.py.
4. Mutation gate, first run: 65.6% (21/32 killed, FAIL under 80). The 7 new tests were
   written to kill all 11 survivors. Second mutation run was STARTED but KILLED on HOLD
   before finishing; score after strengthening is UNKNOWN.

## What is in progress
- Second `testteeth run --base 347d51f8 --fail-under 80` (scoped to the 5 camera test
  files, venv python): killed on HOLD, no result.

## Exact next step (when resumed)
1. Re-run the mutation gate:
   `cd ~/workspace/builders/away-fixA && PYTHONPATH=$HOME/workspace/testteeth/src ~/.venvs/away-b23/bin/python -m testteeth.cli run --base 347d51f82e8f5976360741889395640d0a5564c7 --fail-under 80 --pytest-args "tests/test_v0129_camera_open_gate.py tests/test_v0115_cameras.py tests/test_v0116_fixes.py tests/test_v0119_winrt_camera.py tests/test_capture_history.py -p no:warnings -q" src/away/multicam/capture.py`
   (Note: plain `~/workspace/bin/testteeth` uses system python3 with wrong deps
   (mcp 2.x); must run the module with the venv python. Baseline needs the
   environmental deselects only when running the FULL suite; the scoped run above
   avoids them.)
2. If score >= 80: push via `~/workspace/github-tools/replicate_branch.py
   fernandogarzaaa/project-away fix/camera-open-gate 347d51f82e8f5976360741889395640d0a5564c7 347d51f82e8f5976360741889395640d0a5564c7:<tip SHA>`,
   open the PR via the gh-api Python helper (title "Fix: serialize camera opens with
   a process-wide gate", no em dashes in body), post `@coderabbitai review`, monitor
   checks to green. Never merge (needs Inan's go).
3. If score < 80: run `testteeth suggest`, strengthen tests, repeat.

## Blockers / risks
- None blocking. Open risk: true native-crash verification needs the real Acer
  (Windows hardware); unit tests prove serialization only.
- CodeRabbit free-OSS quota was reported exhausted repo-wide earlier; the review
  trigger may not produce a review.
- The 2 full-suite failures above are environmental; CI (with full deps) should be green.
