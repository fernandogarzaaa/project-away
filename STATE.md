# Fix C (COM hardening) - paused state (HOLD issued 2026-10-17)

## Branch / worktree
- Worktree: `~/workspace/builders/away-fixC` (branch `fix/com-hardening`, tracking `origin/main`)
- Base: `origin/main` at `347d51f82e8f5976360741889395640d0a5564c7` (Release v0.1.28)
- Changes are UNCOMMITTED. Nothing pushed. No PR opened.

## Files touched
- `src/away/pointer/uia.py` (modified): `_com_error_types()` helper; every use of the shared
  COM objects now under `self._lock` (`_ensure` requires caller-held lock; `thread_init`
  takes it); `_drop_com_objects()` resets `_uia/_cond/_cache` on COMError/OSError and logs
  one warning; `foreground_targets()` returns `[]` and `focus()` returns `False` on disconnect
  so the next call re-creates.
- `src/away/voice/other_audio.py` (modified): `pycaw_reader().read()` now guards
  `AudioUtilities.GetAllSessions()` itself, returning `[]` on failure.
- `src/away/app/tray.py` (modified): new `menu_safe(name)` decorator; applied to all
  click-action menu callbacks in `main()` (pause, dictation, mic, settings, edit config,
  quit, install/check/run update) and `comfort_menu_items` (attention gate, guided first
  run, calibrate tracking) plus the inline lambdas (retry/turn-off cameras, shortcut
  record/stop, stop everything, disconnect agents, reset cursor, check PC, self-test).
  `live_menu_items` (from `away.liveagent.integrate`) was NOT touched (not owned by this fix).
- `tests/test_v0129_com_hardening.py` (new, untracked): 7 tests.

## Done (verified with raw output)
- Regression tests written FIRST; 6 of 7 FAIL on unfixed code (disconnects propagate,
  overlap detected, AttributeError for menu_safe). The 7th
  (`test_sample_records_reader_error_instead_of_raising`) is a characterization test of
  pre-existing behavior and passes both ways.
- All 7 pass on fixed code.
- `ruff check` and `ruff format` clean on all 4 files.
- `mypy` (repo config, CI invocation): "Success: no issues found in 268 source files".
- Targeted existing tests pass: test_v0120_smart_pointer.py, test_v0122_realpc.py,
  test_app.py, test_v0123_features.py.
- Full suite run #1 (away-b27 venv, has anyio): completed, exit 0, no FAILED lines
  (summary line was lost in the pipe; treat as provisional).
- Full suite run #2 (away-b23 venv, lacks anyio): 1 pre-existing env failure only:
  `test_iphone.py::test_receiver_real_udp_loopback` (PermissionError on UDP in sandbox,
  same as the documented 0.1.28 release env failure) and
  `test_safety_hardening.py::test_a_body_bigger_than_its_content_length_is_cut_off_at_the_asgi_level`
  (ModuleNotFoundError: anyio, venv limitation only; CI installs full deps).

## In progress (background processes left running at HOLD)
- Full suite re-run with `-v` to a log (`/tmp/fixC_full_suite3.log`), session
  `proc_978ca3c6b1d1`: was at ~37% (`tests/test_studio_learning.py`, slow training tests
  under heavy system load, NOT a hang/failure). Check `tail /tmp/fixC_full_suite3.log`.
- Mutation gate (`/tmp/fixC_testteeth.log`), session `proc_4fe8ce3e9231`:
  `testteeth run --base 347d51f82e8f5976360741889395640d0a5564c7 --fail-under 80`
  scoped to the 3 source files, pytest-args limited to the 7 relevant test files,
  invoked via the away-b27 venv python so its subprocesses use the right interpreter.
  Was in baseline phase. Check `tr '\r' '\n' < /tmp/fixC_testteeth.log | tail`.

## Mutation gate result (2026-10-17, run finished during HOLD)
- Score: **31.8% (FAIL, need >= 80)**. Log: `/tmp/fixC_testteeth.log`.
- `Mutants: 110 killed: 35 survived: 3 not covered: 72 timeouts: 0`
- Dominant problem: `tray.py::main` and its nested callbacks (58 mutants, all
  `[not covered]`): `main()` never executes in tests (pragma no cover, needs a desktop),
  and testteeth grades whole changed functions, so every mutant in `main` counts as
  survived. The `@menu_safe` decorator lines and wrapped lambdas I added live there.
- `menu_safe`/`deco`/`wrapper` themselves: 100% (4/4 killed). `comfort_menu_items` and
  `on_gate`: 100%.
- Other gaps to close: `uia._com_error_types` ImportError branch never executed (block the
  comtypes import in a test via `sys.modules["comtypes"] = None`); `other_audio.read`
  success path with fake sessions (covers L53-L60); `focus` survived mutants
  (`return False -> return None` for ref=None needs a None-ref test;
  `remove-call self._ensure()` needs an assertion that objects were created, e.g.
  `len(created) == 1` after a successful focus); `foreground_targets` two uncovered
  `return []` early exits (add hwnd=0 / root=None cases to the new test file).
- Resume strategy for 80: extract the menu callbacks out of `main()` into module-level
  (or a small callbacks object) so they are directly callable in tests, then add tests
  invoking each with fakes (assert the underlying action ran; assert exceptions are
  swallowed and logged). That converts the ~58 not-covered tray mutants into covered
  ones. Then re-run testteeth until >= 80.
- Do NOT try to game the gate by narrowing scope; the callbacks are touched code and
  must be genuinely tested.

## Exact next step (on resume)
1. Confirm full suite green (excluding the known UDP-sandbox env test).
2. Strengthen tests per the mutation gaps above; re-run testteeth until >= 80
   (`testteeth suggest` prints per-mutant briefs).
3. Commit (message: Fix C summary, no em dashes), push via
   `~/workspace/github-tools/replicate_branch.py fernandogarzaaa/project-away fix/com-hardening <base SHA> <base SHA>:<tip SHA>`.
4. Open PR via the gh-api python snippet (title: "Fix: harden COM usage (UIA lock,
   re-create on disconnect, guarded audio sessions)"; body covers log evidence, gaps,
   changes, tests, and the stated non-goal: a true native SEH fault cannot be caught
   from Python).
5. Trigger CodeRabbit (`@coderabbitai review` comment); note if quota exhausted.
6. Monitor CI to green; fix failures; NEVER merge (needs Inan's explicit go).

## Full suite result (2026-10-17, received during HOLD)
**GREEN**: `3016 passed, 19 skipped, 1 deselected in 659.52s`, exit 0 (away-b27 venv).
The deselected test is the known UDP-sandbox env failure
(`test_iphone.py::test_receiver_real_udp_loopback`). Log: `/tmp/fixC_full_suite3.log`.
The earlier stall at 37% was slow training tests under heavy system load, not a hang.

## Blockers / notes
- No code blockers. System load was very high (sibling Fix A/B agents running suites);
  that is why the suite was slow, not a code problem.
- mypy is not installed for system python3; use
  `/home/hatch/workspace/builders/away-b27/.venv/bin/mypy` (read-only use of the binary;
  do not touch `~/workspace/builders/away-b27/` otherwise).
- The away-b23 venv lacks `anyio`; prefer the away-b27 venv python for test runs.
- Known env-only failure (do not chase): `test_iphone.py::test_receiver_real_udp_loopback`.
