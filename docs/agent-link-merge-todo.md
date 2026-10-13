# Agent Link: TODO when the other v0.1.20 branches merge

Agent Link (feat/remote-link) is built so other branches plug in without touching its core:
`away.agentlink.registry.register(Domain(...))` adds a kind of setting agents can list/change (validated, undoable),
and `Telemetry.extensions["name"] = callable` adds live context.

- [ ] **feat/gesture-studio**: register a `gestures` domain (per-gesture enable / threshold / cooldown); then retire
      the global `custom_gestures` threshold/cooldown ops in `registry.custom_apply`. Feed Studio's "that was wrong"
      button into `feedback.wrong` on the bus (the misfire detector and agents already listen for it).
- [ ] **feat/macros**: replace the `macros` placeholder in `registry.domains()` with a real domain (list / get /
      create / delete shortcuts); keep `shell`/`launch` refusals from `_check_action`.
- [ ] **feat/smart-pointer**: reuse its UIA matcher in `control.ComputerControl._point` / `desktop.best_match`
      (marked `TODO(merge feat/smart-pointer)` in `service.act`), and register its tunables.
- [ ] **feat/attention (comfort / fatigue)**: set `host.core.telemetry.extensions["fatigue"] = <fn returning the
      fatigue mode>`; `get_context` / `away://about` then report it instead of `null`. Register its settings.
- [ ] Blocked camera zones ("ignore my hand in this corner") need a pipeline hook first; then add a `zones` op.
- [ ] Settings window: each branch adds a tab, so `tests/test_settings_layout.py` / `tests/test_tuning_tab.py`
      tab counts (`len(w._scrollers)`) need the sum of all new tabs.
- [ ] `config.AppConfig`: several branches add sections; resolve the dataclass field order and `CONFIG_VERSION` once.
