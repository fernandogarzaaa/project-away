# Agent Link: integration with the other v0.1.20 features (done in the v0.1.20 merge)

Agent Link plugs other features in through `away.agentlink.registry` domains and `Telemetry.extensions`; the
glue lives in `away/agentlink/integrations.py` (tests: `tests/test_v0120_release_integration.py`).

- [x] **Gesture Studio**: a `gestures` domain (per gesture `enabled` / `threshold` / `hold_s` / `cooldown_s`, the
      same ranges as Studio's sliders); Studio's zone switches (`zones.enabled`, `zones.block_cursor`,
      `zones.anchor`) are agent settings. The global `custom_gestures` threshold/cooldown ops are **kept**: they
      drive the older recorded-custom-gesture model, which still runs next to Studio.
- [x] **Shortcuts (macros)**: a real `macros` domain (list with step descriptions only, create from steps,
      delete with a private undo backup in the shortcut folder; typed text never reaches an agent or the change
      history). `macro:` / `macro_control:` are allowed binding actions; `shell` / program `launch` stay refused.
- [x] **Smart pointer**: agent computer use finds elements through `desktop.SmartPointerUIA` (the smart pointer's
      fast UIA cache + fuzzy spoken-name matcher, falling back to the full `uiautomation` walk); `best_match`
      falls back to the same fuzzy score. Its switches/sliders are agent settings.
- [x] **Attention / comfort**: `telemetry.extensions["fatigue"]` (`ok` / `tired` / `head_control`) so
      `get_context` / `away://about` report it; also `attention_gate`, `smart_pointer`, `shortcuts`,
      `gesture_studio` extensions. Its switches and sliders are agent settings.
- [x] Settings window tab counts follow the notebook (`len(w._scrollers) == len(nb.tabs())`).
- [x] `config.AppConfig`: all sections merged; `CONFIG_VERSION` unchanged (new sections are additive).
- [ ] Not done: an agent op that edits Studio's zone *maps* (draw "ignore here" rectangles), and feeding
      Studio's "That was wrong" button (Settings process) into the tray app's `feedback.wrong` (needs IPC
      between the two processes; the voice command "that was wrong" already reaches agents).
