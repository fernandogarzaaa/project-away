"""Live agent: a real-time assistant inside Project Away.

Say "Hey <name>, ..." and it answers out loud in about a second, does small jobs on this PC through Agent Link's
own tools (in-process, with the same approvals and safety rules), and hands bigger jobs to your other AI agents
through their inbox. It runs on cloud models (free ones through opencode, or your own API keys), so it adds almost
no load to the laptop. It is OFF until you turn it on in Settings > Live agent.

Kept in its own package on purpose: the rest of the app only calls :func:`away.liveagent.integrate.attach`, the
Settings tab builder and the tray menu helper.

Modules:

* :mod:`.settings`   - what the person chose (``<data>/liveagent/settings.json``; API keys live in the OS keyring)
* :mod:`.keys`       - API keys in Windows Credential Manager / macOS Keychain / Secret Service (``keyring``)
* :mod:`.http`       - one tiny HTTP client (``User-Agent: ProjectAway/<version>``, JSON + server-sent events)
* :mod:`.catalog`    - known providers, free-key links, model facts (vision, reasoning, keeps prompts)
* :mod:`.providers`  - OpenAI-compatible, Anthropic, opencode (free models), fakes for tests
* :mod:`.opencode`   - find or download the opencode engine, run a private ``opencode serve``
* :mod:`.jev`        - Jev "System One" decisions (risk gate, element choice, success check, intent routing)
* :mod:`.router`     - which model for which job, the fallback funnel, circuit breakers
* :mod:`.ledger`     - daily cost ledger with a budget stop (paid keys)
* :mod:`.privacy`    - redaction + what may go to providers that may keep prompts
* :mod:`.untrusted`  - fencing for screen / UI / clipboard text put into prompts (prompt-injection defence)
* :mod:`.textstream` - strip reasoning traces, split replies into sentences for speech
* :mod:`.agent`      - the turn loop (chat, computer use, hand-off)
* :mod:`.speech`     - sentence-by-sentence speech with barge-in
* :mod:`.integrate`  - wiring into the running app (voice, Agent Link, tray, updates)
* :mod:`.gui`        - Settings > Live agent

Parts are ported, with attribution, from the author's own MIT projects: GHOST-Chimera (free_router, cost_monitor,
redaction, untrusted fencing) and ape-mcp's freepool (cost ladder, failure classes, Retry-After parsing).
"""

from __future__ import annotations

__all__: list[str] = []
