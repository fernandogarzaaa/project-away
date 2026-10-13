"""Agent Link (v0.1.20): the bridge between you and your AI agents.

Any AI agent that speaks MCP (Claude Desktop, Claude.ai, ChatGPT connectors, Cursor, the OpenAI Agents SDK,
Gemini CLI, a local agent...) or plain HTTP/JSON can connect to the running app, see what you allow it to see,
ask you questions you answer with a gesture or your voice, tune the app to your preferences, and, only when
you allow it, use your computer while you watch.

Layout (each module is importable without a camera, a display or the ``mcp`` package unless it says so):

* :mod:`perms` - the permissions ("See my gestures", "See my camera when I allow"...) and their scope names
* :mod:`auth` - agents, tokens, pairing codes, constant-time checks
* :mod:`ratelimit` - token buckets for requests, failed sign-ins and computer actions
* :mod:`safepaths` - the only files an agent can download (bundles, recordings, logs) and nothing else
* :mod:`telemetry` - fps, latency, CPU/GPU, landmarks, gestures, log tail, live context (presence/attention)
* :mod:`recording` / :mod:`replay` - compact landmark recordings and the headless replay
* :mod:`handsigns` - thumbs up/down, finger count and open palm from landmarks (answers to prompts)
* :mod:`prompts` - ask_user / notify_user / connection approvals, queued until you are paying attention
* :mod:`registry` / :mod:`changes` / :mod:`prefs` - preference-level self-modification with history and undo
* :mod:`control` - "Let it use my computer": gating, takeover pause, risky-action confirmation, action log
* :mod:`service` - the transport-agnostic core; :mod:`api` (REST + MCP over HTTP) and :mod:`bridge` (stdio)
* :mod:`tunnel` - cloudflared quick tunnels (pinned download), :mod:`clients` - Claude Desktop / Cursor config
* :mod:`host` - wires it all into the running app; :mod:`gui` - Settings > Agent Link and the connect wizard
"""

from __future__ import annotations

__all__: list[str] = []
