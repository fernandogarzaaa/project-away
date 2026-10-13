"""The AI apps the Connect wizard knows: find them on this PC, write their MCP config (with a backup and only
after the user confirmed the change), and the copy-paste steps for remote ones.

* Claude Desktop - ``claude_desktop_config.json`` (``%APPDATA%\\Claude`` / ``~/Library/Application Support/Claude``
  / ``~/.config/Claude``), ``mcpServers`` entry that runs ``away mcp`` (stdio).
* Cursor - ``~/.cursor/mcp.json``, same entry.
* ChatGPT, Claude.ai and anything else remote - the pairing link (a Streamable HTTP MCP URL) + the QR code.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SERVER_NAME = "project-away"


@dataclass(frozen=True)
class AgentApp:
    key: str
    name: str
    local: bool  # configured on this PC (stdio) rather than through the pairing link
    download: str
    steps: str  # one line, for the wizard


APPS: dict[str, AgentApp] = {
    "claude_desktop": AgentApp(
        "claude_desktop",
        "Claude Desktop",
        True,
        "https://claude.ai/download",
        'Restart Claude Desktop, then ask Claude: "What can you see in Project Away?"',
    ),
    "cursor": AgentApp(
        "cursor",
        "Cursor",
        True,
        "https://cursor.com/downloads",
        "Restart Cursor (or open Settings > MCP and press refresh), then ask in the chat.",
    ),
    "chatgpt": AgentApp(
        "chatgpt",
        "ChatGPT",
        False,
        "https://chatgpt.com",
        "Settings > Apps & Connectors > Advanced: turn on Developer mode, then Create > "
        "paste the link > Authentication: none > Create.",
    ),
    "claude_ai": AgentApp(
        "claude_ai",
        "Claude (web or phone)",
        False,
        "https://claude.ai",
        "On claude.ai: Settings > Connectors > Add custom connector > paste the link > Add.",
    ),
    "other": AgentApp(
        "other",
        "Other",
        False,
        "",
        "Use the link as a remote MCP server URL (Streamable HTTP, no sign-in needed). "
        "Agents without MCP can use the same link's /v1 API (see the README).",
    ),
}


def _home(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("AWAY_HOME_OVERRIDE") or Path.home())


def config_path(key: str, platform: str | None = None, env: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if env is None else env
    plat = platform or sys.platform
    home = _home(env)
    if key == "claude_desktop":
        if plat.startswith("win"):
            return Path(env.get("APPDATA") or home / "AppData" / "Roaming") / "Claude" / "claude_desktop_config.json"
        if plat == "darwin":
            return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        return Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "Claude" / "claude_desktop_config.json"
    if key == "cursor":
        return home / ".cursor" / "mcp.json"
    return None


def installed(key: str, platform: str | None = None, env: Mapping[str, str] | None = None) -> bool:
    """Best effort: the app's config folder or program exists."""
    env = os.environ if env is None else env
    plat = platform or sys.platform
    home = _home(env)
    p = config_path(key, plat, env)
    if p is not None and p.parent.is_dir():
        return True
    if key == "claude_desktop":
        cands = [
            Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local") / "AnthropicClaude",
            Path("/Applications/Claude.app"),
        ]
        return any(c.exists() for c in cands)
    if key == "cursor":
        cands = [
            Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local") / "Programs" / "cursor",
            Path("/Applications/Cursor.app"),
        ]
        return any(c.exists() for c in cands) or shutil.which("cursor") is not None
    return False


def server_entry(scopes: list[str], command: list[str] | None = None) -> dict[str, Any]:
    if command is None:
        from away.app.frozen import away_command

        command = away_command(["mcp"])
    return {"command": command[0], "args": [*command[1:], "--scopes", ",".join(scopes)]}


def plan(path: Path, entry: dict[str, Any]) -> tuple[str, str | None]:
    """(new file text, old file text or None) - shown to the user before anything is written."""
    old = path.read_text(encoding="utf-8") if path.exists() else None
    data: dict[str, Any] = {}
    if old and old.strip():
        loaded = json.loads(old)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} is not a JSON object; fix or remove it first")
        data = loaded
    servers = data.get("mcpServers")
    if servers is not None and not isinstance(servers, dict):
        raise ValueError(f"{path}: mcpServers is not an object")
    data["mcpServers"] = dict(servers or {})
    data["mcpServers"][SERVER_NAME] = entry
    return json.dumps(data, indent=2) + "\n", old


def write(path: Path, entry: dict[str, Any], clock: Any = time.time) -> Path | None:
    """Merge our entry into the app's config. Returns the backup path (None when there was no file)."""
    new, old = plan(path, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if old is not None:
        backup = path.with_name(path.name + f".bak-{time.strftime('%Y%m%d-%H%M%S', time.localtime(clock()))}")
        backup.write_text(old, encoding="utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, path)
    return backup


def remove(path: Path) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if SERVER_NAME not in (data.get("mcpServers") or {}):
        return False
    del data["mcpServers"][SERVER_NAME]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def snippets(scopes: list[str], link: str | None = None, command: list[str] | None = None) -> dict[str, str]:
    entry = server_entry(scopes, command)
    out = {
        "Claude Desktop": json.dumps({"mcpServers": {SERVER_NAME: entry}}, indent=2),
        "Cursor": json.dumps({"mcpServers": {SERVER_NAME: entry}}, indent=2),
        "Any MCP app (on this PC)": f"command: {entry['command']}\nargs: {' '.join(entry['args'])}\ntransport: stdio",
    }
    if link:
        out["Any MCP app (remote)"] = (
            f"URL: {link}\ntransport: Streamable HTTP\nauthentication: none (the link is the key)"
        )
    return out
