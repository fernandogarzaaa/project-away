"""The logic behind Settings > Agent Link and the "Connect an AI agent" wizard, without Tk (tested headless).

Plain language only: the user sees "See my screen when I allow", never "scope"; "private link", never "tunnel";
errors come with one fix button ("Claude isn't installed." -> "Get it").
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from away.agentlink import clients, perms
from away.agentlink import host as hostmod

TILES = ("claude_desktop", "chatgpt", "cursor", "claude_ai", "other")


@dataclass
class Problem:
    text: str
    fix_label: str = ""
    fix: str = ""  # "url:https://..." | "start_app" | "retry"


EVERYONE = "Every connected agent"  # v0.1.22: the default-agent menu's "no default" choice


class AgentLinkModel:
    def __init__(
        self,
        data_dir: Path | None = None,
        send: Callable[..., str] | None = None,
        state: Callable[[], dict[str, Any]] | None = None,
        alive: Callable[[], Any] | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.send = send or (lambda action, **kw: hostmod.send_request(action, data_dir, **kw))
        self.state = state or (lambda: hostmod.read_state(data_dir))
        self.alive = alive or (lambda: hostmod.host_alive(data_dir))
        self.config_path = config_path

    # ------------------------------------------------------------------ status
    def status_line(self, st: dict[str, Any] | None = None) -> tuple[str, bool]:
        if not self.alive():
            return "Project Away isn't running", False
        st = st if st is not None else self.state()
        connected = [a for a in st.get("agents") or [] if a.get("connected")]
        if connected:
            names = ", ".join(sorted({a["name"] for a in connected}))
            return f"Connected to {names} \u2713", True
        if st.get("agents"):
            return "Your agents are allowed but not connected right now", False
        return "No AI agent connected yet", False

    def agents(self) -> list[dict[str, Any]]:
        return list(self.state().get("agents") or [])

    def wait_result(
        self, rid: str, timeout: float = 90.0, sleep: Callable[[float], None] = time.sleep
    ) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            res = (self.state().get("results") or {}).get(rid)
            if res is not None:
                return dict(res)
            sleep(0.3)
        return {"ok": False, "error": "Project Away didn't answer. Is it running?"}

    def not_running(self) -> Problem | None:
        if self.alive():
            return None
        return Problem("Project Away isn't running.", "Start it", "start_app")

    # ------------------------------------------------------------------ wizard
    @staticmethod
    def tiles() -> list[dict[str, Any]]:
        out = []
        for key in TILES:
            app = clients.APPS[key]
            out.append(
                {
                    "key": key,
                    "name": app.name,
                    "local": app.local,
                    "installed": clients.installed(key) if app.local else None,
                }
            )
        return out

    @staticmethod
    def permission_rows(advanced: bool = False) -> list[perms.Permission]:
        return [p for p in perms.PERMISSIONS if p.advanced == advanced]

    @staticmethod
    def default_scopes() -> list[str]:
        return [p.scope for p in perms.PERMISSIONS if p.default]

    def local_plan(self, key: str, scopes: list[str]) -> tuple[Problem | None, str, Path | None]:
        """(problem, preview text, config path) for connecting a local app."""
        app = clients.APPS[key]
        if not clients.installed(key):
            short = app.name.split()[0]
            return Problem(f"{short} isn't installed.", "Get it", f"url:{app.download}"), "", None
        path = clients.config_path(key)
        assert path is not None
        try:
            _new, old = clients.plan(path, clients.server_entry(perms.to_list(perms.normalize(scopes))))
        except ValueError as exc:
            return (
                Problem(f"{app.name}'s settings file can't be read ({exc}).", "Show the file", f"url:{path.as_uri()}"),
                "",
                None,
            )
        what = "add Project Away to" if old is None or "project-away" not in old else "update Project Away in"
        preview = f"We'll {what} {app.name}'s settings. A backup copy is kept."
        return None, preview, path

    def connect_local(self, key: str, scopes: list[str]) -> tuple[Problem | None, str]:
        problem, _preview, path = self.local_plan(key, scopes)
        if problem is not None or path is None:
            return problem, ""
        clients.write(path, clients.server_entry(perms.to_list(perms.normalize(scopes))))
        if self.alive():
            self.send("serve")
        app = clients.APPS[key]
        return None, f"Done! {app.steps} You'll be asked to Allow it here the first time."

    def start_remote(self, scopes: list[str], relay: str | None = None) -> tuple[Problem | None, dict[str, Any] | None]:
        p = self.not_running()
        if p is not None:
            return p, None
        rid = self.send(
            "pair",
            scopes=perms.to_list(perms.normalize(scopes)),
            remote=True,
            download=True,
            relay=relay or self.link_type(),
        )
        res = self.wait_result(rid, 120.0)
        if not res.get("ok"):
            return Problem(str(res.get("error") or "Something went wrong."), "Try again", "retry"), None
        return None, dict(res["result"])

    @staticmethod
    def remote_steps(key: str) -> list[tuple[str, str]]:
        """(app name, one line) - the chosen app first, then the others."""
        order = [key] + [k for k in ("chatgpt", "claude_ai", "other") if k != key]
        return [(clients.APPS[k].name, clients.APPS[k].steps) for k in order if not clients.APPS[k].local]

    # ------------------------------------------------------------------ actions
    def disconnect_all(self) -> str:
        if not self.alive():
            return "Nothing is connected."
        res = self.wait_result(self.send("disconnect_all"), 15.0)
        n = (res.get("result") or {}).get("disconnected", 0)
        return f"Disconnected {n} agent{'s' if n != 1 else ''}."

    def revoke(self, agent_id: str) -> None:
        self.send("revoke", agent_id=agent_id)

    def set_scopes(self, agent_id: str, scopes: list[str]) -> None:
        self.send("set_scopes", agent_id=agent_id, scopes=perms.to_list(perms.normalize(scopes)))

    # ------------------------------------------------- agent tokens (v0.1.24)
    @staticmethod
    def _ago(t: float, now: float) -> str:
        d = now - t
        if d < 120:
            return "just now"
        if d < 7200:
            return f"{int(d // 60)} minutes ago"
        if d < 2 * 86400:
            return f"{int(d // 3600)} hours ago"
        return time.strftime("%b %d", time.localtime(t))

    @classmethod
    def token_rows(cls, st: dict[str, Any], now: float | None = None) -> list[dict[str, Any]]:
        """Settings > Agent Link > "Agent tokens": one row per agent that stays connected, in plain words."""
        now = time.time() if now is None else now
        rows = []
        for t in st.get("tokens") or []:
            used = "not used yet"
            if t.get("last_used"):
                used = f"last used {cls._ago(float(t['last_used']), now)}"
                if t.get("last_source"):
                    used += f" from {t['last_source']}"
                if t.get("last_host") and t["last_host"] not in ("127.0.0.1", "localhost", "::1"):
                    used += f" (through {t['last_host']})"
            ends = (
                "Stays connected until you revoke it"
                if not t.get("expires")
                else f"Stays connected until {time.strftime('%b %d, %Y', time.localtime(float(t['expires'])))}"
            )
            old = ""
            if t.get("old_key_until"):
                old = f"The old token still works until {time.strftime('%H:%M', time.localtime(t['old_key_until']))}"
            rows.append(
                {
                    "id": t["id"],
                    "agent_id": t.get("agent_id", ""),
                    "name": t.get("agent") or "AI agent",
                    "connected": bool(t.get("connected")),
                    "scopes": list(t.get("scopes") or []),
                    "can": "Can: " + (", ".join(t.get("permissions") or []) or "nothing yet"),
                    "when": f"Added {time.strftime('%b %d', time.localtime(float(t.get('created') or now)))}; {used}",
                    "ends": ends,
                    "old_key": old,
                }
            )
        return rows

    def revoke_token(self, token_id: str) -> str:
        """Revoke: it stops working at once and the agent is disconnected."""
        p = self.not_running()
        if p is not None:
            return p.text
        res = self.wait_result(self.send("token_revoke", token_id=token_id), 15.0)
        if res.get("ok") and (res.get("result") or {}).get("revoked"):
            return "Revoked. It is disconnected now; it can only come back if you allow it again."
        return str(res.get("error") or "That agent was already disconnected.")

    def rotate_token(self, token_id: str) -> tuple[Problem | None, str | None]:
        """A replacement token, shown once (the old one keeps working for an hour)."""
        p = self.not_running()
        if p is not None:
            return p, None
        res = self.wait_result(self.send("token_rotate", token_id=token_id), 15.0)
        if not res.get("ok"):
            return Problem(str(res.get("error") or "Something went wrong."), "Try again", "retry"), None
        secret = hostmod.take_secret(str((res.get("result") or {}).get("secret_ref") or ""), self.data_dir)
        if not secret:
            return Problem("The new token could not be shown. Please try again.", "Try again", "retry"), None
        return None, secret

    # "New agent tokens last": the default lifetime of a token issued when you choose Keep this agent connected
    EXPIRY_CHOICES = (("90d", "90 days"), ("30d", "30 days"), ("never", "Until I revoke it"))

    def token_expiry(self) -> str:
        """``agent_link.token_expiry`` (90 days unless you changed it)."""
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        try:
            cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
            return str(cfg.agent_link.token_expiry or "90d")
        except Exception:
            return "90d"

    def set_token_expiry(self, choice: str) -> None:
        """Accepts the key ("90d") or its label ("90 days"); tokens already given keep their own end date."""
        from away import config as cfgmod
        from away.agentlink.auth import expiry_choice

        by_label = {label: key for key, label in self.EXPIRY_CHOICES}
        value = expiry_choice(by_label.get(choice, choice))
        path = self.config_path or cfgmod.config_path()
        cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
        cfg.agent_link.token_expiry = value
        cfgmod.save(cfg, path)

    def set_token_scopes(self, token_id: str, scopes: list[str]) -> None:
        self.send("token_scopes", token_id=token_id, scopes=perms.to_list(perms.normalize(scopes)))

    def answer(self, prompt_id: str, answer: str) -> None:
        self.send("answer", prompt_id=prompt_id, answer=answer)

    def undo_change(self, batch: str) -> None:
        self.send("undo_change", batch=batch)

    def undo_agent_today(self, agent_id: str) -> None:
        self.send("undo_agent_today", agent_id=agent_id)

    def stop_control(self) -> None:
        self.send("control_stop")

    def undo_action(self) -> None:
        self.send("control_undo")

    def want_health(self) -> None:
        if self.alive():
            self.send("want_health")

    def bundle(self) -> str:
        res = self.wait_result(self.send("bundle"), 60.0)
        return f"Saved {res['result']['path']}" if res.get("ok") else str(res.get("error"))

    def new_link(self, relay: str | None = None) -> tuple[Problem | None, dict[str, Any] | None]:
        """v0.1.22: a fresh secure link (new web address); nobody is disconnected. v0.1.23: of the chosen type."""
        p = self.not_running()
        if p is not None:
            return p, None
        res = self.wait_result(self.send("new_link", relay=relay or self.link_type()), 120.0)
        if not res.get("ok"):
            return Problem(str(res.get("error") or "Something went wrong."), "Try again", "retry"), None
        return None, dict(res["result"])

    def backup_link(self) -> tuple[Problem | None, dict[str, Any] | None]:
        """v0.1.23 "Blocked? Try the backup link": a link that does not go through Cloudflare."""
        p = self.not_running()
        if p is not None:
            return p, None
        res = self.wait_result(self.send("backup_link"), 120.0)
        if not res.get("ok"):
            return Problem(str(res.get("error") or "Something went wrong."), "Try again", "retry"), None
        return None, dict(res["result"])

    def link_type(self) -> str:
        """v0.1.23: "standard" (Cloudflare) or "backup" (SSH relay), saved in agent_link.link_type."""
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        try:
            cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
            return str(cfg.agent_link.link_type or "standard")
        except Exception:
            return "standard"

    def set_link_type(self, kind: str) -> None:
        from away import config as cfgmod

        if kind not in ("standard", "backup"):
            raise ValueError("link type must be standard or backup")
        path = self.config_path or cfgmod.config_path()
        cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
        cfg.agent_link.link_type = kind
        cfgmod.save(cfg, path)

    @staticmethod
    def inbox_lines(st: dict[str, Any]) -> list[str]:
        """v0.1.23: unread messages per agent, and messages kept for agents whose link ended."""
        out = []
        for a in st.get("agents") or []:
            n = int(a.get("unread") or 0)
            if n:
                out.append(f"{a['name']}: {n} unread message{'s' if n != 1 else ''}")
        for w in st.get("waiting") or []:
            n = int(w.get("unread") or 0)
            out.append(
                f"{w['name']}: {n} message{'s' if n != 1 else ''} waiting (not connected; pair it again under the "
                "same name to deliver them)"
            )
        return out

    def set_talk(self, agent_id: str, wake_word: str | None = None, wake_url: str | None = None) -> str:
        """v0.1.22: an agent's own wake word and "wake me" address. Returns "" or a plain error."""
        from away.agentlink.inbox import check_wake_url

        data: dict[str, Any] = {"agent_id": agent_id}
        if wake_word is not None:
            data["wake_word"] = wake_word
        if wake_url is not None:
            try:
                data["wake_url"] = check_wake_url(wake_url)
            except ValueError as exc:
                return str(exc)
        if not self.alive():
            return "Project Away isn't running; start it and try again."
        res = self.wait_result(self.send("set_talk", **data), 15.0)
        return "" if res.get("ok") else str(res.get("error") or "Could not save.")

    def set_default_agent(self, name: str) -> None:
        """v0.1.22: who "Hey agent" and the summon gesture talk to ("" = every connected agent)."""
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
        cfg.agent_link.default_agent = str(name or "").strip()[:60]
        cfgmod.save(cfg, path)

    def set_talk_enabled(self, on: bool) -> None:
        """v0.1.22: listen for "Hey <agent>" at all (off: those words are typed like any other)."""
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
        cfg.agent_link.talk_enabled = bool(on)
        cfgmod.save(cfg, path)

    def control_budget(self) -> int:
        """v0.1.28: steps per approved "use my computer" session (agent_link.control_step_budget)."""
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        try:
            cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
            return max(1, int(cfg.agent_link.control_step_budget or 40))
        except Exception:
            return 40

    def set_control_budget(self, steps: int) -> int:
        """v0.1.28: change the step budget (1-1000); applies to the next approved session. Returns the saved value."""
        from away import config as cfgmod

        value = max(1, min(1000, int(steps)))
        path = self.config_path or cfgmod.config_path()
        cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
        cfg.agent_link.control_step_budget = value
        cfgmod.save(cfg, path)
        return value

    def audit_trail_on(self) -> bool:
        """v0.1.28: the per-session visual audit trail (agent_link.control_audit_trail), on by default."""
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        try:
            cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
            return bool(cfg.agent_link.control_audit_trail)
        except Exception:
            return True

    def set_audit_trail(self, on: bool) -> None:
        """v0.1.28: turn the visual audit trail on or off (no pictures kept and no session files when off)."""
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
        cfg.agent_link.control_audit_trail = bool(on)
        cfgmod.save(cfg, path)

    def talk_enabled(self) -> bool:
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        try:
            cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
            return bool(cfg.agent_link.talk_enabled)
        except Exception:
            return True

    def default_agent(self) -> str:
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        try:
            cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
            return str(cfg.agent_link.default_agent or "")
        except Exception:
            return ""

    @staticmethod
    def default_choices(agents: list[dict[str, Any]]) -> list[str]:
        """The default-agent menu: "Every connected agent" first, then each agent's name."""
        names = [str(a.get("name", "")) for a in agents if a.get("name")]
        return [EVERYONE, *dict.fromkeys(names)]

    def set_autofix(self, mode: str) -> None:
        from away import config as cfgmod

        path = self.config_path or cfgmod.config_path()
        cfg, _ = cfgmod.load(path) if path.exists() else (cfgmod.AppConfig(), [])
        cfg.agent_link.autofix = mode
        cfgmod.save(cfg, path)

    @staticmethod
    def health_lines(h: dict[str, Any] | None) -> list[str]:
        if not h:
            return ["Waiting for numbers from the running app..."]
        out = []
        fps = h.get("fps") or {}
        out.append("Cameras: " + (", ".join(f"camera {k}: {v:g} fps" for k, v in fps.items()) or "none running"))
        lat = h.get("latency") or {}
        if lat.get("p50_ms") is not None:
            out.append(f"Delay from camera to action: {lat['p50_ms']:g} ms typical, {lat['p95_ms']:g} ms slowest")
        cpu = h.get("cpu") or {}
        out.append(f"Processor: Project Away {cpu.get('process_pct', '?')}%, whole PC {cpu.get('system_pct', '?')}%")
        gpu = h.get("gpu")
        if gpu:
            out.append(f"Graphics card: {gpu.get('util_pct'):g}% busy, {gpu.get('mem_used_mb'):g} MB memory used")
        models = h.get("models") or {}
        if models:
            out.append("Models: " + ", ".join(f"{k}: {v}" for k, v in models.items()))
        return out
