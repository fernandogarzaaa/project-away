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

    def start_remote(self, scopes: list[str]) -> tuple[Problem | None, dict[str, Any] | None]:
        p = self.not_running()
        if p is not None:
            return p, None
        rid = self.send("pair", scopes=perms.to_list(perms.normalize(scopes)), remote=True, download=True)
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
