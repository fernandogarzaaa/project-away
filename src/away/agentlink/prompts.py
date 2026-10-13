"""Talking to the human: ask_user, notify_user and the "<agent> wants to connect" approvals.

A prompt is answered by whichever comes first:

* a click on the on-screen prompt (a small always-on-top window run by ``away agent-ui``),
* a hand sign held for ~0.6 s: thumbs up (yes / the first option), thumbs down (no / the last option), one to
  four fingers (option N), an open palm (option 5) - see :mod:`away.agentlink.handsigns`,
* a custom gesture whose name is one of the options,
* the voice: saying an option, "yes"/"no", "allow"/"deny" or "option two",
* or the timeout (the answer is then ``None``, method ``timeout``).

Prompts are queued and shown one at a time. ``urgency="high"`` shows at once; ``normal`` and ``low`` wait for a
polite moment (the user is present, facing the screen and not typing), ``normal`` for at most half its timeout,
``low`` until its timeout.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from away.agentlink.handsigns import SignHold, answer_for

URGENCIES = ("low", "normal", "high")
MAX_TIMEOUT_S = 300.0
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "first": 1, "second": 2, "third": 3, "fourth": 4}
YES_WORDS = ("yes", "yeah", "yep", "allow", "approve", "ok", "okay", "sure", "do it", "accept")
NO_WORDS = ("no", "nope", "deny", "don't", "dont", "reject", "cancel", "stop")


@dataclass
class Prompt:
    id: str
    title: str
    question: str
    options: list[str]
    details: list[str] = field(default_factory=list)
    agent: str = ""
    urgency: str = "normal"
    timeout_s: float = 60.0
    kind: str = "ask"  # ask | approve | confirm
    created: float = 0.0
    shown_at: float | None = None
    answer: str | None = None
    method: str = ""
    done: threading.Event = field(default_factory=threading.Event)

    def spec(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "question": self.question,
            "options": self.options,
            "details": self.details,
            "agent": self.agent,
            "timeout_s": self.timeout_s,
            "kind": self.kind,
        }


class PromptUI(Protocol):
    def show(self, prompt: Prompt) -> None: ...
    def poll(self, prompt: Prompt) -> str | None: ...
    def close(self, prompt: Prompt) -> None: ...
    def toast(self, text: str, level: str = "info", seconds: float = 6.0) -> None: ...


class NullUI:
    """No window (tests, no display): prompts are still answered by gesture or voice; ``clicks`` simulates one."""

    def __init__(self) -> None:
        self.shown: list[Prompt] = []
        self.toasts: list[tuple[str, str]] = []
        self.clicks: dict[str, str] = {}

    def show(self, prompt: Prompt) -> None:
        self.shown.append(prompt)

    def poll(self, prompt: Prompt) -> str | None:
        return self.clicks.pop(prompt.id, None)

    def close(self, prompt: Prompt) -> None:
        pass

    def toast(self, text: str, level: str = "info", seconds: float = 6.0) -> None:
        self.toasts.append((level, text))


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", str(text).lower()).strip()


def voice_answer(text: str, options: list[str]) -> str | None:
    t = normalize(text)
    if not t:
        return None
    low = [normalize(o) for o in options]
    for i, o in enumerate(low):
        if o and (t == o or t.endswith(" " + o) or t.startswith(o + " ")):
            return options[i]
    m = re.search(r"\b(?:option|number)\s+(\w+)", t)
    if m:
        w = m.group(1)
        n = int(w) if w.isdigit() else _WORDS.get(w, 0)
        if 1 <= n <= len(options):
            return options[n - 1]
    words = t.split()
    if any(w in YES_WORDS for w in words) or t in YES_WORDS:
        return answer_for("thumbs_up", options)
    if any(w in NO_WORDS for w in words) or t in NO_WORDS:
        return answer_for("thumbs_down", options)
    return None


class PromptManager:
    def __init__(
        self,
        ui: PromptUI | None = None,
        polite: Callable[[], bool] = lambda: True,
        clock: Callable[[], float] = time.monotonic,
        tts: Callable[[str], None] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.ui: PromptUI = ui if ui is not None else NullUI()
        self.polite = polite
        self.clock = clock
        self.tts = tts
        self.on_event = on_event
        self.queue: list[Prompt] = []
        self.current: Prompt | None = None
        self.history: list[dict[str, Any]] = []
        self._hold = SignHold()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------- lifecycle
    def start(self, every_s: float = 0.1) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(every_s):
                with contextlib.suppress(Exception):
                    self.pump()

        self._thread = threading.Thread(target=loop, daemon=True, name="agentlink-prompts")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for p in [*self.queue, *([self.current] if self.current else [])]:
                self._finish(p, None, "cancelled")
            self.queue = []

    # ------------------------------------------------------------------- core
    def create(
        self,
        question: str,
        options: list[str] | None = None,
        *,
        title: str = "",
        agent: str = "",
        details: list[str] | None = None,
        timeout_s: float = 60.0,
        urgency: str = "normal",
        kind: str = "ask",
    ) -> Prompt:
        opts = [str(o).strip()[:60] for o in (options or ["Yes", "No"]) if str(o).strip()][:5] or ["Yes", "No"]
        if urgency not in URGENCIES:
            raise ValueError(f"urgency must be {', '.join(URGENCIES)}")
        p = Prompt(
            id=uuid.uuid4().hex[:10],
            title=str(title or (f"{agent} asks" if agent else "Your AI agent asks"))[:80],
            question=str(question).strip()[:500] or "?",
            options=opts,
            details=[str(d)[:200] for d in (details or [])][:12],
            agent=agent,
            urgency=urgency,
            timeout_s=max(3.0, min(float(timeout_s), MAX_TIMEOUT_S)),
            kind=kind,
            created=self.clock(),
        )
        with self._lock:
            self.queue.append(p)
        return p

    def ask(self, question: str, options: list[str] | None = None, **kw: Any) -> dict[str, Any]:
        """Show a prompt and wait for the answer (blocking). Use :meth:`create` + :meth:`wait` to not block."""
        p = self.create(question, options, **kw)
        return self.wait(p)

    def wait(self, p: Prompt) -> dict[str, Any]:
        if self._thread is None:  # no pump thread (tests): drive it here
            while not p.done.is_set():
                self.pump()
                if not p.done.is_set():
                    p.done.wait(0.01)
        else:
            p.done.wait(p.timeout_s + 5.0)
            if not p.done.is_set():
                self._finish(p, None, "timeout")
        return self.result(p)

    def result(self, p: Prompt) -> dict[str, Any]:
        waited = round(self.clock() - p.created, 2)
        return {"answer": p.answer, "method": p.method, "waited_s": waited, "question": p.question, "id": p.id}

    def _finish(self, p: Prompt, answer: str | None, method: str) -> None:
        with self._lock:
            if p.done.is_set():
                return
            p.answer, p.method = answer, method
            if p in self.queue:
                self.queue.remove(p)
            if self.current is p:
                self.current = None
                self._hold.reset()
                with contextlib.suppress(Exception):
                    self.ui.close(p)
            self.history.append(
                {
                    "t": time.time(),
                    "question": p.question,
                    "answer": answer,
                    "method": method,
                    "agent": p.agent,
                    "kind": p.kind,
                }
            )
            self.history = self.history[-100:]
            p.done.set()
        if self.on_event:
            self.on_event("prompt.answered", {"id": p.id, "answer": answer, "method": method, "kind": p.kind})

    def pump(self) -> None:
        now = self.clock()
        with self._lock:
            for p in list(self.queue):
                if now - p.created >= p.timeout_s:
                    self._finish(p, None, "timeout")
            cur = self.current
            if cur is not None:
                if now - cur.created >= cur.timeout_s:
                    self._finish(cur, None, "timeout")
                else:
                    try:
                        click = self.ui.poll(cur)
                    except Exception:
                        click = None
                    if click is not None and click in cur.options:
                        self._finish(cur, click, "click")
                return
            if not self.queue:
                return
            order = sorted(self.queue, key=lambda q: (-URGENCIES.index(q.urgency), q.created))
            nxt = order[0]
            waited = now - nxt.created
            ready = (
                nxt.urgency == "high"
                or self._safe_polite()
                or (nxt.urgency == "normal" and waited >= nxt.timeout_s / 2.0)
            )
            if not ready:
                return
            self.queue.remove(nxt)
            self.current = nxt
            nxt.shown_at = now
        with contextlib.suppress(Exception):
            self.ui.show(nxt)
        if self.on_event:
            self.on_event("prompt.shown", {"id": nxt.id, "kind": nxt.kind, "agent": nxt.agent})

    def _safe_polite(self) -> bool:
        try:
            return bool(self.polite())
        except Exception:
            return True

    # ---------------------------------------------------------------- answers
    def feed_sign(self, t: float, sign: str | None) -> bool:
        """Per hand frame: the current hand sign (or None). True when it answered the shown prompt."""
        cur = self.current
        if cur is None:
            return False
        held = self._hold.update(t, sign)
        if held is None:
            return False
        ans = answer_for(held, cur.options)
        if ans is None:
            return False
        self._finish(cur, ans, f"gesture:{held}")
        return True

    def feed_gesture(self, name: str) -> bool:
        cur = self.current
        if cur is None:
            return False
        for o in cur.options:
            if normalize(o).replace(" ", "_") == normalize(name).replace(" ", "_"):
                self._finish(cur, o, f"gesture:{name}")
                return True
        return False

    def feed_voice(self, text: str) -> bool:
        cur = self.current
        if cur is None:
            return False
        ans = voice_answer(text, cur.options)
        if ans is None:
            return False
        self._finish(cur, ans, "voice")
        return True

    def click(self, prompt_id: str, answer: str) -> bool:
        """An answer from the Settings window or the tray (same as clicking the prompt)."""
        with self._lock:
            p = self.current if self.current is not None and self.current.id == prompt_id else None
            if p is None:
                p = next((q for q in self.queue if q.id == prompt_id), None)
        if p is None or answer not in p.options:
            return False
        self._finish(p, answer, "click")
        return True

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            items = ([self.current] if self.current else []) + list(self.queue)
        return [dict(p.spec(), shown=p is self.current) for p in items]

    # ----------------------------------------------------------------- notify
    def notify(self, text: str, level: str = "info", speak: bool = False) -> dict[str, Any]:
        level = level if level in ("info", "success", "warning", "error") else "info"
        text = str(text).strip()[:300]
        with contextlib.suppress(Exception):
            self.ui.toast(text, level)
        spoken = False
        if speak and self.tts is not None:
            try:
                self.tts(text)
                spoken = True
            except Exception:
                spoken = False
        if self.on_event:
            self.on_event("notify", {"text": text, "level": level})
        return {"shown": True, "spoken": spoken}


class SubprocessUI:  # pragma: no cover - starts GUI processes; smoke-tested under Xvfb
    """Shows prompts and toasts with ``away agent-ui`` in its own process (Tk wants its own main thread and the
    tray already owns this one). Answers come back through ``<id>.answer.json``."""

    def __init__(self, folder: Path, command: Callable[[list[str]], list[str]] | None = None) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        if command is None:
            from away.app.frozen import away_command

            command = away_command
        self.command = command
        self.procs: dict[str, Any] = {}

    def _spawn(self, args: list[str]) -> Any:
        import subprocess
        import sys

        kw: dict[str, Any] = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000
        try:
            return subprocess.Popen(self.command(args), **kw)
        except Exception:
            return None

    def show(self, prompt: Prompt) -> None:
        spec = self.folder / f"{prompt.id}.json"
        spec.write_text(json.dumps(prompt.spec()), encoding="utf-8")
        self.procs[prompt.id] = self._spawn(["agent-ui", "prompt", str(spec)])

    def poll(self, prompt: Prompt) -> str | None:
        ans = self.folder / f"{prompt.id}.answer.json"
        if not ans.exists():
            return None
        try:
            data = json.loads(ans.read_text(encoding="utf-8"))
        except Exception:
            return None
        ans.unlink(missing_ok=True)
        return str(data.get("answer")) if data.get("answer") is not None else None

    def close(self, prompt: Prompt) -> None:
        (self.folder / f"{prompt.id}.done").write_text(prompt.method or "", encoding="utf-8")
        for suffix in (".json", ".answer.json"):
            (self.folder / f"{prompt.id}{suffix}").unlink(missing_ok=True)

    def toast(self, text: str, level: str = "info", seconds: float = 6.0) -> None:
        spec = self.folder / f"toast-{uuid.uuid4().hex[:8]}.json"
        spec.write_text(json.dumps({"text": text, "level": level, "seconds": seconds}), encoding="utf-8")
        self._spawn(["agent-ui", "toast", str(spec)])

    def flash(self, text: str) -> None:
        spec = self.folder / f"flash-{uuid.uuid4().hex[:8]}.json"
        spec.write_text(json.dumps({"text": text, "level": "camera", "seconds": 1.5}), encoding="utf-8")
        self._spawn(["agent-ui", "flash", str(spec)])

    def overlay(self, name: str, stop_file: Path, mode: str = "control") -> Any:
        spec = self.folder / f"overlay-{uuid.uuid4().hex[:8]}.json"
        spec.write_text(json.dumps({"agent": name, "stop_file": str(stop_file), "mode": mode}), encoding="utf-8")
        return self._spawn(["agent-ui", "control", str(spec)])
