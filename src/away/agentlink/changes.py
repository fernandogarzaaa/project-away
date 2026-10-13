"""Change history for everything agents (and auto-fix) change: who, why, before/after, undo.

* Every batch first copies the config file to ``agentlink/snapshots/<time>-<batch>.yaml`` (the last 50 are kept).
* Every operation is one history entry with a plain-language line ("Cursor speed: 2.2 -> 1.8"), who made it and
  why, and the inverse operations that undo it.
* :meth:`ChangeManager.undo` undoes one entry, :meth:`undo_batch` a whole batch, :meth:`undo_agent_today`
  everything one agent changed today (newest first). Undo is itself recorded (``who = "undo"``).
"""

from __future__ import annotations

import contextlib
import json
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from away import config as cfgmod
from away.agentlink.registry import Applied, ChangeError, Op, PresetStore, apply_ops

KEEP_SNAPSHOTS = 50
KEEP_ENTRIES = 2000


@dataclass
class Entry:
    id: str
    batch: str
    ts: float
    who: str
    agent_id: str
    why: str
    text: str
    op: dict[str, Any]
    before: Any
    after: Any
    inverse: list[dict[str, Any]] = field(default_factory=list)
    undone: bool = False
    low_risk: bool = False


class ChangeManager:
    def __init__(
        self,
        folder: Path,
        config_path: Callable[[], Path] = cfgmod.config_path,
        on_applied: Callable[[Any], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.folder = Path(folder)
        self.config_path = config_path
        self.on_applied = on_applied
        self.clock = clock
        self.presets = PresetStore(self.folder / "presets")
        self._lock = threading.RLock()
        self.entries: list[Entry] = []
        self._load()

    @property
    def history_file(self) -> Path:
        return self.folder / "changes.jsonl"

    def _load(self) -> None:
        try:
            with self.history_file.open(encoding="utf-8") as fh:
                self.entries = [Entry(**json.loads(line)) for line in fh if line.strip()]
        except FileNotFoundError:
            self.entries = []
        except Exception:
            self.entries = []

    def _save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.entries = self.entries[-KEEP_ENTRIES:]
        tmp = self.history_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for e in self.entries:
                fh.write(json.dumps(asdict(e), default=str) + "\n")
        tmp.replace(self.history_file)

    def _snapshot(self, batch: str) -> Path | None:
        src = self.config_path()
        if not src.exists():
            return None
        snaps = self.folder / "snapshots"
        snaps.mkdir(parents=True, exist_ok=True)
        dst = snaps / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.clock()))}-{batch}.yaml"
        shutil.copy2(src, dst)
        for old in sorted(snaps.glob("*.yaml"))[:-KEEP_SNAPSHOTS]:
            old.unlink(missing_ok=True)
        return dst

    def _load_cfg(self) -> Any:
        p = self.config_path()
        cfg, _ = cfgmod.load(p) if p.exists() else (cfgmod.AppConfig(), [])
        return cfg

    # ------------------------------------------------------------------ public
    def preview(self, ops: list[Op]) -> dict[str, Any]:
        """What ``ops`` would do, without saving anything (raises :class:`ChangeError` when invalid)."""
        cfg = self._load_cfg()
        done = apply_ops(cfg, ops, self.presets)
        return {
            "changes": [a.text for a in done],
            "low_risk": bool(done) and all(a.low_risk for a in done),
            "before_after": [{"key": a.op.key, "before": a.before, "after": a.after} for a in done],
        }

    def apply(self, ops: list[Op], who: str, agent_id: str = "", why: str = "") -> dict[str, Any]:
        if not ops:
            raise ChangeError("nothing to change")
        with self._lock:
            cfg = self._load_cfg()
            done: list[Applied] = apply_ops(cfg, ops, self.presets)
            batch = uuid.uuid4().hex[:8]
            snap = self._snapshot(batch)
            cfgmod.save(cfg, self.config_path())
            now = self.clock()
            new = [
                Entry(
                    id=uuid.uuid4().hex[:10],
                    batch=batch,
                    ts=now,
                    who=who,
                    agent_id=agent_id,
                    why=str(why or "")[:500],
                    text=a.text,
                    op=a.op.to_dict(),
                    before=a.before,
                    after=a.after,
                    inverse=[o.to_dict() for o in a.inverse],
                    low_risk=a.low_risk,
                )
                for a in done
            ]
            self.entries.extend(new)
            self._save()
        if self.on_applied is not None:
            with contextlib.suppress(Exception):  # the file is saved; the app's config watcher applies it anyway
                self.on_applied(cfg)
        return {
            "batch": batch,
            "snapshot": snap.name if snap else None,
            "changes": [{"id": e.id, "text": e.text} for e in new],
        }

    def history(self, limit: int = 50, agent_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self.entries if agent_id is None or e.agent_id == agent_id]
        return [
            {"id": e.id, "batch": e.batch, "ts": e.ts, "who": e.who, "why": e.why, "text": e.text, "undone": e.undone}
            for e in reversed(items[-limit:])
        ]

    def _undo_entries(self, entries: list[Entry], who: str) -> dict[str, Any]:
        ops: list[Op] = []
        for e in reversed(entries):  # newest first (entries are in history order)
            ops.extend(Op.from_dict(o) for o in e.inverse)
        if not ops:
            raise ChangeError("nothing to undo")
        res = self.apply(ops, who=who, why="undo: " + "; ".join(e.text for e in entries)[:400])
        with self._lock:
            for e in entries:
                e.undone = True
            self._save()
        return res

    def undo(self, change_id: str, who: str = "undo") -> dict[str, Any]:
        with self._lock:
            e = next((x for x in self.entries if x.id == change_id), None)
        if e is None:
            raise ChangeError(f"no change {change_id!r}")
        if e.undone:
            raise ChangeError("that change was already undone")
        return self._undo_entries([e], who)

    def undo_last(self, who: str = "undo") -> dict[str, Any]:
        with self._lock:
            e = next((x for x in reversed(self.entries) if not x.undone and not x.who.startswith("undo")), None)
        if e is None:
            raise ChangeError("nothing to undo")
        return self.undo_batch(e.batch, who)

    def undo_batch(self, batch: str, who: str = "undo") -> dict[str, Any]:
        with self._lock:
            items = [x for x in self.entries if x.batch == batch and not x.undone]
        if not items:
            raise ChangeError(f"nothing left to undo in batch {batch!r}")
        return self._undo_entries(items, who)

    def undo_agent_today(self, agent_id: str, who: str = "undo") -> dict[str, Any]:
        start = time.mktime((*time.localtime(self.clock())[:3], 0, 0, 0, 0, 0, -1))
        with self._lock:
            items = [x for x in self.entries if x.agent_id == agent_id and x.ts >= start and not x.undone]
        if not items:
            raise ChangeError("this agent changed nothing today (or it was already undone)")
        return self._undo_entries(items, who)
