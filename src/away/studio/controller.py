"""Everything the Gesture Studio tab does, minus the widgets (tested headless).

Studio changes are saved straight away (only the Studio's own parts of the config: ``zones``, ``studio`` and
the bindings of Studio gestures), so the gestures on disk and the config never disagree and the user never has
to remember a separate Save. Other unsaved edits in the Settings window are left alone.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from away.studio import pack as P
from away.studio.pack import NOTHING, StudioStore, friendly_name, gesture_settings, new_session_id
from away.studio.words import ACTION_CHOICES, HANDS_WORDS, KIND_WORDS, action_text, name_for
from away.studio.zones import ZoneMap, camera_key, lookup_map, store_map

NO_ARG_KINDS = ("none", "mouse", "dictation", "voice", "mic", "pause")


@dataclass
class GestureRow:
    name: str
    title: str
    kind: str
    hands: str
    examples: int
    rounds: int
    status: str
    action: str


class StudioController:
    def __init__(self, model: Any, store: StudioStore | None = None) -> None:
        self.m = model
        self.store = store or StudioStore()

    @property
    def cfg(self) -> Any:
        return self.m.cfg

    # ---------------------------------------------------------------- saving
    def persist(self) -> str | None:
        """Write the Studio parts of the config to the file now. Returns an error message or None."""
        from away import config as cfgmod

        m = self.m
        saved = copy.deepcopy(m._saved)
        saved.zones = copy.deepcopy(m.cfg.zones)
        saved.studio = copy.deepcopy(m.cfg.studio)
        names = set(m.cfg.bindings.custom_gestures) | set(saved.bindings.custom_gestures)
        names |= {g.name for g in self.store.gestures()}
        sb, cb = saved.bindings, m.cfg.bindings
        sb.custom_gestures = list(cb.custom_gestures)
        for prof_name, prof in sb.profiles.items():
            src = cb.profiles.get(prof_name)
            for n in names:
                if src is not None and n in src.gestures:
                    prof.gestures[n] = src.gestures[n]
                else:
                    prof.gestures.pop(n, None)
        try:
            cfgmod.save(saved, m.path)
        except (OSError, ValueError) as exc:
            return f"Could not save: {exc}"
        m._saved = saved
        return None

    # ---------------------------------------------------------------- gestures
    def trained_classes(self) -> set[str]:
        try:
            meta = json.loads((self.store.model_dir / "studio.json").read_text())
        except (OSError, ValueError):
            return set()
        return {c for g in meta.get("groups", []) for c in g.get("classes", []) if c != NOTHING}

    def model_time(self) -> float:
        f = self.store.model_dir / "studio.json"
        return f.stat().st_mtime if f.exists() else 0.0

    def rows(self) -> list[GestureRow]:
        trained = self.trained_classes()
        mtime = self.model_time()
        counts = self.store.counts()
        out = []
        for g in self.store.gestures():
            n = counts.get(g.name, 0)
            newest = max((p.stat().st_mtime for p in self.store.clip_paths(g.name)), default=0.0)
            if n == 0:
                status = "Needs practice"
            elif g.name not in trained:
                status = "Ready to learn"
            elif newest > mtime:
                status = "New examples: learn again"
            else:
                status = "Ready"
            if not gesture_settings(self.cfg, g.name, g.kind).enabled:
                status = "Switched off"
            act = str(self.cfg.bindings.resolve("gestures", g.name, "default"))
            rounds = self.store.sessions(g.name) if n else 0
            out.append(
                GestureRow(
                    g.name, name_for(g.name), KIND_WORDS[g.kind], HANDS_WORDS[g.hands], n, rounds, status, action_text(act)
                )
            )
        return out

    def create_gesture(self, title: str, kind: str = "static", hands: int = 1) -> tuple[str | None, str | None]:
        """``(name, error)``: the wizard's step 1. The gesture gets default settings and no action yet."""
        name = friendly_name(title)
        if not name:
            return None, "Type a name for your gesture first."
        try:
            self.store.add(name, kind, hands)
        except ValueError as exc:
            return None, str(exc)[:1].upper() + str(exc)[1:]
        b = self.cfg.bindings
        if name not in b.custom_gestures:
            b.custom_gestures.append(name)
        self.cfg.studio.gestures[name] = P.settings_from(None, kind).to_dict()
        b.default.gestures.setdefault(name, P.parse_action("none"))
        return name, self.persist()

    def remove_gesture(self, name: str) -> str | None:
        self.store.remove(name, delete_clips=True)
        b = self.cfg.bindings
        if name in b.custom_gestures:
            b.custom_gestures.remove(name)
        for prof in b.profiles.values():
            prof.gestures.pop(name, None)
        self.cfg.studio.gestures.pop(name, None)
        return self.persist()

    # ---------------------------------------------------------------- recordings
    def save_take(self, msg: dict[str, Any]) -> tuple[Path | None, str | None]:
        try:
            p = self.store.save_take(
                str(msg.get("label")),
                list(msg.get("frames") or []),
                str(msg.get("handedness", "Right")),
                float(msg.get("fps") or 30.0),
                str(msg.get("session") or new_session_id()),
            )
        except ValueError as exc:
            return None, str(exc)[:1].upper() + str(exc)[1:]
        return p, None

    def save_negative(self, frames: list[tuple[Any, Any]], fps: float = 30.0) -> str | None:
        """'That was wrong': the last seconds become a 'no gesture' example (its own practice round)."""
        try:
            self.store.save_take(NOTHING, frames, "Right", fps, f"wrong-{uuid.uuid4().hex[:10]}")
        except ValueError as exc:
            return str(exc)[:1].upper() + str(exc)[1:]
        return None

    def delete_all_recordings(self) -> int:
        return self.store.delete_all_clips()

    # ---------------------------------------------------------------- learning
    def learn(self, progress: Any = None, backend: str = "auto") -> Any:
        from away.studio.trainer import train_studio

        return train_studio(self.store, backend=backend, progress=progress)

    def last_result(self) -> dict[str, Any] | None:
        try:
            data: dict[str, Any] = json.loads((self.store.model_dir / "metrics.json").read_text())
        except (OSError, ValueError):
            return None
        return data

    # ---------------------------------------------------------------- what it does
    def current_action(self, name: str) -> tuple[str, str]:
        """``(choice kind, argument)`` for the action picker (``url`` for launch:http...)."""
        act = self.cfg.bindings.resolve("gestures", name, "default")
        if act.kind == "launch" and act.arg.lower().startswith(("http://", "https://")):
            return "url", act.arg
        return act.kind, act.arg

    @staticmethod
    def choice_args(kind: str) -> tuple[str, ...] | None:
        for _label, k, args in ACTION_CHOICES:
            if k == kind:
                return args
        return None

    def set_action(self, name: str, kind: str, arg: str = "") -> str | None:
        """Bind a Studio gesture (default profile) from the plain-words picker."""
        arg = arg.strip()
        if kind == "url":
            if not arg:
                return "Type the website address, like example.com"
            kind, arg = "launch", arg if "://" in arg else f"https://{arg}"
        if kind not in NO_ARG_KINDS and not arg:
            return "Fill in the box next to the action first."
        try:
            self.cfg.bindings.set(f"gesture:{name}", "none" if kind == "none" else f"{kind}:{arg}", "default")
        except ValueError as exc:
            return str(exc)[:1].upper() + str(exc)[1:]
        return self.persist()

    def set_settings(
        self,
        name: str,
        enabled: bool,
        threshold: float,
        hold_s: float,
        cooldown_s: float,
        then_text: str = "",
    ) -> str | None:
        g = self.store.get(name)
        steps = [ln.strip() for ln in then_text.splitlines() if ln.strip()]
        raw = {"enabled": enabled, "threshold": threshold, "hold_s": hold_s, "cooldown_s": cooldown_s, "then": steps}
        try:
            s = P.settings_from(raw, g.kind if g else "static", self.cfg.bindings.allow_shell, name)
        except ValueError as exc:
            msg = str(exc).split(": ", 1)[-1]
            return msg[:1].upper() + msg[1:]
        self.cfg.studio.gestures[name] = s.to_dict()
        return self.persist()

    # ---------------------------------------------------------------- sharing
    def export(self, path: Path, include_clips: bool = False) -> str | None:
        try:
            P.export_pack(self.store, self.cfg, path, include_clips=include_clips)
        except (OSError, ValueError) as exc:
            return str(exc)[:1].upper() + str(exc)[1:]
        return None

    def import_(self, path: Path) -> tuple[list[str], str | None]:
        try:
            notes = P.import_pack(self.store, self.cfg, path)
        except (OSError, ValueError, KeyError) as exc:
            return [], str(exc)[:1].upper() + str(exc)[1:]
        return notes, self.persist()

    # ---------------------------------------------------------------- zones
    def zone_profiles(self) -> list[str]:
        return list(self.cfg.bindings.profiles)

    def zone_map(self, profile: str = "default") -> ZoneMap:
        """The map to edit for this camera + profile (falls back like the runtime; else a fresh 6 x 4 grid)."""
        try:
            zm, _src = lookup_map(self.cfg.zones.maps, camera_key(self.cfg), profile)
        except ValueError:
            zm = None
        return zm if zm is not None else ZoneMap()

    def save_zone_map(self, zm: ZoneMap, profile: str = "default") -> str | None:
        """Save for this camera + profile; an all-'works here' map removes the entry (works everywhere)."""
        store_map(self.cfg.zones.maps, camera_key(self.cfg), profile, None if zm.is_open() else zm)
        return self.persist()

    def set_block_cursor(self, on: bool) -> str | None:
        self.cfg.zones.block_cursor = bool(on)
        return self.persist()


# ---------------------------------------------------------------------- the 3-step wizard's recording plan
@dataclass
class Step:
    kind: str  # "take" | "relax"
    round: int
    index: int
    label: str
    session: str
    seconds: float
    countdown: float


@dataclass
class WizardPlan:
    """Step 2 of 'Teach a new gesture': ``rounds`` practice rounds of ``per_round`` examples, each round
    followed by an automatic relax recording (the 'no gesture' examples). Each round is its own session so the
    learner can check itself on rounds it never saw."""

    name: str
    kind: str = "static"
    rounds: int = 3
    per_round: int = 5
    relax_s: float = 3.0
    steps: list[Step] = field(default_factory=list)
    pos: int = 0
    saved: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        secs = 1.0 if self.kind == "static" else 1.5
        for r in range(self.rounds):
            sid = new_session_id()
            for i in range(self.per_round):
                self.steps.append(Step("take", r, i, self.name, sid, secs, 3.0 if i == 0 else 1.5))
            self.steps.append(Step("relax", r, 0, NOTHING, new_session_id(), self.relax_s, 1.0))

    @property
    def done(self) -> bool:
        return self.pos >= len(self.steps)

    @property
    def current(self) -> Step | None:
        return None if self.done else self.steps[self.pos]

    @property
    def total_takes(self) -> int:
        return self.rounds * self.per_round

    def takes_done(self) -> int:
        return sum(1 for s in self.steps[: self.pos] if s.kind == "take")

    def command(self) -> dict[str, Any] | None:
        s = self.current
        if s is None:
            return None
        return {
            "cmd": "record",
            "label": s.label,
            "session": s.session,
            "seconds": s.seconds,
            "countdown": s.countdown,
            "tag": ("wizard", self.pos),
        }

    def accept(self, path: Path) -> None:
        self.saved.append(path)
        self.pos += 1

    def redo_last(self) -> list[Path]:
        """'That one was bad': step back to the last practice example (and any relax recording after it) so it
        is recorded again. Returns the saved files to delete."""
        removed: list[Path] = []
        while self.pos > 0:
            self.pos -= 1
            if self.saved:
                removed.append(self.saved.pop())
            if self.steps[self.pos].kind == "take":
                break
        return removed

    def progress_text(self) -> str:
        s = self.current
        if s is None:
            return "Done!"
        if s.kind == "relax":
            return f"Relax round {s.round + 1} of {self.rounds}"
        return f"Practice example {self.takes_done() + 1} of {self.total_takes} (round {s.round + 1} of {self.rounds})"
