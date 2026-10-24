"""Agent Link <-> the other v0.1.20 features (Gesture Studio, Shortcuts, smart pointer, attention/comfort).

* :func:`gestures_domain`: per Studio gesture ``enabled`` / ``threshold`` / ``hold_s`` / ``cooldown_s``
  (``studio.gestures.<name>`` in the config; the same ranges Gesture Studio's own sliders use).
* :func:`macros_domain`: recorded shortcuts: list (step descriptions only, never typed text), create (from
  steps an agent sends), delete (kept in a private backup in the shortcut folder so it can be undone). Binding
  a gesture or phrase to a shortcut is an ordinary ``bindings`` change with the ``macro:<name>`` action.
* :func:`context_extensions`: live context for ``get_context`` / ``away://about``: the fatigue mode, the
  attention gate, the smart pointer, shortcuts and Studio.

Pure functions over an :class:`~away.config.AppConfig`; files only through :class:`MacroStore` and
:class:`~away.studio.pack.StudioStore`, so the registry tests can point them at a temporary folder.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

GESTURE_FIELDS: dict[str, tuple[float, float] | None] = {
    "enabled": None,  # a switch
    "threshold": (0.3, 0.99),  # "How sure it must be" (away.studio.pack.settings_from)
    "hold_s": (0.0, 3.0),  # "Hold for"
    "cooldown_s": (0.0, 30.0),  # "Rest time"
}
GESTURE_TITLES = {
    "enabled": "on",
    "threshold": "how sure it must be",
    "hold_s": "hold for (s)",
    "cooldown_s": "rest time (s)",
}
BACKUP_DIR = ".agent-deleted"  # inside the shortcut folder: an agent's deletion stays undoable, privately


def _change_error(msg: str) -> Exception:
    from away.agentlink.registry import ChangeError

    return ChangeError(msg)


# ------------------------------------------------------------------------------------------- Studio
def _studio_names(cfg: Any, store: Any = None) -> dict[str, str]:
    """Studio gesture name -> kind (static / moving): the trained pack plus any saved settings."""
    out: dict[str, str] = {}
    with contextlib.suppress(Exception):
        from away.studio.pack import StudioStore

        for g in (store or StudioStore()).gestures():
            out[g.name] = g.kind
    for name in cfg.studio.gestures:
        out.setdefault(str(name), "static")
    return out


def gestures_domain(store: Any = None) -> Any:
    from away.agentlink.registry import Applied, Domain, Op, _coerce_number

    def lst(cfg: Any) -> list[dict[str, Any]]:
        from away.studio.pack import gesture_settings

        rows: list[dict[str, Any]] = []
        for name, kind in sorted(_studio_names(cfg, store).items()):
            try:
                s = gesture_settings(cfg, name, kind)
            except ValueError:
                continue
            rows.append(
                {
                    "key": name,
                    "kind": f"studio_gesture ({kind})",
                    "enabled": s.enabled,
                    "threshold": s.threshold,
                    "hold_s": s.hold_s,
                    "cooldown_s": s.cooldown_s,
                    "help": "set '<name>.enabled' (true/false), '<name>.threshold' (0.3..0.99), "
                    "'<name>.hold_s' (0..3) or '<name>.cooldown_s' (0..30)",
                }
            )
        return rows

    def apply(cfg: Any, op: Op) -> Applied:
        from away.studio.pack import LABEL_RE, gesture_settings

        if op.op != "set":
            raise _change_error("gestures only support 'set' (train new ones in Settings > Gesture Studio)")
        name, _, fld = op.key.partition(".")
        if not LABEL_RE.match(name) or fld not in GESTURE_FIELDS:
            raise _change_error("set '<gesture>.enabled', '.threshold', '.hold_s' or '.cooldown_s'")
        kinds = _studio_names(cfg, store)
        if name not in kinds:
            raise _change_error(f"there is no Gesture Studio gesture named {name!r} (see list_settings gestures)")
        cur = gesture_settings(cfg, name, kinds[name])
        before = getattr(cur, fld)
        rng = GESTURE_FIELDS[fld]
        if rng is None:
            if not isinstance(op.value, bool):
                raise _change_error(f"{name}.enabled is on/off (true/false)")
            new: Any = op.value
        else:
            new = float(_coerce_number(f"studio.gestures.{op.key}", op.value, rng[0], rng[1], 0.0))
        entry = dict(cfg.studio.gestures.get(name) or {})
        entry[fld] = new
        gestures = dict(cfg.studio.gestures)
        gestures[name] = entry
        cfg.studio.gestures = gestures
        return Applied(
            op,
            before,
            new,
            [Op("gestures", "set", op.key, before)],
            f"Gesture {name}, {GESTURE_TITLES[fld]}: {before} -> {new}",
            fld != "enabled",
        )

    return Domain(
        "gestures",
        "Gesture Studio gestures (per gesture)",
        lst,
        apply,
        ops=("set",),
        note="train or delete gestures in Settings > Gesture Studio",
    )


# ------------------------------------------------------------------------------------------ Shortcuts
def macros_domain(store_factory: Callable[[Any], Any] | None = None) -> Any:
    from away.agentlink.registry import Applied, Domain, Op, is_dry_run

    def store_for(cfg: Any) -> Any:
        if store_factory is not None:
            return store_factory(cfg)
        from away.control.macros import MacroStore

        return MacroStore.default(cfg.macros)

    def lst(cfg: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for m in store_for(cfg).all():
            rows.append(
                {
                    "key": m.name,
                    "title": m.label,
                    "kind": "shortcut",
                    "steps": [s.describe() for s in m.steps],  # typed text is never included
                    "apps": m.apps,
                    "types_private_text": m.sensitive,
                    "run_with": f"macro:{m.name}",
                }
            )
        return rows

    def apply(cfg: Any, op: Op) -> Applied:
        from away.control.macros import Macro, MacroError, name_from_title

        store = store_for(cfg)
        if op.op == "create":
            spec = op.value if isinstance(op.value, dict) else None
            if spec is None:
                raise _change_error(
                    "create a shortcut with value {title, steps: [{type: click|key|text|scroll|wait, ...}]}"
                )
            name = op.key or name_from_title(str(spec.get("title", "")))
            if store.exists(name):
                raise _change_error(f"a shortcut named {name!r} already exists")
            try:
                m = Macro.from_dict({**spec, "name": name, "created": time.time()})
            except MacroError as exc:
                raise _change_error(str(exc)) from None
            if not is_dry_run():
                store.save(m)
            redacted = {**spec, "steps": [s.describe() for s in m.steps]}  # the change history never keeps text
            return Applied(
                Op(op.domain, op.op, name, redacted, op.profile),
                None,
                name,
                [Op("macros", "delete", name)],
                f'New shortcut "{m.label}" ({len(m.steps)} steps); bind it with macro:{name}',
            )
        if op.op == "delete":
            if not store.exists(op.key):
                raise _change_error(f"there is no shortcut named {op.key!r}")
            src = Path(store.path(op.key))
            backup_dir = src.parent / BACKUP_DIR
            backup = backup_dir / f"{op.key}-{int(time.time() * 1000)}.json"
            if not is_dry_run():
                backup_dir.mkdir(parents=True, exist_ok=True)
                src.replace(backup)  # private, on this PC; the change history only keeps the file name
            return Applied(
                op,
                op.key,
                None,
                [Op("macros", "restore", op.key, backup.name)],
                f'Deleted shortcut "{op.key}"',
            )
        if op.op == "restore":  # only produced by undo
            backup = Path(store.folder) / BACKUP_DIR / str(op.value or "")
            if not op.value or not backup.is_file() or backup.parent.name != BACKUP_DIR:
                raise _change_error("that shortcut can no longer be restored")
            if is_dry_run():
                return Applied(op, None, op.key, [Op("macros", "delete", op.key)], f'Restored shortcut "{op.key}"')
            m = Macro.from_dict({**json.loads(backup.read_text(encoding="utf-8")), "name": op.key})
            store.save(m)
            backup.unlink()
            return Applied(op, None, op.key, [Op("macros", "delete", op.key)], f'Restored shortcut "{op.key}"')
        raise _change_error("shortcuts support create and delete (run one by binding macro:<name>)")

    return Domain("macros", "Shortcuts (recorded clicks and keys)", lst, apply, ops=("create", "delete"))


# ------------------------------------------------------------------------------------------- context
def fatigue_mode(pipeline: Any) -> str | None:
    """``head_control`` (switched to the head pointer because the arm is tired), ``tired``, ``ok``, or None."""
    hub = getattr(pipeline, "comfort", None)
    if hub is None:
        return None
    f = hub.attention_state().get("fatigue") or {}
    if f.get("head_control"):
        return "head_control"
    return "tired" if f.get("tired") else "ok"


def context_extensions(app: Any) -> dict[str, Callable[[], Any]]:
    """Live-context callables for :attr:`Telemetry.extensions` (each one is cheap and never raises far)."""
    pipe = app.pipeline
    out: dict[str, Callable[[], Any]] = {}
    if getattr(pipe, "comfort", None) is not None:
        out["fatigue"] = lambda: fatigue_mode(pipe)

        def attention() -> dict[str, Any]:
            st = pipe.comfort.attention_state()
            return {"state": st.get("state"), "hint": st.get("hint")}

        out["attention_gate"] = attention
    smart = getattr(pipe, "smart", None)
    if smart is not None:
        out["smart_pointer"] = lambda: {
            "available": smart.available,
            "snap_to_buttons": smart.magnet_on(),
            "click_by_name": smart.voice_on(),
            "smoothing": str(pipe.cfg.hands.cursor.smoothing),
        }
    macros = getattr(app, "macros", None)
    if macros is not None:
        out["shortcuts"] = lambda: {"recording": bool(macros.recording), "playing": bool(macros.playing)}
    studio = getattr(pipe, "studio", None)
    if studio is not None:
        out["gesture_studio"] = lambda: {
            "gestures": sorted(studio.settings),
            "hand_in_allowed_zone": bool(studio.zone_ok),
        }
    return out


__all__ = ["context_extensions", "fatigue_mode", "gestures_domain", "macros_domain"]
