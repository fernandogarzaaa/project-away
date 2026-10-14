"""v0.1.20 integration: Agent Link sees Gesture Studio, Shortcuts, the smart pointer and attention/comfort;
the guided first run hands its zones and custom-gesture steps to Gesture Studio."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from away import config as C
from away.agentlink import desktop, integrations, registry
from away.agentlink.changes import ChangeManager
from away.agentlink.desktop import Element, SmartPointerUIA, best_match
from away.agentlink.registry import ChangeError, Op
from away.pointer.targets import Snapshot, Target

STEPS = [
    {"type": "click", "target": {"x": 10, "y": 20, "app": "outlook.exe", "element": "New mail"}},
    {"type": "text", "text": "my secret note", "app": "outlook.exe"},
    {"type": "key", "keys": "ctrl+enter", "app": "outlook.exe"},
]


def manager(tmp_path: Path) -> tuple[ChangeManager, Path]:
    cfgp = tmp_path / "config.yaml"
    cfg = C.AppConfig()
    cfg.macros.store_dir = str(tmp_path / "macros")
    C.save(cfg, cfgp)
    return ChangeManager(tmp_path / "al", lambda: cfgp), cfgp


# ------------------------------------------------------------------------------------------- registry
def test_real_domains_replace_the_placeholders() -> None:
    doms = registry.domains()
    assert doms["macros"].available and doms["macros"].apply is not None
    assert set(doms["macros"].ops) == {"create", "delete"}
    assert doms["gestures"].available and doms["gestures"].ops == ("set",)
    assert not any("feat/" in t for t in registry.MERGE_TODO)


def test_new_feature_switches_and_sliders_are_agent_settings() -> None:
    items = {i["key"]: i for i in registry.settings_list(C.AppConfig())}
    for key in (
        "hands.cursor.magnet.enabled",
        "voice.click_by_name",
        "attention.enabled",
        "comfort.fatigue_switch",
        "comfort.break_reminders",
        "zones.enabled",
    ):
        assert items[key]["kind"] == "switch", key
    assert items["hands.cursor.smoothing"]["choices"] == ["adaptive", "one_euro", "predictive"]
    assert items["hands.cursor.magnet.strength"]["kind"] == "number"  # pointer_tunables
    assert any(k.startswith(("attention.", "comfort.")) and v["kind"] == "number" for k, v in items.items())


def test_switching_a_feature_through_agent_link_is_undoable(tmp_path: Path) -> None:
    cm, cfgp = manager(tmp_path)
    r = cm.apply(
        [
            Op("settings", "set", "attention.enabled", True),
            Op("settings", "set", "hands.cursor.smoothing", "predictive"),
            Op("settings", "set", "comfort.fatigue_switch", True),
        ],
        "Claude",
    )
    cfg = C.load(cfgp)[0]
    assert cfg.attention.enabled and cfg.hands.cursor.smoothing == "predictive" and cfg.comfort.fatigue_switch
    cm.undo_batch(r["batch"])
    cfg = C.load(cfgp)[0]
    assert not cfg.attention.enabled and cfg.hands.cursor.smoothing == "adaptive"  # v0.1.25 default
    with pytest.raises(ChangeError):
        cm.apply([Op("settings", "set", "hands.cursor.smoothing", "teleport")], "Claude")


def test_agents_may_bind_a_gesture_to_a_shortcut_but_still_not_to_programs(tmp_path: Path) -> None:
    cm, cfgp = manager(tmp_path)
    cm.apply([Op("bindings", "set", "voice:open my mail", "macro:open_mail")], "Claude")
    assert str(C.load(cfgp)[0].bindings.default.voice["open my mail"]) == "macro:open_mail"
    with pytest.raises(ChangeError):
        cm.apply([Op("bindings", "set", "gesture:pinch", "launch:cmd.exe")], "Claude")


# ------------------------------------------------------------------------------------------- shortcuts
def test_macros_domain_creates_lists_without_text_deletes_and_undoes(tmp_path: Path) -> None:
    cm, cfgp = manager(tmp_path)
    cm.apply([Op("macros", "create", "send_note", {"title": "Send a note", "steps": STEPS})], "Claude")
    store_dir = tmp_path / "macros"
    assert (store_dir / "send_note.json").is_file()
    rows = registry.domains()["macros"].list(C.load(cfgp)[0])
    assert rows[0]["key"] == "send_note" and rows[0]["types_private_text"] is True
    assert "my secret note" not in json.dumps(rows)  # an agent never reads typed text back
    assert rows[0]["steps"][1] == "Type text (14 characters)"

    r = cm.apply([Op("macros", "delete", "send_note")], "Claude")
    assert not (store_dir / "send_note.json").exists()
    history = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "al").rglob("*") if p.is_file())
    assert "my secret note" not in history  # the undo record keeps a file name, not the steps
    cm.undo_batch(r["batch"])
    assert (store_dir / "send_note.json").is_file()
    assert not list((store_dir / integrations.BACKUP_DIR).glob("*.json"))


@pytest.mark.parametrize(
    "op",
    [
        Op("macros", "create", "x"),  # no steps
        Op("macros", "create", "bad name!", {"steps": STEPS}),
        Op("macros", "create", "w", {"steps": [{"type": "shell", "text": "calc"}]}),
        Op("macros", "delete", "nope"),
        Op("macros", "restore", "x", "../../etc/passwd"),
    ],
)
def test_macros_domain_refuses_bad_changes(tmp_path: Path, op: Op) -> None:
    cm, _ = manager(tmp_path)
    with pytest.raises(ChangeError):
        cm.apply([op], "Claude")


def test_macros_domain_will_not_overwrite(tmp_path: Path) -> None:
    cm, _ = manager(tmp_path)
    cm.apply([Op("macros", "create", "a", {"steps": STEPS})], "Claude")
    with pytest.raises(ChangeError):
        cm.apply([Op("macros", "create", "a", {"steps": STEPS})], "Claude")


# ------------------------------------------------------------------------------------- Gesture Studio
def test_gestures_domain_sets_per_gesture_settings_and_undoes(tmp_path: Path) -> None:
    from away.studio.pack import StudioStore

    StudioStore().add("thumbs_up", "static")
    cm, cfgp = manager(tmp_path)
    rows = registry.domains()["gestures"].list(C.load(cfgp)[0])
    assert [r["key"] for r in rows] == ["thumbs_up"] and rows[0]["enabled"] is True
    r = cm.apply(
        [Op("gestures", "set", "thumbs_up.threshold", 0.9), Op("gestures", "set", "thumbs_up.enabled", False)],
        "Claude",
    )
    g = C.load(cfgp)[0].studio.gestures["thumbs_up"]
    assert g["threshold"] == pytest.approx(0.9) and g["enabled"] is False
    cm.undo_batch(r["batch"])
    g = C.load(cfgp)[0].studio.gestures["thumbs_up"]
    assert g["enabled"] is True and g["threshold"] == pytest.approx(rows[0]["threshold"])
    for bad in (
        Op("gestures", "set", "thumbs_up.threshold", 5),
        Op("gestures", "set", "thumbs_up.enabled", "yes"),
        Op("gestures", "set", "thumbs_up.then", ["shell:calc"]),
        Op("gestures", "set", "wave.enabled", True),  # not a Studio gesture
        Op("gestures", "create", "wave"),
    ):
        with pytest.raises(ChangeError):
            cm.apply([bad], "Claude")


# ------------------------------------------------------------------------------------------- context
def test_context_extensions_report_fatigue_attention_pointer_shortcuts_and_studio() -> None:
    fatigue = {"tired": False, "head_control": False}
    comfort = SimpleNamespace(attention_state=lambda: {"state": "engaged", "hint": "", "fatigue": dict(fatigue)})
    smart = SimpleNamespace(available=False, magnet_on=lambda: False, voice_on=lambda: False)
    studio = SimpleNamespace(settings={"thumbs_up": None}, zone_ok=True)
    pipe = SimpleNamespace(comfort=comfort, smart=smart, studio=studio, cfg=SimpleNamespace(hands=C.AppConfig().hands))
    app = SimpleNamespace(pipeline=pipe, macros=SimpleNamespace(recording=True, playing=False))
    ext = integrations.context_extensions(app)
    assert ext["fatigue"]() == "ok"
    fatigue["head_control"] = True
    assert ext["fatigue"]() == "head_control"
    assert ext["attention_gate"]()["state"] == "engaged"
    assert ext["smart_pointer"]()["smoothing"] == "adaptive"
    assert ext["shortcuts"]() == {"recording": True, "playing": False}
    assert ext["gesture_studio"]()["gestures"] == ["thumbs_up"]
    assert integrations.fatigue_mode(SimpleNamespace()) is None


def test_get_context_reports_the_fatigue_mode_instead_of_null(tmp_path: Path) -> None:
    from away.agentlink.telemetry import Telemetry

    tel = Telemetry()
    assert tel.context()["fatigue_mode"] is None
    tel.extensions["fatigue"] = lambda: "tired"
    tel.extensions["attention_gate"] = lambda: {"state": "idle"}
    ctx = tel.context()
    assert ctx["fatigue_mode"] == "tired" and ctx["extensions"]["attention_gate"] == {"state": "idle"}


# ---------------------------------------------------------------------- computer use + smart pointer
class FakeCache:
    def __init__(self, targets: list[Target]) -> None:
        self.targets = targets
        self.calls = 0

    def refresh_now(self) -> Snapshot:
        self.calls += 1
        return Snapshot(tuple(self.targets), 0.0)


def test_agent_computer_use_finds_elements_with_the_smart_pointer_cache() -> None:
    cache = FakeCache(
        [Target("Send message", "button", (100, 200, 180, 230)), Target("Save", "button", (0, 0, 40, 20))]
    )
    smart = SimpleNamespace(available=True, cache=cache)
    uia = SmartPointerUIA(smart, desktop.NullUIA())
    assert uia.available
    el = uia.find("send")
    assert el is not None and el.name == "Send message" and el.center == (140, 215) and cache.calls == 1
    assert uia.find("sav", "Button") is not None
    assert uia.find("print") is None  # no match: falls back to the base (NullUIA)
    assert uia.focused() is None  # password checks etc. stay with the base


def test_smart_pointer_uia_falls_back_when_the_pointer_has_no_ui_automation() -> None:
    class Base(desktop.NullUIA):
        available = True

        def find(self, name: str, control_type: str | None = None) -> Element | None:
            return Element("From base", rect=(0, 0, 10, 10))

    uia = SmartPointerUIA(SimpleNamespace(available=False, cache=FakeCache([])), Base())
    found = uia.find("anything")
    assert found is not None and found.name == "From base"


def test_best_match_uses_the_fuzzy_spoken_name_score_as_a_last_resort() -> None:
    els = [Element("Send", "Button", rect=(0, 0, 10, 10)), Element("Settings", "Button", rect=(20, 0, 30, 10))]
    assert best_match(els, "Send").name == "Send"  # type: ignore[union-attr]
    assert best_match(els, "sned").name == "Send"  # type: ignore[union-attr]
    assert best_match(els, "zzz") is None


# ------------------------------------------------------------------------------------------- first run
def test_first_run_gets_gesture_studio_steps() -> None:
    from away.app import first_run

    calls: list[list[str]] = []
    import away.studio.first_run as sfr

    sfr.register(popen=lambda argv: calls.append(list(argv)))
    try:
        assert "away.studio.first_run" in first_run.discover_plugins()
        keys = [s.key for s in first_run.build_steps()]
        assert keys.index("zones") == keys.index("framing") + 1
        assert keys.index("custom_gesture") == keys.index("gaze") + 1
        steps = {s.key: s for s in first_run.build_steps()}
        assert steps["zones"].run is not None and steps["custom_gesture"].run is not None
        steps["zones"].run()
        steps["custom_gesture"].run()
        assert calls[0][-3:] == ["settings", "--tab", "Gesture Studio"]
        assert calls[1][-1] == "--teach-gesture"
    finally:
        sfr.register()


def test_settings_cli_accepts_a_tab_and_the_teach_guide(monkeypatch: Any) -> None:
    from away import cli
    from away.app import settings_gui

    seen: dict[str, Any] = {}
    monkeypatch.setattr(settings_gui, "main", lambda **kw: seen.update(kw) or 0)
    assert cli.main(["settings", "--tab", "studio", "--teach-gesture"]) == 0
    assert seen["tab"] == "studio" and seen["teach_gesture"] is True


def test_record_away_input_is_on_by_default() -> None:
    assert C.AppConfig().macros.record_away_input is True
    cfg, unknown = C.from_dict({"macros": {"record_away_input": False}})
    assert unknown == [] and cfg.macros.record_away_input is False


# ------------------------------------------------------------------------------------------- Settings tabs
class FakeNotebook:
    def __init__(self, titles: list[str]) -> None:
        self._tabs = [f".t{i}" for i in range(len(titles))]
        self._text = dict(zip(self._tabs, titles))

    def tabs(self) -> list[str]:
        return list(self._tabs)

    def tab(self, tab_id: str, option: str) -> str:
        return self._text[tab_id]

    def insert(self, pos: int, tab_id: str) -> None:
        self._tabs.remove(tab_id)
        self._tabs.insert(pos, tab_id)


def test_settings_tabs_start_with_getting_started_and_end_with_the_expert_pages() -> None:
    from away.app.settings_gui import TAB_ORDER, order_tabs

    built = [
        "Bindings",
        "Shortcuts",
        "Tuning",
        "Voice & input",
        "Cameras",
        "Performance",
        "Gesture Studio",
        "Agent Link",
        "Getting started",
        "Something new",
    ]
    got = order_tabs(FakeNotebook(built), TAB_ORDER)
    assert got[0] == "Getting started" and got[1:4] == ["Bindings", "Gesture Studio", "Shortcuts"]
    assert got[-3:] == ["Tuning", "Performance", "Something new"]  # an unnamed tab keeps a place at the end
    assert sorted(got) == sorted(built)
